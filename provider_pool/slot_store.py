"""Slot state stores — Redis-backed and in-memory.

Each (key_id, model) pair tracks:
  cooldown_until      — epoch float; slot is unavailable until this time
  cooldown_reason     — human-readable reason for the cooldown
  tokens_this_month   — cumulative tokens used this month
  last_error          — last error message seen from this slot
  last_error_at       — ISO timestamp of the last error

tokens_today is tracked in a SEPARATE key with a 24-hour TTL so it resets
automatically at creation-time + 24h (Redis-backed store only — the in-memory
store resets it in-place since it has no TTL mechanism).

Keys live under a configurable prefix so multi-tenant/BYOK pools can be
namespaced:
  platform:  "provider_pool:slot:{key_id}:{safe_model}"
  per-org:   "provider_pool:org:{org_id}:slot:{key_id}:{safe_model}"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis

_DEFAULT_PREFIX = "provider_pool:slot"

_SET_COOLDOWN_LUA = """
local key     = KEYS[1]
local now     = tonumber(ARGV[1])
local until_t = ARGV[2]
local reason  = ARGV[3]
local iso     = ARGV[4]
local ttl     = tonumber(ARGV[5])

local existing = redis.call('HGET', key, 'cooldown_until')
if existing and tonumber(existing) > now then
    return 0
end
redis.call('HSET', key,
    'cooldown_until',  until_t,
    'cooldown_reason', reason,
    'last_error',      reason,
    'last_error_at',   iso)
redis.call('EXPIRE', key, ttl)
return 1
"""

_TRY_ACQUIRE_LUA = """
local state_key    = KEYS[1]
local inflight_key = KEYS[2]
local now          = tonumber(ARGV[1])
local ttl          = tonumber(ARGV[2])

local cu = redis.call('HGET', state_key, 'cooldown_until')
if cu and tonumber(cu) > now then
    return 0
end
local ok = redis.call('SET', inflight_key, '1', 'NX', 'EX', ttl)
return ok and 1 or 0
"""

_INFLIGHT_TTL = 130


@dataclass
class SlotState:
    key_id: str
    model: str
    cooldown_until: float = 0.0
    cooldown_reason: str = ""
    tokens_today: int = 0
    tokens_this_month: int = 0
    last_error: str = ""
    last_error_at: str = ""

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def cooling_for_seconds(self) -> float:
        """Remaining cooldown time in seconds; 0 if available."""
        remaining = self.cooldown_until - time.time()
        return max(0.0, remaining)


class SlotStore:
    """Async Redis store for per-slot health state and token accounting."""

    def __init__(
        self,
        redis: aioredis.Redis,
        prefix: str = _DEFAULT_PREFIX,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._cooldown_sha: str | None = None
        self._acquire_sha: str | None = None

    async def load_scripts(self) -> None:
        """Pre-load Lua scripts into Redis. Call once at startup."""
        self._cooldown_sha = await self._redis.script_load(_SET_COOLDOWN_LUA)
        self._acquire_sha = await self._redis.script_load(_TRY_ACQUIRE_LUA)

    def _key(self, key_id: str, model: str) -> str:
        safe_model = model.replace("/", "_").replace(":", "_")
        return f"{self._prefix}:{key_id}:{safe_model}"

    def _today_key(self, key_id: str, model: str) -> str:
        """Separate key for tokens_today with 24h TTL — resets automatically."""
        safe_model = model.replace("/", "_").replace(":", "_")
        return f"{self._prefix}:{key_id}:{safe_model}:tokens_today"

    async def is_available(self, key_id: str, model: str) -> bool:
        val = await self._redis.hget(self._key(key_id, model), "cooldown_until")
        if val is None:
            return True
        try:
            return time.time() >= float(val)
        except ValueError:
            return True

    async def get_state(self, key_id: str, model: str) -> SlotState:
        data: dict = await self._redis.hgetall(self._key(key_id, model))
        today_raw = await self._redis.get(self._today_key(key_id, model))
        return SlotState(
            key_id=key_id,
            model=model,
            cooldown_until=float(data.get("cooldown_until", 0)),
            cooldown_reason=data.get("cooldown_reason", ""),
            tokens_today=int(float(today_raw)) if today_raw else 0,
            tokens_this_month=int(float(data.get("tokens_this_month", 0))),
            last_error=data.get("last_error", ""),
            last_error_at=data.get("last_error_at", ""),
        )

    def _inflight_key(self, key_id: str, model: str) -> str:
        safe_model = model.replace("/", "_").replace(":", "_")
        return f"{self._prefix}:{key_id}:{safe_model}:inflight"

    async def try_acquire(self, key_id: str, model: str) -> bool:
        """Atomically check availability and set an in-flight NX lock.

        Returns True if the slot is available AND this worker claimed it.
        Returns False if the slot is cooling OR another worker already holds it.
        Callers MUST call release() when the request finishes (success or error).
        """
        if self._acquire_sha:
            result = await self._redis.evalsha(
                self._acquire_sha,
                2,
                self._key(key_id, model),
                self._inflight_key(key_id, model),
                str(time.time()),
                str(_INFLIGHT_TTL),
            )
            return bool(result)

        # Fallback (scripts not yet loaded — non-atomic, safe during startup)
        val = await self._redis.hget(self._key(key_id, model), "cooldown_until")
        if val is not None:
            try:
                if time.time() < float(val):
                    return False
            except ValueError:
                pass
        result = await self._redis.set(
            self._inflight_key(key_id, model),
            "1",
            nx=True,
            ex=_INFLIGHT_TTL,
        )
        return result is not None

    async def release(self, key_id: str, model: str) -> None:
        """Release the in-flight NX lock for this slot."""
        await self._redis.delete(self._inflight_key(key_id, model))

    async def set_cooldown(
        self,
        key_id: str,
        model: str,
        seconds: float,
        reason: str,
    ) -> None:
        """Set a cooldown, skipping the write if the slot is already cooling (CAS)."""
        now = time.time()
        until = now + seconds
        now_iso = datetime.now(timezone.utc).isoformat()
        key = self._key(key_id, model)
        ttl = min(int(seconds) + 3600, 86400 * 31)

        if self._cooldown_sha:
            await self._redis.evalsha(
                self._cooldown_sha,
                1,
                key,
                str(now),
                str(until),
                reason,
                now_iso,
                str(ttl),
            )
        else:
            existing = await self._redis.hget(key, "cooldown_until")
            if existing is not None:
                try:
                    if float(existing) > now:
                        return
                except ValueError:
                    pass
            await self._redis.hset(
                key,
                mapping={
                    "cooldown_until": until,
                    "cooldown_reason": reason,
                    "last_error": reason,
                    "last_error_at": now_iso,
                },
            )
            await self._redis.expire(key, ttl)

    async def clear_cooldown(self, key_id: str, model: str) -> None:
        key = self._key(key_id, model)
        await self._redis.hset(
            key,
            mapping={"cooldown_until": 0, "cooldown_reason": ""},
        )

    async def record_usage(self, key_id: str, model: str, tokens: int) -> None:
        """Increment token counters for today and this month atomically."""
        key = self._key(key_id, model)
        today_key = self._today_key(key_id, model)
        pipe = self._redis.pipeline(transaction=True)
        pipe.incrbyfloat(today_key, tokens)
        pipe.expire(today_key, 86400)
        pipe.hincrbyfloat(key, "tokens_this_month", tokens)
        await pipe.execute()

    async def reset_daily_tokens(self, key_id: str, model: str) -> None:
        await self._redis.delete(self._today_key(key_id, model))

    async def reset_monthly_tokens(self, key_id: str, model: str) -> None:
        await self._redis.hset(self._key(key_id, model), "tokens_this_month", 0)


class MemorySlotStore:
    """In-memory slot state store — no Redis required.

    Suitable for testing, single-process deployments, or dev environments.
    State is lost on process restart and is not shared across workers, so the
    try_acquire()/release() in-flight lock and cross-process CAS semantics that
    SlotStore provides via Redis don't apply here — safe only for single-process use.
    """

    def __init__(self, prefix: str = _DEFAULT_PREFIX) -> None:
        self._data: dict[str, dict] = {}
        self._prefix = prefix

    def _key(self, key_id: str, model: str) -> str:
        safe_model = model.replace("/", "_").replace(":", "_")
        return f"{self._prefix}:{key_id}:{safe_model}"

    def _get(self, key_id: str, model: str) -> dict:
        return self._data.get(self._key(key_id, model), {})

    async def is_available(self, key_id: str, model: str) -> bool:
        try:
            return time.time() >= float(
                self._get(key_id, model).get("cooldown_until", 0)
            )
        except ValueError:
            return True

    async def get_state(self, key_id: str, model: str) -> SlotState:
        data = self._get(key_id, model)
        return SlotState(
            key_id=key_id,
            model=model,
            cooldown_until=float(data.get("cooldown_until", 0)),
            cooldown_reason=data.get("cooldown_reason", ""),
            tokens_today=int(float(data.get("tokens_today", 0))),
            tokens_this_month=int(float(data.get("tokens_this_month", 0))),
            last_error=data.get("last_error", ""),
            last_error_at=data.get("last_error_at", ""),
        )

    async def try_acquire(self, key_id: str, model: str) -> bool:
        return await self.is_available(key_id, model)

    async def release(self, key_id: str, model: str) -> None:
        pass

    async def set_cooldown(
        self, key_id: str, model: str, seconds: float, reason: str
    ) -> None:
        key = self._key(key_id, model)
        if key not in self._data:
            self._data[key] = {}
        self._data[key].update(
            {
                "cooldown_until": time.time() + seconds,
                "cooldown_reason": reason,
                "last_error": reason,
                "last_error_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def clear_cooldown(self, key_id: str, model: str) -> None:
        key = self._key(key_id, model)
        if key in self._data:
            self._data[key].update({"cooldown_until": 0.0, "cooldown_reason": ""})

    async def record_usage(self, key_id: str, model: str, tokens: int) -> None:
        key = self._key(key_id, model)
        if key not in self._data:
            self._data[key] = {}
        d = self._data[key]
        d["tokens_today"] = float(d.get("tokens_today", 0)) + tokens
        d["tokens_this_month"] = float(d.get("tokens_this_month", 0)) + tokens

    async def reset_daily_tokens(self, key_id: str, model: str) -> None:
        key = self._key(key_id, model)
        if key in self._data:
            self._data[key]["tokens_today"] = 0

    async def reset_monthly_tokens(self, key_id: str, model: str) -> None:
        key = self._key(key_id, model)
        if key in self._data:
            self._data[key]["tokens_this_month"] = 0
