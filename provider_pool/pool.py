"""Provider pool — multi-key, multi-model fallback with circuit-breaking.

Architecture
============
A ProviderPool holds an ordered list of Slots, each representing one
(api_key, model) pair on a specific provider backend.  Slots are ordered
column-first across keys so that load is spread before quality degrades:

  configs:
    key_A  → [gemini-2.5-flash, gemini-3-flash-preview]
    key_B  → [gemini-2.5-flash, gemini-3.1-flash-lite-preview]
    openrouter → [gpt-4o-mini]

  slot order:
    0. (key_A, gemini-2.5-flash)         ← try best model on all keys first
    1. (key_B, gemini-2.5-flash)
    2. (openrouter, gpt-4o-mini)
    3. (key_A, gemini-3-flash-preview)   ← only then fall to weaker models
    4. (key_B, gemini-3.1-flash-lite-preview)

Fallback logic
==============
For each slot, the pool starts streaming.  If an ErrorEvent arrives
**before** the first TextDeltaEvent and the error is retryable, the slot
is cooled down and the next slot is tried.  Once the first chunk is yielded
the pool is committed to that slot — a mid-stream error is surfaced to the
caller as a normal ErrorEvent (the client must retry).

Error classification → cooldown
================================
  billing_cap   (monthly spend exceeded)  → 24 h
  rate_limited  (per-minute/day quota)    → 60 s
  server_error  (5xx transient)           → 30 s
  auth_error    (401/403, dead key)       → 30 days (effectively permanent)
  unknown                                 → 30 s

Multi-tenant / BYOK usage
==========================
Build a temporary pool per-tenant from the tenant's stored keys, namespaced
so its cooldown state doesn't collide with the platform pool's:

  org_pool = ProviderPool.from_config(
      org_slot_configs,
      store=SlotStore(redis, prefix=f"provider_pool:org:{org_id}:slot"),
  )
  async for event in org_pool.generate_with_fallback(messages):
      ...

Quick-start
===========
  from provider_pool import ProviderPool, SlotConfig, MemorySlotStore

  pool = ProviderPool.from_config(
      [
          SlotConfig(key_id="primary", provider="google",
                     api_key="AIzaSy...", models=["gemini-2.5-flash"]),
          SlotConfig(key_id="fallback", provider="openai",
                     api_key="sk-...", models=["gpt-4o-mini"]),
      ],
      store=MemorySlotStore(),  # swap for SlotStore(redis) in production
  )

  async for event in pool.generate_with_fallback(messages):
      ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from .alerts import (
    AlertBackend,
    AlertPayload,
    LogAlertBackend,
    SlotEvent,
    build_alert_backend,
)
from .base import (
    ChatMessage,
    ErrorEvent,
    FinishEvent,
    LLMProvider,
    StreamEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolDefinition,
)
from .slot_store import MemorySlotStore, SlotState, SlotStore

logger = logging.getLogger(__name__)

# ── Error classification ───────────────────────────────────────────────────────

_COOLDOWN: dict[str, float] = {
    "billing_cap": 86_400.0,  # 24 h — monthly spend cap
    "rate_limited": 60.0,  # 1 min — TPM / RPD quota
    "server_error": 30.0,  # transient 5xx
    "auth_error": 86_400 * 30,  # 30 days — dead key
    "unknown": 30.0,
}


def _classify_error(message: str) -> tuple[str, float]:
    """Return (error_type, cooldown_seconds) for an error message."""
    msg = message.lower()
    if (
        "billing account has exceeded" in msg
        or "monthly spending cap" in msg
        or "spend cap" in msg
        or "insufficient_quota" in msg
        or "exceeded your current quota" in msg
        or "quota exceeded" in msg
    ):
        return "billing_cap", _COOLDOWN["billing_cap"]
    if (
        "resource_exhausted" in msg
        or "429" in message
        or "rate limit" in msg
        or "ratelimit" in msg
        or "too many requests" in msg
        or "too_many_requests" in msg
    ):
        return "rate_limited", _COOLDOWN["rate_limited"]
    if (
        "401" in message
        or "403" in message
        or "invalid api key" in msg
        or "incorrect api key" in msg
        or "authentication" in msg
    ):
        return "auth_error", _COOLDOWN["auth_error"]
    if (
        any(code in message for code in ("500", "502", "503", "504"))
        or "service unavailable" in msg
        or "overloaded" in msg
    ):
        return "server_error", _COOLDOWN["server_error"]
    return "unknown", _COOLDOWN["unknown"]


def _is_retryable(error_type: str) -> bool:
    return error_type != "auth_error"


# ── JSON extraction (for generate_json) ────────────────────────────────────────


def _extract_json(text: str) -> str:
    """Extract the outermost JSON object or array from LLM output.

    Handles markdown fences (```json ... ```) and leading/trailing prose.
    Uses bracket-depth tracking to find the exact end rather than rfind(),
    which would grab the last closing bracket even inside a nested string.

    Picks whichever of `{`/`[` appears first in the text as the outermost
    value's opening bracket — trying `{` unconditionally before `[` would
    mis-extract a top-level array of objects as just its first element.
    """
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    candidates = [(text.find(ch), ch) for ch in ("{", "[")]
    candidates = [(pos, ch) for pos, ch in candidates if pos != -1]
    if not candidates:
        return text
    start, open_ch = min(candidates)
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_string = False
    skip_next = False
    for i, ch in enumerate(text[start:], start):
        if skip_next:
            skip_next = False
            continue
        if ch == "\\" and in_string:
            skip_next = True
            continue
        if ch == '"':
            in_string = not in_string
        if not in_string:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


SLOT_GENERATION_TIMEOUT = 120


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class SlotConfig:
    """Config for one API key entry — one key can cover many models."""

    key_id: str
    provider: str  # "google" | "openai" | "anthropic"
    api_key: str = field(repr=False)
    models: list[str] = field(default_factory=list)
    base_url: str | None = None  # OpenRouter or custom endpoint
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Slot:
    """A single (key, model) pair the pool can route to."""

    key_id: str
    provider: str
    api_key: str = field(repr=False)
    model: str = field(default="")
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slot_id(self) -> str:
        return f"{self.key_id}/{self.model}"


# ── Pool ──────────────────────────────────────────────────────────────────────


class ProviderPool:
    def __init__(
        self,
        slots: list[Slot],
        store: SlotStore | MemorySlotStore,
        alert: AlertBackend | None = None,
        quota_warning_tokens: int = 0,
    ) -> None:
        self._slots = slots
        self._store = store
        self._alert = alert or LogAlertBackend()
        self._quota_warning_tokens = quota_warning_tokens

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        configs: list[SlotConfig],
        store: SlotStore | MemorySlotStore,
        alert: AlertBackend | None = None,
        quota_warning_tokens: int = 0,
    ) -> ProviderPool:
        """Build interleaved slot list from key configs.

        Column-first ordering: all keys' first model, then all keys' second
        model, etc.  This spreads quota pressure across keys before falling
        back to weaker models.
        """
        slots: list[Slot] = []
        max_models = max((len(c.models) for c in configs), default=0)
        for model_idx in range(max_models):
            for cfg in configs:
                if model_idx < len(cfg.models):
                    slots.append(
                        Slot(
                            key_id=cfg.key_id,
                            provider=cfg.provider,
                            api_key=cfg.api_key,
                            model=cfg.models[model_idx],
                            base_url=cfg.base_url,
                            extra=cfg.extra,
                        )
                    )
        return cls(
            slots=slots,
            store=store,
            alert=alert,
            quota_warning_tokens=quota_warning_tokens,
        )

    # ── Provider instantiation ────────────────────────────────────────────────

    def _build_provider(self, slot: Slot):
        from .providers import get_provider

        kwargs: dict[str, Any] = {
            "api_key": slot.api_key,
            "disable_cascade": True,  # pool manages cascading, not the provider
        }
        if slot.base_url:
            kwargs["base_url"] = slot.base_url
        kwargs.update(slot.extra)
        return get_provider(slot.provider, slot.model, **kwargs)

    # ── Slot availability ─────────────────────────────────────────────────────

    async def _available_slots(self) -> list[Slot]:
        result = []
        for slot in self._slots:
            if await self._store.is_available(slot.key_id, slot.model):
                result.append(slot)
        return result

    async def has_available_slots(self) -> bool:
        """Return True when at least one provider slot can be attempted."""
        for slot in self._slots:
            if await self._store.is_available(slot.key_id, slot.model):
                return True
        return False

    # ── Cooldown & alerting ───────────────────────────────────────────────────

    async def _cool_slot(self, slot: Slot, message: str) -> None:
        error_type, cooldown = _classify_error(message)
        await self._store.set_cooldown(slot.key_id, slot.model, cooldown, message)

        event_map = {
            "billing_cap": SlotEvent.EXHAUSTED,
            "rate_limited": SlotEvent.RATE_LIMITED,
            "auth_error": SlotEvent.AUTH_FAILED,
            "server_error": SlotEvent.UNHEALTHY,
            "unknown": SlotEvent.UNHEALTHY,
        }
        await self._alert.fire(
            AlertPayload(
                event=event_map.get(error_type, SlotEvent.UNHEALTHY),
                key_id=slot.key_id,
                model=slot.model,
                message=message,
                cooldown_seconds=cooldown,
            )
        )

    async def _record_usage(self, slot: Slot, usage: dict[str, int]) -> None:
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if not total:
            return
        await self._store.record_usage(slot.key_id, slot.model, total)
        if self._quota_warning_tokens > 0:
            state = await self._store.get_state(slot.key_id, slot.model)
            if state.tokens_this_month >= self._quota_warning_tokens:
                await self._alert.fire(
                    AlertPayload(
                        event=SlotEvent.QUOTA_WARNING,
                        key_id=slot.key_id,
                        model=slot.model,
                        message=(
                            f"Token usage {state.tokens_this_month:,} reached "
                            f"warning threshold {self._quota_warning_tokens:,}"
                        ),
                        tokens_used=state.tokens_this_month,
                    )
                )

    # ── Core streaming entrypoint ─────────────────────────────────────────────

    async def generate_with_fallback(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from the first healthy slot; fall back on retryable errors.

        Yields the same StreamEvent types as any individual LLMProvider so
        callers are provider-agnostic.
        """
        available = await self._available_slots()
        if not available:
            logger.error(
                "[Pool] No provider slots available — all cooling or misconfigured"
            )
            await self._alert.fire(
                AlertPayload(
                    event=SlotEvent.ALL_EXHAUSTED,
                    key_id="all",
                    model=None,
                    message="No provider slots available — all cooling or misconfigured",
                )
            )
            yield ErrorEvent(
                message="All provider slots are exhausted. Please try again later.",
                error_code="POOL_EXHAUSTED",
            )
            return

        for slot in available:
            provider = self._build_provider(slot)
            first_chunk = False
            slot_failed = False

            try:
                try:
                    async with asyncio.timeout(SLOT_GENERATION_TIMEOUT):
                        async for event in provider.generate(
                            messages=messages,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                        ):
                            if isinstance(event, TextDeltaEvent | ToolCallEvent):
                                first_chunk = True
                                yield event

                            elif isinstance(event, FinishEvent):
                                if not event.model:
                                    event.model = slot.model
                                await self._record_usage(slot, event.usage)
                                yield event
                                return  # clean success

                            elif isinstance(event, ErrorEvent):
                                error_type, _ = _classify_error(event.message)
                                if not first_chunk and _is_retryable(error_type):
                                    logger.warning(
                                        f"[Pool] {slot.slot_id} failed before first chunk "
                                        f"({error_type}) — trying next slot"
                                    )
                                    await self._cool_slot(slot, event.message)
                                    slot_failed = True
                                    break  # try next slot
                                else:
                                    # Mid-stream failure or unretryable — surface to caller
                                    await self._cool_slot(slot, event.message)
                                    yield event
                                    return
                except TimeoutError:
                    logger.warning(
                        "[Pool] %s timed out after %ss — cooling slot",
                        slot.slot_id,
                        SLOT_GENERATION_TIMEOUT,
                    )
                    await self._cool_slot(
                        slot,
                        f"timeout: no response within {SLOT_GENERATION_TIMEOUT}s",
                    )
                    slot_failed = True
            finally:
                await provider.close()

            if slot_failed:
                continue

        # Exhausted all available slots before yielding anything
        await self._alert.fire(
            AlertPayload(
                event=SlotEvent.ALL_EXHAUSTED,
                key_id="all",
                model=None,
                message="All available slots failed before yielding first chunk",
            )
        )
        yield ErrorEvent(
            message="All provider slots are exhausted. Please try again later.",
            error_code="POOL_EXHAUSTED",
        )

    # ── Non-streaming convenience wrapper ────────────────────────────────────

    async def generate_text(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Collect a full text response from the pool.

        Raises RuntimeError if all slots are exhausted or an unrecoverable
        ErrorEvent is returned.  Callers that need streaming should use
        generate_with_fallback() directly.
        """
        full_text: list[str] = []
        async for event in self.generate_with_fallback(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        ):
            if isinstance(event, TextDeltaEvent):
                full_text.append(event.text)
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
            elif isinstance(event, FinishEvent):
                break
        result = "".join(full_text)
        if not result:
            logger.warning(
                "[Pool] generate_text: model yielded no text deltas — returning empty string"
            )
        return result

    async def generate_json(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> dict | list:
        """Collect a response and parse it as JSON.

        Requests each provider's native structured-output mode where one
        exists (see LLMProvider.generate's json_mode). Still extracts the
        outermost JSON value from the raw text before parsing — models
        occasionally wrap output in markdown fences or trailing prose even
        in JSON mode, and providers without a native mode (Anthropic) rely on
        this entirely.

        Raises the same RuntimeError as generate_text on pool exhaustion, or
        json.JSONDecodeError if no valid JSON value can be extracted. Retrying
        on parse failure is the caller's policy, not this method's.
        """
        raw = await self.generate_text(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return json.loads(_extract_json(raw))

    # ── Usage-returning variants (for callers that bill/report per call) ───────

    async def generate_with_usage(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> tuple[str, dict, str | None]:
        """Like generate_text(), but also returns (usage, model) from the
        FinishEvent — generate_text() discards them after extracting text,
        which loses the token counts a billing/usage-tracking caller needs.
        """
        full_text: list[str] = []
        usage: dict = {}
        model: str | None = None
        async for event in self.generate_with_fallback(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        ):
            if isinstance(event, TextDeltaEvent):
                full_text.append(event.text)
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
            elif isinstance(event, FinishEvent):
                usage = event.usage
                model = event.model
                break
        result = "".join(full_text)
        if not result:
            logger.warning(
                "[Pool] generate_with_usage: model yielded no text deltas — returning empty string"
            )
        return result, usage, model

    async def generate_json_with_usage(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> tuple[dict | list, dict, str | None]:
        """Like generate_json(), but also returns (usage, model)."""
        raw, usage, model = await self.generate_with_usage(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return json.loads(_extract_json(raw)), usage, model

    # ── Admin introspection ───────────────────────────────────────────────────

    async def get_status(self) -> dict:
        """Return health summary for an admin/monitoring endpoint."""
        slots_status = []
        healthy = 0
        cooling = 0

        for slot in self._slots:
            state: SlotState = await self._store.get_state(slot.key_id, slot.model)
            now = time.time()
            is_avail = now >= state.cooldown_until

            if is_avail:
                healthy += 1
                status = "healthy"
            else:
                cooling += 1
                status = "cooling"

            slots_status.append(
                {
                    "slot_id": slot.slot_id,
                    "key_id": slot.key_id,
                    "provider": slot.provider,
                    "model": slot.model,
                    "status": status,
                    "cooldown_reason": state.cooldown_reason if not is_avail else None,
                    "cools_in_seconds": round(state.cooling_for_seconds)
                    if not is_avail
                    else None,
                    "tokens_today": state.tokens_today,
                    "tokens_this_month": state.tokens_this_month,
                    "last_error": state.last_error or None,
                    "last_error_at": state.last_error_at or None,
                }
            )

        return {
            "healthy_slots": healthy,
            "cooling_slots": cooling,
            "total_slots": len(self._slots),
            "slots": slots_status,
        }

    async def close(self) -> None:
        """Release pool resources. Idempotent."""
        self._slots.clear()


# ── LLMProvider adapter ─────────────────────────────────────────────────────


class PooledProvider(LLMProvider):
    """Adapts a ProviderPool to the LLMProvider interface.

    Lets provider-agnostic callers route through the pool's multi-key
    rotation/fallback without knowing about slots. The actual model that
    served a call is stamped on the FinishEvent by the pool; the static
    `model` attribute here only keeps `getattr(provider, "model", ...)`
    call sites working.
    """

    model = "provider_pool"

    def __init__(self, pool: ProviderPool) -> None:
        self._pool = pool

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        async for event in self._pool.generate_with_fallback(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        ):
            yield event

    async def close(self) -> None:
        """No-op — the pool is shared and owns its slot providers."""


# ── Platform singleton ────────────────────────────────────────────────────────

_platform_pool: ProviderPool | None = None


def init_platform_pool(
    store: SlotStore | MemorySlotStore,
    configs: list[SlotConfig],
    webhook_url: str | None = None,
    quota_warning_tokens: int = 0,
) -> ProviderPool:
    """Initialize the global platform pool from explicit slot configs.

    Call once at application startup. Pass a SlotStore (Redis) for production
    or MemorySlotStore for development/testing. Building `configs` from your
    app's own settings/env vars is the caller's job — this function has no
    opinion on where API keys come from.

    Example::

        pool = init_platform_pool(
            store=SlotStore(redis_client),
            configs=[
                SlotConfig(key_id="primary", provider="google",
                           api_key=os.environ["GOOGLE_API_KEY"],
                           models=["gemini-2.5-flash"]),
            ],
        )
    """
    global _platform_pool
    if _platform_pool is not None:
        logger.warning(
            "[Pool] init_platform_pool called while a pool already exists — "
            "replacing (old pool slots cleared)"
        )
        _platform_pool._slots.clear()

    alert = build_alert_backend(webhook_url)
    _platform_pool = ProviderPool.from_config(
        configs,
        store=store,
        alert=alert,
        quota_warning_tokens=quota_warning_tokens,
    )
    logger.info(
        f"[Pool] Initialized with {len(_platform_pool._slots)} slots "
        f"across {len(configs)} key(s)"
    )
    return _platform_pool


def get_platform_pool() -> ProviderPool:
    if _platform_pool is None:
        raise RuntimeError(
            "ProviderPool not initialized. Call init_platform_pool() at startup."
        )
    return _platform_pool
