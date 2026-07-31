# provider-pool

Multi-key, multi-model LLM provider pool for Google Gemini, OpenAI, and Anthropic —
with circuit-breaking, cooldown-based fallback, and Redis-backed (or in-memory)
slot state.

## Why

Calling a single API key/model pair directly means one rate limit, billing cap,
or outage takes your app down. `ProviderPool` holds an ordered list of
`(api_key, model)` slots — spread across keys before falling back to weaker
models — and streams from the first healthy one, cooling down and retrying the
next slot on a retryable error.

## Install

```bash
pip install "provider-pool[google,redis]"
# or: openai, anthropic, all
```

## Quick-start (no Redis, single-process)

```python
from provider_pool import ProviderPool, SlotConfig, MemorySlotStore

pool = ProviderPool.from_config(
    [
        SlotConfig(key_id="primary", provider="google",
                   api_key="AIzaSy...", models=["gemini-2.5-flash"]),
        SlotConfig(key_id="fallback", provider="openai",
                   api_key="sk-...", models=["gpt-4o-mini"]),
    ],
    store=MemorySlotStore(),
)

async for event in pool.generate_with_fallback(messages=[...]):
    ...
```

## Quick-start (with Redis, multi-process)

```python
import redis.asyncio as aioredis
from provider_pool import ProviderPool, SlotConfig, SlotStore

redis = aioredis.from_url("redis://localhost")
store = SlotStore(redis)
await store.load_scripts()  # pre-load Lua scripts once at startup

pool = ProviderPool.from_config(configs=[...], store=store)
```

## Discovering available models

Rather than the pool auto-cascading through every model a key can see (which
would break the "best model first" ordering the pool exists to guarantee),
`list_models()` fetches the catalog for a key so *you* still choose and order
what goes into `SlotConfig.models`:

```python
from provider_pool import list_models, ProviderPool, SlotConfig, MemorySlotStore

models = await list_models("google", api_key="AIzaSy...")
# ["gemini-2.5-flash", "gemini-2.5-pro", ...] — generateContent-capable only

pool = ProviderPool.from_config(
    [SlotConfig(key_id="primary", provider="google", api_key="AIzaSy...",
                models=models[:2])],  # still your call which ones and in what order
    store=MemorySlotStore(),
)
```

Filtering by provider:
- **google** — filtered to models where the API reports `generateContent` support.
- **anthropic** — no filtering needed; the Messages API catalog is chat-only.
- **openai** — best-effort filter by name pattern (excludes `embedding`,
  `whisper`, `tts`, `dall-e`, `moderation`, and legacy completion-only models).
  OpenAI's `/models` endpoint doesn't expose a capability flag, so review the
  result — this is a heuristic, not a guarantee.

## Slot ordering

Slots are ordered column-first across keys so load spreads before quality
degrades:

```
configs:
  key_A → [gemini-2.5-flash, gemini-3-flash-preview]
  key_B → [gemini-2.5-flash, gemini-3.1-flash-lite-preview]

slot order:
  0. (key_A, gemini-2.5-flash)         ← try best model on all keys first
  1. (key_B, gemini-2.5-flash)
  2. (key_A, gemini-3-flash-preview)   ← only then fall to weaker models
  3. (key_B, gemini-3.1-flash-lite-preview)
```

## Error classification → cooldown

| error_type    | cause                          | cooldown |
|---------------|---------------------------------|----------|
| billing_cap   | monthly spend cap exceeded      | 24 h     |
| rate_limited  | per-minute/day quota            | 60 s     |
| server_error  | transient 5xx                   | 30 s     |
| auth_error    | 401/403, dead key                | 30 days  |
| unknown       | anything else                   | 30 s     |

## Namespacing (multi-tenant / BYOK)

`SlotStore` and `MemorySlotStore` take a `prefix` so you can run one pool per
tenant against the same Redis instance:

```python
store = SlotStore(redis, prefix=f"provider_pool:org:{org_id}:slot")
```

## Adding a provider

1. Create a module in `provider_pool/providers/`.
2. Subclass `LLMProvider`, implement `generate()` and `close()`.
3. Register it in `provider_pool/providers/__init__.py::get_provider()`.
