"""provider_pool — multi-key, multi-model LLM provider pool with circuit-breaking.

Quick-start (with Redis)::

    import redis.asyncio as aioredis
    from provider_pool import ProviderPool, SlotConfig, SlotStore

    redis = aioredis.from_url("redis://localhost")
    pool = ProviderPool.from_config(
        configs=[
            SlotConfig(
                key_id="primary",
                provider="google",
                api_key="AIzaSy...",
                models=["gemini-2.5-flash", "gemini-3.1-flash-lite-preview"],
            ),
            SlotConfig(
                key_id="fallback",
                provider="openai",
                api_key="sk-...",
                models=["gpt-4o-mini"],
            ),
        ],
        store=SlotStore(redis),
    )

    async for event in pool.generate_with_fallback(messages):
        if isinstance(event, TextDeltaEvent):
            print(event.text, end="", flush=True)

Quick-start (no Redis, single-process)::

    from provider_pool import ProviderPool, SlotConfig, MemorySlotStore

    pool = ProviderPool.from_config(configs=[...], store=MemorySlotStore())
"""

from .alerts import (
    AlertBackend,
    AlertPayload,
    CompositeAlertBackend,
    LogAlertBackend,
    SlotEvent,
    WebhookAlertBackend,
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
    ToolResultEvent,
)
from .catalog import (
    AGENT_MIN_CONTEXT,
    CATALOG,
    ModelInfo,
    annotate_models,
    get_catalog,
    is_agent_capable,
    lookup,
)
from .chained import ChainedPool
from .pool import (
    PooledProvider,
    ProviderPool,
    Slot,
    SlotConfig,
    get_platform_pool,
    init_platform_pool,
)
from .providers import get_provider, list_models
from .slot_store import MemorySlotStore, SlotState, SlotStore

__all__ = [
    # Pool
    "ProviderPool",
    "PooledProvider",
    "ChainedPool",
    "Slot",
    "SlotConfig",
    "init_platform_pool",
    "get_platform_pool",
    "get_provider",
    "list_models",
    # Catalog
    "AGENT_MIN_CONTEXT",
    "CATALOG",
    "ModelInfo",
    "annotate_models",
    "get_catalog",
    "is_agent_capable",
    "lookup",
    # Stores
    "SlotStore",
    "MemorySlotStore",
    "SlotState",
    # Events
    "StreamEvent",
    "TextDeltaEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "FinishEvent",
    "ErrorEvent",
    # Messages / tools
    "ChatMessage",
    "ToolDefinition",
    "LLMProvider",
    # Alerts
    "AlertBackend",
    "AlertPayload",
    "SlotEvent",
    "LogAlertBackend",
    "WebhookAlertBackend",
    "CompositeAlertBackend",
    "build_alert_backend",
]
