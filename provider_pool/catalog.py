"""Curated model catalog with capability flags.

BYOK lets users bring keys for any supported provider, but agentic
workloads (multi-step tool loops) have a hard capability floor: the model
must support tool calling and streaming, and needs a large context window
to survive long tool transcripts (GitHub Copilot CLI enforces the same
floor and recommends >=128k context). This module is the platform-curated
allowlist those flows filter against — model *availability* still comes
from the provider's live catalog (see ``list_models``); this file only
says what the platform vouches for and what each model can do.

Catalog entries use family stems (e.g. ``claude-sonnet-4-5``) that match
dated/suffixed release ids (``claude-sonnet-4-5-20250929``) via
longest-stem lookup, so point releases don't require catalog edits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = [
    "AGENT_MIN_CONTEXT",
    "CATALOG",
    "ModelInfo",
    "annotate_models",
    "get_catalog",
    "is_agent_capable",
    "lookup",
]

# Minimum context window (tokens) for a model to drive agentic tool loops.
AGENT_MIN_CONTEXT = 128_000


@dataclass(frozen=True)
class ModelInfo:
    """Capability flags for one curated model family."""

    provider: str
    model: str  # family stem, matched by longest prefix against live ids
    display_name: str
    context_window: int
    tool_calling: bool = True
    streaming: bool = True
    structured_output: bool = True
    vision: bool = False


CATALOG: tuple[ModelInfo, ...] = (
    # ── Google ────────────────────────────────────────────────────────
    ModelInfo(
        provider="google",
        model="gemini-3-pro",
        display_name="Gemini 3 Pro",
        context_window=1_000_000,
        vision=True,
    ),
    ModelInfo(
        provider="google",
        model="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        context_window=1_000_000,
        vision=True,
    ),
    ModelInfo(
        provider="google",
        model="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=1_000_000,
        vision=True,
    ),
    ModelInfo(
        provider="google",
        model="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
        context_window=1_000_000,
        vision=True,
    ),
    # ── OpenAI ────────────────────────────────────────────────────────
    ModelInfo(
        provider="openai",
        model="gpt-5.1",
        display_name="GPT-5.1",
        context_window=400_000,
        vision=True,
    ),
    ModelInfo(
        provider="openai",
        model="gpt-5-mini",
        display_name="GPT-5 mini",
        context_window=400_000,
        vision=True,
    ),
    ModelInfo(
        provider="openai",
        model="gpt-5-nano",
        display_name="GPT-5 nano",
        context_window=400_000,
        vision=True,
    ),
    ModelInfo(
        provider="openai",
        model="gpt-5",
        display_name="GPT-5",
        context_window=400_000,
        vision=True,
    ),
    ModelInfo(
        provider="openai",
        model="gpt-4o-mini",
        display_name="GPT-4o mini",
        context_window=128_000,
        vision=True,
    ),
    ModelInfo(
        provider="openai",
        model="gpt-4o",
        display_name="GPT-4o",
        context_window=128_000,
        vision=True,
    ),
    # ── Anthropic ─────────────────────────────────────────────────────
    ModelInfo(
        provider="anthropic",
        model="claude-opus-4-5",
        display_name="Claude Opus 4.5",
        context_window=200_000,
        vision=True,
    ),
    ModelInfo(
        provider="anthropic",
        model="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        context_window=200_000,
        vision=True,
    ),
    ModelInfo(
        provider="anthropic",
        model="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        vision=True,
    ),
    ModelInfo(
        provider="anthropic",
        model="claude-sonnet-4",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        vision=True,
    ),
)


def get_catalog(provider: str | None = None) -> list[ModelInfo]:
    """Return curated entries, optionally filtered to one provider."""
    if provider is None:
        return list(CATALOG)
    return [m for m in CATALOG if m.provider == provider]


def lookup(provider: str, model_id: str) -> ModelInfo | None:
    """Match a live model id to its curated family entry.

    A catalog stem matches when the id equals it or extends it at a
    ``-``/``.``/``@`` boundary; the longest matching stem wins so
    ``gemini-2.5-flash-lite`` resolves to its own entry rather than
    ``gemini-2.5-flash``.
    """
    best: ModelInfo | None = None
    for entry in CATALOG:
        if entry.provider != provider:
            continue
        stem = entry.model
        if (
            model_id == stem
            or model_id.startswith((stem + "-", stem + ".", stem + "@"))
        ) and (best is None or len(stem) > len(best.model)):
            best = entry
    return best


def is_agent_capable(info: ModelInfo, min_context: int = AGENT_MIN_CONTEXT) -> bool:
    """Whether a model meets the floor for agentic tool loops."""
    return info.tool_calling and info.streaming and info.context_window >= min_context


def annotate_models(provider: str, model_ids: list[str]) -> list[dict]:
    """Annotate live model ids with curated capability flags.

    Returns one dict per id: ``{"model", "in_catalog", "agent_capable",
    **capability flags}``. Ids not in the catalog get ``in_catalog=False``
    and ``agent_capable=False`` — usable for plain chat at the caller's
    discretion, never for agent flows.
    """
    out: list[dict] = []
    for model_id in model_ids:
        info = lookup(provider, model_id)
        if info is None:
            out.append(
                {
                    "model": model_id,
                    "in_catalog": False,
                    "agent_capable": False,
                }
            )
        else:
            flags = asdict(info)
            flags.pop("provider")
            flags.pop("model")
            out.append(
                {
                    "model": model_id,
                    "in_catalog": True,
                    "agent_capable": is_agent_capable(info),
                    **flags,
                }
            )
    return out
