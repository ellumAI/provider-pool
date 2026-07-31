"""Chain two provider pools: a primary tried first, a fallback safety net.

Built for BYOK routing (org keys first, shared platform keys as backup —
the same semantics OpenRouter uses for user-supplied keys), but the class
is generic over any two pool-like objects.

Duck-typing note: this class deliberately avoids ``isinstance`` checks on
stream events and matches on ``error_code`` instead, so it can wrap pool
objects from a *different copy* of this library (e.g. an app's vendored
``app/providers`` package) whose event classes are distinct types.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol

__all__ = ["ChainedPool"]

# ProviderPool yields this error_code only when it has produced no output
# at all (no healthy slots, or every slot failed before its first chunk).
# That guarantee is what makes chaining safe: re-running the request on
# the fallback pool can never duplicate streamed output.
_EXHAUSTED_CODE = "POOL_EXHAUSTED"


class _PoolLike(Protocol):
    def generate_with_fallback(
        self,
        messages: list,
        tools: list | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncGenerator[Any, None]:
        ...

    async def get_status(self) -> dict:
        ...


class ChainedPool:
    """Try the primary pool; on exhaustion re-run on the fallback pool.

    The primary's POOL_EXHAUSTED error is suppressed and the request is
    replayed against the fallback, so callers see one clean stream. Any
    other error — including exhaustion of the fallback itself — surfaces
    unchanged. Mid-stream errors are never chained (output was already
    delivered; a replay would duplicate it), matching ProviderPool's own
    commit-on-first-chunk rule.
    """

    def __init__(self, primary: _PoolLike, fallback: _PoolLike) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate_with_fallback(
        self,
        messages: list,
        tools: list | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncGenerator[Any, None]:
        yielded_any = False
        primary_exhausted = False

        async for event in self.primary.generate_with_fallback(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        ):
            if (
                not yielded_any
                and getattr(event, "error_code", None) == _EXHAUSTED_CODE
            ):
                primary_exhausted = True
                break
            yielded_any = True
            yield event

        if not primary_exhausted:
            return

        async for event in self.fallback.generate_with_fallback(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        ):
            yield event

    async def get_status(self) -> dict:
        """Merged health summary for admin/status endpoints."""
        return {
            "chained": True,
            "primary": await self.primary.get_status(),
            "fallback": await self.fallback.get_status(),
        }
