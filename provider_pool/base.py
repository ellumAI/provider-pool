"""Abstract base for LLM providers.

Every provider must implement `generate()` — which yields StreamEvent
objects as the model produces text and tool calls. The agent loop is
provider-agnostic; it only speaks StreamEvent.

To add a new provider:
  1. Create a module in provider_pool/providers/
  2. Subclass LLMProvider
  3. Register it in providers/__init__.py::get_provider()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

# ──────────────────────────────────────────────────────────────────────
#  Stream Events — provider-agnostic output format
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TextDeltaEvent:
    """Incremental text chunk from the model."""

    text: str


@dataclass
class ToolCallEvent:
    """The model wants to call a tool."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None  # Gemini 3.x thought signature
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent:
    """Result fed back to the model after tool execution."""

    tool_call_id: str
    result: Any


@dataclass
class FinishEvent:
    """Generation is complete for this turn."""

    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None


@dataclass
class ErrorEvent:
    """Provider-level error."""

    message: str
    error_code: str | None = None
    details: dict | None = None


# Union type for convenience
StreamEvent = TextDeltaEvent | ToolCallEvent | FinishEvent | ErrorEvent


@dataclass
class ToolDefinition:
    """Provider-agnostic tool schema passed to the model.

    `parameters` follows JSON Schema format, matching the
    Gemini / OpenAI / Anthropic function-calling specs.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    requires_approval: bool = False


@dataclass
class ChatMessage:
    """A single message for the LLM context window."""

    role: str  # "user" | "assistant" | "system" | "tool"
    content: str | None = None
    tool_calls: list[ToolCallEvent] | None = None
    tool_call_id: str | None = None  # for role="tool" messages
    name: str | None = None  # tool name for role="tool" messages


# ──────────────────────────────────────────────────────────────────────
#  Provider ABC
# ──────────────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract interface for LLM providers.

    Subclasses implement `generate()` which accepts a conversation
    history + tool definitions and yields stream events.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the model.

        Yields TextDeltaEvent / ToolCallEvent / FinishEvent / ErrorEvent.
        The caller (agent loop) handles tool execution and re-invocation.

        json_mode requests the provider's native structured-output mode where
        one exists (Gemini's response_mime_type, OpenAI's response_format).
        Anthropic has no native equivalent — providers without one accept the
        flag and ignore it; callers still get free-text they must parse.
        """
        ...  # pragma: no cover

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (HTTP clients, etc.)."""
        ...
