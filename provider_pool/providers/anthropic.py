"""Anthropic LLM provider.

Uses the `anthropic` Python SDK for streaming generation with tool use.
Translates Anthropic's native events into the provider-agnostic StreamEvent format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..base import (
    ChatMessage,
    ErrorEvent,
    FinishEvent,
    LLMProvider,
    StreamEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


async def list_models(api_key: str) -> list[str]:
    """List Claude model IDs available to this API key.

    Anthropic's catalog is Messages-API-only, so no capability filtering is
    needed here (unlike OpenAI's, which mixes in embeddings/audio/image models).
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        resp = await client.models.list()
        return [m.id for m in resp.data]
    finally:
        await client.close()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the official SDK."""

    def __init__(self, model: str, api_key: str = "", **_kwargs):
        if not api_key:
            raise ValueError("AnthropicProvider requires an explicit api_key")

        self.model_name = model
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    def _build_tools(self, tools: list[ToolDefinition] | None) -> list[dict] | None:
        """Convert ToolDefinition list → Anthropic tool format."""
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict]]:
        """Convert ChatMessage list → Anthropic format (system, messages)."""
        system = None
        result = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
                continue

            if msg.role == "tool":
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.tool_call_id,
                            "name": tc.tool_name,
                            "input": tc.arguments,
                        }
                    )
                result.append({"role": "assistant", "content": content})
            else:
                result.append(
                    {
                        "role": msg.role,
                        "content": msg.content or "",
                    }
                )

        return system, result

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,  # noqa: ARG002 — no native JSON mode; caller must parse
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Anthropic Claude."""
        self._ensure_client()

        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._build_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                current_tool_id = ""
                current_tool_name = ""
                tool_input_json = ""

                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_id = event.content_block.id
                            current_tool_name = event.content_block.name
                            tool_input_json = ""

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield TextDeltaEvent(text=event.delta.text)
                        elif event.delta.type == "input_json_delta":
                            tool_input_json += event.delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool_id:
                            try:
                                args = (
                                    json.loads(tool_input_json)
                                    if tool_input_json
                                    else {}
                                )
                            except json.JSONDecodeError:
                                args = {}
                            yield ToolCallEvent(
                                tool_call_id=current_tool_id,
                                tool_name=current_tool_name,
                                arguments=args,
                            )
                            current_tool_id = ""
                            current_tool_name = ""
                            tool_input_json = ""

                    elif event.type == "message_stop":
                        usage_info = {}
                        if hasattr(stream, "current_message_snapshot"):
                            snap = stream.current_message_snapshot
                            if snap.usage:
                                usage_info = {
                                    "input_tokens": snap.usage.input_tokens,
                                    "output_tokens": snap.usage.output_tokens,
                                }
                        finish_reason = "end_turn"
                        if hasattr(stream, "current_message_snapshot"):
                            finish_reason = (
                                stream.current_message_snapshot.stop_reason
                                or "end_turn"
                            )
                        yield FinishEvent(finish_reason=finish_reason, usage=usage_info)

        except Exception as e:
            logger.error(f"Anthropic generation error: {e}", exc_info=True)
            yield ErrorEvent(message=str(e))

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
