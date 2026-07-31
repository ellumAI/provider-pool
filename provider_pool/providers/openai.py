"""OpenAI LLM provider.

Uses the `openai` Python SDK for streaming generation with function calling.
Translates OpenAI's native events into the provider-agnostic StreamEvent format.

Also works with OpenRouter and any OpenAI-compatible endpoint via base_url.
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

# Model ID substrings that indicate a non-chat model (embeddings, audio, image,
# moderation, or legacy completion-only) — excluded from list_models() output.
_NON_CHAT_MARKERS = (
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "davinci-002",
    "babbage-002",
)


async def list_models(api_key: str, base_url: str | None = None) -> list[str]:
    """List chat-completion-capable model IDs for this key.

    Best-effort filter by name pattern — OpenAI's /models endpoint doesn't
    expose a capability flag, so embeddings/audio/image/moderation models are
    excluded by substring match. Review the result before using it; also works
    against OpenRouter and other OpenAI-compatible endpoints via base_url,
    though their naming conventions may not match this filter.
    """
    from openai import AsyncOpenAI

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    try:
        resp = await client.models.list()
        return [
            m.id
            for m in resp.data
            if not any(marker in m.id.lower() for marker in _NON_CHAT_MARKERS)
        ]
    finally:
        await client.close()


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the official openai SDK."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        **_kwargs,
    ):
        if not api_key:
            raise ValueError("OpenAIProvider requires an explicit api_key")

        self._api_key = api_key
        self._base_url = base_url  # set to https://openrouter.ai/api/v1 for OpenRouter
        # Explicit timeout so a stalled stream can't ride the SDK's 600s default.
        # read bounds the gap between streamed chunks; the SDK retries transient
        # failures up to max_retries.
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None

        # Strip "openai/" prefix for standard OpenAI calls (when base_url is not OpenRouter)
        is_openrouter = base_url and "openrouter.ai" in base_url
        if model.startswith("openai/") and not is_openrouter:
            self.model_name = model[len("openai/") :]
        else:
            self.model_name = model

    def _ensure_client(self) -> None:
        if self._client is None:
            import httpx
            from openai import AsyncOpenAI

            client_kwargs: dict = {
                "api_key": self._api_key,
                "timeout": httpx.Timeout(
                    connect=10.0, read=self._timeout, write=30.0, pool=10.0
                ),
                "max_retries": self._max_retries,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**client_kwargs)

    def _build_tools(self, tools: list[ToolDefinition] | None) -> list[dict] | None:
        """Convert ToolDefinition list → OpenAI function format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list → OpenAI messages format."""
        result = []
        for msg in messages:
            if msg.role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                m: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    m["content"] = msg.content
                m["tool_calls"] = [
                    {
                        "id": tc.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append(m)
            else:
                result.append(
                    {
                        "role": msg.role,
                        "content": msg.content or "",
                    }
                )
        return result

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from OpenAI."""
        self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        oai_tools = self._build_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools

        try:
            stream = await self._client.chat.completions.create(**kwargs)

            # Buffer for tool calls being assembled from deltas
            tool_call_buffers: dict[int, dict] = {}

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Text delta
                if delta.content:
                    yield TextDeltaEvent(text=delta.content)

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        buf = tool_call_buffers[idx]
                        if tc_delta.id:
                            buf["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                buf["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                buf["arguments"] += tc_delta.function.arguments

                # Check finish reason
                finish_reason = (
                    chunk.choices[0].finish_reason if chunk.choices else None
                )
                if finish_reason:
                    # Emit buffered tool calls
                    for buf in tool_call_buffers.values():
                        try:
                            args = (
                                json.loads(buf["arguments"]) if buf["arguments"] else {}
                            )
                        except json.JSONDecodeError:
                            args = {}
                        yield ToolCallEvent(
                            tool_call_id=buf["id"],
                            tool_name=buf["name"],
                            arguments=args,
                        )
                    tool_call_buffers.clear()

                    usage_info = {}
                    if chunk.usage:
                        usage_info = {
                            "input_tokens": chunk.usage.prompt_tokens or 0,
                            "output_tokens": chunk.usage.completion_tokens or 0,
                        }
                    yield FinishEvent(finish_reason=finish_reason, usage=usage_info)

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}", exc_info=True)
            yield ErrorEvent(message=str(e))

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
