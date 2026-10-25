"""Google Gemini LLM provider.

Uses the `google-genai` SDK for streaming generation with function calling.
Translates Gemini's native events into the provider-agnostic StreamEvent format.

When used through ProviderPool, disable_cascade=True is always set — the pool
manages fallback externally. `cascade_models` is only relevant when running
GeminiProvider standalone (outside a pool): pass the ordered list of models to
step through on quota exhaustion.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

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

# Transient errors that should be retried (includes Gemini 5xx and 429)
_RETRYABLE_ERRORS = (
    OSError,
    ConnectionError,
    TimeoutError,
    httpx.ConnectError,
    ServerError,  # google.genai 502/503/500 responses
)
_MAX_RETRIES = 4


async def list_models(api_key: str) -> list[str]:
    """List Gemini model IDs this API key can access that support generateContent.

    Returns bare model names (e.g. "gemini-2.5-flash", without the "models/"
    prefix the API uses internally) suitable for use directly in
    SlotConfig.models. Excludes models that don't support generateContent
    (e.g. embedding-only models), where the SDK reports that capability.
    """
    client = genai.Client(api_key=api_key)
    models: list[str] = []
    pager = await client.aio.models.list()
    async for model in pager:
        actions = getattr(model, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        name = model.name or ""
        models.append(name.removeprefix("models/"))
    return models


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the google-genai SDK."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        disable_cascade: bool = False,
        cascade_models: list[str] | None = None,
        **_kwargs,
    ):
        if not api_key:
            raise ValueError("GeminiProvider requires an explicit api_key")

        self._primary_model = model
        self._api_key = api_key
        self._client = None
        self._sync_client = None
        # When managed by ProviderPool, cascade is handled externally — use only
        # the specified model so the pool controls which slot is tried next.
        self._disable_cascade = disable_cascade

        if disable_cascade:
            self.model_name = model
            self._cascade_models = [model]
            self._fallback_tier = 0
        else:
            # Standalone: step through the provided models (or just the one model)
            self._cascade_models = cascade_models or [model]
            self.model_name = self._cascade_models[0]
            self._fallback_tier = 0

        self._using_fallback = False

    def _ensure_client(self) -> None:
        """Lazily initialise the Gemini client."""
        if self._client is None:
            self._sync_client = httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
            )
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(httpx_client=self._sync_client),
            )

    # Fields that are valid JSON Schema but unsupported by Gemini's FunctionDeclaration
    _GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
        {
            "additionalProperties",
            "$schema",
            "$ref",
            "definitions",
            "$defs",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "const",
            "examples",
            "default",
            "readOnly",
            "writeOnly",
            "deprecated",
            "contentEncoding",
            "contentMediaType",
        }
    )

    def _sanitize_schema(self, schema: Any) -> Any:
        """Recursively remove/normalize schema fields unsupported by Gemini."""
        if isinstance(schema, str):
            # Bare type string — wrap into a proper schema dict
            return {"type": schema}
        if not isinstance(schema, dict):
            return {"type": "string"}

        result = {}
        for key, value in schema.items():
            if key in self._GEMINI_UNSUPPORTED_SCHEMA_KEYS or key.startswith("x-"):
                continue
            if key == "properties" and isinstance(value, dict):
                # Ensure every property value is a dict schema, not a bare string/type
                sanitized_props = {}
                for prop_name, prop_schema in value.items():
                    if isinstance(prop_schema, str):
                        sanitized_props[prop_name] = {"type": prop_schema}
                    elif isinstance(prop_schema, dict):
                        sanitized_props[prop_name] = self._sanitize_schema(prop_schema)
                    else:
                        sanitized_props[prop_name] = {"type": "string"}
                result[key] = sanitized_props
            elif isinstance(value, dict):
                result[key] = self._sanitize_schema(value)
            elif key == "required" and isinstance(value, list):
                # required must be a list of strings — never recurse-sanitize items
                clean = []
                for item in value:
                    if isinstance(item, str):
                        clean.append(item)
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("type") or item.get("title")
                        if isinstance(name, str):
                            clean.append(name)
                result[key] = clean
            elif key == "enum" and isinstance(value, list):
                result[key] = [str(v) for v in value]
            elif key == "type" and isinstance(value, list):
                # JSON Schema allows type arrays e.g. ["string", "null"] — Gemini doesn't
                non_null = [t for t in value if t != "null"]
                result[key] = non_null[0] if non_null else "string"
            elif isinstance(value, list):
                result[key] = [self._sanitize_schema(item) for item in value]
            else:
                result[key] = value

        schema_type = result.get("type")
        if schema_type == "array":
            items = result.get("items")
            if not isinstance(items, dict) or not items:
                result["items"] = {"type": "object", "properties": {}}
            elif "type" not in items:
                items["type"] = self._infer_schema_type(items) or "object"
                if items["type"] == "object":
                    items.setdefault("properties", {})

        if schema_type == "object":
            result.setdefault("properties", {})

        if not schema_type:
            inferred_type = self._infer_schema_type(result)
            if inferred_type:
                result["type"] = inferred_type

        return result

    def _infer_schema_type(self, schema: dict[str, Any]) -> str | None:
        """Infer a minimal Gemini-compatible type for partial JSON schemas."""
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        if "enum" in schema:
            return "string"
        return None

    def _build_tools(self, tools: list[ToolDefinition] | None) -> list[dict] | None:
        """Convert ToolDefinition list → Gemini function declarations."""
        if not tools:
            return None

        declarations = []
        for tool in tools:
            parameters = self._sanitize_schema(tool.parameters)
            if not isinstance(parameters, dict) or not parameters.get("type"):
                parameters = {"type": "object", "properties": {}, "required": []}
            decl = {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            }
            declarations.append(decl)
        return declarations

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict]]:
        """Convert ChatMessage list → Gemini content format.

        Returns (system_instruction, contents).
        """
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            if msg.role == "tool":
                # Tool result → Gemini function response
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.name,
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant with tool calls — preserve thought_signature
                parts = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    fc_part = {
                        "function_call": {
                            "name": tc.tool_name,
                            "args": tc.arguments,
                        }
                    }
                    # Gemini 3.x requires thought_signature to be replayed
                    if hasattr(tc, "thought_signature") and tc.thought_signature:
                        fc_part["thought_signature"] = tc.thought_signature
                    parts.append(fc_part)
                contents.append({"role": "model", "parts": parts})
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    {
                        "role": role,
                        "parts": [{"text": msg.content or ""}],
                    }
                )

        return system_instruction, contents

    def _stream_in_thread(
        self,
        contents: list[dict],
        generate_config: types.GenerateContentConfig,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run the synchronous Gemini stream in a background thread.

        Pushes parsed StreamEvent objects onto an asyncio.Queue so the
        async generator can yield them without blocking the event loop.
        """
        import time as _time

        # When pool-managed, only try the single assigned model.
        remaining = (
            [self._primary_model]
            if self._disable_cascade
            else self._cascade_models[self._fallback_tier :]
        ) or [self.model_name]

        last_exc: Exception | None = None

        for tier_offset, model in enumerate(remaining):
            current_tier = self._fallback_tier + tier_offset
            self.model_name = model
            self._using_fallback = current_tier > 0

            if tier_offset > 0:
                logger.warning(
                    f"Stepping to model tier {current_tier} ({model}) "
                    f"after exhausting {remaining[tier_offset - 1]}"
                )

            for attempt in range(_MAX_RETRIES):
                try:
                    assert self._client is not None
                    response = self._client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=generate_config,
                    )

                    for chunk in response:
                        if chunk.text:
                            loop.call_soon_threadsafe(
                                queue.put_nowait, TextDeltaEvent(text=chunk.text)
                            )

                        if chunk.candidates:
                            for candidate in chunk.candidates:
                                if candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if (
                                            hasattr(part, "function_call")
                                            and part.function_call
                                        ):
                                            fc = part.function_call
                                            sig = getattr(
                                                part, "thought_signature", None
                                            )
                                            loop.call_soon_threadsafe(
                                                queue.put_nowait,
                                                ToolCallEvent(
                                                    tool_call_id=f"tc-{uuid.uuid4().hex[:12]}",
                                                    tool_name=fc.name,
                                                    arguments=(
                                                        dict(fc.args) if fc.args else {}
                                                    ),
                                                    thought_signature=sig,
                                                ),
                                            )

                    usage_info = {}
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage_info = {
                            "input_tokens": getattr(
                                response.usage_metadata, "prompt_token_count", 0
                            ),
                            "output_tokens": getattr(
                                response.usage_metadata, "candidates_token_count", 0
                            ),
                        }

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        FinishEvent(finish_reason="stop", usage=usage_info),
                    )
                    return  # ── Success ──

                except ClientError as exc:
                    last_exc = exc
                    is_quota = getattr(
                        exc, "status", ""
                    ) == "RESOURCE_EXHAUSTED" or "429" in str(exc)
                    if is_quota:
                        # Quota exhausted — skip remaining retries and step down
                        logger.warning(
                            f"Quota exhausted for {model} — "
                            f"stepping to next model in cascade"
                        )
                        break
                    logger.error(
                        "Gemini request rejected: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ErrorEvent(message=str(exc)),
                    )
                    return

                except _RETRYABLE_ERRORS as exc:
                    last_exc = exc
                    wait = (attempt + 1) * 3
                    logger.warning(
                        f"Gemini transient error (attempt {attempt + 1}/{_MAX_RETRIES}): "
                        f"{type(exc).__name__}: {exc} — retrying in {wait}s"
                    )
                    _time.sleep(wait)
                    # On last attempt fall through to next model in cascade

                except Exception as exc:
                    logger.error(f"Gemini generation error: {exc}", exc_info=True)
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ErrorEvent(message=str(exc)),
                    )
                    return

        # All cascade models exhausted
        loop.call_soon_threadsafe(
            queue.put_nowait,
            ErrorEvent(message=f"All models exhausted. Last error: {last_exc}"),
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Gemini.

        The synchronous google-genai SDK is run in a background thread
        so that it does not block the asyncio event loop. This keeps
        other coroutines responsive while multiple agents stream concurrently.
        """
        self._ensure_client()

        system_instruction, contents = self._convert_messages(messages)
        tool_declarations = self._build_tools(tools)

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        generate_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            # Disable AFC — the caller manages tool execution itself. Without
            # this, the SDK tries to auto-execute function_calls and returns
            # empty responses.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
            # Disable thinking — thinking models emit thought_signatures on
            # function calls, which must be replayed exactly or Gemini returns
            # 400 INVALID_ARGUMENT. Budget=0 turns off the thinking pass
            # entirely, keeping function-call history simple and replayable.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            **config_kwargs,
        )

        if tool_declarations:
            generate_config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**decl) for decl in tool_declarations
                    ]
                )
            ]

        # Bridge between synchronous thread and async generator
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Run the blocking Gemini stream in a thread pool worker. The context
        # copy keeps contextvars (e.g. the caller's OpenTelemetry span) visible
        # in the thread, as asyncio.to_thread does.
        thread_future = loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            contextvars.copy_context().run,
            self._stream_in_thread,
            contents,
            generate_config,
            queue,
            loop,
        )

        # Use a sentinel to signal that the thread is done
        _stream_done = object()

        async def _await_thread():
            """Wait for the thread to complete, then signal the queue."""
            try:
                await thread_future
            except Exception:
                pass  # Exception is retrieved via thread_future.exception()
            finally:
                queue.put_nowait(_stream_done)

        # Start a task that will signal completion
        completion_task = asyncio.ensure_future(_await_thread())

        try:
            while True:
                event = await queue.get()

                # Thread finished — check for exception
                if event is _stream_done:
                    # Drain any remaining items before the sentinel
                    while not queue.empty():
                        remaining = queue.get_nowait()
                        if remaining is _stream_done:
                            continue
                        yield remaining
                        if isinstance(remaining, FinishEvent | ErrorEvent):
                            return

                    # Check if thread raised an exception
                    if thread_future.done():
                        try:
                            thread_exc = thread_future.exception()
                        except asyncio.CancelledError:
                            thread_exc = None
                        if thread_exc:
                            logger.error(f"Stream thread failed: {thread_exc}")
                            yield ErrorEvent(message=str(thread_exc))
                    return

                yield event

                if isinstance(event, FinishEvent | ErrorEvent):
                    return

        except Exception as e:
            logger.error(f"Gemini generate wrapper error: {e}", exc_info=True)
            yield ErrorEvent(message=str(e))
        finally:
            completion_task.cancel()
            # Ensure the thread completes even if the consumer stops early
            if not thread_future.done():
                thread_future.cancel()

    async def close(self) -> None:
        """Release the Gemini client and its underlying httpx connection pool."""
        self._client = None
        if self._sync_client is not None:
            with contextlib.suppress(Exception):
                self._sync_client.close()
            self._sync_client = None
