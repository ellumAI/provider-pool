"""Tool results sent back to Gemini must use a role the API accepts.

Repro: gemini-3.5-flash rejects the follow-up turn after any tool call with
400 "Role 'function' is not supported".
"""

from __future__ import annotations

import asyncio
import os

import pytest

from provider_pool import ChatMessage, ErrorEvent, TextDeltaEvent, ToolCallEvent
from provider_pool import ToolDefinition
from provider_pool.providers.google import GeminiProvider

MODEL = os.environ.get("GEMINI_TEST_MODEL", "gemini-3.5-flash")
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

WEATHER = ToolDefinition(
    name="getCityWeather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def test_tool_result_is_sent_with_an_accepted_role():
    provider = GeminiProvider(api_key="unused", model=MODEL)
    _, contents = provider._convert_messages(
        [
            ChatMessage(role="user", content="Weather in Lagos?"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCallEvent(
                        tool_call_id="tc-1",
                        tool_name="getCityWeather",
                        arguments={"city": "Lagos"},
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                content='{"temp_c": 31}',
                tool_call_id="tc-1",
                name="getCityWeather",
            ),
        ]
    )
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[2]["parts"][0]["function_response"]["name"] == "getCityWeather"


async def _collect(provider, messages, tools):
    events = []
    async for event in provider.generate(messages, tools):
        events.append(event)
    return events


@pytest.mark.skipif(not API_KEY, reason="needs GOOGLE_API_KEY or GEMINI_API_KEY")
def test_live_tool_round_trip():
    provider = GeminiProvider(api_key=API_KEY, model=MODEL, disable_cascade=True)
    messages = [
        ChatMessage(role="system", content="Use tools when they help. Be brief."),
        ChatMessage(
            role="user",
            content="What's the weather in Lagos? Use the tool, then answer.",
        ),
    ]
    first = asyncio.run(_collect(provider, messages, [WEATHER]))
    calls = [e for e in first if isinstance(e, ToolCallEvent)]
    assert calls, first

    messages.append(ChatMessage(role="assistant", content="", tool_calls=calls))
    for call in calls:
        messages.append(
            ChatMessage(
                role="tool",
                content='{"temp_c": 31, "conditions": "light rain"}',
                tool_call_id=call.tool_call_id,
                name=call.tool_name,
            )
        )
    second = asyncio.run(_collect(provider, messages, [WEATHER]))
    errors = [e for e in second if isinstance(e, ErrorEvent)]
    assert not errors, errors
    assert "".join(e.text for e in second if isinstance(e, TextDeltaEvent)).strip()


@pytest.mark.skipif(not API_KEY, reason="needs GOOGLE_API_KEY or GEMINI_API_KEY")
def test_live_parallel_tool_round_trip():
    provider = GeminiProvider(api_key=API_KEY, model=MODEL, disable_cascade=True)
    messages = [
        ChatMessage(role="system", content="Use tools when they help. Be brief."),
        ChatMessage(
            role="user",
            content=(
                "What's the weather in Lagos and in Abuja? Call the tool for "
                "both cities at once, then answer."
            ),
        ),
    ]
    first = asyncio.run(_collect(provider, messages, [WEATHER]))
    calls = [e for e in first if isinstance(e, ToolCallEvent)]
    assert len(calls) >= 2, first

    messages.append(ChatMessage(role="assistant", content="", tool_calls=calls))
    for call in calls:
        messages.append(
            ChatMessage(
                role="tool",
                content=f'{{"city": "{call.arguments.get("city")}", "temp_c": 31}}',
                tool_call_id=call.tool_call_id,
                name=call.tool_name,
            )
        )
    second = asyncio.run(_collect(provider, messages, [WEATHER]))
    errors = [e for e in second if isinstance(e, ErrorEvent)]
    assert not errors, errors
    assert "".join(e.text for e in second if isinstance(e, TextDeltaEvent)).strip()
