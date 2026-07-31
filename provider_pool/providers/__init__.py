"""Provider factory — resolves provider name to LLMProvider instance."""

from __future__ import annotations

from ..base import LLMProvider


def get_provider(provider_name: str, model: str, **kwargs) -> LLMProvider:
    """Instantiate the correct provider.

    Args:
        provider_name: "google" | "openai" | "anthropic"
        model: Model name (e.g. "gemini-2.5-flash", "gpt-4o-mini")
        **kwargs: Provider-specific kwargs (api_key, base_url, disable_cascade, ...)
    """
    if provider_name == "google":
        from .google import GeminiProvider

        return GeminiProvider(model=model, **kwargs)
    elif provider_name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(model=model, **kwargs)
    elif provider_name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. Supported: google, openai, anthropic"
        )


async def list_models(provider_name: str, api_key: str, **kwargs) -> list[str]:
    """List the model IDs an API key can access for the given provider.

    Returns model IDs suitable for use directly in SlotConfig.models — but
    the caller still chooses which ones and in what order; this doesn't rank
    or filter for quality/cost, only for basic chat-capability where the
    provider's API makes that determinable (see each provider's list_models
    docstring for what filtering, if any, is applied).

    Args:
        provider_name: "google" | "openai" | "anthropic"
        api_key: the provider API key to query with
        **kwargs: provider-specific extras (e.g. base_url for openai)
    """
    if provider_name == "google":
        from .google import list_models as _list_models
    elif provider_name == "openai":
        from .openai import list_models as _list_models
    elif provider_name == "anthropic":
        from .anthropic import list_models as _list_models
    else:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. Supported: google, openai, anthropic"
        )
    return await _list_models(api_key, **kwargs)
