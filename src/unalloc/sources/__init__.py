"""Cost source adapters."""

from __future__ import annotations

from unalloc.sources.anthropic import AnthropicSource
from unalloc.sources.base import Source
from unalloc.sources.litellm import LiteLLMSource
from unalloc.sources.openai import OpenAISource
from unalloc.sources.opencost import OpenCostSource

REGISTRY: dict[str, type[Source]] = {
    OpenCostSource.name: OpenCostSource,
    LiteLLMSource.name: LiteLLMSource,
    OpenAISource.name: OpenAISource,
    AnthropicSource.name: AnthropicSource,
}

__all__ = [
    "REGISTRY",
    "AnthropicSource",
    "LiteLLMSource",
    "OpenAISource",
    "OpenCostSource",
    "Source",
]
