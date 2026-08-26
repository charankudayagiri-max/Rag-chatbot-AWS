"""
llm.py — Provider-Independent LLM Service Layer

The ONLY file that communicates with any LLM API.

CONCEPTS DEMONSTRATED
─────────────────────
  REST APIs, SDK vs REST, OpenAI Compatible APIs,
         Provider Independent Design, Request & Response Lifecycle.
  System Role, User Role, Assistant Role, Messages Array.
  Streaming Architecture, Token Streaming.

ARCHITECTURE
────────────
  1. LLMProvider — abstract contract (complete, complete_stream).
  2. GroqProvider — Groq SDK implementation.
  3. OpenAICompatibleProvider — works with any OpenAI-format endpoint.
  4. get_provider() — factory that reads LLM_PROVIDER config.
  5. Public API: get_llm_response(), get_llm_response_stream().

If you ever swap providers, only the config changes — no code edits.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Generator

from app.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    logger,
)

# Provider Abstraction

class LLMProvider(ABC):
    """
    Abstract base class defining the LLM provider contract.

    Every provider must implement two methods:
      • complete()        — return full response text.
      • complete_stream() — yield text tokens as they arrive.

    This allows the rest of the application to be completely
    decoupled from any specific LLM vendor.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> tuple[str, dict]:
        """
        Send messages and return (response_text, usage_metadata).

        Parameters
        ----------
        messages : list[dict]
            The messages array with role/content pairs.
        temperature, top_p, max_tokens : generation parameters.

        Returns
        -------
        tuple[str, dict]
            (response_text, usage_info) where usage_info contains
            prompt_tokens, completion_tokens, total_tokens.
        """
        ...

    @abstractmethod
    def complete_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        """
        Send messages and yield response text chunks.

        Yields
        ------
        str
            Text tokens/chunks from the model.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name for logging."""
        ...

# Groq Provider
class GroqProvider(LLMProvider):
    """
    LLM provider implementation using the Groq SDK.

    Groq provides ultra-fast inference for open-source models
    like Llama, Mixtral, and Gemma via their custom LPU hardware.
    """

    def __init__(self) -> None:
        from groq import Groq

        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        self._client = Groq(api_key=GROQ_API_KEY)
        logger.info("🔌 Groq provider initialised")

    @property
    def provider_name(self) -> str:
        return "groq"

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> tuple[str, dict]:
        chat_completion = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        response_text = chat_completion.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(chat_completion.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(chat_completion.usage, "completion_tokens", 0),
            "total_tokens": getattr(chat_completion.usage, "total_tokens", 0),
        }
        return response_text, usage

    def complete_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# OpenAI-Compatible Provider
class OpenAICompatibleProvider(LLMProvider):
    """
    LLM provider for any OpenAI-compatible API endpoint.

    Works with: OpenAI, Azure OpenAI, Ollama, LM Studio,
    vLLM, Together AI, Anyscale — any endpoint that speaks
    the OpenAI chat completions format.

    Configure via:
      OPENAI_API_KEY  — your API key
      OPENAI_BASE_URL — the base URL (e.g., http://localhost:11434/v1)
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=OPENAI_API_KEY or "no-key-required",
            base_url=OPENAI_BASE_URL,
        )
        logger.info(
            "🔌 OpenAI-compatible provider initialised (base_url=%s)",
            OPENAI_BASE_URL or "default",
        )

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> tuple[str, dict]:
        chat_completion = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        response_text = chat_completion.choices[0].message.content
        usage = {
            "prompt_tokens": getattr(chat_completion.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(chat_completion.usage, "completion_tokens", 0),
            "total_tokens": getattr(chat_completion.usage, "total_tokens", 0),
        }
        return response_text, usage

    def complete_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# Provider Factory
_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """
    Retrieve or initialise the LLM provider (singleton).

    Reads LLM_PROVIDER from config and instantiates the
    appropriate implementation. Supported values:
      • "groq"              — Groq SDK
      • "openai-compatible" — Any OpenAI-format API

    Returns
    -------
    LLMProvider
        The cached provider instance.
    """
    global _provider
    if _provider is None:
        if LLM_PROVIDER == "groq":
            _provider = GroqProvider()
        elif LLM_PROVIDER == "openai-compatible":
            _provider = OpenAICompatibleProvider()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
                f"Supported: groq, openai-compatible"
            )
        logger.info("✅ LLM Provider ready: %s", _provider.provider_name)
    return _provider

# Public API (backwards-compatible interface)
def get_llm_response(
    system_prompt: str,
    user_prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Send a prompt to the LLM and return the model's response text.

    Request Lifecycle:
      1. Build messages array: system → history → user
      2. Select provider
      3. Send request
      4. Parse response
      5. Log usage metrics

    Parameters
    ----------
    system_prompt : str
        The system-level instruction (RAG context or general).
    user_prompt : str
        The current user message.
    conversation_history : list[dict] | None
        Previous messages for multi-turn context.

    Returns
    -------
    str
        The LLM-generated text.
    """
    provider = get_provider()

    # ── Request Lifecycle: Build messages array ──────────────
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_prompt})

    logger.info(
        "📤 LLM Request | provider=%s | model=%s | messages=%d",
        provider.provider_name,
        MODEL_NAME,
        len(messages),
    )

    # ── Send request ─────────────────────────────────────────
    start_time = time.perf_counter()

    response_text, usage = provider.complete(
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    duration = time.perf_counter() - start_time

    # ── Response Lifecycle: Log usage ────────────────────────
    logger.info(
        "📥 LLM Response | chars=%d | tokens=%s | time=%.2fs",
        len(response_text),
        usage.get("total_tokens", "N/A"),
        duration,
    )

    return response_text

def get_llm_response_stream(
    system_prompt: str,
    user_prompt: str,
    conversation_history: list[dict[str, str]] | None = None,
):
    """
    Send a prompt to the LLM and yield response text chunks as they arrive.

    Streaming Architecture:
      • The LLM generates tokens one at a time.
      • Each token is yielded immediately to the caller.
      • The caller (API layer) forwards each token to the client via SSE.
      • This provides real-time, incremental rendering in the UI.

    Parameters
    ----------
    system_prompt : str
        The system-level instruction.
    user_prompt : str
        The current user message.
    conversation_history : list[dict] | None
        Previous messages.

    Yields
    ------
    str
        Text tokens/chunks from the model.
    """
    provider = get_provider()

    # Build messages: system → history → current user
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_prompt})

    logger.info(
        "📤 LLM Stream Request | provider=%s | model=%s | messages=%d",
        provider.provider_name,
        MODEL_NAME,
        len(messages),
    )

    yield from provider.complete_stream(
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
