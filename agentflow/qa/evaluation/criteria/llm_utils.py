"""
Shared LLM calling utilities for criteria.

Provides LLMCallerMixin with _call_llm_score() used by
LLMJudgeCriterion, RubricBasedCriterion, and others.
"""

from __future__ import annotations

import json
import logging

from agentflow.qa.evaluation.token_usage import TokenUsage


logger = logging.getLogger("agentflow.evaluation")


def _parse_model_provider(model: str) -> tuple[str, str]:
    """Derive (provider, clean_model) from a model string.

    Supports:
      - ``"gemini-2.5-flash"``          → ``("google", "gemini-2.5-flash")``
            - ``"gemini/gemini-2.5-flash"``    → ``("google", "gemini-2.5-flash")``
      - ``"gpt-4o"``                     → ``("openai", "gpt-4o")``
      - ``"openai/gpt-4o"``              → ``("openai", "gpt-4o")``

    Returns:
        Tuple of (provider, model_name).
    """
    if "/" in model:
        prefix, name = model.split("/", 1)
        prefix_lower = prefix.lower()
        if prefix_lower in ("gemini", "google"):
            return "google", name
        if prefix_lower in ("openai", "gpt"):
            return "openai", name
        # Unknown prefix — treat the whole string as model name, guess provider
        model = name

    lower = model.lower()
    if lower.startswith("gemini"):
        return "google", model
    if lower.startswith(("gpt", "o1", "o3", "o4")):
        return "openai", model
    # Default to google
    return "google", model


class LLMCallerMixin:
    """Mixin providing shared LLM calling logic for criteria.

    Uses Google GenAI SDK as the primary path, then OpenAI as fallback.
    All LLM-based criteria inherit from this.
    """

    async def _call_llm_json(self, prompt: str) -> tuple[dict, TokenUsage]:
        """Call LLM and return (parsed JSON dict, token usage).

        Tries Google GenAI first, then OpenAI. If none are available,
        returns a default dict with score 0.5 and zero token usage.

        Args:
            prompt: The evaluation prompt to send.

        Returns:
            Tuple of (parsed JSON dict, TokenUsage for this call).
        """
        provider, model_name = _parse_model_provider(self.config.judge_model)

        if provider == "google":
            result = await self._call_google_json(model_name, prompt)
            if result is not None:
                return result

        # OpenAI path (primary for OpenAI models, fallback for Google failures)
        result = await self._call_openai_json(
            self.config.judge_model if provider == "openai" else model_name,
            prompt,
        )
        if result is not None:
            return result

        # Last resort: try Google if we haven't yet
        if provider != "google":
            result = await self._call_google_json(model_name, prompt)
            if result is not None:
                return result

        logger.warning("No LLM library available, returning default score")
        return {"score": 0.5, "reasoning": "No LLM available"}, TokenUsage()

    async def _call_google_json(self, model: str, prompt: str) -> tuple[dict, TokenUsage] | None:
        """Call Google GenAI and return (parsed JSON dict, TokenUsage), or None on failure."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client()
            config = types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Google GenAI returned empty content")
            usage = TokenUsage()
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                usage = TokenUsage(
                    input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                    cache_read_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
                )
            return json.loads(text), usage
        except ImportError:
            return None
        except Exception as e:
            logger.warning("Google GenAI call failed: %s", e)
            return None

    async def _call_openai_json(self, model: str, prompt: str) -> tuple[dict, TokenUsage] | None:
        """Call OpenAI and return (parsed JSON dict, TokenUsage), or None on failure."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("OpenAI returned empty content")
            usage = TokenUsage()
            ru = getattr(response, "usage", None)
            if ru is not None:
                cached = 0
                details = getattr(ru, "prompt_tokens_details", None)
                if details is not None:
                    cached = getattr(details, "cached_tokens", 0) or 0
                usage = TokenUsage(
                    input_tokens=getattr(ru, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(ru, "completion_tokens", 0) or 0,
                    cache_read_tokens=cached,
                )
            return json.loads(text), usage
        except ImportError:
            return None
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)
            return None

    async def _call_llm_score(self, prompt: str) -> tuple[float, str, TokenUsage]:
        """Call LLM and return (score, reasoning, token_usage).

        Convenience wrapper around :meth:`_call_llm_json` that extracts
        the ``score`` and ``reasoning`` fields from the response dict.

        Args:
            prompt: The evaluation prompt to send.

        Returns:
            Tuple of (score float 0-1, reasoning string, TokenUsage).
        """
        result, usage = await self._call_llm_json(prompt)
        return float(result.get("score", 0.0)), result.get("reasoning", ""), usage
