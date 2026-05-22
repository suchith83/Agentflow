"""
Shared LLM calling utilities for evaluation criteria.

Provides LLMCallerMixin with _call_llm_score() used by
LLMJudgeCriterion, RubricBasedCriterion, and others.

Delegates to agentflow.core.llm.caller.call_llm so that all single-turn
provider dispatch lives in one place.
"""

from __future__ import annotations

import json
import logging

from agentflow.core.llm.caller import call_llm
from agentflow.core.llm.client_factory import detect_provider
from agentflow.qa.evaluation.token_usage import TokenUsage


logger = logging.getLogger("agentflow.evaluation")


def _parse_model_provider(model: str, use_vertex_ai: bool = False) -> tuple[str, str]:
    """Return (provider, model_name_without_prefix)."""
    provider = detect_provider(model, use_vertex_ai=use_vertex_ai)
    model_name = model.split("/", 1)[-1] if "/" in model else model
    return provider, model_name


class LLMCallerMixin:
    """Mixin providing shared LLM calling logic for LLM-based criteria.

    Reads ``self.config.judge_model`` to determine provider and model.
    """

    async def _call_google_json(self, prompt: str) -> tuple[dict, TokenUsage] | None:
        """Call Google LLM and return (parsed JSON dict, token usage), or None if unavailable."""
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        use_vertex_ai: bool = getattr(self.config, "use_vertex_ai", False)  # type: ignore[attr-defined]
        try:
            text, inp, out, cache = await call_llm(
                judge_model,
                prompt,
                temperature=0.3,
                json_mode=True,
                use_vertex_ai=use_vertex_ai,
            )
            if not text:
                return None
            data = json.loads(text)
            usage = TokenUsage(input_tokens=inp, output_tokens=out, cache_read_tokens=cache)
            return data, usage
        except (ImportError, ValueError, Exception):
            return None

    async def _call_openai_json(self, prompt: str) -> tuple[dict, TokenUsage] | None:
        """Call OpenAI LLM and return (parsed JSON dict, token usage), or None if unavailable."""
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        try:
            text, inp, out, cache = await call_llm(
                judge_model,
                prompt,
                temperature=0.3,
                json_mode=True,
            )
            if not text:
                return None
            data = json.loads(text)
            usage = TokenUsage(input_tokens=inp, output_tokens=out, cache_read_tokens=cache)
            return data, usage
        except (ImportError, ValueError, Exception):
            return None

    async def _call_llm_json(self, prompt: str) -> tuple[dict, TokenUsage]:
        """Call the judge LLM and return (parsed JSON dict, token usage)."""
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        use_vertex_ai: bool = getattr(self.config, "use_vertex_ai", False)  # type: ignore[attr-defined]
        provider, _ = _parse_model_provider(judge_model, use_vertex_ai=use_vertex_ai)

        # Try provider-specific methods first
        result = None
        if provider == "google":
            result = await self._call_google_json(prompt)
        elif provider == "openai":
            result = await self._call_openai_json(prompt)

        if result is not None:
            return result

        # If provider-specific method failed or is not available, return default
        logger.warning("No LLM provider available for model: %s", judge_model)
        return {"score": 0.5, "reasoning": "No LLM provider available"}, TokenUsage()

    async def _call_llm_score(self, prompt: str) -> tuple[float, str, TokenUsage]:
        """Call the judge LLM and return (score, reasoning, token_usage)."""
        result, usage = await self._call_llm_json(prompt)
        return float(result.get("score", 0.0)), result.get("reasoning", ""), usage
