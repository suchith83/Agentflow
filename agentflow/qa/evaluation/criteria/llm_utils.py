"""
Shared LLM calling utilities for evaluation criteria.

Provides LLMCallerMixin with _call_llm_score() used by
LLMJudgeCriterion, RubricBasedCriterion, and others.

Client creation is delegated to agentflow.core.llm.client_factory so that
the eval judge uses the same provider/Vertex AI logic as the Agent class.
"""

from __future__ import annotations

import json
import logging

from agentflow.core.llm.client_factory import create_llm_client, detect_provider
from agentflow.qa.evaluation.token_usage import TokenUsage


logger = logging.getLogger("agentflow.evaluation")


class LLMCallerMixin:
    """Mixin providing shared LLM calling logic for LLM-based criteria.

    Reads ``self.config.judge_model`` to determine provider and model.
    Supports the same provider/Vertex AI path as the Agent class via the
    shared ``create_llm_client`` factory.
    """

    async def _call_llm_json(self, prompt: str) -> tuple[dict, TokenUsage]:
        """Call the judge LLM and return (parsed JSON dict, token usage).

        Args:
            prompt: The evaluation prompt to send.

        Returns:
            Tuple of (parsed JSON dict, TokenUsage for this call).
        """
        judge_model: str = self.config.judge_model  # type: ignore[attr-defined]
        use_vertex_ai: bool = getattr(self.config, "use_vertex_ai", False)  # type: ignore[attr-defined]

        provider = detect_provider(judge_model, use_vertex_ai=use_vertex_ai)
        # Strip any "provider/" prefix so the SDK gets a clean model name.
        model_name = judge_model.split("/", 1)[-1] if "/" in judge_model else judge_model

        try:
            client = create_llm_client(provider, use_vertex_ai=use_vertex_ai)
        except (ImportError, ValueError) as exc:
            logger.warning("Could not create %s client: %s", provider, exc)
            return {"score": 0.5, "reasoning": f"LLM client unavailable: {exc}"}, TokenUsage()

        if provider == "google":
            result = await self._call_google_json(client, model_name, prompt)
        else:
            result = await self._call_openai_json(client, model_name, prompt)

        if result is None:
            logger.warning("LLM call returned no result, using default score")
            return {"score": 0.5, "reasoning": "No result from LLM"}, TokenUsage()

        return result

    async def _call_google_json(
        self, client: object, model: str, prompt: str
    ) -> tuple[dict, TokenUsage] | None:
        """Call Google GenAI and return (parsed JSON dict, TokenUsage), or None on failure."""
        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
            response = await client.aio.models.generate_content(  # type: ignore[union-attr]
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
        except Exception as e:
            logger.warning("Google GenAI call failed: %s", e)
            return None

    async def _call_openai_json(
        self, client: object, model: str, prompt: str
    ) -> tuple[dict, TokenUsage] | None:
        """Call OpenAI and return (parsed JSON dict, TokenUsage), or None on failure."""
        try:
            response = await client.chat.completions.create(  # type: ignore[union-attr]
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
        except Exception as e:
            logger.warning("OpenAI call failed: %s", e)
            return None

    async def _call_llm_score(self, prompt: str) -> tuple[float, str, TokenUsage]:
        """Call the judge LLM and return (score, reasoning, token_usage).

        Convenience wrapper around :meth:`_call_llm_json` that extracts
        the ``score`` and ``reasoning`` fields.
        """
        result, usage = await self._call_llm_json(prompt)
        return float(result.get("score", 0.0)), result.get("reasoning", ""), usage
