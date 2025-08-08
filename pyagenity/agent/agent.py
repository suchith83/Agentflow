from typing import Any, Dict, List, Optional, Union, Generator, Iterable, Callable
import litellm
import json

from pyagenity.agent.agent_request import AgentRequest
from pyagenity.agent.agent_response import AgentResponse, AgentResponseChunk


# Main Agent class
class Agent:
    def __init__(
        self,
        name: str,
        model: str,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Agent with model and parameters.
        All LiteLLM parameters are supported via kwargs.
        """
        self.name = name
        self.model = model
        self.custom_llm_provider = custom_llm_provider
        self.params = kwargs
        self._final_message_hooks: List[Callable[[AgentResponse], None]] = []

    # --------------------------- Public API --------------------------- #
    def add_final_message_hook(self, hook: Callable[[AgentResponse], None]) -> None:
        """Register a hook executed when a non-streaming run completes."""
        self._final_message_hooks.append(hook)

    def run(
        self,
        prompt: Union[str, AgentRequest, List[Dict[str, Any]]],
        **kwargs,
    ) -> Union[AgentResponse, Generator[AgentResponseChunk, None, AgentResponse]]:
        """
        Run the agent with a prompt or AgentRequest.
        Returns AgentResponse with content, thinking, usage, and raw response.
        """
        # Prepare request
        if isinstance(prompt, AgentRequest):
            req = prompt
        elif isinstance(prompt, str):
            req = AgentRequest(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                **self.params,
                **kwargs,
            )
        elif isinstance(prompt, list):
            req = AgentRequest(
                messages=prompt,
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                **self.params,
                **kwargs,
            )
        else:
            raise ValueError("Prompt must be str, list, or AgentRequest")

        # Prepare LiteLLM arguments
        llm_args = req.__dict__.copy()
        llm_args.update(req.kwargs)
        llm_args.pop("kwargs", None)
        llm_args["model"] = req.model
        llm_args["messages"] = req.messages
        # Remove None values to avoid TypeError in LiteLLM
        llm_args = {k: v for k, v in llm_args.items() if v is not None}

        # Decide streaming vs normal
        stream_requested = llm_args.get("stream", False)
        if stream_requested:
            return self._run_stream_internal(llm_args)
        response = litellm.completion(**llm_args)  # litellm object

        # Robust extraction
        content = ""
        thinking = None
        usage = None
        # Convert litellm response to raw JSON string (prevent leaking object)
        try:
            raw = response.model_dump_json()  # type: ignore[attr-defined]
        except Exception:
            try:
                raw = json.dumps(getattr(response, "__dict__", {}))
            except Exception:
                raw = None
        # Try to extract choices
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", None)
        if choices and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict):
                content = (
                    choice.get("message", {}).get("content")
                    or choice.get("text", "")
                    or ""
                )
                thinking = choice.get("message", {}).get("thinking")
            else:
                message = getattr(choice, "message", None)
                if message:
                    content = getattr(message, "content", "")
                    thinking = getattr(message, "thinking", None)
                else:
                    content = getattr(choice, "text", "")
        # Usage extraction
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage", None)
        if hasattr(usage, "__dict__"):
            usage = usage.__dict__

        finish_reason = None
        if choices and len(choices) > 0:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                finish_reason = choice0.get("finish_reason")
            else:
                finish_reason = getattr(choice0, "finish_reason", None)

        agent_response = AgentResponse(
            content=content or "",
            thinking=thinking,
            usage=usage,
            raw=raw,
            model=getattr(response, "model", None),
            provider=getattr(response, "_hidden_params", {}).get("custom_llm_provider")
            if hasattr(response, "_hidden_params")
            else None,
            finish_reason=finish_reason,
        )
        # Fire hooks
        for hook in self._final_message_hooks:
            try:
                hook(agent_response)
            except Exception:
                pass
        return agent_response

    # ------------------------- Internal Helpers ---------------------- #
    def _run_stream_internal(
        self, llm_args: Dict[str, Any]
    ) -> Generator[AgentResponseChunk, None, AgentResponse]:
        """Streaming execution returning chunks and final AgentResponse when closed."""
        llm_args["stream"] = True
        litestream: Iterable[Any] = litellm.completion(**llm_args)
        aggregated: List[str] = []
        for part in litestream:
            try:
                choices = getattr(part, "choices", None)
                if choices:
                    delta_obj = choices[0]
                    # openai style streaming delta
                    delta = ""
                    if isinstance(delta_obj, dict):
                        delta = delta_obj.get("delta", {}).get("content") or ""
                    else:
                        delta_section = getattr(delta_obj, "delta", None)
                        if delta_section:
                            delta = getattr(delta_section, "content", "") or ""
                    if delta:
                        aggregated.append(delta)
                        yield AgentResponseChunk(delta=delta, done=False)
            except Exception:
                continue
        # Build final response
        final = AgentResponse(content="".join(aggregated))
        for hook in self._final_message_hooks:
            try:
                hook(final)
            except Exception:
                pass
        yield AgentResponseChunk(delta="", done=True)
        return final
