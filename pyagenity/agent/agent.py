from typing import Any, Dict, List, Optional, Union
import litellm


# Custom input class for agent requests
class AgentRequest:
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        modalities: Optional[List[Any]] = None,
        prediction: Optional[Any] = None,
        audio: Optional[Any] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        response_format: Optional[Union[dict, Any]] = None,
        seed: Optional[int] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        parallel_tool_calls: Optional[bool] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        deployment_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,
        extra_headers: Optional[dict] = None,
        thinking: Optional[Any] = None,
        web_search_options: Optional[Any] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs,
    ):
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.stream_options = stream_options
        self.stop = stop
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.modalities = modalities
        self.prediction = prediction
        self.audio = audio
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.user = user
        self.response_format = response_format
        self.seed = seed
        self.tools = tools
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.deployment_id = deployment_id
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.api_version = api_version
        self.api_key = api_key
        self.model_list = model_list
        self.extra_headers = extra_headers
        self.thinking = thinking
        self.web_search_options = web_search_options
        self.custom_llm_provider = custom_llm_provider
        self.kwargs = kwargs


# Custom response class for agent responses
class AgentResponse:
    def __init__(
        self,
        content: str = "",
        thinking: Optional[Any] = None,
        usage: Optional[dict] = None,
        raw: Optional[Any] = None,
    ):
        self.content = content or ""
        self.thinking = thinking
        self.usage = usage
        self.raw = raw


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

    def run(
        self,
        prompt: Union[str, AgentRequest, List[Dict[str, Any]]],
        **kwargs,
    ) -> AgentResponse:
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

        # Call LiteLLM
        response = litellm.completion(**llm_args)

        # Robust extraction
        content = ""
        thinking = None
        usage = None
        raw = response
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

        return AgentResponse(
            content=content or "", thinking=thinking, usage=usage, raw=raw
        )
