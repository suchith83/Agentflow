from typing import List, Dict, Any, Optional, Union


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
