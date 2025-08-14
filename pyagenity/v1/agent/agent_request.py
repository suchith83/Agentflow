# from typing import Any


# class AgentRequest:
#     def __init__(
#         self,
#         messages: list[dict[str, Any]],
#         model: str,
#         temperature: float | None = None,
#         top_p: float | None = None,
#         n: int | None = None,
#         stream: bool | None = None,
#         stream_options: dict | None = None,
#         stop: str | list[str] | None = None,
#         max_tokens: int | None = None,
#         max_completion_tokens: int | None = None,
#         modalities: list[Any] | None = None,
#         prediction: Any | None = None,
#         audio: Any | None = None,
#         presence_penalty: float | None = None,
#         frequency_penalty: float | None = None,
#         logit_bias: dict | None = None,
#         user: str | None = None,
#         response_format: dict | Any | None = None,
#         seed: int | None = None,
#         tools: list[Any] | None = None,
#         tool_choice: str | dict | None = None,
#         parallel_tool_calls: bool | None = None,
#         logprobs: bool | None = None,
#         top_logprobs: int | None = None,
#         deployment_id: str | None = None,
#         reasoning_effort: str | None = None,
#         base_url: str | None = None,
#         api_version: str | None = None,
#         api_key: str | None = None,
#         model_list: list | None = None,
#         extra_headers: dict | None = None,
#         thinking: Any | None = None,
#         web_search_options: Any | None = None,
#         custom_llm_provider: str | None = None,
#         **kwargs,
#     ):
#         self.messages = messages
#         self.model = model
#         self.temperature = temperature
#         self.top_p = top_p
#         self.n = n
#         self.stream = stream
#         self.stream_options = stream_options
#         self.stop = stop
#         self.max_tokens = max_tokens
#         self.max_completion_tokens = max_completion_tokens
#         self.modalities = modalities
#         self.prediction = prediction
#         self.audio = audio
#         self.presence_penalty = presence_penalty
#         self.frequency_penalty = frequency_penalty
#         self.logit_bias = logit_bias
#         self.user = user
#         self.response_format = response_format
#         self.seed = seed
#         self.tools = tools
#         self.tool_choice = tool_choice
#         self.parallel_tool_calls = parallel_tool_calls
#         self.logprobs = logprobs
#         self.top_logprobs = top_logprobs
#         self.deployment_id = deployment_id
#         self.reasoning_effort = reasoning_effort
#         self.base_url = base_url
#         self.api_version = api_version
#         self.api_key = api_key
#         self.model_list = model_list
#         self.extra_headers = extra_headers
#         self.thinking = thinking
#         self.web_search_options = web_search_options
#         self.custom_llm_provider = custom_llm_provider
#         self.kwargs = kwargs
