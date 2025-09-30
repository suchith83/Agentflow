Config is the object that holds configuration for a graph execution,
its has following fields:
- is_stream: bool = False
    - Whether the graph execution is a stream or not.
- user_id: str
    - The user id for whom the graph is being executed.
- run_id: str
    - The unique id for this graph execution.
- timestamp: str
    - The timestamp when the graph execution started.

- thread_id: Optional[str] = None
    - The thread id for this graph execution. If not provided, a new one will be
      generated.

- recursion_limit: int = 25
    - The maximum recursion limit for the graph execution.


If you deployed this using pyagenity cli then `user` as dict will be passed WHICH Is returned from authencation system ...