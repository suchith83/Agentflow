New Job: Add initial state as optional in invoke. ainvoke, stream, asteam ....

Expected Behavior:
If user sent state then only that value should be updated, not the entire state ...
Lets understand with an example:
I have a custom state:
```
class MyState(AgentState):
    """Custom state with additional fields for resume matching."""
    candidate_cv: str = ""
    jd: str = ""  # job description
    match_score: float = 0.0
    analysis_results: dict = field(default_factory=dict)
```

Now say in every invoke I want to pass the latest jd to the state without affecting other fields. I should be able to do something like this:

Can you add a this features