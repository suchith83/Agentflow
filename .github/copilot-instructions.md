# Copilot Coding Agent Instructions for PyAgenity

## Project Overview
PyAgenity is a Python-based agent framework leveraging LLMs (e.g., OpenAI, Groq, Anthropic) via the `litellm` library. The main logic resides in the `pyagenity/agent/agent.py` file, which defines the `Agent` class for running LLM completions. Example usage and extension points are found in the `example/` directory.

## Architecture & Key Components
- **Agent Class** (`pyagenity/agent/agent.py`): Core abstraction for LLM-powered agents. Accepts model, temperature, and max_tokens. Uses `litellm.completion` for chat completions.
- **LLM Integration**: All LLM calls are routed through `litellm`, supporting multiple providers. Model selection and parameters are passed directly to `completion()`.
- **Example Directory**: Contains sample agent usage and is the best starting point for new features or tests.

## Developer Workflows
- **Build/Install**: Use standard Python packaging (`pyproject.toml`). Install dependencies with `pip install -e .` or `pip install -r requirements.txt` if present.
- **Testing**: No explicit test runner or test files detected. Add tests in `example/` or a new `tests/` directory. Use `pytest` for convention.
- **Debugging**: Directly run scripts in `example/` or interact with `Agent` via REPL. For LLM debugging, inspect `litellm` responses and exceptions.

## Project-Specific Patterns
- **Agent Pattern**: All agent logic should subclass or wrap the `Agent` class. Pass prompts as strings; responses are raw LLM outputs.
- **Parameterization**: Prefer passing model parameters (e.g., temperature, max_tokens) via the `Agent` constructor for reproducibility.
- **Extensibility**: Add new agent types in `pyagenity/agent/`. Keep integrations modular and avoid hardcoding provider logic.

## Integration Points
- **litellm**: Central dependency for LLM calls. Configure provider/model via environment variables or directly in code.
- **External Models**: Supported via `litellm` (OpenAI, Groq, Anthropic, Azure, etc.).

## Conventions & Recommendations
- **Directory Structure**: Core logic in `pyagenity/agent/`, examples in `example/`. Follow this pattern for new features.
- **No Custom Build/Test Scripts**: Use standard Python tools unless otherwise documented.
- **Configuration**: Prefer code-based configuration over environment variables for agent parameters.

## Example Usage
```python
from pyagenity.agent.agent import Agent
agent = Agent(name="Demo", model="gpt-4o")
response = agent.run("Hello!")
```

## Key Files
- `pyagenity/agent/agent.py`: Main agent logic
- `example/`: Usage samples
- `pyproject.toml`: Project metadata

---
If any conventions or workflows are unclear, please request clarification or provide feedback for further refinement.
