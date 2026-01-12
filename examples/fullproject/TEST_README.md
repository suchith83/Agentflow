# Testing Guide for react_sync.py

This document explains the comprehensive test suite for the `react_sync.py` ReAct agent example.

## Overview

The test suite consists of two main files:

1. **`test_react_sync.py`** - Unit tests for individual components
2. **`test_react_sync_evaluation.py`** - Evaluation tests for end-to-end behavior

## Test Structure

### Unit Tests (`test_react_sync.py`)

Unit tests verify individual components and functions work correctly in isolation:

- **TestGetWeatherTool**: Tests the `get_weather` tool function
  - Basic functionality
  - Dependency injection (tool_call_id, state)
  - Various location inputs
  
- **TestShouldUseToolsFunction**: Tests the routing logic
  - Empty context handling
  - Assistant messages with/without tool calls
  - Tool result messages
  - Complex conversation flows
  
- **TestToolNode**: Tests ToolNode configuration and execution
  
- **TestAgentConfiguration**: Tests Agent initialization and configuration
  
- **TestGraphConstruction**: Tests StateGraph structure
  - Node configuration
  - Edge connections
  - Conditional routing
  - Entry points
  
- **TestCheckpointerConfiguration**: Tests checkpointer setup
  
- **TestIntegration**: Integration tests for complete setup
  
- **TestMessageFlow**: Tests message formatting and flow
  
- **TestErrorHandling**: Tests error scenarios and edge cases

### Evaluation Tests (`test_react_sync_evaluation.py`)

Evaluation tests assess the agent's end-to-end behavior and performance:

- **TestTrajectoryEvaluation**: Tests trajectory collection and analysis
  - Message capture
  - Tool call tracking
  
- **TestWeatherToolEvaluation**: Evaluates tool functionality
  - Output format correctness
  - Special character handling
  - Dependency injection
  
- **TestAgentResponseQuality**: Evaluates agent decision-making
  - Appropriate tool usage
  - Routing logic correctness
  
- **TestPerformanceMetrics**: Measures performance
  - Tool execution speed
  - Routing decision speed
  
- **TestEndToEndScenarios**: Complete execution scenarios
  - Single tool call flows
  - Conversation integrity
  
- **TestRobustnessAndEdgeCases**: Tests edge cases
  - Empty inputs
  - Very long inputs
  - Many messages in context
  
- **TestEvaluationCriteria**: Custom evaluation criteria
  - Tool call accuracy
  
- **TestConfigurationValidation**: Validates configuration
  - Model settings
  - System prompts
  - Tool node names
  
- **TestComparisonWithExpectedBehavior**: Compares actual vs expected
  - Message sequence patterns
  
- **TestDocumentationAndExamples**: Validates documentation
  - Code comments
  - Docstrings
  
- **TestReproducibility**: Ensures consistent behavior
  - Deterministic outputs
  - Reproducible routing

## Running the Tests

### Prerequisites

Make sure you have the required dependencies installed:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Install agentflow with required extras
pip install -e ".[google-genai]"
```

### Running All Tests

Run all tests in both files:

```bash
# From the react directory
pytest test_react_sync.py test_react_sync_evaluation.py -v

# Or with more detailed output
pytest test_react_sync.py test_react_sync_evaluation.py -v --tb=short
```

### Running Specific Test Files

Run only unit tests:

```bash
pytest test_react_sync.py -v
```

Run only evaluation tests:

```bash
pytest test_react_sync_evaluation.py -v
```

### Running Specific Test Classes

Run a specific test class:

```bash
pytest test_react_sync.py::TestGetWeatherTool -v
pytest test_react_sync_evaluation.py::TestPerformanceMetrics -v
```

### Running Specific Test Methods

Run a specific test:

```bash
pytest test_react_sync.py::TestGetWeatherTool::test_get_weather_basic_call -v
```

### Running with Coverage

Generate a coverage report:

```bash
pip install pytest-cov
pytest test_react_sync.py test_react_sync_evaluation.py --cov=react_sync --cov-report=html
```

View the coverage report:

```bash
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

### Integration Tests

Some tests require actual API calls and are skipped by default. To run them:

```bash
# Set up environment variable with your API key
export GOOGLE_API_KEY="your-api-key-here"

# Run with integration tests enabled
pytest test_react_sync_evaluation.py --run-integration -v
```

## Test Output

### Successful Test Output

```
test_react_sync.py::TestGetWeatherTool::test_get_weather_basic_call PASSED
test_react_sync.py::TestGetWeatherTool::test_get_weather_with_tool_call_id PASSED
...
============================== 45 passed in 2.34s ==============================
```

### Failed Test Output

When a test fails, you'll see detailed information:

```
FAILED test_react_sync.py::TestGetWeatherTool::test_get_weather_basic_call
_________________________ TestGetWeatherTool.test_get_weather_basic_call _________________________

    def test_get_weather_basic_call(self):
        result = get_weather(location="New York City")
>       assert result == "The weather in New York City is sunny"
E       AssertionError: assert 'The weather...' == 'The weather in New York City is sunny'
```

## Best Practices

### Writing New Tests

When adding new tests:

1. **Follow the naming convention**: `test_<what_is_being_tested>`
2. **Use descriptive docstrings**: Explain what the test validates
3. **Keep tests focused**: Each test should verify one specific behavior
4. **Use appropriate fixtures**: Set up common test data in `setup_method` or fixtures
5. **Clean up after tests**: Remove any created files or state

### Test Organization

- **Unit tests**: Test individual functions and components in isolation
- **Integration tests**: Test how components work together
- **Evaluation tests**: Test end-to-end behavior and quality metrics

### Mocking

For tests that would require API calls:

```python
from unittest.mock import patch, MagicMock

def test_with_mock():
    with patch('agentflow.graph.agent.Agent.__call__') as mock_agent:
        mock_agent.return_value = AgentState()
        # Your test code here
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -e ".[google-genai]"
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: |
          pytest examples/react/test_react_sync.py -v
          pytest examples/react/test_react_sync_evaluation.py -v
```

## Evaluation Metrics

The evaluation tests measure:

- **Correctness**: Does the agent produce correct outputs?
- **Completeness**: Does the agent follow the expected execution path?
- **Performance**: How fast does the agent execute?
- **Robustness**: Does the agent handle edge cases?
- **Reproducibility**: Does the agent produce consistent results?

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running tests from the correct directory
   ```bash
   cd /path/to/pyagenity/examples/react
   pytest test_react_sync.py -v
   ```

2. **Missing dependencies**: Install all required packages
   ```bash
   pip install -e ".[google-genai]"
   pip install pytest pytest-asyncio
   ```

3. **API key errors**: Set up your API key for integration tests
   ```bash
   export GOOGLE_API_KEY="your-key"
   ```

4. **Module not found errors**: The tests add the react directory to the path automatically, but if you have issues, run from the react directory

## Contributing

When contributing new tests:

1. Add unit tests for new functions or components
2. Add evaluation tests for new behaviors or features
3. Ensure all tests pass before submitting
4. Update this README with any new test categories

## Example Test Run

Here's what a complete test run looks like:

```bash
$ pytest test_react_sync.py test_react_sync_evaluation.py -v

========================= test session starts =========================
collected 67 items

test_react_sync.py::TestGetWeatherTool::test_get_weather_basic_call PASSED
test_react_sync.py::TestGetWeatherTool::test_get_weather_with_tool_call_id PASSED
test_react_sync.py::TestGetWeatherTool::test_get_weather_with_state PASSED
test_react_sync.py::TestGetWeatherTool::test_get_weather_various_locations PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_empty_context PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_no_context_attribute PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_assistant_with_tool_calls PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_assistant_without_tool_calls PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_tool_result_message PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_user_message PASSED
test_react_sync.py::TestShouldUseToolsFunction::test_complex_conversation_flow PASSED
...

test_react_sync_evaluation.py::TestTrajectoryEvaluation::test_collector_captures_messages PASSED
test_react_sync_evaluation.py::TestTrajectoryEvaluation::test_collector_captures_tool_calls PASSED
test_react_sync_evaluation.py::TestWeatherToolEvaluation::test_weather_tool_correct_output_format PASSED
...

========================= 67 passed in 3.45s =========================
```

## License

These tests are part of the 10xScale Agentflow project and follow the same license.

## Support

For issues or questions about the tests, please:
1. Check this README
2. Review the test code comments
3. Open an issue on the GitHub repository
