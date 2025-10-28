# Callbacks: Interception and Flow Control

The callback system in 10xScale Agentflow provides powerful interception capabilities that allow you to hook into the execution flow of your agent graphs at critical decision points. Unlike simple event observation, callbacks enable you to actively participate in, modify, and control the execution process.

## Core Concepts

### CallbackManager

The `CallbackManager` is the central component that manages all callbacks and validators. You must create your own instance - there is no global default manager.

```python
from agentflow.utils.callbacks import CallbackManager

# Create your callback manager
callback_manager = CallbackManager()
```

### Callback Types

The system supports three types of callbacks:

- **Before Invoke**: Executed before AI models, tools, or MCP functions are called
- **After Invoke**: Executed after successful completion of operations
- **On Error**: Executed when operations fail

### Input Validation

Validators provide a simpler interface for message validation that runs before message processing:

- **BaseValidator**: Abstract base class for custom validators
- **PromptInjectionValidator**: Detects prompt injection attacks
- **MessageContentValidator**: Validates message structure

## Basic Usage

### 1. Creating a Callback Manager

```python
from agentflow.utils.callbacks import CallbackManager

# Create and configure your callback manager
callback_manager = CallbackManager()

# Use it when compiling your graph
compiled_graph = graph.compile(callback_manager=callback_manager)
```

### 2. Before Invoke Callbacks

Before invoke callbacks run before AI models, tools, or MCP functions are called. They can validate inputs, add metadata, or transform data.

```python
from agentflow.utils.callbacks import CallbackManager, CallbackContext, InvocationType

async def validate_tool_input(context: CallbackContext, input_data: dict) -> dict:
    """Validate and potentially modify tool inputs before execution."""
    if context.function_name == "database_query":
        # Apply security validations
        if "DROP" in input_data.get("query", "").upper():
            raise ValueError("Dangerous SQL operations not allowed")

        # Add audit context
        input_data["audit_user"] = context.metadata.get("user_id", "unknown")
        input_data["timestamp"] = datetime.utcnow().isoformat()

    return input_data

# Register the callback
callback_manager.register_before_invoke(InvocationType.TOOL, validate_tool_input)
```

### 3. After Invoke Callbacks

After invoke callbacks run after successful completion and can enrich responses or perform post-processing.

```python
async def enrich_ai_response(context: CallbackContext, input_data: dict, output_data: any) -> any:
    """Enrich AI responses with additional context."""
    if context.invocation_type == InvocationType.AI:
        # Add confidence scoring
        response_text = str(output_data)
        confidence_score = calculate_confidence(response_text)

        if confidence_score < 0.7:
            # Enhance low-confidence responses
            enhanced_response = await get_clarification_prompt(response_text, input_data)
            return enhanced_response

    return output_data

callback_manager.register_after_invoke(InvocationType.AI, enrich_ai_response)
```

### 4. Error Callbacks

Error callbacks handle failures and can provide recovery strategies or custom error responses.

```python
from agentflow.state.message import Message

async def handle_tool_errors(context: CallbackContext, input_data: dict, error: Exception) -> Message | None:
    """Implement intelligent error recovery for tool failures."""
    if context.function_name == "external_api_call":
        if isinstance(error, TimeoutError):
            # Implement retry logic
            return await retry_with_backoff(context, input_data, max_retries=3)

        elif isinstance(error, AuthenticationError):
            # Generate helpful error message
            return Message.from_text(
                "The external service authentication failed. "
                "Please check the API credentials and try again.",
                role="assistant"
            )

    return None  # Re-raise the original error

callback_manager.register_on_error(InvocationType.TOOL, handle_tool_errors)
```

## Input Validation

### Built-in Validators

10xScale Agentflow includes powerful built-in validators:

```python
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.validators import PromptInjectionValidator, MessageContentValidator

callback_manager = CallbackManager()

# Add prompt injection protection
callback_manager.register_input_validator(PromptInjectionValidator(strict_mode=True))

# Add message structure validation
callback_manager.register_input_validator(MessageContentValidator())
```

### Custom Validators

Create custom validators by extending `BaseValidator`:

```python
from agentflow.utils.callbacks import BaseValidator
from agentflow.state.message import Message
from agentflow.utils.validators import ValidationError

class BusinessPolicyValidator(BaseValidator):
    """Custom validator for business rules."""

    def __init__(self, forbidden_topics: list[str] = None, max_length: int = 1000):
        self.forbidden_topics = forbidden_topics or ["politics", "religion"]
        self.max_length = max_length

    async def validate(self, messages: list[Message]) -> bool:
        for msg in messages:
            content = msg.text()

            # Check length
            if len(content) > self.max_length:
                raise ValidationError(
                    f"Message too long: {len(content)} > {self.max_length}",
                    "length_exceeded"
                )

            # Check forbidden topics
            content_lower = content.lower()
            for topic in self.forbidden_topics:
                if topic in content_lower:
                    raise ValidationError(
                        f"Forbidden topic detected: {topic}",
                        "forbidden_topic"
                    )

        return True

# Register custom validator
callback_manager.register_input_validator(BusinessPolicyValidator())
```

### Validator Modes

Validators support strict and lenient modes:

```python
# Strict mode: Raises ValidationError on violations (blocks execution)
strict_validator = PromptInjectionValidator(strict_mode=True)

# Lenient mode: Logs violations but allows execution (for monitoring)
lenient_validator = PromptInjectionValidator(strict_mode=False)
```

## Integration with Graphs

### Basic Graph Setup

```python
from agentflow.graph import StateGraph
from agentflow.utils.callbacks import CallbackManager

# Create graph
graph = StateGraph()
# ... add nodes and edges ...

# Create callback manager with validators
callback_manager = CallbackManager()
callback_manager.register_input_validator(PromptInjectionValidator(strict_mode=True))

# Compile with callbacks
compiled_graph = graph.compile(callback_manager=callback_manager)

# Use the graph - validation runs automatically
result = compiled_graph.invoke({"messages": [user_message]})
```

### Advanced Setup with Multiple Validators

```python
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.validators import register_default_validators, PromptInjectionValidator

# Create manager
callback_manager = CallbackManager()

# Register default validators (prompt injection + message content)
register_default_validators(callback_manager, strict_mode=True)

# Add custom business validator
callback_manager.register_input_validator(
    BusinessPolicyValidator(forbidden_topics=["confidential", "internal"])
)

# Add logging callback
async def log_operations(context: CallbackContext, input_data: any) -> any:
    logger.info(f"Operation: {context.invocation_type} - {context.function_name}")
    return input_data

callback_manager.register_before_invoke(InvocationType.AI, log_operations)

# Compile graph
compiled_graph = graph.compile(callback_manager=callback_manager)
```

## Callback Context

The `CallbackContext` provides information about the current operation:

```python
@dataclass
class CallbackContext:
    invocation_type: InvocationType  # AI, TOOL, MCP, or INPUT_VALIDATION
    node_name: str                   # Name of the graph node
    function_name: str | None        # Name of the function being called
    metadata: dict[str, Any] | None  # Additional context data
```

## Error Handling

Callbacks can raise exceptions to block operations or return recovery values:

```python
# Blocking validation
async def strict_validation(context: CallbackContext, input_data: dict) -> dict:
    if not is_valid(input_data):
        raise ValueError("Invalid input data")
    return input_data

# Recovery on error
async def error_recovery(context: CallbackContext, input_data: dict, error: Exception) -> Message | None:
    if isinstance(error, TemporaryError):
        # Retry logic
        return await retry_operation(input_data)
    return None  # Re-raise error
```

## Best Practices

### 1. Create Dedicated Managers

```python
# Good: Separate managers for different use cases
production_manager = CallbackManager()
production_manager.register_input_validator(PromptInjectionValidator(strict_mode=True))

development_manager = CallbackManager()
development_manager.register_input_validator(PromptInjectionValidator(strict_mode=False))
```

### 2. Use Appropriate Validator Modes

```python
# Production: Strict mode for security
prod_validator = PromptInjectionValidator(strict_mode=True)

# Development: Lenient mode for testing
dev_validator = PromptInjectionValidator(strict_mode=False)
```

### 3. Handle Errors Gracefully

```python
async def robust_error_handler(context: CallbackContext, input_data: any, error: Exception) -> Message | None:
    try:
        # Attempt recovery
        return await recover_from_error(error, input_data)
    except Exception as recovery_error:
        # Log recovery failure but don't crash
        logger.error(f"Recovery failed: {recovery_error}")
        return None
```

### 4. Test Your Callbacks

```python
# Test validation
test_messages = [Message.text_message("Test input")]
try:
    await callback_manager.execute_validators(test_messages)
    print("Validation passed")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Complete Example

```python
from agentflow.graph import StateGraph, ToolNode
from agentflow.state import AgentState, Message
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.validators import PromptInjectionValidator, MessageContentValidator

# Create callback manager
callback_manager = CallbackManager()

# Add security validators
callback_manager.register_input_validator(PromptInjectionValidator(strict_mode=True))
callback_manager.register_input_validator(MessageContentValidator())

# Add custom business validator
class BusinessValidator(BaseValidator):
    async def validate(self, messages: list[Message]) -> bool:
        for msg in messages:
            if "inappropriate" in msg.text().lower():
                raise ValidationError("Inappropriate content detected", "content_policy")
        return True

callback_manager.register_input_validator(BusinessValidator())

# Create and configure graph
graph = StateGraph()
# ... add nodes, edges, entry point ...

# Compile with security
app = graph.compile(callback_manager=callback_manager)

# Safe execution - validation runs automatically
try:
    result = app.invoke({"messages": [Message.text_message("Safe query")]})
    print("Success:", result)
except ValidationError as e:
    print("Blocked:", e)
```

## API Reference

::: agentflow.utils.callbacks
