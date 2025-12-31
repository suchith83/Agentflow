"""
Evaluation set and case data models.

This module defines the core data structures for agent evaluation:
    - EvalSet: Collection of evaluation cases
    - EvalCase: Single test scenario with expected outcomes
    - Invocation: Single turn in a conversation
    - ToolCall: Expected or actual tool call
    - TrajectoryStep: Single step in execution trajectory
    - SessionInput: Initial session configuration
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Type of step in execution trajectory."""

    NODE = "node"
    TOOL = "tool"
    MESSAGE = "message"
    CONDITIONAL = "conditional"


class ToolCall(BaseModel):
    """Represents a tool call (expected or actual).

    Attributes:
        name: Name of the tool/function called.
        args: Arguments passed to the tool.
        call_id: Unique identifier for this tool call.
        result: Optional result from the tool execution.
    """

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None
    result: Any | None = None

    def matches(
        self,
        other: ToolCall,
        check_args: bool = True,
        check_call_id: bool = False,
    ) -> bool:
        """Check if this tool call matches another.

        Args:
            other: The tool call to compare against.
            check_args: Whether to compare arguments.
            check_call_id: Whether to compare call IDs.

        Returns:
            True if the tool calls match based on the specified criteria.
        """
        if self.name != other.name:
            return False
        if check_args and self.args != other.args:
            return False
        if check_call_id and self.call_id != other.call_id:
            return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return False
        return self.name == other.name and self.args == other.args


class TrajectoryStep(BaseModel):
    """Single step in an execution trajectory.

    Attributes:
        step_type: Type of step (node, tool, message, conditional).
        name: Name of the node, tool, or action.
        args: Arguments if this is a tool call.
        timestamp: When this step occurred.
        metadata: Additional step-specific data.
    """

    step_type: StepType
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def node(
        cls,
        name: str,
        timestamp: float | None = None,
        **metadata: Any,
    ) -> TrajectoryStep:
        """Create a node step."""
        return cls(
            step_type=StepType.NODE,
            name=name,
            timestamp=timestamp,
            metadata=metadata,
        )

    @classmethod
    def tool(
        cls,
        name: str,
        args: dict[str, Any] | None = None,
        timestamp: float | None = None,
        **metadata: Any,
    ) -> TrajectoryStep:
        """Create a tool step."""
        return cls(
            step_type=StepType.TOOL,
            name=name,
            args=args or {},
            timestamp=timestamp,
            metadata=metadata,
        )


class MessageContent(BaseModel):
    """Content of a message (simplified from full Message model).

    Attributes:
        role: The role of the message sender (user, assistant, tool).
        content: The message content (text or structured blocks).
        metadata: Additional message metadata.
    """

    role: str
    content: str | list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def user(cls, text: str) -> MessageContent:
        """Create a user message."""
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> MessageContent:
        """Create an assistant message."""
        return cls(role="assistant", content=text)

    def get_text(self) -> str:
        """Extract text content from the message."""
        if isinstance(self.content, str):
            return self.content
        # Handle list of content blocks
        texts = []
        for block in self.content:
            if isinstance(block, dict):
                if "text" in block:
                    texts.append(block["text"])
                elif "type" in block and block["type"] == "text":
                    texts.append(block.get("text", ""))
        return " ".join(texts)


class Invocation(BaseModel):
    """A single turn in the conversation.

    Represents one user query and the expected agent behavior in response.

    Attributes:
        invocation_id: Unique identifier for this invocation.
        user_content: The user's input message.
        expected_tool_trajectory: Expected sequence of tool calls.
        expected_intermediate_responses: Expected intermediate agent responses.
        expected_final_response: Expected final response from the agent.
    """

    invocation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_content: MessageContent
    expected_tool_trajectory: list[ToolCall] = Field(default_factory=list)
    expected_intermediate_responses: list[MessageContent] = Field(default_factory=list)
    expected_final_response: MessageContent | None = None

    @classmethod
    def simple(
        cls,
        user_query: str,
        expected_response: str | None = None,
        expected_tools: list[ToolCall] | None = None,
    ) -> Invocation:
        """Create a simple invocation with minimal configuration.

        Args:
            user_query: The user's question or request.
            expected_response: Expected agent response text.
            expected_tools: Expected tool calls.

        Returns:
            Configured Invocation instance.
        """
        return cls(
            user_content=MessageContent.user(user_query),
            expected_final_response=(
                MessageContent.assistant(expected_response) if expected_response else None
            ),
            expected_tool_trajectory=expected_tools or [],
        )


class SessionInput(BaseModel):
    """Initial session configuration for evaluation.

    Attributes:
        app_name: Name of the agent/application being tested.
        user_id: User identifier for the session.
        state: Initial state values.
        config: Additional configuration for the session.
    """

    app_name: str = ""
    user_id: str = "test_user"
    state: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    """A single evaluation case representing one test scenario.

    An EvalCase contains a conversation (one or more invocations) and
    the expected outcomes for each turn.

    Attributes:
        eval_id: Unique identifier for this evaluation case.
        name: Human-readable name for this test case.
        description: Description of what this test case validates.
        conversation: List of invocations (turns) in the conversation.
        session_input: Initial session configuration.
        tags: Tags for categorizing/filtering test cases.
        metadata: Additional test case metadata.
    """

    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    conversation: list[Invocation] = Field(default_factory=list)
    session_input: SessionInput = Field(default_factory=SessionInput)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def single_turn(
        cls,
        eval_id: str,
        user_query: str,
        expected_response: str | None = None,
        expected_tools: list[ToolCall] | None = None,
        name: str = "",
        description: str = "",
    ) -> EvalCase:
        """Create a single-turn evaluation case.

        Args:
            eval_id: Unique identifier for this case.
            user_query: The user's input.
            expected_response: Expected agent response.
            expected_tools: Expected tool calls.
            name: Human-readable name.
            description: Description of the test.

        Returns:
            Configured EvalCase instance.
        """
        return cls(
            eval_id=eval_id,
            name=name,
            description=description,
            conversation=[
                Invocation.simple(
                    user_query=user_query,
                    expected_response=expected_response,
                    expected_tools=expected_tools,
                )
            ],
        )


class EvalSet(BaseModel):
    """A collection of evaluation cases.

    An EvalSet groups related test cases together for batch evaluation.

    Attributes:
        eval_set_id: Unique identifier for this evaluation set.
        name: Human-readable name for this set.
        description: Description of what this evaluation set tests.
        eval_cases: List of evaluation cases in this set.
        metadata: Additional metadata for the set.
    """

    eval_set_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    eval_cases: list[EvalCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of evaluation cases."""
        return len(self.eval_cases)

    def __iter__(self):
        """Iterate over evaluation cases."""
        return iter(self.eval_cases)

    def add_case(self, case: EvalCase) -> None:
        """Add an evaluation case to the set."""
        self.eval_cases.append(case)

    def get_case(self, eval_id: str) -> EvalCase | None:
        """Get an evaluation case by ID."""
        for case in self.eval_cases:
            if case.eval_id == eval_id:
                return case
        return None

    def filter_by_tags(self, tags: list[str]) -> list[EvalCase]:
        """Filter cases by tags (cases must have all specified tags)."""
        return [case for case in self.eval_cases if all(tag in case.tags for tag in tags)]

    @classmethod
    def from_file(cls, path: str) -> EvalSet:
        """Load an EvalSet from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded EvalSet instance.
        """
        import json
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: str) -> None:
        """Save the EvalSet to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        import json
        from pathlib import Path

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
