"""
AI-powered user simulation for dynamic conversation testing.

This module provides the UserSimulator class which uses an LLM to
simulate realistic user behavior during agent evaluation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from agentflow.evaluation.eval_config import UserSimulatorConfig
    from agentflow.graph.compiled_graph import CompiledGraph

logger = logging.getLogger("agentflow.evaluation")


USER_SIMULATOR_PROMPT = """You are simulating a user interacting with an AI assistant.

SCENARIO:
{scenario}

CONVERSATION PLAN:
{conversation_plan}

CONVERSATION SO FAR:
{conversation_history}

Based on the scenario and plan, generate the user's next message.
The message should:
1. Follow the conversation plan naturally
2. Be realistic and natural
3. Test the agent's capabilities appropriately

Respond with only the user's message (no JSON, no explanation).
"""

USER_RESPONSE_PROMPT = """You are simulating a user responding to an AI assistant.

SCENARIO:
{scenario}

LAST ASSISTANT MESSAGE:
{assistant_message}

REMAINING CONVERSATION GOALS:
{goals}

Generate a natural user response that:
1. Reacts appropriately to the assistant's message
2. Advances toward the conversation goals
3. Feels realistic and natural

Respond with only the user's message (no JSON, no explanation).
"""


class ConversationScenario(BaseModel):
    """Defines a conversation scenario for user simulation.

    Attributes:
        scenario_id: Unique identifier for the scenario.
        description: Description of the overall scenario.
        starting_prompt: Initial user message to start the conversation.
        conversation_plan: High-level plan of conversation flow.
        goals: List of goals to achieve during conversation.
        max_turns: Maximum number of conversation turns.
    """

    scenario_id: str = ""
    description: str = ""
    starting_prompt: str = ""
    conversation_plan: str = ""
    goals: list[str] = Field(default_factory=list)
    max_turns: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Result of a user simulation session.

    Attributes:
        scenario_id: ID of the scenario that was run.
        turns: Number of conversation turns.
        conversation: Full conversation history.
        goals_achieved: List of goals that were achieved.
        completed: Whether the simulation completed successfully.
        error: Error message if simulation failed.
    """

    scenario_id: str = ""
    turns: int = 0
    conversation: list[dict[str, str]] = Field(default_factory=list)
    goals_achieved: list[str] = Field(default_factory=list)
    completed: bool = False
    error: str | None = None


class UserSimulator:
    """AI-powered user simulation for testing agents.

    Uses an LLM to generate realistic user messages for testing
    agents with dynamic conversations rather than fixed prompts.

    Attributes:
        model: The LLM model to use for user simulation.
        temperature: Temperature for user message generation.
        max_turns: Maximum conversation turns per scenario.

    Example:
        ```python
        from agentflow.evaluation.simulators import UserSimulator, ConversationScenario

        # Create simulator
        simulator = UserSimulator(model="gpt-4o-mini")

        # Define scenario
        scenario = ConversationScenario(
            scenario_id="weather_lookup",
            description="User wants to know the weather for travel planning",
            starting_prompt="I'm planning a trip next week",
            conversation_plan="1. Ask about weather\\n2. Ask about packing\\n3. Confirm plans",
            goals=["Get weather info", "Get packing advice"],
            max_turns=6,
        )

        # Run simulation
        result = await simulator.run(graph, scenario)
        print(f"Completed: {result.completed}")
        print(f"Goals achieved: {result.goals_achieved}")
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_turns: int = 10,
        config: UserSimulatorConfig | None = None,
    ):
        """Initialize the user simulator.

        Args:
            model: LLM model to use for user simulation.
            temperature: Temperature for message generation.
            max_turns: Default maximum turns per scenario.
            config: Optional configuration override.
        """
        if config:
            self.model = config.model
            self.temperature = config.temperature
            self.max_turns = config.max_invocations
        else:
            self.model = model
            self.temperature = temperature
            self.max_turns = max_turns

    async def run(
        self,
        graph: CompiledGraph,
        scenario: ConversationScenario,
        config: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Run a user simulation against an agent.

        Args:
            graph: The compiled agent graph to test.
            scenario: The conversation scenario to simulate.
            config: Optional config to pass to graph execution.

        Returns:
            SimulationResult with conversation history and outcomes.
        """
        conversation: list[dict[str, str]] = []
        goals_achieved: list[str] = []
        max_turns = scenario.max_turns or self.max_turns

        try:
            # Start with the initial prompt
            user_message = scenario.starting_prompt
            if not user_message:
                user_message = await self._generate_initial_message(scenario)

            for turn in range(max_turns):
                # Add user message to conversation
                conversation.append({"role": "user", "content": user_message})

                # Get agent response
                from agentflow.state import Message

                input_data = {"messages": [Message.text_message(user_message, role="user")]}

                try:
                    result = await graph.ainvoke(input_data, config=config or {})
                except Exception as e:
                    logger.error("Agent execution failed: %s", e)
                    return SimulationResult(
                        scenario_id=scenario.scenario_id,
                        turns=turn + 1,
                        conversation=conversation,
                        goals_achieved=goals_achieved,
                        completed=False,
                        error=f"Agent error: {e}",
                    )

                # Extract agent response
                agent_response = self._extract_response(result)
                conversation.append({"role": "assistant", "content": agent_response})

                # Check for goal completion
                achieved = await self._check_goals(
                    scenario.goals,
                    goals_achieved,
                    conversation,
                )
                goals_achieved.extend(achieved)

                # Check if we're done
                if len(goals_achieved) >= len(scenario.goals):
                    return SimulationResult(
                        scenario_id=scenario.scenario_id,
                        turns=turn + 1,
                        conversation=conversation,
                        goals_achieved=goals_achieved,
                        completed=True,
                    )

                # Generate next user message
                remaining_goals = [g for g in scenario.goals if g not in goals_achieved]
                user_message = await self._generate_response(
                    scenario=scenario,
                    conversation=conversation,
                    remaining_goals=remaining_goals,
                )

            # Max turns reached
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                turns=max_turns,
                conversation=conversation,
                goals_achieved=goals_achieved,
                completed=len(goals_achieved) >= len(scenario.goals),
            )

        except Exception as e:
            logger.error("Simulation failed: %s", e)
            return SimulationResult(
                scenario_id=scenario.scenario_id,
                turns=len(conversation) // 2,
                conversation=conversation,
                goals_achieved=goals_achieved,
                completed=False,
                error=str(e),
            )

    async def _generate_initial_message(
        self,
        scenario: ConversationScenario,
    ) -> str:
        """Generate the initial user message."""
        prompt = USER_SIMULATOR_PROMPT.format(
            scenario=scenario.description,
            conversation_plan=scenario.conversation_plan,
            conversation_history="(This is the start of the conversation)",
        )

        return await self._call_llm(prompt)

    async def _generate_response(
        self,
        scenario: ConversationScenario,
        conversation: list[dict[str, str]],
        remaining_goals: list[str],
    ) -> str:
        """Generate the next user message."""
        # Get last assistant message
        last_assistant = ""
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        prompt = USER_RESPONSE_PROMPT.format(
            scenario=scenario.description,
            assistant_message=last_assistant,
            goals="\n".join(f"- {g}" for g in remaining_goals),
        )

        return await self._call_llm(prompt)

    async def _check_goals(
        self,
        all_goals: list[str],
        achieved: list[str],
        conversation: list[dict[str, str]],
    ) -> list[str]:
        """Check which goals have been newly achieved.

        This is a simple implementation that checks if goal keywords
        appear in the conversation. Override for more sophisticated checking.
        """
        remaining = [g for g in all_goals if g not in achieved]
        newly_achieved = []

        # Simple keyword matching
        full_text = " ".join(msg["content"].lower() for msg in conversation)

        for goal in remaining:
            # Check if key words from goal appear in conversation
            goal_words = goal.lower().split()
            min_word_length = 3
            if all(word in full_text for word in goal_words if len(word) > min_word_length):
                newly_achieved.append(goal)

        return newly_achieved

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM for user simulation."""
        try:
            from litellm import acompletion

            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            return response.choices[0].message.content.strip()

        except ImportError:
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )

                return response.choices[0].message.content.strip()

            except ImportError:
                logger.warning("No LLM library available for user simulation")
                return "I have a follow-up question."

    def _extract_response(self, result: dict[str, Any]) -> str:
        """Extract text response from graph result."""
        if not result:
            return ""

        messages = result.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if hasattr(msg, "role") and msg.role == "assistant":
                    return msg.get_text() if hasattr(msg, "get_text") else str(msg)
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content

        return ""


class BatchSimulator:
    """Run multiple simulation scenarios in batch.

    Useful for comprehensive agent testing across multiple scenarios.
    """

    def __init__(
        self,
        simulator: UserSimulator | None = None,
        **kwargs,
    ):
        """Initialize batch simulator.

        Args:
            simulator: Optional pre-configured UserSimulator.
            **kwargs: Arguments to pass to UserSimulator if not provided.
        """
        self.simulator = simulator or UserSimulator(**kwargs)

    async def run_batch(
        self,
        graph: CompiledGraph,
        scenarios: list[ConversationScenario],
        config: dict[str, Any] | None = None,
    ) -> list[SimulationResult]:
        """Run multiple scenarios.

        Args:
            graph: The compiled agent graph to test.
            scenarios: List of scenarios to run.
            config: Optional config to pass to graph execution.

        Returns:
            List of simulation results.
        """
        results = []

        for scenario in scenarios:
            logger.info("Running scenario: %s", scenario.scenario_id)
            result = await self.simulator.run(graph, scenario, config)
            results.append(result)

            status = "completed" if result.completed else "incomplete"
            logger.info(
                "Scenario %s %s after %d turns (%d/%d goals)",
                scenario.scenario_id,
                status,
                result.turns,
                len(result.goals_achieved),
                len(scenario.goals),
            )

        return results

    def summary(self, results: list[SimulationResult]) -> dict[str, Any]:
        """Generate summary statistics for batch results."""
        total = len(results)
        completed = sum(1 for r in results if r.completed)
        total_goals = sum(len(r.goals_achieved) for r in results)
        total_turns = sum(r.turns for r in results)
        errors = sum(1 for r in results if r.error)

        return {
            "total_scenarios": total,
            "completed": completed,
            "completion_rate": completed / total if total > 0 else 0.0,
            "total_goals_achieved": total_goals,
            "average_turns": total_turns / total if total > 0 else 0,
            "errors": errors,
        }
