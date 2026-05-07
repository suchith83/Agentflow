"""Skills support for Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentflow.core.graph.tool_node import ToolNode


if TYPE_CHECKING:
    from agentflow.core.skills.models import SkillConfig
    from agentflow.core.skills.registry import SkillsRegistry


logger = logging.getLogger("agentflow.agent")


class AgentSkillsMixin:
    """Skills registration helpers for Agent."""

    # Instance attributes set by _setup_skills
    _skills_config: SkillConfig | None
    _skills_registry: SkillsRegistry | None
    _trigger_table_prompt: dict[str, Any] | None
    _tool_node: ToolNode | None

    def _setup_skills(self, skills: SkillConfig | None) -> None:
        """Initialize skills infrastructure if a SkillConfig is provided.

        Args:
            skills: Optional SkillConfig instance with skills_dir and options.
        """
        self._skills_config = None
        self._skills_registry = None
        self._trigger_table_prompt = None

        if skills is None:
            return

        from agentflow.core.skills.activation import make_set_skill_tool
        from agentflow.core.skills.models import SkillConfig
        from agentflow.core.skills.registry import SkillsRegistry

        if not isinstance(skills, SkillConfig):
            raise TypeError(f"Expected SkillConfig, got {type(skills)}")

        self._skills_config = skills
        self._skills_registry = SkillsRegistry()

        if self._skills_config.skills_dir:
            self._skills_registry.discover(self._skills_config.skills_dir)

        # Create set_skill tool (loads skill content or specific resources)
        set_skill_fn = make_set_skill_tool(
            self._skills_registry,
            hot_reload=self._skills_config.hot_reload,
        )

        # Add skill tool to the tool node; if the agent was configured with a
        # named ToolNode reference (tool_node="TOOL"), queue the tool until the
        # actual ToolNode is resolved at execution time.
        if self._tool_node is None:
            if getattr(self, "tool_node_name", None) is not None:
                # Named ToolNode resolved at execution time via InjectQ; queue the
                # tool so _resolve_tools() registers it on first call.
                extra = getattr(self, "_extra_tools", None)
                if extra is None:
                    self._extra_tools = [set_skill_fn]
                else:
                    extra.append(set_skill_fn)
            else:
                raise RuntimeError(
                    "Skills require an existing ToolNode when skills are enabled. "
                    "Provide a ToolNode to the Agent before configuring skills."
                )
        else:
            self._tool_node.add_tool(set_skill_fn)

        # Build and cache trigger-table prompt once during setup.
        if self._skills_config.inject_trigger_table:
            trigger_table = self._skills_registry.build_trigger_table()
            if trigger_table:
                self._trigger_table_prompt = {"role": "system", "content": trigger_table}

        logger.info(
            "Skills enabled: %d skill(s) discovered",
            len(self._skills_registry.names()),
        )

    def _build_skill_prompts(
        self,
        state: Any,
        system_prompt: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build effective system prompts with skill trigger table if configured.

        Args:
            state: Current AgentState (unused, kept for API compatibility).
            system_prompt: Base system prompt list.

        Returns:
            Effective system prompt list with trigger table appended if configured.
        """
        effective_system_prompt = list(system_prompt)

        if not self._skills_config or not self._skills_registry:
            return effective_system_prompt

        if self._trigger_table_prompt is not None:
            effective_system_prompt.append(self._trigger_table_prompt)

        return effective_system_prompt
