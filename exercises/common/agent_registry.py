"""Named research variants and the compatibility mapping for old BALROG flags."""

import importlib
import os

from exercises.common.agent_io import AgentIO


VARIANTS = {
    "supplied_planning_v1": ("exercise2_planning.agents.history.supplied_planning_v1", "CustomAgent"),
    "compact_planning_pilot_v1": ("exercise2_planning.agents.history.compact_planning_pilot_v1", "CustomAgent"),
    "compact_planning_frozen_v1": ("exercise2_planning.agents.compact_planning_frozen_v1", "CompactPlanningAgent"),
    "grounded_planning_v1": ("exercise2_planning.agents.grounded_planning_v1", "GroundedPlanningAgent"),
    "plan_critique_v1": ("exercise3_reflection.agents.history.plan_critique_v1", "CustomAgent"),
    "action_critique_v2": ("exercise3_reflection.agents.action_critique_v2", "ActionCritiqueAgent"),
    "door_trigger_critique_v3": ("exercise3_reflection.agents.history.door_trigger_critique_v3", "CustomAgent"),
    "goal_state_reflection_v1": ("exercise3_reflection.agents.goal_state_reflection_v1", "GoalStateReflectionAgent"),
}


def record_historical_calls(agent_class):
    """Add raw-call logging to the two earliest agents, without changing their policy."""
    class LoggedHistoricalAgent(agent_class):
        def __init__(self, client_factory, prompt_builder):
            super().__init__(client_factory, prompt_builder)
            self.step_number = 0
            generate = self.client.generate

            def generate_and_record(messages, *args, **kwargs):
                response = generate(messages, *args, **kwargs)
                if hasattr(self, "_extract_plan_and_action"):
                    action = self._extract_plan_and_action(response.completion)[1]
                else:
                    action = self._field(response.completion, "ACTION")
                AgentIO._log_call(self, "action", messages, response, "", action, action, "")
                return response

            self.client.generate = generate_and_record

        def act(self, obs, prev_action=None):
            response = super().act(obs, prev_action)
            self.step_number += 1
            return response

        def reset(self):
            super().reset()
            self.step_number = 0

    return LoggedHistoricalAgent


def create_research_agent(client_factory, prompt_builder):
    variant = os.environ.get("BALROG_AGENT_VARIANT")
    named = variant is not None
    if not named:
        if os.environ.get("BALROG_CUSTOM_GOAL_STATE") == "1":
            variant = "goal_state_reflection_v1"
        elif os.environ.get("BALROG_CUSTOM_MODE") == "reflect":
            variant = "action_critique_v2"
        elif os.environ.get("BALROG_CUSTOM_GUIDE") == "1" or os.environ.get("BALROG_CUSTOM_GATE") == "1":
            variant = "grounded_planning_v1"
        else:
            variant = "compact_planning_frozen_v1"
    module, name = VARIANTS[variant]
    agent_class = getattr(importlib.import_module("exercises." + module), name)
    if variant in ("supplied_planning_v1", "compact_planning_pilot_v1"):
        agent_class = record_historical_calls(agent_class)
    agent = agent_class(client_factory, prompt_builder)
    if variant in ("plan_critique_v1", "door_trigger_critique_v3"):
        agent.mode = os.environ.get("BALROG_CUSTOM_MODE", "reflect")
    guide_default = str(int(getattr(agent, "correct_guide", False))) if named else "0"
    agent.correct_guide = os.environ.get("BALROG_CUSTOM_GUIDE", guide_default) == "1"
    agent.pickup_rule = os.environ.get("BALROG_CUSTOM_GATE", "0") == "1"
    return agent
