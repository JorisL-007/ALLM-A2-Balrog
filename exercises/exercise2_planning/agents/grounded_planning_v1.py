"""E2 follow-up: corrected front-cell guide and optional target pickup rule."""

import re
from .compact_planning_frozen_v1 import CompactPlanningAgent


class GroundedPlanningAgent(CompactPlanningAgent):
    correct_guide = True

    def _prepare_observation(self, obs):
        if self.correct_guide and obs.get("mission"):
            self.prompt_builder.system_prompt = self.prompt_builder.system_prompt.replace(
                "pick up: pick up the object below you",
                "pick up: pick up a portable object in the cell immediately in front of you",
            ).replace(
                "Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.",
                "Use 'pick up' for a portable object immediately in front of you. Use 'toggle' to open or close a door or interact with a switch in front of you.",
            )

    def _apply_pickup_rule(self, obs, observation, model_action):
        if not self.pickup_rule or model_action not in ("go forward", "toggle"):
            return model_action, ""
        mission = obs.get("mission", "")
        target = re.match(r"^pick up the ([a-z]+ [a-z]+)(?:$|[,.])", mission, re.IGNORECASE)
        if not target:
            return model_action, ""
        item = target.group(1).lower()
        lines = [line.strip().lower() for line in observation.splitlines()]
        if f"a {item} 1 step forward" not in lines:
            return model_action, ""
        if any(re.search(r"(?:you carry|you are carrying|holding).*\b" + re.escape(item) + r"\b", line)
               for line in lines):
            return model_action, ""
        return "pick up", f"mission target {item} observed 1 step forward"

