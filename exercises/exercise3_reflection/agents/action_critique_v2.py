"""Accepted E3 v2: bounded critique that chooses the next action directly."""

import re
from exercises.exercise2_planning.agents.grounded_planning_v1 import GroundedPlanningAgent


class ActionCritiqueAgent(GroundedPlanningAgent):
    correct_guide = False

    def _maybe_reflect(self):
        trigger = self._reflection_trigger()
        if trigger and self.reflections_used < 3 and self.step_number - self.last_reflection_step >= 6:
            reflection = self._reflect(trigger)
            self.reflections_used += 1
            self.last_reflection_step = self.step_number
            return reflection, trigger
        return None, ""

    def _reflection_trigger(self):
        # A repeated description alone may just be a corridor. Require a front obstacle.
        if self.action_history and len(self.observation_history) >= 2:
            last_action = self.action_history[-1]
            before, after = self.observation_history[-2:]
            obstacle = re.search(r"a (?:wall|locked .* door|.* key|.* box|.* ball) 1 step forward", before)
            if before == after and obstacle and last_action in ("go forward", "toggle", "pick up"):
                if len(self.action_history) >= 2 and self.action_history[-2] == last_action:
                    return f"Two {last_action} actions left the same front obstacle visible."

        # An alternating pair of observations after undoing turns is a concrete loop.
        if len(self.observation_history) >= 5 and len(self.action_history) >= 4:
            views = self.observation_history[-5:]
            turns = self.action_history[-4:]
            if (views[0] == views[2] == views[4] and views[1] == views[3]
                    and views[0] != views[1]
                    and turns in (["turn left", "turn right"] * 2, ["turn right", "turn left"] * 2)):
                return "Left/right turns returned to the same two observations twice."
        return ""


    def _reflect(self, trigger):
        messages = self.prompt_builder.get_prompt()
        messages[-1].content += f"""

Critique this concrete failure: {trigger}
Last actions: {', '.join(self.action_history[-5:])}
Current subgoals: {self.plan or 'none'}
Use the observed evidence only. Explain why the last plan or action failed.
Choose one allowed next action that tests a different route; do not repeat the loop's last action.
Keep the answer short. A wall more than one step forward does not block a single forward step.
CRITIQUE: <one sentence>
REVISED SUBGOALS: <up to two short subgoals>
LESSON: <one short instruction for the remainder of this episode>
EXPECTED: <one visible result expected from your action>
ACTION: <one allowed action>
"""
        response = self.client.generate(messages)
        revised_action = self._field(response.completion, "ACTION")
        self._log_call("reflection", messages, response, trigger, revised_action, revised_action, "")
        revision = self._field(response.completion, "REVISED SUBGOALS")
        lesson = self._field(response.completion, "LESSON")
        expected = self._field(response.completion, "EXPECTED")
        if revision:
            self.plan = revision[:180]
        if lesson:
            self.lessons.append(lesson[:120])
            self.lessons = self.lessons[-2:]
        if expected:
            self.expected = expected[:120]
        return response

