"""Compact planning agent for Exercises 2 and 3."""

import os
import re
import json

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.mode = os.environ.get("BALROG_CUSTOM_MODE", "planning")
        self.reset()

    def reset(self):
        super().reset()
        self.plan = ""
        self.facts = []
        self.expected = ""
        self.seen_observations = []
        self.observation_history = []
        self.action_history = []
        self.lessons = []
        self.step_number = 0
        self.last_reflection_step = -6
        self.reflections_used = 0

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        observation = obs["text"]["long_term_context"]
        self.seen_observations.append(observation.lower())
        clean_observation = observation.split("\n\nObservation:\n")[-1].strip()
        self.observation_history.append(clean_observation)
        if prev_action:
            self.action_history.append(prev_action)

        reflection = None
        trigger = self._reflection_trigger()
        if self.mode == "reflect" and trigger and self.reflections_used < 3 and self.step_number - self.last_reflection_step >= 6:
            reflection = self._reflect(trigger)
            self.reflections_used += 1
            self.last_reflection_step = self.step_number
            revised_action = self._field(reflection.completion, "ACTION")
            if revised_action:
                self.step_number += 1
                return reflection._replace(
                    reasoning="REFLECTION: " + reflection.completion,
                    completion=revised_action,
                )

        babyai_hint = 'In BabyAI, a target one step forward can block movement: use "pick up" when facing an item to collect it.' if "Crafter" not in self.prompt_builder.system_prompt else ""
        lesson_line = f"Lessons from observed failures: {'; '.join(self.lessons)}\n" if self.lessons else ""

        instructions = f"""
Persistent subgoals: {self.plan or 'none'}
Previously observed facts: {'; '.join(self.facts) or 'none'}
Expected result of the previous action: {self.expected or 'none'}
Previous executed action: {prev_action or 'none'}
{lesson_line}

Compare the expected result with the CURRENT observation before choosing an action.
Keep at most two short subgoals. Change them when the observation disproves them.
Only keep facts copied exactly from an observation; old relative distances are not current positions.
{babyai_hint}
Use exactly one action from the allowed actions in the environment instructions.
Reply in four short lines, with no extra prose:
SUBGOALS: <updated subgoals, or same>
FACTS: <up to two short exact observation phrases separated by ;, or same>
EXPECTED: <visible result expected from the next action>
ACTION: <one allowed action>
""".strip()
        messages = self.prompt_builder.get_prompt()
        messages[-1].content += "\n\n" + instructions
        response = self.client.generate(messages)
        raw = response.completion
        self._log_call("action", messages, response, trigger if reflection else "")

        subgoals = self._field(raw, "SUBGOALS")
        if subgoals and subgoals.lower() != "same":
            self.plan = subgoals[:180]

        facts = self._field(raw, "FACTS")
        if facts and facts.lower() != "same":
            candidates = [part.strip() for part in facts.split(";")]
            self.facts = [
                fact[:100] for fact in candidates
                if fact and any(fact.lower() in seen for seen in self.seen_observations)
            ][:2]

        self.expected = self._field(raw, "EXPECTED")[:120]
        action = self._field(raw, "ACTION")
        # Keep the actual model output in BALROG's trajectory log for inspection.
        reasoning = (f"REFLECTION: {reflection.completion}\n" if reflection else "") + raw
        reasoning += f"\n[stored subgoals: {self.plan}; facts: {'; '.join(self.facts)}]"
        if reflection:
            response = response._replace(
                input_tokens=response.input_tokens + reflection.input_tokens,
                output_tokens=response.output_tokens + reflection.output_tokens,
            )
        self.step_number += 1
        return response._replace(reasoning=reasoning, completion=action)

    def _field(self, response, name):
        match = re.search(rf"^{name}:\s*(.*)$", response, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else ""

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
        self._log_call("reflection", messages, response, trigger)
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

    def _log_call(self, role, messages, response, trigger):
        path = os.environ.get("BALROG_LLM_LOG")
        if path:
            record = {
                "step": self.step_number,
                "role": role,
                "trigger": trigger,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "raw_response": response.completion,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
            with open(path, "a") as file:
                file.write(json.dumps(record) + "\n")
