# SUPERSEDED: initial compact-planning pilot, before the frozen evaluation.
# Original bytes are preserved in archive/refactor_20261018/source_before.tar.gz.
"""Compact planning agent for Exercises 2 and 3."""

import os
import re

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

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        observation = obs["text"]["long_term_context"]
        self.seen_observations.append(observation.lower())

        instructions = f"""
Persistent subgoals: {self.plan or 'none'}
Previously observed facts: {'; '.join(self.facts) or 'none'}
Expected result of the previous action: {self.expected or 'none'}
Previous executed action: {prev_action or 'none'}

Compare the expected result with the CURRENT observation before choosing an action.
Keep at most two short subgoals. Change them when the observation disproves them.
Only keep facts copied exactly from an observation; old relative distances are not current positions.
In BabyAI, a target one step forward can block movement: use "pick up" when facing an item to collect it.
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
        reasoning = raw + f"\n[stored subgoals: {self.plan}; facts: {'; '.join(self.facts)}]"
        return response._replace(reasoning=reasoning, completion=action)

    def _field(self, response, name):
        match = re.search(rf"^{name}:\s*(.*)$", response, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else ""
