"""Accepted E2 planner: two subgoals, observed facts, and expected outcomes.

Extracted without prompt changes from the frozen follow-up agent.
"""

from balrog.agents.base import BaseAgent
from exercises.common.agent_io import AgentIO


class CompactPlanningAgent(AgentIO, BaseAgent):
    state_line = ""
    line_count = "four"
    correct_guide = False
    pickup_rule = False

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
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
        self._prepare_observation(obs)
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        observation = obs["text"]["long_term_context"]
        self.seen_observations.append(observation.lower())
        clean_observation = observation.split("\n\nObservation:\n")[-1].strip()
        self.observation_history.append(clean_observation)
        if prev_action:
            self.action_history.append(prev_action)

        reflection, trigger = self._maybe_reflect()
        if reflection is not None:
            revised_action = self._field(reflection.completion, "ACTION")
            if revised_action:
                self.step_number += 1
                result = reflection._replace(
                    reasoning="REFLECTION: " + reflection.completion,
                    completion=revised_action,
                )
                return self._with_action_metadata(result, revised_action, "")

        babyai_hint = 'In BabyAI, a target one step forward can block movement: use "pick up" when facing an item to collect it.' if "Crafter" not in self.prompt_builder.system_prompt else ""
        lesson_line = f"Lessons from observed failures: {'; '.join(self.lessons)}\n" if self.lessons else ""

        state_line = self.state_line
        line_count = self.line_count
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
Reply in {line_count} short lines, with no extra prose:
SUBGOALS: <updated subgoals, or same>
FACTS: <up to two short exact observation phrases separated by ;, or same>
EXPECTED: <visible result expected from the next action>
{state_line}ACTION: <one allowed action>
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
        model_action = self._field(raw, "ACTION")
        action, override_reason = self._apply_pickup_rule(obs, clean_observation, model_action)
        self._log_call("action", messages, response, trigger if reflection else "",
                       model_action, action, override_reason)
        # Keep the actual model output in BALROG's trajectory log for inspection.
        reasoning = (f"REFLECTION: {reflection.completion}\n" if reflection else "") + raw
        reasoning += f"\n[stored subgoals: {self.plan}; facts: {'; '.join(self.facts)}]"
        if reflection:
            response = response._replace(
                input_tokens=response.input_tokens + reflection.input_tokens,
                output_tokens=response.output_tokens + reflection.output_tokens,
            )
        self.step_number += 1
        result = response._replace(reasoning=reasoning, completion=action)
        return self._with_action_metadata(result, model_action, override_reason)


    def _prepare_observation(self, obs):
        pass

    def _maybe_reflect(self):
        return None, ""

    def _apply_pickup_rule(self, obs, observation, model_action):
        return model_action, ""
