"""Focused E2 changes to the existing assignment agent; no oracle or learned memory.

Course book: 6.1.3 (decomposition), 6.2.3 (external state), 7.3.2
(action consequences), 10.2.5 (small harness interventions).
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'exercise2'))
import ex2_agent as legacy

SHORT_PROMPT = """Choose one immediate subgoal using only the observation and history.
Give one short subgoal sentence, then exactly one primitive action. Reconsider
the subgoal after the next observation. Use exactly these two fields:
SUBGOAL: <one short sentence>
ACTION: <one exact allowed action, without extra text>
Allowed actions: {actions}
"""


def correct_mechanics(text):
    """Replace the two incorrect BabyAI hints; leave all other instructions intact."""
    return text.replace('pick up the object below you',
                        'pick up a key, ball or box directly in front of you, if your hands are empty').replace(
        "- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.",
        "- Use 'pick up' for a key, ball or box one step directly forward; moving forward cannot enter its occupied tile. Use 'toggle' for doors or boxes. 'drop' puts your carried object on an empty tile directly forward.")


class FocusedAgent(legacy.CustomAgent):
    """Change exactly one prompt factor relative to the legacy planning actor."""
    def __init__(self, client_factory, prompt_builder, intervention):
        self.intervention = intervention
        self.previous_text = None
        self.previous_executed = None
        self.latest_transition = None
        super().__init__(client_factory, prompt_builder)
        original_get = prompt_builder.get_prompt

        def get_prompt(*args, **kwargs):
            messages = copy.deepcopy(original_get(*args, **kwargs))
            if intervention == 'mechanics' and self.environment == 'babyai':
                messages[0].content = correct_mechanics(messages[0].content)
            if intervention == 'transition' and self.latest_transition:
                messages[-1].content += '\n\nPrevious executed action and observed effect:\n' + self.latest_transition
            return messages

        prompt_builder.get_prompt = get_prompt

    def reset(self):
        super().reset()
        self.previous_text = self.previous_executed = self.latest_transition = None

    def observe_transition(self, action, before, after):
        """Only the ordinary agent's text observations and actually executed command."""
        changed = before != after
        self.latest_transition = f'Executed: {action}\nText changed: {changed}\n'
        if changed:
            # A mechanical difference, not a generated critic or learned lesson.
            old = set(('\n'.join(before.values())).splitlines())
            new = set(('\n'.join(after.values())).splitlines())
            self.latest_transition += 'Removed lines: ' + repr(sorted(old-new)) + '\nAdded lines: ' + repr(sorted(new-old))
        else:
            self.latest_transition += 'The exposed text is unchanged; this alone does not prove the action failed.'


def make_agent(variant, client_factory, builder):
    if variant == 'naive':
        from balrog.agents.naive import NaiveAgent
        return NaiveAgent(client_factory, builder)
    if variant in ('planning', 'grounded'):
        return legacy.CustomAgent(client_factory, builder)
    if variant == 'short':
        legacy.PLANNING_PROMPT = SHORT_PROMPT
    return FocusedAgent(client_factory, builder, variant)
