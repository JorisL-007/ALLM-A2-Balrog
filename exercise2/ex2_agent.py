"""Two small BALROG agents: structured planning, optionally with external checks."""
import copy
import json
import os
import re
import time
import uuid
from pathlib import Path

from balrog.agents.base import BaseAgent

ACTIONS = {
    'babyai': ('turn left', 'turn right', 'go forward', 'pick up', 'drop', 'toggle'),
    'crafter': ('Noop', 'Move West', 'Move East', 'Move North', 'Move South', 'Do',
                'Sleep', 'Place Stone', 'Place Table', 'Place Furnace', 'Place Plant',
                'Make Wood Pickaxe', 'Make Stone Pickaxe', 'Make Iron Pickaxe',
                'Make Wood Sword', 'Make Stone Sword', 'Make Iron Sword'),
    'babaisai': ('idle', 'up', 'right', 'down', 'left'),
}
FALLBACK = {'babyai': 'turn left', 'crafter': 'Noop', 'babaisai': 'idle'}

PLANNING_PROMPT = """Choose the next action toward the game's objective.
Use the current observation and provided history. Describe the immediate subgoal,
then at most three short steps toward it. State the requirements for your next
primitive action. Do not invent objects, inventory, positions or completed steps;
mark requirements unknown if the observation/history does not establish them.
Execute only ONE primitive action, then wait for the next observation. Keep the
whole response under 150 words. Use exactly these four fields, ACTION last:
SUBGOAL: <immediate objective>
PLAN: <at most three short steps>
PRECONDITION: <requirements for the next action, including any uncertainty>
ACTION: <exactly one action from the list below, without extra text>
Allowed actions: {actions}
"""


def extract_action(text, environment):
    """Require exactly one ACTION line; leave unsupported commands detectable."""
    matches = re.findall(r'^\s*ACTION:\s*([^\n]*)$', text, re.IGNORECASE | re.MULTILINE)
    if len(matches) != 1:
        return None
    action = matches[0].strip()
    canonical = {a.casefold(): a for a in ACTIONS[environment]}
    return canonical.get(action.casefold(), action)


def check_action(action, environment, observation):
    """Use only exposed text. Unknown requirements do not cause rejection.

    These are necessary-condition checks, not a complete game simulator.
    """
    if action not in ACTIONS[environment]:
        return 'Return exactly one ACTION line containing an allowed primitive action.'
    text = observation.get('text', {})
    long_text = text.get('long_term_context', '').lower()
    current = '\n'.join(text.get(k, '') for k in ('short_term_context', 'long_term_context')).lower()
    if environment == 'babyai':
        # Coordinates are relative to the current facing direction. A diagonal
        # object does not match the end-anchored "1 step forward" pattern.
        front = re.findall(r'^a ([a-z ]+?) 1 step forward\s*$', long_text, re.MULTILINE)
        if action == 'go forward' and any(re.search(r'\b(wall|key|ball|box)\b', item) for item in front):
            return 'The observed wall/object directly ahead blocks forward movement.'
        if action == 'pick up':
            # Do not infer carrying state: BabyAI's text does not expose inventory.
            if not any(re.search(r'\b(key|ball|box)\b', item) for item in front):
                return 'No pickup object is described exactly one step directly ahead.'
        if action == 'drop' and front:
            return 'Dropping requires an empty tile directly ahead; it is visibly occupied.'
    elif environment == 'crafter':
        # Inventory lists only nonzero items, so absent entries mean zero only
        # when an explicit inventory section is present in THIS observation.
        if 'you have nothing in your inventory.' in current:
            inventory = {}
        elif 'your inventory:' in current:
            section = current.split('your inventory:', 1)[1].split('\n\n', 1)[0]
            inventory = {k: int(v) for k, v in re.findall(r'^- ([a-z_]+): (\d+)\s*$', section, re.MULTILINE)}
        else:
            inventory = None
        # Check only clearly necessary materials, not exact recipe quantities
        # or proximity to workstations (which may be uncertain).
        required = {
            'Place Stone': 'stone', 'Place Table': 'wood',
            'Place Furnace': 'stone', 'Place Plant': 'sapling',
        }
        if action.startswith('Make '):
            required[action] = action.split()[1].lower()
        resource = required.get(action)
        if inventory is not None and resource and inventory.get(resource, 0) == 0:
            return f'{action} requires some {resource}, but the current inventory contains none.'
    # BabaIsAI: STOP/PUSH/YOU rules can change; do not impose static collision rules.
    return None


class CustomAgent(BaseAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.variant = os.environ['EX2_VARIANT']
        self.environment = os.environ['EX2_ENVIRONMENT']
        if self.variant not in ('planning', 'grounded') or self.environment not in ACTIONS:
            raise ValueError('Unsupported Exercise 2 variant or environment')
        self.log_dir = Path(os.environ['EX2_LOG_DIR'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        super().reset()
        # The upstream builder's reset does not clear previous_reasoning.
        self.prompt_builder.update_reasoning(None)
        self.step = 0
        self.log_path = self.log_dir / f'{os.getpid()}_{uuid.uuid4().hex}.jsonl'

    def act(self, obs, prev_action=None):
        started = time.monotonic()
        if prev_action is not None:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        # Avoid mutating messages held by a prompt builder.
        messages = copy.deepcopy(self.prompt_builder.get_prompt())
        if not messages or messages[-1].role != 'user':
            raise RuntimeError('Expected a final observation message')
        messages[-1].content += '\n\n' + PLANNING_PROMPT.format(actions=', '.join(ACTIONS[self.environment]))
        calls = []
        attempts = 2 if self.variant == 'grounded' else 1
        total_input = total_output = 0
        fallback = False
        final_raw = ''
        for attempt in range(attempts):
            response = self.client.generate(messages)
            final_raw = response.completion or ''
            total_input += response.input_tokens or 0
            total_output += response.output_tokens or 0
            action = extract_action(final_raw, self.environment)
            violation = check_action(action, self.environment, obs)
            calls.append({
                'messages': [{'role': m.role, 'content': m.content} for m in messages],
                'completion': final_raw, 'stop_reason': response.stop_reason,
                'action': action, 'violation': violation,
                'input_tokens': response.input_tokens, 'output_tokens': response.output_tokens,
            })
            if self.variant == 'planning' or violation is None:
                break
            if attempt + 1 < attempts:
                # Constraint feedback only: no diagnosis of the trajectory, critic
                # role, persistent lesson, or self-reflection component.
                messages[-1].content += (
                    '\n\nA Python action check rejected your proposed action '
                    f'{action!r}: {violation}\n'
                    'Produce the four fields again with one feasible allowed ACTION.'
                )
            else:
                action = FALLBACK[self.environment]
                fallback = True
        if action is None:
            # Planning-only: let normal BALROG invalid-action handling apply.
            action = 'INVALID_ACTION'
        self.prompt_builder.update_reasoning(final_raw)
        record = {
            'step': self.step, 'variant': self.variant, 'environment': self.environment,
            'observation': obs.get('text', {}), 'calls': calls, 'action': action,
            'agent_fallback': fallback, 'input_tokens': total_input,
            'output_tokens': total_output, 'elapsed_seconds': time.monotonic() - started,
        }
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.step += 1
        # Both calls count toward BALROG's token usage. Store the raw plan in CSVs.
        reasoning = final_raw
        if self.variant == 'grounded' and any(c['violation'] for c in calls):
            reasoning += '\nPYTHON_CHECK: ' + json.dumps({
                'violations': [c['violation'] for c in calls], 'fallback': fallback,
            })
        return response._replace(completion=action, reasoning=reasoning,
                                 input_tokens=total_input, output_tokens=total_output)
