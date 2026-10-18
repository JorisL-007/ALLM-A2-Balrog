"""Replay saved replies through archived and refactored agents; never call a model."""
import copy
import csv
import importlib.util
import json
import os
import re
import sys
import tarfile
import tempfile
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'BALROG')]
from balrog.prompt_builder.history import HistoryPromptBuilder
from exercises.common.agent_registry import create_research_agent

Response = namedtuple('Response', 'completion input_tokens output_tokens reasoning')
STATE = ('plan', 'facts', 'expected', 'seen_observations', 'observation_history',
         'action_history', 'lessons', 'step_number', 'last_reflection_step', 'reflections_used')

class SavedClient:
    def __init__(self, calls):
        self.calls = calls
        self.index = 0
        self.prompts = []

    def generate(self, messages):
        self.prompts.append([{'role': m.role, 'content': m.content} for m in messages])
        call = self.calls[self.index]
        self.index += 1
        return Response(call['raw_response'], call['input_tokens'], call['output_tokens'], '')


def replay(path, old_class, condition, history=16, missing_action=False):
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    if missing_action:
        index = next(i for i, c in enumerate(calls) if c['role'] == 'reflection')
        calls = copy.deepcopy(calls[:index + 1])
        calls[-1]['raw_response'] = re.sub(r'(?m)^ACTION:.*$', '', calls[-1]['raw_response'])
        calls.append({**calls[-1], 'role': 'action', 'raw_response': 'SUBGOALS: same\nFACTS: same\nEXPECTED: changed view\nACTION: turn left'})
    trajectory = next(path.parent.rglob('*_run_00.csv'))
    rows = list(csv.DictReader(trajectory.open()))
    flags = {'BALROG_CUSTOM_MODE': 'reflect' if condition == 'e3' else 'planning',
             'BALROG_CUSTOM_GUIDE': str(int(condition.startswith('guide'))),
             'BALROG_CUSTOM_GATE': str(int(condition == 'guide_gate')),
             'BALROG_CUSTOM_GOAL_STATE': str(int(condition == 'guide_goal'))}
    os.environ.update(flags)
    os.environ.pop('BALROG_AGENT_VARIANT', None)
    os.environ.pop('BALROG_LLM_LOG', None)
    clients = [SavedClient(calls), SavedClient(calls)]
    system = calls[0]['messages'][0]['content']
    mission = re.search(r'Your goal is to (.*?)\. The following', system)
    builders = [HistoryPromptBuilder(history, 0, system, 1) for _ in range(2)]
    agents = [old_class(lambda: clients[0], builders[0]),
              create_research_agent(lambda: clients[1], builders[1])]
    totals = [0, 0]
    steps = 0
    while clients[0].index < len(calls):
        current = calls[clients[0].index]['messages'][-1]['content']
        observation = re.split(r'\n\n(?:Persistent subgoals:|Critique this concrete failure:)', current)[0]
        observation = observation.removeprefix('Current Observation:\n')
        short = ''
        if 'Crafter' in system and '\nYou see:' in observation:
            short, observation = observation.split('\nYou see:', 1)
            observation = 'You see:' + observation
            if '\n\nYour previous output' in short:
                short, feedback = short.split('\n\nYour previous output', 1)
                observation = '\nYour previous output' + feedback + '\n' + observation
                short = short.removesuffix('\n')
                observation = '\n' + observation
        obs = {'text': {'long_term_context': observation, 'short_term_context': short}}
        if mission:
            obs['mission'] = mission.group(1)
        prev = rows[steps-1].get('Executed Action', rows[steps-1]['Action']) if steps else None
        outputs = [a.act(copy.deepcopy(obs), prev) for a in agents]
        assert vars(outputs[0]) == vars(outputs[1]), (path, steps, outputs)
        assert {k: getattr(agents[0], k) for k in STATE} == {k: getattr(agents[1], k) for k in STATE}, (path, steps, 'state')
        assert clients[0].prompts == clients[1].prompts, (path, steps, 'prompt')
        totals[0] += outputs[0].input_tokens
        totals[1] += outputs[0].output_tokens
        steps += 1
    if not missing_action:
        if clients[0].prompts != [c['messages'] for c in calls]:
            import difflib
            for actual, saved in zip(clients[0].prompts, calls):
                if actual != saved['messages']:
                    print('\n'.join(difflib.unified_diff(json.dumps(saved['messages'], indent=2).splitlines(), json.dumps(actual, indent=2).splitlines())))
                    break
            raise AssertionError((path, 'saved prompts differ'))
    for agent in agents:
        agent.reset()
    assert {k: getattr(agents[0], k) for k in STATE} == {k: getattr(agents[1], k) for k in STATE}
    # Reuse the same objects after reset, as BALROG does.
    for client in clients:
        client.index = 0
        client.prompts = []
    obs_text = calls[0]['messages'][-1]['content'].split('\n\nPersistent subgoals:')[0].removeprefix('Current Observation:\n')
    obs = {'text': {'long_term_context': obs_text, 'short_term_context': ''}}
    if mission:
        obs['mission'] = mission.group(1)
    outputs = [a.act(copy.deepcopy(obs)) for a in agents]
    assert vars(outputs[0]) == vars(outputs[1]) and clients[0].prompts == clients[1].prompts
    return {'source': str(path.relative_to(ROOT)), 'condition': condition, 'history': history,
            'steps': steps, 'calls': len(calls), 'tokens': totals, 'missing_critic_action': missing_action,
            'prompts_actions_state_reset': 'identical'}


def main():
    archive = ROOT / 'archive/refactor_20261018/source_before.tar.gz'
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / 'old_custom.py'
        with tarfile.open(archive) as tar:
            source.write_bytes(tar.extractfile('BALROG/balrog/agents/custom.py').read())
        spec = importlib.util.spec_from_file_location('old_custom', source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        base = ROOT / 'experiment_results/followup_v1'
        cases = [('plan', 'pickup', 30), ('e3', 'goto', 30), ('guide', 'pickup', 30),
                 ('guide_gate', 'pickup', 30), ('guide_goal', 'pickup', 30),
                 ('guide_h4', 'open', 30), ('guide_h32', 'pickup', 30)]
        results = []
        for condition, task, seed in cases:
            path = base / f'episodes/babyai/{task}/{condition}/seed_{seed:03d}/rep_0/llm_calls.jsonl'
            history = 4 if condition == 'guide_h4' else 32 if condition == 'guide_h32' else 16
            results.append(replay(path, old.CustomAgent, condition, history))
        path = base / 'episodes/crafter/default/guide_goal/seed_120/rep_0/llm_calls.jsonl'
        results.append(replay(path, old.CustomAgent, 'guide_goal'))
        for path in sorted((base / 'episodes/babyai/goto/e3').rglob('llm_calls.jsonl')):
            if '"role": "reflection"' in path.read_text():
                results.append(replay(path, old.CustomAgent, 'e3', missing_action=True))
                break
        output = ROOT / 'experiment_results/refactored_v1/verification/replay.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + '\n')
        print(f'{len(results)} saved-call comparisons passed: {output}')

if __name__ == '__main__':
    main()
