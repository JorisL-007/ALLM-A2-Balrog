"""Run with TEST_BALROG_DIR=/path/to/BALROG python3 -m unittest discover -s tests -v.
Loads BALROG's real BaseAgent/history builder without importing model/game clients.
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('TEST_BALROG_DIR', str(PACKAGE.parent / 'BALROG')))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault('balrog', types.ModuleType('balrog'))
sys.modules.setdefault('balrog.agents', types.ModuleType('balrog.agents'))
load_file('balrog.agents.base', REPO / 'balrog/agents/base.py')
History = load_file('test_history', REPO / 'balrog/prompt_builder/history.py').HistoryPromptBuilder
sys.path.insert(0, str(PACKAGE))
from ex2_agent import CustomAgent, check_action, extract_action

Response = namedtuple('Response', 'model_id completion stop_reason input_tokens output_tokens reasoning')


def observation(long='', short=''):
    return {'text': {'long_term_context': long, 'short_term_context': short}}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages[-1].content)
        return Response('test', next(self.responses), 'stop', 10, 20, None)


class AgentTests(unittest.TestCase):
    def test_parser_requires_one_action_line(self):
        self.assertEqual(extract_action('PLAN: collect wood\nACTION: do', 'crafter'), 'Do')
        self.assertIsNone(extract_action('ACTION: left\nACTION: up', 'babaisai'))
        self.assertEqual(extract_action('ACTION: Collect Wood', 'crafter'), 'Collect Wood')

    def test_babyai_front_and_diagonal_objects(self):
        self.assertIsNotNone(check_action('go forward', 'babyai', observation('a blue key 1 step forward')))
        self.assertIsNone(check_action('pick up', 'babyai', observation('a blue key 1 step forward')))
        self.assertIsNotNone(check_action('pick up', 'babyai', observation('a blue key 1 step right and 1 step forward')))
        self.assertIsNone(check_action('go forward', 'babyai', observation('a blue key 1 step right and 1 step forward')))
        self.assertIsNone(check_action('go forward', 'babyai', observation('a green open door 1 step forward')))

    def test_crafter_current_inventory_and_unknowns(self):
        obs = observation('You face a tree.', 'Your status:\n- energy: 8/9\n\nYour inventory:\n- wood: 2\n- sapling: 1')
        self.assertIsNone(check_action('Make Wood Pickaxe', 'crafter', obs))
        self.assertIsNotNone(check_action('Make Stone Pickaxe', 'crafter', obs))
        self.assertIsNone(check_action('Place Plant', 'crafter', obs))
        self.assertIsNotNone(check_action('Place Table', 'crafter', observation('', 'You have nothing in your inventory.')))
        self.assertIsNone(check_action('Make Stone Pickaxe', 'crafter', observation('no inventory information')))

    def test_babaisai_does_not_assume_fixed_collision_rules(self):
        obs = observation('Active rules:\nWALL IS PUSH\nObjects on the map:\nwall 1 step up')
        self.assertIsNone(check_action('up', 'babaisai', obs))
        self.assertIsNotNone(check_action('go forward', 'babaisai', obs))

    def run_agent(self, variant, responses, obs, env='babyai'):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        client = Client(responses)
        builder = History(max_text_history=16, max_image_history=0, max_cot_history=1)
        with patch.dict(os.environ, EX2_VARIANT=variant, EX2_ENVIRONMENT=env, EX2_LOG_DIR=folder.name):
            agent = CustomAgent(lambda: client, builder)
        builder.update_instruction_prompt('Task: put the blue key next to the red ball')
        result = agent.act(obs)
        record = json.loads(agent.log_path.read_text().strip())
        return agent, client, result, record

    def test_identical_first_prompt_in_both_variants(self):
        prompts = []
        for variant in ('planning', 'grounded'):
            _, client, _, _ = self.run_agent(variant, ['PLAN: turn\nACTION: turn left'], observation('a wall 1 step forward'))
            prompts.append(client.prompts[0])
        self.assertEqual(prompts[0], prompts[1])

    def test_planning_does_not_repair_or_hide_invalid_commands(self):
        _, client, result, record = self.run_agent('planning', ['PLAN: get wood\nACTION: Collect Wood'], observation(), 'crafter')
        self.assertEqual(result.completion, 'Collect Wood')
        self.assertEqual(len(client.prompts), 1)
        self.assertIsNotNone(record['calls'][0]['violation'])

    def test_grounding_repairs_and_counts_both_calls(self):
        _, client, result, record = self.run_agent('grounded', ['ACTION: go forward', 'PLAN: turn\nACTION: turn right'], observation('a wall 1 step forward'))
        self.assertEqual(result.completion, 'turn right')
        self.assertEqual((result.input_tokens, result.output_tokens), (20, 40))
        self.assertEqual(len(record['calls']), 2)
        self.assertIn('blocks forward movement', client.prompts[1])
        self.assertFalse(record['agent_fallback'])

    def test_failed_repair_uses_logged_fallback(self):
        _, _, result, record = self.run_agent('grounded', ['ACTION: go forward', 'ACTION: go forward'], observation('a wall 1 step forward'))
        self.assertEqual(result.completion, 'turn left')
        self.assertTrue(record['agent_fallback'])
        self.assertEqual(len(record['calls']), 2)

    def test_reset_clears_reasoning_and_uses_new_log(self):
        agent, _, _, _ = self.run_agent('planning', ['ACTION: turn left'], observation())
        previous = agent.log_path
        agent.reset()
        self.assertIsNone(agent.prompt_builder.previous_reasoning)
        self.assertEqual(agent.step, 0)
        self.assertNotEqual(agent.log_path, previous)
        self.assertEqual(len(agent.prompt_builder._events), 0)


if __name__ == '__main__':
    unittest.main()
