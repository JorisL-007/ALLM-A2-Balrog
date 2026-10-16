"""Reflection limits, exclusions, accounting and reset; fake client, no LLM calls."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_mechanisms import Client, Response
from balrog.prompt_builder.history import HistoryPromptBuilder

spec=importlib.util.spec_from_file_location('pilot_reflection',Path(__file__).resolve().parents[1]/'e3/reflection.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Actor:
    def __init__(self):
        self.client=Client();self.client.tokens=512
        self.prompt_builder=HistoryPromptBuilder(16,0,max_cot_history=1)
    def reset(self):self.prompt_builder.reset()
    def act(self,obs,prev_action=None):
        self.prompt_builder.update_observation(obs)
        return self.client.generate(self.prompt_builder.get_prompt())


class ReflectionTests(unittest.TestCase):
    def wrapper(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        return module.ReflectionAgent(Actor(),'babyai',Path(tmp.name)/'reflections.jsonl')
    def stuck(self,agent,action='go forward'):
        t={'long_term_context':'a wall 1 step forward','short_term_context':''}
        agent.observe_transition(action,t,t)
        return agent.act({'text':t})
    def test_trigger_counts_tokens_restores_actor_budget_and_obeys_caps(self):
        agent=self.wrapper()
        self.stuck(agent);self.stuck(agent);r=self.stuck(agent)
        self.assertEqual(agent.reflections,1)
        self.assertEqual((r.input_tokens,r.output_tokens),(20,24))
        self.assertEqual(agent.client.tokens,512)
        for _ in range(40):self.stuck(agent)
        self.assertEqual(agent.reflections,2)
        self.assertLessEqual(len(agent.lessons),2)
        self.assertTrue(all(len(x)<=400 for x in agent.lessons))
        agent.reset()
        self.assertEqual(agent.lessons,[])
        self.assertEqual(agent.reflections,0)
        self.assertEqual(agent.streak,0)
    def test_resource_gathering_waiting_and_sleep_do_not_trigger(self):
        for action in ['Do','Sleep','Noop','idle']:
            agent=self.wrapper()
            for _ in range(20):self.stuck(agent,action)
            self.assertEqual(agent.reflections,0)
    def test_effectful_actions_clear_streak(self):
        agent=self.wrapper();self.stuck(agent);self.stuck(agent)
        agent.observe_transition('go forward',{'long_term_context':'old'},{'long_term_context':'new'})
        self.assertEqual(agent.streak,0)


if __name__=='__main__':unittest.main()
