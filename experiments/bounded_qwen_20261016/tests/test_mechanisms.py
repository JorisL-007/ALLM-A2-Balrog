"""Mechanics and intervention invariants; no experimental LLM calls."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'BALROG'))
from agents import FocusedAgent, correct_mechanics, legacy
from balrog.prompt_builder.history import HistoryPromptBuilder
from run import budget_remaining

Response = namedtuple('Response','model_id completion stop_reason input_tokens output_tokens reasoning')


class Client:
    def __init__(self): self.prompts=[]
    def generate(self,messages):
        self.prompts.append(copy.deepcopy(messages))
        return Response('fixture','SUBGOAL: approach\nACTION: turn left','stop',10,12,None)


class Tests(unittest.TestCase):
    def test_installed_pickup_requires_front_and_carrying_is_textual(self):
        import gymnasium as gym
        import minigrid
        minigrid.register_minigrid_envs()
        from minigrid.core.world_object import Key
        env=gym.make('MiniGrid-Empty-5x5-v0').unwrapped
        self.addCleanup(env.close)
        env.reset(seed=7)
        env.agent_pos=(1,1);env.agent_dir=0
        key=Key('blue');env.grid.set(2,1,key)
        env.step(env.actions.forward)
        self.assertEqual(tuple(env.agent_pos),(1,1))
        self.assertIsNone(env.carrying)
        env.step(env.actions.toggle)
        self.assertIsNone(env.carrying)
        _,_,_,_,info=env.step(env.actions.pickup)
        self.assertIs(env.carrying,key)
        self.assertIsNone(env.grid.get(2,1))
        self.assertIn('You carry a blue key',info['descriptions'])

    def test_correction_removes_both_wrong_hints_and_preserves_mission(self):
        from balrog.environments.babyai_text import get_instruction_prompt
        old=get_instruction_prompt(None,mission='pick up the blue ball')
        new=correct_mechanics(old)
        self.assertIn('pick up the blue ball',new)
        self.assertNotIn('below you',new)
        self.assertNotIn('interact or pickup',new)
        self.assertEqual(correct_mechanics(new),new)

    def agent(self,variant,environment='babyai'):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        client=Client();builder=HistoryPromptBuilder(16,0,max_cot_history=1)
        with patch.dict(os.environ,EX2_VARIANT='planning',EX2_ENVIRONMENT=environment,EX2_LOG_DIR=folder.name):
            actor=FocusedAgent(lambda:client,builder,variant)
        builder.update_instruction_prompt('Mission. pick up the object below you')
        return actor,client

    def test_transition_records_executed_action_without_a_critic_call(self):
        actor,client=self.agent('transition')
        before={'long_term_context':'a blue key 1 step forward','short_term_context':''}
        after={'long_term_context':'You carry a blue key','short_term_context':''}
        actor.observe_transition('pick up',before,after)
        actor.act({'text':after},prev_action='INVALID_ACTION')
        self.assertEqual(len(client.prompts),1)
        prompt=client.prompts[0][-1].content
        self.assertIn('Executed: pick up',prompt)
        self.assertIn('Removed lines:',prompt)
        actor.reset()
        self.assertIsNone(actor.latest_transition)
        self.assertIsNone(actor.prompt_builder.previous_reasoning)

    def test_mechanics_is_babyai_only(self):
        actor,client=self.agent('mechanics','crafter')
        actor.act({'text':{'long_term_context':'tree','short_term_context':''}})
        self.assertIn('below you',client.prompts[0][0].content)

    def test_unchanged_text_does_not_claim_failure(self):
        actor,_=self.agent('transition','crafter')
        text={'long_term_context':'You face tree','short_term_context':'wood: 1'}
        actor.observe_transition('Do',text,text)
        self.assertIn('does not prove',actor.latest_transition)

    def test_first_limit_wins_and_never_negative(self):
        state=dict(benchmark_stop_unix=1000,limits={'benchmark_seconds':7200},benchmark_seconds_used=7190)
        self.assertEqual(budget_remaining(state,now=100),10)
        self.assertEqual(budget_remaining(state,now=999),1)
        self.assertEqual(budget_remaining(state,now=1001),0)

    def test_development_manifest_is_fixed_stratified_and_heldout_free(self):
        manifest=json.loads((HERE/'development_manifest.json').read_text())
        self.assertEqual(len(manifest),18)
        self.assertEqual(len({c['case_id'] for c in manifest}),18)
        self.assertEqual({c['seed'] for c in manifest},{11,12})
        self.assertEqual(sum(c['environment']=='babyai' for c in manifest),10)
        self.assertEqual(sum(c['environment']=='crafter' for c in manifest),2)
        self.assertEqual(sum(c['environment']=='babaisai' for c in manifest),6)


if __name__=='__main__': unittest.main()
