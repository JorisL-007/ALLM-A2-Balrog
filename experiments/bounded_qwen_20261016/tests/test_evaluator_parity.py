"""Differential audit against the actual BALROG evaluator, with scripted commands.

Two five-step real-environment fixtures; no LLM calls and no scored screening data.
Includes an invalid command to exercise the legacy fallback/history behavior.
"""
import copy
import json
import sys
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'BALROG'))
import episode

Response=namedtuple('Response','model_id completion stop_reason input_tokens output_tokens reasoning')


class ScriptedClient:
    def __init__(self,*args):
        self.count=self.input_tokens=self.output_tokens=0
        self.messages=[]
        self.commands=iter(['bogus','turn right','go forward','turn left','pick up'])
    def generate(self,messages):
        self.messages.append([(m.role,m.content) for m in copy.deepcopy(messages)])
        self.count+=1;self.input_tokens+=10;self.output_tokens+=5
        return Response('scripted_unit_fixture',next(self.commands),'stop',10,5,None)


class EvaluatorParity(unittest.TestCase):
    def test_reference_loop_matches_original_prompts_scores_and_fallback_history(self):
        from omegaconf import OmegaConf
        from balrog.evaluator import Evaluator
        from balrog.agents.naive import NaiveAgent
        from balrog.prompt_builder import create_prompt_builder
        cfg=OmegaConf.load(ROOT/'BALROG/balrog/config/config.yaml')
        cfg.envs.env_kwargs.seed=11
        cfg.eval.max_steps_per_episode=5
        cfg.eval.num_workers=1
        original_client=ScriptedClient()
        original_agent=NaiveAgent(lambda:original_client,create_prompt_builder(cfg.agent))
        case=dict(case_id='babyai:BabyAI-MixedTrainLocal-v0/goto:11',environment='babyai',
                  task='BabyAI-MixedTrainLocal-v0/goto',seed=11)
        replacement=ScriptedClient()
        with tempfile.TemporaryDirectory() as tmp:
            original=Evaluator('babyai',cfg,original_cwd=str(ROOT/'BALROG'),output_dir=str(Path(tmp)/'original'))
            measured=original.run_episode(case['task'],original_agent)
            with patch.object(episode,'LoggedClient',return_value=replacement):
                instrumented=episode.run(case,'naive',Path(tmp)/'instrumented',time.time()+30,setup_steps=5)
            self.assertEqual(original_client.messages,replacement.messages)
            self.assertEqual(measured['progression'],instrumented['progression'])
            self.assertEqual(measured['num_steps'],instrumented['steps'])
            self.assertEqual(measured['input_tokens'],instrumented['input_tokens'])
            self.assertEqual(measured['output_tokens'],instrumented['output_tokens'])
            steps=[json.loads(x) for x in (Path(tmp)/'instrumented/steps.jsonl').read_text().splitlines()]
            self.assertEqual(steps[0]['proposed_action'],'bogus')
            self.assertEqual(steps[0]['executed_action'],'go forward')
            self.assertEqual(instrumented['invalid_commands'],1)
        (HERE/'audit/evaluator_parity.json').write_text(json.dumps(dict(status='passed',
            real_environment_fixtures=2,steps_each=5,experimental_llm_calls=0,
            verified=['identical full prompts over five decisions','identical progression and step count',
                      'identical token accounting with scripted client','invalid proposal vs executed fallback',
                      'legacy proposed-action history retained']),indent=2)+'\n')


if __name__=='__main__':unittest.main()
