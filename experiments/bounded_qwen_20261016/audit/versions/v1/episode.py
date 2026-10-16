"""One instrumented episode using BALROG's installed wrappers and legacy policy.

The action loop mirrors evaluator.py, including legacy previous-proposal feedback.
All experimental LLM calls go to the literal loopback endpoint and required model.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = 'Qwen/Qwen3-4B-Instruct-2507-FP8'
ENDPOINT = 'http://127.0.0.1:8088/v1'
sys.path.insert(0, str(ROOT / 'BALROG'))


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()


def fingerprint(text, mission=None):
    return hashlib.sha256(json.dumps({'text': text, 'mission': mission}, sort_keys=True).encode()).hexdigest()


class LoggedClient:
    def __init__(self, directory, tokens, deadline):
        from openai import OpenAI
        self.api = OpenAI(api_key='EMPTY', base_url=ENDPOINT, max_retries=0, timeout=60)
        self.directory, self.tokens, self.deadline = Path(directory), tokens, deadline
        self.count = self.input_tokens = self.output_tokens = 0

    def generate(self, messages):
        Response = namedtuple('Response', 'model_id completion stop_reason input_tokens output_tokens reasoning')
        payload = [{'role': m.role, 'content': [{'type': 'text', 'text': m.content}]} for m in messages]
        if any(getattr(m, 'attachment', None) is not None for m in messages):
            raise ValueError('Text-only protocol unexpectedly received an image')
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Episode budget exhausted before LLM request')
        call_id = self.count
        self.count += 1
        started = time.monotonic()
        append(self.directory/'calls.jsonl', dict(event='request', call_id=call_id, model=MODEL,
               messages=payload, max_tokens=self.tokens, temperature='model default', at_unix=time.time()))
        try:
            r = self.api.chat.completions.create(model=MODEL, messages=payload,
                    max_tokens=self.tokens, timeout=min(60, remaining))
            c = r.choices[0]
            data = dict(event='response', call_id=call_id, completion=c.message.content,
                        stop_reason=c.finish_reason, input_tokens=r.usage.prompt_tokens,
                        output_tokens=r.usage.completion_tokens, elapsed_seconds=time.monotonic()-started)
            self.input_tokens += r.usage.prompt_tokens
            self.output_tokens += r.usage.completion_tokens
            append(self.directory/'calls.jsonl', data)
            return Response(MODEL, (c.message.content or '').strip(), c.finish_reason,
                            r.usage.prompt_tokens, r.usage.completion_tokens, None)
        except Exception as exc:
            append(self.directory/'calls.jsonl', dict(event='error', call_id=call_id,
                    error=repr(exc), elapsed_seconds=time.monotonic()-started))
            raise


def run(spec, variant, directory, deadline, setup_steps=None):
    import numpy as np
    from omegaconf import OmegaConf
    from balrog.environments import make_env
    from balrog.prompt_builder import create_prompt_builder
    from agents import make_agent
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(ROOT/'BALROG/balrog/config/config.yaml')
    seed, environment = spec['seed'], spec['environment']
    cfg.envs.env_kwargs.seed = seed
    cfg.envs.crafter_kwargs.seed = seed
    cfg.agent.max_text_history = 16
    cfg.agent.max_image_history = 0
    cfg.agent.max_cot_history = 1
    cfg.client.client_name = 'vllm'
    cfg.client.model_id = MODEL
    cfg.client.base_url = ENDPOINT
    cfg.client.generate_kwargs.temperature = None
    cfg.client.generate_kwargs.max_tokens = 128 if variant == 'naive' else 512
    cfg.eval.num_workers = 1
    cfg.eval.max_steps_per_episode = setup_steps
    OmegaConf.save(cfg, directory/'config.yaml')
    os.environ.update(EX2_VARIANT='grounded' if variant == 'grounded' else 'planning',
                      EX2_ENVIRONMENT=environment, EX2_LOG_DIR=str(directory/'agent_raw'))
    client = LoggedClient(directory, cfg.client.generate_kwargs.max_tokens, deadline)
    agent = make_agent(variant, lambda: client, create_prompt_builder(cfg.agent))
    # Seed before construction as well as reset; layout fingerprints are still checked.
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(environment, spec['task'], cfg)
    agent.reset()
    obs, _ = env.reset(seed=seed)
    mission = obs.get('mission')
    initial = dict(text=copy.deepcopy(obs['text']), mission=mission)
    initial['fingerprint'] = fingerprint(initial['text'], mission)
    (directory/'initial.json').write_text(json.dumps(initial, indent=2)+'\n')
    agent.prompt_builder.update_instruction_prompt(env.get_instruction_prompt(instructions=mission))
    previous_proposal = None
    previous_executed = None
    total_reward = 0
    invalid = unchanged = repeated = 0
    terminated = truncated = False
    started = time.monotonic()
    try:
        for step in range(setup_steps or env.max_steps):
            if time.time() >= deadline:
                raise TimeoutError('Episode budget exhausted')
            before = copy.deepcopy(obs['text'])
            response = agent.act(obs, prev_action=previous_proposal)
            proposed = response.completion
            executed = env.check_action_validity(proposed)
            obs, reward, terminated, truncated, _ = env.step(executed)
            after = copy.deepcopy(obs['text'])
            invalid += executed != proposed
            unchanged += before == after
            repeated += executed == previous_executed
            append(directory/'steps.jsonl', dict(step=step, observation=before, proposed_action=proposed,
                executed_action=executed, next_observation=after, text_unchanged=before==after,
                repeated_action=executed==previous_executed, invalid=executed!=proposed,
                reasoning=response.reasoning, reward=float(reward), terminated=bool(terminated),
                truncated=bool(truncated), input_tokens=response.input_tokens, output_tokens=response.output_tokens))
            if hasattr(agent, 'observe_transition'):
                agent.observe_transition(executed, before, after)
            # Preserve legacy history handling in all arms. Transition is an explicit added factor.
            if executed != proposed and cfg.eval.feedback_on_invalid_action:
                obs['text']['long_term_context'] = (
                    f'\n\nYour previous output did not contain a valid action. Defaulted to action: {executed}\n\nObservation:\n'
                    + obs['text']['long_term_context'])
            previous_proposal, previous_executed = proposed, executed
            total_reward += reward
            if terminated or truncated:
                break
        # Only progression is read from stats; never serialize or expose target_plan.
        progression = float(env.get_stats()['progression'])
        result = dict(spec, variant=variant, status='setup' if setup_steps else 'complete',
            progression=progression, steps=step+1, terminated=bool(terminated), truncated=bool(truncated),
            episode_return=float(total_reward), initial_fingerprint=initial['fingerprint'],
            calls=client.count, input_tokens=client.input_tokens, output_tokens=client.output_tokens,
            invalid_commands=invalid, unchanged_text=unchanged, repeated_actions=repeated,
            decision_seconds=time.monotonic()-started, setup_steps=setup_steps)
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--deadline', required=True, type=float)
    p.add_argument('--setup-steps', type=int)
    a = p.parse_args()
    run(json.loads(a.spec), a.variant, a.directory, a.deadline, a.setup_steps)
