"""Numbered exercise entry points share this resumable BALROG episode runner."""
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exercises.common.profiles import STUDIES, study_cases
from exercises.common.results import verify
from exercises.common.server import ensure_server, stop_server


def source_hashes():
    paths = [p for p in (ROOT / 'exercises').rglob('*.py')
             if p.name not in ('replay_saved_calls.py', 'build_submission_archives.py')]
    paths += list((ROOT / 'exercises').rglob('*.json'))
    paths += list((ROOT / 'exercises').rglob('*.sh'))
    paths += list((ROOT / 'BALROG/balrog').rglob('*.py')) + list((ROOT / 'BALROG/balrog/config').rglob('*.yaml'))
    paths += [ROOT / 'BALROG/eval.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def check_manifest(out, cases):
    expected = {'schema': 'refactored_v1', 'cases': cases, 'source_hashes': source_hashes(),
                'server': {'seed': 0, 'generation_config': 'auto'}}
    path = out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise SystemExit(f'Settings or code changed since this run began: {path}')
    else:
        # Never add a new manifest to a directory of legacy or unverified evidence.
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'Output is not empty and has no matching manifest: {out}')
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + '\n')


def target_directory(out, c):
    if c['panel'].startswith('e1_'):
        return out / c['model_label'] / c['agent'] / f'run_{c["repeat"]}' / c['environment'] / f'seed_{c["seed"]}'
    branch = 'repeat' if c['repeat'] else 'episodes'
    return out / branch / c['environment'] / c['task'] / c['condition'] / f'seed_{c["seed"]:03d}' / f'rep_{c["repeat"]}'


def episode_command(c, work, port):
    null = lambda value: 'null' if value is None else str(value)
    command = ['conda', 'run', '--no-capture-output', '-n', 'balrog', 'python', 'eval.py',
               f'agent.type={c["agent"]}', f'agent.max_text_history={c["history"]}',
               'agent.max_image_history=0', 'agent.remember_cot=True', 'agent.max_cot_history=1',
               f'envs.names={c["environment"]}', f'envs.env_kwargs.seed={c["seed"]}',
               f'eval.num_episodes.{c["environment"]}=1', f'eval.num_workers={c["workers"]}',
               f'eval.max_steps_per_episode={null(c["budget"])}', 'eval.save_trajectories=True',
               f'eval.resume_from={work}', 'client.client_name=vllm', f'client.model_id={c["model"]}',
               f'client.base_url=http://127.0.0.1:{port}/v1',
               f'client.generate_kwargs.temperature={null(c["temperature"])}',
               f'client.generate_kwargs.max_tokens={c["tokens"]}', 'client.generate_kwargs.thinking_budget=null']
    if c['environment'] == 'babyai' and c['task'] != 'all':
        command += [f'tasks.babyai_tasks=[BabyAI-MixedTrainLocal-v0/{c["task"]}]']
    if c['environment'] == 'crafter':
        command += [f'envs.crafter_kwargs.seed={c["seed"]}']
    flags = {'BALROG_AGENT_VARIANT': c['variant'], 'BALROG_CUSTOM_GUIDE': str(c['guide']),
             'BALROG_CUSTOM_GATE': str(c['gate']), 'BALROG_CUSTOM_GOAL_STATE': str(c['goal']),
             'BALROG_CUSTOM_MODE': 'reflect' if 'critique' in c['variant'] else 'planning',
             'BALROG_LLM_LOG': str(work / 'llm_calls.jsonl'), 'HF_HUB_OFFLINE': '1'}
    return command, flags


def verify_case(work, c):
    if c['task'] != 'all':
        condition = 'e3' if 'critique' in c['variant'] else c['condition']
        result = verify(work, c['environment'], c['task'], condition)
        if result and result.get('seed') == c['seed']:
            return result
        return None
    # E1 runs all configured tasks for one seed with four workers, as originally.
    files = list(work.rglob('*_run_*.json'))
    expected = {'babyai': 5, 'crafter': 1, 'babaisai': 40}[c['environment']]
    if len(files) != expected or not (work / 'llm_calls.jsonl').exists():
        return None
    results = [json.loads(p.read_text()) for p in files]
    if len({r['task'] for r in results}) != expected or any(r['seed'] != c['seed'] for r in results):
        return None
    for path, result in zip(files, results):
        with path.with_suffix('.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or len(rows) != result['num_steps']:
            return None
    calls = [json.loads(line) for line in (work / 'llm_calls.jsonl').read_text().splitlines()]
    for key in ('input_tokens', 'output_tokens'):
        if sum(c[key] for c in calls) != sum(r[key] for r in results):
            return None
    return {**{key: sum(r[key] for r in results) for key in ('input_tokens', 'output_tokens', 'num_steps')},
            'progression': sum(r['progression'] for r in results) / len(results)}


def completed_case(target, c):
    path = target / 'complete.json'
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    settings_match = all(meta.get(key) == c[key] for key in ('panel', 'environment', 'task', 'seed', 'condition', 'budget', 'repeat'))
    return settings_match and verify_case(target, c) is not None


def run_case(out, c, port):
    target = target_directory(out, c)
    if target.exists():
        if completed_case(target, c):
            return 'skip', 0
        raise SystemExit(f'Final directory failed verification; retained for inspection: {target}')
    work = target.with_name(target.name + '.working')
    if work.exists():
        work.rename(work.with_name(target.name + f'.incomplete_{time.time_ns()}'))
    work.mkdir(parents=True)
    command, flags = episode_command(c, work, port)
    (work / 'command.json').write_text(json.dumps({'command': command, 'environment': flags}, indent=2) + '\n')
    start = time.monotonic()
    with (work / 'console.log').open('w') as log:
        process = subprocess.run(command, cwd=ROOT / 'BALROG', env=dict(os.environ, **flags), stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    result = verify_case(work, c) if process.returncode == 0 else None
    if result is None:
        raise SystemExit(f'Episode failed verification; attempt retained at {work}')
    meta = {key: c[key] for key in ('panel', 'environment', 'task', 'seed', 'condition', 'budget', 'repeat')}
    meta.update(elapsed_seconds=elapsed, **{key: result[key] for key in ('progression', 'num_steps', 'input_tokens', 'output_tokens')})
    (work / 'complete.json').write_text(json.dumps(meta, indent=2) + '\n')
    work.rename(target)
    return 'done', elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=STUDIES, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model', help='Optional model label filter')
    parser.add_argument('--condition', help='Optional historical condition filter')
    args = parser.parse_args()
    cases = [c for c in study_cases(args.study) if (not args.model or c['model_label'] == args.model)
             and (not args.condition or c['condition'] == args.condition)]
    if not cases:
        parser.error('No cases match these filters')
    out = (args.out or ROOT / 'experiment_results/refactored_v1' / args.study).resolve()
    if args.dry_run:
        for c in cases:
            target = target_directory(out, c)
            command, flags = episode_command(c, target.with_name(target.name + '.working'), args.port)
            print(json.dumps(c, sort_keys=True))
            print('cd ' + shlex.quote(str(ROOT / 'BALROG')) + ' && ' + shlex.join(['env'] + [f'{k}={v}' for k,v in flags.items()] + command))
        print(f'{len(cases)} cases; output {out}; wall limit {args.hours:g} hours; no files written')
        return
    check_manifest(out, cases)
    deadline = time.monotonic() + args.hours * 3600
    done = 0
    server = None
    active = None
    try:
        for c in cases:
            if time.monotonic() >= deadline or (args.limit and done >= args.limit):
                print(f'Stopped between episodes after {done} new cases; rerun to resume.', flush=True)
                break
            if completed_case(target_directory(out, c), c):
                continue
            identity = (c['model'], c['context'])
            if identity != active:
                stop_server(server)
                server = ensure_server(c, args.port, out / 'server' / c['model_label'])
                active = identity
            state, elapsed = run_case(out, c, args.port)
            done += state == 'done'
            print(f'{c["condition"]} {c["task"]} seed {c["seed"]}: {state}, {elapsed:.1f}s', flush=True)
    finally:
        stop_server(server)
    # Aggregation is a separate, read-only program; execution never owns statistics.
    if args.study.startswith('followup'):
        subprocess.run([sys.executable, str(ROOT / 'analysis/analyze_e2_e3_followup.py'),
                        '--input', str(out), '--output', str(out / 'analysis')], check=True)
    elif args.study.startswith('e1_'):
        subprocess.run([sys.executable, str(ROOT / 'analysis/summarize_e1_baselines.py'),
                        '--input', str(out), '--output', str(out / 'analysis')], check=True)
    print(f'Finished invocation: {done} new cases. Evidence: {out}')

if __name__ == '__main__':
    main()
