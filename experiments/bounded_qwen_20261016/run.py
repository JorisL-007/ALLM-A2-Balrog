#!/usr/bin/env python3
"""Serial, resumable budget controller. Confirmation requires a separate explicit mode."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-4B-Instruct-2507-FP8'
VARIANTS = ('naive', 'planning', 'grounded', 'mechanics', 'short', 'transition')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value)+'\n')


def budget_remaining(state, now=None):
    now = time.time() if now is None else now
    return max(0, min(state['benchmark_stop_unix']-now,
                      state['limits']['benchmark_seconds']-state['benchmark_seconds_used']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, required=True)
    p.add_argument('--pass-id', required=True)
    p.add_argument('--setup', action='store_true')
    p.add_argument('--case-ids', nargs='+')
    p.add_argument('--python', default='/home/joris/miniconda3/envs/balrog/bin/python')
    p.add_argument('--mode', choices=('development', 'confirmation'), default='development')
    p.add_argument('--output', type=Path, default=HERE)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    manifest = json.loads((HERE/'development_manifest.json').read_text())
    if a.mode == 'confirmation':
        if out == HERE:
            p.error('Confirmation must use a fresh --output directory')
        if not (HERE/'FROZEN.json').exists():
            p.error('Freeze an actor and protocol before confirmation')
        frozen = json.loads((HERE/'FROZEN.json').read_text())
        if a.variants != frozen['confirmation_variants']:
            p.error('Confirmation variants must match FROZEN.json order exactly')
        manifest = [dict(c, seed=s, case_id=f"{c['environment']}:{c['task']}:{s}")
                    for c in { (c['environment'],c['task']):c for c in manifest }.values()
                    for s in range(100,120)]
    elif any(100 <= c['seed'] <= 119 for c in manifest):
        p.error('Held-out seeds forbidden in development')
    if a.case_ids:
        manifest = [c for c in manifest if c['case_id'] in a.case_ids]
        if len(manifest) != len(set(a.case_ids)):
            p.error('Unknown or duplicate case ID')
    if a.setup:
        manifest = [next(c for c in manifest if c['environment']==e) for e in ('babyai','crafter','babaisai')]
    if a.dry_run:
        print(json.dumps(dict(mode=a.mode, variants=a.variants, cases=manifest, episodes=len(manifest)*len(a.variants)),indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = out/'STATE.json'
    if a.mode == 'confirmation' and not state_path.exists():
        # A future, separately authorized run; this does not change today's development budget.
        now=time.time()
        write(state_path, dict(started_at_unix=now, deadline_unix=now+86400,
            benchmark_stop_unix=now+85500, limits=dict(benchmark_seconds=82800,episode_attempts=2160,e2_candidates=6),
            benchmark_seconds_used=0,episode_attempts=0,active_processes=[],e2_candidates_started=[]))
    state = json.loads(state_path.read_text())
    if state.get('active_evaluation'):
        p.error('Unreconciled active evaluation: inspect ownership and account elapsed time before resuming')
    # No network hostname, proxy, external service, or model substitution is permitted.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8088/v1/models', timeout=5) as r:
        ids=[x['id'] for x in json.load(r)['data']]
    if ids != [MODEL]:
        p.error(f'Expected only {MODEL}, got {ids}')
    hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')}
    if (out/'run_hashes.json').exists() and json.loads((out/'run_hashes.json').read_text()) != hashes:
        p.error('Code changed since first run; create a versioned experiment directory')
    write(out/'run_hashes.json', hashes)
    for variant in a.variants:
        if variant not in ('naive','planning','grounded') and variant not in state['e2_candidates_started']:
            if len(state['e2_candidates_started']) >= state['limits']['e2_candidates']:
                p.error('Candidate limit reached')
            state['e2_candidates_started'].append(variant)
            write(state_path,state)
        for case in manifest:
            directory = out/'runs'/a.pass_id/variant/case['case_id'].replace('/','_').replace(':','__')
            if (directory/'result.json').exists():
                continue
            # Never overwrite failed or interrupted output, even when retrying the same case.
            if directory.exists():
                directory = directory.with_name(directory.name+'__retry_'+str(time.time_ns()))
            remaining=budget_remaining(state)
            if remaining < 5 or state['episode_attempts'] >= state['limits']['episode_attempts']:
                print('Resource limit reached',flush=True)
                return
            directory.mkdir(parents=True)
            started=time.time()
            state['episode_attempts'] += 1
            timeout=min(remaining, 120 if a.setup else remaining)
            cmd=[a.python,str(HERE/'episode.py'),'--spec',json.dumps(case),'--variant',variant,
                 '--directory',str(directory),'--deadline',str(started+timeout-1)]
            if a.setup: cmd += ['--setup-steps','5']
            entry=dict(event='episode_started',variant=variant,pass_id=a.pass_id,case=case,
                       directory=str(directory),started_at_unix=started,setup=a.setup)
            append(out/'ledger.jsonl',entry)
            with (directory/'console.log').open('w') as log:
                proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,start_new_session=True)
                state['active_evaluation']=dict(pid=proc.pid,process_group=proc.pid,started_at_unix=started,
                                                directory=str(directory),command=cmd)
                write(state_path,state)
                try:
                    rc=proc.wait(timeout=timeout)
                except BaseException:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    rc=-1
                elapsed=time.time()-started
            state['benchmark_seconds_used'] += elapsed
            state['active_evaluation']=None
            state['updated_at_unix']=time.time()
            write(state_path,state)
            result_path=directory/'result.json'
            result=json.loads(result_path.read_text()) if rc==0 and result_path.exists() else dict(status='incomplete',returncode=rc)
            append(out/'ledger.jsonl',dict(entry,event='episode_finished',elapsed_seconds=elapsed,result=result))
            print(f"{a.pass_id} {variant} {case['case_id']}: {result['status']} progression={result.get('progression')} ({elapsed:.1f}s)",flush=True)
            if result['status']=='incomplete':
                write(directory/'incomplete.json',dict(result,case=case,variant=variant,elapsed_seconds=elapsed))
                sys.exit('Incomplete attempt preserved; inspect before continuing')


if __name__ == '__main__':
    main()
