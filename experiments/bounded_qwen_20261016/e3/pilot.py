"""One paired full-panel E3 pilot, charged to the original session budget."""
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

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from run import append,budget_remaining,write,MODEL


def main():
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    actor=frozen['selected_actor']
    lock=(HERE/'evaluation.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=HERE/'STATE.json';state=json.loads(state_path.read_text())
    if state.get('active_evaluation'):raise RuntimeError('Unreconciled active evaluation')
    manifest=json.loads((HERE/'development_manifest.json').read_text())
    if any(c['seed'] not in (11,12) for c in manifest):raise ValueError('Held-out case forbidden')
    # Require enough observed actor cost plus a generous reflection/variance margin.
    estimate=frozen['estimated_panel_seconds']*2.5
    if budget_remaining(state)<estimate or state['limits']['episode_attempts']-state['episode_attempts']<36:
        raise RuntimeError(f'Pilot deferred: need {estimate:.1f}s and 36 attempts remaining')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8088/v1/models',timeout=5) as r:
        if [x['id'] for x in json.load(r)['data']]!=[MODEL]:raise RuntimeError('Wrong model')
    config_id='stuck3_cooldown12_max2_tokens96'
    if config_id not in state['e3_candidates_started']:
        if len(state['e3_candidates_started'])>=state['limits']['e3_candidates']:raise RuntimeError('E3 candidate limit')
        state['e3_candidates_started'].append(config_id);write(state_path,state)
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list((HERE/'e3').glob('*.py'))+list(HERE.glob('*.py'))+[HERE/'FROZEN.json']}
    hash_path=HERE/'e3/hashes.json'
    if hash_path.exists() and json.loads(hash_path.read_text())!=hashes:raise RuntimeError('Pilot code changed')
    write(hash_path,hashes)
    for case in manifest:
        for reflection in (False,True):
            variant=actor+('_reflection' if reflection else '_frozen')
            directory=HERE/'e3/runs/pilot_1'/variant/case['case_id'].replace('/','_').replace(':','__')
            if (directory/'result.json').exists():continue
            if directory.exists():directory=directory.with_name(directory.name+'__retry_'+str(time.time_ns()))
            remaining=budget_remaining(state)
            if remaining<5 or state['episode_attempts']>=state['limits']['episode_attempts']:return
            directory.mkdir(parents=True)
            started=time.time()
            state['episode_attempts']+=1
            cmd=[sys.executable,str(HERE/'e3/worker.py'),'--directory',str(directory),'--spec',json.dumps(case),
                 '--deadline',str(started+remaining-1)]
            if reflection:cmd+=['--reflection']
            event=dict(stage='E3',event='episode_started',variant=variant,pass_id='pilot_1',case=case,
                       directory=str(directory),started_at_unix=started,setup=False)
            append(HERE/'ledger.jsonl',event)
            with (directory/'console.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_evaluation']=dict(pid=proc.pid,process_group=proc.pid,started_at_unix=started,
                                                directory=str(directory),command=cmd)
                write(state_path,state)
                try:rc=proc.wait(timeout=remaining)
                except BaseException:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    rc=-1
            elapsed=time.time()-started
            state['benchmark_seconds_used']+=elapsed
            state['active_evaluation']=None;state['updated_at_unix']=time.time();write(state_path,state)
            path=directory/'result.json'
            result=json.loads(path.read_text()) if rc==0 and path.exists() else dict(status='incomplete',returncode=rc)
            append(HERE/'ledger.jsonl',dict(event,event='episode_finished',elapsed_seconds=elapsed,result=result))
            print(f"E3 {variant} {case['case_id']}: {result['status']} {result.get('progression')} ({elapsed:.1f}s)",flush=True)
            if result['status']=='incomplete':
                write(directory/'incomplete.json',result)
                sys.exit('Incomplete E3 attempt preserved')


if __name__=='__main__':main()
