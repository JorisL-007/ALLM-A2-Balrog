"""After actor lock only: complete Baba seed 12 for two-seed power calibration.

These 12 episodes are outside the fixed screening panel and cannot select/tune
the actor. They use development seed 12, never any confirmation seed.
"""
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from run import append,budget_remaining,write,MODEL


def main():
    actor_lock=json.loads((HERE/'ACTOR_LOCK.json').read_text())
    actor=actor_lock['selected_actor']
    lock=(HERE/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=HERE/'STATE.json';state=json.loads(state_path.read_text())
    if state.get('active_evaluation'):raise RuntimeError('Unreconciled evaluation')
    if actor=='naive':raise RuntimeError('Calibration requires a distinct actor and Naive comparator')
    for name,digest in actor_lock['frozen_file_hashes'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Actor changed after lock')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8088/v1/models',timeout=5) as r:
        if [x['id'] for x in json.load(r)['data']]!=[MODEL]:raise RuntimeError('Wrong model')
    cases=[dict(c,seed=12,case_id=c['case_id'].rsplit(':',1)[0]+':12')
           for c in json.loads((HERE/'development_manifest.json').read_text()) if c['environment']=='babaisai']
    destination=HERE/'protocol/calibration';destination.mkdir(exist_ok=True)
    write(destination/'manifest.json',cases)
    if budget_remaining(state)<300 or state['limits']['episode_attempts']-state['episode_attempts']<12:
        raise RuntimeError('Insufficient budget for calibration; retain existing evidence')
    for case in cases:
        for variant in ('naive',actor):
            directory=destination/'runs'/variant/case['case_id'].replace('/','_').replace(':','__')
            if (directory/'result.json').exists():continue
            if directory.exists():raise RuntimeError('Incomplete calibration preserved; inspect before retry')
            directory.mkdir(parents=True)
            remaining=budget_remaining(state);started=time.time()
            if remaining<5 or state['episode_attempts']>=state['limits']['episode_attempts']:return
            state['episode_attempts']+=1
            cmd=[sys.executable,str(HERE/'episode.py'),'--variant',variant,'--directory',str(directory),
                 '--spec',json.dumps(case),'--deadline',str(started+remaining-1)]
            event=dict(event='episode_started',stage='variance_calibration',variant=variant,pass_id='calibration_12',
                       case=case,directory=str(directory),started_at_unix=started,setup=False)
            append(HERE/'ledger.jsonl',event)
            with (directory/'console.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_evaluation']=dict(pid=proc.pid,process_group=proc.pid,started_at_unix=started,
                                                directory=str(directory),command=cmd);write(state_path,state)
                try:rc=proc.wait(timeout=remaining)
                except BaseException:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    rc=-1
            elapsed=time.time()-started
            state['benchmark_seconds_used']+=elapsed;state['active_evaluation']=None
            state['updated_at_unix']=time.time();write(state_path,state)
            path=directory/'result.json'
            result=json.loads(path.read_text()) if rc==0 and path.exists() else dict(status='incomplete',returncode=rc)
            result['stage']='variance_calibration';write(path if rc==0 else directory/'incomplete.json',result)
            append(HERE/'ledger.jsonl',dict(event,event='episode_finished',elapsed_seconds=elapsed,result=result))
            print(f"Calibration {variant} {case['case_id']}: {result['status']} {result.get('progression')} ({elapsed:.1f}s)",flush=True)
            if result['status']=='incomplete':raise RuntimeError('Incomplete calibration preserved')
    # Use one decision pass per case to avoid treating repeats as independent.
    scores={}
    for variant in ('naive',actor):
        panel='reference_1' if variant=='naive' else 'screen_1'
        rows=[json.loads(p.read_text()) for p in (HERE/'runs'/panel/variant).glob('*/result.json')]
        rows += [json.loads(p.read_text()) for p in (destination/'runs'/variant).glob('*/result.json')]
        for seed in (11,12):
            means=[]
            for env,count in [('babyai',5),('crafter',1),('babaisai',6)]:
                values=[r['progression'] for r in rows if r['seed']==seed and r['environment']==env]
                if len(values)!=count:raise ValueError('Calibration coverage mismatch')
                means.append(statistics.mean(values))
            scores[variant,seed]=statistics.mean(means)
    differences=[scores[actor,s]-scores['naive',s] for s in (11,12)]
    sd=statistics.stdev(differences);effect=.10
    required=max(2,math.ceil(((1.96+.84)*sd/effect)**2))
    report=dict(actor=actor,primary_baseline='naive',seed_clusters=2,decision_passes_for_power=1,
        paired_seed_differences=differences,paired_cluster_sd=sd,target_effect=effect,
        normal_approx_clusters_for_80pct_power=required,available_reserved_seed_clusters=20,
        selected_confirmation_seed_clusters=20,approx_mde_at_20=(1.96+.84)*sd/math.sqrt(20),
        power_qualified=False,notes=['Two clusters cannot reliably estimate a population variance.',
        'Use all 20 reserved clusters as a bounded precision study; no guaranteed power or significance.',
        'Repeated decisions on the same layout are not independent layouts.',
        'Actor was locked before this calibration; these cases are excluded from screening/selection.'])
    write(destination/'power_report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
