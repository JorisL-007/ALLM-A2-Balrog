"""Rebuild complete-panel metrics and strictly matched paired differences."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]


def csv_write(path,rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze():
    manifest=json.loads((HERE/'development_manifest.json').read_text())
    expected={c['case_id'] for c in manifest}
    execution=defaultdict(float)
    for line in (HERE/'ledger.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['event']=='episode_finished' and not event.get('setup') and event.get('stage')!='E3':
            execution[event['pass_id'],event['variant'],event['case']['case_id']]+=event['elapsed_seconds']
    groups=defaultdict(dict);episodes=[]
    for path in sorted((HERE/'runs').glob('*/*/*/result.json')):
        d=json.loads(path.read_text())
        if d['status']!='complete':continue
        repetition=path.parents[2].name
        d['execution_seconds']=execution[repetition,d['variant'],d['case_id']]
        key=(repetition,d['variant'])
        if d['case_id'] in groups[key]:raise ValueError('Duplicate completed case '+str(key))
        groups[key][d['case_id']]=d
        episodes.append(dict(pass_id=repetition,**d,path=str(path.relative_to(HERE))))
    csv_write(HERE/'per_episode.csv',episodes)
    metrics=[];complete={}
    for (rep,variant),cases in groups.items():
        if set(cases)!=expected:
            print('PARTIAL',rep,variant,len(cases),'/',len(expected));continue
        complete[rep,variant]=cases
        means=[]
        for environment in ('babyai','crafter','babaisai'):
            rows=[d for d in cases.values() if d['environment']==environment]
            mean=statistics.mean(d['progression'] for d in rows)*100;means.append(mean)
            metrics.append(dict(pass_id=rep,variant=variant,benchmark=environment,episodes=len(rows),progression_pct=mean,
                calls=sum(d['calls'] for d in rows),input_tokens=sum(d['input_tokens'] for d in rows),
                output_tokens=sum(d['output_tokens'] for d in rows),steps=sum(d['steps'] for d in rows),
                invalid_commands=sum(d['invalid_commands'] for d in rows),unchanged_text=sum(d['unchanged_text'] for d in rows),
                repeated_actions=sum(d['repeated_actions'] for d in rows),decision_seconds=sum(d['decision_seconds'] for d in rows),
                execution_seconds=sum(d['execution_seconds'] for d in rows)))
        metrics.append(dict(pass_id=rep,variant=variant,benchmark='overall',episodes=18,progression_pct=statistics.mean(means),
            **{k:sum(row[k] for row in metrics[-3:]) for k in ('calls','input_tokens','output_tokens','steps','invalid_commands','unchanged_text','repeated_actions','decision_seconds','execution_seconds')}))
    csv_write(HERE/'comparison.csv',metrics)
    task_rows=[]
    for (rep,variant),cases in complete.items():
        tasks=defaultdict(list)
        for d in cases.values():
            if d['environment']=='babyai':tasks[d['task']].append(d['progression'])
        task_rows.extend(dict(pass_id=rep,variant=variant,task=t,episodes=len(v),success_pct=statistics.mean(v)*100) for t,v in tasks.items())
    csv_write(HERE/'babyai_tasks.csv',task_rows)
    paired=[]
    for (rep,variant),cases in complete.items():
        for baseline in ('naive','planning','grounded'):
            base_key=(rep,baseline) if (rep,baseline) in complete else ('reference_1',baseline)
            base=complete.get(base_key)
            if base is None or variant==baseline:continue
            for cid in sorted(expected):
                x,y=cases[cid],base[cid]
                if x['initial_fingerprint']!=y['initial_fingerprint']:
                    raise ValueError(f'Layout mismatch: {variant} / {baseline}: {cid}')
                paired.append(dict(pass_id=rep,variant=variant,baseline=baseline,baseline_pass_id=base_key[0],case_id=cid,environment=x['environment'],
                    delta_progression_pp=100*(x['progression']-y['progression']),
                    delta_calls=x['calls']-y['calls'],delta_output_tokens=x['output_tokens']-y['output_tokens'],
                    delta_execution_seconds=x['execution_seconds']-y['execution_seconds']))
    csv_write(HERE/'paired_differences.csv',paired)
    paired_groups=defaultdict(list)
    for row in paired:
        paired_groups[row['pass_id'],row['variant'],row['baseline'],row['environment']].append(row)
    paired_summary=[]
    for (rep,variant,baseline,env),rows in sorted(paired_groups.items()):
        paired_summary.append(dict(pass_id=rep,variant=variant,baseline=baseline,baseline_pass_id=rows[0]['baseline_pass_id'],benchmark=env,
            matched_cases=len(rows),delta_progression_pp=statistics.mean(r['delta_progression_pp'] for r in rows),
            delta_calls=sum(r['delta_calls'] for r in rows),delta_output_tokens=sum(r['delta_output_tokens'] for r in rows),
            delta_execution_seconds=sum(r['delta_execution_seconds'] for r in rows)))
    for rep,variant,baseline in sorted({(r['pass_id'],r['variant'],r['baseline']) for r in paired_summary}):
        rows=[r for r in paired_summary if (r['pass_id'],r['variant'],r['baseline'])==(rep,variant,baseline)]
        if len(rows)!=3:raise ValueError('Incomplete paired benchmark coverage')
        paired_summary.append(dict(pass_id=rep,variant=variant,baseline=baseline,baseline_pass_id=rows[0]['baseline_pass_id'],benchmark='overall',matched_cases=18,
            delta_progression_pp=statistics.mean(r['delta_progression_pp'] for r in rows),
            delta_calls=sum(r['delta_calls'] for r in rows),delta_output_tokens=sum(r['delta_output_tokens'] for r in rows),
            delta_execution_seconds=sum(r['delta_execution_seconds'] for r in rows)))
    csv_write(HERE/'paired_summary.csv',paired_summary)
    repeated=defaultdict(list)
    for row in metrics:repeated[row['variant'],row['benchmark']].append(row)
    csv_write(HERE/'repeated_panel_summary.csv',[
        dict(variant=v,benchmark=b,repetitions=len(rs),episodes_per_repetition=18 if b=='overall' else rs[0]['episodes'],
             mean_progression_pct=statistics.mean(r['progression_pct'] for r in rs),
             sd_across_repeated_panels_pct=statistics.stdev(r['progression_pct'] for r in rs) if len(rs)>1 else '',
             total_calls=sum(r['calls'] for r in rs),total_output_tokens=sum(r['output_tokens'] for r in rs),
             total_execution_seconds=sum(r['execution_seconds'] for r in rs))
        for (v,b),rs in sorted(repeated.items())])
    print('Complete panels:',list(complete))
    for row in metrics:
        if row['benchmark']=='overall':print(row)


if __name__=='__main__':analyze()
