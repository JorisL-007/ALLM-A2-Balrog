"""Keep E3 paired pilot results separate from E2 screening and actor selection."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((HERE/'development_manifest.json').read_text())
    expected={c['case_id'] for c in manifest}
    groups=defaultdict(dict);paths={}
    for path in sorted((HERE/'e3/runs/pilot_1').glob('*/*/result.json')):
        d=json.loads(path.read_text())
        if d['status']!='complete':raise ValueError('Incomplete E3 case')
        if d['case_id'] in groups[d['variant']]:raise ValueError('Duplicate E3 case')
        groups[d['variant']][d['case_id']]=d
        paths[d['variant'],d['case_id']]=path.parent
    actor=json.loads((HERE/'FROZEN.json').read_text())['selected_actor']
    base,reflection=actor+'_frozen',actor+'_reflection'
    if set(groups[base])!=expected or set(groups[reflection])!=expected:
        raise ValueError('E3 comparison requires 18 complete cases in each arm')
    paired=[];metrics=[]
    for cid in sorted(expected):
        a,b=groups[base][cid],groups[reflection][cid]
        if a['initial_fingerprint']!=b['initial_fingerprint']:raise ValueError('E3 layout mismatch')
        paired.append(dict(case_id=cid,environment=a['environment'],baseline=a['progression'],reflection=b['progression'],
            delta_progression_pp=100*(b['progression']-a['progression']),delta_calls=b['calls']-a['calls'],
            delta_output_tokens=b['output_tokens']-a['output_tokens'],reflections=b['reflections']))
    for variant,cases in groups.items():
        means=[]
        for env in ('babyai','crafter','babaisai'):
            rows=[d for d in cases.values() if d['environment']==env]
            mean=statistics.mean(d['progression'] for d in rows)*100;means.append(mean)
            metrics.append(dict(variant=variant,benchmark=env,episodes=len(rows),progression_pct=mean,
                calls=sum(r['calls'] for r in rows),input_tokens=sum(r['input_tokens'] for r in rows),
                output_tokens=sum(r['output_tokens'] for r in rows),reflections=sum(r['reflections'] for r in rows),
                steps=sum(r['steps'] for r in rows),invalid_commands=sum(r['invalid_commands'] for r in rows),
                unchanged_text=sum(r['unchanged_text'] for r in rows),decision_seconds=sum(r['decision_seconds'] for r in rows)))
        metrics.append(dict(variant=variant,benchmark='overall',episodes=18,progression_pct=statistics.mean(means),
            **{k:sum(r[k] for r in metrics[-3:]) for k in ('calls','input_tokens','output_tokens','reflections','steps','invalid_commands','unchanged_text','decision_seconds')}))
    for filename,rows in [('paired_differences.csv',paired),('comparison.csv',metrics)]:
        with (HERE/'e3'/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    all_reflections=[]
    for (variant,cid),directory in paths.items():
        file=directory/'reflections.jsonl'
        if file.exists():
            for line in file.read_text().splitlines():all_reflections.append(dict(case_id=cid,**json.loads(line)))
    (HERE/'e3/reflection_events.json').write_text(json.dumps(all_reflections,indent=2)+'\n')
    for row in metrics:
        if row['benchmark']=='overall':print(row)
    print('Reflection-only input/output tokens:',sum(r['input_tokens'] for r in all_reflections),sum(r['output_tokens'] for r in all_reflections))


if __name__=='__main__':main()
