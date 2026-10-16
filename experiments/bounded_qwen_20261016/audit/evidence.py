"""Deterministic trajectory review: every arm/task, first success/failure, all regressions.

Flags describe observable mechanics, not automatic judgments of plan correctness.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]


def main():
    results=[];summaries=[];examples=[];groups=defaultdict(list)
    for path in sorted((HERE/'runs').glob('*/*/*/result.json')):
        r=json.loads(path.read_text())
        if r['status']!='complete':continue
        r['pass_id']=path.parents[2].name
        r['directory']=str(path.parent.relative_to(HERE))
        rows=[json.loads(s) for s in (path.parent/'steps.jsonl').read_text().splitlines()]
        flags=defaultdict(list)
        for d in rows:
            text=d['observation']['long_term_context']
            if re.search(r'^a (?:\w+ )?(?:key|ball|box) 1 step forward$',text,re.M) and d['executed_action']=='go forward':
                flags['forward_into_pickup_object'].append(d)
            if 'You face tree at your front.' in text and d['executed_action'].startswith('Move'):
                flags['movement_while_facing_tree'].append(d)
            if d['text_unchanged'] and d['repeated_action']:
                flags['repeated_unchanged_text'].append(d)
        for kind,ds in flags.items():
            examples.append(dict(pass_id=r['pass_id'],variant=r['variant'],case_id=r['case_id'],kind=kind,
                count=len(ds),directory=r['directory'],first_example=ds[0]))
        raw=list((path.parent/'agent_raw').glob('*.jsonl'))
        rejects=fallbacks=0
        for f in raw:
            for line in f.open():
                d=json.loads(line)
                rejects+=bool(d['calls'][0]['violation'])
                fallbacks+=d['agent_fallback']
        summaries.append(dict(pass_id=r['pass_id'],variant=r['variant'],case_id=r['case_id'],
            forward_into_pickup_object=len(flags['forward_into_pickup_object']),
            movement_while_facing_tree=len(flags['movement_while_facing_tree']),
            repeated_unchanged_text=len(flags['repeated_unchanged_text']),
            initial_check_rejections=rejects,agent_fallbacks=fallbacks,extra_calls=r['calls']-r['steps']))
        groups[r['pass_id'],r['variant'],r['environment'],r['task'],r['progression']>0].append(r)
        results.append(r)
    # First sorted case within every arm/task/outcome category. No hand-picked trajectories.
    review=[dict(category=list(k),episode=sorted(v,key=lambda r:r['case_id'])[0]) for k,v in sorted(groups.items())]
    base={r['case_id']:r for r in results if r['variant']=='planning' and r['pass_id']=='reference_1'}
    regressions=[dict(variant=r['variant'],pass_id=r['pass_id'],case_id=r['case_id'],
        candidate=r['progression'],planning=base[r['case_id']]['progression'],directory=r['directory'])
        for r in results if r['variant'] not in ('naive','planning','grounded') and r['case_id'] in base and r['progression']<base[r['case_id']]['progression']]
    for name,value in [('mechanics_metrics',summaries),('mechanics_examples',examples),('systematic_review',review),('regressions',regressions)]:
        (HERE/'audit'/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print('Reviewed',len(results),'episodes;',len(review),'task/outcome representatives;',len(regressions),'regressions')


if __name__=='__main__':main()
