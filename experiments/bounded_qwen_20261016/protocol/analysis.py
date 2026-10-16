"""Paired seed-cluster confirmation analysis. Never call on partial confirmation data."""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def seed_scores(rows):
    """Average fixed tasks within benchmark, equal-weight benchmarks, then repeats."""
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['seed'],r['variant'],r['pass_id'],r['environment']].append(r['progression'])
    by_repeat=defaultdict(list)
    for (seed,variant,rep,env),values in grouped.items():
        by_repeat[seed,variant,rep].append(statistics.mean(values))
    by_seed=defaultdict(list)
    for (seed,variant,rep),values in by_repeat.items():
        if len(values)!=3:raise ValueError('Missing benchmark in a seed cluster')
        by_seed[seed,variant].append(statistics.mean(values))
    return {k:statistics.mean(v) for k,v in by_seed.items()}


def paired_cluster_interval(differences, draws=10000):
    """Resample layouts, not episodes or repeated decisions on the same layout."""
    rng=random.Random(20261016)
    values=list(differences)
    if len(values)<2:raise ValueError('At least two independent seed clusters required')
    boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(draws))
    return dict(mean_pp=100*statistics.mean(values),lo_pp=100*boot[int(.025*draws)],
                hi_pp=100*boot[min(draws-1,int(.975*draws))],seed_clusters=len(values),
                cluster_sd_pp=100*statistics.stdev(values))


def required_clusters(sd, effect=.10):
    # Conventional normal approximation, planning aid only; pilot n is tiny.
    return max(2,math.ceil(((1.96+.84)*sd/effect)**2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--frozen',type=Path,default=Path(__file__).resolve().parents[1]/'FROZEN.json')
    a=p.parse_args()
    frozen=json.loads(a.frozen.read_text())
    manifest=json.loads((a.frozen.parent/'development_manifest.json').read_text())
    tasks={(c['environment'],c['task']) for c in manifest}
    expected={(e,t,s) for e,t in tasks for s in range(100,120)}
    expected_passes={f'confirmation_{i}' for i in range(1,frozen['confirmation_repetitions']+1)}
    groups=defaultdict(dict);rows=[]
    for path in sorted((a.output/'runs').glob('*/*/*/result.json')):
        r=json.loads(path.read_text())
        if r['status']!='complete':raise ValueError('Incomplete result')
        rep=path.parents[2].name
        if rep not in expected_passes or r['variant'] not in frozen['confirmation_variants']:
            raise ValueError('Unexpected repetition/variant')
        key=(r['environment'],r['task'],r['seed'])
        if key in groups[rep,r['variant']]:raise ValueError('Duplicate matched case')
        groups[rep,r['variant']][key]=r
        rows.append(dict(r,pass_id=rep))
    for rep in expected_passes:
        for variant in frozen['confirmation_variants']:
            if set(groups[rep,variant])!=expected:
                raise ValueError(f'Confirmation incomplete: {rep}/{variant}; no interim efficacy analysis permitted')
    anchor=groups['confirmation_1',frozen['confirmation_variants'][0]]
    for group in groups.values():
        for key,r in group.items():
            if r['initial_fingerprint']!=anchor[key]['initial_fingerprint']:
                raise ValueError('Initial observation mismatch')
    scores=seed_scores(rows)
    selected=frozen['selected_actor']
    primary=frozen['primary_baseline']
    reports=[]
    for baseline in frozen['confirmation_variants']:
        if baseline==selected:continue
        diffs=[scores[s,selected]-scores[s,baseline] for s in range(100,120)]
        reports.append(dict(selected=selected,baseline=baseline,primary=baseline==primary,
                            **paired_cluster_interval(diffs)))
    destination=a.output/'confirmation_analysis.json'
    destination.write_text(json.dumps(dict(comparisons=reports,complete_episodes=len(rows),
        notes=['Primary endpoint is predeclared; other baseline comparisons are descriptive.',
               'Percentile intervals resample shared seed clusters and include all repetitions.']),indent=2)+'\n')
    print(destination.read_text())


if __name__=='__main__':main()
