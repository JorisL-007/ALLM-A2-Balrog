#!/usr/bin/env python3
"""Summarize completed panels, optionally alongside revised Exercise 1 results."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

ENVIRONMENTS = ('babyai', 'crafter', 'babaisai')


def episode_records(folder):
    return [json.loads(p.read_text()) for p in folder.rglob('*_run_*.json')]


def signature(folder):
    return collections.Counter((d['task'], d['seed']) for d in episode_records(folder))


def summarize(folder, environment):
    episodes = episode_records(folder / environment)
    if not episodes:
        raise ValueError(f'No completed episodes in {folder / environment}')
    raw = []
    for path in (folder / environment).rglob('raw_responses/*.jsonl'):
        raw.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    steps = sum(d['num_steps'] for d in episodes)
    result = {
        'episodes': len(episodes), 'steps': steps,
        'progress_pct': 100 * statistics.mean(d['progression'] for d in episodes),
        'input_tokens': sum(d['input_tokens'] for d in episodes),
        'output_tokens': sum(d['output_tokens'] for d in episodes),
        'balrog_fallback_pct': 100 * sum(len(d['failed_candidates']) for d in episodes) / steps if steps else 0,
    }
    # The initial violation statistic is evaluated for both variants, including
    # planning-only, so it also exposes the uncorrected candidate quality.
    result.update({
        'llm_calls': sum(len(r['calls']) for r in raw) if raw else None,
        'retry_pct': 100 * sum(len(r['calls']) > 1 for r in raw) / len(raw) if raw else None,
        'initial_check_rejection_pct': 100 * sum(r['calls'][0]['violation'] is not None for r in raw) / len(raw) if raw else None,
        'agent_fallback_pct': 100 * sum(r['agent_fallback'] for r in raw) / len(raw) if raw else None,
        'decision_seconds': sum(r['elapsed_seconds'] for r in raw) if raw else None,
    })
    if raw and len(raw) != steps:
        raise ValueError(f'Raw log/episode step count mismatch in {folder / environment}: {len(raw)} vs {steps}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--baseline-dir', type=Path)
    args = parser.parse_args()
    root = args.results.resolve()
    sources = [(root, 'planning'), (root, 'grounded')]
    if args.baseline_dir:
        sources = [(args.baseline_dir.resolve(), v) for v in ('naive', 'cot')] + sources
    rows = []
    reference = {}
    for source, variant in sources:
        parent = source / 'qwen4-instruct' / variant
        panels = sorted(parent.glob('run_*'))
        completed = [p for p in panels if (p / '.complete').exists()]
        if not completed:
            raise ValueError(f'No completed panels for {variant} under {parent}')
        for folder in completed:
            run = int(folder.name.split('_')[1])
            sig = signature(folder)
            if run in reference and reference[run] != sig:
                raise ValueError(f'Seed/task panels differ for {variant}, run {run}; compare matched panels only')
            reference[run] = sig
            environment_rows = []
            for env in ENVIRONMENTS:
                record = dict(model='qwen4-instruct', variant=variant, run=run, benchmark=env,
                              **summarize(folder, env))
                rows.append(record)
                environment_rows.append(record)
            overall = dict(model='qwen4-instruct', variant=variant, run=run, benchmark='overall')
            # Progress is an equal-weight mean across environments, not episodes.
            overall['progress_pct'] = statistics.mean(r['progress_pct'] for r in environment_rows)
            for name in ('episodes', 'steps', 'input_tokens', 'output_tokens', 'llm_calls', 'decision_seconds'):
                values = [r[name] for r in environment_rows]
                overall[name] = sum(values) if all(v is not None for v in values) else None
            for name in ('balrog_fallback_pct', 'retry_pct', 'initial_check_rejection_pct', 'agent_fallback_pct'):
                values = [r[name] for r in environment_rows]
                overall[name] = sum(r[name] * r['steps'] for r in environment_rows) / overall['steps'] if all(v is not None for v in values) and overall['steps'] else None
            # Preserve consistent column ordering with the environment rows.
            rows.append({k: overall[k] for k in rows[0]})
    with (root / 'per_run_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'performance_matrix.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'variant', 'benchmark', 'runs', 'episodes_per_run',
                         'mean_progress_pct', 'sd_progress_pct'])
        for _, variant in sources:
            for env in (*ENVIRONMENTS, 'overall'):
                group = [r for r in rows if r['variant'] == variant and r['benchmark'] == env]
                scores = [r['progress_pct'] for r in group]
                sd = statistics.stdev(scores) if len(scores) > 1 else None
                writer.writerow(['qwen4-instruct', variant, env, len(scores), group[0]['episodes'], statistics.mean(scores), sd])
                spread = f'{sd:.2f}' if sd is not None else 'n/a'
                print(f'{variant:9s} {env:8s}: {statistics.mean(scores):6.2f}% (SD {spread}; n={len(scores)})')
    print(f'CSV files saved in {root}')


if __name__ == '__main__':
    main()
