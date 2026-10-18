"""Read E1 results only: repeated-run means, sample SDs, and explicit completeness."""
import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=ROOT / 'exercise1_results/20261015_072525')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root, output = args.input.resolve(), args.output.resolve()
if root == output:
    parser.error('Use separate input and output directories')
output.mkdir(parents=True, exist_ok=True)
models = ('qwen4-instruct', 'qwen14-awq', 'phi4-instruct')
benchmarks = ('babyai', 'crafter', 'babaisai')
expected = {'babyai': 50, 'crafter': 10, 'babaisai': 120}
matrix, statuses = [], []
for model in models:
    for agent in ('naive', 'cot'):
        completed = []
        for run in (1, 2, 3):
            folder = root / model / agent / f'run_{run}'
            if not folder.exists():
                if (root / model / agent).exists():
                    statuses.append([model, agent, run, 0, 'not run', str(folder)])
                continue
            # Ignore retained incomplete attempts in the refactored namespace.
            files = [p for p in folder.rglob('*_run_*.json') if not any('.working' in x or '.incomplete_' in x for x in p.parts)]
            data = [json.loads(p.read_text()) for p in files]
            counts = {env: sum(env in p.relative_to(folder).parts for p in files) for env in benchmarks}
            complete = counts == expected
            statuses.append([model, agent, run, len(files), 'complete' if complete else 'incomplete', str(folder)])
            if not complete:
                continue
            summary_path = folder / 'summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())['environments']
                completed.append({env: summary[env]['progression_percentage'] for env in benchmarks})
            else:
                completed.append({env: 100 * statistics.mean(d['progression'] for p,d in zip(files,data)
                                                             if env in p.relative_to(folder).parts) for env in benchmarks})
        if completed:
            for env in benchmarks:
                scores = [run[env] for run in completed]
                matrix.append([model, agent, env, len(scores), expected[env], statistics.mean(scores),
                               statistics.stdev(scores) if len(scores) > 1 else ''])
for name, fields, rows in [
    ('performance_matrix_all_models.csv', ('model','agent','benchmark','runs','episodes_per_run','mean_progress_pct','sd_progress_pct'), matrix),
    ('run_status.csv', ('model','agent','run','completed_episodes','status','source'), statuses)]:
    with (output / name).open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)
print(output / 'performance_matrix_all_models.csv')
