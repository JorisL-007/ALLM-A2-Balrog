"""Summarize BALROG's saved episode JSON and CSV files."""

import csv
import json
import statistics
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("tags", nargs="*")
parser.add_argument("--stage", choices=("development", "evaluation"), default="development")
parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_results")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.input.resolve()
output = args.output.resolve()
if root == output:
    parser.error("Use separate input and output directories")
output.mkdir(parents=True, exist_ok=True)
manifest_path = root / "manifest.json"
new_cases = json.loads(manifest_path.read_text()).get("cases", []) if manifest_path.exists() else []
completions = [p for p in root.rglob("complete.json") if not any(".incomplete_" in x or ".working" in x for x in p.parts)]
new_groups = {}
for completion in completions:
    meta = json.loads(completion.read_text())
    new_groups.setdefault(meta["condition"], []).append((completion.parent, meta))
all_tags = sorted(new_groups) if new_cases else sorted(p.parent.name for p in root.glob("*/settings.txt"))
tags = args.tags or [tag for tag in all_tags if (tag.startswith(("e2_", "e3_", "cot_"))) == (args.stage == "development")]
episode_rows = []
matrix_rows = []

for tag in tags:
    folder = root / tag
    if new_cases:
        settings = next(c for c in new_cases if c["condition"] == tag)
        group = new_groups.get(tag, [])
        elapsed = sum(meta["elapsed_seconds"] for _, meta in group)
        files = sorted(p for directory, _ in group for p in directory.rglob("*_run_*.json"))
    else:
        settings = dict(line.strip().split("=", 1) for line in (folder / "settings.txt").read_text().splitlines())
        elapsed = sum(float(row["elapsed_seconds"]) for row in csv.DictReader((folder / "timing.csv").open()))
        files = sorted(folder.glob("seed_*/**/*_run_*.json"))
    for path in files:
        data = json.loads(path.read_text())
        steps = list(csv.DictReader(path.with_suffix(".csv").open()))
        seed = data["seed"]
        reflections = sum(1 for row in steps if row["Reasoning"].startswith("REFLECTION:"))
        episode_rows.append({
            "condition": tag,
            "model": settings["model"],
            "environment": settings.get("environment", "babyai"),
            "task": data["task"].split("/")[-1],
            "seed": seed,
            "progress": data["progression"],
            "steps": data["num_steps"],
            "input_tokens": data["input_tokens"],
            "output_tokens": data["output_tokens"],
            "fallback_actions": (
                sum(row["Action"] != row["Executed Action"] for row in steps)
                if "Executed Action" in steps[0] else len(data["failed_candidates"])
            ),
            "reflections": reflections,
            "trajectory": os.path.relpath(path.with_suffix(".csv"), Path(__file__).resolve().parents[1]),
        })

    for task in sorted({row["task"] for row in episode_rows if row["condition"] == tag} | {"ALL"}):
        rows = [row for row in episode_rows if row["condition"] == tag and (task == "ALL" or row["task"] == task)]
        if not rows:
            continue
        progress = [row["progress"] for row in rows]
        matrix_rows.append({
            "condition": tag,
            "model": settings["model"],
            "environment": settings.get("environment", "babyai"),
            "task": task,
            "episodes": len(rows),
            "mean_progress_pct": round(100 * statistics.mean(progress), 2),
            "sd_progress_pct": round(100 * statistics.stdev(progress), 2) if len(progress) > 1 else 0,
            "successes": sum(value == 1 for value in progress),
            "mean_steps": round(statistics.mean(row["steps"] for row in rows), 2),
            "mean_input_tokens": round(statistics.mean(row["input_tokens"] for row in rows), 1),
            "mean_output_tokens": round(statistics.mean(row["output_tokens"] for row in rows), 1),
            "fallback_actions": sum(row["fallback_actions"] for row in rows),
            "reflections": sum(row["reflections"] for row in rows),
            "panel_wall_seconds": round(elapsed, 1),
        })

for name, rows in (("episodes.csv", episode_rows), ("performance_matrix.csv", matrix_rows)):
    if not rows:
        continue
    with (output / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print(output / "performance_matrix.csv")
