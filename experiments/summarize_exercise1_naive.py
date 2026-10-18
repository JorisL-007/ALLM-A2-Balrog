"""Summarize the completed Exercise 1 naive runs for all three local models."""

import csv
import json
import statistics
from pathlib import Path


root = Path(__file__).resolve().parents[1] / "exercise1_results/20261015_072525"
models = ("qwen4-instruct", "qwen14-awq", "phi4-instruct")
benchmarks = ("babyai", "crafter", "babaisai")

with (root / "performance_matrix_all_models.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("model", "agent", "benchmark", "runs", "episodes_per_run",
                     "mean_progress_pct", "sd_progress_pct"))
    for model in models:
        summaries = [
            json.loads((root / model / "naive" / f"run_{run}" / "summary.json").read_text())
            for run in (1, 2, 3)
        ]
        for benchmark in benchmarks:
            results = [summary["environments"][benchmark] for summary in summaries]
            scores = [result["progression_percentage"] for result in results]
            writer.writerow((model, "naive", benchmark, len(scores),
                             results[0]["episodes_played"], statistics.mean(scores),
                             statistics.stdev(scores)))
