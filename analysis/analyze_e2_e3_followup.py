"""Summarize completed follow-up episodes and paired BabyAI comparisons."""

import csv
import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exercises.common.results import result_files

ROOT = Path(__file__).resolve().parents[1]


INPUT = ROOT / "experiment_results/followup_v1"
OUT = ROOT / "experiment_results/refactored_v1/analysis/followup"


def read_episodes():
    episodes = []
    for completion in INPUT.rglob("complete.json"):
        if any(".working" in x or ".incomplete_" in x for x in completion.parts):
            continue
        folder = completion.parent
        meta = json.loads(completion.read_text())
        meta.setdefault("panel", "pilot" if 20 <= meta["seed"] <= 24 else "preflight")
        files = result_files(folder, meta["environment"], meta["task"])
        if not files:
            continue
        result = json.loads(files[0].read_text())
        with files[1].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        calls = [json.loads(line) for line in (folder / "llm_calls.jsonl").read_text().splitlines()]
        stalled = sum(
            rows[i]["Observation"] == rows[i - 1]["Observation"]
            and rows[i]["Executed Action"] == rows[i - 1]["Executed Action"]
            and rows[i]["Executed Action"] in ("go forward", "toggle", "pick up")
            for i in range(1, len(rows))
        )
        turn_loop = sum(
            [r["Executed Action"] for r in rows[i:i + 4]] in
            (["turn left", "turn right"] * 2, ["turn right", "turn left"] * 2)
            and rows[i]["Observation"] == rows[i + 2]["Observation"]
            and rows[i + 1]["Observation"] == rows[i + 3]["Observation"]
            for i in range(max(0, len(rows) - 3))
        )
        episodes.append({**meta,
            "success": int(result.get("progression", 0) == 1) if meta["environment"] == "babyai" else "",
            "progression": result.get("progression", 0),
            "invalid_proposals": sum(r["Action"] != r["Executed Action"] for r in rows),
            "interventions": sum(bool(r["Override Reason"]) for r in rows),
            "stalled_pairs": stalled, "turn_loops": turn_loop,
            "llm_calls": len(calls), "reflection_calls": sum(c["role"] == "reflection" for c in calls),
            "trajectory": str(Path(os.path.relpath(files[1], ROOT))),
            "raw_calls": str(Path(os.path.relpath(folder / "llm_calls.jsonl", ROOT))),
        })
    return episodes


def mean(values):
    return sum(values) / len(values) if values else 0


def quantile(values, fraction):
    values = sorted(values)
    return values[int(fraction * (len(values) - 1))]


def paired_test(episodes, panel, first, second, tasks, seeds):
    index = {(e["environment"], e["task"], e["seed"], e["condition"], e["repeat"]): e
             for e in episodes if e["panel"] == "primary"}
    blocks = []
    gains = losses = ties = 0
    for seed in seeds:
        deltas = []
        for task in tasks:
            a = index.get(("babyai", task, seed, first, 0))
            b = index.get(("babyai", task, seed, second, 0))
            if a is None or b is None:
                continue
            delta = b["success"] - a["success"]
            deltas.append(delta)
        if len(deltas) == len(tasks):
            blocks.append(mean(deltas))
            gains += sum(delta == 1 for delta in deltas)
            losses += sum(delta == -1 for delta in deltas)
            ties += sum(delta == 0 for delta in deltas)
    if not blocks:
        return None
    rng = random.Random(0)
    samples = [mean([rng.choice(blocks) for _ in blocks]) for _ in range(10000)]
    n = gains + losses
    exact_p = min(1, 2 * sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n) if n else 1
    observed = abs(mean(blocks))
    permuted = sum(abs(mean([block * rng.choice((-1, 1)) for block in blocks])) >= observed - 1e-12
                   for _ in range(100000))
    permutation_p = (permuted + 1) / 100001
    return {"panel": panel, "first": first, "second": second, "tasks": "+".join(tasks),
            "paired_seeds": len(blocks), "difference": mean(blocks),
            "ci_low": quantile(samples, .025), "ci_high": quantile(samples, .975),
            "gains": gains, "losses": losses, "ties": ties,
            "exact_mcnemar_p": exact_p if len(tasks) == 1 else "",
            "seed_block_permutation_p": permutation_p,
            "test_p": exact_p if len(tasks) == 1 else permutation_p}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT, INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    INPUT = args.input.resolve()
    OUT = args.output.resolve()
    if INPUT == OUT:
        parser.error("Use a separate output directory to preserve saved evidence")
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = read_episodes()
    write_csv(OUT / "episodes.csv", episodes)
    grouped = defaultdict(list)
    for episode in episodes:
        if episode["repeat"] == 0:
            grouped[(episode["panel"], episode["environment"], episode["task"], episode["condition"])].append(episode)
    summary = []
    for (panel, environment, task, condition), group in sorted(grouped.items()):
        summary.append({"panel": panel, "environment": environment, "task": task, "condition": condition,
                        "n": len(group), "mean_progression": mean([e["progression"] for e in group]),
                        "sd_progression": statistics.stdev([e["progression"] for e in group]) if len(group) > 1 else 0,
                        "successes": sum(e["success"] for e in group if e["success"] != ""),
                        "mean_steps": mean([e["num_steps"] for e in group]),
                        "sd_steps": statistics.stdev([e["num_steps"] for e in group]) if len(group) > 1 else 0,
                        "stalled_pairs": sum(e["stalled_pairs"] for e in group),
                        "turn_loops": sum(e["turn_loops"] for e in group),
                        "invalid_proposals": sum(e["invalid_proposals"] for e in group),
                        "interventions": sum(e["interventions"] for e in group),
                        "input_tokens": sum(e["input_tokens"] for e in group),
                        "output_tokens": sum(e["output_tokens"] for e in group),
                        "llm_calls": sum(e["llm_calls"] for e in group),
                        "reflection_calls": sum(e["reflection_calls"] for e in group),
                        "elapsed_seconds": sum(e["elapsed_seconds"] for e in group)})
    write_csv(OUT / "performance_matrix.csv", summary)
    comparisons = [
        ("primary", "plan", "guide", ["pickup"], range(30, 90)),
        ("secondary", "guide", "guide_gate", ["pickup"], range(30, 90)),
        ("secondary", "guide", "guide_goal", ["goto", "pickup", "open"], range(30, 90)),
        ("secondary", "guide", "guide_h4", ["goto", "pickup", "open"], range(30, 90)),
        ("secondary", "guide", "guide_h32", ["goto", "pickup", "open"], range(30, 90)),
    ]
    paired = [x for request in comparisons if (x := paired_test(episodes, *request))]
    secondary = sorted((x for x in paired if x["panel"] == "secondary"), key=lambda x: x["test_p"])
    for rank, row in enumerate(secondary):
        row["holm_p"] = min(1, max(x["test_p"] * (len(secondary) - i)
                                   for i, x in enumerate(secondary[:rank + 1])))
    for row in paired:
        row.setdefault("holm_p", "")
    write_csv(OUT / "paired_comparisons.csv", paired)

    repeated = defaultdict(list)
    for episode in episodes:
        if (episode["panel"] in ("primary", "repeat") and 30 <= episode["seed"] <= 34
                and episode["task"] in ("goto", "pickup")
                and episode["condition"] in ("plan", "guide_gate")):
            repeated[(episode["task"], episode["seed"], episode["condition"])].append(episode)
    repeat_rows = []
    for (task, seed, condition), group in sorted(repeated.items()):
        group.sort(key=lambda episode: episode["repeat"])
        if len(group) != 4:
            continue
        sequences = []
        first_actions = []
        for episode in group:
            with (ROOT / episode["trajectory"]).open(newline="") as stream:
                actions = list(csv.DictReader(stream))
            sequences.append(tuple(row["Executed Action"] for row in actions))
            first_actions.append(actions[0]["Model Action"])
        repeat_rows.append({
            "task": task, "seed": seed, "condition": condition,
            "success_pattern_primary_then_repeats": "".join(str(e["success"]) for e in group),
            "steps_primary_then_repeats": "/".join(str(e["num_steps"]) for e in group),
            "first_model_actions": "/".join(first_actions),
            "distinct_executed_trajectories": len(set(sequences)),
            "success_varies_across_four": len({e["success"] for e in group}) > 1,
            "success_varies_within_three_repeats": len({e["success"] for e in group[1:]}) > 1,
            "steps_vary_within_three_repeats": len({e["num_steps"] for e in group[1:]}) > 1,
            "interventions_primary_then_repeats": "/".join(str(e["interventions"]) for e in group),
            "raw_calls": "; ".join(e["raw_calls"] for e in group),
        })
    write_csv(OUT / "repeatability_matrix.csv", repeat_rows)
    (OUT / "analysis_sha256.txt").write_text(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "  analysis/analyze_e2_e3_followup.py\n")
    print(f"Analyzed {len(episodes)} completed episodes; results in {OUT}")


if __name__ == "__main__":
    main()
