"""Read saved transfer evidence and recover historical CoT panels; never run models."""
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.analyze_e2_e3_followup import paired_test
from exercises.common.results import result_files, verify
from exercises.common.transport_audit import read_attempts, spending, verify_calls

MODELS = ("qwen4-instruct", "gpt6luna-cloud", "olmo1b-local")
CONDITIONS = ("naive", "cot", "plan", "e3")


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def reference_valid(reference):
    source = source_path(reference["source"])
    manifest = ROOT / "experiment_results/followup_v1/manifest.json"
    return (hashlib.sha256(manifest.read_bytes()).hexdigest() == reference["original_manifest_sha256"]
            and all((source / name).is_file() and hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                    for name, digest in reference["evidence_hashes"].items()))


def recover_history(legacy, output):
    panels, records = [], []
    for agent in ("naive", "cot"):
        for repeat in (1, 2, 3):
            for env, expected in (("babyai", 50), ("crafter", 10), ("babaisai", 120)):
                folder = legacy / "qwen4-instruct" / agent / f"run_{repeat}" / env
                files = sorted(folder.rglob("*_run_*.json")) if folder.exists() else []
                data = [json.loads(p.read_text()) for p in files]
                keys = [(d["task"], d["seed"]) for d in data]
                valid = len(keys) == len(set(keys))
                for path, result in zip(files, data):
                    with path.with_suffix(".csv").open(newline="") as stream:
                        trajectory = list(csv.DictReader(stream))
                    valid &= len(trajectory) == result["num_steps"]
                    records.append(dict(agent=agent, repeat=repeat, environment=env,
                                        task=result["task"], seed=result["seed"], progression=result["progression"],
                                        steps=result["num_steps"], input_tokens=result["input_tokens"],
                                        output_tokens=result["output_tokens"], source=str(path)))
                complete = valid and len(data) == expected
                panels.append(dict(agent=agent, repeat=repeat, environment=env, n=len(data), expected=expected,
                                   status="complete" if complete else ("partial" if data else "not run"),
                                   mean_progression=statistics.mean(d["progression"] for d in data) if data else "",
                                   episode_sd=statistics.stdev(d["progression"] for d in data) if len(data) > 1 else "",
                                   input_tokens=sum(d["input_tokens"] for d in data),
                                   output_tokens=sum(d["output_tokens"] for d in data), source=str(folder)))
    aggregates = []
    for agent in ("naive", "cot"):
        for env in ("babyai", "crafter", "babaisai"):
            group = [p for p in panels if p["agent"] == agent and p["environment"] == env and p["status"] == "complete"]
            values = [p["mean_progression"] for p in group]
            aggregates.append(dict(agent=agent, environment=env, complete_repetitions=len(group),
                                   mean_progression=statistics.mean(values) if values else "",
                                   run_mean_sd=statistics.stdev(values) if len(values) > 1 else "",
                                   unit="repeated panels of identical seeds; not independent new puzzles"))
    write_csv(output / "historical_panels.csv", panels)
    write_csv(output / "historical_episodes.csv", records, ["agent", "repeat", "environment", "task", "seed", "progression", "steps", "input_tokens", "output_tokens", "source"])
    write_csv(output / "historical_matrix.csv", aggregates)
    matched = []
    for panel in panels:
        if panel["agent"] != "cot" or panel["status"] != "complete":
            continue
        baseline = next(p for p in panels if p["agent"] == "naive" and p["repeat"] == panel["repeat"]
                        and p["environment"] == panel["environment"])
        if baseline["status"] == "complete":
            matched.append(dict(environment=panel["environment"], repeat=panel["repeat"], episodes_each=panel["n"],
                                naive_progression=baseline["mean_progression"], cot_progression=panel["mean_progression"],
                                difference=panel["mean_progression"]-baseline["mean_progression"],
                                naive_input_tokens=baseline["input_tokens"], cot_input_tokens=panel["input_tokens"],
                                naive_output_tokens=baseline["output_tokens"], cot_output_tokens=panel["output_tokens"],
                                protocol="native action budget; model-default temperature; naive cap128 vs CoT512; repeated seeds"))
    write_csv(output / "historical_matched_panels.csv", matched,
              ["environment", "repeat", "episodes_each", "naive_progression", "cot_progression", "difference", "naive_input_tokens", "cot_input_tokens", "naive_output_tokens", "cot_output_tokens", "protocol"])
    return panels


def read_episodes(root):
    episodes, invalid = [], []
    for label in MODELS:
        model_root = root / label
        folders = sorted({p.parent for directory in (model_root, root / "pilots" / label)
                          for pattern in ("complete.json", "reference.json") for p in directory.rglob(pattern)
                          if not any(".working" in part or ".incomplete_" in part for part in p.parts)})
        seen = set()
        for folder in folders:
            reference = folder / "reference.json"
            reused = reference.exists()
            if reused:
                record = json.loads(reference.read_text())
                if not reference_valid(record):
                    invalid.append(str(reference)); continue
                source = source_path(record["source"])
            else:
                source = folder
            meta = json.loads((source / "complete.json").read_text())
            result = verify(source, meta["environment"], meta["task"], meta["condition"])
            if (result is None or result["seed"] != meta["seed"] or result["num_steps"] > meta["budget"]
                    or (not reused and not verify_calls(source, meta["context"]))):
                invalid.append(str(folder)); continue
            key = (bool(meta.get("pilot")), meta["environment"], meta["task"], meta["seed"], meta["condition"])
            if key in seen:
                invalid.append(str(folder)); continue
            seen.add(key)
            files = result_files(source, meta["environment"], meta["task"])
            with files[1].open(newline="") as stream:
                trajectory = list(csv.DictReader(stream))
            calls = [json.loads(line) for line in (source / "llm_calls.jsonl").read_text().splitlines()]
            audit_path = source / "api_attempts.jsonl"
            attempts = read_attempts(audit_path)
            responses = [a for a in attempts if a.get("event") == "response"]
            loops = sum([r["Executed Action"] for r in trajectory[i:i+4]] in
                        (["turn left", "turn right"] * 2, ["turn right", "turn left"] * 2)
                        and trajectory[i]["Observation"] == trajectory[i+2]["Observation"]
                        and trajectory[i+1]["Observation"] == trajectory[i+3]["Observation"]
                        for i in range(max(0, len(trajectory)-3)))
            stalled = sum(a["Observation"] == b["Observation"] and a["Executed Action"] == b["Executed Action"]
                          and b["Executed Action"] in ("go forward", "pick up", "toggle")
                          for a, b in zip(trajectory, trajectory[1:]))
            episodes.append(dict(model_label=label, panel="preflight" if meta.get("pilot") else "primary",
                                 environment=meta["environment"], task=meta["task"], seed=meta["seed"],
                                 condition=meta["condition"], repeat=0, reused=reused, budget=meta["budget"],
                                 success=int(result["progression"] == 1) if meta["environment"] == "babyai" else "",
                                 progression=result["progression"], num_steps=result["num_steps"],
                                 input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                                 elapsed_seconds=meta["elapsed_seconds"], llm_calls=len(calls),
                                 reflection_calls=sum(c["role"] == "reflection" for c in calls),
                                 invalid_proposals=sum(r["Action"] != r["Executed Action"] for r in trajectory),
                                 interventions=sum(bool(r["Override Reason"]) for r in trajectory),
                                 stalled_pairs=stalled, turn_loops=loops,
                                 truncations=sum(a["response"]["choices"][0].get("finish_reason") == "length" for a in responses if a["response"].get("choices")) if audit_path.exists() else "",
                                 missing_action_outputs=sum("ACTION:" not in c["raw_response"] for c in calls) if meta["condition"] == "cot" else 0,
                                 transport_attempts=len(attempts) if audit_path.exists() else "",
                                 trajectory=str(files[1]), raw_calls=str(source / "llm_calls.jsonl")))
    return episodes, invalid


def analyze(root, output, legacy):
    if root.resolve() == output.resolve():
        raise ValueError("Use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    historical = recover_history(legacy, output)
    episodes, invalid = read_episodes(root)
    episode_fields = ["model_label", "panel", "environment", "task", "seed", "condition", "repeat", "reused", "budget", "success", "progression", "num_steps", "input_tokens", "output_tokens", "elapsed_seconds", "llm_calls", "reflection_calls", "invalid_proposals", "interventions", "stalled_pairs", "turn_loops", "truncations", "missing_action_outputs", "transport_attempts", "trajectory", "raw_calls"]
    write_csv(output / "episodes.csv", episodes, episode_fields)
    matrix, coverage, samples = [], [], []
    for label in MODELS:
        setup_path = root / "server" / label / "model_setup.json"
        setup = json.loads(setup_path.read_text()) if setup_path.exists() else {}
        model_id = {"qwen4-instruct": "Qwen/Qwen3-4B-Instruct-2507-FP8", "gpt6luna-cloud": "gpt-6-luna",
                    "olmo1b-local": "allenai/OLMo-2-0425-1B-Instruct"}[label]
        precision = setup.get("precision", "API model ID" if label == "gpt6luna-cloud" else ("FP8" if label == "qwen4-instruct" else "not selected"))
        for env, tasks in (("babyai", ("goto", "pickup", "open")), ("crafter", ("default",))):
            for condition in CONDITIONS:
                for task in tasks:
                    group = [e for e in episodes if e["panel"] == "primary" and e["model_label"] == label
                             and e["environment"] == env and e["task"] == task and e["condition"] == condition]
                    seeds = range(30, 90, 2 if condition == "cot" else 1) if env == "babyai" else range(120, 130, 2 if condition == "cot" else 1)
                    coverage.append(dict(model_label=label, environment=env, task=task, condition=condition,
                                         completed=len(group), expected=len(seeds), missing_seeds=" ".join(str(s) for s in seeds if s not in {e["seed"] for e in group})))
                    values = [e["progression"] for e in group]
                    row = dict(model_label=label, model_id=model_id, precision=precision, environment=env, task=task, condition=condition, n=len(group),
                               expected=len(seeds), mean_progression=statistics.mean(values) if values else "",
                               episode_sd=statistics.stdev(values) if len(values) > 1 else "",
                               reused=sum(e["reused"] for e in group))
                    for key in ("num_steps", "input_tokens", "output_tokens", "elapsed_seconds", "llm_calls", "reflection_calls", "invalid_proposals", "interventions", "stalled_pairs", "turn_loops", "missing_action_outputs"):
                        row[key] = sum(e[key] for e in group)
                    row["truncations"] = sum(e["truncations"] for e in group if e["truncations"] != "") if any(e["truncations"] != "" for e in group) else ""
                    row["new_episode_seconds"] = sum(e["elapsed_seconds"] for e in group if not e["reused"])
                    matrix.append(row)
                    for outcome in (True, False):
                        chosen = next((e for e in group if (e["success"] == 1 if env == "babyai" else e["progression"] > 0) == outcome), None)
                        samples.append(dict(model_label=label, environment=env, task=task, condition=condition,
                                            outcome="success/progress" if outcome else "failure/zero progress",
                                            trajectory=chosen["trajectory"] if chosen else "unavailable",
                                            raw_calls=chosen["raw_calls"] if chosen else "unavailable", manual_review="pending"))
    write_csv(output / "performance_matrix.csv", matrix)
    write_csv(output / "coverage.csv", coverage)
    write_csv(output / "inspection_index.csv", samples)
    overall = []
    for label in MODELS:
        for env in ("babyai", "crafter"):
            for condition in CONDITIONS:
                group = [e for e in episodes if e["panel"] == "primary" and e["model_label"] == label
                         and e["environment"] == env and e["condition"] == condition]
                parts = [r for r in matrix if r["model_label"] == label and r["environment"] == env and r["condition"] == condition]
                values = [e["progression"] for e in group]
                row = dict(model_label=label, model_id=parts[0]["model_id"], precision=parts[0]["precision"],
                           environment=env, condition=condition, n=len(group), expected=sum(r["expected"] for r in parts),
                           successes=sum(e["success"] for e in group) if env == "babyai" else "",
                           mean_progression=statistics.mean(values) if values else "",
                           episode_sd=statistics.stdev(values) if len(values) > 1 else "",
                           mean_steps=statistics.mean(e["num_steps"] for e in group) if group else "")
                for key in ("reused", "input_tokens", "output_tokens", "elapsed_seconds", "new_episode_seconds", "llm_calls", "reflection_calls", "invalid_proposals", "turn_loops", "stalled_pairs", "missing_action_outputs"):
                    row[key] = sum(r[key] for r in parts)
                row["truncations"] = sum(r["truncations"] for r in parts if r["truncations"] != "") if any(r["truncations"] != "" for r in parts) else ""
                overall.append(row)
    write_csv(output / "overall_matrix.csv", overall)
    paired = []
    for label in MODELS:
        selected = [e for e in episodes if e["model_label"] == label]
        for first, second in (("naive", "cot"), ("naive", "plan"), ("cot", "plan"), ("plan", "e3")):
            seeds = range(30, 90, 2 if "cot" in (first, second) else 1)
            result = paired_test(selected, "transfer", first, second, ["goto", "pickup", "open"], seeds)
            if result:
                paired.append(dict(model_label=label, **result))
    ordered = sorted(paired, key=lambda x: x["test_p"])
    running = 0
    for rank, row in enumerate(ordered):
        # Always retain the prespecified twelve-test family, even for partial runs.
        running = max(running, row["test_p"] * (12-rank))
        row["holm_p"] = min(1, running)
    write_csv(output / "paired_comparisons.csv", paired,
              ["model_label", "panel", "first", "second", "tasks", "paired_seeds", "difference", "ci_low", "ci_high", "gains", "losses", "ties", "exact_mcnemar_p", "seed_block_permutation_p", "test_p", "holm_p"])
    cost = spending(root)
    status = dict(verified_evaluation_episodes=sum(e["panel"] == "primary" for e in episodes),
                  expected_evaluation_episodes=1995, pilots=sum(e["panel"] == "preflight" for e in episodes),
                  reused=sum(e["reused"] for e in episodes), invalid_evidence=invalid,
                  historical_cot_episodes=sum(p["n"] for p in historical if p["agent"] == "cot"),
                  cloud_cost=cost, manual_interpretation="pending; inspection index is not a claim of manual review")
    status["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (output / "summary.md").write_text(
        "# CoT control and model transfer\n\n"
        f"Verified evaluation episodes: {status['verified_evaluation_episodes']}/1995; reused: {status['reused']}.\n\n"
        f"Historical CoT episodes recovered: {status['historical_cot_episodes']}.\n\n"
        f"Estimated cumulative cloud cost: ${cost['estimated_usd']:.4f}, including ${cost['uncertain_usd']:.4f} uncertain charges.\n\n"
        "See overall_matrix.csv, performance_matrix.csv, coverage.csv, paired_comparisons.csv, historical_matrix.csv, and inspection_index.csv.\n\n"
        "CoT comparisons use its exact half-sized seed subset. Paired intervals resample seed blocks; the twelve-test Holm family remains fixed.\n\n"
        "Historical native-budget panels and new capped panels are separate. Manual interpretation is pending. Missing episodes are not failures.\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiment_results/refactored_v1/model_transfer_luna_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, default=ROOT / "exercise1_results/20261014_221758")
    args = parser.parse_args()
    analyze(args.input.resolve(), args.output.resolve(), args.legacy.resolve())
