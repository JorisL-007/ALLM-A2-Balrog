"""Summarize the pinned teammate contribution without importing or running BALROG.

The archive is read in place. Reported values and verified snapshot values stay
separate, particularly for the incomplete local Qwen2B run.
"""
import argparse
import csv
import hashlib
import io
import json
import statistics
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(archive_path, output):
    metadata = json.loads((archive_path.parent / "provenance.json").read_text())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != metadata["archive_sha256"]:
        raise ValueError("Archive does not match the pinned provenance hash")
    episodes, matrix, task_rows, coverage = [], [], [], []
    with tarfile.open(archive_path) as archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]

        def read_json(name):
            return json.load(archive.extractfile(name))

        for case in metadata["cases"]:
            prefix = "results/" + case["case"] + "/"
            members = [n for n in names if n.startswith(prefix)]
            records = []
            seen = set()
            for name in sorted(members):
                if not (name.endswith(".json") and "_run_" in name):
                    continue
                data = read_json(name)
                key = (data["task"], data["seed"])
                if key in seen:
                    raise ValueError(f"Duplicate task/seed in {case['case']}: {key}")
                seen.add(key)
                if data["client"]["model_id"] != case["model"]:
                    raise ValueError(f"Unexpected model in {name}")
                trajectory = name[:-5] + ".csv"
                rows = list(csv.DictReader(io.StringIO(archive.extractfile(trajectory).read().decode())))
                if len(rows) != data["num_steps"]:
                    raise ValueError(f"Trajectory length mismatch: {name}")
                record = dict(case=case["case"], hosting=case["hosting"],
                              model=case["model"], agent=case["agent"],
                              environment=case["environment"], task=data["task"],
                              seed=data["seed"], progression_pct=100 * data["progression"],
                              steps=data["num_steps"], input_tokens=data["input_tokens"],
                              output_tokens=data["output_tokens"], archive_member=name,
                              trajectory_member=trajectory)
                records.append(record)
                episodes.append(record)
            expected = {(task, seed) for task in case["tasks"] for seed in case["seeds"]}
            if seen - expected:
                raise ValueError(f"Unexpected task/seed in {case['case']}")
            summaries = [read_json(n) for n in sorted(members)
                         if n.endswith(f"/{case['environment']}_summary.json")]
            summary_count = sum(d["episodes_played"] for d in summaries)
            if records:
                count = len(records)
                mean = statistics.mean(r["progression_pct"] for r in records)
                sd = statistics.stdev(r["progression_pct"] for r in records) if count > 1 else ""
                steps = statistics.mean(r["steps"] for r in records)
                tokens_in = sum(r["input_tokens"] for r in records)
                tokens_out = sum(r["output_tokens"] for r in records)
                level = "episode JSON and trajectory CSV"
            else:
                count = summary_count
                mean = sum(d["progression_percentage"] * d["episodes_played"] for d in summaries) / count
                sd = ""  # Do not label seed-summary variability as episode SD.
                steps = sum(d["average_steps"] * d["episodes_played"] for d in summaries) / count
                tokens_in = sum(d["input_tokens"] for d in summaries)
                tokens_out = sum(d["output_tokens"] for d in summaries)
                level = "summary only; no individual trajectories"
            complete = count == case["expected_episodes"]
            status = "verified" if complete and round(mean, 2) == case["reported_progress_pct"] else "incomplete; reported value not verified"
            matrix.append(dict(case=case["case"], model=case["model"], hosting=case["hosting"],
                               agent=case["agent"], environment=case["environment"],
                               reported_progress_pct=case["reported_progress_pct"],
                               snapshot_progress_pct=mean, snapshot_episodes=count,
                               expected_episodes=case["expected_episodes"], episode_sd_pct=sd,
                               mean_steps=steps, input_tokens=tokens_in, output_tokens=tokens_out,
                               evidence_level=level, status=status,
                               temperature=case["temperature"], output_cap=case["output_cap"],
                               history=case["history"], action_budget=case["action_budget"],
                               script_member=case["script"]))
            for task in case["tasks"]:
                values = [r["progression_pct"] for r in records if r["task"] == task]
                task_level = "episode JSON"
                if not records:
                    values = [d["tasks"][task]["progression_percentage"] for d in summaries]
                    task_level = "per-task summaries (one episode per task/seed)"
                task_rows.append(dict(case=case["case"], task=task, episodes=len(values),
                                      progression_pct=statistics.mean(values) if values else "",
                                      evidence_level=task_level))
            orphan_csv = [n for n in members if n.endswith(".csv") and n[:-4] + ".json" not in members]
            coverage.append(dict(case=case["case"], episode_json=len(records),
                                 trajectory_csv=sum(n.endswith(".csv") for n in members),
                                 summary_episode_count=summary_count,
                                 missing_episode_records=[dict(task=t, seed=s) for t, s in sorted(expected - seen)],
                                 orphan_trajectories=orphan_csv))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "performance_matrix.csv", matrix)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "per_task.csv", task_rows)
    (output / "verification.json").write_text(json.dumps(dict(
        commit=metadata["commit"], archive_sha256=digest, cases=coverage,
        raw_call_jsonl_files=sum(n.endswith(".jsonl") for n in names if n.startswith("results/")),
        excluded="results/setup_check: one GPT-4o-mini preflight; not a study condition",
        limitations=metadata["limitations"]), indent=2) + "\n")
    for row in matrix:
        print(f"{row['case']}: {row['snapshot_progress_pct']:.2f}% "
              f"({row['snapshot_episodes']}/{row['expected_episodes']}), {row['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "research/cloud_model/upstream_snapshot.tar.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "research/cloud_model/analysis")
    args = parser.parse_args()
    summarize(args.input, args.output)
