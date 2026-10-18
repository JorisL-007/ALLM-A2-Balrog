"""Run the fixed follow-up BALROG panels, one resumable episode at a time."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
TASKS = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
CONDITIONS = {
    "naive": ("naive", "planning", 16, 0, 0, 0, 256),
    "plan": ("custom", "planning", 16, 0, 0, 0, 256),
    "e3": ("custom", "reflect", 16, 0, 0, 0, 256),
    "guide_h4": ("custom", "planning", 4, 1, 0, 0, 256),
    "guide": ("custom", "planning", 16, 1, 0, 0, 256),
    "guide_h32": ("custom", "planning", 32, 1, 0, 0, 256),
    "guide_gate": ("custom", "planning", 16, 1, 1, 0, 256),
    "guide_goal": ("custom", "planning", 16, 1, 0, 1, 256),
}
SOURCE_FILES = [
    "BALROG/balrog/agents/custom.py",
    "BALROG/balrog/agents/naive.py",
    "BALROG/balrog/evaluator.py",
    "BALROG/balrog/client.py",
    "BALROG/balrog/prompt_builder/history.py",
    "BALROG/balrog/environments/babyai_text/__init__.py",
    "BALROG/balrog/environments/babyai_text/clean_lang_wrapper.py",
    "experiments/followup_runner.py",
    "experiments/run_followup_overnight.sh",
]


def panel_cases(panel):
    if panel == "pilot":
        for seed in range(20, 25):
            for task in ("goto", "pickup", "open"):
                for condition in ("plan", "guide", "guide_gate", "guide_goal", "guide_h32"):
                    yield "babyai", task, seed, condition, 32, 0
    if panel == "primary":
        for seed in range(30, 90):
            for task in ("goto", "pickup", "open"):
                for condition in CONDITIONS:
                    yield "babyai", task, seed, condition, 32, 0
    if panel == "broad":
        for seed in range(100, 112):
            for task in TASKS:
                for condition in ("plan", "guide", "guide_gate", "guide_goal"):
                    yield "babyai", task, seed, condition, 128, 0
    if panel == "crafter":
        for seed in range(120, 130):
            for condition in ("plan", "guide_goal"):
                yield "crafter", "default", seed, condition, 50, 0
    if panel == "repeat":
        for seed in range(30, 35):
            for task in ("goto", "pickup"):
                for condition in ("plan", "guide_gate"):
                    for repeat in range(1, 4):
                        yield "babyai", task, seed, condition, 32, repeat


def file_hash(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def manifest():
    return {
        "model": MODEL,
        "temperature": 0,
        "server_context": 8192,
        "conditions": {name: list(settings) for name, settings in CONDITIONS.items()},
        "panels": {"pilot": "20-24", "primary": "30-89", "broad": "100-111",
                   "crafter": "120-129", "repeat": "30-34 x3"},
        "source_hashes": {path: file_hash(path) for path in SOURCE_FILES},
    }


def check_manifest(out):
    expected = manifest()
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise SystemExit(f"Settings or code changed since this run began: {path}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2) + "\n")


def result_files(directory, environment, task):
    name = "BabyAI-MixedTrainLocal-v0/" + task if environment == "babyai" else task
    files = list(directory.rglob(f"{task}_run_00.json"))
    if len(files) != 1:
        # BALROG may retain the full task name in the result filename.
        files = list(directory.rglob("*_run_00.json"))
    if len(files) != 1 or name not in str(files[0]):
        return None
    trajectory = files[0].with_suffix(".csv")
    return files[0], trajectory


def verify(directory, environment, task, condition):
    files = result_files(directory, environment, task)
    if not files or not all(path.exists() for path in files):
        return None
    result = json.loads(files[0].read_text())
    with files[1].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != result["num_steps"] or not rows:
        return None
    if any("Model Action" not in row or "Override Reason" not in row for row in rows):
        return None
    calls_path = directory / "llm_calls.jsonl"
    if not calls_path.exists():
        return None
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    if len([call for call in calls if call["role"] == "action"]) != len(rows):
        # E3's critic can supply the action directly, so it has fewer action calls.
        if condition != "e3" or len(calls) < len(rows):
            return None
    if sum(call["input_tokens"] for call in calls) != result["input_tokens"]:
        return None
    if sum(call["output_tokens"] for call in calls) != result["output_tokens"]:
        return None
    return result


def run_case(out, environment, task, seed, condition, budget, repeat, port, panel):
    agent, mode, history, guide, gate, goal, tokens = CONDITIONS[condition]
    target = out / ("repeat" if repeat else "episodes") / environment / task / condition / f"seed_{seed:03d}" / f"rep_{repeat}"
    if target.exists():
        if (target / "complete.json").exists() and verify(target, environment, task, condition):
            return "skip", 0
        raise SystemExit(f"Incomplete final directory requires inspection: {target}")
    work = target.parent / (target.name + ".working")
    if work.exists():
        archive = target.parent / f"{target.name}.incomplete_{int(time.time())}"
        work.rename(archive)
    work.mkdir(parents=True)
    task_override = f"[BabyAI-MixedTrainLocal-v0/{task}]"
    command = [
        "conda", "run", "--no-capture-output", "-n", "balrog", "python", "eval.py",
        f"agent.type={agent}", f"agent.max_text_history={history}",
        "agent.max_image_history=0", "agent.remember_cot=True", "agent.max_cot_history=1",
        f"envs.names={environment}", f"envs.env_kwargs.seed={seed}",
        f"eval.num_episodes.{environment}=1", "eval.num_workers=1",
        f"eval.max_steps_per_episode={budget}", "eval.save_trajectories=True",
        f"eval.resume_from={work}", "client.client_name=vllm", f"client.model_id={MODEL}",
        f"client.base_url=http://127.0.0.1:{port}/v1",
        "client.generate_kwargs.temperature=0", f"client.generate_kwargs.max_tokens={tokens}",
        "client.generate_kwargs.thinking_budget=null",
    ]
    command += [f"tasks.babyai_tasks={task_override}"] if environment == "babyai" else [f"envs.crafter_kwargs.seed={seed}"]
    run_env = os.environ.copy()
    run_env.update({"BALROG_CUSTOM_MODE": mode, "BALROG_CUSTOM_GUIDE": str(guide),
                    "BALROG_CUSTOM_GATE": str(gate), "BALROG_CUSTOM_GOAL_STATE": str(goal),
                    "BALROG_LLM_LOG": str(work / "llm_calls.jsonl"), "HF_HUB_OFFLINE": "1"})
    (work / "command.json").write_text(json.dumps({"command": command, "environment": {
        name: run_env[name] for name in run_env if name.startswith("BALROG_CUSTOM_")}}, indent=2) + "\n")
    start = time.monotonic()
    with (work / "console.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT / "BALROG", env=run_env, stdout=log,
                                   stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    result = verify(work, environment, task, condition) if completed.returncode == 0 else None
    if result is None:
        raise SystemExit(f"Episode failed verification; raw attempt retained at {work}")
    (work / "complete.json").write_text(json.dumps({
        "panel": panel, "environment": environment, "task": task, "seed": seed, "condition": condition,
        "budget": budget, "repeat": repeat, "elapsed_seconds": elapsed,
        "progression": result.get("progression"), "num_steps": result["num_steps"],
        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
    }, indent=2) + "\n")
    work.rename(target)
    return "done", elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("pilot", "primary", "broad", "crafter", "repeat", "all"), default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "experiment_results/followup_v1")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Only launch this many unfinished episodes")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    out = args.out.resolve()
    check_manifest(out)
    panels = ("primary", "broad", "crafter", "repeat") if args.panel == "all" else (args.panel,)
    deadline = time.monotonic() + args.hours * 3600
    done = 0
    for panel in panels:
        for case in panel_cases(panel):
            if time.monotonic() >= deadline or (args.limit and done >= args.limit):
                print(f"Stopped between episodes after {done} new episodes; rerun the same command to resume.", flush=True)
                return
            state, elapsed = run_case(out, *case, args.port, panel)
            if state == "done":
                done += 1
                print(f"{panel} {case}: {elapsed:.1f}s", flush=True)
    print(f"All requested panels complete; {done} new episodes.", flush=True)


if __name__ == "__main__":
    main()
