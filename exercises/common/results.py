"""Read-only helpers for saved BALROG evidence; no model execution."""

import csv
import json

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

