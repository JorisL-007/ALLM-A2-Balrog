"""Count narrow, observation-supported loop patterns in saved trajectories."""

import csv
import re
import argparse
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("tags", nargs="*")
parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_results")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.input.resolve()
output = args.output.resolve()
if root == output:
    parser.error("Use separate input and output directories")
output.mkdir(parents=True, exist_ok=True)
out = []
import json
new_groups = {}
for completion in (root.rglob("complete.json") if (root / "manifest.json").exists() else []):
    if any(".working" in x or ".incomplete_" in x for x in completion.parts):
        continue
    meta = json.loads(completion.read_text())
    for path in completion.parent.rglob("*_run_*.csv"):
        new_groups.setdefault(meta["condition"], []).append((path, meta["seed"]))
for tag in args.tags or sorted(new_groups or {p.parent.name: [] for p in root.glob("*/settings.txt")}):
    paths = new_groups.get(tag) if new_groups else [(p, p.relative_to(root / tag).parts[0].replace("seed_", "")) for p in sorted((root / tag).glob("seed_*/**/*_run_*.csv"))]
    for path, seed in paths or []:
        rows = list(csv.DictReader(path.open()))
        blocked = turns = toggles = 0
        for i in range(1, len(rows)):
            previous, current = rows[i - 1], rows[i]
            same_view = current["Observation"] == previous["Observation"]
            front_obstacle = re.search(
                r"a (?:wall|locked [^\n]*door|[^\n]*key|[^\n]*box|[^\n]*ball) 1 step forward",
                previous["Observation"],
            )
            if current.get("Executed Action", current["Action"]) == "go forward" and same_view and front_obstacle:
                blocked += 1
            if (current.get("Executed Action", current["Action"]) == "toggle"
                    and previous.get("Executed Action", previous["Action"]) == "toggle"
                    and same_view and "locked" in current["Observation"] and "door" in current["Observation"]):
                toggles += 1
            if i >= 3:
                segment = rows[i - 3:i + 1]
                actions = [row.get("Executed Action", row["Action"]) for row in segment]
                views = [row["Observation"] for row in segment]
                if (actions in (["turn left", "turn right"] * 2, ["turn right", "turn left"] * 2)
                        and views[0] == views[2] and views[1] == views[3] and views[0] != views[1]):
                    turns += 1
        out.append({
            "condition": tag,
            "seed": seed,
            "task": path.stem.split("_run_")[0],
            "blocked_forward_pairs": blocked,
            "alternating_turn_windows": turns,
            "unchanged_locked_door_toggle_pairs": toggles,
            "trajectory": os.path.relpath(path, Path(__file__).resolve().parents[1]),
        })

with (output / "loop_events.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, out[0].keys() if out else ("condition", "seed", "task", "blocked_forward_pairs", "alternating_turn_windows", "unchanged_locked_door_toggle_pairs", "trajectory"))
    writer.writeheader()
    writer.writerows(out)
print(output / "loop_events.csv")
