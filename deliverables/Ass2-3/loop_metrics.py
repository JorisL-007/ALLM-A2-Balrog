"""Count narrow, observation-supported loop patterns in saved trajectories."""

import csv
import re
import sys
from pathlib import Path


root = Path(__file__).resolve().parent.parent / "experiment_results"
out = []
for tag in sys.argv[1:]:
    for path in sorted((root / tag).glob("seed_*/**/*_run_*.csv")):
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
            "seed": path.relative_to(root / tag).parts[0].replace("seed_", ""),
            "task": path.stem.split("_run_")[0],
            "blocked_forward_pairs": blocked,
            "alternating_turn_windows": turns,
            "unchanged_locked_door_toggle_pairs": toggles,
            "trajectory": str(path.relative_to(root.parent)),
        })

with (root / "loop_events.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, out[0].keys())
    writer.writeheader()
    writer.writerows(out)
print(root / "loop_events.csv")
