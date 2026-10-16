# Exercise 3: feedback-triggered reflection

`custom.py` is the submitted BALROG agent. Set `BALROG_CUSTOM_MODE=reflect` for Exercise 3 and `BALROG_CUSTOM_MODE=planning` for its matched no-reflection comparator. The critic is called only after a repeated front-obstacle interaction leaves the observation unchanged or two alternating turn pairs return to the same views. It revises subgoals, records a short lesson, predicts a visible outcome, and chooses the next action. It is capped at three calls per episode, with at least six actions between calls; at most two lessons remain in memory. All critique calls count toward the logged episode tokens. The evaluator patch returns the executed fallback action to the agent and keeps both proposed and executed actions in trajectories.

The final agent code is also installed at `BALROG/balrog/agents/custom.py` in the parent repository. `custom_reflect_v1.py` and `custom_reflect_v3_door_trigger.py` are rejected development variants; `custom_reflect_v2.py` is the frozen final checkpoint. `performance_matrix.csv`, `development_matrix.csv`, `episodes.csv`, `loop_events.csv`, `REPORT.md`, `research_notes.md`, and `raw_llm_samples.jsonl` are snapshots. `evidence/` contains selected paired success and failure trajectories with full custom-agent raw-call JSONL. The parent repository's `experiment_results/` contains all trajectories, raw calls, settings, timing, and code hashes. The included scripts are copies for reproducibility; place them in the parent repository's `experiments/` directory to run them there.

With BALROG installed in conda environment `balrog` and a local OpenAI-compatible Qwen3-4B server at port 8088, run the matched conditions from the parent repository:

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_plan custom goto,pickup,open 11,12,13,14,15 32 256 planning
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_reflect custom goto,pickup,open 11,12,13,14,15 32 256 reflect
```

Both use the same agent, model, tasks, seeds, prompts except for the reflection branch, temperature, 16-entry text history, and 32-action cap. Crafter transfer used 50 actions, seeds 11–13. Reflection made no calls there because these triggers only cover BabyAI behavior; this limit is reported rather than counted as evidence of Crafter benefit.
