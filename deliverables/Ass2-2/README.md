# Exercise 2: compact persistent planning

`custom.py` is the submitted BALROG agent. Set `BALROG_CUSTOM_MODE=planning` (the default) to use Exercise 2. It retains at most two short subgoals, at most two facts whose text appeared in an observation, and a short expected result of the next action. This state resets for every episode. The evaluator patch records both proposed and executed actions, then sends the executed action back to the agent.

The supplied starting agent is in `custom_supplied.py`; `custom_compact_v1.py` is the development checkpoint. The final agent code is also installed at `BALROG/balrog/agents/custom.py` in the parent repository. `performance_matrix.csv`, `development_matrix.csv`, `episodes.csv`, `REPORT.md`, `research_notes.md`, and `raw_llm_samples.jsonl` are snapshots from the investigation. `evidence/` has selected full success and failure trajectories, with raw JSONL where the custom agent logged it. Full trajectories and raw custom-agent LLM logs remain under the parent repository's `experiment_results/` directory. The included scripts are copies for reproducibility; place them in the parent repository's `experiments/` directory to run them there.

With BALROG installed in conda environment `balrog` and a local OpenAI-compatible Qwen3-4B server at port 8088, run from the parent repository:

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_naive naive goto,pickup,open 11,12,13,14,15 32 128 none
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_plan custom goto,pickup,open 11,12,13,14,15 32 256 planning
```

The conditions share task seeds, observation format, temperature, history, and 32-action cap. The output caps differ because the structured reply needs more than a single action token; actual input/output tokens and elapsed time are reported. The built-in CoT reference and development trials are documented in `research_notes.md`. The raw LLM sample contains full messages and responses, not just extracted actions.
