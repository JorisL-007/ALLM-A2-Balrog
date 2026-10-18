# BALROG reasoning agent investigation

The completed follow-up harness study is documented in [research/FOLLOWUP_README.md](research/FOLLOWUP_README.md), with results in [research/followup_report.md](research/followup_report.md). Raw calls are sampled in [pilot](research/followup_raw_llm_samples.jsonl) and [held-out](research/followup_heldout_raw_samples.jsonl) files, and the full calls remain beside each episode trajectory.

Exercises 2 and 3 of `docs/Ass2Reasoning.pdf` are implemented in [`BALROG/balrog/agents/custom.py`](BALROG/balrog/agents/custom.py). The same agent has two modes: `planning` uses compact persistent subgoals, observation-supported facts, and an expected next observation; `reflect` adds a bounded critique and plan revision when a specific observed loop is detected. The original supplied agent is saved in `experiments/checkpoints/custom_supplied.py`.

The short conclusion is in [`REPORT.md`](REPORT.md), the adaptation record is in [`research_notes.md`](research_notes.md), a direct [trajectory index](research/trajectory_index.md) points to selected successes and failures, and the per-episode and aggregate data are in [`experiment_results/`](experiment_results/). Full custom-agent prompts and raw model responses are saved as `seed_*/llm_calls.jsonl` in each run folder. [`research/raw_llm_samples.jsonl`](research/raw_llm_samples.jsonl) is a small portable sample for manual inspection.

## Reproduce a panel

Use the supplied BALROG checkout and its `balrog` conda environment, and serve a local OpenAI-compatible model at `http://127.0.0.1:8088/v1`. The runs used vLLM in a separate `vllm` conda environment, temperature zero, text observations, history length 16, one episode per task and seed, and one evaluator worker. Set `MODEL` to the server's served model name. Example with the local Qwen3-4B instruction model:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  conda run --no-capture-output -n vllm vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --host 127.0.0.1 --port 8088 --served-model-name Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --gpu-memory-utilization 0.88 --max-model-len 8192 \
  --generation-config auto --seed 0
```

In another terminal, run a panel:

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_plan custom goto,pickup,open 11,12,13,14,15 32 256 planning
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_reflect custom goto,pickup,open 11,12,13,14,15 32 256 reflect
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  heldout_qwen_naive naive goto,pickup,open 11,12,13,14,15 32 128 none
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  transfer_qwen_naive naive default 11,12,13 50 128 none crafter
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  transfer_qwen_plan custom default 11,12,13 50 256 planning crafter
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8 bash experiments/run_panel.sh \
  transfer_qwen_reflect custom default 11,12,13 50 256 reflect crafter
```

For the limited model comparison, restart the server with `Qwen/Qwen3-14B-AWQ` (add `--default-chat-template-kwargs '{"enable_thinking": false}'`) or `microsoft/Phi-4-mini-instruct`, keeping the other server settings. Then run the same `goto,pickup 11,12,13 32` panels with tags `cross_qwen14_*` or `cross_phi4_*`: `naive` uses `128 none`, `custom` planning and reflection use `256 planning` and `256 reflect`. The actual command settings and model names for every panel are saved under `experiment_results/<tag>/settings.txt`; server logs are in `experiment_results/server/`.

The script writes the settings, code hashes, console, timing, BALROG episode JSON/CSV trajectories, and custom-agent raw LLM JSONL under `experiment_results/<tag>/`. It uses a 32-action BabyAI budget; Crafter transfer used 50 actions. These budgets make the matched conditions comparable, but are shorter than BALROG's default evaluation. `BALROG/balrog/evaluator.py` has the small change recorded in [`experiments/evaluator_logging.patch`](experiments/evaluator_logging.patch): CSV distinguishes the model's proposed action from the action BALROG actually executes after a fallback, and the agent receives the executed action as feedback.

To regenerate the final summaries after the runs:

```bash
python3 experiments/summarize.py \
  heldout_qwen_naive heldout_qwen_plan heldout_qwen_reflect \
  transfer_qwen_naive transfer_qwen_plan transfer_qwen_reflect \
  cross_qwen14_naive cross_qwen14_plan cross_qwen14_reflect \
  cross_phi4_naive cross_phi4_plan cross_phi4_reflect
python3 experiments/loop_metrics.py \
  heldout_qwen_naive heldout_qwen_plan heldout_qwen_reflect
```

`performance_matrix.csv` reports mean progression and sample SD. BabyAI progression is binary mission success; Crafter progression is achievements divided by 22. `episodes.csv` links every result to its trajectory. The loop counts are descriptive overlapping observation/action windows, not independent failures.

See [`deliverables/Ass2-2/README.md`](deliverables/Ass2-2/README.md) and [`deliverables/Ass2-3/README.md`](deliverables/Ass2-3/README.md) for the exercise-specific commands and artifact contents. The matching submission archives are [`Ass2-2.zip`](deliverables/Ass2-2.zip) and [`Ass2-3.zip`](deliverables/Ass2-3.zip).
