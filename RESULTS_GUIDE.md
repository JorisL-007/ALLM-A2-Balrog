# Where the results are

Start with the [experiment index](research/experiment_index.csv). It maps each research step to its agent, settings, results, and inspected examples.

| Folder | Contents |
|---|---|
| `exercise1_results/` | Original local naive and CoT baselines. |
| `experiment_results/e2_*`, `e3_*`, `heldout_*`, `cross_*` | Earlier planning and reflection development/evaluation. |
| `experiment_results/followup_v1/` | Larger Qwen guide, pickup-rule, history, and reflection study. |
| `experiment_results/refactored_v1/model_transfer_luna_v1/` | Final matched Qwen, GPT-6 Luna, and OLMo comparison. |
| `research/cloud_model/` | Teammate's separately imported cloud-model baselines. |
| `research/model_transfer_verification/` | Verification records and manually inspected trajectory notes. |

## Final model-transfer study

In [`model_transfer_luna_v1/`](experiment_results/refactored_v1/model_transfer_luna_v1/):

- [`analysis/overall_matrix.csv`](experiment_results/refactored_v1/model_transfer_luna_v1/analysis/overall_matrix.csv): results by model and agent. `performance_matrix.csv` breaks them down by task; `paired_comparisons.csv` gives matched differences, intervals, and tests; `coverage.csv` lists missing cases; `episodes.csv` has one row per episode. `historical_*` files summarize older CoT evidence. `analysis_completed/` and `analysis_rechecked/` are regenerated checks, not extra runs.
- `qwen4-instruct/`, `gpt6luna-cloud/`, `olmo1b-local/`: raw evaluation episodes. Conditions are `naive` (supplied baseline), `cot` (supplied CoT), `plan` (frozen planning), and `e3` (action critique).
- `pilots/`: short checks excluded from evaluation. `luna_continuation/`: resume status and two provider refusals. [`manifest.json`](experiment_results/refactored_v1/model_transfer_luna_v1/manifest.json) pins settings and source hashes.

An episode lives at `<model>/episodes/<environment>/<task>/<condition>/seed_NNN/rep_0/`. For example, [Luna pickup seed 51 with critique](experiment_results/refactored_v1/model_transfer_luna_v1/gpt6luna-cloud/episodes/babyai/pickup/e3/seed_051/rep_0/llm_calls.jsonl) has:

| File | Purpose |
|---|---|
| `llm_calls.jsonl` | Full prompts, replies, proposed actions, critique calls, and tokens. |
| `api_attempts.jsonl` | Transport requests, responses, retries, and usage. |
| `babyai/.../*_run_00.csv` | Observations, proposed and executed actions, rewards, and outcomes by step. |
| `complete.json` | Verified episode settings and totals. |
| `command.json` | Exact BALROG command and agent flags. |
| `console.log`, `eval.log` | Diagnostics. |

Read `llm_calls.jsonl` beside the trajectory CSV to see what the model proposed and what happened after the executed action. A `.working` or `.incomplete_*` folder holds an unfinished attempt; it is not a completed task failure. The two Luna CoT provider refusals remain in `.working` folders and are excluded from success-rate denominators.
