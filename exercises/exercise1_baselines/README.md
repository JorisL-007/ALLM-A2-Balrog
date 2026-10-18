# Exercise 1 — establish baseline behavior

[Study overview](../../README.md) · [Next: compact planning](../exercise2_planning/README.md) · [Experiment index](../../research/experiment_index.csv)

## Step 01: three local models with the supplied naive agent

1. **Observed problem.** Baseline trajectories show malformed proposals, walking into walls, and repeated turns. Small-model performance cannot be inferred from model size or from one successful episode.
2. **Hypothesis.** A common set of environments and repeated runs will establish each model's strengths and the size of run-to-run variation before changing its harness.
3. **Change.** No planning state or critique. BALROG's supplied naive prompt asks: `You always have to output one of the above actions at a time and no other text.`
4. **Experiment.** Three complete repetitions per model, reusing the same seed panels. Models: Qwen3-4B-Instruct-2507-FP8, Qwen3-14B-AWQ, Phi-4-mini-instruct. Text history 16, no images, four workers, model generation defaults (`temperature=null`), 128 output tokens, native environment action limits. E1 Qwen4 server context 16,384; other models 8,192; GPU memory utilization 0.90 (later E2/E3 0.88). Qwen14 native thinking is disabled. These settings differ from the later temperature-zero, capped studies.
5. **Observation.** The table below describes the completed sweep at [20261015_072525](../../exercise1_results/20261015_072525). SD is across three **run means**, not across all individual episodes. Repeated seeds are not independent new puzzles.
6. **Decision.** Reuse these results for baseline context; rerun naive only when later tasks/seeds/action budgets require a matched comparison.

| Per-model repetition | Exact task set | Seeds | Episodes |
|---|---|---|---:|
| BabyAI | `goto`, `pickup`, `open`, `putnext`, `pick_up_seq_go_to` | 1–10 per task | 50 |
| Crafter | `default` survival/achievement task | 1–10 | 10 |
| BabaIsAI | all 40 installed built-in tasks (listed in the experiment index) | 1–3 per task | 120 |

Thus the complete three-model naive sweep contains **1,620 episodes**. TextWorld, MiniHack, and NetHack were not executed. There is no evidenced cloud-model result; this assignment requirement remains missing.

| Model | BabyAI progress | Crafter progress | BabaIsAI progress |
|---|---:|---:|---:|
| Qwen3-4B | 32.00% ± 2.00% | 18.94% ± 2.62% | 2.22% ± 1.27% |
| Qwen3-14B | 21.33% ± 1.15% | 9.55% ± 1.20% | 5.83% ± 0.83% |
| Phi-4-mini | 3.33% ± 1.15% | 2.73% ± 0.91% | 7.50% ± 0.00% |

BabyAI progression is binary mission completion. Crafter progression is unlocked achievements divided by 22. Use each environment's progression definition rather than interpreting the overall average as a success probability. [Complete three-model matrix](../../exercise1_results/20261015_072525/performance_matrix_all_models.csv); the earlier Qwen-only `performance_matrix.csv` remains a historical artifact.

From the repository root:

```bash
bash exercises/exercise1_baselines/runners/run_e1_step01_naive_baselines.sh --dry-run
bash exercises/exercise1_baselines/runners/run_e1_step01_naive_baselines.sh
python3 analysis/summarize_e1_baselines.py --input exercise1_results/20261015_072525 --output experiment_results/refactored_v1/analysis/e1
```

The runner uses [shared profiles](../common/profiles.py), retains four-worker task grouping, and writes fresh evidence under `experiment_results/refactored_v1/e1_naive/`. It does not repeat the old experiment unless explicitly invoked. Model startup/switching and native episode limits can make this sweep much longer than the joint follow-up; its eight-hour invocation limit is resumable.

## Step 02: supplied CoT reference, incomplete broader sweep

1. **Observed problem.** A direct action gives no explicit intermediate reasoning, but adding reasoning can introduce verbose or malformed outputs.
2. **Hypothesis.** Asking the model to reason before its action may help; its output and token costs must be measured too.
3. **Change.** BALROG's supplied CoT instructions are `First think about what's the best course of action step by step.` followed by `Finally, provide a single output action at the end of the message in the form of: ACTION: <action>`. CoT history is retained (`remember_cot=True`, one reasoning-history entry); output cap 512.
4. **Experiment.** The larger Qwen4 sweep under [20261014_221758](../../exercise1_results/20261014_221758) started with the same E1 task panel, model defaults, and native limits. One CoT repetition completed 180 episodes. Repetition two has 65 saved episodes (50 BabyAI, 10 Crafter, 5 BabaIsAI), so it is **incomplete**; repetition three was not run. This timestamp also contains an earlier complete naive sweep; it is not pooled into the primary three-model table.
5. **Observation.** These files cannot support a three-complete-repetition CoT comparison. The later **six-episode development** reference solved 5/6, equaling compact planning, but produced 22 invalid proposals and substantially more output tokens. See [E2's matched comparison](../exercise2_planning/README.md).
6. **Decision.** Keep partial CoT evidence explicit. No claim that compact planning beats CoT generally; few-shot, robust naive, and robust CoT were not evaluated.

```bash
bash exercises/exercise1_baselines/runners/run_e1_step02_cot_baselines.sh --dry-run
bash exercises/exercise1_baselines/runners/run_e1_step02_cot_baselines.sh
python3 analysis/summarize_e1_baselines.py --input exercise1_results/20261014_221758 --output experiment_results/refactored_v1/analysis/e1_partial_cot
```

This reproduction profile requests three fresh Qwen4 CoT repetitions; it does not label the historic incomplete run complete. `run_status.csv` lists complete, incomplete, and unrun repetitions separately. The analyzer accepts independent `--input` and `--output` paths and never launches models.

## Inspected baseline behavior and fair comparisons

The later matched naive control supplies clear successes and failures: Qwen4 `goto` seed 12 reached its yellow key in four actions; `pickup` seed 11 alternated turns for 32 actions. These belong to the E2/E3 **32-action protocol**, not the native-budget E1 table. Their logs and full trajectories are indexed by [the initial per-episode table](../../experiment_results/episodes.csv). The [E2 guide](../exercise2_planning/README.md) explains why the baseline is repeated there.

Raw E1 trajectories are retained under the original model/agent/run/environment/seed paths. Early E1 runs did not save full request JSONL; their `Reasoning` and action columns are the available evidence. New E1 runs save complete raw calls, including CoT, without changing the supplied action parser. The lack of early raw prompts is recorded rather than reconstructed as original evidence.
