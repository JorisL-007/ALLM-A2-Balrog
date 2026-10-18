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

Thus the complete three-model naive sweep contains **1,620 episodes**. TextWorld, MiniHack, and NetHack were not executed. The separately imported teammate contribution below adds cloud-model baselines; its settings and denominators differ from this local sweep.

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

## Cloud-model contribution — teammate

This is a separate Exercise 1 contribution from [Ovitch87/AgenticBALROG](https://github.com/Ovitch87/AgenticBALROG/tree/dc8b49f2848f4b35df87e727f21319f9bcead036), imported on 18 October 2026 at commit `dc8b49f2848f4b35df87e727f21319f9bcead036`. Attribution is to the teammate's GitHub account; personal names remain report placeholders. The [source and result snapshot](../../research/cloud_model/upstream_snapshot.tar.gz) preserves the original scripts, BALROG implementation, configuration, all committed results, and MIT license. [Provenance](../../research/cloud_model/provenance.json) records settings and archive hash; the [member manifest](../../research/cloud_model/snapshot_manifest.csv) hashes every archived file. We did not execute the imported scripts or merge their evaluator into the local implementation.

### Question and observations

1. **Observed problem.** Local baseline performance varied greatly by model; the original local sweep did not address the assignment's cloud-model comparison. Imported GPT pickup trajectories also show wall collisions and alternating turns.
2. **Hypothesis.** Cloud models may handle these tasks better, and supplied CoT may help them plan longer Crafter sequences. This is the integration's organizing question, not a claim that the teammate preregistered it.
3. **Change.** The teammate used BALROG's supplied naive and CoT agents. CoT adds the same step-by-step instruction quoted in Step 02, retains one reasoning-history entry, and increases the output cap from 128 to 512. It does not use our compact planner or critic.
4. **Experiment.** GPT-4.1 mini (`gpt-4.1-mini-2025-04-14`) naive: all five BabyAI tasks, seeds 1–25 (125 episodes); Crafter `default`, seeds 1–10 (10); all 40 BabaIsAI tasks, seeds 1–5 (200 according to summaries). GPT CoT and Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) naive/CoT: Crafter, ten episodes each, nominal seeds 1–10. All use text history 16, no images, native action limits, and four evaluator workers. GPT temperature is 1.0; Gemini temperature and thinking budget are left at provider defaults. Exact task names and commands are preserved in the provenance and archived `Exer1_scripts/`. These are single panels, not three repeated runs.
5. **Observation.** All six reported cloud scores reproduce at two decimal places. GPT's Crafter mean rises by 17.27 percentage points with CoT; Gemini's falls by 1.82. These are descriptive differences, without established paired worlds or a common output-token cap. The table does not establish a causal CoT advantage, or compare cloud performance with our E2/E3 agents under matched conditions.
6. **Decision.** Keep as separately labeled cloud baseline evidence. Use a future matched model-by-agent experiment to investigate whether the custom harness transfers. No new inference was performed during integration.

| Model / hosting | Agent | BabyAI | Crafter | BabaIsAI |
|---|---|---:|---:|---:|
| GPT-4.1 mini / cloud | Naive | 41.60% (125) | 17.27% (10) | 27.50% (200, summaries only) |
| GPT-4.1 mini / cloud | CoT | — | 34.55% (10) | — |
| Gemini 3.5 Flash-Lite / cloud | Naive | — | 42.73% (10) | — |
| Gemini 3.5 Flash-Lite / cloud | CoT | — | 40.91% (10) | — |
| Qwen3.5-2B / **teammate local** | Naive | **24.00% reported; not verified** | — | — |

Parentheses are episodes, not repetitions. The local Qwen row is included to preserve the supplied table, and is explicitly **not a cloud result**. Its exact serving ID is `qwen3.5:2b`, with one worker, temperature 1.0, 128 output tokens, history 16, and reasoning disabled. The snapshot contains **28 successes in 109 completed episode records (25.69%)**, plus one CSV without an episode JSON. Seeds 1–21 are complete, seed 22 has four completed tasks, and seeds 23–25 are absent. The planned denominator is 125; missing episodes cannot be counted as failures or inferred to reconcile the reported 24%.

### Evidence quality, cost, and comparability

- **BabaIsAI:** five seed summaries, each with 40 per-task scores, reproduce 27.50%. No individual episode JSON or trajectory CSV was committed. This is summary-supported evidence, with no manual trajectory verification.
- **Crafter seeds:** the scripts set `envs.env_kwargs.seed` but leave `envs.crafter_kwargs.seed=null`. The imported factory constructs Crafter before evaluator reseeding. A matching folder label therefore does not verify an identical world; the stored CoT starting descriptions also differ between conditions. We do not calculate a paired test from these labels. A future study should explicitly seed the constructor and save initial observations.
- **Raw logs:** the snapshot contains no complete request/reply JSONL. Original CSVs preserve post-action observations, proposals, and CoT reasoning. In this evaluator, `Action` is reassigned to `response.completion` after execution; it is **not a separate record of the finally executed fallback action**. Neither full prompts nor the missing action distinction have been reconstructed as original evidence.
- **Costs:** [the matrix](../../research/cloud_model/analysis/performance_matrix.csv) reports saved input/output totals and mean steps. GPT Crafter naive/CoT used 3,373,053/4,859,513 input tokens and 2,740/397,761 output tokens across ten episodes each; mean lengths were 217.0/262.9 actions. Gemini used 3,220,397/3,774,080 input and 3,870/127,216 output tokens. The imported Gemini client adds candidate and reasoning token counts when the provider supplies them. Per-call responses, retry costs, invoices, and elapsed times are unavailable; these totals cannot establish total API spend.
- **Source version:** the archive pins the repository at import. The imported runs have no per-run source hashes or dependency lock, so the snapshot is not proof that every historical episode used exactly that code revision.
- **Variance:** the analyzer reports episode SD separately (Crafter: GPT naive 8.78 pp, GPT CoT 16.49 pp; Gemini naive 14.72 pp, Gemini CoT 13.55 pp). These are not the three-run SDs in the local E1 table. Different task panels, generation settings, and action budgets prevent pooling with the local or follow-up studies.

### Manually inspected examples

The linked files are exact copies of archived trajectories; [sample provenance](../../research/cloud_model/sample_index.csv) records original paths and hashes. CSV steps are zero-based. We read the GPT pickup trajectories in full, and inspected the opening, crafting transitions, and terminal windows of the Crafter examples; these are qualitative examples, not a systematic sample of all trajectories.

| Example | What the saved trajectory shows | Interpretation |
|---|---|---|
| [GPT pickup seed 1, success](../../research/cloud_model/samples/gpt_pickup_success_seed1.csv) | Alternating turns at steps 20–30; yellow key one step ahead after 37; blocked forward at 38; `pick up` at 39 succeeds. | A cloud model can recover late while retaining the same pickup misunderstanding. Under a 32-action cap this observed path would not have completed. |
| [GPT pickup seed 10, failure](../../research/cloud_model/samples/gpt_pickup_failure_seed10.csv) | Red ball remains diagonally adjacent; right/left alternation from steps 10–63; no pickup before native termination at 64 actions. | More capable base models still exhibit the loop our critic targets. |
| [GPT Crafter CoT seed 1](../../research/cloud_model/samples/gpt_crafter_cot_seed1.csv) | Table at 16 and wood pickaxe at 17 each earn an achievement; eight achievements overall. At 392–393, reasoning insists eating the visible plant will help, but `Do` does not restore food and the agent dies. | A useful crafting sequence and an unsupported recovery belief coexist in one episode. |
| [GPT Crafter naive seed 1](../../research/cloud_model/samples/gpt_crafter_naive_seed1.csv) | One achievement; final `Do` actions face stone, followed by death at step 66. | Low-progress contrast, without claiming a matched-world causal comparison. |
| [Gemini Crafter CoT seed 1](../../research/cloud_model/samples/gemini_crafter_cot_seed1.csv) | At 5, `Place Table` leaves no visible table and earns no reward; at 6, reasoning says a table was successfully placed. Later, stone pickaxe at 103 and furnace at 130 succeed; nine achievements overall. | Reasoning can remember an intended action as an accomplished fact. This is concrete motivation for checking observed outcomes. |
| [Gemini Crafter naive seed 1](../../research/cloud_model/samples/gemini_crafter_naive_seed1.csv) | Seven achievements; terminal observation says the agent is sleeping with a nearby zombie. | Repeated waiting while asleep is not, by itself, evidence of an unproductive reasoning loop. |

### Reproduce the import analysis

This command reads the pinned archive, verifies its hash and episode/trajectory counts, and writes separate tables. It never calls an API, starts a model, or overwrites the local E1/E2/E3 matrices:

```bash
python3 analysis/summarize_e1_cloud_models.py \
  --input research/cloud_model/upstream_snapshot.tar.gz \
  --output research/cloud_model/analysis
```

Outputs: [performance matrix](../../research/cloud_model/analysis/performance_matrix.csv), [episode records](../../research/cloud_model/analysis/episodes.csv), [per-task results](../../research/cloud_model/analysis/per_task.csv), and [coverage audit](../../research/cloud_model/analysis/verification.json). The archive also preserves the excluded one-episode GPT-4o-mini setup check. The original launch commands remain archived for provenance; no new cloud runner or experiment has been added. The [experiment index](../../research/experiment_index.csv) identifies each imported condition separately.

## Inspected baseline behavior and fair comparisons

The later matched naive control supplies clear successes and failures: Qwen4 `goto` seed 12 reached its yellow key in four actions; `pickup` seed 11 alternated turns for 32 actions. These belong to the E2/E3 **32-action protocol**, not the native-budget E1 table. Their logs and full trajectories are indexed by [the initial per-episode table](../../experiment_results/episodes.csv). The [E2 guide](../exercise2_planning/README.md) explains why the baseline is repeated there.

Raw E1 trajectories are retained under the original model/agent/run/environment/seed paths. Early E1 runs did not save full request JSONL; their `Reasoning` and action columns are the available evidence. New E1 runs save complete raw calls, including CoT, without changing the supplied action parser. The lack of early raw prompts is recorded rather than reconstructed as original evidence.
