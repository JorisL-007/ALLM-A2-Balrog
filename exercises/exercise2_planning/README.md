# Exercise 2 — compact planning and grounded actions

[Overview](../../README.md) · [E1 baseline context](../exercise1_baselines/README.md) · [E3 reflection](../exercise3_reflection/README.md) · [Experiment index](../../research/experiment_index.csv)

## Step 01: develop a compact planner

### Supplied plan → compact subgoals, observed facts, expected outcome

1. **Observed problem.** In supplied-planner `goto` seed 3, invalid proposals on steps 11–26 became fallback forward moves into an unchanged front wall. In `pickup` seed 2 the agent spent ten actions with the key directly ahead before collecting it. `PLAN: No changes to the plan` could persist during failure. Inspect the [supplied failure](../../deliverables/Ass2-2/evidence/e2_supplied_seed_3_goto/goto_run_00.csv).
2. **Hypothesis.** A shorter plan, observation-supported memory, and a predicted visible result might make state and intent easier to use.
3. **Change.** Replace free-form `PLAN` with four short output fields shown below. Keep up to two fact phrases only if they occur in an observation; truncate plan to 180 characters, each fact to 100, expectation to 120. Reset these at episode boundaries. Two subgoals are requested in the prompt, not enforced by a semantic parser. Facts can be true historical observations while relative distances have become stale.
4. **Experiment.** Qwen3-4B, `goto`/`pickup`, seeds 1–3, 32 actions, history 16, temperature 0, one worker. Supplied and compact agents: 256 output tokens. Built-in CoT reference: 512. See [recorded settings](../../experiment_results/e2_compact_v1/settings.txt) and the command below.
5. **Observation.** Supplied solved 4/6; compact pilot solved 5/6, with mean steps 16.7 → 9.7 and approximately 12.9k → 6.7k total tokens per episode. [Pickup seed 3](../../deliverables/Ass2-2/evidence/e2_compact_v1_seed_3_pickup/pickup_run_00.csv) became an eight-action success: approach red box, collect from one step ahead. But `goto` seed 1 became a failure, alternating turns between the same yellow-key views. CoT also solved 5/6 but used 10.2k input + 3.4k output tokens per episode and made 22 invalid proposals. Its seed-1 pickup failure is linked by the index.
6. **Decision.** Keep compact planning as the reasoning base; do not attribute the package effect to a single field or claim superiority to CoT from six development episodes.

The preserved [supplied reference](agents/history/supplied_planning_v1.py) and [compact pilot](agents/history/compact_planning_pilot_v1.py) retain their original prompt wording. The [accepted planner](agents/compact_planning_frozen_v1.py) is the later frozen actor shared with E3. Its central instructions are:

```text
Compare the expected result with the CURRENT observation before choosing an action.
Keep at most two short subgoals. Change them when the observation disproves them.
Only keep facts copied exactly from an observation; old relative distances are not current positions.
In BabyAI, a target one step forward can block movement: use "pick up" when facing an item to collect it.
...
SUBGOALS: <updated subgoals, or same>
FACTS: <up to two short exact observation phrases separated by ;, or same>
EXPECTED: <visible result expected from the next action>
ACTION: <one allowed action>
```

### Rejected stricter-expectation trial

1. **Observed problem.** Vague expectations such as “yellow key in front” did not stop a turn cycle.
2. **Hypothesis.** Requesting a specific observable change and avoiding a turn that restores the previous view might help.
3. **Change.** A stricter expectation instruction was tried. **Logs available, source unavailable:** no separate source checkpoint was saved, so no version purporting to reproduce that source is provided.
4. **Experiment.** Same six development episodes, task seeds, and budgets; historical condition `e2_compact_v2`.
5. **Observation.** Success fell to 3/6, mean 17.5 actions. In [seed-1 calls](../../experiment_results/e2_compact_v2/seed_1/llm_calls.jsonl), the model predicted a distance change while repeatedly walking into an unchanged front wall.
6. **Decision.** Reject and revert. Keep the raw evidence and source limitation explicit.

From the repository root:

```bash
bash exercises/exercise2_planning/runners/run_e2_step01_planning_development.sh --dry-run
bash exercises/exercise2_planning/runners/run_e2_step01_planning_development.sh
python3 analysis/summarize_e2_e3_development.py --output experiment_results/refactored_v1/analysis/development
```

The runner reproduces the three available checkpoints/reference conditions (`e2_supplied`, `e2_compact_v1`, `cot_reference`). It does not invent a runnable stricter-expectation version. New outputs go under `experiment_results/refactored_v1/e2_development/`.

## Step 02: freeze and evaluate

1. **Observed problem.** Small development panels varied sharply; the successful compact pilot still made geometry errors.
2. **Hypothesis.** Compact planning should improve matched unseen BabyAI missions if its development gain generalizes.
3. **Change.** Freeze the accepted planner and prompts; compare with the supplied naive control. E3 evaluates the same actor with critique enabled.
4. **Experiment.** Qwen4, `goto`/`pickup`/`open`, new seeds 11–15, 32 actions. History 16, temperature 0, output caps naive 128 / structured 256. Additional transfer: Crafter `default`, seeds 11–13, 50 actions; other models `goto`/`pickup`, seeds 11–13, 32 actions. Five new seeds per primary task were chosen from measured 21–27-second development panels and large binary outcome SDs. These are limited studies, separate from E1's native budgets.
5. **Observation.** The table below shows a Qwen4 gain versus naive, but substantial latency cost and poor transfer. Planning gained five paired successes and lost one against naive. True memory did not guarantee appropriate action: [open seed 14 success](../../deliverables/Ass2-2/evidence/heldout_qwen_plan_seed_14_open/open_run_00.csv) retained the carried blue key but wasted toggles; [seed 15 failure](../../deliverables/Ass2-2/evidence/heldout_qwen_plan_seed_15_open/open_run_00.csv) kept toggling a distant door.
6. **Decision.** Keep as a frozen control; avoid claiming general efficacy or universally acceptable cost. Develop grounded actions only in a subsequent, separately seeded study.

| Initial Qwen4 BabyAI | Successes / 15 | Progress mean ± episode SD | Mean steps | Mean input / output tokens | Panel time |
|---|---:|---:|---:|---:|---:|
| Naive | 2 | 13.33% ± 35.19% | 30.13 | 17,909 / 89 | 25 s |
| Frozen compact planning | 6 | 40.00% ± 50.71% | 28.33 | 21,573 / 1,201 | 142 s |
| Frozen action critique (E3) | 6 | 40.00% ± 50.71% | 26.13 | 19,749 / 1,220 | 146 s |

Crafter naive progression was 10.61% ± 6.94% versus 4.55% ± 0 for planning. Naive earned 2/4/1 achievements by seed; planning earned 1/1/1. In one failure, planning repeatedly used `Do` facing grass while expecting wood. Other-model success counts (six episodes each) were Qwen14 **4 naive / 1 planning / 2 critique**, Phi **0 / 2 / 1**. See [the complete initial matrix](../../experiment_results/performance_matrix.csv) and [E3's transfer interpretation](../exercise3_reflection/README.md).

```bash
bash exercises/exercise2_planning/runners/run_e2_step02_planning_evaluation.sh --dry-run
bash exercises/exercise2_planning/runners/run_e2_step02_planning_evaluation.sh
python3 analysis/summarize_e2_e3_evaluation.py --output experiment_results/refactored_v1/analysis/evaluation
```

The profile includes all recorded E2 evaluation models and tasks. Use `--model qwen4-instruct --out <fresh-directory>` for a Qwen-only subset. Model labels and exact historical tags are in [historical_profiles.json](../common/historical_profiles.json).

## Joint E2/E3 follow-up, step 03: corrected guide and narrow pickup rule

1. **Observed problem.** BALROG's guide said `pick up: pick up the object below you` and suggested `toggle` for pickup. The installed MiniGrid action acts on the **front cell**. A planner hint had not reliably overcome this contradiction.
2. **Hypothesis.** Correcting instructions might help; if the model still walks into the exact mission target, a narrow substitution might correct that immediate error.
3. **Change.** [Grounded planning](agents/grounded_planning_v1.py) changes only custom-agent instructions to `pick up: pick up a portable object in the cell immediately in front of you` and `Use 'pick up' for a portable object immediately in front of you. Use 'toggle' to open or close a door or interact with a switch in front of you.` Optional pickup rule: mission begins `pick up the <color> <object>`; that exact object is observed `1 step forward`; agent is not carrying it; proposal is `go forward` or `toggle`. Only then submit `pick up`. Turns and other objects remain model decisions. Compound missions beginning with `pick up` can qualify; those beginning `go to` cannot.
4. **Experiment.** Pilot seeds 20–24, `goto`/`pickup`/`open`, 32 actions, five conditions (plan, guide, guide+rule, guide+goal-state, guide history32), 75 episodes. Temperature 0, history16 except history32, output256. Goal-state is described in [E3](../exercise3_reflection/README.md).
5. **Observation.** Pickup successes /5: 4 plan, 1 guide, 2 rule, 2 goal-state, 1 history32; all `open` conditions 0/5. [Rule seed 24](../../experiment_results/followup_pilot/episodes/babyai/pickup/guide_gate/seed_024/rep_0/llm_calls.jsonl) replaced forward with pickup facing the grey box and succeeded in 5 steps, versus 17 guide-only. [Guide seed 20 failure](../../experiment_results/followup_pilot/episodes/babyai/pickup/guide/seed_020/rep_0/llm_calls.jsonl) remained a turn loop. Seed 23 had different responses to identical prompts and **zero rule substitutions** despite different outcomes. History32's largest call used 1,689 input tokens, within context8192. Full pilot time: 816 seconds.
6. **Decision.** The local mechanism works, but success-rate benefit is uncertain. Freeze code and prompts, then evaluate on distinct seeds. Equalize final naive output cap to 256. Preserve preflight/summary-fix directories; no held-out-driven agent changes.

```bash
bash exercises/common/run_e2_e3_step03_followup_pilot.sh
```

## Joint follow-up, step 04: paired held-out evaluation

One coordinated schedule preserves matched conditions. All use Qwen4, temperature0, output256, context8192, one worker. Histories are 16 unless explicitly 4/32.

| Panel | Conditions | Tasks/seeds | Budget | Episodes |
|---|---|---|---:|---:|
| Primary | naive, plan, E3 critique, guide h4/h16/h32, guide+rule, guide+goal-state | `goto`, `pickup`, `open`; 30–89 | 32 | 1,440 |
| Broader task transfer | plan, guide, guide+rule, guide+goal-state | all five BabyAI tasks; 100–111 | up to 128, native termination applies | 240 |
| Environment transfer | plan, goal-state | Crafter `default`; 120–129 | 50 | 20 |
| Repeatability | plan, guide+rule | `goto`, `pickup`; 30–34; three extra repeats | 32 | 60 |

```bash
bash exercises/common/run_e2_e3_step04_followup_evaluation.sh
python3 analysis/analyze_e2_e3_followup.py --input experiment_results/followup_v1 --output experiment_results/refactored_v1/analysis/followup
```

This new invocation reproduces the protocol in a separate namespace; the original study is already complete. Runtime, eight-hour resume behavior, and `--dry-run` are explained in [the root guide](../../README.md).

| Primary condition | goto /60 | pickup /60 | open /60 | Total tokens | Minutes |
|---|---:|---:|---:|---:|---:|
| Naive | 28 | 26 | 4 | 2.46M | 7.5 |
| Frozen plan | 40 | 34 | 1 | 3.21M | 25.9 |
| Frozen E3 critic | 45 | 35 | 0 | 3.24M | 27.7 |
| Guide, history4 | 39 | 33 | 13 | 2.14M | 25.0 |
| Guide, history16 | 40 | 29 | 3 | 3.45M | 27.3 |
| Guide, history32 | 38 | 26 | 2 | 4.04M | 27.3 |
| Guide + pickup rule | 38 | 29 | 5 | 3.42M | 26.6 |
| Guide + goal-state | 39 | 29 | 3 | 3.71M | 37.3 |

**Observation and decision — corrected guide:** The primary paired pickup comparison was −8.3 percentage points, 95% interval [−25.0, +8.3], 10 gains / 15 losses / 35 ties, exact McNemar p=0.424. No supported improvement. Inspect a [guide gain, seed32](../../experiment_results/followup_v1/episodes/babyai/pickup/guide/seed_032/rep_0/llm_calls.jsonl) and [guide loss, seed34](../../experiment_results/followup_v1/episodes/babyai/pickup/guide/seed_034/rep_0/llm_calls.jsonl); the latter fails through navigation despite correct instructions.

**Observation and decision — pickup rule:** 29/60 equals guide-only; 3 gains / 3 losses / 54 ties, p=1. There were 15 substitutions, all followed by carrying the target and success. [Seed33](../../experiment_results/followup_v1/episodes/babyai/pickup/guide_gate/seed_033/rep_0/llm_calls.jsonl) clearly shows model forward → harness pickup → executed pickup. Keep the narrow mechanism as an ablation, not a demonstrated overall success improvement. Outcomes also differed where the rule never fired; selected intervention cases are not a causal success-rate estimate.

**Observation and decision — history:** History4 versus guide16: +7.2 points across three tasks, 95% seed-block interval [0, +14.4], permutation p=0.080, Holm p=0.320. Tokens fell about 38%. History32: −3.3 points, interval [−8.9, +1.7], Holm p=0.941, more tokens. The `open` history4 gain (13/60 versus 3/60) is exploratory: 12 gains/2 losses, unadjusted McNemar p=0.0129. [Seed35 success](../../experiment_results/followup_v1/episodes/babyai/open/guide_h4/seed_035/rep_0/llm_calls.jsonl) collects the purple key and opens in nine actions; [history16 failure](../../experiment_results/followup_v1/episodes/babyai/open/guide/seed_035/rep_0/llm_calls.jsonl) carries the key but walks away. Independent replication on new fixed door seeds is the most promising next experiment, not a settled finding.

### Broader-task observations and task floors

| Successes /12 per task | goto | pickup | open | putnext | pick_up_seq_go_to | Total /60 |
|---|---:|---:|---:|---:|---:|---:|
| Plan | 10 | 7 | 1 | 0 | 2 | 20 |
| Guide | 8 | 8 | 1 | 0 | 4 | 21 |
| Rule | 6 | 7 | 1 | 0 | 6 | 20 |
| Goal-state | 7 | 4 | 1 | 0 | 1 | 13 |

The compound [plan seed100](../../experiment_results/followup_v1/episodes/babyai/pick_up_seq_go_to/plan/seed_100/rep_0/llm_calls.jsonl) collected its blue ball at action105 and reached the purple key at107. This cannot be evaluated fairly at a 32-action cap. All four conditions failed all `putnext` episodes; [plan seed100](../../experiment_results/followup_v1/episodes/babyai/putnext/plan/seed_100/rep_0/llm_calls.jsonl) shows turns and ineffective forward moves. The rule made five broader-panel substitutions, one on a compound mission beginning `pick up`. Compound seed102's rule condition succeeded **without substitution** because its mission began `go to`; do not attribute that success to the rule. Keep task floors and negative results; no post-evaluation revision was made.

## Variance and statistical interpretation

BabyAI outcomes are binary, so an episode SD near 0.5 is expected. More distinct seeds narrow uncertainty in the *paired effect*; they do not make binary SD intrinsically small. Pair each condition on the same task seed. The primary pickup test uses exact McNemar on discordant outcomes and 10,000 paired-seed bootstrap samples. Secondary multi-task intervals resample entire seed blocks; permutation tests flip block signs (100,000 samples). RNG seed0 and original algorithms are preserved. Four secondary tests receive Holm adjustment. [Paired results](../../experiment_results/followup_v1/paired_comparisons.csv).

The 60 repeatability episodes are separate from the 60-seed effect estimate. Within three consecutive repeats success stayed constant, but versus the earlier primary run it changed in 3/20 task-seed-condition cells; action sequences differed in12/20. [Primary pickup31](../../experiment_results/followup_v1/episodes/babyai/pickup/plan/seed_031/rep_0/llm_calls.jsonl) and [repeat1](../../experiment_results/followup_v1/repeat/babyai/pickup/plan/seed_031/rep_1/llm_calls.jsonl) had identical first prompts but different first actions and success outcomes. Distinct runs of the same seed are not pooled as independent task evidence. Intervals estimate one rollout per condition and do not separately model backend/time variation.
