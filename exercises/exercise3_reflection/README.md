# Exercise 3 — feedback-triggered critique and goal-state reflection

[Overview](../../README.md) · [E2 planner and joint protocol](../exercise2_planning/README.md) · [Experiment index](../../research/experiment_index.csv)

## Step 01: a critique must affect the next action

### Plan-only critique → action-selecting critique

1. **Observed problem.** Compact planning `goto` seed 1 alternated left/right between the same two views even though actions and observations were in history.
2. **Hypothesis.** A separate critique, triggered by concrete evidence of ineffective behavior, might revise the plan and break the loop.
3. **Change.** [Plan critique v1](agents/history/plan_critique_v1.py) requested a diagnosis, revised subgoals, and one lesson; the ordinary actor still chose the action. It explicitly said `Do not choose the next action here.` The accepted [action critique v2](agents/action_critique_v2.py) instead says `Choose one allowed next action that tests a different route; do not repeat the loop's last action.` Its structured reply is shown below.
4. **Experiment.** Qwen4, `goto`/`pickup`, seeds 1–3, 32 actions, temperature0, history 16, output256, one worker. Compare the same planning actor with critique off and on. Main historical conditions: `e3_off_dev2`, `e3_on_dev2`, `e3_on_dev3`.
5. **Observation.** Plan-only critique stayed at 5/6; three calls added about 590 input and 72 output tokens per episode. At step 4 of [plan-critique seed 1](../../experiment_results/e3_on_dev2/seed_1/llm_calls.jsonl), the critic identified the cycle but the actor still turned right. Action-selecting critique solved 6/6 with two critic calls total. [Action-critique seed 1](../../experiment_results/e3_on_dev3/seed_1/llm_calls.jsonl) selected forward at steps 4 and 12, changed the repeated view, and finished at step 18.
6. **Decision.** Keep action-selecting critique for held-out evaluation; the development gain is not a general performance estimate. The plan-only design remains an explicitly superseded checkpoint.

```text
Critique this concrete failure: <observed trigger>
Last actions: <up to five executed actions>
Current subgoals: <stored plan>
Use the observed evidence only. Explain why the last plan or action failed.
Choose one allowed next action that tests a different route; do not repeat the loop's last action.
Keep the answer short. A wall more than one step forward does not block a single forward step.
CRITIQUE: <one sentence>
REVISED SUBGOALS: <up to two short subgoals>
LESSON: <one short instruction for the remainder of this episode>
EXPECTED: <one visible result expected from your action>
ACTION: <one allowed action>
```

The trigger requires either two identical interaction/forward actions leaving a front obstacle and the view unchanged, or four alternating turns returning to the same two distinct views. **Repeated action or zero reward alone is insufficient.** At most three critiques per episode, at least six steps apart. Retain at most two lessons, each 120 characters; reset them with all episode memory. If the critic supplies an action, use it directly. If it lacks an `ACTION` field, call the ordinary actor and add **both** calls' tokens. A field containing an invalid action still goes through BALROG's usual fallback; extraction is not action validation.

### Rejected broader door trigger

1. **Observed problem.** In `open` seed 1 the agent picked up a purple key, then toggled while the locked door was left and ahead rather than directly in front. The narrow trigger often did not fire.
2. **Hypothesis.** Triggering after two unchanged toggles with a locked door visible might interrupt this failure earlier.
3. **Change.** [Door-trigger critique v3](agents/history/door_trigger_critique_v3.py) broadened the trigger; the critic and bounded lesson memory remained.
4. **Experiment.** Three `open` development seeds 1–3, same 32-action protocol, condition `e3_on_open_dev2`; compare the narrow-trigger probe `e3_on_open_dev` and planning `e3_off_open_dev`.
5. **Observation.** Broader trigger fired earlier (step 8 in seed 1), but still solved 0/3 and used eight critiques. [Raw calls](../../experiment_results/e3_on_open_dev2/seed_1/llm_calls.jsonl) include a claim that the locked door was “not interactable” and forward recommendations that did not face it.
6. **Decision.** Reject broader trigger and retain v2. Diagnosing an apparent failure is not sufficient if the proposed revision is wrong.

```bash
bash exercises/exercise3_reflection/runners/run_e3_step01_critique_development.sh --dry-run
bash exercises/exercise3_reflection/runners/run_e3_step01_critique_development.sh
python3 analysis/summarize_e2_e3_development.py --output experiment_results/refactored_v1/analysis/development
```

This runner selects available historical variants explicitly. Preliminary `e3_off_dev`/`e3_on_dev` runs have a source hash that does not match any saved checkpoint; they remain evidence-only and are not silently assigned another implementation. See [the index](../../research/experiment_index.csv). The later documented off/on comparison has saved source.

## Step 02: held-out critique evaluation

1. **Observed problem.** Critique helped one development cycle but failed on door geometry.
2. **Hypothesis.** The accepted critic might reduce unproductive action loops and improve new mission outcomes at modest extra cost.
3. **Change.** Freeze v2; compare it with exactly the same actor, memory format, and history with critique disabled.
4. **Experiment.** Qwen4 `goto`/`pickup`/`open`, seeds 11–15,32 actions; Crafter seeds 11–13,50 actions; Qwen14/Phi `goto`/`pickup`, seeds 11–13,32 actions. Settings and naive comparison are in [E2 step02](../exercise2_planning/README.md).
5. **Observation.** Both planning and critique solved 6/15 Qwen4 BabyAI episodes. Critique gained one success and lost one. It took 146 seconds versus 142, with 20 counted critic calls. Alternating-turn windows fell 145→22, blocked-forward pairs rose 37→51, and unchanged-door toggle pairs were 51→48. These overlapping windows describe behavior; they are not independent failed missions.
6. **Decision.** Critique changes behavior at a small additional cost relative to the already expensive planner, but there is no net success gain in this panel. Freeze it as a control for the later joint evaluation.

| Inspected trajectory | What happened |
|---|---|
| [Planning goto11](../../deliverables/Ass2-3/evidence/heldout_qwen_plan_seed_11_goto/goto_run_00.csv) vs [critique goto11](../../deliverables/Ass2-3/evidence/heldout_qwen_reflect_seed_11_goto/goto_run_00.csv) | Critic chose forward after a turn cycle; reflective run reached the ball on action32. |
| [Critique pickup13](../../deliverables/Ass2-3/evidence/heldout_qwen_reflect_seed_13_pickup/pickup_run_00.csv) | Critic recommended forward despite a wall one step ahead; failure persisted. |
| [Phi planning pickup12](../../deliverables/Ass2-3/evidence/cross_phi4_plan_seed_12_pickup/pickup_run_00.csv) vs [Phi critique pickup12](../../deliverables/Ass2-3/evidence/cross_phi4_reflect_seed_12_pickup/pickup_run_00.csv) | Planning eventually collected the adjacent key. Critic turned away and failed. Both began with an invalid compound proposal and executed fallback forward. |

Qwen14 success counts were 4/6 naive, 1/6 planning, 2/6 critique; Phi 0/6, 2/6, 1/6. Some Phi critiques placed fields on one line, so `ACTION` was not extracted and the actor was called afterward. Crafter planning and critique each averaged 4.55% progression, below naive 10.61%; **zero critiques fired** because the triggers use BabyAI action/geometry evidence. This is a scope limitation, not a test showing Crafter critiques are ineffective.

```bash
bash exercises/exercise3_reflection/runners/run_e3_step02_reflection_evaluation.sh --dry-run
bash exercises/exercise3_reflection/runners/run_e3_step02_reflection_evaluation.sh
python3 analysis/summarize_e2_e3_evaluation.py --output experiment_results/refactored_v1/analysis/evaluation
python3 analysis/measure_e2_e3_action_loops.py heldout_qwen_plan heldout_qwen_reflect --output experiment_results/refactored_v1/analysis/loops
```

## Joint follow-up, steps03–04: one-call goal-state reflection

1. **Observed problem.** Extra critique calls can add cost; the model may have enough information to compare state and goal during each ordinary action decision.
2. **Hypothesis.** A short explicit state-versus-goal line might help it notice lack of progress without a separate critic call.
3. **Change.** [Goal-state reflection v1](agents/goal_state_reflection_v1.py) extends corrected-guide planning with the exact line below, before `ACTION`. It uses five response lines and **one call per action**, with the extra-call critic disabled. This is a small **ReflAct-inspired adaptation**, not a reproduction of the publication's full setup. The line is model text, not a verified state evaluator or new persistent memory.
4. **Experiment.** Pilot 75 episodes, then the frozen 1,760-episode joint schedule in [E2](../exercise2_planning/README.md). The prespecified comparison is guide+goal-state versus guide at history 16 on all three primary tasks, with paired seed-block uncertainty and Holm adjustment. Crafter compares frozen planning against goal-state.
5. **Observation.** Pilot `goto` seed 22 succeeded with goal-state where guide failed, but some state lines falsely called distant walls blocking. On held-out tasks, goal-state solved 71/180 versus 72/180 guide-only: −0.6 points, 95% interval [−7.2,+6.1], Holm p=1. Tokens rose 3.45M→3.71M; time 27.3→37.3 minutes. [Goto31 failure](../../experiment_results/followup_v1/episodes/babyai/goto/guide_goal/seed_031/rep_0/llm_calls.jsonl) kept alternating turns while explicitly stating no progress. The [guide control](../../experiment_results/followup_v1/episodes/babyai/goto/guide/seed_031/rep_0/llm_calls.jsonl) eventually succeeded.
6. **Decision.** No evidence that this added line justifies its cost. Retain the version for reproducibility and report negative results; do not revise it after looking at held-out outcomes.

```text
STATE_VS_GOAL: <current observed state; unmet mission requirement; whether the previous action advanced it>
ACTION: <one allowed action>
```

The original paper citation and formal interpretation are retained in [the report](../../report/Ass2Reasoning_report.tex). Existing source notes consulted MiniGrid's action implementation to ground pickup semantics; the follow-up guide correction itself is described in E2.

In the primary panel, the frozen E3 critic solved 80/180 versus 75/180 planning. Alternating-turn windows fell 1,210→156, tokens were 3.24M versus 3.21M, and time 27.7 versus 25.9 minutes. There were 197 reflection-mode calls; all supplied actions directly, so total calls equaled 4,031 executed actions. “Critic call” does not automatically mean an additional call beyond the action count. Its success difference was small and task-dependent (goto 45 vs 40, pickup 35 vs 34, open 0 vs 1); it was not the prespecified primary contrast.

Broader-task goal-state results were 13/60 versus 21/60 guide-only. Crafter ten-seed mean progression was 0.0500 (SD 0.0144) versus 0.0591 (SD 0.0307) planning, paired difference −0.0091 with bootstrap interval [−0.0227, 0]. Goal-state used 734,350 versus 696,416 tokens and 247 versus 185 seconds. On [Crafter122 planning](../../experiment_results/followup_v1/episodes/crafter/default/plan/seed_122/rep_0/llm_calls.jsonl), moving toward a tree earned two achievements; [goal-state122](../../experiment_results/followup_v1/episodes/crafter/default/guide_goal/seed_122/rep_0/llm_calls.jsonl) moved west first and earned one. Keep this negative transfer explicit.

Run the coordinated pilot/evaluation once, rather than launching separate unmatched E2 and E3 follow-ups:

```bash
bash exercises/common/run_e2_e3_step03_followup_pilot.sh
bash exercises/common/run_e2_e3_step04_followup_evaluation.sh
python3 analysis/analyze_e2_e3_followup.py --input experiment_results/followup_v1 --output experiment_results/refactored_v1/analysis/followup
```

See E2 for the exact seeds, task budgets, history comparisons, multiplicity correction, and repeatability findings. The repeated seeds are kept separate from independent task evidence. Full prompts and all critic/actor token totals remain beside every recorded trajectory; [portable raw samples](../../research/raw_llm_samples.jsonl) include both helpful and harmful critiques.

## Step05: compare the frozen critic across models

[The coordinated CoT/model-transfer runner](../../README.md#step-05-cot-control-and-model-transfer) includes `action_critique_v2` with its existing prompts, triggers, bounded lessons, and reflection limits. Qwen's complete BabyAI panel solved 80/180 with critique versus 75/180 planning; the paired +2.8-point effect was uncertain after adjustment, though alternating-turn windows fell from 1,210 to 156. On Luna's completed 180-case BabyAI panel, critique solved **105 versus planning 28**, a paired **+42.8-point** difference, 95% seed-block interval [+33.9, +51.7], Holm-adjusted p=.00012. Turn-loop windows fell from 2,266 to 270. Yet naive solved **128/180** at lower estimated cost: critique recovered from a harmful planner, without surpassing the supplied baseline. [Pickup seed51](../../experiment_results/refactored_v1/model_transfer_luna_v1/gpt6luna-cloud/episodes/babyai/pickup/e3/seed_051/rep_0/llm_calls.jsonl) shows the critic choosing forward after repeated alternating views; [pickup seed64](../../experiment_results/refactored_v1/model_transfer_luna_v1/gpt6luna-cloud/episodes/babyai/pickup/e3/seed_064/rep_0/llm_calls.jsonl) instead cycles through four views with left turns for 32 actions and no critique. OLMo's critique solved the same three cases as planning, and 460/462 critiques returned no parsed action, causing extra actor calls. No frozen Crafter critic triggered. [Completed inspection](../../research/model_transfer_verification/postrun_inspection.txt) links successes, failures, and raw calls; two Luna CoT cases remain missing after provider refusals.

The [OLMo short door pilot](../../research/model_transfer_verification/live/olmo1b-local/babyai_open_e3_seed21/llm_calls.jsonl) exercised one real critique at call6 after repeated forward actions left a wall visible. It returned prose without `ACTION`; the ordinary actor then returned the invalid `pick up the green key.` and BALROG executed fallback forward. Thus eight executed actions used **nine calls**, all counted. The [Crafter pilot](../../research/model_transfer_verification/live/olmo1b-local/crafter_default_e3_seed21/llm_calls.jsonl) repeatedly proposed `Place Furnace` with an explicitly empty inventory; the frozen trigger did not fire. These inspected failures establish logging and expose mechanism limits; they do not estimate the effect of reflection on OLMo.
