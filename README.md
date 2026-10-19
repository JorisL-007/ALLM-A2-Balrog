# BALROG: planning, grounded actions, and reflection

**Research question:** Can a small persistent plan and feedback-triggered critique improve task completion and reduce unproductive loops at an acceptable inference cost?

For result folders, summaries, and raw episode logs, see [Results guide](RESULTS_GUIDE.md).

Read the study in order:

| Stage | Question | Guide and reproduction commands |
|---|---|---|
| Exercise 1 — baselines | How do local and cloud models behave with supplied agents? | [Baseline study](exercises/exercise1_baselines/README.md) |
| Exercise 2 — planning | Do compact plans, observed facts, and better action instructions help? | [Planning study](exercises/exercise2_planning/README.md) |
| Exercise 3 — reflection | Can an explicit critique change an ineffective action? | [Reflection study](exercises/exercise3_reflection/README.md) |

Each guide follows **observed problem → hypothesis → change → experiment → observation → decision**. The larger grounded-action/history/goal-state evaluation came *after* the initial E2/E3 evaluation; it is a joint follow-up, not part of the initial development evidence.

## Findings in brief

- On the initial matched Qwen3-4B BabyAI panel, naive solved **2/15**, compact planning **6/15**, and reflection **6/15**. Planning cost substantially more elapsed time. Reflection reduced alternating turns but did not improve aggregate success.
- On the larger primary panel, naive solved **58/180**, frozen planning **75/180**, and frozen reflection **80/180**. Results depended on task and model; this is not a benchmark-wide win.
- Correcting the pickup guide did **not** improve the prespecified pickup comparison: **29/60 versus 34/60**, paired difference −8.3 percentage points, 95% interval [−25.0, +8.3], exact McNemar p=0.424.
- The narrow pickup rule corrected 15 immediate proposals, all followed by successful pickup, but its overall pickup success rate equaled guide-only. Navigation failures remained.
- The one-call goal-state prompt increased inference cost without a reliable gain. History 4 reduced tokens and showed an exploratory door-task improvement; its prespecified comparison did not survive multiple-comparison adjustment.
- Crafter transfer and limited other-model checks constrain generalization. The earlier matched CoT comparison was a small development reference; the new Step 05 Qwen panel below supplies the larger control. Few-shot and robust variants were not evaluated.
- **Step 05:** on 90 exactly matched Qwen BabyAI task-seed cases, naive solved 27, supplied CoT 41, planning 36, and critique 38. No overall Qwen contrast survived the prespecified twelve-test Holm adjustment. OLMo 1B solved 0/180 naive, 0/90 CoT, and the same 3/180 under planning and critique. On GPT-6 Luna's complete 180-case BabyAI panel, naive solved **128**, planning **28**, and critique **105**; supplied CoT solved **38/89**. Critique repaired many planning loops but remained below naive. [Completed analysis and inspected logs](research/model_transfer_verification/postrun_inspection.txt).
- Luna's paired planning-minus-naive difference was **−55.6 points** (seed-block 95% interval [−63.9, −47.2]); critique-minus-planning was **+42.8 points** ([+33.9, +51.7]). Both survived Holm correction across twelve prespecified tests. CoT was **−24.1 points** versus naive on 87 exactly matched task-seed cases; one CoT seed block is missing after a provider refusal. Crafter's ten-seed mean progression was .168 naive, .055 planning, and .073 critique; no Crafter critic fired.
- The separately imported **cloud-model contribution** verifies GPT-4.1 mini's 41.60% BabyAI score and the reported cloud Crafter scores; GPT's 27.50% BabaIsAI result is supported by summaries only. CoT improved GPT's observed Crafter mean but did not improve Gemini's. These native-budget panels do not test our custom agents and are not pooled with the paired follow-up. [Evidence, settings, and limitations](exercises/exercise1_baselines/README.md#cloud-model-contribution--teammate).

The formal argument, assignment answers, and remaining identity/submission placeholders are in [the LaTeX report](report/Ass2Reasoning_report.tex). Historical metrics remain in [the initial matrix](experiment_results/performance_matrix.csv) and [the follow-up matrix](experiment_results/followup_v1/performance_matrix.csv).

## Critical assessment and proposed next work

**Strengths.** The study follows inspected failures through a specific change to a matched evaluation. Frozen controls, held-out seeds, explicit action/fallback logging, bounded reflection, inference-cost accounting, paired confidence intervals, and negative findings make its central claims auditable. The 1,760-episode follow-up is substantial for this assignment. The imported cloud baselines broaden model coverage while preserving the original local results.

**Limits.** BabyAI uses three short tasks and a 32-action cap. The limited Qwen14/Phi checks, Luna planner regression, and OLMo floor show that agent effects vary sharply by model. Crafter transfer has ten seeds per main condition, and the frozen Crafter critic never triggered in Step 05, so that panel does not test an active reflection intervention. The 256-token cap truncated many Qwen CoT replies; few-shot and robust variants remain untested. Two Luna CoT episodes ended in provider refusals and remain missing, including one BabyAI seed block and one Crafter seed. Fixed seeds do not eliminate inference variation. High episode SD is expected for binary success near 50%; uncertainty in the **paired improvement** is the relevant target.

The CoT-control and model-transfer study is implemented as Step 05 below. Qwen and OLMo completed their schedules; Luna completed every non-refused case. The historical CoT panels were recovered without extending the old sweep.

| Priority | Proposed work | Why it adds evidence |
|---|---|---|
| 1 — quickest, no inference | Historical local CoT panels are now recovered; obtaining the teammate's missing Qwen2B episodes and Baba trajectories remains useful. | The recovery retains all 245 CoT episodes and distinguishes complete environment panels from the partial BabaIsAI repetition. |
| 2 — targeted new development | Investigate the same-direction turn cycle missed by the Luna critic, then freeze a revised trigger before evaluation. | [Pickup seed64](experiment_results/refactored_v1/model_transfer_luna_v1/gpt6luna-cloud/episodes/babyai/pickup/e3/seed_064/rep_0/llm_calls.jsonl) repeats left turns for 32 actions without reflection. |
| 3 — independent replication | Compare naive, planning, and revised critique on new BabyAI seeds and a second capable model. | The completed Luna panel shows a large planner regression despite Qwen's modest gain; new seeds would test stability. |
| 4 — cheap focused replication | Freeze history 4 versus history 16 and test a fresh `open` seed panel. | Directly tests the exploratory cost-saving finding from the earlier follow-up. |
| 5 — Crafter reflection | Develop a Crafter-specific failure trigger before claiming a reflection effect there. | The Step 05 frozen critic made no Crafter reflection calls on any model. |

Step 05 used the existing **60 evaluation seeds per BabyAI task**, with CoT on a fixed half-sized subset. It added the missing Qwen CoT control while reusing compatible Qwen evidence, then applied all four agents to GPT-6 Luna and OLMo. Reusing an existing seed panel tests model transfer without adding independent new Qwen puzzles; the earlier repeatability findings remain a limitation.

## Step 05: CoT control and model transfer

Preview the complete schedule without writes, downloads, or inference:

```bash
bash exercises/common/run_e2_e3_step05_cot_and_model_transfer.sh --dry-run
```

With `OPENAI_API_KEY` exported and the local GPU available, launch all three stages in order:

```bash
nohup bash exercises/common/run_e2_e3_step05_cot_and_model_transfer.sh > model_transfer.log 2>&1 < /dev/null &
```

The fixed schedule contains **1,445 new evaluation episodes**, **550 referenced Qwen controls**, and **18 excluded short pilots**. BabyAI `goto`/`pickup`/`open`: seeds30–89, 32 actions; CoT uses even seeds30–88. Crafter: seeds120–129, 50 actions; CoT uses even seeds120–128. All conditions use temperature0, history16, no images, one worker, and 256 output tokens per call. The frozen planner and action critic keep their existing prompts, limits, and reset behavior. The supplied CoT prompt and its one-entry reasoning history are preserved. CoT truncation is measured rather than prompting it differently.

Outputs are separate under `experiment_results/refactored_v1/model_transfer_luna_v1/`. The earlier unrun GPT-4.1 mini preparation remains at `model_transfer_v1/`; its 550 references have no new evaluation episodes. The script recovers historical CoT panels first, runs Qwen → GPT-6 Luna → OLMo, and analyzes after each stage. The default invocation limit is48 hours, checked between episodes. Repeating the command resumes verified unfinished work except for provider refusals, which need the continuation below. `--hours`, `--limit` (counts new pilots too), `--out`, `--port` (default8097), `--model`, and `--condition` are available. Scientific settings/source changes are rejected; filtered schedules require their own output directory. Only owned servers are stopped. Failed/interrupted attempts are retained. Persistent stage failures allow later models to proceed; exit2 means the selected study remains incomplete.

Two Luna CoT requests received HTTP400 `invalid_prompt`. The source-pinned [Luna continuation](continue_luna_eval.py) kept both raw `.working` attempts and completed every other case. It can be audited with `python3 continue_luna_eval.py --dry-run`; rerunning it skips all verified completions and reports the two refusals as missing evidence.

Cloud generation uses [`gpt-6-luna`](https://developers.openai.com/api/docs/models/gpt-6-luna) and the existing OpenAI Chat Completions client. To retain temperature0 and the 256-token cap, requests set `reasoning_effort="none"` and `max_completion_tokens=256`; the full request is logged. **$10 is a target; execution continues until the cumulative estimate exceeds $20**, then proceeds to OLMo. The guard includes pilots, incomplete attempts, and retries across resumes. Known usage is estimated at the [documented standard rates](https://developers.openai.com/api/docs/models/gpt-6-luna): $0.10/M input and $0.50/M output, without cache discounts; unknown/interrupted requests receive conservative provisional charges. These are estimates, not invoices. `--cloud-target-usd` and `--cloud-stop-usd` change operational limits without changing the scientific panel. All request attempts have full payloads/replies, finish reasons, timing, and available usage in `api_attempts.jsonl`; SDK retries are disabled so BALROG's three attempts remain visible. Requests time out after60 seconds and episodes after30 minutes.

Reused controls retain their original actor/critic logs and measured times. Their older logs lack transport-level audits, so unavailable retry/truncation counts remain blank. Newly incurred cloud spending is calculated from the new request ledger.

OLMo prefers the cached official Q8_0 GGUF, official tokenizer, and native4096 context. Startup incompatibility permits official BF16 before any pilot/evaluation, with missing files downloaded through `hf` into the study's server directory. The selection and artifact hashes are pinned across resumes; quality or evaluation outcomes never trigger a precision switch. Serving requires a free dedicated port at first startup. The script reports pilot-based time/cost estimates; the old5.77-hour Qwen follow-up is not a runtime estimate for this study.

Local setup records also pin vLLM, Transformers, PyTorch, and Python versions. A changed serving environment or model artifact is rejected before restarting inference. The final version-pin check made no model calls; its earlier prepared references and source checkpoint were retained separately before initializing the final study namespace.

Analysis is independent and can recover the old CoT evidence without running models:

```bash
python3 analysis/analyze_e2_e3_model_transfer.py \
  --input experiment_results/refactored_v1/model_transfer_luna_v1 \
  --output experiment_results/refactored_v1/model_transfer_luna_v1/analysis
```

Read `historical_panels.csv`, `historical_matched_panels.csv`, `historical_matrix.csv`, `overall_matrix.csv`, `performance_matrix.csv`, `coverage.csv`, `paired_comparisons.csv`, `status.json`, and `inspection_index.csv`. CoT comparisons restrict controls to its exact seed subset. Four overall BabyAI contrasts per model use the original seed-block bootstrap/permutation methods, seed0, and Holm adjustment across twelve tests; per-task and Crafter scores are descriptive. Reused episode times remain original and are distinguished from newly incurred time. Missing episodes remain missing. The [post-run research note](research/model_transfer_verification/postrun_inspection.txt) records manually read successes and failures; the analyzer itself still labels interpretation pending because it cannot verify that human reading.

Implementation verification previously completed six real Qwen and six real OLMo pilots: **91 actions and92 LLM calls**, preserved in [the pilot inspection table](research/model_transfer_verification/live_inspection.csv). The run also completed all six Luna pilots. Q8 startup failed in vLLM0.31.0, so OLMo's tested precision is **BF16**. The final study has **1,993/1,995 verified evaluation observations**: Qwen665/665, OLMo665/665, and Luna663/665. The two missing Luna CoT episodes are BabyAI pickup seed50 and Crafter seed122, both provider refusals with raw calls retained. Estimated cloud expenditure is **$1.5375** including pilots and uncertain rejected requests. The read-only analyzer reproduced all twelve output files byte-identically from the completed evidence. [Paired results](experiment_results/refactored_v1/model_transfer_luna_v1/analysis/paired_comparisons.csv) and [manual trajectory notes](research/model_transfer_verification/postrun_inspection.txt) distinguish successes, failures, and missing episodes.

## Where things live

```text
exercises/exercise1_baselines/  README and numbered baseline runners
exercises/exercise2_planning/  README, planning agents, historical variants, runners
exercises/exercise3_reflection/ README, reflection agents, historical variants, runners
exercises/common/             variant selection, raw logging, profiles, shared execution
analysis/                     read-only aggregation and statistical analysis
research/                     experiment index, portable raw calls, refactor verification
research/cloud_model/         pinned teammate source/results, provenance, separate analysis
report/                       canonical LaTeX report
integration/                  patch against the nested BALROG checkout
archive/refactor_20261018/     original source/docs and filename migration map
```

[Experiment index](research/experiment_index.csv) links historical condition names, named variants, settings, checkpoints, results, and inspected trajectories. Rejected/superseded source is under each exercise's `agents/history/`. The strict-expectation trial has **logs available, source unavailable**. Its original implementation has not been reconstructed.

Existing `exercise1_results/`, `experiment_results/`, raw-log samples, manifests, and submission evidence retain their original paths. All new runs use `experiment_results/refactored_v1/`. Frozen data are never resumed with refactored code.

## Setup and integration

Use the existing `balrog` and `vllm` conda environments, locally cached models, and NVIDIA GPU. The development model is `Qwen/Qwen3-4B-Instruct-2507-FP8`. The baseline profiles also use `Qwen/Qwen3-14B-AWQ` and `microsoft/Phi-4-mini-instruct`. Earlier runners use cached models only (`HF_HUB_OFFLINE=1`). Step05 may download missing pinned OLMo artifacts before serving; it requires the `hf` CLI and an exported API key for its cloud stage.

The nested BALROG checkout is based on commit `b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5`. On a **clean checkout at that commit**, install the tracked research integration:

```bash
git -C BALROG apply --check ../integration/balrog_research.patch
git -C BALROG apply ../integration/balrog_research.patch
```

The current workspace already has that integration. It retains BALROG's clients, history prompt builder, evaluator, environment wrappers, and result aggregation. `BALROG/balrog/agents/custom.py` only selects a named research agent. The patch also records executed fallback actions and raw calls from supplied naive/CoT agents. For E1, the original four-worker task grouping, model generation defaults, and native episode limits are preserved; E2/E3 use temperature zero and capped budgets. See [explicit profiles](exercises/common/profiles.py) and [historical panel settings](exercises/common/historical_profiles.json).

New runners select `BALROG_AGENT_VARIANT` explicitly. Legacy `BALROG_CUSTOM_MODE`, `BALROG_CUSTOM_GUIDE`, `BALROG_CUSTOM_GATE`, and `BALROG_CUSTOM_GOAL_STATE` flags remain supported by [the registry](exercises/common/agent_registry.py). History length is a profile setting, not a separate implementation.

## Run or inspect the joint follow-up

From the repository root, preview without starting a server or writing files:

```bash
bash exercises/common/run_e2_e3_step04_followup_evaluation.sh --dry-run
```

One command runs the fixed primary, broader-task, Crafter, and repeatability panels and then invokes the separate analyzer:

```bash
bash exercises/common/run_e2_e3_step04_followup_evaluation.sh
```

The original fixed study took **5.77 hours** of summed episode time, with **55,963 calls, 44.57M input tokens, and 2.51M output tokens**. Allow roughly **5–7 hours** on the original setup, plus server startup; backend and system load can change this. The runner checks its eight-hour deadline **between episodes**, so an episode already started may finish after it. Invoke the same command again to resume.

`--hours`, `--limit`, `--port`, and `--out` are available on every entry point. `--condition` and `--model` select a subset and become part of the pinned schedule; use a distinct output directory for a different selection. Completed episodes are skipped only after trajectory/call/token checks. Interrupted `.working` attempts are retained as `.incomplete_*` on retry. Changed settings or source hashes are rejected. The runner starts or verifies the model and context; it stops only servers it starts. A conflicting server is left intact with an error explaining the port mismatch.

Analysis reads saved evidence and writes elsewhere; it never launches a model or imports execution runners:

```bash
python3 analysis/summarize_e1_baselines.py --output experiment_results/refactored_v1/analysis/e1
python3 analysis/summarize_e1_cloud_models.py --output research/cloud_model/analysis
python3 analysis/summarize_e2_e3_development.py --output experiment_results/refactored_v1/analysis/development
python3 analysis/summarize_e2_e3_evaluation.py --output experiment_results/refactored_v1/analysis/evaluation
python3 analysis/measure_e2_e3_action_loops.py heldout_qwen_naive heldout_qwen_plan heldout_qwen_reflect --output experiment_results/refactored_v1/analysis/loops
python3 analysis/analyze_e2_e3_followup.py --input experiment_results/followup_v1 --output experiment_results/refactored_v1/analysis/followup
```

Use `--input` to analyze a new run namespace. Per-episode tables link to trajectories. For E1, incomplete repetitions appear in `run_status.csv` and are excluded from complete-run means; one complete repetition has no estimable run-level SD.

## Evidence and verification

Read actual observations and model replies in [initial raw-call samples](research/raw_llm_samples.jsonl), [pilot samples](research/followup_raw_llm_samples.jsonl), and [held-out samples](research/followup_heldout_raw_samples.jsonl). Each sample has a `source` pointing to its complete log. Exercise guides link both successes and failures. In CSV trajectories, `Observation` is **after** the executed action; use the raw call's current observation for the corresponding decision. `Model Action` is the model proposal, `Action` is the harness submission, `Override Reason` describes a substitution, and `Executed Action` includes BALROG's fallback. Earlier CSVs lack the newer metadata columns; do not infer an intervention from that absence.

The organizational refactor preserves behavior. [Verification records](research/refactor_verification) contain saved-call replay, reproduced numerical tables, live-run inspection, resume checks, and evidence hashes. Replay compares archived and refactored prompts, state, action metadata, tokens, and resets, including critique without an action:

```bash
conda run --no-capture-output -n balrog python exercises/common/replay_saved_calls.py
python3 exercises/common/run_experiments.py --study smoke --out experiment_results/refactored_v1/smoke_reproduction
```

The final smoke profile completed eight short Qwen3-4B episodes covering planning, critique mode, corrected guide, pickup substitution, goal-state reflection, histories 4/32, and Crafter. [Inspected observations and outcomes](research/refactor_verification/live_inspection.csv) link full prompts and trajectories; [portable raw calls](research/refactor_verification/inspected_raw_calls.jsonl) preserve the actual replies. Planning pickup20 succeeded; guide/goal-state pickup20 remained turn-loop failures; the pickup rule collected the grey box at step4; Crafter's invalid `Move South-East` became a separately logged `Noop`. History32 peaked at1,689 input tokens.

The critique-mode navigation episode succeeded before any trigger. A separate door probe exercised **two real critiques**, both counted; each chose forward into a visible wall and failed to recover. The accepted agent files have identical hashes across that probe and the final run. Earlier smoke runs and their source checkpoint remain available; the final eight were repeated after correcting historical source mappings in the runner. These are integration checks, not new statistical evidence. Exact outcomes can vary even at temperature zero.

Verification passed: nine saved-call replays; all original numerical rows (including all1,760 follow-up episodes and all three E1 models); completed smoke rerun with **zero new episodes and byte-identical files**; rejection of changed settings and a deliberately changed hash in a disposable manifest; eight numbered dry-run entry points; clean upstream patch application. The evidence audit found **24,023 original files unchanged in place**, with22 redundant source copies preserved in the dated archive and no missing/changed evidence. The formal report has source/link checks; PDF compilation remains pending a TeX installation.

## Archive, report, and submission

Before migration, original code and notes were saved in [source_before.tar.gz](archive/refactor_20261018/source_before.tar.gz). [migration_map.csv](archive/refactor_20261018/migration_map.csv) records old and canonical paths. The original frozen follow-up source bundle remains at [its recorded path](experiments/checkpoints/followup_v1_code.tar.gz). Historical submission ZIPs [E2](deliverables/Ass2-2.zip) and [E3](deliverables/Ass2-3.zip) and their raw evidence remain intact. There is one maintained copy of source and prose.

Build refreshed packages directly from canonical files:

```bash
python3 exercises/common/build_submission_archives.py
```

They are written separately under `deliverables/refactored_v1/`. These packages include source, integration patch, report, portable raw samples, the pinned cloud contribution, and selected evidence; the full original local results are retained in the workspace and indexed separately. A regenerated E1 package does not establish that an E1 archive was previously submitted. The teammate's imported cloud evaluations have their own source and evidence provenance.

Compile the report with a local TeX installation:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error Ass2Reasoning_report.tex
pdflatex -interaction=nonstopmode -halt-on-error Ass2Reasoning_report.tex
```

No TeX engine was available in the authoring environment; source checks do not replace compiling the PDF. Assignment sources: [assignment](docs/Ass2Reasoning.pdf), [book](docs/Agentic_LLM_Book-full.pdf), and the installed BALROG documentation. The investigation draws on book Sections 6.2 and 10.4; this refactor adds no agent framework or unit-test suite.
