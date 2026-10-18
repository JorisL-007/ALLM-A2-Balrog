# BALROG: planning, grounded actions, and reflection

**Research question:** Can a small persistent plan and feedback-triggered critique improve task completion and reduce unproductive loops at an acceptable inference cost?

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
- Crafter transfer and limited other-model checks constrain generalization. The matched local CoT comparison was a small development reference; a larger local CoT sweep has one complete repetition and an incomplete second. Few-shot and robust variants were not evaluated.
- The separately imported **cloud-model contribution** verifies GPT-4.1 mini's 41.60% BabyAI score and the reported cloud Crafter scores; GPT's 27.50% BabaIsAI result is supported by summaries only. CoT improved GPT's observed Crafter mean but did not improve Gemini's. These native-budget panels do not test our custom agents and are not pooled with the paired follow-up. [Evidence, settings, and limitations](exercises/exercise1_baselines/README.md#cloud-model-contribution--teammate).

The formal argument, assignment answers, and remaining identity/submission placeholders are in [the LaTeX report](report/Ass2Reasoning_report.tex). Historical metrics remain in [the initial matrix](experiment_results/performance_matrix.csv) and [the follow-up matrix](experiment_results/followup_v1/performance_matrix.csv).

## Critical assessment and proposed next work

**Strengths.** The study follows inspected failures through a specific change to a matched evaluation. Frozen controls, held-out seeds, explicit action/fallback logging, bounded reflection, inference-cost accounting, paired confidence intervals, and negative findings make its central claims auditable. The 1,760-episode follow-up is substantial for this assignment. The imported cloud baselines broaden model coverage while preserving the original local results.

**Limits.** Most custom-agent evidence comes from one Qwen4 model and short BabyAI tasks. The limited Qwen14/Phi checks already suggest that harness gains do not reliably transfer. Crafter transfer is small, and the initial Crafter critic never triggered, so that panel does not really test the reflection mechanism. There is no adequately matched, held-out CoT comparator for the main custom-agent study; few-shot and robust agents remain untested. The cloud contribution evaluates supplied agents only, with different settings and incomplete logging. Fixed seeds do not eliminate inference variation. High episode SD is expected for binary success near 50%; uncertainty in the **paired improvement** is the relevant target.

The next CoT-control and model-transfer study is now implemented as Step 05 below. Its evaluation results remain pending. The historical CoT panels have been recovered without extending the old sweep.

| Priority | Proposed work | Why it adds evidence |
|---|---|---|
| 1 — quickest, no inference | Historical local CoT panels are now recovered; obtaining the teammate's missing Qwen2B episodes and Baba trajectories remains useful. | The recovery retains all 245 CoT episodes and distinguishes complete environment panels from the partial BabaIsAI repetition. |
| 2 — strongest missing control | Add supplied CoT to a matched Qwen4 BabyAI comparison with naive, frozen planning, and frozen action critique. | Tests whether persistent state/selective critique adds anything beyond ordinary step-by-step prompting. Fewer repeated runs are acceptable; prioritize distinct paired seeds over repeatedly solving the same puzzles. |
| 3 — best expansion | Run those same four agents with one cloud model, preferably the teammate's already-used GPT-4.1 mini if the pinned version is available. | Tests transfer of the **agent improvement within a model**, beyond a table of different models' naive scores. It also shows whether a stronger base model still benefits from our loop intervention. |
| 4 — cheap focused replication | Freeze history 4 versus history 16 and test a fresh `open` seed panel. | Directly tests the most promising cost-saving finding, which is currently exploratory and did not pass the adjusted primary-panel comparison. |
| 5 — additional model family | Step 05 includes OLMo-2-0425-1B-Instruct, preferring the official Q8_0 quantization. | This adds a small model from a different family to the same core-agent comparison. |

The chosen panel uses the existing **60 evaluation seeds per BabyAI task**, with CoT on a fixed half-sized subset. It adds the missing Qwen CoT control while reusing compatible Qwen evidence, then applies all four agents to GPT-4.1 mini and OLMo. Reusing an existing evaluation seed panel tests transfer without adding independent new Qwen puzzles. Compare effects within each model and retain the earlier repeatability findings as a limitation.

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

Outputs are separate under `experiment_results/refactored_v1/model_transfer_v1/`. The script recovers historical CoT panels first, runs Qwen → GPT → OLMo, and analyzes after each stage. The default invocation limit is48 hours, checked between episodes. Repeating the command resumes verified unfinished work; `--hours`, `--limit` (counts new pilots too), `--out`, `--port` (default8097), `--model`, and `--condition` are available. Scientific settings/source changes are rejected; filtered schedules require their own output directory. Only owned servers are stopped. Failed/interrupted attempts are retained. Persistent stage failures allow later models to proceed; exit2 means the selected study remains incomplete.

Cloud generation uses `gpt-4.1-mini-2025-04-14` and the existing OpenAI client. **$10 is a target; execution continues until the cumulative estimate exceeds $20**, then proceeds to OLMo. The guard includes pilots, incomplete attempts, and retries across resumes. Known usage is estimated at $0.40/M input and $1.60/M output without cache discounts; unknown/interrupted requests receive conservative provisional charges. These are estimates, not invoices. `--cloud-target-usd` and `--cloud-stop-usd` change operational limits without changing the scientific panel. All request attempts have full payloads/replies, finish reasons, timing, and available usage in `api_attempts.jsonl`; SDK retries are disabled so BALROG's three attempts remain visible. Requests time out after60 seconds and episodes after30 minutes.

Reused controls retain their original actor/critic logs and measured times. Their older logs lack transport-level audits, so unavailable retry/truncation counts remain blank. Newly incurred cloud spending is calculated from the new request ledger.

OLMo prefers the cached official Q8_0 GGUF, official tokenizer, and native4096 context. Startup incompatibility permits official BF16 before any pilot/evaluation, with missing files downloaded through `hf` into the study's server directory. The selection and artifact hashes are pinned across resumes; quality or evaluation outcomes never trigger a precision switch. Serving requires a free dedicated port at first startup. The script reports pilot-based time/cost estimates; the old5.77-hour Qwen follow-up is not a runtime estimate for this study.

Local setup records also pin vLLM, Transformers, PyTorch, and Python versions. A changed serving environment or model artifact is rejected before restarting inference. The final version-pin check made no model calls; its earlier prepared references and source checkpoint were retained separately before initializing the final study namespace.

Analysis is independent and can recover the old CoT evidence without running models:

```bash
python3 analysis/analyze_e2_e3_model_transfer.py \
  --input experiment_results/refactored_v1/model_transfer_v1 \
  --output experiment_results/refactored_v1/model_transfer_v1/analysis
```

Read `historical_panels.csv`, `historical_matched_panels.csv`, `historical_matrix.csv`, `overall_matrix.csv`, `performance_matrix.csv`, `coverage.csv`, `paired_comparisons.csv`, `status.json`, and `inspection_index.csv`. CoT comparisons restrict controls to its exact seed subset. Four overall BabyAI contrasts per model use the original seed-block bootstrap/permutation methods, seed0, and Holm adjustment across twelve tests; per-task and Crafter scores are descriptive. Reused episode times remain original and are distinguished from newly incurred time. Missing episodes remain missing. Automated sample selection leaves manual review explicitly pending. [Verification evidence](research/model_transfer_verification) records which checks and live pilots actually completed.

Implementation verification completed six real Qwen and six real OLMo pilots: **91 actions and92 LLM calls**, including an OLMo critique without an `ACTION` followed by a separately counted actor call. All twelve full trajectories and raw calls were personally inspected and preserved in [the inspection table](research/model_transfer_verification/live_inspection.csv). Qwen navigation/pickup succeeded; its CoT calls truncated4/15 times. OLMo had no pilot successes and showed rotation, wall-collision, and failed-crafting loops; these are eight-action integration checks, not transfer-effect estimates. Q8 startup failed in vLLM0.31.0, so OLMo's tested precision is **BF16**. The six GPT pilots remain blocked by the absent environment API key. Transport replay and disposable control-flow checks verified token accounting, retries, completed-run skips, interrupted-attempt retention, source/settings rejection, and the10/20-dollar spending rules. The prepared main namespace contains only the550 referenced controls; no new bulk evaluation has been started.

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
