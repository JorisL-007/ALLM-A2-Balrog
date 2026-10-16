# Exercise 2 — structured planning and action checks

This package runs Qwen/Qwen3-4B-Instruct-2507-FP8 on BabyAI, Crafter and BabaIsAI.
It uses the existing `vllm` and `balrog` conda environments and BALROG installation.
The launcher selects our `CustomAgent` for that Python process; it does not overwrite
BALROG's `custom.py`, prompts, configuration or evaluator.

## 1. Put the package beside BALROG

Unzip the download in your existing project directory. The layout should be:

```
ALLM-A2-Balrog/
    BALROG/
        eval.py
    exercise2/
        run_ex2_local.sh
        eval_ex2.py
        ex2_agent.py
        summarize_ex2.py
        README.md
        tests/
```

Run the commands below from `ALLM-A2-Balrog/exercise2/`.
If your checkout is elsewhere, prefix runner commands with
`BALROG_DIR=/absolute/path/to/BALROG`.
Stop a manually launched vLLM server on port 8088 before running this script.
The script starts and stops its own server, once for the entire sweep.

## 2. Smoke test first

```bash
SMOKE=1 bash run_ex2_local.sh
```

This tests BOTH variants, ALL three environments and ALL configured tasks, with
one seed per task, one repetition and a ten-step limit. With the default task
lists, that is 92 short episodes. Inspect the console output for errors and check
that `performance_matrix.csv` is created. These are setup checks, not reportable
benchmark scores. Smoke and full runs receive separate timestamped directories.

If BabaIsAI is missing from your current installation, follow BALROG's setup in
`BALROG/docs/installation.md`. Its standard setup includes `pip install -e .` and
`balrog-post-install`; run these in the existing `balrog` conda environment from
inside the BALROG checkout if setup was not completed previously.

## 3. Overnight full run

If tmux is not already running:

```bash
tmux new -s balrog-ex2
```

From the `exercise2/` directory inside tmux:

```bash
SWEEP_ID=ex2_overnight bash run_ex2_local.sh 2>&1 | tee ex2_overnight.log
```

Detach with Ctrl+B, then D. Reconnect with:

```bash
tmux attach -t balrog-ex2
```

Default settings:

| Setting | Value |
| --- | --- |
| Local model | Qwen3-4B-Instruct-2507-FP8 |
| Variants | `planning`, `grounded` |
| Repetitions | 3 per variant |
| BabyAI panel | Seeds 1–10; one episode per task/seed = 50 episodes/repetition |
| Crafter panel | Seeds 1–10 = 10 episodes/repetition |
| BabaIsAI panel | Seeds 1–3; one episode per task/seed = 120 episodes/repetition |
| Total | 1,080 episodes across both variants and three repetitions |
| History | 16 text observations; 0 images; 1 previous reasoning trace |
| Temperature/sampling | Model generation defaults loaded by vLLM; no temperature override |
| Output budget | 512 tokens per call, matching Exercise 1 CoT |
| Workers | 4 |
| Step limit | Environment/BALROG default |
| Extra validation call | At most one per step, only for `grounded` |

Panels use the installed config's default episode counts and task lists. The
counts above assume the standard 5 BabyAI and 40 BabaIsAI tasks. Each panel is
repeated three times; it is not three different collections of layouts. This
estimates variation in decisions on a fixed set of environments. Fixed layouts
and the server seed improve repeatability, but stochastic GPU inference need not
be identical. Crafter's constructor seed and the shared environment seed are
both set explicitly, as in the revised Exercise 1 runner.

To use your teammate's larger panels instead, use the same overrides for E1 and E2:

```bash
BABYAI_SEEDS=25 BABAISAI_SEEDS=5 SWEEP_ID=ex2_large bash run_ex2_local.sh
```

That increases the total to 2,010 episodes. Runtime depends on episode lengths
and retries; the runner has no old per-model/whole-sweep cutoff. Full raw prompts
and completions are saved and can take substantial disk space.

## 4. What the variants do

Both variants use exactly the same prompt in `ex2_agent.py`:

1. Specify the immediate subgoal.
2. Give at most three short steps.
3. State the next action's preconditions, marking unknowns explicitly.
4. Output one primitive `ACTION`.

The plan is regenerated at each observation. Only BALROG's normal history and
one previous reasoning trace are retained. There is no extra memory store,
learned lesson retrieval, critic or failure-triggered self-reflection module.

`planning`: parses the ACTION field and passes the proposal to BALROG. Unsupported
commands still go through BALROG's standard invalid-action fallback; they are not
silently repaired by this variant.

`grounded`: runs the same proposal through conservative Python checks. If a
check fails, it provides the concrete constraint and requests one corrected
response. A second failure executes a logged fallback: `turn left` in BabyAI,
`Noop` in Crafter, `idle` in BabaIsAI. Fallbacks are always vocabulary-valid;
they are not guaranteed to be strategically useful.

The checks are deliberately limited:

| Benchmark | Python checks |
| --- | --- |
| All three | Exactly one ACTION field and membership in the action vocabulary |
| BabyAI | Known wall/pickup-object ahead blocks forward movement; pickup requires a visible pickup object directly ahead; drop requires a visibly unoccupied front tile |
| Crafter | Placement/crafting requires at least some of its necessary material, when the current inventory is explicitly available |
| BabaIsAI | Vocabulary only: no fixed collision rules, since STOP/PUSH/YOU can change |

Unknown inventory, hidden state and unobserved conditions are not treated as
known facts. BabyAI's carrying state is not inferred. Crafter's exact recipe
quantities, tool requirements and workstation distance are not fully validated.
These checks do not prove overall plan correctness. All checks read only the
same exposed text available to the LLM, not internal game state or target plans.

BabaIsAI therefore tests transfer of the planning prompt and the action-interface
checks, rather than the BabyAI/Crafter-specific semantic rules. The grounded
variant combines checks with a retry policy; interpret it as that combined
intervention. It can spend up to twice the output budget at a problematic step.
Both calls' tokens are included in BALROG totals and the summary metrics.

## 5. Results and inspection

Results are written beneath `exercise2_results/<SWEEP_ID>/`:

- `performance_matrix.csv`: progression mean and sample SD across the three full panels.
- `per_run_metrics.csv`: episodes, steps, progression, input/output tokens, LLM calls,
  initial check rejections, retry rate, agent fallback rate, BALROG fallback rate,
  and total measured decision time.
- `qwen4-instruct/<variant>/run_<n>/summary.json`: standard BALROG aggregated summary.
- `<run>/<environment>/seed_<n>/...`: original episode JSON/CSV files and console log.
- `<run>/<environment>/seed_<n>/raw_responses/*.jsonl`: full text prompts, completions,
  extracted actions, check decisions, retries, chosen actions and per-step time.
- `seed_panels.tsv`, `settings.txt`, `system_info.txt`, `status.csv`, and the vLLM log.

BALROG summary `agent.type` is `custom` for both variants; directory names and raw
log metadata distinguish them. Each JSONL file is one episode. Its parent path
identifies environment and seed, and its first prompt includes the task/mission.
The JSONL `observation` is the input to the decision; BALROG's CSV observation is
the feedback AFTER the executed action.

Initial check rejection rates are computed for BOTH variants, even though
`planning` only logs the result and does not enforce the check. BALROG and agent
fallback rates are distinct: a vocabulary-valid agent fallback is not rejected
by BALROG. Overall progression gives each environment equal weight; overall
fallback/retry percentages are pooled over steps. SD is across repeated panels,
not BALROG's within-panel standard error. Three panels alone do not establish
statistical significance, and fixed-seed repetition does not measure all layout
variability.

To compare with the REVISED Exercise 1 Naive/CoT results:

```bash
python3 summarize_ex2.py exercise2_results/ex2_overnight \
  --baseline-dir ../exercise1_results/YOUR_E1_SWEEP_ID
```

This adds Naive/CoT rows to the two CSV files and rejects mismatched task/seed
panels. Use the matching 16-history E1 run, not the older 8-history results.
No need to repeat E1 from this package. If your E1 script lives elsewhere, supply
its absolute results path. Baseline-only raw-call metrics are blank when no
matching raw-response logs exist.

## 6. Resume and tests

To resume an interrupted full run, use the SAME settings and ID:

```bash
SWEEP_ID=ex2_overnight bash run_ex2_local.sh
```

Completed seed evaluations are skipped. Incomplete attempts are preserved under
`incomplete/` and that seed evaluation is restarted; they are excluded from the
aggregated full panels. The script rejects changed code/config/settings when
resuming. Use a new sweep ID after changing them. GPU model failures and incomplete
panels stop the sweep and are not reported as successful evaluations.

Run the included agent tests without a GPU:

```bash
TEST_BALROG_DIR=../BALROG python3 -m unittest discover -s tests -v
```

The tests load the real BALROG BaseAgent and history builder directly. They
exercise relative coordinates, missing inventory information, dynamic BabaIsAI
rules, identical prompts, repair limits, token accounting and episode resets.
Syntax, tests and simulated launcher/runner/summary integration were checked in
the authoring environment. Real vLLM/game evaluations still need the smoke test
on your computer.

## 7. Teammate's API model

The launcher is provider-independent. After setting the API provider's key, your
teammate can run one seed using the same custom agent, for example from this folder:

```bash
conda run --no-capture-output -n balrog python eval_ex2.py \
  --balrog-dir ../BALROG --variant planning --environment babyai \
  --log-dir "$PWD/api_planning/babyai/seed_1/raw_responses" -- \
  agent.max_text_history=16 agent.max_image_history=0 \
  agent.remember_cot=True agent.max_cot_history=1 \
  envs.env_kwargs.seed=1 eval.num_episodes.babyai=1 \
  eval.num_workers=4 eval.max_steps_per_episode=null \
  eval.save_trajectories=True eval.resume_from="$PWD/api_planning/babyai/seed_1" \
  client.client_name=openai client.model_id=gpt-4.1-mini-2025-04-14 \
  client.generate_kwargs.temperature=null client.generate_kwargs.max_tokens=512 \
  client.generate_kwargs.thinking_budget=null
```

Create the output directory first (`mkdir -p api_planning/babyai/seed_1`), since
BALROG expects `resume_from` to exist. Repeat with `--variant grounded`, each
benchmark, matching seeds, and three separately stored repetitions. For Crafter,
include `envs.crafter_kwargs.seed=1` alongside `envs.env_kwargs.seed=1`. Use his
chosen model/provider rather than necessarily the example model. Provider/model
sampling defaults can differ, so report them as part of the comparison.

## Research connection

This is a small adaptation of task decomposition and action grounding, not a
reproduction of a published trained planner. It follows the course book's
sections 6.1.3, 6.2.3, 7.3.2 and 10.2.5 and the assignment's suggestions to change
prompts/plans and try benchmark-specific harness improvements.

- Wang et al. (ACL 2023), Plan-and-Solve: https://aclanthology.org/2023.acl-long.147/
- Erdogan et al. (ICML 2025), Plan-and-Act: https://arxiv.org/abs/2503.09572
- Kambhampati et al. (2024), LLM-Modulo: https://arxiv.org/abs/2402.01817

Exercise 3 can later add failure-triggered reflection and memory of useful
corrections on top of one selected E2 variant.
