# Autonomous BALROG experiment brief

You are working in my local ALLM Assignment 2 project on my Linux GPU desktop.
Carry out the work below, including running experiments and inspecting actual
outputs. Make routine implementation and experiment choices independently within
these bounds. Do not stop after producing a plan or ask me to approve each
iteration. If a required resource is unavailable, do the independent work first,
then report the precise blocker. Do not invent benchmark results.

## Objective and stopping limits

Make evidence-backed progress on Exercise 2 through small, targeted experiments.
If time permits, implement and pilot a clearly separated Exercise 3 extension.
Finish with reproducible code, an auditable experiment ledger, a concise analysis,
and a selected next experiment. An honest negative result is a valid outcome.

Limits for this session, including retries, debugging evaluations and baselines:

- At most FOUR HOURS of total elapsed work, including a final 15-minute reporting buffer.
- At most TWO HOURS of cumulative benchmark-execution wall time, including failed
  attempts; measure elapsed execution time rather than estimating GPU utilization.
- At most 300 episode attempts, including failed/incomplete attempts and repeats.
- At most six new E2 candidate configurations and two E3 pilot configurations.
- One local GPU-serving model and one evaluation job at a time.

Stop at the first limit reached, or sooner if the evidence is sufficient or no
productive next experiment remains. Record deadlines and consumption in a state
file so continuation or compaction does not reset the budget. Set process
timeouts from the remaining budget. Save completed episodes, classify unfinished
attempts as incomplete, and stop only processes you started. Do not leave a long
evaluation running after declaring the session finished.

Do not run another full overnight matrix or use held-out confirmation seeds in
this session. Prepare the larger confirmation command for later execution.

## Model and environment

All benchmark action generation, candidate evaluation by an LLM, reflection and
memory summarization must use ONLY the existing local model:

`Qwen/Qwen3-4B-Instruct-2507-FP8`

This is our Qwen3-4B-Instruct evaluation model. Do not substitute Thinking, Qwen
14B, Phi, or an API model. My teammate handles API-model evaluations. Codex itself
can use its normal coding tools; the restriction concerns experimental LLMs.

The existing setup uses conda environments `vllm` and `balrog`, a local vLLM
OpenAI-compatible endpoint normally at `http://127.0.0.1:8088/v1`, and an RTX 5080
with 16 GB VRAM. Inspect the actual installation before changing it. Reuse an
existing compatible model server where possible. Do not kill unrelated jobs,
upgrade the working stack, or download another model just to begin experimenting.

Do not spend this session on a full standard-CoT sweep. The previous one was
stopped because of excessive runtime; CoT is not a prerequisite for this work.

## Read first and establish the current state

Locate and read the actual files in '/docs':

- `Ass2Reasoning.pdf` and relevant sections of `Agentic_LLM_Book-full.pdf`.
- The current BALROG evaluator, environment wrappers, prompt builder, agents,
  model client and configuration, together with repository instructions.
- Existing E1/E2 runners and the `exercise2/` custom-agent package and README.
- Existing `exercise1_results/`, `exercise2_results/`, or `exercise1.zip` and
  `exercise2.zip` if supplied. Read archive contents directly where practical.

Reuse completed measurements when model, prompt, task, seed, configuration and
protocol match. Preserve original agents, results and logs. Work in a separate
experiment directory or branch; preserve any uncommitted user work. Keep code
simple, commented and suitable for a student assignment. Avoid rebuilding the
framework or performing unrelated refactors.

The previously inspected results provide context, not substitutes for files:

| Qwen 4B agent | BabyAI | Crafter | BabaIsAI | Overall |
| --- | ---: | ---: | ---: | ---: |
| Naive | 32.0% | 18.9% | 2.2% | 17.7% |
| Structured planning | 22.7% | 7.4% | 9.2% | 13.1% |
| Planning with Python checks | 23.3% | 5.8% | 8.3% | 12.5% |

These are means across three repetitions. The panels used 50 BabyAI, 10 Crafter
and 120 BabaIsAI episodes per repetition. There is no completed standard-CoT
baseline in these archives. Planning generated about 27 times the Naive output
tokens and took about 66 minutes per full panel versus 10 minutes for Naive.

Observed failures to investigate:

- Planning improves BabyAI goto but nearly eliminates successful pickup.
- A key directly ahead often produces `go forward` instead of `pick up`.
- The grounded agent rejects blocked forward movement, then often turns away.
- Crafter plans to collect wood while moving away from a tree it already faces.
- BabaIsAI gains are on goto_win tasks; explicit rule-construction tasks remain
  unsolved, and most steps still have unchanged textual observations.
- Token truncation was rare in E2; do not assume a larger token budget fixes it.

## Stage 1: audit and build a small development panel

First audit the public game mechanics, the action descriptions given to the
model, coordinate conventions, action parsing, observation/action alignment and
history handling. In particular, verify pickup semantics against the installed
environment. Our source inspection raised concerns about the wording of the
upstream BabyAI pickup instructions. Check this rather than assuming either the
prompt or our interpretation is correct.

Create and save a deterministic, stratified development manifest, approximately:

- BabyAI: all five default tasks with seeds 11 and 12: 10 episodes.
- Crafter: seeds 11 and 12: 2 episodes.
- BabaIsAI: seed 11 for six default tasks covering plain/distractor navigation,
  plain/distractor rule construction, and two-room navigation/rule construction:
  `env/goto_win`, `env/goto_win-distr_obj_rule`, `env/make_win`,
  `env/make_win-distr_obj_rule`, `env/two_room-goto_win`, `env/two_room-make_win`.

Use the same manifest for every complete screening comparison. Measure time on
a tiny setup check, then choose a feasible candidate count under the budget.

Run one episode per task/seed per screening pass. BALROG reuses a fixed seed
for every episode in one eval call, so do not request multiple episodes with
the same fixed seed and count them as different layouts. Set Crafter's
constructor seed as well as the shared environment seed. Record initial text
observations/fingerprints to verify layouts actually match, not just seed labels.
Reserve seeds 100–119 for a later confirmation protocol; do not evaluate or use
their trajectories while selecting or tuning candidates.

## Stage 2: screen focused Exercise 2 interventions

Establish Naive and the original E2 planning/grounded references on the development
manifest, using matching cached results if available. Reproduce the legacy token
budgets for references; keep budgets fixed among comparable candidates and log
any intentional budget change as a separate factor. Track actual calls/tokens
so extra retries or reflection are never treated as free.

Choose at most six E2 candidates from the audit and measured failures. Reasonable
hypotheses include:

- Correct, concise action-mechanics instructions, tested as a distinct change.
- A shorter immediate-subgoal-and-action prompt with less verbose planning.
- A small number of independently constructed, general action demonstrations.
- Better constraint feedback that explains an action's requirements.
- Explicitly presenting the previous executed action and observed change,
  without a critic, learned lessons or additional persistent memory.

These are suggestions, not a mandatory grid. Aim to screen at least three distinct
E2 candidates when feasible within the limits; explain if cost or blockers prevent
that. Change one main factor at a time.
Prefer the smallest change that can test a concrete hypothesis. Keep explicit
self-reflection and learned-lesson memory separate for Exercise 3. Game-specific
rules are allowed if they use public mechanics, are documented and are not
presented as generic reasoning improvements.

Start candidates with a few diagnostic episodes. A candidate that fails basic
execution checks need not receive a full panel. Label such partial diagnostics
clearly; never compare their partial scores with complete-panel scores as if
sample coverage were equal. For completed panels, inspect a consistent sample
of successes, failures and regressions, including pickup, Crafter interaction and
BabaIsAI rule construction. Do not choose only persuasive examples.

After each candidate, record its hypothesis, exact change, results, cost, failure
examples and the next decision. Drop unproductive candidates. Repeat only the
best one or two complete-panel candidates, up to two additional passes if budget
permits. Do not automatically run three passes for every idea.

## Stage 3: optional bounded Exercise 3 pilot

If the audit and E2 experiments leave enough budget, freeze the chosen E2 actor
and implement one small, separately labeled self-reflection/memory extension.
A negative E2 result does not justify hiding the outcome; choose the most
defensible actor and explain why.

A suitable pilot triggers a short reflection after repeated ineffective actions
or an observed action-effect mismatch, then stores a bounded corrective lesson
that the actor receives on later decisions. Use a cooldown and strict reflection
budget. Triggering on repeated observation alone is a heuristic: allow for
legitimate waiting or repeated resource-gathering actions.

Use the SAME local Qwen model. Define the memory reset boundary; default to
within-episode memory and clear it between episodes. Do not let lessons leak
between variants, repetitions or later confirmation episodes. Log triggers,
reflection prompts/responses, memory updates, timing and all additional tokens.
Compare the frozen actor with and without this extension on the same development
panel. At most two E3 configurations; otherwise leave tested scaffolding and a
specific next experiment rather than launching a large run.

## Evaluation integrity and reporting

During action selection, the agent may use only observations/history exposed to
the ordinary benchmark agent and public rules. No hidden game state, target_plan,
reference trajectories, future observations, seed-specific solutions or oracle
actions. Do not change rewards, success definitions, task difficulty or step
limits to manufacture an improvement. A debug step cap is allowed only in a
clearly separate setup check, never pooled with full-episode screening results.

Keep sampling settings, history, tasks/seeds and scoring fixed unless they are
the intervention. Log every LLM call, including validation retries, repairs and
reflection. Preserve proposed and executed actions, current input observations,
next observations, raw responses, finish reasons, tokens and timings. Mark failed
or unfinished episodes explicitly and verify expected episode counts before
reporting complete results. Record the BALROG commit, dependency versions,
configuration, prompts, code hashes, initial observations and model defaults.

Report per-benchmark progression and BabyAI task-level success. Keep overall
progression as an equal-weight mean across benchmarks. Also report invalid
commands, ineffective/repeated actions, plan/action mismatch examples, actual
calls/tokens, and evaluation time. Give paired differences on matched cases.
A lower fallback rate or smoother trajectory is not proof of better task solving.

Do not claim significance from the tiny development panel, treat repeated seeds
as new independent layouts, or select a variant solely on one lucky aggregate.
Prefer substantial gains with an understandable mechanism and reasonable cost.
If results trade off across benchmarks, retain and explain that trade-off.

## Required checkpoints and completion evidence

Create a resumable experiment folder containing:

- `STATE.json`: budgets, completed work, active process ownership and next step.
- A fixed development manifest and machine-readable experiment ledger.
- Versioned candidate code/prompts and meaningful tests for changed mechanisms.
- Raw logs, per-episode measurements and concise comparison CSVs.
- `FINDINGS.md`: what was tried, what worked or regressed, representative evidence,
  limitations, and material suitable for the assignment's E2/E3 questions.
- A selected frozen agent, or an explicit no-improvement conclusion supported by
  the completed experiments.
- A runnable command and protocol for a later held-out confirmation experiment;
  specify its primary endpoint, matched baselines, seeds, repetitions and analysis.
  Use paired uncertainty estimates that account for repeated layouts; choose
  sample size from pilot variability and an effect worth detecting. Do not promise
  significance or repeatedly inspect confirmation data until it becomes significant.

Give concise progress updates after substantive checkpoints and maintain the
ledger even if the session is interrupted. End with actual completed experiment
counts, budget used, measured changes, important blockers and the recommended
next command. Stop when this evidence exists or the limits are reached. Do not
continue searching indefinitely for a positive result, spend money on external
evaluation APIs, push/publish changes, or claim to have completed the entire
assignment before the remaining teammate work is available.
