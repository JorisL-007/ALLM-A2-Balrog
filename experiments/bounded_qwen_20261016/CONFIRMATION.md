# Held-out confirmation protocol (future run only)

This session does **not** execute seeds 100–119 or inspect their observations.
The actor and protocol are frozen in `FROZEN.json`. Actor selection was locked
before the separately labelled development-seed variance calibration.

## Design and endpoint

Use all 20 reserved seeds, 100 through 119 inclusive. For each seed use the
five default BabyAI tasks, Crafter default, and the same six BabaIsAI tasks as
the development manifest: 12 tasks per seed. Two independent decision passes
repeat those layouts. Use the selected E2 actor and all three references:
Naive, original planning, original grounded. This is 240 episodes per arm per
pass and 1,920 completed episodes overall. Repeated seeds are not new layouts.

The primary endpoint is the selected actor minus Naive in equal-weight overall
progression, averaged over the fixed tasks within each benchmark, then over the
three benchmark means. Average the two decision passes within each seed. The
primary practically worthwhile effect is **+10 percentage points** overall.
Original planning is the mechanistic, equal-output-budget comparator; grounded
is another descriptive reference. Benchmark/task-specific effects and token,
call, invalid-command and runtime costs are secondary descriptive outcomes.

Maintain exactly the frozen prompt/code, 16 text history, no images, one previous
reasoning trace, model sampling defaults (.7 temperature/.8 top-p/top-k 20),
128-token Naive and 512-token E2 caps, actual environment default step limits,
and constructor/reset seeding. Run serially against the same cached local
Qwen/Qwen3-4B-Instruct-2507-FP8 revision. Never expose hidden state or solutions
to the actor. Memory is reset between episodes; E3 is excluded from this E2
confirmation experiment.

## Uncertainty and stopping

Primary uncertainty is a paired percentile bootstrap with 10,000 resamples of
the **20 shared seed clusters**. Keep every task, benchmark, arm and repeated
pass together when resampling a seed. This protects against treating repeated
decisions on a layout as independent samples. Report the paired point estimate,
95% interval and per-benchmark trade-offs. Other comparator intervals are
descriptive and do not constitute additional confirmatory claims.

Report evidence for a practically useful improvement only if the primary point
estimate is at least +10 pp and its predeclared interval excludes zero. Otherwise
report the measured result and uncertainty; an interval spanning zero is not
proof of equivalence. A tiny development panel cannot certify power. The final
sample-size calculation and its limitations are recorded in `FROZEN.json`.

Post-lock calibration completed the six BabaIsAI tasks at development seed 12
for Naive and the frozen actor. Combined with the first screening passes, this
gives two complete paired seed clusters, without using confirmation data. Their
overall paired differences were +23.64 pp and -9.09 pp (SD 23.14 pp). A normal
planning approximation, `(1.96 + 0.84)^2 * SD^2 / effect^2`, suggests **42 seed
clusters** for 80% power against a 10 pp effect; the estimate itself is unreliable
with only two pilot clusters. Twenty reserved clusters correspond to an estimated
14.49 pp detectable effect under that approximation.

The runnable protocol therefore uses all **20 reserved seeds as a bounded
precision study**. It is **not power-qualified for a 10 pp effect**, and smaller
worthwhile gains can remain inconclusive. Do not replace missing independent
seeds with more repetitions or expand the bank after seeing confirmation
results. A power-qualified larger study would need a newly preregistered untouched
seed bank and revised resource plan before any confirmation execution.

Do not inspect interim performance or change prompts, seeds, endpoints or
sample size in response to confirmation outcomes. Stop at the predeclared
sample count. The launcher uses a separate future-session ceiling of 24 elapsed
hours, 23 cumulative benchmark hours, 2,160 attempts (including failures), one
server and one evaluation job. These are future-run limits, not permission to
extend the present session. If the panel cannot finish, retain all logs and
report an incomplete protocol, without an efficacy conclusion. Inspect failures
only to diagnose infrastructure; do not turn those trajectories into tuning data.

The analysis refuses missing/duplicate cases or mismatched initial fingerprints.
No efficacy summary is emitted until all four arms and both passes are present.

## Runnable command

From the project root, after reading the final `FINDINGS.md` and `FROZEN.json`:

```bash
bash experiments/bounded_qwen_20261016/protocol/confirm.sh \
  "$PWD/confirmation_qwen_frozen_$(date +%Y%m%d_%H%M%S)"
```

This verifies frozen file hashes, starts the exact cached model if the endpoint
is absent, verifies its model ID, saves raw logs, checks full coverage and runs
the paired analysis once. If reusing a server, verify its launch flags and model
revision match `provenance.json`; an ID alone cannot prove sampling compatibility.
It stops only a server it starts. Do not run this command during development.

The analysis can later be reproduced without model calls:

```bash
python3 experiments/bounded_qwen_20261016/protocol/analysis.py \
  /absolute/path/to/the/completed/confirmation_output
```
