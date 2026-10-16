# Bounded autonomous experiment

This directory is separate from the original BALROG/E1/E2 code and results.
Read `FINDINGS.md` for the measured outcome and `AUDIT.md` for mechanics and
harness findings. `STATE.json` records the session's original deadlines and
charged budget; `ledger.jsonl` records every attempted episode, including setup
and incomplete work. Development cases are fixed in `development_manifest.json`.

Use the existing `balrog` Python environment, from the project root:

```bash
/home/joris/miniconda3/envs/balrog/bin/python -m unittest discover \
  -s experiments/bounded_qwen_20261016/tests -v
python3 experiments/bounded_qwen_20261016/audit/analyze.py
python3 experiments/bounded_qwen_20261016/audit/evidence.py
```

The three references are original Naive, original E2 planning and grounded.
The focused agents subclass the original E2 custom actor, itself a BALROG
BaseAgent. No upstream files are changed. `episode.py` uses actual BALROG
wrappers, default limits and score definitions, and preserves the legacy
proposed-action history policy while logging both proposal and execution.

Every episode saves config, initial text/mission and fingerprint, all request
prompts and responses with finish reasons/tokens/timing, and aligned pre-action
and next observations. SDK retries are disabled; grounded validation retries
are explicit logged calls. A single loopback endpoint and exact model ID are
hard coded. The runner rejects a changed code hash, overlapping evaluator,
exhausted budgets and held-out seeds in development mode. Incomplete output
is preserved in place; retries receive unique directories.

To inspect a command without executing model calls:

```bash
python3 experiments/bounded_qwen_20261016/run.py --variants mechanics \
  --pass-id example --dry-run
```

Do not restart the old session with a new budget. A resumed controller checks
`STATE.json`. If an interruption leaves `active_evaluation` set, first inspect
that owned process and its output, charge all elapsed execution time, and mark
its attempt incomplete before clearing the entry. Never kill unrelated jobs.

The later held-out command and predeclared analysis are in `CONFIRMATION.md`.
`protocol/confirm.sh` creates a fresh output directory, starts the cached exact
model only if needed, and stops only a server it starts. It is not executed
during development. Host execution is needed when a sandbox hides the GPU or
loopback endpoint.


The metadata-only v2 correction is documented in `audit/CONFIGURATION_ERRATUM.md`.
Original saved configs remain available with `effective_agent.json` sidecars.
Internal legacy agent logs use `planning`/`grounded` for their validation mode;
`result.json` variant, directory and sidecar identify the intervention.

`ACTOR_LOCK.json` fixes E2 selection before all later calibration/E3 work.
`protocol/calibration/` contains 12 separately labelled Baba seed12 episodes for
sample-size planning; they are excluded from E2 screening scores. `FROZEN.json`
fixes the future confirmation protocol and explicitly reports its precision limit.
E3 logs/results live only under `e3/`; reproduce its paired analysis with:

```bash
python3 experiments/bounded_qwen_20261016/e3/analyze.py
python3 experiments/bounded_qwen_20261016/audit/integrity.py
```
