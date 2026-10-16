# Saved loader defaults in experiment code v1

The v1 `config.yaml` retained several **unused** BALROG loader defaults:
`agent.type=naive`, all environment/task lists, default episode counts, and the
client retry count. The explicit `make_agent(variant, ...)` factory selected the
correct actor independently of `agent.type`; the direct single-episode loop
selected its environment/task from the case manifest. `LoggedClient` explicitly
disabled transport retries. These effective settings and all actual requests,
responses, trajectories, seeds and costs are present in the raw evidence.

All E2 screening/repetition results were generated with v1. Their original
configs are preserved. Added `effective_agent.json` sidecars reconstruct the
effective actor/settings from the preserved factory code and result identity.
They are annotations, not replacements for the original files.

After E2 repetitions, v2 corrected these saved labels and records the actual
actor class. **No prompt, factory choice, action, score, history, step limit,
sampling parameter or retry behavior changed.** These fields are not read by
the explicit factory/action loop. The code revision is recorded in `ledger.jsonl`
and `STATE.json`; exact v1/v2 source and hashes are in `audit/versions/`.

A differential real-environment test against BALROG's evaluator passed again
after the metadata correction. Later calibration, E3 and confirmation use v2.
This distinction must be retained when reproducing the saved configuration.
