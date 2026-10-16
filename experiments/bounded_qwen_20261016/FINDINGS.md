# Bounded Qwen experiment findings

The clearest E2 result is a cheaper planner. The short subgoal/action actor used
**73.2% fewer output tokens and 65.8% less evaluator time** than original planning,
with similar development progression. A reliable task-solving improvement is
unproven. Correcting BabyAI instructions alone and adding action-effect feedback
did not improve this panel. The E3 pilot produced **zero progression gain on every
benchmark**, with higher cost.

The frozen E2 actor is **short**. ACTOR_LOCK.json records selection before
calibration or E3; FROZEN.json records the later confirmation protocol. This
completes the bounded cycle, not the entire assignment or teammate's API work.

## Protocol and research connection

All experimental calls used **local Qwen/Qwen3-4B-Instruct-2507-FP8**, cached revision
8591804019c8b22094c3b5b4454e0edc05dffc98, on the existing RTX 5080. The fixed panel
has 10 BabyAI cases (five tasks, seeds 11/12), two Crafter cases (11/12), and six
BabaIsAI tasks at seed 11. Completed comparisons use the same 18 cases and matching
initial text/mission fingerprints. Repeated decisions do not add new layouts.

Naive retains its 128-token cap; planning, grounded and every E2 candidate retain
512. Model defaults are temperature .7, top-p .8, top-k 20. History is 16 text
observations, no images, one previous reasoning trace. One evaluator runs at a
time with unchanged scoring and environment limits. SDK retries are disabled;
grounded's validation retries count as calls. Five-step setup checks are separate.

The assignment and supplied course book were read. Book §§6.1.3 and 10.2.5
motivate immediate subgoals; §6.2.3 discusses external state and the knowing-doing
gap; §7.3.2 discusses action consequences. E3 adapts the actor/critic/memory idea
discussed under Reflexion in §10.4, with bounded within-episode memory. These are
small prompt/harness adaptations, not reproductions of trained planners or full
Reflexion. See AUDIT.md for sources, mechanics and implementation scope.

## Audit findings

Installed MiniGrid and a real environment fixture verify that pickup targets
the tile **directly ahead**, with empty hands. Forward cannot enter a key/ball/box;
toggle does not pick up a key. BALROG says pickup is "below you" and suggests
toggle for pickup. Carrying is exposed as "You carry a ...", contrary to an old
E2 comment. The mechanics candidate fixes those two instruction fragments; it
solved neither direct-pickup case.

The upstream evaluator applies an invalid-action fallback, then records/passes
back the proposal. Our loop preserves reference history semantics while logging
both proposal and execution and aligned pre/post observations. A differential
test against BALROG's actual evaluator verified identical prompts, progression,
steps and scripted token totals, including an invalid command. Reset behavior and
coordinate conventions were audited. Actors never receive hidden state or
BabaIsAI target_plan.

Saved v1 configs retained unused loader defaults. The explicit factory selected
the correct actor and the local client actually disabled transport retries.
Original configs remain preserved with effective-setting sidecars. Version 2
corrects labels without changing prompts/actions/scoring; exact versions/hashes
and the repeated passing parity test are preserved. See
[audit/CONFIGURATION_ERRATUM.md](audit/CONFIGURATION_ERRATUM.md).

## E2 results

Each pass contains 18 cases. Entries are progression percentages; overall gives
each benchmark equal weight. Only Naive and short received two additional passes.

| Actor | Passes | BabyAI | Crafter | BabaIsAI | Overall |
|---|---:|---:|---:|---:|---:|
| Naive | 3 | 13.33 | 28.79 | 16.67 | 19.60 |
| Original planning | 1 | 50.00 | 4.55 | 16.67 | 23.74 |
| Original grounded | 1 | 30.00 | 4.55 | 16.67 | 17.07 |
| Correct mechanics | 1 | 30.00 | 9.09 | 16.67 | 18.59 |
| Short subgoal/action | 3 | 50.00 | 6.82 | 16.67 | 24.49 |
| Previous action/effect | 1 | 30.00 | 6.82 | 16.67 | 17.83 |

First-pass paired differences versus planning: mechanics -5.15 pp, short +0.76 pp,
transition -5.91 pp. Short's three matched differences versus Naive were +7.27,
+5.00 and +2.42 pp (mean +4.90). These selected development results do not
establish significance.

Short's overall SD across passes was zero; Naive's was 2.43 pp. Flat aggregates
masked changed trajectories: short lost an open success and gained a sequence
success in its third pass. BabyAI task-level success exposes that brittleness:

| BabyAI task | Naive, 3 passes | Planning, 1 pass | Short, 3 passes |
|---|---:|---:|---:|
| goto | 50.0 | 100.0 | 100.0 |
| pickup | 16.7 | 0.0 | 50.0 |
| open | 0.0 | 100.0 | 33.3 |
| putnext | 0.0 | 0.0 | 0.0 |
| pickup then goto | 0.0 | 50.0 | 66.7 |

Mean cost per 18-case pass, including evaluator startup/shutdown:

| Actor | Calls | Input tokens | Output tokens | Execution minutes |
|---|---:|---:|---:|---:|
| Naive | 1,870.7 | 2,874,767 | 5,059 | 1.91 |
| Planning | 1,599 | 2,946,219 | 99,499 | 13.63 |
| Short | 1,427.3 | 2,542,742 | 26,676 | 4.66 |

Short retains a cost advantage over planning. Naive remains much cheaper and
stronger in Crafter. The frozen short actor retains the original BabyAI wording
to preserve its tested single-factor definition; the separate mechanics fix and
its negative result remain available.

## Representative evidence

Review selection is systematic: first sorted positive/zero-progression case per
arm/task/outcome, all regressions, and fixed mechanics flags. Full samples are in
audit/systematic_review.json, mechanics_examples.json and regressions.json.

- **Pickup:** short pickup11 step 17 executes "pick up" and observes carrying the
  grey ball. Pickup12 step 63 moves forward into an adjacent yellow key.
- **Corrected instructions:** mechanics open12 step 127 says its subgoal is picking
  up the purple key one step ahead, but executes forward and receives unchanged
  text. The actual sent instruction was verified corrected.
- **Crafter:** planning seed11 step 16 plans to collect wood but moves north while
  already facing a tree. Short retains such errors (seed12 step 12, Move West).
- **Rules:** make_win11 starts with plans to break YOU, followed by long runs of
  repeated "up". No tested actor solves a development rule-construction case.
- **Proxy metrics:** grounded had zero unchanged BabyAI text transitions but lower
  success than planning. Naive emitted 131 unsupported commands across three passes
  while outperforming planning in Crafter. Grounded spent 158 validation calls.
  Smoother trajectories alone do not establish better task solving.

The old completed results reproduce the brief's larger-panel means; their
seeds/task coverage differ, so they were not substituted for matched episodes.
This panel covers six of 40 default BabaIsAI tasks and few layouts. One Qwen model,
single-pass rejected candidates, and stochastic decisions limit generalization.

## E3 paired pilot

Freeze short, then compare one fresh 18-case pass with/without reflection.
Trigger after three identical actions with unchanged full text; cooldown 12;
at most two 96-token reflections and two 400-character lessons per episode.
Do/Sleep/Noop/idle are exempt. Memory clears between episodes.

| Arm | BabyAI | Crafter | BabaIsAI | Overall | Calls | Output tokens | Minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen actor | 70.00 | 4.55 | 16.67 | 30.40 | 1,362 | 25,278 | 4.43 |
| Reflection | 70.00 | 4.55 | 16.67 | 30.40 | 1,460 | 29,232 | 5.52 |

There were 23 reflection calls (12,237 input/748 output tokens, 5.73 seconds).
Changed trajectories also increased actor calls: total calls rose by 98 and
execution time by 24.6%. Unchanged text fell from 842 to 600 without progression
gain. Lessons mostly repeated existing no-repeat advice. Pickup12's apparent
gain occurred with **zero** reflections; sequence12 regressed. One paired pass
cannot distinguish small effects from sampling variation. Stop this generic
reflection configuration; do not claim an E3 improvement. Full evidence is in e3/.

## Confirmation and stopping evidence

After actor lock, 12 separate Baba seed12 episodes completed two full paired
development seed clusters for variance calibration. They are excluded from E2
selection/scoring. Paired overall differences were +23.64 and -9.09 pp.
The rough normal power calculation suggests **42 independent seed clusters**
for a 10 pp effect; its variance estimate is unreliable with two clusters.

The runnable protocol uses all **20 reserved seeds, 100–119**, two decision
passes, all three matched references and short: 1,920 episodes. It explicitly
reports limited power (approximate detectable effect 14.49 pp), averages repeated
layouts before uncertainty estimation, and bootstraps whole paired seed clusters.
It requires complete counts/fingerprints and one final analysis. No held-out
observations have been generated. Read [CONFIRMATION.md](CONFIRMATION.md).

~~~bash
bash experiments/bounded_qwen_20261016/protocol/confirm.sh \
  "$PWD/confirmation_qwen_frozen"
~~~

The command was dry-run validated: 240 unique task/seed cases, four arms,
two passes; frozen hashes and shell syntax pass. This later precision check is
the selected next experiment. It does not promise significance or replace
independent seeds with repetitions.

Completed: **216 scored episodes** (180 E2, 36 E3), **12 calibration** and **3 setup**
episodes. All 231 benchmark attempts completed; none failed/incomplete. Including
13 audit fixture attempts gives **244/300** attempts. One initial unit fixture
registration failure was corrected and remains counted. Consumption was
**86.83/120 benchmark minutes**, including measured audit-fixture execution;
total elapsed time and original deadlines are in STATE.json.

Three of six allowed E2 candidates and one of two E3 configurations were used.
Seventeen new tests and all nine existing E2 tests pass; parity and reflection
checks passed again after final logging changes. Integrity reconciles **20,105
local Qwen calls, 35,895,615 input and 563,271 output tokens**. All calls finished
normally, with no token-limit truncation. Original results/user changes are
preserved. The owned model server is stopped; no owned evaluation remains.
There is no remaining technical blocker. Evidence is sufficient to stop, with a
clear confirmation-precision limitation and no demonstrated E3 task-solving gain.
