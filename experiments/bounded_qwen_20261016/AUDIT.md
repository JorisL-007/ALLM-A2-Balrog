# Harness audit

Audited the supplied assignment PDF; course book §§6.1.3, 6.2.3, 7.3.2,
10.2.5 and 10.4; BALROG evaluator, agent factory, custom and naive actors,
history builder, model client, config and all three game wrappers; existing E1/E2
launchers, E2 agent/tests and completed result tables. BALROG commit and hashes
are in `provenance.json`. Existing files are preserved.

## Mechanics and observations

* Installed MiniGrid `minigrid_env.py:574–589` picks up the object at `front_pos`
  with empty hands. `forward` cannot enter a key/ball/box. `toggle` invokes the
  object's toggle method; a key is not picked up by toggling. The old instruction
  says “below you” and its tip incorrectly groups pickup with toggle. A real
  installed-environment unit fixture verifies forward/toggle/pickup separately.
* Installed MiniGrid `gen_graph()` emits `You carry a ...` into descriptions;
  the BabyAI wrapper passes that into the agent's text. The old E2 comment that
  carrying is not exposed is inaccurate. Its checks nevertheless do not use
  carrying; changing that checker would be an additional intervention and is
  excluded from these first three candidates.
* BabyAI object offsets are relative to facing. A diagonal object's distance is
  not equivalent to exactly one step forward. Closed/locked doors need toggle;
  keys must be carried for matching locked doors. The instruction correction is
  explicitly game specific, rather than a claim of generic reasoning progress.
* Crafter object coordinates are absolute north/south/east/west. The separately
  exposed facing line identifies the front tile. `Do` interacts with that tile;
  movement also changes facing. Inventory/status is short-term context, retained
  only on the current observation by BALROG's history builder. Repeated `Do`
  or waiting can be legitimate even if the spatial text stays unchanged.
* BabaIsAI coordinates are horizontal right/left and vertical down/up, relative
  to YOU. Its active rules can change object properties. Its wrapper can reset
  internally when YOU breaks, and includes that event in text. Neither actors
  nor the logging client receive its `target_plan`/reference solution. Only the
  public observations, instruction and actual command are used in interventions.

## Evaluation and integrity

* The upstream evaluator obtains an executed command through
  `check_action_validity`, but then overwrites the history/CSV action with the
  original proposal. Thus CSV action is not reliable evidence of actual fallback
  behavior. The new local loop mirrors the reference policy and feedback but
  records both proposed and executed actions and pre/post text independently.
  The transition candidate adds the executed action/effect as its sole factor.
* Upstream CSV observations are **after** the action. Old E2 JSONL observations
  are **before** it. Current `steps.jsonl` explicitly holds both.
* `HistoryPromptBuilder.reset()` does not clear `previous_reasoning`. E2 already
  clears it. A fresh agent is created per episode here, and candidate reset is
  tested, so there is no carryover between cases or arms.
* The upstream custom agent retains `plan` across its inherited reset and does
  not always update reasoning history. We build on the existing E2 subclass of
  BALROG BaseAgent instead; we do not change upstream `custom.py`.
* Fixed eval seed repeats layouts on multiple episodes. Manifest has exactly one
  episode per task/seed. Crafter constructor **and** shared seed are set. Global
  Python/NumPy seeds are set before construction as well as environment reset.
  Initial text/mission fingerprints must match before any paired comparison.
* Scoring is unchanged: BabyAI mission success (0/1), BabaIsAI positive terminal
  reward (0/1), Crafter unlocked achievement fraction out of 22. Overall is the
  equal-weight mean of the three benchmark means, not pooled episode success.
* Full episodes use unchanged installed environment limits (including Crafter's
  configured 2000-step maximum). Only the clearly separate setup check has a
  five-step cap. Failed requests have no automatic transport retry; requests and
  errors are logged, and incomplete episodes are preserved and excluded.
* Model is the cached exact Qwen FP8 revision. Serving flags match the legacy
  launchers: model defaults (temperature .7, top-p .8, top-k 20), seed 0, 16384
  context, flashinfer sampler/deep-gemm disabled. Naive retains 128 tokens;
  references and all three E2 candidates retain 512 per call. Grounded retains
  its one possible validation retry, counted in costs. There is one sequential
  evaluator and one serving model. Worker count is 1 versus legacy 4; historical
  times are therefore context, not a controlled speed comparison.
* The sandbox cannot see the host GPU or model endpoint. Host execution was
  required and successfully exposes the RTX 5080. No packages were upgraded,
  model downloaded, unrelated process stopped, or external evaluation API used.

## Prior evidence and new hypotheses

Completed old panels were read from `exercise1_results/20261015_072525` and
`exercise2/exercise2_results/20261015_202925`. They reproduce the brief's means
(Naive overall 17.72%, planning 13.09%, grounded 12.47%). They use seeds 1–10
or 1–3, not the present 11/12 panel, and cannot be reused as matched episodes.
The smoke results and interrupted CoT directories are not pooled with full runs.

Three planned single-factor E2 candidates, all relative to legacy planning:

1. `mechanics`: replace only the two incorrect BabyAI instruction fragments.
   Hypothesis: correctly describing pickup helps interaction; other games are
   unchanged prompt controls and can still vary through sampling.
2. `short`: replace the four-field planning instruction by one immediate subgoal
   and one action. Hypothesis: less generation devoted to speculative plans may
   reduce cost and plan/action disconnect. The maximum token cap stays 512.
3. `transition`: add the previous executed primitive, whether text changed and
   exact added/removed text lines. This is mechanical one-step feedback with
   no critic call, diagnosis, learned lesson or persistent memory.

These implement small prompt/harness adaptations supported by the course book:
§6.1.3 motivates intermediate subgoals; §6.2.3 discusses the knowing-doing gap
and unreliable state tracking; §7.3.2 discusses action consequences; §10.2.5
encourages small benchmark-specific improvements. They do not reproduce a
trained planner, symbolic simulator, or the full WorldCoder/Reflexion methods.
