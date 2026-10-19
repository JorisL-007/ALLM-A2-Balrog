# Assignment 2 report outline

**Proposed title:** *When planning helps—and when it hurts: reasoning and self-reflection in BALROG*

**Central research question:** Do compact persistent plans and feedback-triggered critique improve task completion and reduce unproductive action loops at an acceptable inference cost?

**Central finding:** Effects depend on the model, task, and harness. Compact planning showed a modest Qwen improvement but severely reduced Luna performance. Critique often broke planning loops, yet did not consistently outperform the supplied naive agent. Negative results and observed failure mechanisms are central findings.

This outline follows the questions in [the assignment](docs/Ass2Reasoning.pdf). Keep the actual chronology: baselines → small planning experiments → critique development → joint harness evaluation → matched CoT and model-transfer study. For each development iteration, use **observed problem → hypothesis → change → experiment → observation → decision**.

## Abstract

Write last, in one short paragraph: research question, BALROG environments and local/cloud models, the planning and critique mechanisms, principal paired results, and the conclusion that more explicit reasoning does not guarantee better actions. Mention both effectiveness and inference cost.

## 1. Introduction and background

- Explain why sequential games test observation use, memory, action selection, and recovery from mistakes.
- Introduce BALROG and distinguish model capability from the behavior of its surrounding agent code.
- Relate the work to the assigned book chapters 6–9 and the chapter 10 discussion, BALROG, ReAct, Reflexion, and the later ReflAct-inspired experiment. Describe the latter as an adaptation, not a reproduction.
- State three questions: Does planning beat naive/CoT? Does critique repair failures economically? Do effects transfer across models and environments?

## 2. Experimental design and evidence

- **Environment scope:** E1 used all five installed BabyAI tasks (`goto`, `pickup`, `open`, `putnext`, `pick_up_seq_go_to`), Crafter, and 40 BabaIsAI tasks. E2/E3 concentrated on the first three BabyAI tasks, with broader-task and smaller Crafter transfer panels. TextWorld, MiniHack, and NetHack were not executed.
- **Models:** list exact model IDs, local/cloud hosting, quantization, and their role. E1 local: Qwen3-4B, Qwen3-14B, Phi-4-mini. Imported cloud baselines: GPT-4.1 mini and Gemini 3.5 Flash-Lite. Final matched transfer: Qwen3-4B, GPT-6 Luna, and OLMo-2-0425-1B-Instruct (BF16 after Q8 startup failed). Label the teammate's incomplete Qwen3.5-2B evidence separately.
- **Metrics:** define progression separately for each environment: BabyAI mission completion, Crafter achievements divided by 22, and the installed BabaIsAI wrapper's progression definition. Report task success/progression, steps, loop patterns, invalid proposals, interventions, tokens, and elapsed time. Do not interpret a cross-environment average as a success probability.
- **Fairness:** distinguish E1 model defaults/native action limits from later temperature-zero/capped protocols. Give seeds, action/output caps, history, worker counts, memory reset, and which existing controls were reused. Count actor and critic calls, including an extra actor call when critique lacks an action. Keep proposed, harness-submitted, and executed fallback actions distinct.
- **Uncertainty:** distinguish SD across run means from episode SD. Pair on task and seed; do not pool repeated seeds as new puzzles. Explain bootstrap intervals, seed-block permutations, McNemar where used, and Holm adjustment within each prespecified comparison family. Discuss the separate repeatability experiment and temperature-zero variability.
- **Evidence:** full prompt/reply logs, trajectories, manifests, source checkpoints, and manual reading of both successes and failures. Provider refusals remain missing observations.

**Table 1:** Compact protocol table by study stage; retain detailed settings in an appendix.

## 3. Exercise 1 — baseline behavior

### 3.1 Models, benchmarks, and performance matrix

- Explain why these models and environments were selected under available compute and API resources.
- Present the three-model local naive matrix: 1,620 episodes, three repetitions of the same seed panels. State that its SD measures variation across three run means.
- Present the teammate's cloud-model matrix separately, with attribution, differing budgets/settings, and evidence limitations. GPT BabaIsAI is supported by summaries; the reported Qwen2B score is not fully verified.
- Recover the 245 historical CoT episodes by complete environment panel: two BabyAI, two Crafter, one BabaIsAI, plus five partial BabaIsAI episodes. Do not label the whole interrupted sweep complete.

### 3.2 What the trajectories taught us

- Show wall collisions, alternating turns, malformed actions, and pickup misunderstandings using short observation → reply → executed-action examples.
- Explain what different environments expose and why model size alone did not predict the observed ranking.
- Answer whether the evidence supports reliable reasoning, what failed during setup/running, and how these observations motivated E2.
- State that supplied few-shot and robust variants were not evaluated; CoT was evaluated in both historical and later matched panels.

**Tables 2–3:** Local baseline matrix and separately labeled cloud contribution.

Evidence: [E1 guide](exercises/exercise1_baselines/README.md).

## 4. Exercise 2 — compact planning and grounded actions

### 4.1 Idea and implementation

Begin with the required two-sentence description: “The agent carries a short plan, observed facts, and an expected outcome between actions. It compares them with the new observation before choosing its next action.” Explain that the general structure is reusable, while pickup instructions and action substitution are BabyAI-specific and development focused on Qwen4.

Show a short exact prompt excerpt with `SUBGOALS`, `FACTS`, `EXPECTED`, and `ACTION`. Describe the memory bounds and reset, and distinguish observation-supported historical facts from a reliable current spatial map.

### 4.2 Iterative evidence

- Supplied planning → compact pilot: 4/6 → 5/6 development successes; CoT also 5/6 at greater cost. This small panel does not establish superiority.
- Stricter-expectation trial: 3/6, rejected; logs survive but its source checkpoint is unavailable.
- Initial held-out Qwen panel: naive 2/15 versus planning 6/15, with greater inference cost and mixed transfer.
- Later guide correction, narrow pickup substitution, and history lengths 4/16/32: explain the exact behavioral changes, then refer to the joint evaluation in Section 6.

Answer the assignment's questions about generic versus benchmark-specific improvements, implementation difficulty, brittleness, and which model/task benefited most. Report task-level regressions alongside gains.

**Table 4:** E2 development decisions and initial evaluation matrix. **Prompt box 1:** Frozen planning format.

Evidence: [E2 guide](exercises/exercise2_planning/README.md).

## 5. Exercise 3 — feedback-triggered self-reflection

### 5.1 Idea and implementation

Required two-sentence description: “When observations show a blocked interaction or a repeated two-view turn loop, a critic diagnoses the failure and revises the plan. It can select the next action and retain a small lesson memory for the rest of the episode.”

Show `CRITIQUE`, `REVISED SUBGOALS`, `LESSON`, `EXPECTED`, and `ACTION`. Explain the maximum three critiques, six-step spacing, two bounded lessons, reset, and missing-action fallback. Repetition or zero reward alone does not trigger critique.

### 5.2 Iterative evidence

- Plan-only critique diagnosed a loop but the actor repeated it; action-selecting critique changed behavior and was retained.
- Broader door trigger was rejected after failing the development door cases.
- Initial held-out planning and critique both solved 6/15; alternating-turn windows fell 145 → 22. Explain why overlapping loop windows are not independent failures.
- Introduce the later one-call `STATE_VS_GOAL` prompt as the ReflAct-inspired alternative, with the extra critic disabled.

**Table 5:** E3 variants and matched performance/cost. **Prompt box 2:** Critic format and trigger example.

Evidence: [E3 guide](exercises/exercise3_reflection/README.md).

## 6. Joint E2/E3 evaluation and transfer

### 6.1 Larger Qwen harness study

- Describe the 1,760-episode schedule: 1,440 primary, 240 broader-task, 20 Crafter, and 60 separate repeatability episodes.
- Primary Qwen totals: naive 58/180, planning 75/180, critique 80/180. Show per-task results so the door-task weakness remains visible.
- Prespecified guide-versus-plan pickup result: 29/60 versus 34/60; −8.3 points, 95% interval [−25.0, +8.3], McNemar p=.424.
- Pickup rule: 15 immediate substitutions succeeded, but overall pickup success equaled guide-only. Goal-state: 71/180 versus guide 72/180, at greater cost. History4 reduced tokens and showed an exploratory door gain, but its overall comparison did not survive adjustment.
- Discuss broader-task floors, weak Crafter transfer, and the repeatability results without claiming that equal seeds ensure deterministic outputs.

### 6.2 Supplied CoT control and model transfer

Present the completed study: 1,993/1,995 verified evaluation observations, including 550 reused Qwen controls, plus 18 excluded pilots. Two Luna CoT provider refusals remain missing. These are not 1,993 newly independent puzzles.

| BabyAI successes | Naive | Supplied CoT | Planning | Critique |
|---|---:|---:|---:|---:|
| Qwen3-4B | 58/180 | 41/90 | 75/180 | 80/180 |
| GPT-6 Luna | 128/180 | 38/89 | 28/180 | 105/180 |
| OLMo-2-1B | 0/180 | 0/90 | 3/180 | 3/180 |

Explain that CoT uses fewer seeds: its formal Luna comparisons use 29 complete seed blocks (87 task-seed pairs), excluding seed50 as a block. Raw percentages in this table are descriptive until restricted to matched cases.

- Qwen: modest planning/critique gains, but no overall contrast survives the twelve-test adjustment; many CoT outputs hit the 256-token cap.
- Luna: planning-minus-naive −55.6 points (95% CI [−63.9, −47.2]); critique-minus-planning +42.8 ([+33.9, +51.7]); both survive adjustment. Critique still trails naive. CoT is also below naive on the matched subset, with little truncation.
- OLMo: near-zero task performance limits what can be inferred about reflection.
- Crafter: report progression and matched denominators separately. No frozen critic triggered, so this does not demonstrate the efficacy of an active Crafter critique mechanism.
- Cost: compare input/output tokens, calls, steps, and elapsed time. Luna's recorded total cost estimate is $1.5375 including pilots and incomplete attempts. A critic call that supplies the next action can replace an ordinary actor call; it is not automatically an additional call per step.

**Tables 6–7:** Final model/agent matrix and paired effects with intervals, gain/loss/tie counts, and adjusted tests. **Figure 1:** Success-versus-cost comparison. **Figure 2:** One useful critique and one missed loop, using actual trajectory excerpts.

Evidence: [follow-up results](experiment_results/followup_v1/performance_matrix.csv), [transfer results](experiment_results/refactored_v1/model_transfer_luna_v1/analysis/overall_matrix.csv), [paired tests](experiment_results/refactored_v1/model_transfer_luna_v1/analysis/paired_comparisons.csv), [inspection notes](research/model_transfer_verification/postrun_inspection.txt).

## 7. Discussion and critical reflection

- Explain the difference between useful reasoning text and useful executed actions. True remembered facts and a plausible critique can coexist with ineffective behavior.
- Use Luna pickup seed51 to show critique breaking a planning loop, while naive solves faster. Use seed64 to show 32 same-direction turns with no critique trigger.
- Discuss brittleness across models, prompt wording, history, decoding variation, parsing, and environment semantics. Avoid attributing all paired differences solely to critic calls.
- Assess strengths: iterative trajectory reading, frozen comparisons, cost accounting, reproducible logs, negative findings, and local/cloud coverage.
- Assess limitations: small development panels, unequal early output caps, reused seeds, incomplete imported evidence, provider refusals, task floors, narrow triggers, and absent few-shot/robust controls. Latency comparisons across historical runs are descriptive.
- Prioritize future work: independent seed replication, a revised same-direction-cycle trigger tested first on development seeds, history4 door replication, and an actual Crafter critique mechanism. Do not portray proposals as completed experiments.
- Answer directly: Are agentic LLMs suited to reasoning problems? Is self-reflection useful and stable? What would we change with more time?
- Optional bonus: fewer than five sentences comparing verbal LLM reflection with human reflection, clearly marked as philosophical interpretation.

## 8. Conclusion

Answer the three research questions without introducing new results. Emphasize model-dependent planning effects, substantial behavioral repair by critique in some cases, and the absence of a consistent advantage over naive across models and environments. State the cost trade-off and the importance of grounded action feedback and external evaluation.

## References and appendices

- References: assignment book, BALROG, ReAct, Reflexion, ReflAct, and implementation sources actually used; verify bibliographic details before final submission.
- Appendix A: complete model/settings/task/seed tables and metric definitions.
- Appendix B: exact prompts, version map, bounded memory/trigger pseudocode, and unavailable-source limitations.
- Appendix C: full performance matrices, repeatability and statistical methods, cost details, and missing-case records.
- Appendix D: linked success/failure trajectories, commands, and [result-file guide](RESULTS_GUIDE.md).
- Submission checklist: `Ass2-1.zip`, `Ass2-2.zip`, and `Ass2-3.zip`, each with scripts, prompts, code, and a README as required. Verify packages exist and reproduce the named versions; use placeholders for author names, student IDs, contribution split, hardware details not yet verified, and any missing submission metadata.

Suggested emphasis, rather than an assignment-imposed page limit: brief background/setup; most space for E2/E3 mechanisms, matched results, inspected failures, and critical reflection. Put exhaustive matrices and commands in appendices.
