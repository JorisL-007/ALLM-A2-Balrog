# BALROG follow-up: grounded actions, history, and goal-state reflection

**Status:** Complete. All 1,760 scheduled held-out episodes finished and passed the runner's result, trajectory, call-count, and token-total checks. The complete run took 5.77 hours; invoking the same command again added zero episodes.

## Design

The primary comparison is corrected-guide planning versus the frozen E2 planner on 60 paired `pickup` seeds (30–89), 32 actions each. The same three BabyAI tasks also include naive, frozen E3 critique, history 4/16/32, corrected guide plus target pickup rule, and corrected guide plus one-call goal-state reflection. Secondary panels test all five BabyAI tasks at up to 128 actions, Crafter at 50 actions, and threefold temperature-zero repeats on five `goto`/`pickup` seeds. All conditions use local Qwen3-4B FP8, temperature zero, 256 output tokens, text observations, and one worker. The [manifest](../experiment_results/followup_v1/manifest.json) pins settings and source hashes.

The guide correction follows [MiniGrid's front-cell pickup implementation](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/minigrid_env.py), replacing BALROG's contradictory below-cell pickup and toggle tip only in the custom-agent conditions. The one-call goal-state line is inspired by [ReflAct](https://arxiv.org/html/2505.15182); the prompt and evaluation here are a small adaptation, not a reproduction of its full method.

## Development evidence

The 75-episode [pilot matrix](../experiment_results/followup_pilot/performance_matrix.csv) showed pickup success in 4/5 frozen-plan, 1/5 corrected-guide, 2/5 guide-plus-rule, and 2/5 guide-plus-goal-state episodes. Open was 0/5 in every pilot condition. These small numbers guided checks and are kept separate from held-out estimates. The rule made one substitution: on [pickup seed 24](../experiment_results/followup_pilot/episodes/babyai/pickup/guide_gate/seed_024/rep_0/llm_calls.jsonl), it replaced a proposed `go forward` with `pick up` while the mission's grey box was directly ahead, and the agent succeeded in five actions. The [guide-only run](../experiment_results/followup_pilot/episodes/babyai/pickup/guide/seed_024/rep_0/llm_calls.jsonl) succeeded in 17. On [pickup seed 20](../experiment_results/followup_pilot/episodes/babyai/pickup/guide/seed_020/rep_0/llm_calls.jsonl), the guide failed in a repeated turn loop while the frozen planner succeeded; the pickup rule could not help because the model never faced the key.

The pilot also found temperature-zero variation: guide-only and guide-plus-rule [seed 23 calls](../experiment_results/followup_pilot/episodes/babyai/pickup/guide_gate/seed_023/rep_0/llm_calls.jsonl) reached an identical prompt but produced different pickup actions. The gate made **zero** substitutions in that seed, so its apparent success difference is not a rule effect. The separate repeatability panel addresses this. The maximum history-32 input call was 1,689 tokens, below the 8,192-token server context.

## Held-out results

The 60-seed primary panel is complete. Its [performance matrix](../experiment_results/followup_v1/performance_matrix.csv), [episode table](../experiment_results/followup_v1/episodes.csv), and [paired comparisons](../experiment_results/followup_v1/paired_comparisons.csv) are generated from verified episode directories. BabyAI progression is binary mission success; the sample SD is consequently about 0.5 for many conditions. The primary analysis uses distinct paired task seeds. Repeated runs of seeds 30–34 are analyzed separately.

| Condition | `goto` / 60 | `pickup` / 60 | `open` / 60 | Total tokens, 180 episodes | Elapsed min |
|---|---:|---:|---:|---:|---:|
| Naive | 28 | 26 | 4 | 2.46M | 7.5 |
| Frozen E2 plan | 40 | 34 | 1 | 3.21M | 25.9 |
| Frozen E3 critic | 45 | 35 | 0 | 3.24M | 27.7 |
| Corrected guide, history 4 | 39 | 33 | 13 | 2.14M | 25.0 |
| Corrected guide, history 16 | 40 | 29 | 3 | 3.45M | 27.3 |
| Corrected guide, history 32 | 38 | 26 | 2 | 4.04M | 27.3 |
| Guide + pickup rule | 38 | 29 | 5 | 3.42M | 26.6 |
| Guide + goal-state line | 39 | 29 | 3 | 3.71M | 37.3 |

**Primary test:** corrected guide versus frozen E2 planning on paired `pickup` seeds: 29/60 versus 34/60 successes; guide minus plan = **−8.3 percentage points**, 95% paired seed-bootstrap interval **−25.0 to +8.3 points**; 10 guide gains, 15 losses, 35 ties; exact McNemar **p=0.424**. The guide did not improve pickup success in this panel, and the interval still permits a modest benefit or a substantial loss. It also changed navigation behavior on `goto` and `open`, so its impact is wider than the pickup description alone.

**Secondary tests:** guide + pickup rule versus guide alone had zero net pickup success difference (29/60 each; 3 gains, 3 losses, 54 ties; exact McNemar p=1). It made 15 logged substitutions, all in pickup episodes. Guide + goal-state line changed three-task success by −0.6 points (seed-block 95% interval −7.2 to +6.1; adjusted p=1), while increasing total tokens from 3.45M to 3.71M and elapsed time from 27.3 to 37.3 minutes. History 4 versus guide history 16 changed three-task success by +7.2 points (interval 0 to +14.4; seed-block permutation p=0.080, Holm p=0.320) and used 2.14M instead of 3.45M tokens. History 32 changed it by −3.3 points (interval −8.9 to +1.7; Holm p=0.941) and used 4.04M tokens. The history-4 `open` gain, 13/60 versus 3/60, is a notable **exploratory task-specific** result: paired +16.7 points, 12 gains and 2 losses, exact McNemar p=0.0129 before any wider family of exploratory tests. It needs independent replication.

All 15 primary pickup substitutions occurred on different episodes and were followed by mission success and an observation that the agent carried the named target. On those seeds, guide-only failed three times; the other 12 guide-only runs also succeeded, often after more actions. The rule directly fixes a local wrong proposal, but its **net 60-seed success effect was zero** because outcomes also changed in non-intervention episodes. Conditioning on rule activation does not produce a causal success-rate estimate.

The frozen E3 critic had 80 successes across the 180 primary episodes versus 75 for frozen planning, with 197 reflection-mode calls. Every triggered reflection in this panel supplied the next action directly, so total E3 calls equaled its 4,031 actions; there were no extra calls on top of those actions. Its total tokens were 3.24M versus 3.21M and elapsed time 27.7 versus 25.9 minutes. Descriptive alternating-turn windows fell from 1,210 to 156. These overlapping windows describe behavior, not independent mission outcomes. The critic's success difference is small and task dependent (`goto` 45 versus 40, `pickup` 35 versus 34, `open` 0 versus 1).

## Broader BabyAI transfer: 12 seeds, up to 128 actions

| Condition | `goto` | `pickup` | `open` | `putnext` | `pick_up_seq_go_to` | Total / 60 | Tokens | Elapsed min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen E2 plan | 10 | 7 | 1 | 0 | 2 | 20 | 4.59M | 28.7 |
| Corrected guide | 8 | 8 | 1 | 0 | 4 | 21 | 4.61M | 28.4 |
| Guide + pickup rule | 6 | 7 | 1 | 0 | 6 | 20 | 4.79M | 29.1 |
| Guide + goal-state line | 7 | 4 | 1 | 0 | 1 | 13 | 5.43M | 42.7 |

`putnext` had zero success in all four conditions; it remains a task floor even at 128 actions. `open` had one success per condition. The compound task had some successes only with the longer budget; inspected examples are below. These 12-seed per-task results are descriptive and variable (a 6/12 success rate has sample SD 0.52). The narrow rule made five transfer substitutions: four on `pickup`, one on a compound mission that began `pick up`. On compound seed 102, guide + rule succeeded without any substitution because its mission began `go to`; that difference is run variation, not rule evidence.

## Crafter transfer: 10 seeds, 50 actions

The [Crafter episode records](../experiment_results/followup_v1/episodes.csv) show mean achievement progression 0.0591 (sample SD 0.0307) for frozen planning and 0.0500 (SD 0.0144) for the one-call goal-state condition. The paired difference was −0.0091 of the 22-achievement scale, with a seed-bootstrap 95% interval of −0.0227 to 0. No episode ended in a terminal success; all used 50 actions. Goal-state reflection used 734,350 versus 696,416 tokens and 247 versus 185 elapsed seconds. On [seed 122](../experiment_results/followup_v1/episodes/crafter/default/plan/seed_122/rep_0/llm_calls.jsonl), planning earned two achievements after moving toward a tree and acting nearby, while [goal-state reflection](../experiment_results/followup_v1/episodes/crafter/default/guide_goal/seed_122/rep_0/llm_calls.jsonl) moved west first and earned one. This small transfer panel provides no evidence of a benefit that justifies the extra inference cost.

## Temperature-zero repeatability

The [repeatability matrix](../experiment_results/followup_v1/repeatability_matrix.csv) keeps the primary run and three repeats separate for each of 20 task-seed-condition cells (`goto`/`pickup`, seeds 30–34, frozen plan and guide + rule). Across the **three consecutive repeats**, success never changed within a cell, although action count changed in 2/20 cells. Comparing those repeats with the earlier primary run, success changed in 3/20 cells and executed action sequences differed in 12/20. The primary and repeats were separated by hours; the three repeats of a cell were consecutive, so this audit does not quantify all time effects.

The [primary pickup seed-31 planner call](../experiment_results/followup_v1/episodes/babyai/pickup/plan/seed_031/rep_0/llm_calls.jsonl) and its [first repeat](../experiment_results/followup_v1/repeat/babyai/pickup/plan/seed_031/rep_1/llm_calls.jsonl) had identical first prompts and the same grey-ball mission. The primary proposed `go forward` and succeeded in 16 actions; the repeat proposed `turn right` and failed after 32, as did repeats two and three. On pickup seed 32, the primary planner failed but all three repeats succeeded in eight actions, again with identical first prompts and different replies. Thus temperature zero was not perfectly reproducible in this setup. The 60-seed paired interval reflects one rollout per condition and does not separately model this run-to-run variation.

## Cost, validity, and interpretation

The full study took **5.77 wall hours** (3.41 hours primary, 2.15 hours broader BabyAI, 0.12 hours Crafter, 0.09 hours repeats by summed episode time). It recorded 55,963 LLM calls, including 197 reflection-mode calls, and 44.57M input plus 2.51M output tokens across all panels. About 349 MB of per-episode trajectories and full raw calls were saved. The [manifest](../experiment_results/followup_v1/manifest.json) records model settings and behavior-affecting source hashes; the [analysis hash](../experiment_results/followup_v1/analysis_sha256.txt) records the reporting script version. A second invocation of `bash experiments/run_followup_overnight.sh --limit 1` returned **0 new episodes**, confirming completed-episode resume behavior. A compact [pilot raw-call sample](followup_raw_llm_samples.jsonl) and [held-out raw-call sample](followup_heldout_raw_samples.jsonl) are available for manual inspection; complete prompts and replies remain beside every episode.

The [frozen source checkpoint](../experiments/checkpoints/followup_v1_code.tar.gz) preserves the evaluated agent, runner, evaluator logging, prompt builder, environment guide, and analysis script together. The original E2/E3 performance matrices remain separate and unchanged.

The strongest practical follow-up is **history length 4**, not the corrected guide or one-call goal-state line. It reduced tokens by 38% versus guide history 16 and showed an exploratory `open` gain, but the prespecified three-task comparison did not survive multiplicity adjustment. A new predeclared `open` panel and a backend determinism audit would test whether the apparent gain persists. The pickup rule is mechanistically correct and bounded, with clear immediate-action benefits when the target is ahead, but it did not improve overall success in this panel. The goal-state line did not improve primary or transfer success and increased cost. The E3 critic sharply reduced observed turn-loop windows at nearly the same token cost, while its success improvement stayed small and task dependent.

### Inspected primary trajectories

- On `pickup` seed 32, [frozen planning calls](../experiment_results/followup_v1/episodes/babyai/pickup/plan/seed_032/rep_0/llm_calls.jsonl) began with `go forward`, then spent the 32-action budget in left/right turns and failed. [Corrected-guide calls](../experiment_results/followup_v1/episodes/babyai/pickup/guide/seed_032/rep_0/llm_calls.jsonl) reached the grey key, briefly repeated a blocked forward action, then picked it up at action 22.
- On `pickup` seed 34, [frozen planning calls](../experiment_results/followup_v1/episodes/babyai/pickup/plan/seed_034/rep_0/llm_calls.jsonl) picked up the yellow ball at action 12 after several blocked forward proposals. [Corrected-guide calls](../experiment_results/followup_v1/episodes/babyai/pickup/guide/seed_034/rep_0/llm_calls.jsonl) alternated left/right turns for much of the budget and failed. These two seeds show gains and losses; neither alone estimates the guide's effect.
- On `pickup` seed 33, the [guide-plus-rule call](../experiment_results/followup_v1/episodes/babyai/pickup/guide_gate/seed_033/rep_0/llm_calls.jsonl) proposed `go forward` with the named yellow key one step ahead. The logged harness action and BALROG executed action were `pick up`; the next observation said `You carry a yellow key`, ending the episode successfully.
- On `goto` seed 31, [guide-only](../experiment_results/followup_v1/episodes/babyai/goto/guide/seed_031/rep_0/llm_calls.jsonl) eventually escaped a turn loop and reached the grey ball at action 23, while [one-call goal-state reflection](../experiment_results/followup_v1/episodes/babyai/goto/guide_goal/seed_031/rep_0/llm_calls.jsonl) kept alternating turns through action 32 despite repeatedly stating that no progress was made.
- On `open` seed 35, the [history-4 calls](../experiment_results/followup_v1/episodes/babyai/open/guide_h4/seed_035/rep_0/llm_calls.jsonl) picked up a purple key and opened the matching door at action 9. The [history-16 guide calls](../experiment_results/followup_v1/episodes/babyai/open/guide/seed_035/rep_0/llm_calls.jsonl) also picked up the key, but walked away and later toggled while the locked door was not directly ahead, failing at action 32. This is a concrete history-condition gain; temperature-zero response variation remains a possible contributor.

### Inspected broader-task trajectories

- On `pick_up_seq_go_to` seed 100, the [frozen planner](../experiment_results/followup_v1/episodes/babyai/pick_up_seq_go_to/plan/seed_100/rep_0/llm_calls.jsonl) eventually picked up the blue ball at action 105 and reached the purple key at action 107. The long path illustrates why the 32-action primary budget cannot assess this compound mission.
- On compound-task seed 102, [guide + pickup rule](../experiment_results/followup_v1/episodes/babyai/pick_up_seq_go_to/guide_gate/seed_102/rep_0/llm_calls.jsonl) escaped an early turn loop and completed the mission in 51 actions, while [guide-only](../experiment_results/followup_v1/episodes/babyai/pick_up_seq_go_to/guide/seed_102/rep_0/llm_calls.jsonl) stayed in the loop through action 128. The mission began `go to ... after you pick up ...`, so the narrow rule made **zero** substitutions. This outcome difference is not evidence that the rule helped.
- On `putnext` seed 100, the [frozen planner](../experiment_results/followup_v1/episodes/babyai/putnext/plan/seed_100/rep_0/llm_calls.jsonl) initially alternated left/right and later repeatedly proposed forward moves with a wall directly beside it. It failed at action 128, representative of the early task floor.
