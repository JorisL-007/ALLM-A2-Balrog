# Trajectories read during analysis

The CSV `Reasoning` column keeps the model's full visible output; `Action` is its proposed action and `Executed Action` is BALROG's actual action after any fallback. The `Observation` column is the result after that executed action. Selected runs below are copied into the exercise packages so they can be inspected without navigating the full result tree. Matching `llm_calls.jsonl` files contain full prompts and raw replies for custom-agent calls.

| Question | Trajectory | What to inspect |
|---|---|---|
| Supplied E2 failure | [seed 3 `goto`](../deliverables/Ass2-2/evidence/e2_supplied_seed_3_goto/goto_run_00.csv) | Invalid proposed actions and fallback forward moves into an unchanged wall. |
| Compact E2 success | [seed 3 `pickup`](../deliverables/Ass2-2/evidence/e2_compact_v1_seed_3_pickup/pickup_run_00.csv) | Approach to the red box and pickup at one step forward. |
| Memory helped but toggling lingered | [held-out seed 14 `open`](../deliverables/Ass2-2/evidence/heldout_qwen_plan_seed_14_open/open_run_00.csv) | Blue-key carrying fact, ineffective toggles, eventual success. |
| Memory did not fix geometry | [held-out seed 15 `open`](../deliverables/Ass2-2/evidence/heldout_qwen_plan_seed_15_open/open_run_00.csv) | Door remains distant while toggle repeats. |
| Paired reflection gain | [planning seed 11 `goto`](../deliverables/Ass2-3/evidence/heldout_qwen_plan_seed_11_goto/goto_run_00.csv), [reflection seed 11 `goto`](../deliverables/Ass2-3/evidence/heldout_qwen_reflect_seed_11_goto/goto_run_00.csv) | Critique selects forward movement after a turn cycle; reflective run succeeds at the cap. |
| Wrong critique | [reflection seed 13 `pickup`](../deliverables/Ass2-3/evidence/heldout_qwen_reflect_seed_13_pickup/pickup_run_00.csv) | Critic recommends moving into a front wall. |
| Harmful transfer | [Phi-4 planning seed 12 `pickup`](../deliverables/Ass2-3/evidence/cross_phi4_plan_seed_12_pickup/pickup_run_00.csv), [reflection seed 12 `pickup`](../deliverables/Ass2-3/evidence/cross_phi4_reflect_seed_12_pickup/pickup_run_00.csv) | Planning finally picks up an adjacent key; critique turns away from it. Both begin with an invalid proposed compound action and fallback forward move. |

For every other episode, `experiment_results/episodes.csv` gives the full trajectory path. The sample raw calls in `research/raw_llm_samples.jsonl` include a `source` field pointing to their complete call log.
