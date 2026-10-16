# BALROG Exercises 2 and 3: compact planning and reflection

## Question and method

I tested whether a small persistent plan and feedback-triggered critique improve BabyAI task completion and reduce unproductive action loops at a tolerable inference cost. I read the assignment, the supplied custom agent, BALROG evaluator and logs, and the book's Sections 6.2 and 10.4 before changing the code. The final agent keeps BALROG's existing prompt builder, clients, environment wrappers, evaluator, and result format. A small evaluator patch records proposed versus actually executed fallback actions, and feeds the executed action back to the agent.

Qwen3-4B instruction was the development model. Initial trials used three seeds each for `goto` and `pickup`, plus a three-seed `open` probe. I read the model outputs and observation/action trajectories after each trial, changed one issue at a time, and froze the final code before evaluation on new seeds 11–15. Held-out conditions used the same BabyAI tasks, seeds, text observations, temperature zero, 16-entry text history, and 32-action limit. The naive output cap was 128 tokens; the structured agent used 256 so its required fields could fit. Actual token use is included below. Crafter transfer used seeds 11–13 and 50 actions for all conditions. These panels are small and shorter than BALROG's default episode budgets.

Exercise 1 already supplied broader naive-agent results. Its Qwen3-4B BabyAI aggregate was 32.0% ± 2.0% across three runs of 50 episodes, and Crafter was 18.94% ± 2.62% across three runs of 10 episodes. Those task mixtures and budgets differ from this investigation, so I used them as background and reran naive only on matched tasks, seeds, and action limits.

## Development and adaptation

The supplied planning agent solved 4/6 initial BabyAI episodes but often kept its plan while ineffective actions continued; one `goto` run proposed invalid text for steps 11–26 and BALROG executed fallback forward moves into a wall. A compact four-line output (`SUBGOALS`, `FACTS`, `EXPECTED`, `ACTION`) with bounded persistent state solved 5/6 and reduced mean steps from 16.7 to 9.7. Built-in CoT solved 5/6 on the same small panel, with more output tokens and 22 invalid proposed actions. A stricter expectation instruction then reduced compact planning to 3/6; I reverted it.

The compact agent still alternated left/right between two identical views in a `goto` episode. A first critique design identified the loop but passed the next action back to the normal actor, which repeated a turn. Letting the critique choose the next action broke that particular loop: the development panel rose from 5/6 without reflection to 6/6 with it. A broader door trigger failed in a separate three-episode probe and was reverted. The final critic requires an unchanged observation after repeated interaction with a visible front obstacle, or a four-turn alternating-view cycle; it allows at most three calls per episode, at least six steps apart, and stores at most two short lessons. Repeated actions or zero reward by themselves are never sufficient triggers. The full adaptation record, including unsuccessful trials, is in `research_notes.md`.

## Frozen Qwen3-4B evaluation

| BabyAI condition | Successes / 15 | Mean progression ± sample SD | Mean steps | Mean input / output tokens per episode | Panel elapsed |
|---|---:|---:|---:|---:|---:|
| Naive | 2 | 13.33% ± 35.19% | 30.13 | 17,909 / 89 | 25 s |
| Compact planning (E2) | 6 | 40.00% ± 50.71% | 28.33 | 21,573 / 1,201 | 142 s |
| Reflection enabled (E3) | 6 | 40.00% ± 50.71% | 26.13 | 19,749 / 1,220 | 146 s |

E2 gained five paired successes and lost one versus naive. It used about 27% more total tokens per episode and 5.7 times the panel wall time. The improvement is promising within this action budget, but the latency and wide episode variation prevent a general claim that its cost is acceptable for every use. E3 gained one `goto` success and lost one `pickup` success relative to the same agent with reflection off; 13 outcomes were tied. Its 20 critique calls were included in the token totals. Its panel elapsed time was 4 seconds higher, while total tokens per episode were lower because trajectories differed. Thus the observed loop reduction did not produce an aggregate held-out completion gain.

The most concrete positive E3 trace is seed 11 `goto`: a critique after a turn cycle selected `go forward`, and the grey ball was reached on action 32. In seed 12 `pickup`, a critique selected forward motion and the key was picked up on action 9 rather than action 23 without reflection. The negative cases matter too: seed 13 `pickup` received a forward recommendation despite a wall one step ahead, and failed. Planning without reflection remembered that it carried a blue key in seed 15 `open`, yet kept toggling while the door was several steps away. True memory did not ensure correct spatial use.

As a narrow behavioral check, overlapping windows of alternating turns fell from 145 with planning to 22 with reflection, but blocked-forward pairs rose from 37 to 51 and unchanged locked-door toggle pairs remained high (51 versus 48). These are descriptive counts of observation/action patterns, not independent failures or statistical tests. The critic changed *which* loops occurred more reliably than it changed success.

## Crafter transfer

| Condition, three seeds | Achievements by seed | Mean progression ± sample SD | Mean input / output tokens | Elapsed |
|---|---:|---:|---:|---:|
| Naive | 2, 4, 1 | 10.61% ± 6.94% | 65,487 / 154 | 17 s |
| Compact planning | 1, 1, 1 | 4.55% ± 0 | 67,575 / 2,086 | 58 s |
| Reflection enabled | 1, 1, 1 | 4.55% ± 0 | 70,777 / 2,147 | 59 s |

The planning prompt and BabyAI-specific loop triggers did not transfer well. In one Crafter run, the planner repeatedly used `Do` while facing grass and predicted wood collection; in another, naive collected four achievements while planning persisted with “Collect Wood; Move toward cow” and collected only wood. No Crafter reflection call fired, so the last row tests the reflecting *mode* on Crafter but does not establish whether actual critique helps there. This negative transfer limits the scope of the BabyAI findings.

## Limited comparison with another local model

On six matched held-out `goto`/`pickup` episodes (seeds 11–13), Qwen3-14B-AWQ solved 4/6 with naive actions, 1/6 with compact planning, and 2/6 with reflection. Mean progression sample SDs were 51.64, 40.82, and 51.64 percentage points. Mean input/output tokens were 9,126/52; 23,903/1,530; and 20,118/1,363, and panel times were 16/125/121 seconds. The 14B planner repeatedly turned left/right between the same views for entire episodes. In seed 12 `goto`, a critique chose forward movement at action 5 and the agent reached the key at action 12; the matched planner failed. Its paired `pickup` run still failed after three critiques. The effect of the compact prompt therefore depended strongly on the model, and reflection did not restore naive performance in this small comparison.

On the same six tasks and seeds, Phi-4-mini-instruct solved 0/6 naive, 2/6 planning, and 1/6 reflective (sample SD 0, 51.64, and 40.82 percentage points). Mean input/output tokens were 16,230/99; 14,615/958; and 18,290/1,181, with panel times 15/68/83 seconds. In seed 12 `pickup`, planning succeeded after the key came one step ahead: it tried forward several times and eventually picked it up on action 11. The reflective critic instead chose `turn left` on action 6 when forward left that key in place; it moved away and failed. That is a concrete harmful revision, despite the trigger being grounded in the observation. The first action in both runs was an invalid compound proposal; BALROG executed fallback `go forward`, which is separately shown in the CSV. Some Phi-4 critiques also omitted a standalone `ACTION:` line, leaving the normal actor to choose after the critique. The comparison is small, but it does not support a model-independent gain from reflection.

## Evidence, cost, and limits

`experiment_results/performance_matrix.csv` has task-level and pooled results, and `episodes.csv` links each outcome to a CSV trajectory. Every custom run's `seed_*/llm_calls.jsonl` records the full prompt messages, raw reply, call type, trigger, and input/output tokens. A seven-call portable sample is `research/raw_llm_samples.jsonl`. I inspected both successful and failed trajectories and compared the JSONL call-token sums with every final planning and reflective episode total. The evaluator CSV records both the proposed action and the executed fallback action. Development and held-out observations, exact examples, and rejected adaptations are in `research_notes.md`.

Fifteen held-out BabyAI episodes and three Crafter episodes per condition give high variance and limited power; the other-model checks have only six episodes each, and temperature-zero local inference was not perfectly repeatable across reruns. The tasks and action caps were deliberately selected for a small investigation, not a benchmark-wide claim. On the main Qwen3-4B panel, E2 showed a completion gain over naive at substantial elapsed-time cost. E3 reduced one loop pattern at little additional elapsed time over E2, but gave no net held-out completion gain. The cross-model reversals and the negative Crafter transfer further limit generalization. More seeds and a more reliable spatial action policy are needed before arguing that reflection itself pays off.
