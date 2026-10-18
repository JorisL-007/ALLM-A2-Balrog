# Follow-up study: grounded BabyAI actions and history

The follow-up keeps the Exercise 2 planning and Exercise 3 critic controls frozen while testing three off-by-default changes in `BALROG/balrog/agents/custom.py`.

| Condition | Mode | BabyAI guide | Pickup rule | Goal-state line | Text observations |
|---|---|---:|---:|---:|---:|
| `naive` | supplied naive | original | off | off | 16 |
| `plan` | E2 planning | original | off | off | 16 |
| `e3` | E3 triggered critic | original | off | off | 16 |
| `guide_h4` | E2 planning | corrected | off | off | 4 |
| `guide` | E2 planning | corrected | off | off | 16 |
| `guide_h32` | E2 planning | corrected | off | off | 32 |
| `guide_gate` | E2 planning | corrected | on | off | 16 |
| `guide_goal` | E2 planning | corrected | off | on | 16 |

The corrected guide changes only the custom agent's BabyAI instructions. MiniGrid picks up a portable object in the **front cell**, whereas the supplied BALROG guide says **below you** and suggests `toggle` for pickup. The gate substitutes `pick up` only for a proposed `go forward` or `toggle` when the current observation says the mission's exact named target is one step forward and the agent is not carrying it. The goal-state line is a one-call [ReflAct-inspired](https://arxiv.org/html/2505.15182) adaptation, not a reproduction. It disables the E3 triggered critic. In the observed E3 runs, the critic supplied the next action on its trigger step, so it replaced the regular action call rather than adding a second call. The actual MiniGrid action logic is in [MiniGridEnv.step](https://github.com/Farama-Foundation/Minigrid/blob/main/minigrid/minigrid_env.py).

## Run

From the repository root, with the `balrog` and `vllm` conda environments installed:

```bash
bash experiments/run_followup_overnight.sh
```

The command starts or checks the local Qwen3-4B FP8 vLLM server at port 8088, then runs the fixed panels. It stops between episodes after eight hours; run the **same command** again to resume. The 75-episode pilot took 816 seconds, or 10.9 seconds per episode. The complete 1,760-episode study took **5.77 wall hours**, within the projected 5–7 hours. You can run an individual panel with `--panel primary`, `--panel broad`, `--panel crafter`, or `--panel repeat`. `--panel pilot --out experiment_results/followup_pilot` runs development seeds 20–24. `--limit N` limits new episodes for a short check and does not change the experiment settings. `PORT=8088` may be changed if the server uses a different local port.

The fixed evaluation has 1,440 primary episodes (eight conditions, three BabyAI tasks, seeds 30–89, 32 actions), 240 broader-task episodes (four conditions, all five tasks, seeds 100–111, 128 actions), 20 Crafter episodes (two conditions, seeds 120–129, 50 actions), and 60 repeatability episodes (two conditions, `goto`/`pickup`, seeds 30–34 repeated three times). One task, seed, condition, and repeat occupies one directory. Temperature is zero; the history length is the condition setting; all other model and environment settings match within each panel.

Each completed directory contains BALROG's episode JSON and CSV trajectory, `llm_calls.jsonl` with full prompts, raw replies, and token counts, `command.json`, `console.log`, and `complete.json`. The CSV separates model action, harness action (`Action`), override reason, and BALROG's executed fallback action. The runner skips an episode only after checking its trajectory, call count, and token totals. Interrupted `.working` directories are preserved as `.incomplete_*` on restart. A manifest with code hashes and settings prevents mixing versions. The original E2/E3 matrices are left intact.

The complete result has 1,760 `complete.json` files and no `.working` directories. A second invocation with `--limit 1` reported **0 new episodes** after verifying them, confirming resume behavior on the finished study.

The final behavior-affecting source and analysis script are also saved as a [frozen code checkpoint](../experiments/checkpoints/followup_v1_code.tar.gz). The manifest pins the behavior-affecting files by SHA-256; `analysis_sha256.txt` pins the reporting script.

Summarize completed episodes with:

```bash
python3 experiments/analyze_followup.py
```

The overnight command refreshes these summaries automatically, and this command can refresh them on demand. They are `episodes.csv`, `performance_matrix.csv`, `paired_comparisons.csv`, and `repeatability_matrix.csv` in `experiment_results/followup_v1/`. The primary estimand is the paired `pickup` success difference for `guide` versus `plan` on seeds 30–89, with gain/loss/tie counts, a seed bootstrap 95% interval, and an exact McNemar test. The four secondary comparisons use seed-block intervals and Holm-adjusted tests: exact McNemar for the single-task rule comparison and seed-block sign permutations for the three-task goal and history comparisons. `repeat` records are kept separate and never treated as independent task seeds. Loop counts are descriptive trajectory patterns, not proof that the agent failed.

Pilot observations and the final evaluation report are recorded in [followup_notes.md](followup_notes.md) and [followup_report.md](followup_report.md). Portable samples of complete raw prompts and replies are in [pilot samples](followup_raw_llm_samples.jsonl) and [held-out samples](followup_heldout_raw_samples.jsonl); every episode directory retains its full call log.
