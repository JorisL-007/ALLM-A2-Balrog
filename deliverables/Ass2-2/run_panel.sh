#!/usr/bin/env bash
# Example: bash experiments/run_panel.sh supplied custom goto,pickup 1,2,3 32 256
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$1"
AGENT="$2"
TASK_NAMES="$3"
SEEDS="$4"
STEPS="$5"
TOKENS="$6"
MODE="${7:-planning}"
ENVIRONMENT="${8:-babyai}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}"
PORT="${PORT:-8088}"
OUT="$ROOT/experiment_results/$TAG"

mkdir -p "$OUT"
printf 'agent=%s\nenvironment=%s\ntasks=%s\nseeds=%s\nmax_steps=%s\nmax_tokens=%s\nmode=%s\nmodel=%s\n' \
  "$AGENT" "$ENVIRONMENT" "$TASK_NAMES" "$SEEDS" "$STEPS" "$TOKENS" "$MODE" "$MODEL" > "$OUT/settings.txt"
sha256sum "$ROOT/BALROG/balrog/agents/custom.py" "$ROOT/BALROG/balrog/evaluator.py" > "$OUT/code_sha256.txt"
printf 'seed,elapsed_seconds\n' > "$OUT/timing.csv"

IFS=, read -ra TASK_ARRAY <<< "$TASK_NAMES"
TASK_OVERRIDE='['
for task in "${TASK_ARRAY[@]}"; do
  [[ "$TASK_OVERRIDE" == '[' ]] || TASK_OVERRIDE+=','
  TASK_OVERRIDE+="BabyAI-MixedTrainLocal-v0/$task"
done
TASK_OVERRIDE+=']'

IFS=, read -ra SEED_ARRAY <<< "$SEEDS"
for seed in "${SEED_ARRAY[@]}"; do
  SEED_OUT="$OUT/seed_$seed"
  mkdir -p "$SEED_OUT"
  start=$SECONDS
  env_args=("envs.names=$ENVIRONMENT" "envs.env_kwargs.seed=$seed" "eval.num_episodes.$ENVIRONMENT=1")
  if [[ "$ENVIRONMENT" == babyai ]]; then
    env_args+=("tasks.babyai_tasks=$TASK_OVERRIDE")
  elif [[ "$ENVIRONMENT" == crafter ]]; then
    env_args+=("envs.crafter_kwargs.seed=$seed")
  fi
  (
    cd "$ROOT/BALROG"
    BALROG_CUSTOM_MODE="$MODE" BALROG_LLM_LOG="$SEED_OUT/llm_calls.jsonl" \
      conda run --no-capture-output -n balrog python eval.py \
      agent.type="$AGENT" agent.max_text_history=16 agent.max_image_history=0 \
      agent.remember_cot=True agent.max_cot_history=1 \
      "${env_args[@]}" \
      eval.num_workers=1 "eval.max_steps_per_episode=$STEPS" \
      eval.save_trajectories=True "eval.resume_from=$SEED_OUT" \
      client.client_name=vllm "client.model_id=$MODEL" \
      "client.base_url=http://127.0.0.1:$PORT/v1" \
      client.generate_kwargs.temperature=0 \
      "client.generate_kwargs.max_tokens=$TOKENS" \
      client.generate_kwargs.thinking_budget=null
  ) > "$SEED_OUT/console.log" 2>&1
  printf '%s,%s\n' "$seed" "$((SECONDS-start))" >> "$OUT/timing.csv"
  echo "$TAG seed $seed: $((SECONDS-start)) s"
done
