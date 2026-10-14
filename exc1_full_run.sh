#!/usr/bin/env bash
# Run beside BALROG/eval.py: bash run_ex1_local_final.sh
# Three repetitions of the SAME seed panels for each model and agent.
# Default panels preserve BALROG episode totals: 50 BabyAI, 10 Crafter, 120 BabaIsAI.
# To match the teammate's full panels: BABYAI_SEEDS=25 BABAISAI_SEEDS=5 bash run_ex1_local_final.sh
# Model generation defaults are used; output limits match the teammate (Naive=128, CoT=512).
# Fixed environment seeds reproduce layouts, not necessarily stochastic LLM outputs.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BALROG_DIR="${BALROG_DIR:-$ROOT/BALROG}"
VLLM_ENV="${VLLM_ENV:-vllm}"
BALROG_ENV="${BALROG_ENV:-balrog}"
PORT="${PORT:-8088}"
WORKERS="${WORKERS:-4}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/exercise1_results/$SWEEP_ID"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
VLLM_PID=""
MODELS=(
    'qwen4-instruct|Qwen/Qwen3-4B-Instruct-2507-FP8|16384'
    'qwen14-awq|Qwen/Qwen3-14B-AWQ|8192'
    'phi4-instruct|microsoft/Phi-4-mini-instruct|8192'
)
AGENTS=(naive cot)

[[ -f "$BALROG_DIR/eval.py" ]] || { echo "Missing $BALROG_DIR/eval.py" >&2; exit 1; }
for bin in conda curl setsid python3; do
    command -v "$bin" >/dev/null || { echo "Missing program: $bin" >&2; exit 1; }
done
mkdir -p "$OUT"

stop_server() {
    if [[ -n "$VLLM_PID" ]]; then
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        for ((i=0; i<20; i++)); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
    local model="$1" start=$SECONDS
    while (( SECONDS - start < STARTUP_TIMEOUT )); do
        if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
            python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(m["id"]==sys.argv[1] for m in d["data"]) else 1)' "$model" 2>/dev/null; then
            return 0
        fi
        kill -0 "$VLLM_PID" 2>/dev/null || return 1
        sleep 5
    done
    return 1
}

# BALROG's fixed env seed is reused for EVERY episode in one eval.py call.
# Run one episode per task/seed, then aggregate the full panel per repetition.
# Read task counts and panel sizes from the installed BALROG configuration.
conda run --no-capture-output -n "$BALROG_ENV" python - \
    "$BALROG_DIR" "${BABYAI_SEEDS:-}" "${CRAFTER_SEEDS:-}" "${BABAISAI_SEEDS:-}" \
    > "$OUT/seed_panels.new.tsv" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load(Path(sys.argv[1]) / 'balrog/config/config.yaml')
for env, override in zip(('babyai', 'crafter', 'babaisai'), sys.argv[2:]):
    seeds = int(override or cfg.eval.num_episodes[env])
    if seeds < 1:
        raise ValueError('Seed panel sizes must be positive')
    print(f'{env}\t{seeds}\t{len(cfg.tasks[env + "_tasks"])}')
PY
if [[ -f "$OUT/seed_panels.tsv" ]] && ! cmp -s "$OUT/seed_panels.tsv" "$OUT/seed_panels.new.tsv"; then
    echo "Seed panels changed. Use a new SWEEP_ID." >&2
    exit 1
fi
mv "$OUT/seed_panels.new.tsv" "$OUT/seed_panels.tsv"
mapfile -t ENV_SPECS < "$OUT/seed_panels.tsv"

# Refuse to mix earlier code/config/settings into a resumed sweep.
{
    sha256sum "$0" "$BALROG_DIR/balrog/config/config.yaml"
    git -C "$BALROG_DIR" rev-parse HEAD 2>/dev/null || true
    printf 'workers=%s\nmodel_generation_defaults=true\n' "$WORKERS"
} > "$OUT/settings.new.txt"
if [[ -f "$OUT/settings.txt" ]] && ! cmp -s "$OUT/settings.txt" "$OUT/settings.new.txt"; then
    echo "Code or settings changed. Use a new SWEEP_ID." >&2
    exit 1
fi
mv "$OUT/settings.new.txt" "$OUT/settings.txt"
if [[ ! -f "$OUT/system_info.txt" ]]; then
    {
        date -Is
        conda run -n "$VLLM_ENV" python -c 'import vllm; print("vLLM", vllm.__version__)'
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    } > "$OUT/system_info.txt" 2>&1
fi
[[ -f "$OUT/status.csv" ]] || printf 'model,agent,run,environment,seed,status,elapsed_seconds\n' > "$OUT/status.csv"
echo "Sweep: $SWEEP_ID; three repetitions per model/agent; output: $OUT"
cat "$OUT/seed_panels.tsv"

for record in "${MODELS[@]}"; do
    IFS='|' read -r label model max_len <<< "$record"
    already_done=1
    for agent in "${AGENTS[@]}"; do
        for run in 1 2 3; do
            [[ -f "$OUT/$label/$agent/run_$run/.complete" ]] || already_done=0
        done
    done
    (( already_done )) && continue
    mkdir -p "$OUT/$label"
    python3 - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    try:
        sock.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError:
        sys.exit('Port occupied: stop the existing server or choose another PORT.')
PY
    extra_args=()
    # CoT is provided by BALROG's agent prompt, not Qwen's native thinking mode.
    if [[ "$label" == qwen14-awq ]]; then
        extra_args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
    fi
    echo "Starting $model"
    setsid env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
        conda run --no-capture-output -n "$VLLM_ENV" vllm serve "$model" \
        --host 127.0.0.1 --port "$PORT" --served-model-name "$model" \
        --gpu-memory-utilization 0.90 --max-model-len "$max_len" \
        --generation-config auto --seed 0 "${extra_args[@]}" \
        > "$OUT/$label/vllm.log" 2>&1 &
    VLLM_PID=$!
    if ! wait_for_server "$model"; then
        tail -40 "$OUT/$label/vllm.log" >&2
        exit 1
    fi

    for agent in "${AGENTS[@]}"; do
        tokens=128
        [[ "$agent" == cot ]] && tokens=512
        for run in 1 2 3; do
            run_dir="$OUT/$label/$agent/run_$run"
            [[ -f "$run_dir/.complete" ]] && continue
            for spec in "${ENV_SPECS[@]}"; do
                IFS=$'\t' read -r environment seed_count task_count <<< "$spec"
                for ((seed=1; seed<=seed_count; seed++)); do
                    result_dir="$run_dir/$environment/seed_$seed"
                    [[ -f "$result_dir/.complete" ]] && continue
                    # Preserve incomplete attempts outside the aggregated run.
                    if [[ -d "$result_dir" ]]; then
                        backup="$OUT/incomplete/$label/$agent/run_$run/$environment"
                        mkdir -p "$backup"
                        mv "$result_dir" "$backup/seed_${seed}_$(date +%s%N)"
                    fi
                    mkdir -p "$result_dir"
                    seed_args=("envs.env_kwargs.seed=$seed")
                    if [[ "$environment" == crafter ]]; then
                        # Crafter initializes its RNG in its constructor, before reset.
                        seed_args+=("envs.crafter_kwargs.seed=$seed")
                    fi
                    echo "$label / $agent / repetition $run / $environment / seed $seed"
                    start=$SECONDS
                    if (
                        cd "$BALROG_DIR"
                        conda run --no-capture-output -n "$BALROG_ENV" python eval.py \
                            agent.type="$agent" agent.max_text_history=16 \
                            agent.max_image_history=0 agent.remember_cot=True agent.max_cot_history=1 \
                            envs.names="$environment" "${seed_args[@]}" \
                            "eval.num_episodes.$environment=1" eval.num_workers="$WORKERS" \
                            eval.max_steps_per_episode=null eval.save_trajectories=True \
                            eval.resume_from="$result_dir" \
                            client.client_name=vllm client.model_id="$model" \
                            client.base_url="http://127.0.0.1:$PORT/v1" \
                            client.generate_kwargs.temperature=null \
                            client.generate_kwargs.max_tokens="$tokens" \
                            client.generate_kwargs.thinking_budget=null
                    ) > "$result_dir/console.log" 2>&1 && \
                        python3 - "$result_dir" "$task_count" "$seed" <<'PY'
import collections, json, sys
from pathlib import Path
folder, expected, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
episodes = [json.loads(p.read_text()) for p in folder.rglob('*_run_*.json')]
tasks = collections.Counter(d['task'] for d in episodes)
if len(episodes) != expected or len(tasks) != expected or any(v != 1 for v in tasks.values()):
    sys.exit(f'Incomplete panel: {len(episodes)}/{expected} episodes; task counts: {dict(tasks)}')
if any(d.get('seed') != seed for d in episodes):
    sys.exit('Unexpected recorded environment seed')
PY
                    then
                        touch "$result_dir/.complete"
                        printf '%s,%s,%s,%s,%s,ok,%s\n' "$label" "$agent" "$run" "$environment" "$seed" "$((SECONDS-start))" >> "$OUT/status.csv"
                    else
                        printf '%s,%s,%s,%s,%s,failed,%s\n' "$label" "$agent" "$run" "$environment" "$seed" "$((SECONDS-start))" >> "$OUT/status.csv"
                        tail -25 "$result_dir/console.log" >&2
                        echo "Failed or incomplete evaluation. Resume with SWEEP_ID=$SWEEP_ID." >&2
                        exit 1
                    fi
                done
            done
            # BALROG traverses nested seed folders and computes equal-weight environment progress.
            (
                cd "$BALROG_DIR"
                conda run --no-capture-output -n "$BALROG_ENV" python - "$run_dir" <<'PY'
import sys
from balrog.utils import collect_and_summarize_results, print_summary_table
print_summary_table(collect_and_summarize_results(sys.argv[1]))
PY
            )
            touch "$run_dir/.complete"
        done
    done
    stop_server
done

# Sample SD across the three repeated full panels; each environment has equal overall weight.
python3 - "$OUT" <<'PY'
import csv, json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / 'performance_matrix.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['model', 'agent', 'benchmark', 'runs', 'episodes_per_run', 'mean_progress_pct', 'sd_progress_pct'])
    for model in ('qwen4-instruct', 'qwen14-awq', 'phi4-instruct'):
        for agent in ('naive', 'cot'):
            summaries = [json.loads((root / model / agent / f'run_{r}' / 'summary.json').read_text()) for r in (1, 2, 3)]
            for env in ('babyai', 'crafter', 'babaisai', 'overall'):
                scores = [s['average_progress'] if env == 'overall' else s['environments'][env]['progression_percentage'] for s in summaries]
                counts = [sum(e['episodes_played'] for e in s['environments'].values()) if env == 'overall' else s['environments'][env]['episodes_played'] for s in summaries]
                writer.writerow([model, agent, env, 3, counts[0], statistics.mean(scores), statistics.stdev(scores)])
print(f"Done: {root / 'performance_matrix.csv'}")
PY
