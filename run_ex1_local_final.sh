#!/usr/bin/env bash
# BALROG Exercise 1 - five local models, sequential, in the requested order.
# Project layout: this file next to BALROG/eval.py.
# Defaults: 3 full repetitions/model, 50 BabyAI + 10 Crafter episodes/repetition.
# Completed runs are marked with .complete and skipped when resuming with SWEEP_ID.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BALROG_DIR="$ROOT/BALROG"
VLLM_ENV="${VLLM_ENV:-vllm}"
BALROG_ENV="${BALROG_ENV:-balrog}"
PORT="${PORT:-8088}"
RUNS="${RUNS:-3}"
TEMPERATURE="${TEMPERATURE:-0.7}"
SMOKE="${SMOKE:-0}"
ONLY_MODEL="${ONLY_MODEL:-}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ROOT/exercise1_results/$SWEEP_ID"
MAX_SWEEP_SECONDS="${MAX_SWEEP_SECONDS:-43200}"  # 12-hour ceiling
THINKING_TOKENS="${THINKING_TOKENS:-16384}"
GPT_OSS_CPU_OFFLOAD_GB="${GPT_OSS_CPU_OFFLOAD_GB:-4}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
VLLM_PID=""
SWEEP_START=$(date +%s)

[[ -f "$BALROG_DIR/eval.py" ]] || { echo "Missing $BALROG_DIR/eval.py" >&2; exit 1; }
for bin in conda curl setsid timeout python3; do
    command -v "$bin" >/dev/null || { echo "Missing program: $bin" >&2; exit 1; }
done
mkdir -p "$OUT"

# Full benchmark uses upstream episode defaults; smoke is one episode for each BabyAI task.
ENVS="babyai-crafter-babaisai"
EPISODE_OVERRIDES=()
EXPECTED_EPISODES=60
if [[ "$SMOKE" == "1" ]]; then
    ENVS="babyai"
    RUNS=1
    EPISODE_OVERRIDES=(eval.num_episodes.babyai=1)
    EXPECTED_EPISODES=5
fi

stop_server() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM process group $VLLM_PID ..."
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        for ((i=0; i<20; i++)); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        sleep 3
    fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_is_free() {
    python3 - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError:
        sys.exit(1)
PY
}

wait_for_server() {
    local model="$1" elapsed=0
    while (( elapsed < STARTUP_TIMEOUT )); do
        if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
             python3 -c 'import sys,json; name=sys.argv[1]; data=json.load(sys.stdin); sys.exit(0 if any(m.get("id")==name for m in data.get("data",[])) else 1)' \
             "$model" 2>/dev/null; then
            echo "Server ready: $model"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM exited during startup."
            return 1
        fi
        sleep 5
        ((elapsed+=5))
    done
    echo "Server startup exceeded ${STARTUP_TIMEOUT}s."
    return 1
}

# Preflight: reject models that cannot emit a single valid action. Not run for
# Thinking; it previously spent its output budget reasoning on these synthetic prompts.
# User explicitly wants to try Thinking, and it is the FINAL model.
check_chat() {
    local model="$1" tokens="$2"
    python3 - "$PORT" "$model" "$tokens" "$TEMPERATURE" <<'PY'
import json, sys, urllib.request
port, model, limit, temperature = sys.argv[1:]
valid = {'turn left', 'turn right', 'go forward', 'pick up', 'drop', 'toggle'}
questions = [
    'Task: Pick up the blue box.\nCurrent observation: a wall 4 steps forward, a wall 1 step right, a blue box 3 steps left and 1 step forward.',
    'Task: Go to the purple ball.\nCurrent observation: a wall 6 steps forward, a wall 1 step left, a purple ball 3 steps right and 5 steps forward.',
]
for i, question in enumerate(questions, 1):
    prompt = 'Possible actions: ' + ', '.join(sorted(valid)) + '\n' + question + '\nOutput exactly one of the actions above and no other text.'
    data = json.dumps({
        'model': model, 'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': int(limit), 'temperature': float(temperature)
    }).encode()
    try:
        req = urllib.request.Request('http://127.0.0.1:' + port + '/v1/chat/completions',
                                     data=data, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=240) as reply:
            response = json.load(reply)
        choice = response['choices'][0]
        msg = choice['message']
        content = (msg.get('content') or '').strip().lower()
        finish = choice.get('finish_reason')
        print(f"Preflight {i}: finish={finish}; output_tokens={response.get('usage',{}).get('completion_tokens')}; action={content[:100]!r}", flush=True)
        if finish == 'length' or content not in valid:
            print('Raw final content:', repr((msg.get('content') or '')[:500]), flush=True)
            raise ValueError('Model did not return a valid single action')
    except Exception as exc:
        print(f'Preflight failed on case {i}: {exc}', file=sys.stderr)
        sys.exit(1)
PY
}

# Exact user-requested order: label|model_id|workers|max_context|gpu_memory_utilization
MODELS=(
    'qwen4-instruct|Qwen/Qwen3-4B-Instruct-2507-FP8|8|16384|0.90'
    'qwen14-awq|Qwen/Qwen3-14B-AWQ|2|8192|0.90'
    # 'gpt-oss-20b|openai/gpt-oss-20b|2|8192|0.85'
    'phi4-instruct|microsoft/Phi-4-mini-instruct|4|8192|0.90'
    # 'qwen4-thinking|Qwen/Qwen3-4B-Thinking-2507-FP8|2|32768|0.90'
)

printf 'Sweep: %s\nMode: %s\nRuns/model: %s\nEnvironments: %s\nOutput: %s\nMax wall time: %ss\n' \
    "$SWEEP_ID" "$([[ "$SMOKE" == 1 ]] && echo smoke || echo full)" "$RUNS" "$ENVS" "$OUT" "$MAX_SWEEP_SECONDS"
{
    date -Is
    git -C "$BALROG_DIR" rev-parse HEAD 2>/dev/null || true
    conda run -n "$VLLM_ENV" python -c 'import vllm; print("vLLM",vllm.__version__)' || true
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
} > "$OUT/system_info.txt" 2>&1
if [[ ! -f "$OUT/status.csv" ]]; then
    printf 'model,run,status,elapsed_seconds\n' > "$OUT/status.csv"
fi

for record in "${MODELS[@]}"; do
    IFS='|' read -r label model workers max_len gpu_util <<< "$record"
    [[ -z "$ONLY_MODEL" || "$label" == "$ONLY_MODEL" ]] || continue

    # Resuming should not waste time loading a model whose runs are already finished.
    already_done=1
    for ((run=1; run<=RUNS; run++)); do
        [[ -f "$OUT/$label/run_$run/.complete" ]] || { already_done=0; break; }
    done
    if (( already_done )); then
        echo "Already complete: $label ($RUNS repetitions); skipping server startup."
        continue
    fi
    remaining=$((MAX_SWEEP_SECONDS - ($(date +%s) - SWEEP_START)))
    if (( remaining < 300 )); then
        echo "12-hour budget nearly exhausted; no more models will start."
        break
    fi

    # Per-model generation and history settings. All models use the same
    # temperature, naive agent, BabyAI/Crafter tasks, and default episode counts.
    case "$label" in
        qwen4-instruct)  model_tokens=256; model_history=16; model_run_limit=600  ;;
        qwen14-awq)     model_tokens=256; model_history=16; model_run_limit=1800 ;;
        gpt-oss-20b)   model_tokens=1024; model_history=16; model_run_limit=5400 ;;
        phi4-instruct)  model_tokens=256; model_history=16; model_run_limit=1800 ;;
        qwen4-thinking) model_tokens="$THINKING_TOKENS"; model_history=4; model_run_limit="$MAX_SWEEP_SECONDS" ;;
    esac

    model_dir="$OUT/$label"
    mkdir -p "$model_dir"
    echo
    echo "========== $label ($model) =========="
    echo "Configuration: workers=$workers, history=$model_history, max_tokens=$model_tokens, context=$max_len"
    if ! port_is_free; then
        echo "Port $PORT is occupied. Stop your manually launched vLLM before starting this runner." >&2
        exit 1
    fi
    extra_args=()
    case "$label" in
        qwen14-awq)
            extra_args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
            ;;
        gpt-oss-20b)
            # Proven workaround on this RTX 5080: CPU offload + Triton + eager.
            extra_args+=(--reasoning-parser openai_gptoss
                         --default-chat-template-kwargs '{"reasoning_effort": "low"}'
                         --attention-backend TRITON_ATTN --enforce-eager
                         --cpu-offload-gb "$GPT_OSS_CPU_OFFLOAD_GB")
            ;;
        qwen4-thinking)
            extra_args+=(--reasoning-parser deepseek_r1)
            ;;
    esac

    echo "Starting vLLM; log: $model_dir/vllm.log"
    setsid env VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
        conda run --no-capture-output -n "$VLLM_ENV" \
        vllm serve "$model" \
        --host 127.0.0.1 --port "$PORT" \
        --served-model-name "$model" \
        --gpu-memory-utilization "$gpu_util" \
        --max-model-len "$max_len" \
        --generation-config vllm \
        "${extra_args[@]}" > "$model_dir/vllm.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_server "$model"; then
        tail -60 "$model_dir/vllm.log" || true
        printf '%s,-,server_failed,0\n' "$label" >> "$OUT/status.csv"
        stop_server
        continue
    fi
    if [[ "$label" == "qwen4-thinking" ]]; then
        echo "Skipping synthetic preflight for Thinking (explicit experimental attempt; monitor its actual BALROG log)."
    elif ! check_chat "$model" "$model_tokens"; then
        echo "Preflight failed; skipping this model instead of wasting a full evaluation."
        printf '%s,-,chat_failed,0\n' "$label" >> "$OUT/status.csv"
        stop_server
        continue
    fi

    for ((run=1; run<=RUNS; run++)); do
        result_dir="$model_dir/run_$run"
        mkdir -p "$result_dir"
        if [[ -f "$result_dir/.complete" ]]; then
            echo "Skipping completed run $run for $label"
            continue
        fi
        remaining=$((MAX_SWEEP_SECONDS - ($(date +%s) - SWEEP_START) - 120))
        if (( remaining <= 0 )); then
            echo "Sweep time budget reached; not starting another repetition."
            printf '%s,%s,not_started_budget,0\n' "$label" "$run" >> "$OUT/status.csv"
            break
        fi
        run_limit="$model_run_limit"
        if (( remaining < run_limit )); then run_limit="$remaining"; fi
        echo "Running $label; repetition $run/$RUNS; $workers workers; timeout=${run_limit}s"
        start=$SECONDS
        # resume_from directs BALROG to this exact folder and enables episode-level resume.
        if (
            cd "$BALROG_DIR"
            timeout --signal=TERM --kill-after=30s "${run_limit}s" \
                conda run --no-capture-output -n "$BALROG_ENV" \
                python eval.py \
                agent.type=naive \
                agent.max_image_history=0 \
                agent.max_text_history="$model_history" \
                eval.num_workers="$workers" \
                eval.save_trajectories=True \
                eval.resume_from="$result_dir" \
                "${EPISODE_OVERRIDES[@]}" \
                envs.names="$ENVS" \
                client.client_name=vllm \
                client.model_id="$model" \
                client.base_url="http://127.0.0.1:$PORT/v1" \
                client.timeout=300 \
                client.max_retries=1 \
                client.generate_kwargs.temperature="$TEMPERATURE" \
                client.generate_kwargs.max_tokens="$model_tokens"
        ) > "$result_dir/console.log" 2>&1; then
            completed=$(find "$result_dir" -type f -name '*_run_*.json' | wc -l)
            if (( completed >= EXPECTED_EPISODES )); then
                touch "$result_dir/.complete"
                status=ok
                echo "Completed $completed episodes in $((SECONDS-start)) seconds"
                if grep -RIl --include='*.csv' 'Defaulted to action:' "$result_dir" >/dev/null 2>&1; then
                    echo "WARNING: Invalid-action fallbacks found in trajectory CSVs; inspect these before reporting scores."
                fi
            else
                status=incomplete
                echo "Only $completed/$EXPECTED_EPISODES episode JSON files found."
            fi
        else
            code=$?
            if (( code == 124 || code == 137 )); then
                status=timed_out
                echo "Repetition $run exceeded its time budget."
            else
                status=failed
                echo "Evaluation returned exit status $code."
            fi
            tail -15 "$result_dir/console.log" || true
        fi
        printf '%s,%s,%s,%s\n' "$label" "$run" "$status" "$((SECONDS-start))" >> "$OUT/status.csv"
        if [[ "$status" != "ok" ]]; then
            echo "Skipping remaining repetitions for $label; proceeding to next model."
            break
        fi
    done
    stop_server
 done

echo
echo "Done. Status: $OUT/status.csv"
echo "Results: $OUT/<model>/run_<n>/"
echo "Resume unfinished runs with: SWEEP_ID=$SWEEP_ID ./$(basename "$0")"
echo "Elapsed sweep time: $(( $(date +%s) - SWEEP_START )) seconds"
