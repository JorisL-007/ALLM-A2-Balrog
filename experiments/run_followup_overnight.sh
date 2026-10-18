#!/usr/bin/env bash
# Start/check Qwen3-4B, then run the frozen follow-up panels for at most eight hours.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8088}"
MODEL="Qwen/Qwen3-4B-Instruct-2507-FP8"
OUT="$ROOT/experiment_results/followup_v1"
mkdir -p "$OUT/server"

if ! curl --silent --fail "http://127.0.0.1:$PORT/v1/models" > "$OUT/server/models.json"; then
  HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
    nohup conda run --no-capture-output -n vllm vllm serve "$MODEL" \
      --host 127.0.0.1 --port "$PORT" --served-model-name "$MODEL" \
      --gpu-memory-utilization 0.88 --max-model-len 8192 \
      --generation-config auto --seed 0 \
      > "$OUT/server/qwen4.log" 2>&1 &
  echo "$!" > "$OUT/server/qwen4.pid"
  for attempt in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:$PORT/v1/models" > "$OUT/server/models.json"; then
      break
    fi
    sleep 2
  done
fi

if ! curl --silent --fail "http://127.0.0.1:$PORT/v1/models" > "$OUT/server/models.json"; then
  echo "Qwen3-4B server did not become ready; see $OUT/server/qwen4.log" >&2
  exit 1
fi

if ! python3 - "$OUT/server/models.json" "$MODEL" <<'PY'
import json
import sys
models = json.load(open(sys.argv[1]))
sys.exit(0 if any(item.get("id") == sys.argv[2] for item in models.get("data", [])) else 1)
PY
then
  echo "Port $PORT serves a different model; stop it before running this study." >&2
  exit 1
fi

python3 "$ROOT/experiments/followup_runner.py" "$@" --port "$PORT"

# Refresh summaries even when the eight-hour limit leaves later panels for another night.
ANALYSIS_OUT="$OUT"
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  if [[ "${arguments[$i]}" == "--out" ]]; then
    ANALYSIS_OUT="${arguments[$((i+1))]}"
  fi
done
python3 "$ROOT/experiments/analyze_followup.py" --out "$ANALYSIS_OUT"
