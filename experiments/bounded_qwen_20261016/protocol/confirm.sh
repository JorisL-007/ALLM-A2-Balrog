#!/usr/bin/env bash
# FUTURE ONLY: run after reading CONFIRMATION.md and FROZEN.json.
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:?Usage: bash protocol/confirm.sh NEW_OUTPUT_DIRECTORY}
python_bin=${BALROG_PYTHON:-/home/joris/miniconda3/envs/balrog/bin/python}
vllm_bin=${VLLM_BIN:-/home/joris/miniconda3/envs/vllm/bin/vllm}
model='Qwen/Qwen3-4B-Instruct-2507-FP8'
"$python_bin" - "$task_dir" <<'PY'
import hashlib,json,pathlib,sys
folder=pathlib.Path(sys.argv[1])
frozen=json.loads((folder/'FROZEN.json').read_text())
for path,digest in frozen['frozen_file_hashes'].items():
    file=(folder/path).resolve()
    if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Frozen file changed: '+str(file))
PY
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -e "$output" ]]; then
    echo 'Choose a fresh output directory; preserve any previous confirmation attempt.' >&2
    exit 1
fi
mkdir -p -- "$output"
if ! curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:8088/v1/models > "$output/endpoint.json"; then
    setsid timeout --signal=TERM --kill-after=15s 24h env HF_HUB_OFFLINE=1 \
        VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
        "$vllm_bin" serve "$model" --host 127.0.0.1 --port 8088 \
        --served-model-name "$model" --gpu-memory-utilization 0.90 \
        --max-model-len 16384 --generation-config auto --seed 0 \
        > "$output/vllm.log" 2>&1 &
    server_pid=$!
    printf '%s\n' "$server_pid" > "$output/owned_server_pid.txt"
    ready=0
    for ((attempt=0; attempt<120; attempt++)); do
        if curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:8088/v1/models > "$output/endpoint.json" 2>/dev/null; then
            ready=1
            break
        fi
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 2
    done
    [[ "$ready" == 1 ]] || { echo 'Model server unavailable; inspect vllm.log' >&2; exit 1; }
fi
mapfile -t variants < <("$python_bin" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["confirmation_variants"]))' "$task_dir/FROZEN.json")
repetitions=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["confirmation_repetitions"])' "$task_dir/FROZEN.json")
cp -- "$task_dir/FROZEN.json" "$output/FROZEN.json"
for ((rep=1; rep<=repetitions; rep++)); do
    "$python_bin" "$task_dir/run.py" --mode confirmation --output "$output" \
        --pass-id "confirmation_$rep" --variants "${variants[@]}" --python "$python_bin"
done
"$python_bin" "$task_dir/protocol/analysis.py" "$output" --frozen "$task_dir/FROZEN.json"
