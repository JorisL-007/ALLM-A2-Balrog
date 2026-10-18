#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "$ROOT/exercises/common/run_experiments.py" --study model_transfer_v1 --port 8097 "$@"
