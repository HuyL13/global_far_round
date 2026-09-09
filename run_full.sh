#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
OUTPUT="${OUTPUT:-$ROOT/results}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
mkdir -p "$OUTPUT"

python -u scripts/validate_environment.py
python -u scripts/run_rtn.py --config configs/rtn4.yaml --output "$OUTPUT/rtn4" --device "$DEVICE" --dtype "$DTYPE" 2>&1 | tee "$OUTPUT/rtn4.log"
python -u scripts/run_rtn.py --config configs/rtn3.yaml --output "$OUTPUT/rtn3" --device "$DEVICE" --dtype "$DTYPE" 2>&1 | tee "$OUTPUT/rtn3.log"
python -u scripts/run_far.py --config configs/far.yaml --output "$OUTPUT/far" --device "$DEVICE" --dtype "$DTYPE" 2>&1 | tee "$OUTPUT/far.log"
python -u scripts/make_summary.py --results "$OUTPUT" 2>&1 | tee "$OUTPUT/summary.log"
