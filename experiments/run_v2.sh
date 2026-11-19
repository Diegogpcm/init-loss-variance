#!/usr/bin/env bash
# v2 experiments: full architecture grid, K=128 inits, 0-1 loss proxies, random-label controls.
# Usage: bash experiments/run_v2.sh {digits|fashion}   (from the repo root)
set -euo pipefail
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/bin/python}
mkdir -p results/v2 logs
case "${1:-digits}" in
  digits)
    $PY experiments/run_sweep.py --n-archs 0 --K 128 --workers 32 \
        --out results/v2/grid_full.jsonl > logs/grid_full.log 2>&1 &
    $PY experiments/run_sweep.py --n-archs 0 --K 128 --n-train 200 --epochs 150 --probe 200 --workers 24 \
        --out results/v2/grid_n200.jsonl > logs/grid_n200.log 2>&1 &
    $PY experiments/run_gain_sweep.py --K 128 --workers 12 \
        --out results/v2/gain_sweep.jsonl > logs/gain_sweep.log 2>&1 &
    wait ;;
  fashion)
    $PY experiments/run_sweep.py --dataset fashion --n-archs 0 --K 128 --epochs 20 --workers 40 \
        --out results/v2/fashion_full.jsonl > logs/fashion_full.log 2>&1 &
    $PY experiments/run_sweep.py --dataset fashion --n-archs 0 --K 128 --n-train 1000 --epochs 100 --workers 24 \
        --out results/v2/fashion_n1000.jsonl > logs/fashion_n1000.log 2>&1 &
    wait ;;
esac
echo "done $1"
