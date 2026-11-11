#!/bin/bash
# Stage 1.3 / 1.5: run from repo root. Single-thread = py_<cfg>.json, default threads = py_<cfg>_default.json
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=baseline/results
unset OMP_NUM_THREADS
run(){ # name threads cfg reps warmup
  if [ "$2" = 1 ]; then export OPENBLAS_NUM_THREADS=1; else unset OPENBLAS_NUM_THREADS; fi
  echo "$(date +%T) start $1"; $PY baseline/bench.py --config $3 --reps $4 --warmup $5 --out $R/$1.json > /dev/null; echo "$(date +%T) done $1"
}
[ "$1" = small ] && { for i in 1 2 3; do run py_small_rep$i 1 small 30 3; done; run py_small 1 small 30 3; run py_small_default 0 small 30 3; }
[ "$1" = medium ] && { run py_medium 1 medium 10 2; run py_medium_default 0 medium 10 2; }
[ "$1" = large ] && { run py_large 1 large 5 1; run py_large_default 0 large 5 1; }
