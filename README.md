# NumPy Language Model Profiling (Phase 1.2)

Baseline profiling of a pure-NumPy transformer language model, as groundwork for an FPGA-based hardware accelerator.
Measures the same model three ways: Python/NumPy, a C port calling OpenBLAS, and (planned) Intel VTune.
Team: Abrar, Geetha, Richa. Mentor: Sriram. MGIT FPGA / accelerator final project.

## Status
| Stage | What | State |
|---|---|---|
| 0 | Setup, machine record, smoke test | done |
| 1 | Python baseline (small/medium/large, 1 and 2 BLAS threads) | done |
| 2 | cProfile, snakeviz, line_profiler, py-spy, memory, FLOPs | done |
| 3 | C + OpenBLAS port, validation, benchmarks, flag/thread studies | done |
| 4 | Intel VTune | not started |
| 5 | Fallback tools (perf, gprof, callgrind, uProf) | not started |
| 6 | Consolidated comparison and kernel ranking | not started |
| 7 | Checkpoint | not started |

## Headline results (1 BLAS thread, i5-3210M)
| Config | Python step | C step | Speedup |
|---|---|---|---|
| small | 109.8 ms | 60.2 ms | 1.8x |
| medium | 2294 ms | 1452 ms | 1.6x |
| large | 9463 ms | 6574 ms | 1.4x |

Forward ~41% / backward ~55% / Adam 1-4% of a training step. Machine ceilings: 21.5 GFLOP/s (1-thread dgemm), 10.7 GB/s (STREAM triad). The C port matches NumPy to ~1e-14.

## Layout
- `src/`: unmodified `transformer.py` and `train.py` from the mentor. `data/sample.txt`: training text.
- `baseline/`: benchmark and profiling scripts; `baseline/results/`: all measurements (`prev_unverified_power/` holds an early superseded set, do not use).
- `c/`: C port (`tx.c`), GEMM and STREAM benchmarks, Makefile. Reference data in `c/ref*/` is regenerated, not committed.
- `docs/`: `machine.md`, `sw_baseline.md`, `kernel_ranking.md`, `c_port.md`, screenshots, and `logs/`.

## Reproduce
Run from the repo root.

    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/train.py --config small --steps 50 --log_every 25      # smoke test
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config small --reps 30 --warmup 3 --out baseline/results/py_small.json
    python baseline/export_ref.py --config small --steps 20           # writes c/ref
    cd c && make && ./tx check ref && ./tx train ref && ./tx bench ref 50

Needs gcc, make and OpenBLAS headers/library for the C part. See `docs/c_port.md` for the toolchain caveat used for the recorded numbers (OpenBLAS 0.3.15 vs NumPy's 0.3.29).

## Caveats
Measured inside a Linux guest on a 2012 laptop CPU; timings drifted ~12% during the session. VTune hardware counters are probably unavailable in that guest. See `docs/machine.md`.
