# C + OpenBLAS port (Stage 3)

## What was built
`c/tx.c` (unmodified from the guide) implements forward, cross-entropy loss, full backward and Adam in float64, calling `cblas_dgemm` for every matrix product, and checks itself against reference data exported from the unmodified NumPy code (`baseline/export_ref.py` -> `c/ref`, `c/ref_medium`, `c/ref_large`; the reference data is git-ignored and regenerable).

## Toolchain deviation (important)
`libopenblas-dev` could not be installed (no network on this machine's shell). Instead:
- gcc 11.4.0 and make were already present.
- OpenBLAS **0.3.15** (DYNAMIC_ARCH, core auto-detected as Sandybridge) was taken from the copy bundled inside the installed `opencv_python_headless` package, together with its `libgfortran`/`libquadmath`, and a minimal hand-written `cblas.h` declaring only `cblas_dgemm` and the `openblas_*` helpers. They live outside the repo in `$HOME/openblas_shim`. Build with:
  `make EXTRA_CFLAGS="-I$S/include -L$S/lib -Wl,-rpath-link,$S/lib"` and `LD_LIBRARY_PATH=$S/lib` at run time.
- NumPy uses a different build (OpenBLAS 0.3.29). The two BLAS versions differ, so a small part of any Python-vs-C gap may come from the BLAS build. When a proper `libopenblas-dev` (or MSYS2 OpenBLAS on Windows) is available, rebuild and rerun the benchmarks to remove this caveat.
- `hello_dgemm` printed `C = [19 22; 43 50]` as expected.

## Machine ceilings (step 3.3 / 3.4; also in machine.md)
- Practical compute ceiling: **21.5 GFLOP/s with 1 thread** (best of N=64..2048, N=256), **33.2 GFLOP/s with 2 threads** (default). Logs: `docs/logs/gemm_bench.txt`.
- Peak memory bandwidth (STREAM triad, best of 3): **10.7 GB/s**. Log: `docs/logs/stream_triad.txt`.
- Theoretical single-thread peak: AVX without FMA = 8 FLOP/cycle x 2.5 GHz = 20 GFLOP/s; the measured 21.5 exceeds this, which implies the CPU turbo clock (up to 3.1 GHz for this chip -> about 25 GFLOP/s) was active. Use the measured number for the roofline.

## Validation (step 3.7): all PASS
Log: `docs/logs/c_check_train.txt`.

| Config | max abs logits diff | max abs grad diff | worst loss diff over steps | Result |
|---|---|---|---|---|
| small (20 steps) | 1.02e-14 | 1.55e-15 | 6.2e-15 | PASS |
| medium (20 steps) | 1.38e-14 | 1.61e-15 | 9.8e-15 | PASS |
| large (6 steps*) | 1.33e-14 | 1.39e-15 | 4.9e-15 | PASS |

*Large reference exported with 6 training steps instead of 20 to fit the time limit of one shell call (NumPy needs ~10 s per step on this machine). The build is warning free with `-Wall -Wextra`.

## Python versus C (1 BLAS thread, step 3.8)
Logs: `docs/logs/c_bench_1thread.txt`; Python from `docs/sw_baseline.md`.

| Config | Python step ms | C step ms | Speedup | Python GFLOP/s | C GFLOP/s |
|---|---|---|---|---|---|
| small | 109.8 | 60.2 | 1.83x | 3.36 | 6.14 |
| medium | 2294.4 | 1452.2 | 1.58x | 5.00 | 7.89 |
| large | 9463.0 | 6574.0 | 1.44x | 7.20 | 10.36 |

C phase medians (ms, forward / loss / backward / Adam): small 23.9 / 0.46 / 33.1 / 2.38; medium 594 / 2.26 / 828 / 22.4; large 2605 / 2.29 / 3781 / 134.

Readings: the speedup shrinks with size (1.8x, 1.6x, 1.4x) because larger matrices put more time into BLAS, which both versions share. Even C reaches only 6-10 GFLOP/s, 29-48% of the 21.5 GFLOP/s dgemm ceiling, so most of the remaining time is elementwise work, many small per-head products and memory traffic (the gap Stage 4 should explain).
**Noise caveat:** the machine got about 12% slower during the session (Python small step 110 ms in Stage 1, 123 ms when re-measured next to the later C runs; C small 60 ms then 69 ms). Back-to-back pairs give small speedup of 123/69 = 1.78x, consistent with the table's 1.82x, so the ratios hold but absolute times drift. Check power state/thermals before the final numbers in Stage 6.

## Compiler flag study (step 3.9, small, 15 reps, two runs each, train-step median ms)
| Flags | Run 1 | Run 2 |
|---|---|---|
| -O0 | 73.3 | 74.1 |
| -O2 | 69.9 | 70.4 |
| -O3 | 68.0 | 68.0 |
| -O3 -march=native | 68.0 | 68.0 |

-O0 is only about 8% slower than -O3, because nearly all time is inside prebuilt OpenBLAS (and libm tanh/exp) which the flags do not touch. -O3 vs -O3 -march=native is not distinguishable (identical to measurement spread), and -O2 vs -O3 differs by about 2 ms, near the noise (max-min spread is several ms). Note these runs happened at the slower machine state (~68 ms vs 60 ms earlier), so compare flags only with each other.

## Thread study (step 3.10, ms, train step median)
| Threads | C small | C medium | Python medium | Speedup vs 1 thread (C medium) |
|---|---|---|---|---|
| 1 | 68.3 | 1373.8 | 2294.4 | 1.0x |
| 2 | 73.0 | 1238.5 | 2260.8 (default) | 1.11x |
| 4 | 67.9 | 1271.2 | n/a (capped to 2) | 1.08x |
| all (2 logical) | = 2 threads | = 2 threads | = 2 threads | |

Setting 4 threads made OpenBLAS report `threads: 2` (capped at the 2 logical CPUs), so "4" and "all" equal the 2-thread run. Scaling is weak: the 2 logical CPUs are hyper-threads of one physical core, small is flat to slightly worse with 2 threads, and the elementwise loops in tx.c are single threaded. (Python 4-thread run skipped because it equals the 2-thread default.) dgemm alone scales 21.5 -> 33 GFLOP/s with 2 threads, but the whole step does not.

## Prediction before running VTune (to be checked in Stage 4)
Arithmetic: if every matrix FLOP ran at the 21.5 GFLOP/s ceiling, dgemm would need about 17 ms of the 60 ms small step (29%), 0.53 s of 1.45 s for medium (37%) and 3.2 s of 6.6 s for large (48%). The rest should be elementwise code. My prediction:
- **small:** the hottest single function will be the GELU pair (`gelu_fwd`/`gelu_bwd`, dominated by libm `tanh` and cube computation), roughly as costly as or more than `dgemm_kernel_SANDYBRIDGE`; layer-norm backward and softmax follow.
- **medium:** `dgemm_kernel_SANDYBRIDGE` becomes the top symbol (about 40-50% of samples) with GELU next.
- **large:** `dgemm_kernel` clearly hottest (about 50% or more), with attention-related products (head-sized dgemm calls at ld = D) and softmax growing.
- VTune should show the dgemm kernel as core bound/retiring-heavy, and the elementwise loops (GELU, layer norm, Adam) as memory/latency bound.

## tx.c <-> transformer.py map
See the table in the guide (step 3.11); I did not modify tx.c.
