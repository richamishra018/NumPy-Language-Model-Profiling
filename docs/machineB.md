# Machine B (Samsung Galaxy Book4)

All Stage 4 (Intel VTune) measurements come from this machine. Machine A numbers (`docs/machine.md`) are a different computer and must not be mixed with these.

## Machine record
| Item | Value |
|---|---|
| CPU | Intel Core 5 120U (Raptor Lake-U; VTune reports "Raptorlake-P"), 2 P-cores + 8 E-cores = 10 cores / 12 threads, 1.4 GHz base |
| RAM | 15.7 GiB (16,847,933,440 bytes) |
| OS | Windows 11 Pro 10.0.26300 |
| Python / NumPy | 3.14.7 / 2.5.4 (NumPy uses its bundled scipy-openblas64) |
| C toolchain | MSYS2 UCRT64, gcc 16.2.0, OpenBLAS 0.3.34 (Haswell kernels, OpenMP build) |
| VTune | 2026.6.0 (build 633341) |
| Power | Windows plan "Samsung Mode" (Best performance requested), on AC power |

## Important: single-thread setting
MSYS2's OpenBLAS is an OpenMP build and ignores `OPENBLAS_NUM_THREADS`. A "1 thread" run needs `OMP_NUM_THREADS=1` as well (`tx.exe` prints `threads: 1` when it worked). All numbers below and in `docs/vtune/` are single thread unless marked otherwise.

## Baselines, 1 BLAS thread
| Config | Python step ms | C step ms | Speedup | C GFLOP/s |
|---|---|---|---|---|
| small | 49.1 (runs 48.9 / 49.2 / 49.1) | 18.5 | 2.7x | 20.0 |
| medium | 899 | 376 | 2.4x | 30.5 |
| large | 3722 | 1758 | 2.1x | 38.7 |

FLOPs per step: small 369.4, medium 11,460.7, large 68,118.4 MFLOP. Raw data: `baseline/results/machineB_py_*.json`, `docs/logs/machineB_c_bench.txt`.
For reference, a first run with OpenBLAS accidentally on 12 threads gave C steps of 19.6 / 388 / 1294 ms (`docs/logs/machineB_c_bench_12thread.txt`), i.e. threads do not help at small and medium sizes.

## Ceilings
| Ceiling | Value |
|---|---|
| dgemm, 1 thread | peak 73.4 GFLOP/s (N=512); 67.4 at N=1024 |
| dgemm, 12 threads | 209.6 GFLOP/s (N=1024) |
| STREAM triad (best of 3) | 26.93 GB/s |
| DRAM platform maximum (VTune) | 55 GB/s |

Raw data: `docs/logs/machineB_ceilings.txt`. Validation (all three configs PASS, max error about 1e-14): `docs/logs/machineB_check.txt`.

## Where the VTune results are
`docs/vtune/stage4_machineB_results.md` (tables and interpretation), `docs/vtune/README.md` (exact commands), `docs/vtune/self_check.txt`, screenshots and CSV exports in the same folder.
