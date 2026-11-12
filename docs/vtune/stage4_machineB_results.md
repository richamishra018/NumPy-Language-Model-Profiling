# Stage 4 results - Machine B (Samsung Galaxy Book4), single BLAS thread

All tables are **Machine B (Samsung Galaxy Book4)**. Machine A numbers appear only where marked "Machine A" for comparison.
Valid data set: OpenBLAS forced to 1 thread (`OMP_NUM_THREADS=1`; MSYS2's OpenBLAS is an OpenMP build and ignores
`OPENBLAS_NUM_THREADS`). tx.exe printed `threads: 1`. A first attempt ran with 12 threads by mistake; it was discarded and only its
C benchmark log is kept (`docs/logs/machineB_c_bench_12thread.txt`), used below for the thread comparison.

## Machine B record
| Item | Value |
|---|---|
| CPU / cores / threads | Intel Core 5 120U (Raptor Lake-U; VTune: Raptorlake-P), 2 P + 8 E = 10 cores / 12 threads |
| RAM | 15.7 GiB |
| OS, Python, NumPy, BLAS | Windows 11 Pro 10.0.26300, Python 3.14.7, NumPy 2.5.4 (scipy_openblas64), C: MSYS2 gcc 16.2.0 + OpenBLAS 0.3.34 (Haswell kernels, OpenMP) |
| VTune version | 2026.6.0 (build 633341); hardware event-based sampling works |
| Power mode, AC | "Samsung Mode" plan (Best performance requested), on AC |
| dgemm ceiling GFLOP/s | 1 thread: peak 73.4 (N=512), 67.4 at N=1024; default 12 threads: 209.6 (N=1024) |
| STREAM triad GB/s (best of 3) | 26.93 (26.88 / 26.63 / 26.93); VTune platform max DRAM 55 GB/s |

Validation: all three configs PASS (max error about 1e-14).

## Baselines (Machine B (Samsung Galaxy Book4)), 1 BLAS thread
| Config | Python step ms | C step ms | Speedup | C GFLOP/s | % of 1-thread dgemm peak |
|---|---|---|---|---|---|
| small | 49.1 (runs 48.9 / 49.2 / 49.1, spread 0.7%) | 18.5 | 2.7x | 20.0 | 27% |
| medium | 899 | 376 | 2.4x | 30.5 | 42% |
| large | 3722 | 1758 | 2.1x | 38.7 | 53% |

(FLOPs per step 369.4 / 11,460.7 / 68,118.4 MFLOP.) C phase split, small: forward 8.2 ms, backward 9.6, adam 0.51, loss 0.14.
Machine A for comparison (1 thread): C GFLOP/s 6.1 / 7.9 / 10.4 = 28% / 37% / 48% of its 21.5 ceiling, the same shape.

## Run 2 (C hotspots, hardware sampling, small): own vs dgemm
| Component | CPU time % (run 1 / run 2) |
|---|---|
| dgemm family (libopenblas: kernel 29-32, copy/beta helpers about 9) | 41.5 / 40.9 |
| libm `tanh` | 18.1 / 19.9 |
| libm `exp` + one unsymbolised ucrtbase helper | 9.9 + 4.3 / 9.5 + 4.2 |
| gelu_bwd | 6.5 / 5.9 |
| gelu_fwd | 2.0 / 1.7 |
| ln_fwd + ln_bwd | 2.1 / 2.0 |
| adam_step | 3.2 / 3.3 |
| forward / backward own loops | 2.5 + 0.8 / 2.9 + 1.0 |
| own code total (tx.exe) | 19.4 / 19.2 |
| wait/spin | 1.5 / 1.5 |

Top five agree in both runs (dgemm_kernel, tanh, exp, gelu_bwd, ucrtbase helper). Machine A exact dgemm share (small) was 43.5%.
tanh+exp (libm, called from GELU and softmax) are 28-29% of the run: the GELU pair including its tanh is about 26-28%.

## Run 3 (Microarchitecture Exploration, small), P-core, whole program and per function
| Item | Retiring | Bad Speculation | Front-End | Back-End: Memory | Back-End: Core | CPI |
|---|---|---|---|---|---|---|
| whole program, run 1 | 43.5% | 8.7% | 10.2% | 16.9% | 20.7% | 0.392 |
| whole program, run 2 | 37.9% | 15.9% | 10.1% | 16.2% | 19.9% | 0.394 |
| dgemm_kernel (runs 1 / 2) | 65 / 58% | 0 / 5% | 2 / 3% | 1 / 4% | 35 / 30% | 0.26 / 0.28 |
| tanh | 38 / 31% | 22 / 36% | 28 / 26% | 0 | 11 / 8% | 0.54 / 0.56 |
| gelu_bwd | 34 / 34% | 36 / 41% | 11 / 12% | 6 / 0% | 13 / 13% | 0.49 / 0.46 |
| adam_step (few samples, noisy) | 18 / 24% | 6 / 0% | 3 / 4% | 18 / 56% | 55 / 44% | 0.80 / 0.69 |

Memory-bound parts (whole program, % of clockticks, run 1 / run 2): L1 5.4 / 5.1, L2 0.1 / 0.7, L3 1.8 / 1.4, DRAM 4.8 / 4.4.
Highlighted issues: CPI is low (0.39, good); Bad Speculation 9-16% is the only bucket that moves between runs.
Some samples land on E-cores (the process is not pinned; E-core Retiring 26-42%), which explains part of the run-to-run spread.
Buckets above are P-core values.

## Run 4 (Memory Access, medium)
DRAM bandwidth observed peak 14.0 GB/s, average 5.3 GB/s, 0.0% of elapsed time at high bandwidth utilisation.
Peak is 52% of the STREAM ceiling (26.9 GB/s) and 25% of platform max (55 GB/s). Memory Bound 13.3% of slots (L1 4.5, L3 1.6, DRAM 6.3% of clockticks).
**Repeat (c_memory_2):** peak 16.1 GB/s (60% of STREAM), average 5.9 GB/s, 0.0% of time at high bandwidth, Memory Bound 14.8% of slots (L1 3.2, L3 2.0, DRAM 7.6% of clockticks). Same conclusion: not bandwidth limited.

## Run 5 (Threading, medium, OMP_NUM_THREADS=4)
Effective CPU utilization 5.3% (0.64 of 12 logical CPUs); spin and overhead 11.9 s = 58.4% of CPU time, mostly
WaitForMultipleObjects / semaphores. **Repeat (c_threading_2):** effective CPU utilization 5.4% (0.65 of 12), spin and overhead 9.97 s = 58.6% of CPU time. Same as run 1.
Observed wall speedup from threads (12-thread run vs 1-thread run, same machine):
small 19.6 vs 18.5 ms (0.94x), medium 388 vs 376 ms (0.97x), large 1294 vs 1758 ms (1.36x). Threads do not help at these sizes.

## Run 1 (Python, small, VTune user-mode, two runs)
By module: NumPy `_multiarray_umath` (elementwise loops) 46-50%, ucrtbase (libm exp/tanh, malloc/free) 27-29%, OpenBLAS 12-15%,
python314.dll interpreter 3.5-6.7%. Symbols for NumPy are unresolved (`func@0x...`), so function-level attribution is by module.
ITT Top Tasks (100 steps): backward 2.79 s (55%), forward 2.08 s (41%), adam 0.136 s (2.7%), loss 0.055 s (1.1%).
Machine A step shares (forward 41%, backward 55%, Adam 1-4%, loss <1%) agree.

## Repeatability
Memory Access and Threading were each repeated once more (runs 4 and 5 now have two results each); the repeats agree within 2 GB/s on DRAM peak and 0.1 point on CPU utilization.
C hotspots: top five identical in both runs. Microarchitecture: whole-program Retiring 43.5 vs 37.9, Bad Speculation 8.7 vs 15.9
(differ beyond a few points; unpinned hybrid cores); dgemm_kernel per-function values agree within about 7 points.
Python hotspots: same top modules in both runs, function order differs.

## Interpretation (guide step 4.13)
| Observation | Meaning | Accelerator implication |
|---|---|---|
| dgemm 41% (small) but only 19% own code and 28% libm tanh/exp | matrix multiply is the largest single block, not the majority; elementwise math is about half as much again | GEMM engine plus a fused GELU/softmax datapath (tanh, exp) |
| dgemm_kernel Retiring 58-65%, Core Bound 30-35%, Memory under 5% | compute limited | more multipliers than a CPU core; count DSP slices |
| tanh/gelu_bwd Retiring about 35%, Bad Speculation 22-41%, Front-End about 25% | libm branches and latency, not memory | hardware tanh/exp pipelines with no branches |
| Bandwidth peak 52% of STREAM, 0% of time at high bandwidth | not bandwidth limited | host-FPGA bandwidth is not the first constraint at these sizes |
| Threads: 0.94x / 0.97x / 1.36x, spin 58% | small per-head dgemm calls do not scale | baseline is effectively single threaded for small and medium models |
| Python: umath 46-50%, BLAS 12-15%, interpreter 4-7% | mostly compiled frames | interpreter overhead is small; do not count it as an accelerator gain |

### (a) Machine A prediction versus Machine B hardware data
- small: prediction "GELU pair as costly as dgemm or more" is **not confirmed**: GELU incl. tanh is 26-28%, dgemm 41%. This matches the Machine A finding (43.5%), now confirmed with hardware sampling.
- medium (40-50%) and large (50% or more): **not measured by VTune in this run** (hotspots were only run on small). Machine A's preload timer (57.9% / 65.7%) remains the only evidence; a medium hotspots run would close this.
- Expected pipeline picture: dgemm Retiring/Core Bound **confirmed**. Elementwise loops "Memory or latency bound": **partly wrong**: they are libm compute/branch/front-end bound, with Memory Bound near zero for tanh.
### (b) Compute-bound versus memory-bound verdict per kernel
dgemm kernel: compute bound (Retiring plus Core Bound about 95%). tanh/exp and gelu_bwd: latency and branch bound inside libm, not memory bound. adam_step: ambiguous (few samples; mix of Core and Memory). Whole run: not bandwidth limited.
### (c) For Phase 1.3 roofline
Model the dgemm kernel family first (largest and cleanly compute bound), then GELU (tanh) as the second point. Use this machine's ceilings:
1-thread dgemm 73.4 GFLOP/s (67.4 at N=1024), 12-thread 209.6 GFLOP/s, STREAM 26.9 GB/s, platform 55 GB/s. Achieved C throughput 20.0 / 30.5 / 38.7 GFLOP/s.

## Caveats
Hotspots/uarch ran small only; medium only for Memory Access and Threading; large not profiled. Python symbols unresolved. Process was not pinned
on a hybrid CPU. No source-line view was exported (function level used; `gelu_bwd` source file `tx.c` is resolved).
