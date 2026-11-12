# Stage 5: fallback profiling of the C port (no VTune hardware counters)

Why: VTune 2026.6 on this i5-3210M (Ivy Bridge) offers only user-mode sampling; hardware analyses are "not applicable to the current machine microarchitecture" (see `docs/vtune/README.md`). Stage 1-3 numbers came from the Linux guest, so the same environment is used here. All runs: 1 BLAS thread, small/medium configs unless noted, OpenBLAS 0.3.15 (see c_port.md).

## What worked and what did not
| Tool | Result here |
|---|---|
| gprof (`-pg`) | Works. Sees only the instrumented binary, not OpenBLAS or libm. |
| perf | Installed (6.8.12) but unusable: `perf_event_paranoid=4`, no unprivileged capability, and the guest exposes no CPU PMU (only software/tracepoint events). Not changed: that would be modifying system settings. So no cycles, IPC or cache-miss counters. |
| valgrind/callgrind | Not installed and no network, so skipped. |
| Xcode Instruments, AMD uProf | Not applicable (Windows/Intel). |
| `c/dgemm_timer.c` (LD_PRELOAD wrapper, my addition) | Works. Times every `cblas_dgemm` call exactly. Not in the guide; it answers the guide's key question ("how much is dgemm vs everything else") without a sampling profiler. |

## 1. dgemm share of runtime (exact, `docs/logs/dgemm_timer.txt`)
| Config | dgemm calls | dgemm time / wall | **dgemm share** | achieved dgemm GFLOP/s | avg time per call |
|---|---|---|---|---|---|
| small (30 reps) | 13,959 | 1.04 / 2.39 s | **43.5%** | 11.7 | 74 us |
| medium (6 reps) | 14,499 | 8.93 / 15.41 s | **57.9%** | 11.6 | 616 us |
| large (3 reps) | 14,490 | 35.71 / 54.33 s | **65.7%** | 11.4 | 2,465 us |

(The wall time is the whole `tx bench` process, including its harness and start-up, so shares are slightly conservative.)
Key reading: dgemm is the largest single component at every size and grows with model size, but it runs at only about 11.5 GFLOP/s, about half the 21.5 GFLOP/s ceiling measured by `gemm_bench` on big square matrices. The reason is the ~420 dgemm calls per step, many of them small per-head products (qk and attention-times-V per batch and head), which cannot reach the large-matrix ceiling.

## 2. Own-code ranking from gprof (`-O2 -pg -g -fno-inline`, `docs/logs/gprof_noinline.txt`)
Self-time share of the samples that fall inside `tx` itself (this excludes OpenBLAS and libm):

| Rank | Small | % | Medium | % |
|---|---|---|---|---|
| 1 | gelu_bwd | 30.9 | gelu_bwd | 33.3 |
| 2 | adam_step | 17.7 | forward (own loops) | 14.4 |
| 3 | backward (own loops) | 13.2 | adam_step | 10.1 |
| 4 | forward (own loops) | 10.3 | gelu_fwd | 9.3 |
| 5 | colsum_acc / ln_fwd / gelu_fwd | 5.9 each | backward (own loops) | 9.3 |
| 6 | add_bias | 4.4 | ln_bwd | 6.8 |

(With normal inlining, gprof folds gelu into forward/backward: `backward` 46.8%, `adam_step` 19.1%, `forward` 14.9%, `linear` 6.4%, `ln_fwd` 6.4%, `ln_bwd` 4.3%; small, 30 reps.)

## 3. Where the whole run goes (small / medium; approximate)
Combining the exact dgemm share with the gprof-sampled own-code share (gprof total / wall in the same run):

| Component | Small | Medium |
|---|---|---|
| OpenBLAS dgemm | 41.7% | 57.2% |
| Own C code (gprof samples) | 26.6% | 14.1% |
| Everything else (inferred by subtraction: libm `tanh`/`exp`, other shared-library and start-up time) | about 32% | about 29% |

The last row is **inferred, not measured**: gprof cannot see inside libm, so I cannot separate `tanh`/`exp` time from other overhead. Because GELU's `tanh` lives in libm, the GELU functions' true cost is higher than their own-code self time in section 2. gprof perturbs timing (-pg and -fno-inline), so treat section 2 and 3 as rough.

## Check against the prediction in `c_port.md`
- **Medium: dgemm top symbol at about 40-50%:** right on the family, slightly low (measured 57%).
- **Large: dgemm clearly hottest, about 50% or more:** right (66%).
- **Small: GELU pair as costly as or more than the dgemm kernel:** not supported. dgemm is the largest component (43%); the GELU pair's maximum possible share (own code about 10% of the run plus the part of the ~32% "everything else" that is `tanh`) cannot be resolved without a sampling profiler, so "GELU >= dgemm" is neither confirmed nor excluded. It cannot be much above 42%.
- **Where I was wrong:** I estimated dgemm time from the 21.5 GFLOP/s large-matrix ceiling (29% / 37% / 48% for small / medium / large). Real dgemm runs at about 11.5 GFLOP/s because of the many small per-head calls, so its share is 14-20 percentage points higher. Lesson for the accelerator: the per-head attention products and their call overhead matter, not only big GEMMs.
- Counter-style claims (core vs memory bound, cache misses) in c_port.md's prediction remain **unchecked**: there are no counters here.

## What this means for the accelerator (partial, evidence-limited)
- The best software spends 43-66% of its time in matrix multiply and 34-57% elsewhere (elementwise math, softmax, layer norm, Adam, data movement), so both a GEMM engine and a fused streaming elementwise datapath are in scope, with the GEMM share rising with model size.
- The small-matrix dgemm inefficiency (11.5 vs 21.5 GFLOP/s) suggests batching or fusing the per-head attention products.
- Not answered here: whether the kernels are memory or compute bound. That needs hardware counters from a newer Intel machine (VTune Microarchitecture Exploration and Memory Access), or perf on bare-metal Linux.

## Status against the guide's Stage 5 done-when
- Hotspot ranking for the C program from a fallback tool, saved in docs/: **done** (this file and `docs/logs/`).
- Read a teammate's VTune results and explain them in your own words: **still open**: needs a teammate with a supported Intel CPU to run Stage 4 and share `docs/vtune/` results.
