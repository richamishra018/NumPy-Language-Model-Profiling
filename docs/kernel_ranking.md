# Python kernel ranking (Stage 2)

Machine: [machine.md](machine.md) (AC power, Best performance, 1 BLAS thread: `OPENBLAS_NUM_THREADS=1`). Raw evidence in `baseline/results/` (`cprofile_*`, `lineprofile_*`, `spy_small.*`, `mem_small.png`, `flops_*.txt`).
cProfile runs: small 30 steps, medium 5 steps, large 2 steps (cProfile adds overhead: trust ranking and proportions, not seconds). Large uses only 2 steps, so it is a rougher measurement.

## First table: small config (30 steps)
| Rank | Function | cProfile self % | Line-level finding | py-spy agrees? |
|---|---|---|---|---|
| 1 | gelu_grad | 22.4 | L27 `x + 0.044715*x**3` inside tanh arg: 48% of function; L28 `np.tanh`: 35% | Yes, 21.1% |
| 2 | gelu | 20.1 | single line is 100% (power, tanh, several temporaries) | Yes, 20.4% |
| 3 | mha_backward | 10.5 | self time includes ~a dozen matmuls plus softmax backward | Yes, 9.9% |
| 4 | ff_backward | 9.2 | L203 `dh_act * gelu_grad(h)` = 72% of its cumulative time (the GELU gradient call) | Yes, 9.5% |
| 5 | softmax | 5.6 | L18 `np.exp(x)` = 59% | Yes, 5.8% |
| 6 | ff_forward | 5.4 | L189 `gelu(h)` = 79% of cumulative time | Yes, 5.7% |
| 7 | layer_norm_backward | 5.4 | L99-L101 each 27-31% (many elementwise passes, `** -1.5`, sqrt) | Yes, 5.3% |
| 8 | mha_forward | 5.4 | L130 `softmax(scores)` = 51% of cumulative time | Yes, 5.8% |
| 9 | AdamOptimizer.step | 3.6 | elementwise update | Yes, 4.2% |

Top five by self time: gelu_grad, gelu, mha_backward, ff_backward, softmax. **The three tools agree on the top group** (GELU pair, then mha/ff backward, softmax, layer norm).

## Native view (py-spy --native, small, 100 steps)
Leaf samples: `dgemm_kernel_SANDYBRIDGE` 16.1% (+ `dgemm_beta` 1.2%, so matrix multiply is only ~17% of wall time), versus elementwise work: `expm1`/`tanh` ~12%, `DOUBLE_multiply` 5.6%, `DOUBLE_add` 4.8%, `DOUBLE_divide` 3.0%, `pow` 1.2%, plus temp-array copies. (~9.5% sits under `_call_with_frames_removed`, unresolved Python-side frames; 322 of 4594 samples had errors, so treat the native percentages as +/- a few points.) This agrees with the FLOP analysis below: matmul is the minority of time.

## Scaling study (step 2.7), 1 BLAS thread
| Config | Step ms | Forward % | Backward % | Top-3 by cProfile self time | Achieved GFLOP/s |
|---|---|---|---|---|---|
| small | 109.8 | 41.4 | 53.7 | gelu_grad 22%, gelu 20%, mha_backward 10.5% | 3.36 |
| medium | 2294.4 | 41.3 | 57.0 | gelu_grad 21%, gelu 19%, ff_backward 13% | 5.00 |
| large | 9463.0 | 41.3 | 56.6 | mha_backward 18%, ff_backward 17%, gelu_grad 15% | 7.20 |

GELU pair (gelu + gelu_grad) share: small 42.5%, medium 39.8%, large 26.3%. Softmax grows with size: 5.6% -> 7.2% -> 10.7% (quadratic attention). mha_backward climbs from 10.5% to 18.3%. **The hottest kernel changes with config**: elementwise GELU at small/medium, attention backward at large. The measured single-thread dgemm ceiling on this CPU is 21.5 GFLOP/s (Stage 3), so the Python step achieves 16-33% of it even counting only matmul FLOPs.

## FLOPs (analytic, `flops_*.txt`, N = batch x (seq_len+1))
| Config | Forward MFLOP | Backward MFLOP | Step MFLOP |
|---|---|---|---|
| small | 123.1 | 246.3 | 369.4 |
| medium | 3,820 | 7,640 | 11,461 |
| large | 22,706 | 45,412 | 68,118 |

These count matmul FLOPs only (elementwise work such as GELU, softmax, layer norm is excluded), so achieved GFLOP/s understates total arithmetic but shows how little of the step is efficient matrix math.

## Memory (step 2.5)
`mprof` small, 30 steps: samples ranged 5.5 to 72.7 MiB (peak RSS from bench 83 MB), see `baseline/results/mem_small.png`. Attention probability matrix per layer (float64, T = seq_len+1): small 1.1 MB, medium 8.5 MB, large 33.8 MB. Peak RSS: 83 MB / 711 MB / 2354 MB.

## Hardware reading
The top Python hotspots are elementwise (tanh, power, exp), memory-bound with little arithmetic per byte, and fit a streaming datapath well. But a C port (Stage 3) fuses these passes and removes temporaries, so the Python ranking is not the final target; Stage 6 compares before and after.

## Caveats
- snakeviz (step 2.2): screenshots of the icicle view for all three configs are in `docs/snakeviz_small.png`, `docs/snakeviz_medium.png` and `docs/snakeviz_large.png` (rendered with snakeviz 2.2.2 from the cProfile `.prof` files, captured with a headless browser). Small shows `gelu_grad` 0.745 s and `gelu` 0.670 s of a 3.33 s run, with backward 1.75 s vs forward 1.37 s. A bar chart of the same data is in `docs/cprofile_selftime.png`.
- py-spy raw was 200 steps (6,912 samples); the native/speedscope run was 100 steps with some sampling errors.
- The py-spy and mprof runs executed inside the Linux guest under a hypervisor.


---

# Stage 6: Python vs C kernel ranking, with evidence from several tools

Machine labels: **A** = i5-3210M laptop (no hardware counters), **B** = Samsung Galaxy Book4 (VTune with counters). Never mixed in a column without a label. Tool evidence: cProfile and py-spy (Python, A), the exact dgemm per-call-site timer (C, A: `c/dgemm_sites.c`, `baseline/dgemm_sites_report.py`, `docs/logs/dgemm_by_kernel.txt`), gprof (C, A), VTune hotspots, Microarchitecture Exploration, Memory Access and Threading (C, B: `docs/vtune/`), flops.py (analytic).

## Candidate kernels, small config unless stated
"Share in Python" = cProfile self-time share of the matching functions (small / medium / large). "Share in C" = percent of the whole C run. dgemm per-kernel splits are exact per call site on Machine A (small / medium / large); Machine A's total dgemm share is 43.5% / 57.9% / 65.7%. Machine B's VTune measured the small config only (dgemm 41%), without splitting dgemm by call site.

| Candidate kernel | Share in Python (cProfile) | Share in C, Machine A (exact dgemm, % of run) | Share in C, Machine B (VTune, small) | FLOPs per step, medium (flops.py; % of step) | Bound by (Machine B VTune) | Notes |
|---|---|---|---|---|---|---|
| QKV and output projections (GEMM) | inside mha_forward/backward: 15.9 / 20.2 / 26.9% (shared with attention products, cProfile cannot separate) | 11.2 / 15.6 / 16.4% (13-15 GFLOP/s) | part of dgemm 41% | 3,246 MFLOP (28.3%) | compute: dgemm kernel Retiring 58-65%, Core Bound 30-35%, Memory <5% | regular, large M; GEMM engine target |
| Attention scores, softmax, context (per head) | mha_* (above) + softmax 5.6 / 7.2 / 10.7% | products: 9.3 / 11.7 / 16.2% (only 5-10 GFLOP/s); softmax not separable in C on A | softmax's libm exp 9.5-9.9% of run; own softmax loops are part of "forward own loops" 2.5-2.9% | 1,636 MFLOP (14.3%) products only | products: same dgemm kernel but inefficient (dgemm packing helpers `oncopy/otcopy/itcopy` are 46-100% Memory Bound, about 7% of the run); exp: latency/branch bound in libm | 12,672 / 13,824 / 13,824 small calls per run; grows as T^2; per-head tiles are 129 x 32 | 
| Feed-forward W1 and W2 (GEMM) | inside ff_forward/backward: 14.6 / 21.0 / 26.0% (includes bias add) | 21.3 / 30.0 / 32.9% (14-16 GFLOP/s) | part of dgemm 41% | 6,493 MFLOP (56.7%) | compute (as dgemm) | largest single kernel by FLOPs and by C time |
| GELU forward and backward (elementwise, tanh) | gelu + gelu_grad: 42.5 / 39.8 / 26.3% | not measured exactly on A (gprof own code: gelu_bwd 30.9% + gelu_fwd 5.9% of own-code samples, tanh in libm not seen) | **26.6-27.6%** (tanh 18.1-19.9, gelu_bwd 5.9-6.5, gelu_fwd 1.7-2.0) | about 118 MFLOP (about 1%) hand estimate, tanh counted as one op | latency, branches, front-end in libm tanh: Retiring 31-38%, Bad Speculation 22-36%, Front-End 25-28%, Memory about 0 | tiny FLOP share, large time share: pure instruction-latency cost; a hardware tanh pipeline removes it |
| Layer norm forward and backward | 7.1 / 5.3 / 4.1% | not measured exactly | ln_fwd + ln_bwd 2.0-2.1% | about 52 MFLOP (0.5%) hand estimate | not separately measured (too few samples) | small share, regular streaming pattern |
| Vocabulary projection and loss | loss 0.5 / 0.1 / 0.0% (projection is inside forward/backward) | vocab dgemm 1.7 / 0.6 / 0.3%; loss 0.8% / 0.2% / 0.0% of step | not separated | 86 MFLOP GEMM (0.7%) + about 1 MFLOP loss | n/a | negligible once V = 54; grows if the vocabulary grows |
| Adam update | 3.6 / 1.2 / 1.6% | step share 4.0 / 1.5 / 2.0% (2.38 / 22.4 / 134 ms of C step, Stage 3) | adam_step 3.2-3.3% | about 10 MFLOP (0.1%) hand estimate | ambiguous: Core Bound 44-55%, Memory Bound 18-56% across runs, few samples | touches p, g, m and v per step (about 26 MB of float64 state at medium) so it is memory heavy, but a small share |

Cross-tool agreement: cProfile and py-spy agree on the Python top group (GELU pair, mha, ff, softmax, layer norm); in C, the exact timer on A (dgemm 43.5% small) and VTune on B (dgemm 41%) agree within 2.5 points; both gprof (A) and VTune (B) put gelu_bwd first among own functions. Machine B's VTune uarch CSV reports lower tanh/exp percentages (about 16% and 4%) than its hotspots run (about 18-20% and 10%); the two analyses sample differently on this hybrid CPU (unpinned P/E cores), so treat libm shares as roughly 15-28% for tanh+exp together and not as a precise number.

Python vs C change: in Python the GELU pair is 42% of small-config time; in C the same work is about 27% (VTune, small). The C speedup mostly removes temporaries and interpreter cost, not the tanh itself, so GELU stays a leading non-GEMM cost in C.

## Minimum data movement per GEMM (medium, shapes for Phase 1.3)
| GEMM (medium, forward) | M x N x K | FLOP per call | minimum bytes per call (float64: A+B+C once) | FLOP/byte | vs ridge (A 2.0, B 2.7) |
|---|---|---|---|---|---|
| QKV / output projection | 2064 x 128 x 128 | 67.63 M | 4358 kB | 15.5 | above ridge: compute side |
| Attention scores QK^T (per head) | 129 x 129 x 32 | 1.07 M | 199 kB | 5.3 | above ridge: compute side |
| Attention x V (per head) | 129 x 32 x 129 | 1.07 M | 199 kB | 5.3 | above ridge: compute side |
| FF W1 | 2064 x 512 x 128 | 270.53 M | 11092 kB | 24.4 | above ridge: compute side |
| FF W2 | 2064 x 128 x 512 | 270.53 M | 11092 kB | 24.4 | above ridge: compute side |
| Vocab projection | 2064 x 54 x 128 | 28.53 M | 3060 kB | 9.3 | above ridge: compute side |

All GEMMs sit above the ridge points (Machine A 2.0, Machine B 2.7 FLOP/byte), so the roofline places them on the compute side, in agreement with VTune (dgemm compute bound; Memory Access: DRAM bandwidth peak 14-16 GB/s, 52-60% of Machine B's STREAM 26.9 GB/s, 0% of time at high bandwidth). The per-head attention tiles have the lowest intensity (5.3) and the lowest measured efficiency. Elementwise kernels stream whole tensors with about 1 FLOP per byte or less (GELU reads and writes 16 bytes per element for roughly 10-20 operations), so they sit near or below the ridge; their cost in practice comes from libm instruction latency, not bandwidth.

## Selection criteria applied
- **Share.** GEMM family (all dgemm) is the largest block of the C version: 43.5 / 57.9 / 65.7% (A, exact) and 41% (B, VTune, small). Feed-forward W1/W2 alone is 21-33% (A). GELU with tanh is the largest non-GEMM block (27%, B, small), softmax exp next (about 10%).
- **Parallelism and regularity.** GEMMs are regular multiply-accumulate nests (Retiring 58-65%, Core Bound 30-35% on B); GELU and softmax are elementwise with a transcendental per element, easy to pipeline in a datapath.
- **Data reuse.** Large GEMMs have 15-24 FLOP/byte (high reuse, tiling helps); per-head tiles 5.3; elementwise about 0.5-1: no reuse, so buffering helps only by fusion (one pass, no temporaries).
- **Size.** Attention probability matrices: 1.1 MB (small), 8.5 MB (medium), 33.8 MB (large) per layer; the large one will not fit in block RAM without tiling.
- **Honest ceiling (Amdahl).** If only a kernel's own share speeds up, the best total speedup is 1 / (1 - share):

| Candidate | Share (basis) | Best possible total speedup |
|---|---|---|
| All GEMMs (dgemm family) | 43.5% / 57.9% / 65.7% (A, exact, small / medium / large) | 1.77x / 2.38x / 2.92x |
| All GEMMs | 41% (B, VTune, small) | 1.69x |
| Feed-forward GEMMs only | 21.3% / 30.0% / 32.9% (A) | 1.27x / 1.43x / 1.49x |
| GEMMs + GELU (tanh) + softmax exp, fused | about 41 + 27 + 10 = 78% (B, small) | about 4.5x |

Amdahl ignores host to FPGA transfer cost and the CPU-side work left; real gains are lower.

## Provisional proposal (to be confirmed by the Phase 1.3 roofline)
Accelerate the **GEMM family first** (compute bound, regular, 41-66% of the best software), with a **fused GELU and softmax datapath** as the second block, because that is where the remaining time goes and a pipelined tanh/exp removes the libm latency that dominates it. Three measured pieces of evidence for the GEMM block: (1) dgemm is the largest block at every size (43.5 / 57.9 / 65.7% on A, 41% on B); (2) it is compute bound (Retiring 58-65%, Core Bound 30-35%, Memory under 5% of slots, DRAM bandwidth peak only 52-60% of STREAM and 0% of time at high bandwidth, on B); (3) the best software still reaches only 39-42% of its own peak at medium size (A 8.3 of 21.5, B 30.5 of 73.4 GFLOP/s) and the per-head attention products run at 5-10 GFLOP/s on A, so there is room. Best total speedup from GEMM alone: about 1.7-2.4x at small and medium sizes (up to 2.9x at large); about 4.5x if GELU and softmax exp are also fused (B, small).
What would change this: if the target model is much larger (GEMM share rises toward 65%+ and attention memory dominates), the vocabulary grows (vocab projection stops being negligible), or a medium/large VTune hotspots run on Machine B shows the GELU share staying near 27% instead of falling as Python suggests (42% to 26%).

## Evidence gaps (named owners)
- VTune hotspots, uarch and ITT per-phase times for the **medium and large** C configs, and a dgemm per-call-site split on Machine B: Richa (Samsung Book). Until then the medium/large C shares are Machine A exact timings, and Machine B's per-kernel picture is small-config only.
- Machine B Python default-thread run and C peak memory: Richa (Samsung Book), commands in `stage6_comparison.md`.
- Hand-count FLOPs for the elementwise kernels (rough, tanh as one op): refine in Phase 1.3 (team).
