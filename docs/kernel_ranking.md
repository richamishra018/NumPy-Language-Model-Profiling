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
