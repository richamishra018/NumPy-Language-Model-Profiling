# Python / NumPy software baseline (Stage 1)

Measured 2026-11-11 on the machine in [machine.md](machine.md) (Intel i5-3210M, 2 logical CPUs, 3.8 GiB RAM, Linux guest under a Microsoft hypervisor).
**Laptop on AC power, Windows power mode "Best performance".** Python 3.10.12, NumPy 2.2.6, BLAS = bundled OpenBLAS 0.3.29 (scipy-openblas), core Sandybridge.
Unmodified `src/transformer.py`; fixed batch, seed 42; tokens per step = batch_size x (seq_len + 1).

## Thread setting
Two runs per config: **1 BLAS thread** (`OPENBLAS_NUM_THREADS=1`, the fair single-core reference; files `py_<cfg>.json`) and **default** (2 threads; files `py_<cfg>_default.json`). Thread counts confirmed with threadpoolctl-style check: default 2, variable gives 1.

## Commands (repo root; wrapper is `baseline/run_stage1.sh <small|medium|large>`)
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config small  --reps 30 --warmup 3 --out baseline/results/py_small.json
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config medium --reps 10 --warmup 2 --out baseline/results/py_medium.json
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config large  --reps 5  --warmup 1 --out baseline/results/py_large.json
    (same three without the variable -> py_<cfg>_default.json)

## Results, 1 BLAS thread (median ms)
| Config | Forward | Loss | Backward | Adam | Train step | Train tok/s | Inference fwd | Inference tok/s | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| small | 45.4 | 1.3 | 59.0 | 3.7 | 109.8 | 4,734 | 45.8 | 11,364 | 83 MB |
| medium | 947.0 | 5.4 | 1308.4 | 29.1 | 2294.4 | 900 | 941.2 | 2,193 | 711 MB |
| large | 3911.5 | 5.4 | 5352.4 | 156.7 | 9463.0 | 217 | 3839.4 | 536 | 2354 MB |

## Results, default threads (2)
| Config | Forward | Loss | Backward | Adam | Train step | Train tok/s | Inference fwd | Inference tok/s | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| small | 55.2 | 1.5 | 73.5 | 5.0 | 137.8 | 3,774 | 55.9 | 9,295 | 84 MB |
| medium | 951.8 | 7.7 | 1236.4 | 32.1 | 2260.8 | 913 | 964.3 | 2,140 | 710 MB |
| large | 3609.3 | 6.0 | 4215.4 | 157.5 | 7976.8 | 258 | 3404.2 | 604 | 2352 MB |

## Share of one training step (1 thread)
| Config | Forward | Loss | Backward | Adam |
|---|---|---|---|---|
| small | 41.4% | 1.2% | 53.7% | 3.4% |
| medium | 41.3% | 0.2% | 57.0% | 1.3% |
| large | 41.3% | 0.1% | 56.6% | 1.7% |

Rule of thumb: **forward ~41%, backward ~55%, Adam ~1-4%**, loss ~0-1%.

## Repeatability (small, 1 thread, 30 reps each)
Train-step medians: 111.1, 110.3, 110.6, 109.8 ms. Spread 1.1%, within the "few percent" requirement.

## Power-state check (why these numbers replaced the first set)
The first set was taken without confirming AC power or power mode. After plugging in and selecting Best performance, small train step dropped from 136.7 to 109.8 ms (-20%), so all three configs were rerun. Medium moved 2382 -> 2294 ms (-4%) and large 9946 -> 9463 ms (-5%). Old files are kept in `baseline/results/prev_unverified_power/` and should not be used.

## Caveats
- Machine is a VM guest on an older (2012) laptop CPU, so absolute times are 2-3x slower than the guide's worked example (small train step ~110 ms vs 58 ms). Ratios match the example.
- Large peak RSS is 2.35 GB of 3.8 GiB RAM with no swap; it fit, but there is little headroom.
- Default (2-thread) is slower than 1 thread for small (138 vs 110 ms) and faster for large (7.98 vs 9.46 s); with 2 logical CPUs on one physical core, extra threads only pay off for large matrices.

## What was surprising
Time grows slower than work: medium does roughly 30x the arithmetic of small (about 8x per token from layers x d_model^2, times 4x more tokens) but its forward pass takes only ~21x longer (45 -> 947 ms), so small is dominated by per-call and Python overhead, and efficiency improves with size (rough estimate; exact FLOPs come in Stage 2). Also, the power state mattered most for small (-20%) and little for medium/large, and Adam is only 1-4% of a step with loss under 1%, so almost all training cost is forward/backward matrix math, with backward ~1.3x forward.

## Training versus inference (updated in Stage 6; provisional until the Phase 1.3 roofline)
Full numbers and machine labels are in `docs/stage6_comparison.md`. With the C port (medium config): forward is about 43-44% of a training step, backward 53-55%, Adam about 1%, loss under 0.2% (Machine A and Machine B agree; NumPy gives 41% / 55-57% / 1-4% / under 1%). Inference (forward only) runs about 2.3x faster in tokens per second than training (about 3,450 vs 1,496 tokens/s on Machine A, 12,550 vs 5,485 on Machine B). Backward costs about 1.3x forward because each forward GEMM needs two backward GEMMs; of all dgemm time, forward GEMMs are 33-40% and backward 60-67%. Training must also keep activations for backward; the attention probability matrices alone are 1.1 / 8.5 / 33.8 MB per layer for small / medium / large. Kernels shared by training and inference: all forward GEMMs, softmax, GELU forward, layer-norm forward. Training only: the backward GEMMs, gelu_bwd, ln_bwd, softmax backward and Adam.
