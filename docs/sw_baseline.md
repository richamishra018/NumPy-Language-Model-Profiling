# Python / NumPy software baseline (Stage 1)

Measured 2026-11-11 on the machine in [machine.md](machine.md) (Intel i5-3210M, 2 logical CPUs, 3.8 GiB RAM, Linux guest under a Microsoft hypervisor).
Python 3.10.12, NumPy 2.2.6, BLAS = bundled OpenBLAS 0.3.29 (scipy-openblas), core Sandybridge.
Unmodified `src/transformer.py`; fixed batch, seed 42; tokens per step = batch_size x (seq_len + 1).

## Thread setting
Two runs per config: **1 BLAS thread** (`OPENBLAS_NUM_THREADS=1`, the fair single-core reference; files `py_<cfg>.json`) and **default** (2 threads; files `py_<cfg>_default.json`). Default thread count was confirmed as 2 and the variable as changing it to 1.

## Commands (repo root; wrapper is `baseline/run_stage1.sh <small|medium|large>`)
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config small  --reps 30 --warmup 3 --out baseline/results/py_small.json
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config medium --reps 10 --warmup 2 --out baseline/results/py_medium.json
    OPENBLAS_NUM_THREADS=1 python baseline/bench.py --config large  --reps 5  --warmup 1 --out baseline/results/py_large.json
    (same three without the variable -> py_<cfg>_default.json)

## Results, 1 BLAS thread (median ms)
| Config | Forward | Loss | Backward | Adam | Train step | Train tok/s | Inference fwd | Inference tok/s | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| small  | 55.9 | 1.5 | 72.9 | 4.9 | 136.7 | 3,804 | 55.4 | 9,389 | 83 MB |
| medium | 989.5 | 5.9 | 1356.1 | 28.9 | 2382.0 | 866 | 979.4 | 2,107 | 701 MB |
| large  | 4150.9 | 5.3 | 5614.4 | 155.6 | 9946.1 | 207 | 3983.0 | 516 | 2354 MB |

## Results, default threads (2)
| Config | Forward | Loss | Backward | Adam | Train step | Train tok/s | Inference fwd | Inference tok/s | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| small  | 60.6 | 2.0 | 78.8 | 5.8 | 147.7 | 3,522 | 63.8 | 8,155 | 84 MB |
| medium | 968.9 | 7.7 | 1264.9 | 32.1 | 2298.0 | 898 | 1002.1 | 2,060 | 708 MB |
| large  | 3925.8 | 5.3 | 4855.4 | 175.3 | 8968.3 | 229 | 3724.9 | 552 | 2352 MB |

## Share of one training step (1 thread)
| Config | Forward | Loss | Backward | Adam |
|---|---|---|---|---|
| small | 40.9% | 1.1% | 53.3% | 3.6% |
| medium | 41.5% | 0.2% | 56.9% | 1.2% |
| large | 41.7% | 0.1% | 56.4% | 1.6% |

Rule of thumb: **forward ~41%, backward ~55%, Adam ~2-4%**, loss ~0-1%.

## Repeatability (small, 1 thread, 30 reps each)
Train-step medians: 135.8, 137.9, 137.6 ms (plus the final 136.7 ms run). Spread 1.5%, within the "few percent" requirement. Max/median of train_step was 1.1-1.36 for small, 1.02-1.14 for medium/large, below the 1.5x disturbance threshold.

## Caveats
- Machine is a VM guest on an older (2012) laptop CPU, so absolute times are 2-3x slower than the guide's worked example (small train step 137 ms vs 58 ms). Ratios match the example.
- Power mode and AC status are not recorded (see machine.md). Timing runs were not verified as thermally quiet beyond the repeatability check.
- Large peak RSS is 2.35 GB of 3.8 GiB RAM with no swap; it fit, but there is little headroom.
- Default (2-thread) is only 0-10% faster than 1 thread for medium/large and slightly slower for small: with 2 logical CPUs on one physical core, extra threads add little and add overhead for small matrices.

## What was surprising
Small-config forward (56 ms) is about 18x faster than medium (990 ms) even though medium has only ~8x more FLOPs per layer-token times 2x layers; i.e. small fits in cache and medium/large spill out of the 3 MB L3, so the per-FLOP cost rises with size (memory-bound behavior, relevant for the roofline). Second, Adam is only 1-4% of a step and the loss under 1%, so almost all training cost is the forward/backward matrix math; backward is ~1.35x forward.
