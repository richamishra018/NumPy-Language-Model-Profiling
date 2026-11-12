# Stage 6: comparison table, Phase 1.3 inputs, training vs inference

Two computers produced the evidence and are **never mixed inside one row**:
- **Machine A** = Richa's older laptop (Intel i5-3210M Ivy Bridge, 2 logical CPUs in the Linux guest, 3.8 GiB). Stages 1-3 and 5. VTune hardware analyses are unsupported on it.
- **Machine B** = Samsung Galaxy Book4 (Intel Core 5 120U Raptor Lake-U, 2 P + 8 E cores, 12 threads, 15.7 GiB). Stage 4 (VTune with hardware counters) and its own baselines.
Headline config: **medium** (d_model 128, 4 layers, batch 16, seq 128; 2,064 positions per step; 11,460.7 MFLOP per training step). Medians from quiet, plugged-in runs; small and large are in `sw_baseline.md`, `c_port.md`, `machineB.md`.

## 6.1 Comparison table, medium config

### Machine A (i5-3210M)
| Version | Threads | Forward ms | Backward ms | Step ms | Tokens/s (train) | Achieved GFLOP/s | Peak memory |
|---|---|---|---|---|---|---|---|
| Python + NumPy | 1 | 947.0 | 1308.4 | 2294.4 | 900 | 5.00 | 711 MB |
| Python + NumPy | default (2) | 951.8 | 1236.4 | 2260.8 | 913 | 5.07 | 710 MB |
| C + OpenBLAS | 1 | 598.0 | 725.7 | 1379.5 | 1,496 | 8.31 | 260 MB |
| C + OpenBLAS | all (2) | 508.2 | 656.4 | 1187.9 | 1,738 | 9.65 | 265 MB |
| Reference ceiling (gemm_bench) | 1 / all (2) | | | | | 21.5 / 33.2 | |

Notes: the two C rows are one back-to-back measurement (6 reps, `docs/logs/c_medium_threads_peakmem.txt`); the 1-thread C step was 1452 ms in the Stage 3 run, so allow about 5% run-to-run drift. The 2-thread run had a 7 s outlier (max); medians used. C peak memory from `/usr/bin/time -v` (maximum resident set size). Machine A speedup of C over Python at 1 thread: 1.66x.

### Machine B (Samsung Galaxy Book4)
| Version | Threads | Forward ms | Backward ms | Step ms | Tokens/s (train) | Achieved GFLOP/s | Peak memory |
|---|---|---|---|---|---|---|---|
| Python + NumPy | 1 | 351.4 | 530.7 | 899.4 | 2,295 | 12.74 | 671 MB |
| Python + NumPy | default | not measured | | | | | |
| C + OpenBLAS | 1 | 164.5 | 207.1 | 376.3 | 5,485 | 30.46 | not measured |
| C + OpenBLAS | all (12) | 173.4 | 210.2 | 388.3 | 5,316 | 29.52 | not measured |
| Reference ceiling (gemm_bench) | 1 / all (12) | | | | | 73.4 / 209.6 | |

Speedup of C over Python at 1 thread: 2.39x. Threads do not help the C step at medium size (0.97x). Memory bandwidth ceiling: 26.9 GB/s (STREAM triad), platform max 55 GB/s (VTune).
Machine B's OpenBLAS is an OpenMP build that ignores `OPENBLAS_NUM_THREADS`; "1 thread" requires `OMP_NUM_THREADS=1` (`tx.exe` printed `threads: 1`).

**Empty cells and owners** (the Stage 6 done-when allows a named owner): Machine B "Python default threads" and Machine B "C peak memory" were not measured. Owner: Richa, on the Samsung Book. Commands: `python baseline\bench.py --config medium --reps 10 --warmup 2 --out baseline\results\machineB_py_medium_default.json` with `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` unset, and for peak memory `Get-Process tx | Select PeakWorkingSet64` while `.\c\tx.exe bench c/ref_medium 5` runs (or Task Manager's peak working set). Machine A has no empty cells.

### Reading the table
- The C port is 1.7x (Machine A) to 2.4x (Machine B) faster than NumPy at medium size; the gain falls with model size (A: 1.8x, 1.6x, 1.4x; B: 2.7x, 2.4x, 2.1x for small, medium, large) because larger matrices spend more time in OpenBLAS, which both versions share.
- Even C reaches only 39-42% of its machine's single-thread dgemm ceiling at medium size (28-48% across all three configs) (A: 8.3 of 21.5 = 39%; B: 30.5 of 73.4 = 42%). The rest of the step is not efficient matrix math (see `kernel_ranking.md`).
- C needs about 2.7x less memory than NumPy at medium size on Machine A (260 vs 711 MB), because NumPy creates temporaries.

## 6.3 Inputs for Phase 1.3 (roofline)
Use **one machine's ceilings with that same machine's measurements**. Machine B is the primary candidate (counters work); Machine A is a second dataset.

| Input | Machine A | Machine B | Source | Status / owner |
|---|---|---|---|---|
| Peak compute, 1 thread | 21.5 GFLOP/s | 73.4 (67.4 at N=1024) | gemm_bench | collected |
| Peak compute, all threads | 33.2 (2 logical CPUs) | 209.6 (12 threads) | gemm_bench | collected |
| Peak memory bandwidth | 10.7 GB/s | 26.9 GB/s (platform max 55 per VTune) | stream_triad | collected |
| Ridge point, 1 thread (compute / bandwidth) | 2.0 FLOP/B | 2.7 FLOP/B | derived | collected |
| FLOPs per step per kernel (matmul) | see `kernel_ranking.md` | same (model is identical) | flops.py | collected |
| FLOPs per step for elementwise kernels | hand estimate in `kernel_ranking.md` | same | hand count | estimate; refine in Phase 1.3 (owner: team) |
| Time per kernel in the C version | dgemm per kernel exact (`docs/logs/dgemm_by_kernel.txt`), small/medium/large | VTune function shares, small only (`docs/vtune/stage4_machineB_results.md`) | preload timer, VTune hotspots | partial: VTune on medium/large and C ITT tasks not run. Owner: Richa (Samsung) |
| Achieved GFLOP/s per kernel | dgemm per kernel from the timer (14-16 GFLOP/s for big GEMMs, 5-10 for per-head attention products) | per-kernel time x FLOPs once the above exist | derived | partial, same owner |
| Minimum bytes moved per kernel | tensor shapes in `kernel_ranking.md` | same | computed from shapes | started; finish in Phase 1.3 (owner: team) |

## 6.4 Training versus inference (provisional until the Phase 1.3 roofline)
- **Forward vs backward share of the C step (medium):** Machine A 598 / 1380 = 43% forward, 726 / 1380 = 53% backward, Adam 1.6%, loss 0.2%. Machine B 164.5 / 376.3 = 44% forward, 55% backward, Adam 0.9%, loss 0.1%. In Python (both machines) forward is about 41% and backward 55-57%. Backward costs about 1.3x forward, because every GEMM in forward needs two GEMMs in backward.
- **Tokens per second at medium, C, 1 thread:** training 1,496 (A) / 5,485 (B); inference (forward only) about 3,450 (A) / 12,550 (B), i.e. about 2.3x the training rate.
- **Memory to keep activations for backward:** the attention probability matrices alone are 1.1 MB (small), 8.5 MB (medium) and 33.8 MB (large) per layer in float64, and they must be kept only for training; inference does not need to keep them (the C and NumPy forward passes still allocate them, so this is an upper bound on savings).
- **Kernels shared by both:** QKV, output and feed-forward GEMMs, per-head attention GEMMs, softmax, GELU forward, layer-norm forward, vocabulary projection. **Training only:** the extra backward GEMMs (about 2x forward), gelu_bwd, ln_bwd, softmax backward, loss gradient, Adam.
- Consequence: forward kernels serve both modes and are 44% of a training step but 100% of inference. Of the dgemm time, forward GEMMs are only about a third (33% small, 34% medium, 40% large) and backward GEMMs 60-67% (`docs/logs/dgemm_by_kernel.txt`), so a GEMM engine aimed only at inference covers a smaller share of training than of inference.
