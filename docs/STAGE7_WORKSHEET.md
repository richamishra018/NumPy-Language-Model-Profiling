# Stage 7 worksheet: Checkpoint, end of Phase 1.2

This is an **evidence worksheet**, not the answers. The guide says copying defeats the point, so each question below gives (a) what the question is really asking, (b) the measured evidence with the file to open, (c) pitfalls, and (d) a blank for **your own written answer** (a few sentences, in your words). Then be ready to explain each aloud to Sriram.

Machine labels: **A** = older laptop (i5-3210M, no hardware counters), **B** = Samsung Galaxy Book4 (Core 5 120U, VTune counters work). Never mix A and B numbers in one comparison. Repo paths are relative to the repo root.

---

## 0. The question behind the whole phase: which part of the model goes on the hardware?

**Where the evidence stands (provisional, to be confirmed by the Phase 1.3 roofline):**
1. **GEMM family first** (feed-forward W1/W2, QKV and output projections, per-head attention products). Largest block of the best software: dgemm = 43.5% / 57.9% / 65.7% of the C run for small / medium / large (A, exact, `docs/logs/dgemm_timer.txt`) and 41% for small (B, VTune, `docs/vtune/stage4_machineB_results.md`). Compute bound: dgemm kernel Retiring 58-65%, Core Bound 30-35%, Memory Bound under 5% (B, `docs/vtune/c_uarch_*_summary.txt`).
2. **Fused GELU (tanh) + softmax (exp) second.** GELU incl. tanh is 26.6-27.6% of the small C run and softmax's exp about 10% (B, VTune hotspots). They are limited by libm instruction latency and branches (Retiring 31-38%, Bad Speculation 22-36%, Memory about 0), not by memory.
3. **Honest ceiling (Amdahl, `docs/kernel_ranking.md`):** GEMM alone gives at most 1.77x / 2.38x / 2.92x (A) or 1.69x (B, small); GEMM + GELU + softmax exp about 4.5x (B, small). Host-to-FPGA transfer and CPU leftovers make real gains lower.

**What this phase did not settle:** VTune ran on the small config only (medium/large shares are Machine A timings); FPGA resources and numeric precision (the port is float64) are not yet part of the analysis; the roofline is Phase 1.3.

Write your own one-paragraph answer here: ______________________________________________

---

## The twelve questions

### Q1. Why is tottime the right cProfile column for hotspots and cumtime for structure?
- **Asking:** the difference between time inside a function and time including its callees.
- **Evidence** (`baseline/results/cprofile_small.txt`, 30 steps): `gelu_grad` tottime 0.745 s = cumtime 0.745 s (a leaf, so the two agree); `ff_backward` tottime 0.307 s but cumtime 1.066 s (most of its time is in `gelu_grad`, which it calls); `backward` tottime 0.037 s but cumtime 1.753 s (it only orchestrates). Ranking by cumtime would put `backward`/`forward` on top, which tells you nothing about where work happens.
- **Pitfall:** cProfile inflates every call; trust ranking and proportions, not seconds (the guide's own warning).
- **Also see:** `docs/snakeviz_small.png` (icicle view shows exactly this nesting).
- Your answer: ______________________________________________

### Q2. The `@` operator never appears in cProfile. Where does matmul time show up, and which two tools separate it from elementwise time?
- **Asking:** `@` is not a Python function call, so its time is charged to the function containing it; then which tools see inside.
- **Evidence:** cProfile charges matmul time to `mha_backward`, `ff_backward`, etc. (their tottime includes their dgemm calls). **py-spy with `--native`** (`baseline/results/spy_small.json`): leaf samples `dgemm_kernel_SANDYBRIDGE` 16.1% + `dgemm_beta` 1.2% (matrix multiply about 17%), against `expm1`/`tanh` about 12%, `DOUBLE_multiply` 5.6%, `DOUBLE_add` 4.8%. **line_profiler** (`baseline/results/lineprofile_small.txt`) separates lines inside a function (e.g. `x ** 3` and `tanh` lines in `gelu_grad` at 48% and 35%). **VTune on the Python run** (B, `docs/vtune/py_hotspots_*_summary.txt`): NumPy elementwise loops 46-50%, libm 27-29%, OpenBLAS 12-15%, interpreter 3.5-6.7%.
- **Pitfall:** the guide names line_profiler (step 2.3) and native py-spy (step 2.4) as the two; VTune is a third way to say the same thing.
- Your answer: ______________________________________________

### Q3. Which function has the highest self time in your Python profile? Is it a matmul? Why can elementwise functions dominate at small sizes?
- **Evidence** (`docs/kernel_ranking.md`, scaling study): small: `gelu_grad` 22.4%, `gelu` 20.1%, `mha_backward` 10.5%; medium: `gelu_grad` 20.7%, `gelu` 19.1%, `ff_backward` 13.3%; large: `mha_backward` 18.3%, `ff_backward` 17.3%, `gelu_grad` 14.8%. The GELU pair falls from 42.5% to 26.3% as size grows.
- **For the "why":** GELU is `tanh`, `x ** 3` and several temporaries per element, 4.2 M elements per pass at medium, with almost no arithmetic per byte; FLOPs per step for the GEMMs grow with d_model squared while elementwise work grows only linearly (compare `baseline/results/flops_*.txt` with the hand estimate of about 118 MFLOP for GELU, about 1% of the step's FLOPs, in `docs/kernel_ranking.md`). In C (B, VTune) GELU incl. tanh is still about 27% of the small run.
- **Pitfall:** the answer to "is it a matmul" is no, but the hottest function depends on the config (large: `mha_backward`).
- Your answer: ______________________________________________

### Q4. forward() runs on seq_len + 1 positions. Where does that come from and which numbers does it change?
- **Evidence:** `get_batch` returns rows of seq_len + 1 tokens; `forward()` runs on the whole row; the loss shifts by one to make targets (guide's pitfall box, and `pos_emb` has seq_len + 1 rows). So T = 65 for small, 129 for medium, 257 for large; tokens per step = batch x (seq_len + 1) = 520 / 2,064 / 2,056.
- **Numbers it changes:** analytic FLOPs (`baseline/flops.py` uses T = seq_len + 1: 369.4 / 11,460.7 / 68,118.4 MFLOP per step), tokens per second, and the attention matrix size (8 x 4 x 65 x 65 x 8 B = 1.1 MB for small, 8.5 MB medium, 33.8 MB large per layer).
- **Pitfall:** using seq_len is a silent error: it understates attention FLOPs and tokens per step.
- Your answer: ______________________________________________

### Q5. NumPy already calls OpenBLAS for `@`. Where does the C port's speedup come from, and why does it shrink as the model grows?
- **Evidence:** speedup C over Python, 1 thread: A 1.83x / 1.58x / 1.44x; B 2.7x / 2.4x / 2.1x (small / medium / large; `docs/c_port.md`, `docs/machineB.md`). The sources are not the matmul: interpreter overhead, new temporary arrays in almost every NumPy line, and elementwise functions that make several passes (guide, Stage 3). Memory: C peaks at 260 MB vs Python 711 MB at medium on A (`docs/logs/c_medium_threads_peakmem.txt`).
- **Why it shrinks:** the dgemm share of the C run rises with size (A: 43.5% / 57.9% / 65.7%), and both versions run the same kind of BLAS for that part, so a growing fraction of the time is untouched by the port.
- **Pitfall:** the two BLAS builds differ (A: C used OpenBLAS 0.3.15 vs NumPy's 0.3.29; B: 0.3.34 vs NumPy's bundled build), which is a small caveat on the gap.
- Your answer: ______________________________________________

### Q6. What did your thread study show? Why do small matrices scale poorly with more threads?
- **Evidence:** A (`docs/c_port.md`, `docs/logs/c_flag_thread_study.txt`): C medium 1373.8 ms (1 thread) vs 1238.5 (2) = 1.11x, small flat to slightly worse; asking for 4 threads gave `threads: 2` (capped at the 2 logical CPUs). Python medium 2294 vs 2261 ms (default 2 threads). dgemm alone scales 21.5 to 33.2 GFLOP/s. B: 12-thread C step vs 1-thread: small 0.94x, medium 0.97x, large 1.36x (`docs/machineB.md`); VTune Threading run (medium, 4 threads): effective CPU utilization about 5.3%, spin and overhead about 58% (`docs/vtune/c_threading_summary.txt`, `c_threading_platform.png`).
- **Why:** tiny matrices make the coordination cost (thread wake-up, semaphores) as large as the work saved; per-head tiles are 129 x 32; the elementwise loops in `tx.c` are single threaded regardless; on A the 2 logical CPUs are hyper-threads of one core.
- **Pitfall:** the OpenMP build on B ignores `OPENBLAS_NUM_THREADS` (needs `OMP_NUM_THREADS`), a caveat for any thread-count claim.
- Your answer: ______________________________________________

### Q7. Your C train step reaches X GFLOP/s but gemm_bench reaches Y. What fills the gap, according to VTune?
- **Evidence:** A: 8.3 vs 21.5 GFLOP/s at medium; B: 30.5 vs 73.4 GFLOP/s at medium (39-42%). VTune on B (small config, `docs/vtune/stage4_machineB_results.md`): the run is dgemm 41%, libm tanh 18-20%, exp about 10%, own code 19%; the dgemm kernel itself is Retiring 58-65% with Core Bound 30-35%. The exact per-kernel dgemm timer on A (`docs/logs/dgemm_by_kernel.txt`) shows big GEMMs run at 14-16 GFLOP/s but the per-head attention products at only 5-10 GFLOP/s, and the dgemm packing helpers (`dgemm_oncopy/otcopy/itcopy`, about 7% of the run on B) are 46-100% Memory Bound.
- **Pitfall:** VTune data covers the small config only, so say so; the shares at medium/large come from A's timer.
- Your answer: ______________________________________________

### Q8. Which top-down bucket dominated in your microarchitecture run? What does that say about compute vs memory?
- **Evidence** (B, small, whole program, two runs, `docs/vtune/c_uarch_1_summary.txt`, `c_uarch_2_summary.txt`, `c_uarch_summary.png`): Retiring 43.5% / 37.9% is the largest bucket; Back-End Bound 36-38% (Core Bound about 20%, Memory Bound about 16-17%); Front-End about 10%; Bad Speculation 8.7% / 15.9% (the only bucket that moved much); CPI 0.39. Per function: dgemm kernel Retiring 58-65%, Memory under 5%; tanh and gelu_bwd Bad Speculation 22-41%, Front-End 11-28%, Memory about 0. Memory Access (medium): DRAM peak 14-16 GB/s = 52-60% of the 26.9 GB/s STREAM ceiling, 0% of the time in the high-bandwidth band.
- **What it says:** the best software is compute and instruction-latency limited, not bandwidth limited.
- **Pitfall:** the run-to-run spread (hybrid P/E cores, process not pinned); report ranges.
- Your answer: ______________________________________________

### Q9. What fraction of the C train step is forward vs backward? What does that say about training vs inference?
- **Evidence** (`docs/stage6_comparison.md`): medium, C, 1 thread: A 598 / 726 ms of 1380 = 43% forward, 53% backward; B 164.5 / 207.1 ms of 376.3 = 44% / 55%; Adam about 1%. NumPy: 41% / 55-57%. Backward is about 1.3x forward (each forward GEMM needs two backward GEMMs). Inference (forward only) runs about 2.3x the training tokens per second (A 3,450 vs 1,496; B 12,550 vs 5,485). Of all dgemm time, forward GEMMs are 33-40%.
- **For the decision:** a forward-only GEMM engine serves both modes; supporting training means backward GEMMs (about two thirds of GEMM time), `gelu_bwd`, `ln_bwd` and Adam, plus memory for saved activations (attention matrices 1.1 / 8.5 / 33.8 MB per layer).
- Your answer: ______________________________________________

### Q10. Why must profiling use an optimized build with debug symbols, not -O0?
- **Evidence:** the guide's flag table (-O3 is what runs in production, -g for symbols and source lines, -fno-omit-frame-pointer for reliable call stacks) and VTune troubleshooting ("function names but no source lines" without -g). Your flag study (`docs/c_port.md`, small, A): -O0 73-74 ms, -O2 70, -O3 68, -O3 -march=native 68 ms; the difference is only about 8% because nearly all time is inside prebuilt OpenBLAS and libm, which the flags do not touch; -O2 vs -O3 is within noise.
- **Be careful:** your own data shows -O0 is not wildly slower here; the argument is about what the profile represents (own-code functions are inflated and inlining/vectorization change, so the ranking of own code at -O0 is not the ranking of the program you would ship), and that without -g tools cannot attribute time to functions and lines.
- Your answer: ______________________________________________

### Q11. Which kernel do you propose to accelerate? Give three pieces of measured evidence and the best possible overall speedup from Amdahl's law.
- **Evidence to cite** (`docs/kernel_ranking.md`, "Provisional proposal"): (1) dgemm is the largest block at every size: 43.5 / 57.9 / 65.7% (A, exact timer) and 41% (B, VTune, small); (2) it is compute bound: Retiring 58-65%, Core Bound 30-35%, Memory under 5% of slots, DRAM peak only 52-60% of STREAM (B); (3) the best software reaches only 39-42% of peak at medium (A 8.3 of 21.5, B 30.5 of 73.4 GFLOP/s) and per-head attention products only 5-10 GFLOP/s (A), so there is room.
- **Amdahl:** S = 1 / (1 - p). All GEMMs: p = 0.435 / 0.579 / 0.657 gives 1.77x / 2.38x / 2.92x (A); p = 0.41 gives 1.69x (B, small). Feed-forward GEMMs alone: 1.27-1.49x. GEMM + GELU + softmax fused: p about 0.78 gives about 4.5x (B, small).
- **Pitfall:** state the evidence gaps (VTune only small; FPGA resources and precision not yet analysed) and that transfer cost is ignored.
- Your answer: ______________________________________________

### Q12. What result, or what change of model size, would make you change your mind?
- **Things the evidence already hints at** (think through each in your own words): the GELU share falls with size in Python (42.5% to 26.3%), so if VTune on medium/large C shows GELU staying near 27%, the fused-elementwise block matters more; attention memory grows with T squared (33.8 MB per layer at large), so at large sizes the attention path may force tiling and become the limiting block; a much larger vocabulary would make the vocabulary projection non-negligible (it is 0.3-1.7% now); if the roofline (Phase 1.3) shows the GEMMs would sit under the FPGA's memory roof rather than the compute roof, the on-chip buffering design matters more than raw multipliers; if the FPGA board's DSP count or block RAM cannot hold the working tiles; and if lower precision (fixed point or float32/bfloat16) changes accuracy or the speed comparison (the C port is float64).
- Your answer: ______________________________________________

---

## Checkpoint bullets (from the guide)
- [ ] Short notes answering all twelve questions, in your own words.
- [ ] One screenshot to explain "why the best software is limited by what it is limited by". Suggested: `docs/vtune/c_uarch_summary.png` (top-down buckets) together with `docs/vtune/c_hotspots_summary.png` (dgemm vs libm vs own code); fallback if no VTune: `docs/fallback_profiling.md` section 1 (dgemm share table).
- [ ] **Roofline prediction (before Phase 1.3):** will your chosen kernel sit under the memory roof or the compute roof, and why? It is a prediction, not a measurement. Evidence to reason from: ridge points A 2.0 and B 2.7 FLOP/byte (`docs/stage6_comparison.md`); GEMM arithmetic intensity 5.3 (per-head tile) to 24.4 (feed-forward) FLOP/byte (`docs/kernel_ranking.md`, shapes table); VTune Memory Access found no bandwidth limit on the CPU. Remember the roofline that matters is the FPGA board's, not the CPU's.
  Your prediction: ______________________________________________
- [ ] Deliverables committed (table below) and the repo can be cloned and rerun using only the README.

## Deliverables check
| Deliverable | Path | Present in repo |
|---|---|---|
| Machine description and ceilings | `docs/machine.md`, `docs/machineB.md` | yes |
| Python baseline numbers and write-up | `baseline/results/py_*.json`, `machineB_py_*.json`, `docs/sw_baseline.md` | yes |
| Python profiles | `baseline/results/cprofile_*.txt/.prof`, `lineprofile_*.txt`, `spy_*.json/.txt`, `mem_*.png`, `flops_*.txt` | yes (mem plot small only) |
| C port, logs and write-up | `c/` sources, `docs/logs/`, `docs/c_port.md` | yes |
| VTune evidence | `docs/vtune/` (self_check, CSV exports, screenshots, README) | yes (Machine B); Machine A: not supported |
| Fallback evidence | `docs/fallback_profiling.md` | yes (Machine A) |
| Ranking and Phase 1.3 inputs | `docs/kernel_ranking.md`, `docs/stage6_comparison.md` | yes |

**README gap to fix before handing in:** the README's reproduction section covers Linux/macOS commands and the OpenBLAS caveat for Machine A; it does not yet describe the Windows/MSYS2 build used on Machine B (`scripts/windows/build_tx_msys2.sh`, `OMP_NUM_THREADS=1`). A teammate rerunning only from the README would miss it. A short "Windows (MSYS2)" subsection should be added.

## Known open items (do later)
VTune hotspots/uarch/ITT for the medium and large C configs on Machine B; Machine B Python default-thread run and C peak memory; refine the elementwise FLOP hand-counts in Phase 1.3; C ITT markers.
