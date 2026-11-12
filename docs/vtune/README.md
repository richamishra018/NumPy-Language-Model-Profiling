# VTune log (Stage 4)

Machine: Windows PC "richanotebook", Intel i5-3210M (Ivy Bridge), Intel VTune Profiler 2026.6 (GUI at C:\Program Files (x86)\Intel\oneAPI\vtune\2026.6).
Runs are on **native Windows**, not the Linux guest used for Stages 1-3 (the guest hides hardware counters).
Result folders live in `vtune/` (git-ignored); reports/screenshots go in this folder.

## Setup done
- VTune opened; the Intel Software Improvement Program prompt was answered "I do NOT consent" (privacy-preserving choice).

## Run log (exact settings per run are added below as they are run)

## Finding: hardware analyses are not available on this CPU (checked in the VTune GUI, 2026-11-12)
- Performance Snapshot: "VTune Profiler cannot recognize the processor ... not applicable to the current machine microarchitecture".
- Microarchitecture Exploration: "This analysis type is not applicable to the current machine microarchitecture" (plus a note about the sampling driver / running as administrator).
- Hotspots: only **User-Mode Sampling** is selectable; **Hardware Event-Based Sampling** is greyed out.
- Cause: the i5-3210M is Ivy Bridge (2012), older than the microarchitectures supported by VTune 2026.6.
- Consequence per the guide (step 4.1 / 4.14): do Hotspots with `sampling-mode=sw` and Threading here; get counter-style evidence for Runs 3 and 4 (uarch, memory access) from Stage 5 tools, or run them on a newer Intel machine on the team.

---

# Machine B (Samsung Galaxy Book4), Stage 4 log, 2026-11-12

Machine: Samsung Galaxy Book4, Intel Core 5 120U (Raptor Lake, 2 P + 8 E cores, 12 threads), 15.7 GiB, Windows 11 Pro 10.0.26300,
VTune 2026.6.0, Python 3.14.7, NumPy 2.5.4, C via MSYS2 gcc 16.2.0 + OpenBLAS 0.3.34 (OpenMP build).
Hardware event-based sampling and Microarchitecture Exploration work here (unlike Machine A). Run from an Administrator PowerShell.
All runs use ONE BLAS thread (`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; the OpenMP build ignores the latter alone) except the threading run.
A first 12-thread attempt was discarded (invalid as 1-thread data); only its C benchmark log is kept as `docs/logs/machineB_c_bench_12thread.txt`. Full results: `stage4_machineB_results.md`.

| Result | Command (repo root) | What it showed |
|---|---|---|
| py_hotspots_1/2 | `vtune -collect hotspots -knob sampling-mode=sw -knob enable-stack-collection=true -result-dir vtune\py_hotspots_N -- python baseline\profile_steps.py --config small --steps 100` | NumPy elementwise 46-50%, libm 27-29%, BLAS 12-15%, interpreter 4-7% |
| c_hotspots_1/2 | `vtune -collect hotspots -knob sampling-mode=hw -knob enable-stack-collection=true -result-dir vtune\c_hotspots_N -search-dir .\c -source-search-dir .\c -- .\c\tx.exe bench c/ref 300` | dgemm family 41%, tanh+exp 28-29%, own code 19% |
| c_uarch_1/2 | `vtune -collect uarch-exploration -result-dir vtune\c_uarch_N -source-search-dir .\c -- .\c\tx.exe bench c/ref 300` | Retiring 38-44%, Back-End 36-38% (Core 20, Memory 16-17), CPI 0.39 |
| c_memory_1 | `vtune -collect memory-access -result-dir vtune\c_memory_1 -source-search-dir .\c -- .\c\tx.exe bench c/ref_medium 30` | DRAM peak 14.0 GB/s, 52% of STREAM 26.9 |
| c_threading | `vtune -collect threading -result-dir vtune\c_threading -- .\c\tx.exe bench c/ref_medium 20` (OMP/OPENBLAS threads = 4) | effective CPU use 5.3%, spin 58% |
| py_itt_1 | `vtune -collect hotspots -start-paused -knob sampling-mode=sw -result-dir vtune\py_itt_1 -- python baseline\profile_steps_itt.py --config small --steps 100` | backward 55%, forward 41%, Adam 2.7%, loss 1.1% |

## Self-check (Machine B, 2026-11-12)
`self_check.txt`: ready for Performance Snapshot, Hotspots and Threading (user-mode), Hotspots with hardware sampling (with call stacks; stack depth limited to hardware LBR), HPC Performance Characterization, Microarchitecture Exploration, Memory Access, Threading with hardware sampling. Only the GPU Compute/Media checks failed (no DPC++/GPU runtime; not used in this project).

## Screenshots (Machine B, single BLAS thread results in `vtune/`)
| File | Result and view |
|---|---|
| py_hotspots_summary.png | py_hotspots_1, Summary |
| py_itt_summary.png | py_itt_1, Summary with Top Tasks |
| c_hotspots_summary.png | c_hotspots_1, Summary |
| c_hotspots_bottomup.png | c_hotspots_1, Bottom-up by function |
| c_hotspots_source.png | c_hotspots_1, Assembly view of a hot ucrtbase (libm) function; no source lines exist for library code |
| c_uarch_summary.png | c_uarch_1, Summary (top-down buckets) |
| c_uarch_bottomup.png | c_uarch_1, Bottom-up with P-core bucket columns (not sorted by clockticks; hot rows are in `c_uarch_1_by_function.csv`) |
| c_memory_summary.png | c_memory_1, Summary with Bandwidth Utilization Histogram (all time in the Low band) |
| c_threading_summary.png | c_threading, Summary |
| c_threading_platform.png | c_threading, Platform timeline (one busy thread, three mostly waiting) |

## Repeats of runs 4 and 5 (2026-11-12)
| Result | Command | What it showed |
|---|---|---|
| c_memory_2 | same as c_memory_1 | DRAM peak 16.1 GB/s (run 1: 14.0), average 5.9, 0% time at high bandwidth; agrees |
| c_threading_2 | same as c_threading (4 threads) | effective CPU use 5.4% (run 1: 5.3%), spin 58.6%; agrees |
Summary text files: `c_memory_2_summary.txt`, `c_threading_2_summary.txt`.

## Run 4 CSV
`c_memory_1_by_function.csv`: Memory Access (medium config) per-function export, 224 rows with P-core Memory Bound, L1/L2/L3 columns. Top function by CPU time: `dgemm_kernel_HASWELL` about 45%, then `tanh` about 13%, `gelu_bwd` about 5%.

## Not done (optional in the guide)
C ITT markers (step 4.10, C half) and the call-stack report (step 4.11); the Python ITT run is done.
