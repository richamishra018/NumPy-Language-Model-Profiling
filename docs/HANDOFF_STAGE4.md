# Handoff: Stage 4 (Intel VTune) on the Samsung Galaxy Book4

Project: NumPy Language Model Profiling (Phase 1.2, "Baseline Profiling: Python, C + OpenBLAS, Intel VTune"). Team: Abrar, Geetha, Richa. Mentor: Sriram.
Repo: https://github.com/richamishra018/NumPy-Language-Model-Profiling (branch `main`)
Guide: `Phase_1_2_Baseline_Profiling.docx` (Stage 4 starts at "STAGE 4 · INTEL VTUNE PROFILER"; the stage numbers below match it).

How to use this note: paste it into the new project chat, then say "continue Stage 4 from this handoff". Machine rule for this chat: **everything in Stage 4 is Machine B (Samsung Galaxy Book4). Never mix Machine B numbers with Machine A numbers in one comparison.**

---

## 1. Why Stage 4 is being redone on a different laptop

- Machine A = Richa's old laptop "richanotebook": Intel Core i5-3210M (Ivy Bridge, 2012), 2 cores / 4 threads (the Linux guest sees 2 CPUs), 3.8 GiB RAM visible, Windows host, VTune 2026.6.
- On Machine A, VTune 2026.6 **cannot do hardware analyses**: Performance Snapshot says "cannot recognize the processor"; Microarchitecture Exploration says "not applicable to the current machine microarchitecture"; Hotspots only offered User-Mode Sampling (Hardware Event-Based Sampling greyed out). Details are in `docs/vtune/README.md`.
- Because of that, Machine A got Stage 5 fallback evidence instead (gprof, an LD_PRELOAD dgemm timer; perf/valgrind unavailable). See `docs/fallback_profiling.md`.
- Machine B is expected to be supported, so it can run the hardware analyses Machine A could not: Microarchitecture Exploration (Run 3) and Memory Access (Run 4), plus hardware-sampling Hotspots on the C program (Run 2).

**First action on Machine B: confirm support** (open VTune GUI, New Project, Configure Analysis, pick Microarchitecture Exploration). If it says "not applicable", stop and report: Stage 4 hardware runs are impossible there too.

## 2. What is already done (Machine A, do not redo)

| Stage | State | Where |
|---|---|---|
| 0 Setup | done | `docs/machine.md` |
| 1 Python baseline (3 configs, 1 and 2 BLAS threads) | done | `docs/sw_baseline.md`, `baseline/results/py_*.json` |
| 2 Python profiling (cProfile, snakeviz, line_profiler, py-spy, mprof, FLOPs) | done | `docs/kernel_ranking.md`, `baseline/results/` |
| 3 C + OpenBLAS port (validated, benchmarked, flag and thread studies) | done | `docs/c_port.md`, `docs/logs/` |
| 4 VTune | **not done on Machine A (CPU unsupported); to be done on Machine B** | this handoff |
| 5 Fallback tools | done on Machine A (gprof + dgemm timer) | `docs/fallback_profiling.md` |
| 6 Consolidate, 7 Checkpoint | not started | |

### Machine A headline numbers (1 BLAS thread; for later comparison only, do not mix with Machine B)
| Config | Python step | C step | Speedup | C GFLOP/s |
|---|---|---|---|---|
| small | 109.8 ms | 60.2 ms | 1.8x | 6.1 |
| medium | 2294 ms | 1452 ms | 1.6x | 7.9 |
| large | 9463 ms | 6574 ms | 1.4x | 10.4 |

- Step share (all configs): forward about 41%, backward about 55%, Adam 1-4%, loss under 1%.
- Machine A ceilings: dgemm 21.5 GFLOP/s (1 thread), 33.2 (2 threads); STREAM triad 10.7 GB/s.
- C port matches NumPy to about 1e-14 on all three configs.
- Python profile (small): top self time `gelu_grad` 22%, `gelu` 20%, `mha_backward` 10.5%, `ff_backward` 9%, `softmax` 5.6%. py-spy with native frames: matrix-multiply kernel only about 17% of samples.
- Exact dgemm share of the C run (Machine A, preload timer): small 43.5%, medium 57.9%, large 65.7%; dgemm ran at only about 11.5 GFLOP/s (about half the ceiling) because of the many small per-head calls (about 420 per step for small).
- gprof own-code top (small): `gelu_bwd` 31%, `adam_step` 18%, `backward` loops 13%, `forward` loops 10% (excludes OpenBLAS and libm).

### The written prediction that Stage 4 must check (from `docs/c_port.md` and `docs/fallback_profiling.md`)
- small: GELU pair (`gelu_fwd`/`gelu_bwd`, dominated by libm `tanh`) roughly as costly as `dgemm_kernel` or more. Machine A evidence did not confirm this (dgemm 43%).
- medium: dgemm kernel is the top symbol (prediction 40-50%; Machine A measured 57%).
- large: dgemm clearly hottest (prediction 50% or more; Machine A measured 66%).
- Expected pipeline picture: dgemm kernels Retiring-heavy or Core Bound; elementwise loops (GELU, layer norm, Adam) Memory or latency bound.
- Where the Machine A estimate was wrong: dgemm share was 14-20 points higher than predicted because small per-head dgemm calls are inefficient.

## 3. Open questions Stage 4 on Machine B should answer

1. What fraction of the C program is the dgemm family vs own functions (hardware sampling)? Compare with the 43/58/66% from Machine A.
2. Retiring / Bad Speculation / Front-End Bound / Back-End Bound (Memory Bound vs Core Bound), per whole run and per function (dgemm kernel vs `gelu_bwd`, `ln_bwd`, `adam_step`).
3. Is the code bandwidth limited? Peak DRAM bandwidth reached (medium config) vs the STREAM triad ceiling measured **on Machine B**.
4. Does threading help (Effective CPU utilization, spin and wait time) and does it match Machine B's thread study?
5. Python run: how much of time is interpreter vs BLAS vs NumPy elementwise loops?
6. Which row of the interpretation table (guide step 4.13) fits the data; what does it imply for the FPGA accelerator?

## 4. Prerequisites on Machine B

Do these before any measurement.
1. Windows plugged in, power mode "Best performance", browsers/video calls closed, do not touch the PC during runs. Record this.
2. Software: Git, Python 3.10+ (Windows), VTune (version noted), MSYS2 (https://www.msys2.org). Use the **Intel oneAPI command prompt** run as **Administrator** for every vtune command, and the **MSYS2 UCRT64** shell for building C.
3. VTune sampling driver installed (the self-check in section 6 tells you).
4. If the VTune GUI shows the Intel Software Improvement Program prompt, either answer is fine; Machine A used "I do NOT consent".

Tool note from Machine A: terminals can be clicked but not typed into by Claude's computer-use feature, so you (the human) run the commands in sections 5 to 8 and paste outputs back; Claude can write scripts, read the exported reports and analyse them. VTune GUI can be driven by Claude if the session is linked to Machine B and VTune is allowed.

## 5. Setup on Machine B (PowerShell, in a folder such as `C:\NumPy Language Model Profiling`)

```powershell
git clone https://github.com/richamishra018/NumPy-Language-Model-Profiling
cd NumPy-Language-Model-Profiling
py -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install numpy matplotlib psutil line_profiler snakeviz py-spy memory_profiler threadpoolctl ittapi tomli
```

Create `docs/machineB.md` with: CPU model, core and thread count, RAM, Windows version, Python and NumPy versions, VTune version, power mode, AC yes/no.
```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
python -c "import numpy; numpy.show_config()"
python -c "from threadpoolctl import threadpool_info; [print(i['internal_api'], i['version'], 'threads', i['num_threads']) for i in threadpool_info()]"
```

Regenerate the reference data (git-ignored, so it is not in the clone). Run from the repo root with 1 BLAS thread:
```powershell
$env:OPENBLAS_NUM_THREADS = "1"
python baseline/export_ref.py --config small  --steps 20
python baseline/export_ref.py --config medium --steps 20 --out c/ref_medium
python baseline/export_ref.py --config large  --steps 6  --out c/ref_large      # large is slow; 6 steps is enough
```

Build the C port natively. In the **MSYS2 UCRT64** shell:
```bash
cd "/c/NumPy Language Model Profiling/NumPy-Language-Model-Profiling"     # adjust the path to where you cloned
bash scripts/windows/build_tx_msys2.sh
```
That script installs gcc and OpenBLAS (`pacman`), builds `c/tx.exe`, `c/gemm_bench.exe`, `c/stream_triad.exe` with `-O3 -march=native -g -fno-omit-frame-pointer`, and runs `tx.exe check ref` (must print PASS). The `-g` is required so VTune can show functions and source lines. If source lines show as unknown, use function-level views and note it (guide step 4.4). Make sure `C:\msys64\ucrt64\bin` is on PATH when you launch `tx.exe` from the oneAPI prompt (OpenBLAS DLLs live there).

Then validate all three configs (every one must print PASS before anything is benchmarked):
```powershell
.\c\tx.exe check c/ref ;        .\c\tx.exe train c/ref
.\c\tx.exe check c/ref_medium ; .\c\tx.exe train c/ref_medium
.\c\tx.exe check c/ref_large  ; .\c\tx.exe train c/ref_large
```
Save the logs in `docs/logs/machineB_check.txt`.

## 6. Machine B baselines (so VTune data pairs with same-machine numbers)

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
python baseline/bench.py --config small  --reps 30 --warmup 3 --out baseline/results/machineB_py_small.json
python baseline/bench.py --config medium --reps 10 --warmup 2 --out baseline/results/machineB_py_medium.json
python baseline/bench.py --config large  --reps 5  --warmup 1 --out baseline/results/machineB_py_large.json
.\c\tx.exe bench c/ref 50        # save outputs to docs/logs/machineB_c_bench.txt
.\c\tx.exe bench c/ref_medium 10
.\c\tx.exe bench c/ref_large 5
.\c\gemm_bench.exe               # compute ceiling; run once with OPENBLAS_NUM_THREADS=1 and once with the variable removed
.\c\stream_triad.exe             # run 3 times, record the best (needs about 800 MB free RAM)
```
Repeat the small Python run three times: the medians must agree within a few percent, otherwise fix the cause (background load, power mode, thermals) first.
Record: Python step ms, C step ms, speedup, achieved GFLOP/s (FLOPs per step: small 369.4 MFLOP, medium 11,460.7, large 68,118.4), dgemm ceiling GFLOP/s, STREAM GB/s. Put them in `docs/machineB.md`.

## 7. VTune: self-check, then the runs

Use the **Intel oneAPI command prompt as Administrator**, from the repo root. Result folders go in `vtune/` (git-ignored). Run each analysis at least twice and check that the top five functions agree. Log every exact command in `docs/vtune/README.md`.

### 7.1 Self-check (step 4.3)
```
vtune-self-checker.bat      > docs\vtune\self_check.txt 2>&1
```
It takes several minutes and reports which analyses work. If it reports a missing sampling driver, rerun the VTune installer and choose the driver.

### 7.2 Run 1: Hotspots, Python (user-mode sampling)
```
vtune -collect hotspots -knob sampling-mode=sw -knob enable-stack-collection=true -result-dir vtune\py_hotspots -- python baseline\profile_steps.py --config small --steps 100
```
Summary: Top Hotspots, Effective CPU Utilization. Bottom-up grouped by Function: how much is Python frames, how much is BLAS, how much is NumPy elementwise loops.

### 7.3 Run 2: Hotspots, C (hardware sampling)
```
vtune -collect hotspots -knob sampling-mode=hw -knob enable-stack-collection=true -result-dir vtune\c_hotspots -search-dir .\c -source-search-dir .\c -- .\c\tx.exe bench c/ref 300
```
Add up the dgemm family share vs own functions (`ln_fwd`, `gelu_bwd`, `forward`, `backward`, `adam_step`). This is the key number; compare with Machine A (43.5% for small) and with the prediction.

### 7.4 Run 3: Microarchitecture Exploration
```
vtune -collect uarch-exploration -result-dir vtune\c_uarch -source-search-dir .\c -- .\c\tx.exe bench c/ref 300
```
Record the four top-down buckets, CPI, every highlighted (red) metric with its value, and the Memory Bound parts (L1, L2, L3, DRAM) vs Core Bound. In Bottom-up, add the bucket columns and sort by function: dgemm kernels vs elementwise loops.

### 7.5 Run 4: Memory Access (medium config)
```
vtune -collect memory-access -result-dir vtune\c_memory -source-search-dir .\c -- .\c\tx.exe bench c/ref_medium 30
```
Record DRAM Bandwidth peak and the Bandwidth Utilization Histogram. Compare the peak with Machine B's own stream_triad result. Close to the ceiling means bandwidth limited; far below means latency or compute limited.

### 7.6 Run 5: Threading (only if BLAS uses several threads)
```
$env:OPENBLAS_NUM_THREADS = "4"
vtune -collect threading -result-dir vtune\c_threading -- .\c\tx.exe bench c/ref_medium 20
```
Record Effective CPU Utilization, Spin Time, Wait Time, and the thread timeline. Optional: `vtune -collect hpc-performance -result-dir vtune\c_hpc -- .\c\tx.exe bench c/ref_medium 30`.

### 7.7 ITT markers (step 4.10)
Python:
```
vtune -collect hotspots -start-paused -knob sampling-mode=sw -result-dir vtune\py_itt -- python baseline\profile_steps_itt.py --config small --steps 100
```
Read Top Tasks for time per phase (forward, loss, backward, adam). C: build ittapi (`git clone https://github.com/intel/ittapi`, then cmake build as in the guide), `make itt ITT=../ittapi`, then `vtune -collect hotspots -knob sampling-mode=hw -result-dir vtune\c_itt -- .\c\tx_itt.exe bench c/ref 300`. Compare VTune's per-phase times with `tx bench`; differences are measurement overhead. ITT is optional if time is short.

### 7.8 Reports and screenshots (step 4.11 and 4.12)
```
mkdir docs\vtune
vtune -report summary   -result-dir vtune\c_hotspots > docs\vtune\c_hotspots_summary.txt
vtune -report hotspots  -result-dir vtune\c_hotspots -group-by function -format csv -csv-delimiter comma -report-output docs\vtune\c_hotspots_by_function.csv
vtune -report top-down  -result-dir vtune\c_hotspots -report-output docs\vtune\c_hotspots_topdown.txt
vtune -report summary   -result-dir vtune\c_uarch    > docs\vtune\c_uarch_summary.txt
vtune -report hw-events -result-dir vtune\c_uarch -format csv -report-output docs\vtune\c_uarch_events.csv
vtune -report summary   -result-dir vtune\c_memory   > docs\vtune\c_memory_summary.txt
vtune -report summary   -result-dir vtune\c_threading > docs\vtune\c_threading_summary.txt
vtune -report summary   -result-dir vtune\py_hotspots > docs\vtune\py_hotspots_summary.txt
vtune -help collect
```
Screenshots (GUI): Summary for every run; Bottom-up for runs 2 and 3; Platform view for the threading run; Source/Assembly of the hottest function. Save them in `docs/vtune/` with the run name in the file name. If a report command is refused, `vtune -help report` shows what that version supports.

## 8. What to record (paste these tables back into the chat with real values)

**Machine B record**
| Item | Value |
|---|---|
| CPU / cores / threads | |
| RAM | |
| OS, Python, NumPy, BLAS | |
| VTune version | |
| Power mode, AC | |
| dgemm ceiling GFLOP/s (1 thread / default) | |
| STREAM triad GB/s (best of 3) | |

**Baselines (Machine B)**
| Config | Python step ms | C step ms | Speedup | C GFLOP/s |
|---|---|---|---|---|
| small | | | | |
| medium | | | | |
| large | | | | |

**Run 2 (C hotspots): own vs dgemm**
| Component | CPU time % |
|---|---|
| dgemm family (OpenBLAS) | |
| gelu_bwd | |
| gelu_fwd | |
| ln_bwd / ln_fwd | |
| adam_step | |
| softmax / forward / backward own loops | |
| other | |

**Run 3 (top-down), whole program and per function**
| Item | Retiring | Bad Speculation | Front-End | Back-End: Memory | Back-End: Core | CPI |
|---|---|---|---|---|---|---|
| whole program | | | | | | |
| dgemm kernel | | | | | | |
| gelu_bwd | | | | | | |
| other hot functions | | | | | | |
Also list every highlighted metric with its value.

**Run 4 (memory):** DRAM bandwidth peak GB/s, % of STREAM ceiling, L3 and DRAM bound percentages.
**Run 5 (threading):** Effective CPU utilization, spin time, wait time, speedup observed.
**Run 1 (Python):** % Python frames / BLAS / NumPy elementwise; Top Tasks if ITT was used.

Repeatability: for each analysis note the top five functions of both runs and whether they agree.

## 9. Interpretation (guide step 4.13)

Fill the right-hand column from Machine B's own results:
| Observation | Meaning | Accelerator implication |
|---|---|---|
| Most time in dgemm kernels | matrix-multiply dominated, BLAS already tuned | GEMM engine is the natural kernel; beating OpenBLAS needs real parallelism and data reuse |
| Large share in own elementwise functions | leftover work is memory and math-library bound | fused streaming datapath; check arithmetic intensity first |
| Retiring high, Memory Bound low | compute limited | more parallel multipliers than a CPU; count DSP slices |
| Back-End Memory Bound high | waiting for data | on-chip buffering and reuse; plan block-RAM budget |
| Bandwidth near the STREAM ceiling | bandwidth limited | host-to-FPGA bandwidth becomes critical |
| Bad Speculation notable | branch-heavy (causal-mask loop?) | hardware masks without branches |
| Low CPU utilization, high spin/wait | threads do not help at this size | software baseline is effectively single threaded for small models |
| Python: mostly compiled frames | interpreter overhead small | do not count interpreter overhead as an accelerator benefit |

Then write: (a) where the Machine A prediction was right or wrong against Machine B hardware data, (b) the compute-bound vs memory-bound verdict per kernel (the question Machine A could not answer), and (c) which kernel Phase 1.3 should model on the roofline, with Machine B's ceilings.

## 10. Rules for this chat

- Label every table "Machine B (Samsung Galaxy Book4)". Machine A numbers are only quoted for comparison and clearly marked. The roofline in Phase 1.3 must use one machine's ceilings with that same machine's measurements.
- Never benchmark or profile a program that has not printed PASS.
- Profile a quiet, plugged-in machine; report ranges, not single numbers. If two runs differ a lot, check power mode and thermals.
- Long runs: Python `large` and any VTune run on `ref_large` are slow; use fewer steps and say so.
- Commit exports, reports and screenshots (`docs/vtune/`, `docs/machineB.md`, `docs/logs/machineB_*`), not the `vtune/` result folders. `.gitignore` already ignores `/vtune/` (anchored; an earlier unanchored pattern also hid `docs/vtune/`, now fixed), `.venv/`, `c/ref*/`, binaries.
- Push with `git push` from your own PowerShell. Claude sessions in this project could not reach GitHub, so commit locally and push yourself; add Machine B results in a new commit.
- When Stage 4 is finished, update the status table in `README.md` (Stage 4 row) and `docs/vtune/README.md`, then Stage 6 (consolidate, with Machine A and B shown side by side) can begin.

## 11. Gotchas learned on Machine A

- Machine A C build used OpenBLAS 0.3.15 taken from the OpenCV Python wheel (no `libopenblas-dev` was installable); NumPy used 0.3.29. On Machine B the MSYS2 OpenBLAS differs again; record both versions.
- The large reference export takes about 10 s per NumPy step; export only 6 steps.
- `line_profiler` on Python 3.10 needs the `tomli` package (already in the install line above).
- Drive-mounted repos on Machine A left stale `.git/*.lock` files after commits; if a git command says "File exists", delete the named `.lock` file only when no git process is running.
- VTune GUI: Hotspots with Hardware Event-Based Sampling greyed out or an "unsupported microarchitecture" message means the CPU generation is unsupported; do not keep retrying.
- gprof cannot see OpenBLAS or libm, so on a machine with working VTune prefer VTune's function view for the dgemm share.
