# Machine record

Recorded 2026-11-11 (Stage 0.5). Measured inside the Linux environment that runs on the Windows laptop "richanotebook".

## CPU
- Model: Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz (Ivy Bridge, family 6 model 58)
- Cores / threads: 1 core visible, 2 logical CPUs (2 threads per core). Note: the Linux guest exposes 2 CPUs; the physical i5-3210M has 2 cores / 4 threads.
- Clock: 2.5 GHz nominal (reported 2494 MHz)
- Caches: L1d 32 KiB, L1i 32 KiB, L2 256 KiB, L3 3 MiB
- SIMD: AVX (no AVX2, no FMA) -> peak FP64 per core = 8 FLOP/cycle (4-wide AVX add + 4-wide mul)
- Vendor: GenuineIntel (VTune hardware sampling possible only on native hardware; see caveat)

## Memory
- RAM visible to the environment: 3.8 GiB total, no swap

## OS / software
- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0 (Microsoft hypervisor guest, full virtualization)
- Python 3.10.12, NumPy 2.2.6 (bundled scipy-openblas, OpenBLAS 0.3.29, Haswell kernels selected via DYNAMIC_ARCH), matplotlib 3.10.9, psutil 7.2.2
- Virtualenv: .venv created with system site-packages (PyPI not reachable from this shell, so line_profiler, snakeviz, py-spy, memory_profiler, threadpoolctl and ittapi are NOT yet installed)

## Power state
- Power mode: NOT RECORDED (not visible from the guest). Richa to fill in: Windows power mode ______
- On AC power: NOT RECORDED. Richa to fill in: yes / no ______

## Caveats
- This is a VM guest (Hyper-V/WSL2-style, "Hypervisor vendor: Microsoft"). Per the guide, hardware counters are normally NOT exposed here, so run VTune (Stage 4) on native Windows, or use Stage 5 tools. Python/C timings from here carry some virtualization overhead; note this when comparing against teammates.
- Cache sizes, core count and the 2-CPU view come from the guest and may differ from the physical chip.
- NumPy BLAS picked Haswell kernels (AVX2) even though the CPU reports no AVX2 flag; worth checking during Stage 3.

## Smoke test (Stage 0.6)
`python src/train.py --config small --steps 50 --log_every 25`: 4654 characters, vocab size **54**, loss 5.3385 (step 1) -> 2.7806 (step 25) -> 2.4876 (step 50), wall time ~17 s. checkpoint.pkl and loss_curve.png deleted afterwards.
