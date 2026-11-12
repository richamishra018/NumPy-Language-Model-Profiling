#!/bin/bash
# Build the C port natively on Windows for VTune (guide step 3.1 / 3.6 / 4.4).
# Run inside the "MSYS2 UCRT64" shell (NOT the plain MSYS shell):
#   cd "/d/NumPy Language Model Profiling/numpy-lm-profiling"
#   bash scripts/windows/build_tx_msys2.sh
set -e
pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-openblas make
gcc --version | head -1
cd c
# MSYS2 OpenBLAS >= 0.3.2x installs cblas.h under include/openblas
INC=""
for d in /ucrt64/include/openblas /ucrt64/include; do [ -f "$d/cblas.h" ] && INC="-I$d" && break; done
echo "cblas include flag: $INC"
# -g gives DWARF debug info for VTune; -fno-omit-frame-pointer for call stacks (guide 4.4)
gcc -O3 -march=native -g -fno-omit-frame-pointer -Wall -Wextra $INC -o tx.exe tx.c -lopenblas -lm
gcc -O3 -march=native $INC -o gemm_bench.exe gemm_bench.c -lopenblas -lm
gcc -O3 -march=native -o stream_triad.exe stream_triad.c
echo "--- smoke test (must print PASS) ---"
# NOTE: tx.exe needs the reference folder c/ref (already exists in this folder) and the OpenBLAS DLL on PATH (UCRT64 shell has it).
./tx.exe check ref
./tx.exe train ref | tail -2
./tx.exe bench ref 20 | grep "train step"
echo "Built c/tx.exe, c/gemm_bench.exe, c/stream_triad.exe"
