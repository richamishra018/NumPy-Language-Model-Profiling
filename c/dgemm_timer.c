/* LD_PRELOAD timer: wraps cblas_dgemm, accumulates time, call count and FLOPs, prints a summary at exit.
   Stage 5 substitute for the VTune "dgemm family vs own code" split on machines without hardware counters.
   Build: gcc -O2 -shared -fPIC -o dgemm_timer.so dgemm_timer.c -ldl
   Run:   LD_PRELOAD=./dgemm_timer.so ./tx bench ref 30                                                    */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <time.h>
typedef void (*dgemm_fn)(int, int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
static dgemm_fn real; static double tns; static long calls; static double flops;
static struct timespec t0;
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
__attribute__((constructor)) static void init(void) { clock_gettime(CLOCK_MONOTONIC, &t0); }
void cblas_dgemm(int o, int ta, int tb, int M, int N, int K, double al, const double *A, int lda, const double *B, int ldb, double be, double *C, int ldc) {
    if (!real) real = (dgemm_fn)dlsym(RTLD_NEXT, "cblas_dgemm");
    double s = now_ns(); real(o, ta, tb, M, N, K, al, A, lda, B, ldb, be, C, ldc); tns += now_ns() - s;
    calls++; flops += 2.0 * M * N * K;
}
__attribute__((destructor)) static void fini(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    double wall = (t.tv_sec - t0.tv_sec) * 1e9 + (t.tv_nsec - t0.tv_nsec);
    fprintf(stderr, "DGEMM_TIMER calls=%ld dgemm_s=%.3f wall_s=%.3f share=%.1f%% dgemm_GFLOPs=%.2f avg_us=%.1f\n",
            calls, tns / 1e9, wall / 1e9, 100.0 * tns / wall, flops / tns, tns / 1e3 / calls);
}
