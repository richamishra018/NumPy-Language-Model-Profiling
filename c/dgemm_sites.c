/* LD_PRELOAD timer, per call site: records time/calls per (return address, M, N, K) for cblas_dgemm and prints
   them at exit as "SITE offset M N K calls ns". Offsets are relative to the main binary (resolve with addr2line).
   Build: gcc -O2 -shared -fPIC -o dgemm_sites.so dgemm_sites.c -ldl      Used by baseline/dgemm_sites_report.py */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
typedef void (*fn_t)(int, int, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
typedef struct { unsigned long off; int M, N, K; long calls; double ns; } Rec;
static Rec recs[256]; static int nrec; static fn_t real;
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
void cblas_dgemm(int o, int ta, int tb, int M, int N, int K, double al, const double *A, int lda, const double *B, int ldb, double be, double *C, int ldc) {
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "cblas_dgemm");
    void *ra = __builtin_return_address(0); Dl_info di; unsigned long off = 0;
    if (dladdr(ra, &di)) off = (unsigned long)ra - (unsigned long)di.dli_fbase;
    double s = now_ns(); real(o, ta, tb, M, N, K, al, A, lda, B, ldb, be, C, ldc); double d = now_ns() - s;
    for (int i = 0; i < nrec; i++) if (recs[i].off == off && recs[i].M == M && recs[i].N == N && recs[i].K == K) { recs[i].calls++; recs[i].ns += d; return; }
    if (nrec < 256) { recs[nrec].off = off; recs[nrec].M = M; recs[nrec].N = N; recs[nrec].K = K; recs[nrec].calls = 1; recs[nrec].ns = d; nrec++; }
}
__attribute__((destructor)) static void fini(void) {
    for (int i = 0; i < nrec; i++) fprintf(stderr, "SITE %lx %d %d %d %ld %.0f\n", recs[i].off, recs[i].M, recs[i].N, recs[i].K, recs[i].calls, recs[i].ns);
}
