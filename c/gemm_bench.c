/* Measures what OpenBLAS dgemm actually achieves on this machine (the practical compute ceiling).
 * gcc -O3 -march=native -o gemm_bench gemm_bench.c -lopenblas -lm && ./gemm_bench */
#define _POSIX_C_SOURCE 200809L
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int main(void){
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    printf("OpenBLAS: %s | threads %d\n", openblas_get_config(), openblas_get_num_threads());
    for (int s = 0; s < 6; s++) {
        int n = sizes[s]; size_t e = (size_t)n*n;
        double *a = malloc(e*8), *b = malloc(e*8), *c = malloc(e*8);
        for (size_t i = 0; i < e; i++) { a[i] = (i%7)*0.1; b[i] = (i%5)*0.2; c[i] = 0; }
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,1.0,a,n,b,n,0.0,c,n);          /* warm-up */
        int reps = n <= 256 ? 200 : (n <= 1024 ? 10 : 3); double best = 1e9;
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,1.0,a,n,b,n,0.0,c,n);
            double dt = now() - t0; if (dt < best) best = dt;
        }
        printf("N=%5d  best %9.3f ms  %8.2f GFLOP/s\n", n, best*1e3, 2.0*n*n*n/best/1e9);
        free(a); free(b); free(c);
    }
    return 0;
}
