/* Minimal STREAM-style triad: a[i] = b[i] + s*c[i]. Gives a practical DRAM bandwidth ceiling for the roofline.
 * gcc -O3 -march=native -o stream_triad stream_triad.c && ./stream_triad */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int main(void){
    size_t n = 32u*1024*1024;                     /* 3 arrays x 256 MB = 768 MB, far larger than any cache */
    double *a = malloc(n*8), *b = malloc(n*8), *c = malloc(n*8);
    if (!a || !b || !c) { fprintf(stderr, "not enough memory, lower n\n"); return 1; }
    for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1.0; c[i] = 2.0; }
    double best = 1e9;
    for (int r = 0; r < 10; r++) {
        double t0 = now();
        for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0*c[i];
        double dt = now() - t0; if (dt < best) best = dt;
    }
    printf("triad best %.3f ms  ->  %.2f GB/s (24 bytes moved per element)\n", best*1e3, 24.0*n/best/1e9);
    return (int)a[n/2] == 7 ? 0 : 1;
}
