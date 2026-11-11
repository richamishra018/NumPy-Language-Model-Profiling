/* Smallest possible OpenBLAS test: C = A x B for 2x2 row-major matrices.  gcc hello_dgemm.c -o hello_dgemm -lopenblas */
#include <cblas.h>
#include <stdio.h>
int main(void) {
    double A[4] = {1, 2, 3, 4}, B[4] = {5, 6, 7, 8}, C[4] = {0, 0, 0, 0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
    printf("C = [%g %g; %g %g]   (expected [19 22; 43 50])\n", C[0], C[1], C[2], C[3]);
    printf("%s\n", openblas_get_config());
    return 0;
}
