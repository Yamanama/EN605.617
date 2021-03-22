#include "common.h"
// Derived from cublas_example.cu Module 8 activity code
#define index(i,j,ld) (((j)*(ld))+(i))
/**
 *  Print Device Information
 */
void printDevice() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("  Clock Rate: %d\n", prop.clockRate);
    printf("  Warp Size: %d\n\n", prop.warpSize);
}

/**
 * Print a matrix
 * Derived from cublas_example.cu Module 8 Activity code
 *
 * P - the matrix
 * uHP - the height
 * uWP - the width
 * message - the printable header message
 */
void print_matrix(float*P,int columns,int rows, const char * message) {
    int i,j;
    printf("%s\n", message);
    for(i=0;i<rows;i++){
        printf("\n");
        for(j=0;j<columns;j++)
            printf(" %f",P[index(i,j,rows)]);
    }
    printf("\n");
}

void print_matrices(Matrix A, Matrix B, Matrix product) {
    print_matrix(A.host, A_COLUMNS, A_ROWS, "A:");
    print_matrix(B.host, B_COLUMNS, B_ROWS, "B:");
    print_matrix(product.host, C_COLUMNS, C_ROWS, "Product:");
}