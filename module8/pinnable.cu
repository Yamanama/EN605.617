
#include "common.cu"
#include "kernels.cu"
/**
 * Allocate the pinnable memory for fast access
 *
 * A - The first matrix
 * B - The second matrix
 * C - The product matrix
 */
void allocate(Matrix* A, Matrix* B, Matrix* C, Dot* dot) {
    // compute sizes
    unsigned int aSize = A_ROWS * A_COLUMNS * sizeof(float);
    unsigned int bSize = B_ROWS * B_COLUMNS * sizeof(float);
    unsigned int cSize = C_ROWS * C_COLUMNS * sizeof(float);
    unsigned int dotSize = sizeof(double) * (size_t)DOT_LENGTH;
    // device allocate
    cudaMalloc((void**)&A->device, aSize);
    cudaMalloc((void**)&B->device, bSize);
    cudaMalloc((void**)&C->device, cSize);
    cudaMalloc((void**)&dot->vectorOne, dotSize);
    cudaMalloc((void**)&dot->vectorTwo, dotSize);
    cudaMalloc((void**)&dot->result, sizeof(double));
    // host allocate
    cudaMallocHost((void**)&A->host, aSize);
    cudaMallocHost((void**)&B->host, bSize);
    cudaMallocHost((void**)&C->host, cSize);
    
}

/**
 * Cleanup memory
 *
 * A - the first matrix
 * B - the second matrix
 * C - the product matrix
 * dot - the dot product structure 
 */
 void freeMemory(Matrix A, Matrix B, Matrix C, Dot dot) {
    // free host
    cudaFreeHost(A.host);
    cudaFreeHost(B.host);
    cudaFreeHost(C.host);
    // free device
    cudaFree(A.device);
    cudaFree(B.device);
    cudaFree(C.device);
    cudaFree(dot.vectorOne);
    cudaFree(dot.vectorTwo);
    cudaFree(dot.result);
}

/**
 * Main entrypoint
 */
int main(int argc, char** argv) {
    Matrix A, B, C;
    Dot dot;
    // print device info
    printDevice();
    // allocate memory
    allocate(&A, &B, &C, &dot);
    // generate
    printf("Pinnable\n");
    printf("  A %dx%d Array Generation:\n", A_ROWS, A_COLUMNS);
    generateRandomMatrix(A, A_ROWS, A_COLUMNS);
    printf("  B %dx%d Array Generation:\n", B_ROWS, B_COLUMNS);
    generateRandomMatrix(B, B_ROWS, B_COLUMNS);
    // calculate
    perform(A, B, C, "Executing");
    // print
    print_matrices(A, B, C);
    // calculate a dot product
    printf("  Dot Product:\n");
    generateRandomVectors(dot, DOT_LENGTH);
    computeDotProduct(dot);
    // free
    freeMemory(A, B, C, dot);
}




