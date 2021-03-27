
#include "common.cu"
#include "kernels.cu"

/**
 * Allocate the pinnable memory for fast access
 *
 * vectors - the vectors array
 */
 void allocate(Vector vectors[]) {
    // compute sizes
    for (int i = 0; i < VECTORS; i++) {
        cudaMalloc((void**)&vectors[i].device, VECTOR_SIZE * sizeof(int));
        cudaMallocHost((void**)&vectors[i].host, VECTOR_SIZE * sizeof(int));
    }
}

/**
 * Cleanup memory
 *
 * vectors - the vectors array
 */
void freeMemory(Vector vectors[]) {
    for (int i = 0; i < VECTORS; i++) {
        cudaFreeHost(vectors[i].host);
        cudaFree(vectors[i].device);
    }
}
/**
 * Main entrypoint
 */
int main(int argc, char** argv) {
    Vector vectors[VECTORS];
    // print device info
    printDevice();
    // allocate memory
    allocate(vectors);
    printf("Pinned\n");
    // generate inputs
    generate(vectors);
    // perform operations
    perform(vectors);
    // print
    // printVectors(vectors);
    // free
    freeMemory(vectors);
}




