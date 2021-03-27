#include "common.h"

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

void printVectors(Vector vectors[]) {
    for (int i =0; i < VECTOR_SIZE; i++) {
        printf("%d ", *(vectors[0].host + i));
        printf("%d ", *(vectors[1].host + i));
        printf("%d ", *(vectors[2].host + i));
        printf("%d ", *(vectors[3].host + i));
        printf("%d ", *(vectors[4].host + i));
        printf("%d\n", *(vectors[5].host + i));
    }
}