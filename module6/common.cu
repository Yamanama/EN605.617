#include "common.h"

/**
 * Parse CLI arguments and store in an Options Structure
 *
 * argc - argument count
 * argv - argument variables
 * options - options structure
 */
 void parseArgs(unsigned int argc, char** argv, Options* options){
    options->totalThreads = 1<<20;
    options->blockSize = 256;
    // first arg is the total thread count
    if (argc >= 2) {
        options->totalThreads = atoi(argv[1]);
    }
    // second arg is the block size
    if (argc >= 3) {
        options->blockSize = atoi(argv[2]);
    }
    // calc total blocks
    options->totalBlocks = options->totalThreads/options->blockSize;
    // validate and optimize
    if (options->totalThreads % options->blockSize != 0) {
        ++options->totalBlocks;
        options->totalThreads = options->totalBlocks*options->blockSize;
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", options->totalThreads);
    }
    options->arraySize = (sizeof(unsigned int) * options->totalThreads);
}
/**
 *  Print Device Information
 */
void printDevice() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("  Clock Rate: %d\n", prop.clockRate);
    printf("  Warp Size: %d\n", prop.warpSize);
}

/**
 * Print results
 *
 * results - results structure
 * options - options structure
 */
 void printResults(Results results, Options options) {
    // iterate results and print
    for(unsigned int i = 0; i < options.totalThreads; i++) {
        printf("Count %3u ", *(results.sequence + i));
        printf("Random: %1u ", *(results.random + i));
        printf("Sum: %3u ", *(results.sum + i));
        printf("Difference: %3d ", *(results.difference + i));
        printf("Product: %5u ", *(results.product + i));
        printf("Modulus: %3d\n", *(results.modulus + i));
    }
}