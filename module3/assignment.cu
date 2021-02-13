
#include "assignment.h"
#include "kernels.cu"
/**
 * Structure to contain derived settings from CLI input
 */
typedef struct {
    unsigned int totalThreads;
    unsigned int blockSize;
    unsigned int totalBlocks;
    unsigned int arraySize;
} Options;
/**
 * Structure to hold cuda allocations
 */
typedef struct {
    unsigned int * block;
    unsigned int * sequence;
    unsigned int * random;
    curandState_t* states;
} Device;
/**
 * Structure to results from GPU operations
 */
typedef struct {
    unsigned int * sequence;
    unsigned int * random;
    unsigned int * sum;
    unsigned int * difference;
    unsigned int * product;
    unsigned int * modulus;
} Results;
/**
 * Parse CLI arguments and store in an Options Structure
 *
 * argc - argument count
 * argv - argument variables
 * options - options structure
 */
void parseArgs(unsigned int argc, char** argv, Options* options){
    options->totalThreads = (1 << 20);
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
}
/**
 * Allocate necessary space for all calculations
 *
 * results - results structure
 * device - device structure
 * options - options structure
 */
void allocate(Results* results, Device* device, Options options){
    // array size
    unsigned int size = (sizeof(unsigned int) * options.totalThreads);
    // allocate the device structure
    cudaMalloc((void**)&device->states,
                options.totalThreads * sizeof(curandState_t));
    cudaMalloc((void **)&device->block, size);
    cudaMalloc((void **)&device->sequence, size);
    cudaMalloc((void **)&device->random, size);
    // allocate storage for all results
    results->sequence = (unsigned int*)calloc(sizeof(unsigned int),
                         options.totalThreads);
    results->random = (unsigned int*)calloc(sizeof(unsigned int),
                         options.totalThreads);
    results->sum = (unsigned int*)calloc(sizeof(unsigned int),
                         options.totalThreads);
    results->difference = (unsigned int*)calloc(sizeof(unsigned int),
                         options.totalThreads);
    results->product = (unsigned int*)calloc(sizeof(unsigned int),
                         options.totalThreads);
    results->modulus = (unsigned int*)calloc(sizeof(unsigned int),
                         options.totalThreads);
}
/**
 * Clean up allocations
 *
 * device - the device structure to clean up
 */
void clean(Device device) {
    cudaFree(device.states);
    cudaFree(device.block);
    cudaFree(device.sequence);
    cudaFree(device.random);
}
/**
 * Generate both required arrays
 *
 * results - results structure
 * device - device structure
 * options - options structure
 */
void generate(Results* results, Device device, Options options) {
    // array sizes
    unsigned int size = (sizeof(unsigned int) * options.totalThreads);

    // execute the sequence kernel
    sequence<<<options.totalBlocks, options.blockSize>>>(device.block);
    // store generated sequences
    cudaMemcpy(results->sequence,
               device.block,
               size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(device.sequence,
               device.block,
               size,
               cudaMemcpyDeviceToHost);
    // execute the random kernels
    initRandoms<<<options.totalThreads, 1>>>(time(0), device.states);
    randoms<<<options.totalThreads, 1>>>(device.states, device.block);
    // store generated randoms
    cudaMemcpy(results->random,
               device.block,
               size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(device.random,
               device.block,
               size,
               cudaMemcpyDeviceToHost);
}
/**
 * Calculate all relevant metrics
 *
 * results - results structure
 * device - device structure
 * options - options structure
 */
void calc(Results* results, Device device, Options options) {
    // array size
    unsigned int size = (sizeof(unsigned int) * options.totalThreads);
    // execute add kernel and store
    add<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                    device.sequence,
                                                    device.random);
    cudaMemcpy(results->sum, device.block, size, cudaMemcpyDeviceToHost);
    // execute the subtract kernel and store
    subtract<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                         device.sequence, device.random);
    cudaMemcpy(results->difference, device.block, size, cudaMemcpyDeviceToHost);
    // execute the mult kernel and store
    mult<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                     device.sequence,
                                                     device.random);
    cudaMemcpy(results->product, device.block, size, cudaMemcpyDeviceToHost);
    // execute mod kernel and store
    mod<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                    device.sequence,
                                                    device.random);
    cudaMemcpy(results->modulus, device.block, size, cudaMemcpyDeviceToHost);
}
/**
 * Print results
 *
 * results - results structure
 * options - options structure
 */
void print(Results results, Options options) {
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
/**
 * Main entrypoint
 */
int main(int argc, char** argv)
{
    // structures
    Options options;
    Device device;
    Results results;
    // parse CLI arguements and store
    parseArgs(argc, argv, &options);
    // allocate memory
    allocate(&results, &device, options);
    // generate source arrays
    generate(&results, device, options);
    // perform calculations
    calc(&results, device, options);
    // clean up
    clean(device);
    // print results
    print(results, options);
    // end
    return EXIT_SUCCESS;
}
