#include "common.cu"
#include "kernels.cu"
/**
 * Allocate the pinnable memory for fast access
 *
 * results - the pinned results
 * device - the device memory structure
 * options - cli options 
 */
void allocate(Results* results, Device* device, Options options) {
    unsigned int size = (sizeof(unsigned int) * options.totalThreads);
    // allocate device memory
    cudaMalloc((void**)&device->states,
                options.totalThreads * sizeof(curandState_t));
    cudaMalloc((void **)&device->block, size);
    cudaMalloc((void **)&device->sequence, size);
    cudaMalloc((void **)&device->random, size);
    // allocate pageable memory
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
 * Cleanup memory
 *
 * results - the results
 * device - the device memory structure 
 */
void freeMemory(Results results, Device device) {
    free(results.sequence);
    free(results.random);
    free(results.sum);
    free(results.difference);
    free(results.product);
    free(results.modulus);
    cudaFree(device.states);
    cudaFree(device.block);
    cudaFree(device.sequence);
    cudaFree(device.random);
}
/**
 * Main entrypoint
 */
int main(int argc, char** argv) {
    Options options;
    Results results;
    Device device;
    // parse cli args
    parseArgs(argc, argv, &options);
    // print device info
    printDevice();
    // allocate
    allocate(&results, &device, options);
    // perform pageable strategy
    perform(results, device, options, "Pageable");
    // printResults(results, options);
    freeMemory(results, device);
}




