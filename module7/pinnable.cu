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
    cudaMalloc((void**)&device->states,
                options.totalThreads * sizeof(curandState_t));
    cudaMalloc((void **)&device->block, size);
    cudaMalloc((void **)&device->sequence, size);
    cudaMalloc((void **)&device->random, size);

    cudaMallocHost((void**)&results->sequence, options.arraySize);
    cudaMallocHost((void**)&results->random, options.arraySize);
    cudaMallocHost((void**)&results->sum, options.arraySize);
    cudaMallocHost((void**)&results->product, options.arraySize);
    cudaMallocHost((void**)&results->difference, options.arraySize);
    cudaMallocHost((void**)&results->modulus, options.arraySize);
}

/**
 * Cleanup memory
 *
 * results - the results
 * device - the device memory structure 
 */
 void freeMemory(Results results, Device device) {
    // free host
    cudaFreeHost(results.sequence);
    cudaFreeHost(results.random);
    cudaFreeHost(results.sum);
    cudaFreeHost(results.difference);
    cudaFreeHost(results.product);
    cudaFreeHost(results.modulus);
    // free device
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
    // allocate memory
    allocate(&results, &device, options);
    perform(results, device, options, "Pinnable");
    // printResults(results, options);
    freeMemory(results, device);
}




