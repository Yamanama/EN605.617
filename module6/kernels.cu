#include <curand.h>
#include <curand_kernel.h>

// RANDOM_MAX ensures 0 <= result < 4 to meet the requirement
#define RANDOM_MAX 4
#define KERNEL_STRING "    Kernel"
#define TRANSFER_STRING "    Transfer"
/**
 * Kernel to initial curand states
 * Derived from https://docs.nvidia.com/cuda/curand
 */
__global__
void initRandoms(unsigned int seed, curandState_t* states)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init(seed, thread_idx, 0, &states[thread_idx]);
}
/**
 * Kernel to initialize randoms using curand states 
 * Derived from https://docs.nvidia.com/cuda/curand
 */
__global__
void randoms(curandState_t* states, unsigned int* result)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int random = curand(&states[thread_idx]) % RANDOM_MAX; 
    result[thread_idx] = random;
}
/**
 * Kernel to generate a sequence of numbers
 */
__global__
void sequence(unsigned int* result)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[thread_idx] = thread_idx;
}
/**
 * Kernel to add two arrays and put the sums in a third array
 */
__global__
void add(unsigned int* result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int sum = sequence[thread_idx] + random[thread_idx];
    result[thread_idx] = sum;
}
/**
 * Kernel to subtract two arrays and put the differences in a third array
 */
__global__
void subtract(unsigned int * result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int difference = sequence[thread_idx] - random[thread_idx];
    result[thread_idx] = difference;
}
/**
 * Kernel to multiply two arrays and put the product in a third array
 */
__global__
void mult(unsigned int * result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int product = sequence[thread_idx] * random[thread_idx];
    result[thread_idx] = product;
}
/**
 * Kernel to modulus two arrays and put the remainder in a third array
 */
__global__
void mod(unsigned int * result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int modulus = sequence[thread_idx] % random[thread_idx]; 
    result[thread_idx] = modulus;
}
/**
 * Setup Timer
 *
 * start - start marker
 * stop - stop marker
 */
void setupTimer(cudaEvent_t* start, cudaEvent_t* stop){
    cudaEventCreate(start);
    cudaEventCreate(stop);
}
/**
 * Log the computed time
 * 
 * start - start marker
 * stop - stop marker
 * message - log message
 */
void logTime(cudaEvent_t start, cudaEvent_t stop, const char* message){
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  %8s: %f\n", message, elapsed);
}
/**
 *  Clean up memory for timers
 *
 * start - start marker
 * stop - stop marker
 */
void cleanTimer(cudaEvent_t start, cudaEvent_t stop){
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
/**
 * Generate Sequence Array
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 * message - log message
 */
void generateSequence(Results* results, 
                      Device device,
                      Options options, 
                      const char* message) 
{
    // markers
    cudaEvent_t start, stop;
    printf("%sSequence:\n", message);
    // setup timer
    setupTimer(&start, &stop);
    // mart start
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    sequence<<<options.totalBlocks, options.blockSize>>>(device.block);
    // log time
    logTime(start, stop, KERNEL_STRING);
    // sync
    cudaDeviceSynchronize();
    // check error
    cudaGetLastError();
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(results->sequence,
               device.block,
               options.arraySize,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(device.sequence,
               device.block,
               options.arraySize,
               cudaMemcpyDeviceToHost);
    //log time
    logTime(start, stop, TRANSFER_STRING);
    //cleanup timer
    cleanTimer(start, stop);
}
/**
 * Generate Random Array
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 * message - log message
 */
void generateRandom(Results* results, 
                    Device device, 
                    Options options, 
                    const char* message) 
{
    // markers
    cudaEvent_t start, stop;
    printf("%sRandom:\n", message);
    // setup timer
    setupTimer(&start, &stop);
    cudaMalloc((void**)&device.states,
                options.totalThreads * sizeof(curandState_t));
    // mark time
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    initRandoms<<<options.totalBlocks, options.blockSize>>>(time(0), device.states);
    randoms<<<options.totalBlocks, options.blockSize>>>(device.states, device.block);
    // log time
    logTime(start, stop, KERNEL_STRING);
    // sync
    cudaDeviceSynchronize();
    // check error
    cudaGetLastError();
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(results->random,
               device.block,
               options.arraySize,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(device.random,
               device.block,
               options.arraySize,
               cudaMemcpyDeviceToHost);
    // log time
    logTime(start, stop, TRANSFER_STRING);
    // clean timer
    cleanTimer(start, stop);
}
/**
 * Perform addition
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 */
void performAdd(Results* results, Device device, Options options)
{
    // markers
    cudaEvent_t start, stop;
    printf("  Adding\n");
    // set up timer
    setupTimer(&start, &stop);
    // mark time
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    add<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                    device.sequence,
                                                    device.random);
    // log time
    logTime(start, stop, KERNEL_STRING);
    // sync
    cudaDeviceSynchronize();
    // check error
    cudaGetLastError();
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(results->sum, 
               device.block, 
               options.arraySize,
               cudaMemcpyDeviceToHost);
    // log time
    logTime(start, stop, TRANSFER_STRING);
    // clean up
    cleanTimer(start, stop);
}
/**
 * Perform subtraction
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 */
void performSubtract(Results* results, Device device, Options options)
{
    // markers
    cudaEvent_t start, stop;
    printf("  Subtracting\n");
    setupTimer(&start, &stop);
    // mark time
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    subtract<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                    device.sequence,
                                                    device.random);
    // log time
    logTime(start, stop, KERNEL_STRING);
    // sync
    cudaDeviceSynchronize();
    // check error
    cudaGetLastError();
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(results->difference, 
               device.block, 
               options.arraySize,
               cudaMemcpyDeviceToHost);
    // log time
    logTime(start, stop, TRANSFER_STRING);
    // clean up
    cleanTimer(start, stop);
}
/**
 * Perform modulus
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 */
void performModulus(Results* results, Device device, Options options)
{
    // marker
    cudaEvent_t start, stop;
    printf("  Modulus\n");
    setupTimer(&start, &stop);
    // mark time
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    mod<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                    device.sequence,
                                                    device.random);
    // log time
    logTime(start, stop, KERNEL_STRING);
    // sync
    cudaDeviceSynchronize();
    // check error
    cudaGetLastError();
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(results->modulus, 
               device.block, 
               options.arraySize,
               cudaMemcpyDeviceToHost);
    // log time
    logTime(start, stop, TRANSFER_STRING);
    // clean up
    cleanTimer(start, stop);
}
/**
 * Perform multiple
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 */
void performMult(Results* results, Device device, Options options)
{
    // array sizes
    cudaEvent_t start, stop;
    printf("  Multiplying\n");
    setupTimer(&start, &stop);
    // mark time
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    mult<<<options.totalBlocks, options.blockSize>>>(device.block,
                                                    device.sequence,
                                                    device.random);
    // log time
    logTime(start, stop, KERNEL_STRING);
    // sync
    cudaDeviceSynchronize();
    // check error
    cudaGetLastError();
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(results->product, 
               device.block, 
               options.arraySize,
               cudaMemcpyDeviceToHost);
    // log time
    logTime(start, stop, TRANSFER_STRING);
    // clean up timers
    cleanTimer(start, stop);
}
/**
 * Perform addition
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 * message - log message
 */
void perform(Results results, 
             Device device, 
             Options options, 
             const char * message)
{
    printf("%s\n  Generate:\n", message);
    // generate
    generateSequence(&results, device, options, "    ");
    generateRandom(&results, device, options, "    ");
    // perform
    performAdd(&results, device, options);
    performSubtract(&results, device, options);
    performMult(&results, device, options);
    performModulus(&results, device, options);
}
