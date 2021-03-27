// #include <curand.h>
// #include <curand_kernel.h>
// #include <cublas_v2.h>

// #include "common.h"

#define RANDOM_MAX 4

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
 void randoms(curandState_t* states, int* result)
 {
     const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
     result[thread_idx] = curand(&states[thread_idx]) % RANDOM_MAX;
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
    printf("      %8s: %f\n", message, elapsed);
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
 * Generate Sequence
 *
 * vector - the vector
 * message - the message
 */
void generateSequence(Vector vector, const char* message) {
    printf("    %s:\n", message);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // thrust pointer
    thrust::device_ptr<int> sequenceOne(vector.device);
    // generate
    cudaEventRecord(start, 0);
    thrust::sequence(sequenceOne, sequenceOne + VECTOR_SIZE);
    logTime(start, stop, "Generation");
    cudaEventRecord(start, 0);
    thrust::copy(sequenceOne, sequenceOne + VECTOR_SIZE, vector.host);
    logTime(start, stop, "Transfer");
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
 void generateRandom(Vector vector, const char* message) 
{
    // markers
    cudaEvent_t start, stop;
    printf("    %s:\n", message);
    // setup timer
    setupTimer(&start, &stop);
    // states
    curandState_t* states;
    cudaMalloc((void**)&states, VECTOR_SIZE * sizeof(curandState_t));
    // mark time
    cudaEventRecord(start, 0);
    // execute the sequence kernel
    initRandoms<<<VECTOR_SIZE, 1>>>(time(0), states);
    randoms<<<VECTOR_SIZE, 1>>>(states, vector.device);
    // log time
    logTime(start, stop, "Generation");
    cudaEventRecord(start, 0);
    // store generated sequences
    cudaMemcpy(vector.host, vector.device, VECTOR_SIZE * sizeof(int),
               cudaMemcpyDeviceToHost);
    // log time
    logTime(start, stop, "Transfer");
    // clean timer
    cleanTimer(start, stop);
}
/**
 * Generate vectors
 * 
 * vectors - array of vectors
 */
void generate(Vector vectors[]) {
    printf("  Generation:\n");
    generateSequence(vectors[0], "Sequence");
    generateRandom(vectors[1], "Random");
}
/**
 * Modulus operation
 *
 * one - first vector
 * two - second vector
 * result - the results vector
 */
void modulus(Vector one, Vector two, Vector result) {
    printf("    Product:\n");
    // make device pointers
    thrust::device_ptr<int> first(one.device);
    thrust::device_ptr<int> second(two.device);
    thrust::device_ptr<int> third(result.device);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // operate
    cudaEventRecord(start, 0);
    thrust::transform(first, first + VECTOR_SIZE, second, third, thrust::modulus<int>());
    logTime(start, stop, "Operation");
    // transfer
    cudaEventRecord(start, 0);
    thrust::copy(third, third + VECTOR_SIZE, result.host);
    logTime(start, stop, "Transfer");
    // clean
    cleanTimer(start, stop);
}
/**
 * Product operation
 *
 * one - first vector
 * two - second vector
 * result - the results vector
 */
void product(Vector one, Vector two, Vector result) {
    printf("    Product:\n");
    // make device pointers
    thrust::device_ptr<int> first(one.device);
    thrust::device_ptr<int> second(two.device);
    thrust::device_ptr<int> third(result.device);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // operate
    cudaEventRecord(start, 0);
    thrust::transform(first, first + VECTOR_SIZE, second, third, thrust::multiplies<int>());
    logTime(start, stop, "Operation");
    // transfer
    cudaEventRecord(start, 0);
    thrust::copy(third, third + VECTOR_SIZE, result.host);
    logTime(start, stop, "Transfer");
    // clean
    cleanTimer(start, stop);
}
/**
 * Subtract operation
 *
 * one - first vector
 * two - second vector
 * result - the results vector
 */
void subtract(Vector one, Vector two, Vector result) {
    printf("    Subtraction:\n");
    // make device pointers
    thrust::device_ptr<int> first(one.device);
    thrust::device_ptr<int> second(two.device);
    thrust::device_ptr<int> third(result.device);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // operate
    cudaEventRecord(start, 0);
    thrust::transform(first, first + VECTOR_SIZE, second, third, thrust::minus<int>());
    logTime(start, stop, "Operation");
    // transfer
    cudaEventRecord(start, 0);
    thrust::copy(third, third + VECTOR_SIZE, result.host);
    logTime(start, stop, "Transfer");
    // clean
    cleanTimer(start, stop);
}
/**
 * Add operation
 *
 * one - first vector
 * two - second vector
 * result - the results vector
 */
void add(Vector one, Vector two, Vector result) {
    printf("    Addition:\n");
    // make device pointers
    thrust::device_ptr<int> first(one.device);
    thrust::device_ptr<int> second(two.device);
    thrust::device_ptr<int> third(result.device);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // operate
    cudaEventRecord(start, 0);
    thrust::transform(first, first + VECTOR_SIZE, second, third, thrust::plus<int>());
    logTime(start, stop, "Operation");
    // transfer
    cudaEventRecord(start, 0);
    thrust::copy(third, third + VECTOR_SIZE, result.host);
    logTime(start, stop, "Transfer");
    // clean
    cleanTimer(start, stop);
}
/**
 * Perform operations
 *
 * vectors - array of vectors
 */
void perform(Vector vectors[]) {
    printf("  Operations:\n");
    add(vectors[0], vectors[1], vectors[2]);
    subtract(vectors[0], vectors[1], vectors[3]);
    product(vectors[0], vectors[1], vectors[4]);
    modulus(vectors[0], vectors[1], vectors[5]);
}