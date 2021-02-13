#include <curand.h>
#include <curand_kernel.h>

// RANDOM_MAX ensures 0 <= result < 4 to meet the requirement
#define RANDOM_MAX 4
/**
 * Kernel to initial curand states
 * Derived from https://docs.nvidia.com/cuda/curand
 */
__global__
void initRandoms(unsigned int seed, curandState_t* states)
{
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}
/**
 * Kernel to initialize randoms using curand states 
 * Derived from https://docs.nvidia.com/cuda/curand
 */
__global__
void randoms(curandState_t* states, unsigned int* result)
{
    result[blockIdx.x] = curand(&states[blockIdx.x]) % RANDOM_MAX;
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
    result[thread_idx] = sequence[thread_idx] + random[thread_idx];
}
/**
 * Kernel to subtract two arrays and put the differences in a third array
 */
__global__
void subtract(unsigned int * result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[thread_idx] = sequence[thread_idx] - random[thread_idx];
}
/**
 * Kernel to multiply two arrays and put the product in a third array
 */
__global__
void mult(unsigned int * result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[thread_idx] = sequence[thread_idx] * random[thread_idx];
}
/**
 * Kernel to modulus two arrays and put the remainder in a third array
 */
__global__
void mod(unsigned int * result, unsigned int* sequence, unsigned int* random)
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    result[thread_idx] = sequence[thread_idx] % random[thread_idx];
}
