#include <stdio.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

#define CONSTANT_MAX 512

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

void parseArgs(unsigned int argc, char** argv, Options* options);
void allocate(Results* results, Device* device, Options options);
void clean(Device device);
void generate(Results* results, Device device, Options options);
void calc(Results* results, Device device, Options options);
void print(Results results, Options options);

