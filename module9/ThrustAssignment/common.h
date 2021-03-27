#include <stdio.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/iterator/counting_iterator.h>

#define VECTORS 6
#define VECTOR_SIZE 100

typedef struct {
    int * host;
    int * device;
} Vector;