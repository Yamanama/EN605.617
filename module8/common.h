#include <stdio.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>


#define A_ROWS 1
#define A_COLUMNS 2
#define B_COLUMNS 3
#define B_ROWS A_COLUMNS
#define C_COLUMNS B_COLUMNS
#define C_ROWS A_ROWS

#define DOT_LENGTH 100

typedef struct {
    float * host;
    float * device;
} Matrix;

typedef struct {
    double * vectorOne;
    double * vectorTwo;
    double * result;
} Dot;
