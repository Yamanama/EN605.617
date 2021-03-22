#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>


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
 * Generate a random matrix on the device
 *
 * array - the array
 * rows - the rows
 * columns - the columns
 */
void generateRandomMatrix(Matrix array, int rows, int columns) {
    // generator
    curandGenerator_t numberGenerator;
    curandCreateGenerator(&numberGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // generate
    cudaEventRecord(start, 0);
    curandSetPseudoRandomGeneratorSeed(numberGenerator, 
                                       (unsigned long long) clock());
    curandGenerateUniform(numberGenerator, array.device, rows * columns);
    logTime(start, stop, "  Random Array Generation");
    cudaEventRecord(start, 0);
    // transfer
    cudaMemcpy(array.host,
        array.device,
        rows * columns * sizeof(float),
        cudaMemcpyDeviceToHost);
    logTime(start, stop, "  Transfer to host");
    // clean up timers
    cleanTimer(start, stop);
}

/**
 * Generate a random vector on the device
 *
 * array - the array
 * rows - the rows
 * columns - the columns
 */
 void generateRandomVectors(Dot dot, int size) {
    // generator
    curandGenerator_t numberGenerator;
    curandCreateGenerator(&numberGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // generate
    cudaEventRecord(start, 0);
    curandSetPseudoRandomGeneratorSeed(numberGenerator, 
                                       (unsigned long long) clock());
    curandGenerateUniform(numberGenerator, (float*)dot.vectorOne, size);
    curandGenerateUniform(numberGenerator, (float*)dot.vectorTwo, size);
    logTime(start, stop, "  Random Vectors Generation");
    // clean up timers
    cleanTimer(start, stop);
}

/**
 * Compute a dot product of two vectors
 * 
 * dot - the dot structure
 */
void computeDotProduct(Dot dot) {
    // timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // handler
    cublasHandle_t handler;
    cublasCreate(&handler);
    // calculate
    cudaEventRecord(start, 0);
    cublasDdot(handler, DOT_LENGTH, dot.vectorOne, 1, dot.vectorTwo, 1, dot.result);
    logTime(start, stop, "  Random Vectors Generation");
    double result;
    cudaMemcpy(&result, dot.result, sizeof(double), cudaMemcpyDeviceToHost);
    // clean up timers
    printf("  Dot Product Result: %f\n", result);
    cleanTimer(start, stop);
    cublasDestroy(handler);
}

/**
 * Perform addition
 *
 * results - results structure
 * device - device structure
 * options - cli structure
 * message - log message
 */
 void perform(Matrix A, Matrix B, Matrix C, const char * message) {
    printf("%s\n  Computing Product:\n", message);
    // setup timer
    cudaEvent_t start, stop;
    setupTimer(&start, &stop);
    // temps
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // handler
    cublasHandle_t handle;
    cublasCreate(&handle);
    // calculate
    cudaEventRecord(start, 0);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                A_ROWS, B_COLUMNS, A_COLUMNS, &alpha, 
                A.device, A_ROWS, B.device, A_COLUMNS, &beta, 
                C.device, A_ROWS);
    logTime(start, stop, "  Matrices Product Calculation");
    cudaEventRecord(start, 0);
    // transfer
    cudaMemcpy(C.host,
               C.device,
               C_ROWS * C_COLUMNS * sizeof(float),
               cudaMemcpyDeviceToHost);
    logTime(start, stop, "  Transfer to host");
    cublasDestroy(handle);
 }