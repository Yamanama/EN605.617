#include "caesar_cipher.h"

/**
 * Setup the cuda timers
 *
 * start - the start marker
 * stop - the stop marker
 */
void setupTimer(cudaEvent_t* start, cudaEvent_t* stop){
    cudaEventCreate(start);
    cudaEventCreate(stop);
}
/**
 * Setup the cuda timers
 *
 * start - the start marker
 * stop - the stop marker
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
 * Setup the cuda timers
 *
 * start - the start marker
 * stop - the stop marker
 */
void cleanTimer(cudaEvent_t start, cudaEvent_t stop){
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
/**
 * Kernel to modulus two arrays and put the remainder in a third array
 */
__global__
void caesarCipher(char * result, int offset) 
{
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    char d_tmp = (result[thread_idx] + offset) % ASCII_MAX;
    result[thread_idx] = d_tmp;
}
/**
 * Parse CLI arguments and store in an Options Structure
 *
 * argc - argument count
 * argv - argument variables
 * options - options structure
 */
void parseArgs(unsigned int argc, char** argv, Options* options){
    // set defaults
    options->filename = "cipher.txt";
    options->offset = 3;
    // first arg is file name
    if (argc >= 2) {
        options->filename = argv[1];
    }
    // second arg cipher offset
    if (argc >= 3) {
        options->offset = atoi(argv[2]);
    }
}
/**
 * Allocate the pinnable memory for fast access
 *
 * results - the pinned results
 * device - the device memory structure
 * size - the array size
 */
void allocate(Results* cipher, Results* decrypt, Device* device, unsigned int size) {
    cudaMalloc((void **)&device->block, size);
    cudaMalloc((void **)&device->encrypted, size);
    cudaMalloc((void **)&device->decrypted, size);

    cudaMallocHost((void**)&cipher->output, size);
    cudaMallocHost((void**)&cipher->input, size);
    cudaMallocHost((void**)&decrypt->output, size);
    cudaMallocHost((void**)&decrypt->input, size);
    
}
/**
 * Allocate the pageable memory for fast access
 *
 * results - the pinned results
 * device - the device memory structure
 * size - the array size
 */
void allocatePageable(Results* cipher, Results* decrypt, Device* device, unsigned int size) {
    cudaMalloc((void **)&device->block, size);
    cudaMalloc((void **)&device->encrypted, size);
    cudaMalloc((void **)&device->decrypted, size);

    cipher->output = (char*)malloc(size);
    cipher->input = (char*)malloc(size);
    decrypt->output = (char*)malloc(size);
    decrypt->input = (char*)malloc(size);
    
}
/**
 * Perform the encryption
 *
 * cipher - the string to translate structure
 * device - the device memory structure
 * size - the array size
 * offset - the cipher offset
 */
void translate(Results * cipher, Device * device, int size, int offset) {
    cudaEvent_t start, stop;
    // setup timer
    setupTimer(&start, &stop);
    // transfer in
    cudaEventRecord(start, 0);
    cudaMemcpy(device->block, cipher->input, size, cudaMemcpyHostToDevice);
    logTime(start, stop, TRANSFER_STRING);
    // execute kernel
    cudaEventRecord(start, 0);
    caesarCipher<<<64,64>>>(device->block, offset);
    logTime(start, stop, KERNEL_STRING);
    // wait for the work to be done
    cudaDeviceSynchronize();
    // check errors
    cudaGetLastError();
    // transfer out
    cudaEventRecord(start, 0);
    // copy it out
    cudaMemcpy(cipher->output,
        device->block,
        size,
        cudaMemcpyDeviceToHost);
    logTime(start, stop, TRANSFER_STRING);
    // clean up the timer
    cleanTimer(start, stop);
}
/**
 * Get the message from a file
 *
 * Options - the CLI options
 */
const char * getMessage(Options options) {
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    // open the file
    fp = fopen(options.filename, "r");  
    if (fp == NULL){
        printf("Couldn't find file %s\n", options.filename);
        exit(EXIT_FAILURE);
    }
    // read the first line
    getline(&line, &len, fp);
    // close the file
    fclose(fp);
    return line;
}
/**
 * Clean up the pinned memory allocations
 *
 * encrypted - results structure for encryption
 * decrypted - results structure for decryption
 * device - device structure
 */
void freePinned(Results encrypted, Results decrypted, Device device) {
    // free host side
    cudaFreeHost(encrypted.input);
    cudaFreeHost(encrypted.output);
    cudaFreeHost(decrypted.input);
    cudaFreeHost(decrypted.output);
    // free device side
    cudaFree(device.encrypted);
    cudaFree(device.decrypted);
    cudaFree(device.block);
}
/**
 * Execute pinned strategy
 *
 * message - the message to translatee
 * options - the cli options
 */
void execute_pinned(const char * message, Options options) {
    Results encrypted;
    Results decrypted;
    Device device;
    // store the string length
    encrypted.length = strlen(message);
    // calculate array size
    unsigned int size = sizeof(char) * encrypted.length;
    printf("  Pinned:\n");
    // allocate
    allocate(&encrypted, &decrypted, &device, size);
    memcpy(encrypted.input, message, sizeof(char) * encrypted.length);
    // perform encryption
    translate(&encrypted, &device, size, options.offset);
    printf("   Encrypted: %s\n", encrypted.output);
    // swap
    decrypted.input = encrypted.output;
    // perform decryption
    translate(&decrypted, &device, size, -options.offset);
    printf("   Decrypted: %s\n", decrypted.output);
    // free memory
    freePinned(encrypted, decrypted, device);
}
/**
 * Clean up the pageable memory allocations
 *
 * encrypted - results structure for encryption
 * decrypted - results structure for decryption
 * device - device structure
 */
void freePageable(Results encrypted, Results decrypted, Device device) {
    free(encrypted.input);
    free(encrypted.output);
    cudaFree(device.encrypted);
    cudaFree(device.decrypted);
    cudaFree(device.block);
}
/**
 * Execute pageable strategy
 *
 * message - the message to translatee
 * options - the cli options
 */
void execute_pageable(const char * message, Options options) {
    Results encrypted;
    Results decrypted;
    Device device;
    // store length
    encrypted.length = strlen(message);
    unsigned int size = sizeof(char) * encrypted.length;
    printf("  Pageable:\n");
    // allocate
    allocatePageable(&encrypted, &decrypted, &device, size);
    memcpy(encrypted.input, message, sizeof(char) * encrypted.length);
    // perform encryption
    translate(&encrypted, &device, size, options.offset);
    printf("   Encrypted: %s\n", encrypted.output);
    // swap
    decrypted.input = encrypted.output;
    // perform decryption
    translate(&decrypted, &device, size, -options.offset);
    printf("   Decrypted: %s\n", decrypted.output);
    // free memory
    freePageable(encrypted, decrypted, device);
}
/**
 * Main entrypoint
 */
int main(int argc, char** argv) {
    Options options;
    // parse cli args
    parseArgs(argc, argv, &options);
    // get the message from file
    const char * message = getMessage(options);
    printf("Message: %s\n", message);
    // execute pageable
    execute_pageable(message, options);
    // execute pinned
    execute_pinned(message, options);
    // reset
    cudaDeviceReset();
    // exit
    exit(EXIT_SUCCESS);
}