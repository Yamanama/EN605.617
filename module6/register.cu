#include <stdio.h>
#include <stdlib.h>

#define KERNEL_LOOP 1024
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
__host__ void wait_exit(void)
{
        char ch;

        printf("\nPress any key to exit");
        ch = getchar();
}

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
        for(unsigned int i=0; i < KERNEL_LOOP; i++)
        {
                host_data_ptr[i] = (unsigned int) rand();
        }
}

__global__ void test_gpu_register(unsigned int * const data, const unsigned int num_elements)
{
        const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if(tid < num_elements)
        {
                unsigned int d_tmp = data[tid];
                d_tmp = d_tmp * 2;
                data[tid] = d_tmp;
        }
}

__host__ void gpu_kernel(void)
{
        const unsigned int num_elements = KERNEL_LOOP;
        const unsigned int num_threads = KERNEL_LOOP;
        const unsigned int num_blocks = num_elements/num_threads;
        const unsigned int num_bytes = num_elements * sizeof(unsigned int);

        unsigned int * data_gpu;

        unsigned int host_packed_array[num_elements];
        unsigned int host_packed_array_output[num_elements];
        cudaEvent_t start, stop;
        // setup timer
        setupTimer(&start, &stop);
        cudaMalloc(&data_gpu, num_bytes);

        generate_rand_data(host_packed_array);

        cudaMemcpy(data_gpu, host_packed_array, num_bytes,cudaMemcpyHostToDevice);
        cudaEventRecord(start, 0);
        test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements);
        logTime(start, stop, "Time: ");
        cudaThreadSynchronize();        // Wait for the GPU launched work to complete
        cudaGetLastError();

        cudaMemcpy(host_packed_array_output, data_gpu, num_bytes,cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_elements; i++){
                printf("Input value: %x, device output: %x\n",host_packed_array[i], host_packed_array_output[i]);
        }

        cudaFree((void* ) data_gpu);
        cudaDeviceReset();
        wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	gpu_kernel();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return EXIT_SUCCESS;
}
