#include <stdio.h>

// #define ARRAY_SIZE 256
// #define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

// /* Declare  statically two arrays of ARRAY_SIZE each */
// unsigned int cpu_block[ARRAY_SIZE];
// unsigned int cpu_thread[ARRAY_SIZE];


__global__
void what_is_my_id(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub0(unsigned int array_size)
{
	printf("****** %4u *******\n", array_size);
	unsigned int cpu_block[array_size];
	unsigned int cpu_thread[array_size];
	unsigned int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	cudaMalloc((void **)&gpu_block, array_size_in_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_in_bytes);
	cudaMemcpy( cpu_block, gpu_block, array_size_in_bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( cpu_thread, gpu_thread, array_size_in_bytes, cudaMemcpyHostToDevice );

	const unsigned int num_blocks = array_size/16;
	const unsigned int num_threads = array_size/num_blocks;

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, array_size_in_bytes, cudaMemcpyDeviceToHost );
	cudaMemcpy( cpu_thread, gpu_thread, array_size_in_bytes, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < array_size; i++)
	{
		printf("Thread: %2u - Block: %2u\n",cpu_thread[i],cpu_block[i]);
	}
}

int main()
{
	main_sub0(256);
	main_sub0(128);
	main_sub0(13);
	main_sub0(3);
	main_sub0(32);
	return EXIT_SUCCESS;
}
