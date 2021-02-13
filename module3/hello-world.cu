// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>

__global__ 
void hello(int * block)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	printf("%2u %2u %2u %2u\n",threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
	block[thread_idx] = threadIdx.x;
}

void main_sub(int n, int block_size)
{
	unsigned int cpu_block[n];
	int num_blocks = n / block_size;
	int array_size_in_bytes = (sizeof(unsigned int) * n);
	printf("******** N:%2u Block Size:%2u Number of Blocks: %2u **********\n", n, block_size, num_blocks);
	
	/* Declare pointers for GPU based params */
	int *gpu_block;

	cudaMalloc((void **)&gpu_block, array_size_in_bytes);
	cudaMemcpy( gpu_block, cpu_block, array_size_in_bytes, cudaMemcpyHostToDevice );

	/* Execute our kernel */
	hello<<<num_blocks, block_size>>>(gpu_block);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, array_size_in_bytes, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);

	/* Iterate through the arrays and print */
	for(unsigned int i = 0; i < n; i++)
	{
		printf("Calculated Thread: - Block: %2u\n",cpu_block[i]);
	}
}

int main()
{
	main_sub(16,16);
	main_sub(128,32);
	main_sub(32,128);
	// main_sub(1280,128);
	main_sub(13, 7);
	return EXIT_SUCCESS;
}

