#include <stdio.h> 
#include <stdlib.h>

/* 
 * The maximum and minimum integer values of the range of printable characters 
 * in the ASCII alphabet. Used by encrypt kernel to wrap adjust values to that 
 * ciphertext is always printable. 
 */ 
#define MAX_PRINTABLE 64 
#define MIN_PRINTABLE 91
#define NUM_ALPHA MAX_PRINTABLE - MIN_PRINTABLE
#define STRING_LENGTH 14

__global__ void encrypt(char *text, char *key, char* result) { /* Calculate the current index */ const unsigned int
	idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	 /* 
	  * Adjust value of text and key to be based at 0 
	  * Printable ASCII starts at MIN_PRINTABLE, but 0 start is easier to work with 
	  */ 
	char adjusted_text = text[idx] - MIN_PRINTABLE; 
	char adjusted_key = key[idx] - MIN_PRINTABLE;

	 /* The cipher character is the text char added to the key char modulo the number of chars in the alphabet*/ 
	char cipherchar = (adjusted_text + adjusted_key) % (NUM_ALPHA);

	 /* adjust back to normal ascii (starting at MIN_PRINTABLE) and save to result */ 
	result[idx] = (char)cipherchar + MIN_PRINTABLE ;
}

cudaEvent_t get_time() {
	cudaEvent_t marker;
	cudaEventCreate(&marker);
	cudaEventRecord(marker);
	return marker;
}

void print_encryption_results(char* cpu_text, 
							  char* cpu_key, 
							  char* cpu_result,
							  int array_size) 
{
	for (int i = 0; i<STRING_LENGTH; i++){
		printf("%c", *(cpu_text + i));
	}
	printf("\n");
	for (int i = 0; i<STRING_LENGTH; i++){
		printf("%c", *(cpu_key + i));
	}
	printf("\n");
	for (int i = 0; i<STRING_LENGTH; i++){
		printf("%c", *(cpu_result + i));
	}
	printf("\n");
}

void pageable_transfer_execution(int array_size, int threads_per_block, const char* input_fp, const char* key_fp) { 
	/* Calculate the size of the array */
	int array_size_in_bytes = (sizeof(char) * strlen(input_fp)); 

	 char *cpu_text = (char *) malloc(array_size_in_bytes); 
	 char *cpu_key = (char *) malloc(array_size_in_bytes); 
	 char *cpu_result = (char *) malloc(array_size_in_bytes);
	 memcpy(cpu_text, input_fp, array_size_in_bytes);
	 memcpy(cpu_key, key_fp, array_size_in_bytes);
	 /* Read characters from the input and key files into the text and key arrays respectively */ 
	 char *gpu_text;
	 char *gpu_key;
	 char *gpu_result;


	 cudaMalloc((void **)&gpu_text, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_key, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_result, array_size_in_bytes);

	 /* Copy the CPU memory to the GPU memory */ 
	 cudaMemcpy( gpu_text, cpu_text, array_size_in_bytes, cudaMemcpyHostToDevice); 
	 cudaMemcpy( gpu_key, cpu_key, array_size_in_bytes, cudaMemcpyHostToDevice);

	 /* Designate the number of blocks and threads */ 
	 const unsigned int num_blocks = array_size/threads_per_block; 
	 const unsigned int num_threads = array_size/num_blocks;

	 /* Execute the encryption kernel and keep track of start and end time for duration */ 
	 float duration = 0; 
	 cudaEvent_t start_time = get_time();

	 encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);

	 cudaEvent_t end_time = get_time(); 
	 cudaEventSynchronize(end_time); 
	 cudaEventElapsedTime(&duration, start_time, end_time);

	 /* Copy the changed GPU memory back to the CPU */ 
	 cudaMemcpy( cpu_result, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

	 printf("Pageable Transfer- Duration: %fmsn\n", duration); 
	 print_encryption_results(cpu_text, cpu_key, cpu_result, array_size);

	 /* Free the GPU memory */ 
	 cudaFree(gpu_text);
	 cudaFree(gpu_key);
	 cudaFree(gpu_result);

	 /* Free the CPU memory */ 
	 free(cpu_text);
	 free(cpu_key);
	 free(cpu_result);

}

void pinned_transfer_execution(int array_size, int threads_per_block, const char *input_fp, const char *key_fp) { 
	int array_size_in_bytes = (sizeof(char) * (array_size)); 


	char *cpu_text_pinned;
	char *cpu_key_pinned;
	char *cpu_result_pinned;
	//pin it 
	cudaMallocHost((void **)&cpu_text_pinned, array_size_in_bytes); cudaMallocHost((void **)&cpu_key_pinned, array_size_in_bytes);
	cudaMallocHost((void **)&cpu_result_pinned, array_size_in_bytes);
	memcpy(cpu_text_pinned, input_fp, array_size_in_bytes);
	memcpy(cpu_key_pinned, key_fp, array_size_in_bytes);
	 /* Copy the memory over */ 
	

	 /* Declare and allocate pointers for GPU based parameters */ 
	char *gpu_text; 
	char *gpu_key; 
	char *gpu_result;

	cudaMalloc((void **)&gpu_text, array_size_in_bytes); 
	cudaMalloc((void **)&gpu_key, array_size_in_bytes); 
	cudaMalloc((void **)&gpu_result, array_size_in_bytes);
	

	/* Copy the CPU memory to the GPU memory */ 
	cudaMemcpy( gpu_text, cpu_text_pinned, array_size_in_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy( gpu_key, cpu_key_pinned, array_size_in_bytes, cudaMemcpyHostToDevice);

	 /* Designate the number of blocks and threads */ 
	const unsigned int num_blocks = array_size/threads_per_block; 
	const unsigned int num_threads = array_size/num_blocks;

	 /* Execute the encryption kernel and keep track of start and end time for duration */ 
	float duration = 0; 
	cudaEvent_t start_time = get_time();

	 encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);

	 cudaEvent_t end_time = get_time(); 
	 cudaEventSynchronize(end_time); 
	 cudaEventElapsedTime(&duration, start_time, end_time);

	 /* Copy the changed GPU memory back to the CPU */ 
	 cudaMemcpy( cpu_result_pinned, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

	 printf("Pinned Transfer- Duration: %fmsn\n", duration); 
	 print_encryption_results(cpu_text_pinned, cpu_key_pinned, cpu_result_pinned, array_size);

	 /* Free the GPU memory */ 
	cudaFree(gpu_text); 
	cudaFree(gpu_key); 
	cudaFree(gpu_result);

	 /* Free the pinned CPU memory */ 
	cudaFreeHost(cpu_text_pinned); 
	cudaFreeHost(cpu_key_pinned); 
	cudaFreeHost(cpu_result_pinned);

	 /* Free the pageable CPU memory */ 
	// INSERT CODE HERE 
}

/** * Prints the correct usage of this file * @name is the name of the executable (argv[0]) */ 
void print_usage(char *name) {
	printf("Usage: %s <total_num_threads> <threads_per_block> <input_file> <key_file>\n", name); 
}

/** 
  * Performs simple setup functions before calling the pageable_transfer_execution() 
  * function. * Makes sure the files are valid, handles opening and closing of file pointers. 
  */ void pageable_transfer(int num_threads, int threads_per_block, char *input_file, char *key_file) { 

	const char * input_fp = "THIS IS A TEST";
	const char * key_fp =   "NOTENOTENOTENO";

 	/* Perform the pageable transfer */ 
	pageable_transfer_execution(num_threads, threads_per_block, input_fp, key_fp);

 	
}

/** 
  * Performs setup functions before calling the pageable_transfer_execution() 
  * function. 
  * Makes sure the files are valid, handles opening and closing of file pointers. 
  */ void pinned_transfer(int num_threads, int threads_per_block, char *input_file, char *key_file) {

	const char * input_fp = "THIS IS A TEST";
	const char * key_fp =   "NOTENOTENOTENO";

    /* Perform the pageable transfer */ 
    pinned_transfer_execution(num_threads, threads_per_block, input_fp, key_fp);

 	
}

	/** 
	  * Entry point for excution. Checks command line arguments and 
	  * opens input files, then passes execution to subordinate main_sub() 
	  */
int main(int argc, char *argv[]) { 
	/* Check the number of arguments, print usage if wrong 
	*/ 
	if(argc != 5) { 
		printf("Error: Incorrect number of command line arguments\n"); 
		print_usage(argv[0]); exit(-1); 
	}

 	/* Check the values for num_threads and threads_per_block */ 
	int num_threads = atoi(argv[1]); 
	int threads_per_block = atoi(argv[2]);
	if(num_threads <= 0 || threads_per_block <= 0) { 
		printf("Error: num_threads and threads_per_block must be integer > 0");
		print_usage(argv[0]); exit(-1); 
	}

 	if(threads_per_block > num_threads) { 
		printf("Error: threads per block is greater than number of threads\n"); 
		print_usage(argv[0]);
		exit(-1); 
	}

 	printf("\n"); 
	/* Perform the pageable transfer */ 
	pageable_transfer(num_threads, threads_per_block, argv[3], argv[4]);

 	printf("-----------------------------------------------------------------\n");

 	/* Perform the pinned transfer */ 
	pinned_transfer(num_threads, threads_per_block, argv[3], argv[4]);

 	return EXIT_SUCCESS; 
}
