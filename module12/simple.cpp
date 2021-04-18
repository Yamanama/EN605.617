//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define NUM_BUFFER_ELEMENTS 16
const unsigned int width = 2;
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * Create a command queue
 * From Module10/HelloWorld.cpp
 */
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

/**
 * Stopwatch for time keeping
 */
struct Stopwatch {
	clock_t begin;
	clock_t end;
	/**
	 * Constructor
	 */
	Stopwatch() {}
	/**
	 * Start the watch
	 */
	void start() {
		this->begin = clock();
	}
	/**
	 * Stop the watch
	 *
	 * message - the log message
	 */
	void stop(const char * message) {
		this->end = clock();
		double duration = double(this->end - this->begin) / double(CLOCKS_PER_SEC);
		std::cout << message << std::fixed << duration << std::setprecision(5);
		std::cout << std::endl;
	}
	/**
	 * Stop then start the watch
	 *
	 * message - the log message
	 */
	void lap(const char * message) {
		this->stop(message);
		this->start();
	}
};

/**
 * Platform information
 */
struct Platform {
	cl_int errNum;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
	cl_context context;
	cl_uint i;
	/**
	 * Constructor
	 */
	Platform() {
		// get platforms
		this->getPlatforms();
		// setup devices
		this->getDevices();
	}
	/**
	 * Get available platforms
	 */
	void getPlatforms() {
		// get number of platforms
		this->errNum = clGetPlatformIDs(0, NULL, &this->numPlatforms);
		checkErr( 
			(this->errNum != CL_SUCCESS) ? this->errNum : 
			(this->numPlatforms <= 0 ? -1 : CL_SUCCESS), 
			"clGetPlatformIDs"); 
		this->deviceIDs = NULL;
		// allocate
		this->platformIDs = (cl_platform_id *)alloca(
				sizeof(cl_platform_id) * this->numPlatforms);
		// get platform ids
		this->errNum = clGetPlatformIDs(this->numPlatforms, this->platformIDs, NULL);
		checkErr( 
			(this->errNum != CL_SUCCESS) ? this->errNum : 
			(this->numPlatforms <= 0 ? -1 : CL_SUCCESS)
			, "clGetPlatformIDs");
		}
	/**
	 * Setup the devices
	 */
	void getDevices() {
		cl_uint i;
		// iterate platforms
		for (this->i = 0; i < this->numPlatforms; this->i++) {
			// get devices
			this->errNum = clGetDeviceIDs(platformIDs[this->i], CL_DEVICE_TYPE_ALL, 0, NULL, &this->numDevices);
			// check
			if (this->errNum != CL_SUCCESS && 
				this->errNum != CL_DEVICE_NOT_FOUND) {
				checkErr(this->errNum, "clGetDeviceIDs");
			}
			else if (this->numDevices > 0) 
			{
				// allocate
				this->deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * this->numDevices);
				// get device ids
				this->errNum = clGetDeviceIDs(
					this->platformIDs[this->i], CL_DEVICE_TYPE_ALL,
					this->numDevices, &this->deviceIDs[0], NULL);
				// check
				checkErr(this->errNum, "clGetDeviceIDs");
				break;
			}
		}
		// context properties
		cl_context_properties contextProperties[] = {
			CL_CONTEXT_PLATFORM, 
			(cl_context_properties)this->platformIDs[this->i], 0 };
		// setup context
		this->context = clCreateContext(
			contextProperties, this->numDevices, 
			this->deviceIDs, NULL,
			NULL, &this->errNum);
		// check
		checkErr(this->errNum, "clCreateContext");
	}
};
/**
 * Program
 */
struct Program {
	Platform* platform;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	/**
	 * Constructor
	 *
	 * platform - the platform
	 */
	Program(Platform* platform) {
		this->platform = platform;
		// initialize
		this->init();
		// build program
		this->build();
	}
	/**
	 * Initialize
	 */
	void init() {
		cl_int errNum;
		// get the src file
		std::ifstream srcFile("simple.cl");
		checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");
		// use it
		std::string srcProg(
			std::istreambuf_iterator<char>(srcFile),
			(std::istreambuf_iterator<char>()));
		// book keeping
		const char * src = srcProg.c_str();
		size_t length = srcProg.length();
		// Create program from source
		this->program = clCreateProgramWithSource(
			this->platform->context, 
			1, 
			&src, 
			&length, 
			&errNum);
		// check
		checkErr(errNum, "clCreateProgramWithSource");
	}
	/**
	 * Build
	 */
	void build() {
		cl_int errNum;
		// build program
		errNum = clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
		// check
		if (errNum != CL_SUCCESS)
		{
			// Determine the reason for the error
			char buildLog[16384];
			clGetProgramBuildInfo(
				this->program, 
				this->platform->deviceIDs[0], 
				CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), 
				buildLog, 
				NULL);
			std::cerr << "Error in kernel: " << std::endl;
			std::cerr << buildLog;
			checkErr(errNum, "clBuildProgram");
		}
	}
};
/**
 * Buffers Structure
 */
struct Buffers {
    int * inputOutput;
    cl_mem_flags flag;
    Program * program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    /**
     * Constructor
     *
     * program - the program
     */
    Buffers (Program * program, cl_mem_flags flag) {
        this->flag = flag;
        this->program = program;
        // create the buffer
        this->createBuffer();
        // create the sub buffers
        this->createSubBuffers();
        // create the command queues
        this->createCommandQueues();
       
    }
    /**
     * Create the buffer
     */
    void createBuffer() {
        cl_int errNum;
        // size
        int size = NUM_BUFFER_ELEMENTS * this->program->platform->numDevices;
        // i/o
        this->inputOutput = new int[size];
        // iterate
        for (unsigned int i = 0; i < size; i++) {
            inputOutput[i] = i;
        }
        // create the buffer
        cl_mem buffer = clCreateBuffer(
            this->program->platform->context,
            this->flag,
            sizeof(int) * size,
            NULL,
            &errNum);
        // check it
        checkErr(errNum, "clCreateBuffer");
        // push it
        this->buffers.push_back(buffer);
    }
    /**
     * Create sub buffers
     */
    void createSubBuffers() {
        cl_int errNum;
        // iterate
        for (unsigned int i = 1; i < this->program->platform->numDevices; i++)
        {
            // region
            cl_buffer_region region = 
                {
                    width * i * sizeof(int), 
                    width * sizeof(int)
                };
            // make a sub buffer
            cl_mem buffer = clCreateSubBuffer(
                this->buffers[0],
                CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &errNum);
            // check it
            checkErr(errNum, "clCreateSubBuffer");
            // push it
            this->buffers.push_back(buffer);
        }
    }
    /**
     * Create Command Queues
     */
    void createCommandQueues() {
        cl_int errNum;
        // iterate
        for (unsigned int i = 0; i < this->program->platform->numDevices; i++)
        {
            cl_device_id device;
            // create the command queue
            cl_command_queue queue = CreateCommandQueue(            
                this->program->platform->context, 
                &device);
            // push it
            this->queues.push_back(queue);
            // create kernel
            cl_kernel kernel = clCreateKernel(
                this->program->program,
                "average",
                &errNum);
            // check it
            checkErr(errNum, "clCreateKernel(average)");
            // set it
            errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&this->buffers[i]);
            errNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), (void*)&width);
            // check it
            checkErr(errNum, "clSetKernelArg(average)");
            // push it
            this->kernels.push_back(kernel);
        }
    }
    /**
     * Enqueue the buffers
     */
    void enqueue() {
        int size = NUM_BUFFER_ELEMENTS * this->program->platform->numDevices;
        cl_int errNum = clEnqueueWriteBuffer(
            this->queues[0],
            this->buffers[0],
            CL_TRUE,
            0,
            sizeof(int) * size,
            (void*)this->inputOutput,
            0,
            NULL,
            NULL);
    }
    /**
     * Perform Operation
     */
    void doit() {
        cl_int errNum;
        std::vector<cl_event> events;
        // iterate
        for (unsigned int i = 0; i < this->queues.size(); i++)
        {
            cl_event event;
            size_t gWI = NUM_BUFFER_ELEMENTS;
            // perform
            errNum = clEnqueueNDRangeKernel(
                this->queues[i], 
                this->kernels[i], 
                1, 
                NULL,
                (const size_t*)&gWI, 
                (const size_t*)NULL, 
                0, 
                0, 
                &event);
            // push it
            events.push_back(event);
        }
        // wait
        clWaitForEvents(events.size(), &events[0]);
    }
    /**
     * Read from buffer
     */
    void read() {
        int size = NUM_BUFFER_ELEMENTS * this->program->platform->numDevices;
        clEnqueueReadBuffer(
            this->queues[0],
            this->buffers[0],
            CL_TRUE,
            0,
            sizeof(int) * size,
            (void*)this->inputOutput,
            0,
            NULL,
            NULL);
    }
    /**
     * Print helper
     */
    void print() {
        // iterate and print
        for (unsigned i = 0; i < this->program->platform->numDevices; i++) {
            for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i+1) * NUM_BUFFER_ELEMENTS); elems++)
            {
                std::cout << " " << this->inputOutput[elems];
            }
            std::cout << std::endl;
        }
    }
};
/** 
 * Execute the operation
 */
void execute(cl_mem_flags flag) {
    // make stopwatch
    Stopwatch * stopwatch = new Stopwatch();
	stopwatch->start();
	// make a new platform
	Platform *platform = new Platform();
	stopwatch->lap("  Platform: ");
    // make a new program
	Program *program = new Program(platform);
    stopwatch->lap("  Program: ");
    // make buffers
    Buffers *b = new Buffers(program, flag);
    stopwatch->lap("  Buffers: ");
    // enqueue
    b->enqueue();
    stopwatch->lap("  Enqueue: ");
    // perform the operation
    b->doit();
    stopwatch->lap("  Perform: ");
    // read from buffers
    b->read();
    stopwatch->stop("  Read: ");
    // print
    // b->print();
}
/**
 * Pageable Execution
 */
void executePageable() {
    std::cout << "Executing with pageable CL memory" << std::endl;
    execute(CL_MEM_READ_WRITE);
}
/**
 * Pinned Execution
 */
void executePinned() {
    std::cout << "Executing with pinned CL memory" << std::endl;
    execute(CL_MEM_ALLOC_HOST_PTR);
}
/**
 * Main
 */
int main(int argc, char** argv)
{
    // start
    std::cout << "Simple buffer and sub-buffer Example adjusted" << std::endl;
    // execute pageable
    executePageable();
    // execute pinned
    executePinned();
    // done
    std::cout << "Program completed successfully" << std::endl;
    return 0;
}
