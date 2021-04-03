//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
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

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}


/**
 *  Modifications to support the requirements for Module 10 assignment below
 *  this line.
 */

/**
 * The input arrays
 */
struct Arrays {
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    /**
     * Constructor
     */
    Arrays() {}
};
/**
 * The Operation Structure
 */
struct Operation {
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_device_id device;
    cl_kernel kernel;
    cl_int errNum;
    cl_mem memObjects[3];
    const char * name;
    float result[ARRAY_SIZE];
    /**
     * Constructor
     */
    Operation(const char * kernelName, Arrays * results) {
        this->name = kernelName;
        this->device = 0;
        this->context = CreateContext();
        this->commandQueue = CreateCommandQueue(this->context, &this->device);
        this->program = CreateProgram(this->context, 
                                      this->device, 
                                      "HelloWorld.cl");
        this->kernel = clCreateKernel(this->program, kernelName, NULL);
    }
};

/**
 * Clean up
 *
 * operation - the operation
 */
void clean(Operation operation) {
    clReleaseCommandQueue(operation.commandQueue);
    clReleaseKernel(operation.kernel);
    clReleaseProgram(operation.program);
    clReleaseContext(operation.context);
}
/**
 * Create memory objects
 *
 * op - the operation
 * arrays - the input arrays
 * memObjects - the memory objects
 */
void createMemObjects(Operation * op, Arrays* arrays, cl_mem memObjects[3]) {
    cl_int errNum;
    // make the memory object and check it
    if (!CreateMemObjects(op->context, memObjects, arrays->a, arrays->b)) {
        clean(*op);
        exit(1);
    }
    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(op->kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(op->kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(op->kernel, 2, sizeof(cl_mem), &memObjects[2]);
    // check the status
    if (errNum != CL_SUCCESS) {
        std::cerr << "Error setting kernel arguments." << std::endl;
        clean(*op);
        exit(1);
    }
}
/**
 * Populate the arrays
 *
 * arrays - the arrays
 */
void populateArrays(Arrays * arrays) {
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        arrays->a[i] = (float)i;
        arrays->b[i] = (float)(i * 2);
    }
}
/**
 * Perform the operation
 * 
 * op - the operation
 * arrays - the input arrays
 */
void doStuff(Operation *op, Arrays * arrays) {
    std::cout << "Executing " << op->name << std::endl;
    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };
    cl_mem memObjects[3] = {0,0,0};
    createMemObjects(op, arrays, memObjects);
    clock_t start, end;
    start = clock();
    // Queue the kernel up for execution across the array
    clEnqueueNDRangeKernel(op->commandQueue, op->kernel, 
                                    1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    end = clock();
    double duration = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "  Kernel: " << std::fixed << duration << std::setprecision(5);
    std::cout << std::endl;
    start = clock();
    // Read the output buffer back to the Host
    clEnqueueReadBuffer(op->commandQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), op->result, 0, NULL, NULL);
    end = clock();
    duration = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "  Read Buffer: " << duration << std::endl;
}
/**
 *  Print the results
 *
 *  operations - the operations
 *  arrays - the input arrays
 */
void printOps(Operation operations[], Arrays* arrays) {
    for (int j =0; j < ARRAY_SIZE; j++) {
        for (int i =0; i < 5; i++) { 
            std::cout << operations[i].result[j] << " ";
        }
        std::cout << std::endl;
    }
}
/**
 * Main
 */
int main(int argc, char** argv)
{
    // make the arrays
    Arrays *arrays = new Arrays();
    populateArrays(arrays);
    // make the operations
    Operation operations[5] = {
        Operation("sum", arrays),
        Operation("difference", arrays),
        Operation("product", arrays),
        Operation("quotient", arrays),
        Operation("power", arrays)
    };
    // perform the operations
    for (int i =0; i < 5; i++) {
        doStuff(&operations[i], arrays);
    }
    // print
    // printOps(operations, arrays);
    // cleanup
    for (int i=0; i <5; i++) {
        clean(operations[i]);
    }
    return 0;
}
