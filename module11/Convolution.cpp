//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 47;
const unsigned int inputSignalHeight = 47;
// a massive input signal
cl_uint inputSignal[inputSignalHeight][inputSignalWidth] =
{
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 4, 4, 4, 3, 2, 2, 2, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 8, 3, 8, 9, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 3, 3, 9, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{0, 9, 0, 8, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{5, 9, 8, 1, 8, 1, 1, 1, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 4, 4, 4, 3, 2, 2, 2, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 8, 3, 8, 9, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 3, 3, 9, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{0, 9, 0, 8, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{5, 9, 8, 1, 8, 1, 1, 1, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 4, 4, 4, 3, 2, 2, 2, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 8, 3, 8, 9, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 3, 3, 9, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{0, 9, 0, 8, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{5, 9, 8, 1, 8, 1, 1, 1, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 4, 4, 4, 3, 2, 2, 2, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 8, 3, 8, 9, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 3, 3, 9, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{0, 9, 0, 8, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{5, 9, 8, 1, 8, 1, 1, 1, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 4, 4, 4, 3, 2, 2, 2, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 8, 3, 8, 9, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 3, 3, 9, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{0, 9, 0, 8, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{5, 9, 8, 1, 8, 1, 1, 1, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 2, 1, 1, 2, 1, 2, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{4, 4, 4, 4, 3, 2, 2, 2, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 8, 3, 8, 9, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{9, 3, 3, 9, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{0, 9, 0, 8, 0, 0, 0, 0, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1}
};
// output signal
const unsigned int outputSignalWidth  = 6;
const unsigned int outputSignalHeight = 6;

cl_float outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_float mask[maskHeight][maskWidth] =
{
	{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
	{0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25}, 
	{0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25}, 
	{0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25}, 
	{0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25}, 
	{0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25}, 
	{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25}
};

/**
 * Check for errors
 * Provided in initial Convolution.cpp
 */
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
/**
 * Callback context
 * Provided in initial Convolution.cpp
 */
void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
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
 *
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
			this->errNum = clGetDeviceIDs(platformIDs[this->i], CL_DEVICE_TYPE_GPU, 0, NULL, &this->numDevices);
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
					this->platformIDs[this->i], CL_DEVICE_TYPE_GPU,
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
			this->deviceIDs, &contextCallback,
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
		// create kernel
		this->createKernel();
	}
	/**
	 * Initialize
	 */
	void init() {
		cl_int errNum;
		// get the src file
		std::ifstream srcFile("Convolution.cl");
		checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");
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
	/**
	 * Create Kernel
	 */
	void createKernel() {
		cl_int errNum;
		// Create kernel object
		this->kernel = clCreateKernel(
			this->program,
			"convolve",
			&errNum);
		// check
		checkErr(errNum, "clCreateKernel");
	}
};
/**
 * Memory Objects
 */
struct Buffers {
	cl_int errNum;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;
	/** 
	 * Constructor
	 */
	Buffers(Program* program) {
		std::cout << "Creating Buffers " << std::endl;
		clock_t start, end;
    	start = clock();
		// make input buffer
		this->inputSignalBuffer = clCreateBuffer(
		program->platform->context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
		// check input
		checkErr(errNum, "clCreateBuffer(inputSignal)");
		// make mask buffer
		this->maskBuffer = clCreateBuffer(
			program->platform->context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(cl_uint) * maskHeight * maskWidth,
			static_cast<void *>(mask),
			&errNum);
		// check mask
		checkErr(errNum, "clCreateBuffer(mask)");
		// make output signal buffer
		this->outputSignalBuffer = clCreateBuffer(
			program->platform->context,
			CL_MEM_WRITE_ONLY,
			sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
			NULL,
			&errNum);
		// check output
		checkErr(errNum, "clCreateBuffer(outputSignal)");
		end = clock();
		double duration = double(end - start) / double(CLOCKS_PER_SEC);
		std::cout << "  Buffer made: " << std::fixed 
				  << duration << std::setprecision(5);
		std::cout << std::endl;
	}
	/**
	 * Add memory objects to kernel
	 */
	void addToKernel(Program* program) {
		cl_int errNum;
		// add
		errNum  = clSetKernelArg(program->kernel, 0, sizeof(cl_mem), &this->inputSignalBuffer);
		errNum |= clSetKernelArg(program->kernel, 1, sizeof(cl_mem), &this->maskBuffer);
		errNum |= clSetKernelArg(program->kernel, 2, sizeof(cl_mem), &this->outputSignalBuffer);
		errNum |= clSetKernelArg(program->kernel, 3, sizeof(cl_uint), &inputSignalWidth);
		errNum |= clSetKernelArg(program->kernel, 4, sizeof(cl_uint), &maskWidth);
		// check
		checkErr(errNum, "clSetKernelArg");
	}
};
/**
 * Perform the convolution operation
 */
void doStuff(Program* program, Buffers* b) {
	std::cout << "Executing Convolution" << std::endl;
	cl_int errNum;
	// Work sizes
	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };
	clock_t start, end;
    start = clock();
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		program->queue, 
		program->kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	end = clock();
	double duration = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "  Kernel: " << std::fixed << duration << std::setprecision(5);
    std::cout << std::endl;
	// check
	checkErr(errNum, "clEnqueueNDRangeKernel");
    start = clock();
	// Read buffer
	errNum = clEnqueueReadBuffer(
		program->queue, 
		b->outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	// check
	checkErr(errNum, "clEnqueueReadBuffer");
	end = clock();
    duration = double(end - start) / double(CLOCKS_PER_SEC);
    std::cout << "  Read Buffer: " << duration << std::endl;
}
/**
 * Print the output
 */
void printOutput() {
	// Output the result buffer
    for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}
}
/**
 * Main
 */
int main(int argc, char** argv)
{
	std::cout << "Executing Convolution" << std::endl;
	Stopwatch * stopwatch = new Stopwatch();
	stopwatch->start();
	// make a new platform
	Platform *platform = new Platform();
	stopwatch->lap("  Platform: ");
	// make a new program
	Program *p= new Program(platform);
	stopwatch->lap("  Program: ");
	// make buffers
	Buffers *buffers = new Buffers(p);
	stopwatch->lap("  Buffers: ");
	// command queue
	cl_device_id device;
	p->queue = CreateCommandQueue(platform->context, &device);
	stopwatch->lap("  Command Queue: ");
	// use buffers
	buffers->addToKernel(p);
	stopwatch->lap("  Loaded Buffers: ");
	// do convolution
	doStuff(p, buffers);
	stopwatch->stop("Convolution Execution: ");
	// print output
	printOutput();
	// exit
    std::cout << std::endl << "Executed program succesfully." << std::endl;
	return 0;
}
