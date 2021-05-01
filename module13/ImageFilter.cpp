/**
 * This code was derived from the following provided examples
 * Book:      OpenCL(R) Programming Guide
 * Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, 
 *            James Fung, Dan Ginsburg
 * ISBN-10:   0-321-74964-2
 * ISBN-13:   978-0-321-74964-2
 * Publisher: Addison-Wesley Professional
 * URLs:      http://safari.informit.com/9780132488006/
 *            http://www.openclprogrammingguide.com
 *
 * ImageFilter2D.cpp
 *
 *    This example demonstrates performing 
 *    gaussian filtering on a 2D image using
 *    OpenCL
 *
 *    Requires FreeImage library for image I/O:
 *      http://freeimage.sourceforge.net/
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "FreeImage.h"

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
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

/**
 * Cleanup any created OpenCL resources
 * Modified from module13/ImageFilter2D.cpp activity code
 */
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem imageObjects, cl_mem imageObjects2, cl_sampler sampler)
{
    if (imageObjects != 0)
        clReleaseMemObject(imageObjects);
    if (imageObjects2 != 0)
        clReleaseMemObject(imageObjects2);
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (sampler != 0)
        clReleaseSampler(sampler);

    if (context != 0)
        clReleaseContext(context);

}
/**
 * Load an image using the FreeImage library and create an OpenCL
 * image out of it
 * From module13/ImageFilter2D.cpp activity code
 */
cl_mem LoadImage(cl_context context, char *fileName, int &width, int &height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(fileName, 0);
    FIBITMAP* image = FreeImage_Load(format, fileName);

    // Convert to 32-bit image
    FIBITMAP* temp = image;
    image = FreeImage_ConvertTo32Bits(image);
    FreeImage_Unload(temp);

    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);

    char *buffer = new char[width * height * 4];
    memcpy(buffer, FreeImage_GetBits(image), width * height * 4);

    FreeImage_Unload(image);

    // Create OpenCL image
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;

    cl_int errNum;
    cl_mem clImage;
    clImage = clCreateImage2D(context,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            &clImageFormat,
                            width,
                            height,
                            0,
                            buffer,
                            &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error creating CL image object" << std::endl;
        return 0;
    }

    return clImage;
}
/**
 * Save an image using the FreeImage library
 * From module13/ImageFilter2D.cpp activity code
 */
bool SaveImage(char *fileName, char *buffer, int width, int height)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(fileName);
    FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)buffer, width,
                        height, width * 4, 32,
                        0xFF000000, 0x00FF0000, 0x0000FF00);
    return (FreeImage_Save(format, image, fileName) == TRUE) ? true : false;
}
/**
 * Round up to the nearest multiple of the group size
 * From module13/ImageFilter2D.cpp activity code
 */
size_t RoundUp(int groupSize, int globalSize)
{
    int r = globalSize % groupSize;
    if(r == 0)
    {
     	return globalSize;
    }
    else
    {
     	return globalSize + groupSize - r;
    }
}
/**
 * Check CLI arguments
 */
void checkArgs(int argc, char** argv) {
    if (argc != 3)
    {
        std::cerr << "USAGE: " << argv[0] << " <inputImageFile> <outputImageFiles>" << std::endl;
        exit(-1);
    }
}

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
		std::ifstream srcFile("ImageFilter2D.cl");
		checkErr(srcFile.is_open() ? CL_SUCCESS : -1, 
                 "reading ImageFilter2D.cl");
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

struct Image {
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> imageObjects;
    std::vector<cl_mem> imageObjects2;
    char* buffer;
    cl_sampler sampler;
    cl_image_format clImageFormat;
    Program * program;
    int width, height;
    char** argv;
    /**
	 * Constructor
	 *
	 * platform - the platform
	 */
	Image(Program* program, char** argv, int width, int height) {
		this->width = width;
        this->height = height;
        this->program = program;
        this->argv = argv;
		// initialize
		this->init();
	}
    /**
     * Handle cleanup
     */
    void clean() {
        Cleanup(this->program->context, this->program->queue,
                    this->program->program, this->program->kernel, 
                    this->imageObjects[0], this->imageObjects2[0], this->sampler);
    }
    /**
     * create the image object
     */
    void createImageObject() {
        cl_int errNum;
        cl_image_format clImageFormat;
        cl_mem iObjects[2] = {0, 0};
        // load image
        iObjects[0] = LoadImage(this->program->platform->context, 
                                          this->argv[1], 
                                          this->width, 
                                          this->height);
        // check it
        if (iObjects[0] == 0) {
            std::cerr << "Error loading: " << std::string(this->argv[1]) << std::endl;
            this->clean();
            exit(-1);
        }
        // format
        clImageFormat.image_channel_order = CL_RGBA;
        clImageFormat.image_channel_data_type = CL_UNORM_INT8;
        // create
        iObjects[1] = clCreateImage2D(this->program->platform->context,
                                          CL_MEM_WRITE_ONLY,
                                          &clImageFormat,
                                          this->width,
                                          this->height,
                                          0,
                                          NULL,
                                          &errNum);
        // check it
        checkErr(errNum, "clCreateImage2D");
        // push the image objects
        this->imageObjects.push_back(iObjects[0]);
        this->imageObjects2.push_back(iObjects[1]);
    }
    /**
     * Create the sampler
     */
    void createSampler() {
        cl_int errNum;
        // create the sampler
        this->sampler = clCreateSampler(this->program->platform->context,
                                        CL_FALSE,
                                        CL_ADDRESS_CLAMP_TO_EDGE,
                                        CL_FILTER_NEAREST,
                                        &errNum);
        checkErr(errNum, "clCreateSampler");
    }
    /**
     * Create Command Queues
     */
    void createCommandQueues() {
        cl_int errNum;
        // iterate
        for (unsigned int i = 0; i < this->program->platform->numDevices; i++) {
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
                "gaussian_filter",
                &errNum);
            // check it
            checkErr(errNum, "clCreateKernel(gaussian_filter)");
            // set it
            errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &this->imageObjects[0]);
            errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->imageObjects2[0]);
            errNum |= clSetKernelArg(kernel, 2, sizeof(cl_sampler), &this->sampler);
            errNum |= clSetKernelArg(kernel, 3, sizeof(cl_int), &this->width);
            errNum |= clSetKernelArg(kernel, 4, sizeof(cl_int), &this->height);
            // check it
            checkErr(errNum, "clSetKernelArg(gaussian_filter)");
            // push it
            this->kernels.push_back(kernel);
        }
    }
    /**
     * Enqueue the buffers
     */
    void enqueue() {
        // Work sizes
        size_t localWorkSize[2] = { 16, 16 };
        size_t globalWorkSize[2] =  { RoundUp(localWorkSize[0], width),
                                    RoundUp(localWorkSize[1], height) };
        // Queue the kernel up for execution
        cl_int errNum = clEnqueueNDRangeKernel(this->queues[0], 
                                               this->kernels[0],
                                               2, NULL,
                                               globalWorkSize, localWorkSize,
                                               0, NULL, NULL);
        checkErr(errNum, "clEnqueueNDRangeKernel");
    }
    /**
     * Read the output back to host
     */
    void read() {
        cl_int errNum;
        // instantiate the buffer
        this->buffer = new char [this->width * this->height * 4];
        // declaration
        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { this->width, this->height, 1};
        // queue up the read image
        errNum = clEnqueueReadImage(this->queues[0], this->imageObjects2[0],
                                    CL_TRUE, origin, region, 0, 0, buffer,
                                    0, NULL, NULL);
        // check it
        checkErr(errNum, "clEnqueueReadImage");
    }
    /**
     * Save the image
     */
    void save () {
        if (!SaveImage(this->argv[2], this->buffer, this->width, this->height))
        {
            std::cerr << "Error writing output image: " << this->argv[2] << std::endl;
            clean();
            delete [] this->buffer;
            exit(-1);
        }
    }
    /**
     *  Initialization 
     */
    void init() {
        // create imaage objects
        this->createImageObject();
        // create sampler
        this->createSampler();
        // create command queues
        this->createCommandQueues();
    }
};

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
 * Main
 */
int main(int argc, char** argv)
{
    checkArgs(argc, argv);
    int w, h;
    std::cout << "Enqueued Image Handler" << std::endl;
    std::cout << "Input file: " << argv[1] << std::endl;
    std::cout << "Output file: " << argv[2] << std::endl;
    std::cout << "Runtime Statistics: " << std::endl;
    // make stopwatch
    Stopwatch * stopwatch = new Stopwatch();
	stopwatch->start();
    // make platform
    Platform *platform = new Platform();
    stopwatch->lap("  Platform: ");
    // make program
    Program *program = new Program(platform);
    stopwatch->lap("  Program: ");
    // make image
    Image *image = new Image(program, argv, w, h);
    stopwatch->lap("  ImageObjects: ");
    // enqueue
    image->enqueue();
    stopwatch->lap("  Enqueue: ");
    // read
    image->read();
    stopwatch->lap("  Read: ");
    // save
    image->save();
    stopwatch->stop("  Save: ");
    // image->clean();
    return 0;
}
