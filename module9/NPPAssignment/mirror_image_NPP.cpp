/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * This program was derived from boxFilterNPP.cpp as given by the instructor for
 * the Module 9 NPP activity. Adjusted for constraints and chose to flip the picture
 * instead of filter
 *
 */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

/**
 * Initialize the device
 */
inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}
/**
 * Print some info
 */
bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}
/**
 * Validate the file exists
 *
 * sFilename - the file name
 */
void checkFile(std::string sFilename) {
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);
    if (infile.good())
    {
        std::cout << "Opened: <" << sFilename.data() << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
    }
    else
    {
        std::cout << "Unable to open: <" << sFilename.data() << ">" << std::endl;
        file_errors++;
        infile.close();
    }

    if (file_errors > 0)
    {
        exit(EXIT_FAILURE);
    }
}
/**
 * Get the result name
 *
 * filename - the name of the original
 */
std::string getResultName(std::string filename) {
    // get the name
    std::string sResultFilename = filename;
    // find dot
    std::string::size_type dot = sResultFilename.rfind('.');
    // handle the presence of a dot
    if (dot != std::string::npos)
    {
        sResultFilename = sResultFilename.substr(0, dot);
    }
    // append with changed.pgm
    sResultFilename += "_changed.pgm";
    return sResultFilename;

}
/**
 * Mirror the image
 * 
 * sFilename - the path to the original
 * sResultFilename - the mirrored image name
 */
void flip(std::string sFilename, std::string sResultFilename) {
    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // flip it
    NPP_CHECK_NPP (
        nppiMirror_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), 
                          oDeviceDst.data(), oDeviceDst.pitch(), 
                          oSizeROI, NPP_VERTICAL_AXIS));
    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

}
/**
 * Main
 */
int main(int argc, char *argv[]) {
    try {
        char *filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        std::string sFilename = filePath;
        // init
        cudaDeviceInit(argc, (const char **)argv);
        // print some info
        if (printfNPPinfo(argc, argv) == false) {
            exit(EXIT_SUCCESS);
        }
        // validate the file
        checkFile(sFilename);
        // make the result file name
        std::string sResultFilename = getResultName(sFilename);    
        // flip the picture
        flip(sFilename, sResultFilename);
        // exit
        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException) {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...) {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
