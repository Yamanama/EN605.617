#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{    
    const int width = 640, height = 480;

    //Create an 8 bit single channel image
    IplImage* img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
    //Set All Image Pixels To 0
    cvZero(img);

    cvShowImage("Input",img);
    cvWaitKey();


    const int step = img->widthStep;
    const int bytes = img->widthStep * img->height;

    unsigned char *dSrc, *dDst;
    cudaMalloc<unsigned char>(&dSrc,bytes);
    cudaMalloc<unsigned char>(&dDst,bytes);

    //Copy Data From IplImage to Device Pointer
    cudaMemcpy(dSrc,img->imageData,bytes,cudaMemcpyHostToDevice);

    NppiSize size;
    size.width = width;
    size.height = height;

    const Npp8u value = 150;

    //Call NPP function to add a constant value to each pixel of the image
    nppiAddC_8u_C1RSfs(dSrc,step,value,dDst,step,size,1);

    //Copy back the result from device to IplImage
    cudaMemcpy(img->imageData,dDst,bytes,cudaMemcpyDeviceToHost);

    cudaFree(dSrc);
    cudaFree(dDst);

    cvShowImage("Output",img);
    cvWaitKey();

    cvReleaseImage(&img);

    return 0;
}
