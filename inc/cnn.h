
#ifndef PROJECT_CNN_H_
#define PROJECT_CNN_H_

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <string>
#include <optional>


const int kBlockSize = 16;

struct Args {
    std::string image_path;
    int kernel_size;
    std::optional<std::string> kernel_path;
};

namespace image_processing {
    void GenerateSobelKernels(int size,
                              float*& kernelX,
                              float*& kernelY);

    double ApplyConv2DKernelCuda(const unsigned char* image, 
                                 unsigned char*& result, 
                                 uint32_t width, 
                                 uint32_t height, 
                                 uint32_t poolWidth,
                                 uint32_t poolHeight,
                                 const float* kernelX, 
                                 const float* kernelY, 
                                 int kernel_size); 

    double ApplyConv2DTransposeKernelCuda(
        const unsigned char* input,
        unsigned char*& result,
        uint32_t input_width,
        uint32_t input_height,
        const float* kernelX,
        const float* kernelY,
        int kernel_size);

                            
    __global__ void Conv2DKernel(const unsigned char* image, 
                                 float* result, 
                                 uint32_t width, 
                                 uint32_t height, 
                                 const float* kernelX, 
                                 const float* kernelY, 
                                 int kernel_size,
                                 bool useSharedMem);

    __global__ void Conv2DTransposeKernel(
        const unsigned char* input,
        float* output,
        int input_width,
        int input_height,
        const float* kernel,
        int kernel_size);

    __global__ void ReLUKernel(float* image, int width, int height);
    
    __global__ void MaxPool2D(float* input, unsigned char* result, int width, int height, int poolWidth, int poolHeight);

}  // namespace image_processing

#endif  // PROJECT_BLUR_H_