
#ifndef PROJECT_CNN_H_
#define PROJECT_CNN_H_

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cfloat>


const int kBlockSize = 16;

namespace image_processing {
    void GenerateSobelKernels(int size,
                              std::vector<float>& kernelX,
                              std::vector<float>& kernelY);

    double ApplyConv2DKernelCuda(const std::vector<unsigned char>& image, 
                                 std::vector<unsigned char>& result, 
                                 uint32_t width, 
                                 uint32_t height, 
                                 uint32_t poolWidth,
                                 uint32_t poolHeight,
                                 const std::vector<float>& kernelX, 
                                 const std::vector<float>& kernelY, 
                                 int kernel_size); 
                            
    __global__ void Conv2DKernel(const unsigned char* image, 
                                 float* result, 
                                 uint32_t width, 
                                 uint32_t height, 
                                 const float* kernelX, 
                                 const float* kernelY, 
                                 int kernel_size,
                                 bool useSharedMem);

    __global__ void ReLUKernel(float* image, int width, int height);
    
    __global__ void MaxPool2D(float* input, unsigned char* result, int width, int height, int poolWidth, int poolHeight);

}  // namespace image_processing

#endif  // PROJECT_BLUR_H_