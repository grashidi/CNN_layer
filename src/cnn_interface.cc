// src/cnn_interface.cpp
#include "cnn_interface.h"

extern "C" void run_conv_2d_kernel(
    const unsigned char* image, 
    unsigned char*& result, 
    uint32_t width, 
    uint32_t height, 
    uint32_t poolWidth,
    uint32_t poolHeight,
    const float* kernelX, 
    const float* kernelY, 
    int kernel_size 
) {
    image_processing::ApplyConv2DKernelCuda(
        image,
        result,
        width,
        height,
        poolWidth,
        poolHeight,
        kernelX,
        kernelY,
        kernel_size
    );
}

extern "C" void run_conv_2d_transpose_kernel(
    const unsigned char* image, 
    unsigned char*& result, 
    uint32_t width, 
    uint32_t height, 
    const float* kernelX, 
    const float* kernelY, 
    int kernel_size 
) {
    image_processing::ApplyConv2DTransposeKernelCuda(
        image,
        result,
        width,
        height,
        kernelX,
        kernelY,
        kernel_size
    );
}