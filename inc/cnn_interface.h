#include "cnn.h"

#pragma once

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
);

extern "C" void run_conv_2d_transpose_kernel(
    const unsigned char* input,
    unsigned char*& result,
    uint32_t input_width,
    uint32_t input_height,
    const float* kernelX,
    const float* kernelY,
    int kernel_size
);
