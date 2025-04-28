#include<cnn.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            return 0.0;                                                       \
        }                                                                     \
    } while (0)


namespace image_processing {
    
    double ApplyConv2DKernelCuda(const unsigned char* image, 
                                 unsigned char*& result, 
                                 uint32_t width, 
                                 uint32_t height,
                                 uint32_t poolWidth,
                                 uint32_t poolHeight,
                                 const float* kernelX, 
                                 const float* kernelY, 
                                 int kernel_size) {
        unsigned char* d_image = nullptr;
        unsigned char* d_result = nullptr;
        float* d_tmp = nullptr;
        float* d_kernelX = nullptr;
        float* d_kernelY = nullptr;

        size_t image_size = width * height * sizeof(unsigned char);
        size_t tmp_size = width * height * sizeof(float);
        size_t result_size = static_cast<uint32_t>((width / poolWidth) * (height / poolHeight)) * sizeof(unsigned char);
        size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));
        CUDA_CHECK(cudaMalloc(&d_image, image_size));
        CUDA_CHECK(cudaMalloc(&d_result, result_size));
        CUDA_CHECK(cudaMalloc(&d_kernelX, kernel_size_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernelY, kernel_size_bytes));

        CUDA_CHECK(cudaMemcpy(d_image, image, image_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernelX, kernelX, kernel_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernelY, kernelY, kernel_size_bytes, cudaMemcpyHostToDevice));

        dim3 block_dim(kBlockSize, kBlockSize);
        dim3 grid_dim((width + kBlockSize - 1) / kBlockSize, (height + kBlockSize - 1) / kBlockSize);

        bool useSharedMem = true;  // or false

        size_t sharedMemSize = 0;
        if (useSharedMem) {
            sharedMemSize = (kBlockSize + kernel_size - 1) *
                            (kBlockSize + kernel_size - 1) * sizeof(unsigned char);
        }

//        std::cout << "Launching kenrnel with block size: " << kBlockSize << " and " << "grid size: " << (width + kBlockSize - 1) / kBlockSize << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        Conv2DKernel<<<grid_dim, block_dim, sharedMemSize>>>(d_image, d_tmp, width, height,
                                                             d_kernelX, d_kernelY, kernel_size,
                                                             useSharedMem);
        ReLUKernel<<<grid_dim, block_dim>>>(d_tmp, width, height);
        MaxPool2D<<<grid_dim, block_dim>>>(d_tmp, d_result, width, height, poolWidth, poolHeight);
        auto end = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(result, d_result, result_size, cudaMemcpyDeviceToHost));

        cudaFree(d_image);
        cudaFree(d_tmp);
        cudaFree(d_result);
        cudaFree(d_kernelX);
        cudaFree(d_kernelY);

        return std::chrono::duration<double>(end - start).count();
    }

    double ApplyConv2DTransposeKernelCuda(const unsigned char* input,
                                          unsigned char*& result,
                                          uint32_t input_width,
                                          uint32_t input_height,
                                          const float* kernelX,
                                          const float* kernelY,
                                          int kernel_size) {
        unsigned char* d_input = nullptr;
        float *d_output1 = nullptr, *d_output2 = nullptr;
        float *d_kernelX = nullptr, *d_kernelY = nullptr;

        int output_width = (input_width - 1) * 2 + kernel_size;
        int output_height = (input_height - 1) * 2 + kernel_size;;
        size_t input_size = input_width * input_height * sizeof(unsigned char);
        size_t output_size = output_width * output_height * sizeof(float);
        size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMalloc(&d_output1, output_size));
        CUDA_CHECK(cudaMalloc(&d_output2, output_size));
        CUDA_CHECK(cudaMalloc(&d_kernelX, kernel_bytes));
        CUDA_CHECK(cudaMalloc(&d_kernelY, kernel_bytes));

        CUDA_CHECK(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernelX, kernelX, kernel_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernelY, kernelY, kernel_bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_output1, 0, output_size));
        CUDA_CHECK(cudaMemset(d_output2, 0, output_size));

        dim3 block_dim(kBlockSize, kBlockSize);
        dim3 grid_dim((input_width + kBlockSize - 1) / kBlockSize,
                    (input_height + kBlockSize - 1) / kBlockSize);

        auto start = std::chrono::high_resolution_clock::now();

        Conv2DTransposeKernel<<<grid_dim, block_dim>>>(
            d_input, d_output1, input_width, input_height, d_kernelX, kernel_size
        );
        Conv2DTransposeKernel<<<grid_dim, block_dim>>>(
            d_input, d_output2, input_width, input_height, d_kernelY, kernel_size
        );

        auto end = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Allocate and download results
        std::vector<float> h_out1(output_width * output_height);
        std::vector<float> h_out2(output_width * output_height);
//        result = new unsigned char[output_width * output_height];

        CUDA_CHECK(cudaMemcpy(h_out1.data(), d_output1, output_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_out2.data(), d_output2, output_size, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < h_out1.size(); ++i) {
            float val = (h_out1[i] + h_out2[i]) / 2.0f;
            val = fminf(fmaxf(val, 0.0f), 255.0f);  // Clamp
            result[i] = static_cast<unsigned char>(val);
        }

        cudaFree(d_input);
        cudaFree(d_output1);
        cudaFree(d_output2);
        cudaFree(d_kernelX);
        cudaFree(d_kernelY);

        return std::chrono::duration<double>(end - start).count();
    }


    __global__ void Conv2DKernel(const unsigned char* image, 
                                 float* result, 
                                 uint32_t width, 
                                 uint32_t height, 
                                 const float* kernelX, 
                                 const float* kernelY, 
                                 int kernel_size,
                                 bool useSharedMem) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int x = blockIdx.x * blockDim.x + tx;
        int y = blockIdx.y * blockDim.y + ty;

        int half_kernel = kernel_size / 2;

        if (useSharedMem) {
            extern __shared__ unsigned char sharedImage[];

            int shared_width = blockDim.x + kernel_size - 1;
            int shared_height = blockDim.y + kernel_size - 1;

            // Load the shared memory tile
            for (int dy = ty; dy < shared_height; dy += blockDim.y) {
                for (int dx = tx; dx < shared_width; dx += blockDim.x) {
                    int global_x = min(max(blockIdx.x * blockDim.x + dx - half_kernel, 0), width - 1);
                    int global_y = min(max(blockIdx.y * blockDim.y + dy - half_kernel, 0), height - 1);
                    sharedImage[dy * shared_width + dx] = image[global_y * width + global_x];
                }
            }

            __syncthreads();

            if (x < width && y < height) {
                float gradientX = 0.0f, gradientY = 0.0f;

                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int sx = tx + kx;
                        int sy = ty + ky;
                        float pixel = static_cast<float>(sharedImage[sy * shared_width + sx]);

                        gradientX += pixel * kernelX[ky * kernel_size + kx];
                        gradientY += pixel * kernelY[ky * kernel_size + kx];
                    }
                }

                result[y * width + x] = sqrtf(gradientX * gradientX + gradientY * gradientY);
            }
        } else {
            // Fallback to global memory
            if (x < width && y < height) {
                float gradientX = 0.0f, gradientY = 0.0f;

                for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                    for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                        int ix = min(max(x + kx, 0), width - 1);
                        int iy = min(max(y + ky, 0), height - 1);
                        float pixel = static_cast<float>(image[iy * width + ix]);

                        gradientX += pixel * kernelX[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                        gradientY += pixel * kernelY[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                    }
                }

                result[y * width + x] = sqrtf(gradientX * gradientX + gradientY * gradientY);
            }
        }
    }

    __global__ void Conv2DTransposeKernel(
        const unsigned char* input,
        float* output,
        int input_width,
        int input_height,
        const float* kernel,
        int kernel_size
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= input_width || y >= input_height) return;

        int stride = 2;

        int out_w = (input_width - 1) * stride + kernel_size;
        int out_h = (input_height - 1) * stride + kernel_size;

        int in_idx = y * input_width + x;
        float val = static_cast<float>(input[in_idx]);

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int out_x = x * stride + j;
                int out_y = y * stride + i;
                if ((out_y * out_w + out_x) < (out_w * out_h)) {
                    atomicAdd(&output[out_y * out_w + out_x], val * kernel[i * kernel_size + j]);
                }
            }
        }
    }

    
    __global__ void ReLUKernel(float* image, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = y * width + x;
            image[idx] = min(max(static_cast<int>(roundf(image[idx])), 0), 255);
        }
    }

    __global__ void MaxPool2D(float* input, unsigned char* result, int width, int height, int poolWidth, int poolHeight) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x * poolWidth >= width || y * poolHeight >= height) return;

        float maxVal = -FLT_MAX;
        for (int i = 0; i < poolHeight; ++i) {
            for (int j = 0; j < poolWidth; ++j) {
                int curX = x * poolWidth + j;
                int curY = y * poolHeight + i;
                if (curX < width && curY < height) {
                    float val = input[curY * width + curX];
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
        }
        result[y * (width / poolWidth) + x] = static_cast<unsigned char>(maxVal);
    }

    void GenerateSobelKernels(int size, float*& kernelX, float*& kernelY) {
        // Ensure the size is odd (e.g., 3x3, 5x5, etc.)
        if (size % 2 == 0) {
            std::cerr << "Kernel size must be odd!" << std::endl;
            return;
        }

        int mid = size / 2;

        // Generate horizontal (X) kernel
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                kernelX[i * size + j] = (j - mid) * (mid - abs(i - mid));
            }
        }

        // Generate vertical (Y) kernel
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                kernelY[i * size + j] = (i - mid) * (mid - abs(j - mid));
            }
        }
    }

}