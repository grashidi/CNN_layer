#include "utils.h"
#include "cnn.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << "<image_path> <kernel_size> <kernel_path>" << std::endl;
    return 1;
  }

  int kernel_size = std::stoi(argv[1]);
  std::string tmp = argv[2];
  const char* image_path = tmp.c_str();

  unsigned char* image;
  uint32_t width, height;
  uint32_t poolWidth = 2, poolHeight = 2;

  if (!image_utils::LoadTiff(image_path, image, width, height)) {
    return -1;
  }

  uint32_t outWidth = static_cast<int>(width / poolWidth);
  uint32_t outHeight = static_cast<int>(height / poolHeight);

  unsigned char* processed_image = new unsigned char[outWidth * outHeight];

  float* kernelX = new float[kernel_size * kernel_size];
  float* kernelY = new float[kernel_size * kernel_size];

  image_processing::GenerateSobelKernels(kernel_size, kernelX, kernelY);
  double cuda_time = image_processing::ApplyConv2DKernelCuda(image,
                                                             processed_image,
                                                             width,
                                                             height,
                                                             poolWidth,
                                                             poolHeight,
                                                             kernelX,
                                                             kernelY,
                                                             kernel_size);

  std::string path = "images/processed";
  size_t dotPos = tmp.find_last_of('.');
  size_t slashPos = tmp.find_last_of('/');
  std::string out_file = path + tmp.substr(slashPos);
  image_utils::SaveTiff(out_file.c_str(), processed_image, outWidth, outHeight);

  std::cout << kernel_size << "\t" << cuda_time << std::endl;

  delete image;
  delete processed_image;
  delete kernelX;
  delete kernelY;

  return 0;
}