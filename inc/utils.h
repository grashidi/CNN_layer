
#ifndef PROJECT_UTILS_H_
#define PROJECT_UTILS_H_

#include <vector>
#include <cstdint>

namespace image_utils {

// Loads a TIFF image from a file.
bool LoadTiff(const char* filename, uint8_t*& image, uint32_t& width, uint32_t& height);

// Saves a grayscale image as a TIFF file.
bool SaveTiff(const char* filename, const uint8_t* image, uint32_t width, uint32_t height);

}  // namespace image_utils

#endif  // PROJECT_UTILS_H_