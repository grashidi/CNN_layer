#include "utils.h"

#include <tiffio.h>
#include <iostream>

namespace image_utils {

//  bool LoadTiff(const char* filename, std::vector<uint8_t>& image, uint32_t& width, uint32_t& height) {
//    TIFF* tif = TIFFOpen(filename, "r");
//    if (!tif) {
//      std::cerr << "Failed to open TIFF file: " << filename << std::endl;
//      return false;
//    }
//
//    uint16_t photometric;
//    uint16_t samples_per_pixel;
//
//    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);
//    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
//
//    bool is_gray = ((photometric == PHOTOMETRIC_MINISWHITE || photometric == PHOTOMETRIC_MINISBLACK) && samples_per_pixel == 1);
//
//    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
//    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
//
//    uint32_t num_pixels = width * height;
//    image.resize(num_pixels);
//
//    uint32_t row_size = is_gray ? width : width * 3;
//    std::vector<uint8_t> row_buffer(row_size);
//
//    for (uint32_t row = 0; row < height; ++row) {
//      TIFFReadScanline(tif, row_buffer.data(), row);
//      for (uint32_t col = 0; col < width; ++col) {
//        if (is_gray) {
//          image[row * width + col] = row_buffer[col];
//        } else {
//          uint8_t r = row_buffer[col * 3];
//          uint8_t g = row_buffer[col * 3 + 1];
//          uint8_t b = row_buffer[col * 3 + 2];
//          image[row * width + col] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
//        }
//      }
//    }
//
//    TIFFClose(tif);
//    return true;
//  }
  bool LoadTiff(const char* filename, uint8_t*& image, uint32_t& width, uint32_t& height) {
      TIFF* tif = TIFFOpen(filename, "r");
      if (!tif) {
          std::cerr << "Failed to open TIFF file: " << filename << std::endl;
          return false;
      }

      uint16_t photometric;
      uint16_t samples_per_pixel;

      TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);
      TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);

      bool is_gray = ((photometric == PHOTOMETRIC_MINISWHITE || photometric == PHOTOMETRIC_MINISBLACK) && samples_per_pixel == 1);

      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

      uint32_t num_pixels = width * height;
      image = new uint8_t[num_pixels];  // Allocation here

      uint32_t row_size = is_gray ? width : width * 3;
      std::vector<uint8_t> row_buffer(row_size);

      for (uint32_t row = 0; row < height; ++row) {
          TIFFReadScanline(tif, row_buffer.data(), row);
          for (uint32_t col = 0; col < width; ++col) {
              if (is_gray) {
                  image[row * width + col] = row_buffer[col];
              } else {
                  uint8_t r = row_buffer[col * 3];
                  uint8_t g = row_buffer[col * 3 + 1];
                  uint8_t b = row_buffer[col * 3 + 2];
                  image[row * width + col] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
              }
          }
      }

      TIFFClose(tif);
      return true;
  }

//  bool SaveTiff(const char* filename, const std::vector<uint8_t>& image, uint32_t width, uint32_t height) {
//      TIFF* tif = TIFFOpen(filename, "w");
//      if (!tif) {
//          std::cerr << "Failed to create TIFF file: " << filename << std::endl;
//          return false;
//      }
//
//      // Set TIFF parameters
//      TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
//      TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
//      TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);  // Grayscale image
//      TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
//      TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
//      TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
//      TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // Grayscale format
//
//      // Write image row by row
//      for (uint32_t row = 0; row < height; row++) {
//          if (TIFFWriteScanline(tif, (void*)&image[row * width], row) < 0) {
//              std::cerr << "Error writing TIFF file at row: " << row << std::endl;
//              TIFFClose(tif);
//              return false;
//          }
//      }
//
//      TIFFClose(tif);
//      return true;
//  }
  bool SaveTiff(const char* filename, const uint8_t* image, uint32_t width, uint32_t height) {
      TIFF* tif = TIFFOpen(filename, "w");
      if (!tif) {
          std::cerr << "Failed to create TIFF file: " << filename << std::endl;
          return false;
      }

      // Set TIFF parameters
      TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
      TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
      TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);  // Grayscale image
      TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
      TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
      TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
      TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // Grayscale format

      // Write image row by row
      for (uint32_t row = 0; row < height; row++) {
          if (TIFFWriteScanline(tif, (void*)(image + row * width), row) < 0) {
              std::cerr << "Error writing TIFF file at row: " << row << std::endl;
              TIFFClose(tif);
              return false;
          }
      }

      TIFFClose(tif);
      return true;
  }


}  // namespace image_utils