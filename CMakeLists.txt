cmake_minimum_required(VERSION 3.10)
project(CUDA_CNN LANGUAGES CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA paths
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR /opt/cuda/include)
set(CMAKE_CUDA_RUNTIME_LIBRARY /opt/cuda/lib/libcudart.so)

# Enable CUDA debug information and optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Set CUDA file extensions
set(CMAKE_CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_source_files_properties(src/cnn.cu PROPERTIES LANGUAGE CUDA)

set(CMAKE_CUDA_ARCHITECTURES 50)  # Target Compute Capability 5.0

# Find required packages
find_package(CUDA REQUIRED)
find_package(TIFF REQUIRED)

# Print CUDA information
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Include Directory: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

# Include directories
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR}
    ${TIFF_INCLUDE_DIR}
    inc
)

# Source files
set(SOURCES
    src/main.cc
    src/utils.cc
    src/cnn.cu
)

# Define the executable
add_executable(cnn ${SOURCES})

# Enable CUDA for the target
set_target_properties(cnn PROPERTIES LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# Link directories
link_directories(/opt/cuda/lib)

# Link libraries
target_link_libraries(cnn PRIVATE
    ${CUDA_LIBRARIES}
    ${TIFF_LIBRARIES}
    nppc
    nppial
    nppig
    nppif
    cudart
)

# Print target properties
get_target_property(CUDA_ENABLED cnn CUDA_ENABLED)
message(STATUS "CUDA Enabled: ${CUDA_ENABLED}")

# Print CUDA flags
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")