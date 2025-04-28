cmake_minimum_required(VERSION 3.10)
project(CUDA_CNN LANGUAGES CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA paths
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR /opt/cuda/include)
set(CMAKE_CUDA_RUNTIME_LIBRARY /opt/cuda/lib/libcudart.so)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --use_fast_math")
set(CMAKE_CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set(CMAKE_CUDA_ARCHITECTURES 50)

# Find packages
find_package(CUDA REQUIRED)
find_package(TIFF REQUIRED)

# Print debug info
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Include Directory: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

# Include directories
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTOR}
    ${TIFF_INCLUDE_DIR}
    inc
)

# === Executable target (original) ===
set(SOURCES_EXEC
    src/main.cc
    src/utils.cc
    src/cnn.cu
)

add_executable(cnn ${SOURCES_EXEC})
set_target_properties(cnn PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(cnn PRIVATE
    ${CUDA_LIBRARIES}
    ${TIFF_LIBRARIES}
    nppc nppial nppig nppif cudart
)

# === Shared library target for Python ===
set(SOURCES_SHARED
    src/utils.cc
    src/cnn.cu
    src/cnn_interface.cc
)

add_library(cnn_shared SHARED ${SOURCES_SHARED})
set_target_properties(cnn_shared PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(cnn_shared PRIVATE
    ${CUDA_LIBRARIES}
    ${TIFF_LIBRARIES}
    nppc nppial nppig nppif cudart
)

# === Debug info ===
get_target_property(CUDA_ENABLED cnn CUDA_ENABLED)
message(STATUS "CUDA Enabled: ${CUDA_ENABLED}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")