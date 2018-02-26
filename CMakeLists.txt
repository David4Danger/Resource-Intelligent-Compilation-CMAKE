cmake_minimum_required (VERSION 2.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# Determine if the cmake version is high enough to support the -x feature.
set(GPU_SUPPORTED FALSE)
if (CMAKE_VERSION VERSION_LESS 3.3)
  message(STATUS "CMake version not high enough to support GPU acceleration!")
  set(GPU_SUPPORTED TRUE)
endif()
project(stockModel)

# Make sure NVCC uses the proper host compiler on OSX
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
  add_compile_options(-stdlib=libc++)
endif()

# Grab the CUDA package, populate variables
find_package(CUDA)
set(GPU_ACCELERATED ${CUDA_FOUND})

# Set directory and compilation flags for both g++ and nvcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_53,code=sm_53; -gencode arch=compute_53,code=compute_53; -std=c++11; -g; -G;")
set(CUDA_PROPAGATE_HOST_FLAGS off)

# Add directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc/)
if (${GPU_ACCELERATED} AND ${GPU_SUPPORTED})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/incGPU/)
endif()

# Setup environments, depending on GPU accel. status
set(SRCS src/main.cpp src/callModels.cpp)
set(INCS inc/callModels.h)

if (${GPU_ACCELERATED} AND ${GPU_SUPPORTED})
  message(STATUS "Creating GPU accelerated version of executable")
  set(SRCS ${SRCS} srcGPU/prng.cu srcGPU/debugCFP.cu)
  set(INCS ${INCS} incGPU/prng.h incGPU/debugCFP.h)
  
  # Switch the source file format for any cpp files that support GPU acceleration,
  # i.e anything that would normally have '-x cu' passed in during compilation.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/callModels.cpp
                              PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
                             )

  # Create GPU executable and link CUDA library
  cuda_add_executable(stockModel ${SRCS} ${INCS})
  target_link_libraries(stockModel -L/usr/local/cuda/lib64 -lcurand)
else()
  # Create executable
  add_executable(stockModel ${SRCS} ${INCS})
endif()
