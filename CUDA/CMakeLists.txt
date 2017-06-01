# CMake entry point
cmake_minimum_required (VERSION 2.8)
project (Stereo)

set(OpenCV_DIR "/home/long/depend/opencv/build")

find_package(OpenCV REQUIRED)
find_package(CUDA)

include_directories(${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../)
CUDA_ADD_LIBRARY(CUDA_TARGETS cuda_interface.cu)       

