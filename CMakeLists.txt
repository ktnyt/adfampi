cmake_minimum_required(VERSION 2.8)

option(ADFA_CIFAR "Build CIFAR classifier" ON)

set(PROJECT_NAME ADFA)
project(${PROJECT_NAME})
set(ADFA_VERSION_MAJOR 1)
set(ADFA_VERSION_MINOR 0)
set(ADFA_VERSION_PATCH 0)

# Include directories for ADFA
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add compiler flags
add_definitions(-DEIGEN_USE_BLAS)

# Copy mnist files
file(COPY mnist DESTINATION ${CMAKE_BINARY_DIR})

# Set MACOSX_RPATH for Mac OS
if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()

add_subdirectory(src)
