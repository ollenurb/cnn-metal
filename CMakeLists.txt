# Minimum CMake version
cmake_minimum_required(VERSION 3.16)

# Project name and language
project(CNN CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler to clang++
set(CMAKE_CXX_COMPILER clang++)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)

# Source files
set(
    SOURCES
    main.cc
)

set(
    HEADERS
    tensor.hpp
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Set compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -O2
)
