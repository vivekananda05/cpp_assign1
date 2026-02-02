cmake_minimum_required(VERSION 3.10)
project(custom_dl_framework)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Download pybind11 if not found
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.0
)

# Options
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(BUILD_TESTS "Build tests" OFF)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")

# Fetch pybind11
FetchContent_MakeAvailable(pybind11)

# Check for CUDA
if(ENABLE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        enable_language(CUDA)
    else()
        message(WARNING "CUDA not found. Building without GPU support.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Add subdirectories
add_subdirectory(src/cpp)

# Installation
install(DIRECTORY src/python/ DESTINATION python
        FILES_MATCHING PATTERN "*.py")
install(DIRECTORY scripts/ DESTINATION scripts
        FILES_MATCHING PATTERN "*.py")
install(DIRECTORY models/ DESTINATION models
        FILES_MATCHING PATTERN "*.py")

# Summary
message(STATUS "========================================")
message(STATUS "Build Configuration Summary")
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Enable CUDA: ${ENABLE_CUDA}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "========================================")