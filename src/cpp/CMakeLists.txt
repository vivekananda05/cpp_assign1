cmake_minimum_required(VERSION 3.10)

# Set source files
set(CPP_SOURCES
    tensor.cpp
    layers.cpp
    optimizers.cpp
    utils.cpp
    bindings.cpp
)

set(CPP_HEADERS
    tensor.h
    layers.h
    optimizers.h
    utils.h
)

# Add CUDA sources if enabled
if(ENABLE_CUDA AND CUDA_FOUND)
    set(CUDA_SOURCES tensor_cuda.cu)
    list(APPEND CPP_SOURCES ${CUDA_SOURCES})
    
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 --expt-relaxed-constexpr")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()
endif()

# Create main C++ library
add_library(custom_dl_framework SHARED ${CPP_SOURCES})

# Set properties
set_target_properties(custom_dl_framework PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(custom_dl_framework 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Python3_INCLUDE_DIRS}
)

# Add compile definitions
if(ENABLE_CUDA AND CUDA_FOUND)
    target_compile_definitions(custom_dl_framework PRIVATE HAS_CUDA)
    target_include_directories(custom_dl_framework PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(custom_dl_framework ${CUDA_LIBRARIES})
endif()

# Create Python module using pybind11
pybind11_add_module(custom_dl_framework_pybind ${CPP_SOURCES})

# Set output name
set_target_properties(custom_dl_framework_pybind PROPERTIES
    OUTPUT_NAME "custom_dl_framework"
    PREFIX ""
    SUFFIX "${Python3_MODULE_EXTENSION}"
)

# Link with pybind11 and Python
target_link_libraries(custom_dl_framework_pybind 
    PRIVATE 
        pybind11::module
        ${Python3_LIBRARIES}
)

# Include directories for Python module
target_include_directories(custom_dl_framework_pybind 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Python3_INCLUDE_DIRS}
)

# Installation
install(TARGETS custom_dl_framework_pybind
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/python
)

install(TARGETS custom_dl_framework
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${CPP_HEADERS}
    DESTINATION include/custom_dl_framework
)

# Print configuration
message(STATUS "Building with pybind11 support")