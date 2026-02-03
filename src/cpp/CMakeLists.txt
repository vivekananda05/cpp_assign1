

# cmake_minimum_required(VERSION 3.10)

# # Set source files
# set(CPP_SOURCES
#     tensor.cpp
#     layers.cpp
#     optimizers.cpp
#     utils.cpp
#     bindings.cpp
# )

# set(CPP_HEADERS
#     tensor.h
#     layers.h
#     optimizers.h
#     utils.h
# )

# # Add CUDA sources if enabled
# if(ENABLE_CUDA AND CUDA_FOUND)
#     set(CUDA_SOURCES tensor_cuda.cu)
#     list(APPEND CPP_SOURCES ${CUDA_SOURCES})
    
#     set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 --expt-relaxed-constexpr")
#     if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#         set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
#     else()
#         set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
#     endif()
# endif()

# # Create Python module manually
# add_library(custom_dl_framework_py MODULE ${CPP_SOURCES})

# # Set properties
# set_target_properties(custom_dl_framework_py PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED ON
#     CXX_EXTENSIONS OFF
#     OUTPUT_NAME "custom_dl_framework"
#     PREFIX ""
#     SUFFIX ".cpython-310-x86_64-linux-gnu.so"  # Force .so extension
#     POSITION_INDEPENDENT_CODE ON
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# # Get Python extension suffix
# execute_process(
#     COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
#     OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# message(STATUS "Python extension suffix: ${PYTHON_EXTENSION_SUFFIX}")

# # Update suffix with actual Python extension
# set_target_properties(custom_dl_framework_py PROPERTIES
#     SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
# )

# # Include directories
# target_include_directories(custom_dl_framework_py 
#     PRIVATE 
#         ${CMAKE_CURRENT_SOURCE_DIR}
#         ${Python3_INCLUDE_DIRS}
#         ${pybind11_INCLUDE_DIR}
# )

# # Add compile definitions
# if(ENABLE_CUDA AND CUDA_FOUND)
#     target_compile_definitions(custom_dl_framework_py PRIVATE HAS_CUDA)
#     target_include_directories(custom_dl_framework_py PRIVATE ${CUDA_INCLUDE_DIRS})
#     target_link_libraries(custom_dl_framework_py ${CUDA_LIBRARIES})
# endif()

# # Link with pybind11 and Python
# target_link_libraries(custom_dl_framework_py 
#     PRIVATE 
#         pybind11::module
#         ${Python3_LIBRARIES}
# )

# # Don't install for now, just build
# message(STATUS "Building Python module: custom_dl_framework")
# message(STATUS "Output directory: ${CMAKE_BINARY_DIR}")
# message(STATUS "Module will be: custom_dl_framework${PYTHON_EXTENSION_SUFFIX}")



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
    
    # Enable CUDA language
    enable_language(CUDA)

    # set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86")
    
    # Set CUDA compilation flags
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 --expt-relaxed-constexpr")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()
    
    # Add CUDA specific flags for your GPU architecture
    # Find your GPU architecture (common ones: sm_50, sm_52, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")  # Change to your GPU architecture
    
    message(STATUS "CUDA enabled: Using CUDA flags: ${CUDA_NVCC_FLAGS}")
endif()

# Create Python module
add_library(custom_dl_framework_py MODULE ${CPP_SOURCES})

# Set properties
set_target_properties(custom_dl_framework_py PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "custom_dl_framework"
    PREFIX ""
    SUFFIX "${Python3_MODULE_EXTENSION}"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Get Python extension suffix
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python extension suffix: ${PYTHON_EXTENSION_SUFFIX}")

# Update suffix with actual Python extension
set_target_properties(custom_dl_framework_py PROPERTIES
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
)

# Include directories
target_include_directories(custom_dl_framework_py 
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIR}
)

# Add compile definitions for CUDA
if(ENABLE_CUDA AND CUDA_FOUND)
    target_compile_definitions(custom_dl_framework_py PRIVATE HAS_CUDA)
    target_include_directories(custom_dl_framework_py PRIVATE ${CUDA_INCLUDE_DIRS})
    
    # Link CUDA libraries
    find_library(CUDART_LIBRARY cudart PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUDNN_LIBRARY cudnn PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    
    target_link_libraries(custom_dl_framework_py 
        PRIVATE 
            ${CUDA_LIBRARIES}
            ${CUDART_LIBRARY}
            ${CUBLAS_LIBRARY}
            # ${CUDNN_LIBRARY}  # Uncomment if you have cuDNN installed
    )
    
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
endif()

# Link with pybind11 and Python
target_link_libraries(custom_dl_framework_py 
    PRIVATE 
        pybind11::module
        ${Python3_LIBRARIES}
)

message(STATUS "Building Python module: custom_dl_framework")