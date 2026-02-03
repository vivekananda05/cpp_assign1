# test_cuda.py
import sys
import os

build_dir = "/home/user/Downloads/GNR638/ASSIGN1/custom_dl_framework/build"
sys.path.insert(0, build_dir)


import custom_dl_framework as dl

print("Testing CUDA support...")

# Check if CUDA is available
cuda_available = dl.isCUDAAvailable()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Test basic tensor operations (they might use CUDA internally)
    tensor = dl.Tensor([1000, 1000])
    tensor.randomNormal(0.0, 1.0)
    print(f"Created large tensor on {'GPU' if cuda_available else 'CPU'}")
    
    # Test convolution (if implemented with CUDA)
    print("\nTesting CUDA operations...")
    
    # Create input and weights for convolution
    input_tensor = dl.Tensor([1, 3, 32, 32])  # batch=1, channels=3, height=32, width=32
    input_tensor.randomNormal(0.0, 0.1)
    
    # The conv2d operation in tensor.cpp might use CUDA if tensor_cuda.cu is properly implemented
    print("Note: For actual CUDA operations, you need to implement CUDA kernels")
    print("in tensor_cuda.cu and call them from your Tensor class methods.")
else:
    print("\nCUDA not available. Possible reasons:")
    print("1. No NVIDIA GPU detected")
    print("2. CUDA toolkit not installed")
    print("3. NVIDIA drivers not installed")
    print("4. CMake didn't find CUDA (check cmake output)")