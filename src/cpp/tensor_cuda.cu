

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void mulKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void reluKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" {
    void cuda_add(const float* a, const float* b, float* c, int n) {
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        
        cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        
        cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    
    void cuda_mul(const float* a, const float* b, float* c, int n) {
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, n * sizeof(float));
        
        cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        mulKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        
        cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    
    void cuda_relu(const float* input, float* output, int n) {
        float *d_input, *d_output;
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        reluKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
        
        cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
}

#endif // HAS_CUDA