// #include "tensor.h"
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// // CUDA error checking
// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t error = call; \
//         if (error != cudaSuccess) { \
//             printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
//                    cudaGetErrorString(error)); \
//             exit(EXIT_FAILURE); \
//         } \
//     } while(0)

// // Kernel for elementwise addition
// __global__ void addKernel(const float* a, const float* b, float* c, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         c[idx] = a[idx] + b[idx];
//     }
// }

// // Kernel for elementwise multiplication
// __global__ void mulKernel(const float* a, const float* b, float* c, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         c[idx] = a[idx] * b[idx];
//     }
// }

// // Kernel for ReLU activation
// __global__ void reluKernel(const float* input, float* output, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         output[idx] = fmaxf(0.0f, input[idx]);
//     }
// }

// // Kernel for matrix multiplication
// __global__ void matmulKernel(const float* A, const float* B, float* C,
//                              int M, int N, int K) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (row < M && col < K) {
//         float sum = 0.0f;
//         for (int i = 0; i < N; ++i) {
//             sum += A[row * N + i] * B[i * K + col];
//         }
//         C[row * K + col] = sum;
//     }
// }

// // Kernel for 2D convolution
// __global__ void conv2dKernel(const float* input, const float* weights,
//                              const float* bias, float* output,
//                              int batch_size, int in_channels,
//                              int in_height, int in_width,
//                              int out_channels, int kernel_h, int kernel_w,
//                              int stride, int padding,
//                              int out_height, int out_width) {
    
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_threads = gridDim.x * blockDim.x;
//     int total_outputs = batch_size * out_channels * out_height * out_width;
    
//     for (int i = idx; i < total_outputs; i += total_threads) {
//         int b = i / (out_channels * out_height * out_width);
//         int rest = i % (out_channels * out_height * out_width);
//         int oc = rest / (out_height * out_width);
//         rest = rest % (out_height * out_width);
//         int oh = rest / out_width;
//         int ow = rest % out_width;
        
//         float sum = 0.0f;
        
//         for (int ic = 0; ic < in_channels; ++ic) {
//             for (int kh = 0; kh < kernel_h; ++kh) {
//                 for (int kw = 0; kw < kernel_w; ++kw) {
//                     int ih = oh * stride + kh - padding;
//                     int iw = ow * stride + kw - padding;
                    
//                     if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
//                         int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
//                         int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
//                         sum += input[input_idx] * weights[weight_idx];
//                     }
//                 }
//             }
//         }
        
//         sum += bias[oc];
//         output[i] = sum;
//     }
// }

// // Kernel for max pooling
// __global__ void maxPool2dKernel(const float* input, float* output,
//                                int batch_size, int channels,
//                                int in_height, int in_width,
//                                int kernel_size, int stride, int padding,
//                                int out_height, int out_width) {
    
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_threads = gridDim.x * blockDim.x;
//     int total_outputs = batch_size * channels * out_height * out_width;
    
//     for (int i = idx; i < total_outputs; i += total_threads) {
//         int b = i / (channels * out_height * out_width);
//         int rest = i % (channels * out_height * out_width);
//         int c = rest / (out_height * out_width);
//         rest = rest % (out_height * out_width);
//         int oh = rest / out_width;
//         int ow = rest % out_width;
        
//         float max_val = -INFINITY;
        
//         for (int kh = 0; kh < kernel_size; ++kh) {
//             for (int kw = 0; kw < kernel_size; ++kw) {
//                 int ih = oh * stride + kh - padding;
//                 int iw = ow * stride + kw - padding;
                
//                 if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
//                     int input_idx = ((b * channels + c) * in_height + ih) * in_width + iw;
//                     max_val = fmaxf(max_val, input[input_idx]);
//                 }
//             }
//         }
        
//         output[i] = max_val;
//     }
// }

// // CUDA wrapper functions
// extern "C" {
//     void cuda_add(const float* a, const float* b, float* c, int n) {
//         float *d_a, *d_b, *d_c;
//         CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
        
//         CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
//         CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));
        
//         int blockSize = 256;
//         int gridSize = (n + blockSize - 1) / blockSize;
//         addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        
//         CUDA_CHECK(cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
        
//         CUDA_CHECK(cudaFree(d_a));
//         CUDA_CHECK(cudaFree(d_b));
//         CUDA_CHECK(cudaFree(d_c));
//     }
    
//     void cuda_matmul(const float* A, const float* B, float* C,
//                      int M, int N, int K) {
//         float *d_A, *d_B, *d_C;
//         CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_C, M * K * sizeof(float)));
        
//         CUDA_CHECK(cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
//         CUDA_CHECK(cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
        
//         dim3 blockDim(16, 16);
//         dim3 gridDim((K + blockDim.x - 1) / blockDim.x,
//                      (M + blockDim.y - 1) / blockDim.y);
        
//         matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        
//         CUDA_CHECK(cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost));
        
//         CUDA_CHECK(cudaFree(d_A));
//         CUDA_CHECK(cudaFree(d_B));
//         CUDA_CHECK(cudaFree(d_C));
//     }
    
//     void cuda_conv2d(const float* input, const float* weights, const float* bias,
//                      float* output, int batch_size, int in_channels,
//                      int in_height, int in_width, int out_channels,
//                      int kernel_h, int kernel_w, int stride, int padding) {
        
//         int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
//         int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;
        
//         float *d_input, *d_weights, *d_bias, *d_output;
//         int input_size = batch_size * in_channels * in_height * in_width;
//         int weights_size = out_channels * in_channels * kernel_h * kernel_w;
//         int output_size = batch_size * out_channels * out_height * out_width;
        
//         CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_weights, weights_size * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_bias, out_channels * sizeof(float)));
//         CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
        
//         CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
//         CUDA_CHECK(cudaMemcpy(d_weights, weights, weights_size * sizeof(float), cudaMemcpyHostToDevice));
//         CUDA_CHECK(cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice));
        
//         int blockSize = 256;
//         int gridSize = (output_size + blockSize - 1) / blockSize;
        
//         conv2dKernel<<<gridSize, blockSize>>>(d_input, d_weights, d_bias, d_output,
//                                               batch_size, in_channels, in_height, in_width,
//                                               out_channels, kernel_h, kernel_w,
//                                               stride, padding, out_height, out_width);
        
//         CUDA_CHECK(cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        
//         CUDA_CHECK(cudaFree(d_input));
//         CUDA_CHECK(cudaFree(d_weights));
//         CUDA_CHECK(cudaFree(d_bias));
//         CUDA_CHECK(cudaFree(d_output));
//     }
// }


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