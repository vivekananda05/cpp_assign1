#include "utils.h"
#include <cmath>
#include <chrono>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace utils {

float crossEntropyLoss(const Tensor& predictions, const Tensor& targets) {
    const auto& pred_data = predictions.getData();
    const auto& target_data = targets.getData();
    
    if (pred_data.size() != target_data.size()) {
        throw std::runtime_error("Size mismatch in crossEntropyLoss");
    }
    
    float loss = 0.0f;
    int batch_size = predictions.getShape()[0];
    int num_classes = predictions.getShape()[1];
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            float p = std::max(pred_data[idx], 1e-8f);
            loss -= target_data[idx] * std::log(p);
        }
    }
    
    return loss / batch_size;
}

float mseLoss(const Tensor& predictions, const Tensor& targets) {
    const auto& pred_data = predictions.getData();
    const auto& target_data = targets.getData();
    
    if (pred_data.size() != target_data.size()) {
        throw std::runtime_error("Size mismatch in mseLoss");
    }
    
    float loss = 0.0f;
    for (size_t i = 0; i < pred_data.size(); ++i) {
        float diff = pred_data[i] - target_data[i];
        loss += diff * diff;
    }
    
    return loss / pred_data.size();
}

float accuracy(const Tensor& predictions, const Tensor& targets) {
    const auto& pred_data = predictions.getData();
    const auto& target_data = targets.getData();
    
    int batch_size = predictions.getShape()[0];
    int num_classes = predictions.getShape()[1];
    
    int correct = 0;
    for (int i = 0; i < batch_size; ++i) {
        int pred_class = 0;
        float max_prob = -1e9;
        
        // Find predicted class
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (pred_data[idx] > max_prob) {
                max_prob = pred_data[idx];
                pred_class = j;
            }
        }
        
        // Find true class
        int true_class = 0;
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (target_data[idx] > 0.5f) {  // Assuming one-hot encoding
                true_class = j;
                break;
            }
        }
        
        if (pred_class == true_class) {
            ++correct;
        }
    }
    
    return static_cast<float>(correct) / batch_size;
}

ModelStats analyzeModel(const std::vector<Layer*>& layers, 
                       const std::vector<int>& input_shape) {
    ModelStats stats = {0, 0, 0, 0.0f};
    
    // Simple analysis
    for (Layer* layer : layers) {
        if (auto conv = dynamic_cast<Conv2d*>(layer)) {
            stats.total_parameters += conv->getNumParameters();
            stats.total_macs += conv->getMACs(input_shape);
            stats.total_flops += conv->getFLOPs(input_shape);
        } else if (auto linear = dynamic_cast<Linear*>(layer)) {
            stats.total_parameters += linear->getNumParameters();
            stats.total_macs += linear->getMACs(input_shape);
            stats.total_flops += linear->getFLOPs(input_shape);
        }
    }
    
    stats.memory_mb = stats.total_parameters * 4.0f / (1024 * 1024);
    return stats;
}

Tensor normalizeImage(const Tensor& image, float mean, float std) {
    std::vector<int> shape = image.getShape();
    Tensor result(shape);
    
    const auto& input_data = image.getData();
    auto& output_data = result.getData();
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = (input_data[i] - mean) / std;
    }
    
    return result;
}

bool isCUDAAvailable() {
#ifdef HAS_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

void setDevice(int device_id) {
#ifdef HAS_CUDA
    cudaSetDevice(device_id);
#endif
}

void cudaSynchronize() {
#ifdef HAS_CUDA
    cudaDeviceSynchronize();
#endif
}

// Timer implementation
void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
}

} // namespace utils