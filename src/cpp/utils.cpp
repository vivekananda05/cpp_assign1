
#include "utils.h"
#include "layers.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iostream>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace utils {

float crossEntropyLoss(const Tensor& predictions, const Tensor& targets) {
    const auto& pred_data = predictions.getData();
    const auto& target_data = targets.getData();
    
    // Get shapes
    std::vector<int> pred_shape = predictions.getShape();
    std::vector<int> target_shape = targets.getShape();
    
    // Validate shapes
    if (pred_shape.size() != 2) {
        throw std::runtime_error("crossEntropyLoss: predictions must be 2D [batch_size, num_classes]");
    }
    
    if (target_shape.size() != 2) {
        throw std::runtime_error("crossEntropyLoss: targets must be 2D one-hot encoded [batch_size, num_classes]");
    }
    
    int batch_size = pred_shape[0];
    int num_classes = pred_shape[1];
    
    if (target_shape[0] != batch_size || target_shape[1] != num_classes) {
        std::cerr << "Warning: crossEntropyLoss shape mismatch: pred=[" 
                  << pred_shape[0] << "," << pred_shape[1] 
                  << "], target=[" << target_shape[0] << "," << target_shape[1] << "]" << std::endl;
        throw std::runtime_error("crossEntropyLoss: predictions and targets shape mismatch");
    }
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; ++i) {
        // Find max logit for numerical stability
        float max_logit = pred_data[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (pred_data[idx] > max_logit) {
                max_logit = pred_data[idx];
            }
        }
        
        // Compute log sum exp
        float log_sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            log_sum_exp += std::exp(pred_data[idx] - max_logit);
        }
        log_sum_exp = std::log(log_sum_exp) + max_logit;
        
        // Find true class
        int true_class = -1;
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (target_data[idx] > 0.5f) {  // One-hot encoding
                true_class = j;
                break;
            }
        }
        
        if (true_class == -1) {
            // No valid true class found
            std::cerr << "Warning: No valid true class found for sample " << i << std::endl;
            continue;
        }
        
        // Compute loss for this sample: -logit_true + log_sum_exp
        int true_idx = i * num_classes + true_class;
        float sample_loss = -pred_data[true_idx] + log_sum_exp;
        total_loss += sample_loss;
    }
    
    return total_loss / std::max(1, batch_size);
}

float mseLoss(const Tensor& predictions, const Tensor& targets) {
    const auto& pred_data = predictions.getData();
    const auto& target_data = targets.getData();
    
    if (pred_data.size() != target_data.size()) {
        throw std::runtime_error("mseLoss: size mismatch");
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
    
    std::vector<int> pred_shape = predictions.getShape();
    if (pred_shape.size() != 2) {
        throw std::runtime_error("accuracy: predictions must be 2D");
    }
    
    int batch_size = pred_shape[0];
    int num_classes = pred_shape[1];
    
    int correct = 0;
    for (int i = 0; i < batch_size; ++i) {
        // Find predicted class (argmax)
        int pred_class = 0;
        float max_val = pred_data[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (pred_data[idx] > max_val) {
                max_val = pred_data[idx];
                pred_class = j;
            }
        }
        
        // Find true class (from one-hot)
        int true_class = -1;
        for (int j = 0; j < num_classes; ++j) {
            int idx = i * num_classes + j;
            if (target_data[idx] > 0.5f) {
                true_class = j;
                break;
            }
        }
        
        if (true_class != -1 && pred_class == true_class) {
            ++correct;
        }
    }
    
    return static_cast<float>(correct) / std::max(1, batch_size);
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
    const auto& input_data = image.getData();
    std::vector<float> output_data(input_data.size());
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        output_data[i] = (input_data[i] - mean) / std;
    }
    
    return Tensor(output_data, shape);
}

bool isCUDAAvailable() {
#ifdef HAS_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        // CUDA driver not installed or not compatible
        return false;
    }
    return (device_count > 0);
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

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
}

} // namespace utils