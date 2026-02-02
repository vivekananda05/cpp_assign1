#ifndef UTILS_H
#define UTILS_H

#include "tensor.h"
#include <vector>
#include <string>

namespace utils {
    
    // Loss functions
    float crossEntropyLoss(const Tensor& predictions, const Tensor& targets);
    float mseLoss(const Tensor& predictions, const Tensor& targets);
    
    // Metrics
    float accuracy(const Tensor& predictions, const Tensor& targets);
    
    // Model analysis
    struct ModelStats {
        int total_parameters;
        int total_macs;
        int total_flops;
        float memory_mb;
    };
    
    ModelStats analyzeModel(const std::vector<Layer*>& layers, 
                           const std::vector<int>& input_shape);
    
    // Data processing
    Tensor normalizeImage(const Tensor& image, float mean = 0.5f, 
                         float std = 0.5f);
    
    // CUDA utils
    bool isCUDAAvailable();
    void setDevice(int device_id);
    void cudaSynchronize();
    
    // Timing utilities
    class Timer {
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        
    public:
        void start();
        double elapsed() const;  // Returns time in seconds
    };
    
} // namespace utils

#endif // UTILS_H