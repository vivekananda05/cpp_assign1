
#ifndef UTILS_H
#define UTILS_H

#include "tensor.h"
#include <vector>
#include <string>

// Forward declaration for Layer
class Layer;

namespace utils {

struct ModelStats {
    int total_parameters = 0;
    int total_macs = 0;
    int total_flops = 0;
    float memory_mb = 0.0f;
};

// Function declarations
float crossEntropyLoss(const Tensor& predictions, const Tensor& targets);
float mseLoss(const Tensor& predictions, const Tensor& targets);
float accuracy(const Tensor& predictions, const Tensor& targets);
ModelStats analyzeModel(const std::vector<Layer*>& layers,
                       const std::vector<int>& input_shape);
Tensor normalizeImage(const Tensor& image, float mean = 0.0f, float std = 1.0f);

// CUDA utilities
bool isCUDAAvailable();
void setDevice(int device_id);
void cudaSynchronize();

// Timer class
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    
public:
    Timer() = default;
    void start();
    double elapsed() const;
};

} // namespace utils

#endif // UTILS_H