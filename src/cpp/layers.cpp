
#include "layers.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

Layer::Layer(const std::string& name, bool trainable) 
    : name(name), trainable(trainable) {}

// Conv2d implementation
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, const std::string& name)
    : Layer(name, true), in_channels(in_channels), 
      out_channels(out_channels), kernel_size(kernel_size),
      stride(stride), padding(padding) {
    
    // Initialize weights [out_channels, in_channels, kernel_size, kernel_size]
    std::vector<int> weight_shape = {out_channels, in_channels, 
                                     kernel_size, kernel_size};
    weights = Tensor(weight_shape, true);
    weights.xavierUniform();
    
    // Initialize bias [out_channels]
    std::vector<int> bias_shape = {out_channels};
    bias = Tensor(bias_shape, true);
    bias.zeros();
}

Tensor Conv2d::forward(const Tensor& input) {
    input_cache = input;  // Store for backward pass
    return input.conv2d(weights, bias, stride, padding);
}

Tensor Conv2d::backward(const Tensor& grad_output) {
    // Simplified backward - in practice, need to implement proper conv2d backward
    return grad_output;
}

std::vector<Tensor*> Conv2d::getParameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> Conv2d::getGradients() {
    return {&weights, &bias};
}

void Conv2d::zeroGrad() {
    weights.zeroGrad();
    bias.zeroGrad();
}

int Conv2d::getNumParameters() const {
    return out_channels * in_channels * kernel_size * kernel_size + out_channels;
}

int Conv2d::getMACs(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) return 0;
    
    int batch_size = input_shape[0];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // Each output element requires kernel_size*kernel_size*in_channels MACs
    return batch_size * out_channels * out_h * out_w * 
           kernel_size * kernel_size * in_channels;
}

int Conv2d::getFLOPs(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) return 0;
    
    int batch_size = input_shape[0];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // For convolution: MACs * 2 (multiply-add counts as 2 FLOPS)
    int macs = getMACs(input_shape);
    return macs * 2 + batch_size * out_channels * out_h * out_w;  // Add bias
}

// MaxPool2d implementation
MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding, 
                     const std::string& name)
    : Layer(name, false), kernel_size(kernel_size), 
      stride(stride), padding(padding) {}

Tensor MaxPool2d::forward(const Tensor& input) {
    input_cache = input;
    return input.maxPool2d(kernel_size, stride, padding);
}

Tensor MaxPool2d::backward(const Tensor& grad_output) {
    // Simplified backward
    return grad_output;
}

int MaxPool2d::getFLOPs(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 4) return 0;
    
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int in_h = input_shape[2];
    int in_w = input_shape[3];
    
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    // Each output requires kernel_size*kernel_size comparisons
    return batch_size * channels * out_h * out_w * kernel_size * kernel_size;
}

// Linear layer implementation
Linear::Linear(int in_features, int out_features, const std::string& name)
    : Layer(name, true), in_features(in_features), 
      out_features(out_features) {
    
    // Initialize weights [out_features, in_features]
    std::vector<int> weight_shape = {out_features, in_features};
    weights = Tensor(weight_shape, true);
    weights.xavierUniform();
    
    // Initialize bias [out_features]
    std::vector<int> bias_shape = {out_features};
    bias = Tensor(bias_shape, true);
    bias.zeros();
}

Tensor Linear::forward(const Tensor& input) {
    // Reshape input if needed
    Tensor flat_input = input.flatten();
    if (flat_input.getShape()[0] != in_features) {
        throw std::runtime_error("Input size mismatch in Linear layer");
    }
    
    input_cache = flat_input;
    Tensor result = weights.matmul(flat_input.reshape({in_features, 1}));
    
    // Add bias
    for (int i = 0; i < out_features; ++i) {
        result[i] += bias[i];
    }
    
    return result;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // Simplified backward
    return grad_output;
}

std::vector<Tensor*> Linear::getParameters() {
    return {&weights, &bias};
}

std::vector<Tensor*> Linear::getGradients() {
    return {&weights, &bias};
}

void Linear::zeroGrad() {
    weights.zeroGrad();
    bias.zeroGrad();
}

int Linear::getNumParameters() const {
    return out_features * in_features + out_features;
}

int Linear::getMACs(const std::vector<int>& input_shape) const {
    return out_features * in_features;
}

int Linear::getFLOPs(const std::vector<int>& input_shape) const {
    // Each output requires in_features multiplies and (in_features-1) adds
    return out_features * (2 * in_features - 1) + out_features;  // + bias
}

// ReLU implementation
ReLU::ReLU(const std::string& name) : Layer(name, false) {}

Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    return input.relu();
}

Tensor ReLU::backward(const Tensor& grad_output) {
    const auto& input_data = input_cache.getData();
    const auto& grad_data = grad_output.getData();
    std::vector<float> result_data(grad_data);
    
    // Zero out gradient where input was negative
    for (size_t i = 0; i < input_data.size(); ++i) {
        if (input_data[i] <= 0) {
            result_data[i] = 0;
        }
    }
    
    return Tensor(result_data, grad_output.getShape());
}

// Softmax implementation
Softmax::Softmax(int axis, const std::string& name) 
    : Layer(name, false), axis(axis) {}

Tensor Softmax::forward(const Tensor& input) {
    return input.softmax(axis);
}

Tensor Softmax::backward(const Tensor& grad_output) {
    // Simplified - in practice, need Jacobian
    return grad_output;
}

// Dropout implementation
Dropout::Dropout(float p, const std::string& name) 
    : Layer(name, false), p(p), training(true) {}

Tensor Dropout::forward(const Tensor& input) {
    if (!training || p == 0.0f) {
        return input;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<int> shape = input.getShape();
    mask = Tensor(shape);
    Tensor result = input;
    
    for (size_t i = 0; i < result.getData().size(); ++i) {
        if (dist(gen) < p) {
            mask[i] = 0.0f;
            result[i] = 0.0f;
        } else {
            mask[i] = 1.0f / (1.0f - p);
            result[i] *= mask[i];
        }
    }
    
    return result;
}

Tensor Dropout::backward(const Tensor& grad_output) {
    if (!training || p == 0.0f) {
        return grad_output;
    }
    
    const auto& mask_data = mask.getData();
    const auto& grad_data = grad_output.getData();
    std::vector<float> result_data(grad_data.size());
    
    for (size_t i = 0; i < grad_data.size(); ++i) {
        result_data[i] = grad_data[i] * mask_data[i];
    }
    
    return Tensor(result_data, grad_output.getShape());
}

// BatchNorm2d implementation
BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum,
                         const std::string& name)
    : Layer(name, true), num_features(num_features), eps(eps),
      momentum(momentum), training(true) {
    
    // Initialize gamma and beta
    std::vector<int> param_shape = {num_features};
    gamma = Tensor(param_shape, true);
    gamma.ones();
    
    beta = Tensor(param_shape, true);
    beta.zeros();
    
    running_mean = Tensor(param_shape);
    running_mean.zeros();
    
    running_var = Tensor(param_shape);
    running_var.ones();
}

Tensor BatchNorm2d::forward(const Tensor& input) {
    // Simplified implementation
    return input;
}

Tensor BatchNorm2d::backward(const Tensor& grad_output) {
    return grad_output;
}

std::vector<Tensor*> BatchNorm2d::getParameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> BatchNorm2d::getGradients() {
    return {&gamma, &beta};
}

void BatchNorm2d::zeroGrad() {
    gamma.zeroGrad();
    beta.zeroGrad();
}