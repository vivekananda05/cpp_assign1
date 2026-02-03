#include "tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <random>
#include <cmath>

Tensor::Tensor() : requires_grad(false), creation_op("none") {
    shape = {1};
    data = {0.0f};
    grad.resize(1, 0.0f);
}

Tensor::Tensor(const std::vector<int>& shape, bool requires_grad) 
    : shape(shape), requires_grad(requires_grad), creation_op("none") {
    int total_size = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Tensor dimensions must be positive");
        }
        total_size *= dim;
    }
    data.resize(total_size, 0.0f);
    grad.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, 
               bool requires_grad)
    : data(data), shape(shape), requires_grad(requires_grad), creation_op("none") {
    int total_size = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Tensor dimensions must be positive");
        }
        total_size *= dim;
    }
    if (data.size() != (size_t)total_size) {
        throw std::runtime_error("Data size doesn't match shape");
    }
    grad.resize(total_size, 0.0f);
}

// Tensor::~Tensor() {
//     // Default destructor - vector members will be automatically destroyed
// }

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : shape(other.shape),
      data(other.data),
      grad(other.grad),
      requires_grad(other.requires_grad),
      creation_op(other.creation_op) {
    // children vector is not copied
}

// Copy assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape = other.shape;
        data = other.data;
        grad = other.grad;
        requires_grad = other.requires_grad;
        creation_op = other.creation_op;
        // children is not copied
    }
    return *this;
}

// Destructor
Tensor::~Tensor() {
    // Automatic cleanup of vector members
}

// Elementwise operations
Tensor Tensor::elementwiseMultiply(const Tensor& other) const {
    if (data.size() != other.data.size()) {
        throw std::runtime_error("Tensor size mismatch for elementwise multiplication");
    }
    
    if (shape != other.shape) {
        throw std::runtime_error("Tensor shape mismatch for elementwise multiplication");
    }
    
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    
    return result;
}

Tensor Tensor::elementwiseAdd(const Tensor& other) const {
    if (data.size() != other.data.size()) {
        throw std::runtime_error("Tensor size mismatch for elementwise addition");
    }
    
    if (shape != other.shape) {
        throw std::runtime_error("Tensor shape mismatch for elementwise addition");
    }
    
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    
    return result;
}

Tensor Tensor::elementwiseSubtract(const Tensor& other) const {
    if (data.size() != other.data.size()) {
        throw std::runtime_error("Tensor size mismatch for elementwise subtraction");
    }
    
    if (shape != other.shape) {
        throw std::runtime_error("Tensor shape mismatch for elementwise subtraction");
    }
    
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    
    return result;
}

int Tensor::getTotalSize() const {
    int total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    return total;
}

float& Tensor::operator[](int index) {
    if (index < 0 || index >= (int)data.size()) {
        throw std::runtime_error("Tensor index out of bounds");
    }
    return data[index];
}

const float& Tensor::operator[](int index) const {
    if (index < 0 || index >= (int)data.size()) {
        throw std::runtime_error("Tensor index out of bounds");
    }
    return data[index];
}

void Tensor::zeros() {
    std::fill(data.begin(), data.end(), 0.0f);
}

void Tensor::ones() {
    std::fill(data.begin(), data.end(), 1.0f);
}

void Tensor::randomNormal(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    for (float& val : data) {
        val = dist(gen);
    }
}

void Tensor::randomUniform(float low, float high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    
    for (float& val : data) {
        val = dist(gen);
    }
}

void Tensor::xavierUniform() {
    if (shape.size() < 2) {
        randomUniform(-0.1f, 0.1f);
        return;
    }
    
    int fan_in = shape[1];
    int fan_out = shape[0];
    for (size_t i = 2; i < shape.size(); ++i) {
        fan_in *= shape[i];
    }
    
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    randomUniform(-limit, limit);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::runtime_error("Matmul requires 2D tensors");
    }
    
    if (shape[1] != other.shape[0]) {
        throw std::runtime_error("Dimension mismatch for matmul");
    }
    
    int m = shape[0];
    int n = shape[1];
    int p = other.shape[1];
    
    std::vector<int> result_shape = {m, p};
    Tensor result(result_shape);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += data[i * n + k] * other.data[k * p + j];
            }
            result.data[i * p + j] = sum;
        }
    }
    
    return result;
}

Tensor Tensor::conv2d(const Tensor& weights, const Tensor& bias, 
                     int stride, int padding) const {
    if (shape.size() != 4) {
        throw std::runtime_error("Conv2d requires 4D input tensor");
    }
    
    int batch_size = shape[0];
    int in_channels = shape[1];
    int in_height = shape[2];
    int in_width = shape[3];
    
    if (weights.shape.size() != 4) {
        throw std::runtime_error("Conv2d requires 4D weight tensor");
    }
    
    int out_channels = weights.shape[0];
    int kernel_h = weights.shape[2];
    int kernel_w = weights.shape[3];
    
    if (weights.shape[1] != in_channels) {
        throw std::runtime_error("Input/weight channel mismatch");
    }
    
    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;
    
    std::vector<int> output_shape = {batch_size, out_channels, 
                                     out_height, out_width};
    Tensor output(output_shape);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                
                                if (ih >= 0 && ih < in_height && 
                                    iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_channels + ic) * 
                                                   in_height + ih) * in_width + iw;
                                    int weight_idx = ((oc * in_channels + ic) * 
                                                    kernel_h + kh) * kernel_w + kw;
                                    
                                    sum += data[input_idx] * weights.data[weight_idx];
                                }
                            }
                        }
                    }
                    
                    sum += bias.data[oc];
                    
                    int output_idx = ((b * out_channels + oc) * 
                                    out_height + oh) * out_width + ow;
                    output.data[output_idx] = sum;
                }
            }
        }
    }
    
    return output;
}

Tensor Tensor::maxPool2d(int kernel_size, int stride, int padding) const {
    if (shape.size() != 4) {
        throw std::runtime_error("maxPool2d requires 4D input tensor");
    }
    
    int batch_size = shape[0];
    int channels = shape[1];
    int in_height = shape[2];
    int in_width = shape[3];
    
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    std::vector<int> output_shape = {batch_size, channels, 
                                     out_height, out_width};
    Tensor output(output_shape);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            
                            if (ih >= 0 && ih < in_height && 
                                iw >= 0 && iw < in_width) {
                                int input_idx = ((b * channels + c) * 
                                               in_height + ih) * in_width + iw;
                                max_val = std::max(max_val, data[input_idx]);
                            }
                        }
                    }
                    
                    int output_idx = ((b * channels + c) * 
                                    out_height + oh) * out_width + ow;
                    output.data[output_idx] = max_val;
                }
            }
        }
    }
    
    return output;
}

Tensor Tensor::relu() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = std::max(0.0f, data[i]);
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
    return result;
}

Tensor Tensor::softmax(int axis) const {
    if (axis < 0) axis = shape.size() + axis;
    if (axis != shape.size() - 1) {
        throw std::runtime_error("Only last axis softmax implemented");
    }
    
    Tensor result(shape);
    
    if (shape.size() == 2) {
        int batch_size = shape[0];
        int num_classes = shape[1];
        
        for (int i = 0; i < batch_size; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            
            // Find max for numerical stability
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                max_val = std::max(max_val, data[idx]);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                result.data[idx] = std::exp(data[idx] - max_val);
                sum_exp += result.data[idx];
            }
            
            for (int j = 0; j < num_classes; ++j) {
                int idx = i * num_classes + j;
                result.data[idx] /= sum_exp;
            }
        }
    }
    
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int old_size = getTotalSize();
    int new_size = 1;
    for (int dim : new_shape) {
        if (dim <= 0) {
            throw std::runtime_error("Reshape dimensions must be positive");
        }
        new_size *= dim;
    }
    
    if (old_size != new_size) {
        throw std::runtime_error("Total size mismatch in reshape");
    }
    
    Tensor result(new_shape);
    result.data = data;
    return result;
}

Tensor Tensor::flatten() const {
    std::vector<int> flat_shape = {getTotalSize()};
    return reshape(flat_shape);
}

void Tensor::backward(const Tensor& grad_output) {
    if (!requires_grad) return;
    
    // Simplified backward: just accumulate gradients
    if (grad.size() != grad_output.data.size()) {
        throw std::runtime_error("Gradient size mismatch");
    }
    
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] += grad_output.data[i];
    }
}

void Tensor::zeroGrad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

std::string Tensor::toString() const {
    std::string result = "Tensor([";
    
    if (data.size() <= 10) {
        // Print all elements for small tensors
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(data[i]);
        }
    } else {
        // Print first and last few elements for large tensors
        for (int i = 0; i < 3; ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(data[i]);
        }
        result += ", ..., ";
        for (size_t i = data.size() - 3; i < data.size(); ++i) {
            if (i > data.size() - 3) result += ", ";
            result += std::to_string(data[i]);
        }
    }
    
    result += "], shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "])";
    
    return result;
}