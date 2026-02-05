#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>

class Tensor {
private:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> grad;
    bool requires_grad;
    std::string creation_op;
    std::vector<Tensor*> children;
    
    void computeGradient(Tensor& grad_output);
    
public:
    Tensor();
    Tensor(const std::vector<int>& shape, bool requires_grad = false);
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, 
           bool requires_grad = false);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data, bool requires_grad = false);
    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Destructor
    ~Tensor();
    
    // Getters
    const std::vector<int>& getShape() const { return shape; }
    const std::vector<float>& getData() const { return data; }
    std::vector<float>& getGrad() { return grad; }
    int getTotalSize() const;
    
    // Setters
    void setRequiresGrad(bool requires) { requires_grad = requires; }
    void setCreationOp(const std::string& op) { creation_op = op; }
    void addChild(Tensor* child) { children.push_back(child); }
    
    // Accessors
    float& operator[](int index);
    const float& operator[](int index) const;
    float at(const std::vector<int>& indices) const;
    
    // Initialization methods
    void zeros();
    void ones();
    void randomNormal(float mean = 0.0f, float stddev = 1.0f);
    void randomUniform(float low = 0.0f, float high = 1.0f);
    void xavierUniform();
    
    // Operations
    Tensor matmul(const Tensor& other) const;
    Tensor elementwiseMultiply(const Tensor& other) const;
    Tensor elementwiseAdd(const Tensor& other) const;
    Tensor elementwiseSubtract(const Tensor& other) const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax(int axis = -1) const;
    Tensor logSoftmax(int axis = -1) const;
    Tensor transpose(const std::vector<int>& dims) const;
    Tensor sum(int axis = -1, bool keepdims = false) const;
    Tensor mean(int axis = -1, bool keepdims = false) const;
    Tensor max(int axis = -1, bool keepdims = false) const;
    
    // Convolution operations
    Tensor conv2d(const Tensor& weights, const Tensor& bias, 
                  int stride = 1, int padding = 0) const;
    Tensor maxPool2d(int kernel_size = 2, int stride = 2, 
                     int padding = 0) const;
    Tensor avgPool2d(int kernel_size = 2, int stride = 2, 
                     int padding = 0) const;
    
    // Backward pass
    void backward(const Tensor& grad_output = Tensor());
    void zeroGrad();
    
    // Utility methods
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor flatten() const;
    Tensor pad2d(int padding) const;
    std::string toString() const;
    
    // Static methods
    static Tensor fromFile(const std::string& filename);
    void saveToFile(const std::string& filename) const;
};

// Operator overloads
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

#endif // TENSOR_H