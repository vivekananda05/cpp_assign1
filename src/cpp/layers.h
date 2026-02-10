#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include <vector>
#include <string>

class Layer {
protected:
    std::string name;
    bool trainable;
    
public:
    Layer(const std::string& name = "", bool trainable = true);
    virtual ~Layer() {}
    
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> getParameters() = 0;
    virtual std::vector<Tensor*> getGradients() = 0;
    virtual void zeroGrad() = 0;
    
    std::string getName() const { return name; }
    bool isTrainable() const { return trainable; }
};

class Conv2d : public Layer {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    
public:
    Conv2d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0,
           const std::string& name = "Conv2d");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override;
    std::vector<Tensor*> getGradients() override;
    void zeroGrad() override;
    
    // Analysis methods
    int getNumParameters() const;
    int getMACs(const std::vector<int>& input_shape) const;
    int getFLOPs(const std::vector<int>& input_shape) const;
};

class MaxPool2d : public Layer {
private:
    int kernel_size;
    int stride;
    int padding;
    Tensor input_cache;
    
public:
    MaxPool2d(int kernel_size = 2, int stride = 2, int padding = 0,
              const std::string& name = "MaxPool2d");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override { return {}; }
    std::vector<Tensor*> getGradients() override { return {}; }
    void zeroGrad() override {}
    
    int getFLOPs(const std::vector<int>& input_shape) const;
};

class Linear : public Layer {
private:
    int in_features;
    int out_features;
    
    Tensor weights;
    Tensor bias;
    Tensor input_cache;
    
public:
    Linear(int in_features, int out_features, 
           const std::string& name = "Linear");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override;
    std::vector<Tensor*> getGradients() override;
    void zeroGrad() override;
    
    int getNumParameters() const;
    int getMACs(const std::vector<int>& input_shape) const;
    int getFLOPs(const std::vector<int>& input_shape) const;
};

class ReLU : public Layer {
private:
    Tensor input_cache;
    
public:
    ReLU(const std::string& name = "ReLU");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override { return {}; }
    std::vector<Tensor*> getGradients() override { return {}; }
    void zeroGrad() override {}
};

class Softmax : public Layer {
private:
    int axis;
    
public:
    Softmax(int axis = -1, const std::string& name = "Softmax");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override { return {}; }
    std::vector<Tensor*> getGradients() override { return {}; }
    void zeroGrad() override {}
};

class Dropout : public Layer {
private:
    float p;
    bool training;
    Tensor mask;
    
public:
    Dropout(float p = 0.5, const std::string& name = "Dropout");
    
    void setTraining(bool is_training) { training = is_training; }
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override { return {}; }
    std::vector<Tensor*> getGradients() override { return {}; }
    void zeroGrad() override {}
};

class BatchNorm2d : public Layer {
private:
    int num_features;
    float eps;
    float momentum;
    bool training;
    
    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;
    
    Tensor input_cache;
    Tensor normalized_cache;
    
public:
    BatchNorm2d(int num_features, float eps = 1e-5, float momentum = 0.1,
                const std::string& name = "BatchNorm2d");
    
    void setTraining(bool is_training) { training = is_training; }
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> getParameters() override;
    std::vector<Tensor*> getGradients() override;
    void zeroGrad() override;
};

#endif // LAYERS_H