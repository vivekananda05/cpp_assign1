#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "tensor.h"
#include "layers.h"
#include <vector>
#include <memory>

class Optimizer {
protected:
    float learning_rate;
    
public:
    Optimizer(float lr = 0.001) : learning_rate(lr) {}
    virtual ~Optimizer() {}
    
    virtual void step(Layer* layer) = 0;
    virtual void zeroGrad(Layer* layer) = 0;
};

class SGD : public Optimizer {
private:
    float momentum;
    float weight_decay;
    
    std::vector<Tensor> velocity;
    
public:
    SGD(float lr = 0.01, float momentum = 0.9, float weight_decay = 0.0);
    
    void step(Layer* layer) override;
    void zeroGrad(Layer* layer) override;
};

class Adam : public Optimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    
    int t;
    std::vector<Tensor> m;
    std::vector<Tensor> v;
    
public:
    Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999,
         float epsilon = 1e-8, float weight_decay = 0.0);
    
    void step(Layer* layer) override;
    void zeroGrad(Layer* layer) override;
};

#endif // OPTIMIZERS_H