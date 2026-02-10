#include "optimizers.h"
#include <cmath>

SGD::SGD(float lr, float momentum, float weight_decay)
    : Optimizer(lr), momentum(momentum), weight_decay(weight_decay), 
      velocity() {}

void SGD::step(Layer* layer) {
    if (!layer->isTrainable()) return;
    
    auto params = layer->getParameters();
    auto grads = layer->getGradients();
    
    if (velocity.empty()) {
        for (auto& param : params) {
            velocity.push_back(Tensor(param->getShape()));
            velocity.back().zeros();
        }
    }
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto& param = *params[i];
        auto& grad = *grads[i];
        auto& vel = velocity[i];
        
        // Apply weight decay
        if (weight_decay > 0) {
            for (size_t j = 0; j < param.getData().size(); ++j) {
                grad[j] += weight_decay * param[j];
            }
        }
        
        // Update velocity
        if (momentum > 0) {
            for (size_t j = 0; j < vel.getData().size(); ++j) {
                vel[j] = momentum * vel[j] + learning_rate * grad[j];
            }
            
            // Update parameters
            for (size_t j = 0; j < param.getData().size(); ++j) {
                param[j] -= vel[j];
            }
        } else {
            // Simple SGD
            for (size_t j = 0; j < param.getData().size(); ++j) {
                param[j] -= learning_rate * grad[j];
            }
        }
    }
}

void SGD::zeroGrad(Layer* layer) {
    layer->zeroGrad();
}

Adam::Adam(float lr, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(lr), beta1(beta1), beta2(beta2), 
      epsilon(epsilon), weight_decay(weight_decay), t(0) {}

void Adam::step(Layer* layer) {
    if (!layer->isTrainable()) return;
    
    auto params = layer->getParameters();
    auto grads = layer->getGradients();
    
    if (m.empty()) {
        for (auto& param : params) {
            m.push_back(Tensor(param->getShape()));
            m.back().zeros();
            v.push_back(Tensor(param->getShape()));
            v.back().zeros();
        }
    }
    
    t++;
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto& param = *params[i];
        auto& grad = *grads[i];
        auto& mt = m[i];
        auto& vt = v[i];
        
        // Apply weight decay
        if (weight_decay > 0) {
            for (size_t j = 0; j < param.getData().size(); ++j) {
                grad[j] += weight_decay * param[j];
            }
        }
        
        // Update biased first moment estimate
        for (size_t j = 0; j < mt.getData().size(); ++j) {
            mt[j] = beta1 * mt[j] + (1 - beta1) * grad[j];
        }
        
        // Update biased second raw moment estimate
        for (size_t j = 0; j < vt.getData().size(); ++j) {
            vt[j] = beta2 * vt[j] + (1 - beta2) * grad[j] * grad[j];
        }
        
        // Compute bias-corrected first moment estimate
        float m_hat_factor = 1.0f / (1.0f - std::pow(beta1, t));
        float v_hat_factor = 1.0f / (1.0f - std::pow(beta2, t));
        
        // Update parameters
        for (size_t j = 0; j < param.getData().size(); ++j) {
            float m_hat = mt[j] * m_hat_factor;
            float v_hat = vt[j] * v_hat_factor;
            
            param[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

void Adam::zeroGrad(Layer* layer) {
    layer->zeroGrad();
}