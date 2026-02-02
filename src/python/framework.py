import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import pickle
import json


# Import from the same package
from . import HAS_CPP_BACKEND

if HAS_CPP_BACKEND:
    import custom_dl_framework as cpp

    

class Tensor:
    """Python wrapper for C++ Tensor class"""
    
    def __init__(self, data=None, shape=None, requires_grad=False):
        if HAS_CPP_BACKEND:
            if HAS_CPP_BACKEND:
                self._init_cpp(data, shape, requires_grad)
            else:
                self._init_python(data, shape, requires_grad)
    
    def _init_cpp(self, data, shape, requires_grad):
        """Initialize with C++ backend"""
        if data is not None and shape is not None:
            if isinstance(data, list):
                self._tensor = cpp.Tensor(data, shape, requires_grad)
            elif isinstance(data, np.ndarray):
                self._tensor = cpp.Tensor(data.flatten().tolist(), 
                                         list(data.shape), requires_grad)
            else:
                raise ValueError("Unsupported data type")
        elif shape is not None:
            self._tensor = cpp.Tensor(shape, requires_grad)
        else:
            self._tensor = cpp.Tensor()
            self._is_cpp = True

    def _init_python(self, data, shape, requires_grad):
        """Initialize with pure Python"""
        self._is_cpp = False
        if data is not None:
            self.data = np.array(data, dtype=np.float32)
            if data.ndim == 0:  # Scalar
                self.data = np.array([data], dtype=np.float32)
        elif shape is not None:
            self.data = np.zeros(shape, dtype=np.float32)
        else:
            self.data = np.array([0.0], dtype=np.float32)
        
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None


    @property
    def _tensor(self):
        """For compatibility with C++ access"""
        return self            
    @property
    def shape(self):
        if HAS_CPP_BACKEND:
            return tuple(self._tensor.getShape())
        else:
            return self.shape
    
    @property
    def data(self):
        if HAS_CPP_BACKEND:
            return np.array(self._tensor.getData()).reshape(self.shape)
        else:
            return self.data
    
    def numpy(self):
        """Convert to numpy array"""
        return self.data
    
    def zero_grad(self):
        if HAS_CPP_BACKEND:
            self._tensor.zeroGrad()
        elif self.grad is not None:
            self.grad.fill(0)
    
    def backward(self, grad_output=None):
        if HAS_CPP_BACKEND:
            if grad_output is not None:
                if isinstance(grad_output, Tensor):
                    self._tensor.backward(grad_output._tensor)
                else:
                    grad_tensor = Tensor(grad_output)
                    self._tensor.backward(grad_tensor._tensor)
            else:
                self._tensor.backward()
    
    def __add__(self, other):
        if HAS_CPP_BACKEND:
            return Tensor._from_cpp(self._tensor + other._tensor)
        else:
            return Tensor(self.data + other.data, self.shape)
    
    def __mul__(self, other):
        if HAS_CPP_BACKEND:
            return Tensor._from_cpp(self._tensor * other._tensor)
        else:
            return Tensor(self.data * other.data, self.shape)
    
    def __repr__(self):
        if HAS_CPP_BACKEND:
            return self._tensor.toString()
        else:
            return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    @classmethod
    def _from_cpp(cls, cpp_tensor):
        """Create Tensor from C++ tensor"""
        tensor = cls()
        tensor._tensor = cpp_tensor
        return tensor
    
    @classmethod
    def zeros(cls, shape, requires_grad=False):
        tensor = cls(shape=shape, requires_grad=requires_grad)
        if HAS_CPP_BACKEND:
            tensor._tensor.zeros()
        else:
            tensor.data = np.zeros(shape, dtype=np.float32)
        return tensor
    
    @classmethod
    def ones(cls, shape, requires_grad=False):
        tensor = cls(shape=shape, requires_grad=requires_grad)
        if HAS_CPP_BACKEND:
            tensor._tensor.ones()
        else:
            tensor.data = np.ones(shape, dtype=np.float32)
        return tensor
    
    @classmethod
    def randn(cls, shape, requires_grad=False):
        tensor = cls(shape=shape, requires_grad=requires_grad)
        if HAS_CPP_BACKEND:
            tensor._tensor.randomNormal()
        else:
            tensor.data = np.random.randn(*shape).astype(np.float32)
        return tensor
    
    @classmethod
    def rand(cls, shape, requires_grad=False):
        tensor = cls(shape=shape, requires_grad=requires_grad)
        if HAS_CPP_BACKEND:
            tensor._tensor.randomUniform()
        else:
            tensor.data = np.random.rand(*shape).astype(np.float32)
        return tensor

class Module:
    """Base class for all neural network modules"""
    
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True
    
    def forward(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = []
        for name, param in self._parameters.items():
            params.append(param)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
    
    def eval(self):
        self.train(False)
    
    def add_module(self, name, module):
        self._modules[name] = module
    
    def add_parameter(self, name, parameter):
        self._parameters[name] = parameter
    
    def save(self, path):
        state_dict = {
            'parameters': {k: v.numpy().tolist() for k, v in self._parameters.items()},
            'modules': {k: v.state_dict() for k, v in self._modules.items()}
        }
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        
        for name, param_data in state_dict['parameters'].items():
            if name in self._parameters:
                self._parameters[name].data = np.array(param_data, dtype=np.float32)
        
        for name, module_state in state_dict['modules'].items():
            if name in self._modules:
                self._modules[name].load_state_dict(module_state)
    
    def state_dict(self):
        return {k: v.numpy().tolist() for k, v in self._parameters.items()}
    
    def load_state_dict(self, state_dict):
        for name, param_data in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = np.array(param_data, dtype=np.float32)

class Sequential(Module):
    """Sequential container for modules"""
    
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x