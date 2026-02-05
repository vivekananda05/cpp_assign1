import sys
import os
import math

from . import HAS_CPP_BACKEND

if HAS_CPP_BACKEND:
    import custom_dl_framework as cpp

class Tensor:
    """Python wrapper for C++ Tensor class"""
    
    def __init__(self, data=None, shape=None, requires_grad=False):
        self._is_cpp = True
        self._tensor_cpp = None
        
        if HAS_CPP_BACKEND:
            self._init_cpp(data, shape, requires_grad)
        else:
            raise RuntimeError("C++ backend is required but not available")
    
    def _init_cpp(self, data, shape, requires_grad):
        """Initialize with C++ backend"""
        if data is not None and shape is not None:
            if isinstance(data, list):
                self._tensor_cpp = cpp.Tensor(data, shape, requires_grad)
            else:
                
                if hasattr(data, '__iter__'):
                    data_list = list(data)
                else:
                    data_list = [data]
                self._tensor_cpp = cpp.Tensor(data_list, shape, requires_grad)
        elif shape is not None:
            self._tensor_cpp = cpp.Tensor(shape, requires_grad)
        else:
            self._tensor_cpp = cpp.Tensor()
    
    @property
    def _tensor(self):
        """Get C++ tensor"""
        return self._tensor_cpp
    
    @property
    def shape(self):
        if self._tensor_cpp:
            return tuple(self._tensor_cpp.getShape())
        return (1,)
    
    def to_list(self):
        """Convert tensor data to Python list"""
        return self._tensor_cpp.getData()
    
    def reshape(self, new_shape):
        """Reshape the tensor - handle -1 for automatic dimension inference"""
        if not isinstance(new_shape, (list, tuple)):
            new_shape = [new_shape]
        
        # Handle -1 in reshape (automatic dimension inference)
        if -1 in new_shape:
            total_elements = self.getTotalSize()
            specified_elements = 1
            neg_one_index = -1
            
            for i, dim in enumerate(new_shape):
                if dim == -1:
                    if neg_one_index != -1:
                        raise ValueError("Only one dimension can be -1")
                    neg_one_index = i
                elif dim > 0:
                    specified_elements *= dim
                else:
                    raise ValueError(f"Invalid dimension {dim} in reshape")
            
            if neg_one_index != -1:
                inferred_dim = total_elements // specified_elements
                if total_elements % specified_elements != 0:
                    raise ValueError(f"Cannot reshape tensor of size {total_elements} into shape {new_shape}")
                
                new_shape = list(new_shape)
                new_shape[neg_one_index] = inferred_dim
        
        # Check for negative dimensions
        for dim in new_shape:
            if dim <= 0:
                raise ValueError(f"Reshape dimensions must be positive, got {new_shape}")
        
        # Create a new tensor with the reshaped data
        reshaped_tensor = Tensor()
        reshaped_tensor._tensor_cpp = self._tensor_cpp.reshape(list(new_shape))
        reshaped_tensor._is_cpp = True
        return reshaped_tensor
    
    def flatten(self):
        """Flatten the tensor to 1D"""
        total_elements = self.getTotalSize()
        return self.reshape([total_elements])
    
    # def backward(self, grad_output=None):
    #     """Backward pass"""
    #     if grad_output is not None:
    #         if isinstance(grad_output, Tensor):
    #             self._tensor_cpp.backward(grad_output._tensor_cpp)
    #         else:
    #             grad_tensor = Tensor(grad_output)
    #             self._tensor_cpp.backward(grad_tensor._tensor_cpp)
    #     else:
    #         self._tensor_cpp.backward()

    def backward(self, grad_output=None):
        """Backward pass"""
        # Always create a gradient tensor
        if grad_output is None:
            # For scalar loss, create gradient of 1.0
            grad_tensor = Tensor([1.0], [1])
        elif isinstance(grad_output, Tensor):
            grad_tensor = grad_output
        else:
            # Convert to tensor
            if hasattr(grad_output, '__len__'):
                grad_tensor = Tensor(grad_output)
            else:
                # Scalar
                grad_tensor = Tensor([grad_output], [1])
        
        # Call C++ backward with the gradient tensor
        self._tensor_cpp.backward(grad_tensor._tensor_cpp)
    
    def zero_grad(self):
        """Zero gradients"""
        self._tensor_cpp.zeroGrad()
    
    def __repr__(self):
        if hasattr(self._tensor_cpp, 'toString'):
            return self._tensor_cpp.toString()
        else:
            try:
                shape_str = str(self.shape) if hasattr(self, 'shape') else 'unknown shape'
                return f"Tensor({shape_str}, dtype=float32)"
            except:
                return "Tensor(C++ backend)"
    
    # Arithmetic operations
    def __add__(self, other):
        """Elementwise addition"""
        if isinstance(other, Tensor):
            # Check if elementwiseAdd method exists
            if hasattr(self._tensor_cpp, 'elementwiseAdd'):
                result = Tensor()
                result._tensor_cpp = self._tensor_cpp.elementwiseAdd(other._tensor_cpp)
                result._is_cpp = True
                return result
            else:
                # Fallback: create new tensor with added values
                data1 = self.to_list()
                data2 = other.to_list()
                if len(data1) != len(data2):
                    raise ValueError("Tensor sizes must match for addition")
                new_data = [a + b for a, b in zip(data1, data2)]
                return Tensor(new_data, self.getShape())
        else:
            # Create a tensor from the scalar
            data = self.to_list()
            new_data = [val + other for val in data]
            return Tensor(new_data, self.getShape())
    
    def __mul__(self, other):
        """Elementwise multiplication"""
        if isinstance(other, Tensor):
            # Check if elementwiseMultiply method exists
            if hasattr(self._tensor_cpp, 'elementwiseMultiply'):
                result = Tensor()
                result._tensor_cpp = self._tensor_cpp.elementwiseMultiply(other._tensor_cpp)
                result._is_cpp = True
                return result
            else:
                # Fallback: create new tensor with multiplied values
                data1 = self.to_list()
                data2 = other.to_list()
                if len(data1) != len(data2):
                    raise ValueError("Tensor sizes must match for multiplication")
                new_data = [a * b for a, b in zip(data1, data2)]
                return Tensor(new_data, self.getShape())
        else:
            # Scalar multiplication
            data = self.to_list()
            new_data = [val * other for val in data]
            return Tensor(new_data, self.getShape())
    
    def __sub__(self, other):
        """Elementwise subtraction"""
        if isinstance(other, Tensor):
            # Check if elementwiseSubtract method exists
            if hasattr(self._tensor_cpp, 'elementwiseSubtract'):
                result = Tensor()
                result._tensor_cpp = self._tensor_cpp.elementwiseSubtract(other._tensor_cpp)
                result._is_cpp = True
                return result
            else:
                # Fallback: create new tensor with subtracted values
                data1 = self.to_list()
                data2 = other.to_list()
                if len(data1) != len(data2):
                    raise ValueError("Tensor sizes must match for subtraction")
                new_data = [a - b for a, b in zip(data1, data2)]
                return Tensor(new_data, self.getShape())
        else:
            # Scalar subtraction
            data = self.to_list()
            new_data = [val - other for val in data]
            return Tensor(new_data, self.getShape())
    
    def matmul(self, other):
        """Matrix multiplication"""
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires a Tensor argument")
        
        if hasattr(self._tensor_cpp, 'matmul'):
            result = Tensor()
            result._tensor_cpp = self._tensor_cpp.matmul(other._tensor_cpp)
            result._is_cpp = True
            return result
        else:
            # Fallback Python implementation
            shape1 = self.getShape()
            shape2 = other.getShape()
            
            if len(shape1) != 2 or len(shape2) != 2:
                raise ValueError("matmul requires 2D tensors")
            
            if shape1[1] != shape2[0]:
                raise ValueError(f"Shape mismatch: {shape1} and {shape2}")
            
            m, n = shape1
            p = shape2[1]
            data1 = self.to_list()
            data2 = other.to_list()
            
            result_data = [0.0] * (m * p)
            
            for i in range(m):
                for j in range(p):
                    sum_val = 0.0
                    for k in range(n):
                        sum_val += data1[i * n + k] * data2[k * p + j]
                    result_data[i * p + j] = sum_val
            
            return Tensor(result_data, [m, p])
    
    def conv2d(self, weight, bias, stride=1, padding=0):
        """2D convolution"""
        if not isinstance(weight, Tensor) or not isinstance(bias, Tensor):
            raise TypeError("conv2d requires Tensor arguments for weight and bias")
        
        if hasattr(self._tensor_cpp, 'conv2d'):
            result = Tensor()
            result._tensor_cpp = self._tensor_cpp.conv2d(weight._tensor_cpp, bias._tensor_cpp, stride, padding)
            result._is_cpp = True
            return result
        else:
            # Fallback: return a dummy tensor for now
            print("Warning: conv2d not available in C++, using dummy tensor")
            return Tensor.randn([1, 1, 32, 32])
    
    def max_pool2d(self, kernel_size, stride=None, padding=0):
        """2D max pooling"""
        if stride is None:
            stride = kernel_size
        
        if hasattr(self._tensor_cpp, 'maxPool2d'):
            result = Tensor()
            result._tensor_cpp = self._tensor_cpp.maxPool2d(kernel_size, stride, padding)
            result._is_cpp = True
            return result
        else:
            # Fallback: return a dummy tensor for now
            print("Warning: maxPool2d not available in C++, using dummy tensor")
            return self
    
    def relu(self):
        """ReLU activation"""
        if hasattr(self._tensor_cpp, 'relu'):
            result = Tensor()
            result._tensor_cpp = self._tensor_cpp.relu()
            result._is_cpp = True
            return result
        else:
            # Fallback: apply ReLU manually
            data = self.to_list()
            new_data = [max(0.0, val) for val in data]
            return Tensor(new_data, self.getShape())
    
    def sigmoid(self):
        """Sigmoid activation"""
        if hasattr(self._tensor_cpp, 'sigmoid'):
            result = Tensor()
            result._tensor_cpp = self._tensor_cpp.sigmoid()
            result._is_cpp = True
            return result
        else:
            # Fallback: apply sigmoid manually
            import math
            data = self.to_list()
            new_data = [1.0 / (1.0 + math.exp(-val)) for val in data]
            return Tensor(new_data, self.getShape())
    
    def softmax(self, axis=-1):
        """Softmax activation"""
        if hasattr(self._tensor_cpp, 'softmax'):
            result = Tensor()
            result._tensor_cpp = self._tensor_cpp.softmax(axis)
            result._is_cpp = True
            return result
        else:
            # Fallback: apply softmax manually
            import math
            shape = self.getShape()
            data = self.to_list()
            
            if len(shape) == 2:
                batch_size, num_classes = shape
                new_data = [0.0] * len(data)
                
                for i in range(batch_size):
                    # Find max for numerical stability
                    start_idx = i * num_classes
                    end_idx = start_idx + num_classes
                    max_val = max(data[start_idx:end_idx])
                    
                    # Compute exponentials
                    exp_vals = []
                    sum_exp = 0.0
                    for j in range(num_classes):
                        exp_val = math.exp(data[start_idx + j] - max_val)
                        exp_vals.append(exp_val)
                        sum_exp += exp_val
                    
                    # Normalize
                    for j in range(num_classes):
                        new_data[start_idx + j] = exp_vals[j] / sum_exp
                
                return Tensor(new_data, shape)
            else:
                # Simple case
                max_val = max(data)
                exp_vals = [math.exp(val - max_val) for val in data]
                sum_exp = sum(exp_vals)
                new_data = [val / sum_exp for val in exp_vals]
                return Tensor(new_data, shape)
    
    def getTotalSize(self):
        """Get total number of elements"""
        return self._tensor_cpp.getTotalSize()
    
    def getData(self):
        """Get data as list"""
        return self._tensor_cpp.getData()
    
    def getShape(self):
        """Get shape as list"""
        return list(self.shape)
    
    @classmethod
    def zeros(cls, shape, requires_grad=False):
        """Create zero tensor"""
        tensor = cls(shape=shape, requires_grad=requires_grad)
        tensor._tensor_cpp.zeros()
        return tensor
    
    @classmethod
    def ones(cls, shape, requires_grad=False):
        """Create ones tensor"""
        tensor = cls(shape=shape, requires_grad=requires_grad)
        tensor._tensor_cpp.ones()
        return tensor
    
    @classmethod
    def randn(cls, shape, requires_grad=False):
        """Create random normal tensor"""
        tensor = cls(shape=shape, requires_grad=requires_grad)
        tensor._tensor_cpp.randomNormal()
        return tensor
    
    @classmethod
    def rand(cls, shape, requires_grad=False):
        """Create random uniform tensor"""
        tensor = cls(shape=shape, requires_grad=requires_grad)
        tensor._tensor_cpp.randomUniform()
        return tensor
    
    @classmethod
    def xavier_uniform(cls, shape, requires_grad=False):
        """Create tensor with Xavier uniform initialization"""
        tensor = cls(shape=shape, requires_grad=requires_grad)
        tensor._tensor_cpp.xavierUniform()
        return tensor


# Rest of the framework.py (Module, Sequential classes) remains the same...

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
        """Return all trainable parameters"""
        params = []
        # Collect parameters from this module
        for param in self._parameters.values():
            params.append(param)
        # Collect parameters from submodules
        for module in self._modules.values():
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def add_module(self, name, module):
        """Add a submodule"""
        self._modules[name] = module
    
    def add_parameter(self, name, parameter):
        """Add a parameter"""
        self._parameters[name] = parameter
    
    def save(self, path):
       
        import pickle
      
        param_data = {}
        for k, v in self._parameters.items():
            param_data[k] = {
                'data': v.getData(),
                'shape': v.getShape()
            }
        
        # Save module states
        module_states = {}
        for k, v in self._modules.items():
            if hasattr(v, 'state_dict'):
                module_states[k] = v.state_dict()
        
        state = {
            '_parameters': param_data,
            '_modules': module_states
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path):
        
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Load parameters
        for name, param_info in state.get('_parameters', {}).items():
            if name in self._parameters:
                data = param_info['data']
                shape = param_info['shape']
                self._parameters[name]._tensor_cpp = cpp.Tensor(data, shape)
        
        # Load submodules
        for name, module_state in state.get('_modules', {}).items():
            if name in self._modules and hasattr(self._modules[name], 'load_state_dict'):
                self._modules[name].load_state_dict(module_state)
    
    def state_dict(self):
      
        state = {}
        for k, v in self._parameters.items():
            state[k] = {
                'data': v.getData(),
                'shape': v.getShape()
            }
        return state
    
    def load_state_dict(self, state_dict):
        
        for name, param_info in state_dict.items():
            if name in self._parameters:
                data = param_info['data']
                shape = param_info['shape']
                self._parameters[name]._tensor_cpp = cpp.Tensor(data, shape)


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