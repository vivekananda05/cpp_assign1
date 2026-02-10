

# loss.py
from .framework import Tensor
from . import HAS_CPP_BACKEND

if HAS_CPP_BACKEND:
    import custom_dl_framework as cpp

class CrossEntropyLoss:
    """Cross-entropy loss using C++ backend"""
    
    def __init__(self):
        self.predictions = None
        self.targets = None
    
    def __call__(self, predictions, targets):
        """Compute cross-entropy loss"""
        # Store references for backward pass
        self.predictions = predictions
        self.targets = targets
        
        if HAS_CPP_BACKEND:
            loss_val = cpp.crossEntropyLoss(predictions._tensor_cpp, targets._tensor_cpp)
        else:
            loss_val = self._compute_loss_python(predictions, targets)
        
        # Create loss tensor
        loss_tensor = Tensor([loss_val], [1], requires_grad=True)
        
        # Store backward function
        loss_tensor._backward_fn = self._backward
        
        return loss_tensor
    
    def _compute_loss_python(self, predictions, targets):
        """Python implementation of cross-entropy loss"""
        import math
        
        pred_data = predictions.getData()
        target_data = targets.getData()
        shape = predictions.getShape()
        
        if len(shape) == 1:
            batch_size = 1
            num_classes = shape[0]
        else:
            batch_size = shape[0]
            num_classes = shape[1]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            if batch_size == 1:
                pred_slice = pred_data
                target_slice = target_data
            else:
                start_idx = i * num_classes
                end_idx = start_idx + num_classes
                pred_slice = pred_data[start_idx:end_idx]
                target_slice = target_data[start_idx:end_idx]
            
            # Find target class
            target_class = 0
            max_val = target_slice[0]
            for j in range(1, num_classes):
                if target_slice[j] > max_val:
                    max_val = target_slice[j]
                    target_class = j
            
            # Softmax for numerical stability
            max_logit = max(pred_slice)
            exp_vals = [math.exp(p - max_logit) for p in pred_slice]
            sum_exp = sum(exp_vals)
            probs = [e / sum_exp for e in exp_vals]
            
            # Cross-entropy
            loss = -math.log(probs[target_class] + 1e-10)
            total_loss += loss
        
        return total_loss / batch_size
    
    def _backward(self):
        """Backward pass - compute gradients"""
        if self.predictions is None:
            return
        
        pred_data = self.predictions.getData()
        target_data = self.targets.getData()
        shape = self.predictions.getShape()
        
        if len(shape) == 1:
            batch_size = 1
            num_classes = shape[0]
        else:
            batch_size = shape[0]
            num_classes = shape[1]
        
        # Compute gradient
        import math
        grad_data = [0.0] * len(pred_data)
        
        for i in range(batch_size):
            if batch_size == 1:
                pred_slice = pred_data
                target_slice = target_data
                grad_slice = grad_data
            else:
                start_idx = i * num_classes
                end_idx = start_idx + num_classes
                pred_slice = pred_data[start_idx:end_idx]
                target_slice = target_data[start_idx:end_idx]
                grad_slice = grad_data[start_idx:end_idx]
            
            # Find target class
            target_class = 0
            max_val = target_slice[0]
            for j in range(1, num_classes):
                if target_slice[j] > max_val:
                    max_val = target_slice[j]
                    target_class = j
            
            # Compute softmax
            max_logit = max(pred_slice)
            exp_vals = [math.exp(p - max_logit) for p in pred_slice]
            sum_exp = sum(exp_vals)
            probs = [e / sum_exp for e in exp_vals]
            
            # Gradient: p - y
            for j in range(num_classes):
                grad_slice[j] = probs[j]
                if j == target_class:
                    grad_slice[j] -= 1.0
        
        # Scale by batch size
        grad_data = [g / batch_size for g in grad_data]
        
        # Create gradient tensor
        grad_tensor = Tensor(grad_data, shape, requires_grad=False)
        
        # Set gradient on predictions
        if hasattr(self.predictions, '_grad'):
            self.predictions._grad = grad_tensor