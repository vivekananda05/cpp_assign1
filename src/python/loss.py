

from .framework import Tensor
from . import HAS_CPP_BACKEND

if HAS_CPP_BACKEND:
    import custom_dl_framework as cpp

class CrossEntropyLoss:
    """Cross-entropy loss using C++ backend"""
    
    def __init__(self):
        self.loss_value = 0.0
        
    def __call__(self, predictions, targets):
        """Compute cross-entropy loss"""
        if HAS_CPP_BACKEND:

            loss_val = cpp.crossEntropyLoss(predictions._tensor_cpp, targets._tensor_cpp)
        
        self.loss_value = loss_val
        loss_tensor = Tensor([loss_val], requires_grad=True)
        return loss_tensor
    
