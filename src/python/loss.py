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
            # Get data from tensors
            pred_data = predictions.getData()
            target_data = targets.getData()
            pred_shape = predictions.getShape()
            target_shape = targets.getShape()
            
            # Compute loss manually (fallback if C++ function doesn't exist)
            try:
                # Try to use C++ implementation
                loss_val = cpp.crossEntropyLoss(predictions._tensor_cpp, targets._tensor_cpp)
            except AttributeError:
                # Fallback to Python implementation
                loss_val = self._compute_loss_manually(pred_data, target_data, pred_shape)
            
            self.loss_value = loss_val
            
            # Create loss tensor
            loss_tensor = Tensor([loss_val], requires_grad=True)
            return loss_tensor
        
        else:
            # Fallback to manual computation
            pred_data = predictions.getData()
            target_data = targets.getData()
            pred_shape = predictions.getShape()
            
            loss_val = self._compute_loss_manually(pred_data, target_data, pred_shape)
            
            loss_tensor = Tensor([loss_val], requires_grad=True)
            return loss_tensor
    
    def _compute_loss_manually(self, pred_data, target_data, pred_shape):
        
        if len(pred_shape) == 1:
            batch_size = 1
            num_classes = pred_shape[0]
            pred_data = [pred_data]
            target_data = [target_data]
        else:
            batch_size = pred_shape[0]
            num_classes = pred_shape[1]
            
            # Reshape data for batch processing
            pred_data = [pred_data[i*num_classes:(i+1)*num_classes] for i in range(batch_size)]
            target_data = [target_data[i*num_classes:(i+1)*num_classes] for i in range(batch_size)]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            # Find true class
            true_class = 0
            max_label = target_data[i][0]
            for j in range(1, num_classes):
                if target_data[i][j] > max_label:
                    max_label = target_data[i][j]
                    true_class = j
            
            # Compute softmax
            # Find max for numerical stability
            max_val = max(pred_data[i])
            
            # Compute exponentials
            exp_vals = []
            for val in pred_data[i]:
                import math
                exp_vals.append(math.exp(val - max_val))
            
            # Compute sum of exponentials
            sum_exp = sum(exp_vals)
            
            # Compute probability for true class
            prob_true = exp_vals[true_class] / sum_exp
            
            # Compute cross-entropy loss
            # Add epsilon to avoid log(0)
            import math
            loss = -math.log(prob_true + 1e-8)
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        return avg_loss