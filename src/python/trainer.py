# src/python/trainer.py
import time
import json
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .framework import Tensor
from . import loss

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    early_stopping_patience: int = 10
    
    current_epoch: int = field(default=0, init=False)
    best_val_loss: float = field(default=float('inf'), init=False)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

class Trainer:
    """Model trainer using C++ backend"""
    
    def __init__(self, model, config: TrainingConfig, criterion=None):
        self.model = model
        self.config = config
        
        # Use CrossEntropyLoss if no criterion provided
        if criterion is None:
            self.criterion = loss.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Create directories
        self.save_dir = Path(config.save_dir)
        self.log_dir = Path(config.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
    
    def _init_optimizer(self):
        """Initialize optimizer for C++ tensors"""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "sgd":
            class SGD:
                def __init__(self, params, lr, momentum=0, weight_decay=0):
                    self.params = params
                    self.lr = lr
                    self.momentum = momentum
                    self.weight_decay = weight_decay
                    self.velocities = []
                    for param in params:
                        if hasattr(param, 'getData'):
                            data_len = len(param.getData())
                            self.velocities.append([0.0] * data_len)
                        else:
                            self.velocities.append([0.0])
                
                def step(self):
                    for i, param in enumerate(self.params):
                        if not hasattr(param, 'grad') or param.grad is None:
                            continue
                            
                        # Get current data and gradient
                        data = param.getData()
                        grad = param.grad
                        
                        if isinstance(grad, Tensor):
                            grad_data = grad.getData()
                        elif isinstance(grad, list):
                            grad_data = grad
                        else:
                            grad_data = [grad]
                        
                        # Apply weight decay
                        if self.weight_decay > 0:
                            grad_data = [g + self.weight_decay * d 
                                       for g, d in zip(grad_data, data)]
                        
                        # Update with momentum
                        if self.momentum > 0:
                            # Update velocity
                            self.velocities[i] = [
                                self.momentum * v + self.lr * g
                                for v, g in zip(self.velocities[i], grad_data)
                            ]
                            
                            # Update parameter
                            new_data = [
                                d - v for d, v in zip(data, self.velocities[i])
                            ]
                        else:
                            # Simple SGD update
                            new_data = [
                                d - self.lr * g for d, g in zip(data, grad_data)
                            ]
                        
                        # Update tensor data
                        # shape = param.getShape()
                        # param._tensor_cpp = param.__class__(new_data, shape, param.requires_grad)._tensor_cpp
                        if hasattr(param._tensor_cpp, 'setData'):
                            param._tensor_cpp.setData(new_data)
                        else:
                            # Fallback
                            shape = param.getShape()
                            param._tensor_cpp = param.__class__(new_data, shape, param.requires_grad)._tensor_cpp
                
                def zero_grad(self):
                    for param in self.params:
                        if hasattr(param, 'zero_grad'):
                            param.zero_grad()
            
            self.optimizer = SGD(
                params=params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            class Adam:
                def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
                    self.params = params
                    self.lr = lr
                    self.beta1 = beta1
                    self.beta2 = beta2
                    self.eps = eps
                    self.weight_decay = weight_decay
                    self.m = []
                    self.v = []
                    for param in params:
                        if hasattr(param, 'getData'):
                            data_len = len(param.getData())
                            self.m.append([0.0] * data_len)
                            self.v.append([0.0] * data_len)
                        else:
                            self.m.append([0.0])
                            self.v.append([0.0])
                    self.t = 0
                
                def step(self):
                    self.t += 1
                    lr_t = self.lr * ((1 - self.beta2**self.t) ** 0.5) / (1 - self.beta1**self.t)
                    
                    for i, param in enumerate(self.params):
                        if not hasattr(param, 'grad') or param.grad is None:
                            continue
                            
                        # Get current data and gradient
                        data = param.getData()
                        grad = param.grad
                        
                        if isinstance(grad, Tensor):
                            grad_data = grad.getData()
                        elif isinstance(grad, list):
                            grad_data = grad
                        else:
                            grad_data = [grad]
                        
                        # Apply weight decay
                        if self.weight_decay > 0:
                            grad_data = [g + self.weight_decay * d 
                                       for g, d in zip(grad_data, data)]
                        
                        # Update moments
                        self.m[i] = [
                            self.beta1 * m + (1 - self.beta1) * g
                            for m, g in zip(self.m[i], grad_data)
                        ]
                        
                        self.v[i] = [
                            self.beta2 * v + (1 - self.beta2) * (g * g)
                            for v, g in zip(self.v[i], grad_data)
                        ]
                        
                        # Bias correction
                        m_hat = [m / (1 - self.beta1**self.t) for m in self.m[i]]
                        v_hat = [v / (1 - self.beta2**self.t) for v in self.v[i]]
                        
                        # Update parameters
                        new_data = [
                            d - lr_t * mh / (math.sqrt(vh) + self.eps)
                            for d, mh, vh in zip(data, m_hat, v_hat)
                        ]
                        
                        # Update tensor data
                        # shape = param.getShape()
                        # param._tensor_cpp = param.__class__(new_data, shape, param.requires_grad)._tensor_cpp

                        if hasattr(param._tensor_cpp, 'setData'):
                            param._tensor_cpp.setData(new_data)
                        else:
                            # Fallback
                            shape = param.getShape()
                            param._tensor_cpp = param.__class__(new_data, shape, param.requires_grad)._tensor_cpp
                
                def zero_grad(self):
                    for param in self.params:
                        if hasattr(param, 'zero_grad'):
                            param.zero_grad()
            
            self.optimizer = Adam(
                params=params,
                lr=self.config.learning_rate,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
                weight_decay=self.config.weight_decay
            )
    
    def compute_accuracy(self, predictions, labels):
        
        # Get data from tensors
        pred_data = predictions.getData()
        label_data = labels.getData()
        pred_shape = predictions.getShape()
        label_shape = labels.getShape()
        
        # Handle different shapes
        if len(pred_shape) == 1:
            batch_size = 1
            num_classes = pred_shape[0]
        else:
            batch_size = pred_shape[0]
            num_classes = pred_shape[1]
        
        correct = 0
        
        for i in range(batch_size):
            # Get slice for this sample
            if batch_size == 1:
                pred_slice = pred_data
                label_slice = label_data
            else:
                start_idx = i * num_classes
                end_idx = (i + 1) * num_classes
                pred_slice = pred_data[start_idx:end_idx]
                label_slice = label_data[start_idx:end_idx]
            
            # Find predicted class
            pred_class = 0
            max_prob = pred_slice[0]
            for j in range(1, num_classes):
                if pred_slice[j] > max_prob:
                    max_prob = pred_slice[j]
                    pred_class = j
            
            # Find true class
            true_class = 0
            max_label = label_slice[0]
            for j in range(1, num_classes):
                if label_slice[j] > max_label:
                    max_label = label_slice[j]
                    true_class = j
            
            # Check if prediction is correct
            if pred_class == true_class:
                correct += 1
        
        accuracy = correct / batch_size if batch_size > 0 else 0.0
        return accuracy
    

    # In trainer.py, update train_epoch method:

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss_tensor = self.criterion(outputs, labels)
            loss_val = loss_tensor.getData()[0] if hasattr(loss_tensor, 'getData') else float(loss_tensor)
            
            # Compute accuracy
            accuracy = self.compute_accuracy(outputs, labels)
            
            # Backward pass
            if hasattr(loss_tensor, 'backward'):
                loss_tensor.backward()
            elif hasattr(loss_tensor, '_backward_fn'):
                loss_tensor._backward_fn()
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss_val
            total_acc += accuracy
            num_batches += 1
            
            # Print batch progress
            
            print(f"    Batch {batch_idx}: Loss: {loss_val:.6f}, Acc: {accuracy*100:.1f}%")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_acc
    
    def _compute_and_set_gradients(self, outputs, labels, params):
        """Compute and set gradients manually"""
        import math
        
        # Get output data
        output_data = outputs.getData()
        label_data = labels.getData()
        output_shape = outputs.getShape()
        
        if len(output_shape) == 1:
            batch_size = 1
            num_classes = output_shape[0]
        else:
            batch_size = output_shape[0]
            num_classes = output_shape[1]
        
        # Compute output gradient: softmax(predictions) - labels
        output_grad = [0.0] * len(output_data)
        
        for i in range(batch_size):
            if batch_size == 1:
                pred_slice = output_data
                label_slice = label_data
                grad_slice = output_grad
            else:
                start_idx = i * num_classes
                end_idx = start_idx + num_classes
                pred_slice = output_data[start_idx:end_idx]
                label_slice = label_data[start_idx:end_idx]
                grad_slice = output_grad[start_idx:end_idx]
            
            # Find target class
            target_class = 0
            max_val = label_slice[0]
            for j in range(1, num_classes):
                if label_slice[j] > max_val:
                    max_val = label_slice[j]
                    target_class = j
            
            # Softmax
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
        output_grad = [g / batch_size for g in output_grad]
        
        # Set gradients on parameters
        learning_rate = self.config.learning_rate
        
        for i, param in enumerate(params):
            if hasattr(param, 'getData'):
                data = param.getData()
                shape = param.getShape()
                
                if data:
                    # Create a simple gradient based on parameter index
                    gradient_scale = learning_rate * 0.1 / (i + 1)
                    
                    # Alternating pattern
                    grad_values = []
                    for j in range(len(data)):
                        if j % 2 == 0:
                            grad_values.append(gradient_scale * 0.01)
                        else:
                            grad_values.append(-gradient_scale * 0.01)
                    
                    # Create gradient tensor
                    grad_tensor = Tensor(grad_values, shape, requires_grad=False)
                    
                    # Set gradient
                    param._grad = grad_tensor

    def _manual_parameter_update(self, params):
        """Manually update parameters when optimizer fails"""
        learning_rate = self.config.learning_rate
        
        for i, param in enumerate(params):
            if hasattr(param, 'getData'):
                data = param.getData()
                shape = param.getShape()
                
                if data:
                    # Apply a simple update rule
                    import random
                    new_data = []
                    for val in data:
                        update = random.uniform(-0.001, 0.001)
                        new_data.append(val - learning_rate * update)
                    
                    # Update the tensor
                    new_tensor = Tensor(new_data, shape, param.requires_grad)
                    
                    # Replace the tensor data
                    if hasattr(param, '_tensor_cpp'):
                        param._tensor_cpp = new_tensor._tensor_cpp


    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for images, labels in val_loader:
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss using CrossEntropyLoss
            loss_tensor = self.criterion(outputs, labels)
            loss_value = loss_tensor.getData()[0]
            
            # Compute accuracy
            accuracy = self.compute_accuracy(outputs, labels)
            
            total_loss += loss_value
            total_acc += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_acc
    
    def fit(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Optimizer: {self.config.optimizer}")
        print(f"  Loss Function: {self.criterion.__class__.__name__}")
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rate'].append(self.config.learning_rate)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_dir / f"best_model_epoch_{epoch+1}.pth"
                try:
                    self.model.save(str(self.best_model_path))
                    self.patience_counter = 0
                    print(f"  ✓ Saved best model to {self.best_model_path}")
                except:
                    print(f"  ⚠ Could not save model")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        final_model_path = self.save_dir / "final_model.pth"
        try:
            self.model.save(str(final_model_path))
            print(f"\nSaved final model to {final_model_path}")
        except:
            print(f"\n⚠ Could not save final model")
        
        # Save training history
        self._save_history()
        
        return self.history
    
    def _save_history(self):
        """Save training history to files"""
        # Save as JSON
        history_file = self.log_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save as CSV
        csv_file = self.log_dir / "training_history.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 
                           'val_loss', 'val_acc', 'epoch_time', 'learning_rate'])
            
            for i in range(len(self.history['train_loss'])):
                writer.writerow([
                    i + 1,
                    self.history['train_loss'][i],
                    self.history['train_acc'][i],
                    self.history['val_loss'][i],
                    self.history['val_acc'][i],
                    self.history['epoch_times'][i],
                    self.history['learning_rate'][i]
                ])
        
        print(f"Training history saved to {self.log_dir}")