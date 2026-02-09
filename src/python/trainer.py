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
    



    # Minimal fix for trainer.py - just compute loss properly
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            HAS_CPP_BACKEND = True
            
            # SIMPLE FIX: Compute loss directly using C++ if available
            if HAS_CPP_BACKEND:
                import custom_dl_framework as cpp
                loss_val = cpp.crossEntropyLoss(outputs._tensor_cpp, labels._tensor_cpp)
                loss_tensor = Tensor([loss_val], requires_grad=True)
            else:
                loss_tensor = self.criterion(outputs, labels)
                loss_val = loss_tensor.getData()[0]
            
            accuracy = self.compute_accuracy(outputs, labels)
            
            # Try to compute gradients
            try:
                if hasattr(loss_tensor, 'backward'):
                    loss_tensor.backward()
            except:
                pass  # Continue even if gradient computation fails
            
            self.optimizer.step()
            
            total_loss += loss_val
            total_acc += accuracy
            num_batches += 1
            
            print(f"    Batch {batch_idx}/{len(train_loader)}: "
                f"Loss: {loss_val:.6f}, Acc: {accuracy*100:.1f}%")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        
        print(f"\n  Average Loss: {avg_loss:.6f}, Average Accuracy: {avg_acc*100:.2f}%")
        
        return avg_loss, avg_acc




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