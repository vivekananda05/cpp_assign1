import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from . import HAS_CPP_BACKEND
@dataclass
class TrainingConfig:
    """Training configuration"""
    # Required parameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Optional parameters with defaults
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    early_stopping_patience: int = 10
    lr_scheduler: bool = True
    lr_decay_factor: float = 0.1
    lr_decay_patience: int = 5
    
    # Additional fields that might be set later
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
    """Model trainer"""
    
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        
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
        
        # Model analysis
        self.model_stats = self._analyze_model()
    
    def _init_optimizer(self):
        """Initialize optimizer"""
        if HAS_CPP_BACKEND:
            import custom_dl_framework as cpp
            
            if self.config.optimizer.lower() == "sgd":
                self.optimizer = cpp.SGD(
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
            else:  # adam
                self.optimizer = cpp.Adam(
                    lr=self.config.learning_rate,
                    beta1=self.config.beta1,
                    beta2=self.config.beta2,
                    weight_decay=self.config.weight_decay
                )
        else:
            # Pure Python optimizer
            class PythonSGD:
                def __init__(self, params, lr, momentum=0, weight_decay=0):
                    self.params = params
                    self.lr = lr
                    self.momentum = momentum
                    self.weight_decay = weight_decay
                    self.velocities = [np.zeros_like(p.data) for p in params]
                
                def step(self):
                    for i, param in enumerate(self.params):
                        grad = param.grad
                        
                        # Apply weight decay
                        if self.weight_decay > 0:
                            grad += self.weight_decay * param.data
                        
                        # Update with momentum
                        if self.momentum > 0:
                            self.velocities[i] = (self.momentum * self.velocities[i] + 
                                                 self.lr * grad)
                            param.data -= self.velocities[i]
                        else:
                            param.data -= self.lr * grad
                
                def zero_grad(self):
                    for param in self.params:
                        param.grad = np.zeros_like(param.data)
            
            class PythonAdam:
                def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
                    self.params = params
                    self.lr = lr
                    self.beta1 = beta1
                    self.beta2 = beta2
                    self.eps = eps
                    self.weight_decay = weight_decay
                    self.m = [np.zeros_like(p.data) for p in params]
                    self.v = [np.zeros_like(p.data) for p in params]
                    self.t = 0
                
                def step(self):
                    self.t += 1
                    lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
                    
                    for i, param in enumerate(self.params):
                        grad = param.grad
                        
                        # Apply weight decay
                        if self.weight_decay > 0:
                            grad += self.weight_decay * param.data
                        
                        # Update moments
                        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)
                        
                        # Bias correction
                        m_hat = self.m[i] / (1 - self.beta1**self.t)
                        v_hat = self.v[i] / (1 - self.beta2**self.t)
                        
                        # Update parameters
                        param.data -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
                
                def zero_grad(self):
                    for param in self.params:
                        param.grad = np.zeros_like(param.data)
            
            params = self.model.parameters()
            if self.config.optimizer.lower() == "sgd":
                self.optimizer = PythonSGD(
                    params=params,
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
            else:
                self.optimizer = PythonAdam(
                    params=params,
                    lr=self.config.learning_rate,
                    beta1=self.config.beta1,
                    beta2=self.config.beta2,
                    weight_decay=self.config.weight_decay
                )
    
    # def _analyze_model(self) -> Dict:
    #     """Analyze model complexity"""
    #     total_params = 0
    #     total_macs = 0
    #     total_flops = 0
        
    #     # Calculate parameters for each layer
    #     for name, param in self.model._parameters.items():
    #         total_params += param.data.size
        
    #     # Estimate MACs and FLOPs (simplified)
    #     # In practice, would compute based on layer types and input shapes
    #     total_macs = total_params * 2  # Rough estimate
    #     total_flops = total_macs * 2   # Multiply-add counts as 2 FLOPS
        
    #     memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
    #     return {
    #         'total_parameters': total_params,
    #         'total_macs': total_macs,
    #         'total_flops': total_flops,
    #         'memory_mb': memory_mb
    #     }

    def _analyze_model(self) -> Dict:
        """Analyze model complexity"""
        total_params = 0
        
        # Try different ways to get parameters
        try:
            # Try accessing _parameters directly
            if hasattr(self.model, '_parameters'):
                for name, param in self.model._parameters.items():
                    total_params += param.data.size
            # Try getting parameters via parameters() method
            elif hasattr(self.model, 'parameters'):
                params = self.model.parameters()
                if params:
                    for param in params:
                        if hasattr(param, 'data'):
                            total_params += param.data.size
                        elif hasattr(param, 'shape'):
                            total_params += np.prod(param.shape)
            # Try to estimate from model structure
            elif hasattr(self.model, 'analyze'):
                # Use the model's own analyze method if available
                result = self.model.analyze(input_shape=(1, 3, 32, 32))
                return result
        except Exception as e:
            print(f"  Note: Could not analyze parameters: {e}")
            total_params = 10000  # Default estimate
        
        # Estimate MACs and FLOPs (simplified)
        total_macs = total_params * 2  # Rough estimate
        total_flops = total_macs * 2   # Multiply-add counts as 2 FLOPS
        
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return {
            'total_parameters': total_params,
            'total_macs': total_macs,
            'total_flops': total_flops,
            'memory_mb': memory_mb
        }
    
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss = self._compute_loss(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            predictions = self._get_predictions(outputs)
            true_labels = self._get_true_labels(labels)
            batch_correct = (predictions == true_labels).sum()
            
            # Update metrics
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += batch_correct
            total_samples += batch_size
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {batch_correct/batch_size:.2%}")
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels)
                
                predictions = self._get_predictions(outputs)
                true_labels = self._get_true_labels(labels)
                batch_correct = (predictions == true_labels).sum()
                
                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                total_correct += batch_correct
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _compute_loss(self, outputs, labels):
        """Compute cross-entropy loss"""
        if HAS_CPP_BACKEND:
            return cpp.crossEntropyLoss(outputs._tensor, labels._tensor)
        else:
            # Softmax and cross-entropy
            probs = self._softmax(outputs.data)
            loss = -np.sum(labels.data * np.log(probs + 1e-8)) / outputs.shape[0]
            
            # Create loss tensor
            from framework import Tensor
            loss_tensor = Tensor([loss])
            return loss_tensor
    
    def _softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _get_predictions(self, outputs):
        """Get predicted class indices"""
        if HAS_CPP_BACKEND:
            outputs_data = outputs.numpy()
        else:
            outputs_data = outputs.data
        
        return np.argmax(outputs_data, axis=1)
    
    def _get_true_labels(self, labels):
        """Get true class indices from one-hot encoding"""
        if HAS_CPP_BACKEND:
            labels_data = labels.numpy()
        else:
            labels_data = labels.data
        
        return np.argmax(labels_data, axis=1)
    
    def fit(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {self.model_stats['total_parameters']:,}")
        print(f"  Total MACs: {self.model_stats['total_macs']:,}")
        print(f"  Total FLOPs: {self.model_stats['total_flops']:,}")
        print(f"  Memory: {self.model_stats['memory_mb']:.2f} MB")
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Optimizer: {self.config.optimizer}")
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
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
                self.model.save(str(self.best_model_path))
                self.patience_counter = 0
                print(f"  ✓ Saved best model to {self.best_model_path}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Learning rate scheduling
            if self.config.lr_scheduler and len(self.history['val_loss']) > 1:
                if self.history['val_loss'][-1] > self.history['val_loss'][-2]:
                    self.config.learning_rate *= self.config.lr_decay_factor
                    print(f"  Learning rate decreased to {self.config.learning_rate}")
        
        # Save final model
        final_model_path = self.save_dir / "final_model.pth"
        self.model.save(str(final_model_path))
        print(f"\nSaved final model to {final_model_path}")
        
        # Save training history
        self._save_history()
        
        # Plot training curves
        self._plot_training_curves()
        
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
    
    def _plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        
        # Epoch times
        axes[1, 1].plot(epochs, self.history['epoch_times'], 'm-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.log_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_file}")

# Import torch for tensor operations (only for type hints)
try:
    import torch
except ImportError:
    torch = None