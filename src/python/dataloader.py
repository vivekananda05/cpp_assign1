import os
import cv2
import random
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Iterator
from pathlib import Path
import threading
from queue import Queue

# Import Tensor from the same module
from .framework import Tensor

class ImageDataset:
    """Dataset loader for image classification"""
    
    def __init__(self, root_dir: str, image_size: Tuple[int, int] = (32, 32),
                 augment: bool = False, normalize: bool = True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        # Load dataset
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load images and labels from directory structure"""
        start_time = time.time()
        
        # Get class names from subdirectories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() 
                              if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), class_idx))
        
        self.load_time = time.time() - start_time
        print(f"✓ Dataset loaded in {self.load_time:.2f} seconds")
        print(f"✓ Found {len(self.samples)} images in {len(self.classes)} classes")
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Data augmentation (simplified)
        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)  # Horizontal flip
        
        # Convert to tensor [C, H, W]
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        if self.normalize:
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img = (img - mean) / std
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image_data = self._load_image(img_path)
        
        # Convert label to one-hot encoding
        label_onehot = np.zeros(len(self.classes), dtype=np.float32)
        label_onehot[label] = 1.0
        
        return Tensor(image_data), Tensor(label_onehot)
    
    def split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
              shuffle=True, seed=42):
        """Split dataset into train, validation, and test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.samples)
        
        n_total = len(self.samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_samples = self.samples[:n_train]
        val_samples = self.samples[n_train:n_train + n_val]
        test_samples = self.samples[n_train + n_val:]
        
        # Create separate datasets
        train_dataset = self._create_subset(train_samples)
        val_dataset = self._create_subset(val_samples)
        test_dataset = self._create_subset(test_samples)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, samples):
        """Create a subset dataset"""
        subset = ImageDataset.__new__(ImageDataset)
        subset.root_dir = self.root_dir
        subset.image_size = self.image_size
        subset.augment = self.augment
        subset.normalize = self.normalize
        subset.classes = self.classes
        subset.class_to_idx = self.class_to_idx
        subset.samples = samples
        subset.load_time = self.load_time
        return subset

class DataLoader:
    """Simple data loader"""
    
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        """Return number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Create iterator"""
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Get next batch"""
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        # Get batch indices
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        
        # Load batch
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            image, label = self.dataset[idx]
            batch_images.append(image.data)
            batch_labels.append(label.data)
        
        # Stack and return
        batch_images_tensor = Tensor(np.stack(batch_images))
        batch_labels_tensor = Tensor(np.stack(batch_labels))
        
        return batch_images_tensor, batch_labels_tensor
    
    # Alternative: Generator version
    def batches(self):
        """Generator for batches"""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                image, label = self.dataset[idx]
                batch_images.append(image.data)
                batch_labels.append(label.data)
            
            yield Tensor(np.stack(batch_images)), Tensor(np.stack(batch_labels))