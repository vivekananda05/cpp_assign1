# src/python/dataloader.py
import os
import random
import time
from typing import List, Tuple, Dict, Optional, Iterator
from pathlib import Path

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
            
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    self.samples.append((str(img_path), class_idx))
        
        self.load_time = time.time() - start_time
        print(f"✓ Dataset loaded in {self.load_time:.2f} seconds")
        print(f"✓ Found {len(self.samples)} images in {len(self.classes)} classes")
    
    def _load_image(self, img_path: str):
        """Load and preprocess a single image"""
        try:
            # Try to use PIL if available
            from PIL import Image
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((self.image_size[1], self.image_size[0]))
                img_data = list(img.getdata())
                width, height = img.size
                
                # Convert to 2D list
                img_2d = []
                for h in range(height):
                    row = []
                    for w in range(width):
                        pixel = img_data[h * width + w]
                        row.append(list(pixel))
                    img_2d.append(row)
        except ImportError:
            # PIL not available, create dummy image
            print(f"⚠ PIL not available, creating dummy image for {img_path}")
            height, width = self.image_size
            img_2d = [[[random.randint(0, 255) for _ in range(3)] 
                      for _ in range(width)] 
                     for _ in range(height)]
        except Exception as e:
            # Any error, create dummy image
            print(f"⚠ Error loading image {img_path}: {e}, creating dummy image")
            height, width = self.image_size
            img_2d = [[[random.randint(0, 255) for _ in range(3)] 
                      for _ in range(width)] 
                     for _ in range(height)]
        
        # Convert to CHW format [channels, height, width]
        height = len(img_2d)
        width = len(img_2d[0])
        channels = 3
        
        chw_img = []
        for c in range(channels):
            channel_data = []
            for h in range(height):
                row = []
                for w in range(width):
                    row.append(float(img_2d[h][w][c]) / 255.0)
                channel_data.append(row)
            chw_img.append(channel_data)
        
        # Normalize if requested
        if self.normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        chw_img[c][h][w] = (chw_img[c][h][w] - mean[c]) / std[c]
        
        # Data augmentation
        if self.augment and random.random() > 0.5:
            for c in range(channels):
                for h in range(height):
                    chw_img[c][h] = chw_img[c][h][::-1]
        
        return chw_img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image_data = self._load_image(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            height, width = self.image_size
            image_data = [[[random.random() for _ in range(width)] 
                          for _ in range(height)] 
                         for _ in range(3)]
        
        # Create label tensor with one-hot encoding
        num_classes = len(self.classes)
        label_onehot = [0.0] * num_classes
        label_onehot[label] = 1.0
        
        # Return as lists, not tensors (tensors will be created in DataLoader)
        return image_data, label_onehot
    
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
        
        # Create subset datasets
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
    """Simple data loader that creates Tensor objects"""
    
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
        """Get next batch as Tensor objects"""
        if self.current_idx >= len(self.indices):
            raise StopIteration
        
        # Get batch indices
        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        
        # Load batch data
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            image_data, label_data = self.dataset[idx]
            batch_images.append(image_data)
            batch_labels.append(label_data)
        
        # Convert to proper tensor format
        # Image data is currently [channels, height, width] for each sample
        # We need to create a 4D tensor: [batch_size, channels, height, width]
        
        # First, flatten the image data to a 1D list for each sample
        batch_size = len(batch_images)
        height, width = self.dataset.image_size
        channels = 3
        
        # Flatten batch data to 1D list
        flattened_images = []
        for img in batch_images:
            # img is [channels, height, width]
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        flattened_images.append(img[c][h][w])
        
        # Create shape: [batch_size, channels, height, width]
        tensor_shape = [batch_size, channels, height, width]
        
        # Create image tensor
        batch_images_tensor = Tensor(flattened_images, tensor_shape)
        
        # Create labels tensor
        # Labels are already one-hot encoded lists
        # We need to flatten them to 1D list
        flattened_labels = []
        for label in batch_labels:
            flattened_labels.extend(label)
        
        # Create shape: [batch_size, num_classes]
        num_classes = len(batch_labels[0]) if batch_labels else len(self.dataset.classes)
        label_shape = [batch_size, num_classes]
        batch_labels_tensor = Tensor(flattened_labels, label_shape)
        
        # Debug: Print tensor shapes for first batch
        if not hasattr(self, '_printed_shapes'):
            print(f"\nBatch tensor shapes:")
            print(f"  Images shape: {batch_images_tensor.getShape()}")
            print(f"  Labels shape: {batch_labels_tensor.getShape()}")
            print(f"  Image data length: {len(flattened_images)}")
            print(f"  Label data length: {len(flattened_labels)}")
            self._printed_shapes = True
        
        return batch_images_tensor, batch_labels_tensor

