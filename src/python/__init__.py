"""
Custom DL Framework - Python Package
"""

import sys
import os

# Add the C++ module path if available
cpp_build_path = os.path.join(os.path.dirname(__file__), '../../build')
if os.path.exists(cpp_build_path):
    sys.path.insert(0, cpp_build_path)

# Try to import C++ backend
try:
    import custom_dl_framework as cpp
    HAS_CPP_BACKEND = True
    print("✓ Custom DL Framework: Using C++ backend")
except ImportError:
    HAS_CPP_BACKEND = False
    print("⚠ Custom DL Framework: Using pure Python implementation")

# Export modules
from .framework import Tensor, Module, Sequential
from .dataloader import ImageDataset, DataLoader
from .trainer import Trainer, TrainingConfig
from .evaluator import Evaluator

__version__ = "1.0.0"
__all__ = [
    'Tensor', 'Module', 'Sequential',
    'ImageDataset', 'DataLoader',
    'Trainer', 'TrainingConfig',
    'Evaluator',
    'HAS_CPP_BACKEND'
]