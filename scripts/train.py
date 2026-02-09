#!/usr/bin/env python3
"""
Training script for custom CNN framework
"""

import os
import sys
import argparse
import time
import random
import json
from pathlib import Path

# ============================================================================
# SETUP PATHS
# ============================================================================

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add all necessary paths to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'python'))
sys.path.insert(0, str(project_root / 'models'))
sys.path.insert(0, str(project_root / 'build'))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")

# ============================================================================
# DETECT C++ BACKEND
# ============================================================================

# First try to import the C++ backend directly
try:
    import custom_dl_framework as cpp
    print(f"✓ Successfully imported C++ backend: custom_dl_framework")
    HAS_CPP_BACKEND = True
except ImportError as e:
    print(f"✗ Could not import C++ backend: {e}")
    HAS_CPP_BACKEND = False

# ============================================================================
# IMPORT PYTHON MODULES
# ============================================================================

# Import framework
try:
    # First, patch HAS_CPP_BACKEND if we detected it
    if HAS_CPP_BACKEND:
        # Create/update __init__ file
        init_path = project_root / 'src' / 'python' / '__init__.py'
        with open(init_path, 'w') as f:
            f.write(f"HAS_CPP_BACKEND = {HAS_CPP_BACKEND}\n")
        print(f"✓ Updated HAS_CPP_BACKEND = {HAS_CPP_BACKEND}")
    
    from src.python.framework import Tensor
    print(f"✓ Successfully imported Tensor from framework.py")
    
except ImportError as e:
    print(f"✗ Error importing framework: {e}")
    sys.exit(1)

# Import other modules
try:
    from src.python.dataloader import ImageDataset, DataLoader
    print(f"✓ Successfully imported ImageDataset and DataLoader")
except ImportError as e:
    print(f"✗ Error importing dataloader: {e}")
    sys.exit(1)

try:
    from src.python.trainer import Trainer, TrainingConfig
    print(f"✓ Successfully imported Trainer and TrainingConfig")
except ImportError as e:
    print(f"✗ Error importing trainer: {e}")
    sys.exit(1)

try:
    from models.simple_cnn import SimpleCNN
    print(f"✓ Successfully imported SimpleCNN from models.simple_cnn")
except ImportError as e:
    print(f"✗ Error importing SimpleCNN: {e}")
    sys.exit(1)

# ============================================================================
# MAIN TRAINING CODE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN model using custom framework')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Input image size (default: 32)')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--conv1_channels', type=int, default=32,
                       help='Channels in first conv layer (default: 32)')
    parser.add_argument('--conv2_channels', type=int, default=64,
                       help='Channels in second conv layer (default: 64)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size (default: 128)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam'],
                       help='Optimizer (default: adam)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay (default: 0.0001)')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs (default: ./logs)')
    
    # Other arguments
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directories
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. LOAD DATASET
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: LOADING DATASET")
    print('='*60)
    
    try:
        print(f"Loading dataset from: {args.data_dir}")
        start_time = time.time()
        
        # Check if data directory exists
        if not os.path.exists(args.data_dir):
            print(f"✗ Data directory '{args.data_dir}' does not exist.")
            print("Please provide a valid data directory with the structure:")
            print("  data_dir/")
            print("  ├── class1/")
            print("  │   ├── image1.jpg")
            print("  │   └── image2.jpg")
            print("  ├── class2/")
            print("  │   ├── image1.jpg")
            print("  │   └── image2.jpg")
            print("  └── ...")
            sys.exit(1)
        
        # Create dataset
        dataset = ImageDataset(
            root_dir=args.data_dir,
            image_size=(args.image_size, args.image_size),
            augment=args.augment,
            normalize=True
        )
        
        load_time = time.time() - start_time
        print(f"✓ Dataset loaded in {load_time:.2f} seconds")
        
        # Get number of classes
        if hasattr(dataset, 'classes'):
            print(f"✓ Found {len(dataset.classes)} classes")
            if args.num_classes is None:
                args.num_classes = len(dataset.classes)
                print(f"✓ Auto-detected num_classes: {args.num_classes}")
        
        if hasattr(dataset, 'samples'):
            print(f"✓ Found {len(dataset.samples)} images")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the data directory exists")
        print("2. Ensure the directory structure is: data_dir/class_name/*.jpg")
        print("3. Check file permissions")
        print("4. Make sure PIL/Pillow is installed: pip install pillow")
        sys.exit(1)
    
    # ========================================================================
    # 2. SPLIT DATASET AND CREATE DATA LOADERS
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: PREPARING DATA LOADERS")
    print('='*60)
    
    try:
        # Split dataset
        train_dataset, val_dataset, test_dataset = dataset.split(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            shuffle=True,
            seed=args.seed
        )
        
        print(f"✓ Dataset split:")
        print(f"  Training: {len(train_dataset)} images")
        print(f"  Validation: {len(val_dataset)} images")
        print(f"  Test: {len(test_dataset)} images")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        print(f"✓ Data loaders created")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
    except Exception as e:
        print(f"✗ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # 3. CREATE MODEL
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: CREATING MODEL")
    print('='*60)
    
    model_config = {
        'conv1_channels': args.conv1_channels,
        'conv2_channels': args.conv2_channels,
        'hidden_size': args.hidden_size,
        'dropout_rate': args.dropout_rate
    }
    
    print(f"Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    try:
        # Create model - using the CNNModel from cnn_model.py
        model = SimpleCNN(
            num_classes=args.num_classes,
            config=model_config
        )
        
        print(f"✓ CNN model created successfully")
        print(f"  Model type: {model.__class__.__name__}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # 4. ANALYZE MODEL
    # ========================================================================
    try:
        model_stats = model.analyze(input_shape=(1, 3, args.image_size, args.image_size))
        print(f"\nModel Analysis:")
        print(f"  Total Parameters: {model_stats.get('total_parameters', 'N/A'):,}")
        print(f"  Total MACs: {model_stats.get('total_macs', 'N/A'):,}")
        print(f"  Total FLOPs: {model_stats.get('total_flops', 'N/A'):,}")
        print(f"  Memory: {model_stats.get('memory_mb', 'N/A'):.2f} MB")
    except Exception as e:
        print(f"\n⚠ Model analysis error: {e}")
        model_stats = {}
    
    # ========================================================================
    # 5. CREATE TRAINER AND TRAIN
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: TRAINING MODEL")
    print('='*60)
    
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Batch Size: {args.batch_size}")
    
    try:
        # Create training configuration
        train_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            save_dir=str(save_dir),
            log_dir=str(log_dir)
        )
        
        print(f"✓ Training config created")
        
        # Create trainer
        trainer = Trainer(model, train_config)
        print(f"✓ Trainer created successfully")
        
        # Test forward pass

        # In the training section of train3.py, add this debug code:

        # Test forward pass
        print(f"\nTesting forward pass on one batch...")
        try:
            test_images, test_labels = next(iter(train_loader))
            
            print(f"  Input type: {type(test_images)}")
            print(f"  Input shape: {test_images.getShape()}")
            print(f"  Input data length: {len(test_images.getData())}")
            
            # Check if we need to reshape
            shape = test_images.getShape()
            if len(shape) == 2:
                batch_size, features = shape
                print(f"  Input is 2D: batch_size={batch_size}, features={features}")
                
                # For CIFAR-10, features should be 3072 (32*32*3)
                if features == 3072:
                    print(f"  ✓ Looks like CIFAR-10 data (32x32x3)")
                elif features == 1024:
                    print(f"  ✓ Looks like grayscale data (32x32)")
            
            # Test forward pass
            test_outputs = model(test_images)
            
            print(f"  Output type: {type(test_outputs)}")
            print(f"  Output shape: {test_outputs.getShape()}")
            print(f"  ✓ Forward pass test successful")
            
        except Exception as e:
            print(f"  ⚠ Forward pass test failed: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing with training anyway...")
        
        # Start training
        print(f"\nStarting training...")
        start_time = time.time()
        
        history = trainer.fit(train_loader, val_loader)
        
        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time:.2f} seconds!")
        
        # Show training results
        if hasattr(trainer, 'best_val_loss'):
            print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
        
        if hasattr(trainer, 'best_model_path') and trainer.best_model_path:
            print(f"  Best model saved to: {trainer.best_model_path}")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # 6. SAVE TRAINING REPORT
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 5: SAVING RESULTS")
    print('='*60)
    
    # Prepare report
    report = {
        'training_args': vars(args),
        'model_config': model_config,
        'model_stats': model_stats,
        'cpp_backend_available': HAS_CPP_BACKEND,
        'dataset_info': {
            'path': args.data_dir,
            'classes': args.num_classes,
            'image_size': args.image_size,
            'total_samples': len(dataset) if hasattr(dataset, '__len__') else 'N/A'
        },
        'training_summary': {
            'epochs_completed': len(history.get('train_loss', [])),
            'best_val_loss': min(history.get('val_loss', [])) if history.get('val_loss') else 'N/A',
            'best_val_acc': max(history.get('val_acc', [])) if history.get('val_acc') else 'N/A',
        }
    }
    
    # Save report
    report_file = log_dir / "training_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Training summary saved to: {report_file}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print('='*60)
    
    if history.get('train_loss'):
        print(f"Epochs completed: {len(history['train_loss'])}")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    print(f"\nFramework status: {'C++ backend active' if HAS_CPP_BACKEND else 'Python fallback mode'}")
    print(f"Model: {model.__class__.__name__}")
    
    return history

if __name__ == '__main__':
    main()