#!/usr/bin/env python3
"""
Evaluation script for custom CNN framework
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.python.dataloader import ImageDataset, DataLoader
from models.cnn_model import CNNModel
from src.python.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test dataset directory')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Input image size (default: 32)')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config_path', type=str,
                       help='Path to model configuration file')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--output_dir', type=str, default='./reports',
                       help='Output directory for reports (default: ./reports)')
    
    return parser.parse_args()

def load_model_config(config_path):
    """Load model configuration from file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get('model_config', {}), config.get('training_args', {})

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model configuration if provided
    if args.config_path:
        model_config, training_args = load_model_config(args.config_path)
        num_classes = training_args.get('num_classes', 10)
    else:
        # Try to infer from data directory
        model_config = {}
        num_classes = len([d for d in os.listdir(args.data_dir) 
                          if os.path.isdir(os.path.join(args.data_dir, d))])
    
    # Load dataset
    print(f"Loading test dataset from {args.data_dir}")
    
    dataset = ImageDataset(
        root_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        augment=False,  # No augmentation for testing
        normalize=True
    )
    
    print(f"Loaded {len(dataset)} test samples")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    
    # Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create model
    print("\nCreating model...")
    model = CNNModel(num_classes=num_classes, config=model_config)
    
    # Load weights
    print(f"Loading model weights from {args.model_path}")
    model.load(args.model_path)
    
    # Create evaluator
    evaluator = Evaluator(model, dataset)
    
    # Evaluate model
    report = evaluator.evaluate(test_loader, verbose=True)
    
    # Analyze predictions
    evaluator.analyze_predictions(test_loader, num_samples=5)
    
    # Save final evaluation
    eval_file = output_dir / "final_evaluation.json"
    with open(eval_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nEvaluation completed!")
    print(f"Detailed report saved to {eval_file}")
    
    return report

if __name__ == '__main__':
    main()