#!/usr/bin/env python3
"""
Benchmark script for custom CNN framework
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.python.framework import Tensor
try:
    import custom_dl_framework as cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

def benchmark_tensor_operations():
    """Benchmark tensor operations"""
    print("\n" + "="*60)
    print("TENSOR OPERATIONS BENCHMARK")
    print("="*60)
    
    results = {}
    
    # Benchmark sizes
    sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
    
    for size in sizes:
        print(f"\nBenchmarking size {size}")
        
        # Create tensors
        a = Tensor.randn(size)
        b = Tensor.randn(size)
        
        # Addition
        start = time.time()
        for _ in range(100):
            c = a + b
        add_time = (time.time() - start) / 100
        results[f'add_{size}'] = add_time
        
        # Multiplication
        start = time.time()
        for _ in range(100):
            c = a * b
        mul_time = (time.time() - start) / 100
        results[f'mul_{size}'] = mul_time
        
        # Matrix multiplication (if 2D)
        if len(size) == 2:
            start = time.time()
            for _ in range(100):
                c = a.matmul(b)
            matmul_time = (time.time() - start) / 100
            results[f'matmul_{size}'] = matmul_time
        
        print(f"  Addition: {add_time:.6f}s")
        print(f"  Multiplication: {mul_time:.6f}s")
        if len(size) == 2:
            print(f"  Matrix Multiply: {matmul_time:.6f}s")
    
    return results

def benchmark_model_forward():
    """Benchmark model forward pass"""
    print("\n" + "="*60)
    print("MODEL FORWARD PASS BENCHMARK")
    print("="*60)
    
    results = {}
    
    from models.cnn_model import CNNModel
    
    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]
    
    model = CNNModel(num_classes=10)
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size {batch_size}")
        
        # Create input tensor
        input_tensor = Tensor.randn((batch_size, 3, 32, 32))
        
        # Warm up
        for _ in range(10):
            output = model(input_tensor)
        
        # Benchmark
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            output = model(input_tensor)
        
        forward_time = (time.time() - start) / iterations
        fps = batch_size / forward_time
        
        results[f'forward_batch_{batch_size}'] = {
            'time_per_batch': forward_time,
            'fps': fps,
            'time_per_sample': forward_time / batch_size
        }
        
        print(f"  Time per batch: {forward_time:.6f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Time per sample: {forward_time/batch_size:.6f}s")
    
    return results

def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARK")
    print("="*60)
    
    results = {}
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Test different tensor sizes
    sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
    
    for i, size in enumerate(sizes):
        print(f"\nTesting tensor size {size}")
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create tensor
        tensor = Tensor.randn(size)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        
        mem_used = mem_after - mem_before
        expected_mem = (size[0] * size[1] * 4) / (1024 * 1024)  # 4 bytes per float32
        
        results[f'memory_size_{size}'] = {
            'actual_mb': mem_used,
            'expected_mb': expected_mem,
            'efficiency': (expected_mem / mem_used) if mem_used > 0 else 0
        }
        
        print(f"  Memory used: {mem_used:.2f} MB")
        print(f"  Expected: {expected_mem:.2f} MB")
        print(f"  Efficiency: {(expected_mem/mem_used)*100:.1f}%" if mem_used > 0 else "N/A")
        
        # Clean up
        del tensor
    
    return results

def benchmark_data_loading():
    """Benchmark data loading performance"""
    print("\n" + "="*60)
    print("DATA LOADING BENCHMARK")
    print("="*60)
    
    results = {}
    
    from src.python.dataloader import ImageDataset, DataLoader
    
    # Create a synthetic dataset for testing
    import tempfile
    import cv2
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic dataset
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create 3 classes
        for class_idx in range(3):
            class_dir = dataset_path / f"class_{class_idx}"
            class_dir.mkdir(exist_ok=True)
            
            # Create 100 synthetic images per class
            for img_idx in range(100):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img_path = class_dir / f"image_{img_idx}.png"
                cv2.imwrite(str(img_path), img)
        
        # Benchmark loading
        print("\nBenchmarking dataset loading...")
        
        start = time.time()
        dataset = ImageDataset(
            root_dir=str(dataset_path),
            image_size=(32, 32),
            augment=False,
            normalize=True
        )
        load_time = time.time() - start
        
        results['dataset_loading'] = {
            'load_time': load_time,
            'images_per_second': len(dataset) / load_time,
            'total_images': len(dataset)
        }
        
        print(f"  Load time: {load_time:.4f}s")
        print(f"  Images per second: {len(dataset)/load_time:.2f}")
        
        # Benchmark data loader
        print("\nBenchmarking DataLoader...")
        
        batch_sizes = [16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Time first epoch
            start = time.time()
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= 10:  # Only test first 10 batches
                    break
            first_batches_time = time.time() - start
            
            # Time full epoch
            start = time.time()
            for images, labels in dataloader:
                pass
            full_epoch_time = time.time() - start
            
            results[f'dataloader_batch_{batch_size}'] = {
                'first_10_batches_time': first_batches_time,
                'full_epoch_time': full_epoch_time,
                'images_per_second': len(dataset) / full_epoch_time
            }
            
            print(f"    First 10 batches: {first_batches_time:.4f}s")
            print(f"    Full epoch: {full_epoch_time:.4f}s")
            print(f"    Images per second: {len(dataset)/full_epoch_time:.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark custom DL framework')
    parser.add_argument('--output', type=str, default='./benchmark_results.json',
                       help='Output file for benchmark results')
    args = parser.parse_args()
    
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'has_cpp_backend': HAS_CPP,
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform
        }
    }
    
    # Run benchmarks
    print("Starting benchmarks...")
    
    # Tensor operations
    tensor_results = benchmark_tensor_operations()
    all_results['tensor_operations'] = tensor_results
    
    # Model forward pass
    model_results = benchmark_model_forward()
    all_results['model_forward'] = model_results
    
    # Memory usage
    memory_results = benchmark_memory_usage()
    all_results['memory_usage'] = memory_results
    
    # Data loading
    data_results = benchmark_data_loading()
    all_results['data_loading'] = data_results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if HAS_CPP:
        print("✓ C++ backend available")
    else:
        print("✗ C++ backend not available (using pure Python)")
    
    # Find fastest operation
    if tensor_results:
        fastest_op = min(tensor_results.items(), key=lambda x: x[1])
        print(f"Fastest tensor operation: {fastest_op[0]} ({fastest_op[1]:.6f}s)")
    
    if model_results:
        fastest_batch = min(model_results.items(), 
                           key=lambda x: x[1]['time_per_sample'])
        print(f"Fastest inference: {fastest_batch[0]} "
              f"({fastest_batch[1]['time_per_sample']:.6f}s per sample)")

if __name__ == '__main__':
    main()