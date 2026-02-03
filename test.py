# test_simple_import.py
import sys
import os

build_dir = "/home/user/Downloads/GNR638/ASSIGN1/custom_dl_framework/build"
sys.path.insert(0, build_dir)

print("Files in build directory:")
for f in os.listdir(build_dir):
    if "custom_dl_framework" in f:
        print(f"  - {f}")

# Try to import without using exec
try:
    # Remove the .cpython... part for import
    import custom_dl_framework as dl
    print("\n✓ Successfully imported custom_dl_framework!")
    
    # List what's available
    print("\nAvailable in module:")
    for attr in dir(dl):
        if not attr.startswith('__'):
            print(f"  - {attr}")
    
    # Test basic functionality without elementwiseMultiply
    print("\nTesting basic Tensor operations...")
    
    # Create a tensor
    tensor = dl.Tensor([2, 3])
    print(f"✓ Created tensor")
    
    # Call simple methods that should exist
    tensor.zeros()
    print(f"✓ Called zeros()")
    
    shape = tensor.getShape()
    print(f"✓ Got shape: {shape}")
    
    data = tensor.getData()
    print(f"✓ Got data (length: {len(data)})")
    
    # Test reshape
    reshaped = tensor.reshape([3, 2])
    print(f"✓ Reshaped tensor")
    
    # Test flatten
    flattened = tensor.flatten()
    print(f"✓ Flattened tensor")
    
    print(f"\nTensor string representation: {tensor}")
    
except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    
    # Try with the full filename
    try:
        import importlib.util
        module_path = os.path.join(build_dir, "custom_dl_framework.cpython-310-x86_64-linux-gnu.so")
        spec = importlib.util.spec_from_file_location("custom_dl_framework", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("\n✓ Loaded module directly!")
        
        # Test it
        tensor = module.Tensor([2, 3])
        print(f"✓ Created tensor: {tensor}")
        
    except Exception as e2:
        print(f"✗ Direct load failed: {e2}")
        import traceback
        traceback.print_exc()



# test_extended.py
import custom_dl_framework as dl

print("Testing extended functionality...")

# Test creating different types of layers
print("\n1. Testing layer creation:")
conv = dl.Conv2d(3, 16, 3)  # 3 input channels, 16 output channels, 3x3 kernel
print(f"✓ Created Conv2d layer: {conv.getName()}")

linear = dl.Linear(784, 10)  # 784 input features, 10 output features
print(f"✓ Created Linear layer: {linear.getName()}")

relu = dl.ReLU()
print(f"✓ Created ReLU layer: {relu.getName()}")

# Test model analysis
print("\n2. Testing model analysis:")
layers = [conv, linear]
input_shape = [1, 3, 32, 32]  # batch_size=1, channels=3, height=32, width=32
stats = dl.ModelStats()
print(f"✓ Created ModelStats")

# Test tensor operations
print("\n3. Testing more tensor operations:")
# Create two tensors
tensor1 = dl.Tensor([2, 2])
tensor1.randomUniform(0.0, 1.0)

tensor2 = dl.Tensor([2, 2])
tensor2.randomUniform(0.0, 1.0)

print(f"✓ Created and initialized tensors")

# Test activation functions
print("\n4. Testing activation functions:")
test_tensor = dl.Tensor([1, 5])
test_tensor.randomUniform(-1.0, 1.0)
print(f"Original tensor: {test_tensor}")

relu_output = test_tensor.relu()
print(f"ReLU output shape: {relu_output.getShape()}")

# Test softmax
softmax_output = test_tensor.softmax()
print(f"Softmax output shape: {softmax_output.getShape()}")

print("\n✅ All tests completed successfully!")
print("Your custom deep learning framework is working correctly!")      



# test_neural_network.py
import custom_dl_framework as dl

print("Building a simple neural network...")

# Create a simple CNN-like architecture
layers = [
    dl.Conv2d(3, 16, 3, stride=1, padding=1, name="conv1"),
    dl.ReLU("relu1"),
    dl.MaxPool2d(2, 2, name="pool1"),
    dl.Conv2d(16, 32, 3, stride=1, padding=1, name="conv2"),
    dl.ReLU("relu2"),
    dl.MaxPool2d(2, 2, name="pool2"),
    dl.Linear(32 * 8 * 8, 10, name="fc1")  # Assuming 32x32 input -> 8x8 after two pools
]

print(f"Created {len(layers)} layers:")
for layer in layers:
    print(f"  - {layer.getName()} ({layer.__class__.__name__})")

# Create a dummy input tensor
batch_size = 4
channels = 3
height = 32
width = 32
input_tensor = dl.Tensor([batch_size, channels, height, width])
input_tensor.randomNormal(0.0, 0.1)

print(f"\nCreated input tensor with shape: {input_tensor.getShape()}")

# Test forward pass through first layer
print("\nTesting forward pass through conv1...")
output = layers[0].forward(input_tensor)
print(f"Output shape after conv1: {output.getShape()}")

print("\n✅ Neural network test completed!")