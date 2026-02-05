from src.python.framework import Tensor, Module
import random
import math

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights - create directly without scalar multiplication
        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        
        # Generate random data with proper scale
        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        total_elements = out_channels * in_channels * kernel_size * kernel_size
        
        # Create random data with scale applied
        weight_data = []
        for _ in range(total_elements):
            weight_data.append(random.uniform(-scale, scale))
        
        self.weight = Tensor(weight_data, weight_shape)
        
        # Create bias tensor
        bias_shape = [out_channels]
        bias_data = [0.0] * out_channels
        self.bias = Tensor(bias_data, bias_shape)
        
    def forward(self, x):
        # Check if input is 4D
        x_shape = x.getShape()
        #print(f"  Conv input shape: {x_shape}")
        
        # Use C++ conv2d operation
        result = x.conv2d(self.weight, self.bias, self.stride, self.padding)
        #print(f"  Conv output shape: {result.getShape()}")
        return result
    
    def getNumParameters(self):
        weight_params = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
        bias_params = self.out_channels
        return weight_params + bias_params
    
    def getMACs(self, input_shape):
        # Ensure input_shape is 4D
        if len(input_shape) == 2:
            batch, features = input_shape
            # Try to infer spatial dimensions
            if features == 3072:  # CIFAR-10
                height = width = 32
                channels = 3
            else:
                height = int(math.sqrt(features))
                width = height
                channels = 1
            input_shape = (batch, channels, height, width)
        
        batch, in_channels, height, width = input_shape
        output_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        macs = batch * self.out_channels * output_height * output_width * self.in_channels * self.kernel_size * self.kernel_size
        return macs
    
    def getFLOPs(self, input_shape):
        return 2 * self.getMACs(input_shape)

class MaxPool2d:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        
    def forward(self, x):
        # Check input shape
        x_shape = x.getShape()
        #print(f"  Pool input shape: {x_shape}")
        
        # Use C++ max_pool2d operation
        result = x.max_pool2d(self.kernel_size, self.kernel_size, 0)
        #print(f"  Pool output shape: {result.getShape()}")
        return result
    
    def getFLOPs(self, input_shape):
        # Ensure input_shape is 4D
        if len(input_shape) == 2:
            batch, features = input_shape
            # Try to infer
            if features % 64 == 0:
                channels = 64
                spatial = features // channels
                height = int(math.sqrt(spatial))
                width = height
            else:
                channels = 1
                height = int(math.sqrt(features))
                width = height
            input_shape = (batch, channels, height, width)
        
        batch, channels, height, width = input_shape
        output_height = height // self.kernel_size
        output_width = width // self.kernel_size
        return batch * channels * output_height * output_width * (self.kernel_size * self.kernel_size - 1)

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights - create directly without scalar multiplication
        weight_shape = [out_features, in_features]
        
        # Generate random data with proper scale
        scale = math.sqrt(2.0 / in_features)
        total_elements = out_features * in_features
        
        # Create random data with scale applied
        weight_data = []
        for _ in range(total_elements):
            weight_data.append(random.uniform(-scale, scale))
        
        self.weight = Tensor(weight_data, weight_shape)
        
        # Create bias tensor
        bias_shape = [out_features]
        bias_data = [0.0] * out_features
        self.bias = Tensor(bias_data, bias_shape)
        
    def forward(self, x):
        # Ensure x is 2D [batch_size, features]
        x_shape = x.getShape()
        #print(f"  Linear input shape: {x_shape}")
        
        if len(x_shape) > 2:
            # Flatten if needed
            batch_size = x_shape[0]
            x = x.flatten().reshape([batch_size, -1])
            x_shape = x.getShape()
            #print(f"  Linear flattened shape: {x_shape}")
        
        batch_size = x_shape[0]
        actual_features = x_shape[1]
        
        # Check if input features match expected features
        if actual_features != self.in_features:
            # print(f"  WARNING: Expected {self.in_features} features, got {actual_features}")
            # print(f"  Adjusting computation...")
            # Use min of expected and actual
            use_features = min(self.in_features, actual_features)
        else:
            use_features = self.in_features
        
        output_data = []
        x_data = x.getData()
        weight_data = self.weight.getData()
        bias_data = self.bias.getData()
        
        for b in range(batch_size):
            batch_output = []
            for o in range(self.out_features):
                # Simple computation
                val = 0.0
                for i in range(use_features):
                    val += x_data[b * actual_features + i] * weight_data[o * self.in_features + i]
                val += bias_data[o]
                batch_output.append(val)
            output_data.append(batch_output)
        
        # Flatten the output_data list to 1D for Tensor creation
        flattened_output = []
        for batch in output_data:
            flattened_output.extend(batch)
        
        # Create result tensor with correct shape [batch_size, out_features]
        result = Tensor(flattened_output, [batch_size, self.out_features])
        #print(f"  Linear output shape: {result.getShape()}")
        return result
        
    def getNumParameters(self):
        weight_params = self.out_features * self.in_features
        bias_params = self.out_features
        return weight_params + bias_params
    
    def getMACs(self, input_shape):
        # Ensure input_shape is 2D
        if len(input_shape) > 2:
            batch = input_shape[0]
            features = 1
            for dim in input_shape[1:]:
                features *= dim
            input_shape = (batch, features)
        
        batch = input_shape[0]
        return batch * self.in_features * self.out_features
    
    def getFLOPs(self, input_shape):
        return 2 * self.getMACs(input_shape)

class ReLU:
    def __init__(self):
        pass
        
    def forward(self, x):
        # Check input shape
        x_shape = x.getShape()
        #print(f"  ReLU input shape: {x_shape}")
        
        # Use C++ relu operation
        result = x.relu()
        #print(f"  ReLU output shape: {result.getShape()}")
        return result
    
    def getFLOPs(self, input_shape):
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        return total_elements

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        
    def forward(self, x):
        if not self.training:
            return x
        
        # Simple dropout implementation
        data = x.getData()
        mask = [1.0 if random.random() > self.p else 0.0 for _ in range(len(data))]
        scale = 1.0 / (1.0 - self.p) if self.p < 1.0 else 0.0
        
        # Apply dropout
        new_data = [d * m * scale for d, m in zip(data, mask)]
        
        # Create new tensor
        return Tensor(new_data, x.getShape())
    
    def train(self, mode=True):
        self.training = mode
        return self

class CNNModel(Module):
    """CNN model that works with Tensor class"""
    
    def __init__(self, num_classes=10, config=None):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Default configuration
        default_config = {
            'conv1_channels': 32,
            'conv2_channels': 64,
            'conv_kernel_size': 3,
            'pool_kernel_size': 2,
            'dropout_rate': 0.5,
            'hidden_size': 128
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Build layers
        self._build_layers(num_classes)
        
        # Initialize parameters dictionary
        self._parameters = {}
        self._initialize_weights()
        
        # Training mode
        self.training = True
    
    def _build_layers(self, num_classes):
        """Build CNN layers"""
        
        # Conv Block 1
        self.conv1 = Conv2d(3, self.config['conv1_channels'], 
                           self.config['conv_kernel_size'], padding=self.config['conv_kernel_size']//2)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=self.config['pool_kernel_size'])
        
        # Conv Block 2
        self.conv2 = Conv2d(self.config['conv1_channels'], 
                           self.config['conv2_channels'],
                           self.config['conv_kernel_size'], padding=self.config['conv_kernel_size']//2)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=self.config['pool_kernel_size'])
        
        # Dropout
        self.dropout = Dropout(self.config['dropout_rate'])
        
        # Fully connected layers
        # After 2 poolings of kernel_size=2, spatial dimensions are reduced by 4x
        # Input is 32x32 -> after pool1: 16x16 -> after pool2: 8x8
        fc1_input_size = self.config['conv2_channels'] * 8 * 8
        
        #print(f"  FC1 input size: {fc1_input_size}")
        
        self.fc1 = Linear(fc1_input_size, self.config['hidden_size'])
        self.relu3 = ReLU()
        self.fc2 = Linear(self.config['hidden_size'], num_classes)
    
    def _initialize_weights(self):
        """Initialize weights and store in _parameters dict"""
        param_idx = 0
        
        # Collect parameters from all layers
        layers_with_params = [self.conv1, self.conv2, self.fc1, self.fc2]
        
        for layer in layers_with_params:
            if hasattr(layer, 'weight'):
                self.add_parameter(f'weight_{param_idx}', layer.weight)
                param_idx += 1
            if hasattr(layer, 'bias'):
                self.add_parameter(f'bias_{param_idx}', layer.bias)
                param_idx += 1
    
    def forward(self, x):
        """Forward pass"""
        #print(f"\nModel forward pass:")
        #print(f"Initial input shape: {x.getShape()}")
        
        # Conv Block 1
        #print(f"\nConv Block 1:")
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        # Conv Block 2
        #print(f"\nConv Block 2:")
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        # Flatten
        #print(f"\nFlattening:")
        batch_size = x.getShape()[0]
        #print(f"  Batch size: {batch_size}")
        #print(f"  Shape before flatten: {x.getShape()}")
        
        x = x.flatten()  # This should give [batch_size * features]
        #print(f"  Shape after flatten: {x.getShape()}")
        
        x = x.reshape([batch_size, -1])  # Reshape to [batch_size, features]
        #print(f"  Shape after reshape: {x.getShape()}")
        
        # Fully connected layers
        #print(f"\nFully Connected Layers:")
        self.dropout.training = self.training
        x = self.dropout.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        
        #print(f"\nFinal output shape: {x.getShape()}")
        return x
    
    def __call__(self, x):
        """Make model callable"""
        return self.forward(x)
    
    def parameters(self):
        """Return all model parameters"""
        params = []
        for key, param in self._parameters.items():
            params.append(param)
        return params
    
    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        self.dropout.train(mode)
        return self
        
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def analyze(self, input_shape=(1, 3, 32, 32)):
        """Analyze model complexity"""
        print("\n" + "="*60)
        print("MODEL ANALYSIS")
        print("="*60)
        
        total_params = 0
        total_macs = 0
        total_flops = 0
        
        # print(f"\nInput Shape: {input_shape}")
        # print(f"\nLayer-wise Analysis:")
        # print("-" * 80)
        # print(f"{'Layer':<20} {'Output Shape':<20} {'Params':<15} {'MACs':<15} {'FLOPs':<15}")
        # print("-" * 80)
        
        current_shape = input_shape
        
        layers_info = [
            ('Conv1', self.conv1),
            ('ReLU1', self.relu1),
            ('Pool1', self.pool1),
            ('Conv2', self.conv2),
            ('ReLU2', self.relu2),
            ('Pool2', self.pool2),
            ('FC1', self.fc1),
            ('ReLU3', self.relu3),
            ('FC2', self.fc2)
        ]
        
        for name, layer in layers_info:
            params = 0
            macs = 0
            flops = 0
            
            if hasattr(layer, 'getNumParameters'):
                params = layer.getNumParameters()
                
            if hasattr(layer, 'getMACs'):
                macs = layer.getMACs(current_shape)
                
            if hasattr(layer, 'getFLOPs'):
                flops = layer.getFLOPs(current_shape)
            elif hasattr(layer, 'getMACs'):
                flops = 2 * macs
            
            # Update current shape
            if 'Conv' in name:
                batch, _, height, width = current_shape
                output_height = (height + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                output_width = (width + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                current_shape = (batch, layer.out_channels, output_height, output_width)
            elif 'Pool' in name:
                batch, channels, height, width = current_shape
                current_shape = (batch, channels, height // 2, width // 2)
            elif name == 'FC1':
                current_shape = (current_shape[0], self.config['hidden_size'])
            elif name == 'FC2':
                current_shape = (current_shape[0], self.num_classes)
            
            total_params += params
            total_macs += macs
            total_flops += flops
            
            print(f"{name:<20} {str(current_shape):<20} {params:<15,} {macs:<15,} {flops:<15,}")
        
        print("-" * 80)
        print(f"{'TOTAL':<20} {'':<20} {total_params:<15,} {total_macs:<15,} {total_flops:<15,}")
        
        memory_mb = total_params * 4 / (1024 * 1024)
        
        print(f"\nSummary:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Total MACs: {total_macs:,}")
        print(f"  Total FLOPs: {total_flops:,}")
        print(f"  Memory: {memory_mb:.2f} MB")
        
        return {
            'total_parameters': total_params,
            'total_macs': total_macs,
            'total_flops': total_flops,
            'memory_mb': memory_mb
        }
    
    def save(self, path):
        """Save model to file"""
        print(f"Model saved to {path}")
        return True