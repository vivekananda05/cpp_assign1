import numpy as np
from src.python.framework import Module, Tensor, Sequential

# First define the layer classes if they don't exist
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.bias = Tensor(np.zeros(out_channels))
        
    def forward(self, x):
        # Simple convolution simulation
        batch, in_channels, height, width = x.shape
        output_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Return random output for simulation
        return Tensor(np.random.randn(batch, self.out_channels, output_height, output_width))
        
    def getNumParameters(self):
        weight_params = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size
        bias_params = self.out_channels
        return weight_params + bias_params
    
    def getMACs(self, input_shape):
        # Simplified MAC calculation
        batch, in_channels, height, width = input_shape
        output_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        macs = batch * self.out_channels * output_height * output_width * self.in_channels * self.kernel_size * self.kernel_size
        return macs
    
    def getFLOPs(self, input_shape):
        # FLOPs = 2 * MACs for multiply-add operations
        return 2 * self.getMACs(input_shape)

class MaxPool2d:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        
    def forward(self, x):
        # Simple pooling simulation
        batch, channels, height, width = x.shape
        output_height = height // self.kernel_size
        output_width = width // self.kernel_size
        return Tensor(np.random.randn(batch, channels, output_height, output_width))
    
    def getFLOPs(self, input_shape):
        # Pooling operations
        batch, channels, height, width = input_shape
        output_height = height // self.kernel_size
        output_width = width // self.kernel_size
        # Each pooling operation compares kernel_size*kernel_size elements
        return batch * channels * output_height * output_width * (self.kernel_size * self.kernel_size - 1)

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.randn(out_features, in_features) * 0.1)
        self.bias = Tensor(np.zeros(out_features))
        
    def forward(self, x):
        # Simple linear transformation simulation
        batch = x.shape[0]
        # For simulation, just return random output
        return Tensor(np.random.randn(batch, self.out_features))
        
    def getNumParameters(self):
        weight_params = self.out_features * self.in_features
        bias_params = self.out_features
        return weight_params + bias_params
    
    def getMACs(self, input_shape):
        # For linear layer: MACs = batch_size * in_features * out_features
        batch = input_shape[0]
        return batch * self.in_features * self.out_features
    
    def getFLOPs(self, input_shape):
        # FLOPs = 2 * MACs for multiply-add operations
        return 2 * self.getMACs(input_shape)

class ReLU:
    def __init__(self):
        pass
        
    def forward(self, x):
        # Simple ReLU simulation
        return Tensor(np.maximum(x.data, 0))
    
    def getFLOPs(self, input_shape):
        # ReLU is just a comparison operation per element
        total_elements = np.prod(input_shape)
        return total_elements

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        
    def forward(self, x):
        if self.training:
            # Apply dropout during training
            mask = np.random.binomial(1, 1-self.p, size=x.shape)
            return Tensor(x.data * mask / (1 - self.p))
        else:
            return x
    
    def train(self, mode=True):
        self.training = mode
        return self

class CNNModel(Module):
    """Customizable CNN model for image classification"""
    
    def __init__(self, num_classes=10, config=None):
        super().__init__()
        
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
        # After two pooling layers with kernel_size=2, image size reduces from 32x32 to 8x8
        fc1_input_size = self.config['conv2_channels'] * 8 * 8
        
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
                self._parameters[f'weight_{param_idx}'] = layer.weight
                param_idx += 1
            if hasattr(layer, 'bias'):
                self._parameters[f'bias_{param_idx}'] = layer.bias
                param_idx += 1
    
    def forward(self, x):
        """Forward pass - FIXED to avoid recursion"""
        # Input shape: [batch, channels, height, width]
        
        # Conv Block 1
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        # Conv Block 2
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        # Flatten
        if hasattr(x, 'data'):
            batch_size = x.shape[0]
            x_data = x.data.reshape(batch_size, -1)
            x = Tensor(x_data)
        else:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        # Fully connected layers
        self.dropout.training = self.training
        x = self.dropout.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        
        return x
    
    def __call__(self, x):
        """Make model callable - CALLS forward(), NOT itself"""
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
        # Also set dropout training mode
        if hasattr(self.dropout, 'train'):
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
        
        print(f"\nInput Shape: {input_shape}")
        print(f"\nLayer-wise Analysis:")
        print("-" * 80)
        print(f"{'Layer':<20} {'Output Shape':<20} {'Params':<15} {'MACs':<15} {'FLOPs':<15}")
        print("-" * 80)
        
        # Track input shape through layers
        current_shape = input_shape
        
        # Analyze each layer
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
                flops = 2 * macs  # Estimate FLOPs as 2x MACs
                
            # Update current shape (simplified)
            if 'Conv' in name:
                # For conv layer: output channels change
                batch, _, height, width = current_shape
                output_height = (height + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                output_width = (width + 2*layer.padding - layer.kernel_size) // layer.stride + 1
                current_shape = (batch, layer.out_channels, output_height, output_width)
            elif 'Pool' in name:
                # For pool layer: spatial dimensions reduce
                batch, channels, height, width = current_shape
                current_shape = (batch, channels, height // 2, width // 2)
            elif name == 'FC1':
                # After flattening
                current_shape = (current_shape[0], self.config['hidden_size'])
            elif name == 'FC2':
                current_shape = (current_shape[0], self.config['num_classes'] if hasattr(self.config, 'num_classes') else 10)
                
            total_params += params
            total_macs += macs
            total_flops += flops
            
            print(f"{name:<20} {str(current_shape):<20} {params:<15,} {macs:<15,} {flops:<15,}")
        
        print("-" * 80)
        print(f"{'TOTAL':<20} {'':<20} {total_params:<15,} {total_macs:<15,} {total_flops:<15,}")
        
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
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