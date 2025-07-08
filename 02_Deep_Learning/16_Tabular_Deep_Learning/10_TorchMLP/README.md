# TorchMLP: PyTorch Multi-Layer Perceptron for Tabular Data

## Overview
TorchMLP is a PyTorch-based implementation of multi-layer perceptrons specifically optimized for tabular data. Ranking 7th in TabArena with an Elo score of 1350, it represents a solid, production-ready neural network solution built on the PyTorch framework.

## Key Features
- **PyTorch Native**: Built using PyTorch primitives for flexibility
- **Production Ready**: Optimized for deployment and scaling
- **Modular Design**: Easy to customize and extend
- **GPU Acceleration**: Full CUDA support for training and inference

## Performance Characteristics
- **TabArena Ranking**: #7 with Elo score of 1350
- **Default Performance**: Moderate baseline (~1100 Elo)
- **Tuning Benefits**: Good improvement with optimization (+150 points)
- **Ensemble Gains**: Solid ensemble contribution (+200 points from default)

## Architecture
Standard multi-layer perceptron with:
- Configurable hidden layers
- Batch normalization
- Dropout regularization
- Modern activation functions
- Embedding layers for categorical features

## Use Cases
- **PyTorch Ecosystems**: When PyTorch integration is required
- **Custom Extensions**: Need for custom loss functions or layers
- **Research**: Experimental neural network architectures
- **Production**: Scalable deployment requirements

## Implementation
Built on PyTorch framework with standard practices:
- Standard MLP architecture
- Embedding layers for categorical data
- Batch normalization and dropout
- Adam optimizer with learning rate scheduling

## Mathematical Framework

### Input Processing
Given tabular dataset with features X = {x₁, x₂, ..., xₙ}:

**Numerical Features:**
```
x_num = StandardScaler(x_raw) = (x_raw - μ) / σ
```

**Categorical Features:**
```
E_cat = Embedding(x_cat) ∈ ℝᵈᵉᵐᵇ
```

Where d_emb is the embedding dimension.

**Feature Concatenation:**
```
X_input = Concat(x_num, E_cat₁, E_cat₂, ..., E_catₘ)
```

### Multi-Layer Perceptron Architecture
TorchMLP uses the standard feedforward neural network formulation:

**Hidden Layer Computation:**
```
h⁽ˡ⁾ = f(BatchNorm(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾))
```

Where:
- W⁽ˡ⁾ ∈ ℝᵈˡˣᵈˡ⁻¹ are layer weights
- b⁽ˡ⁾ ∈ ℝᵈˡ are layer biases
- f is the activation function
- h⁽⁰⁾ = X_input

### Batch Normalization
Applied before activation functions:

```
BN(x) = γ · (x - μ_batch) / √(σ²_batch + ε) + β
```

Where:
- μ_batch and σ²_batch are batch statistics
- γ and β are learnable parameters
- ε is a small constant for numerical stability

### Activation Functions
TorchMLP supports multiple activation functions:

**ReLU:**
```
ReLU(x) = max(0, x)
```

**GELU (Gaussian Error Linear Unit):**
```
GELU(x) = x · Φ(x) = x · (1/2)[1 + erf(x/√2)]
```

**SiLU/Swish:**
```
SiLU(x) = x · sigmoid(x) = x / (1 + e⁻ˣ)
```

### Dropout Regularization
Applied during training:

```
Dropout(x) = x · Bernoulli(1-p) / (1-p)
```

Where p is the dropout probability.

### Complete Forward Pass
The full forward computation:

```
h⁽⁰⁾ = X_input
For l = 1 to L:
    z⁽ˡ⁾ = W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾
    h⁽ˡ⁾ = Dropout(f(BatchNorm(z⁽ˡ⁾)))
```

### Output Layer
**Binary Classification:**
```
ŷ = sigmoid(W_out h⁽ᴸ⁾ + b_out)
```

**Multi-class Classification:**
```
ŷ = softmax(W_out h⁽ᴸ⁾ + b_out)
```

**Regression:**
```
ŷ = W_out h⁽ᴸ⁾ + b_out
```

### Loss Functions
**Binary Cross-Entropy:**
```
L_BCE = -∑ᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Categorical Cross-Entropy:**
```
L_CE = -∑ᵢ ∑ⱼ yᵢⱼ log(ŷᵢⱼ)
```

**Mean Squared Error:**
```
L_MSE = ∑ᵢ (yᵢ - ŷᵢ)²
```

### Optimization with Adam
TorchMLP uses the Adam optimizer:

**Parameter Updates:**
```
mₜ = β₁mₜ₋₁ + (1-β₁)∇θₜ
vₜ = β₂vₜ₋₁ + (1-β₂)(∇θₜ)²
m̂ₜ = mₜ/(1-β₁ᵗ)
v̂ₜ = vₜ/(1-β₂ᵗ)
θₜ₊₁ = θₜ - η · m̂ₜ/(√v̂ₜ + ε)
```

Where:
- η is the learning rate
- β₁, β₂ are momentum parameters (typically 0.9, 0.999)
- ε is numerical stability constant (typically 1e-8)

### Learning Rate Scheduling
**Step Decay:**
```
ηₜ = η₀ · γ^⌊t/step_size⌋
```

**Cosine Annealing:**
```
ηₜ = η_min + (η_max - η_min) · (1 + cos(πt/T))/2
```

**Exponential Decay:**
```
ηₜ = η₀ · e^(-λt)
```

### Regularization Terms
**L2 Weight Decay:**
```
L_reg = λ ∑ₗ ||W⁽ˡ⁾||²_F
```

**Total Loss with Regularization:**
```
L_total = L_task + L_reg
```

### Gradient Computation
Using backpropagation:

```
∂L/∂W⁽ˡ⁾ = (∂L/∂h⁽ˡ⁾) · (h⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = ∂L/∂h⁽ˡ⁾
```

### Feature Importance via Gradients
TorchMLP can compute feature importance:

```
Importance_i = ||∇L/∇xᵢ||₂
```

### PyTorch Implementation Details
```python
class TorchMLP(nn.Module):
    def forward(self, x_num, x_cat):
        # Embedding categorical features
        x_emb = [self.embeddings[i](x_cat[:, i]) 
                for i in range(x_cat.shape[1])]
        
        # Concatenate all features
        x = torch.cat([x_num] + x_emb, dim=1)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
        
        return self.output(x)
```

### Memory Optimization
For large datasets, TorchMLP implements:

**Gradient Accumulation:**
```
Loss_accumulated += Loss_batch / accumulation_steps
```

**Mixed Precision Training:**
```
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

## Advantages

### Framework Integration
- **Native PyTorch**: Seamless integration with PyTorch ecosystem
- **GPU Acceleration**: Automatic CUDA optimization
- **Distributed Training**: Built-in support for multi-GPU training
- **JIT Compilation**: TorchScript compilation for production

### Development Flexibility
- **Custom Layers**: Easy to add custom neural network layers
- **Loss Functions**: Custom loss function implementation
- **Hooks**: Access to intermediate activations and gradients
- **Debugging**: Comprehensive debugging tools available

### Production Benefits
- **Deployment**: Multiple deployment options (ONNX, TorchScript)
- **Scaling**: Horizontal and vertical scaling capabilities
- **Monitoring**: Integration with MLOps tools
- **Versioning**: Model versioning and experiment tracking

## Limitations

### Framework Dependency
- **PyTorch Requirement**: Tied to PyTorch ecosystem
- **Memory Usage**: Higher memory usage than lightweight frameworks
- **Startup Time**: Framework initialization overhead

### Performance Considerations
- **Training Speed**: May be slower than specialized implementations
- **Model Size**: Larger model files due to framework overhead
- **Inference**: Additional latency from framework abstractions

## Hyperparameter Configuration

### Architecture Parameters
- **Hidden Layers**: 3-6 layers for most applications
- **Layer Width**: 128-512 neurons per layer
- **Dropout Rate**: 0.1-0.3 for regularization
- **Activation**: ReLU, GELU, or SiLU

### Training Parameters
- **Learning Rate**: 1e-3 to 1e-4 with scheduling
- **Batch Size**: 256-1024 depending on memory
- **Epochs**: 50-200 with early stopping
- **Weight Decay**: 1e-5 to 1e-3

### Embedding Parameters
- **Embedding Dimension**: min(50, cardinality//2)
- **Numerical Scaling**: StandardScaler or MinMaxScaler

## Advanced Features

### Custom Loss Functions
```python
def custom_loss(predictions, targets, weights=None):
    base_loss = F.cross_entropy(predictions, targets)
    if weights is not None:
        base_loss = base_loss * weights
    return base_loss.mean()
```

### Model Checkpointing
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoint.pth')
```

### Inference Optimization
```python
# JIT compilation for production
model_jit = torch.jit.script(model)

# ONNX export
torch.onnx.export(model, dummy_input, "model.onnx")
```

*TorchMLP provides a solid, framework-native implementation of neural networks for tabular data with excellent integration into PyTorch-based ML pipelines.* 