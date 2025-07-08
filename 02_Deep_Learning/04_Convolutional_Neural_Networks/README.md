# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks represent a revolutionary architecture in deep learning, specifically designed to process grid-like data such as images. CNNs leverage the mathematical operation of convolution to automatically learn spatial hierarchies of features, making them exceptionally effective for computer vision tasks. The architecture's design is inspired by the visual cortex, where neurons respond to specific visual patterns in localized receptive fields.

## Mathematical Foundation

### Discrete Convolution Operation

**2D Convolution (Fundamental Operation):**
```
(I * K)(i,j) = ∑∑ I(m,n) · K(i-m, j-n)
               m n
```

Where:
- **I**: Input feature map (image)
- **K**: Kernel (filter)
- **i,j**: Output position coordinates

**Cross-Correlation (Commonly Used in Deep Learning):**
```
(I ⊗ K)(i,j) = ∑∑ I(i+m, j+n) · K(m,n)
               m n
```

**Valid Convolution (No Padding):**
For input size H×W and kernel size K_h×K_w:
```
Output size: (H - K_h + 1) × (W - K_w + 1)
```

### Multi-Channel Convolution

**3D Convolution (RGB Images):**
```
Y(i,j) = ∑∑∑ X(i+u, j+v, c) · W(u,v,c) + b
         u v c
```

Where:
- **X ∈ ℝ^{H×W×C}**: Input with C channels
- **W ∈ ℝ^{K_h×K_w×C}**: Filter weights
- **b ∈ ℝ**: Bias term
- **Y ∈ ℝ^{H'×W'}**: Output feature map

**Multiple Output Channels:**
```
Y^{(k)}(i,j) = ∑∑∑ X(i+u, j+v, c) · W^{(k)}(u,v,c) + b^{(k)}
               u v c
```

For k = 1, 2, ..., K output channels.

## Convolutional Layer Mathematics

### Forward Pass

**Layer Parameters:**
- **Input**: X ∈ ℝ^{N×H×W×C_in}
- **Weights**: W ∈ ℝ^{K_h×K_w×C_in×C_out}
- **Bias**: b ∈ ℝ^{C_out}
- **Output**: Y ∈ ℝ^{N×H'×W'×C_out}

**Convolution Computation:**
```
Y[n,i,j,k] = ∑∑∑ X[n, i·s+u, j·s+v, c] · W[u,v,c,k] + b[k]
             u v c
```

Where:
- **N**: Batch size
- **s**: Stride
- **u,v**: Kernel spatial indices
- **c**: Input channel index
- **k**: Output channel index

### Output Size Calculation

**With Padding and Stride:**
```
H_out = ⌊(H_in + 2p - K_h)/s⌋ + 1
W_out = ⌊(W_in + 2p - K_w)/s⌋ + 1
```

Where:
- **p**: Padding size
- **s**: Stride
- **⌊·⌋**: Floor function

**Padding Types:**
- **Valid**: p = 0
- **Same**: p = (K-1)/2 (for odd K)
- **Full**: p = K-1

### Receptive Field Analysis

**Receptive Field Size:**
For a layer l with kernel size K_l and stride s_l:
```
RF_l = RF_{l-1} + (K_l - 1) · ∏_{i=1}^{l-1} s_i
```

**Initial Condition:**
```
RF_0 = 1  (input pixel)
```

**Jump (effective stride):**
```
J_l = J_{l-1} · s_l
J_0 = 1
```

## Pooling Operations

### Max Pooling

**Operation:**
```
Y(i,j) = max{X(i·s+u, j·s+v) : 0 ≤ u,v < K}
```

**Gradient (Backpropagation):**
```
∂L/∂X(m,n) = {∂L/∂Y(i,j) if (m,n) = argmax, 0 otherwise}
```

### Average Pooling

**Operation:**
```
Y(i,j) = (1/K²) ∑∑ X(i·s+u, j·s+v)
                u v
```

**Gradient:**
```
∂L/∂X(m,n) = (1/K²) ∑ ∂L/∂Y(i,j)
                     {(i,j): (m,n) in pooling region}
```

### Global Average Pooling

**Operation:**
```
Y_c = (1/HW) ∑∑ X(i,j,c)
              i j
```

Reduces spatial dimensions to 1×1 per channel.

## Advanced Pooling Variants

### Adaptive Pooling

**Adaptive Average Pooling:**
```
kernel_size = ⌈input_size / output_size⌉
stride = ⌊input_size / output_size⌋
```

### Fractional Max Pooling

**Stochastic Pooling Regions:**
Uses random overlapping regions for regularization.

## Parameter Calculation

### Layer Parameters

**Convolutional Layer:**
```
Parameters = K_h × K_w × C_in × C_out + C_out
           = Weights + Biases
```

**Memory for Activations:**
```
Memory = N × H_out × W_out × C_out × sizeof(float)
```

### Computational Complexity

**Forward Pass:**
```
FLOPs = N × H_out × W_out × C_out × (K_h × K_w × C_in + 1)
```

**Backward Pass:**
```
FLOPs_backward ≈ 2 × FLOPs_forward
```

## Backpropagation in CNNs

### Gradient Computation

**Loss w.r.t. Feature Map:**
```
∂L/∂X[i,j,c] = ∑∑∑ ∂L/∂Y[u,v,k] · W[i-u·s, j-v·s, c, k]
                u v k
```

**Loss w.r.t. Weights:**
```
∂L/∂W[u,v,c,k] = ∑∑∑ ∂L/∂Y[i,j,k] · X[i·s+u, j·s+v, c]
                  n i j
```

**Loss w.r.t. Bias:**
```
∂L/∂b[k] = ∑∑∑ ∂L/∂Y[n,i,j,k]
           n i j
```

### Efficient Convolution Implementation

**im2col Transformation:**
Converts convolution to matrix multiplication:
```
Y = X_unfolded · W_reshaped
```

Where X_unfolded ∈ ℝ^{(H_out×W_out)×(K_h×K_w×C_in)}

## Activation Functions in CNNs

### ReLU and Variants

**ReLU:**
```
f(x) = max(0, x)
f'(x) = {1 if x > 0, 0 if x ≤ 0}
```

**Leaky ReLU:**
```
f(x) = {x if x > 0, αx if x ≤ 0}
f'(x) = {1 if x > 0, α if x ≤ 0}
```

**ELU (Exponential Linear Unit):**
```
f(x) = {x if x > 0, α(e^x - 1) if x ≤ 0}
f'(x) = {1 if x > 0, f(x) + α if x ≤ 0}
```

## Normalization Techniques

### Batch Normalization

**Forward Pass:**
```
μ_B = (1/m) ∑ x_i                    (batch mean)
σ²_B = (1/m) ∑ (x_i - μ_B)²         (batch variance)
x̂_i = (x_i - μ_B)/√(σ²_B + ε)      (normalize)
y_i = γx̂_i + β                      (scale and shift)
```

**Backward Pass:**
```
∂L/∂x̂_i = ∂L/∂y_i · γ
∂L/∂σ²_B = ∑ ∂L/∂x̂_i · (x_i - μ_B) · (-1/2)(σ²_B + ε)^{-3/2}
∂L/∂μ_B = ∑ ∂L/∂x̂_i · (-1/√(σ²_B + ε)) + ∂L/∂σ²_B · (-2/m)∑(x_i - μ_B)
∂L/∂x_i = ∂L/∂x̂_i/√(σ²_B + ε) + ∂L/∂σ²_B · 2(x_i - μ_B)/m + ∂L/∂μ_B/m
```

### Layer Normalization

**Operation:**
```
LN(x) = γ ⊙ (x - μ)/σ + β
```

Where μ and σ are computed across features (not batch).

## Weight Initialization

### Xavier/Glorot Initialization

**For Symmetric Activations (tanh):**
```
W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
```

**Gaussian Version:**
```
W ~ N(0, 2/(n_in + n_out))
```

### He Initialization

**For ReLU Networks:**
```
W ~ N(0, 2/n_in)
```

**Uniform Version:**
```
W ~ U[-√(6/n_in), √(6/n_in)]
```

## CNN Architectures

### LeNet-5 (1998)

**Architecture:**
```
Input(32×32) → Conv(6@5×5) → Pool(2×2) → Conv(16@5×5) → Pool(2×2) → FC(120) → FC(84) → FC(10)
```

**Parameter Count:**
```
≈ 60,000 parameters
```

### AlexNet (2012)

**Architecture:**
```
Input(224×224×3) → Conv(96@11×11,s=4) → Pool → Conv(256@5×5) → Pool → 
Conv(384@3×3) → Conv(384@3×3) → Conv(256@3×3) → Pool → FC(4096) → FC(4096) → FC(1000)
```

**Innovations:**
- ReLU activation
- Dropout regularization
- Data augmentation
- GPU training

### VGGNet (2014)

**Design Principle:**
Use very small (3×3) convolution filters consistently.

**VGG-16 Architecture:**
```
[Conv(64@3×3)]×2 → Pool → [Conv(128@3×3)]×2 → Pool → 
[Conv(256@3×3)]×3 → Pool → [Conv(512@3×3)]×3 → Pool → 
[Conv(512@3×3)]×3 → Pool → FC(4096) → FC(4096) → FC(1000)
```

**Parameter Count:**
```
≈ 138 million parameters
```

## Advanced CNN Concepts

### Dilated Convolution

**Operation:**
```
Y(i,j) = ∑∑ X(i + r·u, j + r·v) · W(u,v)
         u v
```

Where r is the dilation rate.

**Receptive Field:**
```
RF = 1 + (K-1) × r
```

### Separable Convolution

**Depthwise Separable Convolution:**
1. **Depthwise**: Apply one filter per input channel
2. **Pointwise**: 1×1 convolution to combine channels

**Parameter Reduction:**
```
Standard: K × K × C_in × C_out
Separable: K × K × C_in + C_in × C_out
Ratio: (K × K × C_in × C_out)/(K × K × C_in + C_in × C_out)
```

### Transposed Convolution

**Operation (Upsampling):**
```
Y[i,j] = ∑∑ X[⌊i/s⌋ - u, ⌊j/s⌋ - v] · W[u,v]
         u v
```

**Output Size:**
```
H_out = (H_in - 1) × s - 2p + K
```

## Regularization in CNNs

### Dropout

**Standard Dropout:**
```
y = {x/p with probability p, 0 with probability 1-p}
```

**Spatial Dropout:**
Drop entire feature maps rather than individual neurons.

### Data Augmentation

**Geometric Transformations:**
- Rotation: R(θ)
- Translation: T(δx, δy)
- Scaling: S(sx, sy)
- Shearing: Sh(sx, sy)

**Photometric Transformations:**
- Brightness: I' = I + β
- Contrast: I' = αI
- Saturation: Adjust color saturation
- Hue: Shift color hue

### Cutout/Random Erasing

**Random Rectangular Masks:**
```
I'(x,y) = {0 if (x,y) in mask region, I(x,y) otherwise}
```

## Loss Functions for CNNs

### Classification Losses

**Cross-Entropy:**
```
L = -∑ y_i log(ŷ_i)
    i
```

**Focal Loss (for Imbalanced Data):**
```
FL(p_t) = -α_t(1-p_t)^γ log(p_t)
```

### Segmentation Losses

**Dice Loss:**
```
Dice = 1 - (2|P ∩ T|)/(|P| + |T|)
```

**IoU Loss:**
```
IoU = |P ∩ T|/|P ∪ T|
L_IoU = 1 - IoU
```

## Optimization for CNNs

### Gradient-Based Optimizers

**SGD with Momentum:**
```
v_t = μv_{t-1} + ∇L
θ_{t+1} = θ_t - ηv_t
```

**Adam:**
```
m_t = β_1m_{t-1} + (1-β_1)∇L
v_t = β_2v_{t-1} + (1-β_2)(∇L)²
θ_{t+1} = θ_t - η(m̂_t)/(√v̂_t + ε)
```

### Learning Rate Scheduling

**Step Decay:**
```
η_t = η_0 × γ^⌊t/step_size⌋
```

**Cosine Annealing:**
```
η_t = η_min + (η_max - η_min)(1 + cos(πt/T))/2
```

## CNN Visualization and Interpretability

### Feature Visualization

**Gradient-Based Methods:**
```
∂L/∂I = ∇_I L(f(I), target)
```

**Grad-CAM:**
```
L_c^{Grad-CAM} = ReLU(∑_k α_k^c A^k)
```

Where:
```
α_k^c = (1/Z) ∑∑ ∂y^c/∂A_{i,j}^k
              i j
```

### Saliency Maps

**Vanilla Gradients:**
```
S(i,j) = |∂L/∂I(i,j)|
```

**Guided Backpropagation:**
Modify ReLU backward pass to only propagate positive gradients.

## Memory Optimization

### Gradient Checkpointing

**Trade Computation for Memory:**
Store only subset of activations, recompute others during backward pass.

**Memory Reduction:**
```
Memory_saved = O(√n) vs O(n)
Computation_overhead = 1 additional forward pass
```

### Mixed Precision Training

**FP16 + FP32:**
```
Forward: FP16
Gradients: FP16 → FP32 accumulation
Weights: FP32
```

**Loss Scaling:**
```
L_scaled = scale_factor × L
∇_scaled = scale_factor × ∇L
∇_unscaled = ∇_scaled / scale_factor
```

## Modern CNN Innovations

### Residual Connections (ResNet)

**Skip Connection:**
```
y = F(x, W) + x
```

**Bottleneck Block:**
```
y = F_3(F_2(F_1(x))) + x
```

Where F_1: 1×1 conv, F_2: 3×3 conv, F_3: 1×1 conv

### Dense Connections (DenseNet)

**Dense Block:**
```
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

**Growth Rate:**
```
Feature maps per layer = k (growth rate)
Total features in layer l = k_0 + k × l
```

### Attention Mechanisms

**Spatial Attention:**
```
A(i,j) = softmax(f(F(i,j)))
F' = A ⊙ F
```

**Channel Attention (SENet):**
```
s = GlobalAvgPool(F)
α = σ(W_2(ReLU(W_1(s))))
F' = α ⊙ F
```

## Computational Efficiency

### Quantization

**8-bit Quantization:**
```
x_quantized = round((x - min_val) × 255/(max_val - min_val))
```

**Dynamic Range:**
```
scale = (max_val - min_val)/255
zero_point = round(-min_val/scale)
```

### Knowledge Distillation

**Soft Targets:**
```
L = αL_CE(y, y_hard) + (1-α)τ²L_KL(σ(z_s/τ), σ(z_t/τ))
```

Where:
- τ: Temperature parameter
- z_s, z_t: Student and teacher logits

## Mathematical Properties

### Translation Equivariance

**Property:**
```
f(T_δ(x)) = T_δ(f(x))
```

Where T_δ is translation by δ.

### Scale and Rotation Invariance

**Data Augmentation Approach:**
Train on transformed versions to approximate invariance.

**Theoretical Invariance:**
Requires specific architectural designs (e.g., group convolutions).

## Implementation Considerations

### Numerical Stability

**Overflow Prevention:**
```
# Softmax stability
def stable_softmax(x):
    return exp(x - max(x)) / sum(exp(x - max(x)))
```

**Gradient Clipping:**
```
if ||∇|| > threshold:
    ∇ = ∇ × threshold/||∇||
```

### Memory-Efficient Implementation

**In-Place Operations:**
```
# ReLU in-place
x[x < 0] = 0
```

**Gradient Accumulation:**
```
for mini_batch in batches:
    loss = forward(mini_batch) / accumulation_steps
    loss.backward()
    if step % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Performance Metrics

### Classification Metrics

**Top-k Accuracy:**
```
Acc@k = (1/N) ∑ I(true_label ∈ top_k_predictions)
                i
```

**Precision/Recall:**
```
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2 × (Precision × Recall)/(Precision + Recall)
```

### Computational Metrics

**FLOPs (Floating Point Operations):**
```
FLOPs = ∑ (Operations per layer)
        layers
```

**Model Size:**
```
Size = ∑ (Parameters per layer × bits per parameter)/8
       layers
```

## Mathematical Summary

Convolutional Neural Networks represent a profound application of mathematical principles to computer vision:

1. **Convolution Operation**: Exploits translation equivariance for efficient feature detection
2. **Hierarchical Feature Learning**: Combines local patterns into increasingly complex representations  
3. **Parameter Sharing**: Dramatically reduces model complexity compared to fully connected networks
4. **Spatial Inductive Bias**: Architecture naturally handles 2D spatial relationships

**Key Mathematical Insight**: The discrete convolution operation, when combined with nonlinear activations and learned parameters, creates a powerful function approximator that respects the spatial structure of visual data. The mathematical framework enables automatic feature hierarchy learning, from edges and textures to complex object representations.

**Theoretical Foundation**: CNNs can be viewed as implementing a form of template matching with learnable templates. The mathematical properties of convolution (linearity, translation equivariance) combined with nonlinear activations and hierarchical composition create universal approximators for visual pattern recognition tasks. 