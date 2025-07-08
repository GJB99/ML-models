# Feedforward and Backpropagation

Feedforward and backpropagation constitute the fundamental training mechanism of neural networks, enabling them to learn complex patterns from data through iterative parameter optimization. The feedforward pass computes predictions by propagating inputs through the network, while backpropagation uses the chain rule of calculus to efficiently compute gradients for parameter updates. This elegant mathematical framework, formalized by Rumelhart, Hinton, and Williams in 1986, revolutionized machine learning and remains the cornerstone of modern deep learning.

## Mathematical Framework

### Network Architecture

**Multi-Layer Neural Network:**
```
Input Layer: x ∈ ℝⁿ⁰
Hidden Layers: h⁽ˡ⁾ ∈ ℝⁿˡ, l = 1, 2, ..., L-1
Output Layer: y ∈ ℝⁿᴸ
```

**Layer Dimensions:**
- **n⁰**: input dimension
- **nˡ**: number of neurons in layer l
- **L**: total number of layers (including output)

**Network Parameters:**
```
Weights: W⁽ˡ⁾ ∈ ℝⁿˡ⁻¹ˣⁿˡ
Biases: b⁽ˡ⁾ ∈ ℝⁿˡ
```

For layers l = 1, 2, ..., L.

### Activation Functions

**Common Activation Functions:**

**Sigmoid:**
```
σ(z) = 1/(1 + e⁻ᶻ)
σ'(z) = σ(z)(1 - σ(z))
```

**Hyperbolic Tangent:**
```
tanh(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)
tanh'(z) = 1 - tanh²(z)
```

**ReLU (Rectified Linear Unit):**
```
ReLU(z) = max(0, z)
ReLU'(z) = {1 if z > 0, 0 if z ≤ 0}
```

**Leaky ReLU:**
```
LeakyReLU(z) = max(αz, z), α ∈ (0, 1)
LeakyReLU'(z) = {1 if z > 0, α if z ≤ 0}
```

**Softmax (Output Layer):**
```
Softmax(z)ᵢ = eᶻⁱ / ∑ⱼ₌₁ᵏ eᶻʲ
```

## Feedforward Propagation

### Layer-wise Computation

**Linear Transformation:**
```
z⁽ˡ⁾ = W⁽ˡ⁾ᵀa⁽ˡ⁻¹⁾ + b⁽ˡ⁾
```

**Activation:**
```
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
```

Where f⁽ˡ⁾ is the activation function for layer l.

**Complete Forward Pass:**
```
a⁽⁰⁾ = x                           (input)
z⁽¹⁾ = W⁽¹⁾ᵀa⁽⁰⁾ + b⁽¹⁾          (first hidden layer)
a⁽¹⁾ = f⁽¹⁾(z⁽¹⁾)
⋮
z⁽ᴸ⁾ = W⁽ᴸ⁾ᵀa⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾        (output layer)
a⁽ᴸ⁾ = f⁽ᴸ⁾(z⁽ᴸ⁾)
```

**Network Output:**
```
ŷ = a⁽ᴸ⁾
```

### Matrix Formulation

**Batch Processing:**
For a batch of m samples X ∈ ℝᵐˣⁿ⁰:

```
Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾W⁽ˡ⁾ + 1ₘb⁽ˡ⁾ᵀ
A⁽ˡ⁾ = f⁽ˡ⁾(Z⁽ˡ⁾)
```

Where:
- **Z⁽ˡ⁾ ∈ ℝᵐˣⁿˡ**: pre-activation values for batch
- **A⁽ˡ⁾ ∈ ℝᵐˣⁿˡ**: activation values for batch
- **1ₘ**: m-dimensional vector of ones

## Loss Functions

### Regression Losses

**Mean Squared Error (MSE):**
```
L(y, ŷ) = (1/2)||y - ŷ||² = (1/2)∑ᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Mean Absolute Error (MAE):**
```
L(y, ŷ) = ||y - ŷ||₁ = ∑ᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

### Classification Losses

**Cross-Entropy (Multi-class):**
```
L(y, ŷ) = -∑ᵢ₌₁ᵏ yᵢ log(ŷᵢ)
```

**Binary Cross-Entropy:**
```
L(y, ŷ) = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Logistic Loss:**
```
L(y, ŷ) = log(1 + e⁻ʸŷ)
```

## Backpropagation Algorithm

### Chain Rule Foundation

**Objective:**
Compute gradients ∂L/∂W⁽ˡ⁾ and ∂L/∂b⁽ˡ⁾ for all layers l.

**Chain Rule Application:**
```
∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂b⁽ˡ⁾
```

### Error Propagation

**Output Layer Error:**
```
δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = ∂L/∂a⁽ᴸ⁾ ⊙ ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = ∂L/∂a⁽ᴸ⁾ ⊙ f'⁽ᴸ⁾(z⁽ᴸ⁾)
```

Where ⊙ denotes element-wise multiplication.

**Hidden Layer Errors (Backward Propagation):**
```
δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ = (W⁽ˡ⁺¹⁾δ⁽ˡ⁺¹⁾) ⊙ f'⁽ˡ⁾(z⁽ˡ⁾)
```

For l = L-1, L-2, ..., 1.

### Gradient Computation

**Weight Gradients:**
```
∂L/∂W⁽ˡ⁾ = a⁽ˡ⁻¹⁾δ⁽ˡ⁾ᵀ
```

**Bias Gradients:**
```
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### Matrix Form (Batch Processing)

**Batch Error Computation:**
```
Δ⁽ᴸ⁾ = ∂L/∂A⁽ᴸ⁾ ⊙ f'⁽ᴸ⁾(Z⁽ᴸ⁾)
Δ⁽ˡ⁾ = (Δ⁽ˡ⁺¹⁾W⁽ˡ⁺¹⁾ᵀ) ⊙ f'⁽ˡ⁾(Z⁽ˡ⁾)
```

**Batch Gradients:**
```
∂L/∂W⁽ˡ⁾ = (1/m)A⁽ˡ⁻¹⁾ᵀΔ⁽ˡ⁾
∂L/∂b⁽ˡ⁾ = (1/m)∑ᵢ₌₁ᵐ Δᵢ⁽ˡ⁾
```

## Detailed Algorithm

### Complete Backpropagation Algorithm

```
Algorithm: Backpropagation
Input: Training set {(x⁽ⁱ⁾, y⁽ⁱ⁾)}ᵢ₌₁ᵐ, Network architecture, Learning rate η
Output: Trained network parameters

1. Initialize weights W⁽ˡ⁾ and biases b⁽ˡ⁾ randomly

2. For each epoch:
   For each mini-batch:
   
   // Forward Pass
   a. Set a⁽⁰⁾ = x
   b. For l = 1 to L:
      z⁽ˡ⁾ = W⁽ˡ⁾ᵀa⁽ˡ⁻¹⁾ + b⁽ˡ⁾
      a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
   
   // Compute Loss
   c. L = loss_function(y, a⁽ᴸ⁾)
   
   // Backward Pass
   d. Compute δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾
   e. For l = L-1 down to 1:
      δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾δ⁽ˡ⁺¹⁾) ⊙ f'⁽ˡ⁾(z⁽ˡ⁾)
   
   // Gradient Computation
   f. For l = 1 to L:
      ∂L/∂W⁽ˡ⁾ = a⁽ˡ⁻¹⁾δ⁽ˡ⁾ᵀ
      ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
   
   // Parameter Update
   g. For l = 1 to L:
      W⁽ˡ⁾ = W⁽ˡ⁾ - η∂L/∂W⁽ˡ⁾
      b⁽ˡ⁾ = b⁽ˡ⁾ - η∂L/∂b⁽ˡ⁾
```

## Specific Loss Function Derivatives

### Mean Squared Error

**Loss Function:**
```
L = (1/2)||y - a⁽ᴸ⁾||²
```

**Output Layer Error:**
```
δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ f'⁽ᴸ⁾(z⁽ᴸ⁾)
```

**For Linear Output (f⁽ᴸ⁾(z) = z):**
```
δ⁽ᴸ⁾ = a⁽ᴸ⁾ - y
```

### Cross-Entropy with Softmax

**Softmax Output:**
```
a⁽ᴸ⁾ᵢ = e^{z⁽ᴸ⁾ᵢ} / ∑ⱼ e^{z⁽ᴸ⁾ⱼ}
```

**Cross-Entropy Loss:**
```
L = -∑ᵢ yᵢ log(a⁽ᴸ⁾ᵢ)
```

**Combined Derivative:**
```
δ⁽ᴸ⁾ = a⁽ᴸ⁾ - y
```

This remarkably simple result emerges from the mathematical properties of softmax and cross-entropy.

### Binary Cross-Entropy with Sigmoid

**Sigmoid Output:**
```
a⁽ᴸ⁾ = σ(z⁽ᴸ⁾) = 1/(1 + e^{-z⁽ᴸ⁾})
```

**Binary Cross-Entropy:**
```
L = -[y log(a⁽ᴸ⁾) + (1-y) log(1-a⁽ᴸ⁾)]
```

**Combined Derivative:**
```
δ⁽ᴸ⁾ = a⁽ᴸ⁾ - y
```

## Computational Complexity

### Forward Pass

**Per Layer Computation:**
- **Matrix Multiplication**: O(n^{l-1} × n^l)
- **Activation Function**: O(n^l)

**Total Forward Pass:**
```
O(∑_{l=1}^L n^{l-1} × n^l)
```

### Backward Pass

**Error Propagation:**
- **Per Layer**: O(n^l × n^{l+1})

**Gradient Computation:**
- **Weight Gradients**: O(n^{l-1} × n^l)
- **Bias Gradients**: O(n^l)

**Total Backward Pass:**
```
O(∑_{l=1}^L n^{l-1} × n^l)
```

**Overall Complexity:**
Same order as forward pass: O(∑_{l=1}^L n^{l-1} × n^l)

## Activation Function Derivatives

### Detailed Derivative Computations

**Sigmoid Derivative:**
```
f(z) = 1/(1 + e^{-z})
f'(z) = f(z)(1 - f(z))
```

**Tanh Derivative:**
```
f(z) = tanh(z)
f'(z) = 1 - tanh²(z) = 1 - f²(z)
```

**ReLU Derivative:**
```
f'(z) = {1 if z > 0, 0 if z ≤ 0}
```

**Softmax Derivative:**
For softmax vector s = [s₁, s₂, ..., sₖ]:
```
∂sᵢ/∂zⱼ = {sᵢ(1-sᵢ) if i=j, -sᵢsⱼ if i≠j}
```

**Jacobian Matrix:**
```
J = diag(s) - ssᵀ
```

## Vanishing and Exploding Gradients

### Gradient Flow Analysis

**Gradient Magnitude:**
```
||∂L/∂W⁽¹⁾|| = ||∂L/∂z⁽ᴸ⁾|| × ∏_{l=2}^L ||W⁽ˡ⁾|| × ||f'⁽ˡ⁾(z⁽ˡ⁾)||
```

**Vanishing Gradients:**
When ∏_{l=2}^L ||W⁽ˡ⁾|| × ||f'⁽ˡ⁾(z⁽ˡ⁾)|| → 0

**Exploding Gradients:**
When ∏_{l=2}^L ||W⁽ˡ⁾|| × ||f'⁽ˡ⁾(z⁽ˡ⁾)|| → ∞

### Mitigation Strategies

**Weight Initialization:**
- **Xavier/Glorot**: Var(W) = 1/n_{in}
- **He**: Var(W) = 2/n_{in} (for ReLU)

**Gradient Clipping:**
```
g = ∇L
if ||g|| > threshold:
    g = g × threshold/||g||
```

**Batch Normalization:**
```
x̂ = (x - μ)/√(σ² + ε)
y = γx̂ + β
```

## Advanced Techniques

### Momentum

**Momentum Update:**
```
v^{(t+1)} = βv^{(t)} + ∇L
θ^{(t+1)} = θ^{(t)} - ηv^{(t+1)}
```

**Nesterov Momentum:**
```
v^{(t+1)} = βv^{(t)} + ∇L(θ^{(t)} - ηβv^{(t)})
θ^{(t+1)} = θ^{(t)} - ηv^{(t+1)}
```

### Adaptive Learning Rates

**AdaGrad:**
```
G^{(t+1)} = G^{(t)} + (∇L)²
θ^{(t+1)} = θ^{(t)} - η∇L/√(G^{(t+1)} + ε)
```

**Adam:**
```
m^{(t+1)} = β₁m^{(t)} + (1-β₁)∇L
v^{(t+1)} = β₂v^{(t)} + (1-β₂)(∇L)²
θ^{(t+1)} = θ^{(t)} - η(m̂^{(t+1)})/(√v̂^{(t+1)} + ε)
```

Where m̂ and v̂ are bias-corrected estimates.

## Regularization Techniques

### L2 Regularization (Weight Decay)

**Modified Loss:**
```
L_total = L + λ∑_{l=1}^L ||W⁽ˡ⁾||²_F
```

**Modified Gradients:**
```
∂L_total/∂W⁽ˡ⁾ = ∂L/∂W⁽ˡ⁾ + 2λW⁽ˡ⁾
```

### Dropout

**Training Phase:**
```
r⁽ˡ⁾ ~ Bernoulli(p)
ã⁽ˡ⁾ = r⁽ˡ⁾ ⊙ a⁽ˡ⁾
```

**Inference Phase:**
```
a⁽ˡ⁾_inference = p × a⁽ˡ⁾
```

## Automatic Differentiation

### Computational Graph

**Forward Mode:**
Compute derivatives alongside function values:
```
∂f/∂x = ∂f/∂u · ∂u/∂x  (chain rule)
```

**Reverse Mode (Backpropagation):**
Traverse graph backward, accumulating gradients:
```
∂L/∂x = ∑_children ∂L/∂child · ∂child/∂x
```

### Implementation Considerations

**Memory Trade-off:**
- **Forward pass**: Store activations for backward pass
- **Gradient checkpointing**: Recompute some activations to save memory

**Numerical Stability:**
- Use numerically stable implementations of activation functions
- Careful handling of edge cases (e.g., log(0), exp(large_number))

## Universal Approximation Theorem

### Theoretical Foundation

**Theorem (Cybenko, 1989):**
A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary accuracy, provided the activation function is non-constant, bounded, and monotonically-increasing.

**Mathematical Statement:**
For any continuous function f: [0,1]ⁿ → ℝ and ε > 0, there exists an integer N and parameters such that:
```
|F(x) - f(x)| < ε  for all x ∈ [0,1]ⁿ
```

Where:
```
F(x) = ∑ᵢ₌₁ᴺ αᵢσ(wᵢᵀx + bᵢ)
```

## Practical Implementation

### Numerical Considerations

**Overflow Prevention:**
```
# Softmax computation
def stable_softmax(x):
    exp_x = exp(x - max(x))
    return exp_x / sum(exp_x)
```

**Underflow Prevention:**
```
# Log-softmax for numerical stability
log_softmax(x) = x - log(sum(exp(x)))
```

### Efficient Implementation

**Vectorization:**
```python
# Instead of loops, use matrix operations
Z = np.dot(A, W) + b  # Vectorized computation
A = activation_function(Z)
```

**Memory Layout:**
- Use row-major order for cache efficiency
- Minimize memory allocations in inner loops

### Debugging Techniques

**Gradient Checking:**
```
gradient_numerical = (f(θ + ε) - f(θ - ε))/(2ε)
gradient_analytical = backprop_gradient
assert abs(gradient_numerical - gradient_analytical) < tolerance
```

**Learning Rate Validation:**
- Too high: Loss oscillates or diverges
- Too low: Slow convergence
- Just right: Smooth, steady decrease

## Mathematical Summary

Feedforward and backpropagation represent one of the most elegant applications of calculus to machine learning:

1. **Chain Rule**: Enables efficient gradient computation through composite functions
2. **Matrix Calculus**: Provides compact representation for batch processing
3. **Function Composition**: Networks as compositions of simple functions
4. **Optimization Theory**: Gradient-based parameter optimization

The mathematical beauty lies in how the chain rule transforms a seemingly intractable optimization problem into a series of local computations that can be efficiently executed.

**Key Insight**: Backpropagation is simply the chain rule applied systematically to neural networks. By decomposing complex functions into elementary operations and caching intermediate results, it achieves computational efficiency that scales linearly with network depth. This mathematical framework enables training of arbitrarily deep networks, forming the foundation of modern deep learning. 