# Recurrent Neural Networks (RNNs)

Recurrent Neural Networks represent a fundamental paradigm shift in neural architecture design, specifically engineered to process sequential data by maintaining hidden state information across time steps. Unlike feedforward networks that process fixed-size inputs independently, RNNs can handle variable-length sequences by sharing parameters across time and maintaining memory through recurrent connections. This mathematical framework enables modeling of temporal dependencies, making RNNs particularly powerful for time series analysis, natural language processing, and any domain where sequential patterns matter.

## Mathematical Foundation

### Basic RNN Architecture

**Hidden State Evolution:**
```
h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
```

**Output Generation:**
```
y_t = g(W_hy · h_t + b_y)
```

Where:
- **h_t ∈ ℝⁿʰ**: Hidden state at time t
- **x_t ∈ ℝⁿˣ**: Input at time t  
- **y_t ∈ ℝⁿʸ**: Output at time t
- **W_hh ∈ ℝⁿʰˣⁿʰ**: Hidden-to-hidden weight matrix
- **W_xh ∈ ℝⁿˣˣⁿʰ**: Input-to-hidden weight matrix
- **W_hy ∈ ℝⁿʰˣⁿʸ**: Hidden-to-output weight matrix
- **b_h ∈ ℝⁿʰ, b_y ∈ ℝⁿʸ**: Bias vectors
- **f, g**: Activation functions (typically tanh, ReLU, or sigmoid)

### Unfolded Network Representation

**Computational Graph Over Time:**
```
h_0 → h_1 → h_2 → ... → h_T
 ↓     ↓     ↓           ↓
y_0   y_1   y_2  ...   y_T
 ↑     ↑     ↑           ↑
x_0   x_1   x_2  ...   x_T
```

**Recursive Definition:**
```
h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
h_0 = initial state (often zeros)
```

**Explicit Expansion:**
```
h_1 = f(W_hh · h_0 + W_xh · x_1 + b_h)
h_2 = f(W_hh · f(W_hh · h_0 + W_xh · x_1 + b_h) + W_xh · x_2 + b_h)
h_3 = f(W_hh · h_2 + W_xh · x_3 + b_h)
...
```

## RNN Variants and Configurations

### Many-to-One (Sequence Classification)
```
Input:  [x_1, x_2, ..., x_T]
Output: y (single prediction)
```

**Implementation:**
```
for t in 1..T:
    h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
y = g(W_hy · h_T + b_y)
```

### One-to-Many (Sequence Generation)
```
Input:  x (single input)
Output: [y_1, y_2, ..., y_T]
```

**Implementation:**
```
h_0 = f(W_xh · x + b_h)
for t in 1..T:
    y_t = g(W_hy · h_{t-1} + b_y)
    h_t = f(W_hh · h_{t-1} + W_yh · y_t + b_h)  # Use previous output as input
```

### Many-to-Many (Sequence-to-Sequence)
```
Input:  [x_1, x_2, ..., x_T]
Output: [y_1, y_2, ..., y_T]
```

**Synchronized (Same Length):**
```
for t in 1..T:
    h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
    y_t = g(W_hy · h_t + b_y)
```

**Encoder-Decoder (Different Lengths):**
```
# Encoder
for t in 1..T_enc:
    h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
c = h_{T_enc}  # Context vector

# Decoder  
h_0^{dec} = c
for t in 1..T_dec:
    h_t^{dec} = f(W_hh^{dec} · h_{t-1}^{dec} + W_xh^{dec} · x_t^{dec} + b_h^{dec})
    y_t = g(W_hy^{dec} · h_t^{dec} + b_y^{dec})
```

## Backpropagation Through Time (BPTT)

### Forward Pass

**Complete Forward Computation:**
```
for t in 1..T:
    a_t = W_hh · h_{t-1} + W_xh · x_t + b_h
    h_t = f(a_t)
    o_t = W_hy · h_t + b_y  
    y_t = g(o_t)
```

Store all intermediate values: {a_t, h_t, o_t, y_t} for t = 1..T

### Loss Function

**Sequence Loss (Sum Over Time Steps):**
```
L = ∑_{t=1}^T L_t(y_t, ŷ_t)
```

**Common Loss Functions:**
- **MSE**: L_t = ½||y_t - ŷ_t||²
- **Cross-Entropy**: L_t = -∑_i ŷ_{t,i} log(y_{t,i})

### Backward Pass

**Output Layer Gradients:**
```
∂L/∂W_hy = ∑_{t=1}^T ∂L_t/∂o_t · h_t^T
∂L/∂b_y = ∑_{t=1}^T ∂L_t/∂o_t
```

**Hidden State Gradients (Recursive):**
```
∂L/∂h_T = W_hy^T · ∂L_T/∂o_T

∂L/∂h_t = W_hy^T · ∂L_t/∂o_t + W_hh^T · ∂L/∂h_{t+1} · f'(a_{t+1})
```

For t = T-1, T-2, ..., 1.

**Weight Gradients:**
```
∂L/∂W_hh = ∑_{t=1}^T ∂L/∂h_t · f'(a_t) · h_{t-1}^T
∂L/∂W_xh = ∑_{t=1}^T ∂L/∂h_t · f'(a_t) · x_t^T
∂L/∂b_h = ∑_{t=1}^T ∂L/∂h_t · f'(a_t)
```

### Gradient Flow Analysis

**Gradient Through Time:**
```
∂L/∂h_0 = ∂L/∂h_T · ∏_{t=1}^T (W_hh^T · f'(a_t))
```

**Jacobian Product:**
```
J_t = W_hh^T · diag(f'(a_t))
∂L/∂h_0 = ∂L/∂h_T · ∏_{t=1}^T J_t
```

## The Vanishing and Exploding Gradient Problem

### Mathematical Analysis

**Gradient Magnitude:**
```
||∂L/∂h_0|| = ||∂L/∂h_T|| · ||∏_{t=1}^T J_t||
```

**Spectral Radius Condition:**
Let λ_max be the largest eigenvalue of W_hh:

**Vanishing Gradients:**
```
If λ_max · max_t |f'(a_t)| < 1, then ||∏_{t=1}^T J_t|| → 0 as T → ∞
```

**Exploding Gradients:**
```
If λ_max · max_t |f'(a_t)| > 1, then ||∏_{t=1}^T J_t|| → ∞ as T → ∞
```

### Activation Function Impact

**Sigmoid Saturation:**
```
σ(z) = 1/(1 + e^{-z})
σ'(z) = σ(z)(1 - σ(z)) ≤ 0.25
```

When |z| is large, σ'(z) ≈ 0, causing vanishing gradients.

**Tanh Saturation:**
```
tanh(z) = (e^z - e^{-z})/(e^z + e^{-z})
tanh'(z) = 1 - tanh²(z) ≤ 1
```

Similar saturation issues for large |z|.

**ReLU Advantages:**
```
ReLU(z) = max(0, z)
ReLU'(z) = {1 if z > 0, 0 if z ≤ 0}
```

No saturation for positive values, but can cause "dead neurons."

### Mitigation Strategies

**Gradient Clipping:**
```
if ||∇|| > threshold:
    ∇ = ∇ · threshold/||∇||
```

**Proper Weight Initialization:**
```
W_hh ~ N(0, σ²) where σ² = 1/n_h  (Xavier initialization)
```

**Orthogonal Initialization:**
```
W_hh = orthogonal_matrix · λ
```

Where λ controls the spectral radius.

**Skip Connections:**
```
h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h) + αh_{t-1}
```

## Advanced RNN Architectures

### Bidirectional RNNs

**Forward Pass:**
```
h_t^{forward} = f(W_hh^f · h_{t-1}^{forward} + W_xh^f · x_t + b_h^f)
```

**Backward Pass:**
```
h_t^{backward} = f(W_hh^b · h_{t+1}^{backward} + W_xh^b · x_t + b_h^b)
```

**Combined Hidden State:**
```
h_t = [h_t^{forward}; h_t^{backward}]  # Concatenation
```

**Output:**
```
y_t = g(W_hy · h_t + b_y)
```

### Deep RNNs (Stacked)

**Layer-wise Computation:**
```
h_t^{(l)} = f(W_hh^{(l)} · h_{t-1}^{(l)} + W_xh^{(l)} · h_t^{(l-1)} + b_h^{(l)})
```

Where:
- **l**: Layer index
- **h_t^{(0)} = x_t**: Input layer
- **h_t^{(L)}**: Top layer output

**Skip Connections in Deep RNNs:**
```
h_t^{(l)} = f(W_hh^{(l)} · h_{t-1}^{(l)} + W_xh^{(l)} · h_t^{(l-1)} + b_h^{(l)}) + h_t^{(l-1)}
```

## LSTM (Long Short-Term Memory)

### Mathematical Formulation

**Cell State Evolution:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate values
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t         # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate
h_t = o_t ⊙ tanh(C_t)                   # Hidden state
```

Where:
- **⊙**: Element-wise multiplication (Hadamard product)
- **σ**: Sigmoid function
- **[h_{t-1}, x_t]**: Concatenation

**Detailed Gate Analysis:**

**Forget Gate:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
Decides what information to discard from cell state.

**Input Gate:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```
Decides what new information to store in cell state.

**Output Gate:**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```
Controls what parts of cell state to output.

### LSTM Gradient Flow

**Cell State Gradient:**
```
∂L/∂C_t = ∂L/∂h_t · o_t · (1 - tanh²(C_t)) + ∂L/∂C_{t+1} · f_{t+1}
```

**Improved Gradient Flow:**
The additive structure C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t allows gradients to flow more easily through the forget gate connection.

## GRU (Gated Recurrent Unit)

### Mathematical Formulation

**Simplified Gating:**
```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)           # Update gate
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)           # Reset gate  
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)  # Candidate hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t         # Hidden state
```

**Parameter Comparison:**
- **LSTM**: 4 gates × (n_h + n_x + 1) × n_h parameters
- **GRU**: 3 gates × (n_h + n_x + 1) × n_h parameters

**Simplified Architecture:**
GRU combines forget and input gates into a single update gate and merges cell state and hidden state.

## Attention Mechanisms in RNNs

### Basic Attention

**Context Vector:**
```
c_t = ∑_{s=1}^T α_{t,s} · h_s^{encoder}
```

**Attention Weights:**
```
e_{t,s} = a(h_{t-1}^{decoder}, h_s^{encoder})    # Alignment model
α_{t,s} = exp(e_{t,s}) / ∑_{s'=1}^T exp(e_{t,s'}) # Softmax normalization
```

**Decoder with Attention:**
```
h_t^{decoder} = f(W_hh · h_{t-1}^{decoder} + W_ch · c_t + W_xh · x_t + b_h)
```

### Attention Alignment Models

**Additive (Bahdanau) Attention:**
```
a(h^{dec}, h^{enc}) = v_a^T tanh(W_a h^{dec} + U_a h^{enc})
```

**Multiplicative (Luong) Attention:**
```
a(h^{dec}, h^{enc}) = h^{dec,T} W_a h^{enc}
```

**Dot-Product Attention:**
```
a(h^{dec}, h^{enc}) = h^{dec,T} h^{enc}
```

## Optimization and Training

### Learning Rate Scheduling

**Time-Based Decay:**
```
η_t = η_0 / (1 + decay_rate × epoch)
```

**Exponential Decay:**
```
η_t = η_0 × decay_rate^{epoch}
```

### Teacher Forcing vs. Free Running

**Teacher Forcing (Training):**
```
# Use ground truth as input at each step
for t in 1..T:
    h_t = f(W_hh · h_{t-1} + W_xh · y_{t-1}^{true} + b_h)
    y_t = g(W_hy · h_t + b_y)
```

**Free Running (Inference):**
```
# Use model's own predictions
for t in 1..T:
    h_t = f(W_hh · h_{t-1} + W_xh · y_{t-1}^{pred} + b_h)
    y_t = g(W_hy · h_t + b_y)
    y_{t-1}^{pred} = y_t
```

**Scheduled Sampling:**
```
ε_t = ε_0 × λ^t  # Decay probability
x_t = {y_{t-1}^{true} with prob ε_t, y_{t-1}^{pred} with prob 1-ε_t}
```

## Regularization Techniques

### Dropout in RNNs

**Standard Dropout (Input/Output):**
```
h_t = f(W_hh · h_{t-1} + W_xh · dropout(x_t) + b_h)
y_t = dropout(g(W_hy · h_t + b_y))
```

**Recurrent Dropout:**
```
h_t = f(dropout(W_hh) · h_{t-1} + W_xh · x_t + b_h)
```

**Variational Dropout:**
Use same dropout mask across all time steps:
```
m ~ Bernoulli(1-p)
h_t = f(W_hh · (m ⊙ h_{t-1}) + W_xh · x_t + b_h)
```

### Weight Regularization

**L2 Regularization:**
```
L_total = L_sequence + λ(||W_hh||²_F + ||W_xh||²_F + ||W_hy||²_F)
```

**Recurrent Weight Decay:**
```
W_hh ← (1 - λ_r)W_hh - η∇W_hh
```

## Computational Complexity

### Time Complexity

**Forward Pass:**
```
O(T × (n_h² + n_h × n_x + n_h × n_y))
```

**BPTT:**
```
O(T × (n_h² + n_h × n_x + n_h × n_y))
```

**Memory Complexity:**
```
O(T × n_h)  # Store hidden states for BPTT
```

### Truncated BPTT

**Limited Backpropagation:**
```
# Only backpropagate k steps
∂L/∂h_{t-k} = 0  (approximate)
```

**Computational Savings:**
```
Memory: O(k × n_h) instead of O(T × n_h)
Time: O(k × operations) per update
```

## Applications and Use Cases

### Natural Language Processing

**Language Modeling:**
```
P(w_1, w_2, ..., w_T) = ∏_{t=1}^T P(w_t | w_1, ..., w_{t-1})
```

**Machine Translation:**
```
P(y_1, ..., y_T | x_1, ..., x_S) = ∏_{t=1}^T P(y_t | y_1, ..., y_{t-1}, x_1, ..., x_S)
```

### Time Series Prediction

**Autoregressive Model:**
```
x_t = f(x_{t-1}, x_{t-2}, ..., x_{t-p}) + ε_t
```

**State Space Model:**
```
x_t = F_t x_{t-1} + w_t    # State transition
y_t = H_t x_t + v_t        # Observation
```

## Modern Alternatives and Improvements

### Transformer Comparison

**RNN Sequential Processing:**
```
O(T) time complexity (inherently sequential)
O(T × n_h) memory for BPTT
```

**Transformer Parallel Processing:**
```
O(1) time complexity (parallelizable)
O(T²) memory for attention
```

### State Space Models

**Linear State Space:**
```
x_t = A x_{t-1} + B u_t
y_t = C x_t + D u_t
```

**Advantages:**
- Efficient parallel training
- Better long-range dependencies
- Stable gradients

## Implementation Considerations

### Numerical Stability

**Gradient Clipping:**
```
total_norm = sqrt(sum(param.grad.data.norm()**2 for param in parameters))
clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:
    for param in parameters:
        param.grad.data.mul_(clip_coef)
```

**LSTM Initialization:**
```
# Forget gate bias initialization
b_f = 1.0  # Start with remembering everything
```

### Memory Optimization

**Gradient Checkpointing:**
```
# Trade computation for memory
# Recompute forward pass during backward
```

**Dynamic Batching:**
```
# Group sequences of similar length
# Minimize padding overhead
```

## Mathematical Properties

### Universal Approximation

**Theorem:** RNNs with sufficient hidden units can approximate any measurable sequence-to-sequence mapping to arbitrary accuracy.

**Turing Completeness:** RNNs are Turing complete, meaning they can simulate any algorithm given enough time and memory.

### Stability Analysis

**Fixed Points:** 
```
h* = f(W_hh · h* + W_xh · x + b_h)
```

**Lyapunov Stability:** Requires eigenvalues of W_hh to have magnitude < 1.

## Advanced Training Techniques

### Curriculum Learning

**Easy-to-Hard Training:**
```
# Start with short sequences
# Gradually increase sequence length
T_epoch = min(T_max, T_start + epoch × T_increment)
```

### Sequence-Level Training

**REINFORCE Algorithm:**
```
∇_θ J = E[∇_θ log π_θ(y|x) · R(y, y*)]
```

Where R(y, y*) is reward comparing predicted and target sequences.

## Mathematical Summary

Recurrent Neural Networks represent a profound mathematical framework for sequence modeling:

1. **Parameter Sharing**: Same weights applied across time steps enable generalization to variable-length sequences
2. **Memory Mechanism**: Hidden state provides a compressed representation of sequence history
3. **Compositional Structure**: Recursive application of simple transformations creates complex temporal patterns
4. **Gradient Flow**: BPTT enables end-to-end learning but faces fundamental challenges with long sequences

**Key Mathematical Insight**: The recurrent connection h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h) creates a dynamical system where the hidden state evolves over time. The mathematical properties of this system (eigenvalues of W_hh, activation function derivatives) determine the network's ability to capture long-term dependencies.

**Theoretical Foundation**: RNNs can be viewed as discrete-time dynamical systems. The quality of long-term memory depends on the spectral radius of the recurrent weight matrix and the saturation properties of activation functions. Advanced architectures (LSTM, GRU) solve gradient flow problems through careful gating mechanisms that create more favorable optimization landscapes.

The mathematical elegance of RNNs lies in their simplicity - a single recursive equation that, when unfolded in time, creates a deep network capable of processing sequences of arbitrary length while maintaining a fixed parameter budget. 