# Transformers Architecture

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized deep learning by demonstrating that attention mechanisms alone, without recurrence or convolution, can achieve state-of-the-art performance in sequence-to-sequence tasks. This paradigm shift from sequential processing to parallel attention-based computation has become the foundation of modern large language models and numerous other applications. The mathematical elegance of the Transformer lies in its ability to capture complex dependencies through scaled dot-product attention while maintaining computational efficiency through parallelization.

## Mathematical Foundation

### Core Attention Mechanism

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- **Q ∈ ℝ^{n×d_k}**: Query matrix
- **K ∈ ℝ^{m×d_k}**: Key matrix  
- **V ∈ ℝ^{m×d_v}**: Value matrix
- **n**: Number of queries (sequence length)
- **m**: Number of key-value pairs
- **d_k**: Dimension of keys and queries
- **d_v**: Dimension of values

**Step-by-Step Computation:**

1. **Similarity Scores:**
```
S = QK^T ∈ ℝ^{n×m}
S_{ij} = ∑_{k=1}^{d_k} Q_{ik} K_{jk}
```

2. **Scaled Scores:**
```
S̃ = S / √d_k
```

3. **Attention Weights:**
```
A = softmax(S̃) ∈ ℝ^{n×m}
A_{ij} = exp(S̃_{ij}) / ∑_{k=1}^m exp(S̃_{ik})
```

4. **Output:**
```
O = AV ∈ ℝ^{n×d_v}
O_i = ∑_{j=1}^m A_{ij} V_j
```

### Self-Attention Mathematics

**Self-Attention Definition:**
When Q, K, V are all derived from the same input sequence X:
```
Q = XW_Q, K = XW_K, V = XW_V
```

Where:
- **X ∈ ℝ^{n×d_{model}}**: Input sequence
- **W_Q ∈ ℝ^{d_{model}×d_k}**: Query projection matrix
- **W_K ∈ ℝ^{d_{model}×d_k}**: Key projection matrix
- **W_V ∈ ℝ^{d_{model}×d_v}**: Value projection matrix

**Complete Self-Attention:**
```
SelfAttention(X) = softmax(XW_Q(XW_K)^T / √d_k)(XW_V)
                 = softmax(XW_QW_K^TX^T / √d_k)(XW_V)
```

### Multi-Head Attention

**Parallel Attention Heads:**
```
head_i = Attention(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})
```

**Concatenation and Projection:**
```
MultiHead(X) = Concat(head_1, head_2, ..., head_h)W_O
```

Where:
- **h**: Number of attention heads
- **W_Q^{(i)} ∈ ℝ^{d_{model}×d_k}**: Query weights for head i
- **W_K^{(i)} ∈ ℝ^{d_{model}×d_k}**: Key weights for head i
- **W_V^{(i)} ∈ ℝ^{d_{model}×d_v}**: Value weights for head i
- **W_O ∈ ℝ^{hd_v×d_{model}}**: Output projection matrix

**Typical Dimension Setting:**
```
d_k = d_v = d_{model} / h
```

This ensures the total computational cost remains similar to single-head attention.

**Mathematical Expansion:**
```
MultiHead(X) = [head_1; head_2; ...; head_h]W_O
             = [Attention(XW_Q^{(1)}, XW_K^{(1)}, XW_V^{(1)}); ...;
                Attention(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})]W_O
```

## Positional Encoding

### Sinusoidal Positional Encoding

**Mathematical Definition:**
```
PE(pos, 2i) = sin(pos / 10000^{2i/d_{model}})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_{model}})
```

Where:
- **pos**: Position in the sequence (0, 1, 2, ...)
- **i**: Dimension index (0, 1, 2, ..., d_{model}/2-1)

**Matrix Form:**
```
PE ∈ ℝ^{max_len×d_{model}}
PE_{pos,dim} = {sin(pos/10000^{dim/d_{model}}) if dim is even
               {cos(pos/10000^{(dim-1)/d_{model}}) if dim is odd
```

**Properties:**
1. **Periodicity**: Different frequencies for different dimensions
2. **Relative Position**: PE(pos+k) can be expressed as linear function of PE(pos)
3. **Boundedness**: All values in [-1, 1]

**Wavelength Analysis:**
```
Wavelength(dim) = 2π × 10000^{dim/d_{model}}
```

- Lower dimensions: Shorter wavelengths (high frequency)
- Higher dimensions: Longer wavelengths (low frequency)

### Learned Positional Embeddings

**Alternative Approach:**
```
PE ∈ ℝ^{max_len×d_{model}}  # Learnable parameters
```

**Advantages:**
- Task-specific optimization
- No fixed frequency constraints

**Disadvantages:**
- Limited to training sequence length
- Requires more parameters

## Transformer Layer Architecture

### Encoder Layer

**Mathematical Formulation:**
```
# Multi-Head Self-Attention
Z = MultiHeadAttention(X)
X₁ = LayerNorm(X + Z)  # Residual connection + Layer Norm

# Feed-Forward Network
F = FFN(X₁)
X₂ = LayerNorm(X₁ + F)  # Residual connection + Layer Norm
```

**Complete Encoder Layer:**
```
EncoderLayer(X) = LayerNorm(X₁ + FFN(X₁))
where X₁ = LayerNorm(X + MultiHeadAttention(X))
```

### Decoder Layer

**Mathematical Formulation:**
```
# Masked Multi-Head Self-Attention
Z₁ = MaskedMultiHeadAttention(Y)
Y₁ = LayerNorm(Y + Z₁)

# Cross-Attention (Encoder-Decoder Attention)
Z₂ = MultiHeadAttention(Q=Y₁, K=encoder_output, V=encoder_output)
Y₂ = LayerNorm(Y₁ + Z₂)

# Feed-Forward Network
F = FFN(Y₂)
Y₃ = LayerNorm(Y₂ + F)
```

### Layer Normalization

**Mathematical Definition:**
```
LayerNorm(x) = γ ⊙ (x - μ)/σ + β
```

Where:
```
μ = (1/d) ∑ᵢ xᵢ  # Mean across features
σ = √((1/d) ∑ᵢ (xᵢ - μ)²)  # Standard deviation
γ, β ∈ ℝᵈ  # Learnable scale and shift parameters
```

### Feed-Forward Network

**Mathematical Structure:**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Where:
- **W₁ ∈ ℝ^{d_{model}×d_{ff}}**: First linear transformation
- **W₂ ∈ ℝ^{d_{ff}×d_{model}}**: Second linear transformation
- **d_{ff}**: Feed-forward dimension (typically 4 × d_{model})
- **max(0, ·)**: ReLU activation function

**Alternative Activations:**
```
GELU: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
SwiGLU: FFN(x) = (σ(xW₁ + b₁) ⊙ (xW₃ + b₃))W₂ + b₂
```

## Masking Mechanisms

### Causal (Auto-regressive) Masking

**Lower Triangular Mask:**
```
Mask_{ij} = {0 if i < j (future positions)
           {-∞ if i ≥ j (past and current positions)
```

**Applied to Attention:**
```
S̃ = (QK^T / √d_k) + Mask
A = softmax(S̃)
```

**Mathematical Effect:**
```
A_{ij} = {0 if i < j
        {softmax(S̃_{ij}) if i ≥ j
```

### Padding Mask

**Sequence Length Variations:**
```
Mask_{ij} = {0 if position j is valid
           {-∞ if position j is padding
```

**Batch Processing:**
For batch of sequences with different lengths, mask ensures attention only to valid positions.

### Key-Value Masking

**Cross-Attention Constraints:**
```
Mask^{KV}_{ij} = {0 if key j is valid for query i
                {-∞ if key j should be ignored
```

## Complete Transformer Architecture

### Encoder Stack

**Multi-Layer Encoder:**
```
H₀ = Embed(X) + PE
H₁ = EncoderLayer₁(H₀)
H₂ = EncoderLayer₂(H₁)
⋮
H_N = EncoderLayer_N(H_{N-1})
```

**Final Encoder Output:**
```
encoder_output = H_N ∈ ℝ^{n×d_{model}}
```

### Decoder Stack

**Multi-Layer Decoder:**
```
D₀ = Embed(Y) + PE
D₁ = DecoderLayer₁(D₀, encoder_output)
D₂ = DecoderLayer₂(D₁, encoder_output)
⋮
D_M = DecoderLayer_M(D_{M-1}, encoder_output)
```

**Output Projection:**
```
logits = D_M W_{output} + b_{output}
P(y_t | y_{<t}, x) = softmax(logits_t)
```

## Attention Pattern Analysis

### Attention Weight Properties

**Row-wise Normalization:**
```
∑_{j=1}^m A_{ij} = 1  for all i
```

**Information Aggregation:**
```
O_i = ∑_{j=1}^m A_{ij} V_j  # Weighted average of values
```

**Gradient Flow:**
```
∂L/∂V_j = ∑_{i=1}^n A_{ij} ∂L/∂O_i
∂L/∂K_j ∝ ∑_{i=1}^n A_{ij} (1 - A_{ij}) Q_i
```

### Self-Attention Patterns

**Identity Attention:**
```
A_{ii} ≈ 1, A_{ij} ≈ 0 for i ≠ j
```

**Local Attention:**
```
A_{ij} ≈ exp(-|i-j|/τ) / Z  # Focus on nearby positions
```

**Global Attention:**
```
A_{ij} ≈ 1/m  # Uniform attention across all positions
```

## Computational Complexity

### Attention Complexity

**Time Complexity:**
```
O(n²d_k + nmd_v)
```

For self-attention (n = m):
```
O(n²d_k + n²d_v) = O(n²d_{model})
```

**Space Complexity:**
```
O(n²)  # Attention weight matrix storage
```

### Multi-Head Attention

**Total Complexity:**
```
Time: h × O(n²d_k) = O(n²d_{model})  # Since hd_k = d_{model}
Space: h × O(n²) = O(hn²)
```

### Comparison with RNNs

**RNN Sequential Processing:**
```
Time: O(nd_{model}²)  # Sequential, cannot parallelize
Space: O(nd_{model})
```

**Transformer Parallel Processing:**
```
Time: O(n²d_{model})  # Parallelizable
Space: O(n²)  # Attention matrices
```

**Trade-off:**
- **Short sequences**: Transformer faster due to parallelization
- **Long sequences**: RNN more memory efficient (n² vs n)

## Advanced Attention Variants

### Sparse Attention

**Local Window Attention:**
```
A_{ij} = {softmax(S̃_{ij}) if |i-j| ≤ w
        {0 otherwise
```

**Strided Attention:**
```
A_{ij} = {softmax(S̃_{ij}) if j ∈ {i-w, i-w+s, ..., i+w}
        {0 otherwise
```

**Complexity Reduction:**
```
O(nw) instead of O(n²)  # where w is window size
```

### Linear Attention

**Kernel Trick Approximation:**
```
Attention(Q,K,V) ≈ φ(Q)(φ(K)ᵀV)
```

Where φ is a feature map:
```
φ(x) = [cos(Wx), sin(Wx)]  # Random Fourier features
```

**Complexity:**
```
O(nd_k) instead of O(n²d_k)
```

### Cross-Attention Variants

**Encoder-Decoder Attention:**
```
Q = decoder_hidden_state
K = V = encoder_output
```

**Memory-Augmented Attention:**
```
K = [encoder_output; memory_bank]
V = [encoder_output; memory_values]
```

## Training Dynamics

### Attention Optimization

**Gradient w.r.t. Queries:**
```
∂L/∂Q = (∂L/∂A) K / √d_k
```

**Gradient w.r.t. Keys:**
```
∂L/∂K = (∂L/∂A)ᵀ Q / √d_k
```

**Gradient w.r.t. Values:**
```
∂L/∂V = Aᵀ (∂L/∂O)
```

### Initialization Strategies

**Xavier Initialization for Projections:**
```
W_Q, W_K, W_V ~ N(0, 1/d_{model})
```

**Output Projection:**
```
W_O ~ N(0, 1/(hd_v))
```

### Learning Rate Scheduling

**Warm-up Schedule:**
```
lr(step) = d_{model}^{-0.5} × min(step^{-0.5}, step × warmup_steps^{-1.5})
```

**Rationale:** Gradual increase prevents attention collapse early in training.

## Attention Interpretability

### Attention Weight Analysis

**Head Specialization:**
Different heads learn different types of relationships:
```
Head₁: Syntactic dependencies
Head₂: Semantic relationships  
Head₃: Positional patterns
```

**Attention Entropy:**
```
H(A_i) = -∑_j A_{ij} log A_{ij}
```

- **Low entropy**: Focused attention
- **High entropy**: Distributed attention

### Attention Rollout

**Multi-Layer Attention Flow:**
```
Ã = I + (1/h)∑_{head} A_{head}  # Average attention + identity
Attention_flow = Ã₁ × Ã₂ × ... × Ã_L
```

### Gradient-Based Attribution

**Attention × Gradient:**
```
Attribution_{ij} = A_{ij} × |∂L/∂A_{ij}|
```

Combines attention weights with gradient importance.

## Transformer Variants

### BERT (Encoder-Only)

**Bidirectional Attention:**
```
# No causal masking - full self-attention
A_{ij} = softmax(S̃_{ij})  for all i,j
```

**Masked Language Modeling:**
```
P(x_i | x_{≠i}) = softmax(h_i W_V + b_V)
```

### GPT (Decoder-Only)

**Causal Self-Attention:**
```
# Lower triangular masking
A_{ij} = {softmax(S̃_{ij}) if i ≥ j
        {0 if i < j
```

**Language Modeling:**
```
P(x_t | x_{<t}) = softmax(h_t W_V + b_V)
```

### T5 (Encoder-Decoder)

**Relative Position Encoding:**
```
S̃_{ij} = (Q_i K_j^T + R_{i-j}) / √d_k
```

Where R_{i-j} encodes relative position bias.

## Memory and Efficiency Optimizations

### Gradient Checkpointing

**Activation Recomputation:**
```
# Forward: Store only subset of activations
# Backward: Recompute missing activations
Memory: O(√L) instead of O(L)  # L = number of layers
```

### Memory-Efficient Attention

**Flash Attention Algorithm:**
```
# Compute attention in blocks to fit in SRAM
# Avoid materializing full n×n attention matrix
Memory: O(n) instead of O(n²)
```

### Mixed Precision Training

**FP16 Attention:**
```
Q, K, V: FP16
Attention weights: FP32 (for numerical stability)
Output: FP16
```

## Theoretical Analysis

### Universal Approximation

**Attention as Function Approximation:**
Transformers can approximate any sequence-to-sequence function given sufficient depth and width.

**Representation Power:**
```
# Single attention layer can represent:
- Copy operations: A ≈ I
- Shifting: A ≈ permutation matrix
- Averaging: A ≈ uniform distribution
```

### Expressivity Analysis

**Turing Completeness:**
Transformers with sufficient precision and layer depth are Turing complete.

**Circuit Complexity:**
Self-attention can implement Boolean circuits efficiently:
```
AND, OR, NOT gates ↔ Specific attention patterns
```

### Optimization Landscape

**Attention Loss Surface:**
- Non-convex optimization problem
- Multiple local minima
- Gradient flow depends on initialization and learning rate

**Convergence Properties:**
- Attention weights can converge to meaningful patterns
- Over-parameterization helps optimization
- Residual connections stabilize training

## Applications Beyond NLP

### Computer Vision

**Vision Transformer (ViT):**
```
# Patch embedding
x_patch = Reshape(image, patch_size)
x_embedded = x_patch W_embed + pos_embed

# Standard transformer on patches
output = Transformer(x_embedded)
```

**Cross-Modal Attention:**
```
Q = text_features
K = V = image_features
cross_attention = Attention(Q, K, V)
```

### Time Series Analysis

**Temporal Attention:**
```
# Query: current time step
# Keys/Values: historical time steps
forecast = Attention(current, history, history)
```

### Graph Neural Networks

**Graph Attention Networks:**
```
# Attention over graph neighbors
e_{ij} = LeakyReLU(a^T [W h_i || W h_j])
α_{ij} = softmax_j(e_{ij})
h_i' = σ(∑_{j∈N(i)} α_{ij} W h_j)
```

## Mathematical Summary

The Transformer architecture represents a paradigm shift in sequence modeling through several key mathematical innovations:

1. **Attention Mechanism**: Replaces recurrence with parallel computation of relationships between all sequence positions
2. **Scaled Dot-Product**: Provides efficient similarity computation with scaling for numerical stability
3. **Multi-Head Design**: Enables learning of multiple types of relationships simultaneously
4. **Positional Encoding**: Injects sequence order information without recurrence

**Key Mathematical Insight**: The attention mechanism computes a weighted average of values, where weights are determined by the similarity between queries and keys. This simple operation, when applied in multiple heads and layers, creates a powerful model capable of capturing complex sequential dependencies.

**Theoretical Foundation**: The Transformer's power lies in its ability to directly model relationships between any two positions in a sequence, regardless of their distance. The mathematical structure enables:
- **Parallel Processing**: All positions computed simultaneously
- **Long-Range Dependencies**: Direct connections between distant positions  
- **Flexible Representations**: Multiple attention heads capture different relationship types
- **Scalability**: Architecture scales effectively with increased data and compute

The mathematical elegance of the Transformer has made it the foundation for modern AI systems, from language models to computer vision, demonstrating the power of attention-based architectures across diverse domains. 