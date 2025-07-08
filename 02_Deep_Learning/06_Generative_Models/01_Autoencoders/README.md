# Autoencoders

Autoencoders represent a fundamental unsupervised learning paradigm in deep learning, designed to learn efficient data representations by training networks to reconstruct their inputs through a compressed latent space. The mathematical elegance of autoencoders lies in their encoder-decoder architecture, where the encoder maps high-dimensional inputs to lower-dimensional latent representations, and the decoder reconstructs the original data from these compressed codes. This framework enables dimensionality reduction, feature learning, denoising, and generative modeling, making autoencoders versatile tools for understanding and manipulating data structure.

## Mathematical Foundation

### Basic Autoencoder Architecture

**Encoder Function:**
```
z = f_θ(x) = σ(W_e x + b_e)
```

**Decoder Function:**
```
x̂ = g_φ(z) = σ(W_d z + b_d)
```

**Complete Autoencoder:**
```
x̂ = g_φ(f_θ(x)) = (g_φ ∘ f_θ)(x)
```

Where:
- **x ∈ ℝⁿ**: Input data
- **z ∈ ℝᵈ**: Latent representation (typically d < n)
- **x̂ ∈ ℝⁿ**: Reconstructed output
- **θ, φ**: Encoder and decoder parameters
- **σ**: Activation function

### Multi-Layer Autoencoder

**Deep Encoder:**
```
h₁ = σ₁(W₁x + b₁)
h₂ = σ₂(W₂h₁ + b₂)
⋮
z = σₗ(Wₗh_{L-1} + bₗ)
```

**Deep Decoder:**
```
h'₁ = σ'₁(W'₁z + b'₁)
h'₂ = σ'₂(W'₂h'₁ + b'₂)
⋮
x̂ = σ'ₘ(W'ₘh'_{M-1} + b'ₘ)
```

**Symmetric Architecture:**
Often L = M and decoder weights are transposes: W'ᵢ = W^T_{L+1-i}

## Loss Functions

### Reconstruction Loss

**Mean Squared Error (Continuous Data):**
```
L_reconstruction = ||x - x̂||² = ∑ᵢ₌₁ⁿ (xᵢ - x̂ᵢ)²
```

**Binary Cross-Entropy (Binary Data):**
```
L_reconstruction = -∑ᵢ₌₁ⁿ [xᵢ log(x̂ᵢ) + (1-xᵢ) log(1-x̂ᵢ)]
```

**Cross-Entropy (Categorical Data):**
```
L_reconstruction = -∑ᵢ₌₁ⁿ ∑ⱼ₌₁ᶜ xᵢⱼ log(x̂ᵢⱼ)
```

### Total Loss Function

**Basic Autoencoder:**
```
L_total = L_reconstruction
```

**Regularized Autoencoder:**
```
L_total = L_reconstruction + λ R(z)
```

Where R(z) is a regularization term.

## Autoencoder Variants

### Undercomplete Autoencoders

**Constraint:**
```
d < n  (latent dimension < input dimension)
```

**Capacity Control:**
The dimensionality bottleneck forces the model to learn meaningful representations.

**Information Bottleneck:**
```
I(X; Z) ≤ H(Z) ≤ d log(2)  (for binary latent codes)
```

### Overcomplete Autoencoders

**Constraint:**
```
d ≥ n  (latent dimension ≥ input dimension)
```

**Regularization Necessity:**
Without regularization, the model can learn the identity function:
```
f_θ(x) = x, g_φ(z) = z
```

**Sparsity Regularization:**
```
R(z) = λ ∑ᵢ₌₁ᵈ |zᵢ|  (L1 regularization)
```

### Sparse Autoencoders

**Sparsity Constraint:**
```
ρ̂ⱼ = (1/m) ∑ᵢ₌₁ᵐ aⱼ⁽ⁱ⁾  # Average activation of unit j
```

**KL Divergence Penalty:**
```
R_sparse = β ∑ⱼ₌₁ᵈ KL(ρ || ρ̂ⱼ)
         = β ∑ⱼ₌₁ᵈ [ρ log(ρ/ρ̂ⱼ) + (1-ρ) log((1-ρ)/(1-ρ̂ⱼ))]
```

Where:
- **ρ**: Target sparsity (typically 0.05)
- **β**: Sparsity penalty weight

**L1 Sparsity:**
```
R_sparse = λ ||z||₁ = λ ∑ᵢ₌₁ᵈ |zᵢ|
```

### Denoising Autoencoders (DAE)

**Corruption Process:**
```
x̃ = C(x)  # Corrupted input
```

**Training Objective:**
```
L = ||x - g_φ(f_θ(x̃))||²
```

**Common Corruption Types:**

**Gaussian Noise:**
```
x̃ = x + ε, where ε ~ N(0, σ²I)
```

**Masking Noise:**
```
x̃ᵢ = {xᵢ with probability 1-p
      {0 with probability p
```

**Salt-and-Pepper Noise:**
```
x̃ᵢ = {0 with probability p/2
      {1 with probability p/2  
      {xᵢ with probability 1-p
```

### Contractive Autoencoders (CAE)

**Contractive Penalty:**
```
R_contractive = λ ||J_f(x)||²_F
```

Where J_f(x) is the Jacobian of the encoder:
```
J_f(x) = ∂f(x)/∂x ∈ ℝᵈˣⁿ
```

**Frobenius Norm:**
```
||J_f(x)||²_F = ∑ᵢ₌₁ᵈ ∑ⱼ₌₁ⁿ (∂fᵢ(x)/∂xⱼ)²
```

**Total Loss:**
```
L_CAE = ||x - x̂||² + λ ||∂f(x)/∂x||²_F
```

**Robustness Property:**
Small changes in input produce smaller changes in latent representation.

## Deep Autoencoders

### Stacked Architecture

**Layer-wise Pre-training:**
1. Train first encoder-decoder pair
2. Use encoder output as input for next layer
3. Repeat until desired depth
4. Fine-tune entire network end-to-end

**Mathematical Formulation:**
```
# Layer 1
z₁ = f₁(x), x̂₁ = g₁(z₁)
Minimize: ||x - x̂₁||²

# Layer 2  
z₂ = f₂(z₁), ẑ₁ = g₂(z₂)
Minimize: ||z₁ - ẑ₁||²

# Combined
z = f_L(f_{L-1}(...f₁(x)...))
x̂ = g₁(g₂(...g_L(z)...))
```

### Skip Connections

**Residual Autoencoder:**
```
z = f_θ(x)
x̂ = g_φ(z) + W_skip x  # Skip connection
```

**U-Net Style:**
```
# Encoder with skip connections stored
h₁, skip₁ = encoder_block₁(x)
h₂, skip₂ = encoder_block₂(h₁)
z = bottleneck(h₂)

# Decoder with skip connections
h'₂ = decoder_block₂(z, skip₂)  
x̂ = decoder_block₁(h'₂, skip₁)
```

## Regularization Techniques

### Weight Decay

**L2 Regularization:**
```
R_weight = λ (||W_e||²_F + ||W_d||²_F)
```

**Total Loss:**
```
L_total = L_reconstruction + λ (||W_e||²_F + ||W_d||²_F)
```

### Dropout

**Training Phase:**
```
z_dropped = dropout(z, p)
x̂ = g_φ(z_dropped)
```

**Inference Phase:**
```
x̂ = g_φ(z)  # No dropout
```

### Batch Normalization

**Encoder with BatchNorm:**
```
h = σ(BN(Wx + b))
```

**Decoder with BatchNorm:**
```
x̂ = σ(BN(W'z + b'))
```

## Variational Autoencoders (VAE)

### Probabilistic Framework

**Encoder (Recognition Model):**
```
q_φ(z|x) = N(z; μ_φ(x), σ²_φ(x)I)
```

**Decoder (Generative Model):**
```
p_θ(x|z) = N(x; μ_θ(z), σ²_θ(z)I)  # or Bernoulli for binary data
```

**Prior:**
```
p(z) = N(z; 0, I)
```

### ELBO Derivation

**Evidence Lower Bound:**
```
log p(x) ≥ E_{q_φ(z|x)}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
         = L_reconstruction - L_KL
```

**KL Divergence (Closed Form for Gaussian):**
```
KL(q_φ(z|x) || p(z)) = ½ ∑ᵢ₌₁ᵈ [μᵢ² + σᵢ² - log(σᵢ²) - 1]
```

**Reparameterization Trick:**
```
z = μ_φ(x) + σ_φ(x) ⊙ ε, where ε ~ N(0, I)
```

This enables backpropagation through the stochastic node.

### β-VAE

**Modified Loss:**
```
L_β-VAE = L_reconstruction + β · KL(q_φ(z|x) || p(z))
```

**Disentanglement:**
- **β > 1**: Encourages disentangled representations
- **β < 1**: Emphasizes reconstruction quality

## Sequence Autoencoders

### Recurrent Autoencoders

**LSTM Encoder:**
```
h_t^{enc} = LSTM_{enc}(x_t, h_{t-1}^{enc})
z = h_T^{enc}  # Final hidden state as encoding
```

**LSTM Decoder:**
```
h_0^{dec} = z
h_t^{dec} = LSTM_{dec}(x_{t-1}^{dec}, h_{t-1}^{dec})
x̂_t = softmax(W_{out} h_t^{dec} + b_{out})
```

### Sequence-to-Sequence Autoencoders

**Variable Length Sequences:**
```
Encoder: x₁, x₂, ..., x_T → z
Decoder: z → x̂₁, x̂₂, ..., x̂_T
```

**Attention Mechanism:**
```
c_t = ∑_{s=1}^T α_{ts} h_s^{enc}
h_t^{dec} = LSTM_{dec}([x_{t-1}; c_t], h_{t-1}^{dec})
```

## Convolutional Autoencoders

### 2D Convolutional Architecture

**Encoder:**
```
h₁ = σ(conv₁(x))
h₂ = σ(conv₂(h₁))
z = flatten(h₂)
```

**Decoder:**
```
h'₁ = reshape(z)
h'₂ = σ(deconv₂(h'₁))
x̂ = σ(deconv₁(h'₂))
```

**Transposed Convolutions:**
```
deconv(z) = conv_transpose(z)
```

Upsamples spatial dimensions while reducing channels.

### U-Net Autoencoder

**Skip Connections:**
```
# Encoder path
conv1, skip1 = encoder_block(x)
conv2, skip2 = encoder_block(conv1)
bottleneck = encoder_block(conv2)

# Decoder path
upconv2 = decoder_block(bottleneck, skip2)
upconv1 = decoder_block(upconv2, skip1)
```

## Training Strategies

### Learning Rate Scheduling

**Cosine Annealing:**
```
η_t = η_min + (η_max - η_min) × (1 + cos(πt/T))/2
```

**Exponential Decay:**
```
η_t = η_0 × γ^{⌊t/step_size⌋}
```

### Early Stopping

**Validation Loss Monitoring:**
```
if val_loss(epoch) > val_loss(epoch - patience):
    stop_training()
```

### Gradient Clipping

**Global Norm Clipping:**
```
if ||∇||₂ > threshold:
    ∇ ← ∇ × threshold/||∇||₂
```

## Evaluation Metrics

### Reconstruction Quality

**Peak Signal-to-Noise Ratio (PSNR):**
```
PSNR = 10 log₁₀(MAX²/MSE)
```

**Structural Similarity Index (SSIM):**
```
SSIM(x, x̂) = (2μₓμₓ̂ + c₁)(2σₓₓ̂ + c₂) / ((μₓ² + μₓ̂² + c₁)(σₓ² + σₓ̂² + c₂))
```

### Latent Space Quality

**Latent Space Interpolation:**
```
z_interp = (1-α)z₁ + αz₂, α ∈ [0,1]
x_interp = g_φ(z_interp)
```

**Latent Space Arithmetic:**
```
z_result = z_A - z_B + z_C
```

### Disentanglement Metrics

**β-VAE Metric:**
```
Score = E_z[Var_i[E_x[classifier(factor_i | encoder(x))]]
```

**MIG (Mutual Information Gap):**
```
MIG = (1/K) ∑ₖ (I(z_j; v_k)⁽¹⁾ - I(z_j; v_k)⁽²⁾) / H(v_k)
```

## Applications

### Dimensionality Reduction

**Principal Component Analysis Comparison:**
```
PCA: Linear projection
Autoencoder: Nonlinear projection
```

**Manifold Learning:**
```
z = f_θ(x)  # Learn nonlinear manifold embedding
```

### Anomaly Detection

**Reconstruction Error Threshold:**
```
anomaly_score = ||x - x̂||²
threshold = percentile(anomaly_scores, 95)
```

**One-Class Classification:**
```
normal_data → low reconstruction error
anomalous_data → high reconstruction error
```

### Data Compression

**Compression Ratio:**
```
ratio = (n × bits_per_pixel) / (d × bits_per_latent)
```

**Rate-Distortion Trade-off:**
```
R(D) = min_{encoder,decoder} I(X; Z) subject to E[d(X, X̂)] ≤ D
```

### Image Denoising

**Training:**
```
clean_image → add_noise → corrupted_image
L = ||clean_image - autoencoder(corrupted_image)||²
```

**Inference:**
```
denoised_image = autoencoder(noisy_image)
```

## Advanced Architectures

### Adversarial Autoencoders (AAE)

**Adversarial Training:**
```
L_reconstruction = ||x - x̂||²
L_adversarial = E[log D(z)] + E[log(1 - D(q(z|x)))]
L_total = L_reconstruction + λ L_adversarial
```

**Generator (Encoder):** q_φ(z|x)
**Discriminator:** Distinguishes between p(z) and q_φ(z|x)

### Vector Quantized VAE (VQ-VAE)

**Discrete Latent Space:**
```
z_q = argmin_k ||z_e - e_k||₂
```

Where:
- **z_e**: Encoder output
- **e_k**: Codebook vectors
- **z_q**: Quantized latent

**Training Loss:**
```
L = ||x - x̂||² + ||sg[z_e] - e||² + β||z_e - sg[e]||²
```

Where sg[·] denotes stop gradient.

## Implementation Considerations

### Numerical Stability

**Activation Function Choice:**
```
# Avoid saturation
encoder: ReLU or ELU
decoder: sigmoid (for [0,1] outputs) or tanh (for [-1,1])
```

**Weight Initialization:**
```
Xavier: W ~ N(0, 2/(n_in + n_out))
He: W ~ N(0, 2/n_in)  # For ReLU networks
```

### Memory Optimization

**Gradient Checkpointing:**
```
# Store only subset of activations
# Recompute others during backward pass
```

**Mixed Precision Training:**
```
# Use FP16 for forward pass
# Use FP32 for loss computation and gradients
```

## Theoretical Analysis

### Information Theory Perspective

**Information Bottleneck:**
```
min I(X; Z) subject to I(Y; Z) ≥ I_min
```

For autoencoders: Y = X (reconstruction task)

**Rate-Distortion Theory:**
```
R(D) = min_{p(z|x): E[d(X,X̂)]≤D} I(X; Z)
```

### Manifold Learning

**Manifold Assumption:**
High-dimensional data lies on lower-dimensional manifold.

**Autoencoder as Manifold Learning:**
```
Encoder: x → z (chart map)
Decoder: z → x̂ (inverse chart map)
```

### Universal Approximation

**Theorem:** Autoencoders with sufficient capacity can approximate any continuous function mapping inputs to reconstructions.

**Limitations:** Approximation quality depends on:
- Network width and depth
- Activation function choice
- Training data distribution

## Mathematical Summary

Autoencoders represent a fundamental approach to unsupervised representation learning:

1. **Encoder-Decoder Framework**: Compression and reconstruction through learned transformations
2. **Information Bottleneck**: Dimensional reduction forces meaningful feature extraction
3. **Regularization**: Various techniques prevent trivial solutions and encourage useful representations
4. **Generative Capability**: Learned latent spaces enable data generation and manipulation

**Key Mathematical Insight**: The reconstruction objective L = ||x - g_φ(f_θ(x))||² creates an optimization problem where the network must discover compact, informative representations. The bottleneck constraint z ∈ ℝᵈ with d < n forces the model to compress information, naturally leading to feature learning.

**Theoretical Foundation**: Autoencoders can be viewed as approximating the identity function under constraints. The mathematical structure enables:
- **Dimensionality Reduction**: Learning nonlinear manifold embeddings
- **Feature Learning**: Discovering meaningful data representations  
- **Generative Modeling**: Sampling from learned latent distributions
- **Regularization**: Controlling representation complexity through various mathematical constraints

The mathematical elegance of autoencoders lies in their simplicity - optimizing reconstruction error leads naturally to meaningful representation learning, making them fundamental building blocks for many advanced generative models and representation learning techniques.

This section includes:
- [**Variational Autoencoders (VAEs)**](./01_Variational_Autoencoders/): A probabilistic take on autoencoders for generative purposes. 