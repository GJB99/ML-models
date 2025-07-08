# Diffusion Models

Diffusion models represent a groundbreaking approach to generative modeling that has revolutionized image synthesis, achieving unprecedented quality in generated samples. Inspired by non-equilibrium thermodynamics, diffusion models learn to reverse a gradual noising process, progressively denoising random noise into coherent data samples. The mathematical foundation combines stochastic differential equations, variational inference, and score-based modeling to create a powerful generative framework. Unlike GANs which learn through adversarial training, diffusion models optimize a tractable likelihood-based objective, providing stable training and high-quality generation.

## Mathematical Foundation

### Forward Diffusion Process

**Markov Chain Definition:**
```
q(x₁:T | x₀) = ∏_{t=1}^T q(x_t | x_{t-1})
```

**Gaussian Transition:**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Where:
- **x₀**: Original data sample
- **x_t**: Noisy sample at timestep t
- **β_t**: Noise schedule (variance added at step t)
- **T**: Total diffusion steps

**Recursive Formulation:**
```
x_t = √(1-β_t) x_{t-1} + √β_t ε_{t-1}
```

Where ε_{t-1} ~ N(0, I)

### Closed-Form Forward Process

**Direct Sampling at Any Timestep:**
```
q(x_t | x₀) = N(x_t; √ᾱ_t x₀, (1-ᾱ_t) I)
```

**Reparameterization:**
```
x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
```

Where:
```
α_t = 1 - β_t
ᾱ_t = ∏_{s=1}^t α_s
ε ~ N(0, I)
```

**Noise Schedule Properties:**
```
ᾱ₀ = 1 (no noise)
ᾱ_T ≈ 0 (pure noise)
```

### Reverse Diffusion Process

**Reverse Markov Chain:**
```
p_θ(x₀:T) = p(x_T) ∏_{t=1}^T p_θ(x_{t-1} | x_t)
```

**Parameterized Reverse Step:**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Simplified Variance (Common Choice):**
```
Σ_θ(x_t, t) = σ_t² I
```

Where σ_t² can be fixed or learned.

## Variational Lower Bound

### ELBO Derivation

**Evidence Lower Bound:**
```
log p_θ(x₀) ≥ E_q[-log q(x₁:T|x₀) + log p_θ(x₀:T)]
```

**Decomposed ELBO:**
```
L = E_q[log p_θ(x₀|x₁)] - KL(q(x_T|x₀) || p(x_T)) - ∑_{t=2}^T KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t))
```

**Term Analysis:**
```
L₀ = E_q[log p_θ(x₀|x₁)]                    # Reconstruction term
L_T = KL(q(x_T|x₀) || p(x_T))               # Prior matching term  
L_{t-1} = KL(q(x_{t-1}|x_t,x₀) || p_θ(x_{t-1}|x_t))  # Denoising terms
```

### Posterior Distribution

**True Reverse Step (Tractable):**
```
q(x_{t-1} | x_t, x₀) = N(x_{t-1}; μ̃_t(x_t, x₀), β̃_t I)
```

**Posterior Mean:**
```
μ̃_t(x_t, x₀) = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x₀ + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
```

**Posterior Variance:**
```
β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t
```

## Loss Functions

### Simplified Training Objective

**L₂ Loss on Noise:**
```
L_simple = E_{t,x₀,ε}[||ε - ε_θ(x_t, t)||²]
```

Where:
- **ε**: True noise added at step t
- **ε_θ(x_t, t)**: Predicted noise by neural network
- **x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε**

**Connection to ELBO:**
The simplified loss is a weighted version of the ELBO terms.

### Variational Loss (Full ELBO)

**Weighted ELBO:**
```
L_vlb = L₀ + L_T + ∑_{t=2}^T w_t L_{t-1}
```

**Weight Schedule:**
```
w_t = β_t²/(2σ_t²α_t(1-ᾱ_t))
```

### Alternative Parameterizations

**Predict x₀ Directly:**
```
L_x₀ = E_{t,x₀,ε}[||x₀ - x₀_θ(x_t, t)||²]
```

**Predict Velocity (v-parameterization):**
```
v_t = √ᾱ_t ε - √(1-ᾱ_t) x₀
L_v = E_{t,x₀,ε}[||v_t - v_θ(x_t, t)||²]
```

## Sampling Algorithms

### DDPM Sampling

**Ancestral Sampling:**
```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    if t > 1:
        z ~ N(0, I)
    else:
        z = 0
    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + σ_t z
```

**Mean Prediction:**
```
μ_θ(x_t, t) = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t))
```

### DDIM Sampling

**Deterministic Sampling:**
```
x_{t-1} = √ᾱ_{t-1} (x_t - √(1-ᾱ_t) ε_θ(x_t, t))/√ᾱ_t + √(1-ᾱ_{t-1}) ε_θ(x_t, t)
```

**Generalized Formula:**
```
x_{t-1} = √ᾱ_{t-1} x₀_pred + √(1-ᾱ_{t-1} - σ_t²) ε_θ(x_t, t) + σ_t ε
```

Where:
```
x₀_pred = (x_t - √(1-ᾱ_t) ε_θ(x_t, t))/√ᾱ_t
```

**Deterministic (σ_t = 0) vs Stochastic (σ_t > 0)**

### Accelerated Sampling

**Skip Steps:**
```
τ = {T, T-k, T-2k, ..., k, 0}  # Subsequence
```

**Update Rule:**
```
x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} x₀_pred + √(1-ᾱ_{τ_{i-1}}) ε_θ(x_{τ_i}, τ_i)
```

## Noise Schedules

### Linear Schedule

**Beta Schedule:**
```
β_t = β_start + (β_end - β_start) × t/T
```

**Typical Values:**
```
β_start = 0.0001
β_end = 0.02
T = 1000
```

### Cosine Schedule

**Improved Schedule:**
```
ᾱ_t = cos²(π/2 × (t/T + s)/(1 + s))
```

Where s is a small offset (e.g., 0.008).

**Beta Computation:**
```
β_t = 1 - ᾱ_t/ᾱ_{t-1}
```

### Learned Schedules

**Adaptive Noise:**
```
β_t = MLP(t, x_t)  # Context-dependent noise
```

## Classifier Guidance

### Conditional Generation

**Classifier-Guided Sampling:**
```
ε̃_θ(x_t, t, y) = ε_θ(x_t, t) - s√(1-ᾱ_t) ∇_{x_t} log p_φ(y|x_t)
```

Where:
- **y**: Class label
- **p_φ(y|x_t)**: Pretrained classifier
- **s**: Guidance scale

**Modified Score:**
```
s_θ(x_t, t, y) = s_θ(x_t, t) + s ∇_{x_t} log p_φ(y|x_t)
```

### Classifier-Free Guidance

**Joint Training:**
```
ε_θ(x_t, t, y) with probability p
ε_θ(x_t, t, ∅) with probability 1-p  # Unconditional
```

**Guided Prediction:**
```
ε̃_θ(x_t, t, y) = ε_θ(x_t, t, ∅) + s(ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

**Implicit Classifier:**
```
ε̃_θ(x_t, t, y) = (1 + s)ε_θ(x_t, t, y) - s ε_θ(x_t, t, ∅)
```

## Score-Based Perspective

### Score Function

**Definition:**
```
s_θ(x_t, t) = ∇_{x_t} log q(x_t)
```

**Connection to Noise Prediction:**
```
ε_θ(x_t, t) = -√(1-ᾱ_t) s_θ(x_t, t)
```

### Score Matching Loss

**Denoising Score Matching:**
```
L_DSM = E_{t,x₀,ε}[||s_θ(x_t, t) - (-ε/√(1-ᾱ_t))||²]
```

**Equivalent to Noise Prediction:**
```
L_DSM ∝ L_simple
```

### Langevin Dynamics

**Score-Based Sampling:**
```
x_{t-1} = x_t + δ s_θ(x_t, t) + √(2δ) z
```

Where δ is step size and z ~ N(0, I).

## Continuous-Time Formulation

### Stochastic Differential Equation

**Forward SDE:**
```
dx = f(x,t)dt + g(t)dw
```

**Reverse SDE:**
```
dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄
```

Where w̄ is reverse-time Brownian motion.

### Variance Preserving (VP) SDE

**Forward Process:**
```
dx = -½β(t)x dt + √β(t) dw
```

**Solution:**
```
x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
```

Where ᾱ_t = exp(-½∫₀ᵗ β(s)ds)

### Variance Exploding (VE) SDE

**Forward Process:**
```
dx = √(dσ²(t)/dt) dw
```

**Solution:**
```
x_t = x₀ + σ(t) ε
```

## Neural Network Architectures

### U-Net Architecture

**Encoder-Decoder with Skip Connections:**
```
# Encoder
h₁ = DownBlock₁(x, t_emb)
h₂ = DownBlock₂(h₁, t_emb)
...
h_n = DownBlock_n(h_{n-1}, t_emb)

# Decoder  
h'₁ = UpBlock₁(h_n, h_{n-1}, t_emb)
h'₂ = UpBlock₂(h'₁, h_{n-2}, t_emb)
...
output = OutputLayer(h'_n)
```

**Time Embedding:**
```
t_emb = MLP(SinusoidalEmbedding(t))
```

### Sinusoidal Time Embedding

**Position Encoding for Time:**
```
PE(t, 2i) = sin(t / 10000^{2i/d})
PE(t, 2i+1) = cos(t / 10000^{2i/d})
```

**Learned Projection:**
```
t_emb = MLP([PE(t); PE(t)])  # Double dimension
```

### Attention Mechanisms

**Self-Attention in U-Net:**
```
# At resolution 16×16 and 8×8
attn_out = SelfAttention(conv_features)
```

**Cross-Attention for Conditioning:**
```
# Text conditioning
cross_attn = CrossAttention(visual_features, text_embeddings)
```

## Advanced Techniques

### Improved Training

**v-Parameterization:**
```
v_t = √ᾱ_t ε - √(1-ᾱ_t) x₀
```

**Benefits:**
- Better training dynamics
- Reduced variance in loss
- Improved sample quality

**Progressive Distillation:**
```
# Distill T-step model into T/2-step model
L_distill = E[||ε_teacher(x_t, t) - ε_student(x_{2t}, 2t)||²]
```

### Conditioning Strategies

**Cross-Attention Conditioning:**
```
Q = linear_q(visual_features)
K = linear_k(condition_embeddings)  
V = linear_v(condition_embeddings)
```

**FiLM (Feature-wise Linear Modulation):**
```
h' = γ(condition) ⊙ h + β(condition)
```

**AdaIN (Adaptive Instance Normalization):**
```
h' = σ(condition) ⊙ ((h - μ(h))/σ(h)) + μ(condition)
```

## Latent Diffusion Models

### VAE Integration

**Latent Space Training:**
```
z = Encoder(x₀)  # Encode to latent space
# Train diffusion model on z instead of x₀
L = E[||ε - ε_θ(z_t, t)||²]
```

**Two-Stage Training:**
```
1. Train VAE: x ↔ z
2. Train diffusion in latent space: z_noise → z_clean
```

**Sampling Process:**
```
z_T ~ N(0, I)
z₀ = DDPM_sampling(z_T)
x₀ = Decoder(z₀)
```

### Advantages of Latent Space

**Computational Efficiency:**
```
Image resolution: 512×512×3 → Latent: 64×64×4
Speedup: ~48× reduction in pixels
```

**Memory Savings:**
Quadratic reduction in memory requirements.

## Evaluation Metrics

### Fréchet Inception Distance (FID)

**Feature-Based Distance:**
```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real Σ_gen))
```

**Lower is Better:** FID ∈ [0, ∞)

### Inception Score (IS)

**Quality and Diversity:**
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```

**Higher is Better:** IS ∈ [1, num_classes]

### CLIP Score

**Text-Image Alignment:**
```
CLIP_score = E[cos_sim(CLIP_text(caption), CLIP_image(generated))]
```

### Human Evaluation

**Preference Studies:**
Human raters compare generated vs. real images.

## Applications

### Image Generation

**Unconditional Generation:**
```
x_T ~ N(0, I) → x₀
```

**Class-Conditional:**
```
x_T ~ N(0, I) + class_label → x₀
```

**Text-to-Image:**
```
x_T ~ N(0, I) + text_embedding → x₀
```

### Image Editing

**Inpainting:**
```
# Combine known and generated pixels at each step
x_t^{known} = √ᾱ_t x₀^{known} + √(1-ᾱ_t) ε
x_t = mask ⊙ x_t^{known} + (1-mask) ⊙ x_t^{generated}
```

**Outpainting:**
```
# Extend image boundaries
x_t = [original_region; diffusion_region]
```

### Super-Resolution

**Conditional Upsampling:**
```
x_high_res = Diffusion(x_low_res, noise)
```

**Progressive Refinement:**
```
64×64 → 128×128 → 256×256 → 512×512
```

## Implementation Details

### Training Procedure

**Algorithm: DDPM Training**
```
repeat:
    x₀ ~ q(x₀)                    # Sample data
    t ~ Uniform({1, ..., T})      # Sample timestep
    ε ~ N(0, I)                   # Sample noise
    x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε   # Add noise
    L = ||ε - ε_θ(x_t, t)||²      # Compute loss
    Update θ using ∇_θ L           # Gradient step
```

### Numerical Stability

**Clipping Gradients:**
```
if ||∇_θ|| > max_grad_norm:
    ∇_θ = ∇_θ × max_grad_norm/||∇_θ||
```

**EMA (Exponential Moving Average):**
```
θ_ema = β θ_ema + (1-β) θ
```

Use θ_ema for inference, θ for training.

### Memory Optimization

**Gradient Checkpointing:**
```
# Recompute activations during backward pass
# Trade computation for memory
```

**Mixed Precision:**
```
# Use FP16 for forward pass
# Use FP32 for loss and gradients
```

## Theoretical Analysis

### Convergence Properties

**Score Estimation Error:**
```
E[||s_θ(x_t, t) - ∇_x log q(x_t)||²] → 0  as dataset_size → ∞
```

**Sampling Error:**
```
W₂(p_sample, p_data) ≤ C × √(score_error + discretization_error)
```

### Approximation Theory

**Universal Approximation:**
Deep networks can approximate score functions arbitrarily well.

**Sample Complexity:**
Number of samples needed for ε-accurate score estimation:
```
N ≥ O(d/ε²)
```

Where d is data dimension.

### Connection to Other Models

**VAE Connection:**
```
ELBO_VAE ≈ ELBO_Diffusion  (with T → ∞)
```

**Flow-Based Models:**
```
Diffusion = Continuous normalizing flow with Gaussian base
```

**Energy-Based Models:**
```
Score function = Gradient of energy function
```

## Recent Advances

### Consistency Models

**Direct Sampling:**
```
x₀ = f_θ(x_t, t)  # Single-step generation
```

**Consistency Training:**
```
L = E[||f_θ(x_t, t) - f_θ(x_{t+1}, t+1)||²]
```

### Rectified Flow

**Straight Paths:**
```
dx_t = (x₁ - x₀) dt
```

**Optimal Transport:**
Minimize transport cost between distributions.

### Video Diffusion

**Temporal Consistency:**
```
3D U-Net: (T, H, W, C) → (T, H, W, C)
```

**Frame Conditioning:**
```
ε_θ(x_t, t, previous_frames)
```

## Mathematical Summary

Diffusion models represent a principled approach to generative modeling through several key mathematical innovations:

1. **Forward-Reverse Process**: Systematic addition and removal of noise through Markov chains
2. **Variational Framework**: Tractable training objective via ELBO optimization
3. **Score-Based Perspective**: Connection to score matching and Langevin dynamics
4. **Continuous Formulation**: SDE framework unifying discrete and continuous processes

**Key Mathematical Insight**: The forward diffusion process transforms any data distribution into a simple Gaussian distribution through a sequence of small noise additions. The reverse process, parameterized by neural networks, learns to undo this transformation, enabling high-quality sample generation.

**Theoretical Foundation**: Diffusion models can be viewed as:
- **Hierarchical VAEs**: With specific encoder structure (Gaussian noise)
- **Score-Based Models**: Learning gradients of log-density
- **Optimal Transport**: Moving between noise and data distributions
- **Langevin MCMC**: Sampling via gradient-based dynamics

The mathematical elegance of diffusion models lies in their principled training objective and stable optimization, contrasting with the adversarial training challenges of GANs while achieving superior sample quality. The theoretical guarantees and practical performance have made diffusion models the current state-of-the-art in generative modeling across multiple domains. 