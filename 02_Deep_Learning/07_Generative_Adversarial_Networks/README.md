# Generative Adversarial Networks (GANs)

Generative Adversarial Networks, introduced by Ian Goodfellow et al. in 2014, represent a revolutionary approach to generative modeling through adversarial training. GANs consist of two neural networks engaged in a minimax game: a generator network that learns to create realistic data samples, and a discriminator network that learns to distinguish between real and generated data. This adversarial framework creates a powerful training dynamic where both networks improve simultaneously, leading to high-quality sample generation. The mathematical foundation of GANs combines game theory, probability theory, and deep learning optimization to create one of the most influential architectures in modern AI.

## Mathematical Foundation

### Basic GAN Formulation

**Generator Network:**
```
G_θ: z → x
x_fake = G_θ(z), where z ~ p_z(z)
```

**Discriminator Network:**
```
D_φ: x → [0, 1]
D_φ(x) = P(x is real)
```

**Minimax Game:**
```
min_θ max_φ V(D, G) = E_{x~p_data(x)}[log D_φ(x)] + E_{z~p_z(z)}[log(1 - D_φ(G_θ(z)))]
```

Where:
- **θ**: Generator parameters
- **φ**: Discriminator parameters
- **p_data(x)**: Real data distribution
- **p_z(z)**: Prior noise distribution (typically N(0, I))

### Optimal Discriminator Analysis

**Fixed Generator Optimal Discriminator:**
For fixed G, the optimal discriminator is:
```
D*_G(x) = p_data(x) / (p_data(x) + p_g(x))
```

Where p_g(x) is the generator's distribution.

**Derivation:**
```
V(D, G) = ∫_x [p_data(x) log D(x) + p_g(x) log(1 - D(x))] dx
```

Taking derivative w.r.t. D(x) and setting to zero:
```
∂V/∂D(x) = p_data(x)/D(x) - p_g(x)/(1 - D(x)) = 0
```

Solving yields the optimal discriminator formula.

### Global Optimum Analysis

**Value Function with Optimal Discriminator:**
```
C(G) = max_D V(D, G) = E_{x~p_data}[log(p_data(x)/(p_data(x) + p_g(x)))] + 
                       E_{x~p_g}[log(p_g(x)/(p_data(x) + p_g(x)))]
```

**Jensen-Shannon Divergence:**
```
C(G) = -log(4) + 2 × JS(p_data || p_g)
```

Where:
```
JS(p || q) = ½KL(p || (p+q)/2) + ½KL(q || (p+q)/2)
```

**Global Minimum:**
```
C(G) is minimized when p_g = p_data
At optimum: D*(x) = ½ for all x
```

## Training Dynamics

### Alternating Optimization

**Algorithm 1: Basic GAN Training**
```
for epoch in epochs:
    # Train Discriminator
    for k steps:
        Sample {x⁽ⁱ⁾}ᵢ₌₁ᵐ ~ p_data(x)
        Sample {z⁽ⁱ⁾}ᵢ₌₁ᵐ ~ p_z(z)
        L_D = -(1/m)[∑ᵢ log D(x⁽ⁱ⁾) + ∑ᵢ log(1 - D(G(z⁽ⁱ⁾)))]
        φ ← φ + η_D ∇_φ L_D
    
    # Train Generator  
    Sample {z⁽ⁱ⁾}ᵢ₌₁ᵐ ~ p_z(z)
    L_G = -(1/m)∑ᵢ log D(G(z⁽ⁱ⁾))
    θ ← θ - η_G ∇_θ L_G
```

### Gradient Analysis

**Discriminator Gradients:**
```
∇_φ L_D = E_{x~p_data}[∇_φ log D_φ(x)] + E_{z~p_z}[∇_φ log(1 - D_φ(G_θ(z)))]
```

**Generator Gradients:**
```
∇_θ L_G = E_{z~p_z}[∇_θ log D_φ(G_θ(z))]
        = E_{z~p_z}[∇_θ G_θ(z) ⋅ ∇_x log D_φ(x)|_{x=G_θ(z)}]
```

## Common Training Problems

### Vanishing Gradients

**Problem:** When discriminator becomes too strong:
```
D(G(z)) ≈ 0 ⟹ log D(G(z)) ≈ -∞
∇_θ log D(G(z)) ≈ 0
```

**Solution - Alternative Generator Loss:**
```
L_G = -E_{z~p_z}[log D(G(z))]  # Instead of E[log(1 - D(G(z)))]
```

**Mathematical Justification:**
Both losses have the same fixed point, but different gradients:
```
Original: ∇_θ E[log(1 - D(G(z)))] = -E[(∇_θ G(z)) ⋅ (∇_x D(x)|_{x=G(z)})/(1 - D(G(z)))]
Alternative: ∇_θ E[log D(G(z))] = E[(∇_θ G(z)) ⋅ (∇_x D(x)|_{x=G(z)})/D(G(z))]
```

### Mode Collapse

**Mathematical Description:**
Generator maps multiple z values to same x:
```
G(z₁) = G(z₂) = ... = x* for diverse z₁, z₂, ...
```

**Lack of Diversity:**
```
H(p_g) ≪ H(p_data)  # Generated distribution has low entropy
```

### Nash Equilibrium

**Simultaneous Optimization:**
```
∇_θ L_G(θ*, φ*) = 0
∇_φ L_D(θ*, φ*) = 0
```

**Challenges:**
- Non-convex optimization landscape
- No guarantee of convergence
- Oscillatory behavior possible

## GAN Variants and Improvements

### Deep Convolutional GAN (DCGAN)

**Architecture Guidelines:**
```
Generator:
- Use transposed convolutions for upsampling
- BatchNorm in all layers except output
- ReLU activation in all layers except output (Tanh)
- No fully connected layers

Discriminator:
- Use strided convolutions for downsampling  
- BatchNorm in all layers except input
- LeakyReLU activation
- No fully connected layers except output
```

**Mathematical Formulation:**
```
G: z ∈ ℝ¹⁰⁰ → x ∈ ℝ³ˣ⁶⁴ˣ⁶⁴
D: x ∈ ℝ³ˣ⁶⁴ˣ⁶⁴ → [0, 1]
```

### Wasserstein GAN (WGAN)

**Earth Mover's Distance:**
```
W(p_data, p_g) = inf_{γ∈Π(p_data,p_g)} E_{(x,y)~γ}[||x - y||]
```

**Kantorovich-Rubinstein Duality:**
```
W(p_data, p_g) = sup_{||f||_L≤1} E_{x~p_data}[f(x)] - E_{x~p_g}[f(x)]
```

**WGAN Objective:**
```
min_G max_{D∈𝒟} E_{x~p_data}[D(x)] - E_{z~p_z}[D(G(z))]
```

Where 𝒟 is the set of 1-Lipschitz functions.

**Weight Clipping:**
```
w ← clip(w, -c, c)  # After each discriminator update
```

### WGAN-GP (Gradient Penalty)

**Gradient Penalty Term:**
```
λ E_{x̂~p_x̂}[(||∇_{x̂} D(x̂)||₂ - 1)²]
```

**Interpolated Samples:**
```
x̂ = εx + (1-ε)G(z), where ε ~ U[0,1]
```

**Complete WGAN-GP Loss:**
```
L = E_{x~p_data}[D(x)] - E_{z~p_z}[D(G(z))] + λ E_{x̂~p_x̂}[(||∇_{x̂} D(x̂)||₂ - 1)²]
```

### Least Squares GAN (LSGAN)

**Modified Loss Functions:**
```
L_D = ½E_{x~p_data}[(D(x) - 1)²] + ½E_{z~p_z}[D(G(z))²]
L_G = ½E_{z~p_z}[(D(G(z)) - 1)²]
```

**Motivation:** Penalizes samples far from decision boundary.

### Spectral Normalization GAN (SNGAN)

**Spectral Norm:**
```
SN(W) = W / σ(W)
```

Where σ(W) is the largest singular value of W.

**Lipschitz Constraint:**
```
||f||_L ≤ ∏_l σ(W^{(l)})
```

**Power Iteration Method:**
```
u^{(t+1)} = W^T v^{(t)} / ||W^T v^{(t)}||₂
v^{(t+1)} = W u^{(t+1)} / ||W u^{(t+1)}||₂
σ(W) ≈ u^{(T)T} W v^{(T)}
```

## Conditional GANs

### Class-Conditional Generation

**Generator:**
```
x = G(z, y), where y is class label
```

**Discriminator:**
```
D(x, y) → [0, 1]
```

**Loss Functions:**
```
L_D = -E_{(x,y)~p_data}[log D(x,y)] - E_{z~p_z,y~p_y}[log(1 - D(G(z,y),y))]
L_G = -E_{z~p_z,y~p_y}[log D(G(z,y),y)]
```

### Auxiliary Classifier GAN (AC-GAN)

**Discriminator with Classifier:**
```
D(x) → (real/fake score, class probabilities)
```

**Loss Functions:**
```
L_S = E[(log P(S=real|x_real)) + (log P(S=fake|G(z)))]  # Source loss
L_C = E[(log P(C=c|x_real)) + (log P(C=c|G(z)))]        # Class loss
L_D = L_S + L_C
L_G = L_C - L_S
```

### Pix2Pix

**Paired Image Translation:**
```
G: (x, z) → y
D: (x, y) → [0, 1]
```

**L1 Regularization:**
```
L_L1 = E_{x,y,z}[||y - G(x,z)||₁]
```

**Total Loss:**
```
L_G = E_{x,z}[log D(x,G(x,z))] + λ L_L1
```

### CycleGAN

**Cycle Consistency:**
```
F: X → Y, G: Y → X
L_cyc = E_{x~X}[||G(F(x)) - x||₁] + E_{y~Y}[||F(G(y)) - y||₁]
```

**Total Loss:**
```
L = L_GAN(G,D_Y,X,Y) + L_GAN(F,D_X,Y,X) + λ L_cyc
```

## Progressive Training

### Progressive GAN

**Resolution Scaling:**
```
4×4 → 8×8 → 16×16 → 32×32 → 64×64 → ...
```

**Smooth Transition:**
```
output = α × higher_res + (1-α) × upsampled_lower_res
α: 0 → 1 during transition
```

**Mathematical Formulation:**
```
G_k(z) = ToRGB_k(g_k(...g_1(z)...))
G_{k+1}(z) = α × ToRGB_{k+1}(g_{k+1}(g_k(...g_1(z)...))) + 
             (1-α) × Upsample(ToRGB_k(g_k(...g_1(z)...)))
```

### StyleGAN

**Style-Based Generator:**
```
w = MLP(z)                    # Mapping network
y_{l,i} = Affine_l(w)         # Style vector for layer l
x_l = Modulation(x_{l-1}, y_{l,i})  # Feature modulation
```

**Adaptive Instance Normalization:**
```
AdaIN(x_i, y) = y_{s,i} × (x_i - μ(x_i))/σ(x_i) + y_{b,i}
```

Where:
- y_{s,i}: Style scale
- y_{b,i}: Style bias

**Noise Injection:**
```
x_l = x_l + B_l × noise_l
```

## Training Techniques

### Feature Matching

**Modified Generator Loss:**
```
L_G = ||E_{x~p_data}[f(x)] - E_{z~p_z}[f(G(z))]||₂²
```

Where f(x) is intermediate features from discriminator.

### Mini-batch Discrimination

**Batch Statistics:**
```
M_i = f(x_i) ∈ ℝᴬ
o_i = ∑_{j=1}^n exp(-||M_i - M_j||_B)
```

**Augmented Features:**
```
h'_i = [h_i, o_i]  # Concatenate original and batch features
```

### Historical Averaging

**Parameter Regularization:**
```
L_reg = ||θ - (1/t)∑_{i=1}^t θ_i||₂²
```

**Total Loss:**
```
L_total = L_original + λ L_reg
```

### Experience Replay

**Discriminator Training:**
```
# Mix current and historical generated samples
batch = [current_generated, sample_from_history]
```

**Generator History Buffer:**
Maintain buffer of previously generated samples.

## Evaluation Metrics

### Inception Score (IS)

**Mathematical Definition:**
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```

Where:
- p(y|x): Inception network predictions
- p(y): Marginal distribution

**Properties:**
- Higher IS → Better quality and diversity
- Range: [1, num_classes]

### Fréchet Inception Distance (FID)

**Gaussian Assumption:**
```
X_real ~ N(μ_r, Σ_r)
X_generated ~ N(μ_g, Σ_g)
```

**FID Calculation:**
```
FID = ||μ_r - μ_g||₂² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
```

**Lower FID → Better quality**

### Precision and Recall

**Manifold-Based Metrics:**
```
Precision = |{generated samples in real manifold}| / |{generated samples}|
Recall = |{real samples in generated manifold}| / |{real samples}|
```

**k-NN Estimation:**
Use k-nearest neighbors in feature space to estimate manifolds.

## Theoretical Analysis

### Convergence Analysis

**Fixed Point Analysis:**
For simultaneous gradient descent:
```
θ_{t+1} = θ_t - η_G ∇_θ L_G(θ_t, φ_t)
φ_{t+1} = φ_t + η_D ∇_φ L_D(θ_t, φ_t)
```

**Jacobian Matrix:**
```
J = [∂∇_θ L_G/∂θ    ∂∇_θ L_G/∂φ]
    [∂∇_φ L_D/∂θ    ∂∇_φ L_D/∂φ]
```

**Stability Condition:**
Eigenvalues of J must have negative real parts.

### Non-Saturating Game

**Alternative Formulation:**
```
max_θ min_φ V(θ, φ) = E_{z~p_z}[log D_φ(G_θ(z))] - E_{x~p_data}[log D_φ(x)]
```

**Comparison with Original:**
- Same Nash equilibria
- Different gradient dynamics
- Better convergence properties

### Mode Collapse Theory

**Perfect Discriminator Scenario:**
If D is perfect classifier:
```
∇_θ L_G = 0  (everywhere)
```

**Unrolled GANs Solution:**
```
θ_{t+1} = θ_t - η_G ∇_θ L_G(θ_t, φ_t^{(k)})
```

Where φ_t^{(k)} is k-step lookahead discriminator update.

## Information-Theoretic GANs

### InfoGAN

**Mutual Information Maximization:**
```
min_{G,Q} max_D V_I(D,G) = V(D,G) - λ I(c; G(z,c))
```

Where:
- c: Latent code
- I(c; G(z,c)): Mutual information

**Variational Lower Bound:**
```
I(c; G(z,c)) ≥ E_{x~G(z,c)}[E_{c'~p(c|x)}[log Q(c'|x)]] + H(c)
```

### BiGAN/ALI

**Bidirectional Mapping:**
```
Generator: G(z) → x
Encoder: E(x) → z
```

**Joint Discriminator:**
```
D(x, z) → [0, 1]
```

**Adversarial Loss:**
```
L = E_{x~p_data}[log D(x, E(x))] + E_{z~p_z}[log(1 - D(G(z), z))]
```

## Advanced Architectures

### Self-Attention GAN (SAGAN)

**Self-Attention Module:**
```
Attention(x) = softmax(f(x)^T g(x)) h(x)
```

Where:
```
f(x) = W_f x  # Query
g(x) = W_g x  # Key  
h(x) = W_h x  # Value
```

**Generator Integration:**
```
x_{l+1} = γ × Attention(x_l) + x_l
```

### BigGAN

**Class-Conditional Batch Normalization:**
```
BN(x, y) = γ(y) × (x - μ)/σ + β(y)
```

**Orthogonal Regularization:**
```
R_orth = λ ||W^T W - I||_F²
```

**Truncated Normal Sampling:**
```
z ~ TruncatedNormal(0, I, threshold)
```

### Progressive Growing

**Smooth Layer Introduction:**
```
G_α = (1-α) × Upsample(G_{k-1}) + α × G_k
D_α = (1-α) × Downsample(D_{k-1}) + α × D_k
```

**α Schedule:**
```
α(t) = min(1, (t - t_start)/t_transition)
```

## Regularization Techniques

### Spectral Normalization

**Implementation:**
```
W_SN = W / σ_1(W)
```

**Power Iteration:**
```
for i in range(n_iterations):
    v = W^T u / ||W^T u||
    u = W v / ||W v||
σ_1 ≈ u^T W v
```

### Gradient Penalty Variants

**Zero-Centered GP:**
```
L_GP = λ E_{x̂}[(||∇_{x̂} D(x̂)||₂ - 0)²]
```

**LP (Lipschitz Penalty):**
```
L_LP = λ max(0, ||∇_{x̂} D(x̂)||₂ - 1)²
```

## Applications

### Image Generation

**High-Resolution Synthesis:**
```
z ∈ ℝ^d → x ∈ ℝ^{3×1024×1024}
```

**Style Transfer:**
```
content + style → stylized_image
```

### Data Augmentation

**Training Set Expansion:**
```
Original: {x₁, x₂, ..., x_n}
Augmented: {x₁, ..., x_n, G(z₁), ..., G(z_m)}
```

**Domain Adaptation:**
```
Source domain → Target domain
```

### Anomaly Detection

**Reconstruction-Based:**
```
anomaly_score = ||x - G(E(x))||₂²
```

**Discriminator-Based:**
```
anomaly_score = 1 - D(x)
```

## Implementation Considerations

### Numerical Stability

**Label Smoothing:**
```
Real labels: 0.9 instead of 1.0
Fake labels: 0.1 instead of 0.0
```

**Noisy Labels:**
```
P(label_flip) = α
Real → Fake with probability α
Fake → Real with probability α
```

### Hyperparameter Tuning

**Learning Rate Balance:**
```
η_G / η_D ≈ 1:1 to 1:5  # Generator : Discriminator
```

**Batch Size Effects:**
```
Larger batches → Better gradient estimates
Smaller batches → More noise, potential regularization
```

### Monitoring Training

**Loss Patterns:**
```
Healthy: L_D ≈ log(2), L_G ≈ log(2)
Mode collapse: L_G → -∞, L_D → 0
Discriminator wins: L_G → ∞, L_D → 0
```

**Generated Sample Quality:**
Monitor visual quality and diversity throughout training.

## Mathematical Summary

Generative Adversarial Networks represent a game-theoretic approach to generative modeling:

1. **Adversarial Framework**: Two networks compete in a minimax game, driving each other to improve
2. **Nash Equilibrium**: Theoretical optimal point where generator matches data distribution  
3. **Implicit Density Models**: No explicit likelihood function, but powerful sample generation
4. **Rich Theoretical Foundation**: Connections to optimal transport, information theory, and game theory

**Key Mathematical Insight**: The adversarial loss function transforms generative modeling from a density estimation problem into a discrimination problem. The minimax objective creates a training dynamic where the generator learns to produce samples indistinguishable from real data.

**Theoretical Foundation**: GANs can be viewed as:
- **Density Ratio Estimation**: Discriminator estimates density ratios  
- **Optimal Transport**: Wasserstein GANs minimize earth mover's distance
- **Variational Divergence Minimization**: Different GAN variants minimize different f-divergences
- **Game Theory**: Two-player zero-sum game with provable optimal strategies

The mathematical elegance of GANs lies in their simplicity - a single adversarial objective leads to complex, realistic data generation, making them one of the most influential architectures in modern generative modeling. 