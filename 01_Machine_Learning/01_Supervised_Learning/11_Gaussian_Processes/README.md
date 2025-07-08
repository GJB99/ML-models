# Gaussian Processes

Gaussian Processes (GPs) are a powerful, elegant non-parametric Bayesian approach to supervised learning that treats functions as random variables. Unlike traditional machine learning methods that learn specific function forms, GPs define probability distributions over functions, enabling both predictions and uncertainty quantification. This probabilistic framework makes GPs particularly valuable when uncertainty estimates are crucial for decision-making.

## Mathematical Framework

### Gaussian Process Definition

**Formal Definition:**
A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution.

**Function Distribution:**
```
f(x) ~ GP(μ(x), k(x, x'))
```

Where:
- **μ(x)** is the mean function
- **k(x, x')** is the covariance function (kernel)
- **f(x)** represents function values at input x

### Finite-Dimensional Distributions

**Joint Distribution:**
For any finite set of inputs X = {x₁, x₂, ..., xₙ}:
```
f = [f(x₁), f(x₂), ..., f(xₙ)]ᵀ ~ N(μ, K)
```

Where:
- **μ = [μ(x₁), μ(x₂), ..., μ(xₙ)]ᵀ** is the mean vector
- **K** is the n×n covariance matrix with K_{ij} = k(x_i, x_j)

### Mean and Covariance Functions

**Mean Function:**
Typically assumed to be zero for simplicity:
```
μ(x) = 0  (can be generalized)
```

**Covariance Function (Kernel):**
Must be positive semi-definite:
```
k(x, x') = Cov[f(x), f(x')]
```

**Kernel Properties:**
1. **Symmetry**: k(x, x') = k(x', x)
2. **Positive Semi-definite**: K ⪰ 0 for any finite set of inputs

## Kernel Functions

### Radial Basis Function (RBF/Squared Exponential)

**RBF Kernel:**
```
k(x, x') = σf² exp(-||x - x'||²/(2ℓ²))
```

Parameters:
- **σf²**: signal variance (output scale)
- **ℓ**: length scale (input scale)

**Properties:**
- Infinitely differentiable
- Produces smooth functions
- Universal approximator

### Matérn Kernel

**Matérn Class:**
```
k(x, x') = σf² (2^(1-ν)/Γ(ν)) (√(2ν)r/ℓ)^ν K_ν(√(2ν)r/ℓ)
```

Where:
- **r = ||x - x'||**
- **ν > 0** controls smoothness
- **K_ν** is the modified Bessel function
- **Γ(ν)** is the gamma function

**Special Cases:**
```
ν = 1/2: k(x,x') = σf² exp(-r/ℓ)  (Exponential)
ν = 3/2: k(x,x') = σf² (1 + √3r/ℓ) exp(-√3r/ℓ)
ν = 5/2: k(x,x') = σf² (1 + √5r/ℓ + 5r²/(3ℓ²)) exp(-√5r/ℓ)
ν → ∞: Converges to RBF kernel
```

### Linear Kernel

**Linear Kernel:**
```
k(x, x') = σf² + σb² + x^T Σ_p x'
```

Where:
- **σb²**: bias variance
- **Σ_p**: input covariance matrix

### Periodic Kernel

**Periodic Kernel:**
```
k(x, x') = σf² exp(-2sin²(π|x-x'|/p)/ℓ²)
```

Where:
- **p**: period
- **ℓ**: length scale within period

### Kernel Composition

**Addition:**
```
k(x, x') = k₁(x, x') + k₂(x, x')
```

**Multiplication:**
```
k(x, x') = k₁(x, x') × k₂(x, x')
```

**Scaling:**
```
k(x, x') = σ² k₀(x, x')
```

## Gaussian Process Regression

### Training Data

**Observed Data:**
```
D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ
```

**Likelihood Model:**
```
yᵢ = f(xᵢ) + εᵢ,  εᵢ ~ N(0, σₙ²)
```

**Joint Distribution:**
```
y = f + ε ~ N(0, K + σₙ²I)
```

Where **K** is the n×n kernel matrix.

### Posterior Inference

**Predictive Distribution:**
For a new input x*, the posterior distribution is:
```
f* | X, y, x* ~ N(μ*, σ*²)
```

**Predictive Mean:**
```
μ* = k*^T (K + σₙ²I)⁻¹ y
```

**Predictive Variance:**
```
σ*² = k** - k*^T (K + σₙ²I)⁻¹ k*
```

Where:
- **k* = [k(x*, x₁), ..., k(x*, xₙ)]^T** (n×1 vector)
- **k** = k(x*, x*) (scalar)

### Matrix Notation

**Kernel Matrices:**
```
K = [k(xᵢ, xⱼ)]ᵢ,ⱼ₌₁ⁿ  (n×n training kernel matrix)
K* = [k(x*ᵢ, xⱼ)]ᵢ₌₁ᵐ,ⱼ₌₁ⁿ  (m×n test-train kernel matrix)
K** = [k(x*ᵢ, x*ⱼ)]ᵢ,ⱼ₌₁ᵐ  (m×m test kernel matrix)
```

**Multi-point Prediction:**
```
f* | X, y, X* ~ N(μ*, Σ*)
```

**Mean Vector:**
```
μ* = K*^T (K + σₙ²I)⁻¹ y
```

**Covariance Matrix:**
```
Σ* = K** - K*^T (K + σₙ²I)⁻¹ K*
```

## Hyperparameter Optimization

### Marginal Likelihood

**Log Marginal Likelihood:**
```
log p(y|X, θ) = -½y^T(K + σₙ²I)⁻¹y - ½log|K + σₙ²I| - (n/2)log(2π)
```

Where θ represents hyperparameters.

**Gradient Computation:**
```
∂log p(y|X, θ)/∂θⱼ = ½y^T α α^T ∂K/∂θⱼ - ½tr(K⁻¹ ∂K/∂θⱼ)
```

Where α = (K + σₙ²I)⁻¹y.

### Optimization Methods

**Gradient-Based Optimization:**
```
θ^(t+1) = θ^(t) + η ∇_θ log p(y|X, θ)
```

**Common Optimizers:**
- L-BFGS: Quasi-Newton method
- Adam: Adaptive learning rates
- Conjugate Gradient: For large-scale problems

### Hyperparameter Initialization

**Length Scale:**
```
ℓ₀ = median{||xᵢ - xⱼ|| : i ≠ j}
```

**Signal Variance:**
```
σf²₀ = var(y)
```

**Noise Variance:**
```
σₙ²₀ = 0.1 × var(y)
```

## Computational Complexity

### Exact GP Inference

**Training Complexity:**
```
O(n³)  for matrix inversion
O(n²)  for each prediction
```

**Storage:**
```
O(n²)  for kernel matrix storage
```

### Scalability Issues

**Computational Bottlenecks:**
1. **Matrix Inversion**: O(n³) scales poorly
2. **Storage**: O(n²) memory requirement
3. **Hyperparameter Learning**: Multiple O(n³) operations

**Practical Limits:**
Standard GP methods become intractable for n > 10⁴ observations.

## Sparse Gaussian Processes

### Inducing Point Methods

**Inducing Points:**
Select m << n representative points Z = {z₁, ..., zₘ}.

**FITC (Fully Independent Training Conditional):**
```
q(f|u) = ∏ᵢ₌₁ⁿ p(fᵢ|u)
```

Where u = [f(z₁), ..., f(zₘ)]^T are inducing function values.

**Approximate Posterior:**
```
μ* ≈ kz*^T (Kzz + σₙ⁻²Kzx Kxz)⁻¹ σₙ⁻²Kzx y
```

**Complexity Reduction:**
```
O(nm²)  training time
O(m²)   prediction time
O(nm)   storage
```

### Variational Sparse GPs

**Variational Lower Bound:**
```
ℒ = ∫ q(f) log p(y|f) df - KL[q(u)||p(u)]
```

**Optimal Variational Distribution:**
```
q(u) = N(m, S)
```

With variational parameters m and S optimized via gradient ascent.

## GP Classification

### Binary Classification

**Likelihood:**
```
p(yᵢ|fᵢ) = Φ(yᵢfᵢ)  (Probit link)
p(yᵢ|fᵢ) = σ(yᵢfᵢ)  (Logistic link)
```

Where yᵢ ∈ {-1, +1}.

**Non-Gaussian Likelihood:**
Posterior is no longer Gaussian, requiring approximation methods.

### Laplace Approximation

**Gaussian Approximation:**
```
q(f|y) = N(f̂, (K⁻¹ + W)⁻¹)
```

Where:
- **f̂** is the mode of p(f|y)
- **W = -∇²log p(y|f)**

**Predictive Distribution:**
```
p(y*|X, y, x*) = ∫ p(y*|f*) q(f*|X, y, x*) df*
```

### Expectation Propagation

**Local Approximations:**
```
p(yᵢ|fᵢ) ≈ Zᵢ⁻¹ exp(aᵢfᵢ - ½bᵢfᵢ²)
```

**Global Approximation:**
```
q(f) = N(μ, Σ)
```

With parameters updated iteratively.

## Multi-output Gaussian Processes

### Independent GPs

**Separate GPs:**
```
fⱼ(x) ~ GP(0, kⱼ(x, x'))  for j = 1, ..., J
```

**Computational Cost:**
```
O(J × n³)  independent training
```

### Linear Model of Coregionalization

**Covariance Structure:**
```
Cov[fᵢ(x), fⱼ(x')] = ∑ᵩ₌₁ᵠ Aᵢⱼ^(q) kᵩ(x, x')
```

Where A^(q) are positive semi-definite matrices.

### Multi-task Learning

**Shared Components:**
```
fⱼ(x) = ∑ᵩ₌₁ᵠ aⱼᵩ gᵩ(x)
```

Where gᵩ(x) are shared latent functions.

## Advanced Topics

### Non-stationary Kernels

**Input-Dependent Length Scales:**
```
k(x, x') = σf² exp(-½(x-x')^T M(x,x')⁻¹ (x-x'))
```

Where M(x,x') varies with input location.

### Deep Gaussian Processes

**Hierarchical Composition:**
```
f^(L)(x) = f^(L-1)(f^(L-2)(...f^(1)(x)...))
```

Each layer f^(ℓ) is a GP.

### Gaussian Process Latent Variable Models

**Dimensionality Reduction:**
```
y = f(z) + ε,  z ~ N(0, I)
```

Where z are latent variables and f ~ GP.

## Model Selection and Validation

### Cross-Validation

**Leave-One-Out (LOO):**
```
CV = ∑ᵢ₌₁ⁿ (yᵢ - μ₋ᵢ(xᵢ))²
```

**Efficient LOO for GPs:**
```
μ₋ᵢ = μᵢ - αᵢ/Kᵢᵢ⁻¹
```

Where αᵢ = (K + σₙ²I)⁻¹y.

### Information Criteria

**Widely Applicable Information Criterion (WAIC):**
```
WAIC = -2∑ᵢ₌₁ⁿ log p(yᵢ|y₋ᵢ) + 2p_eff
```

Where p_eff is the effective number of parameters.

### Hyperparameter Uncertainty

**Bayesian Treatment:**
```
p(f*|y) = ∫ p(f*|y, θ) p(θ|y) dθ
```

**MCMC for Hyperparameters:**
Use Hamiltonian Monte Carlo or Elliptical Slice Sampling.

## Practical Implementation

### Numerical Stability

**Cholesky Decomposition:**
```
K + σₙ²I = LL^T
```

**Solving Linear Systems:**
```
α = L⁻ᵀ(L⁻¹y)
```

**Log Determinant:**
```
log|K + σₙ²I| = 2∑ᵢ₌₁ⁿ log Lᵢᵢ
```

### Preconditioning

**Incomplete Cholesky:**
Approximate K ≈ R^T R with sparse R.

**Conjugate Gradients:**
For solving (K + σₙ²I)α = y iteratively.

### Automatic Relevance Determination (ARD)

**Individual Length Scales:**
```
k(x, x') = σf² exp(-½∑ᵈⱼ₌₁ (xⱼ-x'ⱼ)²/ℓⱼ²)
```

**Feature Selection:**
Large ℓⱼ indicates irrelevant feature j.

## Advantages and Limitations

### Advantages

**Strengths:**
- **Uncertainty Quantification**: Natural uncertainty estimates
- **Non-parametric**: No fixed functional form
- **Automatic Model Selection**: Via marginal likelihood
- **Interpretable**: Clear probabilistic interpretation
- **Small Data Friendly**: Works well with limited training data
- **Flexible**: Wide variety of kernel functions

### Limitations

**Weaknesses:**
- **Computational Complexity**: O(n³) scaling
- **Kernel Selection**: Requires domain knowledge
- **Hyperparameter Sensitivity**: Performance depends on hyperparameters
- **High-Dimensional Inputs**: Curse of dimensionality
- **Non-Gaussian Likelihoods**: Requires approximations

### When to Use GPs

**Optimal Scenarios:**
- **Small-Medium Datasets**: n < 10⁴ observations
- **Uncertainty Critical**: When prediction confidence is important
- **Smooth Functions**: When underlying function is expected to be smooth
- **Active Learning**: When deciding where to collect new data
- **Bayesian Framework**: When probabilistic modeling is preferred

**Avoid When:**
- **Large Datasets**: n > 10⁵ without approximations
- **High-Dimensional Inputs**: p > 20 without careful kernel design
- **Real-time Requirements**: When fast prediction is critical
- **Non-smooth Functions**: When function has discontinuities

## Mathematical Summary

Gaussian Processes represent a paradigm shift from parametric to non-parametric Bayesian modeling:

1. **Function-Space Probability**: Treats functions as random variables
2. **Kernel-Driven Flexibility**: Kernel choice determines function properties
3. **Exact Bayesian Inference**: Closed-form posterior (for regression)
4. **Uncertainty Quantification**: Natural prediction intervals

The mathematical elegance of GPs lies in their ability to perform exact Bayesian inference over an infinite-dimensional function space using finite-dimensional computations through kernel evaluations.

**Key Takeaway**: Gaussian Processes demonstrate that sophisticated probabilistic modeling can be both mathematically elegant and practically useful. By placing probability distributions over functions rather than parameters, GPs provide a principled framework for uncertainty quantification that is particularly valuable in scientific applications where understanding prediction confidence is as important as the predictions themselves. 