# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a fundamental dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving the maximum amount of variance. Developed by Karl Pearson in 1901 and formalized by Harold Hotelling in 1933, PCA has become one of the most widely used techniques in data science, machine learning, and statistical analysis.

## Mathematical Framework

### Data Representation

**Original Data Matrix:**
```
X ∈ ℝ^{n×p}
```

Where:
- **n**: number of observations (samples)
- **p**: number of features (dimensions)
- **xᵢⱼ**: value of feature j for observation i

**Centered Data Matrix:**
```
X_centered = X - 1_n μᵀ
```

Where:
- **μ ∈ ℝᵖ**: sample mean vector μⱼ = (1/n)∑ᵢ₌₁ⁿ xᵢⱼ
- **1_n**: n-dimensional vector of ones

### Covariance Matrix

**Sample Covariance Matrix:**
```
C = (1/(n-1)) X_centered^T X_centered ∈ ℝ^{p×p}
```

**Element-wise Form:**
```
Cⱼₖ = (1/(n-1)) ∑ᵢ₌₁ⁿ (xᵢⱼ - μⱼ)(xᵢₖ - μₖ)
```

**Properties:**
- **Symmetric**: C = Cᵀ
- **Positive Semi-definite**: C ⪰ 0
- **Diagonal Elements**: Cⱼⱼ = Var(Xⱼ)
- **Off-diagonal Elements**: Cⱼₖ = Cov(Xⱼ, Xₖ)

### Eigenvalue Decomposition

**Spectral Decomposition:**
```
C = PΛPᵀ
```

Where:
- **P ∈ ℝ^{p×p}**: matrix of eigenvectors (principal directions)
- **Λ = diag(λ₁, λ₂, ..., λₚ)**: diagonal matrix of eigenvalues
- **λ₁ ≥ λ₂ ≥ ... ≥ λₚ ≥ 0**: eigenvalues in descending order

**Eigenvalue Equation:**
```
Cpⱼ = λⱼpⱼ
```

**Orthonormality:**
```
PᵀP = I  (columns of P are orthonormal)
```

## Principal Components

### Component Definition

**k-th Principal Component:**
```
PCₖ = Xpₖ ∈ ℝⁿ
```

Where pₖ is the k-th eigenvector (column of P).

**Principal Component Scores:**
```
Y = XP ∈ ℝ^{n×p}
```

Where Y contains all principal component scores.

### Variance Properties

**Variance of k-th Principal Component:**
```
Var(PCₖ) = λₖ
```

**Total Variance:**
```
Total Variance = ∑ⱼ₌₁ᵖ λⱼ = tr(C) = ∑ⱼ₌₁ᵖ Var(Xⱼ)
```

**Variance Explained by k-th Component:**
```
Proportion_k = λₖ / (∑ⱼ₌₁ᵖ λⱼ)
```

**Cumulative Variance Explained:**
```
Cumulative_k = (∑ⱼ₌₁ᵏ λⱼ) / (∑ⱼ₌₁ᵖ λⱼ)
```

## Optimization Perspective

### Variance Maximization

**First Principal Component:**
Solve the optimization problem:
```
max w^T C w
subject to ||w||₂ = 1
```

**Lagrangian:**
```
L(w, λ) = w^T C w - λ(w^T w - 1)
```

**First-Order Condition:**
```
∇_w L = 2Cw - 2λw = 0  →  Cw = λw
```

**Solution:**
w = p₁ (first eigenvector), with maximum value λ₁.

**Subsequent Components:**
For k-th component, maximize variance subject to orthogonality:
```
max w^T C w
subject to ||w||₂ = 1, w^T pⱼ = 0 for j = 1,...,k-1
```

### Projection and Reconstruction

**k-dimensional Projection:**
```
Y_k = XP_k ∈ ℝ^{n×k}
```

Where P_k contains the first k eigenvectors.

**Reconstruction:**
```
X̂_k = Y_k P_k^T ∈ ℝ^{n×p}
```

**Reconstruction Error:**
```
Error = ||X - X̂_k||_F^2 = ∑ⱼ₌ₖ₊₁ᵖ λⱼ
```

## SVD Approach

### Singular Value Decomposition

**SVD of Centered Data:**
```
X_centered = UΣVᵀ
```

Where:
- **U ∈ ℝ^{n×n}**: left singular vectors
- **Σ = diag(σ₁, σ₂, ..., σᵣ) ∈ ℝ^{n×p}**: singular values
- **V ∈ ℝ^{p×p}**: right singular vectors
- **r = rank(X_centered)**: rank of the data matrix

### Connection to Eigendecomposition

**Relationship to Covariance Matrix:**
```
C = (1/(n-1)) X_centered^T X_centered = (1/(n-1)) VΣ²Vᵀ
```

**Principal Components via SVD:**
- **Eigenvectors**: V (right singular vectors)
- **Eigenvalues**: λⱼ = σⱼ²/(n-1)
- **Principal Component Scores**: Y = UΣ

### Computational Advantages

**Direct Computation:**
SVD avoids explicit covariance matrix computation, which is beneficial when:
- **n >> p**: SVD is more efficient
- **Numerical stability**: Avoids potential issues with Cᵀ C formation

**Computational Complexity:**
- **Covariance + Eigen**: O(np² + p³)
- **SVD**: O(min(n²p, np²))

## Dimensionality Selection

### Scree Plot Analysis

**Eigenvalue Plot:**
Plot λⱼ vs. j and look for the "elbow" point where eigenvalues level off.

**Scree Test:**
Choose k where λₖ₊₁/λₖ shows a significant drop.

### Variance Threshold

**Cumulative Variance:**
Choose k such that:
```
(∑ⱼ₌₁ᵏ λⱼ) / (∑ⱼ₌₁ᵖ λⱼ) ≥ threshold
```

Common thresholds: 80%, 90%, 95%.

**Kaiser Criterion:**
Retain components with λⱼ > 1 (for standardized data).

### Cross-Validation

**Reconstruction Error:**
Use cross-validation to select k minimizing reconstruction error:
```
CV(k) = (1/V) ∑ᵥ₌₁ⱽ ||X_val^{(v)} - X̂_k^{(v)}||_F²
```

## Kernel PCA

### Nonlinear Extension

**Feature Mapping:**
```
φ: ℝᵖ → ℋ  (mapping to reproducing kernel Hilbert space)
```

**Kernel Matrix:**
```
K ∈ ℝ^{n×n}, Kᵢⱼ = κ(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩
```

**Centering in Feature Space:**
```
K_centered = K - 1_n K/n - K1_n/n + 1_n K 1_n/n²
```

**Principal Components in Feature Space:**
Solve eigenvalue problem:
```
K_centered α = λα
```

**Projection onto k-th Component:**
```
PC_k(x) = ∑ᵢ₌₁ⁿ αᵢₖ κ(xᵢ, x)
```

### Common Kernels

**Polynomial Kernel:**
```
κ(x, y) = (x^T y + c)^d
```

**RBF (Gaussian) Kernel:**
```
κ(x, y) = exp(-γ||x - y||²)
```

**Sigmoid Kernel:**
```
κ(x, y) = tanh(γx^T y + c)
```

## Probabilistic PCA

### Generative Model

**Latent Variable Model:**
```
z ~ N(0, I_q)  (latent variables)
x = Wz + μ + ε  (observation model)
ε ~ N(0, σ²I_p)  (noise)
```

Where:
- **W ∈ ℝ^{p×q}**: loading matrix
- **q < p**: latent dimension

**Marginal Distribution:**
```
x ~ N(μ, C)  where C = WWᵀ + σ²I
```

### Parameter Estimation

**Maximum Likelihood:**
```
W_ML = U_q(Λ_q - σ²I)^{1/2} R
```

Where:
- **U_q**: first q eigenvectors of sample covariance
- **Λ_q**: first q eigenvalues
- **R**: arbitrary orthogonal matrix

**Noise Variance:**
```
σ²_ML = (1/(p-q)) ∑ⱼ₌ₑ₊₁ᵖ λⱼ
```

## Factor Analysis Connection

### Factor Model

**Factor Analysis:**
```
x = Λf + ε
```

Where:
- **Λ ∈ ℝ^{p×q}**: factor loadings
- **f ~ N(0, I)**: common factors
- **ε ~ N(0, Ψ)**: specific factors (Ψ diagonal)

**Covariance Structure:**
```
C = ΛΛᵀ + Ψ
```

**Difference from PCA:**
- **PCA**: Ψ = σ²I (isotropic noise)
- **Factor Analysis**: Ψ diagonal (heteroscedastic noise)

## Applications and Use Cases

### Data Compression

**Dimensionality Reduction:**
Store only first k principal components:
```
Storage: n × k + k × p  instead of  n × p
```

**Compression Ratio:**
```
Ratio = (nk + kp)/(np) = k(n + p)/(np)
```

### Noise Reduction

**Signal vs. Noise Separation:**
- **Signal**: captured by first k components (large eigenvalues)
- **Noise**: remaining components (small eigenvalues)

**Denoising:**
```
X_denoised = X P_k P_k^T
```

### Visualization

**2D/3D Projection:**
Project onto first 2 or 3 principal components:
```
Y_{vis} = X P_{1:2} or X P_{1:3}
```

**Biplot:**
Simultaneous display of observations and variables.

### Feature Engineering

**Uncorrelated Features:**
Principal components are uncorrelated by construction:
```
Cov(PC_i, PC_j) = 0 for i ≠ j
```

**Multicollinearity Removal:**
Eliminates linear dependencies among original features.

## Theoretical Properties

### Optimality

**Variance Maximization:**
PCA maximizes the variance captured by each component sequentially.

**Reconstruction Error Minimization:**
PCA minimizes the sum of squared reconstruction errors:
```
min_P_k ||X - XP_k P_k^T||_F²
```

**Eckart-Young Theorem:**
PCA provides the best rank-k approximation to the data matrix in Frobenius norm.

### Statistical Properties

**Consistency:**
As n → ∞, sample eigenvalues and eigenvectors converge to population values.

**Asymptotic Distribution:**
Under normality assumptions:
```
√n(λ̂_j - λ_j) → N(0, 2λ_j²)
```

**Perturbation Theory:**
Small changes in data lead to small changes in principal components (under eigengap conditions).

## Computational Considerations

### Algorithms

**Power Method:**
For computing dominant eigenvector:
```
v^{(t+1)} = Cv^{(t)} / ||Cv^{(t)}||
```

**Lanczos Algorithm:**
Efficient for sparse matrices, computes k largest eigenvalues.

**Randomized SVD:**
Fast approximation for large matrices:
```
A ≈ QQ^T A  where Q has orthonormal columns
```

### Memory and Complexity

**Memory Requirements:**
- **Covariance Method**: O(p²)
- **SVD Method**: O(np + p²)

**Time Complexity:**
- **Eigendecomposition**: O(p³)
- **SVD**: O(min(n²p, np²))
- **Randomized methods**: O(npk) for k components

### Numerical Stability

**Centering:**
Always center data to avoid numerical issues:
```
X_centered = X - 1_n μ^T
```

**Standardization:**
For mixed-scale data:
```
X_std = (X - μ) / σ
```

**Condition Number:**
Monitor C's condition number to detect numerical issues.

## Limitations and Assumptions

### Assumptions

**Linearity:**
PCA assumes linear relationships between variables.

**Orthogonality:**
Principal components are constrained to be orthogonal.

**Gaussian Distribution:**
Many theoretical results assume multivariate normality.

**Stationarity:**
Assumes relationships between variables are constant.

### Limitations

**Interpretability:**
Principal components are linear combinations of all original features.

**Sensitivity to Scaling:**
Results heavily depend on variable scales without standardization.

**Linear Relationships Only:**
Cannot capture nonlinear patterns (use Kernel PCA or nonlinear methods).

**Outlier Sensitivity:**
Sensitive to outliers due to variance maximization objective.

### When Not to Use PCA

**Sparse Data:**
PCA creates dense representations from sparse data.

**Interpretability Required:**
When individual feature interpretation is crucial.

**Nonlinear Relationships:**
When relationships are fundamentally nonlinear.

**Small Sample Size:**
When n < p, covariance matrix may be singular.

## Extensions and Variants

### Sparse PCA

**Sparsity Constraint:**
```
max w^T C w - λ||w||_1
subject to ||w||_2 = 1
```

Produces sparse loadings for interpretability.

### Robust PCA

**Robust Covariance:**
Use robust estimators less sensitive to outliers:
```
C_robust = median-based covariance estimate
```

**L1-PCA:**
Minimize L1 norm instead of L2:
```
min ∑ᵢⱼ |X_ij - (XP_k P_k^T)_ij|
```

### Incremental PCA

**Online Updates:**
Update principal components as new data arrives:
```
C^{(t+1)} = (1-α)C^{(t)} + α·outer(x_{t+1})
```

**Streaming PCA:**
For large datasets that don't fit in memory.

### Functional PCA

**Continuous Functions:**
For functional data f(t):
```
f(t) = μ(t) + ∑_{k=1}^∞ ξ_k φ_k(t)
```

Where φ_k(t) are functional principal components.

## Implementation Guidelines

### Preprocessing Pipeline

```python
# 1. Center the data
X_centered = X - np.mean(X, axis=0)

# 2. Optional: Standardize
X_std = X_centered / np.std(X_centered, axis=0)

# 3. Compute PCA
U, s, Vt = np.linalg.svd(X_std)
principal_components = Vt.T
eigenvalues = s**2 / (n-1)
```

### Validation Strategy

**Cross-Validation:**
```python
def pca_cv(X, k_values):
    errors = []
    for k in k_values:
        error = cross_val_reconstruction_error(X, k)
        errors.append(error)
    return errors
```

**Information Criteria:**
```python
def aic_pca(X, k):
    reconstruction_error = compute_error(X, k)
    penalty = 2 * k * (2*p - k) / n
    return reconstruction_error + penalty
```

## Mathematical Summary

Principal Component Analysis represents one of the most elegant applications of linear algebra to data analysis:

1. **Eigenvalue Problem**: PCA transforms the variance maximization problem into an eigenvalue decomposition
2. **Geometric Interpretation**: Finds orthogonal directions of maximum variance in data space
3. **Optimality Properties**: Provides optimal low-rank approximations in least-squares sense
4. **Spectral Analysis**: Reveals the intrinsic dimensionality and structure of data

The mathematical foundation of PCA demonstrates how linear algebraic concepts (eigenvalues, eigenvectors, SVD) directly solve practical data analysis problems.

**Key Insight**: PCA's power lies in its dual optimization property—it simultaneously maximizes variance in the projected space and minimizes reconstruction error. This duality, established through the spectral theorem for symmetric matrices, makes PCA both theoretically principled and practically effective for dimensionality reduction, visualization, and feature engineering tasks. 