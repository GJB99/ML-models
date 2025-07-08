# Regularized Regression

Regularized regression methods are variants of linear regression that include penalty terms in the loss function to prevent overfitting and handle high-dimensional data. These techniques are essential for dealing with multicollinearity, feature selection, and situations where the number of features exceeds the number of samples. They represent a fundamental approach to controlling model complexity through mathematical regularization.

## Mathematical Framework

### General Regularized Regression

**Objective Function:**
```
L(β) = ||y - Xβ||₂² + λR(β)
```

Where:
- **||y - Xβ||₂²** is the sum of squared residuals (data fitting term)
- **R(β)** is the regularization penalty
- **λ ≥ 0** is the regularization parameter controlling the penalty strength

### Matrix Notation

**Data Setup:**
- **X ∈ ℝⁿˣᵖ**: design matrix (n samples, p features)
- **y ∈ ℝⁿ**: response vector
- **β ∈ ℝᵖ**: coefficient vector

**Regularized Objective:**
```
β̂ = argmin_β [½||y - Xβ||₂² + λR(β)]
```

## Ridge Regression (L₂ Regularization)

### Mathematical Formulation

**Ridge Objective Function:**
```
L_Ridge(β) = ½||y - Xβ||₂² + λ||β||₂²
```

**Expanded Form:**
```
L_Ridge(β) = ½∑ᵢ₌₁ⁿ(yᵢ - xᵢᵀβ)² + λ∑ⱼ₌₁ᵖβⱼ²
```

### Analytical Solution

**Closed-Form Solution:**
```
β̂_Ridge = (XᵀX + λI)⁻¹Xᵀy
```

**Matrix Derivation:**
Taking the derivative and setting to zero:
```
∂L/∂β = -Xᵀ(y - Xβ) + 2λβ = 0
```

**Solving:**
```
XᵀXβ + λβ = Xᵀy
(XᵀX + λI)β = Xᵀy
```

### Properties

**Shrinkage Factor:**
For the i-th singular value dᵢ of X:
```
shrinkage_factor = dᵢ²/(dᵢ² + λ)
```

**Effective Degrees of Freedom:**
```
df(λ) = tr[X(XᵀX + λI)⁻¹Xᵀ] = ∑ᵢ₌₁ᵖ dᵢ²/(dᵢ² + λ)
```

## Lasso Regression (L₁ Regularization)

### Mathematical Formulation

**Lasso Objective Function:**
```
L_Lasso(β) = ½||y - Xβ||₂² + λ||β||₁
```

**Expanded Form:**
```
L_Lasso(β) = ½∑ᵢ₌₁ⁿ(yᵢ - xᵢᵀβ)² + λ∑ⱼ₌₁ᵖ|βⱼ|
```

### Optimization Challenges

**Non-Differentiability:**
The L₁ penalty is not differentiable at β = 0, requiring specialized optimization techniques.

**Subdifferential:**
```
∂|βⱼ| = {
    +1      if βⱼ > 0
    [-1,1]  if βⱼ = 0
    -1      if βⱼ < 0
}
```

### Coordinate Descent Algorithm

**Soft Thresholding:**
For feature j, holding all other coefficients fixed:
```
β̂ⱼ = S(ρⱼ, λ) = sign(ρⱼ) max(|ρⱼ| - λ, 0)
```

Where:
```
ρⱼ = xⱼᵀ(y - X₋ⱼβ₋ⱼ)
```

**Algorithm Steps:**
1. Initialize β⁽⁰⁾
2. For each iteration t and feature j:
   ```
   βⱼ⁽ᵗ⁺¹⁾ = S(ρⱼ⁽ᵗ⁾, λ)
   ```
3. Repeat until convergence

### Feature Selection Property

**Sparsity Induction:**
Lasso can set coefficients exactly to zero:
```
P(βⱼ = 0) > 0 for any λ > 0
```

**Sparsity Level:**
As λ increases, more coefficients become zero:
```
||β̂_Lasso||₀ decreases with λ
```

## Elastic Net

### Mathematical Formulation

**Elastic Net Objective:**
```
L_ElasticNet(β) = ½||y - Xβ||₂² + λ₁||β||₁ + λ₂||β||₂²
```

**Alternative Parameterization:**
```
L_ElasticNet(β) = ½||y - Xβ||₂² + λ[α||β||₁ + (1-α)||β||₂²]
```

Where:
- **α ∈ [0,1]** is the mixing parameter
- **λ > 0** is the overall regularization strength

### Properties

**Grouped Variable Selection:**
Elastic Net tends to select groups of correlated variables:
```
If corr(xⱼ, xₖ) ≈ 1, then |βⱼ - βₖ| ≈ 0
```

**Sparsity and Smoothness:**
- **α = 0**: Pure Ridge (no sparsity, smooth coefficients)
- **α = 1**: Pure Lasso (sparse solution)
- **0 < α < 1**: Compromise between sparsity and grouping

### Optimization

**Coordinate Descent for Elastic Net:**
```
βⱼ⁽ᵗ⁺¹⁾ = S(ρⱼ⁽ᵗ⁾, λα) / (1 + λ(1-α))
```

## Regularization Path

### Solution Path Analysis

**Ridge Path:**
```
β̂_Ridge(λ) = (XᵀX + λI)⁻¹Xᵀy
```

The path is smooth and all coefficients shrink continuously.

**Lasso Path:**
The path is piecewise linear with kinks at points where coefficients enter/leave the active set.

**Critical Points:**
```
λₖ = max{|xⱼᵀr⁽ᵏ⁾| : j ∉ Active⁽ᵏ⁾}
```

Where r⁽ᵏ⁾ is the residual at step k.

### LARS Algorithm

**Least Angle Regression (LARS):**
Efficiently computes the entire Lasso path:

1. Start with all coefficients at zero
2. Find feature most correlated with response
3. Move in direction of highest correlation until another feature has equal correlation
4. Continue with both features until another joins
5. Repeat until all features are included

## Cross-Validation for λ Selection

### K-Fold Cross-Validation

**CV Error:**
```
CV(λ) = (1/K) ∑ₖ₌₁ᴷ ||y⁽ᵏ⁾ - X⁽ᵏ⁾β̂₋ₖ(λ)||₂²
```

Where β̂₋ₖ(λ) is trained on all folds except k.

**Optimal λ:**
```
λ* = argmin_λ CV(λ)
```

**One Standard Error Rule:**
```
λ₁ₛₑ = max{λ : CV(λ) ≤ CV(λ*) + SE(λ*)}
```

### Information Criteria

**AIC (Akaike Information Criterion):**
```
AIC(λ) = n log(RSS(λ)/n) + 2df(λ)
```

**BIC (Bayesian Information Criterion):**
```
BIC(λ) = n log(RSS(λ)/n) + log(n)df(λ)
```

Where df(λ) is the effective degrees of freedom.

## Theoretical Properties

### Statistical Consistency

**Oracle Property (Lasso):**
Under certain conditions, Lasso achieves variable selection consistency:
```
P(Ŝ = S*) → 1 as n → ∞
```

Where S* is the true active set.

**Irrepresentability Condition:**
```
||XᵀₙₛXₛ(XᵀₛXₛ)⁻¹sign(β*ₛ)||∞ < 1
```

### Prediction Error Bounds

**Ridge Regression Bound:**
```
E[||β̂_Ridge - β*||₂²] ≤ (σ²tr[(XᵀX + λI)⁻¹XᵀX] + λ²||β*||₂²) / n
```

**Lasso Bound (under restricted eigenvalue condition):**
```
||β̂_Lasso - β*||₂² ≤ C(σ²s log p/n + λ²s)
```

Where s = ||β*||₀ is the sparsity level.

## Computational Complexity

### Ridge Regression

**Direct Solution:**
```
O(p³ + np²) for matrix inversion
```

**SVD Approach:**
```
O(np²) for SVD + O(p) for each λ
```

### Lasso/Elastic Net

**Coordinate Descent:**
```
O(np × iterations)
```

**Path Algorithm (LARS):**
```
O(np²) for full path
```

## Extensions and Variants

### Group Lasso

**Group Penalty:**
```
R_group(β) = ∑ₘ₌₁ᴹ ||β_Gₘ||₂
```

Where Gₘ are predefined groups of variables.

### Fused Lasso

**Fusion Penalty:**
```
R_fused(β) = ||β||₁ + λ₂∑ⱼ₌₁ᵖ⁻¹|βⱼ₊₁ - βⱼ|
```

Encourages sparsity and smoothness in coefficient differences.

### Adaptive Lasso

**Weighted L₁ Penalty:**
```
R_adaptive(β) = ∑ⱼ₌₁ᵖ wⱼ|βⱼ|
```

Where weights wⱼ = 1/|β̂_OLS,j|^γ for some γ > 0.

## High-Dimensional Asymptotics

### Scaling Regimes

**Classical Setting:** n → ∞, p fixed
**High-Dimensional:** p = O(n^α) for some α > 0
**Ultra-High-Dimensional:** p >> n

### Performance Guarantees

**Restricted Eigenvalue Condition:**
For subset S ⊆ {1,...,p} with |S| ≤ s:
```
min_{v∈C(S)} (vᵀXᵀXv)/(n||v||₂²) ≥ φ₀² > 0
```

Where C(S) is the restricted cone.

**Rate of Convergence:**
```
||β̂ - β*||₂² = O_P(s log p/n)
```

## Practical Implementation

### Preprocessing

**Standardization:**
```
X̃ⱼ = (Xⱼ - μⱼ)/σⱼ, ỹ = (y - μᵧ)/σᵧ
```

**Why Necessary:**
Regularization penalties are scale-dependent.

### Warm Starts

**Path-Following:**
Use solution at λₖ as initial point for λₖ₊₁:
```
β⁽⁰⁾(λₖ₊₁) = β̂(λₖ)
```

### Active Set Methods

**Identification:**
```
Active(λ) = {j : βⱼ(λ) ≠ 0}
```

**Coordinate Screening:**
```
Skip update if |xⱼᵀr| < λ - λ_old
```

## Model Selection and Validation

### Hyperparameter Tuning

**Grid Search:**
```
λ_grid = [λ_max × r^k : k = 0,1,...,K]
```

Where r = 0.1 and λ_max = ||Xᵀy||∞.

**Random Search:**
```
log(λ) ~ Uniform(log(λ_min), log(λ_max))
```

### Performance Metrics

**Prediction Error:**
```
MSE = (1/n_test) ||y_test - X_test β̂||₂²
```

**Feature Selection Metrics:**
```
Precision = |Ŝ ∩ S*| / |Ŝ|
Recall = |Ŝ ∩ S*| / |S*|
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

## When to Use Each Method

### Ridge Regression

**Optimal for:**
- **Multicollinearity**: When features are highly correlated
- **Small n, Large p**: More samples than features
- **Shrinkage Without Selection**: Want to keep all features
- **Stability**: When consistent predictions are important

### Lasso Regression

**Optimal for:**
- **Feature Selection**: When interpretability is crucial
- **Sparse Solutions**: When most features are irrelevant
- **Large p, Small n**: High-dimensional data
- **Automatic Variable Selection**: Want built-in feature selection

### Elastic Net

**Optimal for:**
- **Grouped Variables**: When correlated features should be selected together
- **Large p >> n**: Ultra-high-dimensional problems
- **Compromise**: Want both shrinkage and selection
- **Robustness**: When neither pure Ridge nor Lasso performs well

## Mathematical Summary

Regularized regression methods demonstrate the power of adding mathematical constraints to improve generalization:

1. **Ridge**: L₂ penalty provides smooth shrinkage through eigenvalue modification
2. **Lasso**: L₁ penalty induces sparsity through soft thresholding
3. **Elastic Net**: Combined penalties balance shrinkage and selection

The key insight is that by controlling model complexity through regularization, we can achieve better bias-variance trade-offs, especially in high-dimensional settings where traditional OLS fails.

**Key Takeaway**: Regularization transforms the ill-posed problem of high-dimensional regression into well-posed optimization problems with unique solutions. The choice of penalty function (L₁, L₂, or mixed) determines whether the solution emphasizes smoothness, sparsity, or both, making regularized regression a cornerstone technique for modern high-dimensional data analysis. 