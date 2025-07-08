# k-Nearest Neighbors (k-NN)

k-Nearest Neighbors (k-NN) is a non-parametric, instance-based, lazy learning algorithm that makes predictions based on the k most similar training examples in the feature space. Despite its simplicity, k-NN can be highly effective for many classification and regression problems, especially when the decision boundary is irregular or when local patterns are important.

## Mathematical Framework

### Core Algorithm

**Prediction Process:**
Given a query point x_q, find the k nearest neighbors in the training set and make a prediction based on their labels.

**Neighbor Set:**
```
N_k(x_q) = {x_i ∈ D : x_i is among the k nearest neighbors of x_q}
```

Where D is the training dataset.

### Distance Metrics

The choice of distance metric fundamentally affects k-NN performance.

**Euclidean Distance (L₂ norm):**
```
d(x_i, x_j) = √(∑_{d=1}^p (x_i^d - x_j^d)²)
```

**Manhattan Distance (L₁ norm):**
```
d(x_i, x_j) = ∑_{d=1}^p |x_i^d - x_j^d|
```

**Minkowski Distance (L_p norm):**
```
d(x_i, x_j) = (∑_{d=1}^p |x_i^d - x_j^d|^p)^(1/p)
```

**Mahalanobis Distance:**
```
d(x_i, x_j) = √((x_i - x_j)ᵀ Σ⁻¹ (x_i - x_j))
```

Where Σ is the covariance matrix.

**Cosine Distance:**
```
d(x_i, x_j) = 1 - (x_i · x_j)/(||x_i|| ||x_j||)
```

**Hamming Distance (for categorical features):**
```
d(x_i, x_j) = ∑_{d=1}^p I(x_i^d ≠ x_j^d)
```

Where I(·) is the indicator function.

## Classification

### Majority Voting

**Unweighted Voting:**
```
ŷ = argmax_c ∑_{x_i ∈ N_k(x_q)} I(y_i = c)
```

Where c represents the class labels.

**Class Probability Estimation:**
```
P(y = c|x_q) = (1/k) ∑_{x_i ∈ N_k(x_q)} I(y_i = c)
```

### Distance-Weighted Voting

**Weighted Prediction:**
```
ŷ = argmax_c ∑_{x_i ∈ N_k(x_q)} w_i · I(y_i = c)
```

**Weight Functions:**

**Inverse Distance Weighting:**
```
w_i = 1/d(x_q, x_i) + ε
```

Where ε is a small constant to avoid division by zero.

**Gaussian Weighting:**
```
w_i = exp(-d(x_q, x_i)²/(2σ²))
```

**Linear Decay:**
```
w_i = max(0, 1 - d(x_q, x_i)/τ)
```

Where τ is a threshold parameter.

### Probability Estimation

**Weighted Probability:**
```
P(y = c|x_q) = ∑_{x_i ∈ N_k(x_q)} w_i · I(y_i = c) / ∑_{x_i ∈ N_k(x_q)} w_i
```

## Regression

### Simple Averaging

**Unweighted Average:**
```
ŷ = (1/k) ∑_{x_i ∈ N_k(x_q)} y_i
```

### Distance-Weighted Regression

**Weighted Average:**
```
ŷ = ∑_{x_i ∈ N_k(x_q)} w_i · y_i / ∑_{x_i ∈ N_k(x_q)} w_i
```

### Local Linear Regression

**Local Polynomial Fitting:**
Fit a linear model using the k nearest neighbors:
```
ŷ = β₀ + ∑_{j=1}^p β_j x_q^j
```

Where β parameters are estimated using weighted least squares:
```
β̂ = (X_k^T W X_k)⁻¹ X_k^T W y_k
```

## Optimal k Selection

### Cross-Validation Approach

**Leave-One-Out Cross-Validation (LOOCV):**
```
CV(k) = (1/n) ∑_{i=1}^n L(y_i, ŷ_i^{(-i)}(k))
```

Where ŷ_i^{(-i)}(k) is the prediction for x_i using k-NN trained on all samples except x_i.

**Optimal k:**
```
k* = argmin_k CV(k)
```

### Bias-Variance Trade-off

**Bias:**
- Small k: Low bias (flexible, can capture local patterns)
- Large k: High bias (smoother decision boundary)

**Variance:**
- Small k: High variance (sensitive to noise)
- Large k: Low variance (more stable predictions)

**Optimal k Heuristics:**
```
k ≈ √n  (rule of thumb)
```

Where n is the number of training samples.

### Error Analysis

**Expected Error Decomposition:**
```
E[Error] = Bias² + Variance + Irreducible Error
```

**Asymptotic Error Rate:**
For large n and optimal k ∝ n^(4/(4+p)):
```
Error Rate ≈ O(n^(-4/(4+p)))
```

Where p is the number of dimensions.

## Curse of Dimensionality

### Distance Concentration

**High-Dimensional Distance Behavior:**
As dimensions increase, distances between points become more similar:
```
lim_{p→∞} (d_max - d_min)/d_min = 0
```

**Relative Distance Variance:**
```
Var[d]/E[d]² → 0 as p → ∞
```

### Volume of Hypersphere

**p-dimensional Unit Hypersphere Volume:**
```
V_p = π^(p/2) / Γ(p/2 + 1)
```

**Implication:**
Most volume in high dimensions is near the surface, making nearest neighbors less meaningful.

### Mitigation Strategies

**Feature Selection:**
Select relevant features to reduce dimensionality:
```
Select features with high relevance score: R(f_i) = |corr(f_i, y)|
```

**Dimensionality Reduction:**
Apply PCA or other techniques:
```
x_reduced = W^T x
```

Where W contains the top-r principal components.

## Computational Complexity

### Naive Implementation

**Training Time:**
```
O(1)  (lazy learning - no training phase)
```

**Prediction Time:**
```
O(np)  per query
```

Where:
- n: number of training samples
- p: number of features

**Space Complexity:**
```
O(np)  (stores entire training set)
```

### Optimized Implementations

### k-d Tree

**Construction Time:**
```
O(np log n)
```

**Query Time:**
```
O(log n)  in low dimensions (p ≤ 10)
O(n)      in high dimensions
```

**Effective Dimension Threshold:**
k-d trees become inefficient when p > 10-20.

### Ball Tree

**Construction Time:**
```
O(np log n)
```

**Query Time:**
```
O(log n)  for any dimension with proper distance metric
```

**Distance Metric Requirement:**
Requires triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)

### LSH (Locality Sensitive Hashing)

**Hash Functions:**
For Euclidean space:
```
h(x) = ⌊(a·x + b)/w⌋
```

Where a is a random vector, b is random shift, w is bucket width.

**Query Time:**
```
O(n^ρ)  where ρ < 1
```

**Approximation Factor:**
Returns (1+ε)-approximate nearest neighbors with high probability.

## Feature Scaling

### Standardization

**Z-score Normalization:**
```
x_scaled = (x - μ)/σ
```

**Min-Max Scaling:**
```
x_scaled = (x - x_min)/(x_max - x_min)
```

### Feature Weighting

**Learned Weights:**
```
d_weighted(x_i, x_j) = √(∑_{d=1}^p w_d(x_i^d - x_j^d)²)
```

**Weight Learning:**
```
w_d = 1/σ_d²  (inverse variance weighting)
```

## Advanced Variants

### Adaptive k-NN

**Dynamic k Selection:**
```
k(x_q) = argmin_k CV_local(k, x_q)
```

Based on local density and data distribution.

### Fuzzy k-NN

**Membership Function:**
```
u_i(x_q) = (1/d(x_q, x_i)^(2/(m-1))) / ∑_{j=1}^k (1/d(x_q, x_j)^(2/(m-1)))
```

**Fuzzy Prediction:**
```
ŷ = ∑_{i=1}^k u_i(x_q) y_i
```

### Locally Weighted k-NN

**Kernel-based Weighting:**
```
w_i = K(d(x_q, x_i)/h)
```

Where K is a kernel function (e.g., Gaussian) and h is bandwidth.

## Model Selection and Validation

### Hyperparameter Tuning

**Grid Search Parameters:**
- k: [1, 3, 5, 7, 9, 11, ...]
- Distance metric: [euclidean, manhattan, minkowski]
- Weight function: [uniform, distance, custom]

**Cross-Validation:**
```
Score(k, metric, weights) = (1/n_folds) ∑_{i=1}^{n_folds} Accuracy_i
```

### Performance Metrics

**Classification Metrics:**
```
Accuracy = (TP + TN)/(TP + TN + FP + FN)
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1-Score = 2 × (Precision × Recall)/(Precision + Recall)
```

**Regression Metrics:**
```
MSE = (1/n) ∑_{i=1}^n (y_i - ŷ_i)²
MAE = (1/n) ∑_{i=1}^n |y_i - ŷ_i|
R² = 1 - SS_res/SS_tot
```

## Theoretical Properties

### Consistency

**Stone's Theorem:**
k-NN is universally consistent if:
```
lim_{n→∞} k(n) = ∞  and  lim_{n→∞} k(n)/n = 0
```

**Convergence Rate:**
Under optimal conditions:
```
E[|ŷ - y|] = O(n^(-2/(2+p)))
```

### Bayes Error Rate

**Asymptotic Error:**
For infinite data:
```
lim_{n→∞} Error_kNN ≤ 2 × Bayes_Error × (1 - Bayes_Error)
```

This bound is tight for k = 1.

## Practical Considerations

### Advantages

**Strengths:**
- **Non-parametric**: Makes no assumptions about data distribution
- **Intuitive**: Easy to understand and implement
- **Versatile**: Works for both classification and regression
- **Local Adaptation**: Adapts to local patterns in data
- **No Training Time**: Immediate deployment
- **Multi-class Natural**: Handles multi-class problems naturally

### Disadvantages

**Limitations:**
- **Computational Cost**: Expensive prediction time
- **Memory Requirements**: Stores entire training set
- **Curse of Dimensionality**: Poor performance in high dimensions
- **Sensitivity to Noise**: Outliers can significantly affect results
- **Feature Scaling**: Requires careful preprocessing
- **Imbalanced Data**: Biased toward majority classes

### When to Use k-NN

**Optimal Scenarios:**
- **Small-Medium Datasets**: When computational cost is manageable
- **Irregular Decision Boundaries**: Complex, non-linear patterns
- **Local Patterns**: When locality is important
- **Mixed Data Types**: With appropriate distance metrics
- **Prototype Systems**: Quick implementation and testing

**Avoid When:**
- **High Dimensions**: p > 10-20 without dimensionality reduction
- **Large Datasets**: n > 10⁵ without optimization
- **Real-time Requirements**: When prediction speed is critical
- **Sparse Data**: When most features are zero
- **Linear Separable**: When simple linear models suffice

## Implementation Guidelines

### Preprocessing Pipeline

```python
1. Handle missing values (imputation or removal)
2. Encode categorical variables (one-hot or label encoding)
3. Scale features (standardization or normalization)
4. Apply dimensionality reduction if needed
5. Split data (train/validation/test)
```

### Performance Optimization

**For Large Datasets:**
1. Use approximate algorithms (LSH, random sampling)
2. Implement early stopping in distance computation
3. Use efficient data structures (k-d tree, ball tree)
4. Consider parallel processing

**For High Dimensions:**
1. Apply feature selection or PCA
2. Use sparse data structures
3. Consider alternative distance metrics
4. Implement approximate nearest neighbor search

## Mathematical Summary

k-Nearest Neighbors demonstrates that sophisticated machine learning doesn't always require complex mathematics. The algorithm's power lies in:

1. **Distance-based Learning**: Using geometric proximity as a proxy for similarity
2. **Non-parametric Flexibility**: Adapting to any data distribution
3. **Local Decision Making**: Making predictions based on local neighborhoods
4. **Simplicity**: Minimal assumptions and easy implementation

The key mathematical insight is that in many real-world problems, similar inputs tend to produce similar outputs, and this locality can be captured through distance metrics in feature space.

**Key Takeaway**: k-NN showcases the principle that "you are known by the company you keep" in machine learning. While simple in concept, it requires careful consideration of distance metrics, dimensionality, and computational trade-offs to be effective in practice. Understanding k-NN provides fundamental insights into instance-based learning and the importance of similarity measures in machine learning. 