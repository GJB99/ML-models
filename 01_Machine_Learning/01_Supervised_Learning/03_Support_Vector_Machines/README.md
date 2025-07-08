# Support Vector Machines (SVM)

Support Vector Machines represent sophisticated supervised learning algorithms that find optimal hyperplanes to separate data classes by maximizing margins between different classes. Developed by Vladimir Vapnik and colleagues in the 1990s, SVMs excel in both classification and regression tasks through the principle of structural risk minimization.

## Mathematical Framework

### Linear SVM (Hard Margin)

**Hyperplane Equation:**
A separating hyperplane in d-dimensional space is defined as:
```
w^T x + b = 0
```

Where:
- **w** ∈ ℝᵈ is the weight vector (normal to hyperplane)
- **b** ∈ ℝ is the bias term
- **x** ∈ ℝᵈ is the input vector

**Distance from Point to Hyperplane:**
```
distance = |w^T x + b| / ||w||₂
```

**Classification Rule:**
```
f(x) = sign(w^T x + b)
```

### Margin Maximization

**Functional Margin:**
```
γ̂ᵢ = yᵢ(w^T xᵢ + b)
```

**Geometric Margin:**
```
γᵢ = γ̂ᵢ / ||w||₂ = yᵢ(w^T xᵢ + b) / ||w||₂
```

**Optimization Problem (Primal Form):**
```
max γ subject to yᵢ(w^T xᵢ + b) ≥ γ, i = 1,...,n
```

**Equivalent Formulation:**
```
min (1/2)||w||₂²
subject to yᵢ(w^T xᵢ + b) ≥ 1, i = 1,...,n
```

### Lagrangian Dual Problem

**Lagrangian:**
```
L(w, b, α) = (1/2)||w||₂² - Σᵢ₌₁ⁿ αᵢ[yᵢ(w^T xᵢ + b) - 1]
```

**KKT Conditions:**
```
∇w L = w - Σᵢ₌₁ⁿ αᵢyᵢxᵢ = 0  →  w = Σᵢ₌₁ⁿ αᵢyᵢxᵢ
∇b L = -Σᵢ₌₁ⁿ αᵢyᵢ = 0       →  Σᵢ₌₁ⁿ αᵢyᵢ = 0
αᵢ ≥ 0
αᵢ[yᵢ(w^T xᵢ + b) - 1] = 0  (complementary slackness)
```

**Dual Problem:**
```
max Σᵢ₌₁ⁿ αᵢ - (1/2) Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ αᵢαⱼyᵢyⱼxᵢ^T xⱼ
subject to Σᵢ₌₁ⁿ αᵢyᵢ = 0, αᵢ ≥ 0
```

**Decision Function:**
```
f(x) = sign(Σᵢ₌₁ⁿ αᵢyᵢxᵢ^T x + b)
```

### Soft Margin SVM

**Slack Variables:**
To handle non-separable data, introduce slack variables ξᵢ ≥ 0:
```
yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ
```

**Primal Optimization:**
```
min (1/2)||w||₂² + C Σᵢ₌₁ⁿ ξᵢ
subject to yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Dual Formulation:**
```
max Σᵢ₌₁ⁿ αᵢ - (1/2) Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ αᵢαⱼyᵢyⱼxᵢ^T xⱼ
subject to Σᵢ₌₁ⁿ αᵢyᵢ = 0, 0 ≤ αᵢ ≤ C
```

### Support Vectors

**Support Vector Classification:**
- **αᵢ = 0**: Not support vectors (correctly classified with margin ≥ 1)
- **0 < αᵢ < C**: Support vectors on margin boundary (ξᵢ = 0)
- **αᵢ = C**: Support vectors inside margin or misclassified (ξᵢ > 0)

**Bias Calculation:**
```
b = yⱼ - Σᵢ∈SV αᵢyᵢxᵢ^T xⱼ
```
where j is any support vector with 0 < αⱼ < C.

### Kernel Methods

**Kernel Trick:**
Replace inner products xᵢ^T xⱼ with kernel function K(xᵢ, xⱼ):
```
f(x) = sign(Σᵢ₌₁ⁿ αᵢyᵢK(xᵢ, x) + b)
```

**Common Kernels:**

**Linear Kernel:**
```
K(xᵢ, xⱼ) = xᵢ^T xⱼ
```

**Polynomial Kernel:**
```
K(xᵢ, xⱼ) = (γxᵢ^T xⱼ + r)ᵈ
```

**Radial Basis Function (RBF/Gaussian) Kernel:**
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||₂²)
```

**Sigmoid Kernel:**
```
K(xᵢ, xⱼ) = tanh(γxᵢ^T xⱼ + r)
```

### Kernel Properties

**Mercer's Condition:**
A function K(x, y) is a valid kernel if and only if the kernel matrix is positive semi-definite:
```
K = [K(xᵢ, xⱼ)]ᵢ,ⱼ ⪰ 0
```

**Reproducing Kernel Hilbert Space (RKHS):**
Kernels correspond to inner products in some feature space φ(x):
```
K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩
```

### Multi-class SVM

**One-vs-One (OvO):**
Train K(K-1)/2 binary classifiers for K classes.

**One-vs-Rest (OvR):**
Train K binary classifiers, each separating one class from all others.

**Crammer-Singer Multi-class:**
```
min (1/2) Σₖ₌₁ᴷ ||wₖ||₂² + C Σᵢ₌₁ⁿ ξᵢ
subject to wᵧᵢ^T xᵢ + bᵧᵢ ≥ wₖ^T xᵢ + bₖ + 2 - ξᵢ, ∀k ≠ yᵢ
```

### SVM Regression (SVR)

**ε-insensitive Loss:**
```
L_ε(y, f(x)) = max(0, |y - f(x)| - ε)
```

**Optimization Problem:**
```
min (1/2)||w||₂² + C Σᵢ₌₁ⁿ (ξᵢ + ξᵢ*)
subject to yᵢ - w^T xᵢ - b ≤ ε + ξᵢ
          w^T xᵢ + b - yᵢ ≤ ε + ξᵢ*
          ξᵢ, ξᵢ* ≥ 0
```

**Dual Formulation:**
```
max -ε Σᵢ₌₁ⁿ (αᵢ + αᵢ*) + Σᵢ₌₁ⁿ yᵢ(αᵢ - αᵢ*) - (1/2) Σᵢ,ⱼ (αᵢ - αᵢ*)(αⱼ - αⱼ*)K(xᵢ, xⱼ)
subject to Σᵢ₌₁ⁿ (αᵢ - αᵢ*) = 0, 0 ≤ αᵢ, αᵢ* ≤ C
```

**Prediction Function:**
```
f(x) = Σᵢ₌₁ⁿ (αᵢ - αᵢ*)K(xᵢ, x) + b
```

### Optimization Algorithms

**Sequential Minimal Optimization (SMO):**
Decomposes the QP problem into smallest possible sub-problems involving two Lagrange multipliers.

**SMO Update Rules:**
For two variables αᵢ, αⱼ:
```
αᵢ^new = αᵢ^old + yᵢyⱼ(αⱼ^old - αⱼ^new)
```

**Working Set Selection:**
Choose violating pair based on KKT conditions.

### Model Selection

**Hyperparameter Grid Search:**
- **C**: Regularization parameter (soft margin trade-off)
- **γ**: Kernel parameter (for RBF kernel)
- **ε**: Tube width (for SVR)

**Cross-Validation:**
k-fold CV to estimate generalization performance:
```
CV_error = (1/k) Σᵢ₌₁ᵏ Error(Test_fold_i)
```

### Computational Complexity

**Training:**
- **SMO**: O(n²) to O(n³) depending on sparsity
- **Working Set Methods**: O(n²·⁵) typically

**Prediction:**
- **Dense**: O(n_sv · d) where n_sv is number of support vectors
- **Sparse**: O(n_sv · d_eff) where d_eff is effective dimensionality

**Memory:**
- **Kernel Matrix**: O(n²) (can be prohibitive for large n)
- **Support Vectors**: O(n_sv · d)

### Probabilistic Outputs

**Platt Scaling:**
Map SVM outputs to probabilities using sigmoid:
```
P(y = 1|x) = 1 / (1 + exp(Af(x) + B))
```

Where A and B are fitted using cross-validation.

### Feature Scaling

**Standardization:**
```
x_scaled = (x - μ) / σ
```

**Min-Max Scaling:**
```
x_scaled = (x - min) / (max - min)
```

Essential for SVM performance, especially with RBF kernels.

## Advantages

### Theoretical Foundation
- **Maximum Margin Principle**: Provides good generalization bounds
- **Global Optimum**: Convex optimization problem has unique solution
- **Kernel Trick**: Can handle non-linear problems elegantly
- **Sparsity**: Solution depends only on support vectors

### Practical Benefits
- **High-Dimensional Data**: Effective when features >> samples
- **Memory Efficiency**: Only stores support vectors
- **Versatile**: Works for classification and regression
- **Robust**: Less prone to overfitting in high dimensions

## Limitations

### Computational Issues
- **Scalability**: Poor scaling with large datasets (O(n²) or O(n³))
- **Memory Requirements**: Kernel matrix can be prohibitive
- **Parameter Sensitivity**: Performance heavily depends on C and kernel parameters
- **No Probabilistic Output**: Requires additional calibration

### Practical Considerations
- **Feature Scaling**: Requires careful preprocessing
- **Kernel Selection**: Choice of kernel significantly affects performance
- **Interpretability**: Limited interpretability, especially with non-linear kernels
- **Noise Sensitivity**: Sensitive to outliers and noise

## Use Cases

### Ideal Scenarios
- **High-Dimensional Data**: Text classification, gene expression analysis
- **Small to Medium Datasets**: When n < 100,000 approximately
- **Non-Linear Patterns**: With appropriate kernel selection
- **Binary Classification**: Natural formulation for binary problems

### Applications
- **Text Mining**: Document classification, spam detection
- **Bioinformatics**: Gene classification, protein analysis
- **Image Recognition**: Face recognition, object detection
- **Finance**: Credit scoring, algorithmic trading

### When NOT to Use
- **Very Large Datasets**: Consider linear SVM or other algorithms
- **Real-Time Requirements**: Prediction time can be prohibitive
- **Interpretability Critical**: Use tree-based methods instead
- **Noisy Data**: Consider robust alternatives

## Implementation Considerations

### Preprocessing Steps
1. **Feature Scaling**: Always standardize features
2. **Missing Values**: Handle appropriately (imputation/removal)
3. **Categorical Variables**: One-hot encoding or ordinal encoding
4. **Outlier Detection**: Consider robust preprocessing

### Hyperparameter Tuning
- **Grid Search**: Systematic exploration of parameter space
- **Random Search**: More efficient for high-dimensional parameter spaces
- **Bayesian Optimization**: Model-based optimization
- **Cross-Validation**: Always use CV for unbiased estimates

### Kernel Selection Guidelines
- **Linear**: When features >> samples, text data
- **RBF**: Default choice, handles non-linear patterns well
- **Polynomial**: Specific polynomial relationships expected
- **Custom**: Domain-specific kernels for specialized applications

### Implementation Tips
- **Start Simple**: Begin with linear SVM, then try RBF
- **Feature Engineering**: Can be more important than algorithm choice
- **Ensemble Methods**: Combine with other algorithms
- **Probabilistic Calibration**: Use Platt scaling for probability estimates

## Advanced Topics

### ν-SVM
Alternative formulation using ν parameter:
```
0 ≤ ν ≤ 1
```
Controls both error fraction and support vector fraction.

### One-Class SVM
For novelty detection and outlier identification:
```
min (1/2)||w||₂² - ρ + (1/νn) Σᵢ₌₁ⁿ ξᵢ
```

### Least Squares SVM
Reformulation with equality constraints:
```
yᵢ(w^T φ(xᵢ) + b) = 1 - eᵢ
```

### Online SVM
Incremental learning algorithms for streaming data.

## Conclusion

Support Vector Machines provide a principled approach to classification and regression with strong theoretical foundations. While computationally intensive for large datasets, they excel in high-dimensional spaces and offer excellent performance with proper parameter tuning and preprocessing.

**Key Takeaway**: SVMs demonstrate the power of optimization theory in machine learning, showing how mathematical elegance (maximum margin principle) translates to practical performance. Understanding SVMs provides crucial insights into kernel methods, optimization, and the bias-variance trade-off. 