# Gradient Boosting Machines (GBMs)

Gradient Boosting is a powerful machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. Developed by Jerome Friedman in 1999, it builds the model in a stage-wise fashion and generalizes other boosting methods by allowing optimization of an arbitrary differentiable loss function through gradient descent in function space.

## Mathematical Framework

### Gradient Boosting Algorithm

**Core Principle:**
Gradient Boosting builds an additive model by sequentially fitting weak learners to the negative gradients of the loss function.

**Additive Model:**
```
F_M(x) = ∑_{m=0}^M γ_m h_m(x)
```

Where:
- **F_M(x)** is the final ensemble model after M iterations
- **h_m(x)** is the m-th weak learner (typically a decision tree)
- **γ_m** is the step size (learning rate) for the m-th iteration
- **M** is the total number of boosting rounds

### Algorithm Steps

**1. Initialize Model:**
```
F_0(x) = argmin_γ ∑_{i=1}^n L(y_i, γ)
```

**2. For m = 1 to M:**

**a) Compute Negative Gradients (Pseudo-residuals):**
```
r_{i,m} = -∂L(y_i, F_{m-1}(x_i))/∂F_{m-1}(x_i)
```

**b) Fit Weak Learner:**
```
h_m = argmin_h ∑_{i=1}^n (r_{i,m} - h(x_i))²
```

**c) Compute Optimal Step Size:**
```
γ_m = argmin_γ ∑_{i=1}^n L(y_i, F_{m-1}(x_i) + γh_m(x_i))
```

**d) Update Model:**
```
F_m(x) = F_{m-1}(x) + γ_m h_m(x)
```

## Loss Functions

### Regression Loss Functions

**Mean Squared Error (MSE):**
```
L(y, F(x)) = (1/2)(y - F(x))²
```

**Gradient:**
```
∂L/∂F = -(y - F(x)) = -residual
```

**Mean Absolute Error (MAE):**
```
L(y, F(x)) = |y - F(x)|
```

**Gradient:**
```
∂L/∂F = -sign(y - F(x))
```

**Huber Loss (δ-robust):**
```
L(y, F(x)) = {
    (1/2)(y - F(x))²           if |y - F(x)| ≤ δ
    δ|y - F(x)| - (1/2)δ²     otherwise
}
```

### Classification Loss Functions

**Binomial Deviance (Log-Loss):**
```
L(y, F(x)) = log(1 + exp(-2yF(x)))
```

Where y ∈ {-1, +1}.

**Gradient:**
```
∂L/∂F = -2y / (1 + exp(2yF(x)))
```

**Multinomial Deviance:**
For K classes with probability estimates p_k(x):
```
L(y, F(x)) = -∑_{k=1}^K y_k log(p_k(x))
```

Where F(x) = (F_1(x), ..., F_K(x)) and:
```
p_k(x) = exp(F_k(x)) / ∑_{j=1}^K exp(F_j(x))
```

## Tree-Specific Implementation

### Terminal Node Optimization

For regression trees with J terminal nodes:

**Node Predictions:**
```
γ_{jm} = argmin_γ ∑_{x_i ∈ R_{jm}} L(y_i, F_{m-1}(x_i) + γ)
```

Where R_{jm} is the j-th terminal node region of tree m.

**MSE Solution:**
```
γ_{jm} = (1/|R_{jm}|) ∑_{x_i ∈ R_{jm}} r_{i,m}
```

**MAE Solution:**
```
γ_{jm} = median{r_{i,m} : x_i ∈ R_{jm}}
```

### Tree Growing Strategy

**Greedy Split Selection:**
At each node, find the best split that maximizes:
```
Gain = ∑_{i ∈ left} (r_{i,m})² / |left| + ∑_{i ∈ right} (r_{i,m})² / |right| - ∑_{i ∈ node} (r_{i,m})² / |node|
```

## Regularization Techniques

### Learning Rate (Shrinkage)

**Modified Update Rule:**
```
F_m(x) = F_{m-1}(x) + ν · γ_m h_m(x)
```

Where ν ∈ (0, 1] is the learning rate.

**Bias-Variance Trade-off:**
- Smaller ν: Lower variance, higher bias, requires more iterations
- Larger ν: Higher variance, lower bias, faster convergence

### Subsampling

**Stochastic Gradient Boosting:**
At each iteration, train on a random subsample:
```
Sample size = η × n, where η ∈ (0, 1]
```

**Benefits:**
- Reduces overfitting
- Improves computational efficiency
- Increases robustness

### Tree Constraints

**Maximum Depth:**
```
depth ≤ d_max (typically 3-8)
```

**Minimum Samples per Leaf:**
```
|R_j| ≥ min_samples_leaf
```

**Maximum Number of Leaf Nodes:**
```
J ≤ J_max
```

## Advanced Techniques

### Newton-Raphson Boosting

**Second-Order Approximation:**
Use both first and second derivatives:
```
r_{i,m} = -g_i / h_i
```

Where:
- **g_i = ∂L/∂F** (first derivative)
- **h_i = ∂²L/∂F²** (second derivative)

**Optimal Step Size:**
```
γ_{jm} = -∑_{x_i ∈ R_{jm}} g_i / ∑_{x_i ∈ R_{jm}} h_i
```

### Feature Importance

**Split-based Importance:**
```
I_j = ∑_{m=1}^M ∑_{t ∈ T_m} I_j^(t) · p(t) · |Δ_t|
```

Where:
- **I_j^(t)** = 1 if node t splits on feature j, 0 otherwise
- **p(t)** = proportion of samples reaching node t
- **|Δ_t|** = impurity reduction at node t

**Permutation Importance:**
```
PI_j = ∑_{i=1}^n L(y_i, F(x_i^{perm_j})) - ∑_{i=1}^n L(y_i, F(x_i))
```

## Convergence Analysis

### Functional Gradient Descent

**Function Space Optimization:**
Gradient boosting performs gradient descent in function space:
```
F_{m+1} = F_m - ν ∇_F ∑_{i=1}^n L(y_i, F(x_i))
```

**Convergence Rate:**
Under certain conditions:
```
E[L(F_m)] - L(F*) ≤ (1 - ν)^m (E[L(F_0)] - L(F*))
```

### Overfitting Control

**Early Stopping:**
Monitor validation loss:
```
Stop when: L_val(F_m) > L_val(F_{m-k}) for k consecutive rounds
```

**Optimal Number of Trees:**
```
M* = argmin_M L_val(F_M)
```

## Computational Complexity

### Training Complexity

**Per Iteration:**
```
O(n × d × log(n) + n × J)
```

Where:
- **n**: number of samples
- **d**: number of features
- **J**: number of terminal nodes per tree

**Total Training:**
```
O(M × n × d × log(n))
```

### Memory Complexity

**Model Storage:**
```
O(M × J × log(d))
```

**Training Memory:**
```
O(n + M × J)
```

## Statistical Properties

### Consistency

**Universal Approximation:**
Gradient boosting with sufficient trees can approximate any continuous function arbitrarily well.

**Consistency Theorem:**
Under regularity conditions:
```
lim_{n→∞} E[L(F_n)] = L(F*)
```

### Generalization Bounds

**Rademacher Complexity Bound:**
With probability ≥ 1-δ:
```
L(F_M) ≤ L̂(F_M) + 2R_n(H_M) + √(log(1/δ)/(2n))
```

Where R_n(H_M) is the Rademacher complexity of the function class.

## Implementation Variants

This section covers several of the most popular and powerful Gradient Boosting implementations:
-   [**XGBoost**](./01_XGBoost/): The standard for performance in tabular competitions.
-   [**CatBoost**](./02_CatBoost/): Excellent handling of categorical features.
-   [**LightGBM**](./03_LightGBM/): The fastest implementation.
-   [**FastTree**](./04_FastTree/): An optimized implementation for the .NET ecosystem.
-   [**H2O GBM**](./05_H2O_GBM/): A highly scalable implementation for distributed environments.

### Key Concept: Boosting

Boosting trains models sequentially, with each new model focusing on correcting the errors made by its predecessors. This is achieved through several key mechanisms:

**Sequential Learning:** Models are built one after another, each one learning from the previous one's mistakes through gradient computation.

**Adaptive Weighting:** In gradient boosting, difficult examples automatically receive more attention through larger gradient magnitudes.

**Gradient Descent Optimization:** New models are fit to the negative gradients of the loss function, effectively performing gradient descent in function space.

## Practical Considerations

### Hyperparameter Tuning

**Critical Parameters:**
1. **Number of Trees (M)**: Use early stopping
2. **Learning Rate (ν)**: 0.01-0.3 typical range
3. **Tree Depth**: 3-8 for most problems
4. **Subsampling Rate**: 0.5-1.0

**Tuning Strategy:**
```
1. Fix learning_rate = 0.1
2. Tune tree parameters (depth, min_samples)
3. Tune regularization (subsampling)
4. Lower learning_rate and increase n_estimators
```

### When to Use Gradient Boosting

**Optimal Scenarios:**
- **Tabular Data**: Excellent performance on structured data
- **Competition Settings**: Often wins ML competitions
- **Feature Importance**: When interpretability is needed
- **Mixed Data Types**: Handles numerical and categorical features
- **Non-linear Patterns**: Captures complex interactions

**Avoid When:**
- **High-Dimensional Sparse Data**: May overfit
- **Very Small Datasets**: Risk of overfitting
- **Real-time Prediction**: Can be slow for inference
- **Simple Linear Relationships**: Overkill for linear problems

## Mathematical Summary

Gradient Boosting achieves superior performance through:

1. **Functional Gradient Descent**: Optimizing in function space rather than parameter space
2. **Sequential Error Correction**: Each model corrects errors of previous models
3. **Adaptive Learning**: Automatic focus on difficult examples through gradients
4. **Flexible Loss Functions**: Can optimize any differentiable loss function

The algorithm's power comes from the mathematical insight that minimizing a loss function can be viewed as a gradient descent problem in the space of functions, where each weak learner represents a step in the steepest descent direction.

**Key Takeaway**: Gradient Boosting transforms the complex problem of function optimization into a sequence of simple regression problems (fitting to gradients), making it both mathematically elegant and computationally tractable. This principle underlies all modern boosting implementations and explains their exceptional performance on tabular data.

This systematic focus on misclassified examples through gradient-based learning is what allows boosting algorithms to build highly accurate "strong learners" from simple "weak learners" (typically decision trees). 