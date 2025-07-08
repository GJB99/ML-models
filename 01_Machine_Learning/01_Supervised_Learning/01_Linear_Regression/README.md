# Linear Regression

Linear regression is a fundamental statistical method that models the relationship between a dependent variable and one or more independent variables using a linear approach. It forms the foundation for many machine learning algorithms and statistical techniques.

## Mathematical Framework

### Simple Linear Regression
For a single independent variable, the linear relationship is modeled as:

```
y = β₀ + β₁x + ε
```

Where:
- **y** is the dependent variable (target)
- **x** is the independent variable (feature)
- **β₀** is the y-intercept (bias term)
- **β₁** is the slope (weight/coefficient)
- **ε** is the error term (residual)

### Multiple Linear Regression
For multiple features, the model extends to:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

**Matrix Notation:**
```
Y = Xβ + ε
```

Where:
- **Y** ∈ ℝⁿ is the target vector
- **X** ∈ ℝⁿˣ⁽ᵖ⁺¹⁾ is the design matrix (including intercept column)
- **β** ∈ ℝᵖ⁺¹ is the parameter vector
- **ε** ∈ ℝⁿ is the error vector

### Cost Function (Mean Squared Error)
The objective is to minimize the sum of squared residuals:

```
J(β) = (1/2n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Matrix Form:**
```
J(β) = (1/2n) ||Y - Xβ||²₂ = (1/2n)(Y - Xβ)ᵀ(Y - Xβ)
```

### Normal Equation (Closed-Form Solution)
The optimal parameters can be found analytically:

```
β̂ = (XᵀX)⁻¹XᵀY
```

**Derivation:**
Taking the gradient with respect to β and setting to zero:
```
∇β J(β) = (1/n)Xᵀ(Xβ - Y) = 0
```

Solving for β:
```
XᵀXβ = XᵀY
β = (XᵀX)⁻¹XᵀY
```

### Gradient Descent Optimization
When the normal equation is computationally expensive, use iterative optimization:

**Gradient:**
```
∇β J(β) = (1/n)Xᵀ(Xβ - Y)
```

**Update Rule:**
```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - α∇β J(β⁽ᵗ⁾)
```

Where α is the learning rate.

**Stochastic Gradient Descent (SGD):**
```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - α(xᵢᵀβ⁽ᵗ⁾ - yᵢ)xᵢ
```

### Assumptions of Linear Regression

1. **Linearity**: The relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
   ```
   Var(ε|X) = σ²I
   ```
4. **Normality**: Errors are normally distributed
   ```
   ε ~ N(0, σ²I)
   ```
5. **No Multicollinearity**: Features are not perfectly correlated

### Statistical Properties

**Unbiasedness:**
```
E[β̂] = β
```

**Variance:**
```
Var(β̂) = σ²(XᵀX)⁻¹
```

**Gauss-Markov Theorem:**
Under the classical assumptions, OLS estimators are BLUE (Best Linear Unbiased Estimators).

### Confidence Intervals
For parameter βⱼ:
```
β̂ⱼ ± t_{α/2,n-p-1} · SE(β̂ⱼ)
```

Where:
```
SE(β̂ⱼ) = √(σ̂² · [(XᵀX)⁻¹]ⱼⱼ)
```

### Hypothesis Testing
**Null Hypothesis:** H₀: βⱼ = 0
**Test Statistic:**
```
t = β̂ⱼ / SE(β̂ⱼ) ~ t_{n-p-1}
```

### Model Evaluation Metrics

**R-squared (Coefficient of Determination):**
```
R² = 1 - (SSres/SStot) = 1 - (Σ(yᵢ - ŷᵢ)²)/(Σ(yᵢ - ȳ)²)
```

**Adjusted R-squared:**
```
R²adj = 1 - (1 - R²) · (n-1)/(n-p-1)
```

**Mean Squared Error:**
```
MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Root Mean Squared Error:**
```
RMSE = √MSE
```

**Mean Absolute Error:**
```
MAE = (1/n) Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

### Feature Scaling
When features have different scales, normalization improves convergence:

**Standardization (Z-score):**
```
x_scaled = (x - μ)/σ
```

**Min-Max Scaling:**
```
x_scaled = (x - min(x))/(max(x) - min(x))
```

### Polynomial Regression
Extending linear regression to capture non-linear relationships:

```
y = β₀ + β₁x + β₂x² + ... + βₐxᵈ + ε
```

Still linear in parameters, so same techniques apply.

### Prediction and Inference

**Point Prediction:**
```
ŷ = Xβ̂
```

**Prediction Interval:**
```
ŷ ± t_{α/2,n-p-1} · √(σ̂²(1 + xᵀ(XᵀX)⁻¹x))
```

### Residual Analysis

**Residuals:**
```
eᵢ = yᵢ - ŷᵢ
```

**Standardized Residuals:**
```
rᵢ = eᵢ/√(σ̂²(1 - hᵢᵢ))
```

Where hᵢᵢ is the leverage (diagonal element of hat matrix H = X(XᵀX)⁻¹Xᵀ).

### Outlier Detection

**Cook's Distance:**
```
Dᵢ = (rᵢ²/p) · (hᵢᵢ/(1-hᵢᵢ))
```

**Leverage:**
```
hᵢᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ
```

### Computational Complexity

- **Normal Equation**: O(p³ + p²n) where p is features, n is samples
- **Gradient Descent**: O(kpn) where k is iterations
- **Memory**: O(p²) for normal equation, O(p) for gradient descent

## Advantages

### Simplicity and Interpretability
- **Clear Relationship**: Direct interpretation of coefficients
- **Statistical Inference**: Well-established hypothesis testing framework
- **Fast Computation**: Analytical solution available
- **Baseline Model**: Excellent starting point for analysis

### Mathematical Properties
- **Unbiased Estimates**: Under classical assumptions
- **Minimum Variance**: Among linear unbiased estimators
- **Normal Distribution**: Asymptotically normal parameter estimates
- **Confidence Intervals**: Exact confidence intervals available

## Limitations

### Assumptions
- **Linearity**: Relationship must be approximately linear
- **Homoscedasticity**: Constant error variance required
- **Independence**: Observations must be independent
- **Normality**: For exact inference, errors should be normal

### Practical Issues
- **Overfitting**: With many features relative to samples
- **Multicollinearity**: Unstable estimates with correlated features
- **Outliers**: Sensitive to extreme values
- **Feature Scaling**: Performance affected by feature scales

## Use Cases

### Ideal Scenarios
- **Linear Relationships**: When true relationship is approximately linear
- **Statistical Inference**: When understanding significance is important
- **Baseline Modeling**: As starting point for analysis
- **Small Datasets**: When sample size is limited

### Applications
- **Economics**: Demand forecasting, price modeling
- **Finance**: Risk assessment, return prediction
- **Healthcare**: Dose-response relationships
- **Engineering**: Calibration models, quality control

## Implementation Considerations

### When to Use Normal Equation
- Small to medium datasets (p < 10,000)
- When exact solution is needed
- Sufficient memory available

### When to Use Gradient Descent
- Large datasets (n > 100,000)
- Many features (p > 10,000)
- Online learning scenarios
- Memory constraints

### Preprocessing Steps
1. **Handle Missing Values**: Imputation or removal
2. **Feature Scaling**: Standardization or normalization
3. **Outlier Treatment**: Detection and handling
4. **Feature Engineering**: Polynomial terms, interactions

### Regularization Extensions
When overfitting occurs, consider:
- **Ridge Regression**: L2 penalty
- **Lasso Regression**: L1 penalty
- **Elastic Net**: Combined L1 and L2 penalties

## Conclusion

Linear regression provides a fundamental framework for understanding relationships between variables. While simple, it offers powerful insights through its mathematical foundation and serves as the building block for more complex machine learning algorithms. Its interpretability and statistical properties make it invaluable for both prediction and inference tasks.

**Key Takeaway**: Master linear regression thoroughly as it forms the mathematical foundation for understanding more complex machine learning algorithms and provides essential insights into the bias-variance tradeoff, overfitting, and statistical inference. 