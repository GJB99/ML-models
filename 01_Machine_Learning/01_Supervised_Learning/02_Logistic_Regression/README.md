# Logistic Regression

Logistic regression is a statistical method used for binary and multiclass classification problems. Unlike linear regression, it models the probability that an instance belongs to a particular class using the logistic function, ensuring outputs are bounded between 0 and 1.

## Mathematical Framework

### Sigmoid Function (Logistic Function)
The core of logistic regression is the sigmoid function:

```
σ(z) = 1 / (1 + e^(-z)) = e^z / (1 + e^z)
```

**Properties:**
- Domain: (-∞, +∞)
- Range: (0, 1)
- S-shaped curve
- σ(0) = 0.5
- σ(-z) = 1 - σ(z)

### Binary Logistic Regression

**Linear Combination:**
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ = β^T x
```

**Probability Model:**
```
P(Y = 1|x) = σ(β^T x) = 1 / (1 + e^(-β^T x))
P(Y = 0|x) = 1 - P(Y = 1|x) = e^(-β^T x) / (1 + e^(-β^T x))
```

**Odds and Log-Odds:**
```
Odds = P(Y=1|x) / P(Y=0|x) = e^(β^T x)
Log-Odds = ln(Odds) = β^T x
```

The log-odds (logit) is linear in the parameters, hence "logistic regression."

### Maximum Likelihood Estimation

**Likelihood Function:**
For n independent observations:
```
L(β) = ∏ᵢ₌₁ⁿ P(yᵢ|xᵢ) = ∏ᵢ₌₁ⁿ [P(Y=1|xᵢ)]^yᵢ [P(Y=0|xᵢ)]^(1-yᵢ)
```

**Log-Likelihood:**
```
ℓ(β) = ln L(β) = Σᵢ₌₁ⁿ [yᵢ ln(pᵢ) + (1-yᵢ) ln(1-pᵢ)]
```

Where pᵢ = P(Y=1|xᵢ) = σ(β^T xᵢ)

**Cost Function (Negative Log-Likelihood):**
```
J(β) = -ℓ(β) = -Σᵢ₌₁ⁿ [yᵢ ln(pᵢ) + (1-yᵢ) ln(1-pᵢ)]
```

### Gradient Computation

**Gradient of Cost Function:**
```
∇β J(β) = Σᵢ₌₁ⁿ (pᵢ - yᵢ)xᵢ = X^T(p - y)
```

Where:
- **p** = [p₁, p₂, ..., pₙ]^T (predicted probabilities)
- **y** = [y₁, y₂, ..., yₙ]^T (actual labels)

**Hessian Matrix:**
```
H = ∇²β J(β) = Σᵢ₌₁ⁿ pᵢ(1-pᵢ)xᵢxᵢ^T = X^T W X
```

Where W = diag(p₁(1-p₁), ..., pₙ(1-pₙ))

### Optimization Algorithms

**Gradient Descent:**
```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - α∇β J(β⁽ᵗ⁾)
```

**Newton-Raphson Method:**
```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - H⁻¹∇β J(β⁽ᵗ⁾)
```

**Iteratively Reweighted Least Squares (IRLS):**
```
β⁽ᵗ⁺¹⁾ = (X^T W⁽ᵗ⁾ X)⁻¹ X^T W⁽ᵗ⁾ z⁽ᵗ⁾
```

Where z⁽ᵗ⁾ = Xβ⁽ᵗ⁾ + (W⁽ᵗ⁾)⁻¹(y - p⁽ᵗ⁾)

### Regularization

**Ridge Logistic Regression (L2):**
```
J(β) = -ℓ(β) + λ Σⱼ₌₁ᵖ βⱼ²
```

**Lasso Logistic Regression (L1):**
```
J(β) = -ℓ(β) + λ Σⱼ₌₁ᵖ |βⱼ|
```

**Elastic Net:**
```
J(β) = -ℓ(β) + λ₁ Σⱼ₌₁ᵖ |βⱼ| + λ₂ Σⱼ₌₁ᵖ βⱼ²
```

### Multiclass Logistic Regression

**One-vs-Rest (OvR):**
Train K binary classifiers for K classes:
```
P(Y = k|x) = σ(β₍ₖ₎^T x) / (1 + Σⱼ₌₁ᴷ σ(β₍ⱼ₎^T x))
```

**Multinomial Logistic Regression (Softmax):**
```
P(Y = k|x) = exp(β₍ₖ₎^T x) / Σⱼ₌₁ᴷ exp(β₍ⱼ₎^T x)
```

**Cross-Entropy Loss:**
```
J(β) = -Σᵢ₌₁ⁿ Σₖ₌₁ᴷ yᵢₖ ln(pᵢₖ)
```

### Statistical Properties

**Asymptotic Normality:**
```
√n(β̂ - β) →ᵈ N(0, I(β)⁻¹)
```

Where I(β) is the Fisher Information Matrix.

**Fisher Information:**
```
I(β) = E[H] = X^T W X
```

**Standard Errors:**
```
SE(β̂ⱼ) = √([X^T W X]⁻¹ⱼⱼ)
```

### Hypothesis Testing

**Wald Test:**
```
W = (β̂ⱼ / SE(β̂ⱼ))² ~ χ²₁
```

**Likelihood Ratio Test:**
```
LR = 2(ℓ(β̂) - ℓ(β₀)) ~ χ²ᵨ
```

**Score Test:**
```
S = (∇ℓ(β₀))^T I(β₀)⁻¹ ∇ℓ(β₀) ~ χ²ᵨ
```

### Model Evaluation

**Deviance:**
```
D = -2ℓ(β̂) = -2 Σᵢ₌₁ⁿ [yᵢ ln(p̂ᵢ) + (1-yᵢ) ln(1-p̂ᵢ)]
```

**AIC (Akaike Information Criterion):**
```
AIC = 2p - 2ℓ(β̂)
```

**BIC (Bayesian Information Criterion):**
```
BIC = p ln(n) - 2ℓ(β̂)
```

### Classification Metrics

**Confusion Matrix Elements:**
- True Positives (TP), False Positives (FP)
- True Negatives (TN), False Negatives (FN)

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**
```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```

**Specificity:**
```
Specificity = TN / (TN + FP)
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC Curve and AUC

**ROC Curve:** Plot of True Positive Rate vs False Positive Rate
```
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

**AUC (Area Under Curve):**
```
AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
```

### Probability Calibration

**Platt Scaling:**
Apply sigmoid to logistic regression outputs:
```
P_calibrated = 1 / (1 + exp(a × f(x) + b))
```

Where f(x) is the uncalibrated score.

### Decision Boundary

**Binary Classification:**
Decision boundary occurs when P(Y=1|x) = 0.5:
```
β^T x = 0
```

**Distance from Point to Decision Boundary:**
```
distance = |β^T x| / ||β||₂
```

### Confidence Intervals

**For Coefficients:**
```
β̂ⱼ ± z_{α/2} × SE(β̂ⱼ)
```

**For Predictions (Linear Predictor):**
```
β̂^T x ± z_{α/2} × √(x^T (X^T W X)⁻¹ x)
```

### Odds Ratios

**Interpretation of Coefficients:**
```
OR = exp(βⱼ)
```

An increase of one unit in xⱼ multiplies the odds by exp(βⱼ).

### Assumptions

1. **Independence**: Observations are independent
2. **Linearity**: Log-odds are linear in parameters
3. **No Perfect Multicollinearity**: Features are not perfectly correlated
4. **Large Sample Size**: For asymptotic properties to hold

### Computational Complexity

- **Training**: O(kpn) where k is iterations, p is features, n is samples
- **Prediction**: O(p) per sample
- **Memory**: O(p²) for Hessian computation

## Advantages

### Probabilistic Output
- **Probability Estimates**: Natural probability interpretation
- **Uncertainty Quantification**: Confidence in predictions
- **Calibrated Outputs**: Well-calibrated probability estimates
- **Threshold Flexibility**: Easy to adjust decision thresholds

### Statistical Properties
- **No Distributional Assumptions**: Only assumes logistic relationship
- **Robust to Outliers**: Less sensitive than linear regression
- **Efficient**: Converges quickly with good initialization
- **Interpretable**: Clear coefficient interpretation via odds ratios

## Limitations

### Model Assumptions
- **Linear Decision Boundary**: Assumes linear relationship in log-odds
- **Feature Independence**: Assumes features contribute independently
- **Sample Size**: Requires sufficient samples for stable estimates
- **Convergence Issues**: May not converge with perfect separation

### Practical Considerations
- **Feature Scaling**: Sensitive to feature scales
- **Multicollinearity**: Unstable with highly correlated features
- **Outliers**: Can influence coefficient estimates
- **Imbalanced Data**: May perform poorly with extreme class imbalance

## Use Cases

### Ideal Scenarios
- **Binary Classification**: Natural choice for binary problems
- **Probability Estimates**: When probability interpretation is needed
- **Medical Diagnosis**: Risk assessment and diagnostic models
- **Marketing**: Customer response prediction

### Applications
- **Healthcare**: Disease diagnosis, treatment response
- **Finance**: Credit approval, fraud detection
- **Marketing**: Click-through rate prediction, customer segmentation
- **Sports**: Win probability, player performance analysis

## Implementation Considerations

### Preprocessing
1. **Feature Scaling**: Standardization recommended
2. **Missing Values**: Handle appropriately (imputation/removal)
3. **Categorical Variables**: One-hot encoding or label encoding
4. **Feature Selection**: Remove irrelevant/redundant features

### Optimization Tips
- **Learning Rate**: Start with α = 0.01-0.1
- **Convergence Criteria**: Monitor log-likelihood change
- **Regularization**: Use when overfitting occurs
- **Initialization**: Random or zero initialization

### Diagnostic Tools
- **Residual Analysis**: Pearson and deviance residuals
- **Influence Measures**: Cook's distance, leverage
- **Goodness of Fit**: Hosmer-Lemeshow test
- **Model Comparison**: AIC, BIC, cross-validation

## Advanced Topics

### Bayesian Logistic Regression
```
P(β|Data) ∝ P(Data|β) × P(β)
```

### Hierarchical Logistic Regression
For grouped data with random effects.

### Robust Logistic Regression
Methods to handle outliers and influential points.

## Conclusion

Logistic regression provides a fundamental framework for classification problems with strong theoretical foundations and practical utility. Its probabilistic interpretation, computational efficiency, and interpretability make it a cornerstone algorithm in machine learning and statistics.

**Key Takeaway**: Logistic regression extends linear regression concepts to classification, providing probabilistic outputs and serving as the foundation for understanding more complex classification algorithms and neural networks. 