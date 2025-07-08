# Decision Trees

Decision trees are non-parametric supervised learning algorithms that create a model to predict target values by learning simple decision rules inferred from data features. They create a tree-like model of decisions through recursive binary splitting, making them highly interpretable and effective for both classification and regression tasks.

## Mathematical Framework

### Basic Tree Structure

**Node Representation:**
A decision tree consists of:
- **Root Node**: Starting point containing all data
- **Internal Nodes**: Decision nodes with splitting conditions
- **Leaf Nodes**: Terminal nodes containing predictions

**Splitting Rule:**
At each internal node, data is split based on feature threshold:
```
if x_j ≤ threshold then go_left else go_right
```

### Information Theory Foundations

**Entropy (Classification):**
For a dataset S with classes C = {c₁, c₂, ..., cₖ}:
```
Entropy(S) = -Σᵢ₌₁ᵏ pᵢ log₂(pᵢ)
```

Where pᵢ = |Sᵢ|/|S| is the proportion of examples in class cᵢ.

**Information Gain:**
Reduction in entropy after splitting on feature A:
```
IG(S, A) = Entropy(S) - Σᵥ∈Values(A) (|Sᵥ|/|S|) × Entropy(Sᵥ)
```

**Gain Ratio:**
Normalized information gain to handle bias toward multi-valued attributes:
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
```

Where:
```
SplitInfo(S, A) = -Σᵥ∈Values(A) (|Sᵥ|/|S|) × log₂(|Sᵥ|/|S|)
```

### Impurity Measures

**Gini Impurity:**
```
Gini(S) = 1 - Σᵢ₌₁ᵏ pᵢ²
```

**Gini Gain:**
```
GiniGain(S, A) = Gini(S) - Σᵥ∈Values(A) (|Sᵥ|/|S|) × Gini(Sᵥ)
```

**Misclassification Error:**
```
Error(S) = 1 - max(pᵢ)
```

### Splitting Criteria Comparison

**For Binary Classification:**
- **Entropy**: More sensitive to changes in node probabilities
- **Gini**: Computationally efficient, tends to isolate frequent classes
- **Error**: Less sensitive, often used for pruning

**Mathematical Relationship:**
For binary classification with probability p for class 1:
```
Entropy = -p log₂(p) - (1-p) log₂(1-p)
Gini = 2p(1-p)
Error = 1 - max(p, 1-p)
```

### Regression Trees

**Variance Reduction:**
For regression, minimize sum of squared errors:
```
Variance(S) = (1/|S|) Σᵢ₌₁|S| (yᵢ - ȳ)²
```

**Sum of Squared Errors (SSE):**
```
SSE(S) = Σᵢ₌₁|S| (yᵢ - ȳ)²
```

**Variance Gain:**
```
VarGain(S, A) = Variance(S) - Σᵥ∈Values(A) (|Sᵥ|/|S|) × Variance(Sᵥ)
```

**Prediction at Leaf:**
```
ŷ = (1/|S_leaf|) Σᵢ∈S_leaf yᵢ
```

### Tree Construction Algorithm (ID3/C4.5/CART)

**Recursive Splitting:**
```
function BuildTree(S, Attributes):
    if S is pure or stopping criteria met:
        return LeafNode(majority_class(S))
    
    best_attribute = argmax_{A ∈ Attributes} IG(S, A)
    tree = DecisionNode(best_attribute)
    
    for each value v of best_attribute:
        S_v = subset of S where best_attribute = v
        subtree = BuildTree(S_v, Attributes - {best_attribute})
        tree.add_branch(v, subtree)
    
    return tree
```

### Optimal Split Finding

**For Continuous Features:**
Sort feature values and consider split points:
```
Split_points = {(xᵢ + xᵢ₊₁)/2 : i = 1, ..., n-1}
```

**Binary Split Optimization:**
```
threshold* = argmax_{t} IG(S, x_j ≤ t)
```

**Multi-way vs Binary Splits:**
- **Binary**: Always creates two children
- **Multi-way**: Creates child for each unique value

### Pruning Techniques

**Pre-pruning (Early Stopping):**
Stop splitting based on:
- Minimum samples per leaf: |S_leaf| ≥ min_samples
- Maximum depth: depth ≤ max_depth
- Minimum information gain: IG ≥ min_gain
- Statistical significance: χ² test for independence

**Post-pruning:**

**Reduced Error Pruning:**
```
if accuracy(pruned_tree) ≥ accuracy(original_tree):
    prune subtree
```

**Cost-Complexity Pruning (Minimal Cost-Complexity):**
Define cost-complexity measure:
```
R_α(T) = R(T) + α|leaves(T)|
```

Where:
- R(T) is misclassification cost
- α is complexity parameter
- |leaves(T)| is number of leaves

**Pruning Sequence:**
Find sequence of nested trees T₀ ⊃ T₁ ⊃ ... ⊃ {root} by increasing α.

### Handling Missing Values

**C4.5 Approach:**
Distribute samples proportionally:
```
weight(S_v) = (|S_v| / |S_known|) × |S_unknown|
```

**Surrogate Splits:**
Find backup splits highly correlated with primary split:
```
correlation(split1, split2) = agreement_rate(split1, split2)
```

### Feature Importance

**Impurity-based Importance:**
```
Importance(feature_j) = Σ_{nodes using j} (N_node / N_total) × impurity_decrease
```

**Permutation Importance:**
```
Importance(feature_j) = score_original - score_permuted_j
```

### Ensemble Extensions

**Bagging Preparation:**
Each tree uses:
- Bootstrap sample of data
- Random subset of features at each split

**Random Forest Integration:**
```
prediction = (1/B) Σᵦ₌₁ᴮ tree_b(x)
```

### Probability Estimates

**Class Probabilities:**
At leaf node with samples S_leaf:
```
P(class = c | x) = |{i ∈ S_leaf : yᵢ = c}| / |S_leaf|
```

**Laplace Smoothing:**
```
P(class = c | x) = (count(c) + 1) / (|S_leaf| + num_classes)
```

### Tree Complexity Measures

**Tree Size:**
- Number of nodes: |nodes(T)|
- Number of leaves: |leaves(T)|
- Tree depth: max_depth(T)

**Description Length:**
```
DL(T) = encoding_cost(tree_structure) + encoding_cost(data|tree)
```

### Decision Boundaries

**Axis-parallel Splits:**
Standard decision trees create rectangular decision regions:
```
Region = {x : x_j₁ ≤ t₁ ∧ x_j₂ > t₂ ∧ ... ∧ x_jₖ ≤ tₖ}
```

**Oblique Trees:**
Linear combinations of features:
```
Σⱼ wⱼxⱼ ≤ threshold
```

### Computational Complexity

**Training:**
- Best case: O(n log n × d) for balanced splits
- Worst case: O(n² × d) for unbalanced data
- Average: O(n log n × d × log n)

**Prediction:**
- O(log n) for balanced trees
- O(n) for degenerate trees (linear chains)

**Memory:**
- O(n) for tree storage
- O(d) for each internal node

### Statistical Tests for Splitting

**Chi-square Test:**
Test independence between split and target:
```
χ² = Σᵢ Σⱼ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ
```

**G-test (Likelihood Ratio):**
```
G = 2 Σᵢ Σⱼ Oᵢⱼ ln(Oᵢⱼ / Eᵢⱼ)
```

### Categorical Feature Handling

**Nominal Features:**
Consider all possible binary partitions:
- 2ᵏ⁻¹ - 1 possible splits for k categories

**Ordinal Features:**
Maintain natural ordering:
- Only k-1 possible splits for k ordered categories

### Incremental Tree Construction

**Online Learning:**
Update tree structure as new data arrives:
```
if new_data changes best_split:
    restructure affected subtree
```

**Hoeffding Trees:**
Use Hoeffding bound to determine split confidence:
```
ε = √(R² ln(1/δ) / 2n)
```

Where R is range of splitting metric, δ is confidence parameter.

## Advantages

### Interpretability
- **White Box Model**: Easy to understand and visualize
- **Feature Importance**: Clear ranking of feature relevance
- **Rule Extraction**: Can extract human-readable rules
- **No Black Box**: Complete transparency in decision process

### Flexibility
- **No Assumptions**: Non-parametric, no distributional assumptions
- **Mixed Data Types**: Handles numerical and categorical features naturally
- **Missing Values**: Can handle missing data gracefully
- **Non-linear Relationships**: Captures complex interactions automatically

### Computational Efficiency
- **Fast Training**: Efficient recursive splitting algorithm
- **Fast Prediction**: O(log n) prediction time for balanced trees
- **Low Memory**: Compact representation for deployment
- **Parallelizable**: Tree construction can be parallelized

## Limitations

### Instability
- **High Variance**: Small changes in data can drastically change tree structure
- **Overfitting**: Tendency to create overly complex trees
- **Bias**: Greedy algorithm may not find globally optimal tree
- **Sensitive to Noise**: Outliers can significantly affect splits

### Representational Limitations
- **Linear Boundaries**: Cannot naturally represent diagonal decision boundaries
- **Smooth Functions**: Poor approximation of smooth continuous functions
- **Balanced Classes**: Bias toward features with more levels
- **Extrapolation**: Cannot extrapolate beyond training data range

### Statistical Issues
- **Multiple Testing**: No correction for multiple hypothesis testing
- **Selection Bias**: Bias toward features with many possible splits
- **Stopping Criteria**: Difficult to determine optimal tree size
- **Confidence**: Limited statistical inference capabilities

## Use Cases

### Ideal Scenarios
- **Interpretability Required**: When model explainability is crucial
- **Mixed Data Types**: Datasets with numerical and categorical features
- **Rule-based Logic**: When decision rules are desired output
- **Feature Selection**: When identifying important features

### Applications
- **Medical Diagnosis**: Symptom-based diagnostic systems
- **Credit Scoring**: Loan approval decision rules
- **Customer Segmentation**: Marketing campaign targeting
- **Quality Control**: Manufacturing defect classification

### When NOT to Use
- **High-Dimensional Data**: Performance degrades with many irrelevant features
- **Smooth Relationships**: When relationships are continuously smooth
- **Small Datasets**: Prone to overfitting with limited data
- **Noisy Data**: Sensitive to label noise and measurement errors

## Implementation Considerations

### Hyperparameter Tuning
- **Max Depth**: Limit tree depth to prevent overfitting
- **Min Samples Split**: Minimum samples required to split node
- **Min Samples Leaf**: Minimum samples required at leaf
- **Max Features**: Random subset of features to consider

### Preprocessing
- **Missing Values**: Imputation or native handling
- **Categorical Encoding**: Ordinal vs nominal treatment
- **Feature Scaling**: Not required but may help with interpretability
- **Outlier Detection**: Consider robust splitting criteria

### Model Selection
- **Cross-Validation**: Use CV for hyperparameter tuning
- **Pruning Strategy**: Choose appropriate pruning method
- **Ensemble vs Single**: Consider Random Forest for better performance
- **Interpretability Trade-off**: Balance accuracy vs explainability

### Validation Strategies
- **Holdout Validation**: Simple train/test split
- **K-fold CV**: More robust performance estimation
- **Time Series CV**: For temporal data
- **Nested CV**: For unbiased hyperparameter optimization

## Advanced Techniques

### Alternative Splitting Criteria
- **MDL Principle**: Minimum Description Length
- **Distance-based**: Maximum distance between class distributions
- **Kernel-based**: Using kernel methods for splits

### Ensemble Methods
- **Random Forest**: Bootstrap aggregating with feature randomness
- **Extra Trees**: Extremely randomized trees
- **Gradient Boosting**: Sequential error correction

### Tree Regularization
- **Cost-Complexity**: α-parameterized family of trees
- **Bayesian**: Bayesian priors on tree structure
- **Structural Risk Minimization**: Trade-off between fit and complexity

## Conclusion

Decision trees provide an intuitive and interpretable approach to machine learning with strong performance on many real-world problems. While they suffer from high variance and can overfit, their transparency and ability to handle mixed data types make them invaluable for applications requiring interpretability.

**Key Takeaway**: Decision trees demonstrate the power of greedy, recursive algorithms in machine learning while highlighting the bias-variance trade-off. Understanding trees is essential for comprehending ensemble methods and provides insights into feature importance and model interpretability. 