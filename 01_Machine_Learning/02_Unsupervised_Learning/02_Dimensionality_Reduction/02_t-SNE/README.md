# t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a powerful nonlinear dimensionality reduction technique specifically designed for visualization of high-dimensional data. Developed by Laurens van der Maaten and Geoffrey Hinton in 2008, t-SNE excels at revealing local structures and clusters in data by preserving neighborhood relationships while mapping to lower dimensions (typically 2D or 3D for visualization).

## Mathematical Framework

### Core Concept

**Objective:**
Given high-dimensional data X = {x₁, x₂, ..., xₙ} where xᵢ ∈ ℝᵈ, find low-dimensional embeddings Y = {y₁, y₂, ..., yₙ} where yᵢ ∈ ℝᵈ' (typically d' = 2 or 3) such that similar points in high-dimensional space remain close in low-dimensional space.

**Key Insight:**
Convert distances to probabilities, then minimize divergence between probability distributions in high-dimensional and low-dimensional spaces.

### High-Dimensional Probability Distribution

**Pairwise Similarities (Conditional Probabilities):**
```
p_{j|i} = exp(-||xᵢ - xⱼ||²/(2σᵢ²)) / ∑_{k≠i} exp(-||xᵢ - xₖ||²/(2σᵢ²))
```

Where σᵢ is the bandwidth parameter for point xᵢ.

**Symmetric Probabilities:**
```
pᵢⱼ = (p_{j|i} + p_{i|j})/(2n)
```

**Properties:**
- pᵢᵢ = 0 (no self-similarity)
- ∑ᵢ ∑ⱼ pᵢⱼ = 1 (valid probability distribution)
- pᵢⱼ = pⱼᵢ (symmetric)

### Perplexity and Bandwidth Selection

**Perplexity Definition:**
```
Perp(Pᵢ) = 2^{H(Pᵢ)}
```

Where H(Pᵢ) is the Shannon entropy of the probability distribution Pᵢ:
```
H(Pᵢ) = -∑ⱼ p_{j|i} log₂ p_{j|i}
```

**Bandwidth Optimization:**
For each point i, find σᵢ such that Perp(Pᵢ) equals a target perplexity value (typically 5-50):
```
∑ⱼ≠ᵢ p_{j|i} log₂ p_{j|i} = -log₂(Perplexity)
```

**Binary Search Algorithm:**
```
While |Perp(current) - Perp(target)| > tolerance:
    if Perp(current) > Perp(target):
        σ_max = σ_current
    else:
        σ_min = σ_current
    σ_current = (σ_min + σ_max) / 2
    Recompute p_{j|i} with new σ_current
```

### Low-Dimensional Probability Distribution

**t-Distribution with One Degree of Freedom:**
```
qᵢⱼ = (1 + ||yᵢ - yⱼ||²)⁻¹ / ∑_{k≠l} (1 + ||yₖ - yₗ||²)⁻¹
```

**Why t-Distribution?**
- **Heavy tails**: Allow dissimilar points to be farther apart
- **Addresses crowding problem**: More space in low dimensions for moderately distant points
- **Student's t with ν=1 degree of freedom**: Reduces to Cauchy distribution

**Alternative Formulation:**
```
qᵢⱼ = ((1 + ||yᵢ - yⱼ||²)⁻¹) / Z
```

Where Z is the normalization constant:
```
Z = ∑_{k≠l} (1 + ||yₖ - yₗ||²)⁻¹
```

### Objective Function

**Kullback-Leibler Divergence:**
```
KL(P||Q) = ∑ᵢ ∑ⱼ≠ᵢ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
```

**Cost Function:**
```
C = KL(P||Q) = ∑ᵢ ∑ⱼ≠ᵢ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
```

**Expanded Form:**
```
C = ∑ᵢ ∑ⱼ≠ᵢ pᵢⱼ log pᵢⱼ - ∑ᵢ ∑ⱼ≠ᵢ pᵢⱼ log qᵢⱼ
```

Since the first term is constant w.r.t. Y, we minimize:
```
C = -∑ᵢ ∑ⱼ≠ᵢ pᵢⱼ log qᵢⱼ
```

## Gradient Computation

### Gradient Derivation

**Gradient of Cost Function:**
```
∂C/∂yᵢ = 4∑ⱼ≠ᵢ (pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹
```

**Derivation Steps:**

1. **Gradient of qᵢⱼ:**
   ```
   ∂qᵢⱼ/∂yᵢ = ∂/∂yᵢ [(1 + ||yᵢ - yⱼ||²)⁻¹ / Z]
   ```

2. **Chain Rule Application:**
   ```
   ∂qᵢⱼ/∂yᵢ = -2(1 + ||yᵢ - yⱼ||²)⁻² (yᵢ - yⱼ) / Z + (1 + ||yᵢ - yⱼ||²)⁻¹ ∂Z⁻¹/∂yᵢ
   ```

3. **Simplification:**
   After algebraic manipulation, the gradient becomes:
   ```
   ∂C/∂yᵢ = 4∑ⱼ (pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹
   ```

### Physical Interpretation

**Attractive Forces:**
When pᵢⱼ > qᵢⱼ (points should be closer):
- Force pulls yᵢ toward yⱼ
- Magnitude: (pᵢⱼ - qᵢⱼ)

**Repulsive Forces:**
When pᵢⱼ < qᵢⱼ (points should be farther):
- Force pushes yᵢ away from yⱼ
- Magnitude: |qᵢⱼ - pᵢⱼ|

**Heavy-Tail Effect:**
The term (1 + ||yᵢ - yⱼ||²)⁻¹ creates heavy tails, allowing:
- Strong repulsion for nearby dissimilar points
- Weak interaction for distant points

## Optimization Algorithm

### Gradient Descent with Momentum

**Update Rule:**
```
Y^{(t+1)} = Y^{(t)} + η^{(t)} ∂C/∂Y + α^{(t)}(Y^{(t)} - Y^{(t-1)})
```

Where:
- **η^{(t)}**: learning rate at iteration t
- **α^{(t)}**: momentum coefficient at iteration t

**Adaptive Learning Rate:**
```
η^{(t)} = {
    η₀                    if t ≤ T_switch
    η₀ × decay_factor     if t > T_switch
}
```

**Momentum Scheduling:**
```
α^{(t)} = {
    α_initial    if t ≤ T_switch  (e.g., 0.5)
    α_final      if t > T_switch  (e.g., 0.8)
}
```

### Early Exaggeration

**Modified Probabilities:**
```
pᵢⱼ^{exag} = α_exag × pᵢⱼ  for t ≤ T_exag
```

Where α_exag > 1 (typically 4-12) for the first T_exag iterations.

**Purpose:**
- Encourage formation of tight clusters early
- Overcome poor local minima
- Create better global structure

**Modified Gradient:**
```
∂C/∂yᵢ = 4∑ⱼ (α_exag × pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹
```

### Algorithm Summary

```
Algorithm: t-SNE
Input: X ∈ ℝ^{n×d}, perplexity, max_iterations
Output: Y ∈ ℝ^{n×d'}

1. Compute pairwise distances in X
2. For each i, binary search for σᵢ to achieve target perplexity
3. Compute symmetric probabilities pᵢⱼ
4. Initialize Y randomly: Y ~ N(0, 10⁻⁴I)
5. For t = 1 to max_iterations:
   a. Compute low-dimensional probabilities qᵢⱼ
   b. Compute gradient ∂C/∂Y
   c. Update Y using gradient descent with momentum
   d. If t = T_exag, turn off early exaggeration
   e. If t = T_switch, reduce learning rate and increase momentum

Return Y
```

## Barnes-Hut Approximation

### Motivation

**Computational Complexity:**
- **Exact t-SNE**: O(n²) per iteration
- **Barnes-Hut t-SNE**: O(n log n) per iteration

**Key Insight:**
Approximate repulsive forces using spatial data structures (quadtrees/octrees).

### Quadtree Construction

**Hierarchical Space Partitioning:**
```
QuadTree Node:
- center_of_mass: weighted center of all points in subtree
- total_mass: sum of all point masses in subtree
- bounding_box: spatial extent of node
- children: up to 4 child nodes (2D) or 8 (3D)
```

**Tree Construction Algorithm:**
```
BuildTree(points, bounding_box):
    if len(points) ≤ 1:
        return LeafNode(points)
    
    subdivisions = partition(bounding_box, 4)  # or 8 for 3D
    children = []
    for subdivision in subdivisions:
        child_points = points_in_region(points, subdivision)
        if child_points:
            children.append(BuildTree(child_points, subdivision))
    
    return InternalNode(children, center_of_mass, total_mass)
```

### Force Approximation

**θ-Criterion:**
For a node with bounding box width w and distance d from query point:
```
if w/d < θ:
    Use node's center of mass for force computation
else:
    Recursively examine children
```

Where θ (typically 0.5) controls accuracy vs. speed trade-off.

**Approximated Repulsive Force:**
```
F_rep(yᵢ) ≈ ∑_nodes mass_node × (yᵢ - center_node) × (1 + ||yᵢ - center_node||²)⁻¹
```

**Total Gradient (Barnes-Hut):**
```
∂C/∂yᵢ = 4[∑ⱼ pᵢⱼ(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹ - F_rep(yᵢ)]
```

### Implementation Details

**Attractive Forces (Exact):**
Still computed exactly since they involve sparse pᵢⱼ matrix.

**Repulsive Forces (Approximated):**
Computed using tree traversal with θ-criterion.

**Complexity Analysis:**
- **Tree Construction**: O(n log n)
- **Force Computation**: O(n log n) average case
- **Total per Iteration**: O(n log n)

## Theoretical Properties

### Convergence Analysis

**Local Minima:**
t-SNE objective is non-convex with many local minima.

**Convergence Guarantee:**
With proper learning rate scheduling:
```
lim_{t→∞} ||∇C(Y^{(t)})|| = 0
```

**Typical Convergence:**
- Fast initial convergence (first 100-250 iterations)
- Slow refinement phase (250-1000+ iterations)

### Crowding Problem

**High-Dimensional Volume:**
Volume of d-dimensional hypersphere grows as:
```
V_d ∝ r^d
```

**Neighbor Preservation:**
In high dimensions, many points are equidistant from a query point, making it impossible to preserve all neighborhoods in low dimensions.

**t-SNE Solution:**
Heavy tails of t-distribution provide more space for moderately dissimilar points:
```
t-distribution: (1 + ||y||²)⁻¹
Gaussian: exp(-||y||²)
```

### Symmetry and Invariance

**Translation Invariance:**
```
C(Y + c1ₙ) = C(Y)  for any constant c
```

**Rotation Invariance:**
```
C(YR) = C(Y)  for any rotation matrix R
```

**Scale Dependence:**
t-SNE is not scale-invariant; results depend on initialization scale.

## Extensions and Variants

### Parametric t-SNE

**Neural Network Mapping:**
Train a neural network fθ to approximate the embedding:
```
yᵢ = fθ(xᵢ)
```

**Joint Optimization:**
```
min_θ KL(P||Q_θ) + λ||θ||²
```

**Out-of-Sample Extension:**
For new point x*, compute y* = fθ(x*).

### Dynamic t-SNE

**Streaming Data:**
Update embeddings as new data arrives:
```
Y^{(t+1)} = (1-α)Y^{(t)} + α × UpdateEmbedding(x_new)
```

**Incremental Learning:**
Modify existing embeddings minimally when adding new points.

### Multi-Scale t-SNE

**Multiple Perplexities:**
```
C_total = ∑ₖ wₖ × KL(P^{(perp_k)}||Q)
```

**Hierarchical Structure:**
Captures both local and global structure simultaneously.

### Symmetric SNE (SNE)

**Original SNE Formulation:**
Uses Gaussian distributions in both high and low dimensions:
```
qᵢⱼ = exp(-||yᵢ - yⱼ||²) / ∑_{k≠l} exp(-||yₖ - yₗ||²)
```

**Crowding Problem:**
SNE suffers more from crowding; t-SNE addresses this with heavy tails.

## Practical Considerations

### Hyperparameter Selection

**Perplexity:**
- **Range**: 5-50 (typical: 20-30)
- **Small perplexity**: Focus on very local structure
- **Large perplexity**: Preserve more global structure
- **Rule of thumb**: 3 × log(n) ≤ perplexity ≤ n/3

**Learning Rate:**
- **Range**: 10-1000 (typical: 100-500)
- **Too small**: Slow convergence
- **Too large**: Unstable dynamics

**Number of Iterations:**
- **Minimum**: 1000 iterations
- **Complex data**: 2000-5000 iterations
- **Monitor**: KL divergence for convergence

### Initialization

**Random Initialization:**
```
Y^{(0)} ~ N(0, σ²I)
```

Where σ = 10⁻⁴ provides good starting point.

**PCA Initialization:**
```
Y^{(0)} = first_2_components(PCA(X)) × scale_factor
```

Can provide better initialization for some datasets.

### Interpretation Guidelines

**Cluster Interpretation:**
- **Tight clusters**: Strong evidence of similarity
- **Distant points**: Likely dissimilar
- **Distances**: Not meaningful beyond nearest neighbors

**What t-SNE Preserves:**
- Local neighborhoods
- Cluster structure
- Relative cluster sizes (approximately)

**What t-SNE May Distort:**
- Global distances
- Density variations
- Outlier positions

## Applications

### Biological Data Analysis

**Single-Cell RNA Sequencing:**
```
X ∈ ℝ^{n×genes}  →  Y ∈ ℝ^{n×2}
```

Visualize cell types and developmental trajectories.

**Protein Structure Analysis:**
Embed protein conformations in low-dimensional space.

### Computer Vision

**Feature Visualization:**
```
X = CNN_features(images)  →  Y ∈ ℝ^{n×2}
```

Visualize learned representations from deep networks.

**Image Similarity:**
Embed image descriptors for similarity visualization.

### Natural Language Processing

**Word Embeddings:**
Visualize word2vec, GloVe, or BERT embeddings.

**Document Clustering:**
Embed TF-IDF or neural text representations.

### Anomaly Detection

**Outlier Visualization:**
Anomalous points often appear isolated in t-SNE plots.

**Quality Control:**
Visual inspection of data quality and preprocessing effects.

## Evaluation Metrics

### Quantitative Measures

**Trustworthiness:**
```
T(k) = 1 - (2/(nk(2n-3k-1))) ∑ᵢ ∑_{j∈U_k(i)} (r(i,j) - k)
```

Where U_k(i) are k-nearest neighbors of i in low-dimensional space that are not k-nearest neighbors in high-dimensional space.

**Continuity:**
```
C(k) = 1 - (2/(nk(2n-3k-1))) ∑ᵢ ∑_{j∈V_k(i)} (r̂(i,j) - k)
```

Where V_k(i) are k-nearest neighbors of i in high-dimensional space that are not k-nearest neighbors in low-dimensional space.

**Neighborhood Preservation:**
```
NP(k) = (1/n) ∑ᵢ |NN_k(i) ∩ NN_k'(i)| / k
```

Where NN_k(i) are k-nearest neighbors in original space and NN_k'(i) in embedding space.

### Qualitative Assessment

**Visual Inspection:**
- Cluster separation and cohesion
- Preservation of known class structure
- Absence of artificial clustering

**Stability Analysis:**
Run multiple times with different random seeds to assess stability.

## Computational Optimization

### Implementation Strategies

**Vectorization:**
```python
# Compute all pairwise distances at once
distances = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)

# Compute q_ij matrix
Q = (1 + distances) ** -1
np.fill_diagonal(Q, 0)
Q = Q / np.sum(Q)
```

**Memory Management:**
- Use sparse matrices for P when perplexity << n
- Implement incremental gradient computation
- Use float32 instead of float64 when appropriate

### Parallel Computing

**Multi-threading:**
- Parallelize gradient computation across points
- Parallelize tree construction (Barnes-Hut)

**GPU Acceleration:**
- CUDA implementations for large-scale t-SNE
- Approximate methods for very large datasets (n > 10⁶)

## Limitations and Considerations

### Algorithmic Limitations

**Non-Convexity:**
Multiple runs may yield different results.

**Computational Complexity:**
O(n²) scaling limits applicability to large datasets.

**Hyperparameter Sensitivity:**
Results depend significantly on perplexity and learning rate.

**Global Structure:**
May not preserve global relationships accurately.

### Interpretive Cautions

**Distance Interpretation:**
Distances in t-SNE plots are not meaningful beyond nearest neighbors.

**Cluster Size:**
Cluster sizes in t-SNE may not reflect true cluster sizes.

**Topology:**
May create artificial clusters or split natural clusters.

### Best Practices

**Multiple Perplexities:**
Try different perplexity values to understand data structure.

**Initialization:**
Use multiple random initializations to assess consistency.

**Validation:**
Compare with other dimensionality reduction methods (PCA, UMAP).

**Domain Knowledge:**
Interpret results in context of domain expertise.

## Mathematical Summary

t-SNE represents a sophisticated approach to nonlinear dimensionality reduction that elegantly addresses the crowding problem:

1. **Probability-Based Framework**: Converts distances to probabilities for robust comparison
2. **Heavy-Tailed Distributions**: Uses t-distribution to provide more space for dissimilar points
3. **KL Divergence Minimization**: Principled objective function with clear interpretation
4. **Gradient-Based Optimization**: Efficient optimization with momentum and adaptive learning

The mathematical beauty of t-SNE lies in its probabilistic interpretation of similarity and the clever use of different distributions in high and low dimensions to address fundamental limitations of linear methods.

**Key Insight**: t-SNE's success stems from recognizing that preserving all pairwise distances in lower dimensions is impossible, so it focuses on preserving neighborhood relationships through probability distributions. The asymmetric treatment of attractive and repulsive forces, combined with heavy-tailed distributions, enables faithful preservation of local structure while maintaining global organization. 