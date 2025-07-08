# K-Means Clustering

K-means clustering is one of the most fundamental and widely-used unsupervised learning algorithms for partitioning data into k distinct clusters. Developed by Stuart Lloyd in 1957 and formalized by James MacQueen in 1967, k-means seeks to minimize within-cluster variance by iteratively optimizing cluster centroids and assignments. Despite its simplicity, k-means remains highly effective for many real-world clustering tasks.

## Mathematical Framework

### Objective Function

**K-means Optimization Problem:**
```
minimize J = ∑_{i=1}^n ∑_{j=1}^k r_{ij} ||x_i - μ_j||²
```

Where:
- **n**: number of data points
- **k**: number of clusters
- **x_i ∈ ℝᵈ**: i-th data point
- **μ_j ∈ ℝᵈ**: centroid of cluster j
- **r_{ij} ∈ {0,1}**: assignment indicator (r_{ij} = 1 if x_i belongs to cluster j)

**Constraint:**
```
∑_{j=1}^k r_{ij} = 1  for all i
```

Each data point belongs to exactly one cluster.

### Within-Cluster Sum of Squares (WCSS)

**WCSS Definition:**
```
WCSS = ∑_{j=1}^k ∑_{x_i ∈ C_j} ||x_i - μ_j||²
```

Where C_j is the set of points assigned to cluster j.

**Alternative Formulation:**
```
WCSS = ∑_{j=1}^k |C_j| · Var(C_j)
```

Where |C_j| is the size of cluster j and Var(C_j) is its variance.

## Algorithm Steps

### 1. Initialization

**Random Initialization:**
```
μ_j^{(0)} = random_sample(X)  for j = 1, ..., k
```

**K-means++ Initialization:**
Choose centroids with probability proportional to squared distance:
```
P(x_i selected) = D(x_i)² / ∑_{x ∈ X} D(x)²
```

Where D(x_i) is the distance to the nearest already chosen centroid.

### 2. Assignment Step

**Cluster Assignment:**
```
c_i^{(t)} = argmin_j ||x_i - μ_j^{(t-1)}||²
```

**Indicator Variables:**
```
r_{ij}^{(t)} = {
    1  if j = c_i^{(t)}
    0  otherwise
}
```

### 3. Update Step

**Centroid Update:**
```
μ_j^{(t)} = (1/|C_j^{(t)}|) ∑_{x_i ∈ C_j^{(t)}} x_i
```

**Matrix Form:**
```
μ_j^{(t)} = (∑_{i=1}^n r_{ij}^{(t)} x_i) / (∑_{i=1}^n r_{ij}^{(t)})
```

### 4. Convergence Criteria

**Centroid Convergence:**
```
||μ_j^{(t)} - μ_j^{(t-1)}|| < ε  for all j
```

**Assignment Convergence:**
```
c_i^{(t)} = c_i^{(t-1)}  for all i
```

**Objective Function Convergence:**
```
|J^{(t)} - J^{(t-1)}| < δ
```

## Distance Metrics

### Euclidean Distance (Default)

**Standard Euclidean:**
```
d(x_i, μ_j) = √(∑_{l=1}^d (x_{il} - μ_{jl})²)
```

**Squared Euclidean (Optimization):**
```
d²(x_i, μ_j) = ∑_{l=1}^d (x_{il} - μ_{jl})²
```

### Alternative Distance Metrics

**Manhattan Distance:**
```
d(x_i, μ_j) = ∑_{l=1}^d |x_{il} - μ_{jl}|
```

**Mahalanobis Distance:**
```
d(x_i, μ_j) = √((x_i - μ_j)ᵀ Σ⁻¹ (x_i - μ_j))
```

**Cosine Distance:**
```
d(x_i, μ_j) = 1 - (x_i · μ_j)/(||x_i|| ||μ_j||)
```

## Theoretical Properties

### Convergence Analysis

**Monotonic Decrease:**
The objective function J monotonically decreases:
```
J^{(t+1)} ≤ J^{(t)}
```

**Finite Convergence:**
K-means converges in finite iterations since:
- Finite number of possible partitions: k^n
- Objective decreases at each step
- Lower bounded by 0

**Convergence Guarantee:**
```
lim_{t→∞} J^{(t)} = J*
```

Where J* is a local minimum.

### Optimal Solutions

**Global Optimum:**
Finding the global optimum is NP-hard for k ≥ 2 and d ≥ 2.

**Local Optimum:**
K-means finds a local optimum that depends on initialization.

**Multiple Runs Strategy:**
```
J_best = min{J₁, J₂, ..., J_R}
```

Run algorithm R times with different initializations.

## Computational Complexity

### Time Complexity

**Per Iteration:**
```
O(nkd)
```

Where:
- **n**: number of data points
- **k**: number of clusters  
- **d**: dimensionality

**Total Complexity:**
```
O(t · n · k · d)
```

Where t is the number of iterations until convergence.

**Typical Convergence:**
```
t = O(log n)  in practice
```

### Space Complexity

**Memory Requirements:**
```
O(nd + kd)
```

For storing data points and centroids.

## Variants and Improvements

### K-means++

**Improved Initialization:**
```
E[J_kmeans++] ≤ 8(ln k + 2) · J_optimal
```

**Selection Probability:**
```
P(x selected) = D²(x) / ∑_{y} D²(y)
```

### Mini-batch K-means

**Stochastic Updates:**
```
μ_j^{(t)} = (1-η) μ_j^{(t-1)} + η · (1/|B_j|) ∑_{x_i ∈ B_j} x_i
```

Where B_j is the mini-batch assigned to cluster j and η is the learning rate.

**Learning Rate:**
```
η = |C_j| / (|C_j| + |B_j|)
```

### Fuzzy C-means

**Soft Assignment:**
```
u_{ij} = 1 / (∑_{l=1}^k (||x_i - μ_j|| / ||x_i - μ_l||)^{2/(m-1)})
```

Where m > 1 is the fuzziness parameter.

**Fuzzy Centroids:**
```
μ_j = (∑_{i=1}^n u_{ij}^m x_i) / (∑_{i=1}^n u_{ij}^m)
```

## Determining Optimal k

### Elbow Method

**Within-cluster Sum of Squares:**
```
WCSS(k) = ∑_{j=1}^k ∑_{x_i ∈ C_j} ||x_i - μ_j||²
```

**Elbow Point:**
Find k where the rate of WCSS decrease sharply changes:
```
k* = argmax_k [WCSS(k-1) - WCSS(k)] - [WCSS(k) - WCSS(k+1)]
```

### Silhouette Analysis

**Silhouette Coefficient:**
For point i in cluster A:
```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

Where:
- **a(i)**: average distance to points in same cluster
- **b(i)**: average distance to points in nearest different cluster

**Average Silhouette:**
```
S(k) = (1/n) ∑_{i=1}^n s(i)
```

**Optimal k:**
```
k* = argmax_k S(k)
```

### Gap Statistic

**Gap Definition:**
```
Gap(k) = E[log(WCSS_ref(k))] - log(WCSS(k))
```

Where WCSS_ref(k) is computed on reference (random) data.

**Optimal k:**
```
k* = smallest k such that Gap(k) ≥ Gap(k+1) - s_{k+1}
```

Where s_{k+1} is the standard error.

### Information Criteria

**Akaike Information Criterion (AIC):**
```
AIC(k) = 2kd - 2 log(L)
```

**Bayesian Information Criterion (BIC):**
```
BIC(k) = kd log(n) - 2 log(L)
```

Where L is the likelihood under Gaussian assumption.

## Cluster Validation

### Internal Validation

**Calinski-Harabasz Index:**
```
CH(k) = [tr(B)/(k-1)] / [tr(W)/(n-k)]
```

Where:
- **B**: between-cluster scatter matrix
- **W**: within-cluster scatter matrix

**Davies-Bouldin Index:**
```
DB(k) = (1/k) ∑_{j=1}^k max_{l≠j} [(σ_j + σ_l) / ||μ_j - μ_l||]
```

Where σ_j is the average distance within cluster j.

### External Validation

**Adjusted Rand Index (ARI):**
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

**Normalized Mutual Information (NMI):**
```
NMI = MI(C, T) / √(H(C) × H(T))
```

Where C are predicted clusters and T are true clusters.

## Practical Considerations

### Data Preprocessing

**Standardization:**
```
x̃_{il} = (x_{il} - μ_l) / σ_l
```

Essential when features have different scales.

**Outlier Removal:**
Remove points with z-scores > threshold:
```
|x_{il} - μ_l| / σ_l > z_threshold
```

### Handling Categorical Data

**One-hot Encoding:**
Transform categorical variables to binary vectors.

**Hamming Distance:**
For categorical data:
```
d(x_i, x_j) = ∑_{l=1}^d I(x_{il} ≠ x_{jl})
```

## Limitations and Assumptions

### Assumptions

**Spherical Clusters:**
K-means assumes clusters are spherical (isotropic).

**Similar Sizes:**
Algorithm biased toward clusters of similar sizes.

**Well-separated:**
Works best when clusters are well-separated.

### Limitations

**Fixed k:**
Number of clusters must be specified in advance.

**Sensitive to Initialization:**
Different initializations can yield different results.

**Outlier Sensitivity:**
Means are sensitive to outliers.

**Non-convex Clusters:**
Cannot handle non-convex cluster shapes.

## Extensions and Applications

### Kernel K-means

**Feature Mapping:**
```
φ: ℝᵈ → ℋ
```

**Kernel Objective:**
```
J = ∑_{i=1}^n ∑_{j=1}^k r_{ij} ||φ(x_i) - μ_j^φ||²
```

**Kernel Trick:**
```
||φ(x_i) - μ_j^φ||² = k(x_i, x_i) - 2∑_{l ∈ C_j} r_{lj}k(x_i, x_l)/|C_j| + ...
```

### Spherical K-means

**Cosine Similarity:**
For text clustering with TF-IDF vectors:
```
sim(x_i, μ_j) = (x_i · μ_j) / (||x_i|| ||μ_j||)
```

**Normalized Centroids:**
```
μ_j = (∑_{i ∈ C_j} x_i) / ||∑_{i ∈ C_j} x_i||
```

### Applications

**Image Segmentation:**
- Pixel clustering based on color/intensity
- Feature extraction for computer vision

**Customer Segmentation:**
- Behavioral pattern analysis
- Marketing strategy optimization

**Data Compression:**
- Vector quantization
- Codebook generation

**Anomaly Detection:**
- Outlier identification
- Quality control

## Implementation Guidelines

### Algorithm Implementation

```python
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Initialize centroids
    centroids = kmeans_plus_plus_init(X, k)
    
    for iter in range(max_iters):
        # Assignment step
        distances = compute_distances(X, centroids)
        assignments = np.argmin(distances, axis=1)
        
        # Update step
        new_centroids = update_centroids(X, assignments, k)
        
        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=tol):
            break
            
        centroids = new_centroids
    
    return centroids, assignments
```

### Performance Optimization

**Vectorization:**
Use matrix operations instead of loops.

**Early Stopping:**
Monitor convergence criteria.

**Parallel Processing:**
Parallelize distance computations.

**Memory Efficiency:**
Use in-place operations when possible.

## Mathematical Summary

K-means clustering demonstrates the power of iterative optimization in unsupervised learning:

1. **Coordinate Descent**: Alternating between assignment and update steps
2. **Lloyd's Algorithm**: Guaranteed convergence to local optimum
3. **Geometric Intuition**: Minimizing within-cluster variance through centroid optimization
4. **Computational Efficiency**: Simple operations with polynomial complexity

The algorithm's effectiveness stems from its mathematical simplicity: by alternately optimizing cluster assignments and centroids, k-means efficiently finds locally optimal partitions that minimize within-cluster variance.

**Key Takeaway**: K-means exemplifies how elegant mathematical formulations can lead to practical algorithms. Its coordinate descent approach to the clustering optimization problem provides a template for many other machine learning algorithms, while its geometric interpretation makes it intuitive and interpretable. Understanding k-means provides fundamental insights into optimization-based clustering and iterative algorithm design. 