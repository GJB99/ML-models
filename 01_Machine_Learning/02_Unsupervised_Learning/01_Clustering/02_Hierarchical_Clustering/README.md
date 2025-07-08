# Hierarchical Clustering

Hierarchical clustering is a fundamental unsupervised learning method that builds a tree-like hierarchy of clusters, revealing the nested structure of data at multiple scales. Unlike partition-based methods that produce a single clustering, hierarchical methods create a dendrogram showing relationships between clusters at different levels of granularity. This approach provides rich insights into data structure and allows flexible cluster selection based on application needs.

## Mathematical Framework

### Basic Definitions

**Dataset:**
```
X = {x₁, x₂, ..., xₙ}  where xᵢ ∈ ℝᵈ
```

**Distance Matrix:**
```
D = [dᵢⱼ]  where dᵢⱼ = dist(xᵢ, xⱼ)
```

**Cluster Hierarchy:**
A sequence of partitions:
```
C⁰, C¹, C², ..., Cⁿ⁻¹
```

Where:
- C⁰ = {{x₁}, {x₂}, ..., {xₙ}} (singleton clusters)
- Cⁿ⁻¹ = {X} (single cluster containing all points)

**Dendrogram:**
Tree structure T = (V, E) where:
- V: nodes representing clusters
- E: edges representing merges/splits
- Each internal node has height h indicating merge distance

## Agglomerative Hierarchical Clustering

### Algorithm Framework

**Agglomerative Algorithm:**
```
Input: Distance matrix D, Linkage criterion L
Output: Dendrogram T

1. Initialize: Each point as singleton cluster
   C = {{x₁}, {x₂}, ..., {xₙ}}

2. While |C| > 1:
   a. Find closest cluster pair (Cᵢ, Cⱼ):
      (i*, j*) = argmin_{i<j} L(Cᵢ, Cⱼ)
   
   b. Merge clusters:
      Cₖ = Cᵢ* ∪ Cⱼ*
      C = (C \ {Cᵢ*, Cⱼ*}) ∪ {Cₖ}
   
   c. Update distances:
      ∀ Cₗ ∈ C: d(Cₖ, Cₗ) = f(d(Cᵢ*, Cₗ), d(Cⱼ*, Cₗ))

3. Return dendrogram T
```

### Linkage Criteria

**Single Linkage (Minimum):**
```
d_single(A, B) = min{d(a, b) | a ∈ A, b ∈ B}
```

**Mathematical Properties:**
- Tends to create elongated clusters
- Sensitive to noise and outliers
- Can handle non-convex cluster shapes

**Complete Linkage (Maximum):**
```
d_complete(A, B) = max{d(a, b) | a ∈ A, b ∈ B}
```

**Properties:**
- Creates compact, spherical clusters
- Less sensitive to outliers
- Prefers balanced cluster sizes

**Average Linkage (UPGMA):**
```
d_average(A, B) = (1/(|A||B|)) ∑_{a∈A} ∑_{b∈B} d(a, b)
```

**Weighted Average:**
```
d_weighted(A, B) = (1/2)[d_average(A, B)]
```

**Centroid Linkage:**
```
d_centroid(A, B) = ||μ_A - μ_B||²
```

Where μ_A and μ_B are cluster centroids.

**Ward's Linkage:**
```
d_ward(A, B) = √((|A||B|)/(|A|+|B|)) ||μ_A - μ_B||²
```

**Ward's Criterion (ESS):**
Minimizes increase in within-cluster sum of squares:
```
Δ(A, B) = ESS(A ∪ B) - ESS(A) - ESS(B)
```

Where:
```
ESS(C) = ∑_{x∈C} ||x - μ_C||²
```

**Ward's Formula:**
```
Δ(A, B) = (|A||B|)/(|A|+|B|) ||μ_A - μ_B||²
```

### Distance Update Formulas

**Lance-Williams Formula:**
General update formula for cluster distances:
```
d(A∪B, C) = αᴬd(A,C) + αᴮd(B,C) + βd(A,B) + γ|d(A,C) - d(B,C)|
```

**Parameters for Different Linkages:**

**Single Linkage:**
```
αᴬ = αᴮ = 1/2, β = 0, γ = -1/2
```

**Complete Linkage:**
```
αᴬ = αᴮ = 1/2, β = 0, γ = 1/2
```

**Average Linkage:**
```
αᴬ = |A|/(|A|+|B|), αᴮ = |B|/(|A|+|B|), β = γ = 0
```

**Ward's Linkage:**
```
αᴬ = (|A|+|C|)/(|A|+|B|+|C|)
αᴮ = (|B|+|C|)/(|A|+|B|+|C|)
β = -|C|/(|A|+|B|+|C|), γ = 0
```

## Divisive Hierarchical Clustering

### Algorithm Framework

**Divisive Algorithm:**
```
Input: Dataset X, Split criterion S
Output: Dendrogram T

1. Initialize: C = {X}

2. While stopping criterion not met:
   a. Select cluster to split:
      C* = argmax_{C∈C} split_priority(C)
   
   b. Split C* into subclusters:
      {C₁, C₂, ..., Cₖ} = split(C*)
   
   c. Update cluster set:
      C = (C \ {C*}) ∪ {C₁, C₂, ..., Cₖ}

3. Return dendrogram T
```

### Splitting Strategies

**Bisecting k-means:**
```
1. Apply k-means with k=2 to cluster C
2. Select split with minimum total SSE:
   SSE = SSE(C₁) + SSE(C₂)
```

**Principal Component Splitting:**
```
1. Compute principal component v₁ of cluster C
2. Split along hyperplane perpendicular to v₁:
   C₁ = {x ∈ C | (x - μ_C) · v₁ ≤ 0}
   C₂ = {x ∈ C | (x - μ_C) · v₁ > 0}
```

**Maximum Distance Split:**
```
1. Find diameter points: (a, b) = argmax_{x,y∈C} d(x, y)
2. Assign points to nearest diameter point
```

## Dendrogram Mathematics

### Height Function

**Merge Height:**
```
h(Cᵢ ∪ Cⱼ) = d(Cᵢ, Cⱼ)
```

**Monotonicity Property:**
For valid hierarchical clustering:
```
h(merge_t) ≤ h(merge_{t+1})
```

**Ultra-metric Property:**
For any three points x, y, z:
```
d_ultra(x, z) ≤ max{d_ultra(x, y), d_ultra(y, z)}
```

### Cophenetic Correlation

**Cophenetic Distance:**
```
c(xᵢ, xⱼ) = height of lowest common ancestor in dendrogram
```

**Cophenetic Correlation Coefficient:**
```
r = corr(original_distances, cophenetic_distances)
```

**Mathematical Formula:**
```
r = (Σᵢ<ⱼ (dᵢⱼ - d̄)(cᵢⱼ - c̄)) / √(Σᵢ<ⱼ (dᵢⱼ - d̄)² Σᵢ<ⱼ (cᵢⱼ - c̄)²)
```

Where d̄ and c̄ are means of original and cophenetic distances.

## Theoretical Properties

### Complexity Analysis

**Time Complexity:**

**Naive Implementation:**
```
O(n³)
```

**With Efficient Data Structures:**
```
O(n² log n)
```

**Optimal Algorithms:**
Single and complete linkage can be computed in O(n²) time.

**Space Complexity:**
```
O(n²)
```

For storing the distance matrix.

### Linkage Properties

**Single Linkage:**
- **Monotonic**: Yes
- **Space-dilating**: No
- **Space-contracting**: Yes
- **Tends to**: Chain-like clusters

**Complete Linkage:**
- **Monotonic**: Yes  
- **Space-dilating**: Yes
- **Space-contracting**: No
- **Tends to**: Compact, spherical clusters

**Average Linkage:**
- **Monotonic**: Yes
- **Space-dilating**: Weakly
- **Space-contracting**: Weakly
- **Tends to**: Balanced clusters

**Ward's Linkage:**
- **Monotonic**: Yes
- **Minimizes**: Within-cluster variance
- **Tends to**: Equal-sized, spherical clusters

### Space Properties

**Space-Dilating:**
```
d(A∪B, C) ≥ max{d(A,C), d(B,C)}
```

**Space-Contracting:**
```
d(A∪B, C) ≤ min{d(A,C), d(B,C)}
```

## Distance Metrics

### Euclidean Distance
```
d_E(x, y) = √(Σᵢ₌₁ᵈ (xᵢ - yᵢ)²)
```

### Manhattan Distance
```
d_M(x, y) = Σᵢ₌₁ᵈ |xᵢ - yᵢ|
```

### Cosine Distance
```
d_cos(x, y) = 1 - (x·y)/(||x|| ||y||)
```

### Correlation Distance
```
d_corr(x, y) = 1 - corr(x, y)
```

### Mahalanobis Distance
```
d_Maha(x, y) = √((x-y)ᵀ Σ⁻¹ (x-y))
```

## Optimal Number of Clusters

### Gap Statistic

**Gap Definition:**
```
Gap(k) = E[log(W_k)] - log(W_k)
```

Where W_k is within-cluster dispersion for k clusters.

**Optimal k:**
```
k* = argmin_k {k : Gap(k) ≥ Gap(k+1) - s_{k+1}}
```

### Silhouette Analysis

**Silhouette Width:**
```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

**Average Silhouette:**
```
S(k) = (1/n) Σᵢ₌₁ⁿ s(i)
```

### Calinski-Harabasz Index

**CH Index:**
```
CH(k) = [tr(B)/(k-1)] / [tr(W)/(n-k)]
```

Where:
- B: between-cluster scatter matrix
- W: within-cluster scatter matrix

### Elbow Method

**Within-Cluster Sum of Squares:**
```
WCSS(k) = Σⱼ₌₁ᵏ Σₓ∈Cⱼ ||x - μⱼ||²
```

**Elbow Point:**
Find k where WCSS decrease rate changes significantly.

## Advanced Techniques

### Constrained Hierarchical Clustering

**Must-Link Constraints:**
```
ML = {(xᵢ, xⱼ) | xᵢ and xⱼ must be in same cluster}
```

**Cannot-Link Constraints:**
```
CL = {(xᵢ, xⱼ) | xᵢ and xⱼ cannot be in same cluster}
```

**Modified Distance:**
```
d'(xᵢ, xⱼ) = {
    0           if (xᵢ, xⱼ) ∈ ML
    ∞           if (xᵢ, xⱼ) ∈ CL
    d(xᵢ, xⱼ)   otherwise
}
```

### Robust Hierarchical Clustering

**Medoid-Based Linkage:**
```
d_medoid(A, B) = d(medoid(A), medoid(B))
```

Where medoid minimizes sum of distances within cluster:
```
medoid(C) = argmin_{x∈C} Σ_{y∈C} d(x, y)
```

### Probabilistic Hierarchical Clustering

**Likelihood-Based Merging:**
```
merge_score(A, B) = log P(A ∪ B) - log P(A) - log P(B)
```

**Gaussian Assumption:**
```
P(C) = (2π)^{-|C|d/2} |Σ_C|^{-1/2} exp(-1/2 Σₓ∈C (x-μ_C)ᵀ Σ_C⁻¹ (x-μ_C))
```

## Cluster Validation

### Internal Validation

**Dunn Index:**
```
DI = min_{1≤i≤k} {min_{1≤j≤k,j≠i} {δ(Cᵢ, Cⱼ) / max_{1≤l≤k} Δ(Cₗ)}}
```

Where:
- δ(Cᵢ, Cⱼ): inter-cluster distance
- Δ(Cₗ): intra-cluster distance

**Davies-Bouldin Index:**
```
DB = (1/k) Σᵢ₌₁ᵏ max_{j≠i} [(σᵢ + σⱼ) / d(cᵢ, cⱼ)]
```

### External Validation

**Adjusted Rand Index:**
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

**Normalized Mutual Information:**
```
NMI = MI(C, T) / √(H(C) × H(T))
```

## Applications

### Phylogenetic Analysis

**Evolutionary Trees:**
Construct species evolution hierarchies using genetic distances.

**UPGMA Method:**
Assume molecular clock: equal evolution rates.

### Social Network Analysis

**Community Detection:**
Hierarchical community structure in networks.

**Modularity Optimization:**
```
Q = (1/2m) Σᵢⱼ [Aᵢⱼ - (kᵢkⱼ)/(2m)] δ(cᵢ, cⱼ)
```

### Image Segmentation

**Region Growing:**
Merge similar adjacent regions hierarchically.

**Color/Texture Hierarchies:**
Multi-scale image analysis.

### Gene Expression Analysis

**Co-expression Networks:**
Group genes with similar expression patterns.

**Functional Annotation:**
Discover gene functional groups.

## Implementation Considerations

### Memory Optimization

**Sparse Distance Matrices:**
Store only non-zero distances for sparse data.

**Streaming Algorithms:**
Process large datasets in chunks.

### Parallelization

**Distance Computation:**
Parallelize distance matrix calculation.

**Linkage Updates:**
Parallel cluster distance updates.

### Efficient Data Structures

**Priority Queues:**
For finding minimum distance pairs.

**Union-Find:**
For tracking cluster membership.

**Nearest Neighbor Chains:**
Optimize single/complete linkage computation.

## Computational Optimizations

### SLINK Algorithm (Single Linkage)

**Time Complexity:** O(n²)
**Space Complexity:** O(n)

**Key Insight:**
Maintain pointer representation of dendrogram.

### CLINK Algorithm (Complete Linkage)

**Time Complexity:** O(n²)

**Reciprocal Nearest Neighbor Chains:**
Exploit geometric properties for optimization.

### BIRCH Integration

**CF-Tree Construction:**
Build compact clustering features tree.

**Agglomerative Phase:**
Apply hierarchical clustering to CF-tree nodes.

## Mathematical Summary

Hierarchical clustering provides a mathematically elegant framework for multi-scale data analysis:

1. **Linkage Criteria**: Mathematical definitions of cluster proximity
2. **Ultra-metric Spaces**: Theoretical foundation for dendrogram construction  
3. **Optimization Objectives**: Ward's linkage minimizes within-cluster variance
4. **Geometric Properties**: Space-dilating vs. space-contracting behaviors

The Lance-Williams formula unifies different linkage methods under a single mathematical framework, while the ultra-metric property ensures valid hierarchical structure.

**Key Insight**: Hierarchical clustering reveals the inherent multi-scale structure of data through mathematically principled merging/splitting criteria. The choice of linkage criterion fundamentally determines the geometric properties of resulting clusters, making it crucial to match the linkage method to the underlying data structure and application requirements. 