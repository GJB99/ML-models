# 07_k_Nearest_Neighbors

k-Nearest Neighbors (k-NN) is a non-parametric, lazy learning algorithm. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.

### Key Concepts:

-   **Lazy Learning**: k-NN is a lazy algorithm because it does not have a dedicated training phase. All computation is deferred until classification.
-   **k-Value**: The number of nearest neighbors to consider. Choosing the right `k` is crucial for the model's performance.
-   **Distance Metric**: A function that measures the "distance" or "similarity" between two data points. The most common metric is Euclidean distance, but others like Manhattan distance can also be used. 