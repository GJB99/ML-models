# 03_DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm. It is particularly good at finding non-linearly separable clusters and handling noise in the data.

### How it works:

DBSCAN groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It is based on two key parameters:

-   **`eps` (epsilon)**: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
-   **`min_samples`**: The number of samples in a neighborhood for a point to be considered as a core point.

### Key Concepts:

-   **Core Point**: A point that has at least `min_samples` points (including itself) within its `eps` radius.
-   **Border Point**: A point that is not a core point but is in the neighborhood of a core point.
-   **Noise Point (Outlier)**: A point that is not a core point and not a border point. 