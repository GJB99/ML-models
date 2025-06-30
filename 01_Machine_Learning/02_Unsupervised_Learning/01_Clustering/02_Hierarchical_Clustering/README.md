# 02_Hierarchical_Clustering

Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering fall into two types:

-   **Agglomerative**: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
-   **Divisive**: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

### Key Concepts:

-   **Dendrogram**: A tree-like diagram that records the sequences of merges or splits. It illustrates the arrangement of the clusters produced by the corresponding analyses.
-   **Linkage Criteria**: Specifies the distance between clusters. Common criteria include:
    -   **Single Linkage**: The distance between the closest members of the two clusters.
    -   **Complete Linkage**: The distance between the members that are farthest apart.
    -   **Average Linkage**: The average distance between each member in one cluster to every member in the other cluster.
    -   **Ward's Linkage**: Minimizes the variance of the clusters being merged. 