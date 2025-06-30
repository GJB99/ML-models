# 03_Graph_Neural_Networks

Graph Neural Networks (GNNs) are a class of deep learning methods designed to perform inference on data described by graphs. They are essential for tasks where the relationships and connections between data points are crucial, such as social network analysis, molecular chemistry, and recommendation systems.

### How they work:

The core idea of GNNs is **message passing**. Each node in the graph has a feature vector (an embedding). In each layer of the GNN, every node aggregates the feature vectors of its neighbors to compute its new feature vector. This process is repeated for `k` layers, allowing a node's representation to be informed by its `k`-hop neighborhood.

### Key Architectures:

-   **Graph Convolutional Networks (GCN)**: A popular type of GNN that uses a simplified, efficient neighborhood aggregation scheme. It can be seen as a spectral approach that has been adapted to be more scalable.

-   **GraphSAGE (Graph SAmple and aggreGatE)**: A framework for inductive representation learning on large graphs. Instead of training on the entire graph, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. This allows it to generalize to unseen nodes.

-   **Graph Attention Networks (GAT)**: Incorporates the attention mechanism into the message passing framework. GATs learn to assign different importance weights to different nodes within a neighborhood, allowing for a more expressive and powerful aggregation. 