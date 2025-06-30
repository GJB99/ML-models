# 10_Temporal_Convolutional_Networks

Temporal Convolutional Networks (TCNs) are a type of neural network architecture designed for sequence modeling tasks, serving as a powerful alternative to Recurrent Neural Networks (RNNs).

### Core Components:

TCNs are built upon two key principles:
1.  **Causal Convolutions**: These are convolutions where the output at time `t` is convolved only with elements from time `t` and earlier in the previous layer. This ensures that the model cannot violate the ordering of the data (i.e., it can't look into the future).
2.  **Dilated Convolutions**: To capture long-range dependencies, TCNs use dilated convolutions. This involves applying the filter over an area larger than its length by skipping input values with a certain step. By stacking layers with exponentially increasing dilation factors, the network can have a very large receptive field, allowing it to look far back into the sequence's history.

### Advantages over RNNs:

-   **Parallelism**: Convolutions can be performed in parallel, as the same filter is used across the sequence, making TCNs potentially faster than RNNs for training and evaluation.
-   **Stable Gradients**: TCNs are not as susceptible to the vanishing/exploding gradient problems that affect simple RNNs.
-   **Large Receptive Field**: The use of dilated convolutions allows for a controlled and large receptive field. 