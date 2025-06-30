# 15_Advanced_Training_Techniques

Beyond model architecture, the specific techniques used for training and optimization are critical for achieving state-of-the-art performance. This section covers advanced methods for optimization and regularization.

### Advanced Optimizers:

-   **AdamW**: An evolution of the Adam optimizer that improves weight decay regularization. It decouples the weight decay from the gradient update, which can lead to better generalization. It has become the default choice for training Transformers.
-   **RAdam (Rectified Adam)**: A variant of Adam that introduces a term to rectify the variance of the adaptive learning rate, seeking to overcome bad local optima in the early stages of training.

### Regularization Techniques:

Regularization is crucial for preventing overfitting and improving a model's ability to generalize to unseen data.

-   **Batch Normalization**: Normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. It helps stabilize and accelerate training, and also acts as a regularizer.
-   **Layer Normalization**: Normalizes the inputs across the features, instead of across the batch dimension. It is not dependent on the batch size and is particularly effective for RNNs and Transformers.
-   **Group Normalization**: A compromise between LayerNorm and BatchNorm. It divides channels into groups and normalizes within each group. It is useful when the batch size is too small for effective BatchNorm.
-   **Advanced Dropout Variants**:
    -   **DropConnect**: Instead of dropping activation units, DropConnect drops individual weights in the network.
    -   **Spatial Dropout**: A dropout technique specifically for convolutional layers. It drops entire feature maps rather than individual pixels, which is more effective at preventing correlations between feature maps. 