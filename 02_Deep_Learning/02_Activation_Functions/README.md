# 02_Activation_Functions

In a neural network, the activation function is responsible for transforming the summed weighted input from the node into the activation of the node or output for that input. These functions introduce non-linear properties to the network, which is critical for learning complex data patterns. Without them, the neural network would simply be a linear regression model.

### Common Activation Functions:

-   **Sigmoid**: S-shaped curve. It squashes values between 0 and 1.
-   **Tanh (Hyperbolic Tangent)**: Also S-shaped, but squashes values between -1 and 1.
-   **ReLU (Rectified Linear Unit)**: Outputs the input directly if it is positive, otherwise, it outputs zero. It is the most widely used activation function.
-   **Leaky ReLU**: A variant of ReLU that allows a small, non-zero gradient when the unit is not active.
-   **Softmax**: Often used in the output layer of a multi-class classification network. It converts a vector of numbers into a probability distribution. 