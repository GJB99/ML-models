# Feedforward and Backpropagation

Neural networks learn from data through two core processes: **Feedforward** and **Backpropagation**. Together, they form the foundation of modern deep learning, enabling the training of complex models.

Backpropagation, short for "backward propagation of error," is the algorithm that efficiently computes the gradients needed to optimize the network's parameters (weights and biases).

### The Training Cycle

1.  **Forward Pass**: Input data is fed into the network and flows from the input layer through the hidden layers to the output layer. At each layer, neurons compute a weighted sum of their inputs and apply an activation function. The final output is the network's prediction.

2.  **Loss Calculation**: The network's prediction is compared to the actual target value using a loss function, which quantifies the error.

3.  **Backward Pass (Backpropagation)**: The error is propagated backward through the network. The algorithm uses the chain rule of calculus to compute the gradient of the loss function with respect to each weight and bias in the network. This gradient indicates the direction in which the parameters should be adjusted to minimize the error.

4.  **Weight Update**: The network's parameters are adjusted in the opposite direction of their gradients, typically using an optimization algorithm like Gradient Descent. This step aims to reduce the error on the next forward pass.

This iterative process of forward pass, loss calculation, backward pass, and weight update enables the neural network to learn complex non-linear relationships in the data. 