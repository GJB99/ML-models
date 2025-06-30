# 02_Multi-Layer_Perceptron

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.

### Key Features:

-   **Hidden Layers**: MLPs have one or more hidden layers between the input and output layers. These hidden layers allow the network to learn more complex, non-linear patterns. This overcomes the limitations of the single-layer perceptron (e.g., it can solve the XOR problem).
-   **Non-linear Activation Functions**: Unlike the simple step function in a perceptron, MLPs use non-linear activation functions (like Sigmoid, Tanh, or ReLU) in their hidden layers. This is crucial for learning complex relationships in the data.
-   **Backpropagation**: MLPs are trained using the backpropagation algorithm, which adjusts the weights of the network based on the error in the output. 