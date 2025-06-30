# 01_Perceptron

The Perceptron is the simplest type of artificial neural network. It is a linear classifier that can solve binary classification problems.

### How it works:

1.  It takes multiple binary inputs `x1, x2, ..., xn`.
2.  Each input has a weight `w1, w2, ..., wn` associated with it.
3.  The neuron computes a weighted sum of the inputs.
4.  It applies an activation function (typically a step function) to this sum. If the sum is above a certain threshold, the output is 1; otherwise, it is 0.
5.  The weights are updated based on the error of the prediction.

### Limitations:

-   A single-layer perceptron can only learn linearly separable patterns. It famously cannot solve the XOR problem. 