# Gradient Descent and Optimizers

Gradient descent serves as a fundamental optimization algorithm for training machine learning models and neural networks by minimizing errors between predicted and actual results. This first-order iterative algorithm finds the minimum of a differentiable function.

### Mathematical Foundation

The algorithm operates on the principle that a function decreases fastest in the direction of the negative gradient. The update rule for a parameter `a` is:
$$ a_{n+1} = a_n - \eta \nabla f(a_n) $$

where `η` (eta) is the **learning rate** (controlling the step size) and `∇f(a_n)` is the gradient of the loss function `f` at point `a_n`.

### Types of Gradient Descent

-   **Batch Gradient Descent**: Computes the gradient of the cost function with respect to the parameters for the entire training dataset. It can be very slow and intractable for large datasets.
-   **Stochastic Gradient Descent (SGD)**: Performs a parameter update for each training example. It is much faster, but the updates have high variance, causing the objective function to fluctuate heavily. The user's provided text mentions that this is a simple extension that serves as the basic algorithm for training most deep networks, updating parameters using individual data points or small batches, making it computationally efficient for large datasets.
-   **Mini-Batch Gradient Descent**: A compromise between Batch and Stochastic GD. It performs an update for every mini-batch of training examples. This is the most common implementation of gradient descent.

### Advanced Optimizers

While standard Gradient Descent is effective, several advanced optimizers have been developed to improve its performance:

-   **Momentum**: Helps accelerate SGD in the relevant direction and dampens oscillations.
-   **AdaGrad**: Adapts the learning rate for each parameter, performing larger updates for infrequent and smaller updates for frequent parameters.
-   **RMSprop**: Also adapts the learning rate, but resolves some of AdaGrad's issues with a rapidly diminishing learning rate.
-   **Adam (Adaptive Moment Estimation)**: The most popular optimizer. It combines the ideas of Momentum and RMSprop and is generally a good starting choice for most problems. 