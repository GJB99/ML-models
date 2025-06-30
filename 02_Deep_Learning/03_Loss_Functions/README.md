# 03_Loss_Functions

A loss function (or cost function) is a method of evaluating how well your algorithm models your dataset. If your predictions are totally off, your loss function will output a higher number. If they're pretty good, it'll output a lower number. As you change pieces of your algorithm to try and improve your model, your loss function will tell you if you're getting warmer.

### Common Loss Functions for Regression:

-   **Mean Squared Error (MSE) / L2 Loss**: Calculates the average of the squares of the errors. It is one of the most common loss functions, but it is sensitive to outliers.
-   **Mean Absolute Error (MAE) / L1 Loss**: Calculates the average of the absolute differences between the target values and the predicted values. It is more robust to outliers than MSE.
-   **Huber Loss**: A combination of MSE and MAE. It is less sensitive to outliers than MSE but is still differentiable at 0.

### Common Loss Functions for Classification:

-   **Binary Cross-Entropy**: Used for binary classification tasks. It measures the performance of a classification model whose output is a probability value between 0 and 1.
-   **Categorical Cross-Entropy**: Used for multi-class classification tasks. It is a generalization of binary cross-entropy.
-   **Hinge Loss**: Primarily used with Support Vector Machines (SVMs). It is intended for "maximum-margin" classification. 