# 09_Regularized_Regression

Regularized regression methods are variants of linear regression that include a penalty term in the loss function to prevent overfitting. This is particularly useful when dealing with a large number of features or multicollinearity.

### Key Algorithms:

-   **Lasso Regression (L1 Regularization)**: Adds a penalty equal to the absolute value of the magnitude of coefficients. This can shrink some coefficients to exactly zero, effectively performing feature selection.
-   **Ridge Regression (L2 Regularization)**: Adds a penalty equal to the square of the magnitude of coefficients. This shrinks the coefficients towards zero, but they never become exactly zero. It is useful for handling multicollinearity.
-   **Elastic Net**: A combination of both L1 and L2 regularization. It has a mixing parameter to control the blend of the two penalties. 