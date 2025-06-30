# Gradient Boosting Machines (GBMs)

Gradient Boosting is a powerful machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion and generalizes other boosting methods by allowing optimization of an arbitrary differentiable loss function.

This section covers several of the most popular and powerful Gradient Boosting implementations:
-   [**XGBoost**](./01_XGBoost/): The standard for performance in tabular competitions.
-   [**CatBoost**](./02_CatBoost/): Excellent handling of categorical features.
-   [**LightGBM**](./03_LightGBM/): The fastest implementation.
-   [**FastTree**](./04_FastTree/): An optimized implementation for the .NET ecosystem.
-   [**H2O GBM**](./05_H2O_GBM/): A highly scalable implementation for distributed environments.

### Key Concept: Boosting

Boosting trains models sequentially, with each new model focusing on correcting the errors made by its predecessors. This is achieved through a few key mechanisms:

-   **Sequential Learning**: Models are built one after another, each one learning from the previous one's mistakes.
-   **Adaptive Weighting**: In some boosting algorithms like AdaBoost, difficult examples (those that were misclassified) are given higher weights in subsequent iterations.
-   **Gradient Descent Optimization**: In Gradient Boosting, new models are fit to the residual errors of the previous model, effectively performing gradient descent on the loss function.

This systematic focus on misclassified examples is what allows boosting algorithms to build highly accurate "strong learners" from simple "weak learners" (typically decision trees). 