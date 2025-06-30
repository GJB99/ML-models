# 02_SHAP

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.

### The Core Idea:

SHAP aims to explain a prediction by computing the contribution of each feature to the prediction. It answers the question: how much did each feature contribute to the difference between the model's prediction and the base value (the average prediction over the entire dataset)?

The Shapley value is the average marginal contribution of a feature value across all possible coalitions (or subsets) of features.

### How it works:

-   For a given prediction, SHAP considers all possible subsets of features.
-   It trains a model on each subset and evaluates the model's output with and without the feature being evaluated.
-   The difference in output is the marginal contribution of that feature for that specific subset.
-   The Shapley value is the weighted average of all these marginal contributions.

### Advantages:

-   **Global and Local Interpretability**: SHAP values can be aggregated to understand the model's global behavior, but they also provide detailed local explanations for individual predictions.
-   **Solid Theoretical Foundation**: It is based on game theory and provides guarantees like efficiency and consistency that other methods lack.
-   **Versatility**: There are optimized solvers for specific model types (like `TreeExplainer` for tree-based models and `DeepExplainer` for deep learning models), making it efficient in practice. 