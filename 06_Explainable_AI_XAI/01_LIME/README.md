# 01_LIME

LIME, which stands for **Local Interpretable Model-agnostic Explanations**, is a technique that explains the predictions of any classifier or regressor in an interpretable and faithful manner by approximating it locally with an interpretable model.

### The Core Idea:

It's often hard to understand the complex decision boundary of a high-performance model (like a deep neural network or a gradient boosted tree). Instead of trying to understand the entire global behavior, LIME focuses on explaining *individual predictions*.

### How it works:

1.  **Select an Instance**: Choose the specific prediction you want to explain.
2.  **Perturb the Instance**: Generate a new dataset of perturbed samples around the instance of interest. For text, this could mean removing words; for images, it could mean hiding super-pixels.
3.  **Get Predictions**: Get predictions on these perturbed samples using the complex, "black box" model.
4.  **Weight the Samples**: Assign weights to the new samples based on their proximity to the original instance.
5.  **Train an Interpretable Model**: Train a simple, interpretable model (like a linear regression or decision tree) on this new dataset with the proximity weights. The features for this simple model are the interpretable components (e.g., words or super-pixels).
6.  **Explain the Prediction**: The learned interpretable model can now be used to explain the individual prediction. For example, the coefficients of the linear model show which features (words/pixels) contributed most to the prediction.

Because LIME is **model-agnostic**, it can be applied to any machine learning model without needing to understand its internal workings. 