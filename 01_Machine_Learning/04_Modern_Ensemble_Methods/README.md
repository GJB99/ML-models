# Modern Ensemble Methods

Ensemble methods combine multiple machine learning models to achieve better predictive performance than any single model could. The core idea is that a diverse group of learners can make better decisions than a single expert.

The two most foundational ensemble techniques are **Bagging** and **Boosting**. More advanced methods like **Stacking** and **Blending** build on these core ideas.

### Bagging (Bootstrap Aggregating)

Bagging focuses on reducing variance. It involves training multiple independent models in parallel on different random subsets of the training data (selected with replacement). The predictions from each model are then aggregated, typically by voting (for classification) or averaging (for regression).

-   **Key Benefit**: Reduces overfitting and improves stability.
-   **Famous Example**: Random Forest.

### Boosting

Boosting focuses on reducing bias. It involves training models sequentially, where each new model is trained to correct the errors made by its predecessors. Incorrectly classified examples are given more weight in subsequent models.

-   **Key Benefit**: Can build very powerful and accurate models.
-   **Famous Examples**: AdaBoost, Gradient Boosting Machines (GBM).

### Stacking (Stacked Generalization)

Stacking is an ensemble learning technique that uses a meta-model to learn how to best combine the predictions from two or more base models.

#### How it works:
1.  **Split the data**: The training data is split into folds (similar to cross-validation).
2.  **Train Base Models**: A set of diverse "Level 0" base models (e.g., a Random Forest, an SVM, a KNN) are trained on the folds of the training data.
3.  **Create a New Dataset**: The predictions made by these base models on the hold-out folds are collected. These "out-of-fold" predictions become the features for a new dataset.
4.  **Train a Meta-Model**: A "Level 1" meta-model (e.g., Logistic Regression) is trained on this new dataset to learn the optimal way to combine the base model predictions.

### Blending

Blending is a simpler variant of stacking. Instead of using k-fold cross-validation, it uses a single holdout validation set to generate the predictions for training the meta-model. It's faster but can be more prone to overfitting.

## Covered Methods

-   [**Stacking**](./01_Stacking/): Meta-learning approach to combine multiple base models
-   [**Voting Classifiers**](./02_Voting_Classifiers/): Simple averaging or voting of model predictions
-   [**NODE**](./03_NODE/): Neural Oblivious Decision Ensembles
-   [**LinearBoost**](./04_LinearBoost/): Linear boosting techniques
-   [**EBM**](./05_EBM/): **TabArena #9** - Explainable Boosting Machine combining performance with interpretability (Elo: 1300) 