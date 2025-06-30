# Naive Bayes Classification

Naive Bayes represents a family of simple "probabilistic classifiers" based on applying Bayes' theorem with the strong (and often naive) assumption that the features are conditionally independent. Despite its simplicity, Naive Bayes often performs surprisingly well in real-world applications, especially in text classification.

### Mathematical Foundation

The algorithm applies Bayes' theorem to find the probability of a class `y` given a set of features `x1, ..., xn`:
$$ P(y|x_1, \ldots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i|y)}{P(x_1, \ldots, x_n)} $$

The "naive" assumption—that all features are independent of each other given the class—dramatically simplifies the computation of the `P(features | class)` term.

### Practical Advantages

-   **High Speed**: Extremely fast to train and classify compared to more sophisticated methods.
-   **Small Data Requirements**: Requires a relatively small amount of training data to estimate the necessary parameters.
-   **Performs Well with High-Dimensional Data**: It tends to work well on datasets with many features (high dimensionality), such as text.
-   **Proven Effectiveness**: Works particularly well for document classification and spam filtering.

### Common Types of Naive Bayes:

-   **Gaussian Naive Bayes**: Used for continuous features, assuming they follow a Gaussian (normal) distribution.
-   **Multinomial Naive Bayes**: Used for discrete counts, common in text classification where the features are word counts or frequencies.
-   **Bernoulli Naive Bayes**: Used for binary/boolean features (i.e., features that are either present or absent).

### Use Cases:

-   Spam filtering
-   Text classification (e.g., sentiment analysis, topic categorization)
-   Real-time prediction 