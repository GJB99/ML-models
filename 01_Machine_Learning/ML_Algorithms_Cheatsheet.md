# Machine Learning Algorithms Cheatsheet

## Overview
This comprehensive cheatsheet covers the most important machine learning algorithms, their characteristics, use cases, and practical considerations for real-world applications.

---

## Supervised Learning Algorithms

### Linear Regression
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Regression |
| **Best Use Case** | Predicting continuous values with linear relationships |
| **Key Formula** | Y = b0 + b1X1 + b2X2 + ... |
| **Assumptions** | Linearity, independence, homoscedasticity |
| **Pros** | Simple, interpretable, fast, good baseline |
| **Cons** | Sensitive to outliers, assumes linearity, limited flexibility |
| **When NOT to Use** | Non-linear relationships, complex patterns |
| **Real-World Example** | House price prediction, sales forecasting |

### Logistic Regression
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Classification |
| **Best Use Case** | Binary classification with probability interpretation |
| **Key Formula** | P = 1 / (1 + e^-(b0 + b1X + ...)) |
| **Assumptions** | Log-odds linearity, independence |
| **Pros** | Probabilistic output, interpretable, no tuning needed |
| **Cons** | Assumes linear decision boundary, sensitive to outliers |
| **When NOT to Use** | Highly non-linear data, complex feature interactions |
| **Real-World Example** | Spam detection, medical diagnosis |

### Decision Trees
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Classification/Regression |
| **Best Use Case** | Problems requiring interpretability and rule extraction |
| **Key Formula** | Recursive binary splits based on feature thresholds |
| **Assumptions** | None (non-parametric) |
| **Pros** | Highly interpretable, handles mixed data types, no preprocessing |
| **Cons** | Prone to overfitting, unstable, biased splits |
| **When NOT to Use** | Need stable predictions, small datasets |
| **Real-World Example** | Loan approval, medical diagnosis |

### Random Forest
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Ensemble |
| **Best Use Case** | High accuracy with good generalization |
| **Key Formula** | Bootstrap aggregating + feature randomness |
| **Assumptions** | Tree independence (approximate) |
| **Pros** | High accuracy, robust to overfitting, handles missing values |
| **Cons** | Less interpretable, slower inference, memory intensive |
| **When NOT to Use** | Real-time applications, interpretability critical |
| **Real-World Example** | Feature selection, fraud detection |

### Gradient Boosting
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Ensemble |
| **Best Use Case** | Maximizing predictive performance |
| **Key Formula** | Sequential weak learners minimizing residual loss |
| **Assumptions** | Sequential dependency between learners |
| **Pros** | Excellent accuracy, handles mixed data, feature importance |
| **Cons** | Prone to overfitting, requires tuning, computationally expensive |
| **When NOT to Use** | Small datasets, real-time constraints |
| **Real-World Example** | Kaggle competitions, credit scoring |

### Support Vector Machine (SVM)
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Classification/Regression |
| **Best Use Case** | High-dimensional data, clear margin separation |
| **Key Formula** | Maximize margin using kernel transformations |
| **Assumptions** | Separability (with kernels), feature scaling important |
| **Pros** | Effective in high dimensions, memory efficient, versatile |
| **Cons** | Slow on large datasets, sensitive to scaling, no probability estimates |
| **When NOT to Use** | Very large datasets, noisy data |
| **Real-World Example** | Text classification, image recognition |

### K-Nearest Neighbors (KNN)
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Classification/Regression |
| **Best Use Case** | Simple problems with local patterns |
| **Key Formula** | Distance-based majority voting |
| **Assumptions** | Local similarity, appropriate distance metric |
| **Pros** | Simple implementation, no training phase, works with small datasets |
| **Cons** | Computationally expensive, sensitive to irrelevant features |
| **When NOT to Use** | High-dimensional data, real-time applications |
| **Real-World Example** | Recommendation systems, pattern recognition |

### Naive Bayes
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Classification |
| **Best Use Case** | Text classification, categorical features |
| **Key Formula** | P(class|features) = P(features|class) × P(class) / P(features) |
| **Assumptions** | Feature independence (naive assumption) |
| **Pros** | Fast training and prediction, works well with small data |
| **Cons** | Strong independence assumption, poor with continuous features |
| **When NOT to Use** | Features are highly dependent |
| **Real-World Example** | Spam filtering, sentiment analysis |

---

## Unsupervised Learning Algorithms

### K-Means Clustering
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Unsupervised - Clustering |
| **Best Use Case** | Customer segmentation, spherical clusters |
| **Key Formula** | Minimize intra-cluster distance to centroids |
| **Assumptions** | Spherical clusters, similar cluster sizes |
| **Pros** | Fast, easy to implement, scales well |
| **Cons** | Requires choosing K, sensitive to initialization |
| **When NOT to Use** | Non-spherical clusters, varying cluster sizes |
| **Real-World Example** | Market segmentation, image compression |

### Hierarchical Clustering
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Unsupervised - Clustering |
| **Best Use Case** | Understanding data structure, creating dendrograms |
| **Key Formula** | Agglomerative or divisive tree building |
| **Assumptions** | Appropriate distance metric exists |
| **Pros** | No need to specify cluster count, creates hierarchy |
| **Cons** | Computationally expensive O(n³), sensitive to noise |
| **When NOT to Use** | Very large datasets |
| **Real-World Example** | Phylogenetic analysis, organizational structure |

### Principal Component Analysis (PCA)
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Unsupervised - Dimensionality Reduction |
| **Best Use Case** | Reducing feature dimensionality, visualization |
| **Key Formula** | Eigenvalue decomposition of covariance matrix |
| **Assumptions** | Linear relationships, features are centered |
| **Pros** | Reduces overfitting, removes multicollinearity, speeds up algorithms |
| **Cons** | Less interpretable features, assumes linearity |
| **When NOT to Use** | Interpretability is crucial, non-linear relationships |
| **Real-World Example** | Image compression, data visualization |

### Autoencoders
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Unsupervised - Dimensionality Reduction/Generation |
| **Best Use Case** | Complex pattern learning, anomaly detection |
| **Key Formula** | Neural network with bottleneck architecture |
| **Assumptions** | Sufficient data for training, appropriate architecture |
| **Pros** | Handles non-linear relationships, flexible architecture |
| **Cons** | Requires large datasets, computationally expensive |
| **When NOT to Use** | Small datasets, simple linear relationships |
| **Real-World Example** | Anomaly detection, data denoising |

### DBSCAN
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Unsupervised - Density-based Clustering |
| **Best Use Case** | Finding arbitrarily shaped clusters, handling noise |
| **Key Formula** | Density-based spatial clustering with noise |
| **Assumptions** | Clusters have sufficient density |
| **Pros** | Finds arbitrary shapes, robust to outliers, auto-determines clusters |
| **Cons** | Sensitive to hyperparameters, struggles with varying densities |
| **When NOT to Use** | High-dimensional sparse data |
| **Real-World Example** | Geospatial clustering, fraud detection |

---

## Deep Learning Algorithms

### Multilayer Perceptron (MLP)
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Neural Network |
| **Best Use Case** | Complex non-linear pattern recognition |
| **Key Formula** | Weighted sums with activation functions |
| **Assumptions** | Sufficient data, appropriate architecture |
| **Pros** | Universal approximator, handles complex patterns |
| **Cons** | Requires large datasets, prone to overfitting |
| **When NOT to Use** | Small datasets, interpretability needed |
| **Real-World Example** | Image classification, function approximation |

### Convolutional Neural Network (CNN)
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Deep Learning |
| **Best Use Case** | Image/video analysis, spatial data |
| **Key Formula** | Convolution + pooling + fully connected layers |
| **Assumptions** | Grid-like spatial structure, translation invariance |
| **Pros** | Excellent for images, parameter sharing, translation invariant |
| **Cons** | Computationally expensive, requires large datasets |
| **When NOT to Use** | Non-spatial data, limited compute resources |
| **Real-World Example** | Computer vision, medical imaging |

### Recurrent Neural Network (RNN)
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised - Deep Learning |
| **Best Use Case** | Sequential data, time series |
| **Key Formula** | Hidden state updates over time steps |
| **Assumptions** | Sequential structure in data |
| **Pros** | Handles variable-length sequences, memory capability |
| **Cons** | Vanishing gradient problem, computationally expensive |
| **When NOT to Use** | Non-sequential data, very long sequences |
| **Real-World Example** | Language modeling, time series prediction |

### Transformer
| **Attribute** | **Details** |
|---------------|-------------|
| **Type** | Supervised/Self-supervised - Deep Learning |
| **Best Use Case** | NLP tasks, sequence-to-sequence problems |
| **Key Formula** | Self-attention mechanism with position encoding |
| **Assumptions** | Large training datasets, sufficient compute |
| **Pros** | Parallelizable, captures long-range dependencies, state-of-the-art results |
| **Cons** | Requires massive datasets and compute resources |
| **When NOT to Use** | Small datasets, limited computational resources |
| **Real-World Example** | Language translation, chatbots (GPT, BERT) |

---

## Algorithm Selection Framework

### By Problem Type
| **Problem Type** | **Recommended Algorithms** |
|------------------|----------------------------|
| **Linear Relationships** | Linear/Logistic Regression |
| **Non-linear Patterns** | Random Forest, Gradient Boosting, Neural Networks |
| **High Dimensions** | SVM, PCA + other algorithms |
| **Large Datasets** | SGD-based algorithms, Random Forest |
| **Small Datasets** | KNN, Naive Bayes, Simple models |
| **Need Interpretability** | Decision Trees, Linear Models |
| **Time Series** | ARIMA, LSTM, Prophet |
| **Images** | CNN, Transfer Learning |
| **Text** | Transformers, Naive Bayes, TF-IDF + ML |

### By Data Characteristics
| **Data Characteristic** | **Best Algorithms** |
|-------------------------|---------------------|
| **Mixed Data Types** | Tree-based methods, Gradient Boosting |
| **Missing Values** | Random Forest, XGBoost |
| **Outliers Present** | Tree-based methods, Robust algorithms |
| **High Noise** | Ensemble methods, Regularized models |
| **Imbalanced Classes** | Cost-sensitive learning, SMOTE + ML |

### Performance vs. Interpretability
```
High Performance, Low Interpretability:
└── Gradient Boosting, Deep Learning, Ensemble Methods

Balanced:
└── Random Forest, SVM

High Interpretability, Moderate Performance:
└── Decision Trees, Linear Models, Naive Bayes
```

## Best Practices
1. **Start Simple**: Begin with baseline models (Linear/Logistic Regression)
2. **Understand Your Data**: Exploratory analysis before algorithm selection
3. **Cross-Validate**: Always use proper validation techniques
4. **Feature Engineering**: Often more important than algorithm choice
5. **Ensemble Methods**: Combine multiple models for better performance
6. **Monitor Overfitting**: Balance bias-variance trade-off
7. **Scale Features**: Important for distance-based and gradient-based algorithms

---

*This cheatsheet provides a practical guide for selecting and applying machine learning algorithms based on problem characteristics and requirements.* 