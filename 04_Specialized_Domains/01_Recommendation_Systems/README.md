# 01_Recommendation_Systems

Recommendation Systems are one of the most popular applications of machine learning in the industry. They are designed to predict the "rating" or "preference" a user would give to an item.

### Core Approaches:

-   **Collaborative Filtering**: This method builds a model from a user's past behaviors (items previously purchased or rated) as well as similar decisions made by other users. The core idea is that if user A has the same opinion as user B on an issue, A is more likely to have B's opinion on a different issue than that of a randomly chosen person.
    -   **User-Based Collaborative Filtering**: Find users who are similar to the target user and recommend items that those similar users liked. Cosine similarity is often used to compare user preference vectors to identify users with similar tastes.
    -   **Item-Based Collaborative Filtering**: Recommend items that are similar to the items the target user has liked in the past. Here, items are compared based on user interaction patterns, again often using cosine similarity.

-   **Content-Based Filtering**: This method uses the attributes of an item to recommend other items with similar properties. For example, if a user has watched many action movies, the system will recommend other action movies.

-   **Matrix Factorization**: A class of collaborative filtering algorithms used in recommender systems. Matrix factorization models work by decomposing the user-item interaction matrix into the product of two lower-dimensionality rectangular matrices.
    -   **Singular Value Decomposition (SVD)**: A popular matrix factorization technique.

-   **Deep Learning for Recommender Systems**: Modern recommenders often use deep learning to capture more complex patterns.
    -   **Neural Collaborative Filtering**: Replaces the standard matrix factorization dot product with a neural network to learn the user-item interaction function.

### Measuring Similarity: Cosine Similarity

Cosine similarity plays a pivotal role in modern recommender systems by measuring the similarity between users or items based on their vector representations. This mathematical measure calculates the cosine of the angle between two vectors in multi-dimensional space, producing values between -1 and 1, where values closer to 1 indicate high similarity.

For example, in user-based collaborative filtering, if two users have similar rating patterns regardless of their rating scales, their vectors will align closely, yielding high cosine scores. Similarly, in item-based filtering, if two books are frequently purchased by the same users, their vectors in the user-purchase matrix will align closely. 