# 01_Q_Learning

Q-Learning is a model-free, off-policy reinforcement learning algorithm that seeks to find the best action to take given the current state. It's considered off-policy because the q-learning function learns from actions that are outside the current policy, like taking random actions, and therefore a policy is not needed.

### How it works:

Q-Learning is based on the Bellman equation. The core idea is to maintain a table of Q-values for each state-action pair. The Q-value represents the "quality" of an action taken from a state.

The formula to update the Q-value is:
`Q(state, action) = Q(state, action) + α * (R(state, action) + γ * max(Q(new_state, all_actions)) - Q(state, action))`

Where:
-   **α (alpha)** is the learning rate.
-   **γ (gamma)** is the discount factor.
-   **R** is the reward.

### Key Concepts:

-   **Q-Table**: A matrix where the rows represent states and the columns represent actions.
-   **Exploration vs. Exploitation**: The agent needs to balance exploring new actions to find potentially better rewards with exploiting known actions to maximize its current reward. This is often handled with an epsilon-greedy strategy. 