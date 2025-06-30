# 02_Deep_Q_Networks

Deep Q-Networks (DQN) are a type of reinforcement learning algorithm that combines Q-learning with deep neural networks. This allows the agent to learn in environments with high-dimensional state spaces (like image-based environments).

### How it works:

Instead of using a Q-table to store Q-values, DQN uses a neural network to approximate the Q-value function. The state is given as input to the network, and the output is the Q-value for each possible action.

### Key Innovations:

-   **Experience Replay**: To break the correlation between consecutive samples, DQN stores the agent's experiences (state, action, reward, next_state) in a replay buffer. The network is then trained on a random mini-batch of these experiences.
-   **Target Network**: To stabilize learning, a second, separate "target" network is used to generate the target Q-values for the Bellman equation. This target network's weights are updated less frequently than the main Q-network. 