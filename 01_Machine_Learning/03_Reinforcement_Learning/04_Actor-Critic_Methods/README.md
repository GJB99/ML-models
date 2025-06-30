# 04_Actor-Critic_Methods

Actor-Critic methods are a family of reinforcement learning algorithms that combine the strengths of both value-based methods (like Q-learning) and policy-based methods (like REINFORCE).

### How they work:

The architecture consists of two components, which are typically neural networks:
1.  **The Actor**: This is the policy part of the algorithm. It controls how the agent behaves by learning a policy that maps states to actions.
2.  **The Critic**: This is the value function part of the algorithm. It measures how good the action taken by the Actor is by estimating a value function (like the Q-value or the state-value).

The learning process is a feedback loop:
-   The **Actor** takes an action based on the current state.
-   The **Critic** evaluates this action by computing a value (e.g., the TD error), which tells the Actor how much better or worse its action was than expected.
-   The **Actor** updates its policy based on this feedback from the Critic, encouraging actions that led to better-than-expected outcomes.
-   The **Critic** updates its value function based on the reward received from the environment, improving its ability to evaluate the Actor's actions.

This separation allows for more stable training than policy-based methods alone, as the Critic's value estimate reduces the high variance of the policy gradient.

### Key Variants:

-   **A3C (Asynchronous Advantage Actor-Critic)**: Uses multiple worker agents in parallel to explore the environment, providing a stream of diverse experiences that helps to stabilize training.
-   **PPO (Proximal Policy Optimization)**: A state-of-the-art actor-critic method that improves training stability by using a clipped objective function to prevent the policy from changing too much in one update.
-   **SAC (Soft Actor-Critic)**: An off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework. It encourages exploration by adding an entropy term to the objective, leading to highly efficient and robust performance. 