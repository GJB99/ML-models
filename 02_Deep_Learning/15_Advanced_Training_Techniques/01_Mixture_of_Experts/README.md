# Mixture of Experts (MoE)

Mixture of Experts is a neural network architecture paradigm designed for building extremely large-scale models in a computationally efficient manner. Instead of a single, dense network that processes every input, an MoE model consists of numerous smaller "expert" sub-networks. For each input, a trainable gating network dynamically chooses which experts to activate.

This allows the model to have a massive number of parameters (trillions, in some cases) while keeping the computational cost for training and inference constant. This is the key technology behind models like Google's Switch Transformer and is reportedly used in GPT-4. 