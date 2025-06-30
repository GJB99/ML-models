# Generative Models

Generative models are a class of statistical models that are used to generate new data instances. Rather than discriminating between different kinds of data (like a classifier), generative models learn the underlying distribution of the data and can create new samples from that distribution.

This section covers several prominent generative modeling paradigms:

-   [**Autoencoders**](./01_Autoencoders/): Neural networks used for unsupervised learning of efficient codings.
-   [**Generative Adversarial Networks (GANs)**](./02_GANs/): A framework where two neural networks contest with each other in a zero-sum game.
-   [**Diffusion Models**](./03_Diffusion_Models/): Models that work by corrupting training data with gradual noise and then learning to reverse the process.
-   [**Normalizing Flows**](./04_Normalizing_Flows/): Construct complex distributions through a series of invertible transformations.
-   [**Energy-Based Models (EBMs)**](./05_Energy_Based_Models/): Model probability distributions through an energy function. 