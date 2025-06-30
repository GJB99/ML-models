# State Space Models

State Space Models (SSMs) are a class of sequential models that map an input sequence to a latent state sequence, from which an output is generated. Recently, they have emerged as a powerful and efficient alternative to Transformer-based architectures for long-sequence modeling. Models like Mamba leverage a selective SSM mechanism to achieve linear scaling with sequence length, a significant advantage over the quadratic complexity of attention.

This section covers:
- [**Mamba**](./01_Mamba/): The foundational selective state space model.
- [**Vision Mamba (Vim)**](./02_Vision_Mamba/): An adaptation of Mamba for computer vision tasks. 