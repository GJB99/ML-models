# Tabular Deep Learning

While Gradient Boosting Decision Trees (GBDTs) have traditionally dominated tabular data tasks, recent advancements in deep learning have produced a new class of models that show exceptional promise. These models adapt architectures like Transformers and ResNets, which have been highly successful in vision and NLP, to the unique challenges of structured, tabular data.

This section covers state-of-the-art deep learning models for tabular data:

## Foundation Models and Transformers
-   [**TabPFN**](./01_TabPFN/): A foundation model that shows remarkable performance on small tabular datasets.
-   [**TabNet**](./02_TabNet/): Uses a sequential attention mechanism for feature selection and interpretability.
-   [**FT-Transformer**](./03_FT-Transformer/): Adapts the Transformer architecture for tabular data.
-   [**TabM**](./06_TabM/): **TabArena #1** - State-of-the-art transformer-based model achieving highest performance (Elo: 1600).

## Advanced Neural Architectures
-   [**TabR**](./04_TabR/): Combines feed-forward networks with k-NN for improved performance.
-   [**ResNet for Tabular**](./05_ResNet_for_Tabular/): A specially adapted ResNet architecture for tabular data.
-   [**RealMLP**](./07_RealMLP/): **TabArena #2** - High-performance MLP architecture optimized for tabular data (Elo: 1580).
-   [**ModernNCA**](./08_ModernNCA/): **TabArena #6** - Modern Neural Component Analysis for structured representation learning (Elo: 1480).

## Probabilistic and Specialized Models
-   [**TabDPT**](./09_TabDPT/): **TabArena #8** - Diffusion Probabilistic Transformer with uncertainty quantification (Elo: 1350).
-   [**TorchMLP**](./10_TorchMLP/): **TabArena #7** - PyTorch-based MLP implementation for tabular data (Elo: 1350).
-   [**FastaiMLP**](./11_FastaiMLP/): **TabArena #10** - Fast.ai framework implementation for tabular neural networks (Elo: 1300). 