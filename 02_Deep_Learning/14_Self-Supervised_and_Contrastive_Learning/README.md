# Self-Supervised and Contrastive Learning

Self-supervised learning is a paradigm where a model learns representations from unlabeled data by solving a pretext task. Contrastive learning is a popular approach within this paradigm.

### Contrastive Learning
The core idea of contrastive learning is to learn an embedding space where similar sample pairs are pulled closer together and dissimilar sample pairs are pushed apart.

#### SimCLR (Simple Contrastive Learning)
SimCLR is a prominent framework that advanced the state-of-the-art in self-supervised visual representation learning. It works by:
1.  Creating two different augmented views of the same image (a positive pair).
2.  Treating all other images in the batch as negative samples.
3.  Training a model to maximize the agreement between the positive pair using a contrastive loss function.

Through careful design of data augmentations and a non-linear projection head, SimCLR learns representations that are highly effective for downstream tasks.

This section also covers:
- SimSiam
- MoCo (Momentum Contrast)
- BYOL (Bootstrap Your Own Latent)

### Key Frameworks:

-   **SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)**: Uses a large batch size to generate many negative samples and employs a learnable nonlinear projection head on top of the encoder to improve the quality of the learned representations.
-   **MoCo (Momentum Contrast)**: Improves upon SimCLR by maintaining a queue of negative samples, allowing it to use a much larger set of negatives without requiring a huge batch size. It uses a momentum-updated encoder to keep the representations in the queue consistent.
-   **BYOL (Bootstrap Your Own Latent)**: A non-contrastive approach that learns by predicting the representation of one augmented view of an image from the representation of another augmented view, without using any negative samples.
-   **Masked Autoencoders (MAE)**: A self-supervised method particularly for Vision Transformers. It masks a large portion of the input image's patches and trains the model to reconstruct the missing patches. 