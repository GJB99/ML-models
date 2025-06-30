# 08_Vision_Transformers_ViT

The Vision Transformer (ViT) represents a significant shift in computer vision, moving away from the traditional CNN-based approach. It applies the highly successful Transformer architecture, originally designed for NLP, to image classification tasks.

### How it works:

1.  **Image Patching**: The ViT model splits an input image into a sequence of fixed-size, non-overlapping patches.
2.  **Patch Embedding**: Each patch is flattened and linearly projected into an embedding vector.
3.  **Positional Encoding**: Just like in NLP, positional encodings are added to the patch embeddings to retain positional information.
4.  **Transformer Encoder**: The resulting sequence of vectors is fed into a standard Transformer encoder, which uses self-attention to weigh the importance of different patches relative to each other.
5.  **Classification Head**: A final classification head (typically a simple MLP) is attached to the output of the Transformer to make the final prediction.

### Significance:

ViT demonstrated that a pure Transformer architecture, without any convolutions, can achieve state-of-the-art results on image classification tasks, especially when pre-trained on very large datasets. 