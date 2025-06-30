# 02_Transformers_Architecture

The Transformer is a novel network architecture introduced in the paper "Attention Is All You Need." It has become the foundation for most state-of-the-art Large Language Models. Unlike RNNs, Transformers do not process data sequentially and instead rely entirely on a self-attention mechanism.

### Key Components:

-   **Self-Attention**: The core innovation of the Transformer. It allows the model to weigh the importance of different words in the input sequence when processing a particular word. This enables it to capture long-range dependencies more effectively than RNNs.
-   **Multi-Head Attention**: An extension of self-attention where the attention mechanism is run multiple times in parallel. This allows the model to jointly attend to information from different representation subspaces at different positions.
-   **Positional Encodings**: Since the model contains no recurrence, it needs some way to make use of the order of the sequence. Positional encodings are added to the input embeddings to give the model information about the relative or absolute position of the tokens in the sequence.
-   **Encoder-Decoder Structure**: The original Transformer had an encoder stack to process the input sequence and a decoder stack to generate the output sequence.
    -   **Encoder-only models** (like BERT) are good for understanding-based tasks (e.g., classification).
    -   **Decoder-only models** (like GPT) are good for generative tasks.
-   **Feed-Forward Networks**: Each layer in the encoder and decoder contains a fully connected feed-forward network. 