# 05_BERT_and_Variants

BERT, which stands for Bidirectional Encoder Representations from Transformers, is a landmark model in Natural Language Processing. Unlike earlier models like GPT which were unidirectional, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

### How it works:

BERT is an **encoder-only** Transformer model. It is pre-trained on two novel unsupervised tasks:
1.  **Masked Language Model (MLM)**: Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a `[MASK]` token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked words in the sequence.
2.  **Next Sentence Prediction (NSP)**: The model receives pairs of sentences as input and learns to predict if the second sentence is the subsequent sentence in the original document.

### Fine-Tuning:

After pre-training, BERT can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

### Key Variants:

-   **RoBERTa (Robustly optimized BERT approach)**: A modification of BERT that optimizes key hyperparameters, resulting in a model that significantly outperforms the original. It removes the Next Sentence Prediction task and trains with much larger mini-batches and learning rates.
-   **DeBERTa (Decoding-enhanced BERT with disentangled attention)**: Improves upon BERT and RoBERTa models using two novel techniques: a disentangled attention mechanism and an enhanced mask decoder. 