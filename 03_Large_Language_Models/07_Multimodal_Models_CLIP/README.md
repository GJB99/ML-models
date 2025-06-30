# 07_Multimodal_Models_CLIP

Multimodal models are a class of AI models that can understand and process information from multiple data types, or "modalities," such as text, images, video, and audio. A key example is connecting text and images.

### CLIP (Contrastive Language-Image Pre-training)

CLIP is a revolutionary multimodal model developed by OpenAI that learns visual concepts from natural language supervision. It is trained on a massive dataset of image and text pairs collected from the internet.

The core idea is to train an image encoder and a text encoder jointly using a contrastive objective. The model learns to predict which caption goes with which image in a batch. By doing so, it learns a shared embedding space where corresponding images and text descriptions are located close to each other.

This training paradigm allows CLIP to perform zero-shot classification on a wide variety of tasks without any direct training on those tasks. For example, to classify an image, you can simply embed the image and compare its similarity to the embeddings of text prompts like "a photo of a dog" or "a photo of a cat".

### How it works:

CLIP is trained on a massive dataset of (image, text) pairs collected from the internet. It has two main components:
1.  **Image Encoder**: A model (like a Vision Transformer or ResNet) that processes an image and outputs a vector representation (embedding).
2.  **Text Encoder**: A model (like a Transformer) that processes a text description and also outputs a vector embedding.

During training, the model learns to project these image and text embeddings into a shared latent space. The goal is to maximize the **cosine similarity** between the embeddings of *correct* (image, text) pairs, while minimizing the similarity between incorrect pairs. This is a form of **contrastive learning**.

### Zero-Shot Classification:

The power of CLIP is its ability to perform **zero-shot classification**. You can provide it with an image and a list of potential text descriptions (e.g., "a photo of a dog," "a photo of a cat," "a photo of a car"). CLIP will embed the image and all the text descriptions, and then predict the description with the highest cosine similarity to the image embedding, without ever having been explicitly trained on that specific classification task. 