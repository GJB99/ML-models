# Diffusion Models (DDPM/DDIM)

Denoising Diffusion Probabilistic Models (DDPMs) are a class of generative models that have revolutionized the field, achieving state-of-the-art results in image and audio generation. These models work by learning to reverse a gradual noising process.

The core idea involves two processes:
1.  **Forward Process:** Gradually add Gaussian noise to an image over a series of steps until it becomes pure noise.
2.  **Reverse Process:** Train a neural network (typically a U-Net) to reverse this process, starting from noise and gradually denoising it to generate a clean image.

Variants like Denoising Diffusion Implicit Models (DDIMs) offer faster sampling. This is the technology behind leading models like DALL-E 2, Midjourney, and Stable Diffusion. 