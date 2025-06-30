# U-Net

The U-Net architecture is a convolutional neural network (CNN) that was originally developed for biomedical image segmentation. Its architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. The key feature is the use of "skip connections" that concatenate feature maps from the contracting path with the corresponding feature maps in the expanding path.

This design has made U-Net the de-facto standard for a wide variety of image segmentation tasks. Furthermore, its powerful structure has been adapted for use as the core denoising network in modern high-performance Diffusion Models.

### How it works:

The U-Net architecture consists of two main paths:
1.  **The Contracting Path (Encoder)**: This is a typical CNN stack, consisting of repeated convolutional and max pooling layers. Its purpose is to capture the context in the image. The feature map size is reduced at each step, while the number of feature channels is increased.
2.  **The Expansive Path (Decoder)**: This path uses transposed convolutions (or up-convolutions) to upsample the feature maps. Its purpose is to enable precise localization. A key feature of U-Net is the use of **skip connections**, which concatenate the feature maps from the contracting path with the upsampled feature maps from the expansive path.

### Why it's effective:

The skip connections are crucial. They combine the high-level, semantic feature maps from the decoder with the low-level, high-resolution feature maps from the encoder. This allows the network to make precise predictions (localization) while still using the learned context from the deeper layers. It is highly effective for segmentation tasks, especially with limited data, as is common in the medical field. 