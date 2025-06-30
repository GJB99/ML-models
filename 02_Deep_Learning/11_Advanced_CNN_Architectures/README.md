# 11_Advanced_CNN_Architectures

While architectures like AlexNet and VGGNet were foundational, modern computer vision relies on more advanced and efficient CNN designs that enable much deeper networks.

### Key Architectures:

-   **ResNet (Residual Network)**: The key innovation of ResNet is the "identity shortcut connection" or "skip connection." These connections allow the gradient to be directly backpropagated to earlier layers, making it possible to train networks that are hundreds or even thousands of layers deep without suffering from the vanishing gradient problem.
    -   **ResNeXt**: An evolution of ResNet that introduces the concept of "cardinality" (the number of parallel pathways), providing a more effective way to increase model capacity than just going deeper or wider.
    -   **Wide ResNet**: Argues that wider and shallower residual networks can be more effective and efficient than the very deep, thin ones.

-   **EfficientNet**: Proposes a new, principled method for model scaling called "compound scaling." Instead of scaling network dimensions (width, depth, resolution) arbitrarily, EfficientNet uniformly scales each dimension with a fixed set of scaling coefficients. This provides a better balance between accuracy and efficiency.

-   **RegNet (Designing Network Design Spaces)**: Rather than focusing on designing individual networks, RegNet aims to design the *design space* of networks itself. It discovers general principles of network design that lead to simpler, better-performing models. 