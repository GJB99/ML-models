# 04_Convolutional_Neural_Networks

Convolutional Neural Networks (CNNs or ConvNets) are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are known for their ability to automatically and adaptively learn spatial hierarchies of features from input images.

### Key Layers and Concepts:

-   **Convolutional Layer**: The core building block of a CNN. The layer's parameters consist of a set of learnable filters (or kernels), which have a small receptive field, but extend through the full depth of the input volume.
-   **Pooling (or Subsampling) Layer**: This layer is responsible for reducing the spatial size (width and height) of the convolved features. This decreases the computational power required to process the data through dimensionality reduction. Common methods are Max Pooling and Average Pooling.
-   **Fully Connected Layer**: After several convolutional and pooling layers, the high-level reasoning in the neural network is done via fully connected layers. Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks.
-   **Strides and Padding**:
    -   **Stride**: The number of pixels shifts over the input matrix.
    -   **Padding**: Adding pixels to the border of an image to allow for more space for the filter to cover the image.

### Common CNN Architectures:

-   LeNet-5
-   AlexNet
-   VGGNet
-   ResNet
-   InceptionNet 