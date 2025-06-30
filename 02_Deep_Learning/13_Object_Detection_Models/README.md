# Object Detection Models

Object detection is a computer vision task that involves identifying and locating objects within an image or video. Object detection models predict a bounding box around each object of interest, along with a class label for that object.

This section covers several key architectures:

-   **R-CNN (Region-based CNN)** and its faster successors, **Fast R-CNN** and **Faster R-CNN**, which are two-stage detectors that first propose regions of interest and then classify them.
-   **YOLO (You Only Look Once)**, which is a family of single-stage detectors that treat object detection as a single regression problem, directly predicting bounding boxes and class probabilities from the full image in one pass. This makes YOLO extremely fast and suitable for real-time applications.
-   **SSD (Single Shot MultiBox Detector)**, another popular single-stage detector.

### Key Architectures:

-   **R-CNN (Region-based CNN)**: The original R-CNN model proposed a multi-stage approach: first, generate region proposals using an algorithm like Selective Search, then extract features from each region using a CNN, and finally classify the regions with an SVM. It was slow and cumbersome.

-   **Fast R-CNN**: Improved upon R-CNN by feeding the entire image to the CNN to generate a feature map. Region proposals are then projected onto this feature map, which speeds up the process significantly.

-   **Faster R-CNN**: This was a major breakthrough. It introduced the **Region Proposal Network (RPN)**, a fully convolutional network that predicts object bounds and "objectness" scores at each position. This replaced the slow Selective Search algorithm and allowed the model to be trained end-to-end.

-   **Mask R-CNN**: Extends Faster R-CNN to the task of **instance segmentation**. In addition to predicting a class and bounding box, Mask R-CNN also outputs a high-quality segmentation mask for each instance. It does this by adding a parallel branch to the network that predicts a binary mask for each Region of Interest (RoI). 