Overview:

A hybrid deep neural architecture combining EfficientNet-B0, multi-residual refinement, LSTM temporal modeling, and an Extreme Learning Machine (ELM) classifier for efficient and accurate land-use/land-cover classification on satellite imagery.

Core Architecture:

1. Backbone: EfficientNet-B0
   Input: 128×128×3 RGB image

   Pretrained: ImageNet weights

   Output: 1280-channel feature map 

   Purpose: Efficient baseline feature extraction with optimized depth/width scaling


2. Multi-Residual Refinement Blocks

   Two sequential residual blocks with projection shortcuts:

   Block 1: 1280 → 64 channels, 3×3 conv, MaxPool2d(2), Dropout(0.3)

   Block 2: 64 → 128 channels, Dropout(0.4)

   Projection: 1×1 convolutions align dimensions for residual addition

   Feature Fusion: fused = residual_output + projected_input


3. Multi-Level Feature Fusion
   Upsample refined features (128 channels) to match backbone spatial dimensions

   Concatenate with original EfficientNet features → 1408 channels

   Reduce via 1×1 convolution → 256 channels → BatchNorm → ReLU


4. Temporal Context Modeling
   Global Average Pooling → 256-dimensional vector

   Reshape to sequence format: (batch, 1, 256)

   Single-layer LSTM (256 units) captures contextual dependencies


5. Classification: Extreme Learning Machine (ELM)
   Hidden layer: Random fixed weights/biases

   Output layer: Closed-form solution (implemented as trainable linear layer)

   Output: 10-class probabilities (EuroSAT categories)


   Key Innovations:

   Multi-Residual Learning: Enhanced gradient flow and feature representation via projection shortcuts

   Cross-Scale Fusion: Combines deep backbone features with refined shallow features

   Temporal Modeling: LSTM captures sequential patterns in global features

   Lightweight Classification: ELM enables fast, non-iterative training of final layer


   Performance Advantages:

   Efficiency: ~7.8M trainable parameters

   Accuracy: Enhanced feature learning through residual pathways


Scalability: ELM enables rapid training compared to iterative deep classifiers

Robustness: Multi-level fusion improves spatial feature representation

Application
Optimized for satellite image classification tasks, particularly the EuroSAT & UC mercedland dataset, balancing computational efficiency with high classification accuracy.