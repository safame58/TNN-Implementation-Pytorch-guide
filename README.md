# Tensorized Neural Networks (TNN) Pytorch Tutorial

This repository provides a minimal PyTorch implementation of tensorized neural networks using:

- Tucker decomposition (for convolution layers)
- Matrix Product Operator (MPO) layers (for linear layers)

## Experiment and results:

We Compare between two architectures CNN and TNN:

### Dense CNN
We use a simple convolutional neural network composed of:
- One convolutional layer (1 → 32 channels, 3×3 kernel)
- ReLU activation
- Max-pooling (downsampling by 2)
- A fully connected layer for classification (10 classes)

This serves as the baseline model.


### Tensorized Neural Network (TNN)

The tensorized model follows the same architecture but replaces layers with tensorized versions:

- **TuckerConv (convolution layer):**
  - Factorizes the convolution into:
    - Input projection (1×1 convolution)
    - Core convolution (3×3)
    - Output projection (1×1 convolution)

- **MPO Linear Layer:**
  - The flattened feature vector is reshaped into a higher-order tensor
  - The weight matrix is represented as a sequence of tensor cores
  - This provides a structured and parameter-efficient representation

We obtained the following results:

- The tensorized model maintains high accuracy with only a small performance drop
- It achieves over **93% parameter reduction**
- This demonstrates that tensorization can significantly compress neural networks while preserving performance

<img width="826" height="374" alt="image" src="https://github.com/user-attachments/assets/cf7723c1-a6fd-4e3c-92ea-8fed243863ed" />

## Run

Open the notebook: jupyter notebook tnn_mnist.ipynb

## Notes

- MPO is implemented explicitly via tensor contraction
- This is a minimal educational example (not optimized)

## Purpose

This repository accompanies our position paper "Position: Tensorization is a powerful but underexplored tool for compression and interpretability of neural networks".
