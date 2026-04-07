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
- There exists many optimized TNNs that are able to maintain the dense model accuracy 
  and significantly reduce the parameters for more complex tasks, eg. check [this review]([tnn_mnist.ipynb](https://arxiv.org/pdf/2302.09019))

## Purpose

This repository accompanies our position paper "Position: Tensorization is a powerful but underexplored tool for compression and interpretability of neural networks".

## Extended Experiments (ResNet-34 Tensorization)

Below, we report empirical results from our work *“Activation-aware Tensorization and Compression of Neural Networks”* (to appear soon), demonstrating that tensorized models are both practical and effective at scale.

---

### Setup

We compare a standard CNN (**ResNet-34**, He et al., 2016) with its tensorized versions on **CIFAR-10** and **CIFAR-100**.

- Tensorization is performed using **Tucker decomposition**  
- Compression rate (CR) = 0.5 (50% parameter reduction at the model level)  
- Only **3×3 convolution layers** are tensorized  

After decomposition, models are **fine-tuned to recover accuracy** using two strategies:

#### Local approach (proposed)
- Feature-based distillation (MSE loss)
- Each tensorized layer learns to match the corresponding original layer
- The original ResNet-34 acts as a teacher model

#### Global approach
- End-to-end fine-tuning with cross-entropy loss
- Non-tensorized layers are frozen

---

### Experimental Environment

- AWS g5.8xlarge instance  
- 32 vCPUs, 128 GB RAM  
- NVIDIA A10G GPU (24 GB VRAM)

---

### CIFAR-10

#### Experiment 1: Top-1 Accuracy (CR = 0.5)

| Model                     | Accuracy (%) |
|--------------------------|-------------|
| ResNet-34                | 95.04       |
| T-ResNet-34 (50k local)  | 94.70       |
| T-ResNet-34 (30k local)  | 94.69       |
| T-ResNet-34 (10k local)  | 94.61       |
| T-ResNet-34 (50k global) | 89.47       |

#### Experiment 2: Fine-tuning Time (minutes)

| Method        | Time |
|---------------|------|
| Local (50k)   | 51.38 |
| Global (50k)  | 120.88 |

---

### CIFAR-100

#### Experiment 1: Top-1 Accuracy (CR = 0.5)

| Model                     | Accuracy (%) |
|--------------------------|-------------|
| ResNet-34                | 79.79       |
| T-ResNet-34 (50k local)  | 78.81       |
| T-ResNet-34 (30k local)  | 78.80       |
| T-ResNet-34 (10k local)  | 78.74       |
| T-ResNet-34 (50k global) | 68.19       |

#### Experiment 2: Fine-tuning Time (minutes)

| Method | Time |
|--------|------|
| Local  | 124.12 |
| Global | 145.99 |

---

### Key Observations

Tensorization provides several practical advantages:

1. **Accuracy preservation**  
   Maintains strong performance with minimal degradation (<1% for CR = 0.5).

2. **Efficiency**  
   Local fine-tuning is significantly faster than global fine-tuning  
   - **2.35× faster on CIFAR-10**  
   - **1.18× faster on CIFAR-100**

3. **Reduced training data requirements**  
   Strong performance is maintained even with smaller training datasets.

4. **Lower memory consumption**  
   Local tensorization processes layers independently:
   - Only one layer needs to be loaded into memory  
   - Enables efficient parallelization across multiple GPUs  

5. **Reduced model complexity**  
   Tensorized models use roughly **half the parameters** while achieving comparable accuracy.

---

### Takeaway

These results demonstrate that tensorization is not only a compression technique, but also supports efficient training strategies and scalable model design.
