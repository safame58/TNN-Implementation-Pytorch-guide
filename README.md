# Tensorized Neural Networks (TNN) Pytorch Tutorial

This repository provides a minimal PyTorch implementation of tensorized neural networks using:

- Tucker decomposition (for convolution layers)
- Matrix Product Operator (MPO) layers (for linear layers)

## Experiment

We compare:

- Standard CNN
- Tensorized CNN (TNN)

on MNIST.

## Key Results

- Comparable accuracy
- Significant parameter reduction

## Run

Open the notebook: jupyter notebook tnn_mnist.ipynb


## Notes

- MPO is implemented explicitly via tensor contraction
- This is a minimal educational example (not optimized)

## Purpose

This repository accompanies our position paper "Position: Tensorization is a powerful but underexplored tool for compression and interpretability of neural networks".
