---
title: Convolutional Neural Networks
track: general
level: foundation
status: draft
module_id: convolutional-neural-networks
implementation: pytorch-with-keras-alternative
---

# Convolutional Neural Networks

A dense network treats an image as an unordered list of pixels. A
convolutional network instead slides small learned filters across the image,
so a feature such as an edge or a stroke is detected wherever it appears, and
pooling layers reduce the resolution so that later filters respond to larger
patterns. The same filter weights are shared across all positions, which
makes the network far more data-efficient on images.

These notebooks classify the same MNIST digits, with the same split, as the
[Neural Networks](../neural-networks/index.md) chapter, using two convolution
and pooling stages before the dense classifier. The convolutional network
makes fewer than half as many test errors as the dense one.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)
