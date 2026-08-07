---
title: Convolutional Neural Networks
track: general
level: foundation
status: draft
module_id: convolutional-neural-networks
implementation: pytorch-with-keras-alternative
---

# Convolutional Neural Networks

Convolutional neural networks preserve the spatial structure of an image.
Their filters are applied across the image to learn local patterns, while
pooling reduces spatial resolution and helps build progressively higher-level
features.

These notebooks classify the same MNIST digits as the dense-network example,
but use two convolution and pooling stages before the final classifier. They
share the same split and evaluation figures so the implementation remains easy
to compare.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)
