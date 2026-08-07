---
title: CIFAR-10 CNN Progression
track: general
level: applied
status: draft
module_id: cifar10-cnn-progression
implementation: pytorch-with-keras-alternative
---

# CIFAR-10 CNN Progression

CIFAR-10 contains small color images from ten classes and is more varied than
MNIST. This example moves from a simple two-stage convolutional network to the
deeper model represented in the source material, adding convolutional stages,
normalization, dropout, and a larger classifier.

The same training, validation, and test examples are used throughout so the
simple and advanced models can be viewed as one clear progression.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)
