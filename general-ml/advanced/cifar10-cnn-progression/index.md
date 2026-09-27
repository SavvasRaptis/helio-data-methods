---
title: CIFAR-10 CNN Progression
track: general
level: applied
status: draft
module_id: cifar10-cnn-progression
implementation: pytorch-with-keras-alternative
---

# CIFAR-10 CNN Progression

CIFAR-10 contains 60,000 colour images of 32×32 pixels in ten classes, from
airplanes to trucks. Objects vary in pose, scale, and background, so the task
is much harder than MNIST. The notebooks train a small two-layer network and
a deeper network with batch normalization and dropout on the same split,
choose between them on validation accuracy, and evaluate only the chosen one
on the test set. The deeper network's learning curves show overfitting
setting in, and why restoring the best validation epoch matters.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)
