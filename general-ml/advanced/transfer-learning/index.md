---
title: Transfer Learning
track: general
level: applied
status: draft
module_id: transfer-learning
implementation: pytorch-with-keras-alternative
---

# Transfer Learning

Transfer learning reuses features learned for one task as the starting point
for another. Here an ImageNet-pretrained VGG16 feature extractor is frozen and
a new classifier is trained for CIFAR-10.

The notebooks use the same CIFAR-10 split and classifier structure while
showing the native PyTorch workflow and the shorter Keras-on-Torch equivalent.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)
