---
title: Transfer Learning
track: general
level: applied
status: draft
module_id: transfer-learning
implementation: pytorch-with-keras-alternative
---

# Transfer Learning

Transfer learning reuses a network trained on one large dataset as a starting
point for a different task. Here the convolutional layers of VGG16, trained
on the 1.3 million images of ImageNet, are frozen and used as a feature
extractor for CIFAR-10, and only a small classifier on top is trained.

The benefit is largest when labelled examples are few, which is the usual
situation for rare events in heliophysics. The notebooks therefore train on
1,000, 5,000, and 45,000 images and compare the transfer model with a small
network trained from scratch on the same images.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)
