---
title: Generative Models
track: general
level: applied
status: draft
module_id: generative-models
implementation: pytorch-with-keras-alternative
---

# Generative Models

A generative model learns to produce new samples that resemble its training
data. A generative adversarial network (GAN) does so by training two
networks against each other: a generator that turns random noise into
images, and a discriminator that tries to tell generated images from real
ones.

The notebooks train a deep convolutional GAN (DCGAN) on MNIST for 50 epochs
and follow a fixed set of noise vectors through training, so the same
samples can be compared from epoch to epoch. They also explain why the GAN
losses, unlike the loss of a classifier, say little about sample quality.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)
