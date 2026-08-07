---
title: Generative Models
track: general
level: applied
status: draft
module_id: generative-models
implementation: pytorch-with-keras-alternative
---

# Generative Models

A generative adversarial network trains two models together. A generator turns
random noise into images, while a discriminator tries to distinguish generated
images from real training images.

These notebooks preserve the source DCGAN structure for MNIST, including the
latent dimension, batch size, optimizer choices, and fixed-noise image grids
used to follow training progress.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)
