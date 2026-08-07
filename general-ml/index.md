---
title: General ML
track: general
level: foundation
status: draft
module_id: general-ml-track
implementation: none
---

# General ML

These tutorials introduce simple machine-learning examples primarely using neural networks with familiar datasets typically used for pedagogical purposes, keeping the focus on implementation.

## Topics

- [Software Toolkit](foundations/software-toolkit/index.md): a quick map of the
  Python packages and data structures used in the examples.
- [Neural Networks](foundations/neural-networks/index.md) and
  [Convolutional Neural Networks](foundations/convolutional-neural-networks/index.md):
  complete MNIST workflows using native PyTorch and Keras on the Torch backend.
- [CIFAR-10 CNN Progression](advanced/cifar10-cnn-progression/index.md),
  [Transfer Learning](advanced/transfer-learning/index.md), and
  [Generative Models](advanced/generative-models/index.md): progressively more
  involved vision examples.
- [Tree Models and Ensembles](advanced/tree-models/index.md) and
  [Hyperparameter Tuning](advanced/hyperparameter-tuning/index.md): boosted
  trees and some examples for hyperparameter tuning.

Native PyTorch is the canonical neural-network implementation. The shorter
Keras 3 notebooks use the same Torch runtime and show the equivalent
high-level `compile()` and `fit()` workflow that one may prefer to use for simplicity. The notebooks also include ideas to experiment interactively.
