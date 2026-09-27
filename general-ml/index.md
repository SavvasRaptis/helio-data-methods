---
title: General ML
track: general
level: foundation
status: draft
module_id: general-ml-track
implementation: none
---

# General ML

These chapters introduce the methods used later in the book on standard
teaching datasets, where the answer is known and the focus can stay on the
method. They build on one another, but each notebook runs on its own.

## Foundations

- [Software Toolkit](foundations/software-toolkit/index.md): the Python
  packages used in the examples and what each one does.
- [Data Splits and Leakage](foundations/data-splits-and-leakage/index.md):
  training, validation, and test sets, and how information leaks between them.
- [Model Evaluation](foundations/model-evaluation/index.md): baselines,
  confusion matrices, and the metrics used throughout the book.
- [Neural Networks](foundations/neural-networks/index.md): a dense network
  for handwritten digits, with every step of the training loop explained.
- [Convolutional Neural Networks](foundations/convolutional-neural-networks/index.md):
  the same task with a network that exploits the image's spatial structure.

## Further methods

- [CIFAR-10 CNN Progression](advanced/cifar10-cnn-progression/index.md):
  a harder image task, a deeper network, and overfitting.
- [Tree Models and Ensembles](advanced/tree-models/index.md): gradient-boosted
  trees on flattened images.
- [Transfer Learning](advanced/transfer-learning/index.md): reusing an
  ImageNet network when labelled data are scarce.
- [Hyperparameter Tuning](advanced/hyperparameter-tuning/index.md): choosing
  architecture and learning rate on validation data.
- [Generative Models](advanced/generative-models/index.md): a generative
  adversarial network that produces new digits.

Each neural-network chapter has a PyTorch notebook, which writes the training
loop out explicitly, and a Keras 3 notebook, which runs the same model through
the shorter `compile()` and `fit()` interface on the same PyTorch backend.
Each notebook ends with suggestions for changes to try.
