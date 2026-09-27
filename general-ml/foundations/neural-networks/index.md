---
title: Neural Networks
track: general
level: foundation
status: draft
module_id: neural-networks
implementation: pytorch-with-keras-alternative
---

# Neural Networks

A neural network is a sequence of simple transformations with adjustable
weights. A dense layer forms weighted sums of its inputs; a nonlinear
activation such as ReLU, $\max(0, x)$, then lets the stack of layers
represent relationships that no single linear map could. Training adjusts
the weights by gradient descent to reduce a loss that measures how far the
predictions are from the known answers.

The example classifies MNIST, 70,000 grayscale images of handwritten digits,
28×28 pixels each. The network is

```text
784 pixels → 200 ReLU → 150 ReLU → dropout (0.5) → 10 logits
```

and it is trained for five epochs with the Adam optimizer and the
cross-entropy loss. It reads about 97.5% of the test digits correctly. The
PyTorch notebook explains each step of the training loop; the Keras notebook
runs the same model through `fit()`.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)
