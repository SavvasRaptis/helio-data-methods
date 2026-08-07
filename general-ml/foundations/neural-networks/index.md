---
title: Neural Networks
track: general
level: foundation
status: draft
module_id: neural-networks
implementation: pytorch-with-keras-alternative
---

# Neural Networks

A neural network learns a sequence of transformations from inputs to
predictions. Dense layers combine the inputs using learned weights and biases,
while nonlinear activation functions such as ReLU allow the network to learn
relationships that cannot be represented by one linear transformation.

The example uses MNIST, a collection of small grayscale images of handwritten
digits from 0 to 9. The task is to assign each image to the correct digit. The
model keeps the structure of the source example:

```text
Flatten → 200 ReLU → 150 ReLU → Dropout(0.5) → 10 logits
```

Both notebooks use the same data split, architecture, five-epoch training
budget, and diagnostic figures.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)
