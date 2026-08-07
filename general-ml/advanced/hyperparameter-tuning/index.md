---
title: Hyperparameter Tuning
track: general
level: applied
status: draft
module_id: hyperparameter-tuning
implementation: pytorch-with-keras-alternative
---

# Hyperparameter Tuning

Hyperparameters are choices made before training, such as layer width and
learning rate. A tuning library evaluates several configurations using the
validation set and records which choices work best within a fixed search
budget.

The PyTorch notebook uses Optuna and the Keras-on-Torch notebook uses
KerasTuner. Both search the same compact MNIST CNN space and evaluate the
selected model once on the test set.

- [Complete PyTorch and Optuna workflow](pytorch/demo.ipynb)
- [KerasTuner alternative using the Torch backend](keras/demo.ipynb)
