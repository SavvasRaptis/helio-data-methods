---
title: Hyperparameter Tuning
track: general
level: applied
status: draft
module_id: hyperparameter-tuning
implementation: pytorch-with-keras-alternative
---

# Hyperparameter Tuning

Hyperparameters are the choices made before training, such as the number of
filters in a layer or the learning rate. A tuner trains one model per
candidate configuration and keeps the one with the best validation score.
The test set plays no part in the search.

The PyTorch notebook uses Optuna and the Keras notebook uses KerasTuner. Both
search the same small space of MNIST convolutional networks with a budget of
four trials, retrain the selected configuration, and evaluate it once on the
test set. With so few trials, the notebooks also show how little a small
difference in validation accuracy means.

- [PyTorch and Optuna notebook](pytorch/demo.ipynb)
- [Keras 3 and KerasTuner notebook](keras/demo.ipynb)
