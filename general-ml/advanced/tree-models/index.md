---
title: Tree Models and Ensembles
track: general
level: applied
status: draft
module_id: tree-models
implementation: framework-neutral
library: xgboost
---

# Tree Models and Ensembles

Decision trees divide a feature space using learned thresholds. XGBoost builds
an ensemble sequentially, with each new tree helping correct errors made by the
existing collection of trees.

This framework-neutral notebook flattens the MNIST images and applies XGBoost
directly. A short **Example thought** compares two maximum tree depths using
validation error and runtime.

- [Complete XGBoost workflow](xgboost/demo.ipynb)
