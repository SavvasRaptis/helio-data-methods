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

A decision tree splits the data on thresholds of one feature at a time.
Gradient boosting builds an ensemble of shallow trees, each fitted to the
errors of the trees before it. Boosted trees, implemented here with XGBoost,
are often the strongest method for tabular data such as solar-wind
measurements or event catalogues.

The notebook applies XGBoost to MNIST with each image flattened to 784 pixel
values, so the comparison with the convolutional network shows the cost of
ignoring spatial structure. A short experiment then compares two tree depths
on validation error and runtime.

- [XGBoost notebook](xgboost/demo.ipynb)
