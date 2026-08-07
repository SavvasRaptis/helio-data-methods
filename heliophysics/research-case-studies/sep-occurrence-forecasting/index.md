---
title: SEP Occurrence Forecasting
track: heliophysics
level: research
status: draft
module_id: sep-occurrence-forecasting
implementation: mixed-model-case-study
artifacts:
  - pytorch/demo
  - keras/demo
  - xgboost/demo
  - xgboost/validation
  - xgboost/interpretability
---

# SEP Occurrence Forecasting

This example is adapted from Aminalragia-Giamini et al. (2021),
[*Solar Energetic Particle Event occurrence prediction using Solar Flare Soft
X-ray measurements and Machine Learning*](https://www.swsc-journal.org/articles/swsc/full_html/2021/01/swsc210024/swsc210024.html).

The saved data contain 49 numerical features, binary labels, and an
existing train/test assignment. The notebooks can therefore demonstrate sample-level classification, class imbalance, validation, and SHAP mechanics, but they cannot establish event-aware performance or physical feature attribution. For a more detailed discussion contact the authors.

- [Native PyTorch neural workflow](pytorch/demo.ipynb)
- [Keras 3 neural alternative using the Torch backend](keras/demo.ipynb)
- [XGBoost, sample-level validation, and SHAP](xgboost.md)

Reference: S. Aminalragia-Giamini et al. (2021), *Journal of Space Weather and Space Climate*, 11, 59.
