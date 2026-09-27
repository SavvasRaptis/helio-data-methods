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

Solar energetic particle (SEP) events are bursts of high-energy protons
accelerated by flares and coronal mass ejections; they endanger astronauts
and spacecraft and can arrive within tens of minutes. This example is adapted
from Aminalragia-Giamini et al. (2021),
[*Solar Energetic Particle Event occurrence prediction using Solar Flare Soft
X-ray measurements and Machine Learning*](https://www.swsc-journal.org/articles/swsc/full_html/2021/01/swsc210024/swsc210024.html),
which predicts SEP occurrence from soft X-ray flare measurements.

The saved data hold 49 standardized predictors per sample, a binary label,
and a fixed train/test assignment. Events make up about 1.3% of the samples.
The archive has no column names, event identifiers, or timestamps, which sets
two limits: the predictors cannot be interpreted physically, and samples from
the same event may fall in both the training and the test set, so all scores
are probably optimistic. The notebooks show the methods for a rare-event
problem (class weighting, verification scores, threshold choice, bootstrap
uncertainty, and attribution) within those limits.

- [PyTorch neural network](pytorch/demo.ipynb)
- [Keras 3 neural network](keras/demo.ipynb)
- [XGBoost, repeated validation, and SHAP](xgboost.md)

Reference: S. Aminalragia-Giamini et al. (2021), *Journal of Space Weather and
Space Climate*, 11, 59.
