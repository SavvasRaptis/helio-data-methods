# SEP Occurrence Forecasting with XGBoost

Three notebooks use gradient-boosted trees on the same split as the neural
networks.

- [XGBoost classifier](xgboost/demo.ipynb): training with early stopping,
  verification scores at three thresholds, and bootstrap intervals.
- [Repeated validation](xgboost/validation.ipynb): how much the scores vary
  across 15 cross-validation folds.
- [SHAP attributions](xgboost/interpretability.ipynb): which predictors the
  model relies on, and whether that ranking survives a change of random seed.
