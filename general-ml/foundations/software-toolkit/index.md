---
title: Software Toolkit
track: general
level: foundation
status: draft
module_id: software-toolkit
implementation: none
---

# Software Toolkit

The examples in this book use a small collection of Python packages. You do
not need to master them before starting. It is enough to know which job each
package performs and to recognize the main data structures as they move
through a modeling workflow. However, reading their documentation will be beneficial in the long run.

## Jupyter notebooks and Google Colab

A Jupyter notebook mixes explanatory text, executable Python, figures, and
saved results. Run cells from top to bottom because later cells usually depend
on variables created earlier. Google Colab runs the same kind of notebook in a
temporary cloud environment, so downloaded data and installed extras may need
to be recreated when the runtime restarts.

## Package map

| Package | Typical import | Role in this book |
| --- | --- | --- |
| [NumPy](https://numpy.org/doc/stable/) | `import numpy as np` | Numerical arrays and vectorized calculations |
| [pandas](https://pandas.pydata.org/docs/) | `import pandas as pd` | Dataframes, timestamps, and tabular data cleaning |
| [Matplotlib](https://matplotlib.org/stable/) | `import matplotlib.pyplot as plt` | Scientific Figures |
| [scikit-learn](https://scikit-learn.org/stable/user_guide.html) | `from sklearn import ...` | Data splits, preprocessing, baselines, metrics, and diagnostic displays |
| [PyTorch](https://pytorch.org/docs/stable/) | `import torch` | Canonical neural betwork models, and automatic differentiation |
| [Keras 3](https://keras.io/) | `import keras` | A higher-level model API using the same PyTorch backend |
| [XGBoost](https://xgboost.readthedocs.io/) | `import xgboost as xgb` | Gradient-boosted tree models for tabular and flattened inputs |
| [Optuna](https://optuna.readthedocs.io/) / [KerasTuner](https://keras.io/keras_tuner/) | `import optuna` / `import keras_tuner` | Validation-based hyperparameter searches |
| [SHAP](https://shap.readthedocs.io/) | `import shap` | Model-behavior diagnostics that require careful interpretation |

## Other useful packages

The following packages are also particularly useful to use in future exampels or experimentation.

| Package | Typical import | When it may be useful |
| --- | --- | --- |
| [tslearn](https://tslearn.readthedocs.io/en/stable/) | `import tslearn` | Time-series distances, clustering, classification, and related learning tools |
| [sktime](https://www.sktime.net/) | `import sktime` | A unified interface for forecasting, classification, transformation, and other time-series tasks |
| [Seaborn](https://seaborn.pydata.org/) | `import seaborn as sns` | Higher-level statistical graphics built on Matplotlib |
| [LightGBM](https://lightgbm.readthedocs.io/en/stable/) | `import lightgbm as lgb` | Efficient gradient-boosted tree models, especially for tabular data, similar to XGBoost |
| [imbalanced-learn](https://imbalanced-learn.org/stable/) | `import imblearn` | Resampling methods, pipelines, and metrics for imbalanced classification problems |
| [statsmodels](https://www.statsmodels.org/stable/index.html) | `import statsmodels.api as sm` | Statistical models, hypothesis tests, classical time-series methods, and detailed inference summaries |


## Three common data representations

A `pandas.DataFrame` keeps column names and timestamps, which is helpful while
auditing and cleaning scientific tables. NumPy arrays provide compact
numerical matrices for preprocessing and many classical models.

```python
import pandas as pd
import torch

frame = pd.DataFrame({"speed": [400.0, 525.0], "bz": [-2.0, 4.0]})
array = frame[["speed", "bz"]].to_numpy(dtype="float32")
tensor = torch.from_numpy(array)
```

## PyTorch is the canonical path

PyTorch exposes the important steps explicitly: create tensors, define a
model, calculate a loss, backpropagate, update parameters, and evaluate with
gradients disabled. Keras 3 offers a shorter `compile()` and `fit()` workflow while still using PyTorch underneath. The backend must be selected before importing Keras:

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras

assert keras.backend.backend() == "torch"
```

The Keras notebooks are alternative implementations, not a second modeling
method. Their data boundaries, architecture intent, metrics, and scientific
interpretation are almost identical to the PyTorch workflow.
