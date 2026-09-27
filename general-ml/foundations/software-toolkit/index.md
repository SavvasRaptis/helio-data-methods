---
title: Software Toolkit
track: general
level: foundation
status: draft
module_id: software-toolkit
implementation: none
---

# Software Toolkit

The examples use a small set of Python packages. You do not need to know
them before starting; it is enough to know what each one is for and to
recognize the main data structures as they pass through a workflow. Their
documentation, linked below, is worth reading as you go further.

## Jupyter notebooks and Google Colab

A Jupyter notebook mixes text, executable Python, figures, and saved results.
Run the cells from top to bottom, because later cells use variables created
by earlier ones. Google Colab runs the same notebooks in a temporary cloud
environment; downloaded data and installed packages disappear when the
runtime restarts, and the notebooks fetch them again.

## Packages used in the book

| Package | Typical import | Role in this book |
| --- | --- | --- |
| [NumPy](https://numpy.org/doc/stable/) | `import numpy as np` | Arrays and vectorized arithmetic |
| [pandas](https://pandas.pydata.org/docs/) | `import pandas as pd` | Tables with named columns and timestamps |
| [Matplotlib](https://matplotlib.org/stable/) | `import matplotlib.pyplot as plt` | Figures |
| [scikit-learn](https://scikit-learn.org/stable/user_guide.html) | `from sklearn import ...` | Data splits, preprocessing, and metrics |
| [PyTorch](https://pytorch.org/docs/stable/) | `import torch` | Neural networks and automatic differentiation |
| [Keras 3](https://keras.io/) | `import keras` | A higher-level neural-network API, run here on PyTorch |
| [XGBoost](https://xgboost.readthedocs.io/) | `import xgboost as xgb` | Gradient-boosted trees |
| [Optuna](https://optuna.readthedocs.io/) / [KerasTuner](https://keras.io/keras_tuner/) | `import optuna` / `import keras_tuner` | Hyperparameter search |
| [SHAP](https://shap.readthedocs.io/) | `import shap` | Attributing a model's predictions to its inputs |

## Other useful packages

These are not used in the notebooks yet but are worth knowing.

| Package | Typical import | Use |
| --- | --- | --- |
| [statsmodels](https://www.statsmodels.org/stable/index.html) | `import statsmodels.api as sm` | Regression with full inference, hypothesis tests, and classical time-series models |
| [sktime](https://www.sktime.net/) | `import sktime` | One interface for time-series forecasting, classification, and transformation |
| [tslearn](https://tslearn.readthedocs.io/en/stable/) | `import tslearn` | Time-series distances, clustering, and classification |
| [imbalanced-learn](https://imbalanced-learn.org/stable/) | `import imblearn` | Resampling and metrics for rare-class problems |
| [LightGBM](https://lightgbm.readthedocs.io/en/stable/) | `import lightgbm as lgb` | Gradient-boosted trees, often faster than XGBoost on large tables |
| [Seaborn](https://seaborn.pydata.org/) | `import seaborn as sns` | Statistical graphics built on Matplotlib |
| [SunPy](https://sunpy.org/) | `import sunpy` | Solar data search, download, and coordinate handling |
| [cdasws](https://cdaweb.gsfc.nasa.gov/WebServices/py/cdasws/) | `from cdasws import CdasWs` | Programmatic access to NASA CDAWeb, including OMNI |

## Three data representations

A pandas `DataFrame` keeps column names and timestamps, which helps while
inspecting and cleaning scientific tables. A NumPy array is a plain numerical
matrix, the input most preprocessing and classical models expect. A PyTorch
tensor is the array a neural network consumes; it can live on a GPU and
record the operations needed for gradients. Converting between them is one
line each:

```python
import pandas as pd
import torch

frame = pd.DataFrame({"speed": [400.0, 525.0], "bz": [-2.0, 4.0]})
array = frame[["speed", "bz"]].to_numpy(dtype="float32")
tensor = torch.from_numpy(array)
```

## PyTorch and Keras

PyTorch makes each step of training explicit: build tensors, define a model,
compute a loss, backpropagate, update the weights, and evaluate with
gradients switched off. Keras 3 wraps the same steps in `compile()` and
`fit()`, and in this book it runs on the PyTorch backend. The backend must be
chosen before Keras is imported:

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras

assert keras.backend.backend() == "torch"
```

The Keras notebooks use the same data, split, architecture, and metrics as
the PyTorch notebooks, so their results should agree closely. Small
differences come from the frameworks' different default weight
initializations and random-number streams.
