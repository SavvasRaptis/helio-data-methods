# Statistical Modeling and Machine Learning in Heliophysics

This site is a practical collection of tutorials, worked examples, and
references for statistical modeling and machine learning in heliophysics. The notebooks use a focused scientific Python stack. Native PyTorch is the
main neural-network path, with concise Keras 3 alternatives running on the
same Torch backend. NumPy, pandas, Matplotlib, scikit-learn, XGBoost, and other
specialized tools appear where they fit the problem. The
[Software Toolkit](general-ml/foundations/software-toolkit/index.md) gives a
brief introduction to what each package does.

## Why AI and ML matter here

Heliophysics and space weather present an unusual data challenge: decades of
observations coexist with sparse sampling across enormous regions and
relatively few examples of the most extreme events. The recent Eos article
[“Vast Space, Sparse Data”](https://eos.org/science-updates/vast-space-sparse-data-an-ai-answer-to-twin-space-weather-challenges)
discusses how AI and machine learning can complement physical understanding
under these conditions. Community efforts such as
[LMAG25](https://www.lmag25.com/) bring together heliophysicists, geospace and
space-weather researchers, forecasters, and machine-learning specialists to
develop useful, interpretable, and carefully validated approaches. This
collection offers a practical entry point into that broader effort.

## Run the notebooks

Notebook pages provide an **Open in Colab** button for running an example in
the browser. For a reproducible local environment, install
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
open a terminal in the repository, and run:

```bash
conda env create -f environment.yml
conda activate helio-data-methods
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --frozen --group notebooks
jupyter lab
```

The first command creates the environment; the remaining commands activate
it, install the locked notebook dependencies, and open JupyterLab or JupyterNotebook.

## Explore the material

### [General ML](general-ml/index.md)

Introductions to neural networks, convolutional models, tree models, transfer
learning, tuning, and generative models. The notebooks include suggestions
that can be explored interactively in Colab.

### [Statistical Modeling](statistical-modeling/index.md)

A space for future examples on classical statistical models, uncertainty, and
time-series methods.

### [Heliophysics](heliophysics/index.md)

Applied examples from Heliophysics including Dst prediction, SEP occurrence, coronal loop reconstruction, and plasmasheet data modeling.

### [Resources](resources/index.md)

A curated set of books, interactive references, lectures, and exercises for
going further than simple demonstrations.
