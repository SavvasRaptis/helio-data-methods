# Statistical Modeling and Machine Learning in Heliophysics

This book collects worked examples of statistical modeling and machine
learning, first on standard teaching datasets and then on heliophysics
problems: forecasting the Dst index, predicting solar energetic particle
events, reconstructing coronal loops, and modeling the plasma sheet. Every
example is a notebook that runs from start to finish, and each one reports
what its results can and cannot support.

## Why machine learning in heliophysics

Heliophysics has decades of observations, but they sample an enormous volume
sparsely, and the most extreme events, which matter most for space weather,
are the rarest. Machine learning can find structure in such data, but only if
the validation is as careful as the model. The Eos article
[“Vast Space, Sparse Data”](https://eos.org/science-updates/vast-space-sparse-data-an-ai-answer-to-twin-space-weather-challenges)
describes these challenges, and the [LMAG25](https://www.lmag25.com/) community
brings together heliophysicists, forecasters, and machine-learning researchers
to address them. The examples here are a practical entry point to that work.

## How the book is organized

- **[General ML](general-ml/index.md).** Data splits, evaluation, and a
  sequence of models on MNIST and CIFAR-10: dense and convolutional networks,
  boosted trees, transfer learning, hyperparameter tuning, and a generative
  adversarial network. Start here if the methods are new to you.
- **[Statistical Modeling](statistical-modeling/index.md).** Classical
  statistical models, uncertainty, and time-series methods. This part is
  still being written.
- **[Heliophysics](heliophysics/index.md).** Research-style examples in which
  the difficulties are physical: data gaps, time-ordered splits, rare events,
  physical baselines, and limited provenance.
- **[Resources](resources/index.md).** Books, courses, and lectures for going
  further.

## Run the notebooks

Each notebook page has an **Open in Colab** button (the rocket icon at the
top) that runs the notebook in the browser. To run locally, install
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html),
open a terminal in the repository, and run:

```bash
conda env create -f environment.yml
conda activate helio-data-methods
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --frozen --group notebooks
jupyter lab
```

The first two commands create and activate the environment, the third
installs the locked package versions, and the last opens JupyterLab.

PyTorch is the main framework, and each neural-network example also has a
shorter Keras 3 version running on the same PyTorch backend. The
[Software Toolkit](general-ml/foundations/software-toolkit/index.md) page
explains what each package does.
