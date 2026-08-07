# Statistical Modeling and Machine Learning in Heliophysics

[![Book](https://img.shields.io/badge/read-Jupyter%20Book-4c72b0)](https://savvasraptis.github.io/helio-data-methods/)
[![CI](https://github.com/SavvasRaptis/helio-data-methods/actions/workflows/ci.yml/badge.svg)](https://github.com/SavvasRaptis/helio-data-methods/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE-CODE)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-lightgrey.svg)](LICENSE-CONTENT)

A practical collection of tutorials, reproducible examples, and references for
statistical modeling and machine learning in heliophysics. The material is
organized by topic and can be explored in any order.

## Explore

- **General ML** introduces the software toolkit and demonstrates neural
  networks, convolutional networks, tree models, transfer learning,
  hyperparameter tuning, and generative models.
- **Statistical Modeling** provides a home for future material on regression,
  uncertainty, classical models, and time-series analysis.
- **Heliophysics** applies these methods to Dst forecasting, SEP occurrence,
  coronal-loop reconstruction, and plasma-sheet modeling.
- **Resources** collects recommended books, courses, and supporting libraries.

[Read the Jupyter Book](https://savvasraptis.github.io/helio-data-methods/) or
[open the introductory PyTorch notebook in Colab](https://colab.research.google.com/github/SavvasRaptis/helio-data-methods/blob/main/general-ml/foundations/neural-networks/pytorch/demo.ipynb).

PyTorch is the primary teaching framework. Neural-network examples also include
concise Keras 3 alternatives using the PyTorch backend.

## Run locally

Python 3.11 is required. With Conda installed:

```bash
git clone https://github.com/SavvasRaptis/helio-data-methods.git
cd helio-data-methods
conda env create -f environment.yml
conda activate helio-data-methods
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --frozen --group notebooks
jupyter lab
```

To build and preview the book locally:

```bash
make book
make serve
```

Then open <http://localhost:8000>.

The heliophysics notebooks retrieve only their declared datasets, cache them
locally, and verify their SHA-256 checksums. Notebook cells are never executed
during a Jupyter Book build; verified outputs are stored in the notebooks.

## Reproducibility

The environment is locked with `uv.lock`. Continuous integration validates the
book structure, runs the test suite, builds the book with warnings treated as
errors, and checks every published notebook.

## License

- Code and supporting tooling: [MIT](LICENSE-CODE)
- Prose, notebooks, and figures: [CC BY 4.0](LICENSE-CONTENT)

## Author

Savvas Raptis — [website](https://savvasraptis.github.io) ·
[APL email](mailto:Savvas.raptis@jhuapl.edu) ·
[personal email](mailto:savvasraptis@pm.me)
