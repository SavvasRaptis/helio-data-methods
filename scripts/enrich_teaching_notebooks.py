"""Add concise exploration prompts to the published teaching notebooks.

The generators create complete workflows. This module appends a short
``Try it yourself`` section to each and adds the validation-only tree-depth
experiment to the XGBoost example.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Iterable

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEMOS = {
    "neural-networks": ROOT / "general-ml/foundations/neural-networks/pytorch/demo.ipynb",
    "convolutional-neural-networks": ROOT
    / "general-ml/foundations/convolutional-neural-networks/pytorch/demo.ipynb",
    "cifar10-cnn-progression": ROOT
    / "general-ml/advanced/cifar10-cnn-progression/pytorch/demo.ipynb",
    "tree-models": ROOT / "general-ml/advanced/tree-models/xgboost/demo.ipynb",
    "transfer-learning": ROOT / "general-ml/advanced/transfer-learning/pytorch/demo.ipynb",
    "hyperparameter-tuning": ROOT
    / "general-ml/advanced/hyperparameter-tuning/pytorch/demo.ipynb",
    "generative-models": ROOT / "general-ml/advanced/generative-models/pytorch/demo.ipynb",
    "dst-forecasting": ROOT
    / "heliophysics/applications/dst-forecasting/pytorch/demo.ipynb",
}
KERAS_DEMOS = {
    module_id: path.parent.parent / "keras" / "demo.ipynb"
    for module_id, path in DEMOS.items()
    if module_id != "tree-models"
}


def md(source: str, *tags: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(dedent(source).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def code(source: str, *tags: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def heading_index(notebook: nbformat.NotebookNode, heading: str) -> int:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip().startswith(heading):
            return index
    raise ValueError(f"missing heading {heading!r}")


def strip_retired_sections(notebook: nbformat.NotebookNode) -> None:
    """Remove previously generated course-style tail sections."""

    retired = (
        "## Interpretation",
        "## Conclusion",
        "## Controlled experiment",
        "## What changed?",
        "## Try it yourself",
        "## Try it yourself in Keras",
        "## Example thought",
        "## Experiment: tree depth",
    )
    cutoffs = []
    for heading in retired:
        try:
            cutoffs.append(heading_index(notebook, heading))
        except ValueError:
            pass
    if cutoffs:
        notebook.cells = notebook.cells[: min(cutoffs)]


PYTORCH_SUGGESTIONS = {
    "neural-networks": (
        "change dropout from `0.5` to `0.25` and compare the gap between training and validation accuracy",
        "halve the width of both hidden layers and compare the parameter count with the change in accuracy",
        "set the Adam learning rate to `1e-4` and to `1e-2` and compare the learning curves",
    ),
    "convolutional-neural-networks": (
        "change the first convolution from 32 to 64 filters and compare accuracy with parameter count",
        "compare 3×3 and 5×5 kernels; the dense layer's input size changes, so print the feature-map shape first",
        "remove the dropout layers and watch the gap between training and validation loss",
    ),
    "cifar10-cnn-progression": (
        "add random horizontal flips to the training images of the deeper network and compare the learning curves",
        "change the dropout rates of the deeper network and see where overfitting starts",
        "train the small network for 25 epochs to separate the effect of capacity from that of training time",
    ),
    "transfer-learning": (
        "add a 250-image subset to `SUBSET_SIZES` to see how far the frozen features carry",
        "replace the head with a single linear layer, a logistic regression on the VGG16 features",
        "unfreeze the last VGG16 block, train it with a learning rate of `1e-5`, and compare with the frozen model",
    ),
    "hyperparameter-tuning": (
        "rerun the search with a different sampler seed and see whether the same configuration wins",
        "increase `TRIALS` to 12 and check how much the best validation accuracy improves",
        "add dropout rates to the search space",
    ),
    "generative-models": (
        "set `LABEL_SMOOTHING` to `0.1` and compare the samples and losses",
        "halve the discriminator's learning rate and watch the balance of the two losses",
        "reduce `LATENT_DIM` to 10 and look for less variety in the samples",
    ),
    "dst-forecasting": (
        "change `HORIZON_HOURS` from 1 to 3 and then 6 while keeping the three-hour input history and 2015 as the final test year; persistence degrades quickly, so watch the skill score",
        "compare three- and six-hour input histories while keeping the one-hour forecast horizon fixed",
        "replace $B_z$ with the rectified coupling term $V B_s$, where $B_s = -B_z$ for southward field and 0 otherwise, and fit its scaling on training years only",
        "rebuild the data loading with NASA CDAWeb's [`cdasws` Python API](https://cdaweb.gsfc.nasa.gov/WebServices/py/cdasws/) and reproduce the hourly OMNI variables and time range used here",
    ),
}


def add_suggestions(
    module_id: str, notebook: nbformat.NotebookNode, *, keras: bool = False
) -> None:
    suggestions = PYTORCH_SUGGESTIONS[module_id]
    qualifier = " in Keras" if keras else ""
    bullets = "\n".join(f"- {suggestion}." for suggestion in suggestions)
    notebook.cells.append(
        md(
            f"""
## Try it yourself{qualifier}

Change one choice at a time and keep the data split and evaluation unchanged:

{bullets}
""",
            "try-it-yourself",
        )
    )


def add_xgboost_example_thought(notebook: nbformat.NotebookNode) -> None:
    notebook.cells.extend(
        [
            md(
                """
## Experiment: tree depth

How does maximum tree depth trade validation error against runtime? The two
models below use the same training and validation images and the same
50-round ceiling. The test set is not used.
""",
                "example-thought",
            ),
            code(
                """
EXAMPLE_ROUNDS = 50  # Reduce to 10 or 25 for a quicker comparison.
EXAMPLE_DEPTHS = [3, 6]
print(f"depths: {EXAMPLE_DEPTHS}; round ceiling: {EXAMPLE_ROUNDS}")
""",
                "example-thought",
            ),
            code(
                """
import time
import pandas as pd


def run_depth_example(depth):
    candidate_parameters = dict(parameters)
    candidate_parameters["max_depth"] = depth
    started = time.perf_counter()
    candidate = xgb.train(
        candidate_parameters,
        dtrain,
        num_boost_round=EXAMPLE_ROUNDS,
        evals=[(dvalidation, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    predictions = candidate.predict(dvalidation).argmax(axis=1)
    return {
        "configuration": f"depth={depth}",
        "validation_error": float(1.0 - accuracy_score(y_validation, predictions)),
        "runtime_seconds": float(time.perf_counter() - started),
        "boosting_rounds": int(candidate.best_iteration + 1),
        "max_depth": depth,
    }
""",
                "example-thought",
                "hide-input",
            ),
            code(
                """
example_results = [run_depth_example(depth) for depth in EXAMPLE_DEPTHS]
example_table = pd.DataFrame(example_results)
display(example_table)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].bar(example_table["configuration"], example_table["validation_error"])
axes[0].set(title="Validation error", ylabel="Classification error")
axes[1].bar(example_table["configuration"], example_table["runtime_seconds"])
axes[1].set(title="Runtime", ylabel="Seconds")
plt.tight_layout()
plt.show()

experiment_evidence = {
    "experiment_id": "maximum-tree-depth",
    "configurations": example_table.to_dict(orient="records"),
    "budget": {"rounds": EXAMPLE_ROUNDS, "mode": "compact"},
    "comparison_metrics": ["validation_error", "runtime_seconds", "boosting_rounds"],
    "test_used": False,
}
print("HELIO_EXPERIMENT " + json.dumps(experiment_evidence, sort_keys=True))
""",
                "example-thought",
            ),
            md(
                """
## Try it yourself

- Try depths 2, 4, and 8 and plot validation error against runtime.
- Hold the depth fixed and vary `eta` between `0.03` and `0.15`.
- Change `subsample` from `0.8` to `0.6` or `1.0` and rerun with a second seed to see how much the error moves.
""",
                "try-it-yourself",
            ),
        ]
    )


def enrich_all(module_ids: Iterable[str] | None = None) -> None:
    selected = set(module_ids or DEMOS)
    for module_id in selected:
        path = DEMOS[module_id]
        notebook = nbformat.read(path, 4)
        strip_retired_sections(notebook)
        if module_id == "tree-models":
            add_xgboost_example_thought(notebook)
        else:
            add_suggestions(module_id, notebook)
        nbformat.write(notebook, path)
        print(f"updated {path.relative_to(ROOT)}")

        keras_path = KERAS_DEMOS.get(module_id)
        if keras_path is not None:
            keras = nbformat.read(keras_path, 4)
            strip_retired_sections(keras)
            add_suggestions(module_id, keras, keras=True)
            nbformat.write(keras, keras_path)
            print(f"updated {keras_path.relative_to(ROOT)}")


if __name__ == "__main__":
    enrich_all()
