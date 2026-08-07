"""Add concise exploration prompts to the published teaching notebooks.

The source generators create complete workflows. This module keeps those
workflows intact, adds a short ``Try it yourself`` section, and adds the one
retained validation-only comparison for the XGBoost example.
"""

from __future__ import annotations

import copy
import uuid
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


def clone(cell: nbformat.NotebookNode) -> nbformat.NotebookNode:
    result = copy.deepcopy(cell)
    result["id"] = uuid.uuid4().hex[:8]
    if result.cell_type == "code":
        result["outputs"] = []
        result["execution_count"] = None
    return result


def heading_index(notebook: nbformat.NotebookNode, heading: str) -> int:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip().startswith(heading):
            return index
    raise ValueError(f"missing heading {heading!r}")


def code_after(notebook: nbformat.NotebookNode, heading: str) -> nbformat.NotebookNode:
    start = heading_index(notebook, heading)
    for cell in notebook.cells[start + 1 :]:
        if cell.cell_type == "code":
            return clone(cell)
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("## "):
            break
    raise ValueError(f"no code after {heading!r}")


def cells_before(notebook: nbformat.NotebookNode, heading: str) -> list[nbformat.NotebookNode]:
    return [clone(cell) for cell in notebook.cells[: heading_index(notebook, heading)]]


def teaching_notebook(
    source: nbformat.NotebookNode, cells: list[nbformat.NotebookNode]
) -> nbformat.NotebookNode:
    metadata = copy.deepcopy(dict(source.metadata))
    teaching = metadata["helio_data_methods"]
    teaching["artifact"] = "demo"
    teaching["budget"] = "teaching"
    teaching.pop("exercise_id", None)
    return nbformat.v4.new_notebook(cells=cells, metadata=metadata)


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
        "change dropout from `0.5` to `0.25` and compare the validation curves",
        "change one dense-layer width and inspect the parameter count",
        "try a smaller Adam learning rate while keeping the split fixed",
    ),
    "convolutional-neural-networks": (
        "change the first convolution from 32 to 64 filters",
        "compare 3×3 and 5×5 kernels",
        "remove one pooling operation and inspect the tensor shapes",
    ),
    "cifar10-cnn-progression": (
        "change the dropout in the advanced model",
        "vary the third-stage filter count",
        "compare how quickly the two models learn under a shorter budget",
    ),
    "transfer-learning": (
        "replace the 512→256 classifier with one 256-unit layer",
        "change the classifier dropout while keeping VGG16 frozen",
        "unfreeze only the final VGG16 block and use a smaller learning rate",
    ),
    "hyperparameter-tuning": (
        "add dropout values to the search space",
        "increase the number of trials and inspect whether the result is stable",
        "repeat the search with a second sampler seed",
    ),
    "generative-models": (
        "try discriminator label smoothing of `0.1`",
        "vary the latent dimension while keeping the fixed noise samples",
        "save the fixed-noise grid at several points during training",
    ),
    "dst-forecasting": (
        "compare three- and six-hour input histories",
        "change the forecast horizon while keeping 2015 as the final test year",
        "add one solar-wind variable and fit its scaling on training years only",
        "go one step further and rebuild the data-loading stage with NASA CDAWeb's official [`cdasws` Python API](https://cdaweb.gsfc.nasa.gov/WebServices/py/cdasws/), then reproduce the hourly OMNI variables and time range used here",
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
## Example thought

How does maximum tree depth affect validation error and runtime? The two
models below use the same training and validation samples and the same
50-round ceiling. The final test set is not used in this comparison.
""",
                "example-thought",
            ),
            code(
                """
EXAMPLE_ROUNDS = 50  # Reduce to 10 or 25 for a quicker comparison.
EXAMPLE_DEPTHS = [3, 6]
print(f"depths: {EXAMPLE_DEPTHS}; round ceiling={EXAMPLE_ROUNDS}")
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
                "provided",
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

- Try depths 2, 4, and 8 and compare validation error with runtime.
- Hold depth fixed and vary the learning rate between `0.03` and `0.15`.
- Change `subsample` from `0.8` to `0.6` or `1.0` and inspect stability.
""",
                "try-it-yourself",
            ),
        ]
    )


PYTORCH_ADVANCED_CIFAR = """
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2), nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, stride=2), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 6 * 6, 600), nn.BatchNorm1d(600),
            nn.LeakyReLU(0.1), nn.Dropout(0.25), nn.Linear(600, 150),
            nn.BatchNorm1d(150), nn.LeakyReLU(0.1), nn.Dropout(0.5),
            nn.Linear(150, 10),
        )

    def forward(self, values):
        return self.classifier(self.features(values))


model = ImageClassifier()
print(model)
"""

KERAS_ADVANCED_CIFAR = """
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3), layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Conv2D(64, 3, strides=2), layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Conv2D(128, 3, strides=2), layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1), layers.Dropout(0.2),
        layers.Flatten(), layers.Dense(600), layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1), layers.Dropout(0.25),
        layers.Dense(150), layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1), layers.Dropout(0.5),
        layers.Dense(10),
    ],
    name="cifar10_advanced",
)
"""


def cifar_progression(notebook: nbformat.NotebookNode, framework: str) -> nbformat.NotebookNode:
    prefix = cells_before(notebook, "## Define the model")
    simple_model = code_after(notebook, "## Define the model")
    training = code_after(notebook, "## Train with validation evidence")
    evaluation = code_after(notebook, "## Evaluate once on the test set")
    advanced_model = code(
        PYTORCH_ADVANCED_CIFAR if framework == "pytorch" else KERAS_ADVANCED_CIFAR
    )
    if framework == "pytorch":
        capture_simple = """
simple_model = model
simple_history = {key: list(value) for key, value in history.items()}
simple_validation = float(max(history["val_accuracy"]))
"""
    else:
        capture_simple = """
simple_model = model
simple_history = {key: list(value) for key, value in history.history.items()}
simple_validation = float(max(history.history["val_accuracy"]))
"""
    capture_advanced = capture_simple.replace("simple", "advanced")
    cells = [
        md(
            f"""
# CIFAR-10 CNN Progression with {"Native PyTorch" if framework == "pytorch" else "Keras 3 — PyTorch Backend"}

This workflow preserves both archived stages: the simple model receives five
epochs and the advanced model receives its documented 25-epoch budget. Only
the model selected from validation accuracy is evaluated on the test set.
"""
        ),
        *prefix[1:],
        md("## Simple source model — five epochs"),
        code("EPOCHS = 5  # Reduce to 1 or 2 for a quicker run."),
        simple_model,
        clone(training),
        code(capture_simple),
        md("## Advanced source model — 25 epochs"),
        code("EPOCHS = 25  # Reduce to 1 or 2 for a quicker run."),
        advanced_model,
        clone(training),
        code(capture_advanced),
        md("## Compare the source progression"),
        code(
            """
if advanced_validation > simple_validation:
    selected_name, model = "advanced source model", advanced_model
else:
    selected_name, model = "simple source model", simple_model
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(simple_history["val_accuracy"], label="simple (5 epochs)")
ax.plot(advanced_history["val_accuracy"], label="advanced (25 epochs)")
ax.set(title="Source-model validation progression", xlabel="Epoch", ylabel="Accuracy")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
print(f"selected from validation evidence: {selected_name}")
"""
        ),
        md("## Evaluate the selected source model once"),
        evaluation,
    ]
    return teaching_notebook(notebook, cells)


def enrich_all(module_ids: Iterable[str] | None = None) -> None:
    selected = set(module_ids or DEMOS)
    if "cifar10-cnn-progression" in selected:
        for framework, path in (
            ("pytorch", DEMOS["cifar10-cnn-progression"]),
            ("keras", KERAS_DEMOS["cifar10-cnn-progression"]),
        ):
            notebook = cifar_progression(nbformat.read(path, 4), framework)
            nbformat.write(notebook, path)

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
