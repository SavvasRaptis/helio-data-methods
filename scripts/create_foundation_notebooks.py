"""Generate the PyTorch-first dense-MNIST teaching notebooks.

The generator keeps section order, prose prompts, experimental constants, and
metadata synchronized. Re-running it intentionally clears stored outputs; run
and verify the generated notebooks before publishing them.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "general-ml" / "foundations" / "neural-networks"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


def notebook(
    *,
    framework: str,
    artifact: str,
    title: str,
    cells: list[nbformat.NotebookNode],
) -> nbformat.NotebookNode:
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "helio_data_methods": {
                "module_id": "neural-networks",
                "framework": framework,
                "backend": "torch",
                "implementation_role": (
                    "alternative" if framework == "keras" else "primary"
                ),
                "artifact": artifact,
                "budget": "teaching",
                "runtime": ["local", "colab"],
                "datasets": [],
            },
            "title": title,
        },
    )


def opening(title: str, framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    role = "complete workflow"
    required = (
        {"keras": "keras", "torch": "torch"}
        if framework == "Keras 3 — PyTorch backend"
        else {"torch": "torch", "torchvision": "torchvision"}
    )
    return [
        markdown(
            f"""
# {title}

This {role} implements the dense MNIST experiment from the
[Neural Networks](../index.md) chapter in **{framework}**.

The official 60,000-example training collection is divided with a seeded,
stratified split into 50,000 training and 10,000 validation samples. The
official 10,000-example test set remains untouched until final evaluation.

The example uses five epochs. To run it more quickly, set `EPOCHS` to 1 or 2
in the data-loading cell.

The notebook is self-contained in Google Colab. Select **Runtime → Run all**;
the runtime will use its installed PyTorch stack. The Keras alternative uses
that same runtime through the Keras 3 high-level API.
"""
        ),
        markdown("## Runtime dependency check"),
        code(
            f"""
import importlib.util

required = {required!r}
missing = [
    package for module, package in required.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    raise RuntimeError(
        "Missing notebook dependencies: "
        + ", ".join(missing)
        + ". Locally run `uv sync --group notebooks`; Colab normally "
        "provides these frameworks, so restart the runtime and try again."
    )
print("runtime dependency check passed")
"""
        ),
        code(
            """
%matplotlib inline
"""
        ),
    ]


COMMON_DATA_KERAS = """
SEED = 42
EPOCHS = 5  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128

keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True)

(x_development, y_development), (x_test, y_test) = keras.datasets.mnist.load_data()
all_indices = np.arange(len(y_development))
train_indices, validation_indices = train_test_split(
    all_indices,
    test_size=10_000,
    random_state=SEED,
    stratify=y_development,
)

split_signature = hashlib.sha256(
    validation_indices.astype("<i8").tobytes()
).hexdigest()[:16]

x_train = x_development[train_indices].astype("float32") / 255.0
y_train = y_development[train_indices]
x_validation = x_development[validation_indices].astype("float32") / 255.0
y_validation = y_development[validation_indices]
x_test = x_test.astype("float32") / 255.0

assert set(train_indices).isdisjoint(validation_indices)
print(f"split signature: {split_signature}")
print(
    f"train={len(y_train):,}, validation={len(y_validation):,}, "
    f"test={len(y_test):,}, epochs={EPOCHS}"
)
"""

COMMON_DATA_TORCH = """
SEED = 42
EPOCHS = 5  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

data_root = Path(
    os.getenv(
        "HELIO_DATA_DIR",
        Path.home() / ".cache" / "helio-data-methods",
    )
)
development_dataset = datasets.MNIST(data_root, train=True, download=True)
test_dataset = datasets.MNIST(data_root, train=False, download=True)
x_development = development_dataset.data.numpy()
y_development = development_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

all_indices = np.arange(len(y_development))
train_indices, validation_indices = train_test_split(
    all_indices,
    test_size=10_000,
    random_state=SEED,
    stratify=y_development,
)

split_signature = hashlib.sha256(
    validation_indices.astype("<i8").tobytes()
).hexdigest()[:16]

x_train = x_development[train_indices].astype("float32") / 255.0
y_train = y_development[train_indices]
x_validation = x_development[validation_indices].astype("float32") / 255.0
y_validation = y_development[validation_indices]
x_test = x_test.astype("float32") / 255.0

assert set(train_indices).isdisjoint(validation_indices)
print(f"device: {DEVICE}")
print(f"split signature: {split_signature}")
print(
    f"train={len(y_train):,}, validation={len(y_validation):,}, "
    f"test={len(y_test):,}, epochs={EPOCHS}"
)
"""

CLASS_DISTRIBUTION = """
training_counts = np.bincount(y_train, minlength=10)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(np.arange(10), training_counts)
ax.set(
    title="Training-set class distribution",
    xlabel="Digit",
    ylabel="Number of samples",
    xticks=np.arange(10),
)
plt.show()
"""

KERAS_IMPORTS = """
import hashlib
import os

os.environ["KERAS_BACKEND"] = "torch"

import matplotlib.pyplot as plt
import numpy as np
import torch
import keras
from keras import layers
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

assert keras.backend.backend() == "torch"
print(
    f"Keras {keras.__version__}; backend={keras.backend.backend()}; "
    f"PyTorch {torch.__version__}"
)
"""

TORCH_IMPORTS = """
import hashlib
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

print(f"PyTorch {torch.__version__}")
"""

KERAS_MODEL = """
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(200, activation="relu"),
        layers.Dense(150, activation="relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(10),
    ],
    name="dense_mnist",
)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()
"""

KERAS_TRAIN = """
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
"""

KERAS_CURVES = """
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(history.history["loss"], marker="o", label="training")
axes[0].plot(history.history["val_loss"], marker="o", label="validation")
axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
axes[1].plot(history.history["accuracy"], marker="o", label="training")
axes[1].plot(history.history["val_accuracy"], marker="o", label="validation")
axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
for axis in axes:
    axis.legend()
    axis.grid(alpha=0.25)
plt.tight_layout()
plt.show()
"""

KERAS_EVALUATE = """
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_logits = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
test_predictions = test_logits.argmax(axis=1)
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))

print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(10),
        digits=3,
        zero_division=0,
    )
)
assert cm.shape == (10, 10)
"""

TORCH_DATA_LOADERS = """
def tensor_dataset(images, labels):
    return TensorDataset(
        torch.from_numpy(images),
        torch.from_numpy(labels).long(),
    )


loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    tensor_dataset(x_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
)
validation_loader = DataLoader(
    tensor_dataset(x_validation, y_validation),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_loader = DataLoader(
    tensor_dataset(x_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)
"""

TORCH_MODEL = """
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 200),
    nn.ReLU(),
    nn.Linear(200, 150),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(150, 10),
).to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)
"""

TORCH_TRAIN = """
def run_epoch(data_loader, training):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += len(labels)

    return total_loss / total_examples, total_correct / total_examples


history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
for epoch in range(EPOCHS):
    train_loss, train_accuracy = run_epoch(train_loader, training=True)
    validation_loss, validation_accuracy = run_epoch(
        validation_loader, training=False
    )
    history["loss"].append(train_loss)
    history["accuracy"].append(train_accuracy)
    history["val_loss"].append(validation_loss)
    history["val_accuracy"].append(validation_accuracy)
    print(
        f"epoch {epoch + 1}/{EPOCHS} - "
        f"loss={train_loss:.4f} - accuracy={train_accuracy:.4f} - "
        f"val_loss={validation_loss:.4f} - "
        f"val_accuracy={validation_accuracy:.4f}"
    )
"""

TORCH_CURVES = """
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(history["loss"], marker="o", label="training")
axes[0].plot(history["val_loss"], marker="o", label="validation")
axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
axes[1].plot(history["accuracy"], marker="o", label="training")
axes[1].plot(history["val_accuracy"], marker="o", label="validation")
axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
for axis in axes:
    axis.legend()
    axis.grid(alpha=0.25)
plt.tight_layout()
plt.show()
"""

TORCH_EVALUATE = """
model.eval()
all_logits = []
all_labels = []
test_loss_total = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(DEVICE))
        test_loss_total += loss_function(logits, labels.to(DEVICE)).item() * len(labels)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

test_logits = torch.cat(all_logits).numpy()
test_targets = torch.cat(all_labels).numpy()
test_predictions = test_logits.argmax(axis=1)
test_loss = test_loss_total / len(test_targets)
test_accuracy = accuracy_score(test_targets, test_predictions)
cm = confusion_matrix(test_targets, test_predictions, labels=np.arange(10))

print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
print(
    classification_report(
        test_targets,
        test_predictions,
        labels=np.arange(10),
        digits=3,
        zero_division=0,
    )
)
assert cm.shape == (10, 10)
"""

CONFUSION = """
fig, ax = plt.subplots(figsize=(7, 7))
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("Test-set confusion matrix")
plt.show()
"""

MISCLASSIFICATIONS_KERAS = """
wrong = np.flatnonzero(test_predictions != y_test)
chosen = wrong[:12]
fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for axis, sample_index in zip(axes.flat, chosen):
    axis.imshow(x_test[sample_index], cmap="gray")
    axis.set_title(
        f"true {y_test[sample_index]} | predicted {test_predictions[sample_index]}"
    )
    axis.axis("off")
for axis in axes.flat[len(chosen):]:
    axis.axis("off")
fig.suptitle("Selected test misclassifications")
plt.tight_layout()
plt.show()
"""

MISCLASSIFICATIONS_TORCH = """
wrong = np.flatnonzero(test_predictions != test_targets)
chosen = wrong[:12]
fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for axis, sample_index in zip(axes.flat, chosen):
    axis.imshow(x_test[sample_index], cmap="gray")
    axis.set_title(
        f"true {test_targets[sample_index]} | "
        f"predicted {test_predictions[sample_index]}"
    )
    axis.axis("off")
for axis in axes.flat[len(chosen):]:
    axis.axis("off")
fig.suptitle("Selected test misclassifications")
plt.tight_layout()
plt.show()
"""


def configuration_cells() -> list[nbformat.NotebookNode]:
    return [
        markdown(
            """
## Model configuration

This complete workflow preserves the source architecture and dropout rate.
"""
        ),
        code(
            """
DROPOUT_RATE = 0.5
print(f"dropout rate: {DROPOUT_RATE}")
"""
        ),
    ]


def keras_cells(artifact: str) -> list[nbformat.NotebookNode]:
    title = "Dense MNIST with Keras 3 — PyTorch Backend"
    return (
        opening(title, "Keras 3 — PyTorch backend", artifact)
        + [
            markdown(
                """
## Keras-to-PyTorch crosswalk

This optional notebook uses the same Torch runtime as the canonical PyTorch
path. `compile()` selects the loss and optimizer, `fit()` owns the explicit
epoch/batch loop shown in the canonical PyTorch workflow, and callbacks provide the
high-level hook for training control.
"""
            ),
            markdown("## Imports and reproducibility"),
            code(KERAS_IMPORTS),
            markdown("## Load data and create the shared split"),
            code(COMMON_DATA_KERAS),
            markdown("## Inspect class coverage"),
            code(CLASS_DISTRIBUTION),
        ]
        + configuration_cells()
        + [
            markdown("## Build the dense classifier"),
            code(KERAS_MODEL),
            markdown("## Train with validation evidence"),
            code(KERAS_TRAIN),
            markdown("## Inspect learning curves"),
            code(KERAS_CURVES),
            markdown("## Evaluate once on the held-out test set"),
            code(KERAS_EVALUATE),
            markdown("## Inspect class-level errors"),
            code(CONFUSION),
            markdown("## Inspect representative mistakes"),
            code(MISCLASSIFICATIONS_KERAS),
        ]
    )


def pytorch_cells(artifact: str) -> list[nbformat.NotebookNode]:
    title = "Dense MNIST with PyTorch"
    return (
        opening(title, "PyTorch", artifact)
        + [
            markdown("## Imports and reproducibility"),
            code(TORCH_IMPORTS),
            markdown("## Load data and create the shared split"),
            code(COMMON_DATA_TORCH),
            markdown("## Inspect class coverage"),
            code(CLASS_DISTRIBUTION),
        ]
        + configuration_cells()
        + [
            markdown("## Create data loaders"),
            code(TORCH_DATA_LOADERS),
            markdown("## Build the dense classifier"),
            code(TORCH_MODEL),
            markdown("## Train with validation evidence"),
            code(TORCH_TRAIN),
            markdown("## Inspect learning curves"),
            code(TORCH_CURVES),
            markdown("## Evaluate once on the held-out test set"),
            code(TORCH_EVALUATE),
            markdown("## Inspect class-level errors"),
            code(CONFUSION),
            markdown("## Inspect representative mistakes"),
            code(MISCLASSIFICATIONS_TORCH),
        ]
    )


def main() -> None:
    (MODULE / "keras").mkdir(parents=True, exist_ok=True)
    (MODULE / "pytorch").mkdir(parents=True, exist_ok=True)
    outputs = {
        "keras/demo.ipynb": ("keras", "demo", keras_cells("demo")),
        "pytorch/demo.ipynb": ("pytorch", "demo", pytorch_cells("demo")),
    }
    for filename, (framework, artifact, cells) in outputs.items():
        title = cells[0].source.splitlines()[0].removeprefix("# ").strip()
        destination = MODULE / filename
        nbformat.write(
            notebook(
                framework=framework,
                artifact=artifact,
                title=title,
                cells=cells,
            ),
            destination,
        )
        print(f"wrote {destination.relative_to(ROOT)}")

    from enrich_teaching_notebooks import enrich_all

    enrich_all(["neural-networks"])


if __name__ == "__main__":
    main()
