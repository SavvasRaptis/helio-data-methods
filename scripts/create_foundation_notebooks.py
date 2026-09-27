"""Generate the PyTorch-first dense-MNIST teaching notebooks.

The generator keeps the two notebooks' sections, prose, constants, and
metadata in step. Re-running it intentionally clears stored outputs; run
and verify the generated notebooks before publishing them.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "general-ml" / "foundations" / "neural-networks"


def markdown(text: str, *tags: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(text.strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def code(text: str, *tags: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(text.strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


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
    required = (
        {"keras": "keras", "torch": "torch"}
        if framework == "keras"
        else {"torch": "torch", "torchvision": "torchvision"}
    )
    other = (
        "the [PyTorch version](../pytorch/demo.ipynb) writes the same loop out by hand"
        if framework == "keras"
        else "the [Keras 3 version](../keras/demo.ipynb) runs the same model through `fit()`"
    )
    return [
        markdown(
            f"""
# {title}

A neural network is a chain of simple functions with adjustable weights.
Training adjusts those weights so the network's output for each input moves
closer to the known answer. This notebook trains a small dense network to
recognize handwritten digits from the MNIST collection, and goes through each
step: preparing the data, defining the network, the training loop, and an
honest evaluation. The [Neural Networks](../index.md) chapter gives an
overview; {other}.

Training runs for five epochs, about a minute on a laptop CPU. Set `EPOCHS`
to 1 or 2 in the data cell for a quicker run.
"""
        ),
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
""",
            "remove-cell",
        ),
        code("%matplotlib inline", "remove-cell"),
    ]


SPLIT_CODE = """
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

# Scale pixel values from 0-255 to 0-1.
x_train = x_development[train_indices].astype("float32") / 255.0
y_train = y_development[train_indices]
x_validation = x_development[validation_indices].astype("float32") / 255.0
y_validation = y_development[validation_indices]
x_test = x_test.astype("float32") / 255.0

assert set(train_indices).isdisjoint(validation_indices)
print(
    f"train={len(y_train):,}, validation={len(y_validation):,}, "
    f"test={len(y_test):,}, image shape={x_train.shape[1:]}"
)
"""

COMMON_DATA_KERAS = """
SEED = 42
EPOCHS = 5  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128  # Images per weight update.

keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

(x_development, y_development), (x_test, y_test) = keras.datasets.mnist.load_data()
""" + SPLIT_CODE

COMMON_DATA_TORCH = """
SEED = 42
EPOCHS = 5  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128  # Images per weight update.

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {DEVICE}")

data_root = Path(os.getenv("HELIO_DATA_DIR", Path.home() / ".cache" / "helio-data-methods"))
development_dataset = datasets.MNIST(data_root, train=True, download=True)
test_dataset = datasets.MNIST(data_root, train=False, download=True)
x_development = development_dataset.data.numpy()
y_development = development_dataset.targets.numpy()
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()
""" + SPLIT_CODE

CLASS_DISTRIBUTION = """
fig = plt.figure(figsize=(12, 3.6))
grid = fig.add_gridspec(2, 10, height_ratios=[1, 1.2])
for digit in range(10):
    axis = fig.add_subplot(grid[0, digit])
    axis.imshow(x_train[np.flatnonzero(y_train == digit)[0]], cmap="gray")
    axis.set_title(str(digit), fontsize=9)
    axis.axis("off")
counts_axis = fig.add_subplot(grid[1, :])
counts_axis.bar(np.arange(10), np.bincount(y_train, minlength=10), color="0.45")
counts_axis.set(xticks=np.arange(10), xlabel="Digit", ylabel="Training images")
plt.tight_layout()
plt.show()
"""

KERAS_IMPORTS = """
import hashlib
import json
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.

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
print(f"Keras {keras.__version__}, backend {keras.backend.backend()}, PyTorch {torch.__version__}")
"""

TORCH_IMPORTS = """
import hashlib
import json
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
DROPOUT_RATE = 0.5
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
history = history.history
"""

CURVES = """
epochs = np.arange(1, len(history["loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for axis, key, label in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
    axis.plot(epochs, history[key], marker="o", label="training")
    axis.plot(epochs, history[f"val_{key}"], marker="o", label="validation")
    axis.set(title=label, xlabel="Epoch", xticks=epochs)
    axis.grid(alpha=0.25)
axes[0].legend()
plt.tight_layout()
plt.show()
"""

KERAS_EVALUATE = """
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, test_predictions, digits=3, zero_division=0))
"""

TORCH_DATA_LOADERS = """
def make_loader(images, labels, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels).long())
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


# Shuffle the training images each epoch; keep validation and test in order.
train_loader = make_loader(x_train, y_train, shuffle=True)
validation_loader = make_loader(x_validation, y_validation)
test_loader = make_loader(x_test, y_test)
images, labels = next(iter(train_loader))
print(f"one batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}")
"""

TORCH_MODEL = """
DROPOUT_RATE = 0.5
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
print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
"""

TORCH_TRAIN = """
def run_epoch(data_loader, training):
    model.train(training)  # Dropout is active only in training mode.
    total_loss, total_correct = 0.0, 0
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.set_grad_enabled(training):
            logits = model(images)  # 1. Forward pass.
            loss = loss_function(logits, labels)  # 2. Loss.
            if training:
                optimizer.zero_grad()
                loss.backward()  # 3. Gradients by backpropagation.
                optimizer.step()  # 4. Weight update.
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(data_loader.dataset), total_correct / len(data_loader.dataset)


history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
for epoch in range(EPOCHS):
    train_loss, train_accuracy = run_epoch(train_loader, training=True)
    validation_loss, validation_accuracy = run_epoch(validation_loader, training=False)
    for key, value in zip(history, (train_loss, train_accuracy, validation_loss, validation_accuracy)):
        history[key].append(value)
    print(
        f"epoch {epoch + 1}/{EPOCHS}: loss={train_loss:.4f}  accuracy={train_accuracy:.4f}  "
        f"val_loss={validation_loss:.4f}  val_accuracy={validation_accuracy:.4f}"
    )
"""

TORCH_EVALUATE = """
test_loss, test_accuracy = run_epoch(test_loader, training=False)
model.eval()
with torch.no_grad():
    test_predictions = model(torch.from_numpy(x_test).to(DEVICE)).argmax(dim=1).cpu().numpy()
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, test_predictions, digits=3, zero_division=0))
"""

ERRORS = """
fig, ax = plt.subplots(figsize=(6.5, 6))
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("Test confusion matrix (rows: true digit)")
plt.show()

mistakes = np.flatnonzero(test_predictions != y_test)[:12]
fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
for axis, index in zip(axes.flat, mistakes):
    axis.imshow(x_test[index], cmap="gray")
    axis.set_title(f"{y_test[index]} read as {test_predictions[index]}", fontsize=9)
    axis.axis("off")
fig.suptitle("First twelve test mistakes")
plt.tight_layout()
plt.show()
"""

RECORD = """
print(f"split signature: {split_signature}")
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "split_signature": split_signature,
            "test_accuracy": float(test_accuracy),
            "confusion_shape": list(cm.shape),
        },
        sort_keys=True,
    )
)
assert cm.shape == (10, 10)
"""

DATA_TEXT = """## Load MNIST and split it

MNIST contains 70,000 grayscale images of handwritten digits, 28×28 pixels
each, divided by its authors into 60,000 training and 10,000 test images. We
split the training images again, with a fixed seed and stratified by digit:

- 50,000 **training** images, used to fit the weights;
- 10,000 **validation** images, used to watch for overfitting while we make
  choices about the model;
- the 10,000 official **test** images, used once, after all choices are
  made, to estimate how the model will do on new digits."""

LOOK_TEXT = """## Look at the data

One example of each digit and the number of training images per digit. The
classes are nearly balanced, so accuracy is a fair summary of performance."""

MODEL_TEXT = """## Define the network

The network flattens each image into 784 numbers and passes them through two
**dense** layers of 200 and 150 units. Each unit computes a weighted sum of
its inputs plus a bias and applies the ReLU function, $\\max(0, x)$; without
such nonlinearities a stack of layers would reduce to a single linear map.
**Dropout** sets half of the second layer's outputs to zero at random during
training, which discourages the network from relying on any one unit. The
last layer has ten outputs, the **logits**: unnormalized scores, one per
digit, which a softmax turns into probabilities.

The **loss** is the cross-entropy, the negative log of the probability the
network assigns to the correct digit. **Adam** is the optimizer: a variant of
gradient descent that adapts the step size for each weight."""

TORCH_LOOP_TEXT = """## Train

An **epoch** is one pass over the training images. Within an epoch the
images arrive in **batches** of 128, and for each batch the loop performs
four steps, marked in the code:

1. the forward pass computes the logits;
2. the loss compares them with the true labels;
3. `loss.backward()` computes the gradient of the loss with respect to every
   weight (backpropagation);
4. `optimizer.step()` moves each weight a small step against its gradient.

After each epoch the same function scores the validation images with
gradients switched off."""

KERAS_LOOP_TEXT = """## Train

An **epoch** is one pass over the training images, in **batches** of 128.
For every batch, `fit()` computes the logits, the loss, and the gradient of
the loss with respect to every weight (backpropagation), and moves each
weight a small step against its gradient. After each epoch it scores the
validation images. The [PyTorch version](../pytorch/demo.ipynb) writes these
steps out explicitly."""

CURVES_TEXT = """## Learning curves

Training loss falls steadily. If validation loss started to rise while
training loss kept falling, the network would be memorizing the training
images rather than learning digits in general, which is **overfitting**.
In the first epochs validation accuracy is even higher than training
accuracy: dropout is off during validation, and the training figure averages
over an epoch in which the weights were still improving."""

EVALUATE_TEXT = """## Evaluate on the test set

The test images are used once, now that the model is fixed."""

ERRORS_TEXT = """## Which digits are confused?

Each row of the confusion matrix is a true digit and each column a predicted
one; off-diagonal counts are errors. The images below are the first twelve
test mistakes."""

RESULTS_TEXT = """## What the results show

The network reads about 97.5 of every 100 test digits correctly. Its errors
concentrate on a few pairs of similar shapes, and some of the mistakes shown
above are hard for a person too. Treating the image as a flat list of pixels
discards the fact that neighbouring pixels belong together; the
[convolutional network](../../convolutional-neural-networks/index.md) uses
that structure and makes fewer than half as many errors."""


def keras_cells(artifact: str) -> list[nbformat.NotebookNode]:
    return (
        opening("Dense Neural Network for MNIST: Keras 3", "keras", artifact)
        + [
            markdown(
                """
## Keras and PyTorch

This notebook runs the Keras 3 API on the PyTorch backend. `compile()` sets
the optimizer and loss, and `fit()` runs the loop over batches and epochs
that the PyTorch version writes by hand.
"""
            ),
            markdown("## Setup\n\nThe libraries used below."),
            code(KERAS_IMPORTS, "hide-input"),
            markdown(DATA_TEXT),
            code(COMMON_DATA_KERAS),
            markdown(LOOK_TEXT),
            code(CLASS_DISTRIBUTION),
            markdown(MODEL_TEXT),
            code(KERAS_MODEL),
            markdown(KERAS_LOOP_TEXT),
            code(KERAS_TRAIN),
            markdown(CURVES_TEXT),
            code(CURVES),
            markdown(EVALUATE_TEXT),
            code(KERAS_EVALUATE),
            markdown(ERRORS_TEXT),
            code(ERRORS),
            code(RECORD, "remove-cell"),
            markdown(RESULTS_TEXT),
        ]
    )


def pytorch_cells(artifact: str) -> list[nbformat.NotebookNode]:
    return (
        opening("Dense Neural Network for MNIST: PyTorch", "pytorch", artifact)
        + [
            markdown("## Setup\n\nThe libraries used below."),
            code(TORCH_IMPORTS, "hide-input"),
            markdown(DATA_TEXT),
            code(COMMON_DATA_TORCH),
            markdown(LOOK_TEXT),
            code(CLASS_DISTRIBUTION),
            markdown(
                """## Batch the data

A `DataLoader` serves the images in batches of 128, reshuffling the
training images every epoch so that consecutive updates see different
digits."""
            ),
            code(TORCH_DATA_LOADERS),
            markdown(MODEL_TEXT),
            code(TORCH_MODEL),
            markdown(TORCH_LOOP_TEXT),
            code(TORCH_TRAIN),
            markdown(CURVES_TEXT),
            code(CURVES),
            markdown(EVALUATE_TEXT),
            code(TORCH_EVALUATE),
            markdown(ERRORS_TEXT),
            code(ERRORS),
            code(RECORD, "remove-cell"),
            markdown(RESULTS_TEXT),
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
