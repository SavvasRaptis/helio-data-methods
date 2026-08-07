"""Generate aligned notebooks for the published tutorial collection.

The generated notebooks are intentionally self-contained for Colab. Re-running
this script clears outputs; execute and verify ready notebooks before release.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ["local", "colab"]


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(text).strip())


def make_notebook(
    *,
    title: str,
    module_id: str,
    framework: str,
    artifact: str,
    datasets: list[str],
    cells: list[nbformat.NotebookNode],
    library: str | None = None,
    implementation_role: str | None = None,
) -> nbformat.NotebookNode:
    if implementation_role is None:
        implementation_role = {
            "keras": "alternative",
            "pytorch": "primary",
            "framework-neutral": "comparison",
        }[framework]
    teaching_metadata: dict[str, object] = {
        "module_id": module_id,
        "framework": framework,
        "implementation_role": implementation_role,
        "artifact": artifact,
        "budget": "teaching",
        "runtime": RUNTIME,
        "datasets": datasets,
    }
    if framework in {"keras", "pytorch"}:
        teaching_metadata["backend"] = "torch"
    if library:
        teaching_metadata["library"] = library
    required = {
        "keras": {"keras": "keras", "torch": "torch"},
        "pytorch": {"torch": "torch"},
        "framework-neutral": {},
    }[framework].copy()
    optional_install: dict[str, str] = {}
    if module_id == "transfer-learning" and framework == "pytorch":
        required["torchvision"] = "torchvision"
    if module_id == "hyperparameter-tuning":
        package = "keras-tuner" if framework == "keras" else "optuna"
        module = "keras_tuner" if framework == "keras" else "optuna"
        optional_install[module] = package
    if library in {"xgboost", "shap"}:
        optional_install[library] = library
    diagnostics = code(
        f"""
import importlib.util
import subprocess
import sys
from pathlib import Path

REQUIRED_RUNTIME = {required!r}
COLAB_EXTRAS = {optional_install!r}
missing_required = [
    package for module, package in REQUIRED_RUNTIME.items()
    if importlib.util.find_spec(module) is None
]
missing_extras = [
    package for module, package in COLAB_EXTRAS.items()
    if importlib.util.find_spec(module) is None
]
if missing_extras and "google.colab" in sys.modules:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing_extras]
    )
    missing_extras = []
if missing_required or missing_extras:
    missing = ", ".join(missing_required + missing_extras)
    raise RuntimeError(
        f"Missing notebook dependencies: {{missing}}. Locally run "
        "`uv sync --group notebooks`; in Colab restart the runtime if an "
        "installation cell just changed the environment."
    )
print("runtime dependency check passed")
"""
    )
    for cell in cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("](index.md)", "](../index.md)")

    inserted_cells = [
        cells[0],
        md("## Runtime dependency check"),
        diagnostics,
        code("%matplotlib inline"),
    ]
    if framework == "keras":
        inserted_cells.append(
            md(
                """
## Keras-to-PyTorch crosswalk

This optional implementation uses the same Torch runtime as the canonical
native PyTorch path. Keras `compile()` selects the optimizer and loss,
`fit()` owns the explicit batch and epoch loop, and callbacks provide
high-level training control. Data, splits, budgets, evidence, and scientific
conclusions remain aligned with the canonical PyTorch workflow.
"""
            )
        )
    inserted_cells.extend(cells[1:])
    return nbformat.v4.new_notebook(
        cells=inserted_cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "helio_data_methods": teaching_metadata,
            "title": title,
        },
    )


def write_notebook(directory: Path, filename: str, notebook: nbformat.NotebookNode) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    nbformat.write(notebook, path)
    print(f"wrote {path.relative_to(ROOT)}")


DST_IMPORTS = r"""
import hashlib
import json
import os
import random
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
"""

DST_DATA_BOOTSTRAP = r"""
DATASET_ID = "dst-omni-2010-2015"
DATA_FILENAME = "omni2_2010-2015.dat"
DATA_RELATIVE_PATH = "data/dst-omni-2010-2015/omni2_2010-2015.dat"
DATA_SHA256 = "18a4ce192bdcc481bdef699a6e11f7f0441b4e933def8dd0c9cd25fc766bcecf"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_file():
    candidates = []
    override = os.getenv("HELIO_DATA_DIR")
    if override:
        root = Path(override).expanduser()
        candidates.extend([root / DATASET_ID / DATA_FILENAME, root / DATA_FILENAME])
    for root in [Path.cwd(), *Path.cwd().parents]:
        candidates.append(root / DATA_RELATIVE_PATH)
    for candidate in candidates:
        if candidate.is_file() and file_sha256(candidate) == DATA_SHA256:
            return candidate

    cache = (
        Path(os.getenv("HELIO_DATA_CACHE", Path.home() / ".cache" / "helio-data-methods"))
        / "datasets"
        / DATASET_ID
        / DATA_FILENAME
    )
    if not cache.is_file() or file_sha256(cache) != DATA_SHA256:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ref = os.getenv("HELIO_DATA_REF", "main")
        url = (
            "https://raw.githubusercontent.com/SavvasRaptis/helio-data-methods/"
            f"{quote(ref, safe='')}/{quote(DATA_RELATIVE_PATH, safe='/')}"
        )
        try:
            with urlopen(url, timeout=60) as response, cache.open("wb") as output:
                output.write(response.read())
        except Exception as exc:
            cache.unlink(missing_ok=True)
            raise RuntimeError(
                "Dst data could not be downloaded. Check network access or set "
                "HELIO_DATA_DIR to the archived data directory."
            ) from exc
    if file_sha256(cache) != DATA_SHA256:
        cache.unlink(missing_ok=True)
        raise ValueError("Dst dataset checksum mismatch; the invalid file was removed.")
    return cache


data_path = resolve_data_file()
print(f"data file: {DATA_FILENAME} (checksum verified)")
print(f"dataset SHA-256: {file_sha256(data_path)}")
"""

DST_PREPARE = r"""
HEADERS = [
    "year", "day", "hour", "Bartels", "IMF_spacecraft", "plasma_spacecraft",
    "IMF_av_npoints", "plasma_av_npoints", "av_|B|", "|av_B|",
    "lat_av_B_GSE", "lon_av_B_GSE", "Bx", "By_GSE", "Bz_GSE", "By_GSM",
    "Bz_GSM", "sigma_|B|", "sigma_B", "sigma_Bx", "sigma_By", "sigma_Bz",
    "Tp", "Np", "V_plasma", "phi_V_angle", "theta_V_angle", "Na/Np",
    "P_dyn", "sigma_Tp", "sigma_Np", "sigma_V", "sigma_phi_V",
    "sigma_theta_V", "sigma_Na/Np", "E", "beta", "Ma", "Kp", "R", "Dst",
    "AE", "p_flux_>1MeV", "p_flux_>2MeV", "p_flux_>4MeV",
    "p_flux_>10MeV", "p_flux_>30MeV", "p_flux_>60MeV", "flag", "Ap",
    "f10.7", "PC", "AL", "AU", "M_ms",
]
FILL_VALUES = {
    "av_|B|": 999.9,
    "Bz_GSM": 999.9,
    "V_plasma": 9999.0,
    "Dst": 99999.0,
}
INPUT_COLUMNS = ["V_plasma", "Bz_GSM", "av_|B|", "Dst"]


def read_omni(path):
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=HEADERS)
    frame["timestamp"] = (
        pd.to_datetime(frame["year"].astype(str), format="%Y")
        + pd.to_timedelta(frame["day"] - 1, unit="D")
        + pd.to_timedelta(frame["hour"], unit="h")
    )
    for column, fill_value in FILL_VALUES.items():
        frame.loc[frame[column] == fill_value, column] = np.nan
    return frame


def make_windows(frame, years, history_hours, horizon_hours):
    selected = frame.loc[frame["year"].isin(years)].reset_index(drop=True)
    times = selected["timestamp"].to_numpy(dtype="datetime64[h]")
    values = selected[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    dst = selected["Dst"].to_numpy(dtype=np.float32)
    features, targets, persistence, target_times = [], [], [], []
    for origin in range(history_hours - 1, len(selected) - horizon_hours):
        first = origin - history_hours + 1
        target_index = origin + horizon_hours
        history_times = times[first : origin + 1]
        if not np.all(np.diff(history_times) == np.timedelta64(1, "h")):
            continue
        if times[target_index] - times[origin] != np.timedelta64(horizon_hours, "h"):
            continue
        history = values[first : origin + 1]
        target = dst[target_index]
        if not np.isfinite(history).all() or not np.isfinite(target):
            continue
        features.append(history.reshape(-1))
        targets.append(target)
        persistence.append(history[-1, INPUT_COLUMNS.index("Dst")])
        target_times.append(times[target_index])
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(persistence, dtype=np.float32),
        np.asarray(target_times),
    )


FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
SEED = 42
HISTORY_HOURS = HISTORY_HOURS_SETTING
HORIZON_HOURS = 3
EPOCHS = 1 if FAST_RUN else 10
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
frame = read_omni(data_path)
x_train_raw, y_train, persistence_train, time_train = make_windows(
    frame, range(2010, 2014), HISTORY_HOURS, HORIZON_HOURS
)
x_validation_raw, y_validation, persistence_validation, time_validation = make_windows(
    frame, [2014], HISTORY_HOURS, HORIZON_HOURS
)
x_test_raw, y_test, persistence_test, time_test = make_windows(
    frame, [2015], HISTORY_HOURS, HORIZON_HOURS
)

if FAST_RUN:
    x_train_raw, y_train = x_train_raw[:6000], y_train[:6000]
    x_validation_raw, y_validation = x_validation_raw[:2000], y_validation[:2000]
    x_test_raw, y_test = x_test_raw[:2000], y_test[:2000]
    persistence_test, time_test = persistence_test[:2000], time_test[:2000]

scaler = StandardScaler().fit(x_train_raw)
x_train = scaler.transform(x_train_raw).astype(np.float32)
x_validation = scaler.transform(x_validation_raw).astype(np.float32)
x_test = scaler.transform(x_test_raw).astype(np.float32)

split_signature = hashlib.sha256(
    time_test.astype("<M8[h]").astype("<i8").tobytes()
    + np.asarray([HISTORY_HOURS, HORIZON_HOURS], dtype="<i8").tobytes()
).hexdigest()[:16]

assert time_train.max() < time_validation.min() < time_test.min()
print(f"split signature: {split_signature}")
print(
    f"train={len(y_train):,}, validation={len(y_validation):,}, "
    f"test={len(y_test):,}, history={HISTORY_HOURS} h, horizon={HORIZON_HOURS} h"
)
"""

DST_DISTRIBUTION = r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(frame["timestamp"], frame["Dst"], linewidth=0.5)
axes[0].axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2015-01-01"), alpha=0.15)
axes[0].axvspan(pd.Timestamp("2015-01-01"), frame["timestamp"].max(), alpha=0.15)
axes[0].set(title="Archived hourly Dst and fixed year partitions", ylabel="Dst [nT]")
axes[1].hist(y_train, bins=50, alpha=0.8)
axes[1].set(title="Training-target distribution", xlabel="Dst at forecast target [nT]")
plt.tight_layout()
plt.show()
"""

DST_METRICS = r"""
def regression_metrics(y_true, y_prediction):
    return {
        "mae": float(mean_absolute_error(y_true, y_prediction)),
        "rmse": float(mean_squared_error(y_true, y_prediction) ** 0.5),
        "r2": float(r2_score(y_true, y_prediction)),
    }


persistence_metrics = regression_metrics(y_test, persistence_test)
print("persistence baseline:", persistence_metrics)
"""

DST_EVALUATION = r"""
model_metrics = regression_metrics(y_test, predictions)
persistence_skill = 1.0 - (
    model_metrics["rmse"] ** 2 / persistence_metrics["rmse"] ** 2
)
print("model:", model_metrics)
print(f"persistence skill: {persistence_skill:.4f}")
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "split_signature": split_signature,
            "model_rmse": model_metrics["rmse"],
            "persistence_rmse": persistence_metrics["rmse"],
            "persistence_skill": persistence_skill,
            "prediction_shape": list(predictions.shape),
        },
        sort_keys=True,
    )
)
assert predictions.shape == y_test.shape
assert np.isfinite(predictions).all()
"""

DST_DIAGNOSTICS = r"""
display_count = min(1000, len(y_test))
worst = np.argsort(np.abs(y_test - predictions))[-8:][::-1]
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(time_test[:display_count], y_test[:display_count], label="observed", linewidth=1)
axes[0].plot(time_test[:display_count], predictions[:display_count], label="neural model")
axes[0].plot(
    time_test[:display_count],
    persistence_test[:display_count],
    label="persistence",
    linestyle=":",
)
axes[0].set(title="First test interval", ylabel="Dst [nT]")
axes[0].legend()
axes[1].scatter(predictions, y_test - predictions, s=8, alpha=0.35)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(xlabel="Predicted Dst [nT]", ylabel="Residual [nT]", title="Test residuals")
plt.tight_layout()
plt.show()

print("Largest absolute test errors:")
for index in worst:
    print(
        str(time_test[index]),
        f"observed={y_test[index]:.1f}",
        f"predicted={predictions[index]:.1f}",
        f"persistence={persistence_test[index]:.1f}",
    )
"""


def dst_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    role = "complete workflow"
    history = 3
    framework_label = (
        "Keras 3 — PyTorch Backend" if framework == "keras" else "Native PyTorch"
    )
    cells = [
        md(
            f"""
# Dst Forecasting with {framework_label}

This {role} implements the experiment in the
[Dst Forecasting](index.md) chapter. The notebook forms gap-safe windows
inside fixed year partitions and evaluates a three-hour-ahead forecast against
true forecast-origin persistence.

Set `HELIO_FAST_RUN=1` for the reduced one-epoch smoke configuration. In Colab,
select **Runtime → Run all**; the data cell downloads and verifies only the
archived OMNI file when no local checkout is available.
"""
        ),
        md("## Imports and reproducibility"),
        code(DST_IMPORTS),
        md("## Resolve the archived dataset"),
        code(DST_DATA_BOOTSTRAP),
        md("## Parse data, form windows, and fit training-only preprocessing"),
        code(DST_PREPARE.replace("HISTORY_HOURS_SETTING", str(history))),
        md("## Inspect coverage and the training target"),
        code(DST_DISTRIBUTION),
        md("## Establish the persistence baseline"),
        code(DST_METRICS),
    ]
    if framework == "keras":
        cells.extend(
            [
                md("## Define the Keras model"),
                code(
                    r"""
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
from keras import layers

keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True)
assert keras.backend.backend() == "torch"
model = keras.Sequential(
    [
        keras.Input(shape=(x_train.shape[1],)),
        layers.Dense(50, activation="relu"),
        layers.Dense(30, activation="relu"),
        layers.Dense(1),
    ],
    name="dst_forecast",
)
model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
model.summary()
"""
                ),
                md("## Train with validation-based early stopping"),
                code(
                    r"""
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
]
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(history.history["loss"], marker="o", label="training")
ax.plot(history.history["val_loss"], marker="o", label="validation")
ax.set(title="Mean-squared error", xlabel="Epoch", ylabel="MSE")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""
                ),
                md("## Evaluate the untouched test year"),
                code(
                    r"""
predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
"""
                    + DST_EVALUATION
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## Define the PyTorch model"),
                code(
                    r"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class DstRegressor(nn.Module):
    def __init__(self, number_inputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(number_inputs, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )

    def forward(self, values):
        return self.network(values).squeeze(1)


model = DstRegressor(x_train.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()
print(model)
"""
                ),
                md("## Train and restore the best validation state"),
                code(
                    r"""
generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)
x_validation_tensor = torch.from_numpy(x_validation).to(DEVICE)
y_validation_tensor = torch.from_numpy(y_validation).to(DEVICE)
training_loss, validation_loss = [], []
best_state, best_validation, stale_epochs = None, float("inf"), 0

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for features, target in train_loader:
        optimizer.zero_grad()
        loss = loss_function(model(features.to(DEVICE)), target.to(DEVICE))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
    training_loss.append(total / len(train_loader.dataset))
    model.eval()
    with torch.no_grad():
        current_validation = loss_function(
            model(x_validation_tensor), y_validation_tensor
        ).item()
    validation_loss.append(current_validation)
    print(
        f"epoch {epoch + 1}: loss={training_loss[-1]:.3f}, "
        f"val_loss={current_validation:.3f}"
    )
    if current_validation < best_validation:
        best_validation = current_validation
        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        stale_epochs = 0
    else:
        stale_epochs += 1
        if stale_epochs >= 3:
            break

model.load_state_dict(best_state)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(training_loss, marker="o", label="training")
ax.plot(validation_loss, marker="o", label="validation")
ax.set(title="Mean-squared error", xlabel="Epoch", ylabel="MSE")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""
                ),
                md("## Evaluate the untouched test year"),
                code(
                    r"""
model.eval()
with torch.no_grad():
    predictions = model(torch.from_numpy(x_test).to(DEVICE)).cpu().numpy()
"""
                    + DST_EVALUATION
                ),
            ]
        )
    cells.extend(
        [
            md("## Diagnose timing and residual errors"),
            code(DST_DIAGNOSTICS),
        ]
    )
    return cells


def generate_dst() -> None:
    directory = ROOT / "heliophysics" / "applications" / "dst-forecasting"
    for framework in ("pytorch", "keras"):
        artifacts = ("demo",)
        for artifact in artifacts:
            notebook = make_notebook(
                title=f"Dst Forecasting — Complete Workflow ({'Keras on Torch' if framework == 'keras' else 'Native PyTorch'})",
                module_id="dst-forecasting",
                framework=framework,
                artifact=artifact,
                datasets=["dst-omni-2010-2015"],
                cells=dst_cells(framework, artifact),
            )
            write_notebook(directory / framework, f"{artifact}.ipynb", notebook)


IMAGE_KERAS_IMPORTS = r"""
import hashlib
import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
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

IMAGE_TORCH_IMPORTS = r"""
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


def image_data_code(framework: str, dataset: str, epochs: int, batch_size: int) -> str:
    validation_size = 10_000 if dataset == "mnist" else 5_000
    train_fast = 5_000 if dataset == "mnist" else 2_000
    validation_fast = 2_000 if dataset == "mnist" else 500
    test_fast = 2_000 if dataset == "mnist" else 500
    shape_keras = (
        'x_train = x_train[..., np.newaxis]\n'
        'x_validation = x_validation[..., np.newaxis]\n'
        'x_test = x_test[..., np.newaxis]'
        if dataset == "mnist"
        else ""
    )
    shape_torch = (
        'x_train = x_train[:, np.newaxis, ...]\n'
        'x_validation = x_validation[:, np.newaxis, ...]\n'
        'x_test = x_test[:, np.newaxis, ...]'
        if dataset == "mnist"
        else (
            'x_train = np.transpose(x_train, (0, 3, 1, 2))\n'
            'x_validation = np.transpose(x_validation, (0, 3, 1, 2))\n'
            'x_test = np.transpose(x_test, (0, 3, 1, 2))'
        )
    )
    if framework == "keras":
        load = (
            "(x_development, y_development), (x_test, y_test) = "
            "keras.datasets.mnist.load_data()"
            if dataset == "mnist"
            else (
                "(x_development, y_development), (x_test, y_test) = "
                "keras.datasets.cifar10.load_data()\n"
                "y_development = y_development.reshape(-1)\n"
                "y_test = y_test.reshape(-1)"
            )
        )
        seed = """
keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True)
"""
        shape = shape_keras
    else:
        dataset_class = "MNIST" if dataset == "mnist" else "CIFAR10"
        x_attr = "development_dataset.data.numpy()" if dataset == "mnist" else "development_dataset.data"
        y_attr = (
            "development_dataset.targets.numpy()"
            if dataset == "mnist"
            else "np.asarray(development_dataset.targets)"
        )
        test_x_attr = "test_dataset.data.numpy()" if dataset == "mnist" else "test_dataset.data"
        test_y_attr = (
            "test_dataset.targets.numpy()"
            if dataset == "mnist"
            else "np.asarray(test_dataset.targets)"
        )
        load = f"""
data_root = Path(
    os.getenv("HELIO_DATA_DIR", Path.home() / ".cache" / "helio-data-methods")
)
development_dataset = datasets.{dataset_class}(data_root, train=True, download=True)
test_dataset = datasets.{dataset_class}(data_root, train=False, download=True)
x_development = {x_attr}
y_development = {y_attr}
x_test = {test_x_attr}
y_test = {test_y_attr}
"""
        seed = """
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
"""
        shape = shape_torch
    return dedent(
        f"""
FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
SEED = 42
EPOCHS = 1 if FAST_RUN else {epochs}
BATCH_SIZE = {batch_size}
{seed}
{load}
all_indices = np.arange(len(y_development))
train_indices, validation_indices = train_test_split(
    all_indices,
    test_size={validation_size},
    random_state=SEED,
    stratify=y_development,
)
split_signature = hashlib.sha256(
    validation_indices.astype("<i8").tobytes()
).hexdigest()[:16]

if FAST_RUN:
    train_indices = train_indices[:{train_fast}]
    validation_indices = validation_indices[:{validation_fast}]
    x_test = x_test[:{test_fast}]
    y_test = y_test[:{test_fast}]

x_train = x_development[train_indices].astype("float32") / 255.0
y_train = y_development[train_indices].astype(np.int64)
x_validation = x_development[validation_indices].astype("float32") / 255.0
y_validation = y_development[validation_indices].astype(np.int64)
x_test = x_test.astype("float32") / 255.0
y_test = y_test.astype(np.int64)
{shape}

assert set(train_indices).isdisjoint(validation_indices)
print(f"split signature: {{split_signature}}")
print(
    f"train={{len(y_train):,}}, validation={{len(y_validation):,}}, "
    f"test={{len(y_test):,}}, epochs={{EPOCHS}}"
)
"""
    ).strip()


IMAGE_DISTRIBUTION = r"""
training_counts = np.bincount(y_train, minlength=10)
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(np.arange(10), training_counts)
ax.set(
    title="Training-set class distribution",
    xlabel="Class",
    ylabel="Samples",
    xticks=np.arange(10),
)
plt.show()
"""

IMAGE_KERAS_TRAIN = r"""
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(history.history["loss"], marker="o", label="training")
axes[0].plot(history.history["val_loss"], marker="o", label="validation")
axes[0].set(title="Cross-entropy loss", xlabel="Epoch")
axes[1].plot(history.history["accuracy"], marker="o", label="training")
axes[1].plot(history.history["val_accuracy"], marker="o", label="validation")
axes[1].set(title="Accuracy", xlabel="Epoch")
for axis in axes:
    axis.legend()
    axis.grid(alpha=0.25)
plt.tight_layout()
plt.show()
"""

IMAGE_TORCH_TRAIN = r"""
def tensor_dataset(images, labels):
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(labels).long())


generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    tensor_dataset(x_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)
validation_loader = DataLoader(
    tensor_dataset(x_validation, y_validation), batch_size=BATCH_SIZE
)
test_loader = DataLoader(tensor_dataset(x_test, y_test), batch_size=BATCH_SIZE)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()
history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}


def epoch_pass(loader, training):
    model.train(training)
    total_loss = 0.0
    correct = 0
    for features, target in loader:
        features, target = features.to(DEVICE), target.to(DEVICE)
        if training:
            optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, target)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(target)
        correct += (logits.argmax(1) == target).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


for epoch in range(EPOCHS):
    train_loss, train_accuracy = epoch_pass(train_loader, True)
    with torch.no_grad():
        validation_loss, validation_accuracy = epoch_pass(validation_loader, False)
    history["loss"].append(train_loss)
    history["accuracy"].append(train_accuracy)
    history["val_loss"].append(validation_loss)
    history["val_accuracy"].append(validation_accuracy)
    print(
        f"epoch {epoch + 1}: loss={train_loss:.4f}, accuracy={train_accuracy:.4f}, "
        f"val_loss={validation_loss:.4f}, val_accuracy={validation_accuracy:.4f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(history["loss"], marker="o", label="training")
axes[0].plot(history["val_loss"], marker="o", label="validation")
axes[0].set(title="Cross-entropy loss", xlabel="Epoch")
axes[1].plot(history["accuracy"], marker="o", label="training")
axes[1].plot(history["val_accuracy"], marker="o", label="validation")
axes[1].set(title="Accuracy", xlabel="Epoch")
for axis in axes:
    axis.legend()
    axis.grid(alpha=0.25)
plt.tight_layout()
plt.show()
"""


def image_evaluation_code(framework: str) -> str:
    inference = (
        """
test_logits = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
test_predictions = test_logits.argmax(axis=1)
"""
        if framework == "keras"
        else """
model.eval()
prediction_parts = []
with torch.no_grad():
    for features, _ in test_loader:
        prediction_parts.append(model(features.to(DEVICE)).argmax(1).cpu().numpy())
test_predictions = np.concatenate(prediction_parts)
"""
    )
    return dedent(
        f"""
{inference}
test_accuracy = accuracy_score(y_test, test_predictions)
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
print(f"test accuracy: {{test_accuracy:.4f}}")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(10),
        digits=3,
        zero_division=0,
    )
)
print(
    "HELIO_RESULT "
    + json.dumps(
        {{
            "split_signature": split_signature,
            "test_accuracy": float(test_accuracy),
            "confusion_shape": list(cm.shape),
        }},
        sort_keys=True,
    )
)
assert cm.shape == (10, 10)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(
    ax=ax, colorbar=False, values_format="d"
)
ax.set_title("Test confusion matrix")
plt.show()

mistakes = np.flatnonzero(test_predictions != y_test)[:12]
if len(mistakes):
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for axis, index in zip(axes.flat, mistakes):
        image = x_test[index]
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        axis.imshow(image.squeeze(), cmap="gray" if image.squeeze().ndim == 2 else None)
        axis.set_title(f"true={{y_test[index]}}, pred={{test_predictions[index]}}")
        axis.axis("off")
    plt.tight_layout()
    plt.show()
"""
    ).strip()


def image_model_code(framework: str, architecture: str) -> str:
    if framework == "keras":
        definitions = {
            "mnist": r"""
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(200, activation="relu"),
        layers.Dense(150, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10),
    ],
    name="mnist_cnn",
)
""",
            "cifar_simple": r"""
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, 3),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Conv2D(32, 3, strides=2),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Flatten(),
        layers.Dense(100),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.5),
        layers.Dense(10),
    ],
    name="cifar10_simple",
)
""",
            "cifar_advanced": r"""
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Conv2D(64, 3, strides=2),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Conv2D(128, 3, strides=2),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(600),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.25),
        layers.Dense(150),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.5),
        layers.Dense(10),
    ],
    name="cifar10_advanced",
)
""",
        }
        return definitions[architecture]

    definitions = {
        "mnist": r"""
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 5 * 5, 200), nn.ReLU(),
            nn.Linear(200, 150), nn.ReLU(), nn.Dropout(0.5), nn.Linear(150, 10),
        )

    def forward(self, values):
        return self.classifier(self.features(values))


model = ImageClassifier()
print(model)
""",
        "cifar_simple": r"""
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 3, stride=2), nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 14 * 14, 100), nn.BatchNorm1d(100),
            nn.LeakyReLU(0.1), nn.Dropout(0.5), nn.Linear(100, 10),
        )

    def forward(self, values):
        return self.classifier(self.features(values))


model = ImageClassifier()
print(model)
""",
        "cifar_advanced": r"""
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
""",
    }
    return definitions[architecture]


def paired_image_cells(
    *,
    title: str,
    chapter: str,
    framework: str,
    artifact: str,
    dataset: str,
    architecture: str,
    epochs: int,
    batch_size: int,
) -> list[nbformat.NotebookNode]:
    label = "Keras 3" if framework == "keras" else "PyTorch"
    role = "complete workflow"
    cells = [
        md(
            f"""
# {title} with {label}

This {role} belongs to [{chapter}](index.md). Keras and PyTorch use the same
seeded indices, normalization, architecture intent, training budget, metrics,
and diagnostic figures.

Set `HELIO_FAST_RUN=1` for a reduced one-epoch smoke run. In Colab select
**Runtime → Run all**; the canonical dataset is downloaded by the framework.
"""
        ),
        md("## Imports and reproducibility"),
        code(IMAGE_KERAS_IMPORTS if framework == "keras" else IMAGE_TORCH_IMPORTS),
        md("## Load data and create the shared split"),
        code(image_data_code(framework, dataset, epochs, batch_size)),
        md("## Inspect class coverage"),
        code(IMAGE_DISTRIBUTION),
    ]
    cells.extend(
        [
            md("## Define the model"),
            code(image_model_code(framework, architecture)),
            md("## Train with validation evidence"),
            code(IMAGE_KERAS_TRAIN if framework == "keras" else IMAGE_TORCH_TRAIN),
            md("## Evaluate once on the test set"),
            code(image_evaluation_code(framework)),
        ]
    )
    return cells


def generate_image_modules() -> None:
    modules = [
        {
            "directory": ROOT
            / "general-ml"
            / "foundations"
            / "convolutional-neural-networks",
            "module_id": "convolutional-neural-networks",
            "title": "MNIST Convolutional Neural Network",
            "chapter": "Convolutional Neural Networks",
            "dataset": "mnist",
            "epochs": {"demo": 5},
            "batch_size": 256,
            "architectures": {
                "demo": "mnist",
            },
        },
        {
            "directory": ROOT
            / "general-ml"
            / "advanced"
            / "cifar10-cnn-progression",
            "module_id": "cifar10-cnn-progression",
            "title": "CIFAR-10 CNN Progression",
            "chapter": "CIFAR-10 CNN Progression",
            "dataset": "cifar10",
            "epochs": {"demo": 5},
            "batch_size": 64,
            "architectures": {
                "demo": "cifar_simple",
            },
        },
    ]
    for settings in modules:
        for framework in ("pytorch", "keras"):
            artifacts = ("demo",)
            for artifact in artifacts:
                notebook = make_notebook(
                    title=f"{settings['title']} — {artifact.title()} ({'Keras on Torch' if framework == 'keras' else 'Native PyTorch'})",
                    module_id=str(settings["module_id"]),
                    framework=framework,
                    artifact=artifact,
                    datasets=[],
                    cells=paired_image_cells(
                        title=str(settings["title"]),
                        chapter=str(settings["chapter"]),
                        framework=framework,
                        artifact=artifact,
                        dataset=str(settings["dataset"]),
                        architecture=settings["architectures"][artifact],
                        epochs=settings["epochs"][artifact],
                        batch_size=int(settings["batch_size"]),
                    ),
                )
                write_notebook(settings["directory"] / framework, f"{artifact}.ipynb", notebook)


def tree_cells(artifact: str) -> list[nbformat.NotebookNode]:
    depth = 6
    role = "complete workflow"
    return [
        md(
            f"""
# MNIST with XGBoost

This {role} provides the framework-neutral tree-model comparison from
[Tree Models and Ensembles](index.md). It uses the same seed-42 split and
classification evidence as the neural examples.

Set `HELIO_FAST_RUN=1` for a reduced ten-round smoke run. In Colab, run all
cells; the first import cell installs XGBoost only if it is missing.
"""
        ),
        md("## Imports and shared split"),
        code(
            r"""
import hashlib
import gzip
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

if importlib.util.find_spec("xgboost") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost>=2.1,<4"])

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
SEED = 42
raw_root = Path(
    os.getenv("HELIO_DATA_DIR", Path.home() / ".cache" / "helio-data-methods" / "torchvision")
) / "MNIST" / "raw"
raw_root.mkdir(parents=True, exist_ok=True)
mnist_files = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def fetch_mnist_file(filename, checksum):
    destination = raw_root / filename
    if not destination.exists():
        url = f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}"
        with urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != checksum:
        raise RuntimeError(f"MNIST checksum mismatch for {filename}: {actual}")
    return destination


def read_images(path):
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise RuntimeError(f"unexpected MNIST image magic number: {magic}")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, rows, columns)


def read_labels(path):
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise RuntimeError(f"unexpected MNIST label magic number: {magic}")
        return np.frombuffer(stream.read(), dtype=np.uint8, count=count)


resolved_mnist = {
    filename: fetch_mnist_file(filename, checksum)
    for filename, checksum in mnist_files.items()
}
import xgboost as xgb

x_development = read_images(resolved_mnist["train-images-idx3-ubyte.gz"])
y_development = read_labels(resolved_mnist["train-labels-idx1-ubyte.gz"])
x_test = read_images(resolved_mnist["t10k-images-idx3-ubyte.gz"])
y_test = read_labels(resolved_mnist["t10k-labels-idx1-ubyte.gz"])
indices = np.arange(len(y_development))
train_indices, validation_indices = train_test_split(
    indices, test_size=10_000, random_state=SEED, stratify=y_development
)
split_signature = hashlib.sha256(
    validation_indices.astype("<i8").tobytes()
).hexdigest()[:16]
if FAST_RUN:
    train_indices = train_indices[:5_000]
    validation_indices = validation_indices[:1_000]
    x_test, y_test = x_test[:2_000], y_test[:2_000]
x_train = x_development[train_indices].reshape(len(train_indices), -1).astype("float32") / 255
y_train = y_development[train_indices]
x_validation = (
    x_development[validation_indices].reshape(len(validation_indices), -1).astype("float32") / 255
)
y_validation = y_development[validation_indices]
x_test = x_test.reshape(len(x_test), -1).astype("float32") / 255
print(f"XGBoost {xgb.__version__}; split signature: {split_signature}")
"""
        ),
        md("## Inspect the training distribution"),
        code(IMAGE_DISTRIBUTION),
        md("## Train with validation-only early stopping"),
        code(
            f"""
parameters = {{
    "objective": "multi:softprob",
    "num_class": 10,
    "eta": 0.08,
    "max_depth": {depth},
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "alpha": 8,
    "lambda": 2,
    "eval_metric": "merror",
    "seed": SEED,
    "nthread": 2,
}}
rounds = 10 if FAST_RUN else 100
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalidation = xgb.DMatrix(x_validation, label=y_validation)
model = xgb.train(
    parameters,
    dtrain,
    num_boost_round=rounds,
    evals=[(dtrain, "training"), (dvalidation, "validation")],
    early_stopping_rounds=10,
    verbose_eval=10,
)
"""
        ),
        md("## Final test evidence"),
        code(
            r"""
probabilities = model.predict(xgb.DMatrix(x_test))
test_predictions = probabilities.argmax(axis=1)
test_accuracy = accuracy_score(y_test, test_predictions)
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
print(f"test accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, test_predictions, digits=3, zero_division=0))
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
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, values_format="d")
ax.set_title("XGBoost test confusion matrix")
plt.show()
"""
        ),
    ]


def generate_tree_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "tree-models"
    for artifact in ("demo",):
        notebook = make_notebook(
            title=f"MNIST XGBoost — {artifact.title()}",
            module_id="tree-models",
            framework="framework-neutral",
            artifact=artifact,
            datasets=[],
            library="xgboost",
            implementation_role="primary",
            cells=tree_cells(artifact),
        )
        write_notebook(directory / "xgboost", f"{artifact}.ipynb", notebook)


def transfer_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    role = "complete workflow"
    keras_head = (
        "layers.Dense(512, activation='relu'), layers.Dropout(0.25), "
        "layers.Dense(256, activation='relu'), layers.Dropout(0.25),"
    )
    torch_head = (
        "nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.25), "
        "nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.25),"
    )
    cells = [
        md(
            f"""
# CIFAR-10 Transfer Learning with {"Keras 3" if framework == "keras" else "PyTorch"}

This {role} freezes an ImageNet-pretrained VGG16 feature extractor and trains a
CIFAR-10 classifier. The split, head intent, ten-epoch maximum, and evaluation
are aligned across frameworks.

Set `HELIO_FAST_RUN=1` for a reduced one-epoch run. A network connection is
required the first time the pretrained weights are cached.
"""
        ),
        md("## Imports and shared CIFAR-10 split"),
        code(IMAGE_KERAS_IMPORTS if framework == "keras" else IMAGE_TORCH_IMPORTS),
        code(image_data_code(framework, "cifar10", 10, 256)),
        md("## Inspect class coverage"),
        code(IMAGE_DISTRIBUTION),
    ]
    if framework == "keras":
        cells.extend(
            [
                md("## Preprocess for VGG16 and define the trainable head"),
                code(
                    f"""
x_train = keras.applications.vgg16.preprocess_input(x_train * 255.0)
x_validation = keras.applications.vgg16.preprocess_input(x_validation * 255.0)
x_test_images = x_test.copy()
x_test = keras.applications.vgg16.preprocess_input(x_test * 255.0)
base_model = keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(32, 32, 3)
)
base_model.trainable = False
print("extracting frozen VGG16 features once per split")
x_train = base_model.predict(
    x_train, batch_size=BATCH_SIZE, verbose=1
).reshape(len(x_train), -1)
x_validation = base_model.predict(
    x_validation, batch_size=BATCH_SIZE, verbose=1
).reshape(len(x_validation), -1)
x_test = base_model.predict(
    x_test, batch_size=BATCH_SIZE, verbose=1
).reshape(len(x_test), -1)
model = keras.Sequential(
    [
        keras.Input(shape=(x_train.shape[1],)),
        {keras_head}
        layers.Dense(10),
    ],
    name="vgg16_frozen_features",
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print(f"trainable parameters: {{sum(np.prod(v.shape) for v in model.trainable_weights):,}}")
model.summary()
"""
                ),
                md("## Train with validation-based early stopping"),
                code(
                    r"""
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ],
    verbose=2,
)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(history.history["loss"], label="training")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set(title="Cross-entropy", xlabel="Epoch")
axes[1].plot(history.history["accuracy"], label="training")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set(title="Accuracy", xlabel="Epoch")
for axis in axes:
    axis.legend()
plt.tight_layout()
plt.show()
"""
                ),
                md("## Final test evidence"),
                code(
                    image_evaluation_code("keras").replace(
                        "image = x_test[index]", "image = x_test_images[index]"
                    )
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## Normalize for VGG16 and define the trainable head"),
                code(
                    f"""
from torchvision.models import VGG16_Weights, vgg16

mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
x_train = ((x_train - mean) / std).astype(np.float32)
x_validation = ((x_validation - mean) / std).astype(np.float32)
x_test_images = x_test.copy()
x_test = ((x_test - mean) / std).astype(np.float32)


class TransferClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        source = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = source.features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()

    def forward(self, values):
        return torch.flatten(self.pool(self.features(values)), 1)


feature_extractor = TransferClassifier().to(DEVICE)


def extract_features(images):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(images)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    batches = []
    with torch.no_grad():
        for (batch,) in loader:
            batches.append(feature_extractor(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


print("extracting frozen VGG16 features once per split")
x_train = extract_features(x_train)
x_validation = extract_features(x_validation)
x_test = extract_features(x_test)
model = nn.Sequential(
    {torch_head}
    nn.Linear(256, 10),
)
print(
    "trainable parameters:",
    f"{{sum(value.numel() for value in model.parameters() if value.requires_grad):,}}",
)
"""
                ),
                md("## Train under the shared budget"),
                code(IMAGE_TORCH_TRAIN.replace(
                    "optimizer = torch.optim.Adam(model.parameters())",
                    (
                        "optimizer = torch.optim.Adam(\n"
                        "    (value for value in model.parameters() if value.requires_grad),\n"
                        "    lr=0.001,\n"
                        ")"
                    ),
                )),
                md("## Final test evidence"),
                code(
                    image_evaluation_code("pytorch").replace(
                        "image = x_test[index]", "image = x_test_images[index]"
                    )
                ),
            ]
        )
    return cells


def generate_transfer_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "transfer-learning"
    for framework in ("pytorch", "keras"):
        artifacts = ("demo",)
        for artifact in artifacts:
            notebook = make_notebook(
                title=f"CIFAR-10 Transfer Learning — {artifact.title()} ({'Keras on Torch' if framework == 'keras' else 'Native PyTorch'})",
                module_id="transfer-learning",
                framework=framework,
                artifact=artifact,
                datasets=[],
                cells=transfer_cells(framework, artifact),
            )
            write_notebook(directory / framework, f"{artifact}.ipynb", notebook)


def tuning_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    role = "complete workflow"
    cells = [
        md(
            f"""
# MNIST CNN Tuning with {"KerasTuner" if framework == "keras" else "Optuna and PyTorch"}

This {role} treats tuning as a validation experiment. Both frameworks use the
same seed-42 split, discrete search space, four-trial budget, ten-epoch maximum,
and validation accuracy objective.

Set `HELIO_FAST_RUN=1` for two one-epoch trials.
"""
        ),
        md("## Imports and shared split"),
        code(
            (
                IMAGE_KERAS_IMPORTS
                + r"""
import importlib.util
import subprocess
import sys
from pathlib import Path
if importlib.util.find_spec("keras_tuner") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "keras-tuner>=1.4,<2"]
    )
import keras_tuner as kt
"""
            )
            if framework == "keras"
            else (
                IMAGE_TORCH_IMPORTS
                + r"""
import importlib.util
import subprocess
import sys
import warnings
if importlib.util.find_spec("optuna") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "optuna>=4,<5"]
    )
warnings.filterwarnings("ignore", message="IProgress not found.*")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
"""
            )
        ),
        code(image_data_code(framework, "mnist", 5, 128)),
        md("## Fixed search contract"),
        code(
            IMAGE_DISTRIBUTION
            + r"""
TRIALS = 2 if FAST_RUN else 4
SEARCH_EPOCHS = 1 if FAST_RUN else 10
SEARCH_SPACE = {
    "filters_1": [16, 32],
    "filters_2": [32, 64],
    "dense_units": [100, 200],
    "learning_rate": [1e-2, 1e-3, 1e-4],
}
"""
        ),
    ]
    if framework == "keras":
        cells.extend(
            [
                md("## Run KerasTuner"),
                code(
                    r"""
def build_model(hp):
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(hp.Choice("filters_1", SEARCH_SPACE["filters_1"]), 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(hp.Choice("filters_2", SEARCH_SPACE["filters_2"]), 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(hp.Choice("dense_units", SEARCH_SPACE["dense_units"]), activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", SEARCH_SPACE["learning_rate"])
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=TRIALS,
    seed=SEED,
    overwrite=True,
    directory=str(Path(os.getenv("HELIO_TUNER_DIR", "/tmp")) / "helio-keras-tuner"),
    project_name="mnist-cnn",
)
tuner.search(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=SEARCH_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2)],
    verbose=0,
)
best_parameters = tuner.get_best_hyperparameters(1)[0]
print("best parameters:", best_parameters.values)
"""
                ),
                md("## Rebuild the selected model and evaluate once"),
                code(
                    r"""
model = tuner.hypermodel.build(best_parameters)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
"""
                ),
                code(image_evaluation_code("keras")),
            ]
        )
    else:
        cells.extend(
            [
                md("## Run Optuna"),
                code(
                    r"""
def build_trial_model(trial):
    filters_1 = trial.suggest_categorical("filters_1", SEARCH_SPACE["filters_1"])
    filters_2 = trial.suggest_categorical("filters_2", SEARCH_SPACE["filters_2"])
    dense_units = trial.suggest_categorical("dense_units", SEARCH_SPACE["dense_units"])
    return nn.Sequential(
        nn.Conv2d(1, filters_1, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(filters_1, filters_2, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Dropout(0.25), nn.Flatten(), nn.Linear(filters_2 * 5 * 5, dense_units),
        nn.ReLU(), nn.Dropout(0.5), nn.Linear(dense_units, 10),
    )


train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long())
validation_features = torch.from_numpy(x_validation)
validation_targets = torch.from_numpy(y_validation).long()


def objective(trial):
    torch.manual_seed(SEED + trial.number)
    candidate = build_trial_model(trial)
    learning_rate = trial.suggest_categorical(
        "learning_rate", SEARCH_SPACE["learning_rate"]
    )
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + trial.number),
    )
    best_accuracy = 0.0
    for epoch in range(SEARCH_EPOCHS):
        candidate.train()
        for features, target in loader:
            candidate_optimizer.zero_grad()
            loss = criterion(candidate(features), target)
            loss.backward()
            candidate_optimizer.step()
        candidate.eval()
        with torch.no_grad():
            accuracy = (
                candidate(validation_features).argmax(1) == validation_targets
            ).float().mean().item()
        best_accuracy = max(best_accuracy, accuracy)
        trial.report(accuracy, epoch)
    return best_accuracy


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=TRIALS)
print("best parameters:", study.best_params)
"""
                ),
                md("## Rebuild the selected model and evaluate once"),
                code(
                    r"""
class FixedTrial:
    def suggest_categorical(self, name, values):
        return study.best_params[name]


model = build_trial_model(FixedTrial())
"""
                ),
                code(
                    IMAGE_TORCH_TRAIN.replace(
                        "optimizer = torch.optim.Adam(model.parameters())",
                        'optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params["learning_rate"])',
                    )
                ),
                code(image_evaluation_code("pytorch")),
            ]
        )
    return cells


def generate_tuning_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "hyperparameter-tuning"
    for framework in ("pytorch", "keras"):
        artifacts = ("demo",)
        for artifact in artifacts:
            notebook = make_notebook(
                title=f"MNIST CNN Tuning — {artifact.title()} ({'KerasTuner on Torch' if framework == 'keras' else 'Optuna and PyTorch'})",
                module_id="hyperparameter-tuning",
                framework=framework,
                artifact=artifact,
                datasets=[],
                cells=tuning_cells(framework, artifact),
            )
            write_notebook(directory / framework, f"{artifact}.ipynb", notebook)


def gan_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    role = "complete workflow"
    smoothing = 0.0
    common_opening = [
        md(
            f"""
# MNIST DCGAN with {"Keras 3 — PyTorch Backend" if framework == "keras" else "Native PyTorch"}

This {role} modernizes the archived DCGAN. A fixed 100-dimensional noise panel
tracks the same generated samples across training. Full execution uses 50
epochs and batch size 128; `HELIO_FAST_RUN=1` uses one reduced epoch.

Losses and selected images are diagnostics, not proof that the generator
learned the complete data distribution.
"""
        ),
        md("## Imports, data, and fixed seeds"),
    ]
    if framework == "keras":
        cells = common_opening + [
            code(
                IMAGE_KERAS_IMPORTS
                + f"""
FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
SEED = 42
EPOCHS = 1 if FAST_RUN else 50
BATCH_SIZE = 128
LATENT_DIM = 100
LABEL_SMOOTHING = {smoothing}
keras.utils.set_random_seed(SEED)
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train[..., np.newaxis]
if FAST_RUN:
    x_train = x_train[:5_000]
dataset = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(SEED),
)
fixed_noise = keras.random.normal((16, LATENT_DIM), seed=SEED)
"""
            ),
            md("## Define generator and discriminator"),
            code(
                r"""
generator = keras.Sequential(
    [
        keras.Input(shape=(LATENT_DIM,)),
        layers.Dense(7 * 7 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, 5, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh"),
    ],
    name="generator",
)
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, 5, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ],
    name="discriminator",
)
class AdversarialModel(keras.Model):
    def __init__(self, generator, discriminator, latent_dim, label_smoothing):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.label_smoothing = label_smoothing
        self.seed_generator = keras.random.SeedGenerator(SEED)
        self.generator_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.built = True

    @property
    def metrics(self):
        return [self.generator_loss_tracker, self.discriminator_loss_tracker]

    def compile(self, generator_optimizer, discriminator_optimizer, loss_function):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_function = loss_function

    def train_step(self, real_images):
        if isinstance(real_images, (tuple, list)):
            real_images = real_images[0]
        batch_size = real_images.shape[0]

        noise = keras.random.normal(
            (batch_size, self.latent_dim), seed=self.seed_generator
        )
        generated_images = self.generator(noise, training=True)

        self.zero_grad()
        real_logits = self.discriminator(real_images, training=True)
        generated_logits = self.discriminator(generated_images.detach(), training=True)
        real_targets = torch.ones_like(real_logits) * (1.0 - self.label_smoothing)
        discriminator_loss = self.loss_function(real_targets, real_logits) + self.loss_function(
            torch.zeros_like(generated_logits), generated_logits
        )
        discriminator_loss.backward()
        discriminator_weights = list(self.discriminator.trainable_weights)
        discriminator_gradients = [weight.value.grad for weight in discriminator_weights]
        with torch.no_grad():
            self.discriminator_optimizer.apply(
                discriminator_gradients, discriminator_weights
            )

        noise = keras.random.normal(
            (batch_size, self.latent_dim), seed=self.seed_generator
        )
        self.zero_grad()
        generated_logits = self.discriminator(
            self.generator(noise, training=True), training=True
        )
        generator_loss = self.loss_function(
            torch.ones_like(generated_logits), generated_logits
        )
        generator_loss.backward()
        generator_weights = list(self.generator.trainable_weights)
        generator_gradients = [weight.value.grad for weight in generator_weights]
        with torch.no_grad():
            self.generator_optimizer.apply(generator_gradients, generator_weights)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
        }


gan = AdversarialModel(generator, discriminator, LATENT_DIM, LABEL_SMOOTHING)
gan.compile(
    generator_optimizer=keras.optimizers.Adam(1e-4),
    discriminator_optimizer=keras.optimizers.Adam(1e-4),
    loss_function=keras.losses.BinaryCrossentropy(from_logits=True),
)
"""
            ),
            md("## Train adversarially through the high-level fit workflow"),
            code(
                r"""
history = gan.fit(dataset, epochs=EPOCHS, verbose=2, shuffle=False)
generator_losses = [float(value) for value in history.history["generator_loss"]]
discriminator_losses = [
    float(value) for value in history.history["discriminator_loss"]
]
"""
            ),
            md("## Fixed-noise evidence"),
            code(
                r"""
generated = generator(fixed_noise, training=False).detach().cpu().numpy()
"""
                + _gan_diagnostics_code()
            ),
        ]
    else:
        cells = common_opening + [
            code(
                IMAGE_TORCH_IMPORTS
                + f"""
FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
SEED = 42
EPOCHS = 1 if FAST_RUN else 50
BATCH_SIZE = 128
LATENT_DIM = 100
LABEL_SMOOTHING = {smoothing}
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
data_root = Path(os.getenv("HELIO_DATA_DIR", Path.home() / ".cache" / "helio-data-methods"))
mnist = datasets.MNIST(data_root, train=True, download=True)
x_train = (mnist.data.numpy().astype("float32") - 127.5) / 127.5
x_train = x_train[:, np.newaxis, ...]
if FAST_RUN:
    x_train = x_train[:5_000]
loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(SEED),
)
fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, generator=torch.Generator().manual_seed(SEED))
"""
            ),
            md("## Define generator and discriminator"),
            code(
                r"""
generator = nn.Sequential(
    nn.ConvTranspose2d(LATENT_DIM, 256, 7, 1, 0, bias=False),
    nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
    nn.ConvTranspose2d(256, 128, 5, 1, 2, bias=False),
    nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
    nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh(),
).to(DEVICE)
discriminator = nn.Sequential(
    nn.Conv2d(1, 64, 5, 2, 2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
    nn.Conv2d(64, 128, 5, 2, 2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
    nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
"""
            ),
            md("## Train adversarially"),
            code(
                r"""
generator_losses, discriminator_losses = [], []
for epoch in range(EPOCHS):
    epoch_generator, epoch_discriminator = [], []
    for (real_images,) in loader:
        real_images = real_images.to(DEVICE)
        batch = len(real_images)
        noise = torch.randn(batch, LATENT_DIM, 1, 1, device=DEVICE)
        generated_images = generator(noise)

        discriminator_optimizer.zero_grad()
        real_logits = discriminator(real_images)
        generated_logits = discriminator(generated_images.detach())
        real_targets = torch.ones_like(real_logits) * (1.0 - LABEL_SMOOTHING)
        discriminator_loss = criterion(real_logits, real_targets) + criterion(
            generated_logits, torch.zeros_like(generated_logits)
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        generated_logits = discriminator(generated_images)
        generator_loss = criterion(generated_logits, torch.ones_like(generated_logits))
        generator_loss.backward()
        generator_optimizer.step()
        epoch_generator.append(generator_loss.item())
        epoch_discriminator.append(discriminator_loss.item())
    generator_losses.append(float(np.mean(epoch_generator)))
    discriminator_losses.append(float(np.mean(epoch_discriminator)))
    print(
        f"epoch {epoch + 1}: generator={generator_losses[-1]:.4f}, "
        f"discriminator={discriminator_losses[-1]:.4f}"
    )
"""
            ),
            md("## Fixed-noise evidence"),
            code(
                r"""
generator.eval()
with torch.no_grad():
    generated = generator(fixed_noise.to(DEVICE)).cpu().numpy()
generated = np.transpose(generated, (0, 2, 3, 1))
"""
                + _gan_diagnostics_code()
            ),
        ]
    return cells


def _gan_diagnostics_code() -> str:
    return r"""
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for axis, image in zip(axes.flat, generated):
    axis.imshow(image.squeeze(), cmap="gray", vmin=-1, vmax=1)
    axis.axis("off")
plt.suptitle("Fixed-noise generated samples")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(generator_losses, label="generator")
ax.plot(discriminator_losses, label="discriminator")
ax.set(title="Adversarial training losses", xlabel="Epoch", ylabel="Loss")
ax.legend()
plt.show()

pixel_diversity = float(generated.std(axis=0).mean())
print(f"mean per-pixel sample standard deviation: {pixel_diversity:.4f}")
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "epochs": EPOCHS,
            "generated_shape": list(generated.shape),
            "generator_loss": generator_losses[-1],
            "discriminator_loss": discriminator_losses[-1],
            "pixel_diversity": pixel_diversity,
        },
        sort_keys=True,
    )
)
assert generated.shape == (16, 28, 28, 1)
assert np.isfinite(generated).all()
assert pixel_diversity > 0
"""


def generate_gan_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "generative-models"
    for framework in ("pytorch", "keras"):
        artifacts = ("demo",)
        for artifact in artifacts:
            notebook = make_notebook(
                title=f"MNIST DCGAN — {artifact.title()} ({'Keras on Torch' if framework == 'keras' else 'Native PyTorch'})",
                module_id="generative-models",
                framework=framework,
                artifact=artifact,
                datasets=[],
                cells=gan_cells(framework, artifact),
            )
            write_notebook(directory / framework, f"{artifact}.ipynb", notebook)


def dataset_bootstrap_code(
    dataset_id: str, files: dict[str, tuple[str, str]]
) -> str:
    manifest = repr(files)
    return f"""
import hashlib
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

DATASET_ID = {dataset_id!r}
DATASET_FILES = {manifest}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset():
    resolved = {{}}
    override = os.getenv("HELIO_DATA_DIR")
    cache_root = Path(
        os.getenv("HELIO_DATA_CACHE", Path.home() / ".cache" / "helio-data-methods")
    ) / "datasets" / DATASET_ID
    for filename, (relative_path, checksum) in DATASET_FILES.items():
        candidates = []
        if override:
            root = Path(override).expanduser()
            candidates.extend([root / DATASET_ID / filename, root / filename])
        for root in [Path.cwd(), *Path.cwd().parents]:
            candidates.append(root / relative_path)
        target = cache_root / filename
        candidates.append(target)
        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file() and file_sha256(candidate) == checksum
            ),
            None,
        )
        if match is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            ref = os.getenv("HELIO_DATA_REF", "main")
            url = (
                "https://raw.githubusercontent.com/SavvasRaptis/helio-data-methods/"
                f"{{quote(ref, safe='')}}/{{quote(relative_path, safe='/')}}"
            )
            try:
                with urlopen(url, timeout=120) as response, target.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            except Exception as exc:
                target.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not retrieve {{DATASET_ID}}/{{filename}}. Check network "
                    "access or set HELIO_DATA_DIR to the archived data directory."
                ) from exc
            if file_sha256(target) != checksum:
                target.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum mismatch for {{DATASET_ID}}/{{filename}}; "
                    "the invalid download was removed."
                )
            match = target
        resolved[filename] = match
    return resolved


dataset_files = resolve_dataset()
print("verified dataset:", DATASET_ID)
for name in dataset_files:
    print(f"  {{name}} (checksum verified)")
"""


SEP_FILES = {
    "x_train.pkl": (
        "data/sep-curated/x_train.pkl",
        "e809bf00498633f509a223d61f9b0006e6ed1803f6de22118bcf654f2ce8ba3b",
    ),
    "x_test.pkl": (
        "data/sep-curated/x_test.pkl",
        "1d0c5f84713d4fde34d567cdb62e9081c4d723f6fef9abd543137376350d5955",
    ),
    "y_train.pkl": (
        "data/sep-curated/y_train.pkl",
        "d7aa048f6b081a9fb1fc00dde19872c0f67ae5b4c8620daa5984b679f9f9dbdc",
    ),
    "y_test.pkl": (
        "data/sep-curated/y_test.pkl",
        "d44c5af108bab2b19f5f8082548282edd8aee89d57e15469516de1ca3f400ee5",
    ),
}

SEP_IMPORTS = r"""
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
random.seed(SEED)
np.random.seed(SEED)
"""

SEP_PREPARE = r"""
x_supplied_train = pd.read_pickle(dataset_files["x_train.pkl"]).to_numpy(dtype=np.float32)
x_test = pd.read_pickle(dataset_files["x_test.pkl"]).to_numpy(dtype=np.float32)
y_supplied_train = (
    pd.read_pickle(dataset_files["y_train.pkl"]).to_numpy().reshape(-1).astype(np.int64)
)
y_test = pd.read_pickle(dataset_files["y_test.pkl"]).to_numpy().reshape(-1).astype(np.int64)
feature_names = np.asarray([f"anonymous feature {i}" for i in range(x_test.shape[1])])

train_indices, validation_indices = train_test_split(
    np.arange(len(y_supplied_train)),
    test_size=0.15,
    random_state=SEED,
    stratify=y_supplied_train,
)
if FAST_RUN:
    train_indices, _ = train_test_split(
        train_indices,
        train_size=min(4000, len(train_indices)),
        random_state=SEED,
        stratify=y_supplied_train[train_indices],
    )
    validation_indices, _ = train_test_split(
        validation_indices,
        train_size=min(1200, len(validation_indices)),
        random_state=SEED,
        stratify=y_supplied_train[validation_indices],
    )

x_train_raw = x_supplied_train[train_indices]
y_train = y_supplied_train[train_indices]
x_validation_raw = x_supplied_train[validation_indices]
y_validation = y_supplied_train[validation_indices]
scaler = StandardScaler().fit(x_train_raw)
x_train = scaler.transform(x_train_raw).astype(np.float32)
x_validation = scaler.transform(x_validation_raw).astype(np.float32)
x_test_scaled = scaler.transform(x_test).astype(np.float32)

counts = np.bincount(y_train, minlength=2)
majority_class = int(np.argmax(counts))
majority_prediction = np.full_like(y_test, majority_class)
class_weights = len(y_train) / (2.0 * np.maximum(counts, 1))
print(
    f"train={len(y_train):,}, validation={len(y_validation):,}, "
    f"supplied test={len(y_test):,}, positive prevalence={y_train.mean():.4f}"
)
print("class weights:", dict(enumerate(class_weights.round(3))))
"""

SEP_METRICS = r"""
def classification_evidence(y_true, probability, label):
    prediction = (probability >= 0.5).astype(np.int64)
    evidence = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }
    print(label, json.dumps(evidence, indent=2))
    print(classification_report(y_true, prediction, digits=3, zero_division=0))
    return evidence


majority_probability = np.full(len(y_test), float(majority_class))
majority_evidence = classification_evidence(
    y_test, majority_probability, "majority-class baseline"
)
"""

SEP_DIAGNOSTICS = r"""
model_evidence = classification_evidence(y_test, probabilities, "model")
matrix = np.asarray(model_evidence["confusion_matrix"])
precision, recall, _ = precision_recall_curve(y_test, probabilities)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].bar([0, 1], np.bincount(y_train, minlength=2))
axes[0].set(title="Training class imbalance", xlabel="Class", ylabel="Samples")
image = axes[1].imshow(matrix, cmap="Blues")
for (row, column), value in np.ndenumerate(matrix):
    axes[1].text(column, row, str(value), ha="center", va="center")
axes[1].set(title="Supplied-test confusion matrix", xlabel="Predicted", ylabel="True")
fig.colorbar(image, ax=axes[1], fraction=0.046)
axes[2].plot(recall, precision)
axes[2].axhline(y_test.mean(), linestyle=":", color="black", label="prevalence")
axes[2].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
axes[2].legend()
plt.tight_layout()
plt.show()

print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "balanced_accuracy": model_evidence["balanced_accuracy"],
            "majority_balanced_accuracy": majority_evidence["balanced_accuracy"],
            "roc_auc": model_evidence["roc_auc"],
            "pr_auc": model_evidence["pr_auc"],
            "prediction_shape": list(probabilities.shape),
        },
        sort_keys=True,
    )
)
assert probabilities.shape == y_test.shape
assert np.isfinite(probabilities).all()
"""


def sep_neural_cells(framework: str) -> list[nbformat.NotebookNode]:
    framework_label = (
        "Keras 3 — PyTorch Backend" if framework == "keras" else "Native PyTorch"
    )
    cells = [
        md(
            f"""
# SEP Occurrence Forecasting — {framework_label}

This research-style workflow uses the archived train/test pickles exactly
as supplied. The 49 predictors are anonymous, and the archive has no event IDs
or timestamps. Consequently, this is sample-level teaching evidence: it cannot
establish event-aware generalization or physical feature attribution.

Set `HELIO_FAST_RUN=1` for a one-epoch smoke run. In Colab, choose
**Runtime → Run all**; the bootstrap downloads and verifies only four pickles.
"""
        ),
        md("## Imports and deterministic configuration"),
        code(SEP_IMPORTS),
        md("## Resolve the immutable archive"),
        code(dataset_bootstrap_code("sep-curated", SEP_FILES)),
        md("## Preserve the supplied test set and split training samples"),
        code(SEP_PREPARE),
        md("## Establish the majority-class baseline"),
        code(SEP_METRICS),
    ]
    if framework == "keras":
        cells.extend(
            [
                md("## Train the weighted Keras classifier"),
                code(
                    r"""
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
from keras import layers

keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True)
assert keras.backend.backend() == "torch"
model = keras.Sequential(
    [
        keras.Input(shape=(x_train.shape[1],)),
        layers.Dense(40, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(30, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=1 if FAST_RUN else 40,
    batch_size=256,
    class_weight={0: class_weights[0], 1: class_weights[1]},
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ],
    verbose=2,
)
probabilities = model.predict(x_test_scaled, verbose=0).reshape(-1)
"""
                ),
                md("## Inspect learning behavior"),
                code(
                    r"""
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(history.history["loss"], label="training")
ax.plot(history.history["val_loss"], label="validation")
ax.set(title="Weighted binary cross-entropy", xlabel="Epoch", ylabel="Loss")
ax.legend()
plt.show()
"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## Train the weighted PyTorch classifier"),
                code(
                    r"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = nn.Sequential(
    nn.Linear(x_train.shape[1], 40, bias=False),
    nn.BatchNorm1d(40),
    nn.ReLU(),
    nn.Linear(40, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
).to(DEVICE)
positive_weight = torch.tensor(
    [counts[0] / max(counts[1], 1)], dtype=torch.float32, device=DEVICE
)
loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
optimizer = torch.optim.Adam(model.parameters())
loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).float()),
    batch_size=256,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
training_losses, validation_losses = [], []
best_state, best_loss, patience_left = None, float("inf"), 5
for epoch in range(1 if FAST_RUN else 40):
    model.train()
    batch_losses = []
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x.to(DEVICE)).squeeze(1)
        loss = loss_function(logits, batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_validation).to(DEVICE)).squeeze(1)
        val_loss = loss_function(
            val_logits, torch.from_numpy(y_validation).float().to(DEVICE)
        ).item()
    training_losses.append(float(np.mean(batch_losses)))
    validation_losses.append(val_loss)
    print(f"epoch {epoch + 1}: loss={training_losses[-1]:.4f}, val={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        patience_left = 5
    else:
        patience_left -= 1
        if patience_left == 0:
            break
model.load_state_dict(best_state)
model.to(DEVICE).eval()
with torch.no_grad():
    probabilities = (
        torch.sigmoid(model(torch.from_numpy(x_test_scaled).to(DEVICE)).squeeze(1))
        .cpu()
        .numpy()
    )
"""
                ),
                md("## Inspect learning behavior"),
                code(
                    r"""
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(training_losses, label="training")
ax.plot(validation_losses, label="validation")
ax.set(title="Weighted binary cross-entropy", xlabel="Epoch", ylabel="Loss")
ax.legend()
plt.show()
"""
                ),
            ]
        )
    cells.extend(
        [
            md("## Evaluate the untouched supplied test set"),
            code(SEP_DIAGNOSTICS),
        ]
    )
    return cells


SEP_XGB_TRAIN = r"""
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=20 if FAST_RUN else 300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=float(counts[0] / max(counts[1], 1)),
    random_state=SEED,
    n_jobs=2,
)
model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
probabilities = model.predict_proba(x_test_scaled)[:, 1]
"""


def sep_framework_neutral_cells(kind: str) -> list[nbformat.NotebookNode]:
    title = {
        "demo": "XGBoost Demonstration",
        "validation": "Repeated Sample-Level Validation",
        "interpretability": "SHAP Demonstration",
    }[kind]
    cells = [
        md(
            f"""
# SEP Occurrence Forecasting — {title}

The archived 49 predictors are anonymous and have no timestamps or event IDs.
Every result below is therefore sample-level teaching evidence, not an
event-aware forecast claim or a physical attribution.

Set `HELIO_FAST_RUN=1` for the reduced smoke configuration.
"""
        ),
        md("## Imports and deterministic configuration"),
        code(SEP_IMPORTS),
        md("## Resolve the immutable archive"),
        code(dataset_bootstrap_code("sep-curated", SEP_FILES)),
        md("## Preserve the supplied test set and split training samples"),
        code(SEP_PREPARE),
    ]
    if kind == "validation":
        cells.extend(
            [
                md("## Repeated stratified validation within supplied training samples"),
                code(
                    r"""
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

features = scaler.fit_transform(x_supplied_train).astype(np.float32)
folds = RepeatedStratifiedKFold(
    n_splits=2 if FAST_RUN else 5,
    n_repeats=1 if FAST_RUN else 3,
    random_state=SEED,
)
scores = []
for fold, (fold_train, fold_validation) in enumerate(
    folds.split(features, y_supplied_train), start=1
):
    fold_scaler = StandardScaler().fit(x_supplied_train[fold_train])
    fold_x_train = fold_scaler.transform(x_supplied_train[fold_train])
    fold_x_validation = fold_scaler.transform(x_supplied_train[fold_validation])
    fold_y_train = y_supplied_train[fold_train]
    fold_counts = np.bincount(fold_y_train, minlength=2)
    fold_model = XGBClassifier(
        n_estimators=20 if FAST_RUN else 200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=float(fold_counts[0] / max(fold_counts[1], 1)),
        random_state=SEED + fold,
        n_jobs=2,
    )
    fold_model.fit(fold_x_train, fold_y_train)
    fold_probability = fold_model.predict_proba(fold_x_validation)[:, 1]
    score = {
        "balanced_accuracy": balanced_accuracy_score(
            y_supplied_train[fold_validation], fold_probability >= 0.5
        ),
        "roc_auc": roc_auc_score(y_supplied_train[fold_validation], fold_probability),
        "pr_auc": average_precision_score(
            y_supplied_train[fold_validation], fold_probability
        ),
    }
    scores.append(score)
    print(f"fold {fold}:", score)
summary = {
    metric: {
        "mean": float(np.mean([score[metric] for score in scores])),
        "std": float(np.std([score[metric] for score in scores])),
    }
    for metric in scores[0]
}
print(json.dumps(summary, indent=2))
print("HELIO_RESULT " + json.dumps({"folds": len(scores), **summary}, sort_keys=True))
"""
                ),
                md(
                    """
## Interpretation

These folds estimate sensitivity to a sample-level partition. Without event
identifiers, they cannot detect leakage between measurements from the same
physical event.
"""
                ),
            ]
        )
        return cells

    if kind == "interpretability":
        cells.extend(
            [
                md("## Establish the majority-class baseline"),
                code(SEP_METRICS),
            ]
        )
    cells.extend(
        [
            md("## Train the weighted boosted-tree model"),
            code(SEP_XGB_TRAIN),
        ]
    )
    if kind == "interpretability":
        cells.extend(
            [
                md("## Summarize anonymous feature influence with SHAP"),
                code(
                    r"""
import warnings

warnings.filterwarnings("ignore", message="IProgress not found.*")
import shap

background_count = min(500, len(x_train))
explain_count = min(300, len(x_test_scaled))
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test_scaled[:explain_count])
if isinstance(shap_values, list):
    shap_values = shap_values[-1]
importance = np.abs(np.asarray(shap_values)).mean(axis=0)
order = np.argsort(importance)[-12:]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feature_names[order], importance[order])
ax.set(title="Mean absolute SHAP value (anonymous columns)", xlabel="mean |SHAP|")
plt.tight_layout()
plt.show()
probabilities = model.predict_proba(x_test_scaled)[:, 1]
model_evidence = classification_evidence(y_test, probabilities, "model")
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "explained_samples": explain_count,
            "shap_shape": list(np.asarray(shap_values).shape),
            "balanced_accuracy": model_evidence["balanced_accuracy"],
        },
        sort_keys=True,
    )
)
"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## Evaluate the untouched supplied test set"),
                code(SEP_METRICS),
                code(SEP_DIAGNOSTICS),
            ]
        )
    return cells


def generate_sep_module() -> None:
    directory = ROOT / "heliophysics" / "research-case-studies" / "sep-occurrence-forecasting"
    for framework in ("keras", "pytorch"):
        framework_title = "Keras 3 on Torch" if framework == "keras" else "Native PyTorch"
        notebook = make_notebook(
            title=f"SEP Occurrence Forecasting — {framework_title}",
            module_id="sep-occurrence-forecasting",
            framework=framework,
            artifact="demo",
            datasets=["sep-curated"],
            cells=sep_neural_cells(framework),
        )
        write_notebook(directory / framework, "demo.ipynb", notebook)
    for kind, filename, library in (
        ("demo", "demo.ipynb", "xgboost"),
        ("validation", "validation.ipynb", "xgboost"),
        ("interpretability", "interpretability.ipynb", "shap"),
    ):
        notebook = make_notebook(
            title=f"SEP Occurrence Forecasting — {kind.title()}",
            module_id="sep-occurrence-forecasting",
            framework="framework-neutral",
            artifact=kind,
            datasets=["sep-curated"],
            cells=sep_framework_neutral_cells(kind),
            library=library,
        )
        write_notebook(directory / "xgboost", filename, notebook)


CORONAL_FILES = {
    "X2D.npy": (
        "data/coronal-loops/X2D.npy",
        "48fdc54c387642ed1dba563b3a27e5aa666327689d0b8db67355aa3015c0d658",
    ),
    "Y2D.npy": (
        "data/coronal-loops/Y2D.npy",
        "f5500bd93a542facd44bb0710abc2880bb0ea8ff7e769f64eb8332898c1bd35b",
    ),
    "LNGTH_L2D.npy": (
        "data/coronal-loops/LNGTH_L2D.npy",
        "38511b43978701e420cfd262fb98dfa2c816dbe8bb7f2ecd2759124073beb988",
    ),
    "DST2D_FP.npy": (
        "data/coronal-loops/DST2D_FP.npy",
        "ff6e05c3679aa876beb390a2fe4531992e2752360b66fe0bcfbb8057bd9c6a0e",
    ),
    "angle_top.npy": (
        "data/coronal-loops/angle_top.npy",
        "4fb89537bb58b44f75ff222dbb53ea8596b1f2900657bbbf3d0eff2fc45796b2",
    ),
    "Z3D.npy": (
        "data/coronal-loops/Z3D.npy",
        "9c04fd96c1c158607259cfbc9ac8e758cfaaf50f627622ee3e9761323b6b8bac",
    ),
}

CORONAL_IMPORTS = r"""
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
FAST_RUN = os.getenv("HELIO_FAST_RUN", "0") == "1"
random.seed(SEED)
np.random.seed(SEED)
"""

CORONAL_PREPARE = r"""
x_coordinates = np.load(dataset_files["X2D.npy"], mmap_mode="r")
y_coordinates = np.load(dataset_files["Y2D.npy"], mmap_mode="r")
z_coordinates = np.load(dataset_files["Z3D.npy"], mmap_mode="r")
length = np.load(dataset_files["LNGTH_L2D.npy"]).astype(np.float32)
footpoint_distance = np.load(dataset_files["DST2D_FP.npy"]).astype(np.float32)
top_angle = np.load(dataset_files["angle_top.npy"]).astype(np.float32)

point_slice = slice(None, None, 10) if FAST_RUN else slice(None)
train_indices = np.arange(0, 300 if FAST_RUN else 3000)
validation_indices = np.arange(3000, 3150 if FAST_RUN else 3750)
test_indices = np.arange(3750, 3900 if FAST_RUN else 5000)


def assemble(indices):
    x = np.asarray(x_coordinates[point_slice, indices], dtype=np.float32).T
    y = np.asarray(y_coordinates[point_slice, indices], dtype=np.float32).T
    z = np.asarray(z_coordinates[point_slice, indices], dtype=np.float32).T
    points = x.shape[1]
    scalars = np.stack(
        [length[indices], footpoint_distance[indices], top_angle[indices]], axis=1
    )
    scalar_channels = np.repeat(scalars[:, None, :], points, axis=1)
    features = np.concatenate([x[..., None], y[..., None], scalar_channels], axis=2)
    return features, z


x_train_raw, y_train_raw = assemble(train_indices)
x_validation_raw, y_validation_raw = assemble(validation_indices)
x_test_raw, y_test_raw = assemble(test_indices)

feature_mean = x_train_raw.mean(axis=(0, 1), keepdims=True)
feature_std = x_train_raw.std(axis=(0, 1), keepdims=True)
feature_std[feature_std < 1e-7] = 1
target_mean = y_train_raw.mean(axis=0, keepdims=True)
target_std = y_train_raw.std(axis=0, keepdims=True)
target_std[target_std < 1e-7] = 1

x_train = ((x_train_raw - feature_mean) / feature_std).astype(np.float32)
x_validation = ((x_validation_raw - feature_mean) / feature_std).astype(np.float32)
x_test = ((x_test_raw - feature_mean) / feature_std).astype(np.float32)
y_train = ((y_train_raw - target_mean) / target_std).astype(np.float32)
y_validation = ((y_validation_raw - target_mean) / target_std).astype(np.float32)
training_mean_prediction = np.repeat(target_mean, len(y_test_raw), axis=0)

assert train_indices.max() < validation_indices.min() < test_indices.min()
print(
    f"train={len(train_indices)}, validation={len(validation_indices)}, "
    f"test={len(test_indices)}, points per loop={y_train.shape[1]}"
)
"""

CORONAL_EVIDENCE = r"""
def regression_evidence(y_true, y_prediction):
    point_mae = float(mean_absolute_error(y_true.ravel(), y_prediction.ravel()))
    point_rmse = float(mean_squared_error(y_true.ravel(), y_prediction.ravel()) ** 0.5)
    loop_mae = np.mean(np.abs(y_true - y_prediction), axis=1)
    loop_rmse = np.sqrt(np.mean((y_true - y_prediction) ** 2, axis=1))
    return {
        "point_mae": point_mae,
        "point_rmse": point_rmse,
        "r2": float(r2_score(y_true.ravel(), y_prediction.ravel())),
        "loop_mae_mean": float(loop_mae.mean()),
        "loop_mae_median": float(np.median(loop_mae)),
        "loop_rmse_mean": float(loop_rmse.mean()),
    }


model_evidence = regression_evidence(y_test_raw, predictions)
baseline_evidence = regression_evidence(y_test_raw, training_mean_prediction)
print("model:", json.dumps(model_evidence, indent=2))
print("training-mean z-profile baseline:", json.dumps(baseline_evidence, indent=2))

residuals = y_test_raw - predictions
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(residuals.ravel(), bins=60)
axes[0].set(title="Point-wise residuals", xlabel="observed z − predicted z")
loop_rmse = np.sqrt(np.mean(residuals**2, axis=1))
axes[1].hist(loop_rmse, bins=30)
axes[1].set(title="Loop-wise RMSE", xlabel="RMSE")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 4))
for panel, index in enumerate([0, len(y_test_raw) // 2, len(y_test_raw) - 1], start=1):
    axis = fig.add_subplot(1, 3, panel, projection="3d")
    axis.plot(
        x_test_raw[index, :, 0],
        x_test_raw[index, :, 1],
        y_test_raw[index],
        label="observed",
    )
    axis.plot(
        x_test_raw[index, :, 0],
        x_test_raw[index, :, 1],
        predictions[index],
        label="reconstructed",
        linestyle="--",
    )
    axis.set_title(f"test loop {test_indices[index]}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
axes = fig.axes
axes[0].legend()
plt.tight_layout()
plt.show()

print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "model_point_rmse": model_evidence["point_rmse"],
            "baseline_point_rmse": baseline_evidence["point_rmse"],
            "model_loop_rmse": model_evidence["loop_rmse_mean"],
            "prediction_shape": list(predictions.shape),
            "split": {"train": [0, 2999], "validation": [3000, 3749], "test": [3750, 4999]},
        },
        sort_keys=True,
    )
)
assert predictions.shape == y_test_raw.shape
assert np.isfinite(predictions).all()
"""


def coronal_neural_cells(framework: str) -> list[nbformat.NotebookNode]:
    framework_label = (
        "Keras 3 — PyTorch Backend" if framework == "keras" else "Native PyTorch"
    )
    cells = [
        md(
            f"""
# Coronal-Loop Reconstruction — {framework_label}

This research workflow reconstructs a loop's z profile from its projected
x/y coordinates and three archived scalar descriptors. It corrects the
overlapping split in the legacy notebook: loops 0–2999 train, 3000–3749
validate, and 3750–4999 form the untouched final test set. All normalization is
fit on training loops only.

Set `HELIO_FAST_RUN=1` to use fewer loops and every tenth point. In Colab,
choose **Runtime → Run all**; the bootstrap verifies each archived array.
"""
        ),
        md("## Imports and deterministic configuration"),
        code(CORONAL_IMPORTS),
        md("## Resolve the immutable arrays"),
        code(dataset_bootstrap_code("coronal-loops", CORONAL_FILES)),
        md("## Assemble features and apply the corrected split"),
        code(CORONAL_PREPARE),
    ]
    if framework == "keras":
        cells.extend(
            [
                md("## Train the Keras convolutional regressor"),
                code(
                    r"""
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
from keras import layers

keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True)
assert keras.backend.backend() == "torch"
model = keras.Sequential(
    [
        keras.Input(shape=x_train.shape[1:]),
        layers.Conv1D(32, 25, padding="same", activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 15, padding="same", activation="relu"),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 7, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(y_train.shape[1]),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=1 if FAST_RUN else 10,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ],
    verbose=2,
)
predictions_scaled = model.predict(x_test, verbose=0)
predictions = predictions_scaled * target_std + target_mean
losses = history.history
"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md("## Train the PyTorch convolutional regressor"),
                code(
                    r"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class LoopRegressor(nn.Module):
    def __init__(self, output_points):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, 25, padding=12),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, output_points)
        )

    def forward(self, inputs):
        return self.regressor(self.features(inputs.transpose(1, 2)))


model = LoopRegressor(y_train.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()
loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
training_losses, validation_losses = [], []
best_state, best_loss, patience_left = None, float("inf"), 2
for epoch in range(1 if FAST_RUN else 10):
    model.train()
    batch_losses = []
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = loss_function(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        val_loss = loss_function(
            model(torch.from_numpy(x_validation).to(DEVICE)),
            torch.from_numpy(y_validation).to(DEVICE),
        ).item()
    training_losses.append(float(np.mean(batch_losses)))
    validation_losses.append(val_loss)
    print(f"epoch {epoch + 1}: loss={training_losses[-1]:.4f}, val={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        patience_left = 2
    else:
        patience_left -= 1
        if patience_left == 0:
            break
model.load_state_dict(best_state)
model.to(DEVICE).eval()
with torch.no_grad():
    predictions_scaled = model(torch.from_numpy(x_test).to(DEVICE)).cpu().numpy()
predictions = predictions_scaled * target_std + target_mean
losses = {"loss": training_losses, "val_loss": validation_losses}
"""
                ),
            ]
        )
    cells.extend(
        [
            md("## Inspect learning curves"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(losses["loss"], label="training")
ax.plot(losses["val_loss"], label="validation")
ax.set(title="Normalized z-profile loss", xlabel="Epoch", ylabel="MSE")
ax.legend()
plt.show()
"""
            ),
            md("## Compare with the training-mean profile"),
            code(CORONAL_EVIDENCE),
            md(
                """
## Scientific boundary

The reconstruction is evaluated only against the supplied arrays. Their
archive does not contain enough provenance to make claims about broader solar
populations, measurement uncertainty, or out-of-distribution performance.
"""
            ),
        ]
    )
    return cells


def generate_coronal_module() -> None:
    directory = ROOT / "heliophysics" / "research-case-studies" / "coronal-loop-reconstruction"
    for framework in ("keras", "pytorch"):
        framework_title = "Keras 3 on Torch" if framework == "keras" else "Native PyTorch"
        notebook = make_notebook(
            title=f"Coronal-Loop Reconstruction — {framework_title}",
            module_id="coronal-loop-reconstruction",
            framework=framework,
            artifact="demo",
            datasets=["coronal-loops"],
            cells=coronal_neural_cells(framework),
        )
        write_notebook(directory / framework, "demo.ipynb", notebook)


def main() -> None:
    generate_dst()
    generate_image_modules()
    generate_tree_module()
    generate_transfer_module()
    generate_tuning_module()
    generate_gan_module()
    generate_sep_module()
    generate_coronal_module()
    from enrich_teaching_notebooks import enrich_all

    enrich_all(
        [
            "dst-forecasting",
            "convolutional-neural-networks",
            "cifar10-cnn-progression",
            "tree-models",
            "transfer-learning",
            "hyperparameter-tuning",
            "generative-models",
        ]
    )


if __name__ == "__main__":
    main()
