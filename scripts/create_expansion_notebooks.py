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


def md(text: str, *tags: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(dedent(text).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def code(text: str, *tags: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(dedent(text).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


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

    # The dependency check and backend magic matter when a notebook is run, not
    # when it is read, so the book omits them.
    diagnostics.metadata["tags"] = ["remove-cell"]
    inserted_cells = [cells[0], diagnostics, code("%matplotlib inline", "remove-cell")]
    if framework == "keras":
        inserted_cells.append(
            md(
                """
## Keras and PyTorch

This notebook repeats the PyTorch version with the Keras 3 API, running on
the same PyTorch backend. `compile()` sets the optimizer and loss, `fit()`
runs the loop over batches and epochs, and callbacks such as early stopping
replace hand-written control flow. The data split, model, and evaluation
match the PyTorch notebook.
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


SEED = 42
HISTORY_HOURS = HISTORY_HOURS_SETTING
HORIZON_HOURS = 1
EPOCHS = 10  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128  # Hourly windows per parameter update.

random.seed(SEED)
np.random.seed(SEED)
# Build each year partition independently so information never crosses a split.
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

# Estimate the scaling only from 2010–2013.
scaler = StandardScaler().fit(x_train_raw)
x_train = scaler.transform(x_train_raw).astype(np.float32)
x_validation = scaler.transform(x_validation_raw).astype(np.float32)
x_test = scaler.transform(x_test_raw).astype(np.float32)

split_signature = hashlib.sha256(
    time_test.astype("<M8[h]").astype("<i8").tobytes()
    + np.asarray([HISTORY_HOURS, HORIZON_HOURS], dtype="<i8").tobytes()
).hexdigest()[:16]

assert time_train.max() < time_validation.min() < time_test.min()
print(
    f"train={len(y_train):,}, validation={len(y_validation):,}, "
    f"test={len(y_test):,}, history={HISTORY_HOURS} h, horizon={HORIZON_HOURS} h"
)
"""

DST_DISTRIBUTION = r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [2.2, 1]})
axes[0].plot(frame["timestamp"], frame["Dst"], linewidth=0.5, color="0.2")
axes[0].axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2015-01-01"), alpha=0.15, label="validation")
axes[0].axvspan(pd.Timestamp("2015-01-01"), frame["timestamp"].max(), alpha=0.15, color="tab:red", label="test")
axes[0].set(title="Hourly Dst, 2010-2015", ylabel="Dst [nT]")
axes[0].legend(loc="lower left")
axes[1].hist(y_train, bins=60, color="0.4")
axes[1].set(title="Training targets", xlabel="Dst [nT]", yscale="log", ylabel="Hours")
plt.tight_layout()
plt.show()
print(f"hours with Dst < -50 nT: train={np.sum(y_train < -50):,}, test={np.sum(y_test < -50):,}")
"""

DST_METRICS = r"""
def regression_metrics(y_true, y_prediction):
    return {
        "MAE [nT]": float(mean_absolute_error(y_true, y_prediction)),
        "RMSE [nT]": float(mean_squared_error(y_true, y_prediction) ** 0.5),
        "R²": float(r2_score(y_true, y_prediction)),
    }


persistence_metrics = regression_metrics(y_test, persistence_test)
print("persistence on 2015:", {key: round(value, 3) for key, value in persistence_metrics.items()})
"""

DST_EVALUATION = r"""
model_metrics = regression_metrics(y_test, predictions)
# MSE skill score: the fraction of persistence's squared error that the model removes.
persistence_skill = 1.0 - model_metrics["RMSE [nT]"] ** 2 / persistence_metrics["RMSE [nT]"] ** 2
storm_hours = y_test < -50
comparison = pd.DataFrame(
    {
        "neural model": model_metrics,
        "persistence": persistence_metrics,
        "neural model, Dst < -50 nT": regression_metrics(
            y_test[storm_hours], predictions[storm_hours]
        ),
        "persistence, Dst < -50 nT": regression_metrics(
            y_test[storm_hours], persistence_test[storm_hours]
        ),
    }
).T
display(comparison.round(3))
print(f"MSE skill score relative to persistence: {persistence_skill:.3f}")
"""

DST_RECORD = r"""
print(f"split signature: {split_signature}")
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "split_signature": split_signature,
            "model_rmse": model_metrics["RMSE [nT]"],
            "persistence_rmse": persistence_metrics["RMSE [nT]"],
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
def plot_interval(axis, start, stop, title):
    window = (time_test >= np.datetime64(start)) & (time_test < np.datetime64(stop))
    axis.plot(time_test[window], y_test[window], color="black", label="observed")
    axis.plot(time_test[window], predictions[window], label="neural model")
    axis.plot(time_test[window], persistence_test[window], linestyle=":", label="persistence")
    model_rmse = regression_metrics(y_test[window], predictions[window])["RMSE [nT]"]
    persistence_rmse = regression_metrics(y_test[window], persistence_test[window])["RMSE [nT]"]
    axis.set(title=f"{title}  (RMSE: model {model_rmse:.1f}, persistence {persistence_rmse:.1f} nT)", ylabel="Dst [nT]")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")


fig = plt.figure(figsize=(12, 10))
grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
plot_interval(fig.add_subplot(grid[0, :]), "2015-01-01", "2015-02-12", "January 2015")
plot_interval(
    fig.add_subplot(grid[1, :]), "2015-03-16T12", "2015-03-20",
    "St Patrick's Day storm, 17 March 2015",
)
residual_axis = fig.add_subplot(grid[2, 0])
residual_axis.scatter(y_test, predictions - y_test, s=6, alpha=0.3)
residual_axis.axhline(0, color="black", linewidth=1)
residual_axis.set(xlabel="Observed Dst [nT]", ylabel="Model minus observed [nT]", title="Test residuals")
change_axis = fig.add_subplot(grid[2, 1])
observed_change = y_test - persistence_test
change_axis.scatter(observed_change, predictions - persistence_test, s=6, alpha=0.3)
limits = [observed_change.min(), observed_change.max()]
change_axis.plot(limits, limits, color="black", linewidth=1, linestyle=":")
change_axis.set(
    xlabel="Observed 1-h change in Dst [nT]", ylabel="Predicted 1-h change [nT]",
    title="Does the model anticipate changes?",
)
plt.tight_layout()
plt.show()

change_correlation = np.corrcoef(observed_change, predictions - persistence_test)[0, 1]
print(f"correlation between observed and predicted 1-h change: {change_correlation:.2f}")
worst = np.argsort(np.abs(y_test - predictions))[-8:][::-1]
display(
    pd.DataFrame(
        {
            "target time": [str(time_test[index])[:13] for index in worst],
            "observed": y_test[worst],
            "neural model": predictions[worst],
            "persistence": persistence_test[worst],
        }
    ).round(1)
)
"""


def dst_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    history = 3
    framework_label = "Keras 3" if framework == "keras" else "PyTorch"
    cells = [
        md(
            f"""
# Dst Forecasting: {framework_label}

The Dst index measures the depression of the horizontal geomagnetic field at
low latitudes, mainly from the storm-time ring current. This notebook
forecasts hourly Dst one hour ahead from the previous three hours of solar
wind and Dst, and asks a single question: does a small neural network beat
persistence, the forecast that nothing changes?

The data are hourly OMNI2 values for 2010-2015. The model trains on
2010-2013, 2014 serves for early stopping, and 2015, which contains the
strongest storm of solar cycle 24, is the test year. See the
[Dst Forecasting](index.md) chapter for an overview. Set `EPOCHS` to 1 or 2 in
the data cell for a quicker run.
"""
        ),
        md(
            """## Imports

A fixed seed makes repeated runs comparable."""
        ),
        code(DST_IMPORTS, "hide-input"),
        md(
            """## Load the OMNI data

The archived OMNI2 file is downloaded once and checked against its SHA-256
checksum. OMNI solar-wind values are already time-shifted from the upstream
spacecraft to the bow-shock nose, so the inputs at hour $t$ describe the solar
wind arriving at Earth at hour $t$."""
        ),
        code(DST_DATA_BOOTSTRAP, "hide-input"),
        md(
            """## Build forecast windows

Each sample contains the preceding three hourly values of solar-wind speed
$V$, the GSM $B_z$ component, field magnitude $|B|$, and Dst: twelve inputs
in all. The target is Dst one hour after the latest input. Windows that span a
data gap are dropped, and each year partition is windowed separately so no
sample crosses a split boundary. The input scaling is fitted on 2010-2013
only."""
        ),
        code(DST_PREPARE.replace("HISTORY_HOURS_SETTING", str(history))),
        md(
            """## Look at the data

Storms are rare: most hours sit within a few tens of nanotesla of zero, so the
training targets are dominated by quiet conditions."""
        ),
        code(DST_DISTRIBUTION),
        md(
            """## Persistence baseline

For a one-hour forecast, persistence assumes that Dst remains at its value at
the forecast origin: $\\widehat{Dst}(t+1) = Dst(t)$. Dst changes slowly
outside storms, so this is a demanding reference, and a model is useful only
if it beats it."""
        ),
        code(DST_METRICS),
    ]
    if framework == "keras":
        model_cells = [
            md(
                """## Define the Keras model

Two hidden layers of 50 and 30 ReLU units map the twelve inputs to one Dst
value. The loss is the mean squared error."""
            ),
            code(
                r"""
os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
import keras
import torch
from keras import layers

keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
assert keras.backend.backend() == "torch"
# Define the neural network used for the Dst forecast.
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
            md(
                """## Train with early stopping

The model fits on 2010-2013 and is checked on 2014 after each epoch. Training
stops after three epochs without improvement, and the weights with the lowest
validation loss are restored."""
            ),
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
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ],
    verbose=0,
)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(history.history["loss"], marker="o", label="training")
ax.plot(history.history["val_loss"], marker="o", label="validation")
ax.set(title="Mean squared error", xlabel="Epoch", ylabel="MSE [nT²]")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""
            ),
            md(
                """## Evaluate on 2015

The test year is used once, after training is complete. Storm hours
(Dst < -50 nT) are scored separately, because an overall RMSE is dominated by
quiet time."""
            ),
            code(
                "predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)\n"
                + DST_EVALUATION
            ),
        ]
    else:
        model_cells = [
            md(
                """## Define the PyTorch model

Two hidden layers of 50 and 30 ReLU units map the twelve inputs to one Dst
value. The loss is the mean squared error."""
            ),
            code(
                r"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# Define the neural network used for the Dst forecast.
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
            md(
                """## Train with early stopping

The loop fits on 2010-2013 and checks 2014 after each epoch. Training stops
after three epochs without improvement, and the state with the lowest
validation loss is restored."""
            ),
            code(
                r"""
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
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
    if current_validation < best_validation:
        best_validation, stale_epochs = current_validation, 0
        best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    else:
        stale_epochs += 1
        if stale_epochs >= 3:
            break

model.load_state_dict(best_state)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(training_loss, marker="o", label="training")
ax.plot(validation_loss, marker="o", label="validation")
ax.set(title="Mean squared error", xlabel="Epoch", ylabel="MSE [nT²]")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""
            ),
            md(
                """## Evaluate on 2015

The test year is used once, after training is complete. Storm hours
(Dst < -50 nT) are scored separately, because an overall RMSE is dominated by
quiet time."""
            ),
            code(
                r"""
model.eval()
with torch.no_grad():
    predictions = model(torch.from_numpy(x_test).to(DEVICE)).cpu().numpy()
"""
                + DST_EVALUATION
            ),
        ]
    cells.extend(model_cells)
    cells.extend(
        [
            code(DST_RECORD, "remove-cell"),
            md(
                """## Where the errors are

The top panels follow a quiet month and the St Patrick's Day storm, whose
Dst minimum of about -223 nT is the lowest in this six-year record. The lower-left
panel shows residuals against observed Dst; the lower-right panel asks
whether the model predicts the *change* in Dst over the next hour, which is
the only part of the forecast persistence cannot supply."""
            ),
            code(DST_DIAGNOSTICS),
            md(
                """## What the results show

The network beats persistence on 2015 as a whole, with an RMSE of about
3.7 nT against 4.8 nT, and by a wider margin during storm hours. It does so
by predicting part of the change over the next hour: the correlation between
predicted and observed one-hour changes is about 0.6. The largest errors are
the abrupt changes, among them the positive jump at the St Patrick's Day
sudden commencement and the storm minimum itself, which both the network and
persistence miss by more than 30 nT. With one hour of lead time and three
hours of history, the inputs carry little warning of such changes. One test
year with one major storm is a small sample, so the storm-hour scores in
particular are uncertain."""
            ),
        ]
    )
    return cells


def generate_dst() -> None:
    directory = ROOT / "heliophysics" / "applications" / "dst-forecasting"
    for framework in ("pytorch", "keras"):
        artifacts = ("demo",)
        for artifact in artifacts:
            notebook = make_notebook(
                title=f"Dst Forecasting: {'Keras 3' if framework == 'keras' else 'PyTorch'}",
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.

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
print(f"Keras {keras.__version__}, backend {keras.backend.backend()}, PyTorch {torch.__version__}")
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

CLASS_NAMES = {
    "mnist": [str(digit) for digit in range(10)],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}


def image_data_code(framework: str, dataset: str, epochs: int | None, batch_size: int) -> str:
    validation_size = 10_000 if dataset == "mnist" else 5_000
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
            '# PyTorch expects (sample, channel, height, width).\n'
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
torch.use_deterministic_algorithms(True, warn_only=True)
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
"""
        shape = shape_torch
    epochs_line = f"EPOCHS = {epochs}  # Reduce to 1 or 2 for a quicker run.\n" if epochs else ""
    return dedent(
        f"""
SEED = 42
{epochs_line}BATCH_SIZE = {batch_size}
CLASS_NAMES = {CLASS_NAMES[dataset]!r}
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

# Scale pixel values from 0-255 to 0-1.
x_train = x_development[train_indices].astype("float32") / 255.0
y_train = y_development[train_indices].astype(np.int64)
x_validation = x_development[validation_indices].astype("float32") / 255.0
y_validation = y_development[validation_indices].astype(np.int64)
x_test = x_test.astype("float32") / 255.0
y_test = y_test.astype(np.int64)
{shape}

assert set(train_indices).isdisjoint(validation_indices)
print(
    f"train={{len(y_train):,}}, validation={{len(y_validation):,}}, "
    f"test={{len(y_test):,}}, image shape={{x_train.shape[1:]}}"
)
"""
    ).strip()


AS_IMAGE = r"""
def as_image(sample):
    # Convert one stored sample to (height, width[, channel]) for plotting.
    if sample.ndim == 1:
        sample = sample.reshape(28, 28)
    if sample.ndim == 3 and sample.shape[0] in (1, 3):
        sample = np.transpose(sample, (1, 2, 0))
    return sample.squeeze()
"""

IMAGE_DISTRIBUTION = AS_IMAGE + r"""

fig = plt.figure(figsize=(12, 3.6))
grid = fig.add_gridspec(2, 10, height_ratios=[1, 1.2])
for label in range(10):
    axis = fig.add_subplot(grid[0, label])
    image = as_image(x_train[np.flatnonzero(y_train == label)[0]])
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axis.set_title(CLASS_NAMES[label], fontsize=9)
    axis.axis("off")
counts_axis = fig.add_subplot(grid[1, :])
counts_axis.bar(CLASS_NAMES, np.bincount(y_train, minlength=10), color="0.45")
counts_axis.set(ylabel="Training images")
plt.tight_layout()
plt.show()
"""

TORCH_TRAINING_HELPERS = r"""
loss_function = nn.CrossEntropyLoss()


def make_loader(images, labels, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels).long())
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def run_epoch(model, loader, optimizer=None):
    # One pass over `loader`; the weights are updated only if an optimizer is given.
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def fit(model, epochs, learning_rate=1e-3, train=None, validation=None, verbose=True):
    # Train with Adam and restore the epoch with the best validation accuracy.
    train_loader = make_loader(*(train or (x_train, y_train)), shuffle=True)
    validation_loader = make_loader(*(validation or (x_validation, y_validation)))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_state, best_accuracy = None, -1.0
    for epoch in range(epochs):
        scores = (*run_epoch(model, train_loader, optimizer), *run_epoch(model, validation_loader))
        for key, value in zip(history, scores):
            history[key].append(value)
        if verbose:
            print(
                f"epoch {epoch + 1:2d}: loss={scores[0]:.4f}  accuracy={scores[1]:.4f}  "
                f"val_loss={scores[2]:.4f}  val_accuracy={scores[3]:.4f}"
            )
        if scores[3] > best_accuracy:
            best_accuracy = scores[3]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    if verbose:
        best_epoch = int(np.argmax(history["val_accuracy"])) + 1
        print(f"restored epoch {best_epoch} (validation accuracy {best_accuracy:.4f})")
    return history


def predict_classes(model, images):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [model(batch.to(DEVICE)).argmax(dim=1).cpu().numpy()
             for (batch, _) in make_loader(images, np.zeros(len(images), dtype=np.int64))]
        )
"""

KERAS_TRAINING_HELPERS = r"""
def fit(model, epochs, learning_rate=1e-3, train=None, validation=None, verbose=True):
    # Train with Adam and restore the epoch with the best validation accuracy.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    keep_best = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=epochs, restore_best_weights=True
    )
    history = model.fit(
        *(train or (x_train, y_train)),
        validation_data=validation or (x_validation, y_validation),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[keep_best],
        verbose=2 if verbose else 0,
    ).history
    if verbose:
        best_epoch = int(np.argmax(history["val_accuracy"]))
        print(
            f"restored epoch {best_epoch + 1} "
            f"(validation accuracy {history['val_accuracy'][best_epoch]:.4f})"
        )
    return history


def predict_classes(model, images):
    return model.predict(images, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
"""

PLOT_HISTORY = r"""
def plot_history(history, title=None):
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = epochs[int(np.argmax(history["val_accuracy"]))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for axis, key, label in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
        axis.plot(epochs, history[key], marker="o", label="training")
        axis.plot(epochs, history[f"val_{key}"], marker="o", label="validation")
        axis.axvline(best, color="0.6", linestyle=":", label="restored epoch")
        axis.set(title=label, xlabel="Epoch")
        axis.grid(alpha=0.25)
    axes[0].legend()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
"""


def training_helpers(framework: str) -> str:
    helpers = KERAS_TRAINING_HELPERS if framework == "keras" else TORCH_TRAINING_HELPERS
    return helpers + "\n\n" + PLOT_HISTORY


def image_evaluation_code(images: str = "x_test", inputs: str = "x_test") -> str:
    return dedent(
        f"""
test_predictions = predict_classes(model, {inputs})
test_accuracy = accuracy_score(y_test, test_predictions)
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
print(f"test accuracy: {{test_accuracy:.4f}}")
print(
    classification_report(
        y_test, test_predictions, labels=np.arange(10),
        target_names=CLASS_NAMES, digits=3, zero_division=0,
    )
)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    ax=ax, colorbar=False, values_format="d", xticks_rotation=45
)
ax.set_title("Test confusion matrix (rows: true class)")
plt.show()

mistakes = np.flatnonzero(test_predictions != y_test)[:12]
fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
for axis in axes.flat:
    axis.axis("off")
for axis, index in zip(axes.flat, mistakes):
    image = as_image({images}[index])
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axis.set_title(f"{{CLASS_NAMES[y_test[index]]}} as {{CLASS_NAMES[test_predictions[index]]}}", fontsize=9)
fig.suptitle("First twelve test mistakes (true as predicted)")
plt.tight_layout()
plt.show()
"""
    ).strip()


IMAGE_RECORD = r"""
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
model.summary()
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
    name="cifar10_small",
)
model.summary()
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
    name="cifar10_deeper",
)
model.summary()
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
print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
""",
        "cifar_simple": r"""
class SmallNetwork(nn.Module):
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


model = SmallNetwork()
print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
""",
        "cifar_advanced": r"""
class DeeperNetwork(nn.Module):
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


model = DeeperNetwork()
print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
""",
    }
    return definitions[architecture]


def framework_name(framework: str) -> str:
    return "Keras 3" if framework == "keras" else "PyTorch"


SETUP_TEXT = "## Setup\n\nThe libraries used below."

MNIST_SPLIT_TEXT = """## Load MNIST and split it

The 60,000 official training images are split, with a fixed seed and
stratified by class, into 50,000 for training and 10,000 for validation. The
10,000 official test images are used once, at the end."""

CIFAR_SPLIT_TEXT = """## Load CIFAR-10 and split it

The 50,000 official training images are split, with a fixed seed and
stratified by class, into 45,000 for training and 5,000 for validation. The
10,000 official test images are used once, at the end."""

LOOK_TEXT = """## Look at the data

One example per class and the number of training images in each. The classes
are balanced, so accuracy is a fair summary of performance."""


def training_text(framework: str) -> str:
    if framework == "keras":
        return """## Training loop

`fit()` below compiles the model with the Adam optimizer and the
cross-entropy loss, trains it, and uses a callback to restore the weights
from the epoch with the best validation accuracy."""
    return """## Training loop

Each epoch passes once over the training images in mini-batches. For each
batch, `run_epoch` computes the cross-entropy loss (the negative
log-probability given to the correct class), backpropagates its gradient, and
lets the Adam optimizer update the weights. After each epoch `fit` scores the
validation set, and at the end it restores the weights from the epoch with
the best validation accuracy."""


def mnist_cnn_cells(framework: str) -> list[nbformat.NotebookNode]:
    return [
        md(
            f"""
# Convolutional Network for MNIST: {framework_name(framework)}

A dense network treats an image as a list of 784 numbers and ignores which
pixels are neighbours. A convolutional network slides small learned filters
across the image, so the same stroke detector is applied everywhere, and
pooling coarsens the image so that later filters see larger regions. This
notebook trains one on the MNIST digits, with the split used in the
[Neural Networks](../../neural-networks/index.md) chapter. Set `EPOCHS` to 1
or 2 in the data cell for a quicker run.
"""
        ),
        md(SETUP_TEXT),
        code(IMAGE_KERAS_IMPORTS if framework == "keras" else IMAGE_TORCH_IMPORTS, "hide-input"),
        md(MNIST_SPLIT_TEXT),
        code(image_data_code(framework, "mnist", 5, 256)),
        md(LOOK_TEXT),
        code(IMAGE_DISTRIBUTION),
        md(
            """## Define the model

Two 3×3 convolution layers with 32 and 64 filters, each followed by a ReLU
and 2×2 max pooling, turn a 28×28 image into 64 feature maps of 5×5 pixels.
Dropout then zeroes a random quarter of those activations during training, a
guard against relying on any single feature. Two dense layers produce ten
logits, one unnormalized score per digit."""
        ),
        code(image_model_code(framework, "mnist")),
        md(training_text(framework)),
        code(training_helpers(framework)),
        code("history = fit(model, EPOCHS)\nplot_history(history)"),
        md(
            """## Evaluate on the test set

The confusion matrix counts, for each true digit (rows), how often each digit
was predicted (columns). The images below are the first twelve test
mistakes."""
        ),
        code(image_evaluation_code()),
        code(IMAGE_RECORD, "remove-cell"),
        md(
            """## What the results show

The convolutional network misclassifies roughly one test digit in a hundred,
fewer than half the errors of the dense network in the
[Neural Networks](../../neural-networks/index.md) chapter after the same
number of epochs. Weight sharing is the reason: a filter that
detects a stroke in one corner detects it everywhere. Many of the remaining
mistakes are digits a person would also hesitate over."""
        ),
    ]


def cifar_progression_cells(framework: str) -> list[nbformat.NotebookNode]:
    capture = (
        "{name}_model, {name}_history = model, history\n"
        "{name}_validation = max(history['val_accuracy'])"
    )
    return [
        md(
            f"""
# CIFAR-10 CNN Progression: {framework_name(framework)}

CIFAR-10 holds 60,000 colour images of 32×32 pixels in ten classes. Objects
vary in pose, scale, and background, which makes the task much harder than
MNIST. This notebook trains two networks on the same split: a small network
for 5 epochs and a deeper one with more regularization for 25 epochs. The
better of the two on validation accuracy is then evaluated once on the test
set.
"""
        ),
        md(SETUP_TEXT),
        code(IMAGE_KERAS_IMPORTS if framework == "keras" else IMAGE_TORCH_IMPORTS, "hide-input"),
        md(CIFAR_SPLIT_TEXT),
        code(image_data_code(framework, "cifar10", 5, 64)),
        md(LOOK_TEXT),
        code(IMAGE_DISTRIBUTION),
        md(training_text(framework)),
        code(training_helpers(framework)),
        md(
            """## Small network, 5 epochs

Two convolution layers, the second with stride 2 to halve the resolution,
each followed by batch normalization and a leaky ReLU. Batch normalization
rescales each layer's outputs with batch statistics, which stabilizes and
speeds up training. One dense layer of 100 units and dropout precede the
output."""
        ),
        code(image_model_code(framework, "cifar_simple")),
        code(
            "history = fit(model, EPOCHS)\nplot_history(history, 'Small network')\n"
            + capture.format(name="small")
        ),
        md(
            """## Deeper network, 25 epochs

Three convolution stages and dense layers of 600 and 150 units, with dropout
after the convolutions and between the dense layers. The extra capacity lets
the network learn richer features, but also makes it easier to memorize the
training images; the gap between training and validation accuracy shows when
that begins."""
        ),
        code("EPOCHS = 25  # Reduce to 1 or 2 for a quicker run."),
        code(image_model_code(framework, "cifar_advanced")),
        code(
            "history = fit(model, EPOCHS)\nplot_history(history, 'Deeper network')\n"
            + capture.format(name="deeper")
        ),
        md(
            """## Choose on validation accuracy

The choice between the two networks uses the validation set only, so the
test score of the chosen network remains an unbiased estimate."""
        ),
        code(
            r"""
if deeper_validation > small_validation:
    selected_name, model = "deeper network", deeper_model
else:
    selected_name, model = "small network", small_model
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.arange(1, len(small_history["val_accuracy"]) + 1), small_history["val_accuracy"], marker="o", label="small (5 epochs)")
ax.plot(np.arange(1, len(deeper_history["val_accuracy"]) + 1), deeper_history["val_accuracy"], marker="o", label="deeper (25 epochs)")
ax.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
print(
    f"best validation accuracy: small {small_validation:.4f}, deeper {deeper_validation:.4f}; "
    f"selected: {selected_name}"
)
"""
        ),
        md(
            """## Evaluate the chosen network on the test set

The confusion matrix shows which classes are mistaken for which."""
        ),
        code(image_evaluation_code()),
        code(IMAGE_RECORD, "remove-cell"),
        md(
            """## What the results show

The deeper network wins on validation accuracy. Its training accuracy keeps
rising long after validation accuracy has levelled off, the signature of
overfitting, so restoring the best validation epoch matters more here than it
did for MNIST. Errors are not spread evenly: the animal classes, cats in
particular, are confused with one another far more often than the vehicle
classes. Data
augmentation (random flips and crops of the training images) is the usual
next step for closing the gap."""
        ),
    ]


def generate_image_modules() -> None:
    modules = (
        ("convolutional-neural-networks", ROOT / "general-ml" / "foundations" / "convolutional-neural-networks",
         "Convolutional Network for MNIST", mnist_cnn_cells),
        ("cifar10-cnn-progression", ROOT / "general-ml" / "advanced" / "cifar10-cnn-progression",
         "CIFAR-10 CNN Progression", cifar_progression_cells),
    )
    for module_id, directory, title, builder in modules:
        for framework in ("pytorch", "keras"):
            notebook = make_notebook(
                title=f"{title}: {framework_name(framework)}",
                module_id=module_id,
                framework=framework,
                artifact="demo",
                datasets=[],
                cells=builder(framework),
            )
            write_notebook(directory / framework, "demo.ipynb", notebook)


def tree_cells(artifact: str) -> list[nbformat.NotebookNode]:
    depth = 6
    return [
        md(
            """
# MNIST with XGBoost

Gradient-boosted trees are the method of choice for many tabular problems.
Each tree splits the data on thresholds of individual features; boosting adds
trees one at a time, each fitted to the errors of the ensemble so far. Here
each 28×28 image is flattened to 784 pixel features, which throws away the
spatial layout a convolutional network exploits. The comparison shows what
that costs. See [Tree Models and Ensembles](index.md) for the chapter
overview.
"""
        ),
        md(
            """## Setup and data

The MNIST files are downloaded once and verified. The split is the same as
in the neural-network notebooks: 50,000 training, 10,000 validation, and
10,000 test images."""
        ),
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

SEED = 42
CLASS_NAMES = [str(digit) for digit in range(10)]
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
# Flatten each image into 784 pixel features scaled to 0-1.
x_train = x_development[train_indices].reshape(len(train_indices), -1).astype("float32") / 255
y_train = y_development[train_indices]
x_validation = (
    x_development[validation_indices].reshape(len(validation_indices), -1).astype("float32") / 255
)
y_validation = y_development[validation_indices]
x_test = x_test.reshape(len(x_test), -1).astype("float32") / 255
print(f"XGBoost {xgb.__version__}; train={len(y_train):,}, validation={len(y_validation):,}, test={len(y_test):,}")
""",
            "hide-input",
        ),
        md(LOOK_TEXT),
        code(IMAGE_DISTRIBUTION),
        md(
            """## Train with early stopping

Each boosting round adds one tree per class. `eta` shrinks each tree's
contribution, `max_depth` limits how many splits a tree may make, and
`subsample` and `colsample_bytree` fit each tree on a random 80% of the
images and pixels. `alpha` and `lambda` penalize large leaf weights.
Training stops once the validation error has not improved for 10 rounds."""
        ),
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
ROUNDS = 300  # Reduce to 25 or 50 for a quicker run.
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalidation = xgb.DMatrix(x_validation, label=y_validation)
evaluation_log = {{}}
model = xgb.train(
    parameters,
    dtrain,
    num_boost_round=ROUNDS,
    evals=[(dtrain, "training"), (dvalidation, "validation")],
    early_stopping_rounds=10,
    evals_result=evaluation_log,
    verbose_eval=50,
)
print(f"best round: {{model.best_iteration + 1}} of {{ROUNDS}}")

fig, ax = plt.subplots(figsize=(7, 3.4))
for name, log in evaluation_log.items():
    ax.plot(np.arange(1, len(log["merror"]) + 1), log["merror"], label=name)
ax.set(title="Classification error during boosting", xlabel="Boosting round", ylabel="Error", yscale="log")
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""
        ),
        md(
            """## Evaluate on the test set

Predictions use the trees up to the best validation round."""
        ),
        code(
            r"""
probabilities = model.predict(
    xgb.DMatrix(x_test), iteration_range=(0, model.best_iteration + 1)
)
test_predictions = probabilities.argmax(axis=1)
test_accuracy = accuracy_score(y_test, test_predictions)
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
print(f"test accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, test_predictions, digits=3, zero_division=0))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, values_format="d")
ax.set_title("Test confusion matrix (rows: true digit)")
plt.show()
"""
        ),
        code(IMAGE_RECORD, "remove-cell"),
        md(
            """## What the results show

Boosted trees classify flattened pixels well, but they make two to three
times as many errors as the convolutional network on the same split. A tree that splits on pixel
412 learns nothing about pixel 413, so every stroke position has to be
learned separately. On tabular data, where features have no spatial
arrangement, the comparison usually goes the other way."""
        ),
    ]


def generate_tree_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "tree-models"
    notebook = make_notebook(
        title="MNIST with XGBoost",
        module_id="tree-models",
        framework="framework-neutral",
        artifact="demo",
        datasets=[],
        library="xgboost",
        implementation_role="primary",
        cells=tree_cells("demo"),
    )
    write_notebook(directory / "xgboost", "demo.ipynb", notebook)


TRANSFER_TORCH_FEATURES = r"""
from torchvision.models import VGG16_Weights, vgg16

IMAGE_SIZE = 64  # VGG16 was trained on 224-pixel images; 32 pixels leaves a 1x1 map.
HEAD_EPOCHS = 30
SUBSET_SIZES = [1_000, 5_000, 45_000]
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

backbone = vgg16(weights=VGG16_Weights.DEFAULT).features.to(DEVICE).eval()
for parameter in backbone.parameters():
    parameter.requires_grad = False  # Freeze the pretrained filters.


def extract_features(images):
    # Resize, normalize with ImageNet statistics, and average-pool VGG16's last maps.
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), 256):
            batch = torch.from_numpy(images[start : start + 256])
            batch = nn.functional.interpolate(batch, size=IMAGE_SIZE, mode="bilinear")
            batch = ((batch - imagenet_mean) / imagenet_std).to(DEVICE)
            outputs.append(backbone(batch).mean(dim=(2, 3)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


started = time.perf_counter()
features_train = extract_features(x_train)
features_validation = extract_features(x_validation)
features_test = extract_features(x_test)
print(
    f"feature vectors: {features_train.shape[1]} per image; "
    f"extraction took {time.perf_counter() - started:.0f} s"
)


def make_head():
    # The only trainable part of the transfer model.
    return nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 10)
    )
"""

TRANSFER_KERAS_FEATURES = r"""
IMAGE_SIZE = 64  # VGG16 was trained on 224-pixel images; 32 pixels leaves a 1x1 map.
HEAD_EPOCHS = 30
SUBSET_SIZES = [1_000, 5_000, 45_000]
backbone = keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling="avg",
)
backbone.trainable = False  # Freeze the pretrained filters.


def extract_features(images):
    # Resize, apply VGG16's own preprocessing, and average-pool its last maps.
    outputs = []
    for start in range(0, len(images), 256):
        batch = keras.ops.image.resize(images[start : start + 256] * 255.0, (IMAGE_SIZE, IMAGE_SIZE))
        batch = keras.applications.vgg16.preprocess_input(batch)
        outputs.append(keras.ops.convert_to_numpy(backbone(batch, training=False)))
    return np.concatenate(outputs).astype(np.float32)


started = time.perf_counter()
features_train = extract_features(x_train)
features_validation = extract_features(x_validation)
features_test = extract_features(x_test)
print(
    f"feature vectors: {features_train.shape[1]} per image; "
    f"extraction took {time.perf_counter() - started:.0f} s"
)


def make_head():
    # The only trainable part of the transfer model.
    return keras.Sequential(
        [
            keras.Input(shape=(512,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(10),
        ]
    )
"""

TRANSFER_COMPARISON = r"""
rows, heads = [], {}
for size in SUBSET_SIZES:
    subset = np.arange(min(size, len(y_train)))  # The training order is already shuffled.
    head = make_head()
    head_history = fit(
        head, HEAD_EPOCHS,
        train=(features_train[subset], y_train[subset]),
        validation=(features_validation, y_validation),
        verbose=False,
    )
    scratch = make_scratch_network()
    scratch_history = fit(
        scratch, EPOCHS, train=(x_train[subset], y_train[subset]), verbose=False
    )
    heads[size] = head
    rows.append(
        {
            "training images": size,
            "frozen VGG16 + head": max(head_history["val_accuracy"]),
            "small CNN from scratch": max(scratch_history["val_accuracy"]),
        }
    )
    print(f"{size:>6,} images: transfer {rows[-1]['frozen VGG16 + head']:.3f}, "
          f"scratch {rows[-1]['small CNN from scratch']:.3f}")

comparison = pd.DataFrame(rows).set_index("training images")
display(comparison.round(3))
fig, ax = plt.subplots(figsize=(6, 3.8))
for column in comparison:
    ax.plot(comparison.index, comparison[column], marker="o", label=column)
ax.set(
    xscale="log", xlabel="Labelled training images", ylabel="Best validation accuracy",
    title="Transfer learning pays most when labels are scarce",
)
ax.legend()
ax.grid(alpha=0.25)
plt.show()
"""


def transfer_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    imports = (IMAGE_KERAS_IMPORTS if framework == "keras" else IMAGE_TORCH_IMPORTS) + (
        "import time\n\nimport pandas as pd\n"
    )
    scratch = image_model_code(framework, "cifar_simple").replace(
        "model = SmallNetwork()", "def make_scratch_network():\n    return SmallNetwork()\n\n\nmodel = SmallNetwork()"
    ) if framework == "pytorch" else (
        "def make_scratch_network():\n"
        + "\n".join(
            "    " + line if line else line
            for line in image_model_code(framework, "cifar_simple")
            .replace("model = keras.Sequential(", "return keras.Sequential(")
            .replace("model.summary()", "")
            .strip()
            .splitlines()
        )
        + "\n"
    )
    return [
        md(
            f"""
# CIFAR-10 Transfer Learning: {framework_name(framework)}

Transfer learning reuses a network trained on one large dataset as a feature
extractor for a new task. Here the convolutional part of VGG16, trained on 1.3
million ImageNet photographs, is frozen, and only a small classifier on top
of it is trained. The benefit should be largest when labelled examples are
scarce, which is the usual situation for rare heliophysical events. We
therefore train on 1,000, 5,000, and 45,000 CIFAR-10 images and compare with
the small CNN of the [CIFAR-10 progression](../../cifar10-cnn-progression/index.md)
trained from scratch on the same images.

The pretrained weights (about 500 MB) are downloaded on first use. Feature
extraction takes several minutes on a CPU.
"""
        ),
        md(SETUP_TEXT),
        code(imports, "hide-input"),
        md(CIFAR_SPLIT_TEXT),
        code(image_data_code(framework, "cifar10", 10, 256)),
        md(LOOK_TEXT),
        code(IMAGE_DISTRIBUTION),
        md(training_text(framework)),
        code(training_helpers(framework)),
        md(
            """## Extract frozen VGG16 features

Each image is enlarged to 64×64 pixels, normalized as VGG16 expects, and
passed once through the frozen convolutional layers. Averaging the last
feature maps gives a 512-number summary per image. Only the small head
defined here is trained."""
        ),
        code(TRANSFER_KERAS_FEATURES if framework == "keras" else TRANSFER_TORCH_FEATURES),
        md(
            """## The comparison network

The same small CNN as in the CIFAR-10 progression, trained from raw pixels
for 10 epochs."""
        ),
        code(scratch),
        md(
            """## Train both on growing subsets

For each subset size, both models see the same training images and are
scored on the full validation set. The test set is not used here."""
        ),
        code(TRANSFER_COMPARISON),
        md(
            """## Evaluate the transfer model on the test set

The head trained on all 45,000 images is evaluated once."""
        ),
        code(
            "model = heads[SUBSET_SIZES[-1]]\n"
            + image_evaluation_code(images="x_test", inputs="features_test")
        ),
        code(IMAGE_RECORD, "remove-cell"),
        md(
            """## What the results show

The experiment tests a specific prediction. With 1,000 labelled images, the
frozen ImageNet features should give a better classifier than a network that
must learn its filters from those images alone. As the training set grows,
the network trained from scratch should close the gap, because it can learn
features matched to small CIFAR images instead of reusing features learned
on large photographs. Check the table against both statements. If the gap
persists at 45,000 images, fine-tuning the last VGG16 block instead of
keeping all of it frozen is the natural next step."""
        ),
    ]


def generate_transfer_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "transfer-learning"
    for framework in ("pytorch", "keras"):
        notebook = make_notebook(
            title=f"CIFAR-10 Transfer Learning: {framework_name(framework)}",
            module_id="transfer-learning",
            framework=framework,
            artifact="demo",
            datasets=[],
            cells=transfer_cells(framework, "demo"),
        )
        write_notebook(directory / framework, "demo.ipynb", notebook)


def tuning_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    library = "KerasTuner" if framework == "keras" else "Optuna"
    cells = [
        md(
            f"""
# Hyperparameter Tuning with {library}: {framework_name(framework)}

Hyperparameters are the choices made before training: here the number of
filters in each convolution layer, the width of the dense layer, and the
learning rate. A tuner trains one model per configuration and keeps the
configuration with the best validation accuracy. The test set plays no part
in the search; it scores the selected configuration once at the end. Reduce
`TRIALS` or `SEARCH_EPOCHS` for a quicker run.
"""
        ),
        md(SETUP_TEXT),
        code(
            (
                IMAGE_KERAS_IMPORTS
                + r"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

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

import pandas as pd

if importlib.util.find_spec("optuna") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "optuna>=4,<5"]
    )
warnings.filterwarnings("ignore", message="IProgress not found.*")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
"""
            ),
            "hide-input",
        ),
        md(MNIST_SPLIT_TEXT),
        code(image_data_code(framework, "mnist", None, 128)),
        md(training_text(framework)),
        code(training_helpers(framework)),
        md(
            f"""## The search space

Every combination of the choices below defines one candidate model, 24 in
all. The budget allows `TRIALS` of them, each trained for up to
`SEARCH_EPOCHS` epochs and scored by its best validation accuracy.
{"KerasTuner's random search draws configurations uniformly." if framework == "keras" else "Optuna's TPE sampler proposes configurations that resemble the best so far; with only four trials it behaves much like a random search."}"""
        ),
        code(
            r"""
TRIALS = 4  # Reduce to 2 for a quicker search.
SEARCH_EPOCHS = 10  # Reduce to 1 or 2 for quicker trials.
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
                md(f"## Run the search with {library}"),
                code(
                    r"""
def build_model(hp):
    return keras.Sequential(
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


class CompiledHyperModel(kt.HyperModel):
    def build(self, hp):
        model = build_model(hp)
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", SEARCH_SPACE["learning_rate"])),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        return model


tuner = kt.RandomSearch(
    CompiledHyperModel(),
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
    verbose=0,
)
trials = pd.DataFrame(
    [
        {**trial.hyperparameters.values, "best validation accuracy": trial.score}
        for trial in tuner.oracle.get_best_trials(TRIALS)
    ]
)
display(trials.round(4))
best_parameters = tuner.get_best_hyperparameters(1)[0]
"""
                ),
                md(
                    """## Retrain the selected configuration and evaluate once

The selected configuration is retrained from scratch with the same epoch
budget, the best validation epoch is restored, and the model is scored on the
test set."""
                ),
                code(
                    r"""
keras.utils.set_random_seed(SEED)
model = build_model(best_parameters)
history = fit(model, SEARCH_EPOCHS, learning_rate=best_parameters.get("learning_rate"))
plot_history(history)
"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md(f"## Run the search with {library}"),
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


def objective(trial):
    torch.manual_seed(SEED + trial.number)
    model = build_trial_model(trial)
    learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"])
    history = fit(model, SEARCH_EPOCHS, learning_rate=learning_rate, verbose=False)
    return max(history["val_accuracy"])


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=TRIALS)
trials = study.trials_dataframe(attrs=("number", "params", "value"))
trials.columns = [column.removeprefix("params_") for column in trials.columns]
display(trials.rename(columns={"value": "best validation accuracy"}).round(4))
"""
                ),
                md(
                    """## Retrain the selected configuration and evaluate once

The selected configuration is retrained from scratch with the same epoch
budget, the best validation epoch is restored, and the model is scored on the
test set. `optuna.trial.FixedTrial` replays the chosen values through the
same model-building function."""
                ),
                code(
                    r"""
torch.manual_seed(SEED)
model = build_trial_model(optuna.trial.FixedTrial(study.best_params))
history = fit(model, SEARCH_EPOCHS, learning_rate=study.best_params["learning_rate"])
plot_history(history)
"""
                ),
            ]
        )
    cells.extend(
        [
            md(
                """## Evaluate on the test set

The first twelve test mistakes follow the confusion matrix."""
            ),
            code(AS_IMAGE + "\n\n" + image_evaluation_code()),
            code(IMAGE_RECORD, "remove-cell"),
            md(
                """## What the results show

On MNIST the configurations differ by fractions of a percent once the
learning rate is reasonable, so the search mainly rules out the learning
rates that are too high or too low. With four trials the ranking of the
remaining configurations is not reliable: the differences are comparable to
the change caused by a new random seed. A larger budget, or repeated trials
per configuration, is needed before a small difference means anything."""
            ),
        ]
    )
    return cells


def generate_tuning_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "hyperparameter-tuning"
    for framework in ("pytorch", "keras"):
        library = "KerasTuner" if framework == "keras" else "Optuna"
        notebook = make_notebook(
            title=f"Hyperparameter Tuning with {library}: {framework_name(framework)}",
            module_id="hyperparameter-tuning",
            framework=framework,
            artifact="demo",
            datasets=[],
            cells=tuning_cells(framework, "demo"),
        )
        write_notebook(directory / framework, "demo.ipynb", notebook)


GAN_KERAS_IMPORTS = r"""
import json
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
from keras import layers

assert keras.backend.backend() == "torch"
print(f"Keras {keras.__version__}, backend {keras.backend.backend()}, PyTorch {torch.__version__}")
"""

GAN_TORCH_IMPORTS = r"""
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

print(f"PyTorch {torch.__version__}")
"""


def gan_cells(framework: str, artifact: str) -> list[nbformat.NotebookNode]:
    smoothing = 0.0
    common_opening = [
        md(
            f"""
# Generative Adversarial Network for MNIST: {framework_name(framework)}

A generative adversarial network (GAN) trains two networks against each
other. The generator turns a vector of 100 random numbers into a 28×28 image;
the discriminator sees real and generated images and outputs the log-odds
that its input is real. The generator improves by fooling the discriminator,
the discriminator by not being fooled. This is the DCGAN design of Radford et
al. (2016), trained on the 60,000 MNIST training images for 50 epochs with
batches of 128. Reduce `EPOCHS` for a quicker run.
"""
        ),
        md(
            """## Setup and data

Pixel values are scaled to [-1, 1] to match the generator's tanh output. A
fixed set of 16 noise vectors is kept aside so the same samples can be
followed through training."""
        ),
    ]
    if framework == "keras":
        return common_opening + [
            code(
                GAN_KERAS_IMPORTS
                + f"""
SEED = 42
EPOCHS = 50  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128
LATENT_DIM = 100
LABEL_SMOOTHING = {smoothing}
SNAPSHOT_EPOCHS = sorted({{1, max(1, EPOCHS // 10), max(1, EPOCHS // 2), EPOCHS}})
keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train[..., np.newaxis]
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
            md(
                """## Define the generator and discriminator

The generator projects the noise to 7×7×256 and upsamples twice with
transposed convolutions to 28×28×1. The discriminator mirrors it with two
strided convolutions and dropout. `AdversarialModel` overrides `train_step`
so that `fit()` alternates one discriminator update and one generator update
per batch."""
            ),
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

        # Discriminator step: real images are labelled 1, generated images 0.
        noise = keras.random.normal((batch_size, self.latent_dim), seed=self.seed_generator)
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
        with torch.no_grad():
            self.discriminator_optimizer.apply(
                [weight.value.grad for weight in discriminator_weights], discriminator_weights
            )

        # Generator step: reward generated images that the discriminator calls real.
        noise = keras.random.normal((batch_size, self.latent_dim), seed=self.seed_generator)
        self.zero_grad()
        generated_logits = self.discriminator(self.generator(noise, training=True), training=True)
        generator_loss = self.loss_function(torch.ones_like(generated_logits), generated_logits)
        generator_loss.backward()
        generator_weights = list(self.generator.trainable_weights)
        with torch.no_grad():
            self.generator_optimizer.apply(
                [weight.value.grad for weight in generator_weights], generator_weights
            )

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
            md(
                """## Train

A callback stores the images generated from the fixed noise at a few epochs,
so progress can be judged by eye."""
            ),
            code(
                r"""
snapshots = {}


class FixedNoiseSnapshots(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 in SNAPSHOT_EPOCHS:
            images = generator(fixed_noise, training=False)
            snapshots[epoch + 1] = images.detach().cpu().numpy()


history = gan.fit(
    dataset, epochs=EPOCHS, verbose=0, shuffle=False, callbacks=[FixedNoiseSnapshots()]
)
generator_losses = [float(value) for value in history.history["generator_loss"]]
discriminator_losses = [float(value) for value in history.history["discriminator_loss"]]
generated = snapshots[EPOCHS]
"""
            ),
            md(GAN_DIAGNOSTICS_TEXT),
            code(GAN_DIAGNOSTICS),
            code(GAN_RECORD, "remove-cell"),
            md(GAN_RESULTS_TEXT),
        ]
    return common_opening + [
        code(
            GAN_TORCH_IMPORTS
            + f"""
SEED = 42
EPOCHS = 50  # Reduce to 1 or 2 for a quicker run.
BATCH_SIZE = 128
LATENT_DIM = 100
LABEL_SMOOTHING = {smoothing}
SNAPSHOT_EPOCHS = sorted({{1, max(1, EPOCHS // 10), max(1, EPOCHS // 2), EPOCHS}})
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
data_root = Path(os.getenv("HELIO_DATA_DIR", Path.home() / ".cache" / "helio-data-methods"))
mnist = datasets.MNIST(data_root, train=True, download=True)
x_train = (mnist.data.numpy().astype("float32") - 127.5) / 127.5
x_train = x_train[:, np.newaxis, ...]
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
        md(
            """## Define the generator and discriminator

The generator upsamples the noise from 1×1 to 7×7, 14×14, and 28×28 with
transposed convolutions, with batch normalization between them and a tanh
output. The discriminator mirrors it with two strided convolutions and
dropout, and ends in one logit. Both use Adam with a small learning rate, as
GAN training is unstable at larger steps."""
        ),
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
        md(
            """## Train

Each batch performs one discriminator update (real images labelled 1,
generated images 0) and one generator update (rewarding generated images the
discriminator calls real). Images from the fixed noise are stored at a few
epochs."""
        ),
        code(
            r"""
generator_losses, discriminator_losses, snapshots = [], [], {}
for epoch in range(EPOCHS):
    generator.train()
    epoch_generator, epoch_discriminator = [], []
    for (real_images,) in loader:
        real_images = real_images.to(DEVICE)
        noise = torch.randn(len(real_images), LATENT_DIM, 1, 1, device=DEVICE)
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
    if epoch + 1 in SNAPSHOT_EPOCHS:
        generator.eval()
        with torch.no_grad():
            images = generator(fixed_noise.to(DEVICE)).cpu().numpy()
        snapshots[epoch + 1] = np.transpose(images, (0, 2, 3, 1))
        print(
            f"epoch {epoch + 1}: generator loss {generator_losses[-1]:.3f}, "
            f"discriminator loss {discriminator_losses[-1]:.3f}"
        )
generated = snapshots[EPOCHS]
"""
        ),
        md(GAN_DIAGNOSTICS_TEXT),
        code(GAN_DIAGNOSTICS),
        code(GAN_RECORD, "remove-cell"),
        md(GAN_RESULTS_TEXT),
    ]


GAN_DIAGNOSTICS_TEXT = """## Follow the fixed-noise samples

Each row shows the images generated from the same eight noise vectors at a
later epoch. The loss curves follow; for a GAN they show the balance between
the two networks, not the quality of the images."""

GAN_DIAGNOSTICS = r"""
fig, axes = plt.subplots(len(snapshots), 8, figsize=(9, 1.25 * len(snapshots) + 0.4), squeeze=False)
for row, (epoch, images) in zip(axes, sorted(snapshots.items())):
    for axis, image in zip(row, images):
        axis.imshow(image.squeeze(), cmap="gray", vmin=-1, vmax=1)
        axis.set_xticks([])
        axis.set_yticks([])
    row[0].set_ylabel(f"epoch {epoch}", rotation=0, ha="right", va="center")
fig.suptitle("Images generated from fixed noise")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.arange(1, len(generator_losses) + 1), generator_losses, label="generator")
ax.plot(np.arange(1, len(discriminator_losses) + 1), discriminator_losses, label="discriminator")
ax.axhline(np.log(2), color="0.6", linestyle=":", label="generator loss at equilibrium, ln 2")
ax.set(title="Adversarial losses", xlabel="Epoch", ylabel="Binary cross-entropy")
ax.legend()
ax.grid(alpha=0.25)
plt.show()

pixel_diversity = float(generated.std(axis=0).mean())
print(f"mean per-pixel standard deviation across the 16 final samples: {pixel_diversity:.3f}")
"""

GAN_RECORD = r"""
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

GAN_RESULTS_TEXT = """## What the results show

The samples sharpen from blobs into recognizable digits within the first few
epochs and improve slowly after that. The losses settle near a fixed ratio
rather than falling, which is the expected behaviour when neither network
dominates. A GAN has no held-out score as simple as test accuracy: the
per-pixel spread across samples only guards against collapse onto a single
image, and judging whether the generator covers all ten digits in the right
proportions needs a separate classifier or a metric such as the Fréchet
inception distance."""


def generate_gan_module() -> None:
    directory = ROOT / "general-ml" / "advanced" / "generative-models"
    for framework in ("pytorch", "keras"):
        notebook = make_notebook(
            title=f"Generative Adversarial Network for MNIST: {framework_name(framework)}",
            module_id="generative-models",
            framework=framework,
            artifact="demo",
            datasets=[],
            cells=gan_cells(framework, "demo"),
        )
        write_notebook(directory / framework, "demo.ipynb", notebook)


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
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
"""

SEP_PREPARE = r"""
x_supplied_train = pd.read_pickle(dataset_files["x_train.pkl"]).to_numpy(dtype=np.float32)
x_test_raw = pd.read_pickle(dataset_files["x_test.pkl"]).to_numpy(dtype=np.float32)
y_supplied_train = (
    pd.read_pickle(dataset_files["y_train.pkl"]).to_numpy().reshape(-1).astype(np.int64)
)
y_test = pd.read_pickle(dataset_files["y_test.pkl"]).to_numpy().reshape(-1).astype(np.int64)
feature_names = np.asarray([f"feature {i}" for i in range(x_test_raw.shape[1])])

train_indices, validation_indices = train_test_split(
    np.arange(len(y_supplied_train)),
    test_size=0.15,
    random_state=SEED,
    stratify=y_supplied_train,
)
x_train_raw = x_supplied_train[train_indices]
y_train = y_supplied_train[train_indices]
x_validation_raw = x_supplied_train[validation_indices]
y_validation = y_supplied_train[validation_indices]
# Standardize with statistics from the training part only.
scaler = StandardScaler().fit(x_train_raw)
x_train = scaler.transform(x_train_raw).astype(np.float32)
x_validation = scaler.transform(x_validation_raw).astype(np.float32)
x_test = scaler.transform(x_test_raw).astype(np.float32)

counts = np.bincount(y_train, minlength=2)
positive_weight = counts[0] / counts[1]  # About 79 non-events per event.
print(
    f"train={len(y_train):,} ({counts[1]} events), "
    f"validation={len(y_validation):,} ({y_validation.sum()} events), "
    f"test={len(y_test):,} ({y_test.sum()} events)"
)
print(f"event rate in training: {y_train.mean():.4f}; positive-class weight: {positive_weight:.1f}")
"""

SEP_METRICS = r"""
def verification_scores(y_true, probability, threshold):
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    pod = tp / max(tp + fn, 1)  # Probability of detection (recall).
    pofd = fp / max(fp + tn, 1)  # Probability of false detection.
    far = fp / max(tp + fp, 1)  # False-alarm ratio.
    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "hits": int(tp), "misses": int(fn), "false alarms": int(fp),
        "POD": pod, "FAR": far, "TSS": pod - pofd,
        "HSS": 2 * (tp * tn - fn * fp) / hss_denominator if hss_denominator else 0.0,
    }


def choose_threshold(y_true, probability, score):
    # Pick the probability threshold that maximizes `score` on validation data.
    candidates = np.unique(np.quantile(probability, np.linspace(0.5, 0.999, 400)))
    values = [verification_scores(y_true, probability, t)[score] for t in candidates]
    return float(candidates[int(np.argmax(values))])


def bootstrap_intervals(y_true, probability, threshold, repeats=1000):
    # Resample the test set to show how much the scores move with 23 events.
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(repeats):
        sample = rng.integers(0, len(y_true), len(y_true))
        if y_true[sample].sum() == 0:
            continue
        scores = verification_scores(y_true[sample], probability[sample], threshold)
        scores["PR-AUC"] = average_precision_score(y_true[sample], probability[sample])
        draws.append(scores)
    draws = pd.DataFrame(draws)[["POD", "FAR", "TSS", "HSS", "PR-AUC"]]
    return draws.quantile([0.025, 0.975]).T.rename(columns={0.025: "2.5%", 0.975: "97.5%"})
"""

SEP_EVALUATION = r"""
thresholds = {
    "TSS": choose_threshold(y_validation, validation_probabilities, "TSS"),
    "HSS": choose_threshold(y_validation, validation_probabilities, "HSS"),
}
rows = {
    "always 'no event'": verification_scores(y_test, np.zeros(len(y_test)), 0.5),
    "threshold 0.5": verification_scores(y_test, probabilities, 0.5),
}
for score, value in thresholds.items():
    rows[f"threshold {value:.3f} (best {score} on validation)"] = verification_scores(
        y_test, probabilities, value
    )
table = pd.DataFrame(rows).T
table["PR-AUC"] = [y_test.mean()] + [average_precision_score(y_test, probabilities)] * 3
display(table.round(3))
print("95% bootstrap intervals on the test set:")
display(
    pd.concat(
        {
            f"best-{score} threshold": bootstrap_intervals(y_test, probabilities, value)
            for score, value in thresholds.items()
        },
        axis=1,
    ).round(3)
)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for axis, (score, value) in zip(axes[:2], thresholds.items()):
    matrix = confusion_matrix(y_test, probabilities >= value, labels=[0, 1])
    axis.imshow(matrix, cmap="Blues", norm="log")
    for (row, column), count in np.ndenumerate(matrix):
        axis.text(
            column, row, str(count), ha="center", va="center",
            color="white" if count > matrix.max() / 10 else "black",
        )
    axis.set(
        title=f"Best-{score} threshold ({value:.3f})",
        xticks=[0, 1], yticks=[0, 1], xticklabels=["no event", "event"],
        yticklabels=["no event", "event"], xlabel="Forecast", ylabel="Observed",
    )
precision, recall, _ = precision_recall_curve(y_test, probabilities)
axes[2].plot(recall, precision)
axes[2].axhline(y_test.mean(), linestyle=":", color="black", label="event rate")
axes[2].set(title="Precision-recall curve", xlabel="Recall (POD)", ylabel="Precision")
axes[2].legend()
observed, forecast = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
axes[3].plot([0, 1], [0, 1], linestyle=":", color="black", label="perfect reliability")
axes[3].plot(forecast, observed, marker="o", label="model")
axes[3].set(
    title="Reliability (8 equal-count bins)",
    xlabel="Forecast probability", ylabel="Observed event frequency",
)
axes[3].legend()
plt.tight_layout()
plt.show()
"""

SEP_RECORD = r"""
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "thresholds": thresholds,
            "tss_at_best_tss": float(table.iloc[2]["TSS"]),
            "hss_at_best_hss": float(table.iloc[3]["HSS"]),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "prediction_shape": list(probabilities.shape),
        },
        sort_keys=True,
    )
)
assert probabilities.shape == y_test.shape
assert np.isfinite(probabilities).all()
"""

SEP_INTRO = """
The data come from Aminalragia-Giamini et al. (2021), who predicted solar
energetic particle (SEP) events from soft X-ray flare measurements. Each sample
has 49 standardized predictors and a binary label. The archive has no column
names, event identifiers, or timestamps, so two limits apply throughout:
predictors cannot be interpreted physically, and samples from the same event
may fall on both sides of the train/test split.
"""

SEP_SCORES_TEXT = """## Verification scores

SEP events make up about 1.3% of the samples, so a forecast of "no event"
is right 98.7% of the time and useless. We therefore report the scores used
in space-weather verification, computed from hits, misses, and false alarms:

- **POD**, the probability of detection: the fraction of events forecast.
- **FAR**, the false-alarm ratio: the fraction of event forecasts that were wrong.
- **TSS** = POD minus the probability of false detection. It ranges from −1
  to 1 and does not depend on the event rate.
- **HSS**, the Heidke skill score: accuracy relative to random forecasts with
  the same event rate.

A probabilistic model needs a threshold to issue yes/no forecasts. We choose
two on the validation set, one maximizing TSS and one maximizing HSS, and
apply both unchanged to the test set. With only 23 test events, a bootstrap
shows how far each score could move by chance."""

SEP_RESULTS_TEXT = """## Evaluate on the supplied test set

The table compares the model at three thresholds with a forecast of "no
event". The reliability
diagram checks whether a forecast probability of $p$ corresponds to an event
frequency of $p$. Class weighting inflates predicted probabilities, so points
below the diagonal are expected."""


def sep_neural_cells(framework: str) -> list[nbformat.NotebookNode]:
    framework_label = "Keras 3" if framework == "keras" else "PyTorch"
    cells = [
        md(
            f"""
# SEP Occurrence Forecasting: {framework_label}

{SEP_INTRO.strip()}

This notebook trains a small neural classifier and evaluates it with
verification scores suited to rare events.
"""
        ),
        md("## Imports"),
        code(SEP_IMPORTS, "hide-input"),
        md(
            """## Load the data

The four pickles are downloaded once and checked against their SHA-256
checksums."""
        ),
        code(dataset_bootstrap_code("sep-curated", SEP_FILES), "hide-input"),
        md(
            """## Split and standardize

The supplied test set is kept for the final evaluation. A stratified 15% of
the supplied training samples becomes the validation set, used for early
stopping and for choosing the decision threshold."""
        ),
        code(SEP_PREPARE),
        md(SEP_SCORES_TEXT),
        code(SEP_METRICS),
    ]
    if framework == "keras":
        cells.extend(
            [
                md(
                    """## Train a weighted classifier

The network outputs the probability of an SEP event. Class weights make each
event count about 79 times as much as a non-event in the loss, so the rare
class is not ignored. Early stopping restores the epoch with the lowest
validation loss."""
                ),
                code(
                    r"""
os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
import keras
import torch
from keras import layers

EPOCHS = 40  # Reduce to 5 or 10 for a quicker run.
keras.utils.set_random_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
assert keras.backend.backend() == "torch"
# Define the neural network used for SEP occurrence classification.
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
model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=256,
    class_weight={0: 1.0, 1: positive_weight},
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ],
    verbose=0,
)
validation_probabilities = model.predict(x_validation, verbose=0).reshape(-1)
probabilities = model.predict(x_test, verbose=0).reshape(-1)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(history.history["loss"], label="training")
ax.plot(history.history["val_loss"], label="validation")
ax.set(title="Binary cross-entropy", xlabel="Epoch", ylabel="Loss")
ax.legend()
plt.show()
"""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md(
                    """## Train a weighted classifier

The network outputs one logit, the log-odds of an SEP event. The loss weights
each event about 79 times as much as a non-event, so the rare class is not
ignored. Training stops when the validation loss has not improved for five
epochs, and the best state is restored."""
                ),
                code(
                    r"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 40  # Reduce to 5 or 10 for a quicker run.
torch.manual_seed(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Define the neural network used for SEP occurrence classification.
model = nn.Sequential(
    nn.Linear(x_train.shape[1], 40, bias=False),
    nn.BatchNorm1d(40),
    nn.ReLU(),
    nn.Linear(40, 30),
    nn.ReLU(),
    nn.Linear(30, 1),
).to(DEVICE)
loss_function = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=DEVICE)
)
optimizer = torch.optim.Adam(model.parameters())
loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).float()),
    batch_size=256,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
x_validation_tensor = torch.from_numpy(x_validation).to(DEVICE)
y_validation_tensor = torch.from_numpy(y_validation).float().to(DEVICE)
training_losses, validation_losses = [], []
best_state, best_loss, stale_epochs = None, float("inf"), 0
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = loss_function(model(batch_x.to(DEVICE)).squeeze(1), batch_y.to(DEVICE))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch_y)
    model.eval()
    with torch.no_grad():
        validation_loss = loss_function(
            model(x_validation_tensor).squeeze(1), y_validation_tensor
        ).item()
    training_losses.append(total / len(loader.dataset))
    validation_losses.append(validation_loss)
    if validation_loss < best_loss:
        best_loss, stale_epochs = validation_loss, 0
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    else:
        stale_epochs += 1
        if stale_epochs >= 5:
            break
model.load_state_dict(best_state)
model.eval()


def predict_probability(x):
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(DEVICE)).squeeze(1)
    return torch.sigmoid(logits).cpu().numpy()


validation_probabilities = predict_probability(x_validation)
probabilities = predict_probability(x_test)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(training_losses, label="training")
ax.plot(validation_losses, label="validation")
ax.axvline(int(np.argmin(validation_losses)), color="0.6", linestyle=":", label="restored epoch")
ax.set(title="Weighted binary cross-entropy", xlabel="Epoch", ylabel="Loss")
ax.legend()
plt.show()
"""
                ),
            ]
        )
    cells.extend(
        [
            md(SEP_RESULTS_TEXT),
            code(SEP_EVALUATION),
            code(SEP_RECORD, "remove-cell"),
            md(
                """## What the results show

The threshold matters as much as the model. When events are rare, even a
hundred false alarms barely raise the probability of false detection, so
maximizing TSS pushes the threshold down until nearly every event is caught
and most alarms are false. HSS penalizes false alarms more and selects a
higher threshold. Which to use depends on the relative cost of a missed event
and a false alarm, a decision for the forecaster rather than the model.

Read the bootstrap intervals before the point estimates: with 23 events, each
missed event moves POD by about 0.04. Because the split is by sample rather
than by event, all of these scores are likely optimistic."""
            ),
        ]
    )
    return cells


SEP_XGB_TRAIN = r"""
from xgboost import XGBClassifier

ROUNDS = 300  # Reduce to 50 or 100 for a quicker run.
# Define the boosted-tree model used for comparison.
model = XGBClassifier(
    n_estimators=ROUNDS,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=30,
    scale_pos_weight=positive_weight,
    random_state=SEED,
    n_jobs=2,
)
model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
print(f"boosting rounds kept by early stopping: {model.best_iteration + 1} of {ROUNDS}")
validation_probabilities = model.predict_proba(x_validation)[:, 1]
probabilities = model.predict_proba(x_test)[:, 1]
"""

SEP_XGB_TEXT = """## Train a weighted boosted-tree model

XGBoost fits an ensemble of shallow trees, each correcting the errors of the
ones before it. `scale_pos_weight` gives events the same weight as in the
neural notebooks, and early stopping on the validation set chooses the
number of trees."""


def sep_framework_neutral_cells(kind: str) -> list[nbformat.NotebookNode]:
    title, purpose = {
        "demo": (
            "XGBoost",
            "This notebook trains a gradient-boosted tree classifier on the same "
            "split as the neural notebooks and scores it the same way.",
        ),
        "validation": (
            "Repeated Validation",
            "One train/test split gives one number. This notebook repeats "
            "stratified cross-validation within the supplied training set to show "
            "how much the scores vary from one partition to the next.",
        ),
        "interpretability": (
            "SHAP Attributions",
            "This notebook computes SHAP values for the XGBoost model and checks "
            "whether the feature ranking survives a change of random seed.",
        ),
    }[kind]
    cells = [
        md(
            f"""
# SEP Occurrence Forecasting: {title}

{SEP_INTRO.strip()}

{purpose}
"""
        ),
        md("## Imports"),
        code(SEP_IMPORTS, "hide-input"),
        md(
            """## Load the data

The four pickles are downloaded once and checked against their SHA-256
checksums."""
        ),
        code(dataset_bootstrap_code("sep-curated", SEP_FILES), "hide-input"),
        md(
            """## Split and standardize

The supplied test set is kept for the final evaluation. A stratified 15% of
the supplied training samples becomes the validation set. Trees do not need
standardized inputs; we standardize anyway so all SEP notebooks share one
preprocessing step."""
        ),
        code(SEP_PREPARE),
        md(SEP_SCORES_TEXT),
        code(SEP_METRICS),
    ]
    if kind == "validation":
        cells.extend(
            [
                md(
                    """## Repeated stratified cross-validation

Five folds, repeated three times with different shuffles, give 15 estimates.
Each fold keeps the 1.3% event rate. The threshold is fixed at 0.5 here so
that the spread reflects the data partition alone."""
                ),
                code(
                    r"""
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

N_SPLITS = 5  # Reduce to 2 for a quicker validation run.
N_REPEATS = 3  # Reduce to 1 for a quicker validation run.
ROUNDS = 200  # Reduce to 20 or 50 for a quicker validation run.
# Repeat the sample-level validation with the same class balance in each fold.
folds = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=SEED,
)
scores = []
for fold, (fold_train, fold_validation) in enumerate(
    folds.split(x_supplied_train, y_supplied_train), start=1
):
    fold_counts = np.bincount(y_supplied_train[fold_train], minlength=2)
    fold_model = XGBClassifier(
        n_estimators=ROUNDS,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=fold_counts[0] / fold_counts[1],
        random_state=SEED + fold,
        n_jobs=2,
    )
    fold_model.fit(x_supplied_train[fold_train], y_supplied_train[fold_train])
    fold_probability = fold_model.predict_proba(x_supplied_train[fold_validation])[:, 1]
    fold_truth = y_supplied_train[fold_validation]
    score = verification_scores(fold_truth, fold_probability, 0.5)
    score["PR-AUC"] = average_precision_score(fold_truth, fold_probability)
    scores.append(score)

scores = pd.DataFrame(scores, index=pd.RangeIndex(1, len(scores) + 1, name="fold"))
metrics = ["POD", "FAR", "TSS", "HSS", "PR-AUC"]
display(scores[metrics].describe().loc[["mean", "std", "min", "max"]].round(3))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.boxplot([scores[metric] for metric in metrics], tick_labels=metrics)
for position, metric in enumerate(metrics, start=1):
    ax.scatter(np.full(len(scores), position), scores[metric], s=10, color="tab:blue", alpha=0.6)
ax.set(title=f"Scores across {len(scores)} folds (threshold 0.5)", ylim=(0, 1))
ax.grid(alpha=0.25, axis="y")
plt.show()
"""
                ),
                code(
                    r"""
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "folds": len(scores),
            **{
                metric: {"mean": float(scores[metric].mean()), "std": float(scores[metric].std())}
                for metric in metrics
            },
        },
        sort_keys=True,
    )
)
""",
                    "remove-cell",
                ),
                md(
                    """
## What the results show

The fold-to-fold range is the honest uncertainty on any single split. A
difference between two models smaller than this range is not evidence that
one is better. These folds are still sample-level: without event
identifiers, they cannot detect leakage between samples from the same event,
so the scores remain an upper bound on performance for unseen events.
"""
                ),
            ]
        )
        return cells

    cells.extend([md(SEP_XGB_TEXT), code(SEP_XGB_TRAIN)])
    if kind == "interpretability":
        cells.extend(
            [
                md(
                    """## Compute SHAP values

For each test sample, SHAP splits the model's output (in log-odds) into one
contribution per feature. The beeswarm plot shows the contributions of the
twelve most influential features; colour gives the feature value. Because the
features are unnamed, this describes what the model relies on, not the
physics of SEP production."""
                ),
                code(
                    r"""
import warnings

warnings.filterwarnings("ignore", message="IProgress not found.*")
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")
import shap

# Compute model-attribution values for every test sample.
explainer = shap.TreeExplainer(model)
shap_values = np.asarray(explainer.shap_values(x_test))
importance = np.abs(shap_values).mean(axis=0)
shap.summary_plot(
    shap_values, x_test, feature_names=feature_names, max_display=12, show=False
)
plt.title("SHAP values on the test set")
plt.tight_layout()
plt.show()
"""
                ),
                md(
                    """## Is the ranking stable?

An explanation is only useful if it does not change when nothing important
changes. We refit the model with a different random seed, recompute SHAP
values, and compare the two importance rankings."""
                ),
                code(
                    r"""
from scipy.stats import spearmanr

refit = XGBClassifier(**{**model.get_params(), "random_state": SEED + 1})
refit.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
refit_importance = np.abs(np.asarray(shap.TreeExplainer(refit).shap_values(x_test))).mean(axis=0)

top = 10
first_top = set(np.argsort(importance)[-top:])
second_top = set(np.argsort(refit_importance)[-top:])
rank_correlation = spearmanr(importance, refit_importance).statistic
print(f"Spearman rank correlation of mean |SHAP|: {rank_correlation:.3f}")
print(f"features shared by the two top-{top} lists: {len(first_top & second_top)}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(importance, refit_importance, s=15)
limit = 1.05 * max(importance.max(), refit_importance.max())
ax.plot([0, limit], [0, limit], linestyle=":", color="black")
ax.set(
    xlabel=f"mean |SHAP|, seed {SEED}", ylabel=f"mean |SHAP|, seed {SEED + 1}",
    title="Feature importance under two seeds", xlim=(0, limit), ylim=(0, limit),
)
plt.show()
"""
                ),
                code(
                    r"""
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            "explained_samples": int(shap_values.shape[0]),
            "shap_shape": list(shap_values.shape),
            "rank_correlation": float(rank_correlation),
        },
        sort_keys=True,
    )
)
""",
                    "remove-cell",
                ),
                md(
                    """## What the results show

Importance falls off gradually from the top feature rather than resting on
one or two predictors. The ranking is largely preserved under a new seed (a
rank correlation near 0.96, with most of the top ten features shared), so
the model's reliance on these features is not an accident of one fit. What those features measure cannot be recovered from
this archive. Correlated predictors can also share or swap attribution, so a
low rank does not mean a feature is irrelevant."""
                ),
            ]
        )
    else:
        cells.extend(
            [
                md(SEP_RESULTS_TEXT),
                code(SEP_EVALUATION),
                code(SEP_RECORD, "remove-cell"),
                md(
                    """## What the results show

The two validation-chosen thresholds differ by almost two orders of
magnitude, and they produce very different forecasts: one catches nearly
every event at the cost of about five false alarms per hit, the other issues
few false alarms but misses about half of the events. Compare models using
the bootstrap intervals rather than the point values: with 23 test events,
differences of a few hundredths in TSS are within the noise. As in the neural notebooks, the split is by
sample, not by event, so the scores are likely optimistic."""
                ),
            ]
        )
    return cells


def generate_sep_module() -> None:
    directory = ROOT / "heliophysics" / "research-case-studies" / "sep-occurrence-forecasting"
    for framework in ("keras", "pytorch"):
        framework_title = "Keras 3" if framework == "keras" else "PyTorch"
        notebook = make_notebook(
            title=f"SEP Occurrence Forecasting: {framework_title}",
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
            title=f"SEP Occurrence Forecasting: {kind.title()}",
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
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
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

# The arrays store one loop per column; transpose to (loop, point).
point_slice = slice(None)
x_projected = np.asarray(x_coordinates[point_slice], dtype=np.float32).T
y_projected = np.asarray(y_coordinates[point_slice], dtype=np.float32).T
heights = np.asarray(z_coordinates[point_slice], dtype=np.float32).T
loop_count, point_count = heights.shape

# Express each projected loop in its own footpoint frame: `along` runs parallel
# to the line joining the two footpoints and `across` is perpendicular to it.
# The network then sees the shape of a loop, not its position on the disk.
baseline_x = x_projected[:, -1:] - x_projected[:, :1]
baseline_y = y_projected[:, -1:] - y_projected[:, :1]
baseline_length = np.hypot(baseline_x, baseline_y)
unit_x, unit_y = baseline_x / baseline_length, baseline_y / baseline_length
relative_x = x_projected - x_projected[:, :1]
relative_y = y_projected - y_projected[:, :1]
along = relative_x * unit_x + relative_y * unit_y
across = relative_y * unit_x - relative_x * unit_y
descriptors = np.stack([length, footpoint_distance, top_angle], axis=1)
features = np.concatenate(
    [
        along[..., None],
        across[..., None],
        np.repeat(descriptors[:, None, :], point_count, axis=1),
    ],
    axis=2,
).astype(np.float32)
print(
    f"loops={loop_count:,}, points per loop={point_count}, "
    f"input channels={features.shape[2]} (along, across, length, "
    "footpoint distance, apex angle)"
)
"""

CORONAL_SPLITS = r"""
GROUP_SIZE = 50  # Consecutive loops kept together in the grouped split.
rng = np.random.default_rng(SEED)
n_train, n_validation = int(0.60 * loop_count), int(0.15 * loop_count)


def by_fraction(order):
    return (
        np.sort(order[:n_train]),
        np.sort(order[n_train : n_train + n_validation]),
        np.sort(order[n_train + n_validation :]),
    )


group_order = rng.permutation(loop_count // GROUP_SIZE)
grouped_order = np.concatenate(
    [np.arange(group * GROUP_SIZE, (group + 1) * GROUP_SIZE) for group in group_order]
)
splits = {
    "random loops": by_fraction(rng.permutation(loop_count)),
    "grouped loops": by_fraction(grouped_order),
    "spatial block": by_fraction(np.arange(loop_count)),
}
for name, (train, validation, test) in splits.items():
    assert not set(train) & set(test) and not set(validation) & set(test)
    print(f"{name:14s} train={len(train):,}  validation={len(validation):,}  test={len(test):,}")

neighbour_gap = np.abs(heights[1:] - heights[:-1]).mean(axis=1)
print(
    "median mean-absolute height difference between neighbouring loops: "
    f"{np.median(neighbour_gap):.3f} (typical apex height {np.median(heights.max(axis=1)):.1f})"
)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
roles = (("train", "0.75"), ("validation", "tab:blue"), ("test", "tab:red"))
for axis, (name, members) in zip(axes, splits.items()):
    for (role, color), indices in zip(roles, members):
        axis.scatter(
            x_projected[indices, 0], y_projected[indices, 0],
            s=3, color=color, label=role, rasterized=True,
        )
    axis.set(title=name, xlabel="x of first footpoint")
axes[0].set_ylabel("y of first footpoint")
axes[0].legend(markerscale=4, loc="lower right")
fig.suptitle("Where the training, validation, and test loops start")
plt.tight_layout()
plt.show()
"""

CORONAL_NORMALIZE = r"""
def prepare_split(split):
    train, validation, test = split
    # Normalization statistics come from the training loops of this split only.
    feature_mean = features[train].mean(axis=(0, 1), keepdims=True)
    feature_std = features[train].std(axis=(0, 1), keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = heights[train].mean(axis=0, keepdims=True)
    target_std = heights[train].std(axis=0, keepdims=True)
    target_std[target_std < 1e-6] = 1.0

    def scale_x(indices):
        return ((features[indices] - feature_mean) / feature_std).astype(np.float32)

    def scale_y(indices):
        return ((heights[indices] - target_mean) / target_std).astype(np.float32)

    return {
        "x_train": scale_x(train), "y_train": scale_y(train),
        "x_validation": scale_x(validation), "y_validation": scale_y(validation),
        "x_test": scale_x(test), "y_test": heights[test],
        "target_mean": target_mean, "target_std": target_std,
    }
"""

CORONAL_EVALUATE = r"""
def rmse(y_true, y_prediction):
    return float(mean_squared_error(y_true.ravel(), y_prediction.ravel()) ** 0.5)


runs, rows = {}, []
for name, split in splits.items():
    data = prepare_split(split)
    predict, losses = train_regressor(data)
    predictions = predict(data["x_test"]) * data["target_std"] + data["target_mean"]
    # Reference: predict the mean training height at every point along the loop.
    baseline = np.repeat(data["target_mean"], len(split[2]), axis=0)
    model_rmse, baseline_rmse = rmse(data["y_test"], predictions), rmse(data["y_test"], baseline)
    runs[name] = {"predictions": predictions, "losses": losses, "test": split[2]}
    rows.append(
        {
            "split": name,
            "model RMSE": model_rmse,
            "mean-profile RMSE": baseline_rmse,
            "model R²": float(r2_score(data["y_test"].ravel(), predictions.ravel())),
            "skill vs mean profile": 1.0 - model_rmse**2 / baseline_rmse**2,
            "epochs": len(losses["loss"]),
        }
    )
    print(f"{name}: model RMSE {model_rmse:.2f}, mean-profile RMSE {baseline_rmse:.2f}")

summary = pd.DataFrame(rows).set_index("split")
display(summary.round(3))
"""

CORONAL_DIAGNOSTICS = r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for axis, (name, run) in zip(axes, runs.items()):
    axis.plot(run["losses"]["loss"], marker="o", label="training")
    axis.plot(run["losses"]["val_loss"], marker="o", label="validation")
    axis.set(title=name, xlabel="Epoch", yscale="log")
    axis.grid(alpha=0.25)
axes[0].set_ylabel("MSE (normalized height)")
axes[0].legend()
plt.tight_layout()
plt.show()

grouped = runs["grouped loops"]
grouped_heights = heights[grouped["test"]]
loop_rmse = np.sqrt(np.mean((grouped_heights - grouped["predictions"]) ** 2, axis=1))
examples = np.argsort(loop_rmse)[[len(loop_rmse) // 10, len(loop_rmse) // 2, -len(loop_rmse) // 10]]

fig = plt.figure(figsize=(13, 4))
for panel, (index, label) in enumerate(zip(examples, ("10th", "50th", "90th")), start=1):
    loop = grouped["test"][index]
    axis = fig.add_subplot(1, 3, panel, projection="3d")
    axis.plot(x_projected[loop], y_projected[loop], heights[loop], label="true")
    axis.plot(
        x_projected[loop], y_projected[loop], grouped["predictions"][index],
        linestyle="--", label="reconstructed",
    )
    axis.set(title=f"loop {loop}: {label} percentile RMSE", xlabel="x", ylabel="y", zlabel="z")
fig.axes[0].legend()
plt.tight_layout()
plt.show()
"""

CORONAL_RECORD = r"""
print(
    "HELIO_RESULT "
    + json.dumps(
        {
            split: {
                "model_rmse": row["model RMSE"],
                "baseline_rmse": row["mean-profile RMSE"],
            }
            for split, row in summary.to_dict(orient="index").items()
        },
        sort_keys=True,
    )
)
for run in runs.values():
    assert run["predictions"].shape == (len(run["test"]), point_count)
    assert np.isfinite(run["predictions"]).all()
"""

CORONAL_TORCH_MODEL = r"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 10  # Reduce to 3 or 5 for a quicker run.
PATIENCE = 3
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# Define the neural network used to reconstruct the loop height profile:
# 1-D convolutions read the ordered points, a dense head returns all heights.
class LoopRegressor(nn.Module):
    def __init__(self, input_channels, output_points):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, 25, padding=12), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 15, padding=7), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, output_points)
        )

    def forward(self, inputs):
        return self.regressor(self.features(inputs.transpose(1, 2)))


def train_regressor(data):
    torch.manual_seed(SEED)
    model = LoopRegressor(features.shape[2], point_count).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["x_train"]), torch.from_numpy(data["y_train"])),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    x_validation = torch.from_numpy(data["x_validation"]).to(DEVICE)
    y_validation = torch.from_numpy(data["y_validation"]).to(DEVICE)
    losses = {"loss": [], "val_loss": []}
    best_state, best_loss, stale_epochs = None, float("inf"), 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(x_validation), y_validation).item()
        losses["loss"].append(total / len(loader.dataset))
        losses["val_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(x):
        with torch.no_grad():
            return model(torch.from_numpy(x).to(DEVICE)).cpu().numpy()

    return predict, losses
"""

CORONAL_KERAS_MODEL = r"""
os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Deterministic cuBLAS on GPUs.
import keras
import torch
from keras import layers

EPOCHS = 10  # Reduce to 3 or 5 for a quicker run.
PATIENCE = 3
torch.use_deterministic_algorithms(True, warn_only=True)
assert keras.backend.backend() == "torch"


def train_regressor(data):
    keras.utils.set_random_seed(SEED)
    # Define the neural network used to reconstruct the loop height profile:
    # 1-D convolutions read the ordered points, a dense head returns all heights.
    model = keras.Sequential(
        [
            keras.Input(shape=data["x_train"].shape[1:]),
            layers.Conv1D(32, 25, padding="same", activation="relu"),
            layers.MaxPooling1D(4),
            layers.Conv1D(64, 15, padding="same", activation="relu"),
            layers.MaxPooling1D(4),
            layers.Conv1D(64, 7, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation="relu"),
            layers.Dense(point_count),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_validation"], data["y_validation"]),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            )
        ],
        verbose=0,
    )

    def predict(x):
        return model.predict(x, batch_size=256, verbose=0)

    return predict, history.history
"""


def coronal_neural_cells(framework: str) -> list[nbformat.NotebookNode]:
    framework_label = "Keras 3" if framework == "keras" else "PyTorch"
    cells = [
        md(
            f"""
# Coronal-Loop Reconstruction: {framework_label}

EUV imagers see coronal loops only in projection. This notebook asks
whether a network can recover the height profile $z(s)$ of a loop from its
projected shape and three scalar descriptors, following Chifu and Gafeira
(2021). The data are 5,000 loops, each sampled at 1,500 points.

The answer depends on which loops the model is tested on. Neighbouring loops
in the archive are nearly identical, and the archive is ordered by position,
so we compare three splits: random loops, groups of 50 consecutive loops, and
one contiguous spatial block.
"""
        ),
        md(
            """## Imports

A fixed seed makes repeated runs comparable."""
        ),
        code(CORONAL_IMPORTS, "hide-input"),
        md(
            """## Load the loops

The arrays are downloaded once and checked against their SHA-256 checksums."""
        ),
        code(dataset_bootstrap_code("coronal-loops", CORONAL_FILES), "hide-input"),
        md(
            """## Describe each loop by its shape

A loop's height depends on its geometry, not on where it sits on the disk. We
therefore rotate and translate each projected loop into a frame fixed by its
footpoints and add the archived projected length, footpoint separation, and
apex angle as extra channels. The target is the height at each of the 1,500
points."""
        ),
        code(CORONAL_PREPARE),
        md(
            """## Three ways to split the loops

- **Random loops** assigns individual loops at random. Because a loop's
  neighbours are near-copies of it, the test set contains close relatives of
  training loops, and the score is optimistic.
- **Grouped loops** assigns blocks of 50 consecutive loops at random. Close
  relatives stay on the same side of the split, while every region still
  appears in training. This is our main estimate.
- **Spatial block** keeps the archive order: the first 60% train, the next 15%
  validate, and the last 25% test. The test loops come from a region the model
  has never seen, so this measures extrapolation."""
        ),
        code(CORONAL_SPLITS),
        md(
            """## Normalize within each split

Inputs are standardized per channel and heights per point, using the training
loops of the split in question."""
        ),
        code(CORONAL_NORMALIZE),
        md(
            f"""## Define the {framework_label} regressor

Three convolution and pooling stages summarize the loop; a dense layer maps
that summary to all 1,500 heights. Training stops when the validation loss has
not improved for three epochs, and the best state is restored."""
        ),
        code(CORONAL_KERAS_MODEL if framework == "keras" else CORONAL_TORCH_MODEL),
        md(
            """## Train and evaluate under each split

The reference prediction is the mean training height at each point along the
loop. A skill above zero means the model beats that profile; R² is computed
over all points of all test loops."""
        ),
        code(CORONAL_EVALUATE),
        md(
            """## Learning curves and example reconstructions

The upper panels compare training and validation loss for each split. The
lower panels show grouped-split test loops at the 10th, 50th, and 90th
percentiles of loop RMSE."""
        ),
        code(CORONAL_DIAGNOSTICS),
        code(CORONAL_RECORD, "remove-cell"),
        md(
            """
## What the results show

With random loops, the network appears almost perfect. That number is inflated:
most test loops have a neighbour in the training set whose heights differ by
about 0.2, or 2% of a typical apex height. Grouping neighbouring loops removes this leak;
the grouped score is the honest estimate for loops drawn from the regions the
model was trained on.

The spatial block is a different question. Its test loops are shorter and
relatively taller than typical training loops, and the model does not beat the
mean profile there. More data or physical constraints, not a larger network,
are what would be needed to reconstruct loops from an unseen region.
"""
        ),
    ]
    return cells


def generate_coronal_module() -> None:
    directory = ROOT / "heliophysics" / "research-case-studies" / "coronal-loop-reconstruction"
    for framework in ("keras", "pytorch"):
        framework_title = "Keras 3" if framework == "keras" else "PyTorch"
        notebook = make_notebook(
            title=f"Coronal-Loop Reconstruction: {framework_title}",
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
