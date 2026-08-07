"""Build the compact PRIME-PS/TM03 results bundle used by the book.

The script intentionally accepts all source paths as command-line arguments.
It performs no model training: it aligns already-saved chronological results
and extracts the single synthetic density condition used in the demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


EXPECTED_SOURCE_SHA256 = {
    "prime_csv": "1e3b98c36312fd07f0c8674966a111392a25e18a0e6752b1e9b75cbd440aff3f",
    "tm03_csv": "8348f6d1bbd268742e3a526dd38a8759f82d5a1767dea6594b5084924a902948",
    "tm03_grid": "e2479cbd735e78d1c023380f2e7b6c4b6e3692fe048c4cf9388acc248b0ecb61",
    "prime_grid": "58ff6819518333bc05f1dc0094d0d5d0de96c9ff603ce7daa929e127416c7700",
}
EXPECTED_COMMON_SAMPLES = 46_595
GRID_CASE_INDEX = 3
SMOOTHING_SIGMA = 1.0
DENSITY_CASES = (
    ("low_density_southward", 3.0, -5.0),
    ("low_density_northward", 3.0, 5.0),
    ("high_density_southward", 20.0, -5.0),
    ("high_density_northward", 20.0, 5.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(name: str, path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} source file: {path}")
    observed = sha256(path)
    expected = EXPECTED_SOURCE_SHA256[name]
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {name}: expected {expected}, observed {observed}"
        )
    return observed


def load_temperature_results(
    prime_csv: Path, tm03_csv: Path
) -> dict[str, np.ndarray]:
    prime = pd.read_csv(
        prime_csv, usecols=["datetime", "temp_tar", "temp_pred"]
    ).rename(columns={"datetime": "timestamp"})
    tm03 = pd.read_csv(tm03_csv, usecols=["Epoch", "temp"]).rename(
        columns={"Epoch": "timestamp", "temp": "tm03_pred"}
    )
    prime["timestamp"] = pd.to_datetime(prime["timestamp"], utc=True)
    tm03["timestamp"] = pd.to_datetime(tm03["timestamp"], utc=True)

    if not prime["timestamp"].is_unique or not tm03["timestamp"].is_unique:
        raise ValueError("Temperature result timestamps must be unique")
    if len(prime) != len(tm03) or not prime["timestamp"].equals(tm03["timestamp"]):
        raise ValueError("PRIME-PS and TM03 timestamps are not exactly aligned")

    aligned = prime.merge(tm03, on="timestamp", how="inner", validate="one_to_one")
    values = aligned[["temp_tar", "temp_pred", "tm03_pred"]].to_numpy(dtype=float)
    common = np.isfinite(values).all(axis=1)
    aligned = aligned.loc[common].reset_index(drop=True)
    if len(aligned) != EXPECTED_COMMON_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_COMMON_SAMPLES:,} common samples, found {len(aligned):,}"
        )

    return {
        "timestamp_ns": aligned["timestamp"].astype("int64").to_numpy(),
        "temperature_observed_kev": aligned["temp_tar"].to_numpy(dtype=np.float64),
        "temperature_prime_ps_kev": aligned["temp_pred"].to_numpy(dtype=np.float64),
        "temperature_tm03_kev": aligned["tm03_pred"].to_numpy(dtype=np.float64),
    }


def load_density_maps(prime_grid_path: Path, tm03_grid_path: Path) -> dict[str, np.ndarray]:
    prime_raw = np.load(prime_grid_path, allow_pickle=False)
    tm03_raw = np.load(tm03_grid_path, allow_pickle=False)
    if prime_raw.shape != (4, 250, 300, 4):
        raise ValueError(f"Unexpected PRIME-PS grid shape: {prime_raw.shape}")
    if tm03_raw.shape != (4, 50, 50):
        raise ValueError(f"Unexpected TM03 grid shape: {tm03_raw.shape}")

    # Density is output index 0. Convert PRIME-PS to the TM03 50 x 50 grid
    # exactly as in the supplied figure script.
    prime_density = np.transpose(prime_raw[..., 0], axes=(0, 2, 1))
    prime_density = prime_density[:, 25:275, :]
    prime_density = prime_density.reshape(4, 50, 5, 50, 5).mean(axis=(2, 4))

    for index in range(prime_density.shape[0]):
        grid = prime_density[index]
        nan_mask = np.isnan(grid)
        filled = np.where(nan_mask, np.nanmean(grid), grid)
        smoothed = gaussian_filter(filled, sigma=SMOOTHING_SIGMA)
        smoothed[nan_mask] = np.nan
        prime_density[index] = smoothed

    unified_mask = (
        (tm03_raw < 0.1) | np.isnan(tm03_raw) | np.isnan(prime_density)
    )
    prime_masked = np.where(unified_mask, np.nan, prime_density)
    tm03_masked = np.where(unified_mask, np.nan, tm03_raw)

    return {
        "x_edges_re": np.linspace(-30.0, -5.0, 51, dtype=np.float64),
        "y_edges_re": np.linspace(-12.5, 12.5, 51, dtype=np.float64),
        "density_case_keys": np.array(
            [case[0] for case in DENSITY_CASES], dtype="<U26"
        ),
        "density_case_n_sw_cm3": np.array(
            [case[1] for case in DENSITY_CASES], dtype=np.float64
        ),
        "density_case_bz_nt": np.array(
            [case[2] for case in DENSITY_CASES], dtype=np.float64
        ),
        "density_prime_ps_cases_cm3": prime_masked.astype(np.float64),
        "density_tm03_cases_cm3": tm03_masked.astype(np.float64),
        "smoothing_sigma_cells": np.array(SMOOTHING_SIGMA, dtype=np.float64),
    }


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    return {
        "mae_kev": float(np.mean(np.abs(residual))),
        "rmse_kev": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator),
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a reproducible compressed NumPy archive with fixed ZIP metadata."""

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue())


def build_bundle(args: argparse.Namespace) -> tuple[Path, Path]:
    sources = {
        "prime_csv": args.prime_csv.resolve(),
        "tm03_csv": args.tm03_csv.resolve(),
        "tm03_grid": args.tm03_grid.resolve(),
        "prime_grid": args.prime_grid.resolve(),
    }
    verified = {name: verify_source(name, path) for name, path in sources.items()}
    arrays = load_temperature_results(sources["prime_csv"], sources["tm03_csv"])
    arrays.update(load_density_maps(sources["prime_grid"], sources["tm03_grid"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "plasma_sheet_prime_tm03_results.npz"
    manifest_path = args.output_dir / "manifest.json"
    write_deterministic_npz(bundle_path, arrays)

    observed = arrays["temperature_observed_kev"]
    manifest = {
        "dataset_id": "plasma-sheet-prime-tm03",
        "description": "Saved PRIME-PS and TM03 results for a no-training book demonstration.",
        "manuscript": {
            "title": "Machine Learning Modeling of Earth's Plasma Sheet using Multi-Spacecraft Observations",
            "status": "under review",
        },
        "sources": {
            name: {"filename": path.name, "sha256": verified[name]}
            for name, path in sources.items()
        },
        "derivation": {
            "temperature_split": "chronological final 20 percent; strict data",
            "temperature_common_samples": int(len(observed)),
            "temperature_metrics": {
                "prime_ps": regression_metrics(
                    observed, arrays["temperature_prime_ps_kev"]
                ),
                "tm03": regression_metrics(observed, arrays["temperature_tm03_kev"]),
            },
            "density_default_case": DENSITY_CASES[GRID_CASE_INDEX][0],
            "density_cases": {
                key: {"index": index, "n_sw_cm3": n_sw, "bz_nt": bz}
                for index, (key, n_sw, bz) in enumerate(DENSITY_CASES)
            },
            "prime_density_output_index": 0,
            "prime_y_crop_indices": [25, 275],
            "block_average": [5, 5],
            "gaussian_sigma_cells": SMOOTHING_SIGMA,
            "common_mask": "TM03 < 0.1 or either model is non-finite",
            "valid_density_cells": {
                key: int(
                    np.isfinite(arrays["density_prime_ps_cases_cm3"][index]).sum()
                )
                for index, (key, _, _) in enumerate(DENSITY_CASES)
            },
        },
        "bundle": {"filename": bundle_path.name, "sha256": sha256(bundle_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return bundle_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime-csv", type=Path, required=True)
    parser.add_argument("--tm03-csv", type=Path, required=True)
    parser.add_argument("--tm03-grid", type=Path, required=True)
    parser.add_argument("--prime-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    bundle, manifest = build_bundle(parse_args())
    print(f"Wrote {bundle}")
    print(f"Wrote {manifest}")
