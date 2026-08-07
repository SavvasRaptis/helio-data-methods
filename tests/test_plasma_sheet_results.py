from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/plasma-sheet-prime-tm03"
NOTEBOOK = ROOT / "heliophysics/research-case-studies/plasma-sheet-modeling/index.ipynb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_compact_result_bundle_matches_manifest() -> None:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    bundle = DATA_ROOT / manifest["bundle"]["filename"]
    assert _sha256(bundle) == manifest["bundle"]["sha256"]
    assert manifest["derivation"]["temperature_common_samples"] == 46_595
    assert manifest["derivation"]["density_default_case"] == "high_density_northward"
    expected_cases = {
        "low_density_southward": {"index": 0, "n_sw_cm3": 3.0, "bz_nt": -5.0},
        "low_density_northward": {"index": 1, "n_sw_cm3": 3.0, "bz_nt": 5.0},
        "high_density_southward": {"index": 2, "n_sw_cm3": 20.0, "bz_nt": -5.0},
        "high_density_northward": {"index": 3, "n_sw_cm3": 20.0, "bz_nt": 5.0},
    }
    assert manifest["derivation"]["density_cases"] == expected_cases
    assert manifest["derivation"]["valid_density_cells"] == {
        key: 2_236 for key in expected_cases
    }

    with np.load(bundle, allow_pickle=False) as results:
        observed = results["temperature_observed_kev"]
        prime = results["temperature_prime_ps_kev"]
        tm03 = results["temperature_tm03_kev"]
        assert observed.shape == prime.shape == tm03.shape == (46_595,)
        assert results["timestamp_ns"].shape == (46_595,)
        assert results["density_prime_ps_cases_cm3"].shape == (4, 50, 50)
        assert results["density_tm03_cases_cm3"].shape == (4, 50, 50)
        assert results["density_case_keys"].tolist() == list(expected_cases)
        assert np.array_equal(results["density_case_n_sw_cm3"], [3, 3, 20, 20])
        assert np.array_equal(results["density_case_bz_nt"], [-5, 5, -5, 5])
        assert np.array_equal(results["x_edges_re"], np.linspace(-30, -5, 51))
        assert np.array_equal(results["y_edges_re"], np.linspace(-12.5, 12.5, 51))

        for model, predicted in {"prime_ps": prime, "tm03": tm03}.items():
            expected = manifest["derivation"]["temperature_metrics"][model]
            assert np.isclose(mean_absolute_error(observed, predicted), expected["mae_kev"])
            assert np.isclose(
                np.sqrt(mean_squared_error(observed, predicted)), expected["rmse_kev"]
            )
            assert np.isclose(r2_score(observed, predicted), expected["r2"])
            assert np.isclose(
                np.corrcoef(observed, predicted)[0, 1], expected["pearson_r"]
            )


def test_result_notebook_has_verified_outputs() -> None:
    notebook = nbformat.read(NOTEBOOK, 4)
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type in {"display_data", "execute_result"}
    )
    assert image_count == 2
    output_text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert output_text.count("HELIO_RESULTS ") == 1
    assert "FigureCanvasAgg is non-interactive" not in output_text
