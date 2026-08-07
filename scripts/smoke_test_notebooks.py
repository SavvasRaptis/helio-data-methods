"""Execute published notebooks in isolated fast-run mode.

The script reads the reader-facing TOC, so hidden placeholders are never
executed accidentally. Source notebooks are loaded into memory and executed
there, so smoke testing never rewrites published outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import nbformat
import yaml
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def toc_files(nodes: list[object]) -> list[str]:
    files: list[str] = []
    for node in nodes:
        if isinstance(node, dict):
            if isinstance(node.get("file"), str):
                files.append(node["file"])
            for value in node.values():
                if isinstance(value, list):
                    files.extend(toc_files(value))
        elif isinstance(node, list):
            files.extend(toc_files(node))
    return files


def published_notebooks(module_filter: str | None) -> list[Path]:
    toc = yaml.safe_load((ROOT / "_toc.yml").read_text(encoding="utf-8"))
    paths: list[Path] = []
    for entry in toc_files([toc]):
        path = ROOT / f"{entry}.ipynb"
        if not path.is_file():
            continue
        notebook = nbformat.read(path, as_version=4)
        metadata = notebook.metadata.get("helio_data_methods", {})
        if module_filter and metadata.get("module_id") != module_filter:
            continue
        paths.append(path)
    if module_filter and not paths:
        raise ValueError(f"No published notebooks found for module {module_filter!r}")
    return paths


def output_text(notebook: nbformat.NotebookNode) -> str:
    pieces: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                pieces.append(output.get("text", ""))
            elif output.output_type in {"execute_result", "display_data"}:
                plain = output.get("data", {}).get("text/plain", "")
                pieces.extend(plain if isinstance(plain, list) else [plain])
    return "\n".join(pieces)


def validate_experiment_evidence(path: Path, source: str, text: str) -> None:
    """Require structured evidence from every canonical guided experiment."""

    if "HELIO_EXPERIMENT" not in source:
        return
    payloads = []
    for line in text.splitlines():
        if line.startswith("HELIO_EXPERIMENT "):
            try:
                payloads.append(json.loads(line.removeprefix("HELIO_EXPERIMENT ")))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: invalid HELIO_EXPERIMENT JSON") from exc
    if len(payloads) != 1:
        raise RuntimeError(f"{path}: expected one HELIO_EXPERIMENT output, found {len(payloads)}")
    required = {
        "experiment_id",
        "configurations",
        "budget",
        "comparison_metrics",
        "test_used",
    }
    if set(payloads[0]) != required:
        raise RuntimeError(
            f"{path}: HELIO_EXPERIMENT keys differ from {sorted(required)!r}"
        )
    evidence = payloads[0]
    if not isinstance(evidence["experiment_id"], str) or not evidence["experiment_id"]:
        raise RuntimeError(f"{path}: experiment_id must be a non-empty string")
    if not isinstance(evidence["configurations"], list) or len(evidence["configurations"]) != 2:
        raise RuntimeError(f"{path}: guided experiment must report two configurations")
    if not all(isinstance(item, dict) and item for item in evidence["configurations"]):
        raise RuntimeError(f"{path}: configurations must be non-empty mappings")
    if not isinstance(evidence["budget"], dict) or not evidence["budget"]:
        raise RuntimeError(f"{path}: experiment budget must be a non-empty mapping")
    if not isinstance(evidence["comparison_metrics"], list) or not evidence["comparison_metrics"]:
        raise RuntimeError(f"{path}: comparison_metrics must be a non-empty list")
    if evidence["test_used"] is not False:
        raise RuntimeError(f"{path}: guided experiment must report test_used=false")


def validate_results_evidence(path: Path, source: str, text: str) -> None:
    """Validate structured evidence from a no-training results demonstration."""

    if "HELIO_RESULTS" not in source:
        return
    payloads = []
    for line in text.splitlines():
        if line.startswith("HELIO_RESULTS "):
            try:
                payloads.append(json.loads(line.removeprefix("HELIO_RESULTS ")))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: invalid HELIO_RESULTS JSON") from exc
    if len(payloads) != 1:
        raise RuntimeError(f"{path}: expected one HELIO_RESULTS output, found {len(payloads)}")
    evidence = payloads[0]
    required = {
        "dataset_id",
        "bundle_sha256",
        "sample_count",
        "split",
        "metrics",
        "map_condition",
        "training_run",
    }
    if set(evidence) != required:
        raise RuntimeError(f"{path}: HELIO_RESULTS keys differ from {sorted(required)!r}")
    if evidence["dataset_id"] != "plasma-sheet-prime-tm03":
        raise RuntimeError(f"{path}: unexpected results dataset")
    if evidence["sample_count"] != 46_595:
        raise RuntimeError(f"{path}: unexpected common chronological sample count")
    if evidence["split"] != "chronological-final-20-percent-strict":
        raise RuntimeError(f"{path}: unexpected results split")
    if set(evidence["metrics"]) != {"PRIME-PS", "TM03"}:
        raise RuntimeError(f"{path}: both model metric records are required")
    if evidence["map_condition"].get("case_key") != "high_density_northward":
        raise RuntimeError(f"{path}: unexpected default density-map case")
    if evidence["training_run"] is not False:
        raise RuntimeError(f"{path}: results demonstration must report training_run=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        help="Execute one published module ID; omit to execute every published notebook.",
    )
    args = parser.parse_args()
    notebooks = published_notebooks(args.module)
    signatures: dict[str, set[str]] = {}

    with tempfile.TemporaryDirectory(prefix="helio-notebook-smoke-") as temporary:
        temporary_path = Path(temporary)
        cache_root = Path(
            os.getenv("HELIO_SMOKE_CACHE", ROOT / ".helio-cache")
        ).expanduser()
        environment = {
            "HELIO_FAST_RUN": "1",
            "HELIO_DATA_CACHE": str(cache_root / "heliophysics"),
            "HELIO_DATA_DIR": str(cache_root / "torchvision"),
            "KERAS_HOME": str(cache_root / "keras"),
            "MPLCONFIGDIR": str(temporary_path / "matplotlib"),
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        }
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        try:
            for path in notebooks:
                print(f"running {path.relative_to(ROOT)}", flush=True)
                notebook = nbformat.read(path, as_version=4)
                metadata = notebook.metadata["helio_data_methods"]
                client = NotebookClient(
                    notebook,
                    timeout=1800,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(path.parent)}},
                )
                executed = client.execute()
                text = output_text(executed)
                source = "\n".join(cell.get("source", "") for cell in notebook.cells)
                if "FigureCanvasAgg is non-interactive" in text:
                    raise RuntimeError(f"{path}: Matplotlib used a non-interactive backend")
                if "plt.show()" in source:
                    image_count = sum(
                        "image/png" in output.get("data", {})
                        or "image/svg+xml" in output.get("data", {})
                        for cell in executed.cells
                        for output in cell.get("outputs", [])
                        if output.output_type in {"display_data", "execute_result"}
                    )
                    if image_count == 0:
                        raise RuntimeError(f"{path}: plotting code produced no figure output")
                if "HELIO_RESULT" in source and "HELIO_RESULT" not in text:
                    raise RuntimeError(f"{path}: missing HELIO_RESULT smoke evidence")
                validate_experiment_evidence(path, source, text)
                validate_results_evidence(path, source, text)
                for line in text.splitlines():
                    if (
                        line.startswith("split signature:")
                        and metadata["artifact"] == "demo"
                    ):
                        signature_group = (
                            f"{metadata['module_id']}:{metadata['artifact']}"
                        )
                        signatures.setdefault(signature_group, set()).add(
                            line.split(":", 1)[1].strip()
                        )
                print(f"passed {path.relative_to(ROOT)}")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    mismatched = {module: values for module, values in signatures.items() if len(values) > 1}
    if mismatched:
        raise RuntimeError(f"framework split signatures differ: {mismatched}")
    print(f"passed {len(notebooks)} published notebook smoke runs")


if __name__ == "__main__":
    main()
