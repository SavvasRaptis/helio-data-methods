"""Execute and store full-budget outputs for reader-visible notebooks."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from smoke_test_notebooks import (
    output_text,
    published_notebooks,
    validate_experiment_evidence,
    validate_results_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
STORED_ARTIFACTS = {"demo", "validation", "interpretability"}


def verify_figure_outputs(
    path: Path, source: nbformat.NotebookNode, executed: nbformat.NotebookNode
) -> None:
    """Reject headless runs that silently replace figures with warnings."""

    source_text = "\n".join(cell.get("source", "") for cell in source.cells)
    output_text: list[str] = []
    image_count = 0
    for cell in executed.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                output_text.append(output.get("text", ""))
            elif output.output_type in {"display_data", "execute_result"}:
                data = output.get("data", {})
                image_count += int("image/png" in data or "image/svg+xml" in data)
                plain = data.get("text/plain", "")
                output_text.extend(plain if isinstance(plain, list) else [plain])

    combined_output = "\n".join(output_text)
    if "FigureCanvasAgg is non-interactive" in combined_output:
        raise RuntimeError(f"{path}: Matplotlib used a non-interactive backend")
    if "plt.show()" in source_text and image_count == 0:
        raise RuntimeError(f"{path}: plotting code produced no stored figure output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="Published module ID to execute.")
    args = parser.parse_args()
    cache_root = ROOT / ".helio-cache"
    paths = published_notebooks(args.module)

    with tempfile.TemporaryDirectory(prefix="helio-full-run-") as temporary:
        environment = {
            "HELIO_DATA_CACHE": str(cache_root / "heliophysics"),
            "HELIO_DATA_DIR": str(cache_root / "torchvision"),
            "KERAS_HOME": str(cache_root / "keras"),
            "MPLCONFIGDIR": str(Path(temporary) / "matplotlib"),
            "MPLBACKEND": "module://matplotlib_inline.backend_inline",
            "HELIO_TUNER_DIR": str(Path(temporary) / "tuning"),
        }
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        os.environ.pop("HELIO_FAST_RUN", None)
        try:
            for path in paths:
                notebook = nbformat.read(path, as_version=4)
                metadata = notebook.metadata["helio_data_methods"]
                if metadata["artifact"] not in STORED_ARTIFACTS:
                    raise RuntimeError(f"{path}: unsupported stored artifact")
                executed = NotebookClient(
                    notebook,
                    timeout=7200,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(path.parent)}},
                ).execute()
                verify_figure_outputs(path, notebook, executed)
                validate_experiment_evidence(
                    path,
                    "\n".join(cell.get("source", "") for cell in notebook.cells),
                    output_text(executed),
                )
                validate_results_evidence(
                    path,
                    "\n".join(cell.get("source", "") for cell in notebook.cells),
                    output_text(executed),
                )
                for cell in executed.cells:
                    if (
                        cell.get("cell_type") == "code"
                        and cell.get("source", "").strip() == "%matplotlib inline"
                    ):
                        cell["outputs"] = []
                nbformat.write(executed, path)
                print(f"stored verified outputs: {path.relative_to(ROOT)}")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    main()
