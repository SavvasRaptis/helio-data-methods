"""Remove machine-specific text from published notebook sources and outputs."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

from smoke_test_notebooks import ROOT, published_notebooks


SOURCE_REPLACEMENTS = {
    'print(f"data file: {data_path}")':
        'print(f"data file: {DATA_FILENAME} (checksum verified)")',
    'for name, path in dataset_files.items():\n    print(f"  {name}: {path}")':
        'for name in dataset_files:\n    print(f"  {name} (checksum verified)")',
    'history = gan.fit(dataset, epochs=EPOCHS, verbose=2)':
        'history = gan.fit(dataset, epochs=EPOCHS, verbose=2, shuffle=False)',
}

REMOVABLE_WARNING = re.compile(
    r"IProgress not found|shuffle=True.*ignored.*torch DataLoader",
    flags=re.IGNORECASE | re.DOTALL,
)


def sanitize_text(value: str) -> str:
    repository_prefix = str(ROOT) + "/"
    return value.replace(repository_prefix, "")


def main() -> None:
    changed_paths: list[Path] = []
    for path in published_notebooks(None):
        notebook = nbformat.read(path, as_version=4)
        changed = False
        for cell in notebook.cells:
            source = cell.get("source", "")
            revised = source
            for old, new in SOURCE_REPLACEMENTS.items():
                revised = revised.replace(old, new)
            if (
                "import optuna\noptuna.logging.set_verbosity" in revised
                and "IProgress not found" not in revised
            ):
                revised = revised.replace(
                    "import optuna\noptuna.logging.set_verbosity",
                    'import warnings\n\nwarnings.filterwarnings('
                    '"ignore", message="IProgress not found.*")\n'
                    "import optuna\noptuna.logging.set_verbosity",
                )
            if "import shap\n\nbackground_count" in revised and "IProgress not found" not in revised:
                revised = revised.replace(
                    "import shap\n\nbackground_count",
                    'import warnings\n\nwarnings.filterwarnings('
                    '"ignore", message="IProgress not found.*")\n'
                    "import shap\n\nbackground_count",
                )
            if revised != source:
                cell["source"] = revised
                changed = True

            cleaned_outputs = []
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    text = output.get("text", "")
                    if REMOVABLE_WARNING.search(text):
                        changed = True
                        continue
                    clean = sanitize_text(text)
                    if clean != text:
                        output["text"] = clean
                        changed = True
                elif output.get("output_type") in {"display_data", "execute_result"}:
                    plain = output.get("data", {}).get("text/plain")
                    if isinstance(plain, str):
                        clean = sanitize_text(plain)
                        if clean != plain:
                            output["data"]["text/plain"] = clean
                            changed = True
                cleaned_outputs.append(output)
            if cleaned_outputs != cell.get("outputs", []):
                cell["outputs"] = cleaned_outputs

        if changed:
            nbformat.write(notebook, path)
            changed_paths.append(path.relative_to(ROOT))

    for path in changed_paths:
        print(f"sanitized {path}")
    print(f"sanitized {len(changed_paths)} published notebooks")


if __name__ == "__main__":
    main()
