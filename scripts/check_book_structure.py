"""Validate the book's module contract without executing notebook cells."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from helio_data_methods.data import DATASETS


TRACKS = {"general", "statistical", "heliophysics"}
LEVELS = {"foundation", "applied", "research"}
STATUSES = {"placeholder", "draft", "reviewed"}
IMPLEMENTATIONS = {
    "pytorch-with-keras-alternative",
    "framework-neutral",
    "mixed-model-case-study",
    "results-demonstration",
    "none",
}
FRAMEWORKS = {"keras", "pytorch", "framework-neutral"}
BACKENDS = {"torch"}
IMPLEMENTATION_ROLES = {"primary", "alternative", "comparison"}
ARTIFACTS = {"demo", "validation", "interpretability"}
RUNTIMES = {"local", "colab"}
BUDGETS = {"teaching"}
REQUIRED_CHAPTER_FIELDS = {
    "title",
    "track",
    "level",
    "status",
    "module_id",
    "implementation",
}
MODULE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
NEURAL_FILES = {
    "pytorch/demo.ipynb": ("pytorch", "torch", "primary", "demo"),
    "keras/demo.ipynb": ("keras", "torch", "alternative", "demo"),
}
FRAMEWORK_NEUTRAL_FILES = {
    "xgboost/demo.ipynb": ("framework-neutral", "primary", "demo", "xgboost"),
}
LEGACY_NOTEBOOK_PATTERN = re.compile(
    r"^(?:demo|exercise|solutions)_(?:keras|pytorch|xgboost)\.ipynb$"
)
TRY_IT_YOURSELF_MODULES = {
    "neural-networks",
    "convolutional-neural-networks",
    "cifar10-cnn-progression",
    "tree-models",
    "transfer-learning",
    "hyperparameter-tuning",
    "generative-models",
    "dst-forecasting",
}
EXAMPLE_THOUGHT_MODULES = {"tree-models"}
BALANCED_CLASSIFICATION_MODULES = {
    "neural-networks",
    "convolutional-neural-networks",
    "cifar10-cnn-progression",
    "tree-models",
    "transfer-learning",
    "hyperparameter-tuning",
}
TOP_LEVEL_PAGES = [
    "general-ml/index",
    "statistical-modeling/index",
    "heliophysics/index",
    "resources/index",
]


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read valid YAML: {exc}") from exc


def _front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening YAML front matter delimiter")
    try:
        end = next(index for index, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("missing closing YAML front matter delimiter") from exc
    data = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(data, dict):
        raise ValueError("front matter must be a YAML mapping")
    return data


def _toc_files(node: Any) -> list[str]:
    files: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "file" and isinstance(value, str):
                files.append(value)
            else:
                files.extend(_toc_files(value))
    elif isinstance(node, list):
        for value in node:
            files.extend(_toc_files(value))
    return files


def _chapter_for_notebook(notebook: Path, root: Path) -> Path | None:
    parent = notebook.parent
    while parent != root:
        chapter = parent / "index.md"
        if chapter.exists():
            return chapter
        parent = parent.parent
    return None


def publication_errors(
    chapters: dict[Path, dict[str, Any]], toc_set: set[str], root: Path
) -> list[str]:
    """Validate the relationship between publication status and navigation."""

    errors: list[str] = []
    for chapter, metadata in chapters.items():
        relative = chapter.relative_to(root)
        entry = relative.with_suffix("").as_posix()
        status = metadata.get("status")
        if status == "placeholder" and entry in toc_set:
            errors.append(f"{relative}: placeholder chapter must not appear in _toc.yml")
        elif status in {"draft", "reviewed"} and entry not in toc_set:
            errors.append(f"{relative}: {status} chapter is missing from _toc.yml")
    return errors


def validate_repository(root: Path) -> list[str]:
    """Return all structural validation errors found below *root*."""

    root = root.resolve()
    errors: list[str] = []
    if (root / "source-material").exists():
        errors.append("legacy source-material/ archive must not be present")
    chapters: dict[Path, dict[str, Any]] = {}
    standalone_results: dict[Path, dict[str, Any]] = {}
    module_ids: dict[str, Path] = {}

    for relative_track, expected_track in (
        ("general-ml", "general"),
        ("statistical-modeling", "statistical"),
        ("heliophysics", "heliophysics"),
    ):
        track_root = root / relative_track
        for chapter in sorted(track_root.rglob("index.md")):
            relative = chapter.relative_to(root)
            try:
                metadata = _front_matter(chapter)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                errors.append(f"{relative}: {exc}")
                continue

            chapters[chapter] = metadata
            missing = sorted(REQUIRED_CHAPTER_FIELDS - metadata.keys())
            if missing:
                errors.append(f"{relative}: missing metadata fields: {', '.join(missing)}")

            if metadata.get("track") not in TRACKS:
                errors.append(f"{relative}: invalid track {metadata.get('track')!r}")
            elif metadata["track"] != expected_track:
                errors.append(
                    f"{relative}: track must be {expected_track!r} for this directory"
                )
            if metadata.get("level") not in LEVELS:
                errors.append(f"{relative}: invalid level {metadata.get('level')!r}")
            expected_level = None
            if "foundations" in relative.parts:
                expected_level = "foundation"
            elif "advanced" in relative.parts or "applications" in relative.parts:
                expected_level = "applied"
            elif "research-case-studies" in relative.parts:
                expected_level = "research"
            if expected_level and metadata.get("level") != expected_level:
                errors.append(
                    f"{relative}: level must be {expected_level!r} for this directory"
                )
            if metadata.get("status") not in STATUSES:
                errors.append(f"{relative}: invalid status {metadata.get('status')!r}")
            if metadata.get("implementation") not in IMPLEMENTATIONS:
                errors.append(
                    f"{relative}: invalid implementation "
                    f"{metadata.get('implementation')!r}"
                )

            module_id = metadata.get("module_id")
            if not isinstance(module_id, str) or not MODULE_ID_PATTERN.fullmatch(module_id):
                errors.append(f"{relative}: invalid module_id {module_id!r}")
            elif module_id in module_ids:
                errors.append(
                    f"{relative}: duplicate module_id {module_id!r}; "
                    f"first used by {module_ids[module_id].relative_to(root)}"
                )
            else:
                module_ids[module_id] = chapter
            if chapter.parent != track_root and module_id != chapter.parent.name:
                errors.append(
                    f"{relative}: module_id must match directory name "
                    f"{chapter.parent.name!r}"
                )
            if "source_material" in metadata:
                errors.append(f"{relative}: retired source_material metadata is forbidden")

    config_path = root / "_config.yml"
    try:
        config = _load_yaml(config_path)
        if config.get("title") != "Statistical Modeling and Machine Learning in Heliophysics":
            errors.append("_config.yml: unexpected project title")
        if config.get("author") != "Savvas Raptis":
            errors.append("_config.yml: unexpected author")
        if config.get("execute", {}).get("execute_notebooks") != "off":
            errors.append("_config.yml: notebook execution must be off")
        footer = config.get("html", {}).get("extra_footer", "")
        for expected_link in (
            "mailto:Savvas.raptis@jhuapl.edu",
            "mailto:savvasraptis@pm.me",
            "https://savvasraptis.github.io",
        ):
            if expected_link not in footer:
                errors.append(f"_config.yml: footer is missing {expected_link}")
        js_files = config.get("sphinx", {}).get("config", {}).get("html_js_files", [])
        if "github-star.js" not in js_files:
            errors.append("_config.yml: global GitHub Star control is not loaded")
    except (AttributeError, ValueError) as exc:
        errors.append(f"_config.yml: {exc}")

    toc_path = root / "_toc.yml"
    toc_files: list[str] = []
    try:
        toc = _load_yaml(toc_path)
        if "parts" in toc:
            errors.append("_toc.yml: duplicate track captions are forbidden")
        top_level = [
            chapter.get("file")
            for chapter in toc.get("chapters", [])
            if isinstance(chapter, dict)
        ]
        if top_level != TOP_LEVEL_PAGES:
            errors.append(
                "_toc.yml: top-level pages must be ordered "
                f"{TOP_LEVEL_PAGES!r}"
            )
        statistical_entry = next(
            (
                chapter
                for chapter in toc.get("chapters", [])
                if isinstance(chapter, dict)
                and chapter.get("file") == "statistical-modeling/index"
            ),
            None,
        )
        if statistical_entry is not None and statistical_entry.get("sections"):
            errors.append(
                "_toc.yml: Statistical Modeling must not expose unfinished child pages"
            )
        toc_files = [toc.get("root", "")] + _toc_files(toc)
        if len(toc_files) != len(set(toc_files)):
            errors.append("_toc.yml: duplicate page reference")
        for entry in toc_files:
            if "source-material" in Path(entry).parts:
                errors.append(f"_toc.yml: source archive must not appear in TOC: {entry}")
            if not (root / f"{entry}.md").exists() and not (root / f"{entry}.ipynb").exists():
                errors.append(f"_toc.yml: referenced page does not exist: {entry}")
    except (AttributeError, ValueError) as exc:
        errors.append(f"_toc.yml: {exc}")

    toc_set = set(toc_files)
    errors.extend(publication_errors(chapters, toc_set, root))

    homepage = (root / "index.md").read_text(encoding="utf-8")
    for retired_heading in ("Available now", "How the book grows"):
        if retired_heading in homepage:
            errors.append(f"index.md: retired heading remains: {retired_heading!r}")
    for required_text in ("Software Toolkit", "Run the notebooks"):
        if required_text not in homepage:
            errors.append(f"index.md: missing reader introduction {required_text!r}")

    resources = (root / "resources/index.md").read_text(encoding="utf-8")
    if "DRAFT" in resources or "To be curated" in resources:
        errors.append("resources/index.md: published resources must not contain placeholders")

    star_script = root / "_static/github-star.js"
    if not star_script.exists():
        errors.append("_static/github-star.js: missing GitHub Star control")
    else:
        star_source = star_script.read_text(encoding="utf-8")
        for expected_text in (
            "https://github.com/SavvasRaptis/helio-data-methods",
            "Star on GitHub",
            "aria-label",
        ):
            if expected_text not in star_source:
                errors.append(
                    f"_static/github-star.js: missing {expected_text!r}"
                )

    for notebook in sorted(
        path
        for track in ("general-ml", "heliophysics")
        for path in (root / track).rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    ):
        relative = notebook.relative_to(root)
        try:
            payload = json.loads(notebook.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative}: cannot read valid notebook JSON: {exc}")
            continue
        metadata = payload.get("metadata", {}).get("helio_data_methods")
        if not isinstance(metadata, dict):
            errors.append(f"{relative}: missing metadata.helio_data_methods")
            continue
        is_standalone_result = (
            notebook.name == "index.ipynb"
            and metadata.get("implementation") == "results-demonstration"
        )
        chapter: Path | None = None
        if is_standalone_result:
            standalone_results[notebook] = metadata
            chapter_metadata = metadata
            missing = sorted(REQUIRED_CHAPTER_FIELDS - metadata.keys())
            if missing:
                errors.append(
                    f"{relative}: missing standalone page metadata fields: "
                    f"{', '.join(missing)}"
                )
            expected_track = "general" if relative.parts[0] == "general-ml" else "heliophysics"
            if metadata.get("track") != expected_track:
                errors.append(
                    f"{relative}: standalone page track must be {expected_track!r}"
                )
            if metadata.get("level") not in LEVELS:
                errors.append(f"{relative}: invalid level {metadata.get('level')!r}")
            if "research-case-studies" in relative.parts and metadata.get("level") != "research":
                errors.append(f"{relative}: research case study level must be 'research'")
            if metadata.get("status") not in STATUSES:
                errors.append(f"{relative}: invalid status {metadata.get('status')!r}")
            module_id = metadata.get("module_id")
            if not isinstance(module_id, str) or not MODULE_ID_PATTERN.fullmatch(module_id):
                errors.append(f"{relative}: invalid module_id {module_id!r}")
            elif module_id != notebook.parent.name:
                errors.append(
                    f"{relative}: module_id must match directory name "
                    f"{notebook.parent.name!r}"
                )
            elif module_id in module_ids:
                errors.append(
                    f"{relative}: duplicate module_id {module_id!r}; "
                    f"first used by {module_ids[module_id].relative_to(root)}"
                )
            else:
                module_ids[module_id] = notebook
        else:
            chapter = _chapter_for_notebook(notebook, root)
            if chapter is None or chapter not in chapters:
                errors.append(f"{relative}: notebook has no owning module index.md")
                continue
            chapter_metadata = chapters[chapter]
            if metadata.get("module_id") != chapter_metadata.get("module_id"):
                errors.append(f"{relative}: module_id does not match its chapter")
        if metadata.get("framework") not in FRAMEWORKS:
            errors.append(f"{relative}: invalid framework {metadata.get('framework')!r}")
        framework = metadata.get("framework")
        if framework in {"keras", "pytorch"}:
            if metadata.get("backend") not in BACKENDS:
                errors.append(f"{relative}: neural notebook backend must be 'torch'")
        elif "backend" in metadata:
            errors.append(f"{relative}: framework-neutral notebook must omit backend")
        if metadata.get("implementation_role") not in IMPLEMENTATION_ROLES:
            errors.append(
                f"{relative}: invalid implementation_role "
                f"{metadata.get('implementation_role')!r}"
            )
        if metadata.get("artifact") not in ARTIFACTS:
            errors.append(f"{relative}: invalid artifact {metadata.get('artifact')!r}")
        artifact = metadata.get("artifact")
        if metadata.get("budget") not in BUDGETS:
            errors.append(f"{relative}: invalid budget {metadata.get('budget')!r}")
        if "exercise_id" in metadata:
            errors.append(f"{relative}: retired exercise_id metadata is forbidden")
        runtime = metadata.get("runtime")
        if (
            not isinstance(runtime, list)
            or set(runtime) != RUNTIMES
            or len(runtime) != len(RUNTIMES)
        ):
            errors.append(
                f"{relative}: runtime must contain exactly {sorted(RUNTIMES)!r}"
            )
        datasets = metadata.get("datasets")
        if not isinstance(datasets, list):
            errors.append(f"{relative}: datasets must be a list")
        else:
            invalid_datasets = sorted(
                value
                for value in datasets
                if not isinstance(value, str) or value not in DATASETS
            )
            if invalid_datasets:
                errors.append(
                    f"{relative}: invalid dataset identifiers: {invalid_datasets!r}"
                )
            if len(datasets) != len(set(datasets)):
                errors.append(f"{relative}: duplicate dataset identifier")
        if "source_material" in metadata:
            errors.append(f"{relative}: retired source_material metadata is forbidden")
        all_notebook_source = "\n".join(
            "".join(cell.get("source", "")) for cell in payload.get("cells", [])
        )
        if "source-material" in all_notebook_source:
            errors.append(f"{relative}: legacy source-material reference is forbidden")
        notebook_source = "\n".join(
            "".join(cell.get("source", ""))
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        if "%matplotlib inline" not in notebook_source:
            errors.append(f"{relative}: notebook must select the inline Matplotlib backend")
        stored_output_text = json.dumps(
            [cell.get("outputs", []) for cell in payload.get("cells", [])]
        )
        if "FigureCanvasAgg is non-interactive" in stored_output_text:
            errors.append(f"{relative}: stored output contains a headless Matplotlib warning")
        all_tags = [
            tag
            for cell in payload.get("cells", [])
            for tag in cell.get("metadata", {}).get("tags", [])
        ]
        forbidden_tags = {"student-task", "solution-comparison"}.intersection(all_tags)
        if forbidden_tags:
            errors.append(f"{relative}: retired cell tags are forbidden: {sorted(forbidden_tags)!r}")
        all_cell_source = "\n".join(
            "".join(cell.get("source", "")) for cell in payload.get("cells", [])
        )
        if "HELIO_COMPARISON" in all_cell_source:
            errors.append(f"{relative}: retired HELIO_COMPARISON evidence is forbidden")
        if "HELIO_FAST_RUN" in all_cell_source or re.search(
            r"\bFAST_RUN\b", all_cell_source
        ):
            errors.append(
                f"{relative}: reader-facing fast-run environment switches are forbidden"
            )
        if re.search(r"\bTODO\b", all_cell_source):
            errors.append(f"{relative}: TODO exercise language is forbidden")
        if re.search(
            r"(?im)^#{1,6}\s+(?:exercise|worked solution)\b", all_cell_source
        ):
            errors.append(f"{relative}: retired exercise/solution headings are forbidden")
        module_id = metadata.get("module_id")
        is_teaching_primary = (
            module_id in TRY_IT_YOURSELF_MODULES
            and artifact == "demo"
            and metadata.get("implementation_role") == "primary"
        )
        if is_teaching_primary:
            if "try-it-yourself" not in all_tags:
                errors.append(f"{relative}: teaching workflow requires a try-it-yourself cell")
        if module_id in EXAMPLE_THOUGHT_MODULES and is_teaching_primary:
            if "example-thought" not in all_tags:
                errors.append(f"{relative}: XGBoost workflow requires example-thought cells")
            if all_cell_source.count("HELIO_EXPERIMENT") != 1:
                errors.append(f"{relative}: XGBoost workflow must emit exactly one HELIO_EXPERIMENT record")
            experiment_source = "\n".join(
                "".join(cell.get("source", ""))
                for cell in payload.get("cells", [])
                if cell.get("cell_type") == "code"
                and "example-thought" in cell.get("metadata", {}).get("tags", [])
            )
            forbidden_test_identifiers = {
                "x_test", "y_test", "test_loader", "test_accuracy",
                "test_predictions", "model_metrics",
            }
            used = sorted(
                identifier for identifier in forbidden_test_identifiers
                if re.search(rf"\b{re.escape(identifier)}\b", experiment_source)
            )
            if used:
                errors.append(
                    f"{relative}: example thought must not access final-test identifiers: {used!r}"
                )
        elif is_teaching_primary:
            if "guided-experiment" in all_tags or "example-thought" in all_tags:
                errors.append(f"{relative}: retired controlled-study tags are forbidden")
            if "HELIO_EXPERIMENT" in all_cell_source:
                errors.append(f"{relative}: only XGBoost may emit HELIO_EXPERIMENT")
        if framework == "keras" and module_id in TRY_IT_YOURSELF_MODULES:
            if "try-it-yourself" not in all_tags:
                errors.append(f"{relative}: Keras workflow requires concise exploration suggestions")
            if (
                "guided-experiment" in all_tags
                or "example-thought" in all_tags
                or "HELIO_EXPERIMENT" in all_cell_source
            ):
                errors.append(f"{relative}: Keras alternative must not contain a controlled study")
        if re.search(r"(?im)^#{1,6}\s+(?:controlled experiment|what changed\??)\s*$", all_cell_source):
            errors.append(f"{relative}: retired controlled-study heading remains")
        if module_id in BALANCED_CLASSIFICATION_MODULES and re.search(
            r"(?i)(?:majority[- ]class|baseline_accuracy)",
            all_cell_source + stored_output_text,
        ):
            errors.append(f"{relative}: majority-class baseline is forbidden for this balanced example")
        if module_id == "dst-forecasting" and re.search(
            r"(?i)training[- ]mean", all_cell_source + stored_output_text
        ):
            errors.append(f"{relative}: Dst must use persistence as its only baseline")
        if module_id == "dst-forecasting" and artifact == "demo":
            if not re.search(r"(?m)^HISTORY_HOURS\s*=\s*3\s*$", notebook_source):
                errors.append(f"{relative}: Dst must use three hours of input history")
            if not re.search(r"(?m)^HORIZON_HOURS\s*=\s*1\s*$", notebook_source):
                errors.append(f"{relative}: Dst must use a one-hour forecast horizon")
            if not (
                "change `HORIZON_HOURS` from 1 to 3 and then 6" in all_cell_source
                and "2015 as the final test year" in all_cell_source
            ):
                errors.append(
                    f"{relative}: Dst exploration must mention three- and six-hour horizons"
                )
            if re.search(r"(?i)three-hour-ahead forecast", all_cell_source):
                errors.append(f"{relative}: stale three-hour default wording remains")
        if re.search(
            r"(?:/Users/|[A-Za-z]:\\\\)", notebook_source + stored_output_text
        ):
            errors.append(f"{relative}: notebook contains an absolute local path")
        if re.search(
            r"IProgress not found|shuffle=True.*ignored.*torch DataLoader",
            stored_output_text,
            flags=re.IGNORECASE,
        ):
            errors.append(f"{relative}: stored output contains a removable runtime warning")
        if framework == "keras":
            lowered_source = notebook_source.lower()
            legacy_framework = "tensor" + "flow"
            legacy_alias = r"\b" + "t" + "f" + r"\."
            if legacy_framework in lowered_source or re.search(legacy_alias, notebook_source):
                errors.append(f"{relative}: Keras alternative contains legacy backend code")
            backend_assignment = notebook_source.find('KERAS_BACKEND"] = "torch"')
            keras_import = notebook_source.find("import keras")
            if backend_assignment < 0 or keras_import < 0 or backend_assignment > keras_import:
                errors.append(
                    f"{relative}: KERAS_BACKEND=torch must be set before importing Keras"
                )
            if 'keras.backend.backend() == "torch"' not in notebook_source:
                errors.append(f"{relative}: Keras alternative must assert the Torch backend")

        notebook_entry = relative.with_suffix("").as_posix()
        chapter_status = chapter_metadata.get("status")
        if chapter_status == "placeholder" and notebook_entry in toc_set:
            errors.append(
                f"{relative}: notebook owned by placeholder chapter must not be in _toc.yml"
            )
        elif chapter_status in {"draft", "reviewed"} and notebook_entry not in toc_set:
            errors.append(
                f"{relative}: notebook owned by {chapter_status} chapter is missing "
                "from _toc.yml"
            )

        if chapter is not None:
            module_relative = notebook.relative_to(chapter.parent).as_posix()
            expected = NEURAL_FILES.get(module_relative)
            if expected and (
                metadata.get("framework"),
                metadata.get("backend"),
                metadata.get("implementation_role"),
                metadata.get("artifact"),
            ) != expected:
                errors.append(f"{relative}: metadata does not match filename")
            neutral_expected = (
                FRAMEWORK_NEUTRAL_FILES.get(module_relative)
                if chapter_metadata.get("implementation") == "framework-neutral"
                else None
            )
            if neutral_expected and (
                metadata.get("framework"),
                metadata.get("implementation_role"),
                metadata.get("artifact"),
                metadata.get("library"),
            ) != neutral_expected:
                errors.append(
                    f"{relative}: framework-neutral metadata does not match filename"
                )

    for notebook, metadata in standalone_results.items():
        relative = notebook.relative_to(root)
        if (notebook.parent / "index.md").exists():
            errors.append(f"{relative}: standalone result page must not have index.md")
        sibling_notebooks = list(notebook.parent.glob("*.ipynb"))
        if sibling_notebooks != [notebook]:
            errors.append(
                f"{relative}: results demonstration must be the directory's only notebook"
            )
        if (
            metadata.get("framework"),
            metadata.get("implementation_role"),
            metadata.get("artifact"),
        ) != ("framework-neutral", "primary", "demo"):
            errors.append(
                f"{relative}: results demonstration metadata must identify one "
                "framework-neutral primary demo"
            )
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        all_source = "\n".join(
            "".join(cell.get("source", "")) for cell in payload.get("cells", [])
        )
        code_source = "\n".join(
            "".join(cell.get("source", ""))
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        tags = [
            tag
            for cell in payload.get("cells", [])
            for tag in cell.get("metadata", {}).get("tags", [])
        ]
        if tags.count("results-figure") != 2:
            errors.append(f"{relative}: results demonstration requires two figure cells")
        if "try-it-yourself" not in tags:
            errors.append(f"{relative}: results demonstration requires exploration prompts")
        if all_source.count("HELIO_RESULTS") != 1:
            errors.append(f"{relative}: results demonstration requires one HELIO_RESULTS record")
        if re.search(r"(?m)^\s*(?:from|import)\s+(?:torch|keras|xgboost|lightgbm)\b", code_source):
            errors.append(f"{relative}: results demonstration must not import model libraries")
        if re.search(r"\.(?:fit|backward)\s*\(", code_source):
            errors.append(f"{relative}: results demonstration must not train models")

    for chapter, metadata in chapters.items():
        if (
            metadata.get("implementation") == "pytorch-with-keras-alternative"
            and metadata.get("status") in {"draft", "reviewed"}
        ):
            for landing in ("pytorch.md", "keras.md"):
                landing_path = chapter.parent / landing
                if landing_path.exists():
                    errors.append(
                        f"{landing_path.relative_to(root)}: redundant single-notebook "
                        "landing page is forbidden"
                    )
            for filename, expected in NEURAL_FILES.items():
                notebook = chapter.parent / filename
                if not notebook.exists():
                    errors.append(
                        f"{chapter.relative_to(root)}: {metadata['status']} PyTorch-first "
                        f"module is missing {filename}"
                    )
                    continue
                notebook_entry = notebook.relative_to(root).with_suffix("").as_posix()
                if notebook_entry not in toc_set:
                    errors.append(
                        f"{notebook.relative_to(root)}: paired artifact is missing "
                        "from _toc.yml"
                    )
                try:
                    payload = json.loads(notebook.read_text(encoding="utf-8"))
                    notebook_metadata = payload["metadata"]["helio_data_methods"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue
                actual = (
                    notebook_metadata.get("framework"),
                    notebook_metadata.get("backend"),
                    notebook_metadata.get("implementation_role"),
                    notebook_metadata.get("artifact"),
                )
                if actual != expected:
                    errors.append(
                        f"{notebook.relative_to(root)}: expected framework/artifact "
                        f"{expected}, found {actual}"
                    )
            for candidate in chapter.parent.glob("*.ipynb"):
                if LEGACY_NOTEBOOK_PATTERN.fullmatch(candidate.name):
                    errors.append(
                        f"{candidate.relative_to(root)}: legacy root-level notebook is forbidden"
                    )
            for retired in chapter.parent.rglob("*.ipynb"):
                if retired.name in {"exercise.ipynb", "solution.ipynb"}:
                    errors.append(
                        f"{retired.relative_to(root)}: retired teaching artifact is forbidden"
                    )
        if (
            metadata.get("implementation") == "framework-neutral"
            and metadata.get("library") == "xgboost"
            and metadata.get("status") in {"draft", "reviewed"}
        ):
            landing = chapter.parent / "xgboost.md"
            if landing.exists():
                errors.append(
                    f"{landing.relative_to(root)}: redundant single-notebook "
                    "landing page is forbidden"
                )
            for filename, expected in FRAMEWORK_NEUTRAL_FILES.items():
                notebook = chapter.parent / filename
                if not notebook.exists():
                    errors.append(
                        f"{chapter.relative_to(root)}: {metadata['status']} "
                        f"framework-neutral module is missing {filename}"
                    )
                    continue
                notebook_entry = notebook.relative_to(root).with_suffix("").as_posix()
                if notebook_entry not in toc_set:
                    errors.append(
                        f"{notebook.relative_to(root)}: framework-neutral artifact "
                        "is missing from _toc.yml"
                    )
                try:
                    payload = json.loads(notebook.read_text(encoding="utf-8"))
                    notebook_metadata = payload["metadata"]["helio_data_methods"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue
                actual = (
                    notebook_metadata.get("framework"),
                    notebook_metadata.get("implementation_role"),
                    notebook_metadata.get("artifact"),
                    notebook_metadata.get("library"),
                )
                if actual != expected:
                    errors.append(
                        f"{notebook.relative_to(root)}: expected metadata {expected}, "
                        f"found {actual}"
                    )
            for retired in chapter.parent.rglob("*.ipynb"):
                if retired.name in {"exercise.ipynb", "solution.ipynb"}:
                    errors.append(
                        f"{retired.relative_to(root)}: retired teaching artifact is forbidden"
                    )
        if (
            metadata.get("implementation") == "mixed-model-case-study"
            and metadata.get("status") in {"draft", "reviewed"}
        ):
            artifacts = metadata.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                errors.append(
                    f"{chapter.relative_to(root)}: mixed case study requires a "
                    "non-empty artifacts list"
                )
                continue
            if len(artifacts) != len(set(artifacts)):
                errors.append(
                    f"{chapter.relative_to(root)}: duplicate mixed artifact"
                )
            for stem in artifacts:
                if (
                    not isinstance(stem, str)
                    or Path(stem).is_absolute()
                    or ".." in Path(stem).parts
                    or not all(MODULE_ID_PATTERN.fullmatch(part) for part in Path(stem).parts)
                ):
                    errors.append(
                        f"{chapter.relative_to(root)}: invalid artifact stem {stem!r}"
                    )
                    continue
                notebook = chapter.parent / f"{stem}.ipynb"
                if not notebook.exists():
                    errors.append(
                        f"{chapter.relative_to(root)}: mixed case study is missing "
                        f"{stem}.ipynb"
                    )
                    continue
                entry = notebook.relative_to(root).with_suffix("").as_posix()
                if entry not in toc_set:
                    errors.append(
                        f"{notebook.relative_to(root)}: mixed artifact is missing "
                        "from _toc.yml"
                    )
                try:
                    payload = json.loads(notebook.read_text(encoding="utf-8"))
                    notebook_metadata = payload["metadata"]["helio_data_methods"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError):
                    continue
                group, artifact = Path(stem).parts
                expected_framework = (
                    group if group in {"pytorch", "keras"} else "framework-neutral"
                )
                expected_role = {
                    "pytorch": "primary",
                    "keras": "alternative",
                    "xgboost": "comparison",
                }.get(group)
                actual = (
                    notebook_metadata.get("framework"),
                    notebook_metadata.get("implementation_role"),
                    notebook_metadata.get("artifact"),
                )
                expected = (expected_framework, expected_role, artifact)
                if actual != expected:
                    errors.append(
                        f"{notebook.relative_to(root)}: expected mixed metadata "
                        f"{expected}, found {actual}"
                    )
            groups = {
                Path(stem).parts[0]: sum(
                    isinstance(candidate, str)
                    and Path(candidate).parts[0] == Path(stem).parts[0]
                    for candidate in artifacts
                )
                for stem in artifacts
                if isinstance(stem, str)
            }
            for group, artifact_count in groups.items():
                landing = chapter.parent / f"{group}.md"
                landing_entry = landing.relative_to(root).with_suffix("").as_posix()
                if artifact_count > 1:
                    if not landing.exists():
                        errors.append(
                            f"{chapter.relative_to(root)}: multi-notebook group is "
                            f"missing implementation landing page {group}.md"
                        )
                    elif landing_entry not in toc_set:
                        errors.append(
                            f"{landing.relative_to(root)}: multi-notebook landing "
                            "page is missing from _toc.yml"
                        )
                elif landing.exists():
                    errors.append(
                        f"{landing.relative_to(root)}: redundant single-notebook "
                        "landing page is forbidden"
                    )

    for dataset_id, files in DATASETS.items():
        if not files:
            errors.append(f"dataset {dataset_id!r}: no files declared")
        for filename, (source_path, checksum) in files.items():
            if not filename or Path(filename).is_absolute():
                errors.append(f"dataset {dataset_id!r}: invalid filename {filename!r}")
            if (
                Path(source_path).is_absolute()
                or not Path(source_path).parts
                or Path(source_path).parts[0] != "data"
                or not (root / source_path).is_file()
            ):
                errors.append(
                    f"dataset {dataset_id!r}: invalid source path {source_path!r}"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                errors.append(
                    f"dataset {dataset_id!r}: invalid SHA-256 for {filename!r}"
                )

    published_notebooks = [
        entry for entry in toc_files if (root / f"{entry}.ipynb").exists()
    ]
    if len(published_notebooks) != 23:
        errors.append(
            f"_toc.yml: expected 23 published notebooks, found {len(published_notebooks)}"
        )

    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = validate_repository(root)
    if errors:
        print("Book structure validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    chapter_count = len(list((root / "general-ml").rglob("index.md"))) + len(
        list((root / "heliophysics").rglob("index.md"))
    )
    standalone_count = sum(
        json.loads(path.read_text(encoding="utf-8"))
        .get("metadata", {})
        .get("helio_data_methods", {})
        .get("implementation")
        == "results-demonstration"
        for path in (root / "heliophysics").rglob("index.ipynb")
    )
    print(
        "Book structure is valid "
        f"({chapter_count} chapter pages and {standalone_count} standalone results page checked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
