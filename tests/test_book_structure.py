from pathlib import Path

import nbformat
import yaml

from scripts.check_book_structure import publication_errors, validate_repository


def test_book_structure() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    assert validate_repository(repository_root) == []


def test_placeholder_chapter_cannot_be_published() -> None:
    root = Path("/book")
    chapter = root / "general-ml" / "future" / "index.md"
    chapters = {chapter: {"status": "placeholder"}}
    toc = {"general-ml/future/index"}
    assert publication_errors(chapters, toc, root) == [
        "general-ml/future/index.md: placeholder chapter must not appear in _toc.yml"
    ]


def test_ready_chapter_must_be_published() -> None:
    root = Path("/book")
    chapter = root / "general-ml" / "ready" / "index.md"
    chapters = {chapter: {"status": "draft"}}
    assert publication_errors(chapters, set(), root) == [
        "general-ml/ready/index.md: draft chapter is missing from _toc.yml"
    ]


def test_ready_chapter_in_toc_passes_publication_gate() -> None:
    root = Path("/book")
    chapter = root / "general-ml" / "ready" / "index.md"
    chapters = {chapter: {"status": "reviewed"}}
    toc = {"general-ml/ready/index"}
    assert publication_errors(chapters, toc, root) == []


def test_teaching_workflows_offer_exploration_without_controlled_studies() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    workflow_roots = [
        repository_root / "general-ml/foundations/neural-networks/pytorch",
        repository_root / "general-ml/foundations/convolutional-neural-networks/pytorch",
        repository_root / "general-ml/advanced/cifar10-cnn-progression/pytorch",
        repository_root / "general-ml/advanced/tree-models/xgboost",
        repository_root / "general-ml/advanced/transfer-learning/pytorch",
        repository_root / "general-ml/advanced/hyperparameter-tuning/pytorch",
        repository_root / "general-ml/advanced/generative-models/pytorch",
        repository_root / "heliophysics/applications/dst-forecasting/pytorch",
    ]

    for workflow_root in workflow_roots:
        demo = nbformat.read(workflow_root / "demo.ipynb", 4)
        metadata = demo.metadata["helio_data_methods"]
        assert metadata["artifact"] == "demo"
        assert metadata["budget"] == "teaching"
        assert "exercise_id" not in metadata
        tags = {tag for cell in demo.cells for tag in cell.metadata.get("tags", [])}
        assert "try-it-yourself" in tags
        source = "\n".join(cell.source for cell in demo.cells)
        assert "HELIO_COMPARISON" not in source
        assert "## Controlled experiment" not in source
        assert "## What changed?" not in source
        if metadata["module_id"] == "tree-models":
            assert "example-thought" in tags
            assert source.count("HELIO_EXPERIMENT") == 1
            assert "## Example thought" in source
        else:
            assert "guided-experiment" not in tags
            assert "example-thought" not in tags
            assert "HELIO_EXPERIMENT" not in source


def test_retired_teaching_artifacts_are_absent() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    retired = [
        path
        for track in ("general-ml", "heliophysics")
        for path in (repository_root / track).rglob("*.ipynb")
        if path.name in {"exercise.ipynb", "solution.ipynb"}
    ]
    assert retired == []


def test_keras_workflows_offer_suggestions_without_duplicate_studies() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    for path in repository_root.glob("general-ml/**/keras/demo.ipynb"):
        notebook = nbformat.read(path, 4)
        tags = {tag for cell in notebook.cells for tag in cell.metadata.get("tags", [])}
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "try-it-yourself" in tags
        assert "guided-experiment" not in tags
        assert "HELIO_EXPERIMENT" not in source
    dst = nbformat.read(
        repository_root / "heliophysics/applications/dst-forecasting/keras/demo.ipynb",
        4,
    )
    assert "try-it-yourself" in {
        tag for cell in dst.cells for tag in cell.metadata.get("tags", [])
    }


def test_reader_facing_notebook_inventory_is_23() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    toc = (repository_root / "_toc.yml").read_text(encoding="utf-8")
    entries = [
        line.split("file:", 1)[1].strip()
        for line in toc.splitlines()
        if "file:" in line
    ]
    notebooks = [entry for entry in entries if (repository_root / f"{entry}.ipynb").exists()]
    assert len(notebooks) == 23


def test_plasma_sheet_results_are_one_self_contained_page() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    module = (
        repository_root
        / "heliophysics/research-case-studies/plasma-sheet-modeling"
    )
    assert not (module / "index.md").exists()
    assert list(module.glob("*.ipynb")) == [module / "index.ipynb"]
    notebook = nbformat.read(module / "index.ipynb", 4)
    metadata = notebook.metadata["helio_data_methods"]
    assert metadata["implementation"] == "results-demonstration"
    assert metadata["status"] == "draft"
    assert metadata["datasets"] == ["plasma-sheet-prime-tm03"]
    source = "\n".join(cell.source for cell in notebook.cells)
    assert source.count("HELIO_RESULTS") == 1
    assert "DRAFT" not in source
    tags = [tag for cell in notebook.cells for tag in cell.metadata.get("tags", [])]
    assert tags.count("results-figure") == 2
    assert "try-it-yourself" in tags


def test_single_notebook_implementations_are_flattened_in_toc() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    toc = yaml.safe_load((repository_root / "_toc.yml").read_text(encoding="utf-8"))
    entries: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("file"), str):
                entries.append(value["file"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(toc)
    redundant = [
        entry
        for entry in entries
        if entry.endswith(("/pytorch", "/keras"))
        or (entry.endswith("/xgboost") and "sep-occurrence-forecasting" not in entry)
    ]
    assert redundant == []

    landing_pages = [
        path.relative_to(repository_root).as_posix()
        for track in ("general-ml", "heliophysics")
        for path in (repository_root / track).rglob("*.md")
        if path.name in {"pytorch.md", "keras.md", "xgboost.md"}
    ]
    assert landing_pages == [
        "heliophysics/research-case-studies/sep-occurrence-forecasting/xgboost.md"
    ]


def test_software_toolkit_is_published_first() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    toc = (repository_root / "_toc.yml").read_text(encoding="utf-8")
    toolkit = "general-ml/foundations/software-toolkit/index"
    assert toolkit in toc
    for hidden_page in (
        "general-ml/foundations/machine-learning-problems/index",
        "general-ml/foundations/data-splits-and-leakage/index",
        "general-ml/foundations/model-evaluation/index",
    ):
        assert hidden_page not in toc
    content = (repository_root / f"{toolkit}.md").read_text(encoding="utf-8")
    for package in ("pandas", "PyTorch", "scikit-learn", "Keras 3"):
        assert package in content
    assert "## References" not in content


def test_removed_coronal_hybrid_and_plasma_sheet_title() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    assert not (
        repository_root
        / "heliophysics/research-case-studies/coronal-loop-reconstruction/xgboost/demo.ipynb"
    ).exists()
    plasma = nbformat.read(
        repository_root
        / "heliophysics/research-case-studies/plasma-sheet-modeling/index.ipynb",
        4,
    )
    source = "\n".join(cell.source for cell in plasma.cells)
    assert plasma.metadata["title"] == "Plasma-Sheet Modeling"
    assert source.startswith("# Plasma-Sheet Modeling\n")
    assert "Research status" not in source
    assert "Learning objectives" not in source


def test_reader_navigation_has_four_unique_top_level_pages() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    toc = yaml.safe_load((repository_root / "_toc.yml").read_text(encoding="utf-8"))
    assert "parts" not in toc
    chapters = toc["chapters"]
    assert [chapter["file"] for chapter in chapters] == [
        "general-ml/index",
        "statistical-modeling/index",
        "heliophysics/index",
        "resources/index",
    ]
    assert "sections" not in chapters[1]


def test_reader_landing_pages_are_concise_and_complete() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    homepage = (repository_root / "index.md").read_text(encoding="utf-8")
    assert "Available now" not in homepage
    assert "How the book grows" not in homepage
    assert "Software Toolkit" in homepage
    assert "Run the notebooks" in homepage

    resources = (repository_root / "resources/index.md").read_text(encoding="utf-8")
    assert "To be curated" not in resources
    assert "DRAFT" not in resources
    for link in (
        "https://www.statlearning.com/",
        "https://d2l.ai/",
        "https://www.deeplearningbook.org/",
        "https://mlu-explain.github.io/",
    ):
        assert link in resources


def test_github_star_control_is_global_and_accessible() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository_root / "_config.yml").read_text(encoding="utf-8"))
    assert "github-star.js" in config["sphinx"]["config"]["html_js_files"]
    script = (repository_root / "_static/github-star.js").read_text(encoding="utf-8")
    assert "https://github.com/SavvasRaptis/helio-data-methods" in script
    assert "Star on GitHub" in script
    assert "aria-label" in script
