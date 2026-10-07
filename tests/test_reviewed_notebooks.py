"""Regression checks for the reviewed branch's threshold and runtime changes."""

from pathlib import Path

import nbformat
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from scripts.create_expansion_notebooks import SEP_METRICS, TRANSFER_COMPARISON
from scripts.smoke_test_notebooks import reduce_notebook_for_test


@pytest.mark.parametrize("score", ["TSS", "HSS"])
def test_threshold_search_includes_all_decisions(score):
    namespace = {"np": np, "confusion_matrix": confusion_matrix}
    exec(SEP_METRICS, namespace)
    choose = namespace["choose_threshold"]
    metric = namespace["verification_scores"]
    truth = np.array([1, 1, 1, 0, 0, 1])
    probability = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.9], dtype=np.float32)
    candidates = [*np.unique(probability), float(probability.max()) + 1]
    scores = [metric(truth, probability, t)[score] for t in candidates]
    assert choose(truth, probability, score) == candidates[int(np.argmax(scores))]
    assert metric(truth, probability, candidates[0])["misses"] == 0
    assert metric(truth, probability, candidates[-1])["hits"] == 0
    assert choose(np.array([0, 1]), np.array([0.5, 0.5]), score) == 0.5


def test_transfer_temporary_run_reduces_head_budget():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "HEAD_EPOCHS = 30\nEPOCHS = 10  # Reduce to 1 or 2 for a quicker run."
    )])
    reduce_notebook_for_test(notebook, Path("general-ml/advanced/transfer-learning/pytorch/demo.ipynb"))
    assert "HEAD_EPOCHS = 1" in notebook.cells[0].source
    assert "EPOCHS = 1" in notebook.cells[0].source


def test_transfer_subset_labels_are_actual_sizes():
    header = TRANSFER_COMPARISON.split("    head = make_head()", 1)[0]
    namespace = {"np": np, "SUBSET_SIZES": [1000, 5000, 45000], "y_train": np.zeros(2000)}
    exec(header + "    rows.append((size, len(subset)))\n", namespace)
    assert namespace["rows"] == [(1000, 1000), (2000, 2000)]
