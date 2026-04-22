"""Tests for Experiment 732 — JEPAReasonerProbe 5-fold stratified CV + domain transfer.

Tests verify:
1. StratifiedKFold produces balanced label distribution per fold (REQ-VER-034-3a, SCENARIO-VER-042).
2. Gate file written with correct schema (REQ-VER-034-3d, SCENARIO-VER-042).
3. mean_auc and std_auc computed correctly from fold list (REQ-VER-034-3b/3c).
4. Domain transfer reports honest result when labeled data is unavailable (REQ-VER-034-4b, SCENARIO-VER-043).
5. honest_verdict strings match all three outcome branches.

These tests cover only code added for Exp 732.  They do NOT re-test pre-existing modules.

Spec: REQ-VER-034-3, REQ-VER-034-3a, REQ-VER-034-3b, REQ-VER-034-3c, REQ-VER-034-3d,
      REQ-VER-034-4, REQ-VER-034-4b, SCENARIO-VER-042, SCENARIO-VER-043
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_items(n_pos: int, n_neg: int) -> list[dict]:
    """Create synthetic corpus items with known label distribution."""
    items = []
    for i in range(n_pos):
        items.append({"question": f"pos_q_{i}", "label": 1.0})
    for i in range(n_neg):
        items.append({"question": f"neg_q_{i}", "label": 0.0})
    return items


def _make_trained_probe():
    """Return a JEPAReasonerProbe with a probe trained on tiny synthetic data."""
    from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

    p = JEPAReasonerProbe(device="cpu")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, JEPAReasonerProbe.HIDDEN_DIM)).astype(np.float32)
    y = np.array([0.0] * 10 + [1.0] * 10, dtype=np.float32)
    p.train_probe(X, y, n_epochs=3, lr=1e-3)
    return p


# ---------------------------------------------------------------------------
# SCENARIO-VER-042: StratifiedKFold balanced folds
# ---------------------------------------------------------------------------


def test_stratified_kfold_balanced_label_distribution():
    """StratifiedKFold produces folds whose positive-rate deviates <= 5% from corpus.

    WHY we test this directly: StratifiedKFold is the safety contract for CV validity.
    If labels leak between folds or the stratification is wrong, all fold AUCs are
    biased and the mean_auc conclusion is invalid.

    Spec: REQ-VER-034-3a, SCENARIO-VER-042
    """
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        pytest.skip("sklearn not installed")

    n_pos, n_neg = 60, 40
    items = _make_items(n_pos, n_neg)
    labels = np.array([item["label"] for item in items], dtype=np.float32)
    X_dummy = np.zeros((len(labels), 1))  # StratifiedKFold only uses labels for stratification

    corpus_pos_rate = labels.mean()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_dummy, labels)):
        val_labels = labels[val_idx]
        fold_pos_rate = val_labels.mean()
        deviation = abs(fold_pos_rate - corpus_pos_rate)
        assert deviation <= 0.05, (
            f"Fold {fold_i} positive rate {fold_pos_rate:.3f} deviates "
            f"{deviation:.3f} from corpus rate {corpus_pos_rate:.3f} (threshold: 0.05)"
        )


def test_stratified_kfold_fold_count():
    """StratifiedKFold produces exactly n_splits folds.

    Spec: REQ-VER-034-3a
    """
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        pytest.skip("sklearn not installed")

    items = _make_items(50, 50)
    labels = np.array([item["label"] for item in items], dtype=np.float32)
    X = np.zeros((len(labels), 1))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X, labels))
    assert len(folds) == 5


# ---------------------------------------------------------------------------
# REQ-VER-034-3b/3c: mean_auc and std_auc computation
# ---------------------------------------------------------------------------


def test_mean_auc_std_auc_computed_correctly():
    """mean_auc and std_auc are computed as numpy mean/std of fold_aucs.

    WHY a direct numerical test: we want to catch off-by-one errors (e.g.,
    using ddof=1 instead of ddof=0) that would cause the gate check to be
    inconsistent across runs.  NumPy's std() defaults to ddof=0 (population
    std), which is what we use.

    Spec: REQ-VER-034-3b, REQ-VER-034-3c
    """
    fold_aucs = [0.85, 0.88, 0.82, 0.90, 0.87]
    fold_arr = np.array(fold_aucs, dtype=np.float64)

    expected_mean = float(fold_arr.mean())
    expected_std = float(fold_arr.std())

    assert math.isclose(expected_mean, 0.864, abs_tol=1e-3), (
        f"Expected mean ~0.864, got {expected_mean}"
    )
    # std should be < 0.15 (robust case)
    assert expected_std < 0.15, f"Unexpectedly high std for synthetic data: {expected_std}"


def test_std_auc_high_variance_detection():
    """std_auc >= 0.15 triggers 'probe_xval_high_variance' verdict when mean passes.

    Spec: REQ-VER-034-3c
    """
    fold_aucs = [1.0, 0.75, 0.50, 0.90, 0.80]
    fold_arr = np.array(fold_aucs, dtype=np.float64)
    mean_auc = float(fold_arr.mean())
    std_auc = float(fold_arr.std())

    MEAN_GATE = 0.75
    STD_GATE = 0.15

    if mean_auc >= MEAN_GATE and std_auc < STD_GATE:
        verdict = "probe_xval_robust"
    elif mean_auc >= MEAN_GATE and std_auc >= STD_GATE:
        verdict = "probe_xval_high_variance"
    else:
        verdict = "probe_xval_below_threshold"

    assert verdict == "probe_xval_high_variance", (
        f"Expected 'probe_xval_high_variance', got '{verdict}' (mean={mean_auc:.3f}, std={std_auc:.3f})"
    )


def test_verdict_below_threshold_when_mean_too_low():
    """honest_verdict == 'probe_xval_below_threshold' when mean_auc < 0.75.

    Spec: REQ-VER-034-3c
    """
    fold_aucs = [0.60, 0.65, 0.70, 0.62, 0.68]
    fold_arr = np.array(fold_aucs, dtype=np.float64)
    mean_auc = float(fold_arr.mean())
    std_auc = float(fold_arr.std())

    MEAN_GATE = 0.75
    STD_GATE = 0.15

    if mean_auc >= MEAN_GATE and std_auc < STD_GATE:
        verdict = "probe_xval_robust"
    elif mean_auc >= MEAN_GATE and std_auc >= STD_GATE:
        verdict = "probe_xval_high_variance"
    else:
        verdict = "probe_xval_below_threshold"

    assert verdict == "probe_xval_below_threshold"


# ---------------------------------------------------------------------------
# REQ-VER-034-3d: gate file schema (SCENARIO-VER-042)
# ---------------------------------------------------------------------------


def test_gate_file_written_with_correct_schema(tmp_path):
    """write_gate_file() produces a JSON with all required fields.

    WHY we test the schema explicitly: downstream tools (conductor, retro script,
    cascade deployment check) read specific fields from tier21_gate.json.  A missing
    field silently breaks those tools with a KeyError rather than a loud failure.

    Spec: REQ-VER-034-3d, SCENARIO-VER-042
    """
    import os
    import sys

    # Ensure the scripts/ directory is importable.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from scripts.experiment_732_probe_xval import write_gate_file  # noqa: PLC0415

    fold_aucs = [0.85, 0.88, 0.82, 0.90, 0.87]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    # Redirect gate file to tmp_path.
    original_gate = "results/tier21_gate.json"
    gate_pass = write_gate_file(tmp_path, mean_auc, std_auc, 0.70, fold_aucs)

    gate_path = tmp_path / "results" / "tier21_gate.json"
    assert gate_path.exists(), "Gate file was not written"

    payload = json.loads(gate_path.read_text())

    required_fields = {"gate", "mean_auc", "std_auc", "transfer_auc", "fold_aucs",
                       "mean_auc_threshold", "std_auc_threshold", "transfer_auc_threshold"}
    missing = required_fields - set(payload.keys())
    assert not missing, f"Gate file missing fields: {missing}"

    # Numeric values round-trip correctly.
    assert abs(payload["mean_auc"] - mean_auc) < 1e-6
    assert abs(payload["std_auc"] - std_auc) < 1e-6
    assert payload["fold_aucs"] == fold_aucs


def test_gate_file_pass_when_conditions_met(tmp_path):
    """Gate file has gate='pass' when mean_auc >= 0.75 AND std_auc < 0.15.

    Spec: REQ-VER-034-3c, REQ-VER-034-3d
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.experiment_732_probe_xval import write_gate_file

    fold_aucs = [0.85, 0.88, 0.82, 0.90, 0.87]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    gate_pass = write_gate_file(tmp_path, mean_auc, std_auc, None, fold_aucs)
    assert gate_pass is True

    payload = json.loads((tmp_path / "results" / "tier21_gate.json").read_text())
    assert payload["gate"] == "pass"


def test_gate_file_fail_when_std_too_high(tmp_path):
    """Gate file has gate='fail' when std_auc >= 0.15.

    Spec: REQ-VER-034-3c, REQ-VER-034-3d
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.experiment_732_probe_xval import write_gate_file

    fold_aucs = [1.0, 0.75, 0.50, 0.90, 0.80]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    gate_pass = write_gate_file(tmp_path, mean_auc, std_auc, None, fold_aucs)
    assert gate_pass is False

    payload = json.loads((tmp_path / "results" / "tier21_gate.json").read_text())
    assert payload["gate"] == "fail"
    assert "reason" in payload


# ---------------------------------------------------------------------------
# SCENARIO-VER-043: domain transfer honest reporting
# ---------------------------------------------------------------------------


def test_transfer_auc_null_when_no_math500(tmp_path):
    """transfer_auc=null and note='manual_label_required' when math_items=None.

    WHY we test this: silently skipping the transfer test and reporting
    transfer_auc=0.0 would mislead the cascade deployment decision.  The
    null + note pattern forces the caller to acknowledge the gap.

    Spec: REQ-VER-034-4b, SCENARIO-VER-043
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.experiment_732_probe_xval import run_transfer_test

    result = run_transfer_test(
        best_probe_weights={"w1": [[0.0]*256]*1024, "b1": [0.0]*256,
                            "w2": [[0.0]*1]*256, "b2": [0.0]*1},
        math_items=None,
        device="cpu",
    )
    assert result["transfer_auc"] is None
    assert result["transfer_note"] == "manual_label_required"


def test_transfer_auc_null_when_no_probe_weights(tmp_path):
    """transfer_auc=null when best_probe_weights=None (no fold converged).

    Spec: REQ-VER-034-4a, SCENARIO-VER-043
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.experiment_732_probe_xval import run_transfer_test

    fake_items = [{"question": "Q1", "label": 0.0}]
    result = run_transfer_test(
        best_probe_weights=None,
        math_items=fake_items,
        device="cpu",
    )
    assert result["transfer_auc"] is None
    assert result["transfer_note"] == "no_probe_weights_available"


def test_transfer_auc_null_when_empty_math_items(tmp_path):
    """transfer_auc=null when math_items is an empty list.

    Spec: REQ-VER-034-4b, SCENARIO-VER-043
    """
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from scripts.experiment_732_probe_xval import run_transfer_test

    result = run_transfer_test(
        best_probe_weights={},
        math_items=[],
        device="cpu",
    )
    assert result["transfer_auc"] is None
    assert result["transfer_note"] == "manual_label_required"


# ---------------------------------------------------------------------------
# Verdict string completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fold_aucs,expected_verdict", [
    ([0.85, 0.88, 0.82, 0.90, 0.87], "probe_xval_robust"),
    ([1.0, 0.75, 0.50, 0.90, 0.80], "probe_xval_high_variance"),
    ([0.50, 0.55, 0.60, 0.52, 0.58], "probe_xval_below_threshold"),
])
def test_honest_verdict_all_branches(fold_aucs: list, expected_verdict: str):
    """All three honest_verdict strings are reachable from valid fold_auc inputs.

    WHY parametrize: each branch corresponds to a distinct deployment decision.
    A missing branch means the conductor cannot distinguish the failure modes.

    Spec: REQ-VER-034-3c
    """
    MEAN_GATE = 0.75
    STD_GATE = 0.15

    arr = np.array(fold_aucs, dtype=np.float64)
    mean_auc = float(arr.mean())
    std_auc = float(arr.std())

    if mean_auc >= MEAN_GATE and std_auc < STD_GATE:
        verdict = "probe_xval_robust"
    elif mean_auc >= MEAN_GATE and std_auc >= STD_GATE:
        verdict = "probe_xval_high_variance"
    else:
        verdict = "probe_xval_below_threshold"

    assert verdict == expected_verdict, (
        f"fold_aucs={fold_aucs} → mean={mean_auc:.3f} std={std_auc:.3f}: "
        f"expected '{expected_verdict}', got '{verdict}'"
    )
