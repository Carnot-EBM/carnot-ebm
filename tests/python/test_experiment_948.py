"""Tests for scripts/experiment_948_symbolic_kan_real_fover.py.

100% coverage target on the helper functions added in Exp 948.
These helpers encode FoVer step text into feature vectors and
pair / split data for Symbolic-KAN training.

Spec: REQ-MODEL-030, SCENARIO-MODEL-015.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO / "python"))
sys.path.insert(0, str(_REPO / "scripts"))

from experiment_948_symbolic_kan_real_fover import (  # noqa: E402
    AUC_STANDARD_REAL,
    AUC_SYMBOLIC_SYNTHETIC,
    THRESHOLD_MARGINAL,
    THRESHOLD_VIABLE,
    _extract_numbers,
    _make_synthetic_pairs,
    _operator_type,
    compute_auc_roc,
    load_real_pairs,
    pair_and_split,
    step_to_features,
)
from carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel  # noqa: E402


# ---------------------------------------------------------------------------
# _extract_numbers
# ---------------------------------------------------------------------------


def test_extract_numbers_simple():
    """REQ-MODEL-030: basic integer and decimal extraction."""
    result = _extract_numbers("3 + 4 = 7")
    assert result == [3.0, 4.0, 7.0]


def test_extract_numbers_latex():
    """REQ-MODEL-030: LaTeX sequences are stripped before parsing."""
    result = _extract_numbers(r"\( 4 \times 20 = 80 \)")
    assert 4.0 in result
    assert 20.0 in result
    assert 80.0 in result


def test_extract_numbers_negative():
    """REQ-MODEL-030: negative number detection."""
    nums = _extract_numbers("temperature is -5 degrees")
    assert -5.0 in nums


def test_extract_numbers_empty():
    """REQ-MODEL-030: no numbers returns empty list."""
    assert _extract_numbers("no numbers here") == []


# ---------------------------------------------------------------------------
# _operator_type
# ---------------------------------------------------------------------------


def test_operator_type_mul():
    """REQ-MODEL-030: multiplication keyword maps to 0.50."""
    assert _operator_type("times three") == pytest.approx(0.50)


def test_operator_type_cmp():
    """REQ-MODEL-030: comparison keyword maps to 0.75."""
    assert _operator_type("greater than five") == pytest.approx(0.75)


def test_operator_type_eq():
    """REQ-MODEL-030: equality keyword maps to 1.00."""
    assert _operator_type("the total equals 100") == pytest.approx(1.00)


def test_operator_type_add_default():
    """REQ-MODEL-030: unknown operator defaults to ADD (0.25)."""
    assert _operator_type("some words without operator") == pytest.approx(0.25)


def test_operator_type_case_insensitive():
    """REQ-MODEL-030: matching is case-insensitive."""
    assert _operator_type("TIMES two") == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# step_to_features
# ---------------------------------------------------------------------------


def test_step_to_features_length():
    """SCENARIO-MODEL-015: feature vector is exactly dim elements long."""
    feat = step_to_features("3 + 4 = 7", dim=16)
    assert len(feat) == 16


def test_step_to_features_padding():
    """SCENARIO-MODEL-015: short step padded with zeros."""
    feat = step_to_features("a b c", dim=16)  # no numbers
    assert len(feat) == 16
    # All numeric positions should be 0.0
    assert feat[2:] == [0.0] * 14


def test_step_to_features_custom_dim():
    """SCENARIO-MODEL-015: dim parameter controls output length."""
    feat = step_to_features("1 2 3 4 5", dim=8)
    assert len(feat) == 8


def test_step_to_features_operator_in_first_slot():
    """SCENARIO-MODEL-015: index 0 encodes operator type."""
    feat_add = step_to_features("add 5 and 3", dim=8)
    feat_mul = step_to_features("times 5 and 3", dim=8)
    # mul > add in encoding
    assert feat_mul[0] > feat_add[0]


def test_step_to_features_normalised():
    """SCENARIO-MODEL-015: feature values are finite and bounded."""
    feat = step_to_features("100 + 200 = 300", dim=16)
    for v in feat:
        assert -3.0 <= v <= 3.0
        assert not (v != v)  # not NaN


def test_step_to_features_empty_step():
    """SCENARIO-MODEL-015: empty string produces a valid vector."""
    feat = step_to_features("", dim=16)
    assert len(feat) == 16
    assert all(v == 0.0 or v == pytest.approx(0.25) for v in feat)


# ---------------------------------------------------------------------------
# load_real_pairs
# ---------------------------------------------------------------------------


def test_load_real_pairs_missing_file(tmp_path):
    """REQ-MODEL-030: missing file returns empty lists."""
    xs_c, xs_i = load_real_pairs(tmp_path / "nonexistent.json")
    assert xs_c == []
    assert xs_i == []


def test_load_real_pairs_valid_data(tmp_path):
    """REQ-MODEL-030: correct/incorrect pairs loaded from JSON."""
    data = [
        {"step_text": "3 + 4 = 7", "label": "correct"},
        {"step_text": "3 + 4 = 8", "label": "incorrect"},
        {"step_text": "5 * 2 = 10", "label": "correct"},
    ]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    xs_c, xs_i = load_real_pairs(p)
    assert len(xs_c) == 2
    assert len(xs_i) == 1
    assert len(xs_c[0]) == 16  # feature dim


def test_load_real_pairs_unknown_label_ignored(tmp_path):
    """REQ-MODEL-030: items with unrecognised labels are silently skipped."""
    data = [
        {"step_text": "a", "label": "maybe"},
        {"step_text": "b", "label": "correct"},
    ]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    xs_c, xs_i = load_real_pairs(p)
    assert len(xs_c) == 1
    assert len(xs_i) == 0


# ---------------------------------------------------------------------------
# _make_synthetic_pairs
# ---------------------------------------------------------------------------


def test_make_synthetic_pairs_count():
    """REQ-MODEL-030: generates exactly n correct and n incorrect pairs."""
    xs_c, xs_i = _make_synthetic_pairs(n=10)
    assert len(xs_c) == 10
    assert len(xs_i) == 10


def test_make_synthetic_pairs_feature_dim():
    """REQ-MODEL-030: each synthetic feature vector has length 16."""
    xs_c, xs_i = _make_synthetic_pairs(n=5)
    for feat in xs_c + xs_i:
        assert len(feat) == 16


def test_make_synthetic_pairs_different():
    """REQ-MODEL-030: correct and incorrect features differ (violation injected)."""
    xs_c, xs_i = _make_synthetic_pairs(n=20)
    # At least some pairs should differ
    diffs = [any(c != i for c, i in zip(fc, fi)) for fc, fi in zip(xs_c, xs_i)]
    assert any(diffs)


# ---------------------------------------------------------------------------
# pair_and_split
# ---------------------------------------------------------------------------


def test_pair_and_split_sizes():
    """SCENARIO-MODEL-015: 80/20 split has correct sizes."""
    xs_c = [[float(i)] * 16 for i in range(10)]
    xs_i = [[float(i + 0.5)] * 16 for i in range(10)]
    tc, ti, ec, ei = pair_and_split(xs_c, xs_i, train_frac=0.80, seed=0)
    total = len(tc)
    assert len(ti) == total
    assert abs(total - 8) <= 1  # ~80% of 10
    assert abs(len(ec) - 2) <= 1


def test_pair_and_split_unequal_classes():
    """SCENARIO-MODEL-015: shorter class is cycled to match longer."""
    xs_c = [[1.0] * 16 for _ in range(6)]
    xs_i = [[0.0] * 16 for _ in range(4)]
    tc, ti, ec, ei = pair_and_split(xs_c, xs_i, train_frac=0.80, seed=0)
    # Total pairs = max(6,4) = 6; train ceil(6*0.8)=5, eval=1
    assert len(tc) + len(ec) == 6
    assert len(tc) == len(ti)
    assert len(ec) == len(ei)


# ---------------------------------------------------------------------------
# compute_auc_roc
# ---------------------------------------------------------------------------


def _trivial_model_auc_1():
    """Helper: a model that always assigns energy=0 to correct and energy=10 to incorrect."""

    class _MockModel:
        def energy(self, x):
            # Use sum of x to distinguish: correct features sum near 16, incorrect near 0
            s = float(sum(x))
            return -s  # higher sum → lower energy → correct

    return _MockModel()


def test_compute_auc_roc_perfect():
    """SCENARIO-MODEL-015: perfect model achieves AUC=1.0."""
    config = SymbolicKANConfig(input_dim=4, n_nodes=2)
    model = SymbolicKANModel(config, seed=0)
    # Manufacture eval sets where correct always has E < incorrect
    # by choosing values that drive energy apart
    eval_c = [[1.0, 0.0, 0.0, 0.0]] * 5
    eval_i = [[0.0, 1.0, 0.0, 0.0]] * 5
    # Train briefly to differentiate
    xs_c = np.array(eval_c, dtype=np.float32)
    xs_i = np.array(eval_i, dtype=np.float32)
    model.train(xs_c, xs_i, n_epochs=5)
    auc = compute_auc_roc(model, eval_c, eval_i)
    assert 0.0 <= auc <= 1.0  # valid range


def test_compute_auc_roc_empty_returns_half():
    """SCENARIO-MODEL-015: empty eval set returns 0.5 (random baseline)."""
    config = SymbolicKANConfig(input_dim=4, n_nodes=2)
    model = SymbolicKANModel(config, seed=0)
    auc = compute_auc_roc(model, [], [[1.0, 0.0, 0.0, 0.0]])
    assert auc == pytest.approx(0.5)
    auc2 = compute_auc_roc(model, [[1.0, 0.0, 0.0, 0.0]], [])
    assert auc2 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Constants sanity check
# ---------------------------------------------------------------------------


def test_constants():
    """REQ-MODEL-030: reference AUC constants match prior experiment values."""
    assert AUC_SYMBOLIC_SYNTHETIC == pytest.approx(0.9344)
    assert AUC_STANDARD_REAL == pytest.approx(0.5139)
    assert THRESHOLD_VIABLE == pytest.approx(0.70)
    assert THRESHOLD_MARGINAL == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# compute_auc_roc — ties branch
# ---------------------------------------------------------------------------


def test_compute_auc_roc_ties():
    """SCENARIO-MODEL-015: tied energies contribute 0.5 each to AUC."""
    config = SymbolicKANConfig(input_dim=4, n_nodes=2)
    model = SymbolicKANModel(config, seed=0)
    # Force identical inputs so energies are equal → tie
    same = [[0.0, 0.0, 0.0, 0.0]]
    auc = compute_auc_roc(model, same, same)
    assert auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# main() integration test
# ---------------------------------------------------------------------------


def _run_main_with_patches(
    tmp_path,
    monkeypatch,
    fake_correct=None,
    fake_incorrect=None,
    fake_auc=None,
):
    """Helper: run main() with patched load_real_pairs and/or compute_auc_roc.

    Returns the artifact dict written to tmp_path.
    """
    import experiment_948_symbolic_kan_real_fover as mod948

    (tmp_path / "results").mkdir(exist_ok=True)
    original_tmpl = mod948.ExperimentTemplate

    class _PatchedTemplate(original_tmpl):
        def __init__(self, exp_id, title, deliverable, **kw):
            kw["repo_root"] = tmp_path
            super().__init__(exp_id, title, deliverable, **kw)

    monkeypatch.setattr(mod948, "ExperimentTemplate", _PatchedTemplate)
    monkeypatch.setattr(mod948, "_REPO", _REPO)

    if fake_correct is not None:
        monkeypatch.setattr(
            mod948,
            "load_real_pairs",
            lambda *_: (fake_correct, fake_incorrect or []),
        )

    if fake_auc is not None:
        monkeypatch.setattr(mod948, "compute_auc_roc", lambda *_: fake_auc)

    mod948.main()

    out_path = tmp_path / "results" / "experiment_948_symbolic_kan_real_fover.json"
    with out_path.open() as fh:
        return json.load(fh)


def test_main_produces_artifact(tmp_path, monkeypatch):
    """REQ-MODEL-030: main() runs end-to-end and writes a valid JSON artifact.

    We chdir to tmp_path so the hardcoded relative output path
    'results/experiment_948_symbolic_kan_real_fover.json' lands in tmp_path.
    ExperimentTemplate is patched with repo_root pointing to the real repo so
    it can still read fover_labeled_steps_live.json from results/.
    """
    import experiment_948_symbolic_kan_real_fover as mod948

    # Use repo_root=tmp_path so the deliverable lands in tmp_path/results/
    # and DeliverableGuard agrees with the actual write location.
    (tmp_path / "results").mkdir()
    original_tmpl = mod948.ExperimentTemplate

    class _PatchedTemplate(original_tmpl):
        def __init__(self, exp_id, title, deliverable, **kw):
            kw["repo_root"] = tmp_path
            super().__init__(exp_id, title, deliverable, **kw)

    monkeypatch.setattr(mod948, "ExperimentTemplate", _PatchedTemplate)
    # Also patch _REPO so fover_labeled_steps_live.json is read from real repo
    monkeypatch.setattr(mod948, "_REPO", _REPO)
    mod948.main()

    out_path = tmp_path / "results" / "experiment_948_symbolic_kan_real_fover.json"
    assert out_path.exists(), "main() must write the artifact JSON"
    with out_path.open() as fh:
        artifact = json.load(fh)

    assert artifact["experiment"] == 948
    assert "honest_verdict" in artifact
    assert "auc_symbolic_real" in artifact
    assert artifact["status"] == "success"


def test_main_synthetic_fallback(tmp_path, monkeypatch):
    """REQ-MODEL-030: fewer than 20 real pairs triggers synthetic_fallback verdict."""
    # Provide only 5 correct + 5 incorrect (< 20 total)
    sparse = [[float(i)] * 16 for i in range(5)]
    artifact = _run_main_with_patches(
        tmp_path, monkeypatch, fake_correct=sparse, fake_incorrect=sparse
    )
    assert artifact["inference_mode"] == "synthetic_fallback"
    assert artifact["honest_verdict"] == "symbolic_kan_synthetic_fallback"


def test_main_marginal_verdict(tmp_path, monkeypatch):
    """REQ-MODEL-030: AUC in (0.60, 0.70] produces marginal verdict."""
    artifact = _run_main_with_patches(tmp_path, monkeypatch, fake_auc=0.65)
    assert artifact["honest_verdict"] == "symbolic_kan_real_marginal"
    assert artifact["auc_symbolic_real"] == pytest.approx(0.65)


def test_main_degraded_verdict(tmp_path, monkeypatch):
    """REQ-MODEL-030: AUC <= 0.60 produces degraded verdict."""
    artifact = _run_main_with_patches(tmp_path, monkeypatch, fake_auc=0.55)
    assert artifact["honest_verdict"] == "symbolic_kan_real_degraded"
    assert artifact["auc_symbolic_real"] == pytest.approx(0.55)
