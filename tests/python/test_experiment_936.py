"""Tests for Exp 936: KAN Tier 4 Real Data — AutoKnots on FoVer-Labeled Pairs.

Covers: _extract_features, _load_real_fover_pairs, _make_synthetic_fallback,
        _compute_auc, _run_kan_eval, _train_split, and the deliverable JSON.

Spec: REQ-SELF-008, SCENARIO-SELF-008
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_936_kan_tier4_real_data import (  # noqa: E402
    INPUT_DIM,
    MIN_REAL_PAIRS,
    SEED,
    TRAIN_FRAC,
    _compute_auc,
    _extract_features,
    _load_real_fover_pairs,
    _make_synthetic_fallback,
    _run_kan_eval,
    _train_split,
)


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------

def test_extract_features_shape():
    # REQ-SELF-008: feature vector must have INPUT_DIM elements
    feats = _extract_features("The answer is 42.")
    assert feats.shape == (INPUT_DIM,)
    assert feats.dtype == np.float32


def test_extract_features_binary():
    # All values must be 0 or 1 (binary indicator flags)
    feats = _extract_features("x = 10 therefore total is 10 km")
    assert set(np.unique(feats)).issubset({0.0, 1.0})


def test_extract_features_digit_flag():
    # Bit 0 should fire when digits are present
    feats_digit = _extract_features("3 + 4 = 7")
    assert feats_digit[0] == 1.0
    feats_nodigit = _extract_features("no numbers here")
    assert feats_nodigit[0] == 0.0


def test_extract_features_conclusion_flag():
    # Bit 2 should fire for conclusion words
    feats = _extract_features("Therefore the answer is correct.")
    assert feats[2] == 1.0


def test_extract_features_negation_flag():
    # Bit 8 should fire for negation words
    feats = _extract_features("This is not correct.")
    assert feats[8] == 1.0


def test_extract_features_no_conclusion_flag():
    # Bit 15 should fire when no conclusion word present
    feats = _extract_features("Some random text with 5 digits.")
    assert feats[15] == 1.0

    feats_conc = _extract_features("Therefore the total is 100.")
    assert feats_conc[15] == 0.0


def test_extract_features_empty_string():
    # Should not raise on empty input
    feats = _extract_features("")
    assert feats.shape == (INPUT_DIM,)


# ---------------------------------------------------------------------------
# _load_real_fover_pairs
# ---------------------------------------------------------------------------

def test_load_real_fover_pairs_correct_and_incorrect():
    data = [
        {"step_text": "The answer is 3 + 4 = 7. Therefore correct.", "label": "correct"},
        {"step_text": "Not sure if this is right.", "label": "incorrect"},
        {"step_text": "2 * 5 = 10. Total is 10.", "label": "correct"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    correct, incorrect = _load_real_fover_pairs(path)
    assert len(correct) == 2
    assert len(incorrect) == 1
    assert all(arr.shape == (INPUT_DIM,) for arr in correct)


def test_load_real_fover_pairs_wrong_label_alias():
    # 'wrong' should be treated same as 'incorrect'
    data = [
        {"step_text": "Definitely wrong approach.", "label": "wrong"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    correct, incorrect = _load_real_fover_pairs(path)
    assert len(correct) == 0
    assert len(incorrect) == 1


def test_load_real_fover_pairs_unknown_label_skipped():
    data = [
        {"step_text": "Some text.", "label": "ambiguous"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    correct, incorrect = _load_real_fover_pairs(path)
    assert len(correct) == 0
    assert len(incorrect) == 0


# ---------------------------------------------------------------------------
# _make_synthetic_fallback
# ---------------------------------------------------------------------------

def test_make_synthetic_fallback_shapes():
    correct, wrong = _make_synthetic_fallback(50, INPUT_DIM, SEED)
    assert correct.shape == (25, INPUT_DIM)
    assert wrong.shape == (25, INPUT_DIM)


def test_make_synthetic_fallback_binary():
    correct, wrong = _make_synthetic_fallback(20, INPUT_DIM, SEED)
    assert set(np.unique(correct)).issubset({0.0, 1.0})
    assert set(np.unique(wrong)).issubset({0.0, 1.0})


def test_make_synthetic_fallback_deterministic():
    c1, w1 = _make_synthetic_fallback(50, INPUT_DIM, 42)
    c2, w2 = _make_synthetic_fallback(50, INPUT_DIM, 42)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(w1, w2)


# ---------------------------------------------------------------------------
# _compute_auc
# ---------------------------------------------------------------------------

def test_compute_auc_perfect():
    # Perfect model: all correct energies lower than wrong energies
    energies_correct = np.array([0.1, 0.2, 0.3])
    energies_wrong = np.array([0.8, 0.9, 1.0])
    auc = _compute_auc(energies_correct, energies_wrong)
    assert auc == 1.0


def test_compute_auc_worst():
    # Inverted model: all correct energies higher than wrong energies
    energies_correct = np.array([0.8, 0.9, 1.0])
    energies_wrong = np.array([0.1, 0.2, 0.3])
    auc = _compute_auc(energies_correct, energies_wrong)
    assert auc == 0.0


def test_compute_auc_random():
    # Mixed: ~0.5 expected
    energies_correct = np.array([0.5, 0.5])
    energies_wrong = np.array([0.5, 0.5])
    auc = _compute_auc(energies_correct, energies_wrong)
    assert auc == 0.5


def test_compute_auc_empty():
    # Empty arrays: should return 0.0 without error
    auc = _compute_auc(np.array([]), np.array([]))
    assert auc == 0.0


# ---------------------------------------------------------------------------
# _train_split
# ---------------------------------------------------------------------------

def test_train_split_sizes():
    correct = np.ones((10, INPUT_DIM), dtype=np.float32)
    wrong = np.ones((10, INPUT_DIM), dtype=np.float32) * 2
    c_train, c_test, w_train, w_test = _train_split(correct, wrong, 0.8, SEED)
    assert len(c_train) == 8
    assert len(c_test) == 2
    assert len(w_train) == 8
    assert len(w_test) == 2


def test_train_split_no_overlap():
    correct = np.arange(30, dtype=np.float32).reshape(10, 3)
    wrong = np.arange(30, 60, dtype=np.float32).reshape(10, 3)
    c_train, c_test, _, _ = _train_split(correct, wrong, 0.8, SEED)
    # All test rows must not appear in train
    train_set = set(map(tuple, c_train.tolist()))
    for row in c_test:
        assert tuple(row.tolist()) not in train_set


def test_train_split_small_data_at_least_one():
    # With just 2 samples per class, train gets 1 minimum
    correct = np.ones((2, INPUT_DIM), dtype=np.float32)
    wrong = np.ones((2, INPUT_DIM), dtype=np.float32)
    c_train, c_test, w_train, w_test = _train_split(correct, wrong, 0.8, SEED)
    assert len(c_train) >= 1
    assert len(w_train) >= 1


# ---------------------------------------------------------------------------
# _run_kan_eval (integration — uses real KANModel, CPU JAX)
# ---------------------------------------------------------------------------

def test_run_kan_eval_returns_float():
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.random as jrandom
    from carnot.models.kan import KANConfig, KANModel

    config = KANConfig(input_dim=4, num_knots=4, degree=3, sparse=False)
    kan = KANModel(config, key=jrandom.PRNGKey(0))
    correct_embs = np.ones((3, 4), dtype=np.float32)
    wrong_embs = np.zeros((3, 4), dtype=np.float32)
    auc = _run_kan_eval(kan, correct_embs, wrong_embs)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# Deliverable JSON schema
# ---------------------------------------------------------------------------

def test_deliverable_json_required_fields():
    path = Path("results/experiment_936_kan_tier4_real_data.json")
    assert path.exists(), "Deliverable JSON must exist after running the experiment"
    with path.open() as f:
        d = json.load(f)

    required = [
        "experiment", "title", "run_date", "status", "honest_verdict",
        "inference_mode", "n_real_pairs", "baseline_auc", "post_refinement_auc",
        "signed_auc_improvement", "delta_vs_exp910_post", "n_knots_added",
        "n_knots_removed", "refinement_rounds", "round_summaries",
    ]
    for field in required:
        assert field in d, f"Missing field: {field}"


def test_deliverable_honest_verdict_valid():
    path = Path("results/experiment_936_kan_tier4_real_data.json")
    with path.open() as f:
        d = json.load(f)
    valid_verdicts = {
        "real_data_improves_over_synthetic",
        "real_data_comparable",
        "real_data_below_synthetic",
        "real_data_insufficient_synthetic_fallback",
    }
    assert d["honest_verdict"] in valid_verdicts


def test_deliverable_auc_in_range():
    path = Path("results/experiment_936_kan_tier4_real_data.json")
    with path.open() as f:
        d = json.load(f)
    assert 0.0 <= d["baseline_auc"] <= 1.0
    assert 0.0 <= d["post_refinement_auc"] <= 1.0
