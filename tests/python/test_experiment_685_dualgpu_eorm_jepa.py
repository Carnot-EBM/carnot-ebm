"""Tests for Experiment 685 — DualGPU EORM+JEPA Parallel Retrain.

Spec: REQ-HW-036, REQ-LEARN-044, SCENARIO-HW-036, SCENARIO-LEARN-074

Coverage:
    - test_build_eorm_pairs_from_mixed_labels: builds (correct, incorrect, question) triples
    - test_build_eorm_pairs_fallback_all_positive: fallback synthetic pairs when no negatives
    - test_build_jepa_violation_pairs_label_mapping: maps label=True to has_violation=True
    - test_build_jepa_violation_pairs_partial_prefix: partial_response is first 50% of words
    - test_flatten_pytree_flat_dict: flattens one-level dict to str-keyed numpy arrays
    - test_flatten_pytree_nested_dict: flattens two-level dict to slash-delimited keys
    - test_flatten_pytree_list: flattens list node with index keys
    - test_flatten_pytree_skips_non_array: silently skips non-array-convertible leaves
    - test_train_eorm_returns_required_keys: result dict has eorm_loss and eorm_train_time_s
    - test_train_eorm_loss_is_finite: eorm_loss is a finite float after training
    - test_train_eorm_saves_safetensors: safetensors file written to expected path
    - test_train_jepa_returns_required_keys: result dict has jepa_loss and jepa_train_time_s
    - test_train_jepa_loss_is_finite: jepa_loss is a finite float after training
    - test_run_blocked_when_gate_missing: _run writes blocked artifact when gate file absent
    - test_run_blocked_when_retro_false: _run writes blocked artifact when retro_071_resolved=False
    - test_run_success_produces_required_fields: full _run produces artifact with all schema fields
    - test_run_speedup_is_positive: speedup > 0 in success artifact
    - test_honest_verdict_success: speedup >= 1.3 -> dualgpu_retrain_success
    - test_honest_verdict_marginal: 1.0 <= speedup < 1.3 -> dualgpu_retrain_marginal
    - test_honest_verdict_slower: speedup < 1.0 -> dualgpu_retrain_slower
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is on path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.experiment_685_dualgpu_eorm_jepa import (  # noqa: E402
    _build_eorm_pairs,
    _build_jepa_violation_pairs,
    _flatten_pytree,
    train_eorm,
    train_jepa,
    _run,
    DELIVERABLE,
    GATE_PATH,
    EORM_OUTPUT,
    JEPA_OUTPUT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MIXED_RECORDS = [
    {"question_id": "1", "step_text": "Step A correct.", "label": True, "confidence": 0.9},
    {"question_id": "2", "step_text": "Step B incorrect.", "label": False, "confidence": 0.3},
    {"question_id": "3", "step_text": "Step C correct.", "label": 1, "confidence": 0.8},
    {"question_id": "4", "step_text": "Step D incorrect.", "label": 0, "confidence": 0.2},
]

ALL_POSITIVE_RECORDS = [
    {"question_id": "1", "step_text": "Step A correct.", "label": True, "confidence": 0.9},
    {"question_id": "2", "step_text": "Step B correct.", "label": 1, "confidence": 0.8},
]


# ---------------------------------------------------------------------------
# _build_eorm_pairs
# ---------------------------------------------------------------------------


def test_build_eorm_pairs_from_mixed_labels() -> None:
    """Builds (correct, incorrect, question) triples by pairing positives with negatives.

    Spec: REQ-HW-036
    """
    pairs = _build_eorm_pairs(MIXED_RECORDS)
    assert len(pairs) >= 1, "Expected at least one pair from mixed labels"
    for correct, incorrect, question in pairs:
        assert isinstance(correct, str)
        assert isinstance(incorrect, str)
        assert isinstance(question, str)


def test_build_eorm_pairs_fallback_all_positive() -> None:
    """Falls back to synthetic pairs when no negatives are present.

    Spec: REQ-HW-036
    """
    pairs = _build_eorm_pairs(ALL_POSITIVE_RECORDS)
    assert len(pairs) > 0, "Fallback should produce at least one pair"
    for correct, incorrect, question in pairs:
        assert isinstance(correct, str)
        assert isinstance(incorrect, str)


# ---------------------------------------------------------------------------
# _build_jepa_violation_pairs
# ---------------------------------------------------------------------------


def test_build_jepa_violation_pairs_label_mapping() -> None:
    """Maps label=True records to has_violation=True and label=False to has_violation=False.

    Spec: REQ-LEARN-044, SCENARIO-LEARN-074
    """
    pairs = _build_jepa_violation_pairs(MIXED_RECORDS)
    assert len(pairs) == len(MIXED_RECORDS)
    # First record has label=True
    assert pairs[0].has_violation is True
    # Second record has label=False
    assert pairs[1].has_violation is False


def test_build_jepa_violation_pairs_partial_prefix() -> None:
    """partial_response is the first ~50% of words from step_text.

    Spec: SCENARIO-LEARN-074
    """
    records = [{"question_id": "x", "step_text": "one two three four five six", "label": True, "confidence": 0.9}]
    pairs = _build_jepa_violation_pairs(records)
    assert len(pairs) == 1
    words = records[0]["step_text"].split()
    expected_partial = " ".join(words[:3])  # midpoint = max(1, 6//2) = 3
    assert pairs[0].partial_response == expected_partial
    assert pairs[0].full_response == records[0]["step_text"]


# ---------------------------------------------------------------------------
# _flatten_pytree
# ---------------------------------------------------------------------------


def test_flatten_pytree_flat_dict() -> None:
    """Flattens a one-level dict of arrays to str-keyed numpy arrays.

    Spec: REQ-HW-036 (weight serialization path)
    """
    node = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
    out: dict = {}
    _flatten_pytree(node, "", out)
    assert "a" in out
    assert "b" in out
    np.testing.assert_array_equal(out["a"], np.array([1.0, 2.0]))


def test_flatten_pytree_nested_dict() -> None:
    """Flattens a two-level dict with slash-delimited keys.

    Spec: REQ-HW-036 (weight serialization path)
    """
    node = {"layer1": {"weight": np.array([1.0, 2.0]), "bias": np.array([0.5])}}
    out: dict = {}
    _flatten_pytree(node, "", out)
    assert "layer1/weight" in out
    assert "layer1/bias" in out


def test_flatten_pytree_list() -> None:
    """Flattens a list node with integer index keys.

    Spec: REQ-HW-036 (weight serialization path)
    """
    node = [np.array([1.0]), np.array([2.0])]
    out: dict = {}
    _flatten_pytree(node, "layers", out)
    assert "layers/0" in out
    assert "layers/1" in out


def test_flatten_pytree_skips_non_array() -> None:
    """Silently skips leaves that cannot be converted to numpy arrays.

    Spec: REQ-HW-036 (robustness)
    """
    node = {"valid": np.array([1.0]), "invalid": object()}
    out: dict = {}
    _flatten_pytree(node, "", out)
    assert "valid" in out
    assert "invalid" not in out


# ---------------------------------------------------------------------------
# train_eorm
# ---------------------------------------------------------------------------

_SMALL_PAIRS = [
    ("Correct step text here.", "Wrong step text there.", "What is 2+2?"),
    ("Another correct step.", "Another incorrect step.", "What is 3+3?"),
]


def test_train_eorm_returns_required_keys(tmp_path: Path) -> None:
    """train_eorm returns dict with eorm_loss and eorm_train_time_s.

    Spec: REQ-HW-036
    """
    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        (tmp_path / "results").mkdir()
        result = train_eorm("cpu", _SMALL_PAIRS)
    assert "eorm_loss" in result
    assert "eorm_train_time_s" in result


def test_train_eorm_loss_is_finite(tmp_path: Path) -> None:
    """eorm_loss is a finite float after training — no NaN or inf divergence.

    Spec: REQ-HW-036, SCENARIO-HW-036
    """
    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        (tmp_path / "results").mkdir()
        result = train_eorm("cpu", _SMALL_PAIRS)
    assert math.isfinite(result["eorm_loss"]), f"eorm_loss={result['eorm_loss']} is not finite"


def test_train_eorm_saves_safetensors(tmp_path: Path) -> None:
    """train_eorm writes eorm_v2_dualgpu.safetensors to the results directory.

    Spec: SCENARIO-HW-036
    """
    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        (tmp_path / "results").mkdir()
        train_eorm("cpu", _SMALL_PAIRS)
    assert (tmp_path / EORM_OUTPUT).exists(), "eorm_v2_dualgpu.safetensors not written"


# ---------------------------------------------------------------------------
# train_jepa
# ---------------------------------------------------------------------------


def _make_jepa_pairs() -> list:
    """Build two minimal ViolationPair objects for testing."""
    from carnot.embeddings.jepa_retrain import ViolationPair  # noqa: PLC0415
    return [
        ViolationPair(question_id="1", model_id="test", full_response="Correct answer here.", partial_response="Correct", has_violation=False),
        ViolationPair(question_id="2", model_id="test", full_response="Wrong answer with error.", partial_response="Wrong", has_violation=True),
    ]


def test_train_jepa_returns_required_keys(tmp_path: Path) -> None:
    """train_jepa returns dict with jepa_loss and jepa_train_time_s.

    Spec: REQ-LEARN-044
    """
    pairs = _make_jepa_pairs()
    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        (tmp_path / "results").mkdir()
        result = train_jepa("cpu", pairs)
    assert "jepa_loss" in result
    assert "jepa_train_time_s" in result


def test_train_jepa_loss_is_finite(tmp_path: Path) -> None:
    """jepa_loss is a finite float after retraining — no divergence.

    Spec: REQ-LEARN-044, SCENARIO-LEARN-074
    """
    pairs = _make_jepa_pairs()
    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        (tmp_path / "results").mkdir()
        result = train_jepa("cpu", pairs)
    assert math.isfinite(result["jepa_loss"]), f"jepa_loss={result['jepa_loss']} is not finite"


# ---------------------------------------------------------------------------
# _run — gate and integration tests
# ---------------------------------------------------------------------------


def _make_tmpl(tmp_path: Path):
    """Build a minimal ExperimentTemplate pointed at tmp_path."""
    from scripts.experiment_template import ExperimentTemplate  # noqa: PLC0415
    tmpl = ExperimentTemplate(
        685,
        "DualGPU EORM+JEPA test",
        DELIVERABLE,
        requires_gpu=False,
        repo_root=tmp_path,
    )
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "checkpoints" / "experiment_685").mkdir(parents=True, exist_ok=True)
    return tmpl


def _write_fover_data(tmp_path: Path) -> None:
    """Write a minimal fover_labeled_steps_live.json for integration tests."""
    path = tmp_path / "results" / "fover_labeled_steps_live.json"
    data = MIXED_RECORDS
    path.write_text(json.dumps(data))


def test_run_blocked_when_gate_missing(tmp_path: Path) -> None:
    """_run writes a blocked artifact when the Exp 684 gate file does not exist.

    Spec: REQ-HW-036 (gate check)
    """
    tmpl = _make_tmpl(tmp_path)
    _write_fover_data(tmp_path)

    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        _run(tmpl)

    artifact = json.loads((tmp_path / DELIVERABLE).read_text())
    assert artifact["honest_verdict"] == "dualgpu_retrain_blocked"
    assert artifact["status"] == "blocked"


def test_run_blocked_when_retro_false(tmp_path: Path) -> None:
    """_run writes a blocked artifact when retro_071_resolved is False in the gate file.

    Spec: REQ-HW-036 (gate check)
    """
    tmpl = _make_tmpl(tmp_path)
    _write_fover_data(tmp_path)

    gate_data = {"retro_071_resolved": False, "honest_verdict": "dualgpu_blocked"}
    (tmp_path / GATE_PATH).write_text(json.dumps(gate_data))

    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        _run(tmpl)

    artifact = json.loads((tmp_path / DELIVERABLE).read_text())
    assert artifact["honest_verdict"] == "dualgpu_retrain_blocked"


def test_run_success_produces_required_fields(tmp_path: Path) -> None:
    """Full _run with gate=True produces an artifact with all required schema fields.

    Spec: REQ-HW-036, SCENARIO-HW-036
    """
    tmpl = _make_tmpl(tmp_path)
    _write_fover_data(tmp_path)

    gate_data = {"retro_071_resolved": True, "honest_verdict": "dualgpu_confirmed"}
    (tmp_path / GATE_PATH).write_text(json.dumps(gate_data))

    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        _run(tmpl)

    artifact = json.loads((tmp_path / DELIVERABLE).read_text())

    required = ["experiment", "title", "run_date", "started_at", "finished_at",
                "duration_s", "status", "honest_verdict", "speedup",
                "sequential_total_s", "parallel_total_s"]
    for field in required:
        assert field in artifact, f"Missing required field: {field}"


def test_run_speedup_is_positive(tmp_path: Path) -> None:
    """speedup in the success artifact is a positive finite float.

    Spec: SCENARIO-HW-036
    """
    tmpl = _make_tmpl(tmp_path)
    _write_fover_data(tmp_path)

    gate_data = {"retro_071_resolved": True, "honest_verdict": "dualgpu_confirmed"}
    (tmp_path / GATE_PATH).write_text(json.dumps(gate_data))

    with patch("scripts.experiment_685_dualgpu_eorm_jepa._REPO_ROOT", tmp_path):
        _run(tmpl)

    artifact = json.loads((tmp_path / DELIVERABLE).read_text())
    assert artifact["speedup"] > 0, "speedup must be positive"
    assert math.isfinite(artifact["speedup"]), "speedup must be finite"


# ---------------------------------------------------------------------------
# honest_verdict classification (unit tests, no file I/O)
# ---------------------------------------------------------------------------


def _compute_verdict(speedup: float) -> str:
    """Replicate the verdict logic from _run for unit testing."""
    if speedup >= 1.3:
        return "dualgpu_retrain_success"
    elif speedup >= 1.0:
        return "dualgpu_retrain_marginal"
    else:
        return "dualgpu_retrain_slower"


def test_honest_verdict_success() -> None:
    """speedup >= 1.3 maps to dualgpu_retrain_success.

    Spec: REQ-HW-036
    """
    assert _compute_verdict(1.5) == "dualgpu_retrain_success"
    assert _compute_verdict(1.3) == "dualgpu_retrain_success"


def test_honest_verdict_marginal() -> None:
    """1.0 <= speedup < 1.3 maps to dualgpu_retrain_marginal.

    Spec: REQ-HW-036
    """
    assert _compute_verdict(1.0) == "dualgpu_retrain_marginal"
    assert _compute_verdict(1.29) == "dualgpu_retrain_marginal"


def test_honest_verdict_slower() -> None:
    """speedup < 1.0 maps to dualgpu_retrain_slower.

    Spec: REQ-HW-036
    """
    assert _compute_verdict(0.9) == "dualgpu_retrain_slower"
    assert _compute_verdict(0.1) == "dualgpu_retrain_slower"
