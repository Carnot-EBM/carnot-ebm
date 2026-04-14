"""Tests for spilled_energy_extractor.py — logit-based hallucination detection.

Spec: REQ-VERIFY-076
SCENARIO-VERIFY-093 (peaked logits produce positive spill)
SCENARIO-VERIFY-094 (uniform logits produce near-zero spill)
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from carnot.pipeline.spilled_energy_extractor import (
    SpilledEnergyExtractor,
    SpilledEnergyResult,
    compute_lookahead_energy,
    compute_spilled_energy,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _uniform_logits(n_tokens: int = 4, vocab_size: int = 8) -> np.ndarray:
    """Logit array where every vocab entry is equal → uniform distribution."""
    return np.zeros((n_tokens, vocab_size), dtype=np.float64)


def _peaked_logits(n_tokens: int = 4, vocab_size: int = 8, peak: float = 10.0) -> np.ndarray:
    """Logit array where the first vocab entry dominates via large logit value."""
    arr = np.zeros((n_tokens, vocab_size), dtype=np.float64)
    arr[:, 0] = peak   # token 0 receives a large logit → high probability
    return arr


def _single_token_logits(vocab_size: int = 8) -> np.ndarray:
    """Degenerate case: exactly one token in the response."""
    return np.zeros((1, vocab_size), dtype=np.float64)


# ---------------------------------------------------------------------------
# compute_spilled_energy — uniform logits (SCENARIO-VERIFY-094)
# ---------------------------------------------------------------------------


def test_uniform_logits_near_zero_spill() -> None:
    """Uniform logits → per-token spill near zero.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-094
    """
    # REQ-VERIFY-076: spill = entropy + max_log_prob; for uniform both cancel.
    logits = _uniform_logits(n_tokens=5, vocab_size=16)
    result = compute_spilled_energy(logits, threshold=1.0)

    # Entropy = log(16) ≈ 2.77, max log-prob = -log(16) ≈ -2.77 → sum ≈ 0.
    assert abs(result.mean_spilled) < 1e-10
    assert abs(result.max_spilled) < 1e-10


def test_uniform_logits_suspected_hallucination_false() -> None:
    """Uniform logits with default threshold → suspected_hallucination is False.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-094
    """
    logits = _uniform_logits()
    result = compute_spilled_energy(logits, threshold=1.0)
    assert result.suspected_hallucination is False
    assert result.threshold_used == 1.0


def test_uniform_logits_lookahead_energy() -> None:
    """Uniform logits → lookahead energy ≈ log(vocab_size).

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-094
    """
    vocab_size = 8
    logits = _uniform_logits(n_tokens=4, vocab_size=vocab_size)
    result = compute_spilled_energy(logits)
    # max log-prob for uniform over V is -log(V); lookahead = -mean(max_lp) = log(V).
    expected_lookahead = float(np.log(vocab_size))
    assert abs(result.lookahead_energy - expected_lookahead) < 1e-10


# ---------------------------------------------------------------------------
# compute_spilled_energy — peaked logits (SCENARIO-VERIFY-093)
# ---------------------------------------------------------------------------


def test_peaked_logits_positive_spill() -> None:
    """Peaked logits → per-token spill values are positive.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093
    """
    # REQ-VERIFY-076: peaked but not degenerate → moderate entropy, not near zero.
    logits = _peaked_logits(n_tokens=3, vocab_size=8, peak=3.0)
    result = compute_spilled_energy(logits, threshold=1.0)
    assert result.mean_spilled > 0.0
    assert result.max_spilled > 0.0
    assert all(v > 0.0 for v in result.per_token_spilled)


def test_peaked_logits_mean_max_p95_populated() -> None:
    """Spilled energy result has mean, max, and p95 populated for peaked logits.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093
    """
    logits = _peaked_logits()
    result = compute_spilled_energy(logits)
    assert isinstance(result.mean_spilled, float)
    assert isinstance(result.max_spilled, float)
    assert isinstance(result.p95_spilled, float)
    # For uniform token count, mean == max == p95 (all rows identical).
    assert abs(result.mean_spilled - result.max_spilled) < 1e-10
    assert abs(result.mean_spilled - result.p95_spilled) < 1e-10


def test_peaked_very_high_logit_low_spill() -> None:
    """Very high peak logit → near-degenerate distribution → low spill.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093
    """
    # When peak is very large, one token has nearly all the probability mass.
    # Entropy ≈ 0, max_log_prob ≈ 0 → spill ≈ 0.
    logits = _peaked_logits(n_tokens=2, vocab_size=8, peak=100.0)
    result = compute_spilled_energy(logits, threshold=1.0)
    assert result.mean_spilled < 1e-6
    assert result.suspected_hallucination is False


# ---------------------------------------------------------------------------
# Threshold firing
# ---------------------------------------------------------------------------


def test_threshold_fires_when_mean_spilled_exceeds_threshold() -> None:
    """suspected_hallucination=True when mean_spilled > threshold.

    Spec: REQ-VERIFY-076
    """
    # Use a moderate peak to get non-zero spill, then set threshold just below it.
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=2.0)
    result_high = compute_spilled_energy(logits, threshold=1e-9)
    # With threshold near 0, any positive spill should fire.
    assert result_high.suspected_hallucination is True


def test_threshold_does_not_fire_when_mean_below_threshold() -> None:
    """suspected_hallucination=False when mean_spilled <= threshold.

    Spec: REQ-VERIFY-076
    """
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=100.0)
    result = compute_spilled_energy(logits, threshold=1.0)
    assert result.suspected_hallucination is False


def test_threshold_stored_in_result() -> None:
    """threshold_used is preserved in the result.

    Spec: REQ-VERIFY-076
    """
    logits = _uniform_logits()
    result = compute_spilled_energy(logits, threshold=2.5)
    assert result.threshold_used == 2.5


# ---------------------------------------------------------------------------
# Single token edge case
# ---------------------------------------------------------------------------


def test_single_token() -> None:
    """Degenerate case with a single response token works without error.

    Spec: REQ-VERIFY-076
    """
    logits = _single_token_logits(vocab_size=4)
    result = compute_spilled_energy(logits, threshold=1.0)
    assert result.per_token_spilled.shape == (1,)
    assert isinstance(result.mean_spilled, float)
    assert isinstance(result.lookahead_energy, float)


# ---------------------------------------------------------------------------
# compute_lookahead_energy — standalone function
# ---------------------------------------------------------------------------


def test_compute_lookahead_energy_uniform() -> None:
    """compute_lookahead_energy on uniform logits returns log(vocab_size).

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-094
    """
    vocab_size = 32
    logits = np.zeros((5, vocab_size), dtype=np.float64)
    la = compute_lookahead_energy(logits)
    assert abs(la - np.log(vocab_size)) < 1e-10


def test_compute_lookahead_energy_peaked() -> None:
    """compute_lookahead_energy on very peaked logits returns near zero.

    Spec: REQ-VERIFY-076
    """
    logits = _peaked_logits(n_tokens=3, vocab_size=8, peak=100.0)
    la = compute_lookahead_energy(logits)
    # max log_prob ≈ 0 for near-degenerate distribution → lookahead ≈ 0.
    assert la < 1e-6


def test_compute_lookahead_energy_raises_for_1d() -> None:
    """compute_lookahead_energy raises ValueError for non-2D input.

    Spec: REQ-VERIFY-076
    """
    with pytest.raises(ValueError, match="2-D"):
        compute_lookahead_energy(np.zeros(8, dtype=np.float64))


def test_compute_lookahead_energy_raises_for_empty() -> None:
    """compute_lookahead_energy raises ValueError for zero tokens.

    Spec: REQ-VERIFY-076
    """
    with pytest.raises(ValueError, match="at least one token"):
        compute_lookahead_energy(np.zeros((0, 8), dtype=np.float64))


# ---------------------------------------------------------------------------
# SpilledEnergyResult serialization (SCENARIO-VERIFY-093)
# ---------------------------------------------------------------------------


def test_result_to_dict_json_round_trip() -> None:
    """SpilledEnergyResult.to_dict() round-trips through JSON without error.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093
    """
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=3.0)
    result = compute_spilled_energy(logits, threshold=1.0)
    d = result.to_dict()
    # Must be JSON-serializable with no custom encoder.
    serialized = json.dumps(d)
    parsed = json.loads(serialized)

    assert isinstance(parsed["per_token_spilled"], list)
    assert len(parsed["per_token_spilled"]) == 4
    assert isinstance(parsed["mean_spilled"], float)
    assert isinstance(parsed["suspected_hallucination"], bool)
    assert parsed["threshold_used"] == 1.0


def test_result_to_json_deterministic() -> None:
    """to_json() produces byte-identical output on repeated calls.

    Spec: REQ-VERIFY-076
    """
    logits = _uniform_logits(n_tokens=3, vocab_size=4)
    result = compute_spilled_energy(logits)
    assert result.to_json() == result.to_json()


def test_result_to_dict_contains_all_fields() -> None:
    """to_dict() contains all required SpilledEnergyResult fields.

    Spec: REQ-VERIFY-076
    """
    logits = _uniform_logits()
    result = compute_spilled_energy(logits)
    d = result.to_dict()
    required = {
        "per_token_spilled",
        "mean_spilled",
        "max_spilled",
        "p95_spilled",
        "lookahead_energy",
        "suspected_hallucination",
        "threshold_used",
    }
    assert required == set(d.keys())


# ---------------------------------------------------------------------------
# compute_spilled_energy — input validation
# ---------------------------------------------------------------------------


def test_raises_for_1d_input() -> None:
    """compute_spilled_energy raises ValueError for 1-D input.

    Spec: REQ-VERIFY-076
    """
    with pytest.raises(ValueError, match="2-D"):
        compute_spilled_energy(np.zeros(8, dtype=np.float64))


def test_raises_for_empty_tokens() -> None:
    """compute_spilled_energy raises ValueError for (0, V) input.

    Spec: REQ-VERIFY-076
    """
    with pytest.raises(ValueError, match="at least one token"):
        compute_spilled_energy(np.zeros((0, 8), dtype=np.float64))


# ---------------------------------------------------------------------------
# SpilledEnergyExtractor — extract_from_array
# ---------------------------------------------------------------------------


def test_extractor_extract_from_array_uniform() -> None:
    """SpilledEnergyExtractor.extract_from_array() works on uniform logits.

    Spec: REQ-VERIFY-076
    """
    extractor = SpilledEnergyExtractor()
    logits = _uniform_logits()
    result = extractor.extract_from_array(logits, threshold=1.0)
    assert isinstance(result, SpilledEnergyResult)
    assert result.suspected_hallucination is False


def test_extractor_extract_from_array_peaked() -> None:
    """SpilledEnergyExtractor.extract_from_array() works on peaked logits.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093
    """
    extractor = SpilledEnergyExtractor()
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=3.0)
    result = extractor.extract_from_array(logits, threshold=1e-9)
    assert isinstance(result, SpilledEnergyResult)
    assert result.suspected_hallucination is True


# ---------------------------------------------------------------------------
# SpilledEnergyExtractor — extract_from_file
# ---------------------------------------------------------------------------


def test_extractor_extract_from_file(tmp_path: pathlib.Path) -> None:
    """SpilledEnergyExtractor.extract_from_file() loads .npy and returns result.

    Spec: REQ-VERIFY-076
    """
    logits = _peaked_logits(n_tokens=3, vocab_size=8, peak=2.0)
    npy_path = tmp_path / "test_logits.npy"
    np.save(npy_path, logits)

    extractor = SpilledEnergyExtractor()
    result = extractor.extract_from_file(str(npy_path), threshold=1.0)
    assert isinstance(result, SpilledEnergyResult)
    assert result.per_token_spilled.shape == (3,)


def test_extractor_extract_from_file_not_found(tmp_path: pathlib.Path) -> None:
    """extract_from_file raises FileNotFoundError for missing path.

    Spec: REQ-VERIFY-076
    """
    extractor = SpilledEnergyExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_file(str(tmp_path / "nonexistent.npy"))


def test_extractor_extract_from_file_pathlib(tmp_path: pathlib.Path) -> None:
    """extract_from_file accepts a pathlib.Path object.

    Spec: REQ-VERIFY-076
    """
    logits = _uniform_logits()
    npy_path = tmp_path / "logits.npy"
    np.save(npy_path, logits)

    extractor = SpilledEnergyExtractor()
    result = extractor.extract_from_file(npy_path)
    assert isinstance(result, SpilledEnergyResult)


# ---------------------------------------------------------------------------
# Loading saved logits from data/research/logits_282_*.npy (if Exp 282 ran)
# ---------------------------------------------------------------------------


def test_load_exp282_logits_if_present() -> None:
    """If Exp 282 logit files exist, load and analyze them without error.

    Spec: REQ-VERIFY-076
    """
    data_dir = pathlib.Path("data/research")
    npy_files = sorted(data_dir.glob("logits_282_*.npy")) if data_dir.exists() else []

    if not npy_files:
        pytest.skip("No Exp 282 logit files found — Exp 282 has not been run")

    extractor = SpilledEnergyExtractor()
    for npy_path in npy_files[:3]:   # Check up to 3 files to keep test fast.
        result = extractor.extract_from_file(npy_path)
        assert isinstance(result, SpilledEnergyResult)
        assert result.per_token_spilled.shape[0] >= 1
        assert result.lookahead_energy >= 0.0


# ---------------------------------------------------------------------------
# Pipeline integration — VerifyRepairPipeline.verify_spilled_energy
# ---------------------------------------------------------------------------


def test_pipeline_verify_spilled_energy_from_array() -> None:
    """VerifyRepairPipeline.verify_spilled_energy() accepts a numpy array.

    Spec: REQ-VERIFY-076
    """
    pipeline = VerifyRepairPipeline()
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=3.0)
    result = pipeline.verify_spilled_energy(logits, threshold=1.0)
    assert isinstance(result, SpilledEnergyResult)


def test_pipeline_verify_spilled_energy_from_file(tmp_path: pathlib.Path) -> None:
    """VerifyRepairPipeline.verify_spilled_energy() accepts a file path string.

    Spec: REQ-VERIFY-076
    """
    logits = _uniform_logits(n_tokens=2, vocab_size=4)
    npy_path = tmp_path / "pipeline_logits.npy"
    np.save(npy_path, logits)

    pipeline = VerifyRepairPipeline()
    result = pipeline.verify_spilled_energy(str(npy_path), threshold=1.0)
    assert isinstance(result, SpilledEnergyResult)
    assert result.suspected_hallucination is False


def test_pipeline_verify_spilled_energy_does_not_break_verify() -> None:
    """verify_spilled_energy() is additive: existing verify() still works after calling it.

    Spec: REQ-VERIFY-076
    """
    pipeline = VerifyRepairPipeline()
    logits = _uniform_logits()
    _se_result = pipeline.verify_spilled_energy(logits)

    # Existing verify() path must be unaffected.
    vr = pipeline.verify(question="What is 2+2?", response="4")
    assert hasattr(vr, "verified")


def test_pipeline_export_from_init() -> None:
    """SpilledEnergyResult and related symbols are importable from carnot.pipeline.

    Spec: REQ-VERIFY-076
    """
    from carnot.pipeline import (  # noqa: F401
        SpilledEnergyExtractor,
        SpilledEnergyResult,
        compute_lookahead_energy,
        compute_spilled_energy,
    )
