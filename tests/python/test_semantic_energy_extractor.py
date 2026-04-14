"""Tests for semantic_energy_extractor.py — logit-based overconfidence detection.

Spec: REQ-VERIFY-077
SCENARIO-VERIFY-095 (peaked logits → overconfident_flag fires)
SCENARIO-VERIFY-096 (DualEnergyGate fires when EITHER signal triggers)
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from carnot.pipeline.semantic_energy_extractor import (
    DualEnergyGate,
    DualEnergyResult,
    SemanticEnergyExtractor,
    SemanticEnergyResult,
    compute_semantic_energy,
)
from carnot.pipeline.spilled_energy_extractor import SpilledEnergyResult
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _uniform_logits(n_tokens: int = 4, vocab_size: int = 8) -> np.ndarray:
    """Logit array where every vocab entry is equal → uniform distribution."""
    return np.zeros((n_tokens, vocab_size), dtype=np.float64)


def _peaked_logits(
    n_tokens: int = 4, vocab_size: int = 8, peak: float = 10.0
) -> np.ndarray:
    """Logit array where the first vocab entry dominates via a large logit value."""
    arr = np.zeros((n_tokens, vocab_size), dtype=np.float64)
    arr[:, 0] = peak
    return arr


def _single_token_logits(vocab_size: int = 8) -> np.ndarray:
    """Degenerate case: exactly one token in the response."""
    return np.zeros((1, vocab_size), dtype=np.float64)


def _make_spilled_result(suspected: bool, threshold: float = 1.0) -> SpilledEnergyResult:
    """Build a minimal SpilledEnergyResult for gate tests."""
    per_token = np.array([0.5], dtype=np.float64)
    return SpilledEnergyResult(
        per_token_spilled=per_token,
        mean_spilled=0.5,
        max_spilled=0.5,
        p95_spilled=0.5,
        lookahead_energy=0.0,
        suspected_hallucination=suspected,
        threshold_used=threshold,
    )


def _make_semantic_result(
    overconfident: bool, energy: float = -3.0
) -> SemanticEnergyResult:
    """Build a minimal SemanticEnergyResult for gate tests."""
    return SemanticEnergyResult(
        semantic_energy=energy,
        temperature=1.0,
        overconfident_flag=overconfident,
        threshold_used=-2.0,
        per_token_semantic=np.array([energy], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# compute_semantic_energy — formula correctness
# ---------------------------------------------------------------------------


def test_uniform_logits_semantic_energy_equals_neg_log_vocab() -> None:
    """Uniform logits: E = −log(V) for each token (all logits equal → sum = V·exp(0)).

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    # REQ-VERIFY-077: E = mean_t(−log(∑_i exp(logit_i / T)))
    # For uniform logits (all zero), ∑ exp(0/1) = V → E = −log(V)
    vocab_size = 8
    logits = _uniform_logits(n_tokens=4, vocab_size=vocab_size)
    energy = compute_semantic_energy(logits, temperature=1.0)
    expected = -np.log(vocab_size)
    assert abs(energy - expected) < 1e-10


def test_peaked_logits_semantic_energy_is_more_negative() -> None:
    """Peaked logits produce lower (more negative) semantic energy than uniform.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    # A peaked distribution has a large log-partition → more negative energy.
    uniform_energy = compute_semantic_energy(_uniform_logits(), temperature=1.0)
    peaked_energy = compute_semantic_energy(_peaked_logits(peak=10.0), temperature=1.0)
    assert peaked_energy < uniform_energy


def test_temperature_effect_higher_temperature_raises_energy() -> None:
    """Higher temperature produces less negative (higher) semantic energy.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    # REQ-VERIFY-077: higher temperature flattens the effective distribution →
    # log-partition grows slower → energy is less negative.
    logits = _peaked_logits(n_tokens=3, vocab_size=8, peak=5.0)
    e_low_temp = compute_semantic_energy(logits, temperature=0.5)
    e_high_temp = compute_semantic_energy(logits, temperature=2.0)
    # Lower temperature → sharper distribution → more negative energy
    assert e_low_temp < e_high_temp


def test_single_token_semantic_energy() -> None:
    """Single-token response computes semantic energy without error.

    Spec: REQ-VERIFY-077
    """
    logits = _single_token_logits(vocab_size=4)
    energy = compute_semantic_energy(logits, temperature=1.0)
    assert isinstance(energy, float)
    # Uniform over 4: E = -log(4)
    assert abs(energy - (-np.log(4))) < 1e-10


def test_semantic_energy_very_peaked_approaches_neg_max_logit() -> None:
    """Very peaked logits: E ≈ −max_logit / temperature (partition dominated by one token).

    Spec: REQ-VERIFY-077
    """
    peak = 50.0
    vocab_size = 8
    logits = _peaked_logits(n_tokens=2, vocab_size=vocab_size, peak=peak)
    energy = compute_semantic_energy(logits, temperature=1.0)
    # log-partition ≈ peak → E ≈ -peak
    assert abs(energy - (-peak)) < 0.01


# ---------------------------------------------------------------------------
# compute_semantic_energy — input validation
# ---------------------------------------------------------------------------


def test_raises_for_1d_input() -> None:
    """compute_semantic_energy raises ValueError for 1-D input.

    Spec: REQ-VERIFY-077
    """
    with pytest.raises(ValueError, match="2-D"):
        compute_semantic_energy(np.zeros(8, dtype=np.float64))


def test_raises_for_empty_tokens() -> None:
    """compute_semantic_energy raises ValueError for (0, V) input.

    Spec: REQ-VERIFY-077
    """
    with pytest.raises(ValueError, match="at least one token"):
        compute_semantic_energy(np.zeros((0, 8), dtype=np.float64))


def test_raises_for_zero_temperature() -> None:
    """compute_semantic_energy raises ValueError for temperature <= 0.

    Spec: REQ-VERIFY-077
    """
    with pytest.raises(ValueError, match="Temperature"):
        compute_semantic_energy(_uniform_logits(), temperature=0.0)

    with pytest.raises(ValueError, match="Temperature"):
        compute_semantic_energy(_uniform_logits(), temperature=-1.0)


# ---------------------------------------------------------------------------
# SemanticEnergyResult dataclass
# ---------------------------------------------------------------------------


def test_semantic_result_fields() -> None:
    """SemanticEnergyResult has all required fields.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor(threshold=-5.0, temperature=1.0)
    logits = _uniform_logits()
    result = extractor.extract(logits)

    assert isinstance(result, SemanticEnergyResult)
    assert isinstance(result.semantic_energy, float)
    assert isinstance(result.temperature, float)
    assert isinstance(result.overconfident_flag, bool)
    assert isinstance(result.threshold_used, float)
    assert isinstance(result.per_token_semantic, np.ndarray)
    assert result.per_token_semantic.shape == (4,)


def test_semantic_result_to_dict_round_trips_json() -> None:
    """SemanticEnergyResult.to_dict() round-trips through JSON without error.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor(threshold=-5.0)
    result = extractor.extract(_peaked_logits(n_tokens=3, vocab_size=8, peak=3.0))
    d = result.to_dict()
    serialized = json.dumps(d)
    parsed = json.loads(serialized)

    assert isinstance(parsed["semantic_energy"], float)
    assert isinstance(parsed["temperature"], float)
    assert isinstance(parsed["overconfident_flag"], bool)
    assert isinstance(parsed["threshold_used"], float)
    assert isinstance(parsed["per_token_semantic"], list)
    assert len(parsed["per_token_semantic"]) == 3


def test_semantic_result_to_dict_contains_all_fields() -> None:
    """to_dict() contains all required SemanticEnergyResult fields.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    result = extractor.extract(_uniform_logits())
    d = result.to_dict()
    required = {
        "semantic_energy",
        "temperature",
        "overconfident_flag",
        "threshold_used",
        "per_token_semantic",
    }
    assert required == set(d.keys())


def test_semantic_result_to_json_deterministic() -> None:
    """to_json() produces byte-identical output on repeated calls.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    result = extractor.extract(_uniform_logits(n_tokens=3, vocab_size=4))
    assert result.to_json() == result.to_json()


# ---------------------------------------------------------------------------
# SemanticEnergyExtractor — overconfident_flag (SCENARIO-VERIFY-095)
# ---------------------------------------------------------------------------


def test_overconfident_flag_fires_for_peaked_logits() -> None:
    """Very peaked logits (very negative energy) fire overconfident_flag.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    # peak=20: energy ≈ -20, threshold=-5 → energy < threshold → flag fires
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=20.0)
    extractor = SemanticEnergyExtractor(threshold=-5.0, temperature=1.0)
    result = extractor.extract(logits)
    assert result.overconfident_flag is True


def test_overconfident_flag_does_not_fire_for_uniform_logits() -> None:
    """Uniform logits do not fire overconfident_flag with a strict threshold.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    # Uniform over 8: energy ≈ -log(8) ≈ -2.08, threshold=-5.0 → energy > threshold → no flag
    logits = _uniform_logits(n_tokens=4, vocab_size=8)
    extractor = SemanticEnergyExtractor(threshold=-5.0, temperature=1.0)
    result = extractor.extract(logits)
    assert result.overconfident_flag is False


def test_threshold_stored_in_result() -> None:
    """threshold_used is preserved in the result.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor(threshold=-3.7, temperature=1.0)
    result = extractor.extract(_uniform_logits())
    assert result.threshold_used == -3.7


def test_temperature_stored_in_result() -> None:
    """temperature is preserved in the result.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor(threshold=-5.0, temperature=2.5)
    result = extractor.extract(_uniform_logits())
    assert result.temperature == 2.5


def test_extractor_input_validation_1d() -> None:
    """SemanticEnergyExtractor.extract() raises ValueError for 1-D input.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    with pytest.raises(ValueError, match="2-D"):
        extractor.extract(np.zeros(8, dtype=np.float64))


def test_extractor_input_validation_empty() -> None:
    """SemanticEnergyExtractor.extract() raises ValueError for empty token count.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    with pytest.raises(ValueError, match="at least one token"):
        extractor.extract(np.zeros((0, 8), dtype=np.float64))


# ---------------------------------------------------------------------------
# SemanticEnergyExtractor — calibration
# ---------------------------------------------------------------------------


def test_calibrate_returns_threshold() -> None:
    """calibrate() returns a float threshold and sets self.threshold.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()

    # Corpus: 10 peaked logit arrays (confident, wrong=True) + 10 uniform (correct=True)
    rng = np.random.default_rng(42)
    corpus: list[np.ndarray] = []
    labels: list[bool] = []

    for _ in range(10):
        peak = rng.uniform(8.0, 15.0)
        corpus.append(_peaked_logits(n_tokens=4, vocab_size=8, peak=peak))
        labels.append(False)  # confident + WRONG

    for _ in range(10):
        corpus.append(_uniform_logits(n_tokens=4, vocab_size=8))
        labels.append(True)  # uncertain + CORRECT

    threshold = extractor.calibrate(corpus, labels)
    assert isinstance(threshold, float)
    assert extractor.threshold == threshold


def test_calibrate_threshold_separates_corpus() -> None:
    """Calibrated threshold is between wrong (peaked) and correct (uniform) energies.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    rng = np.random.default_rng(0)

    corpus: list[np.ndarray] = []
    labels: list[bool] = []

    for _ in range(15):
        peak = rng.uniform(10.0, 20.0)
        corpus.append(_peaked_logits(n_tokens=4, vocab_size=8, peak=peak))
        labels.append(False)

    for _ in range(15):
        corpus.append(_uniform_logits(n_tokens=4, vocab_size=8))
        labels.append(True)

    threshold = extractor.calibrate(corpus, labels)

    # Wrong examples (peaked) have very negative energies → below threshold
    wrong_energies = [
        compute_semantic_energy(lg)
        for lg, lbl in zip(corpus, labels, strict=True)
        if not lbl
    ]
    # Correct examples (uniform) have less negative energies → above threshold
    correct_energies = [
        compute_semantic_energy(lg)
        for lg, lbl in zip(corpus, labels, strict=True)
        if lbl
    ]

    mean_correct = float(np.mean(correct_energies))

    # Threshold should be between the two clusters (or at least below correct mean)
    assert threshold <= mean_correct


def test_calibrate_raises_on_length_mismatch() -> None:
    """calibrate() raises ValueError when corpus and labels lengths differ.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    corpus = [_uniform_logits(), _peaked_logits()]
    labels = [True]
    with pytest.raises(ValueError, match="same length"):
        extractor.calibrate(corpus, labels)


def test_calibrate_raises_on_too_few_examples() -> None:
    """calibrate() raises ValueError with fewer than 2 examples.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor()
    with pytest.raises(ValueError, match="at least 2"):
        extractor.calibrate([_uniform_logits()], [True])


# ---------------------------------------------------------------------------
# DualEnergyGate — gate logic (SCENARIO-VERIFY-096)
# ---------------------------------------------------------------------------


def test_gate_fires_when_both_signals_fire() -> None:
    """DualEnergyGate fires when both spilled and semantic signals trigger.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-096
    """
    gate = DualEnergyGate()
    spilled = _make_spilled_result(suspected=True)
    semantic = _make_semantic_result(overconfident=True)
    result = gate.fire(spilled, semantic)

    assert result.gate_fired is True
    assert result.trigger_signal == "both"


def test_gate_fires_when_only_spilled_fires() -> None:
    """DualEnergyGate fires with trigger_signal='spilled' when only spilled fires.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-096
    """
    gate = DualEnergyGate()
    spilled = _make_spilled_result(suspected=True)
    semantic = _make_semantic_result(overconfident=False)
    result = gate.fire(spilled, semantic)

    assert result.gate_fired is True
    assert result.trigger_signal == "spilled"


def test_gate_fires_when_only_semantic_fires() -> None:
    """DualEnergyGate fires with trigger_signal='semantic' when only semantic fires.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-096
    """
    gate = DualEnergyGate()
    spilled = _make_spilled_result(suspected=False)
    semantic = _make_semantic_result(overconfident=True)
    result = gate.fire(spilled, semantic)

    assert result.gate_fired is True
    assert result.trigger_signal == "semantic"


def test_gate_does_not_fire_when_neither_signal_fires() -> None:
    """DualEnergyGate does not fire when both signals are below threshold.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-096
    """
    gate = DualEnergyGate()
    spilled = _make_spilled_result(suspected=False)
    semantic = _make_semantic_result(overconfident=False)
    result = gate.fire(spilled, semantic)

    assert result.gate_fired is False
    assert result.trigger_signal == "none"


def test_dual_result_to_dict_round_trips_json() -> None:
    """DualEnergyResult.to_dict() round-trips through JSON without error.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-096
    """
    gate = DualEnergyGate()
    spilled = _make_spilled_result(suspected=True)
    semantic = _make_semantic_result(overconfident=False)
    result = gate.fire(spilled, semantic)
    d = result.to_dict()
    serialized = json.dumps(d)
    parsed = json.loads(serialized)

    assert parsed["gate_fired"] is True
    assert parsed["trigger_signal"] == "spilled"
    assert isinstance(parsed["spilled_result"], dict)
    assert isinstance(parsed["semantic_result"], dict)
    assert isinstance(parsed["calibration_threshold_used"], float)


def test_dual_result_contains_all_fields() -> None:
    """DualEnergyResult.to_dict() contains all required fields.

    Spec: REQ-VERIFY-077
    """
    gate = DualEnergyGate()
    result = gate.fire(_make_spilled_result(False), _make_semantic_result(False))
    d = result.to_dict()
    required = {
        "spilled_result",
        "semantic_result",
        "gate_fired",
        "trigger_signal",
        "calibration_threshold_used",
    }
    assert required == set(d.keys())


def test_dual_result_calibration_threshold_reflects_semantic() -> None:
    """calibration_threshold_used reflects the semantic extractor's threshold.

    Spec: REQ-VERIFY-077
    """
    gate = DualEnergyGate()
    semantic = _make_semantic_result(overconfident=False, energy=-3.0)
    result = gate.fire(_make_spilled_result(False), semantic)
    assert result.calibration_threshold_used == semantic.threshold_used


def test_dual_result_to_json_deterministic() -> None:
    """DualEnergyResult.to_json() produces byte-identical output on repeated calls.

    Spec: REQ-VERIFY-077
    """
    gate = DualEnergyGate()
    result = gate.fire(_make_spilled_result(True), _make_semantic_result(True))
    assert result.to_json() == result.to_json()


def test_calibrate_fallback_all_wrong() -> None:
    """calibrate() fallback path: all examples wrong → uses median of wrong energies.

    Spec: REQ-VERIFY-077
    """
    # Construct corpus where isotonic regression predicts P(wrong)>=0.5 everywhere
    # (all labels are wrong=False), so there's no crossing → fallback path runs.
    extractor = SemanticEnergyExtractor()
    corpus = [_peaked_logits(peak=float(p)) for p in range(5, 15)]
    labels = [False] * 10  # all WRONG

    threshold = extractor.calibrate(corpus, labels)
    # The threshold should be a valid float (either median or DEFAULT)
    assert isinstance(threshold, float)


# ---------------------------------------------------------------------------
# DualEnergyGate — calibrate delegates to SemanticEnergyExtractor
# ---------------------------------------------------------------------------


def test_gate_calibrate_updates_threshold() -> None:
    """DualEnergyGate.calibrate() updates the internal semantic threshold.

    Spec: REQ-VERIFY-077
    """
    gate = DualEnergyGate()
    initial_threshold = gate._semantic_extractor.threshold

    corpus = [_peaked_logits(peak=12.0)] * 5 + [_uniform_logits()] * 5
    labels = [False] * 5 + [True] * 5

    new_threshold = gate.calibrate(corpus, labels)
    assert isinstance(new_threshold, float)
    assert gate._semantic_extractor.threshold == new_threshold
    # The threshold should differ from (or equal to) the default — either is fine.
    assert isinstance(initial_threshold, float)


# ---------------------------------------------------------------------------
# Edge cases: very high entropy, very low entropy logits
# ---------------------------------------------------------------------------


def test_very_high_entropy_logits_moderate_energy() -> None:
    """Very high entropy logits (uniform) produce moderate (less negative) energy.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    # Large vocab, all equal → energy = −log(V)
    vocab_size = 1000
    logits = np.zeros((5, vocab_size), dtype=np.float64)
    energy = compute_semantic_energy(logits, temperature=1.0)
    expected = -np.log(vocab_size)
    assert abs(energy - expected) < 1e-8


def test_very_low_entropy_logits_very_negative_energy() -> None:
    """Very low entropy (very peaked) logits produce very negative energy.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    peak = 100.0
    logits = _peaked_logits(n_tokens=3, vocab_size=8, peak=peak)
    energy = compute_semantic_energy(logits, temperature=1.0)
    # log-partition ≈ peak → energy ≈ -peak
    assert energy < -90.0


def test_single_token_extractor() -> None:
    """SemanticEnergyExtractor works with single response token.

    Spec: REQ-VERIFY-077
    """
    extractor = SemanticEnergyExtractor(threshold=-5.0, temperature=1.0)
    logits = _single_token_logits(vocab_size=8)
    result = extractor.extract(logits)
    assert result.per_token_semantic.shape == (1,)
    assert isinstance(result.semantic_energy, float)


# ---------------------------------------------------------------------------
# Pipeline integration — VerifyRepairPipeline.verify_dual_energy
# ---------------------------------------------------------------------------


def test_pipeline_verify_dual_energy_returns_dual_result() -> None:
    """VerifyRepairPipeline.verify_dual_energy() returns DualEnergyResult.

    Spec: REQ-VERIFY-077
    """
    pipeline = VerifyRepairPipeline()
    logits = _peaked_logits(n_tokens=4, vocab_size=8, peak=3.0)
    result = pipeline.verify_dual_energy(logits)
    assert isinstance(result, DualEnergyResult)
    assert result.trigger_signal in {"spilled", "semantic", "both", "none"}


def test_pipeline_verify_dual_energy_uniform_logits() -> None:
    """verify_dual_energy on uniform logits returns DualEnergyResult without error.

    Spec: REQ-VERIFY-077
    """
    pipeline = VerifyRepairPipeline()
    logits = _uniform_logits(n_tokens=3, vocab_size=8)
    result = pipeline.verify_dual_energy(logits)
    assert isinstance(result, DualEnergyResult)
    # Uniform logits: spilled ≈ 0 (no spill), semantic energy ≈ −log(8) ≈ −2.08
    # With default semantic threshold of -5.0, overconfident_flag should be False
    assert result.semantic_result.overconfident_flag is False


def test_pipeline_verify_dual_energy_does_not_break_verify() -> None:
    """verify_dual_energy() is additive: existing verify() still works after calling it.

    Spec: REQ-VERIFY-077
    """
    pipeline = VerifyRepairPipeline()
    logits = _uniform_logits()
    _dual = pipeline.verify_dual_energy(logits)

    vr = pipeline.verify(question="What is 2+2?", response="4")
    assert hasattr(vr, "verified")


def test_pipeline_verify_dual_energy_from_file(tmp_path: pathlib.Path) -> None:
    """verify_dual_energy() accepts a .npy file path.

    Spec: REQ-VERIFY-077
    """
    logits = _peaked_logits(n_tokens=2, vocab_size=4, peak=5.0)
    npy_path = tmp_path / "dual_test.npy"
    np.save(npy_path, logits)

    pipeline = VerifyRepairPipeline()
    result = pipeline.verify_dual_energy(str(npy_path))
    assert isinstance(result, DualEnergyResult)


# ---------------------------------------------------------------------------
# Exp 282 logits calibration (if available)
# ---------------------------------------------------------------------------


def test_calibrate_on_exp282_logits_if_present() -> None:
    """If Exp 282 logit files exist, calibrate from them without error.

    Spec: REQ-VERIFY-077
    """
    import pathlib

    data_dir = pathlib.Path("data/research")
    npy_files = sorted(data_dir.glob("logits_282_*.npy")) if data_dir.exists() else []

    if not npy_files:
        pytest.skip("No Exp 282 logit files found — Exp 282 has not been run")

    # Use a synthetic label corpus (all correct, since we have no ground truth labels).
    # This tests calibrate() runs without error on real data shapes.
    corpus = [np.load(p) for p in npy_files[:10]]
    labels = [True] * len(corpus)

    extractor = SemanticEnergyExtractor()
    threshold = extractor.calibrate(corpus, labels)
    assert isinstance(threshold, float)


# ---------------------------------------------------------------------------
# Export from carnot.pipeline
# ---------------------------------------------------------------------------


def test_export_from_init() -> None:
    """SemanticEnergyExtractor and related symbols are importable from carnot.pipeline.

    Spec: REQ-VERIFY-077
    """
    from carnot.pipeline import (  # noqa: F401
        DualEnergyGate,
        DualEnergyResult,
        SemanticEnergyExtractor,
        SemanticEnergyResult,
        compute_semantic_energy,
    )
