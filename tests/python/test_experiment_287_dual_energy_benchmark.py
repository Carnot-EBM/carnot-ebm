"""Tests for experiment_287_dual_energy_benchmark.py — dual-energy gate on Apple corpus.

**Researcher summary:**
    Verifies AUROC computation, threshold sweep monotonicity, per-variant-type breakdown
    of signal attribution, and gap-filler coverage analysis.  All tests use small synthetic
    logits so they run fast on CPU without live inference.

Spec: REQ-VERIFY-078
SCENARIO-VERIFY-097 (AUROC computation correctness)
SCENARIO-VERIFY-098 (threshold sweep monotonicity)
SCENARIO-VERIFY-099 (per-variant-type signal attribution)
SCENARIO-VERIFY-100 (gap-filler coverage analysis)
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers — allow running before script is on sys.path
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "experiment_287_dual_energy_benchmark.py"
)

_MODULE_NAME = "experiment_287_dual_energy_benchmark"


def _import_script():
    """Import the benchmark script module."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Load once at module level so import errors are reported clearly.
_mod = _import_script()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_logits(n_tokens: int = 10, vocab_size: int = 20) -> np.ndarray:
    """Uniform logit array — near-zero spilled energy."""
    return np.zeros((n_tokens, vocab_size), dtype=np.float64)


def _peaked_logits(
    n_tokens: int = 10, vocab_size: int = 20, peak: float = 8.0
) -> np.ndarray:
    """Peaked logit array — token 0 dominates, low spilled energy."""
    arr = np.zeros((n_tokens, vocab_size), dtype=np.float64)
    arr[:, 0] = peak
    return arr


def _confused_logits(
    n_tokens: int = 10, vocab_size: int = 20, n_peaks: int = 5, peak: float = 3.0
) -> np.ndarray:
    """Multi-peaked logits — several tokens share moderate probability → higher spill."""
    arr = np.zeros((n_tokens, vocab_size), dtype=np.float64)
    for i in range(n_peaks):
        arr[:, i] = peak
    return arr


# ---------------------------------------------------------------------------
# compute_auroc — SCENARIO-VERIFY-097
# ---------------------------------------------------------------------------


def test_compute_auroc_perfect_separation() -> None:
    """All errors scored above all non-errors → AUROC = 1.0.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    # REQ-VERIFY-078: AUROC must be computed via sklearn.metrics.roc_auc_score.
    scores = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]  # First 3 are errors, last 3 are not.
    labels = [True, True, True, False, False, False]
    auroc = _mod.compute_auroc(scores, labels)
    assert abs(auroc - 1.0) < 1e-6


def test_compute_auroc_reverse_separation() -> None:
    """All errors scored BELOW all non-errors → AUROC = 0.0.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    scores = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]  # First 3 are errors (low score).
    labels = [True, True, True, False, False, False]
    auroc = _mod.compute_auroc(scores, labels)
    assert abs(auroc - 0.0) < 1e-6


def test_compute_auroc_random_equal_scores() -> None:
    """All scores identical → AUROC = 0.5.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    scores = [1.0] * 10
    labels = [True] * 5 + [False] * 5
    auroc = _mod.compute_auroc(scores, labels)
    assert abs(auroc - 0.5) < 1e-6


def test_compute_auroc_returns_float() -> None:
    """compute_auroc returns a Python float in [0.0, 1.0].

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    scores = [0.5, 0.8, 0.2, 0.9, 0.1, 0.6]
    labels = [False, True, False, True, False, True]
    result = _mod.compute_auroc(scores, labels)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# threshold_sweep — SCENARIO-VERIFY-098
# ---------------------------------------------------------------------------


def test_threshold_sweep_returns_correct_length() -> None:
    """threshold_sweep returns exactly N_THRESHOLD_STEPS records.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-098
    """
    scores = list(np.linspace(0.0, 2.0, 50))
    labels = [True] * 25 + [False] * 25
    n_steps = 11
    sweep = _mod.threshold_sweep(scores, labels, n_steps)
    assert len(sweep) == n_steps


def test_threshold_sweep_record_keys() -> None:
    """Every sweep record has threshold, precision, recall, and f1.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-098
    """
    scores = [1.0, 2.0, 0.5, 1.5, 0.8, 0.3]
    labels = [True, True, True, False, False, False]
    sweep = _mod.threshold_sweep(scores, labels, n_steps=5)
    required = {"threshold", "precision", "recall", "f1"}
    for record in sweep:
        assert required == set(record.keys()), f"Missing keys: {required - set(record.keys())}"


def test_threshold_sweep_recall_non_increasing() -> None:
    """Recall is non-increasing as threshold increases.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-098
    """
    rng = np.random.default_rng(42)
    # Errors get higher scores than non-errors on average.
    error_scores = (rng.normal(2.0, 0.3, 30)).tolist()
    nonerror_scores = (rng.normal(0.5, 0.3, 30)).tolist()
    scores = error_scores + nonerror_scores
    labels = [True] * 30 + [False] * 30
    sweep = _mod.threshold_sweep(scores, labels, n_steps=21)
    recalls = [r["recall"] for r in sweep]
    for i in range(len(recalls) - 1):
        # Allow small floating-point tolerance.
        assert recalls[i] >= recalls[i + 1] - 1e-9, (
            f"Recall increased at step {i}: {recalls[i]:.4f} → {recalls[i+1]:.4f}"
        )


def test_threshold_sweep_precision_non_decreasing() -> None:
    """Precision is non-decreasing as threshold increases.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-098
    """
    rng = np.random.default_rng(99)
    error_scores = (rng.normal(2.0, 0.3, 30)).tolist()
    nonerror_scores = (rng.normal(0.5, 0.3, 30)).tolist()
    scores = error_scores + nonerror_scores
    labels = [True] * 30 + [False] * 30
    sweep = _mod.threshold_sweep(scores, labels, n_steps=21)
    precisions = [r["precision"] for r in sweep]
    for i in range(len(precisions) - 1):
        # Precision is non-decreasing when threshold is stricter.
        assert precisions[i] <= precisions[i + 1] + 1e-9, (
            f"Precision decreased at step {i}: {precisions[i]:.4f} → {precisions[i+1]:.4f}"
        )


def test_threshold_sweep_at_target_precision() -> None:
    """At least one sweep step achieves precision ≥ 0.65 when data is separable.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-098
    """
    # Well-separated: error scores clearly above non-error scores.
    scores = [2.5, 2.3, 2.1, 1.9, 1.7, 0.5, 0.4, 0.3, 0.2, 0.1]
    labels = [True] * 5 + [False] * 5
    sweep = _mod.threshold_sweep(scores, labels, n_steps=51)
    max_precision = max(r["precision"] for r in sweep)
    assert max_precision >= 0.65


# ---------------------------------------------------------------------------
# generate_synthetic_logits — internal helper used by benchmark
# ---------------------------------------------------------------------------


def test_generate_synthetic_logits_shape() -> None:
    """generate_synthetic_logits returns (N_TOKENS, VOCAB_SIZE) array.

    Spec: REQ-VERIFY-078
    """
    rng = np.random.default_rng(0)
    logits = _mod.generate_synthetic_logits("correct", "number_swap", rng)
    assert logits.ndim == 2
    assert logits.shape == (_mod.N_TOKENS, _mod.VOCAB_SIZE)


def test_generate_synthetic_logits_correct_low_spill() -> None:
    """Correct-class logits produce mean_spilled below spilled threshold.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    from carnot.pipeline.spilled_energy_extractor import compute_spilled_energy

    rng = np.random.default_rng(1)
    # Average over several draws to smooth noise.
    spills = []
    for _ in range(20):
        logits = _mod.generate_synthetic_logits("correct", "number_swap", rng)
        res = compute_spilled_energy(logits)
        spills.append(res.mean_spilled)
    assert np.mean(spills) < _mod.SPILLED_THRESHOLD, (
        f"Correct logits have mean spill {np.mean(spills):.3f} ≥ threshold {_mod.SPILLED_THRESHOLD}"
    )


def test_generate_synthetic_logits_wrong_ns_low_spill() -> None:
    """number_swap wrong logits produce very low mean_spilled (overconfident, not confused).

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    from carnot.pipeline.spilled_energy_extractor import compute_spilled_energy

    rng = np.random.default_rng(2)
    spills = []
    for _ in range(20):
        logits = _mod.generate_synthetic_logits("wrong", "number_swap", rng)
        res = compute_spilled_energy(logits)
        spills.append(res.mean_spilled)
    assert np.mean(spills) < _mod.SPILLED_THRESHOLD, (
        f"number_swap wrong should have low spill but got {np.mean(spills):.3f}"
    )


def test_generate_synthetic_logits_wrong_is_high_spill() -> None:
    """irrelevant_sentence wrong logits produce mean_spilled above spilled threshold.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    from carnot.pipeline.spilled_energy_extractor import compute_spilled_energy

    rng = np.random.default_rng(3)
    spills = []
    for _ in range(20):
        logits = _mod.generate_synthetic_logits("wrong", "irrelevant_sentence", rng)
        res = compute_spilled_energy(logits)
        spills.append(res.mean_spilled)
    assert np.mean(spills) > _mod.SPILLED_THRESHOLD, (
        f"irrelevant_sentence wrong should have high spill but got {np.mean(spills):.3f}"
    )


# ---------------------------------------------------------------------------
# per_variant_breakdown — SCENARIO-VERIFY-099
# ---------------------------------------------------------------------------


def test_per_variant_breakdown_keys() -> None:
    """per_variant_breakdown result contains 'number_swap' and 'irrelevant_sentence' keys.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    # Build minimal benchmark rows.
    rows = [
        {
            "variant_type": "number_swap",
            "trigger_signal": "semantic",
            "gate_fired": True,
            "is_error": True,
        },
        {
            "variant_type": "irrelevant_sentence",
            "trigger_signal": "spilled",
            "gate_fired": True,
            "is_error": True,
        },
        {
            "variant_type": "number_swap",
            "trigger_signal": "none",
            "gate_fired": False,
            "is_error": False,
        },
    ]
    breakdown = _mod.per_variant_breakdown(rows)
    assert "number_swap" in breakdown
    assert "irrelevant_sentence" in breakdown


def test_per_variant_breakdown_signal_fractions() -> None:
    """number_swap errors show higher semantic fraction than irrelevant_sentence errors.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    rows = []
    # 10 number_swap errors: semantic fires
    for _ in range(10):
        rows.append({"variant_type": "number_swap", "trigger_signal": "semantic",
                     "gate_fired": True, "is_error": True})
    # 10 irrelevant_sentence errors: spilled fires
    for _ in range(10):
        rows.append({"variant_type": "irrelevant_sentence", "trigger_signal": "spilled",
                     "gate_fired": True, "is_error": True})
    breakdown = _mod.per_variant_breakdown(rows)
    ns = breakdown["number_swap"]
    is_ = breakdown["irrelevant_sentence"]
    # number_swap should have higher semantic fraction.
    assert ns["signal_fractions"]["semantic"] > is_["signal_fractions"].get("semantic", 0.0)
    # irrelevant_sentence should have higher spilled fraction.
    assert is_["signal_fractions"]["spilled"] > ns["signal_fractions"].get("spilled", 0.0)


def test_per_variant_breakdown_counts() -> None:
    """per_variant_breakdown records correct total and fired counts.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    rows = [
        {"variant_type": "number_swap", "trigger_signal": "semantic",
         "gate_fired": True, "is_error": True},
        {"variant_type": "number_swap", "trigger_signal": "none",
         "gate_fired": False, "is_error": False},
        {"variant_type": "number_swap", "trigger_signal": "both",
         "gate_fired": True, "is_error": True},
    ]
    breakdown = _mod.per_variant_breakdown(rows)
    ns = breakdown["number_swap"]
    assert ns["total"] == 3
    assert ns["gate_fired_count"] == 2


# ---------------------------------------------------------------------------
# signal_attribution_counts — SCENARIO-VERIFY-099
# ---------------------------------------------------------------------------


def test_signal_attribution_sums_to_total() -> None:
    """signal attribution counts sum to total number of rows.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    rows = [
        {"trigger_signal": "spilled", "gate_fired": True},
        {"trigger_signal": "semantic", "gate_fired": True},
        {"trigger_signal": "both", "gate_fired": True},
        {"trigger_signal": "none", "gate_fired": False},
        {"trigger_signal": "none", "gate_fired": False},
    ]
    counts = _mod.signal_attribution_counts(rows)
    total = counts["spilled"] + counts["semantic"] + counts["both"] + counts["none"]
    assert total == len(rows)


def test_signal_attribution_correct_values() -> None:
    """signal_attribution_counts returns correct per-signal tallies.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    rows = [
        {"trigger_signal": "spilled", "gate_fired": True},
        {"trigger_signal": "spilled", "gate_fired": True},
        {"trigger_signal": "semantic", "gate_fired": True},
        {"trigger_signal": "both", "gate_fired": True},
        {"trigger_signal": "none", "gate_fired": False},
    ]
    counts = _mod.signal_attribution_counts(rows)
    assert counts["spilled"] == 2
    assert counts["semantic"] == 1
    assert counts["both"] == 1
    assert counts["none"] == 1


# ---------------------------------------------------------------------------
# gap_filler_coverage_analysis — SCENARIO-VERIFY-100
# ---------------------------------------------------------------------------


def test_gap_filler_returns_fraction_in_range() -> None:
    """gap_filler_coverage_analysis returns a fraction in [0.0, 1.0].

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-100
    """
    sweep = [
        {"threshold": 0.5, "precision": 0.70, "recall": 0.80, "f1": 0.75},
        {"threshold": 1.0, "precision": 0.80, "recall": 0.60, "f1": 0.69},
        {"threshold": 1.5, "precision": 0.90, "recall": 0.40, "f1": 0.55},
    ]
    result = _mod.gap_filler_coverage_analysis(
        sweep,
        n_not_formalizable=1302,
        target_precision=0.65,
    )
    assert 0.0 <= result["coverage_at_target_precision"] <= 1.0


def test_gap_filler_n_not_formalizable() -> None:
    """gap_filler_coverage_analysis records n_not_formalizable = 1302.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-100
    """
    sweep = [{"threshold": 0.5, "precision": 0.70, "recall": 0.80, "f1": 0.75}]
    result = _mod.gap_filler_coverage_analysis(
        sweep, n_not_formalizable=1302, target_precision=0.65
    )
    assert result["n_not_formalizable"] == 1302


def test_gap_filler_uses_recall_at_target_precision() -> None:
    """Coverage equals the recall at the threshold achieving target precision.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-100
    """
    # Sweep: only step with precision=0.70 (≥ 0.65), recall=0.55.
    sweep = [
        {"threshold": 1.0, "precision": 0.50, "recall": 0.90, "f1": 0.64},
        {"threshold": 2.0, "precision": 0.70, "recall": 0.55, "f1": 0.61},
        {"threshold": 3.0, "precision": 0.85, "recall": 0.30, "f1": 0.44},
    ]
    result = _mod.gap_filler_coverage_analysis(
        sweep, n_not_formalizable=1302, target_precision=0.65
    )
    # First step meeting ≥ 0.65 precision has recall=0.55 → coverage=0.55.
    assert abs(result["coverage_at_target_precision"] - 0.55) < 1e-6


def test_gap_filler_no_threshold_meets_precision() -> None:
    """Coverage is 0.0 when no sweep step reaches target precision.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-100
    """
    sweep = [{"threshold": 0.5, "precision": 0.40, "recall": 0.90, "f1": 0.55}]
    result = _mod.gap_filler_coverage_analysis(
        sweep, n_not_formalizable=1302, target_precision=0.65
    )
    assert result["coverage_at_target_precision"] == 0.0


# ---------------------------------------------------------------------------
# Full benchmark integration — run on small synthetic corpus
# ---------------------------------------------------------------------------


def test_run_benchmark_returns_expected_keys() -> None:
    """run_benchmark returns a dict with primary_criterion_met and metric keys.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    # Use a tiny synthetic run to keep the test fast (no disk I/O needed).
    result = _mod.run_benchmark(n_benchmark=20, n_calibration=20, seed=287)
    required = {
        "experiment",
        "run_date",
        "synthetic_logits",
        "primary_criterion_met",
        "primary_criterion_threshold",
        "auroc_spilled",
        "auroc_semantic",
        "auroc_combined",
        "n_benchmark",
        "adversarial_accuracy_simulated",
        "variant_breakdown",
        "signal_attribution",
        "threshold_sweep",
        "gap_filler",
    }
    assert required.issubset(set(result.keys())), (
        f"Missing keys: {required - set(result.keys())}"
    )


def test_run_benchmark_primary_criterion_met() -> None:
    """run_benchmark primary criterion (combined AUROC ≥ 0.65) is satisfied on synthetic data.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    result = _mod.run_benchmark(n_benchmark=200, n_calibration=100, seed=287)
    assert result["primary_criterion_met"] is True, (
        f"Primary criterion FAILED: combined AUROC = {result['auroc_combined']:.4f}"
    )
    assert result["auroc_combined"] >= 0.65, (
        f"Combined AUROC {result['auroc_combined']:.4f} < 0.65"
    )


def test_run_benchmark_combined_auroc_exceeds_individuals() -> None:
    """Combined AUROC exceeds both individual signal AUROCs on adversarial corpus.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-097
    """
    result = _mod.run_benchmark(n_benchmark=200, n_calibration=100, seed=287)
    assert result["auroc_combined"] >= result["auroc_spilled"] - 1e-6, (
        "Combined AUROC should be >= spilled AUROC"
    )
    assert result["auroc_combined"] >= result["auroc_semantic"] - 1e-6, (
        "Combined AUROC should be >= semantic AUROC"
    )


def test_run_benchmark_variant_breakdown_has_both_types() -> None:
    """Benchmark variant breakdown includes both number_swap and irrelevant_sentence.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    result = _mod.run_benchmark(n_benchmark=40, n_calibration=20, seed=287)
    assert "number_swap" in result["variant_breakdown"]
    assert "irrelevant_sentence" in result["variant_breakdown"]


def test_run_benchmark_signal_attribution_sums_correctly() -> None:
    """Signal attribution counts sum to the total benchmark size.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-099
    """
    result = _mod.run_benchmark(n_benchmark=40, n_calibration=20, seed=287)
    attr = result["signal_attribution"]
    total = attr["spilled"] + attr["semantic"] + attr["both"] + attr["none"]
    assert total == result["n_benchmark"]


def test_run_benchmark_gap_filler_n_not_formalizable() -> None:
    """Benchmark gap_filler always records n_not_formalizable = 1302.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-100
    """
    result = _mod.run_benchmark(n_benchmark=40, n_calibration=20, seed=287)
    assert result["gap_filler"]["n_not_formalizable"] == 1302


def test_run_benchmark_artifact_json_round_trip() -> None:
    """run_benchmark result round-trips through json.dumps/loads without error.

    Spec: REQ-VERIFY-078
    """
    result = _mod.run_benchmark(n_benchmark=20, n_calibration=20, seed=287)
    serialized = json.dumps(result)
    parsed = json.loads(serialized)
    assert parsed["experiment"] == 287
    assert isinstance(parsed["primary_criterion_met"], bool)


def test_run_benchmark_synthetic_logits_flag() -> None:
    """run_benchmark sets synthetic_logits=True when no real logits are present.

    Spec: REQ-VERIFY-078
    """
    result = _mod.run_benchmark(n_benchmark=20, n_calibration=20, seed=287)
    assert result["synthetic_logits"] is True


def test_run_benchmark_threshold_sweep_length() -> None:
    """Threshold sweep has exactly N_THRESHOLD_STEPS entries.

    Spec: REQ-VERIFY-078, SCENARIO-VERIFY-098
    """
    result = _mod.run_benchmark(n_benchmark=40, n_calibration=20, seed=287)
    assert len(result["threshold_sweep"]) == _mod.N_THRESHOLD_STEPS
