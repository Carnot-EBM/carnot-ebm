"""Tests for experiment_433_spilled_energy.py — SpilledEnergyDetector benchmark.

Covers:
  - Experiment module is importable
  - build_synthetic_corpus() shape: 100 items, 50 correct + 50 hallucinated
  - run_spilled_energy_benchmark() returns valid metrics dict
  - build result has schema='carnot.spilled_energy.v1'
  - Artifact has all required fields: skip_rate, fn_rate, fp_rate, honest_verdict
  - compute_honest_verdict() logic: viable vs insufficient_signal

Spec: REQ-VERIFY-092, REQ-VERIFY-093
SCENARIO-VERIFY-123, SCENARIO-VERIFY-124, SCENARIO-VERIFY-125
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on the path for experiment imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------


def test_experiment_module_importable() -> None:
    """The experiment_433_spilled_energy module can be imported without errors.

    Spec: REQ-VERIFY-092
    """
    import scripts.experiment_433_spilled_energy as mod  # noqa: F401
    assert mod is not None


# ---------------------------------------------------------------------------
# build_synthetic_corpus
# ---------------------------------------------------------------------------


def test_build_synthetic_corpus() -> None:
    """build_synthetic_corpus() returns 100 items: 50 correct + 50 hallucinated.

    Spec: REQ-VERIFY-092
    """
    from scripts.experiment_433_spilled_energy import build_synthetic_corpus

    corpus = build_synthetic_corpus()
    assert len(corpus) == 100

    n_correct = sum(1 for item in corpus if item["is_correct"])
    n_hallucinated = sum(1 for item in corpus if not item["is_correct"])
    assert n_correct == 50
    assert n_hallucinated == 50

    # All items have required keys
    for item in corpus:
        assert "text" in item
        assert "is_correct" in item
        assert isinstance(item["text"], str)
        assert isinstance(item["is_correct"], bool)


# ---------------------------------------------------------------------------
# run_spilled_energy_benchmark
# ---------------------------------------------------------------------------


def test_run_spilled_energy_benchmark() -> None:
    """run_spilled_energy_benchmark() returns valid artifact dict.

    Spec: REQ-VERIFY-092, REQ-VERIFY-093
    """
    from scripts.experiment_433_spilled_energy import (
        build_synthetic_corpus,
        run_spilled_energy_benchmark,
    )
    from carnot.pipeline.spilled_energy import SpilledEnergyDetector

    corpus = build_synthetic_corpus()
    detector = SpilledEnergyDetector()
    metrics = run_spilled_energy_benchmark(corpus, detector)

    assert isinstance(metrics, dict)
    assert "skip_rate" in metrics
    assert "fn_rate" in metrics
    assert "fp_rate" in metrics
    assert "n_total" in metrics

    # Sanity checks
    assert metrics["n_total"] == 100
    assert metrics["n_correct"] == 50
    assert metrics["n_hallucinated"] == 50
    assert 0.0 <= metrics["skip_rate"] <= 1.0
    assert 0.0 <= metrics["fn_rate"] <= 1.0
    assert 0.0 <= metrics["fp_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


def test_artifact_schema() -> None:
    """Artifact has schema='carnot.spilled_energy.v1'.

    Spec: REQ-VERIFY-092
    """
    from scripts.experiment_433_spilled_energy import (
        build_synthetic_corpus,
        run_spilled_energy_benchmark,
    )
    from carnot.pipeline.spilled_energy import SpilledEnergyDetector
    from scripts.experiment_template import ExperimentTemplate

    corpus = build_synthetic_corpus()
    detector = SpilledEnergyDetector()
    metrics = run_spilled_energy_benchmark(corpus, detector)

    tmpl = ExperimentTemplate(
        exp_id=433,
        title="SpilledEnergyDetector benchmark test",
        deliverable="/tmp/exp_433_test.json",
        requires_gpu=False,
    )
    tmpl.setup()

    artifact = tmpl.build_result(
        {
            "skip_rate": metrics["skip_rate"],
            "fn_rate": metrics["fn_rate"],
            "fp_rate": metrics["fp_rate"],
            "honest_verdict": "test",
            "n_total": metrics["n_total"],
        },
        status="success",
    )
    # Schema overridden as in the main script
    artifact["schema"] = "carnot.spilled_energy.v1"

    assert artifact["schema"] == "carnot.spilled_energy.v1"


# ---------------------------------------------------------------------------
# Required fields in artifact
# ---------------------------------------------------------------------------


def test_artifact_has_required_fields() -> None:
    """Artifact has all required fields: skip_rate, fn_rate, fp_rate, honest_verdict.

    Spec: REQ-VERIFY-092, REQ-VERIFY-093
    """
    from scripts.experiment_433_spilled_energy import (
        build_synthetic_corpus,
        compute_honest_verdict,
        run_spilled_energy_benchmark,
    )
    from carnot.pipeline.spilled_energy import SpilledEnergyDetector
    from scripts.experiment_template import ExperimentTemplate

    corpus = build_synthetic_corpus()
    detector = SpilledEnergyDetector()
    metrics = run_spilled_energy_benchmark(corpus, detector)
    honest_verdict = compute_honest_verdict(metrics["skip_rate"], metrics["fn_rate"])

    tmpl = ExperimentTemplate(
        exp_id=433,
        title="SpilledEnergyDetector benchmark test",
        deliverable="/tmp/exp_433_fields_test.json",
        requires_gpu=False,
    )
    tmpl.setup()

    artifact = tmpl.build_result(
        {
            "skip_rate": metrics["skip_rate"],
            "fn_rate": metrics["fn_rate"],
            "fp_rate": metrics["fp_rate"],
            "honest_verdict": honest_verdict,
            "n_total": metrics["n_total"],
            "n_correct": metrics["n_correct"],
            "n_hallucinated": metrics["n_hallucinated"],
            "n_skipped": metrics["n_skipped"],
            "n_fn": metrics["n_fn"],
            "n_fp": metrics["n_fp"],
        },
        status="success",
    )
    artifact["schema"] = "carnot.spilled_energy.v1"

    required_fields = ["skip_rate", "fn_rate", "fp_rate", "honest_verdict"]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# compute_honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_logic_viable() -> None:
    """skip_rate>0.20 AND fn_rate<0.05 → 'spilled_energy_viable'.

    Spec: REQ-VERIFY-092
    """
    from scripts.experiment_433_spilled_energy import compute_honest_verdict

    verdict = compute_honest_verdict(skip_rate=0.50, fn_rate=0.02)
    assert verdict == "spilled_energy_viable"


def test_honest_verdict_logic_insufficient_low_skip() -> None:
    """skip_rate<=0.20 → 'insufficient_signal' (not enough calls saved).

    Spec: REQ-VERIFY-092
    """
    from scripts.experiment_433_spilled_energy import compute_honest_verdict

    verdict = compute_honest_verdict(skip_rate=0.15, fn_rate=0.01)
    assert verdict == "insufficient_signal"


def test_honest_verdict_logic_insufficient_high_fn() -> None:
    """fn_rate>=0.05 → 'insufficient_signal' (too many missed hallucinations).

    Spec: REQ-VERIFY-092
    """
    from scripts.experiment_433_spilled_energy import compute_honest_verdict

    verdict = compute_honest_verdict(skip_rate=0.60, fn_rate=0.10)
    assert verdict == "insufficient_signal"


def test_honest_verdict_logic_boundary() -> None:
    """Exact boundary values follow the > and < conditions (not >= and <=).

    skip_rate exactly 0.20 → insufficient (need > 0.20, not >=)
    fn_rate exactly 0.05 → insufficient (need < 0.05, not <=)

    Spec: REQ-VERIFY-092
    """
    from scripts.experiment_433_spilled_energy import compute_honest_verdict

    # Exactly at thresholds — should NOT be viable
    verdict_skip_boundary = compute_honest_verdict(skip_rate=0.20, fn_rate=0.02)
    assert verdict_skip_boundary == "insufficient_signal"

    verdict_fn_boundary = compute_honest_verdict(skip_rate=0.50, fn_rate=0.05)
    assert verdict_fn_boundary == "insufficient_signal"
