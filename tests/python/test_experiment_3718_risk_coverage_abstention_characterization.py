"""Tests for Exp 3718 FoVer risk-coverage abstention characterization.

Spec: REQ-SPOE-3718, SCENARIO-SPOE-3718.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.pipeline import risk_coverage_abstention_3718 as exp


def _examples(outcome: str, *, n: int = 120) -> list[exp.AbstentionExample]:
    examples: list[exp.AbstentionExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        if outcome == "energy_better_abstention_signal":
            energy = 0.95 - 0.0001 * idx if label else 0.05 + 0.0001 * (idx - n // 2)
            baseline = 0.30 + 0.0001 * idx
        elif outcome == "energy_ties_or_loses_to_entropy":
            energy = 0.30 + 0.0001 * idx
            baseline = 0.95 - 0.0001 * idx if label else 0.05 + 0.0001 * (idx - n // 2)
        else:  # pragma: no cover - guarded by parametrization choices.
            raise ValueError(outcome)
        examples.append(
            exp.AbstentionExample(
                label=label,
                energy_score=energy,
                baseline_score=baseline,
                example_id=f"{outcome}-{idx}",
            )
        )
    return examples


@pytest.mark.parametrize(
    ("case_name", "examples", "expected_verdict", "expected_beats"),
    [
        (
            "energy_better_abstention_signal",
            _examples("energy_better_abstention_signal"),
            "complete: energy_is_a_better_selective_prediction_signal_than_entropy_deployable_abstention_gate",
            True,
        ),
        (
            "energy_ties_or_loses_to_entropy",
            _examples("energy_ties_or_loses_to_entropy"),
            "complete: energy_ties_or_loses_to_entropy_as_abstention_signal_honest_negative",
            False,
        ),
        (
            "blocked",
            [],
            "complete: blocked_fover_perstep_scores_unavailable",
            False,
        ),
    ],
)
def test_scenario_spoe_3718_parametrized_honest_outcomes(
    case_name: str,
    examples: list[exp.AbstentionExample],
    expected_verdict: str,
    expected_beats: bool,
) -> None:
    """SCENARIO-SPOE-3718: positive, honest-negative, and blocked outcomes are distinct."""

    artifact = exp.build_artifact_from_examples(
        examples,
        started_s=1.0,
        now_s=2.5,
        seeds=[11, 12, 13, 14, 15],
        n_bootstrap=16,
        tests_run=[f"SCENARIO-SPOE-3718 {case_name}"],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["energy_beats_baseline_abstention"] is expected_beats
    assert type(artifact["energy_beats_baseline_abstention"]) is bool
    assert artifact["n_seeds"] == 5
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["tests_run"] == [f"SCENARIO-SPOE-3718 {case_name}"]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_req_spoe_3718_risk_coverage_helpers_are_distinct() -> None:
    """REQ-SPOE-3718: AURC, AUROC, and risk@coverage are separate metrics."""

    rows = _examples("energy_better_abstention_signal", n=80)
    labels = [row.label for row in rows]
    energy = [row.energy_score for row in rows]
    baseline = [row.baseline_score for row in rows]

    energy_summary = exp.risk_coverage_summary(labels, energy, fixed_coverages=[0.5, 0.9])
    baseline_summary = exp.risk_coverage_summary(labels, baseline, fixed_coverages=[0.5, 0.9])
    comparison = exp.compare_abstention_signals(
        labels,
        energy,
        baseline,
        seeds=[3, 4, 5, 6, 7],
        n_bootstrap=8,
        fixed_coverages=[0.5, 0.9],
    )

    assert energy_summary["aurc"] < baseline_summary["aurc"]
    assert energy_summary["risk_at_fixed_coverage"]["0.50"] == pytest.approx(0.0)
    assert energy_summary["coverage_at_5pct_risk"] >= 0.5
    assert comparison["energy_aurc"] != comparison["energy_auroc"]
    assert comparison["energy_beats_baseline_abstention"] is True
    assert comparison["risk_at_fixed_coverage"]["0.50"]["energy"] < (
        comparison["risk_at_fixed_coverage"]["0.50"]["baseline"]
    )


def test_req_spoe_3718_validation_and_write_artifact(tmp_path: Path) -> None:
    """REQ-SPOE-3718: artifact schema, bare bool, and metric guards are strict."""

    output = exp.write_artifact_from_examples(
        tmp_path,
        output_path="results/exp3718.json",
        examples=_examples("energy_better_abstention_signal"),
        started_s=0.0,
        now_s=1.0,
        seeds=[21, 22, 23, 24, 25],
        n_bootstrap=8,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["energy_beats_baseline_abstention"] is True
    assert artifact["coverage_at_5pct_risk"] >= 0.5

    broken_bool = dict(artifact, energy_beats_baseline_abstention={"value": True})
    with pytest.raises(ValueError, match="energy_beats_baseline_abstention"):
        exp.validate_artifact(broken_bool)

    missing = dict(artifact)
    missing.pop("energy_aurc")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        exp.validate_artifact(bad_verdict)

    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        exp.validate_artifact(bad_duration)

    bad_clean = dict(artifact, adversarial_verify_clean="yes")
    with pytest.raises(ValueError, match="adversarial_verify_clean"):
        exp.validate_artifact(bad_clean)

    bad_seeds = dict(artifact, n_seeds=4)
    with pytest.raises(ValueError, match="n_seeds"):
        exp.validate_artifact(bad_seeds)

    bad_numeric = dict(artifact, energy_aurc=None)
    with pytest.raises(ValueError, match="energy_aurc"):
        exp.validate_artifact(bad_numeric)

    bad_ci = dict(artifact, energy_aurc_ci=None)
    with pytest.raises(ValueError, match="energy_aurc_ci"):
        exp.validate_artifact(bad_ci)

    tautological = dict(artifact, energy_auroc=artifact["energy_aurc"])
    with pytest.raises(ValueError, match="distinct metrics"):
        exp.validate_artifact(tautological)

    fixed_risk_tautology = dict(
        artifact,
        risk_at_fixed_coverage={
            "0.50": {"energy": artifact["energy_aurc"], "baseline": 0.4}
        },
    )
    with pytest.raises(ValueError, match="distinct metrics"):
        exp.validate_artifact(fixed_risk_tautology)

    leak_positive = dict(
        artifact,
        leak_guard={"triggered": True},
        energy_beats_baseline_abstention=True,
    )
    with pytest.raises(ValueError, match="leak guard"):
        exp.validate_artifact(leak_positive)


def test_req_spoe_3718_cached_preconditions_block_missing_fover(tmp_path: Path) -> None:
    """REQ-SPOE-3718: missing cached per-step scores block without metrics."""

    artifact = exp.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: blocked_fover_perstep_scores_unavailable"
    assert artifact["energy_beats_baseline_abstention"] is False
    assert artifact["energy_aurc"] is None
    assert artifact["baseline_aurc"] is None
    assert artifact["preconditions_checked"][0]["available"] is False


def test_req_spoe_3718_leak_guard_blocks_positive_verdict() -> None:
    """REQ-SPOE-3718: suspicious near-perfect AUROC cannot publish a positive signal."""

    artifact = exp.build_artifact_from_examples(
        _examples("energy_better_abstention_signal", n=1000),
        started_s=0.0,
        now_s=1.0,
        seeds=[31, 32, 33, 34, 35],
        n_bootstrap=4,
    )

    exp.validate_artifact(artifact)
    assert artifact["energy_auroc"] >= 0.99
    assert artifact["leak_guard"]["triggered"] is True
    assert artifact["energy_beats_baseline_abstention"] is False
    assert artifact["honest_verdict"] == (
        "complete: energy_ties_or_loses_to_entropy_as_abstention_signal_honest_negative"
    )


def test_req_spoe_3718_helper_edge_cases() -> None:
    """REQ-SPOE-3718: helper fallbacks do not fabricate selective metrics."""

    empty_curve = exp.risk_coverage_summary([], [], fixed_coverages=[0.5])
    assert empty_curve["aurc"] is None
    assert empty_curve["risk_at_fixed_coverage"] == {"0.50": None}

    assert exp.compare_abstention_signals([], [], [], seeds=[1, 2, 3, 4, 5])[
        "energy_beats_baseline_abstention"
    ] is False
    assert exp.bootstrap_metric(
        [1, 1],
        [0.8, 0.9],
        metric_fn=lambda labels, scores: 0.5,
        seeds=[1, 2, 3, 4, 5],
        n_bootstrap=2,
    )["point"] is None
    assert exp.paired_bootstrap_delta(
        [1, 1],
        [0.8, 0.9],
        [0.7, 0.6],
        metric_fn=lambda labels, scores: 0.5,
        seeds=[1, 2, 3, 4, 5],
        n_bootstrap=2,
    )["point"] is None

    no_boot = exp.bootstrap_metric(
        [0, 1],
        [0.1, 0.9],
        metric_fn=lambda labels, scores: 0.25,
        seeds=[1, 2, 3, 4, 5],
        n_bootstrap=0,
    )
    assert no_boot["ci95"] == [0.25, 0.25]

    values, seed_means = exp._bootstrap_values(
        np.asarray([0, 1], dtype=np.int64),
        [np.asarray([0.1, 0.9], dtype=np.float64)],
        metric_fn=lambda idx: None,
        seeds=[0],
        n_bootstrap=5,
    )
    assert values == []
    assert seed_means == [0.0]

    assert exp.calibration_brier_ece([exp.AbstentionExample(1, 0.8, 0.7)]) == {
        "brier": None,
        "ece": None,
        "n_holdout": 0,
    }
    assert exp._round(float("inf")) == float("inf")


def test_req_spoe_3718_adversarial_and_write_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3718: write_artifact records adversarial status and reports."""

    synthetic = exp.build_artifact_from_examples(
        _examples("energy_better_abstention_signal"),
        started_s=0.0,
        now_s=1.0,
        seeds=[41, 42, 43, 44, 45],
        n_bootstrap=4,
    )
    real_adversarial_runner = exp.run_adversarial_verify_report
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: dict(synthetic))
    monkeypatch.setattr(
        exp,
        "run_adversarial_verify_report",
        lambda path: {"flags": [{"severity": "critical", "id": "fixture"}, "skip"]},
    )

    output = exp.write_artifact(
        tmp_path,
        output_path="results/critical-exp3718.json",
        tests_run=["REQ-SPOE-3718 write_artifact"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["adversarial_verify_clean"] is False
    assert artifact["adversarial_verify_report"]["flag_count"] == 1
    assert artifact["energy_beats_baseline_abstention"] is False
    assert artifact["honest_verdict"] == (
        "complete: energy_ties_or_loses_to_entropy_as_abstention_signal_honest_negative"
    )
    assert exp.adversarial_report_is_clean({"flags": 3}) is False
    assert exp.adversarial_report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert exp.compact_adversarial_report({"flags": [{"severity": "warn"}, "skip"]}) == {
        "flag_count": 1,
        "flags": [{"severity": "warn"}],
    }

    monkeypatch.setattr(exp, "run_adversarial_verify_report", real_adversarial_runner)
    clean_output = exp.write_artifact_from_examples(
        tmp_path,
        output_path="results/clean-exp3718.json",
        examples=_examples("energy_ties_or_loses_to_entropy"),
        started_s=0.0,
        now_s=1.0,
        seeds=[51, 52, 53, 54, 55],
        n_bootstrap=4,
    )
    report = exp.run_adversarial_verify_report(clean_output)
    assert "flags" in report

    saved = exp.importlib.util.spec_from_file_location
    try:
        exp.importlib.util.spec_from_file_location = lambda *args, **kwargs: None
        with pytest.raises(ImportError, match="adversarial_verify"):
            exp.run_adversarial_verify_report(clean_output)
    finally:
        exp.importlib.util.spec_from_file_location = saved
