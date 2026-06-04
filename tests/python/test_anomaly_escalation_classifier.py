"""Tests for the advisory anomaly-escalation classifier.

Spec refs: REQ-AUTO-015, REQ-AUTO-017, SCENARIO-AUTO-012, SCENARIO-AUTO-014.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import anomaly_escalation_classifier as classifier


SPEC_PATH = Path("openspec/capabilities/autoresearch/spec.md")


def test_req_auto_015_spec_anchor_exists() -> None:
    """REQ-AUTO-015: OpenSpec declares the advisory anomaly classifier."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-AUTO-015" in spec
    assert "SCENARIO-AUTO-012" in spec
    assert "scripts/anomaly_escalation_classifier.py" in spec
    assert "MUST NOT prune" in spec
    assert "recommend relaxing verification" in spec


def test_req_auto_017_spec_anchor_exists() -> None:
    """REQ-AUTO-017: OpenSpec declares the tuned false-escalation target."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-AUTO-017" in spec
    assert "SCENARIO-AUTO-014" in spec
    assert "false escalation" in spec
    assert "at most 0.2" in spec
    assert "P1 v1/v2" in spec


def test_scenario_auto_012_planned_bounded_negative_is_clean() -> None:
    """SCENARIO-AUTO-012: planned kill-gate negatives remain auto-reconcilable."""

    artifact = {
        "honest_verdict": "complete: thesis_a_bounded_no_improvement_at_kill_gate",
        "prior_expectation": {
            "expected_negative": True,
            "expected_negative_tokens": ["no_improvement", "bounded"],
            "kill_gate_expected": "no_improvement",
            "known_bounded_lineage": "Thesis-A-bounded",
        },
        "acceptance_gate": {
            "name": "bounded Thesis-A kill gate",
            "expected_negative": True,
            "observed_verdict_token": "no_improvement",
        },
    }

    result = classifier.classify_artifact(artifact)

    assert result.classification == "clean_bounded_negative"
    assert result.recommendation == "standard_auto_reconcile"
    assert result.verification_relaxation_recommended is False
    assert "expected negative" in result.rationale
    assert "Thesis-A-bounded" in result.rationale


def test_scenario_auto_012_load_bearing_positive_control_failure_is_anomaly() -> None:
    """SCENARIO-AUTO-012: P1 v2 INCONCLUSIVE-style control failure escalates."""

    artifact = {
        "honest_verdict": "inconclusive: p1_v2_ar_positive_control_failed_ar_best_below_0_3",
        "positive_control": {
            "name": "AR positive control",
            "load_bearing": True,
            "passed": False,
            "threshold": "ar_best >= 0.3",
            "observed": 0.0,
        },
        "prior_expectation": {
            "assumptions": ["AR positive control must pass before judging energy landscape"],
        },
    }

    result = classifier.classify_artifact(artifact)

    assert result.classification == "frame_violating_anomaly"
    assert result.recommendation == "halt_pruning_escalate_to_human"
    assert result.verification_relaxation_recommended is False
    assert "positive control" in result.rationale
    assert "pause" in result.rationale
    assert "relax" not in result.recommendation


def test_scenario_auto_012_clean_positive_is_clean_positive() -> None:
    """SCENARIO-AUTO-012: terminal positive results are not escalated."""

    artifact = {
        "honest_verdict": "complete: verifier_product_gate_passed_selection_lift_positive",
        "acceptance_gate_passed": True,
        "selection_lift": 0.08,
    }

    result = classifier.classify_artifact(artifact)

    assert result.classification == "clean_positive"
    assert result.recommendation == "standard_positive_reconcile"
    assert result.verification_relaxation_recommended is False
    assert "terminal positive" in result.rationale


def test_req_auto_015_assumption_contradiction_and_envelope_escape_escalate() -> None:
    """REQ-AUTO-015: explicit frame violations override clean verdict wording."""

    assumption_artifact = {
        "honest_verdict": "complete: no_improvement_observed",
        "assumption_contradicted": True,
        "assumption_note": "The corpus was assumed non-degenerate but headroom was zero.",
    }
    envelope_artifact = {
        "honest_verdict": "complete: metric_direction_unexpected",
        "predicted_envelope": {
            "metric": "selection_lift",
            "min": 0.01,
            "max": 0.08,
        },
        "selection_lift": -0.12,
    }

    assumption_result = classifier.classify_artifact(assumption_artifact)
    envelope_result = classifier.classify_artifact(envelope_artifact)

    assert assumption_result.classification == "frame_violating_anomaly"
    assert "contradict" in assumption_result.rationale
    assert envelope_result.classification == "frame_violating_anomaly"
    assert "predicted envelope" in envelope_result.rationale


def test_req_auto_015_cli_reads_json_and_emits_recommendation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-015: CLI emits an advisory verdict without mutating input."""

    artifact_path = tmp_path / "artifact.json"
    artifact = {
        "honest_verdict": "complete: bounded_negative_expected_no_delta",
        "expected_negative": True,
    }
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")

    rc = classifier.main([str(artifact_path)])
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output["classification"] == "clean_bounded_negative"
    assert output["recommendation"] == "standard_auto_reconcile"
    assert output["verification_relaxation_recommended"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_auto_015_alternate_frame_violation_shapes_are_detected() -> None:
    """REQ-AUTO-015: bool flags and textual assumption violations are real signals."""

    bool_control = classifier.classify_artifact(
        {
            "honest_verdict": "complete: no_improvement",
            "positive_control_passed": False,
        }
    )
    failure_control = classifier.classify_artifact(
        {
            "honest_verdict": "complete: no_improvement",
            "positive_control_failure": True,
        }
    )
    textual_assumption = classifier.classify_artifact(
        {
            "honest_verdict": "complete: no_improvement",
            "assumption_note": "The non-degeneracy assumption was violated.",
        }
    )

    assert bool_control.classification == "frame_violating_anomaly"
    assert failure_control.classification == "frame_violating_anomaly"
    assert textual_assumption.classification == "frame_violating_anomaly"


def test_req_auto_015_expected_negative_metadata_variants_are_clean() -> None:
    """REQ-AUTO-015: token-only and lineage-only expected negatives are bounded."""

    token_only = classifier.classify_artifact(
        {
            "honest_verdict": "complete: no_delta_at_declared_gate",
            "prior_expectation": {"expected_negative_tokens": "no_delta"},
        }
    )
    lineage_only = classifier.classify_artifact(
        {
            "honest_verdict": "complete: bounded_negative_confirmed",
            "prior_expectation": {"lineage": "Route retired as bounded"},
        }
    )

    assert token_only.classification == "clean_bounded_negative"
    assert "declared expected negative" in token_only.rationale
    assert lineage_only.classification == "clean_bounded_negative"
    assert "known bounded lineage" in lineage_only.rationale


def test_req_auto_015_unexpected_negative_and_neutral_artifact_defaults() -> None:
    """REQ-AUTO-015: unplanned negatives escalate; neutral non-negatives do not."""

    unexpected_negative = classifier.classify_artifact(
        {"honest_verdict": "complete: no_improvement_without_prior_gate"}
    )
    neutral = classifier.classify_artifact({"honest_verdict": "queued_for_operator_note"})

    assert unexpected_negative.classification == "frame_violating_anomaly"
    assert "lacks expected kill-gate" in unexpected_negative.rationale
    assert neutral.classification == "clean_positive"
    assert "non-negative artifact" in neutral.rationale


def test_scenario_auto_014_real_earned_negatives_are_not_false_escalated() -> None:
    """SCENARIO-AUTO-014: exp3791 false escalations become bounded negatives."""

    matched_compute = classifier.classify_file(Path("results/thesis_a_part_b_matched_compute.json"))
    part_b_not_run = classifier.classify_file(
        Path("results/experiment_3739_kill_gate_part_b_verdict.json")
    )

    assert matched_compute.classification == "clean_bounded_negative"
    assert matched_compute.recommendation == "standard_auto_reconcile"
    assert matched_compute.verification_relaxation_recommended is False
    assert "bounded-negative verdict text" in matched_compute.rationale

    assert part_b_not_run.classification == "clean_bounded_negative"
    assert part_b_not_run.recommendation == "standard_auto_reconcile"
    assert part_b_not_run.verification_relaxation_recommended is False
    assert "bounded-negative verdict text" in part_b_not_run.rationale


def test_scenario_auto_014_p1_positive_control_failures_still_escalate() -> None:
    """SCENARIO-AUTO-014: P1 v1/v2 positive controls preserve anomaly recall."""

    p1_v1 = classifier.classify_file(Path("results/thesis_a_p1_discrete_search.json"))
    p1_v2 = classifier.classify_file(Path("results/thesis_a_p1_discrete_search_v2.json"))

    assert p1_v1.classification == "frame_violating_anomaly"
    assert p1_v1.recommendation == "halt_pruning_escalate_to_human"
    assert p1_v1.verification_relaxation_recommended is False
    assert "positive control failure" in p1_v1.rationale

    assert p1_v2.classification == "frame_violating_anomaly"
    assert p1_v2.recommendation == "halt_pruning_escalate_to_human"
    assert p1_v2.verification_relaxation_recommended is False
    assert "positive control failure" in p1_v2.rationale


def test_req_auto_015_envelope_non_anomalies_and_bad_shapes_are_tolerated() -> None:
    """REQ-AUTO-015: malformed or in-range envelopes do not poison classification."""

    in_range = classifier.classify_artifact(
        {
            "honest_verdict": "complete: metric_checked",
            "predicted_envelope": {"metric": "selection_lift", "min": 0.01, "max": 0.08},
            "selection_lift": 0.05,
        }
    )
    bad_metric = classifier.classify_artifact(
        {
            "honest_verdict": "complete: metric_checked",
            "predicted_envelope": {"metric": 7, "min": 0.01, "max": 0.08},
            "selection_lift": 0.05,
        }
    )
    bad_numbers = classifier.classify_artifact(
        {
            "honest_verdict": "complete: metric_checked",
            "predicted_envelope": {"metric": "selection_lift", "min": "low", "max": 0.08},
            "selection_lift": True,
        }
    )

    assert in_range.classification == "clean_positive"
    assert bad_metric.classification == "clean_positive"
    assert bad_numbers.classification == "clean_positive"


def test_req_auto_015_classify_file_rejects_non_object_json(tmp_path: Path) -> None:
    """REQ-AUTO-015: artifact files must be JSON objects, not arbitrary JSON."""

    artifact_path = tmp_path / "not_object.json"
    artifact_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object artifact"):
        classifier.classify_file(artifact_path)
