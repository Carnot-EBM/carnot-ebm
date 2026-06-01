"""Tests for Exp 3660 FR-11 continuous self-learning v9.

Spec: REQ-LEARN-3660, SCENARIO-LEARN-3660.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.fr11 import continuous_self_learning_v9 as exp3660
from carnot.fr11.continuous_self_learning_v9 import (
    BLOCKED_VERDICT,
    NO_GAIN_VERDICT,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT,
    build_artifact,
    build_artifact_from_examples,
    select_honest_verdict,
    validate_artifact,
    write_artifact,
)
from carnot.pipeline.second_pair_detector import LabeledDetectorExample


def _fusion_fixture(outcome: str) -> list[LabeledDetectorExample]:
    examples: list[LabeledDetectorExample] = []
    if outcome == "blocked":
        return examples
    domains = ("math_reasoning", "code_validation")
    for domain_index, domain in enumerate(domains):
        for idx in range(120):
            label = 1 if idx < 30 else 0
            if outcome == "holds_no_collapse":
                ensemble = 0.90 - 0.001 * idx if label else 0.10 + 0.001 * (idx - 30)
                confidence = 0.10 + 0.001 * idx if label else 0.90 - 0.001 * (idx - 30)
            elif outcome == "no_gain":
                ensemble = 0.90 - 0.001 * idx if label else 0.10 + 0.001 * (idx - 30)
                confidence = ensemble
            else:  # pragma: no cover - parametrization guards valid outcomes.
                raise ValueError(outcome)
            examples.append(
                LabeledDetectorExample(
                    domain=domain,
                    label=label,
                    ensemble_energy=ensemble,
                    confidence_error=confidence,
                    example_id=f"{outcome}-{domain_index}-{idx}",
                )
            )
    return examples


@pytest.mark.parametrize(
    ("outcome", "expected_verdict"),
    [
        ("holds_no_collapse", SUCCESS_VERDICT),
        ("no_gain", NO_GAIN_VERDICT),
        ("blocked", BLOCKED_VERDICT),
    ],
)
def test_req_learn_3660_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-LEARN-3660: honest outcomes are classified from fixtures."""

    artifact = build_artifact_from_examples(
        _fusion_fixture(outcome),
        started_s=1.0,
        now_s=3.0,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3660.INFERENCE_SUBSTRATE
    if outcome == "blocked":
        assert artifact["n_online_updates"] == 0
        assert artifact["acceptance_gate"]["passed"] is False
    else:
        assert artifact["n_online_updates"] == 240
        assert artifact["collapse_detected_deploy_arm"] is False
        assert artifact["collapse_detected_control"] is True
        assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
        assert artifact["quality_maintained"] is True
        assert min(artifact["fusion_weight_deploy_final_by_domain"].values()) >= 0.2
        assert max(artifact["fusion_weight_deploy_final_by_domain"].values()) <= 0.8
        assert max(artifact["fusion_weight_control_final_by_domain"].values()) == 1.0
    if outcome == "holds_no_collapse":
        assert artifact["online_fusion_auroc_gain"] > 0.0
        assert artifact["calibration_improved"] is True
    if outcome == "no_gain":
        assert artifact["online_fusion_auroc_gain"] == 0.0


def test_req_learn_3660_build_artifact_preconditions_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3660-1/2: cached traces are required and JSON is written."""

    blocked = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_VERDICT
    assert blocked["preconditions_checked"][0]["resource"] == "fr11_module"

    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3660,
        "load_cached_labeled_examples",
        lambda root: ([], {"synthetic": {"status": "blocked"}}),
    )
    no_traces = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(no_traces)
    assert no_traces["honest_verdict"] == BLOCKED_VERDICT
    assert no_traces["preconditions_checked"][1]["available"] is False

    def _raise_load(root: Path) -> tuple[list[LabeledDetectorExample], dict[str, object]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3660, "load_cached_labeled_examples", _raise_load)
    failed_load = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(failed_load)
    assert failed_load["preconditions_checked"][1]["resource"] == (
        "cached_traces_with_ensemble_and_confidence"
    )

    monkeypatch.setattr(
        exp3660,
        "load_cached_labeled_examples",
        lambda root: (_fusion_fixture("holds_no_collapse"), {"synthetic": {"status": "loaded"}}),
    )
    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(artifact)
    assert artifact["honest_verdict"] in {SUCCESS_VERDICT, NO_GAIN_VERDICT}
    assert artifact["preconditions_checked"][1]["available"] is True

    output = write_artifact(
        tmp_path,
        output_path="results/experiment_3660_fixture.json",
        examples=_fusion_fixture("no_gain"),
        started_s=0.0,
        now_s=1.0,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"] == NO_GAIN_VERDICT

    blocked_output = write_artifact(
        tmp_path / "missing-fr11-root",
        output_path="results/experiment_3660_blocked.json",
        started_s=0.0,
        now_s=1.0,
    )
    blocked_payload = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked_payload["honest_verdict"] == BLOCKED_VERDICT


def test_req_learn_3660_validation_and_verdict_edges() -> None:
    """REQ-LEARN-3660-4/6: schema and terminal verdict selection are strict."""

    assert select_honest_verdict(gate_passed=True, online_fusion_auroc_gain=0.01) == SUCCESS_VERDICT
    assert select_honest_verdict(gate_passed=True, online_fusion_auroc_gain=0.0) == NO_GAIN_VERDICT
    assert select_honest_verdict(gate_passed=False, online_fusion_auroc_gain=1.0) == NO_GAIN_VERDICT

    artifact = build_artifact_from_examples(
        _fusion_fixture("holds_no_collapse"),
        started_s=0.0,
        now_s=1.0,
    )
    validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unsupported")
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(bad_verdict)

    no_principles = dict(artifact)
    no_principles.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(no_principles)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(missing_principle)

    bad_gate = dict(artifact, acceptance_gate={"passed": "yes"})
    with pytest.raises(ValueError, match="acceptance_gate"):
        validate_artifact(bad_gate)

    bad_duration = dict(artifact, duration_s="fast")
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)

    bad_n = dict(artifact, n_online_updates=199)
    with pytest.raises(ValueError, match="at least"):
        validate_artifact(bad_n)

    bad_bool = dict(artifact, collapse_detected_control="true")
    with pytest.raises(ValueError, match="collapse_detected_control"):
        validate_artifact(bad_bool)

    bad_gain = dict(artifact, online_fusion_auroc_gain=float("nan"))
    with pytest.raises(ValueError, match="online_fusion_auroc_gain"):
        validate_artifact(bad_gain)

    with pytest.raises(ValueError, match="same length"):
        exp3660.fusion_scores([0.5], [0.1, 0.2], alpha=0.5)
    with pytest.raises(ValueError, match="same length"):
        exp3660.calibrate_scores_by_domain(_fusion_fixture("no_gain")[:1], [0.5, 0.6])
    with pytest.raises(ValueError, match="same length"):
        exp3660.online_metric_trajectories([0], [0.2, 0.8])

    pass_rate, true_accuracy = exp3660.online_metric_trajectories([0, 1], [0.2, 0.8], n_windows=2)
    assert pass_rate != true_accuracy
    assert exp3660.detect_collapse({}, require_all=True) is False

    blocked_nonfinite = build_artifact_from_examples(
        [LabeledDetectorExample("nonfinite", 1, float("nan"), 0.5, "nonfinite-case")],
        started_s=0.0,
        now_s=1.0,
    )
    assert blocked_nonfinite["honest_verdict"] == BLOCKED_VERDICT
