"""Tests for Exp6147 task-aware admission energy calibration.

Spec refs: REQ-VERIFY-6147, REQ-VERIFY-6147-1, REQ-VERIFY-6147-2,
REQ-VERIFY-6147-3, REQ-VERIFY-6147-4, REQ-VERIFY-6147-5,
REQ-VERIFY-6147-6, REQ-VERIFY-6147-7, REQ-VERIFY-6147-8,
REQ-VERIFY-6147-9, REQ-LEARN-6147,
SCENARIO-VERIFY-6147-FEATURES, SCENARIO-VERIFY-6147-REPLAY,
SCENARIO-VERIFY-6147-CONTROLS, SCENARIO-LEARN-6147-FREEZE,
SCENARIO-LEARN-6147-HELD-UNREAD.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6147_task_aware_energy_calibration as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _run_artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.25,
        write=write,
    )


def test_req_6147_specs_declare_oracle_distinct_calibration_contract() -> None:
    """REQ-VERIFY-6147/REQ-LEARN-6147: specs name fields and scenarios."""

    verify_text = VERIFY_SPEC.read_text(encoding="utf-8")
    learn_text = LEARN_SPEC.read_text(encoding="utf-8")
    verify_section = verify_text[verify_text.index("### REQ-VERIFY-6147") :]
    learn_section = learn_text[learn_text.index("## REQ-LEARN-6147") :]
    normalized = " ".join(verify_section.split())

    for marker in (
        "REQ-VERIFY-6147-1",
        "REQ-VERIFY-6147-2",
        "REQ-VERIFY-6147-3",
        "REQ-VERIFY-6147-4",
        "REQ-VERIFY-6147-5",
        "REQ-VERIFY-6147-6",
        "REQ-VERIFY-6147-7",
        "REQ-VERIFY-6147-8",
        "REQ-VERIFY-6147-9",
        "SCENARIO-VERIFY-6147-FEATURES",
        "SCENARIO-VERIFY-6147-REPLAY",
        "SCENARIO-VERIFY-6147-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in verify_section
    for marker in (
        "REQ-LEARN-6147",
        "SCENARIO-LEARN-6147-FREEZE",
        "SCENARIO-LEARN-6147-HELD-UNREAD",
    ):
        assert marker in learn_section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in verify_section
        assert " ".join(principle.split()) in normalized


def test_scenario_6147_features_replay_and_held_outcomes_are_oracle_distinct(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6147-FEATURES/REPLAY: no current or held label enters scores."""

    artifact = _run_artifact(tmp_path)

    scan = artifact["decision_time_feature_allowlist_and_forbidden_field_scan"]
    assert scan["forbidden_found_count"] == 0
    assert scan["ready_zero_if_forbidden"] is True
    assert set(mod.DECISION_TIME_FEATURE_ALLOWLIST) == set(scan["allowlist"])
    assert all(token in scan["forbidden_tokens"] for token in mod.FORBIDDEN_SCORE_TOKENS)

    held = artifact["held_outcomes_unread_receipt"]
    assert held["held_label_read_count"] == 0
    assert held["future_known_label_read_count"] == 0
    assert held["sealed_shifted_family_label_read_count"] == 0
    assert held["calibration_label_read_count"] == 240
    assert held["evaluated_partitions"] == ["calibration"]

    replay = artifact["chronological_replay_statistics"]
    assert replay["current_label_visible_before_score_count"] == 0
    assert replay["future_event_visible_before_score_count"] == 0
    assert replay["memory_budget_events_per_task"] == mod.MEMORY_BUDGET_EVENTS_PER_TASK
    assert replay["per_model"][mod.MANDATED_MODEL_IDS[0]]["scored_calibration_event_count"] == 120
    samples = replay["sample_replay_receipts"]
    assert samples
    assert samples[0]["prior_same_task_count_before_score"] == 0
    assert samples[0]["label_added_after_score"] is True


def test_req_6147_task_aware_lift_controls_threshold_and_manifest(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6147-5/6/7/8: task-aware lift gates a frozen policy."""

    artifact = _run_artifact(tmp_path, write=True)

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["task_aware_energy_calibration_ready_score"] == 1.0
    assert artifact["retirement_triggered"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    per_model = artifact["per_model_grouped_metrics_and_intervals"]["by_model"]
    for model_id, model_metrics in per_model.items():
        assert model_id in mod.MANDATED_MODEL_IDS
        delta = model_metrics["primary_metric_delta_task_aware_minus_global"]
        assert delta["observed"] > 0.08
        assert delta["ci95"][0] > 0.0
        assert model_metrics["scores"]["task_aware_energy"]["auroc"] > model_metrics["scores"]["global_energy"]["auroc"]
        assert model_metrics["scores"]["task_frequency"]["auroc"] < 0.6
        assert model_metrics["scores"]["random"]["auroc"] < 0.7
        assert model_metrics["grouping"]["group_key"] == "base_template_id"

    controls = artifact[
        "alias_frequency_identity_length_timestamp_duplicate_outcome_permutation_and_shuffle_controls"
    ]
    assert controls["all_required_controls_present"] is True
    assert controls["all_controls_passed"] is True
    assert controls["label_shuffle"]["passed"] is True
    assert controls["outcome_permutation"]["passed"] is True
    assert controls["timestamp"]["timestamp_direct_feature_used"] is False
    assert controls["duplicate"]["duplicate_event_id_count"] == 0

    selected = artifact["selected_score_threshold_abstention_and_memory_budget"]
    assert selected["selected_score"] == "task_aware_energy"
    assert selected["memory_budget_events_per_task"] == mod.MEMORY_BUDGET_EVENTS_PER_TASK
    assert selected["selection_uses_held_outcomes"] is False
    assert selected["abstention_rule"]["margin"] == mod.ABSTENTION_MARGIN
    assert artifact["selection_manifest_hash"] == mod.selection_manifest_hash(selected)

    confusion = artifact["calibration_coverage_risk_and_confusion_matrices"]["by_model"]
    for matrix in confusion.values():
        assert matrix["confusion_matrix"]["unsafe_total"] > 0
        assert matrix["confusion_matrix"]["safe_total"] > 0
        assert matrix["coverage"] > 0.5


def test_req_6147_validation_forces_zero_on_forbidden_held_or_manifest_drift(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6147-3/8: forbidden fields and held reads fail closed."""

    artifact = _run_artifact(tmp_path)

    bad_forbidden = deepcopy(artifact)
    bad_forbidden["decision_time_feature_allowlist_and_forbidden_field_scan"][
        "forbidden_found_count"
    ] = 1
    bad_forbidden["task_aware_energy_calibration_ready_score"] = mod.ready_score(
        bad_forbidden
    )
    bad_forbidden["status"] = mod.status(bad_forbidden)
    bad_forbidden["honest_verdict"] = mod.honest_verdict(bad_forbidden)
    bad_forbidden["reproducibility_checksum"] = mod.reproducibility_checksum(bad_forbidden)
    assert bad_forbidden["task_aware_energy_calibration_ready_score"] == 0.0
    with pytest.raises(ValueError, match="forbidden"):
        mod.validate_artifact(bad_forbidden)

    bad_held = deepcopy(artifact)
    bad_held["held_outcomes_unread_receipt"]["held_label_read_count"] = 1
    bad_held["task_aware_energy_calibration_ready_score"] = mod.ready_score(bad_held)
    bad_held["status"] = mod.status(bad_held)
    bad_held["honest_verdict"] = mod.honest_verdict(bad_held)
    bad_held["reproducibility_checksum"] = mod.reproducibility_checksum(bad_held)
    assert bad_held["task_aware_energy_calibration_ready_score"] == 0.0
    with pytest.raises(ValueError, match="held_outcomes_unread"):
        mod.validate_artifact(bad_held)

    bad_manifest = deepcopy(artifact)
    bad_manifest["selection_manifest_hash"] = mod.sha256_text("wrong")
    bad_manifest["reproducibility_checksum"] = mod.reproducibility_checksum(bad_manifest)
    with pytest.raises(ValueError, match="selection_manifest_hash"):
        mod.validate_artifact(bad_manifest)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance
    )
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_score = deepcopy(artifact)
    bad_score["task_aware_energy_calibration_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="task_aware_energy_calibration_ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_null"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "simulation"
    bad_substrate["task_aware_energy_calibration_ready_score"] = mod.ready_score(
        bad_substrate
    )
    bad_substrate["status"] = mod.status(bad_substrate)
    bad_substrate["honest_verdict"] = mod.honest_verdict(bad_substrate)
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_substrate
    )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = True
    bad_verifier["task_aware_energy_calibration_ready_score"] = mod.ready_score(
        bad_verifier
    )
    bad_verifier["status"] = mod.status(bad_verifier)
    bad_verifier["honest_verdict"] = mod.honest_verdict(bad_verifier)
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_verifier
    )
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    no_lift = deepcopy(artifact)
    first_model = mod.MANDATED_MODEL_IDS[0]
    no_lift["per_model_grouped_metrics_and_intervals"]["by_model"][first_model][
        "primary_metric_delta_task_aware_minus_global"
    ]["positive_lower_95"] = False
    assert any("nonpositive_task_aware_lift" in item for item in mod._blocked_reasons(no_lift))

    retired = deepcopy(artifact)
    retired["retirement_triggered"] = True
    assert mod.status(retired) == "retired"
    assert mod.honest_verdict(retired).startswith("retired:")

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)


def test_req_6147_metric_helpers_cover_empty_and_abstain_edges() -> None:
    """REQ-VERIFY-6147-6/7: edge cases fail closed without hidden labels."""

    assert mod._quantile([], 0.5) == 0.0
    assert mod._quantile([3.0], 0.5) == 3.0
    assert mod._shuffled_task_labels([]) == []
    assert mod._shuffled_task_labels(["only"]) == ["only"]
    assert mod._auprc([0, 0], [0.1, 0.2]) == 0.0
    assert mod._brier([], []) == 0.0
    assert mod._ece([], []) == 0.0

    scan = mod._scan_score_inputs(
        [
            {
                "event_id": "fixture",
                "features": {"family": "calibration", "note": "exact_answer"},
                "replay": {},
            }
        ]
    )
    assert scan["forbidden_found_count"] > 0

    confusion = mod._confusion_for_entries(
        [
            {
                "unsafe_label": 1,
                "scores": {"task_aware_energy": 0.0},
            }
        ],
        threshold=0.0,
        margin=0.01,
    )
    assert confusion["abstained_count"] == 1


def test_req_6147_blocks_when_upstream_gate_is_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-6147-1/8/9: upstream readiness gates calibration."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        exp6146_artifact={"sota_constraint_event_corpus_ready_score": 0},
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.5,
        write=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["task_aware_energy_calibration_ready_score"] == 0.0
    assert "exp6146_ready_score" in artifact["honest_verdict"]
    assert artifact["structured_gate_receipt"]["calibration_permitted"] is False
    assert artifact["per_model_grouped_metrics_and_intervals"]["by_model"] == {}
    assert mod.validate_artifact(artifact) is True


def test_req_6147_adversarial_verify_recognizes_cached_no_llm_substrate(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6147-9: cached SOTA row scoring is not live GGUF inference."""

    artifact = _run_artifact(tmp_path)
    report = adversarial_verify.verify_artifact(
        tmp_path / mod.RESULT_RELATIVE_PATH.name
        if (tmp_path / mod.RESULT_RELATIVE_PATH.name).exists()
        else _write_artifact(tmp_path, artifact)
    )
    kinds = {flag["kind"] for flag in report["flags"]}

    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds


def _write_artifact(tmp_path: Path, artifact: dict[str, Any]) -> Path:
    path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path
