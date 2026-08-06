"""Tests for Exp6159 fresh decision-calibrated exact stream.

Spec refs: REQ-VERIFY-6159, REQ-LEARN-6159,
SCENARIO-VERIFY-6159-FRESH, SCENARIO-VERIFY-6159-BOUNDARY,
SCENARIO-VERIFY-6159-ENDPOINT, SCENARIO-VERIFY-6159-CONTROLS,
SCENARIO-LEARN-6159-PREREGISTERED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6159_decision_calibrated_stream as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _write_artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.write_decision_calibrated_stream_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_output_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        split_output_path=tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name,
        outcome_output_path=tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name,
        preregistration_output_path=tmp_path / mod.PREREGISTRATION_FILE_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
    )


def test_req_6159_specs_declare_fresh_stream_endpoint_and_principles() -> None:
    """REQ-VERIFY-6159, REQ-LEARN-6159: specs anchor the complete contract."""

    verify = VERIFY_SPEC.read_text(encoding="utf-8")
    learn = LEARN_SPEC.read_text(encoding="utf-8")
    verify_section = verify[verify.index("### REQ-VERIFY-6159") :]
    learn_section = learn[learn.index("## REQ-LEARN-6159") :]
    normalized = " ".join(verify_section.split())

    for marker in (
        "REQ-VERIFY-6159-1",
        "REQ-VERIFY-6159-2",
        "REQ-VERIFY-6159-3",
        "REQ-VERIFY-6159-4",
        "REQ-VERIFY-6159-5",
        "REQ-VERIFY-6159-6",
        "REQ-VERIFY-6159-7",
        "REQ-VERIFY-6159-8",
        "REQ-VERIFY-6159-9",
        "REQ-LEARN-6159-1",
        "REQ-LEARN-6159-2",
        "REQ-LEARN-6159-3",
        "SCENARIO-VERIFY-6159-FRESH",
        "SCENARIO-VERIFY-6159-BOUNDARY",
        "SCENARIO-VERIFY-6159-ENDPOINT",
        "SCENARIO-VERIFY-6159-CONTROLS",
        "SCENARIO-LEARN-6159-PREREGISTERED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.PREREGISTRATION_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in verify_section or marker in learn_section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in verify_section
        assert " ".join(principle.split()) in normalized


def test_scenario_6159_stream_sidecars_are_fresh_pre_post_separated(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6159-FRESH/BOUNDARY: sidecars are disjoint and fresh."""

    artifact = _write_artifact(tmp_path)
    rows = _load_jsonl(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)
    outcomes = _load_jsonl(tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name)
    splits = json.loads((tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name).read_text())
    preregistration = json.loads(
        (tmp_path / mod.PREREGISTRATION_FILE_RELATIVE_PATH.name).read_text()
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert mod.validate_artifact(artifact) is True
    assert len(rows) == len(outcomes) == 240
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert "prospective_preregistration_valid=true" in artifact["honest_verdict"]
    assert artifact["decision_calibrated_stream_ready_score"] == 1.0
    assert artifact["llm_invocation_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)

    counts = artifact["event_template_family_partition_and_shift_counts"]
    assert counts["event_count"] == 240
    assert counts["family_count"] >= 6
    assert counts["base_template_count"] == 30
    assert counts["structural_shift_event_count"] > 0
    assert counts["alias_event_count"] > 0
    assert counts["alias_counted_as_shift_count"] == 0
    assert set(splits["partition_counts"]) == set(mod.PARTITIONS)

    nonreuse = artifact["exposed_fixture_overlap_counts"]
    assert nonreuse["event_overlap_count"] == 0
    assert nonreuse["template_overlap_count"] == 0
    assert nonreuse["seed_overlap_count"] == 0
    assert artifact["never_used_seed_and_identity_receipts"]["fresh"] is True
    assert preregistration["preregistration_hash"] == mod.preregistration_hash(preregistration)
    replay = mod.replay_sidecars(
        tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name,
        tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name,
        tmp_path / mod.PREREGISTRATION_FILE_RELATIVE_PATH.name,
    )
    assert replay["ok"] is True
    assert replay["preregistration_sha256"].startswith("sha256:")

    for index, row in enumerate(rows):
        assert row["schema"] == mod.ROW_SCHEMA
        assert row["event_id"] == f"exp6159-event-{index:06d}"
        assert row["chronological_index"] == index
        assert row["row_hash"] == mod.row_hash(row)
        assert set(row["pre_decision"]) == set(mod.PRE_OUTCOME_SCHEMA["pre_decision"])
        row_text = json.dumps(row, sort_keys=True)
        assert "exact_answer" not in row_text
        assert "current_outcome" not in row_text
        assert "future_label" not in row_text
        assert "held_label" not in row_text
        assert "post_outcome" not in row_text
    for outcome in outcomes:
        assert outcome["schema"] == mod.OUTCOME_SCHEMA
        assert outcome["outcome_hash"] == mod.outcome_hash(outcome)


def test_scenario_6159_preregistration_freezes_endpoint_before_access(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6159-ENDPOINT: endpoint commitments match preregistration."""

    artifact = _write_artifact(tmp_path)
    preregistration = json.loads(
        (tmp_path / mod.PREREGISTRATION_FILE_RELATIVE_PATH.name).read_text()
    )

    assert artifact["frozen_utility_cost_table"] == preregistration["frozen_utility_cost_table"]
    assert (
        artifact["primary_cluster_unit_bootstrap_and_sample_size_plan"]
        == preregistration["primary_cluster_unit_bootstrap_and_sample_size_plan"]
    )
    assert (
        artifact["safety_and_noninferiority_margins"]
        == preregistration["safety_and_noninferiority_margins"]
    )
    assert (
        artifact["brier_ece_and_descriptive_auroc_plan"]
        == preregistration["brier_ece_and_descriptive_auroc_plan"]
    )
    assert artifact["held_loader_one_shot_contract"]["held_access_count"] == 0
    assert artifact["held_loader_one_shot_contract"]["max_pre_inference_held_access_count"] == 0
    assert artifact["held_loader_one_shot_contract"]["one_shot_after_model_rows"] is True
    assert preregistration["frozen_before_inference"] is True
    assert preregistration["held_materialization_count_at_freeze"] == 0
    assert preregistration["llm_invocation_count_at_freeze"] == 0
    assert artifact["brier_ece_and_descriptive_auroc_plan"]["auroc_role"] == "descriptive_only"
    assert "brier" in artifact["brier_ece_and_descriptive_auroc_plan"]["proper_score_endpoints"]
    assert "ece" in artifact["brier_ece_and_descriptive_auroc_plan"]["proper_score_endpoints"]


def test_scenario_6159_nonreuse_exact_controls_and_rebuild() -> None:
    """SCENARIO-VERIFY-6159-CONTROLS: exact validation and controls fail closed."""

    bundle = mod.build_stream_bundle()
    validation = mod.validate_stream_bundle(bundle)
    rebuild = mod.deterministic_rebuild_receipt()

    assert validation["exact_validator_agreement"]["python_z3_compared_count"] > 0
    assert validation["exact_validator_agreement"]["disagreement_count"] == 0
    assert validation["exact_validator_agreement"]["unresolved_disagreement_count"] == 0
    assert validation["control_counts"]["alias"]["events"] > 0
    assert validation["control_counts"]["alias"]["counted_as_shift"] == 0
    assert validation["control_counts"]["contradiction"]["rejected"] > 0
    assert validation["control_counts"]["malformed_strategy"]["rejected"] > 0
    assert validation["control_counts"]["poison"]["rejected"] > 0
    assert validation["control_counts"]["threshold_boundary"]["events"] > 0
    assert validation["shift_counts"]["structural_shift_alias_confusion_count"] == 0
    assert validation["overlap_counts"]["base_template_overlap_count"] == 0
    assert validation["prior_fixture_nonreuse"]["event_overlap_count"] == 0
    assert validation["prior_fixture_nonreuse"]["template_overlap_count"] == 0
    assert validation["prior_fixture_nonreuse"]["seed_overlap_count"] == 0
    assert rebuild["matches"] is True
    assert validation["bundle_checksum"] == rebuild["checksum"]


def test_req_6159_validation_rejects_leaks_split_drift_and_endpoint_drift(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6159-2/3/5/9: readiness rejects leakage and freeze drift."""

    artifact = _write_artifact(tmp_path)
    bundle = mod.build_stream_bundle()

    short = deepcopy(bundle)
    short.rows.pop()
    with pytest.raises(mod.DecisionCalibratedStreamError, match="chronology row count"):
        mod.validate_stream_bundle(short)

    prereg_hash_drift = deepcopy(bundle)
    prereg_hash_drift.preregistration["preregistration_hash"] = mod.sha256_text("wrong")
    with pytest.raises(mod.DecisionCalibratedStreamError, match="preregistration hash"):
        mod.validate_stream_bundle(prereg_hash_drift)

    duplicate = deepcopy(bundle)
    duplicate.rows[1]["event_id"] = duplicate.rows[0]["event_id"]
    with pytest.raises(mod.DecisionCalibratedStreamError, match="chronology event"):
        mod.validate_stream_bundle(duplicate)

    bad_index = deepcopy(bundle)
    bad_index.rows[0]["chronological_index"] = 9
    with pytest.raises(mod.DecisionCalibratedStreamError, match="chronology index"):
        mod.validate_stream_bundle(bad_index)

    row_hash_drift = deepcopy(bundle)
    row_hash_drift.rows[0]["row_hash"] = mod.sha256_text("wrong-row")
    with pytest.raises(mod.DecisionCalibratedStreamError, match="row hash"):
        mod.validate_stream_bundle(row_hash_drift)

    row_drift = deepcopy(bundle)
    row_drift.rows[0]["control_kind"] = "normal_but_drifted"
    row_drift.rows[0]["row_hash"] = mod.row_hash(row_drift.rows[0])
    with pytest.raises(mod.DecisionCalibratedStreamError, match="row drift"):
        mod.validate_stream_bundle(row_drift)

    forbidden = deepcopy(bundle)
    forbidden.rows[0]["pre_decision"]["task_descriptor"]["exact_answer"] = "leak"
    forbidden.rows[0]["row_hash"] = mod.row_hash(forbidden.rows[0])
    with pytest.raises(mod.DecisionCalibratedStreamError, match="forbidden"):
        mod.validate_stream_bundle(forbidden)

    split_drift = deepcopy(bundle)
    split_drift.rows[1]["partition"] = (
        "calibration" if split_drift.rows[1]["partition"] != "calibration" else "future_known"
    )
    split_drift.rows[1]["row_hash"] = mod.row_hash(split_drift.rows[1])
    with pytest.raises(mod.DecisionCalibratedStreamError, match="partition"):
        mod.validate_stream_bundle(split_drift)

    split_hash_drift = deepcopy(bundle)
    split_hash_drift.splits["split_hash"] = mod.sha256_text("wrong-split")
    with pytest.raises(mod.DecisionCalibratedStreamError, match="split hash"):
        mod.validate_stream_bundle(split_hash_drift)

    derivative_drift = deepcopy(bundle)
    base = derivative_drift.rows[0]["base_template_id"]
    derivative_drift.splits["base_template_to_partition"][base] = "future_known"
    derivative_drift.splits["split_hash"] = mod.split_hash(derivative_drift.splits)
    with pytest.raises(mod.DecisionCalibratedStreamError, match="partition"):
        mod.validate_stream_bundle(derivative_drift)

    outcome_chronology = deepcopy(bundle)
    outcome_chronology.outcomes[0]["event_id"] = "exp6159-event-999999"
    with pytest.raises(mod.DecisionCalibratedStreamError, match="outcome chronology"):
        mod.validate_stream_bundle(outcome_chronology)

    outcome_hash = deepcopy(bundle)
    outcome_hash.outcomes[0]["outcome_hash"] = mod.sha256_text("wrong-outcome")
    with pytest.raises(mod.DecisionCalibratedStreamError, match="outcome hash"):
        mod.validate_stream_bundle(outcome_hash)

    outcome_drift = deepcopy(bundle)
    outcome_drift.outcomes[2]["post_outcome"]["exact_labels"]["accepted"] = False
    outcome_drift.outcomes[2]["outcome_hash"] = mod.outcome_hash(outcome_drift.outcomes[2])
    with pytest.raises(mod.DecisionCalibratedStreamError, match="outcome drift"):
        mod.validate_stream_bundle(outcome_drift)

    prereg_drift = deepcopy(bundle)
    prereg_drift.preregistration["frozen_utility_cost_table"]["false_unsafe_acceptance"] = -1.0
    prereg_drift.preregistration["preregistration_hash"] = mod.preregistration_hash(
        prereg_drift.preregistration
    )
    with pytest.raises(mod.DecisionCalibratedStreamError, match="preregistration drift"):
        mod.validate_stream_bundle(prereg_drift)

    bad_access = deepcopy(artifact)
    bad_access["held_loader_one_shot_contract"]["held_access_count"] = 1
    bad_access["decision_calibrated_stream_ready_score"] = mod.ready_score(bad_access)
    bad_access["status"] = mod.status(bad_access)
    bad_access["honest_verdict"] = mod.honest_verdict(bad_access)
    bad_access["reproducibility_checksum"] = mod.reproducibility_checksum(bad_access)
    assert bad_access["decision_calibrated_stream_ready_score"] == 0.0
    with pytest.raises(ValueError, match="held_loader_one_shot_contract"):
        mod.validate_artifact(bad_access)

    bad_overlap = deepcopy(artifact)
    bad_overlap["exposed_fixture_overlap_counts"]["event_overlap_count"] = 1
    assert "prior_fixture_overlap" in mod._blocked_reasons(bad_overlap)

    compared = deepcopy(bundle.outcomes[0])
    compared["post_outcome"]["cross_backend_agreement"]["agrees"] = False
    unresolved = deepcopy(bundle.outcomes[5])
    unresolved["post_outcome"]["cross_backend_agreement"]["agrees"] = False
    receipt = mod._exact_validator_agreement([compared, unresolved])
    assert receipt["disagreement_count"] == 1
    assert receipt["unresolved_disagreement_count"] == 1

    list_forbidden = [{"event_id": "e", "nested": ["ok", {"held_label": 1}]}]
    assert mod.scan_forbidden_pre_outcome_fields(list_forbidden)["violation_count"] == 1
    assert mod._values_for_keys({"outer": [{"event_id": "e"}]}, {"event_id"}) == ["e"]

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)


def test_req_6159_schema_provenance_and_adversarial_verify(tmp_path: Path) -> None:
    """REQ-VERIFY-6159-1/9: artifact schema, provenance, and no-LLM audit hold."""

    artifact = _write_artifact(tmp_path)

    assert set(mod.FIELD_PRINCIPLES) <= set(artifact["field_provenance"])
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_score = deepcopy(artifact)
    bad_score["decision_calibrated_stream_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="decision_calibrated_stream_ready_score"):
        mod.validate_artifact(bad_score)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = False
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    bad_llm = deepcopy(artifact)
    bad_llm["llm_invocation_count"] = 1
    bad_llm["reproducibility_checksum"] = mod.reproducibility_checksum(bad_llm)
    with pytest.raises(ValueError, match="llm_invocation_count"):
        mod.validate_artifact(bad_llm)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["preconditions_ready"] = False
    blocked["forbidden_field_scan"]["violation_count"] = 1
    blocked["exact_validator_agreement"]["disagreement_count"] = 1
    blocked["alias_contradiction_malformed_poison_and_boundary_controls"][
        "all_required_controls_present"
    ] = False
    blocked["held_loader_one_shot_contract"]["held_access_count"] = 1
    blocked["llm_invocation_count"] = 1
    blocked["deterministic_rebuild_checksum"] = mod.sha256_text("wrong-rebuild")
    blocked["protected_files_unchanged"]["unchanged"] = False
    blocked["test_exit_codes"][mod.DEFAULT_TEST_COMMANDS[0]] = 1
    assert mod.status(blocked) == "blocked"
    assert mod.honest_verdict(blocked).startswith("blocked:")
    for reason in (
        "preconditions",
        "forbidden_pre_outcome_fields",
        "exact_validator_agreement",
        "missing_controls",
        "endpoint_preregistration_mismatch",
        "held_access_not_zero",
        "llm_invocation_count",
        "deterministic_rebuild",
        "protected_files",
        "test_commands",
    ):
        assert reason in mod._blocked_reasons(blocked)

    report = adversarial_verify.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH.name)
    kinds = {flag["kind"] for flag in report["flags"]}
    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds
