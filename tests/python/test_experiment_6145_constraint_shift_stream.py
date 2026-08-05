"""Tests for Exp6145 exact constraint-shift event stream.

Spec refs: REQ-VERIFY-6145, SCENARIO-VERIFY-6145-STREAM,
SCENARIO-VERIFY-6145-EXACT, SCENARIO-VERIFY-6145-SHIFT,
SCENARIO-VERIFY-6145-REBUILD, REQ-LEARN-6145,
SCENARIO-LEARN-6145-BOUNDARY, SCENARIO-LEARN-6145-PARTITIONS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6145_constraint_shift_stream as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_6145_specs_declare_stream_boundary_and_principles() -> None:
    """REQ-VERIFY-6145, REQ-LEARN-6145: specs anchor the exact stream contract."""

    verify = VERIFY_SPEC.read_text(encoding="utf-8")
    learn = LEARN_SPEC.read_text(encoding="utf-8")
    verify_section = verify[verify.index("### REQ-VERIFY-6145") :]
    learn_section = learn[learn.index("## REQ-LEARN-6145") :]
    normalized = " ".join(verify_section.split())

    for marker in (
        "SCENARIO-VERIFY-6145-STREAM",
        "SCENARIO-VERIFY-6145-EXACT",
        "SCENARIO-VERIFY-6145-SHIFT",
        "SCENARIO-VERIFY-6145-REBUILD",
        "SCENARIO-LEARN-6145-BOUNDARY",
        "SCENARIO-LEARN-6145-PARTITIONS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_FILE_RELATIVE_PATH.as_posix(),
        mod.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        mod.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in verify_section or marker in learn_section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in verify_section
        assert " ".join(principle.split()) in normalized


def test_scenario_6145_stream_materializes_pre_post_separated_sidecars(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6145-STREAM: rows are chronological and pre-outcome only."""

    artifact = mod.write_constraint_shift_stream_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_output_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        split_output_path=tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name,
        outcome_output_path=tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
    )
    rows = _load_jsonl(tmp_path / mod.ROW_FILE_RELATIVE_PATH.name)
    outcomes = _load_jsonl(tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name)
    splits = json.loads((tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name).read_text())
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert mod.validate_artifact(artifact) is True
    assert len(rows) == len(outcomes) == 240
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["constraint_shift_stream_ready_score"] == 1.0
    assert artifact["llm_invocation_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.FIELD_PRINCIPLES) <= set(artifact["field_provenance"])

    counts = artifact["event_base_template_family_partition_and_shift_counts"]
    assert counts["event_count"] == 240
    assert counts["family_count"] >= 6
    assert counts["base_template_count"] == 48
    assert counts["structural_shift_event_count"] > 0
    assert counts["alias_event_count"] > 0
    assert counts["alias_counted_as_shift_count"] == 0

    receipt = mod.replay_sidecars(
        row_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        split_path=tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name,
        outcome_path=tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name,
    )
    assert receipt["ok"] is True
    assert receipt["chronological_order"]["monotone"] is True
    assert receipt["forbidden_pre_outcome_field_scan"]["violation_count"] == 0
    assert receipt["overlap_counts"]["base_template_overlap_count"] == 0
    assert receipt["exact_validator_agreement"]["unresolved_disagreement_count"] == 0
    assert splits["partition_counts"] == counts["partition_counts"]

    for index, row in enumerate(rows):
        assert row["schema"] == mod.ROW_SCHEMA
        assert row["event_id"] == f"exp6145-event-{index:06d}"
        assert row["chronological_index"] == index
        assert row["row_hash"] == mod.row_hash(row)
        assert set(row["pre_decision"]) == set(mod.PRE_OUTCOME_SCHEMA["pre_decision"])
        assert "exact_answer" not in json.dumps(row, sort_keys=True)
        assert "current_validator_result" not in json.dumps(row, sort_keys=True)
    for outcome in outcomes:
        assert outcome["schema"] == mod.OUTCOME_SCHEMA
        assert outcome["outcome_hash"] == mod.outcome_hash(outcome)


def test_scenario_6145_exact_labels_controls_and_shift_alias_receipts() -> None:
    """SCENARIO-VERIFY-6145-EXACT, SCENARIO-VERIFY-6145-SHIFT: controls fail closed."""

    bundle = mod.build_stream_bundle()
    receipt = mod.validate_stream_bundle(bundle)

    assert receipt["exact_validator_agreement"]["python_z3_compared_count"] > 0
    assert receipt["exact_validator_agreement"]["disagreement_count"] == 0
    assert receipt["exact_validator_agreement"]["unresolved_disagreement_count"] == 0
    assert receipt["control_counts"]["contradiction"]["rejected"] > 0
    assert receipt["control_counts"]["malformed_proposal"]["rejected"] > 0
    assert receipt["control_counts"]["strategy_poison"]["rejected"] > 0
    assert receipt["control_counts"]["alias"]["accepted"] > 0
    assert receipt["control_counts"]["alias"]["counted_as_shift"] == 0
    assert receipt["shift_counts"]["sealed_structural_shift_family_count"] >= 1
    assert receipt["shift_counts"]["structural_shift_alias_confusion_count"] == 0
    assert receipt["overlap_counts"]["derivative_partition_mismatch_count"] == 0

    contradictions = [
        outcome
        for outcome in bundle.outcomes
        if outcome["post_outcome"]["control_kind"] == "contradiction"
    ]
    assert contradictions
    assert all(
        item["post_outcome"]["current_validator_result"] == "rejected" for item in contradictions
    )
    aliases = [row for row in bundle.rows if row["variant_kind"] == "alias"]
    assert aliases
    assert all(row["alias_only"] is True and row["structural_shift"] is False for row in aliases)


def test_scenario_6145_rejects_forbidden_pre_fields_split_drift_and_posthoc_labels() -> None:
    """SCENARIO-LEARN-6145-BOUNDARY/PARTITIONS: tampering rejects by interface."""

    bundle = mod.build_stream_bundle()
    mod.validate_stream_bundle(bundle)

    forbidden = deepcopy(bundle)
    forbidden.rows[0]["pre_decision"]["task_descriptor"]["exact_answer"] = ["leak"]
    with pytest.raises(mod.ConstraintShiftStreamError, match="forbidden pre-outcome"):
        mod.validate_stream_bundle(forbidden)

    split_drift = deepcopy(bundle)
    split_drift.rows[1]["partition"] = (
        "calibration" if split_drift.rows[1]["partition"] != "calibration" else "future_known"
    )
    with pytest.raises(mod.ConstraintShiftStreamError, match="partition drift"):
        mod.validate_stream_bundle(split_drift)

    posthoc = deepcopy(bundle)
    posthoc.outcomes[2]["post_outcome"]["python"]["status"] = "tampered"
    with pytest.raises(mod.ConstraintShiftStreamError, match="outcome drift"):
        mod.validate_stream_bundle(posthoc)

    duplicate = deepcopy(bundle)
    duplicate.rows[1]["event_id"] = duplicate.rows[0]["event_id"]
    with pytest.raises(mod.ConstraintShiftStreamError, match="chronology"):
        mod.validate_stream_bundle(duplicate)


def test_scenario_6145_ready_score_and_artifact_validation_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6145-REBUILD: readiness is exactly gated and deterministic."""

    artifact = mod.write_constraint_shift_stream_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_output_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        split_output_path=tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name,
        outcome_output_path=tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
    )
    rebuild = mod.deterministic_rebuild_receipt()

    assert rebuild["matches"] is True
    assert artifact["deterministic_rebuild_checksum"] == rebuild["checksum"]
    assert mod.ready_score(artifact) == 1.0

    bad_llm = deepcopy(artifact)
    bad_llm["llm_invocation_count"] = 1
    bad_llm["constraint_shift_stream_ready_score"] = mod.ready_score(bad_llm)
    bad_llm["status"] = mod.status(bad_llm)
    bad_llm["honest_verdict"] = mod.honest_verdict(bad_llm)
    bad_llm["reproducibility_checksum"] = mod.reproducibility_checksum(bad_llm)
    assert bad_llm["constraint_shift_stream_ready_score"] == 0.0
    with pytest.raises(ValueError, match="llm_invocation_count"):
        mod.validate_artifact(bad_llm)

    bad_score = deepcopy(artifact)
    bad_score["constraint_shift_stream_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_6145_fail_closed_guard_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-6145: schema, split, chronology, and verdict guards reject drift."""

    artifact = mod.write_constraint_shift_stream_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_output_path=tmp_path / mod.ROW_FILE_RELATIVE_PATH.name,
        split_output_path=tmp_path / mod.SPLIT_FILE_RELATIVE_PATH.name,
        outcome_output_path=tmp_path / mod.OUTCOME_FILE_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
    )
    bundle = mod.build_stream_bundle()

    short = deepcopy(bundle)
    short.rows.pop()
    with pytest.raises(mod.ConstraintShiftStreamError, match="chronology row count"):
        mod.validate_stream_bundle(short)

    bad_index = deepcopy(bundle)
    bad_index.rows[0]["chronological_index"] = 7
    with pytest.raises(mod.ConstraintShiftStreamError, match="chronology index"):
        mod.validate_stream_bundle(bad_index)

    row_hash_drift = deepcopy(bundle)
    row_hash_drift.rows[0]["row_hash"] = mod.sha256_text("wrong-row")
    with pytest.raises(mod.ConstraintShiftStreamError, match="row hash"):
        mod.validate_stream_bundle(row_hash_drift)

    row_drift = deepcopy(bundle)
    row_drift.rows[0]["control_kind"] = "normal-but-drifted"
    row_drift.rows[0]["row_hash"] = mod.row_hash(row_drift.rows[0])
    with pytest.raises(mod.ConstraintShiftStreamError, match="row drift"):
        mod.validate_stream_bundle(row_drift)

    outcome_chronology = deepcopy(bundle)
    outcome_chronology.outcomes[0]["event_id"] = "exp6145-event-999999"
    with pytest.raises(mod.ConstraintShiftStreamError, match="outcome chronology"):
        mod.validate_stream_bundle(outcome_chronology)

    outcome_drift = deepcopy(bundle)
    outcome_drift.outcomes[0]["post_outcome"]["exact_labels"]["accepted"] = False
    outcome_drift.outcomes[0]["outcome_hash"] = mod.outcome_hash(outcome_drift.outcomes[0])
    with pytest.raises(mod.ConstraintShiftStreamError, match="outcome drift"):
        mod.validate_stream_bundle(outcome_drift)

    split_hash_drift = deepcopy(bundle)
    split_hash_drift.splits["split_hash"] = mod.sha256_text("wrong-split")
    with pytest.raises(mod.ConstraintShiftStreamError, match="split hash"):
        mod.validate_stream_bundle(split_hash_drift)

    derivative_drift = deepcopy(bundle)
    base = derivative_drift.rows[0]["base_template_id"]
    derivative_drift.splits["base_template_to_partition"][base] = "calibration"
    derivative_drift.splits["split_hash"] = mod.split_hash(derivative_drift.splits)
    with pytest.raises(mod.ConstraintShiftStreamError, match="partition drift"):
        mod.validate_stream_bundle(derivative_drift)

    compared = deepcopy(bundle.outcomes[0])
    compared["post_outcome"]["cross_backend_agreement"]["agrees"] = False
    unresolved = deepcopy(bundle.outcomes[9])
    unresolved["post_outcome"]["cross_backend_agreement"]["agrees"] = False
    receipt = mod._exact_validator_agreement([compared, unresolved])
    assert receipt["disagreement_count"] == 1
    assert receipt["unresolved_disagreement_count"] == 1

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

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
    blocked["llm_invocation_count"] = 1
    blocked["forbidden_pre_outcome_field_scan"]["violation_count"] = 1
    blocked["exact_validator_agreement"]["disagreement_count"] = 1
    blocked["calibration_future_known_shifted_overlap_counts"]["base_template_overlap_count"] = 1
    blocked["protected_files_unchanged"]["unchanged"] = False
    blocked["deterministic_rebuild_checksum"] = mod.sha256_text("wrong-rebuild")
    first_command = mod.DEFAULT_TEST_COMMANDS[0]
    blocked["test_exit_codes"].pop(first_command)
    blocked["test_exit_codes"][mod.DEFAULT_TEST_COMMANDS[1]] = 1
    assert mod.status(blocked) == "blocked"
    verdict = mod.honest_verdict(blocked)
    assert verdict.startswith("blocked:")
    for reason in (
        "preconditions",
        "llm_invocation_count",
        "forbidden_pre_outcome_fields",
        "exact_validator_agreement",
        "split_overlap",
        "protected_files",
        "deterministic_rebuild",
        "missing_test_commands",
    ):
        assert reason in mod._blocked_reasons(blocked)

    assert mod._read_json_if_exists(tmp_path / "missing.json") == {}
