"""Tests for Exp6346 certified factor evolution safety audit.

Spec refs: REQ-LEARN-6346, SCENARIO-LEARN-6346-MANIFEST,
SCENARIO-LEARN-6346-EPROCESS, SCENARIO-LEARN-6346-LIFECYCLE,
SCENARIO-LEARN-6346-PROTECTED, SCENARIO-LEARN-6346-ROLLBACK,
SCENARIO-LEARN-6346-BOUNDARY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6346_certified_factor_evolution_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    return mod.run(
        date="20260812",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, Any]) -> dict[str, Any]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_learn_6346_spec_declares_contract() -> None:
    """REQ-LEARN-6346: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6346") :]
    for token in (
        "SCENARIO-LEARN-6346-MANIFEST",
        "SCENARIO-LEARN-6346-EPROCESS",
        "SCENARIO-LEARN-6346-LIFECYCLE",
        "SCENARIO-LEARN-6346-PROTECTED",
        "SCENARIO-LEARN-6346-ROLLBACK",
        "SCENARIO-LEARN-6346-BOUNDARY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in " ".join(section.split())


def test_scenario_learn_6346_manifest_precedes_outcome_sensitive_reads(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6346-MANIFEST: attack choices freeze first."""

    artifact = _artifact(tmp_path)
    manifest_receipt = artifact["attack_manifest_path_hash_and_preoutcome_receipt"]
    manifest = json.loads(Path(manifest_receipt["path"]).read_text(encoding="utf-8"))
    preconditions = artifact["preconditions_checked"]

    assert manifest_receipt["sha256"] == mod.sha256_file(Path(manifest_receipt["path"]))
    assert manifest_receipt["manifest_written_before_outcome_sensitive_reads"] is True
    assert manifest["attack_classes"] == list(mod.ATTACK_CLASSES)
    assert [row["attack_class"] for row in manifest["attacks"]] == list(mod.ATTACK_CLASSES)
    assert all(row["expected_terminal_decision"] in mod.FAIL_CLOSED_ACTIONS for row in manifest["attacks"])
    assert all(row["corruption_location"] for row in manifest["attacks"])
    assert preconditions["manifest_written_before_outcome_sensitive_reads"] is True
    assert preconditions["outcome_sensitive_reads_after_manifest_hash"] is True
    assert artifact["information_isolation_contract"]["outcome_sensitive_fields_read_after_manifest"] is True
    assert artifact["upstream_paths_hashes_and_terminal_classes"]["exp6345"]["terminal"] is True


def test_scenario_learn_6346_eprocess_and_lifecycle_attacks_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6346-EPROCESS: copied-state attacks never release."""

    artifact = _artifact(tmp_path)
    eprocess = artifact[
        "optional_stopping_peeking_reset_duplicate_cross_factor_selected_null_and_identity_attack_results"
    ]
    lifecycle = artifact[
        "rationale_counterexample_lineage_certificate_merge_delete_and_eviction_attack_results"
    ]

    assert eprocess["all_attacks_fail_closed"] is True
    assert lifecycle["all_attacks_fail_closed"] is True
    assert eprocess["released_attack_count"] == 0
    assert lifecycle["became_active_count"] == 0
    assert eprocess["optional_stopping_attack"]["state_results"]["clean"]["terminal_decision"] == "reject"
    assert eprocess["evalue_reset_attack"]["state_results"]["corrupted"]["terminal_decision"] == "abort"
    assert lifecycle["rationale_laundering_attack"]["state_results"]["clean"]["terminal_decision"] == "reject"
    assert lifecycle["capacity_eviction_attack"]["state_results"]["clean"]["terminal_decision"] == "quarantine"
    assert artifact["fail_closed_count_by_attack_class"]["all_attack_classes_fail_closed"] is True
    assert artifact["fail_closed_count_by_attack_class"]["attack_count"] == len(mod.ATTACK_CLASSES)
    assert all(
        row["fail_closed_count"] == len(mod.UPSTREAM_STATE_CLASSES)
        for row in artifact["fail_closed_count_by_attack_class"]["by_attack"].values()
    )


def test_scenario_learn_6346_protected_missing_skipped_and_corrupt_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6346-PROTECTED: missing evidence is blocked."""

    artifact = _artifact(tmp_path)
    protected = artifact["protected_validation_read_reuse_and_budget_asymmetry_results"]
    missing = artifact["missing_upstream_and_skipped_utility_handling"]

    assert protected["all_attacks_fail_closed"] is True
    assert protected["protected_validation_read_attack"]["state_results"]["clean"][
        "protected_validation_leak_count"
    ] == 0
    assert missing["missing_evidence_counts_as_safety_success"] is False
    assert missing["skipped_utility_counts_as_utility_success"] is False
    assert missing["skipped_utility_task_still_runs_safety_audit"] is True
    assert set(missing["synthetic_state_probe_results"]) == set(mod.UPSTREAM_STATE_CLASSES)
    assert missing["synthetic_state_probe_results"]["missing"]["terminal_decision"] == "abort"
    assert missing["synthetic_state_probe_results"]["skipped"]["utility_promotion_count"] == 0
    assert missing["synthetic_state_probe_results"]["corrupted"]["counts_as_safety_success"] is False


def test_scenario_learn_6346_source_restart_rollback_and_zero_boundary(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6346-ROLLBACK: parent bytes restore exactly."""

    artifact = _artifact(tmp_path)
    source = artifact["source_model_mutation_results"]
    restart = artifact["restart_corruption_and_rollback_failure_results"]
    rollback = artifact["rollback_byte_identity"]

    assert source["all_attacks_fail_closed"] is True
    assert restart["all_attacks_fail_closed"] is True
    assert artifact["source_model_weight_mutation_count"] == 0
    assert rollback["all_parent_bytes_match_after_restart"] is True
    assert rollback["byte_identical_parent_restoration"] is True
    assert rollback["parent_restore_count"] >= 1
    assert all(row["parent_bytes_sha256"] == row["restored_bytes_sha256"] for row in rollback["receipts"])
    for field in (
        "unsafe_commit_count",
        "undetected_harmful_attack_count",
        "protected_validation_leak_count",
        "source_model_weight_mutation_count",
        "utility_promotion_count",
        "generated_label_count",
        "llm_call_count",
    ):
        assert type(artifact[field]) is int
        assert artifact[field] == 0


def test_req_learn_6346_schema_ready_checksum_and_negative_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6346-BOUNDARY: safety never promotes utility."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260812", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["safety_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["utility_promotion_count"] == 0
    assert artifact["exact_oracle_claim_boundary"]["claim_boundary"] == "mixed"
    assert artifact["verifier_is_oracle"]["mixed_boundary"] is True
    assert "deterministic_exact_outcome_checker" in artifact["verifier_is_oracle"]["exact_oracle_checks"]
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    utility_abuse = json.loads(json.dumps(artifact))
    utility_abuse["utility_promotion_count"] = 1
    _refresh(utility_abuse)
    assert utility_abuse["safety_ready_score"] == 0.0
    with pytest.raises(ValueError, match="utility_promotion_count"):
        mod.validate_artifact(utility_abuse)

    bad_attack = json.loads(json.dumps(artifact))
    bad_attack["fail_closed_count_by_attack_class"]["by_attack"][mod.ATTACK_CLASSES[0]][
        "all_states_fail_closed"
    ] = False
    _refresh(bad_attack)
    assert bad_attack["safety_ready_score"] == 0.0

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6346_upstream_state_classifiers_and_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-6346: missing, skipped, and corrupt upstream classes close."""

    skipped = tmp_path / "skipped.json"
    skipped.write_text(
        json.dumps({"status": "skipped", "honest_verdict": "skipped: upstream utility gated"}),
        encoding="utf-8",
    )
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")

    assert mod.terminal_path_receipt(tmp_path / "missing.json")["terminal_class"] == "missing"
    assert mod.terminal_path_receipt(skipped)["terminal_class"] == "skipped"
    assert mod.terminal_path_receipt(corrupt)["terminal_class"] == "malformed"
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.read_json_object(corrupt) is None
    assert mod.classify_upstream_state(mod.terminal_path_receipt(skipped)) == "skipped"
    assert mod.classify_upstream_state(mod.terminal_path_receipt(corrupt)) == "corrupted"
    assert mod.classify_upstream_state(mod.terminal_path_receipt(tmp_path / "missing.json")) == "missing"
    assert mod.test_exit_codes(None, ["cmd"]) == {"cmd": 0}
    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_bytes(b"abc").startswith("sha256:")
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.expected_decision("not_an_attack")
    with pytest.raises(ValueError, match="bad"):
        mod.require(False, "bad")
