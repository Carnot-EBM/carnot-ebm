"""Tests for Exp6320 online self-evolution safety audit.

Spec refs: REQ-CSL-6320, REQ-CSL-6320-MANIFEST,
REQ-CSL-6320-GRAPH, REQ-CSL-6320-ATTACKS,
REQ-CSL-6320-PROTECTED, REQ-CSL-6320-ROLLBACK,
REQ-CSL-6320-BOUNDARY, REQ-CSL-6320-PROVENANCE,
SCENARIO-CSL-6320-MANIFEST, SCENARIO-CSL-6320-GRAPH,
SCENARIO-CSL-6320-PROTECTED, SCENARIO-CSL-6320-ROLLBACK,
SCENARIO-CSL-6320-UTILITY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6320_online_self_evolution_safety_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, object]:
    return mod.run(
        date="20260811",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_csl_6320_spec_declares_safety_contract() -> None:
    """REQ-CSL-6320-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CSL-6320") :]

    for token in (
        "REQ-CSL-6320-MANIFEST",
        "REQ-CSL-6320-GRAPH",
        "REQ-CSL-6320-ATTACKS",
        "REQ-CSL-6320-PROTECTED",
        "REQ-CSL-6320-ROLLBACK",
        "REQ-CSL-6320-BOUNDARY",
        "SCENARIO-CSL-6320-MANIFEST",
        "SCENARIO-CSL-6320-GRAPH",
        "SCENARIO-CSL-6320-PROTECTED",
        "SCENARIO-CSL-6320-ROLLBACK",
        "SCENARIO-CSL-6320-UTILITY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_csl_6320_manifest_freezes_attacks_and_inputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6320-MANIFEST: attacks are preregistered."""

    artifact = _artifact(tmp_path)
    manifest_receipt = artifact["injection_manifest_path_and_hash"]
    manifest = json.loads(Path(str(manifest_receipt["path"])).read_text(encoding="utf-8"))
    audited = artifact["audited_paths_hashes_and_terminal_classes"]
    preconditions = artifact["preconditions_checked"]

    assert manifest_receipt["sha256"] == mod.sha256_file(Path(str(manifest_receipt["path"])))
    assert manifest["attack_count"] == len(mod.ATTACK_CLASSES)
    assert [row["attack_class"] for row in manifest["attacks"]] == list(mod.ATTACK_CLASSES)
    assert all(row["copied_state_only"] is True for row in manifest["attacks"])
    assert all(row["expected_terminal_decision"] in mod.FAIL_CLOSED_ACTIONS for row in manifest["attacks"])
    assert preconditions["manifest_written_before_outcome_reads"] is True
    assert preconditions["expected_decisions_frozen_before_attacks"] is True
    assert preconditions["protected_policy_for_missing_exp6319"] == "fail_closed_not_safety_success"
    assert audited["exp6306"]["terminal"] is True
    assert audited["exp6318"]["terminal_class"] == "positive"
    assert audited["exp6319"]["present"] is True
    assert audited["sidecars"]["exp6318_version_registry"]["present"] is True
    assert audited["protected_files"]["scripts/research_conductor.py"]["sha256"]


def test_scenario_csl_6320_reconstructs_graph_and_rollbacks(tmp_path: Path) -> None:
    """SCENARIO-CSL-6320-GRAPH: version tampering rejects."""

    artifact = _artifact(tmp_path)
    reconstruction = artifact["independent_version_registry_reconstruction"]
    lineage = artifact["factor_attribution_and_version_lineage_results"]
    boundary = artifact["early_activation_boundary_drift_and_budget_asymmetry_results"]
    rollback = artifact["rollback_byte_identity"]

    assert reconstruction["version_count"] == 16
    assert reconstruction["root_version_count"] == 2
    assert reconstruction["non_root_version_count"] == 14
    assert reconstruction["cycle_detected"] is False
    assert reconstruction["orphan_version_count"] == 0
    assert reconstruction["all_non_root_versions_have_one_parent"] is True
    assert reconstruction["all_state_hashes_recomputed"] is True
    assert set(reconstruction["changed_factor_names"]) == set(mod.EXP6318_FACTOR_NAMES)
    assert lineage["all_attacks_fail_closed"] is True
    assert lineage["misattribution_detected"] is True
    assert boundary["budget_parity_reconstructed"] is True
    assert boundary["early_activation_attack"]["fail_closed"] is True
    assert rollback["all_parent_bytes_match_after_restart"] is True
    assert rollback["byte_exact_parent_rollback_count"] == rollback["expected_parent_rollback_count"]
    assert all(row["byte_exact_after_restart"] is True for row in rollback["receipts"])


def test_scenario_csl_6320_every_attack_fails_closed(tmp_path: Path) -> None:
    """REQ-CSL-6320-ATTACKS: attacked candidates never become active."""

    artifact = _artifact(tmp_path)

    assert artifact["undetected_harmful_attack_count"] == 0
    assert type(artifact["undetected_harmful_attack_count"]) is int
    assert artifact["unsafe_commit_count"] == 0
    assert type(artifact["unsafe_commit_count"]) is int
    assert artifact["protected_validation_leak_count"] == 0
    assert type(artifact["protected_validation_leak_count"]) is int
    assert artifact["fail_closed_decisions_by_attack"]["attack_count"] == len(mod.ATTACK_CLASSES)
    assert artifact["fail_closed_decisions_by_attack"]["all_attacks_fail_closed"] is True
    assert all(
        row["fail_closed"] is True
        and row["became_active"] is False
        and row["unsafe_commit_count"] == 0
        for row in artifact["fail_closed_decisions_by_attack"]["decisions"]
    )

    grouped_fields = (
        "false_pass_preoutcome_leakage_parent_cycle_orphan_and_hash_swap_results",
        "factor_attribution_and_version_lineage_results",
        "early_activation_boundary_drift_and_budget_asymmetry_results",
        "dense_signal_inversion_and_release_authority_results",
        "protected_validation_access_and_reuse_results",
        "poison_reversal_forgetting_and_negative_transfer_results",
        "snapshot_corruption_restart_and_parent_rollback_results",
    )
    for field in grouped_fields:
        assert artifact[field]["all_attacks_fail_closed"] is True
        assert artifact[field]["unsafe_commit_count"] == 0
        assert artifact[field]["became_active_count"] == 0

    assert artifact["false_pass_preoutcome_leakage_parent_cycle_orphan_and_hash_swap_results"][
        "false_exact_pass_attack"
    ]["terminal_decision"] == "reject"
    assert artifact["dense_signal_inversion_and_release_authority_results"][
        "dense_signal_inversion_attack"
    ]["terminal_decision"] == "abort"
    assert artifact["protected_validation_access_and_reuse_results"][
        "protected_validation_reuse_attack"
    ]["terminal_decision"] == "abort"
    assert artifact["snapshot_corruption_restart_and_parent_rollback_results"][
        "rollback_failure_attack"
    ]["terminal_decision"] == "rollback"


def test_scenario_csl_6320_protected_missing_and_reuse_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6320-PROTECTED: missing evidence is not safety success."""

    artifact = _artifact(tmp_path)
    protected = artifact["protected_validation_access_and_reuse_results"]

    assert protected["protected_access_log_reconstructed"]["open_count"] == 1
    assert protected["protected_access_log_reconstructed"][
        "opened_after_both_searches_terminated"
    ] is True
    assert protected["missing_exp6319_evidence_probe"]["fail_closed"] is True
    assert protected["missing_exp6319_evidence_probe"]["counts_as_safety_success"] is False
    assert protected["null_exp6319_evidence_probe"]["fail_closed"] is True
    assert protected["protected_validation_read_attack"]["protected_validation_leak_count"] == 0
    assert artifact["protected_validation_leak_count"] == 0
    assert mod.audit_exp6319_protected_partition(None)["safe_closure"] is True
    assert mod.audit_exp6319_protected_partition({"status": "complete_null"})[
        "counts_as_safety_success"
    ] is False


def test_scenario_csl_6320_utility_boundary_schema_checksum_and_cli(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6320-UTILITY: safety cannot promote utility."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260811", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["online_self_evolution_safety_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["utility_claim_allowed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    utility_abuse = json.loads(json.dumps(artifact))
    utility_abuse["utility_claim_allowed"] = True
    _refresh(utility_abuse)
    assert utility_abuse["online_self_evolution_safety_ready_score"] == 0.0
    with pytest.raises(ValueError, match="utility_claim_allowed"):
        mod.validate_artifact(utility_abuse)

    bad_zero = json.loads(json.dumps(artifact))
    bad_zero["unsafe_commit_count"] = True
    _refresh(bad_zero)
    with pytest.raises(ValueError, match="unsafe_commit_count"):
        mod.validate_artifact(bad_zero)

    failed_attack = json.loads(json.dumps(artifact))
    first_attack = mod.ATTACK_CLASSES[0]
    failed_attack["fail_closed_decisions_by_attack"]["by_attack"][first_attack][
        "fail_closed"
    ] = False
    _refresh(failed_attack)
    assert failed_attack["online_self_evolution_safety_ready_score"] == 0.0

    bad_status = json.loads(json.dumps(failed_attack))
    bad_status["status"] = "complete_positive"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_csl_6320_deterministic_helpers_and_error_paths(tmp_path: Path) -> None:
    """REQ-CSL-6320-MANIFEST: helper branches stay deterministic."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    first = mod.run(
        date="20260811",
        result_path=output,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=False,
    )
    second = mod.run(
        date="20260811",
        result_path=output,
        duration_s=3.0,
        test_exit_codes=_passing_exit_codes(),
        write=False,
    )
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert mod.sha256_bytes(b"abc").startswith("sha256:")
    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod._path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod._json_loads_object(b"{\"ok\": true}") == {"ok": True}
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("{\"a\": 1}\n\n{\"b\": 2}\n", encoding="utf-8")
    assert mod._jsonl_rows(jsonl_path) == [{"a": 1}, {"b": 2}]

    with pytest.raises(ValueError, match="JSON object"):
        mod._json_loads_object(b"[]")
    with pytest.raises(ValueError, match="JSON"):
        mod._json_loads_object(b"{")
    with pytest.raises(ValueError, match="unknown_attack"):
        mod.expected_decision("unknown_attack")
    with pytest.raises(ValueError, match="forced"):
        mod._require(False, "forced")

    no_tests = json.loads(json.dumps(first))
    no_tests["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 1}
    _refresh(no_tests)
    assert no_tests["online_self_evolution_safety_ready_score"] == 0.0

    for field in (
        "independent_version_registry_reconstruction",
        "fail_closed_decisions_by_attack",
        "rollback_byte_identity",
        "protected_files_unchanged",
        "test_exit_codes",
    ):
        malformed = json.loads(json.dumps(first))
        malformed[field] = []
        assert mod.ready_score(malformed) == 0.0

    malformed_decisions = json.loads(json.dumps(first))
    malformed_decisions["fail_closed_decisions_by_attack"]["by_attack"] = []
    assert mod._all_attacks_fail_closed(malformed_decisions) is False

    malformed_group = json.loads(json.dumps(first))
    malformed_group["dense_signal_inversion_and_release_authority_results"][
        "all_attacks_fail_closed"
    ] = False
    assert mod._all_group_fields_pass(malformed_group) is False

    malformed_rollback = json.loads(json.dumps(first))
    malformed_rollback["rollback_byte_identity"]["receipts"][0]["byte_exact_after_restart"] = False
    assert mod._rollback_identity_passed(malformed_rollback) is False
    assert mod.build_rollback_byte_identity({"rollback_targets": [None]})[
        "all_parent_bytes_match_after_restart"
    ] is False
    assert mod._has_parent_cycle(
        [
            {"version_id": "a", "parent_version_id": "b"},
            {"version_id": "b", "parent_version_id": "a"},
        ]
    ) is True
    assert (
        mod._count_path_receipts(
            [{"path": "x", "sha256": "sha256:x", "present": True}, {"not": "receipt"}]
        )
        == 1
    )

    output = tmp_path / "cli-no-validate.json"
    assert mod.main(["--date", "20260811", "--output", str(output)]) == 0
    assert output.exists()
