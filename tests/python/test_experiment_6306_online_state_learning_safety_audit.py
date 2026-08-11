"""Tests for Exp6306 online state learning safety audit.

Spec refs: REQ-CSL-6306, REQ-CSL-6306-INDEPENDENCE,
REQ-CSL-6306-FAULTS, REQ-CSL-6306-AUDIT,
REQ-CSL-6306-LEAKAGE, REQ-CSL-6306-ROLLBACK,
REQ-CSL-6306-PROVENANCE, SCENARIO-CSL-6306-RECONSTRUCT,
SCENARIO-CSL-6306-FAIL-CLOSED, SCENARIO-CSL-6306-APPEND-ONLY,
SCENARIO-CSL-6306-ROLLBACK.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6306_online_state_learning_safety_audit as mod
from carnot.terminal_artifacts import path_sha256


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
UPSTREAM = REPO / mod.EXP6304_RELATIVE_PATH


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


def _reload(path_text: str) -> dict[str, object]:
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def test_req_csl_6306_spec_declares_safety_contract() -> None:
    """REQ-CSL-6306-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CSL-6306") :]

    for token in (
        "REQ-CSL-6306-INDEPENDENCE",
        "REQ-CSL-6306-FAULTS",
        "REQ-CSL-6306-AUDIT",
        "REQ-CSL-6306-LEAKAGE",
        "REQ-CSL-6306-ROLLBACK",
        "REQ-CSL-6306-PROVENANCE",
        "SCENARIO-CSL-6306-RECONSTRUCT",
        "SCENARIO-CSL-6306-FAIL-CLOSED",
        "SCENARIO-CSL-6306-APPEND-ONLY",
        "SCENARIO-CSL-6306-ROLLBACK",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_csl_6306_reconstructs_pinned_upstream_before_faults(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6306-RECONSTRUCT: pinned bytes match before injection."""

    artifact = _artifact(tmp_path)
    upstream = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    receipts = artifact["snapshot_and_log_reconstruction_receipts"]
    utility = artifact["producer_utility_determination_preserved"]
    independence = artifact["evaluator_independence_receipts"]

    assert artifact["upstream_path_hash_and_terminal_class"]["exp6304"]["sha256"] == path_sha256(
        UPSTREAM
    )
    assert artifact["upstream_path_hash_and_terminal_class"]["exp6304"][
        "terminal_class"
    ] == "positive"
    assert artifact["upstream_path_hash_and_terminal_class"]["exp6298"]["terminal"] is True
    assert receipts["reconstructed_before_fault_injection"] is True
    assert receipts["all_byte_identities_match"] is True
    assert receipts["predecision_snapshot_count"] == 80
    assert receipts["postdecision_outcome_count"] == 16
    assert receipts["manifest_event_count"] == 16
    assert all(row["byte_identity"] is True for row in receipts["sidecar_receipts"])
    assert utility["upstream_honest_verdict"] == upstream["honest_verdict"]
    assert utility["verbatim_honest_verdict"] == upstream["honest_verdict"]
    assert utility["preserved"] is True
    assert independence["uses_copied_temporary_state"] is True
    assert independence["producer_readiness_trusted_as_safety_authority"] is False
    assert independence["canonical_exp6304_outputs_mutated"] is False


def test_scenario_csl_6306_every_injection_fails_closed_and_appends(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6306-FAIL-CLOSED: unsafe commits and leaks stay zero."""

    artifact = _artifact(tmp_path)
    manifest_receipt = artifact["injection_manifest_path_and_hash"]
    manifest = _reload(str(manifest_receipt["path"]))

    assert manifest_receipt["sha256"] == path_sha256(Path(str(manifest_receipt["path"])))
    assert manifest["fault_count"] == len(mod.FAULT_CLASSES)
    assert [row["fault_class"] for row in manifest["faults"]] == list(mod.FAULT_CLASSES)
    assert artifact["unsafe_commit_count"] == 0
    assert type(artifact["unsafe_commit_count"]) is int
    assert artifact["predecision_leak_count"] == 0
    assert type(artifact["predecision_leak_count"]) is int
    assert artifact["base_model_mutation_count"] == 0
    assert type(artifact["base_model_mutation_count"]) is int
    assert artifact["audit_log_mutation_count"] == 0
    assert type(artifact["audit_log_mutation_count"]) is int

    for field in mod.FAULT_RESULT_FIELDS:
        result = artifact[field]
        assert result["fault_class"] in mod.FAULT_CLASSES
        assert result["fail_closed"] is True
        assert result["unsafe_commit_count"] == 0
        assert result["predecision_leak_count"] == 0
        assert result["base_model_mutation_count"] == 0
        assert result["audit_append_only"] is True
        assert result["audit_prefix_preserved"] is True
        assert result["byte_exact_rollback"] is True

    assert artifact["false_pass_results"]["claimed_exact_pass"] is True
    assert artifact["missing_validator_results"]["validator_present"] is False
    assert artifact["nonfinite_update_results"]["nonfinite_update_detected"] is True
    assert artifact["poison_results"]["terminal_action"] == "quarantine"


def test_scenario_csl_6306_rollback_restart_and_safety_gate(tmp_path: Path) -> None:
    """SCENARIO-CSL-6306-ROLLBACK: rollback counts are byte exact."""

    artifact = _artifact(tmp_path)
    rollback = artifact["byte_exact_rollback_count_and_expected"]
    safety = artifact["safety_determination"]
    promotion = artifact["safety_cannot_promote_utility_receipt"]

    assert rollback["actual"] == rollback["expected"] == len(mod.FAULT_CLASSES)
    assert all(row["byte_exact"] is True for row in rollback["receipts"])
    assert artifact["restart_results"]["restart_identity"] is True
    assert artifact["rollback_results"]["explicit_rollback_request_honored"] is True
    assert artifact["online_learning_safety_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert safety["safety_ready"] is True
    assert safety["producer_utility_is_safety_authority"] is False
    assert promotion["utility_output_ready_score"] == promotion["utility_input_ready_score"]
    assert promotion["safety_only_promotion_blocked"] is True

    unsafe = json.loads(json.dumps(artifact))
    unsafe["unsafe_commit_count"] = 1
    mod.refresh_terminal_fields(unsafe)
    assert unsafe["online_learning_safety_ready_score"] == 0.0
    assert unsafe["status"] == "complete_null"
    with pytest.raises(ValueError, match="unsafe_commit_count"):
        mod.validate_artifact({**artifact, "unsafe_commit_count": True})

    bad_rollback = json.loads(json.dumps(artifact))
    bad_rollback["byte_exact_rollback_count_and_expected"]["actual"] -= 1
    mod.refresh_terminal_fields(bad_rollback)
    assert bad_rollback["online_learning_safety_ready_score"] == 0.0


def test_req_csl_6306_schema_checksum_cli_and_validation(tmp_path: Path) -> None:
    """REQ-CSL-6306: artifact fields, checksum, and CLI stay stable."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260811", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["test_commands"] == list(mod.DEFAULT_TEST_COMMANDS)
    assert all(code == 0 for code in artifact["test_exit_codes"].values())
    assert artifact["protected_files_unchanged"]["unchanged"] is True

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "running"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_csl_6306_defensive_edges_and_unknown_faults(tmp_path: Path) -> None:
    """REQ-CSL-6306-FAULTS: malformed helper paths fail closed."""

    assert mod.sha256_bytes(b"abc").startswith("sha256:")
    assert mod._json_loads_object(b"{\"ok\": true}") == {"ok": True}
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("{\"a\": 1}\n\n{\"b\": 2}\n", encoding="utf-8")
    assert mod._jsonl_rows(jsonl_path) == [{"a": 1}, {"b": 2}]
    with pytest.raises(ValueError, match="JSON object"):
        mod._json_loads_object(b"[]")
    with pytest.raises(ValueError, match="JSON"):
        mod._json_loads_object(b"{")
    with pytest.raises(ValueError, match="unknown_fault"):
        mod._fault_payload("unknown_fault", 1)
    with pytest.raises(ValueError, match="unknown_fault"):
        mod._expected_action("unknown_fault")
    with pytest.raises(ValueError, match="unknown_fault"):
        mod._evaluate_fault("unknown_fault", b"{}", b"", 1)
    assert mod._relative_or_absolute(tmp_path / "outside.txt").endswith("outside.txt")

    artifact = _artifact(tmp_path, write=False)
    no_producer = json.loads(json.dumps(artifact))
    no_producer["producer_utility_determination_preserved"] = []
    mod.refresh_terminal_fields(no_producer)
    assert "safety_cannot_promote_utility_receipt" in no_producer

    for malformed_field in (
        "test_exit_codes",
        "snapshot_and_log_reconstruction_receipts",
        "safety_cannot_promote_utility_receipt",
        "protected_files_unchanged",
    ):
        malformed = json.loads(json.dumps(artifact))
        malformed[malformed_field] = []
        assert mod.ready_score(malformed) == 0.0

    no_tests = json.loads(json.dumps(artifact))
    no_tests["test_exit_codes"] = {list(mod.DEFAULT_TEST_COMMANDS)[0]: 2}
    mod.refresh_terminal_fields(no_tests)
    assert no_tests["online_learning_safety_ready_score"] == 0.0

    for field, key, value in (
        ("false_pass_results", None, None),
        ("false_pass_results", "fail_closed", False),
        ("false_pass_results", "byte_exact_rollback", False),
        ("false_pass_results", "audit_append_only", False),
        ("false_pass_results", "unsafe_commit_count", 1),
    ):
        mutated = json.loads(json.dumps(artifact))
        if key is None:
            mutated[field] = []
        else:
            mutated[field][key] = value
        assert mod._all_faults_fail_closed(mutated) is False

    no_rollback = json.loads(json.dumps(artifact))
    no_rollback["byte_exact_rollback_count_and_expected"] = []
    assert mod._rollback_counts_match(no_rollback) is False

    output = tmp_path / "cli-no-validate.json"
    assert mod.main(["--date", "20260811", "--output", str(output)]) == 0
    assert output.exists()
