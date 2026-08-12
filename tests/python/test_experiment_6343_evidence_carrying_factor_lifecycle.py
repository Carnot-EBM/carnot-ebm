"""Tests for Exp6343 evidence-carrying factor lifecycle.

Spec refs: REQ-LEARN-6343, REQ-LEARN-6343-EVIDENCE,
REQ-LEARN-6343-LIFECYCLE, REQ-LEARN-6343-GATES,
REQ-LEARN-6343-BOUNDS, REQ-LEARN-6343-RESTART,
REQ-LEARN-6343-PROVENANCE, SCENARIO-LEARN-6343-LIFECYCLE,
SCENARIO-LEARN-6343-GATED-MERGE-DELETE,
SCENARIO-LEARN-6343-ATTACKS, SCENARIO-LEARN-6343-BOUNDED,
SCENARIO-LEARN-6343-RESTART-ROLLBACK.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6343_evidence_carrying_factor_lifecycle as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, object]:
    return mod.run(
        date="20260812",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def _read_json(receipt: dict[str, object]) -> dict[str, object]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def test_req_learn_6343_spec_declares_contract_and_principles() -> None:
    """REQ-LEARN-6343-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6343") :]

    for token in (
        "REQ-LEARN-6343-EVIDENCE",
        "REQ-LEARN-6343-LIFECYCLE",
        "REQ-LEARN-6343-GATES",
        "REQ-LEARN-6343-BOUNDS",
        "REQ-LEARN-6343-RESTART",
        "SCENARIO-LEARN-6343-LIFECYCLE",
        "SCENARIO-LEARN-6343-GATED-MERGE-DELETE",
        "SCENARIO-LEARN-6343-ATTACKS",
        "SCENARIO-LEARN-6343-BOUNDED",
        "SCENARIO-LEARN-6343-RESTART-ROLLBACK",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_6343_registry_replay_is_byte_identical(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6343-LIFECYCLE: registry rows replay to the same state."""

    artifact = _artifact(tmp_path)
    registry_receipt = artifact["version_registry_path_and_hash"]
    lifecycle_schema = _read_json(artifact["factor_lifecycle_schema_path_and_hash"])
    evidence_schema = _read_json(artifact["evidence_bundle_schema_path_and_hash"])
    manifest = _read_json(artifact["synthetic_lifecycle_stream_manifest_path_and_hash"])
    rows = mod.read_jsonl(Path(str(registry_receipt["path"])))
    replay = mod.replay_registry_rows(rows)

    assert lifecycle_schema["schema"] == mod.FACTOR_LIFECYCLE_SCHEMA
    assert evidence_schema["schema"] == mod.EVIDENCE_BUNDLE_SCHEMA
    assert manifest["operation_names"] == list(mod.OPERATION_NAMES)
    assert registry_receipt["sha256"] == mod.sha256_file(Path(str(registry_receipt["path"])))
    assert registry_receipt["row_count"] == len(rows)
    assert replay["state_hash"] == artifact["restart_and_byte_exact_rollback_results"][
        "restart_state_hash"
    ]
    assert replay["state_bytes_sha256"] == artifact["restart_and_byte_exact_rollback_results"][
        "restart_state_bytes_sha256"
    ]
    assert replay["registry_hash"] == artifact["restart_and_byte_exact_rollback_results"][
        "registry_hash"
    ]
    assert artifact["factor_add_merge_delete_quarantine_and_restore_results"][
        "all_required_operations_executed"
    ] is True
    for operation in ("add", "retain", "merge", "quarantine", "delete", "restore"):
        assert artifact["factor_add_merge_delete_quarantine_and_restore_results"][
            "operation_counts"
        ][operation] >= 1
    assert any(row["operation"] == "capacity_evict" for row in rows)
    assert rows[0]["previous_row_hash"] == mod.GENESIS_ROW_HASH
    assert all(
        rows[index]["previous_row_hash"] == rows[index - 1]["row_hash"]
        for index in range(1, len(rows))
    )


def test_scenario_learn_6343_gates_retention_bounds_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6343-GATED-MERGE-DELETE: destructive changes need gates."""

    artifact = _artifact(tmp_path)
    lifecycle = artifact["factor_add_merge_delete_quarantine_and_restore_results"]
    exact = artifact["exact_historical_replay_results"]
    retention = artifact["protected_retention_results"]
    bounds = artifact["bounded_memory_growth_results"]
    rollback = artifact["restart_and_byte_exact_rollback_results"]
    remembering = artifact["catastrophic_remembering_event_definition_and_counts"]

    assert exact["all_state_changes_checked"] is True
    assert exact["all_committed_state_changes_passed"] is True
    assert retention["all_protected_retention_passed"] is True
    assert retention["protected_regression_count"] == 0
    assert bounds["max_active_count"] <= mod.ACTIVE_FACTOR_CAPACITY
    assert bounds["max_quarantine_count"] <= mod.QUARANTINE_FACTOR_CAPACITY
    assert bounds["capacity_eviction_count"] >= 1
    assert bounds["deterministic_compaction"] is True
    assert rollback["restart_byte_identical"] is True
    assert rollback["all_destructive_rollbacks_byte_identical"] is True
    assert remembering["catastrophic_remembering_event_count"] == 0
    for row in lifecycle["valid_operation_receipts"]:
        if row["operation"] in {"merge", "delete"}:
            checks = row["operation_checks"]
            assert checks["exact_historical_replay_passed"] is True
            assert checks["protected_retention_passed"] is True
            assert checks["byte_identical_rollback_passed"] is True


def test_scenario_learn_6343_attacks_and_store_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6343-ATTACKS: invalid evidence does not mutate state."""

    artifact = _artifact(tmp_path)
    attacks = artifact[
        "stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results"
    ]
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["mutated_attack_count"] == 0
    for name in (
        "stale_certificate",
        "circular_parent",
        "cross_family_evidence",
        "duplicate_evidence",
        "rationale_only_evidence",
        "witness_swap",
        "harmful_merge",
        "harmful_deletion",
    ):
        assert attacks[name]["fail_closed"] is True
        assert attacks[name]["mutated"] is False

    store = mod.LifecycleStore()
    added = store.apply_event(mod.lifecycle_event("add", "accept_guard", event_index=0))
    assert added["accepted"] is True
    duplicate = store.try_apply_event(mod.lifecycle_event("add", "accept_guard", event_index=1))
    assert duplicate["reason"] == "duplicate_evidence"
    stale = mod.lifecycle_event("add", "stale_certificate_guard", event_index=2)
    stale["evidence"]["release_certificate"]["ledger_state_hash"] = "sha256:stale"
    assert store.try_apply_event(stale)["reason"] == "stale_certificate"
    circular = mod.lifecycle_event("add", "cycle_guard", event_index=3)
    circular["evidence"]["parent_version"] = "cycle_guard:v001"
    assert store.try_apply_event(circular)["reason"] == "circular_lineage"
    cross_family = mod.lifecycle_event("add", "cross_family_guard", event_index=4)
    cross_family["evidence"]["family_id"] = "other_family"
    assert store.try_apply_event(cross_family)["reason"] == "cross_family_evidence"
    rationale_only = mod.lifecycle_event("add", "rationale_only_guard", event_index=5)
    rationale_only["evidence"].pop("replay_witness")
    assert store.try_apply_event(rationale_only)["reason"] == "rationale_only_evidence"
    swapped = mod.lifecycle_event("add", "witness_swap_guard", event_index=6)
    swapped["evidence"]["replay_witness"]["case_id"] = "case_repair_01"
    assert store.try_apply_event(swapped)["reason"] == "witness_swap"
    with pytest.raises(ValueError, match="unsupported_operation"):
        store.apply_event(mod.lifecycle_event("unsupported", "accept_guard", event_index=7))


def test_req_learn_6343_cli_schema_checksum_and_readiness(tmp_path: Path) -> None:
    """REQ-LEARN-6343-PROVENANCE: CLI writes a valid terminal artifact."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260812", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert artifact["generated_label_count"] == 0
    assert type(artifact["generated_label_count"]) is int
    assert artifact["llm_call_count"] == 0
    assert type(artifact["llm_call_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["evidence_factor_lifecycle_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_zero = json.loads(json.dumps(artifact))
    bad_zero["llm_call_count"] = True
    _refresh(bad_zero)
    with pytest.raises(ValueError, match="llm_call_count"):
        mod.validate_artifact(bad_zero)

    failed_attack = json.loads(json.dumps(artifact))
    failed_attack[
        "stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results"
    ]["all_attacks_fail_closed"] = False
    _refresh(failed_attack)
    assert failed_attack["evidence_factor_lifecycle_ready_score"] == 0.0

    bad_status = json.loads(json.dumps(failed_attack))
    bad_status["status"] = "complete_positive"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_learn_6343_properties_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6343-EVIDENCE: helpers enforce the evidence contract."""

    artifact = _artifact(tmp_path, write=False)

    assert mod.sha256_json({"ok": True}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod._rounded(1.2345678912349) == 1.234567891235
    assert mod._as_mapping([]) == {}
    with pytest.raises(ValueError, match="forced"):
        mod._require(False, "forced")
    with pytest.raises(ValueError, match="unknown_factor_template"):
        mod.factor_template("missing")

    accepted_store = mod.LifecycleStore()
    accepted = accepted_store.try_apply_event(
        mod.lifecycle_event("add", "repair_guard", event_index=20)
    )
    assert accepted["accepted"] is True
    assert accepted["mutated"] is True

    alias = dict(mod.factor_template("accept_guard"))
    alias["factor_id"] = "alias_guard"
    monkeypatch.setitem(mod.FACTOR_TEMPLATES, "alias_key", alias)
    assert mod.factor_template("alias_guard")["factor_id"] == "alias_guard"

    mismatch_store = mod.LifecycleStore()
    affected_mismatch = mod.lifecycle_event("add", "accept_guard", event_index=21)
    affected_mismatch["evidence"]["affected_variables"] = ["repair_cue"]
    assert mismatch_store.try_apply_event(affected_mismatch)["reason"] == (
        "affected_variables_mismatch"
    )

    stale_parent = mod.lifecycle_event("add", "accept_guard", event_index=22)
    stale_parent["evidence"]["evidence_identity"] = "unique:stale_parent"
    stale_parent["evidence"]["parent_version"] = "missing:v001"
    assert mod.LifecycleStore().try_apply_event(stale_parent)["reason"] == (
        "stale_parent_version"
    )

    missing_rollback = mod.lifecycle_event("add", "accept_guard", event_index=23)
    missing_rollback["evidence"]["evidence_identity"] = "unique:rollback"
    missing_rollback["evidence"]["rollback_target"] = "missing:v001"
    assert mod.LifecycleStore().try_apply_event(missing_rollback)["reason"] == (
        "rollback_target_missing"
    )

    already_active = mod.LifecycleStore()
    already_active.apply_event(mod.lifecycle_event("add", "accept_guard", event_index=24))
    duplicate_factor = mod.lifecycle_event("add", "accept_guard", event_index=25)
    duplicate_factor["evidence"]["evidence_identity"] = "unique:already_active"
    assert already_active.try_apply_event(duplicate_factor)["reason"] == (
        "factor_already_active"
    )

    with pytest.raises(ValueError, match="unsupported_operation"):
        mod.LifecycleStore()._mutate("bad", {}, {"factor_id": "accept_guard"})

    replay_failure_store = mod.LifecycleStore()
    replay_failure_store.apply_event(mod.lifecycle_event("add", "accept_guard", event_index=26))
    replay_failure_store.active["accept_guard"]["prediction"] = "repair"
    assert replay_failure_store.exact_replay_receipt()["failure_count"] == 1
    assert replay_failure_store.protected_retention_receipt()["failure_count"] == 1
    assert (
        mod.factor_prediction(
            mod.factor_record(
                mod.lifecycle_event("add", "accept_guard", event_index=27)["evidence"],
                0,
            ),
            mod.HISTORICAL_CASES["case_repair_01"],
        )
        == "abstain"
    )

    evidence = mod.lifecycle_event("add", "accept_guard", event_index=28)["evidence"]
    template = mod.factor_template("accept_guard")
    unknown_case = json.loads(json.dumps(evidence))
    unknown_case["replay_witness"]["case_id"] = "missing"
    assert mod.witness_rejection_reason(unknown_case, template) == "witness_unknown_case"
    swapped_case = json.loads(json.dumps(evidence))
    swapped_case["replay_witness"]["case_id"] = "case_repair_01"
    swapped_case["replay_witness"]["case_hash"] = mod.CASE_HASHES["case_repair_01"]
    assert mod.witness_rejection_reason(swapped_case, template) == "witness_swap"
    cross_family = json.loads(json.dumps(evidence))
    cross_family["family_id"] = "other_family"
    assert mod.witness_rejection_reason(cross_family, template) == "cross_family_evidence"
    exact_failed = json.loads(json.dumps(evidence))
    exact_failed["replay_witness"]["observed"] = "reject"
    assert mod.witness_rejection_reason(exact_failed, template) == "exact_replay_failed"
    bad_counterexample = json.loads(json.dumps(evidence))
    bad_counterexample["minimized_exact_counterexample"]["case_hash"] = "sha256:bad"
    assert mod.witness_rejection_reason(bad_counterexample, template) == "counterexample_hash"

    for field in (
        "factor_add_merge_delete_quarantine_and_restore_results",
        "exact_historical_replay_results",
        "protected_retention_results",
        "bounded_memory_growth_results",
        "restart_and_byte_exact_rollback_results",
        "stale_circular_cross_family_duplicate_and_rationale_laundering_attack_results",
        "test_exit_codes",
        "protected_files_unchanged",
    ):
        malformed = json.loads(json.dumps(artifact))
        malformed[field] = []
        assert mod.ready_score(malformed) == 0.0

    no_tests = json.loads(json.dumps(artifact))
    no_tests["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 1}
    _refresh(no_tests)
    assert no_tests["evidence_factor_lifecycle_ready_score"] == 0.0

    tampered_rows = mod.build_version_registry().rows
    tampered_rows[0]["evidence_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="evidence_hash"):
        mod.replay_registry_rows(tampered_rows)

    output = tmp_path / "cli-no-validate.json"
    assert mod.main(["--date", "20260812", "--output", str(output)]) == 0
    assert output.exists()
