"""Tests for Exp5825 certified adaptive memory event/state contract.

Spec refs: REQ-LEARN-5825, SCENARIO-LEARN-5825-ADAPTERS,
SCENARIO-LEARN-5825-FAIL-CLOSED, SCENARIO-LEARN-5825-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5825_certified_adaptive_memory_contract as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5825_certified_adaptive_memory_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5825_certified_adaptive_memory_contract.py "
    "-m pytest tests/python/test_experiment_5825_certified_adaptive_memory_contract.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5825_certified_adaptive_memory_contract.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
ROADMAP_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5824_v519_source_delta_ingestion.py "
    "-q --no-cov -n 0"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ROOT_CLUTTER_COMMAND,
    ROADMAP_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _refresh_event(event: dict[str, Any]) -> dict[str, Any]:
    event["payload_hash"] = mod.sha256_json(event["payload"])
    event["event_hash"] = mod.canonical_event_hash(event)
    event["event_id"] = mod.expected_event_id(event)
    return event


def test_req_learn_5825_spec_declares_contract_fields_and_principles() -> None:
    """REQ-LEARN-5825: OpenSpec anchors schema fields, adapters, and ready gate."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5825") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5825",
        "SCENARIO-LEARN-5825-ADAPTERS",
        "SCENARIO-LEARN-5825-FAIL-CLOSED",
        "SCENARIO-LEARN-5825-ARTIFACT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`adaptive_memory_contract_ready_score`",
    ):
        assert marker in section
    for event_type in mod.REQUIRED_EVENT_TYPES:
        assert event_type.replace("_", " ") in normalized or event_type in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5825_artifact_ready_and_written(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5825-ARTIFACT: clean receipts produce bare readiness 1.0."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )
    written = mod.build_and_write_artifact(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    assert mod.validate_artifact(artifact) is True
    assert mod.validate_artifact(written) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == written
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["adaptive_memory_contract_ready_score"] == 1.0
    assert isinstance(artifact["adaptive_memory_contract_ready_score"], float)
    assert artifact["schema_errors"] == []
    assert artifact["duration_s"] == 1.25
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["output_paths"]["result_writable"] is True
    assert artifact["preconditions_checked"]["resources"]["disk"]["ok"] is True
    assert artifact["preconditions_checked"]["resources"]["memory"]["ok"] is True
    assert artifact["preconditions_checked"]["upstream_row_counts"] == {
        "exp5761_instances": 120,
        "exp5785_rows": 360,
    }
    assert all(value.startswith("sha256:") for value in artifact["upstream_artifact_hashes"].values())
    assert artifact["canonical_event_schema"]["event_types"] == list(mod.REQUIRED_EVENT_TYPES)
    assert artifact["canonical_state_schema"]["identity_rule"] == mod.STATE_IDENTITY_RULE
    assert artifact["chronology_and_visibility_checks"]["all_passed"] is True
    assert artifact["chronology_and_visibility_checks"]["hidden_label_access_count"] == 0
    assert all(
        receipt["round_trip_ok"] is True
        for receipt in artifact["adapter_round_trip_receipts"]["adapters"].values()
    )
    assert all(
        receipt["passed"] is True
        for receipt in artifact["adversarial_contract_results"].values()
    )
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_learn_5825_adapters_round_trip_every_available_source() -> None:
    """SCENARIO-LEARN-5825-ADAPTERS: every upstream row/receipt has a canonical event."""

    bundle = mod.adapt_all_upstreams(REPO)
    events = bundle["events"]
    states = bundle["states"]
    receipts = bundle["receipts"]

    assert mod.validate_event_stream(events, states) == []
    assert len({event["event_id"] for event in events}) == len(events)
    assert len({state["state_id"] for state in states}) == len(states)
    assert all(event["event_hash"] == mod.canonical_event_hash(event) for event in events)
    assert all(state["state_hash"] == mod.canonical_state_hash(state) for state in states)
    assert [event["causal_sequence_index"] for event in events] == sorted(
        event["causal_sequence_index"] for event in events
    )
    assert set(event["event_type"] for event in events) >= set(mod.REQUIRED_EVENT_TYPES)
    assert receipts["adapters"]["exp5761"]["source_row_count"] == 120
    assert receipts["adapters"]["exp5761"]["canonical_event_count"] == 600
    assert receipts["adapters"]["exp5762"]["source_event_count"] == 520
    assert receipts["adapters"]["exp5763"]["source_event_count"] == 288
    assert receipts["adapters"]["exp5785"]["source_row_count"] == 360
    assert receipts["total_canonical_event_count"] == len(events)
    assert receipts["total_canonical_state_count"] == len(states)


def test_scenario_learn_5825_adversarial_fixtures_fail_closed() -> None:
    """SCENARIO-LEARN-5825-FAIL-CLOSED: adversarial event/state mutations are rejected."""

    results = mod.adversarial_contract_results()

    assert set(results) == set(mod.REQUIRED_ADVERSARIAL_CASES)
    assert all(receipt["passed"] is True for receipt in results.values())
    assert results["leakage"]["error_code"] == "hidden_science_label_access"
    assert results["forged_oracle_labels"]["error_code"] == "forged_oracle_label"
    assert results["collision_without_split"]["error_code"] == "collision_without_split"
    assert results["stale_supersession"]["error_code"] == "stale_supersession"
    assert results["rollback_mismatch"]["error_code"] == "rollback_mismatch"
    assert results["missing_protected_prefix_evidence"]["error_code"] == (
        "missing_protected_prefix_evidence"
    )

    bundle = mod.adapt_all_upstreams(REPO)
    events = bundle["events"]
    states = bundle["states"]
    bad = deepcopy(events)
    bad[3]["causal_sequence_index"] = bad[2]["causal_sequence_index"]
    errors = mod.validate_event_stream(bad, states)

    assert errors
    assert errors[0]["error_code"] == "non_monotone_chronology"
    with pytest.raises(mod.ContractValidationError, match="non_monotone_chronology"):
        mod.assert_valid_event_stream(bad, states)


def test_req_learn_5825_missing_upstream_writes_terminal_blocked_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-5825: missing upstream evidence blocks instead of fabricating readiness."""

    artifact = mod.build_and_write_artifact(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["adaptive_memory_contract_ready_score"] == 0.0
    assert "missing_upstream_artifact" in artifact["schema_errors"]
    assert artifact["preconditions_checked"]["preconditions_ready"] is False
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_5825_artifact_validation_rejects_tampering() -> None:
    """SCENARIO-LEARN-5825-ARTIFACT: checksum, fields, gates, and exits are audited."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    for mutate, match in (
        (lambda item: item.pop("status"), "missing required artifact fields"),
        (
            lambda item: item.update({"inference_substrate": "live_llm_inference"}),
            "inference_substrate",
        ),
        (
            lambda item: item.update({"adaptive_memory_contract_ready_score": 0.0}),
            "adaptive_memory_contract_ready_score",
        ),
        (lambda item: item["schema_errors"].append("late_error"), "adaptive_memory_contract_ready_score"),
        (
            lambda item: item["chronology_and_visibility_checks"].update({"all_passed": False}),
            "adaptive_memory_contract_ready_score",
        ),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance"),
        (lambda item: item.update({"honest_verdict": "ready"}), "honest_verdict"),
        (
            lambda item: item["test_exit_codes"].update({TEST_COMMAND: 1}),
            "adaptive_memory_contract_ready_score",
        ),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if "reproducibility_checksum" in bad and match != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)


def test_req_learn_5825_precondition_and_io_edge_cases_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5825: malformed upstreams and unsafe local gates are explicit."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{\"ok\": true}\n", encoding="utf-8")
    scalar_jsonl = tmp_path / "scalar.jsonl"
    scalar_jsonl.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)
    assert mod._read_jsonl(blank_jsonl) == [{"ok": True}]
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(scalar_jsonl)
    assert mod._source_hash({"receipt_hash": "sha256:receipt"}) == "sha256:receipt"
    assert mod._source_hash({"unhashed": "payload"}).startswith("sha256:")

    present_hashes = {
        name: "sha256:" + name for name in mod.UPSTREAM_PATHS
    }
    monkeypatch.setattr(
        mod,
        "_hash_path",
        lambda root, relative: present_hashes[
            next(name for name, path in mod.UPSTREAM_PATHS.items() if path == relative)
        ],
    )
    monkeypatch.setattr(mod, "_load_upstream_bundle", lambda root: (_ for _ in ()).throw(ValueError("bad")))
    corrupt = mod.collect_preconditions(tmp_path, result_path=tmp_path / "out.json")
    assert "corrupt_upstream_artifact" in corrupt["blocked_reasons"]
    assert corrupt["corrupt_upstream_errors"] == ["ValueError"]

    bad_artifacts = {
        "exp5761": {
            "status": "blocked",
            "honest_verdict": "not-terminal",
            "verifier_is_oracle": False,
            "instance_count": 2,
            "solver_versions": {},
            "split_manifest": {},
        },
        "exp5762": {
            "status": "complete",
            "honest_verdict": "complete: ok",
            "verifier_is_oracle": True,
            "inference_substrate": "oracle",
            "science_split_hash": "",
        },
        "exp5763": {
            "status": "complete",
            "honest_verdict": "complete: ok",
            "verifier_is_oracle": True,
            "query_label_receipts": [],
            "stream_root_hash": "",
        },
        "exp5785": {
            "status": "complete",
            "honest_verdict": "complete: ok",
            "verifier_is_oracle": True,
            "row_file_sha256": "sha256:different",
            "exact_validator_receipts": [],
            "chronological_split_receipts": {},
        },
    }
    bad_rows = {"exp5761_instances": [{"row_hash": "sha256:a"}], "exp5785_rows": []}
    monkeypatch.setattr(mod, "_load_upstream_bundle", lambda root: (bad_artifacts, bad_rows))
    monkeypatch.setattr(
        mod,
        "_memory_probe",
        lambda: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    monkeypatch.setattr(
        mod,
        "_disk_probe",
        lambda root: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    blocked = mod.collect_preconditions(
        tmp_path,
        result_path=tmp_path / "missing" / "out.json",
    )

    assert set(blocked["blocked_reasons"]) >= {
        "upstream_status_not_complete",
        "upstream_honest_verdict_not_terminal",
        "upstream_verifier_not_oracle",
        "exp5761_row_count_mismatch",
        "exp5785_row_hash_mismatch",
        "missing_exact_validator_versions",
        "missing_split_hashes",
        "insufficient_free_ram",
        "insufficient_free_disk",
        "output_path_not_writable",
    }


def test_scenario_learn_5825_schema_validation_edge_cases() -> None:
    """SCENARIO-LEARN-5825-FAIL-CLOSED: low-level schema defects get typed errors."""

    events, states = mod._positive_control_stream()
    assert mod.assert_valid_event_stream(events, states) is True

    duplicate_state_errors = mod.validate_event_stream(events, [states[0], states[0]])
    assert any(error["error_code"] == "duplicate_state_identity" for error in duplicate_state_errors)

    bad_state = deepcopy(states[0])
    bad_state["source_artifact_hash"] = "missing"
    bad_state["state_hash"] = mod.canonical_state_hash(bad_state)
    errors = mod.validate_event_stream(events, [bad_state, states[1]])
    assert any(error["error_code"] == "missing_hash" for error in errors)

    bad_state_hash = deepcopy(states[0])
    bad_state_hash["state_hash"] = "sha256:bad"
    errors = mod.validate_event_stream(events, [bad_state_hash, states[1]])
    assert any(error["error_code"] == "state_hash_mismatch" for error in errors)

    bad_state_id = deepcopy(states[0])
    bad_state_id["state_id"] = "state::wrong"
    errors = mod.validate_event_stream(events, [bad_state_id, states[1]])
    assert any(error["error_code"] == "state_id_mismatch" for error in errors)

    bad_state_row_hash = deepcopy(states[0])
    bad_state_row_hash["source_row_hash"] = ""
    bad_state_row_hash["state_hash"] = mod.canonical_state_hash(bad_state_row_hash)
    errors = mod.validate_event_stream(events, [bad_state_row_hash, states[1]])
    assert any(error["error_code"] == "missing_hash" for error in errors)

    duplicate_event = deepcopy(events[0])
    duplicate_event["causal_sequence_index"] = 1
    duplicate_event["event_hash"] = mod.canonical_event_hash(duplicate_event)
    errors = mod.validate_event_stream([events[0], duplicate_event], states)
    assert any(error["error_code"] == "duplicate_event_identity" for error in errors)

    bad_event_id = deepcopy(events[0])
    bad_event_id["event_id"] = "event::wrong"
    errors = mod.validate_event_stream([bad_event_id], states)
    assert any(error["error_code"] == "event_id_mismatch" for error in errors)

    stale_payload_hash = deepcopy(events[0])
    stale_payload_hash["payload"]["extra"] = "tamper"
    stale_payload_hash["event_hash"] = mod.canonical_event_hash(stale_payload_hash)
    errors = mod.validate_event_stream([stale_payload_hash], states)
    assert any(error["error_code"] == "payload_hash_mismatch" for error in errors)

    bad_event = deepcopy(events[0])
    bad_event["event_type"] = "not_supported"
    _refresh_event(bad_event)
    errors = mod.validate_event_stream([bad_event], states)
    assert any(error["error_code"] == "unsupported_event_type" for error in errors)

    bad_visibility = deepcopy(events[0])
    bad_visibility["visibility"] = "hidden"
    _refresh_event(bad_visibility)
    errors = mod.validate_event_stream([bad_visibility], states)
    assert any(error["error_code"] == "invalid_visibility" for error in errors)

    missing_hash = deepcopy(events[0])
    missing_hash["source_row_hash"] = ""
    missing_hash["event_hash"] = mod.canonical_event_hash(missing_hash)
    errors = mod.validate_event_stream([missing_hash], states)
    assert any(error["error_code"] == "missing_hash" for error in errors)

    missing_state = deepcopy(events[0])
    missing_state["resulting_state_id"] = "state::missing"
    missing_state["event_hash"] = mod.canonical_event_hash(missing_state)
    errors = mod.validate_event_stream([missing_state], states)
    assert any(error["error_code"] == "missing_resulting_state" for error in errors)

    missing_parent = deepcopy(events[0])
    missing_parent["parent_state_id"] = "state::missing-parent"
    missing_parent["event_hash"] = mod.canonical_event_hash(missing_parent)
    errors = mod.validate_event_stream([missing_parent], states)
    assert any(error["error_code"] == "missing_parent_state" for error in errors)

    parent_mismatch_states = deepcopy(states)
    parent_mismatch_states[1]["parent_state_hash"] = mod.sha256_text("wrong-parent")
    parent_mismatch_states[1]["state_hash"] = mod.canonical_state_hash(parent_mismatch_states[1])
    parent_mismatch_states[1]["state_id"] = mod.expected_state_id(parent_mismatch_states[1])
    parent_mismatch = deepcopy(events[0])
    parent_mismatch["resulting_state_id"] = parent_mismatch_states[1]["state_id"]
    parent_mismatch["event_hash"] = mod.canonical_event_hash(parent_mismatch)
    parent_mismatch["event_id"] = mod.expected_event_id(parent_mismatch)
    errors = mod.validate_event_stream([parent_mismatch], parent_mismatch_states)
    assert any(error["error_code"] == "parent_state_hash_mismatch" for error in errors)

    no_receipt_states = deepcopy(states)
    no_receipt_states[1]["mutation_receipt_hash"] = ""
    no_receipt_states[1]["state_hash"] = mod.canonical_state_hash(no_receipt_states[1])
    no_receipt_states[1]["state_id"] = mod.expected_state_id(no_receipt_states[1])
    no_receipt = deepcopy(events[0])
    no_receipt["resulting_state_id"] = no_receipt_states[1]["state_id"]
    no_receipt["event_hash"] = mod.canonical_event_hash(no_receipt)
    no_receipt["event_id"] = mod.expected_event_id(no_receipt)
    errors = mod.validate_event_stream([no_receipt], no_receipt_states)
    assert any(error["error_code"] == "state_mutation_without_receipt" for error in errors)

    collision_no_receipt = deepcopy(events[0])
    collision_no_receipt["event_type"] = "collision_split"
    _refresh_event(collision_no_receipt)
    errors = mod.validate_event_stream([collision_no_receipt], states)
    assert any(error["error_code"] == "collision_without_split" for error in errors)

    rollback_missing_hashes = deepcopy(events[0])
    rollback_missing_hashes["event_type"] = "rollback"
    _refresh_event(rollback_missing_hashes)
    errors = mod.validate_event_stream([rollback_missing_hashes], states)
    assert any(error["error_code"] == "rollback_mismatch" for error in errors)

    supersession_missing_active = deepcopy(events[0])
    supersession_missing_active["event_type"] = "supersession"
    _refresh_event(supersession_missing_active)
    errors = mod.validate_event_stream([supersession_missing_active], states)
    assert any(error["error_code"] == "stale_supersession" for error in errors)

    with pytest.raises(ValueError, match="unsupported_event_type"):
        mod.make_event(
            event_type="not_supported",
            source_adapter="x",
            sequence=0,
            source_artifact="x",
            source_artifact_hash=mod.sha256_text("artifact"),
            source_hash=mod.sha256_text("row"),
            visibility="train",
            axes={"family": "x"},
            payload={},
            parent_state=states[0],
            resulting_state=states[1],
            oracle_provenance={"authority": "exact_solver_or_validator"},
        )

    bad_chronology = deepcopy(events)
    bad_chronology.append(deepcopy(events[0]))
    checks = mod.chronology_and_visibility_checks(bad_chronology, states)
    assert checks["all_passed"] is False
    assert checks["schema_error_counts"]["non_monotone_chronology"] >= 1


def test_req_learn_5825_validation_and_wrapper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LEARN-5825: artifact validators and CLI wrappers have deterministic behavior."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )

    for mutate, match in (
        (lambda item: item.update({"status": "ready"}), "status"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"field_provenance": []}), "field_provenance"),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        bad["adaptive_memory_contract_ready_score"] = (
            mod.adaptive_memory_contract_ready_score_from_artifact(bad)
        )
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    complete_with_blocked_verdict = deepcopy(artifact)
    complete_with_blocked_verdict["schema_errors"] = ["forced"]
    complete_with_blocked_verdict["adaptive_memory_contract_ready_score"] = 0.0
    complete_with_blocked_verdict["honest_verdict"] = "blocked: forced"
    complete_with_blocked_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        complete_with_blocked_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(complete_with_blocked_verdict)

    blocked_with_complete_verdict = deepcopy(artifact)
    blocked_with_complete_verdict["status"] = "blocked"
    blocked_with_complete_verdict["adaptive_memory_contract_ready_score"] = 0.0
    blocked_with_complete_verdict["honest_verdict"] = "complete: forced"
    blocked_with_complete_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        blocked_with_complete_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_with_complete_verdict)

    monkeypatch.setattr(
        mod,
        "build_and_write_artifact",
        lambda **kwargs: {"test_commands": kwargs["test_commands"]},
    )
    assert mod.run()["test_commands"] == mod.DEFAULT_TEST_COMMANDS
    called: list[bool] = []
    monkeypatch.setattr(mod, "run", lambda: called.append(True))
    mod.main()
    assert called == [True]


def test_req_learn_5825_failed_validation_command_blocks_artifact() -> None:
    """REQ-LEARN-5825: failed validation commands cannot become readiness."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )

    assert artifact["status"] == "blocked"
    assert artifact["adaptive_memory_contract_ready_score"] == 0.0
    assert artifact["schema_errors"] == []
    assert artifact["honest_verdict"] == "blocked: failed_test_exit_codes"
    assert mod.validate_artifact(artifact) is True

    missing_exits = mod.build_artifact(
        root=REPO,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes={},
    )
    assert missing_exits["status"] == "blocked"
    assert missing_exits["adaptive_memory_contract_ready_score"] == 0.0
    assert missing_exits["honest_verdict"] == "blocked: failed_test_exit_codes"
