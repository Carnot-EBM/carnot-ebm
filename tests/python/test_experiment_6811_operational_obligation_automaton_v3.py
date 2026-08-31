"""Tests for the Exp6811 typed operational-obligation automaton.

Spec refs: REQ-AGENTIC-6810-1 and SCENARIO-AGENTIC-6810-1-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.agentic.arc_trajectory_supervisor import (
    AUTOMATON_SCHEMA,
    OBLIGATION_SCHEMA,
    OperationalObligationError,
    OperationalObligationSupervisor,
    TraceAutomatonSupervisor,
    canonical_event_bytes,
    canonical_json_bytes,
    canonical_obligation_bytes,
    compile_operational_obligations,
    read_supervisor_contract,
)
from carnot.agentic import arc_trajectory_supervisor as supervisor_module
from carnot import experiment_6811_operational_obligation_automaton_v3 as exp


def _action(kind: str) -> dict:
    return {"data": None, "kind": kind}


def _record(
    obligation_id: str,
    action: str,
    *,
    priority: str = "hard",
    issuer: str = "exact_authority",
    order: int = 0,
    weight: int | None = None,
    all_of: list[str] | None = None,
    none_of: list[str] | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> dict:
    if weight is None:
        weight = 1 if priority == "soft" else 0
    return {
        "action": _action(action),
        "contract": {
            "authority": {"issuer": issuer, "order": order},
            "execution_consequence": {
                "add": add if add is not None else [f"done:{obligation_id}"],
                "remove": remove if remove is not None else [],
            },
            "fallback": {
                "action": _action("NOOP"),
                "reason": "No exact candidate is available.",
            },
            "prerequisite": {
                "all_of": all_of if all_of is not None else ["ready"],
                "none_of": none_of if none_of is not None else [],
            },
            "priority": {"class": priority, "weight": weight},
        },
        "obligation_id": obligation_id,
    }


def _event(
    event_id: str,
    sequence: int,
    obligation_ids: list[str],
    candidates: list[dict],
    *,
    facts: list[str] | None = None,
) -> dict:
    return {
        "candidates": candidates,
        "event_id": event_id,
        "obligation_ids": obligation_ids,
        "observed_facts": facts if facts is not None else ["ready"],
        "sequence": sequence,
    }


def _candidate(
    candidate_id: str,
    action: str,
    *authorities: str,
    soft_progress: int = 0,
) -> dict:
    return {
        "action": _action(action),
        "authority_chain": list(authorities),
        "candidate_id": candidate_id,
        "soft_progress": soft_progress,
    }


def _supervisor(records: list[dict]) -> OperationalObligationSupervisor:
    compiled = compile_operational_obligations(canonical_obligation_bytes(records))
    return OperationalObligationSupervisor(compiled)


def test_scenario_agentic_6810_1_canonical_compiler_bytes_and_hash() -> None:
    """SCENARIO-AGENTIC-6810-1-CANONICAL-COMPILER fixes exact bytes."""

    records = [_record("obligation-0", "SAFE")]
    source = canonical_obligation_bytes(records)
    compiled = compile_operational_obligations(source)

    assert source == canonical_json_bytes({"obligations": records, "schema": OBLIGATION_SCHEMA})
    assert compiled["schema"] == AUTOMATON_SCHEMA
    assert compiled["priority_order"] == ["hard", "binding", "soft"]
    assert compiled == compile_operational_obligations(source)
    assert compiled["automaton_hash"].startswith("sha256:")

    pretty = json.dumps(json.loads(source), indent=2, sort_keys=True).encode("ascii")
    with pytest.raises(OperationalObligationError, match="non_canonical_serialization") as error:
        compile_operational_obligations(pretty)
    assert error.value.code == "non_canonical_serialization"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("unknown_priority", "unknown_priority"),
        ("absent_authority", "absent_authority"),
        ("missing_fallback", "missing_fallback"),
        ("ambiguous_prerequisite", "ambiguous_prerequisite"),
        ("consequence_deletion", "consequence_deletion"),
        ("duplicate_identity", "duplicate_obligation_id"),
        ("unbounded_cycle", "unbounded_cycle"),
    ),
)
def test_scenario_agentic_6810_1_schema_attacks_fail_closed(
    mutation: str, expected_code: str
) -> None:
    """REQ-AGENTIC-6810-1 rejects every malformed five-part contract."""

    records = [_record("obligation-0", "SAFE")]
    if mutation == "unknown_priority":
        records[0]["contract"]["priority"]["class"] = "urgent"
    elif mutation == "absent_authority":
        del records[0]["contract"]["authority"]
    elif mutation == "missing_fallback":
        del records[0]["contract"]["fallback"]
    elif mutation == "ambiguous_prerequisite":
        records[0]["contract"]["prerequisite"]["none_of"] = ["ready"]
    elif mutation == "consequence_deletion":
        records[0]["contract"]["execution_consequence"] = {"add": [], "remove": []}
    elif mutation == "duplicate_identity":
        records.append(deepcopy(records[0]))
    elif mutation == "unbounded_cycle":
        records = [
            _record("obligation-a", "A", all_of=["done:b"], add=["done:a"]),
            _record("obligation-b", "B", all_of=["done:a"], add=["done:b"]),
        ]

    with pytest.raises(OperationalObligationError) as error:
        compile_operational_obligations(
            canonical_json_bytes(
                {
                    "obligations": records,
                    "schema": OBLIGATION_SCHEMA,
                }
            )
        )
    assert error.value.code == expected_code


def test_scenario_agentic_6810_1_hard_precedes_unbounded_soft_score() -> None:
    """REQ-AGENTIC-6810-1 makes finite soft score unable to pay for hard error."""

    records = [
        _record("00-hard", "SAFE", issuer="hard_authority"),
        _record(
            "01-soft",
            "RISKY",
            priority="soft",
            issuer="soft_adviser",
            order=20,
            weight=7,
        ),
    ]
    event = _event(
        "event-0",
        0,
        ["00-hard", "01-soft"],
        [
            _candidate("candidate-safe", "SAFE", "hard_authority"),
            _candidate(
                "candidate-risky",
                "RISKY",
                "soft_adviser",
                soft_progress=10**30,
            ),
        ],
    )

    (row,) = _supervisor(records).replay(canonical_event_bytes([event]))

    assert row["selected_candidate_id"] == "candidate-safe"
    assert row["hard_violation_count"] == 0
    risky = next(
        item for item in row["candidate_energies"] if item["candidate_id"] == "candidate-risky"
    )
    assert risky["energy"][0] == 1
    assert risky["accepted"] is False


def test_scenario_agentic_6810_1_binding_authority_precedes_soft_progress() -> None:
    """REQ-AGENTIC-6810-1 orders binding duties before finite progress."""

    records = [
        _record("00-first", "FIRST", priority="binding", issuer="issuer-0", order=0),
        _record("01-second", "SECOND", priority="binding", issuer="issuer-1", order=1),
    ]
    event = _event(
        "event-0",
        0,
        ["00-first", "01-second"],
        [
            _candidate("candidate-first", "FIRST", "issuer-0"),
            _candidate("candidate-second", "SECOND", "issuer-1", soft_progress=10**30),
        ],
    )

    (row,) = _supervisor(records).replay(canonical_event_bytes([event]))

    assert row["selected_candidate_id"] == "candidate-first"
    assert row["selected_energy"][1] == [0, 1]


def test_scenario_agentic_6810_1_no_candidate_and_no_op_certificates() -> None:
    """SCENARIO-AGENTIC-6810-1-CERTIFICATES emits both exact certificates."""

    records = [_record("obligation-0", "SAFE", add=["done"])]
    supervisor = _supervisor(records)
    stale = _event(
        "event-0",
        0,
        ["obligation-0"],
        [_candidate("candidate-0", "SAFE", "exact_authority")],
        facts=[],
    )
    (stale_row,) = supervisor.replay(canonical_event_bytes([stale]))

    assert stale_row["certificate"]["kind"] == "no_candidate"
    assert stale_row["selected_candidate_id"] is None
    assert stale_row["selected_action"] == _action("NOOP")
    assert stale_row["certificate"]["reason"] == "stale_prerequisite"

    ready = _event(
        "event-0",
        0,
        ["obligation-0"],
        [_candidate("candidate-0", "SAFE", "exact_authority")],
    )
    (no_op_row,) = supervisor.replay(canonical_event_bytes([ready]), initial_facts=["done"])

    assert no_op_row["certificate"]["kind"] == "no_op"
    assert no_op_row["state_changed"] is False


def test_scenario_agentic_6810_1_authority_spoof_fails_closed() -> None:
    """REQ-AGENTIC-6810-1 rejects an action without its declared issuer."""

    event = _event(
        "event-0",
        0,
        ["obligation-0"],
        [_candidate("candidate-0", "SAFE", "spoofed_authority")],
    )
    (row,) = _supervisor([_record("obligation-0", "SAFE")]).replay(canonical_event_bytes([event]))

    assert row["certificate"]["kind"] == "no_candidate"
    assert row["certificate"]["reason"] == "authority_spoof"
    assert row["conflict_certificates"][0]["reason"] == "authority_spoof"


def test_scenario_agentic_6810_1_duplicate_and_reordered_replay_rejected() -> None:
    """SCENARIO-AGENTIC-6810-1-ATTACKS rejects non-canonical event order."""

    records = [
        _record("obligation-0", "A"),
        _record("obligation-1", "B", issuer="issuer-b"),
    ]
    events = [
        _event("event-0", 0, ["obligation-0"], [_candidate("candidate-0", "A", "exact_authority")]),
        _event("event-1", 1, ["obligation-1"], [_candidate("candidate-1", "B", "issuer-b")]),
    ]
    supervisor = _supervisor(records)

    duplicate = deepcopy(events)
    duplicate[1]["event_id"] = "event-0"
    with pytest.raises(OperationalObligationError) as duplicate_error:
        supervisor.replay(canonical_event_bytes(duplicate))
    assert duplicate_error.value.code == "duplicate_event_id"

    reordered = list(reversed(events))
    with pytest.raises(OperationalObligationError) as reorder_error:
        supervisor.replay(canonical_event_bytes(reordered))
    assert reorder_error.value.code == "replay_reorder"


def test_scenario_agentic_6810_1_v1_reader_remains_compatible() -> None:
    """SCENARIO-AGENTIC-6810-1-CERTIFICATES preserves v1 reads and behavior."""

    v1 = {
        "schema": "carnot.arc.trace_fsm.v1",
        "thresholds": {
            "actions_since_observed_change": 2,
            "consecutive_navigation_or_replay": 2,
            "same_action_run": 2,
        },
    }
    supervisor = read_supervisor_contract(v1)

    assert isinstance(supervisor, TraceAutomatonSupervisor)
    assert supervisor.select_action(
        ("ACTION1", None),
        previous_frame_changed=None,
        level_progress_since_previous_action=False,
    ) == ("ACTION1", None)
    supervisor.finalize()
    assert supervisor.receipt()["fsm_schema"] == "carnot.arc.trace_fsm.v1"


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output = tmp_path_factory.mktemp("exp6811") / "artifact.json"
    return exp.build_artifact(
        result_path=output,
        duration_s=1.25,
        run_date="20260831",
        write=True,
    )


def test_scenario_agentic_6810_1_artifact_reuses_every_frozen_trace(
    built_artifact: dict,
) -> None:
    """REQ-AGENTIC-6810-1 emits one row per source trace and attack."""

    artifact = built_artifact
    trace_rows = [row for row in artifact["rows"] if row["row_type"] == "trace"]
    attack_rows = [row for row in artifact["rows"] if row["row_type"] == "attack"]

    assert artifact["operational_automaton_fixture_ready"] is True
    assert len(trace_rows) == artifact["compiler_manifest"]["source_trace_count"]
    assert len(trace_rows) == 1244 + 30
    assert len(attack_rows) == len(exp.ATTACK_IDS)
    assert {row["attack_id"] for row in attack_rows} == set(exp.ATTACK_IDS)
    assert all(row["source_free"] is True for row in trace_rows)
    assert artifact["hard_violation_count"] == 0
    assert artifact["solve_claim"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert exp.validate_artifact(artifact) == []


def test_scenario_agentic_6810_1_fresh_process_and_attacks_are_complete(
    built_artifact: dict,
) -> None:
    """SCENARIO-AGENTIC-6810-1-ATTACKS proves fresh identity and fail-closed coverage."""

    receipt = built_artifact["canonical_byte_receipts"]

    assert receipt["fresh_process"] is True
    assert receipt["states_byte_identical"] is True
    assert receipt["legal_action_sets_byte_identical"] is True
    assert receipt["selected_actions_byte_identical"] is True
    assert receipt["conflict_certificates_byte_identical"] is True
    assert receipt["hashes_byte_identical"] is True
    assert [row["attack_id"] for row in built_artifact["attack_results"]] == list(exp.ATTACK_IDS)
    assert all(row["failed_closed"] is True for row in built_artifact["attack_results"])
    assert built_artifact["backward_compatibility_receipts"]["v1_readable"] is True


def test_req_agentic_6810_1_artifact_is_stable_and_validator_catches_drift(
    built_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-AGENTIC-6810-1 binds deterministic inputs and outputs."""

    second = exp.build_artifact(
        result_path=tmp_path / "second.json",
        duration_s=9.75,
        run_date="20260831",
        write=False,
    )
    drift = deepcopy(built_artifact)
    drift["priority_order"] = ["soft", "binding", "hard"]

    assert second["reproducibility_checksum"] == built_artifact["reproducibility_checksum"]
    assert "priority order mismatch" in exp.validate_artifact(drift)
    assert "reproducibility checksum mismatch" in exp.validate_artifact(drift)


def test_scenario_agentic_6810_1_blocked_precondition_stops_before_rows(
    tmp_path: Path,
) -> None:
    """REQ-AGENTIC-6810-1 emits the exact blocked diagnostic and stops."""

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "blocked.json",
        duration_s=0.1,
        run_date="20260831",
        write=False,
    )

    assert artifact["status"] == "complete_blocked_operational_obligation_automaton_v3"
    assert artifact["honest_verdict"] == "complete_blocked_operational_obligation_automaton_v3"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["operational_automaton_fixture_ready"] is False
    assert artifact["rows"] == []
    assert artifact["attack_results"] == []
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_checks"]
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []


def test_req_agentic_6810_1_all_schema_rejection_paths_are_named() -> None:
    """REQ-AGENTIC-6810-1 covers every strict schema rejection branch."""

    def rejects(callable_value, code: str) -> None:
        with pytest.raises(OperationalObligationError) as error:
            callable_value()
        assert error.value.code == code

    rejects(
        lambda: supervisor_module._require_exact_keys({}, {"required"}, code="wrong_keys"),
        "wrong_keys",
    )
    rejects(
        lambda: supervisor_module._require_sorted_fact_list(
            "not-a-list", code="bad_facts", field="facts"
        ),
        "bad_facts",
    )
    rejects(
        lambda: supervisor_module._require_sorted_fact_list(
            ["z", "a"], code="bad_facts", field="facts"
        ),
        "bad_facts",
    )
    for action in (None, {"data": None}, {"data": None, "kind": True}):
        rejects(lambda action=action: supervisor_module._validate_action(action, code="bad"), "bad")
    rejects(
        lambda: supervisor_module._validate_action(
            {"data": {"not-json"}, "kind": "SAFE"}, code="bad"
        ),
        "bad",
    )
    rejects(lambda: supervisor_module._validate_obligation(None), "invalid_obligation")

    malformed_records = []
    invalid_id = _record("obligation-0", "SAFE")
    invalid_id["obligation_id"] = ""
    malformed_records.append((invalid_id, "invalid_obligation_id"))
    invalid_contract = _record("obligation-0", "SAFE")
    invalid_contract["contract"] = None
    malformed_records.append((invalid_contract, "invalid_contract"))
    invalid_prerequisite = _record("obligation-0", "SAFE")
    invalid_prerequisite["contract"]["prerequisite"] = None
    malformed_records.append((invalid_prerequisite, "ambiguous_prerequisite"))
    invalid_authority = _record("obligation-0", "SAFE")
    invalid_authority["contract"]["authority"] = None
    malformed_records.append((invalid_authority, "absent_authority"))
    empty_issuer = _record("obligation-0", "SAFE")
    empty_issuer["contract"]["authority"]["issuer"] = ""
    malformed_records.append((empty_issuer, "absent_authority"))
    invalid_order = _record("obligation-0", "SAFE")
    invalid_order["contract"]["authority"]["order"] = True
    malformed_records.append((invalid_order, "invalid_authority_order"))
    missing_reason = _record("obligation-0", "SAFE")
    missing_reason["contract"]["fallback"]["reason"] = ""
    malformed_records.append((missing_reason, "missing_fallback"))
    invalid_consequence = _record("obligation-0", "SAFE")
    invalid_consequence["contract"]["execution_consequence"] = None
    malformed_records.append((invalid_consequence, "consequence_deletion"))
    ambiguous_consequence = _record("obligation-0", "SAFE", add=["same"], remove=["same"])
    malformed_records.append((ambiguous_consequence, "ambiguous_consequence"))
    invalid_priority = _record("obligation-0", "SAFE")
    invalid_priority["contract"]["priority"] = None
    malformed_records.append((invalid_priority, "unknown_priority"))
    bool_weight = _record("obligation-0", "SAFE")
    bool_weight["contract"]["priority"]["weight"] = True
    malformed_records.append((bool_weight, "invalid_priority_weight"))
    zero_soft = _record("obligation-0", "SAFE", priority="soft", weight=0)
    malformed_records.append((zero_soft, "invalid_priority_weight"))
    weighted_hard = _record("obligation-0", "SAFE", weight=1)
    malformed_records.append((weighted_hard, "invalid_priority_weight"))
    for record, code in malformed_records:
        rejects(lambda record=record: supervisor_module._validate_obligation(record), code)

    dag = [
        _record("obligation-a", "A", add=["made"], remove=[]),
        _record("obligation-b", "B", all_of=["made"]),
    ]
    assert (
        compile_operational_obligations(canonical_obligation_bytes(dag))["schema"]
        == AUTOMATON_SCHEMA
    )
    rejects(lambda: compile_operational_obligations({}), "non_canonical_serialization")
    rejects(lambda: compile_operational_obligations(b"{"), "non_canonical_serialization")
    rejects(lambda: compile_operational_obligations(b"[]"), "invalid_obligation_envelope")
    rejects(
        lambda: compile_operational_obligations(
            canonical_json_bytes({"obligations": [_record("a", "A")], "schema": "wrong"})
        ),
        "unsupported_obligation_schema",
    )
    rejects(
        lambda: compile_operational_obligations(
            canonical_json_bytes({"obligations": [], "schema": OBLIGATION_SCHEMA})
        ),
        "empty_obligation_set",
    )
    rejects(
        lambda: compile_operational_obligations(
            canonical_obligation_bytes([_record("z", "Z"), _record("a", "A")])
        ),
        "non_canonical_obligation_order",
    )
    rejects(
        lambda: OperationalObligationSupervisor({"schema": "wrong"}), "unsupported_automaton_schema"
    )
    compiled = compile_operational_obligations(canonical_obligation_bytes([_record("a", "A")]))
    compiled["states"] = []
    rejects(lambda: OperationalObligationSupervisor(compiled), "compiled_automaton_mismatch")
    rejects(
        lambda: read_supervisor_contract({"schema": "wrong"}), "unsupported_supervisor_contract"
    )


def test_req_agentic_6810_1_all_event_rejection_paths_are_named() -> None:
    """REQ-AGENTIC-6810-1 covers strict event and replay validation."""

    supervisor = _supervisor([_record("obligation-0", "SAFE")])
    valid = _event(
        "event-0",
        0,
        ["obligation-0"],
        [_candidate("candidate-0", "SAFE", "exact_authority")],
    )

    def rejects_event(event, code: str) -> None:
        with pytest.raises(OperationalObligationError) as error:
            supervisor._validate_event(event, 0)
        assert error.value.code == code

    rejects_event(None, "invalid_event")
    rejects_event({}, "invalid_event_fields")
    mutation = deepcopy(valid)
    mutation["event_id"] = ""
    rejects_event(mutation, "invalid_event_id")
    mutation = deepcopy(valid)
    mutation["obligation_ids"] = ["unknown"]
    rejects_event(mutation, "unknown_obligation_id")
    mutation = deepcopy(valid)
    mutation["candidates"] = None
    rejects_event(mutation, "invalid_candidates")
    mutation = deepcopy(valid)
    mutation["candidates"] = [None]
    rejects_event(mutation, "invalid_candidate")
    mutation = deepcopy(valid)
    mutation["candidates"][0]["candidate_id"] = ""
    rejects_event(mutation, "invalid_candidate_id")
    mutation = deepcopy(valid)
    mutation["candidates"][0]["authority_chain"] = []
    rejects_event(mutation, "invalid_candidate_authority")
    mutation = deepcopy(valid)
    mutation["candidates"][0]["soft_progress"] = True
    rejects_event(mutation, "invalid_soft_progress")
    mutation = deepcopy(valid)
    mutation["candidates"].append(deepcopy(mutation["candidates"][0]))
    rejects_event(mutation, "duplicate_candidate_id")

    soft = _record("obligation-0", "SAFE", priority="soft")
    unbound = _event(
        "event-0",
        0,
        ["obligation-0"],
        [_candidate("candidate-0", "OTHER", "exact_authority")],
    )
    (unbound_row,) = _supervisor([soft]).replay(canonical_event_bytes([unbound]))
    assert unbound_row["certificate"] == {"kind": "no_candidate", "reason": "no_legal_candidate"}

    invalid_replays = (
        ({}, "non_canonical_event_serialization"),
        (b"{", "non_canonical_event_serialization"),
        (
            json.dumps(json.loads(canonical_event_bytes([valid])), indent=2).encode(),
            "non_canonical_event_serialization",
        ),
        (b"[]", "invalid_event_envelope"),
        (canonical_json_bytes({"events": [], "schema": "wrong"}), "unsupported_event_schema"),
        (
            canonical_json_bytes({"events": None, "schema": supervisor_module.EVENT_SCHEMA}),
            "invalid_event_envelope",
        ),
    )
    for source, code in invalid_replays:
        with pytest.raises(OperationalObligationError) as error:
            supervisor.replay(source)
        assert error.value.code == code
    with pytest.raises(OperationalObligationError, match="invalid_initial_facts"):
        supervisor.replay(canonical_event_bytes([]), initial_facts=[""])
    assert isinstance(
        read_supervisor_contract(supervisor.compiled_automaton), OperationalObligationSupervisor
    )


def test_req_agentic_6810_1_experiment_error_cli_and_validator_coverage(
    built_artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-AGENTIC-6810-1 covers blocked writes, workers, CLI, and drift diagnostics."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp._load_json(list_json)
    text = tmp_path / "text.md"
    text.write_text("no owned requirement or supervisor here", encoding="utf-8")
    assert exp._requirement_bytes(text, "REQ-MISSING") is None
    assert exp._v1_supervisor_bytes(text) is None

    malformed_root = tmp_path / "malformed-root"
    preflight = malformed_root / exp.PREFLIGHT_PATH
    preflight.parent.mkdir(parents=True)
    preflight.write_text("[]", encoding="utf-8")
    blocked_output = tmp_path / "blocked-written.json"
    blocked = exp.build_artifact(
        repo_root=malformed_root,
        result_path=blocked_output,
        duration_s=0.1,
        write=True,
    )
    assert blocked_output.is_file()
    assert blocked["gate_check_summary"]["checks"]["preflight_read_error"] == "ValueError"

    unexpected = exp._compiler_attack("unexpected", lambda records: None, "some_error")
    assert unexpected["failed_closed"] is False
    assert unexpected["outcome"] == "accepted_unexpectedly"

    drift_cases: list[tuple[dict, str]] = []
    missing = deepcopy(built_artifact)
    del missing["schema"]
    drift_cases.append((missing, "missing required fields"))
    wrong_principles = deepcopy(built_artifact)
    wrong_principles["field_principles"] = {}
    drift_cases.append((wrong_principles, "field principles do not cover artifact"))
    scalar_drifts = (
        ("verdict_class", "outside", "verdict class outside closed enum"),
        ("solve_claim", True, "solve claim must be false"),
        ("solve_provenance", "live", "solve provenance mismatch"),
        ("verifier_is_oracle", True, "verifier is oracle must be false"),
        ("hard_violation_count", 1, "accepted action has hard violation"),
        ("verdict_class", "partial", "ready verdict mismatch"),
    )
    for field, value, message in scalar_drifts:
        drift = deepcopy(built_artifact)
        drift[field] = value
        drift_cases.append((drift, message))
    bad_gate = deepcopy(built_artifact)
    bad_gate["gate_check_summary"]["passed"] = False
    drift_cases.append((bad_gate, "ready artifact has failed precondition"))
    bad_component = deepcopy(built_artifact)
    bad_component["readiness_components"]["compiler"] = False
    drift_cases.append((bad_component, "ready artifact has incomplete readiness component"))
    bad_attack_ids = deepcopy(built_artifact)
    bad_attack_ids["attack_results"][0]["attack_id"] = "wrong"
    drift_cases.append((bad_attack_ids, "attack coverage mismatch"))
    bad_attack_result = deepcopy(built_artifact)
    bad_attack_result["attack_results"][0]["failed_closed"] = False
    drift_cases.append((bad_attack_result, "attack did not fail closed"))
    bad_rows = deepcopy(built_artifact)
    bad_rows["rows"] = []
    drift_cases.append((bad_rows, "row coverage mismatch"))
    bad_ready_verdict = deepcopy(built_artifact)
    bad_ready_verdict["honest_verdict"] = "unfinished"
    drift_cases.append((bad_ready_verdict, "ready verdict mismatch"))
    for drift, expected in drift_cases:
        assert any(expected in error for error in exp.validate_artifact(drift))

    blocked_drifts = (
        (
            "gate_check_summary",
            {"passed": False, "failed_checks": []},
            "blocked artifact missing diagnostic",
        ),
        ("rows", [{"row_type": "trace"}], "blocked artifact did not stop before replay"),
        ("verdict_class", "null", "blocked verdict mismatch"),
    )
    for field, value, expected in blocked_drifts:
        drift = deepcopy(blocked)
        drift[field] = value
        assert expected in exp.validate_artifact(drift)

    obligation = [_record("obligation-0", "SAFE")]
    event = [
        _event(
            "event-0",
            0,
            ["obligation-0"],
            [_candidate("candidate-0", "SAFE", "exact_authority")],
        )
    ]
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    exp._write_atomic(
        request,
        {
            "events": event,
            "obligations": obligation,
            "schema": "carnot.experiment_6811.fresh_request.v1",
        },
        pretty=False,
    )
    assert exp._fresh_worker(request, response) == 0
    assert exp._load_json(response)["projection"]["states"]
    wrong_request = tmp_path / "wrong-request.json"
    exp._write_atomic(wrong_request, {"schema": "wrong"}, pretty=False)
    assert exp._fresh_worker(wrong_request, response) == 2
    assert exp.main(["--fresh-request", str(request)]) == 2
    assert exp.main(["--fresh-request", str(request), "--fresh-response", str(response)]) == 0

    cli_output = tmp_path / "cli-blocked.json"
    assert (
        exp.main(["--project-root", str(tmp_path / "empty-root"), "--output", str(cli_output)]) == 0
    )
    assert exp.main(["--validate", "--output", str(cli_output)]) == 0
    assert "valid" in capsys.readouterr().out
    invalid_output = tmp_path / "invalid.json"
    invalid_output.write_text('{"bad":true}', encoding="utf-8")
    assert exp.main(["--validate", "--output", str(invalid_output)]) == 1
    assert "missing required fields" in capsys.readouterr().out

    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: {"bad": True})
    assert exp.main(["--output", str(tmp_path / "never-written.json")]) == 1
    assert "missing required fields" in capsys.readouterr().out
