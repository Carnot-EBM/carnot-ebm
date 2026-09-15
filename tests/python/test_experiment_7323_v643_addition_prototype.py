"""Behavior tests for REQ-CL-7323 structural constraint addition."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7323_v643_addition_prototype as exp


def _request(request_id: str = "request-0") -> dict[str, object]:
    return {
        "request_id": request_id,
        "activities": ["a", "b", "c", "d"],
        "allowed_slots": {name: [0, 1, 2, 3] for name in ("a", "b", "c", "d")},
    }


def _executor(version: str = "executor-v1") -> exp.BooleanScheduleExecutor:
    return exp.BooleanScheduleExecutor(
        version,
        {"pairwise_separation": {"a|b": 2}, "capacity": 2},
    )


def test_scenario_cl_7323_language_requires_localizing_subqueries() -> None:
    """SCENARIO-CL-7323-LANGUAGE: a compound rejection proves no atom alone."""

    request = _request()
    learner = exp.StructuralAdditionLearner("executor-v1")
    oracle = exp.ChargedOracle(_executor(), request)
    decision = learner.propose(request)

    assert oracle.query(decision["plan"], "main") is False
    assert learner.active_atoms() == []
    atoms = learner.localize_rejection(request, decision["plan"], oracle)

    assert {atom["kind"] for atom in atoms} == {"pairwise_separation", "capacity"}
    assert all(atom["version"] == "executor-v1" for atom in atoms)
    assert all(atom["witness"] and atom["query_receipts"] for atom in atoms)
    assert learner.uncertain_candidates()
    assert oracle.call_count <= exp.QUERY_BUDGET
    assert "_rules" not in inspect.getsource(exp.StructuralAdditionLearner)


def test_scenario_cl_7323_version_scopes_authority_and_exposes_hidden_change() -> None:
    """SCENARIO-CL-7323-VERSION: only authenticated same-version atoms reactivate."""

    learner = exp.StructuralAdditionLearner("executor-v1")
    request = _request()
    oracle = exp.ChargedOracle(_executor(), request)
    plan = learner.propose(request)["plan"]
    assert oracle.query(plan, "main") is False
    learner.localize_rejection(request, plan, oracle)
    v1_atoms = learner.active_atoms()
    assert v1_atoms

    learner.activate_version("executor-v2")
    assert learner.active_atoms() == []
    learner.activate_version("executor-v1")
    assert learner.active_atoms() == v1_atoms

    challenge = exp.run_unannounced_change_challenge()
    assert challenge["passed"] is True
    assert challenge["assumption_limit_exposed"] is True
    assert challenge["learned_recovery_claim"] is False


def test_scenario_cl_7323_bounds_rejects_bad_evidence_and_exact_cache_aliasing() -> None:
    """SCENARIO-CL-7323-BOUNDS: receipts, bytes, hypotheses, and cache stay bounded."""

    request = _request()
    executor = _executor()
    cache: dict[str, bool] = {}
    oracle = exp.ChargedOracle(executor, request, exact_cache=cache)
    plan = {"request_id": request["request_id"], "assignments": {name: 3 for name in "abcd"}}
    first = oracle.query(plan, "main", allow_cache=True)
    assert oracle.query(deepcopy(plan), "main", allow_cache=True) is first
    assert oracle.call_count == 1
    changed = deepcopy(plan)
    changed["assignments"]["a"] = 2
    oracle.query(changed, "main", allow_cache=True)
    assert oracle.call_count == 2

    learner = exp.StructuralAdditionLearner("executor-v1")
    learner.retain_accepted_plan(
        {"request_id": "accepted", "assignments": {"a": 0, "b": 0}},
        {"accepted": True, "sequence": 1},
    )
    atom = exp.make_atom(
        "pairwise_separation",
        "executor-v1",
        {"pair": ["a", "b"], "minimum": 1},
        {"accepted": False, "sequence": 2},
        [{"accepted": False, "sequence": 2}],
    )
    with pytest.raises(exp.AdditionRejected, match="contradicts_retained_acceptance"):
        learner.admit_atom(atom, current_query_index=2)
    stale = deepcopy(atom)
    stale["version"] = "executor-v2"
    with pytest.raises(exp.AdditionRejected, match="version_mismatch"):
        learner.admit_atom(stale, current_query_index=2)
    early = deepcopy(atom)
    early["query_receipts"][0]["sequence"] = 9
    with pytest.raises(exp.AdditionRejected, match="early_evidence"):
        learner.admit_atom(early, current_query_index=2)

    tiny = exp.StructuralAdditionLearner("executor-v1", memory_cap_bytes=400)
    with pytest.raises(exp.AdditionRejected, match="persistent_state_cap"):
        tiny.admit_atom(atom, current_query_index=2)
    assert len(learner.pair_hypotheses(("a", "b"))) <= 16


def test_scenario_cl_7323_streams_seal_independent_public_and_private_views() -> None:
    """SCENARIO-CL-7323-STREAMS: all stream counts and public limits are frozen."""

    streams = exp.build_stream_manifest()
    assert exp.stream_conformance_errors(streams) == []
    assert len(streams["development"]["environments"]) == 8
    assert len(streams["held_out"]["environments"]) == 24
    assert streams["held_out"]["strata"] == {
        "stationary": 8,
        "announced_version_change": 8,
        "return_to_known_version": 8,
    }
    assert streams["challenge"]["disjoint"] is True
    for split in ("development", "held_out"):
        for environment in streams[split]["environments"]:
            assert len(environment["public_requests"]) == 24
            assert all(len(row["activities"]) <= 6 for row in environment["public_requests"])
            assert all(
                len(slots) <= 4
                for row in environment["public_requests"]
                for slots in row["allowed_slots"].values()
            )
            assert "private_rules" not in environment


def test_scenario_cl_7323_arms_record_cost_safety_and_causal_use() -> None:
    """SCENARIO-CL-7323-ARMS: all arms run and persistent atoms affect later requests."""

    panel = exp.run_development_panel(exp.build_stream_manifest())
    assert {row["arm"] for row in panel["rows"]} == set(exp.ALL_ARMS)
    assert len(panel["rows"]) == 8 * 24 * len(exp.ALL_ARMS)
    assert all(row["oracle_calls"] <= exp.QUERY_BUDGET for row in panel["rows"])
    assert all(row["state_bytes"] <= exp.STATE_CAP_BYTES for row in panel["rows"])
    assert all(not row["returned_infeasible"] for row in panel["rows"])
    assert all(row["censored"] is False for row in panel["rows"])
    assert panel["causal_later_distinct_request_count"] > 0
    assert panel["label_shuffled_is_diagnostic_only"] is True


def test_scenario_cl_7323_e2e_adapter_is_default_off_and_durable(tmp_path: Path) -> None:
    """SCENARIO-CL-7323-E2E: transactional commits survive restart and invalidate by version."""

    request = _request()
    disabled = exp.AdditionMemoryAdapter(tmp_path / "disabled")
    assert disabled.predict(request, _executor())["status"] == "abstain"
    assert not (tmp_path / "disabled").exists()

    receipt = exp.run_adapter_e2e(tmp_path / "enabled")
    assert receipt["passed"] is True
    assert receipt["charged_final_check"] is True
    assert receipt["later_distinct_prediction_changed"] is True
    assert receipt["commit_persisted"] is True
    assert receipt["crash_reload_parity"] is True
    assert receipt["version_invalidated"] is True


def test_scenario_cl_7323_e2e_control_can_rerun_in_one_output_root(tmp_path: Path) -> None:
    """SCENARIO-CL-7323-E2E: prior control state cannot poison a later run."""

    first = exp.run_development_controls(tmp_path / "state")
    second = exp.run_development_controls(tmp_path / "state")

    assert all(row["passed"] for row in first)
    assert all(row["passed"] for row in second)


def test_scenario_cl_7323_terminal_is_complete_before_held_out_efficacy(tmp_path: Path) -> None:
    """SCENARIO-CL-7323-TERMINAL: fixture readiness and unmeasured efficacy stay separate."""

    artifact = exp.build_artifact(exp.REPO_ROOT, tmp_path / "state", progress=False)
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["addition_fixture_ready_score"] == 1
    assert artifact["addition_promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["sample_size_budget"]["held_out"]["attempted_environments"] == 0
    assert exp.validate_artifact(artifact, require_validation=False) == []

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["not-current"]
    assert "model_contract" in exp.validate_artifact(changed, require_validation=False)
    changed = deepcopy(artifact)
    changed["rows"][0]["oracle_calls"] = exp.QUERY_BUDGET + 1
    assert "row_query_budget" in exp.validate_artifact(changed, require_validation=False)

    checks, hashes = exp.collect_preconditions(
        exp.REPO_ROOT, overrides={"executor_available": False}
    )
    blocked = exp.build_blocked_artifact(checks, hashes)
    failure = blocked["gate_check_summary"]["first_failure"]
    assert blocked["verdict_class"] == "blocked"
    assert blocked["addition_fixture_ready_score"] == 0
    assert failure["upstream"] == "host_cpu_executor"
    assert failure["check"] == "executor_available"
    assert failure["field"] == "available"
    assert failure["expected_value"] is True
    assert failure["observed_value"] is False


def test_scenario_cl_7323_terminal_main_writes_only_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7323-TERMINAL: the entrypoint seals current validation receipts."""

    validation = {
        "validation_receipts": [
            {
                "name": name,
                "command": name,
                "scope": "test",
                "exit_code": 0,
                "duration_s": 0.01,
                "log_sha256": "sha256:" + "0" * 64,
                "passed": True,
                "timed_out": False,
            }
            for name in exp.REQUIRED_CHECK_NAMES
        ],
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": exp.repository_health(),
    }
    monkeypatch.setattr(exp, "run_scoped_validation", lambda *args, **kwargs: validation)
    monkeypatch.setattr(exp, "run_terminal_validators", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        exp,
        "run_full_python_suite",
        lambda *args, **kwargs: exp.passing_receipt("full_python_suite"),
    )

    assert exp.main(["--date", "20260915", "--output-root", str(tmp_path)]) == 0
    result_path = tmp_path / "results" / "experiment_7323_v643_addition_prototype.json"
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    assert artifact["required_checks_passed"] is True
    assert artifact["addition_fixture_ready_score"] == 1
    assert exp.validate_artifact(artifact, require_validation=True) == []


def test_scenario_cl_7323_bounds_covers_executor_and_learner_failure_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7323-BOUNDS: malformed plans and unsafe state fail closed."""

    request = _request()
    executor = _executor()
    assert exp._utility(request, None) == 0.0
    assert executor.check(request, {"request_id": "wrong", "assignments": {"a": 0}}) is False
    assert executor.check(request, {"request_id": "request-0", "assignments": {}}) is False
    assert (
        executor.check(request, {"request_id": "request-0", "assignments": {"unknown": 0}}) is False
    )
    assert executor.check(request, {"request_id": "request-0", "assignments": {"a": 99}}) is False

    budgeted = exp.ChargedOracle(executor, request)
    budgeted.call_count = exp.QUERY_BUDGET
    with pytest.raises(exp.AdditionRejected, match="query_budget"):
        budgeted.query({"request_id": "request-0", "assignments": {"a": 0}}, "main")

    learner = exp.StructuralAdditionLearner("executor-v1")
    receipt = {"accepted": False, "sequence": 1}
    atom = exp.make_atom(
        "pairwise_separation",
        "executor-v1",
        {"pair": ["a", "b"], "minimum": 1},
        receipt,
        [receipt],
    )
    partial = {"request_id": "partial", "assignments": {"a": 0}}
    assert exp._atom_allows_plan(atom, partial) is True
    missing = deepcopy(atom)
    missing["query_receipts"] = []
    with pytest.raises(exp.AdditionRejected, match="missing_query_receipts"):
        learner.admit_atom(missing, current_query_index=1)
    unsupported = deepcopy(atom)
    unsupported["kind"] = "unsupported"
    with pytest.raises(exp.AdditionRejected, match="unsupported_atom"):
        learner.admit_atom(unsupported, current_query_index=1)
    admitted = learner.admit_atom(atom, current_query_index=1)
    assert learner.admit_atom(atom, current_query_index=1) == admitted

    corrupt = deepcopy(atom)
    corrupt["payload"]["minimum"] = 2
    fresh = exp.StructuralAdditionLearner("executor-v1")
    with pytest.raises(exp.AdditionRejected, match="atom_hash_mismatch"):
        fresh.import_certified_atoms([corrupt])

    capacity_zero = exp.make_atom(
        "capacity",
        "executor-v1",
        {"maximum": 0},
        receipt,
        [receipt],
    )
    empty_solver = exp.StructuralAdditionLearner("executor-v1")
    empty_solver.admit_atom(capacity_zero, current_query_index=1)
    assert empty_solver.propose(request)["plan"] is None
    assert (
        empty_solver.run_request(request, exp.ChargedOracle(executor, request))["returned_plan"]
        is None
    )
    frozen = exp.StructuralAdditionLearner("executor-v1", frozen=True)
    assert (
        frozen.localize_rejection(
            request,
            {"request_id": "request-0", "assignments": {name: 0 for name in "abcd"}},
            exp.ChargedOracle(executor, request),
        )
        == []
    )
    frozen_result = frozen.run_request(request, exp.ChargedOracle(executor, request))
    assert frozen_result["returned_plan"] is None

    ticks = iter((0.0, 3.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    with pytest.raises(exp.AdditionRejected, match="solver_time_limit"):
        exp.StructuralAdditionLearner("executor-v1").propose(request)


def test_scenario_cl_7323_streams_reports_each_malformed_seal() -> None:
    """SCENARIO-CL-7323-STREAMS: malformed counts and public views are named."""

    malformed = exp.build_stream_manifest()
    malformed["development"]["environments"] = malformed["development"]["environments"][:7]
    malformed["held_out"]["environments"] = malformed["held_out"]["environments"][:23]
    first = malformed["held_out"]["environments"][0]
    first["public_requests"] = first["public_requests"][:23]
    first["public_requests"][0]["activities"] = list("abcdefg")
    first["public_requests"][0]["allowed_slots"]["a"] = [0, 1, 2, 3, 4]
    first["public_requests"][0]["private_hint"] = True
    assert exp.stream_conformance_errors(malformed) == [
        "activity_limit",
        "development_count",
        "held_out_count",
        "manifest_hash",
        "private_view_leak",
        "request_count",
        "slot_limit",
    ]


def test_scenario_cl_7323_e2e_adapter_rejects_disabled_duplicate_and_over_cap(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7323-E2E: adapter writes fail without authority or byte room."""

    disabled = exp.AdditionMemoryAdapter(tmp_path / "disabled")
    assert disabled._atoms() == []
    assert disabled.execute(_request(), _executor())["status"] == "abstain"
    receipt = {"accepted": False, "sequence": 1}
    atom = exp.make_atom(
        "pairwise_separation",
        "executor-v1",
        {"pair": ["a", "b"], "minimum": 1},
        receipt,
        [receipt],
    )
    with pytest.raises(exp.AdditionRejected, match="adapter_disabled"):
        disabled._commit_atom(atom, 1)

    enabled = exp.AdditionMemoryAdapter(tmp_path / "enabled", enabled=True)
    enabled._commit_atom(atom, 1)
    with pytest.raises(exp.AdditionRejected, match="admission_check_failed"):
        enabled._commit_atom(atom, 1)

    bounded = exp.AdditionMemoryAdapter(tmp_path / "bounded", enabled=True, memory_cap_bytes=1)
    with pytest.raises(exp.AdditionRejected, match="persistent_state_cap"):
        bounded._commit_atom(atom, 1)


def test_scenario_cl_7323_terminal_covers_preflight_and_validator_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7323-TERMINAL: each terminal inconsistency has a stable name."""

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    checks, hashes = exp.collect_preconditions(empty_root)
    assert hashes == {}
    assert exp.gate_summary(checks)["passed"] is False
    monkeypatch.setattr(exp, "collect_preconditions", lambda *_args, **_kwargs: (checks, hashes))
    assert (
        exp.build_artifact(empty_root, tmp_path / "blocked-state", progress=False)["status"]
        == "blocked"
    )
    monkeypatch.undo()

    artifact = exp.build_artifact(exp.REPO_ROOT, tmp_path / "state", progress=False)
    mutations = {
        "identity": {"schema": "wrong"},
        "substrate": {"inference_substrate": "wrong"},
        "venue": {"execution_venue": "wrong"},
        "verdict_class": {"verdict_class": "wrong"},
        "blocked_rows": {"status": "blocked"},
        "row_state_cap": {
            "rows": [{**artifact["rows"][0], "state_bytes": exp.STATE_CAP_BYTES + 1}]
        },
        "returned_infeasible": {"rows": [{**artifact["rows"][0], "returned_infeasible": True}]},
        "premature_promotion": {"addition_promotion_score": 1},
        "required_validation": {"required_checks_passed": False},
        "validation_receipts": {"validation_receipts": []},
        "fixture_ready_class": {"verdict_class": "null"},
        "failed_readiness": {"verdict_class": "disqualified"},
    }
    for expected, updates in mutations.items():
        changed = deepcopy(artifact)
        changed.update(updates)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        errors = exp.validate_artifact(
            changed,
            require_validation=expected in {"required_validation", "validation_receipts"},
        )
        assert expected in errors

    output = tmp_path / "atomic.json"
    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        exp._atomic_write(output, {"value": 1})
    assert not list(tmp_path.glob(".atomic.json.*.tmp"))


def test_scenario_cl_7323_terminal_covers_command_wrappers_and_main_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7323-TERMINAL: command and entrypoint failure paths stay terminal."""

    calls: list[list[exp.CommandSpec]] = []

    def fake_commands(
        _root: Path, commands: list[exp.CommandSpec], **_kwargs: object
    ) -> list[dict]:
        calls.append(commands)
        return [exp.passing_receipt(command.name) for command in commands]

    monkeypatch.setattr(exp, "run_commands", fake_commands)
    assert exp.run_full_python_suite(exp.REPO_ROOT, tmp_path)["passed"] is True
    assert len(exp.run_terminal_validators(exp.REPO_ROOT, tmp_path / "candidate", tmp_path)) == 2
    assert [command.name for group in calls for command in group] == [
        "full_python_suite",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]

    with pytest.raises(SystemExit, match="requires"):
        exp.main(["--date", "20260914", "--output-root", str(tmp_path)])

    candidate = tmp_path / "validate.json"
    artifact = exp.build_artifact(exp.REPO_ROOT, tmp_path / "validate-state", progress=False)
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--date", "20260915", "--validate", str(candidate)]) == 1

    checks, hashes = exp.collect_preconditions(
        exp.REPO_ROOT, overrides={"executor_available": False}
    )
    blocked = exp.build_blocked_artifact(checks, hashes)
    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: blocked)
    blocked_root = tmp_path / "blocked"
    assert exp.main(["--date", "20260915", "--output-root", str(blocked_root)]) == 0

    complete = artifact
    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    failed_validation = {
        "validation_receipts": [],
        "required_checks_passed": False,
        "repository_health": exp.repository_health(),
    }
    monkeypatch.setattr(exp, "run_scoped_validation", lambda *_args, **_kwargs: failed_validation)
    monkeypatch.setattr(
        exp,
        "run_terminal_validators",
        lambda *_args, **_kwargs: [{**exp.passing_receipt("adversarial_verify"), "passed": False}],
    )
    monkeypatch.setattr(
        exp,
        "run_full_python_suite",
        lambda *_args, **_kwargs: {
            **exp.passing_receipt("full_python_suite"),
            "passed": False,
            "output_tail": "test_experiment_7323_v643_addition_prototype.py failed",
        },
    )
    disqualified_root = tmp_path / "disqualified"
    assert exp.main(["--date", "20260915", "--output-root", str(disqualified_root)]) == 0
    payload = json.loads(
        (disqualified_root / "results" / exp.RESULT_NAME).read_text(encoding="utf-8")
    )
    assert payload["verdict_class"] == "disqualified"
    assert payload["addition_fixture_ready_score"] == 0

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(RuntimeError, match="terminal_artifact_invalid"):
        exp.main(["--date", "20260915", "--output-root", str(tmp_path / "invalid")])
