"""Tests for REQ-CL-7403 and SCENARIO-CL-7403-*.

The tests keep completion, safety, and oracle-defined value separate. This
prevents a valid null result from being mislabeled as an execution failure.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7403_v649_synthetic_memory as experiment


@pytest.fixture(scope="module")
def protocol() -> dict:
    """REQ-CL-7403: Load the exact sealed V647 protocol bytes."""

    return json.loads((experiment.REPO_ROOT / experiment.PROTOCOL_PATH).read_text())


@pytest.fixture(scope="module")
def replay(protocol: dict) -> experiment.ReplayEvidence:
    """SCENARIO-CL-7403-REPLAY: Execute the complete frozen cohort once."""

    return experiment.replay_protocol(protocol)


def test_preconditions_authenticate_exact_qualified_inputs() -> None:
    """SCENARIO-CL-7403-GATE: Exact local identities qualify dependent work."""

    checks, hashes, sidecars = experiment.collect_preconditions(experiment.REPO_ROOT)

    assert checks
    assert all(row["passed"] for row in checks if row["terminal_blocking"])
    assert (
        hashes[experiment.UPSTREAM_RESULT_PATH.as_posix()]
        == experiment.EXPECTED_HASHES[experiment.UPSTREAM_RESULT_PATH.as_posix()]
    )
    assert (
        hashes[experiment.PROTOCOL_PATH.as_posix()]
        == experiment.EXPECTED_HASHES[experiment.PROTOCOL_PATH.as_posix()]
    )
    assert sidecars == [
        {
            "label": "historical_exp7371_receipt_not_current_inference",
            "path": experiment.UPSTREAM_RESULT_PATH.as_posix(),
            "sha256": experiment.EXPECTED_HASHES[experiment.UPSTREAM_RESULT_PATH.as_posix()],
            "counted_as_current": False,
        }
    ]


def test_preconditions_fail_closed_on_changed_artifact(tmp_path: Path) -> None:
    """SCENARIO-CL-7403-GATE: Changed upstream bytes name the exact failed field."""

    changed = tmp_path / "changed.json"
    changed.write_text("{}\n", encoding="utf-8")
    checks, _hashes, _sidecars = experiment.collect_preconditions(
        experiment.REPO_ROOT, upstream_result_path=changed
    )
    failures = [row for row in checks if row["terminal_blocking"] and not row["passed"]]

    assert failures
    assert {row["check"] for row in failures} >= {
        "upstream_result_sha256",
        "upstream_experiment_id",
        "upstream_proof_boundary_ready_score",
    }
    assert all(
        {"upstream", "artifact_field", "expected", "observed"} <= set(row) for row in failures
    )


def test_protocol_gate_rejects_drift(protocol: dict) -> None:
    """SCENARIO-CL-7403-GATE: The original 32 by 24 cohort cannot drift."""

    assert experiment.protocol_errors(protocol) == []
    changed = deepcopy(protocol)
    changed["evaluation_streams"][0]["requests"].pop()
    assert "evaluation_request_count" in experiment.protocol_errors(changed)
    changed = deepcopy(protocol)
    changed["arms"].reverse()
    assert "arm_order" in experiment.protocol_errors(changed)
    changed = deepcopy(protocol)
    changed["run_date"] = "changed"
    assert "protocol_identity" in experiment.protocol_errors(changed)
    with pytest.raises(ValueError, match="protocol_invalid"):
        experiment.replay_protocol(changed)


def test_object_loader_and_replay_missing_proof_fail_closed(
    tmp_path: Path, protocol: dict, monkeypatch
) -> None:
    """SCENARIO-CL-7403-GATE: Malformed bytes and absent proof evidence stay explicit."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert experiment._load_object(missing) == {}
    assert experiment._load_object(malformed) == {}
    assert experiment._load_object(scalar) == {}

    monkeypatch.setattr(experiment.boundary, "evaluate_stream", lambda _stream: ([], []))
    with pytest.raises(RuntimeError, match="stream_has_no_proof_for_restart"):
        experiment.replay_protocol(protocol)


def test_replay_has_all_equal_input_rows_and_persistent_controls(
    replay: experiment.ReplayEvidence,
) -> None:
    """SCENARIO-CL-7403-REPLAY: Five arms share inputs and retain control state."""

    assert len(replay.rows) == 32 * 24 * 5
    assert {row["arm"] for row in replay.rows} == set(experiment.ARMS)
    assert all(row["complete_service_cost"]["total"] > 0 for row in replay.rows)
    assert all("witnessed_earlier_feedback" in row for row in replay.rows)
    for stream_id in {row["stream_id"] for row in replay.rows}:
        stream_rows = [row for row in replay.rows if row["stream_id"] == stream_id]
        for request_index in range(24):
            paired = [row for row in stream_rows if row["request_index"] == request_index]
            assert [row["arm"] for row in paired] == list(experiment.ARMS)
            assert len({row["assumption_bytes_sha256"] for row in paired}) == 1
        incremental = [
            row for row in stream_rows if row["arm"] == "persistent_incremental_exact_solver"
        ]
        reachability = [
            row for row in stream_rows if row["arm"] == "persistent_source_graph_reachability_cache"
        ]
        assert len({row["state_instance_id"] for row in incremental[:20]}) == 1
        assert len({row["state_instance_id"] for row in reachability}) == 1


def test_causal_witnesses_name_earlier_feedback_and_restart(
    replay: experiment.ReplayEvidence,
) -> None:
    """SCENARIO-CL-7403-AUTHORITY: Erasure links later work to earlier feedback."""

    assert len(replay.witnesses) >= 8
    assert len({row["stream_id"] for row in replay.witnesses}) >= 4
    assert all(
        row["earlier_verified_feedback_request_id"]
        and row["earlier_request_index"] < row["later_request_index"]
        and row["different_later_query"]
        and row["changed_exact_work"]
        and row["independent_path_errors"] == []
        for row in replay.witnesses
    )
    assert len(replay.restart_rows) == 32
    assert all(
        row["exact_bytes_restored"] and row["stale_version_invalidated"]
        for row in replay.restart_rows
    )


def test_authority_controls_preserve_stale_forged_and_missing_cases(
    replay: experiment.ReplayEvidence,
) -> None:
    """SCENARIO-CL-7403-AUTHORITY: Unauthorized proof authority fails closed."""

    names = {row["attack"] for row in replay.attacks}
    assert {"forged_source_hash", "stale_version_proof", "missing_proof_fallback"} <= names
    assert all(row["passed"] for row in replay.attacks)


def test_metrics_use_fixed_block_bootstrap_and_both_controls(
    replay: experiment.ReplayEvidence,
) -> None:
    """SCENARIO-CL-7403-GATES: Frozen cost gates compare both persistent controls."""

    metrics = experiment.synthetic_metrics(replay.rows, replay.witnesses)

    assert metrics["bootstrap_draws"] == 10_000
    assert metrics["bootstrap_seed"] == 7_371_307
    assert metrics["unsafe_decisions"] == 0
    assert metrics["exact_decision_coverage"] == 1.0
    assert metrics["valid_utility"] is True
    for comparator in experiment.PERSISTENT_CONTROLS:
        assert metrics["paid_query_ratio_ci95_upper"][comparator] < 0.90
        assert metrics["full_cost_ratio_ci95_upper"][comparator] <= 1.0


def test_independent_reducer_separates_completion_safety_and_value(
    protocol: dict, replay: experiment.ReplayEvidence
) -> None:
    """SCENARIO-CL-7403-GATES: A valid oracle-defined benefit is circular."""

    artifact = experiment.build_artifact_for_test(protocol, replay)
    reduced = experiment.independent_reduce(artifact)

    assert reduced["synthetic_memory_capture_complete_score"] == 1
    assert reduced["proof_safety_ready_score"] == 1
    assert reduced["synthetic_memory_value_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["promotion_score"] == 0
    assert experiment.validate_artifact(artifact) == []

    null = deepcopy(artifact)
    proof_rows = [row for row in null["rows"] if row["arm"] == "proof_memory"]
    for row in proof_rows:
        row["paid_exact_query"] = True
    experiment.finalize_artifact(null)
    assert null["synthetic_memory_capture_complete_score"] == 1
    assert null["synthetic_memory_value_score"] == 0
    assert null["verdict_class"] == "null"

    disqualified = experiment.build_artifact_for_test(protocol, replay, receipts=[])
    assert disqualified["verdict_class"] == "disqualified"
    assert experiment.validate_artifact(disqualified) == []


def test_blocked_artifact_has_exact_gate_summary(protocol: dict) -> None:
    """SCENARIO-CL-7403-GATE: Missing unchanged input is blocked, not partial."""

    failed = experiment.precondition_row(
        "upstream_result_path",
        "results/missing.json",
        "path",
        "readable_nonempty_json_object",
        "missing",
        False,
    )
    artifact = experiment.build_artifact_for_test(
        protocol, experiment.ReplayEvidence([], [], [], []), preconditions=[failed]
    )

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["first_failure"] == {
        "upstream": "results/missing.json",
        "path": "results/missing.json",
        "check": "upstream_result_path",
        "field": "path",
        "expected": "readable_nonempty_json_object",
        "observed": "missing",
    }
    assert experiment.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(schema="bad"), "identity_mismatch"),
        (lambda value: value.update(MODEL_SPECS=["model"]), "model_declaration_mismatch"),
        (
            lambda value: value["invocation_counts"].update(model_loads_attempted=1),
            "invocation_counts_nonzero",
        ),
        (lambda value: value.update(inference_substrate={}), "substrate_mismatch"),
        (lambda value: value.update(execution_venue="host_cpu"), "substrate_mismatch"),
        (lambda value: value.update(verifier_is_oracle=False), "oracle_declaration_mismatch"),
        (lambda value: value.update(promotion_score=1), "promotion_nonzero"),
        (lambda value: value.update(field_principles={}), "field_principles_incomplete"),
        (
            lambda value: value.update(reproducibility_checksum="bad"),
            "reproducibility_checksum_mismatch",
        ),
        (
            lambda value: value.update(independent_reduction={}),
            "independent_reduction_mismatch",
        ),
        (
            lambda value: value.update(synthetic_memory_value_score=-1),
            "synthetic_memory_value_score_mismatch",
        ),
        (
            lambda value: value.update(acceptance_gate_results=[]),
            "acceptance_gates_mismatch",
        ),
        (
            lambda value: value.update(gate_check_summary={}),
            "gate_check_summary_mismatch",
        ),
        (lambda value: value.update(verdict_class="null"), "verdict_class_mismatch"),
        (lambda value: value.update(honest_verdict="wrong"), "honest_verdict_mismatch"),
    ],
)
def test_artifact_validation_fails_closed(
    protocol: dict,
    replay: experiment.ReplayEvidence,
    mutation,
    expected: str,
) -> None:
    """SCENARIO-CL-7403-ARTIFACT: Terminal declarations cannot drift."""

    artifact = experiment.build_artifact_for_test(protocol, replay)
    mutation(artifact)
    assert expected in experiment.validate_artifact(artifact)


def test_artifact_helpers_cover_progress_plan_and_raw_replay(
    tmp_path: Path, protocol: dict, replay: experiment.ReplayEvidence, capsys
) -> None:
    """SCENARIO-CL-7403-ARTIFACT: Raw evidence and exact commands stay replayable."""

    started = time.monotonic()
    experiment.progress(started, "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    assert experiment.utc_now().endswith("+00:00")
    span = experiment._span("test", started, started, experiment.utc_now(), checkpoints=1)
    assert span["duration_s"] >= 0
    assert span["checkpoints"] == 1

    checkpointed = experiment.replay_protocol(
        protocol,
        checkpoint_dir=tmp_path / experiment.RAW_DIR / "checkpoints",
        emit_progress=True,
        started=started,
    )
    assert len(checkpointed.rows) == len(replay.rows)
    assert "stream_complete" in capsys.readouterr().out

    raw_hashes = experiment.write_raw_evidence(tmp_path, protocol, replay, [])
    assert raw_hashes
    artifact = experiment.build_artifact_for_test(
        protocol,
        replay,
        source_hashes={path: digest for path, digest in raw_hashes.items()},
    )
    assert experiment.cold_reload_errors(artifact, tmp_path, recompute_rows=False) == []

    commands = experiment.scoped_command_plan(experiment.REPO_ROOT, tmp_path / "private")
    assert experiment.validate_scoped_command_plan(experiment.REPO_ROOT, commands) == []
    assert {command.name for command in commands} == set(experiment.REQUIRED_CHECK_NAMES)
    assert "required_command_names_changed" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, commands[:-1]
    )
    full_suite = [
        *commands,
        experiment.validation_scope.CommandSpec("full_python_suite", ("true",), "repository"),
    ]
    assert "full_python_suite_forbidden" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, full_suite
    )
    terminal = experiment.terminal_commands(tmp_path / "candidate.json")
    assert {row.spec.name for row in terminal} == set(experiment.TERMINAL_CHECK_NAMES[:-1])
    normalized = experiment._normalized_receipt(
        {
            "name": "worktree_imports",
            "command": "python check",
            "command_argv": ["python", "check"],
            "command_environment": {"A": "B"},
            "scope": "changed_modules",
            "exit_code": 0,
            "duration_s": 0.1,
            "log_path": "/tmp/log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "resolved_imports": {"carnot.x": "/worktree/x.py"},
        }
    )
    assert normalized["resolved_imports"]
    assert normalized["environment"] == {"A": "B"}
    args = experiment.parse_args(["--date", "20260919", "--output", str(tmp_path / "out.json")])
    assert args.date == "20260919"


def test_cold_reload_detects_row_and_raw_hash_drift(
    tmp_path: Path, protocol: dict, replay: experiment.ReplayEvidence, monkeypatch
) -> None:
    """SCENARIO-CL-7403-ARTIFACT: Fresh readers reject changed evidence."""

    raw_hashes = experiment.write_raw_evidence(tmp_path, protocol, replay, [])
    artifact = experiment.build_artifact_for_test(
        protocol,
        replay,
        source_hashes={path: digest for path, digest in raw_hashes.items()},
    )
    artifact["source_artifact_hashes"]["external/not-raw.json"] = "sha256:" + "2" * 64
    experiment.finalize_artifact(artifact)
    evidence_path = tmp_path / experiment.RAW_DIR / "synthetic_evidence.json"
    payload = json.loads(evidence_path.read_text())
    payload["rows"].pop()
    payload["erasure_witness_rows"].pop()
    payload["restart_rows"].pop()
    payload["cost_rows"].pop()
    experiment.atomic_json(evidence_path, payload)
    attacks_path = tmp_path / experiment.RAW_DIR / "authority_attacks.json"
    attack_payload = json.loads(attacks_path.read_text())
    attack_payload["authority_attack_rows"].pop()
    experiment.atomic_json(attacks_path, attack_payload)
    changed = deepcopy(replay)
    changed.rows.pop()
    monkeypatch.setattr(experiment, "replay_protocol", lambda _protocol: changed)

    errors = experiment.cold_reload_errors(artifact, tmp_path, recompute_rows=True)
    assert "raw_rows_mismatch" in errors
    assert "raw_witnesses_mismatch" in errors
    assert "raw_restart_rows_mismatch" in errors
    assert "raw_cost_rows_mismatch" in errors
    assert "raw_attacks_mismatch" in errors
    assert "independent_row_replay_mismatch" in errors
    assert any(error.startswith("raw_hash_mismatch:") for error in errors)


def test_blocked_validator_rejects_bad_honest_verdict(protocol: dict) -> None:
    """SCENARIO-CL-7403-GATE: A blocked class must retain a blocked verdict prefix."""

    failed = experiment.precondition_row("missing", "input", "path", "present", None, False)
    artifact = experiment.build_artifact_for_test(
        protocol, experiment.ReplayEvidence([], [], [], []), preconditions=[failed]
    )
    artifact["honest_verdict"] = "complete_wrong"
    assert "honest_verdict_mismatch" in experiment.validate_artifact(artifact)


def test_validator_rejects_non_object() -> None:
    """SCENARIO-CL-7403-ARTIFACT: A scalar cannot become a terminal result."""

    assert experiment.validate_artifact([]) == ["artifact_not_object"]
