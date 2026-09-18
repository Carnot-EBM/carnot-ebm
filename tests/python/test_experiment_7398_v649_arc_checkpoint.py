"""REQ-ARC-WMTE-7398: qualify exact-once ARC episode checkpoint recovery."""

from __future__ import annotations

from copy import deepcopy
import json
import time
from pathlib import Path

import pytest

from carnot import experiment_7398_v649_arc_checkpoint as subject


pytestmark = pytest.mark.memory_watchdog_skip
ROOT = Path(__file__).resolve().parents[2]


def _schedule() -> list[dict[str, object]]:
    return [
        {"episode_id": "fixture:0", "game": "fixture", "seed": 0},
        {"episode_id": "fixture:1", "game": "fixture", "seed": 1},
    ]


class FixtureExecutor:
    """Create deterministic request and action events for restart tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, episode, emit):
        episode_id = str(episode["episode_id"])
        self.calls.append(episode_id)
        emit("request_completed", {"request_index": 0, "response_hash": "sha256:reply"})
        row = {
            "episode_id": episode_id,
            "game": episode["game"],
            "seed": episode["seed"],
            "action": 1,
            "action_data": {"source": "fixture"},
            "policy_class": "E3AgentPolicy",
            "factory": "make_carnot_agent",
            "adapter_disabled": True,
            "solution_paths_denied": True,
            "tool_result_consumed": True,
            "later_policy_action": True,
        }
        emit("action_completed", row)
        return row


@pytest.mark.parametrize(
    "boundary",
    [
        "before_first_request",
        "after_request",
        "after_action",
        "between_episodes",
        "before_final_aggregation",
    ],
)
def test_interruption_matrix_recovers_actions_exactly_once(tmp_path: Path, boundary: str) -> None:
    """SCENARIO-ARC-WMTE-7398-INTERRUPTION-MATRIX: resume all boundaries."""

    executor = FixtureExecutor()
    with pytest.raises(subject.InjectedInterruption, match=boundary):
        subject.run_checkpointed_episodes(
            tmp_path / boundary,
            _schedule(),
            executor,
            interrupt_at=boundary,
        )

    resumed = subject.run_checkpointed_episodes(tmp_path / boundary, _schedule(), executor)

    assert [row["episode_id"] for row in resumed["rows"]] == ["fixture:0", "fixture:1"]
    assert len({row["action_event_id"] for row in resumed["rows"]}) == 2
    assert resumed["accounting"] == {
        "planned_units": 2,
        "attempted_units": 2,
        "completed_units": 2,
        "censored_units": 0,
        "unstarted_units": 0,
    }
    assert all(row["restart_parity"] for row in resumed["episode_recovery_rows"])
    assert resumed["independent_reader_passed"] is True
    expected_calls = {
        "before_first_request": ["fixture:0", "fixture:0", "fixture:1"],
        "after_request": ["fixture:0", "fixture:0", "fixture:1"],
        "after_action": ["fixture:0", "fixture:1"],
        "between_episodes": ["fixture:0", "fixture:1"],
        "before_final_aggregation": ["fixture:0", "fixture:1"],
    }
    assert executor.calls == expected_calls[boundary]


def test_valid_completed_episode_is_not_executed_again(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: completed checkpoint rows are authoritative."""

    executor = FixtureExecutor()
    first = subject.run_checkpointed_episodes(tmp_path, _schedule(), executor)
    second = subject.run_checkpointed_episodes(tmp_path, _schedule(), executor)

    assert first["rows"] == second["rows"]
    assert executor.calls == ["fixture:0", "fixture:1"]
    assert all(row["recovered_from_checkpoint"] for row in second["episode_recovery_rows"])


def test_integrity_reader_rejects_truncated_duplicate_tampered_and_state_mismatch(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7398-JOURNAL-INTEGRITY: reject invalid journals."""

    executor = FixtureExecutor()
    base = tmp_path / "base"
    subject.run_checkpointed_episodes(base, _schedule(), executor)
    source = base / "fixture_0.json"

    truncated = tmp_path / "truncated.json"
    truncated.write_bytes(source.read_bytes()[:-3])
    with pytest.raises(subject.CheckpointIntegrityError, match="malformed"):
        subject.read_episode_checkpoint(truncated, _schedule()[0])

    duplicate = tmp_path / "duplicate.json"
    value = json.loads(source.read_text())
    value["events"].append(deepcopy(value["events"][0]))
    duplicate.write_text(json.dumps(value))
    with pytest.raises(subject.CheckpointIntegrityError, match="duplicate"):
        subject.read_episode_checkpoint(duplicate, _schedule()[0])

    tampered = tmp_path / "tampered.json"
    value = json.loads(source.read_text())
    value["events"][1]["detail"]["response_hash"] = "sha256:changed"
    tampered.write_text(json.dumps(value))
    with pytest.raises(subject.CheckpointIntegrityError, match="event hash"):
        subject.read_episode_checkpoint(tampered, _schedule()[0])

    changed = deepcopy(_schedule()[0])
    changed["seed"] = 99
    with pytest.raises(subject.CheckpointIntegrityError, match="state hash"):
        subject.read_episode_checkpoint(source, changed)


def test_reader_rejects_reordered_chain_and_invalid_event_shape(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: ordering and required event fields fail closed."""

    subject.run_checkpointed_episodes(tmp_path, _schedule(), FixtureExecutor())
    source = tmp_path / "fixture_0.json"
    value = json.loads(source.read_text())
    value["events"][0], value["events"][1] = value["events"][1], value["events"][0]
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(value))
    with pytest.raises(subject.CheckpointIntegrityError, match="sequence|hash chain"):
        subject.read_episode_checkpoint(reordered, _schedule()[0])

    value = json.loads(source.read_text())
    del value["events"][0]["event_id"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(value))
    with pytest.raises(subject.CheckpointIntegrityError, match="event shape"):
        subject.read_episode_checkpoint(invalid, _schedule()[0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema="bad"), "schema"),
        (lambda value: value.update(episode_id="wrong"), "episode identity"),
        (lambda value: value.update(events={}), "events must be a list"),
        (lambda value: value["events"].__setitem__(0, []), "event shape"),
        (lambda value: value["events"][0].update(episode_id="wrong"), "event episode"),
        (lambda value: value["events"][0].update(state_hash="bad"), "event state hash"),
        (lambda value: value["events"][0].update(prior_event_hash="bad"), "hash chain"),
        (lambda value: value.update(journal_checksum="bad"), "journal checksum"),
    ],
)
def test_reader_rejects_each_structural_boundary(tmp_path: Path, mutation, message: str) -> None:
    """SCENARIO-ARC-WMTE-7398-JOURNAL-INTEGRITY: fail each typed boundary."""

    case = tmp_path / message.replace(" ", "_")
    subject.run_checkpointed_episodes(case, _schedule(), FixtureExecutor())
    source = case / "fixture_0.json"
    value = json.loads(source.read_text())
    mutation(value)
    source.write_text(json.dumps(value))
    with pytest.raises(subject.CheckpointIntegrityError, match=message):
        subject.read_episode_checkpoint(source, _schedule()[0])


def test_checkpoint_helpers_reject_changed_duplicates_and_missing_actions(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: idempotence never accepts changed or absent action data."""

    assert not issubclass(subject.InjectedInterruption, Exception)
    checkpoint = tmp_path / "one.json"
    episode = _schedule()[0]
    subject._append_event(checkpoint, episode, "episode_started", 0, {"episode_index": 0})
    with pytest.raises(subject.CheckpointIntegrityError, match="changed content"):
        subject._append_event(checkpoint, episode, "episode_started", 0, {"episode_index": 1})
    with pytest.raises(ValueError, match="unknown interruption"):
        subject.run_checkpointed_episodes(
            tmp_path / "bad", _schedule(), FixtureExecutor(), interrupt_at="bad"
        )

    def missing_action(source, emit):
        del emit
        return {"episode_id": source["episode_id"]}

    with pytest.raises(subject.CheckpointIntegrityError, match="exactly one matching action"):
        subject.run_checkpointed_episodes(tmp_path / "missing-action", _schedule(), missing_action)


def test_real_scored_path_uses_factory_denies_paths_and_reaches_action(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7398-SCORED-PATH: exercise the real submitted policy."""

    emitted: list[tuple[str, dict[str, object]]] = []
    row = subject.run_scripted_scored_episode(
        {"episode_id": "r11l:7398", "game": "r11l", "seed": 7398},
        tmp_path,
        lambda kind, detail: emitted.append((kind, detail)),
    )

    assert row["factory"] == "make_carnot_agent"
    assert row["policy_class"] == "E3AgentPolicy"
    assert row["adapter_disabled"] is True
    assert row["solution_paths_denied"] is True
    assert row["tool_result_consumed"] is True
    assert row["later_policy_action"] is True
    assert [kind for kind, _ in emitted] == [
        "request_completed",
        "request_completed",
        "action_completed",
    ]


def test_real_scored_path_restores_existing_environment(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: fixture flags never change production caller state."""

    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "caller")
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "caller-disabled")
    emitted = []
    subject.run_scripted_scored_episode(
        {"episode_id": "r11l:restore", "game": "r11l", "seed": 7398},
        tmp_path,
        lambda kind, detail: emitted.append((kind, detail)),
    )
    assert emitted
    assert subject.os.environ["CARNOT_ARC_INDUCE_THINK"] == "caller"
    assert subject.os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "caller-disabled"


def test_interruption_matrix_reduces_all_controls(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7398-INTERRUPTION-MATRIX: reduce the fixed five-case panel."""

    def scripted(episode, work_dir, emit):
        del work_dir
        emit("request_completed", {"request_index": 0, "response_hash": "sha256:reply"})
        row = {
            "episode_id": episode["episode_id"],
            "game": episode["game"],
            "seed": episode["seed"],
            "action": 1,
            "adapter_disabled": True,
            "solution_paths_denied": True,
            "tool_result_consumed": True,
            "later_policy_action": True,
        }
        emit("action_completed", row)
        return row

    monkeypatch.setattr(subject, "run_scripted_scored_episode", scripted)
    reduced = subject.run_interruption_matrix(tmp_path)
    assert len(reduced["rows"]) == 10
    assert len(reduced["episode_recovery_rows"]) == 10
    assert len(reduced["checkpoint_receipts"]) == 10
    assert all(row["passed"] for row in reduced["interruption_controls"])
    assert reduced["accounting"]["completed_units"] == 10
    assert reduced["independent_reader_passed"] is True


def test_historical_diagnosis_and_sidecars_keep_provenance_separate(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7398-PROVENANCE-BOUNDARY: isolate old and scripted calls."""

    diagnosis = subject.diagnose_historical_timeout(ROOT)
    assert diagnosis["cause_reproduced"] is True
    assert diagnosis["terminal_summary_action_count"] == 0
    assert diagnosis["durable_completed_action_count"] == 624
    assert diagnosis["corrected_accounting"] == {
        "planned_units": 6,
        "attempted_units": 6,
        "completed_units": 5,
        "censored_units": 1,
        "unstarted_units": 0,
    }

    scripted = {"rows": [{"episode_id": "fixture:0", "action": 1}]}
    receipts = subject.write_provenance_sidecars(tmp_path, diagnosis, scripted)
    assert set(receipts) == {"historical_runtime", "scripted_transport"}
    assert receipts["historical_runtime"]["scope"] == "historical"
    assert receipts["scripted_transport"]["scope"] == "development_proxy"
    assert receipts["historical_runtime"]["sha256"].startswith("sha256:")
    historical = json.loads(Path(receipts["historical_runtime"]["path"]).read_text())
    assert historical["counts_as_current_invocation"] is False


def _passing_receipts() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "argv": [name],
            "environment": {},
            "duration_s": 0.1,
            "log_sha256": "sha256:fixture",
        }
        for name in (*subject.REQUIRED_CHECK_NAMES, *subject.REQUIRED_E2E_NAMES)
    ]


def _artifact_fixture(tmp_path: Path) -> dict[str, object]:
    run = subject.run_checkpointed_episodes(
        tmp_path / "checkpoints", _schedule(), FixtureExecutor()
    )
    diagnosis = subject.diagnose_historical_timeout(ROOT)
    sidecars = subject.write_provenance_sidecars(tmp_path / "sidecars", diagnosis, run)
    return subject.build_terminal_artifact(
        run_date=subject.RUN_DATE,
        started_at_utc="2026-09-18T00:00:00+00:00",
        ended_at_utc="2026-09-18T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[{"check": "fixture", "passed": True}],
        diagnosis=diagnosis,
        checkpoint_run=run,
        validation_receipts=_passing_receipts(),
        source_hashes={},
        provenance_sidecars=sidecars,
        phase_spans=[],
        terminal_lints_passed=True,
    )


def test_artifact_is_schema_complete_and_current_invocations_are_zero(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: readiness binds restart, accounting, denial, and readers."""

    artifact = _artifact_fixture(tmp_path)

    assert subject.validate_artifact(artifact) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == subject.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == subject.INFERENCE_SUBSTRATE
    assert isinstance(artifact["inference_substrate"], str)
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["arc_checkpoint_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["credited_solve"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert all(
        "generation_calls_attempted" not in row for row in artifact["provenance_sidecars"].values()
    )
    assert set(artifact) <= set(artifact["field_principles"])


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("schema", "bad", "identity"),
        ("run_date", "bad", "date"),
        ("MODEL_SPECS", ["bad"], "model declaration"),
        ("model_invoked", True, "model declaration"),
        ("invocation_counts", {}, "invocation counts"),
        ("inference_substrate", {}, "substrate"),
        ("inference_substrate_class", "bad", "substrate class"),
        ("execution_venue", "bad", "venue"),
        ("promotion_score", 1, "promotion"),
        ("credited_solve", True, "promotion"),
        ("arc_checkpoint_ready_score", 0, "readiness"),
        ("field_principles", {}, "principles"),
        ("reproducibility_checksum", "bad", "checksum"),
    ],
)
def test_artifact_validator_rejects_drift(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    """REQ-ARC-WMTE-7398: cold validation rejects terminal field drift."""

    artifact = _artifact_fixture(tmp_path)
    artifact[field] = bad
    assert any(message in error for error in subject.validate_artifact(artifact))


def test_independent_reducer_and_gate_failures_are_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: independent readers disqualify contradictions."""

    artifact = _artifact_fixture(tmp_path)
    path = tmp_path / "candidate.json"
    subject.atomic_json(path, artifact)
    reduced = subject.independent_reduce_file(path)
    assert reduced["matches_declared"] is True
    assert reduced["reduced_arc_checkpoint_ready_score"] == 1

    failed = deepcopy(artifact)
    failed["acceptance_gate_results"][0]["passed"] = False
    failed["arc_checkpoint_ready_score"] = 0
    failed["verdict_class"] = "disqualified"
    failed["honest_verdict"] = "complete_disqualified_required_evidence"
    failed["gate_check_summary"] = subject.gate_summary(failed["acceptance_gate_results"])
    failed["reproducibility_checksum"] = subject.artifact_checksum(failed)
    assert subject.validate_artifact(failed) == []
    subject.atomic_json(path, failed)
    reduced = subject.independent_reduce_file(path)
    assert reduced["reduced_arc_checkpoint_ready_score"] == 0
    assert reduced["matches_declared"] is True


def test_preconditions_validation_plans_and_helpers_are_exact(tmp_path: Path, capsys) -> None:
    """REQ-ARC-WMTE-7398: authenticate inputs and freeze Exp7358 affected checks."""

    checks, hashes, historical = subject.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert historical["flagged_adversarial"] is True
    assert historical["authorizes_current_readiness"] is False
    assert hashes[subject.HISTORICAL_ARTIFACT.as_posix()]["sha256"].startswith("sha256:")

    commands = subject.build_validation_plan(ROOT, tmp_path / "scoped")
    assert subject.validate_validation_plan(ROOT, commands) == []
    assert [row.name for row in commands] == list(subject.REQUIRED_CHECK_NAMES)
    coverage = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert dict(coverage.command_environment)["COVERAGE_FILE"].endswith("/.coverage")

    e2e = subject.e2e_command_specs(ROOT, tmp_path / "e2e")
    assert [row.name for row in e2e] == list(subject.REQUIRED_E2E_NAMES)
    assert "12" in e2e[-1].argv
    terminal = subject.terminal_command_specs(ROOT, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == [
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]

    assert subject.utc_now().endswith("+00:00")
    started = time.monotonic()
    subject.progress(started, "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    assert subject.parse_args(["--date", subject.RUN_DATE]).date == subject.RUN_DATE
    assert subject.gate_row("equal", "test", 1, 1)["passed"] is True
    assert subject.gate_row("identity", "test", None, None, operator="is")["passed"] is True
    with pytest.raises(ValueError, match="unsupported"):
        subject.gate_row("bad", "test", 1, 1, operator="!=")


def test_missing_or_malformed_checkpoint_and_blocked_precondition(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: missing journals and gates never look complete."""

    assert subject.read_episode_checkpoint(tmp_path / "missing.json", _schedule()[0]) is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]")
    with pytest.raises(subject.CheckpointIntegrityError, match="object"):
        subject.read_episode_checkpoint(malformed, _schedule()[0])
    assert subject._load_object(malformed) == {}
    malformed.write_text("{")
    assert subject._load_object(malformed) == {}
    assert subject._load_object(tmp_path / "absent.json") == {}

    artifact = _artifact_fixture(tmp_path)
    artifact["preconditions_checked"] = [
        subject.gate_row("missing", "precondition", True, None, upstream="x", artifact_field="y")
    ]
    artifact["acceptance_gate_results"][0] = subject.gate_row(
        "preconditions", "precondition", True, False, upstream="preconditions_checked"
    )
    artifact["arc_checkpoint_ready_score"] = 0
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = "blocked_required_input_or_current_gate"
    artifact["gate_check_summary"] = subject.gate_summary(artifact["acceptance_gate_results"])
    artifact["reproducibility_checksum"] = subject.artifact_checksum(artifact)
    assert subject.validate_artifact(artifact) == []


def test_builder_covers_blocked_disqualified_and_non_mapping_rows(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7398: terminal classes preserve failed and malformed inputs."""

    run = subject.run_checkpointed_episodes(
        tmp_path / "checkpoints", _schedule(), FixtureExecutor()
    )
    run["rows"].append("invalid")
    run["interruption_controls"] = [{"passed": True}]
    diagnosis = subject.diagnose_historical_timeout(ROOT)
    diagnosis["rows"].append("invalid")
    sidecars = subject.write_provenance_sidecars(tmp_path / "sidecars", diagnosis, run)
    common = {
        "run_date": subject.RUN_DATE,
        "started_at_utc": "2026-09-18T00:00:00+00:00",
        "ended_at_utc": "2026-09-18T00:00:01+00:00",
        "duration_s": 1.0,
        "diagnosis": diagnosis,
        "checkpoint_run": run,
        "validation_receipts": _passing_receipts(),
        "source_hashes": {},
        "provenance_sidecars": sidecars,
        "phase_spans": [],
    }
    blocked = subject.build_terminal_artifact(
        **common,
        preconditions=[{"passed": False}],
        terminal_lints_passed=True,
    )
    assert blocked["verdict_class"] == "blocked"
    disqualified = subject.build_terminal_artifact(
        **common,
        preconditions=[{"passed": True}],
        terminal_lints_passed=False,
    )
    assert disqualified["verdict_class"] == "disqualified"
    disqualified["arc_checkpoint_ready_score"] = 1
    assert "unsafe readiness on non-ready artifact" in subject.validate_artifact(disqualified)
    span = subject._phase("fixture", time.monotonic(), time.monotonic(), 1)
    assert span["phase"] == "fixture" and span["completed_units"] == 1
