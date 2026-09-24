"""Tests for REQ-ARC-WMTE-7589 output custody and causal alias observation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot import experiment_7589_v663_arc_output_boundary as exp


def frame(value: int, *, level: int = 0) -> SimpleNamespace:
    """Build one public observation without game state or source access."""

    grid = np.full((3, 4), value, dtype=np.int16)
    return SimpleNamespace(frame=[grid.tolist()], levels_completed=level)


def action(action_id: int = 6, *, x: int = 2, y: int = 3) -> tuple[int, dict[str, int]]:
    return action_id, {"x": x, "y": y}


def test_default_off_and_exact_opt_in(monkeypatch) -> None:
    """REQ-ARC-WMTE-7589 keeps the observer default-off and exact-value gated."""

    monkeypatch.delenv(exp.OBSERVER_ENV, raising=False)
    assert exp.maybe_make_observable_aliasing_observer("xx11") is None
    monkeypatch.setenv(exp.OBSERVER_ENV, "true")
    assert exp.maybe_make_observable_aliasing_observer("xx11") is None
    monkeypatch.setenv(exp.OBSERVER_ENV, "1")
    assert isinstance(
        exp.maybe_make_observable_aliasing_observer("xx11"),
        exp.ObservableStateAliasingObserver,
    )


def test_causal_keys_exist_before_and_ignore_next_frame() -> None:
    """SCENARIO-ARC-WMTE-7589-CAUSAL-HISTORY forbids future-frame keys."""

    left = exp.ObservableStateAliasingObserver("xx11")
    right = exp.ObservableStateAliasingObserver("xx11")
    chosen = action()
    assert left.observe(frame(1), chosen, level=0) == chosen
    assert right.observe(frame(1), chosen, level=0) == chosen
    left_before = left.snapshot()["pending"]
    right_before = right.snapshot()["pending"]

    assert left_before["keys"] == right_before["keys"]
    assert left_before["target_sha256"] is None
    assert right_before["target_sha256"] is None
    assert set(left_before["keys"]) == {"0", "1", "2", "4"}

    left.observe(frame(2), action(1), level=0)
    right.observe(frame(9), action(1), level=0)
    left_event = left.snapshot()["events"][0]
    right_event = right.snapshot()["events"][0]
    assert left_event["keys"] == right_event["keys"]
    assert left_event["target_sha256"] != right_event["target_sha256"]


def test_coordinate_identity_and_history_lengths() -> None:
    """REQ-ARC-WMTE-7589 retains action coordinates and 0/1/2/4 histories."""

    left = exp.ObservableStateAliasingObserver("xx11")
    right = exp.ObservableStateAliasingObserver("xx11")
    left.observe(frame(1), action(x=2, y=3), level=0)
    right.observe(frame(1), action(x=3, y=2), level=0)
    left_pending = left.snapshot()["pending"]
    right_pending = right.snapshot()["pending"]

    assert left_pending["action"] == {"action": 6, "coordinates": {"x": 2, "y": 3}}
    assert right_pending["action"] == {"action": 6, "coordinates": {"x": 3, "y": 2}}
    assert left_pending["keys"]["0"] != right_pending["keys"]["0"]

    observer = exp.ObservableStateAliasingObserver("xx11")
    for value in range(1, 7):
        observer.observe(frame(value), action(1, x=value, y=0), level=0)
    pending = observer.snapshot()["pending"]
    assert pending["history_items_used"] == {"0": 0, "1": 1, "2": 2, "4": 4}


def test_repeated_key_conflict_and_singleton_unknown_support() -> None:
    """REQ-ARC-WMTE-7589 separates unknown singleton support from conflicts."""

    observer = exp.ObservableStateAliasingObserver("xx11")
    observer.observe(frame(1), action(), level=0)
    observer.observe(frame(2), action(1), level=0)
    first = observer.snapshot()["support_summary"]["0"]
    assert first["singleton_unknown_support"] == 1
    assert first["conflicting_keys"] == 0

    observer.observe(frame(1), action(), level=0)
    observer.observe(frame(9), action(1), level=0)
    summary = observer.snapshot()["support_summary"]["0"]
    assert summary["conflicting_keys"] == 1
    assert summary["contradictory_target_count"] == 2


def test_level_boundary_finalizes_then_clears_history() -> None:
    """SCENARIO-ARC-WMTE-7589-PARITY-AND-LEVELS clears prior-level context."""

    observer = exp.ObservableStateAliasingObserver("xx11")
    observer.observe(frame(1, level=0), action(1), level=0)
    observer.observe(frame(2, level=0), action(2), level=0)
    observer.observe(frame(3, level=1), action(3), level=1)
    snapshot = observer.snapshot()

    assert snapshot["events"][-1]["level_boundary"] is True
    assert snapshot["events"][-1]["level_before"] == 0
    assert snapshot["events"][-1]["level_after"] == 1
    assert snapshot["level_boundary_count"] == 1
    assert snapshot["pending"]["history_items_used"] == {"0": 0, "1": 0, "2": 0, "4": 0}


def test_ring_buffer_evicts_support_and_emits_receipts() -> None:
    """SCENARIO-ARC-WMTE-7589-BOUNDS retains only bounded hashed support."""

    observer = exp.ObservableStateAliasingObserver("xx11", max_events=3)
    for value in range(7):
        observer.observe(frame(value), action(1, x=value, y=value + 1), level=0)
    snapshot = observer.snapshot()

    assert snapshot["max_events"] == 3
    assert snapshot["finalized_event_count"] == 6
    assert snapshot["live_event_count"] == 3
    assert snapshot["evictions"]["count"] == 3
    assert len(snapshot["evictions"]["recent"]) == 3
    assert snapshot["evictions"]["hash_chain"].startswith("sha256:")
    assert sum(row["live_key_count"] for row in snapshot["support_summary"].values()) <= 12
    serialized = json.dumps(snapshot, sort_keys=True)
    assert "[[" not in serialized
    assert "game_state" not in serialized
    assert "source" not in serialized
    assert "adapter" not in serialized


def test_protected_output_failure_is_reproduced_in_miniature(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7589-OUTPUT-BOUNDARY preserves the results guard."""

    receipt = exp.reproduce_protected_output_failure(tmp_path / "contract")

    assert receipt["passed"] is True
    assert receipt["exit_code"] == 2
    assert receipt["guard_message"] == "e3 requires --output outside results/ (immutable evidence)"
    assert receipt["output_exists"] is False
    assert Path(receipt["miniature_root"]).is_relative_to(tmp_path)
    assert Path(receipt["attempted_output"]).is_relative_to(
        Path(receipt["miniature_root"]) / "results"
    )


def test_e2e_commands_use_only_private_output_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 moves every live pytest and smoke output below /tmp."""

    root = Path(__file__).resolve().parents[2]
    private = tmp_path / "owned"
    commands = exp.build_e2e_commands(root, private)
    assert [command.name for command in commands] == [
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "e2e_012",
        "e2e_013",
        "foreign_cwd_llm_off_e3_smoke",
    ]
    for command in commands:
        joined = " ".join(command.argv)
        for argument in command.argv:
            if argument.startswith("--basetemp="):
                assert Path(argument.split("=", 1)[1]).is_relative_to(private)
        if command.name == "foreign_cwd_llm_off_e3_smoke":
            output = Path(command.argv[command.argv.index("--output") + 1])
            foreign = Path(command.argv[command.argv.index("-C") + 1])
            assert output.is_relative_to(private)
            assert foreign.is_relative_to(private)
            assert "CARNOT_ARC_DISABLE_INDUCTION=1" in command.argv
        assert str(root / "results") not in joined

    for command in commands:
        exp.prepare_command_parent(command)
    assert (private / "foreign-cwd").is_dir()


def test_real_e3_boundary_policy_parity(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7589-PARITY-AND-LEVELS compares the final action seam."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = exp.measure_policy_parity(n_actions=10, seed=7589)

    assert result["passed"] is True
    assert result["action_count"] == len(result["rows"]) > 0
    assert result["observer_diagnostics"]["enabled"] is True
    assert result["observer_diagnostics"]["finalized_event_count"] > 0
    for row in result["rows"]:
        assert row["equal"] is True
        assert row["off_action"] == row["on_action"]
        assert row["off_coordinates"] == row["on_coordinates"]
        assert row["off_terminated"] == row["on_terminated"]


def test_learning_lifecycle_persists_and_rejects_duplicate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 exercises delayed durable learning without weights."""

    receipt = exp.exercise_learning_lifecycle(tmp_path / "learning")

    assert receipt["operations"] == [
        "predict",
        "release",
        "update",
        "persist",
        "reload",
        "duplicate_rejection",
    ]
    assert receipt["prediction_label_available"] is False
    assert receipt["state_changed_after_update"] is True
    assert receipt["reload_equal"] is True
    assert receipt["duplicate_rejected"] is True
    assert receipt["state_unchanged_after_duplicate"] is True
    assert receipt["model_weights_changed"] is False
    assert receipt["passed"] is True


def test_fixture_measurement_covers_every_required_causal_case() -> None:
    """REQ-ARC-WMTE-7589 reduces all observer requirements from raw fixtures."""

    fixtures = exp.measure_observer_fixtures()

    assert fixtures["passed"] is True
    assert set(fixtures["checks"]) == {
        "history_causality",
        "level_reset_clearing",
        "coordinate_identity",
        "repeated_key_conflicts",
        "singleton_unknown_support",
        "bounded_memory",
    }
    assert all(row["passed"] for row in fixtures["checks"].values())
    assert exp.independent_reduce(fixtures["rows"]) == fixtures["independent_reduction"]


def test_complete_artifact_has_required_fields_and_null_verdict(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7589-TERMINAL keeps readiness separate from benefit."""

    artifact = exp.build_test_artifact(tmp_path)
    errors = exp.validate_artifact(artifact)

    assert errors == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "null"
    assert artifact["flagged_adversarial"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["planned_inference_substrate_class"] == "no_model_load"
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["arc_output_boundary_ready_score"] == 1
    assert artifact["history_observer_ready_score"] == 1
    assert artifact["acceptance_gate_results"]["validity"]["passed"] is True
    assert artifact["acceptance_gate_results"]["readiness"]["passed"] is True
    assert artifact["acceptance_gate_results"]["benefit"]["passed"] is False
    assert artifact["acceptance_gate_results"]["retention"]["passed"] is True
    assert artifact["acceptance_gate_results"]["freshness"]["passed"] is True
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"].values())
    assert artifact["verifier_is_oracle"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["solve_improvement_claimed"] is False
    assert artifact["production_defaults_changed"] is False
    assert artifact["world_model_acceptance_changed"] is False
    assert artifact["hud_masks_changed"] is False
    assert artifact["thinking_settings_changed"] is False
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)


def test_artifact_rows_and_policy_rows_are_auditable(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 preserves raw row operands and per-action equality."""

    artifact = exp.build_test_artifact(tmp_path)
    required = {
        "unit",
        "arm",
        "absolute_metric",
        "numerator",
        "denominator",
        "seed",
        "direction",
        "missing",
        "censored",
        "provenance",
    }

    assert artifact["rows"]
    assert all(required <= set(row) for row in artifact["rows"])
    assert artifact["policy_parity_rows"]
    assert all(row["equal"] for row in artifact["policy_parity_rows"])
    assert artifact["sample_size_budget"] == {
        "intended_independent_units": 8,
        "observed_independent_units": 8,
        "excluded_independent_units": 0,
        "censored_independent_units": 0,
        "seeds_or_windows_multiply_source_groups": False,
    }
    assert artifact["independent_reduction"] == exp.independent_reduce(artifact["rows"])


def test_failed_e2e_cannot_open_output_readiness(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 derives readiness from exact unchanged E2E receipts."""

    artifact = exp.build_test_artifact(tmp_path)
    artifact["e2e_receipts"][0]["passed"] = False
    artifact["e2e_receipts"][0]["exit_code"] = 1
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)

    errors = exp.validate_artifact(artifact)
    assert "arc_output_boundary_ready_score" in errors
    assert "readiness_gate" in errors


def test_blocked_artifact_names_every_gate_operand() -> None:
    """REQ-ARC-WMTE-7589 records exact external precondition failures."""

    artifact = exp.build_blocked_artifact(
        run_date="20260924",
        duration_s=0.1,
        check="missing_source",
        upstream="worktree",
        path="missing.json",
        field="is_file",
        op="==",
        expected=True,
        observed=False,
    )

    assert artifact["honest_verdict"] == "complete_blocked_missing_source"
    assert artifact["verdict_class"] == "blocked"
    assert set(artifact["gate_check_summary"]) == {
        "check",
        "upstream",
        "path",
        "field",
        "op",
        "expected",
        "observed",
    }
    assert exp.validate_artifact(artifact) == []


def test_preconditions_authenticate_sources_not_future_outputs(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 treats new output paths as products, not prerequisites."""

    root = Path(__file__).resolve().parents[2]
    checks, hashes = exp.collect_preconditions(root, tmp_path / "not-created-yet")

    assert checks and all(row["passed"] for row in checks)
    assert exp.REQUIREMENT_ID in (root / exp.SPEC_REL).read_text(encoding="utf-8")
    assert exp.RESULT_REL.as_posix() not in {row["path"] for row in checks}
    assert exp.RAW_REL.as_posix() not in {row["path"] for row in checks}
    assert all(value.startswith("sha256:") for value in hashes.values())


def test_manifest_freezes_only_affected_files() -> None:
    """SCENARIO-ARC-WMTE-7589-TERMINAL freezes the validation scope."""

    manifest = exp.affected_validation_manifest()
    assert manifest["test_paths"] == [
        "tests/python/test_experiment_7589_v663_arc_output_boundary.py"
    ]
    assert manifest["changed_modules"] == [
        "python/carnot/experiment_7589_v663_arc_output_boundary.py"
    ]
    assert manifest["static_paths"] == [
        "scripts/experiments/experiment_7589_v663_arc_output_boundary.py",
        "python/carnot/agentic/arc_competition_agent.py",
    ]


def test_json_hash_fallbacks_and_observer_error_counter(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 keeps hashing stable for supported public values."""

    value = {
        "array": np.asarray([1, 2], dtype=np.int16),
        "scalar": np.int64(3),
        "path": tmp_path,
    }
    assert exp._jsonable(value) == {
        "array": [1, 2],
        "scalar": 3,
        "path": str(tmp_path),
    }
    object_frame = np.asarray([["a", "b"]], dtype=object)
    assert exp.full_frame_hash(object_frame).startswith("sha256:")
    observer = exp.ObservableStateAliasingObserver("xx11")
    move = action()
    assert observer.observe(None, move) == move
    observer.note_error()
    assert observer.snapshot()["error_count"] == 1


def test_invalid_row_operands_are_rejected() -> None:
    """REQ-ARC-WMTE-7589 refuses denominator and numerator corruption."""

    with pytest.raises(ValueError, match="invalid_row_operands"):
        exp.independent_reduce([{"unit": "bad", "numerator": 2, "denominator": 1}])


def test_failed_parity_child_is_explicit(monkeypatch) -> None:
    """REQ-ARC-WMTE-7589 does not invent parity after child failure."""

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7, stdout="", stderr="failed"),
    )
    result = exp.measure_policy_parity(n_actions=1, seed=1)
    assert result["passed"] is False
    assert result["child_exit_code"] == 7
    assert result["rows"] == []


def test_progress_and_command_builders(capsys, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 emits progress and declares every scoped command."""

    root = Path(__file__).resolve().parents[2]
    exp.progress(0.0, "test", "plain")
    exp.progress(0.0, "test", "detailed", units=1)
    output = capsys.readouterr().out
    assert "event=plain" in output
    assert "event=detailed" in output and "units=1" in output

    validation = exp.build_validation_commands(root, tmp_path / "validation")
    assert [row.name for row in validation] == [
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
    ]
    coverage_commands = [
        row
        for row in validation
        if row.name in {"changed_module_coverage", "changed_module_coverage_report"}
    ]
    assert all(
        any(arg.startswith("COVERAGE_FILE=") for arg in row.argv) for row in coverage_commands
    )
    terminal = exp.build_terminal_commands(root, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == [
        "declared_entrypoint",
        "fresh_process_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]


def test_cold_and_independent_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7589-TERMINAL replays exact persisted rows."""

    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "artifact.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path) == []
    assert exp.independent_replay(path) == []

    artifact["rows"] = "invalid"
    exp.atomic_json(path, artifact)
    assert exp.independent_replay(path) == ["rows"]
    artifact["rows"] = [{"unit": "bad", "numerator": 2, "denominator": 1}]
    exp.atomic_json(path, artifact)
    assert exp.independent_replay(path)[0].startswith("independent_reduction:")


def test_source_hash_extension() -> None:
    """REQ-ARC-WMTE-7589 binds the current implementation and static seam."""

    root = Path(__file__).resolve().parents[2]
    hashes = exp._source_hashes(root, {"upstream": "sha256:0"})
    assert hashes["upstream"] == "sha256:0"
    assert hashes[exp.MODULE_REL.as_posix()].startswith("sha256:")
    assert hashes[exp.WRAPPER_REL.as_posix()].startswith("sha256:")


def test_complete_artifact_mutations_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7589 validates each terminal declaration independently."""

    base = exp.build_test_artifact(tmp_path)

    def check(mutator, expected: str, *, refresh_checksum: bool = True) -> None:
        value = deepcopy(base)
        mutator(value)
        if refresh_checksum:
            value["reproducibility_checksum"] = exp.reproducibility_checksum(value)
        assert expected in exp.validate_artifact(value)

    check(lambda value: value.update(honest_verdict="bad"), "honest_verdict")
    check(lambda value: value.update(verdict_class="unknown"), "verdict_class")
    check(lambda value: value.update(MODEL_SPECS=[{"id": "forbidden"}]), "model_declaration")
    check(lambda value: value["invocation_counts"].update(input_tokens=1), "invocation_counts")
    check(lambda value: value.update(field_principles={}), "field_principles")
    check(lambda value: value.update(verifier_is_oracle=False), "verifier_is_oracle")
    check(lambda value: value.update(solve_provenance="stored_route"), "solve_provenance")
    check(lambda value: value.update(solve_improvement_claimed=True), "solve_improvement_claimed")
    for field in (
        "production_defaults_changed",
        "world_model_acceptance_changed",
        "hud_masks_changed",
        "thinking_settings_changed",
        "observer_default_enabled",
    ):
        check(lambda value, field=field: value.update({field: True}), field)
    check(
        lambda value: value.update(duration_s=2.0),
        "reproducibility_checksum",
        refresh_checksum=False,
    )
    check(
        lambda value: value.update(inference_substrate_class="aggregation"),
        "inference_substrate_class",
    )
    check(
        lambda value: value.update(planned_inference_substrate_class="model_bounded_generation"),
        "planned_inference_substrate_class",
    )
    check(lambda value: value.update(rows=[]), "rows")
    check(lambda value: value.update(independent_reduction={}), "independent_reduction")
    check(
        lambda value: value["rows"][0].update(numerator=2, denominator=1),
        "independent_reduction",
    )
    check(
        lambda value: value.update(history_observer_ready_score=0), "history_observer_ready_score"
    )
    check(lambda value: value.update(acceptance_gate_results={}), "acceptance_gate_results")
    check(
        lambda value: value["acceptance_gate_results"]["readiness"].update(passed=False),
        "readiness_gate",
    )
    check(
        lambda value: value["acceptance_gate_results"]["benefit"].update(passed=True),
        "benefit_gate",
    )
    check(
        lambda value: value["acceptance_gate_results"]["validity"].update(principle=""),
        "gate_principles",
    )
    check(lambda value: value.update(policy_parity_rows=[]), "policy_parity_rows")
    check(
        lambda value: value["sample_size_budget"].update(observed_independent_units=7),
        "sample_size_budget",
    )
    check(
        lambda value: value["source_artifact_hashes"].update(authenticated_sources={}),
        "source_artifact_hashes",
    )
    check(
        lambda value: value["acceptance_gate_results"]["validity"].update(passed=False),
        "null_operational_gates",
    )


def test_blocked_artifact_mutations_fail_closed() -> None:
    """REQ-ARC-WMTE-7589 keeps blocked substrate and scores exact."""

    base = exp.build_blocked_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        check="missing",
        upstream="worktree",
        path="missing",
        field="is_file",
        op="==",
        expected=True,
        observed=False,
    )

    def check(field: str, value: Any, expected: str) -> None:
        artifact = deepcopy(base)
        artifact[field] = value
        artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
        assert expected in exp.validate_artifact(artifact)

    check("gate_check_summary", {}, "gate_check_summary")
    check("inference_substrate_class", "no_model_load", "inference_substrate_class")
    check(
        "planned_inference_substrate_class",
        "aggregation",
        "planned_inference_substrate_class",
    )
    check("arc_output_boundary_ready_score", 1, "arc_output_boundary_ready_score")
    check("history_observer_ready_score", 1, "history_observer_ready_score")


def test_replay_reports_fixture_and_reduction_mismatches(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7589-TERMINAL names persisted replay drift."""

    artifact = exp.build_test_artifact(tmp_path)
    artifact["observer_fixture_evidence"]["rows"] = []
    artifact["policy_parity_rows"] = []
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    path = tmp_path / "mutated.json"
    exp.atomic_json(path, artifact)
    errors = exp.cold_replay(path)
    assert "cold_fixture_rows_mismatch" in errors
    assert "cold_policy_parity_rows_mismatch" in errors

    artifact = exp.build_test_artifact(tmp_path / "second")
    artifact["independent_reduction"] = {}
    exp.atomic_json(path, artifact)
    assert exp.independent_replay(path) == ["independent_reduction"]
