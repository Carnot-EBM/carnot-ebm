"""Tests for Exp 4861 generation-wall induce-plan fork probe.

Spec refs: REQ-ARC-WMTE-4861,
SCENARIO-ARC-WMTE-4861-BLOCKED-PRECONDITION,
SCENARIO-ARC-WMTE-4861-JOINT-FORK,
SCENARIO-ARC-WMTE-4861-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4861_generation_wall_fork_probe as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _act(action: int, data: dict[str, int] | None = None) -> dict[str, Any]:
    return {"action": action, "data": data}


def _ground_truth() -> dict[str, list[dict[str, Any]]]:
    return {
        "cd82": [_act(1), _act(5)],
        "cn04": [_act(2)],
        "ls20": [_act(3)],
        "m0r0": [_act(4)],
        "tu93": [_act(1)],
    }


def _row(
    game: str,
    *,
    accuracy: float,
    bucket: str,
    prefix_len: int = 1,
) -> dict[str, Any]:
    return {
        "game": game,
        "engine_heldout_accuracy": accuracy,
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED" and game != "tu93",
        "winning_prefix_len": prefix_len,
        "planned_prefix_len": prefix_len if bucket != "NEVER_ENUMERATED" else 0,
        "planned_pool_size": 1 if bucket != "NEVER_ENUMERATED" else 0,
        "heldout_transition_count": 5,
        "plan_length": prefix_len if bucket != "NEVER_ENUMERATED" else 0,
        "planner_reached_l1_win": bucket == "COVERED",
        "live_path_methods_called": [
            "E3AgentPolicy._induce_and_plan",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _control(bucket: str = "COVERED", accuracy: float = 0.9) -> dict[str, Any]:
    return _row("tu93", accuracy=accuracy, bucket=bucket)


def test_req_arc_wmte_4861_spec_declares_fork_probe_contract() -> None:
    """REQ-ARC-WMTE-4861: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4861",
        "SCENARIO-ARC-WMTE-4861-BLOCKED-PRECONDITION",
        "SCENARIO-ARC-WMTE-4861-JOINT-FORK",
        "SCENARIO-ARC-WMTE-4861-PARTIAL-CHECKPOINT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4861_classifies_planned_pool_against_banked_prefix() -> None:
    """SCENARIO-ARC-WMTE-4861-JOINT-FORK: planned pools get the three coverage buckets."""

    winner = [_act(1), _act(6, {"x": 3, "y": 4})]

    assert (
        mod.classify_planned_pool(
            "aa00",
            winner,
            [_act(1), _act(6, {"x": 3, "y": 4})],
            planner_reached_l1_win=True,
        )["planned_bucket"]
        == "COVERED"
    )
    lost = mod.classify_planned_pool(
        "aa00",
        winner,
        [_act(1), _act(6, {"x": 3, "y": 4})],
        planner_reached_l1_win=False,
    )
    missing = mod.classify_planned_pool(
        "aa00",
        winner,
        [_act(1)],
        planner_reached_l1_win=False,
    )
    first_mismatch = mod.classify_planned_pool(
        "aa00",
        winner,
        [_act(2)],
        planner_reached_l1_win=False,
    )

    assert lost["planned_bucket"] == "ENUMERATED_BUT_LOST"
    assert lost["planned_prefix_len"] == 2
    assert lost["migrated"] is False
    assert missing["planned_bucket"] == "NEVER_ENUMERATED"
    assert missing["planned_prefix_len"] == 1
    assert first_mismatch["planned_prefix_len"] == 0


def test_scenario_arc_wmte_4861_joint_fork_verdicts_are_deterministic() -> None:
    """SCENARIO-ARC-WMTE-4861-JOINT-FORK: accuracy x migration names the fork."""

    high_migration = mod.build_artifact(
        per_game_fork={
            "cd82": _row("cd82", accuracy=0.8, bucket="COVERED"),
            "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
            "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
    )
    high_no_migration = mod.build_artifact(
        per_game_fork={
            "cd82": _row("cd82", accuracy=0.8, bucket="NEVER_ENUMERATED"),
            "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
            "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
    )
    low_no_migration = mod.build_artifact(
        per_game_fork={
            "cd82": _row("cd82", accuracy=0.2, bucket="NEVER_ENUMERATED"),
            "cn04": _row("cn04", accuracy=0.3, bucket="NEVER_ENUMERATED"),
            "ls20": _row("ls20", accuracy=0.4, bucket="NEVER_ENUMERATED"),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
    )

    assert high_migration["fork_verdict"] == "GUIDANCE_WALL"
    assert high_migration["honest_verdict"] == (
        "complete_generation_wall_guidance_wall_high_accuracy_migration"
    )
    assert high_no_migration["fork_verdict"] == "PLANNER_GAP"
    assert high_no_migration["honest_verdict"] == (
        "complete_generation_wall_planner_gap_high_accuracy_no_migration"
    )
    assert low_no_migration["fork_verdict"] == "INDUCER_CEILING"
    assert low_no_migration["honest_verdict"] == (
        "complete_generation_wall_inducer_ceiling_low_accuracy_no_migration"
    )
    assert high_migration["coverage_migration_count"] == 1
    assert mod.artifact_schema_errors(high_migration) == []
    assert mod.artifact_schema_errors(high_no_migration) == []
    assert mod.artifact_schema_errors(low_no_migration) == []


def test_scenario_arc_wmte_4861_run_blocks_missing_preconditions(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4861-BLOCKED-PRECONDITION: missing resources never fabricate a fork."""

    common = {
        "root": tmp_path,
        "ground_truth_loader": lambda _root: _ground_truth(),
        "game_measurer": lambda game, **_kwargs: _row(game, accuracy=0.8, bucket="COVERED"),
        "positive_control_runner": lambda **_kwargs: _control(),
        "live_path_checker": lambda _root: True,
        "now": iter([1.0, 1.1]).__next__,
        "write": False,
    }

    blocked_arcade = mod.run(
        **common,
        offline_arcade_checker=lambda: False,
        generator_checker=lambda: {"ok": True},
        environment_games_loader=lambda _arcade: set(_ground_truth()),
    )
    blocked_generator = mod.run(
        **{**common, "now": iter([2.0, 2.1]).__next__},
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": False, "detail": "missing_qwen"},
        environment_games_loader=lambda _arcade: set(_ground_truth()),
    )
    blocked_games = mod.run(
        **{**common, "now": iter([3.0, 3.1]).__next__},
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": True},
        environment_games_loader=lambda _arcade: {"cd82", "tu93"},
    )

    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_arcade["per_game_fork"] == {}
    assert blocked_generator["honest_verdict"] == "blocked_generator_unavailable"
    assert blocked_generator["preconditions_checked"]["generator"]["ok"] is False
    assert blocked_games["honest_verdict"] == "blocked_no_heldout_games"
    assert blocked_games["n_games_measured"] == 0


def test_scenario_arc_wmte_4861_run_writes_checkpoints_and_partial(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4861-PARTIAL-CHECKPOINT: elapsed cap emits a resumable artifact."""

    times = iter([10.0, 10.1, 10.2])
    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": True},
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: _row(game, accuracy=0.2, bucket="NEVER_ENUMERATED"),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=times.__next__,
        write=True,
        soft_elapsed_budget_s=0.05,
        heldout_games=("cd82", "cn04", "ls20"),
    )

    checkpoint = tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json"
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert checkpoint.exists()
    assert loaded == artifact
    assert artifact["partial"] is True
    assert artifact["checkpoint_emitted"] is True
    assert artifact["n_games_measured"] == 1
    assert artifact["honest_verdict"] == "complete_generation_wall_fork_probe_partial_budget_stop"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4861_schema_errors_are_explicit() -> None:
    """REQ-ARC-WMTE-4861: malformed artifacts fail closed with named errors."""

    artifact = mod.build_artifact(
        per_game_fork={
            "cd82": _row("cd82", accuracy=0.8, bucket="COVERED"),
            "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
            "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
    )
    malformed = dict(artifact)
    malformed.update(
        {
            "honest_verdict": "not_terminal",
            "fork_verdict": "MAYBE",
            "per_game_fork": {"cd82": {"planned_bucket": "MAYBE"}},
            "positive_control_migrated": False,
            "planner_blind_to_banked_answer": False,
            "verifier_is_oracle": False,
            "live_path_reachable": False,
            "solve_provenance": "live_agent_self_discovery",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "checkpoint_emitted": "yes",
            "reproducibility_checksum": "sha256:bad",
        }
    )

    errors = mod.artifact_schema_errors(malformed)

    assert "honest_verdict_terminal_prefix" in errors
    assert "fork_verdict" in errors
    assert "per_game_fork.cd82.planned_bucket" in errors
    assert "positive_control_migrated" in errors
    assert "planner_blind_to_banked_answer" in errors
    assert "verifier_is_oracle" in errors
    assert "live_path_reachable" in errors
    assert "solve_provenance" in errors
    assert "inference_substrate" in errors
    assert "checkpoint_emitted" in errors
    assert "reproducibility_checksum" in errors

    assert mod.artifact_schema_errors({})[0].startswith("missing_field:")
    assert "field_principles" in mod.artifact_schema_errors(dict(artifact, field_principles=[]))
    bad_principles = {
        **artifact["field_principles"],
        "honest_verdict": {"principle": "different"},
    }
    assert "field_principles.honest_verdict" in mod.artifact_schema_errors(
        dict(artifact, field_principles=bad_principles)
    )
    assert "per_game_fork" in mod.artifact_schema_errors(
        dict(artifact, per_game_fork=[], n_games_measured=0)
    )
    assert "per_game_fork.cd82" in mod.artifact_schema_errors(
        dict(artifact, per_game_fork={"cd82": []}, n_games_measured=1)
    )
    invalid_rows = dict(artifact)
    invalid_rows.update(
        {
            "per_game_fork": {
                "cd82": {
                    "planned_bucket": "COVERED",
                    "engine_heldout_accuracy": 1.5,
                    "winning_prefix_len": 0,
                    "planned_pool_size": -1,
                    "heldout_transition_count": -1,
                    "migrated": "yes",
                }
            },
            "n_games_measured": "bad",
            "median_engine_heldout_accuracy": 1.5,
            "coverage_migration_count": 9,
            "retire_if_same_verdict": False,
        }
    )
    row_errors = mod.artifact_schema_errors(invalid_rows)
    assert "per_game_fork.cd82.engine_heldout_accuracy" in row_errors
    assert "per_game_fork.cd82.winning_prefix_len" in row_errors
    assert "per_game_fork.cd82.planned_pool_size" in row_errors
    assert "per_game_fork.cd82.heldout_transition_count" in row_errors
    assert "per_game_fork.cd82.migrated" in row_errors
    assert "n_games_measured" in row_errors
    assert "coverage_migration_count" in row_errors
    assert "median_engine_heldout_accuracy" in row_errors
    assert "retire_if_same_verdict" in row_errors

    blocked = mod.build_blocked_artifact(
        "blocked_test",
        preconditions_checked={},
        duration_s=0.0,
    )
    blocked_bad = dict(
        blocked, per_game_fork={"cd82": _row("cd82", accuracy=0.2, bucket="COVERED")}
    )
    blocked_bad["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_bad)
    assert "blocked_artifact_has_fork_rows" in mod.artifact_schema_errors(blocked_bad)

    partial_without_rows = mod.build_artifact(
        per_game_fork={},
        positive_control_game="tu93",
        positive_control_row=None,
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=0.0,
        partial=True,
        checkpoint_emitted=False,
    )
    assert "partial_without_rows" in mod.artifact_schema_errors(partial_without_rows)

    retired_positive = mod.build_artifact(
        per_game_fork={
            "cd82": _row("cd82", accuracy=0.8, bucket="COVERED"),
            "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
            "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
        },
        positive_control_game="tu93",
        positive_control_row=_control(bucket="NEVER_ENUMERATED", accuracy=0.1),
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
    )
    assert retired_positive["honest_verdict"] == (
        "complete_generation_wall_fork_probe_retired_positive_control_failed"
    )

    retired_no_table = mod.build_artifact(
        per_game_fork={"cd82": _row("cd82", accuracy=0.8, bucket="COVERED")},
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
    )
    assert retired_no_table["honest_verdict"] == (
        "complete_generation_wall_fork_probe_retired_no_joint_table"
    )

    try:
        mod._validate_or_raise({"bad": True})
    except mod.DiagnosticError as exc:
        assert "missing_field:" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("DiagnosticError not raised")


def test_req_arc_wmte_4861_run_full_and_helper_branches(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-ARC-WMTE-4861: deterministic helper and resume branches are covered."""

    assert mod._normalise_generator_result(False) == {"ok": False}
    assert (
        mod._positive_control_passed(
            {"planned_bucket": "COVERED", "engine_heldout_accuracy": "bad"}
        )
        is False
    )
    assert mod._load_checkpoint("missing", root=tmp_path) is None
    bad_checkpoint = tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "bad.json"
    bad_checkpoint.parent.mkdir(parents=True)
    bad_checkpoint.write_text("{", encoding="utf-8")
    assert mod._load_checkpoint("bad", root=tmp_path) is None

    blocked = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: False,
        generator_checker=lambda: {"ok": True},
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: _row(game, accuracy=0.8, bucket="COVERED"),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([40.0, 40.1]).__next__,
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    proposer = object()
    monkeypatch.setattr(mod, "make_live_qwen_proposer", lambda: proposer)
    monkeypatch.setattr(
        mod,
        "generator_available",
        lambda *, proposer: {"ok": True, "proposer_id": id(proposer)},
    )
    blocked_live_path = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=None,
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: _row(game, accuracy=0.8, bucket="COVERED"),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: False,
        now=iter([45.0, 45.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
    )
    assert blocked_live_path["honest_verdict"] == "blocked_live_path_unreachable"
    assert blocked_live_path["preconditions_checked"]["generator"]["proposer_id"] == id(proposer)

    rows = {
        "cd82": _row("cd82", accuracy=0.8, bucket="COVERED"),
        "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
        "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
    }
    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: True,
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: dict(rows[game]),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([50.0, 50.1, 50.2, 50.3, 50.4]).__next__,
        write=True,
        heldout_games=("cd82", "cn04", "ls20"),
    )

    assert artifact["fork_verdict"] == "GUIDANCE_WALL"
    assert artifact["positive_control_migrated"] is True
    assert mod.artifact_schema_errors(artifact) == []

    resumed = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": True},
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: dict(rows[game]),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([60.0, 60.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
    )

    assert resumed["n_games_measured"] == 3
    assert resumed["checkpoint_emitted"] is True


def test_req_arc_wmte_4861_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4861: final artifact is the requested diagnostic deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["positive_control_game"] == "tu93"
    assert artifact["planner_blind_to_banked_answer"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    if not artifact["honest_verdict"].startswith("blocked_"):
        assert artifact["n_games_measured"] >= 1
        assert artifact["checkpoint_emitted"] is True
