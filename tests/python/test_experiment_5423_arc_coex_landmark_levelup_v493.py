"""Tests for Exp5423 ARC CoEx landmark level-up attempt.

Spec refs: REQ-ARC-FCP-5423,
SCENARIO-ARC-FCP-5423.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5423_arc_coex_landmark_levelup_v493 as exp5423
from carnot.agentic import arc_live_trajectory_frontier as live_frontier
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_live_trajectory_frontier import LiveCoExLandmarkFrontierGenerator


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(lf52_levels: int = 2, *, include_alternate: bool = True) -> dict[str, Any]:
    games: list[dict[str, Any]] = [
        {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": lf52_levels},
    ]
    if include_alternate:
        games.append(
            {"game": "re86", "reproducibility": "reproduced", "levels_reproduced": 2}
        )
    return {"reproducible_total_levels": 69, "games": games}


def _frame(*, level: int = 0, changed: bool = False) -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[2:10, 2:18] = 8
    grid[14:16, 14:16] = 9
    if changed:
        grid[14, 14] = 10
    return SimpleNamespace(frame=grid, available_actions=[1, 6], levels_completed=level)


def _candidates() -> list[ArcAction]:
    return [
        ArcAction(6, {"x": 14, "y": 14}, "button_like_blob"),
        ArcAction(6, {"x": 4, "y": 4}, "large_flat_blob"),
        ArcAction(1, None, "keyboard"),
    ]


def _preconditions() -> dict[str, Any]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "spec_has_req_5423": True,
        "registry_present": True,
        "offline_arcade_available": True,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
    }


def _lint_result(passed: bool = True) -> dict[str, Any]:
    return {"command": "arc_levelup_guarantee_lint", "passed": passed, "returncode": 0}


def _null_attempt() -> dict[str, Any]:
    return {
        "attempt_count": 4,
        "reset_count": 1,
        "max_level_reached": 2,
        "offline_reproduced": False,
        "failure_mode": "bounded_budget_no_levelup",
        "frontier_expansion_count": 1,
        "landmark_count": 1,
        "discovered_landmarks": [{"frame_hash": "abc", "level_after": 2, "score": 4.0}],
        "frontier_transitions": [
            {"from_hash": "root", "to_hash": "abc", "action": 6, "accepted": True}
        ],
        "action_sequence_receipts": [
            {
                "sequence": [{"action": 6, "data": {"x": 14, "y": 14}}],
                "measurement_receipts": [{"receipt_id": "r1", "changed_cells": 1}],
                "replayable": True,
            }
        ],
        "runtime_observations": [{"action": 6, "changed_cells": 1, "level_after": 2}],
        "newly_reached_levels": [],
        "solution_labels": [],
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
        "runtime_self_discovery": True,
    }


def test_req_arc_fcp_5423_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5423: OpenSpec anchors the CoEx landmark artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5423" in spec
    assert "SCENARIO-ARC-FCP-5423" in spec
    assert exp5423.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5423.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5423_precheck_prefers_unbanked_lf52_l3() -> None:
    """SCENARIO-ARC-FCP-5423: registry precheck avoids duplicate solved levels."""

    selected = exp5423.select_target_after_precheck(_registry(lf52_levels=2))
    rotated = exp5423.select_target_after_precheck(_registry(lf52_levels=3))
    blocked = exp5423.select_target_after_precheck(
        _registry(lf52_levels=3, include_alternate=False),
        alternates=(),
    )

    assert selected["status"] == "selected"
    assert selected["registry_precheck"] is True
    assert selected["target_game"] == "lf52"
    assert selected["target_level"] == "L3"
    assert selected["target_level_number"] == 3
    assert selected["target_level_before"] == 2
    assert selected["duplicate_solve_avoided"] is True
    assert selected["selection_reason"] == "preferred_lf52_l3_not_banked"
    assert rotated["target_game"] == "re86"
    assert rotated["target_level"] == "L3"
    assert blocked["status"] == "blocked_duplicate_solve"
    assert blocked["duplicate_solve_avoided"] is True
    assert (
        exp5423.select_target_after_precheck(
            _registry(lf52_levels=3),
            alternates=("missing", "re86"),
        )["target_game"]
        == "re86"
    )
    assert exp5423._action_label(6, {"x": 1, "y": 2}) == (  # noqa: SLF001
        '{"action":6,"data":{"x":1,"y":2}}'
    )
    assert exp5423._fallback_sequence_receipt(  # noqa: SLF001
        ["RESET", "not-json", '{"action":1,"data":null}']
    )[0]["sequence"] == [{"action": 1, "data": None}]
    assert exp5423._fallback_sequence_receipt(["RESET"]) == []  # noqa: SLF001


def test_scenario_arc_fcp_5423_generator_persists_frontier_landmarks_and_receipts() -> None:
    """SCENARIO-ARC-FCP-5423: live observations promote persistent landmark prefixes."""

    generator = LiveCoExLandmarkFrontierGenerator(
        min_support=2,
        max_uncertainty=0.51,
        landmark_min_changed_cells=1,
    )
    before = _frame(level=2)
    after = _frame(level=2, changed=True)
    candidates = _candidates()

    assert generator.best_sequence(before, candidates) == tuple()
    generator.observe_transition(before, 6, {"x": 14, "y": 14}, after)
    assert generator.best_sequence(before, candidates) == tuple()
    generator.observe_transition(before, 6, {"x": 14, "y": 14}, after)

    sequence = generator.best_sequence(before, candidates)
    generator.record_reset(level=2)
    post_reset_sequence = generator.best_sequence(before, candidates)
    diagnostics = generator.diagnostics()

    assert sequence[0] == {"action": 6, "data": {"x": 14, "y": 14}}
    assert post_reset_sequence[0] == sequence[0]
    assert diagnostics["source"] == "live_coex_landmark_frontier"
    assert diagnostics["frontier_expansion_count"] == 2
    assert diagnostics["landmark_count"] >= 1
    assert diagnostics["reset_count"] == 1
    assert diagnostics["frontier_transitions"][0]["accepted"] is False
    assert diagnostics["frontier_transitions"][-1]["accepted"] is True
    assert diagnostics["action_history_clusters"][0]["support_count"] == 2
    receipt = diagnostics["action_sequence_receipts"][0]
    assert receipt["replayable"] is True
    assert receipt["measurement_receipts"][0]["changed_cells"] == 1
    assert generator.as_dict()["coex_frontier_persistence_enabled"] is True


def test_scenario_arc_fcp_5423_generator_defensive_branches() -> None:
    """REQ-ARC-FCP-5423: defensive live-observation branches are deterministic."""

    generator = LiveCoExLandmarkFrontierGenerator(min_support=1, max_uncertainty=0.51)
    bad_level = SimpleNamespace(frame=np.zeros((3, 3), dtype=np.int16), levels_completed="bad")
    no_level = np.ones((3, 3), dtype=np.int16)

    assert live_frontier._first_action_allowed([], _candidates()) is False  # noqa: SLF001
    generator.observe_transition(bad_level, 6, {"x": 1, "y": 1}, no_level)
    generator.reset(level=0, reset_to_prior=True)
    generator.record_reset(level=0)
    generator.observe_transition(bad_level, 6, {"x": 1, "y": 1}, no_level)
    before_landmarks = generator.diagnostics()["landmark_count"]

    generator._persistent_frontiers.clear()  # noqa: SLF001 - force accepted-row path
    one_candidate = [ArcAction(6, {"x": 1, "y": 1}, "solo")]
    sequence = generator.best_sequence(bad_level, one_candidate)
    generator._record_landmark(  # noqa: SLF001 - duplicate landmark guard
        after_hash=live_frontier._frame_hash(no_level),  # noqa: SLF001
        level_after=0,
        changed=1,
        receipt_id="m0002",
    )
    generator._record_sequence_receipt(  # noqa: SLF001 - fallback to latest measurement
        [{"action": 1, "data": None}],
        "manual",
        [],
    )
    diagnostics = generator.diagnostics()

    assert sequence == (
        {"action": 6, "data": {"x": 1, "y": 1}},
        {"action": 6, "data": {"x": 1, "y": 1}},
    )
    assert diagnostics["reset_count"] == 1
    assert diagnostics["landmark_count"] == before_landmarks
    assert diagnostics["action_sequence_receipts"][-1]["measurement_receipts"][0]["receipt_id"] == "m0002"


def test_scenario_arc_fcp_5423_live_e3_path_reaches_coex_hooks() -> None:
    """SCENARIO-ARC-FCP-5423: E3 can consume the CoEx generator on the live path."""

    generator = LiveCoExLandmarkFrontierGenerator(min_support=1, max_uncertainty=0.51)
    policy = E3AgentPolicy(
        "lf52",
        proposer=None,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=None,
        action_effect_expansion_prior=False,
        action_prior=generator,
        qd_generator=generator,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )
    candidates = policy.explorer._candidates(_frame(level=2))  # noqa: SLF001 - live hook fixture
    generator.observe_transition(_frame(level=2), 6, {"x": 14, "y": 14}, _frame(level=2, changed=True))
    node = {"frame": _frame(level=2), "untested": candidates}
    sequence = policy.explorer._qd_sequence_for_node(node)  # noqa: SLF001 - live hook fixture
    qd_diagnostics = policy.explorer.qd_generation_diagnostics()

    assert sequence[0] == {"action": 6, "data": {"x": 14, "y": 14}}
    assert qd_diagnostics["sequences_injected"] == 1
    assert qd_diagnostics["generator"]["coex_frontier_persistence_enabled"] is True
    assert policy.explorer.action_salience_diagnostics()["connected_component_salience_enabled"] is True


def test_scenario_arc_fcp_5423_artifact_schema_gates_live_credit() -> None:
    """REQ-ARC-FCP-5423: only reproduced live self-discovery can bank a level."""

    selection = exp5423.select_target_after_precheck(_registry())
    artifact = exp5423.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt=_null_attempt(),
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5423"],
        duration_s=0.2,
    )

    exp5423.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["registry_precheck"] is True
    assert artifact["target_game"] == "lf52"
    assert artifact["target_level"] == "L3"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["arc_new_level_banked"] is False
    assert artifact["attempt_count"] == 4
    assert artifact["frontier_expansion_count"] == 1
    assert artifact["landmark_count"] == 1
    assert artifact["action_sequence_receipts"]
    assert artifact["no_offline_bfs"] is True
    assert artifact["no_per_game_adapter"] is True
    assert artifact["arc_levelup_lint_passed"] is True
    assert artifact["inference_substrate"] == exp5423.INFERENCE_SUBSTRATE

    success = exp5423.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt={
            **_null_attempt(),
            "offline_reproduced": True,
            "max_level_reached": 3,
            "new_reproduced_levels": 1,
            "solution_labels": ['{"action":6,"data":{"x":14,"y":14}}'],
            "failure_mode": None,
        },
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5423"],
        duration_s=0.2,
    )
    exp5423.validate_artifact(success)
    assert success["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1

    bad = {**success, "action_sequence_receipts": []}
    errors = exp5423.artifact_schema_errors(bad)
    assert "action_sequence_receipts must be a non-empty list" in errors

    corrupt = {
        **success,
        "status": "complete",
        "solve_provenance": "outer_loop_re",
        "inference_substrate": "offline_proxy",
        "registry_precheck": "yes",
        "duplicate_solve_avoided": False,
        "attempt_count": "4",
        "target_game": "",
        "honest_verdict": "done",
        "offline_reproduced": False,
        "arc_new_level_banked": False,
        "reproduced_levels": 0,
    }
    errors = exp5423.artifact_schema_errors(corrupt)
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert f"inference_substrate must be {exp5423.INFERENCE_SUBSTRATE}" in errors
    assert "registry_precheck must be bare bool" in errors
    assert "duplicate_solve_avoided must be true" in errors
    assert "attempt_count must be bare int" in errors
    assert "target_game must be non-empty string" in errors
    assert "complete artifact requires offline_reproduced true" in errors
    assert "complete artifact requires arc_new_level_banked true" in errors
    assert "complete artifact requires reproduced_levels >= 1" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5423.validate_artifact(corrupt)

    impossible_null = {**success, "status": "honest_null", "arc_new_level_banked": True}
    assert "arc_new_level_banked requires complete status" in exp5423.artifact_schema_errors(
        impossible_null
    )

    invalid = {
        **success,
        "status": "weird",
        "action_sequence_receipts": "not-list",
        "offline_reproduced": True,
        "solve_provenance": "outer_loop_re",
    }
    invalid_errors = exp5423.artifact_schema_errors(invalid)
    assert "status must be complete, honest_null, or blocked" in invalid_errors
    assert "action_sequence_receipts must be list" in invalid_errors
    assert "offline_reproduced true requires live_agent_self_discovery" in invalid_errors

    blocked = exp5423.build_artifact(
        selection={
            **selection,
            "status": "blocked_duplicate_solve",
            "registry_precheck": True,
        },
        registry_total_before=69,
        attempt={"blocked": True, "no_offline_bfs": True, "no_per_game_adapter": True},
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5423"],
        duration_s=0.2,
    )
    exp5423.validate_artifact(blocked)
    assert blocked["failure_mode"] == "duplicate_solve_precheck"

    no_failure_null = exp5423.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt={
            **_null_attempt(),
            "failure_mode": None,
            "runtime_self_discovery": False,
        },
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5423"],
        duration_s=0.2,
    )
    assert no_failure_null["failure_mode"] == "bounded_budget_no_levelup"


def test_scenario_arc_fcp_5423_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5423: run wrapper writes a validated deliverable JSON."""

    (tmp_path / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")
    spec_path = tmp_path / exp5423.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-ARC-FCP-5423\n", encoding="utf-8")
    registry_path = tmp_path / exp5423.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(_registry()), encoding="utf-8")

    artifact = exp5423.run_experiment(
        root=tmp_path,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: True,
        lint_runner=lambda _root: _lint_result(),
        tests_run=["unit 5423"],
    )

    written = tmp_path / exp5423.RESULT_RELATIVE_PATH
    assert written.exists()
    assert artifact["registry_total_before"] == 69
    assert artifact["registry_total_after"] == 69
    assert artifact["preconditions_checked"]["offline_arcade_available"] is True
    assert artifact["arc_levelup_lint"]["passed"] is True
    assert artifact["status"] == "honest_null"

    blocked = exp5423.run_experiment(
        root=tmp_path,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: False,
        lint_runner=lambda _root: _lint_result(False),
        tests_run=["unit 5423"],
    )
    assert blocked["status"] == "blocked"
    assert blocked["failure_mode"] == "missing_harness_access"
    assert blocked["arc_levelup_lint_passed"] is False

    missing_codex_root = tmp_path / "missing_codex"
    missing_codex_root.mkdir()
    (missing_codex_root / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    spec_path = missing_codex_root / exp5423.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-ARC-FCP-5423\n", encoding="utf-8")
    registry_path = missing_codex_root / exp5423.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(_registry()), encoding="utf-8")

    precondition_blocked = exp5423.run_experiment(
        root=missing_codex_root,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: True,
        lint_runner=lambda _root: _lint_result(),
        tests_run=["unit 5423"],
    )
    assert precondition_blocked["status"] == "blocked"
    assert precondition_blocked["preconditions_checked"]["CODEX.md"] is False
