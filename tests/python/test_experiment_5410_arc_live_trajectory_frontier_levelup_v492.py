"""Tests for Exp5410 ARC live trajectory-frontier level-up attempt.

Spec refs: REQ-ARC-FCP-5410,
SCENARIO-ARC-FCP-5410.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5410_arc_live_trajectory_frontier_levelup_v492 as exp5410
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_live_trajectory_frontier import LiveTrajectoryFrontierGenerator


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(re86_levels: int = 2, *, include_alternate: bool = True) -> dict[str, Any]:
    games: list[dict[str, Any]] = [
        {"game": "re86", "reproducibility": "reproduced", "levels_reproduced": re86_levels},
    ]
    if include_alternate:
        games.append(
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2}
        )
    return {"reproducible_total_levels": 69, "games": games}


def _frame(*, changed: bool = False) -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[2:10, 2:18] = 8
    grid[14:16, 14:16] = 9
    if changed:
        grid[14, 14] = 10
    return SimpleNamespace(frame=grid, available_actions=[1, 6])


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
        "spec_has_req_5410": True,
        "registry_present": True,
        "offline_arcade_available": True,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
    }


def _null_attempt() -> dict[str, Any]:
    return {
        "attempt_count": 4,
        "max_level_reached": 2,
        "offline_reproduced": False,
        "failure_mode": "bounded_budget_no_levelup",
        "frontier_expansions": [
            {"prefix": [{"action": 6, "data": {"x": 14, "y": 14}}], "accepted": True}
        ],
        "frontier_expansion_count": 1,
        "salience_routes_used": ["blob_tier_0_button_like"],
        "uncertainty_rejections": 1,
        "verifier_observations": [{"support_count": 2, "accepted": True}],
        "newly_reached_levels": [],
        "solution_labels": [],
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
        "runtime_self_discovery": True,
    }


def test_req_arc_fcp_5410_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5410: OpenSpec anchors the trajectory-frontier artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5410" in spec
    assert "SCENARIO-ARC-FCP-5410" in spec
    assert exp5410.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5410.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5410_generator_rejects_low_support_then_emits_prefix() -> None:
    """SCENARIO-ARC-FCP-5410: observed prefixes need support before promotion."""

    generator = LiveTrajectoryFrontierGenerator(min_support=2, max_uncertainty=0.4)
    before = _frame()
    after = _frame(changed=True)
    candidates = _candidates()

    assert generator.best_sequence(before, candidates) == tuple()
    assert generator.diagnostics()["uncertainty_rejections"] == 1

    generator.observe_transition(before, 6, {"x": 14, "y": 14}, after)
    assert generator.best_sequence(before, candidates) == tuple()
    assert generator.diagnostics()["uncertainty_rejections"] == 2

    generator.observe_transition(before, 6, {"x": 14, "y": 14}, after)
    sequence = generator.best_sequence(before, candidates)
    diagnostics = generator.diagnostics()

    assert len(sequence) == 2
    assert sequence[0] == {"action": 6, "data": {"x": 14, "y": 14}}
    assert diagnostics["frontier_expansion_count"] == 1
    assert diagnostics["salience_routes_used"] == ["blob_tier_0_button_like"]
    assert diagnostics["verifier_observations"][0]["support_count"] == 2
    assert diagnostics["verifier_observations"][0]["accepted"] is True


def test_scenario_arc_fcp_5410_generator_edge_routes_and_fallbacks() -> None:
    """REQ-ARC-FCP-5410: salience routes and fallback grid coercions are explicit."""

    generator = LiveTrajectoryFrontierGenerator(min_support=1, max_uncertainty=0.51)
    frame_3d = SimpleNamespace(frame=np.zeros((2, 20, 20), dtype=np.int16), available_actions=[6])
    frame_bad = SimpleNamespace(frame=np.zeros((2,), dtype=np.int16), available_actions=[6])
    color_frame = SimpleNamespace(frame=np.zeros((8, 8), dtype=np.int16), available_actions=[6])
    color_frame.frame[3, 3] = 9
    color_after = SimpleNamespace(frame=np.array(color_frame.frame, copy=True), available_actions=[6])
    color_after.frame[3, 3] = 10

    generator.observe_transition(frame_3d, 6, {"x": 0, "y": 0}, _frame(changed=True))
    generator.observe_transition(frame_bad, 6, None, frame_bad)
    generator.observe_transition(color_frame, 6, {"x": 3, "y": 3}, color_after)

    one_action_sequence = generator.best_sequence(color_frame, [ArcAction(6, {"x": 3, "y": 3}, "solo")])
    diagnostics = generator.diagnostics()

    assert len(one_action_sequence) == 2
    assert one_action_sequence[0] == one_action_sequence[1]
    assert "blob_tier_2_color" in {
        row["salience_route"] for row in diagnostics["verifier_observations"]
    }
    assert "blob_tier_unknown" in {
        row["salience_route"] for row in diagnostics["verifier_observations"]
    }
    assert generator.as_dict()["trajectory_frontier_generation_enabled"] is True


def test_scenario_arc_fcp_5410_live_e3_path_reaches_generator_hooks() -> None:
    """SCENARIO-ARC-FCP-5410: E3 uses salience and sequence-prefix hooks."""

    generator = LiveTrajectoryFrontierGenerator(min_support=1, max_uncertainty=0.51)
    policy = E3AgentPolicy(
        "re86",
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
    candidates = policy.explorer._candidates(_frame())  # noqa: SLF001 - live hook fixture
    policy.explorer.qd_generator.observe_transition(_frame(), 6, {"x": 14, "y": 14}, _frame(changed=True))
    node = {"frame": _frame(), "untested": candidates}
    sequence = policy.explorer._qd_sequence_for_node(node)  # noqa: SLF001 - live hook fixture
    salience = policy.explorer.action_salience_diagnostics()

    assert candidates[0]["data"] == {"x": 14, "y": 14}
    assert sequence[0] == {"action": 6, "data": {"x": 14, "y": 14}}
    assert salience["connected_component_salience_enabled"] is True
    assert salience["salience_tiers_emitted"] is True
    assert policy.explorer.qd_generation_diagnostics()["sequences_injected"] == 1


def test_scenario_arc_fcp_5410_registry_precheck_selects_or_blocks_duplicate() -> None:
    """REQ-ARC-FCP-5410: registry precheck avoids duplicate solved levels."""

    selected = exp5410.select_target_after_precheck(_registry(re86_levels=2))
    rotated = exp5410.select_target_after_precheck(_registry(re86_levels=3))
    blocked = exp5410.select_target_after_precheck(
        _registry(re86_levels=3, include_alternate=False),
        alternates=(),
    )

    assert selected["status"] == "selected"
    assert selected["registry_precheck_done"] is True
    assert selected["target_game"] == "re86"
    assert selected["target_level"] == "L3"
    assert selected["duplicate_solve_avoided"] is True
    assert rotated["target_game"] == "sb26"
    assert rotated["target_level"] == "L3"
    assert (
        exp5410.select_target_after_precheck(
            _registry(re86_levels=3),
            alternates=("missing", "sb26"),
        )["target_game"]
        == "sb26"
    )
    assert blocked["status"] == "blocked_duplicate_solve"
    assert blocked["duplicate_solve_avoided"] is True
    assert exp5410._action_label(6, {"x": 1, "y": 2}) == (  # noqa: SLF001
        '{"action":6,"data":{"x":1,"y":2}}'
    )


def test_scenario_arc_fcp_5410_artifact_schema_gates_live_credit() -> None:
    """REQ-ARC-FCP-5410: only live self-discovery can set offline_reproduced true."""

    selection = exp5410.select_target_after_precheck(_registry())
    artifact = exp5410.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt=_null_attempt(),
        preconditions_checked=_preconditions(),
        tests_run=["unit 5410"],
        duration_s=0.2,
    )

    exp5410.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["offline_reproduced"] is False
    assert artifact["attempt_count"] == 4
    assert artifact["frontier_expansion_count"] == 1
    assert artifact["uncertainty_rejections"] == 1
    assert artifact["arc_new_level_banked"] is False
    assert artifact["no_offline_bfs"] is True
    assert artifact["no_per_game_adapter"] is True
    assert artifact["inference_substrate"] == exp5410.INFERENCE_SUBSTRATE

    success = exp5410.build_artifact(
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
        tests_run=["unit 5410"],
        duration_s=0.2,
    )
    exp5410.validate_artifact(success)
    assert success["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["arc_new_level_banked"] is True

    bad = dict(success)
    bad["solve_provenance"] = "development_proxy"
    bad["offline_reproduced"] = True
    bad["arc_new_level_banked"] = True
    errors = exp5410.artifact_schema_errors(bad)
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "offline_reproduced true requires live_agent_self_discovery" in errors
    with pytest.raises(ValueError):
        exp5410.validate_artifact(bad)

    default_null = exp5410.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt={"offline_reproduced": False},
        preconditions_checked=_preconditions(),
        tests_run=["unit 5410"],
        duration_s=0.1,
    )
    default_blocked = exp5410.build_artifact(
        selection=exp5410.select_target_after_precheck(
            _registry(re86_levels=3, include_alternate=False),
            alternates=(),
        ),
        registry_total_before=69,
        attempt={},
        preconditions_checked=_preconditions(),
        tests_run=["unit 5410"],
        duration_s=0.1,
    )
    assert default_null["failure_mode"] == "bounded_budget_no_levelup"
    assert default_blocked["failure_mode"] == "duplicate_solve_precheck"

    invalid = dict(default_null)
    invalid.update(
        {
            "status": "maybe",
            "inference_substrate": "offline_bfs",
            "registry_precheck_done": "yes",
            "duplicate_solve_avoided": False,
            "no_offline_bfs": False,
            "attempt_count": "4",
            "target_game": "",
            "target_level": "",
            "salience_routes_used": "blob",
            "honest_verdict": "unclear",
        }
    )
    invalid_errors = exp5410.artifact_schema_errors(invalid)
    assert "status must be complete, honest_null, or blocked" in invalid_errors
    assert f"inference_substrate must be {exp5410.INFERENCE_SUBSTRATE}" in invalid_errors
    assert "registry_precheck_done must be bare bool" in invalid_errors
    assert "duplicate_solve_avoided must be true" in invalid_errors
    assert "no_offline_bfs must be true" in invalid_errors
    assert "attempt_count must be bare int" in invalid_errors
    assert "target_game must be non-empty string" in invalid_errors
    assert "target_level must be non-empty string" in invalid_errors
    assert "salience_routes_used must be list" in invalid_errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in invalid_errors

    invalid_complete = dict(default_null)
    invalid_complete.update(
        {
            "status": "complete",
            "offline_reproduced": False,
            "arc_new_level_banked": False,
            "reproduced_levels": 0,
            "honest_verdict": "complete: invalid",
        }
    )
    complete_errors = exp5410.artifact_schema_errors(invalid_complete)
    assert "complete artifact requires offline_reproduced true" in complete_errors
    assert "complete artifact requires arc_new_level_banked true" in complete_errors
    assert "complete artifact requires reproduced_levels >= 1" in complete_errors

    invalid_noncomplete = dict(default_null)
    invalid_noncomplete.update({"offline_reproduced": True, "arc_new_level_banked": True})
    noncomplete_errors = exp5410.artifact_schema_errors(invalid_noncomplete)
    assert "non-complete artifact cannot set offline_reproduced true" in noncomplete_errors
    assert "arc_new_level_banked requires complete status" in noncomplete_errors


def test_scenario_arc_fcp_5410_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5410: run writes the bounded live-attempt artifact."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5410.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5410\nSCENARIO-ARC-FCP-5410\n",
        encoding="utf-8",
    )
    (root / exp5410.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["selection"]["target_game"] == "re86"
        return _null_attempt()

    artifact = exp5410.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5410 run"],
    )
    written = json.loads((root / exp5410.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["registry_precheck_done"] is True
    assert artifact["target_level"] == "L3"
    assert artifact["attempt_count"] == 4
    assert artifact["tests_run"] == ["unit 5410 run"]

    arcade_blocked = exp5410.run_experiment(
        root=root,
        offline_arcade_check=lambda: False,
        tests_run=["arcade blocked"],
    )
    assert arcade_blocked["honest_verdict"].startswith("blocked:")
    assert arcade_blocked["preconditions_checked"]["offline_arcade_available"] is False

    blocked_root = tmp_path / "blocked"
    blocked = exp5410.run_experiment(
        root=blocked_root,
        offline_arcade_check=lambda: True,
        tests_run=["missing preconditions"],
    )
    assert blocked["honest_verdict"].startswith("blocked:")
    assert blocked["attempt_count"] == 0
