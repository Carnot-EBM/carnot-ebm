"""Tests for Exp5437 ARC registry-guided live reinduction level-up attempt.

Spec refs: REQ-ARC-FCP-5437,
SCENARIO-ARC-FCP-5437.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5437_arc_live_reinduction_levelup_v494 as exp5437


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(
    *,
    cn04_levels: int = 3,
    vc33_levels: int = 2,
    include_alternate: bool = True,
) -> dict[str, Any]:
    games: list[dict[str, Any]] = [
        {"game": "cn04", "reproducibility": "reproduced", "levels_reproduced": cn04_levels},
        {"game": "vc33", "reproducibility": "reproduced", "levels_reproduced": vc33_levels},
    ]
    if include_alternate:
        games.append({"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 2})
    return {"reproducible_total_levels": 69, "games": games}


def _preconditions() -> dict[str, Any]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "spec_has_req_5437": True,
        "registry_present": True,
        "offline_arcade_available": True,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
    }


def _lint_result(passed: bool = True) -> dict[str, Any]:
    return {"command": "arc_levelup_guarantee_lint", "passed": passed, "returncode": 0}


def _runtime_predicates() -> list[dict[str, Any]]:
    return [
        {
            "predicate": "observed_action_effect",
            "action": 6,
            "data": {"x": 14, "y": 14},
            "support_count": 2,
            "accepted": True,
            "source": "runtime_observation_cluster",
        }
    ]


def _null_attempt() -> dict[str, Any]:
    return {
        "attempt_count": 6,
        "reset_count": 1,
        "max_level_reached": 3,
        "offline_reproduced": False,
        "new_reproduced_levels": 0,
        "failure_mode": "bounded_budget_no_levelup",
        "frontier_expansion_count": 1,
        "runtime_predicate_count": 1,
        "runtime_predicates": _runtime_predicates(),
        "frontier_transitions": [
            {"from_hash": "root", "to_hash": "next", "action": 6, "accepted": True}
        ],
        "action_sequence_receipts": [
            {
                "sequence": [{"action": 6, "data": {"x": 14, "y": 14}}],
                "measurement_receipts": [{"receipt_id": "m0001", "changed_cells": 4}],
                "replayable": True,
            }
        ],
        "runtime_observations": [{"action": 6, "changed_cells": 4, "level_after": 3}],
        "generic_verifier_routes": [{"route": "runtime_observation_cluster", "accepted": True}],
        "newly_reached_levels": [],
        "solution_labels": [],
        "reproduction_gate": {"reproduced": False, "reached_level": 3},
        "runtime_self_discovery": True,
        "no_offline_bfs": True,
        "no_per_game_adapter": True,
    }


def test_req_arc_fcp_5437_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5437: OpenSpec anchors the reinduction artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5437" in spec
    assert "SCENARIO-ARC-FCP-5437" in spec
    assert exp5437.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5437.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5437_precheck_prefers_cn04_l4_then_vc33_l3() -> None:
    """SCENARIO-ARC-FCP-5437: registry precheck avoids duplicate solved levels."""

    selected = exp5437.select_target_after_precheck(_registry())
    cn04_not_next = exp5437.select_target_after_precheck(_registry(cn04_levels=2))
    rotated = exp5437.select_target_after_precheck(_registry(cn04_levels=4))
    alternate = exp5437.select_target_after_precheck(_registry(cn04_levels=4, vc33_levels=3))
    skipped_then_alternate = exp5437.select_target_after_precheck(
        {
            "reproducible_total_levels": 69,
            "games": [
                {"game": "cn04", "reproducibility": "reproduced", "levels_reproduced": 4},
                {"game": "vc33", "reproducibility": "reproduced", "levels_reproduced": 3},
                {"game": "bad", "reproducibility": "dry", "levels_reproduced": 3},
                {"game": "zero", "reproducibility": "reproduced", "levels_reproduced": 0},
                {"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 2},
            ],
        },
        alternates=("missing", "bad", "zero", "sk48"),
    )
    blocked = exp5437.select_target_after_precheck(
        _registry(cn04_levels=4, vc33_levels=3, include_alternate=False),
        alternates=(),
    )

    assert selected["status"] == "selected"
    assert selected["registry_precheck"] is True
    assert selected["target_game"] == "cn04"
    assert selected["target_level"] == "L4"
    assert selected["target_level_before"] == 3
    assert selected["duplicate_solve_avoided"] is True
    assert selected["target_eligible_reason"] == "cn04 L4 is next unbanked frontier"
    assert cn04_not_next["target_game"] == "vc33"
    assert "cn04 L4 not next frontier" in cn04_not_next["selection_reason"]
    assert rotated["target_game"] == "vc33"
    assert rotated["target_level"] == "L3"
    assert "preferred cn04 L4 already banked" in rotated["selection_reason"]
    assert alternate["target_game"] == "sk48"
    assert alternate["target_level"] == "L3"
    assert skipped_then_alternate["target_game"] == "sk48"
    assert blocked["status"] == "blocked_duplicate_solve"
    assert blocked["duplicate_solve_avoided"] is True
    assert exp5437._action_label(6, {"x": 1, "y": 2}) == (  # noqa: SLF001
        '{"action":6,"data":{"x":1,"y":2}}'
    )
    assert exp5437._fallback_sequence_receipt(  # noqa: SLF001
        ["RESET", "not-json", '{"action":1,"data":null}']
    )[0]["sequence"] == [{"action": 1, "data": None}]
    assert exp5437._fallback_sequence_receipt(["RESET"]) == []  # noqa: SLF001


def test_scenario_arc_fcp_5437_runtime_predicates_summarize_live_evidence() -> None:
    """SCENARIO-ARC-FCP-5437: runtime predicates come from observed live transitions."""

    predicates = exp5437.runtime_predicates_from_diagnostics(
        {
            "verifier_observations": [
                object(),
                {
                    "action": 6,
                    "data": {"x": 4, "y": 5},
                    "support_count": 2,
                    "effect_count": 2,
                    "accepted": True,
                    "salience_route": "blob_tier_0_button_like",
                },
                {"action": 1, "data": None, "support_count": 0, "accepted": False},
            ]
        },
        [
            object(),
            {"reason": "", "transition_count": 0},
            {
                "reason": "level_up_reinduction",
                "transition_count": 3,
                "planned": False,
                "skipped": "proposer_failed",
            },
        ],
    )
    routes = exp5437.generic_verifier_routes_from_predicates(predicates)

    assert [row["predicate"] for row in predicates] == [
        "observed_action_effect",
        "level_up_reinduction_route",
    ]
    assert predicates[0]["source"] == "runtime_observation_cluster"
    assert predicates[1]["source"] == "generic_verifier_routing"
    assert routes == [
        {"route": "runtime_observation_cluster", "accepted": True},
        {"route": "generic_verifier_routing", "accepted": False},
    ]


def test_scenario_arc_fcp_5437_artifact_schema_gates_live_credit() -> None:
    """REQ-ARC-FCP-5437: only reproduced live self-discovery can bank a level."""

    selection = exp5437.select_target_after_precheck(_registry())
    artifact = exp5437.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt=_null_attempt(),
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5437"],
        duration_s=0.2,
    )

    exp5437.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["registry_precheck"] is True
    assert artifact["target_game"] == "cn04"
    assert artifact["target_level"] == "L4"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["arc_new_level_banked"] is False
    assert artifact["attempt_count"] == 6
    assert artifact["frontier_expansion_count"] == 1
    assert artifact["runtime_predicate_count"] == 1
    assert artifact["action_sequence_receipts"]
    assert artifact["arc_levelup_lint_passed"] is True
    assert artifact["inference_substrate"] == exp5437.INFERENCE_SUBSTRATE

    success = exp5437.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt={
            **_null_attempt(),
            "offline_reproduced": True,
            "max_level_reached": 4,
            "new_reproduced_levels": 1,
            "solution_labels": ['{"action":6,"data":{"x":14,"y":14}}'],
            "failure_mode": None,
        },
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5437"],
        duration_s=0.2,
    )
    exp5437.validate_artifact(success)
    assert success["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["arc_new_level_banked"] is True

    corrupt = {
        **success,
        "status": "complete",
        "solve_provenance": "outer_loop_re",
        "inference_substrate": "offline_proxy",
        "registry_precheck": "yes",
        "duplicate_solve_avoided": False,
        "attempt_count": "6",
        "frontier_expansion_count": "1",
        "runtime_predicate_count": "1",
        "target_game": "",
        "target_level": "",
        "action_sequence_receipts": [],
        "honest_verdict": "done",
        "offline_reproduced": False,
        "arc_new_level_banked": False,
        "reproduced_levels": 0,
    }
    errors = exp5437.artifact_schema_errors(corrupt)
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert f"inference_substrate must be {exp5437.INFERENCE_SUBSTRATE}" in errors
    assert "registry_precheck must be bare bool" in errors
    assert "duplicate_solve_avoided must be true" in errors
    assert "attempt_count must be bare int" in errors
    assert "frontier_expansion_count must be bare int" in errors
    assert "runtime_predicate_count must be bare int" in errors
    assert "target_game must be non-empty string" in errors
    assert "target_level must be non-empty string" in errors
    assert "action_sequence_receipts must be a non-empty list" in errors
    assert "complete artifact requires offline_reproduced true" in errors
    assert "complete artifact requires arc_new_level_banked true" in errors
    assert "complete artifact requires reproduced_levels >= 1" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5437.validate_artifact(corrupt)

    weird = {
        **success,
        "status": "weird",
        "action_sequence_receipts": "not-list",
        "offline_reproduced": True,
        "solve_provenance": "outer_loop_re",
    }
    weird_errors = exp5437.artifact_schema_errors(weird)
    assert "status must be complete, honest_null, or blocked" in weird_errors
    assert "action_sequence_receipts must be list" in weird_errors
    assert "offline_reproduced true requires live_agent_self_discovery" in weird_errors

    no_runtime_evidence = {
        **success,
        "runtime_predicate_count": 0,
        "frontier_expansion_count": 0,
        "frontier_transitions": [],
    }
    assert "complete artifact requires runtime predicate or frontier evidence" in (
        exp5437.artifact_schema_errors(no_runtime_evidence)
    )

    invalid_null = {**artifact, "offline_reproduced": True, "arc_new_level_banked": True}
    null_errors = exp5437.artifact_schema_errors(invalid_null)
    assert "non-complete artifact cannot set offline_reproduced true" in null_errors
    assert "arc_new_level_banked requires complete status" in null_errors

    blocked = exp5437.build_artifact(
        selection={
            **selection,
            "status": "blocked_duplicate_solve",
            "registry_precheck": True,
        },
        registry_total_before=69,
        attempt={"blocked": True, "no_offline_bfs": True, "no_per_game_adapter": True},
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(False),
        tests_run=["unit 5437"],
        duration_s=0.2,
    )
    exp5437.validate_artifact(blocked)
    assert blocked["failure_mode"] == "duplicate_solve_precheck"
    assert blocked["arc_levelup_lint_passed"] is False

    default_failure = exp5437.build_artifact(
        selection=selection,
        registry_total_before=69,
        attempt={"offline_reproduced": False, "action_sequence_receipts": [object()]},
        preconditions_checked=_preconditions(),
        lint_result=_lint_result(),
        tests_run=["unit 5437"],
        duration_s=0.1,
    )
    assert default_failure["failure_mode"] == "bounded_budget_no_levelup"


def test_scenario_arc_fcp_5437_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5437: run wrapper writes a validated deliverable JSON."""

    (tmp_path / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")
    spec_path = tmp_path / exp5437.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-ARC-FCP-5437\nSCENARIO-ARC-FCP-5437\n", encoding="utf-8")
    registry_path = tmp_path / exp5437.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(_registry()), encoding="utf-8")

    artifact = exp5437.run_experiment(
        root=tmp_path,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: True,
        lint_runner=lambda _root: _lint_result(),
        tests_run=["unit 5437"],
    )
    written = json.loads((tmp_path / exp5437.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["registry_total_before"] == 69
    assert artifact["registry_total_after"] == 69
    assert artifact["preconditions_checked"]["offline_arcade_available"] is True
    assert artifact["status"] == "honest_null"

    arcade_blocked = exp5437.run_experiment(
        root=tmp_path,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: False,
        lint_runner=lambda _root: _lint_result(),
        tests_run=["arcade blocked"],
    )
    assert arcade_blocked["status"] == "blocked"
    assert arcade_blocked["failure_mode"] == "missing_harness_access"

    missing_codex_root = tmp_path / "missing_codex"
    missing_codex_root.mkdir()
    (missing_codex_root / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    spec_path = missing_codex_root / exp5437.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-ARC-FCP-5437\n", encoding="utf-8")
    registry_path = missing_codex_root / exp5437.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(_registry()), encoding="utf-8")

    precondition_blocked = exp5437.run_experiment(
        root=missing_codex_root,
        attempt_runner=lambda **_kwargs: _null_attempt(),
        offline_arcade_check=lambda: True,
        lint_runner=lambda _root: _lint_result(),
        tests_run=["unit 5437"],
    )
    assert precondition_blocked["status"] == "blocked"
    assert precondition_blocked["preconditions_checked"]["CODEX.md"] is False
