"""Tests for Exp5450 ARC measurement-access live level-up attempt.

Spec refs: REQ-ARC-FCP-5450,
SCENARIO-ARC-FCP-5450.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5450_arc_measurement_access_live_levelup_v495 as exp5450


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, include_ka59: bool = True) -> dict[str, Any]:
    games: list[dict[str, Any]] = [
        {
            "game": "cn04",
            "reproducibility": "reproduced",
            "levels_reproduced": 3,
            "dead_ends": ["Exp5012 cn04 no-bank no_grounded_l4_delta"],
        },
        {
            "game": "re86",
            "reproducibility": "reproduced",
            "levels_reproduced": 2,
            "dead_ends": ["sprite overlay L3 salience attempts bounded_budget_no_levelup"],
        },
        {"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 2},
    ]
    if include_ka59:
        games.append({"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1})
    return {"reproducible_total_levels": 69, "games": games}


def _loop_results() -> dict[str, dict[str, Any]]:
    return {
        "cn04": {"offline_reproduced": True, "reproduced_levels": 3},
        "re86": {"offline_reproduced": True, "reproduced_levels": 2},
        "ka59": {"offline_reproduced": True, "reproduced_levels": 1},
        "sk48": {"offline_reproduced": True, "reproduced_levels": 2},
    }


def _attempt() -> dict[str, Any]:
    return {
        "live_attempt_count": 7,
        "max_level_reached": 1,
        "frontier_expansion_count": 2,
        "frontier_transitions": [
            {
                "from_hash": "root",
                "to_hash": "next",
                "action": 6,
                "data": {"x": 12, "y": 18},
                "changed_cells": 5,
                "accepted": True,
            }
        ],
        "runtime_observations": [
            {
                "level_before": 1,
                "level_after": 1,
                "before_hash": "root",
                "after_hash": "next",
                "changed_cells": 5,
            }
        ],
        "action_sequence_receipts": [
            {
                "sequence": [{"action": 6, "data": {"x": 12, "y": 18}}],
                "measurement_receipts": [{"receipt_id": "m0001", "changed_cells": 5}],
                "replayable": True,
            }
        ],
        "generic_verifier_routes": [{"route": "runtime_observation_cluster", "accepted": True}],
        "induction_attempts": [
            {
                "reason": "level_up_reinduction",
                "transition_count": 1,
                "planned": False,
                "skipped": "disabled_exp5450_no_llm",
            }
        ],
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "residual_wall": "bounded_budget_no_levelup",
        "runtime_self_discovery": True,
        "no_offline_bfs": True,
        "no_source_reading": True,
        "no_per_game_adapter_credited": True,
    }


def _preconditions() -> dict[str, Any]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "spec_has_req_5450": True,
        "registry_present": True,
        "arc_loop_results_checked": True,
        "offline_arcade_available": True,
        "no_offline_bfs": True,
        "no_source_reading": True,
        "no_per_game_adapter_credited": True,
    }


def test_req_arc_fcp_5450_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5450: OpenSpec anchors the measurement-access artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5450" in spec
    assert "SCENARIO-ARC-FCP-5450" in spec
    assert exp5450.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5450.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5450_rotation_avoids_recent_no_bank_targets() -> None:
    """SCENARIO-ARC-FCP-5450: target rotation avoids stale cn04/re86 reruns."""

    selection = exp5450.select_rotated_target(
        _registry(),
        _loop_results(),
        recent_no_bank_targets=("cn04:L4", "re86:L3"),
        frontier_priority=("cn04", "re86", "ka59", "sk48"),
    )
    blocked = exp5450.select_rotated_target(
        _registry(include_ka59=False),
        _loop_results(),
        recent_no_bank_targets=("cn04:L4", "re86:L3", "sk48:L3"),
        frontier_priority=("cn04", "re86", "sk48"),
    )
    skipped_missing_then_selected = exp5450.select_rotated_target(
        _registry(),
        _loop_results(),
        recent_no_bank_targets=("cn04:L4", "re86:L3"),
        frontier_priority=("missing", "cn04", "re86", "ka59"),
    )

    assert selection["status"] == "selected"
    assert selection["registry_precheck_total_levels"] == 69
    assert selection["selected_game"] == "ka59"
    assert selection["selected_target_level"] == 2
    assert selection["registry_level_before"] == 1
    assert "rotated_away_from_recent_no_bank" in selection["target_rotation_reason"]
    assert "cn04:L4" in selection["skipped_recent_no_bank_targets"]
    assert selection["frontier_precheck"]["re86"]["loop_reproduced_levels"] == 2
    assert blocked["status"] == "blocked"
    assert blocked["selected_game"] == ""
    assert blocked["selected_target_level"] == 0
    assert "no_eligible_frontier_after_rotation" in blocked["target_rotation_reason"]
    assert skipped_missing_then_selected["selected_game"] == "ka59"
    assert exp5450._loop_depth({"x": {"offline_reproduced": False}}, "x") == 0  # noqa: SLF001
    assert exp5450._dead_end_summary({"dead_ends": "string dead end"}) == (  # noqa: SLF001
        "string dead end"
    )
    assert exp5450._dead_end_summary({"dead_ends": object()}) == ""  # noqa: SLF001
    assert exp5450._action_label(6, {"x": 1, "y": 2}) == (  # noqa: SLF001
        '{"action":6,"data":{"x":1,"y":2}}'
    )


def test_scenario_arc_fcp_5450_runtime_predicates_capture_measurement_access() -> None:
    """SCENARIO-ARC-FCP-5450: predicates come from runtime measurement access."""

    predicates = exp5450.induce_runtime_predicates(_attempt())
    empty = exp5450.induce_runtime_predicates(
        {
            "runtime_observations": [object()],
            "frontier_transitions": [object()],
            "generic_verifier_routes": [object()],
            "induction_attempts": [object()],
        }
    )

    assert [row["predicate"] for row in predicates] == [
        "frame_level_measurement",
        "action_effect_observation",
        "state_change_summary",
        "verifier_routed_predicate",
    ]
    assert predicates[0]["level_before"] == 1
    assert predicates[1]["changed_cells"] == 5
    assert predicates[2]["from_hash"] == "root"
    assert predicates[3]["source"] == "generic_verifier_route"
    assert empty == []


def test_req_arc_fcp_5450_artifact_schema_gates_reproduction_credit() -> None:
    """REQ-ARC-FCP-5450: only reproduced +1 live self-discovery banks a level."""

    selection = exp5450.select_rotated_target(_registry(), _loop_results())
    artifact = exp5450.build_artifact(
        selection=selection,
        attempt=_attempt(),
        preconditions_checked=_preconditions(),
        tests_run=["unit 5450"],
        duration_s=0.1,
    )

    exp5450.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["selected_game"] == "ka59"
    assert artifact["selected_target_level"] == 2
    assert artifact["live_attempt_count"] == 7
    assert artifact["runtime_predicates_induced"]
    assert artifact["offline_reproduced"] is False
    assert artifact["new_level_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["arc_new_level_banked"] is False
    assert artifact["inference_substrate"] == exp5450.INFERENCE_SUBSTRATE

    success = exp5450.build_artifact(
        selection=selection,
        attempt={
            **_attempt(),
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "max_level_reached": 2,
            "residual_wall": "",
        },
        preconditions_checked=_preconditions(),
        tests_run=["unit 5450"],
        duration_s=0.1,
    )
    exp5450.validate_artifact(success)
    assert success["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["new_level_reproduced"] is True
    assert success["new_levels_banked"] == 1
    assert success["arc_new_level_banked"] is True

    corrupt = {
        **success,
        "solve_provenance": "development_proxy",
        "inference_substrate": "offline_proxy",
        "registry_precheck_total_levels": "69",
        "selected_game": "",
        "selected_target_level": "2",
        "target_rotation_reason": "",
        "live_attempt_count": "7",
        "runtime_predicates_induced": [],
        "offline_reproduced": "False",
        "reproduced_levels": "1",
        "new_levels_banked": "1",
        "new_level_reproduced": "False",
        "no_offline_bfs": False,
        "no_source_reading": False,
        "no_per_game_adapter_credited": False,
        "arc_new_level_banked": False,
        "honest_verdict": "done",
    }
    errors = exp5450.artifact_schema_errors(corrupt)
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert f"inference_substrate must be {exp5450.INFERENCE_SUBSTRATE}" in errors
    assert "registry_precheck_total_levels must be bare int" in errors
    assert "selected_game must be non-empty string" in errors
    assert "selected_target_level must be bare int" in errors
    assert "target_rotation_reason must be non-empty string" in errors
    assert "live_attempt_count must be bare int" in errors
    assert "runtime_predicates_induced must be a non-empty list" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "new_levels_banked must be bare int" in errors
    assert "new_level_reproduced must be bare bool" in errors
    assert "no_offline_bfs must be true" in errors
    assert "no_source_reading must be true" in errors
    assert "no_per_game_adapter_credited must be true" in errors
    assert "complete artifact requires offline_reproduced true" in errors
    assert "complete artifact requires new_level_reproduced true" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5450.validate_artifact(corrupt)

    weird = {**artifact, "status": "weird", "selected_target_level": 0}
    weird_errors = exp5450.artifact_schema_errors(weird)
    assert "status must be complete, honest_null, or blocked" in weird_errors
    assert "selected_target_level must be >= 1" in weird_errors

    zero_level_complete = {**success, "reproduced_levels": 0, "new_levels_banked": 0}
    zero_level_errors = exp5450.artifact_schema_errors(zero_level_complete)
    assert "complete artifact requires reproduced_levels >= 1" in zero_level_errors
    assert "complete artifact requires new_levels_banked >= 1" in zero_level_errors

    invalid_null = {**artifact, "offline_reproduced": True, "arc_new_level_banked": True}
    invalid_null_errors = exp5450.artifact_schema_errors(invalid_null)
    assert "non-complete artifact cannot set offline_reproduced true" in invalid_null_errors
    assert "arc_new_level_banked requires complete status" in invalid_null_errors

    blocked = exp5450.build_artifact(
        selection={
            "status": "blocked",
            "registry_precheck_total_levels": 69,
            "selected_game": "",
            "selected_target_level": 0,
            "target_rotation_reason": "no_eligible_frontier_after_rotation",
            "registry_level_before": 0,
        },
        attempt={"blocked": True},
        preconditions_checked={**_preconditions(), "offline_arcade_available": False},
        tests_run=["unit 5450"],
        duration_s=0.1,
    )
    exp5450.validate_artifact(blocked)
    assert blocked["honest_verdict"].startswith("blocked:")


def test_scenario_arc_fcp_5450_run_experiment_writes_deliverable(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5450: run wrapper writes a validated deliverable JSON."""

    (tmp_path / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")
    spec_path = tmp_path / exp5450.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-ARC-FCP-5450\nSCENARIO-ARC-FCP-5450\n", encoding="utf-8")
    registry_path = tmp_path / exp5450.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(_registry()), encoding="utf-8")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for game, row in _loop_results().items():
        (results_dir / f"arc_loop_solve_{game}.json").write_text(
            json.dumps({"game": game, **row}),
            encoding="utf-8",
        )

    artifact = exp5450.run_experiment(
        root=tmp_path,
        attempt_runner=lambda **_kwargs: _attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5450"],
    )
    written = json.loads((tmp_path / exp5450.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["registry_precheck_total_levels"] == 69
    assert artifact["preconditions_checked"]["arc_loop_results_checked"] is True
    assert artifact["status"] == "honest_null"

    blocked = exp5450.run_experiment(
        root=tmp_path,
        attempt_runner=lambda **_kwargs: _attempt(),
        offline_arcade_check=lambda: False,
        tests_run=["unit 5450"],
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
