"""Tests for Exp5385 ARC geometric salience live-path attempt.

Spec refs: REQ-ARC-FCP-5385,
SCENARIO-ARC-FCP-5385.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5385_arc_geometric_salience_live_path_v490 as exp5385
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_competition_agent import StepwiseExplorer
from carnot.agentic.arc_frame_change_predictor import rank_arc_actions
from carnot.agentic.arc_geometric_salience import GeometricSaliencePrior


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(re86_levels: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "re86", "levels_reproduced": re86_levels},
            {"game": "sb26", "levels_reproduced": 2},
        ],
    }


def _two_button_frame() -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[3:5, 3:5] = 8
    grid[14:16, 14:16] = 8
    return SimpleNamespace(frame=grid, available_actions=[6])


def _two_button_candidates() -> list[ArcAction]:
    return [
        ArcAction(6, {"x": 3, "y": 3}, "far_equal_button"),
        ArcAction(6, {"x": 14, "y": 14}, "near_changed_button"),
    ]


def _observed_near_transition() -> tuple[SimpleNamespace, SimpleNamespace]:
    before = _two_button_frame().frame.copy()
    after = before.copy()
    after[14:16, 14:16] = 9
    return SimpleNamespace(frame=before), SimpleNamespace(frame=after)


def test_req_arc_fcp_5385_spec_declares_required_fields() -> None:
    """REQ-ARC-FCP-5385: OpenSpec anchors the 5385 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5385" in spec
    assert "SCENARIO-ARC-FCP-5385" in spec
    assert exp5385.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5385.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5385_geodesic_anchor_breaks_equal_salience_tie() -> None:
    """SCENARIO-ARC-FCP-5385: observed transition geometry reorders equal blobs."""

    frame = _two_button_frame()
    candidates = _two_button_candidates()
    prior = GeometricSaliencePrior(geodesic_weight=500.0)
    before, after = _observed_near_transition()

    base_ranked = rank_arc_actions(frame, candidates, prior=prior.base_prior)
    assert base_ranked[0].source == "far_equal_button"

    prior.observe_transition(before, 6, {"x": 14, "y": 14}, after)
    geometric_ranked = rank_arc_actions(frame, candidates, prior=prior)

    assert geometric_ranked[0].source == "near_changed_button"
    assert prior.observed_transition_count == 1
    assert prior.as_dict()["hyperbolic_or_geodesic_ranking_enabled"] is True
    assert prior.diagnostics()["geodesic_anchor_count"] == 1


def test_scenario_arc_fcp_5385_stepwise_explorer_feeds_action_prior_observations() -> None:
    """SCENARIO-ARC-FCP-5385: live ingestion updates transition-aware action priors."""

    frame = _two_button_frame()
    before, after = _observed_near_transition()
    prior = GeometricSaliencePrior(geodesic_weight=500.0)
    explorer = StepwiseExplorer(
        frame_change_scorer=None,
        action_prior=prior,
        candidate_router=None,
    )
    origin = explorer._hash(before)  # noqa: SLF001 - live-path observation fixture
    explorer.graph[origin] = {
        "path": [],
        "untested": [],
        "value": 0.0,
        "frame": before,
    }
    explorer.awaiting = {
        "origin": origin,
        "action": 6,
        "data": {"x": 14, "y": 14},
        "grid": before.frame,
        "previous_frame": before,
        "level_before": 0,
    }

    explorer._ingest(after)  # noqa: SLF001 - verifies submitted live-path hook
    ranked = rank_arc_actions(frame, _two_button_candidates(), prior=prior)

    assert prior.observed_transition_count == 1
    assert ranked[0].source == "near_changed_button"


def test_scenario_arc_fcp_5385_registry_precheck_selects_re86_l3_or_blocks_duplicate() -> None:
    """SCENARIO-ARC-FCP-5385: registry precheck avoids duplicate level credit."""

    selected = exp5385.select_target_after_precheck(_registry(re86_levels=2))
    duplicate = exp5385.select_target_after_precheck(
        {"reproducible_total_levels": 69, "games": [{"game": "re86", "levels_reproduced": 3}]},
        alternates=(),
    )

    assert selected["status"] == "selected"
    assert selected["registry_precheck_done"] is True
    assert selected["target_game"] == "re86"
    assert selected["target_level_before"] == 2
    assert selected["attempted_level"] == 3
    assert selected["no_duplicate_solve"] is True
    assert duplicate["status"] == "duplicate_blocked"
    assert duplicate["no_duplicate_solve"] is False


def test_scenario_arc_fcp_5385_rank_measurement_and_live_diagnostics() -> None:
    """SCENARIO-ARC-FCP-5385: diagnostics prove the live policy can consume the prior."""

    measurement = exp5385.measure_geometric_rank_delta()
    diagnostics = exp5385.geometric_salience_live_diagnostics()

    assert measurement["before_rank"] == 1
    assert measurement["after_rank"] == 0
    assert measurement["geometric_rank_delta"] == 1
    assert diagnostics["geometric_salience_live_reachable"] is True
    assert diagnostics["hyperbolic_or_geodesic_ranking_enabled"] is True
    assert diagnostics["action_prior_source"] == "geometric_geodesic_blob_salience"


def test_scenario_arc_fcp_5385_artifact_honesty_gates() -> None:
    """SCENARIO-ARC-FCP-5385: only live reproduced progress receives credit."""

    selection = exp5385.select_target_after_precheck(_registry())
    diagnostics = exp5385.geometric_salience_live_diagnostics()
    rank_measurement = exp5385.measure_geometric_rank_delta()
    no_bank = exp5385.build_artifact(
        selection=selection,
        registry_total_before=69,
        live_diagnostics=diagnostics,
        rank_measurement=rank_measurement,
        attempt={
            "offline_reproduced": False,
            "max_level_reached": 2,
            "reproduced_levels": 2,
            "failure_mode": "bounded_budget_no_levelup",
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5385.validate_artifact(no_bank)
    assert no_bank["status"] == "honest_null"
    assert no_bank["target_game"] == "re86"
    assert no_bank["target_level_before"] == 2
    assert no_bank["attempted_level"] == 3
    assert no_bank["reproduced_levels"] == 2
    assert no_bank["new_level_banked"] is False
    assert no_bank["offline_reproduced"] is False
    assert no_bank["failure_mode"] == "bounded_budget_no_levelup"
    assert no_bank["honest_verdict"].startswith("no-bank:")

    success = exp5385.build_artifact(
        selection=selection,
        registry_total_before=69,
        live_diagnostics=diagnostics,
        rank_measurement=rank_measurement,
        attempt={
            "offline_reproduced": True,
            "max_level_reached": 3,
            "reproduced_levels": 3,
            "failure_mode": "",
            "solution_labels": ['{"action":6,"data":{"x":14,"y":14}}'],
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5385.validate_artifact(success)
    assert success["status"] == "complete"
    assert success["offline_reproduced"] is True
    assert success["new_level_banked"] is True
    assert success["reproduced_levels"] == 3
    assert success["honest_verdict"].startswith("banked:")

    bad = dict(success)
    bad["solve_provenance"] = "outer_loop_re"
    bad["no_per_game_adapter"] = False
    bad["reproduced_levels"] = 2
    errors = exp5385.artifact_schema_errors(bad)

    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "no_per_game_adapter must be bare true" in errors
    assert "credited solve must reproduce at least attempted_level" in errors
    with pytest.raises(ValueError):
        exp5385.validate_artifact(bad)


def test_scenario_arc_fcp_5385_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5385: runner writes stable JSON with required bare fields."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5385.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5385\nSCENARIO-ARC-FCP-5385\n",
        encoding="utf-8",
    )
    (root / exp5385.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["selection"]["target_game"] == "re86"
        return {
            "target_game": "re86",
            "target_level_before": 2,
            "attempted_level": 3,
            "offline_reproduced": False,
            "reproduced_levels": 2,
            "new_level_banked": False,
            "actions_taken": 8,
            "max_level_reached": 2,
            "failure_mode": "bounded_budget_no_levelup",
        }

    artifact = exp5385.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5385 geometric salience"],
    )
    written = json.loads((root / exp5385.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["status"] == "honest_null"
    assert artifact["registry_precheck_done"] is True
    assert artifact["geometric_salience_live_reachable"] is True
    assert artifact["hyperbolic_or_geodesic_ranking_enabled"] is True
    assert artifact["no_outer_loop_re"] is True
    assert artifact["no_per_game_adapter"] is True
    assert artifact["no_duplicate_solve"] is True
    assert artifact["live_attempt_count"] == 1
    assert artifact["tests_run"] == ["unit 5385 geometric salience"]
