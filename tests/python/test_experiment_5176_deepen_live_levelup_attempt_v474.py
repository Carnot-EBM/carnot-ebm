"""Tests for Exp 5176 B1/B2-gated ARC deepen live level-up attempt.

Spec refs: REQ-REPORT-5176,
SCENARIO-REPORT-5176-BLOCKED-NO-VALIDATED-LEVER,
SCENARIO-REPORT-5176-VALIDATED-LEVER-SELECTION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from carnot import experiment_5176_deepen_live_levelup_attempt_v474 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
EXP5174_PATH = REPO / "results" / "experiment_5174_gap_live_integration_reconciliation_v474.json"
EXP5175_PATH = REPO / "results" / "experiment_5175_gap4891_relational_mask_pruner_ab_v474.json"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _exp5174(value: str = "re-scoped") -> dict:
    return {"gap_status_recommendation": {"value": value}}


def _exp5175(
    *,
    levels_banked: list[dict] | None = None,
    reduction: float = 0.0,
    pruned_edges: int = 0,
) -> dict:
    return {
        "levels_banked": levels_banked or [],
        "gap4891_status_recommendation": "building_with_new_lever_named",
        "states_expanded_reduction_pct": {"cd82": reduction, "cn04": 0.0},
        "move_pruned_edges": {"cd82": pruned_edges, "cn04": 3},
        "target_games": ["cd82", "sk48", "sp80"],
        "negative_control_game": "cn04",
        "per_game": [
            {
                "game": "cd82",
                "pruned": {"pruner_stats": {"pruned": pruned_edges}},
                "unpruned": {"states_expanded": 4000},
            },
            {
                "game": "cn04",
                "pruned": {"pruner_stats": {"pruned": 3}},
                "unpruned": {"states_expanded": 4000},
            },
        ],
    }


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ar25
  levels_reproduced: 3
- game: bp35
  levels_reproduced: 2
- game: cd82
  levels_reproduced: 2
- game: cn04
  levels_reproduced: 3
- game: dc22
  levels_reproduced: 2
- game: ft09
  levels_reproduced: 3
"""


def test_req_report_5176_spec_declares_required_artifact_contract() -> None:
    """REQ-REPORT-5176: OpenSpec declares the Exp5176 artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-5176" in spec
    assert "SCENARIO-REPORT-5176-BLOCKED-NO-VALIDATED-LEVER" in spec
    assert "SCENARIO-REPORT-5176-VALIDATED-LEVER-SELECTION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_report_5176_blocked_no_validated_lever_from_actual_upstreams() -> None:
    """SCENARIO-REPORT-5176-BLOCKED-NO-VALIDATED-LEVER: actual B1/B2 nulls stop."""

    exp5174 = json.loads(EXP5174_PATH.read_text(encoding="utf-8"))
    exp5175 = json.loads(EXP5175_PATH.read_text(encoding="utf-8"))
    registry_levels = mod.load_registry_levels(REGISTRY_PATH.read_text(encoding="utf-8"))

    artifact = mod.build_artifact(
        exp5174=exp5174,
        exp5175=exp5175,
        registry_levels=registry_levels,
        live_path_reachable=True,
        arc_orphan_solver_lint={"passed": True},
        duration_s=1.0,
    )

    assert artifact["lever_used"] == "none_available"
    assert artifact["target_games"] == [
        {"game": "cd82", "level_before": 2, "level_attempted": 3},
        {"game": "cn04", "level_before": 3, "level_attempted": 4},
    ]
    assert artifact["levels_banked"] == []
    assert artifact["reproducible_levels_delta"] == 0
    assert artifact["live_path_reachable"] is True
    assert artifact["solve_provenance"] == "not_applicable_blocked_no_runtime_discovery"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["stopped_before_levelup_attempt"] is True
    assert "MAP (arXiv:2605.13037)" in artifact["next_direction_if_null"]
    assert artifact["honest_verdict"].startswith("complete_")
    assert "zero_levels_banked" in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)


def test_scenario_report_5176_validated_lever_selection_cases() -> None:
    """SCENARIO-REPORT-5176-VALIDATED-LEVER-SELECTION: banks or state wins unlock."""

    assert mod.determine_lever(_exp5174(), _exp5175()) == "none_available"
    assert mod.determine_lever(_exp5174("fixable_live_path_wiring_gap"), _exp5175()) == "exp5174"
    assert (
        mod.determine_lever(
            _exp5174(),
            _exp5175(
                levels_banked=[
                    {
                        "game": "cd82",
                        "new_level": 3,
                        "offline_reproduced": True,
                        "reproducibility_checksum": "sha256:banked",
                    }
                ]
            ),
        )
        == "exp5175"
    )
    assert mod.determine_lever(_exp5174(), _exp5175(reduction=12.5)) == "exp5175"
    assert (
        mod.determine_lever(
            _exp5174("fixable_live_path_wiring_gap"),
            _exp5175(reduction=12.5),
        )
        == "both"
    )
    assert mod.select_target_games({"cd82": 0, "cn04": 3}, _exp5175(), limit=2) == [
        {"game": "cn04", "level_before": 3, "level_attempted": 4}
    ]


def test_req_report_5176_artifact_records_banked_counterfactual_without_live_claim() -> None:
    """REQ-REPORT-5176: reproduce-confirmed upstream banks are counted explicitly."""

    registry_levels = mod.load_registry_levels(_registry_text())
    banked = {
        "game": "cd82",
        "new_level": 3,
        "offline_reproduced": True,
        "reproducibility_checksum": "sha256:banked",
    }
    artifact = mod.build_artifact(
        exp5174=_exp5174(),
        exp5175=_exp5175(levels_banked=[copy.deepcopy(banked)], pruned_edges=9),
        registry_levels=registry_levels,
        live_path_reachable=True,
        arc_orphan_solver_lint={"passed": True},
        duration_s=2.0,
    )

    assert artifact["lever_used"] == "exp5175"
    assert artifact["levels_banked"] == [banked]
    assert artifact["reproducible_levels_delta"] == 1
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["next_direction_if_null"] == ""
    assert artifact["preconditions_checked"]["stopped_before_levelup_attempt"] is False
    assert artifact["honest_verdict"].startswith("success_")
    assert "one_level_banked" in artifact["honest_verdict"]

    two_banks = [copy.deepcopy(banked), {**copy.deepcopy(banked), "game": "cn04", "new_level": 4}]
    plural = mod.build_artifact(
        exp5174=_exp5174(),
        exp5175=_exp5175(levels_banked=two_banks, pruned_edges=9),
        registry_levels=registry_levels,
        live_path_reachable=True,
        arc_orphan_solver_lint={"passed": True},
        duration_s=2.0,
    )

    assert plural["reproducible_levels_delta"] == 2
    assert plural["honest_verdict"].startswith("success_2_level_banked")


def test_req_report_5176_write_artifact_is_stable(tmp_path: Path) -> None:
    """REQ-REPORT-5176: the terminal JSON artifact is reproducibly written."""

    artifact = mod.build_artifact(
        exp5174=_exp5174(),
        exp5175=_exp5175(),
        registry_levels=mod.load_registry_levels(_registry_text()),
        live_path_reachable=False,
        arc_orphan_solver_lint={"passed": False},
        duration_s=0.5,
    )

    written = mod.write_artifact(tmp_path, artifact)
    loaded = json.loads(written.read_text(encoding="utf-8"))

    assert written == tmp_path / mod.RESULT_RELATIVE_PATH
    assert loaded == artifact
