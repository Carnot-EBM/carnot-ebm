"""Tests for Exp5711 relational goal-energy live-path qualification.

Spec refs: REQ-ARC-WMTE-5711,
SCENARIO-ARC-WMTE-5711-LIVE-HOOK-REACHABILITY,
SCENARIO-ARC-WMTE-5711-SAFE-FALLBACK-AND-LEAKAGE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5711_arc_relational_goal_energy_live_qualification as mod


REPO = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
pytestmark = pytest.mark.memory_watchdog_skip


def test_req_arc_wmte_5711_artifact_builder_emits_required_gate_fields() -> None:
    """REQ-ARC-WMTE-5711: build_artifact emits every required scalar and audit field."""

    artifact = mod.build_artifact(root=REPO)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["new_levels_claimed"] == 0
    assert artifact["live_path_reachable_score"] == 1.0
    assert artifact["relational_goal_energy_ready_score"] == 1.0
    assert artifact["game_source_read_count"] == 0
    assert artifact["game_adapter_count"] == 0
    assert artifact["outer_loop_bfs_used"] is False
    assert artifact["per_game_leakage_detected"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["reproducibility_checksum"].removeprefix("sha256:")) == 64


def test_req_arc_wmte_5711_positive_and_negative_controls_are_visible() -> None:
    """REQ-ARC-WMTE-5711: exact separation, fallback, and route confusion are auditable."""

    artifact = mod.build_artifact(root=REPO)

    for fixture, variance in artifact["score_variance_by_fixture"].items():
        if fixture.startswith("positive_"):
            assert variance > mod.VARIANCE_FLOOR
            assert artifact["strict_separation_by_fixture"][fixture] is True
    assert artifact["zero_variance_fallback_count"] >= 1
    assert artifact["fallback_order_equivalence"] is True
    assert all(row["unsafe_route_accepted"] is False for row in artifact["negative_control_results"])
    assert all(row["unsafe_route_accepted"] is False for row in artifact["corrupted_control_results"])
    assert artifact["route_confusion_matrix"]["macro_precision"] == 1.0
    assert artifact["route_confusion_matrix"]["macro_recall"] == 1.0


def test_req_arc_wmte_5711_registry_precheck_excludes_duplicates_and_claims_no_solve() -> None:
    """REQ-ARC-WMTE-5711: reproduced fixtures are prechecked and duplicate-free."""

    artifact = mod.build_artifact(root=REPO)
    precheck = artifact["registry_precheck"]
    reproduced = artifact["reproduced_level_fixture_manifest"]

    assert precheck["solve_provenance"] == "development_proxy"
    assert precheck["duplicates_excluded"] is True
    assert precheck["duplicate_count"] == 0
    assert all(row["already_reproduced"] for row in reproduced)
    assert {row["source"] for row in reproduced} == {"agent_owned_reproduced_receipt"}
    assert artifact["new_levels_claimed"] == 0


def test_req_arc_wmte_5711_helper_defensive_branches(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5711: malformed receipts and classifier mismatches stay explicit."""

    assert mod._load_upstream_receipt(tmp_path) == {}  # noqa: SLF001

    monkeypatch.setattr(
        mod,
        "_load_upstream_receipt",
        lambda _root: {
            "per_game": [
                {},
                {"game": "aa00", "prefix_level": 0},
                {"game": "aa00", "prefix_level": 1},
                {"game": "aa00", "prefix_level": 1},
            ]
        },
    )
    precheck, rows = mod.registry_precheck(tmp_path)
    assert precheck["duplicate_count"] == 1
    assert rows == [
        {
            "game": "aa00",
            "level": 1,
            "already_reproduced": True,
            "source": "agent_owned_reproduced_receipt",
            "receipt_path": mod.UPSTREAM_REPRODUCED_RECEIPT,
        }
    ]

    fixture = mod._positive_fixtures()[0]  # noqa: SLF001
    mismatched = dict(fixture)
    mismatched["route_class"] = "centroid_alignment"
    monkeypatch.setattr(mod, "_positive_fixtures", lambda: [mismatched])
    _variance, _separation, matrix = mod._score_positive_fixtures()  # noqa: SLF001
    assert matrix["by_class"]["centroid_alignment"]["fn"] == 1
    assert matrix["by_class"]["region_pair_equality"]["fp"] == 1


def test_req_arc_wmte_5711_repository_artifact_is_stable_and_schema_valid() -> None:
    """REQ-ARC-WMTE-5711: checked-in artifact is the stable qualification receipt."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
    assert result["schema"] == mod.SCHEMA
    assert result["solve_provenance"] == "development_proxy"
    assert result["inference_substrate"] == "arc_visible_state_relational_energy_no_llm"
    assert result["live_path_reachable_score"] == 1.0
    assert result["relational_goal_energy_ready_score"] == 1.0
    assert result["new_levels_claimed"] == 0
    assert result["per_game_constant_scan"]["per_game_constants_detected"] is False
    assert result["game_source_read_count"] == 0
    assert result["game_adapter_count"] == 0
    assert result["outer_loop_bfs_used"] is False
    assert result["per_game_leakage_detected"] is False
    assert result["honest_verdict"].startswith("complete:")
