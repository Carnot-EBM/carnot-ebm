"""Tests for Exp 5591 blob-topology offline-sim prototype.

Spec refs: REQ-ARC-FCP-5591, SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY,
SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5591_blob_topology_offline_sim_prototype as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5591_spec_declares_topology_contract() -> None:
    """REQ-ARC-FCP-5591: OpenSpec declares the object_hash/blob_topology contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5591") :]

    for marker in (
        "REQ-ARC-FCP-5591",
        "SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY",
        "SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY",
        "object_hash",
        "blob_topology",
        "children",
        "adjacency_list",
    ):
        assert marker in section


def test_scenario_arc_fcp_5591_blocked_precondition_never_measures(monkeypatch) -> None:
    """A missing resource fails closed without attempting any game."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "blob_topology_import": True,
            "ok": False,
        },
    )

    def _fail_if_called(game, **_kwargs):
        raise AssertionError("_measure_one_game must not run when a precondition is missing")

    monkeypatch.setattr(mod, "_measure_one_game", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["per_game_rows"] == []
    assert artifact["total_games_measured"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_fcp_5591_synthetic_persistence_confirmed(monkeypatch) -> None:
    """A synthetic all-persisted result is classified as identity-confirmed."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "blob_topology_import": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "_measure_one_game",
        lambda game, **_kwargs: {
            "game": game,
            "initial_blob_count": 5,
            "initial_max_containment_depth": 1,
            "initial_adjacency_edge_count": 4,
            "actions_taken": [1],
            "cross_frame_hash_persisted": True,
        },
    )

    artifact = mod.build_artifact(roster=("cd82", "m0r0"))

    assert artifact["games_with_cross_frame_hash_persistence"] == 2
    assert artifact["total_games_measured"] == 2
    assert "cross_frame_identity_confirmed_2_of_2_games" in artifact["honest_verdict"]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_fcp_5591_synthetic_no_persistence_is_honest_null(monkeypatch) -> None:
    """A synthetic zero-persistence result reports an honest, not-fabricated null."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "blob_topology_import": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "_measure_one_game",
        lambda game, **_kwargs: {
            "game": game,
            "initial_blob_count": 5,
            "initial_max_containment_depth": 1,
            "initial_adjacency_edge_count": 4,
            "actions_taken": [1, 2, 3, 4],
            "cross_frame_hash_persisted": False,
        },
    )

    artifact = mod.build_artifact(roster=("cd82",))

    assert artifact["games_with_cross_frame_hash_persistence"] == 0
    assert artifact["honest_verdict"] == (
        "complete: blob_topology_prototype_ran_but_no_cross_frame_persistence_observed"
    )


def test_req_arc_fcp_5591_max_containment_depth_helper() -> None:
    """_max_containment_depth returns the tree height, 1 for a flat forest, 0 for no blobs."""

    assert mod._max_containment_depth({}) == 0
    assert mod._max_containment_depth({0: [], 1: []}) == 1
    assert mod._max_containment_depth({0: [1], 1: [2], 2: []}) == 3


def test_req_arc_fcp_5591_repository_artifact_confirms_real_cross_frame_identity() -> None:
    """REQ-ARC-FCP-5591: the checked-in real run confirms identity persistence on real frames."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["total_games_measured"] > 0
    assert result["games_with_cross_frame_hash_persistence"] > 0
    assert (
        result["inference_substrate"] == "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert result["solve_provenance"] == "development_proxy"
    for row in result["per_game_rows"]:
        if "error" in row:
            continue
        assert row["initial_blob_count"] >= 1
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
