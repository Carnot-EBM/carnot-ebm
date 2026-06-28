"""Tests for Exp 4933 MATM similarity retrieval.

Spec refs: REQ-ARC-WMTE-4933, SCENARIO-ARC-WMTE-4933-LIVE-WIRING,
SCENARIO-ARC-WMTE-4933-FLAG-OFF-PARITY, SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from carnot import experiment_4933_matm_similarity_retrieval_efficiency as mod
from carnot.agentic import arc_competition_agent as comp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _frame(value: int) -> np.ndarray:
    return np.asarray([[value, 0], [0, 0]], dtype=np.int16)


def _row(game: str, *, hit_rate: float, actions: int | None, reached: int) -> dict[str, Any]:
    return {
        "game": game,
        "forward_walk_hit_rate": float(hit_rate),
        "actions_to_first_levelup": actions,
        "reached_level": int(reached),
    }


def _rows(*, improved: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = [
        _row("tu93", hit_rate=0.10, actions=10, reached=1),
        _row("lp85", hit_rate=0.10, actions=20, reached=4),
        _row("sp80", hit_rate=0.10, actions=30, reached=1),
        _row("cn04", hit_rate=0.10, actions=40, reached=1),
        _row("m0r0", hit_rate=0.10, actions=50, reached=1),
    ]
    similarity = [
        _row("tu93", hit_rate=0.20, actions=9 if improved else 10, reached=1),
        _row("lp85", hit_rate=0.20, actions=19 if improved else 20, reached=4),
        _row("sp80", hit_rate=0.20, actions=30, reached=1),
        _row("cn04", hit_rate=0.20, actions=40, reached=1),
        _row("m0r0", hit_rate=0.20, actions=50, reached=1),
    ]
    return baseline, similarity


def _preconditions() -> dict[str, Any]:
    return {
        "arcade_importable": True,
        "metaharness_present": True,
        "fixtures_present": {game: True for game in mod.GAME_IDS},
        "generator_required": False,
        "blocked_resource": "",
    }


def test_req_arc_wmte_4933_spec_declares_similarity_retrieval_contract() -> None:
    """REQ-ARC-WMTE-4933: OpenSpec anchors the live flag and artifact gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4933")
    end = spec.index("## Implementation Status", start)
    section = _normalise(spec[start:end])

    for marker in (
        "REQ-ARC-WMTE-4933",
        "SCENARIO-ARC-WMTE-4933-LIVE-WIRING",
        "SCENARIO-ARC-WMTE-4933-FLAG-OFF-PARITY",
        "SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE",
        mod.RESULT_RELATIVE_PATH,
        "cross_game_features_v2",
        "2606.19911",
        "2603.10600",
        "2605.18871",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


def test_scenario_arc_wmte_4933_flag_off_preserves_submitted_exact_hash_default() -> None:
    """SCENARIO-ARC-WMTE-4933-FLAG-OFF-PARITY: submitted defaults stay exact-hash."""

    explorer = comp.StepwiseExplorer()

    assert comp.SUBMITTED_AGENT_CONFIG["matm_similarity_retrieval_enabled"] is False
    assert explorer.similarity_retrieval_enabled is False
    assert explorer.navigation_diagnostics()["similarity_retrieval_enabled"] is False


def test_scenario_arc_wmte_4933_similarity_prefix_routes_before_return() -> None:
    """SCENARIO-ARC-WMTE-4933-LIVE-WIRING: similar prefixes pass only through router."""

    explorer = comp.StepwiseExplorer(
        similarity_retrieval=True,
        value_head=lambda frame: float(np.asarray(frame).sum()),
        value_weight=1.0,
        goal_bias=lambda frame: float(np.asarray(frame).sum()),
        goal_bias_lower_is_better=True,
    )
    explorer._similarity_descriptor = lambda _frame: (7,)  # type: ignore[method-assign]
    explorer.graph = {
        "cur": {"path": [], "untested": [], "value": None, "frame": _frame(9)},
        "prior": {"path": [{"action": 1, "data": None}], "untested": [], "value": None, "frame": _frame(8)},
        "dst": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 3, "data": None}],
            "value": None,
            "frame": _frame(1),
        },
    }
    explorer.adj = {"prior": [({"action": 2, "data": None}, "dst")]}
    for node_hash, node in explorer.graph.items():
        explorer._index_similarity_state(node_hash, node["frame"])

    assert explorer._shortest_path("cur", "dst") == [{"action": 2, "data": None}]
    diagnostics = explorer.navigation_diagnostics()
    assert diagnostics["similarity_router_accepts"] == 1
    assert diagnostics["similarity_world_model_verifier_checks"] == 1
    assert diagnostics["exact_shortest_path_hits"] == 0


def test_scenario_arc_wmte_4933_similarity_prefix_rejects_bad_value_route() -> None:
    """SCENARIO-ARC-WMTE-4933-LIVE-WIRING: value/goal routing can reject a prefix."""

    explorer = comp.StepwiseExplorer(
        similarity_retrieval=True,
        value_head=lambda frame: float(np.asarray(frame).sum()),
        value_weight=1.0,
    )
    explorer._similarity_descriptor = lambda _frame: (7,)  # type: ignore[method-assign]
    explorer.graph = {
        "cur": {"path": [], "untested": [], "value": None, "frame": _frame(1)},
        "prior": {"path": [{"action": 1, "data": None}], "untested": [], "value": None, "frame": _frame(2)},
        "dst": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [],
            "value": None,
            "frame": _frame(9),
        },
    }
    explorer.adj = {"prior": [({"action": 2, "data": None}, "dst")]}
    for node_hash, node in explorer.graph.items():
        explorer._index_similarity_state(node_hash, node["frame"])

    assert explorer._shortest_path("cur", "dst") is None
    assert explorer.navigation_diagnostics()["similarity_router_rejects"] == 1


def test_scenario_arc_wmte_4933_artifact_gate_pass_and_null_are_explicit() -> None:
    """SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE: pass requires the full falsifiable gate."""

    baseline, similarity = _rows(improved=True)
    artifact = mod.build_artifact(
        baseline_rows=baseline,
        similarity_rows=similarity,
        preconditions_checked=_preconditions(),
        submitted_parity_test={"passed": True, "command": "pytest parity"},
        live_path_reachable=True,
        lazy_value_in_budget=True,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["flag_eligible_to_default_on"] is True
    assert artifact["actions_to_first_levelup_delta"]["tu93"] == 1
    assert artifact["actions_to_first_levelup_delta"]["lp85"] == 1
    assert artifact["reached_level_regression"] is False
    assert artifact["moves_reproducible_total_levels"] is False

    baseline, similarity = _rows(improved=False)
    null_artifact = mod.build_artifact(
        baseline_rows=baseline,
        similarity_rows=similarity,
        preconditions_checked=_preconditions(),
        submitted_parity_test={"passed": True},
        live_path_reachable=True,
        lazy_value_in_budget=True,
    )

    mod.validate_artifact(null_artifact)
    assert null_artifact["honest_verdict"] == mod.RETIRED_VERDICT
    assert null_artifact["flag_eligible_to_default_on"] is False
    assert null_artifact["retire_if_same_verdict"] is True


def test_scenario_arc_wmte_4933_blocked_artifact_names_missing_resource() -> None:
    """SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE: missing resources block honestly."""

    blocked = dict(_preconditions())
    blocked["arcade_importable"] = False
    blocked["blocked_resource"] = "arcade"

    artifact = mod.build_blocked_artifact(preconditions_checked=blocked)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_arcade"
    assert artifact["preconditions_checked"]["blocked_resource"] == "arcade"
