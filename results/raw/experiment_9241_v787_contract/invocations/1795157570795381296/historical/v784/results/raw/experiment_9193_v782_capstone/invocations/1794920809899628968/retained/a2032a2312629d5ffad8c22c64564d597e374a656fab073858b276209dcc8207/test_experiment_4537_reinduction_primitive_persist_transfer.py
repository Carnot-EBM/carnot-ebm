"""Tests for Exp 4537 reusable per-level re-induction transfer.

Spec refs: REQ-ARC-WMTE-4537, SCENARIO-ARC-WMTE-4537.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4537_reinduction_primitive_persist_transfer as exp4537
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def test_req_arc_wmte_4537_spec_declares_transfer_contract() -> None:
    """REQ-ARC-WMTE-4537: OpenSpec declares the persisted primitive and artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4537",
        "SCENARIO-ARC-WMTE-4537",
        "per_level_reinduction_operator",
        exp4537.RESULT_RELATIVE_PATH,
        "primitive_persisted",
        "representation_transfer",
        "arc_solver_kit.reproduce()",
    ):
        assert marker in spec


def test_req_arc_wmte_4537_solver_kit_operator_reinduces_and_routes() -> None:
    """REQ-ARC-WMTE-4537: level-up detection re-induces L_{n+1} and returns a route."""

    def inducer(goal_level: int, context: dict[str, object]) -> dict[str, object]:
        return {
            "predicate_id": f"L{goal_level}_predicate",
            "signature": f"sig-{goal_level}",
            "representation_correct": True,
            "source_level": context["from_level"],
        }

    result = kit.per_level_reinduction_operator(
        [
            SimpleNamespace(levels_completed=0),
            SimpleNamespace(levels_completed=1),
            SimpleNamespace(levels_completed=2),
        ],
        predicate_inducer=inducer,
        route_builder=lambda event: {
            "route": "depth_primary_goal_bias",
            "goal_bias_label": event["predicate"]["predicate_id"],
        },
    )

    assert result["operator"] == "per_level_reinduction_operator"
    assert result["level_ups_detected"] == 2
    assert result["stale_state_cleared"] is True
    assert [event["next_goal_level"] for event in result["events"]] == [2, 3]
    assert result["events"][0]["predicate"]["signature"] == "sig-2"
    assert result["events"][1]["representation_transfer"] is True
    assert result["events"][1]["route"]["goal_bias_label"] == "L3_predicate"

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "per_level_reinduction_operator" in operators
    selected = kit.select_primitive_operators(mechanic_class="program_editor", game="tn36")
    assert selected[0].operator == "per_level_reinduction_operator"


def test_req_arc_wmte_4537_recommend_approach_surfaces_operator_and_registry_gotcha() -> None:
    """REQ-ARC-WMTE-4537: routing and registry expose the reusable primitive."""

    recommendation = arc_solve_learning.recommend_approach("tu93")
    selected = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert selected[0] == "per_level_reinduction_operator"

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotcha = next(
        row
        for row in registry["general_gotchas"]
        if row.get("id") == exp4537.PRIMITIVE_GOTCHA_ID
    )
    assert gotcha["operator"] == "per_level_reinduction_operator"
    assert "detect level-up" in gotcha["note"]


def test_req_arc_wmte_4537_transfer_predicate_route_and_registry_helpers() -> None:
    """REQ-ARC-WMTE-4537: transfer helpers expose representation-correct route data."""

    registry = {
        "general_gotchas": [
            {"id": exp4537.PRIMITIVE_GOTCHA_ID, "operator": "per_level_reinduction_operator"}
        ],
        "games": [{"game": "tn36", "levels_reproduced": "7", "mechanic_class": "program_editor"}],
    }

    assert exp4537._registry_has_primitive_gotcha(registry) is True
    entry = exp4537._registry_game(registry, "tn36")
    assert exp4537._as_int(entry["levels_reproduced"]) == 7
    assert exp4537._as_int("not-an-int") == 0
    assert exp4537._registry_game(registry, "missing") == {}
    predicate = exp4537._predicate_for_game("tn36", 8, entry)
    assert predicate["predicate_id"] == "tn36_L8_program_editor_predicate"
    assert predicate["representation_correct"] is True
    fallback = exp4537._predicate_for_game(
        "zz99",
        2,
        {"mechanic_class": "custom_mechanic", "win_condition": "custom predicate"},
    )
    assert fallback["predicate_id"] == "zz99_L2_custom_mechanic_predicate"
    route = exp4537._route_for_event({"predicate": predicate})
    assert route["depth_primary"] is True
    assert route["goal_bias_label"] == predicate["predicate_id"]
    assert exp4537._route_for_event({})["goal_bias_label"] == ""


def test_scenario_arc_wmte_4537_artifact_schema_records_transfer_null() -> None:
    """SCENARIO-ARC-WMTE-4537: transfer nulls still preserve representation wins."""

    artifact = exp4537.build_artifact(
        a1_summary={"honest_verdict": "complete: reinduction_no_deeper_level_barrier_refined_honest_null"},
        preconditions_checked={
            "offline_arcade_import_smoke": True,
            "a1_artifact_present": True,
            "spec_has_req_4537": True,
        },
        transfer_results=[
            {
                "game": "tu93",
                "prior_reproduced_level": 5,
                "deepest_level_reached": 5,
                "offline_reproduced": True,
                "new_level_banked": False,
                "representation_transfer": True,
                "predicate": {
                    "predicate_id": "tu93_L6_goal_distance_predicate",
                    "representation_correct": True,
                    "signature": "tu93:L6:goal_distance",
                },
            },
            {
                "game": "sc25",
                "prior_reproduced_level": 5,
                "deepest_level_reached": 5,
                "offline_reproduced": True,
                "new_level_banked": False,
                "representation_transfer": True,
                "predicate": {
                    "predicate_id": "sc25_L6_cast_grid_exit_predicate",
                    "representation_correct": True,
                    "signature": "sc25:L6:cast_grid_exit",
                },
            },
        ],
        registry_updated=True,
        random_seed=4537,
        duration_s=0.0,
    )

    assert exp4537.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == (
        "complete: reinduction_primitive_persisted_transfer_null_characterized"
    )
    assert artifact["primitive_persisted"]["operator"] == "per_level_reinduction_operator"
    assert artifact["transfer_games"] == ["tu93", "sc25"]
    assert artifact["transfer_deepest_level_per_game"] == {"tu93": 5, "sc25": 5}
    assert artifact["representation_transfer"] == {"tu93": True, "sc25": True}
    assert artifact["offline_reproduced"] is False


def test_scenario_arc_wmte_4537_success_and_blocked_artifact_branches() -> None:
    """SCENARIO-ARC-WMTE-4537: success is reserved for reproduced new levels."""

    success = exp4537.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "tn36",
                "deepest_level_reached": 8,
                "new_level_banked": True,
                "offline_reproduced": True,
                "representation_transfer": True,
            },
            {
                "game": "tu93",
                "deepest_level_reached": 5,
                "new_level_banked": False,
                "offline_reproduced": True,
                "representation_transfer": True,
            },
        ],
        registry_updated=True,
        random_seed=4537,
        duration_s=None,
    )
    assert success["honest_verdict"] == "success: reinduction_primitive_persisted_transfer_tn36_L8"
    assert success["offline_reproduced"] is True
    assert success["new_levels_banked"] == 1
    assert exp4537.artifact_schema_errors(success) == []

    blocked = exp4537.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=4537,
        duration_s=0.0,
    )
    assert blocked["honest_verdict"] == "blocked_reinduction_primitive_transfer_precondition"


def test_req_arc_wmte_4537_schema_errors_reject_bad_artifacts() -> None:
    """REQ-ARC-WMTE-4537: schema validation catches false transfer claims."""

    errors = exp4537.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match" in errors
    assert "primitive_persisted must name per_level_reinduction_operator" in errors
    assert "transfer_games must contain at least two games" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    bad_success = exp4537.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "tn36",
                "deepest_level_reached": 8,
                "new_level_banked": True,
                "offline_reproduced": True,
                "representation_transfer": True,
            },
            {
                "game": "tu93",
                "deepest_level_reached": 5,
                "new_level_banked": False,
                "offline_reproduced": True,
                "representation_transfer": True,
            },
        ],
        registry_updated=True,
        random_seed=4537,
        duration_s=0.0,
    )
    bad_success["offline_reproduced"] = False
    bad_success["new_levels_banked"] = 0
    bad_success["reproducibility_checksum"] = exp4537.payload_checksum(bad_success)
    success_errors = exp4537.artifact_schema_errors(bad_success)
    assert "success requires offline_reproduced=true" in success_errors
    assert "success requires at least one new level banked" in success_errors

    false_null = exp4537.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "tu93",
                "deepest_level_reached": 5,
                "new_level_banked": False,
                "offline_reproduced": True,
                "representation_transfer": True,
            },
            {
                "game": "sc25",
                "deepest_level_reached": 5,
                "new_level_banked": False,
                "offline_reproduced": True,
                "representation_transfer": True,
            },
        ],
        registry_updated=True,
        random_seed=4537,
        duration_s=0.0,
    )
    false_null["offline_reproduced"] = True
    false_null["reproducibility_checksum"] = exp4537.payload_checksum(false_null)
    assert "non-success cannot claim offline_reproduced=true for a new level" in (
        exp4537.artifact_schema_errors(false_null)
    )

    tampered = exp4537.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {"game": "tu93", "deepest_level_reached": 5},
            {"game": "sc25", "deepest_level_reached": 5},
        ],
        registry_updated=True,
        random_seed=4537,
        duration_s=0.0,
    )
    tampered["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in exp4537.artifact_schema_errors(
        tampered
    )
