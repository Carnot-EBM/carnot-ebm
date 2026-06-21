"""Tests for Exp 4549 reusable LLM-proposer re-induction transfer.

Spec refs: REQ-ARC-WMTE-4549, SCENARIO-ARC-WMTE-4549.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4549_llm_proposer_primitive_persist_transfer as exp4549
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def test_req_arc_wmte_4549_spec_declares_llm_primitive_transfer_contract() -> None:
    """REQ-ARC-WMTE-4549: OpenSpec declares the persisted LLM primitive and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4549",
        "SCENARIO-ARC-WMTE-4549",
        "llm_proposer_reinduction_operator",
        exp4549.RESULT_RELATIVE_PATH,
        "reachable_plan_produced",
        "primitive_per_level_reinduction_operator",
        "arc_solver_kit.reproduce()",
    ):
        assert marker in spec


def test_req_arc_wmte_4549_solver_kit_operator_ranks_refines_and_degrades() -> None:
    """REQ-ARC-WMTE-4549: LLM proposal transfer ranks, refines, and has DSL fallback."""

    calls: list[dict[str, object]] = []

    def proposal_provider(goal_level: int, context: dict[str, object]):
        calls.append({"goal_level": goal_level, "round": context["refinement_round"]})
        if context["refinement_round"] == 1:
            return [
                {
                    "name": "weak_unreachable",
                    "goal_predicate": f"L{goal_level}_weak",
                    "dynamics_model": "stuck",
                    "trust_energy": 2.0,
                    "reachable_plan": False,
                    "representation_correct": True,
                    "signature": "weak",
                }
            ]
        return [
            {
                "name": "trusted_reachable",
                "goal_predicate": f"L{goal_level}_goal",
                "dynamics_model": "progress",
                "plan": [{"action": 1}],
                "trust_energy": 0.0,
                "reachable_plan": True,
                "representation_correct": True,
                "signature": f"L{goal_level}:trusted",
            }
        ]

    result = kit.llm_proposer_reinduction_operator(
        [SimpleNamespace(levels_completed=0), SimpleNamespace(levels_completed=1)],
        proposal_provider=proposal_provider,
        initial_predicate={"signature": "L1:seed"},
    )

    assert result["operator"] == "llm_proposer_reinduction_operator"
    assert result["base_operator"] == "per_level_reinduction_operator"
    assert result["reachable_plan_produced"] is True
    assert result["representation_transfer"] is True
    assert len(calls) == 2
    event = result["events"][0]
    assert event["trust_energy_ranked"] is True
    assert event["refinement_rounds_used"] == 2
    assert event["selected_candidate_name"] == "trusted_reachable"
    assert event["predicate"]["signature"] == "L2:trusted"
    assert event["route"]["depth_primary"] is True

    fallback = kit.llm_proposer_reinduction_operator(
        [SimpleNamespace(levels_completed=0), SimpleNamespace(levels_completed=1)],
        proposal_provider=None,
        fallback_predicate_inducer=lambda goal_level, _context: {
            "predicate_id": f"fallback_L{goal_level}",
            "signature": f"fallback:{goal_level}",
            "representation_correct": True,
        },
        initial_predicate={"signature": "seed"},
    )
    assert fallback["reachable_plan_produced"] is False
    assert fallback["events"][0]["proposal_mode"] == "dsl_fallback"
    assert fallback["events"][0]["llm_proposer_invoked"] is False
    assert fallback["events"][0]["predicate"]["predicate_id"] == "fallback_L2"


def test_req_arc_wmte_4549_routing_and_registry_extend_existing_gotcha() -> None:
    """REQ-ARC-WMTE-4549: routing surfaces the primitive and registry extends, not duplicates."""

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "llm_proposer_reinduction_operator" in operators

    recommendation = arc_solve_learning.recommend_approach("tr87")
    selected = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert "llm_proposer_reinduction_operator" in selected

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row
        for row in registry["general_gotchas"]
        if row.get("id") == exp4549.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == "per_level_reinduction_operator"
    assert "llm_proposer_reinduction_operator" in gotchas[0]["note"]


def test_scenario_arc_wmte_4549_artifact_schema_records_transfer_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4549: transfer nulls preserve plan and representation evidence."""

    artifact = exp4549.build_artifact(
        a1_summary={
            "honest_verdict": "complete: llm_proposer_positive_control_failed_false_negative_risk_open",
            "barrier_refinement": "positive_control_failed",
        },
        preconditions_checked={
            "offline_arcade_import_smoke": True,
            "a1_artifact_present": True,
            "spec_has_req_4549": True,
            "registry_has_extended_primitive_gotcha": True,
            "ok": True,
        },
        transfer_results=[
            {
                "game": "tu93",
                "prior_reproduced_level": 5,
                "deepest_level_reached": 5,
                "current_depth_reproduced": True,
                "new_level_banked": False,
                "reachable_plan_produced": False,
                "representation_transfer": True,
            },
            {
                "game": "tr87",
                "prior_reproduced_level": 6,
                "deepest_level_reached": 6,
                "current_depth_reproduced": True,
                "new_level_banked": False,
                "reachable_plan_produced": False,
                "representation_transfer": True,
            },
        ],
        registry_updated=True,
        random_seed=4549,
        duration_s=0.0,
        inference_substrate=exp4549.DSL_FALLBACK_INFERENCE_SUBSTRATE,
    )

    assert exp4549.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == (
        "complete: llm_proposer_primitive_persisted_transfer_null_characterized"
    )
    assert artifact["primitive_persisted"]["operator"] == "llm_proposer_reinduction_operator"
    assert artifact["primitive_persisted"]["registry_general_gotcha_id"] == (
        "primitive_per_level_reinduction_operator"
    )
    assert artifact["transfer_deepest_level_per_game"] == {"tu93": 5, "tr87": 6}
    assert artifact["reachable_plan_produced"] == {"tu93": False, "tr87": False}
    assert artifact["representation_transfer"] == {"tu93": True, "tr87": True}
    assert artifact["offline_reproduced"] is False

    out = exp4549.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_scenario_arc_wmte_4549_success_blocked_and_schema_guards() -> None:
    """SCENARIO-ARC-WMTE-4549: only reproduced new levels can produce success."""

    success = exp4549.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "tn36",
                "deepest_level_reached": 8,
                "new_level_banked": True,
                "current_depth_reproduced": True,
                "reachable_plan_produced": True,
                "representation_transfer": True,
            },
            {
                "game": "tu93",
                "deepest_level_reached": 5,
                "new_level_banked": False,
                "current_depth_reproduced": True,
                "reachable_plan_produced": False,
                "representation_transfer": True,
            },
        ],
        registry_updated=True,
        random_seed=4549,
        duration_s=None,
        inference_substrate="live_llm_inference + model_specs",
    )
    assert success["honest_verdict"] == (
        "success: llm_proposer_primitive_persisted_transfer_tn36_L8"
    )
    assert success["offline_reproduced"] is True
    assert success["new_levels_banked"] == 1
    assert exp4549.artifact_schema_errors(success) == []

    blocked = exp4549.build_artifact(
        a1_summary={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=4549,
        duration_s=0.0,
        inference_substrate=exp4549.DSL_FALLBACK_INFERENCE_SUBSTRATE,
    )
    assert blocked["honest_verdict"] == "blocked_llm_proposer_primitive_transfer_precondition"

    errors = exp4549.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert "primitive_persisted must name llm_proposer_reinduction_operator" in errors
    assert "transfer_games must contain at least two games" in errors
    assert "reachable_plan_produced must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    bad_success = dict(success)
    bad_success["offline_reproduced"] = False
    bad_success["new_levels_banked"] = 0
    bad_success["reproducibility_checksum"] = exp4549.payload_checksum(bad_success)
    success_errors = exp4549.artifact_schema_errors(bad_success)
    assert "success requires offline_reproduced=true" in success_errors
    assert "success requires at least one new level banked" in success_errors

    tampered = dict(success)
    tampered["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in exp4549.artifact_schema_errors(
        tampered
    )
