"""Tests for Exp 4561 primitive persistence and cross-game transfer.

Spec refs: REQ-ARC-WMTE-4561, SCENARIO-ARC-WMTE-4561.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4561_primitive_persist_transfer as exp4561
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _attempt(game: str, mode: str, *, solved: bool, actions: int, reached: int = 0) -> dict[str, object]:
    return {
        "game": game,
        "router_mode": mode,
        "variant_signature": f"{game}~color01",
        "solved": solved,
        "reached_level": reached,
        "actions": actions,
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached,
            "reached_level": reached,
            "reproduced": solved,
        },
    }


def test_req_arc_wmte_4561_spec_declares_primitive_transfer_contract() -> None:
    """REQ-ARC-WMTE-4561: OpenSpec declares the persisted primitive and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4561",
        "SCENARIO-ARC-WMTE-4561",
        "verifier_router_candidate_ranking_operator",
        exp4561.RESULT_RELATIVE_PATH,
        "transfer_value_per_game",
        "primitive_verifier_router_candidate_ranking_operator",
    ):
        assert marker in spec
    for field, principle in exp4561.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4561_solver_kit_operator_ranks_candidates_and_reports_gain() -> None:
    """REQ-ARC-WMTE-4561: verifier-router ranking is deterministic and reports value-add."""

    candidates = [
        {"candidate_id": "baseline", "verifier_score": 0.2, "reaches_goal": False},
        {"candidate_id": "target", "verifier_score": 0.9, "reaches_goal": True},
        {"candidate_id": "tie_after_target", "verifier_score": 0.9, "reaches_goal": False},
    ]

    result = kit.verifier_router_candidate_ranking_operator(
        candidates,
        score_key="verifier_score",
        target_key="reaches_goal",
    )

    assert result["operator"] == "verifier_router_candidate_ranking_operator"
    assert [row["candidate_id"] for row in result["ranked_candidates"]] == [
        "target",
        "tie_after_target",
        "baseline",
    ]
    assert result["target_rank_before"] == 1
    assert result["target_rank_after"] == 0
    assert result["ordering_gain"] == 1
    assert result["value_added"] is True

    no_gain = kit.verifier_router_candidate_ranking_operator(
        [
            {"candidate_id": "already_first", "verifier_score": 0.5, "reaches_goal": True},
            {"candidate_id": "same_score", "verifier_score": 0.5, "reaches_goal": False},
        ],
        score_key="verifier_score",
        target_key="reaches_goal",
    )
    assert [row["candidate_id"] for row in no_gain["ranked_candidates"]] == [
        "already_first",
        "same_score",
    ]
    assert no_gain["ordering_gain"] == 0
    assert no_gain["value_added"] is False


def test_req_arc_wmte_4561_routing_and_registry_surface_persisted_primitive() -> None:
    """REQ-ARC-WMTE-4561: routing and registry expose the reusable candidate ranker."""

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert exp4561.PRIMITIVE_OPERATOR in operators

    recommendation = arc_solve_learning.recommend_approach("tu93")
    selected = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert exp4561.PRIMITIVE_OPERATOR in selected

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row
        for row in registry["general_gotchas"]
        if row.get("id") == exp4561.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == exp4561.PRIMITIVE_OPERATOR
    assert "ordering gain" in gotchas[0]["note"]


def test_req_arc_wmte_4561_selects_a1_when_a2_positive_control_failed() -> None:
    """REQ-ARC-WMTE-4561: the best-characterized A1 primitive is persisted on A2 failure."""

    decision = exp4561.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "complete: verifier_router_no_value_added_honest_null_gap_sharpened",
            "inference_substrate": exp4561.INFERENCE_SUBSTRATE,
            "generic_transfer_rate_with_verifier": 0.04,
            "generic_transfer_rate_baseline": 0.04,
            "generic_transfer_delta": 0.0,
            "offline_reproduced": True,
        },
        a2_artifact={
            "honest_verdict": "complete: executable_proposer_positive_control_failed_no_deeper_barrier_refined",
            "positive_control_passed": False,
            "llm_proposer_value": {"rate": 0.0},
        },
    )

    assert decision["source"] == "A1_verifier_router"
    assert decision["operator"] == exp4561.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == exp4561.PRIMITIVE_GOTCHA_ID
    assert decision["persisted_as_best_characterized_null"] is True


def test_req_arc_wmte_4561_cached_transfer_measurement_uses_persisted_ranker() -> None:
    """REQ-ARC-WMTE-4561: cached transfer rows report ordering gain on held-out games."""

    a1_artifact = {
        "baseline_measurement": {
            "variant_attempts": [_attempt("zz99", "baseline", solved=False, actions=50)]
        },
        "verifier_measurement": {
            "variant_attempts": [_attempt("zz99", "verifier", solved=True, actions=8, reached=1)]
        },
        "random_router_measurement": {
            "variant_attempts": [_attempt("zz99", "random", solved=False, actions=49)]
        },
    }

    row = exp4561.measure_cached_verifier_transfer_game(
        "zz99",
        a1_artifact=a1_artifact,
        incoming_order=("baseline", "random", "verifier"),
    )

    assert row["game"] == "zz99"
    assert row["value_added"] is True
    assert row["transfer_value"]["ordering_gain"] == 2
    assert row["transfer_value"]["best_candidate_id"] == "verifier"
    assert row["transfer_value"]["target_rank_after"] == 0


def test_scenario_arc_wmte_4561_artifact_schema_records_success_and_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4561: transfer artifacts distinguish value-add from null."""

    decision = {
        "source": "A1_verifier_router",
        "operator": exp4561.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": exp4561.PRIMITIVE_GOTCHA_ID,
    }
    success = exp4561.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": True, "offline_arcade_import_smoke": True},
        transfer_results=[
            {
                "game": "tu93",
                "value_added": True,
                "transfer_value": {"ordering_gain": 2, "value_added": True},
                "offline_reproduced_new_level": False,
                "dead_end": "",
            },
            {
                "game": "tr87",
                "value_added": False,
                "transfer_value": {"ordering_gain": 0, "value_added": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no cached candidate reached the goal",
            },
        ],
        registry_updated=True,
        random_seed=4561,
        duration_s=0.0,
    )

    assert success["honest_verdict"] == "success: primitive_persisted_transfer_tu93_value_added"
    assert success["offline_reproduced"] is False
    assert success["new_levels_banked"] == 0
    assert success["transfer_value_per_game"]["tu93"]["ordering_gain"] == 2
    assert exp4561.artifact_schema_errors(success) == []

    out = exp4561.write_artifact(success, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == success

    null = exp4561.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "tu93",
                "value_added": False,
                "transfer_value": {"ordering_gain": 0, "value_added": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no cached candidate reached the goal",
            },
            {
                "game": "tr87",
                "value_added": False,
                "transfer_value": {"ordering_gain": 0, "value_added": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no cached candidate reached the goal",
            },
        ],
        registry_updated=True,
        random_seed=4561,
        duration_s=0.0,
    )
    assert null["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert null["transfer_dead_ends"]["tu93"] == "no cached candidate reached the goal"
    assert exp4561.artifact_schema_errors(null) == []


def test_req_arc_wmte_4561_schema_errors_reject_false_claims() -> None:
    """REQ-ARC-WMTE-4561: schema validation catches malformed transfer claims."""

    errors = exp4561.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert "primitive_persisted must name verifier_router_candidate_ranking_operator" in errors
    assert "transfer_games must contain at least two games" in errors
    assert "transfer_value_per_game must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    artifact = exp4561.build_artifact(
        upstream_decision={
            "source": "A1_verifier_router",
            "operator": exp4561.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": exp4561.PRIMITIVE_GOTCHA_ID,
        },
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "tu93",
                "value_added": True,
                "transfer_value": {"ordering_gain": 1, "value_added": True},
                "offline_reproduced_new_level": False,
            },
            {
                "game": "tr87",
                "value_added": False,
                "transfer_value": {"ordering_gain": 0, "value_added": False},
                "offline_reproduced_new_level": False,
            },
        ],
        registry_updated=True,
        random_seed=4561,
        duration_s=0.0,
    )
    artifact["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in exp4561.artifact_schema_errors(
        artifact
    )
