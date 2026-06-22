"""Tests for Exp 4596 primitive persistence and transfer.

Spec refs: REQ-CAPSTONE-4596, SCENARIO-CAPSTONE-4596,
SCENARIO-CAPSTONE-4596-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4596_primitive_persist_transfer as mod
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(
    approach: str,
    *,
    winner: bool,
    candidate_generated: bool = True,
    reached_level: int = 1,
) -> dict[str, Any]:
    return {
        "approach": approach,
        "candidate_id": f"{approach}-candidate",
        "candidate_generated": candidate_generated,
        "winner_generated": winner,
        "win_reached": winner,
        "reached_level": reached_level if winner else 0,
        "actions": 5 if winner else 0,
        "reproduction_gate": {
            "game": "zz99",
            "claimed_level": reached_level if winner else 0,
            "reached_level": reached_level if winner else 0,
            "reproduced": winner,
            "mode": "offline_reproduction_gate_no_quota",
        },
    }


def test_req_capstone_4596_spec_declares_dispatcher_transfer_contract() -> None:
    """REQ-CAPSTONE-4596: OpenSpec declares the persisted dispatcher primitive."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4596",
        "SCENARIO-CAPSTONE-4596",
        "SCENARIO-CAPSTONE-4596-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4596_solver_kit_dispatcher_selects_routed_winner() -> None:
    """REQ-CAPSTONE-4596: dispatcher executes the routed approach candidate."""

    result = kit.approach_dispatcher_operator(
        {"mechanic_class": "avatar_navigation", "approach": "goal_distance_astar"},
        [
            _candidate("default_graph_explore", winner=False, candidate_generated=False),
            _candidate("goal_distance_astar", winner=True, reached_level=2),
        ],
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["selected_approach"] == "goal_distance_astar"
    assert result["candidate_generated"] is True
    assert result["winner_generated"] is True
    assert result["win_reached"] is True
    assert result["baseline_winner_generated"] is False
    assert result["value_added"] is True
    assert result["selected_candidate"]["candidate_id"] == "goal_distance_astar-candidate"

    missing = kit.approach_dispatcher_operator(
        {"mechanic_class": "keyboard_graph", "approach": "systematic_bfs"},
        [_candidate("default_graph_explore", winner=False, candidate_generated=False)],
    )
    assert missing["selected_approach"] == "systematic_bfs"
    assert missing["executed_approach"] == "default_graph_explore"
    assert missing["value_added"] is False
    assert missing["dead_end"].startswith("no generated candidate for routed approach")


def test_req_capstone_4596_routing_and_registry_surface_dispatcher() -> None:
    """REQ-CAPSTONE-4596: routing and registry expose the reusable dispatcher."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}

    selected = kit.select_primitive_operators(mechanic_class="avatar_navigation", action_model="mixed")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("ar25")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert mod.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "dispatcher" in gotchas[0]["note"].lower()
    assert "latest_exp4596_transfer" in gotchas[0]


def test_req_capstone_4596_selects_a1_dispatcher_over_a3_null() -> None:
    """REQ-CAPSTONE-4596: A1 wins as the strongest persisted primitive."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "success: generation_completeness_winner_generated_2of25_above_1of25",
            "winner_generated_delta": 0.04,
            "generic_transfer_rate_with_wiring": 0.08,
            "generic_transfer_rate_baseline": 0.04,
            "newly_solved_variants": ["sp80~color01"],
        },
        a3_artifact={
            "honest_verdict": "complete: goal_energy_prior_no_value_honest_null_gap_sharpened",
            "winner_generated_delta": 0.0,
            "generic_transfer_rate_with_energy": 0.0,
        },
    )

    assert decision["source"] == "A1_approach_dispatcher"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] > 0.0
    assert decision["source_tuning_games"] == ["sp80"]


def test_req_capstone_4596_transfer_measurement_reports_value_and_null() -> None:
    """REQ-CAPSTONE-4596: transfer records per-game dispatcher value-add."""

    value = mod.measure_dispatcher_transfer_game(
        "ar25",
        route={"mechanic_class": "avatar_navigation", "approach": "goal_distance_astar"},
        candidates=[
            _candidate("default_graph_explore", winner=False, candidate_generated=False),
            _candidate("goal_distance_astar", winner=True, reached_level=2),
        ],
        source_tuning_games=("sp80",),
    )
    assert value["game"] == "ar25"
    assert value["not_tuned_on_source"] is True
    assert value["value_added"] is True
    assert value["transfer_value"]["winner_generated"] is True
    assert value["transfer_value"]["win_reached"] is True
    assert value["transfer_value"]["existing_reproduced_level"] == 2

    null = mod.measure_dispatcher_transfer_game(
        "bp35",
        route={"mechanic_class": "avatar_navigation", "approach": "goal_distance_astar"},
        candidates=[_candidate("default_graph_explore", winner=False, candidate_generated=False)],
        source_tuning_games=("sp80",),
    )
    assert null["value_added"] is False
    assert null["transfer_value"]["winner_generated"] is False
    assert "no generated candidate" in null["dead_end"]


def test_scenario_capstone_4596_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4596: artifact schema records value-add or honest null."""

    decision = {
        "source": "A1_approach_dispatcher",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": ["sp80"],
        "selection_rationale": "A1 raised winner-generated rate while A3 was null.",
    }
    transfer_results = [
        {
            "game": "ar25",
            "value_added": True,
            "transfer_value": {
                "winner_generated": True,
                "win_reached": True,
                "offline_reproduced_new_level": False,
                "existing_reproduced_level": 2,
                "value_added": True,
            },
            "dead_end": "",
        },
        {
            "game": "bp35",
            "value_added": False,
            "transfer_value": {
                "winner_generated": False,
                "win_reached": False,
                "offline_reproduced_new_level": False,
                "existing_reproduced_level": 0,
                "value_added": False,
            },
            "dead_end": "no generated candidate for routed approach",
        },
    ]
    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={"A1_approach_dispatcher": {"measured_signal": 0.04}},
        preconditions_checked={"ok": True},
        transfer_results=transfer_results,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success: primitive_persisted_transfer_ar25_value_added"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["offline_reproduced"]["new_levels_banked"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    null_artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[dict(row, value_added=False, dead_end="no transfer") for row in transfer_results],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert null_artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert mod.artifact_schema_errors(null_artifact) == []

    errors = mod.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least two games" in errors


def test_scenario_capstone_4596_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4596: run writes the requested result JSON."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry = {
        "schema_version": 1,
        "general_gotchas": [
            {"id": mod.PRIMITIVE_GOTCHA_ID, "operator": mod.PRIMITIVE_OPERATOR, "note": "fixture"}
        ],
        "games": [],
    }
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    _write_json(
        tmp_path / mod.A1_RELATIVE_PATH,
        {
            "honest_verdict": "success: generation_completeness_winner_generated_2of25_above_1of25",
            "winner_generated_delta": 0.04,
            "newly_solved_variants": ["sp80~color01"],
            "wired_measurement": {
                "variant_attempts": [
                    {
                        "game": "ar25",
                        "selected_approach": "goal_distance_astar",
                        "selected_feature_route": {
                            "mechanic_class": "avatar_navigation",
                            "approach": "goal_distance_astar",
                            "confidence": 1.0,
                        },
                    },
                    {
                        "game": "bp35",
                        "selected_approach": "goal_distance_astar",
                        "selected_feature_route": {
                            "mechanic_class": "avatar_navigation",
                            "approach": "goal_distance_astar",
                            "confidence": 1.0,
                        },
                    },
                ]
            },
            "baseline_measurement": {
                "variant_attempts": [
                    {"game": "ar25", "executed_approach": "default_graph_explore"},
                    {"game": "bp35", "executed_approach": "default_graph_explore"},
                ]
            },
        },
    )
    _write_json(
        tmp_path / mod.A3_RELATIVE_PATH,
        {"honest_verdict": "complete: goal_energy_prior_no_value_honest_null", "winner_generated_delta": 0.0},
    )
    _write_json(
        tmp_path / "results" / "arc_loop_solve_ar25.json",
        {
            "game": "ar25",
            "solution_labels": ["1", "2"],
            "reproduced_levels": 2,
            "reproduction_gate": {"reproduced": True, "reached_level": 2, "claimed_level": 2},
        },
    )

    artifact = mod.run(
        tmp_path,
        transfer_games=("ar25", "bp35"),
        offline_arcade_checker=lambda: True,
        reproduce_checker=lambda game, labels, claimed: {
            "game": game,
            "reproduced": bool(labels),
            "reached_level": claimed if labels else 0,
            "claimed_level": claimed,
            "mode": "fixture",
        },
        now=iter([5.0, 5.25]).__next__,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["transfer_games"] == ["ar25", "bp35"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    blocked = mod.build_artifact(
        selected_upstream={
            "source": "A1_approach_dispatcher",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
            "source_tuning_games": ["sp80"],
        },
        upstream_signals={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, root=tmp_path)


def test_req_capstone_4596_defensive_branches_are_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4596: helper branches stay deterministic and honest."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._load_registry(tmp_path) == {}
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(":\n", encoding="utf-8")
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(True) == 0
    assert mod._as_int("bad") == 0
    assert mod._normalise_action_label("3") == '{"action":3}'
    assert mod._normalise_action_label('{"action":6,"data":{"x":1,"y":2}}').startswith("{")
    assert mod._normalise_action_label('{"action":6,"x":1,"y":2}') == '{"action":6,"data":{"x":1,"y":2}}'
    assert mod._normalise_action_label({"action": 2}) == '{"action":2}'
    assert mod._normalise_action_label({"action": 6, "data": {"x": 1, "y": 2}}) == (
        '{"action":6,"data":{"x":1,"y":2}}'
    )
    assert mod._normalise_action_label(4) == '{"action":4}'

    all_null = mod.select_primitive_from_upstreams(a1_artifact={}, a3_artifact={})
    assert "All upstreams were value-null" in all_null["selection_rationale"]
    a3_larger = mod.select_primitive_from_upstreams(
        a1_artifact={"winner_generated_delta": 0.0},
        a3_artifact={"winner_generated_delta": 1.0},
    )
    assert "A3 had the larger numeric signal" in a3_larger["selection_rationale"]

    no_route = mod.measure_dispatcher_transfer_game(
        "zz99",
        route={},
        candidates=[],
        source_tuning_games=("zz99",),
    )
    assert no_route["not_tuned_on_source"] is False
    assert no_route["dead_end"].startswith("no generated candidate")

    assert mod._cached_solution_candidate(  # noqa: SLF001
        tmp_path, "missing", "goal_distance_astar", reproduce_checker=lambda *_args: {}
    ) is None
    _write_json(
        tmp_path / "results" / "arc_loop_solve_zero.json",
        {"solution_labels": ["1"], "reproduced_levels": 0},
    )
    assert mod._cached_solution_candidate(  # noqa: SLF001
        tmp_path, "zero", "goal_distance_astar", reproduce_checker=lambda *_args: {}
    ) is None
    assert mod._route_for_game("missing", {}) == {  # noqa: SLF001
        "mechanic_class": "",
        "approach": "default_graph_explore",
    }
    assert mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )["offline_arcade"] is False

    artifact = mod.build_artifact(
        selected_upstream={
            "source": "A1_approach_dispatcher",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": True,
                "transfer_value": {
                    "winner_generated": True,
                    "win_reached": True,
                    "offline_reproduced_new_level": True,
                    "existing_reproduced_level": 0,
                    "value_added": True,
                },
                "dead_end": "",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {"existing_reproduced_level": 0, "offline_reproduced_new_level": False},
                "dead_end": "no candidate",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert artifact["offline_reproduced"]["new_levels_banked"] == 1
    assert mod.artifact_schema_errors(artifact) == []

    wrong_gotcha = dict(artifact)
    wrong_gotcha["primitive_persisted"] = {
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": "wrong",
    }
    wrong_gotcha["reproducibility_checksum"] = mod.payload_checksum(wrong_gotcha)
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    no_value_success = dict(artifact)
    no_value_success["transfer_value_per_game"] = {"zz99": {"value_added": False}}
    no_value_success["reproducibility_checksum"] = mod.payload_checksum(no_value_success)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        no_value_success
    )

    mismatched_offline = dict(artifact)
    mismatched_offline["offline_reproduced"] = {"new_levels_banked": 1, "new_level_records": []}
    mismatched_offline["reproducibility_checksum"] = mod.payload_checksum(mismatched_offline)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        mismatched_offline
    )

    tampered = dict(artifact)
    tampered["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in mod.artifact_schema_errors(
        tampered
    )

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"honest_verdict": "bad"})
    with pytest.raises(ValueError, match="honest_verdict must start"):
        mod.run(tmp_path, offline_arcade_checker=lambda: True, write=False)
