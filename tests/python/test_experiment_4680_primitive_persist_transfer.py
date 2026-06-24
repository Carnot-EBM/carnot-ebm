"""Tests for Exp 4680 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4680,
SCENARIO-ARC-WMTE-4680-PERSIST-STRONGEST-COMPONENT,
SCENARIO-ARC-WMTE-4680-TRANSFER-MEASUREMENT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4680_primitive_persist_transfer as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attempt(
    game: str,
    *,
    first_win: bool = False,
    reached_level: int = 0,
    actions_to_first_levelup: int | None = None,
    reproduced: bool = False,
) -> dict[str, Any]:
    return {
        "game": game,
        "first_win": first_win,
        "solved": reached_level >= 2,
        "reached_level": reached_level,
        "actions_to_first_levelup": actions_to_first_levelup,
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level if reproduced else 0,
            "reproduced": reproduced,
        },
    }


def _a1_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating",
        "chosen_submitted_config": "unchanged",
        "subgoal_decomposition": [],
        "per_subgoal_reachable": [],
        "generic_agent_reached_level": 0,
        "reproduced_levels": 0,
        "generic_first_win_by_config": {
            "explore_budget_800": {
                "variant_attempts": [
                    _attempt("bp35"),
                    _attempt("cd82"),
                    _attempt("dc22"),
                ],
            }
        },
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: poe_world_factored_planner_no_coverage_gain_residual_logged",
        "chosen_submitted_config": "unchanged",
        "candidate_generation_coverage_factored": 0.0,
        "candidate_generation_coverage_flat_baseline": 0.0,
        "coverage_delta": 0.0,
        "first_win_rate_delta": -0.04,
        "solve_rate_delta": 0.0,
        "expert_trust_weights": [
            {
                "game": "ar25",
                "name": "center_color_rewrite",
                "object_class": "color_rewrite",
                "trust": 0.0,
                "heldout_correct": 0,
                "heldout_total": 1,
                "kept": False,
            }
        ],
        "target_arm_results": {
            "candidate_generation_probe": {
                "rows": [
                    {
                        "game": "ar25",
                        "factored_winner_in_pool": False,
                        "flat_winner_in_pool": False,
                        "winner_transition_observed": False,
                    }
                ]
            }
        },
        "offline_reproduced": False,
    }


def test_req_arc_wmte_4680_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4680: OpenSpec declares the persist-transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4680",
        "SCENARIO-ARC-WMTE-4680-PERSIST-STRONGEST-COMPONENT",
        "SCENARIO-ARC-WMTE-4680-TRANSFER-MEASUREMENT",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4680_solver_kit_trust_operator_filters_prefix_overfit() -> None:
    """REQ-ARC-WMTE-4680: the persisted primitive keeps only held-out-stable experts."""

    result = kit.programmatic_expert_trust_weighting_operator(
        [
            {
                "name": "overfit",
                "object_class": "color",
                "heldout_correct": 0,
                "heldout_total": 2,
            },
            {
                "name": "stable",
                "object_class": "color",
                "heldout_correct": 3,
                "heldout_total": 3,
            },
            {"name": "bad", "object_class": "shape", "trust": "not-a-number"},
        ],
        trust_threshold=0.75,
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["expert_count"] == 3
    assert result["kept_expert_count"] == 1
    assert result["coverage_ready"] is True
    assert result["expert_trust_weights"][0]["name"] == "stable"
    assert result["expert_trust_weights"][0]["trust"] == 1.0
    assert result["expert_trust_weights"][1]["kept"] is False

    empty = kit.programmatic_expert_trust_weighting_operator([])
    assert empty["coverage_ready"] is False
    assert empty["residual"] == "expert_factors_not_independent"

    edge = kit.programmatic_expert_trust_weighting_operator(
        [
            {"name": "bool_total", "heldout_correct": True, "heldout_total": True},
            {"name": "bad_total", "heldout_correct": "bad", "heldout_total": "bad"},
        ],
        trust_threshold=True,
    )
    assert edge["trust_threshold"] == 0.75
    assert edge["best_trust"] == 0.0
    assert edge["coverage_ready"] is False


def test_req_arc_wmte_4680_routing_and_registry_surface_persisted_operator() -> None:
    """REQ-ARC-WMTE-4680: routing and registry expose the persisted trust primitive."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "programmatic expert trust" in gotchas[0]["note"]
    assert "latest_exp4680_transfer" in gotchas[0]


def test_scenario_arc_wmte_4680_selects_a2_trust_weighting_when_a1_a2_null() -> None:
    """SCENARIO-ARC-WMTE-4680-PERSIST-STRONGEST-COMPONENT: null A1/A2 persists trust weighting."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert decision["source"] == "A2_programmatic_expert_trust_weighting"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["upstream_signal_rank"][0]["source"] == "A2_programmatic_expert_trust_weighting"
    assert "both A1 and A2 were value-null" in decision["selection_rationale"]

    a1_cleared = dict(
        _a1_artifact(),
        honest_verdict="success: hierarchical_subgoal_generic_agent_new_level_lp85_L2",
        generic_agent_reached_level=2,
        reproduced_levels=1,
    )
    assert (
        mod.select_primitive_from_upstreams(a1_artifact=a1_cleared, a2_artifact=_a2_artifact())[
            "source"
        ]
        == "A1_hierarchical_subgoal_search"
    )

    a2_cleared = dict(
        _a2_artifact(),
        honest_verdict="success: poe_world_factored_planner_coverage_up_live_firstwin_lift_lp85",
        coverage_delta=1.0,
    )
    assert (
        mod.select_primitive_from_upstreams(a1_artifact={}, a2_artifact=a2_cleared)["source"]
        == "A2_poe_world_factored_planner"
    )


def test_scenario_arc_wmte_4680_transfer_measurement_reports_cached_null_and_value() -> None:
    """SCENARIO-ARC-WMTE-4680-TRANSFER-MEASUREMENT: transfer rows report value or null."""

    null = mod.measure_transfer_game(
        "bp35", a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert null["game"] == "bp35"
    assert null["value_added"] is False
    assert null["transfer_value"]["candidate_generation_coverage_delta"] == 0.0
    assert null["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert null["transfer_value"]["first_win_rate_delta"] == 0.0
    assert null["transfer_value"]["offline_reproduced_new_level"] is False
    assert "no cached programmatic-expert transfer rows" in null["dead_end"]

    lifted_a1 = _a1_artifact()
    lifted_a1["generic_first_win_by_config"]["explore_budget_800"]["variant_attempts"] = [
        _attempt("bp35")
    ]
    lifted_a2 = _a2_artifact()
    lifted_a2["expert_trust_weights"].append(
        {
            "game": "bp35",
            "name": "stable_bp35",
            "object_class": "color",
            "trust": 1.0,
            "heldout_correct": 2,
            "heldout_total": 2,
            "kept": True,
        }
    )
    lifted_a2["target_arm_results"]["candidate_generation_probe"]["rows"].append(
        {
            "game": "bp35",
            "factored_winner_in_pool": True,
            "flat_winner_in_pool": False,
            "first_win": True,
            "reached_level": 1,
            "offline_reproduced_new_level": True,
        }
    )
    value = mod.measure_transfer_game("bp35", a1_artifact=lifted_a1, a2_artifact=lifted_a2)

    assert value["value_added"] is True
    assert value["transfer_value"]["candidate_generation_coverage_delta"] == 1.0
    assert value["transfer_value"]["first_win_rate_delta"] == 1.0
    assert value["transfer_value"]["offline_reproduced_new_level"] is True

    clipped_a1 = _a1_artifact()
    clipped_a1["generic_first_win_by_config"]["explore_budget_800"]["variant_attempts"] = [
        _attempt("cd82", first_win=True, reached_level=2)
    ]
    clipped_a2 = _a2_artifact()
    clipped_a2["expert_trust_weights"].append(
        {
            "game": "cd82",
            "name": "stable_cd82",
            "object_class": "color",
            "trust": 1.0,
            "heldout_correct": 1,
            "heldout_total": 1,
        }
    )
    clipped_a2["target_arm_results"]["candidate_generation_probe"]["rows"].append(
        {
            "game": "cd82",
            "factored_winner_in_pool": False,
            "flat_winner_in_pool": True,
            "first_win": False,
            "reached_level": 0,
        }
    )
    clipped = mod.measure_transfer_game("cd82", a1_artifact=clipped_a1, a2_artifact=clipped_a2)
    assert clipped["transfer_value"]["candidate_generation_coverage_delta"] == 0.0
    assert clipped["transfer_value"]["first_win_rate_delta"] == 0.0
    assert clipped["transfer_value"]["live_solve_rate_delta"] == 0.0

    rejected_a2 = _a2_artifact()
    rejected_a2["expert_trust_weights"].append(
        {
            "game": "dc22",
            "name": "overfit_dc22",
            "object_class": "color",
            "trust": 0.0,
            "heldout_correct": 0,
            "heldout_total": 1,
        }
    )
    rejected_a2["target_arm_results"]["candidate_generation_probe"]["rows"].append(
        {
            "game": "dc22",
            "factored_winner_in_pool": False,
            "flat_winner_in_pool": False,
        }
    )
    rejected = mod.measure_transfer_game("dc22", a1_artifact=_a1_artifact(), a2_artifact=rejected_a2)
    assert "failed held-out trust" in rejected["dead_end"]

    no_lift_a2 = _a2_artifact()
    no_lift_a2["expert_trust_weights"].append(
        {
            "game": "dc22",
            "name": "stable_no_lift",
            "object_class": "color",
            "trust": 1.0,
            "heldout_correct": 1,
            "heldout_total": 1,
        }
    )
    no_lift_a2["target_arm_results"]["candidate_generation_probe"]["rows"].append(
        {
            "game": "dc22",
            "factored_winner_in_pool": False,
            "flat_winner_in_pool": False,
        }
    )
    no_lift = mod.measure_transfer_game("dc22", a1_artifact=_a1_artifact(), a2_artifact=no_lift_a2)
    assert "trusted experts produced no coverage" in no_lift["dead_end"]


def test_scenario_arc_wmte_4680_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4680-TRANSFER-MEASUREMENT: artifact schema records transfer value."""

    decision = {
        "source": "A2_programmatic_expert_trust_weighting",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "selection_rationale": "fixture",
    }
    rows = [
        {
            "game": game,
            "value_added": False,
            "transfer_value": {
                "candidate_generation_coverage_delta": 0.0,
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "offline_reproduced_new_level": False,
            },
            "operator_result": {"operator": mod.PRIMITIVE_OPERATOR},
            "dead_end": "null",
        }
        for game in ("bp35", "cd82", "dc22")
    ]

    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["offline_reproduced_new_level"] is False
    assert "programmatic expert trust-weighting" in artifact["residual_dead_end"]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    success_rows = [dict(row) for row in rows]
    success_rows[0] = {
        **success_rows[0],
        "value_added": True,
        "dead_end": "",
        "transfer_value": {
            **success_rows[0]["transfer_value"],
            "candidate_generation_coverage_delta": 1.0,
            "offline_reproduced_new_level": True,
        },
    }
    success = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=success_rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.25,
    )
    assert success["honest_verdict"] == "success: primitive_persisted_transfer_value_characterized"
    assert success["residual_dead_end"] == ""
    assert mod.artifact_schema_errors(success) == []


def test_scenario_arc_wmte_4680_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4680-TRANSFER-MEASUREMENT: run writes a stable artifact."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "general_gotchas": [
                    {
                        "id": mod.PRIMITIVE_GOTCHA_ID,
                        "operator": mod.PRIMITIVE_OPERATOR,
                        "note": "fixture",
                    }
                ],
                "games": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())

    artifact = mod.run(
        tmp_path,
        transfer_games=("bp35", "cd82", "dc22"),
        offline_arcade_checker=lambda: True,
        now=iter([4.0, 4.5]).__next__,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["transfer_games"] == ["bp35", "cd82", "dc22"]
    assert artifact["duration_s"] == 1.0
    assert artifact["preconditions_checked"]["ok"] is True
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4680_defensive_branches_are_schema_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4680: blocked and malformed inputs remain explicit."""

    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
    assert mod._load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert mod._load_json(bad) == {}
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(False) == 0
    assert mod._as_int("bad") == 0
    assert mod._attempt_by_game({"variant_attempts": "bad"}, "bp35") is None
    assert mod._baseline_attempt(_a1_artifact(), "missing") is None
    assert mod._coverage_probe_by_game(_a2_artifact(), "missing") is None
    assert (
        mod._coverage_probe_by_game(
            {"target_arm_results": {"candidate_generation_probe": {"rows": "bad"}}},
            "bp35",
        )
        is None
    )

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]

    blocked = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    malformed = mod.build_artifact(
        selected_upstream={
            "source": "A2_programmatic_expert_trust_weighting",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "aa00",
                "value_added": True,
                "transfer_value": {"offline_reproduced_new_level": True},
                "dead_end": "",
            },
            {
                "game": "bb00",
                "value_added": False,
                "transfer_value": {"offline_reproduced_new_level": False},
                "dead_end": "null",
            },
            {
                "game": "cc00",
                "value_added": False,
                "transfer_value": {"offline_reproduced_new_level": False},
                "dead_end": "null",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert malformed["offline_reproduced_new_level"] is True

    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "bad"
    malformed["verifier_is_oracle"] = True
    malformed["primitive_persisted"] = {}
    malformed["transfer_games"] = []
    malformed["transfer_value_per_game"] = []
    malformed["offline_reproduced_new_level"] = "yes"
    malformed["residual_dead_end"] = []
    malformed["random_seed"] = "bad"
    malformed["registry_updated"] = "yes"
    malformed["field_principles"] = {}
    malformed["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match REQ-ARC-WMTE-4680" in errors
    assert "verifier_is_oracle must be false" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "offline_reproduced_new_level must be a bare bool" in errors
    assert "reproducibility_checksum must match artifact content" in errors

    wrong_gotcha = mod.build_artifact(
        selected_upstream={
            "source": "A2_programmatic_expert_trust_weighting",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": "wrong",
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {"game": game, "value_added": False, "transfer_value": {}, "dead_end": "null"}
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    success_without_value = dict(wrong_gotcha)
    success_without_value["honest_verdict"] = (
        "success: primitive_persisted_transfer_value_characterized"
    )
    success_without_value["primitive_persisted"] = {
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
    }
    success_without_value["reproducibility_checksum"] = mod.payload_checksum(success_without_value)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        success_without_value
    )

    offline_mismatch = dict(success_without_value)
    offline_mismatch["honest_verdict"] = "complete: primitive_persisted_transfer_null_characterized"
    offline_mismatch["offline_reproduced"] = {"new_levels_banked": 2, "new_level_records": []}
    offline_mismatch["reproducibility_checksum"] = mod.payload_checksum(offline_mismatch)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        offline_mismatch
    )

    with pytest.raises(ValueError):
        mod.write_artifact({}, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(tmp_path, offline_arcade_checker=lambda: False, now=iter([1.0, 1.1]).__next__)
