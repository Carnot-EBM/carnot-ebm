"""Tests for Exp4644 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4644, SCENARIO-ARC-WMTE-4644.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4644_primitive_persist_transfer as mod
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit
from carnot.agentic.arc_goal_energy_live import GoalSatisfactionEnergy


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"

PREDICATE_CODE = 'def is_goal(state):\n    return state["unsatisfied_targets"] == 0\n'


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attempt(game: str, *, solved: bool, actions: int = 200) -> dict[str, Any]:
    return {
        "game": game,
        "variant_signature": f"{game}~color01",
        "attempted": True,
        "solved": solved,
        "first_win": solved,
        "actions": actions,
        "actions_to_first_levelup": actions if solved else None,
        "reproduction_gate": {
            "game": game,
            "claimed_level": 1 if solved else 0,
            "reached_level": 1 if solved else 0,
            "reproduced": solved,
        },
    }


def _a1_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: goal_energy_no_live_lift_honest_null_gap_sharpened",
        "live_solve_rate_goal_energy": 0.04,
        "live_solve_rate_baseline": 0.04,
        "solve_rate_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "median_actions_to_win_delta": 0.0,
        "offline_reproduced": True,
        "goal_energy_measurement": {
            "variant_attempts": [_attempt("bp35", solved=False), _attempt("dc22", solved=False)]
        },
        "baseline_measurement": {
            "variant_attempts": [_attempt("bp35", solved=False), _attempt("dc22", solved=False)]
        },
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened",
        "live_solve_rate_expansion": 0.0,
        "live_solve_rate_ranker_baseline": 0.0,
        "solve_rate_delta": 0.0,
        "depth_of_live_solve_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "offline_reproduced": True,
    }


def test_req_arc_wmte_4644_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4644: OpenSpec declares the primitive-transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4644",
        "SCENARIO-ARC-WMTE-4644",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4644_solver_kit_goal_energy_operator_ranks_without_oracle() -> None:
    """REQ-ARC-WMTE-4644: the persisted operator improves rank by graded energy."""

    goal_energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    result = kit.graded_goal_energy_search_heuristic_operator(
        [
            {
                "candidate_id": "flat",
                "navigation_energy": 0.0,
                "goal_state": {"total_targets": 2, "satisfied_targets": 0, "unsatisfied_targets": 2},
                "reaches_goal": False,
            },
            {
                "candidate_id": "winner",
                "navigation_energy": 0.0,
                "goal_state": {"total_targets": 2, "satisfied_targets": 2, "unsatisfied_targets": 0},
                "reaches_goal": True,
            },
        ],
        goal_energy=goal_energy,
        alpha=0.5,
        beta=0.5,
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["best_candidate_id"] == "winner"
    assert result["actions_to_first_goal_before"] == 2
    assert result["actions_to_first_goal_after"] == 1
    assert result["action_efficiency_lift"] == 1.0
    assert result["value_added"] is True
    assert result["ranked_candidates"][0]["goal_predicate_pass"] is True


def test_req_arc_wmte_4644_routing_and_registry_surface_goal_energy_operator() -> None:
    """REQ-ARC-WMTE-4644: routing and registry expose the persisted primitive."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="keyboard")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("ka59")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert mod.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "goal-energy" in gotchas[0]["note"].lower()
    assert "latest_exp4644_transfer" in gotchas[0]


def test_req_arc_wmte_4644_selects_a1_best_characterized_null() -> None:
    """REQ-ARC-WMTE-4644: all-null upstreams persist the best-characterized A1 primitive."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        source_tuning_games=("r11l",),
    )

    assert decision["source"] == "A1_goal_energy_heuristic"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] == 0.0
    assert decision["source_tuning_games"] == ["r11l"]
    assert "best-characterized" in decision["selection_rationale"]


def test_req_arc_wmte_4644_transfer_measurement_reports_null_and_lift() -> None:
    """REQ-ARC-WMTE-4644: transfer reports cached deltas per untuned game."""

    null = mod.measure_goal_energy_transfer_game(
        "bp35",
        a1_artifact=_a1_artifact(),
        source_tuning_games=("r11l",),
    )
    assert null["game"] == "bp35"
    assert null["not_tuned_on_source"] is True
    assert null["value_added"] is False
    assert null["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert null["transfer_value"]["first_win_rate_delta"] == 0.0
    assert "zero solve/first-win/action-efficiency lift" in null["dead_end"]

    lift_artifact = _a1_artifact()
    lift_artifact["goal_energy_measurement"] = {
        "variant_attempts": [_attempt("cd82", solved=True, actions=4)]
    }
    lift_artifact["baseline_measurement"] = {
        "variant_attempts": [_attempt("cd82", solved=True, actions=7)]
    }
    lift = mod.measure_goal_energy_transfer_game(
        "cd82",
        a1_artifact=lift_artifact,
        source_tuning_games=("r11l",),
    )
    assert lift["value_added"] is True
    assert lift["transfer_value"]["action_efficiency_lift"] == 3.0
    assert lift["transfer_value"]["offline_reproduced_new_level"] is False


def test_scenario_arc_wmte_4644_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4644: artifact schema records transfer value or honest null."""

    decision = {
        "source": "A1_goal_energy_heuristic",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": ["r11l"],
        "selection_rationale": "A1 was best-characterized.",
    }
    transfer_results = [
        {
            "game": "cd82",
            "value_added": True,
            "transfer_value": {
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "action_efficiency_lift": 3.0,
                "offline_reproduced_new_level": False,
                "existing_reproduced_level": 1,
                "value_added": True,
            },
            "dead_end": "",
        },
        {
            "game": "bp35",
            "value_added": False,
            "transfer_value": {
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "action_efficiency_lift": 0.0,
                "offline_reproduced_new_level": False,
                "existing_reproduced_level": 0,
                "value_added": False,
            },
            "dead_end": "no transfer",
        },
    ]

    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={"A1_goal_energy_heuristic": {"measured_signal": 0.0}},
        preconditions_checked={"ok": True},
        transfer_results=transfer_results,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success: primitive_persisted_transfer_cd82_value_added"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
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


def test_scenario_arc_wmte_4644_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4644: run writes the requested result JSON."""

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
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())
    _write_json(
        tmp_path / mod.EXP4020_RELATIVE_PATH,
        {"game": "r11l", "goal_predicate_code": PREDICATE_CODE},
    )

    artifact = mod.run(
        tmp_path,
        transfer_games=("bp35", "dc22"),
        offline_arcade_checker=lambda: True,
        now=iter([5.0, 5.25]).__next__,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["transfer_games"] == ["bp35", "dc22"]
    assert artifact["primitive_persisted"]["source_tuning_games"] == ["r11l"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4644_defensive_branches_are_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4644: defensive branches remain explicit and schema-gated."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_json) == {}
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": {}}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_int(False) == 0
    assert mod._as_int(object()) == 0

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]
    assert mod._source_tuning_games(tmp_path, {"source_tuning_games": ["g2", "g1", "g1"]}) == ["g1", "g2"]
    assert mod._goal_energy_for_root(tmp_path) is None

    a1_wins = mod.select_primitive_from_upstreams(
        a1_artifact={"solve_rate_delta": 0.25},
        a2_artifact={"depth_of_live_solve_delta": 0.0},
        source_tuning_games=("r11l",),
    )
    assert "strongest clean" in a1_wins["selection_rationale"]
    a2_wins = mod.select_primitive_from_upstreams(
        a1_artifact={"solve_rate_delta": 0.0},
        a2_artifact={"depth_of_live_solve_delta": 0.5},
        source_tuning_games=("r11l",),
    )
    assert "A2 had the larger" in a2_wins["selection_rationale"]
    assert mod._attempt_actions({"solved": True}) is None

    source_game = mod.measure_goal_energy_transfer_game(
        "r11l",
        a1_artifact=_a1_artifact(),
        source_tuning_games=("r11l",),
    )
    assert source_game["value_added"] is False
    assert "source tuning game" in source_game["dead_end"]

    no_row = mod.measure_goal_energy_transfer_game(
        "missing",
        a1_artifact=_a1_artifact(),
        source_tuning_games=("r11l",),
    )
    assert "no cached matched variant" in no_row["dead_end"]

    first_win_artifact = _a1_artifact()
    first_win_artifact["goal_energy_measurement"] = {
        "variant_attempts": [_attempt("wa30", solved=True, actions=5)]
    }
    first_win_artifact["baseline_measurement"] = {
        "variant_attempts": [_attempt("wa30", solved=False)]
    }
    first_win = mod.measure_goal_energy_transfer_game(
        "wa30",
        a1_artifact=first_win_artifact,
        source_tuning_games=("r11l",),
    )
    assert first_win["value_added"] is True
    assert first_win["operator_probe"]["value_added"] is True

    artifact = mod.build_artifact(
        selected_upstream={
            "source": "A1_goal_energy_heuristic",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
            "source_tuning_games": ["r11l"],
        },
        upstream_signals={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert artifact["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(artifact) == []

    banked = mod.build_artifact(
        selected_upstream={
            "source": "A1_goal_energy_heuristic",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
            "source_tuning_games": ["r11l"],
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "wa30",
                "value_added": True,
                "transfer_value": {
                    "live_solve_rate_delta": 1.0,
                    "first_win_rate_delta": 1.0,
                    "action_efficiency_lift": 0.0,
                    "offline_reproduced_new_level": True,
                    "existing_reproduced_level": 1,
                    "value_added": True,
                },
                "dead_end": "",
            },
            {
                "game": "bp35",
                "value_added": False,
                "transfer_value": {"value_added": False},
                "dead_end": "no transfer",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert banked["offline_reproduced"]["new_levels_banked"] == 1

    wrong_gotcha = dict(banked)
    wrong_gotcha["primitive_persisted"] = dict(banked["primitive_persisted"])
    wrong_gotcha["primitive_persisted"]["registry_general_gotcha_id"] = "wrong"
    wrong_gotcha["reproducibility_checksum"] = mod.payload_checksum(wrong_gotcha)
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    fake_success = dict(banked)
    fake_success["transfer_value_per_game"] = {"wa30": {"value_added": False}}
    fake_success["reproducibility_checksum"] = mod.payload_checksum(fake_success)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        fake_success
    )

    bank_mismatch = dict(banked)
    bank_mismatch["offline_reproduced"] = dict(banked["offline_reproduced"])
    bank_mismatch["offline_reproduced"]["new_level_records"] = []
    bank_mismatch["reproducibility_checksum"] = mod.payload_checksum(bank_mismatch)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        bank_mismatch
    )

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum must match artifact content" in mod.artifact_schema_errors(
        bad_checksum
    )

    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        mod.run(tmp_path, offline_arcade_checker=lambda: True, write=False)
