"""Tests for Exp 4656 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4656, SCENARIO-ARC-WMTE-4656.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4656_primitive_persist_transfer as mod
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attempt(
    game: str,
    *,
    policy_mode: str,
    first_win: bool = False,
    solved: bool = False,
    actions: int = 200,
    actions_to_first_levelup: int | None = None,
    reproduced: bool = False,
) -> dict[str, Any]:
    return {
        "game": game,
        "variant_signature": f"{game}~color01",
        "attempted": True,
        "policy_mode": policy_mode,
        "first_win": first_win,
        "solved": solved,
        "actions": actions,
        "actions_to_first_levelup": actions_to_first_levelup,
        "reached_level": 1 if first_win or solved else 0,
        "lazy_value_diagnostics": {
            "enabled": policy_mode == "value_routed",
            "value_head_evals": 4 if policy_mode == "value_routed" else 0,
            "cache_hits": 1 if policy_mode == "value_routed" else 0,
            "cached_frame_hashes": 4 if policy_mode == "value_routed" else 0,
        },
        "reproduction_gate": {
            "game": game,
            "claimed_level": 1 if first_win or solved else 0,
            "reached_level": 1 if reproduced else 0,
            "reproduced": reproduced,
        },
    }


def _a1_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration.",
        "chosen_submitted_config": "unchanged",
        "feature_output_identical_verified": True,
        "per_node_feature_cost_ms": 0.397451,
        "sim_timed_out": False,
        "feature_subset": mod.FEATURE_SUBSET,
        "value_weight_set": 1e-12,
        "solve_rate_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "live_solve_rate_value_routed": 0.0,
        "live_first_win_rate_value_routed": 0.0,
        "value_routed_measurement": {
            "variant_attempts": [
                _attempt("bp35", policy_mode="value_routed", actions=195),
                _attempt("cd82", policy_mode="value_routed", actions=197),
                _attempt("dc22", policy_mode="value_routed", actions=197),
            ]
        },
        "baseline_measurement": {
            "variant_attempts": [
                _attempt("bp35", policy_mode="baseline", actions=195),
                _attempt("cd82", policy_mode="baseline", actions=197),
                _attempt("dc22", policy_mode="baseline", actions=197),
            ]
        },
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: energy_fitness_qd_no_winner_generated_honest_null_gap_sharpened",
        "chosen_submitted_config": "unchanged",
        "winner_generated": False,
        "winner_generated_count": 0,
        "solve_rate_delta": 0.0,
        "first_win_rate_delta": 0.0,
    }


def test_req_arc_wmte_4656_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4656: OpenSpec declares the .429 persist-transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4656",
        "SCENARIO-ARC-WMTE-4656",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4656_solver_kit_cheap_cost_fix_operator_wraps_bridge_ranker() -> None:
    """REQ-ARC-WMTE-4656: cheap cost-fix primitive ranks without oracle leakage."""

    calls: list[str] = []

    def value_head(candidate: Mapping[str, Any]) -> float:
        calls.append(str(candidate["state_key"]))
        return float(candidate["value_score"])

    result = kit.cheap_value_routing_cost_fix_operator(
        [
            {"candidate_id": "noop", "state_key": "a", "value_score": 0.9},
            {
                "candidate_id": "winner",
                "state_key": "b",
                "value_score": 0.1,
                "reaches_levelup": True,
            },
            {
                "candidate_id": "winner_cached",
                "state_key": "b",
                "value_score": 0.1,
                "reaches_levelup": True,
            },
        ],
        value_head=value_head,
        first_win_budget=1,
        per_node_feature_cost_ms=0.397451,
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["base_operator"] == "value_head_bridge_fix_operator"
    assert result["feature_subset"] == mod.FEATURE_SUBSET
    assert result["cost_fix_applied"] is True
    assert result["per_node_feature_cost_ms"] == 0.397451
    assert result["verifier_is_oracle"] is False
    assert result["value_added"] is True
    assert result["first_win_lift"] is True
    assert result["efficiency_lift"] == 1
    assert result["cache_hits"] == 1
    assert calls == ["a", "b"]


def test_req_arc_wmte_4656_routing_and_registry_surface_persisted_operator() -> None:
    """REQ-ARC-WMTE-4656: routing and registry expose the persisted cost fix."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("bp35")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert mod.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "cheap-feature value-routing" in gotchas[0]["note"]
    assert "latest_exp4656_transfer" in gotchas[0]


def test_req_arc_wmte_4656_selects_cheap_feature_cost_fix_when_a1_a2_null() -> None:
    """REQ-ARC-WMTE-4656: all-null A1/A2 persists the strongest characterized component."""

    decision = mod.select_primitive_from_upstreams(a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact())

    assert decision["source"] == "A1_cheap_value_routing_cost_fix"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] == pytest.approx(1.0)
    assert "both A1 and A2 were live-value null" in decision["selection_rationale"]
    assert decision["upstream_signal_rank"][0]["source"] == "A1_cheap_value_routing_cost_fix"

    a1_cleared = dict(_a1_artifact(), honest_verdict="success: value_routing_cost_fixed_live_firstwin_up_1")
    a1_cleared["first_win_rate_delta"] = 0.1
    success_decision = mod.select_primitive_from_upstreams(
        a1_artifact=a1_cleared, a2_artifact=_a2_artifact()
    )
    assert "A1 value-routing cost fix cleared" in success_decision["selection_rationale"]

    a2_cleared = dict(_a2_artifact(), winner_generated=True, winner_generated_count=2)
    a2_decision = mod.select_primitive_from_upstreams(a1_artifact={}, a2_artifact=a2_cleared)
    assert "A2 generated a winner" in a2_decision["selection_rationale"]
    weak_component = mod.upstream_signal_summary(
        a1_artifact={"per_node_feature_cost_ms": 13.0, "sim_timed_out": True},
        a2_artifact={},
    )
    assert weak_component["A1_value_routing_cost_fix_live"]["component_signal"] == 0.0


def test_req_arc_wmte_4656_transfer_measurement_reports_cached_null_and_lift() -> None:
    """REQ-ARC-WMTE-4656: transfer rows report solve/first-win/efficiency deltas."""

    null = mod.measure_transfer_game("bp35", a1_artifact=_a1_artifact())

    assert null["game"] == "bp35"
    assert null["value_added"] is False
    assert null["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert null["transfer_value"]["first_win_rate_delta"] == 0.0
    assert null["transfer_value"]["action_efficiency_lift"] == 0.0
    assert null["transfer_value"]["offline_reproduced_new_level"] is False
    assert "zero solve-rate, first-win, and action-efficiency lift" in null["dead_end"]

    lift_artifact = _a1_artifact()
    lift_artifact["value_routed_measurement"] = {
        "variant_attempts": [
            _attempt(
                "cd82",
                policy_mode="value_routed",
                first_win=True,
                solved=True,
                actions=7,
                actions_to_first_levelup=7,
                reproduced=True,
            )
        ]
    }
    lift_artifact["baseline_measurement"] = {
        "variant_attempts": [
            _attempt("cd82", policy_mode="baseline", first_win=True, actions=11, actions_to_first_levelup=11)
        ]
    }
    lift = mod.measure_transfer_game("cd82", a1_artifact=lift_artifact)

    assert lift["value_added"] is True
    assert lift["transfer_value"]["first_win_rate_delta"] == 0.0
    assert lift["transfer_value"]["action_efficiency_lift"] == 4.0
    assert lift["transfer_value"]["offline_reproduced_new_level"] is False


def test_scenario_arc_wmte_4656_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4656: artifact schema records transfer value or residual null."""

    decision = {
        "source": "A1_cheap_value_routing_cost_fix",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "measured_signal": 1.0,
        "selection_rationale": "fixture",
    }
    rows = [
        {
            "game": "bp35",
            "value_added": False,
            "transfer_value": {
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "action_efficiency_lift": 0.0,
                "offline_reproduced_new_level": False,
            },
            "dead_end": "null",
        },
        {
            "game": "cd82",
            "value_added": False,
            "transfer_value": {
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "action_efficiency_lift": 0.0,
                "offline_reproduced_new_level": False,
            },
            "dead_end": "null",
        },
        {
            "game": "dc22",
            "value_added": False,
            "transfer_value": {
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "action_efficiency_lift": 0.0,
                "offline_reproduced_new_level": False,
            },
            "dead_end": "null",
        },
    ]

    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["offline_reproduced_new_level"] is False
    assert "zero transfer lift" in artifact["residual_dead_end"]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    success_rows = [dict(row) for row in rows]
    success_rows[1] = {
        **success_rows[1],
        "value_added": True,
        "dead_end": "",
        "transfer_value": {
            **success_rows[1]["transfer_value"],
            "action_efficiency_lift": 2.0,
            "value_added": True,
        },
    }
    success = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=success_rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert success["honest_verdict"] == "success: primitive_persisted_transfer_value_characterized"
    assert success["residual_dead_end"] == ""
    assert mod.artifact_schema_errors(success) == []

    errors = mod.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least three games" in errors


def test_scenario_arc_wmte_4656_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4656: run writes a stable three-game transfer artifact."""

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
        now=iter([4.0, 4.25]).__next__,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["transfer_games"] == ["bp35", "cd82", "dc22"]
    assert artifact["duration_s"] == 1.0
    assert artifact["preconditions_checked"]["ok"] is True
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4656_defensive_branches_are_schema_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-4656: blocked and malformed inputs remain explicit."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert mod._load_json(bad) == {}
    not_dict = tmp_path / "list.json"
    not_dict.write_text("[]", encoding="utf-8")
    assert mod._load_json(not_dict) == {}
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(False) == 0
    assert mod._as_int("bad") == 0
    assert mod._attempt_by_game({"variant_attempts": "bad"}, "bp35") is None
    assert mod._attempt_actions(None) is None
    assert mod._attempt_actions({"first_win": True, "actions": 5}) == 5
    assert mod._attempt_reproduced(None) is False

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]

    missing = mod.measure_transfer_game("missing", a1_artifact=_a1_artifact())
    assert "no cached matched value-routed/baseline attempts" in missing["dead_end"]

    blocked = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    banked = mod.build_artifact(
        selected_upstream={
            "source": "A1_cheap_value_routing_cost_fix",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "aa00",
                "value_added": True,
                "transfer_value": {
                    "live_solve_rate_delta": 1.0,
                    "first_win_rate_delta": 1.0,
                    "action_efficiency_lift": 0.0,
                    "offline_reproduced_new_level": True,
                },
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
                "dead_end": "",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert banked["offline_reproduced_new_level"] is True
    assert banked["offline_reproduced"]["new_levels_banked"] == 1

    malformed = dict(banked)
    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "bad"
    malformed["verifier_is_oracle"] = True
    malformed["transfer_value_per_game"] = []
    malformed["offline_reproduced_new_level"] = "yes"
    malformed["residual_dead_end"] = []
    malformed["random_seed"] = "bad"
    malformed["registry_updated"] = "yes"
    malformed["field_principles"] = {}
    malformed["reproducibility_checksum"] = "bad"
    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match REQ-ARC-WMTE-4656" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "transfer_value_per_game must be a mapping" in errors
    assert "offline_reproduced_new_level must be a bare bool" in errors
    assert "residual_dead_end must be a string" in errors
    assert "random_seed must be a bare int" in errors
    assert "registry_updated must be a bare bool" in errors
    assert "field_principles must match REQ-ARC-WMTE-4656" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    wrong_gotcha = dict(banked)
    wrong_gotcha["primitive_persisted"] = dict(banked["primitive_persisted"])
    wrong_gotcha["primitive_persisted"]["registry_general_gotcha_id"] = "wrong"
    wrong_gotcha["reproducibility_checksum"] = mod.payload_checksum(wrong_gotcha)
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    fake_success = dict(banked)
    fake_success["transfer_value_per_game"] = {"aa00": {"value_added": False}}
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

    drifted = dict(banked)
    drifted["duration_s"] = 99.0
    assert "reproducibility_checksum must match artifact content" in mod.artifact_schema_errors(drifted)

    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, root=tmp_path)

    runnable = tmp_path / "runnable"
    runnable.mkdir()
    (runnable / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (runnable / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = runnable / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (runnable / "ops").mkdir()
    (runnable / mod.REGISTRY_RELATIVE_PATH).write_text(
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(runnable / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(runnable / mod.A2_RELATIVE_PATH, _a2_artifact())
    not_written = mod.run(
        runnable,
        transfer_games=("bp35", "cd82", "dc22"),
        offline_arcade_checker=lambda: True,
        now=iter([2.0, 2.1]).__next__,
        write=False,
    )
    assert not_written["preconditions_checked"]["ok"] is True
    assert not (runnable / mod.RESULT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        mod.run(runnable, offline_arcade_checker=lambda: True, write=False)
    monkeypatch.undo()

    monkeypatch.setattr(mod, "run", lambda _root: {"sentinel": True})
    assert mod.main() == 0
    assert json.loads(capsys.readouterr().out)["sentinel"] is True
