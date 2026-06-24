"""Tests for Exp 4668 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4668, SCENARIO-ARC-WMTE-4668.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4668_primitive_persist_transfer as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_label(action: int, data: Any = None) -> str:
    return json.dumps({"action": action, "data": data}, sort_keys=True, separators=(",", ":"))


def _attempt(
    game: str,
    *,
    policy_mode: str,
    first_win: bool = False,
    solved: bool = False,
    actions: int = 200,
    actions_to_first_levelup: int | None = None,
    reached_level: int | None = None,
    reproduced: bool = False,
) -> dict[str, Any]:
    level = int(reached_level if reached_level is not None else (1 if first_win or solved else 0))
    return {
        "game": game,
        "variant_signature": f"{game}~color01",
        "attempted": True,
        "policy_mode": policy_mode,
        "first_win": first_win,
        "solved": solved,
        "actions": actions,
        "actions_to_first_levelup": actions_to_first_levelup,
        "reached_level": level,
        "lazy_value_diagnostics": {
            "enabled": policy_mode in {"value_routed", "dagger_corrected"},
            "value_head_evals": 4 if policy_mode in {"value_routed", "dagger_corrected"} else 0,
            "cache_hits": 1 if policy_mode in {"value_routed", "dagger_corrected"} else 0,
        },
        "reproduction_gate": {
            "game": game,
            "claimed_level": level,
            "reached_level": level if reproduced else 0,
            "reproduced": reproduced,
        },
    }


def _a1_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: l2_goal_induction_no_deepening_residual_single_exemplar_goal_insufficient",
        "generic_agent_reached_level": {"lp85": 1, "sc25": 0},
        "goal_predicate_satisfiable": {"lp85": False, "sc25": False},
        "l2_plan_reaches_goal": {"lp85": False, "sc25": False},
        "bare_control_passed": False,
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: dagger_distribution_corrected_no_live_lift_residual_logged.",
        "chosen_submitted_config": "unchanged",
        "distribution_shift_score_before": 0.699108,
        "distribution_shift_score_after": 0.0,
        "shift_score_delta": -0.699108,
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
        "dagger_dataset": {
            "frontier_count": 1042,
            "negative_count": 1041,
            "positive_count": 595,
            "total_count": 1636,
            "winning_path_count": 594,
        },
        "distribution_shift_probe_after": {
            "distribution_shift_score": 0.0,
            "frontier_count": 1042,
            "aggregate_reference_count": 1042,
            "method": "frontier_vs_aggregated_search_distribution_score_gap",
        },
        "corrected_measurement": {
            "variant_attempts": [
                _attempt("bp35", policy_mode="dagger_corrected", actions=195),
                _attempt("cd82", policy_mode="dagger_corrected", actions=197),
                _attempt("dc22", policy_mode="dagger_corrected", actions=197),
            ]
        },
        "baseline_measurement": {
            "variant_attempts": [
                _attempt("bp35", policy_mode="value_routed", actions=195),
                _attempt("cd82", policy_mode="value_routed", actions=197),
                _attempt("dc22", policy_mode="value_routed", actions=197),
            ]
        },
        "live_baseline_winning_path_trained": {
            "measurement": {
                "variant_attempts": [
                    _attempt("bp35", policy_mode="value_routed", actions=195),
                    _attempt("cd82", policy_mode="value_routed", actions=197),
                    _attempt("dc22", policy_mode="value_routed", actions=197),
                ]
            }
        },
    }


def test_req_arc_wmte_4668_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4668: OpenSpec declares the .430 persist-transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4668",
        "SCENARIO-ARC-WMTE-4668",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4668_solver_kit_dagger_operator_relabels_off_path_rows() -> None:
    """REQ-ARC-WMTE-4668: DAgger primitive collects off-path rows without oracle inference."""

    result = kit.dagger_off_path_data_collection_operator(
        [
            {
                "source": "live_frontier",
                "features": [0.1, 0.2],
                "path": [{"action": 1, "data": None}],
            },
            {
                "source": "live_frontier",
                "features": [4.0, 5.0],
                "path": [{"action": 9, "data": {"x": 1}}],
            },
            {"source": "empty_features", "features": [], "path": []},
        ],
        winning_labels=[_path_label(1), _path_label(2)],
        winning_rows=[{"source": "winning_path", "features": [0.0, 0.0], "label": 1.0}],
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["frontier_count"] == 3
    assert result["relabeled_frontier_count"] == 2
    assert result["aggregate_total_count"] == 3
    assert result["positive_count"] == 2
    assert result["negative_count"] == 1
    assert result["off_path_negative_count"] == 1
    assert result["rows"][0]["label"] == 1.0
    assert result["rows"][1]["label"] == 0.0
    assert result["rows"][1]["path"] == [{"action": 9, "data": {"x": 1}}]


def test_req_arc_wmte_4668_routing_and_registry_surface_persisted_operator() -> None:
    """REQ-ARC-WMTE-4668: routing and registry expose the persisted DAgger primitive."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "DAgger-lite off-path" in gotchas[0]["note"]
    assert "latest_exp4668_transfer" in gotchas[0]


def test_req_arc_wmte_4668_selects_dagger_component_when_a1_a2_null() -> None:
    """REQ-ARC-WMTE-4668: all-null A1/A2 persists the strongest characterized component."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert decision["source"] == "A2_dagger_off_path_data_collection"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] == pytest.approx(0.699108)
    assert "both A1 and A2 were value-null" in decision["selection_rationale"]
    assert decision["upstream_signal_rank"][0]["source"] == "A2_dagger_off_path_data_collection"

    a1_cleared = dict(
        _a1_artifact(),
        honest_verdict="success: l2_goal_induction_generic_agent_reached_L2_lp85",
        generic_agent_reached_level={"lp85": 2},
    )
    a1_decision = mod.select_primitive_from_upstreams(
        a1_artifact=a1_cleared, a2_artifact=_a2_artifact()
    )
    assert a1_decision["source"] == "A1_l2_goal_induction"
    assert "A1 L2 goal induction cleared" in a1_decision["selection_rationale"]

    a2_cleared = dict(
        _a2_artifact(),
        honest_verdict="success: dagger_distribution_shift_value_routing_live_firstwin_up_1",
        first_win_rate_delta=0.1,
    )
    a2_decision = mod.select_primitive_from_upstreams(a1_artifact={}, a2_artifact=a2_cleared)
    assert a2_decision["source"] == "A2_dagger_distribution_corrected_value_routing"
    assert "A2 DAgger-corrected value routing cleared" in a2_decision["selection_rationale"]


def test_req_arc_wmte_4668_transfer_measurement_reports_cached_null_and_lift() -> None:
    """REQ-ARC-WMTE-4668: transfer rows report solve/first-win/efficiency deltas."""

    null = mod.measure_transfer_game("bp35", a2_artifact=_a2_artifact())

    assert null["game"] == "bp35"
    assert null["value_added"] is False
    assert null["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert null["transfer_value"]["first_win_rate_delta"] == 0.0
    assert null["transfer_value"]["action_efficiency_lift"] == 0.0
    assert null["transfer_value"]["offline_reproduced_new_level"] is False
    assert "zero solve-rate, first-win, and action-efficiency lift" in null["dead_end"]

    lift_artifact = _a2_artifact()
    lift_artifact["corrected_measurement"] = {
        "variant_attempts": [
            _attempt(
                "cd82",
                policy_mode="dagger_corrected",
                first_win=True,
                solved=True,
                actions=7,
                actions_to_first_levelup=7,
                reached_level=1,
                reproduced=True,
            )
        ]
    }
    lift_artifact["baseline_measurement"] = {
        "variant_attempts": [
            _attempt(
                "cd82",
                policy_mode="value_routed",
                first_win=True,
                actions=11,
                actions_to_first_levelup=11,
                reached_level=1,
                reproduced=True,
            )
        ]
    }
    lift = mod.measure_transfer_game("cd82", a2_artifact=lift_artifact)

    assert lift["value_added"] is True
    assert lift["transfer_value"]["first_win_rate_delta"] == 0.0
    assert lift["transfer_value"]["action_efficiency_lift"] == 4.0
    assert lift["transfer_value"]["offline_reproduced_new_level"] is False


def test_scenario_arc_wmte_4668_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4668: artifact schema records transfer value or residual null."""

    decision = {
        "source": "A2_dagger_off_path_data_collection",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "measured_signal": 0.699108,
        "selection_rationale": "fixture",
    }
    rows = [
        {
            "game": game,
            "value_added": False,
            "transfer_value": {
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "action_efficiency_lift": 0.0,
                "offline_reproduced_new_level": False,
            },
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


def test_scenario_arc_wmte_4668_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4668: run writes a stable three-game transfer artifact."""

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


def test_req_arc_wmte_4668_defensive_branches_are_schema_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4668: blocked and malformed inputs remain explicit."""

    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
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
    assert mod._measurement_from_a2(
        {"live_baseline_winning_path_trained": {"measurement": {"variant_attempts": []}}},
        "baseline_measurement",
    ) == {"variant_attempts": []}
    assert mod._measurement_from_a2({}, "corrected_measurement") == {}

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]

    missing = mod.measure_transfer_game("missing", a2_artifact=_a2_artifact())
    assert "no cached matched corrected/baseline attempts" in missing["dead_end"]

    blocked = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    malformed = mod.build_artifact(
        selected_upstream={
            "source": "A2_dagger_off_path_data_collection",
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
    assert malformed["offline_reproduced_new_level"] is True
    assert malformed["offline_reproduced"]["new_levels_banked"] == 1

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
    assert "inference_substrate must match REQ-ARC-WMTE-4668" in errors
    assert "verifier_is_oracle must be false" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "offline_reproduced_new_level must be a bare bool" in errors
    assert "reproducibility_checksum must match artifact content" in errors

    malformed_gotcha = mod.build_artifact(
        selected_upstream={
            "source": "A2_dagger_off_path_data_collection",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": "wrong",
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {"game": game, "value_added": False, "transfer_value": {}, "dead_end": ""}
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        malformed_gotcha
    )

    success_without_value = dict(malformed_gotcha)
    success_without_value["honest_verdict"] = "success: primitive_persisted_transfer_value_characterized"
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

    bad_checksum_prefix = dict(offline_mismatch)
    bad_checksum_prefix["reproducibility_checksum"] = "bad"
    assert "reproducibility_checksum must be sha256-prefixed" in mod.artifact_schema_errors(
        bad_checksum_prefix
    )

    with pytest.raises(ValueError):
        mod.write_artifact({}, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(tmp_path, offline_arcade_checker=lambda: False, now=iter([1.0, 1.1]).__next__)
