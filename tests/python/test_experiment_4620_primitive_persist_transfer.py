"""Tests for Exp 4620 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4620, SCENARIO-ARC-WMTE-4620.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4620_primitive_persist_transfer as exp4620
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4620_spec_declares_bridge_fix_transfer_contract() -> None:
    """REQ-ARC-WMTE-4620: OpenSpec declares the persisted bridge-fix primitive."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4620",
        "SCENARIO-ARC-WMTE-4620",
        exp4620.PRIMITIVE_OPERATOR,
        exp4620.RESULT_RELATIVE_PATH,
        exp4620.PRIMITIVE_GOTCHA_ID,
        "decision-point-only/cached value evaluation",
    ):
        assert marker in spec
    for field, payload in exp4620.FIELD_PRINCIPLES.items():
        assert field in spec
        assert payload["principle"] in spec


def test_req_arc_wmte_4620_solver_kit_bridge_fix_caches_and_lifts_rank() -> None:
    """REQ-ARC-WMTE-4620: bridge fix ranks only decision points and caches state scores."""

    calls: list[str] = []

    def value_head(candidate: Mapping[str, Any]) -> float:
        calls.append(str(candidate["state_key"]))
        return float(candidate["value_score"])

    candidates = [
        {"candidate_id": "noop", "state_key": "s0", "value_score": 0.8},
        {
            "candidate_id": "slow_duplicate",
            "state_key": "win",
            "value_score": 0.1,
            "reaches_levelup": True,
        },
        {
            "candidate_id": "fast_duplicate",
            "state_key": "win",
            "value_score": 0.1,
            "reaches_levelup": True,
        },
        {
            "candidate_id": "non_decision",
            "state_key": "s2",
            "value_score": 0.0,
            "decision_point": False,
        },
    ]

    result = kit.value_head_bridge_fix_operator(
        candidates,
        value_head=value_head,
        max_value_evals=2,
        first_win_budget=1,
    )

    assert result["operator"] == exp4620.PRIMITIVE_OPERATOR
    assert calls == ["s0", "win"]
    assert result["value_head_evals"] == 2
    assert result["cache_hits"] == 1
    assert [row["candidate_id"] for row in result["ranked_candidates"][:3]] == [
        "slow_duplicate",
        "fast_duplicate",
        "noop",
    ]
    assert result["actions_to_first_levelup_before"] == 2
    assert result["actions_to_first_levelup_after"] == 1
    assert result["efficiency_lift"] == 1
    assert result["first_win_lift"] is True
    assert result["value_added"] is True
    assert result["verifier_is_oracle"] is False

    no_target = kit.value_head_bridge_fix_operator([{"candidate_id": "x", "value_score": 0.0}])
    assert no_target["value_added"] is False
    assert no_target["dead_end"] == "no level-up/win candidate in decision set"


def test_req_arc_wmte_4620_routing_and_registry_surface_bridge_fix_operator() -> None:
    """REQ-ARC-WMTE-4620: routing and registry expose the reusable bridge fix."""

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert exp4620.PRIMITIVE_OPERATOR in operators

    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert exp4620.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("bp35")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert exp4620.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == exp4620.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == exp4620.PRIMITIVE_OPERATOR
    assert "decision-point" in gotchas[0]["note"]


def test_req_arc_wmte_4620_selects_a1_bridge_fix_over_a2_live_null() -> None:
    """REQ-ARC-WMTE-4620: A1 bridge fix wins when A2 reports no live value."""

    decision = exp4620.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "success: bridge_cause_isolated_compute_fix_identified",
            "binding_bridge_cause": "compute_cost",
            "indicated_fix": "decision-point-only eval/cached features for live frontier nodes",
            "offline_win_confirmed": True,
            "positive_control_passed": True,
            "compute_cost_evidence": {
                "equal_node_budget": {"value_head_wins": True, "value_head_first_wins": 1},
                "equal_wall_clock": {"value_head_loses": True},
            },
            "diagnostic_corpus": {"games": ["ls20", "cn04", "sk48", "live_25_game_sim"]},
        },
        a2_artifact={
            "honest_verdict": "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened",
            "first_win_rate_graduated": 0.04,
            "first_win_rate_linear_baseline": 0.04,
            "actions_delta": 0.0,
        },
    )

    assert decision["source"] == "A1_bridge_fix_helper"
    assert decision["operator"] == exp4620.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == exp4620.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] > 0.0
    assert decision["source_tuning_games"] == ["cn04", "ls20", "sk48"]


def test_req_arc_wmte_4620_transfer_measurement_reports_efficiency_lift() -> None:
    """REQ-ARC-WMTE-4620: transfer measurement applies the persisted bridge fix."""

    row = exp4620.measure_bridge_fix_transfer_game(
        "bp35",
        source_tuning_games=("cn04", "ls20", "sk48"),
        candidates=[
            {"candidate_id": "baseline_noop", "state_key": "a", "value_score": 0.8},
            {
                "candidate_id": "target",
                "state_key": "b",
                "value_score": 0.1,
                "reaches_levelup": True,
            },
        ],
        first_win_budget=1,
    )

    assert row["game"] == "bp35"
    assert row["not_tuned_on_source"] is True
    assert row["value_added"] is True
    assert row["transfer_value"]["efficiency_lift"] == 1
    assert row["transfer_value"]["first_win_lift"] is True
    assert row["transfer_value"]["offline_reproduced_new_level"] is False

    tuned = exp4620.measure_bridge_fix_transfer_game(
        "cn04",
        source_tuning_games=("cn04",),
        candidates=[{"candidate_id": "target", "value_score": 0.0, "reaches_levelup": True}],
    )
    assert tuned["value_added"] is False
    assert tuned["dead_end"] == "source tuning game excluded from transfer value"


def test_scenario_arc_wmte_4620_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4620: artifact schema distinguishes transfer win and null."""

    decision = {
        "source": "A1_bridge_fix_helper",
        "operator": exp4620.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": exp4620.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": ["cn04", "ls20", "sk48"],
    }
    success = exp4620.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True, "offline_arcade": True},
        transfer_results=[
            {
                "game": "bp35",
                "value_added": True,
                "transfer_value": {
                    "efficiency_lift": 1,
                    "first_win_lift": True,
                    "offline_reproduced_new_level": False,
                },
                "dead_end": "",
            },
            {
                "game": "dc22",
                "value_added": False,
                "transfer_value": {
                    "efficiency_lift": 0,
                    "first_win_lift": False,
                    "offline_reproduced_new_level": False,
                },
                "dead_end": "value-head order matched baseline",
            },
        ],
        registry_updated=True,
        random_seed=exp4620.RANDOM_SEED,
        duration_s=0.1,
    )

    assert success["honest_verdict"] == "success: primitive_persisted_transfer_bp35_value_added"
    assert success["verifier_is_oracle"] is False
    assert success["solve_provenance"] == exp4620.SOLVE_PROVENANCE
    assert success["offline_reproduced"]["new_levels_banked"] == 0
    assert exp4620.artifact_schema_errors(success) == []
    assert (
        json.loads(exp4620.write_artifact(success, root=tmp_path).read_text(encoding="utf-8"))
        == success
    )

    null = exp4620.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "bp35",
                "value_added": False,
                "transfer_value": {"efficiency_lift": 0, "offline_reproduced_new_level": False},
                "dead_end": "value-head order matched baseline",
            },
            {
                "game": "dc22",
                "value_added": False,
                "transfer_value": {"efficiency_lift": 0, "offline_reproduced_new_level": False},
                "dead_end": "no level-up candidate generated",
            },
        ],
        registry_updated=True,
        random_seed=exp4620.RANDOM_SEED,
        duration_s=0.1,
    )
    assert null["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert exp4620.artifact_schema_errors(null) == []

    errors = exp4620.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert f"primitive_persisted must name {exp4620.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least two games" in errors


def test_scenario_arc_wmte_4620_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4620: run writes a stable result from upstream artifacts."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / exp4620.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / exp4620.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "general_gotchas": [
                    {
                        "id": exp4620.PRIMITIVE_GOTCHA_ID,
                        "operator": exp4620.PRIMITIVE_OPERATOR,
                        "note": "fixture",
                    }
                ],
                "games": [],
                "reproducible_total_levels": 55,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(
        tmp_path / exp4620.A1_RELATIVE_PATH,
        {
            "honest_verdict": "success: bridge_cause_isolated_compute_fix_identified",
            "binding_bridge_cause": "compute_cost",
            "indicated_fix": "decision-point-only eval/cached features",
            "offline_win_confirmed": True,
            "positive_control_passed": True,
            "diagnostic_corpus": {"games": ["ls20", "cn04", "sk48"]},
            "compute_cost_evidence": {
                "equal_node_budget": {"value_head_wins": True},
                "equal_wall_clock": {"value_head_loses": True},
            },
        },
    )
    _write_json(
        tmp_path / exp4620.A2_RELATIVE_PATH,
        {
            "honest_verdict": "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened",
            "first_win_rate_graduated": 0.04,
            "first_win_rate_linear_baseline": 0.04,
            "actions_delta": 0.0,
        },
    )

    artifact = exp4620.run(
        tmp_path,
        offline_arcade_checker=lambda: True,
        now=iter([10.0, 10.25]).__next__,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["transfer_games"] == ["bp35", "dc22", "g50t"]
    assert artifact["preconditions_checked"]["ok"] is True
    assert (tmp_path / exp4620.RESULT_RELATIVE_PATH).exists()

    blocked = exp4620.build_artifact(
        selected_upstream={
            "operator": exp4620.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": exp4620.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=exp4620.RANDOM_SEED,
        duration_s=None,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert exp4620.artifact_schema_errors(blocked) == []

    with pytest.raises(ValueError, match="missing required field"):
        exp4620.write_artifact({}, root=tmp_path)


def test_req_arc_wmte_4620_defensive_branches_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-4620: defensive branches stay honest and deterministic."""

    assert exp4620._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp4620._load_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp4620._load_json(list_json) == {}

    assert exp4620._load_registry(tmp_path) == {}
    registry_path = tmp_path / exp4620.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text("[", encoding="utf-8")
    assert exp4620._load_registry(tmp_path) == {}
    assert exp4620._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert exp4620._as_float(True) == 0.0
    assert exp4620._as_float("bad") == 0.0
    assert exp4620._as_int(True) == 0
    assert exp4620._as_int("bad") == 0

    checks = exp4620.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert checks["offline_arcade"] is False
    assert checks["ok"] is False
    blocked = exp4620.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"

    a2_selected = exp4620.select_primitive_from_upstreams(
        a1_artifact={},
        a2_artifact={"first_win_rate_graduated": 0.2, "first_win_rate_linear_baseline": 0.0},
    )
    assert "A2 has the larger live metric" in a2_selected["selection_rationale"]
    null_selected = exp4620.select_primitive_from_upstreams(a1_artifact={}, a2_artifact={})
    assert "All upstreams were value-null" in null_selected["selection_rationale"]
    summary = exp4620.upstream_signal_summary(
        a1_artifact={"compute_cost_evidence": "bad", "diagnostic_corpus": {"games": ["live_x"]}},
        a2_artifact={"actions_delta": True},
    )
    assert summary["A1_bridge_fix_helper"]["source_tuning_games"] == []
    assert summary["A2_graduated_spatial_value_head"]["actions_delta"] == 0.0

    artifact = exp4620.build_artifact(
        selected_upstream={
            "operator": exp4620.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": exp4620.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": False,
                "transfer_value": {
                    "offline_reproduced_new_level": True,
                    "existing_reproduced_level": 1,
                },
                "dead_end": "banked by reproduce",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {"offline_reproduced_new_level": False},
                "dead_end": "",
            },
        ],
        registry_updated=True,
        random_seed=exp4620.RANDOM_SEED,
        duration_s=None,
    )
    assert artifact["offline_reproduced"]["new_levels_banked"] == 1
    assert artifact["duration_s"] is None

    malformed = dict(artifact)
    malformed["honest_verdict"] = "bad"
    malformed["verifier_is_oracle"] = True
    malformed["solve_provenance"] = "outer_loop_re"
    malformed["transfer_value_per_game"] = []
    malformed["reproducibility_checksum"] = "bad"
    errors = exp4620.artifact_schema_errors(malformed)
    assert "honest_verdict must use a terminal prefix" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "solve_provenance must be development_proxy or live_agent_self_discovery" in errors
    assert "transfer_value_per_game must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    drifted = dict(artifact)
    drifted["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in exp4620.artifact_schema_errors(
        drifted
    )

    monkeypatch.setattr(exp4620, "run", lambda _root: {"sentinel": True})
    assert exp4620.main() == 0
    assert json.loads(capsys.readouterr().out)["sentinel"] is True
