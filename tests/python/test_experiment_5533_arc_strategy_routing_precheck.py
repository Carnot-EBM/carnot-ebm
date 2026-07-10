"""Tests for Exp5533 ARC strategy-routing precheck.

Spec refs: REQ-ARC-FCP-5533,
SCENARIO-ARC-FCP-5533.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from carnot import experiment_5533_arc_strategy_routing_precheck as exp5533
from carnot.agentic import arc_bounded_strategy_router as bounded_router
from carnot.agentic.arc_strategy_router import BoundedStrategyCandidateRouter


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, sb26_levels: int = 2, g50t_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": sb26_levels},
            {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": g50t_levels},
            {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _exp5520() -> dict[str, Any]:
    return {
        "experiment": "experiment_5520_arc_action_diversity_target_precheck",
        "selected_game": "sb26",
        "selected_level": "L3",
        "arc_levelup_candidate_ready": True,
        "solve_provenance": "live_agent_self_discovery",
    }


def _exp5521_stale() -> dict[str, Any]:
    return {
        "experiment": "experiment_5521_arc_live_action_diverse_levelup",
        "selected_game": "sb26",
        "selected_level": "L3",
        "selected_target_level": 3,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "banking_gate": False,
        "repeated_coordinate_rate": 0.5263157894736842,
        "honest_verdict": (
            "honest_null: sb26 L3 bounded_budget_no_target_level_reproduction; "
            "entropy=2.481; repeat_rate=0.526; registry_delta=0"
        ),
    }


def _candidate(
    label: str,
    x: int,
    y: int,
    *,
    salience: float = 0.0,
    effect: float = 0.0,
    verifier: float = 0.0,
    reset: float = 0.0,
) -> dict[str, Any]:
    return {
        "label": label,
        "action": 6,
        "data": {"x": x, "y": y},
        "salience_score": float(salience),
        "effect_score": float(effect),
        "verifier_score": float(verifier),
        "reset_score": float(reset),
    }


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5533.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5533\nSCENARIO-ARC-FCP-5533\n",
        encoding="utf-8",
    )
    (root / exp5533.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5533.EXP5520_RELATIVE_PATH).write_text(
        json.dumps(_exp5520()),
        encoding="utf-8",
    )
    (root / exp5533.EXP5521_RELATIVE_PATH).write_text(
        json.dumps(_exp5521_stale()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5533_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5533: OpenSpec anchors the precheck artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5533" in spec
    assert "SCENARIO-ARC-FCP-5533" in spec
    assert exp5533.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5533.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5533_router_suppresses_repeats_before_selection() -> None:
    """SCENARIO-ARC-FCP-5533: suppression changes actual routed candidate order."""

    candidates = [
        _candidate("top-salience", 10, 10, salience=10.0, effect=1.0, verifier=1.0),
        _candidate("top-effect-same-coord", 10, 10, salience=9.0, effect=10.0, verifier=2.0),
        _candidate("top-verifier-same-coord", 10, 10, salience=8.0, effect=2.0, verifier=10.0),
        _candidate("effect-fallback", 14, 14, salience=7.0, effect=8.0, verifier=3.0),
        _candidate("verifier-fallback", 20, 20, salience=6.0, effect=3.0, verifier=8.0),
        _candidate("reset-fallback", 25, 25, salience=5.0, effect=4.0, verifier=4.0, reset=9.0),
    ]
    unsuppressed = BoundedStrategyCandidateRouter(
        max_candidates=4,
        per_strategy_limit=1,
        suppress_repeated_coordinates=False,
    ).rank(None, candidates)
    router = BoundedStrategyCandidateRouter(
        max_candidates=4,
        per_strategy_limit=1,
        suppress_repeated_coordinates=True,
    )
    suppressed = router.rank(None, candidates)
    diagnostics = router.last_diagnostics

    assert [row["data"] for row in unsuppressed[:3]].count({"x": 10, "y": 10}) == 3
    assert [row["data"] for row in suppressed[:4]] == [
        {"x": 10, "y": 10},
        {"x": 14, "y": 14},
        {"x": 20, "y": 20},
        {"x": 25, "y": 25},
    ]
    assert diagnostics["selection_changed_by_suppression"] is True
    assert diagnostics["suppressed_coordinate_count"] >= 2
    assert diagnostics["selected_strategy_count"] >= 3


def test_scenario_arc_fcp_5533_selects_non_duplicate_rotated_target() -> None:
    """SCENARIO-ARC-FCP-5533: stale Exp5521 target is rejected before readiness."""

    artifact = exp5533.build_precheck(
        exp5533.StrategyPrecheckEvidence(
            registry=_registry(),
            exp5520=_exp5520(),
            exp5521=_exp5521_stale(),
        ),
        tests_run=["unit 5533"],
        duration_s=0.1,
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )

    exp5533.validate_artifact(artifact)
    assert artifact["registry_precheck_passed"] is True
    assert artifact["selected_game"] == "g50t"
    assert artifact["selected_level"] == "L3"
    assert artifact["already_reproduced"] is False
    assert artifact["strategy_routing_live_path_reachable"] is True
    assert artifact["repeated_coordinate_suppression_enabled"] is True
    assert artifact["repeated_coordinate_rate_precheck"] == pytest.approx(0.0)
    assert artifact["action_entropy_precheck"] >= 2.0
    assert artifact["salience_coverage_rate_precheck"] == pytest.approx(1.0)
    assert len(artifact["strategy_portfolio"]) >= 3
    assert artifact["llm_strategy_proposer_used"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["inference_substrate"] == "arc_live_path_precheck_no_solve_claim"
    assert artifact["arc_sge_candidate_ready"] is True
    assert artifact["target_audit"]["sb26:L3"]["decision"] == (
        "rejected_stale_exp5521_repeated_coordinate"
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no solve claimed" in artifact["honest_verdict"]


def test_req_arc_fcp_5533_schema_rejects_bad_required_fields() -> None:
    """REQ-ARC-FCP-5533: schema rejects malformed readiness and solve-credit drift."""

    artifact = exp5533.build_precheck(
        exp5533.StrategyPrecheckEvidence(
            registry=_registry(),
            exp5520=_exp5520(),
            exp5521=_exp5521_stale(),
        ),
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    invalid = {
        **artifact,
        "selected_game": 7,
        "selected_level": [],
        "already_reproduced": True,
        "registry_precheck_passed": "true",
        "strategy_portfolio": [],
        "strategy_routing_live_path_reachable": "true",
        "repeated_coordinate_suppression_enabled": "true",
        "repeated_coordinate_rate_precheck": 1.5,
        "action_entropy_precheck": "2.0",
        "salience_coverage_rate_precheck": -0.1,
        "model_specs": "none",
        "llm_strategy_proposer_used": "false",
        "solve_provenance": "development_proxy",
        "arc_sge_candidate_ready": True,
        "tests_added_or_reused": "unit",
        "field_principles": [],
        "inference_substrate": "offline_bfs",
        "honest_verdict": "solved g50t",
    }

    errors = exp5533.artifact_schema_errors(invalid)

    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "ready artifacts require already_reproduced false" in errors
    assert "registry_precheck_passed must be bare bool" in errors
    assert "strategy_portfolio must contain at least three strategies" in errors
    assert "strategy_routing_live_path_reachable must be bare bool" in errors
    assert "repeated_coordinate_suppression_enabled must be bare bool" in errors
    assert "repeated_coordinate_rate_precheck must be in [0, 1]" in errors
    assert "action_entropy_precheck must be bare float" in errors
    assert "salience_coverage_rate_precheck must be in [0, 1]" in errors
    assert "model_specs must be a list" in errors
    assert "llm_strategy_proposer_used must be bare bool" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "tests_added_or_reused must be a non-empty list" in errors
    assert "field_principles must be a mapping" in errors
    assert "inference_substrate must be arc_live_path_precheck_no_solve_claim" in errors
    assert "honest_verdict must start with complete: or blocked:" in errors
    assert "honest_verdict must not claim a solve" in errors
    with pytest.raises(ValueError):
        exp5533.validate_artifact(invalid)


def test_req_arc_fcp_5533_helper_fallbacks_and_blockers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-FCP-5533: defensive helpers fail closed without solve credit."""

    artifact = exp5533.build_precheck(
        exp5533.StrategyPrecheckEvidence(
            registry=_registry(),
            exp5520=_exp5520(),
            exp5521=_exp5521_stale(),
        ),
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    blocked_no_depth = exp5533.build_precheck(
        exp5533.StrategyPrecheckEvidence(
            registry={"games": [{"game": "zero", "levels_reproduced": 0}]},
            exp5520={},
            exp5521={},
        ),
        live_path_reachability={"ok": False, "checks": {"unit": False}},
    )
    blocked_missing_registry = exp5533.build_precheck(
        exp5533.StrategyPrecheckEvidence(registry={}, exp5520={}, exp5521={}),
        live_path_reachability={"ok": False, "checks": {"unit": False}},
    )
    original_probe = exp5533._run_strategy_probe  # noqa: SLF001

    def _small_portfolio_probe(_game: str, _level: str) -> dict[str, Any]:
        probe = original_probe(_game, _level)
        probe["strategy_portfolio"] = probe["strategy_portfolio"][:2]
        return probe

    monkeypatch.setattr(exp5533, "_run_strategy_probe", _small_portfolio_probe)
    blocked_small_portfolio = exp5533.build_precheck(
        exp5533.StrategyPrecheckEvidence(
            registry=_registry(),
            exp5520=_exp5520(),
            exp5521=_exp5521_stale(),
        ),
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    ready_threshold_errors = exp5533.artifact_schema_errors(
        {
            **artifact,
            "selected_game": "",
            "selected_level": "",
            "strategy_routing_live_path_reachable": False,
            "repeated_coordinate_suppression_enabled": False,
            "action_entropy_precheck": 0.0,
            "repeated_coordinate_rate_precheck": 1.0,
            "salience_coverage_rate_precheck": 0.0,
            "arc_sge_candidate_ready": True,
        }
    )
    metric_no_coord = exp5533._selection_metrics(  # noqa: SLF001
        [{"action": 5}],
        total_salience_candidates=2,
    )

    assert exp5533._as_int("bad", 4) == 4  # noqa: SLF001
    assert exp5533._as_float("bad", 0.5) == 0.5  # noqa: SLF001
    assert exp5533._parse_level("3") == 3  # noqa: SLF001
    assert exp5533._parse_level("bad") == 0  # noqa: SLF001
    assert exp5533._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    assert exp5533._read_yaml(tmp_path / "missing.yaml") == {  # noqa: SLF001
        "reproducible_total_levels": 0,
        "games": [],
    }
    assert exp5533._read_text(tmp_path / "missing.md") == ""  # noqa: SLF001
    assert exp5533._candidate_signature({"action": 5}) == "A5"  # noqa: SLF001
    assert exp5533._exp5521_stale_target(  # noqa: SLF001
        {**_exp5521_stale(), "repeated_coordinate_rate": 0.0}
    ) == ""
    assert blocked_no_depth["honest_verdict"].startswith("blocked:")
    assert "strategy_routing_live_path_not_reachable" in blocked_no_depth["honest_verdict"]
    assert "no_registry_safe_adjacent_target" in blocked_no_depth["honest_verdict"]
    assert "registry_missing" in blocked_missing_registry["honest_verdict"]
    assert "strategy_portfolio_too_small" in blocked_small_portfolio["honest_verdict"]
    assert metric_no_coord["repeated_coordinate_rate_precheck"] == pytest.approx(0.0)
    assert metric_no_coord["salience_coverage_rate_precheck"] == pytest.approx(0.0)
    assert "ready artifacts require selected_game" in ready_threshold_errors
    assert "ready artifacts require selected_level" in ready_threshold_errors
    assert "ready artifacts require strategy_routing_live_path_reachable true" in ready_threshold_errors
    assert "ready artifacts require repeated_coordinate_suppression_enabled true" in ready_threshold_errors
    assert "ready artifacts require action_entropy_precheck above threshold" in ready_threshold_errors
    assert "ready artifacts require repeated_coordinate_rate_precheck below threshold" in ready_threshold_errors
    assert "ready artifacts require salience_coverage_rate_precheck above threshold" in ready_threshold_errors

    object_candidate = SimpleNamespace(action_id="bad", data={"x": "bad", "y": 2}, score="bad")
    payload_candidate = SimpleNamespace(action_id=5, data={"foo": "bar"}, score=1.5)
    empty_candidate = SimpleNamespace(action_id=5, data=None, score=1.0)
    assert bounded_router._candidate_action(object_candidate) == 0  # noqa: SLF001
    assert bounded_router._candidate_coordinate(object_candidate) is None  # noqa: SLF001
    assert bounded_router._candidate_score(object_candidate, "score") == 0.0  # noqa: SLF001
    assert bounded_router._candidate_coordinate({"x": 1, "y": 2}) == (1, 2)  # noqa: SLF001
    assert bounded_router._candidate_coordinate({"x": "bad", "y": 2}) is None  # noqa: SLF001
    assert bounded_router._candidate_score(SimpleNamespace(score=2.0), "score") == 2.0  # noqa: SLF001
    assert bounded_router._candidate_signature(payload_candidate) == "A5@foo=bar"  # noqa: SLF001
    assert bounded_router._candidate_signature(empty_candidate) == "A5"  # noqa: SLF001

    router = BoundedStrategyCandidateRouter(
        strategies=[
            {"name": "first", "score_field": "score", "bound": 1},
            {"name": "second", "score_field": "score", "bound": 1},
        ],
        max_candidates=3,
        suppress_repeated_coordinates=True,
    )
    ranked = router.rank(
        None,
        [
            {"action": 6, "data": {"x": 1, "y": 1}, "score": 3.0},
            {"action": 6, "data": {"x": 2, "y": 2}, "score": 2.0},
        ],
    )
    assert [row["data"] for row in ranked] == [{"x": 1, "y": 1}, {"x": 2, "y": 2}]
    assert router.last_diagnostics["strategies_used"] == ["first", "second"]
    fill_router = BoundedStrategyCandidateRouter(
        strategies=[{"name": "only", "score_field": "score", "bound": 1}],
        max_candidates=2,
        suppress_repeated_coordinates=True,
    )
    fill_ranked = fill_router.rank(
        None,
        [
            {"action": 6, "data": {"x": 4, "y": 4}, "score": 3.0},
            {"action": 6, "data": {"x": 5, "y": 5}, "score": 2.0},
        ],
    )
    assert [row["data"] for row in fill_ranked] == [{"x": 4, "y": 4}, {"x": 5, "y": 5}]
    assert fill_router.last_diagnostics["strategies_used"] == ["only", "fallback_fill"]
    assert isinstance(exp5533.strategy_routing_live_path_reachability()["ok"], bool)


def test_scenario_arc_fcp_5533_run_experiment_writes_result(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5533: run_experiment writes the no-solve precheck artifact."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5533.run_experiment(
        root=root,
        tests_run=["unit 5533 run"],
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    written = json.loads((root / exp5533.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "g50t"
    assert artifact["selected_level"] == "L3"
    assert artifact["arc_sge_candidate_ready"] is True
    assert artifact["tests_added_or_reused"] == ["unit 5533 run"]
    assert artifact["preconditions_checked"]["spec_has_req_5533"] is True
