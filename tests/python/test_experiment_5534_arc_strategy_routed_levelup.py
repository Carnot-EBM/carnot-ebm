"""Tests for Exp5534 ARC strategy-routed live level-up attempt.

Spec refs: REQ-ARC-FCP-5534,
SCENARIO-ARC-FCP-5534.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5534_arc_strategy_routed_levelup as exp5534


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _portfolio() -> list[dict[str, Any]]:
    return [
        {"name": "salience_first", "score_field": "salience_score", "bound": 1},
        {"name": "action_effect_memory", "score_field": "effect_score", "bound": 1},
        {
            "name": "verifier_router_candidate_ranking",
            "score_field": "verifier_score",
            "bound": 1,
        },
        {"name": "conservative_reset_reinduction", "score_field": "reset_score", "bound": 1},
    ]


def _registry(*, g50t_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": g50t_levels},
        ],
    }


def _exp5533(*, ready: bool = True, game: str = "g50t", level: str = "L3") -> dict[str, Any]:
    return {
        "experiment": "experiment_5533_arc_strategy_routing_precheck",
        "arc_sge_candidate_ready": ready,
        "selected_game": game,
        "selected_level": level,
        "already_reproduced": False,
        "strategy_portfolio": _portfolio(),
        "strategy_routing_live_path_reachable": True,
        "repeated_coordinate_suppression_enabled": True,
        "solve_provenance": "live_agent_self_discovery",
    }


def _target() -> dict[str, Any]:
    return exp5534.select_target_from_exp5533(_exp5533(), _registry())


def _null_attempt() -> dict[str, Any]:
    action_rows = [
        {
            "step": 1,
            "action": 6,
            "data": {"x": 10, "y": 10},
            "label": "A6@10,10",
            "strategy": "salience_first",
            "verifier_route": "candidate_router.rank",
            "level_before": 2,
            "level_after": 2,
        },
        {
            "step": 2,
            "action": 6,
            "data": {"x": 14, "y": 10},
            "label": "A6@14,10",
            "strategy": "action_effect_memory",
            "verifier_route": "candidate_router.rank",
            "level_before": 2,
            "level_after": 2,
        },
        {
            "step": 3,
            "action": 6,
            "data": {"x": 10, "y": 10},
            "label": "A6@10,10b",
            "strategy": "verifier_router_candidate_ranking",
            "verifier_route": "candidate_router.rank",
            "level_before": 2,
            "level_after": 2,
        },
    ]
    proposed_rows = [
        {"step": 1, "action": 6, "data": {"x": 10, "y": 10}},
        {"step": 1, "action": 6, "data": {"x": 14, "y": 10}},
        {"step": 1, "action": 6, "data": {"x": 20, "y": 10}},
    ]
    return {
        "attempts": len(action_rows),
        "post_levels_reproduced": 2,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "action_rows": action_rows,
        "proposed_action_rows": proposed_rows,
        "observations": [{"step": 0, "event": "reset", "level": 2}],
        "verifier_routes": [{"step": 1, "route": "candidate_router.rank"}],
        "suppression_events": [{"step": 1, "suppressed_coordinate_count": 2}],
        "level_counter_changes": [],
        "reproduction_gate": {"reproduced": False, "reached_level": 2},
        "verifier_feedback": {"reproduction_gate": {"reproduced": False}},
        "solution_labels": [],
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "methodology_receipt": (
            "bounded_live_runtime budget=3 mechanism=strategy_routed_live_agent "
            "gate=standard_reproduction prohibited_inputs=false"
        ),
    }


def _success_attempt() -> dict[str, Any]:
    return {
        **_null_attempt(),
        "attempts": 4,
        "post_levels_reproduced": 3,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "action_rows": [
            *_null_attempt()["action_rows"],
            {
                "step": 4,
                "action": 6,
                "data": {"x": 20, "y": 10},
                "label": "A6@20,10",
                "strategy": "conservative_reset_reinduction",
                "verifier_route": "candidate_router.rank",
                "level_before": 2,
                "level_after": 3,
            },
        ],
        "level_counter_changes": [{"step": 4, "level_before": 2, "level_after": 3}],
        "reproduction_gate": {"reproduced": True, "claimed_level": 3, "reached_level": 3},
        "verifier_feedback": {
            "reproduction_gate": {"reproduced": True, "claimed_level": 3, "reached_level": 3}
        },
        "solution_labels": ['{"action":6,"data":{"x":20,"y":10}}'],
        "failure_mode": "",
    }


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5534.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5534\nSCENARIO-ARC-FCP-5534\n",
        encoding="utf-8",
    )
    (root / exp5534.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5534.EXP5533_RELATIVE_PATH).write_text(
        json.dumps(_exp5533()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5534_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5534: OpenSpec anchors the live attempt artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5534" in spec
    assert "SCENARIO-ARC-FCP-5534" in spec
    assert exp5534.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5534.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5534_target_comes_from_exp5533_and_duplicate_blocks() -> None:
    """SCENARIO-ARC-FCP-5534: duplicate Exp5533 targets do not rotate."""

    selected = exp5534.select_target_from_exp5533(_exp5533(), _registry())
    duplicate = exp5534.select_target_from_exp5533(_exp5533(), _registry(g50t_levels=3))
    not_ready = exp5534.select_target_from_exp5533(_exp5533(ready=False), _registry())
    malformed = exp5534.select_target_from_exp5533(_exp5533(level="bad"), _registry())

    assert selected["blocked"] is False
    assert selected["selected_game"] == "g50t"
    assert selected["selected_level"] == "L3"
    assert selected["target_level"] == 3
    assert selected["prior_levels_reproduced"] == 2
    assert duplicate["blocked"] is True
    assert duplicate["blocker"] == "blocked_duplicate_target"
    assert duplicate["selected_game"] == "g50t"
    assert duplicate["selected_level"] == "L3"
    assert not_ready["blocker"] == "exp5533_candidate_not_ready"
    assert malformed["blocker"] == "exp5533_selected_level_malformed"


def test_scenario_arc_fcp_5534_artifact_gates_honest_null_and_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5534: only reproduced live self-discovery trajectories bank."""

    null_artifact = exp5534.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_path="results/null-5534.json",
        exp5533=_exp5533(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    success = exp5534.build_artifact(
        target=_target(),
        attempt=_success_attempt(),
        registry_updated=True,
        trajectory_path="results/success-5534.json",
        exp5533=_exp5533(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    log_path = tmp_path / "trajectory.json"
    log = exp5534.build_trajectory_log(
        target=_target(),
        attempt=_null_attempt(),
        artifact=null_artifact,
        exp5533=_exp5533(),
    )
    exp5534.write_trajectory_log(log_path, log)

    exp5534.validate_artifact(null_artifact)
    exp5534.validate_artifact(success)
    assert null_artifact["status"] == "honest_null"
    assert null_artifact["strategy_portfolio_used"] == _portfolio()
    assert null_artifact["strategy_switch_count"] == 2
    assert null_artifact["attempts"] == 3
    assert null_artifact["action_entropy"] == pytest.approx(0.9182958340544896)
    assert null_artifact["repeated_coordinate_rate"] == pytest.approx(1.0 / 3.0)
    assert null_artifact["repeated_coordinate_suppression_events"] == 2
    assert null_artifact["salience_coverage_rate"] == pytest.approx(2.0 / 3.0)
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["registry_delta"] == 0
    assert null_artifact["honest_verdict"].startswith("honest_null:")
    assert success["status"] == "complete"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["registry_delta"] == 1
    assert json.loads(log_path.read_text(encoding="utf-8"))["suppression_events"] == (
        _null_attempt()["suppression_events"]
    )


def test_req_arc_fcp_5534_schema_rejects_malformed_credit() -> None:
    """REQ-ARC-FCP-5534: schema rejects off-path or malformed banking credit."""

    artifact = exp5534.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_path="results/null-5534.json",
        exp5533=_exp5533(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "selected_game": 7,
        "selected_level": [],
        "solve_provenance": "development_proxy",
        "strategy_portfolio_used": [],
        "strategy_switch_count": "2",
        "attempts": "3",
        "action_entropy": "1.0",
        "repeated_coordinate_rate": 1.5,
        "repeated_coordinate_suppression_events": -1,
        "salience_coverage_rate": -0.1,
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_delta": -1,
        "trajectory_path": "",
        "model_specs": "none",
        "llm_strategy_proposer_used": "false",
        "arc_live_levelup_ready": "true",
        "tests_added_or_reused": "unit",
        "field_principles": [],
        "inference_substrate": "offline_bfs",
        "honest_verdict": "solved",
        "offline_bfs_used": True,
        "game_source_read": True,
        "hand_built_per_game_adapter_used": True,
        "methodology_receipt": "",
    }
    missing_gate = {
        **artifact,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "registry_delta": 0,
    }
    malformed_types = {
        **artifact,
        "offline_reproduced": "false",
        "reproduced_levels": "0",
        "registry_delta": "0",
        "offline_bfs_used": "false",
        "registry_updated": True,
    }
    negative_levels = {**artifact, "reproduced_levels": -1}

    errors = exp5534.artifact_schema_errors(invalid)
    gate_errors = exp5534.artifact_schema_errors(missing_gate)
    type_errors = exp5534.artifact_schema_errors(malformed_types)
    negative_errors = exp5534.artifact_schema_errors(negative_levels)

    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "strategy_portfolio_used must contain at least three strategies" in errors
    assert "strategy_switch_count must be bare int" in errors
    assert "attempts must be bare int" in errors
    assert "action_entropy must be bare float" in errors
    assert "repeated_coordinate_rate must be in [0, 1]" in errors
    assert "repeated_coordinate_suppression_events must be non-negative" in errors
    assert "salience_coverage_rate must be in [0, 1]" in errors
    assert "offline_reproduced requires reproduced_levels >= 1" in errors
    assert "registry_delta must be non-negative" in errors
    assert "trajectory_path must be a non-empty string" in errors
    assert "model_specs must be a list" in errors
    assert "llm_strategy_proposer_used must be bare bool" in errors
    assert "arc_live_levelup_ready must be bare bool" in errors
    assert "tests_added_or_reused must be a non-empty list" in errors
    assert "field_principles must be a mapping" in errors
    assert "inference_substrate must be arc_live_agent_self_discovery" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    assert "offline_bfs_used must be false" in errors
    assert "game_source_read must be false" in errors
    assert "hand_built_per_game_adapter_used must be false" in errors
    assert "methodology_receipt must be a non-empty string" in errors
    assert "offline_reproduced true requires registry_delta == reproduced_levels" in gate_errors
    assert "offline_reproduced must be bare bool" in type_errors
    assert "reproduced_levels must be bare int" in type_errors
    assert "registry_delta must be bare int" in type_errors
    assert "offline_bfs_used must be bare bool" in type_errors
    assert "registry_updated requires offline_reproduced true" in type_errors
    assert "reproduced_levels must be non-negative" in negative_errors
    with pytest.raises(ValueError):
        exp5534.validate_artifact(invalid)


def test_scenario_arc_fcp_5534_run_experiment_writes_null_without_registry_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5534: honest null writes both result and trajectory log."""

    root = _tmp_ready_root(tmp_path)

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["target"]["selected_game"] == "g50t"
        assert kwargs["target"]["target_level"] == 3
        assert kwargs["budget"] == 3
        assert kwargs["strategy_portfolio"] == _portfolio()
        return _null_attempt()

    artifact = exp5534.run_experiment(
        root=root,
        budget=3,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5534 null"],
    )
    written = json.loads((root / exp5534.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    log = json.loads((root / exp5534.TRAJECTORY_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((root / exp5534.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["registry_delta"] == 0
    assert artifact["trajectory_path"] == exp5534.TRAJECTORY_RELATIVE_PATH
    assert log["selected_game"] == "g50t"
    assert log["verifier_routes"] == _null_attempt()["verifier_routes"]
    assert registry["reproducible_total_levels"] == 69
    assert registry["games"][1]["levels_reproduced"] == 2
    assert artifact["tests_added_or_reused"] == ["unit 5534 null"]


def test_scenario_arc_fcp_5534_run_experiment_updates_registry_only_on_banked_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5534: reproduced live trajectory updates registry once."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5534.run_experiment(
        root=root,
        budget=4,
        attempt_runner=lambda **_kwargs: _success_attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5534 success"],
    )
    registry = yaml.safe_load((root / exp5534.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["registry_delta"] == 1
    assert registry["reproducible_total_levels"] == 70
    assert registry["games"][1]["levels_reproduced"] == 3
    assert registry["games"][1]["latest_exp5534_strategy_routed_levelup"]["artifact"] == (
        exp5534.RESULT_RELATIVE_PATH
    )


def test_scenario_arc_fcp_5534_blocked_preconditions_write_schema_valid_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5534: missing target, duplicate target, or harness blocks runtime."""

    missing = exp5534.run_experiment(root=tmp_path, tests_run=["missing 5534"])
    duplicate_root = _tmp_ready_root(tmp_path / "duplicate", registry=_registry(g50t_levels=3))
    duplicate = exp5534.run_experiment(
        root=duplicate_root,
        offline_arcade_check=lambda: True,
        tests_run=["duplicate 5534"],
    )
    no_harness_root = _tmp_ready_root(tmp_path / "no-harness")
    no_harness = exp5534.run_experiment(
        root=no_harness_root,
        offline_arcade_check=lambda: False,
        tests_run=["no harness 5534"],
    )

    assert missing["status"] == "blocked"
    assert missing["selected_game"] == ""
    assert missing["attempts"] == 0
    assert "exp5533_target_missing" in missing["honest_verdict"]
    assert duplicate["status"] == "blocked"
    assert duplicate["selected_game"] == "g50t"
    assert duplicate["registry_delta"] == 0
    assert "blocked_duplicate_target" in duplicate["honest_verdict"]
    assert no_harness["status"] == "blocked"
    assert "missing_harness_access" in no_harness["honest_verdict"]


def test_req_arc_fcp_5534_defensive_helpers_and_registry_new_game(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5534: fallback branches stay explicit and no-credit."""

    artifact = exp5534.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_path="results/null-5534.json",
        exp5533=_exp5533(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    metrics = exp5534.trajectory_metrics(
        [{"action": 6, "x": "1", "y": "2"}, {"action": 6, "x": "1", "y": "2"}],
        [{"action": 6, "x": 1, "y": 2}],
    )
    no_coordinate_metrics = exp5534.trajectory_metrics([{"action": 5}], [])
    wrong_provenance = exp5534.select_target_from_exp5533(
        {**_exp5533(), "solve_provenance": "development_proxy"},
        _registry(),
    )
    missing_game = exp5534.select_target_from_exp5533(_exp5533(game=""), _registry())
    blocked_credit = exp5534._accepted_reproduced_levels(  # noqa: SLF001
        _target(),
        {**_success_attempt(), "offline_bfs_used": True},
    )
    shallow_credit = exp5534._accepted_reproduced_levels(  # noqa: SLF001
        _target(),
        {**_success_attempt(), "post_levels_reproduced": 2},
    )
    new_root = _tmp_ready_root(tmp_path / "new-game", registry={"reproducible_total_levels": 0, "games": []})
    new_artifact = exp5534.build_artifact(
        target={
            **_target(),
            "selected_game": "zz99",
            "selected_level": "L1",
            "target_level": 1,
            "prior_levels_reproduced": 0,
            "registry_before_levels": 0,
        },
        attempt={**_success_attempt(), "post_levels_reproduced": 1, "reproduced_levels": 1},
        registry_updated=True,
        trajectory_path="results/success-5534.json",
        exp5533=_exp5533(game="zz99", level="L1"),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    assert exp5534._as_int("bad", 4) == 4  # noqa: SLF001
    assert exp5534._parse_level_label("3") == 3  # noqa: SLF001
    assert exp5534._row_coordinate({"action": 6, "x": "1", "y": "2"}) == (1, 2)  # noqa: SLF001
    assert exp5534._row_signature({"action": 5}) == "A5"  # noqa: SLF001
    assert exp5534._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    assert exp5534._read_yaml(tmp_path / "missing.yaml") == {"reproducible_total_levels": 0, "games": []}  # noqa: SLF001
    assert metrics["repeated_coordinate_rate"] == pytest.approx(0.5)
    assert no_coordinate_metrics["repeated_coordinate_rate"] == pytest.approx(0.0)
    assert no_coordinate_metrics["salience_coverage_rate"] == pytest.approx(0.0)
    assert wrong_provenance["blocker"] == "exp5533_wrong_provenance"
    assert missing_game["blocker"] == "exp5533_target_missing"
    assert blocked_credit == 0
    assert shallow_credit == 0
    assert exp5534.update_registry_if_banked(
        root=tmp_path,
        artifact=artifact,
        registry=_registry(),
    ) is False
    assert exp5534.update_registry_if_banked(
        root=new_root,
        artifact=new_artifact,
        registry={"reproducible_total_levels": 0, "games": []},
    ) is True
