"""Tests for Exp5521 ARC live action-diverse level-up attempt.

Spec refs: REQ-ARC-FCP-5521,
SCENARIO-ARC-FCP-5521.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5521_arc_live_action_diverse_levelup as exp5521


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, sb26_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": sb26_levels},
        ],
    }


def _precheck(*, ready: bool = True, game: str = "sb26", level: str = "L3") -> dict[str, Any]:
    return {
        "experiment": "experiment_5520_arc_action_diversity_target_precheck",
        "arc_levelup_candidate_ready": ready,
        "selected_game": game,
        "selected_level": level,
        "already_reproduced": False,
        "exp5508_pattern_reused": False,
        "action_entropy": 3.0,
        "repeated_coordinate_rate": 0.0,
        "salience_coverage_rate": 1.0,
        "solve_provenance": "live_agent_self_discovery",
        "exp5508_pattern": {
            "selected_game": "dc22",
            "selected_level": "L3",
            "target_level": 3,
            "coordinates": [{"x": 24, "y": 20}, {"x": 10, "y": 40}],
            "action_entropy": 1.49,
            "repeated_coordinate_rate": 0.93,
        },
    }


def _target() -> dict[str, Any]:
    return exp5521.select_target_from_precheck(_precheck(), _registry())


def _null_attempt() -> dict[str, Any]:
    action_rows = [
        {
            "step": 1,
            "action": 6,
            "data": {"x": 19, "y": 58},
            "level_before": 2,
            "level_after": 2,
            "changed_cells": 4,
        },
        {
            "step": 2,
            "action": 6,
            "data": {"x": 27, "y": 58},
            "level_before": 2,
            "level_after": 2,
            "changed_cells": 0,
        },
        {
            "step": 3,
            "action": 5,
            "data": None,
            "level_before": 2,
            "level_after": 2,
            "changed_cells": 0,
        },
    ]
    proposed = [
        {"step": 1, "action": 6, "data": {"x": 19, "y": 58}, "score": 4000.0},
        {"step": 1, "action": 6, "data": {"x": 27, "y": 58}, "score": 3999.0},
        {"step": 1, "action": 6, "data": {"x": 35, "y": 58}, "score": 3998.0},
    ]
    return {
        "live_agent_attempts": len(action_rows),
        "post_levels_reproduced": 2,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "action_rows": action_rows,
        "proposed_action_rows": proposed,
        "observations": [{"step": 0, "event": "reset", "level": 2}],
        "verifier_feedback": {
            "reproduction_gate": {"reproduced": False, "reached_level": 2}
        },
        "reproduction_gate": {"reproduced": False, "reached_level": 2},
        "solution_labels": [],
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "methodology_receipt": (
            "bounded_live_runtime budget=3 mechanism=action_diverse_perception_generation "
            "gate=standard_reproduction prohibited_inputs=false"
        ),
    }


def _success_attempt() -> dict[str, Any]:
    return {
        **_null_attempt(),
        "live_agent_attempts": 4,
        "post_levels_reproduced": 3,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "action_rows": [
            *_null_attempt()["action_rows"],
            {
                "step": 4,
                "action": 6,
                "data": {"x": 35, "y": 58},
                "level_before": 2,
                "level_after": 3,
                "changed_cells": 8,
            },
        ],
        "reproduction_gate": {
            "reproduced": True,
            "claimed_level": 3,
            "reached_level": 3,
        },
        "verifier_feedback": {
            "reproduction_gate": {
                "reproduced": True,
                "claimed_level": 3,
                "reached_level": 3,
            }
        },
        "solution_labels": ['{"action":6,"data":{"x":19,"y":58}}'],
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
    (root / exp5521.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5521\nSCENARIO-ARC-FCP-5521\n",
        encoding="utf-8",
    )
    (root / exp5521.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5521.PRECHECK_RELATIVE_PATH).write_text(
        json.dumps(_precheck()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5521_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5521: OpenSpec anchors the live attempt artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5521" in spec
    assert "SCENARIO-ARC-FCP-5521" in spec
    assert exp5521.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5521.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5521_generator_suppresses_failed_and_attempted_coordinates() -> None:
    """SCENARIO-ARC-FCP-5521: changed live generator suppresses repeated coordinates."""

    frame = SimpleNamespace(frame=np.zeros((6, 6), dtype=np.int16), levels_completed=2)
    generator = exp5521.ActionDiverseLiveGenerator(
        max_candidates=3,
        avoid_coordinates={(1, 1)},
    )
    generator._last_features = {  # noqa: SLF001 - pins the candidate rows under test.
        "action_affordance_rows": [
            {"x": 1, "y": 1, "score": 40.0},
            {"x": 2, "y": 2, "score": 30.0},
            {"x": 3, "y": 3, "score": 20.0},
            {"x": 4, "y": 4, "score": 10.0},
        ],
        "motion_affordance_rows": [],
    }

    first_points = generator.click_points(frame, max_points=3)
    generator.observe_transition(frame, 6, {"x": 2, "y": 2}, frame)
    second_points = generator.click_points(frame, max_points=3)
    diagnostics = generator.diagnostics()

    assert first_points == [(2, 2), (3, 3), (4, 4)]
    assert second_points == [(3, 3), (4, 4)]
    assert diagnostics["action_diversity"]["attempted_coordinates"] == [
        {"x": 2, "y": 2}
    ]
    assert diagnostics["action_diversity"]["suppressed_coordinate_count"] >= 2
    assert generator.for_path([{"action": 6, "data": {"x": 3, "y": 3}}]) is generator
    assert generator.action_tier_rows(
        frame,
        [{"action": 6, "data": {"x": 1, "y": 1}}],
    )
    assert generator.action_tier_rows(
        frame,
        [{"action": 6, "data": {"x": 5, "y": 5}}],
    )[0]["data"] == {"x": 5, "y": 5}


def test_scenario_arc_fcp_5521_selects_only_ready_registry_safe_target() -> None:
    """SCENARIO-ARC-FCP-5521: Exp5520 readiness and registry depth gate live runtime."""

    selected = exp5521.select_target_from_precheck(_precheck(), _registry())
    duplicate = exp5521.select_target_from_precheck(_precheck(), _registry(sb26_levels=3))
    not_ready = exp5521.select_target_from_precheck(_precheck(ready=False), _registry())
    bad_level = exp5521.select_target_from_precheck(_precheck(level="bad"), _registry())

    assert selected["blocked"] is False
    assert selected["selected_game"] == "sb26"
    assert selected["selected_level"] == "L3"
    assert selected["target_level"] == 3
    assert selected["prior_levels_reproduced"] == 2
    assert duplicate["blocked"] is True
    assert duplicate["blocker"] == "selected_level_already_reproducible"
    assert not_ready["blocker"] == "exp5520_candidate_not_ready"
    assert bad_level["blocker"] == "exp5520_selected_level_malformed"


def test_scenario_arc_fcp_5521_artifact_gates_honest_null_and_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5521: only reproduced live self-discovery trajectories bank."""

    null_artifact = exp5521.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_log_path="results/null-log.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    success = exp5521.build_artifact(
        target=_target(),
        attempt=_success_attempt(),
        registry_updated=True,
        trajectory_log_path="results/success-log.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    log_path = tmp_path / "trajectory.json"
    log = exp5521.build_trajectory_log(
        target=_target(),
        attempt=_null_attempt(),
        artifact=null_artifact,
        precheck=_precheck(),
    )
    exp5521.write_trajectory_log(log_path, log)

    exp5521.validate_artifact(null_artifact)
    exp5521.validate_artifact(success)
    assert null_artifact["status"] == "honest_null"
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["banking_gate"] is False
    assert null_artifact["registry_delta"] == 0
    assert null_artifact["reproduction_command"] is None
    assert null_artifact["honest_verdict"].startswith("honest_null:")
    assert null_artifact["action_entropy"] == pytest.approx(1.584962500721156)
    assert null_artifact["repeated_coordinate_rate"] == pytest.approx(0.0)
    assert null_artifact["salience_coverage_rate"] == pytest.approx(2.0 / 3.0)
    assert success["status"] == "complete"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["banking_gate"] is True
    assert success["registry_delta"] == 1
    assert success["reproduction_command"] == (
        ".venv/bin/python -m carnot.experiment_5521_arc_live_action_diverse_levelup "
        "--reproduce-log results/success-log.json"
    )
    assert json.loads(log_path.read_text(encoding="utf-8"))["metrics"]["action_entropy"] == (
        null_artifact["action_entropy"]
    )


def test_req_arc_fcp_5521_schema_rejects_malformed_credit() -> None:
    """REQ-ARC-FCP-5521: schema rejects off-path or malformed banking credit."""

    artifact = exp5521.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_log_path="results/null-log.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "selected_game": 7,
        "selected_level": [],
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "banking_gate": True,
        "registry_delta": 2,
        "solve_provenance": "development_proxy",
        "live_attempts": "3",
        "action_entropy": "1.0",
        "repeated_coordinate_rate": 1.5,
        "salience_coverage_rate": -0.1,
        "trajectory_log_path": "",
        "reproduction_command": 5,
        "arc_live_levelup_ready": "true",
        "inference_substrate": "offline_bfs",
        "honest_verdict": "solved",
        "offline_bfs_used": True,
        "game_source_read": True,
        "hand_built_per_game_adapter_used": True,
    }
    missing_gate = {
        **artifact,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "banking_gate": False,
        "registry_delta": 0,
    }

    errors = exp5521.artifact_schema_errors(invalid)
    gate_errors = exp5521.artifact_schema_errors(missing_gate)

    assert "selected_game must be a string" in errors
    assert "selected_level must be a string or int" in errors
    assert "offline_reproduced requires reproduced_levels >= 1" in errors
    assert "banking_gate true requires registry_delta == reproduced_levels" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "live_attempts must be bare int" in errors
    assert "action_entropy must be bare float" in errors
    assert "repeated_coordinate_rate must be in [0, 1]" in errors
    assert "salience_coverage_rate must be in [0, 1]" in errors
    assert "trajectory_log_path must be a non-empty string" in errors
    assert "reproduction_command must be string or null" in errors
    assert "arc_live_levelup_ready must be bare bool" in errors
    assert "inference_substrate must be arc_live_agent_self_discovery" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    assert "offline_bfs_used must be false" in errors
    assert "game_source_read must be false" in errors
    assert "hand_built_per_game_adapter_used must be false" in errors
    assert "offline_reproduced true requires banking_gate true" in gate_errors
    with pytest.raises(ValueError):
        exp5521.validate_artifact(invalid)


def test_scenario_arc_fcp_5521_run_experiment_writes_null_without_registry_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5521: honest null writes both result and trajectory log."""

    root = _tmp_ready_root(tmp_path)

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["target"]["selected_game"] == "sb26"
        assert kwargs["target"]["target_level"] == 3
        assert kwargs["budget"] == 3
        return _null_attempt()

    artifact = exp5521.run_experiment(
        root=root,
        budget=3,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5521 null"],
    )
    written = json.loads((root / exp5521.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    log = json.loads((root / exp5521.TRAJECTORY_LOG_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((root / exp5521.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["registry_delta"] == 0
    assert artifact["trajectory_log_path"] == exp5521.TRAJECTORY_LOG_RELATIVE_PATH
    assert log["selected_game"] == "sb26"
    assert log["verifier_feedback"]["reproduction_gate"]["reproduced"] is False
    assert registry["reproducible_total_levels"] == 69
    assert registry["games"][1]["levels_reproduced"] == 2
    assert artifact["tests_run"] == ["unit 5521 null"]


def test_scenario_arc_fcp_5521_run_experiment_updates_registry_only_on_banked_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5521: reproduced live trajectory updates registry once."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5521.run_experiment(
        root=root,
        budget=4,
        attempt_runner=lambda **_kwargs: _success_attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5521 success"],
    )
    registry = yaml.safe_load((root / exp5521.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["banking_gate"] is True
    assert artifact["registry_delta"] == 1
    assert artifact["reproduction_command"] is not None
    assert registry["reproducible_total_levels"] == 70
    assert registry["games"][1]["levels_reproduced"] == 3
    assert registry["games"][1]["latest_exp5521_levelup_attempt"]["artifact"] == (
        exp5521.RESULT_RELATIVE_PATH
    )


def test_scenario_arc_fcp_5521_blocked_preconditions_write_schema_valid_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5521: missing target or harness blocks before live runtime."""

    missing = exp5521.run_experiment(root=tmp_path, tests_run=["missing 5521"])
    duplicate_root = _tmp_ready_root(tmp_path / "duplicate", registry=_registry(sb26_levels=3))
    duplicate = exp5521.run_experiment(
        root=duplicate_root,
        offline_arcade_check=lambda: True,
        tests_run=["duplicate 5521"],
    )
    no_harness_root = _tmp_ready_root(tmp_path / "no-harness")
    no_harness = exp5521.run_experiment(
        root=no_harness_root,
        offline_arcade_check=lambda: False,
        tests_run=["no harness 5521"],
    )

    assert missing["status"] == "blocked"
    assert missing["selected_game"] == ""
    assert missing["live_attempts"] == 0
    assert "exp5520_target_missing" in missing["honest_verdict"]
    assert duplicate["status"] == "blocked"
    assert duplicate["registry_delta"] == 0
    assert "selected_level_already_reproducible" in duplicate["honest_verdict"]
    assert no_harness["status"] == "blocked"
    assert "missing_harness_access" in no_harness["honest_verdict"]


def test_req_arc_fcp_5521_defensive_branches_and_registry_helpers(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5521: fallback branches stay explicit and no-credit."""

    artifact = exp5521.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_log_path="results/null-log.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    repeated_metrics = exp5521.trajectory_metrics(
        [
            {"action": 6, "x": "1", "y": "2"},
            {"action": 6, "x": "1", "y": "2"},
            {"action": 5},
        ],
        [{"action": 6, "x": 1, "y": 2}],
    )
    wrong_provenance = exp5521.select_target_from_precheck(
        {**_precheck(), "solve_provenance": "development_proxy"},
        _registry(),
    )
    missing_game = exp5521.select_target_from_precheck(_precheck(game=""), _registry())
    blocked_credit = exp5521._accepted_reproduced_levels(  # noqa: SLF001
        _target(),
        {**_success_attempt(), "offline_bfs_used": True},
    )
    shallow_credit = exp5521._accepted_reproduced_levels(  # noqa: SLF001
        _target(),
        {**_success_attempt(), "post_levels_reproduced": 2},
    )
    malformed = {
        **artifact,
        "offline_reproduced": "false",
        "reproduced_levels": -1,
        "banking_gate": "false",
        "registry_delta": -1,
        "live_attempts": -1,
        "offline_bfs_used": "false",
        "methodology_receipt": "",
        "registry_updated": True,
    }
    malformed_types = {
        **artifact,
        "reproduced_levels": "0",
        "registry_delta": "0",
    }
    bad_banking = {
        **artifact,
        "offline_reproduced": False,
        "reproduced_levels": 1,
        "banking_gate": True,
        "registry_delta": 1,
        "reproduction_command": None,
    }
    new_root = _tmp_ready_root(tmp_path / "new-game", registry={"reproducible_total_levels": 0, "games": []})
    new_artifact = {
        **exp5521.build_artifact(
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
            trajectory_log_path="results/success-log.json",
            precheck=_precheck(game="zz99", level="L1"),
            preconditions_checked={"unit": True},
            tests_run=["unit"],
            duration_s=0.1,
        ),
        "selected_game": "zz99",
    }

    assert exp5521._as_int("bad", 4) == 4  # noqa: SLF001
    assert exp5521._parse_level_label("3") == 3  # noqa: SLF001
    assert exp5521._row_coordinate({"action": 6, "x": "1", "y": "2"}) == (1, 2)  # noqa: SLF001
    assert exp5521._pattern_coordinates(_precheck()) == {(10, 40), (24, 20)}  # noqa: SLF001
    assert exp5521._pattern_coordinates({}) == set()  # noqa: SLF001
    direct_policy = SimpleNamespace(action_salience_diagnostics=lambda: {"action_tier_rows": [1]})
    nested_policy = SimpleNamespace(
        explorer=SimpleNamespace(action_salience_diagnostics=lambda: {"action_tier_rows": [2]})
    )
    empty_policy = SimpleNamespace()
    assert exp5521._policy_salience_diagnostics(direct_policy)["action_tier_rows"] == [1]  # noqa: SLF001
    assert exp5521._policy_salience_diagnostics(nested_policy)["action_tier_rows"] == [2]  # noqa: SLF001
    assert exp5521._policy_salience_diagnostics(empty_policy)["action_tier_rows"] == []  # noqa: SLF001
    assert repeated_metrics["repeated_coordinate_rate"] == pytest.approx(0.5)
    assert wrong_provenance["blocker"] == "exp5520_wrong_provenance"
    assert missing_game["blocker"] == "exp5520_target_missing"
    assert blocked_credit == 0
    assert shallow_credit == 0
    assert exp5521.update_registry_if_banked(
        root=tmp_path,
        artifact=artifact,
        registry=_registry(),
    ) is False
    assert exp5521.update_registry_if_banked(
        root=new_root,
        artifact=new_artifact,
        registry={"reproducible_total_levels": 0, "games": []},
    ) is True
    malformed_errors = exp5521.artifact_schema_errors(malformed)
    malformed_type_errors = exp5521.artifact_schema_errors(malformed_types)
    bad_banking_errors = exp5521.artifact_schema_errors(bad_banking)
    assert "offline_reproduced must be bare bool" in malformed_errors
    assert "reproduced_levels must be non-negative" in malformed_errors
    assert "banking_gate must be bare bool" in malformed_errors
    assert "registry_delta must be non-negative" in malformed_errors
    assert "live_attempts must be non-negative" in malformed_errors
    assert "offline_bfs_used must be bare bool" in malformed_errors
    assert "methodology_receipt must be a non-empty string" in malformed_errors
    assert "registry_updated requires banking_gate true" in malformed_errors
    assert "reproduced_levels must be bare int" in malformed_type_errors
    assert "registry_delta must be bare int" in malformed_type_errors
    assert "banking_gate true requires offline_reproduced true" in bad_banking_errors
    assert "banking_gate true requires reproduction_command" in bad_banking_errors
