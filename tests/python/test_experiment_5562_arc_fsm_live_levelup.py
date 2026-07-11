"""Tests for Exp5562 gated ARC FSM live level-up attempt.

Spec refs: REQ-ARC-FCP-5562,
SCENARIO-ARC-FCP-5562.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5562_arc_fsm_live_levelup as exp5562


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, r11l_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": r11l_levels},
            {"game": "ls20", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _precheck(*, ready: bool = True, game: str = "r11l", level: str = "L3") -> dict[str, Any]:
    return {
        "experiment": "experiment_5561_arc_fsm_target_rotation_precheck",
        "arc_fsm_precheck_ready": ready,
        "selected_game": game,
        "selected_level": level,
        "random_seed": 5561,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "fsm_action_abstraction_ready": True,
        "repeated_coordinate_suppression_enabled": True,
        "action_entropy_precheck": 2.0,
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "arc_live_path_precheck_no_llm",
        "target_selection": {
            "selected_game": game,
            "selected_level": level,
            "target_level": 3,
            "prior_levels_reproduced": 2,
            "registry_precheck_passed": True,
        },
    }


def _target() -> dict[str, Any]:
    return exp5562.select_target_from_precheck(_precheck(), _registry())


def _null_attempt() -> dict[str, Any]:
    action_rows = [
        {
            "step": 1,
            "action": 6,
            "data": {"x": 10, "y": 10},
            "label": "A6@10,10",
            "fsm_phase": "observe_select",
            "fsm_action": "observe_select:A6@10,10",
            "verifier_route": "candidate_router.rank",
            "level_before": 2,
            "level_after": 2,
        },
        {
            "step": 2,
            "action": 6,
            "data": {"x": 14, "y": 10},
            "label": "A6@14,10",
            "fsm_phase": "effect_transition",
            "fsm_action": "effect_transition:A6@14,10",
            "verifier_route": "candidate_router.rank",
            "level_before": 2,
            "level_after": 2,
        },
        {
            "step": 3,
            "action": 6,
            "data": {"x": 10, "y": 10},
            "label": "A6@10,10-again",
            "fsm_phase": "verify_candidate",
            "fsm_action": "verify_candidate:A6@10,10",
            "verifier_route": "candidate_router.rank",
            "level_before": 2,
            "level_after": 2,
        },
    ]
    return {
        "attempts": len(action_rows),
        "post_levels_reproduced": 2,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "action_rows": action_rows,
        "proposed_action_rows": [{"action": 6, "data": {"x": 10, "y": 10}}],
        "observations": [{"step": 0, "event": "reset", "level": 2}],
        "verifier_routes": [
            {"step": 1, "route": "candidate_router.rank", "fsm_phase": "observe_select"}
        ],
        "suppression_events": [{"step": 1, "suppressed_coordinate_count": 2}],
        "level_counter_changes": [],
        "reproduction_gate": {"reproduced": False, "reached_level": 2},
        "verifier_feedback": {"reproduction_gate": {"reproduced": False}},
        "runtime_reverse_engineering": [
            {"step": 1, "signal": "fsm_phase_observed", "phase": "observe_select"}
        ],
        "solution_labels": [],
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "terminal_state": {"status": "budget_exhausted", "max_level": 2, "target_level": 3},
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "llm_strategy_proposer_used": False,
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
                "fsm_phase": "reset_reinduce",
                "fsm_action": "reset_reinduce:A6@20,10",
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
        "runtime_reverse_engineering": [
            *_null_attempt()["runtime_reverse_engineering"],
            {"step": 4, "signal": "level_counter_changed", "level_after": 3},
        ],
        "solution_labels": ['{"action":6,"data":{"x":20,"y":10}}'],
        "failure_mode": "",
        "terminal_state": {"status": "reproduced", "max_level": 3, "target_level": 3},
    }


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5562.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5562\nSCENARIO-ARC-FCP-5562\n",
        encoding="utf-8",
    )
    (root / exp5562.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5562.UPSTREAM_RELATIVE_PATH).write_text(
        json.dumps(_precheck()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5562_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5562: OpenSpec anchors the gated FSM live-attempt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5562" in spec
    assert "SCENARIO-ARC-FCP-5562" in spec
    assert exp5562.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5562.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5562_uses_exp5561_target_and_prevents_duplicates() -> None:
    """SCENARIO-ARC-FCP-5562: registry reread gates live runtime before attempts."""

    selected = exp5562.select_target_from_precheck(_precheck(), _registry())
    duplicate = exp5562.select_target_from_precheck(_precheck(), _registry(r11l_levels=3))
    blocked = exp5562.select_target_from_precheck(_precheck(ready=False), _registry())
    malformed = exp5562.select_target_from_precheck(_precheck(level="bad"), _registry())
    missing = exp5562.select_target_from_precheck(_precheck(game="missing"), _registry())
    empty = exp5562.select_target_from_precheck(_precheck(game=""), _registry())

    assert selected["blocked"] is False
    assert selected["duplicate_prevented"] is False
    assert selected["selected_game"] == "r11l"
    assert selected["selected_level"] == "L3"
    assert selected["target_level"] == 3
    assert selected["arc_live_levelup_ready"] is True
    assert duplicate["duplicate_prevented"] is True
    assert duplicate["blocked"] is True
    assert duplicate["blocker"] == "target_already_reproduced_duplicate_prevented"
    assert duplicate["arc_live_levelup_ready"] is False
    assert blocked["blocker"] == "exp5561_precheck_not_ready"
    assert malformed["blocker"] == "exp5561_selected_level_malformed"
    assert missing["blocker"] == "exp5561_target_missing_from_registry"
    assert empty["blocker"] == "exp5561_target_missing"


def test_scenario_arc_fcp_5562_artifact_gates_null_success_and_duplicate(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5562: only reproduced self-discovery can bank a level."""

    null_artifact = exp5562.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_path="results/null-5562.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
        attempt_budget=3,
    )
    success = exp5562.build_artifact(
        target=_target(),
        attempt=_success_attempt(),
        registry_updated=True,
        trajectory_path="results/success-5562.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
        attempt_budget=4,
    )
    duplicate_target = exp5562.select_target_from_precheck(_precheck(), _registry(r11l_levels=3))
    duplicate_artifact = exp5562.build_artifact(
        target=duplicate_target,
        attempt=exp5562.blocked_attempt("target_already_reproduced_duplicate_prevented"),
        registry_updated=False,
        trajectory_path="results/duplicate-5562.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
        attempt_budget=0,
    )
    log_path = tmp_path / "trajectory.json"
    log = exp5562.build_trajectory_log(
        target=_target(),
        attempt=_null_attempt(),
        artifact=null_artifact,
        precheck=_precheck(),
    )
    exp5562.write_trajectory_log(log_path, log)

    exp5562.validate_artifact(null_artifact)
    exp5562.validate_artifact(success)
    exp5562.validate_artifact(duplicate_artifact)
    assert null_artifact["status"] == "honest_null"
    assert null_artifact["llm_invoked"] is False
    assert null_artifact["llm_strategy_proposer_used"] is False
    assert null_artifact["no_model_specs_required"] is True
    assert null_artifact["upstream_arc_precheck"] == exp5562.UPSTREAM_RELATIVE_PATH
    assert null_artifact["solve_provenance"] == "live_agent_self_discovery"
    assert null_artifact["random_seed"] == 5561
    assert null_artifact["attempts"] == 3
    assert null_artifact["action_entropy"] == pytest.approx(0.9182958340544896)
    assert null_artifact["repeated_coordinate_rate"] == pytest.approx(1.0 / 3.0)
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["registry_delta"] == 0
    assert null_artifact["arc_live_levelup_ready"] is True
    assert null_artifact["inference_substrate"] == "arc_live_agent_self_discovery_no_llm"
    assert null_artifact["honest_verdict"].startswith("honest_null:")
    assert "model_specs" not in null_artifact
    assert "target_model" not in null_artifact
    assert success["status"] == "complete"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["registry_delta"] == 1
    assert success["honest_verdict"].startswith("complete:")
    assert duplicate_artifact["status"] == "duplicate_prevented"
    assert duplicate_artifact["attempts"] == 0
    assert duplicate_artifact["arc_live_levelup_ready"] is False
    assert duplicate_artifact["honest_verdict"].startswith("duplicate_prevented:")
    assert json.loads(log_path.read_text(encoding="utf-8"))["terminal_state"]["status"] == (
        "budget_exhausted"
    )


def test_req_arc_fcp_5562_schema_rejects_model_specs_and_malformed_credit() -> None:
    """REQ-ARC-FCP-5562: schema rejects model metadata and invalid banking gates."""

    artifact = exp5562.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        trajectory_path="results/null-5562.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
        attempt_budget=3,
    )
    invalid = {
        **artifact,
        "llm_invoked": True,
        "no_model_specs_required": False,
        "upstream_arc_precheck": "",
        "solve_provenance": "development_proxy",
        "llm_strategy_proposer_used": True,
        "random_seed": "5561",
        "reproducibility_checksum": "",
        "selected_game": 7,
        "selected_level": [],
        "attempts": "3",
        "trajectory_path": "",
        "action_entropy": "1.0",
        "repeated_coordinate_rate": 1.1,
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "registry_delta": -1,
        "arc_live_levelup_ready": "true",
        "tests_added_or_reused": "unit",
        "field_principles": [],
        "inference_substrate": "live_llm_inference",
        "honest_verdict": "solved",
        "model_specs": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "target_model": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "offline_bfs_used": True,
        "game_source_read": True,
        "hand_built_per_game_adapter_used": True,
        "registry_updated": True,
    }
    missing_gate = {**artifact, "offline_reproduced": True, "reproduced_levels": 1}
    registry_update_error = {**artifact, "registry_updated": True}
    type_errors = {
        **artifact,
        "offline_reproduced": "false",
        "reproduced_levels": "0",
        "registry_delta": "0",
        "offline_bfs_used": "false",
    }
    negative_errors = {**artifact, "attempts": -1, "reproduced_levels": -1}
    repeated_type_errors = {**artifact, "repeated_coordinate_rate": "0.0"}
    incomplete_principles = {
        **artifact,
        "field_principles": {"llm_invoked": exp5562.FIELD_PRINCIPLES["llm_invoked"]},
    }

    errors = exp5562.artifact_schema_errors(invalid)
    gate_errors = exp5562.artifact_schema_errors(missing_gate)
    registry_update_errors = exp5562.artifact_schema_errors(registry_update_error)
    bare_type_errors = exp5562.artifact_schema_errors(type_errors)
    negative_field_errors = exp5562.artifact_schema_errors(negative_errors)
    repeated_field_type_errors = exp5562.artifact_schema_errors(repeated_type_errors)
    principle_errors = exp5562.artifact_schema_errors(incomplete_principles)

    assert "llm_invoked must be false" in errors
    assert "no_model_specs_required must be true" in errors
    assert "upstream_arc_precheck must be a non-empty string" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "llm_strategy_proposer_used must be false" in errors
    assert "random_seed must be an int" in errors
    assert "reproducibility_checksum must be a sha256 string" in errors
    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "attempts must be bare int" in errors
    assert "trajectory_path must be a non-empty string" in errors
    assert "action_entropy must be bare float" in errors
    assert "repeated_coordinate_rate must be in [0, 1]" in errors
    assert "offline_reproduced requires reproduced_levels >= 1" in errors
    assert "registry_delta must be non-negative" in errors
    assert "arc_live_levelup_ready must be bare bool" in errors
    assert "tests_added_or_reused must be a non-empty list" in errors
    assert "field_principles must be a mapping" in errors
    assert "field_principles must cover every required field" in principle_errors
    assert "inference_substrate must be arc_live_agent_self_discovery_no_llm" in errors
    assert (
        "honest_verdict must start with complete:, honest_null:, duplicate_prevented:, or blocked:"
        in errors
    )
    assert "model_specs must be omitted for no-LLM substrate" in errors
    assert "target_model must be omitted for no-LLM substrate" in errors
    assert "offline_bfs_used must be false" in errors
    assert "game_source_read must be false" in errors
    assert "hand_built_per_game_adapter_used must be false" in errors
    assert "registry_updated requires offline_reproduced true" in registry_update_errors
    assert "offline_reproduced true requires registry_delta == reproduced_levels" in gate_errors
    assert "offline_reproduced must be bare bool" in bare_type_errors
    assert "reproduced_levels must be bare int" in bare_type_errors
    assert "registry_delta must be bare int" in bare_type_errors
    assert "offline_bfs_used must be bare bool" in bare_type_errors
    assert "attempts must be non-negative" in negative_field_errors
    assert "reproduced_levels must be non-negative" in negative_field_errors
    assert "repeated_coordinate_rate must be bare float" in repeated_field_type_errors
    with pytest.raises(ValueError):
        exp5562.validate_artifact(invalid)


def test_scenario_arc_fcp_5562_run_experiment_writes_honest_null_without_registry_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5562: no-bank FSM live attempt writes result and trajectory."""

    root = _tmp_ready_root(tmp_path)

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["target"]["selected_game"] == "r11l"
        assert kwargs["target"]["target_level"] == 3
        assert kwargs["budget"] == 3
        assert kwargs["random_seed"] == 5561
        return _null_attempt()

    artifact = exp5562.run_experiment(
        root=root,
        budget=3,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5562 null"],
    )
    written = json.loads((root / exp5562.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    log = json.loads((root / exp5562.TRAJECTORY_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((root / exp5562.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["registry_delta"] == 0
    assert artifact["trajectory_path"] == exp5562.TRAJECTORY_RELATIVE_PATH
    assert log["selected_game"] == "r11l"
    assert log["attempt_budget"] == 3
    assert registry["reproducible_total_levels"] == 69
    assert registry["games"][1]["levels_reproduced"] == 2


def test_scenario_arc_fcp_5562_run_experiment_updates_registry_only_on_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5562: reproduced FSM live trajectory updates registry once."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5562.run_experiment(
        root=root,
        budget=4,
        attempt_runner=lambda **_kwargs: _success_attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5562 success"],
    )
    registry = yaml.safe_load((root / exp5562.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["registry_delta"] == 1
    assert registry["reproducible_total_levels"] == 70
    assert registry["games"][1]["levels_reproduced"] == 3
    assert registry["games"][1]["latest_exp5562_fsm_live_levelup"]["artifact"] == (
        exp5562.RESULT_RELATIVE_PATH
    )


def test_req_arc_fcp_5562_blocked_duplicate_and_helper_edges(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5562: blocked and defensive helper paths stay no-credit."""

    missing = exp5562.run_experiment(root=tmp_path, tests_run=["missing 5562"])
    duplicate_root = _tmp_ready_root(tmp_path / "duplicate", registry=_registry(r11l_levels=3))
    duplicate = exp5562.run_experiment(
        root=duplicate_root,
        attempt_runner=lambda **_kwargs: pytest.fail("duplicate gate must skip runtime"),
        tests_run=["duplicate 5562"],
    )
    no_harness_root = _tmp_ready_root(tmp_path / "no-harness")
    no_harness = exp5562.run_experiment(
        root=no_harness_root,
        offline_arcade_check=lambda: False,
        tests_run=["no harness 5562"],
    )
    not_adjacent = exp5562.select_target_from_precheck(_precheck(level="L4"), _registry())
    artifact = exp5562.build_artifact(
        target=_target(),
        attempt={**_success_attempt(), "offline_bfs_used": True},
        registry_updated=True,
        trajectory_path="results/blocked-credit-5562.json",
        precheck=_precheck(),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
        attempt_budget=4,
    )
    shallow_delta = exp5562.accepted_reproduced_levels(
        _target(),
        {**_success_attempt(), "post_levels_reproduced": 2},
    )
    new_game_artifact = {
        **artifact,
        "selected_game": "zz99",
        "selected_level": "L1",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "registry_delta": 1,
        "post_levels_reproduced": 1,
        "prior_levels_reproduced": 0,
    }

    assert missing["status"] == "blocked"
    assert missing["selected_game"] == ""
    assert "exp5561_precheck_missing" in missing["honest_verdict"]
    assert duplicate["status"] == "duplicate_prevented"
    assert duplicate["attempts"] == 0
    assert duplicate["registry_delta"] == 0
    assert no_harness["status"] == "blocked"
    assert "missing_harness_access" in no_harness["honest_verdict"]
    assert not_adjacent["blocker"] == "exp5561_target_not_adjacent_frontier"
    assert artifact["offline_reproduced"] is False
    assert artifact["registry_delta"] == 0
    assert shallow_delta == 0
    assert exp5562.parse_level_label("3") == 3
    assert exp5562.row_signature({"action": 6, "data": {"x": "1", "y": "2"}}) == "A6@1,2"
    assert exp5562.row_signature({"action": 6, "x": "3", "y": "4"}) == "A6@3,4"
    assert exp5562.row_signature({"action": 5}) == "A5"
    assert exp5562.read_json(tmp_path / "missing.json") == {}
    assert exp5562.read_yaml(tmp_path / "missing.yaml") == {
        "reproducible_total_levels": 0,
        "games": [],
    }
    assert exp5562.update_registry_if_banked(
        root=tmp_path,
        artifact=artifact,
        registry=_registry(),
    ) is False
    assert exp5562.update_registry_if_banked(
        root=tmp_path / "new-game",
        artifact=new_game_artifact,
        registry={"reproducible_total_levels": 0, "games": []},
    ) is True
