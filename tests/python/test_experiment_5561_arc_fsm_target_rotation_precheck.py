"""Tests for Exp5561 ARC FSM target rotation precheck.

Spec refs: REQ-ARC-FCP-5561,
SCENARIO-ARC-FCP-5561.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5561_arc_fsm_target_rotation_precheck as exp5561


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, g50t_levels: int = 2, r11l_levels: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": g50t_levels},
            {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": r11l_levels},
            {"game": "ls20", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _exp5548_null(*, game: str = "g50t", level: str = "L3") -> dict[str, Any]:
    return {
        "experiment": "experiment_5548_arc_clean_live_levelup",
        "selected_game": game,
        "selected_level": level,
        "status": "honest_null",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "registry_delta": 0,
        "solve_provenance": "live_agent_self_discovery",
    }


def _evidence() -> exp5561.FsmPrecheckEvidence:
    return exp5561.FsmPrecheckEvidence(registry=_registry(), exp5548=_exp5548_null())


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5561.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5561\nSCENARIO-ARC-FCP-5561\n",
        encoding="utf-8",
    )
    (root / exp5561.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5561.EXP5548_RELATIVE_PATH).write_text(
        json.dumps(_exp5548_null()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5561_spec_declares_fsm_precheck_artifact_fields() -> None:
    """REQ-ARC-FCP-5561: OpenSpec anchors the FSM rotation precheck contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5561" in spec
    assert "SCENARIO-ARC-FCP-5561" in spec
    assert exp5561.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5561.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5561_rotates_away_from_exp5548_no_bank_target() -> None:
    """SCENARIO-ARC-FCP-5561: recent no-bank targets are rejected before selection."""

    selection = exp5561.select_target(_evidence())

    assert selection["blocked"] is False
    assert selection["selected_game"] == "r11l"
    assert selection["selected_level"] == "L3"
    assert selection["already_reproduced"] is False
    assert selection["registry_precheck_passed"] is True
    assert selection["recent_no_bank_targets_avoided"] == ["g50t:L3"]
    assert selection["target_audit"]["g50t:L3"]["decision"] == (
        "rejected_recent_no_bank_target"
    )
    assert selection["target_audit"]["r11l:L3"]["decision"] == "selected"


def test_req_arc_fcp_5561_target_selection_blocks_duplicates_and_missing_registry() -> None:
    """REQ-ARC-FCP-5561: duplicate, non-adjacent, and missing targets fail closed."""

    duplicate = exp5561.select_target(
        exp5561.FsmPrecheckEvidence(registry=_registry(r11l_levels=3), exp5548=_exp5548_null()),
        target_candidates=(("g50t", 3), ("r11l", 3), ("ls20", 3)),
    )
    blocked = exp5561.select_target(
        exp5561.FsmPrecheckEvidence(registry={"games": []}, exp5548=_exp5548_null()),
        target_candidates=(("missing", 1), ("g50t", 3)),
    )
    non_adjacent = exp5561.select_target(
        exp5561.FsmPrecheckEvidence(registry=_registry(), exp5548=_exp5548_null()),
        target_candidates=(("r11l", 4),),
    )

    assert duplicate["selected_game"] == "ls20"
    assert duplicate["target_audit"]["r11l:L3"]["decision"] == "rejected_already_reproduced"
    assert blocked["blocked"] is True
    assert blocked["registry_precheck_passed"] is False
    assert blocked["target_audit"]["missing:L1"]["decision"] == (
        "rejected_game_missing_from_registry"
    )
    assert non_adjacent["target_audit"]["r11l:L4"]["decision"] == (
        "rejected_not_adjacent_frontier"
    )
    assert exp5561._candidate_signature({"action": 6, "x": "3", "y": "4"}) == "A6@3,4"  # noqa: SLF001
    assert exp5561._candidate_signature({"action": 5}) == "A5"  # noqa: SLF001

    blocked_artifact = exp5561.build_precheck(
        exp5561.FsmPrecheckEvidence(registry={"games": []}, exp5548=_exp5548_null()),
        tests_run=["unit blocked 5561"],
        live_path_reachability={"ok": False, "checks": {"unit": False}},
    )
    exp5561.validate_artifact(blocked_artifact)
    assert blocked_artifact["status"] == "blocked"
    assert "fsm_action_abstraction_not_live_reachable" in blocked_artifact["honest_verdict"]
    assert "repeated_coordinate_suppression_not_enabled" in blocked_artifact["honest_verdict"]
    assert "action_entropy_precheck_below_threshold" in blocked_artifact["honest_verdict"]


def test_scenario_arc_fcp_5561_fsm_abstraction_suppresses_repeated_coordinates() -> None:
    """SCENARIO-ARC-FCP-5561: FSM routing is live-reachable and entropy-gated."""

    probe = exp5561.run_fsm_precheck_probe("r11l", "L3", random_seed=5561)
    artifact = exp5561.build_precheck(
        _evidence(),
        tests_run=["unit 5561"],
        duration_s=0.1,
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )

    exp5561.validate_artifact(artifact)
    assert probe["fsm_action_abstraction_ready"] is True
    assert probe["repeated_coordinate_suppression_enabled"] is True
    assert probe["action_entropy_precheck"] >= exp5561.MIN_ACTION_ENTROPY
    assert probe["suppressed_coordinate_count"] > 0
    assert len({row["fsm_phase"] for row in probe["selected_rows"]}) >= 3
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert artifact["selected_game"] == "r11l"
    assert artifact["selected_level"] == "L3"
    assert artifact["recent_no_bank_targets_avoided"] == ["g50t:L3"]
    assert artifact["fsm_action_abstraction_ready"] is True
    assert artifact["repeated_coordinate_suppression_enabled"] is True
    assert artifact["action_entropy_precheck"] >= exp5561.MIN_ACTION_ENTROPY
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["arc_fsm_precheck_ready"] is True
    assert artifact["inference_substrate"] == "arc_live_path_precheck_no_llm"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_specs" not in artifact
    assert "target_model" not in artifact


def test_req_arc_fcp_5561_schema_rejects_model_specs_and_bad_gate_fields() -> None:
    """REQ-ARC-FCP-5561: schema rejects substrate, target, and gate drift."""

    artifact = exp5561.build_precheck(
        _evidence(),
        tests_run=["unit 5561"],
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    invalid = {
        **artifact,
        "llm_invoked": True,
        "no_model_specs_required": False,
        "selected_game": 7,
        "selected_level": [],
        "registry_precheck_passed": "true",
        "already_reproduced": True,
        "recent_no_bank_targets_avoided": "g50t:L3",
        "fsm_action_abstraction_ready": "true",
        "repeated_coordinate_suppression_enabled": "true",
        "action_entropy_precheck": "2.0",
        "solve_provenance": "development_proxy",
        "arc_fsm_precheck_ready": True,
        "tests_added_or_reused": "unit",
        "field_principles": [],
        "inference_substrate": "live_llm_inference",
        "honest_verdict": "solved r11l",
        "reproducibility_checksum": "",
        "model_specs": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "target_model": "unsloth/gemma-4-26B-A4B-it-GGUF",
    }
    low_entropy_ready = {
        **artifact,
        "action_entropy_precheck": 0.0,
    }

    errors = exp5561.artifact_schema_errors(invalid)
    low_entropy_errors = exp5561.artifact_schema_errors(low_entropy_ready)

    assert "llm_invoked must be false" in errors
    assert "no_model_specs_required must be true" in errors
    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "registry_precheck_passed must be bare bool" in errors
    assert "ready artifacts require already_reproduced false" in errors
    assert "recent_no_bank_targets_avoided must be a list" in errors
    assert "fsm_action_abstraction_ready must be bare bool" in errors
    assert "repeated_coordinate_suppression_enabled must be bare bool" in errors
    assert "action_entropy_precheck must be bare float" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "tests_added_or_reused must be a non-empty list" in errors
    assert "field_principles must be a mapping" in errors
    assert "inference_substrate must be arc_live_path_precheck_no_llm" in errors
    assert "honest_verdict must start with complete: or blocked:" in errors
    assert "honest_verdict must not claim a solve" in errors
    assert "reproducibility_checksum must be a sha256 string" in errors
    assert "model_specs must be omitted for no-LLM substrate" in errors
    assert "target_model must be omitted for no-LLM substrate" in errors
    assert "ready artifacts require action_entropy_precheck above threshold" in low_entropy_errors

    ready_missing_target = {**artifact, "selected_game": ""}
    assert "ready artifacts require selected_game" in exp5561.artifact_schema_errors(
        ready_missing_target
    )
    with pytest.raises(ValueError):
        exp5561.validate_artifact(invalid)


def test_req_arc_fcp_5561_run_experiment_writes_stable_artifact(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5561: run_experiment emits a deterministic precheck artifact."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5561.run_experiment(
        root=root,
        tests_run=["unit 5561"],
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    written = json.loads((root / exp5561.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "r11l"
    assert artifact["selected_level"] == "L3"
    assert artifact["preconditions_checked"]["spec_has_req_5561"] is True
    assert artifact["preconditions_checked"]["llm_invoked"] is False
    assert artifact["preconditions_checked"]["offline_bfs_used"] is False
    assert artifact["preconditions_checked"]["game_source_read"] is False
    assert artifact["preconditions_checked"]["hand_built_per_game_adapter_used"] is False
    assert artifact["reproducibility_checksum"] == exp5561.compute_reproducibility_checksum(
        selected_game=artifact["selected_game"],
        selected_level=artifact["selected_level"],
        registry_total=artifact["registry_total_levels"],
        registry_depth=artifact["prior_levels_reproduced"],
        recent_no_bank_targets_avoided=artifact["recent_no_bank_targets_avoided"],
        action_entropy_precheck=artifact["action_entropy_precheck"],
    )
