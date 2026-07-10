"""Tests for Exp5547 ARC no-LLM substrate precheck.

Spec refs: REQ-ARC-FCP-5547,
SCENARIO-ARC-FCP-5547.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5547_arc_no_llm_substrate_precheck as exp5547


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, g50t_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2},
            {
                "game": "g50t",
                "reproducibility": "reproduced",
                "levels_reproduced": g50t_levels,
            },
            {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5547.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5547\nSCENARIO-ARC-FCP-5547\n",
        encoding="utf-8",
    )
    (root / exp5547.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5547_spec_declares_clean_no_llm_artifact_fields() -> None:
    """REQ-ARC-FCP-5547: OpenSpec anchors the clean precheck artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5547" in spec
    assert "SCENARIO-ARC-FCP-5547" in spec
    assert exp5547.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5547.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5547_selects_non_duplicate_no_llm_target() -> None:
    """SCENARIO-ARC-FCP-5547: target choice is registry-safe and model-free."""

    artifact = exp5547.build_precheck(
        exp5547.CleanArcPrecheckEvidence(registry=_registry()),
        tests_run=["unit 5547"],
        duration_s=0.1,
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )

    exp5547.validate_artifact(artifact)
    assert artifact["selected_game"] == "g50t"
    assert artifact["selected_level"] == "L3"
    assert artifact["registry_precheck_passed"] is True
    assert artifact["already_reproduced"] is False
    assert artifact["llm_strategy_proposer_used"] is False
    assert artifact["no_model_specs_required"] is True
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert artifact["random_seed"] == exp5547.DEFAULT_RANDOM_SEED
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["strategy_routing_live_path_reachable"] is True
    assert artifact["repeated_coordinate_suppression_enabled"] is True
    assert artifact["action_entropy_precheck"] >= exp5547.MIN_ACTION_ENTROPY
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert artifact["arc_clean_precheck_ready"] is True
    assert artifact["target_audit"]["g50t:L3"]["decision"] == "selected"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no solve claimed" in artifact["honest_verdict"]


def test_scenario_arc_fcp_5547_duplicate_candidate_rotates_before_readiness() -> None:
    """SCENARIO-ARC-FCP-5547: already-banked candidate levels are rejected."""

    artifact = exp5547.build_precheck(
        exp5547.CleanArcPrecheckEvidence(registry=_registry(g50t_levels=3)),
        tests_run=["unit 5547"],
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )

    exp5547.validate_artifact(artifact)
    assert artifact["selected_game"] == "lf52"
    assert artifact["selected_level"] == "L3"
    assert artifact["already_reproduced"] is False
    assert artifact["target_audit"]["g50t:L3"]["decision"] == "rejected_already_reproduced"
    assert artifact["target_audit"]["lf52:L3"]["decision"] == "selected"


def test_req_arc_fcp_5547_target_selection_blocks_without_clean_candidate() -> None:
    """REQ-ARC-FCP-5547: missing and non-adjacent candidates fail closed."""

    registry = {
        "reproducible_total_levels": 4,
        "games": [
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": 1}
        ],
    }
    selection = exp5547.select_target(
        exp5547.CleanArcPrecheckEvidence(registry=registry),
        target_candidates=(("missing", 1), ("g50t", 3)),
    )
    artifact = exp5547.build_precheck(
        exp5547.CleanArcPrecheckEvidence(registry={"games": []}),
        tests_run=["unit 5547"],
        live_path_reachability={"ok": False, "checks": {"unit": False}},
    )

    exp5547.validate_artifact(artifact)
    assert selection["blocked"] is True
    assert selection["target_audit"]["missing:L1"]["decision"] == (
        "rejected_game_missing_from_registry"
    )
    assert selection["target_audit"]["g50t:L3"]["decision"] == "rejected_not_adjacent_frontier"
    assert artifact["status"] == "blocked"
    assert artifact["selected_game"] == ""
    assert artifact["action_entropy_precheck"] == 0.0
    assert "strategy_routing_live_path_not_reachable" in artifact["honest_verdict"]
    assert exp5547._candidate_signature({"action": 5, "data": {}}) == "A5"


def test_req_arc_fcp_5547_schema_rejects_model_specs_and_bad_gate_fields() -> None:
    """REQ-ARC-FCP-5547: schema rejects substrate/provenance and model-spec drift."""

    artifact = exp5547.build_precheck(
        exp5547.CleanArcPrecheckEvidence(registry=_registry()),
        tests_run=["unit 5547"],
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    invalid = {
        **artifact,
        "selected_game": 7,
        "selected_level": [],
        "registry_precheck_passed": "true",
        "already_reproduced": True,
        "llm_strategy_proposer_used": True,
        "no_model_specs_required": False,
        "random_seed": "5547",
        "reproducibility_checksum": "",
        "strategy_routing_live_path_reachable": "true",
        "repeated_coordinate_suppression_enabled": "true",
        "action_entropy_precheck": "2.0",
        "solve_provenance": "development_proxy",
        "arc_clean_precheck_ready": True,
        "tests_added_or_reused": "unit",
        "field_principles": [],
        "inference_substrate": "live_llm_inference",
        "honest_verdict": "solved g50t",
        "model_specs": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "target_model": "unsloth/gemma-4-26B-A4B-it-GGUF",
    }

    errors = exp5547.artifact_schema_errors(invalid)

    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "registry_precheck_passed must be bare bool" in errors
    assert "ready artifacts require already_reproduced false" in errors
    assert "llm_strategy_proposer_used must be false" in errors
    assert "no_model_specs_required must be true" in errors
    assert "random_seed must be an int" in errors
    assert "reproducibility_checksum must be a sha256 string" in errors
    assert "strategy_routing_live_path_reachable must be bare bool" in errors
    assert "repeated_coordinate_suppression_enabled must be bare bool" in errors
    assert "action_entropy_precheck must be bare float" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "tests_added_or_reused must be a non-empty list" in errors
    assert "field_principles must be a mapping" in errors
    assert (
        "inference_substrate must be offline_arcade_live_agent_runtime_self_discovery_no_llm"
        in errors
    )
    assert "model_specs must be omitted for no-LLM substrate" in errors
    assert "target_model must be omitted for no-LLM substrate" in errors
    assert "honest_verdict must start with complete: or blocked:" in errors
    assert "honest_verdict must not claim a solve" in errors
    with pytest.raises(ValueError):
        exp5547.validate_artifact(invalid)

    ready_missing_target = {
        **artifact,
        "selected_game": "",
        "selected_level": "",
        "action_entropy_precheck": 0.0,
    }
    ready_missing_errors = exp5547.artifact_schema_errors(ready_missing_target)
    assert "ready artifacts require selected_game" in ready_missing_errors
    assert "ready artifacts require selected_level" in ready_missing_errors
    assert "ready artifacts require action_entropy_precheck above threshold" in ready_missing_errors


def test_req_arc_fcp_5547_run_experiment_writes_stable_artifact(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5547: run_experiment emits deterministic seed/checksum metadata."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5547.run_experiment(
        root=root,
        tests_run=["unit 5547"],
        live_path_reachability={"ok": True, "checks": {"unit": True}},
    )
    written = json.loads((root / exp5547.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["preconditions_checked"]["spec_has_req_5547"] is True
    assert artifact["preconditions_checked"]["llm_strategy_proposer_used"] is False
    assert artifact["preconditions_checked"]["model_specs_present"] is False
    assert artifact["reproducibility_checksum"] == exp5547.compute_reproducibility_checksum(
        selected_game=artifact["selected_game"],
        selected_level=artifact["selected_level"],
        random_seed=artifact["random_seed"],
        registry_total=artifact["registry_total_levels"],
        registry_depth=artifact["prior_levels_reproduced"],
        action_entropy_precheck=artifact["action_entropy_precheck"],
    )
