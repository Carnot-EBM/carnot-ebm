"""Tests for Exp5508 ARC live perception-generation level-up attempt.

Spec refs: REQ-ARC-FCP-5508,
SCENARIO-ARC-FCP-5508.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5508_arc_live_perception_generation_levelup_v499 as exp5508
from carnot.agentic import arc_perception_generation as perception
from carnot.agentic.arc_perception_generation import (
    ClassicalPerceptionGenerator,
    REQUIRED_PERCEPTION_FEATURES,
)


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, dc22_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {
                "game": "dc22",
                "reproducibility": "reproduced",
                "levels_reproduced": dc22_levels,
                "mechanic_class": "config_toggle_navigation",
            },
            {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _precheck(*, ready: bool = True, game: str = "dc22", level: str = "L3") -> dict[str, Any]:
    return {
        "levelup_attempt_ready": ready,
        "selected_game": game,
        "selected_level": level,
        "selected_mechanism": (
            "perception-grounded connected-component/color-blob segmentation plus "
            "salience-tiered action generation with trajectory taxonomy receipts"
        ),
        "reproducible_total_levels_before": 69,
        "solve_claimed": False,
    }


def _target() -> dict[str, Any]:
    return exp5508.select_target_from_precheck(_precheck(), _registry())


def _taxonomy_counts(**overrides: int) -> dict[str, int]:
    counts = {
        "factual": 1,
        "referential": 0,
        "logical": 2,
        "procedural": 0,
        "scope_based": 1,
    }
    counts.update(overrides)
    return counts


def _null_attempt() -> dict[str, Any]:
    return {
        "live_agent_attempts": 6,
        "runtime_observation_steps": 5,
        "post_levels_reproduced": 2,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "perception_features_enabled": list(REQUIRED_PERCEPTION_FEATURES),
        "trajectory_taxonomy_counts": _taxonomy_counts(),
        "trajectory_taxonomy_steps": [
            {"step": 1, "failure_kind": "factual", "changed_cells": 0},
            {"step": 2, "failure_kind": "logical", "changed_cells": 4},
        ],
        "candidate_action_count": 8,
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "offline_bfs_used": False,
        "game_source_read": False,
        "hand_built_per_game_adapter_used": False,
        "methodology_receipt": (
            "bounded_live_runtime budget=6 mechanism=classical_perception_generation "
            "gate=standard_reproduction prohibited_inputs=false"
        ),
    }


def _success_attempt() -> dict[str, Any]:
    return {
        **_null_attempt(),
        "live_agent_attempts": 7,
        "runtime_observation_steps": 7,
        "post_levels_reproduced": 3,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "trajectory_taxonomy_counts": _taxonomy_counts(factual=0, logical=1, scope_based=0),
        "trajectory_taxonomy_steps": [{"step": 7, "failure_kind": "", "level_after": 3}],
        "solution_labels": ['{"action":6,"data":{"x":24,"y":20}}'],
        "failure_mode": "",
        "reproduction_gate": {
            "game": "dc22",
            "claimed_level": 3,
            "reached_level": 3,
            "reproduced": True,
        },
    }


def _frame_pair() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    before = np.zeros((10, 10), dtype=np.int16)
    before[0, :] = 16
    before[2:8, 1:7] = 1
    before[6:8, 7:9] = 9
    before[3:5, 7:9] = 7
    after = before.copy()
    after[6:8, 7:9] = 10
    return (
        SimpleNamespace(frame=before, levels_completed=0, available_actions=[1, 6]),
        SimpleNamespace(frame=after, levels_completed=0, available_actions=[1, 6]),
        SimpleNamespace(frame=after, levels_completed=1, available_actions=[1, 6]),
    )


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "docs" / "research-notes").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5508.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5508\nSCENARIO-ARC-FCP-5508\n",
        encoding="utf-8",
    )
    (root / exp5508.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5508.PRECHECK_RELATIVE_PATH).write_text(
        json.dumps(_precheck()),
        encoding="utf-8",
    )
    (root / exp5508.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "ARC standing floor perception-generation attempt.",
        encoding="utf-8",
    )
    (root / exp5508.LEVERS_NOTE_RELATIVE_PATH).write_text(
        "Prior ARC levers recorded; Exp5508 is perception-generation.",
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5508_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5508: OpenSpec anchors the Exp5508 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5508" in spec
    assert "SCENARIO-ARC-FCP-5508" in spec
    assert exp5508.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5508.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5508_perception_generator_emits_runtime_features() -> None:
    """SCENARIO-ARC-FCP-5508: classical perception grounds candidate generation."""

    before, after, levelup = _frame_pair()
    generator = ClassicalPerceptionGenerator(max_candidates=5)
    fresh_generator = ClassicalPerceptionGenerator(max_candidates=5)

    features = generator.inspect(after, previous_frame=before)
    assert generator.for_path([]) is generator
    assert fresh_generator.click_points(after, max_points=1)
    assert fresh_generator.tier_rows(after)
    points = generator.click_points(after, max_points=3)
    candidates = [
        {"action": 6, "data": {"x": 7, "y": 6}, "source": "unit"},
        {"action": 1, "data": None, "source": "unit"},
        {"action": 6, "data": {}, "source": "unit"},
        SimpleNamespace(action_id=6, data={"x": 7, "y": 6}),
    ]
    action_rows = generator.action_tier_rows(after, candidates)
    sequence = generator.best_sequence(after, candidates, min_len=2)

    assert set(REQUIRED_PERCEPTION_FEATURES) == {
        "connected_components",
        "color_blobs",
        "sprite_overlays",
        "salient_motion",
        "action_affordances",
    }
    assert features["connected_component_rows"]
    assert features["color_blob_rows"]
    assert any(row["motion_overlap"] > 0 for row in features["sprite_overlay_rows"])
    assert features["motion_affordance_rows"][0]["changed_pixels"] == 4
    assert features["action_affordance_rows"][0]["action"] == 6
    assert points[0] == (7, 6)
    assert action_rows[0]["action"] == 6
    assert action_rows[-1]["score"] == 0.0
    assert len(sequence) == 2
    assert sequence[0]["action"] == 6
    generator.inspect(after, previous_frame=before)
    assert generator.score(after, {"action": 6, "data": {"x": 7, "y": 6}}) > 4000.0
    assert generator.score(after, {"action": 1, "data": None}) >= 0.0
    duplicate_generator = ClassicalPerceptionGenerator()
    duplicate_generator._last_features = {  # noqa: SLF001 - pins duplicate point stability.
        "action_affordance_rows": [
            {"x": 1, "y": 1},
            {"x": 1, "y": 1},
            {"x": 2, "y": 2},
        ]
    }
    assert duplicate_generator.click_points(after, max_points=2) == [(1, 1), (2, 2)]
    assert perception._as_int("bad", 4) == 4  # noqa: SLF001
    assert perception._overlap_count(np.zeros((0, 0), dtype=bool), [0, 0, 0, 0]) == 0  # noqa: SLF001

    generator.observe_transition(before, 6, {"x": 7, "y": 6}, before)
    generator.observe_transition(before, 6, None, after)
    generator.observe_transition(before, 6, {"x": 7, "y": 6}, after)
    generator.observe_transition(before, 6, {"x": 7, "y": 6}, SimpleNamespace(frame=np.zeros((3, 3))))
    generator.observe_transition(before, 6, {"x": 7, "y": 6}, levelup)
    generator.record_scope_failure("bounded_budget_no_target_level_reproduction")
    diagnostics = generator.diagnostics()

    assert diagnostics["perception_features_enabled"] == list(REQUIRED_PERCEPTION_FEATURES)
    assert diagnostics["runtime_observation_steps"] == 5
    assert diagnostics["candidate_generation_count"] >= 2
    assert diagnostics["trajectory_taxonomy_counts"] == {
        "factual": 1,
        "referential": 1,
        "logical": 1,
        "procedural": 1,
        "scope_based": 1,
    }
    assert generator.as_dict()["generation_stage_action_prioritization"] is True
    generator.reset(level=0)
    assert generator.diagnostics()["last_feature_counts"]["action_affordances"] == 0


def test_scenario_arc_fcp_5508_rechecks_precheck_target_and_blocks_duplicates() -> None:
    """SCENARIO-ARC-FCP-5508: registry is re-read before live budget is spent."""

    selected = exp5508.select_target_from_precheck(_precheck(), _registry())
    duplicate = exp5508.select_target_from_precheck(_precheck(), _registry(dc22_levels=3))
    missing = exp5508.select_target_from_precheck({}, _registry())
    not_ready = exp5508.select_target_from_precheck(_precheck(ready=False), _registry())
    missing_game = exp5508.select_target_from_precheck(_precheck(game=""), _registry())
    malformed = exp5508.select_target_from_precheck(_precheck(level="bad"), _registry())
    numeric_level = exp5508.select_target_from_precheck(_precheck(level="3"), _registry())

    assert selected["blocked"] is False
    assert selected["selected_game"] == "dc22"
    assert selected["selected_level"] == "L3"
    assert selected["target_level"] == 3
    assert selected["prior_levels_reproduced"] == 2
    assert selected["registry_before_levels"] == 69
    assert duplicate["blocked"] is True
    assert duplicate["blocker"] == "selected_level_already_reproducible"
    assert missing["blocker"] == "precheck_target_missing"
    assert not_ready["blocker"] == "precheck_target_not_ready"
    assert missing_game["blocker"] == "precheck_target_missing"
    assert malformed["blocker"] == "precheck_selected_level_malformed"
    assert numeric_level["selected_level"] == "L3"


def test_scenario_arc_fcp_5508_artifact_gates_null_and_success() -> None:
    """SCENARIO-ARC-FCP-5508: only reproduced new levels change registry totals."""

    null_artifact = exp5508.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    success = exp5508.build_artifact(
        target=_target(),
        attempt=_success_attempt(),
        registry_updated=True,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5508.validate_artifact(null_artifact)
    exp5508.validate_artifact(success)
    assert null_artifact["selected_game"] == "dc22"
    assert null_artifact["selected_level"] == "L3"
    assert null_artifact["registry_before_levels"] == 69
    assert null_artifact["registry_after_levels"] == 69
    assert null_artifact["arc_registry_delta"] == 0
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["solve_provenance"] == "live_agent_self_discovery"
    assert null_artifact["live_agent_attempts"] == 6
    assert null_artifact["runtime_observation_steps"] == 5
    assert null_artifact["offline_bfs_used"] is False
    assert null_artifact["game_source_read"] is False
    assert null_artifact["hand_built_per_game_adapter_used"] is False
    assert null_artifact["inference_substrate"] == exp5508.INFERENCE_SUBSTRATE
    assert null_artifact["honest_verdict"].startswith("honest_null:")
    assert "standard_reproduction" in null_artifact["methodology_receipt"]

    assert success["registry_after_levels"] == 70
    assert success["arc_registry_delta"] == 1
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["registry_updated"] is True
    assert success["honest_verdict"].startswith("complete:")


def test_scenario_arc_fcp_5508_schema_rejects_bad_required_fields() -> None:
    """REQ-ARC-FCP-5508: schema rejects off-path credit and malformed counts."""

    artifact = exp5508.build_artifact(
        target=_target(),
        attempt=_null_attempt(),
        registry_updated=False,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "selected_game": 7,
        "selected_level": 3,
        "registry_before_levels": "69",
        "registry_after_levels": 68,
        "arc_registry_delta": "0",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "solve_provenance": "development_proxy",
        "live_agent_attempts": "6",
        "runtime_observation_steps": -1,
        "perception_features_enabled": {},
        "trajectory_taxonomy_counts": [],
        "offline_bfs_used": True,
        "game_source_read": True,
        "hand_built_per_game_adapter_used": True,
        "methodology_receipt": "",
        "inference_substrate": "arc_live_agent_self_discovery",
        "honest_verdict": "solved",
    }

    errors = exp5508.artifact_schema_errors(invalid)

    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "registry_before_levels must be bare int" in errors
    assert "registry_after_levels must be >= registry_before_levels" in errors
    assert "arc_registry_delta must be bare int" in errors
    assert "offline_reproduced requires reproduced_levels >= 1" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "live_agent_attempts must be bare int" in errors
    assert "runtime_observation_steps must be non-negative" in errors
    assert "perception_features_enabled must be a list" in errors
    assert "trajectory_taxonomy_counts must be a dict" in errors
    assert "offline_bfs_used must be false" in errors
    assert "game_source_read must be false" in errors
    assert "hand_built_per_game_adapter_used must be false" in errors
    assert "methodology_receipt must be a non-empty string" in errors
    assert f"inference_substrate must be {exp5508.INFERENCE_SUBSTRATE}" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5508.validate_artifact(invalid)

    assert exp5508._accepted_reproduced_levels(  # noqa: SLF001
        _target(),
        {**_success_attempt(), "offline_bfs_used": True},
    ) == 0
    assert exp5508._accepted_reproduced_levels(  # noqa: SLF001
        _target(),
        {**_success_attempt(), "post_levels_reproduced": 2},
    ) == 0
    assert "arc_registry_delta must equal registry_after_levels - registry_before_levels" in (
        exp5508.artifact_schema_errors({**artifact, "arc_registry_delta": 1})
    )
    assert "registry_after_levels must be >= registry_before_levels" in exp5508.artifact_schema_errors(
        {**artifact, "registry_before_levels": 69, "registry_after_levels": 68}
    )
    assert "offline_reproduced must be bare bool" in exp5508.artifact_schema_errors(
        {**artifact, "offline_reproduced": "false"}
    )
    assert "offline_reproduced requires arc_registry_delta == reproduced_levels" in (
        exp5508.artifact_schema_errors(
            {
                **artifact,
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "registry_after_levels": 71,
                "arc_registry_delta": 2,
            }
        )
    )
    assert "perception_features_enabled missing required feature" in exp5508.artifact_schema_errors(
        {**artifact, "perception_features_enabled": ["connected_components"]}
    )
    assert "trajectory_taxonomy_counts.factual must be bare int" in exp5508.artifact_schema_errors(
        {**artifact, "trajectory_taxonomy_counts": {**_taxonomy_counts(), "factual": "1"}}
    )
    assert "offline_bfs_used must be bare bool" in exp5508.artifact_schema_errors(
        {**artifact, "offline_bfs_used": "false"}
    )
    assert "registry_updated requires offline_reproduced true" in exp5508.artifact_schema_errors(
        {**artifact, "registry_updated": True}
    )


def test_scenario_arc_fcp_5508_run_experiment_writes_honest_null_without_registry_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5508: no reproduced target level leaves registry unchanged."""

    root = _tmp_ready_root(tmp_path)

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["target"]["selected_game"] == "dc22"
        assert kwargs["target"]["target_level"] == 3
        assert kwargs["budget"] == 6
        return _null_attempt()

    artifact = exp5508.run_experiment(
        root=root,
        budget=6,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5508 null"],
    )
    written = json.loads((root / exp5508.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((root / exp5508.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["arc_registry_delta"] == 0
    assert artifact["registry_updated"] is False
    assert registry["reproducible_total_levels"] == 69
    assert registry["games"][0]["levels_reproduced"] == 2
    assert artifact["tests_run"] == ["unit 5508 null"]
    assert exp5508.update_registry_if_banked(
        root=root,
        artifact=artifact,
        registry=registry,
    ) is False


def test_scenario_arc_fcp_5508_run_experiment_updates_registry_only_on_banked_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5508: reproduced new-level success updates registry."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5508.run_experiment(
        root=root,
        budget=7,
        attempt_runner=lambda **_kwargs: _success_attempt(),
        offline_arcade_check=lambda: True,
        tests_run=["unit 5508 success"],
    )
    registry = yaml.safe_load((root / exp5508.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["arc_registry_delta"] == 1
    assert artifact["registry_updated"] is True
    assert registry["reproducible_total_levels"] == 70
    assert registry["games"][0]["levels_reproduced"] == 3
    assert registry["games"][0]["latest_exp5508_levelup_attempt"]["artifact"] == (
        exp5508.RESULT_RELATIVE_PATH
    )
    new_root = _tmp_ready_root(tmp_path / "new-game", registry={"reproducible_total_levels": 0, "games": []})
    new_artifact = {
        **artifact,
        "selected_game": "zz99",
        "post_levels_reproduced": 1,
        "registry_before_levels": 0,
        "registry_after_levels": 1,
        "arc_registry_delta": 1,
        "reproduced_levels": 1,
    }
    assert exp5508.update_registry_if_banked(
        root=new_root,
        artifact=new_artifact,
        registry={"reproducible_total_levels": 0, "games": []},
    ) is True


def test_scenario_arc_fcp_5508_blocked_preconditions_write_schema_valid_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5508: missing or duplicate targets fail closed."""

    missing = exp5508.run_experiment(root=tmp_path, tests_run=["missing 5508"])
    duplicate_root = _tmp_ready_root(tmp_path / "duplicate", registry=_registry(dc22_levels=3))
    duplicate = exp5508.run_experiment(
        root=duplicate_root,
        offline_arcade_check=lambda: True,
        tests_run=["duplicate 5508"],
    )
    no_harness_root = _tmp_ready_root(tmp_path / "no-harness")
    no_harness = exp5508.run_experiment(
        root=no_harness_root,
        offline_arcade_check=lambda: False,
        tests_run=["no harness 5508"],
    )

    assert missing["status"] == "blocked"
    assert missing["selected_game"] == ""
    assert missing["selected_level"] == ""
    assert missing["live_agent_attempts"] == 0
    assert "precheck_target_missing" in missing["honest_verdict"]
    assert duplicate["status"] == "blocked"
    assert duplicate["selected_game"] == ""
    assert duplicate["selected_level"] == ""
    assert duplicate["arc_registry_delta"] == 0
    assert "selected_level_already_reproducible" in duplicate["honest_verdict"]
    assert no_harness["status"] == "blocked"
    assert no_harness["perception_features_enabled"] == list(REQUIRED_PERCEPTION_FEATURES)
    assert "missing_harness_access" in no_harness["honest_verdict"]
