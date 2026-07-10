"""Tests for Exp5520 ARC action-diversity target precheck.

Spec refs: REQ-ARC-FCP-5520,
SCENARIO-ARC-FCP-5520.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5520_arc_action_diversity_target_precheck as exp5520
from carnot.agentic.arc_perception_generation import ActionDiversityPerceptionGenerator


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, sb26_levels: int = 2, total: int = 69) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": sb26_levels},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _exp5508() -> dict[str, Any]:
    steps = []
    for index, point in enumerate(
        [(24, 20), (10, 40), (24, 20), (19, 31), (24, 20), (10, 40), (24, 20)],
        start=1,
    ):
        steps.append(
            {
                "step": index,
                "action": 6,
                "data": {"x": point[0], "y": point[1]},
                "changed_cells": 1 if point == (24, 20) else 0,
                "failure_kind": "logical" if point == (24, 20) else "factual",
            }
        )
    return {
        "selected_game": "dc22",
        "selected_level": "L3",
        "target_level": 3,
        "status": "honest_null",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "arc_registry_delta": 0,
        "attempt": {
            "live_agent_attempts": len(steps),
            "trajectory_taxonomy_steps": steps,
            "perception_features_enabled": [
                "connected_components",
                "color_blobs",
                "sprite_overlays",
                "salient_motion",
                "action_affordances",
            ],
        },
    }


def _salience_artifact(*, game: str = "sb26", level: int = 3) -> dict[str, Any]:
    rows = [
        {"action": 6, "data": {"x": 19, "y": 58}, "score": 4000.0, "tier": 0},
        {"action": 6, "data": {"x": 19, "y": 58}, "score": 3999.0, "tier": 0},
        {"action": 6, "data": {"x": 27, "y": 58}, "score": 3998.0, "tier": 0},
        {"action": 6, "data": {"x": 35, "y": 58}, "score": 3997.0, "tier": 0},
        {"action": 6, "data": {"x": 43, "y": 58}, "score": 3996.0, "tier": 0},
    ]
    return {
        "target_game": game,
        "target_level_attempted": level,
        "new_level_banked": False,
        "attempt": {
            "salience_diagnostics": {
                "action_tier_rows": rows,
            }
        },
    }


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5520.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5520\nSCENARIO-ARC-FCP-5520\n",
        encoding="utf-8",
    )
    (root / exp5520.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5520.EXP5508_RELATIVE_PATH).write_text(
        json.dumps(_exp5508()),
        encoding="utf-8",
    )
    (root / exp5520.EXP5480_RELATIVE_PATH).write_text(
        json.dumps(_salience_artifact()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5520_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5520: OpenSpec anchors the precheck artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5520" in spec
    assert "SCENARIO-ARC-FCP-5520" in spec
    assert exp5520.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5520.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5520_diversity_generator_suppresses_repeated_coordinates() -> None:
    """SCENARIO-ARC-FCP-5520: live-path generator diversity is measurable."""

    rows = _salience_artifact()["attempt"]["salience_diagnostics"]["action_tier_rows"]
    generator = ActionDiversityPerceptionGenerator(max_candidates=4)

    selected = generator.prioritize_rows(
        rows + [{"action": 6, "data": {"x": 99, "y": 99}, "score": None, "tier": None}],
        avoid_coordinates={(24, 20), (10, 40)},
    )
    metrics = generator.diversity_metrics(selected, total_salience_candidates=4)

    assert [(row["data"]["x"], row["data"]["y"]) for row in selected] == [
        (19, 58),
        (27, 58),
        (35, 58),
        (43, 58),
    ]
    assert metrics["action_entropy"] == pytest.approx(2.0)
    assert metrics["repeated_coordinate_rate"] == pytest.approx(0.0)
    assert metrics["salience_coverage_rate"] == pytest.approx(1.0)
    assert "repeated_coordinate_suppression" in generator.change_receipts


def test_scenario_arc_fcp_5520_selects_registry_safe_rotated_target() -> None:
    """SCENARIO-ARC-FCP-5520: Exp5508 pattern is rejected before target readiness."""

    evidence = exp5520.PrecheckEvidence(
        registry=_registry(),
        exp5508=_exp5508(),
        candidate_artifacts=[_salience_artifact()],
    )

    artifact = exp5520.build_precheck(evidence, tests_run=["unit"], duration_s=0.1)

    exp5520.validate_artifact(artifact)
    assert artifact["registry_precheck_done"] is True
    assert artifact["selected_game"] == "sb26"
    assert artifact["selected_level"] == "L3"
    assert artifact["already_reproduced"] is False
    assert artifact["exp5508_pattern_reused"] is False
    assert artifact["action_entropy"] >= 2.0
    assert artifact["repeated_coordinate_rate"] == pytest.approx(0.0)
    assert artifact["salience_coverage_rate"] == pytest.approx(1.0)
    assert artifact["no_credit_probe_attempts"] == 4
    assert artifact["arc_levelup_candidate_ready"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["inference_substrate"] == "arc_live_precheck"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no solve" in artifact["honest_verdict"]


def test_scenario_arc_fcp_5520_blocks_when_only_candidates_are_duplicate_or_repeated() -> None:
    """SCENARIO-ARC-FCP-5520: duplicate or Exp5508-shaped probes fail closed."""

    duplicate = exp5520.build_precheck(
        exp5520.PrecheckEvidence(
            registry=_registry(sb26_levels=3),
            exp5508=_exp5508(),
            candidate_artifacts=[_salience_artifact()],
        )
    )
    repeated = exp5520.build_precheck(
        exp5520.PrecheckEvidence(
            registry=_registry(),
            exp5508=_exp5508(),
            candidate_artifacts=[
                _salience_artifact(game="dc22", level=3),
            ],
        )
    )

    assert duplicate["arc_levelup_candidate_ready"] is False
    assert duplicate["already_reproduced"] is True
    assert duplicate["honest_verdict"].startswith("blocked:")
    assert repeated["arc_levelup_candidate_ready"] is False
    assert repeated["exp5508_pattern_reused"] is True
    assert "exp5508_pattern_reused" in repeated["honest_verdict"]
    exp5520.validate_artifact(duplicate)
    exp5520.validate_artifact(repeated)


def test_req_arc_fcp_5520_schema_rejects_bad_required_fields() -> None:
    """REQ-ARC-FCP-5520: schema rejects malformed readiness and solve-credit drift."""

    artifact = exp5520.build_precheck(
        exp5520.PrecheckEvidence(
            registry=_registry(),
            exp5508=_exp5508(),
            candidate_artifacts=[_salience_artifact()],
        )
    )
    invalid = {
        **artifact,
        "registry_precheck_done": "true",
        "selected_game": 7,
        "selected_level": [],
        "already_reproduced": True,
        "exp5508_pattern_reused": True,
        "candidate_generator_changes": [],
        "action_entropy": "2.0",
        "repeated_coordinate_rate": 1.5,
        "salience_coverage_rate": -0.1,
        "no_credit_probe_attempts": "4",
        "arc_levelup_candidate_ready": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "offline_bfs",
        "honest_verdict": "solved sb26",
    }
    invalid_bool_and_negative = {
        **artifact,
        "already_reproduced": "false",
        "no_credit_probe_attempts": -1,
    }
    ready_missing_thresholds = {
        **artifact,
        "selected_game": "",
        "selected_level": "",
        "action_entropy": 0.0,
        "repeated_coordinate_rate": 1.0,
        "salience_coverage_rate": 0.0,
        "arc_levelup_candidate_ready": True,
    }

    errors = exp5520.artifact_schema_errors(invalid)
    bool_errors = exp5520.artifact_schema_errors(invalid_bool_and_negative)
    threshold_errors = exp5520.artifact_schema_errors(ready_missing_thresholds)

    assert "registry_precheck_done must be bare bool" in errors
    assert "selected_game must be a string" in errors
    assert "selected_level must be a string or int" in errors
    assert "ready artifacts require already_reproduced false" in errors
    assert "ready artifacts require exp5508_pattern_reused false" in errors
    assert "candidate_generator_changes must be a non-empty list" in errors
    assert "action_entropy must be bare float" in errors
    assert "repeated_coordinate_rate must be in [0, 1]" in errors
    assert "salience_coverage_rate must be in [0, 1]" in errors
    assert "no_credit_probe_attempts must be bare int" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "inference_substrate must be arc_live_precheck" in errors
    assert "honest_verdict must start with complete: or blocked:" in errors
    assert "already_reproduced must be bare bool" in bool_errors
    assert "no_credit_probe_attempts must be non-negative" in bool_errors
    assert "ready artifacts require selected_game" in threshold_errors
    assert "ready artifacts require selected_level" in threshold_errors
    assert "ready artifacts require action_entropy above threshold" in threshold_errors
    assert "ready artifacts require repeated_coordinate_rate below threshold" in threshold_errors
    assert "ready artifacts require salience_coverage_rate above threshold" in threshold_errors
    with pytest.raises(ValueError):
        exp5520.validate_artifact(invalid)


def test_req_arc_fcp_5520_helper_fallback_branches(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5520: helper fallbacks stay explicit and no-credit."""

    registry = _registry()
    feature_receipt_artifact = {
        "game": "bp35",
        "feature_receipts": {
            "color_blob_rows": [
                [],
                {"centroid_x": "12", "centroid_y": "13", "tier": "bad"},
            ]
        },
    }
    mixed_exp5508 = {
        "selected_game": "dc22",
        "target_level": "3",
        "attempt": {
            "trajectory_taxonomy_steps": [
                {"action": 6},
                {"action": 6, "x": "1", "y": "2"},
            ]
        },
    }

    assert exp5520._as_int("bad", 4) == 4  # noqa: SLF001
    assert exp5520._as_float("bad", 0.5) == 0.5  # noqa: SLF001
    assert exp5520._parse_level("3") == 3  # noqa: SLF001
    assert exp5520._parse_level("bad") == 0  # noqa: SLF001
    assert exp5520._read_yaml(tmp_path / "missing.yaml") == {  # noqa: SLF001
        "reproducible_total_levels": 0,
        "games": [],
    }
    assert exp5520._read_text(tmp_path / "missing.md") == ""  # noqa: SLF001
    assert exp5520._row_coordinate({"action": 6, "x": "1", "y": "2"}) == (1, 2)  # noqa: SLF001
    assert exp5520._row_coordinate({"action": 6}) is None  # noqa: SLF001
    assert exp5520._row_signature({"action": 6}) == "A6"  # noqa: SLF001
    assert exp5520._candidate_level({"game": "bp35"}, registry, "bp35") == 3  # noqa: SLF001

    fallback_rows = exp5520._candidate_rows(feature_receipt_artifact)  # noqa: SLF001
    pattern = exp5520.extract_exp5508_pattern(mixed_exp5508)
    no_candidate = exp5520.build_precheck(
        exp5520.PrecheckEvidence(
            registry=registry,
            exp5508=_exp5508(),
            candidate_artifacts=[{}, [], {"target_game": "", "target_level_attempted": 0}],
        )
    )
    below_threshold = exp5520.build_precheck(
        exp5520.PrecheckEvidence(
            registry=registry,
            exp5508=_exp5508(),
            candidate_artifacts=[
                {
                    "target_game": "bp35",
                    "target_level_attempted": 3,
                    "attempt": {
                        "salience_diagnostics": {
                            "action_tier_rows": [
                                {"action": 6, "data": {"x": 91, "y": 92}, "score": 1.0}
                            ]
                        }
                    },
                }
            ],
        )
    )

    assert fallback_rows == [
        {"action": 6, "data": {"x": 12, "y": 13}, "score": 3996.0, "tier": 4}
    ]
    assert exp5520._candidate_rows({"game": "bp35"}) == []  # noqa: SLF001
    assert pattern["selected_level"] == "L3"
    assert pattern["unique_coordinate_count"] == 1
    assert exp5520._runs_exp5508_pattern(  # noqa: SLF001
        game="bp35",
        level=3,
        pattern=pattern,
        probe_rows=[],
        repeated_coordinate_rate=0.0,
        action_entropy=2.0,
    ) is True
    assert no_candidate["honest_verdict"] == "blocked: no_candidate_salience_artifacts"
    assert below_threshold["honest_verdict"].endswith("diversity_probe_below_threshold")
    exp5520.validate_artifact(no_candidate)
    exp5520.validate_artifact(below_threshold)


def test_scenario_arc_fcp_5520_run_experiment_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5520: runner writes the stable no-credit artifact."""

    root = _tmp_ready_root(tmp_path)

    artifact = exp5520.run_experiment(root=root, tests_run=["unit 5520"])
    written = json.loads((root / exp5520.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "sb26"
    assert artifact["arc_levelup_candidate_ready"] is True
    assert artifact["tests_run"] == ["unit 5520"]
