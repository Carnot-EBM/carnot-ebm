"""Tests for Exp5507 ARC null-coordinate and perception precheck.

Spec refs: REQ-ARC-FCP-5507,
SCENARIO-ARC-FCP-5507.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5507_arc_null_coordinate_perception_precheck_v499 as exp5507


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(*, dc22_level: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": dc22_level},
            {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _exp5493() -> dict[str, Any]:
    return {
        "selected_game": "dc22",
        "selected_target_level": 3,
        "prior_levels_reproduced": 2,
        "proposed_live_mechanism": (
            "E3AgentPolicy + LiveCoExLandmarkFrontierGenerator option induction "
            "over visible toggle-navigation action effects"
        ),
        "excluded_recent_no_bank_targets": ["sb26:L3", "bp35:L3", "ka59:L2"],
        "levels_reproduced_by_candidate_game": {"bp35": 2, "sb26": 2, "dc22": 2},
    }


def _exp5494() -> dict[str, Any]:
    receipts = [
        {
            "receipt_id": "m0001",
            "action": 6,
            "data": {"x": 24, "y": 20},
            "changed_cells": 1,
            "before_hash": "a",
            "after_hash": "b",
        },
        {
            "receipt_id": "m0002",
            "action": 6,
            "data": {"x": 24, "y": 20},
            "changed_cells": 0,
            "before_hash": "b",
            "after_hash": "b",
        },
        {
            "receipt_id": "m0003",
            "action": 6,
            "data": {"x": 10, "y": 40},
            "changed_cells": 1,
            "before_hash": "b",
            "after_hash": "c",
        },
        {
            "receipt_id": "m0004",
            "action": 6,
            "data": {"x": 10, "y": 40},
            "changed_cells": 0,
            "before_hash": "c",
            "after_hash": "c",
        },
    ]
    return {
        "selected_game": "dc22",
        "target_level": 3,
        "prior_levels_reproduced": 2,
        "post_levels_reproduced": 2,
        "new_level_banked": False,
        "offline_reproduced": False,
        "status": "honest_null",
        "failure_mode": "bounded_budget_no_target_level_reproduction",
        "attempt": {
            "measurement_access_receipts": receipts,
            "action_history_clusters": [
                {
                    "sequence": [{"action": 6, "data": {"x": 10, "y": 40}}],
                    "changed_cells": 1,
                    "support_count": 2,
                },
                {
                    "sequence": [{"action": 6, "data": {"x": 24, "y": 20}}],
                    "changed_cells": 1,
                    "support_count": 2,
                },
            ],
        },
        "verifier_checks": [
            {
                "action": 6,
                "data": {"x": 10, "y": 40},
                "effect_count": 1,
                "effect_rate": 0.25,
                "salience_route": "blob_tier_0_button_like",
                "support_count": 4,
                "accepted": False,
            },
            {
                "action": 6,
                "data": {"x": 24, "y": 20},
                "effect_count": 3,
                "effect_rate": 0.75,
                "salience_route": "blob_tier_0_button_like",
                "support_count": 4,
                "accepted": True,
            },
        ],
        "target_selection": {
            "recent_no_bank_targets": ["sb26:L3", "bp35:L3", "ka59:L2"],
        },
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "METHODOLOGY_MISSING"}],
    }


def _exp5464() -> dict[str, Any]:
    return {
        "null_coordinate_audit": {
            "checked_reproduced_loop_artifacts": 22,
            "contaminated_artifacts": [],
            "null_coordinate_exploit_valid": False,
        },
        "metric_integrity_probe_receipts": {
            "duplicate_probe": {
                "game": "r11l",
                "target_level": 2,
                "registry_depth": 2,
                "duplicate_rejected": True,
                "rejection_reasons": ["duplicate_depth"],
            }
        },
    }


def _salience_artifact(game: str) -> dict[str, Any]:
    return {
        "target_game": game,
        "target_level_attempted": 3,
        "target_level_before": 2,
        "new_level_banked": False,
        "feature_receipts": {
            "connected_component_rows": [{"color": 9, "pixel_count": 4}],
            "color_blob_rows": [{"color": 9, "tier": 0, "button_like": True}],
            "changed_pixel_rows": [{"x": 14, "y": 14, "before": 9, "after": 10}],
            "salience_tier_rows": [{"tier": 0, "color": 9}],
            "action_effect_observations": [
                {"action": 6, "data": {"x": 14, "y": 14}, "changed_pixels": 4}
            ],
        },
    }


def test_req_arc_fcp_5507_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5507: OpenSpec anchors the no-solve artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5507" in spec
    assert "SCENARIO-ARC-FCP-5507" in spec
    assert exp5507.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5507.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle in spec


def test_scenario_arc_fcp_5507_audits_null_coordinates_as_valid_actions() -> None:
    """SCENARIO-ARC-FCP-5507: no-op receipts are not missing/null coordinates."""

    audit = exp5507.audit_null_coordinates(_exp5494(), _exp5464())

    assert audit["receipts_checked"] == 4
    assert audit["valid_coordinate_receipts"] == 4
    assert audit["null_or_missing_coordinate_receipts"] == 0
    assert audit["zero_change_receipts"] == 2
    assert audit["metric_artifact_detected"] is False
    assert audit["null_coordinate_exploit_valid"] is False
    assert audit["prior_noop_behaviors_valid_game_actions"] is True
    assert audit["coordinate_effect_summary"]["24,20"]["changed_receipts"] == 1
    assert audit["coordinate_effect_summary"]["10,40"]["zero_change_receipts"] == 1


def test_scenario_arc_fcp_5507_malformed_receipts_block_null_audit() -> None:
    """SCENARIO-ARC-FCP-5507: missing coordinates are metric artifacts, not actions."""

    malformed_exp5494 = {
        "attempt": {
            "measurement_access_receipts": [
                [],
                {"receipt_id": "bad-data", "data": "not-a-coordinate", "changed_cells": 0},
                {"receipt_id": "missing-x", "data": {"x": None, "y": 1}, "changed_cells": 0},
            ]
        }
    }

    audit = exp5507.audit_null_coordinates(malformed_exp5494, _exp5464())
    blocked = exp5507.build_precheck(
        exp5507.UpstreamEvidence(
            registry=_registry(),
            exp5493=_exp5493(),
            exp5494={
                **_exp5494(),
                "attempt": malformed_exp5494["attempt"],
            },
            exp5464=_exp5464(),
            prior_salience=[_salience_artifact("bp35")],
        )
    )

    assert audit["receipts_checked"] == 3
    assert audit["valid_coordinate_receipts"] == 0
    assert audit["null_or_missing_coordinate_receipts"] == 2
    assert audit["metric_artifact_detected"] is True
    assert audit["prior_noop_behaviors_valid_game_actions"] is False
    assert audit["verdict"] == "blocked_null_coordinate_metric_artifact"
    assert blocked["levelup_attempt_ready"] is False
    assert "null_coordinate_audit_not_clean" in blocked["honest_verdict"]


def test_scenario_arc_fcp_5507_selects_dc22_with_materially_changed_mechanism() -> None:
    """SCENARIO-ARC-FCP-5507: dc22 L3 is eligible only through changed perception."""

    evidence = exp5507.UpstreamEvidence(
        registry=_registry(),
        exp5493=_exp5493(),
        exp5494=_exp5494(),
        exp5464=_exp5464(),
        prior_salience=[_salience_artifact("bp35"), _salience_artifact("sb26")],
    )
    precheck = exp5507.build_precheck(evidence)

    assert precheck["selected_game"] == "dc22"
    assert precheck["selected_level"] == "L3"
    assert "connected-component/color-blob" in precheck["selected_mechanism"]
    assert precheck["levelup_attempt_ready"] is True
    assert precheck["solve_claimed"] is False
    assert precheck["inference_substrate"] == exp5507.INFERENCE_SUBSTRATE
    assert precheck["duplicate_targets_rejected"][0]["target"] == "r11l:L2"
    assert any(row["target"] == "dc22:L3" for row in precheck["recent_no_bank_targets_rejected"])
    assert len(precheck["perception_grounding_findings"]) >= 3
    exp5507.validate_artifact(precheck)


def test_scenario_arc_fcp_5507_blocks_duplicate_selected_target() -> None:
    """SCENARIO-ARC-FCP-5507: already reproduced selected levels fail closed."""

    evidence = exp5507.UpstreamEvidence(
        registry=_registry(dc22_level=3),
        exp5493=_exp5493(),
        exp5494=_exp5494(),
        exp5464=_exp5464(),
        prior_salience=[_salience_artifact("bp35")],
    )
    precheck = exp5507.build_precheck(evidence)

    assert precheck["selected_game"] == ""
    assert precheck["selected_level"] == ""
    assert precheck["selected_mechanism"] == ""
    assert precheck["levelup_attempt_ready"] is False
    assert precheck["solve_claimed"] is False
    assert precheck["honest_verdict"].startswith("blocked:")
    assert any(row["target"] == "dc22:L3" for row in precheck["duplicate_targets_rejected"])
    exp5507.validate_artifact(precheck)


def test_scenario_arc_fcp_5507_schema_rejects_bad_required_fields() -> None:
    """SCENARIO-ARC-FCP-5507: schema rejects solve claims and wrong substrates."""

    evidence = exp5507.UpstreamEvidence(
        registry=_registry(),
        exp5493=_exp5493(),
        exp5494=_exp5494(),
        exp5464=_exp5464(),
        prior_salience=[_salience_artifact("bp35")],
    )
    artifact = exp5507.build_precheck(evidence)
    invalid = {
        **artifact,
        "registry_path": "ops/wrong.yaml",
        "reproducible_total_levels_before": "69",
        "duplicate_targets_rejected": {},
        "recent_no_bank_targets_rejected": {},
        "null_coordinate_audit": [],
        "perception_grounding_findings": {},
        "selected_game": 7,
        "selected_level": 3,
        "selected_mechanism": 8,
        "levelup_attempt_ready": "true",
        "solve_claimed": True,
        "inference_substrate": "arc_live_agent_self_discovery",
        "honest_verdict": "complete: solved dc22 L3",
    }
    ready_invalid = {
        **artifact,
        "selected_game": "",
        "selected_level": "",
        "selected_mechanism": "",
        "perception_grounding_findings": [],
        "levelup_attempt_ready": True,
        "honest_verdict": "pending",
    }

    errors = exp5507.artifact_schema_errors(invalid)
    ready_errors = exp5507.artifact_schema_errors(ready_invalid)

    assert "registry_path must be ops/arc_solve_registry.yaml" in errors
    assert "reproducible_total_levels_before must be bare int" in errors
    assert "duplicate_targets_rejected must be a list" in errors
    assert "recent_no_bank_targets_rejected must be a list" in errors
    assert "null_coordinate_audit must be a dict" in errors
    assert "perception_grounding_findings must be a list" in errors
    assert "selected_game must be a string" in errors
    assert "selected_level must be a string" in errors
    assert "selected_mechanism must be a string" in errors
    assert "levelup_attempt_ready must be bare bool" in errors
    assert "solve_claimed must be false" in errors
    assert "inference_substrate must be aggregation_from_upstream_artifacts" in errors
    assert "honest_verdict must not claim a solve" in errors
    assert "honest_verdict must start with complete: or blocked:" in ready_errors
    assert "levelup_attempt_ready requires selected_game" in ready_errors
    assert "levelup_attempt_ready requires selected_level" in ready_errors
    assert "levelup_attempt_ready requires selected_mechanism" in ready_errors
    assert "levelup_attempt_ready requires perception_grounding_findings" in ready_errors
    with pytest.raises(ValueError):
        exp5507.validate_artifact(invalid)


def test_scenario_arc_fcp_5507_run_experiment_writes_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5507: runner writes the required deliverable JSON."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "docs" / "research-notes").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("# CLAUDE\nARC\n", encoding="utf-8")
    (root / exp5507.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5507\nSCENARIO-ARC-FCP-5507\n",
        encoding="utf-8",
    )
    (root / exp5507.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5507.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "ARC standing floor, null-coordinate and perception grounding.",
        encoding="utf-8",
    )
    (root / exp5507.LEVERS_NOTE_RELATIVE_PATH).write_text(
        "Existing lever verdicts; no new verdict from precheck.",
        encoding="utf-8",
    )
    (root / exp5507.EXP5493_RELATIVE_PATH).write_text(json.dumps(_exp5493()), encoding="utf-8")
    (root / exp5507.EXP5494_RELATIVE_PATH).write_text(json.dumps(_exp5494()), encoding="utf-8")
    (root / exp5507.EXP5464_RELATIVE_PATH).write_text(json.dumps(_exp5464()), encoding="utf-8")
    (root / exp5507.EXP5465_RELATIVE_PATH).write_text(
        json.dumps(_salience_artifact("bp35")),
        encoding="utf-8",
    )
    (root / exp5507.EXP5480_RELATIVE_PATH).write_text(
        json.dumps(_salience_artifact("sb26")),
        encoding="utf-8",
    )

    artifact = exp5507.run_experiment(root=root, tests_run=["unit 5507"])
    written = json.loads((root / exp5507.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "dc22"
    assert artifact["selected_level"] == "L3"
    assert artifact["levelup_attempt_ready"] is True
    assert artifact["solve_claimed"] is False
    assert artifact["tests_run"] == ["unit 5507"]


def test_scenario_arc_fcp_5507_missing_upstream_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5507: absent upstream files produce exact blocked reasons."""

    artifact = exp5507.run_experiment(root=tmp_path, tests_run=["missing upstream"])

    assert artifact["status"] == "blocked"
    assert artifact["selected_game"] == ""
    assert artifact["selected_level"] == ""
    assert artifact["levelup_attempt_ready"] is False
    assert artifact["solve_claimed"] is False
    assert "missing_selected_target" in artifact["honest_verdict"]
    assert "no_perception_grounding_findings" in artifact["honest_verdict"]
    assert "recent_no_bank_audit_missing" in artifact["honest_verdict"]
    assert (tmp_path / exp5507.RESULT_RELATIVE_PATH).exists()
