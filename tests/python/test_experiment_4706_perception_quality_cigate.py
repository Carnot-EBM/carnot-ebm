"""Tests for Exp 4706 perception-quality CI-gates.

Spec refs: REQ-ARC-WMTE-4706,
SCENARIO-ARC-WMTE-4706-LOO-DISCRIMINATION,
SCENARIO-ARC-WMTE-4706-OFFPATH-DISCRIMINATION,
SCENARIO-ARC-WMTE-4706-PERCEPTION-QUALITY-FLOOR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loo_rows(*, separable: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(("aa00", "bb00", "cc00")):
        offset = float(index) / 100.0
        if separable:
            rows.extend(
                [
                    {"game": game, "label": 1, "features": [2.0 + offset, 0.1]},
                    {"game": game, "label": 1, "features": [1.8 + offset, 0.2]},
                    {"game": game, "label": 0, "features": [0.2 + offset, 0.1]},
                    {"game": game, "label": 0, "features": [0.0 + offset, 0.2]},
                ]
            )
        else:
            rows.extend(
                [
                    {"game": game, "label": 1, "features": [0.5, 0.5]},
                    {"game": game, "label": 1, "features": [0.5, 0.5]},
                    {"game": game, "label": 0, "features": [0.5, 0.5]},
                    {"game": game, "label": 0, "features": [0.5, 0.5]},
                ]
            )
    return rows


def _offpath_rows(*, calibrated: bool) -> list[dict[str, Any]]:
    offpath_scores = (
        [(0.86, 1), (0.74, 1), (0.30, 0), (0.15, 0)]
        if calibrated
        else [(0.50, 1), (0.50, 1), (0.50, 0), (0.50, 0)]
    )
    rows = [
        {"split": "winning_path", "score": 0.95, "label": 1},
        {"split": "winning_path", "score": 0.82, "label": 1},
        {"split": "winning_path", "score": 0.24, "label": 0},
        {"split": "winning_path", "score": 0.10, "label": 0},
    ]
    rows.extend(
        {"split": "off_path_frontier", "score": score, "label": label}
        for score, label in offpath_scores
    )
    return rows


def test_req_arc_wmte_4706_spec_declares_perception_quality_contract() -> None:
    """REQ-ARC-WMTE-4706: OpenSpec anchors the gate fields and principles."""

    from carnot import experiment_4706_perception_quality_cigate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4706" in spec
    assert "SCENARIO-ARC-WMTE-4706-LOO-DISCRIMINATION" in spec
    assert "SCENARIO-ARC-WMTE-4706-OFFPATH-DISCRIMINATION" in spec
    assert "SCENARIO-ARC-WMTE-4706-PERCEPTION-QUALITY-FLOOR" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4706_loo_gate_fails_chance_and_passes_rich() -> None:
    """SCENARIO-ARC-WMTE-4706-LOO-DISCRIMINATION: chance fixtures fail."""

    from carnot import experiment_4706_perception_quality_cigate as mod

    chance = mod.compute_loo_discrimination(_loo_rows(separable=False))
    rich = mod.compute_loo_discrimination(_loo_rows(separable=True))
    failed = mod.validate_loo_discrimination_gate(
        order1_baseline_loo=mod.ORDER1_CHANCE_BASELINE_AUROC,
        richer_loo=chance["loo_auroc"],
    )
    passed = mod.validate_loo_discrimination_gate(
        order1_baseline_loo=mod.ORDER1_CHANCE_BASELINE_AUROC,
        richer_loo=rich["loo_auroc"],
    )

    assert chance["loo_auroc"] == pytest.approx(0.5)
    assert rich["loo_auroc"] == pytest.approx(1.0)
    assert failed["passed"] is False
    assert "richer_loo_at_or_near_chance" in failed["errors"]
    assert "richer_loo_below_target" in failed["errors"]
    assert passed["passed"] is True
    assert passed["order1_chance_baseline_loo_auroc"] == pytest.approx(0.503096)
    assert passed["richer_representation_loo_auroc"] == pytest.approx(1.0)
    with pytest.raises(mod.GateFailure, match="richer_loo_at_or_near_chance"):
        mod.assert_gate_passed(failed)


def test_scenario_arc_wmte_4706_offpath_metric_flags_large_gap() -> None:
    """SCENARIO-ARC-WMTE-4706-OFFPATH-DISCRIMINATION: off-path nulls are flagged."""

    from carnot import experiment_4706_perception_quality_cigate as mod

    large_gap = mod.validate_offpath_discrimination_metric(_offpath_rows(calibrated=False))
    calibrated = mod.validate_offpath_discrimination_metric(_offpath_rows(calibrated=True))

    assert large_gap["passed"] is False
    assert large_gap["winning_path_auroc"] == pytest.approx(1.0)
    assert large_gap["off_path_frontier_auroc"] == pytest.approx(0.5)
    assert large_gap["winning_path_vs_offpath_gap"] == pytest.approx(0.5)
    assert "offpath_auroc_at_or_near_chance" in large_gap["errors"]
    assert "winning_path_vs_offpath_gap_too_large" in large_gap["errors"]
    assert calibrated["passed"] is True
    assert calibrated["winning_path_vs_offpath_gap"] == pytest.approx(0.0)
    with pytest.raises(mod.GateFailure, match="winning_path_vs_offpath_gap_too_large"):
        mod.assert_gate_passed(large_gap)


def test_scenario_arc_wmte_4706_perception_quality_floor_flags_regression() -> None:
    """SCENARIO-ARC-WMTE-4706-PERCEPTION-QUALITY-FLOOR: LOO regressions fail."""

    from carnot import experiment_4706_perception_quality_cigate as mod

    regressed = mod.validate_perception_quality_floor(0.61, source="fixture_regression")
    honest = mod.validate_perception_quality_floor(
        mod.A1_ESTABLISHED_PERCEPTION_LOO_FLOOR,
        source="fixture_honest",
    )

    assert regressed["passed"] is False
    assert "perception_loo_below_a1_floor" in regressed["errors"]
    assert honest["passed"] is True
    assert honest["measured_loo_auroc"] == pytest.approx(mod.A1_ESTABLISHED_PERCEPTION_LOO_FLOOR)
    with pytest.raises(mod.GateFailure, match="perception_loo_below_a1_floor"):
        mod.assert_gate_passed(regressed)


def test_req_arc_wmte_4706_artifact_schema_and_run_write_terminal_json(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4706: terminal artifact is checksummed and schema-validated."""

    from carnot import experiment_4706_perception_quality_cigate as mod

    (tmp_path / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4706\n"
        "SCENARIO-ARC-WMTE-4706-LOO-DISCRIMINATION\n"
        "SCENARIO-ARC-WMTE-4706-OFFPATH-DISCRIMINATION\n"
        "SCENARIO-ARC-WMTE-4706-PERCEPTION-QUALITY-FLOOR\n",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={
            "ok": True,
            "offline_arcade": True,
            "spec_has_req_4706": True,
            "cached_loo_source_present": True,
            "live_offpath_source_present": True,
        },
        loo_rows=_loo_rows(separable=True),
        offpath_rows=_offpath_rows(calibrated=True),
        duration_s=1.0,
        write=True,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == (
        "success: perception_quality_loo_plus_offpath_cigate_shipped_tests_green"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["loo_discrimination_gate_added"]["passed"] is True
    assert artifact["offpath_discrimination_metric_added"]["passed"] is True
    assert artifact["perception_quality_floor_cigate_added"]["passed"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.0,
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["loo_discrimination_gate_added"]["passed"] is False
    assert mod.artifact_schema_errors(blocked) == []


def test_req_arc_wmte_4706_helper_edges_and_schema_errors_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4706: edge helpers fail closed with deterministic errors."""

    from carnot import experiment_4706_perception_quality_cigate as mod

    assert mod.tie_aware_auroc([0.1, 0.2], [1, 1]) == pytest.approx(0.5)
    assert mod.compute_loo_discrimination([])["loo_auroc"] == pytest.approx(0.5)
    assert mod._as_float("not-a-number", default=7.0) == pytest.approx(7.0)
    assert mod._row_features({"representations": {"rich": [1.25, 2.5]}}, "rich") == [
        1.25,
        2.5,
    ]
    assert mod._row_features({"score": 0.75}, "missing") == [0.75]
    assert mod._row_features({"features": 0.33}, "features") == [0.33]
    assert mod._mean_vector([], 2) == [0.0, 0.0]
    missing_offpath = mod.validate_offpath_discrimination_metric(
        [{"split": "winning_path", "score": 0.9, "label": 1}]
    )
    low_delta = mod.validate_loo_discrimination_gate(
        order1_baseline_loo=0.60,
        richer_loo=0.61,
        min_delta=0.05,
    )

    assert "offpath_rows_missing" in missing_offpath["errors"]
    assert "richer_loo_delta_too_small" in low_delta["errors"]

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        loo_discrimination_gate_added={"passed": True, "errors": []},
        offpath_discrimination_metric_added={"passed": False, "errors": ["gap"]},
        perception_quality_floor_cigate_added={"passed": True, "errors": []},
        tests_added={"passed": True},
        duration_s=1.0,
    )
    assert artifact["honest_verdict"] == "failed: perception_quality_cigate_failed"
    assert "offpath_discrimination_metric_added" in mod.artifact_schema_errors(artifact)

    broken = dict(artifact)
    broken["honest_verdict"] = "running"
    broken["inference_substrate"] = "live_llm_inference"
    broken["verifier_is_oracle"] = True
    broken["field_principles"] = {}
    broken["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "verifier_is_oracle_false" in errors
    assert "field_principles.honest_verdict" in errors
    assert "reproducibility_checksum" in errors

    malformed = dict(artifact)
    malformed["field_principles"] = []
    assert "field_principles" in mod.artifact_schema_errors(malformed)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])
    with pytest.raises(mod.GateFailure, match="forced_schema_error"):
        mod.run(
            root=tmp_path,
            preconditions_checked={"ok": True},
            loo_rows=_loo_rows(separable=True),
            offpath_rows=_offpath_rows(calibrated=True),
            duration_s=1.0,
            write=False,
        )
