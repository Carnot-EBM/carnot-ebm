"""Tests for Exp 4610 world-model trust pass-rate metric helper.

Spec refs: REQ-ARC-WMTE-4610, SCENARIO-ARC-WMTE-4610.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _a1_artifact(rows: list[dict[str, Any]], baseline: float = 0.0) -> dict[str, Any]:
    return {
        "honest_verdict": "success: world_model_trust_energy_pass_rate_up_2_first_win_up",
        "world_model_trust_pass_rate_binary": baseline,
        "first_win_rate_new": 0.5,
        "median_actions_to_first_levelup_new": 11.0,
        "measurements": rows,
    }


def _row(
    game: str,
    *,
    trust_pass: bool,
    planner_used: bool,
    candidate: str = "change_weighted_partial",
    correct_changed_cells: int = 1,
) -> dict[str, Any]:
    return {
        "game": game,
        "new_trust_pass": trust_pass,
        "new_planner_used": planner_used,
        "new_selected_candidate_name": candidate,
        "new_correct_changed_cells": correct_changed_cells,
    }


def test_req_arc_wmte_4610_spec_declares_metric_contract() -> None:
    """REQ-ARC-WMTE-4610: OpenSpec declares the explicit numerator contract."""

    from carnot import experiment_4610_world_model_trust_pass_rate_metric as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4610" in spec
    assert "SCENARIO-ARC-WMTE-4610" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4610_synthetic_k_over_n_metric_is_explicit() -> None:
    """SCENARIO-ARC-WMTE-4610: k/n is reported with numerator and denominator."""

    from carnot import experiment_4610_world_model_trust_pass_rate_metric as mod

    metric = mod.compute_world_model_trust_pass_rate(
        _a1_artifact(
            [
                _row("g1", trust_pass=True, planner_used=True),
                _row("g2", trust_pass=True, planner_used=False),
                _row("g3", trust_pass=False, planner_used=True),
                _row("g4", trust_pass=True, planner_used=True),
            ],
            baseline=0.25,
        )
    )

    assert metric["trust_pass_numerator"] == 2
    assert metric["trust_pass_denominator"] == 4
    assert metric["world_model_trust_pass_rate"] == pytest.approx(0.5)
    assert metric["world_model_trust_pass_rate_baseline"] == pytest.approx(0.25)
    assert metric["world_model_trust_pass_rate_delta"] == pytest.approx(0.25)
    assert metric["passed_games"] == ["g1", "g4"]
    assert metric["tried_games"] == ["g1", "g2", "g3", "g4"]


def test_scenario_arc_wmte_4610_identity_pass_is_not_counted() -> None:
    """SCENARIO-ARC-WMTE-4610: degenerate identity rows cannot inflate numerator."""

    from carnot import experiment_4610_world_model_trust_pass_rate_metric as mod

    metric = mod.compute_world_model_trust_pass_rate(
        _a1_artifact(
            [
                _row(
                    "degenerate",
                    trust_pass=True,
                    planner_used=True,
                    candidate="identity",
                    correct_changed_cells=0,
                ),
                _row("genuine", trust_pass=True, planner_used=True),
            ]
        )
    )

    assert metric["trust_pass_numerator"] == 1
    assert metric["trust_pass_denominator"] == 2
    assert metric["world_model_trust_pass_rate"] == pytest.approx(0.5)
    assert metric["excluded_degenerate_identity_games"] == ["degenerate"]
    assert metric["passed_games"] == ["genuine"]


def test_req_arc_wmte_4610_baseline_equal_artifact_sets_null_delta_note() -> None:
    """REQ-ARC-WMTE-4610: baseline-equal trust rate emits null-delta methodology note."""

    from carnot import experiment_4610_world_model_trust_pass_rate_metric as mod

    a1_artifact = _a1_artifact(
        [
            _row("g1", trust_pass=True, planner_used=True),
            _row("g2", trust_pass=False, planner_used=False),
        ],
        baseline=0.5,
    )
    registry = {"reproducible_total_levels": 55}
    coheadline = mod.build_coheadline_block(
        a1_artifact=a1_artifact,
        registry=registry,
        live_submittable_artifact={"live_submittable_level_count": 54},
        action_efficiency_artifact={
            "generic_transfer_rate_over_variants": 0.04,
            "generic_transfer_ci": [0.0, 0.1],
            "action_efficiency_score": 1.0,
            "action_efficiency_ci": [1.0, 1.0],
            "median_actions_to_first_levelup": 20.0,
        },
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        a1_artifact=a1_artifact,
        registry=registry,
        coheadline_block=coheadline,
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert artifact["honest_verdict"] == (
        "success: world_model_trust_pass_rate_metric_helper_shipped_tests_green"
    )
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["world_model_trust_pass_rate"] == pytest.approx(0.5)
    assert artifact["trust_pass_numerator"] == 1
    assert artifact["trust_pass_denominator"] == 2
    assert artifact["coheadline_block"]["reported_side_by_side"] == [
        "world_model_trust_pass_rate",
        "reproducible_total_levels",
        "live_submittable_level_count",
        "first_win_rate",
        "generic_transfer_rate_over_variants",
        "action_efficiency_score",
    ]
    assert artifact["coheadline_block"]["reproducible_total_levels"] == 55
    assert artifact["coheadline_block"]["live_submittable_level_count"] == 54
    assert artifact["coheadline_block"]["generic_transfer_rate_over_variants"] == pytest.approx(0.04)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4610_schema_and_helper_defensive_branches() -> None:
    """REQ-ARC-WMTE-4610: helper rejects malformed schema without fabricating metrics."""

    from carnot import experiment_4610_world_model_trust_pass_rate_metric as mod

    assert mod._as_float(True, 7.5) == 7.5
    assert mod._as_float("not-a-float", 2.25) == 2.25
    assert mod._as_int(False, 8) == 8
    assert mod._as_int("not-an-int", 9) == 9
    assert mod._tests_added()["passed"] is True

    a1_artifact = _a1_artifact([_row("g1", trust_pass=True, planner_used=True)], baseline=0.0)
    registry = {"reproducible_total_levels": "55"}
    coheadline = mod.build_coheadline_block(a1_artifact=a1_artifact, registry=registry)
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        a1_artifact=a1_artifact,
        registry=registry,
        coheadline_block=coheadline,
        duration_s=0.001,
        tests_added=mod._tests_added(),
    )

    assert "null_delta_methodology_note" not in artifact
    assert artifact["registry_reproducible_total_levels"] == 55

    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["inference_substrate"] = "wrong"
    broken["trust_pass_numerator"] = "1"
    broken["trust_pass_denominator"] = -1
    broken["world_model_trust_pass_rate"] = 2.0
    broken["world_model_trust_pass_rate_delta"] = 0.0
    broken["coheadline_block"] = []
    broken.pop("tests_added")
    broken["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(broken)

    assert "missing required field tests_added" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "trust_pass_numerator" in errors
    assert "trust_pass_denominator" in errors
    assert "world_model_trust_pass_rate" in errors
    assert "null_delta_methodology_note" in errors
    assert "coheadline_block" in errors
    assert "reproducibility_checksum" in errors

    wrong_fraction = dict(artifact)
    wrong_fraction["world_model_trust_pass_rate"] = 0.25
    wrong_fraction["reproducibility_checksum"] = mod.payload_checksum(wrong_fraction)
    assert "world_model_trust_pass_rate_fraction" in mod.artifact_schema_errors(wrong_fraction)

    wrong_block = dict(artifact)
    wrong_block["coheadline_block"] = {"reported_side_by_side": []}
    wrong_block["reproducibility_checksum"] = mod.payload_checksum(wrong_block)
    assert "coheadline_block.reported_side_by_side" in mod.artifact_schema_errors(wrong_block)
