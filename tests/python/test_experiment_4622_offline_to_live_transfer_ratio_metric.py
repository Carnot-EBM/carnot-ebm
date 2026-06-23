"""Tests for Exp 4622 offline-to-live transfer-ratio metric helper.

Spec refs: REQ-ARC-WMTE-4622, SCENARIO-ARC-WMTE-4622.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _offline_artifact(auroc: float = 0.72) -> dict[str, Any]:
    return {
        "experiment": "synthetic_offline_signal",
        "loo_auroc_mean": auroc,
        "loo_auroc_ci": [max(0.0, auroc - 0.05), min(1.0, auroc + 0.05)],
    }


def _live_artifact(
    *,
    graduated_first_win: float,
    linear_first_win: float,
    bare_first_win: float,
    graduated_actions: float = 20.0,
    linear_actions: float = 20.0,
    bare_actions: float = 20.0,
) -> dict[str, Any]:
    return {
        "first_win_rate_graduated": graduated_first_win,
        "first_win_rate_linear_baseline": linear_first_win,
        "first_win_rate_bare": bare_first_win,
        "median_actions_to_first_levelup_graduated": graduated_actions,
        "median_actions_to_first_levelup_linear_baseline": linear_actions,
        "median_actions_to_first_levelup_bare": bare_actions,
        "first_win_delta": graduated_first_win - linear_first_win,
        "actions_delta": linear_actions - graduated_actions,
        "matched_variant_signatures": ["aa00~color01", "bb00~color01"],
    }


def test_req_arc_wmte_4622_spec_declares_metric_contract() -> None:
    """REQ-ARC-WMTE-4622: OpenSpec declares the transfer-ratio artifact contract."""

    from carnot import experiment_4622_offline_to_live_transfer_ratio_metric as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4622" in spec
    assert "SCENARIO-ARC-WMTE-4622" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4622_high_offline_zero_live_lift_is_visible_gap() -> None:
    """SCENARIO-ARC-WMTE-4622: high offline AUROC plus zero live lift reports ratio zero."""

    from carnot import experiment_4622_offline_to_live_transfer_ratio_metric as mod

    metric = mod.compute_offline_to_live_transfer_ratio(
        _offline_artifact(0.72),
        _live_artifact(
            graduated_first_win=0.04,
            linear_first_win=0.04,
            bare_first_win=0.04,
            graduated_actions=20.0,
            linear_actions=20.0,
            bare_actions=20.0,
        ),
    )

    assert metric["offline_auroc_component"] == pytest.approx(0.72)
    assert metric["live_lift_component"] == pytest.approx(0.0)
    assert metric["offline_to_live_transfer_ratio"] == pytest.approx(0.0)
    assert metric["bridge_crossed"] is False
    assert metric["first_win_lift_component"] == pytest.approx(0.0)
    assert metric["action_efficiency_lift_component"] == pytest.approx(0.0)


def test_scenario_arc_wmte_4622_positive_first_win_lift_crosses_bridge() -> None:
    """SCENARIO-ARC-WMTE-4622: positive live first-win lift yields positive transfer ratio."""

    from carnot import experiment_4622_offline_to_live_transfer_ratio_metric as mod

    metric = mod.compute_offline_to_live_transfer_ratio(
        _offline_artifact(0.72),
        _live_artifact(
            graduated_first_win=0.20,
            linear_first_win=0.04,
            bare_first_win=0.04,
            graduated_actions=20.0,
            linear_actions=20.0,
            bare_actions=20.0,
        ),
    )

    assert metric["offline_auroc_component"] == pytest.approx(0.72)
    assert metric["first_win_lift_component"] == pytest.approx(0.16)
    assert metric["live_lift_component"] == pytest.approx(0.16)
    assert metric["offline_to_live_transfer_ratio"] == pytest.approx(0.16 / 0.72, abs=1e-6)
    assert metric["bridge_crossed"] is True


def test_scenario_arc_wmte_4622_positive_efficiency_lift_crosses_bridge() -> None:
    """SCENARIO-ARC-WMTE-4622: positive action efficiency also yields positive ratio."""

    from carnot import experiment_4622_offline_to_live_transfer_ratio_metric as mod

    metric = mod.compute_offline_to_live_transfer_ratio(
        _offline_artifact(0.72),
        _live_artifact(
            graduated_first_win=0.04,
            linear_first_win=0.04,
            bare_first_win=0.04,
            graduated_actions=10.0,
            linear_actions=20.0,
            bare_actions=30.0,
        ),
    )

    assert metric["first_win_lift_component"] == pytest.approx(0.0)
    assert metric["action_efficiency_lift_component"] == pytest.approx(0.5)
    assert metric["live_lift_component"] == pytest.approx(0.5)
    assert metric["offline_to_live_transfer_ratio"] == pytest.approx(0.5 / 0.72, abs=1e-6)
    assert metric["bridge_crossed"] is True


def test_req_arc_wmte_4622_baseline_equal_artifact_sets_null_delta_note() -> None:
    """REQ-ARC-WMTE-4622: baseline-equal live lift emits null-delta methodology note."""

    from carnot import experiment_4622_offline_to_live_transfer_ratio_metric as mod

    offline = _offline_artifact(0.72)
    live = _live_artifact(
        graduated_first_win=0.04,
        linear_first_win=0.04,
        bare_first_win=0.04,
        graduated_actions=20.0,
        linear_actions=20.0,
        bare_actions=20.0,
    )
    registry = {"reproducible_total_levels": 55}
    coheadline = mod.build_coheadline_block(
        offline_artifact=offline,
        live_artifact=live,
        registry=registry,
        live_submittable_artifact={"live_submittable_level_count": 54},
        action_efficiency_artifact={
            "action_efficiency_score": 1.0,
            "action_efficiency_ci": [1.0, 1.0],
            "median_actions_to_first_levelup": 20.0,
        },
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        offline_artifact=offline,
        live_artifact=live,
        registry=registry,
        coheadline_block=coheadline,
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert artifact["honest_verdict"] == (
        "success: offline_to_live_transfer_ratio_metric_helper_shipped_tests_green"
    )
    assert artifact["offline_auroc_component"] == pytest.approx(0.72)
    assert artifact["live_lift_component"] == pytest.approx(0.0)
    assert artifact["offline_to_live_transfer_ratio"] == pytest.approx(0.0)
    assert "bridge-not-crossed" in artifact["null_delta_methodology_note"]
    assert artifact["coheadline_block"]["reported_side_by_side"] == [
        "offline_to_live_transfer_ratio",
        "offline_auroc_component",
        "live_lift_component",
        "reproducible_total_levels",
        "live_submittable_level_count",
        "first_win_rate",
        "action_efficiency_score",
    ]
    assert artifact["coheadline_block"]["reproducible_total_levels"] == 55
    assert artifact["coheadline_block"]["live_submittable_level_count"] == 54
    assert artifact["coheadline_block"]["first_win_rate"] == pytest.approx(0.04)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4622_schema_rejects_malformed_payload() -> None:
    """REQ-ARC-WMTE-4622: schema validation rejects hidden or malformed metrics."""

    from carnot import experiment_4622_offline_to_live_transfer_ratio_metric as mod

    assert mod._as_float(True, 7.5) == 7.5
    assert mod._as_int(False, 8) == 8

    feature_metric = mod.compute_offline_to_live_transfer_ratio(
        {"feature_class_loo_auroc": {"v3_full": 0.63}},
        _live_artifact(
            graduated_first_win=0.0,
            linear_first_win=0.0,
            bare_first_win=0.0,
            graduated_actions=0.0,
            linear_actions=0.0,
            bare_actions=0.0,
        ),
    )
    assert feature_metric["offline_auroc_component"] == pytest.approx(0.63)
    assert feature_metric["baseline_first_win_rate"] == pytest.approx(0.0)
    assert feature_metric["baseline_median_actions_to_first_levelup"] == pytest.approx(0.0)

    per_game_metric = mod.compute_offline_to_live_transfer_ratio(
        {
            "per_game_variance": {
                "per_game_loo_auroc": {
                    "aa00": 0.5,
                    "bb00": 0.7,
                }
            }
        },
        _live_artifact(
            graduated_first_win=0.0,
            linear_first_win=0.0,
            bare_first_win=0.0,
            graduated_actions=0.0,
            linear_actions=0.0,
            bare_actions=0.0,
        ),
    )
    assert per_game_metric["offline_auroc_component"] == pytest.approx(0.6)

    offline = _offline_artifact(0.72)
    live = _live_artifact(
        graduated_first_win=0.10,
        linear_first_win=0.04,
        bare_first_win=0.04,
    )
    registry = {"reproducible_total_levels": "55"}
    coheadline = mod.build_coheadline_block(
        offline_artifact=offline,
        live_artifact=live,
        registry=registry,
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        offline_artifact=offline,
        live_artifact=live,
        registry=registry,
        coheadline_block=coheadline,
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert "null_delta_methodology_note" not in artifact
    assert artifact["registry_reproducible_total_levels"] == 55

    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["inference_substrate"] = "wrong"
    broken["offline_auroc_component"] = 0.0
    broken["live_lift_component"] = -1.0
    broken["offline_to_live_transfer_ratio"] = "0.0"
    broken["coheadline_block"] = []
    broken.pop("tests_added")
    broken["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(broken)

    assert "missing required field tests_added" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "offline_auroc_component" in errors
    assert "live_lift_component" in errors
    assert "offline_to_live_transfer_ratio" in errors
    assert "coheadline_block" in errors
    assert "reproducibility_checksum" in errors

    missing_note = mod.build_artifact(
        preconditions_checked={"ok": True},
        offline_artifact=offline,
        live_artifact=_live_artifact(
            graduated_first_win=0.04,
            linear_first_win=0.04,
            bare_first_win=0.04,
            graduated_actions=20.0,
            linear_actions=20.0,
            bare_actions=20.0,
        ),
        registry=registry,
        coheadline_block=mod.build_coheadline_block(
            offline_artifact=offline,
            live_artifact=_live_artifact(
                graduated_first_win=0.04,
                linear_first_win=0.04,
                bare_first_win=0.04,
                graduated_actions=20.0,
                linear_actions=20.0,
                bare_actions=20.0,
            ),
            registry=registry,
        ),
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )
    missing_note.pop("null_delta_methodology_note")
    missing_note["reproducibility_checksum"] = mod.payload_checksum(missing_note)
    assert "null_delta_methodology_note" in mod.artifact_schema_errors(missing_note)

    wrong_ratio = dict(artifact)
    wrong_ratio["offline_to_live_transfer_ratio"] = 9.0
    wrong_ratio["reproducibility_checksum"] = mod.payload_checksum(wrong_ratio)
    assert "offline_to_live_transfer_ratio_formula" in mod.artifact_schema_errors(wrong_ratio)

    wrong_block = dict(artifact)
    wrong_block["coheadline_block"] = {"reported_side_by_side": []}
    wrong_block["reproducibility_checksum"] = mod.payload_checksum(wrong_block)
    assert "coheadline_block.reported_side_by_side" in mod.artifact_schema_errors(wrong_block)
