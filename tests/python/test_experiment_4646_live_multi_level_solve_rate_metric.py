"""Tests for Exp 4646 live multi-level solve-rate helper.

Spec refs: REQ-ARC-WMTE-4646, SCENARIO-ARC-WMTE-4646.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _artifact(rows: list[dict[str, Any]], *, key: str = "attempts") -> dict[str, Any]:
    return {"live_measurement": {key: rows}}


def _row(depth: int, *, game: str = "aa00", attempted: bool = True) -> dict[str, Any]:
    return {
        "game": game,
        "attempted": attempted,
        "depth_of_live_solve": depth,
        "variant_signature": f"{game}~synthetic",
    }


def test_req_arc_wmte_4646_spec_declares_live_multi_level_contract() -> None:
    """REQ-ARC-WMTE-4646: OpenSpec declares the helper and required fields."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4646" in spec
    assert "SCENARIO-ARC-WMTE-4646" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4646_one_of_four_attempts_reaching_depth_two_reports_quarter() -> None:
    """SCENARIO-ARC-WMTE-4646: one of four live attempts at depth>=2 reports 0.25."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    metric = mod.compute_live_multi_level_solve_rate(
        _artifact(
            [
                _row(0, game="aa00"),
                _row(1, game="bb00"),
                _row(2, game="cc00"),
                _row(1, game="dd00"),
            ]
        )
    )

    assert metric["live_multi_level_solve_rate"] == pytest.approx(0.25)
    assert metric["live_attempt_count"] == 4
    assert metric["multi_level_attempt_count"] == 1
    assert metric["depth_histogram"] == {
        "depth_0": 1,
        "depth_1": 2,
        "depth_2": 1,
        "depth_3_plus": 0,
    }
    assert metric["null_delta_methodology_note"] is None


def test_scenario_arc_wmte_4646_all_first_win_only_reports_zero_with_null_note() -> None:
    """REQ-ARC-WMTE-4646: depth-one-only attempts report 0.0 with the null note."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    a2 = _artifact([_row(1, game="aa00"), _row(1, game="bb00"), _row(1, game="cc00")])
    metric = mod.compute_live_multi_level_solve_rate(a2)
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        a1_artifact={},
        a2_artifact=a2,
        a6_artifact={"live_submittable_level_count_integrated": 57},
        registry={"reproducible_total_levels": 57},
        live_action_efficiency_artifact={"live_action_efficiency": 0.758102},
        offline_to_live_artifact={"offline_to_live_transfer_ratio": 0.0},
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert metric["live_multi_level_solve_rate"] == pytest.approx(0.0)
    assert metric["depth_histogram"] == {
        "depth_0": 0,
        "depth_1": 3,
        "depth_2": 0,
        "depth_3_plus": 0,
    }
    assert "no live attempt reached depth>=2" in metric["null_delta_methodology_note"]
    assert artifact["live_multi_level_solve_rate"] == pytest.approx(0.0)
    assert "no live attempt reached depth>=2" in artifact["null_delta_methodology_note"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4646_depth_histogram_counts_zero_one_two_and_three_plus() -> None:
    """SCENARIO-ARC-WMTE-4646: the per-depth histogram bins depth 3+ explicitly."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    metric = mod.compute_live_multi_level_solve_rate(
        _artifact(
            [
                _row(0, game="aa00"),
                _row(1, game="bb00"),
                _row(2, game="cc00"),
                _row(3, game="dd00"),
                _row(5, game="ee00"),
                _row(2, game="ff00", attempted=False),
            ],
            key="variant_attempts",
        )
    )

    assert metric["depth_histogram"] == {
        "depth_0": 1,
        "depth_1": 1,
        "depth_2": 1,
        "depth_3_plus": 2,
    }
    assert metric["live_attempt_count"] == 5
    assert metric["multi_level_attempt_count"] == 3
    assert metric["live_multi_level_solve_rate"] == pytest.approx(0.6)
    assert [row["depth_bin"] for row in metric["attempt_depths"]] == [
        "depth_0",
        "depth_1",
        "depth_2",
        "depth_3_plus",
        "depth_3_plus",
    ]


def test_scenario_arc_wmte_4646_multi_artifact_rate_and_coheadline_block_are_correct() -> None:
    """REQ-ARC-WMTE-4646: A1/A2 attempts combine into the canonical coheadline block."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    a1 = {
        "goal_energy_measurement": {
            "variant_attempts": [
                {"game": "aa00", "attempted": True, "reached_level": 0},
                {"game": "bb00", "attempted": True, "reached_level": 1},
            ],
            "first_win_rate": 0.5,
        }
    }
    a2 = {"expansion_measurement": {"attempts": [_row(2, game="cc00"), _row(3, game="dd00")]}}
    coheadline = mod.build_coheadline_block(
        a1_artifact=a1,
        a2_artifact=a2,
        a6_artifact={
            "live_submittable_level_count_integrated": "57",
            "offline_to_live_transfer_ratio_integrated": 0.125,
        },
        registry={"reproducible_total_levels": "57"},
        live_action_efficiency_artifact={"live_action_efficiency": 0.758102},
        offline_to_live_artifact={"offline_to_live_transfer_ratio": 0.25},
    )

    assert coheadline["live_multi_level_solve_rate"] == pytest.approx(0.5)
    assert coheadline["depth_histogram"] == {
        "depth_0": 1,
        "depth_1": 1,
        "depth_2": 1,
        "depth_3_plus": 1,
    }
    assert coheadline["reproducible_total_levels"] == 57
    assert coheadline["live_submittable_level_count"] == 57
    assert coheadline["first_win_rate"] == pytest.approx(0.5)
    assert coheadline["live_action_efficiency"] == pytest.approx(0.758102)
    assert coheadline["offline_to_live_transfer_ratio"] == pytest.approx(0.25)
    assert coheadline["offline_to_live_transfer_ratio_integrated"] == pytest.approx(0.125)
    assert coheadline["reported_side_by_side"] == [
        "live_multi_level_solve_rate",
        "reproducible_total_levels",
        "live_submittable_level_count",
        "first_win_rate",
        "live_action_efficiency",
        "offline_to_live_transfer_ratio",
    ]


def test_req_arc_wmte_4646_defensive_parsing_paths_are_deterministic() -> None:
    """REQ-ARC-WMTE-4646: malformed rows are skipped and checksum validation is stable."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    metric = mod.compute_live_multi_level_solve_rate(
        {
            "expansion_measurement": {
                "attempts": [
                    {"game": "aa00", "attempted": False, "depth_of_live_solve": 3},
                    {"game": "bb00", "attempted": True, "depth_of_live_solve": "bad"},
                    {"game": "cc00", "attempted": True, "reproduction_gate": {"reached_level": 2}},
                ]
            }
        }
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        a1_artifact={},
        a2_artifact={"expansion_measurement": {"attempts": [{"reached_level": 2}]}},
        a6_artifact={},
        registry={},
        live_action_efficiency_artifact={},
        offline_to_live_artifact={},
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert mod._as_float(True, 7.5) == pytest.approx(7.5)
    assert mod._as_int(False, 8) == 8
    assert mod._as_int("not-an-int", 9) == 9
    assert metric["live_attempt_count"] == 1
    assert metric["multi_level_attempt_count"] == 1
    assert metric["depth_histogram"]["depth_2"] == 1
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4646_schema_error_paths_are_explicit() -> None:
    """REQ-ARC-WMTE-4646: schema validation reports each auditable error path."""

    from carnot import experiment_4646_live_multi_level_solve_rate_metric as mod

    fallback_metric = mod.compute_live_multi_level_solve_rate(
        {"attempts": ["bad-row", {"game": "aa00", "first_win": True}]}
    )
    base = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        a1_artifact={},
        a2_artifact={"expansion_measurement": {"attempts": [{"reached_level": 2}]}},
        a6_artifact={},
        registry={},
        live_action_efficiency_artifact={},
        offline_to_live_artifact={},
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert fallback_metric["live_attempt_count"] == 1
    assert fallback_metric["depth_histogram"]["depth_1"] == 1
    assert {
        "missing required field honest_verdict",
        "honest_verdict_terminal_prefix",
        "inference_substrate",
        "live_multi_level_solve_rate",
        "depth_histogram",
        "coheadline_block",
        "reproducibility_checksum",
    }.issubset(set(mod.artifact_schema_errors({})))

    wrong_keys = {**base, "depth_histogram": {"bad": 1}}
    wrong_count = {**base, "depth_histogram": dict(base["depth_histogram"])}
    wrong_count["depth_histogram"]["depth_0"] = 99
    wrong_rate = {**base, "live_multi_level_solve_rate": 0.0}
    null_without_note = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        a1_artifact={},
        a2_artifact={"expansion_measurement": {"attempts": [{"reached_level": 1}]}},
        a6_artifact={},
        registry={},
        live_action_efficiency_artifact={},
        offline_to_live_artifact={},
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )
    del null_without_note["null_delta_methodology_note"]
    wrong_side = {**base, "coheadline_block": {**base["coheadline_block"]}}
    wrong_side["coheadline_block"]["reported_side_by_side"] = []
    wrong_block_rate = {**base, "coheadline_block": {**base["coheadline_block"]}}
    wrong_block_rate["coheadline_block"]["live_multi_level_solve_rate"] = 0.0
    wrong_block_histogram = {**base, "coheadline_block": {**base["coheadline_block"]}}
    wrong_block_histogram["coheadline_block"]["depth_histogram"] = {}

    assert "depth_histogram.keys" in mod.artifact_schema_errors(wrong_keys)
    assert "depth_histogram.count" in mod.artifact_schema_errors(wrong_count)
    assert "live_multi_level_solve_rate_formula" in mod.artifact_schema_errors(wrong_rate)
    assert "null_delta_methodology_note" in mod.artifact_schema_errors(null_without_note)
    assert "coheadline_block.reported_side_by_side" in mod.artifact_schema_errors(wrong_side)
    assert "coheadline_block.live_multi_level_solve_rate" in mod.artifact_schema_errors(
        wrong_block_rate
    )
    assert "coheadline_block.depth_histogram" in mod.artifact_schema_errors(wrong_block_histogram)
