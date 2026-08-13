"""Tests for Exp6393 ARC scalar gate-metric contract.

Spec refs: REQ-ARC-ARM-6393, SCENARIO-ARC-ARM-6393-ROW-REPLAY,
SCENARIO-ARC-ARM-6393-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6393-GATE-REPLAY, REQ-HARNESS-6393.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from carnot import experiment_6393_arc_scalar_gate_metric_contract as exp6393


REPO = Path(__file__).resolve().parents[2]


def _exp6388() -> dict:
    return exp6393.load_json(REPO / exp6393.EXP6388_REL_PATH)


def test_req_arc_arm_6393_recomputes_v549_scalars_from_frozen_rows() -> None:
    """REQ-ARC-ARM-6393: scalar metrics come from rows, not nested deltas."""

    metrics = exp6393.recompute_metrics_from_exp6388(_exp6388())

    assert metrics["pooled_admission_precision_scalar"] == pytest.approx(1.0)
    assert metrics["delta_admission_precision_scalar"] == pytest.approx(0.75)
    assert metrics["false_accept_count_scalar"] == 0
    assert metrics["delta_false_accept_count_scalar"] == -9
    assert metrics["immutable_row_count_receipts"]["raw_receipt_count"] == 48
    assert metrics["immutable_row_count_receipts"]["source_field"] == (
        "raw_model_output_and_evidence_binding_receipts"
    )
    assert metrics["immutable_row_count_receipts"]["aggregate_delta_used_as_source"] is False
    assert set(metrics["admission_precision_by_model_detail"]) == set(
        exp6393.MANDATED_MODEL_IDS
    )
    assert all(
        row["delta_admission_precision"] == pytest.approx(0.75)
        for row in metrics["admission_precision_by_model_detail"].values()
    )


def test_req_harness_6393_rejects_non_scalar_and_nonfinite_gate_values() -> None:
    """REQ-HARNESS-6393: conductor numeric fields reject coercion values."""

    for value in (
        {"pooled_unrounded": 0.75},
        [0.75],
        "0.75",
        True,
        math.nan,
        math.inf,
        -math.inf,
    ):
        with pytest.raises(ValueError):
            exp6393.validate_gate_scalar(value, "delta_admission_precision_scalar")

    assert exp6393.validate_gate_scalar(0.75, "delta_admission_precision_scalar") == 0.75
    assert exp6393.validate_gate_scalar(-9, "delta_false_accept_count_scalar") == -9.0
    with pytest.raises(ValueError):
        exp6393.reject_rounded_sign_change(0.0004, 0.0, ">", 0.0)
    exp6393.reject_rounded_sign_change(0.75, 0.75, ">", 0.0)
    assert exp6393.expect_value_error("not-an-attack", lambda: None)["fail_closed"] is False


def test_scenario_arc_arm_6393_row_hash_duplicate_missing_and_order_attacks_fail() -> None:
    """SCENARIO-ARC-ARM-6393-ATTACKS-FAIL-CLOSED."""

    baseline = _exp6388()
    missing = copy.deepcopy(baseline)
    missing["raw_model_output_and_evidence_binding_receipts"] = [
        row
        for row in missing["raw_model_output_and_evidence_binding_receipts"]
        if row["model_id"] != exp6393.MANDATED_MODEL_IDS[0]
    ]
    with pytest.raises(ValueError, match="missing row"):
        exp6393.recompute_metrics_from_exp6388(missing)

    duplicate = copy.deepcopy(baseline)
    duplicate["raw_model_output_and_evidence_binding_receipts"].append(
        copy.deepcopy(duplicate["raw_model_output_and_evidence_binding_receipts"][0])
    )
    with pytest.raises(ValueError, match="duplicate row"):
        exp6393.recompute_metrics_from_exp6388(duplicate)

    swapped = copy.deepcopy(baseline)
    swapped["models_used"] = list(reversed(swapped["models_used"]))
    with pytest.raises(ValueError, match="model order"):
        exp6393.recompute_metrics_from_exp6388(swapped)

    with pytest.raises(ValueError, match="stale hash"):
        exp6393.require_expected_hash("actual", "expected", "results/experiment_6388.json")


def test_scenario_arc_arm_6393_defensive_row_contract_errors(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6393-ATTACKS-FAIL-CLOSED covers malformed evidence."""

    not_object = tmp_path / "not_object.json"
    not_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON"):
        exp6393.load_json(not_object)

    assert exp6393._terminal_class({"status": "running"}) == "nonterminal_or_unknown"
    counts = exp6393._empty_counts()
    exp6393._add_count(counts, {"status": "rejected", "admissible_goal": True})
    assert counts["false_reject"] == 1
    with pytest.raises(ValueError, match="unknown row status"):
        exp6393._add_count(counts, {"status": "maybe", "admissible_goal": False})
    with pytest.raises(ValueError, match="zero accepts"):
        exp6393._precision(exp6393._empty_counts())

    baseline = _exp6388()
    missing_manifest = copy.deepcopy(baseline)
    missing_manifest.pop("sealed_visible_trajectory_prefix_manifest")
    with pytest.raises(ValueError, match="receipts or manifest"):
        exp6393.recompute_metrics_from_exp6388(missing_manifest)

    no_prefixes = copy.deepcopy(baseline)
    no_prefixes["sealed_visible_trajectory_prefix_manifest"]["prefixes"] = []
    with pytest.raises(ValueError, match="no prefixes"):
        exp6393.recompute_metrics_from_exp6388(no_prefixes)

    extra = copy.deepcopy(baseline)
    extra_row = copy.deepcopy(extra["raw_model_output_and_evidence_binding_receipts"][0])
    extra_row["prefix_id"] = "extra-prefix"
    extra["raw_model_output_and_evidence_binding_receipts"].append(extra_row)
    with pytest.raises(ValueError, match="unexpected row"):
        exp6393.recompute_metrics_from_exp6388(extra)

    reordered = copy.deepcopy(baseline)
    rows = reordered["raw_model_output_and_evidence_binding_receipts"]
    reordered["raw_model_output_and_evidence_binding_receipts"] = rows[32:] + rows[:32]
    with pytest.raises(ValueError, match="model order in row receipts"):
        exp6393.recompute_metrics_from_exp6388(reordered)


def test_scenario_arc_arm_6393_gate_replay_receives_only_finite_numbers(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6393-GATE-REPLAY."""

    output = tmp_path / "experiment_6393_arc_scalar_gate_metric_contract.json"
    artifact = exp6393.build_artifact(
        REPO,
        date="20260813",
        output_path=output,
        tests_run=("unit",),
        duration_s=0.25,
    )

    assert output.exists()
    assert set(exp6393.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["arc_gate_metric_contract_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_live_route_or_solve_claim"]["arc_solve_claim"] is False
    assert artifact["historical_artifacts_modified"] is False
    assert artifact["conductor_modified"] is False
    assert all(artifact["protected_files_unchanged"].values())
    assert all(
        row["comparison_surface_finite_bare_number"]
        for row in artifact["structured_gate_replay_results"]
    )
    assert all(row["passed"] for row in artifact["structured_gate_replay_results"])
    assert artifact["structured_gate_replay_results"][1]["actual"] == pytest.approx(0.75)
    assert all(
        row["finite_bare_number"]
        for row in artifact["scalar_type_and_finiteness_checks"].values()
    )
    assert all(
        row["fail_closed"]
        for row in artifact["coercion_rounding_missing_duplicate_and_order_attack_matrix"]
    )
    assert "downstream Exp6400 readiness gate" in artifact["field_principles"][
        "arc_gate_metric_contract_ready_score"
    ]


def test_req_arc_arm_6393_main_writes_atomic_artifact(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6393: the command writes the required JSON artifact."""

    output = tmp_path / "artifact.json"

    assert exp6393.main(["--date", "20260813", "--output", str(output)]) == 0

    artifact = exp6393.load_json(output)
    assert artifact["arc_gate_metric_contract_ready_score"] == 1.0
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_req_harness_6393_blocks_when_gate_replay_or_schema_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-HARNESS-6393: readiness is withheld when the gate replay fails."""

    output = tmp_path / "blocked.json"

    monkeypatch.setattr(exp6393, "_eval_op", lambda actual, op, expected: (False, "forced"))
    artifact = exp6393.build_artifact(
        REPO,
        date="20260813",
        output_path=output,
        tests_run=("unit",),
        duration_s=0.25,
    )
    assert artifact["status"] == "blocked"
    assert artifact["arc_gate_metric_contract_ready_score"] == 0.0
    assert artifact["honest_verdict"] == "blocked: scalar_gate_replay_failed"

    monkeypatch.setattr(
        exp6393,
        "REQUIRED_ARTIFACT_FIELDS",
        exp6393.REQUIRED_ARTIFACT_FIELDS + ("forced_missing_field",),
    )
    with pytest.raises(ValueError, match="missing required fields"):
        exp6393.build_artifact(
            REPO,
            date="20260813",
            output_path=tmp_path / "schema_fail.json",
            tests_run=("unit",),
            duration_s=0.25,
        )
