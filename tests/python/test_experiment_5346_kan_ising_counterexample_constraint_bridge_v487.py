"""Tests for Exp 5346 KAN/Ising counterexample constraint bridge.

Spec refs: REQ-KAN-5346, SCENARIO-KAN-5346.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5346_kan_ising_counterexample_constraint_bridge_v487 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5346_spec_declares_constraint_bridge_contract() -> None:
    """REQ-KAN-5346: OpenSpec anchors the bounded bridge artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("## REQ-KAN-5346") : spec.index("## Implementation Status")
    ]

    for marker in (
        "REQ-KAN-5346",
        "SCENARIO-KAN-5346",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "explicit downstream cut constraints",
        "equivalent tiny Ising penalty records",
        "`constraint_bridge_ready` MUST be true only",
        "`certificate_success_delta` MUST remain separate",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_kan_5346_cells_reuse_localized_counterexamples() -> None:
    """REQ-KAN-5346: localized Exp5332 rows become deterministic bridge cells."""

    cells = mod.define_localized_counterexample_cells()

    assert len(cells) == mod.FIXTURE_COUNT
    assert [cell.cell_id for cell in cells] == [
        "kan_unit_0_piece_3_counterexample_cell",
        "kan_unit_1_piece_2_counterexample_cell",
        "kan_unit_2_piece_2_counterexample_cell",
    ]
    assert [cell.unit_index for cell in cells] == [0, 1, 2]
    assert [cell.piece_index for cell in cells] == [3, 2, 2]
    assert cells[0].region == pytest.approx((0.35, 0.6))
    assert cells[1].region == pytest.approx((0.2666666667, 0.6))
    assert cells[2].region == pytest.approx((0.2666666667, 0.6))
    assert all(cell.false_property_rejected_by_source is True for cell in cells)
    assert all(cell.localized_by_source is True for cell in cells)


def test_req_kan_5346_generates_explicit_cuts_and_ising_penalties() -> None:
    """REQ-KAN-5346: every localized cell has a cut and Ising penalty."""

    cells = mod.define_localized_counterexample_cells()
    cuts = mod.generate_counterexample_cuts(cells)

    assert len(cuts) == mod.FIXTURE_COUNT
    assert [cut.cut_id for cut in cuts] == [
        "cut_forbid_kan_unit_0_piece_3",
        "cut_forbid_kan_unit_1_piece_2",
        "cut_forbid_kan_unit_2_piece_2",
    ]
    for cut, cell in zip(cuts, cells, strict=True):
        assert cut.cell_id == cell.cell_id
        assert cut.unit_index == cell.unit_index
        assert cut.piece_index == cell.piece_index
        assert cut.linear_constraint == (
            f"z_unit_{cell.unit_index}_piece_{cell.piece_index} <= 0"
        )
        assert cut.ising_penalty["penalty_weight"] == pytest.approx(1.0)
        assert cut.ising_penalty["active_energy"] == pytest.approx(1.0)


def test_scenario_kan_5346_injected_constraints_reject_false_cells_preserve_true() -> None:
    """SCENARIO-KAN-5346: cuts improve false rejection with no true-property loss."""

    diagnostic = mod.run_constraint_bridge()

    assert diagnostic["fixture_count"] == mod.FIXTURE_COUNT
    assert diagnostic["counterexample_cut_count"] == mod.FIXTURE_COUNT
    assert diagnostic["baseline_false_property_rejection_rate"] == pytest.approx(0.0)
    assert diagnostic["injected_false_property_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["false_property_rejection_delta"] == pytest.approx(1.0)
    assert diagnostic["true_property_preservation_rate"] == pytest.approx(1.0)
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["certificate_success_delta"] == pytest.approx(0.0)
    assert diagnostic["no_broad_certificate_claim"] is True
    assert diagnostic["constraint_bridge_ready"] is True
    assert diagnostic["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert diagnostic["solve_time_delta_s"] >= 0.0

    for row in diagnostic["false_property_checks"]:
        assert row["baseline_accepted"] is True
        assert row["baseline_rejected"] is False
        assert row["injected_accepted"] is False
        assert row["injected_rejected"] is True
        assert row["cut_id"].startswith("cut_forbid_")
        assert row["injected_ising_energy"] == pytest.approx(1.0)

    assert diagnostic["true_property_checks"]
    for row in diagnostic["true_property_checks"]:
        assert row["baseline_accepted"] is True
        assert row["injected_accepted"] is True
        assert row["injected_ising_energy"] == pytest.approx(0.0)


def test_req_kan_5346_artifact_schema_and_validation(tmp_path: Path) -> None:
    """REQ-KAN-5346: artifact exposes principle fields and bare scalar gates."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5346", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.2,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "tests_run") == tests_run
    assert artifact["fixture_count"] == mod.FIXTURE_COUNT
    assert artifact["counterexample_cut_count"] == mod.FIXTURE_COUNT
    assert artifact["false_property_rejection_delta"] == pytest.approx(1.0)
    assert artifact["true_property_preservation_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["certificate_success_delta"] == pytest.approx(0.0)
    assert artifact["no_broad_certificate_claim"] is True
    assert artifact["constraint_bridge_ready"] is True
    assert "REQ-KAN-5346" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5346_validation_fails_closed_on_schema_drift() -> None:
    """REQ-KAN-5346: invalid scope, readiness, or scalar shape fails."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5346", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["no_broad_certificate_claim"] = False
    with pytest.raises(AssertionError, match="broad certificate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["constraint_bridge_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["fixture_count"] = {"value": mod.FIXTURE_COUNT}
    with pytest.raises(AssertionError, match="bare integer"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["unsafe_false_accepts"] = 1
    with pytest.raises(AssertionError, match="unsafe false accepts"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["false_property_rejection_delta"] = 0.0
    with pytest.raises(AssertionError, match="rejection delta"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "wrong")
    with pytest.raises(AssertionError, match="inference"):
        mod.validate_artifact(broken)


def test_req_kan_5346_honest_verdict_prefixes_cover_blocked_cases() -> None:
    """REQ-KAN-5346: terminal verdict prefixes cover ready and blocked cases."""

    diagnostic = mod.run_constraint_bridge()

    assert mod.honest_verdict(diagnostic).startswith("complete:")

    blocked = copy.deepcopy(diagnostic)
    blocked["no_broad_certificate_claim"] = False
    assert mod.honest_verdict(blocked).startswith("blocked_")

    blocked = copy.deepcopy(diagnostic)
    blocked["true_property_preservation_rate"] = 0.5
    assert mod.honest_verdict(blocked).startswith("blocked_")

    blocked = copy.deepcopy(diagnostic)
    blocked["unsafe_false_accepts"] = 1
    assert mod.honest_verdict(blocked).startswith("blocked_")

    blocked = copy.deepcopy(diagnostic)
    blocked["false_property_rejection_delta"] = 0.0
    assert mod.honest_verdict(blocked).startswith("blocked_")

    blocked = copy.deepcopy(diagnostic)
    blocked["constraint_bridge_ready"] = False
    assert mod.honest_verdict(blocked).startswith("blocked_")


def test_deliverable_file_validates_for_scenario_kan_5346() -> None:
    """SCENARIO-KAN-5346: committed deliverable JSON satisfies the V487 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["constraint_bridge_ready"] is True
    assert artifact["no_broad_certificate_claim"] is True
    assert artifact["unsafe_false_accepts"] == 0
