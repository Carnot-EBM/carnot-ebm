"""Tests for Exp 2893 tiny KAN PWA/MILP hardware complexity accounting.

Spec refs: REQ-KAN-2893, SCENARIO-KAN-2893.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.hardware.kan_pwa_milp_hardware_complexity_accounting import (  # noqa: E402
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    build_artifact,
    compare_with_existing_accounting_helpers,
    compute_complexity_metrics,
    extract_tiny_pwa_structure,
    load_json,
    run_experiment,
    validate_artifact,
)
from carnot.verify.kan_pwa_milp_corrigendum import build_corrigendum_fixture  # noqa: E402


def _exp2876() -> dict[str, object]:
    fixture = build_corrigendum_fixture()
    return {
        "schema": "carnot.kan_pwa_milp_corrigendum.v2",
        "experiment": 2876,
        "artifact": "experiment_2876_kan_pwa_milp_corrigendum_v2",
        "honest_verdict": "complete_corrigendum_z3_milp_bounds_distinct_no_general_kan_claim",
        "local_error_bound": fixture.local_error_bound,
        "global_error_bound": fixture.global_error_bound,
        "solver_status": "optimal",
        "pwa_fixture": fixture.as_serializable(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_req_kan_2893_is_spec_anchored() -> None:
    """REQ-KAN-2893: the accounting requirement exists before implementation."""
    spec = Path("openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-2893" in spec
    assert "SCENARIO-KAN-2893" in spec


def test_tiny_pwa_metrics_are_deterministic() -> None:
    """REQ-KAN-2893: RM/BOP/NABS and structural counts are deterministic."""
    structure = extract_tiny_pwa_structure(_exp2876())
    metrics = compute_complexity_metrics(structure)

    assert structure.unit_count == 2
    assert structure.segments_per_unit == (4, 4)
    assert structure.output_weights == (1.0, 2.0)
    assert metrics.rm_count == 2
    assert metrics.nabs_count == 4
    assert metrics.bop_count == 96
    assert metrics.memory_table_entries == 8
    assert metrics.pwa_regions == 4
    assert metrics.milp_constraints == 27

    payload = metrics.as_serializable()
    assert payload["branch_count"] == 4
    assert payload["branch_comparison_count"] == 3
    assert payload["milp_binary_variables"] == 4
    assert payload["milp_continuous_variables"] == 2
    assert payload["assumed_rm_bit_pressure"] == 32
    assert payload["assumed_nabs_bit_pressure"] == 8


def test_helper_comparison_uses_existing_accounting_conventions() -> None:
    """SCENARIO-KAN-2893: local KANELE/QuantKAN helpers provide comparisons."""
    metrics = compute_complexity_metrics(extract_tiny_pwa_structure(_exp2876()))

    comparison = compare_with_existing_accounting_helpers(metrics)

    assert comparison["quantkan_kaem_conventions"]["quantkan_proxy_bits"] == 3
    assert comparison["quantkan_kaem_conventions"]["q8_table_bop_proxy"] == 64
    assert comparison["quantkan_kaem_conventions"]["bram36_blocks_for_table_bytes"] == 1
    assert comparison["kanele_node_convention"]["total_luts_per_node"] == 14
    assert comparison["kanele_node_convention"]["fan_in_edges_per_node"] == 2


def test_build_artifact_has_required_schema_and_no_hardware_claims() -> None:
    """REQ-KAN-2893: artifact fields are complete and claims stay disabled."""
    artifact = build_artifact(exp2876=_exp2876(), duration_s=0.125, run_date="20260523")

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260523"
    assert artifact["kan_complexity_accounting_ready"] is True
    assert artifact["hardware_execution_claim_made"] is False
    assert artifact["analog_kan_claim_made"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["local_error_bound"] == pytest.approx(0.0625)
    assert artifact["global_error_bound"] == pytest.approx(0.09375)
    assert artifact["complexity_metrics"]["rm_count"] == artifact["rm_count"] == 2
    assert artifact["complexity_metrics"]["bop_count"] == artifact["bop_count"] == 96
    assert artifact["complexity_metrics"]["nabs_count"] == artifact["nabs_count"] == 4
    assert artifact["complexity_metrics"]["milp_constraints"] == artifact["milp_constraints"] == 27
    assert (
        "results/experiment_2876_kan_pwa_milp_corrigendum_v2.json"
        in artifact["source_artifacts"]
    )
    assert artifact["field_principles"]["claim_boundary"].startswith("No FPGA")


def test_run_experiment_writes_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-2893: runner writes the deterministic accounting JSON."""
    exp2876_path = tmp_path / "experiment_2876_kan_pwa_milp_corrigendum_v2.json"
    deliverable_path = tmp_path / "experiment_2893_kan_hardware_complexity_accounting_v1.json"
    _write_json(exp2876_path, _exp2876())

    artifact = run_experiment(exp2876_path=exp2876_path, deliverable_path=deliverable_path)

    payload = json.loads(deliverable_path.read_text(encoding="utf-8"))
    assert payload == artifact
    assert artifact_has_required_fields(payload)
    assert payload["pwa_regions"] == 4
    assert payload["milp_constraints"] == 27
    assert payload["hardware_execution_claim_made"] is False


def test_invalid_inputs_and_claim_drift_fail_clearly(tmp_path: Path) -> None:
    """REQ-KAN-2893: malformed inputs and hardware-claim drift are rejected."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(bad_json)
    with pytest.raises(ValueError, match="missing Exp 2876 field: pwa_fixture"):
        extract_tiny_pwa_structure({})
    with pytest.raises(ValueError, match="expected Exp 2876.pwa_fixture to be a JSON object"):
        extract_tiny_pwa_structure({"pwa_fixture": []})
    with pytest.raises(ValueError, match="missing Exp 2876.pwa_fixture field: units"):
        extract_tiny_pwa_structure({"pwa_fixture": {"output_segments": [{}]}})
    with pytest.raises(ValueError, match="expected Exp 2876.pwa_fixture.units to be a list"):
        extract_tiny_pwa_structure({"pwa_fixture": {"units": {}, "output_segments": [{}]}})

    bad_fixture = _exp2876()
    bad_fixture["pwa_fixture"] = {"units": [], "output_segments": []}
    with pytest.raises(ValueError, match="expected at least one Exp 2876 unit"):
        extract_tiny_pwa_structure(bad_fixture)
    with pytest.raises(ValueError, match="expected at least one Exp 2876 output segment"):
        extract_tiny_pwa_structure(
            {
                "local_error_bound": 0.0,
                "global_error_bound": 0.0,
                "pwa_fixture": {"units": [{}], "output_segments": []},
            }
        )
    with pytest.raises(ValueError, match="expected Exp 2876 unit 0 to be a JSON object"):
        extract_tiny_pwa_structure(
            {
                "local_error_bound": 0.0,
                "global_error_bound": 0.0,
                "pwa_fixture": {"units": [[]], "output_segments": [{}]},
            }
        )
    with pytest.raises(ValueError, match="expected Exp 2876 unit 0 to contain segments"):
        extract_tiny_pwa_structure(
            {
                "local_error_bound": 0.0,
                "global_error_bound": 0.0,
                "pwa_fixture": {"units": [{"segments": []}], "output_segments": [{}]},
            }
        )

    artifact = build_artifact(exp2876=_exp2876(), duration_s=0.125, run_date="20260523")
    assert not artifact_has_required_fields({})
    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact({})
    artifact["hardware_execution_claim_made"] = True
    assert not artifact_has_required_fields(artifact)
    with pytest.raises(ValueError, match="failed no-hardware-claim validation"):
        validate_artifact(artifact)
