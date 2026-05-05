"""Tests for Exp 1319 KAN hardware complexity audit.

Spec refs: REQ-KAN-1319, SCENARIO-KAN-1319.
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

from carnot.hardware.kan_hardware_complexity_audit import (  # noqa: E402
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    build_artifact,
    build_candidate_audit,
    load_json,
    run_experiment,
)
from scripts import experiment_1319_kan_hardware_complexity_audit as exp1319  # noqa: E402


def _exp1148() -> dict[str, object]:
    return {
        "auroc_compressed": 0.971753,
        "size_compressed_bytes": 2704,
        "n_centroids": 32,
        "size_reduction_factor": 5.026627,
    }


def _exp1162() -> dict[str, object]:
    return {
        "sos_kan_n_inputs": 3,
        "sos_kan_k_splines": 8,
        "sos_kan_n_knots": 8,
        "sos_kan_rank": 4,
        "sos_kan_hidden_dim": 16,
        "rm_per_inference": 24,
        "bop_per_inference": 192,
        "nabs_per_inference": 75,
        "lut_storage_bytes": 6144,
        "n_lut_points": 256,
    }


def _exp1174() -> dict[str, object]:
    return {
        "standard_kan_rm": 2547,
        "standard_kan_bop": 81504,
        "standard_kan_nabs": 2352,
        "compressed_kan_rm": 1523,
        "compressed_kan_bop": 48736,
        "compressed_kan_nabs": 1328,
        "bika_kan_rm": 0,
        "bika_kan_bop": 40752,
        "bika_kan_nabs": 4899,
        "npu_feasibility_verdict": "npu_feasible",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_candidate_audit_uses_local_soskan_lut_and_bika_metrics() -> None:
    """REQ-KAN-1319: RM/BOP/NABS/LUT estimates come from local KAN artifacts."""
    estimate = build_candidate_audit(_exp1148(), _exp1162(), _exp1174())

    assert estimate.candidate_name == "compressed_soskan_v3_q8_lut_bika"
    assert estimate.rm_per_inference == 24
    assert estimate.bop_per_inference == 192
    assert estimate.nabs_per_inference == 75
    assert estimate.lookup_table_bytes == 6144
    assert estimate.analog_kan_candidate is True
    assert estimate.npu_or_fpga_best_target == "FPGA"
    assert estimate.nonlinear_activation_budget == {
        "q8_hat_basis_lookups": 24,
        "q8_interpolations": 24,
        "relu_hidden_units": 16,
        "gram_feature_blocks": 3,
    }


def test_build_artifact_has_required_schema_and_conservative_claim() -> None:
    """REQ-KAN-1319: artifact has required fields and no hardware claim."""
    estimate = build_candidate_audit(_exp1148(), _exp1162(), _exp1174())

    artifact = build_artifact(
        estimate=estimate,
        exp1148=_exp1148(),
        exp1162=_exp1162(),
        exp1174=_exp1174(),
        duration_s=0.1,
        run_date="20260505",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260505"
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == (
        "hardware_portability_audit_only_no_fpga_npu_or_analog_execution"
    )
    assert "python/carnot/analysis/kanele_sos_kan_fpga.py" in artifact["relevant_local_modules"]
    assert "python/carnot/hardware/bika_analysis.py" in artifact["relevant_local_modules"]
    assert artifact["platform_classification"]["analog"] == "future_speculative_only"


def test_run_experiment_writes_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-1319: runner writes the deterministic Exp 1319 JSON."""
    exp1148 = tmp_path / "exp1148.json"
    exp1162 = tmp_path / "exp1162.json"
    exp1174 = tmp_path / "exp1174.json"
    deliverable = tmp_path / "experiment_1319_kan_hardware_complexity_audit.json"
    _write_json(exp1148, _exp1148())
    _write_json(exp1162, _exp1162())
    _write_json(exp1174, _exp1174())

    artifact = run_experiment(
        exp1148_path=exp1148,
        exp1162_path=exp1162,
        exp1174_path=exp1174,
        deliverable_path=deliverable,
    )

    payload = json.loads(deliverable.read_text())
    assert payload == artifact
    assert artifact_has_required_fields(payload)
    assert payload["rm_per_inference"] == 24
    assert payload["lookup_table_bytes"] == 6144
    assert payload["hardware_claim_allowed"] is False


def test_script_main_uses_configured_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-KAN-1319: script entrypoint writes its configured artifact path."""
    deliverable = tmp_path / "experiment_1319.json"
    monkeypatch.setattr(exp1319, "DELIVERABLE", deliverable)

    assert exp1319.main() == 0
    payload = json.loads(deliverable.read_text())

    assert artifact_has_required_fields(payload)
    assert payload["run_date"] == "20260505"
    assert payload["honest_verdict"].endswith("no_fpga_npu_or_analog_execution")


def test_invalid_json_and_missing_source_fields_fail_clearly(tmp_path: Path) -> None:
    """REQ-KAN-1319: malformed inputs and incomplete artifacts are rejected."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(bad_json)
    with pytest.raises(ValueError, match="missing Exp 1162 field"):
        build_candidate_audit(_exp1148(), {}, _exp1174())

    estimate = build_candidate_audit(_exp1148(), _exp1162(), _exp1174())
    artifact = build_artifact(
        estimate=estimate,
        exp1148=_exp1148(),
        exp1162=_exp1162(),
        exp1174=_exp1174(),
        duration_s=0.1,
        run_date="20260505",
    )
    assert not artifact_has_required_fields({})
    artifact["hardware_claim_allowed"] = True
    assert not artifact_has_required_fields(artifact)
