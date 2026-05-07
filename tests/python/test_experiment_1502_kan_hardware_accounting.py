"""Tests for Exp 1502 no-synthesis KAN hardware accounting.

Spec refs: REQ-KAN-1502, SCENARIO-KAN-1502.
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

from carnot.hardware.kan_hardware_accounting_quantkan_kaem import (  # noqa: E402
    BRAM36_BYTES,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    bram36_blocks,
    build_artifact,
    build_variant_estimates,
    load_json,
    run_experiment,
)


def _exp1148() -> dict[str, object]:
    return {
        "auroc_original": 0.9902,
        "auroc_compressed": 0.971753,
        "size_original_bytes": 13592,
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
        "n_lut_points": 256,
        "lut_storage_bytes": 6144,
        "rm_per_inference": 24,
        "bop_per_inference": 192,
        "nabs_per_inference": 75,
        "auroc_compressed": 0.9718,
    }


def _exp1174() -> dict[str, object]:
    return {
        "standard_kan_rm": 2547,
        "standard_kan_bop": 81504,
        "standard_kan_nabs": 2352,
        "standard_estimated_lut_count": 27822,
        "compressed_kan_rm": 1523,
        "compressed_kan_bop": 48736,
        "compressed_kan_nabs": 1328,
        "compressed_estimated_lut_count": 16558,
        "bika_kan_rm": 0,
        "bika_kan_bop": 40752,
        "bika_kan_nabs": 4899,
        "bika_estimated_lut_count": 16795,
    }


def _exp1199() -> dict[str, object]:
    return {
        "soskan_full_precision_auroc": 0.990228,
        "soskan_4bit_auroc": 0.990137,
        "soskan_4bit_size_mb": 0.001228,
        "soskan_4bit_inference_latency_ms": 0.038038,
    }


def _exp1266() -> dict[str, object]:
    return {
        "auroc_curve": {
            "full_precision": 0.9902,
            "8bit_ptq": 0.9902,
            "4bit_ptq": 0.9901,
            "3bit_ptq": 0.9801,
            "3bit_lut": 0.9791,
        },
        "quantkan_3bit_auroc": 0.9801,
        "lut_kan_speedup": 2.5,
        "lut_table_size_kb": 12.5,
    }


def _exp1319() -> dict[str, object]:
    return {
        "hardware_claim_allowed": False,
        "hardware_execution": {
            "cpu_artifact_generation_only": True,
            "fpga_execution": False,
            "npu_execution": False,
            "analog_execution": False,
        },
        "honest_verdict": "hardware_portability_audit_only_no_fpga_npu_or_analog_execution",
    }


def _exp1372() -> dict[str, object]:
    return {
        "formal_property_verified": True,
        "hardware_correctness_claimed": False,
        "hardware_execution_claimed": False,
        "kan_layer_tested": "GSKANEnergy(n_vars=16,n_groups=4,n_knots=8)",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_variant_estimates_compare_naive_quantkan_and_kaem() -> None:
    """REQ-KAN-1502: operation, memory, LUT, and BRAM proxies are deterministic."""
    estimates = build_variant_estimates(
        _exp1148(), _exp1162(), _exp1174(), _exp1199(), _exp1266()
    )

    naive = estimates["naive_full_precision_soskan"]
    quantkan = estimates["quantkan_3bit_lut_soskan"]
    kaem = estimates["kaem_univariate_table_approx"]

    assert naive.rm_per_inference == 2547
    assert naive.bop_per_inference == 81504
    assert naive.memory_bytes == 13592
    assert naive.lut_proxy == 27822
    assert naive.bram36_blocks == 3

    assert quantkan.rm_per_inference == 24
    assert quantkan.bop_per_inference == 72
    assert quantkan.memory_bytes == 7065
    assert quantkan.lut_proxy == 6298
    assert quantkan.bram36_blocks == 2
    assert quantkan.accuracy_reference["quantkan_3bit_auroc"] == pytest.approx(0.9801)

    assert kaem.rm_per_inference == 6
    assert kaem.bop_per_inference == 48
    assert kaem.memory_bytes == 768
    assert kaem.lut_proxy == 240
    assert kaem.bram36_blocks == 1
    assert kaem.accuracy_reference["univariate_separable_assumption"] is True


def test_build_artifact_has_required_schema_and_blocks_hardware_claims() -> None:
    """REQ-KAN-1502: artifact is complete but explicitly no-synthesis/no-claim."""
    estimates = build_variant_estimates(
        _exp1148(), _exp1162(), _exp1174(), _exp1199(), _exp1266()
    )
    artifact = build_artifact(
        variant_estimates=estimates,
        exp1148=_exp1148(),
        exp1162=_exp1162(),
        exp1174=_exp1174(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        exp1319=_exp1319(),
        exp1372=_exp1372(),
        duration_s=0.125,
        run_date="20260507",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260507"
    assert artifact["kan_hardware_accounting_ready"] is True
    assert artifact["accounting_only_no_synthesis_claim"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["lut_proxy_estimate"] == {
        "kaem_univariate_table_approx": 240,
        "naive_full_precision_soskan": 27822,
        "quantkan_3bit_lut_soskan": 6298,
    }
    assert artifact["bram_proxy_estimate"]["bram36_bytes"] == BRAM36_BYTES
    assert artifact["bram_proxy_estimate"]["variants"]["quantkan_3bit_lut_soskan"] == 2
    assert "no_vivado_synthesis_or_board_measurement_for_exp1502" in artifact["blockers"]
    assert any(
        item["module"] == "python/carnot/verify/kan_pwa_formal.py"
        for item in artifact["kan_components_audited"]
    )


def test_run_experiment_writes_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-1502: runner writes the deterministic accounting JSON."""
    paths = {
        "exp1148_path": tmp_path / "exp1148.json",
        "exp1162_path": tmp_path / "exp1162.json",
        "exp1174_path": tmp_path / "exp1174.json",
        "exp1199_path": tmp_path / "exp1199.json",
        "exp1266_path": tmp_path / "exp1266.json",
        "exp1319_path": tmp_path / "exp1319.json",
        "exp1372_path": tmp_path / "exp1372.json",
    }
    _write_json(paths["exp1148_path"], _exp1148())
    _write_json(paths["exp1162_path"], _exp1162())
    _write_json(paths["exp1174_path"], _exp1174())
    _write_json(paths["exp1199_path"], _exp1199())
    _write_json(paths["exp1266_path"], _exp1266())
    _write_json(paths["exp1319_path"], _exp1319())
    _write_json(paths["exp1372_path"], _exp1372())
    deliverable = tmp_path / "experiment_1502_kan_hardware_accounting_quantkan_kaem.json"

    artifact = run_experiment(deliverable_path=deliverable, **paths)

    payload = json.loads(deliverable.read_text())
    assert payload == artifact
    assert artifact_has_required_fields(payload)
    assert payload["quantkan_proxy_estimates"]["memory_bytes"] == 7065
    assert payload["kaem_proxy_estimates"]["lut_proxy"] == 240
    assert payload["hardware_claim_allowed"] is False


def test_invalid_inputs_and_claim_drift_fail_clearly(tmp_path: Path) -> None:
    """REQ-KAN-1502: malformed inputs and hardware-claim drift are rejected."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(bad_json)
    with pytest.raises(ValueError, match="missing Exp 1174 field"):
        build_variant_estimates(_exp1148(), _exp1162(), {}, _exp1199(), _exp1266())

    assert bram36_blocks(0) == 0
    assert bram36_blocks(1) == 1
    assert not artifact_has_required_fields({})

    estimates = build_variant_estimates(
        _exp1148(), _exp1162(), _exp1174(), _exp1199(), _exp1266()
    )
    artifact = build_artifact(
        variant_estimates=estimates,
        exp1148=_exp1148(),
        exp1162=_exp1162(),
        exp1174=_exp1174(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        exp1319=_exp1319(),
        exp1372=_exp1372(),
        duration_s=0.125,
        run_date="20260507",
    )
    artifact["hardware_claim_allowed"] = True
    assert not artifact_has_required_fields(artifact)
