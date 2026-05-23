"""Tests for Exp 2904 KV260-anchored KAN hardware complexity accounting v2.

Spec refs: REQ-KAN-2904, SCENARIO-KAN-2904.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.hardware import kan_hardware_complexity_accounting_v2 as exp2904


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: str | Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(root: Path, rel_path: str | Path) -> str:
    return hashlib.sha256((root / rel_path).read_bytes()).hexdigest()


def _exp2893_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: tiny KAN PWA/MILP complexity accounting ready",
        "tiny_pwa_structure": {"unit_count": 2},
        "complexity_metrics": {"unit_count": 2},
    }


def _exp2898_payload(bitstream_sha256: str) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "bitstream_sha256": bitstream_sha256,
        "bitstream_sha256_source": "board:/lib/firmware/xilinx/carnot_ising_v4",
        "kv260_overlay_loaded": "carnot_ising_v2_n64",
    }


def _vivado_report() -> str:
    return """
| CLB LUTs                   | 52564 |     0 |          0 |    117120 | 44.88 |
|   LUT as Logic             | 52562 |     0 |          0 |    117120 | 44.88 |
| Block RAM Tile             |     0 |     0 |          0 |       144 |  0.00 |
| DSPs                       |     0 |     0 |          0 |      1248 |  0.00 |
"""


def _fixture_root(tmp_path: Path) -> str:
    bitstream_bytes = b"exp2898-kv260-bitstream"
    bitstream_sha = hashlib.sha256(bitstream_bytes).hexdigest()
    _write_json(tmp_path, exp2904.EXP2893_REL_PATH, _exp2893_payload())
    _write_json(tmp_path, exp2904.EXP2898_REL_PATH, _exp2898_payload(bitstream_sha))
    _write_text(tmp_path, exp2904.UTILIZATION_REPORT_REL_PATH, _vivado_report())
    bitstream = tmp_path / exp2904.BITSTREAM_REL_PATH
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(bitstream_bytes)
    return bitstream_sha


def test_req_kan_2904_spec_anchor_exists() -> None:
    """REQ-KAN-2904: the KV260-anchored accounting contract is in OpenSpec."""

    spec = (exp2904.REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-KAN-2904" in spec
    assert "SCENARIO-KAN-2904" in spec
    assert "experiment_2904_kan_hardware_complexity_accounting_v2.json" in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_parse_vivado_report_extracts_kv260_utilization_counts() -> None:
    """REQ-KAN-2904: LUT, BRAM, and DSP counts come from bitstream metadata."""

    utilization = exp2904.parse_vivado_utilization_report(_vivado_report())

    assert utilization.kv260_lut_used == 52564
    assert utilization.kv260_bram_used == 0
    assert utilization.kv260_dsp_used == 0
    assert utilization.kv260_lut_available == 117120
    assert utilization.kv260_bram_available == 144
    assert utilization.kv260_dsp_available == 1248


def test_scenario_kan_2904_builds_required_artifact_fields(tmp_path: Path) -> None:
    """SCENARIO-KAN-2904: clean upstream evidence produces the v2 artifact."""

    bitstream_sha = _fixture_root(tmp_path)

    artifact = exp2904.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert exp2904.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert exp2904.artifact_has_required_fields(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["kan_node_count"] == 2
    assert artifact["kv260_lut_used"] == 52564
    assert artifact["kv260_bram_used"] == 0
    assert artifact["kv260_dsp_used"] == 0
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["exp2898_bitstream_sha256"] == bitstream_sha
    assert artifact["kan_synthesis_claim_made"] is False
    assert artifact["new_board_execution_claim_made"] is False

    scaling = artifact["scaling_estimate_to_next_size"]
    assert scaling["current_kan_node_count"] == 2
    assert scaling["next_kan_node_count"] == 4
    assert scaling["estimated_kv260_lut_used"] == 105128
    assert scaling["estimated_kv260_bram_used"] == 0
    assert scaling["estimated_kv260_dsp_used"] == 0
    assert scaling["fits_kv260_lut_budget"] is True
    assert scaling["estimated_lut_utilization_pct"] == pytest.approx(89.76, abs=0.01)
    assert "not a KAN synthesis" in scaling["claim_boundary"]

    assert artifact["cited_upstream_artifacts"] == [
        {
            "experiment_id": "exp2893",
            "artifact_path": exp2904.EXP2893_REL_PATH.as_posix(),
            "fields_imported": ["tiny_pwa_structure.unit_count"],
            "sha256": _sha256(tmp_path, exp2904.EXP2893_REL_PATH),
        },
        {
            "experiment_id": "exp2898",
            "artifact_path": exp2904.EXP2898_REL_PATH.as_posix(),
            "fields_imported": [
                "honest_verdict",
                "inference_substrate",
                "bitstream_sha256",
                "bitstream_sha256_source",
                "kv260_overlay_loaded",
            ],
            "sha256": _sha256(tmp_path, exp2904.EXP2898_REL_PATH),
        },
        {
            "experiment_id": "exp2898",
            "artifact_path": exp2904.UTILIZATION_REPORT_REL_PATH.as_posix(),
            "fields_imported": ["CLB LUTs", "Block RAM Tile", "DSPs"],
            "sha256": _sha256(tmp_path, exp2904.UTILIZATION_REPORT_REL_PATH),
        },
        {
            "experiment_id": "exp2898",
            "artifact_path": exp2904.BITSTREAM_REL_PATH.as_posix(),
            "fields_imported": ["sha256"],
            "sha256": _sha256(tmp_path, exp2904.BITSTREAM_REL_PATH),
        },
    ]


def test_write_artifact_writes_requested_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-2904: writer emits the requested stable JSON deliverable."""

    _fixture_root(tmp_path)
    out = exp2904.write_artifact(tmp_path, started_s=4.0, now_s=5.0)

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["artifact"] == "experiment_2904_kan_hardware_complexity_accounting_v2"
    assert exp2904.artifact_has_required_fields(payload)
    assert payload["duration_s"] == pytest.approx(1.0)
    assert payload["kv260_lut_used"] == 52564


def test_invalid_inputs_fail_with_clear_errors(tmp_path: Path) -> None:
    """REQ-KAN-2904: missing or inconsistent upstream evidence is rejected."""

    with pytest.raises(ValueError, match="missing utilization row"):
        exp2904.parse_vivado_utilization_report("| CLB LUTs | 1 | 0 | 0 | 2 | 50.0 |")

    with pytest.raises(ValueError, match="expected JSON object"):
        exp2904.load_json(tmp_path / "missing.json")

    _write_json(tmp_path, "array.json", {"x": 1})
    assert exp2904.load_json(tmp_path / "array.json") == {"x": 1}
    (tmp_path / "bad.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp2904.load_json(tmp_path / "bad.json")

    with pytest.raises(ValueError, match="missing Exp 2893 KAN node count"):
        exp2904.extract_kan_node_count({})
    with pytest.raises(ValueError, match="positive"):
        exp2904.extract_kan_node_count({"tiny_pwa_structure": {"unit_count": 0}})

    bitstream_sha = _fixture_root(tmp_path)
    exp2898 = _exp2898_payload(bitstream_sha)
    exp2898["inference_substrate"] = "software_proxy"
    with pytest.raises(ValueError, match="hardware_smoke"):
        exp2904.validate_exp2898_upstream(exp2898, bitstream_sha)

    exp2898 = _exp2898_payload(bitstream_sha)
    exp2898["honest_verdict"] = "blocked"
    with pytest.raises(ValueError, match="terminal"):
        exp2904.validate_exp2898_upstream(exp2898, bitstream_sha)

    exp2898 = _exp2898_payload("0" * 64)
    with pytest.raises(ValueError, match="bitstream SHA"):
        exp2904.validate_exp2898_upstream(exp2898, bitstream_sha)

    bad_artifact = exp2904.build_artifact(tmp_path, started_s=0.0, now_s=0.5)
    bad_artifact.pop("honest_verdict")
    assert not exp2904.artifact_has_required_fields(bad_artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        exp2904.validate_artifact(bad_artifact)

    invalid_claim = exp2904.build_artifact(tmp_path, started_s=0.0, now_s=0.5)
    invalid_claim["kan_synthesis_claim_made"] = True
    with pytest.raises(ValueError, match="failed required schema"):
        exp2904.validate_artifact(invalid_claim)

    utilization = exp2904.parse_vivado_utilization_report(_vivado_report())
    with pytest.raises(ValueError, match="positive"):
        exp2904.build_scaling_estimate(kan_node_count=0, utilization=utilization)
