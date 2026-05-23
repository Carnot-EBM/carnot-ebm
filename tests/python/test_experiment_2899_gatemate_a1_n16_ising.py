"""Tests for Exp 2899 GateMate A1 n=16 Ising tile bitstream build.

Spec refs: REQ-HW-061, SCENARIO-HW-061.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_2899_gatemate_a1_n16_ising_tile_bitstream_build_v1.json"
)
RTL_PATH = REPO_ROOT / "hardware" / "gatemate" / "ising_n16_gatemate.v"

REQUIRED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "yosys_version",
    "nextpnr_gatemate_version",
    "synth_succeeded",
    "place_and_route_succeeded",
    "lut_mapping_error_log",
    "bitstream_path",
    "bitstream_sha256",
    "duration_s",
)

VALID_VERDICTS = {
    "blocked_gatemate_toolchain_missing",
    "blocked_gatemate_usb_not_attached",
    "blocked_gatemate_synthesis_failed",
    "blocked_gatemate_place_and_route_failed",
    "complete_gatemate_bitstream_generated_flash_staged",
}


def _load_artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_exp2899_artifact_has_required_schema_fields() -> None:
    """REQ-HW-061: the deliverable JSON contains every required task field."""
    artifact = _load_artifact()
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert not missing, f"missing required fields: {missing}"
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["honest_verdict"] in VALID_VERDICTS
    assert isinstance(artifact["preconditions_checked"], list)
    assert isinstance(artifact["duration_s"], int | float)
    assert artifact["duration_s"] >= 0.0


def test_exp2899_preconditions_record_toolchain_and_dirtyjtag() -> None:
    """REQ-HW-061: yosys, nextpnr-gatemate, and DirtyJTAG USB are recorded."""
    artifact = _load_artifact()
    resources = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}

    assert "yosys" in resources
    assert "nextpnr-gatemate" in resources
    assert "dirtyjtag_usb_1209_0xc0ca" in resources
    assert artifact["yosys_version"] == resources["yosys"].get("version", "")
    assert artifact["nextpnr_gatemate_version"] == resources["nextpnr-gatemate"].get(
        "version",
        "",
    )


def test_exp2899_blocked_and_failure_states_are_internally_consistent() -> None:
    """REQ-HW-061: blocked states stop before overclaiming synthesis or bitstreams."""
    artifact = _load_artifact()
    resources = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}

    if artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing":
        assert not (
            resources["yosys"]["available"] and resources["nextpnr-gatemate"]["available"]
        )
        assert artifact["synth_succeeded"] is False
        assert artifact["place_and_route_succeeded"] is False
        assert artifact["bitstream_path"] is None
        assert artifact["bitstream_sha256"] is None
    elif artifact["honest_verdict"] == "blocked_gatemate_usb_not_attached":
        assert resources["dirtyjtag_usb_1209_0xc0ca"]["available"] is False
        assert artifact["synth_succeeded"] is False
        assert artifact["place_and_route_succeeded"] is False
    elif not artifact["synth_succeeded"] or not artifact["place_and_route_succeeded"]:
        assert artifact["lut_mapping_error_log"], "failed builds must preserve stderr"
        assert artifact["bitstream_path"] is None
        assert artifact["bitstream_sha256"] is None
    else:
        bitstream_path = REPO_ROOT / artifact["bitstream_path"]
        assert bitstream_path.exists()
        actual_sha256 = hashlib.sha256(bitstream_path.read_bytes()).hexdigest()
        assert artifact["bitstream_sha256"] == actual_sha256


def test_exp2899_does_not_record_a_flash_attempt() -> None:
    """SCENARIO-HW-061: the build may stage a bitstream but must not flash it."""
    artifact_text = ARTIFACT_PATH.read_text(encoding="utf-8")
    assert "flash_succeeded" not in artifact_text
    assert "-b olimex_gatemateevb" not in artifact_text


def test_exp2899_gatemate_rtl_is_fixed_n16_discrete_sb_top() -> None:
    """REQ-HW-061: RTL exists as the n=16 GateMate adaptation of Discrete SB."""
    rtl = RTL_PATH.read_text(encoding="utf-8")

    assert "Spec: REQ-HW-061, SCENARIO-HW-061" in rtl
    assert re.search(r"\bmodule\s+ising_n16_gatemate\b", rtl)
    assert re.search(r"localparam\s+integer\s+N_VARIABLES\s*=\s*16\b", rtl)
    assert "output reg [15:0]                  spin_out" in rtl
    assert "input  wire [7:0]                   coupling_addr" in rtl
    assert "reg signed [COUPLING_BITS-1:0] j_matrix [0:COUPLING_COUNT-1]" in rtl
    assert "N_VARIABLES = 256" not in rtl
