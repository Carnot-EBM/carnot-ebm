"""Tests for Exp 2985 SSQA dual-BRAM register-map projection.

Spec refs: REQ-HW-081, SCENARIO-HW-081.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_2985_ssqa_dual_bram_register_map_plan as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _write_json(root: Path, rel_path: Path, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_sources(root: Path) -> None:
    bitstream = root / "build" / "gatemate" / "experiment_2956_gatemate_n16" / "ising.bit"
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(b"exp2985-bitstream-fixture")
    ccf = root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    ccf.parent.mkdir(parents=True, exist_ok=True)
    ccf.write_text("# no physical Pin_in/Pin_out\n# allow-unconstrained\n", encoding="utf-8")
    test_vector = root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json"
    test_vector.write_text(
        json.dumps(
            {
                "init_spins_hex": "0xace1",
                "max_steps": 8,
                "n_spins": 16,
                "top_module": "ising_n16_gatemate",
            }
        ),
        encoding="utf-8",
    )

    _write_json(
        root,
        exp.EXP2955_REL_PATH,
        {
            "clock_assumption": "12.0 MHz nextpnr target frequency",
            "constraints_sha256": "ccf-sha",
            "constraints_file_paths": [str(ccf)],
            "device": "CCGM1A1",
            "dirtyjtag_detected": True,
            "gatemate_constraints_ready": True,
            "nextpnr_options_required": ["--freq 12.0", "--vopt allow-unconstrained"],
            "pin_assumption": "Non-clock IO pins are intentionally unconstrained.",
            "test_vector_paths": [str(test_vector)],
        },
    )
    _write_json(
        root,
        exp.EXP2956_REL_PATH,
        {
            "bitstream_path": str(bitstream),
            "bitstream_sha256": "bitstream-sha",
            "device": "CCGM1A1",
            "gatemate_bitstream_built": True,
            "timing_summary": {
                "max_frequency_mhz": 15.69,
                "requested_frequency_mhz": 12.0,
                "timing_met": True,
            },
            "top_module": "ising_n16_gatemate",
            "utilization_summary": {
                "nextpnr_resource_lines": [],
                "yosys_cell_counts": {},
                "yosys_cells_total": None,
            },
        },
    )
    _write_json(
        root,
        exp.EXP2972_REL_PATH,
        {
            "bitstream_path": str(bitstream),
            "bitstream_sha256": "bitstream-sha",
            "board_detected": True,
            "flash_succeeded": True,
            "honest_verdict": "complete: gatemate_flash_contact_smoke_no_readback",
            "observed_output_sha256": "post-flash-transcript-sha",
            "smoke_vector_passed": False,
            "timing_observation": {
                "readback_supported": False,
                "readback_reason": "No host-visible sampler output/readback path is defined.",
            },
        },
    )
    _write_json(
        root,
        exp.EXP2984_REL_PATH,
        {
            "board_detected": True,
            "board_id": "idcode 0x20000001; colognechip; GateMate Series; GM1Ax",
            "expected_smoke_output": "unavailable_no_host_visible_io_path",
            "flash_succeeded": True,
            "honest_verdict": "complete: gatemate_no_readback_no_host_smoke_io",
            "readback_attempted": False,
            "readback_supported": False,
            "sampler_claim_allowed": False,
            "smoke_vector_attempted": False,
            "smoke_vector_passed": False,
            "speedup_claim_allowed": False,
            "timing_observation": {
                "smoke_vector_reason": "No JTAG, UART, GPIO, or host register protocol.",
            },
        },
    )


def _flatten_registers(register_map: dict) -> list[dict]:
    return [
        register
        for group in register_map["register_groups"].values()
        for register in group["registers"]
    ]


def test_req_hw_081_spec_anchor_exists() -> None:
    """REQ-HW-081: OpenSpec anchors the Exp 2985 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-081" in spec
    assert "SCENARIO-HW-081" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert exp.INFERENCE_SUBSTRATE in spec


def test_req_hw_081_register_map_groups_and_offsets_are_stable() -> None:
    """REQ-HW-081: register fields are grouped and 32-bit aligned."""

    register_map = exp.build_register_map()
    registers = _flatten_registers(register_map)
    offsets = [register["offset"] for register in registers]
    names = {register["name"] for register in registers}

    assert set(register_map["register_groups"]) == {
        "input_control",
        "seed_state",
        "energy_verifier",
        "output",
        "status_error",
    }
    assert len(offsets) == len(set(offsets))
    assert all(offset % 4 == 0 for offset in offsets)
    assert {
        "CONTROL",
        "STATUS",
        "ERROR_CODE",
        "INIT_SPINS_LO",
        "RNG_SEED_LO",
        "ENERGY_ACCUM_LO",
        "SAMPLE_OUT_LO",
        "BANK_A_CRC32",
        "BANK_B_CRC32",
    } <= names
    assert register_map["memory_windows"]["BANK_A_READ_SNAPSHOT"]["base"] == 0x1000
    assert register_map["memory_windows"]["BANK_B_DELAYED_WRITE"]["base"] == 0x2000


def test_req_hw_081_dual_bram_memory_accounting_is_explicit() -> None:
    """REQ-HW-081: memory banks include formulas, bits, and BRAM rounding."""

    layout = exp.build_memory_layout()

    assert layout["assumptions"]["n_spins"] == 16
    assert layout["current_gatemate_rtl_floor"]["dense_coupling_bits_q7"] == 2048
    assert layout["banks"]["bank_a_read_snapshot"]["total_bits"] == 4368
    assert layout["banks"]["bank_b_delayed_write"]["total_bits"] == 672
    assert layout["total_projected_bits"] == 5040
    assert layout["total_projected_bytes"] == 630
    assert layout["kv260_bram36_blocks_min_by_bank"] == {
        "bank_a_read_snapshot": 1,
        "bank_b_delayed_write": 1,
        "total_if_banks_are_separate": 2,
    }
    assert layout["bank_swap_semantics"].startswith("Bank A is stable")


def test_scenario_hw_081_artifact_extracts_current_constraints_and_io_gaps(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-081: current GateMate evidence is preserved without sampler claims."""

    _write_sources(tmp_path)

    artifact = exp.run_experiment(tmp_path, started_s=10.0, now_s=12.5)

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["register_map_plan_ready"] is True
    assert artifact["projection_only"] is True
    assert artifact["sampler_claim_allowed"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)

    gate = artifact["resource_accounting"]["current_gatemate_evidence"]
    assert gate["device"] == "CCGM1A1"
    assert gate["bitstream_built"] is True
    assert gate["flash_succeeded"] is True
    assert gate["readback_supported"] is False
    assert gate["host_visible_smoke_io_available"] is False
    assert gate["timing_summary"]["max_frequency_mhz"] == pytest.approx(15.69)
    assert gate["utilization_counts_available"] is False

    constraints = artifact["resource_accounting"]["constraints_evidence"]
    assert constraints["constraints_ready"] is True
    assert constraints["pin_assumption"].startswith("Non-clock IO")
    assert "--vopt allow-unconstrained" in constraints["nextpnr_options_required"]


def test_req_hw_081_smoke_vectors_and_readbacks_are_later_milestone_checks() -> None:
    """REQ-HW-081: smoke/readback plans require real host-visible IO evidence."""

    vectors = exp.build_smoke_vectors()
    checks = exp.build_readback_checks()

    vector_by_name = {vector["name"]: vector for vector in vectors}
    assert vector_by_name["n16_ring_chord_from_exp2955"]["software_reference_spin_out_hex"] == "0xe7ac"
    assert vector_by_name["n16_ring_chord_from_exp2955"]["pass_condition"] == (
        "after STATUS.done=1, SAMPLE_OUT_LO[15:0] == 0xe7ac and STEP_COUNT == 8"
    )
    assert all(vector["requires_host_visible_io"] for vector in vectors)
    assert checks[0]["name"] == "bitstream_hash_recheck"
    assert any(check["name"] == "bank_crc32_round_trip" for check in checks)
    assert all(check["claim_unlocked_by_check"] == "none" for check in checks)


def test_scenario_hw_081_writes_stable_deliverable(tmp_path: Path) -> None:
    """SCENARIO-HW-081: run_experiment writes the required JSON artifact."""

    _write_sources(tmp_path)

    artifact = exp.run_experiment(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["target_boards"] == ["GateMate A1-EVB-2M", "AMD/Xilinx KV260"]
    assert saved["projection_only"] is True
    assert saved["source_artifacts"] == [path.as_posix() for path in exp.SOURCE_REL_PATHS]


def test_req_hw_081_validation_rejects_claim_boundary_break() -> None:
    """REQ-HW-081: validation refuses accidental sampler or speedup claims."""

    artifact = exp.build_artifact(
        source_payloads=exp.SourcePayloads({}, {}, {}, {}),
        duration_s=0.0,
    )
    artifact["speedup_claim_allowed"] = True

    with pytest.raises(ValueError, match="speedup_claim_allowed"):
        exp.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("projection_only", False, "projection_only"),
        ("register_map_plan_ready", False, "register_map_plan_ready"),
        ("sampler_claim_allowed", True, "sampler_claim_allowed"),
        ("inference_substrate", "hardware_smoke", "inference_substrate"),
    ],
)
def test_req_hw_081_validation_rejects_schema_invariant_breaks(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-HW-081: every projection-only invariant is enforced."""

    artifact = exp.build_artifact(
        source_payloads=exp.SourcePayloads({}, {}, {}, {}),
        duration_s=0.0,
    )
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        exp.validate_artifact(artifact)


def test_req_hw_081_validation_rejects_missing_required_field() -> None:
    """REQ-HW-081: required artifact fields cannot be omitted."""

    artifact = exp.build_artifact(
        source_payloads=exp.SourcePayloads({}, {}, {}, {}),
        duration_s=0.0,
    )
    del artifact["register_map"]

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(artifact)
