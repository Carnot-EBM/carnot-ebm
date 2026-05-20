"""Schema + structural-invariant tests for exp2559 GateMate .cfg parser-mismatch fix.

The experiment itself is a hardware bring-up step (re-pack nextpnr-himbaechel
.cfg into native binary .bit via /opt/oss-cad-suite/bin/gmpack, then flash
via openFPGALoader). There is no in-tree Python module to unit-test; the
deliverable is the JSON artifact + the packed .bit binary on disk. These
tests verify the deliverable's schema is well-formed, its fields are
internally consistent with the hardware-flash claim, and the packed .bit
artifact it references actually exists.

The tests deliberately stay narrow (cover only what this experiment added).
"""

import hashlib
import json
from pathlib import Path

# REQ-HARDWARE-GATEMATE: GateMate A1-EVB-2M bitstream flashed via
# nextpnr-himbaechel -> gmpack -> openFPGALoader pipeline.
# SCENARIO-EXP2559: parser-mismatch root-cause diagnosed + native .bit
# repacked + flashed live.

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_2559_gatemate_cfg_fix.json"

REQUIRED_TOP_LEVEL_FIELDS = (
    "experiment",
    "title",
    "run_date",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
    "gatemate_jtag_detected",
    "gatemate_bitstream_flashed",
    "approach_b_attempted",
    "approach_a_attempted",
    "cfg_inspection_note",
    "gatemate_smoke_test_result",
    "preconditions_checked",
)


def _load_artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_exp2559_artifact_has_required_schema_fields():
    """Every REQUIRED ARTIFACT FIELD from the task prompt is present."""
    artifact = _load_artifact()
    missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in artifact]
    assert not missing, f"missing required fields: {missing}"


def test_exp2559_honest_verdict_has_terminal_prefix():
    """Verdict Terminal-Prefix Discipline: start with success_/complete_/etc."""
    artifact = _load_artifact()
    verdict = artifact["honest_verdict"]
    valid_prefixes = (
        "complete:", "complete_",
        "success:", "success_",
        "passed:", "passed_",
        "shipped:", "shipped_",
    )
    assert any(verdict.startswith(p) for p in valid_prefixes), (
        f"honest_verdict must lead with a terminal prefix; got: {verdict[:40]!r}"
    )


def test_exp2559_acceptance_gate_satisfied():
    """Gate: bitstream_flashed=true OR (both approaches attempted + diagnosis)."""
    artifact = _load_artifact()
    flashed = artifact["gatemate_bitstream_flashed"] is True
    both_attempted = (
        artifact["approach_a_attempted"] is True
        and artifact["approach_b_attempted"] is True
        and artifact.get("preconditions_checked") is not None
    )
    assert flashed or both_attempted, (
        "acceptance gate not met: need either physical flash success "
        "OR documented attempts of both approaches"
    )


def test_exp2559_duration_plausible_for_real_hardware_flash():
    """Real hardware flash + repack work takes >60s wall-clock."""
    artifact = _load_artifact()
    assert artifact["duration_s"] >= 60, (
        f"duration_s={artifact['duration_s']} suspiciously short for hardware flash"
    )


def test_exp2559_reproducibility_checksum_present_and_well_formed():
    """Compute-bound artifact requires reproducibility_checksum field."""
    artifact = _load_artifact()
    checksum = artifact["reproducibility_checksum"]
    assert checksum.startswith("md5:"), (
        f"reproducibility_checksum should declare its hash family; got {checksum!r}"
    )
    assert len(checksum.split(":", 1)[1]) == 32, "md5 hex digest must be 32 chars"


def test_exp2559_preconditions_record_gmpack_and_jtag():
    """The Pre-Launch Preconditions block must record the load-bearing tools."""
    artifact = _load_artifact()
    resources = {p["resource"] for p in artifact["preconditions_checked"]}
    assert "gmpack_binary" in resources, "gmpack was the load-bearing fix tool"
    assert "dirtyjtag_usb_1209_c0ca" in resources, "JTAG cable presence is required"
    assert "openFPGALoader_binary" in resources, "flasher version must be recorded"


def test_exp2559_packed_bit_artifact_exists_and_matches_recorded_size():
    """The packed .bit referenced in the artifact actually exists on disk."""
    artifact = _load_artifact()
    packed_path = REPO_ROOT / artifact["artifacts"]["packed_bit_path"]
    assert packed_path.exists(), f"packed .bit missing at {packed_path}"
    recorded_size = artifact["artifacts"]["packed_bit_size_bytes"]
    assert packed_path.stat().st_size == recorded_size, (
        f"packed .bit size on disk ({packed_path.stat().st_size}) does not match "
        f"artifact-recorded size ({recorded_size})"
    )


def test_exp2559_packed_bit_md5_matches_recorded():
    """The packed .bit md5 on disk matches the value recorded in the artifact."""
    artifact = _load_artifact()
    packed_path = REPO_ROOT / artifact["artifacts"]["packed_bit_path"]
    recorded_md5 = artifact["artifacts"]["packed_bit_md5"]
    on_disk_md5 = hashlib.md5(packed_path.read_bytes()).hexdigest()
    assert on_disk_md5 == recorded_md5, (
        f"packed .bit md5 mismatch: on-disk {on_disk_md5} vs recorded {recorded_md5}"
    )


def test_exp2559_input_cfg_checksum_matches_exp2551_reference():
    """The .cfg whose parse-error we diagnose is the same one exp2551 flashed.

    exp2551's reproducibility_checksum recorded md5 73ebc2e50f83ffebc6b54d3e989fb8b7
    on rtl/gatemate_ising_n16.cfg. exp2559 must operate on the same byte-identical
    .cfg so the diagnosis is anchored to the actual prior failure, not a different
    bitstream regenerated mid-flight.
    """
    artifact = _load_artifact()
    cfg_md5 = artifact["artifacts"]["input_cfg_md5"]
    # exp2551 prior_failure cross-reference checksum
    assert cfg_md5 == "73ebc2e50f83ffebc6b54d3e989fb8b7", (
        "input .cfg checksum drifted from exp2551 — diagnosis no longer anchored"
    )


def test_exp2559_post_flash_jtag_idcode_unchanged():
    """Post-flash JTAG re-enumeration must still see the GateMate IDCODE.

    Sanity check that the flash did not brick the chip. exp2551 captured
    pre-flash IDCODE 0x20000001 (GM1Ax); exp2559 confirms post-flash IDCODE
    is the same — chip is alive and configured.
    """
    artifact = _load_artifact()
    flash_output = artifact["flash_output"]
    assert flash_output["post_flash_jtag_idcode"] == "0x20000001"
    assert flash_output["post_flash_jtag_model"] == "GM1Ax"
    assert flash_output["exit_code"] == 0
