"""Unit tests for the exp2477 KV260 bitstream-flash driver.

These tests cover the pure helper functions in
``scripts/experiment_2477_kv260_bitstream_flash.py``:

* ``compute_bitstream_sha256`` -- digest matches a deterministic fixture.
* ``detect_kv260_programmer`` -- parser correctly classifies KV260 USB IDs
  vs. non-Xilinx IDs (e.g. the DirtyJTAG cable for the GateMate board).
* ``build_artifact`` -- always emits the full REQUIRED_ARTIFACT_FIELDS set.
* ``validate_artifact`` -- enforces the terminal-prefix discipline.
* ``write_artifact`` -- round-trips a built artifact through JSON.

Vivado itself is NOT invoked from the tests; ``run_vivado`` is a thin
``subprocess.run`` wrapper that would require a 70+s real run, and the
test job is to verify the bookkeeping, not the EDA toolchain.

REQ traceability:
    REQ-FPGA-006 (KV260 bitstream artifact contract).
SCENARIO traceability:
    SCENARIO-FPGA-006a (bitstream sha256 stable across re-runs).
    SCENARIO-FPGA-006b (programmer-detect honest when board absent).
    SCENARIO-FPGA-006c (artifact schema validates terminal-prefix verdict).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Tests live in tests/python/; scripts/ is a sibling of tests/, so we
# insert the repo root on sys.path before importing the driver module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_2477_kv260_bitstream_flash import (  # noqa: E402
    REQUIRED_ARTIFACT_FIELDS,
    TERMINAL_VERDICT_PREFIXES,
    build_artifact,
    compute_bitstream_sha256,
    detect_kv260_programmer,
    validate_artifact,
    write_artifact,
)


@pytest.fixture
def fixture_binary(tmp_path: Path) -> Path:
    """Write 16 bytes of deterministic content and return its path."""

    fp = tmp_path / "fake.bit"
    fp.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07" * 2)
    return fp


def test_compute_bitstream_sha256_matches_known_digest(fixture_binary: Path) -> None:
    # Pre-computed via:  printf '...' | sha256sum
    expected = "d2a1c23b0d2514a61539937da65e56acf4fbd42f22631050e07284a8703dccb2"
    actual = compute_bitstream_sha256(fixture_binary)
    # We don't hardcode the digest above (a typo there would mask a real
    # bug). Instead, recompute via stdlib hashlib and require parity --
    # the value is only there for human eyeballing.
    import hashlib

    h = hashlib.sha256(fixture_binary.read_bytes()).hexdigest()
    assert actual == h
    # The fixture bytes hash to the recorded value -- sanity-check the
    # comment so a future contributor noticing the literal can trust it.
    assert h == expected


def test_compute_bitstream_sha256_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_bitstream_sha256(tmp_path / "nonexistent.bit")


def test_detect_kv260_programmer_finds_ft4232h() -> None:
    # FTDI FT4232H -- the KV260's onboard JTAG bridge.
    lsusb = (
        "0bda:8153 (bus 4, device 5) path: 2.4.5\n"
        "0403:6011 (bus 3, device 4) path: 2.5\n"
        "1d6b:0003 (bus 4, device 1)\n"
    )
    detected, reason = detect_kv260_programmer(lsusb)
    assert detected is True
    assert "0403:6011" in reason


def test_detect_kv260_programmer_finds_xilinx_cable() -> None:
    # Xilinx Platform Cable USB II.
    lsusb = "03fd:0008 Xilinx, Inc.\n"
    detected, reason = detect_kv260_programmer(lsusb)
    assert detected is True
    assert "03fd:0008" in reason


def test_detect_kv260_programmer_rejects_other_jtag_cables() -> None:
    # GateMate uses DirtyJTAG (1209:c0ca); PolarFire uses FlashPro5
    # (1514:2008). Neither is a KV260 programmer, so the detector must
    # honestly report "not found".
    lsusb = (
        "1209:c0ca (bus 3, device 6) path: 2.3\n"
        "1514:2008 (bus 3, device 5) path: 2.1\n"
        "1d6b:0002 (bus 1, device 1)\n"
    )
    detected, reason = detect_kv260_programmer(lsusb)
    assert detected is False
    assert "no KV260 programmer" in reason
    # The reason must enumerate the required VID/PID pairs so a human
    # reader can confirm which programmer would unblock the flow.
    assert "0403:6011" in reason


def test_detect_kv260_programmer_handles_empty_input() -> None:
    detected, reason = detect_kv260_programmer("")
    assert detected is False
    assert "no KV260 programmer" in reason


def test_detect_kv260_programmer_old_format() -> None:
    # Older `lsusb` format: "Bus 003 Device 004: ID 0403:6011 ..."
    lsusb = "Bus 003 Device 004: ID 0403:6011 FTDI FT4232H Future Tech\n"
    detected, _ = detect_kv260_programmer(lsusb)
    assert detected is True


def test_build_artifact_has_all_required_fields() -> None:
    art = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="complete: smoke",
        kv260_bitstream_flashed=False,
        bitstream_path="/tmp/test.bit",
        bitstream_tool_used="vivado",
        vivado_available=True,
        nextpnr_xilinx_available=False,
        rtl_file_count=20,
        latency_ns=None,
        duration_s=42.5,
        preconditions_checked=[{"resource": "vivado", "available": True}],
    )
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in art, f"build_artifact dropped required field: {field}"


def test_build_artifact_extras_are_merged() -> None:
    art = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="complete_smoke",
        kv260_bitstream_flashed=False,
        bitstream_path=None,
        bitstream_tool_used="none",
        vivado_available=False,
        nextpnr_xilinx_available=False,
        rtl_file_count=0,
        latency_ns=None,
        duration_s=1.0,
        preconditions_checked=[],
        extras={"yosys_version": "0.64", "random_seed": 0},
    )
    assert art["yosys_version"] == "0.64"
    assert art["random_seed"] == 0


def test_build_artifact_extras_do_not_overwrite_required() -> None:
    # The CLAUDE.md fabrication-defence rule is: extras may augment but
    # must not silently overwrite a required field.
    art = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="complete_smoke",
        kv260_bitstream_flashed=False,
        bitstream_path="/tmp/real.bit",
        bitstream_tool_used="vivado",
        vivado_available=True,
        nextpnr_xilinx_available=False,
        rtl_file_count=20,
        latency_ns=None,
        duration_s=1.0,
        preconditions_checked=[],
        extras={"bitstream_path": "/tmp/spoofed.bit"},
    )
    assert art["bitstream_path"] == "/tmp/real.bit"


def test_validate_artifact_accepts_each_terminal_prefix() -> None:
    base = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="placeholder",
        kv260_bitstream_flashed=False,
        bitstream_path=None,
        bitstream_tool_used="none",
        vivado_available=False,
        nextpnr_xilinx_available=False,
        rtl_file_count=0,
        latency_ns=None,
        duration_s=1.0,
        preconditions_checked=[],
    )
    for prefix in TERMINAL_VERDICT_PREFIXES:
        base["honest_verdict"] = prefix + "kv260 ok"
        ok, reason = validate_artifact(base)
        assert ok, f"validate_artifact rejected legal prefix {prefix!r}: {reason}"


def test_validate_artifact_rejects_non_terminal_verdict() -> None:
    art = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="blocked_board_not_connected",
        kv260_bitstream_flashed=False,
        bitstream_path=None,
        bitstream_tool_used="none",
        vivado_available=False,
        nextpnr_xilinx_available=False,
        rtl_file_count=0,
        latency_ns=None,
        duration_s=1.0,
        preconditions_checked=[],
    )
    ok, reason = validate_artifact(art)
    assert ok is False
    assert "terminal prefix" in reason


def test_validate_artifact_rejects_missing_field() -> None:
    art = {
        "experiment_id": "2477",
        "honest_verdict": "complete: smoke",
    }
    ok, reason = validate_artifact(art)
    assert ok is False
    assert "missing required field" in reason


def test_validate_artifact_rejects_non_string_verdict() -> None:
    art = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="complete: tmp",
        kv260_bitstream_flashed=False,
        bitstream_path=None,
        bitstream_tool_used="none",
        vivado_available=False,
        nextpnr_xilinx_available=False,
        rtl_file_count=0,
        latency_ns=None,
        duration_s=1.0,
        preconditions_checked=[],
    )
    art["honest_verdict"] = 12345  # type: ignore[assignment]
    ok, reason = validate_artifact(art)
    assert ok is False
    assert "string" in reason


def test_write_artifact_roundtrip(tmp_path: Path) -> None:
    art = build_artifact(
        experiment_id="2477",
        milestone="2026.05.239",
        honest_verdict="complete: smoke",
        kv260_bitstream_flashed=False,
        bitstream_path="/tmp/real.bit",
        bitstream_tool_used="vivado",
        vivado_available=True,
        nextpnr_xilinx_available=False,
        rtl_file_count=20,
        latency_ns=None,
        duration_s=42.5,
        preconditions_checked=[{"resource": "vivado", "available": True}],
    )
    out = tmp_path / "out" / "experiment_2477.json"
    write_artifact(art, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == art


def test_deliverable_artifact_validates() -> None:
    """The shipped results/experiment_2477_kv260_bitstream_flash.json
    must satisfy the same validator the runtime uses; this catches
    schema drift if a future edit deletes a required field by hand."""

    deliverable = (
        _REPO_ROOT
        / "results"
        / "experiment_2477_kv260_bitstream_flash.json"
    )
    if not deliverable.exists():  # pragma: no cover - only on partial runs
        pytest.skip("deliverable not yet written")
    art = json.loads(deliverable.read_text())
    ok, reason = validate_artifact(art)
    assert ok, reason
