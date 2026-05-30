"""Exp 3443 GateMate opportunistic detect + toolchain continuity audit.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

Why this module exists:
    Hardware-Task Continuity Discipline (CLAUDE.md) requires at least one task
    per attached board per milestone to keep the board visible in retros and
    prevent the forget-pattern. GateMate is classified as opportunistic-only
    (north-star §3), so this is a LIGHT audit: toolchain presence check + board
    enumerate via dirtyJtag. No full synth/pnr/flash cycle is run.

    exp3432 ran the synth/pnr/pack/flash flow but the bitstream did not reach
    the board (pnr returned CC_LUT4 unsupported error). This audit records the
    current toolchain + board state so the next milestone can plan accordingly
    without re-running a doomed flash attempt.

    All three precondition checks (yosys, nextpnr-himbaechel, openFPGALoader)
    mirror the CLAUDE.md Pre-Launch Preconditions table for GateMate. The
    board-detect command uses '--detect' (not '--scan' which does not exist in
    current oss-cad-suite), and matches the openFPGALoader idcode pattern from
    the CLAUDE.md "GateMate board reachable" precondition row.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

EXPERIMENT_ID = 3443
SCHEMA = "carnot.gatemate_detect.v1"
SPEC_REFS = ["REQ-HW-106", "SCENARIO-HW-106"]
INFERENCE_SUBSTRATE = "hardware_smoke"

# The IDCODE emitted by openFPGALoader for the GateMate A1-EVB-2M.
# Source: CLAUDE.md "GateMate board reachable" precondition row.
EXPECTED_IDCODE = "0x20000001"
GATEMATE_MANUFACTURER = "colognechip"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "gatemate_board_detected",
    "duration_s",
}


def check_toolchain() -> list[dict[str, Any]]:
    """Check whether each required GateMate toolchain binary is on PATH.

    Why we check these three specifically:
        - yosys: synthesis (RTL -> netlist); absent → cannot synthesise
        - nextpnr-himbaechel: place-and-route with the himbaechel backend
          required for GateMate in the 2026-era oss-cad-suite (a standalone
          'nextpnr-gatemate' binary does NOT exist — see CLAUDE.md Pre-Launch
          Preconditions table)
        - openFPGALoader: JTAG flashing + board detect; uses dirtyJtag cable

    Returns a list of dicts with keys: resource, available, path_or_none.
    """
    tools = ["yosys", "nextpnr-himbaechel", "openFPGALoader"]
    results = []
    for tool in tools:
        path = shutil.which(tool)
        results.append({
            "resource": f"toolchain:{tool}",
            "available": path is not None,
            "path_or_none": path,
        })
    return results


def detect_board() -> dict[str, Any]:
    """Run 'openFPGALoader -c dirtyJtag --detect' and parse the output.

    Why dirtyJtag:
        The GateMate A1-EVB-2M has an onboard DirtyJTAG MCU (USB PID 0xc0ca).
        CLAUDE.md "GateMate board reachable" specifies this exact command.

    Why '--detect' not '--scan' or '--scan-usb':
        '--scan' does not exist in current oss-cad-suite openFPGALoader.
        '--detect' enumerates the JTAG chain and prints the IDCODE.

    Returns a dict with keys:
        board_detected (bool)
        idcode (str | None)
        manufacturer (str | None)
        raw_output (str)
        returncode (int)
        duration_s (float)
    """
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["openFPGALoader", "-c", "dirtyJtag", "--detect"],
            capture_output=True,
            text=True,
        )
        raw = result.stdout + result.stderr
        duration_s = time.monotonic() - t0
    except FileNotFoundError:
        duration_s = time.monotonic() - t0
        return {
            "board_detected": False,
            "idcode": None,
            "manufacturer": None,
            "raw_output": "openFPGALoader not found on PATH",
            "returncode": -1,
            "duration_s": duration_s,
        }

    # Parse the IDCODE from oss-cad-suite output lines like:
    #   idcode 0x20000001
    #   manufacturer colognechip
    idcode: str | None = None
    manufacturer: str | None = None
    for line in raw.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("idcode"):
            parts = line.strip().split()
            if len(parts) >= 2:
                idcode = parts[1].lower()
        if "manufacturer" in stripped or "colognechip" in stripped:
            manufacturer = line.strip()

    # Board is detected if returncode is 0 AND we found the expected idcode.
    # returncode=0 alone is not sufficient — some cable configs succeed but
    # find no devices; idcode presence is the load-bearing signal.
    board_detected = (result.returncode == 0) and (idcode == EXPECTED_IDCODE.lower())

    return {
        "board_detected": board_detected,
        "idcode": idcode,
        "manufacturer": manufacturer,
        "raw_output": raw,
        "returncode": result.returncode,
        "duration_s": duration_s,
    }


def run_audit() -> dict[str, Any]:
    """Run the full GateMate continuity audit and return an artifact dict.

    Performs exactly two things:
      1. Toolchain presence check (yosys, nextpnr-himbaechel, openFPGALoader)
      2. Board enumerate via 'openFPGALoader -c dirtyJtag --detect'

    No synthesis, P&R, or flash cycle is run.

    The honest_verdict follows Verdict Terminal-Prefix Discipline: both
    board-detected and blocked paths begin with 'complete:'.
    """
    t0 = time.monotonic()

    toolchain_results = check_toolchain()
    toolchain_ok = all(r["available"] for r in toolchain_results)

    preconditions_checked: list[dict[str, Any]] = list(toolchain_results)

    if not toolchain_ok:
        duration_s = time.monotonic() - t0
        missing = [r["resource"] for r in toolchain_results if not r["available"]]
        preconditions_checked.append({
            "resource": "gatemate_board_detect",
            "available": False,
            "detail": f"skipped — toolchain missing: {missing}",
        })
        return {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "spec_refs": SPEC_REFS,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "honest_verdict": "complete: blocked_gatemate_toolchain_missing",
            "gatemate_board_detected": False,
            "toolchain_ok": False,
            "preconditions_checked": preconditions_checked,
            "duration_s": duration_s,
            "continuity_note": (
                f"Toolchain incomplete — missing: {missing}. "
                "Cannot proceed to board detect without toolchain."
            ),
        }

    detect_result = detect_board()
    board_detected = detect_result["board_detected"]

    preconditions_checked.append({
        "resource": "gatemate_board_detect",
        "available": board_detected,
        "idcode": detect_result["idcode"],
        "returncode": detect_result["returncode"],
        "detail": detect_result["raw_output"][:512],  # cap log size in preconditions
    })

    duration_s = time.monotonic() - t0

    if board_detected:
        honest_verdict = "complete: gatemate toolchain present and board detected"
        continuity_note = (
            f"Toolchain OK; board detected with idcode={detect_result['idcode']} "
            f"(manufacturer: {detect_result['manufacturer']}). "
            "Continuity from exp3432: flow ran but bitstream did not reach board "
            "(pnr CC_LUT4 unsupported). Next milestone should resolve LUT4 issue "
            "before re-attempting flash."
        )
    else:
        honest_verdict = "complete: blocked_gatemate_board_unreachable"
        continuity_note = (
            f"Toolchain OK but board not detected "
            f"(returncode={detect_result['returncode']}, "
            f"idcode={detect_result['idcode']!r}). "
            "Opportunistic board; blocked verdict is acceptable per north-star §3."
        )

    return {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "gatemate_board_detected": board_detected,
        "toolchain_ok": toolchain_ok,
        "preconditions_checked": preconditions_checked,
        "detect_idcode": detect_result["idcode"],
        "detect_returncode": detect_result["returncode"],
        "detect_raw_output": detect_result["raw_output"],
        "duration_s": duration_s,
        "continuity_note": continuity_note,
    }
