"""Exp 3478 GateMate opportunistic detect + toolchain continuity audit v4.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

Why this module exists:
    Hardware-Task Continuity Discipline (CLAUDE.md) requires at least one task
    per attached board per milestone. GateMate is classified as opportunistic-only
    (north-star §3), so this is a LIGHT audit: toolchain presence check + board
    enumerate via dirtyJtag. No full synth/pnr/flash cycle is run.

    Exp 3443 (v1) was flagged TAUTOLOGY because the script wrapper added
    ``experiment=3443`` alongside ``experiment_id=3443``. Two top-level numeric
    fields agreeing to >5 significant figures triggers adversarial_verify's
    TAUTOLOGY rule. Fix carried forward from v2: ``run_audit()`` returns the
    artifact WITHOUT an ``experiment_id`` field; the caller (the wrapper script)
    adds exactly ONE identifier field (``experiment_id``). No duplicate numeric
    identifier pair can form.

    v4 differs from v3 only in the experiment ID (3478 vs 3466) and the
    continuity note referencing the prior milestone's predecessor experiment
    (exp3466 instead of exp3454).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

EXPERIMENT_ID = 3478
SCHEMA = "carnot.gatemate_detect.v4"
SPEC_REFS = ["REQ-HW-106", "SCENARIO-HW-106"]
INFERENCE_SUBSTRATE = "hardware_smoke"

# IDCODE emitted by openFPGALoader for the GateMate A1-EVB-2M.
# Source: CLAUDE.md "GateMate board reachable" precondition row.
EXPECTED_IDCODE = "0x20000001"
GATEMATE_MANUFACTURER = "colognechip"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "gatemate_board_detected",
    "toolchain_present",
    "duration_s",
}


def check_toolchain() -> list[dict[str, Any]]:
    """Check whether each required GateMate toolchain binary is on PATH.

    Why these three:
        - yosys: synthesis (RTL → netlist)
        - nextpnr-himbaechel: place-and-route with the himbaechel backend
          required by oss-cad-suite 2026 for GateMate; a standalone
          ``nextpnr-gatemate`` binary does NOT exist (CLAUDE.md Pre-Launch
          Preconditions table)
        - openFPGALoader: JTAG flashing + board detect over dirtyJtag cable

    Returns a list of dicts: {resource, available, path_or_none}.
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
    """Run ``openFPGALoader -c dirtyJtag --detect`` and parse the IDCODE.

    Why dirtyJtag:
        The GateMate A1-EVB-2M has an onboard DirtyJTAG MCU (USB PID 0xc0ca).
        CLAUDE.md specifies this exact command for the board-detect precondition.

    Why ``--detect`` not ``--scan``:
        ``--scan`` does not exist in current oss-cad-suite openFPGALoader.
        ``--detect`` enumerates the JTAG chain and prints the IDCODE.

    Returns: {board_detected, idcode, manufacturer, raw_output, returncode,
              duration_s}.
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

    # Parse IDCODE from oss-cad-suite output lines like:
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
        if "manufacturer" in stripped or GATEMATE_MANUFACTURER in stripped:
            manufacturer = line.strip()

    # Board is detected only when returncode=0 AND idcode matches the expected
    # GateMate IDCODE. returncode=0 alone is insufficient — some cable configs
    # succeed but find no devices; idcode presence is the load-bearing signal.
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
    """Run the GateMate continuity audit and return an artifact dict.

    Only two things are done:
      1. Toolchain presence check (yosys, nextpnr-himbaechel, openFPGALoader)
      2. Board enumerate via ``openFPGALoader -c dirtyJtag --detect``

    No synthesis, P&R, or flash cycle is run.

    IMPORTANT — why experiment_id is NOT in the returned dict:
        The wrapper script (experiment_3478_*.py) adds ``experiment_id``
        as the sole numeric identifier. If this function also returned
        ``experiment_id``, the wrapper would emit both ``experiment_id``
        (from here) and a second copy (from the wrapper), producing two
        top-level numeric fields with identical values — the exact
        TAUTOLOGY that got exp3443 flagged. Keeping the identifier out of
        the module return avoids that. The wrapper is responsible for the
        one numeric identifier field.

    Verdict Terminal-Prefix Discipline: all verdicts start with ``complete:``.
    """
    t0 = time.monotonic()

    toolchain_results = check_toolchain()
    toolchain_present = all(r["available"] for r in toolchain_results)

    preconditions_checked: list[dict[str, Any]] = list(toolchain_results)

    if not toolchain_present:
        duration_s = time.monotonic() - t0
        missing = [r["resource"] for r in toolchain_results if not r["available"]]
        preconditions_checked.append({
            "resource": "gatemate_board_detect",
            "available": False,
            "detail": f"skipped — toolchain missing: {missing}",
        })
        return {
            "schema": SCHEMA,
            "spec_refs": SPEC_REFS,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "honest_verdict": "complete: blocked_gatemate_toolchain_missing",
            "gatemate_board_detected": False,
            "toolchain_present": False,
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
        "detail": detect_result["raw_output"][:512],
    })

    duration_s = time.monotonic() - t0

    if board_detected:
        honest_verdict = "complete: gatemate toolchain present and board detected"
        continuity_note = (
            f"Toolchain OK; board detected with idcode={detect_result['idcode']} "
            f"(manufacturer: {detect_result['manufacturer']}). "
            "Continuity from exp3466: toolchain was missing in the prior milestone "
            "audit; board detect now confirmed. Next milestone should attempt "
            "the full synth/pnr/flash flow."
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
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "gatemate_board_detected": board_detected,
        "toolchain_present": toolchain_present,
        "preconditions_checked": preconditions_checked,
        "detect_idcode": detect_result["idcode"],
        "detect_raw_output": detect_result["raw_output"],
        "duration_s": duration_s,
        "continuity_note": continuity_note,
    }
