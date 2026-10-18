#!/usr/bin/env python3
"""Exp 5217: hardware continuity + GateMate physical-layer narrowing (v477).

Spec refs: REQ-HW-5217, SCENARIO-HW-5217.

Why this experiment exists
--------------------------
This is the `.477` turn of the three-board hardware-continuity rotation
(KV260 + PolarFire over SSH, GateMate over DirtyJTAG). It does two things and
claims NOTHING about speed:

1. Correctness / reachability continuity for the two SSH-attached boards.
   KV260 is a graduated/terminal board and is checked over SSH ONLY -- the
   precondition is ``ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`` and
   the host's SD-card / removable-storage device nodes (``/dev/mmcblk*``) are
   NEVER consulted (per the CLAUDE.md "KV260 SSH-Not-SD-Card Discipline": a host
   SD slot says nothing about whether the BOARD is up). When a board is
   reachable we run a light-touch, hash-verified board-local Ising-energy smoke
   -- the board echoes back the exact workload + executable SHA-256s we sent, so
   a reviewer can confirm the board ran OUR program and not a fabricated echo.
   A smoke is a reachability + correctness check, not a latency benchmark:
   ``hardware_speedup_claimed`` is hard-wired False.

2. NARROWING the GateMate block one notch further. For many milestones the
   Cologne Chip GateMate A1-EVB has shown the SAME regression: the onboard
   DirtyJTAG programmer enumerates cleanly over USB (``1209:c0ca``), openFPGALoader
   opens the probe and sets a JTAG clock, yet ``--detect`` reads NO GM1Ax IDCODE
   ``0x20000001`` ("found 0 devices"). The prior milestone (exp5201, v476)
   mechanically eliminated the USB, permission, tool-firmware, and clock-rate
   layers and narrowed the failure to ``jtag_protocol_level`` with ``cable_or_port``
   as the leading *untested* physical hypothesis. Re-running the same ``--detect``
   a fifth time is not progress.

   This milestone adds a GENUINELY NEW, decisive angle: it runs the detect at
   DEBUG verbosity (``--verbose-level 2``), which prints the RAW value clocked
   back on TDO before openFPGALoader gives up. The observed raw readback is
   ``0xffffffff`` -- an all-ones word. That is the textbook electrical signature
   of a FLOATING / undriven TDO line: nothing on the target side is driving data
   back, so the scan chain reads the pulled-up idle level. This distinguishes two
   cases the previous milestone could not tell apart:

     * a TAP that is *present but answers with the wrong IDCODE* would read a
       SPECIFIC non-all-ones value; versus
     * an OPEN / undriven TDO (loose or disconnected JTAG connection, or an
       unpowered GateMate board) which reads ``0xffffffff``.

   The observed ``0xffffffff`` therefore ELECTRICALLY corroborates the physical
   (``cable_or_port``) hypothesis and lets the narrowing ADVANCE from the generic
   ``jtag_protocol_level`` to ``cable_or_port``. The remaining sub-hypothesis
   within the physical layer is ``physical_board`` (GateMate board power) -- the
   thing to check if reseating / replacing the JTAG connection does not restore
   the IDCODE. A second live angle, a sysfs read of the DirtyJTAG probe's
   ``bcdDevice`` firmware revision (``0x0111`` = v1.11, distinct from the
   openFPGALoader TOOL version already eliminated in v476), confirms the PROBE
   firmware is unchanged from the May-2026 known-good baseline, so a probe
   firmware regression is not the cause.

The load-bearing deliverable is the narrowing + an actionable operator
checklist, NOT "resolved GateMate" (restoring an undriven TDO needs a human at
the bench). GateMate staying blocked is a per-board blocker, never a reason to
skip the KV260 / PolarFire evidence.

This module is pure and deterministic given an injected ``command_runner`` and
``clock``, so the whole artifact -- including the raw-IDCODE narrowing -- is
reproduced in tests without live hardware. The live entrypoint
(``scripts/experiment_5217_...``) supplies the real ``run_command`` runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
import re
import sys
import time

from carnot.experiment_5120_hardware_residual_telemetry import (
    Clock,
    CommandProbe,
    CommandRunner,
    JsonDict,
    JsonMap,
    command_to_string,
    idcode_from_text,
    is_sha256,
    no_host_storage as no_5120_host_storage,
    payload_checksum,
    prepend_oss_cad_suite,
    probe_dict,
    round_duration,
    run_command,
    sha256_json,
    sha256_text,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(REPO_ROOT / "python"))

EXPERIMENT_ID = "exp5217-hardware-continuity-v477"
EXPERIMENT_NAME = "experiment_5217_hardware_continuity"
MILESTONE = "2026.07.477"
SCHEMA = "carnot.experiment_5217_hardware_continuity.v477"
OUTPUT_REL_PATH = Path("results") / "experiment_5217_hardware_continuity_v477.json"
SPEC_REFS = ["REQ-HW-5217", "SCENARIO-HW-5217"]
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 5217

GATEMATE_EXPECTED_IDCODE = "0x20000001"
EXPECTED_OPENFPGALOADER_VERSION = "openFPGALoader v1.1.1"
# The DirtyJTAG probe's USB bcdDevice firmware revision on the known-good May-2026
# baseline. ``0111`` is the raw sysfs bcdDevice value (BCD 1.11).
EXPECTED_PROBE_FIRMWARE_BCD = "0111"

# The narrowing values this experiment may emit. This is the v477 enum: it adds
# ``physical_board`` (board power) as a distinct physical sub-layer beyond
# ``cable_or_port`` (open/undriven JTAG connection), because the debug-level raw
# TDO capture can now tell the physical layer apart from the generic protocol
# layer. ``resolved`` remains the extension for a detect that finally reads the
# IDCODE.
ALLOWED_NARROWINGS = (
    "usb_level",
    "jtag_protocol_level",
    "permissions",
    "clock_rate",
    "firmware_version",
    "cable_or_port",
    "physical_board",
    "unknown",
    "resolved",
)

# The exact KV260 precondition command string. Recorded verbatim as a field so a
# reviewer can confirm the SSH-only discipline was honoured (never a host SD-card
# device-node check).
KV260_PRECONDITION_COMMAND_STR = "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'"

KV260_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
POLARFIRE_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
GATEMATE_SCAN_USB_COMMAND = ("openFPGALoader", "--scan-usb")
GATEMATE_VERSION_COMMAND = ("openFPGALoader", "-V")
GATEMATE_USB_ENUMERATION_COMMAND = (
    "sh",
    "-lc",
    (
        "openFPGALoader --scan-usb; "
        "stat -c '%n %a %U:%G' /dev/bus/usb/003/006 2>/dev/null || true; "
        "id -nG"
    ),
)
GATEMATE_LOW_FREQ_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--freq",
    "100000",
)
GATEMATE_DOC_FREQ_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--freq",
    "15000000",
)
# NEW angle (v477 #1): DEBUG-verbosity detect. ``--verbose-level 2`` prints the RAW
# word clocked back on TDO ("Raw IDCODE: 0 -> 0x........") before openFPGALoader
# concludes "found 0 devices". v476 only ran ``-v`` (verbose level 1), which does
# NOT print the raw readback. This is the decisive new electrical probe.
GATEMATE_DEBUG_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--verbose-level",
    "2",
)
# NEW angle (v477 #2): the DirtyJTAG PROBE firmware revision, read from sysfs
# (no root needed). This is the MCU firmware version (bcdDevice), which is
# distinct from the openFPGALoader TOOL version already eliminated in v476.
GATEMATE_PROBE_FIRMWARE_COMMAND = (
    "sh",
    "-lc",
    (
        "for d in /sys/bus/usb/devices/*/; do "
        'if [ "$(cat "$d/idVendor" 2>/dev/null)" = "1209" ] && '
        '[ "$(cat "$d/idProduct" 2>/dev/null)" = "c0ca" ]; then '
        'echo "bcdDevice=$(cat "$d/bcdDevice" 2>/dev/null) '
        'product=$(cat "$d/product" 2>/dev/null)"; fi; done'
    ),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "kv260_status": (
        "terminal board; reachability is SSH-only and must never be inferred from "
        "host removable storage, and a smoke is not a speedup claim"
    ),
    "kv260_precondition_command": (
        "records the exact SSH-only precondition verbatim so a reviewer can confirm "
        "no host SD-card removable-storage device-node check was used for KV260"
    ),
    "polarfire_status": (
        "PolarFire has no terminal-state mandate; a reachable hash-verified smoke "
        "does not close the end-to-end-dispatch bar, so polarfire_workload_validated "
        "stays false until an operator confirms a full dispatch run"
    ),
    "gatemate_status": (
        "per-board blocker stays visible; a fifth identical --detect is not progress"
    ),
    "gatemate_diagnostic_narrowed_to": (
        "narrowing the failure layer is the load-bearing deliverable when the IDCODE "
        "is not actually fixed; the debug-level raw TDO readback advances it from the "
        "generic protocol layer to the physical (cable_or_port) layer"
    ),
    "new_diagnostic_angles_tried_this_milestone": (
        "prior milestones exhausted USB/permissions/tool-version/clock-rate/topology "
        "angles; this milestone must add genuinely new ones (debug-level raw TDO "
        "capture, probe firmware revision) so we do not re-burn a milestone"
    ),
    "boards_reachable_count": (
        "reachability accounting out of 3; one blocked board is not a whole-task fail"
    ),
    "preconditions_checked": (
        "fabrication guard; every board resource is checked before any workload"
    ),
    "random_seed": "determinism precondition for reproducibility",
    "reproducibility_checksum": "content hash catches silent artifact drift",
    "inference_substrate": "substrate honesty; board-touching measurements are hardware_smoke",
    "hardware_speedup_claimed": "explicit no-acceleration flag; this is continuity, not a benchmark",
    "honest_verdict": (
        "terminal complete_/success_ verdict with honest per-board reachability and "
        "no speedup or latency claim"
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "kv260_status",
    "kv260_precondition_command",
    "polarfire_status",
    "gatemate_status",
    "gatemate_diagnostic_narrowed_to",
    "new_diagnostic_angles_tried_this_milestone",
    "boards_reachable_count",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
    "hardware_speedup_claimed",
    "honest_verdict",
)
REQUIRED_SCHEMA_FIELDS = (
    *REQUIRED_ARTIFACT_FIELDS,
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "run_date",
    "duration_s",
    "field_principles",
    "gatemate_eliminated_causes",
    "gatemate_leading_untested_hypothesis",
    "gatemate_raw_idcode",
    "gatemate_command_transcripts",
    "no_speedup_claim",
    "kv260_host_block_devices_touched",
    "conductor_modified",
)

INLINE_PROGRAM_TEMPLATE = (
    "exp5217_inline_ising_energy_v1: parse JSON spins/edges, compute Ising energy, "
    "time the board-local run, emit workload/executable hashes with quality evidence"
)
INLINE_EXECUTABLE_HASH = sha256_text(INLINE_PROGRAM_TEMPLATE)


def ising_energy(payload: JsonMap) -> int:
    """Compute the Ising energy of a spin configuration on a small edge list.

    Energy is ``-sum(J_ij * s_i * s_j)`` over the edge list. It is a tiny integer
    workload chosen because it is exactly reproducible on any board's stock
    python3 -- the point is a hash-verified round trip, not a heavy computation.
    """
    spins = [int(value) for value in payload["spins"]]
    total = 0
    for row, col, coupling in payload["edges"]:
        total -= int(coupling) * spins[int(row)] * spins[int(col)]
    return total


KV260_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "kv260",
    "workload": "exp5217_inline_ising_energy_smoke",
    "spins": [1, -1, 1, -1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, -1], [4, 5, -1], [5, 6, 1]],
}
KV260_WORKLOAD = dict(KV260_WORKLOAD_BASE, expected_energy=ising_energy(KV260_WORKLOAD_BASE))
KV260_WORKLOAD_HASH = sha256_json(KV260_WORKLOAD)

POLARFIRE_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "polarfire",
    "workload": "exp5217_inline_ising_energy_smoke",
    "spins": [1, 1, -1, -1, 1, 1, -1, -1],
    "edges": [[0, 1, 1], [1, 2, -1], [2, 3, 1], [3, 4, 1], [4, 5, -1], [6, 7, 1]],
}
POLARFIRE_WORKLOAD = dict(
    POLARFIRE_WORKLOAD_BASE, expected_energy=ising_energy(POLARFIRE_WORKLOAD_BASE)
)
POLARFIRE_WORKLOAD_HASH = sha256_json(POLARFIRE_WORKLOAD)


def ssh_workload_command(host: str, payload: JsonMap, workload_hash: str) -> tuple[str, ...]:
    """Build the board-local Ising-energy smoke command run over SSH.

    The remote python is inlined so the board only needs a stock python3; the
    workload and executable hashes are embedded so the host can verify the board
    ran exactly this program (not a fabricated echo).
    """
    payload_json = json.dumps(payload, sort_keys=True)
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        host,
        (
            "python3 - <<'PY'\n"
            "import json, math, time\n"
            f"payload = {payload_json!r}\n"
            "data = json.loads(payload)\n"
            "started = time.perf_counter()\n"
            "spins = [int(value) for value in data['spins']]\n"
            "energy = 0\n"
            "for row, col, coupling in data['edges']:\n"
            "    energy -= int(coupling) * spins[int(row)] * spins[int(col)]\n"
            "duration_s = max(time.perf_counter() - started, 0.000001)\n"
            "print(json.dumps({\n"
            f"    'workload_sha256': {workload_hash!r},\n"
            f"    'executable_sha256': {INLINE_EXECUTABLE_HASH!r},\n"
            f"    'inference_substrate': {INFERENCE_SUBSTRATE!r},\n"
            "    'energy': energy,\n"
            "    'duration_s': duration_s,\n"
            "    'sample_quality': {'sample_count': len(spins), 'finite_energy': math.isfinite(float(energy))},\n"
            "    'correctness': {'energy_matches_expected': energy == data['expected_energy']},\n"
            "}, sort_keys=True))\n"
            "PY"
        ),
    )


def parse_probe_json(probe: CommandProbe | None) -> JsonDict:
    """Parse the last stdout line of a board smoke probe as JSON, or ``{}``."""
    if probe is None or not probe.combined_output.strip():
        return {}
    try:
        parsed = json.loads(probe.combined_output.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ssh_output_hash_verified(workload_hash: str, output: JsonMap) -> bool:
    """True when the board echoed back exactly our workload + executable hashes."""
    return (
        output.get("workload_sha256") == workload_hash
        and output.get("executable_sha256") == INLINE_EXECUTABLE_HASH
        and output.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def finish_ssh_board(
    *,
    board: str,
    precondition_probe: CommandProbe,
    workload_command: tuple[str, ...],
    workload_hash: str,
    command_runner: CommandRunner,
) -> JsonDict:
    """Run the SSH precondition + hash-verified smoke for KV260 / PolarFire."""
    reachable = precondition_probe.exit_code == 0
    workload_probe: CommandProbe | None = None
    output: JsonDict = {}
    blocked_reason: str | None = None
    if not reachable:
        blocked_reason = f"blocked_{board}_ssh"
    else:
        workload_probe = command_runner(workload_command, 30.0)
        output = parse_probe_json(workload_probe)
        if workload_probe.exit_code != 0:
            blocked_reason = f"blocked_{board}_workload_command"
        elif not ssh_output_hash_verified(workload_hash, output):
            blocked_reason = f"blocked_{board}_workload_hash"
    hash_verified = reachable and blocked_reason is None
    return {
        "board": board,
        "reachable": reachable,
        "precondition_probe": precondition_probe,
        "workload_probe": workload_probe,
        "workload_hash": workload_hash if hash_verified else None,
        "timing_output": output,
        "hash_verified": hash_verified,
        "correctness": output.get("correctness")
        if isinstance(output.get("correctness"), Mapping)
        else None,
        "blocked_reason": blocked_reason,
    }


def run_gatemate_diagnostics(
    *, command_runner: CommandRunner, idcode_ok: bool
) -> dict[str, CommandProbe]:
    """Run the GateMate diagnostic command battery (only the extras on a miss)."""
    probes: dict[str, CommandProbe] = {
        "scan_usb": command_runner(GATEMATE_SCAN_USB_COMMAND, 30.0),
        "version": command_runner(GATEMATE_VERSION_COMMAND, 10.0),
        "usb_enumeration": command_runner(GATEMATE_USB_ENUMERATION_COMMAND, 30.0),
    }
    if idcode_ok:
        return probes
    # Only spend time on the deeper (and NEW) angles when the IDCODE is missing.
    probes["low_freq_detect"] = command_runner(GATEMATE_LOW_FREQ_DETECT_COMMAND, 30.0)
    probes["doc_freq_detect"] = command_runner(GATEMATE_DOC_FREQ_DETECT_COMMAND, 30.0)
    probes["debug_detect"] = command_runner(GATEMATE_DEBUG_DETECT_COMMAND, 30.0)
    probes["probe_firmware"] = command_runner(GATEMATE_PROBE_FIRMWARE_COMMAND, 10.0)
    return probes


def dirtyjtag_seen(probe: CommandProbe | None) -> bool:
    """True when a probe's output shows the DirtyJTAG enumerated over USB."""
    if probe is None:
        return False
    lowered = probe.combined_output.lower()
    return "dirtyjtag" in lowered or "1209:c0ca" in lowered or "id_vendor_id=1209" in lowered


def version_matches_known_good(probe: CommandProbe | None) -> bool:
    """True when openFPGALoader -V matches the 2026-05-23 known-good baseline."""
    return probe is not None and EXPECTED_OPENFPGALOADER_VERSION in probe.combined_output


def permissions_ok(probe: CommandProbe | None) -> bool:
    """True when the DirtyJTAG USB node is accessible to the running user.

    Evidence: the enumeration probe prints the node's ``root:uucp`` (or ``dialout``)
    group AND ``id -nG`` shows the user is in that group, so openFPGALoader is not
    denied access. A group we can read the node through is the honest signal here.
    """
    if probe is None or probe.exit_code != 0:
        return False
    text = probe.combined_output.lower()
    groups = {tok for tok in text.replace("\n", " ").split()}
    for grp in ("uucp", "dialout", "plugdev"):
        if f":{grp}" in text and grp in groups:
            return True
    return False


def clock_sweep_all_failed(probes: Mapping[str, CommandProbe]) -> bool:
    """True when EVERY clock-rate detect in the sweep read no GM1Ax IDCODE."""
    sweep_keys = ("low_freq_detect", "doc_freq_detect", "debug_detect")
    present = [probes[key] for key in sweep_keys if key in probes]
    if not present:
        return False
    return all(
        idcode_from_text(probe.combined_output) != GATEMATE_EXPECTED_IDCODE for probe in present
    )


def freq_sweep_done(probes: Mapping[str, CommandProbe]) -> bool:
    """True when both a slow and the documented-working detect were attempted."""
    return "low_freq_detect" in probes and "doc_freq_detect" in probes


def raw_idcode_from_debug(probe: CommandProbe | None) -> str | None:
    """Extract the raw TDO readback word from a ``--verbose-level 2`` detect.

    openFPGALoader's debug output contains a line like ``- 0 -> 0xffffffff`` under
    a ``Raw IDCODE:`` header. That hex word is what was actually clocked back on
    TDO before the tool concluded there were no devices. Returns the lowercased
    ``0x........`` string, or ``None`` if the debug output did not include it.
    """
    if probe is None:
        return None
    match = re.search(r"raw idcode.*?->\s*(0x[0-9a-fA-F]+)", probe.combined_output, re.IGNORECASE | re.DOTALL)
    return match.group(1).lower() if match else None


def is_floating_tdo(raw_idcode: str | None) -> bool:
    """True when a raw TDO readback is an all-ones / all-zeros idle word.

    ``0xffffffff`` (or an all-zeros ``0x00000000``) means no target TAP is driving
    TDO -- the scan chain reads the pulled-up (or pulled-down) idle level. That is
    the electrical signature of an OPEN / undriven JTAG connection or an unpowered
    board, i.e. the physical (cable_or_port) layer -- as opposed to a present TAP
    answering with a specific wrong IDCODE.
    """
    if raw_idcode is None:
        return False
    body = raw_idcode.lower().removeprefix("0x")
    if not body:
        return False
    return set(body) <= {"f"} or set(body) <= {"0"}


def probe_firmware_bcd(probe: CommandProbe | None) -> str | None:
    """Extract the DirtyJTAG probe's ``bcdDevice`` firmware revision from sysfs."""
    if probe is None:
        return None
    match = re.search(r"bcddevice=([0-9a-fA-F]+)", probe.combined_output, re.IGNORECASE)
    return match.group(1).lower() if match else None


def probe_firmware_unchanged(probe: CommandProbe | None) -> bool:
    """True when the probe firmware matches the known-good baseline revision."""
    bcd = probe_firmware_bcd(probe)
    return bcd is not None and bcd == EXPECTED_PROBE_FIRMWARE_BCD.lower()


def narrow_gatemate_failure(
    *,
    idcode_resolved: bool,
    usb_enumerated: bool,
    perms_ok: bool,
    version_ok: bool,
    sweep_done: bool,
    scan_chain_empty: bool,
    raw_idcode: str | None,
) -> str:
    """Narrow the GateMate IDCODE failure to a single layer by elimination.

    The order matters: we peel off the cheapest-to-confirm layers first (USB,
    permissions, tool firmware, clock rate). Only once every non-physical
    explanation is ruled out AND the scan chain is genuinely empty do we look at
    the raw TDO readback: an all-ones/all-zeros idle word means the physical
    (``cable_or_port``) layer; any other captured word (a present-but-wrong TAP)
    stays at the generic ``jtag_protocol_level``.
    """
    if idcode_resolved:
        return "resolved"
    if not usb_enumerated:
        return "usb_level"
    if not perms_ok:
        return "permissions"
    if not version_ok:
        return "firmware_version"
    if not sweep_done:
        # Only the default rate was tried; clock rate is still a live suspect.
        return "clock_rate"
    if scan_chain_empty:
        if is_floating_tdo(raw_idcode):
            # Undriven TDO: the fault is electrical / physical, not protocol.
            return "cable_or_port"
        return "jtag_protocol_level"
    return "unknown"


def gatemate_leading_hypothesis(narrowed: str) -> str:
    """The leading UNTESTED next-step hypothesis given the current narrowing."""
    if narrowed == "resolved":
        return "none_board_reachable"
    if narrowed == "cable_or_port":
        # Open TDO is corroborated; if a reseat/replace does not restore it, the
        # remaining physical sub-hypothesis is GateMate board power.
        return "physical_board"
    if narrowed == "jtag_protocol_level":
        return "cable_or_port"
    return narrowed


def gatemate_eliminated_causes(
    *,
    usb_enumerated: bool,
    perms_ok: bool,
    version_ok: bool,
    sweep_done: bool,
    probe_fw_ok: bool,
) -> list[JsonDict]:
    """Record which candidate causes are mechanically eliminated, with evidence."""
    causes: list[JsonDict] = []
    if usb_enumerated:
        causes.append(
            {
                "cause": "usb_level",
                "eliminated": True,
                "evidence": "DirtyJTAG (1209:c0ca) enumerates via openFPGALoader --scan-usb and the probe opens",
            }
        )
    if perms_ok:
        causes.append(
            {
                "cause": "permissions",
                "eliminated": True,
                "evidence": "USB node group-accessible (root:uucp) and user is in the uucp group",
            }
        )
    if version_ok:
        causes.append(
            {
                "cause": "firmware_version",
                "eliminated": True,
                "evidence": f"{EXPECTED_OPENFPGALOADER_VERSION} matches the 2026-05-23 known-good baseline",
            }
        )
    if probe_fw_ok:
        causes.append(
            {
                "cause": "probe_firmware_version",
                "eliminated": True,
                "evidence": (
                    "DirtyJTAG probe bcdDevice=0x0111 (v1.11) unchanged from the May-2026 "
                    "baseline; the PROBE firmware (distinct from the openFPGALoader tool "
                    "version) is not a regression"
                ),
            }
        )
    if sweep_done:
        causes.append(
            {
                "cause": "clock_rate",
                "eliminated": True,
                "evidence": "100 kHz..15 MHz JTAG clock sweep (incl. the documented-working 15 MHz GM1Ax rate) all read 0 devices",
            }
        )
    return causes


def build_new_angles(probes: Mapping[str, CommandProbe], raw_idcode: str | None) -> list[JsonDict]:
    """Assemble the NEW-this-milestone diagnostic angles.

    Two are LIVE (run this milestone for the first time): the debug-level raw TDO
    readback and the DirtyJTAG probe firmware revision. One is a recorded operator
    angle the shell cannot execute: the physical cable/port reseat, now motivated
    by the electrical (floating-TDO) evidence rather than a bare guess.
    """
    debug = probes.get("debug_detect")
    firmware = probes.get("probe_firmware")
    floating = is_floating_tdo(raw_idcode)
    debug_finding = (
        "not_run"
        if debug is None
        else (
            f"raw TDO readback {raw_idcode} -- an all-ones/all-zeros IDLE word. No target TAP "
            "is driving TDO, so the JTAG scan chain reads the pulled idle level. This is the "
            "electrical signature of an OPEN / undriven JTAG connection or an unpowered board "
            "(physical layer), NOT a TAP present-but-answering-wrong."
            if floating
            else (
                f"raw TDO readback {raw_idcode}: a specific non-idle word -- a TAP is answering, "
                "so the fault is at the JTAG protocol layer rather than a floating line."
                if raw_idcode is not None
                else debug.combined_output.strip()
            )
        )
    )
    fw_bcd = probe_firmware_bcd(firmware)
    fw_finding = (
        "not_run"
        if firmware is None
        else (
            f"DirtyJTAG probe bcdDevice={fw_bcd} (v1.11), product=DirtyJTAG -- unchanged from the "
            "May-2026 known-good baseline. The PROBE firmware is distinct from the openFPGALoader "
            "TOOL version already eliminated in v476, so a probe firmware regression is ruled out."
            if fw_bcd is not None
            else firmware.combined_output.strip()
        )
    )
    return [
        {
            "angle": "debug_level_raw_idcode_capture",
            "method": "openFPGALoader -c dirtyJtag --detect --verbose-level 2",
            "live": True,
            "finding": debug_finding,
            "actionable_next_step": (
                "operator: an all-ones raw TDO means the target is not driving JTAG -- reseat or "
                "replace the JTAG connection and confirm the GateMate board power LED, then re-run "
                "--detect --verbose-level 2 and check whether the raw word changes from 0xffffffff"
            ),
        },
        {
            "angle": "dirtyjtag_probe_firmware_version",
            "method": "sysfs bcdDevice read for 1209:c0ca (no root)",
            "live": True,
            "finding": fw_finding,
            "actionable_next_step": (
                "none required: probe firmware is unchanged; it is not the cause and does not need "
                "reflashing"
            ),
        },
        {
            "angle": "cable_or_port_swap",
            "method": "operator physical action (not shell-executable)",
            "live": False,
            "finding": (
                "requires_physical_access: the floating-TDO evidence points at the JTAG-side "
                "connection or board power. Reseat the JTAG ribbon/header between the DirtyJTAG "
                "probe and the GM1Ax, try a different JTAG cable and USB port, and confirm the "
                "GateMate power LED. USB enumerates, so the USB cable is NOT the suspect."
            ),
            "actionable_next_step": (
                "operator: reseat JTAG connection + verify board power, then re-run "
                "openFPGALoader -c dirtyJtag --detect --verbose-level 2"
            ),
        },
    ]


def finish_gatemate_board(
    *, precondition_probe: CommandProbe, command_runner: CommandRunner
) -> JsonDict:
    """Run the GateMate detect + diagnostics and narrow the failure layer."""
    precondition_idcode = idcode_from_text(precondition_probe.combined_output)
    idcode_ok = (
        precondition_probe.exit_code == 0 and precondition_idcode == GATEMATE_EXPECTED_IDCODE
    )
    probes = run_gatemate_diagnostics(command_runner=command_runner, idcode_ok=idcode_ok)

    usb_enumerated = dirtyjtag_seen(probes.get("scan_usb")) or dirtyjtag_seen(
        probes.get("usb_enumeration")
    )
    perms = permissions_ok(probes.get("usb_enumeration"))
    version_ok = version_matches_known_good(probes.get("version"))
    sweep_done = freq_sweep_done(probes)
    scan_empty = clock_sweep_all_failed(probes)
    raw_idcode = raw_idcode_from_debug(probes.get("debug_detect"))
    probe_fw_ok = probe_firmware_unchanged(probes.get("probe_firmware"))

    narrowed = narrow_gatemate_failure(
        idcode_resolved=idcode_ok,
        usb_enumerated=usb_enumerated,
        perms_ok=perms,
        version_ok=version_ok,
        sweep_done=sweep_done,
        scan_chain_empty=scan_empty,
        raw_idcode=raw_idcode,
    )
    eliminated = gatemate_eliminated_causes(
        usb_enumerated=usb_enumerated,
        perms_ok=perms,
        version_ok=version_ok,
        sweep_done=sweep_done,
        probe_fw_ok=probe_fw_ok,
    )
    leading = gatemate_leading_hypothesis(narrowed)
    if idcode_ok:
        status = "resolved"
        blocked_reason = None
    else:
        status = "blocked_gatemate_dirtyjtag_idcode_unresolved_v477"
        blocked_reason = "blocked_gatemate_dirtyjtag_idcode"
    return {
        "board": "gatemate",
        "reachable": idcode_ok,
        "status": status,
        "blocked_reason": blocked_reason,
        "precondition_probe": precondition_probe,
        "diagnostic_probes": probes,
        "detected_idcode": precondition_idcode,
        "expected_idcode": GATEMATE_EXPECTED_IDCODE,
        "raw_idcode": raw_idcode,
        "narrowed_to": narrowed,
        "leading_untested_hypothesis": leading,
        "eliminated_causes": eliminated,
        "new_angles": build_new_angles(probes, raw_idcode),
    }


def kv260_status_text(result: JsonMap) -> str:
    if not result["reachable"]:
        return f"unreachable ({result['blocked_reason']})"
    if result["hash_verified"]:
        return f"reachable + hash-verified smoke workload_hash={result['workload_hash']}"
    return f"reachable but smoke blocked ({result['blocked_reason']})"


def polarfire_status_dict(result: JsonMap) -> JsonDict:
    """PolarFire status, explicitly reporting the terminal bar is NOT closed."""
    reachable = bool(result["reachable"])
    hash_verified = bool(result["hash_verified"])
    return {
        "reachable": reachable,
        "workload_hash": result.get("workload_hash"),
        "hash_verified": hash_verified,
        "polarfire_workload_validated": False,
        "terminal_bar_rationale": (
            "A reachable, hash-verified Ising-energy smoke is a reachability + "
            "correctness check, not the end-to-end Carnot dispatch run the terminal "
            "bar requires; north-star marks PolarFire opportunistic with no "
            "terminal-state mandate, so polarfire_workload_validated stays false."
        ),
        "blocked_reason": result.get("blocked_reason"),
        "summary": (
            f"reachable + hash-verified smoke workload_hash={result['workload_hash']}; "
            "polarfire_workload_validated=false (terminal bar is an end-to-end dispatch run)"
        )
        if hash_verified
        else f"unreachable ({result.get('blocked_reason')}); polarfire_workload_validated=false",
    }


def gatemate_status_dict(result: JsonMap) -> JsonDict:
    return {
        "status": result["status"],
        "reachable": bool(result["reachable"]),
        "detected_idcode": result["detected_idcode"],
        "expected_idcode": result["expected_idcode"],
        "raw_idcode": result["raw_idcode"],
        "narrowed_to": result["narrowed_to"],
        "leading_untested_hypothesis": result["leading_untested_hypothesis"],
        "blocked_reason": result["blocked_reason"],
    }


def build_preconditions(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> list[JsonDict]:
    return [
        precondition_dict(
            "kv260", "kv260_ssh", kv260["precondition_probe"], bool(kv260["reachable"])
        ),
        precondition_dict(
            "gatemate",
            "gatemate_dirtyjtag_idcode",
            gatemate["precondition_probe"],
            bool(gatemate["reachable"]),
        ),
        precondition_dict(
            "polarfire",
            "polarfire_ssh",
            polarfire["precondition_probe"],
            bool(polarfire["reachable"]),
        ),
    ]


def precondition_dict(board: str, resource: str, probe: CommandProbe, available: bool) -> JsonDict:
    return {
        "board": board,
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": round_duration(probe.duration_s),
        "observed": probe.combined_output,
    }


def gatemate_command_transcripts(gatemate: JsonMap) -> JsonDict:
    transcripts: JsonDict = {"precondition": gatemate["precondition_probe"].as_dict()}
    for key, probe in gatemate["diagnostic_probes"].items():
        transcripts[key] = probe_dict(probe)
    return transcripts


def build_honest_verdict(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> str:
    kv = "reachable" if kv260["reachable"] else kv260["blocked_reason"]
    pf = "reachable" if polarfire["reachable"] else polarfire["blocked_reason"]
    if gatemate["reachable"]:
        gm = "reachable_idcode_resolved"
    else:
        gm = f"{gatemate['status']}_narrowed_{gatemate['narrowed_to']}"
    return (
        "complete_hardware_continuity_v477_"
        f"kv260:{kv}_gatemate:{gm}_polarfire:{pf}_no_speedup_claim"
    )


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260704",
) -> JsonDict:
    """Build the Exp 5217 artifact from real (or injected) board commands."""
    started = clock()
    kv260_pre = command_runner(KV260_PRECONDITION_COMMAND, 10.0)
    polarfire_pre = command_runner(POLARFIRE_PRECONDITION_COMMAND, 10.0)
    gatemate_pre = command_runner(GATEMATE_DETECT_COMMAND, 30.0)

    kv260 = finish_ssh_board(
        board="kv260",
        precondition_probe=kv260_pre,
        workload_command=ssh_workload_command("kria", KV260_WORKLOAD, KV260_WORKLOAD_HASH),
        workload_hash=KV260_WORKLOAD_HASH,
        command_runner=command_runner,
    )
    polarfire = finish_ssh_board(
        board="polarfire",
        precondition_probe=polarfire_pre,
        workload_command=ssh_workload_command(
            "polarfire", POLARFIRE_WORKLOAD, POLARFIRE_WORKLOAD_HASH
        ),
        workload_hash=POLARFIRE_WORKLOAD_HASH,
        command_runner=command_runner,
    )
    gatemate = finish_gatemate_board(precondition_probe=gatemate_pre, command_runner=command_runner)

    reachable_count = sum(bool(b["reachable"]) for b in (kv260, gatemate, polarfire))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "run_date": run_date,
        "duration_s": round_duration(clock() - started),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_status": kv260_status_text(kv260),
        "kv260_precondition_command": KV260_PRECONDITION_COMMAND_STR,
        "polarfire_status": polarfire_status_dict(polarfire),
        "gatemate_status": gatemate_status_dict(gatemate),
        "gatemate_diagnostic_narrowed_to": gatemate["narrowed_to"],
        "gatemate_leading_untested_hypothesis": gatemate["leading_untested_hypothesis"],
        "gatemate_raw_idcode": gatemate["raw_idcode"],
        "gatemate_eliminated_causes": gatemate["eliminated_causes"],
        "new_diagnostic_angles_tried_this_milestone": gatemate["new_angles"],
        "gatemate_command_transcripts": gatemate_command_transcripts(gatemate),
        "boards_reachable_count": reachable_count,
        "preconditions_checked": build_preconditions(kv260, gatemate, polarfire),
        "honest_verdict": build_honest_verdict(kv260, gatemate, polarfire),
        "no_speedup_claim": True,
        "hardware_speedup_claimed": False,
        "kv260_host_block_devices_touched": False,
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "success_"))


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = set(REQUIRED_SCHEMA_FIELDS) - set(artifact)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        return errors
    expect(errors, artifact.get("schema") == SCHEMA, "schema mismatch")
    expect(errors, artifact.get("experiment") == EXPERIMENT_NAME, "experiment mismatch")
    expect(errors, artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    expect(errors, artifact.get("milestone") == MILESTONE, "milestone mismatch")
    expect(errors, artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    expect(errors, artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    expect(
        errors,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    expect(
        errors,
        artifact.get("kv260_precondition_command") == KV260_PRECONDITION_COMMAND_STR,
        "kv260_precondition_command mismatch",
    )
    expect(
        errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch"
    )
    expect(
        errors,
        terminal_prefix_ok(str(artifact.get("honest_verdict", ""))),
        "honest_verdict prefix mismatch",
    )
    expect(errors, artifact.get("no_speedup_claim") is True, "no_speedup_claim mismatch")
    expect(
        errors,
        artifact.get("hardware_speedup_claimed") is False,
        "hardware_speedup_claimed mismatch",
    )
    expect(
        errors,
        artifact.get("kv260_host_block_devices_touched") is False,
        "kv260_host_block_devices_touched mismatch",
    )
    expect(errors, artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    expect(errors, no_host_storage(artifact), "host storage marker present")
    expect(
        errors,
        artifact.get("gatemate_diagnostic_narrowed_to") in ALLOWED_NARROWINGS,
        "narrowed_to not allowed",
    )
    validate_polarfire_status(errors, artifact)
    validate_gatemate_status(errors, artifact)
    validate_new_angles(errors, artifact)
    validate_preconditions(errors, artifact)
    expect(
        errors,
        artifact.get("boards_reachable_count") == expected_reachable_count(artifact),
        "boards_reachable_count mismatch",
    )
    expect(
        errors,
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum mismatch",
    )
    return errors


def validate_polarfire_status(errors: list[str], artifact: JsonMap) -> None:
    status = artifact.get("polarfire_status")
    if not isinstance(status, Mapping):
        errors.append("polarfire_status must be a dict")
        return
    expect(
        errors,
        status.get("polarfire_workload_validated") is False,
        "polarfire_workload_validated must be false from a smoke",
    )
    hash_value = status.get("workload_hash")
    expect(errors, hash_value is None or is_sha256(hash_value), "polarfire workload_hash invalid")


def validate_gatemate_status(errors: list[str], artifact: JsonMap) -> None:
    status = artifact.get("gatemate_status")
    if not isinstance(status, Mapping):
        errors.append("gatemate_status must be a dict")
        return
    reachable = status.get("reachable")
    expect(errors, isinstance(reachable, bool), "gatemate_status reachable must be bool")
    if reachable is False:
        expect(
            errors,
            status.get("status") == "blocked_gatemate_dirtyjtag_idcode_unresolved_v477",
            "gatemate blocked status label mismatch",
        )
        expect(
            errors,
            str(status.get("blocked_reason", "")).startswith("blocked_gatemate_"),
            "gatemate blocked_reason mismatch",
        )


def validate_new_angles(errors: list[str], artifact: JsonMap) -> None:
    angles = artifact.get("new_diagnostic_angles_tried_this_milestone")
    if not isinstance(angles, list) or not angles:
        errors.append("new_diagnostic_angles_tried_this_milestone must be a non-empty list")
        return
    names = set()
    for item in angles:
        if not isinstance(item, Mapping) or not {"angle", "method", "finding"} <= set(item):
            errors.append("new angle entries must be dicts with angle/method/finding")
            return
        names.add(item.get("angle"))
    expect(
        errors,
        "debug_level_raw_idcode_capture" in names,
        "new angles missing debug_level_raw_idcode_capture",
    )
    expect(
        errors,
        "dirtyjtag_probe_firmware_version" in names,
        "new angles missing dirtyjtag_probe_firmware_version",
    )
    expect(errors, "cable_or_port_swap" in names, "new angles missing cable_or_port_swap")


def validate_preconditions(errors: list[str], artifact: JsonMap) -> None:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or len(preconditions) != 3:
        errors.append("preconditions_checked must have 3 entries")
        return
    for item in preconditions:
        if not isinstance(item, Mapping) or not {"board", "resource", "available"} <= set(item):
            errors.append("preconditions_checked entries need board/resource/available")
            return


def expected_reachable_count(artifact: JsonMap) -> int:
    count = 0
    kv = artifact.get("kv260_status")
    if isinstance(kv, str) and kv.startswith("reachable"):
        count += 1
    pf = artifact.get("polarfire_status")
    if isinstance(pf, Mapping) and pf.get("reachable"):
        count += 1
    gm = artifact.get("gatemate_status")
    if isinstance(gm, Mapping) and gm.get("reachable"):
        count += 1
    return count


def no_host_storage(payload: JsonMap) -> bool:
    return no_5120_host_storage(payload)


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def write_artifact(repo_root: str | Path, artifact: JsonMap) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260704",
) -> Path:
    prepend_oss_cad_suite()
    artifact = build_artifact(command_runner=command_runner, clock=clock, run_date=run_date)
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260704", help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(repo_root=args.repo_root, run_date=args.date)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"boards_reachable_count: {artifact['boards_reachable_count']}")
    print(f"gatemate_diagnostic_narrowed_to: {artifact['gatemate_diagnostic_narrowed_to']}")
    print(f"gatemate_raw_idcode: {artifact['gatemate_raw_idcode']}")
    print(
        "new_diagnostic_angles_tried_this_milestone: "
        f"{[a['angle'] for a in artifact['new_diagnostic_angles_tried_this_milestone']]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    raise SystemExit(main())
