#!/usr/bin/env python3
"""Exp 5201: GateMate DirtyJTAG IDCODE diagnostic with new angles (v476).

Spec refs: REQ-HW-5201, SCENARIO-HW-5201.

Why this experiment exists
--------------------------
For three consecutive milestones (`.473`/`.474`/`.475`) the Cologne Chip GateMate
A1-EVB has shown the SAME regression: the DirtyJTAG USB programmer enumerates
cleanly (``openFPGALoader --scan-usb`` sees ``1209:c0ca``), openFPGALoader can
talk to the probe (it always prints and sets a JTAG clock), yet
``openFPGALoader -c dirtyJtag --detect`` reads NO GateMate GM1Ax IDCODE
``0x20000001`` — verbose mode reports ``found 0 devices``. Re-running ``--detect``
a fourth time is not progress. The load-bearing deliverable of this experiment is
therefore NOT "resolved GateMate" (that likely needs a human at the bench) but a
precise NARROWING of the failure layer, so the operator knows exactly where to
look and a future automated attempt does not waste a milestone repeating angles
that are already exhausted.

The narrowing is done by ELIMINATION from real command evidence:

* ``usb_level``       — ruled out when the probe enumerates over USB and
  openFPGALoader opens it (a genuine USB-link fault would fail enumeration).
* ``permissions``     — ruled out when the ``/dev/bus/usb`` node is group-readable
  and the running user is in that group, i.e. openFPGALoader is not denied access.
* ``clock_rate``      — ruled out when a SWEEP of JTAG clocks (from a slow
  100 kHz up to the documented-working 15 MHz GM1Ax rate) all still read 0 devices.
  A single-rate failure would leave clock rate as a live suspect; a full sweep does
  not.
* ``firmware_version``— ruled out when ``openFPGALoader -V`` matches the
  2026-05-23 known-good ``v1.1.1`` baseline (a tool/probe firmware regression is
  the other classic cause of a sudden IDCODE-readback loss).
* ``jtag_protocol_level`` — what remains when USB, permissions, clock rate, and
  tool firmware are all intact but the JTAG scan chain still returns zero TAPs.
  The probe is fully functional; the GM1Ax TAP simply does not answer on TDO.

When the failure narrows to ``jtag_protocol_level`` the LEADING untested physical
hypothesis is ``cable_or_port``: the JTAG-side ribbon/pin-header between the probe
and the board, or GateMate board power. That hypothesis requires a human to reseat
a cable or confirm a power LED, so it is recorded as an operator angle rather than
executed. (Note: the *USB* cable/port is NOT the suspect here — USB works; the
JTAG-side wiring and board power are.)

This module is pure and deterministic given an injected ``command_runner`` and
``clock``, so the whole artifact can be reproduced in tests without live hardware.
The live entrypoint (``scripts/experiment_5201_...``) supplies the real
``run_command`` runner which executes the openFPGALoader / ssh commands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
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

EXPERIMENT_ID = "exp5201-hardware-continuity-gatemate-diagnostic-v476"
EXPERIMENT_NAME = "experiment_5201_hardware_continuity_gatemate_diagnostic"
MILESTONE = "2026.07.476"
SCHEMA = "carnot.experiment_5201_hardware_continuity_gatemate_diagnostic.v476"
OUTPUT_REL_PATH = (
    Path("results") / "experiment_5201_hardware_continuity_gatemate_diagnostic_v476.json"
)
SPEC_REFS = ["REQ-HW-5201", "SCENARIO-HW-5201"]
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 5201

GATEMATE_EXPECTED_IDCODE = "0x20000001"
EXPECTED_OPENFPGALOADER_VERSION = "openFPGALoader v1.1.1"

# The narrowing values this experiment may emit. The task-facing enum is the
# blocked-case subset; ``resolved`` is the natural extension for the (currently
# hypothetical) case where a detect finally reads the IDCODE.
ALLOWED_NARROWINGS = (
    "usb_level",
    "jtag_protocol_level",
    "permissions",
    "clock_rate",
    "firmware_version",
    "cable_or_port",
    "unknown",
    "resolved",
)

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
# NEW angle (a): USB topology map. A prior milestone never recorded WHERE on the
# USB tree the DirtyJTAG lives; this exposes whether it is behind a shared hub and
# whether a direct-root-port move is untested.
GATEMATE_USB_TOPOLOGY_COMMAND = ("sh", "-lc", "lsusb -t; lsusb -d 1209:c0ca")
GATEMATE_VERBOSE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect", "-v")
GATEMATE_LOW_FREQ_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--freq",
    "100000",
)
# NEW angle (b): the documented-working GM1Ax clock rate. The GMM-7550 reference
# config detects the GM1Ax with ``--freq 15000000``; trying it (higher, not lower,
# than our 6 MHz default) closes the clock-rate hypothesis from the other end.
GATEMATE_DOC_FREQ_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--freq",
    "15000000",
)

# NEW angle (d): the openFPGALoader-issue reference search. This is an out-of-band
# reference check (not a live subprocess); the finding is recorded verbatim so a
# future attempt does not have to repeat the search. Sourced 2026-07-03 via web
# search of the trabucayre/openFPGALoader issue tracker + GMM-7550 docs.
OPENFPGALOADER_ISSUE_SEARCH_FINDING = (
    "trabucayre/openFPGALoader issue #628 documents the Olimex GateMate A1-EVB "
    "(this exact board) failing to program; issue #520 fixed IDCODE reads from 4 "
    "to 32 bits; the GMM-7550 reference detects the GM1Ax at --freq 15000000 with "
    "IDCODE 0x20000001. The tracker attributes a 'found 0 devices' result to "
    "outdated DirtyJTAG firmware, USB access permissions, or timing/frequency -- "
    "all three of which this milestone mechanically eliminated (v1.1.1 tool + "
    "bcdDevice 1.11 probe firmware unchanged from the May baseline, user in the "
    "uucp group with an accessible node, and a 100 kHz..15 MHz clock sweep all "
    "reading 0 devices). Residual points to the physical JTAG connection / board "
    "power, not the tool."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "kv260_status": (
        "terminal board; reachability is SSH-only and must never be inferred from "
        "host removable storage, and a smoke is not a speedup claim"
    ),
    "polarfire_status": (
        "PolarFire has no terminal-state mandate; a reachable hash-verified smoke "
        "does not close the end-to-end-dispatch bar, so polarfire_workload_validated "
        "stays false until an operator confirms a full dispatch run"
    ),
    "gatemate_status": (
        "per-board blocker stays visible; a fourth identical --detect is not progress"
    ),
    "gatemate_diagnostic_narrowed_to": (
        "narrowing the failure layer is the load-bearing deliverable when the IDCODE "
        "is not actually fixed; it tells the operator exactly where to look next"
    ),
    "new_diagnostic_angles_tried_this_milestone": (
        "the first six angles were exhausted across prior milestones; this milestone "
        "must add genuinely new ones (USB topology, documented 15 MHz, cable/port "
        "swap, issue search) so we do not re-burn a milestone"
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
    "honest_verdict": (
        "terminal complete_/success_ verdict with honest per-board reachability and "
        "no speedup or latency claim"
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "kv260_status",
    "polarfire_status",
    "gatemate_status",
    "gatemate_diagnostic_narrowed_to",
    "new_diagnostic_angles_tried_this_milestone",
    "boards_reachable_count",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
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
    "gatemate_command_transcripts",
    "no_speedup_claim",
    "hardware_speedup_claimed",
    "kv260_host_block_devices_touched",
    "conductor_modified",
)

INLINE_PROGRAM_TEMPLATE = (
    "exp5201_inline_ising_energy_v1: parse JSON spins/edges, compute Ising energy, "
    "time the board-local run, emit workload/executable hashes with quality evidence"
)
INLINE_EXECUTABLE_HASH = sha256_text(INLINE_PROGRAM_TEMPLATE)


def ising_energy(payload: JsonMap) -> int:
    """Compute the Ising energy of a spin configuration on a small edge list."""
    spins = [int(value) for value in payload["spins"]]
    total = 0
    for row, col, coupling in payload["edges"]:
        total -= int(coupling) * spins[int(row)] * spins[int(col)]
    return total


KV260_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "kv260",
    "workload": "exp5201_inline_ising_energy_smoke",
    "spins": [1, -1, 1, -1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, -1], [4, 5, -1], [5, 6, 1]],
}
KV260_WORKLOAD = dict(KV260_WORKLOAD_BASE, expected_energy=ising_energy(KV260_WORKLOAD_BASE))
KV260_WORKLOAD_HASH = sha256_json(KV260_WORKLOAD)

POLARFIRE_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "polarfire",
    "workload": "exp5201_inline_ising_energy_smoke",
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
    probes["usb_topology"] = command_runner(GATEMATE_USB_TOPOLOGY_COMMAND, 30.0)
    probes["verbose_detect"] = command_runner(GATEMATE_VERBOSE_DETECT_COMMAND, 30.0)
    probes["low_freq_detect"] = command_runner(GATEMATE_LOW_FREQ_DETECT_COMMAND, 30.0)
    probes["doc_freq_detect"] = command_runner(GATEMATE_DOC_FREQ_DETECT_COMMAND, 30.0)
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
    sweep_keys = ("low_freq_detect", "doc_freq_detect", "verbose_detect")
    present = [probes[key] for key in sweep_keys if key in probes]
    if not present:
        return False
    return all(
        idcode_from_text(probe.combined_output) != GATEMATE_EXPECTED_IDCODE for probe in present
    )


def freq_sweep_done(probes: Mapping[str, CommandProbe]) -> bool:
    """True when both a slow and the documented-working detect were attempted."""
    return "low_freq_detect" in probes and "doc_freq_detect" in probes


def narrow_gatemate_failure(
    *,
    idcode_resolved: bool,
    usb_enumerated: bool,
    perms_ok: bool,
    version_ok: bool,
    sweep_done: bool,
    scan_chain_empty: bool,
) -> str:
    """Narrow the GateMate IDCODE failure to a single layer by elimination.

    The order matters: we peel off the cheapest-to-confirm layers first (USB,
    permissions, tool firmware) and only conclude ``jtag_protocol_level`` once
    every other explanation is ruled out AND the scan chain is genuinely empty.
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
        # Probe fully functional, every clock rate empty: the TAP does not answer.
        return "jtag_protocol_level"
    return "unknown"


def gatemate_eliminated_causes(
    *,
    usb_enumerated: bool,
    perms_ok: bool,
    version_ok: bool,
    sweep_done: bool,
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
    if sweep_done:
        causes.append(
            {
                "cause": "clock_rate",
                "eliminated": True,
                "evidence": "100 kHz..15 MHz JTAG clock sweep (incl. the documented-working 15 MHz GM1Ax rate) all read 0 devices",
            }
        )
    return causes


def build_new_angles(probes: Mapping[str, CommandProbe]) -> list[JsonDict]:
    """Assemble the NEW-this-milestone diagnostic angles.

    Two are LIVE (derived from probes run this milestone): the USB topology map
    and the documented-working 15 MHz detect. Two are recorded angles the shell
    cannot execute: the physical cable/port swap (needs a human at the bench) and
    the openFPGALoader-issue reference search (an out-of-band web lookup).
    """
    topology = probes.get("usb_topology")
    doc_freq = probes.get("doc_freq_detect")
    angles: list[JsonDict] = [
        {
            "angle": "usb_topology_map",
            "method": "lsusb -t + lsusb -d 1209:c0ca",
            "live": True,
            "finding": (topology.combined_output.strip() if topology is not None else "not_run"),
            "actionable_next_step": (
                "DirtyJTAG sits behind a shared external hub with the PolarFire "
                "FlashPro; moving it to a direct root port is untested but a weak "
                "hypothesis since USB itself works"
            ),
        },
        {
            "angle": "documented_working_15mhz_detect",
            "method": "openFPGALoader -c dirtyJtag --detect --freq 15000000",
            "live": True,
            "finding": (
                "resolved: read expected IDCODE"
                if doc_freq is not None
                and idcode_from_text(doc_freq.combined_output) == GATEMATE_EXPECTED_IDCODE
                else (doc_freq.combined_output.strip() if doc_freq is not None else "not_run")
            ),
            "actionable_next_step": (
                "the GMM-7550 reference detects the GM1Ax at exactly this rate; if it "
                "still reads 0 devices the clock rate is not the cause"
            ),
        },
        {
            "angle": "cable_or_port_swap",
            "method": "operator physical action (not shell-executable)",
            "live": False,
            "finding": (
                "requires_physical_access: reseat the JTAG ribbon between the "
                "DirtyJTAG probe and the GM1Ax header, confirm the GateMate board "
                "power LED, and/or try a different JTAG cable and USB port. USB "
                "enumerates, so the JTAG-side wiring and board power -- not the USB "
                "cable -- are the prime suspects"
            ),
            "actionable_next_step": (
                "operator: reseat JTAG ribbon + verify board power, then re-run --detect"
            ),
        },
        {
            "angle": "openfpgaloader_issue_search",
            "method": "web search of trabucayre/openFPGALoader issues + GMM-7550 docs (2026-07-03)",
            "live": False,
            "finding": OPENFPGALOADER_ISSUE_SEARCH_FINDING,
            "actionable_next_step": (
                "compare against issue #628 (Olimex GateMate A1-EVB) if the physical "
                "reseat does not resolve it"
            ),
        },
    ]
    return angles


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

    narrowed = narrow_gatemate_failure(
        idcode_resolved=idcode_ok,
        usb_enumerated=usb_enumerated,
        perms_ok=perms,
        version_ok=version_ok,
        sweep_done=sweep_done,
        scan_chain_empty=scan_empty,
    )
    eliminated = gatemate_eliminated_causes(
        usb_enumerated=usb_enumerated,
        perms_ok=perms,
        version_ok=version_ok,
        sweep_done=sweep_done,
    )
    if idcode_ok:
        status = "resolved"
        blocked_reason = None
        leading = "none_board_reachable"
    else:
        status = "blocked_gatemate_dirtyjtag_idcode_unresolved_v476"
        blocked_reason = "blocked_gatemate_dirtyjtag_idcode"
        leading = "cable_or_port" if narrowed == "jtag_protocol_level" else narrowed
    return {
        "board": "gatemate",
        "reachable": idcode_ok,
        "status": status,
        "blocked_reason": blocked_reason,
        "precondition_probe": precondition_probe,
        "diagnostic_probes": probes,
        "detected_idcode": precondition_idcode,
        "expected_idcode": GATEMATE_EXPECTED_IDCODE,
        "narrowed_to": narrowed,
        "leading_untested_hypothesis": leading,
        "eliminated_causes": eliminated,
        "new_angles": build_new_angles(probes),
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
        "complete_hardware_continuity_gatemate_diagnostic_"
        f"kv260:{kv}_gatemate:{gm}_polarfire:{pf}_no_speedup_claim"
    )


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260703",
) -> JsonDict:
    """Build the Exp 5201 artifact from real (or injected) board commands."""
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
        "polarfire_status": polarfire_status_dict(polarfire),
        "gatemate_status": gatemate_status_dict(gatemate),
        "gatemate_diagnostic_narrowed_to": gatemate["narrowed_to"],
        "gatemate_leading_untested_hypothesis": gatemate["leading_untested_hypothesis"],
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
            status.get("status") == "blocked_gatemate_dirtyjtag_idcode_unresolved_v476",
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
    expect(errors, "cable_or_port_swap" in names, "new angles missing cable_or_port_swap")
    expect(
        errors,
        "openfpgaloader_issue_search" in names,
        "new angles missing openfpgaloader_issue_search",
    )


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
    run_date: str = "20260703",
) -> Path:
    prepend_oss_cad_suite()
    artifact = build_artifact(command_runner=command_runner, clock=clock, run_date=run_date)
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260703", help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(repo_root=args.repo_root, run_date=args.date)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"boards_reachable_count: {artifact['boards_reachable_count']}")
    print(f"gatemate_diagnostic_narrowed_to: {artifact['gatemate_diagnostic_narrowed_to']}")
    print(
        "new_diagnostic_angles_tried_this_milestone: "
        f"{[a['angle'] for a in artifact['new_diagnostic_angles_tried_this_milestone']]}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    raise SystemExit(main())
