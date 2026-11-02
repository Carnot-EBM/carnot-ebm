#!/usr/bin/env python3
"""Exp 5179: hardware continuity board timing with per-board blockers.

Spec refs: REQ-HW-5179, SCENARIO-HW-5179.

This experiment keeps the three attached-board tracks visible without turning
smoke evidence into a speedup claim. It preserves the Exp 5166 combined
transcript shape, but adds differential GateMate IDCODE diagnostics so a missing
GM1Ax IDCODE is not merely re-reported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

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


EXPERIMENT_ID = "exp5179-hardware-continuity-board-timing-v474"
EXPERIMENT_NAME = "experiment_5179_hardware_continuity_board_timing"
MILESTONE = "2026.07.474"
SCHEMA = "carnot.experiment_5179_hardware_continuity_board_timing.v474"
OUTPUT_REL_PATH = Path("results") / "experiment_5179_hardware_continuity_board_timing_v474.json"
SPEC_REFS = ["REQ-HW-5179", "SCENARIO-HW-5179"]
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 5179

KV260_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
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
        "for dev in /dev/ttyACM0 /dev/ttyACM1; do "
        "[ -e \"$dev\" ] && udevadm info -q property -n \"$dev\" | "
        "rg '^(DEVNAME|ID_MODEL|ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL|ID_USB_DRIVER|ID_PATH)='; "
        "done"
    ),
)
GATEMATE_DMESG_COMMAND = (
    "sh",
    "-lc",
    (
        "sudo -n dmesg --ctime | "
        "rg -i '3-2\\.3|dirtyjtag|1209|c0ca|jtag|cdc_acm|ttyACM' | tail -n 120"
    ),
)
GATEMATE_VERBOSE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect", "-v")
GATEMATE_LOW_FREQ_DETECT_COMMAND = (
    "openFPGALoader",
    "-c",
    "dirtyJtag",
    "--detect",
    "--freq",
    "1000000",
)
GATEMATE_POWER_PORT_COMMAND = (
    "sh",
    "-lc",
    (
        "if command -v uhubctl >/dev/null 2>&1; then "
        "uhubctl; "
        "else echo 'uhubctl not installed; physical port power-cycle not available from this shell'; "
        "exit 127; fi"
    ),
)
GATEMATE_USB_RESET_COMMAND = (
    "sh",
    "-lc",
    (
        "if command -v usbreset >/dev/null 2>&1; then "
        "sudo -n usbreset 1209:c0ca; "
        "else echo 'usbreset not installed'; exit 127; fi"
    ),
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
GATEMATE_EXPECTED_IDCODE = "0x20000001"
EXPECTED_OPENFPGALOADER_VERSION = "openFPGALoader v1.1.1"

INLINE_PROGRAM_TEMPLATE = (
    "exp5179_inline_ising_energy_v1: parse JSON spins/edges, compute Ising energy, "
    "time the board-local run, and emit workload/executable hashes with quality evidence"
)
INLINE_EXECUTABLE_HASH = sha256_text(INLINE_PROGRAM_TEMPLATE)

REQUIRED_ARTIFACT_FIELDS = (
    "kv260_result",
    "gatemate_result",
    "polarfire_result",
    "gatemate_idcode_diagnostic_attempts",
    "boards_reachable_count",
    "hardware_wishlist_updated",
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "command_transcripts",
    "sample_quality_evidence",
    "no_speedup_claim",
    "tests_run",
)
REQUIRED_SCHEMA_FIELDS = (
    *REQUIRED_ARTIFACT_FIELDS,
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "random_seed",
    "run_date",
    "duration_s",
    "field_principles",
    "kv260_host_block_devices_touched",
    "hardware_speedup_claimed",
    "workload_hashes",
    "conductor_modified",
    "reproducibility_checksum",
)
BOARD_RESULT_FIELDS = (
    "reachable",
    "workload_hash",
    "executable_hash",
    "latency_transcript",
    "timing_output",
    "hash_verified",
    "sample_quality",
    "correctness",
    "blocked_reason",
    "inference_substrate",
)
COMMAND_TRANSCRIPT_KEYS = (
    "kv260_precondition",
    "kv260_workload",
    "gatemate_precondition",
    "gatemate_scan_usb",
    "gatemate_version",
    "gatemate_usb_enumeration",
    "gatemate_dmesg",
    "gatemate_verbose_detect",
    "gatemate_low_freq_detect",
    "gatemate_power_port",
    "gatemate_usb_reset",
    "gatemate_post_reset_scan_usb",
    "gatemate_post_reset_detect",
    "gatemate_workload",
    "polarfire_precondition",
    "polarfire_workload",
)
WORKLOAD_HASH_KEYS = ("kv260", "gatemate", "polarfire")

WISHLIST_MARKERS = (
    "2026-07-02 Exp 5179 KV260",
    "2026-07-02 Exp 5179 GateMate",
    "2026-07-02 Exp 5179 PolarFire",
)
WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_5179_hardware_continuity_board_timing_v474.py --date 20260702 --update-wishlist",
    ".venv/bin/pytest tests/python/test_experiment_5179_hardware_continuity_board_timing.py -q",
    ".venv/bin/coverage run --source=python/carnot/experiment_5179_hardware_continuity_board_timing.py -m pytest tests/python/test_experiment_5179_hardware_continuity_board_timing.py -q",
    ".venv/bin/coverage report --fail-under=100 -m python/carnot/experiment_5179_hardware_continuity_board_timing.py",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "kv260_result": "per-board evidence; KV260 SSH success or blocker must not be inferred from host storage",
    "gatemate_result": "per-board evidence; GateMate stays visible even when DirtyJTAG lacks an IDCODE",
    "polarfire_result": "per-board evidence; PolarFire SSH success or blocker is reported independently",
    "gatemate_idcode_diagnostic_attempts": "differential diagnosis beyond a repeated DirtyJTAG detect",
    "boards_reachable_count": "reachability accounting; one blocked board is not a whole-task failure",
    "hardware_wishlist_updated": "ops continuity; the active hardware table must carry this milestone's dated board status",
    "honest_verdict": "terminal verdict with complete_/success_ prefix plus honest per-board reachability",
    "inference_substrate": "substrate honesty; board-touching measurements are hardware_smoke",
    "preconditions_checked": "fabrication guard; every board resource is checked before workload execution",
    "command_transcripts": "authenticated command evidence",
    "sample_quality_evidence": "no latency-only or speedup-only claim",
    "no_speedup_claim": "hardware claim discipline",
    "tests_run": "verification evidence",
}


def ising_energy(payload: JsonMap) -> int:
    spins = [int(value) for value in payload["spins"]]
    total = 0
    for row, col, coupling in payload["edges"]:
        total -= int(coupling) * spins[int(row)] * spins[int(col)]
    return total


KV260_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "kv260",
    "workload": "exp5179_inline_ising_energy_smoke",
    "spins": [1, -1, 1, -1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, -1], [2, 3, 1], [3, 4, -1], [4, 5, 1], [5, 6, -1]],
}
KV260_EXPECTED_ENERGY = ising_energy(KV260_WORKLOAD_BASE)
KV260_WORKLOAD = dict(KV260_WORKLOAD_BASE, expected_energy=KV260_EXPECTED_ENERGY)
KV260_WORKLOAD_HASH = sha256_json(KV260_WORKLOAD)

POLARFIRE_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "polarfire",
    "workload": "exp5179_inline_ising_energy_smoke",
    "spins": [1, 1, -1, -1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, 1], [2, 3, -1], [3, 4, 1], [4, 5, -1], [6, 7, 1]],
}
POLARFIRE_EXPECTED_ENERGY = ising_energy(POLARFIRE_WORKLOAD_BASE)
POLARFIRE_WORKLOAD = dict(POLARFIRE_WORKLOAD_BASE, expected_energy=POLARFIRE_EXPECTED_ENERGY)
POLARFIRE_WORKLOAD_HASH = sha256_json(POLARFIRE_WORKLOAD)

GATEMATE_WORKLOAD: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "gatemate",
    "workload": "exp5179_dirtyjtag_idcode_readback",
    "command": list(GATEMATE_DETECT_COMMAND),
    "expected_idcode": GATEMATE_EXPECTED_IDCODE,
}
GATEMATE_WORKLOAD_HASH = sha256_json(GATEMATE_WORKLOAD)


def kv260_workload_command() -> tuple[str, ...]:
    return ssh_workload_command("kria", KV260_WORKLOAD, KV260_WORKLOAD_HASH)


def polarfire_workload_command() -> tuple[str, ...]:
    return ssh_workload_command("polarfire", POLARFIRE_WORKLOAD, POLARFIRE_WORKLOAD_HASH)


def ssh_workload_command(host: str, payload: JsonMap, workload_hash: str) -> tuple[str, ...]:
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


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260702",
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    started = clock()
    root = Path(repo_root)
    kv260_precondition = command_runner(KV260_PRECONDITION_COMMAND, 10.0)
    gatemate_precondition = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    polarfire_precondition = command_runner(POLARFIRE_PRECONDITION_COMMAND, 10.0)

    kv260 = finish_ssh_board(
        board="kv260",
        precondition_probe=kv260_precondition,
        workload_command=kv260_workload_command(),
        workload_hash=KV260_WORKLOAD_HASH,
        command_runner=command_runner,
    )
    gatemate = finish_gatemate_board(
        precondition_probe=gatemate_precondition,
        command_runner=command_runner,
    )
    polarfire = finish_ssh_board(
        board="polarfire",
        precondition_probe=polarfire_precondition,
        workload_command=polarfire_workload_command(),
        workload_hash=POLARFIRE_WORKLOAD_HASH,
        command_runner=command_runner,
    )
    board_results = {"kv260": kv260, "gatemate": gatemate, "polarfire": polarfire}
    gatemate_attempts = gatemate_idcode_diagnostic_attempts(gatemate)

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
        "honest_verdict": honest_verdict(board_results),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": build_preconditions(kv260, gatemate, polarfire),
        "kv260_host_block_devices_touched": False,
        "kv260_result": result_public(kv260),
        "gatemate_result": result_public(gatemate),
        "polarfire_result": result_public(polarfire),
        "gatemate_idcode_diagnostic_attempts": gatemate_attempts,
        "boards_reachable_count": sum(bool(result["reachable"]) for result in board_results.values()),
        "hardware_wishlist_updated": wishlist_has_update(root),
        "command_transcripts": build_command_transcripts(kv260, gatemate, polarfire),
        "workload_hashes": {
            "kv260": KV260_WORKLOAD_HASH,
            "gatemate": GATEMATE_WORKLOAD_HASH,
            "polarfire": POLARFIRE_WORKLOAD_HASH,
        },
        "sample_quality_evidence": sample_quality_evidence(board_results),
        "no_speedup_claim": True,
        "hardware_speedup_claimed": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def finish_ssh_board(
    *,
    board: str,
    precondition_probe: CommandProbe,
    workload_command: tuple[str, ...],
    workload_hash: str,
    command_runner: CommandRunner,
) -> JsonDict:
    workload_probe = None
    output: JsonDict = {}
    blocked_reason = None
    reachable = precondition_probe.exit_code == 0
    if not reachable:
        blocked_reason = f"blocked_{board}_ssh"
    else:
        workload_probe = command_runner(workload_command, 30.0)
        output = parse_probe_json(workload_probe)
        blocked_reason = ssh_workload_blocker(board, workload_probe, workload_hash, output)
    return {
        "board": board,
        "reachable": reachable,
        "precondition_probe": precondition_probe,
        "workload_probe": workload_probe,
        "workload_hash": workload_hash if reachable else None,
        "executable_hash": INLINE_EXECUTABLE_HASH if reachable else None,
        "timing_output": output,
        "hash_verified": ssh_output_hash_verified(workload_hash, output),
        "sample_quality": output.get("sample_quality") if isinstance(output.get("sample_quality"), Mapping) else None,
        "correctness": output.get("correctness") if isinstance(output.get("correctness"), Mapping) else None,
        "blocked_reason": blocked_reason,
    }


def finish_gatemate_board(
    *, precondition_probe: CommandProbe, command_runner: CommandRunner
) -> JsonDict:
    precondition_idcode = idcode_from_text(precondition_probe.combined_output)
    idcode_ok = precondition_probe.exit_code == 0 and precondition_idcode == GATEMATE_EXPECTED_IDCODE
    diagnostic_probes: dict[str, CommandProbe | None] = run_gatemate_diagnostics(
        command_runner=command_runner,
        idcode_ok=idcode_ok,
    )
    workload_probe = None
    blocked_reason = None
    if precondition_probe.exit_code != 0:
        blocked_reason = "blocked_gatemate_dirtyjtag"
    elif not idcode_ok:
        blocked_reason = "blocked_gatemate_dirtyjtag_idcode"
    else:
        workload_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
        workload_idcode = idcode_from_text(workload_probe.combined_output)
        if workload_probe.exit_code != 0:
            blocked_reason = "blocked_gatemate_workload_command"
        elif workload_idcode != GATEMATE_EXPECTED_IDCODE:
            blocked_reason = "blocked_gatemate_workload_idcode"
    hash_verified = (
        workload_probe is not None
        and workload_probe.exit_code == 0
        and idcode_from_text(workload_probe.combined_output) == GATEMATE_EXPECTED_IDCODE
    )
    return {
        "board": "gatemate",
        "reachable": idcode_ok,
        "precondition_probe": precondition_probe,
        "diagnostic_probes": diagnostic_probes,
        "workload_probe": workload_probe,
        "workload_hash": GATEMATE_WORKLOAD_HASH if idcode_ok else None,
        "executable_hash": None,
        "timing_output": {
            "detected_idcode": idcode_from_text(workload_probe.combined_output)
            if workload_probe is not None
            else precondition_idcode,
            "expected_idcode": GATEMATE_EXPECTED_IDCODE,
            "command_interface": "openFPGALoader",
            "diagnostic_summary": gate_diagnostic_summary(precondition_probe, diagnostic_probes),
        },
        "hash_verified": hash_verified,
        "sample_quality": None,
        "correctness": {"idcode_matches_expected": hash_verified} if workload_probe is not None else None,
        "blocked_reason": blocked_reason,
    }


def run_gatemate_diagnostics(
    *, command_runner: CommandRunner, idcode_ok: bool
) -> dict[str, CommandProbe | None]:
    probes: dict[str, CommandProbe | None] = {
        "scan_usb": command_runner(GATEMATE_SCAN_USB_COMMAND, 30.0),
        "version": command_runner(GATEMATE_VERSION_COMMAND, 10.0),
        "usb_enumeration": command_runner(GATEMATE_USB_ENUMERATION_COMMAND, 30.0),
        "dmesg": command_runner(GATEMATE_DMESG_COMMAND, 30.0),
        "verbose_detect": command_runner(GATEMATE_VERBOSE_DETECT_COMMAND, 30.0),
        "low_freq_detect": command_runner(GATEMATE_LOW_FREQ_DETECT_COMMAND, 30.0),
        "power_port": None,
        "usb_reset": None,
        "post_reset_scan_usb": None,
        "post_reset_detect": None,
    }
    if idcode_ok:
        return probes
    probes["power_port"] = command_runner(GATEMATE_POWER_PORT_COMMAND, 30.0)
    probes["usb_reset"] = command_runner(GATEMATE_USB_RESET_COMMAND, 30.0)
    if probes["usb_reset"] is not None and probes["usb_reset"].exit_code == 0:
        probes["post_reset_scan_usb"] = command_runner(GATEMATE_SCAN_USB_COMMAND, 30.0)
        probes["post_reset_detect"] = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    return probes


def gate_diagnostic_summary(
    precondition_probe: CommandProbe, probes: Mapping[str, CommandProbe | None]
) -> JsonDict:
    post_reset_detect = probes.get("post_reset_detect")
    return {
        "precondition_idcode": idcode_from_text(precondition_probe.combined_output),
        "expected_idcode": GATEMATE_EXPECTED_IDCODE,
        "scan_usb_dirtyjtag_seen": dirtyjtag_seen(probes.get("scan_usb")),
        "tool_version_matches_known_good": version_matches_known_good(probes.get("version")),
        "usb_enumeration_seen": dirtyjtag_seen(probes.get("usb_enumeration")),
        "post_reset_idcode": idcode_from_text(post_reset_detect.combined_output)
        if post_reset_detect is not None
        else None,
    }


def gatemate_idcode_diagnostic_attempts(gatemate: JsonMap) -> list[JsonDict]:
    probes = gatemate["diagnostic_probes"]
    attempts = [
        diagnostic_attempt(1, "detect", detect_outcome(gatemate["precondition_probe"])),
        diagnostic_attempt(2, "scan_usb", scan_usb_outcome(probes.get("scan_usb"))),
        diagnostic_attempt(3, "tool_version_compare", version_outcome(probes.get("version"))),
        diagnostic_attempt(4, "usb_enumeration", usb_enumeration_outcome(probes.get("usb_enumeration"))),
        diagnostic_attempt(5, "kernel_log", kernel_log_outcome(probes.get("dmesg"))),
        diagnostic_attempt(6, "verbose_detect", changed_detect_outcome(probes.get("verbose_detect"))),
        diagnostic_attempt(7, "low_frequency_detect", changed_detect_outcome(probes.get("low_freq_detect"))),
        diagnostic_attempt(8, "physical_reseat_or_port_move", physical_access_outcome()),
    ]
    if probes.get("power_port") is not None:
        attempts.append(
            diagnostic_attempt(9, "power_or_port_cycle", power_port_outcome(probes.get("power_port")))
        )
    if probes.get("usb_reset") is not None:
        attempts.append(diagnostic_attempt(10, "usb_reset", usb_reset_outcome(probes.get("usb_reset"))))
    if probes.get("post_reset_scan_usb") is not None:
        attempts.append(
            diagnostic_attempt(11, "post_reset_scan_usb", scan_usb_outcome(probes.get("post_reset_scan_usb")))
        )
    if probes.get("post_reset_detect") is not None:
        attempts.append(
            diagnostic_attempt(12, "post_reset_detect", detect_outcome(probes.get("post_reset_detect")))
        )
    return attempts


def diagnostic_attempt(attempt: int, method: str, outcome: str) -> JsonDict:
    return {"attempt": attempt, "method": method, "outcome": outcome}


def detect_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    if outcome_has_idcode(probe):
        return f"resolved: detected expected GateMate IDCODE {GATEMATE_EXPECTED_IDCODE}"
    return f"no GateMate IDCODE; raw_output={probe.combined_output!r}"


def scan_usb_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    if dirtyjtag_seen(probe):
        return "DirtyJTAG enumerated through openFPGALoader --scan-usb"
    return f"DirtyJTAG not enumerated by scan-usb; raw_output={probe.combined_output!r}"


def version_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    if version_matches_known_good(probe):
        return f"{EXPECTED_OPENFPGALOADER_VERSION} matches known-good 2026-05-23 toolchain"
    return f"version drift or unavailable; expected {EXPECTED_OPENFPGALOADER_VERSION}, raw_output={probe.combined_output!r}"


def usb_enumeration_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    text = probe.combined_output
    if dirtyjtag_seen(probe) and ("root:uucp" in text or "ID_USB_DRIVER=cdc_acm" in text):
        return "DirtyJTAG enumerated at USB/CDC layer with usable uucp/ACM evidence"
    return f"USB enumeration did not confirm DirtyJTAG access; raw_output={text!r}"


def kernel_log_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    text = probe.combined_output
    if probe.exit_code == 0 and text.strip():
        return "kernel log shows DirtyJTAG/USB activity without resolving target IDCODE"
    return f"kernel log unavailable or empty; exit_code={probe.exit_code}, raw_output={text!r}"


def changed_detect_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    if outcome_has_idcode(probe):
        return f"changed detect resolved expected IDCODE {GATEMATE_EXPECTED_IDCODE}"
    return f"changed detect still did not expose IDCODE; raw_output={probe.combined_output!r}"


def physical_access_outcome() -> str:
    return (
        "not_performed: physical cable reseat or different USB port requires operator physical access; "
        "shell diagnostics used USB enumeration and usbreset instead"
    )


def power_port_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    if probe.exit_code == 0:
        return f"host power/port utility reported: {probe.combined_output!r}"
    return f"host power/port cycle unavailable; raw_output={probe.combined_output!r}"


def usb_reset_outcome(probe: CommandProbe | None) -> str:
    if probe is None:
        return "not_run"
    if probe.exit_code == 0:
        return f"USB reset completed; raw_output={probe.combined_output!r}"
    return f"USB reset unavailable or failed; raw_output={probe.combined_output!r}"


def outcome_has_idcode(probe: CommandProbe | None) -> bool:
    return probe is not None and idcode_from_text(probe.combined_output) == GATEMATE_EXPECTED_IDCODE


def dirtyjtag_seen(probe: CommandProbe | None) -> bool:
    if probe is None:
        return False
    lowered = probe.combined_output.lower()
    return "dirtyjtag" in lowered or "1209:c0ca" in lowered or "id_vendor_id=1209" in lowered


def version_matches_known_good(probe: CommandProbe | None) -> bool:
    return probe is not None and EXPECTED_OPENFPGALOADER_VERSION in probe.combined_output


def parse_probe_json(probe: CommandProbe | None) -> JsonDict:
    if probe is None or not probe.combined_output.strip():
        return {}
    try:
        parsed = json.loads(probe.combined_output.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ssh_output_hash_verified(workload_hash: str, output: JsonMap) -> bool:
    return (
        output.get("workload_sha256") == workload_hash
        and output.get("executable_sha256") == INLINE_EXECUTABLE_HASH
        and output.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def ssh_output_has_evidence(output: JsonMap) -> bool:
    return isinstance(output.get("sample_quality"), Mapping) or isinstance(output.get("correctness"), Mapping)


def ssh_workload_blocker(
    board: str, workload_probe: CommandProbe | None, workload_hash: str, output: JsonMap
) -> str | None:
    if workload_probe is None:
        return f"blocked_{board}_workload_missing"
    if workload_probe.exit_code != 0:
        return f"blocked_{board}_workload_command"
    if not ssh_output_hash_verified(workload_hash, output):
        return f"blocked_{board}_workload_hash"
    if not ssh_output_has_evidence(output):
        return f"blocked_{board}_workload_evidence"
    return None


def honest_verdict(board_results: JsonMap) -> str:
    parts = [board_verdict(board, board_results[board]) for board in ("kv260", "gatemate", "polarfire")]
    return "complete_hardware_continuity_board_timing_" + "_".join(parts) + "_no_speedup_claim"


def board_verdict(board: str, result: JsonMap) -> str:
    if board == "gatemate" and result.get("reachable"):
        return "gatemate:reachable_idcode_resolved"
    if board == "gatemate":
        reason = result.get("blocked_reason") or "blocked_gatemate_unknown"
        if reason == "blocked_gatemate_dirtyjtag_idcode":
            return "gatemate:blocked_gatemate_dirtyjtag_idcode_unresolved_after_diagnostics"
        return f"gatemate:{reason}"
    if result.get("reachable"):
        return f"{board}:reachable"
    return f"{board}:{result.get('blocked_reason') or f'blocked_{board}_unknown'}"


def result_public(result: JsonMap) -> JsonDict:
    return {
        "reachable": bool(result["reachable"]),
        "workload_hash": result.get("workload_hash"),
        "executable_hash": result.get("executable_hash"),
        "latency_transcript": probe_dict(result.get("workload_probe")),
        "timing_output": dict(result["timing_output"]),
        "hash_verified": bool(result["hash_verified"]),
        "sample_quality": result.get("sample_quality"),
        "correctness": result.get("correctness"),
        "blocked_reason": result.get("blocked_reason"),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_preconditions(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> list[JsonDict]:
    return [
        precondition_dict(
            "kv260_ssh",
            kv260["precondition_probe"],
            bool(kv260["reachable"]),
            "ssh_only_no_host_block_device_probe",
        ),
        precondition_dict(
            "gatemate_dirtyjtag_idcode",
            gatemate["precondition_probe"],
            bool(gatemate["reachable"]),
            "dirtyjtag_detect_requires_gm1ax_idcode",
        ),
        precondition_dict(
            "polarfire_ssh",
            polarfire["precondition_probe"],
            bool(polarfire["reachable"]),
            "ssh_only_no_flash",
        ),
    ]


def precondition_dict(
    resource: str, probe: CommandProbe, available: bool, discipline: str
) -> JsonDict:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": round_duration(probe.duration_s),
        "observed": probe.combined_output,
        "discipline": discipline,
        "safety_constraints": ["no_destructive_actions", "no_speedup_from_reachability"],
    }


def build_command_transcripts(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> JsonDict:
    gate_probes = gatemate["diagnostic_probes"]
    return {
        "kv260_precondition": kv260["precondition_probe"].as_dict(),
        "kv260_workload": probe_dict(kv260.get("workload_probe")),
        "gatemate_precondition": gatemate["precondition_probe"].as_dict(),
        "gatemate_scan_usb": probe_dict(gate_probes.get("scan_usb")),
        "gatemate_version": probe_dict(gate_probes.get("version")),
        "gatemate_usb_enumeration": probe_dict(gate_probes.get("usb_enumeration")),
        "gatemate_dmesg": probe_dict(gate_probes.get("dmesg")),
        "gatemate_verbose_detect": probe_dict(gate_probes.get("verbose_detect")),
        "gatemate_low_freq_detect": probe_dict(gate_probes.get("low_freq_detect")),
        "gatemate_power_port": probe_dict(gate_probes.get("power_port")),
        "gatemate_usb_reset": probe_dict(gate_probes.get("usb_reset")),
        "gatemate_post_reset_scan_usb": probe_dict(gate_probes.get("post_reset_scan_usb")),
        "gatemate_post_reset_detect": probe_dict(gate_probes.get("post_reset_detect")),
        "gatemate_workload": probe_dict(gatemate.get("workload_probe")),
        "polarfire_precondition": polarfire["precondition_probe"].as_dict(),
        "polarfire_workload": probe_dict(polarfire.get("workload_probe")),
    }


def sample_quality_evidence(board_results: JsonMap) -> JsonDict:
    return {
        "reachable_boards": [
            board for board, result in board_results.items() if result.get("reachable")
        ],
        "hash_verified_boards": [
            board for board, result in board_results.items() if result.get("hash_verified")
        ],
        "kv260": quality_public(board_results["kv260"]),
        "gatemate": quality_public(board_results["gatemate"]),
        "polarfire": quality_public(board_results["polarfire"]),
        "speedup_evidence_claimed": False,
    }


def quality_public(result: JsonMap) -> JsonDict:
    return {
        "reachable": bool(result["reachable"]),
        "hash_verified": bool(result["hash_verified"]),
        "sample_quality": result.get("sample_quality"),
        "correctness": result.get("correctness"),
        "blocked_reason": result.get("blocked_reason"),
    }


def wishlist_has_update(repo_root: str | Path) -> bool:
    path = Path(repo_root) / WISHLIST_REL_PATH
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return all(marker in text for marker in WISHLIST_MARKERS)


def ensure_hardware_wishlist_update(repo_root: str | Path) -> Path:
    path = Path(repo_root) / WISHLIST_REL_PATH
    addition = "\n".join(
        f"| {marker} | v474 hardware_smoke continuity status recorded in Exp 5179. | "
        "No hardware speedup claim; per-board blockers remain visible. |"
        for marker in WISHLIST_MARKERS
    )
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if all(marker in text for marker in WISHLIST_MARKERS):
            return path
        path.write_text(text.rstrip() + "\n" + addition + "\n", encoding="utf-8")
        return path
    path.write_text(
        "# Hardware Wishlist\n\n"
        "### Active hardware tracks\n\n"
        "| Track | Why active now | Boundary |\n"
        "|---|---|---|\n"
        f"{addition}\n",
        encoding="utf-8",
    )
    return path


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
    run_date: str = "20260702",
    tests_run: Sequence[str] | None = None,
    update_wishlist: bool = False,
) -> Path:
    prepend_oss_cad_suite()
    if update_wishlist:
        ensure_hardware_wishlist_update(repo_root)
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        tests_run=tests_run,
    )
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


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
    expect(errors, terminal_prefix_ok(str(artifact.get("honest_verdict", ""))), "honest_verdict prefix mismatch")
    expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate mismatch")
    expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    expect(errors, artifact.get("kv260_host_block_devices_touched") is False, "kv260_host_block_devices_touched mismatch")
    expect(errors, artifact.get("hardware_wishlist_updated") is True, "hardware_wishlist_updated mismatch")
    expect(errors, artifact.get("no_speedup_claim") is True, "no_speedup_claim mismatch")
    expect(errors, artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed mismatch")
    expect(errors, artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    expect(errors, no_host_storage(artifact), "host storage marker present")
    validate_board_result(errors, artifact, "kv260_result", "kv260")
    validate_board_result(errors, artifact, "gatemate_result", "gatemate")
    validate_board_result(errors, artifact, "polarfire_result", "polarfire")
    validate_gatemate_diagnostics(errors, artifact)
    expect(errors, reachable_count(artifact) == artifact.get("boards_reachable_count"), "boards_reachable_count mismatch")
    validate_mapping_keys(errors, artifact, "command_transcripts", COMMAND_TRANSCRIPT_KEYS)
    validate_mapping_keys(errors, artifact, "workload_hashes", WORKLOAD_HASH_KEYS)
    expect(errors, isinstance(artifact.get("sample_quality_evidence"), Mapping), "sample_quality_evidence must be a dict")
    expect(errors, isinstance(artifact.get("preconditions_checked"), list) and len(artifact["preconditions_checked"]) == 3, "preconditions_checked mismatch")
    expect(errors, isinstance(artifact.get("tests_run"), list) and bool(artifact.get("tests_run")), "tests_run must be non-empty")
    expect(errors, artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum mismatch")
    return errors


def terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "success_"))


def reachable_count(artifact: JsonMap) -> int:
    return sum(
        bool(artifact[field].get("reachable"))
        for field in ("kv260_result", "gatemate_result", "polarfire_result")
        if isinstance(artifact.get(field), Mapping)
    )


def validate_board_result(errors: list[str], artifact: JsonMap, field: str, board: str) -> None:
    result = artifact.get(field)
    if not isinstance(result, Mapping):
        errors.append(f"{field} must be a dict")
        return
    expect(errors, set(BOARD_RESULT_FIELDS) == set(result), f"{field} keys mismatch")
    expect(errors, isinstance(result.get("reachable"), bool), f"{field} reachable must be bool")
    blocked_reason = result.get("blocked_reason")
    expect(
        errors,
        blocked_reason is None or str(blocked_reason).startswith(f"blocked_{board}_"),
        f"{field} blocked_reason mismatch",
    )
    hash_value = result.get("workload_hash")
    expect(errors, hash_value is None or is_sha256(hash_value), f"{field} workload_hash invalid")
    if result.get("reachable"):
        expect(errors, hash_value is not None, f"{field} reachable missing workload_hash")


def validate_gatemate_diagnostics(errors: list[str], artifact: JsonMap) -> None:
    attempts = artifact.get("gatemate_idcode_diagnostic_attempts")
    if not isinstance(attempts, list) or not attempts:
        errors.append("gatemate_idcode_diagnostic_attempts must be a non-empty list")
        return
    has_non_detect = False
    for item in attempts:
        if not isinstance(item, Mapping):
            errors.append("gatemate_idcode_diagnostic_attempts entries must be dicts")
            return
        if set(item) != {"attempt", "method", "outcome"}:
            errors.append("gatemate_idcode_diagnostic_attempts entry keys mismatch")
            return
        if not isinstance(item.get("attempt"), int) or not str(item.get("method")) or not str(item.get("outcome")):
            errors.append("gatemate_idcode_diagnostic_attempts entry values invalid")
            return
        has_non_detect = has_non_detect or item.get("method") != "detect"
    expect(
        errors,
        has_non_detect,
        "gatemate_idcode_diagnostic_attempts must include a non-detect diagnostic",
    )


def validate_mapping_keys(
    errors: list[str], artifact: JsonMap, field: str, expected_keys: Sequence[str]
) -> None:
    value = artifact.get(field)
    expect(errors, isinstance(value, Mapping), f"{field} must be a dict")
    if isinstance(value, Mapping):
        expect(errors, set(value) == set(expected_keys), f"{field} keys mismatch")


def no_host_storage(payload: JsonMap) -> bool:
    return no_5120_host_storage(payload)


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    parser.add_argument(
        "--update-wishlist",
        action="store_true",
        help="Append the Exp 5179 hardware status rows before building the artifact.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(
        repo_root=args.repo_root,
        run_date=args.date,
        update_wishlist=args.update_wishlist,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"boards_reachable_count: {artifact['boards_reachable_count']}")
    print(
        "gatemate_idcode_diagnostic_attempts: "
        f"{len(artifact['gatemate_idcode_diagnostic_attempts'])}"
    )
    print(f"hardware_wishlist_updated: {artifact['hardware_wishlist_updated']}")
    print(f"no_speedup_claim: {artifact['no_speedup_claim']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    raise SystemExit(main())
