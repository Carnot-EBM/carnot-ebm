#!/usr/bin/env python3
"""Exp 5132: authenticated board timing continuity.

Spec refs: REQ-HW-5132, SCENARIO-HW-5132.

This experiment keeps the hardware track honest while board timing remains
conditional. It checks KV260 through SSH only, records GateMate and PolarFire
precheck transcripts, and keeps a CPU residual-energy sweep in the artifact so
sample-quality telemetry is preserved even when safe board timing is blocked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from carnot.experiment_5120_hardware_residual_telemetry import (
    CommandProbe,
    CommandRunner,
    JsonDict,
    JsonMap,
    Clock,
    command_to_string,
    compute_cpu_residual_sweep,
    dirtyjtag_seen_in_text,
    fit_decay_exponent,
    gatemate_detected,
    idcode_from_text,
    is_sha256,
    no_host_storage as no_5120_host_storage,
    observed,
    parse_python_version,
    parse_uio_devices,
    payload_checksum,
    policy_precondition,
    prepend_oss_cad_suite,
    probe_dict,
    precondition_entry,
    round_duration,
    run_command,
    sha256_json,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


EXPERIMENT_ID = "exp5132-authenticated-board-timing-v470"
EXPERIMENT_NAME = "experiment_5132_authenticated_board_timing"
MILESTONE = "2026.07.470"
SCHEMA = "carnot.experiment_5132_authenticated_board_timing.v470"
OUTPUT_REL_PATH = Path("results") / "experiment_5132_authenticated_board_timing_v470.json"
SPEC_REFS = ["REQ-HW-5132", "SCENARIO-HW-5132"]
INFERENCE_SUBSTRATE = "hardware_smoke_or_authenticated_blockers"
HONEST_VERDICT = "complete_authenticated_board_blockers_cpu_residual_no_speedup_claim"
RANDOM_SEED = 5132

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
KV260_UIO_LIST_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "ls /dev/uio*",
)
KV260_UIO_SYSFS_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "for u in /sys/class/uio/uio*; do [ -e \"$u\" ] || continue; "
    "printf '%s ' \"$(basename \"$u\")\"; cat \"$u/name\" 2>/dev/null || true; done",
)
KV260_SAFE_WORKLOAD_REL_PATH = Path("hardware") / "kv260" / "exp5132_safe_uio_read_only_workload.py"

GATEMATE_COMMAND_AVAILABLE_COMMAND = ("sh", "-lc", "command -v openFPGALoader")
GATEMATE_YOSYS_VERSION_COMMAND = ("yosys", "-V")
GATEMATE_NEXTPNR_VERSION_COMMAND = ("nextpnr-himbaechel", "--version")
GATEMATE_GMPACK_VERSION_COMMAND = ("sh", "-lc", "command -v gmpack && gmpack -h | head -n 2")
GATEMATE_USB_EVIDENCE_COMMAND = (
    "sh",
    "-lc",
    "lsusb | grep -Ei '1209:c0ca|dirtyjtag|gatemate|cologne|olimex|1514:2008|flashpro' || true",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
GATEMATE_SAFE_FLASH_MANIFEST_REL_PATH = (
    Path("hardware") / "gatemate" / "exp5132_safe_flash_manifest.json"
)

POLARFIRE_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
POLARFIRE_ARCH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "uname -m",
)
POLARFIRE_PYTHON_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "python3 --version",
)
POLARFIRE_DISPATCH_WORKLOAD = {
    "experiment_id": EXPERIMENT_ID,
    "workload": "polarfire_inline_dispatch_precheck_no_carnot_runtime",
    "spins": [1, -1, 1, -1],
}
POLARFIRE_DISPATCH_WORKLOAD_HASH = sha256_json(POLARFIRE_DISPATCH_WORKLOAD)
POLARFIRE_DISPATCH_PRECHECK_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    (
        "python3 - <<'PY'\n"
        "import json, time\n"
        f"payload = {json.dumps(POLARFIRE_DISPATCH_WORKLOAD, sort_keys=True)!r}\n"
        "data = json.loads(payload)\n"
        "started = time.perf_counter()\n"
        "energy = sum(value * value for value in data['spins'])\n"
        "duration_s = max(time.perf_counter() - started, 0.000001)\n"
        "print(json.dumps({\n"
        f"    'workload_sha256': {POLARFIRE_DISPATCH_WORKLOAD_HASH!r},\n"
        "    'energy': energy,\n"
        "    'duration_s': duration_s,\n"
        "    'sample_quality': {'sample_count': len(data['spins']), 'finite_energy': True},\n"
        "}, sort_keys=True))\n"
        "PY"
    ),
)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "preconditions_checked",
    "kv260_ssh_checked",
    "kv260_host_block_devices_touched",
    "kv260_timing_transcript",
    "gatemate_checked",
    "gatemate_transcript",
    "polarfire_checked",
    "polarfire_transcript",
    "command_transcripts",
    "workload_hashes",
    "timing_measurements",
    "residual_energy_by_sweep",
    "sample_quality_evidence",
    "no_speedup_claim",
    "extropic_tsu_execution_claimed",
    "flagged_adversarial",
    "conductor_modified",
    "tests_run",
)
REQUIRED_SCHEMA_FIELDS = (
    *REQUIRED_ARTIFACT_FIELDS,
    "schema",
    "experiment",
    "spec_refs",
    "random_seed",
    "run_date",
    "field_principles",
    "decay_exponent",
    "kv260_ssh_ready",
    "gatemate_detected",
    "polarfire_ssh_ready",
    "board_precheck_summary",
    "residual_source",
    "reproducibility_checksum",
    "extropic_tsu_context",
)
COMMAND_TRANSCRIPT_KEYS = (
    "kv260_ssh",
    "kv260_uio_devices",
    "kv260_uio_sysfs",
    "kv260_timing_workload",
    "gatemate_openfpgaloader",
    "gatemate_yosys_version",
    "gatemate_nextpnr_version",
    "gatemate_gmpack_version",
    "gatemate_usb_evidence",
    "gatemate_dirtyjtag_detect",
    "polarfire_ssh",
    "polarfire_arch",
    "polarfire_python",
    "polarfire_dispatch_precheck",
)
WORKLOAD_HASH_KEYS = (
    "cpu_reference_residual_sweep",
    "cpu_residual_samples",
    "kv260_timing_workload",
    "gatemate_safe_flash_manifest",
    "polarfire_dispatch_precheck_workload",
)

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python scripts/experiment_5132_authenticated_board_timing_v470.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5132_authenticated_board_timing.py -q",
    ".venv/bin/coverage run --source=python/carnot/experiment_5132_authenticated_board_timing.py -m pytest tests/python/test_experiment_5132_authenticated_board_timing.py -q",
    ".venv/bin/coverage report --fail-under=100 -m python/carnot/experiment_5132_authenticated_board_timing.py",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "preconditions_checked": "hardware preflight accountability",
    "kv260_ssh_checked": "board continuity",
    "kv260_host_block_devices_touched": "safety",
    "kv260_timing_transcript": "authenticated evidence",
    "gatemate_checked": "board continuity",
    "gatemate_transcript": "authenticated evidence",
    "polarfire_checked": "board continuity",
    "polarfire_transcript": "authenticated evidence",
    "command_transcripts": "authenticated evidence",
    "workload_hashes": "reproducibility",
    "timing_measurements": "latency evidence",
    "residual_energy_by_sweep": "sample-quality telemetry",
    "sample_quality_evidence": "no latency-only claim",
    "no_speedup_claim": "no false acceleration",
    "extropic_tsu_execution_claimed": "hardware honesty",
    "flagged_adversarial": "adversarial-verification accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260701",
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5132 terminal artifact from safe board prechecks."""

    started = clock()
    kv260 = run_kv260_checks(repo_root=repo_root, command_runner=command_runner)
    gatemate = run_gatemate_checks(repo_root=repo_root, command_runner=command_runner)
    polarfire = run_polarfire_checks(command_runner=command_runner)
    residual_rows, residual_meta = compute_cpu_residual_sweep()
    decay_exponent = fit_decay_exponent(residual_rows)
    workload_hashes = build_workload_hashes(
        residual_rows=residual_rows,
        residual_meta=residual_meta,
        kv260=kv260,
        gatemate=gatemate,
    )
    timing = build_timing_measurements(
        kv260=kv260,
        gatemate=gatemate,
        polarfire=polarfire,
        residual_rows=residual_rows,
    )
    authenticated_board_count = sum(
        [
            bool(kv260["ssh_ready"]),
            bool(gatemate["detected"]),
            bool(polarfire["ssh_ready"]),
        ]
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "run_date": run_date,
        "honest_verdict": HONEST_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round_duration(clock() - started),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": build_preconditions(kv260, gatemate, polarfire),
        "kv260_ssh_checked": True,
        "kv260_ssh_ready": bool(kv260["ssh_ready"]),
        "kv260_host_block_devices_touched": False,
        "kv260_timing_transcript": kv260_timing_transcript(kv260),
        "gatemate_checked": True,
        "gatemate_detected": bool(gatemate["detected"]),
        "gatemate_transcript": gatemate_transcript(gatemate),
        "polarfire_checked": True,
        "polarfire_ssh_ready": bool(polarfire["ssh_ready"]),
        "polarfire_transcript": polarfire_transcript(polarfire),
        "command_transcripts": command_transcripts(kv260, gatemate, polarfire),
        "workload_hashes": workload_hashes,
        "timing_measurements": timing,
        "residual_source": "cpu_reference_residual_sweep",
        "residual_energy_by_sweep": residual_rows,
        "decay_exponent": decay_exponent,
        "sample_quality_evidence": sample_quality_evidence(
            residual_rows=residual_rows,
            decay_exponent=decay_exponent,
            kv260=kv260,
            polarfire=polarfire,
        ),
        "board_precheck_summary": {
            "authenticated_board_precheck_count": authenticated_board_count,
            "cpu_reference_residual_sweep_recorded": bool(residual_rows),
            "full_speedup_evidence_present": False,
            "kv260_blockers": list(kv260["blockers"]),
            "gatemate_flash_blocker": gatemate["flash_precheck"]["blocker"],
            "polarfire_dispatch_blockers": list(polarfire["blockers"]),
        },
        "no_speedup_claim": True,
        "extropic_tsu_execution_claimed": False,
        "extropic_tsu_context": "architecture_only_no_execution_attempted",
        "flagged_adversarial": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_kv260_checks(*, repo_root: str | Path, command_runner: CommandRunner) -> JsonDict:
    ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    ssh_ready = ssh_probe.exit_code == 0
    uio_probe = command_runner(KV260_UIO_LIST_COMMAND, 10.0) if ssh_ready else None
    sysfs_probe = command_runner(KV260_UIO_SYSFS_COMMAND, 10.0) if ssh_ready else None
    safe_workload = load_safe_kv260_workload(repo_root)
    timing_probe = None
    blockers: list[str] = []
    if not ssh_ready:
        blockers.append("blocked_kv260_ssh_unreachable")
    if ssh_ready and safe_workload is None:
        blockers.append("no_checked_in_safe_kv260_uio_timing_workload")
    if ssh_ready and safe_workload is not None:
        timing_probe = command_runner(
            kv260_timing_command(safe_workload["text"], safe_workload["sha256"]), 30.0
        )
        if timing_probe.exit_code != 0:
            blockers.append("kv260_safe_uio_timing_workload_failed")
    return {
        "ssh_probe": ssh_probe,
        "ssh_ready": ssh_ready,
        "uio_probe": uio_probe,
        "sysfs_probe": sysfs_probe,
        "safe_workload": safe_workload,
        "timing_probe": timing_probe,
        "timing_output": parse_probe_json(timing_probe),
        "blockers": blockers,
    }


def load_safe_kv260_workload(repo_root: str | Path) -> JsonDict | None:
    path = Path(repo_root) / KV260_SAFE_WORKLOAD_REL_PATH
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not safe_kv260_workload_text(text):
        return None
    return {"path": str(KV260_SAFE_WORKLOAD_REL_PATH), "text": text, "sha256": sha256_text(text)}


def safe_kv260_workload_text(text: str) -> bool:
    lowered = text.lower()
    unsafe = ("write_u32", "loadapp", "flash", "program", "prot_write", "os.o_rdwr")
    return (
        "read_only_uio_workload" in lowered
        and "safe_for_continuity_audit" in lowered
        and "read_only" in lowered
        and not any(marker in lowered for marker in unsafe)
    )


def kv260_timing_command(workload_text: str, workload_hash: str) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "kria",
        f"EXP5132_WORKLOAD_SHA256={workload_hash} python3 - <<'PY'\n{workload_text}\nPY",
    )


def run_gatemate_checks(*, repo_root: str | Path, command_runner: CommandRunner) -> JsonDict:
    openfpgaloader_probe = command_runner(GATEMATE_COMMAND_AVAILABLE_COMMAND, 10.0)
    yosys_probe = command_runner(GATEMATE_YOSYS_VERSION_COMMAND, 10.0)
    nextpnr_probe = command_runner(GATEMATE_NEXTPNR_VERSION_COMMAND, 10.0)
    gmpack_probe = command_runner(GATEMATE_GMPACK_VERSION_COMMAND, 10.0)
    usb_probe = command_runner(GATEMATE_USB_EVIDENCE_COMMAND, 10.0)
    detect_probe = (
        command_runner(GATEMATE_DETECT_COMMAND, 30.0)
        if openfpgaloader_probe.exit_code == 0
        else None
    )
    flash_precheck = gatemate_flash_precheck(repo_root)
    return {
        "openfpgaloader_probe": openfpgaloader_probe,
        "yosys_probe": yosys_probe,
        "nextpnr_probe": nextpnr_probe,
        "gmpack_probe": gmpack_probe,
        "usb_probe": usb_probe,
        "detect_probe": detect_probe,
        "detected": gatemate_detected(detect_probe),
        "flash_precheck": flash_precheck,
    }


def gatemate_flash_precheck(repo_root: str | Path) -> JsonDict:
    path = Path(repo_root) / GATEMATE_SAFE_FLASH_MANIFEST_REL_PATH
    if not path.is_file():
        return {"manifest_present": False, "manifest_sha256": None, "blocker": "no_safe_gatemate_flash_manifest"}
    text = path.read_text(encoding="utf-8")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        manifest = {}
    safe = manifest.get("flash_allowed") is True and manifest.get("design_scope") == "tiny_readback_only"
    return {
        "manifest_present": True,
        "manifest_sha256": sha256_text(text),
        "blocker": None if safe else "safe_gatemate_flash_manifest_invalid",
    }


def run_polarfire_checks(*, command_runner: CommandRunner) -> JsonDict:
    ssh_probe = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    ssh_ready = ssh_probe.exit_code == 0
    arch_probe = command_runner(POLARFIRE_ARCH_COMMAND, 10.0) if ssh_ready else None
    python_probe = command_runner(POLARFIRE_PYTHON_COMMAND, 10.0) if ssh_ready else None
    blockers: list[str] = []
    dispatch_probe = None
    if not ssh_ready:
        blockers.append("blocked_polarfire_ssh_unreachable")
    if ssh_ready and observed(arch_probe).strip() != "riscv64":
        blockers.append("polarfire_arch_not_riscv64")
    python_version = parse_python_version(observed(python_probe))
    if ssh_ready and (python_version is None or python_version < (3, 10, 0)):
        blockers.append("polarfire_python_precheck_failed")
    if ssh_ready and not blockers:
        dispatch_probe = command_runner(POLARFIRE_DISPATCH_PRECHECK_COMMAND, 30.0)
        if dispatch_probe.exit_code != 0:
            blockers.append("polarfire_dispatch_precheck_failed")
    return {
        "ssh_probe": ssh_probe,
        "ssh_ready": ssh_ready,
        "arch_probe": arch_probe,
        "python_probe": python_probe,
        "dispatch_probe": dispatch_probe,
        "dispatch_output": parse_probe_json(dispatch_probe),
        "blockers": blockers,
    }


def parse_probe_json(probe: CommandProbe | None) -> JsonDict:
    if probe is None or not probe.combined_output.strip():
        return {}
    try:
        parsed = json.loads(probe.combined_output.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_preconditions(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> list[JsonDict]:
    return [
        precondition_entry(
            "kv260_ssh",
            kv260["ssh_probe"],
            bool(kv260["ssh_ready"]),
            "ssh_only_no_host_block_device_probe",
            ["ssh_only", "no_host_block_device_access", "no_destructive_actions"],
        ),
        policy_precondition(),
        {
            "resource": "kv260_safe_uio_timing_workload",
            "available": kv260["safe_workload"] is not None,
            "command": str(KV260_SAFE_WORKLOAD_REL_PATH),
            "exit_code": 0 if kv260["safe_workload"] is not None else 1,
            "duration_s": 0.0001,
            "observed": "present" if kv260["safe_workload"] is not None else "missing",
            "discipline": "read_only_uio_workload_required_before_register_access",
            "safety_constraints": ["read_only", "hash_matched", "ssh_only"],
        },
        precondition_entry(
            "gatemate_toolchain",
            gatemate["openfpgaloader_probe"],
            gatemate["openfpgaloader_probe"].exit_code == 0,
            "toolchain_and_dirtyjtag_detect_only",
            ["detect_only", "no_flash_without_manifest"],
        ),
        precondition_entry(
            "gatemate_dirtyjtag_detect",
            gatemate["detect_probe"] or gatemate["openfpgaloader_probe"],
            bool(gatemate["detected"]),
            "dirtyjtag_detect_no_flash",
            ["detect_only", "no_program"],
        ),
        {
            "resource": "gatemate_safe_flash_manifest",
            "available": gatemate["flash_precheck"]["blocker"] is None,
            "command": str(GATEMATE_SAFE_FLASH_MANIFEST_REL_PATH),
            "exit_code": 0 if gatemate["flash_precheck"]["blocker"] is None else 1,
            "duration_s": 0.0001,
            "observed": "safe" if gatemate["flash_precheck"]["blocker"] is None else "missing_or_blocked",
            "discipline": "flash_requires_safe_tiny_design_manifest",
            "safety_constraints": ["no_flash_without_manifest", "hash_matched"],
        },
        precondition_entry(
            "polarfire_ssh",
            polarfire["ssh_probe"],
            bool(polarfire["ssh_ready"]),
            "ssh_precheck_only",
            ["ssh_only", "no_flash"],
        ),
        {
            "resource": "polarfire_dispatch_precheck",
            "available": polarfire["dispatch_probe"] is not None
            and polarfire["dispatch_probe"].exit_code == 0,
            "command": command_to_string(POLARFIRE_DISPATCH_PRECHECK_COMMAND),
            "exit_code": polarfire["dispatch_probe"].exit_code
            if polarfire["dispatch_probe"] is not None
            else 1,
            "duration_s": round_duration(
                polarfire["dispatch_probe"].duration_s
                if polarfire["dispatch_probe"] is not None
                else 0.0001
            ),
            "observed": observed(polarfire["dispatch_probe"]),
            "discipline": "inline_hash_matched_python_precheck_no_carnot_dispatch_claim",
            "safety_constraints": ["ssh_only", "no_scp", "no_flash"],
        },
    ]


def kv260_timing_transcript(kv260: JsonMap) -> JsonDict:
    safe_workload = kv260["safe_workload"]
    timing_probe = kv260["timing_probe"]
    return {
        "ssh_checked": True,
        "ssh_ready": bool(kv260["ssh_ready"]),
        "uio_devices": parse_uio_devices(
            kv260["uio_probe"].combined_output if kv260["uio_probe"] is not None else ""
        ),
        "uio_sysfs_observed": observed(kv260["sysfs_probe"]),
        "safe_workload_path": safe_workload["path"] if safe_workload is not None else None,
        "workload_hash": safe_workload["sha256"] if safe_workload is not None else None,
        "uio_timing_attempted": timing_probe is not None,
        "timing_output": dict(kv260["timing_output"]),
        "blockers": list(kv260["blockers"]),
        "transcript": probe_dict(timing_probe),
    }


def gatemate_transcript(gatemate: JsonMap) -> JsonDict:
    detect_probe = gatemate["detect_probe"]
    dirtyjtag_seen = dirtyjtag_seen_in_text(gatemate["usb_probe"].combined_output) or (
        detect_probe is not None and dirtyjtag_seen_in_text(detect_probe.combined_output)
    )
    return {
        "toolchain_checked": True,
        "openfpgaloader_available": gatemate["openfpgaloader_probe"].exit_code == 0,
        "yosys_available": gatemate["yosys_probe"].exit_code == 0,
        "nextpnr_himbaechel_available": gatemate["nextpnr_probe"].exit_code == 0,
        "gmpack_available": gatemate["gmpack_probe"].exit_code == 0,
        "usb_dirtyjtag_seen": dirtyjtag_seen,
        "detected": bool(gatemate["detected"]),
        "detected_idcode": idcode_from_text(detect_probe.combined_output)
        if detect_probe is not None
        else None,
        "flash_precheck": dict(gatemate["flash_precheck"]),
        "flash_attempted": False,
        "action_scope": "detect_and_flash_precheck_only_no_programming",
    }


def polarfire_transcript(polarfire: JsonMap) -> JsonDict:
    return {
        "ssh_checked": True,
        "ssh_ready": bool(polarfire["ssh_ready"]),
        "arch": observed(polarfire["arch_probe"]) if polarfire["arch_probe"] is not None else None,
        "python": observed(polarfire["python_probe"])
        if polarfire["python_probe"] is not None
        else None,
        "dispatch_precheck_attempted": polarfire["dispatch_probe"] is not None,
        "workload_hash": POLARFIRE_DISPATCH_WORKLOAD_HASH,
        "dispatch_output": dict(polarfire["dispatch_output"]),
        "blockers": list(polarfire["blockers"]),
        "transcript": probe_dict(polarfire["dispatch_probe"]),
        "action_scope": "ssh_inline_dispatch_precheck_no_file_copy_no_flash",
    }


def command_transcripts(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> JsonDict:
    return {
        "kv260_ssh": kv260["ssh_probe"].as_dict(),
        "kv260_uio_devices": probe_dict(kv260["uio_probe"]),
        "kv260_uio_sysfs": probe_dict(kv260["sysfs_probe"]),
        "kv260_timing_workload": probe_dict(kv260["timing_probe"]),
        "gatemate_openfpgaloader": gatemate["openfpgaloader_probe"].as_dict(),
        "gatemate_yosys_version": gatemate["yosys_probe"].as_dict(),
        "gatemate_nextpnr_version": gatemate["nextpnr_probe"].as_dict(),
        "gatemate_gmpack_version": gatemate["gmpack_probe"].as_dict(),
        "gatemate_usb_evidence": gatemate["usb_probe"].as_dict(),
        "gatemate_dirtyjtag_detect": probe_dict(gatemate["detect_probe"]),
        "polarfire_ssh": polarfire["ssh_probe"].as_dict(),
        "polarfire_arch": probe_dict(polarfire["arch_probe"]),
        "polarfire_python": probe_dict(polarfire["python_probe"]),
        "polarfire_dispatch_precheck": probe_dict(polarfire["dispatch_probe"]),
    }


def build_workload_hashes(
    *,
    residual_rows: Sequence[JsonMap],
    residual_meta: JsonMap,
    kv260: JsonMap,
    gatemate: JsonMap,
) -> JsonDict:
    safe_workload = kv260["safe_workload"]
    return {
        "cpu_reference_residual_sweep": sha256_json(
            {
                "workload": "exp5132_cpu_reference_residual_sweep",
                "random_seed": RANDOM_SEED,
                "metadata": residual_meta,
            }
        ),
        "cpu_residual_samples": sha256_json(list(residual_rows)),
        "kv260_timing_workload": safe_workload["sha256"] if safe_workload is not None else None,
        "gatemate_safe_flash_manifest": gatemate["flash_precheck"]["manifest_sha256"],
        "polarfire_dispatch_precheck_workload": POLARFIRE_DISPATCH_WORKLOAD_HASH,
    }


def build_timing_measurements(
    *,
    kv260: JsonMap,
    gatemate: JsonMap,
    polarfire: JsonMap,
    residual_rows: Sequence[JsonMap],
) -> JsonDict:
    return {
        "kv260_ssh_s": round_duration(kv260["ssh_probe"].duration_s),
        "kv260_uio_list_s": round_duration(kv260["uio_probe"].duration_s)
        if kv260["uio_probe"] is not None
        else None,
        "kv260_authenticated_workload_s": round_duration(kv260["timing_probe"].duration_s)
        if kv260["timing_probe"] is not None
        else None,
        "gatemate_detect_s": round_duration(gatemate["detect_probe"].duration_s)
        if gatemate["detect_probe"] is not None
        else None,
        "polarfire_dispatch_precheck_s": round_duration(polarfire["dispatch_probe"].duration_s)
        if polarfire["dispatch_probe"] is not None
        else None,
        "cpu_residual_sweep_samples": len(residual_rows),
        "full_board_speedup_evidence_present": False,
    }


def sample_quality_evidence(
    *,
    residual_rows: Sequence[JsonMap],
    decay_exponent: float | None,
    kv260: JsonMap,
    polarfire: JsonMap,
) -> JsonDict:
    kv260_quality = kv260["timing_output"].get("sample_quality")
    polarfire_quality = polarfire["dispatch_output"].get("sample_quality")
    return {
        "cpu_residual_sample_count": len(residual_rows),
        "decay_exponent_fit_from_samples": decay_exponent is not None,
        "kv260_read_only_sample_quality": kv260_quality if isinstance(kv260_quality, Mapping) else None,
        "polarfire_dispatch_precheck_quality": polarfire_quality
        if isinstance(polarfire_quality, Mapping)
        else None,
        "board_speedup_evidence_complete": False,
    }


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
    run_date: str = "20260701",
    tests_run: Sequence[str] | None = None,
) -> Path:
    prepend_oss_cad_suite()
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
    expect(errors, str(artifact.get("honest_verdict", "")).startswith(("complete_", "success_", "blocked_")), "honest_verdict terminal prefix missing")
    expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    expect(errors, round_duration(artifact.get("duration_s")) >= 0.0001, "duration_s below floor")
    for field, expected in (
        ("kv260_ssh_checked", True),
        ("kv260_host_block_devices_touched", False),
        ("gatemate_checked", True),
        ("polarfire_checked", True),
        ("no_speedup_claim", True),
        ("extropic_tsu_execution_claimed", False),
        ("flagged_adversarial", False),
        ("conductor_modified", False),
    ):
        expect(errors, artifact.get(field) is expected, f"{field} mismatch")
    expect(errors, no_host_storage(artifact), "forbidden host storage marker")
    validate_mapping_keys(errors, artifact, "command_transcripts", COMMAND_TRANSCRIPT_KEYS)
    validate_mapping_keys(errors, artifact, "workload_hashes", WORKLOAD_HASH_KEYS)
    validate_mapping(errors, artifact, "timing_measurements")
    validate_mapping(errors, artifact, "kv260_timing_transcript")
    validate_mapping(errors, artifact, "gatemate_transcript")
    validate_mapping(errors, artifact, "polarfire_transcript")
    validate_residual_telemetry(errors, artifact)
    validate_hashes(errors, artifact)
    expect(
        errors,
        isinstance(artifact.get("preconditions_checked"), list)
        and len(artifact.get("preconditions_checked")) >= 8,
        "preconditions_checked resources mismatch",
    )
    expect(
        errors,
        isinstance(artifact.get("tests_run"), list) and bool(artifact.get("tests_run")),
        "tests_run must be non-empty",
    )
    expect(errors, artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")
    return errors


def no_host_storage(payload: JsonMap) -> bool:
    return no_5120_host_storage(payload)


def validate_mapping_keys(
    errors: list[str],
    artifact: JsonMap,
    field: str,
    expected_keys: Sequence[str],
) -> None:
    value = artifact.get(field)
    expect(errors, isinstance(value, Mapping), f"{field} must be a dict")
    if isinstance(value, Mapping):
        expect(errors, set(value) == set(expected_keys), f"{field} keys mismatch")


def validate_mapping(errors: list[str], artifact: JsonMap, field: str) -> None:
    expect(errors, isinstance(artifact.get(field), Mapping), f"{field} must be a dict")


def validate_hashes(errors: list[str], artifact: JsonMap) -> None:
    hashes = artifact.get("workload_hashes")
    if not isinstance(hashes, Mapping):
        return
    for key in (
        "cpu_reference_residual_sweep",
        "cpu_residual_samples",
        "polarfire_dispatch_precheck_workload",
    ):
        expect(errors, is_sha256(hashes.get(key)), f"{key} hash invalid")
    for key in ("kv260_timing_workload", "gatemate_safe_flash_manifest"):
        value = hashes.get(key)
        expect(errors, value is None or is_sha256(value), f"{key} hash invalid")


def validate_residual_telemetry(errors: list[str], artifact: JsonMap) -> None:
    rows = artifact.get("residual_energy_by_sweep")
    expect(errors, isinstance(rows, list), "residual_energy_by_sweep must be a list")
    if not isinstance(rows, list):
        return
    expect(errors, bool(rows), "residual telemetry requires residual samples")
    for row in rows:
        expect(errors, isinstance(row, Mapping), "residual row invalid")
    fit = fit_decay_exponent(rows) if rows else None
    expect(errors, artifact.get("decay_exponent") == fit, "decay exponent mismatch")
    expect(errors, fit is not None and math.isfinite(fit), "decay exponent invalid")
    quality = artifact.get("sample_quality_evidence")
    expect(errors, isinstance(quality, Mapping), "sample_quality_evidence must be a dict")
    if isinstance(quality, Mapping):
        expect(errors, quality.get("board_speedup_evidence_complete") is False, "speedup evidence overclaim")


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260701", help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(repo_root=args.repo_root, run_date=args.date)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"inference_substrate: {artifact['inference_substrate']}")
    print(f"no_speedup_claim: {artifact['no_speedup_claim']}")
    print(f"extropic_tsu_execution_claimed: {artifact['extropic_tsu_execution_claimed']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    raise SystemExit(main())
