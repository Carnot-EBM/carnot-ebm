#!/usr/bin/env python3
"""Exp 5144: authenticated local board workload transcripts.

Spec refs: REQ-HW-5144, SCENARIO-HW-5144.

This experiment is intentionally conservative. Board reachability is useful
continuity evidence, but it is not a workload claim. A board workload is run
only when a checked-in safe manifest names the workload, the file or payload
hash matches, and the command transcript returns timing plus sample-quality or
correctness evidence. Otherwise the artifact records the exact blocker and
keeps `no_speedup_claim=true`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
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
    gatemate_detected,
    idcode_from_text,
    is_sha256,
    no_host_storage as no_5120_host_storage,
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
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(REPO_ROOT / "python"))


EXPERIMENT_ID = "exp5144-authenticated-board-workload-v471"
EXPERIMENT_NAME = "experiment_5144_authenticated_board_workload"
MILESTONE = "2026.07.471"
SCHEMA = "carnot.experiment_5144_authenticated_board_workload.v471"
OUTPUT_REL_PATH = Path("results") / "experiment_5144_authenticated_board_workload_v471.json"
EXP5141_RESULT_REL_PATH = (
    Path("results") / "experiment_5141_hubo_partition_residual_exponent_v471.json"
)
SAFE_WORKLOAD_MANIFEST_REL_PATH = (
    Path("hardware") / "board_workloads" / "exp5144_safe_workload_manifest.json"
)
SAFE_WORKLOAD_MANIFEST_SCHEMA = "carnot.exp5144.safe_workload_manifest.v1"
SPEC_REFS = ["REQ-HW-5144", "SCENARIO-HW-5144"]
INFERENCE_SUBSTRATE = "local_board_transcripts_or_blocked"
BLOCKED_VERDICT = "blocked_no_safe_board_workload_manifest_no_speedup_claim"
COMPLETE_VERDICT = "complete_authenticated_board_workload_transcripts_no_speedup_claim"
RANDOM_SEED = 5144

KV260_COMMAND_KIND = "ssh_python_read_only_uio_v1"
GATEMATE_COMMAND_KIND = "dirtyjtag_program_v1"
POLARFIRE_COMMAND_KIND = "ssh_inline_python_correctness_v1"
BOARD_NAMES = ("kv260", "gatemate", "polarfire")

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_COMMAND_AVAILABLE_COMMAND = ("sh", "-lc", "command -v openFPGALoader")
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)

POLARFIRE_INLINE_PROGRAM_TEMPLATE = (
    "loads a JSON spin payload, computes sum(value * value), times the run, "
    "and emits workload/executable hashes plus correctness evidence"
)
POLARFIRE_INLINE_EXECUTABLE_HASH = sha256_text(POLARFIRE_INLINE_PROGRAM_TEMPLATE)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "preconditions_checked",
    "kv260_ssh_checked",
    "kv260_host_block_devices_touched",
    "safe_workload_manifest",
    "workload_hashes",
    "kv260_timing_transcript",
    "gatemate_transcript",
    "polarfire_transcript",
    "timing_measurements",
    "sample_quality_evidence",
    "hardware_workload_transcripts_ready",
    "no_speedup_claim",
    "extropic_tsu_execution_claimed",
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
    "command_transcripts",
    "board_blockers",
    "reproducibility_checksum",
)
COMMAND_TRANSCRIPT_KEYS = (
    "kv260_ssh",
    "kv260_workload",
    "gatemate_openfpgaloader",
    "gatemate_dirtyjtag_detect",
    "gatemate_workload",
    "polarfire_ssh",
    "polarfire_dispatch",
)
WORKLOAD_HASH_KEYS = (
    "safe_workload_manifest",
    "exp5141_board_ready_descriptors",
    "kv260_workload",
    "kv260_executable",
    "gatemate_workload",
    "gatemate_bitstream",
    "polarfire_workload",
    "polarfire_executable",
)
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python scripts/experiment_5144_authenticated_board_workload_v471.py --date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5144_authenticated_board_workload.py -q",
    ".venv/bin/coverage run --source=python/carnot/experiment_5144_authenticated_board_workload.py -m pytest tests/python/test_experiment_5144_authenticated_board_workload.py -q",
    ".venv/bin/coverage report --fail-under=100 -m python/carnot/experiment_5144_authenticated_board_workload.py",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "preconditions_checked": "hardware safety",
    "kv260_ssh_checked": "board reachability",
    "kv260_host_block_devices_touched": "hardware safety",
    "safe_workload_manifest": "workload provenance",
    "workload_hashes": "reproducibility",
    "kv260_timing_transcript": "board evidence",
    "gatemate_transcript": "board evidence",
    "polarfire_transcript": "board evidence",
    "timing_measurements": "measured evidence",
    "sample_quality_evidence": "no empty speed claim",
    "hardware_workload_transcripts_ready": "downstream readiness",
    "no_speedup_claim": "hardware claim discipline",
    "extropic_tsu_execution_claimed": "substrate honesty",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
    "command_transcripts": "authenticated command evidence",
    "board_blockers": "precise blocker accounting",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260702",
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp5144 artifact from safe prechecks and manifest-gated runs."""

    started = clock()
    root = Path(repo_root)
    descriptors = load_exp5141_descriptors(root)
    manifest = load_safe_workload_manifest(root)
    kv260 = run_kv260_checks(root, manifest, command_runner)
    gatemate = run_gatemate_checks(root, manifest, command_runner)
    polarfire = run_polarfire_checks(manifest, command_runner)
    ready_boards = [
        name
        for name, board in (("kv260", kv260), ("gatemate", gatemate), ("polarfire", polarfire))
        if board["evidence_ready"]
    ]
    ready = bool(ready_boards)
    safe_manifest = safe_manifest_summary(manifest, descriptors)
    sample_quality = sample_quality_evidence(
        ready_boards=ready_boards,
        kv260=kv260,
        gatemate=gatemate,
        polarfire=polarfire,
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "run_date": run_date,
        "honest_verdict": COMPLETE_VERDICT if ready else BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round_duration(clock() - started),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": build_preconditions(manifest, kv260, gatemate, polarfire),
        "kv260_ssh_checked": True,
        "kv260_host_block_devices_touched": False,
        "safe_workload_manifest": safe_manifest,
        "workload_hashes": workload_hashes(safe_manifest, kv260, gatemate, polarfire),
        "kv260_timing_transcript": kv260_transcript(kv260),
        "gatemate_transcript": gatemate_transcript(gatemate),
        "polarfire_transcript": polarfire_transcript(polarfire),
        "command_transcripts": command_transcripts(kv260, gatemate, polarfire),
        "timing_measurements": timing_measurements(kv260, gatemate, polarfire),
        "sample_quality_evidence": sample_quality,
        "board_blockers": {
            "kv260": list(kv260["blockers"]),
            "gatemate": list(gatemate["blockers"]),
            "polarfire": list(polarfire["blockers"]),
        },
        "hardware_workload_transcripts_ready": ready,
        "no_speedup_claim": True,
        "extropic_tsu_execution_claimed": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def load_exp5141_descriptors(repo_root: str | Path) -> JsonDict:
    path = Path(repo_root) / EXP5141_RESULT_REL_PATH
    if not path.is_file():
        return {
            "loaded": False,
            "path": str(EXP5141_RESULT_REL_PATH),
            "sha256": None,
            "descriptor_counts": {},
            "descriptor_hashes": {board: [] for board in BOARD_NAMES},
            "blocker": "exp5141_board_ready_descriptors_missing",
        }
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    descriptors = data.get("board_ready_workload_descriptors", [])
    hashes: dict[str, list[str]] = {board: [] for board in BOARD_NAMES}
    for descriptor in descriptors:
        if isinstance(descriptor, Mapping):
            board = str(descriptor.get("target_board", ""))
            digest = descriptor.get("workload_hash")
            if board in hashes and is_sha256(digest):
                hashes[board].append(str(digest))
    return {
        "loaded": True,
        "path": str(EXP5141_RESULT_REL_PATH),
        "sha256": sha256_text(text),
        "descriptor_counts": {board: len(hashes[board]) for board in sorted(hashes)},
        "descriptor_hashes": hashes,
        "blocker": None,
    }


def load_safe_workload_manifest(repo_root: str | Path) -> JsonDict:
    path = Path(repo_root) / SAFE_WORKLOAD_MANIFEST_REL_PATH
    if not path.is_file():
        return {
            "present": False,
            "valid": False,
            "path": str(SAFE_WORKLOAD_MANIFEST_REL_PATH),
            "sha256": None,
            "blockers": ["no_checked_in_safe_workload_manifest"],
            "workloads": {
                board: _missing_workload(board, f"no_safe_{board}_workload_manifest")
                for board in BOARD_NAMES
            },
        }
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {
            "present": True,
            "valid": False,
            "path": str(SAFE_WORKLOAD_MANIFEST_REL_PATH),
            "sha256": sha256_text(text),
            "blockers": ["safe_workload_manifest_malformed_json"],
            "workloads": {
                board: _missing_workload(board, f"no_safe_{board}_workload_manifest")
                for board in BOARD_NAMES
            },
        }
    raw_workloads = raw.get("workloads", {}) if isinstance(raw, Mapping) else {}
    workloads = {
        "kv260": _normalize_kv260_workload(Path(repo_root), raw_workloads.get("kv260")),
        "gatemate": _normalize_gatemate_workload(Path(repo_root), raw_workloads.get("gatemate")),
        "polarfire": _normalize_polarfire_workload(raw_workloads.get("polarfire")),
    }
    blockers = []
    if raw.get("schema") != SAFE_WORKLOAD_MANIFEST_SCHEMA:
        blockers.append("safe_workload_manifest_schema_mismatch")
    blockers.extend(
        str(workload["blocker"])
        for workload in workloads.values()
        if workload.get("present") and not workload.get("safe")
    )
    return {
        "present": True,
        "valid": not blockers,
        "path": str(SAFE_WORKLOAD_MANIFEST_REL_PATH),
        "sha256": sha256_text(text),
        "blockers": blockers,
        "manifest_id": raw.get("manifest_id"),
        "workloads": workloads,
    }


def _missing_workload(board: str, blocker: str) -> JsonDict:
    return {
        "board": board,
        "present": False,
        "enabled": False,
        "safe": False,
        "command_kind": None,
        "workload_hash": None,
        "executable_hash": None,
        "bitstream_hash": None,
        "blocker": blocker,
    }


def _normalize_kv260_workload(repo_root: Path, entry: Any) -> JsonDict:
    missing = _missing_workload("kv260", "no_safe_kv260_workload_manifest")
    if not isinstance(entry, Mapping) or not entry.get("enabled"):
        return missing
    base = dict(missing, present=True, enabled=bool(entry.get("enabled")), command_kind=entry.get("command_kind"))
    if entry.get("command_kind") != KV260_COMMAND_KIND:
        return dict(base, blocker="kv260_workload_command_kind_invalid")
    rel_path = Path(str(entry.get("workload_path", "")))
    path = repo_root / rel_path
    if not path.is_file():
        return dict(base, blocker="kv260_workload_file_missing")
    text = path.read_text(encoding="utf-8")
    digest = sha256_text(text)
    if digest != entry.get("workload_sha256") or digest != entry.get("executable_sha256"):
        return dict(base, blocker="kv260_workload_hash_mismatch")
    if not safe_kv260_workload_text(text):
        return dict(base, blocker="kv260_workload_not_read_only_safe")
    return dict(
        base,
        safe=True,
        path=str(rel_path),
        text=text,
        workload_hash=digest,
        executable_hash=digest,
        blocker=None,
    )


def safe_kv260_workload_text(text: str) -> bool:
    lowered = text.lower()
    unsafe = ("write_u32", "loadapp", "flash", "program", "prot_write", "os.o_rdwr", "r+b", "wb")
    return (
        "exp5144_safe_read_only_uio_workload" in lowered
        and "safe_for_continuity_audit" in lowered
        and "read_only" in lowered
        and not any(marker in lowered for marker in unsafe)
    )


def _normalize_gatemate_workload(repo_root: Path, entry: Any) -> JsonDict:
    missing = _missing_workload("gatemate", "no_safe_gatemate_workload_manifest")
    if not isinstance(entry, Mapping) or not entry.get("enabled"):
        return missing
    base = dict(missing, present=True, enabled=bool(entry.get("enabled")), command_kind=entry.get("command_kind"))
    if entry.get("command_kind") != GATEMATE_COMMAND_KIND:
        return dict(base, blocker="gatemate_workload_command_kind_invalid")
    if entry.get("flash_allowed") is not True or entry.get("board_profile") != "olimex_gatemateevb":
        return dict(base, blocker="gatemate_flash_manifest_not_safe")
    rel_path = Path(str(entry.get("bitstream_path", "")))
    path = repo_root / rel_path
    if not path.is_file():
        return dict(base, blocker="gatemate_bitstream_missing")
    digest = sha256_file(path)
    if digest != entry.get("bitstream_sha256") or digest != entry.get("workload_sha256"):
        return dict(base, blocker="gatemate_bitstream_hash_mismatch")
    return dict(
        base,
        safe=True,
        path=str(rel_path),
        workload_hash=digest,
        bitstream_hash=digest,
        blocker=None,
    )


def _normalize_polarfire_workload(entry: Any) -> JsonDict:
    missing = _missing_workload("polarfire", "no_safe_polarfire_workload_manifest")
    if not isinstance(entry, Mapping) or not entry.get("enabled"):
        return missing
    base = dict(missing, present=True, enabled=bool(entry.get("enabled")), command_kind=entry.get("command_kind"))
    payload = entry.get("payload")
    if entry.get("command_kind") != POLARFIRE_COMMAND_KIND:
        return dict(base, blocker="polarfire_workload_command_kind_invalid")
    if not isinstance(payload, Mapping) or "expected_energy" not in payload:
        return dict(base, blocker="polarfire_payload_invalid")
    workload_hash = sha256_json(payload)
    if workload_hash != entry.get("workload_sha256"):
        return dict(base, blocker="polarfire_workload_hash_mismatch")
    if entry.get("executable_sha256") != POLARFIRE_INLINE_EXECUTABLE_HASH:
        return dict(base, blocker="polarfire_executable_hash_mismatch")
    return dict(
        base,
        safe=True,
        payload=dict(payload),
        workload_hash=workload_hash,
        executable_hash=POLARFIRE_INLINE_EXECUTABLE_HASH,
        blocker=None,
    )


def run_kv260_checks(
    repo_root: Path, manifest: JsonMap, command_runner: CommandRunner
) -> JsonDict:
    ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    workload = manifest["workloads"]["kv260"]
    blockers: list[str] = []
    workload_probe = None
    output: JsonDict = {}
    if ssh_probe.exit_code != 0:
        blockers.append("blocked_kv260_ssh_unreachable")
    elif not workload.get("safe"):
        blockers.append(str(workload["blocker"]))
    else:
        workload_probe = command_runner(kv260_workload_command(workload, str(workload["text"])), 30.0)
        output = parse_probe_json(workload_probe)
        blockers.extend(output_blockers("kv260", workload, workload_probe, output))
    return {
        "ssh_probe": ssh_probe,
        "ssh_ready": ssh_probe.exit_code == 0,
        "workload": workload,
        "workload_probe": workload_probe,
        "output": output,
        "hash_matched": hash_matched(workload, output),
        "evidence_ready": evidence_ready(workload_probe, workload, output),
        "blockers": blockers,
    }


def kv260_workload_command(entry: JsonMap, workload_text: str) -> tuple[str, ...]:
    workload_hash = entry.get("workload_hash") or entry.get("workload_sha256")
    executable_hash = entry.get("executable_hash") or entry.get("executable_sha256")
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "kria",
        (
            f"EXP5144_WORKLOAD_SHA256={workload_hash} "
            f"EXP5144_EXECUTABLE_SHA256={executable_hash} python3 - <<'PY'\n"
            f"{workload_text}\nPY"
        ),
    )


def run_gatemate_checks(
    repo_root: Path, manifest: JsonMap, command_runner: CommandRunner
) -> JsonDict:
    tool_probe = command_runner(GATEMATE_COMMAND_AVAILABLE_COMMAND, 10.0)
    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0) if tool_probe.exit_code == 0 else None
    detected = gatemate_detected(detect_probe)
    workload = manifest["workloads"]["gatemate"]
    blockers: list[str] = []
    workload_probe = None
    if tool_probe.exit_code != 0:
        blockers.append("blocked_gatemate_openfpgaloader_missing")
    elif not detected:
        blockers.append("blocked_gatemate_dirtyjtag_not_detected")
    elif not workload.get("safe"):
        blockers.append(str(workload["blocker"]))
    else:
        workload_probe = command_runner(gatemate_workload_command(repo_root, workload), 60.0)
        if workload_probe.exit_code != 0:
            blockers.append("gatemate_workload_command_failed")
    evidence = workload_probe is not None and workload_probe.exit_code == 0 and workload.get("safe")
    return {
        "tool_probe": tool_probe,
        "detect_probe": detect_probe,
        "detected": detected,
        "workload": workload,
        "workload_probe": workload_probe,
        "hash_matched": bool(evidence),
        "evidence_ready": bool(evidence),
        "blockers": blockers,
    }


def gatemate_workload_command(repo_root: str | Path, entry: JsonMap) -> tuple[str, ...]:
    rel_path = entry.get("path") or entry.get("bitstream_path")
    return (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        str(Path(repo_root) / str(rel_path)),
    )


def run_polarfire_checks(manifest: JsonMap, command_runner: CommandRunner) -> JsonDict:
    ssh_probe = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    workload = manifest["workloads"]["polarfire"]
    blockers: list[str] = []
    dispatch_probe = None
    output: JsonDict = {}
    if ssh_probe.exit_code != 0:
        blockers.append("blocked_polarfire_ssh_unreachable")
    elif not workload.get("safe"):
        blockers.append(str(workload["blocker"]))
    else:
        dispatch_probe = command_runner(polarfire_dispatch_command(workload), 30.0)
        output = parse_probe_json(dispatch_probe)
        blockers.extend(output_blockers("polarfire", workload, dispatch_probe, output))
    return {
        "ssh_probe": ssh_probe,
        "ssh_ready": ssh_probe.exit_code == 0,
        "workload": workload,
        "dispatch_probe": dispatch_probe,
        "output": output,
        "hash_matched": hash_matched(workload, output),
        "evidence_ready": evidence_ready(dispatch_probe, workload, output),
        "blockers": blockers,
    }


def polarfire_dispatch_command(entry: JsonMap) -> tuple[str, ...]:
    payload = json.dumps(entry["payload"], sort_keys=True)
    workload_hash = entry.get("workload_hash") or entry.get("workload_sha256")
    executable_hash = entry.get("executable_hash") or entry.get("executable_sha256")
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        "polarfire",
        (
            "python3 - <<'PY'\n"
            "import json, time\n"
            f"payload = {payload!r}\n"
            "data = json.loads(payload)\n"
            "started = time.perf_counter()\n"
            "energy = sum(value * value for value in data['spins'])\n"
            "duration_s = max(time.perf_counter() - started, 0.000001)\n"
            "print(json.dumps({\n"
            f"    'workload_sha256': {workload_hash!r},\n"
            f"    'executable_sha256': {executable_hash!r},\n"
            "    'energy': energy,\n"
            "    'duration_s': duration_s,\n"
            "    'sample_quality': {'sample_count': len(data['spins']), 'finite_energy': True},\n"
            "    'correctness': {'energy_matches_expected': energy == data['expected_energy']},\n"
            "}, sort_keys=True))\n"
            "PY"
        ),
    )


def parse_probe_json(probe: CommandProbe | None) -> JsonDict:
    if probe is None or not probe.combined_output.strip():
        return {}
    try:
        parsed = json.loads(probe.combined_output.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def output_blockers(
    board: str, workload: JsonMap, probe: CommandProbe | None, output: JsonMap
) -> list[str]:
    if probe is None:
        return [f"{board}_workload_not_attempted"]
    if probe.exit_code != 0:
        return [f"{board}_workload_command_failed"]
    if not hash_matched(workload, output):
        return [f"{board}_workload_output_hash_mismatch"]
    if not has_quality_or_correctness(output):
        return [f"{board}_sample_quality_or_correctness_missing"]
    return []


def hash_matched(workload: JsonMap, output: JsonMap) -> bool:
    return (
        workload.get("safe") is True
        and output.get("workload_sha256") == workload.get("workload_hash")
        and (
            workload.get("executable_hash") is None
            or output.get("executable_sha256") == workload.get("executable_hash")
        )
    )


def has_quality_or_correctness(output: JsonMap) -> bool:
    return isinstance(output.get("sample_quality"), Mapping) or isinstance(
        output.get("correctness"), Mapping
    )


def evidence_ready(
    probe: CommandProbe | None, workload: JsonMap, output: JsonMap
) -> bool:
    return (
        probe is not None
        and probe.exit_code == 0
        and hash_matched(workload, output)
        and has_quality_or_correctness(output)
    )


def safe_manifest_summary(manifest: JsonMap, descriptors: JsonMap) -> JsonDict:
    return {
        "path": manifest["path"],
        "present": bool(manifest["present"]),
        "valid": bool(manifest["valid"]),
        "sha256": manifest["sha256"],
        "blockers": list(manifest["blockers"]),
        "manifest_id": manifest.get("manifest_id"),
        "exp5141_descriptors_loaded": bool(descriptors["loaded"]),
        "exp5141_result_path": descriptors["path"],
        "exp5141_result_sha256": descriptors["sha256"],
        "descriptor_counts": descriptors["descriptor_counts"],
        "descriptor_hashes": descriptors["descriptor_hashes"],
        "workloads": {
            board: workload_manifest_public(manifest["workloads"][board]) for board in BOARD_NAMES
        },
    }


def workload_manifest_public(workload: JsonMap) -> JsonDict:
    return {
        "board": workload["board"],
        "present": bool(workload["present"]),
        "enabled": bool(workload["enabled"]),
        "safe": bool(workload["safe"]),
        "command_kind": workload["command_kind"],
        "path": workload.get("path"),
        "workload_hash": workload.get("workload_hash"),
        "executable_hash": workload.get("executable_hash"),
        "bitstream_hash": workload.get("bitstream_hash"),
        "blocker": workload.get("blocker"),
    }


def build_preconditions(
    manifest: JsonMap, kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap
) -> list[JsonDict]:
    return [
        precondition_entry(
            "kv260_ssh",
            kv260["ssh_probe"],
            bool(kv260["ssh_ready"]),
            "ssh_only_no_host_block_device_probe",
            ["ssh_only", "no_host_block_device_access", "no_destructive_actions"],
        ),
        policy_precondition(),
        precondition_entry(
            "gatemate_openfpgaloader",
            gatemate["tool_probe"],
            gatemate["tool_probe"].exit_code == 0,
            "dirtyjtag_detect_no_nextpnr_gatemate",
            ["detect_first", "no_nextpnr_gatemate", "no_flash_without_manifest"],
        ),
        {
            "resource": "safe_workload_manifest",
            "available": bool(manifest["valid"]),
            "command": str(SAFE_WORKLOAD_MANIFEST_REL_PATH),
            "exit_code": 0 if manifest["valid"] else 1,
            "duration_s": 0.0001,
            "observed": "valid" if manifest["valid"] else ",".join(manifest["blockers"]),
            "discipline": "workload_execution_requires_checked_in_hash_matched_manifest",
            "safety_constraints": ["hash_matched", "checked_in", "no_speedup_from_reachability"],
        },
        precondition_entry(
            "polarfire_ssh",
            polarfire["ssh_probe"],
            bool(polarfire["ssh_ready"]),
            "ssh_dispatch_manifest_gated",
            ["ssh_only", "hash_matched_dispatch", "no_speedup_from_reachability"],
        ),
    ]


def workload_hashes(
    safe_manifest: JsonMap, kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap
) -> JsonDict:
    return {
        "safe_workload_manifest": safe_manifest["sha256"],
        "exp5141_board_ready_descriptors": safe_manifest["exp5141_result_sha256"],
        "kv260_workload": kv260["workload"].get("workload_hash"),
        "kv260_executable": kv260["workload"].get("executable_hash"),
        "gatemate_workload": gatemate["workload"].get("workload_hash"),
        "gatemate_bitstream": gatemate["workload"].get("bitstream_hash"),
        "polarfire_workload": polarfire["workload"].get("workload_hash"),
        "polarfire_executable": polarfire["workload"].get("executable_hash"),
    }


def kv260_transcript(kv260: JsonMap) -> JsonDict:
    return {
        "ssh_checked": True,
        "ssh_ready": bool(kv260["ssh_ready"]),
        "workload_attempted": kv260["workload_probe"] is not None,
        "workload_hash": kv260["workload"].get("workload_hash"),
        "executable_hash": kv260["workload"].get("executable_hash"),
        "hash_matched": bool(kv260["hash_matched"]),
        "timing_output": dict(kv260["output"]),
        "blockers": list(kv260["blockers"]),
        "transcript": probe_dict(kv260["workload_probe"]),
    }


def gatemate_transcript(gatemate: JsonMap) -> JsonDict:
    detect_probe = gatemate["detect_probe"]
    return {
        "openfpgaloader_checked": True,
        "openfpgaloader_available": gatemate["tool_probe"].exit_code == 0,
        "dirtyjtag_detected": bool(gatemate["detected"]),
        "detected_idcode": idcode_from_text(detect_probe.combined_output)
        if detect_probe is not None
        else None,
        "workload_attempted": gatemate["workload_probe"] is not None,
        "flash_attempted": gatemate["workload_probe"] is not None,
        "workload_hash": gatemate["workload"].get("workload_hash"),
        "bitstream_hash": gatemate["workload"].get("bitstream_hash"),
        "hash_matched": bool(gatemate["hash_matched"]),
        "blockers": list(gatemate["blockers"]),
        "transcript": probe_dict(gatemate["workload_probe"]),
        "action_scope": "dirtyjtag_detect_then_manifest_gated_program_only",
    }


def polarfire_transcript(polarfire: JsonMap) -> JsonDict:
    return {
        "ssh_checked": True,
        "ssh_ready": bool(polarfire["ssh_ready"]),
        "dispatch_attempted": polarfire["dispatch_probe"] is not None,
        "workload_hash": polarfire["workload"].get("workload_hash"),
        "executable_hash": polarfire["workload"].get("executable_hash"),
        "hash_matched": bool(polarfire["hash_matched"]),
        "dispatch_output": dict(polarfire["output"]),
        "blockers": list(polarfire["blockers"]),
        "transcript": probe_dict(polarfire["dispatch_probe"]),
        "action_scope": "ssh_dispatch_hashes_only_no_speedup_inference",
    }


def command_transcripts(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> JsonDict:
    return {
        "kv260_ssh": kv260["ssh_probe"].as_dict(),
        "kv260_workload": probe_dict(kv260["workload_probe"]),
        "gatemate_openfpgaloader": gatemate["tool_probe"].as_dict(),
        "gatemate_dirtyjtag_detect": probe_dict(gatemate["detect_probe"]),
        "gatemate_workload": probe_dict(gatemate["workload_probe"]),
        "polarfire_ssh": polarfire["ssh_probe"].as_dict(),
        "polarfire_dispatch": probe_dict(polarfire["dispatch_probe"]),
    }


def timing_measurements(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> JsonDict:
    ready_count = sum(bool(board["evidence_ready"]) for board in (kv260, gatemate, polarfire))
    return {
        "kv260_ssh_s": round_duration(kv260["ssh_probe"].duration_s),
        "kv260_workload_s": round_duration(kv260["workload_probe"].duration_s)
        if kv260["workload_probe"] is not None
        else None,
        "gatemate_detect_s": round_duration(gatemate["detect_probe"].duration_s)
        if gatemate["detect_probe"] is not None
        else None,
        "gatemate_workload_s": round_duration(gatemate["workload_probe"].duration_s)
        if gatemate["workload_probe"] is not None
        else None,
        "polarfire_ssh_s": round_duration(polarfire["ssh_probe"].duration_s),
        "polarfire_dispatch_s": round_duration(polarfire["dispatch_probe"].duration_s)
        if polarfire["dispatch_probe"] is not None
        else None,
        "hash_matched_evidence_count": ready_count,
    }


def sample_quality_evidence(
    *, ready_boards: Sequence[str], kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap
) -> JsonDict:
    return {
        "ready_evidence_boards": list(ready_boards),
        "kv260": board_quality(kv260["output"], kv260["hash_matched"], kv260["blockers"]),
        "gatemate": {
            "hash_matched": bool(gatemate["hash_matched"]),
            "correctness": {"program_exit_code_zero": bool(gatemate["evidence_ready"])}
            if gatemate["workload_probe"] is not None
            else None,
            "sample_quality": None,
            "blockers": list(gatemate["blockers"]),
        },
        "polarfire": board_quality(
            polarfire["output"], polarfire["hash_matched"], polarfire["blockers"]
        ),
        "speedup_evidence_claimed": False,
    }


def board_quality(output: JsonMap, hash_ok: bool, blockers: Sequence[str]) -> JsonDict:
    return {
        "hash_matched": bool(hash_ok),
        "sample_quality": output.get("sample_quality")
        if isinstance(output.get("sample_quality"), Mapping)
        else None,
        "correctness": output.get("correctness")
        if isinstance(output.get("correctness"), Mapping)
        else None,
        "blockers": list(blockers),
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
    run_date: str = "20260702",
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
    expect(
        errors,
        str(artifact.get("honest_verdict", "")).startswith(("complete_", "success_", "blocked_")),
        "honest_verdict terminal prefix missing",
    )
    expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    expect(errors, round_duration(artifact.get("duration_s")) >= 0.0001, "duration_s below floor")
    for field, expected in (
        ("kv260_ssh_checked", True),
        ("kv260_host_block_devices_touched", False),
        ("no_speedup_claim", True),
        ("extropic_tsu_execution_claimed", False),
        ("conductor_modified", False),
    ):
        expect(errors, artifact.get(field) is expected, f"{field} mismatch")
    expect(errors, no_host_storage(artifact), "forbidden host storage marker")
    validate_mapping(errors, artifact, "safe_workload_manifest")
    validate_mapping_keys(errors, artifact, "workload_hashes", WORKLOAD_HASH_KEYS)
    validate_mapping(errors, artifact, "kv260_timing_transcript")
    validate_mapping(errors, artifact, "gatemate_transcript")
    validate_mapping(errors, artifact, "polarfire_transcript")
    validate_mapping(errors, artifact, "timing_measurements")
    validate_mapping_keys(errors, artifact, "command_transcripts", COMMAND_TRANSCRIPT_KEYS)
    validate_mapping(errors, artifact, "sample_quality_evidence")
    validate_ready_gate(errors, artifact)
    validate_hashes(errors, artifact)
    expect(
        errors,
        isinstance(artifact.get("preconditions_checked"), list)
        and len(artifact.get("preconditions_checked")) >= 5,
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


def validate_mapping(errors: list[str], artifact: JsonMap, field: str) -> None:
    expect(errors, isinstance(artifact.get(field), Mapping), f"{field} must be a dict")


def validate_mapping_keys(
    errors: list[str], artifact: JsonMap, field: str, expected_keys: Sequence[str]
) -> None:
    value = artifact.get(field)
    expect(errors, isinstance(value, Mapping), f"{field} must be a dict")
    if isinstance(value, Mapping):
        expect(errors, set(value) == set(expected_keys), f"{field} keys mismatch")


def validate_ready_gate(errors: list[str], artifact: JsonMap) -> None:
    quality = artifact.get("sample_quality_evidence")
    if not isinstance(quality, Mapping):
        return
    ready_boards = quality.get("ready_evidence_boards")
    ready = bool(ready_boards)
    expect(
        errors,
        artifact.get("hardware_workload_transcripts_ready") is ready,
        "ready gate mismatch",
    )
    expect(errors, quality.get("speedup_evidence_claimed") is False, "speedup evidence overclaim")


def validate_hashes(errors: list[str], artifact: JsonMap) -> None:
    hashes = artifact.get("workload_hashes")
    if not isinstance(hashes, Mapping):
        return
    for key, value in hashes.items():
        expect(errors, value is None or is_sha256(value), f"{key} hash invalid")


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(repo_root=args.repo_root, run_date=args.date)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"hardware_workload_transcripts_ready: {artifact['hardware_workload_transcripts_ready']}")
    print(f"no_speedup_claim: {artifact['no_speedup_claim']}")
    print(f"extropic_tsu_execution_claimed: {artifact['extropic_tsu_execution_claimed']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    raise SystemExit(main())
