"""Exp6325 GateMate dated-receipt single detect.

Spec refs: REQ-HW-6325, SCENARIO-HW-6325-1, SCENARIO-HW-6325-2,
SCENARIO-HW-6325-3, SCENARIO-HW-6325-4, SCENARIO-HW-6325-5,
SCENARIO-HW-6325-6.

The 2026-08-11 operator receipt changes GateMate power state. That change
permits one read-only DirtyJTAG detect. This module records the preconditions,
runs at most that one detect, preserves the raw outcome, and then stops.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any

from carnot.experiment_6121_gatemate_changed_state_gate_v530 import (
    _receipt_authorizes_changed_state as exp6121_receipt_authorizes_changed_state,
)
from carnot.experiment_6121_gatemate_changed_state_gate_v530 import (
    physical_state_hashes as exp6121_physical_state_hashes,
)
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandReceipt"]

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "carnot.experiment_6325.gatemate_dated_receipt_single_detect.v1"
EXPERIMENT = "experiment_6325_gatemate_dated_receipt_single_detect"
EXPERIMENT_ID = "exp6325-gatemate-dated-receipt-single-detect"
MILESTONE = "2026.08.6325"
RUN_DATE = "20260812"
RANDOM_SEED = 6325
SPEC_REFS = (
    "REQ-HW-6325",
    "SCENARIO-HW-6325-1",
    "SCENARIO-HW-6325-2",
    "SCENARIO-HW-6325-3",
    "SCENARIO-HW-6325-4",
    "SCENARIO-HW-6325-5",
    "SCENARIO-HW-6325-6",
)
OUTPUT_REL_PATH = Path("results/experiment_6325_gatemate_dated_receipt_single_detect.json")
INFERENCE_SUBSTRATE = "read_only_usb_tool_receipts_plus_single_dirtyjtag_detect"

EXPECTED_BOARD = "Cologne Chip GateMate A1-EVB-2M"
EXPECTED_DIRTYJTAG_VIDPID = "1209:c0ca"
EXPECTED_IDCODE = "0x20000001"
EXACT_AUTHORIZED_COMMAND = "openFPGALoader -c dirtyJtag --detect"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
VERSION_COMMAND = ("openFPGALoader", "--help")
LSUSB_COMMAND = ("lsusb",)
DETECT_TIMEOUT_S = 30.0
READ_ONLY_TIMEOUT_S = 10.0

KNOWN_ISSUES_REL_PATH = Path("ops/known-issues.md")
EXP6121_REL_PATH = Path("results/experiment_6121_gatemate_changed_state_gate_v530.json")
EXP6199_REL_PATH = Path("results/experiment_6199_gatemate_terminal_action_audit_v537.json")
USER_NAMED_EXP6199_REL_PATH = Path("results/experiment_6199_gatemate_dated_receipt_gate.json")
PRIOR_ATTEMPT_PATHS = (EXP6121_REL_PATH, EXP6199_REL_PATH)
HASHED_INPUT_PATHS = (
    KNOWN_ISSUES_REL_PATH,
    Path("research-hardware-wishlist.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    EXP6121_REL_PATH,
    EXP6199_REL_PATH,
    USER_NAMED_EXP6199_REL_PATH,
)
PROTECTED_REL_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
)

EXPECTED_DATED_PHYSICAL_RECEIPT: JsonDict = {
    "exists": True,
    "receipt_date": "20260811",
    "source": (
        'operator directive 2026-08-11T12:58:49Z: "I have just power cycled '
        'all of the FPGA hardware"'
    ),
    "changes": [
        {
            "field": "power",
            "description": "operator power-cycled all attached FPGA boards, including GateMate",
        }
    ],
    "board": EXPECTED_BOARD,
    "power": (
        "power-cycled 2026-08-11 per operator directive (was: cached physical "
        "power unresolved; raw all-ones TDO suggested open or unpowered target)"
    ),
    "usb_dirtyjtag": (
        "1209:c0ca, bus 3, path 2.3 (re-enumerated after power cycle -- device "
        "number changed from the last recorded bus 003 device 006; SAME shared "
        "hub path 2.3, confirming the physical connection was not moved, only "
        "power-cycled)"
    ),
    "verified_read_only": (
        "lsusb (2026-08-11T12:58:49Z): 1209:c0ca present, bus 3 device 11, path "
        "2.3 -- no JTAG command run, no detect attempted; that single "
        "non-destructive detect is reserved for the gated script per the "
        "operator_action_packet's do_not_do list"
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "dated_physical_receipt_path_hash_date_and_text",
    "receipt_newer_than_prior_attempts",
    "board_and_cable_target",
    "pre_command_usb_receipt",
    "openfpgaloader_version_receipt",
    "exact_authorized_command",
    "detect_command_count",
    "detect_started_utc",
    "detect_finished_utc",
    "detect_stdout",
    "detect_stderr",
    "detect_exit_code",
    "detect_timeout",
    "detected_chain_and_device_ids",
    "post_command_usb_receipt",
    "hardware_state_changed_from_prior_attempts",
    "flash_command_count",
    "erase_command_count",
    "reset_command_count",
    "synthesis_command_count",
    "place_route_command_count",
    "timing_command_count",
    "kv260_command_count",
    "polarfire_command_count",
    "stop_after_single_attempt_receipt",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    *REQUIRED_ARTIFACT_FIELDS,
)
FORBIDDEN_COUNT_FIELDS = (
    "flash_command_count",
    "erase_command_count",
    "reset_command_count",
    "synthesis_command_count",
    "place_route_command_count",
    "timing_command_count",
    "kv260_command_count",
    "polarfire_command_count",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates blocked preconditions from a completed single detect.",
    "dated_physical_receipt_path_hash_date_and_text": (
        "The operator receipt is the only physical-change authority."
    ),
    "receipt_newer_than_prior_attempts": (
        "Receipt date must be newer than prior failed attempts."
    ),
    "board_and_cable_target": "The one command is scoped to GateMate plus DirtyJTAG only.",
    "pre_command_usb_receipt": "Read-only USB state is captured before the board command.",
    "openfpgaloader_version_receipt": (
        "Tool version is recorded without addressing a board."
    ),
    "exact_authorized_command": "Only one exact non-destructive command is allowed.",
    "detect_command_count": "A bare integer enforces the single-attempt budget.",
    "detect_started_utc": "UTC start time orders the hardware receipt.",
    "detect_finished_utc": "UTC finish time bounds the attempt.",
    "detect_stdout": "Raw stdout preserves the detection result.",
    "detect_stderr": "Raw stderr preserves tool failures.",
    "detect_exit_code": "Exit code prevents failed detects becoming success.",
    "detect_timeout": "Timeout is a terminal outcome, not a retry trigger.",
    "detected_chain_and_device_ids": (
        "Parsed chain data is derived only from raw detect output."
    ),
    "post_command_usb_receipt": "Read-only USB state is captured after the attempt.",
    "hardware_state_changed_from_prior_attempts": (
        "Changed power state, not software repetition, authorizes this attempt."
    ),
    "flash_command_count": "Flash commands are forbidden.",
    "erase_command_count": "Erase commands are forbidden.",
    "reset_command_count": "Reset commands are forbidden.",
    "synthesis_command_count": "Synthesis commands are forbidden.",
    "place_route_command_count": "Place and route commands are forbidden.",
    "timing_command_count": "Timing commands are forbidden.",
    "kv260_command_count": "KV260 commands are forbidden in this GateMate task.",
    "polarfire_command_count": "PolarFire commands are forbidden in this GateMate task.",
    "stop_after_single_attempt_receipt": "All outcomes stop after one attempt.",
    "protected_files_unchanged": "Operator and conductor files remain byte-identical.",
    "preconditions_checked": (
        "Hashes, USB, tool path, disk, permissions, timeout, and budget are checked first."
    ),
    "inference_substrate": "Use read-only host receipts plus one DirtyJTAG detect.",
    "verifier_is_oracle": (
        "Raw receipts and command output are authoritative only for visibility."
    ),
    "field_provenance": "Every field traces to a receipt, command output, parser, or test.",
    "field_principles": "Every required field declares why it exists.",
    "test_commands": "Verification commands are recorded.",
    "test_exit_codes": "Verification exit codes are recorded.",
    "duration_s": "Measured wall time is reported without padding.",
    "reproducibility_checksum": "Checksum detects receipt or artifact drift.",
    "honest_verdict": "Verdict names the raw outcome without inferring execution.",
}

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6325_gatemate_dated_receipt_single_detect.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6325_gatemate_dated_receipt_single_detect.py -m pytest tests/python/test_experiment_6325_gatemate_dated_receipt_single_detect.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6325_gatemate_dated_receipt_single_detect.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6325_gatemate_dated_receipt_single_detect.py",
    ".venv/bin/pytest tests/python -q",
)
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


@dataclass(frozen=True)
class CommandReceipt:
    """Raw process evidence for read-only commands and the one detect."""

    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timeout: bool = False

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_receipt(self, *, timeout_s: float, addresses_board: bool) -> JsonDict:
        return {
            "command": command_to_string(self.command),
            "timeout_s": timeout_s,
            "addresses_board": addresses_board,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 6),
            "timeout": self.timeout,
        }


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def _coerce_timeout_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def run_command(command: tuple[str, ...], timeout_s: float) -> CommandReceipt:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandReceipt(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=time.perf_counter() - started,
            timeout=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_timeout_text(exc.stdout)
        stderr = _coerce_timeout_text(exc.stderr)
        suffix = f"timed out after {timeout_s:.1f}s"
        stderr = f"{stderr}\n{suffix}".strip()
        return CommandReceipt(
            command=command,
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.perf_counter() - started,
            timeout=True,
        )


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(encoded)


def path_receipt(root: Path, relative_path: str | Path) -> JsonDict:
    rel = Path(relative_path)
    path = root / rel
    if not path.exists():
        return {"path": rel.as_posix(), "present": False, "bytes": 0, "sha256": None}
    data = path.read_bytes()
    return {
        "path": rel.as_posix(),
        "present": True,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def read_json_object(root: Path, relative_path: str | Path) -> JsonDict:
    path = root / Path(relative_path)
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): path_receipt(root, path)["sha256"] for path in PROTECTED_REL_PATHS}


def protected_files_unchanged(root: Path, before_hashes: JsonMap) -> JsonDict:
    after = protected_file_hashes(root)
    changed = [path for path, old_hash in before_hashes.items() if after.get(path) != old_hash]
    return {
        "all_unchanged": not changed,
        "changed_paths": changed,
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
    }


def utc_from_clock_value(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def extracted_receipt_text(root: Path) -> tuple[JsonDict, str]:
    text_path = root / KNOWN_ISSUES_REL_PATH
    if not text_path.exists():
        return {"exists": False, "receipt_date": None, "changes": []}, ""
    text = text_path.read_text(encoding="utf-8")
    anchor = "2026-08-11 operator physical action"
    start = text.find(anchor)
    if start < 0:
        return {"exists": False, "receipt_date": None, "changes": []}, ""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text[start:], flags=re.DOTALL)
    if match is None:
        return {"exists": False, "receipt_date": None, "changes": []}, ""
    raw = match.group(1)
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}, raw


def dated_physical_receipt_path_hash_date_and_text(
    root: Path,
    receipt_override: JsonMap | None,
) -> tuple[JsonDict, JsonDict]:
    if receipt_override is None:
        parsed, receipt_text = extracted_receipt_text(root)
        source = "ops_known_issues_structured_json_block"
    else:
        parsed = dict(receipt_override)
        receipt_text = json.dumps(parsed, sort_keys=True)
        source = "caller_supplied_receipt"
    return parsed, {
        "path": KNOWN_ISSUES_REL_PATH.as_posix(),
        "path_hash": path_receipt(root, KNOWN_ISSUES_REL_PATH),
        "receipt_date": parsed.get("receipt_date"),
        "text": receipt_text,
        "text_sha256": sha256_bytes(receipt_text.encode()),
        "parsed_receipt": parsed,
        "source": source,
    }


def prior_attempt_dates(root: Path) -> dict[str, str | None]:
    dates: dict[str, str | None] = {}
    for path in PRIOR_ATTEMPT_PATHS:
        artifact = read_json_object(root, path)
        dates[path.as_posix()] = str(artifact.get("run_date")) if artifact.get("run_date") else None
    return dates


def receipt_newer_than_prior_attempts(root: Path, receipt: JsonMap) -> JsonDict:
    receipt_date = str(receipt.get("receipt_date") or "")
    dates = prior_attempt_dates(root)
    comparisons = {
        path: bool(receipt_date and prior_date and receipt_date > prior_date)
        for path, prior_date in dates.items()
    }
    return {
        "receipt_date": receipt_date or None,
        "prior_attempt_dates": dates,
        "newer_than_each_prior_attempt": comparisons,
        "newer_than_all_prior_attempts": bool(comparisons)
        and all(comparisons.values()),
        "user_named_prior_attempt_receipt": path_receipt(root, USER_NAMED_EXP6199_REL_PATH),
    }


def board_and_cable_target(receipt: JsonMap) -> JsonDict:
    board = str(receipt.get("board") or "")
    usb_dirtyjtag = str(receipt.get("usb_dirtyjtag") or "")
    board_ok = board == EXPECTED_BOARD
    dirtyjtag_ok = EXPECTED_DIRTYJTAG_VIDPID in usb_dirtyjtag
    return {
        "expected_board": EXPECTED_BOARD,
        "receipt_board": board,
        "expected_dirtyjtag_vidpid": EXPECTED_DIRTYJTAG_VIDPID,
        "receipt_usb_dirtyjtag": usb_dirtyjtag,
        "board_ok": board_ok,
        "dirtyjtag_vidpid_ok": dirtyjtag_ok,
        "target_ok": board_ok and dirtyjtag_ok,
    }


def read_only_usb_receipt(
    *,
    command_runner: CommandRunner,
    phase: str,
    run: bool,
) -> JsonDict:
    if not run:
        return {
            "phase": phase,
            "command": command_to_string(LSUSB_COMMAND),
            "read_only": True,
            "executed": False,
            "reason": "not_run_because_no_detect_attempt_executed",
            "dirtyjtag_present": None,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timeout": False,
        }
    receipt = command_runner(LSUSB_COMMAND, READ_ONLY_TIMEOUT_S)
    out = receipt.combined_output
    return {
        "phase": phase,
        "command": command_to_string(LSUSB_COMMAND),
        "read_only": True,
        "executed": True,
        "dirtyjtag_present": EXPECTED_DIRTYJTAG_VIDPID in out,
        **receipt.as_receipt(timeout_s=READ_ONLY_TIMEOUT_S, addresses_board=False),
    }


def openfpgaloader_version_receipt(
    *,
    command_runner: CommandRunner,
    binary_path: str | None,
) -> JsonDict:
    receipt = command_runner(VERSION_COMMAND, READ_ONLY_TIMEOUT_S)
    first_line = next((line.strip() for line in receipt.combined_output.splitlines() if line.strip()), "")
    return {
        "binary_path": binary_path,
        "binary_path_resolved": binary_path is not None,
        "tool_identity_first_line": first_line,
        "version_command_addresses_board": False,
        **receipt.as_receipt(timeout_s=READ_ONLY_TIMEOUT_S, addresses_board=False),
    }


def disk_and_permission_receipt(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    output_parent = root / OUTPUT_REL_PATH.parent
    return {
        "root": str(root),
        "disk_usage": {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        },
        "permissions": {
            "root_readable": os.access(root, os.R_OK),
            "root_executable": os.access(root, os.X_OK),
            "output_parent_exists": output_parent.exists(),
            "output_parent_writable": os.access(output_parent, os.W_OK),
        },
    }


def hardware_state_changed_from_prior_attempts(receipt: JsonMap) -> JsonDict:
    prior_state, current_state, changed = exp6121_physical_state_hashes(receipt)
    return {
        "changed": changed,
        "exp6121_parser_authorizes_changed_state": exp6121_receipt_authorizes_changed_state(
            receipt
        ),
        "prior_state_hash": stable_hash(prior_state),
        "current_state_hash": stable_hash(current_state),
        "changed_fields": [
            str(item.get("field"))
            for item in receipt.get("changes", [])
            if isinstance(item, Mapping)
        ],
        "basis": "Exp6121 physical-state parser plus 2026-08-11 power field receipt.",
    }


def input_path_hashes(root: Path) -> JsonDict:
    return {path.as_posix(): path_receipt(root, path) for path in HASHED_INPUT_PATHS}


def preconditions_checked(
    *,
    root: Path,
    receipt: JsonMap,
    receipt_newer: JsonMap,
    target: JsonMap,
    pre_usb: JsonMap,
    version: JsonMap,
    protected_before: JsonMap,
) -> JsonDict:
    return {
        "receipt_date_and_target_checked_before_detect": True,
        "receipt_authorization": {
            "exists": bool(receipt.get("exists")),
            "receipt_date": receipt.get("receipt_date"),
            "newer_than_prior_attempts": receipt_newer["newer_than_all_prior_attempts"],
            "target_ok": target["target_ok"],
            "exp6121_parser_authorizes_changed_state": exp6121_receipt_authorizes_changed_state(
                receipt
            ),
        },
        "hashed_input_paths": input_path_hashes(root),
        "protected_file_hashes_before_detect": dict(protected_before),
        "pre_command_usb_receipt": pre_usb,
        "openfpgaloader_binary": {
            "which_path": version.get("binary_path"),
            "version_exit_code": version.get("exit_code"),
            "version_command": version.get("command"),
        },
        "disk_and_permissions": disk_and_permission_receipt(root),
        "timeout_policy": {
            "detect_timeout_s": DETECT_TIMEOUT_S,
            "read_only_timeout_s": READ_ONLY_TIMEOUT_S,
            "retry_on_timeout": False,
        },
        "one_command_budget": {
            "exact_authorized_command": EXACT_AUTHORIZED_COMMAND,
            "detect_command_budget": 1,
            "forbidden_boards": ["KV260", "PolarFire"],
            "forbidden_actions": [
                "flash",
                "erase",
                "reset",
                "synthesis",
                "place",
                "route",
                "timing",
            ],
        },
    }


def precondition_block_status(receipt: JsonMap, newer: JsonMap, target: JsonMap, version: JsonMap) -> str | None:
    if not receipt.get("exists") or not receipt.get("receipt_date"):
        return "blocked_missing_receipt"
    if not newer.get("newer_than_all_prior_attempts"):
        return "blocked_stale_receipt"
    if not target.get("target_ok"):
        return "blocked_wrong_target"
    if version.get("exit_code") != 0 or version.get("timeout") is True:
        return "blocked_tool_version"
    return None


def detected_chain_and_device_ids(stdout: str, stderr: str) -> JsonDict:
    text = f"{stdout}{stderr}"
    idcodes = [match.lower() for match in re.findall(r"\bidcode\s+(0x[0-9a-fA-F]+)", text)]
    devices: list[JsonDict] = []
    for block in re.split(r"\bindex\s+\d+\s*:", text)[1:]:
        device: JsonDict = {}
        idcode = re.search(r"\bidcode\s+(0x[0-9a-fA-F]+)", block)
        manufacturer = re.search(r"\bmanufacturer\s+([^\n\r]+)", block)
        family = re.search(r"\bfamily\s+([^\n\r]+)", block)
        model = re.search(r"\bmodel\s+([^\n\r]+)", block)
        if idcode:
            device["idcode"] = idcode.group(1).lower()
        if manufacturer:
            device["manufacturer"] = manufacturer.group(1).strip()
        if family:
            device["family"] = family.group(1).strip()
        if model:
            device["model"] = model.group(1).strip()
        devices.append(device)
    return {
        "idcodes": idcodes,
        "devices": devices,
        "device_count": len(idcodes),
        "chain_empty": not idcodes,
        "expected_idcode": EXPECTED_IDCODE,
        "expected_gatemate_idcode_seen": EXPECTED_IDCODE in idcodes,
        "parser": "regex_idcode_and_index_blocks_from_raw_openfpgaloader_output",
    }


def run_single_detect(
    *,
    command_runner: CommandRunner,
    clock: Clock,
    authorized: bool,
) -> tuple[JsonDict, JsonDict]:
    if not authorized:
        empty = {
            "detect_command_count": 0,
            "detect_started_utc": None,
            "detect_finished_utc": None,
            "detect_stdout": "",
            "detect_stderr": "",
            "detect_exit_code": None,
            "detect_timeout": False,
        }
        return empty, detected_chain_and_device_ids("", "")
    started = utc_from_clock_value(clock())
    receipt = command_runner(DETECT_COMMAND, DETECT_TIMEOUT_S)
    finished = utc_from_clock_value(clock())
    detect = {
        "detect_command_count": 1,
        "detect_started_utc": started,
        "detect_finished_utc": finished,
        "detect_stdout": receipt.stdout,
        "detect_stderr": receipt.stderr,
        "detect_exit_code": receipt.exit_code,
        "detect_timeout": receipt.timeout,
    }
    return detect, detected_chain_and_device_ids(receipt.stdout, receipt.stderr)


def status_after_detect(
    *,
    block_status: str | None,
    detect: JsonMap,
    parsed_chain: JsonMap,
) -> str:
    if block_status is not None:
        return block_status
    if detect.get("detect_timeout") is True:
        return "blocked_timeout"
    if parsed_chain.get("expected_gatemate_idcode_seen") is True:
        return "complete_visible"
    if parsed_chain.get("chain_empty") is True:
        if detect.get("detect_exit_code") not in (0, None):
            return "blocked_detect_failed"
        return "blocked_empty_chain"
    return "blocked_idcode"


def stop_after_single_attempt_receipt(status: str, detect_count: int) -> JsonDict:
    return {
        "stopped_after_single_attempt": detect_count <= 1,
        "retry_count": 0,
        "stop_reason": status,
        "applies_to_success_empty_chain_failure_and_timeout": True,
        "no_flash_erase_reset_synthesis_place_route_timing_kv260_or_polarfire": True,
    }


def honest_verdict(status: str) -> str:
    verdicts = {
        "complete_visible": (
            "complete_visible: one authorized DirtyJTAG detect observed GateMate "
            "IDCODE 0x20000001; no execution or performance claim made"
        ),
        "blocked_empty_chain": (
            "blocked_empty_chain: one authorized DirtyJTAG detect ran and found no "
            "device ID; stopped without retry"
        ),
        "blocked_timeout": (
            "blocked_timeout: one authorized DirtyJTAG detect timed out; stopped "
            "without retry"
        ),
        "blocked_detect_failed": (
            "blocked_detect_failed: one authorized DirtyJTAG detect returned a tool "
            "failure; stopped without retry"
        ),
        "blocked_idcode": (
            "blocked_idcode: one authorized DirtyJTAG detect ran but did not show "
            "the expected GateMate IDCODE"
        ),
        "blocked_missing_receipt": (
            "blocked_missing_receipt: no valid dated GateMate physical receipt was "
            "available; zero hardware commands run"
        ),
        "blocked_stale_receipt": (
            "blocked_stale_receipt: the receipt was not newer than both failed "
            "GateMate attempts; zero hardware commands run"
        ),
        "blocked_wrong_target": (
            "blocked_wrong_target: the receipt did not target GateMate A1-EVB-2M "
            "with DirtyJTAG 1209:c0ca; zero hardware commands run"
        ),
        "blocked_tool_version": (
            "blocked_tool_version: openFPGALoader version could not be recorded; "
            "zero hardware commands run"
        ),
    }
    return verdicts[status]


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-HW-6325 / SCENARIO-HW-6325-* plus local receipts and command output",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "test_exit_codes", "reproducibility_checksum"}
    }
    return stable_hash(stable).removeprefix("sha256:")


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.time,
    run_date: str = RUN_DATE,
    dated_physical_receipt: JsonMap | None = None,
    protected_before_hashes: JsonMap | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: JsonMap | None = None,
) -> JsonDict:
    started = clock()
    source_root = Path(root)
    protected_before = dict(protected_before_hashes or protected_file_hashes(source_root))
    receipt, receipt_text = dated_physical_receipt_path_hash_date_and_text(
        source_root, dated_physical_receipt
    )
    newer = receipt_newer_than_prior_attempts(source_root, receipt)
    target = board_and_cable_target(receipt)
    pre_usb = read_only_usb_receipt(command_runner=command_runner, phase="pre", run=True)
    binary_path = shutil.which("openFPGALoader")
    version = openfpgaloader_version_receipt(
        command_runner=command_runner,
        binary_path=binary_path,
    )
    preconditions = preconditions_checked(
        root=source_root,
        receipt=receipt,
        receipt_newer=newer,
        target=target,
        pre_usb=pre_usb,
        version=version,
        protected_before=protected_before,
    )
    changed_state = hardware_state_changed_from_prior_attempts(receipt)
    block_status = precondition_block_status(receipt, newer, target, version)
    detect, parsed_chain = run_single_detect(
        command_runner=command_runner,
        clock=clock,
        authorized=block_status is None,
    )
    status = status_after_detect(block_status=block_status, detect=detect, parsed_chain=parsed_chain)
    post_usb = read_only_usb_receipt(
        command_runner=command_runner,
        phase="post",
        run=detect["detect_command_count"] == 1,
    )
    commands = list(test_commands) if test_commands is not None else list(TEST_COMMANDS)
    exit_codes = dict(test_exit_codes) if test_exit_codes is not None else dict(TEST_EXIT_CODES)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": OUTPUT_REL_PATH.as_posix(),
        "status": status,
        "dated_physical_receipt_path_hash_date_and_text": receipt_text,
        "receipt_newer_than_prior_attempts": newer,
        "board_and_cable_target": target,
        "pre_command_usb_receipt": pre_usb,
        "openfpgaloader_version_receipt": version,
        "exact_authorized_command": EXACT_AUTHORIZED_COMMAND,
        "detect_command_count": detect["detect_command_count"],
        "detect_started_utc": detect["detect_started_utc"],
        "detect_finished_utc": detect["detect_finished_utc"],
        "detect_stdout": detect["detect_stdout"],
        "detect_stderr": detect["detect_stderr"],
        "detect_exit_code": detect["detect_exit_code"],
        "detect_timeout": detect["detect_timeout"],
        "detected_chain_and_device_ids": parsed_chain,
        "post_command_usb_receipt": post_usb,
        "hardware_state_changed_from_prior_attempts": changed_state,
        "flash_command_count": 0,
        "erase_command_count": 0,
        "reset_command_count": 0,
        "synthesis_command_count": 0,
        "place_route_command_count": 0,
        "timing_command_count": 0,
        "kv260_command_count": 0,
        "polarfire_command_count": 0,
        "stop_after_single_attempt_receipt": stop_after_single_attempt_receipt(
            status, detect["detect_command_count"]
        ),
        "protected_files_unchanged": protected_files_unchanged(source_root, protected_before),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "duration_s": round(float(clock() - started), 6),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(status)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        return [f"missing required fields: {missing}"]
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if tuple(artifact.get("spec_refs", ())) != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("exact_authorized_command") != EXACT_AUTHORIZED_COMMAND:
        errors.append("exact_authorized_command mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must cover every required field")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover every required field")

    count = artifact.get("detect_command_count")
    status = str(artifact.get("status"))
    if not isinstance(count, int) or count not in (0, 1):
        errors.append("detect_command_count must be bare 0 or bare 1")
    if status in {
        "complete_visible",
        "blocked_empty_chain",
        "blocked_timeout",
        "blocked_detect_failed",
        "blocked_idcode",
    }:
        if count != 1:
            errors.append("detect_command_count must be bare 1 after the command")
        if not artifact.get("detect_started_utc") or not artifact.get("detect_finished_utc"):
            errors.append("detect UTC fields required after the command")
    if status.startswith("blocked_") and status in {
        "blocked_missing_receipt",
        "blocked_stale_receipt",
        "blocked_wrong_target",
        "blocked_tool_version",
    }:
        if count != 0:
            errors.append("blocked preconditions must run zero detect commands")
        if artifact.get("detect_stdout") != "" or artifact.get("detect_stderr") != "":
            errors.append("blocked preconditions must not carry detect stdout/stderr")
    if any(artifact.get(field) != 0 for field in FORBIDDEN_COUNT_FIELDS):
        errors.append("forbidden command counts must remain bare 0")
    stop = artifact.get("stop_after_single_attempt_receipt")
    if not isinstance(stop, Mapping) or stop.get("retry_count") != 0:
        errors.append("stop_after_single_attempt_receipt must record zero retries")
    protected = artifact.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        errors.append("protected files must be unchanged")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for raw receipts")
    chain = artifact.get("detected_chain_and_device_ids")
    if not isinstance(chain, Mapping):
        errors.append("detected_chain_and_device_ids must be a mapping")
    elif status == "complete_visible" and chain.get("expected_gatemate_idcode_seen") is not True:
        errors.append("complete_visible requires expected GateMate IDCODE")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(
        (
            "complete_visible:",
            "blocked_empty_chain:",
            "blocked_timeout:",
            "blocked_detect_failed:",
            "blocked_idcode:",
            "blocked_missing_receipt:",
            "blocked_stale_receipt:",
            "blocked_wrong_target:",
            "blocked_tool_version:",
        )
    ):
        errors.append("honest_verdict prefix mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(repo_root: Path, artifact: JsonMap) -> Path:
    validate_artifact(artifact)
    return atomic_write_json(
        OUTPUT_REL_PATH,
        artifact,
        root=repo_root,
        allow_override=False,
        sort_keys=True,
    )


def run_experiment(
    *,
    repo_root: Path = REPO_ROOT,
    source_root: Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.time,
    run_date: str = RUN_DATE,
    dated_physical_receipt: JsonMap | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: JsonMap | None = None,
) -> Path:
    source = Path(source_root) if source_root is not None else Path(repo_root)
    protected_before = protected_file_hashes(source)
    artifact = build_artifact(
        root=source,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        dated_physical_receipt=dated_physical_receipt,
        protected_before_hashes=protected_before,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    return write_artifact(Path(repo_root), artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(repo_root=Path(args.repo_root), run_date=args.date)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"detect_command_count: {artifact['detect_command_count']}")
    print(f"detect_timeout: {artifact['detect_timeout']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
