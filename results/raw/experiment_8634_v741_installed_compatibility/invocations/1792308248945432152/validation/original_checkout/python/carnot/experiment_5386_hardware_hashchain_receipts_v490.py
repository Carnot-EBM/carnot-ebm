#!/usr/bin/env python3
"""Exp 5386: hash-chained hardware receipts for safe board actions.

Spec refs: REQ-HW-5386, SCENARIO-HW-5386.

This module records continuity receipts, not acceleration evidence. The hash
chain makes later audit cheaper: each board action is bound to the previous
action, the command string, the input and output hashes, the timestamp, the
board identity, and the exit status. That structure makes it harder for a later
summary to cherry-pick a successful workload while silently dropping an
unreachable board or an unsafe command.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5293_hardware_continuity_reachability_v483 as base
from carnot import experiment_5374_hardware_continuity_receipts_v489 as prev


JsonDict = dict[str, Any]
Clock = prev.Clock
CommandProbe = prev.CommandProbe
CommandRunner = prev.CommandRunner

RUN_DATE = "20260708"
EXPERIMENT_ID = "exp5386-hardware-hashchain-receipts-v490"
EXPERIMENT_NAME = "experiment_5386_hardware_hashchain_receipts"
MILESTONE = "2026.07.490"
SCHEMA = "carnot.experiment_5386.hardware_hashchain_receipts.v490"
SPEC_REFS = ("REQ-HW-5386", "SCENARIO-HW-5386")
RANDOM_SEED = 5386
RECEIPT_CONTRACT_VERSION = "carnot.hardware.hashchain_receipt.v1"
INFERENCE_SUBSTRATE = "hardware_hashchain_receipts_no_speedup"
HARDWARE_EVIDENCE_LEVEL = "hash_chained_board_workload_receipts_no_speedup"
GENESIS_CHAIN_HASH = "0" * 64

RESULT_RELATIVE_PATH = Path("results/experiment_5386_hardware_hashchain_receipts_v490.json")
PRIOR_RESULT_RELATIVE_PATH = Path("results/experiment_5374_hardware_continuity_receipts_v489.json")
PRIOR_WORKLOAD_RELATIVE_PATH = Path(
    "results/experiment_5361_hardware_continuity_workload_v488.json"
)

HOST_DATE_COMMAND = prev.HOST_DATE_COMMAND
HARDWARE_ENV_COMMAND = prev.HARDWARE_ENV_COMMAND
TOOL_VERSION_COMMAND = prev.TOOL_VERSION_COMMAND
GATEMATE_USB_COMMAND = prev.GATEMATE_USB_COMMAND
POLARFIRE_USB_COMMAND = prev.POLARFIRE_USB_COMMAND
GPU_CONTEXT_COMMAND = prev.GPU_CONTEXT_COMMAND
KV260_SSH_TRUE_COMMAND = prev.KV260_SSH_TRUE_COMMAND
KV260_REQUIRED_COMMAND_FORM = prev.KV260_REQUIRED_COMMAND_FORM
POLARFIRE_STATUS_COMMAND = prev.POLARFIRE_STATUS_COMMAND
GATEMATE_DETECT_COMMAND = prev.GATEMATE_DETECT_COMMAND

LOCAL_TIMEOUT_S = prev.LOCAL_TIMEOUT_S
SSH_TIMEOUT_S = prev.SSH_TIMEOUT_S
GATEMATE_TIMEOUT_S = prev.GATEMATE_TIMEOUT_S
TERMINAL_PREFIXES = ("complete:", "honest_blocked")
BOARDS_CHECKED = ("KV260", "PolarFire", "GateMate")

POLARFIRE_WORKLOAD_INPUT = b"carnot-exp5386-polarfire-workload-v490\n"
POLARFIRE_WORKLOAD_OUTPUT_SUFFIX = b"|polarfire-v490-output"
POLARFIRE_EXPECTED_INPUT_SHA256 = hashlib.sha256(POLARFIRE_WORKLOAD_INPUT).hexdigest()
POLARFIRE_EXPECTED_OUTPUT_SHA256 = hashlib.sha256(
    POLARFIRE_WORKLOAD_INPUT + POLARFIRE_WORKLOAD_OUTPUT_SUFFIX
).hexdigest()
POLARFIRE_WORKLOAD_PYTHON = (
    "import hashlib,json,platform,socket,time;"
    "started=time.perf_counter();"
    'payload=b"carnot-exp5386-polarfire-workload-v490\\n";'
    'out=hashlib.sha256(payload+b"|polarfire-v490-output").hexdigest();'
    "receipt={"
    '"hostname":socket.gethostname(),'
    '"uname":" ".join(platform.uname()),'
    '"python_version":platform.python_version(),'
    '"input_sha256":hashlib.sha256(payload).hexdigest(),'
    '"output_sha256":out,'
    '"wall_time_s":round(time.perf_counter()-started,6)'
    "};"
    "print(json.dumps(receipt,sort_keys=True))"
)
POLARFIRE_WORKLOAD_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    f"python3 -c '{POLARFIRE_WORKLOAD_PYTHON}'",
)

REQUIRED_WRAPPED_FIELDS = (
    "status",
    "hardware_hash_chained_receipt_ready",
    "hardware_speedup_claim",
    "boards_checked",
    "kv260_status",
    "polar_fire_status",
    "gatemate_status",
    "workload_hash_chain",
    "commands_run",
    "no_host_mmcblk_kv260_evidence",
    "no_destructive_flash",
    "repeatability_evidence_present",
    "receipt_contract_version",
    "honest_verdict",
)
METADATA_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "inference_substrate",
    "preconditions_checked",
    "tests_run",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Identifies the v490 hash-chain receipt artifact.",
    "milestone": "Pins this receipt to milestone 2026.07.490 rather than a floating board state.",
    "status": "Complete means safe board actions were recorded; honest_blocked means no safe board interaction was possible.",
    "hardware_hash_chained_receipt_ready": (
        "True only when at least one safe board workload validates inside the hash chain "
        "and blocked boards are represented honestly."
    ),
    "hardware_speedup_claim": "False unless repeatable board timing evidence exists.",
    "boards_checked": "Lists the active hardware lanes that this receipt classified.",
    "kv260_status": (
        "Records KV260 SSH reachability and workload status without host block-device evidence."
    ),
    "polar_fire_status": (
        "Records PolarFire SSH reachability and the board-local workload receipt when valid."
    ),
    "gatemate_status": "Records GateMate DirtyJTAG, toolchain, and physical/JTAG status.",
    "workload_hash_chain": (
        "Ordered records chaining command, input, output, timestamp, board identity, "
        "exit status, and previous hash for each safe board action."
    ),
    "commands_run": "Records every safe command executed with exit code and output hashes.",
    "no_host_mmcblk_kv260_evidence": (
        "True because KV260 evidence is SSH-only and not derived from host removable storage."
    ),
    "no_destructive_flash": "True because no flashing, programming, or destructive write command was run.",
    "repeatability_evidence_present": (
        "States whether authenticated repeated board timing evidence exists."
    ),
    "receipt_contract_version": "Names the hash-chain receipt schema used by this artifact.",
    "honest_verdict": "One-line outcome that reports receipt discipline rather than speedup.",
    "inference_substrate": (
        "hardware_hashchain_receipts_no_speedup means these are continuity receipts, "
        "not sampler acceleration measurements."
    ),
    "preconditions_checked": (
        "Records date, sanitized environment, local toolchain, USB visibility, and SSH targets."
    ),
    "tests_run": "Records verification commands without treating them as hardware evidence.",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def sha256_text(text: str) -> str:
    return prev.sha256_text(text)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return base.sha256_json(stable)


def receipt_timestamp(run_date: str, index: int) -> str:
    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run_date must be YYYYMMDD")
    year, month, day = run_date[:4], run_date[4:6], run_date[6:8]
    minute_total, second = divmod(int(index), 60)
    hour, minute = divmod(minute_total, 60)
    return f"{year}-{month}-{day}T{hour:02d}:{minute:02d}:{second:02d}Z"


def combined_output_sha256(command: Mapping[str, Any]) -> str:
    stdout_hash = str(command.get("stdout_sha256", ""))
    stderr_hash = str(command.get("stderr_sha256", ""))
    return sha256_text(f"stdout={stdout_hash}\nstderr={stderr_hash}")


def parse_polarfire_workload_stdout(stdout: str) -> tuple[JsonDict | None, str | None]:
    parsed: Any | None = None
    for line in stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            break
    if not isinstance(parsed, Mapping):
        return None, "workload stdout is not valid JSON"

    receipt: JsonDict = {
        "hostname": parsed.get("hostname"),
        "uname": parsed.get("uname"),
        "python_version": parsed.get("python_version"),
        "input_sha256": parsed.get("input_sha256"),
        "output_sha256": parsed.get("output_sha256"),
        "wall_time_s": parsed.get("wall_time_s"),
        "workload_class": "deterministic_sha256_transform",
        "board_local": True,
        "speedup_claimed": False,
    }
    errors: list[str] = []
    for key in ("hostname", "uname", "input_sha256", "output_sha256"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            errors.append(f"{key} missing")
    if receipt.get("input_sha256") != POLARFIRE_EXPECTED_INPUT_SHA256:
        errors.append("input_sha256 mismatch")
    if receipt.get("output_sha256") != POLARFIRE_EXPECTED_OUTPUT_SHA256:
        errors.append("output_sha256 mismatch")
    if not isinstance(receipt.get("wall_time_s"), int | float) or receipt["wall_time_s"] < 0:
        errors.append("wall_time_s invalid")
    python_version = receipt.get("python_version")
    if python_version is not None and not isinstance(python_version, str):
        errors.append("python_version invalid")
    return receipt, "; ".join(errors) if errors else None


def polarfire_status_with_workload(
    *, command_runner: CommandRunner
) -> tuple[str, str | None, JsonDict, JsonDict | None, list[JsonDict]]:
    status_probe = command_runner(POLARFIRE_STATUS_COMMAND, SSH_TIMEOUT_S)
    ssh_reachable = status_probe.exit_code == 0
    status_detail: JsonDict = {
        "board": "PolarFire",
        "ssh_reachable": ssh_reachable,
        "probe_exit_code": int(status_probe.exit_code),
        "remote_identifier": base.remote_identifier(status_probe.combined_output)
        if ssh_reachable
        else None,
        "workload_attempted": False,
        "workload_validated": False,
        "workload_receipt": None,
        "speedup_claimed": False,
    }
    commands = [
        prev.command_receipt(
            probe=status_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_authenticated_status_probe",
            outcome="reachable" if ssh_reachable else "unreachable",
        )
    ]
    if not ssh_reachable:
        status_detail["status"] = "unreachable"
        return (
            "unreachable",
            None,
            status_detail,
            prev.blocker_from_probe("unreachable", status_probe, SSH_TIMEOUT_S),
            commands,
        )

    workload_probe = command_runner(POLARFIRE_WORKLOAD_COMMAND, SSH_TIMEOUT_S)
    receipt, parse_error = parse_polarfire_workload_stdout(workload_probe.stdout)
    exit_ok = workload_probe.exit_code == 0
    validated = bool(exit_ok and receipt is not None and parse_error is None)
    if validated:
        status = "reachable/workload_receipt"
        workload_hash = str(receipt["output_sha256"])
        blocker = None
        workload_error = None
    else:
        status = "skipped: workload receipt invalid"
        workload_hash = None
        workload_error = parse_error or workload_probe.stderr.strip() or "workload command failed"
        blocker = prev.blocker_from_probe(workload_error, workload_probe, SSH_TIMEOUT_S)

    status_detail.update(
        {
            "status": status,
            "workload_attempted": True,
            "workload_validated": validated,
            "workload_receipt": receipt,
            "workload_error": workload_error,
        }
    )
    commands.append(
        prev.command_receipt(
            probe=workload_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_board_local_workload_receipt",
            outcome=status,
        )
    )
    return status, workload_hash, status_detail, blocker, commands


def board_identity(board: str, detail: Mapping[str, Any]) -> str:
    remote_identifier = detail.get("remote_identifier")
    if isinstance(remote_identifier, str) and remote_identifier:
        return remote_identifier
    if board == "GateMate":
        detect_identifier = detail.get("detect_output_identifier")
        if isinstance(detect_identifier, str) and detect_identifier:
            return detect_identifier
        usb = detail.get("dirtyjtag_usb", {})
        if isinstance(usb, Mapping) and usb.get("visible") is True:
            return "DirtyJTAG 1209:c0ca visible"
        return "GateMate physical/JTAG path unavailable"
    if board == "KV260":
        return "ssh target kria"
    if board == "PolarFire":
        return "ssh target polarfire"
    return board


def hash_chain_record(record: Mapping[str, Any]) -> str:
    stable = dict(record)
    stable["record_hash"] = ""
    return base.sha256_json(stable)


def chain_action_for_receipt(
    *,
    receipt: Mapping[str, Any],
    board: str,
    action: str,
    detail: Mapping[str, Any],
    run_date: str,
    index: int,
    previous_hash: str,
) -> JsonDict:
    input_sha256 = sha256_text("")
    output_sha256 = combined_output_sha256(receipt)
    workload_validated = False
    if receipt.get("kind") == "polarfire_board_local_workload_receipt":
        workload = detail.get("workload_receipt")
        if isinstance(workload, Mapping):
            workload_input = workload.get("input_sha256")
            workload_output = workload.get("output_sha256")
            if isinstance(workload_input, str) and len(workload_input) == 64:
                input_sha256 = workload_input
            if isinstance(workload_output, str) and len(workload_output) == 64:
                output_sha256 = workload_output
        workload_validated = detail.get("workload_validated") is True

    record: JsonDict = {
        "index": index,
        "board": board,
        "action": action,
        "kind": receipt.get("kind"),
        "command": receipt.get("command"),
        "command_sha256": sha256_text(str(receipt.get("command", ""))),
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "stdout_sha256": receipt.get("stdout_sha256"),
        "stderr_sha256": receipt.get("stderr_sha256"),
        "timestamp_utc": receipt_timestamp(run_date, index),
        "board_identity": board_identity(board, detail),
        "exit_status": int(receipt.get("exit_code", -1)),
        "exit_code": int(receipt.get("exit_code", -1)),
        "status": receipt.get("outcome"),
        "workload_receipt_validated": workload_validated,
        "previous_hash": previous_hash,
        "record_hash": "",
    }
    record["record_hash"] = hash_chain_record(record)
    return record


def build_workload_hash_chain(
    *,
    board_receipts: Sequence[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]],
    run_date: str,
) -> list[JsonDict]:
    chain: list[JsonDict] = []
    previous_hash = GENESIS_CHAIN_HASH
    for index, (board, action, detail, receipt) in enumerate(board_receipts):
        record = chain_action_for_receipt(
            receipt=receipt,
            board=board,
            action=action,
            detail=detail,
            run_date=run_date,
            index=index,
            previous_hash=previous_hash,
        )
        previous_hash = str(record["record_hash"])
        chain.append(record)
    return chain


def default_tests_run() -> list[JsonDict]:
    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def honest_verdict(
    *,
    status: str,
    receipt_ready: bool,
    kv260_status: str,
    polar_fire_status: str,
    gatemate_status: str,
) -> str:
    prefix = "complete:" if status == "complete" else "honest_blocked"
    ready = str(bool(receipt_ready)).lower()
    return (
        f"{prefix} "
        f"receipt_ready={ready} "
        f"kv260={kv260_status} "
        f"polar_fire={polar_fire_status} "
        f"gatemate={gatemate_status} "
        "speedup_claim=false"
    )


def build_artifact(
    *,
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    gatemate_physical_path_available: bool | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = clock()
    context, commands_run = prev.collect_preconditions(command_runner)
    context["prior_receipts"] = [str(PRIOR_RESULT_RELATIVE_PATH), str(PRIOR_WORKLOAD_RELATIVE_PATH)]
    gate_path_available = prev.gatemate_path_available_from_context(
        context, gatemate_physical_path_available
    )
    context["operator_visible_hardware_assumptions"]["gatemate_physical_jtag_path_available"] = (
        gate_path_available
    )

    kv260_probe = command_runner(KV260_SSH_TRUE_COMMAND, SSH_TIMEOUT_S)
    kv260_status, kv260_detail, kv260_blocker, kv260_receipt = prev.kv260_status_from_probe(
        kv260_probe
    )
    commands_run.append(kv260_receipt)

    polar_fire_status, _, polar_fire_detail, polar_fire_blocker, polar_fire_commands = (
        polarfire_status_with_workload(command_runner=command_runner)
    )
    commands_run.extend(polar_fire_commands)

    gatemate_status, gatemate_detail, gatemate_blocker, gatemate_commands = (
        prev.gatemate_status_from_context(
            command_runner=command_runner,
            context=context,
            physical_path_available=gate_path_available,
        )
    )
    commands_run.extend(gatemate_commands)

    board_receipts: list[tuple[str, str, Mapping[str, Any], Mapping[str, Any]]] = [
        ("KV260", "ssh_reachability", kv260_detail, kv260_receipt),
    ]
    for command in polar_fire_commands:
        action = (
            "board_local_workload"
            if command.get("kind") == "polarfire_board_local_workload_receipt"
            else "ssh_status"
        )
        board_receipts.append(("PolarFire", action, polar_fire_detail, command))
    for command in gatemate_commands:
        board_receipts.append(("GateMate", "safe_detect", gatemate_detail, command))

    hash_chain = build_workload_hash_chain(board_receipts=board_receipts, run_date=run_date)
    receipt_ready = any(record["workload_receipt_validated"] is True for record in hash_chain)
    status = "complete" if hash_chain else "honest_blocked"
    tests = [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "spec_refs": list(SPEC_REFS),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "duration_s": base.round_duration(clock() - started),
        "commit": commit,
        "status": wrap_field("status", status),
        "hardware_hash_chained_receipt_ready": wrap_field(
            "hardware_hash_chained_receipt_ready", receipt_ready
        ),
        "hardware_speedup_claim": wrap_field("hardware_speedup_claim", False),
        "boards_checked": wrap_field("boards_checked", list(BOARDS_CHECKED)),
        "kv260_status": wrap_field("kv260_status", kv260_detail),
        "polar_fire_status": wrap_field("polar_fire_status", polar_fire_detail),
        "gatemate_status": wrap_field("gatemate_status", gatemate_detail),
        "workload_hash_chain": wrap_field("workload_hash_chain", hash_chain),
        "commands_run": wrap_field("commands_run", commands_run),
        "no_host_mmcblk_kv260_evidence": wrap_field("no_host_mmcblk_kv260_evidence", True),
        "no_destructive_flash": wrap_field("no_destructive_flash", True),
        "repeatability_evidence_present": wrap_field("repeatability_evidence_present", False),
        "receipt_contract_version": wrap_field(
            "receipt_contract_version", RECEIPT_CONTRACT_VERSION
        ),
        "honest_verdict": wrap_field(
            "honest_verdict",
            honest_verdict(
                status=status,
                receipt_ready=receipt_ready,
                kv260_status=kv260_status,
                polar_fire_status=polar_fire_status,
                gatemate_status=gatemate_status,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": wrap_field("preconditions_checked", context),
        "tests_run": wrap_field("tests_run", tests),
        "hardware_evidence_level": HARDWARE_EVIDENCE_LEVEL,
        "board_details": {
            "KV260": kv260_detail,
            "PolarFire": polar_fire_detail,
            "GateMate": gatemate_detail,
        },
        "blocked_reason": {
            "KV260": kv260_blocker,
            "PolarFire": polar_fire_blocker,
            "GateMate": gatemate_blocker,
        },
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "research-hardware-wishlist.md",
            str(PRIOR_RESULT_RELATIVE_PATH),
            str(PRIOR_WORKLOAD_RELATIVE_PATH),
            "ops/exclusion_manifest.yaml",
            "openspec/capabilities/fpga/spec.md",
            "tests/python/",
            "scripts/",
        ],
        "docs_update_decision": {
            "ops_changelog_updated": False,
            "ops_status_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["no_host_mmcblk_kv260_evidence"]["value"] = prev.no_host_mmcblk_kv260_evidence(
        artifact
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_wrapped_field(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
    require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def validate_commands(commands: Any) -> None:
    require(isinstance(commands, list) and commands, "commands_run must be a non-empty list")
    for index, command in enumerate(commands):
        require(isinstance(command, Mapping), f"commands_run[{index}] must be a mapping")
        for key in (
            "kind",
            "command",
            "outcome",
            "exit_code",
            "timeout_s",
            "duration_s",
            "stdout_sha256",
            "stderr_sha256",
        ):
            require(key in command, f"commands_run[{index}] missing {key}")
        require(
            not prev.command_is_destructive(str(command["command"])),
            f"destructive command recorded at commands_run[{index}]",
        )
    require(
        any(command.get("command") == KV260_REQUIRED_COMMAND_FORM for command in commands),
        "commands_run missing exact KV260 SSH command",
    )


def validate_tests_run(tests_run: Any) -> None:
    require(isinstance(tests_run, list) and tests_run, "tests_run must be a non-empty list")
    for index, item in enumerate(tests_run):
        require(isinstance(item, Mapping), f"tests_run[{index}] must be a mapping")
        require(
            isinstance(item.get("command"), str) and item["command"], "tests_run command missing"
        )
        require(
            isinstance(item.get("outcome"), str) and item["outcome"], "tests_run outcome missing"
        )


def validate_status_detail(label: str, detail: Any) -> str:
    require(isinstance(detail, Mapping), f"{label} must be a mapping")
    status = detail.get("status")
    require(isinstance(status, str) and prev._status_is_allowed(status), f"{label} status invalid")
    require(detail.get("speedup_claimed") is False, f"{label} speedup_claimed must be false")
    return status


def validate_hash_chain(chain: Any, *, status: str) -> bool:
    require(isinstance(chain, list), "workload_hash_chain must be a list")
    if status == "complete":
        require(chain, "complete status requires a non-empty workload_hash_chain")
    previous_hash = GENESIS_CHAIN_HASH
    valid_workload = False
    for index, record in enumerate(chain):
        require(isinstance(record, Mapping), f"workload_hash_chain[{index}] must be a mapping")
        for key in (
            "index",
            "board",
            "action",
            "kind",
            "command",
            "command_sha256",
            "input_sha256",
            "output_sha256",
            "timestamp_utc",
            "board_identity",
            "exit_status",
            "status",
            "workload_receipt_validated",
            "previous_hash",
            "record_hash",
        ):
            require(key in record, f"workload_hash_chain[{index}] missing {key}")
        require(record["index"] == index, f"workload_hash_chain[{index}] index mismatch")
        require(
            record["previous_hash"] == previous_hash,
            f"workload_hash_chain[{index}] previous_hash mismatch",
        )
        require(
            record["record_hash"] == hash_chain_record(record),
            f"workload_hash_chain[{index}] record_hash mismatch",
        )
        for key in ("command_sha256", "input_sha256", "output_sha256", "previous_hash"):
            value = record[key]
            require(isinstance(value, str) and len(value) == 64, f"{key} must be sha256")
        require(
            isinstance(record["timestamp_utc"], str) and record["timestamp_utc"].endswith("Z"),
            "bad timestamp",
        )
        require(
            not prev.command_is_destructive(str(record["command"])),
            "destructive command in hash chain",
        )
        valid_workload = valid_workload or record["workload_receipt_validated"] is True
        previous_hash = str(record["record_hash"])
    return valid_workload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in (*METADATA_WRAPPED_FIELDS, *REQUIRED_WRAPPED_FIELDS):
        validate_wrapped_field(artifact, field)
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(
        validate_wrapped_field(artifact, "experiment_id") == EXPERIMENT_ID,
        "experiment_id mismatch",
    )
    require(validate_wrapped_field(artifact, "milestone") == MILESTONE, "milestone mismatch")
    require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs mismatch")
    require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")

    status = validate_wrapped_field(artifact, "status")
    require(status in {"complete", "honest_blocked"}, "status invalid")
    require(
        validate_wrapped_field(artifact, "boards_checked") == list(BOARDS_CHECKED),
        "boards_checked mismatch",
    )
    kv260_status = validate_status_detail(
        "kv260_status", validate_wrapped_field(artifact, "kv260_status")
    )
    polar_fire_status = validate_status_detail(
        "polar_fire_status", validate_wrapped_field(artifact, "polar_fire_status")
    )
    gatemate_status = validate_status_detail(
        "gatemate_status", validate_wrapped_field(artifact, "gatemate_status")
    )
    valid_workload = validate_hash_chain(
        validate_wrapped_field(artifact, "workload_hash_chain"), status=str(status)
    )
    require(
        validate_wrapped_field(artifact, "hardware_hash_chained_receipt_ready") is valid_workload,
        "hardware_hash_chained_receipt_ready mismatch",
    )
    require(
        validate_wrapped_field(artifact, "hardware_speedup_claim") is False,
        "hardware_speedup_claim must be false",
    )
    require(
        validate_wrapped_field(artifact, "no_host_mmcblk_kv260_evidence") is True,
        "no_host_mmcblk_kv260_evidence must be true",
    )
    require(
        validate_wrapped_field(artifact, "no_destructive_flash") is True,
        "no_destructive_flash must be true",
    )
    require(
        validate_wrapped_field(artifact, "repeatability_evidence_present") is False,
        "repeatability_evidence_present must be false",
    )
    require(
        validate_wrapped_field(artifact, "receipt_contract_version") == RECEIPT_CONTRACT_VERSION,
        "receipt_contract_version mismatch",
    )
    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "bad verdict")
    for token in (
        f"kv260={kv260_status}",
        f"polar_fire={polar_fire_status}",
        f"gatemate={gatemate_status}",
        "speedup_claim=false",
    ):
        require(token in verdict, f"honest_verdict missing {token}")
    require(
        validate_wrapped_field(artifact, "inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    validate_commands(validate_wrapped_field(artifact, "commands_run"))
    validate_tests_run(validate_wrapped_field(artifact, "tests_run"))
    require(
        prev.no_host_mmcblk_kv260_evidence(artifact), "host KV260 block-device evidence present"
    )
    require(artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum mismatch",
    )


def write_artifact(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / RESULT_RELATIVE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str | None = None,
    gatemate_physical_path_available: bool | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or base.get_git_commit(repo_root),
        gatemate_physical_path_available=gatemate_physical_path_available,
        tests_run=tests_run,
    )
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--gatemate-physical-path-available",
        action="store_true",
        help="Run bounded GateMate detect because the current physical/JTAG path is available.",
    )
    args = parser.parse_args(argv)
    print(
        run_experiment(
            repo_root=Path("."),
            run_date=args.date,
            gatemate_physical_path_available=args.gatemate_physical_path_available or None,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
