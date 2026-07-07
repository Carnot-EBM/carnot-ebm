#!/usr/bin/env python3
"""Exp 5374: hardware continuity receipts for reachable boards.

Spec refs: REQ-HW-5374, SCENARIO-HW-5374.

This module records board-continuity evidence, not performance evidence. KV260
is checked only through the required SSH reachability command, PolarFire gets a
tiny hash-checked board-local workload when SSH is reachable, and GateMate uses
only safe toolchain or detect checks when the physical/JTAG path is available.
Those receipts show whether board lanes are still alive; they do not include
baseline timing, repeat timing, or a sampler workload comparison, so they cannot
support a hardware speedup claim.
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
from carnot import experiment_5319_hardware_continuity_no_speedup_v485 as context_prev
from carnot import experiment_5361_hardware_continuity_workload_v488 as prev


JsonDict = dict[str, Any]
Clock = context_prev.Clock
CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner

RUN_DATE = "20260707"
EXPERIMENT_ID = "exp5374-hardware-continuity-receipts-v489"
EXPERIMENT_NAME = "experiment_5374_hardware_continuity_receipts"
MILESTONE = "2026.07.489"
SCHEMA = "carnot.experiment_5374.hardware_continuity_receipts.v489"
SPEC_REFS = ("REQ-HW-5374", "SCENARIO-HW-5374")
RANDOM_SEED = 5374
INFERENCE_SUBSTRATE = "hardware_continuity_receipts_no_speedup"
HARDWARE_EVIDENCE_LEVEL = "board_continuity_receipts_no_speedup"

RESULT_RELATIVE_PATH = Path("results/experiment_5374_hardware_continuity_receipts_v489.json")
PRIOR_RESULT_RELATIVE_PATH = Path("results/experiment_5361_hardware_continuity_workload_v488.json")

HOST_DATE_COMMAND = prev.HOST_DATE_COMMAND
HARDWARE_ENV_COMMAND = prev.HARDWARE_ENV_COMMAND
TOOL_VERSION_COMMAND = prev.TOOL_VERSION_COMMAND
GATEMATE_USB_COMMAND = prev.GATEMATE_USB_COMMAND
POLARFIRE_USB_COMMAND = prev.POLARFIRE_USB_COMMAND
GPU_CONTEXT_COMMAND = prev.GPU_CONTEXT_COMMAND
KV260_SSH_TRUE_COMMAND = prev.KV260_SSH_TRUE_COMMAND
KV260_REQUIRED_COMMAND_FORM = "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'"
POLARFIRE_STATUS_COMMAND = prev.POLARFIRE_STATUS_COMMAND
GATEMATE_DETECT_COMMAND = prev.GATEMATE_DETECT_COMMAND

LOCAL_TIMEOUT_S = prev.LOCAL_TIMEOUT_S
SSH_TIMEOUT_S = prev.SSH_TIMEOUT_S
GATEMATE_TIMEOUT_S = prev.GATEMATE_TIMEOUT_S
MAX_OUTPUT_EXCERPT_CHARS = 480
TERMINAL_PREFIXES = ("complete:", "blocked_")

POLARFIRE_WORKLOAD_INPUT = b"carnot-exp5374-polarfire-workload-v489\n"
POLARFIRE_WORKLOAD_OUTPUT_SUFFIX = b"|polarfire-v489-output"
POLARFIRE_EXPECTED_INPUT_SHA256 = hashlib.sha256(POLARFIRE_WORKLOAD_INPUT).hexdigest()
POLARFIRE_EXPECTED_OUTPUT_SHA256 = hashlib.sha256(
    POLARFIRE_WORKLOAD_INPUT + POLARFIRE_WORKLOAD_OUTPUT_SUFFIX
).hexdigest()
POLARFIRE_WORKLOAD_PYTHON = (
    "import hashlib,json,platform,socket,time;"
    "started=time.perf_counter();"
    'payload=b"carnot-exp5374-polarfire-workload-v489\\n";'
    'out=hashlib.sha256(payload+b"|polarfire-v489-output").hexdigest();'
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
    "hardware_speedup_claim",
    "kv260_checked_via_ssh",
    "kv260_status",
    "polarfire_status",
    "polarfire_workload_hash",
    "gatemate_status",
    "commands_run",
    "no_host_mmcblk_kv260_evidence",
    "no_destructive_flash",
    "repeatability_evidence_present",
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
    "experiment_id": "Identifies the v489 continuity receipt so reconciliation can cite one exact artifact.",
    "milestone": "Pins this receipt to milestone 2026.07.489 rather than a floating board state.",
    "status": "Complete means fresh reachability or receipt attempts were recorded for the requested board lanes.",
    "hardware_speedup_claim": (
        "False unless authenticated baseline timing, board timing, workload hash, "
        "and repeatability evidence are all present."
    ),
    "kv260_checked_via_ssh": (
        "True only when the required BatchMode SSH true command was attempted for KV260."
    ),
    "kv260_status": "KV260 is classified as reachable, unreachable, or skipped with an honest reason.",
    "polarfire_status": (
        "PolarFire is classified as reachable/workload_receipt, unreachable, "
        "or skipped with an honest reason."
    ),
    "polarfire_workload_hash": (
        "The value is the board-local workload output hash only when that receipt validates."
    ),
    "gatemate_status": (
        "GateMate is detected, blocked on physical/JTAG availability, unreachable, "
        "or skipped with an honest reason."
    ),
    "commands_run": (
        "Records exact command strings, exit codes, and concise redacted output excerpts "
        "or hashes for every attempted status or workload command."
    ),
    "no_host_mmcblk_kv260_evidence": (
        "True because KV260 evidence is SSH-only and not derived from host removable storage."
    ),
    "no_destructive_flash": "True because no flashing, programming, or destructive write command was run.",
    "repeatability_evidence_present": (
        "States whether authenticated repeat timing evidence exists for a hardware speedup claim."
    ),
    "honest_verdict": "One-line continuity summary naming KV260, PolarFire, GateMate, and no-speedup status.",
    "inference_substrate": (
        "hardware_continuity_receipts_no_speedup means these are board continuity receipts, "
        "not sampler acceleration measurements."
    ),
    "preconditions_checked": (
        "Records date, sanitized environment, local toolchain, USB visibility, and SSH targets before board classification."
    ),
    "tests_run": "Records verification commands without treating them as hardware evidence.",
}

HOST_BLOCK_DEVICE_MARKERS = ("/dev/mmcblk", "/dev/disk", "/dev/sd")
DESTRUCTIVE_COMMAND_MARKERS = (
    "--write",
    "--flash",
    " flash ",
    "flashcp",
    "dd if=",
    " dd ",
    "program_hw_devices",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return base.sha256_json(stable)


def short_excerpt(text: str, *, max_chars: int = MAX_OUTPUT_EXCERPT_CHARS) -> str:
    clean = text.replace("\x00", "").strip()
    if len(clean) <= max_chars:
        return clean
    suffix = f"...<truncated {len(clean) - max_chars} chars>"
    return clean[: max(0, max_chars - len(suffix))] + suffix


def command_receipt(
    *,
    probe: CommandProbe,
    timeout_s: float,
    kind: str,
    outcome: str,
    command_override: str | None = None,
    stdout_override: str | None = None,
) -> JsonDict:
    stdout = probe.stdout if stdout_override is None else stdout_override
    return {
        "kind": kind,
        "command": command_override or base.command_to_string(probe.command),
        "timeout_s": float(timeout_s),
        "exit_code": int(probe.exit_code),
        "duration_s": base.round_duration(probe.duration_s),
        "outcome": outcome,
        "stdout_excerpt": short_excerpt(stdout),
        "stderr_excerpt": short_excerpt(probe.stderr),
        "stdout_sha256": sha256_text(stdout),
        "stderr_sha256": sha256_text(probe.stderr),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(probe.stderr.splitlines()),
    }


def collect_preconditions(command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    host_probe = command_runner(HOST_DATE_COMMAND, LOCAL_TIMEOUT_S)
    env_probe = command_runner(HARDWARE_ENV_COMMAND, LOCAL_TIMEOUT_S)
    tool_probe = command_runner(TOOL_VERSION_COMMAND, LOCAL_TIMEOUT_S)
    gatemate_usb_probe = command_runner(GATEMATE_USB_COMMAND, LOCAL_TIMEOUT_S)
    polarfire_usb_probe = command_runner(POLARFIRE_USB_COMMAND, LOCAL_TIMEOUT_S)
    gpu_probe = command_runner(GPU_CONTEXT_COMMAND, LOCAL_TIMEOUT_S)

    env_summary = context_prev.parse_hardware_environment(env_probe)
    gpu_context = context_prev.parse_gpu_context(gpu_probe)
    context = {
        "host_date": context_prev.parse_host_date(host_probe),
        "hardware_environment": env_summary,
        "tool_versions": base.parse_tool_versions(tool_probe),
        "usb_visibility": {
            "GateMate": base.parse_usb_visible(gatemate_usb_probe, "1209:c0ca"),
            "PolarFire": base.parse_usb_visible(polarfire_usb_probe, "1514:2008"),
        },
        "gpu_runtime_context": gpu_context,
        "public_reference_boundaries": context_prev.public_reference_boundaries(env_summary),
        "ssh_targets": {"KV260": "kria", "PolarFire": "polarfire"},
        "kv260_check_method": "ssh_batchmode_true_only",
        "operator_visible_hardware_assumptions": {
            "kv260_checked_by_ssh_only": True,
            "kv260_host_storage_precondition_retired": True,
            "gatemate_physical_jtag_path_available": None,
            "no_destructive_flash": True,
            "hardware_speedup_claim": False,
        },
        "prior_receipts": [str(PRIOR_RESULT_RELATIVE_PATH)],
    }
    commands = [
        command_receipt(
            probe=host_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="step0_host_date_context",
            outcome="recorded" if host_probe.exit_code == 0 else "host_date_unavailable",
        ),
        command_receipt(
            probe=env_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="local_hardware_environment",
            outcome="recorded" if env_probe.exit_code == 0 else "env_unavailable",
            stdout_override=base.sanitized_env_stdout(env_summary),
        ),
        command_receipt(
            probe=tool_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="local_tool_versions",
            outcome="recorded" if tool_probe.exit_code == 0 else "tool_version_probe_failed",
        ),
        command_receipt(
            probe=gatemate_usb_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="local_usb_gatemate_dirtyjtag",
            outcome="visible" if context["usb_visibility"]["GateMate"]["visible"] else "not_visible",
        ),
        command_receipt(
            probe=polarfire_usb_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="local_usb_polarfire_flashpro",
            outcome="visible" if context["usb_visibility"]["PolarFire"]["visible"] else "not_visible",
        ),
        command_receipt(
            probe=gpu_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="local_dual_rtx3090_runtime_context",
            outcome="dual_rtx3090_context"
            if gpu_context["dual_rtx3090_present"]
            else "gpu_context_not_dual_rtx3090",
        ),
    ]
    return context, commands


def blocker_from_probe(
    reason: str,
    probe: CommandProbe,
    timeout_s: float,
    *,
    command_override: str | None = None,
) -> JsonDict:
    return {
        "reason": reason,
        "command": command_override or base.command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "timeout_s": float(timeout_s),
        "stdout_excerpt": short_excerpt(probe.stdout),
        "stderr_excerpt": short_excerpt(probe.stderr),
        "stdout_sha256": sha256_text(probe.stdout),
        "stderr_sha256": sha256_text(probe.stderr),
    }


def kv260_status_from_probe(probe: CommandProbe) -> tuple[str, JsonDict, JsonDict | None, JsonDict]:
    reachable = probe.exit_code == 0
    status = "reachable" if reachable else "unreachable"
    detail = {
        "board": "KV260",
        "status": status,
        "ssh_reachable": reachable,
        "check_method": "ssh_batchmode_true_only",
        "command_form": KV260_REQUIRED_COMMAND_FORM,
        "probe_exit_code": int(probe.exit_code),
        "remote_identifier": base.remote_identifier(probe.combined_output)
        if probe.combined_output
        else None,
        "speedup_claimed": False,
    }
    blocker = None
    if not reachable:
        blocker = blocker_from_probe(
            "unreachable",
            probe,
            SSH_TIMEOUT_S,
            command_override=KV260_REQUIRED_COMMAND_FORM,
        )
    receipt = command_receipt(
        probe=probe,
        timeout_s=SSH_TIMEOUT_S,
        kind="kv260_ssh_true_reachability_probe",
        outcome=status,
        command_override=KV260_REQUIRED_COMMAND_FORM,
    )
    return status, detail, blocker, receipt


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
        command_receipt(
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
            blocker_from_probe("unreachable", status_probe, SSH_TIMEOUT_S),
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
        blocker = blocker_from_probe(
            workload_error,
            workload_probe,
            SSH_TIMEOUT_S,
        )

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
        command_receipt(
            probe=workload_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_board_local_workload_receipt",
            outcome=status,
        )
    )
    return status, workload_hash, status_detail, blocker, commands


def gatemate_path_available_from_context(
    context: Mapping[str, Any], explicit: bool | None
) -> bool:
    if explicit is not None:
        return bool(explicit)
    env_summary = context.get("hardware_environment", {})
    if isinstance(env_summary, Mapping):
        return base.gatemate_setup_changed_from_env(env_summary)
    return False


def openfpgaloader_present(context: Mapping[str, Any]) -> bool:
    tool_versions = context.get("tool_versions", {})
    if not isinstance(tool_versions, Mapping):
        return False
    loader = tool_versions.get("openFPGALoader", {})
    return isinstance(loader, Mapping) and loader.get("present") is True


def gatemate_status_from_context(
    *,
    command_runner: CommandRunner,
    context: Mapping[str, Any],
    physical_path_available: bool,
) -> tuple[str, JsonDict, JsonDict | None, list[JsonDict]]:
    tool_versions = context.get("tool_versions", {})
    usb_visibility = context.get("usb_visibility", {}).get("GateMate", {})
    base_detail = {
        "board": "GateMate",
        "physical_or_jtag_path_available": bool(physical_path_available),
        "dirtyjtag_usb": usb_visibility,
        "tool_versions": tool_versions,
        "speedup_claimed": False,
    }
    if not physical_path_available:
        detail = {
            **base_detail,
            "status": "blocked_physical_or_jtag",
            "jtag_detect_attempted": False,
            "reason": "physical_or_jtag_path_not_available_or_unchanged",
        }
        blocker = {
            "reason": "physical_or_jtag_path_not_available_or_unchanged",
            "prior_evidence": str(PRIOR_RESULT_RELATIVE_PATH),
        }
        return "blocked_physical_or_jtag", detail, blocker, []

    if not openfpgaloader_present(context):
        detail = {
            **base_detail,
            "status": "skipped: openFPGALoader unavailable",
            "jtag_detect_attempted": False,
            "reason": "openFPGALoader unavailable",
        }
        return (
            "skipped: openFPGALoader unavailable",
            detail,
            {"reason": "openFPGALoader unavailable"},
            [],
        )

    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    detected = base.gate_detect_ok(detect_probe)
    status = "detected" if detected else "unreachable"
    detail = {
        **base_detail,
        "status": status,
        "jtag_detect_attempted": True,
        "detect_exit_code": int(detect_probe.exit_code),
        "detect_output_identifier": base.remote_identifier(detect_probe.combined_output)
        if detect_probe.combined_output
        else None,
    }
    blocker = None if detected else blocker_from_probe("detect_failed", detect_probe, GATEMATE_TIMEOUT_S)
    commands = [
        command_receipt(
            probe=detect_probe,
            timeout_s=GATEMATE_TIMEOUT_S,
            kind="gatemate_dirtyjtag_detect",
            outcome=status,
        )
    ]
    return status, detail, blocker, commands


def honest_verdict(
    *,
    kv260_status: str,
    polarfire_status: str,
    gatemate_status: str,
    hardware_speedup_claim: bool,
    repeatability_evidence_present: bool,
) -> str:
    speedup = str(bool(hardware_speedup_claim)).lower()
    repeatability = str(bool(repeatability_evidence_present)).lower()
    return (
        "complete: "
        f"kv260={kv260_status} "
        f"polarfire={polarfire_status} "
        f"gatemate={gatemate_status} "
        f"hardware_speedup_claim={speedup} "
        f"repeatability_evidence_present={repeatability}"
    )


def no_host_mmcblk_kv260_evidence(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return not any(marker in encoded for marker in HOST_BLOCK_DEVICE_MARKERS)


def command_is_destructive(command: str) -> bool:
    lowered = f" {command.lower()} "
    return any(marker in lowered for marker in DESTRUCTIVE_COMMAND_MARKERS)


def _status_is_allowed(status: str) -> bool:
    return bool(status) and (
        status
        in {
            "reachable",
            "unreachable",
            "reachable/workload_receipt",
            "detected",
            "blocked_physical_or_jtag",
        }
        or status.startswith("skipped: ")
    )


def default_tests_run() -> list[JsonDict]:
    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


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
    context, commands_run = collect_preconditions(command_runner)
    gate_path_available = gatemate_path_available_from_context(
        context, gatemate_physical_path_available
    )
    context["operator_visible_hardware_assumptions"]["gatemate_physical_jtag_path_available"] = (
        gate_path_available
    )

    kv260_probe = command_runner(KV260_SSH_TRUE_COMMAND, SSH_TIMEOUT_S)
    kv260_status, kv260_detail, kv260_blocker, kv260_receipt = kv260_status_from_probe(kv260_probe)
    commands_run.append(kv260_receipt)

    polarfire_status, polarfire_hash, polarfire_detail, polarfire_blocker, polarfire_commands = (
        polarfire_status_with_workload(command_runner=command_runner)
    )
    commands_run.extend(polarfire_commands)

    gatemate_status, gatemate_detail, gatemate_blocker, gatemate_commands = (
        gatemate_status_from_context(
            command_runner=command_runner,
            context=context,
            physical_path_available=gate_path_available,
        )
    )
    commands_run.extend(gatemate_commands)

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
        "status": wrap_field("status", "complete"),
        "hardware_speedup_claim": wrap_field("hardware_speedup_claim", False),
        "kv260_checked_via_ssh": wrap_field("kv260_checked_via_ssh", True),
        "kv260_status": wrap_field("kv260_status", kv260_status),
        "polarfire_status": wrap_field("polarfire_status", polarfire_status),
        "polarfire_workload_hash": wrap_field("polarfire_workload_hash", polarfire_hash),
        "gatemate_status": wrap_field("gatemate_status", gatemate_status),
        "commands_run": wrap_field("commands_run", commands_run),
        "no_host_mmcblk_kv260_evidence": wrap_field(
            "no_host_mmcblk_kv260_evidence", True
        ),
        "no_destructive_flash": wrap_field("no_destructive_flash", True),
        "repeatability_evidence_present": wrap_field("repeatability_evidence_present", False),
        "honest_verdict": wrap_field(
            "honest_verdict",
            honest_verdict(
                kv260_status=kv260_status,
                polarfire_status=polarfire_status,
                gatemate_status=gatemate_status,
                hardware_speedup_claim=False,
                repeatability_evidence_present=False,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": wrap_field("preconditions_checked", context),
        "tests_run": wrap_field("tests_run", tests),
        "hardware_evidence_level": HARDWARE_EVIDENCE_LEVEL,
        "board_details": {
            "KV260": kv260_detail,
            "PolarFire": polarfire_detail,
            "GateMate": gatemate_detail,
        },
        "blocked_reason": {
            "KV260": kv260_blocker,
            "PolarFire": polarfire_blocker,
            "GateMate": gatemate_blocker,
        },
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "research-hardware-wishlist.md",
            str(PRIOR_RESULT_RELATIVE_PATH),
            "ops/status.md",
            "openspec/capabilities/fpga/spec.md",
        ],
        "docs_update_decision": {
            "ops_changelog_updated": False,
            "ops_status_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["no_host_mmcblk_kv260_evidence"]["value"] = no_host_mmcblk_kv260_evidence(artifact)
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
            "stdout_excerpt",
            "stderr_excerpt",
            "stdout_sha256",
            "stderr_sha256",
        ):
            require(key in command, f"commands_run[{index}] missing {key}")
        require(
            not command_is_destructive(str(command["command"])),
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
        require(isinstance(item.get("command"), str) and item["command"], "tests_run command missing")
        require(isinstance(item.get("outcome"), str) and item["outcome"], "tests_run outcome missing")


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
    require(validate_wrapped_field(artifact, "status") == "complete", "status must be complete")
    require(
        validate_wrapped_field(artifact, "hardware_speedup_claim") is False,
        "hardware_speedup_claim must be false",
    )
    require(
        validate_wrapped_field(artifact, "kv260_checked_via_ssh") is True,
        "kv260_checked_via_ssh must be true",
    )
    kv260_status = validate_wrapped_field(artifact, "kv260_status")
    polarfire_status = validate_wrapped_field(artifact, "polarfire_status")
    gatemate_status = validate_wrapped_field(artifact, "gatemate_status")
    for label, value in (
        ("kv260_status", kv260_status),
        ("polarfire_status", polarfire_status),
        ("gatemate_status", gatemate_status),
    ):
        require(isinstance(value, str) and _status_is_allowed(value), f"{label} invalid")
    workload_hash = validate_wrapped_field(artifact, "polarfire_workload_hash")
    if polarfire_status == "reachable/workload_receipt":
        require(
            isinstance(workload_hash, str) and len(workload_hash) == 64,
            "polarfire_workload_hash must be a sha256 string",
        )
        require(
            workload_hash == POLARFIRE_EXPECTED_OUTPUT_SHA256,
            "polarfire_workload_hash mismatch",
        )
    else:
        require(workload_hash is None, "polarfire_workload_hash must be null without receipt")
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
    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "bad verdict")
    for token in (
        f"kv260={kv260_status}",
        f"polarfire={polarfire_status}",
        f"gatemate={gatemate_status}",
        "hardware_speedup_claim=false",
    ):
        require(token in verdict, f"honest_verdict missing {token}")
    require(
        validate_wrapped_field(artifact, "inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    validate_commands(validate_wrapped_field(artifact, "commands_run"))
    validate_tests_run(validate_wrapped_field(artifact, "tests_run"))
    require(no_host_mmcblk_kv260_evidence(artifact), "host KV260 block-device evidence present")
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
