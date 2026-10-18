#!/usr/bin/env python3
"""Exp 5411: safe hardware repeatability restoration receipt.

Spec refs: REQ-HW-5411, SCENARIO-HW-5411.

This module records hardware evidence only when the current bench can produce
it safely. KV260 is deliberately limited to the required SSH true precondition,
because any broader probe would blur board reachability with unrelated host
state. PolarFire gets repeated same-workload hash receipts when SSH is
reachable. GateMate uses only non-destructive DirtyJTAG diagnostics. The result
can restore repeatability evidence, but it still cannot support a speedup claim
without a valid same-workload baseline.
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
from carnot import experiment_5374_hardware_continuity_receipts_v489 as receipts
from carnot import experiment_5398_hardware_evidence_graph_repeatability_v491 as prev


JsonDict = dict[str, Any]
Clock = prev.Clock
CommandProbe = prev.CommandProbe
CommandRunner = prev.CommandRunner

RUN_DATE = "20260708"
EXPERIMENT_ID = "exp5411-hardware-repeatability-restoration-v492"
EXPERIMENT_NAME = "experiment_5411_hardware_repeatability_restoration"
MILESTONE = "2026.07.492"
SCHEMA = "carnot.experiment_5411.hardware_repeatability_restoration.v492"
SPEC_REFS = ("REQ-HW-5411", "SCENARIO-HW-5411")
RANDOM_SEED = 5411
INFERENCE_SUBSTRATE = "hardware_smoke"
HARDWARE_EVIDENCE_LEVEL = "safe_board_repeatability_restoration_no_speedup"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5411_hardware_repeatability_restoration_v492.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5398_hardware_evidence_graph_repeatability_v491.json"
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
POLARFIRE_REPEAT_TARGET = 3
TERMINAL_PREFIXES = ("complete:", "blocked:")
BOARDS_CHECKED = ("KV260", "PolarFire", "GateMate")

POLARFIRE_WORKLOAD_INPUT = b"carnot-exp5411-polarfire-repeatability-v492\n"
POLARFIRE_WORKLOAD_OUTPUT_SUFFIX = b"|polarfire-v492-output"
POLARFIRE_EXPECTED_INPUT_SHA256 = hashlib.sha256(POLARFIRE_WORKLOAD_INPUT).hexdigest()
POLARFIRE_EXPECTED_OUTPUT_SHA256 = hashlib.sha256(
    POLARFIRE_WORKLOAD_INPUT + POLARFIRE_WORKLOAD_OUTPUT_SUFFIX
).hexdigest()
POLARFIRE_WORKLOAD_PYTHON = (
    "import hashlib,json,platform,socket,time;"
    "started=time.perf_counter();"
    'payload=b"carnot-exp5411-polarfire-repeatability-v492\\n";'
    'out=hashlib.sha256(payload+b"|polarfire-v492-output").hexdigest();'
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

REQUIRED_RAW_FIELDS = (
    "preconditions_checked",
    "kv260_ssh_reachable",
    "kv260_host_sd_probe_used",
    "polarfire_reachable",
    "polarfire_repeat_count",
    "polarfire_repeat_hashes",
    "gatemate_reachable",
    "gatemate_destructive_probe_used",
    "repeated_same_workload_ready",
    "hardware_speedup_claim",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "True only after local date, environment, tools, USB, and SSH target gates were attempted.",
    "kv260_ssh_reachable": "True only from the required BatchMode SSH true command.",
    "kv260_host_sd_probe_used": "False because KV260 evidence must not depend on host removable storage.",
    "polarfire_reachable": "True only from authenticated PolarFire SSH status.",
    "polarfire_repeat_count": "Counts safe same-workload PolarFire repeats actually attempted after SSH reachability.",
    "polarfire_repeat_hashes": "Records per-repeat board-local output hashes so agreement can be audited.",
    "gatemate_reachable": "True only when non-destructive DirtyJTAG detection reports GateMate identity.",
    "gatemate_destructive_probe_used": "False because GateMate is never flashed, programmed, or written in this receipt.",
    "repeated_same_workload_ready": "True only after at least three valid same-workload repeats agree on the expected hash.",
    "hardware_speedup_claim": "False because this receipt has no valid baseline timing comparison.",
    "inference_substrate": "hardware_smoke because evidence comes from bounded board probes and tiny board workloads.",
    "honest_verdict": "Terminal summary begins with complete: or blocked: and states repeatability and speedup boundaries.",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_text(text: str) -> str:
    return receipts.sha256_text(text)


def sha256_json(payload: Mapping[str, Any]) -> str:
    return base.sha256_json(payload)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def receipt_timestamp(run_date: str, index: int) -> str:
    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run_date must be YYYYMMDD")
    year, month, day = run_date[:4], run_date[4:6], run_date[6:8]
    minute_total, second = divmod(int(index), 60)
    hour, minute = divmod(minute_total, 60)
    return f"{year}-{month}-{day}T{hour:02d}:{minute:02d}:{second:02d}Z"


def timing_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round(sum((value - mean) ** 2 for value in values) / len(values), 12)


def command_board(kind: str) -> str:
    if kind.startswith("kv260"):
        return "KV260"
    if kind.startswith("polarfire"):
        return "PolarFire"
    if kind.startswith("gatemate"):
        return "GateMate"
    return "Host"


def command_path(command_text: str) -> str:
    stripped = command_text.strip()
    return stripped.split(maxsplit=1)[0] if stripped else ""


def enrich_command_receipt(
    command: Mapping[str, Any], *, run_date: str, index: int
) -> JsonDict:
    enriched = dict(command)
    kind = str(enriched.get("kind", ""))
    command_text = str(enriched.get("command", ""))
    enriched["board"] = command_board(kind)
    enriched["timestamp_utc"] = receipt_timestamp(run_date, index)
    enriched["command_path"] = command_path(command_text)
    enriched["command_sha256"] = sha256_text(command_text)
    return enriched


def command_receipt(
    *,
    probe: CommandProbe,
    timeout_s: float,
    kind: str,
    outcome: str,
    run_date: str,
    index: int,
    command_override: str | None = None,
) -> JsonDict:
    receipt = receipts.command_receipt(
        probe=probe,
        timeout_s=timeout_s,
        kind=kind,
        outcome=outcome,
        command_override=command_override,
    )
    return enrich_command_receipt(receipt, run_date=run_date, index=index)


def collect_preconditions(
    command_runner: CommandRunner, *, run_date: str
) -> tuple[JsonDict, list[JsonDict]]:
    context, commands = receipts.collect_preconditions(command_runner)
    context = dict(context)
    context["prior_receipt"] = str(PRIOR_RESULT_RELATIVE_PATH)
    context["ssh_targets"] = {"KV260": "kria", "PolarFire": "polarfire"}
    context["safe_gate"] = {
        "kv260_ssh_only": True,
        "gatemate_non_destructive_only": True,
        "hardware_speedup_claim": False,
    }
    return (
        context,
        [
            enrich_command_receipt(command, run_date=run_date, index=index)
            for index, command in enumerate(commands)
        ],
    )


def blocker_from_probe(
    reason: str,
    probe: CommandProbe,
    timeout_s: float,
    *,
    command_override: str | None = None,
) -> JsonDict:
    return receipts.blocker_from_probe(
        reason,
        probe,
        timeout_s,
        command_override=command_override,
    )


def kv260_reachability(
    *, command_runner: CommandRunner, run_date: str, command_index: int
) -> tuple[JsonDict, JsonDict | None, JsonDict]:
    probe = command_runner(KV260_SSH_TRUE_COMMAND, SSH_TIMEOUT_S)
    reachable = probe.exit_code == 0
    status = "reachable" if reachable else "unreachable"
    detail: JsonDict = {
        "board": "KV260",
        "status": status,
        "ssh_reachable": reachable,
        "check_method": "ssh_batchmode_true_only",
        "command_form": KV260_REQUIRED_COMMAND_FORM,
        "probe_exit_code": int(probe.exit_code),
        "board_identity": (
            "kria ssh target reachable; remote identity intentionally not queried"
            if reachable
            else None
        ),
        "command_paths": ["ssh"],
        "stdout_sha256": sha256_text(probe.stdout),
        "stderr_sha256": sha256_text(probe.stderr),
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
    command = command_receipt(
        probe=probe,
        timeout_s=SSH_TIMEOUT_S,
        kind="kv260_ssh_true_reachability_probe",
        outcome=status,
        run_date=run_date,
        index=command_index,
        command_override=KV260_REQUIRED_COMMAND_FORM,
    )
    return detail, blocker, command


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
    if receipt.get("python_version") is not None and not isinstance(
        receipt.get("python_version"), str
    ):
        errors.append("python_version invalid")
    return receipt, "; ".join(errors) if errors else None


def polarfire_repeatability(
    *, command_runner: CommandRunner, run_date: str, command_index: int
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    status_probe = command_runner(POLARFIRE_STATUS_COMMAND, SSH_TIMEOUT_S)
    reachable = status_probe.exit_code == 0
    commands = [
        command_receipt(
            probe=status_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_authenticated_status_probe",
            outcome="reachable" if reachable else "unreachable",
            run_date=run_date,
            index=command_index,
        )
    ]
    detail: JsonDict = {
        "board": "PolarFire",
        "status": "reachable" if reachable else "unreachable",
        "ssh_reachable": reachable,
        "probe_exit_code": int(status_probe.exit_code),
        "board_identity": base.remote_identifier(status_probe.combined_output)
        if reachable
        else None,
        "command_paths": ["ssh"],
        "repeat_target": POLARFIRE_REPEAT_TARGET,
        "repeat_count": 0,
        "valid_repeat_count": 0,
        "repeat_hashes": [],
        "repeat_timings_s": [],
        "repeat_timing_variance": None,
        "workload_attempts": [],
        "repeatability_class": "blocked_polarfire_ssh_unreachable",
        "speedup_claimed": False,
    }
    if not reachable:
        return detail, blocker_from_probe("unreachable", status_probe, SSH_TIMEOUT_S), commands

    attempts: list[JsonDict] = []
    blocker: JsonDict | None = None
    for repeat_index in range(POLARFIRE_REPEAT_TARGET):
        probe = command_runner(POLARFIRE_WORKLOAD_COMMAND, SSH_TIMEOUT_S)
        receipt, parse_error = parse_polarfire_workload_stdout(probe.stdout)
        valid = probe.exit_code == 0 and receipt is not None and parse_error is None
        output_hash = receipt.get("output_sha256") if isinstance(receipt, Mapping) else None
        timing = receipt.get("wall_time_s") if isinstance(receipt, Mapping) else None
        attempts.append(
            {
                "index": repeat_index + 1,
                "exit_code": int(probe.exit_code),
                "valid": valid,
                "parse_error": parse_error,
                "receipt": receipt,
                "wall_time_s": timing,
                "input_sha256": receipt.get("input_sha256")
                if isinstance(receipt, Mapping)
                else None,
                "output_sha256": output_hash,
            }
        )
        if not valid and blocker is None:
            blocker = blocker_from_probe(
                parse_error or probe.stderr.strip() or "workload command failed",
                probe,
                SSH_TIMEOUT_S,
            )
        commands.append(
            command_receipt(
                probe=probe,
                timeout_s=SSH_TIMEOUT_S,
                kind=f"polarfire_board_local_workload_repeat_{repeat_index + 1}",
                outcome="valid_repeat" if valid else "invalid_repeat",
                run_date=run_date,
                index=command_index + repeat_index + 1,
            )
        )

    output_hashes = [
        str(attempt["output_sha256"])
        for attempt in attempts
        if isinstance(attempt.get("output_sha256"), str)
    ]
    timings = [
        float(attempt["wall_time_s"])
        for attempt in attempts
        if isinstance(attempt.get("wall_time_s"), int | float)
    ]
    valid_attempts = [attempt for attempt in attempts if attempt["valid"] is True]
    stable_outputs = output_hashes == [POLARFIRE_EXPECTED_OUTPUT_SHA256] * POLARFIRE_REPEAT_TARGET
    repeatability_ready = len(valid_attempts) >= POLARFIRE_REPEAT_TARGET and stable_outputs
    if repeatability_ready:
        repeatability_class = "repeatable_board_local_same_output_timing"
    elif len(output_hashes) >= 2 and len(set(output_hashes)) > 1:
        repeatability_class = "non_reproducible_output_hash_drift"
    else:
        repeatability_class = "insufficient_valid_board_local_repeats"
    detail.update(
        {
            "status": "reachable/repeated_workload",
            "repeat_count": len(attempts),
            "valid_repeat_count": len(valid_attempts),
            "repeat_hashes": output_hashes,
            "repeat_timings_s": timings,
            "repeat_timing_variance": timing_variance(timings),
            "workload_attempts": attempts,
            "workload_validated": repeatability_ready,
            "repeatability_class": repeatability_class,
        }
    )
    return detail, blocker, commands


def _mapping_value(root: Mapping[str, Any], *keys: str) -> Any:
    current: Any = root
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def gatemate_status(
    *, command_runner: CommandRunner, context: Mapping[str, Any], run_date: str, command_index: int
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    tool_present = receipts.openfpgaloader_present(context)
    usb_visible = _mapping_value(context, "usb_visibility", "GateMate", "visible") is True
    base_detail: JsonDict = {
        "board": "GateMate",
        "dirtyjtag_usb_visible": usb_visible,
        "openfpgaloader_present": tool_present,
        "command_paths": [],
        "destructive_probe_used": False,
        "speedup_claimed": False,
    }
    if not tool_present:
        detail = {
            **base_detail,
            "status": "unreachable",
            "reachable": False,
            "board_identity": None,
            "reason": "openFPGALoader unavailable",
        }
        return detail, {"reason": "openFPGALoader unavailable"}, []
    if not usb_visible:
        detail = {
            **base_detail,
            "status": "unreachable",
            "reachable": False,
            "board_identity": None,
            "reason": "dirtyjtag usb not visible",
        }
        return detail, {"reason": "dirtyjtag usb not visible"}, []

    probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    detected = base.gate_detect_ok(probe)
    status = "detected" if detected else "unreachable"
    detail = {
        **base_detail,
        "status": status,
        "reachable": detected,
        "board_identity": base.remote_identifier(probe.combined_output)
        if probe.combined_output
        else None,
        "probe_exit_code": int(probe.exit_code),
        "command_paths": ["openFPGALoader"],
        "stdout_sha256": sha256_text(probe.stdout),
        "stderr_sha256": sha256_text(probe.stderr),
    }
    blocker = None if detected else blocker_from_probe("detect_failed", probe, GATEMATE_TIMEOUT_S)
    command = command_receipt(
        probe=probe,
        timeout_s=GATEMATE_TIMEOUT_S,
        kind="gatemate_dirtyjtag_detect",
        outcome=status,
        run_date=run_date,
        index=command_index,
    )
    return detail, blocker, [command]


def repeated_same_workload_ready(polarfire_detail: Mapping[str, Any]) -> bool:
    return (
        polarfire_detail.get("workload_validated") is True
        and polarfire_detail.get("repeat_count") == POLARFIRE_REPEAT_TARGET
        and polarfire_detail.get("repeat_hashes")
        == [POLARFIRE_EXPECTED_OUTPUT_SHA256] * POLARFIRE_REPEAT_TARGET
    )


def honest_verdict(
    *,
    repeated_ready: bool,
    kv260_reachable: bool,
    polarfire_reachable: bool,
    polarfire_repeat_count: int,
    gatemate_reachable: bool,
) -> str:
    prefix = "complete:" if repeated_ready else "blocked:"
    return (
        f"{prefix} "
        f"kv260_ssh_reachable={str(kv260_reachable).lower()} "
        f"polarfire_reachable={str(polarfire_reachable).lower()} "
        f"polarfire_repeat_count={polarfire_repeat_count} "
        f"gatemate_reachable={str(gatemate_reachable).lower()} "
        f"repeated_same_workload_ready={str(repeated_ready).lower()} "
        "hardware_speedup_claim=false"
    )


def default_tests_run() -> list[JsonDict]:
    return [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]


def validate_tests_run(tests_run: Any) -> None:
    require(isinstance(tests_run, list) and tests_run, "tests_run must be non-empty")
    for index, item in enumerate(tests_run):
        require(isinstance(item, Mapping), f"tests_run[{index}] must be mapping")
        require(isinstance(item.get("command"), str) and item["command"], "test command missing")
        require(isinstance(item.get("outcome"), str) and item["outcome"], "test outcome missing")


def build_artifact(
    *,
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = clock()
    context, commands_run = collect_preconditions(command_runner, run_date=run_date)

    kv260_detail, kv260_blocker, kv260_command = kv260_reachability(
        command_runner=command_runner,
        run_date=run_date,
        command_index=len(commands_run),
    )
    commands_run.append(kv260_command)

    polarfire_detail, polarfire_blocker, polarfire_commands = polarfire_repeatability(
        command_runner=command_runner,
        run_date=run_date,
        command_index=len(commands_run),
    )
    commands_run.extend(polarfire_commands)

    gatemate_detail, gatemate_blocker, gatemate_commands = gatemate_status(
        command_runner=command_runner,
        context=context,
        run_date=run_date,
        command_index=len(commands_run),
    )
    commands_run.extend(gatemate_commands)

    repeated_ready = repeated_same_workload_ready(polarfire_detail)
    tests = [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "milestone": MILESTONE,
        "duration_s": base.round_duration(clock() - started),
        "commit": commit,
        "preconditions_checked": True,
        "kv260_ssh_reachable": kv260_detail["ssh_reachable"],
        "kv260_host_sd_probe_used": False,
        "polarfire_reachable": polarfire_detail["ssh_reachable"],
        "polarfire_repeat_count": int(polarfire_detail.get("repeat_count", 0)),
        "polarfire_repeat_hashes": list(polarfire_detail.get("repeat_hashes", [])),
        "gatemate_reachable": gatemate_detail.get("reachable") is True,
        "gatemate_destructive_probe_used": False,
        "repeated_same_workload_ready": repeated_ready,
        "hardware_speedup_claim": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(
            repeated_ready=repeated_ready,
            kv260_reachable=kv260_detail["ssh_reachable"],
            polarfire_reachable=polarfire_detail["ssh_reachable"],
            polarfire_repeat_count=int(polarfire_detail.get("repeat_count", 0)),
            gatemate_reachable=gatemate_detail.get("reachable") is True,
        ),
        "hardware_evidence_level": HARDWARE_EVIDENCE_LEVEL,
        "boards_checked": list(BOARDS_CHECKED),
        "artifact_field_principles": dict(FIELD_PRINCIPLES),
        "precondition_context": context,
        "commands_run": commands_run,
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
        "tests_run": tests,
        "docs_update_decision": {
            "ops_changelog_updated": False,
            "ops_status_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _artifact_mentions_host_storage(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return any(marker in encoded for marker in receipts.HOST_BLOCK_DEVICE_MARKERS)


def _validate_command_receipts(commands: Any) -> None:
    require(isinstance(commands, list) and commands, "commands_run must be non-empty")
    kv260_commands = 0
    for index, command in enumerate(commands):
        require(isinstance(command, Mapping), f"commands_run[{index}] must be mapping")
        command_text = command.get("command")
        require(isinstance(command_text, str) and command_text, "command text missing")
        if command.get("board") == "KV260":
            kv260_commands += 1
            require(
                command_text == KV260_REQUIRED_COMMAND_FORM,
                "KV260 command must be exact SSH true precondition",
            )
        require(
            not receipts.command_is_destructive(command_text),
            f"destructive command recorded: {command_text}",
        )
        require(command.get("command_sha256") == sha256_text(command_text), "command hash mismatch")
        require(isinstance(command.get("timestamp_utc"), str), "timestamp missing")
        require(isinstance(command.get("command_path"), str), "command path missing")
    require(kv260_commands == 1, "exactly one KV260 reachability command required")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_RAW_FIELDS:
        require(field in artifact, f"{field} missing")
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs mismatch")
    require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    require(artifact.get("milestone") == MILESTONE, "milestone mismatch")
    require(artifact.get("preconditions_checked") is True, "preconditions_checked must be true")
    require(
        isinstance(artifact.get("kv260_ssh_reachable"), bool),
        "kv260_ssh_reachable must be bool",
    )
    require(artifact.get("kv260_host_sd_probe_used") is False, "kv260_host_sd_probe_used")
    require(isinstance(artifact.get("polarfire_reachable"), bool), "bad polarfire status")
    repeat_count = artifact.get("polarfire_repeat_count")
    require(isinstance(repeat_count, int) and repeat_count >= 0, "bad polarfire repeat count")
    repeat_hashes = artifact.get("polarfire_repeat_hashes")
    require(isinstance(repeat_hashes, list), "polarfire_repeat_hashes must be list")
    for index, item in enumerate(repeat_hashes):
        require(isinstance(item, str) and len(item) == 64, f"bad repeat hash {index}")
    require(isinstance(artifact.get("gatemate_reachable"), bool), "bad GateMate status")
    require(
        artifact.get("gatemate_destructive_probe_used") is False,
        "gatemate_destructive_probe_used",
    )
    require(isinstance(artifact.get("repeated_same_workload_ready"), bool), "bad repeat ready")
    require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    verdict = artifact.get("honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "bad verdict")
    require("hardware_speedup_claim=false" in verdict, "verdict missing speedup boundary")
    if artifact.get("repeated_same_workload_ready") is True:
        require(repeat_count >= POLARFIRE_REPEAT_TARGET, "ready repeat count too low")
        require(
            repeat_hashes == [POLARFIRE_EXPECTED_OUTPUT_SHA256] * POLARFIRE_REPEAT_TARGET,
            "ready repeat hashes do not agree",
        )
    _validate_command_receipts(artifact.get("commands_run"))
    require(not _artifact_mentions_host_storage(artifact), "host storage evidence present")
    validate_tests_run(artifact.get("tests_run"))
    require(artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum mismatch",
    )


def write_output(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_artifact(artifact)
    root = Path(repo_root)
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or base.get_git_commit(repo_root),
        tests_run=tests_run,
    )
    return write_output(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    print(run_experiment(repo_root=Path("."), run_date=args.date))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
