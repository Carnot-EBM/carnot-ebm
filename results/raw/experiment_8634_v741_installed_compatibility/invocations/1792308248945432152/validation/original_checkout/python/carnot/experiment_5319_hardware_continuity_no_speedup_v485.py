#!/usr/bin/env python3
"""Exp 5319: hardware continuity receipts without a speedup claim.

Spec refs: REQ-HW-5319, SCENARIO-HW-5319.

This experiment is a receipt builder, not a benchmark harness. The important
engineering boundary is that board reachability, USB visibility, local GPU
runtime context, and public TSU/Kona references can explain what hardware is
nearby, but none of those facts proves a sampler ran faster. A speedup claim
requires an authenticated workload transcript with matching hashes,
correctness checks, and a measured hardware-vs-baseline timing comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5293_hardware_continuity_reachability_v483 as base
from carnot import experiment_5305_hardware_continuity_reachability_v484 as prev


JsonDict = dict[str, Any]
Clock = Callable[[], float]
CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner

RUN_DATE = "20260706"
EXPERIMENT_ID = "exp5319-hardware-continuity-no-speedup-v485"
EXPERIMENT_NAME = "experiment_5319_hardware_continuity_no_speedup"
MILESTONE = "2026.07.485"
SCHEMA = "carnot.experiment_5319.hardware_continuity_no_speedup.v485"
SPEC_REFS = ("REQ-HW-5319", "SCENARIO-HW-5319")
RANDOM_SEED = 5319
INFERENCE_SUBSTRATE = "hardware_reachability_receipts_no_speedup"
HARDWARE_EVIDENCE_LEVEL = "reachability_context_receipt_only"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5319_hardware_continuity_no_speedup_v485.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5305_hardware_continuity_reachability_v484.json"
)

HOST_DATE_COMMAND = prev.HOST_DATE_COMMAND
HARDWARE_ENV_COMMAND = base.HARDWARE_ENV_COMMAND
TOOL_VERSION_COMMAND = base.TOOL_VERSION_COMMAND
GATEMATE_USB_COMMAND = base.GATEMATE_USB_COMMAND
POLARFIRE_USB_COMMAND = base.POLARFIRE_USB_COMMAND
GPU_CONTEXT_COMMAND = (
    "sh",
    "-lc",
    (
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
        "else echo 'nvidia-smi_not_found' >&2; exit 127; fi"
    ),
)
KV260_SSH_TRUE_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
POLARFIRE_STATUS_COMMAND = base.POLARFIRE_REACHABILITY_COMMAND
GATEMATE_DETECT_COMMAND = base.GATEMATE_DETECT_COMMAND

LOCAL_TIMEOUT_S = base.LOCAL_TIMEOUT_S
SSH_TIMEOUT_S = base.SSH_TIMEOUT_S
GATEMATE_TIMEOUT_S = base.GATEMATE_TIMEOUT_S
TERMINAL_PREFIXES = ("complete:", "blocked_")
EXTRA_CONTEXT_ENV_KEYS = (
    "KONA_API_KEY",
    "LOGICAL_KONA_TOKEN",
    "LOGICAL_API_KEY",
)

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "commands_run",
)
REQUIRED_BARE_BOOLEAN_FIELDS = (
    "kv260_ssh_reachable",
    "polarfire_status_reachable",
    "gatemate_physical_jtag_changed",
    "authenticated_workload_run",
    "no_speedup_claim",
    "public_hardware_references_used_as_context_only",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Identifies the v485 receipt so later reconciler steps can cite one exact artifact.",
    "milestone": "Pins the receipt to milestone 2026.07.485 rather than a floating current state.",
    "status": "Summarizes whether the receipt is complete or blocked without implying acceleration.",
    "honest_verdict": (
        "Terminal verdict starts with complete: or blocked_ and states board, "
        "public-reference, authenticated-workload, and no-speedup boundaries."
    ),
    "inference_substrate": (
        "hardware_reachability_receipts_no_speedup means these are reachability "
        "and context receipts, not sampler hardware acceleration evidence."
    ),
    "preconditions_checked": (
        "Records host/date context, sanitized hardware environment, local tools, "
        "USB visibility, GPU runtime context, public-reference boundaries, and SSH targets."
    ),
    "commands_run": (
        "Every command receipt is exact enough to audit what ran and bounded so "
        "status checks cannot be confused with a workload benchmark."
    ),
    "board_statuses": "Board statuses separate reachability from authenticated workload execution.",
    "blocked_reason": "Per-board blockers preserve exact command failure text or carried physical/JTAG status.",
    "dual_rtx3090_runtime_context": (
        "Dual RTX 3090 visibility is SOTA runtime context only and not sampler acceleration."
    ),
    "public_hardware_context_boundaries": (
        "Extropic/TSU and Logical/Kona are public-reference context only unless local authenticated access exists."
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return base.sha256_json(stable)


def parse_host_date(probe: CommandProbe) -> JsonDict:
    summary: JsonDict = {
        "host": "not_recorded",
        "date_utc": None,
        "date_local": None,
        "exit_code": int(probe.exit_code),
        "stderr": probe.stderr.strip(),
    }
    for line in probe.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"host", "date_utc", "date_local"}:
            summary[key] = value.strip() or None
    return summary


def parse_hardware_environment(probe: CommandProbe) -> JsonDict:
    summary = base.parse_hardware_environment(probe)
    for key in EXTRA_CONTEXT_ENV_KEYS:
        summary.setdefault(key, {"present": False, "truthy": False, "value_recorded": False})
    for line in probe.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in EXTRA_CONTEXT_ENV_KEYS:
            summary[key]["present"] = True
            summary[key]["truthy"] = value.strip().lower() in base.TRUTHY_VALUES
    return dict(sorted(summary.items()))


def parse_gpu_context(probe: CommandProbe) -> JsonDict:
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    rtx3090 = [line for line in lines if "RTX 3090" in line.upper()]
    return {
        "command_exit_code": int(probe.exit_code),
        "nvidia_smi_available": probe.exit_code == 0,
        "detected_gpu_lines": lines[:8],
        "rtx3090_count": len(rtx3090),
        "dual_rtx3090_present": len(rtx3090) >= 2,
        "runtime_relevance": "local SOTA model-runtime context only",
        "sampler_hardware_acceleration_claimed": False,
        "speedup_claimed": False,
    }


def public_reference_boundaries(env_summary: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    extropic_marker = any(
        bool(env_summary.get(key, {}).get("present"))
        for key in ("EXTROPIC_API_KEY", "EXTROPIC_TOKEN", "THRML_HOME", "TSU_SDK", "XTR0_SDK")
    )
    kona_marker = any(
        bool(env_summary.get(key, {}).get("present"))
        for key in EXTRA_CONTEXT_ENV_KEYS
    )
    return {
        "extropic_tsu": {
            "context_only": True,
            "credential_or_sdk_marker_present": extropic_marker,
            "local_authenticated_access": False,
            "execution_claimed": False,
            "speedup_claimed": False,
            "boundary": "public THRML/Extropic TSU/XTR context only; no local authenticated TSU run",
        },
        "logical_kona": {
            "context_only": True,
            "credential_or_sdk_marker_present": kona_marker,
            "local_authenticated_access": False,
            "execution_claimed": False,
            "speedup_claimed": False,
            "boundary": "public Logical/Kona context only; no local authenticated Kona run",
        },
    }


def command_receipt(
    *,
    probe: CommandProbe,
    timeout_s: float,
    kind: str,
    outcome: str,
    stdout_override: str | None = None,
) -> JsonDict:
    return base.command_receipt(
        probe=probe,
        timeout_s=timeout_s,
        kind=kind,
        outcome=outcome,
        stdout_override=stdout_override,
    )


def collect_preconditions(command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    host_probe = command_runner(HOST_DATE_COMMAND, LOCAL_TIMEOUT_S)
    env_probe = command_runner(HARDWARE_ENV_COMMAND, LOCAL_TIMEOUT_S)
    tool_probe = command_runner(TOOL_VERSION_COMMAND, LOCAL_TIMEOUT_S)
    gatemate_usb_probe = command_runner(GATEMATE_USB_COMMAND, LOCAL_TIMEOUT_S)
    polarfire_usb_probe = command_runner(POLARFIRE_USB_COMMAND, LOCAL_TIMEOUT_S)
    gpu_probe = command_runner(GPU_CONTEXT_COMMAND, LOCAL_TIMEOUT_S)

    env_summary = parse_hardware_environment(env_probe)
    gpu_context = parse_gpu_context(gpu_probe)
    context = {
        "host_date": parse_host_date(host_probe),
        "hardware_environment": env_summary,
        "tool_versions": base.parse_tool_versions(tool_probe),
        "usb_visibility": {
            "GateMate": base.parse_usb_visible(gatemate_usb_probe, "1209:c0ca"),
            "PolarFire": base.parse_usb_visible(polarfire_usb_probe, "1514:2008"),
        },
        "gpu_runtime_context": gpu_context,
        "public_reference_boundaries": public_reference_boundaries(env_summary),
        "ssh_targets": {"KV260": "kria", "PolarFire": "polarfire"},
        "kv260_check_method": "ssh_batchmode_true_only",
        "operator_visible_hardware_assumptions": {
            "kv260_checked_by_ssh_only": True,
            "kv260_host_storage_precondition_retired": True,
            "gatemate_physical_jtag_block_carried_forward_unless_setup_changed": True,
            "public_hardware_references_context_only": True,
            "authenticated_workload_run": False,
            "no_speedup_claim": True,
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


def kv260_status_from_probe(probe: CommandProbe) -> tuple[JsonDict, JsonDict | None, JsonDict]:
    reachable = probe.exit_code == 0
    status = "reachable_ssh_status_only" if reachable else "blocked_kv260_ssh_unreachable"
    receipt = command_receipt(
        probe=probe,
        timeout_s=SSH_TIMEOUT_S,
        kind="kv260_ssh_true_reachability_probe",
        outcome=status,
    )
    board_status = {
        "board": "KV260",
        "status": status,
        "ssh_reachable": reachable,
        "check_method": "ssh_batchmode_true_only",
        "remote_identifier": base.remote_identifier(probe.combined_output) if probe.combined_output else None,
        "probe_exit_code": int(probe.exit_code),
        "speedup_claimed": False,
    }
    blocker = None if reachable else base.blocker_from_probe(status, probe, SSH_TIMEOUT_S)
    return board_status, blocker, receipt


def status_value(
    *,
    kv260_reachable: bool,
    polarfire_reachable: bool,
    gatemate_blocked_after_changed_setup: bool,
) -> str:
    if not kv260_reachable:
        return "blocked_kv260_ssh_no_authenticated_workload"
    if not polarfire_reachable:
        return "blocked_polarfire_status_no_authenticated_workload"
    if gatemate_blocked_after_changed_setup:
        return "blocked_gatemate_jtag_no_authenticated_workload"
    return "complete_status_receipts_no_authenticated_workload"


def honest_verdict(
    *,
    status: str,
    kv260_status: Mapping[str, Any],
    polarfire_status: Mapping[str, Any],
    gatemate_status: Mapping[str, Any],
) -> str:
    summary = (
        f"kv260={kv260_status['status']} "
        f"polarfire={polarfire_status['status']} "
        f"gatemate={gatemate_status['status']} "
        "authenticated_workload_run=false "
        "public_refs=context_only "
        "no_speedup_claim"
    )
    if status.startswith("complete_"):
        return f"complete: {summary}"
    return f"{status}: {summary}"


def build_artifact(
    *,
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    gatemate_setup_changed: bool | None = None,
) -> JsonDict:
    started = clock()
    context, commands_run = collect_preconditions(command_runner)
    env_setup_changed = base.gatemate_setup_changed_from_env(context["hardware_environment"])
    setup_changed = env_setup_changed if gatemate_setup_changed is None else gatemate_setup_changed
    context["operator_visible_hardware_assumptions"][
        "gatemate_physical_setup_changed_assumed"
    ] = bool(setup_changed)

    kv260_probe = command_runner(KV260_SSH_TRUE_COMMAND, SSH_TIMEOUT_S)
    kv260, kv260_blocker, kv260_receipt = kv260_status_from_probe(kv260_probe)
    commands_run.append(kv260_receipt)

    polarfire, polarfire_blocker, polarfire_commands = base.build_ssh_reachability(
        board_label="PolarFire",
        command=POLARFIRE_STATUS_COMMAND,
        blocked_reason="blocked_polarfire_ssh_unreachable",
        kind="polarfire_authenticated_status_probe",
        command_runner=command_runner,
    )
    commands_run.extend(polarfire_commands)

    gatemate, gatemate_blocker, gatemate_commands = base.build_gatemate_reachability(
        command_runner=command_runner,
        setup_changed=bool(setup_changed),
        context=context,
    )
    if gatemate_blocker and gatemate_blocker.get("reason") == (
        "operator_setup_unchanged_physical_jtag_block_carried_forward"
    ):
        gatemate_blocker = dict(gatemate_blocker)
        gatemate_blocker["prior_evidence"] = str(PRIOR_RESULT_RELATIVE_PATH)
    commands_run.extend(gatemate_commands)

    gatemate_blocked_after_changed_setup = bool(setup_changed and gatemate_blocker)
    status = status_value(
        kv260_reachable=bool(kv260["ssh_reachable"]),
        polarfire_reachable=bool(polarfire["ssh_reachable"]),
        gatemate_blocked_after_changed_setup=gatemate_blocked_after_changed_setup,
    )
    blockers = {"KV260": kv260_blocker, "PolarFire": polarfire_blocker, "GateMate": gatemate_blocker}
    board_statuses = {"KV260": kv260, "PolarFire": polarfire, "GateMate": gatemate}

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
        "honest_verdict": wrap_field(
            "honest_verdict",
            honest_verdict(
                status=status,
                kv260_status=kv260,
                polarfire_status=polarfire,
                gatemate_status=gatemate,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": wrap_field("preconditions_checked", context),
        "commands_run": wrap_field("commands_run", commands_run),
        "kv260_ssh_reachable": bool(kv260["ssh_reachable"]),
        "polarfire_status_reachable": bool(polarfire["ssh_reachable"]),
        "gatemate_physical_jtag_changed": bool(setup_changed),
        "authenticated_workload_run": False,
        "no_speedup_claim": True,
        "public_hardware_references_used_as_context_only": True,
        "hardware_evidence_level": HARDWARE_EVIDENCE_LEVEL,
        "hardware_speedup_claimed": False,
        "host_storage_precondition_used": False,
        "board_statuses": wrap_field("board_statuses", board_statuses),
        "blocked_reason": wrap_field("blocked_reason", blockers),
        "dual_rtx3090_runtime_context": wrap_field(
            "dual_rtx3090_runtime_context", context["gpu_runtime_context"]
        ),
        "public_hardware_context_boundaries": wrap_field(
            "public_hardware_context_boundaries", context["public_reference_boundaries"]
        ),
        "docs_update_decision": {
            "research_hardware_wishlist_updated": False,
            "ops_status_updated": False,
            "ops_changelog_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "research-hardware-wishlist.md",
            "ops/changelog.md",
            "ops/status.md",
            str(PRIOR_RESULT_RELATIVE_PATH),
            "results/experiment_5293_hardware_continuity_reachability_v483.json",
        ],
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def no_host_storage_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return "mmcblk" not in encoded and "/dev/disk" not in encoded


def validate_wrapped_field(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
    require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def validate_bare_booleans(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_BARE_BOOLEAN_FIELDS:
        require(isinstance(artifact.get(field), bool), f"{field} must be a bare boolean")
    require(artifact["authenticated_workload_run"] is False, "authenticated_workload_run must be false")
    require(artifact["no_speedup_claim"] is True, "no_speedup_claim must stay true")
    require(
        artifact["public_hardware_references_used_as_context_only"] is True,
        "public hardware references must stay context-only",
    )


def validate_commands(commands: Any) -> None:
    require(isinstance(commands, list) and commands, "commands_run must be a non-empty list")
    for index, command in enumerate(commands):
        require(isinstance(command, Mapping), f"commands_run[{index}] must be a mapping")
        for key in ("command", "outcome", "exit_code", "timeout_s", "duration_s"):
            require(key in command, f"commands_run[{index}] missing {key}")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_WRAPPED_FIELDS:
        validate_wrapped_field(artifact, field)
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(validate_wrapped_field(artifact, "experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    require(validate_wrapped_field(artifact, "milestone") == MILESTONE, "milestone mismatch")
    require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs mismatch")
    require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")

    status = validate_wrapped_field(artifact, "status")
    require(isinstance(status, str) and status, "status missing")
    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "bad verdict")
    for token in (
        "kv260=",
        "polarfire=",
        "gatemate=",
        "authenticated_workload_run=false",
        "public_refs=context_only",
        "no_speedup",
    ):
        require(token in verdict, f"honest_verdict missing {token}")
    require(
        validate_wrapped_field(artifact, "inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    preconditions = validate_wrapped_field(artifact, "preconditions_checked")
    require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    for key in (
        "host_date",
        "hardware_environment",
        "tool_versions",
        "usb_visibility",
        "gpu_runtime_context",
        "public_reference_boundaries",
        "kv260_check_method",
        "operator_visible_hardware_assumptions",
    ):
        require(key in preconditions, f"preconditions_checked missing {key}")
    require(preconditions["kv260_check_method"] == "ssh_batchmode_true_only", "KV260 method mismatch")
    gpu_context = preconditions["gpu_runtime_context"]
    require(gpu_context["sampler_hardware_acceleration_claimed"] is False, "GPU sampler claim drift")
    boundaries = preconditions["public_reference_boundaries"]
    require(boundaries["extropic_tsu"]["context_only"] is True, "Extropic boundary drift")
    require(boundaries["logical_kona"]["context_only"] is True, "Kona boundary drift")
    validate_commands(validate_wrapped_field(artifact, "commands_run"))
    validate_bare_booleans(artifact)
    require(artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed must be false")
    require(artifact.get("host_storage_precondition_used") is False, "host storage precondition drift")
    require(no_host_storage_markers(artifact), "host storage marker present")
    require(artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum mismatch",
    )


def write_artifact(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / RESULT_RELATIVE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str | None = None,
    gatemate_setup_changed: bool | None = None,
) -> Path:
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or base.get_git_commit(repo_root),
        gatemate_setup_changed=gatemate_setup_changed,
    )
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--gatemate-setup-changed",
        action="store_true",
        help="Run the bounded GateMate status probe because the operator changed the setup.",
    )
    args = parser.parse_args(argv)
    print(
        run_experiment(
            repo_root=Path("."),
            run_date=args.date,
            gatemate_setup_changed=args.gatemate_setup_changed,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
