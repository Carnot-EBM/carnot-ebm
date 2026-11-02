#!/usr/bin/env python3
"""Exp 5293: hardware continuity reachability receipts.

Spec refs: REQ-HW-5293, SCENARIO-HW-5293.

This module writes a status receipt, not a benchmark. The reason for keeping
this as a small, explicit artifact builder is to make the boundary easy to
audit: a board can be reachable, a USB probe can enumerate, and a tool can print
its version, but none of those facts is acceleration evidence. KV260 and
PolarFire are checked through authenticated SSH commands, while GateMate carries
the existing physical/JTAG block unless the operator explicitly says the setup
changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]
Clock = Callable[[], float]

RUN_DATE = "20260706"
EXPERIMENT_ID = "exp5293-hardware-continuity-reachability-v483"
EXPERIMENT_NAME = "experiment_5293_hardware_continuity_reachability"
MILESTONE = "2026.07.483"
SCHEMA = "carnot.experiment_5293.hardware_continuity_reachability.v483"
SPEC_REFS = ("REQ-HW-5293", "SCENARIO-HW-5293")
RANDOM_SEED = 5293
INFERENCE_SUBSTRATE = "hardware_probe_no_speedup_claim"
HARDWARE_EVIDENCE_LEVEL = "reachability_status_receipt_only"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5293_hardware_continuity_reachability_v483.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5279_hardware_continuity_reachability_v482.json"
)

HARDWARE_ENV_COMMAND = ("env",)
TOOL_NAMES = (
    "ssh",
    "openFPGALoader",
    "yosys",
    "nextpnr-himbaechel",
    "gmpack",
    "lsusb",
)
TOOL_VERSION_COMMAND = (
    "sh",
    "-lc",
    (
        "for tool in ssh openFPGALoader yosys nextpnr-himbaechel gmpack lsusb; do "
        "if command -v \"$tool\" >/dev/null 2>&1; then "
        "printf '%s_path=%s\\n' \"$tool\" \"$(command -v \"$tool\")\"; "
        "version=\"$($tool -V 2>&1 | head -n 1)\"; "
        "if [ -z \"$version\" ]; then version=\"$($tool --version 2>&1 | head -n 1)\"; fi; "
        "printf '%s_version=%s\\n' \"$tool\" \"$version\"; "
        "else printf '%s_path=\\n%s_version=\\n' \"$tool\" \"$tool\"; fi; "
        "done"
    ),
)
GATEMATE_USB_COMMAND = (
    "sh",
    "-lc",
    "if command -v lsusb >/dev/null 2>&1; then lsusb -d 1209:c0ca; "
    "else echo 'lsusb not found' >&2; exit 127; fi",
)
POLARFIRE_USB_COMMAND = (
    "sh",
    "-lc",
    "if command -v lsusb >/dev/null 2>&1; then lsusb -d 1514:2008; "
    "else echo 'lsusb not found' >&2; exit 127; fi",
)
KV260_REACHABILITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "printf 'hostname='; hostname; "
    "printf 'uname='; uname -a; "
    "printf 'xmutil='; (xmutil --version 2>&1 || command -v xmutil || true); "
    "printf 'uio='; (ls /dev/uio* 2>/dev/null | tr '\\n' ' ' || true)",
)
POLARFIRE_REACHABILITY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "printf 'hostname='; hostname; "
    "printf 'uname='; uname -a; "
    "printf 'python='; (python3 --version 2>&1 || true)",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
GATEMATE_EXPECTED_IDCODE = "0x20000001"

LOCAL_TIMEOUT_S = 5.0
SSH_TIMEOUT_S = 10.0
GATEMATE_TIMEOUT_S = 30.0

HARDWARE_ENV_KEYS = (
    "CARNOT_KV260_HOST",
    "CARNOT_KV260_USER",
    "CARNOT_POLARFIRE_HOST",
    "CARNOT_POLARFIRE_USER",
    "CARNOT_GATEMATE_SETUP_CHANGED",
    "GATEMATE_PHYSICAL_SETUP_CHANGED",
    "CARNOT_MODE",
    "CARNOT_FORCE_LIVE",
    "EXTROPIC_API_KEY",
    "EXTROPIC_TOKEN",
    "THRML_HOME",
    "TSU_SDK",
    "XTR0_SDK",
)
HARDWARE_ENV_PREFIXES = ("CARNOT_", "EXTROPIC_", "THRML_", "TSU_", "XTR0_", "GATEMATE_")
TRUTHY_VALUES = {"1", "true", "yes", "y", "on", "changed"}

TERMINAL_PREFIXES = ("complete:", "blocked_")
REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "hardware_evidence_level",
    "kv260_reachability",
    "polarfire_reachability",
    "gatemate_reachability",
    "hardware_speedup_claimed",
    "blocked_reason",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal verdict starts with complete: or blocked_ and states KV260, "
        "PolarFire, GateMate, and no-speedup outcomes."
    ),
    "inference_substrate": (
        "hardware_probe_no_speedup_claim means this receipt records status probes "
        "only, not model inference or acceleration."
    ),
    "hardware_evidence_level": (
        "Reachability and version evidence can support continuity accounting but "
        "cannot support performance claims."
    ),
    "kv260_reachability": (
        "KV260 is checked only by authenticated SSH board commands; no host "
        "storage state is a precondition."
    ),
    "polarfire_reachability": (
        "PolarFire status is derived from authenticated SSH board commands plus "
        "local FlashPro visibility when available."
    ),
    "gatemate_reachability": (
        "GateMate status records USB/tool/JTAG reachability and carries the "
        "physical setup blocker unless the operator changed it."
    ),
    "hardware_speedup_claimed": (
        "False because no reproducible same-workload hardware-vs-baseline timing "
        "run was performed."
    ),
    "blocked_reason": (
        "Per-board blockers preserve exact command failure text or the carried "
        "physical/JTAG reason."
    ),
    "preconditions_checked": (
        "Local environment, tool-version, and USB receipts are recorded before "
        "board classifications to make the probe context auditable."
    ),
    "host_storage_precondition_used": (
        "False because the KV260 path is SSH-only and does not depend on host "
        "removable storage."
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def round_duration(duration_s: float) -> float:
    return round(max(float(duration_s), 0.000001), 6)


@dataclass(frozen=True)
class CommandProbe:
    """A bounded command receipt with stdout and stderr kept separate."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:  # pragma: no cover
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        return CommandProbe(
            tuple(command),
            int(completed.returncode),
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandProbe(
            tuple(command),
            124,
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else f"command timed out after {timeout_s}s",
            time.perf_counter() - started,
        )
    except OSError as exc:
        return CommandProbe(
            tuple(command),
            127,
            "",
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
        )


def get_git_commit(repo_root: str | Path) -> str:  # pragma: no cover
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def wrap_field(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _env_key_allowed(key: str) -> bool:
    return key in HARDWARE_ENV_KEYS or key.startswith(HARDWARE_ENV_PREFIXES)


def _env_truthy(value: str) -> bool:
    return value.strip().lower() in TRUTHY_VALUES


def parse_hardware_environment(probe: CommandProbe) -> JsonDict:
    parsed = {
        key: {"present": False, "truthy": False, "value_recorded": False}
        for key in HARDWARE_ENV_KEYS
    }
    if probe.exit_code != 0:
        return parsed
    for line in probe.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and _env_key_allowed(key):
            parsed.setdefault(key, {"present": False, "truthy": False, "value_recorded": False})
            parsed[key]["present"] = True
            parsed[key]["truthy"] = _env_truthy(value)
    return dict(sorted(parsed.items()))


def sanitized_env_stdout(env_summary: Mapping[str, Mapping[str, Any]]) -> str:
    present = [key for key, value in env_summary.items() if value.get("present") is True]
    return "\n".join(sorted(present))


def parse_tool_versions(probe: CommandProbe) -> JsonDict:
    tools: JsonDict = {
        tool: {"present": False, "path": None, "version": None} for tool in TOOL_NAMES
    }
    for line in probe.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key.endswith("_path"):
            tool = key[: -len("_path")]
            tools.setdefault(tool, {"present": False, "path": None, "version": None})
            tools[tool]["path"] = value.strip() or None
            tools[tool]["present"] = bool(value.strip())
        elif key.endswith("_version"):
            tool = key[: -len("_version")]
            tools.setdefault(tool, {"present": False, "path": None, "version": None})
            tools[tool]["version"] = value.strip() or None
    return tools


def parse_usb_visible(probe: CommandProbe, usb_id: str) -> JsonDict:
    return {
        "device_identifier": usb_id,
        "visible": probe.exit_code == 0 and usb_id in probe.stdout,
        "stdout": probe.stdout.strip(),
        "stderr": probe.stderr.strip(),
    }


def command_receipt(
    *,
    probe: CommandProbe,
    timeout_s: float,
    kind: str,
    outcome: str,
    stdout_override: str | None = None,
) -> JsonDict:
    return {
        "kind": kind,
        "command": command_to_string(probe.command),
        "timeout_s": float(timeout_s),
        "exit_code": int(probe.exit_code),
        "duration_s": round_duration(probe.duration_s),
        "stdout": probe.stdout if stdout_override is None else stdout_override,
        "stderr": probe.stderr,
        "outcome": outcome,
    }


def collect_local_probe_context(command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    env_probe = command_runner(HARDWARE_ENV_COMMAND, LOCAL_TIMEOUT_S)
    tool_probe = command_runner(TOOL_VERSION_COMMAND, LOCAL_TIMEOUT_S)
    gatemate_usb_probe = command_runner(GATEMATE_USB_COMMAND, LOCAL_TIMEOUT_S)
    polarfire_usb_probe = command_runner(POLARFIRE_USB_COMMAND, LOCAL_TIMEOUT_S)

    env_summary = parse_hardware_environment(env_probe)
    tool_versions = parse_tool_versions(tool_probe)
    context = {
        "hardware_environment": env_summary,
        "tool_versions": tool_versions,
        "usb_visibility": {
            "GateMate": parse_usb_visible(gatemate_usb_probe, "1209:c0ca"),
            "PolarFire": parse_usb_visible(polarfire_usb_probe, "1514:2008"),
        },
        "operator_visible_hardware_assumptions": {
            "gatemate_physical_setup_changed_assumed": False,
            "kv260_checked_by_ssh_only": True,
            "internet_required_for_board_probe": False,
        },
    }
    commands = [
        command_receipt(
            probe=env_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="local_hardware_environment",
            outcome="recorded" if env_probe.exit_code == 0 else "env_unavailable",
            stdout_override=sanitized_env_stdout(env_summary),
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
    ]
    return context, commands


def gatemate_setup_changed_from_env(env_summary: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        bool(env_summary.get(key, {}).get("truthy"))
        for key in ("CARNOT_GATEMATE_SETUP_CHANGED", "GATEMATE_PHYSICAL_SETUP_CHANGED")
    )


def blocker_from_probe(reason: str, probe: CommandProbe, timeout_s: float) -> JsonDict:
    return {
        "reason": reason,
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "timeout_s": float(timeout_s),
        "stdout": probe.stdout,
        "stderr": probe.stderr,
    }


def remote_identifier(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "not_recorded"
    return "\n".join(lines[:6])[:800]


def build_ssh_reachability(
    *,
    board_label: str,
    command: tuple[str, ...],
    blocked_reason: str,
    kind: str,
    command_runner: CommandRunner,
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    probe = command_runner(command, SSH_TIMEOUT_S)
    reachable = probe.exit_code == 0
    status = "reachable_ssh_status_only" if reachable else blocked_reason
    commands = [
        command_receipt(
            probe=probe,
            timeout_s=SSH_TIMEOUT_S,
            kind=kind,
            outcome=status,
        )
    ]
    if not reachable:
        return (
            {
                "board": board_label,
                "status": status,
                "ssh_reachable": False,
                "remote_identifier": None,
                "probe_exit_code": probe.exit_code,
                "speedup_claimed": False,
            },
            blocker_from_probe(blocked_reason, probe, SSH_TIMEOUT_S),
            commands,
        )
    return (
        {
            "board": board_label,
            "status": status,
            "ssh_reachable": True,
            "remote_identifier": remote_identifier(probe.combined_output),
            "probe_exit_code": probe.exit_code,
            "speedup_claimed": False,
        },
        None,
        commands,
    )


def gate_detect_ok(probe: CommandProbe) -> bool:
    text = probe.combined_output
    return probe.exit_code == 0 and (
        GATEMATE_EXPECTED_IDCODE in text or "GM1Ax" in text or "GateMate Series" in text
    )


def build_gatemate_reachability(
    *,
    command_runner: CommandRunner,
    setup_changed: bool,
    context: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    tool_versions = context.get("tool_versions", {})
    usb_visibility = context.get("usb_visibility", {}).get("GateMate", {})
    if not setup_changed:
        return (
            {
                "board": "GateMate",
                "status": "blocked_gatemate_physical_jtag_setup_unchanged",
                "physical_setup_changed": False,
                "jtag_probe_attempted": False,
                "device_identifier": "1209:c0ca",
                "dirtyjtag_usb": usb_visibility,
                "tool_versions": tool_versions,
                "speedup_claimed": False,
            },
            {
                "reason": "operator_setup_unchanged_physical_jtag_block_carried_forward",
                "prior_evidence": str(PRIOR_RESULT_RELATIVE_PATH),
            },
            [],
        )

    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    reachable = gate_detect_ok(detect_probe)
    status = (
        "reachable_dirtyjtag_idcode_status_only"
        if reachable
        else "blocked_gatemate_dirtyjtag_status_probe_failed"
    )
    blocker = (
        None
        if reachable
        else blocker_from_probe("blocked_gatemate_dirtyjtag_status_probe_failed", detect_probe, GATEMATE_TIMEOUT_S)
    )
    return (
        {
            "board": "GateMate",
            "status": status,
            "physical_setup_changed": True,
            "jtag_probe_attempted": True,
            "device_identifier": "1209:c0ca",
            "dirtyjtag_usb": usb_visibility,
            "tool_versions": tool_versions,
            "detect_exit_code": detect_probe.exit_code,
            "detect_excerpt": remote_identifier(detect_probe.combined_output),
            "speedup_claimed": False,
        },
        blocker,
        [
            command_receipt(
                probe=detect_probe,
                timeout_s=GATEMATE_TIMEOUT_S,
                kind="gatemate_dirtyjtag_status_probe",
                outcome=status,
            )
        ],
    )


def build_honest_verdict(
    *,
    kv260: Mapping[str, Any],
    polarfire: Mapping[str, Any],
    gatemate: Mapping[str, Any],
    blockers: Mapping[str, JsonDict | None],
) -> str:
    summary = (
        f"kv260={kv260['status']} "
        f"polarfire={polarfire['status']} "
        f"gatemate={gatemate['status']} no_speedup_claim"
    )
    if any(blockers.values()):
        if blockers.get("KV260") or blockers.get("PolarFire"):
            return f"blocked_board_reachability: {summary}"
        return f"blocked_board_status: {summary}"
    return f"complete: {summary}"


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    gatemate_setup_changed: bool | None = None,
) -> JsonDict:
    started = clock()
    context, commands_run = collect_local_probe_context(command_runner)
    env_setup_changed = gatemate_setup_changed_from_env(context["hardware_environment"])
    setup_changed = env_setup_changed if gatemate_setup_changed is None else gatemate_setup_changed
    context["operator_visible_hardware_assumptions"][
        "gatemate_physical_setup_changed_assumed"
    ] = bool(setup_changed)

    kv260, kv260_blocker, kv260_commands = build_ssh_reachability(
        board_label="KV260",
        command=KV260_REACHABILITY_COMMAND,
        blocked_reason="blocked_kv260_ssh_unreachable",
        kind="kv260_ssh_status_probe",
        command_runner=command_runner,
    )
    commands_run.extend(kv260_commands)
    polarfire, polarfire_blocker, polarfire_commands = build_ssh_reachability(
        board_label="PolarFire",
        command=POLARFIRE_REACHABILITY_COMMAND,
        blocked_reason="blocked_polarfire_ssh_unreachable",
        kind="polarfire_ssh_status_probe",
        command_runner=command_runner,
    )
    commands_run.extend(polarfire_commands)
    gatemate, gatemate_blocker, gatemate_commands = build_gatemate_reachability(
        command_runner=command_runner,
        setup_changed=bool(setup_changed),
        context=context,
    )
    commands_run.extend(gatemate_commands)

    blockers = {
        "KV260": kv260_blocker,
        "PolarFire": polarfire_blocker,
        "GateMate": gatemate_blocker,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "duration_s": round_duration(clock() - started),
        "commit": commit,
        "honest_verdict": wrap_field(
            "honest_verdict",
            build_honest_verdict(
                kv260=kv260,
                polarfire=polarfire,
                gatemate=gatemate,
                blockers=blockers,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "hardware_evidence_level": wrap_field(
            "hardware_evidence_level", HARDWARE_EVIDENCE_LEVEL
        ),
        "kv260_reachability": wrap_field("kv260_reachability", kv260),
        "polarfire_reachability": wrap_field("polarfire_reachability", polarfire),
        "gatemate_reachability": wrap_field("gatemate_reachability", gatemate),
        "hardware_speedup_claimed": wrap_field("hardware_speedup_claimed", False),
        "blocked_reason": wrap_field("blocked_reason", blockers),
        "preconditions_checked": wrap_field("preconditions_checked", context),
        "host_storage_precondition_used": wrap_field("host_storage_precondition_used", False),
        "commands_run": commands_run,
        "docs_update_decision": {
            "research_hardware_wishlist_updated": False,
            "ops_status_updated": False,
            "ops_changelog_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md hardware continuity and KV260 SSH discipline",
            "research-hardware-wishlist.md",
            "ops/status.md hardware sections",
            "ops/exclusion_manifest.yaml",
            str(PRIOR_RESULT_RELATIVE_PATH),
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


def validate_commands(commands: Any) -> None:
    require(isinstance(commands, list) and commands, "commands_run must be a non-empty list")
    for index, command in enumerate(commands):
        require(isinstance(command, Mapping), f"commands_run[{index}] must be a mapping")
        for key in ("command", "outcome"):
            require(key in command, f"commands_run[{index}] missing {key}")


def validate_board_reachability(value: Any, board: str) -> None:
    require(isinstance(value, Mapping), f"{board} reachability must be a mapping")
    require(value.get("board") == board, f"{board} board label mismatch")
    require(isinstance(value.get("status"), str) and value["status"], f"{board} status missing")
    require(value.get("speedup_claimed") is False, f"{board} speedup_claimed must be false")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = set(REQUIRED_WRAPPED_FIELDS) - set(artifact)
    require(not missing, f"missing required field: {sorted(missing)}")
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    require(artifact.get("spec_refs") == list(SPEC_REFS), "spec_refs mismatch")
    require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")

    verdict = validate_wrapped_field(artifact, "honest_verdict")
    require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "bad verdict")
    for token in ("kv260=", "polarfire=", "gatemate=", "no_speedup"):
        require(token in verdict, f"honest_verdict missing {token}")
    require(
        validate_wrapped_field(artifact, "inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate mismatch",
    )
    require(
        validate_wrapped_field(artifact, "hardware_evidence_level") == HARDWARE_EVIDENCE_LEVEL,
        "hardware_evidence_level mismatch",
    )
    validate_board_reachability(
        validate_wrapped_field(artifact, "kv260_reachability"), "KV260"
    )
    validate_board_reachability(
        validate_wrapped_field(artifact, "polarfire_reachability"), "PolarFire"
    )
    validate_board_reachability(
        validate_wrapped_field(artifact, "gatemate_reachability"), "GateMate"
    )
    require(
        validate_wrapped_field(artifact, "hardware_speedup_claimed") is False,
        "hardware_speedup_claimed must be false",
    )
    blockers = validate_wrapped_field(artifact, "blocked_reason")
    require(isinstance(blockers, Mapping), "blocked_reason must be a mapping")
    require(set(blockers) == {"KV260", "PolarFire", "GateMate"}, "blocked_reason keys mismatch")
    require(
        validate_wrapped_field(artifact, "host_storage_precondition_used") is False,
        "host_storage_precondition_used must be false",
    )
    preconditions = validate_wrapped_field(artifact, "preconditions_checked")
    require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    for key in (
        "hardware_environment",
        "tool_versions",
        "usb_visibility",
        "operator_visible_hardware_assumptions",
    ):
        require(key in preconditions, f"preconditions_checked missing {key}")
    validate_commands(artifact.get("commands_run"))
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
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str | None = None,
    gatemate_setup_changed: bool | None = None,
) -> Path:
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or get_git_commit(repo_root),
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
