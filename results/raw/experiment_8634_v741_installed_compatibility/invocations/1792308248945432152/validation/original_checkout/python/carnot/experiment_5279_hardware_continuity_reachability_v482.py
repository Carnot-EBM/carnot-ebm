#!/usr/bin/env python3
"""Exp 5279: hardware continuity reachability receipts.

Spec refs: REQ-HW-5279, SCENARIO-HW-5279.

This module writes a continuity artifact, not a benchmark. The point is to make
the current board state auditable without turning "I can or cannot reach this
board" into a speedup claim. KV260 is checked only over SSH, PolarFire records
SSH reachability plus whether a terminal workload is visible, and GateMate keeps
the physical/JTAG status separate from toolchain visibility unless the operator
explicitly says the physical setup changed.
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

RUN_DATE = "20260705"
EXPERIMENT_ID = "exp5279-hardware-continuity-reachability-v482"
EXPERIMENT_NAME = "experiment_5279_hardware_continuity_reachability"
MILESTONE = "2026.07.482"
SCHEMA = "carnot.experiment_5279.hardware_continuity_reachability.v482"
SPEC_REFS = ("REQ-HW-5279", "SCENARIO-HW-5279")
RANDOM_SEED = 5279
INFERENCE_SUBSTRATE = "hardware_probe_no_speedup_claim"
HARDWARE_EVIDENCE_LEVEL = "reachability_status_receipt_only"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5279_hardware_continuity_reachability_v482.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5266_hardware_thermodynamic_schedule_boundary_v481.json"
)

SSH_CONFIG_KV260_COMMAND = ("ssh", "-G", "kria")
SSH_CONFIG_POLARFIRE_COMMAND = ("ssh", "-G", "polarfire")
HOST_RESOLUTION_KV260_COMMAND = ("getent", "hosts", "kria")
HOST_RESOLUTION_POLARFIRE_COMMAND = ("getent", "hosts", "polarfire")
HARDWARE_ENV_COMMAND = ("env",)
TOOLCHAIN_PRESENCE_COMMAND = (
    "sh",
    "-lc",
    (
        "for tool in ssh scp openFPGALoader yosys nextpnr-himbaechel gmpack vivado "
        "lsusb; do if command -v \"$tool\" >/dev/null 2>&1; then "
        "printf '%s=%s\\n' \"$tool\" \"$(command -v \"$tool\")\"; "
        "else printf '%s=\\n' \"$tool\"; fi; done"
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

KV260_SSH_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "kria", "true")
KV260_BOARD_SUMMARY_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "hostname; uname -a; (xmutil listapps 2>/dev/null || true); "
    "(ls /dev/uio* 2>/dev/null || true)",
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
POLARFIRE_TERMINAL_WORKLOAD_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "if [ -x ./carnot_terminal_workload.sh ] || "
    "[ -x /tmp/carnot_terminal_workload.sh ] || "
    "[ -f ./carnot_terminal_workload.py ] || "
    "[ -f /tmp/carnot_terminal_workload.py ]; then "
    "echo terminal_workload_exists=true; else echo terminal_workload_exists=false; fi",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
GATEMATE_EXPECTED_IDCODE = "0x20000001"

STEP0_TIMEOUT_S = 5.0
SSH_TIMEOUT_S = 10.0
BOARD_DETAIL_TIMEOUT_S = 15.0
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
    "per_board_status",
    "host_sd_card_precondition_used",
    "hardware_speedup_claimed",
    "blocked_reason",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal verdict starts with complete: or blocked_ and summarizes each "
        "board plus no-speedup discipline."
    ),
    "inference_substrate": (
        "hardware_probe_no_speedup_claim means this artifact records status "
        "receipts only, not model inference or acceleration."
    ),
    "hardware_evidence_level": (
        "Reachability and status evidence can keep boards visible, but cannot "
        "support performance claims."
    ),
    "per_board_status": (
        "Each board is reported independently so one blocker cannot be hidden "
        "inside a whole-task summary."
    ),
    "host_sd_card_precondition_used": (
        "False because KV260 continuity uses SSH and board-level checks, not "
        "host removable-storage state."
    ),
    "hardware_speedup_claimed": (
        "False because no same-run matched workload benchmark was performed."
    ),
    "blocked_reason": (
        "Per-board blockers preserve exact command failure or the carried "
        "physical/JTAG reason for follow-up."
    ),
    "preconditions_checked": (
        "Step 0 records SSH targets, environment indicators, toolchain presence, "
        "USB visibility, and operator-visible physical assumptions before board "
        "classification."
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


def parse_ssh_config(probe: CommandProbe) -> JsonDict:
    allowed = {"host", "hostname", "user", "port", "proxyjump", "proxycommand"}
    parsed: JsonDict = {"available": probe.exit_code == 0, "selected": {}}
    if probe.exit_code != 0:
        parsed["error"] = probe.combined_output.strip()
        return parsed
    for line in probe.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key.lower() in allowed and value.strip():
            parsed["selected"][key.lower()] = value.strip()
    return parsed


def sanitized_ssh_config_stdout(config: Mapping[str, Any]) -> str:
    selected = config.get("selected")
    if not isinstance(selected, Mapping):
        return ""
    return "\n".join(f"{key} {selected[key]}" for key in sorted(selected))


def parse_toolchain_presence(probe: CommandProbe) -> JsonDict:
    tools: JsonDict = {}
    for line in probe.stdout.splitlines():
        tool, separator, path = line.partition("=")
        if separator and tool:
            tools[tool] = {"present": bool(path.strip()), "path": path.strip() or None}
    return tools


def parse_usb_visible(probe: CommandProbe, usb_id: str) -> JsonDict:
    return {
        "usb_id": usb_id,
        "visible": probe.exit_code == 0 and bool(probe.stdout.strip()),
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


def collect_step0_preconditions(command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    kv260_config_probe = command_runner(SSH_CONFIG_KV260_COMMAND, STEP0_TIMEOUT_S)
    polarfire_config_probe = command_runner(SSH_CONFIG_POLARFIRE_COMMAND, STEP0_TIMEOUT_S)
    kv260_resolution_probe = command_runner(HOST_RESOLUTION_KV260_COMMAND, STEP0_TIMEOUT_S)
    polarfire_resolution_probe = command_runner(
        HOST_RESOLUTION_POLARFIRE_COMMAND, STEP0_TIMEOUT_S
    )
    env_probe = command_runner(HARDWARE_ENV_COMMAND, STEP0_TIMEOUT_S)
    toolchain_probe = command_runner(TOOLCHAIN_PRESENCE_COMMAND, STEP0_TIMEOUT_S)
    gatemate_usb_probe = command_runner(GATEMATE_USB_COMMAND, STEP0_TIMEOUT_S)
    polarfire_usb_probe = command_runner(POLARFIRE_USB_COMMAND, STEP0_TIMEOUT_S)

    kv260_config = parse_ssh_config(kv260_config_probe)
    polarfire_config = parse_ssh_config(polarfire_config_probe)
    env_summary = parse_hardware_environment(env_probe)
    toolchain = parse_toolchain_presence(toolchain_probe)
    preconditions = {
        "ssh_targets": {
            "KV260": {
                "alias": "kria",
                "ssh_config": kv260_config,
                "host_resolution": {
                    "available": kv260_resolution_probe.exit_code == 0,
                    "stdout": kv260_resolution_probe.stdout.strip(),
                    "stderr": kv260_resolution_probe.stderr.strip(),
                },
            },
            "PolarFire": {
                "alias": "polarfire",
                "ssh_config": polarfire_config,
                "host_resolution": {
                    "available": polarfire_resolution_probe.exit_code == 0,
                    "stdout": polarfire_resolution_probe.stdout.strip(),
                    "stderr": polarfire_resolution_probe.stderr.strip(),
                },
            },
        },
        "hardware_environment": env_summary,
        "toolchain_presence": toolchain,
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
            probe=kv260_config_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_ssh_config_kv260",
            outcome="recorded" if kv260_config_probe.exit_code == 0 else "ssh_config_unavailable",
            stdout_override=sanitized_ssh_config_stdout(kv260_config),
        ),
        command_receipt(
            probe=polarfire_config_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_ssh_config_polarfire",
            outcome=(
                "recorded" if polarfire_config_probe.exit_code == 0 else "ssh_config_unavailable"
            ),
            stdout_override=sanitized_ssh_config_stdout(polarfire_config),
        ),
        command_receipt(
            probe=kv260_resolution_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_host_resolution_kv260",
            outcome="resolved" if kv260_resolution_probe.exit_code == 0 else "not_resolved",
        ),
        command_receipt(
            probe=polarfire_resolution_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_host_resolution_polarfire",
            outcome="resolved" if polarfire_resolution_probe.exit_code == 0 else "not_resolved",
        ),
        command_receipt(
            probe=env_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_hardware_environment",
            outcome="recorded" if env_probe.exit_code == 0 else "env_unavailable",
            stdout_override=sanitized_env_stdout(env_summary),
        ),
        command_receipt(
            probe=toolchain_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_toolchain_presence",
            outcome="recorded" if toolchain_probe.exit_code == 0 else "toolchain_probe_failed",
        ),
        command_receipt(
            probe=gatemate_usb_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_usb_gatemate_dirtyjtag",
            outcome="visible" if gatemate_usb_probe.exit_code == 0 else "not_visible",
        ),
        command_receipt(
            probe=polarfire_usb_probe,
            timeout_s=STEP0_TIMEOUT_S,
            kind="step0_usb_polarfire_flashpro",
            outcome="visible" if polarfire_usb_probe.exit_code == 0 else "not_visible",
        ),
    ]
    return preconditions, commands


def gatemate_setup_changed_from_env(env_summary: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(
        bool(env_summary.get(key, {}).get("truthy"))
        for key in ("CARNOT_GATEMATE_SETUP_CHANGED", "GATEMATE_PHYSICAL_SETUP_CHANGED")
    )


def blocker_from_probe(reason: str, probe: CommandProbe) -> JsonDict:
    return {
        "reason": reason,
        "command": command_to_string(probe.command),
        "exit_code": int(probe.exit_code),
        "stdout": probe.stdout,
        "stderr": probe.stderr,
    }


def build_kv260_status(command_runner: CommandRunner) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    ssh_probe = command_runner(KV260_SSH_COMMAND, SSH_TIMEOUT_S)
    commands = [
        command_receipt(
            probe=ssh_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="kv260_ssh_reachability",
            outcome="reachable" if ssh_probe.exit_code == 0 else "blocked_kv260_ssh_unreachable",
        )
    ]
    if ssh_probe.exit_code != 0:
        return (
            {
                "status": "blocked_kv260_ssh_unreachable",
                "ssh_reachable": False,
                "board_level_checked": False,
                "speedup_claimed": False,
            },
            blocker_from_probe("blocked_kv260_ssh_unreachable", ssh_probe),
            commands,
        )

    summary_probe = command_runner(KV260_BOARD_SUMMARY_COMMAND, BOARD_DETAIL_TIMEOUT_S)
    commands.append(
        command_receipt(
            probe=summary_probe,
            timeout_s=BOARD_DETAIL_TIMEOUT_S,
            kind="kv260_board_level_summary",
            outcome="recorded" if summary_probe.exit_code == 0 else "summary_command_failed",
        )
    )
    return (
        {
            "status": "reachable_ssh_board_level_checked",
            "ssh_reachable": True,
            "board_level_checked": True,
            "board_summary_exit_code": summary_probe.exit_code,
            "board_summary_excerpt": summary_probe.combined_output.strip()[:500],
            "speedup_claimed": False,
        },
        None,
        commands,
    )


def terminal_workload_exists(probe: CommandProbe) -> bool:
    return probe.exit_code == 0 and "terminal_workload_exists=true" in probe.stdout


def build_polarfire_status(
    command_runner: CommandRunner,
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    ssh_probe = command_runner(POLARFIRE_SSH_COMMAND, SSH_TIMEOUT_S)
    commands = [
        command_receipt(
            probe=ssh_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_ssh_reachability",
            outcome="reachable" if ssh_probe.exit_code == 0 else "blocked_polarfire_ssh_unreachable",
        )
    ]
    if ssh_probe.exit_code != 0:
        return (
            {
                "status": "blocked_polarfire_ssh_unreachable",
                "ssh_reachable": False,
                "terminal_workload_exists": None,
                "speedup_claimed": False,
            },
            blocker_from_probe("blocked_polarfire_ssh_unreachable", ssh_probe),
            commands,
        )

    workload_probe = command_runner(POLARFIRE_TERMINAL_WORKLOAD_COMMAND, BOARD_DETAIL_TIMEOUT_S)
    exists = terminal_workload_exists(workload_probe)
    commands.append(
        command_receipt(
            probe=workload_probe,
            timeout_s=BOARD_DETAIL_TIMEOUT_S,
            kind="polarfire_terminal_workload_presence",
            outcome="terminal_workload_present" if exists else "terminal_workload_missing",
        )
    )
    status = (
        "reachable_terminal_workload_present"
        if exists
        else "reachable_terminal_workload_missing"
    )
    blocker = None if exists else blocker_from_probe(
        "blocked_polarfire_terminal_workload_missing", workload_probe
    )
    return (
        {
            "status": status,
            "ssh_reachable": True,
            "terminal_workload_exists": exists,
            "terminal_workload_probe_exit_code": workload_probe.exit_code,
            "speedup_claimed": False,
        },
        blocker,
        commands,
    )


def gate_detect_ok(probe: CommandProbe) -> bool:
    text = probe.combined_output
    return probe.exit_code == 0 and (
        GATEMATE_EXPECTED_IDCODE in text or "GM1Ax" in text or "GateMate Series" in text
    )


def build_gatemate_status(
    *,
    command_runner: CommandRunner,
    setup_changed: bool,
    preconditions: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict | None, list[JsonDict]]:
    toolchain = preconditions.get("toolchain_presence", {})
    usb_visibility = preconditions.get("usb_visibility", {}).get("GateMate", {})
    if not setup_changed:
        return (
            {
                "status": "blocked_gatemate_physical_jtag_setup_unchanged",
                "physical_setup_changed": False,
                "jtag_probe_attempted": False,
                "toolchain": toolchain,
                "dirtyjtag_usb": usb_visibility,
                "speedup_claimed": False,
            },
            {
                "reason": "operator_setup_unchanged_physical_jtag_block_carried_forward",
                "prior_evidence": str(PRIOR_RESULT_RELATIVE_PATH),
            },
            [],
        )

    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, GATEMATE_TIMEOUT_S)
    ok = gate_detect_ok(detect_probe)
    status = (
        "reachable_dirtyjtag_idcode_status_only"
        if ok
        else "blocked_gatemate_dirtyjtag_status_probe_failed"
    )
    blocker = None if ok else blocker_from_probe("blocked_gatemate_dirtyjtag_status_probe_failed", detect_probe)
    return (
        {
            "status": status,
            "physical_setup_changed": True,
            "jtag_probe_attempted": True,
            "toolchain": toolchain,
            "dirtyjtag_usb": usb_visibility,
            "detect_exit_code": detect_probe.exit_code,
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
    statuses: Mapping[str, Mapping[str, Any]],
    blockers: Mapping[str, JsonDict | None],
) -> str:
    summary = (
        f"kv260={statuses['KV260']['status']} "
        f"polarfire={statuses['PolarFire']['status']} "
        f"gatemate={statuses['GateMate']['status']} no_speedup_claim"
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
    preconditions, commands_run = collect_step0_preconditions(command_runner)
    env_setup_changed = gatemate_setup_changed_from_env(preconditions["hardware_environment"])
    setup_changed = env_setup_changed if gatemate_setup_changed is None else gatemate_setup_changed
    preconditions["operator_visible_hardware_assumptions"][
        "gatemate_physical_setup_changed_assumed"
    ] = bool(setup_changed)

    kv260_status, kv260_blocker, kv260_commands = build_kv260_status(command_runner)
    commands_run.extend(kv260_commands)
    polarfire_status, polarfire_blocker, polarfire_commands = build_polarfire_status(command_runner)
    commands_run.extend(polarfire_commands)
    gatemate_status, gatemate_blocker, gatemate_commands = build_gatemate_status(
        command_runner=command_runner,
        setup_changed=bool(setup_changed),
        preconditions=preconditions,
    )
    commands_run.extend(gatemate_commands)

    statuses = {
        "KV260": kv260_status,
        "PolarFire": polarfire_status,
        "GateMate": gatemate_status,
    }
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
        "honest_verdict": wrap_field("honest_verdict", build_honest_verdict(statuses, blockers)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "hardware_evidence_level": wrap_field(
            "hardware_evidence_level", HARDWARE_EVIDENCE_LEVEL
        ),
        "per_board_status": wrap_field("per_board_status", statuses),
        "host_sd_card_precondition_used": wrap_field("host_sd_card_precondition_used", False),
        "hardware_speedup_claimed": wrap_field("hardware_speedup_claimed", False),
        "blocked_reason": wrap_field("blocked_reason", blockers),
        "preconditions_checked": wrap_field("preconditions_checked", preconditions),
        "commands_run": commands_run,
        "docs_update_decision": {
            "research_hardware_wishlist_updated": False,
            "ops_status_updated": False,
            "ops_changelog_updated": False,
            "reason": "no material status change is inferred by this receipt builder",
        },
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md hardware continuity and KV260 SSH discipline",
            "research-hardware-wishlist.md",
            "hardware/kv260/README.md",
            "ops/hardware-bringup-prep.md",
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
    statuses = validate_wrapped_field(artifact, "per_board_status")
    require(isinstance(statuses, Mapping), "per_board_status must be a mapping")
    require(set(statuses) == {"KV260", "PolarFire", "GateMate"}, "per_board_status keys mismatch")
    blockers = validate_wrapped_field(artifact, "blocked_reason")
    require(isinstance(blockers, Mapping), "blocked_reason must be a mapping")
    require(set(blockers) == {"KV260", "PolarFire", "GateMate"}, "blocked_reason keys mismatch")
    require(
        validate_wrapped_field(artifact, "host_sd_card_precondition_used") is False,
        "host_sd_card_precondition_used must be false",
    )
    require(
        validate_wrapped_field(artifact, "hardware_speedup_claimed") is False,
        "hardware_speedup_claimed must be false",
    )
    preconditions = validate_wrapped_field(artifact, "preconditions_checked")
    require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    for key in (
        "ssh_targets",
        "hardware_environment",
        "toolchain_presence",
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
