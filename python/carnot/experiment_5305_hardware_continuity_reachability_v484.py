#!/usr/bin/env python3
"""Exp 5305: hardware continuity status receipts.

Spec refs: REQ-HW-5305, SCENARIO-HW-5305.

This module builds a continuity receipt rather than a performance result. It
records what an operator can safely observe about the attached hardware bench:
local host/date context, environment and tool visibility, SSH status for KV260
and PolarFire, and GateMate USB/JTAG status only when the physical setup is
explicitly marked changed. These receipts are useful for continuity accounting,
but they are deliberately not workload completion or acceleration evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5293_hardware_continuity_reachability_v483 as base


JsonDict = dict[str, Any]
Clock = base.Clock
CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner

RUN_DATE = "20260706"
EXPERIMENT_ID = "exp5305-hardware-continuity-reachability-v484"
EXPERIMENT_NAME = "experiment_5305_hardware_continuity_reachability"
MILESTONE = "2026.07.484"
SCHEMA = "carnot.experiment_5305.hardware_continuity_reachability.v484"
SPEC_REFS = ("REQ-HW-5305", "SCENARIO-HW-5305")
RANDOM_SEED = 5305
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE
HARDWARE_EVIDENCE_LEVEL = base.HARDWARE_EVIDENCE_LEVEL

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5305_hardware_continuity_reachability_v484.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5293_hardware_continuity_reachability_v483.json"
)

HOST_DATE_COMMAND = (
    "sh",
    "-lc",
    (
        "printf 'host='; hostname; "
        "printf 'date_utc='; date -u +%Y-%m-%dT%H:%M:%SZ; "
        "printf 'date_local='; date +%Y-%m-%dT%H:%M:%S%z"
    ),
)
SSH_TARGETS = {"KV260": "kria", "PolarFire": "polarfire"}

HARDWARE_ENV_COMMAND = base.HARDWARE_ENV_COMMAND
TOOL_VERSION_COMMAND = base.TOOL_VERSION_COMMAND
GATEMATE_USB_COMMAND = base.GATEMATE_USB_COMMAND
POLARFIRE_USB_COMMAND = base.POLARFIRE_USB_COMMAND
KV260_REACHABILITY_COMMAND = base.KV260_REACHABILITY_COMMAND
POLARFIRE_REACHABILITY_COMMAND = base.POLARFIRE_REACHABILITY_COMMAND
GATEMATE_DETECT_COMMAND = base.GATEMATE_DETECT_COMMAND

LOCAL_TIMEOUT_S = base.LOCAL_TIMEOUT_S
SSH_TIMEOUT_S = base.SSH_TIMEOUT_S
GATEMATE_TIMEOUT_S = base.GATEMATE_TIMEOUT_S
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES

REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "hardware_evidence_level",
    "kv260_status",
    "polarfire_status",
    "gatemate_status",
    "hardware_speedup_claimed",
    "blocked_reason",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal verdict starts with complete: or blocked_ and states KV260, "
        "PolarFire, GateMate, and no-speedup outcomes."
    ),
    "inference_substrate": base.FIELD_PRINCIPLES["inference_substrate"],
    "hardware_evidence_level": base.FIELD_PRINCIPLES["hardware_evidence_level"],
    "kv260_status": (
        "KV260 status is checked only by authenticated SSH board commands; no "
        "host storage state is a precondition."
    ),
    "polarfire_status": (
        "PolarFire status records authenticated SSH or terminal reachability "
        "only and does not imply workload success."
    ),
    "gatemate_status": (
        "GateMate status records safe USB/tool/JTAG visibility and carries the "
        "physical setup blocker unless the operator changed it."
    ),
    "hardware_speedup_claimed": base.FIELD_PRINCIPLES["hardware_speedup_claimed"],
    "blocked_reason": (
        "Per-board blockers preserve exact command failure text or the carried "
        "physical/JTAG reason."
    ),
    "preconditions_checked": (
        "Step 0 records local host/date context, environment, tools, USB/JTAG "
        "visibility, SSH targets, and operator-visible limitations before board "
        "classification."
    ),
    "host_storage_precondition_used": (
        "False because the KV260 path is SSH-only and does not depend on host "
        "removable storage."
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
    if probe.exit_code != 0:
        return summary
    for line in probe.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"host", "date_utc", "date_local"}:
            summary[key] = value.strip() or None
    return summary


def collect_step0_preconditions(command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    host_probe = command_runner(HOST_DATE_COMMAND, LOCAL_TIMEOUT_S)
    context, commands = base.collect_local_probe_context(command_runner)
    context["host_date"] = parse_host_date(host_probe)
    context["ssh_targets"] = dict(SSH_TARGETS)
    context["operator_visible_hardware_assumptions"].update(
        {
            "kv260_host_storage_precondition_retired": True,
            "no_speedup_claim": True,
            "gatemate_physical_jtag_block_carried_forward_unless_setup_changed": True,
        }
    )
    commands.insert(
        0,
        base.command_receipt(
            probe=host_probe,
            timeout_s=LOCAL_TIMEOUT_S,
            kind="step0_host_date_context",
            outcome="recorded" if host_probe.exit_code == 0 else "host_date_unavailable",
        ),
    )
    return context, commands


def build_artifact(
    *,
    command_runner: CommandRunner = base.run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    gatemate_setup_changed: bool | None = None,
) -> JsonDict:
    started = clock()
    context, commands_run = collect_step0_preconditions(command_runner)
    env_setup_changed = base.gatemate_setup_changed_from_env(context["hardware_environment"])
    setup_changed = env_setup_changed if gatemate_setup_changed is None else gatemate_setup_changed
    context["operator_visible_hardware_assumptions"][
        "gatemate_physical_setup_changed_assumed"
    ] = bool(setup_changed)

    kv260, kv260_blocker, kv260_commands = base.build_ssh_reachability(
        board_label="KV260",
        command=KV260_REACHABILITY_COMMAND,
        blocked_reason="blocked_kv260_ssh_unreachable",
        kind="kv260_ssh_status_probe",
        command_runner=command_runner,
    )
    commands_run.extend(kv260_commands)
    polarfire, polarfire_blocker, polarfire_commands = base.build_ssh_reachability(
        board_label="PolarFire",
        command=POLARFIRE_REACHABILITY_COMMAND,
        blocked_reason="blocked_polarfire_ssh_unreachable",
        kind="polarfire_ssh_status_probe",
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
        "duration_s": base.round_duration(clock() - started),
        "commit": commit,
        "honest_verdict": wrap_field(
            "honest_verdict",
            base.build_honest_verdict(
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
        "kv260_status": wrap_field("kv260_status", kv260),
        "polarfire_status": wrap_field("polarfire_status", polarfire),
        "gatemate_status": wrap_field("gatemate_status", gatemate),
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


def validate_board_status(value: Any, board: str) -> None:
    require(isinstance(value, Mapping), f"{board} status must be a mapping")
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
    validate_board_status(validate_wrapped_field(artifact, "kv260_status"), "KV260")
    validate_board_status(validate_wrapped_field(artifact, "polarfire_status"), "PolarFire")
    validate_board_status(validate_wrapped_field(artifact, "gatemate_status"), "GateMate")
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
        "host_date",
        "hardware_environment",
        "tool_versions",
        "usb_visibility",
        "ssh_targets",
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
