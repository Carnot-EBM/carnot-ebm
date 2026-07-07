#!/usr/bin/env python3
"""Exp 5347: hardware continuity with board-local workload receipts.

Spec refs: REQ-HW-5347, SCENARIO-HW-5347.

This artifact builder records the narrow kind of evidence that is available in
the current hardware lane: board reachability and, for PolarFire when SSH works,
a tiny deterministic workload that executes on the board and reports matching
hashes. The receipt proves authenticated workload reachability only. It does not
compare against a baseline and therefore cannot support a speedup claim.
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
from carnot import experiment_5333_hardware_continuity_no_speedup_v486 as prev


JsonDict = dict[str, Any]
Clock = context_prev.Clock
CommandProbe = base.CommandProbe
CommandRunner = base.CommandRunner

RUN_DATE = "20260707"
EXPERIMENT_ID = "exp5347-hardware-continuity-workload-receipts-v487"
EXPERIMENT_NAME = "experiment_5347_hardware_continuity_workload_receipts"
MILESTONE = "2026.07.487"
SCHEMA = "carnot.experiment_5347.hardware_continuity_workload_receipts.v487"
SPEC_REFS = ("REQ-HW-5347", "SCENARIO-HW-5347")
RANDOM_SEED = 5347
INFERENCE_SUBSTRATE = "hardware_smoke"
HARDWARE_EVIDENCE_LEVEL = "board_reachability_plus_hash_verified_workload_no_speedup"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5347_hardware_continuity_workload_receipts_v487.json"
)
PRIOR_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5333_hardware_continuity_no_speedup_v486.json"
)

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
TERMINAL_PREFIXES = ("complete:", "blocked_")

POLARFIRE_WORKLOAD_INPUT = b"carnot-exp5347-polarfire-workload-v487\n"
POLARFIRE_WORKLOAD_OUTPUT_SUFFIX = b"|polarfire-v487-output"
POLARFIRE_EXPECTED_INPUT_SHA256 = hashlib.sha256(POLARFIRE_WORKLOAD_INPUT).hexdigest()
POLARFIRE_EXPECTED_OUTPUT_SHA256 = hashlib.sha256(
    POLARFIRE_WORKLOAD_INPUT + POLARFIRE_WORKLOAD_OUTPUT_SUFFIX
).hexdigest()
POLARFIRE_WORKLOAD_PYTHON = (
    "import hashlib,json,platform,socket,time;"
    "started=time.perf_counter();"
    'payload=b"carnot-exp5347-polarfire-workload-v487\\n";'
    'out=hashlib.sha256(payload+b"|polarfire-v487-output").hexdigest();'
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
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_status",
    "polarfire_status",
    "gatemate_status",
    "tests_run",
    "commands_run",
)
REQUIRED_BARE_BOOLEAN_FIELDS = (
    "polarfire_workload_validated",
    "authenticated_workload_run",
    "public_refs_context_only",
    "speedup_claim",
    "no_host_block_device_evidence",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Identifies the v487 workload-receipt artifact so reconciliation can cite one exact hardware-smoke result.",
    "milestone": "Pins the receipt to milestone 2026.07.487 rather than a floating current hardware state.",
    "status": "Summarizes whether a PolarFire board-local workload receipt was validated without implying acceleration.",
    "honest_verdict": (
        "Terminal verdict starts with complete: or blocked_ and states board, "
        "public-reference, workload, and no-speedup boundaries."
    ),
    "inference_substrate": (
        "hardware_smoke because the artifact records bounded board probes and "
        "a tiny authenticated workload receipt when PolarFire SSH is available."
    ),
    "preconditions_checked": (
        "Records host/date context, sanitized hardware environment, local tools, "
        "USB visibility, public-reference boundaries, and SSH targets before classification."
    ),
    "kv260_status": (
        "KV260 status comes only from the required BatchMode SSH true command, "
        "so host block-device state cannot masquerade as board evidence."
    ),
    "polarfire_status": (
        "PolarFire status records authenticated SSH reachability and, when present, "
        "the board-local deterministic workload receipt with hashes."
    ),
    "gatemate_status": (
        "GateMate status records physical, JTAG, and toolchain evidence actually "
        "available without flashing or timing a sampler."
    ),
    "tests_run": "Records verification commands so the receipt is tied to passing tests.",
    "commands_run": "Captures bounded command receipts for every hardware-status or workload probe.",
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


def kv260_command_receipt(probe: CommandProbe, outcome: str) -> JsonDict:
    receipt = context_prev.command_receipt(
        probe=probe,
        timeout_s=SSH_TIMEOUT_S,
        kind="kv260_ssh_true_reachability_probe",
        outcome=outcome,
    )
    receipt["command"] = KV260_REQUIRED_COMMAND_FORM
    return receipt


def kv260_status_from_probe(probe: CommandProbe) -> tuple[JsonDict, JsonDict | None, JsonDict]:
    reachable = probe.exit_code == 0
    status = "reachable_ssh_status_only" if reachable else "blocked_kv260_ssh_unreachable"
    board_status = {
        "board": "KV260",
        "status": status,
        "ssh_reachable": reachable,
        "check_method": "ssh_batchmode_true_only",
        "command_form": KV260_REQUIRED_COMMAND_FORM,
        "remote_identifier": base.remote_identifier(probe.combined_output)
        if probe.combined_output
        else None,
        "probe_exit_code": int(probe.exit_code),
        "speedup_claimed": False,
    }
    blocker = None if reachable else base.blocker_from_probe(status, probe, SSH_TIMEOUT_S)
    if blocker is not None:
        blocker = dict(blocker)
        blocker["command"] = KV260_REQUIRED_COMMAND_FORM
    return board_status, blocker, kv260_command_receipt(probe, status)


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
) -> tuple[JsonDict, JsonDict | None, list[JsonDict], bool]:
    status, blocker, commands = base.build_ssh_reachability(
        board_label="PolarFire",
        command=POLARFIRE_STATUS_COMMAND,
        blocked_reason="blocked_polarfire_ssh_unreachable",
        kind="polarfire_authenticated_status_probe",
        command_runner=command_runner,
    )
    status = dict(status)
    status.update(
        {
            "workload_attempted": False,
            "workload_validated": False,
            "workload_receipt": None,
            "workload_error": None,
        }
    )
    if not status["ssh_reachable"]:
        status["workload_error"] = "ssh_unreachable"
        return status, blocker, commands, False

    workload_probe = command_runner(POLARFIRE_WORKLOAD_COMMAND, SSH_TIMEOUT_S)
    receipt, parse_error = parse_polarfire_workload_stdout(workload_probe.stdout)
    exit_ok = workload_probe.exit_code == 0
    validated = bool(exit_ok and receipt is not None and parse_error is None)
    if validated:
        outcome = "reachable_workload_receipt_validated"
        workload_blocker = None
        workload_error = None
    else:
        outcome = "reachable_workload_receipt_invalid" if exit_ok else "reachable_workload_failed"
        workload_error = parse_error or workload_probe.stderr.strip() or "workload command failed"
        workload_blocker = base.blocker_from_probe(
            "blocked_polarfire_workload_receipt_invalid",
            workload_probe,
            SSH_TIMEOUT_S,
        )
        workload_blocker["workload_error"] = workload_error

    status.update(
        {
            "status": outcome,
            "workload_attempted": True,
            "workload_validated": validated,
            "workload_receipt": receipt,
            "workload_error": workload_error,
            "speedup_claimed": False,
        }
    )
    commands.append(
        context_prev.command_receipt(
            probe=workload_probe,
            timeout_s=SSH_TIMEOUT_S,
            kind="polarfire_board_local_workload_receipt",
            outcome=outcome,
        )
    )
    return status, workload_blocker, commands, validated


def status_value(*, polarfire_ssh_reachable: bool, polarfire_workload_validated: bool) -> str:
    if polarfire_workload_validated:
        return "complete_hardware_smoke_workload_receipt_no_speedup"
    if polarfire_ssh_reachable:
        return "blocked_polarfire_workload_receipt_invalid"
    return "blocked_polarfire_ssh_unreachable"


def honest_verdict(
    *,
    status: str,
    kv260_status: Mapping[str, Any],
    polarfire_status: Mapping[str, Any],
    gatemate_status: Mapping[str, Any],
    authenticated_workload_run: bool,
) -> str:
    authenticated = str(bool(authenticated_workload_run)).lower()
    summary = (
        f"kv260={kv260_status['status']} "
        f"polarfire={polarfire_status['status']} "
        f"gatemate={gatemate_status['status']} "
        f"authenticated_workload_run={authenticated} "
        "public_refs=context_only "
        "speedup_claim=false"
    )
    if status.startswith("complete_"):
        return f"complete: {summary}"
    return f"{status}: {summary}"


def no_host_block_device_evidence(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    return "mmcblk" not in encoded and "/dev/disk" not in encoded


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
    gatemate_setup_changed: bool | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = clock()
    context, commands_run = context_prev.collect_preconditions(command_runner)
    env_setup_changed = base.gatemate_setup_changed_from_env(context["hardware_environment"])
    setup_changed = env_setup_changed if gatemate_setup_changed is None else gatemate_setup_changed
    context["operator_visible_hardware_assumptions"]["gatemate_physical_setup_changed_assumed"] = (
        bool(setup_changed)
    )
    context["operator_visible_hardware_assumptions"]["authenticated_workload_run"] = None
    context["operator_visible_hardware_assumptions"]["no_speedup_claim"] = True
    context["ssh_targets"] = {"KV260": "kria", "PolarFire": "polarfire"}

    kv260_probe = command_runner(KV260_SSH_TRUE_COMMAND, SSH_TIMEOUT_S)
    kv260, kv260_blocker, kv260_receipt = kv260_status_from_probe(kv260_probe)
    commands_run.append(kv260_receipt)

    polarfire, polarfire_blocker, polarfire_commands, workload_validated = (
        polarfire_status_with_workload(command_runner=command_runner)
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

    status = status_value(
        polarfire_ssh_reachable=bool(polarfire["ssh_reachable"]),
        polarfire_workload_validated=workload_validated,
    )
    tests = [dict(item) for item in (tests_run if tests_run is not None else default_tests_run())]
    context["operator_visible_hardware_assumptions"]["authenticated_workload_run"] = (
        workload_validated
    )
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
                authenticated_workload_run=workload_validated,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": wrap_field("preconditions_checked", context),
        "kv260_status": wrap_field("kv260_status", kv260),
        "polarfire_status": wrap_field("polarfire_status", polarfire),
        "gatemate_status": wrap_field("gatemate_status", gatemate),
        "tests_run": wrap_field("tests_run", tests),
        "commands_run": wrap_field("commands_run", commands_run),
        "polarfire_workload_validated": workload_validated,
        "authenticated_workload_run": workload_validated,
        "public_refs_context_only": True,
        "speedup_claim": False,
        "no_host_block_device_evidence": True,
        "hardware_evidence_level": HARDWARE_EVIDENCE_LEVEL,
        "blocked_reason": {
            "KV260": kv260_blocker,
            "PolarFire": polarfire_blocker,
            "GateMate": gatemate_blocker,
        },
        "public_hardware_context_boundaries": context["public_reference_boundaries"],
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "research-hardware-wishlist.md",
            str(PRIOR_RESULT_RELATIVE_PATH),
            "results/experiment_5319_hardware_continuity_no_speedup_v485.json",
            "ops/status.md",
            "ops/changelog.md",
        ],
        "docs_update_decision": {
            "ops_changelog_updated": False,
            "ops_status_updated": False,
            "reason": "task stop rule delegates docs/status reconciliation to the conductor",
        },
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["no_host_block_device_evidence"] = no_host_block_device_evidence(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_wrapped_field(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
    require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
    require("value" in wrapped, f"{field} missing value")
    return wrapped["value"]


def validate_bare_booleans(artifact: Mapping[str, Any]) -> None:
    for field in REQUIRED_BARE_BOOLEAN_FIELDS:
        require(isinstance(artifact.get(field), bool), f"{field} must be a bare boolean")
    require(
        artifact["authenticated_workload_run"] == artifact["polarfire_workload_validated"],
        "authenticated_workload_run must match polarfire_workload_validated",
    )
    require(artifact["public_refs_context_only"] is True, "public_refs_context_only must be true")
    require(artifact["speedup_claim"] is False, "speedup_claim must be false")
    require(
        artifact["no_host_block_device_evidence"] is True,
        "no_host_block_device_evidence must be true",
    )


def validate_board_status(value: Any, board: str) -> None:
    require(isinstance(value, Mapping), f"{board} status must be a mapping")
    require(value.get("board") == board, f"{board} board label mismatch")
    require(isinstance(value.get("status"), str) and value["status"], f"{board} status missing")
    require(value.get("speedup_claimed") is False, f"{board} speedup_claimed must be false")


def validate_polarfire_status(value: Any, workload_validated: bool) -> None:
    validate_board_status(value, "PolarFire")
    require(isinstance(value.get("workload_attempted"), bool), "PolarFire workload_attempted missing")
    require(isinstance(value.get("workload_validated"), bool), "PolarFire workload_validated missing")
    require(
        value["workload_validated"] is workload_validated,
        "PolarFire workload validation disagrees with top-level field",
    )
    if workload_validated:
        receipt = value.get("workload_receipt")
        require(isinstance(receipt, Mapping), "PolarFire validated workload missing receipt")
        for key in ("hostname", "uname", "python_version", "input_sha256", "output_sha256"):
            require(isinstance(receipt.get(key), str) and receipt[key], f"receipt missing {key}")
        require(
            receipt["input_sha256"] == POLARFIRE_EXPECTED_INPUT_SHA256,
            "receipt input checksum mismatch",
        )
        require(
            receipt["output_sha256"] == POLARFIRE_EXPECTED_OUTPUT_SHA256,
            "receipt output checksum mismatch",
        )
        require(isinstance(receipt.get("wall_time_s"), int | float), "receipt wall time missing")
        require(receipt["speedup_claimed"] is False, "receipt speedup claim drift")


def validate_commands(commands: Any) -> None:
    require(isinstance(commands, list) and commands, "commands_run must be a non-empty list")
    for index, command in enumerate(commands):
        require(isinstance(command, Mapping), f"commands_run[{index}] must be a mapping")
        for key in ("command", "outcome", "exit_code", "timeout_s", "duration_s"):
            require(key in command, f"commands_run[{index}] missing {key}")
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
    for field in REQUIRED_WRAPPED_FIELDS:
        validate_wrapped_field(artifact, field)
    require(artifact.get("schema") == SCHEMA, "schema mismatch")
    require(
        validate_wrapped_field(artifact, "experiment_id") == EXPERIMENT_ID, "experiment_id mismatch"
    )
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
        "authenticated_workload_run=",
        "public_refs=context_only",
        "speedup_claim=false",
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
        "public_reference_boundaries",
        "kv260_check_method",
        "operator_visible_hardware_assumptions",
    ):
        require(key in preconditions, f"preconditions_checked missing {key}")
    require(
        preconditions["kv260_check_method"] == "ssh_batchmode_true_only", "KV260 method mismatch"
    )
    validate_board_status(validate_wrapped_field(artifact, "kv260_status"), "KV260")
    workload_validated = artifact.get("polarfire_workload_validated")
    validate_polarfire_status(validate_wrapped_field(artifact, "polarfire_status"), workload_validated)
    validate_board_status(validate_wrapped_field(artifact, "gatemate_status"), "GateMate")
    validate_tests_run(validate_wrapped_field(artifact, "tests_run"))
    validate_commands(validate_wrapped_field(artifact, "commands_run"))
    validate_bare_booleans(artifact)
    require(no_host_block_device_evidence(artifact), "host block-device marker present")
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
    gatemate_setup_changed: bool | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or base.get_git_commit(repo_root),
        gatemate_setup_changed=gatemate_setup_changed,
        tests_run=tests_run,
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
