#!/usr/bin/env python3
"""Exp 5266: hardware reachability plus thermodynamic sampler boundary.

Spec refs: REQ-HW-5266, SCENARIO-HW-5266.

This module records what the local bench can prove without inflating public
hardware literature into a local acceleration result. It checks host and
credential preconditions, probes KV260 and PolarFire with bounded SSH
reachability commands from the existing safe hardware-continuity pattern, keeps
GateMate at the known physical/JTAG blocker unless the physical setup changes,
and writes an ops note that turns thermodynamic sampler cost and
autocorrelation into future receipt requirements.
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
EXPERIMENT_ID = "exp5266-hardware-thermodynamic-schedule-boundary-v481"
EXPERIMENT_NAME = "experiment_5266_hardware_thermodynamic_schedule_boundary"
MILESTONE = "2026.07.481"
SCHEMA = "carnot.experiment_5266.hardware_thermodynamic_schedule_boundary.v481"
SPEC_REFS = ("REQ-HW-5266", "SCENARIO-HW-5266")
RANDOM_SEED = 5266
INFERENCE_SUBSTRATE = "hardware_probe_no_speedup_claim"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5266_hardware_thermodynamic_schedule_boundary_v481.json"
)
BOUNDARY_NOTE_RELATIVE_PATH = Path(
    "ops/research-thermodynamic-sampler-boundary-v481.md"
)
SAFE_PROBE_REFERENCE_PATH = Path(
    "python/carnot/experiment_5255_hardware_continuity_pkit_boundary_v480.py"
)

HOST_CPU_COMMAND = ("lscpu",)
HOST_GPU_COMMAND = ("nvidia-smi", "-L")
HARDWARE_ENV_COMMAND = ("env",)
KV260_SSH_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "kria", "true")
POLARFIRE_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)

PRECONDITION_TIMEOUT_S = 5.0
BOARD_SSH_TIMEOUT_S = 10.0
TERMINAL_PREFIXES = ("complete:", "blocked_")
BOARD_STATUSES = {
    "reachable",
    "blocked_kv260_ssh_unreachable",
    "blocked_polarfire_ssh_unreachable",
    "blocked_safe_probe_missing",
}
GATEMATE_STATUSES = {"blocked_physical_jtag", "not_checked_physical_setup_changed"}

PUBLIC_SOURCE_URLS = {
    "thermodynamic_ai_models": "https://arxiv.org/abs/2607.00170",
    "extropic_tsu_101": (
        "https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware"
    ),
    "extropic_xtr0": "https://extropic.ai/writing/inside-x0-and-xtr-0",
    "extropic_hardware": "https://extropic.ai/hardware",
}

HARDWARE_ENV_KEYS = (
    "CARNOT_KV260_HOST",
    "CARNOT_KV260_USER",
    "CARNOT_POLARFIRE_HOST",
    "CARNOT_POLARFIRE_USER",
    "CARNOT_GATEMATE_SETUP_CHANGED",
    "GATEMATE_PHYSICAL_SETUP_CHANGED",
    "EXTROPIC_API_KEY",
    "EXTROPIC_TOKEN",
    "THRML_HOME",
    "TSU_SDK",
    "XTR0_SDK",
)
HARDWARE_ENV_PREFIXES = ("CARNOT_", "EXTROPIC_", "THRML_", "TSU_", "XTR0_", "GATEMATE_")

HONEST_VERDICT_PRINCIPLE = (
    "Value starts with complete: or blocked_ and states board reachability plus "
    "no-speedup status."
)
SPEEDUP_CLAIMED_PRINCIPLE = (
    "False because this task records reachability and literature boundaries only; "
    "no local benchmark receipt compares hardware against CPU or GPU."
)
FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": HONEST_VERDICT_PRINCIPLE,
    "inference_substrate": (
        f"Must be {INFERENCE_SUBSTRATE}; this is reachability and boundary "
        "evidence, not acceleration evidence."
    ),
    "preconditions_checked": (
        "Records host CPU/GPU, network assumptions, and hardware credential or "
        "environment indicators before any board probe."
    ),
    "kv260_status": "Derived only from the SSH-safe KV260 reachability command.",
    "polarfire_status": "Derived only from the SSH-safe PolarFire reachability command.",
    "gatemate_status": (
        "Carry forward blocked_physical_jtag unless the operator changed the "
        "physical JTAG setup."
    ),
    "thermodynamic_boundary_updated": (
        "True only when the ops note records sampler-cost/autocorrelation as "
        "future requirements rather than local execution."
    ),
    "thermodynamic_boundary_note_path": (
        "Repository path to the ops note that documents future thermodynamic "
        "sampler-cost and autocorrelation receipt requirements."
    ),
}
REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_status",
    "polarfire_status",
    "gatemate_status",
    "thermodynamic_boundary_updated",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def round_duration(duration_s: float) -> float:
    return round(max(float(duration_s), 0.000001), 6)


@dataclass(frozen=True)
class CommandProbe:
    """One bounded command receipt with stdout and stderr preserved separately."""

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
            exc.stdout or "",
            exc.stderr or f"command timed out after {timeout_s}s",
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


def safe_probe_scripts_available(repo_root: str | Path) -> bool:
    return (Path(repo_root) / SAFE_PROBE_REFERENCE_PATH).exists()


def _field_from_lscpu(stdout: str, field: str) -> str | None:
    prefix = f"{field}:"
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip() or None
    return None


def parse_host_cpu(probe: CommandProbe) -> JsonDict:
    return {
        "available": probe.exit_code == 0,
        "model": _field_from_lscpu(probe.stdout, "Model name") or "unknown",
        "architecture": _field_from_lscpu(probe.stdout, "Architecture") or "unknown",
        "command": command_to_string(probe.command),
    }


def parse_host_gpu(probe: CommandProbe) -> JsonDict:
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    return {
        "available": probe.exit_code == 0 and bool(lines),
        "devices": lines,
        "command": command_to_string(probe.command),
        "error": "" if probe.exit_code == 0 else probe.combined_output.strip(),
    }


def _hardware_env_key_allowed(key: str) -> bool:
    return key in HARDWARE_ENV_KEYS or key.startswith(HARDWARE_ENV_PREFIXES)


def parse_hardware_environment(probe: CommandProbe) -> JsonDict:
    present_keys: set[str] = set()
    if probe.exit_code == 0:
        for line in probe.stdout.splitlines():
            key, separator, _value = line.partition("=")
            if separator and _hardware_env_key_allowed(key):
                present_keys.add(key)

    parsed = {
        key: {"present": key in present_keys, "value_recorded": False}
        for key in HARDWARE_ENV_KEYS
    }
    for key in sorted(present_keys - set(HARDWARE_ENV_KEYS)):
        parsed[key] = {"present": True, "value_recorded": False}
    return parsed


def sanitized_env_stdout(env_summary: Mapping[str, Mapping[str, Any]]) -> str:
    present = [key for key, value in env_summary.items() if value.get("present") is True]
    return "\n".join(sorted(present))


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


def collect_preconditions(command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    cpu_probe = command_runner(HOST_CPU_COMMAND, PRECONDITION_TIMEOUT_S)
    gpu_probe = command_runner(HOST_GPU_COMMAND, PRECONDITION_TIMEOUT_S)
    env_probe = command_runner(HARDWARE_ENV_COMMAND, PRECONDITION_TIMEOUT_S)

    env_summary = parse_hardware_environment(env_probe)
    preconditions = {
        "host_cpu": parse_host_cpu(cpu_probe),
        "host_gpu": parse_host_gpu(gpu_probe),
        "network_reachability_assumptions": {
            "internet_required_for_board_probe": False,
            "kv260_ssh_alias": "kria",
            "polarfire_ssh_alias": "polarfire",
            "ssh_config_or_dns_may_resolve_board_aliases": True,
            "kv260_host_block_device_required": False,
        },
        "hardware_environment": env_summary,
    }

    commands = [
        command_receipt(
            probe=cpu_probe,
            timeout_s=PRECONDITION_TIMEOUT_S,
            kind="precondition_host_cpu",
            outcome="recorded" if cpu_probe.exit_code == 0 else "blocked_cpu_probe_failed",
        ),
        command_receipt(
            probe=gpu_probe,
            timeout_s=PRECONDITION_TIMEOUT_S,
            kind="precondition_host_gpu",
            outcome="recorded" if gpu_probe.exit_code == 0 else "no_gpu_visible_or_tool_missing",
        ),
        command_receipt(
            probe=env_probe,
            timeout_s=PRECONDITION_TIMEOUT_S,
            kind="precondition_hardware_environment",
            outcome="recorded" if env_probe.exit_code == 0 else "blocked_env_probe_failed",
            stdout_override=sanitized_env_stdout(env_summary),
        ),
    ]
    return preconditions, commands


def board_status_from_probe(board: str, probe: CommandProbe) -> str:
    if probe.exit_code == 0:
        return "reachable"
    return f"blocked_{board}_ssh_unreachable"


def build_honest_verdict(*, kv260_status: str, polarfire_status: str, gatemate_status: str) -> str:
    if "blocked_safe_probe_missing" in {kv260_status, polarfire_status}:
        prefix = "blocked_safe_probe_missing:"
    elif kv260_status != "reachable" or polarfire_status != "reachable":
        prefix = "blocked_board_reachability:"
    else:
        prefix = "complete:"
    return (
        f"{prefix} kv260={kv260_status} polarfire={polarfire_status} "
        f"gatemate={gatemate_status} no_speedup_claim"
    )


def build_thermodynamic_boundary_note() -> str:
    return f"""# Exp 5266 Thermodynamic Sampler Boundary

Status: future requirements only. No speedup claim is allowed.

Sources reviewed:
- Scaling Up Thermodynamic AI Models, arXiv:2607.00170: {PUBLIC_SOURCE_URLS["thermodynamic_ai_models"]}
- Extropic TSU 101: {PUBLIC_SOURCE_URLS["extropic_tsu_101"]}
- Extropic X0/XTR-0: {PUBLIC_SOURCE_URLS["extropic_xtr0"]}
- Extropic hardware overview: {PUBLIC_SOURCE_URLS["extropic_hardware"]}

Boundary:
- The V481 thermodynamic-model reference treats Gibbs-sampled Ising systems as inference substrates and links inference cost, accuracy, and autocorrelation. For Carnot, that becomes a future requirement to record sampler-cost and autocorrelation receipts before any thermodynamic speedup claim.
- Extropic public pages describe TSUs and XTR-0 as programmable EBM/PGM sampling hardware, but this repo still has no local SDK or device, no authenticated TSU transcript, and no local XTR-0 execution.
- KV260 and PolarFire reachability checks are board-continuity receipts only. GateMate remains a physical/JTAG blocker unless the operator changes the physical setup.

Future requirement:
- A valid acceleration claim must include workload hash, executable or bitstream hash, output hash, wall-clock timing, sample-quality/autocorrelation diagnostics, CPU/GPU baseline parity, and device identity from the same reproducible run.

No speedup claim: Exp 5266 records reachability and boundary requirements only.
"""


def write_boundary_note(repo_root: str | Path) -> Path:
    path = Path(repo_root) / BOUNDARY_NOTE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_thermodynamic_boundary_note(), encoding="utf-8")
    return path


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    commit: str = "unknown",
    boundary_note_written: bool = True,
    physical_setup_changed: bool = False,
    safe_probe_scripts_present: bool = True,
) -> JsonDict:
    started = clock()
    preconditions, commands_run = collect_preconditions(command_runner)
    board_probe_notes: JsonDict = {}

    if safe_probe_scripts_present:
        kv260_probe = command_runner(KV260_SSH_COMMAND, BOARD_SSH_TIMEOUT_S)
        kv260_status = board_status_from_probe("kv260", kv260_probe)
        commands_run.append(
            command_receipt(
                probe=kv260_probe,
                timeout_s=BOARD_SSH_TIMEOUT_S,
                kind="kv260_ssh_reachability",
                outcome=kv260_status,
            )
        )

        polarfire_probe = command_runner(POLARFIRE_SSH_COMMAND, BOARD_SSH_TIMEOUT_S)
        polarfire_status = board_status_from_probe("polarfire", polarfire_probe)
        commands_run.append(
            command_receipt(
                probe=polarfire_probe,
                timeout_s=BOARD_SSH_TIMEOUT_S,
                kind="polarfire_ssh_reachability",
                outcome=polarfire_status,
            )
        )
        board_probe_notes["kv260"] = "ssh_only_safe_probe"
        board_probe_notes["polarfire"] = "ssh_only_safe_probe"
    else:
        kv260_status = "blocked_safe_probe_missing"
        polarfire_status = "blocked_safe_probe_missing"
        board_probe_notes["kv260"] = "safe_probe_missing"
        board_probe_notes["polarfire"] = "safe_probe_missing"

    gatemate_status = (
        "not_checked_physical_setup_changed" if physical_setup_changed else "blocked_physical_jtag"
    )
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
                kv260_status=kv260_status,
                polarfire_status=polarfire_status,
                gatemate_status=gatemate_status,
            ),
        ),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": wrap_field("preconditions_checked", preconditions),
        "kv260_status": wrap_field("kv260_status", kv260_status),
        "polarfire_status": wrap_field("polarfire_status", polarfire_status),
        "gatemate_status": wrap_field("gatemate_status", gatemate_status),
        "thermodynamic_boundary_updated": wrap_field(
            "thermodynamic_boundary_updated", bool(boundary_note_written)
        ),
        "thermodynamic_boundary_note_path": wrap_field(
            "thermodynamic_boundary_note_path",
            str(BOUNDARY_NOTE_RELATIVE_PATH) if boundary_note_written else None,
        ),
        "speedup_claimed": False,
        "speedup_claimed_principle": SPEEDUP_CLAIMED_PRINCIPLE,
        "commands_run": commands_run,
        "board_probe_notes": board_probe_notes,
        "safe_probe_reference_path": str(SAFE_PROBE_REFERENCE_PATH),
        "safe_probe_scripts_present": bool(safe_probe_scripts_present),
        "gatemate_carry_forward": {
            "status": gatemate_status,
            "physical_setup_changed": bool(physical_setup_changed),
            "evidence": [
                "results/experiment_5255_hardware_continuity_pkit_boundary_v480.json",
                "results/experiment_5243_hardware_continuity_kan_pbit_boundary_v479.json",
                "results/experiment_5231_hardware_continuity_pbit_boundary_v478.json",
            ],
            "rationale": (
                "no operator cable, port, power, or board setup change was provided"
                if not physical_setup_changed
                else "operator recorded a physical setup change; this task still avoids inventing a software workaround"
            ),
        },
        "thermodynamic_boundary_sources": dict(PUBLIC_SOURCE_URLS),
        "reviewed_inputs": [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md hardware continuity, preconditions, and artifact-field rules",
            "research-hardware-wishlist.md active hardware tracks",
            "research-references.md V481 Energy hardware, Ising, and certificates section",
            "ops/status.md",
            "ops/changelog.md",
            "ops/exclusion_manifest.yaml",
            "results/experiment_5255_hardware_continuity_pkit_boundary_v480.json",
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


def validate_command_receipts(commands: Any) -> None:
    require(isinstance(commands, list) and commands, "commands_run must be a non-empty list")
    for index, command in enumerate(commands):
        require(isinstance(command, Mapping), f"commands_run[{index}] must be a mapping")
        for key in ("command", "outcome", "timeout_s", "exit_code", "duration_s"):
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
    preconditions = validate_wrapped_field(artifact, "preconditions_checked")
    require(isinstance(preconditions, Mapping), "preconditions_checked must be a mapping")
    for key in ("host_cpu", "host_gpu", "network_reachability_assumptions", "hardware_environment"):
        require(key in preconditions, f"preconditions_checked missing {key}")
    require(validate_wrapped_field(artifact, "kv260_status") in BOARD_STATUSES, "bad kv260")
    require(
        validate_wrapped_field(artifact, "polarfire_status") in BOARD_STATUSES,
        "bad polarfire",
    )
    require(
        validate_wrapped_field(artifact, "gatemate_status") in GATEMATE_STATUSES,
        "bad gatemate",
    )
    require(
        isinstance(validate_wrapped_field(artifact, "thermodynamic_boundary_updated"), bool),
        "thermodynamic_boundary_updated type mismatch",
    )
    note_path = validate_wrapped_field(artifact, "thermodynamic_boundary_note_path")
    require(note_path is None or isinstance(note_path, str), "bad thermodynamic note path")
    require(artifact.get("speedup_claimed") is False, "speedup_claimed must be false")
    require(
        artifact.get("speedup_claimed_principle") == SPEEDUP_CLAIMED_PRINCIPLE,
        "speedup_claimed_principle mismatch",
    )
    validate_command_receipts(artifact.get("commands_run"))
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
    physical_setup_changed: bool = False,
) -> Path:
    note_path = write_boundary_note(repo_root)
    artifact = build_artifact(
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        commit=commit or get_git_commit(repo_root),
        boundary_note_written=note_path.exists(),
        physical_setup_changed=physical_setup_changed,
        safe_probe_scripts_present=safe_probe_scripts_available(repo_root),
    )
    return write_artifact(repo_root, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--physical-setup-changed",
        action="store_true",
        help="Record a changed GateMate setup while avoiding software workaround claims.",
    )
    args = parser.parse_args(argv)
    print(
        run_experiment(
            repo_root=Path("."),
            run_date=args.date,
            physical_setup_changed=args.physical_setup_changed,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
