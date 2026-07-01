#!/usr/bin/env python3
"""Exp 5093: hardware continuity v2 for KV260, GateMate, and PolarFire.

Spec refs: REQ-HW-5093, SCENARIO-HW-5093.

This module records live, non-destructive prechecks without turning hardware
visibility into a speedup claim. KV260 is checked only through SSH and an
already-existing safe UIO/register transcript, GateMate is detected through
DirtyJTAG only, and PolarFire runs only the hash-dispatch preconditions that
prove a future dispatch would be meaningful.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandProbe"]

EXPERIMENT_ID = 5093
EXPERIMENT_NAME = "experiment_5093_hardware_continuity"
SCHEMA = "carnot.experiment_5093_hardware_continuity.v467"
OUTPUT_REL_PATH = Path("results") / "experiment_5093_hardware_continuity_v467.json"
SAFE_KV260_UIO_TRANSCRIPT_REL_PATH = (
    Path("results") / "experiment_5093_kv260_uio_register_transcript.jsonl"
)
PRIOR_KV260_TRANSCRIPT_REL_PATH = (
    Path("results") / "experiment_5065_kv260_testbench_timing_packet.transcript.jsonl"
)
SPEC_REFS = ["REQ-HW-5093", "SCENARIO-HW-5093"]
RANDOM_SEED = 5093
INFERENCE_SUBSTRATE = "hardware_precheck_and_transcript_audit"
SUCCESS_VERDICT = "success_hardware_continuity_v467_no_speedup_claim"
PARTIAL_VERDICT = "complete_hardware_continuity_v467_partial_board_blockers"

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_COMMAND_AVAILABLE_COMMAND = ("sh", "-lc", "command -v openFPGALoader")
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
POLARFIRE_ARCH_COMMAND = ("ssh", "polarfire", "uname -m")
POLARFIRE_PYTHON_COMMAND = ("ssh", "polarfire", "python3 --version")
POLARFIRE_UPTIME_COMMAND = ("ssh", "polarfire", "uptime")
POLARFIRE_KERNEL_COMMAND = ("ssh", "polarfire", "uname -r")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_ready",
    "kv260_uio_transcript_path",
    "kv260_speedup_claim_allowed",
    "gatemate_detected",
    "gatemate_terminal_state",
    "polarfire_detected",
    "polarfire_dispatch_precheck_ready",
    "destructive_actions_taken",
    "board_matrix",
    "flagged_adversarial",
)
REQUIRED_SCHEMA_FIELDS = (
    *REQUIRED_ARTIFACT_FIELDS,
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "random_seed",
    "field_principles",
    "destructive_actions_allowed",
    "command_probes",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal-prefixed verdict distinguishes a clean audit from partial board blockers without retrying hardware."
    },
    "duration_s": {
        "principle": "measured wall-clock time makes live prechecks auditable and discourages fabricated hardware state."
    },
    "inference_substrate": {
        "principle": "declares hardware_precheck_and_transcript_audit so SSH/detect transcripts are not mistaken for benchmark inference."
    },
    "preconditions_checked": {
        "principle": "records the resources checked before any follow-on probe, including the explicit no-destructive-actions policy."
    },
    "kv260_ssh_ready": {
        "principle": "true only from the required BatchMode SSH command to kria; host storage checks are invalid for KV260."
    },
    "kv260_uio_transcript_path": {
        "principle": "points only to an already-existing safe read-only UIO/register transcript, or null with an explicit blocker."
    },
    "kv260_speedup_claim_allowed": {
        "principle": "must stay false because this artifact is continuity/precheck evidence, not a latency or speedup benchmark."
    },
    "gatemate_detected": {
        "principle": "true only when non-destructive DirtyJTAG detection reports GateMate/GM1Ax/IDCODE evidence."
    },
    "gatemate_terminal_state": {
        "principle": "names the current GateMate state without flashing, programming, or claiming sampler output."
    },
    "polarfire_detected": {
        "principle": "true only from non-destructive PolarFire BatchMode SSH reachability."
    },
    "polarfire_dispatch_precheck_ready": {
        "principle": "true only when non-mutating hash-dispatch prerequisites pass; no SCP or workload dispatch is performed."
    },
    "destructive_actions_taken": {
        "principle": "must be empty because this continuity audit is precheck-only."
    },
    "board_matrix": {
        "principle": "one row per active board preserves evidence, blockers, terminal state, and safe next step."
    },
    "flagged_adversarial": {
        "principle": "false only when schema checks pass and the artifact preserves no-speedup and no-destructive-action boundaries."
    },
}


class CommandProbe:
    """Captured result for a non-destructive command.

    Hardware continuity work needs the exact command result, not a paraphrase,
    because later operators need to distinguish a missing tool, a missing cable,
    and a reachable board without rerunning the probe.
    """

    def __init__(
        self,
        command: Sequence[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_s: float,
    ) -> None:
        self.command = tuple(command)
        self.exit_code = int(exit_code)
        self.stdout = stdout
        self.stderr = stderr
        self.duration_s = float(duration_s)

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> JsonDict:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "combined_output": self.combined_output,
            "duration_s": _round_duration(self.duration_s),
        }


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepend_oss_cad_suite() -> None:  # pragma: no cover - host environment dependent.
    candidate = Path("/opt/oss-cad-suite/bin")
    if not (candidate / "openFPGALoader").exists():
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if str(candidate) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(candidate), *parts])


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:  # pragma: no cover
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandProbe(
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"command timed out after {timeout_s}s"
        return CommandProbe(command, 124, stdout, stderr, time.perf_counter() - started)
    except OSError as exc:
        return CommandProbe(command, 127, "", str(exc), time.perf_counter() - started)


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Build the v467 continuity artifact from live or injected prechecks."""

    started = clock()
    kv260_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    gatemate_command_probe = command_runner(GATEMATE_COMMAND_AVAILABLE_COMMAND, 10.0)
    gatemate_command_available = gatemate_command_probe.exit_code == 0
    gatemate_detect_probe = (
        command_runner(GATEMATE_DETECT_COMMAND, 30.0) if gatemate_command_available else None
    )
    polarfire_ssh_probe = command_runner(POLARFIRE_SSH_COMMAND, 10.0)
    polarfire_probe_bundle = _run_polarfire_precheck_bundle(
        polarfire_ssh_probe=polarfire_ssh_probe,
        command_runner=command_runner,
    )
    kv260_uio_status = verify_safe_kv260_uio_transcript(repo_root)

    kv260_ssh_ready = kv260_probe.exit_code == 0
    gatemate_detected = _gatemate_detected(gatemate_detect_probe)
    polarfire_detected = polarfire_ssh_probe.exit_code == 0
    preconditions = [
        _precondition_entry(
            "kv260_ssh",
            kv260_probe,
            kv260_ssh_ready,
            "ssh_only_no_host_sd_card",
        ),
        _precondition_entry(
            "gatemate_detect_command",
            gatemate_command_probe,
            gatemate_command_available,
            "command_availability_only_no_detect_side_effect",
        ),
        _precondition_entry(
            "polarfire_ssh",
            polarfire_ssh_probe,
            polarfire_detected,
            "ssh_reachability_before_hash_dispatch_precheck",
        ),
        _policy_precondition(),
    ]
    gatemate_terminal_state = _gatemate_terminal_state(
        command_available=gatemate_command_available,
        detected=gatemate_detected,
        detect_probe=gatemate_detect_probe,
    )
    polarfire_dispatch_ready = bool(polarfire_probe_bundle["ready"])
    polarfire_terminal_state = _polarfire_terminal_state(
        detected=polarfire_detected,
        dispatch_ready=polarfire_dispatch_ready,
    )
    kv260_terminal_state = _kv260_terminal_state(kv260_ssh_ready, kv260_uio_status)
    success = bool(
        kv260_ssh_ready
        and kv260_uio_status["verified"]
        and gatemate_detected
        and polarfire_dispatch_ready
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": SUCCESS_VERDICT if success else PARTIAL_VERDICT,
        "duration_s": _round_duration(clock() - started),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "destructive_actions_allowed": False,
        "kv260_ssh_ready": kv260_ssh_ready,
        "kv260_uio_transcript_path": kv260_uio_status["path"],
        "kv260_speedup_claim_allowed": False,
        "gatemate_detected": gatemate_detected,
        "gatemate_terminal_state": gatemate_terminal_state,
        "polarfire_detected": polarfire_detected,
        "polarfire_dispatch_precheck_ready": polarfire_dispatch_ready,
        "destructive_actions_taken": [],
        "flagged_adversarial": False,
        "command_probes": _command_probes(
            kv260_probe=kv260_probe,
            gatemate_command_probe=gatemate_command_probe,
            gatemate_detect_probe=gatemate_detect_probe,
            polarfire_ssh_probe=polarfire_ssh_probe,
            polarfire_probe_bundle=polarfire_probe_bundle,
        ),
        "board_matrix": {
            "kv260": _kv260_row(
                ssh_ready=kv260_ssh_ready,
                terminal_state=kv260_terminal_state,
                precondition=preconditions[0],
                uio_status=kv260_uio_status,
            ),
            "gatemate": _gatemate_row(
                command_available=gatemate_command_available,
                detected=gatemate_detected,
                terminal_state=gatemate_terminal_state,
                precondition=preconditions[1],
                detect_probe=gatemate_detect_probe,
            ),
            "polarfire": _polarfire_row(
                detected=polarfire_detected,
                terminal_state=polarfire_terminal_state,
                precondition=preconditions[2],
                dispatch_bundle=polarfire_probe_bundle,
            ),
        },
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: JsonMap) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def verify_safe_kv260_uio_transcript(repo_root: str | Path) -> JsonDict:
    """Verify a pre-existing read-only UIO/register transcript if present.

    The continuity task is allowed to audit an established transcript, but it
    should not create a new register-touching script in this precheck-only lane.
    A missing transcript is therefore a recorded blocker, not a failure.
    """

    path = Path(repo_root) / SAFE_KV260_UIO_TRANSCRIPT_REL_PATH
    if not path.is_file():
        return {
            "path": None,
            "verified": False,
            "sha256": None,
            "blocker": "no_existing_safe_kv260_uio_register_transcript_verified",
            "detail": (
                "Exp 5065 lists UIO devices but does not read a register; older "
                "latency scripts write UIO control registers and are outside this audit-only scope."
            ),
            "prior_transcript_path": str(PRIOR_KV260_TRANSCRIPT_REL_PATH),
        }
    text = path.read_text(encoding="utf-8")
    safe = _safe_uio_transcript_text(text)
    status = {
        "path": str(SAFE_KV260_UIO_TRANSCRIPT_REL_PATH) if safe else None,
        "verified": safe,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "blocker": None if safe else "kv260_uio_register_transcript_not_safe_read_only",
        "detail": "existing safe read-only UIO/register transcript verified"
        if safe
        else "transcript exists but does not declare read-only safe UIO/register access",
        "prior_transcript_path": str(PRIOR_KV260_TRANSCRIPT_REL_PATH),
    }
    return status


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = set(REQUIRED_SCHEMA_FIELDS) - set(artifact)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        return errors
    _expect(errors, artifact.get("schema") == SCHEMA, "schema mismatch")
    _expect(errors, artifact.get("experiment") == EXPERIMENT_NAME, "experiment mismatch")
    _expect(errors, artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    _expect(errors, artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _expect(errors, artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _expect(errors, artifact.get("honest_verdict") in {SUCCESS_VERDICT, PARTIAL_VERDICT}, "honest_verdict mismatch")
    _expect(
        errors,
        str(artifact.get("honest_verdict", "")).startswith(("success_", "complete_")),
        "honest_verdict terminal prefix missing",
    )
    _expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _expect(errors, _duration_number(artifact.get("duration_s")) >= 0.0001, "duration_s below floor")
    _expect(errors, artifact.get("destructive_actions_allowed") is False, "destructive actions are not allowed")
    _expect(errors, artifact.get("destructive_actions_taken") == [], "destructive actions are not allowed")
    _expect(errors, artifact.get("kv260_speedup_claim_allowed") is False, "speedup claim is not allowed")
    _expect(errors, artifact.get("flagged_adversarial") is False, "flagged_adversarial must be false")
    _expect(errors, _no_host_storage(artifact), "forbidden host storage marker")
    _validate_bare_required_fields(errors, artifact)
    _validate_preconditions(errors, artifact)
    _validate_command_probes(errors, artifact)
    _validate_board_matrix(errors, artifact)
    _expect(errors, artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")
    return errors


def _run_polarfire_precheck_bundle(
    *,
    polarfire_ssh_probe: CommandProbe,
    command_runner: CommandRunner,
) -> JsonDict:
    probes: dict[str, CommandProbe | None] = {
        "arch": None,
        "python": None,
        "uptime": None,
        "kernel": None,
    }
    if polarfire_ssh_probe.exit_code != 0:
        return {"ready": False, "probes": probes, "blockers": ["polarfire_ssh_unreachable"]}
    probes["arch"] = command_runner(POLARFIRE_ARCH_COMMAND, 10.0)
    probes["python"] = command_runner(POLARFIRE_PYTHON_COMMAND, 10.0)
    probes["uptime"] = command_runner(POLARFIRE_UPTIME_COMMAND, 10.0)
    probes["kernel"] = command_runner(POLARFIRE_KERNEL_COMMAND, 10.0)
    arch = _observed(probes["arch"]).strip()
    python_version = parse_python_version(_observed(probes["python"]))
    blockers: list[str] = []
    if probes["arch"].exit_code != 0 or arch != "riscv64":
        blockers.append("polarfire_arch_not_riscv64")
    if probes["python"].exit_code != 0 or python_version is None or python_version < (3, 10, 0):
        blockers.append("polarfire_python_precheck_failed")
    return {"ready": not blockers, "probes": probes, "blockers": blockers}


def parse_python_version(version_text: str) -> tuple[int, int, int] | None:
    match = re.search(r"Python\s+(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or "0")


def _command_probes(
    *,
    kv260_probe: CommandProbe,
    gatemate_command_probe: CommandProbe,
    gatemate_detect_probe: CommandProbe | None,
    polarfire_ssh_probe: CommandProbe,
    polarfire_probe_bundle: JsonMap,
) -> JsonDict:
    probes = polarfire_probe_bundle["probes"]
    return {
        "kv260_ssh": kv260_probe.as_dict(),
        "gatemate_detect_command": gatemate_command_probe.as_dict(),
        "gatemate_dirtyjtag_detect": gatemate_detect_probe.as_dict()
        if gatemate_detect_probe is not None
        else None,
        "polarfire_ssh": polarfire_ssh_probe.as_dict(),
        "polarfire_arch": _probe_dict(probes["arch"]),
        "polarfire_python": _probe_dict(probes["python"]),
        "polarfire_uptime": _probe_dict(probes["uptime"]),
        "polarfire_kernel": _probe_dict(probes["kernel"]),
    }


def _kv260_row(
    *,
    ssh_ready: bool,
    terminal_state: str,
    precondition: JsonMap,
    uio_status: JsonMap,
) -> JsonDict:
    return {
        "board": "kv260",
        "precondition_resource": "kv260_ssh",
        "precondition": dict(precondition),
        "ssh_ready": ssh_ready,
        "terminal_state": terminal_state,
        "action_scope": "ssh_only_and_existing_transcript_audit_no_register_touch_by_exp5093",
        "uio_transcript_path": uio_status["path"],
        "uio_transcript_status": dict(uio_status),
        "speedup_claim_allowed": False,
        "destructive_actions_taken": [],
        "limitations": [
            "no_kv260_speedup_claim",
            "no_new_uio_register_read_run_by_this_experiment",
        ],
        "next_safe_step": (
            "Run a separate established safe UIO/register transcript script only "
            "when that register-read path is explicitly in scope."
        ),
    }


def _gatemate_row(
    *,
    command_available: bool,
    detected: bool,
    terminal_state: str,
    precondition: JsonMap,
    detect_probe: CommandProbe | None,
) -> JsonDict:
    detect_log = _observed(detect_probe) if detect_probe is not None else ""
    return {
        "board": "gatemate_a1_dirtyjtag",
        "precondition_resource": "gatemate_detect_command",
        "precondition": dict(precondition),
        "command_available": command_available,
        "detected": detected,
        "terminal_state": terminal_state,
        "action_scope": "detect_only_no_flash_no_program_no_latency_claim",
        "cable_state_inference": _gatemate_cable_state(command_available, detected, detect_probe),
        "destructive_actions_taken": [],
        "evidence": {
            "detected_idcode": _idcode_from_text(detect_log),
            "detect_log": detect_log,
            "detect_exit_code": detect_probe.exit_code if detect_probe is not None else None,
        },
        "limitations": [
            "no_gatemate_flash_or_programming_performed",
            "no_gatemate_sampler_output_verified",
        ],
        "next_safe_step": (
            "Resolve DirtyJTAG/target IDCODE detection before any separate "
            "flash or timing task."
        ),
    }


def _polarfire_row(
    *,
    detected: bool,
    terminal_state: str,
    precondition: JsonMap,
    dispatch_bundle: JsonMap,
) -> JsonDict:
    probes = dispatch_bundle["probes"]
    return {
        "board": "polarfire_soc",
        "precondition_resource": "polarfire_ssh",
        "precondition": dict(precondition),
        "detected": detected,
        "terminal_state": terminal_state,
        "action_scope": "ssh_and_hash_dispatch_preconditions_only_no_scp_no_dispatch",
        "dispatch_precheck_ready": bool(dispatch_bundle["ready"]),
        "dispatch_executed": False,
        "destructive_actions_taken": [],
        "dispatch_precheck": {
            "arch": _observed(probes["arch"]) if probes["arch"] is not None else None,
            "python": _observed(probes["python"]) if probes["python"] is not None else None,
            "uptime": _observed(probes["uptime"]) if probes["uptime"] is not None else None,
            "kernel": _observed(probes["kernel"]) if probes["kernel"] is not None else None,
            "blockers": list(dispatch_bundle["blockers"]),
            "known_safe_dispatch_path": "carnot.hardware.polarfire_dispatch_smoke.check_preconditions",
            "mutating_dispatch_steps_skipped": True,
        },
        "limitations": [
            "no_polarfire_scp_performed",
            "no_polarfire_carnot_dispatch_executed",
            "no_hash_verified_workload_completion_claimed_by_exp5093",
        ],
        "next_safe_step": (
            "Use the established PolarFire dispatch smoke only in a separate "
            "task that explicitly allows file copy and workload dispatch."
        ),
    }


def _precondition_entry(
    resource: str,
    probe: CommandProbe,
    available: bool,
    discipline: str,
) -> JsonDict:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": _round_duration(probe.duration_s),
        "observed": _observed(probe),
        "discipline": discipline,
    }


def _policy_precondition() -> JsonDict:
    return {
        "resource": "destructive_actions_allowed",
        "available": False,
        "command": "policy",
        "exit_code": 0,
        "duration_s": 0.0001,
        "observed": "false",
        "discipline": "explicit_no_destructive_actions",
    }


def _safe_uio_transcript_text(text: str) -> bool:
    lowered = text.lower()
    unsafe_markers = ("write_u32", "loadapp", "flash", "program", "mmap.prot_write")
    return (
        "uio_register_read" in lowered
        and "read_only" in lowered
        and "safe_for_continuity_audit" in lowered
        and not any(marker in lowered for marker in unsafe_markers)
    )


def _kv260_terminal_state(ssh_ready: bool, uio_status: JsonMap) -> str:
    if ssh_ready and uio_status["verified"]:
        return "kv260_ssh_ready_safe_uio_register_transcript_verified_no_speedup_claim"
    if ssh_ready:
        return "blocked_kv260_no_safe_uio_register_transcript_no_speedup_claim"
    return "blocked_kv260_ssh_unreachable_no_uio_register_transcript"


def _gatemate_detected(probe: CommandProbe | None) -> bool:
    if probe is None:
        return False
    observed = probe.combined_output.lower()
    markers = ("gatemate", "gm1a", "idcode", "0x20000001", "colognechip")
    return probe.exit_code == 0 and any(marker in observed for marker in markers)


def _gatemate_terminal_state(
    *,
    command_available: bool,
    detected: bool,
    detect_probe: CommandProbe | None,
) -> str:
    if not command_available:
        return "blocked_gatemate_detect_command_unavailable"
    if detected:
        return "gatemate_detected_idcode_no_flash_terminal"
    if detect_probe is not None and _dirtyjtag_cable_seen(detect_probe.combined_output):
        return "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal"
    return "blocked_gatemate_detect_failed_no_idcode_terminal"


def _gatemate_cable_state(
    command_available: bool,
    detected: bool,
    detect_probe: CommandProbe | None,
) -> str:
    if not command_available:
        return "detect_command_unavailable"
    if detected:
        return "dirtyjtag_cable_and_gatemate_idcode_detected"
    if detect_probe is not None and _dirtyjtag_cable_seen(detect_probe.combined_output):
        return "dirtyjtag_cable_seen_no_gatemate_idcode"
    return "detect_failed_or_no_dirtyjtag_response"


def _polarfire_terminal_state(*, detected: bool, dispatch_ready: bool) -> str:
    if not detected:
        return "blocked_polarfire_ssh_unreachable"
    if dispatch_ready:
        return "polarfire_ssh_hash_dispatch_precheck_ready_no_dispatch_executed"
    return "blocked_polarfire_hash_dispatch_precheck_not_ready"


def _dirtyjtag_cable_seen(text: str) -> bool:
    lowered = text.lower()
    return "jtag frequency" in lowered or "dirtyjtag" in lowered


def _idcode_from_text(text: str) -> str | None:
    match = re.search(r"0x[0-9a-fA-F]+", text)
    return match.group(0) if match else None


def _probe_dict(probe: CommandProbe | None) -> JsonDict | None:
    return probe.as_dict() if probe is not None else None


def _observed(probe: CommandProbe | None) -> str:
    if probe is None:
        return ""
    text = probe.combined_output.strip()
    if text:
        return text.splitlines()[0].strip()
    return f"returncode={probe.exit_code}"


def _round_duration(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return round(max(parsed, 0.0001), 6)


def _duration_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _no_host_storage(payload: JsonMap) -> bool:
    encoded = json.dumps(payload, sort_keys=True, default=str).lower()
    return "mmcblk" not in encoded and "/dev/disk" not in encoded


def _validate_bare_required_fields(errors: list[str], artifact: JsonMap) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        value = artifact.get(field)
        _expect(
            errors,
            not (isinstance(value, Mapping) and set(value) == {"value", "principle"}),
            f"{field} must be a bare value",
        )


def _validate_preconditions(errors: list[str], artifact: JsonMap) -> None:
    preconditions = artifact.get("preconditions_checked")
    _expect(errors, isinstance(preconditions, list) and len(preconditions) == 4, "bad preconditions")
    if not isinstance(preconditions, list) or len(preconditions) != 4:
        return
    expected = [
        ("kv260_ssh", command_to_string(KV260_SSH_COMMAND), "ssh_only_no_host_sd_card"),
        (
            "gatemate_detect_command",
            command_to_string(GATEMATE_COMMAND_AVAILABLE_COMMAND),
            "command_availability_only_no_detect_side_effect",
        ),
        (
            "polarfire_ssh",
            command_to_string(POLARFIRE_SSH_COMMAND),
            "ssh_reachability_before_hash_dispatch_precheck",
        ),
        ("destructive_actions_allowed", "policy", "explicit_no_destructive_actions"),
    ]
    for row, (resource, command, discipline) in zip(preconditions, expected, strict=True):
        _expect(errors, isinstance(row, Mapping), "bad precondition row")
        if isinstance(row, Mapping):
            _expect(errors, row.get("resource") == resource, f"{resource} resource mismatch")
            _expect(errors, row.get("command") == command, f"{resource} command mismatch")
            _expect(errors, row.get("discipline") == discipline, f"{resource} discipline mismatch")
            _expect(errors, isinstance(row.get("available"), bool), f"{resource} availability not bool")
    _expect(errors, preconditions[0].get("available") is artifact.get("kv260_ssh_ready"), "kv260_ssh_ready mismatch")
    _expect(errors, preconditions[2].get("available") is artifact.get("polarfire_detected"), "polarfire_detected mismatch")
    _expect(errors, preconditions[3].get("available") is False, "destructive policy mismatch")


def _validate_command_probes(errors: list[str], artifact: JsonMap) -> None:
    probes = artifact.get("command_probes")
    _expect(errors, isinstance(probes, Mapping), "command_probes must be a dict")
    if not isinstance(probes, Mapping):
        return
    for key in (
        "kv260_ssh",
        "gatemate_detect_command",
        "gatemate_dirtyjtag_detect",
        "polarfire_ssh",
        "polarfire_arch",
        "polarfire_python",
        "polarfire_uptime",
        "polarfire_kernel",
    ):
        _expect(errors, key in probes, f"missing command probe {key}")


def _validate_board_matrix(errors: list[str], artifact: JsonMap) -> None:
    matrix = artifact.get("board_matrix")
    _expect(errors, isinstance(matrix, Mapping), "board_matrix must be a dict")
    if not isinstance(matrix, Mapping):
        return
    _expect(errors, set(matrix) == {"kv260", "gatemate", "polarfire"}, "board_matrix keys mismatch")
    if set(matrix) != {"kv260", "gatemate", "polarfire"}:
        return
    kv260 = matrix["kv260"]
    gatemate = matrix["gatemate"]
    polarfire = matrix["polarfire"]
    _expect(errors, isinstance(kv260, Mapping), "kv260 row invalid")
    _expect(errors, isinstance(gatemate, Mapping), "gatemate row invalid")
    _expect(errors, isinstance(polarfire, Mapping), "polarfire row invalid")
    if not all(isinstance(row, Mapping) for row in (kv260, gatemate, polarfire)):
        return
    _expect(errors, kv260.get("ssh_ready") is artifact.get("kv260_ssh_ready"), "kv260 row mismatch")
    _expect(errors, kv260.get("uio_transcript_path") == artifact.get("kv260_uio_transcript_path"), "kv260 transcript mismatch")
    _expect(errors, kv260.get("speedup_claim_allowed") is False, "kv260 row speedup claim")
    _expect(errors, kv260.get("destructive_actions_taken") == [], "kv260 destructive actions")
    _expect(errors, gatemate.get("detected") is artifact.get("gatemate_detected"), "gatemate row mismatch")
    _expect(errors, gatemate.get("terminal_state") == artifact.get("gatemate_terminal_state"), "gatemate terminal state mismatch")
    _expect(errors, gatemate.get("destructive_actions_taken") == [], "gatemate destructive actions")
    _expect(errors, polarfire.get("detected") is artifact.get("polarfire_detected"), "polarfire row mismatch")
    _expect(
        errors,
        polarfire.get("dispatch_precheck_ready") is artifact.get("polarfire_dispatch_precheck_ready"),
        "polarfire precheck mismatch",
    )
    _expect(errors, polarfire.get("dispatch_executed") is False, "polarfire dispatch executed")
    _expect(errors, polarfire.get("destructive_actions_taken") == [], "polarfire destructive actions")


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kv260_ssh_ready: {artifact['kv260_ssh_ready']}")
    print(f"gatemate_detected: {artifact['gatemate_detected']}")
    print(f"polarfire_dispatch_precheck_ready: {artifact['polarfire_dispatch_precheck_ready']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
