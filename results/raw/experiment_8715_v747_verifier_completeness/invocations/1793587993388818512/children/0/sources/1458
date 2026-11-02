#!/usr/bin/env python3
"""Exp 5079: non-destructive board continuity matrix.

Spec refs: REQ-HW-5079, SCENARIO-HW-5079.

This experiment keeps the three active hardware tracks visible without turning
reachability into a benchmark claim. KV260 is checked only by SSH and then
audited against the existing Exp 5065 transcript-backed testbench packet.
GateMate and PolarFire run only non-mutating detection/state commands.
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

from carnot import experiment_5065_kv260_testbench_timing_packet as exp5065


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandProbe"]

EXPERIMENT_ID = 5079
EXPERIMENT_NAME = "experiment_5079_board_continuity_matrix"
SCHEMA = "carnot.experiment_5079_board_continuity_matrix.v466"
OUTPUT_REL_PATH = Path("results") / "experiment_5079_board_continuity_matrix_v466.json"
PRIOR_KV260_ARTIFACT_REL_PATH = Path("results") / "experiment_5065_kv260_testbench_timing_packet.json"
PRIOR_KV260_TRANSCRIPT_REL_PATH = (
    Path("results") / "experiment_5065_kv260_testbench_timing_packet.transcript.jsonl"
)
SPEC_REFS = ["REQ-HW-5079", "SCENARIO-HW-5079"]
RANDOM_SEED = 5079
INFERENCE_SUBSTRATE = "hardware_precheck_and_upstream_artifact_audit"
HONEST_VERDICT = "success_board_continuity_matrix_written_no_speedup_claim"

KV260_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
POLARFIRE_UPTIME_COMMAND = ("ssh", "polarfire", "uptime")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "kv260_ssh_ready",
    "kv260_prior_transcript_verified",
    "kv260_speedup_claim_allowed",
    "gatemate_detected",
    "gatemate_terminal_state",
    "polarfire_detected",
    "polarfire_terminal_state",
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
    "preconditions_checked",
    "command_probes",
    "kv260_prior_artifact",
    "kv260_prior_summary",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal-prefixed verdict for a written board matrix with no speedup claim."
    },
    "duration_s": {
        "principle": "measured wall-clock time for live prechecks plus upstream artifact audit."
    },
    "inference_substrate": {
        "principle": "declares hardware_precheck_and_upstream_artifact_audit because this mixes live reachability with checked-in evidence."
    },
    "kv260_ssh_ready": {
        "principle": "true only when the exact BatchMode SSH precondition to kria exits zero; never a host block-device check."
    },
    "kv260_prior_transcript_verified": {
        "principle": "true only when Exp 5065 artifact validation and transcript SHA-256 verification both pass."
    },
    "kv260_speedup_claim_allowed": {
        "principle": "must remain false because the KV260 evidence is SSH reachability plus an upstream Python testbench packet."
    },
    "gatemate_detected": {
        "principle": "true only from non-destructive DirtyJTAG/openFPGALoader detection with GateMate/GM1Ax/IDCODE evidence."
    },
    "gatemate_terminal_state": {
        "principle": "names the current GateMate state without flashing, timing, or claiming a Carnot tile is deployed."
    },
    "polarfire_detected": {
        "principle": "true only from non-destructive PolarFire BatchMode SSH reachability."
    },
    "polarfire_terminal_state": {
        "principle": "names the current PolarFire SSH state without running Carnot dispatch."
    },
    "destructive_actions_taken": {
        "principle": "must be an empty list; this matrix is detection and audit only."
    },
    "board_matrix": {
        "principle": "one machine-readable row per active board preserving reachability, evidence, limitations, and safe next step."
    },
    "flagged_adversarial": {
        "principle": "false only when schema checks pass and the matrix preserves no-speedup/no-destructive-action boundaries."
    },
}


class CommandProbe:
    """Captured command result for a non-destructive hardware precheck.

    The record stores stdout, stderr, exit code, and duration so the continuity
    artifact can be audited without rerunning board commands. This is important
    for hardware work because a missing board and a reachable board have very
    different next steps, but neither should become an unverified speedup claim.
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


def read_json_object(path: Path) -> JsonDict:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def audit_prior_kv260(repo_root: str | Path) -> tuple[JsonDict, JsonDict]:
    """Verify Exp 5065's artifact/transcript pair and extract claim limits."""

    root = Path(repo_root)
    artifact_path = root / PRIOR_KV260_ARTIFACT_REL_PATH
    transcript_path = root / PRIOR_KV260_TRANSCRIPT_REL_PATH
    payload = read_json_object(artifact_path)
    transcript_text = _read_text(transcript_path)
    errors: list[str] = []
    if not payload:
        errors.append("prior_exp5065_artifact_missing_or_invalid_json")
    if transcript_text is None:
        errors.append("prior_exp5065_transcript_missing")

    claimed_hash = str(payload.get("transcript_sha256") or "")
    observed_hash = hashlib.sha256((transcript_text or "").encode()).hexdigest()
    transcript_hash_verified = bool(transcript_text is not None and claimed_hash == observed_hash)
    if transcript_text is not None and not transcript_hash_verified:
        errors.append("prior_exp5065_transcript_sha256_mismatch")

    if payload and transcript_text is not None:
        try:
            exp5065.validate_artifact(payload, transcript_text=transcript_text)
        except ValueError as exc:
            errors.append(f"prior_exp5065_schema_validation_failed:{exc}")

    parity = _prior_parity(payload)
    timing_packet_present = bool(
        payload.get("timing_ratio_packet_built") is True
        and isinstance(payload.get("timing_ratio_packet"), Mapping)
    )
    verified = bool(not errors and parity == "match" and timing_packet_present)
    limitations = [
        "prior_exp5065_is_ssh_attached_python_testbench_evidence_not_fabric_latency",
        "no_general_fpga_speedup_claim",
        "timing_ratio_includes_cpu_reference_and_ssh_command_context_not_a_board_speedup",
    ]
    if not verified:
        limitations.append("prior_exp5065_transcript_or_parity_unverified")

    artifact_record = {
        "artifact_path": str(PRIOR_KV260_ARTIFACT_REL_PATH),
        "transcript_path": str(PRIOR_KV260_TRANSCRIPT_REL_PATH),
        "artifact_present": artifact_path.is_file(),
        "transcript_present": transcript_path.is_file(),
        "artifact_sha256": file_sha256(artifact_path),
        "transcript_sha256": file_sha256(transcript_path),
        "claimed_transcript_sha256": claimed_hash or None,
        "transcript_sha256_verified": transcript_hash_verified,
    }
    summary = {
        "verified": verified,
        "errors": errors,
        "prior_honest_verdict": str(payload.get("honest_verdict") or ""),
        "prior_inference_substrate": str(payload.get("inference_substrate") or ""),
        "loaded_overlay": payload.get("loaded_overlay"),
        "overlay_loaded": payload.get("overlay_loaded"),
        "uio_devices": list(payload.get("uio_devices") or []),
        "cpu_reference_ok": payload.get("cpu_reference_ok"),
        "kv260_result_ok": payload.get("kv260_result_ok"),
        "cpu_board_parity": parity,
        "timing_packet_present": timing_packet_present,
        "timing_ratio_packet": _prior_timing_packet(payload),
        "transcript_sha256_verified": transcript_hash_verified,
        "local_claim_scope": str(payload.get("local_claim_scope") or ""),
        "limitations": limitations,
    }
    return artifact_record, summary


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Build the Exp 5079 matrix from live prechecks and Exp 5065 evidence."""

    started = clock()
    kv260_probe = command_runner(KV260_PRECONDITION_COMMAND, 10.0)
    gatemate_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    polarfire_probe = command_runner(POLARFIRE_PRECONDITION_COMMAND, 10.0)
    polarfire_uptime_probe = (
        command_runner(POLARFIRE_UPTIME_COMMAND, 20.0) if polarfire_probe.exit_code == 0 else None
    )

    kv260_prior_artifact, kv260_prior_summary = audit_prior_kv260(repo_root)
    kv260_ssh_ready = kv260_probe.exit_code == 0
    gatemate_detected = _gatemate_detected(gatemate_probe)
    polarfire_detected = polarfire_probe.exit_code == 0
    preconditions = [
        _precondition_entry(
            "kv260_ssh",
            kv260_probe,
            kv260_ssh_ready,
            "ssh_only_no_host_sd_card",
        ),
        _precondition_entry(
            "gatemate_dirtyjtag_detect",
            gatemate_probe,
            gatemate_detected,
            "detect_only_no_flash_no_program",
        ),
        _precondition_entry(
            "polarfire_ssh",
            polarfire_probe,
            polarfire_detected,
            "ssh_reachability_only_no_dispatch",
        ),
    ]
    gatemate_terminal_state = _gatemate_terminal_state(gatemate_detected)
    polarfire_terminal_state = _polarfire_terminal_state(polarfire_detected)
    kv260_prior_verified = bool(kv260_prior_summary["verified"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": HONEST_VERDICT,
        "duration_s": _round_duration(clock() - started),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "kv260_ssh_ready": kv260_ssh_ready,
        "kv260_prior_transcript_verified": kv260_prior_verified,
        "kv260_speedup_claim_allowed": False,
        "gatemate_detected": gatemate_detected,
        "gatemate_terminal_state": gatemate_terminal_state,
        "polarfire_detected": polarfire_detected,
        "polarfire_terminal_state": polarfire_terminal_state,
        "destructive_actions_taken": [],
        "flagged_adversarial": False,
        "preconditions_checked": preconditions,
        "command_probes": {
            "kv260_ssh": kv260_probe.as_dict(),
            "gatemate_dirtyjtag_detect": gatemate_probe.as_dict(),
            "polarfire_ssh": polarfire_probe.as_dict(),
            "polarfire_uptime": (
                polarfire_uptime_probe.as_dict() if polarfire_uptime_probe is not None else None
            ),
        },
        "kv260_prior_artifact": kv260_prior_artifact,
        "kv260_prior_summary": kv260_prior_summary,
        "board_matrix": {
            "kv260": _kv260_row(kv260_ssh_ready, preconditions[0], kv260_prior_summary),
            "gatemate": _gatemate_row(gatemate_detected, gatemate_terminal_state, preconditions[1]),
            "polarfire": _polarfire_row(
                polarfire_detected,
                polarfire_terminal_state,
                preconditions[2],
                polarfire_uptime_probe,
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
    _expect(errors, artifact.get("honest_verdict") == HONEST_VERDICT, "honest_verdict mismatch")
    _expect(
        errors,
        str(artifact.get("honest_verdict", "")).startswith(("success_", "success:")),
        "honest_verdict terminal prefix missing",
    )
    _expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _expect(errors, _duration_number(artifact.get("duration_s")) >= 0.0001, "duration_s below floor")
    _expect(errors, artifact.get("kv260_speedup_claim_allowed") is False, "speedup claim is not allowed")
    _expect(errors, artifact.get("destructive_actions_taken") == [], "destructive actions are not allowed")
    _expect(errors, artifact.get("flagged_adversarial") is False, "flagged_adversarial must be false")
    _expect(errors, _no_host_storage(artifact), "forbidden host storage marker")
    _validate_bare_required_fields(errors, artifact)
    _validate_preconditions(errors, artifact)
    _validate_command_probes(errors, artifact)
    _validate_board_matrix(errors, artifact)
    _expect(
        errors,
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "bad checksum",
    )
    return errors


def _kv260_row(ssh_ready: bool, precondition: JsonMap, prior: JsonMap) -> JsonDict:
    prior_verified = bool(prior.get("verified"))
    if ssh_ready and prior_verified:
        terminal_state = "kv260_ssh_ready_prior_transcript_verified_no_speedup_claim"
    elif ssh_ready:
        terminal_state = "blocked_kv260_prior_transcript_unverified_no_speedup_claim"
    elif prior_verified:
        terminal_state = "blocked_kv260_ssh_unreachable_prior_transcript_verified"
    else:
        terminal_state = "blocked_kv260_ssh_unreachable_prior_transcript_unverified"
    return {
        "board": "kv260",
        "precondition_resource": "kv260_ssh",
        "precondition": dict(precondition),
        "ssh_ready": ssh_ready,
        "terminal_state": terminal_state,
        "speedup_claim_allowed": False,
        "destructive_actions_taken": [],
        "evidence": {
            "prior_artifact_path": str(PRIOR_KV260_ARTIFACT_REL_PATH),
            "prior_transcript_path": str(PRIOR_KV260_TRANSCRIPT_REL_PATH),
            "prior_transcript_verified": prior_verified,
            "transcript_sha256_verified": prior.get("transcript_sha256_verified") is True,
            "cpu_board_parity": prior.get("cpu_board_parity"),
            "timing_packet_present": prior.get("timing_packet_present") is True,
            "loaded_overlay": prior.get("loaded_overlay"),
            "uio_devices": list(prior.get("uio_devices") or []),
            "prior_honest_verdict": prior.get("prior_honest_verdict"),
            "timing_ratio_packet": prior.get("timing_ratio_packet"),
        },
        "limitations": list(prior.get("limitations") or []),
        "next_safe_step": (
            "Only a separate board-command transcript using /dev/uio register access "
            "may support a future KV260 speedup or latency claim."
        ),
    }


def _gatemate_row(detected: bool, terminal_state: str, precondition: JsonMap) -> JsonDict:
    observed = str(precondition.get("observed") or "")
    return {
        "board": "gatemate_a1",
        "precondition_resource": "gatemate_dirtyjtag_detect",
        "precondition": dict(precondition),
        "detected": detected,
        "terminal_state": terminal_state,
        "action_scope": "detect_only_no_flash_no_program_no_latency_claim",
        "speedup_claim_allowed": False,
        "destructive_actions_taken": [],
        "evidence": {
            "detected_idcode": _idcode_from_text(observed),
            "detect_observed": observed,
        },
        "limitations": [
            "no_carnot_tile_flashed_or_timed_by_this_experiment",
            "no_host_visible_sampler_output_verified_by_this_experiment",
        ],
        "next_safe_step": (
            "Use a separate established GateMate Carnot tile flash/timing script only "
            "after a known-good bitstream and host-visible output path are in scope."
        ),
    }


def _polarfire_row(
    detected: bool,
    terminal_state: str,
    precondition: JsonMap,
    uptime_probe: CommandProbe | None,
) -> JsonDict:
    return {
        "board": "polarfire_soc",
        "precondition_resource": "polarfire_ssh",
        "precondition": dict(precondition),
        "detected": detected,
        "terminal_state": terminal_state,
        "action_scope": "ssh_precheck_and_optional_uptime_only_no_dispatch",
        "speedup_claim_allowed": False,
        "destructive_actions_taken": [],
        "state_probe": _state_probe("uptime", uptime_probe),
        "limitations": [
            "no_carnot_dispatch_executed_by_this_experiment",
            "no_hash_verified_workload_completion_recorded_by_this_experiment",
        ],
        "next_safe_step": (
            "Run a separate hash-verified Carnot dispatch only when that path is the "
            "explicit task scope."
        ),
    }


def _state_probe(state_type: str, probe: CommandProbe | None) -> JsonDict:
    if probe is None:
        return {"captured": False, "state_type": state_type, "reason": "precondition_not_available"}
    return {
        "captured": True,
        "succeeded": probe.exit_code == 0,
        "state_type": state_type,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": _round_duration(probe.duration_s),
        "observed": _observed(probe),
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


def _prior_parity(payload: JsonMap) -> str:
    evidence = payload.get("structured_testbench_evidence")
    parity = evidence.get("parity") if isinstance(evidence, Mapping) else {}
    mismatches = parity.get("mismatches") if isinstance(parity, Mapping) else None
    if (
        payload.get("cpu_reference_ok") is True
        and payload.get("kv260_result_ok") is True
        and mismatches == []
    ):
        return "match"
    if payload:
        return "not_verified"
    return "missing"


def _prior_timing_packet(payload: JsonMap) -> JsonDict | None:
    packet = payload.get("timing_ratio_packet")
    if not isinstance(packet, Mapping):
        return None
    return {
        "workload_name": packet.get("workload_name"),
        "n_variables": packet.get("n_variables"),
        "iterations": packet.get("iterations"),
        "parity_match": packet.get("parity_match"),
        "cpu_wall_clock_s": packet.get("cpu_wall_clock_s"),
        "kv260_command_wall_clock_s": packet.get("kv260_command_wall_clock_s"),
        "kv260_board_reported_workload_s": packet.get("kv260_board_reported_workload_s"),
        "ratio_claim_scope": packet.get("ratio_claim_scope"),
    }


def _gatemate_detected(probe: CommandProbe) -> bool:
    observed = probe.combined_output.lower()
    markers = ("gatemate", "gm1a", "idcode", "0x20000001", "colognechip")
    return probe.exit_code == 0 and any(marker in observed for marker in markers)


def _gatemate_terminal_state(detected: bool) -> str:
    if detected:
        return "gatemate_detected_toolchain_unblocked_no_carnot_tile_flashed_or_timed"
    return "blocked_gatemate_usb_undetected"


def _polarfire_terminal_state(detected: bool) -> str:
    if detected:
        return "polarfire_ssh_attached_no_carnot_dispatch_executed"
    return "blocked_polarfire_ssh_unreachable"


def _idcode_from_text(text: str) -> str | None:
    match = re.search(r"0x[0-9a-fA-F]+", text)
    return match.group(0) if match else None


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    if text:
        return text.splitlines()[0].strip()
    return f"returncode={probe.exit_code}"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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
    _expect(errors, isinstance(preconditions, list) and len(preconditions) == 3, "bad preconditions")
    if not isinstance(preconditions, list) or len(preconditions) != 3:
        return
    expected = [
        ("kv260_ssh", command_to_string(KV260_PRECONDITION_COMMAND), "ssh_only_no_host_sd_card"),
        (
            "gatemate_dirtyjtag_detect",
            command_to_string(GATEMATE_DETECT_COMMAND),
            "detect_only_no_flash_no_program",
        ),
        (
            "polarfire_ssh",
            command_to_string(POLARFIRE_PRECONDITION_COMMAND),
            "ssh_reachability_only_no_dispatch",
        ),
    ]
    for row, (resource, command, discipline) in zip(preconditions, expected, strict=True):
        _expect(errors, isinstance(row, Mapping), "bad precondition row")
        if not isinstance(row, Mapping):
            continue
        _expect(errors, row.get("resource") == resource, f"{resource} resource mismatch")
        _expect(errors, row.get("command") == command, f"{resource} command mismatch")
        _expect(errors, row.get("discipline") == discipline, f"{resource} discipline mismatch")
        _expect(errors, isinstance(row.get("available"), bool), f"{resource} availability not bool")
    _expect(errors, preconditions[0].get("available") is artifact.get("kv260_ssh_ready"), "kv260_ssh_ready mismatch")
    _expect(errors, preconditions[1].get("available") is artifact.get("gatemate_detected"), "gatemate_detected mismatch")
    _expect(errors, preconditions[2].get("available") is artifact.get("polarfire_detected"), "polarfire_detected mismatch")


def _validate_command_probes(errors: list[str], artifact: JsonMap) -> None:
    probes = artifact.get("command_probes")
    _expect(errors, isinstance(probes, Mapping), "command_probes must be a dict")
    if not isinstance(probes, Mapping):
        return
    for key in ("kv260_ssh", "gatemate_dirtyjtag_detect", "polarfire_ssh", "polarfire_uptime"):
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
    _expect(
        errors,
        bool(kv260.get("evidence", {}).get("prior_transcript_verified"))
        is artifact.get("kv260_prior_transcript_verified"),
        "kv260 prior verification mismatch",
    )
    _expect(errors, kv260.get("speedup_claim_allowed") is False, "kv260 row speedup claim")
    _expect(errors, kv260.get("destructive_actions_taken") == [], "kv260 destructive actions")
    _expect(errors, gatemate.get("detected") is artifact.get("gatemate_detected"), "gatemate row mismatch")
    _expect(
        errors,
        gatemate.get("terminal_state") == artifact.get("gatemate_terminal_state"),
        "gatemate terminal state mismatch",
    )
    _expect(errors, gatemate.get("destructive_actions_taken") == [], "gatemate destructive actions")
    _expect(errors, polarfire.get("detected") is artifact.get("polarfire_detected"), "polarfire row mismatch")
    _expect(
        errors,
        polarfire.get("terminal_state") == artifact.get("polarfire_terminal_state"),
        "polarfire terminal state mismatch",
    )
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
    print(f"polarfire_detected: {artifact['polarfire_detected']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
