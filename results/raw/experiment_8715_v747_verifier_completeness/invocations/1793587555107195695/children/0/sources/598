"""KV260, GateMate, and PolarFire continuity check for Exp 4040.

This experiment keeps the three hardware boards visible while driving KV260
toward the north-star terminal state: a confirmed Carnot overlay plus a
board-side latency transcript. The code treats SSH reachability as the only
KV260 precondition because a host SD-card device says nothing about the running
board.

Spec refs: REQ-HW-4040, SCENARIO-HW-4040.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time
from typing import Any

from carnot import experiment_3709_kv260_drive_to_terminal_latency_transcript as _latency
from carnot import experiment_3972_hardware_continuity as _hardware_base


EXPERIMENT_ID = 4040
SCHEMA = "carnot.hardware_continuity_kv260_terminal_drive.v1"
SPEC_REFS = ["REQ-HW-4040", "SCENARIO-HW-4040"]
OUTPUT_REL_PATH = Path("results") / "experiment_4040_hardware_continuity.json"
RANDOM_SEED = 4040
INFERENCE_SUBSTRATE = "hardware_smoke"

CommandProbe = _latency.CommandProbe
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], str | None, float], CommandProbe]

KV260_SSH_PRECONDITION = _latency.KV260_SSH_COMMAND
KV260_LISTAPPS_COMMAND = _latency.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = _latency.KV260_LISTAPPS_SUDO_COMMAND
KV260_LOADAPP_COMMAND = _latency.KV260_LOADAPP_COMMAND
KV260_LOADAPP_SUDO_COMMAND = _latency.KV260_LOADAPP_SUDO_COMMAND
KV260_LATENCY_COMMAND = _latency.KV260_LATENCY_COMMAND
BOARD_HARNESS_SOURCE = _latency.BOARD_HARNESS_SOURCE.replace("exp3709", "exp4040")

BOARD_SAMPLE_COUNT = _latency.BOARD_SAMPLE_COUNT
BOARD_SPIN_COUNT = _latency.BOARD_SPIN_COUNT
BOARD_MAX_DEGREE = _latency.BOARD_MAX_DEGREE
BOARD_BETA_FINAL_Q88 = _latency.BOARD_BETA_FINAL_Q88

GATEMATE_DETECT_COMMAND = _hardware_base.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _hardware_base.POLARFIRE_SSH_PRECONDITION
POLARFIRE_CONTINUITY_COMMAND = _hardware_base.POLARFIRE_CONTINUITY_COMMAND

BOARD_NAMES = ("kv260", "gatemate", "polarfire")
VALID_KV260_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4", "carnot_ising")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "per_board_reachability",
    "kv260_overlay_loaded",
    "preconditions_checked",
    "inference_substrate",
    "kv260_reachable",
    "gatemate_reachable",
    "polarfire_reachable",
    "per_board_next_step",
    "per_board_terminal_state",
    "per_board_duration_s",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix naming the observed KV260 overlay/latency gate plus "
        "per-board blockers."
    ),
    "per_board_reachability": (
        "principle: the continuity record that keeps boards visible in retros"
    ),
    "kv260_overlay_loaded": (
        "principle: the KV260 terminal-state gate -- overlay present is the "
        "precondition for a board-latency claim"
    ),
    "preconditions_checked": (
        "principle: records which board accesses were verified before the smoke, "
        "pre-empting the fabricate-when-unreachable failure mode"
    ),
    "inference_substrate": (
        "Must remain hardware_smoke because this artifact records board smoke, "
        "not model inference."
    ),
    "kv260_reachable": "Bare bool for KV260 SSH reachability through kria.",
    "gatemate_reachable": "Bare bool for GateMate reachability through DirtyJTAG detect.",
    "polarfire_reachable": "Bare bool for PolarFire SSH reachability.",
    "per_board_next_step": "Bare dict of the next concrete step for each board.",
    "per_board_terminal_state": (
        "Bare dict of observed terminal-state progress or exact per-board blocker."
    ),
    "per_board_duration_s": "Distinct positive wall-clock timer per board.",
    "duration_s": "Total wall-clock for the hardware-smoke continuity run.",
}


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(
    command: tuple[str, ...],
    stdin: str | None = None,
    timeout_s: float = 60.0,
) -> CommandProbe:  # pragma: no cover - live subprocess boundary
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            input=stdin,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout_s}s"
        return CommandProbe(command, 124, stdout, stderr, time.perf_counter() - started)
    except OSError as exc:
        return CommandProbe(command, 127, "", str(exc), time.perf_counter() - started)
    return CommandProbe(
        command,
        completed.returncode,
        completed.stdout,
        completed.stderr,
        time.perf_counter() - started,
    )


def payload_checksum(payload: dict[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4040 hardware continuity artifact from board transcripts."""
    del repo_root
    started = clock()

    kv260_record, kv260_duration = measure_board(clock, lambda: check_kv260(command_runner))
    gatemate_record, gatemate_duration = measure_board(
        clock,
        lambda: check_gatemate(command_runner),
    )
    polarfire_record, polarfire_duration = measure_board(
        clock,
        lambda: check_polarfire(command_runner),
    )

    reachability = {
        "kv260": bool(kv260_record["reachable"]),
        "gatemate": bool(gatemate_record["reachable"]),
        "polarfire": bool(polarfire_record["reachable"]),
    }
    terminal_state = {
        "kv260": str(kv260_record["state"]),
        "gatemate": str(gatemate_record["state"]),
        "polarfire": str(polarfire_record["state"]),
    }
    next_steps = {
        "kv260": str(kv260_record["next_step"]),
        "gatemate": str(gatemate_record["next_step"]),
        "polarfire": str(polarfire_record["next_step"]),
    }
    durations = {
        "kv260": kv260_duration,
        "gatemate": gatemate_duration,
        "polarfire": polarfire_duration,
    }

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict(kv260_record, gatemate_record, polarfire_record),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_reachable": reachability["kv260"],
        "gatemate_reachable": reachability["gatemate"],
        "polarfire_reachable": reachability["polarfire"],
        "per_board_reachability": reachability,
        "per_board_terminal_state": terminal_state,
        "per_board_next_step": next_steps,
        "per_board_duration_s": durations,
        "duration_s": round_timer(clock() - started),
        "preconditions_checked": [
            kv260_record["precondition"],
            gatemate_record["precondition"],
            polarfire_record["precondition"],
        ],
        "kv260_overlay_loaded": bool(kv260_record["overlay_loaded"]),
        "kv260_loaded_overlay_name": kv260_record["loaded_overlay_name"],
        "kv260_latency_step_taken": bool(kv260_record["latency_step_taken"]),
        "kv260_latency_samples_ms": kv260_record["latency_samples_ms"],
        "kv260_latency_median_ms": kv260_record["latency_median_ms"],
        "kv260_latency_batch_ms": kv260_record["latency_batch_ms"],
        "kv260_latency_step_summary": kv260_record["latency_step_summary"],
        "kv260_state": kv260_record["state"],
        "kv260_command_transcripts": kv260_record["command_transcripts"],
        "gatemate_state": gatemate_record["state"],
        "gatemate_detect_output": gatemate_record["detect_output"],
        "gatemate_command_transcript": gatemate_record["command_transcript"],
        "polarfire_state": polarfire_record["state"],
        "polarfire_continuity_output": polarfire_record["continuity_output"],
        "polarfire_command_transcripts": polarfire_record["command_transcripts"],
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def measure_board(clock: Clock, action: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], float]:
    started = clock()
    record = action()
    return record, round_timer(clock() - started)


def check_kv260(command_runner: CommandRunner) -> dict[str, Any]:
    ssh_probe = command_runner(KV260_SSH_PRECONDITION, None, 10.0)
    command_transcripts = empty_kv260_command_transcripts(ssh_probe)
    if ssh_probe.exit_code != 0:
        return kv260_record(
            reachable=False,
            state="blocked_kv260_unreachable",
            next_step="blocked_kv260_unreachable",
            loaded_overlay_name=None,
            command_transcripts=command_transcripts,
            precondition=precondition_entry("kv260_ssh", ssh_probe),
        )

    initial_overlay = probe_kv260_overlay(
        command_runner,
        command_transcripts,
        non_sudo_key="xmutil_listapps_initial",
        sudo_key="xmutil_listapps_initial_sudo",
    )
    loaded_overlay = initial_overlay
    if loaded_overlay is None:
        load_probe = command_runner(KV260_LOADAPP_COMMAND, None, 120.0)
        command_transcripts["xmutil_loadapp"] = load_probe.as_dict()
        if xmutil_requires_root(load_probe):
            load_sudo_probe = command_runner(KV260_LOADAPP_SUDO_COMMAND, None, 120.0)
            command_transcripts["xmutil_loadapp_sudo"] = load_sudo_probe.as_dict()
        loaded_overlay = probe_kv260_overlay(
            command_runner,
            command_transcripts,
            non_sudo_key="xmutil_listapps_after_load",
            sudo_key="xmutil_listapps_after_load_sudo",
        )

    latency = latency_step(command_runner, command_transcripts) if loaded_overlay else None
    latency_step_taken = latency is not None and latency["step_taken"] is True
    state = kv260_state(loaded_overlay, latency_step_taken)
    return kv260_record(
        reachable=True,
        state=state,
        next_step=kv260_next_step(True, loaded_overlay, latency_step_taken),
        loaded_overlay_name=loaded_overlay,
        latency=latency,
        command_transcripts=command_transcripts,
        precondition=precondition_entry("kv260_ssh", ssh_probe),
    )


def empty_kv260_command_transcripts(ssh_probe: CommandProbe) -> dict[str, Any]:
    return {
        "ssh_true": ssh_probe.as_dict(),
        "xmutil_listapps_initial": None,
        "xmutil_listapps_initial_sudo": None,
        "xmutil_loadapp": None,
        "xmutil_loadapp_sudo": None,
        "xmutil_listapps_after_load": None,
        "xmutil_listapps_after_load_sudo": None,
        "latency_harness": None,
    }


def probe_kv260_overlay(
    command_runner: CommandRunner,
    command_transcripts: dict[str, Any],
    *,
    non_sudo_key: str,
    sudo_key: str,
) -> str | None:
    list_probe = command_runner(KV260_LISTAPPS_COMMAND, None, 30.0)
    command_transcripts[non_sudo_key] = list_probe.as_dict()
    overlay = detect_kv260_overlay(list_probe.combined_output) if list_probe.exit_code == 0 else None
    if overlay is None and xmutil_requires_root(list_probe):
        list_sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, None, 30.0)
        command_transcripts[sudo_key] = list_sudo_probe.as_dict()
        overlay = (
            detect_kv260_overlay(list_sudo_probe.combined_output)
            if list_sudo_probe.exit_code == 0
            else None
        )
    return overlay


def latency_step(
    command_runner: CommandRunner,
    command_transcripts: dict[str, Any],
) -> dict[str, Any] | None:
    latency_probe = command_runner(KV260_LATENCY_COMMAND, BOARD_HARNESS_SOURCE, 1800.0)
    command_transcripts["latency_harness"] = latency_probe.as_dict()
    if latency_probe.exit_code != 0:
        return {
            "step_taken": False,
            "blocked_reason": excerpt(command_text(latency_probe)) or "latency_harness_failed",
            "samples_ms": [],
            "median_ms": None,
            "batch_ms": None,
            "summary": {"exit_code": latency_probe.exit_code},
        }
    try:
        board_payload = _latency.extract_board_payload(latency_probe.stdout)
        _latency.validate_board_payload(board_payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "step_taken": False,
            "blocked_reason": f"invalid_latency_payload: {exc}",
            "samples_ms": [],
            "median_ms": None,
            "batch_ms": None,
            "summary": {"exit_code": latency_probe.exit_code},
        }
    samples = [float(sample) for sample in board_payload["per_sample_wall_ms"]]
    batch_ms = float(board_payload["per_batch_wall_ms"])
    return {
        "step_taken": True,
        "blocked_reason": None,
        "samples_ms": samples,
        "median_ms": float(statistics.median(samples)),
        "batch_ms": batch_ms,
        "summary": {
            "schema": board_payload.get("schema"),
            "sample_count": board_payload.get("sample_count"),
            "selected_uio": board_payload.get("selected_uio"),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
        },
    }


def kv260_record(
    *,
    reachable: bool,
    state: str,
    next_step: str,
    loaded_overlay_name: str | None,
    command_transcripts: dict[str, Any],
    precondition: dict[str, Any],
    latency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency_data = latency or {
        "step_taken": False,
        "blocked_reason": None,
        "samples_ms": [],
        "median_ms": None,
        "batch_ms": None,
        "summary": {},
    }
    return {
        "reachable": reachable,
        "state": state,
        "next_step": next_step,
        "overlay_loaded": loaded_overlay_name is not None,
        "loaded_overlay_name": loaded_overlay_name,
        "latency_step_taken": latency_data["step_taken"],
        "latency_samples_ms": latency_data["samples_ms"],
        "latency_median_ms": latency_data["median_ms"],
        "latency_batch_ms": latency_data["batch_ms"],
        "latency_step_summary": latency_data["summary"],
        "latency_blocked_reason": latency_data["blocked_reason"],
        "precondition": precondition,
        "command_transcripts": command_transcripts,
    }


def check_gatemate(command_runner: CommandRunner) -> dict[str, Any]:
    detect_probe = command_runner(GATEMATE_DETECT_COMMAND, None, 30.0)
    reachable = detects_gatemate(detect_probe)
    state = "reachable_detected_gatemate_idcode" if reachable else "blocked_gatemate_unreachable"
    return {
        "reachable": reachable,
        "state": state,
        "next_step": (
            "gatemate_forward_step_run_minimal_ising_tile_smoke"
            if reachable
            else "blocked_gatemate_unreachable"
        ),
        "detect_output": excerpt(command_text(detect_probe)),
        "precondition": precondition_entry("gatemate_jtag_detect", detect_probe),
        "command_transcript": detect_probe.as_dict(),
    }


def check_polarfire(command_runner: CommandRunner) -> dict[str, Any]:
    ssh_probe = command_runner(POLARFIRE_SSH_PRECONDITION, None, 10.0)
    reachable = ssh_probe.exit_code == 0
    continuity_probe: CommandProbe | None = None
    if reachable:
        continuity_probe = command_runner(POLARFIRE_CONTINUITY_COMMAND, None, 15.0)
    state = (
        "reachable_ssh_continuity_recorded"
        if reachable
        else "blocked_polarfire_unreachable"
    )
    return {
        "reachable": reachable,
        "state": state,
        "next_step": (
            "polarfire_forward_step_run_hash_verified_soft_cpu_dispatch"
            if reachable
            else "blocked_polarfire_unreachable"
        ),
        "continuity_output": (
            excerpt(command_text(continuity_probe))
            if continuity_probe is not None
            else "skipped: polarfire unreachable"
        ),
        "precondition": precondition_entry("polarfire_ssh", ssh_probe),
        "command_transcripts": {
            "ssh_true": ssh_probe.as_dict(),
            "continuity": continuity_probe.as_dict() if continuity_probe is not None else None,
        },
    }


def detect_kv260_overlay(text: str) -> str | None:
    for line in text.splitlines():
        lowered = line.lower()
        active = bool(re.search(r"\b\d+->\d+\b", line)) or "active" in lowered or "running" in lowered
        if not active:
            continue
        for overlay in VALID_KV260_OVERLAYS:
            if overlay in line:
                return overlay
    return None


def xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def detects_gatemate(probe: CommandProbe) -> bool:
    text = command_text(probe).lower()
    return (
        probe.exit_code == 0
        and "idcode" in text
        and ("colognechip" in text or "gatemate" in text or "gm1a" in text)
    )


def kv260_state(loaded_overlay: str | None, latency_step_taken: bool) -> str:
    if loaded_overlay and latency_step_taken:
        return "reachable_overlay_loaded_latency_step_recorded"
    if loaded_overlay:
        return "reachable_overlay_loaded_latency_step_blocked"
    return "reachable_overlay_absent_latency_skipped"


def kv260_next_step(
    reachable: bool,
    loaded_overlay: str | None,
    latency_step_taken: bool,
) -> str:
    if not reachable:
        return "blocked_kv260_unreachable"
    if loaded_overlay and latency_step_taken:
        return "kv260_terminal_state_overlay_loaded_latency_transcript_landed"
    if loaded_overlay:
        return "kv260_forward_step_rerun_terminal_latency_transcript"
    return "kv260_forward_step_load_terminal_overlay_per_north_star_section_3"


def honest_verdict(
    kv260_record_value: dict[str, Any],
    gatemate_record: dict[str, Any],
    polarfire_record: dict[str, Any],
) -> str:
    if not (
        kv260_record_value["reachable"]
        or gatemate_record["reachable"]
        or polarfire_record["reachable"]
    ):
        return "blocked_all_boards_unreachable"
    kv260_phrase = kv260_verdict_phrase(
        bool(kv260_record_value["reachable"]),
        bool(kv260_record_value["overlay_loaded"]),
        bool(kv260_record_value["latency_step_taken"]),
    )
    return (
        f"complete: hardware_continuity_{kv260_phrase}_4040_"
        f"gm{state_token(str(gatemate_record['state']))}_"
        f"pf{state_token(str(polarfire_record['state']))}"
    )


def kv260_verdict_phrase(reachable: bool, overlay_loaded: bool, latency_step_taken: bool) -> str:
    if not reachable:
        return "kv260_unreachable"
    if overlay_loaded and latency_step_taken:
        return "kv260_overlay_loaded_latency_step_landed"
    if overlay_loaded:
        return "kv260_overlay_loaded_latency_step_blocked"
    return "kv260_overlay_absent_latency_skipped"


def precondition_entry(resource: str, probe: CommandProbe) -> dict[str, Any]:
    available = detects_gatemate(probe) if resource == "gatemate_jtag_detect" else probe.exit_code == 0
    return {
        "resource": resource,
        "available": available,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "observed": observed(probe),
        "duration_s": round_timer(probe.duration_s),
        "checked_before_board_operations": True,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("schema must identify the Exp 4040 continuity artifact")
    if artifact.get("experiment") != EXPERIMENT_ID:
        raise ValueError("experiment must be 4040")
    if artifact.get("spec_refs") != SPEC_REFS:
        raise ValueError("spec_refs must reference REQ-HW-4040 and SCENARIO-HW-4040")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 4040")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    validate_principles(artifact)
    for field in (
        "kv260_reachable",
        "gatemate_reachable",
        "polarfire_reachable",
        "kv260_overlay_loaded",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be bool")
    if not isinstance(artifact.get("kv260_latency_step_taken"), bool):
        raise ValueError("kv260 latency step must be bool")
    validate_per_board_reachability(artifact)
    validate_per_board_terminal_state(artifact)
    validate_next_steps(artifact)
    validate_durations(artifact)
    validate_preconditions(artifact)
    text = json.dumps(artifact, sort_keys=True, default=str).lower()
    if "/dev/mmcblk" in text:
        raise ValueError("artifact contains forbidden KV260 host storage marker")
    validate_kv260_overlay_and_latency(artifact)
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact.get("fabric_acceleration_claimed") is not False:
        raise ValueError("fabric_acceleration_claimed must be false")
    validate_verdict(artifact)
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        raise ValueError("field_principles must be a dict")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:
        raise ValueError(f"field_principles missing required fields: {sorted(missing_principles)}")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if principles[field] != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match the Exp 4040 contract")
        value = artifact[field]
        if isinstance(value, dict) and set(value) == {"value", "principle"}:
            raise ValueError(f"{field} must remain a bare value, not a principle wrapper")


def validate_per_board_reachability(artifact: dict[str, Any]) -> None:
    reachability = artifact.get("per_board_reachability")
    if not isinstance(reachability, dict) or set(reachability) != set(BOARD_NAMES):
        raise ValueError("per_board_reachability must be keyed by all boards")
    for board in BOARD_NAMES:
        value = reachability[board]
        if not isinstance(value, bool):
            raise ValueError("per_board_reachability values must be bool")
        if value is not artifact[f"{board}_reachable"]:
            raise ValueError("per_board_reachability must match scalar board reachability")


def validate_per_board_terminal_state(artifact: dict[str, Any]) -> None:
    terminal_state = artifact.get("per_board_terminal_state")
    if not isinstance(terminal_state, dict) or set(terminal_state) != set(BOARD_NAMES):
        raise ValueError("per board terminal state must be keyed by all boards")
    expected = {
        "kv260": artifact["kv260_state"],
        "gatemate": artifact["gatemate_state"],
        "polarfire": artifact["polarfire_state"],
    }
    for board in BOARD_NAMES:
        state = terminal_state[board]
        if not isinstance(state, str) or not state:
            raise ValueError("per board terminal state values must be non-empty strings")
        if state != expected[board]:
            raise ValueError("per board terminal state must match observed terminal state")


def validate_next_steps(artifact: dict[str, Any]) -> None:
    next_steps = artifact.get("per_board_next_step")
    if not isinstance(next_steps, dict) or set(next_steps) != set(BOARD_NAMES):
        raise ValueError("per_board_next_step keys must be kv260, gatemate, and polarfire")
    for board in BOARD_NAMES:
        step = next_steps[board]
        if not isinstance(step, str) or not step:
            raise ValueError("per_board_next_step values must be non-empty strings")
        blocked = f"blocked_{board}_unreachable"
        if bool(artifact[f"{board}_reachable"]):
            if step == blocked:
                raise ValueError(f"reachable board {board} cannot use {blocked}")
        elif step != blocked:
            raise ValueError(f"unreachable board {board} must use {blocked}")


def validate_durations(artifact: dict[str, Any]) -> None:
    if positive_number(artifact.get("duration_s")) <= 0.0:
        raise ValueError("duration_s must be positive")
    durations = artifact.get("per_board_duration_s")
    if not isinstance(durations, dict) or set(durations) != set(BOARD_NAMES):
        raise ValueError("per_board_duration_s keys must be kv260, gatemate, and polarfire")
    values = [positive_number(durations[board]) for board in BOARD_NAMES]
    if any(value <= 0.0 for value in values):
        raise ValueError("per_board_duration_s values must be positive")
    if len(set(values)) != len(values):
        raise ValueError("per_board_duration_s values must be distinct")


def validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        raise ValueError("preconditions_checked must be a list")
    expected = {"kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"}
    seen: set[str] = set()
    for entry in preconditions:
        if not isinstance(entry, dict) or not {"resource", "available"} <= set(entry):
            raise ValueError("preconditions_checked entries must include resource and available")
        if not isinstance(entry["available"], bool):
            raise ValueError("preconditions_checked available must be bool")
        seen.add(str(entry["resource"]))
    if seen != expected:
        raise ValueError("preconditions_checked must include kv260, gatemate, and polarfire")


def validate_kv260_overlay_and_latency(artifact: dict[str, Any]) -> None:
    overlay_loaded = artifact["kv260_overlay_loaded"]
    overlay_name = artifact.get("kv260_loaded_overlay_name")
    latency_taken = artifact["kv260_latency_step_taken"]
    if overlay_loaded:
        if not isinstance(overlay_name, str) or "carnot_ising" not in overlay_name:
            raise ValueError("kv260 overlay name must identify a Carnot overlay")
    elif latency_taken:
        raise ValueError("kv260 latency step requires overlay confirmation")
    if latency_taken:
        samples = artifact.get("kv260_latency_samples_ms")
        if not isinstance(samples, list) or not samples:
            raise ValueError("kv260 latency samples must be present when step is taken")
        if any(float(sample) <= 0.0 for sample in samples):
            raise ValueError("kv260 latency samples must be positive")
        if artifact.get("kv260_latency_median_ms") is None:
            raise ValueError("kv260 latency median must be present when step is taken")
        transcript = artifact.get("kv260_command_transcripts", {}).get("latency_harness")
        if not isinstance(transcript, dict) or transcript.get("exit_code") != 0:
            raise ValueError("kv260 latency transcript must show a successful board command")


def validate_verdict(artifact: dict[str, Any]) -> None:
    verdict = str(artifact.get("honest_verdict", ""))
    any_reachable = any(bool(artifact.get(f"{board}_reachable")) for board in BOARD_NAMES)
    if not verdict.startswith(("success:", "complete:", "blocked_")):
        raise ValueError("honest_verdict must start with a terminal prefix or blocked_")
    if any_reachable:
        if not verdict.startswith(("success:", "complete:")):
            raise ValueError("honest_verdict must use a terminal prefix for reachable boards")
    elif verdict != "blocked_all_boards_unreachable":
        raise ValueError("honest_verdict must be blocked_all_boards_unreachable")


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def command_text(probe: CommandProbe | None) -> str:
    if probe is None:
        return ""
    return probe.combined_output.strip()


def observed(probe: CommandProbe) -> str:
    stdout = probe.stdout.strip()
    stderr = probe.stderr.strip()
    if stdout:
        return excerpt(stdout)
    if stderr:
        return excerpt(stderr)
    return f"returncode={probe.exit_code}"


def excerpt(text: str, limit: int = 500) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def state_token(state: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_")
    return token or "unknown"


def positive_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0.0:
        return 0.0
    return number


def round_timer(value: Any) -> float:
    return round(float(value), 6)


def main() -> None:  # pragma: no cover - CLI wrapper
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"per_board_reachability: {artifact['per_board_reachability']}")
    print(f"kv260_overlay_loaded: {artifact['kv260_overlay_loaded']}")
    print(f"kv260_latency_step_taken: {artifact['kv260_latency_step_taken']}")
    print(f"per_board_next_step: {artifact['per_board_next_step']}")
    print(f"per_board_duration_s: {artifact['per_board_duration_s']}")
    print(f"duration_s: {artifact['duration_s']}")


if __name__ == "__main__":  # pragma: no cover
    main()
