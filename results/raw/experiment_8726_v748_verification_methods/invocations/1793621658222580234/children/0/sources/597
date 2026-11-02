"""Exp 4052 hardware continuity with the KV260 latency transcript terminal gate.

Spec refs: REQ-HW-4052, SCENARIO-HW-4052.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4040_hardware_continuity as _base


EXPERIMENT_ID = 4052
SCHEMA = "carnot.hardware_continuity_kv260_latency_transcript.v1"
SPEC_REFS = ["REQ-HW-4052", "SCENARIO-HW-4052"]
OUTPUT_REL_PATH = Path("results") / "experiment_4052_hardware_continuity.json"
RANDOM_SEED = 4052
INFERENCE_SUBSTRATE = _base.INFERENCE_SUBSTRATE

CommandProbe = _base.CommandProbe
Clock = _base.Clock
CommandRunner = _base.CommandRunner

KV260_SSH_PRECONDITION = _base.KV260_SSH_PRECONDITION
KV260_LISTAPPS_COMMAND = _base.KV260_LISTAPPS_COMMAND
KV260_LISTAPPS_SUDO_COMMAND = _base.KV260_LISTAPPS_SUDO_COMMAND
KV260_LOADAPP_COMMAND = _base.KV260_LOADAPP_COMMAND
KV260_LOADAPP_SUDO_COMMAND = _base.KV260_LOADAPP_SUDO_COMMAND
KV260_LATENCY_COMMAND = _base.KV260_LATENCY_COMMAND
GATEMATE_DETECT_COMMAND = _base.GATEMATE_DETECT_COMMAND
POLARFIRE_SSH_PRECONDITION = _base.POLARFIRE_SSH_PRECONDITION
POLARFIRE_CONTINUITY_COMMAND = _base.POLARFIRE_CONTINUITY_COMMAND

BOARD_NAMES = _base.BOARD_NAMES
BOARD_SAMPLE_COUNT = _base.BOARD_SAMPLE_COUNT
BOARD_SPIN_COUNT = _base.BOARD_SPIN_COUNT
BOARD_MAX_DEGREE = _base.BOARD_MAX_DEGREE
BOARD_BETA_FINAL_Q88 = _base.BOARD_BETA_FINAL_Q88

REQUIRED_ARTIFACT_FIELDS = tuple(
    dict.fromkeys((*_base.REQUIRED_ARTIFACT_FIELDS, "kv260_latency_step_taken"))
)
FIELD_PRINCIPLES = {
    **_base.FIELD_PRINCIPLES,
    "honest_verdict": (
        "Terminal prefix naming the observed KV260 latency-transcript gate plus "
        "per-board blockers."
    ),
    "kv260_latency_step_taken": (
        "principle: the KV260 terminal-state gate -- true only after a successful "
        "board-level latency transcript over SSH"
    ),
}

payload_checksum = _base.payload_checksum
run_command = _base.run_command
command_to_string = _base.command_to_string

BOARD_HARNESS_SOURCE = rf'''#!/usr/bin/env python3
import glob
import json
import mmap
import os
import struct
import time

DEFAULT_MAP_SIZE = 0x1000
SAMPLER_BASE_ADDR = 0xA0000000
SAMPLE_COUNT = {BOARD_SAMPLE_COUNT}
SPIN_COUNT = {BOARD_SPIN_COUNT}
MAX_DEGREE = {BOARD_MAX_DEGREE}
BETA_FINAL_Q88 = {BOARD_BETA_FINAL_Q88}


def _read_text(path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return default


def _parse_int(text, default=0):
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        return default


def _discover_uio_devices():
    devices = []
    for sys_path in sorted(glob.glob("/sys/class/uio/uio*")):
        name = os.path.basename(sys_path)
        addr_text = _read_text(os.path.join(sys_path, "maps/map0/addr"))
        size_text = _read_text(os.path.join(sys_path, "maps/map0/size"))
        map_name = _read_text(os.path.join(sys_path, "maps/map0/name"))
        devices.append(
            {{
                "path": "/dev/" + name,
                "addr": _parse_int(addr_text),
                "addr_hex": addr_text,
                "size": _parse_int(size_text, DEFAULT_MAP_SIZE),
                "size_hex": size_text,
                "name": map_name,
            }}
        )
    return devices


def _select_sampler_uio(devices):
    for dev in devices:
        if dev["addr"] == SAMPLER_BASE_ADDR:
            return dev
    for dev in devices:
        lowered = dev["name"].lower()
        if "ising" in lowered or "sampler" in lowered:
            return dev
    if devices:
        return devices[0]
    raise RuntimeError("no UIO device candidates found")


def _candidate_lengths(dev):
    page = int(os.sysconf("SC_PAGE_SIZE") or DEFAULT_MAP_SIZE)
    size = int(dev.get("size") or page)
    values = [min(max(size, 4), page), page, size]
    unique = []
    for value in values:
        if value > 0 and value not in unique:
            unique.append(value)
    return unique


def _open_map(dev):
    fd = os.open(dev["path"], os.O_RDWR | os.O_SYNC)
    last_error = None
    try:
        for length in _candidate_lengths(dev):
            try:
                return fd, mmap.mmap(
                    fd,
                    length,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    flags=mmap.MAP_SHARED,
                )
            except OSError as exc:
                last_error = exc
        raise last_error or OSError("unable to mmap UIO map0")
    except Exception:
        os.close(fd)
        raise


def _read_u32(mm, offset):
    return struct.unpack_from("<I", mm, offset)[0]


def main():
    print("BOARD_HARNESS_START exp4052", flush=True)
    devices = _discover_uio_devices()
    sampler_uio = _select_sampler_uio(devices)
    fd, mm = _open_map(sampler_uio)
    samples = []
    last_value = 0
    batch_start_ns = time.perf_counter_ns()
    try:
        for _ in range(SAMPLE_COUNT):
            start_ns = time.perf_counter_ns()
            last_value = _read_u32(mm, 0)
            end_ns = time.perf_counter_ns()
            samples.append(max((end_ns - start_ns) / 1_000_000.0, 0.000001))
    finally:
        mm.close()
        os.close(fd)
    batch_ms = max((time.perf_counter_ns() - batch_start_ns) / 1_000_000.0, 0.000001)
    print(
        json.dumps(
            {{
                "schema": "carnot.kv260.uio_register_latency_transcript.v1",
                "sample_count": SAMPLE_COUNT,
                "per_sample_wall_ms": samples,
                "per_batch_wall_ms": batch_ms,
                "fixed_compute_budget": {{
                    "spin_count": SPIN_COUNT,
                    "sample_count": SAMPLE_COUNT,
                    "max_degree": MAX_DEGREE,
                    "beta_final_q88": BETA_FINAL_Q88,
                    "trigger_mode": "uio_register_read_once_per_sample",
                }},
                "selected_uio": sampler_uio["path"],
                "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
                "read_offset_hex": "0x0",
                "final_register_value_hex": hex(int(last_value)),
            }},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    """Build the Exp 4052 artifact while preserving the existing board probes."""
    previous_harness = _base.BOARD_HARNESS_SOURCE
    previous_detector = _base.detect_kv260_overlay
    _base.BOARD_HARNESS_SOURCE = BOARD_HARNESS_SOURCE
    _base.detect_kv260_overlay = detect_kv260_overlay
    try:
        artifact = _base.build_artifact(
            repo_root=repo_root,
            command_runner=command_runner,
            clock=clock,
        )
    finally:
        _base.BOARD_HARNESS_SOURCE = previous_harness
        _base.detect_kv260_overlay = previous_detector
    stamp_4052_artifact(artifact)
    validate_artifact(artifact)
    return artifact


def detect_kv260_overlay(text: str) -> str | None:
    for overlay in ("carnot_ising_v2_n64", "carnot_ising_v4", "carnot_ising"):
        for line in text.splitlines():
            lowered = line.lower()
            if overlay in line and (
                "id_ok" in lowered or "active" in lowered or "running" in lowered or "->" in line
            ):
                return overlay
    return None


def stamp_4052_artifact(artifact: dict[str, Any]) -> None:
    artifact["schema"] = SCHEMA
    artifact["experiment"] = EXPERIMENT_ID
    artifact["spec_refs"] = list(SPEC_REFS)
    artifact["random_seed"] = RANDOM_SEED
    artifact["field_principles"] = dict(FIELD_PRINCIPLES)
    artifact["kv260_state"] = kv260_state(
        bool(artifact["kv260_reachable"]),
        bool(artifact["kv260_overlay_loaded"]),
        bool(artifact["kv260_latency_step_taken"]),
    )
    artifact["per_board_terminal_state"]["kv260"] = artifact["kv260_state"]
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def kv260_state(reachable: bool, overlay_loaded: bool, latency_step_taken: bool) -> str:
    if not reachable:
        return "blocked_kv260_unreachable"
    if overlay_loaded and latency_step_taken:
        return "reachable_overlay_loaded_latency_transcript_recorded"
    if overlay_loaded:
        return "reachable_overlay_loaded_latency_transcript_blocked"
    return "reachable_overlay_absent_latency_skipped"


def honest_verdict(artifact: dict[str, Any]) -> str:
    if not any(bool(artifact[f"{board}_reachable"]) for board in BOARD_NAMES):
        return "blocked_all_boards_unreachable"
    return (
        f"complete: hardware_continuity_{kv260_verdict_phrase(artifact)}_4052_"
        f"gm{_base.state_token(str(artifact['gatemate_state']))}_"
        f"pf{_base.state_token(str(artifact['polarfire_state']))}"
    )


def kv260_verdict_phrase(artifact: dict[str, Any]) -> str:
    if not bool(artifact["kv260_reachable"]):
        return "kv260_unreachable"
    if bool(artifact["kv260_overlay_loaded"]) and bool(artifact["kv260_latency_step_taken"]):
        return "kv260_latency_transcript_landed"
    if bool(artifact["kv260_overlay_loaded"]):
        return "kv260_latency_transcript_blocked"
    return "kv260_overlay_absent_latency_skipped"


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify the Exp 4052 artifact")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4052")
    _require(
        artifact.get("spec_refs") == SPEC_REFS,
        "spec_refs must reference REQ-HW-4052 and SCENARIO-HW-4052",
    )
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4052")
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    validate_principles(artifact)
    validate_boards(artifact)
    validate_preconditions(artifact)
    validate_kv260_gate(artifact)
    _require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be hardware_smoke",
    )
    _require(
        artifact.get("fabric_acceleration_claimed") is False,
        "fabric_acceleration_claimed must be false",
    )
    _require("mmcblk" not in json.dumps(artifact, sort_keys=True, default=str).lower(), "")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum does not match artifact content",
    )


def validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be a dict")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in principles, f"field_principles missing {field}")
        _require(principles[field] == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )


def validate_boards(artifact: dict[str, Any]) -> None:
    reachability = artifact.get("per_board_reachability")
    terminal_state = artifact.get("per_board_terminal_state")
    next_steps = artifact.get("per_board_next_step")
    durations = artifact.get("per_board_duration_s")
    for mapping, name in (
        (reachability, "per_board_reachability"),
        (terminal_state, "per_board_terminal_state"),
        (next_steps, "per_board_next_step"),
        (durations, "per_board_duration_s"),
    ):
        _require(isinstance(mapping, dict), f"{name} must be a dict")
        _require(set(mapping) == set(BOARD_NAMES), f"{name} must be keyed by all boards")
    for board in BOARD_NAMES:
        _require(isinstance(reachability[board], bool), "reachability values must be bool")
        _require(
            reachability[board] is artifact[f"{board}_reachable"],
            "reachability must match scalar fields",
        )
        _require(isinstance(terminal_state[board], str) and terminal_state[board], "")
        _require(isinstance(next_steps[board], str) and next_steps[board], "")
        _require(float(durations[board]) > 0.0, "per-board timers must be positive")
        blocked = f"blocked_{board}_unreachable"
        _require(
            (next_steps[board] != blocked) if reachability[board] else (next_steps[board] == blocked),
            "unreachable boards must use blocked next steps",
        )
    _require(len({float(durations[board]) for board in BOARD_NAMES}) == len(BOARD_NAMES), "")
    _require(float(artifact["duration_s"]) > 0.0, "duration_s must be positive")


def validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    seen = {str(entry.get("resource")) for entry in preconditions if isinstance(entry, dict)}
    _require(seen == {"kv260_ssh", "gatemate_jtag_detect", "polarfire_ssh"}, "")
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require({"resource", "available"} <= set(entry), "")
        _require(isinstance(entry["available"], bool), "precondition availability must be bool")


def validate_kv260_gate(artifact: dict[str, Any]) -> None:
    _require(isinstance(artifact["kv260_overlay_loaded"], bool), "overlay gate must be bool")
    _require(isinstance(artifact["kv260_latency_step_taken"], bool), "latency gate must be bool")
    if artifact["kv260_overlay_loaded"]:
        _require(
            "carnot_ising" in str(artifact.get("kv260_loaded_overlay_name")),
            "overlay name must identify Carnot",
        )
    _require(
        not artifact["kv260_latency_step_taken"] or artifact["kv260_overlay_loaded"],
        "latency transcript requires overlay confirmation",
    )
    if artifact["kv260_latency_step_taken"]:
        samples = artifact.get("kv260_latency_samples_ms")
        transcript = artifact.get("kv260_command_transcripts", {}).get("latency_harness")
        _require(isinstance(samples, list) and len(samples) >= 30, "latency samples missing")
        _require(all(float(sample) > 0.0 for sample in samples), "latency samples must be positive")
        _require(artifact.get("kv260_latency_median_ms") is not None, "latency median missing")
        _require(isinstance(transcript, dict) and transcript.get("exit_code") == 0, "")
        _require("kv260_latency_transcript_landed_4052" in artifact["honest_verdict"], "")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)  # pragma: no cover


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


def main() -> None:  # pragma: no cover
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
