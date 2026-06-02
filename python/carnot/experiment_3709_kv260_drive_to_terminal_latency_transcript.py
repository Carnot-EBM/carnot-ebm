"""Build the Exp 3709 KV260 terminal-candidate latency transcript artifact.

Spec refs: REQ-HW-3709, SCENARIO-HW-3709.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time
from typing import Any

EXPERIMENT_ID = "exp3709"
TASK_ID = "exp3709-kv260-drive-to-terminal-latency-transcript"
SCHEMA = "carnot.kv260_terminal_latency_transcript.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"
)
RANDOM_SEED = 3709

KV260_SSH_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
KV260_LISTAPPS_COMMAND = ("ssh", "kria", "xmutil listapps")
KV260_LOADAPP_COMMAND = ("ssh", "kria", "xmutil loadapp carnot_ising_v2_n64")
KV260_LISTAPPS_SUDO_COMMAND = ("ssh", "kria", "sudo xmutil listapps")
KV260_LOADAPP_SUDO_COMMAND = (
    "ssh",
    "kria",
    "sudo xmutil loadapp carnot_ising_v2_n64",
)
KV260_LATENCY_COMMAND = ("ssh", "kria", "sudo python3 -")

VALID_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4")
LEGACY_DEPLOYABLE_OVERLAY = "carnot_ising_v2_n64"

BOARD_SAMPLE_COUNT = 32
BOARD_SPIN_COUNT = 64
BOARD_MAX_DEGREE = 16
BOARD_BETA_FINAL_Q88 = 0x0100

SUCCESS_VERDICT = (
    "complete: kv260_board_latency_transcript_captured_poc_anchor_terminal_candidate"
)
BLOCKED_VERDICT = "complete: blocked_kv260_ssh_unreachable"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_reachable",
    "kv260_overlay_loaded",
    "board_latency_samples",
    "board_latency_median_ms",
    "terminal_condition_met",
    "speedup_claim_avoided_assert",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "SSH-attached board test; per-board duration floor.",
    "preconditions_checked": (
        "Records the SSH-reachability check -- the correct KV260 precondition, "
        "not host SD card."
    ),
    "kv260_ssh_reachable": (
        "The honest board state; an unreachable board is a blocked_*, "
        "not a fabricated pass."
    ),
    "kv260_overlay_loaded": (
        "Confirms the carnot_ising overlay is the latest real-board-deployable "
        "bitstream."
    ),
    "board_latency_samples": (
        "The raw on-board per-sample latency distribution (>=30) -- the "
        "terminal-state transcript, not a single fabricated number."
    ),
    "board_latency_median_ms": (
        "Median on-board latency -- the POC functional anchor (NOT a speedup "
        "claim)."
    ),
    "terminal_condition_met": (
        "True iff a non-fabricated board-latency transcript + overlay "
        "confirmation satisfies the north-star sec-3 terminal condition."
    ),
    "speedup_claim_avoided_assert": (
        "Asserts NO thermalization/equilibrium/hardware-speedup claim is made "
        "(Paper-v6 Narrowing #2/#3)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

DEFAULT_FIXED_COMPUTE_BUDGET = {
    "spin_count": BOARD_SPIN_COUNT,
    "sample_count": BOARD_SAMPLE_COUNT,
    "max_degree": BOARD_MAX_DEGREE,
    "beta_final_q88": BOARD_BETA_FINAL_Q88,
    "trigger_mode": "reset_trigger_poll_done_once_per_sample",
}


BOARD_HARNESS_SOURCE = rf'''#!/usr/bin/env python3
import glob
import json
import mmap
import os
import struct
import time

ADDR_CONTROL = 0x0000
ADDR_STATUS = 0x0004
ADDR_SPIN_COUNT = 0x0008
ADDR_BETA_FINAL = 0x001C
ADDR_BIAS_BASE = 0x1000
ADDR_ADJ_BASE = 0x2000
ADDR_COUPL_BASE = 0x6000
ADDR_SPOUT_BASE = 0xA010
STATUS_DONE_MASK = 0x4
DEFAULT_MAP_SIZE = 0x20000
SAMPLER_BASE_ADDR = 0xA0000000
POLL_TIMEOUT_S = 0.250
SPIN_COUNT = {BOARD_SPIN_COUNT}
MAX_DEGREE = {BOARD_MAX_DEGREE}
SAMPLE_COUNT = {BOARD_SAMPLE_COUNT}
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
        dev_path = "/dev/" + name
        addr_text = _read_text(os.path.join(sys_path, "maps/map0/addr"))
        size_text = _read_text(os.path.join(sys_path, "maps/map0/size"))
        map_name = _read_text(os.path.join(sys_path, "maps/map0/name"))
        devices.append(
            {{
                "path": dev_path,
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
    for dev in devices:
        if dev["path"] == "/dev/uio0":
            return dev
    raise RuntimeError("no UIO device candidates found")


def _open_map(dev):
    size = max(int(dev.get("size") or DEFAULT_MAP_SIZE), DEFAULT_MAP_SIZE)
    fd = os.open(dev["path"], os.O_RDWR | os.O_SYNC)
    try:
        mm = mmap.mmap(
            fd,
            size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            flags=mmap.MAP_SHARED,
        )
    except Exception:
        os.close(fd)
        raise
    return fd, mm


def _read_u32(mm, offset):
    return struct.unpack_from("<I", mm, offset)[0]


def _write_u32(mm, offset, value):
    struct.pack_into("<I", mm, offset, int(value) & 0xFFFFFFFF)


def _pack_i16(value):
    return int(value) & 0xFFFF


def _upload_fixed_problem(mm):
    _write_u32(mm, ADDR_CONTROL, 0x2)
    _write_u32(mm, ADDR_CONTROL, 0x0)
    _write_u32(mm, ADDR_SPIN_COUNT, SPIN_COUNT)
    _write_u32(mm, ADDR_BETA_FINAL, BETA_FINAL_Q88)
    for spin in range(SPIN_COUNT):
        _write_u32(mm, ADDR_BIAS_BASE + 4 * spin, 0)
        for slot in range(MAX_DEGREE):
            neighbor = (spin + slot + 1) % SPIN_COUNT
            coupling = 16 if ((spin + slot + {RANDOM_SEED}) % 2 == 0) else -16
            offset = 4 * (spin * MAX_DEGREE + slot)
            _write_u32(mm, ADDR_ADJ_BASE + offset, _pack_i16(neighbor))
            _write_u32(mm, ADDR_COUPL_BASE + offset, _pack_i16(coupling))


def _read_spin_words(mm):
    words = []
    for word_index in range((SPIN_COUNT + 31) // 32):
        words.append(_read_u32(mm, ADDR_SPOUT_BASE + 4 * word_index))
    return words


def _run_one_sample(mm):
    _write_u32(mm, ADDR_CONTROL, 0x2)
    _write_u32(mm, ADDR_CONTROL, 0x0)
    start_ns = time.perf_counter_ns()
    _write_u32(mm, ADDR_CONTROL, 0x1)
    deadline = time.perf_counter() + POLL_TIMEOUT_S
    while time.perf_counter() < deadline:
        if _read_u32(mm, ADDR_STATUS) & STATUS_DONE_MASK:
            end_ns = time.perf_counter_ns()
            words = _read_spin_words(mm)
            return (end_ns - start_ns) / 1_000_000.0, words
    raise RuntimeError("sampler poll timed out")


def main():
    print("BOARD_HARNESS_START exp3709", flush=True)
    devices = _discover_uio_devices()
    sampler_uio = _select_sampler_uio(devices)
    fd, mm = _open_map(sampler_uio)
    samples = []
    final_words = []
    batch_start_ns = time.perf_counter_ns()
    try:
        _upload_fixed_problem(mm)
        for _ in range(SAMPLE_COUNT):
            elapsed_ms, final_words = _run_one_sample(mm)
            samples.append(elapsed_ms)
    finally:
        mm.close()
        os.close(fd)
    batch_ms = (time.perf_counter_ns() - batch_start_ns) / 1_000_000.0
    print(
        json.dumps(
            {{
                "schema": "carnot.kv260.remote_latency_harness.v1",
                "sample_count": SAMPLE_COUNT,
                "per_sample_wall_ms": samples,
                "per_batch_wall_ms": batch_ms,
                "fixed_compute_budget": {{
                    "spin_count": SPIN_COUNT,
                    "sample_count": SAMPLE_COUNT,
                    "max_degree": MAX_DEGREE,
                    "beta_final_q88": BETA_FINAL_Q88,
                    "trigger_mode": "reset_trigger_poll_done_once_per_sample",
                }},
                "selected_uio": sampler_uio["path"],
                "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
                "final_spin_words_hex": [hex(int(word)) for word in final_words],
            }},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class CommandProbe:
    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> dict[str, object]:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "combined_output": self.combined_output,
        }


CommandRunner = Callable[[tuple[str, ...], str | None, float], CommandProbe]


def command_to_string(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def run_command(
    command: tuple[str, ...],
    stdin: str | None = None,
    timeout_s: float = 60.0,
) -> CommandProbe:
    """Run one command while preserving enough transcript data for audit."""
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
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout_s}s"
        return CommandProbe(
            command=command,
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.perf_counter() - started,
        )
    except OSError as exc:
        return CommandProbe(
            command=command,
            exit_code=127,
            stderr=str(exc),
            duration_s=time.perf_counter() - started,
        )
    return CommandProbe(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=time.perf_counter() - started,
    )


def sha256_payload(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_board_payload(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise ValueError("board harness stdout did not contain a final JSON object")


def validate_board_payload(payload: dict[str, Any]) -> None:
    samples = payload.get("per_sample_wall_ms")
    if not isinstance(samples, list) or len(samples) < 30:
        raise ValueError("board latency transcript must contain at least 30 samples")
    if any(float(sample) <= 0.0 for sample in samples):
        raise ValueError("board latency samples must be positive")
    if float(payload.get("per_batch_wall_ms", 0.0)) <= 0.0:
        raise ValueError("board batch latency must be positive")


def _detect_overlay(text: str) -> str | None:
    for overlay in VALID_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def _precondition_entry(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "resource": "kv260_ssh",
        "command": command_to_string(KV260_SSH_COMMAND),
        "available": ssh_probe.exit_code == 0,
        "exit_code": ssh_probe.exit_code,
    }


def _empty_command_probes(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "kv260_ssh": ssh_probe.as_dict(),
        "kv260_xmutil_listapps_initial": None,
        "kv260_xmutil_listapps_initial_sudo": None,
        "kv260_xmutil_loadapp": None,
        "kv260_xmutil_loadapp_sudo": None,
        "kv260_xmutil_listapps_after_load": None,
        "kv260_xmutil_listapps_after_load_sudo": None,
        "kv260_latency_harness": None,
    }


def _operator_action_item(ssh_reachable: bool, terminal_condition_met: bool) -> str:
    if not ssh_reachable:
        return "Restore kria/kv260.local SSH reachability; do not use host SD-card checks."
    if terminal_condition_met:
        return "none"
    return "Review overlay load and board sampler transcript; a further board action is required."


def _median_or_none(samples: list[float]) -> float | None:
    if not samples:
        return None
    return float(statistics.median(samples))


def _success_from_board_payload(
    board_payload: dict[str, Any],
) -> tuple[list[float], float, dict[str, Any]]:
    validate_board_payload(board_payload)
    samples = [float(sample) for sample in board_payload["per_sample_wall_ms"]]
    batch_ms = float(board_payload["per_batch_wall_ms"])
    fixed_budget = dict(DEFAULT_FIXED_COMPUTE_BUDGET)
    fixed_budget.update(dict(board_payload.get("fixed_compute_budget") or {}))
    return samples, batch_ms, fixed_budget


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _base_payload(
    *,
    ssh_probe: CommandProbe,
    duration_s: float,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "kv260_ssh_reachable": ssh_probe.exit_code == 0,
        "kv260_overlay_loaded": None,
        "board_latency_samples": [],
        "board_latency_median_ms": None,
        "board_latency_batch_ms": None,
        "fixed_compute_budget": dict(DEFAULT_FIXED_COMPUTE_BUDGET),
        "terminal_condition_met": False,
        "speedup_claim_avoided_assert": True,
        "latency_interpretation": (
            "POC functional fixed-compute heuristic-budget latency transcript only; "
            "no comparative CPU performance claim."
        ),
        "operator_action_item": _operator_action_item(ssh_probe.exit_code == 0, False),
        "command_probes": _empty_command_probes(ssh_probe),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(float(duration_s), 0.0001), 4),
    }


def validate_artifact(payload: dict[str, object]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(payload)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if payload["inference_substrate"] != "hardware_smoke":
        raise ValueError("inference_substrate must be hardware_smoke")
    if payload["speedup_claim_avoided_assert"] is not True:
        raise ValueError("speedup_claim_avoided_assert must be true")
    if payload["kv260_ssh_reachable"] is False:
        if payload["honest_verdict"] != BLOCKED_VERDICT:
            raise ValueError("unreachable SSH must use the blocked KV260 verdict")
        if payload["terminal_condition_met"] is not False:
            raise ValueError("unreachable SSH cannot satisfy terminal condition")
        return
    if payload["honest_verdict"] != SUCCESS_VERDICT:
        raise ValueError("reachable terminal transcript must use the success verdict")
    if payload["kv260_overlay_loaded"] not in VALID_OVERLAYS:
        raise ValueError("kv260_overlay_loaded is not a valid Carnot overlay")
    samples = payload["board_latency_samples"]
    if not isinstance(samples, list) or len(samples) < 30:
        raise ValueError("terminal artifact must carry at least 30 samples")
    if any(float(sample) <= 0.0 for sample in samples):
        raise ValueError("terminal artifact samples must be positive")
    if payload["board_latency_median_ms"] is None:
        raise ValueError("terminal artifact must carry median latency")
    if payload["terminal_condition_met"] is not True:
        raise ValueError("terminal condition must be true for success")


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Build the artifact from SSH and board-side timing evidence only."""
    started = time.perf_counter()
    ssh_probe = command_runner(KV260_SSH_COMMAND, None, 10.0)
    raw_elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    payload = _base_payload(ssh_probe=ssh_probe, duration_s=raw_elapsed)

    if ssh_probe.exit_code != 0:
        checksum_payload = dict(payload)
        payload["reproducibility_checksum"] = sha256_payload(checksum_payload)
        validate_artifact(payload)
        return payload

    command_probes = payload["command_probes"]
    list_probe = command_runner(KV260_LISTAPPS_COMMAND, None, 30.0)
    command_probes["kv260_xmutil_listapps_initial"] = list_probe.as_dict()
    overlay_loaded = _detect_overlay(list_probe.combined_output) if list_probe.exit_code == 0 else None
    if overlay_loaded is None and _xmutil_requires_root(list_probe):
        list_sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, None, 30.0)
        command_probes["kv260_xmutil_listapps_initial_sudo"] = (
            list_sudo_probe.as_dict()
        )
        overlay_loaded = (
            _detect_overlay(list_sudo_probe.combined_output)
            if list_sudo_probe.exit_code == 0
            else None
        )

    if overlay_loaded is None:
        load_probe = command_runner(KV260_LOADAPP_COMMAND, None, 120.0)
        command_probes["kv260_xmutil_loadapp"] = load_probe.as_dict()
        if _xmutil_requires_root(load_probe):
            load_sudo_probe = command_runner(KV260_LOADAPP_SUDO_COMMAND, None, 120.0)
            command_probes["kv260_xmutil_loadapp_sudo"] = load_sudo_probe.as_dict()
        list_after = command_runner(KV260_LISTAPPS_COMMAND, None, 30.0)
        command_probes["kv260_xmutil_listapps_after_load"] = list_after.as_dict()
        overlay_loaded = (
            _detect_overlay(list_after.combined_output) if list_after.exit_code == 0 else None
        )
        if overlay_loaded is None and _xmutil_requires_root(list_after):
            list_after_sudo = command_runner(KV260_LISTAPPS_SUDO_COMMAND, None, 30.0)
            command_probes["kv260_xmutil_listapps_after_load_sudo"] = (
                list_after_sudo.as_dict()
            )
            overlay_loaded = (
                _detect_overlay(list_after_sudo.combined_output)
                if list_after_sudo.exit_code == 0
                else None
            )

    if overlay_loaded is None:  # pragma: no cover
        raise RuntimeError("KV260 SSH reachable, but Carnot Ising overlay was not confirmed")

    latency_probe = command_runner(KV260_LATENCY_COMMAND, BOARD_HARNESS_SOURCE, 1800.0)
    command_probes["kv260_latency_harness"] = latency_probe.as_dict()
    if latency_probe.exit_code != 0:  # pragma: no cover
        raise RuntimeError(
            "KV260 SSH reachable and overlay confirmed, but latency harness failed: "
            f"{latency_probe.stderr.strip()[:300]}"
        )

    board_payload = extract_board_payload(latency_probe.stdout)
    samples, batch_ms, fixed_budget = _success_from_board_payload(board_payload)
    terminal_condition_met = overlay_loaded in VALID_OVERLAYS and len(samples) >= 30

    raw_elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    payload.update(
        {
            "honest_verdict": SUCCESS_VERDICT,
            "kv260_overlay_loaded": overlay_loaded,
            "board_latency_samples": samples,
            "board_latency_median_ms": _median_or_none(samples),
            "board_latency_batch_ms": batch_ms,
            "fixed_compute_budget": fixed_budget,
            "terminal_condition_met": terminal_condition_met,
            "operator_action_item": _operator_action_item(True, terminal_condition_met),
            "board_harness_summary": {
                "schema": board_payload.get("schema"),
                "sample_count": board_payload.get("sample_count"),
                "selected_uio": board_payload.get("selected_uio"),
                "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            },
            "duration_s": round(max(float(raw_elapsed), 0.0001), 4),
        }
    )
    checksum_payload = dict(payload)
    payload["reproducibility_checksum"] = sha256_payload(checksum_payload)
    validate_artifact(payload)
    return payload


def write_artifact(repo_root: str | Path, payload: dict[str, object]) -> Path:
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def run_experiment(
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> Path:
    payload = build_artifact(command_runner=command_runner, duration_s=duration_s)
    return write_artifact(repo_root, payload)


def main() -> None:  # pragma: no cover
    out_path = run_experiment(Path("."))
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "honest_verdict": payload["honest_verdict"],
                "terminal_condition_met": payload["terminal_condition_met"],
                "kv260_ssh_reachable": payload["kv260_ssh_reachable"],
                "result": str(out_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
