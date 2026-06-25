"""Exp 4733 KV260 SSH-gated Ising latency continuity artifact.

Spec refs: REQ-HW-4733, SCENARIO-HW-4733.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


if __package__ in {None, ""}:  # pragma: no cover - direct script invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPERIMENT_ID = 4733
SCHEMA = "carnot.kv260_continuity.v1"
SPEC_REFS = ["REQ-HW-4733", "SCENARIO-HW-4733"]
OUTPUT_REL_PATH = Path("results") / "experiment_4733_kv260_continuity.json"
RANDOM_SEED = 4733
INFERENCE_SUBSTRATE = "hardware_smoke"

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
KV260_LISTAPPS_SUDO_COMMAND = ("ssh", "kria", "sudo xmutil listapps")
KV260_LATENCY_COMMAND = ("ssh", "kria", "sudo python3 -")
KV260_BITSTREAM_SHA_COMMAND = (
    "ssh",
    "kria",
    "sudo sh -lc \"find /lib/firmware -maxdepth 4 -type f "
    "\\( -name '*carnot_ising*.bit' -o -name '*carnot_ising*.bit.bin' \\) "
    "-print0 2>/dev/null | sort -z | xargs -0 sha256sum 2>/dev/null | head -n 1\"",
)

VALID_OVERLAYS = ("carnot_ising_v4_n64", "carnot_ising_v2_n64", "carnot_ising_v4")
LOAD_APP_PREFERENCE = ("carnot_ising_v4_n64", "carnot_ising_v2_n64", "carnot_ising_v4")

BOARD_SAMPLE_COUNT = 32
BOARD_SPIN_COUNT = 64
BOARD_MAX_DEGREE = 16
BOARD_BETA_FINAL_Q88 = 0x0100

SUCCESS_VERDICT = "success: kv260_latency_transcript_captured"
BLOCKED_SSH_VERDICT = "complete:/blocked_kv260_ssh_unreachable"

REQUIRED_OPERATOR_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "kv260_ssh_reachable",
    "kv260_latency_numbers",
    "kv260_synthesis_succeeded",
    "preconditions_checked",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_OPERATOR_FIELDS,
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "duration_s",
    "command_probes",
    "overlay_loaded",
    "bitstream_sha256",
    "board_harness_summary",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: kv260_latency_transcript_captured OR "
        "complete:/blocked_kv260_ssh_unreachable (an honest non-terminal continuity record)."
    ),
    "inference_substrate": (
        "hardware_smoke -- an SSH-attached board test; the duration floor is per-board, "
        "not the 60s live-model floor."
    ),
    "kv260_ssh_reachable": (
        "the SSH-reachability result (the CORRECT precondition; NEVER host SD-card "
        "device nodes)."
    ),
    "kv260_latency_numbers": (
        "the on-board per-sample Ising latency transcript (the terminal-state deliverable) "
        "-- present only if reachable; null + blocked verdict otherwise."
    ),
    "kv260_synthesis_succeeded": (
        "true if the carnot Ising overlay loaded + ran on-board -- part of the "
        "terminal-state definition (north-star §3)."
    ),
    "preconditions_checked": (
        "records the SSH-reachability check (NEVER host SD-card); pre-empts the "
        "wrong-mechanism SD-card confusion + missing-resource fabrication."
    ),
    "verifier_is_oracle": "false -- a hardware latency measurement invokes no verifier oracle.",
    "random_seed": "determinism precondition for reproducibility (the sampler seed).",
    "reproducibility_checksum": "content-addressed hash of the bitstream + transcript inputs.",
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
        addr_text = _read_text(os.path.join(sys_path, "maps/map0/addr"))
        size_text = _read_text(os.path.join(sys_path, "maps/map0/size"))
        devices.append(
            {{
                "path": "/dev/" + name,
                "addr": _parse_int(addr_text),
                "addr_hex": addr_text,
                "size": _parse_int(size_text, DEFAULT_MAP_SIZE),
                "size_hex": size_text,
                "name": _read_text(os.path.join(sys_path, "maps/map0/name")),
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
            return (end_ns - start_ns) / 1000.0, words
    raise RuntimeError("sampler poll timed out")


def main():
    print("BOARD_HARNESS_START exp4733", flush=True)
    devices = _discover_uio_devices()
    sampler_uio = _select_sampler_uio(devices)
    fd, mm = _open_map(sampler_uio)
    samples = []
    final_words = []
    batch_start_ns = time.perf_counter_ns()
    try:
        _upload_fixed_problem(mm)
        for _ in range(SAMPLE_COUNT):
            elapsed_us, final_words = _run_one_sample(mm)
            samples.append(elapsed_us)
    finally:
        mm.close()
        os.close(fd)
    batch_us = (time.perf_counter_ns() - batch_start_ns) / 1000.0
    print(
        json.dumps(
            {{
                "schema": "carnot.kv260.remote_latency_harness.v2",
                "sample_count": SAMPLE_COUNT,
                "per_sample_wall_clock_us": samples,
                "per_batch_wall_clock_us": batch_us,
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


def loadapp_command(app_name: str) -> tuple[str, ...]:
    return (
        "ssh",
        "kria",
        f"sudo xmutil unloadapp 2>/dev/null || true; sudo xmutil loadapp {app_name}",
    )


def run_command(
    command: tuple[str, ...],
    stdin: str | None = None,
    timeout_s: float = 60.0,
) -> CommandProbe:
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
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised only by wall-clock timeout.
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


def payload_checksum(payload: dict[str, object]) -> str:
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(
        checksum_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_board_payload(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise ValueError("board harness stdout did not contain a final JSON object")


def validate_board_payload(payload: dict[str, Any]) -> None:
    samples = payload.get("per_sample_wall_clock_us")
    if not isinstance(samples, list) or len(samples) < 30:
        raise ValueError("board latency transcript must contain at least 30 samples")
    if any(float(sample) <= 0.0 for sample in samples):
        raise ValueError("board latency samples must be positive")
    if float(payload.get("per_batch_wall_clock_us", 0.0)) <= 0.0:
        raise ValueError("board batch latency must be positive")


def _precondition_entry(ssh_probe: CommandProbe) -> dict[str, object]:
    observed = ssh_probe.combined_output.strip() or f"returncode={ssh_probe.exit_code}"
    return {
        "resource": "kv260_ssh",
        "available": ssh_probe.exit_code == 0,
        "command": command_to_string(KV260_SSH_COMMAND),
        "exit_code": ssh_probe.exit_code,
        "duration_s": ssh_probe.duration_s,
        "observed": observed.splitlines()[0][:300],
        "discipline": "ssh_only_no_host_sd_card",
    }


def _empty_command_probes(ssh_probe: CommandProbe) -> dict[str, object]:
    return {
        "kv260_ssh": ssh_probe.as_dict(),
        "kv260_xmutil_listapps": None,
        "kv260_xmutil_listapps_sudo": None,
        "kv260_xmutil_loadapp": None,
        "kv260_xmutil_listapps_after_load": None,
        "kv260_bitstream_sha256": None,
        "kv260_latency_harness": None,
    }


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _listed_overlays(text: str) -> list[str]:
    return [overlay for overlay in VALID_OVERLAYS if overlay in text]


def _loaded_overlay_from_xmutil(text: str) -> str | None:
    for line in text.splitlines():
        for overlay in VALID_OVERLAYS:
            if overlay not in line:
                continue
            lowered = line.lower()
            if "running" in lowered or "slot_handle 0" in lowered or "loaded" in lowered:
                return overlay
            if "->" in line and not line.rstrip().endswith("-1"):
                return overlay
    return None


def _select_load_app(listapps_text: str) -> str:
    available = set(_listed_overlays(listapps_text))
    for app_name in LOAD_APP_PREFERENCE:
        if app_name in available:
            return app_name
    return "carnot_ising_v2_n64"


def _parse_bitstream_sha(stdout: str) -> str | None:
    for line in stdout.splitlines():
        parts = line.split()
        if parts and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            return parts[0].lower()
    return None


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return float(ordered[index])


def _latency_numbers(board_payload: dict[str, Any]) -> dict[str, Any]:
    validate_board_payload(board_payload)
    samples = [float(sample) for sample in board_payload["per_sample_wall_clock_us"]]
    fixed_budget = dict(DEFAULT_FIXED_COMPUTE_BUDGET)
    fixed_budget.update(dict(board_payload.get("fixed_compute_budget") or {}))
    return {
        "unit": "us",
        "sample_count": len(samples),
        "per_sample_wall_clock_us": samples,
        "mean_us": float(statistics.fmean(samples)),
        "median_us": float(statistics.median(samples)),
        "min_us": float(min(samples)),
        "max_us": float(max(samples)),
        "p95_us": _p95(samples),
        "per_batch_wall_clock_us": float(board_payload["per_batch_wall_clock_us"]),
        "fixed_compute_budget": fixed_budget,
        "selected_uio": board_payload.get("selected_uio"),
        "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
        "final_spin_words_hex": list(board_payload.get("final_spin_words_hex") or []),
    }


def _base_artifact(ssh_probe: CommandProbe, duration_s: float) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": BLOCKED_SSH_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_ssh_reachable": ssh_probe.exit_code == 0,
        "kv260_latency_numbers": None,
        "kv260_synthesis_succeeded": False,
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(max(float(duration_s), 0.0001), 4),
        "command_probes": _empty_command_probes(ssh_probe),
        "overlay_loaded": None,
        "bitstream_sha256": None,
        "board_harness_summary": None,
    }


def build_artifact(
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    ssh_probe = command_runner(KV260_SSH_COMMAND, None, 10.0)
    raw_elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    artifact = _base_artifact(ssh_probe, raw_elapsed)

    if ssh_probe.exit_code != 0:
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        validate_artifact(artifact)
        return artifact

    command_probes = artifact["command_probes"]
    list_probe = command_runner(KV260_LISTAPPS_COMMAND, None, 30.0)
    command_probes["kv260_xmutil_listapps"] = list_probe.as_dict()
    listapps_text = list_probe.combined_output if list_probe.exit_code == 0 else ""
    overlay_loaded = _loaded_overlay_from_xmutil(listapps_text)
    use_sudo_listapps = _xmutil_requires_root(list_probe)

    if overlay_loaded is None and use_sudo_listapps:
        sudo_list_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, None, 30.0)
        command_probes["kv260_xmutil_listapps_sudo"] = sudo_list_probe.as_dict()
        if sudo_list_probe.exit_code == 0:
            listapps_text = sudo_list_probe.combined_output
            overlay_loaded = _loaded_overlay_from_xmutil(listapps_text)

    if overlay_loaded is None:
        app_name = _select_load_app(listapps_text)
        load_command = loadapp_command(app_name)
        load_probe = command_runner(load_command, None, 120.0)
        command_probes["kv260_xmutil_loadapp"] = load_probe.as_dict()
        list_after_command = KV260_LISTAPPS_SUDO_COMMAND if use_sudo_listapps else KV260_LISTAPPS_COMMAND
        list_after_probe = command_runner(list_after_command, None, 30.0)
        command_probes["kv260_xmutil_listapps_after_load"] = list_after_probe.as_dict()
        after_text = list_after_probe.combined_output
        overlay_loaded = _loaded_overlay_from_xmutil(after_text)
        if overlay_loaded is None and load_probe.exit_code == 0 and "loaded" in load_probe.combined_output.lower():
            overlay_loaded = app_name

    bitstream_probe = command_runner(KV260_BITSTREAM_SHA_COMMAND, None, 30.0)
    command_probes["kv260_bitstream_sha256"] = bitstream_probe.as_dict()
    bitstream_sha256 = _parse_bitstream_sha(bitstream_probe.stdout)

    latency_probe = command_runner(KV260_LATENCY_COMMAND, BOARD_HARNESS_SOURCE, 1800.0)
    command_probes["kv260_latency_harness"] = latency_probe.as_dict()
    if latency_probe.exit_code != 0:
        raise RuntimeError(
            "KV260 latency harness failed after SSH and overlay confirmation: "
            f"{latency_probe.stderr.strip()[:300]}"
        )
    board_payload = extract_board_payload(latency_probe.stdout)
    numbers = _latency_numbers(board_payload)
    raw_elapsed = duration_s if duration_s is not None else time.perf_counter() - started
    artifact.update(
        {
            "honest_verdict": SUCCESS_VERDICT,
            "kv260_latency_numbers": numbers,
            "kv260_synthesis_succeeded": overlay_loaded in VALID_OVERLAYS,
            "overlay_loaded": overlay_loaded,
            "bitstream_sha256": bitstream_sha256,
            "board_harness_summary": {
                "schema": board_payload.get("schema"),
                "sample_count": board_payload.get("sample_count"),
                "selected_uio": board_payload.get("selected_uio"),
                "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            },
            "duration_s": round(max(float(raw_elapsed), 0.0001), 4),
        }
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(repo_root: str | Path, artifact: dict[str, object]) -> Path:
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(command_runner=command_runner, duration_s=duration_s)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: dict[str, object]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    for field in REQUIRED_OPERATOR_FIELDS:
        _require(field in artifact, f"missing operator field: {field}")
        _require(
            not (isinstance(artifact[field], dict) and set(artifact[field]) == {"value", "principle"}),
            f"{field} must remain a bare value",
        )
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require("mmcblk" not in encoded and "/dev/disk" not in encoded, "forbidden host storage marker")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")

    if artifact.get("kv260_ssh_reachable") is False:
        _require(artifact.get("honest_verdict") == BLOCKED_SSH_VERDICT, "bad blocked SSH verdict")
        _require(artifact.get("kv260_latency_numbers") is None, "blocked SSH cannot have latency")
        _require(artifact.get("kv260_synthesis_succeeded") is False, "blocked SSH cannot synthesize")
        return

    _require(artifact.get("kv260_ssh_reachable") is True, "kv260_ssh_reachable must be bool")
    _require(artifact.get("honest_verdict") == SUCCESS_VERDICT, "bad success verdict")
    _require(artifact.get("kv260_synthesis_succeeded") is True, "synthesis did not succeed")
    _require(artifact.get("overlay_loaded") in VALID_OVERLAYS, "invalid overlay")
    numbers = artifact.get("kv260_latency_numbers")
    _require(isinstance(numbers, dict), "success requires latency numbers")
    samples = numbers.get("per_sample_wall_clock_us")
    _require(isinstance(samples, list) and len(samples) >= 30, "success requires at least 30 samples")
    _require(all(float(sample) > 0.0 for sample in samples), "success samples must be positive")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - live hardware entrypoint.
    out_path = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    main()
