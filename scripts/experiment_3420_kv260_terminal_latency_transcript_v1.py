#!/usr/bin/env python3
"""Exp 3420: KV260 terminal on-board latency transcript + board graduation.

Spec: REQ-HW-060 (SCENARIO-HW-060), REQ-HW-061 (SCENARIO-HW-061).

WHAT THIS DOES (in plain terms)
-------------------------------
The KV260 FPGA board is Carnot's "sovereignty story": north-star section 3
says we must show, on real hardware, that the energy function can be evaluated
on a dedicated edge accelerator. To declare that story *finished* we need ONE
honest, non-fabricated latency measurement taken on the actual board. This
script records that measurement and, if it succeeds, marks the board as having
reached its terminal state so the per-milestone hardware mandate lifts.

HOW IT STAYS HONEST
-------------------
The ONLY precondition is that the board answers over SSH. We deliberately do
NOT check the host machine's SD-card slot — that is a retired, wrong-mechanism
check that confused five prior milestones (see CLAUDE.md "KV260 SSH-Not-SD-Card
Discipline"). If the board does not answer over SSH, we write an honest
``blocked_kv260_ssh_unreachable`` artifact and stop. We never invent a
transcript: a fabricated transcript would be worse than no transcript because
it would pollute the headline sovereignty claim.

The board interaction (SSH, scp, UIO register round-trips) is performed through
an injectable command executor so the pure data-shaping and artifact-building
logic can be unit-tested without an attached board, while the real run on the
bench exercises the genuine hardware path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3420_kv260_terminal_latency_transcript_v1.json"
)
TRANSCRIPT_PATH = (
    REPO_ROOT / "results" / "experiment_3420_kv260_terminal_transcript.log"
)
OUTPUT_DIR = REPO_ROOT / "output" / "experiment_3420_kv260_terminal_latency"
LOCAL_PROBLEM_PATH = OUTPUT_DIR / "problem_payload.json"
LOCAL_HARNESS_PATH = OUTPUT_DIR / "board_harness.py"
REMOTE_PROBLEM_PATH = "/tmp/experiment_3420_kv260_problem.json"
REMOTE_HARNESS_PATH = "/tmp/experiment_3420_kv260_board_harness.py"

EXPERIMENT_ID = 3420
RUN_DATE = "20260530"
SCHEMA = "results/v1"
INFERENCE_SUBSTRATE = "hardware_smoke"
KV260_HOST = "kria"
# The legacy overlay name carnot_ising_v2_n64 maps to the current
# XDC-constrained carnot_ising_v4 bitstream (see the KV260 bitstream memory).
OVERLAY_LOAD_COMMAND = (
    "sudo xmutil unloadapp 2>/dev/null; sudo xmutil loadapp carnot_ising_v2_n64"
)
VALID_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4")
RANDOM_SEEDS = [42, 137, 271]
# A single large round-trip count is enough for a terminal transcript: we want
# a stable mean/p50/p99 on the headline problem, not a sample-size sweep.
HEADLINE_SAMPLE_COUNT = 1000
N_SPINS = 64
MAX_DEGREE = 16
BETA_FINAL_Q88 = 0x0100
# Hardware-smoke duration floor (matches adversarial_verify.py for
# inference_substrate=hardware_smoke). A real SSH round-trip campaign takes
# wall time; anything faster is suspect.
DURATION_FLOOR_S = 1.0

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_latency_transcript",
    "kv260_synthesis_succeeded",
    "kv260_terminal_state_reached",
    "kv260_ssh_uptime_at_run",
    "kv260_overlay_loaded",
    "kv260_overlay_load_command",
    "kv260_uio_devices_present",
    "bitstream_sha256",
    "ising_problem_spec",
    "per_iteration_latency_us",
    "random_seeds_used",
    "reproducibility_checksum",
    "board_transcript_path",
    "duration_s",
    "run_date",
}


# The board-side harness runs ON the KV260 (Ubuntu Xilinx). It memory-maps the
# Ising sampler's UIO device, uploads the deterministic problem, and times each
# trigger->done UIO register round-trip. It is a string here because it is
# scp'd to the board and executed there; it never runs on the dev host.
BOARD_HARNESS_SOURCE = r'''#!/usr/bin/env python3
import glob
import json
import mmap
import os
import struct
import sys
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
            {
                "path": dev_path,
                "addr": _parse_int(addr_text),
                "addr_hex": addr_text,
                "size": _parse_int(size_text, DEFAULT_MAP_SIZE),
                "size_hex": size_text,
                "name": map_name,
            }
        )
    return devices


def _select_sampler_uio(devices):
    for dev in devices:
        if dev["addr"] == SAMPLER_BASE_ADDR:
            return dev
    for dev in devices:
        if "ising" in dev["name"].lower() or "sampler" in dev["name"].lower():
            return dev
    for dev in devices:
        if dev["path"] == "/dev/uio0":
            return dev
    raise RuntimeError("no UIO device candidates found")


def _open_map(dev):
    size = dev.get("size") or DEFAULT_MAP_SIZE
    if size < DEFAULT_MAP_SIZE:
        size = DEFAULT_MAP_SIZE
    fd = os.open(dev["path"], os.O_RDWR | os.O_SYNC)
    try:
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_SHARED)
    except Exception:
        os.close(fd)
        raise
    return fd, mm


def _read_u32(mm, offset):
    return struct.unpack_from("<I", mm, offset)[0]


def _write_u32(mm, offset, value):
    struct.pack_into("<I", mm, offset, value & 0xFFFFFFFF)


def _pack_i16(value):
    return int(value) & 0xFFFF


def _upload_problem(mm, problem):
    n = int(problem["n_spins"])
    max_degree = int(problem["upload"]["max_degree"])
    _write_u32(mm, ADDR_CONTROL, 0x2)
    _write_u32(mm, ADDR_CONTROL, 0x0)
    _write_u32(mm, ADDR_SPIN_COUNT, n)
    _write_u32(mm, ADDR_BETA_FINAL, int(problem.get("beta_final_q88", 0x0100)))

    for i, q in enumerate(problem["upload"]["h_q88"]):
        _write_u32(mm, ADDR_BIAS_BASE + 4 * i, _pack_i16(q))

    for i, row in enumerate(problem["upload"]["adjacency"]):
        for k, neighbor in enumerate(row):
            offset = 4 * (i * max_degree + k)
            _write_u32(mm, ADDR_ADJ_BASE + offset, _pack_i16(neighbor))
            _write_u32(mm, ADDR_COUPL_BASE + offset, _pack_i16(problem["upload"]["couplings_q88"][i][k]))


def _read_spins(mm, n):
    words = []
    for word_index in range((n + 31) // 32):
        words.append(_read_u32(mm, ADDR_SPOUT_BASE + 4 * word_index))
    spins = []
    for i in range(n):
        word = words[i // 32]
        spins.append(1 if ((word >> (i % 32)) & 1) else -1)
    return words, spins


def _energy(j_matrix, h_vector, spins):
    n = len(spins)
    total = 0.0
    for i in range(n):
        total -= float(h_vector[i]) * spins[i]
        for j in range(i + 1, n):
            total -= float(j_matrix[i][j]) * spins[i] * spins[j]
    return total


def _run_samples(mm, problem, n_samples):
    n = int(problem["n_spins"])
    latencies_us = []
    failed = 0
    final_energy = None
    final_words = []

    for _ in range(int(n_samples)):
        _write_u32(mm, ADDR_CONTROL, 0x2)
        _write_u32(mm, ADDR_CONTROL, 0x0)

        start_ns = time.perf_counter_ns()
        _write_u32(mm, ADDR_CONTROL, 0x1)
        deadline = time.perf_counter() + POLL_TIMEOUT_S
        done = False
        while time.perf_counter() < deadline:
            if _read_u32(mm, ADDR_STATUS) & STATUS_DONE_MASK:
                done = True
                break
        end_ns = time.perf_counter_ns()
        if not done:
            failed += 1
            continue

        words, spins = _read_spins(mm, n)
        final_words = words
        final_energy = _energy(problem["j_matrix"], problem["h_vector"], spins)
        latencies_us.append((end_ns - start_ns) / 1000.0)

    if not latencies_us:
        raise RuntimeError("no completed samples observed")

    return {
        "seed": int(problem["random_seed"]),
        "n_samples": int(n_samples),
        "latencies_us": latencies_us,
        "final_energy": final_energy,
        "final_spin_words_hex": [hex(int(word)) for word in final_words],
        "failed_samples": failed,
    }


def main():
    problem_path = sys.argv[1]
    with open(problem_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    started = time.perf_counter()
    print("BOARD_HARNESS_START experiment_3420")
    devices = _discover_uio_devices()
    print("UIO_DEVICES " + json.dumps(devices, sort_keys=True))
    sampler_uio = _select_sampler_uio(devices)
    print("SELECTED_UIO " + json.dumps(sampler_uio, sort_keys=True))

    fd, mm = _open_map(sampler_uio)
    runs = []
    try:
        for problem in payload["problems"]:
            _upload_problem(mm, problem)
            n_samples = payload["headline_sample_count"]
            print(f"RUN seed={problem['random_seed']} n_samples={n_samples}", flush=True)
            runs.append(_run_samples(mm, problem, n_samples))
    finally:
        mm.close()
        os.close(fd)

    out = {
        "duration_s": time.perf_counter() - started,
        "selected_uio": sampler_uio["path"],
        "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
        "uio_devices": [dev["path"] for dev in devices],
        "runs": runs,
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
'''


@dataclass
class CommandResult:
    """Captured outcome of a single shell command (local, SSH, or scp)."""

    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


# An executor turns a command + timeout into a CommandResult. The default is the
# real subprocess runner; tests inject a fake to drive the board paths offline.
CommandExecutor = Callable[[list[str], float], CommandResult]


class Transcript:
    """Append-only log of every command issued, for audit provenance."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")

    def record_result(self, label: str, result: CommandResult) -> None:
        self.write(f"$ {label}: {shlex.join(result.cmd)}")
        self.write(f"rc={result.returncode} duration_s={result.duration_s:.6f}")
        if result.stdout:
            self.write("[stdout]")
            self.write(result.stdout.rstrip())
        if result.stderr:
            self.write("[stderr]")
            self.write(result.stderr.rstrip())
        self.write("")

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _path_for_artifact(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def q88(value: float) -> int:
    """Convert a float to Q8.8 fixed-point, clamped to signed-16-bit range."""
    return max(-32768, min(32767, int(round(float(value) * 256.0))))


def generate_ising_problem(seed: int, n_spins: int = N_SPINS) -> dict[str, Any]:
    """Generate a deterministic SK-style J matrix with zero field (h=0)."""
    rng = random.Random(seed)
    sigma = 1.0 / math.sqrt(n_spins)
    matrix = [[0.0 for _ in range(n_spins)] for _ in range(n_spins)]
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            value = rng.gauss(0.0, sigma)
            matrix[i][j] = value
            matrix[j][i] = value
    return {
        "n_spins": n_spins,
        "random_seed": seed,
        "j_distribution": "normal_0_1_over_sqrt_n",
        "j_matrix": matrix,
        "h_vector": [0.0 for _ in range(n_spins)],
    }


def build_sparse_upload(
    problem: dict[str, Any], max_degree: int = MAX_DEGREE
) -> dict[str, Any]:
    """Project dense J into the v2_n64 AXI sparse adjacency upload layout."""
    n_spins = int(problem["n_spins"])
    j_matrix = problem["j_matrix"]
    h_vector = problem["h_vector"]
    adjacency: list[list[int]] = []
    couplings_q88: list[list[int]] = []

    for i in range(n_spins):
        ranked = sorted(
            ((j, float(j_matrix[i][j])) for j in range(n_spins) if j != i),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        chosen = ranked[:max_degree]
        adjacency.append([int(j) for j, _ in chosen])
        couplings_q88.append([q88(value) for _, value in chosen])

    return {
        "layout": "ising_sampler_v2_n64_sparse_axi_q8_8",
        "max_degree": max_degree,
        "h_q88": [q88(value) for value in h_vector],
        "adjacency": adjacency,
        "couplings_q88": couplings_q88,
    }


def problem_spec(problem: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_spins": int(problem["n_spins"]),
        "j_matrix_sha256": sha256_canonical(problem["j_matrix"]),
        "h_vector_sha256": sha256_canonical(problem["h_vector"]),
        "random_seed": int(problem["random_seed"]),
    }


def build_problem_payload() -> dict[str, Any]:
    problems = []
    specs = []
    for seed in RANDOM_SEEDS:
        problem = generate_ising_problem(seed)
        problem["upload"] = build_sparse_upload(problem)
        problem["beta_final_q88"] = BETA_FINAL_Q88
        problems.append(problem)
        specs.append(problem_spec(problem))

    return {
        "experiment_id": EXPERIMENT_ID,
        "n_spins": N_SPINS,
        "max_degree_uploaded": MAX_DEGREE,
        "random_seeds_used": list(RANDOM_SEEDS),
        "headline_sample_count": HEADLINE_SAMPLE_COUNT,
        "ising_problem_specs": specs,
        "problems": problems,
        "upload_note": (
            "Dense J is generated as requested; the deployed v2_n64 register "
            "map accepts MAX_DEGREE sparse adjacency slots, so the board "
            "upload uses the top-|J| couplings per row and preserves the dense "
            "J for energy provenance."
        ),
    }


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0,100]) over a non-empty list."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty list")
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def compute_latency_stats(latencies_us: list[float]) -> dict[str, Any]:
    """Reduce raw per-iteration latencies to the terminal-transcript stats."""
    if not latencies_us:
        raise ValueError("cannot compute latency stats from zero iterations")
    return {
        "n_iterations": len(latencies_us),
        "mean_us": sum(latencies_us) / len(latencies_us),
        "p50_us": _percentile(latencies_us, 50.0),
        "p99_us": _percentile(latencies_us, 99.0),
        "min_us": min(latencies_us),
        "max_us": max(latencies_us),
    }


def _real_run(cmd: list[str], timeout: float) -> CommandResult:
    """Run a command locally via subprocess (the default executor)."""
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return CommandResult(
            cmd=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            cmd=cmd,
            returncode=124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr
            if isinstance(exc.stderr, str)
            else f"timeout after {timeout}s",
            duration_s=time.perf_counter() - started,
        )
    except OSError as exc:
        return CommandResult(
            cmd=cmd,
            returncode=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            duration_s=time.perf_counter() - started,
        )


def _ssh_cmd(remote_cmd: str, batch_mode: bool = False) -> list[str]:
    cmd = ["ssh", "-o", "ConnectTimeout=5"]
    if batch_mode:
        cmd += ["-o", "BatchMode=yes"]
    cmd += [KV260_HOST, remote_cmd]
    return cmd


def _scp_cmd(local: Path, remote: str) -> list[str]:
    return ["scp", "-o", "ConnectTimeout=5", str(local), f"{KV260_HOST}:{remote}"]


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:
    return {"resource": resource, "available": bool(available), "detail": detail}


def _detect_overlay(text: str) -> str | None:
    for overlay in VALID_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def check_preconditions_and_load_overlay(
    executor: CommandExecutor, transcript: Transcript
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    """Verify SSH reachability + overlay, loading the overlay if needed.

    Returns (blocked_verdict_or_None, preconditions_checked, details). The ONLY
    reachability check is SSH; host SD-card checks are forbidden.
    """
    details: dict[str, Any] = {}
    preconditions: list[dict[str, Any]] = []

    ssh_result = executor(_ssh_cmd("true", batch_mode=True), 10)
    transcript.record_result("precondition_ssh", ssh_result)
    ssh_ok = ssh_result.returncode == 0
    preconditions.append(
        _precondition(
            "kv260_ssh",
            ssh_ok,
            f"rc={ssh_result.returncode}; stderr={ssh_result.stderr.strip()[:200]}",
        )
    )
    if not ssh_ok:
        return "blocked_kv260_ssh_unreachable", preconditions, details

    list_result = executor(_ssh_cmd("sudo xmutil listapps 2>&1 | head"), 20)
    transcript.record_result("precondition_overlay_list", list_result)
    listed_overlay = _detect_overlay(list_result.stdout + "\n" + list_result.stderr)
    overlay_ok = list_result.returncode == 0 and listed_overlay is not None
    preconditions.append(
        _precondition(
            "kv260_overlay",
            overlay_ok,
            (list_result.stdout + list_result.stderr).strip()[:500],
        )
    )
    if not overlay_ok:
        return "blocked_kv260_overlay_missing", preconditions, details

    load_result = executor(_ssh_cmd(OVERLAY_LOAD_COMMAND), 60)
    transcript.record_result("overlay_load", load_result)
    list_after_load = executor(_ssh_cmd("sudo xmutil listapps 2>&1"), 20)
    transcript.record_result("overlay_list_after_load", list_after_load)
    loaded_overlay = _detect_overlay(
        load_result.stdout
        + "\n"
        + load_result.stderr
        + "\n"
        + list_after_load.stdout
        + "\n"
        + list_after_load.stderr
    )
    details["loaded_overlay"] = loaded_overlay or listed_overlay

    uio_result = executor(_ssh_cmd("ls /dev/uio0 2>/dev/null && echo ok"), 20)
    transcript.record_result("precondition_uio0", uio_result)
    uio0_ok = uio_result.returncode == 0 and "ok" in uio_result.stdout.split()
    preconditions.append(
        _precondition("kv260_uio0", uio0_ok, uio_result.stdout.strip()[:200])
    )
    if not uio0_ok:
        return "blocked_kv260_uio_devices_absent", preconditions, details

    return None, preconditions, details


def _parse_sha256sum(output: str) -> tuple[str | None, str | None]:
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            return parts[0].lower(), parts[1]
    return None, None


def collect_board_provenance(
    executor: CommandExecutor, transcript: Transcript
) -> dict[str, Any]:
    uptime = executor(_ssh_cmd("uptime"), 20)
    transcript.record_result("uptime", uptime)

    uio = executor(_ssh_cmd("ls /dev/uio* 2>/dev/null"), 20)
    transcript.record_result("list_uio_devices", uio)

    bit = executor(
        _ssh_cmd(
            "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit 2>/dev/null "
            "| head -n 1"
        ),
        30,
    )
    transcript.record_result("bitstream_sha256_bit", bit)
    bitstream_sha, bitstream_path = _parse_sha256sum(bit.stdout)

    if bitstream_sha is None:
        bit_bin = executor(
            _ssh_cmd(
                "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit.bin "
                "2>/dev/null | head -n 1"
            ),
            30,
        )
        transcript.record_result("bitstream_sha256_bit_bin_fallback", bit_bin)
        bitstream_sha, bitstream_path = _parse_sha256sum(bit_bin.stdout)

    return {
        "uptime": uptime.stdout.strip(),
        "uio_devices": [
            line.strip() for line in uio.stdout.splitlines() if line.strip()
        ],
        "bitstream_sha256": bitstream_sha,
        "bitstream_path": bitstream_path,
    }


def _extract_board_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("board harness stdout did not contain a final JSON object")


def run_board_harness(
    executor: CommandExecutor,
    problem_payload: dict[str, Any],
    transcript: Transcript,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROBLEM_PATH.write_text(
        json.dumps(problem_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    LOCAL_HARNESS_PATH.write_text(BOARD_HARNESS_SOURCE, encoding="utf-8")

    problem_scp = executor(_scp_cmd(LOCAL_PROBLEM_PATH, REMOTE_PROBLEM_PATH), 60)
    transcript.record_result("scp_problem_payload", problem_scp)
    if problem_scp.returncode != 0:
        raise RuntimeError(f"problem scp failed rc={problem_scp.returncode}")

    harness_scp = executor(_scp_cmd(LOCAL_HARNESS_PATH, REMOTE_HARNESS_PATH), 60)
    transcript.record_result("scp_board_harness", harness_scp)
    if harness_scp.returncode != 0:
        raise RuntimeError(f"harness scp failed rc={harness_scp.returncode}")

    harness = executor(
        _ssh_cmd(f"sudo python3 {REMOTE_HARNESS_PATH} {REMOTE_PROBLEM_PATH}"), 1800
    )
    transcript.record_result("run_board_harness", harness)
    if harness.returncode != 0:
        raise RuntimeError(
            f"board harness failed rc={harness.returncode}: "
            f"{harness.stderr.strip()[:500]}"
        )
    return _extract_board_json(harness.stdout)


def _headline_latencies(board_payload: dict[str, Any]) -> list[float]:
    """Concatenate every seed's raw round-trip latencies for the headline run."""
    latencies: list[float] = []
    for row in board_payload["runs"]:
        latencies.extend(float(x) for x in row.get("latencies_us", []))
    return latencies


def _primary_problem_spec(problem_payload: dict[str, Any]) -> dict[str, Any]:
    first = dict(problem_payload["ising_problem_specs"][0])
    first["all_seed_specs"] = problem_payload["ising_problem_specs"]
    first["max_degree_uploaded"] = problem_payload["max_degree_uploaded"]
    first["upload_layout"] = "ising_sampler_v2_n64_sparse_axi_q8_8"
    return first


def _reproducibility_checksum(
    problem_payload: dict[str, Any],
    overlay_loaded: str,
    bitstream_sha256: str | None,
) -> str:
    return sha256_canonical(
        {
            "problems": problem_payload["problems"],
            "random_seeds_used": RANDOM_SEEDS,
            "board_overlay_name": overlay_loaded,
            "bitstream_sha256": bitstream_sha256,
            "headline_sample_count": HEADLINE_SAMPLE_COUNT,
        }
    )


def build_blocked_artifact(
    *,
    verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    transcript_path: Path,
    uptime: str = "",
    overlay_loaded: str = "",
    uio_devices_present: list[str] | None = None,
) -> dict[str, Any]:
    """Honest blocked artifact: full schema, terminal-state NOT reached."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp3420-kv260-terminal-latency-transcript",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_latency_transcript": None,
        "kv260_synthesis_succeeded": False,
        "kv260_terminal_state_reached": False,
        "preconditions_checked": preconditions_checked,
        "kv260_ssh_uptime_at_run": uptime,
        "kv260_overlay_loaded": overlay_loaded,
        "kv260_overlay_load_command": OVERLAY_LOAD_COMMAND,
        "kv260_uio_devices_present": uio_devices_present or [],
        "bitstream_sha256": None,
        "ising_problem_spec": {
            "n_spins": N_SPINS,
            "j_matrix_sha256": "",
            "h_vector_sha256": "",
            "random_seed": RANDOM_SEEDS[0],
        },
        "per_iteration_latency_us": [],
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "board_transcript_path": _path_for_artifact(transcript_path),
        "duration_s": duration_s,
    }


def build_success_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    uptime: str,
    overlay_loaded: str,
    uio_devices_present: list[str],
    bitstream_sha256: str | None,
    problem_payload: dict[str, Any],
    board_payload: dict[str, Any],
    duration_s: float,
    transcript_path: Path,
    transcript_text: str,
) -> dict[str, Any]:
    """Terminal artifact: records the transcript and graduates the board."""
    latencies = _headline_latencies(board_payload)
    stats = compute_latency_stats(latencies)
    transcript_record = {
        "stats": stats,
        "headline_sample_count": HEADLINE_SAMPLE_COUNT,
        "per_seed_failed_samples": {
            str(row["seed"]): int(row.get("failed_samples", 0))
            for row in board_payload["runs"]
        },
        "command_transcript_path": _path_for_artifact(transcript_path),
        "command_transcript_excerpt": transcript_text[-4000:],
    }

    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp3420-kv260-terminal-latency-transcript",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": "complete: kv260_terminal_latency_transcript_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_latency_transcript": transcript_record,
        "kv260_synthesis_succeeded": True,
        "kv260_terminal_state_reached": True,
        "hardware_latency_us": stats["mean_us"],
        "preconditions_checked": preconditions_checked,
        "kv260_ssh_uptime_at_run": uptime,
        "kv260_overlay_loaded": overlay_loaded,
        "kv260_overlay_load_command": OVERLAY_LOAD_COMMAND,
        "kv260_uio_devices_present": uio_devices_present,
        "bitstream_sha256": bitstream_sha256,
        "bitstream_sha256_source": "board:/lib/firmware/xilinx/carnot_ising_v4",
        "ising_problem_spec": _primary_problem_spec(problem_payload),
        "per_iteration_latency_us": latencies,
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": _reproducibility_checksum(
            problem_payload, overlay_loaded, bitstream_sha256
        ),
        "board_transcript_path": _path_for_artifact(transcript_path),
        "board_harness_summary": {
            "selected_uio": board_payload.get("selected_uio"),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            "board_harness_duration_s": board_payload.get("duration_s"),
        },
        "duration_s": duration_s,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Assert schema completeness and (for success) terminal-state sanity."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    if artifact["kv260_terminal_state_reached"]:
        if not artifact["honest_verdict"].startswith("complete:"):
            raise ValueError("terminal artifact must carry a complete: verdict")
        if artifact["kv260_overlay_loaded"] not in VALID_OVERLAYS:
            raise ValueError("kv260_overlay_loaded is not a valid Carnot overlay")
        if not artifact.get("bitstream_sha256"):
            raise ValueError("bitstream_sha256 missing on terminal artifact")
        if not artifact["per_iteration_latency_us"]:
            raise ValueError("terminal artifact must carry per-iteration latencies")
        stats = artifact["kv260_latency_transcript"]["stats"]
        if stats["mean_us"] <= 0 or stats["p99_us"] <= 0:
            raise ValueError("latency stats must be positive")
        if float(artifact["duration_s"]) < DURATION_FLOOR_S:
            raise ValueError("duration_s below hardware-smoke floor")
    else:
        if not artifact["honest_verdict"].startswith("blocked_"):
            raise ValueError("non-terminal artifact must carry a blocked_ verdict")


def run_experiment(
    executor: CommandExecutor | None = None,
    *,
    result_path: Path = RESULT_PATH,
    transcript_path: Path = TRANSCRIPT_PATH,
) -> dict[str, Any]:
    """Drive the full flow: preconditions -> board harness -> artifact."""
    if executor is None:
        executor = _real_run
    started = time.perf_counter()
    transcript = Transcript(transcript_path)
    transcript.write(f"experiment_3420 started_at={_utc_now_iso()}")

    blocked, preconditions, load_details = check_preconditions_and_load_overlay(
        executor, transcript
    )
    if blocked is not None:
        artifact = build_blocked_artifact(
            verdict=blocked,
            preconditions_checked=preconditions,
            duration_s=time.perf_counter() - started,
            transcript_path=transcript_path,
            overlay_loaded=load_details.get("loaded_overlay") or "",
        )
        validate_artifact(artifact)
        _write_json(result_path, artifact)
        return artifact

    provenance = collect_board_provenance(executor, transcript)
    if not provenance.get("bitstream_sha256"):
        artifact = build_blocked_artifact(
            verdict="blocked_kv260_bitstream_sha256_missing",
            preconditions_checked=preconditions,
            duration_s=time.perf_counter() - started,
            transcript_path=transcript_path,
            uptime=provenance.get("uptime", ""),
            overlay_loaded=load_details.get("loaded_overlay") or "",
            uio_devices_present=provenance.get("uio_devices", []),
        )
        validate_artifact(artifact)
        _write_json(result_path, artifact)
        return artifact

    problem_payload = build_problem_payload()
    board_payload = run_board_harness(executor, problem_payload, transcript)

    artifact = build_success_artifact(
        preconditions_checked=preconditions,
        uptime=provenance.get("uptime", ""),
        overlay_loaded=load_details.get("loaded_overlay") or "carnot_ising_v2_n64",
        uio_devices_present=provenance.get("uio_devices", []),
        bitstream_sha256=provenance.get("bitstream_sha256"),
        problem_payload=problem_payload,
        board_payload=board_payload,
        duration_s=time.perf_counter() - started,
        transcript_path=transcript_path,
        transcript_text=transcript.read(),
    )
    validate_artifact(artifact)
    _write_json(result_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment()
    if args.print_result_path:
        print(RESULT_PATH)
    else:
        print(
            json.dumps(
                {
                    "honest_verdict": artifact["honest_verdict"],
                    "kv260_terminal_state_reached": artifact[
                        "kv260_terminal_state_reached"
                    ],
                    "result": str(RESULT_PATH),
                }
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
