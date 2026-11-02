#!/usr/bin/env python3
"""Exp 2898: KV260 Ising sampler hardware latency transcript.

Spec: REQ-HW-060, SCENARIO-HW-060.

This script is intentionally narrow. It does not synthesize, flash, or modify
the bitstream. It checks the SSH-attached KV260, loads the existing Carnot
overlay, writes a deterministic n=64 Ising problem through the documented UIO
register map, triggers the sampler, and records board-side wall-clock latency.
If a required board resource is missing, it writes an honest blocked artifact
before any measurement attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
TRANSCRIPT_PATH = REPO_ROOT / "results" / "experiment_2898_kv260_transcript.log"
OUTPUT_DIR = REPO_ROOT / "output" / "experiment_2898_kv260_latency"
LOCAL_PROBLEM_PATH = OUTPUT_DIR / "problem_payload.json"
LOCAL_HARNESS_PATH = OUTPUT_DIR / "board_harness.py"
REMOTE_PROBLEM_PATH = "/tmp/experiment_2898_kv260_problem.json"
REMOTE_HARNESS_PATH = "/tmp/experiment_2898_kv260_board_harness.py"

EXPERIMENT_ID = 2898
RUN_DATE = "20260523"
SCHEMA = "results/v1"
INFERENCE_SUBSTRATE = "hardware_smoke"
KV260_HOST = "kria"
OVERLAY_LOAD_COMMAND = (
    "sudo xmutil unloadapp 2>/dev/null; sudo xmutil loadapp carnot_ising_v2_n64"
)
VALID_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4")
RANDOM_SEEDS = [42, 137, 271]
N_SAMPLE_COUNTS = [100, 1000, 10000]
N_SPINS = 64
MAX_DEGREE = 16
BETA_FINAL_Q88 = 0x0100

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "kv260_ssh_uptime_at_run",
    "kv260_overlay_loaded",
    "kv260_overlay_load_command",
    "kv260_uio_devices_present",
    "bitstream_sha256",
    "ising_problem_spec",
    "per_seed_results",
    "random_seeds_used",
    "reproducibility_checksum",
    "board_transcript_path",
    "duration_s",
    "run_date",
}


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


def _check_uio0_mmap():
    fd = os.open("/dev/uio0", os.O_RDWR | os.O_SYNC)
    try:
        mm = mmap.mmap(fd, 0x1000, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_SHARED)
        mm.close()
    finally:
        os.close(fd)
    return True


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


def _median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _p95(values):
    values = sorted(values)
    index = max(0, min(len(values) - 1, int(0.95 * len(values) + 0.999999) - 1))
    return values[index]


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
        "per_sample_wall_clock_us_median": _median(latencies_us),
        "per_sample_wall_clock_us_p95": _p95(latencies_us),
        "per_sample_wall_clock_us_min": min(latencies_us),
        "per_sample_wall_clock_us_max": max(latencies_us),
        "final_energy": final_energy,
        "final_spin_words_hex": [hex(int(word)) for word in final_words],
        "failed_samples": failed,
    }


def main():
    problem_path = sys.argv[1]
    with open(problem_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    started = time.perf_counter()
    print("BOARD_HARNESS_START experiment_2898")
    devices = _discover_uio_devices()
    print("UIO_DEVICES " + json.dumps(devices, sort_keys=True))
    uio0_mmap_checked = _check_uio0_mmap()
    sampler_uio = _select_sampler_uio(devices)
    print("SELECTED_UIO " + json.dumps(sampler_uio, sort_keys=True))

    fd, mm = _open_map(sampler_uio)
    runs = []
    try:
        for problem in payload["problems"]:
            _upload_problem(mm, problem)
            for n_samples in payload["n_sample_counts"]:
                print(f"RUN seed={problem['random_seed']} n_samples={n_samples}", flush=True)
                runs.append(_run_samples(mm, problem, n_samples))
    finally:
        mm.close()
        os.close(fd)

    out = {
        "duration_s": time.perf_counter() - started,
        "selected_uio": sampler_uio["path"],
        "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
        "uio0_mmap_checked": uio0_mmap_checked,
        "uio_devices": [dev["path"] for dev in devices],
        "uio_device_details": devices,
        "runs": runs,
    }
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
'''


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


class Transcript:
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


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_for_artifact(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def q88(value: float) -> int:
    return max(-32768, min(32767, int(round(float(value) * 256.0))))


def generate_ising_problem(seed: int, n_spins: int = N_SPINS) -> dict[str, Any]:
    """Generate deterministic SK-style J and h=0 for REQ-HW-060."""
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


def build_sparse_upload(problem: dict[str, Any], max_degree: int = MAX_DEGREE) -> dict[str, Any]:
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
        "n_sample_counts": list(N_SAMPLE_COUNTS),
        "ising_problem_specs": specs,
        "problems": problems,
        "upload_note": (
            "Dense J is generated as requested; the deployed v2_n64 register map "
            "accepts MAX_DEGREE sparse adjacency slots, so the board upload uses "
            "the top-|J| couplings per row and preserves the dense J for energy "
            "provenance."
        ),
    }


def _run(cmd: list[str], timeout: int | float) -> CommandResult:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
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
            stderr=exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout}s",
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


def _ssh(remote_cmd: str, timeout: int | float = 30, batch_mode: bool = False) -> CommandResult:
    cmd = ["ssh", "-o", "ConnectTimeout=5"]
    if batch_mode:
        cmd += ["-o", "BatchMode=yes"]
    cmd += [KV260_HOST, remote_cmd]
    return _run(cmd, timeout=timeout)


def _scp(local: Path, remote: str, timeout: int | float = 60) -> CommandResult:
    return _run(
        ["scp", "-o", "ConnectTimeout=5", str(local), f"{KV260_HOST}:{remote}"],
        timeout=timeout,
    )


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:
    return {"resource": resource, "available": bool(available), "detail": detail}


def _detect_overlay(text: str) -> str | None:
    for overlay in VALID_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def check_preconditions_and_load_overlay(
    transcript: Transcript,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    details: dict[str, Any] = {}
    preconditions: list[dict[str, Any]] = []

    ssh_result = _run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", KV260_HOST, "true"],
        timeout=10,
    )
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

    list_result = _ssh("sudo xmutil listapps 2>&1 | head", timeout=20)
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

    load_result = _ssh(OVERLAY_LOAD_COMMAND, timeout=60)
    transcript.record_result("overlay_load", load_result)
    details["overlay_load_stdout"] = load_result.stdout
    details["overlay_load_stderr"] = load_result.stderr

    list_after_load = _ssh("sudo xmutil listapps 2>&1", timeout=20)
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

    uio_result = _ssh("ls /dev/uio0 2>/dev/null && echo ok", timeout=20)
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


def collect_board_provenance(transcript: Transcript) -> dict[str, Any]:
    uptime = _ssh("uptime", timeout=20)
    transcript.record_result("uptime", uptime)

    uio = _ssh("ls /dev/uio* 2>/dev/null", timeout=20)
    transcript.record_result("list_uio_devices", uio)

    bit = _ssh(
        "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit 2>/dev/null | head -n 1",
        timeout=30,
    )
    transcript.record_result("bitstream_sha256_bit", bit)
    bitstream_sha, bitstream_path = _parse_sha256sum(bit.stdout)

    if bitstream_sha is None:
        bit_bin = _ssh(
            "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit.bin 2>/dev/null | head -n 1",
            timeout=30,
        )
        transcript.record_result("bitstream_sha256_bit_bin_fallback", bit_bin)
        bitstream_sha, bitstream_path = _parse_sha256sum(bit_bin.stdout)

    return {
        "uptime": uptime.stdout.strip(),
        "uio_devices": [line.strip() for line in uio.stdout.splitlines() if line.strip()],
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
    problem_payload: dict[str, Any],
    transcript: Transcript,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROBLEM_PATH.write_text(
        json.dumps(problem_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    LOCAL_HARNESS_PATH.write_text(BOARD_HARNESS_SOURCE, encoding="utf-8")

    problem_scp = _scp(LOCAL_PROBLEM_PATH, REMOTE_PROBLEM_PATH, timeout=60)
    transcript.record_result("scp_problem_payload", problem_scp)
    if problem_scp.returncode != 0:
        raise RuntimeError(f"problem scp failed rc={problem_scp.returncode}")

    harness_scp = _scp(LOCAL_HARNESS_PATH, REMOTE_HARNESS_PATH, timeout=60)
    transcript.record_result("scp_board_harness", harness_scp)
    if harness_scp.returncode != 0:
        raise RuntimeError(f"harness scp failed rc={harness_scp.returncode}")

    remote_cmd = f"sudo python3 {REMOTE_HARNESS_PATH} {REMOTE_PROBLEM_PATH}"
    harness = _ssh(remote_cmd, timeout=1800)
    transcript.record_result("run_board_harness", harness)
    if harness.returncode != 0:
        raise RuntimeError(
            f"board harness failed rc={harness.returncode}: {harness.stderr.strip()[:500]}"
        )
    return _extract_board_json(harness.stdout)


def _primary_problem_spec(problem_payload: dict[str, Any]) -> dict[str, Any]:
    first = dict(problem_payload["ising_problem_specs"][0])
    first["all_seed_specs"] = problem_payload["ising_problem_specs"]
    first["max_degree_uploaded"] = problem_payload["max_degree_uploaded"]
    first["upload_layout"] = "ising_sampler_v2_n64_sparse_axi_q8_8"
    return first


def _success_per_seed_results(board_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for row in board_payload["runs"]:
        seed = int(row["seed"])
        if int(row["n_samples"]) == max(N_SAMPLE_COUNTS):
            by_seed[seed] = row
    results = []
    for seed in RANDOM_SEEDS:
        row = by_seed[seed]
        results.append(
            {
                "seed": seed,
                "n_samples": int(row["n_samples"]),
                "per_sample_wall_clock_us_median": float(
                    row["per_sample_wall_clock_us_median"]
                ),
                "per_sample_wall_clock_us_p95": float(row["per_sample_wall_clock_us_p95"]),
                "final_energy": float(row["final_energy"]),
            }
        )
    return results


def _sample_count_sweep(board_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in board_payload["runs"]:
        rows.append(
            {
                "seed": int(row["seed"]),
                "n_samples": int(row["n_samples"]),
                "per_sample_wall_clock_us_median": float(
                    row["per_sample_wall_clock_us_median"]
                ),
                "per_sample_wall_clock_us_p95": float(row["per_sample_wall_clock_us_p95"]),
                "per_sample_wall_clock_us_min": float(row["per_sample_wall_clock_us_min"])
                if "per_sample_wall_clock_us_min" in row
                else None,
                "per_sample_wall_clock_us_max": float(row["per_sample_wall_clock_us_max"])
                if "per_sample_wall_clock_us_max" in row
                else None,
                "final_energy": float(row["final_energy"]),
                "failed_samples": int(row.get("failed_samples", 0)),
                "final_spin_words_hex": row.get("final_spin_words_hex", []),
            }
        )
    return sorted(rows, key=lambda item: (item["seed"], item["n_samples"]))


def _reproducibility_checksum(
    problem_payload: dict[str, Any], overlay_loaded: str, bitstream_sha256: str | None
) -> str:
    return sha256_canonical(
        {
            "problems": problem_payload["problems"],
            "random_seeds_used": RANDOM_SEEDS,
            "board_overlay_name": overlay_loaded,
            "bitstream_sha256": bitstream_sha256,
        }
    )


def build_blocked_artifact(
    *,
    verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    transcript_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2898-kv260-ising-sampler-hardware-latency-benchmark-v1",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "kv260_ssh_uptime_at_run": "",
        "kv260_overlay_loaded": "",
        "kv260_overlay_load_command": OVERLAY_LOAD_COMMAND,
        "kv260_uio_devices_present": [],
        "bitstream_sha256": None,
        "ising_problem_spec": {
            "n_spins": N_SPINS,
            "j_matrix_sha256": "",
            "h_vector_sha256": "",
            "random_seed": RANDOM_SEEDS[0],
        },
        "per_seed_results": [],
        "sample_count_sweep_results": [],
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
    overlay_load_command: str,
    uio_devices_present: list[str],
    bitstream_sha256: str | None,
    problem_payload: dict[str, Any],
    board_payload: dict[str, Any],
    duration_s: float,
    transcript_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2898-kv260-ising-sampler-hardware-latency-benchmark-v1",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "kv260_ssh_uptime_at_run": uptime,
        "kv260_overlay_loaded": overlay_loaded,
        "kv260_overlay_load_command": overlay_load_command,
        "kv260_uio_devices_present": uio_devices_present,
        "bitstream_sha256": bitstream_sha256,
        "bitstream_sha256_source": "board:/lib/firmware/xilinx/carnot_ising_v4",
        "ising_problem_spec": _primary_problem_spec(problem_payload),
        "problem_payload": problem_payload,
        "per_seed_results": _success_per_seed_results(board_payload),
        "sample_count_sweep_results": _sample_count_sweep(board_payload),
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": _reproducibility_checksum(
            problem_payload, overlay_loaded, bitstream_sha256
        ),
        "board_transcript_path": _path_for_artifact(transcript_path),
        "board_harness_summary": {
            "selected_uio": board_payload.get("selected_uio"),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            "uio0_mmap_checked": board_payload.get("uio0_mmap_checked"),
            "board_harness_duration_s": board_payload.get("duration_s"),
        },
        "duration_s": duration_s,
    }


def _validate_success_artifact(artifact: dict[str, Any]) -> None:
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["kv260_overlay_loaded"] not in VALID_OVERLAYS:
        raise ValueError("kv260_overlay_loaded is not a valid Carnot overlay")
    if len(artifact["per_seed_results"]) != 3:
        raise ValueError("per_seed_results must contain exactly three seed summaries")
    for row in artifact["per_seed_results"]:
        if row["per_sample_wall_clock_us_median"] <= 0:
            raise ValueError("median per-sample wall-clock must be positive")
    if float(artifact["duration_s"]) < 30.0:
        raise ValueError("duration_s below 30s hardware-smoke acceptance floor")
    if not artifact.get("bitstream_sha256"):
        raise ValueError("bitstream_sha256 missing")


def run_experiment() -> dict[str, Any]:
    started = time.perf_counter()
    transcript = Transcript(TRANSCRIPT_PATH)
    transcript.write(f"experiment_2898 started_at={_utc_now_iso()}")

    blocked, preconditions, load_details = check_preconditions_and_load_overlay(transcript)
    if blocked is not None:
        artifact = build_blocked_artifact(
            verdict=blocked,
            preconditions_checked=preconditions,
            duration_s=time.perf_counter() - started,
            transcript_path=TRANSCRIPT_PATH,
        )
        _write_json(RESULT_PATH, artifact)
        return artifact

    provenance = collect_board_provenance(transcript)
    if not provenance.get("bitstream_sha256"):
        artifact = build_blocked_artifact(
            verdict="blocked_kv260_bitstream_sha256_missing",
            preconditions_checked=preconditions,
            duration_s=time.perf_counter() - started,
            transcript_path=TRANSCRIPT_PATH,
        )
        artifact["kv260_ssh_uptime_at_run"] = provenance.get("uptime", "")
        artifact["kv260_overlay_loaded"] = load_details.get("loaded_overlay") or ""
        artifact["kv260_uio_devices_present"] = provenance.get("uio_devices", [])
        _write_json(RESULT_PATH, artifact)
        return artifact

    problem_payload = build_problem_payload()
    board_payload = run_board_harness(problem_payload, transcript)

    artifact = build_success_artifact(
        preconditions_checked=preconditions,
        uptime=provenance.get("uptime", ""),
        overlay_loaded=load_details.get("loaded_overlay") or "carnot_ising_v2_n64",
        overlay_load_command=OVERLAY_LOAD_COMMAND,
        uio_devices_present=provenance.get("uio_devices", []),
        bitstream_sha256=provenance.get("bitstream_sha256"),
        problem_payload=problem_payload,
        board_payload=board_payload,
        duration_s=time.perf_counter() - started,
        transcript_path=TRANSCRIPT_PATH,
    )
    _validate_success_artifact(artifact)
    _write_json(RESULT_PATH, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment()
    if args.print_result_path:
        print(RESULT_PATH)
    else:
        print(json.dumps({"honest_verdict": artifact["honest_verdict"], "result": str(RESULT_PATH)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
