"""Exp 3350 KV260 Latency Benchmark.

Spec refs: REQ-HW-101, SCENARIO-HW-101.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3350_kv260_latency_benchmark.json"
)
OUTPUT_DIR = REPO_ROOT / "output" / "experiment_3350_kv260_latency"
LOCAL_PROBLEM_PATH = OUTPUT_DIR / "problem_payload.json"
LOCAL_HARNESS_PATH = OUTPUT_DIR / "board_harness.py"
REMOTE_PROBLEM_PATH = "/tmp/experiment_3350_kv260_problem.json"
REMOTE_HARNESS_PATH = "/tmp/experiment_3350_kv260_board_harness.py"

INFERENCE_SUBSTRATE = "hardware_smoke"
KV260_HOST = "kria"
OVERLAY_LOAD_COMMAND = (
    "sudo xmutil unloadapp 2>/dev/null; sudo xmutil loadapp carnot_ising_v2_n64"
)
N_SPINS = 64
MAX_DEGREE = 16
BETA_FINAL_Q88 = 0x0100
N_PROBLEMS = 100

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
        devices.append(
            {
                "path": "/dev/" + name,
                "addr": _parse_int(_read_text(os.path.join(sys_path, "maps/map0/addr"))),
                "name": _read_text(os.path.join(sys_path, "maps/map0/name")),
            }
        )
    return devices

def _select_sampler_uio(devices):
    for dev in devices:
        if dev["addr"] == SAMPLER_BASE_ADDR:
            return dev
    for dev in devices:
        if dev["path"] == "/dev/uio0":
            return dev
    raise RuntimeError("no UIO device candidates found")

def _open_map(dev):
    fd = os.open(dev["path"], os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, DEFAULT_MAP_SIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_SHARED)
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

def main():
    problem_path = sys.argv[1]
    with open(problem_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    devices = _discover_uio_devices()
    sampler_uio = _select_sampler_uio(devices)
    fd, mm = _open_map(sampler_uio)

    latencies_us = []
    try:
        for problem in payload["problems"]:
            n = int(problem["n_spins"])
            start_ns = time.perf_counter_ns()
            _upload_problem(mm, problem)
            _write_u32(mm, ADDR_CONTROL, 0x2)
            _write_u32(mm, ADDR_CONTROL, 0x0)
            _write_u32(mm, ADDR_CONTROL, 0x1)
            deadline = time.perf_counter() + POLL_TIMEOUT_S
            while time.perf_counter() < deadline:
                if _read_u32(mm, ADDR_STATUS) & STATUS_DONE_MASK:
                    break
            _read_spins(mm, n)
            end_ns = time.perf_counter_ns()
            latencies_us.append((end_ns - start_ns) / 1000.0)
    finally:
        mm.close()
        os.close(fd)

    out = {
        "latencies_us": latencies_us,
        "median_latency_us": sorted(latencies_us)[len(latencies_us)//2] if latencies_us else 0.0,
    }
    print(json.dumps(out, sort_keys=True))

if __name__ == "__main__":
    main()
'''

def q88(value: float) -> int:
    return max(-32768, min(32767, int(round(float(value) * 256.0))))

def generate_ising_problem(seed: int, n_spins: int = N_SPINS) -> dict[str, Any]:
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

def build_problem_payload() -> dict[str, Any]:
    problems = []
    for seed in range(1, N_PROBLEMS + 1):
        problem = generate_ising_problem(seed)
        problem["upload"] = build_sparse_upload(problem)
        problem["beta_final_q88"] = BETA_FINAL_Q88
        problems.append(problem)

    return {
        "problems": problems,
    }

def _run(cmd: list[str], timeout: int | float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def _ssh(remote_cmd: str, timeout: int | float = 30) -> subprocess.CompletedProcess[str]:
    cmd = ["ssh", "-o", "ConnectTimeout=5", KV260_HOST, remote_cmd]
    return _run(cmd, timeout=timeout)

def _scp(local: Path, remote: str, timeout: int | float = 60) -> subprocess.CompletedProcess[str]:
    return _run(["scp", "-o", "ConnectTimeout=5", str(local), f"{KV260_HOST}:{remote}"], timeout=timeout)

def run_cpu_baseline(problems: list[dict[str, Any]]) -> float:
    latencies = []
    for problem in problems:
        n = problem["n_spins"]
        J = np.array(problem["j_matrix"])
        h = np.array(problem["h_vector"])
        state = np.random.choice([-1.0, 1.0], size=n)
        order = np.arange(n)

        t0 = time.perf_counter_ns()
        np.random.shuffle(order)
        for idx in order:
            field = h[idx] + J[idx] @ state
            state[idx] = 1.0 if field >= 0 else -1.0
        t1 = time.perf_counter_ns()

        latencies.append((t1 - t0) / 1000.0)
    return float(np.median(latencies))

def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def run_experiment() -> dict[str, Any]:
    started = time.perf_counter()
    ssh_test = _ssh("true", timeout=10)
    if ssh_test.returncode != 0:
        artifact = {
            "honest_verdict": "blocked_kv260_ssh_unreachable",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "kv260_ssh_uptime_at_run": "",
            "kv260_overlay_loaded": "",
            "hardware_latency_us": 0.0,
            "cpu_latency_us": 0.0,
            "speedup_vs_cpu": 0.0,
            "duration_s": time.perf_counter() - started,
            "blocked_reasons": ["SSH test failed"],
        }
        write_json(RESULT_PATH, artifact)
        return artifact

    uptime = _ssh("uptime").stdout.strip()
    _ssh(OVERLAY_LOAD_COMMAND)

    problem_payload = build_problem_payload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROBLEM_PATH.write_text(json.dumps(problem_payload), encoding="utf-8")
    LOCAL_HARNESS_PATH.write_text(BOARD_HARNESS_SOURCE, encoding="utf-8")

    _scp(LOCAL_PROBLEM_PATH, REMOTE_PROBLEM_PATH)
    _scp(LOCAL_HARNESS_PATH, REMOTE_HARNESS_PATH)

    remote_cmd = f"sudo python3 {REMOTE_HARNESS_PATH} {REMOTE_PROBLEM_PATH}"
    harness = _ssh(remote_cmd, timeout=300)

    if harness.returncode != 0:
        artifact = {
            "honest_verdict": "blocked_kv260_harness_failed",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "kv260_ssh_uptime_at_run": uptime,
            "kv260_overlay_loaded": "",
            "hardware_latency_us": 0.0,
            "cpu_latency_us": 0.0,
            "speedup_vs_cpu": 0.0,
            "duration_s": time.perf_counter() - started,
            "blocked_reasons": [harness.stderr.strip()],
        }
        write_json(RESULT_PATH, artifact)
        return artifact

    try:
        board_output = json.loads(harness.stdout.strip().splitlines()[-1])
        hw_latency_us = float(board_output["median_latency_us"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
        artifact = {
            "honest_verdict": "blocked_kv260_harness_output_invalid",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "kv260_ssh_uptime_at_run": uptime,
            "kv260_overlay_loaded": "",
            "hardware_latency_us": 0.0,
            "cpu_latency_us": 0.0,
            "speedup_vs_cpu": 0.0,
            "duration_s": time.perf_counter() - started,
            "blocked_reasons": [f"Failed to parse board harness output: {exc}"],
        }
        write_json(RESULT_PATH, artifact)
        return artifact

    cpu_latency_us = run_cpu_baseline(problem_payload["problems"])
    speedup = cpu_latency_us / hw_latency_us if hw_latency_us > 0 else 0.0

    assert hw_latency_us > 0, "Hardware latency must be positive"
    assert speedup > 0, "Speedup must be positive"

    artifact = {
        "honest_verdict": "success: hardware latency benchmark complete",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "kv260_ssh_uptime_at_run": uptime,
        "kv260_overlay_loaded": "carnot_ising_v2_n64",
        "hardware_latency_us": hw_latency_us,
        "cpu_latency_us": cpu_latency_us,
        "speedup_vs_cpu": speedup,
        "duration_s": time.perf_counter() - started,
    }
    write_json(RESULT_PATH, artifact)
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

if __name__ == "__main__":
    raise SystemExit(main())