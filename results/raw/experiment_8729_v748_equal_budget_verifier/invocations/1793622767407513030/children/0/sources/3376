"""Exp 2942 KV260 continuation n-scaling latency profile.

Spec refs: REQ-HW-074, SCENARIO-HW-074.

The purpose of this experiment is to replace a CPU/KV260 crossover
extrapolation with board-measured latency rows wherever the active bitstream
actually supports the requested spin count. If the loaded Carnot image is fixed
to a smaller n, the artifact records that boundary instead of inventing larger
n measurements. The hardware path only uses SSH, the existing overlay, and UIO
register access; it does not synthesize, flash, or modify the bitstream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import math
import random
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_2942_kv260_continuation_n_scaling_v1.json")
TRANSCRIPT_REL_PATH = Path("results/experiment_2942_kv260_n_scaling_transcript.log")
OUTPUT_DIR = REPO_ROOT / "output" / "experiment_2942_kv260_n_scaling"
LOCAL_PROBLEM_PATH = OUTPUT_DIR / "problem_payload.json"
LOCAL_HARNESS_PATH = OUTPUT_DIR / "board_harness.py"
REMOTE_PROBLEM_PATH = "/tmp/experiment_2942_kv260_problem.json"
REMOTE_HARNESS_PATH = "/tmp/experiment_2942_kv260_board_harness.py"

EXPERIMENT_ID = 2942
RUN_DATE = "20260523"
SCHEMA = "results/v1"
INFERENCE_SUBSTRATE = "hardware_smoke"
KV260_HOST = "kria"
OVERLAY_LOAD_COMMAND = "sudo xmutil unloadapp 2>/dev/null; sudo xmutil loadapp carnot_ising_v2_n64"
VALID_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4")
TARGET_N_VALUES = [64, 128, 256, 512, 1024]
N_SAMPLES_PER_N = 1000
MAX_DEGREE = 16
BETA_FINAL_Q88 = 0x0100
RANDOM_SEED_BY_N = {n: 2_942_000 + n for n in TARGET_N_VALUES}

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "bitstream_supports_variable_n",
    "per_n_results",
    "bitstream_sha256",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
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
                "addr_hex": _read_text(os.path.join(sys_path, "maps/map0/addr")),
                "size": _parse_int(_read_text(os.path.join(sys_path, "maps/map0/size")), DEFAULT_MAP_SIZE),
                "name": _read_text(os.path.join(sys_path, "maps/map0/name")),
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
    size = max(dev.get("size") or DEFAULT_MAP_SIZE, DEFAULT_MAP_SIZE)
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
    struct.pack_into("<I", mm, offset, int(value) & 0xFFFFFFFF)


def _pack_i16(value):
    return int(value) & 0xFFFF


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _p95(values):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    return ordered[index]


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


def _run_samples(mm, problem, n_samples):
    latencies_us = []
    failed = 0
    for _ in range(int(n_samples)):
        _write_u32(mm, ADDR_CONTROL, 0x2)
        _write_u32(mm, ADDR_CONTROL, 0x0)
        started_ns = time.perf_counter_ns()
        _write_u32(mm, ADDR_CONTROL, 0x1)
        deadline = time.perf_counter() + POLL_TIMEOUT_S
        done = False
        while time.perf_counter() < deadline:
            if _read_u32(mm, ADDR_STATUS) & STATUS_DONE_MASK:
                done = True
                break
        ended_ns = time.perf_counter_ns()
        if done:
            latencies_us.append((ended_ns - started_ns) / 1000.0)
        else:
            failed += 1
    if not latencies_us:
        raise RuntimeError("no completed samples observed")
    return {
        "n": int(problem["n_spins"]),
        "n_samples": int(n_samples),
        "per_sample_us_median": _median(latencies_us),
        "per_sample_us_p95": _p95(latencies_us),
        "per_sample_us_min": min(latencies_us),
        "per_sample_us_max": max(latencies_us),
        "failed_samples": failed,
    }


def main():
    problem_path = sys.argv[1]
    with open(problem_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    started = time.perf_counter()
    print("BOARD_HARNESS_START experiment_2942", flush=True)
    devices = _discover_uio_devices()
    print("UIO_DEVICES " + json.dumps(devices, sort_keys=True), flush=True)
    sampler_uio = _select_sampler_uio(devices)
    print("SELECTED_UIO " + json.dumps(sampler_uio, sort_keys=True), flush=True)
    fd, mm = _open_map(sampler_uio)
    runs = []
    try:
        for problem in payload["problems"]:
            _upload_problem(mm, problem)
            print(f"RUN n={problem['n_spins']} samples={payload['n_samples_per_n']}", flush=True)
            runs.append(_run_samples(mm, problem, int(payload["n_samples_per_n"])))
    finally:
        mm.close()
        os.close(fd)
    print(json.dumps(
        {
            "duration_s": time.perf_counter() - started,
            "selected_uio": sampler_uio["path"],
            "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
            "uio_devices": [dev["path"] for dev in devices],
            "runs": runs,
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
'''


@dataclass(frozen=True)
class BitstreamNSupport:
    """Active bitstream spin-count boundary inferred before board sampling."""

    variable: bool
    supported_n: list[int]
    detail: str


@dataclass(frozen=True)
class HardwareRunResult:
    preconditions_checked: list[dict[str, Any]]
    bitstream_sha256: str
    bitstream_support: BitstreamNSupport
    per_n_results: list[dict[str, float | int]]
    board_summary: dict[str, Any] = field(default_factory=dict)
    blocked_verdict: str = ""
    transcript_path: str = ""


@dataclass(frozen=True)
class CommandResult:  # pragma: no cover - live hardware path
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


class Transcript:  # pragma: no cover - live hardware path
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def q88(value: float) -> int:
    return max(-32768, min(32767, int(round(float(value) * 256.0))))


def _ordered_neighbors(row: int, n_spins: int, max_degree: int) -> list[int]:
    neighbors: list[int] = []
    offset = 1
    while len(neighbors) < min(max_degree, n_spins - 1):
        for candidate in ((row + offset) % n_spins, (row - offset) % n_spins):
            if candidate != row and candidate not in neighbors:
                neighbors.append(candidate)
            if len(neighbors) == min(max_degree, n_spins - 1):
                break
        offset += 1
    return neighbors


def generate_sparse_ising_problem(
    n_spins: int,
    *,
    seed: int,
    max_degree: int = MAX_DEGREE,
) -> dict[str, Any]:
    """Generate the deterministic sparse Ising upload used for timing.

    The board measurement only needs a real register upload and sampler trigger;
    it does not need a dense O(n^2) matrix. A sparse ring-neighborhood graph
    keeps the payload bounded while still exercising the same bias, adjacency,
    and coupling RAM windows that the KV260 sampler consumes.
    """

    if n_spins <= 1:
        raise ValueError("n_spins must be greater than one")
    if max_degree <= 0:
        raise ValueError("max_degree must be positive")

    rng = random.Random(int(seed))
    sigma = 1.0 / math.sqrt(int(n_spins))
    adjacency: list[list[int]] = []
    couplings_q88: list[list[int]] = []
    for row in range(int(n_spins)):
        neighbors = _ordered_neighbors(row, int(n_spins), int(max_degree))
        coupling_row = [q88(rng.gauss(0.0, sigma)) for _ in neighbors]
        while len(neighbors) < int(max_degree):
            neighbors.append(-1)
            coupling_row.append(0)
        adjacency.append(neighbors)
        couplings_q88.append(coupling_row)

    upload = {
        "layout": "ising_sampler_sparse_axi_q8_8_n_scaling",
        "max_degree": int(max_degree),
        "h_q88": [0 for _ in range(int(n_spins))],
        "adjacency": adjacency,
        "couplings_q88": couplings_q88,
    }
    return {
        "n": int(n_spins),
        "n_spins": int(n_spins),
        "random_seed": int(seed),
        "j_distribution": "sparse_ring_neighborhood_normal_0_1_over_sqrt_n",
        "beta_final_q88": BETA_FINAL_Q88,
        "upload": upload,
    }


def problem_spec(problem: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": int(problem["n_spins"]),
        "random_seed": int(problem["random_seed"]),
        "max_degree": int(problem["upload"]["max_degree"]),
        "beta_final_q88": int(problem["beta_final_q88"]),
        "sparse_upload_sha256": sha256_canonical(problem["upload"]),
    }


def build_problem_payload(
    n_values: Sequence[int],
    *,
    seed_by_n: Mapping[int, int] = RANDOM_SEED_BY_N,
) -> dict[str, Any]:
    problems = [
        generate_sparse_ising_problem(int(n), seed=int(seed_by_n[int(n)]))
        for n in n_values
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "requested_n_values": [int(n) for n in n_values],
        "n_samples_per_n": N_SAMPLES_PER_N,
        "problem_specs": [problem_spec(problem) for problem in problems],
        "problems": problems,
        "upload_note": (
            "Sparse deterministic timing problems exercise the KV260 register "
            "upload path without constructing an O(n^2) payload. Unsupported "
            "target n values are not sent to the board."
        ),
    }


def detect_bitstream_support(overlay_loaded: str, board_detail: str = "") -> BitstreamNSupport:
    text = f"{overlay_loaded}\n{board_detail}".lower()
    if "n64" in text:
        return BitstreamNSupport(
            variable=False,
            supported_n=[64],
            detail=(
                "active overlay name includes n64; treating the loaded "
                "carnot_ising_v4 image as fixed n=64 for this run"
            ),
        )
    if "carnot_ising_v4" in text:
        return BitstreamNSupport(
            variable=True,
            supported_n=[64, 128],
            detail=(
                "carnot_ising_v4 local RTL/README expose SPIN_COUNT through "
                "the synthesized maximum n=128; n>=256 is unsupported by this image"
            ),
        )
    return BitstreamNSupport(
        variable=False,
        supported_n=[64],
        detail=(
            "no trustworthy variable-n metadata was found; defaulting to the "
            "previously proven fixed n=64 KV260 boundary"
        ),
    )


def select_measured_n_values(support: BitstreamNSupport) -> list[int]:
    if not support.supported_n:
        return []
    supported_targets = [n for n in TARGET_N_VALUES if n in set(support.supported_n)]
    if support.variable:
        return supported_targets
    return [max(supported_targets or support.supported_n)]


def summarize_board_payload(
    board_payload: Mapping[str, Any],
    *,
    measured_n_values: Sequence[int],
) -> list[dict[str, float | int]]:
    runs = board_payload.get("runs")
    if not isinstance(runs, Sequence):
        raise ValueError("board payload missing runs")
    by_n = {int(row["n"]): row for row in runs if isinstance(row, Mapping) and "n" in row}
    results: list[dict[str, float | int]] = []
    for n in measured_n_values:
        row = by_n.get(int(n))
        if row is None:
            raise ValueError(f"board payload missing n={n} run")
        if int(row.get("failed_samples", 0)) != 0:
            raise ValueError(f"board run for n={n} had failed samples")
        median = float(row["per_sample_us_median"])
        p95 = float(row["per_sample_us_p95"])
        if median <= 0.0 or p95 <= 0.0:
            raise ValueError("board latency rows must be positive")
        results.append(
            {
                "n": int(n),
                "per_sample_us_median": median,
                "per_sample_us_p95": p95,
            }
        )
    return results


def _problem_specs_for_n(
    problem_payload: Mapping[str, Any],
    measured_n_values: Sequence[int],
) -> list[dict[str, Any]]:
    wanted = {int(n) for n in measured_n_values}
    return [
        dict(spec)
        for spec in problem_payload.get("problem_specs", [])
        if int(spec["n"]) in wanted
    ]


def _seeds_for_n(measured_n_values: Sequence[int]) -> list[int]:
    return [int(RANDOM_SEED_BY_N[int(n)]) for n in measured_n_values]


def blocked_artifact(
    *,
    verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    support: BitstreamNSupport | None = None,
) -> dict[str, Any]:
    support = support or BitstreamNSupport(variable=False, supported_n=[], detail="")
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2942-kv260-continuation-n-scaling-v1",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "bitstream_supports_variable_n": bool(support.variable),
        "bitstream_n_support_detail": support.detail,
        "per_n_results": [],
        "bitstream_sha256": "",
        "random_seeds_used": [],
        "reproducibility_checksum": "",
        "requested_n_values": list(TARGET_N_VALUES),
        "measured_n_values": [],
        "unsupported_n_values": list(TARGET_N_VALUES),
        "n_samples_per_n": N_SAMPLES_PER_N,
        "no_bitstream_modified": True,
        "duration_s": float(duration_s),
    }


def _success_verdict(support: BitstreamNSupport, measured_n_values: Sequence[int]) -> str:
    if not support.variable:
        return f"complete: kv260_fixed_n{max(measured_n_values)}_latency_profile_recorded"
    if set(measured_n_values) == set(TARGET_N_VALUES):
        return "complete: kv260_variable_n_latency_profile_recorded"
    return "complete: kv260_variable_n_partial_latency_profile_recorded"


def build_success_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    bitstream_sha256: str,
    support: BitstreamNSupport,
    per_n_results: list[dict[str, float | int]],
    problem_payload: Mapping[str, Any],
    board_summary: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    measured_n_values = [int(row["n"]) for row in per_n_results]
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2942-kv260-continuation-n-scaling-v1",
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "generated_at": _utc_now_iso(),
        "honest_verdict": _success_verdict(support, measured_n_values),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "bitstream_supports_variable_n": bool(support.variable),
        "bitstream_n_support_detail": support.detail,
        "per_n_results": per_n_results,
        "bitstream_sha256": bitstream_sha256,
        "random_seeds_used": _seeds_for_n(measured_n_values),
        "reproducibility_checksum": "",
        "requested_n_values": list(TARGET_N_VALUES),
        "measured_n_values": measured_n_values,
        "unsupported_n_values": [n for n in TARGET_N_VALUES if n not in set(measured_n_values)],
        "n_samples_per_n": N_SAMPLES_PER_N,
        "problem_specs": _problem_specs_for_n(problem_payload, measured_n_values),
        "board_summary": dict(board_summary),
        "no_bitstream_modified": True,
        "methodology_note": (
            "Latency rows are direct KV260 UIO measurements over 1000 samples "
            "for supported n values only. Unsupported requested n values are "
            "left as unsupported, not extrapolated."
        ),
        "duration_s": float(duration_s),
    }
    artifact["reproducibility_checksum"] = sha256_canonical(
        {
            "bitstream_sha256": bitstream_sha256,
            "bitstream_support": {
                "variable": support.variable,
                "supported_n": support.supported_n,
                "detail": support.detail,
            },
            "per_n_results": per_n_results,
            "problem_specs": artifact["problem_specs"],
            "random_seeds_used": artifact["random_seeds_used"],
        }
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        return
    bitstream_sha = str(artifact["bitstream_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", bitstream_sha) is None:
        raise ValueError("complete artifact requires 64-hex bitstream_sha256")
    per_n_results = artifact["per_n_results"]
    if not isinstance(per_n_results, list) or not per_n_results:
        raise ValueError("complete artifact requires non-empty per_n_results")
    for row in per_n_results:
        if set(row) != {"n", "per_sample_us_median", "per_sample_us_p95"}:
            raise ValueError("per_n_results rows must use the required shape")
        if int(row["n"]) <= 0:
            raise ValueError("per_n_results n must be positive")
        if float(row["per_sample_us_median"]) <= 0.0 or float(row["per_sample_us_p95"]) <= 0.0:
            raise ValueError("per_n_results latency values must be positive")
    if not artifact["random_seeds_used"]:
        raise ValueError("complete artifact requires random_seeds_used")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("complete artifact requires reproducibility_checksum")


def run_experiment(
    *,
    root_path: Path = REPO_ROOT,
    hardware_runner: Callable[[dict[str, Any]], HardwareRunResult] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter() if started_s is None else float(started_s)
    problem_payload = build_problem_payload(TARGET_N_VALUES)
    active_runner = hardware_runner or run_live_hardware
    try:
        hardware_result = active_runner(problem_payload)
    except Exception as exc:  # pragma: no cover - defensive live-hardware failure artifact
        duration_s = _duration(started, now_s)
        hardware_result = HardwareRunResult(
            preconditions_checked=[
                {
                    "resource": "kv260_measurement",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
            bitstream_sha256="",
            bitstream_support=BitstreamNSupport(variable=False, supported_n=[], detail=""),
            per_n_results=[],
            blocked_verdict="blocked_kv260_measurement_failed",
        )
    duration_s = _duration(started, now_s)
    if hardware_result.blocked_verdict:
        artifact = blocked_artifact(
            verdict=hardware_result.blocked_verdict,
            preconditions_checked=hardware_result.preconditions_checked,
            duration_s=duration_s,
            support=hardware_result.bitstream_support,
        )
    else:
        artifact = build_success_artifact(
            preconditions_checked=hardware_result.preconditions_checked,
            bitstream_sha256=hardware_result.bitstream_sha256,
            support=hardware_result.bitstream_support,
            per_n_results=hardware_result.per_n_results,
            problem_payload=problem_payload,
            board_summary={
                **hardware_result.board_summary,
                "board_transcript_path": hardware_result.transcript_path,
            },
            duration_s=duration_s,
        )
    validate_artifact(artifact)
    _write_json(root_path / OUTPUT_REL_PATH, artifact)
    return artifact


def _run(cmd: list[str], timeout: int | float) -> CommandResult:  # pragma: no cover
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


def _ssh(remote_cmd: str, timeout: int | float = 30, batch_mode: bool = False) -> CommandResult:  # pragma: no cover
    cmd = ["ssh", "-o", "ConnectTimeout=5"]
    if batch_mode:
        cmd += ["-o", "BatchMode=yes"]
    cmd += [KV260_HOST, remote_cmd]
    return _run(cmd, timeout=timeout)


def _scp(local: Path, remote: str, timeout: int | float = 60) -> CommandResult:  # pragma: no cover
    return _run(
        ["scp", "-o", "ConnectTimeout=5", str(local), f"{KV260_HOST}:{remote}"],
        timeout=timeout,
    )


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:  # pragma: no cover
    return {"resource": resource, "available": bool(available), "detail": detail}


def _detect_overlay(text: str) -> str | None:  # pragma: no cover
    for overlay in VALID_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def _parse_sha256sum(output: str) -> tuple[str | None, str | None]:  # pragma: no cover
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            return parts[0].lower(), parts[1]
    return None, None


def _extract_board_json(stdout: str) -> dict[str, Any]:  # pragma: no cover
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("board harness stdout did not contain a final JSON object")


def _path_for_artifact(path: Path) -> str:  # pragma: no cover
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _check_preconditions_and_load_overlay(
    transcript: Transcript,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:  # pragma: no cover
    details: dict[str, Any] = {}
    preconditions: list[dict[str, Any]] = []

    ssh_result = _run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", KV260_HOST, "true"],
        timeout=10,
    )
    transcript.record_result("precondition_ssh", ssh_result)
    ssh_ok = ssh_result.returncode == 0
    preconditions.append(
        _precondition("kv260_ssh", ssh_ok, f"rc={ssh_result.returncode}; stderr={ssh_result.stderr.strip()[:200]}")
    )
    if not ssh_ok:
        return "blocked_kv260_ssh_unreachable", preconditions, details

    list_result = _ssh("sudo xmutil listapps 2>&1", timeout=20)
    transcript.record_result("precondition_overlay_list", list_result)
    listed_overlay = _detect_overlay(list_result.stdout + "\n" + list_result.stderr)
    overlay_ok = list_result.returncode == 0 and listed_overlay is not None
    preconditions.append(
        _precondition("kv260_overlay", overlay_ok, (list_result.stdout + list_result.stderr).strip()[:500])
    )
    if not overlay_ok:
        return "blocked_kv260_overlay_missing", preconditions, details

    load_result = _ssh(OVERLAY_LOAD_COMMAND, timeout=60)
    transcript.record_result("overlay_load", load_result)
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
    details["loaded_overlay"] = loaded_overlay or listed_overlay or ""
    details["overlay_detail"] = list_after_load.stdout + "\n" + list_after_load.stderr

    uio_result = _ssh("ls /dev/uio0 2>/dev/null && echo ok", timeout=20)
    transcript.record_result("precondition_uio0", uio_result)
    uio0_ok = uio_result.returncode == 0 and "ok" in uio_result.stdout.split()
    preconditions.append(_precondition("kv260_uio0", uio0_ok, uio_result.stdout.strip()[:200]))
    if not uio0_ok:
        return "blocked_kv260_uio_devices_absent", preconditions, details

    return None, preconditions, details


def _collect_board_provenance(transcript: Transcript) -> dict[str, Any]:  # pragma: no cover
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
        "bitstream_sha256": bitstream_sha or "",
        "bitstream_path": bitstream_path or "",
    }


def _run_board_harness(problem_payload: dict[str, Any], transcript: Transcript) -> dict[str, Any]:  # pragma: no cover
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

    harness = _ssh(f"sudo python3 {REMOTE_HARNESS_PATH} {REMOTE_PROBLEM_PATH}", timeout=900)
    transcript.record_result("run_board_harness", harness)
    if harness.returncode != 0:
        raise RuntimeError(
            f"board harness failed rc={harness.returncode}: {harness.stderr.strip()[:500]}"
        )
    return _extract_board_json(harness.stdout)


def run_live_hardware(problem_payload: dict[str, Any]) -> HardwareRunResult:  # pragma: no cover
    transcript_path = REPO_ROOT / TRANSCRIPT_REL_PATH
    transcript = Transcript(transcript_path)
    transcript.write(f"experiment_2942 started_at={_utc_now_iso()}")

    blocked, preconditions, load_details = _check_preconditions_and_load_overlay(transcript)
    empty_support = BitstreamNSupport(variable=False, supported_n=[], detail="")
    if blocked is not None:
        return HardwareRunResult(
            preconditions_checked=preconditions,
            bitstream_sha256="",
            bitstream_support=empty_support,
            per_n_results=[],
            blocked_verdict=blocked,
            transcript_path=_path_for_artifact(transcript_path),
        )

    provenance = _collect_board_provenance(transcript)
    if not provenance.get("bitstream_sha256"):
        return HardwareRunResult(
            preconditions_checked=preconditions,
            bitstream_sha256="",
            bitstream_support=empty_support,
            per_n_results=[],
            blocked_verdict="blocked_kv260_bitstream_sha256_missing",
            transcript_path=_path_for_artifact(transcript_path),
        )

    support = detect_bitstream_support(
        str(load_details.get("loaded_overlay", "")),
        str(load_details.get("overlay_detail", "")),
    )
    preconditions.append(_precondition("bitstream_n_support", True, support.detail))
    measured_n_values = select_measured_n_values(support)
    if not measured_n_values:
        return HardwareRunResult(
            preconditions_checked=preconditions,
            bitstream_sha256=provenance["bitstream_sha256"],
            bitstream_support=support,
            per_n_results=[],
            blocked_verdict="blocked_kv260_no_supported_target_n",
            transcript_path=_path_for_artifact(transcript_path),
        )

    measured_payload = build_problem_payload(measured_n_values)
    board_payload = _run_board_harness(measured_payload, transcript)
    per_n_results = summarize_board_payload(
        board_payload,
        measured_n_values=measured_n_values,
    )
    return HardwareRunResult(
        preconditions_checked=preconditions,
        bitstream_sha256=provenance["bitstream_sha256"],
        bitstream_support=support,
        per_n_results=per_n_results,
        board_summary={
            "kv260_ssh_uptime_at_run": provenance.get("uptime", ""),
            "kv260_uio_devices_present": provenance.get("uio_devices", []),
            "bitstream_path": provenance.get("bitstream_path", ""),
            "selected_uio": board_payload.get("selected_uio", ""),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex", ""),
            "board_harness_duration_s": board_payload.get("duration_s"),
        },
        transcript_path=_path_for_artifact(transcript_path),
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment()
    if args.print_result_path:
        print(REPO_ROOT / OUTPUT_REL_PATH)
    else:
        print(json.dumps({"honest_verdict": artifact["honest_verdict"], "result": str(REPO_ROOT / OUTPUT_REL_PATH)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
