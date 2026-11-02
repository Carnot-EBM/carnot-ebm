"""Exp 2938 KV260 MMD comparison against exact CPU sequential Gibbs.

Spec refs: REQ-HW-071, SCENARIO-HW-071.

This module is deliberately split into a testable statistics/core-sampler layer
and a thin KV260 SSH/UIO layer.  The statistical question is whether the energy
trace produced by the board's synchronous fixed-sweep Glauber update is
indistinguishable from a detailed-balance CPU sequential Gibbs chain on the same
dense n=64 Ising problems from Exp 2898.  If those distributions differ, the
paper cannot call the board output exact Boltzmann sampling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP2898_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
OUTPUT_REL_PATH = Path("results/experiment_2938_kv260_mmd_vs_cpu_sequential_gibbs_v1.json")
TRANSCRIPT_REL_PATH = Path("results/experiment_2938_kv260_mmd_transcript.log")
OUTPUT_DIR = REPO_ROOT / "output" / "experiment_2938_kv260_mmd"
LOCAL_PROBLEM_PATH = OUTPUT_DIR / "problem_payload.json"
LOCAL_HARNESS_PATH = OUTPUT_DIR / "board_harness.py"
REMOTE_PROBLEM_PATH = "/tmp/experiment_2938_kv260_problem.json"
REMOTE_HARNESS_PATH = "/tmp/experiment_2938_kv260_board_harness.py"

EXPERIMENT_ID = 2938
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "hardware_smoke"
KV260_HOST = "kria"
OVERLAY_LOAD_COMMAND = "sudo xmutil unloadapp 2>/dev/null; sudo xmutil loadapp carnot_ising_v2_n64"
VALID_OVERLAYS = ("carnot_ising_v2_n64", "carnot_ising_v4")
RANDOM_SEEDS = [42, 137, 271]
N_SPINS = 64
N_ENERGY_SAMPLES = 10_000
CPU_BURN_IN_SWEEPS = 5_000
N_PERMUTATIONS = 1_000
MAX_PERMUTATION_SAMPLES = 2_048
RFF_FEATURES = 128
SIGNIFICANCE_ALPHA = 0.01
CPU_UPDATE_SCHEDULE = "cpu_sequential_random_spin_order_single_spin_gibbs_dense_j_beta_1"
KV260_UPDATE_SCHEDULE = "kv260_synchronous_parallel_glauber_exp2898_fixed_sweep"

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "cpu_sequential_gibbs_energies_sha256",
    "kv260_synchronous_glauber_energies_sha256",
    "bitstream_sha256_cited",
    "per_seed_mmd_squared",
    "per_seed_mmd_pvalue",
    "per_seed_ks_statistic",
    "per_seed_ks_pvalue",
    "distributions_distinguishable",
    "paper_v6_recommendation",
    "random_seeds_used",
    "reproducibility_checksum",
    "methodology_note",
    "duration_s",
}


BOARD_HARNESS_SOURCE = r"""#!/usr/bin/env python3
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
    return spins


def _energy(j_matrix, h_vector, spins):
    n = len(spins)
    total = 0.0
    for i in range(n):
        total -= float(h_vector[i]) * spins[i]
        for j in range(i + 1, n):
            total -= float(j_matrix[i][j]) * spins[i] * spins[j]
    return total


def _run_energy_trace(mm, problem, n_samples):
    n = int(problem["n_spins"])
    energies = []
    failed = 0
    for _ in range(int(n_samples)):
        _write_u32(mm, ADDR_CONTROL, 0x2)
        _write_u32(mm, ADDR_CONTROL, 0x0)
        _write_u32(mm, ADDR_CONTROL, 0x1)
        deadline = time.perf_counter() + POLL_TIMEOUT_S
        done = False
        while time.perf_counter() < deadline:
            if _read_u32(mm, ADDR_STATUS) & STATUS_DONE_MASK:
                done = True
                break
        if not done:
            failed += 1
            continue
        spins = _read_spins(mm, n)
        energies.append(_energy(problem["j_matrix"], problem["h_vector"], spins))
    if len(energies) != int(n_samples):
        raise RuntimeError(f"completed {len(energies)} of {n_samples} samples; failed={failed}")
    return energies


def main():
    problem_path = sys.argv[1]
    with open(problem_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    started = time.perf_counter()
    print("BOARD_HARNESS_START experiment_2938", flush=True)
    devices = _discover_uio_devices()
    print("UIO_DEVICES " + json.dumps(devices, sort_keys=True), flush=True)
    sampler_uio = _select_sampler_uio(devices)
    print("SELECTED_UIO " + json.dumps(sampler_uio, sort_keys=True), flush=True)
    fd, mm = _open_map(sampler_uio)
    try:
        energies_by_seed = {}
        for problem in payload["problems"]:
            _upload_problem(mm, problem)
            print(f"RUN seed={problem['random_seed']} n_samples={payload['n_samples']}", flush=True)
            energies_by_seed[str(problem["random_seed"])] = _run_energy_trace(
                mm, problem, int(payload["n_samples"])
            )
    finally:
        mm.close()
        os.close(fd)
    print(json.dumps(
        {
            "duration_s": time.perf_counter() - started,
            "selected_uio": sampler_uio["path"],
            "selected_uio_addr_hex": sampler_uio.get("addr_hex", ""),
            "uio_devices": [dev["path"] for dev in devices],
            "energies_by_seed": energies_by_seed,
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
"""


@dataclass(frozen=True)
class DenseIsingProblem:
    """Dense n=64 Ising problem reproduced from the Exp 2898 seed.

    The board still receives the Exp 2898 sparse AXI upload, but both the CPU
    chain and the energy scoring use the dense Hamiltonian because the paper
    claim under audit is about the original n=64 Ising problem, not merely the
    truncated upload tensor.
    """

    seed: int
    n_spins: int
    j_matrix: np.ndarray
    h_vector: np.ndarray
    upload: dict[str, Any]
    beta_final_q88: int
    j_matrix_sha256: str
    h_vector_sha256: str


@dataclass(frozen=True)
class EnergyRunResult:
    seed: int
    energies: list[float]
    energy_sha256: str
    update_schedule: str
    spin_orders_sha256: str = ""


@dataclass(frozen=True)
class HardwareRunResult:
    preconditions_checked: list[dict[str, Any]]
    bitstream_sha256: str
    energies_by_seed: dict[int, list[float]]
    transcript_path: str = ""
    board_summary: dict[str, Any] = field(default_factory=dict)
    blocked_verdict: str = ""


@dataclass(frozen=True)
class CommandResult:  # pragma: no cover - exercised by live hardware run
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


class ProblemReproductionError(ValueError):
    """Raised when Exp 2898 does not reproduce exactly from its recorded seeds."""


class Transcript:  # pragma: no cover - exercised by live hardware run
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_canonical(value: Any) -> str:
    """Hash JSON-compatible values in the stable form used by Exp 2898."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def generate_ising_problem(seed: int, n_spins: int = N_SPINS) -> dict[str, Any]:
    """Regenerate the dense SK-style Exp 2898 problem for a seed."""

    rng = random.Random(int(seed))
    sigma = 1.0 / math.sqrt(int(n_spins))
    matrix = [[0.0 for _ in range(n_spins)] for _ in range(n_spins)]
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            value = rng.gauss(0.0, sigma)
            matrix[i][j] = value
            matrix[j][i] = value
    return {
        "n_spins": int(n_spins),
        "random_seed": int(seed),
        "j_distribution": "normal_0_1_over_sqrt_n",
        "j_matrix": matrix,
        "h_vector": [0.0 for _ in range(n_spins)],
    }


def problem_spec(problem: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_spins": int(problem["n_spins"]),
        "j_matrix_sha256": sha256_canonical(problem["j_matrix"]),
        "h_vector_sha256": sha256_canonical(problem["h_vector"]),
        "random_seed": int(problem["random_seed"]),
    }


def recover_exp2898_problems(exp2898: Mapping[str, Any]) -> list[DenseIsingProblem]:
    """Recover and checksum-verify the three dense Exp 2898 Ising problems."""

    payload = exp2898.get("problem_payload")
    if not isinstance(payload, Mapping):
        raise ProblemReproductionError("problem_payload missing")
    if payload.get("n_spins") != N_SPINS:
        raise ProblemReproductionError("problem_payload.n_spins must be 64")

    seeds = payload.get("random_seeds_used", exp2898.get("random_seeds_used"))
    if list(seeds or []) != RANDOM_SEEDS:
        raise ProblemReproductionError("random_seeds_used must be [42, 137, 271]")

    specs_by_seed = {
        int(spec["random_seed"]): spec
        for spec in payload.get("ising_problem_specs", [])
        if isinstance(spec, Mapping) and isinstance(spec.get("random_seed"), int)
    }
    rows_by_seed = {
        int(row["random_seed"]): row
        for row in payload.get("problems", [])
        if isinstance(row, Mapping) and isinstance(row.get("random_seed"), int)
    }

    problems: list[DenseIsingProblem] = []
    for seed in RANDOM_SEEDS:
        if seed not in specs_by_seed or seed not in rows_by_seed:
            raise ProblemReproductionError(f"seed {seed} missing from Exp 2898 payload")
        row = rows_by_seed[seed]
        spec = specs_by_seed[seed]
        regenerated = generate_ising_problem(seed)
        regenerated_spec = problem_spec(regenerated)
        for field_name in ("j_matrix_sha256", "h_vector_sha256"):
            if regenerated_spec[field_name] != spec.get(field_name):
                raise ProblemReproductionError(f"seed {seed} {field_name} does not reproduce")
        if sha256_canonical(row.get("j_matrix")) != spec.get("j_matrix_sha256"):
            raise ProblemReproductionError(f"seed {seed} artifact j_matrix_sha256 mismatch")
        if sha256_canonical(row.get("h_vector")) != spec.get("h_vector_sha256"):
            raise ProblemReproductionError(f"seed {seed} artifact h_vector_sha256 mismatch")
        upload = row.get("upload")
        if not isinstance(upload, dict):
            raise ProblemReproductionError(f"seed {seed} upload tensor missing")
        problems.append(
            DenseIsingProblem(
                seed=seed,
                n_spins=N_SPINS,
                j_matrix=np.asarray(row["j_matrix"], dtype=np.float64),
                h_vector=np.asarray(row["h_vector"], dtype=np.float64),
                upload=upload,
                beta_final_q88=int(row.get("beta_final_q88", 256)),
                j_matrix_sha256=str(spec["j_matrix_sha256"]),
                h_vector_sha256=str(spec["h_vector_sha256"]),
            )
        )
    return problems


def dense_energy(problem: DenseIsingProblem, state: np.ndarray) -> float:
    """Return E = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i for a spin state."""

    spins = np.asarray(state, dtype=np.float64)
    pair_energy = -0.5 * float(spins @ problem.j_matrix @ spins)
    field_energy = -float(problem.h_vector @ spins)
    return pair_energy + field_energy


def run_cpu_sequential_gibbs(
    problem: DenseIsingProblem,
    *,
    n_samples: int = N_ENERGY_SAMPLES,
    burn_in_sweeps: int = CPU_BURN_IN_SWEEPS,
) -> EnergyRunResult:
    """Run detailed-balance CPU sequential Gibbs with random spin order.

    A recorded sample is one full sweep after burn-in.  The spin order is
    reshuffled every sweep so no site gets a fixed positional advantage; the
    order checksum is recorded to make the stochastic schedule auditable without
    storing a large matrix of indices in the final artifact.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if burn_in_sweeps < 0:
        raise ValueError("burn_in_sweeps must be non-negative")

    rng = np.random.default_rng(problem.seed + 2938)
    state = (rng.integers(0, 2, size=problem.n_spins, dtype=np.int8) * 2 - 1).astype(np.int8)
    orders_hash = hashlib.sha256()
    energies: list[float] = []
    beta = problem.beta_final_q88 / 256.0
    order = np.arange(problem.n_spins)

    for sweep_index in range(int(burn_in_sweeps) + int(n_samples)):
        rng.shuffle(order)
        orders_hash.update(np.asarray(order, dtype=np.uint8).tobytes())
        for spin_index in order:
            local_field = float(problem.h_vector[spin_index] + problem.j_matrix[spin_index] @ state)
            p_plus = _sigmoid(2.0 * beta * local_field)
            state[spin_index] = 1 if rng.random() < p_plus else -1
        if sweep_index >= burn_in_sweeps:
            energies.append(round(dense_energy(problem, state), 12))

    return EnergyRunResult(
        seed=problem.seed,
        energies=energies,
        energy_sha256=sha256_canonical(energies),
        update_schedule=CPU_UPDATE_SCHEDULE,
        spin_orders_sha256=orders_hash.hexdigest(),
    )


def median_pairwise_distance(values: np.ndarray) -> float:
    """Approximate the median non-diagonal pairwise distance for 1-D energies."""

    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = int(ordered.size)
    if n < 2:
        return 1.0
    max_distance = float(ordered[-1] - ordered[0])
    if max_distance <= 0.0:
        return 1.0

    total_pairs = n * (n - 1) // 2

    def count_leq(distance: float) -> int:
        right = 0
        count = 0
        for left in range(n):
            while right < n and ordered[right] - ordered[left] <= distance:
                right += 1
            count += max(0, right - left - 1)
        return count

    def kth_distance(k: int) -> float:
        low = 0.0
        high = max_distance
        for _ in range(64):
            mid = 0.5 * (low + high)
            if count_leq(mid) >= k:
                high = mid
            else:
                low = mid
        return high

    lower_k = (total_pairs + 1) // 2
    if total_pairs % 2:
        median = kth_distance(lower_k)
    else:
        median = 0.5 * (kth_distance(lower_k) + kth_distance(lower_k + 1))
    return float(median if median > 0.0 else max_distance)


def _rbf_sum(x: np.ndarray, y: np.ndarray, bandwidth: float, *, chunk_size: int = 512) -> float:
    gamma = 1.0 / (2.0 * float(bandwidth) * float(bandwidth))
    total = 0.0
    y_values = np.asarray(y, dtype=np.float64).reshape(1, -1)
    for start in range(0, int(x.size), int(chunk_size)):
        block = np.asarray(x[start : start + chunk_size], dtype=np.float64).reshape(-1, 1)
        total += float(np.exp(-gamma * (block - y_values) ** 2).sum(dtype=np.float64))
    return total


def mmd_squared_rbf(x: Sequence[float], y: Sequence[float], bandwidth: float) -> float:
    """Compute the biased RBF-kernel MMD² over all supplied energies."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("MMD requires non-empty samples")
    sigma = float(bandwidth) if bandwidth > 0 else 1.0
    k_xx = _rbf_sum(x_arr, x_arr, sigma) / float(x_arr.size * x_arr.size)
    k_yy = _rbf_sum(y_arr, y_arr, sigma) / float(y_arr.size * y_arr.size)
    k_xy = _rbf_sum(x_arr, y_arr, sigma) / float(x_arr.size * y_arr.size)
    return max(0.0, float(k_xx + k_yy - 2.0 * k_xy))


def _balanced_subset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_permutation_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    per_group = max(1, int(max_permutation_samples) // 2)
    if x.size <= per_group and y.size <= per_group:
        return x, y
    x_idx = rng.choice(x.size, size=min(per_group, x.size), replace=False)
    y_idx = rng.choice(y.size, size=min(per_group, y.size), replace=False)
    return x[np.sort(x_idx)], y[np.sort(y_idx)]


def _rff_mmd_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
    seed: int,
    n_permutations: int,
) -> float:
    rng = np.random.default_rng(seed + 17_071)
    combined = np.concatenate([x, y]).astype(np.float64)
    labels_n = int(x.size)
    sigma = float(bandwidth) if bandwidth > 0 else 1.0
    weights = rng.normal(0.0, 1.0 / sigma, size=(RFF_FEATURES,))
    offsets = rng.uniform(0.0, 2.0 * math.pi, size=(RFF_FEATURES,))
    features = math.sqrt(2.0 / RFF_FEATURES) * np.cos(
        combined[:, None] * weights[None, :] + offsets
    )

    def stat_for_indices(indices: np.ndarray) -> float:
        selected = features[indices].mean(axis=0)
        mask = np.ones(combined.size, dtype=bool)
        mask[indices] = False
        other = features[mask].mean(axis=0)
        diff = selected - other
        return float(diff @ diff)

    observed = stat_for_indices(np.arange(labels_n))
    exceed = 0
    indices = np.arange(combined.size)
    for _ in range(int(n_permutations)):
        rng.shuffle(indices)
        if stat_for_indices(indices[:labels_n]) >= observed:
            exceed += 1
    return float((exceed + 1) / (int(n_permutations) + 1))


def mmd_permutation_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    *,
    bandwidth: float,
    seed: int,
    n_permutations: int = N_PERMUTATIONS,
    max_permutation_samples: int = MAX_PERMUTATION_SAMPLES,
) -> float:
    """Return a balanced permutation p-value for RBF MMD².

    Small unit-test sized samples use the exact quadratic MMD statistic for each
    permutation.  Full 10,000-vs-10,000 traces use a deterministic balanced
    subset and random Fourier features for the permutation loop; the artifact
    methodology note discloses this acceleration while the reported MMD² itself
    remains the exact quadratic statistic over all recorded energies.
    """

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    rng = np.random.default_rng(seed + 2938)
    x_arr, y_arr = _balanced_subset(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        max_permutation_samples=max_permutation_samples,
        rng=rng,
    )
    if x_arr.size + y_arr.size <= 600:
        observed = mmd_squared_rbf(x_arr, y_arr, bandwidth)
        combined = np.concatenate([x_arr, y_arr])
        n_x = int(x_arr.size)
        exceed = 0
        for _ in range(int(n_permutations)):
            permuted = rng.permutation(combined)
            stat_value = mmd_squared_rbf(permuted[:n_x], permuted[n_x:], bandwidth)
            if stat_value >= observed:
                exceed += 1
        return float((exceed + 1) / (int(n_permutations) + 1))
    return _rff_mmd_pvalue(
        x_arr,
        y_arr,
        bandwidth=bandwidth,
        seed=seed,
        n_permutations=n_permutations,
    )


def compare_energy_distributions(
    cpu_energies: Sequence[float],
    kv260_energies: Sequence[float],
    *,
    seed: int,
    n_permutations: int = N_PERMUTATIONS,
    max_permutation_samples: int = MAX_PERMUTATION_SAMPLES,
) -> dict[str, float]:
    """Compute MMD² plus MMD/KS p-values for one seed's energy traces."""

    cpu = np.asarray(cpu_energies, dtype=np.float64)
    kv260 = np.asarray(kv260_energies, dtype=np.float64)
    bandwidth = median_pairwise_distance(np.concatenate([cpu, kv260]))
    mmd_value = mmd_squared_rbf(cpu, kv260, bandwidth)
    mmd_pvalue = mmd_permutation_pvalue(
        cpu,
        kv260,
        bandwidth=bandwidth,
        seed=seed,
        n_permutations=n_permutations,
        max_permutation_samples=max_permutation_samples,
    )
    ks = stats.ks_2samp(cpu, kv260, alternative="two-sided", method="auto")
    return {
        "bandwidth": float(bandwidth),
        "mmd_squared": float(mmd_value),
        "mmd_pvalue": float(mmd_pvalue),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def blocked_artifact(
    *,
    verdict: str,
    preconditions_checked: list[dict[str, Any]],
    duration_s: float,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2938-kv260-mmd-vs-cpu-sequential-gibbs-v1",
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "cpu_sequential_gibbs_energies_sha256": "",
        "kv260_synchronous_glauber_energies_sha256": "",
        "bitstream_sha256_cited": "",
        "per_seed_mmd_squared": [],
        "per_seed_mmd_pvalue": [],
        "per_seed_ks_statistic": [],
        "per_seed_ks_pvalue": [],
        "distributions_distinguishable": False,
        "paper_v6_recommendation": recommendation,
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "methodology_note": "",
        "duration_s": float(duration_s),
    }


def _recommendation(distributions_distinguishable: bool) -> str:
    if distributions_distinguishable:
        return (
            "retract: distributions distinguishable at p<0.01; paper-v6 must retract "
            'the "exact sampling on FPGA" claim and frame KV260 outputs as '
            "fixed-schedule heuristic samples."
        )
    return (
        "retain: all three MMD and KS p-values are >=0.01; paper-v6 can retain "
        "only the narrow approximately Boltzmann claim."
    )


def build_success_artifact(
    *,
    problems: Sequence[DenseIsingProblem],
    cpu_runs: Mapping[int, EnergyRunResult],
    hardware: HardwareRunResult,
    comparisons: Mapping[int, Mapping[str, float]],
    duration_s: float,
    cpu_n_samples: int,
    cpu_burn_in_sweeps: int,
    n_permutations: int,
    max_permutation_samples: int,
) -> dict[str, Any]:
    per_seed_mmd = [float(comparisons[seed]["mmd_squared"]) for seed in RANDOM_SEEDS]
    per_seed_mmd_p = [float(comparisons[seed]["mmd_pvalue"]) for seed in RANDOM_SEEDS]
    per_seed_ks = [float(comparisons[seed]["ks_statistic"]) for seed in RANDOM_SEEDS]
    per_seed_ks_p = [float(comparisons[seed]["ks_pvalue"]) for seed in RANDOM_SEEDS]
    distinguishable = any(
        pvalue < SIGNIFICANCE_ALPHA for pvalue in [*per_seed_mmd_p, *per_seed_ks_p]
    )
    cpu_payload = {str(seed): cpu_runs[seed].energies for seed in RANDOM_SEEDS}
    kv260_payload = {str(seed): hardware.energies_by_seed[seed] for seed in RANDOM_SEEDS}
    reproducibility_payload = {
        "problem_checksums": {
            str(problem.seed): {
                "j_matrix_sha256": problem.j_matrix_sha256,
                "h_vector_sha256": problem.h_vector_sha256,
            }
            for problem in problems
        },
        "cpu_energy_sha256": sha256_canonical(cpu_payload),
        "kv260_energy_sha256": sha256_canonical(kv260_payload),
        "bitstream_sha256": hardware.bitstream_sha256,
        "mmd": per_seed_mmd,
        "mmd_p": per_seed_mmd_p,
        "ks": per_seed_ks,
        "ks_p": per_seed_ks_p,
    }
    methodology = (
        f"CPU baseline: dense n=64 sequential single-spin Gibbs, random spin order, "
        f"beta=1.0, {cpu_burn_in_sweeps} burn-in sweeps, {cpu_n_samples} post-burn-in "
        f"energies per seed. KV260: existing Exp 2898 UIO upload and synchronous "
        f"fixed-sweep Glauber schedule, {cpu_n_samples} energies per seed, bitstream "
        f"SHA256 cited from board. MMD² uses an RBF kernel with median pairwise "
        f"distance bandwidth over the full traces; MMD p-value uses {n_permutations} "
        f"balanced permutations with max_permutation_samples={max_permutation_samples} "
        f"and deterministic random Fourier acceleration for large traces. KS uses "
        f"scipy.stats.ks_2samp."
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment": "exp2938-kv260-mmd-vs-cpu-sequential-gibbs-v1",
        "run_date": RUN_DATE,
        "honest_verdict": "complete: kv260_mmd_vs_cpu_sequential_gibbs_recorded",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": hardware.preconditions_checked,
        "cpu_sequential_gibbs_energies_sha256": sha256_canonical(cpu_payload),
        "kv260_synchronous_glauber_energies_sha256": sha256_canonical(kv260_payload),
        "bitstream_sha256_cited": hardware.bitstream_sha256,
        "per_seed_mmd_squared": per_seed_mmd,
        "per_seed_mmd_pvalue": per_seed_mmd_p,
        "per_seed_ks_statistic": per_seed_ks,
        "per_seed_ks_pvalue": per_seed_ks_p,
        "per_seed_bandwidth": [float(comparisons[seed]["bandwidth"]) for seed in RANDOM_SEEDS],
        "distributions_distinguishable": bool(distinguishable),
        "paper_v6_recommendation": _recommendation(bool(distinguishable)),
        "random_seeds_used": list(RANDOM_SEEDS),
        "reproducibility_checksum": sha256_canonical(reproducibility_payload),
        "methodology_note": methodology,
        "duration_s": float(duration_s),
        "cpu_update_schedule": CPU_UPDATE_SCHEDULE,
        "kv260_update_schedule": KV260_UPDATE_SCHEDULE,
        "energy_samples_per_seed": int(cpu_n_samples),
        "cpu_burn_in_sweeps": int(cpu_burn_in_sweeps),
        "n_permutations": int(n_permutations),
        "max_permutation_samples": int(max_permutation_samples),
        "hardware_transcript_path": hardware.transcript_path,
        "board_summary": hardware.board_summary,
        "cpu_spin_orders_sha256_by_seed": {
            str(seed): cpu_runs[seed].spin_orders_sha256 for seed in RANDOM_SEEDS
        },
        "source_artifacts": [EXP2898_REL_PATH.as_posix()],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    verdict = str(artifact["honest_verdict"])
    if verdict.startswith("blocked_"):
        return
    if not (verdict.startswith("complete:") or verdict.startswith("success:")):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    for field_name in (
        "per_seed_mmd_squared",
        "per_seed_mmd_pvalue",
        "per_seed_ks_statistic",
        "per_seed_ks_pvalue",
    ):
        if len(artifact[field_name]) != 3:
            raise ValueError(f"{field_name} must contain three seed values")
    if float(artifact["duration_s"]) < 60.0:
        raise ValueError("successful duration_s must be >= 60")
    if artifact["random_seeds_used"] != RANDOM_SEEDS:
        raise ValueError("random_seeds_used must match Exp 2898 seeds")
    for field_name in (
        "cpu_sequential_gibbs_energies_sha256",
        "kv260_synchronous_glauber_energies_sha256",
        "bitstream_sha256_cited",
        "reproducibility_checksum",
    ):
        value = artifact[field_name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field_name} must be a sha256 string")


def _precondition(resource: str, available: bool, detail: str) -> dict[str, Any]:
    return {"resource": resource, "available": bool(available), "detail": detail}


def _run(cmd: list[str], timeout: int | float) -> CommandResult:  # pragma: no cover
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return CommandResult(
            cmd, proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - started
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            cmd,
            124,
            exc.stdout if isinstance(exc.stdout, str) else "",
            exc.stderr if isinstance(exc.stderr, str) else f"timeout after {timeout}s",
            time.perf_counter() - started,
        )
    except OSError as exc:
        return CommandResult(
            cmd, 127, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started
        )


def _ssh(
    remote_cmd: str, timeout: int | float = 30, batch_mode: bool = False
) -> CommandResult:  # pragma: no cover
    cmd = ["ssh", "-o", "ConnectTimeout=5"]
    if batch_mode:
        cmd += ["-o", "BatchMode=yes"]
    cmd += [KV260_HOST, remote_cmd]
    return _run(cmd, timeout=timeout)


def _scp(local: Path, remote: str, timeout: int | float = 60) -> CommandResult:  # pragma: no cover
    return _run(["scp", "-o", "ConnectTimeout=5", str(local), f"{KV260_HOST}:{remote}"], timeout)


def _detect_overlay(text: str) -> str | None:
    for overlay in VALID_OVERLAYS:
        if overlay in text:
            return overlay
    return None


def _parse_sha256sum(output: str) -> tuple[str | None, str | None]:
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


def _problem_payload_for_board(
    problems: Sequence[DenseIsingProblem], n_samples: int
) -> dict[str, Any]:
    rows = []
    for problem in problems:
        rows.append(
            {
                "n_spins": problem.n_spins,
                "random_seed": problem.seed,
                "beta_final_q88": problem.beta_final_q88,
                "j_matrix": problem.j_matrix.tolist(),
                "h_vector": problem.h_vector.tolist(),
                "upload": problem.upload,
            }
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "n_spins": N_SPINS,
        "random_seeds_used": list(RANDOM_SEEDS),
        "n_samples": int(n_samples),
        "problems": rows,
    }


def run_kv260_synchronous_glauber(
    problems: Sequence[DenseIsingProblem],
    *,
    n_samples: int = N_ENERGY_SAMPLES,
    root_path: Path = REPO_ROOT,
) -> HardwareRunResult:  # pragma: no cover - requires live KV260
    transcript = Transcript(root_path / TRANSCRIPT_REL_PATH)
    transcript.write("experiment_2938 started")
    preconditions: list[dict[str, Any]] = []

    ssh_result = _run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", KV260_HOST, "true"],
        timeout=10,
    )
    transcript.record_result("precondition_ssh", ssh_result)
    ssh_ok = ssh_result.returncode == 0
    preconditions.append(_precondition("kv260_ssh", ssh_ok, f"rc={ssh_result.returncode}"))
    if not ssh_ok:
        return HardwareRunResult(
            preconditions,
            "",
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_kv260_ssh_unreachable",
        )

    list_result = _ssh("sudo xmutil listapps 2>&1", timeout=20)
    transcript.record_result("precondition_overlay_list", list_result)
    overlay_ok = list_result.returncode == 0 and _detect_overlay(
        list_result.stdout + list_result.stderr
    )
    preconditions.append(
        _precondition(
            "kv260_overlay", bool(overlay_ok), (list_result.stdout + list_result.stderr)[:300]
        )
    )
    if not overlay_ok:
        return HardwareRunResult(
            preconditions,
            "",
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_kv260_overlay_missing",
        )

    load_result = _ssh(OVERLAY_LOAD_COMMAND, timeout=60)
    transcript.record_result("overlay_load", load_result)

    uio_result = _ssh("ls /dev/uio0 2>/dev/null && echo ok", timeout=20)
    transcript.record_result("precondition_uio0", uio_result)
    uio_ok = uio_result.returncode == 0 and "ok" in uio_result.stdout.split()
    preconditions.append(_precondition("kv260_uio0", uio_ok, uio_result.stdout.strip()[:200]))
    if not uio_ok:
        return HardwareRunResult(
            preconditions,
            "",
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_kv260_uio_devices_absent",
        )

    bit_result = _ssh(
        "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit 2>/dev/null | head -n 1",
        timeout=30,
    )
    transcript.record_result("bitstream_sha256_bit", bit_result)
    bitstream_sha, bitstream_path = _parse_sha256sum(bit_result.stdout)
    if bitstream_sha is None:
        bit_bin = _ssh(
            "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit.bin 2>/dev/null | head -n 1",
            timeout=30,
        )
        transcript.record_result("bitstream_sha256_bit_bin", bit_bin)
        bitstream_sha, bitstream_path = _parse_sha256sum(bit_bin.stdout)
    bit_ok = bitstream_sha is not None
    preconditions.append(
        _precondition("active_bitstream_sha256", bit_ok, bitstream_sha or "missing")
    )
    if not bit_ok:
        return HardwareRunResult(
            preconditions,
            "",
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_active_bitstream_sha256_missing",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROBLEM_PATH.write_text(
        json.dumps(
            _problem_payload_for_board(problems, n_samples), sort_keys=True, separators=(",", ":")
        ),
        encoding="utf-8",
    )
    LOCAL_HARNESS_PATH.write_text(BOARD_HARNESS_SOURCE, encoding="utf-8")

    problem_scp = _scp(LOCAL_PROBLEM_PATH, REMOTE_PROBLEM_PATH, timeout=60)
    transcript.record_result("scp_problem_payload", problem_scp)
    if problem_scp.returncode != 0:
        return HardwareRunResult(
            preconditions,
            bitstream_sha,
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_kv260_problem_scp_failed",
        )
    harness_scp = _scp(LOCAL_HARNESS_PATH, REMOTE_HARNESS_PATH, timeout=60)
    transcript.record_result("scp_board_harness", harness_scp)
    if harness_scp.returncode != 0:
        return HardwareRunResult(
            preconditions,
            bitstream_sha,
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_kv260_harness_scp_failed",
        )

    harness = _ssh(f"sudo python3 {REMOTE_HARNESS_PATH} {REMOTE_PROBLEM_PATH}", timeout=1800)
    transcript.record_result("run_board_harness", harness)
    if harness.returncode != 0:
        return HardwareRunResult(
            preconditions,
            bitstream_sha,
            {},
            TRANSCRIPT_REL_PATH.as_posix(),
            {},
            "blocked_kv260_board_harness_failed",
        )
    board_payload = _extract_board_json(harness.stdout)
    energies_by_seed = {
        int(seed): [float(value) for value in values]
        for seed, values in board_payload.get("energies_by_seed", {}).items()
    }
    return HardwareRunResult(
        preconditions_checked=preconditions,
        bitstream_sha256=bitstream_sha,
        energies_by_seed=energies_by_seed,
        transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        board_summary={
            "bitstream_path": bitstream_path,
            "selected_uio": board_payload.get("selected_uio"),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            "board_harness_duration_s": board_payload.get("duration_s"),
        },
    )


def run_experiment(
    *,
    root_path: Path = REPO_ROOT,
    hardware_runner: Callable[[list[DenseIsingProblem]], HardwareRunResult] | None = None,
    cpu_energy_runner: Callable[..., EnergyRunResult] | None = None,
    cpu_n_samples: int = N_ENERGY_SAMPLES,
    cpu_burn_in_sweeps: int = CPU_BURN_IN_SWEEPS,
    n_permutations: int = N_PERMUTATIONS,
    max_permutation_samples: int = MAX_PERMUTATION_SAMPLES,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    exp2898_path = root_path / EXP2898_REL_PATH
    if not exp2898_path.exists():
        artifact = blocked_artifact(
            verdict="blocked_exp2898_artifact_missing",
            preconditions_checked=[],
            duration_s=_duration(started, now_s),
            recommendation="blocked_exp2898_artifact_missing: cannot reproduce Exp 2898 problems.",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    try:
        exp2898 = json.loads(exp2898_path.read_text(encoding="utf-8"))
        problems = recover_exp2898_problems(exp2898)
    except (json.JSONDecodeError, OSError, ProblemReproductionError) as exc:
        artifact = blocked_artifact(
            verdict="blocked_exp2898_problem_reproduction_failed",
            preconditions_checked=[],
            duration_s=_duration(started, now_s),
            recommendation=f"blocked_exp2898_problem_reproduction_failed: {exc}",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    runner = hardware_runner or (
        lambda active_problems: run_kv260_synchronous_glauber(
            active_problems, n_samples=cpu_n_samples, root_path=root_path
        )
    )
    hardware = runner(problems)
    if hardware.blocked_verdict:
        artifact = blocked_artifact(
            verdict=hardware.blocked_verdict,
            preconditions_checked=hardware.preconditions_checked,
            duration_s=_duration(started, now_s),
            recommendation=f"{hardware.blocked_verdict}: KV260 comparison did not run.",
        )
        artifact["bitstream_sha256_cited"] = hardware.bitstream_sha256
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    exp2898_bitstream = str(exp2898.get("bitstream_sha256", ""))
    if hardware.bitstream_sha256 != exp2898_bitstream:
        artifact = blocked_artifact(
            verdict="blocked_active_bitstream_sha256_mismatch",
            preconditions_checked=hardware.preconditions_checked,
            duration_s=_duration(started, now_s),
            recommendation="blocked_active_bitstream_sha256_mismatch: active board SHA does not match Exp 2898.",
        )
        artifact["bitstream_sha256_cited"] = hardware.bitstream_sha256
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    cpu_runner = cpu_energy_runner or run_cpu_sequential_gibbs
    cpu_runs = {
        problem.seed: cpu_runner(
            problem,
            n_samples=cpu_n_samples,
            burn_in_sweeps=cpu_burn_in_sweeps,
        )
        for problem in problems
    }

    comparisons: dict[int, dict[str, float]] = {}
    for seed in RANDOM_SEEDS:
        if len(hardware.energies_by_seed.get(seed, [])) != int(cpu_n_samples):
            artifact = blocked_artifact(
                verdict="blocked_kv260_energy_trace_incomplete",
                preconditions_checked=hardware.preconditions_checked,
                duration_s=_duration(started, now_s),
                recommendation="blocked_kv260_energy_trace_incomplete: missing per-sample hardware energies.",
            )
            artifact["bitstream_sha256_cited"] = hardware.bitstream_sha256
            validate_artifact(artifact)
            _write_json(output_path, artifact)
            return artifact
        comparisons[seed] = compare_energy_distributions(
            cpu_runs[seed].energies,
            hardware.energies_by_seed[seed],
            seed=seed,
            n_permutations=n_permutations,
            max_permutation_samples=max_permutation_samples,
        )

    artifact = build_success_artifact(
        problems=problems,
        cpu_runs=cpu_runs,
        hardware=hardware,
        comparisons=comparisons,
        duration_s=_duration(started, now_s),
        cpu_n_samples=cpu_n_samples,
        cpu_burn_in_sweeps=cpu_burn_in_sweeps,
        n_permutations=n_permutations,
        max_permutation_samples=max_permutation_samples,
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(root_path=args.root)
    if args.print_result_path:
        print(args.root / OUTPUT_REL_PATH)
    else:
        print(
            json.dumps(
                {
                    "honest_verdict": artifact["honest_verdict"],
                    "result": str(args.root / OUTPUT_REL_PATH),
                }
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
