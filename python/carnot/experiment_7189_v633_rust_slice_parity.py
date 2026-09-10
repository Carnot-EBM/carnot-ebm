"""Measure compiled Rust parity for Exp7187's fixed-cardinality pair swaps.

The experiment keeps random-number differences separate from algorithm
differences.  Exact replay uses caller-owned proposal and acceptance tapes,
while the distribution checks deliberately let Rust and Python use their own
random streams.  The Rust executable is a process bridge, not a Python FFI.

Spec: REQ-SAMPLER-7189 and SCENARIO-SAMPLER-7189-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices


JsonDict = dict[str, Any]
State = tuple[int, ...]

RUN_DATE = "20260910"
TASK_ID = "exp7189-rust-slice-parity"
MILESTONE = "2026.09.633"
RESULT_PATH = Path("results/experiment_7189_v633_rust_slice_parity.json")
CHECKPOINT_DIR = Path("results/checkpoints")
SPEC_PATH = Path("openspec/capabilities/samplers/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
UPSTREAM_RESULT_PATH = Path("results/experiment_7187_v633_slice_sampler.json")
BRIDGE_RELATIVE_PATH = Path("target/release/fixed-cardinality-bridge")
TOLERANCE = 1.0e-10

REPLAY_STEPS = 32
REPLAY_SEED = 718900
DISTRIBUTION_SEEDS = tuple(range(718900, 718910))
DISTRIBUTION_BURN_IN = 2_000
DISTRIBUTION_RETAINED = 20_000
TV_LIMIT = 0.15
ENERGY_MEAN_LIMIT = 0.25
THROUGHPUT_SIZES = (32, 64)
THROUGHPUT_CARDINALITIES = (2, 4)
ENERGY_BUDGET = slices.ENERGY_BUDGET
WALL_TIME_BUDGET_S = slices.WALL_TIME_BUDGET_S

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": [
        {
            "upstream": "exp7187-slice-sampler",
            "artifact_field": "slice_sampler_ready_score",
            "op": "==",
            "value": 1,
            "principle": "Require complete valid inputs, not a positive scientific outcome.",
        }
    ],
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    UPSTREAM_RESULT_PATH,
    Path("results/experiment_7145_v627_rust_multiscale_sampler.json"),
    Path("crates/carnot-samplers/Cargo.toml"),
    Path("crates/carnot-samplers/src/lib.rs"),
    Path("crates/carnot-samplers/src/fixed_cardinality.rs"),
    Path("crates/carnot-samplers/src/bin/fixed-cardinality-bridge.rs"),
    Path("crates/carnot-samplers/tests/fixed_cardinality.rs"),
    Path("crates/carnot-ising/src/lib.rs"),
    Path("python/carnot/samplers/backend.py"),
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7189_v633_rust_slice_parity.py"),
    SPEC_PATH,
    Path("scripts/experiments/experiment_7189_v633_rust_slice_parity.py"),
    Path("tests/python/test_experiment_7189_v633_rust_slice_parity.py"),
)
HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

FIELD_PRINCIPLES = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": (
        "Name each resource and record its actual availability before measurement."
    ),
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": (
        "Hash input contracts, code, seeds, and raw rows to expose drift."
    ),
    "gate_check_summary": (
        "Every blocked verdict names the exact failed check, upstream, field, expected value, "
        "and observed value."
    ),
    "verifier_is_oracle": (
        "Declare whether the scored verifier uses the same authority that labels the outcome."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. "
        "Only unfinished own work is partial."
    ),
    "honest_verdict": (
        "A terminal description distinguishes useful evidence, null findings, "
        "disqualification, and external blocks."
    ),
    "inference_substrate_class": (
        "Use cpu_exact_solver_or_simulator when the declared work runs; use blocked_no_run "
        "only before any qualifying work."
    ),
    "rust_slice_parity_score": (
        "One requires compiled cross-language execution and law agreement."
    ),
    "replay_tape_manifest": (
        "Explicit random tapes isolate algorithm parity from RNG differences."
    ),
    "cross_language_rows": "Each proposal and acceptance must be reconstructable.",
    "throughput_rows": "Per-case latency and ESS expose real deployment cost.",
    "e2e_receipts": ("Actual pipeline commands distinguish integration from unit-only evidence."),
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = REQUIRED_ARTIFACT_FIELDS | {
    "task_id",
    "milestone",
    "compiled_rust_execution",
    "distribution_rows",
    "throughput_raw_rows",
    "speedup_rows",
    "nfr_01_10x_met",
    "performance_verdict_class",
    "hardware_execution_claimed",
    "pyo3_execution_claimed",
    "multiscale_execution_claimed",
    "quantized_delayed_acceptance_claimed",
    "mixing_theorem_claimed",
}


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush phase boundaries so a conductor can distinguish work from a stall."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


class _Heartbeat:
    """Print bounded liveness receipts while Python waits on native work."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.started = time.monotonic()
        self.completed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _Heartbeat:
        print(f"[native start] operation={self.operation}", flush=True)
        self._thread.start()
        return self

    def __exit__(self, *_unused: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        elapsed = time.monotonic() - self.started
        print(
            f"[native end] elapsed_s={elapsed:.3f} completed={self.completed} "
            f"operation={self.operation}",
            flush=True,
        )

    def _run(self) -> None:  # pragma: no cover - only a host slower than 50 seconds reaches this.
        while not self._stop.wait(50.0):
            elapsed = time.monotonic() - self.started
            print(
                f"[heartbeat] elapsed_s={elapsed:.3f} completed={self.completed} "
                f"operation={self.operation}",
                flush=True,
            )


def canonical_json(value: Any) -> str:
    """Encode stable JSON after rejecting values that standard JSON cannot replay."""

    def reject_nonfinite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite values are not valid JSON evidence")
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject_nonfinite(key)
                reject_nonfinite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject_nonfinite(child)

    reject_nonfinite(value)
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Name the digest algorithm so stored evidence cannot be misread as another hash."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_json(value: Any) -> str:
    """Hash canonical JSON rather than interpreter-specific object formatting."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a complete file as bytes, including executable and serialized inputs."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every stable input and result while excluding the checksum's own value."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the exact same-milestone task fields from the executable roadmap."""

    path = root / ROADMAP_PATH
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (row for row in document["tasks"] if isinstance(row, Mapping) and row.get("id") == TASK_ID),
        None,
    )
    if task is None:
        return None
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "gated_on": task.get("gated_on"),
    }


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record actual bytes, tools, directories, contracts, and the producer gate."""

    result = result_path or root / RESULT_PATH
    checkpoints = checkpoint_dir or root / CHECKPOINT_DIR
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    hashes = {
        str(path): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[str(path)] is not None and sizes[str(path)] > 0
    }
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    upstream_file = root / UPSTREAM_RESULT_PATH
    try:
        upstream = json.loads(upstream_file.read_text(encoding="utf-8"))
        upstream_score = upstream.get("slice_sampler_ready_score")
    except (OSError, json.JSONDecodeError, AttributeError):
        upstream_score = None
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    tool_observed = {
        "python": bool(sys.executable),
        "numpy": bool(np.__version__),
        "pyyaml": bool(yaml.__version__),
        "cargo": cargo is not None,
        "rustc": rustc is not None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyyaml_version": yaml.__version__,
        "cargo_path": cargo,
        "rustc_path": rustc,
    }
    checks = [
        {
            "check": "driving_capability_spec",
            "upstream": str(SPEC_PATH),
            "field": "REQ-SAMPLER-7189",
            "expected_value": {"exists": True, "req_present": True},
            "observed_value": {
                "exists": spec_file.is_file(),
                "req_present": "REQ-SAMPLER-7189" in spec_text,
            },
            "passed": spec_file.is_file() and "REQ-SAMPLER-7189" in spec_text,
        },
        {
            "check": "required_source_bytes",
            "upstream": "repository",
            "field": "byte_count",
            "expected_value": "every required source is nonempty",
            "observed_value": sizes,
            "passed": all(size is not None and size > 0 for size in sizes.values()),
        },
        {
            "check": "same_milestone_gate_fields",
            "upstream": str(ROADMAP_PATH),
            "field": "task_contract",
            "expected_value": EXPECTED_TASK_CONTRACT,
            "observed_value": _task_contract(root),
            "passed": _task_contract(root) == EXPECTED_TASK_CONTRACT,
        },
        {
            "check": "required_tools",
            "upstream": "host_toolchain",
            "field": "python_numpy_pyyaml_cargo_rustc",
            "expected_value": {
                "python": True,
                "numpy": True,
                "pyyaml": True,
                "cargo": True,
                "rustc": True,
            },
            "observed_value": tool_observed,
            "passed": all(
                tool_observed[name] for name in ("python", "numpy", "pyyaml", "cargo", "rustc")
            ),
        },
        {
            "check": "output_directories",
            "upstream": "filesystem",
            "field": "result_and_checkpoint_parent",
            "expected_value": {"result_parent": True, "checkpoint_dir": True},
            "observed_value": {
                "result_parent": result.parent.is_dir(),
                "checkpoint_dir": checkpoints.is_dir(),
            },
            "passed": result.parent.is_dir() and checkpoints.is_dir(),
        },
        {
            "check": "source_artifact_hashes",
            "upstream": "required_source_bytes",
            "field": "sha256",
            "expected_value": len(REQUIRED_SOURCE_PATHS),
            "observed_value": len(hashes),
            "passed": len(hashes) == len(REQUIRED_SOURCE_PATHS)
            and all(HASH_PATTERN.fullmatch(value) for value in hashes.values()),
        },
        {
            "check": "upstream_slice_sampler_gate",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "slice_sampler_ready_score",
            "expected_value": 1,
            "observed_value": upstream_score,
            "passed": upstream_score == 1,
        },
    ]
    return checks, hashes


def build_rust_bridge(root: Path) -> Path:
    """Compile the small process bridge and return the real executable path."""

    command = [
        "cargo",
        "build",
        "--release",
        "-p",
        "carnot-samplers",
        "--bin",
        "fixed-cardinality-bridge",
    ]
    with _Heartbeat("cargo build fixed-cardinality-bridge") as heartbeat:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
        returncode = process.wait(timeout=480)
        heartbeat.completed = 1
    if returncode != 0:
        raise RuntimeError(f"Rust bridge build failed with exit {returncode}")
    bridge = root / BRIDGE_RELATIVE_PATH
    if not bridge.is_file() or not os.access(bridge, os.X_OK):
        raise RuntimeError(f"Rust bridge is not executable: {bridge}")
    return bridge


def instance_payload(instance: slices.SliceInstance, cardinality: int, beta: float) -> JsonDict:
    """Serialize the exact edge-once energy contract consumed by the Rust core."""

    slices.validate_instance(instance)
    slices.enumerate_slice(instance.n, cardinality)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    return {
        "edges": [
            {"left": left, "right": right, "coupling": coupling}
            for left, right, coupling in instance.edges
        ],
        "fields": list(instance.fields),
        "cardinality": cardinality,
        "beta": beta,
    }


def python_energy(instance: slices.SliceInstance, state: Sequence[int]) -> float:
    """Use Exp7187's edge-once energy as the Python side of conformance."""

    return slices.ising_energy(instance, state)


def make_replay_tape(n: int, k: int, *, seed: int, steps: int) -> list[JsonDict]:
    """Freeze proposal-list indices and uniforms without depending on a chain state."""

    slices.enumerate_slice(n, k)
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    rng = random.Random(seed)
    return [
        {
            "positive_index": rng.randrange(max(1, k)),
            "negative_index": rng.randrange(max(1, n - k)),
            "uniform": rng.random(),
        }
        for _ in range(steps)
    ]


def python_replay(
    instance: slices.SliceInstance,
    cardinality: int,
    beta: float,
    initial_state: Sequence[int],
    tape: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Apply the Rust acceptance contract exactly to a caller-supplied tape."""

    if len(initial_state) != instance.n:
        raise ValueError("state length must equal n")
    if any(spin not in (-1, 1) for spin in initial_state):
        raise ValueError("state spins must be -1 or +1")
    if sum(spin == 1 for spin in initial_state) != cardinality:
        raise ValueError("state cardinality does not match k")
    config = instance_payload(instance, cardinality, beta)
    state = tuple(initial_state)
    outcomes: list[JsonDict] = []
    for draw in tape:
        uniform = draw.get("uniform")
        if (
            not isinstance(uniform, (int, float))
            or not math.isfinite(uniform)
            or not 0.0 <= uniform < 1.0
        ):
            raise ValueError("uniform must be finite and in [0, 1)")
        current_energy = python_energy(instance, state)
        if cardinality in (0, instance.n):
            candidate = state
            candidate_energy = current_energy
            delta = 0.0
            log_threshold = 0.0
            accepted = True
        else:
            positive = [index for index, spin in enumerate(state) if spin == 1]
            negative = [index for index, spin in enumerate(state) if spin == -1]
            try:
                positive_site = positive[int(draw["positive_index"])]
                negative_site = negative[int(draw["negative_index"])]
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise ValueError("proposal index is out of range") from exc
            candidate_list = list(state)
            candidate_list[positive_site], candidate_list[negative_site] = (
                candidate_list[negative_site],
                candidate_list[positive_site],
            )
            candidate = tuple(candidate_list)
            candidate_energy = python_energy(instance, candidate)
            delta = candidate_energy - current_energy
            log_threshold = min(0.0, -config["beta"] * delta)
            accepted = math.log(max(float(uniform), sys.float_info.min)) < log_threshold
        if accepted:
            state = candidate
        outcomes.append(
            {
                "state": list(state),
                "proposed_state": list(candidate),
                "current_energy": current_energy,
                "proposed_energy": candidate_energy,
                "delta_energy": delta,
                "log_acceptance": log_threshold,
                "accepted": accepted,
                "cardinality": cardinality,
            }
        )
    return {
        "initial_state": list(initial_state),
        "final_state": list(state),
        "steps": outcomes,
    }


def replay_request(
    instance: slices.SliceInstance,
    cardinality: int,
    beta: float,
    initial_state: Sequence[int],
    tape: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one complete request so its bytes can be hashed before execution."""

    return {
        "operation": "replay",
        "config": instance_payload(instance, cardinality, beta),
        "initial_state": list(initial_state),
        "tape": [dict(draw) for draw in tape],
    }


def rust_request(
    bridge: Path,
    request: Mapping[str, Any],
    *,
    timeout_s: float = 120.0,
) -> tuple[JsonDict, JsonDict]:
    """Execute compiled Rust and retain command, timing, and exact byte hashes."""

    started = time.monotonic()
    request_bytes = canonical_json(request).encode("utf-8")
    with _Heartbeat(f"compiled Rust bridge operation={request.get('operation')}") as heartbeat:
        completed = subprocess.run(
            [str(bridge)],
            input=request_bytes,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        heartbeat.completed = 1
    elapsed = time.monotonic() - started
    receipt = {
        "command": [str(bridge)],
        "operation": request.get("operation"),
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "request_sha256": sha256_bytes(request_bytes),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stdout_bytes": len(completed.stdout),
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"compiled Rust bridge failed with exit {completed.returncode}: {receipt['stderr']}"
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("compiled Rust bridge returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError("compiled Rust bridge response must be an object")
    return response, receipt


def compare_replays(
    case_id: str,
    tape: Sequence[Mapping[str, Any]],
    python_result: Mapping[str, Any],
    rust_result: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit one reconstructable conformance row for every shared-tape proposal."""

    python_steps = python_result.get("steps", [])
    rust_steps = rust_result.get("steps", [])
    if len(python_steps) != len(tape) or len(rust_steps) != len(tape):
        raise ValueError("replay result length does not match the tape")
    rows: list[JsonDict] = []
    python_before = python_result["initial_state"]
    rust_before = rust_result["initial_state"]
    for index, (draw, python_step, rust_step) in enumerate(
        zip(tape, python_steps, rust_steps, strict=True)
    ):
        delta_error = abs(python_step["delta_energy"] - rust_step["delta_energy"])
        row = {
            "row_type": "cross_language_replay",
            "unit_id": f"{case_id}:step{index}",
            "case_id": case_id,
            "step": index,
            "python_input_state": python_before,
            "rust_input_state": rust_before,
            "positive_index": draw["positive_index"],
            "negative_index": draw["negative_index"],
            "uniform": draw["uniform"],
            "python_accepted": python_step["accepted"],
            "rust_accepted": rust_step["accepted"],
            "python_cardinality": python_step["cardinality"],
            "rust_cardinality": rust_step["cardinality"],
            "python_delta_energy": python_step["delta_energy"],
            "rust_delta_energy": rust_step["delta_energy"],
            "delta_energy_error": delta_error,
            "python_proposed_state": python_step["proposed_state"],
            "rust_proposed_state": rust_step["proposed_state"],
            "python_state": python_step["state"],
            "rust_state": rust_step["state"],
        }
        row["passed"] = (
            python_before == rust_before
            and python_step["accepted"] == rust_step["accepted"]
            and python_step["cardinality"] == rust_step["cardinality"]
            and python_step["proposed_state"] == rust_step["proposed_state"]
            and python_step["state"] == rust_step["state"]
            and delta_error <= TOLERANCE
        )
        row["row_sha256"] = sha256_json(row)
        rows.append(row)
        python_before = python_step["state"]
        rust_before = rust_step["state"]
    return rows


def _python_seeded_chain(
    instance: slices.SliceInstance,
    *,
    cardinality: int,
    beta: float,
    initial_state: Sequence[int],
    seed: int,
    burn_in: int,
    retained: int,
) -> JsonDict:
    """Run Python's own RNG stream while keeping the same transition law."""

    rng = random.Random(seed)
    state = tuple(initial_state)
    current_energy = python_energy(instance, state)
    samples: list[list[int]] = []
    energies: list[float] = []
    accepted = 0
    for index in range(burn_in + retained):
        positive = [site for site, spin in enumerate(state) if spin == 1]
        negative = [site for site, spin in enumerate(state) if spin == -1]
        first = positive[rng.randrange(len(positive))]
        second = negative[rng.randrange(len(negative))]
        candidate_list = list(state)
        candidate_list[first], candidate_list[second] = (
            candidate_list[second],
            candidate_list[first],
        )
        candidate = tuple(candidate_list)
        candidate_energy = python_energy(instance, candidate)
        if math.log(max(rng.random(), sys.float_info.min)) < min(
            0.0, -beta * (candidate_energy - current_energy)
        ):
            state = candidate
            current_energy = candidate_energy
            accepted += 1
        if index >= burn_in:
            samples.append(list(state))
            energies.append(current_energy)
    return {
        "samples": samples,
        "energies": energies,
        "accepted": accepted,
        "attempted": burn_in + retained,
        "energy_evaluations": 1 + burn_in + retained,
        "final_state": list(state),
    }


def _derived_seed(language: str, seed: int) -> int:
    """Domain-separate fixed study seeds so independent arms do not share a stream."""

    material = f"exp7189:{language}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _distribution_row(
    *,
    language: str,
    seed: int,
    chain: Mapping[str, Any],
    law: slices.ExactLaw,
    cardinality: int,
    tv_limit: float,
    energy_mean_limit: float,
    receipt: Mapping[str, Any] | None,
) -> JsonDict:
    """Compare one independent stream with the sealed complete finite law."""

    samples = [tuple(state) for state in chain["samples"]]
    counts = Counter(samples)
    retained = len(samples)
    empirical = [counts[state] / retained for state in law.states]
    tv = 0.5 * sum(
        abs(left - right) for left, right in zip(empirical, law.probabilities, strict=True)
    )
    exact_energy_mean = sum(
        probability * energy
        for probability, energy in zip(law.probabilities, law.energies, strict=True)
    )
    energy_mean = statistics.fmean(chain["energies"])
    cardinality_valid = all(sum(spin == 1 for spin in state) == cardinality for state in samples)
    histogram = [
        {
            "state": list(state),
            "count": counts[state],
            "probability": counts[state] / retained,
            "target_probability": probability,
            "energy": energy,
        }
        for state, probability, energy in zip(
            law.states, law.probabilities, law.energies, strict=True
        )
    ]
    row = {
        "row_type": "independent_distribution",
        "unit_id": f"{language}:seed{seed}",
        "language": language,
        "seed": seed,
        "effective_seed": _derived_seed(language, seed),
        "retained": retained,
        "attempted": chain["attempted"],
        "acceptance": chain["accepted"] / chain["attempted"],
        "total_variation": tv,
        "total_variation_limit": tv_limit,
        "energy_mean": energy_mean,
        "exact_energy_mean": exact_energy_mean,
        "energy_mean_error": abs(energy_mean - exact_energy_mean),
        "energy_mean_error_limit": energy_mean_limit,
        "cardinality": cardinality,
        "cardinality_valid": cardinality_valid,
        "histogram": histogram,
        "trace_sha256": sha256_json({"samples": samples, "energies": chain["energies"]}),
        "process_receipt": dict(receipt) if receipt is not None else None,
    }
    row["passed"] = (
        cardinality_valid and tv <= tv_limit and row["energy_mean_error"] <= energy_mean_limit
    )
    row["row_sha256"] = sha256_json(row)
    return row


def run_distribution_checks(
    bridge: Path,
    *,
    instance: slices.SliceInstance,
    law: slices.ExactLaw,
    seeds: Sequence[int],
    burn_in: int,
    retained: int,
    tv_limit: float,
    energy_mean_limit: float,
) -> list[JsonDict]:
    """Measure separate Python and Rust streams against one enumerated law."""

    if retained <= 0:
        raise ValueError("retained sample count must be positive")
    cardinality = law.states[0].count(1)
    beta = 2.0
    initial = law.states[0]
    rows: list[JsonDict] = []
    started = time.monotonic()
    heartbeat = started
    total = 2 * len(seeds)
    for seed in seeds:
        python_seed = _derived_seed("python", seed)
        python_chain = _python_seeded_chain(
            instance,
            cardinality=cardinality,
            beta=beta,
            initial_state=initial,
            seed=python_seed,
            burn_in=burn_in,
            retained=retained,
        )
        rows.append(
            _distribution_row(
                language="python",
                seed=seed,
                chain=python_chain,
                law=law,
                cardinality=cardinality,
                tv_limit=tv_limit,
                energy_mean_limit=energy_mean_limit,
                receipt=None,
            )
        )
        rust_seed = _derived_seed("rust", seed)
        request = {
            "operation": "seeded",
            "config": instance_payload(instance, cardinality, beta),
            "initial_state": list(initial),
            "seed": rust_seed,
            "burn_in": burn_in,
            "retained": retained,
        }
        rust_chain, receipt = rust_request(bridge, request)
        rows.append(
            _distribution_row(
                language="rust",
                seed=seed,
                chain=rust_chain,
                law=law,
                cardinality=cardinality,
                tv_limit=tv_limit,
                energy_mean_limit=energy_mean_limit,
                receipt=receipt,
            )
        )
        now = time.monotonic()
        if now - heartbeat >= 50.0:  # pragma: no cover - slow host only.
            print(
                f"[heartbeat] elapsed_s={now - started:.3f} completed={len(rows)}/{total} "
                "operation=independent_distribution_checks",
                flush=True,
            )
            heartbeat = now
    return rows


def run_e2e_receipt(bridge: Path, output_dir: Path) -> JsonDict:
    """Serialize one known energy and load the same bytes into both samplers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    instance = slices.SliceInstance(
        n=4,
        seed=7189,
        edges=((0, 1, 1.0), (0, 2, 1.0), (1, 2, -1.0)),
        fields=(0.1, -0.2, 0.3, -0.2),
    )
    initial = (1, 1, -1, -1)
    tape = [{"positive_index": 0, "negative_index": 0, "uniform": 0.5}]
    serialized = {
        "config": instance_payload(instance, 2, 2.0),
        "initial_state": list(initial),
        "tape": tape,
    }
    parameter_path = output_dir / "experiment_7189_e2e_parameters.json"
    parameter_path.write_text(canonical_json(serialized) + "\n", encoding="utf-8")
    loaded = json.loads(parameter_path.read_text(encoding="utf-8"))
    loaded_instance = slices.SliceInstance(
        n=len(loaded["config"]["fields"]),
        seed=7189,
        edges=tuple(
            (edge["left"], edge["right"], edge["coupling"]) for edge in loaded["config"]["edges"]
        ),
        fields=tuple(loaded["config"]["fields"]),
    )
    python_result = python_replay(loaded_instance, 2, 2.0, loaded["initial_state"], loaded["tape"])
    request = {"operation": "replay", **loaded}
    rust_result, process_receipt = rust_request(bridge, request)
    python_energy_value = python_result["steps"][0]["current_energy"]
    rust_energy_value = rust_result["steps"][0]["current_energy"]
    energy_error = abs(python_energy_value - rust_energy_value)
    receipt = {
        "scenario": "SCENARIO-SAMPLER-7189-E2E",
        "e2e_plan_refs": ["E2E-001", "E2E-002"],
        "scope": "known serialized Ising energy to slice-conditioned sampling; no CD-1 claim",
        "parameter_path": str(parameter_path),
        "parameter_sha256": sha256_file(parameter_path),
        "parameter_payload": serialized,
        "command": process_receipt["command"],
        "rust_returncode": process_receipt["returncode"],
        "rust_output_sha256": process_receipt["stdout_sha256"],
        "python_initial_energy": python_energy_value,
        "rust_initial_energy": rust_energy_value,
        "energy_error": energy_error,
        "python_cardinality": python_result["final_state"].count(1),
        "rust_cardinality": rust_result["final_state"].count(1),
    }
    receipt["passed"] = (
        receipt["rust_returncode"] == 0
        and energy_error <= TOLERANCE
        and receipt["python_cardinality"] == receipt["rust_cardinality"] == 2
    )
    return receipt


def _python_benchmark_chain(
    instance: slices.SliceInstance,
    *,
    cardinality: int,
    seed: int,
    energy_budget: int | None,
    wall_time_budget_s: float | None,
) -> JsonDict:
    """Run one Python chain under one of Exp7187's two stopping budgets."""

    if (energy_budget is None) == (wall_time_budget_s is None):
        raise ValueError("provide exactly one throughput budget")
    rng = random.Random(seed)
    initial = list(slices.enumerate_slice(instance.n, cardinality)[0])
    state = tuple(initial)
    started = time.monotonic()
    current_energy = python_energy(instance, state)
    energies: list[float] = [current_energy]
    accepted = 0
    attempted = 0
    while True:
        if energy_budget is not None and len(energies) >= energy_budget:
            break
        if (
            wall_time_budget_s is not None
            and energies
            and time.monotonic() - started >= wall_time_budget_s
        ):
            break
        positive = [site for site, spin in enumerate(state) if spin == 1]
        negative = [site for site, spin in enumerate(state) if spin == -1]
        first = positive[rng.randrange(len(positive))]
        second = negative[rng.randrange(len(negative))]
        candidate_list = list(state)
        candidate_list[first], candidate_list[second] = (
            candidate_list[second],
            candidate_list[first],
        )
        candidate = tuple(candidate_list)
        candidate_energy = python_energy(instance, candidate)
        if math.log(max(rng.random(), sys.float_info.min)) < min(
            0.0, -2.0 * (candidate_energy - current_energy)
        ):
            state = candidate
            current_energy = candidate_energy
            accepted += 1
        attempted += 1
        energies.append(current_energy)
    return {
        "samples": [],
        "energies": energies,
        "accepted": accepted,
        "attempted": attempted,
        "energy_evaluations": len(energies),
        "final_state": list(state),
    }


def _ess_metrics(energies: Sequence[float], latency_s: float) -> tuple[float | None, float | None]:
    """Return null instead of inventing ESS when a short chain is constant."""

    if len(energies) < 2:
        return None, None
    lag = min(slices.LAG_WINDOW, len(energies) - 1)
    ess = slices.effective_sample_size(energies, lag)
    return ess, ess / latency_s if ess is not None and latency_s > 0.0 else None


def run_throughput_benchmarks(
    bridge: Path,
    *,
    sizes: Sequence[int] = THROUGHPUT_SIZES,
    cardinalities: Sequence[int] = THROUGHPUT_CARDINALITIES,
    seeds: Sequence[int] = slices.BENCHMARK_SEEDS,
    energy_budget: int = ENERGY_BUDGET,
    wall_time_budget_s: float = WALL_TIME_BUDGET_S,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Charge serialization, process launch, and bridge output in every timing."""

    raw: list[JsonDict] = []
    started_all = time.monotonic()
    heartbeat = started_all
    total = len(sizes) * len(cardinalities) * len(seeds) * 2 * 2
    for n in sizes:
        for cardinality in cardinalities:
            for seed in seeds:
                for budget_name in ("matched_energy_evaluations", "matched_wall_time"):
                    instance = slices.make_frustrated_instance(n, seed)
                    initial = list(slices.enumerate_slice(n, cardinality)[0])
                    for language in ("python", "rust"):
                        started = time.monotonic()
                        if language == "python":
                            canonical_json(instance_payload(instance, cardinality, 2.0))
                            chain = _python_benchmark_chain(
                                instance,
                                cardinality=cardinality,
                                seed=seed,
                                energy_budget=energy_budget
                                if budget_name == "matched_energy_evaluations"
                                else None,
                                wall_time_budget_s=(
                                    wall_time_budget_s
                                    if budget_name == "matched_wall_time"
                                    else None
                                ),
                            )
                            process_receipt = None
                        else:
                            request: JsonDict = {
                                "operation": (
                                    "budgeted"
                                    if budget_name == "matched_energy_evaluations"
                                    else "timed"
                                ),
                                "config": instance_payload(instance, cardinality, 2.0),
                                "initial_state": initial,
                                "seed": seed,
                                "burn_in": 0,
                                "retained": energy_budget,
                            }
                            if budget_name == "matched_energy_evaluations":
                                request["energy_budget"] = energy_budget
                            if budget_name == "matched_wall_time":
                                request["max_duration_s"] = wall_time_budget_s
                            chain, process_receipt = rust_request(bridge, request)
                        latency = time.monotonic() - started
                        ess, ess_rate = _ess_metrics(chain["energies"], latency)
                        row = {
                            "row_type": "throughput_raw",
                            "unit_id": f"n{n}:k{cardinality}:seed{seed}:{language}:{budget_name}",
                            "n": n,
                            "k": cardinality,
                            "seed": seed,
                            "language": language,
                            "budget": budget_name,
                            "budget_value": (
                                energy_budget
                                if budget_name == "matched_energy_evaluations"
                                else wall_time_budget_s
                            ),
                            "latency_s": latency,
                            "work": chain["energy_evaluations"],
                            "proposal_attempts": chain["attempted"],
                            "acceptance": chain["accepted"] / max(1, chain["attempted"]),
                            "energy_ess": ess,
                            "ess_per_second": ess_rate,
                            "includes_setup_and_bridge_overhead": True,
                            "process_receipt": process_receipt,
                            "status": "complete",
                            "failure": None,
                        }
                        row["row_sha256"] = sha256_json(row)
                        raw.append(row)
                        now = time.monotonic()
                        if now - heartbeat >= 50.0:  # pragma: no cover - slow host only.
                            print(
                                f"[heartbeat] elapsed_s={now - started_all:.3f} "
                                f"completed={len(raw)}/{total} operation=throughput_benchmark",
                                flush=True,
                            )
                            heartbeat = now
    aggregates: list[JsonDict] = []
    for n in sizes:
        for cardinality in cardinalities:
            for language in ("python", "rust"):
                for budget_name in ("matched_energy_evaluations", "matched_wall_time"):
                    selected = [
                        row
                        for row in raw
                        if row["n"] == n
                        and row["k"] == cardinality
                        and row["language"] == language
                        and row["budget"] == budget_name
                    ]
                    latencies = [row["latency_s"] for row in selected]
                    ess_rates = [
                        row["ess_per_second"]
                        for row in selected
                        if row["ess_per_second"] is not None
                    ]
                    aggregate = {
                        "row_type": "throughput_aggregate",
                        "unit_id": f"n{n}:k{cardinality}:{language}:{budget_name}",
                        "n": n,
                        "k": cardinality,
                        "language": language,
                        "budget": budget_name,
                        "case_count": len(selected),
                        "latency_p50_s": float(np.percentile(latencies, 50)),
                        "latency_p95_s": float(np.percentile(latencies, 95)),
                        "ess_per_second_p50": (
                            float(np.percentile(ess_rates, 50)) if ess_rates else None
                        ),
                        "ess_per_second_p95": (
                            float(np.percentile(ess_rates, 95)) if ess_rates else None
                        ),
                        "includes_setup_and_bridge_overhead": True,
                    }
                    aggregate["row_sha256"] = sha256_json(aggregate)
                    aggregates.append(aggregate)
    return raw, aggregates


def _speedup_rows(throughput_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep every deployment-speed result, including ratios below the 10x target."""

    rows: list[JsonDict] = []
    for n in THROUGHPUT_SIZES:
        for cardinality in THROUGHPUT_CARDINALITIES:
            for budget in ("matched_energy_evaluations", "matched_wall_time"):
                by_language = {
                    row["language"]: row
                    for row in throughput_rows
                    if row["n"] == n and row["k"] == cardinality and row["budget"] == budget
                }
                speedup = (
                    by_language["python"]["latency_p50_s"] / by_language["rust"]["latency_p50_s"]
                )
                row = {
                    "unit_id": f"n{n}:k{cardinality}:{budget}",
                    "n": n,
                    "k": cardinality,
                    "budget": budget,
                    "python_over_rust_latency_speedup": speedup,
                    "target": 10.0,
                    "target_met": speedup >= 10.0,
                }
                row["row_sha256"] = sha256_json(row)
                rows.append(row)
    return rows


def _base_artifact(
    *, root: Path, run_date: str, checks: list[JsonDict], hashes: dict[str, str]
) -> JsonDict:
    """Create all terminal fields before choosing blocked or measured evidence."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": checks,
        "run_date": run_date,
        "inference_substrate": "not_started",
        "execution_venue": "host",
        "host_identity": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": [],
        "random_seed": {
            "replay_seed": REPLAY_SEED,
            "distribution_seeds": list(DISTRIBUTION_SEEDS),
            "throughput_seeds": list(slices.BENCHMARK_SEEDS),
            "python_rng": "CPython random.Random",
            "rust_rng": "independent 64-bit LCG in carnot-samplers",
            "distribution_seed_derivation": "first_u64_be(sha256('exp7189:' + language + ':' + seed))",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "inference_substrate_class": "blocked_no_run",
        "rust_slice_parity_score": 0,
        "replay_tape_manifest": [],
        "cross_language_rows": [],
        "distribution_rows": [],
        "throughput_raw_rows": [],
        "throughput_rows": [],
        "e2e_receipts": [],
        "speedup_rows": [],
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "spec_refs": ["REQ-SAMPLER-7189", "SCENARIO-SAMPLER-7189-*"],
        "compiled_rust_execution": False,
        "compiled_bridge_sha256": None,
        "distribution_contract": {
            "n": 8,
            "k": 2,
            "beta": 2.0,
            "graph_seed": 718701,
            "burn_in": DISTRIBUTION_BURN_IN,
            "retained": DISTRIBUTION_RETAINED,
            "seeds": list(DISTRIBUTION_SEEDS),
            "total_variation_limit": TV_LIMIT,
            "energy_mean_error_limit": ENERGY_MEAN_LIMIT,
        },
        "throughput_contract": {
            "sizes": list(THROUGHPUT_SIZES),
            "cardinalities": list(THROUGHPUT_CARDINALITIES),
            "seeds": list(slices.BENCHMARK_SEEDS),
            "energy_budget": ENERGY_BUDGET,
            "wall_time_budget_s": WALL_TIME_BUDGET_S,
            "timing_scope": "setup, serialization, process bridge, execution, and output parsing",
        },
        "nfr_01_10x_met": False,
        "performance_verdict_class": "null",
        "hardware_execution_claimed": False,
        "pyo3_execution_claimed": False,
        "multiscale_execution_claimed": False,
        "quantized_delayed_acceptance_claimed": False,
        "mixing_theorem_claimed": False,
    }


def _run_replay_cases(bridge: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay one ordinary and both singleton slices through compiled Rust."""

    rows: list[JsonDict] = []
    manifest: list[JsonDict] = []
    instance = slices.make_frustrated_instance(8, 718701)
    for offset, cardinality in enumerate((2, 0, 8)):
        case_id = f"n8:k{cardinality}:seed{REPLAY_SEED + offset}"
        initial = slices.enumerate_slice(8, cardinality)[3 if cardinality == 2 else 0]
        tape = make_replay_tape(
            8,
            cardinality,
            seed=REPLAY_SEED + offset,
            steps=REPLAY_STEPS,
        )
        request = replay_request(instance, cardinality, 2.0, initial, tape)
        python_result = python_replay(instance, cardinality, 2.0, initial, tape)
        rust_result, receipt = rust_request(bridge, request)
        case_rows = compare_replays(case_id, tape, python_result, rust_result)
        rows.extend(case_rows)
        manifest.append(
            {
                "case_id": case_id,
                "initial_state": list(initial),
                "tape": tape,
                "tape_sha256": sha256_json(tape),
                "request_sha256": receipt["request_sha256"],
                "python_result_sha256": sha256_json(python_result),
                "rust_result_sha256": receipt["stdout_sha256"],
                "row_hashes": [row["row_sha256"] for row in case_rows],
            }
        )
    return rows, manifest


def _honest_verdict(
    *,
    parity_ready: bool,
    nfr_met: bool,
    replay_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    e2e_receipts: Sequence[Mapping[str, Any]],
) -> str:
    """Describe parity and speed independently so a null cannot become a success."""

    if not parity_ready:
        failed_replay = sum(row["passed"] is not True for row in replay_rows)
        failed_distribution = sum(row["passed"] is not True for row in distribution_rows)
        failed_e2e = sum(receipt["passed"] is not True for receipt in e2e_receipts)
        return (
            "null: compiled execution completed, but the frozen parity gate did not pass; "
            f"failed replay rows={failed_replay}, failed distribution rows={failed_distribution}, "
            f"and failed E2E receipts={failed_e2e}. Every negative row is retained."
        )
    if nfr_met:
        return (
            "positive: compiled Rust matched every explicit Python transition, all independent "
            "streams matched the Exp7187 exact law, the bounded serialized E2E passed, and "
            "every measured deployment latency speedup met 10x."
        )
    return (
        "null: compiled Rust matched every explicit Python transition, all independent streams "
        "matched the Exp7187 exact law, and the bounded serialized E2E passed, but at least one "
        "measured deployment latency speedup was below NFR-01's 10x target. The parity path is "
        "ready; the speed result is retained as a performance null."
    )


def build_artifact(
    *,
    root: Path,
    run_date: str = RUN_DATE,
    bridge_path: Path | None = None,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build complete evidence, or stop before measurement on an external failure."""

    started = time.monotonic()
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None:
        measured_checks, measured_hashes = collect_preconditions(root)
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
    else:
        checks, hashes = preconditions, source_hashes
    artifact = _base_artifact(root=root, run_date=run_date, checks=checks, hashes=hashes)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    _progress(0, "end", "precondition checks")
    if failed is not None:
        artifact.update(
            {
                "status": "blocked_external_precondition",
                "inference_substrate": "no qualifying computation executed",
                "duration_s": time.monotonic() - started,
                "gate_check_summary": {
                    "passed": False,
                    "failed_check": failed.get("check"),
                    "upstream": failed.get("upstream"),
                    "field": failed.get("field"),
                    "expected_value": failed.get("expected_value"),
                    "observed_value": failed.get("observed_value"),
                },
                "verdict_class": "blocked",
                "honest_verdict": (
                    f"blocked_external_precondition: {failed.get('check')} failed before measurement"
                ),
                "inference_substrate_class": "blocked_no_run",
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": "repository_preflight",
        "field": "all_required_preconditions",
        "expected_value": "all pass",
        "observed_value": "all pass",
    }
    _progress(1, "start", "compiled Rust bridge")
    bridge = bridge_path or build_rust_bridge(root)
    if not bridge.is_file() or not os.access(bridge, os.X_OK):
        raise ValueError(f"compiled Rust bridge unavailable: {bridge}")
    artifact["compiled_rust_execution"] = True
    artifact["compiled_bridge_sha256"] = sha256_file(bridge)
    _progress(1, "end", f"compiled Rust bridge path={bridge}")

    _progress(2, "start", "shared-tape replay parity")
    replay_rows, replay_manifest = _run_replay_cases(bridge)
    artifact["cross_language_rows"] = replay_rows
    artifact["replay_tape_manifest"] = replay_manifest
    _progress(2, "end", f"shared-tape replay parity rows={len(replay_rows)}")

    _progress(3, "start", "independent-seed distribution checks")
    distribution_instance = slices.make_frustrated_instance(8, 718701)
    distribution_law = slices.independent_exact_law(distribution_instance, 2, 2.0)
    distribution_rows = run_distribution_checks(
        bridge,
        instance=distribution_instance,
        law=distribution_law,
        seeds=DISTRIBUTION_SEEDS,
        burn_in=DISTRIBUTION_BURN_IN,
        retained=DISTRIBUTION_RETAINED,
        tv_limit=TV_LIMIT,
        energy_mean_limit=ENERGY_MEAN_LIMIT,
    )
    artifact["distribution_rows"] = distribution_rows
    _progress(3, "end", f"independent-seed distribution checks rows={len(distribution_rows)}")

    _progress(4, "start", "serialized Ising E2E")
    with tempfile.TemporaryDirectory(prefix="carnot-exp7189-e2e-") as directory:
        artifact["e2e_receipts"] = [run_e2e_receipt(bridge, Path(directory))]
    _progress(4, "end", "serialized Ising E2E receipts=1")

    _progress(5, "start", "matched-budget throughput benchmark")
    throughput_raw, throughput_rows = run_throughput_benchmarks(bridge)
    artifact["throughput_raw_rows"] = throughput_raw
    artifact["throughput_rows"] = throughput_rows
    speedups = _speedup_rows(throughput_rows)
    artifact["speedup_rows"] = speedups
    _progress(5, "end", f"matched-budget throughput benchmark rows={len(throughput_raw)}")

    _progress(6, "start", "terminal evidence assembly")
    artifact["rows"] = replay_rows + distribution_rows + throughput_raw + throughput_rows
    parity_ready = (
        artifact["compiled_rust_execution"] is True
        and all(row["passed"] is True for row in replay_rows)
        and all(row["passed"] is True for row in distribution_rows)
        and all(receipt["passed"] is True for receipt in artifact["e2e_receipts"])
    )
    nfr_met = all(row["target_met"] is True for row in speedups)
    honest_verdict = _honest_verdict(
        parity_ready=parity_ready,
        nfr_met=nfr_met,
        replay_rows=replay_rows,
        distribution_rows=distribution_rows,
        e2e_receipts=artifact["e2e_receipts"],
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "cpu_exact_solver_or_simulator: compiled release-mode Rust pair-swap process "
                "bridge, independent Python implementation, Exp7187 exact finite-slice law, "
                "explicit replay tapes, and measured host subprocess timings"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "rust_slice_parity_score": int(parity_ready),
            "nfr_01_10x_met": nfr_met,
            "performance_verdict_class": "positive" if nfr_met else "null",
            "verdict_class": "positive" if parity_ready and nfr_met else "null",
            "honest_verdict": honest_verdict,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(6, "end", "terminal evidence assembly")
    return artifact


def validate_artifact(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute completeness, parity readiness, claims, verdict, and hashes."""

    if not REQUIRED_SCHEMA_FIELDS.issubset(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_authority_invalid")
    if any(
        payload.get(field) is not False
        for field in (
            "hardware_execution_claimed",
            "pyo3_execution_claimed",
            "multiscale_execution_claimed",
            "quantized_delayed_acceptance_claimed",
            "mixing_theorem_claimed",
        )
    ):
        errors.append("claim_boundary_invalid")
    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("rust_slice_parity_score") != 0
            or any(
                payload.get(field)
                for field in (
                    "rows",
                    "cross_language_rows",
                    "distribution_rows",
                    "throughput_raw_rows",
                    "throughput_rows",
                    "e2e_receipts",
                )
            )
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not gate.get("failed_check")
            or not gate.get("upstream")
            or not gate.get("field")
        ):
            errors.append("blocked_state_invalid")
        return list(dict.fromkeys(errors))

    cross_rows = payload.get("cross_language_rows", [])
    distribution_rows = payload.get("distribution_rows", [])
    raw_rows = payload.get("throughput_raw_rows", [])
    throughput_rows = payload.get("throughput_rows", [])
    receipts = payload.get("e2e_receipts", [])
    manifest = payload.get("replay_tape_manifest", [])
    expected_cross_units = {
        f"n8:k{cardinality}:seed{REPLAY_SEED + offset}:step{step}"
        for offset, cardinality in enumerate((2, 0, 8))
        for step in range(REPLAY_STEPS)
    }
    if (
        len(cross_rows) != 3 * REPLAY_STEPS
        or {row.get("unit_id") for row in cross_rows} != expected_cross_units
    ):
        errors.append("cross_language_rows_incomplete")
    cross_rows_valid = all(
        row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        and math.isclose(
            row.get("delta_energy_error", math.inf),
            abs(row.get("python_delta_energy", math.inf) - row.get("rust_delta_energy", -math.inf)),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and row.get("passed")
        == (
            row.get("python_input_state") == row.get("rust_input_state")
            and row.get("python_accepted") == row.get("rust_accepted")
            and row.get("python_cardinality") == row.get("rust_cardinality")
            and row.get("python_proposed_state") == row.get("rust_proposed_state")
            and row.get("python_state") == row.get("rust_state")
            and row.get("delta_energy_error", math.inf) <= TOLERANCE
        )
        for row in cross_rows
    )
    if not cross_rows_valid:
        errors.append("cross_language_rows_invalid")
    if (
        len(manifest) != 3
        or any(len(row.get("tape", [])) != REPLAY_STEPS for row in manifest)
        or any(row.get("tape_sha256") != sha256_json(row.get("tape")) for row in manifest)
        or any(
            not all(
                isinstance(row.get(field), str) and HASH_PATTERN.fullmatch(row[field])
                for field in (
                    "tape_sha256",
                    "request_sha256",
                    "python_result_sha256",
                    "rust_result_sha256",
                )
            )
            for row in manifest
        )
        or sorted(hash_value for row in manifest for hash_value in row.get("row_hashes", []))
        != sorted(row.get("row_sha256") for row in cross_rows)
    ):
        errors.append("replay_tape_manifest_invalid")
    expected_distribution = {
        f"{language}:seed{seed}" for language in ("python", "rust") for seed in DISTRIBUTION_SEEDS
    }
    distribution_roster_complete = (
        len(distribution_rows) != 20
        or {row.get("unit_id") for row in distribution_rows} != expected_distribution
    )
    if distribution_roster_complete:
        errors.append("distribution_rows_incomplete")
    distribution_rows_valid = True
    authority = slices.independent_exact_law(slices.make_frustrated_instance(8, 718701), 2, 2.0)
    for row in distribution_rows:
        histogram = row.get("histogram", [])
        retained = row.get("retained", 0)
        histogram_valid = (
            len(histogram) == 28
            and retained == DISTRIBUTION_RETAINED
            and sum(item.get("count", -1) for item in histogram) == retained
            and all(
                item.get("count", -1) >= 0
                and len(item.get("state", [])) == 8
                and item.get("state", []).count(1) == 2
                and item.get("state") == list(authority.states[index])
                and math.isclose(
                    item.get("target_probability", math.inf),
                    authority.probabilities[index],
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                and math.isclose(
                    item.get("energy", math.inf),
                    authority.energies[index],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    item.get("probability", math.inf),
                    item.get("count", -1) / retained,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                for index, item in enumerate(histogram)
            )
        )
        if histogram_valid:
            tv = 0.5 * sum(
                abs(item["probability"] - item["target_probability"]) for item in histogram
            )
            energy_mean = sum(item["probability"] * item["energy"] for item in histogram)
            exact_energy_mean = sum(
                item["target_probability"] * item["energy"] for item in histogram
            )
        else:
            tv = math.inf
            energy_mean = math.inf
            exact_energy_mean = 0.0
        observed_pass = (
            histogram_valid
            and row.get("cardinality_valid") is True
            and tv <= row.get("total_variation_limit", -math.inf)
            and abs(energy_mean - exact_energy_mean)
            <= row.get("energy_mean_error_limit", -math.inf)
        )
        distribution_rows_valid &= (
            row.get("effective_seed") == _derived_seed(row.get("language", ""), row.get("seed", -1))
            and math.isclose(row.get("total_variation", math.inf), tv, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(
                row.get("energy_mean", math.inf), energy_mean, rel_tol=0.0, abs_tol=1.0e-12
            )
            and math.isclose(
                row.get("exact_energy_mean", math.inf),
                exact_energy_mean,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and row.get("passed") == observed_pass
            and row.get("row_sha256")
            == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        )
    if not distribution_rows_valid:
        errors.append("distribution_rows_invalid")
    expected_raw_units = {
        f"n{n}:k{k}:seed{seed}:{language}:{budget}"
        for n in THROUGHPUT_SIZES
        for k in THROUGHPUT_CARDINALITIES
        for seed in slices.BENCHMARK_SEEDS
        for language in ("python", "rust")
        for budget in ("matched_energy_evaluations", "matched_wall_time")
    }
    if (
        len(raw_rows) != 160
        or {row.get("unit_id") for row in raw_rows} != expected_raw_units
        or any(row.get("status") != "complete" for row in raw_rows)
    ):
        errors.append("throughput_raw_rows_incomplete")
    raw_rows_valid = all(
        row.get("latency_s", 0.0) > 0.0
        and row.get("work", 0) > 0
        and row.get("proposal_attempts", 0) > 0
        and 0.0 <= row.get("acceptance", -1.0) <= 1.0
        and row.get("includes_setup_and_bridge_overhead") is True
        and row.get("failure") is None
        and row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        and (
            (
                row.get("budget") == "matched_energy_evaluations"
                and row.get("budget_value") == ENERGY_BUDGET
                and row.get("work") == ENERGY_BUDGET
            )
            or (
                row.get("budget") == "matched_wall_time"
                and row.get("budget_value") == WALL_TIME_BUDGET_S
            )
        )
        and (
            (row.get("language") == "python" and row.get("process_receipt") is None)
            or (
                row.get("language") == "rust"
                and isinstance(row.get("process_receipt"), Mapping)
                and row["process_receipt"].get("returncode") == 0
            )
        )
        for row in raw_rows
    )
    if not raw_rows_valid:
        errors.append("throughput_raw_rows_invalid")
    expected_throughput_units = {
        f"n{n}:k{k}:{language}:{budget}"
        for n in THROUGHPUT_SIZES
        for k in THROUGHPUT_CARDINALITIES
        for language in ("python", "rust")
        for budget in ("matched_energy_evaluations", "matched_wall_time")
    }
    throughput_roster_valid = (
        len(throughput_rows) == 16
        and {row.get("unit_id") for row in throughput_rows} == expected_throughput_units
        and all(row.get("case_count") == 10 for row in throughput_rows)
        and all(
            row.get("latency_p95_s", 0.0) >= row.get("latency_p50_s", 0.0)
            for row in throughput_rows
        )
    )
    if not throughput_roster_valid:
        errors.append("throughput_rows_incomplete")
    throughput_rows_valid = throughput_roster_valid
    for row in throughput_rows:
        selected = [
            raw
            for raw in raw_rows
            if raw.get("n") == row.get("n")
            and raw.get("k") == row.get("k")
            and raw.get("language") == row.get("language")
            and raw.get("budget") == row.get("budget")
        ]
        latencies = [raw["latency_s"] for raw in selected]
        rates = [raw["ess_per_second"] for raw in selected if raw.get("ess_per_second") is not None]
        throughput_rows_valid &= (
            len(selected) == 10
            and math.isclose(
                row.get("latency_p50_s", math.inf),
                float(np.percentile(latencies, 50)),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                row.get("latency_p95_s", math.inf),
                float(np.percentile(latencies, 95)),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and row.get("ess_per_second_p50")
            == (float(np.percentile(rates, 50)) if rates else None)
            and row.get("ess_per_second_p95")
            == (float(np.percentile(rates, 95)) if rates else None)
            and row.get("row_sha256")
            == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        )
    if throughput_roster_valid and not throughput_rows_valid:
        errors.append("throughput_rows_invalid")
    expected_rows = cross_rows + distribution_rows + raw_rows + throughput_rows
    if payload.get("rows") != expected_rows:
        errors.append("rows_incomplete")
    e2e_roster_complete = len(receipts) == 1
    if not e2e_roster_complete:
        errors.append("e2e_receipt_invalid")
    e2e_pass = e2e_roster_complete and receipts[0].get("passed") is True
    if e2e_roster_complete:
        receipt = receipts[0]
        parameter_bytes = (canonical_json(receipt.get("parameter_payload")) + "\n").encode("utf-8")
        recomputed_e2e = (
            receipt.get("rust_returncode") == 0
            and receipt.get("energy_error", math.inf) <= TOLERANCE
            and receipt.get("python_cardinality") == receipt.get("rust_cardinality") == 2
            and receipt.get("parameter_sha256") == sha256_bytes(parameter_bytes)
            and isinstance(receipt.get("rust_output_sha256"), str)
            and bool(HASH_PATTERN.fullmatch(receipt["rust_output_sha256"]))
        )
        if receipt.get("passed") != recomputed_e2e:
            errors.append("e2e_receipt_invalid")
        e2e_pass = recomputed_e2e
    if payload.get("compiled_rust_execution") is not True:
        errors.append("compiled_execution_missing")
    bridge_hash = payload.get("compiled_bridge_sha256")
    if not isinstance(bridge_hash, str) or not HASH_PATTERN.fullmatch(bridge_hash):
        errors.append("compiled_bridge_hash_invalid")
    if root is not None:
        recorded_hashes = payload.get("source_artifact_hashes", {})
        expected_sources = {str(path) for path in REQUIRED_SOURCE_PATHS}
        if (
            not isinstance(recorded_hashes, Mapping)
            or set(recorded_hashes) != expected_sources
            or any(
                not (root / path).is_file() or recorded_hashes[path] != sha256_file(root / path)
                for path in expected_sources
            )
        ):
            errors.append("source_artifact_hashes_invalid")
    parity_ready = (
        "cross_language_rows_incomplete" not in errors
        and cross_rows_valid
        and all(row.get("passed") is True for row in cross_rows)
        and "distribution_rows_incomplete" not in errors
        and distribution_rows_valid
        and all(row.get("passed") is True for row in distribution_rows)
        and e2e_pass
        and "compiled_execution_missing" not in errors
        and "compiled_bridge_hash_invalid" not in errors
    )
    if payload.get("rust_slice_parity_score") != int(parity_ready):
        errors.append("readiness_invalid")
    speedups = payload.get("speedup_rows", [])
    expected_speedups = _speedup_rows(throughput_rows) if throughput_roster_valid else []
    speedups_valid = speedups == expected_speedups
    if not speedups_valid:
        errors.append("speedup_rows_invalid")
    nfr_met = speedups_valid and all(row.get("target_met") is True for row in speedups)
    if payload.get("nfr_01_10x_met") is not nfr_met:
        errors.append("nfr_outcome_invalid")
    expected_performance = "positive" if nfr_met else "null"
    if payload.get("performance_verdict_class") != expected_performance:
        errors.append("performance_verdict_invalid")
    expected_verdict = "positive" if parity_ready and nfr_met else "null"
    if payload.get("verdict_class") != expected_verdict:
        errors.append("terminal_verdict_invalid")
    if payload.get("status") != "complete":
        errors.append("terminal_status_invalid")
    if payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if payload.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("gate_summary_invalid")
    return list(dict.fromkeys(errors))


def atomic_write(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish one complete JSON file through same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # pragma: no cover - interrupted replacement only.
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(*, root: Path, output: Path, run_date: str) -> JsonDict:
    """Compile, measure, independently validate, then atomically publish Exp7189."""

    artifact = build_artifact(root=root, run_date=run_date)
    _progress(7, "start", "artifact validation")
    errors = validate_artifact(artifact, root=root)
    _progress(7, "end", f"artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7189 artifact: {errors}")
    _progress(8, "start", "final atomic write")
    receipt = atomic_write(output, artifact)
    _progress(8, "end", f"final atomic write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed execution date and the read-only artifact check path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or validate existing bytes without changing them."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        _progress(7, "start", f"validation path={args.validate}")
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(7, "end", "validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(7, "end", f"validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    try:
        output = args.output if args.output.is_absolute() else root / args.output
        run_experiment(root=root, output=output, run_date=args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the shipped entrypoint.
    raise SystemExit(main())
