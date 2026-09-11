"""Compare fixed-cardinality sampler boundaries at matched work and wall time.

The study reuses the shipped Python, process Rust, and persistent PyO3 paths.
It measures boundary value separately from sample quality and the 10x NFR.

Spec: REQ-RUSTPY-7202 and SCENARIO-RUSTPY-7202-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7189_v633_rust_slice_parity as exp7189
from carnot import experiment_7201_v634_slice_pyo3 as exp7201


JsonDict = dict[str, Any]

RUN_DATE = "20260911"
TASK_ID = "exp7202-slice-cost-quality"
MILESTONE = "2026.09.634"
RESULT_PATH = Path("results/experiment_7202_v634_slice_cost_quality.json")
CHECKPOINT_DIR = Path("results/checkpoints")
SPEC_PATH = Path("openspec/capabilities/rust-python-boundary/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
UPSTREAM_RESULT_PATH = Path("results/experiment_7201_v634_slice_pyo3.json")

SIZES = (32, 64, 128)
CARDINALITIES = (2, 4)
BATCH_SIZES = (1, 16, 64)
SEEDS = tuple(range(720200, 720210))
ARMS = ("python_control", "subprocess_rust", "persistent_pyo3")
PROTOCOLS = ("equal_work", "equal_wall")
PROPOSALS_PER_CHAIN = 160
WALL_BUDGET_S = 0.05
BETA = 2.0
QUALITY_BURN_IN = 1024
QUALITY_RETAINED = 8192
QUALITY_MIN_RETAINED = 4096
QUALITY_MIN_ESS = 100.0
QUALITY_LAG_WINDOW = 256
REPORTED_LAGS = 16
ENERGY_STANDARDIZED_TOLERANCE = 0.02
ESS_RATE_RATIO_CI_MIN = 0.90
NFR_01_SPEEDUP_TARGET = 10.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 720299
PRIMARY_CELL = {"n": 64, "k": 4, "batch_size": 1, "protocol": "equal_work"}
MODEL_SPECS: list[JsonDict] = []

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": [
        {
            "upstream": "exp7201-slice-pyo3",
            "artifact_field": "pyo3_slice_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    "prior_failures": [
        {
            "experiment_id": "exp7189-rust-slice-parity",
            "verdict": (
                "null: compiled Rust matched every explicit Python transition, all independent "
                "streams matched the Exp7187 exact law, and the bounded serialized E2E passed, "
                "but at least one measured deployment latency speedup was below NFR-01's 10x "
                "target. The parity path is ready; the speed result is retained as a performance "
                "null."
            ),
            "addressed_by": (
                "Benchmark an in-process boundary on matched old and new workload cells; "
                "preserve the prior null and retain the original 10x PRD threshold."
            ),
            "retire_if_same_verdict": True,
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
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    UPSTREAM_RESULT_PATH,
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7189_v633_rust_slice_parity.py"),
    Path("python/carnot/experiment_7201_v634_slice_pyo3.py"),
    Path("python/carnot/experiment_7202_v634_slice_cost_quality.py"),
    Path("crates/carnot-samplers/src/fixed_cardinality.rs"),
    Path("crates/carnot-samplers/src/bin/fixed-cardinality-bridge.rs"),
    Path("crates/carnot-python/src/fixed_cardinality.rs"),
    SPEC_PATH,
    Path("openspec/capabilities/samplers/spec.md"),
    Path("scripts/experiments/experiment_7202_v634_slice_cost_quality.py"),
    Path("tests/python/test_experiment_7202_v634_slice_cost_quality.py"),
)

FIELD_PRINCIPLES = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260911, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": (
        "Every blocked verdict names the failed check, upstream, field, expected and observed "
        "value."
    ),
    "verifier_is_oracle": (
        "True when verification uses the same correctness authority; separate implementations "
        "alone do not remove circularity."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. Only "
        "incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings, including nulls; blocked_* for "
        "external blocks. Never promote infrastructure readiness as scientific benefit."
    ),
    "slice_comparison_complete_score": "Completion does not imply acceleration.",
    "boundary_value_score": "The local claim concerns the measured process boundary.",
    "throughput_rows": "Each size, cardinality, batch, seed and arm retains raw times.",
    "distribution_rows": "Correct target law precedes any speed claim.",
    "nfr_01_10x_met": "The PRD threshold stays 10x; a lesser improvement does not satisfy it.",
    "acceptance_gate_boundary": "The primary deployment cell is fixed before timing.",
    "sample_quality_sufficient": (
        "False unless the independent longer-chain quality panel meets every declared sample "
        "and fidelity threshold."
    ),
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES) | {
    "task_id",
    "milestone",
    "host_identity",
    "root",
    "compiled_binding_receipt",
    "cold_initialization_rows",
    "quality_rows",
    "quality_comparison_rows",
    "control_rows",
    "primary_gate",
    "upstream_performance_null",
    "spec_refs",
}

canonical_json = exp7189.canonical_json
sha256_json = exp7189.sha256_json
sha256_file = exp7189.sha256_file


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush phase boundaries so a stalled native call is visible."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the shared evidence fields before binding the row bytes."""

    row.setdefault("arm", "not_applicable")
    row.setdefault("seed", None)
    row.setdefault("metric", row.get("row_type", "measurement"))
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = sha256_json(row)
    return row


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash all stable evidence except the field that stores this hash."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _task_contract(root: Path) -> JsonDict | None:
    """Read exact roadmap fields so a similar task cannot satisfy this gate."""

    path = root / ROADMAP_PATH
    if not path.is_file():
        return None
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
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
        "prior_failures": task.get("prior_failures"),
    }


def upstream_quarantine_observation(
    upstream: Mapping[str, Any], *, manifest_match: bool
) -> JsonDict:
    """Join artifact and manifest flags before any structured gate is read."""

    fields = (
        "flagged_adversarial",
        "quarantined",
        "quarantine",
        "quarantine_flags",
        "disqualified",
        "invalidated",
    )
    observed = {field: upstream.get(field) for field in fields}
    active = [field for field, value in observed.items() if value not in (None, False, "", [], {})]
    if manifest_match:
        active.append("exclusion_manifest")
    return {
        **observed,
        "exclusion_manifest_match": manifest_match,
        "active_flags": active,
        "quarantined": bool(active),
    }


def gated_upstream_value(
    upstream: Mapping[str, Any], quarantine: Mapping[str, Any], field: str
) -> Any:
    """Refuse to consume a structured value after any quarantine signal."""

    if quarantine.get("quarantined") is True:
        return "not_consumed_due_to_quarantine"
    return upstream.get(field)


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Print before every check and retain each actual prerequisite state."""

    result = result_path or root / RESULT_PATH
    checkpoints = checkpoint_dir or root / CHECKPOINT_DIR
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    def announce(name: str) -> None:
        print(f"[phase 0 check start] {name}", flush=True)

    announce("required source bytes and hashes")
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    hashes = {
        str(path): sha256_file(root / path)
        for path in REQUIRED_SOURCE_PATHS
        if sizes[str(path)] is not None and sizes[str(path)] > 0
    }

    announce("driving capability specification")
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""

    announce("active roadmap task fields")
    task_contract = _task_contract(root)

    announce("upstream artifact bytes")
    upstream_file = root / UPSTREAM_RESULT_PATH
    try:
        decoded = json.loads(upstream_file.read_text(encoding="utf-8"))
        upstream = decoded if isinstance(decoded, Mapping) else {}
    except (OSError, json.JSONDecodeError):
        upstream = {}

    announce("upstream quarantine flags")
    exclusion_file = root / EXCLUSION_PATH
    try:
        exclusion_text = exclusion_file.read_text(encoding="utf-8")
    except OSError:
        exclusion_text = ""
    manifest_match = any(
        token in exclusion_text
        for token in ("exp7201-slice-pyo3", "experiment_7201_v634_slice_pyo3")
    )
    quarantine = upstream_quarantine_observation(upstream, manifest_match=manifest_match)

    announce("exact upstream pyo3 gate")
    gate_value = gated_upstream_value(upstream, quarantine, "pyo3_slice_ready_score")

    announce("known failed upstream NFR value")
    prior_null = upstream.get("upstream_performance_null", {})
    prior_null = prior_null if isinstance(prior_null, Mapping) else {}
    prior_failed_value = gated_upstream_value(prior_null, quarantine, "observed_value")

    announce("host tools")
    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    tools = {
        "python": bool(sys.executable),
        "numpy": bool(np.__version__),
        "pyyaml": bool(yaml.__version__),
        "cargo": cargo is not None,
        "rustc": rustc is not None,
        "python_path": sys.executable,
        "cargo_path": cargo,
        "rustc_path": rustc,
    }

    announce("output directories")
    output_state = {
        "result_parent": result.parent.is_dir() and os.access(result.parent, os.W_OK),
        "checkpoint_dir": checkpoints.is_dir() and os.access(checkpoints, os.W_OK),
        "terminal_is_not_checkpoint": result.parent.resolve() != checkpoints.resolve(),
    }

    checks = [
        {
            "check": "required_source_bytes",
            "upstream": "repository",
            "field": "byte_count",
            "expected_value": "every required source is nonempty",
            "observed_value": sizes,
            "passed": all(size is not None and size > 0 for size in sizes.values()),
        },
        {
            "check": "driving_capability_spec",
            "upstream": str(SPEC_PATH),
            "field": "REQ-RUSTPY-7202",
            "expected_value": {"exists": True, "req_present": True},
            "observed_value": {
                "exists": spec_file.is_file(),
                "req_present": "REQ-RUSTPY-7202" in spec_text,
            },
            "passed": spec_file.is_file() and "REQ-RUSTPY-7202" in spec_text,
        },
        {
            "check": "same_milestone_gate_fields",
            "upstream": str(ROADMAP_PATH),
            "field": "task_contract",
            "expected_value": EXPECTED_TASK_CONTRACT,
            "observed_value": task_contract,
            "passed": task_contract == EXPECTED_TASK_CONTRACT,
        },
        {
            "check": "upstream_artifact_bytes",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "byte_count",
            "expected_value": "nonempty valid JSON object",
            "observed_value": upstream_file.stat().st_size if upstream_file.is_file() else None,
            "passed": bool(upstream),
        },
        {
            "check": "upstream_quarantine_flags",
            "upstream": f"{UPSTREAM_RESULT_PATH} and {EXCLUSION_PATH}",
            "field": "quarantined",
            "expected_value": False,
            "observed_value": quarantine,
            "passed": quarantine["quarantined"] is False,
        },
        {
            "check": "upstream_pyo3_gate",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "pyo3_slice_ready_score",
            "expected_value": 1,
            "observed_value": gate_value,
            "passed": quarantine["quarantined"] is False and gate_value == 1,
        },
        {
            "check": "upstream_known_failed_value",
            "upstream": str(UPSTREAM_RESULT_PATH),
            "field": "upstream_performance_null.observed_value",
            "expected_value": False,
            "observed_value": prior_failed_value,
            "passed": (
                quarantine["quarantined"] is False
                and prior_failed_value is False
                and prior_null.get("promoted") is False
            ),
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
            "observed_value": tools,
            "passed": all(tools[name] for name in ("python", "numpy", "pyyaml", "cargo", "rustc")),
        },
        {
            "check": "output_directories",
            "upstream": "filesystem",
            "field": "result_and_checkpoint_parent",
            "expected_value": {
                "result_parent": True,
                "checkpoint_dir": True,
                "terminal_is_not_checkpoint": True,
            },
            "observed_value": output_state,
            "passed": all(output_state.values()),
        },
        {
            "check": "source_artifact_hashes",
            "upstream": "required_source_bytes",
            "field": "sha256",
            "expected_value": len(REQUIRED_SOURCE_PATHS),
            "observed_value": len(hashes),
            "passed": len(hashes) == len(REQUIRED_SOURCE_PATHS),
        },
    ]
    return checks, hashes


def ess_diagnostics(
    values: Sequence[float],
    *,
    latency_s: float,
    minimum_draws: int = QUALITY_MIN_RETAINED,
    lag_window: int = QUALITY_LAG_WINDOW,
) -> JsonDict:
    """Use the declared initial-positive-sequence ESS and expose insufficiency."""

    count = len(values)
    if count < 2:
        return {
            "draw_count": count,
            "lag_window": None,
            "lag_correlations": None,
            "ess": None,
            "ess_per_second": None,
            "evidence_sufficient": False,
            "estimator": "initial_positive_autocorrelation_sequence",
        }
    used_lag = min(lag_window, count - 1)
    correlations = slices.autocorrelation(values, used_lag)
    ess = slices.effective_sample_size(values, used_lag) if correlations is not None else None
    return {
        "draw_count": count,
        "lag_window": used_lag,
        "lag_correlations": correlations[: REPORTED_LAGS + 1] if correlations else None,
        "ess": ess,
        "ess_per_second": ess / latency_s if ess is not None and latency_s > 0.0 else None,
        "evidence_sufficient": count >= minimum_draws and ess is not None,
        "estimator": "initial_positive_autocorrelation_sequence",
    }


def standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Scale a mean difference by the prespecified pooled sample deviation."""

    if len(left) < 2 or len(right) < 2:
        raise ValueError("standardized comparison needs at least two draws per arm")
    pooled = math.sqrt((statistics.variance(left) + statistics.variance(right)) / 2.0)
    difference = abs(statistics.mean(left) - statistics.mean(right))
    if pooled == 0.0:
        if difference == 0.0:
            return 0.0
        raise ValueError("standardized comparison requires nonconstant pooled draws")
    return difference / pooled


def paired_bootstrap_ci(
    values: Sequence[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float, float]:
    """Bootstrap independent paired-unit ratios with one frozen random stream."""

    if (
        not values
        or resamples <= 0
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
    ):
        raise ValueError("paired ratios and resample count must be finite and positive")
    rng = random.Random(seed)
    estimates = [
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(resamples)
    ]
    low, middle, high = np.percentile(estimates, [2.5, 50.0, 97.5])
    return float(low), float(middle), float(high)


def _ci_dict(values: Sequence[float], *, seed: int) -> JsonDict | None:
    """Name CI coordinates so validators cannot reverse their meaning."""

    if not values:
        return None
    low, estimate, high = paired_bootstrap_ci(values, seed=seed)
    return {"lower": low, "estimate": estimate, "upper": high, "paired_units": len(values)}


def run_exact_law_checks(*, sizes: Sequence[int] = (8, 12)) -> list[JsonDict]:
    """Enumerate both small slices and test the shared pair-swap transition law."""

    rows: list[JsonDict] = []
    for n in sizes:
        instance = slices.make_frustrated_instance(n, 720200 + n)
        law = slices.independent_exact_law(instance, 2, BETA)
        matrix = slices.transition_matrix(instance, 2, BETA, arm=slices.PAIR_SWAP_ARM)
        diagnostics = slices.transition_diagnostics(law, matrix)
        row = {
            "row_type": "distribution_exact_law",
            "unit_id": f"n{n}:k2:exact_law",
            "arm": "independent_exact_enumerator",
            "n": n,
            "k": 2,
            "seed": 720200 + n,
            "metric": "stationary_law_residual",
            "state_count": len(law.states),
            "probability_normalization_error": abs(sum(law.probabilities) - 1.0),
            "transition_normalization_error": diagnostics["transition_normalization_error_max"],
            "transition_support_min": diagnostics["transition_support_min"],
            "detailed_balance_error": diagnostics["detailed_balance_error_max"],
            "stationary_residual": diagnostics["stationary_law_error"],
            "exact_law_claim": True,
        }
        row["passed"] = (
            row["probability_normalization_error"] <= 1.0e-10
            and row["transition_normalization_error"] <= 1.0e-10
            and row["transition_support_min"] >= -1.0e-12
            and row["detailed_balance_error"] <= 1.0e-10
            and row["stationary_residual"] <= 1.0e-10
        )
        rows.append(_finish_row(row))
    return rows


def run_negative_controls() -> list[JsonDict]:
    """Retain constant-chain and wrong-target failures as speed vetoes."""

    constant = ess_diagnostics([1.0] * QUALITY_RETAINED, latency_s=1.0)
    constant_row = _finish_row(
        {
            "row_type": "negative_control",
            "unit_id": "control:constant_chain_ess",
            "control": "constant_chain_ess",
            "metric": "ess_rejection",
            "observed_ess": constant["ess"],
            "observed_lag_correlations": constant["lag_correlations"],
            "control_detected": constant["ess"] is None,
            "speed_eligible": False,
        }
    )
    instance = slices.make_frustrated_instance(8, 718701)
    law = slices.independent_exact_law(instance, 2, BETA)
    mutated = slices._mutation_matrix(instance, 2, BETA, "energy_sign_reversal")
    diagnostics = slices.transition_diagnostics(law, mutated)
    observed = max(diagnostics["detailed_balance_error_max"], diagnostics["stationary_law_error"])
    biased_row = _finish_row(
        {
            "row_type": "negative_control",
            "unit_id": "control:biased_transition_target",
            "control": "biased_transition_target",
            "metric": "wrong_target_rejection",
            "stationary_or_balance_error": observed,
            "control_detected": observed > 1.0e-10,
            "speed_eligible": False,
        }
    )
    return [constant_row, biased_row]


def _initial_state(n: int, k: int, *, seed: int, chain: int) -> list[int]:
    """Create one reproducible slice state without materializing all combinations."""

    rng = random.Random((seed << 12) ^ (n << 4) ^ k ^ chain)
    positive = set(rng.sample(range(n), k))
    return [1 if index in positive else -1 for index in range(n)]


def _batch_inputs(
    n: int, k: int, batch_size: int, seed: int, proposals: int
) -> tuple[list[list[int]], list[list[JsonDict]]]:
    """Freeze caller-owned states and tapes before any arm timing starts."""

    states = [_initial_state(n, k, seed=seed, chain=chain) for chain in range(batch_size)]
    if not 0 <= k <= n or proposals < 0:
        raise ValueError("cardinality and proposal count are out of range")
    tapes = []
    for chain in range(batch_size):
        rng = random.Random((seed << 16) + batch_size * 257 + chain)
        tapes.append(
            [
                {
                    "positive_index": rng.randrange(max(1, k)),
                    "negative_index": rng.randrange(max(1, n - k)),
                    "uniform": rng.random(),
                }
                for _ in range(proposals)
            ]
        )
    return states, tapes


class _SilentHeartbeat:
    """Start a liveness thread but print only when a native call is slow."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _SilentHeartbeat:
        self._thread.start()
        return self

    def __exit__(self, *_unused: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:  # pragma: no cover - bounded calls finish before this interval.
        while not self._stop.wait(60.0):
            print(
                f"[heartbeat] elapsed_s={time.monotonic() - self.started:.3f} "
                f"completed=0 operation={self.operation}",
                flush=True,
            )


def _quiet_rust_request(bridge: Path, request: Mapping[str, Any], *, timeout_s: float) -> JsonDict:
    """Run one short JSON bridge call with a bounded external heartbeat."""

    request_bytes = canonical_json(request).encode("utf-8")
    with _SilentHeartbeat(f"compiled Rust bridge operation={request.get('operation')}"):
        completed = subprocess.run(
            [str(bridge)],
            input=request_bytes,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"compiled Rust bridge failed with exit {completed.returncode}: {stderr}"
        )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("compiled Rust bridge returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError("compiled Rust bridge response must be an object")
    return response


def _run_arm_once(
    arm: str,
    *,
    sampler: Any,
    bridge: Path,
    instance: slices.SliceInstance,
    k: int,
    states: Sequence[Sequence[int]],
    tapes: Sequence[Sequence[Mapping[str, Any]]],
) -> list[JsonDict]:
    """Run one complete caller batch while charging its boundary conversion."""

    if arm == "python_control":
        return [
            exp7189.python_replay(instance, k, BETA, state, tape)
            for state, tape in zip(states, tapes, strict=True)
        ]
    if arm == "persistent_pyo3":
        state_array = np.asarray(states, dtype=np.int8)
        positive, negative, uniforms = exp7201._tape_arrays(tapes)
        return list(sampler.replay_batch(state_array, positive, negative, uniforms))
    if arm == "subprocess_rust":
        outcomes = []
        for state, tape in zip(states, tapes, strict=True):
            request = exp7189.replay_request(instance, k, BETA, state, tape)
            outcomes.append(_quiet_rust_request(bridge, request, timeout_s=120.0))
        return outcomes
    raise ValueError(f"unknown arm: {arm}")


def _result_traces(result: Mapping[str, Any]) -> tuple[list[float], list[float], int, int]:
    """Extract accepted-state energy and occupation from a replay response."""

    energies: list[float] = []
    occupations: list[float] = []
    violations = 0
    accepted = 0
    expected_k = int(result["steps"][0]["cardinality"]) if result.get("steps") else 0
    for step in result.get("steps", []):
        state = step["state"]
        energies.append(step["proposed_energy"] if step["accepted"] else step["current_energy"])
        occupations.append(1.0 if state[0] == 1 else 0.0)
        violations += int(state.count(1) != expected_k)
        accepted += int(step["accepted"])
    return energies, occupations, violations, accepted


def _mean_lags(diagnostics: Sequence[Mapping[str, Any]]) -> list[float] | None:
    """Average like-numbered lag values without pooling independent chains."""

    series = [row["lag_correlations"] for row in diagnostics if row["lag_correlations"]]
    if not series:
        return None
    width = min(len(row) for row in series)
    return [statistics.mean(row[index] for row in series) for index in range(width)]


def _throughput_metrics(
    results_by_cycle: Sequence[Sequence[Mapping[str, Any]]], *, latency_s: float
) -> JsonDict:
    """Summarize independent chain traces without treating a batch as one chain."""

    energy_by_chain: defaultdict[int, list[float]] = defaultdict(list)
    occupation_by_chain: defaultdict[int, list[float]] = defaultdict(list)
    violations = 0
    accepted = 0
    attempts = 0
    for cycle in results_by_cycle:
        for chain_index, result in enumerate(cycle):
            energies, occupations, bad, accepted_count = _result_traces(result)
            energy_by_chain[chain_index].extend(energies)
            occupation_by_chain[chain_index].extend(occupations)
            violations += bad
            accepted += accepted_count
            attempts += len(energies)
    energy_diags = [
        ess_diagnostics(values, latency_s=latency_s) for values in energy_by_chain.values()
    ]
    occupation_diags = [
        ess_diagnostics(values, latency_s=latency_s) for values in occupation_by_chain.values()
    ]
    energy_values = [row["ess"] for row in energy_diags]
    occupation_values = [row["ess"] for row in occupation_diags]
    energy_ess = (
        sum(energy_values) if energy_values and all(v is not None for v in energy_values) else None
    )
    occupation_ess = (
        sum(occupation_values)
        if occupation_values and all(value is not None for value in occupation_values)
        else None
    )
    effective = (
        min(energy_ess, occupation_ess)
        if energy_ess is not None and occupation_ess is not None
        else None
    )
    return {
        "proposal_attempts": attempts,
        "retained_draws_per_chain": min((len(row) for row in energy_by_chain.values()), default=0),
        "acceptance_rate": accepted / max(1, attempts),
        "sector_violations": violations,
        "energy_ess": energy_ess,
        "occupation_ess": occupation_ess,
        "effective_samples_per_second": effective / latency_s
        if effective is not None and latency_s > 0.0
        else None,
        "energy_lag_correlations": _mean_lags(energy_diags),
        "occupation_lag_correlations": _mean_lags(occupation_diags),
        "ess_evidence_sufficient": bool(energy_diags)
        and all(row["evidence_sufficient"] for row in energy_diags + occupation_diags),
        "ess_estimator": "initial_positive_autocorrelation_sequence",
    }


def _arm_order(unit_index: int) -> tuple[str, ...]:
    """Rotate the first arm so order effects do not favor one boundary."""

    offset = unit_index % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def run_throughput_benchmarks(
    binding: ModuleType,
    bridge: Path,
    *,
    sizes: Sequence[int] = SIZES,
    cardinalities: Sequence[int] = CARDINALITIES,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    seeds: Sequence[int] = SEEDS,
    proposals: int = PROPOSALS_PER_CHAIN,
    wall_budget_s: float = WALL_BUDGET_S,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Measure all boundaries with equal proposal work and equal wall windows."""

    cold_rows: list[JsonDict] = []
    rows: list[JsonDict] = []
    samplers: dict[tuple[int, int], Any] = {}
    started_all = time.monotonic()
    last_report = started_all

    for n in sizes:
        for k in cardinalities:
            instance = slices.make_frustrated_instance(n, seeds[0])
            states, tapes = _batch_inputs(n, k, 1, seeds[0], max(1, min(proposals, 4)))
            for arm in ARMS:
                started = time.monotonic()
                sampler = (
                    exp7201._new_sampler(binding, instance, k) if arm == "persistent_pyo3" else None
                )
                _run_arm_once(
                    arm,
                    sampler=sampler,
                    bridge=bridge,
                    instance=instance,
                    k=k,
                    states=states,
                    tapes=tapes,
                )
                elapsed = time.monotonic() - started
                if arm == "persistent_pyo3":
                    samplers[(n, k)] = sampler
                cold_rows.append(
                    _finish_row(
                        {
                            "row_type": "cold_initialization",
                            "unit_id": f"n{n}:k{k}:{arm}:cold",
                            "arm": arm,
                            "n": n,
                            "k": k,
                            "seed": seeds[0],
                            "metric": "cold_initialization_latency_s",
                            "latency_s": elapsed,
                            "excluded_from_warm_statistics": True,
                            "process_launch_charged": arm == "subprocess_rust",
                        }
                    )
                )

    unit_index = 0
    total_units = len(sizes) * len(cardinalities) * len(batch_sizes) * len(seeds) * len(PROTOCOLS)
    for n in sizes:
        for k in cardinalities:
            for batch_size in batch_sizes:
                for seed in seeds:
                    instance = slices.make_frustrated_instance(n, seed)
                    states, tapes = _batch_inputs(n, k, batch_size, seed, proposals)
                    for protocol in PROTOCOLS:
                        order = _arm_order(unit_index)
                        arm_results: dict[str, tuple[float, list[list[JsonDict]], JsonDict]] = {}
                        for order_index, arm in enumerate(order):
                            started = time.monotonic()
                            cycles: list[list[JsonDict]] = []
                            while True:
                                result = _run_arm_once(
                                    arm,
                                    sampler=samplers.get((n, k)),
                                    bridge=bridge,
                                    instance=instance,
                                    k=k,
                                    states=states,
                                    tapes=tapes,
                                )
                                cycles.append(result)
                                elapsed = time.monotonic() - started
                                if protocol == "equal_work" or elapsed >= wall_budget_s:
                                    break
                            metrics = _throughput_metrics(cycles, latency_s=elapsed)
                            metrics["order_index"] = order_index
                            arm_results[arm] = (elapsed, cycles, metrics)

                        parity = True
                        if protocol == "equal_work":
                            signatures = {
                                arm: sha256_json(
                                    [
                                        {
                                            "final_state": result["final_state"],
                                            "accepted": [
                                                step["accepted"] for step in result["steps"]
                                            ],
                                            "states": [step["state"] for step in result["steps"]],
                                        }
                                        for result in cycles[0]
                                    ]
                                )
                                for arm, (_, cycles, _) in arm_results.items()
                            }
                            parity = len(set(signatures.values())) == 1

                        for arm in order:
                            latency, cycles, metrics = arm_results[arm]
                            row = {
                                "row_type": "throughput_raw",
                                "unit_id": (
                                    f"n{n}:k{k}:batch{batch_size}:seed{seed}:{arm}:{protocol}"
                                ),
                                "arm": arm,
                                "n": n,
                                "k": k,
                                "batch_size": batch_size,
                                "seed": seed,
                                "metric": "charged_boundary_latency_s",
                                "protocol": protocol,
                                "arm_order": list(order),
                                "order_index": metrics.pop("order_index"),
                                "latency_s": latency,
                                "planned_proposals_per_chain": proposals,
                                "wall_budget_s": wall_budget_s
                                if protocol == "equal_wall"
                                else None,
                                "deadline_overshoot_s": (
                                    max(0.0, latency - wall_budget_s)
                                    if protocol == "equal_wall"
                                    else None
                                ),
                                "completed_batch_calls": len(cycles),
                                "parity_passed": parity if protocol == "equal_work" else None,
                                "exact_law_claim": False,
                                "data_transfer_and_synchronization_charged": True,
                                "cold_initialization_excluded": True,
                                **metrics,
                            }
                            rows.append(_finish_row(row))

                        unit_index += 1
                        now = time.monotonic()
                        if now - last_report >= 60.0:
                            print(
                                f"[heartbeat] elapsed_s={now - started_all:.3f} "
                                f"completed={unit_index}/{total_units} operation=throughput_benchmark",
                                flush=True,
                            )
                            last_report = now
    return cold_rows, rows


def _quality_chain(
    arm: str,
    *,
    sampler: Any,
    bridge: Path,
    instance: slices.SliceInstance,
    k: int,
    initial: Sequence[int],
    seed: int,
    burn_in: int,
    retained: int,
) -> JsonDict:
    """Run one independent long chain through the selected shipped arm."""

    rust_seed = exp7189._derived_seed("rust", seed)
    if arm == "python_control":
        return exp7189._python_seeded_chain(
            instance,
            cardinality=k,
            beta=BETA,
            initial_state=initial,
            seed=exp7189._derived_seed("python", seed),
            burn_in=burn_in,
            retained=retained,
        )
    if arm == "persistent_pyo3":
        return dict(
            sampler.run_seeded_batch(
                np.asarray([initial], dtype=np.int8), [rust_seed], burn_in, retained
            )[0]
        )
    if arm == "subprocess_rust":
        request = {
            "operation": "seeded",
            "config": exp7189.instance_payload(instance, k, BETA),
            "initial_state": list(initial),
            "seed": rust_seed,
            "burn_in": burn_in,
            "retained": retained,
        }
        return _quiet_rust_request(bridge, request, timeout_s=180.0)
    raise ValueError(f"unknown arm: {arm}")


def run_quality_panels(
    binding: ModuleType,
    bridge: Path,
    *,
    sizes: Sequence[int] = SIZES,
    cardinalities: Sequence[int] = CARDINALITIES,
    seeds: Sequence[int] = SEEDS,
    burn_in: int = QUALITY_BURN_IN,
    retained: int = QUALITY_RETAINED,
) -> list[JsonDict]:
    """Measure long-chain ESS, occupation, energy agreement, and sector safety."""

    rows: list[JsonDict] = []
    samplers: dict[tuple[int, int], Any] = {}
    total = len(sizes) * len(cardinalities) * len(seeds) * len(ARMS)
    started_all = time.monotonic()
    last_report = started_all
    completed = 0
    for n in sizes:
        for k in cardinalities:
            setup_instance = slices.make_frustrated_instance(n, seeds[0])
            samplers[(n, k)] = exp7201._new_sampler(binding, setup_instance, k)
            for seed_index, seed in enumerate(seeds):
                instance = slices.make_frustrated_instance(n, seed)
                initial = _initial_state(n, k, seed=seed, chain=0)
                order = _arm_order(seed_index + n + k)
                for order_index, arm in enumerate(order):
                    if arm == "persistent_pyo3":
                        sampler = exp7201._new_sampler(binding, instance, k)
                    else:
                        sampler = samplers[(n, k)]
                    started = time.monotonic()
                    chain = _quality_chain(
                        arm,
                        sampler=sampler,
                        bridge=bridge,
                        instance=instance,
                        k=k,
                        initial=initial,
                        seed=seed,
                        burn_in=burn_in,
                        retained=retained,
                    )
                    latency = time.monotonic() - started
                    energies = [float(value) for value in chain["energies"]]
                    samples = chain["samples"]
                    occupations = [1.0 if state[0] == 1 else 0.0 for state in samples]
                    violations = sum(state.count(1) != k for state in samples)
                    energy = ess_diagnostics(energies, latency_s=latency)
                    occupation = ess_diagnostics(occupations, latency_s=latency)
                    sufficient = (
                        len(samples) >= QUALITY_MIN_RETAINED
                        and energy["ess"] is not None
                        and energy["ess"] >= QUALITY_MIN_ESS
                        and occupation["ess"] is not None
                        and occupation["ess"] >= QUALITY_MIN_ESS
                        and violations == 0
                    )
                    rows.append(
                        _finish_row(
                            {
                                "row_type": "quality_panel",
                                "unit_id": f"n{n}:k{k}:seed{seed}:{arm}:quality",
                                "arm": arm,
                                "n": n,
                                "k": k,
                                "batch_size": 1,
                                "seed": seed,
                                "metric": "long_chain_quality",
                                "arm_order": list(order),
                                "order_index": order_index,
                                "burn_in": burn_in,
                                "retained": len(samples),
                                "latency_s": latency,
                                "acceptance_rate": chain["accepted"] / max(1, chain["attempted"]),
                                "energy_mean": statistics.mean(energies),
                                "energy_std": statistics.stdev(energies),
                                "occupation_mean": statistics.mean(occupations),
                                "energy_ess": energy["ess"],
                                "energy_ess_per_second": energy["ess_per_second"],
                                "energy_lag_correlations": energy["lag_correlations"],
                                "occupation_ess": occupation["ess"],
                                "occupation_ess_per_second": occupation["ess_per_second"],
                                "occupation_lag_correlations": occupation["lag_correlations"],
                                "ess_estimator": energy["estimator"],
                                "sector_violations": violations,
                                "exact_law_claim": False,
                                "quality_row_sufficient": sufficient,
                                "trace_sha256": sha256_json(
                                    {"energies": energies, "occupations": occupations}
                                ),
                            }
                        )
                    )
                    completed += 1
                    now = time.monotonic()
                    if now - last_report >= 60.0:
                        print(
                            f"[heartbeat] elapsed_s={now - started_all:.3f} "
                            f"completed={completed}/{total} operation=quality_panel",
                            flush=True,
                        )
                        last_report = now
    return rows


def _standardized_from_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Recompute the declared standardization from retained row statistics."""

    pooled = math.sqrt((left["energy_std"] ** 2 + right["energy_std"] ** 2) / 2.0)
    difference = abs(left["energy_mean"] - right["energy_mean"])
    if pooled == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / pooled


def summarize_quality(quality_rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], bool]:
    """Apply per-seed fidelity and paired ESS-rate gates to each large cell."""

    grouped: defaultdict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        grouped[(int(row["n"]), int(row["k"]))].append(row)
    comparisons: list[JsonDict] = []
    for cell_index, ((n, k), selected) in enumerate(sorted(grouped.items())):
        by_seed_arm = {(row["seed"], row["arm"]): row for row in selected}
        seeds = sorted({int(row["seed"]) for row in selected})
        standardized: list[float] = []
        energy_ratios: list[float] = []
        occupation_ratios: list[float] = []
        complete = len(selected) == len(seeds) * len(ARMS)
        for seed in seeds:
            if any((seed, arm) not in by_seed_arm for arm in ARMS):
                complete = False
                continue
            python = by_seed_arm[(seed, "python_control")]
            process = by_seed_arm[(seed, "subprocess_rust")]
            persistent = by_seed_arm[(seed, "persistent_pyo3")]
            standardized.extend(
                (
                    _standardized_from_rows(python, persistent),
                    _standardized_from_rows(process, persistent),
                )
            )
            if (
                persistent["energy_ess_per_second"] is not None
                and process["energy_ess_per_second"] is not None
                and process["energy_ess_per_second"] > 0.0
            ):
                energy_ratios.append(
                    persistent["energy_ess_per_second"] / process["energy_ess_per_second"]
                )
            if (
                persistent["occupation_ess_per_second"] is not None
                and process["occupation_ess_per_second"] is not None
                and process["occupation_ess_per_second"] > 0.0
            ):
                occupation_ratios.append(
                    persistent["occupation_ess_per_second"] / process["occupation_ess_per_second"]
                )
        energy_ci = (
            _ci_dict(energy_ratios, seed=BOOTSTRAP_SEED + cell_index * 2)
            if len(energy_ratios) == len(seeds)
            else None
        )
        occupation_ci = (
            _ci_dict(occupation_ratios, seed=BOOTSTRAP_SEED + cell_index * 2 + 1)
            if len(occupation_ratios) == len(seeds)
            else None
        )
        max_standardized = max(standardized, default=math.inf)
        row_sufficient = complete and all(row["quality_row_sufficient"] is True for row in selected)
        passed = (
            row_sufficient
            and max_standardized <= ENERGY_STANDARDIZED_TOLERANCE
            and energy_ci is not None
            and energy_ci["lower"] >= ESS_RATE_RATIO_CI_MIN
            and occupation_ci is not None
            and occupation_ci["lower"] >= ESS_RATE_RATIO_CI_MIN
        )
        comparisons.append(
            _finish_row(
                {
                    "row_type": "quality_comparison",
                    "unit_id": f"n{n}:k{k}:quality_comparison",
                    "arm": "persistent_pyo3_vs_subprocess_and_python",
                    "n": n,
                    "k": k,
                    "metric": "quality_gate",
                    "independent_seed_count": len(seeds),
                    "comparison_complete": complete,
                    "all_rows_meet_draw_and_ess_minimum": row_sufficient,
                    "max_standardized_energy_difference": max_standardized,
                    "standardized_energy_tolerance": ENERGY_STANDARDIZED_TOLERANCE,
                    "energy_ess_rate_ratio_ci95": energy_ci,
                    "occupation_ess_rate_ratio_ci95": occupation_ci,
                    "ess_rate_ratio_ci_lower_minimum": ESS_RATE_RATIO_CI_MIN,
                    "passed": passed,
                }
            )
        )
    expected_cells = len(SIZES) * len(CARDINALITIES)
    sufficient = len(comparisons) == expected_cells and all(row["passed"] for row in comparisons)
    return comparisons, sufficient


def summarize_primary_gate(
    throughput_rows: Sequence[Mapping[str, Any]], *, sample_quality_sufficient: bool
) -> JsonDict:
    """Compute the fixed batch-one latency gate from paired independent seeds."""

    selected = [
        row
        for row in throughput_rows
        if all(row.get(field) == value for field, value in PRIMARY_CELL.items())
    ]
    by_seed_arm = {(row["seed"], row["arm"]): row for row in selected}
    seeds = sorted({int(row["seed"]) for row in selected})
    boundary_ratios: list[float] = []
    python_ratios: list[float] = []
    for seed in seeds:
        if any((seed, arm) not in by_seed_arm for arm in ARMS):
            continue
        persistent = by_seed_arm[(seed, "persistent_pyo3")]["latency_s"]
        process = by_seed_arm[(seed, "subprocess_rust")]["latency_s"]
        python = by_seed_arm[(seed, "python_control")]["latency_s"]
        boundary_ratios.append(process / persistent)
        python_ratios.append(python / persistent)
    boundary_ci = _ci_dict(boundary_ratios, seed=BOOTSTRAP_SEED + 100)
    python_ci = _ci_dict(python_ratios, seed=BOOTSTRAP_SEED + 101)
    parity = len(selected) == len(seeds) * len(ARMS) and all(
        row.get("parity_passed") is True for row in selected
    )
    zero_sector = bool(selected) and all(row.get("sector_violations") == 0 for row in selected)
    boundary_value = (
        parity
        and zero_sector
        and boundary_ci is not None
        and boundary_ci["lower"] > 1.0
        and sample_quality_sufficient
    )
    nfr_met = (
        parity
        and zero_sector
        and boundary_ci is not None
        and boundary_ci["lower"] >= NFR_01_SPEEDUP_TARGET
        and sample_quality_sufficient
    )
    return {
        "cell": dict(PRIMARY_CELL),
        "single_query_only": True,
        "independent_seed_count": len(seeds),
        "parity_passed": parity,
        "zero_sector_violations": zero_sector,
        "sample_quality_sufficient": sample_quality_sufficient,
        "latency_speedup_over_subprocess_ci95": boundary_ci,
        "python_speedup_ci95": python_ci,
        "local_latency_ci_lower_threshold": 1.0,
        "nfr_01_speedup_target": NFR_01_SPEEDUP_TARGET,
        "nfr_01_10x_met": nfr_met,
        "boundary_value_score": int(boundary_value),
    }


def combined_rows(payload: Mapping[str, Any]) -> list[Any]:
    """Keep one canonical order for the artifact-wide row ledger."""

    return (
        list(payload.get("distribution_rows", []))
        + list(payload.get("control_rows", []))
        + list(payload.get("cold_initialization_rows", []))
        + list(payload.get("throughput_rows", []))
        + list(payload.get("quality_rows", []))
        + list(payload.get("quality_comparison_rows", []))
    )


def _base_artifact(
    *, root: Path, run_date: str, checks: list[JsonDict], hashes: dict[str, str]
) -> JsonDict:
    """Create every required field before choosing complete or blocked status."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": checks,
        "inference_substrate": "not_started",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host_cpu",
        "host_identity": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": [],
        "sample_size_budget": {
            "planned_cold_initialization_rows": len(SIZES) * len(CARDINALITIES) * len(ARMS),
            "completed_cold_initialization_rows": 0,
            "planned_throughput_rows": (
                len(SIZES)
                * len(CARDINALITIES)
                * len(BATCH_SIZES)
                * len(SEEDS)
                * len(ARMS)
                * len(PROTOCOLS)
            ),
            "completed_throughput_rows": 0,
            "planned_distribution_rows": 2,
            "completed_distribution_rows": 0,
            "planned_quality_rows": len(SIZES) * len(CARDINALITIES) * len(SEEDS) * len(ARMS),
            "completed_quality_rows": 0,
            "quality_burn_in_per_chain": QUALITY_BURN_IN,
            "quality_retained_per_chain": QUALITY_RETAINED,
            "equal_work_proposals_per_chain": PROPOSALS_PER_CHAIN,
            "equal_wall_seconds_per_batch_call": WALL_BUDGET_S,
            "independent_units": len(SEEDS),
            "exclusions": [],
        },
        "random_seed": {
            "study_seeds": list(SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "python_rng": "CPython random.Random",
            "rust_rng": "independent 64-bit LCG in carnot-samplers",
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "slice_comparison_complete_score": 0,
        "boundary_value_score": 0,
        "throughput_rows": [],
        "distribution_rows": [],
        "nfr_01_10x_met": False,
        "acceptance_gate_boundary": dict(PRIMARY_CELL),
        "sample_quality_sufficient": False,
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_invoked": False,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "compiled_binding_receipt": {},
        "cold_initialization_rows": [],
        "quality_rows": [],
        "quality_comparison_rows": [],
        "control_rows": [],
        "primary_gate": {},
        "upstream_performance_null": {
            "artifact": str(UPSTREAM_RESULT_PATH),
            "origin_artifact": "results/experiment_7189_v633_rust_slice_parity.json",
            "field": "nfr_01_10x_met",
            "observed_value": False,
            "promoted": False,
        },
        "spec_refs": [
            "REQ-RUSTPY-7202",
            "SCENARIO-RUSTPY-7202-MATCHED-BOUNDARIES",
            "SCENARIO-RUSTPY-7202-QUALITY-GATES-SPEED",
        ],
    }


def _blocked_artifact(artifact: JsonDict, *, failed: Mapping[str, Any], started: float) -> JsonDict:
    """Publish a diagnosed external block without invented computation rows."""

    artifact.update(
        {
            "status": "blocked_external_precondition",
            "inference_substrate": "blocked_no_run",
            "inference_substrate_class": "blocked_no_run",
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
                f"blocked_external_precondition: {failed.get('check')} failed before computation"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _checkpoint(root: Path, phase: int, artifact: Mapping[str, Any]) -> None:
    """Persist measured phase evidence outside the terminal deliverable path."""

    path = root / CHECKPOINT_DIR / f"experiment_7202_phase_{phase}.json"
    payload = {
        "task_id": TASK_ID,
        "phase": phase,
        "status": artifact.get("status"),
        "completed_counts": artifact.get("sample_size_budget"),
        "elapsed_s": artifact.get("duration_s"),
        "artifact": dict(artifact),
    }
    atomic_write(path, payload)


def build_artifact(
    *,
    root: Path,
    run_date: str = RUN_DATE,
    binding_module: ModuleType | None = None,
    binding_receipt: Mapping[str, Any] | None = None,
    bridge_path: Path | None = None,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build measured evidence or stop at one diagnosed external prerequisite."""

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
        return _blocked_artifact(artifact, failed=failed, started=started)
    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": str(UPSTREAM_RESULT_PATH),
        "field": "pyo3_slice_ready_score",
        "expected_value": 1,
        "observed_value": 1,
    }

    _progress(1, "start", "compiled PyO3 load and subprocess build")
    try:
        if binding_module is None or binding_receipt is None:
            print("[phase 1 native start] cargo build and PyO3 load", flush=True)
            extension = exp7201.build_pyo3_extension(root)
            binding, receipt = exp7201.load_compiled_binding(root, extension)
            print("[phase 1 native end] cargo build and PyO3 load", flush=True)
        else:
            binding, receipt = binding_module, dict(binding_receipt)
        print("[phase 1 subprocess start] fixed-cardinality bridge build", flush=True)
        bridge = bridge_path or exp7189.build_rust_bridge(root)
        print("[phase 1 subprocess end] fixed-cardinality bridge build", flush=True)
        if not bridge.is_file() or not os.access(bridge, os.X_OK):
            raise RuntimeError(f"compiled subprocess baseline is unavailable: {bridge}")
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        _progress(1, "end", f"compiled prerequisite failed error={exc}")
        return _blocked_artifact(
            artifact,
            failed={
                "check": "compiled_binding_or_bridge",
                "upstream": "host_toolchain",
                "field": "carnot._rust.RustFixedCardinalitySampler",
                "expected_value": "compiled binding and bridge load successfully",
                "observed_value": str(exc),
            },
            started=started,
        )
    artifact["compiled_binding_receipt"] = receipt
    _progress(1, "end", f"compiled path={receipt['loaded_module_path']}")

    _progress(2, "start", "exact law and negative controls")
    artifact["distribution_rows"] = run_exact_law_checks()
    artifact["control_rows"] = run_negative_controls()
    artifact["sample_size_budget"]["completed_distribution_rows"] = len(
        artifact["distribution_rows"]
    )
    artifact["duration_s"] = time.monotonic() - started
    _checkpoint(root, 2, artifact)
    _progress(2, "end", "exact law rows=2 controls=2")

    _progress(3, "start", "matched boundary benchmarks")
    print("[phase 3 benchmark start] equal-work and 50 ms equal-wall roster", flush=True)
    cold, throughput = run_throughput_benchmarks(binding, bridge)
    print(f"[phase 3 benchmark end] throughput_rows={len(throughput)}", flush=True)
    artifact["cold_initialization_rows"] = cold
    artifact["throughput_rows"] = throughput
    artifact["sample_size_budget"].update(
        {
            "completed_cold_initialization_rows": len(cold),
            "completed_throughput_rows": len(throughput),
        }
    )
    artifact["duration_s"] = time.monotonic() - started
    _checkpoint(root, 3, artifact)
    _progress(3, "end", f"matched boundary rows={len(throughput)}")

    _progress(4, "start", "independent long-chain quality panels")
    print("[phase 4 benchmark start] 1024 burn-in and 8192 retained proposals", flush=True)
    quality = run_quality_panels(binding, bridge)
    print(f"[phase 4 benchmark end] quality_rows={len(quality)}", flush=True)
    artifact["quality_rows"] = quality
    artifact["sample_size_budget"]["completed_quality_rows"] = len(quality)
    artifact["duration_s"] = time.monotonic() - started
    _checkpoint(root, 4, artifact)
    _progress(4, "end", f"quality rows={len(quality)}")

    _progress(5, "start", "quality and primary gate aggregation")
    quality_comparisons, quality_sufficient = summarize_quality(quality)
    primary = summarize_primary_gate(throughput, sample_quality_sufficient=quality_sufficient)
    artifact["quality_comparison_rows"] = quality_comparisons
    artifact["sample_quality_sufficient"] = quality_sufficient
    artifact["primary_gate"] = primary
    artifact["boundary_value_score"] = primary["boundary_value_score"]
    artifact["nfr_01_10x_met"] = primary["nfr_01_10x_met"]
    complete = (
        len(cold) == len(SIZES) * len(CARDINALITIES) * len(ARMS)
        and len(throughput)
        == len(SIZES)
        * len(CARDINALITIES)
        * len(BATCH_SIZES)
        * len(SEEDS)
        * len(ARMS)
        * len(PROTOCOLS)
        and len(quality) == len(SIZES) * len(CARDINALITIES) * len(SEEDS) * len(ARMS)
        and len(artifact["distribution_rows"]) == 2
        and all(row["passed"] for row in artifact["distribution_rows"])
        and all(row["control_detected"] for row in artifact["control_rows"])
        and len(quality_comparisons) == len(SIZES) * len(CARDINALITIES)
    )
    artifact["slice_comparison_complete_score"] = int(complete)
    artifact["status"] = "complete"
    artifact["inference_substrate"] = (
        "cpu_exact_solver_or_simulator: exact finite-slice enumeration, Python pair-swap "
        "simulation, compiled subprocess Rust, and persistent compiled PyO3 execution"
    )
    artifact["inference_substrate_class"] = "cpu_exact_solver_or_simulator"
    artifact["verdict_class"] = "positive" if primary["boundary_value_score"] == 1 else "null"
    quality_text = "sufficient" if quality_sufficient else "insufficient"
    value_text = "passed" if primary["boundary_value_score"] == 1 else "did not pass"
    nfr_text = "met" if primary["nfr_01_10x_met"] else "not met"
    artifact["honest_verdict"] = (
        "complete: all fixed boundary, law, control, and long-chain quality rows were measured. "
        f"Sample-quality evidence was {quality_text}. The primary local boundary gate {value_text}. "
        f"The unchanged NFR-01 10x target was {nfr_text}."
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact["rows"] = combined_rows(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _checkpoint(root, 5, artifact)
    _progress(5, "end", f"boundary_value_score={artifact['boundary_value_score']}")
    return artifact


def _row_hashes_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Verify that every row binds its final field values."""

    return all(
        row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        and "unit_id" in row
        and "arm" in row
        and "seed" in row
        and "metric" in row
        and row.get("error") is None
        and row.get("abstention") is False
        for row in rows
    )


def validate_artifact(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute row coverage, scientific gates, claims, hashes, and terminal state."""

    if not REQUIRED_ARTIFACT_FIELDS.issubset(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if payload.get("MODEL_SPECS") != [] or payload.get("model_invoked") is not False:
        errors.append("model_declaration_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate") != "blocked_no_run"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("slice_comparison_complete_score") != 0
            or combined_rows(payload)
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not gate.get("failed_check")
            or not gate.get("upstream")
            or not gate.get("field")
        ):
            errors.append("blocked_state_invalid")
        return list(dict.fromkeys(errors))

    distribution = payload.get("distribution_rows", [])
    controls = payload.get("control_rows", [])
    cold = payload.get("cold_initialization_rows", [])
    throughput = payload.get("throughput_rows", [])
    quality = payload.get("quality_rows", [])
    quality_comparisons = payload.get("quality_comparison_rows", [])
    all_rows = combined_rows(payload)
    if payload.get("rows") != all_rows or not _row_hashes_valid(all_rows):
        errors.append("rows_invalid")

    distribution_valid = (
        len(distribution) == 2
        and {row.get("n") for row in distribution} == {8, 12}
        and all(
            row.get("k") == 2
            and row.get("passed") is True
            and row.get("stationary_residual", math.inf) <= 1.0e-10
            and row.get("detailed_balance_error", math.inf) <= 1.0e-10
            for row in distribution
        )
    )
    if not distribution_valid:
        errors.append("distribution_rows_invalid")
    controls_valid = (
        len(controls) == 2
        and {row.get("control") for row in controls}
        == {"constant_chain_ess", "biased_transition_target"}
        and all(
            row.get("control_detected") is True and row.get("speed_eligible") is False
            for row in controls
        )
    )
    if not controls_valid:
        errors.append("control_rows_invalid")

    expected_cold = {f"n{n}:k{k}:{arm}:cold" for n in SIZES for k in CARDINALITIES for arm in ARMS}
    cold_valid = (
        len(cold) == len(expected_cold)
        and {row.get("unit_id") for row in cold} == expected_cold
        and all(
            row.get("latency_s", -1.0) >= 0.0 and row.get("excluded_from_warm_statistics") is True
            for row in cold
        )
    )
    if not cold_valid:
        errors.append("cold_initialization_rows_invalid")

    expected_throughput = {
        f"n{n}:k{k}:batch{batch}:seed{seed}:{arm}:{protocol}"
        for n in SIZES
        for k in CARDINALITIES
        for batch in BATCH_SIZES
        for seed in SEEDS
        for arm in ARMS
        for protocol in PROTOCOLS
    }
    throughput_valid = (
        len(throughput) == len(expected_throughput)
        and {row.get("unit_id") for row in throughput} == expected_throughput
        and all(
            row.get("latency_s", 0.0) > 0.0
            and row.get("sector_violations") == 0
            and row.get("data_transfer_and_synchronization_charged") is True
            and row.get("exact_law_claim") is False
            and (
                isinstance(row.get("parity_passed"), bool)
                if row.get("protocol") == "equal_work"
                else row.get("deadline_overshoot_s", -1.0) >= 0.0
            )
            for row in throughput
        )
    )
    if not throughput_valid:
        errors.append("throughput_rows_invalid")

    expected_quality = {
        f"n{n}:k{k}:seed{seed}:{arm}:quality"
        for n in SIZES
        for k in CARDINALITIES
        for seed in SEEDS
        for arm in ARMS
    }
    quality_valid = (
        len(quality) == len(expected_quality)
        and {row.get("unit_id") for row in quality} == expected_quality
        and all(
            row.get("burn_in") == QUALITY_BURN_IN
            and row.get("retained") == QUALITY_RETAINED
            and row.get("sector_violations") == 0
            and row.get("exact_law_claim") is False
            and row.get("quality_row_sufficient")
            is (
                row.get("retained", 0) >= QUALITY_MIN_RETAINED
                and row.get("energy_ess") is not None
                and row.get("energy_ess", 0.0) >= QUALITY_MIN_ESS
                and row.get("occupation_ess") is not None
                and row.get("occupation_ess", 0.0) >= QUALITY_MIN_ESS
                and row.get("sector_violations") == 0
            )
            for row in quality
        )
    )
    if not quality_valid:
        errors.append("quality_rows_invalid")

    recomputed_comparisons, recomputed_quality = summarize_quality(quality)
    if quality_comparisons != recomputed_comparisons:
        errors.append("quality_comparison_rows_invalid")
    if payload.get("sample_quality_sufficient") is not recomputed_quality:
        errors.append("sample_quality_gate_invalid")

    primary = summarize_primary_gate(throughput, sample_quality_sufficient=recomputed_quality)
    if payload.get("primary_gate") != primary:
        errors.append("primary_gate_invalid")
    if payload.get("boundary_value_score") != primary["boundary_value_score"]:
        errors.append("boundary_value_score_invalid")
    if payload.get("nfr_01_10x_met") is not primary["nfr_01_10x_met"]:
        errors.append("nfr_01_gate_invalid")
    if payload.get("acceptance_gate_boundary") != PRIMARY_CELL:
        errors.append("acceptance_gate_boundary_invalid")

    complete = (
        distribution_valid
        and controls_valid
        and cold_valid
        and throughput_valid
        and quality_valid
        and quality_comparisons == recomputed_comparisons
    )
    if payload.get("slice_comparison_complete_score") != int(complete):
        errors.append("completion_score_invalid")
    expected_verdict = "positive" if primary["boundary_value_score"] == 1 else "null"
    if (
        payload.get("status") != "complete"
        or payload.get("verdict_class") != expected_verdict
        or not str(payload.get("honest_verdict", "")).startswith("complete:")
    ):
        errors.append("terminal_verdict_invalid")
    if payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_invalid")
    if payload.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("gate_summary_invalid")
    if payload.get("upstream_performance_null", {}).get("promoted") is not False:
        errors.append("upstream_failed_value_promoted")
    if root is not None:
        recorded = payload.get("source_artifact_hashes", {})
        expected = {str(path) for path in REQUIRED_SOURCE_PATHS}
        if (
            not isinstance(recorded, Mapping)
            or set(recorded) != expected
            or any(
                not (root / path).is_file() or recorded[path] != sha256_file(root / path)
                for path in expected
            )
        ):
            errors.append("source_artifact_hashes_invalid")
    return list(dict.fromkeys(errors))


def atomic_write(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Publish complete JSON through one same-directory atomic replacement."""

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
    """Build, validate, and atomically write the terminal Exp7202 artifact."""

    artifact = build_artifact(root=root, run_date=run_date)
    _progress(6, "start", "final artifact validation")
    errors = validate_artifact(artifact, root=root)
    _progress(6, "end", f"final artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7202 artifact: {errors}")
    _progress(7, "start", "final atomic write")
    receipt = atomic_write(output, artifact)
    _progress(7, "end", f"final atomic write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and optional read-only artifact validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full study or validate durable bytes without repository writes."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        _progress(6, "start", f"validation path={args.validate}")
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(6, "end", "validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(6, "end", f"validation errors={len(errors)}")
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


if __name__ == "__main__":  # pragma: no cover - the script entrypoint owns this branch.
    raise SystemExit(main())
