"""Compare down-up and pair-swap sampling quality at matched charged cost.

The study is deliberately opt-in. It enumerates each finite target before
sampling, executes both Python kernels with independent streams, and keeps
sample quality separate from measurement completion. It does not change a
sampler default or claim the sparse-SK theorem or hardware performance.

Spec: REQ-ISING-7216 and SCENARIO-ISING-7216-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import tempfile
import time
from typing import Any, BinaryIO, Mapping, Sequence, TextIO

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7202_v634_slice_cost_quality as exp7202
from carnot.samplers import experiment_7215_down_up as down_up


JsonDict = dict[str, Any]
Subset = tuple[int, ...]

RUN_DATE = "20260911"
TASK_ID = "exp7216-down-up-quality"
MILESTONE = "2026.09.635"
RESULT_PATH = Path("results/experiment_7216_v635_down_up_quality.json")
TRACE_PATH = Path("results/checkpoints/experiment_7216_v635_down_up_quality_traces.jsonl.gz")
CHECKPOINT_DIR = Path("results/checkpoints")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
UPSTREAM_PATH = Path("results/experiment_7215_v635_down_up_prototype.json")
V634_PATH = Path("results/experiment_7202_v634_slice_cost_quality.json")

CELLS = ((32, 1.0), (32, 2.0), (16, 1.0))
PRIMARY_CELL = {"n": 32, "k": 2, "beta": 1.0}
CARDINALITY = 2
GRAPH_SEEDS = tuple(range(7216001, 7216011))
CHAINS = (0, 1, 2, 3)
ARMS = ("down_up", "pair_swap_metropolis")
PROTOCOLS = ("equal_work", "equal_wall")
ENERGY_EVALUATION_BUDGET = 100_000
WALL_BUDGET_S = 2.0
QUALITY_BURN_IN = 4096
QUALITY_RETAINED = 16384
MEASUREMENT_CAP_S = 1800.0
ESS_MINIMUM = 200.0
SPLIT_RHAT_MAXIMUM = 1.05
OCCUPANCY_ERROR_MAXIMUM = 0.02
ENERGY_STANDARDIZED_ERROR_MAXIMUM = 0.05
ESS_LAG_WINDOW = 1024
BOOTSTRAP_SEED = 7216002
BOOTSTRAP_RESAMPLES = 10_000
EXACT_TOLERANCE = 1.0e-10
MODEL_SPECS: list[JsonDict] = []

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": [
        {
            "upstream": "exp7215-down-up-prototype",
            "artifact_field": "down_up_kernel_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    "prior_failures": [
        {
            "experiment_id": "exp7202-slice-cost-quality",
            "verdict": (
                "complete: all fixed boundary, law, control, and long-chain quality rows "
                "were measured. Sample-quality evidence was insufficient. The primary local "
                "boundary gate did not pass. The unchanged NFR-01 10x target was not met."
            ),
            "addressed_by": (
                "Change the Markov kernel from pair-swap Metropolis to target-weighted down-up "
                "resampling; measure stationary fidelity and mixing cost separately from the "
                "retired 10x bridge claim."
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
    UPSTREAM_PATH,
    V634_PATH,
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7189_v633_rust_slice_parity.py"),
    Path("python/carnot/experiment_7202_v634_slice_cost_quality.py"),
    Path("python/carnot/samplers/backend.py"),
    Path("python/carnot/samplers/experiment_7215_down_up.py"),
    Path("python/carnot/experiment_7216_v635_down_up_quality.py"),
    Path("scripts/experiments/experiment_7216_v635_down_up_quality.py"),
    Path("tests/python/test_experiment_7216_v635_down_up_quality.py"),
    SPEC_PATH,
)

FIELD_PRINCIPLES = {
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": (
        "Use exactly host, kv260, gatemate or polarfire; these tasks execute on host."
    ),
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": (
        "Keep unit_id, arm, seed, metric, error and abstention for each comparison; do not "
        "replace numeric rows with a task roster."
    ),
    "sample_size_budget": (
        "Retain planned, attempted, completed, censored and independent-unit counts."
    ),
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": (
        "Every blocked_* verdict names failed check, upstream, field, expected and observed value."
    ),
    "verifier_is_oracle": (
        "Same correctness authority remains circular even with a separate implementation."
    ),
    "verdict_class": (
        "Use exactly positive | circular_positive | null | blocked | disqualified | partial; "
        "only incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings; blocked_* for external absence. "
        "Readiness is not scientific value."
    ),
    "down_up_comparison_complete_score": (
        "Measurement completion remains separate from sufficient quality."
    ),
    "down_up_value_score": "Only joint fidelity and cost-adjusted quality support the new kernel.",
    "acceptance_gate_quality": ("The primary cell and all quality/throughput clauses are frozen."),
    "quality_rows": "Energy and nondegenerate occupancy probes expose stuck chains.",
    "matched_budget_rows": ("Normalization and target-energy calls count toward actual work."),
    "exact_authority_rows": "Finite-law targets prevent comparing two equally biased chains.",
    "paper_replication_claimed": (
        "False; arbitrary frustrated graphs do not establish the SK theorem."
    ),
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
}

REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES) | {
    "task_id",
    "milestone",
    "root",
    "quality_summary_rows",
    "trace_archive",
    "nfr_01_10x_met",
    "upstream_performance_null",
    "paper_replication_claimed",
    "rust_10x_speed_claimed",
    "tsu_execution_claimed",
    "sparse_sk_theorem_claimed",
    "hardware_power_savings_claimed",
    "hamiltonian_contract",
    "spec_refs",
}


@dataclass(frozen=True)
class ExactAuthority:
    """Store one complete target law and its independently calculated moments."""

    n: int
    k: int
    beta: float
    states: tuple[Subset, ...]
    energies: tuple[float, ...]
    probabilities: tuple[float, ...]
    occupancy_means: tuple[float, ...]
    occupancy_variances: tuple[float, ...]
    energy_mean: float
    energy_variance: float
    normalization_error: float
    energy_parity_error: float
    probes: tuple[int, ...]
    authority_sha256: str
    state_to_index: dict[Subset, int] = field(repr=False, compare=False)


@dataclass(frozen=True)
class ChainRun:
    """Retain one actual kernel run before its raw trace is compressed."""

    trace_indices: tuple[int, ...]
    initial_index: int
    rng_seed: int
    transitions: int
    energy_evaluations: int
    normalizations: int
    unfinished_conditional_energy_evaluations: int
    accepted_moves: int
    self_transitions: int
    sector_violations: int
    elapsed_s: float
    wall_budget_overshoot_s: float | None
    truncated: bool


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush every numbered boundary so the conductor can observe real work."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode deterministic finite JSON for checksums and row identities."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError("nonfinite value in canonical JSON") from exc


def sha256_json(value: Any) -> str:
    """Hash canonical JSON and retain the algorithm name beside the digest."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash every byte of one required input or generated trace archive."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every stable terminal field except the checksum that stores this hash."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the common evidence columns before hashing the final row values."""

    row.setdefault("arm", "not_applicable")
    row.setdefault("seed", None)
    row.setdefault("metric", row.get("row_type", "measurement"))
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = sha256_json(row)
    return row


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the explicit two-key principle/value representation."""

    if (
        isinstance(value, Mapping)
        and set(value) == {"value", "principle"}
        and isinstance(value.get("principle"), str)
    ):
        return value["value"]
    return value


def upstream_quarantine_observation(
    upstream: Mapping[str, Any], *, manifest_match: bool
) -> JsonDict:
    """Join artifact and manifest quarantine signals before any gate is read."""

    names = (
        "flagged_adversarial",
        "quarantined",
        "quarantine",
        "quarantine_flags",
        "disqualified",
        "invalidated",
    )
    observed = {name: upstream.get(name) for name in names}
    active = [name for name, value in observed.items() if value not in (None, False, "", [], {})]
    if manifest_match:
        active.append("exclusion_manifest")
    return {
        **observed,
        "exclusion_manifest_match": manifest_match,
        "active_flags": active,
        "quarantined": bool(active),
    }


def gated_upstream_value(
    upstream: Mapping[str, Any], quarantine: Mapping[str, Any], field_name: str
) -> Any:
    """Reject quarantine before reading and narrowly unwrapping a gate value."""

    if quarantine.get("quarantined") is True:
        return "not_consumed_due_to_quarantine"
    candidate = upstream.get(field_name)
    return unwrap_principled_value(candidate)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the exact active roadmap fields that authorize this experiment."""

    try:
        document = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        return None
    task = next(
        (
            item
            for item in document["tasks"]
            if isinstance(item, Mapping) and item.get("id") == TASK_ID
        ),
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


def _read_json_object(path: Path) -> Mapping[str, Any]:
    """Return one decoded object or an empty object for an unreadable prerequisite."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _check_row(
    check: str,
    upstream: str,
    field_name: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Use one diagnostic shape for successful and failed prerequisites."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field_name,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def collect_preconditions(
    root: Path,
    *,
    result_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str]]:
    """Print first, then retain actual source, tool, contract, and gate observations."""

    result = result_path or root / RESULT_PATH
    checkpoints = checkpoint_dir or root / CHECKPOINT_DIR
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}

    def announce(name: str) -> None:
        print(f"[phase 0 check start] {name}", flush=True)

    announce("required source bytes")
    sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }
    for path in REQUIRED_SOURCE_PATHS:
        if sizes[str(path)] not in (None, 0):
            hashes[str(path)] = sha256_file(root / path)
    checks.append(
        _check_row(
            "required_source_bytes",
            str(root),
            "REQUIRED_SOURCE_PATHS",
            "all files exist and are nonempty",
            sizes,
            all(size is not None and size > 0 for size in sizes.values()),
        )
    )

    announce("driving capability specification")
    spec_file = root / SPEC_PATH
    spec_text = spec_file.read_text(encoding="utf-8") if spec_file.is_file() else ""
    spec_observed = {
        "exists": spec_file.is_file(),
        "req_present": "### REQ-ISING-7216" in spec_text,
        "scenarios_present": "### SCENARIO-ISING-7216-" in spec_text,
    }
    checks.append(
        _check_row(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-* and SCENARIO-*",
            {"exists": True, "req_present": True, "scenarios_present": True},
            spec_observed,
            all(spec_observed.values()),
        )
    )

    announce("active roadmap task and producer gate fields")
    contract = _task_contract(root)
    checks.append(
        _check_row(
            "roadmap_task_contract",
            str(ROADMAP_PATH),
            "id,milestone,deliverable,gated_on,prior_failures",
            EXPECTED_TASK_CONTRACT,
            contract,
            contract == EXPECTED_TASK_CONTRACT,
        )
    )

    announce("upstream artifact bytes and quarantine")
    upstream_file = root / UPSTREAM_PATH
    upstream = _read_json_object(upstream_file)
    exclusion_text = (
        (root / EXCLUSION_PATH).read_text(encoding="utf-8")
        if (root / EXCLUSION_PATH).is_file()
        else ""
    )
    manifest_match = any(
        token in exclusion_text
        for token in ("exp7215-down-up-prototype", "experiment_7215_v635_down_up_prototype")
    )
    quarantine = upstream_quarantine_observation(upstream, manifest_match=manifest_match)
    checks.append(
        _check_row(
            "upstream_quarantine",
            f"{UPSTREAM_PATH} and {EXCLUSION_PATH}",
            "artifact and manifest quarantine signals",
            {"quarantined": False},
            quarantine,
            bool(upstream) and quarantine["quarantined"] is False,
        )
    )

    announce("upstream producer authentication")
    producer_errors = down_up.validate_artifact(upstream) if upstream else ["unreadable_artifact"]
    authentication = {
        "producer_valid": not producer_errors,
        "producer_errors": producer_errors,
        "validator": "carnot.samplers.experiment_7215_down_up.validate_artifact",
    }
    checks.append(
        _check_row(
            "upstream_authentication",
            str(UPSTREAM_PATH),
            "producer validator",
            {"producer_valid": True},
            authentication,
            quarantine["quarantined"] is False and not producer_errors,
        )
    )

    announce("exact producer gate value")
    gate_value = gated_upstream_value(upstream, quarantine, "down_up_kernel_ready_score")
    checks.append(
        _check_row(
            "upstream_gate",
            str(UPSTREAM_PATH),
            "down_up_kernel_ready_score",
            1,
            gate_value,
            quarantine["quarantined"] is False and not producer_errors and gate_value == 1,
        )
    )

    announce("V634 null authentication and non-promotion")
    v634_file = root / V634_PATH
    v634 = _read_json_object(v634_file)
    v634_manifest_match = any(
        token in exclusion_text
        for token in ("exp7202-slice-cost-quality", "experiment_7202_v634_slice_cost_quality")
    )
    v634_quarantine = upstream_quarantine_observation(v634, manifest_match=v634_manifest_match)
    v634_errors = exp7202.validate_artifact(v634) if v634 else ["unreadable_artifact"]
    v634_observed = {
        "producer_valid": not v634_errors,
        "producer_errors": v634_errors,
        "quarantine": v634_quarantine,
        "boundary_value_score": gated_upstream_value(v634, v634_quarantine, "boundary_value_score"),
        "nfr_01_10x_met": gated_upstream_value(v634, v634_quarantine, "nfr_01_10x_met"),
        "promoted": False,
    }
    checks.append(
        _check_row(
            "v634_nfr_null",
            str(V634_PATH),
            "producer validity, quarantine, boundary_value_score,nfr_01_10x_met,promoted",
            {
                "producer_valid": "observed context only",
                "quarantined": False,
                "boundary_value_score": 0,
                "nfr_01_10x_met": False,
                "promoted": False,
            },
            v634_observed,
            (
                v634_quarantine["quarantined"] is False
                and v634_observed["boundary_value_score"] == 0
                and v634_observed["nfr_01_10x_met"] is False
                and v634_observed["promoted"] is False
            ),
        )
    )

    announce("Python imports and validation tools")
    tool_state = {
        "python": bool(os.sys.executable),
        "numpy": bool(np.__version__),
        "pyyaml": bool(yaml.__version__),
        "pytest": (root / ".venv/bin/pytest").is_file(),
        "ruff": (root / ".venv/bin/ruff").is_file() or shutil.which("ruff") is not None,
        "mypy": (root / ".venv/bin/mypy").is_file() or shutil.which("mypy") is not None,
    }
    checks.append(
        _check_row(
            "imports_and_tools",
            "host Python environment",
            "python,numpy,pyyaml,pytest,ruff,mypy",
            "all available",
            tool_state,
            all(tool_state.values()),
        )
    )

    announce("result and checkpoint directories")
    directory_state = {
        "result_parent_writable": result.parent.is_dir() and os.access(result.parent, os.W_OK),
        "checkpoint_dir_writable": checkpoints.is_dir() and os.access(checkpoints, os.W_OK),
        "terminal_is_not_checkpoint": result.parent.resolve() != checkpoints.resolve(),
    }
    checks.append(
        _check_row(
            "output_directories",
            str(root),
            "result and checkpoint locations",
            {
                "result_parent_writable": True,
                "checkpoint_dir_writable": True,
                "terminal_is_not_checkpoint": True,
            },
            directory_state,
            all(directory_state.values()),
        )
    )
    return checks, hashes


def enumerate_exact_authority(instance: slices.SliceInstance, *, beta: float) -> ExactAuthority:
    """Enumerate a complete k=2 target with the independent scalar energy path."""

    slices.validate_instance(instance)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    states = down_up.enumerate_subsets(instance.n, CARDINALITY)
    energies = tuple(
        slices.reference_energy(instance, down_up.subset_to_spins(instance.n, state))
        for state in states
    )
    sampler_energies = tuple(
        slices.ising_energy(instance, down_up.subset_to_spins(instance.n, state))
        for state in states
    )
    log_weights = np.asarray([-beta * energy for energy in energies], dtype=np.float64)
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    probabilities_array = weights / float(np.sum(weights))
    probabilities = tuple(float(value) for value in probabilities_array)
    occupancy_means = tuple(
        math.fsum(probability for state, probability in zip(states, probabilities) if site in state)
        for site in range(instance.n)
    )
    occupancy_variances = tuple(mean * (1.0 - mean) for mean in occupancy_means)
    energy_mean = math.fsum(
        probability * energy for probability, energy in zip(probabilities, energies, strict=True)
    )
    energy_variance = math.fsum(
        probability * (energy - energy_mean) ** 2
        for probability, energy in zip(probabilities, energies, strict=True)
    )
    normalization_error = abs(math.fsum(probabilities) - 1.0)
    energy_parity_error = max(
        abs(reference - sampled)
        for reference, sampled in zip(energies, sampler_energies, strict=True)
    )
    probes = (0, instance.n // 3, 2 * instance.n // 3)
    authority_material = {
        "n": instance.n,
        "k": CARDINALITY,
        "beta": beta,
        "states": states,
        "energies": energies,
        "probabilities": probabilities,
        "occupancy_means": occupancy_means,
        "occupancy_variances": occupancy_variances,
        "energy_mean": energy_mean,
        "energy_variance": energy_variance,
        "energy_convention": slices.EDGE_COUNTING_CONVENTION,
    }
    return ExactAuthority(
        n=instance.n,
        k=CARDINALITY,
        beta=float(beta),
        states=states,
        energies=energies,
        probabilities=probabilities,
        occupancy_means=occupancy_means,
        occupancy_variances=occupancy_variances,
        energy_mean=energy_mean,
        energy_variance=energy_variance,
        normalization_error=normalization_error,
        energy_parity_error=energy_parity_error,
        probes=probes,
        authority_sha256=sha256_json(authority_material),
        state_to_index={state: index for index, state in enumerate(states)},
    )


class TargetEnergyEvaluator:
    """Evaluate target energies in batches while counting every returned state energy."""

    def __init__(self, instance: slices.SliceInstance, authority: ExactAuthority) -> None:
        if authority.n != instance.n or authority.k != CARDINALITY:
            raise ValueError("authority does not match the Ising instance")
        self._instance = instance
        self._authority = authority
        self._spins = np.asarray(
            [down_up.subset_to_spins(instance.n, state) for state in authority.states],
            dtype=np.float64,
        )
        self._left = np.asarray([edge[0] for edge in instance.edges], dtype=np.int64)
        self._right = np.asarray([edge[1] for edge in instance.edges], dtype=np.int64)
        self._couplings = np.asarray([edge[2] for edge in instance.edges], dtype=np.float64)
        self._fields = np.asarray(instance.fields, dtype=np.float64)
        self.evaluations = 0

    def evaluate_indices(self, indices: Sequence[int]) -> tuple[float, ...]:
        """Return exact edge-once energies and charge one evaluation per index."""

        selected = tuple(indices)
        if not selected:
            return ()
        if any(index < 0 or index >= len(self._authority.states) for index in selected):
            raise ValueError("state index lies outside the exact authority")
        batch = self._spins[np.asarray(selected, dtype=np.int64)]
        edge_terms = (batch[:, self._left] * batch[:, self._right]) @ self._couplings
        field_terms = batch @ self._fields
        values = -edge_terms - field_terms
        self.evaluations += len(selected)
        return tuple(float(value) for value in values)


def derive_stream_seed(
    graph_seed: int,
    n: int,
    beta: float,
    chain_id: int,
    arm: str,
    protocol: str,
) -> int:
    """Domain-separate every graph, cell, chain, arm, and protocol stream."""

    material = f"7216:{graph_seed}:{n}:{beta:g}:{chain_id}:{arm}:{protocol}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def overdispersed_initial_indices(authority: ExactAuthority) -> tuple[int, int, int, int]:
    """Choose four fixed states spread across the exact energy order."""

    ordered = sorted(range(len(authority.states)), key=lambda index: authority.energies[index])
    final = len(ordered) - 1
    positions = (0, round(final / 3), round(2 * final / 3), final)
    return tuple(ordered[position] for position in positions)  # type: ignore[return-value]


def randomized_arm_order(
    graph_seed: int, n: int, beta: float, chain_id: int, protocol: str
) -> tuple[str, str]:
    """Randomize first-arm order from a frozen stream unrelated to either chain."""

    order = list(ARMS)
    rng = random.Random(derive_stream_seed(graph_seed, n, beta, chain_id, "arm_order", protocol))
    rng.shuffle(order)
    return order[0], order[1]


def _down_up_candidate_indices(
    authority: ExactAuthority, current_index: int, rng: random.Random
) -> tuple[int, ...]:
    """Return all replacement states after one uniform down choice."""

    source = authority.states[current_index]
    removed_index = rng.randrange(len(source))
    core = source[:removed_index] + source[removed_index + 1 :]
    return tuple(
        authority.state_to_index[tuple(sorted((*core, site)))]
        for site in range(authority.n)
        if site not in core
    )


def _categorical_index(probabilities: np.ndarray, rng: random.Random) -> int:
    """Draw one stable categorical index from a caller-owned random stream."""

    uniform = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if uniform < cumulative:
            return index
    return len(probabilities) - 1


def sample_chain(
    instance: slices.SliceInstance,
    authority: ExactAuthority,
    *,
    arm: str,
    rng_seed: int,
    initial_index: int,
    transition_limit: int | None = None,
    energy_budget: int | None = None,
    wall_budget_s: float | None = None,
    hard_deadline: float | None = None,
) -> ChainRun:
    """Run one kernel under exactly one transition, energy, or wall stopping rule."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    supplied = sum(value is not None for value in (transition_limit, energy_budget, wall_budget_s))
    if supplied != 1:
        raise ValueError("provide exactly one stopping budget")
    if transition_limit is not None and transition_limit < 0:
        raise ValueError("transition limit must be nonnegative")
    if energy_budget is not None and energy_budget < 1:
        raise ValueError("energy budget must include initialization")
    if wall_budget_s is not None and (not math.isfinite(wall_budget_s) or wall_budget_s <= 0.0):
        raise ValueError("wall budget must be positive and finite")
    if initial_index < 0 or initial_index >= len(authority.states):
        raise ValueError("initial state index lies outside the authority")

    rng = random.Random(rng_seed)
    evaluator = TargetEnergyEvaluator(instance, authority)
    started = time.monotonic()
    deadline = started + wall_budget_s if wall_budget_s is not None else None
    current_index = initial_index
    current_energy = evaluator.evaluate_indices((current_index,))[0]
    trace: list[int] = []
    normalizations = 0
    unfinished = 0
    accepted = 0
    last_report = started
    truncated = False

    while True:
        if transition_limit is not None and len(trace) >= transition_limit:
            break
        if hard_deadline is not None and time.monotonic() >= hard_deadline:
            truncated = transition_limit is not None and len(trace) < transition_limit
            break
        if deadline is not None and time.monotonic() >= deadline:
            break

        if arm == "down_up":
            candidate_indices = _down_up_candidate_indices(authority, current_index, rng)
            step_cost = len(candidate_indices)
            if energy_budget is not None:
                remaining = energy_budget - evaluator.evaluations
                if remaining < step_cost:
                    if remaining > 0:
                        evaluator.evaluate_indices(candidate_indices[:remaining])
                        unfinished = remaining
                    break
            energies = evaluator.evaluate_indices(candidate_indices)
            log_weights = -authority.beta * np.asarray(energies, dtype=np.float64)
            weights = np.exp(log_weights - float(np.max(log_weights)))
            probabilities = weights / float(np.sum(weights))
            selected = _categorical_index(probabilities, rng)
            next_index = candidate_indices[selected]
            current_energy = energies[selected]
            normalizations += 1
        else:
            if energy_budget is not None and evaluator.evaluations >= energy_budget:
                break
            source = authority.states[current_index]
            removed = source[rng.randrange(len(source))]
            outside = tuple(site for site in range(authority.n) if site not in source)
            inserted = outside[rng.randrange(len(outside))]
            candidate = tuple(
                sorted(tuple(site for site in source if site != removed) + (inserted,))
            )
            next_index = authority.state_to_index[candidate]
            candidate_energy = evaluator.evaluate_indices((next_index,))[0]
            threshold = min(0.0, -authority.beta * (candidate_energy - current_energy))
            if math.log(max(rng.random(), np.finfo(float).tiny)) < threshold:
                current_energy = candidate_energy
            else:
                next_index = current_index

        accepted += int(next_index != current_index)
        current_index = next_index
        trace.append(current_index)
        now = time.monotonic()
        if now - last_report >= 60.0:
            print(
                f"[heartbeat] elapsed_s={now - started:.3f} completed={len(trace)} "
                f"operation={arm}_chain",
                flush=True,
            )
            last_report = now

    elapsed = time.monotonic() - started
    if energy_budget is not None and evaluator.evaluations < energy_budget:
        remaining = energy_budget - evaluator.evaluations
        filler = tuple(current_index for _ in range(remaining))
        evaluator.evaluate_indices(filler)
        unfinished += remaining
        elapsed = time.monotonic() - started
    previous = initial_index
    self_transitions = 0
    for state_index in trace:
        self_transitions += int(state_index == previous)
        previous = state_index
    violations = sum(len(authority.states[index]) != authority.k for index in trace)
    overshoot = max(0.0, elapsed - wall_budget_s) if wall_budget_s is not None else None
    return ChainRun(
        trace_indices=tuple(trace),
        initial_index=initial_index,
        rng_seed=rng_seed,
        transitions=len(trace),
        energy_evaluations=evaluator.evaluations,
        normalizations=normalizations,
        unfinished_conditional_energy_evaluations=unfinished,
        accepted_moves=accepted,
        self_transitions=self_transitions,
        sector_violations=violations,
        elapsed_s=elapsed,
        wall_budget_overshoot_s=overshoot,
        truncated=truncated,
    )


def _autocorrelations(values: Sequence[float], lag_window: int) -> list[float] | None:
    """Compute Exp7187-compatible lag correlations with an FFT dot-product path."""

    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    variance_sum = float(np.dot(centered, centered))
    if variance_sum == 0.0:
        return None
    used = min(lag_window, len(array) - 1)
    transform_size = 1 << (2 * len(array) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=transform_size)
    products = np.fft.irfft(spectrum * np.conjugate(spectrum), n=transform_size)[: used + 1]
    return [float(value / variance_sum) for value in products]


def ess_diagnostics(
    values: Sequence[float], *, target_variance: float, latency_s: float
) -> JsonDict:
    """Report initial-positive-sequence ESS without rewarding a constant trace."""

    if target_variance < -EXACT_TOLERANCE or not math.isfinite(target_variance):
        raise ValueError("target variance must be finite and nonnegative")
    count = len(values)
    structurally_degenerate = target_variance <= EXACT_TOLERANCE
    if structurally_degenerate:
        return {
            "draw_count": count,
            "target_variance": target_variance,
            "structurally_degenerate": True,
            "constant_observed": len(set(values)) <= 1,
            "ess": None,
            "ess_per_second": None,
            "monte_carlo_standard_error": 0.0,
            "lag_correlations": None,
            "estimator": "initial_positive_autocorrelation_sequence_fft",
            "qualified": True,
        }
    correlations = (
        _autocorrelations(values, min(ESS_LAG_WINDOW, max(1, count - 1))) if count >= 2 else None
    )
    if correlations is None:
        return {
            "draw_count": count,
            "target_variance": target_variance,
            "structurally_degenerate": False,
            "constant_observed": len(set(values)) <= 1,
            "ess": None,
            "ess_per_second": None,
            "monte_carlo_standard_error": None,
            "lag_correlations": None,
            "estimator": "initial_positive_autocorrelation_sequence_fft",
            "qualified": False,
        }
    positive_sum = 0.0
    for correlation in correlations[1:]:
        if correlation <= 0.0:
            break
        positive_sum += correlation
    ess = float(min(count, max(1.0, count / (1.0 + 2.0 * positive_sum))))
    sample_variance = statistics.variance(values)
    mcse = math.sqrt(sample_variance / ess)
    return {
        "draw_count": count,
        "target_variance": target_variance,
        "structurally_degenerate": False,
        "constant_observed": False,
        "ess": ess,
        "ess_per_second": ess / latency_s if latency_s > 0.0 else None,
        "monte_carlo_standard_error": mcse,
        "lag_correlations": correlations[:17],
        "estimator": "initial_positive_autocorrelation_sequence_fft",
        "qualified": ess >= ESS_MINIMUM,
    }


def split_rhat(chains: Sequence[Sequence[float]]) -> float | None:
    """Compute one split-chain potential scale reduction from four chains."""

    if len(chains) != 4:
        raise ValueError("split R-hat requires exactly four chains")
    length = min(len(chain) for chain in chains)
    half = length // 2
    if half < 2:
        return None
    split = [np.asarray(chain[:half], dtype=np.float64) for chain in chains]
    split += [np.asarray(chain[length - half : length], dtype=np.float64) for chain in chains]
    within = float(np.mean([np.var(chain, ddof=1) for chain in split]))
    if within == 0.0:
        return None
    means = np.asarray([np.mean(chain) for chain in split], dtype=np.float64)
    between = half * float(np.var(means, ddof=1))
    estimate = ((half - 1.0) / half) * within + between / half
    return math.sqrt(max(0.0, estimate / within))


def _probe_values(
    authority: ExactAuthority, trace_indices: Sequence[int]
) -> dict[str, list[float]]:
    """Materialize the prespecified energy and occupancy observations."""

    values = {"energy": [authority.energies[index] for index in trace_indices]}
    for site in authority.probes:
        values[f"occupancy_{site}"] = [
            1.0 if site in authority.states[index] else 0.0 for index in trace_indices
        ]
    return values


def _exact_probe(authority: ExactAuthority, name: str) -> tuple[float, float]:
    """Return the frozen exact mean and variance for one named probe."""

    if name == "energy":
        return authority.energy_mean, authority.energy_variance
    site = int(name.removeprefix("occupancy_"))
    return authority.occupancy_means[site], authority.occupancy_variances[site]


def _total_variation(
    authority: ExactAuthority, trace_indices: Sequence[int]
) -> tuple[float | None, float | None]:
    """Estimate finite-state total variation and its multinomial delta-method MCSE."""

    if not trace_indices:
        return None, None
    count = len(trace_indices)
    observed = np.zeros(len(authority.states), dtype=np.float64)
    for index, frequency in Counter(trace_indices).items():
        observed[index] = frequency / count
    exact = np.asarray(authority.probabilities, dtype=np.float64)
    difference = observed - exact
    total_variation = 0.5 * float(np.sum(np.abs(difference)))
    gradient = 0.5 * np.sign(difference)
    variance = float(np.sum(observed * gradient**2) - np.sum(observed * gradient) ** 2)
    return total_variation, math.sqrt(max(0.0, variance) / count)


def quality_row_from_run(
    instance: slices.SliceInstance,
    authority: ExactAuthority,
    run: ChainRun,
    *,
    graph_seed: int,
    chain_id: int,
    arm: str,
    burn_in: int,
    retained: int,
    arm_order: Sequence[str],
    order_index: int,
) -> JsonDict:
    """Score one retained chain against exact energy and occupancy authorities."""

    available = max(0, len(run.trace_indices) - burn_in)
    retained_count = min(retained, available)
    selected = run.trace_indices[burn_in : burn_in + retained_count]
    probe_values = _probe_values(authority, selected)
    diagnostics: dict[str, JsonDict] = {}
    for name, values in probe_values.items():
        exact_mean, exact_variance = _exact_probe(authority, name)
        diagnostic = ess_diagnostics(
            values, target_variance=exact_variance, latency_s=run.elapsed_s
        )
        observed_mean = statistics.mean(values) if values else None
        absolute_error = abs(observed_mean - exact_mean) if observed_mean is not None else None
        standardized_error = (
            absolute_error / math.sqrt(exact_variance)
            if absolute_error is not None and exact_variance > EXACT_TOLERANCE
            else 0.0
            if absolute_error is not None
            else None
        )
        diagnostics[name] = {
            **diagnostic,
            "exact_mean": exact_mean,
            "observed_mean": observed_mean,
            "absolute_mean_error": absolute_error,
            "standardized_mean_error": standardized_error,
        }
    total_variation, total_variation_mcse = _total_variation(authority, selected)
    complete = retained_count == retained and not run.truncated
    return _finish_row(
        {
            "row_type": "quality_chain",
            "unit_id": (
                f"n{instance.n}:k{CARDINALITY}:beta{authority.beta:g}:graph{graph_seed}:"
                f"chain{chain_id}:{arm}:quality"
            ),
            "arm": arm,
            "seed": run.rng_seed,
            "graph_seed": graph_seed,
            "chain_id": chain_id,
            "n": instance.n,
            "k": CARDINALITY,
            "beta": authority.beta,
            "metric": "independent_exact_law_quality",
            "error": None if complete else "measurement_cap_reached",
            "abstention": not complete,
            "arm_order": list(arm_order),
            "order_index": order_index,
            "initial_state_index": run.initial_index,
            "burn_in": burn_in,
            "retained": retained_count,
            "planned_retained": retained,
            "attempted_transitions": run.transitions,
            "latency_s": run.elapsed_s,
            "energy_evaluations": run.energy_evaluations,
            "normalizations": run.normalizations,
            "accepted_moves": run.accepted_moves,
            "self_transitions": run.self_transitions,
            "sector_violations": run.sector_violations,
            "probe_diagnostics": diagnostics,
            "empirical_total_variation": total_variation,
            "total_variation_mcse": total_variation_mcse,
            "exact_authority_sha256": authority.authority_sha256,
            "trace_sha256": sha256_json(run.trace_indices),
            "complete": complete,
            "identical_rng_tape_parity_claimed": False,
        }
    )


class TraceArchiveWriter:
    """Atomically stream compressed raw traces without retaining them in the artifact."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._temporary: Path | None = None
        self._raw: BinaryIO | None = None
        self._gzip: gzip.GzipFile | None = None
        self._text: TextIO | None = None
        self.record_count = 0

    def open(self) -> None:
        """Open one same-directory temporary gzip stream."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        self._temporary = Path(name)
        self._raw = os.fdopen(descriptor, "wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(self._gzip, encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one canonical JSON trace record and flush it to the compressor."""

        if self._text is None:
            raise RuntimeError("trace archive is not open")
        self._text.write(canonical_json(record) + "\n")
        self.record_count += 1

    def close(self) -> JsonDict:
        """Flush, sync, atomically replace, and return a content receipt."""

        if None in (self._text, self._gzip, self._raw, self._temporary):
            raise RuntimeError("trace archive is not open")
        assert self._text is not None
        assert self._gzip is not None
        assert self._raw is not None
        assert self._temporary is not None
        self._text.flush()
        self._text.detach()
        self._gzip.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        os.replace(self._temporary, self.path)
        self._text = None
        self._gzip = None
        self._raw = None
        self._temporary = None
        return {
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "bytes": self.path.stat().st_size,
            "record_count": self.record_count,
            "format": "canonical-json-lines+gzip",
            "compressed": True,
            "atomic_replace": True,
        }

    def abort(self) -> None:
        """Close an interrupted stream and remove only its private temporary file."""

        for handle in (self._text, self._gzip, self._raw):
            if handle is not None:
                try:
                    handle.close()
                except (OSError, ValueError):
                    pass
        if self._temporary is not None and self._temporary.exists():
            self._temporary.unlink()
        self._text = None
        self._gzip = None
        self._raw = None
        self._temporary = None


def _authority_row(
    instance: slices.SliceInstance, authority: ExactAuthority, graph_seed: int
) -> JsonDict:
    """Retain exact finite-law moments, probes, and Hamiltonian scale."""

    validation = slices.validate_instance(instance)
    coupling_values = [edge[2] for edge in instance.edges]
    finite_law_passed = (
        authority.normalization_error <= EXACT_TOLERANCE
        and min(authority.probabilities) > 0.0
        and authority.energy_parity_error <= EXACT_TOLERANCE
        and all(len(state) == CARDINALITY for state in authority.states)
        and validation["frustrated_triangle"] is True
        and all(field_value != 0.0 for field_value in instance.fields)
    )
    return _finish_row(
        {
            "row_type": "exact_authority",
            "unit_id": (
                f"n{instance.n}:k{CARDINALITY}:beta{authority.beta:g}:graph{graph_seed}:exact"
            ),
            "arm": "independent_scalar_enumerator",
            "seed": graph_seed,
            "graph_seed": graph_seed,
            "n": instance.n,
            "k": CARDINALITY,
            "beta": authority.beta,
            "metric": "finite_target_law",
            "instance_hash": instance.instance_hash,
            "state_count": len(authority.states),
            "normalization_error": authority.normalization_error,
            "minimum_probability": min(authority.probabilities),
            "energy_parity_error": authority.energy_parity_error,
            "energy_mean": authority.energy_mean,
            "energy_variance": authority.energy_variance,
            "occupancy_means": list(authority.occupancy_means),
            "occupancy_variances": list(authority.occupancy_variances),
            "probe_indices": list(authority.probes),
            "probe_exact_variances": {
                f"occupancy_{site}": authority.occupancy_variances[site]
                for site in authority.probes
            },
            "authority_sha256": authority.authority_sha256,
            "hamiltonian_scale": {
                "edge_count": len(instance.edges),
                "coupling_min": min(coupling_values),
                "coupling_max": max(coupling_values),
                "coupling_l1": math.fsum(abs(value) for value in coupling_values),
                "field_min": min(instance.fields),
                "field_max": max(instance.fields),
                "field_abs_min": min(abs(value) for value in instance.fields),
                "field_abs_max": max(abs(value) for value in instance.fields),
                "energy_convention": slices.EDGE_COUNTING_CONVENTION,
            },
            "finite_law_passed": finite_law_passed,
            "enumerated_before_sampling": True,
        }
    )


def run_exact_panel() -> tuple[list[JsonDict], dict[tuple[int, float, int], ExactAuthority]]:
    """Enumerate every frozen graph/cell target before any sampler is called."""

    rows: list[JsonDict] = []
    authorities: dict[tuple[int, float, int], ExactAuthority] = {}
    total = len(CELLS) * len(GRAPH_SEEDS)
    started = time.monotonic()
    last_report = started
    for n, beta in CELLS:
        for graph_seed in GRAPH_SEEDS:
            instance = slices.make_frustrated_instance(n, graph_seed)
            authority = enumerate_exact_authority(instance, beta=beta)
            authorities[(n, beta, graph_seed)] = authority
            rows.append(_authority_row(instance, authority, graph_seed))
            now = time.monotonic()
            if now - last_report >= 60.0:
                print(
                    f"[phase 1 progress] completed={len(rows)}/{total} "
                    f"elapsed_s={now - started:.3f}",
                    flush=True,
                )
                last_report = now
    return rows, authorities


def _matched_probe_diagnostics(authority: ExactAuthority, run: ChainRun) -> JsonDict:
    """Report bounded-trace ESS without treating these rows as qualification evidence."""

    probes = _probe_values(authority, run.trace_indices)
    diagnostics: JsonDict = {}
    for name, values in probes.items():
        _, variance = _exact_probe(authority, name)
        diagnostics[name] = ess_diagnostics(
            values, target_variance=variance, latency_s=run.elapsed_s
        )
    nondegenerate = [
        item["ess_per_second"]
        for item in diagnostics.values()
        if not item["structurally_degenerate"] and item["ess_per_second"] is not None
    ]
    diagnostics["minimum_probe_ess_per_second"] = (
        min(nondegenerate) if len(nondegenerate) == len(probes) else None
    )
    return diagnostics


def run_matched_panels(
    authorities: Mapping[tuple[int, float, int], ExactAuthority],
    archive: TraceArchiveWriter,
) -> list[JsonDict]:
    """Execute both arms under equal energy-evaluation and equal wall budgets."""

    rows: list[JsonDict] = []
    total = len(CELLS) * len(GRAPH_SEEDS) * len(CHAINS) * len(PROTOCOLS) * len(ARMS)
    started = time.monotonic()
    last_report = started
    for n, beta in CELLS:
        for graph_seed in GRAPH_SEEDS:
            instance = slices.make_frustrated_instance(n, graph_seed)
            authority = authorities[(n, beta, graph_seed)]
            initials = overdispersed_initial_indices(authority)
            for chain_id in CHAINS:
                for protocol in PROTOCOLS:
                    order = randomized_arm_order(graph_seed, n, beta, chain_id, protocol)
                    for order_index, arm in enumerate(order):
                        stream_seed = derive_stream_seed(
                            graph_seed, n, beta, chain_id, arm, protocol
                        )
                        if protocol == "equal_work":
                            run = sample_chain(
                                instance,
                                authority,
                                arm=arm,
                                rng_seed=stream_seed,
                                initial_index=initials[chain_id],
                                energy_budget=ENERGY_EVALUATION_BUDGET,
                            )
                        else:
                            run = sample_chain(
                                instance,
                                authority,
                                arm=arm,
                                rng_seed=stream_seed,
                                initial_index=initials[chain_id],
                                wall_budget_s=WALL_BUDGET_S,
                            )
                        archive.write(
                            {
                                "unit_id": (
                                    f"n{n}:k{CARDINALITY}:beta{beta:g}:graph{graph_seed}:"
                                    f"chain{chain_id}:{arm}:{protocol}"
                                ),
                                "panel": "matched_budget",
                                "rng_seed": stream_seed,
                                "initial_state_index": initials[chain_id],
                                "trace_indices": list(run.trace_indices),
                            }
                        )
                        row = {
                            "row_type": "matched_budget_chain",
                            "unit_id": (
                                f"n{n}:k{CARDINALITY}:beta{beta:g}:graph{graph_seed}:"
                                f"chain{chain_id}:{arm}:{protocol}"
                            ),
                            "arm": arm,
                            "seed": stream_seed,
                            "graph_seed": graph_seed,
                            "chain_id": chain_id,
                            "n": n,
                            "k": CARDINALITY,
                            "beta": beta,
                            "metric": "charged_kernel_efficiency",
                            "protocol": protocol,
                            "arm_order": list(order),
                            "order_index": order_index,
                            "initialization_charged": True,
                            "normalization_charged": True,
                            "target_energy_calls_charged": True,
                            "energy_evaluation_budget": (
                                ENERGY_EVALUATION_BUDGET if protocol == "equal_work" else None
                            ),
                            "wall_budget_s": WALL_BUDGET_S if protocol == "equal_wall" else None,
                            "energy_evaluations": run.energy_evaluations,
                            "completed_transitions": run.transitions,
                            "normalizations": run.normalizations,
                            "unfinished_conditional_energy_evaluations": (
                                run.unfinished_conditional_energy_evaluations
                            ),
                            "latency_s": run.elapsed_s,
                            "wall_budget_overshoot_s": run.wall_budget_overshoot_s,
                            "accepted_moves": run.accepted_moves,
                            "self_transitions": run.self_transitions,
                            "sector_violations": run.sector_violations,
                            "probe_diagnostics": _matched_probe_diagnostics(authority, run),
                            "trace_sha256": sha256_json(run.trace_indices),
                            "efficiency_evidence_only": True,
                            "quality_qualification_row": False,
                            "identical_rng_tape_parity_claimed": False,
                        }
                        rows.append(_finish_row(row))
                        now = time.monotonic()
                        if now - last_report >= 60.0:
                            print(
                                f"[phase 2 progress] completed={len(rows)}/{total} "
                                f"elapsed_s={now - started:.3f}",
                                flush=True,
                            )
                            last_report = now
    return rows


def summarize_quality_group(
    rows: Sequence[Mapping[str, Any]],
    traces: Mapping[str, Sequence[int]],
    authority: ExactAuthority,
) -> JsonDict:
    """Combine four chains without hiding the weakest ESS or a failed exact mean."""

    ordered = sorted(rows, key=lambda row: int(row["chain_id"]))
    complete = len(ordered) == len(CHAINS) and all(row.get("complete") is True for row in ordered)
    probe_names = ("energy",) + tuple(f"occupancy_{site}" for site in authority.probes)
    probe_summaries: JsonDict = {}
    ess_rates: list[float] = []
    all_chain_ess = True
    all_rhat = True
    exact_means = True
    for name in probe_names:
        chain_values = [
            _probe_values(authority, traces[str(row["unit_id"])])[name] for row in ordered
        ]
        exact_mean, exact_variance = _exact_probe(authority, name)
        structurally_degenerate = exact_variance <= EXACT_TOLERANCE
        chain_ess = [row["probe_diagnostics"][name]["ess"] for row in ordered]
        chain_ess_passed = structurally_degenerate or (
            len(chain_ess) == len(CHAINS)
            and all(value is not None and value >= ESS_MINIMUM for value in chain_ess)
        )
        rhat = split_rhat(chain_values) if len(chain_values) == len(CHAINS) else None
        rhat_passed = structurally_degenerate or (rhat is not None and rhat <= SPLIT_RHAT_MAXIMUM)
        pooled = [value for chain in chain_values for value in chain]
        observed_mean = statistics.mean(pooled) if pooled else None
        absolute_error = abs(observed_mean - exact_mean) if observed_mean is not None else None
        standardized_error = (
            absolute_error / math.sqrt(exact_variance)
            if absolute_error is not None and exact_variance > EXACT_TOLERANCE
            else 0.0
            if absolute_error is not None
            else None
        )
        tolerance = (
            ENERGY_STANDARDIZED_ERROR_MAXIMUM if name == "energy" else OCCUPANCY_ERROR_MAXIMUM
        )
        compared_error = standardized_error if name == "energy" else absolute_error
        mean_tolerance_passed = compared_error is not None and compared_error <= tolerance
        total_ess = (
            math.fsum(float(value) for value in chain_ess if value is not None)
            if chain_ess and all(value is not None for value in chain_ess)
            else None
        )
        total_latency = math.fsum(float(row["latency_s"]) for row in ordered)
        ess_per_second = (
            total_ess / total_latency
            if total_ess is not None and total_latency > 0.0 and not structurally_degenerate
            else None
        )
        if not structurally_degenerate and ess_per_second is not None:
            ess_rates.append(ess_per_second)
        total_mcse = (
            math.sqrt(statistics.variance(pooled) / total_ess)
            if total_ess is not None and total_ess > 0.0 and len(pooled) > 1
            else 0.0
            if structurally_degenerate
            else None
        )
        probe_summaries[name] = {
            "exact_mean": exact_mean,
            "exact_variance": exact_variance,
            "structurally_degenerate": structurally_degenerate,
            "observed_mean": observed_mean,
            "absolute_mean_error": absolute_error,
            "standardized_mean_error": standardized_error,
            "monte_carlo_standard_error": total_mcse,
            "chain_ess": chain_ess,
            "minimum_chain_ess": min(chain_ess)
            if all(value is not None for value in chain_ess)
            else None,
            "chain_ess_passed": chain_ess_passed,
            "split_rhat": rhat,
            "split_rhat_passed": rhat_passed,
            "mean_tolerance": tolerance,
            "mean_tolerance_passed": mean_tolerance_passed,
            "ess_per_second": ess_per_second,
        }
        all_chain_ess = all_chain_ess and chain_ess_passed
        all_rhat = all_rhat and rhat_passed
        exact_means = exact_means and mean_tolerance_passed
    zero_sector = bool(ordered) and all(row.get("sector_violations") == 0 for row in ordered)
    minimum_rate = min(ess_rates) if len(ess_rates) == len(probe_names) else None
    qualified = complete and zero_sector and all_chain_ess and all_rhat and exact_means
    first = ordered[0] if ordered else {}
    return _finish_row(
        {
            "row_type": "quality_summary",
            "unit_id": (
                f"n{authority.n}:k{authority.k}:beta{authority.beta:g}:"
                f"graph{first.get('graph_seed')}:{first.get('arm')}:summary"
            ),
            "arm": first.get("arm", "missing"),
            "seed": first.get("graph_seed"),
            "graph_seed": first.get("graph_seed"),
            "n": authority.n,
            "k": authority.k,
            "beta": authority.beta,
            "metric": "split_chain_quality_qualification",
            "chain_count": len(ordered),
            "complete": complete,
            "zero_sector_violations": zero_sector,
            "all_chain_ess_passed": all_chain_ess,
            "all_split_rhat_passed": all_rhat,
            "exact_mean_tolerances_passed": exact_means,
            "minimum_probe_ess_per_second": minimum_rate,
            "probe_summaries": probe_summaries,
            "quality_qualified": qualified,
        }
    )


def run_quality_panels(
    authorities: Mapping[tuple[int, float, int], ExactAuthority],
    archive: TraceArchiveWriter,
) -> tuple[list[JsonDict], list[JsonDict], bool]:
    """Run every long chain unless the frozen total measurement deadline truncates it."""

    rows: list[JsonDict] = []
    summaries: list[JsonDict] = []
    total = len(CELLS) * len(GRAPH_SEEDS) * len(CHAINS) * len(ARMS)
    started = time.monotonic()
    hard_deadline = started + MEASUREMENT_CAP_S
    last_report = started
    truncated = False
    for n, beta in CELLS:
        if truncated:
            break
        for graph_seed in GRAPH_SEEDS:
            if truncated:
                break
            instance = slices.make_frustrated_instance(n, graph_seed)
            authority = authorities[(n, beta, graph_seed)]
            initials = overdispersed_initial_indices(authority)
            grouped_rows: defaultdict[str, list[JsonDict]] = defaultdict(list)
            traces: dict[str, Sequence[int]] = {}
            for chain_id in CHAINS:
                if truncated:
                    break
                order = randomized_arm_order(graph_seed, n, beta, chain_id, "quality")
                for order_index, arm in enumerate(order):
                    stream_seed = derive_stream_seed(graph_seed, n, beta, chain_id, arm, "quality")
                    run = sample_chain(
                        instance,
                        authority,
                        arm=arm,
                        rng_seed=stream_seed,
                        initial_index=initials[chain_id],
                        transition_limit=QUALITY_BURN_IN + QUALITY_RETAINED,
                        hard_deadline=hard_deadline,
                    )
                    row = quality_row_from_run(
                        instance,
                        authority,
                        run,
                        graph_seed=graph_seed,
                        chain_id=chain_id,
                        arm=arm,
                        burn_in=QUALITY_BURN_IN,
                        retained=QUALITY_RETAINED,
                        arm_order=order,
                        order_index=order_index,
                    )
                    rows.append(row)
                    grouped_rows[arm].append(row)
                    traces[str(row["unit_id"])] = run.trace_indices[
                        QUALITY_BURN_IN : QUALITY_BURN_IN + int(row["retained"])
                    ]
                    archive.write(
                        {
                            "unit_id": row["unit_id"],
                            "panel": "quality",
                            "rng_seed": stream_seed,
                            "initial_state_index": initials[chain_id],
                            "burn_in": QUALITY_BURN_IN,
                            "planned_retained": QUALITY_RETAINED,
                            "trace_indices": list(run.trace_indices),
                            "truncated": run.truncated,
                        }
                    )
                    truncated = truncated or not bool(row["complete"])
                    now = time.monotonic()
                    if now - last_report >= 60.0:
                        print(
                            f"[phase 3 progress] completed={len(rows)}/{total} "
                            f"elapsed_s={now - started:.3f}",
                            flush=True,
                        )
                        last_report = now
                    if truncated:
                        break
            for arm in ARMS:
                selected = grouped_rows.get(arm, [])
                if selected:
                    summaries.append(summarize_quality_group(selected, traces, authority))
    return rows, summaries, truncated


def paired_bootstrap_ci(
    ratios: Sequence[float], *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES
) -> JsonDict:
    """Bootstrap the mean paired graph ratio from one frozen random stream."""

    if (
        not ratios
        or resamples < 1
        or any(value <= 0.0 or not math.isfinite(value) for value in ratios)
    ):
        raise ValueError("paired ratios and resamples must be finite and positive")
    rng = random.Random(seed)
    estimates = [
        statistics.mean(ratios[rng.randrange(len(ratios))] for _ in ratios)
        for _ in range(resamples)
    ]
    lower, estimate, upper = np.percentile(estimates, [2.5, 50.0, 97.5])
    return {
        "lower": float(lower),
        "estimate": float(estimate),
        "upper": float(upper),
        "paired_graphs": len(ratios),
        "resamples": resamples,
        "seed": seed,
        "estimand": "mean paired graph ratio",
    }


def evaluate_primary_gate(
    quality_summaries: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    *,
    panel_complete: bool,
) -> JsonDict:
    """Require exact comparator quality before calculating the throughput interval."""

    selected = [
        row
        for row in quality_summaries
        if all(row.get(key) == value for key, value in PRIMARY_CELL.items())
    ]
    selected_exact = [
        row
        for row in exact_rows
        if all(row.get(key) == value for key, value in PRIMARY_CELL.items())
    ]
    by_graph_arm = {(row.get("graph_seed"), row.get("arm")): row for row in selected}
    expected_pairs = {(graph_seed, arm) for graph_seed in GRAPH_SEEDS for arm in ARMS}
    summary_complete = set(by_graph_arm) == expected_pairs
    finite_law = (
        len(selected_exact) == len(GRAPH_SEEDS)
        and {row.get("graph_seed") for row in selected_exact} == set(GRAPH_SEEDS)
        and all(row.get("finite_law_passed") is True for row in selected_exact)
    )
    zero_sector = summary_complete and all(
        row.get("zero_sector_violations") is True for row in selected
    )
    ess_passed = summary_complete and all(
        row.get("all_chain_ess_passed") is True for row in selected
    )
    rhat_passed = summary_complete and all(
        row.get("all_split_rhat_passed") is True for row in selected
    )
    comparator_quality = summary_complete and all(
        row.get("exact_mean_tolerances_passed") is True for row in selected
    )
    ratios: list[float] = []
    interval: JsonDict | None = None
    if comparator_quality:
        for graph_seed in GRAPH_SEEDS:
            down_rate = by_graph_arm[(graph_seed, "down_up")].get("minimum_probe_ess_per_second")
            pair_rate = by_graph_arm[(graph_seed, "pair_swap_metropolis")].get(
                "minimum_probe_ess_per_second"
            )
            if (
                isinstance(down_rate, (int, float))
                and isinstance(pair_rate, (int, float))
                and down_rate > 0.0
                and pair_rate > 0.0
            ):
                ratios.append(float(down_rate / pair_rate))
        if len(ratios) == len(GRAPH_SEEDS):
            interval = paired_bootstrap_ci(ratios)
    throughput_passed = interval is not None and interval["lower"] > 1.0
    passed = (
        panel_complete
        and summary_complete
        and finite_law
        and zero_sector
        and ess_passed
        and rhat_passed
        and comparator_quality
        and throughput_passed
    )
    abstention_reason = None
    if not comparator_quality:
        abstention_reason = "insufficient_comparator_quality"
    elif interval is None:
        abstention_reason = "incomplete_paired_graph_rates"
    return {
        "cell": dict(PRIMARY_CELL),
        "panel_complete": panel_complete,
        "summary_complete": summary_complete,
        "finite_law_checks_passed": finite_law,
        "zero_sector_violations": zero_sector,
        "all_nondegenerate_chain_ess_at_least_200": ess_passed,
        "all_split_rhat_at_most_1_05": rhat_passed,
        "comparator_quality_sufficient": comparator_quality,
        "exact_mean_tolerances_required_before_throughput": True,
        "paired_graph_ratios": ratios if interval is not None else [],
        "ess_per_second_ratio_ci95": interval,
        "throughput_lower_endpoint_above_one": throughput_passed,
        "abstention_reason": abstention_reason,
        "passed": passed,
    }


def combined_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Keep one stable artifact-wide row order for tamper checks."""

    return [
        *payload.get("exact_authority_rows", []),
        *payload.get("matched_budget_rows", []),
        *payload.get("quality_rows", []),
        *payload.get("quality_summary_rows", []),
    ]


def _sample_size_budget() -> JsonDict:
    """Freeze planned, attempted, completed, censored, and independent counts."""

    exact = len(CELLS) * len(GRAPH_SEEDS)
    matched = exact * len(CHAINS) * len(ARMS) * len(PROTOCOLS)
    quality = exact * len(CHAINS) * len(ARMS)
    return {
        "planned_exact_authority_rows": exact,
        "attempted_exact_authority_rows": 0,
        "completed_exact_authority_rows": 0,
        "censored_exact_authority_rows": 0,
        "planned_matched_budget_rows": matched,
        "attempted_matched_budget_rows": 0,
        "completed_matched_budget_rows": 0,
        "censored_matched_budget_rows": 0,
        "planned_quality_rows": quality,
        "attempted_quality_rows": 0,
        "completed_quality_rows": 0,
        "censored_quality_rows": 0,
        "independent_graph_count": len(GRAPH_SEEDS),
        "independent_chain_count_per_graph_cell_arm": len(CHAINS),
        "equal_work_energy_evaluations_per_chain": ENERGY_EVALUATION_BUDGET,
        "equal_wall_seconds_per_chain": WALL_BUDGET_S,
        "quality_burn_in_per_chain": QUALITY_BURN_IN,
        "quality_retained_per_chain": QUALITY_RETAINED,
        "quality_measurement_cap_s": MEASUREMENT_CAP_S,
    }


def _base_artifact(root: Path, checks: list[JsonDict], hashes: dict[str, str]) -> JsonDict:
    """Create every required field before choosing a terminal finding."""

    return {
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": RUN_DATE,
        "preconditions_checked": checks,
        "inference_substrate": "not_started",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown-host",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": [],
        "sample_size_budget": _sample_size_budget(),
        "random_seed": {
            "graph_seeds": list(GRAPH_SEEDS),
            "chain_ids": list(CHAINS),
            "domain_separation": "sha256(7216:graph:n:beta:chain:arm:protocol) first 64 bits",
            "arm_order": "separate domain-separated shuffled stream",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "identical_explicit_rng_tapes_used": False,
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "down_up_comparison_complete_score": 0,
        "down_up_value_score": 0,
        "acceptance_gate_quality": {
            "primary_cell": dict(PRIMARY_CELL),
            "complete_panels_required": True,
            "zero_sector_violations_required": True,
            "finite_law_checks_required": True,
            "minimum_ess_per_chain": ESS_MINIMUM,
            "maximum_split_rhat": SPLIT_RHAT_MAXIMUM,
            "maximum_absolute_occupancy_mean_error": OCCUPANCY_ERROR_MAXIMUM,
            "maximum_standardized_energy_mean_error": ENERGY_STANDARDIZED_ERROR_MAXIMUM,
            "ess_per_second_ratio_ci95_lower_strictly_greater_than": 1.0,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "both_arms_exact_mean_tolerances_required_before_throughput": True,
        },
        "quality_rows": [],
        "matched_budget_rows": [],
        "exact_authority_rows": [],
        "quality_summary_rows": [],
        "trace_archive": {},
        "paper_replication_claimed": False,
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_invoked": False,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "root": str(root.resolve()),
        "nfr_01_10x_met": False,
        "upstream_performance_null": {
            "artifact": str(V634_PATH),
            "field": "nfr_01_10x_met",
            "observed_value": False,
            "promoted": False,
        },
        "paper_replication_claimed": False,
        "rust_10x_speed_claimed": False,
        "tsu_execution_claimed": False,
        "sparse_sk_theorem_claimed": False,
        "hardware_power_savings_claimed": False,
        "hamiltonian_contract": {
            "energy": slices.EDGE_COUNTING_CONVENTION,
            "cardinality": CARDINALITY,
            "graph_factory": "experiment_7187_v633_slice_sampler.make_frustrated_instance",
            "nonzero_fields_required": True,
            "frustrated_triangle_required": True,
            "fields_tuned_after_mixing": False,
        },
        "primary_gate": {},
        "spec_refs": [
            "REQ-ISING-7216",
            "SCENARIO-ISING-7216-PREFLIGHT",
            "SCENARIO-ISING-7216-LAW-TRACE",
            "SCENARIO-ISING-7216-QUALITY",
            "SCENARIO-ISING-7216-MATCHED-BUDGETS",
            "SCENARIO-ISING-7216-GATE",
            "SCENARIO-ISING-7216-ARTIFACT",
        ],
    }


def _blocked_artifact(artifact: JsonDict, failed: Mapping[str, Any], started: float) -> JsonDict:
    """Publish one terminal external block without invented sampler rows."""

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
                f"blocked_external_precondition: {failed.get('check')} failed before CPU work"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _checkpoint(root: Path, phase: int, artifact: Mapping[str, Any]) -> None:
    """Write bounded progress evidence only beneath the checkpoint directory."""

    path = root / CHECKPOINT_DIR / f"experiment_7216_phase_{phase}.json"
    payload = {
        "task_id": TASK_ID,
        "phase": phase,
        "status": artifact.get("status"),
        "duration_s": artifact.get("duration_s"),
        "sample_size_budget": artifact.get("sample_size_budget"),
    }
    atomic_write(path, payload)


def build_artifact(
    *,
    root: Path,
    output: Path,
    trace_output: Path,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
) -> JsonDict:
    """Build complete measured evidence or stop at an exact external prerequisite."""

    started = time.monotonic()
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None:
        measured_checks, measured_hashes = collect_preconditions(
            root, result_path=output, checkpoint_dir=trace_output.parent
        )
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
    else:
        checks, hashes = preconditions, source_hashes
    artifact = _base_artifact(root, checks, hashes)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    _progress(0, "end", "precondition checks")
    if failed is not None:
        return _blocked_artifact(artifact, failed, started)
    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": str(UPSTREAM_PATH),
        "field": "down_up_kernel_ready_score",
        "expected_value": 1,
        "observed_value": 1,
    }

    _progress(1, "start", "independent exact target laws")
    exact_rows, authorities = run_exact_panel()
    artifact["exact_authority_rows"] = exact_rows
    artifact["sample_size_budget"].update(
        {
            "attempted_exact_authority_rows": len(exact_rows),
            "completed_exact_authority_rows": len(exact_rows),
        }
    )
    artifact["duration_s"] = time.monotonic() - started
    _checkpoint(root, 1, artifact)
    _progress(1, "end", f"independent exact target laws rows={len(exact_rows)}")

    archive = TraceArchiveWriter(trace_output)
    archive.open()
    try:
        _progress(2, "start", "matched energy and wall budget panels")
        matched_rows = run_matched_panels(authorities, archive)
        artifact["matched_budget_rows"] = matched_rows
        artifact["sample_size_budget"].update(
            {
                "attempted_matched_budget_rows": len(matched_rows),
                "completed_matched_budget_rows": len(matched_rows),
            }
        )
        artifact["duration_s"] = time.monotonic() - started
        _checkpoint(root, 2, artifact)
        _progress(2, "end", f"matched budget rows={len(matched_rows)}")

        _progress(3, "start", "long-chain quality qualification panel")
        quality_rows, summaries, truncated = run_quality_panels(authorities, archive)
        artifact["quality_rows"] = quality_rows
        artifact["quality_summary_rows"] = summaries
        planned_quality = artifact["sample_size_budget"]["planned_quality_rows"]
        completed_quality = sum(row.get("complete") is True for row in quality_rows)
        artifact["sample_size_budget"].update(
            {
                "attempted_quality_rows": len(quality_rows),
                "completed_quality_rows": completed_quality,
                "censored_quality_rows": planned_quality - completed_quality,
            }
        )
        artifact["duration_s"] = time.monotonic() - started
        _checkpoint(root, 3, artifact)
        _progress(
            3,
            "end",
            f"quality rows={len(quality_rows)} truncated={truncated}",
        )

        _progress(4, "start", "compressed raw trace finalization")
        trace_receipt = archive.close()
        try:
            trace_receipt["path"] = str(trace_output.relative_to(root))
        except ValueError:
            trace_receipt["path"] = str(trace_output)
        artifact["trace_archive"] = trace_receipt
        _progress(4, "end", f"trace records={trace_receipt['record_count']}")
    except BaseException:
        archive.abort()
        raise

    _progress(5, "start", "quality-first primary gate")
    budget = artifact["sample_size_budget"]
    expected_summaries = len(CELLS) * len(GRAPH_SEEDS) * len(ARMS)
    complete = (
        len(exact_rows) == budget["planned_exact_authority_rows"]
        and all(row["finite_law_passed"] for row in exact_rows)
        and len(matched_rows) == budget["planned_matched_budget_rows"]
        and all(row["sector_violations"] == 0 for row in matched_rows)
        and completed_quality == planned_quality
        and len(summaries) == expected_summaries
        and not truncated
    )
    primary_gate = evaluate_primary_gate(summaries, exact_rows, panel_complete=complete)
    value = primary_gate["passed"] is True
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "host CPU exact finite-slice enumeration plus independently seeded Python "
                "down-up conditional and pair-swap Metropolis simulation"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "down_up_comparison_complete_score": int(complete),
            "down_up_value_score": int(value),
            "primary_gate": primary_gate,
            "verdict_class": "circular_positive" if value else "null",
            "honest_verdict": (
                "complete_circular_positive: both Python kernels met the frozen exact-law "
                "quality clauses and the paired down-up minimum-probe ESS-per-second interval "
                "was strictly above one; this remains circular finite-authority evidence."
                if value
                else (
                    "complete_null: the measurement cap truncated the prespecified quality "
                    "panel, so comparator quality and kernel value are insufficient."
                    if truncated
                    else "complete_null: all prespecified panels were measured, but the joint "
                    "exact-fidelity and cost-adjusted down-up quality gate did not pass."
                )
            ),
            "duration_s": time.monotonic() - started,
        }
    )
    artifact["rows"] = combined_rows(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _checkpoint(root, 5, artifact)
    _progress(
        5,
        "end",
        f"complete_score={int(complete)} value_score={int(value)}",
    )
    return artifact


def _row_hashes_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Verify common fields and the digest of every retained comparison row."""

    return all(
        all(key in row for key in ("unit_id", "arm", "seed", "metric", "error", "abstention"))
        and row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        for row in rows
    )


def validate_artifact(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute roster coverage, gate ordering, claims, and all evidence hashes."""

    if not REQUIRED_ARTIFACT_FIELDS.issubset(payload):
        return ["missing_required_fields"]
    errors: list[str] = []
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if payload.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if payload.get("MODEL_SPECS") != [] or payload.get("model_invoked") is not False:
        errors.append("model_declaration_invalid")
    if payload.get("execution_venue") != "host" or not payload.get("execution_host"):
        errors.append("execution_identity_invalid")
    duration = payload.get("duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        errors.append("duration_invalid")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")

    if payload.get("verdict_class") == "blocked":
        gate = payload.get("gate_check_summary", {})
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate") != "blocked_no_run"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("down_up_comparison_complete_score") != 0
            or payload.get("down_up_value_score") != 0
            or combined_rows(payload)
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not all(
                gate.get(key) is not None
                for key in (
                    "failed_check",
                    "upstream",
                    "field",
                    "expected_value",
                    "observed_value",
                )
            )
        ):
            errors.append("blocked_contract_invalid")
        return list(dict.fromkeys(errors))

    exact_rows = payload.get("exact_authority_rows", [])
    matched_rows = payload.get("matched_budget_rows", [])
    quality_rows = payload.get("quality_rows", [])
    summaries = payload.get("quality_summary_rows", [])
    all_rows = combined_rows(payload)
    if payload.get("rows") != all_rows or not _row_hashes_valid(all_rows):
        errors.append("rows_invalid")

    expected_exact = {
        f"n{n}:k{CARDINALITY}:beta{beta:g}:graph{seed}:exact"
        for n, beta in CELLS
        for seed in GRAPH_SEEDS
    }
    exact_valid = (
        len(exact_rows) == len(expected_exact)
        and {row.get("unit_id") for row in exact_rows} == expected_exact
        and all(
            row.get("state_count") == math.comb(int(row["n"]), CARDINALITY)
            and row.get("normalization_error", math.inf) <= EXACT_TOLERANCE
            and row.get("minimum_probability", 0.0) > 0.0
            and row.get("energy_parity_error", math.inf) <= EXACT_TOLERANCE
            and row.get("finite_law_passed") is True
            and row.get("enumerated_before_sampling") is True
            for row in exact_rows
        )
    )
    if not exact_valid:
        errors.append("exact_authority_rows_invalid")

    expected_matched = {
        (f"n{n}:k{CARDINALITY}:beta{beta:g}:graph{graph_seed}:chain{chain_id}:{arm}:{protocol}")
        for n, beta in CELLS
        for graph_seed in GRAPH_SEEDS
        for chain_id in CHAINS
        for arm in ARMS
        for protocol in PROTOCOLS
    }
    matched_valid = (
        len(matched_rows) == len(expected_matched)
        and {row.get("unit_id") for row in matched_rows} == expected_matched
        and all(
            row.get("sector_violations") == 0
            and row.get("initialization_charged") is True
            and row.get("normalization_charged") is True
            and row.get("target_energy_calls_charged") is True
            and row.get("energy_evaluations", 0) > 0
            and row.get("latency_s", 0.0) > 0.0
            and row.get("normalizations")
            == (row.get("completed_transitions") if row.get("arm") == "down_up" else 0)
            and (
                row.get("energy_evaluations") == ENERGY_EVALUATION_BUDGET
                if row.get("protocol") == "equal_work"
                else row.get("wall_budget_s") == WALL_BUDGET_S
                and row.get("wall_budget_overshoot_s", -1.0) >= 0.0
            )
            for row in matched_rows
        )
    )
    if not matched_valid:
        errors.append("matched_budget_rows_invalid")

    expected_quality = {
        (f"n{n}:k{CARDINALITY}:beta{beta:g}:graph{graph_seed}:chain{chain_id}:{arm}:quality")
        for n, beta in CELLS
        for graph_seed in GRAPH_SEEDS
        for chain_id in CHAINS
        for arm in ARMS
    }
    quality_ids = {row.get("unit_id") for row in quality_rows}
    quality_shape_valid = quality_ids.issubset(expected_quality) and len(quality_ids) == len(
        quality_rows
    )
    quality_content_valid = all(
        row.get("burn_in") == QUALITY_BURN_IN
        and 0 <= row.get("retained", -1) <= QUALITY_RETAINED
        and row.get("sector_violations") == 0
        and set(row.get("probe_diagnostics", {}))
        == {
            "energy",
            f"occupancy_0",
            f"occupancy_{int(row['n']) // 3}",
            f"occupancy_{2 * int(row['n']) // 3}",
        }
        and row.get("identical_rng_tape_parity_claimed") is False
        for row in quality_rows
    )
    if not quality_shape_valid or not quality_content_valid:
        errors.append("quality_rows_invalid")

    expected_summary = {
        f"n{n}:k{CARDINALITY}:beta{beta:g}:graph{seed}:{arm}:summary"
        for n, beta in CELLS
        for seed in GRAPH_SEEDS
        for arm in ARMS
    }
    summary_ids = {row.get("unit_id") for row in summaries}
    if not summary_ids.issubset(expected_summary) or len(summary_ids) != len(summaries):
        errors.append("quality_summary_rows_invalid")

    budget = payload.get("sample_size_budget", {})
    planned_exact = len(expected_exact)
    planned_matched = len(expected_matched)
    planned_quality = len(expected_quality)
    complete_quality = sum(row.get("complete") is True for row in quality_rows)
    budget_valid = (
        isinstance(budget, Mapping)
        and budget.get("planned_exact_authority_rows") == planned_exact
        and budget.get("attempted_exact_authority_rows") == len(exact_rows)
        and budget.get("completed_exact_authority_rows") == len(exact_rows)
        and budget.get("planned_matched_budget_rows") == planned_matched
        and budget.get("attempted_matched_budget_rows") == len(matched_rows)
        and budget.get("completed_matched_budget_rows") == len(matched_rows)
        and budget.get("planned_quality_rows") == planned_quality
        and budget.get("attempted_quality_rows") == len(quality_rows)
        and budget.get("completed_quality_rows") == complete_quality
        and budget.get("censored_quality_rows") == planned_quality - complete_quality
        and budget.get("independent_graph_count") == len(GRAPH_SEEDS)
    )
    if not budget_valid:
        errors.append("sample_size_budget_invalid")

    complete = (
        exact_valid
        and matched_valid
        and quality_shape_valid
        and quality_content_valid
        and len(quality_rows) == planned_quality
        and complete_quality == planned_quality
        and summary_ids == expected_summary
    )
    primary = evaluate_primary_gate(summaries, exact_rows, panel_complete=complete)
    if payload.get("primary_gate") != primary:
        errors.append("primary_gate_invalid")
    if payload.get("down_up_comparison_complete_score") != int(complete):
        errors.append("completion_score_invalid")
    if payload.get("down_up_value_score") != int(primary["passed"] is True):
        errors.append("value_score_invalid")

    expected_verdict = "circular_positive" if primary["passed"] is True else "null"
    if (
        payload.get("status") != "complete"
        or payload.get("verdict_class") != expected_verdict
        or not str(payload.get("honest_verdict", "")).startswith("complete_")
        or payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("terminal_verdict_invalid")
    if payload.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("gate_check_summary_invalid")
    if payload.get("acceptance_gate_quality", {}).get("primary_cell") != PRIMARY_CELL:
        errors.append("acceptance_gate_quality_invalid")
    if (
        payload.get("nfr_01_10x_met") is not False
        or payload.get("upstream_performance_null", {}).get("promoted") is not False
    ):
        errors.append("v634_null_invalid")
    if any(
        payload.get(name) is not False
        for name in (
            "paper_replication_claimed",
            "rust_10x_speed_claimed",
            "tsu_execution_claimed",
            "sparse_sk_theorem_claimed",
            "hardware_power_savings_claimed",
        )
    ):
        errors.append("claim_limits_invalid")

    receipt = payload.get("trace_archive", {})
    expected_trace_records = len(matched_rows) + len(quality_rows)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("compressed") is not True
        or receipt.get("record_count") != expected_trace_records
        or not str(receipt.get("sha256", "")).startswith("sha256:")
    ):
        errors.append("trace_archive_invalid")
    if root is not None:
        recorded = payload.get("source_artifact_hashes", {})
        expected_sources = {str(path) for path in REQUIRED_SOURCE_PATHS}
        if (
            not isinstance(recorded, Mapping)
            or set(recorded) != expected_sources
            or any(
                not (root / path).is_file() or recorded[str(path)] != sha256_file(root / path)
                for path in REQUIRED_SOURCE_PATHS
            )
        ):
            errors.append("source_artifact_hashes_invalid")
        trace_path = Path(str(receipt.get("path", "")))
        trace_path = trace_path if trace_path.is_absolute() else root / trace_path
        if (
            not trace_path.is_file()
            or receipt.get("sha256") != sha256_file(trace_path)
            or receipt.get("bytes") != trace_path.stat().st_size
        ):
            errors.append("trace_archive_bytes_invalid")
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
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(*, root: Path, output: Path, trace_output: Path) -> JsonDict:
    """Build, validate, and atomically write the terminal Exp7216 artifact."""

    artifact = build_artifact(root=root, output=output, trace_output=trace_output)
    _progress(6, "start", "terminal artifact validation")
    errors = validate_artifact(artifact, root=root)
    _progress(6, "end", f"terminal artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7216 artifact: {errors}")
    _progress(7, "start", "final atomic artifact write")
    receipt = atomic_write(output, artifact)
    _progress(7, "end", f"final atomic artifact write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and optional read-only validation path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--trace-output", type=Path, default=TRACE_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quality study or validate caller-selected durable bytes read-only."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        _progress(6, "start", f"read-only validation path={args.validate}")
        try:
            decoded = json.loads(args.validate.read_text(encoding="utf-8"))
            payload = decoded if isinstance(decoded, Mapping) else {}
            errors = validate_artifact(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"validation_error: {exc}", flush=True)
            _progress(6, "end", "read-only validation errors=1")
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        _progress(6, "end", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    try:
        output = args.output if args.output.is_absolute() else root / args.output
        trace_output = (
            args.trace_output if args.trace_output.is_absolute() else root / args.trace_output
        )
        run_experiment(root=root, output=output, trace_output=trace_output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
