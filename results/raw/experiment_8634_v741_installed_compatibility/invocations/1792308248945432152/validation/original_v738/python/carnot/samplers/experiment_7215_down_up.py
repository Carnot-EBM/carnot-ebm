"""Implement and certify a finite target-weighted down-up Ising kernel.

Algorithm 1 of arXiv:2609.08873v1 removes one member of a fixed-size
subset. It then draws a replacement from the target law conditioned on the
remaining subset. This module is opt-in. It does not change a shipped sampler
default, prove a mixing theorem, or measure hardware speed.

Spec: REQ-ISING-7215 and SCENARIO-ISING-7215-*.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import urllib.request

import numpy as np
import yaml

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7202_v634_slice_cost_quality as exp7202


JsonDict = dict[str, Any]
Subset = tuple[int, ...]

RUN_DATE = "20260911"
TASK_ID = "exp7215-down-up-prototype"
MILESTONE = "2026.09.635"
RESULT_PATH = Path("results/experiment_7215_v635_down_up_prototype.json")
CHECKPOINT_DIR = Path("results/checkpoints")
SPEC_PATH = Path("openspec/capabilities/ising-backend/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
UPSTREAM_PATH = Path("results/experiment_7202_v634_slice_cost_quality.json")

PAPER_VERSION = "arXiv:2609.08873v1"
PAPER_URL = "https://arxiv.org/html/2609.08873v1"
PAPER_HTML_SHA256 = "32a2302797b5245ae7921c4a3cf0eab6da86d77f2de87430c88839e1f694af19"
PAPER_LOCATIONS = ("Algorithm 1 in Section 3", "Remark 2 in Section 3.2")
PAPER_ANCHORS = (b'id="algorithm1"', b'id="Thmremark2"')
ENERGY_CONVENTION = slices.EDGE_COUNTING_CONVENTION

SIZES = (8,)
CARDINALITIES = (1, 2, 4)
BETAS = (0.0, 1.0, 2.0)
SEEDS = tuple(range(7215001, 7215011))
EMPIRICAL_DRAWS = 4096
TOLERANCE = 1.0e-10
MODEL_SPECS: list[JsonDict] = []

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": str(RESULT_PATH),
    "gated_on": None,
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
    Path("python/carnot/experiment_7187_v633_slice_sampler.py"),
    Path("python/carnot/experiment_7189_v633_rust_slice_parity.py"),
    Path("python/carnot/experiment_7202_v634_slice_cost_quality.py"),
    Path("python/carnot/samplers/backend.py"),
    Path("python/carnot/samplers/experiment_7215_down_up.py"),
    Path("scripts/experiments/experiment_7215_v635_down_up_prototype.py"),
    Path("tests/python/test_experiment_7215_v635_down_up_prototype.py"),
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = {
    "field_principles",
    "status",
    "run_date",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "down_up_kernel_ready_score",
    "transition_rows",
    "mutation_rows",
    "paper_assumption_rows",
    "kernel_cost_contract",
    "MODEL_SPECS",
    "model_invoked",
}

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
    "down_up_kernel_ready_score": (
        "One requires correct finite-law transitions and effective negative controls."
    ),
    "transition_rows": (
        "Each enumerated cell records stationarity and detailed-balance residuals."
    ),
    "mutation_rows": "Independent laws must reject deliberately incorrect transitions.",
    "paper_assumption_rows": (
        "The SK ensemble theorem does not automatically apply to arbitrary local graphs."
    ),
    "kernel_cost_contract": (
        "All replacement-candidate energies and normalization work are charged."
    ),
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
}


@dataclass(frozen=True)
class ExactSubsetLaw:
    """Store one ordered law whose energies use the independent scalar authority."""

    states: tuple[Subset, ...]
    energies: tuple[float, ...]
    probabilities: tuple[float, ...]


def _progress(phase: int, boundary: str, operation: str) -> None:
    """Flush a phase boundary so the outer process can observe real liveness."""

    print(f"[phase {phase} {boundary}] {operation}", flush=True)


def canonical_json(value: Any) -> str:
    """Encode stable finite JSON so hashes do not depend on dictionary order."""

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


def sha256_bytes(value: bytes) -> str:
    """Name the digest algorithm beside the digest of the complete byte string."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON instead of interpreter-specific object formatting."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a required file as bytes and fail if the file cannot be read."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Bind every terminal field except the digest that stores this calculation."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _finish_row(row: JsonDict) -> JsonDict:
    """Add the common row contract before binding the final row values."""

    row.setdefault("arm", "not_applicable")
    row.setdefault("seed", None)
    row.setdefault("metric", row.get("row_type", "measurement"))
    row.setdefault("error", None)
    row.setdefault("abstention", False)
    row["row_sha256"] = sha256_json(row)
    return row


def _validate_beta(beta: float) -> float:
    """Allow infinite-temperature beta zero but reject invalid numerical inputs."""

    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    return float(beta)


def enumerate_subsets(n: int, k: int) -> tuple[Subset, ...]:
    """Enumerate one stable fixed-cardinality state space, including boundaries."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    return tuple(itertools.combinations(range(n), k))


def _validated_subset(instance: slices.SliceInstance, state: Sequence[int]) -> Subset:
    """Reject ambiguous sets before they can alter a transition probability."""

    subset = tuple(state)
    if any(isinstance(site, bool) or not isinstance(site, int) for site in subset):
        raise ValueError("subset sites must be integers")
    if subset != tuple(sorted(set(subset))):
        raise ValueError("subset must contain sorted unique sites")
    if any(site < 0 or site >= instance.n for site in subset):
        raise ValueError("subset site lies outside the graph")
    return subset


def subset_to_spins(n: int, state: Sequence[int]) -> tuple[int, ...]:
    """Convert a subset to the spin convention used by the shared energy authority."""

    positive = set(state)
    return tuple(1 if site in positive else -1 for site in range(n))


def replacement_distribution(
    instance: slices.SliceInstance,
    core: Sequence[int],
    beta: float,
    *,
    sign: float = -1.0,
    excluded_site: int | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Normalize the full conditional with a log-sum-exp shift.

    The function evaluates every replacement state. This simple prototype
    therefore charges all candidate energies instead of claiming the paper's
    incremental preprocessing bound.
    """

    slices.validate_instance(instance)
    beta_value = _validate_beta(beta)
    fixed = _validated_subset(instance, core)
    candidates = tuple(
        site for site in range(instance.n) if site not in fixed and site != excluded_site
    )
    if not candidates:
        raise ValueError("replacement candidate set must not be empty")
    energies = tuple(
        slices.ising_energy(instance, subset_to_spins(instance.n, (*fixed, site)))
        for site in candidates
    )
    log_weights = tuple(sign * beta_value * energy for energy in energies)
    shift = max(log_weights)
    weights = tuple(math.exp(value - shift) for value in log_weights)
    total = math.fsum(weights)
    probabilities = tuple(value / total for value in weights)
    return candidates, probabilities, energies


def _uniform(value: float | None, rng: random.Random) -> float:
    """Use a caller tape value when present and validate the categorical domain."""

    sample = rng.random() if value is None else value
    if (
        isinstance(sample, bool)
        or not isinstance(sample, (int, float))
        or not math.isfinite(sample)
        or sample < 0.0
        or sample >= 1.0
    ):
        raise ValueError("uniform tape values must satisfy 0 <= value < 1")
    return float(sample)


def down_up_step(
    instance: slices.SliceInstance,
    state: Sequence[int],
    beta: float,
    *,
    down_uniform: float | None = None,
    up_uniform: float | None = None,
    rng: random.Random | None = None,
) -> Subset:
    """Run one elementary down-up transition with optional uniform tapes."""

    slices.validate_instance(instance)
    beta_value = _validate_beta(beta)
    source = _validated_subset(instance, state)
    if len(source) in (0, instance.n):
        return source
    stream = rng if rng is not None else random.Random()
    down_value = _uniform(down_uniform, stream)
    removed_index = min(int(down_value * len(source)), len(source) - 1)
    core = source[:removed_index] + source[removed_index + 1 :]
    candidates, probabilities, _ = replacement_distribution(instance, core, beta_value)
    up_value = _uniform(up_uniform, stream)
    cumulative = 0.0
    selected = candidates[-1]
    for candidate, probability in zip(candidates, probabilities, strict=True):
        cumulative += probability
        if up_value < cumulative:
            selected = candidate
            break
    return tuple(sorted((*core, selected)))


def independent_exact_law(instance: slices.SliceInstance, k: int, beta: float) -> ExactSubsetLaw:
    """Calculate the target law through Exp7187's independent scalar energy path."""

    slices.validate_instance(instance)
    beta_value = _validate_beta(beta)
    states = enumerate_subsets(instance.n, k)
    energies = tuple(
        slices.reference_energy(instance, subset_to_spins(instance.n, state)) for state in states
    )
    log_weights = tuple(-beta_value * energy for energy in energies)
    shift = max(log_weights)
    weights = tuple(math.exp(value - shift) for value in log_weights)
    total = math.fsum(weights)
    return ExactSubsetLaw(
        states=states,
        energies=energies,
        probabilities=tuple(weight / total for weight in weights),
    )


def transition_matrix(
    instance: slices.SliceInstance,
    k: int,
    beta: float,
    *,
    mutation: str | None = None,
) -> tuple[np.ndarray, tuple[Subset, ...]]:
    """Derive the complete matrix by summing every down-then-up path."""

    slices.validate_instance(instance)
    beta_value = _validate_beta(beta)
    states = enumerate_subsets(instance.n, k)
    matrix = np.zeros((len(states), len(states)), dtype=np.float64)
    if k in (0, instance.n):
        matrix[0, 0] = 1.0
        return matrix, states
    state_index = {state: index for index, state in enumerate(states)}
    for source_index, source in enumerate(states):
        for removed_index, removed in enumerate(source):
            core = source[:removed_index] + source[removed_index + 1 :]
            excluded = removed if mutation == "omit_removed_site" else None
            sign = 1.0 if mutation == "wrong_energy_sign" else -1.0
            candidates, probabilities, _ = replacement_distribution(
                instance,
                core,
                beta_value,
                sign=sign,
                excluded_site=excluded,
            )
            for candidate, probability in zip(candidates, probabilities, strict=True):
                target = tuple(sorted((*core, candidate)))
                matrix[source_index, state_index[target]] += probability / k
    if mutation == "drop_self_transitions":
        np.fill_diagonal(matrix, 0.0)
        row_sums = matrix.sum(axis=1)
        matrix = matrix / row_sums[:, None]
    elif mutation not in (None, "omit_removed_site", "wrong_energy_sign"):
        raise ValueError(f"unknown mutation: {mutation}")
    return matrix, states


def transition_diagnostics(
    law: ExactSubsetLaw,
    matrix: np.ndarray,
    states: Sequence[Subset],
) -> JsonDict:
    """Measure stochasticity, reversibility, and the independently scored target."""

    if tuple(states) != law.states or matrix.shape != (len(states), len(states)):
        raise ValueError("law, state order, and transition matrix must agree")
    probabilities = np.asarray(law.probabilities, dtype=np.float64)
    row_residual = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    minimum = float(np.min(matrix))
    flow = probabilities[:, None] * matrix
    detailed_balance = float(np.max(np.abs(flow - flow.T)))
    stationarity = float(np.max(np.abs(probabilities @ matrix - probabilities)))
    k = len(states[0])
    fixed = all(len(state) == k for state in states)
    passed = (
        row_residual <= TOLERANCE
        and minimum >= -TOLERANCE
        and fixed
        and detailed_balance <= TOLERANCE
        and stationarity <= TOLERANCE
    )
    return {
        "row_stochasticity_residual": row_residual,
        "minimum_probability": minimum,
        "fixed_cardinality": fixed,
        "detailed_balance_residual": detailed_balance,
        "stationarity_residual": stationarity,
        "passed": passed,
    }


def empirical_one_step_comparison(
    instance: slices.SliceInstance,
    source: Subset,
    beta: float,
    exact_row: Sequence[float],
    states: Sequence[Subset],
    *,
    seed: int,
    draws: int,
) -> JsonDict:
    """Compare actual sampled steps with one independently derived matrix row."""

    if draws < 1:
        raise ValueError("draws must be positive")
    stream = random.Random(seed)
    state_index = {state: index for index, state in enumerate(states)}
    counts = [0] * len(states)
    violations = 0
    for _ in range(draws):
        target = down_up_step(instance, source, beta, rng=stream)
        violations += int(len(target) != len(source))
        counts[state_index[target]] += 1
    empirical = tuple(count / draws for count in counts)
    exact = tuple(float(value) for value in exact_row)
    max_error = max(abs(observed - expected) for observed, expected in zip(empirical, exact))
    threshold = max(0.02, 5.0 * math.sqrt(0.25 / draws))
    return {
        "draws": draws,
        "counts": counts,
        "exact_probabilities": list(exact),
        "max_absolute_error": max_error,
        "acceptance_threshold": threshold,
        "cardinality_violations": violations,
        "passed": violations == 0 and max_error <= threshold,
    }


def _transition_unit_id(n: int, k: int, beta: float, seed: int) -> str:
    """Name each fixed roster cell without depending on loop position."""

    return f"n{n}:k{k}:beta{beta:g}:seed{seed}"


def run_transition_panel(
    *,
    seeds: Sequence[int] = SEEDS,
    cardinalities: Sequence[int] = CARDINALITIES,
    betas: Sequence[float] = BETAS,
    draws: int = EMPIRICAL_DRAWS,
) -> list[JsonDict]:
    """Retain every exact-law cell and one sampled transition row per cell."""

    rows: list[JsonDict] = []
    started = time.monotonic()
    total = len(seeds) * len(cardinalities) * len(betas)
    last_report = started
    for seed in seeds:
        instance = slices.make_frustrated_instance(8, seed)
        for k in cardinalities:
            for beta in betas:
                law = independent_exact_law(instance, k, beta)
                matrix, states = transition_matrix(instance, k, beta)
                diagnostics = transition_diagnostics(law, matrix, states)
                source_index = seed % len(states)
                empirical = empirical_one_step_comparison(
                    instance,
                    states[source_index],
                    beta,
                    matrix[source_index],
                    states,
                    seed=seed * 101 + k * 11 + int(beta * 10),
                    draws=draws,
                )
                rows.append(
                    _finish_row(
                        {
                            "unit_id": _transition_unit_id(8, k, beta, seed),
                            "row_type": "transition_law",
                            "arm": "down_up_exact_and_sampled",
                            "seed": seed,
                            "metric": "finite_law_certification",
                            "error": None,
                            "abstention": False,
                            "n": 8,
                            "k": k,
                            "beta": beta,
                            "state_count": len(states),
                            "instance_hash": instance.instance_hash,
                            "energy_authority": "Exp7187 reference_energy",
                            **diagnostics,
                            "empirical_source": list(states[source_index]),
                            "empirical_comparison": empirical,
                            "cell_passed": diagnostics["passed"] and empirical["passed"],
                        }
                    )
                )
                now = time.monotonic()
                if now - last_report >= 60.0:
                    print(
                        f"[phase 3 progress] completed={len(rows)}/{total} "
                        f"elapsed_s={now - started:.3f}",
                        flush=True,
                    )
                    last_report = now
    return rows


def run_mutation_checks() -> list[JsonDict]:
    """Apply all three deliberate defects to one nondegenerate exact fixture."""

    instance = slices.make_frustrated_instance(8, 7215099)
    law = independent_exact_law(instance, 2, 1.0)
    rows: list[JsonDict] = []
    for mutation in ("omit_removed_site", "wrong_energy_sign", "drop_self_transitions"):
        matrix, states = transition_matrix(instance, 2, 1.0, mutation=mutation)
        diagnostics = transition_diagnostics(law, matrix, states)
        rows.append(
            _finish_row(
                {
                    "unit_id": f"mutation:{mutation}",
                    "row_type": "mutation_control",
                    "arm": mutation,
                    "seed": 7215099,
                    "metric": "target_law_rejection",
                    "error": None,
                    "abstention": False,
                    "mutation": mutation,
                    **diagnostics,
                    "control_detected": not diagnostics["passed"],
                }
            )
        )
    return rows


def paper_assumption_rows() -> list[JsonDict]:
    """Keep the finite prototype separate from claims that its data cannot prove."""

    return [
        _finish_row(
            {
                "unit_id": "paper-assumption:kernel-only",
                "row_type": "paper_assumption",
                "arm": "scope_check",
                "metric": "claim_applicability",
                "claim": "finite arbitrary frustrated-graph checks reproduce the SK mixing theorem",
                "applies": False,
                "reason": "The paper theorem assumes a random SK ensemble and a stated parameter regime.",
            }
        ),
        _finish_row(
            {
                "unit_id": "paper-assumption:no-speed",
                "row_type": "paper_assumption",
                "arm": "scope_check",
                "metric": "claim_applicability",
                "claim": "the Python exact prototype establishes hardware speed",
                "applies": False,
                "reason": "This experiment executes only a host CPU kernel and exact enumeration.",
            }
        ),
    ]


def unwrap_principled_value(value: Any) -> Any:
    """Unwrap only the explicit two-key value and principle representation."""

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
    """Check artifact and manifest quarantine signals before reading a value."""

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
    """Reject a quarantined value before inspecting its structured contents."""

    if quarantine.get("quarantined") is True:
        return "not_consumed_due_to_quarantine"
    candidate = upstream[field] if field in upstream else upstream
    return unwrap_principled_value(candidate)


def _task_contract(root: Path) -> JsonDict | None:
    """Read the exact roadmap identity, gate, deliverable, and prior-null fields."""

    try:
        document = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
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


def _fetch_paper_bytes() -> bytes:
    """Fetch the immutable versioned HTML with a bounded official-source request."""

    request = urllib.request.Request(PAPER_URL, headers={"User-Agent": "Carnot/Exp7215"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _check_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Use one shape for every prerequisite and every blocked gate diagnostic."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def collect_preconditions(
    root: Path,
    *,
    paper_bytes: bytes | None = None,
    result_path: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:
    """Print and retain every required code, source, tool, and gate observation."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    result = result_path or root / RESULT_PATH
    checkpoints = checkpoint_dir or root / CHECKPOINT_DIR
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

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
    try:
        spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        spec_text = ""
    spec_present = "### REQ-ISING-7215" in spec_text and "SCENARIO-ISING-7215" in spec_text
    checks.append(
        _check_row(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-*",
            "REQ-ISING-7215 present",
            "REQ-ISING-7215 present" if spec_present else "missing",
            spec_present,
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

    announce("contextual upstream bytes and quarantine")
    try:
        decoded = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
        upstream = decoded if isinstance(decoded, Mapping) else {}
    except (OSError, json.JSONDecodeError):
        upstream = {}
    try:
        exclusion_text = (root / EXCLUSION_PATH).read_text(encoding="utf-8")
    except OSError:
        exclusion_text = ""
    manifest_match = any(
        token in exclusion_text
        for token in ("exp7202-slice-cost-quality", "experiment_7202_v634_slice_cost_quality")
    )
    quarantine = upstream_quarantine_observation(upstream, manifest_match=manifest_match)
    checks.append(
        _check_row(
            "upstream_quarantine",
            str(UPSTREAM_PATH),
            "artifact and exclusion-manifest flags",
            {"quarantined": False},
            quarantine,
            quarantine["quarantined"] is False,
        )
    )

    announce("contextual upstream producer authentication")
    producer_errors = exp7202.validate_artifact(upstream) if upstream else ["unreadable_artifact"]
    authentication = {
        "producer_valid": not producer_errors,
        "producer_errors": producer_errors,
        "required_for_authorization": False,
        "structured_values_consumed": False,
    }
    checks.append(
        _check_row(
            "upstream_authentication",
            str(UPSTREAM_PATH),
            "producer validator observation",
            "observe without using it as an execution gate",
            authentication,
            bool(upstream) and not quarantine["quarantined"],
        )
    )

    announce("known failed upstream values")
    failed_values = {
        "boundary_value_score": upstream.get("boundary_value_score"),
        "nfr_01_10x_met": upstream.get("nfr_01_10x_met"),
        "promoted": False,
    }
    checks.append(
        _check_row(
            "upstream_known_failed_values",
            str(UPSTREAM_PATH),
            "boundary_value_score,nfr_01_10x_met,promoted",
            {"boundary_value_score": 0, "nfr_01_10x_met": False, "promoted": False},
            failed_values,
            failed_values
            == {"boundary_value_score": 0, "nfr_01_10x_met": False, "promoted": False},
        )
    )

    announce("paper versioned source bytes and excerpt anchors")
    try:
        source_bytes = _fetch_paper_bytes() if paper_bytes is None else paper_bytes
        paper_hash = sha256_bytes(source_bytes)
        anchor_state = {anchor.decode("ascii"): anchor in source_bytes for anchor in PAPER_ANCHORS}
    except OSError as exc:
        source_bytes = b""
        paper_hash = f"unavailable:{exc}"
        anchor_state = {anchor.decode("ascii"): False for anchor in PAPER_ANCHORS}
    expected_paper_hash = "sha256:" + PAPER_HTML_SHA256
    paper_source = {
        "version": PAPER_VERSION,
        "url": PAPER_URL,
        "sha256": paper_hash,
        "bytes": len(source_bytes),
        "locations": list(PAPER_LOCATIONS),
        "anchors": anchor_state,
        "energy_convention": (
            "paper pi(x) is proportional to exp(beta*(quadratic plus field)); "
            "Carnot uses E=-quadratic-field, so weights are exp(-beta*E)"
        ),
    }
    if source_bytes:
        hashes[PAPER_URL] = paper_hash
    checks.append(
        _check_row(
            "paper_source",
            PAPER_URL,
            "version,sha256,Algorithm 1,Remark 2",
            {
                "version": PAPER_VERSION,
                "sha256": expected_paper_hash,
                "anchors": {anchor.decode("ascii"): True for anchor in PAPER_ANCHORS},
            },
            {
                "version": PAPER_VERSION,
                "sha256": paper_hash,
                "anchors": anchor_state,
            },
            paper_hash == expected_paper_hash and all(anchor_state.values()),
        )
    )

    announce("Python imports and validation tools")
    tool_state = {
        "python": sys.executable,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
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
            all(bool(value) for value in tool_state.values()),
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
    return checks, hashes, paper_source


def _sample_size_budget() -> JsonDict:
    """Freeze planned and actual count fields before any outcomes are inspected."""

    return {
        "planned_transition_cells": 90,
        "attempted_transition_cells": 0,
        "completed_transition_cells": 0,
        "censored_transition_cells": 0,
        "planned_mutation_cells": 3,
        "attempted_mutation_cells": 0,
        "completed_mutation_cells": 0,
        "censored_mutation_cells": 0,
        "independent_unit_count": len(SEEDS),
        "empirical_draws_per_transition_cell": EMPIRICAL_DRAWS,
        "planned_empirical_draws": 90 * EMPIRICAL_DRAWS,
        "attempted_empirical_draws": 0,
        "completed_empirical_draws": 0,
    }


def _base_artifact(
    root: Path,
    checks: list[JsonDict],
    hashes: dict[str, str],
    paper_source: Mapping[str, Any],
) -> JsonDict:
    """Create every required field before selecting a terminal outcome."""

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
            "graph_seeds": list(SEEDS),
            "empirical_stream": "seed*101+k*11+int(beta*10)",
            "mutation_fixture_seed": 7215099,
        },
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "running",
        "down_up_kernel_ready_score": 0,
        "transition_rows": [],
        "mutation_rows": [],
        "paper_assumption_rows": [],
        "kernel_cost_contract": {
            "nonboundary_candidate_count": "n-k+1 for each removed member",
            "down_choices_charged": "k uniform removal paths per exact matrix row",
            "candidate_energy_evaluations_charged": True,
            "normalization_charged": True,
            "normalization_method": "log-sum-exp shift followed by finite sum",
            "incremental_paper_runtime_claimed": False,
        },
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_invoked": False,
        "paper_source": dict(paper_source),
        "paper_replication_claimed": False,
        "general_mixing_theorem_claimed": False,
        "hardware_execution_claimed": False,
        "hardware_speed_claimed": False,
        "upstream_context": {
            "artifact": str(UPSTREAM_PATH),
            "used_as_execution_gate": False,
            "structured_values_consumed": False,
            "known_failed_values_promoted": False,
        },
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "spec_refs": [
            "REQ-ISING-7215",
            "SCENARIO-ISING-7215-KERNEL",
            "SCENARIO-ISING-7215-BOUNDARIES",
            "SCENARIO-ISING-7215-FINITE-LAW",
            "SCENARIO-ISING-7215-MUTATIONS",
            "SCENARIO-ISING-7215-ARTIFACT",
        ],
        "root": str(root.resolve()),
    }


def _blocked_artifact(artifact: JsonDict, failed: Mapping[str, Any], started: float) -> JsonDict:
    """Publish a diagnosed external block without inventing computation rows."""

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


def combined_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Keep a single stable row order across the detailed and common ledgers."""

    return [
        *payload.get("transition_rows", []),
        *payload.get("mutation_rows", []),
        *payload.get("paper_assumption_rows", []),
    ]


def build_artifact(
    root: Path,
    *,
    preconditions: list[JsonDict] | None = None,
    source_hashes: dict[str, str] | None = None,
    paper_source: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build complete finite evidence or one diagnosed external block."""

    started = time.monotonic()
    _progress(0, "start", "precondition checks")
    if preconditions is None or source_hashes is None or paper_source is None:
        measured_checks, measured_hashes, measured_paper = collect_preconditions(root)
        checks = measured_checks if preconditions is None else preconditions
        hashes = measured_hashes if source_hashes is None else source_hashes
        paper = measured_paper if paper_source is None else paper_source
    else:
        checks, hashes, paper = preconditions, source_hashes, paper_source
    artifact = _base_artifact(root, checks, hashes, paper)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    _progress(0, "end", "precondition checks")
    if failed is not None:
        return _blocked_artifact(artifact, failed, started)
    artifact["gate_check_summary"] = {
        "passed": True,
        "failed_check": None,
        "upstream": "required preconditions",
        "field": "all checks",
        "expected_value": True,
        "observed_value": True,
    }

    _progress(1, "start", "paper method and energy convention freeze")
    artifact["paper_source"] = dict(paper)
    _progress(1, "end", f"paper_sha256={paper.get('sha256')}")

    _progress(2, "start", "kernel boundary and input contract")
    boundary_instance = slices.make_frustrated_instance(8, SEEDS[0])
    boundary_passed = down_up_step(
        boundary_instance, (), 0.0, down_uniform=0.0, up_uniform=0.0
    ) == () and down_up_step(
        boundary_instance,
        tuple(range(8)),
        2.0,
        down_uniform=0.0,
        up_uniform=0.0,
    ) == tuple(range(8))
    _progress(2, "end", f"absorbing_boundaries={boundary_passed}")

    _progress(3, "start", "90-cell exact-law and sampled one-step panel")
    transition_rows = run_transition_panel()
    artifact["transition_rows"] = transition_rows
    artifact["sample_size_budget"].update(
        {
            "attempted_transition_cells": len(transition_rows),
            "completed_transition_cells": len(transition_rows),
            "attempted_empirical_draws": len(transition_rows) * EMPIRICAL_DRAWS,
            "completed_empirical_draws": len(transition_rows) * EMPIRICAL_DRAWS,
        }
    )
    _progress(3, "end", f"transition_cells={len(transition_rows)}")

    _progress(4, "start", "deliberately incorrect kernel controls")
    mutation_rows = run_mutation_checks()
    artifact["mutation_rows"] = mutation_rows
    artifact["sample_size_budget"].update(
        {
            "attempted_mutation_cells": len(mutation_rows),
            "completed_mutation_cells": len(mutation_rows),
        }
    )
    _progress(4, "end", f"mutation_cells={len(mutation_rows)}")

    _progress(5, "start", "claim limits and readiness aggregation")
    artifact["paper_assumption_rows"] = paper_assumption_rows()
    ready = (
        boundary_passed
        and len(transition_rows) == 90
        and all(row["cell_passed"] for row in transition_rows)
        and len(mutation_rows) == 3
        and all(row["control_detected"] for row in mutation_rows)
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": (
                "host CPU exact finite-state enumeration plus sampled replay of the "
                "target-weighted down-up kernel"
            ),
            "inference_substrate_class": "cpu_exact_solver_or_simulator",
            "duration_s": time.monotonic() - started,
            "verdict_class": "circular_positive" if ready else "null",
            "honest_verdict": (
                "complete_circular_positive: all 90 finite transition cells and all three "
                "mutation controls passed; this certifies the elementary CPU kernel only."
                if ready
                else "complete_null: the bounded finite kernel certification did not pass."
            ),
            "down_up_kernel_ready_score": int(ready),
        }
    )
    artifact["rows"] = combined_rows(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(5, "end", f"down_up_kernel_ready_score={int(ready)}")
    return artifact


def _row_hashes_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Verify common fields and the digest of every retained row."""

    return all(
        row.get("row_sha256")
        == sha256_json({key: value for key, value in row.items() if key != "row_sha256"})
        and all(key in row for key in ("unit_id", "arm", "seed", "metric", "error", "abstention"))
        and row.get("error") is None
        and row.get("abstention") is False
        for row in rows
    )


def _transition_rows_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Recompute exact matrices and empirical errors for the full frozen roster."""

    expected = {
        _transition_unit_id(8, k, beta, seed)
        for seed in SEEDS
        for k in CARDINALITIES
        for beta in BETAS
    }
    if len(rows) != 90 or {row.get("unit_id") for row in rows} != expected:
        return False
    for row in rows:
        seed = row.get("seed")
        k = row.get("k")
        beta = row.get("beta")
        if seed not in SEEDS or k not in CARDINALITIES or beta not in BETAS:
            return False
        instance = slices.make_frustrated_instance(8, seed)
        law = independent_exact_law(instance, k, beta)
        matrix, states = transition_matrix(instance, k, beta)
        diagnostic = transition_diagnostics(law, matrix, states)
        for field in (
            "row_stochasticity_residual",
            "minimum_probability",
            "detailed_balance_residual",
            "stationarity_residual",
        ):
            if not math.isclose(row.get(field, math.inf), diagnostic[field], abs_tol=1.0e-15):
                return False
        empirical = row.get("empirical_comparison")
        if not isinstance(empirical, Mapping):
            return False
        counts = empirical.get("counts")
        exact = empirical.get("exact_probabilities")
        draws = empirical.get("draws")
        if (
            not isinstance(counts, list)
            or not isinstance(exact, list)
            or draws != EMPIRICAL_DRAWS
            or len(counts) != len(states)
            or len(exact) != len(states)
            or sum(counts) != draws
        ):
            return False
        max_error = max(
            abs(count / draws - probability) for count, probability in zip(counts, exact)
        )
        if not math.isclose(
            empirical.get("max_absolute_error", math.inf), max_error, abs_tol=1.0e-15
        ):
            return False
        if row.get("cell_passed") is not (diagnostic["passed"] and empirical.get("passed") is True):
            return False
    return True


def validate_artifact(payload: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Recompute completeness, scientific gates, claim limits, and all hashes."""

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
        gate = payload.get("gate_check_summary")
        if (
            payload.get("status") != "blocked_external_precondition"
            or payload.get("inference_substrate") != "blocked_no_run"
            or payload.get("inference_substrate_class") != "blocked_no_run"
            or payload.get("down_up_kernel_ready_score") != 0
            or combined_rows(payload)
            or not isinstance(gate, Mapping)
            or gate.get("passed") is not False
            or not all(
                gate.get(key) is not None
                for key in ("failed_check", "upstream", "field", "expected_value", "observed_value")
            )
        ):
            errors.append("blocked_contract_invalid")
        return list(dict.fromkeys(errors))

    transitions = payload.get("transition_rows", [])
    mutations = payload.get("mutation_rows", [])
    assumptions = payload.get("paper_assumption_rows", [])
    if not _transition_rows_valid(transitions):
        errors.append("transition_rows_invalid")
    if (
        len(mutations) != 3
        or {row.get("mutation") for row in mutations}
        != {"omit_removed_site", "wrong_energy_sign", "drop_self_transitions"}
        or not all(
            row.get("control_detected") is True and row.get("passed") is False for row in mutations
        )
    ):
        errors.append("mutation_rows_invalid")
    if len(assumptions) != 2 or not all(row.get("applies") is False for row in assumptions):
        errors.append("paper_assumption_rows_invalid")
    all_rows = combined_rows(payload)
    if payload.get("rows") != all_rows or not _row_hashes_valid(all_rows):
        errors.append("rows_invalid")

    budget = payload.get("sample_size_budget", {})
    if (
        not isinstance(budget, Mapping)
        or budget.get("planned_transition_cells") != 90
        or budget.get("attempted_transition_cells") != 90
        or budget.get("completed_transition_cells") != 90
        or budget.get("censored_transition_cells") != 0
        or budget.get("planned_mutation_cells") != 3
        or budget.get("attempted_mutation_cells") != 3
        or budget.get("completed_mutation_cells") != 3
        or budget.get("censored_mutation_cells") != 0
        or budget.get("independent_unit_count") != 10
        or budget.get("completed_empirical_draws") != 90 * EMPIRICAL_DRAWS
    ):
        errors.append("sample_size_budget_invalid")

    transition_pass = "transition_rows_invalid" not in errors and all(
        row.get("cell_passed") is True for row in transitions
    )
    mutation_pass = "mutation_rows_invalid" not in errors
    ready = transition_pass and mutation_pass
    if payload.get("down_up_kernel_ready_score") != int(ready):
        errors.append("readiness_score_invalid")
    if (
        payload.get("status") != "complete"
        or payload.get("verdict_class") != ("circular_positive" if ready else "null")
        or not str(payload.get("honest_verdict", "")).startswith("complete_")
        or payload.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("terminal_verdict_invalid")
    if payload.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("gate_check_summary_invalid")
    if payload.get("upstream_context", {}).get("known_failed_values_promoted") is not False:
        errors.append("upstream_failed_value_promoted")
    if (
        payload.get("paper_replication_claimed") is not False
        or payload.get("general_mixing_theorem_claimed") is not False
        or payload.get("hardware_execution_claimed") is not False
        or payload.get("hardware_speed_claimed") is not False
    ):
        errors.append("claim_limits_invalid")
    paper = payload.get("paper_source", {})
    if (
        paper.get("version") != PAPER_VERSION
        or paper.get("url") != PAPER_URL
        or paper.get("sha256") != "sha256:" + PAPER_HTML_SHA256
        or paper.get("locations") != list(PAPER_LOCATIONS)
    ):
        errors.append("paper_source_invalid")
    cost = payload.get("kernel_cost_contract", {})
    if (
        cost.get("candidate_energy_evaluations_charged") is not True
        or cost.get("normalization_charged") is not True
        or cost.get("incremental_paper_runtime_claimed") is not False
    ):
        errors.append("kernel_cost_contract_invalid")
    if root is not None:
        recorded = payload.get("source_artifact_hashes", {})
        expected = {str(path) for path in REQUIRED_SOURCE_PATHS} | {PAPER_URL}
        if (
            not isinstance(recorded, Mapping)
            or set(recorded) != expected
            or any(
                not (root / path).is_file() or recorded[str(path)] != sha256_file(root / path)
                for path in REQUIRED_SOURCE_PATHS
            )
            or recorded.get(PAPER_URL) != "sha256:" + PAPER_HTML_SHA256
        ):
            errors.append("source_artifact_hashes_invalid")
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
        if temporary.exists():  # pragma: no cover - only an interrupted replace reaches this.
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "atomic_replace": True,
    }


def run_experiment(root: Path, output: Path) -> JsonDict:
    """Build, validate, and atomically publish the requested terminal artifact."""

    artifact = build_artifact(root)
    _progress(6, "start", "terminal artifact validation")
    errors = validate_artifact(artifact, root=root)
    _progress(6, "end", f"terminal artifact validation errors={len(errors)}")
    if errors:
        raise ValueError(f"invalid Exp7215 artifact: {errors}")
    _progress(7, "start", "final atomic artifact write")
    receipt = atomic_write(output, artifact)
    _progress(7, "end", f"final atomic artifact write bytes={receipt['bytes']}")
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed run date and optional read-only validation target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or validate caller-selected durable bytes without writes."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[3]
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
        run_experiment(root, output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"experiment_error: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script owns normal execution.
    raise SystemExit(main())
