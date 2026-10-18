"""Exp 1477 THRML/Carnot and NPIM-style Ising simulator microprobe.

This module is intentionally small and conservative. It checks whether THRML is
already importable, compares Carnot sampler output to exact CPU references on
tiny Ising systems, and runs a compact hand-fit momentum/schedule heuristic
inspired by NPIM update-rule learning. It never installs THRML and never treats
CPU/JAX simulation as Extropic TSU hardware access.

Spec traces: REQ-SAMPLE-042, SCENARIO-SAMPLE-070.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from carnot.analysis.pbit_sampler_portability import (
    IsingCase,
    enumerate_spin_states,
    exact_boltzmann_distribution,
    ising_energy,
    tiny_ising_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1477_thrml_npim_simulator_parity_microprobe.json"
)

EXPERIMENT_ID = 1477
SCHEMA = "thrml_npim_simulator_parity_microprobe_v1"
RUN_DATE = "20260507"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_available",
    "carnot_sampler_cases",
    "thrml_parity_cases",
    "parity_metric",
    "npim_probe_attempted",
    "npim_energy_delta",
    "npim_time_to_energy_delta",
    "hardware_claim_allowed",
    "simulator_only",
    "blockers",
    "honest_verdict",
}

HONEST_VERDICTS = {
    "in_progress",
    "complete_thrml_unavailable_npim_simulator_probe_recorded",
    "complete_thrml_api_blocked_npim_simulator_probe_recorded",
    "complete_thrml_parity_measured_npim_simulator_probe_recorded",
}

SampleFunc = Callable[
    [IsingCase],
    np.ndarray,
]


@dataclass(frozen=True)
class ThrmlImportProbe:
    """Result of probing the current Python environment for THRML.

    Keeping the import result as data makes tests and artifacts honest: an
    import failure is not silently converted into a simulated parity result,
    and an import success still has to pass the required Ising API checks.

    Spec traces: REQ-SAMPLE-042.
    """

    available: bool
    module: Any | None
    version: str | None
    missing_api_or_dependency: str | None


def _round_metric(value: float | None, digits: int = 12) -> float | None:
    """Round JSON metrics while preserving blocked values as null."""

    if value is None:
        return None
    return round(float(value), digits)


def _describe_exception(exc: BaseException) -> str:
    """Convert an import/API exception into a compact artifact blocker."""

    if isinstance(exc, ModuleNotFoundError):
        return f"missing Python module while importing THRML: {exc.name or 'unknown'}"
    text = str(exc).strip()
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__


def build_toy_ising_cases() -> tuple[IsingCase, ...]:
    """Return three tiny Ising cases small enough for exact enumeration.

    The cases mix ferromagnetic, signed/frustrated, and sparse biased structure
    so the probe is not a single hand-picked Hamiltonian. All cases use the
    same ``IsingCase`` energy convention as the p-bit portability packet.

    Spec traces: REQ-SAMPLE-042.
    """

    chain = IsingCase(
        name="n3_biased_chain",
        j_matrix=np.array(
            [
                [0.0, 0.70, 0.0],
                [0.70, 0.0, 0.35],
                [0.0, 0.35, 0.0],
            ],
            dtype=np.float64,
        ),
        bias=np.array([0.12, -0.04, 0.18], dtype=np.float64),
        beta=1.15,
    )
    sparse = IsingCase(
        name="n5_sparse_signed_bias",
        j_matrix=np.array(
            [
                [0.0, 0.45, 0.0, -0.30, 0.0],
                [0.45, 0.0, 0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.40, -0.20],
                [-0.30, 0.0, 0.40, 0.0, 0.15],
                [0.0, 0.0, -0.20, 0.15, 0.0],
            ],
            dtype=np.float64,
        ),
        bias=np.array([0.05, -0.10, 0.08, 0.02, -0.06], dtype=np.float64),
        beta=1.05,
    )
    return (chain, tiny_ising_case(), sparse)


def exact_case_reference(case: IsingCase) -> dict[str, Any]:
    """Compute exact enumerated energy reference data for a tiny Ising case.

    Spec traces: REQ-SAMPLE-042.
    """

    states = enumerate_spin_states(case.n_spins)
    energies = np.asarray([ising_energy(case, state) for state in states], dtype=np.float64)
    ground_index = int(np.argmin(energies))
    distribution = exact_boltzmann_distribution(case, states)
    exact_mean_energy = float(distribution @ energies)
    return {
        "state_count": int(states.shape[0]),
        "exact_ground_energy": _round_metric(float(energies[ground_index])),
        "exact_mean_energy": _round_metric(exact_mean_energy),
        "ground_state": [int(value) for value in states[ground_index]],
        "beta": float(case.beta),
    }


def _case_edges(case: IsingCase) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return nonzero upper-triangle Ising edges and weights."""

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(case.n_spins):
        for j in range(i + 1, case.n_spins):
            weight = float(case.j_matrix[i, j])
            if weight != 0.0:
                edges.append((i, j))
                weights.append(weight)
    return edges, np.asarray(weights, dtype=np.float64)


def _default_carnot_sample_func(
    case: IsingCase,
    *,
    seed: int,
    n_samples: int,
    n_warmup: int,
    steps_per_sample: int,
) -> np.ndarray:
    """Sample a tiny Ising case with Carnot's existing parallel sampler.

    ``ParallelIsingSampler`` operates on boolean spins but its probability rule
    can represent the same +/-1 Ising conditional by shifting the bias and
    doubling the pairwise couplings. The returned samples are converted back to
    +/-1 states so all energy accounting uses the exact reference convention.
    """

    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

    row_sum = case.j_matrix.sum(axis=1)
    boolean_bias = case.bias - row_sum
    boolean_couplings = 2.0 * case.j_matrix
    sampler = ParallelIsingSampler(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
        schedule=AnnealingSchedule(beta_init=0.2, beta_final=float(case.beta)),
        use_checkerboard=True,
    )
    samples_bool = sampler.sample(
        jrandom.PRNGKey(seed),
        jnp.asarray(boolean_bias, dtype=jnp.float32),
        jnp.asarray(boolean_couplings, dtype=jnp.float32),
        beta=float(case.beta),
    )
    return np.where(np.asarray(samples_bool, dtype=bool), 1, -1).astype(np.int8)


def _energy_vector(case: IsingCase, samples: np.ndarray) -> np.ndarray:
    """Compute one exact Ising energy per sampled +/-1 state."""

    return np.asarray([ising_energy(case, state) for state in samples], dtype=np.float64)


def run_carnot_sampler_cases(
    cases: Sequence[IsingCase],
    *,
    sample_func: Callable[..., np.ndarray] | None = None,
    seed: int = 1477,
    n_samples: int = 64,
    n_warmup: int = 96,
    steps_per_sample: int = 4,
) -> list[dict[str, Any]]:
    """Run Carnot sampler rows and compare them to exact CPU references.

    Spec traces: REQ-SAMPLE-042.
    """

    sampler = sample_func or _default_carnot_sample_func
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        started = time.perf_counter()
        samples = np.asarray(
            sampler(
                case,
                seed=seed + case_index,
                n_samples=n_samples,
                n_warmup=n_warmup,
                steps_per_sample=steps_per_sample,
            ),
            dtype=np.int8,
        )
        energies = _energy_vector(case, samples)
        reference = exact_case_reference(case)
        best_energy = float(np.min(energies))
        rows.append(
            {
                "case": case.name,
                "status": "sampled",
                "n_spins": case.n_spins,
                "sample_count": int(samples.shape[0]),
                "exact_reference_state_count": reference["state_count"],
                "exact_ground_energy": reference["exact_ground_energy"],
                "exact_mean_energy": reference["exact_mean_energy"],
                "carnot_best_energy": _round_metric(best_energy),
                "carnot_mean_energy": _round_metric(float(np.mean(energies))),
                "best_energy_gap_to_exact": _round_metric(
                    best_energy - float(reference["exact_ground_energy"])
                ),
                "wall_time_s": _round_metric(time.perf_counter() - started, digits=6),
            }
        )
    return rows


def probe_thrml_import(
    importer: Callable[[str], Any] = importlib.import_module,
) -> ThrmlImportProbe:
    """Probe local THRML import availability without installing anything.

    Spec traces: REQ-SAMPLE-042.
    """

    try:
        module = importer("thrml")
    except Exception as exc:
        return ThrmlImportProbe(
            available=False,
            module=None,
            version=None,
            missing_api_or_dependency=_describe_exception(exc),
        )
    return ThrmlImportProbe(
        available=True,
        module=module,
        version=str(getattr(module, "__version__", "unknown")),
        missing_api_or_dependency=None,
    )


def _require_thrml_api(module: Any) -> tuple[type, type, type]:
    """Return the minimal THRML objects needed for exact Ising energy parity."""

    missing: list[str] = []
    spin_node = getattr(module, "SpinNode", None)
    block = getattr(module, "Block", None)
    models = getattr(module, "models", None)
    ising_ebm = getattr(models, "IsingEBM", None) if models is not None else None
    if spin_node is None:
        missing.append("SpinNode")
    if block is None:
        missing.append("Block")
    if ising_ebm is None:
        missing.append("models.IsingEBM")
    if missing:
        raise AttributeError(f"THRML import lacks required Ising APIs: {', '.join(missing)}")
    return spin_node, block, ising_ebm


def _measure_thrml_case(case: IsingCase, module: Any) -> dict[str, Any]:
    """Measure exact local-vs-THRML energy parity for one tiny Ising case."""

    spin_node, block_cls, ising_ebm = _require_thrml_api(module)
    nodes = [spin_node() for _ in range(case.n_spins)]
    edge_indices, weights = _case_edges(case)
    edges = [(nodes[i], nodes[j]) for i, j in edge_indices]
    model = ising_ebm(
        nodes,
        edges,
        np.asarray(case.bias, dtype=np.float64),
        weights,
        1.0,
    )
    block = block_cls(nodes)
    states = enumerate_spin_states(case.n_spins)
    errors: list[float] = []
    for state in states:
        local_energy = ising_energy(case, state)
        bool_state = np.asarray(state == 1, dtype=bool)
        thrml_energy = float(model.energy([bool_state], [block]))
        errors.append(abs(local_energy - thrml_energy))
    return {
        "case": case.name,
        "status": "parity_measured",
        "n_spins": case.n_spins,
        "state_count": int(states.shape[0]),
        "edge_count": len(edge_indices),
        "max_abs_energy_error": _round_metric(max(errors) if errors else 0.0),
    }


def measure_thrml_parity_cases(
    cases: Sequence[IsingCase],
    probe: ThrmlImportProbe,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    """Attempt THRML parity only when the local import and API are available.

    Spec traces: REQ-SAMPLE-042, SCENARIO-SAMPLE-070.
    """

    if not probe.available or probe.module is None:
        reason = probe.missing_api_or_dependency or "THRML was not importable"
        return (
            [],
            {
                "metric": "max_abs_energy_error",
                "status": "blocked",
                "value": None,
                "reason": reason,
            },
            [{"blocker": "thrml_not_importable", "detail": reason}],
        )

    try:
        rows = [_measure_thrml_case(case, probe.module) for case in cases]
    except Exception as exc:
        reason = _describe_exception(exc)
        return (
            [],
            {
                "metric": "max_abs_energy_error",
                "status": "blocked",
                "value": None,
                "reason": reason,
            },
            [{"blocker": "thrml_api_incompatible", "detail": reason}],
        )

    max_error = max(float(row["max_abs_energy_error"]) for row in rows) if rows else 0.0
    return (
        rows,
        {
            "metric": "max_abs_energy_error",
            "status": "measured",
            "value": _round_metric(max_error),
            "reason": None,
        },
        [],
    )


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function for the tiny NPIM simulator."""

    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _run_update_policy(
    case: IsingCase,
    *,
    seed: int,
    momentum: float,
    beta_final: float,
    chains: int,
    steps: int,
) -> dict[str, Any]:
    """Run one stochastic momentum/schedule policy on a tiny Ising case."""

    rng = np.random.default_rng(seed)
    states = rng.choice(np.array([-1, 1], dtype=np.int8), size=(chains, case.n_spins))
    field_memory = np.zeros((chains, case.n_spins), dtype=np.float64)
    best_energy = float("inf")
    best_step = 0
    best_by_step: list[float] = []
    for step in range(steps):
        beta_t = 0.2 + (float(beta_final) - 0.2) * (step / max(steps - 1, 1))
        fields = states.astype(np.float64) @ case.j_matrix.T + case.bias
        field_memory = momentum * field_memory + (1.0 - momentum) * fields
        probabilities = _sigmoid_array(2.0 * beta_t * field_memory)
        states = np.where(rng.random(states.shape) < probabilities, 1, -1).astype(np.int8)
        energies = _energy_vector(case, states)
        step_best = float(np.min(energies))
        if step_best < best_energy:
            best_energy = step_best
            best_step = step
        best_by_step.append(best_energy)
    return {
        "momentum": float(momentum),
        "beta_final": float(beta_final),
        "best_energy": best_energy,
        "best_step": int(best_step),
        "best_by_step": best_by_step,
    }


def run_npim_probe(
    cases: Sequence[IsingCase],
    *,
    seed: int = 2602,
    chains: int = 16,
    steps: int = 24,
) -> dict[str, Any]:
    """Run a compact NPIM-style hand-fit schedule/update heuristic.

    The probe is "NPIM-style" rather than a trained NPIM model: it uses
    zeroth-order selection over a tiny grid of momentum and beta-schedule
    parameters, then reports energy and time-to-improvement deltas against a
    fixed local update baseline.

    Spec traces: REQ-SAMPLE-042.
    """

    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    improvement_steps: list[int] = []
    for case_index, case in enumerate(cases):
        baseline = _run_update_policy(
            case,
            seed=seed + 100 * case_index,
            momentum=0.0,
            beta_final=case.beta,
            chains=chains,
            steps=steps,
        )
        candidates = [
            baseline,
            _run_update_policy(
                case,
                seed=seed + 100 * case_index + 1,
                momentum=0.35,
                beta_final=case.beta * 1.6,
                chains=chains,
                steps=steps,
            ),
            _run_update_policy(
                case,
                seed=seed + 100 * case_index + 2,
                momentum=0.65,
                beta_final=case.beta * 2.2,
                chains=chains,
                steps=steps,
            ),
        ]
        best = min(candidates, key=lambda item: float(item["best_energy"]))
        delta = float(best["best_energy"] - baseline["best_energy"])
        first_improvement = None
        if delta < -1e-12:
            for step, energy in enumerate(best["best_by_step"]):
                if float(energy) < float(baseline["best_energy"]):
                    first_improvement = step
                    improvement_steps.append(step)
                    break
        deltas.append(delta)
        rows.append(
            {
                "case": case.name,
                "baseline_best_energy": _round_metric(float(baseline["best_energy"])),
                "npim_best_energy": _round_metric(float(best["best_energy"])),
                "energy_delta_vs_fixed_baseline": _round_metric(delta),
                "time_to_energy_delta_steps": first_improvement,
                "selected_policy": {
                    "momentum": _round_metric(float(best["momentum"])),
                    "beta_final": _round_metric(float(best["beta_final"])),
                },
            }
        )

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return {
        "attempted": True,
        "cases": rows,
        "energy_delta": {
            "metric": "mean_npim_best_energy_minus_fixed_baseline_best_energy",
            "value": _round_metric(mean_delta),
            "unit": "ising_energy",
            "negative_is_better": True,
        },
        "time_to_energy_delta": {
            "metric": "earliest_sweep_with_lower_energy_than_fixed_baseline",
            "value": min(improvement_steps) if improvement_steps else None,
            "unit": "sweeps",
            "lower_is_better": True,
        },
    }


def write_in_progress_artifact(path: str | Path = DELIVERABLE_PATH) -> dict[str, Any]:
    """Write the required Exp 1477 bootstrap artifact before probe completion."""

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
        },
        "status": "in_progress",
        "thrml_available": None,
        "carnot_sampler_cases": [],
        "thrml_parity_cases": [],
        "parity_metric": None,
        "npim_probe_attempted": False,
        "npim_energy_delta": None,
        "npim_time_to_energy_delta": None,
        "hardware_claim_allowed": False,
        "simulator_only": True,
        "blockers": [],
        "honest_verdict": "in_progress",
    }
    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required schema and no-hardware-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("hardware_claim_allowed") is not False:
        raise ValueError("hardware_claim_allowed must remain false for Exp 1477")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1477")
    status = artifact.get("status")
    if status == "in_progress":
        return
    if status != "complete":
        raise ValueError(f"invalid status: {status!r}")
    if not artifact.get("carnot_sampler_cases"):
        raise ValueError("carnot_sampler_cases must be non-empty in the terminal artifact")
    if not isinstance(artifact.get("parity_metric"), Mapping):
        raise ValueError("parity_metric must be an object in the terminal artifact")
    if artifact.get("npim_probe_attempted") is not True:
        raise ValueError("npim_probe_attempted must be true in the terminal artifact")
    if artifact.get("npim_energy_delta") is None:
        raise ValueError("npim_energy_delta is required in the terminal artifact")
    if artifact.get("npim_time_to_energy_delta") is None:
        raise ValueError("npim_time_to_energy_delta is required in the terminal artifact")
    if artifact.get("honest_verdict") not in HONEST_VERDICTS:
        raise ValueError(f"invalid honest_verdict: {artifact.get('honest_verdict')!r}")
    if artifact.get("thrml_available") is False and artifact.get("thrml_parity_cases"):
        raise ValueError("thrml_parity_cases must be empty when THRML is unavailable")


def _terminal_verdict(thrml_available: bool, blockers: Sequence[Mapping[str, str]]) -> str:
    """Choose the terminal honest verdict from THRML availability and blockers."""

    if not thrml_available:
        return "complete_thrml_unavailable_npim_simulator_probe_recorded"
    if blockers:
        return "complete_thrml_api_blocked_npim_simulator_probe_recorded"
    return "complete_thrml_parity_measured_npim_simulator_probe_recorded"


def write_artifact(
    artifact: Mapping[str, Any],
    path: str | Path = DELIVERABLE_PATH,
) -> dict[str, Any]:
    """Write a validated terminal Exp 1477 artifact."""

    payload = dict(artifact)
    validate_artifact(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_microprobe(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = RUN_DATE,
    probe_func: Callable[[], ThrmlImportProbe] = probe_thrml_import,
    sample_func: Callable[..., np.ndarray] | None = None,
    n_samples: int = 64,
    n_warmup: int = 96,
    steps_per_sample: int = 4,
    npim_chains: int = 16,
    npim_steps: int = 24,
) -> dict[str, Any]:
    """Run the complete simulator-only Exp 1477 microprobe and write JSON."""

    write_in_progress_artifact(output_path)
    cases = build_toy_ising_cases()
    probe = probe_func()
    carnot_rows = run_carnot_sampler_cases(
        cases,
        sample_func=sample_func,
        seed=1477,
        n_samples=n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
    )
    thrml_rows, parity_metric, blockers = measure_thrml_parity_cases(cases, probe)
    npim_summary = run_npim_probe(cases, seed=2602, chains=npim_chains, steps=npim_steps)
    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "thrml_version": probe.version,
            "tsu_hardware_execution": False,
            "npim_policy": "zeroth_order_hand_fit_momentum_schedule_simulator",
        },
        "status": "complete",
        "thrml_available": bool(probe.available),
        "carnot_sampler_cases": carnot_rows,
        "thrml_parity_cases": thrml_rows,
        "parity_metric": parity_metric,
        "npim_probe_attempted": bool(npim_summary["attempted"]),
        "npim_energy_delta": npim_summary["energy_delta"],
        "npim_time_to_energy_delta": npim_summary["time_to_energy_delta"],
        "npim_cases": npim_summary["cases"],
        "hardware_claim_allowed": False,
        "simulator_only": True,
        "blockers": blockers,
        "honest_verdict": _terminal_verdict(bool(probe.available), blockers),
    }
    return write_artifact(artifact, output_path)


def main() -> None:  # pragma: no cover
    """CLI entry point for ad hoc local execution."""

    run_microprobe()


if __name__ == "__main__":  # pragma: no cover
    main()
