"""Exp 2916 THRML simulator parity against matched KV260 evidence.

Spec: REQ-HW-067, SCENARIO-HW-067.

This module compares a local THRML software sampler with the same n=64 sparse
Ising problem family already recovered for Exp 2912.  The key boundary is
honesty: this is a simulator-parity run. It does not touch Extropic TSU
hardware, the KV260 board, synthesis tools, bitstreams, or new latency probes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from carnot.hardware import kv260_same_basis_cpu_gibbs_baseline as exp2912


REPO_ROOT = Path(__file__).resolve().parents[3]
KV260_ARTIFACT_REL_PATH = Path(
    "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json"
)
CPU_ARTIFACT_REL_PATH = Path(
    "results/experiment_2912_kv260_same_basis_cpu_gibbs_baseline_v1.json"
)
OUTPUT_REL_PATH = Path("results/experiment_2916_thrml_kv260_sampler_parity_v1.json")

EXPERIMENT_ID = 2916
RUN_DATE = "20260523"
INFERENCE_SUBSTRATE = "simulator_parity"
DEFAULT_THRML_SAMPLE_COUNT_PER_SEED = 16
DEFAULT_THRML_N_WARMUP = 4
DEFAULT_THRML_STEPS_PER_SAMPLE = 1
DEFAULT_ENERGY_BIN_COUNT = 8
FALLBACK_N_SPINS = 16

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "thrml_kv260_parity_ready",
    "thrml_import_ok",
    "thrml_version",
    "matched_full_n64_basis",
    "fallback_subset_used",
    "random_seeds_used",
    "energy_distribution_summary",
    "cpu_vs_thrml_distance",
    "kv260_vs_thrml_summary",
    "no_tsu_hardware_claim",
    "inference_substrate",
    "duration_s",
    "run_date",
}

recover_problem_basis = exp2912.recover_problem_basis

ImportModule = Callable[[str], Any]
ThrmlSampler = Callable[["ThrmlIsingCase", int, int], np.ndarray]


@dataclass(frozen=True)
class ThrmlImportDetails:
    """Observed local THRML import state for the simulator run."""

    ok: bool
    version: str
    import_path: str | None
    error: str | None = None


@dataclass(frozen=True)
class ThrmlIsingCase:
    """THRML-compatible Ising case derived from one uploaded sparse basis.

    The edge weights intentionally carry half of each uploaded directed row
    entry.  Exp 2912's sparse energy is `-0.5 * sum_i,j J_ij s_i s_j`; THRML's
    Ising edge factor sums the edge list directly, so storing `0.5 * J_ij`
    preserves the same unscaled energy surface even when the upload is directed
    or contains asymmetric top-k sparse rows.
    """

    name: str
    seed: int
    n_spins: int
    beta: float
    biases: np.ndarray
    edge_indices: tuple[tuple[int, int], ...]
    edge_weights: np.ndarray
    source_topology_checksum: str
    source_coupling_checksum: str
    source_field_checksum: str

    @property
    def edge_count(self) -> int:
        """Return the number of edge factors passed to THRML."""

        return len(self.edge_indices)


class ThrmlBasisUnsupportedError(RuntimeError):
    """Raised when THRML cannot represent the full sparse upload."""


def _round_metric(value: float | None, digits: int = 12) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def probe_thrml_import(importer: ImportModule = importlib.import_module) -> ThrmlImportDetails:
    """Import THRML and record version/path provenance.

    Spec traces: REQ-HW-067.
    """

    try:
        thrml_module = importer("thrml")
    except BaseException as exc:
        return ThrmlImportDetails(
            ok=False,
            version="unavailable",
            import_path=None,
            error=f"{exc.__class__.__name__}: {exc}",
        )
    version = getattr(thrml_module, "__version__", None)
    if not version:
        try:
            version = importlib.metadata.version("thrml")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return ThrmlImportDetails(
        ok=True,
        version=str(version),
        import_path=getattr(thrml_module, "__file__", None),
        error=None,
    )


def sparse_upload_energy(problem: exp2912.SparseProblemBasis, state: np.ndarray) -> float:
    """Compute the Exp 2912 sparse-upload energy for a full or subset state.

    Spec traces: REQ-HW-067.
    """

    spin_vec = np.asarray(state, dtype=np.float64)
    n_spins = int(spin_vec.shape[0])
    adjacency = np.asarray(problem.adjacency[:n_spins], dtype=np.int64)
    couplings = np.asarray(problem.couplings_q88[:n_spins], dtype=np.float64) / 256.0
    fields = np.asarray(problem.h_q88[:n_spins], dtype=np.float64) / 256.0
    valid = (adjacency >= 0) & (adjacency < n_spins)
    safe_adjacency = np.where(valid, adjacency, 0)
    neighbor_state = spin_vec[safe_adjacency]
    pair_terms = couplings * spin_vec[:, None] * neighbor_state
    return float(-(fields @ spin_vec) - 0.5 * np.sum(pair_terms[valid]))


def thrml_case_from_sparse_basis(
    problem: exp2912.SparseProblemBasis, *, n_spins: int | None = None
) -> ThrmlIsingCase:
    """Convert an Exp 2912 sparse q8.8 upload into a THRML edge-list case.

    Spec traces: REQ-HW-067.
    """

    active_n = int(n_spins if n_spins is not None else problem.n_spins)
    if active_n <= 0 or active_n > int(problem.n_spins):
        raise ValueError("n_spins must be in the recovered basis range")

    biases = np.asarray(problem.h_q88[:active_n], dtype=np.float64) / 256.0
    edge_indices: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    adjacency = np.asarray(problem.adjacency[:active_n], dtype=np.int64)
    couplings = np.asarray(problem.couplings_q88[:active_n], dtype=np.float64) / 256.0
    for left in range(active_n):
        for col, right_raw in enumerate(adjacency[left]):
            right = int(right_raw)
            if right < 0 or right >= active_n or right == left:
                continue
            weight = 0.5 * float(couplings[left, col])
            if weight == 0.0:
                continue
            edge_indices.append((left, right))
            edge_weights.append(weight)

    return ThrmlIsingCase(
        name=f"exp2916_seed_{problem.seed}_n{active_n}_uploaded_sparse",
        seed=int(problem.seed),
        n_spins=active_n,
        beta=float(problem.beta_final_q88) / 256.0,
        biases=biases,
        edge_indices=tuple(edge_indices),
        edge_weights=np.asarray(edge_weights, dtype=np.float64),
        source_topology_checksum=problem.topology_checksum,
        source_coupling_checksum=problem.coupling_tensor_checksum,
        source_field_checksum=problem.field_tensor_checksum,
    )


def energy_for_spin_state(case: ThrmlIsingCase, state: np.ndarray) -> float:
    """Compute the unscaled Ising energy for a THRML-compatible case."""

    spin_vec = np.asarray(state, dtype=np.float64)
    if spin_vec.shape != (case.n_spins,):
        raise ValueError(f"state shape {spin_vec.shape} does not match n_spins={case.n_spins}")
    pair_energy = 0.0
    for (left, right), weight in zip(case.edge_indices, case.edge_weights, strict=True):
        pair_energy += float(weight) * float(spin_vec[left]) * float(spin_vec[right])
    return float(-(case.biases @ spin_vec) - pair_energy)


def _samples_to_energies(case: ThrmlIsingCase, samples: np.ndarray) -> np.ndarray:
    samples_bool = np.asarray(samples, dtype=bool)
    expected_shape = (samples_bool.shape[0], case.n_spins)
    if samples_bool.ndim != 2 or samples_bool.shape != expected_shape:
        raise ValueError(f"THRML sampler returned shape {samples_bool.shape}, expected (*, {case.n_spins})")
    spin_rows = np.where(samples_bool, 1, -1).astype(np.int8)
    return np.asarray([energy_for_spin_state(case, row) for row in spin_rows], dtype=np.float64)


def energy_distribution_summary(
    energies: Sequence[float] | np.ndarray, *, bin_count: int = DEFAULT_ENERGY_BIN_COUNT
) -> dict[str, Any]:
    """Summarize an energy distribution with stable histogram metadata."""

    values = np.asarray(list(energies), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty energy distribution")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper <= lower:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, int(bin_count) + 1)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "sample_count": int(values.size),
        "mean": _round_metric(float(np.mean(values))),
        "variance": _round_metric(float(np.var(values))),
        "min": _round_metric(float(np.min(values))),
        "max": _round_metric(float(np.max(values))),
        "histogram": {
            "bin_count": int(bin_count),
            "bin_edges": [_round_metric(float(edge)) for edge in edges],
            "counts": [int(value) for value in counts],
        },
    }


def histogram_distance(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
) -> float:
    """Return total-variation distance between common-bin energy histograms."""

    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    left_values = np.asarray(list(left), dtype=np.float64)
    right_values = np.asarray(list(right), dtype=np.float64)
    if left_values.size == 0 or right_values.size == 0:
        return 0.0
    combined = np.concatenate([left_values, right_values])
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if upper <= lower:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, int(bin_count) + 1)
    left_counts, _ = np.histogram(left_values, bins=edges)
    right_counts, _ = np.histogram(right_values, bins=edges)
    left_probs = left_counts.astype(np.float64) / float(np.sum(left_counts))
    right_probs = right_counts.astype(np.float64) / float(np.sum(right_counts))
    return round(float(0.5 * np.sum(np.abs(left_probs - right_probs))), 12)


def sample_thrml_case(  # pragma: no cover - exercised by the live deliverable command.
    case: ThrmlIsingCase,
    seed: int,
    n_samples: int,
) -> np.ndarray:
    """Sample a THRML Ising case through the installed software API."""

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    jax.config.update("jax_disable_jit", True)
    thrml = importlib.import_module("thrml")
    models = importlib.import_module("thrml.models")
    spin_node_cls = getattr(thrml, "SpinNode", None)
    ising_cls = getattr(models, "IsingEBM", None)
    program_cls = getattr(models, "IsingSamplingProgram", None)
    if spin_node_cls is None or ising_cls is None or program_cls is None:
        raise ThrmlBasisUnsupportedError("installed THRML lacks the Ising sampling API")

    nodes = [spin_node_cls() for _ in range(case.n_spins)]
    edges = [(nodes[left], nodes[right]) for left, right in case.edge_indices]
    try:
        model = ising_cls(
            nodes,
            edges,
            jnp.asarray(case.biases, dtype=jnp.float32),
            jnp.asarray(case.edge_weights, dtype=jnp.float32),
            jnp.asarray(case.beta, dtype=jnp.float32),
        )
        blocks = [thrml.Block([node]) for node in nodes]
        program = program_cls(model, blocks, [])
        schedule = thrml.SamplingSchedule(
            n_warmup=DEFAULT_THRML_N_WARMUP,
            n_samples=int(n_samples),
            steps_per_sample=DEFAULT_THRML_STEPS_PER_SAMPLE,
        )
        init_state = [jnp.asarray([idx % 2 == 0], dtype=bool) for idx in range(case.n_spins)]
        block_samples = thrml.sample_states(
            jrandom.PRNGKey(int(seed)),
            program,
            schedule,
            init_state,
            [],
            blocks,
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise ThrmlBasisUnsupportedError(f"installed THRML could not sample this basis: {exc}") from exc
    return np.concatenate([np.asarray(item, dtype=bool) for item in block_samples], axis=1)


def _final_energies_from_cpu(payload: Mapping[str, Any]) -> list[float]:
    rows = payload.get("cpu_per_seed_results", [])
    if not isinstance(rows, list):
        return []
    return [
        float(row["final_energy"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("final_energy"), (int, float))
    ]


def _final_energies_from_kv260(payload: Mapping[str, Any]) -> tuple[list[float], str]:
    rows = payload.get("sample_count_sweep_results", [])
    source = "sample_count_sweep_results.final_energy"
    if not isinstance(rows, list) or not rows:
        rows = payload.get("per_seed_results", [])
        source = "per_seed_results.final_energy"
    if not isinstance(rows, list):
        return [], source
    return (
        [
            float(row["final_energy"])
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("final_energy"), (int, float))
        ],
        source,
    )


def _summary_or_empty(values: Sequence[float]) -> dict[str, Any]:
    return energy_distribution_summary(values) if values else {}


def _sample_cases(
    cases: Sequence[ThrmlIsingCase],
    *,
    sampler: ThrmlSampler,
    sample_count_per_seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    all_energies: list[float] = []
    for case in cases:
        samples = sampler(case, int(case.seed), int(sample_count_per_seed))
        energies = _samples_to_energies(case, samples)
        all_energies.extend(float(value) for value in energies)
        rows.append(
            {
                "seed": int(case.seed),
                "n_spins": int(case.n_spins),
                "edge_count": int(case.edge_count),
                "sample_count": int(energies.size),
                "mean_energy": _round_metric(float(np.mean(energies))),
                "variance_energy": _round_metric(float(np.var(energies))),
                "min_energy": _round_metric(float(np.min(energies))),
                "max_energy": _round_metric(float(np.max(energies))),
            }
        )
    return rows, np.asarray(all_energies, dtype=np.float64)


def _blocked_artifact(
    *,
    verdict: str,
    duration_s: float,
    thrml_import: ThrmlImportDetails | None = None,
    random_seeds_used: Sequence[int] | None = None,
    basis_limitation: str = "",
) -> dict[str, Any]:
    import_details = thrml_import or ThrmlImportDetails(False, "unavailable", None, None)
    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "thrml_kv260_parity_ready": False,
        "thrml_import_ok": bool(import_details.ok),
        "thrml_version": str(import_details.version),
        "matched_full_n64_basis": False,
        "fallback_subset_used": False,
        "random_seeds_used": [int(seed) for seed in (random_seeds_used or [])],
        "energy_distribution_summary": {},
        "cpu_vs_thrml_distance": 0.0,
        "kv260_vs_thrml_summary": {},
        "no_tsu_hardware_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
        "thrml_import_path": import_details.import_path,
        "thrml_import_error": import_details.error,
        "basis_limitation": basis_limitation,
        "upstream_artifacts": {
            "kv260": KV260_ARTIFACT_REL_PATH.as_posix(),
            "cpu": CPU_ARTIFACT_REL_PATH.as_posix(),
        },
    }
    validate_artifact(artifact)
    return artifact


def _success_artifact(
    *,
    verdict: str,
    thrml_import: ThrmlImportDetails,
    random_seeds_used: Sequence[int],
    matched_full_n64_basis: bool,
    fallback_subset_used: bool,
    thrml_energies: np.ndarray,
    cpu_energies: Sequence[float],
    kv260_energies: Sequence[float],
    kv260_source: str,
    per_seed_thrml_results: Sequence[Mapping[str, Any]],
    sample_count_per_seed: int,
    duration_s: float,
    basis_limitation: str,
) -> dict[str, Any]:
    thrml_summary = energy_distribution_summary(thrml_energies)
    thrml_summary.update(
        {
            "n_spins": FALLBACK_N_SPINS if fallback_subset_used else exp2912.N_SPINS,
            "sample_count_per_seed": int(sample_count_per_seed),
        }
    )
    cpu_distance = histogram_distance(cpu_energies, thrml_energies)
    kv260_distance = histogram_distance(kv260_energies, thrml_energies)
    kv260_summary = {
        "evidence_source": kv260_source,
        "final_energy_summary": _summary_or_empty(list(kv260_energies)),
        "histogram_distance": kv260_distance,
        "mean_energy_delta_abs": _round_metric(
            abs(float(np.mean(kv260_energies)) - float(np.mean(thrml_energies)))
        )
        if kv260_energies
        else None,
        "min_energy_delta_abs": _round_metric(
            abs(float(np.min(kv260_energies)) - float(np.min(thrml_energies)))
        )
        if kv260_energies
        else None,
    }
    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "thrml_kv260_parity_ready": True,
        "thrml_import_ok": True,
        "thrml_version": thrml_import.version,
        "matched_full_n64_basis": bool(matched_full_n64_basis),
        "fallback_subset_used": bool(fallback_subset_used),
        "random_seeds_used": [int(seed) for seed in random_seeds_used],
        "energy_distribution_summary": {
            "thrml": thrml_summary,
            "cpu_final_energy": _summary_or_empty(list(cpu_energies)),
            "kv260_final_energy": _summary_or_empty(list(kv260_energies)),
        },
        "cpu_vs_thrml_distance": cpu_distance,
        "kv260_vs_thrml_summary": kv260_summary,
        "no_tsu_hardware_claim": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "run_date": RUN_DATE,
        "thrml_import_path": thrml_import.import_path,
        "basis_limitation": basis_limitation,
        "sample_count_per_seed": int(sample_count_per_seed),
        "thrml_schedule": {
            "n_warmup": DEFAULT_THRML_N_WARMUP,
            "steps_per_sample": DEFAULT_THRML_STEPS_PER_SAMPLE,
            "beta_source": "problem.beta_final_q88 / 256",
        },
        "per_seed_thrml_results": [dict(row) for row in per_seed_thrml_results],
        "upstream_artifacts": {
            "kv260": KV260_ARTIFACT_REL_PATH.as_posix(),
            "cpu": CPU_ARTIFACT_REL_PATH.as_posix(),
        },
        "hardware_claims_made": {
            "tsu": False,
            "z1": False,
            "extropic_hardware": False,
            "kv260_board": False,
            "synthesis": False,
            "bitstream": False,
            "new_latency": False,
        },
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the stable Exp 2916 terminal schema."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be simulator_parity")
    if artifact.get("run_date") != RUN_DATE:
        raise ValueError("run_date must be 20260523")
    duration = float(artifact.get("duration_s", -1.0))
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be a non-negative finite value")
    cpu_distance = float(artifact.get("cpu_vs_thrml_distance", -1.0))
    if not math.isfinite(cpu_distance) or cpu_distance < 0.0:
        raise ValueError("cpu_vs_thrml_distance must be non-negative and finite")
    if artifact.get("matched_full_n64_basis") is True and artifact.get("fallback_subset_used") is True:
        raise ValueError("full n64 and fallback subset cannot both be true")
    if artifact.get("thrml_kv260_parity_ready") is True:
        if artifact.get("thrml_import_ok") is not True:
            raise ValueError("ready parity requires thrml_import_ok")
        if not artifact.get("random_seeds_used"):
            raise ValueError("ready parity requires random_seeds_used")
        summary = artifact.get("energy_distribution_summary")
        if not isinstance(summary, Mapping) or not summary.get("thrml"):
            raise ValueError("ready parity requires THRML energy summary")


def run_experiment(
    root_path: Path = REPO_ROOT,
    *,
    importer: ImportModule = importlib.import_module,
    sampler: ThrmlSampler = sample_thrml_case,
    sample_count_per_seed: int = DEFAULT_THRML_SAMPLE_COUNT_PER_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Run Exp 2916 and write the deliverable artifact.

    Spec traces: REQ-HW-067, SCENARIO-HW-067.
    """

    started = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    cpu_payload = _read_json(root_path / CPU_ARTIFACT_REL_PATH)
    if cpu_payload.get("same_basis_cpu_baseline_ready") is not True:
        artifact = _blocked_artifact(
            verdict="blocked_cpu_baseline_not_ready",
            duration_s=_duration(started, now_s),
            basis_limitation="Exp 2912 same_basis_cpu_baseline_ready is not true.",
        )
        _write_json(output_path, artifact)
        return artifact

    thrml_import = probe_thrml_import(importer)
    if not thrml_import.ok:
        artifact = _blocked_artifact(
            verdict="blocked_thrml_import_unavailable",
            duration_s=_duration(started, now_s),
            thrml_import=thrml_import,
            random_seeds_used=cpu_payload.get("random_seeds_used", []),
            basis_limitation="Installed THRML could not be imported.",
        )
        _write_json(output_path, artifact)
        return artifact

    kv260_payload = _read_json(root_path / KV260_ARTIFACT_REL_PATH)
    try:
        basis = exp2912.recover_problem_basis(kv260_payload)
    except (exp2912.ProblemBasisUnrecoverableError, ValueError) as exc:
        artifact = _blocked_artifact(
            verdict="blocked_kv260_problem_basis_unrecoverable",
            duration_s=_duration(started, now_s),
            thrml_import=thrml_import,
            random_seeds_used=cpu_payload.get("random_seeds_used", []),
            basis_limitation=f"Unable to recover Exp 2898 sparse basis: {exc}",
        )
        _write_json(output_path, artifact)
        return artifact

    cpu_seed_set = set(int(seed) for seed in cpu_payload.get("random_seeds_used", []))
    problems = [problem for problem in basis.problems if int(problem.seed) in cpu_seed_set]
    if not problems:
        problems = list(basis.problems)
    full_cases = [thrml_case_from_sparse_basis(problem) for problem in problems]

    fallback_used = False
    matched_full = True
    basis_limitation = ""
    try:
        per_seed_rows, thrml_energies = _sample_cases(
            full_cases,
            sampler=sampler,
            sample_count_per_seed=sample_count_per_seed,
        )
    except ThrmlBasisUnsupportedError as exc:
        fallback_used = True
        matched_full = False
        basis_limitation = (
            "Installed THRML could not represent the full n=64 sparse upload; "
            f"using first {FALLBACK_N_SPINS} spins with in-subset uploaded edges. {exc}"
        )
        fallback_cases = [
            thrml_case_from_sparse_basis(problem, n_spins=FALLBACK_N_SPINS) for problem in problems
        ]
        try:
            per_seed_rows, thrml_energies = _sample_cases(
                fallback_cases,
                sampler=sampler,
                sample_count_per_seed=sample_count_per_seed,
            )
        except ThrmlBasisUnsupportedError as fallback_exc:
            artifact = _blocked_artifact(
                verdict="blocked_thrml_basis_unsupported",
                duration_s=_duration(started, now_s),
                thrml_import=thrml_import,
                random_seeds_used=[problem.seed for problem in problems],
                basis_limitation=f"{basis_limitation}; fallback also failed: {fallback_exc}",
            )
            _write_json(output_path, artifact)
            return artifact
    except BaseException as exc:
        artifact = _blocked_artifact(
            verdict="blocked_thrml_sampling_failed",
            duration_s=_duration(started, now_s),
            thrml_import=thrml_import,
            random_seeds_used=[problem.seed for problem in problems],
            basis_limitation=f"THRML sampling failed: {exc.__class__.__name__}: {exc}",
        )
        _write_json(output_path, artifact)
        return artifact

    cpu_energies = _final_energies_from_cpu(cpu_payload)
    kv260_energies, kv260_source = _final_energies_from_kv260(kv260_payload)
    artifact = _success_artifact(
        verdict=(
            "complete: thrml_kv260_n16_fallback_simulator_parity_ready_no_hardware_claim"
            if fallback_used
            else "complete: thrml_kv260_n64_simulator_parity_ready_no_hardware_claim"
        ),
        thrml_import=thrml_import,
        random_seeds_used=[problem.seed for problem in problems],
        matched_full_n64_basis=matched_full,
        fallback_subset_used=fallback_used,
        thrml_energies=thrml_energies,
        cpu_energies=cpu_energies,
        kv260_energies=kv260_energies,
        kv260_source=kv260_source,
        per_seed_thrml_results=per_seed_rows,
        sample_count_per_seed=sample_count_per_seed,
        duration_s=_duration(started, now_s),
        basis_limitation=basis_limitation,
    )
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
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
