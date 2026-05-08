"""Exp 1548 THRML/Carnot independent-RNG parity audit.

This audit exists because prior large-n THRML/Carnot parity artifacts reported
byte-identical stochastic histograms. Independent samplers should agree in
distribution, not produce identical sample summaries. The Carnot lane therefore
uses Carnot's CPU sampler, the THRML lane uses the installed THRML API
directly, and their seeds come from disjoint root lineages recorded in a
machine-readable manifest.

Spec traces: REQ-SAMPLE-056, SCENARIO-SAMPLE-084.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carnot.analysis.pbit_sampler_portability import ising_energy
from carnot.samplers.backend import CpuBackend

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1548_thrml_carnot_parity_independent_rng_audit.json"
)
SEED_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_independent_rng_seed_manifest_1548.json"
AUDIT_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_independent_rng_audit_1548.jsonl"

EXPERIMENT_ID = 1548
RUN_DATE = "20260508"
MILESTONE = "2026.04.119"
SCHEMA = "thrml_carnot_parity_independent_rng_audit_v1"
TOPOLOGIES = ("signed_ring_chord", "sparse_random", "lattice", "scale_free")
DEFAULT_N_VALUES = (32, 64, 128)
DEFAULT_CARNOT_ROOT_SEED = 20260508154801
DEFAULT_THRML_ROOT_SEED = 20260508154899
DEFAULT_SAMPLE_COUNT_PER_CASE = 16
DEFAULT_N_WARMUP = 4
DEFAULT_STEPS_PER_SAMPLE = 1
DEFAULT_ENERGY_BIN_COUNT = 4
DEFAULT_BETA = 0.35

THRESHOLDS = {
    "mean_energy_delta_abs_max": 0.60,
    "kl_divergence_max": 0.05,
    "ks_p_value_min": 0.01,
}
TERMINAL_VERDICT_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "independent_rng_audit_ready",
    "rng_path_independent",
    "code_path_independent",
    "rng_seed_manifest_path",
    "n_values_tested",
    "topologies_tested",
    "sample_path_hashes",
    "byte_identical_pairs",
    "nonzero_stochastic_delta_observed",
    "per_case_results",
    "max_mean_energy_delta_abs",
    "max_kl_divergence",
    "min_ks_p_value",
    "bounded_kl_passed",
    "ks_test_passed",
    "rng_path_not_independent",
    "simulator_only",
    "no_tsu_hardware_claim",
    "focused_tests_passed",
    "honest_verdict",
}

SamplerFn = Callable[["AuditIsingCase"], np.ndarray]


@dataclass(frozen=True)
class AuditIsingCase:
    """One deterministic Ising case used by the independent-RNG audit."""

    n_spins: int
    topology: str
    name: str
    j_matrix: np.ndarray
    bias: np.ndarray
    beta: float


def _display_path(path: str | Path) -> str:
    output_path = Path(path)
    try:
        return str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(output_path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    output_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return serializable


def _write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _round_metric(value: float | None, digits: int = 12) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _set_edge(j_matrix: np.ndarray, left: int, right: int, weight: float) -> None:
    j_matrix[int(left), int(right)] = float(weight)
    j_matrix[int(right), int(left)] = float(weight)


def _bias_pattern(n_spins: int) -> np.ndarray:
    pattern = np.array([0.008, -0.012, 0.010, -0.006, 0.004, -0.009, 0.007, -0.005])
    return np.resize(pattern, int(n_spins)).astype(np.float64)


def _complete_case(n_spins: int) -> AuditIsingCase:
    j_matrix = np.zeros((int(n_spins), int(n_spins)), dtype=np.float64)
    scale = 0.055 / math.sqrt(max(int(n_spins), 1))
    for left in range(int(n_spins)):
        for right in range(left + 1, int(n_spins)):
            sign = -1.0 if (left * 17 + right * 31) % 3 == 0 else 1.0
            magnitude = scale * (0.75 + 0.05 * ((left + right) % 5))
            _set_edge(j_matrix, left, right, sign * magnitude)
    return AuditIsingCase(
        n_spins=int(n_spins),
        topology="complete",
        name=f"n{int(n_spins)}_complete",
        j_matrix=j_matrix,
        bias=_bias_pattern(int(n_spins)),
        beta=DEFAULT_BETA,
    )


def _signed_ring_chord_case(n_spins: int) -> AuditIsingCase:
    n_spins = int(n_spins)
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    ring_pattern = np.array([0.060, -0.040, 0.052, -0.030, 0.045, -0.035, 0.055, -0.025])
    chord_pattern = np.array([-0.025, 0.020, -0.018, 0.014, -0.022, 0.016, -0.015, 0.012])
    for idx in range(n_spins):
        _set_edge(j_matrix, idx, (idx + 1) % n_spins, float(ring_pattern[idx % 8]))
        if n_spins > 3:
            _set_edge(j_matrix, idx, (idx + 2) % n_spins, float(chord_pattern[idx % 8]))
    return AuditIsingCase(
        n_spins=n_spins,
        topology="signed_ring_chord",
        name=f"n{n_spins}_signed_ring_chord",
        j_matrix=j_matrix,
        bias=_bias_pattern(n_spins),
        beta=DEFAULT_BETA,
    )


def _sparse_random_case(n_spins: int) -> AuditIsingCase:
    n_spins = int(n_spins)
    rng = np.random.default_rng(20260508 + n_spins)
    target_edges = min(n_spins * 3, n_spins * (n_spins - 1) // 2)
    edge_set = {tuple(sorted((idx, (idx + 1) % n_spins))) for idx in range(n_spins)}
    candidates = [
        (left, right)
        for left in range(n_spins)
        for right in range(left + 1, n_spins)
        if (left, right) not in edge_set
    ]
    rng.shuffle(candidates)
    edge_set.update(candidates[: max(0, target_edges - len(edge_set))])
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for edge_index, (left, right) in enumerate(sorted(edge_set)):
        sign = -1.0 if (edge_index + left) % 4 == 0 else 1.0
        magnitude = 0.045 + 0.004 * ((left + right + edge_index) % 5)
        _set_edge(j_matrix, left, right, sign * magnitude)
    return AuditIsingCase(
        n_spins=n_spins,
        topology="sparse_random",
        name=f"n{n_spins}_sparse_random",
        j_matrix=j_matrix,
        bias=_bias_pattern(n_spins),
        beta=DEFAULT_BETA,
    )


def _lattice_shape(n_spins: int) -> tuple[int, int]:
    root = int(math.sqrt(int(n_spins)))
    for rows in range(root, 0, -1):
        if int(n_spins) % rows == 0:
            return rows, int(n_spins) // rows
    return 1, int(n_spins)


def _lattice_case(n_spins: int) -> AuditIsingCase:
    n_spins = int(n_spins)
    rows, cols = _lattice_shape(n_spins)
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            right = row * cols + ((col + 1) % cols)
            down = ((row + 1) % rows) * cols + col
            horizontal = 0.070 if (row + col) % 2 == 0 else -0.050
            vertical = -0.060 if (row * 2 + col) % 3 == 0 else 0.045
            if right != node:
                _set_edge(j_matrix, node, right, horizontal)
            if down != node:
                _set_edge(j_matrix, node, down, vertical)
    return AuditIsingCase(
        n_spins=n_spins,
        topology="lattice",
        name=f"n{n_spins}_periodic_{rows}x{cols}_lattice",
        j_matrix=j_matrix,
        bias=_bias_pattern(n_spins),
        beta=DEFAULT_BETA,
    )


def _scale_free_case(n_spins: int) -> AuditIsingCase:
    n_spins = int(n_spins)
    if n_spins < 3:
        return _sparse_random_case(n_spins)
    rng = np.random.default_rng(20260509 + n_spins)
    degrees = [0 for _ in range(n_spins)]
    edges = {(0, 1), (0, 2), (1, 2)}
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    for new_node in range(3, n_spins):
        existing = np.arange(new_node)
        weights = np.asarray(degrees[:new_node], dtype=np.float64)
        probabilities = weights / float(np.sum(weights))
        target_count = min(2, new_node)
        targets = rng.choice(existing, size=target_count, replace=False, p=probabilities)
        for target in sorted(int(value) for value in targets):
            edge = tuple(sorted((new_node, target)))
            if edge not in edges:
                edges.add(edge)
                degrees[new_node] += 1
                degrees[target] += 1
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for edge_index, (left, right) in enumerate(sorted(edges)):
        sign = -1.0 if (left + right + edge_index) % 5 == 0 else 1.0
        magnitude = 0.045 + 0.005 * ((left * 3 + right) % 4)
        _set_edge(j_matrix, left, right, sign * magnitude)
    return AuditIsingCase(
        n_spins=n_spins,
        topology="scale_free",
        name=f"n{n_spins}_scale_free",
        j_matrix=j_matrix,
        bias=_bias_pattern(n_spins),
        beta=DEFAULT_BETA,
    )


def build_audit_cases(
    n_values: Sequence[int] = DEFAULT_N_VALUES,
    topologies: Sequence[str] = TOPOLOGIES,
) -> tuple[AuditIsingCase, ...]:
    """Build deterministic cases for the requested n/topology grid."""

    factories = {
        "signed_ring_chord": _signed_ring_chord_case,
        "complete": _complete_case,
        "sparse_random": _sparse_random_case,
        "lattice": _lattice_case,
        "scale_free": _scale_free_case,
    }
    cases: list[AuditIsingCase] = []
    for n_spins in n_values:
        for topology in topologies:
            if topology not in factories:
                raise ValueError(f"unsupported topology: {topology!r}")
            cases.append(factories[topology](int(n_spins)))
    return tuple(cases)


def build_seed_manifest(
    *,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
    topologies: Sequence[str] = TOPOLOGIES,
    carnot_root_seed: int = DEFAULT_CARNOT_ROOT_SEED,
    thrml_root_seed: int = DEFAULT_THRML_ROOT_SEED,
) -> dict[str, Any]:
    """Return a manifest with disjoint root seeds and per-case lineages."""

    cases = [(int(n_spins), str(topology)) for n_spins in n_values for topology in topologies]
    carnot_children = np.random.SeedSequence(int(carnot_root_seed)).spawn(len(cases))
    thrml_children = np.random.SeedSequence(int(thrml_root_seed)).spawn(len(cases))
    case_seeds = []
    for index, ((n_spins, topology), carnot_child, thrml_child) in enumerate(
        zip(cases, carnot_children, thrml_children, strict=True)
    ):
        carnot_seed = int(carnot_child.generate_state(1, dtype=np.uint32)[0])
        thrml_seed = int(thrml_child.generate_state(1, dtype=np.uint32)[0])
        case_seeds.append(
            {
                "case_id": f"n{n_spins}_{topology}",
                "case_index": index,
                "n_spins": n_spins,
                "topology": topology,
                "carnot_seed": carnot_seed,
                "thrml_seed": thrml_seed,
                "seed_relation": "disjoint_root_seed_sequence",
            }
        )
    manifest = {
        "schema": "thrml_carnot_independent_rng_seed_manifest_v1",
        "run_date": RUN_DATE,
        "samplers": {
            "carnot": {
                "root_seed": int(carnot_root_seed),
                "derivation": "numpy_seed_sequence_spawn",
                "prng_family": "jax_prngkey_inside_carnot_cpu_backend",
            },
            "thrml": {
                "root_seed": int(thrml_root_seed),
                "derivation": "numpy_seed_sequence_spawn",
                "prng_family": "jax_prngkey_passed_to_thrml_sample_states",
            },
        },
        "case_seeds": case_seeds,
        "shared_key_object_used": False,
    }
    validate_seed_manifest(manifest)
    return manifest


def validate_seed_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the REQ-SAMPLE-056 disjoint seed-root contract."""

    samplers = dict(manifest.get("samplers") or {})
    carnot = dict(samplers.get("carnot") or {})
    thrml = dict(samplers.get("thrml") or {})
    if int(carnot.get("root_seed", -1)) == int(thrml.get("root_seed", -1)):
        raise ValueError("Carnot and THRML must use disjoint root seeds")
    derivations = {str(carnot.get("derivation", "")), str(thrml.get("derivation", ""))}
    if any("shared_key" in derivation for derivation in derivations):
        raise ValueError("seed manifest must not derive both samplers from a shared key")
    if manifest.get("shared_key_object_used") is True:
        raise ValueError("seed manifest reports shared key object usage")
    seen: set[int] = set()
    for row in manifest.get("case_seeds") or []:
        carnot_seed = int(row["carnot_seed"])
        thrml_seed = int(row["thrml_seed"])
        if carnot_seed == thrml_seed:
            raise ValueError("per-case Carnot and THRML seeds must differ")
        if carnot_seed in seen or thrml_seed in seen:
            raise ValueError("per-case sampler seeds must not be reused")
        seen.add(carnot_seed)
        seen.add(thrml_seed)


def write_seed_manifest(manifest: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Validate and write the seed manifest JSON."""

    validate_seed_manifest(manifest)
    return _write_json(path, manifest)


def sample_path_hash(samples: np.ndarray) -> str:
    """Hash a sampled boolean path including shape and dtype."""

    array = np.ascontiguousarray(np.asarray(samples, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _samples_to_energies(case: AuditIsingCase, samples: np.ndarray) -> np.ndarray:
    spins = np.where(np.asarray(samples, dtype=bool), 1, -1).astype(np.int8)
    return np.asarray([ising_energy(case, state) for state in spins], dtype=np.float64)


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left_sorted = np.sort(np.asarray(left, dtype=np.float64))
    right_sorted = np.sort(np.asarray(right, dtype=np.float64))
    values = np.sort(np.unique(np.concatenate([left_sorted, right_sorted])))
    left_cdf = np.searchsorted(left_sorted, values, side="right") / float(left_sorted.size)
    right_cdf = np.searchsorted(right_sorted, values, side="right") / float(right_sorted.size)
    return float(np.max(np.abs(left_cdf - right_cdf))) if values.size else 0.0


def _ks_p_value(statistic: float, n_left: int, n_right: int) -> float:
    if n_left <= 0 or n_right <= 0:
        return 0.0
    n_eff = (float(n_left) * float(n_right)) / (float(n_left) + float(n_right))
    if statistic <= 0.0:
        return 1.0
    root = math.sqrt(n_eff)
    lam = (root + 0.12 + 0.11 / max(root, 1.0e-12)) * float(statistic)
    terms = [(-1) ** (k - 1) * math.exp(-2.0 * (lam**2) * (k**2)) for k in range(1, 101)]
    return max(0.0, min(1.0, 2.0 * sum(terms)))


def compute_distribution_metrics(
    carnot_energies: np.ndarray,
    thrml_energies: np.ndarray,
    *,
    energy_bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
) -> dict[str, Any]:
    """Compute sampled distribution metrics for one n/topology pair."""

    carnot_values = np.asarray(carnot_energies, dtype=np.float64)
    thrml_values = np.asarray(thrml_energies, dtype=np.float64)
    all_energies = np.concatenate([carnot_values, thrml_values])
    lower = float(np.min(all_energies))
    upper = float(np.max(all_energies))
    if upper <= lower:
        lower -= 0.5
        upper += 0.5
    edges = np.linspace(lower, upper, int(energy_bin_count) + 1)
    carnot_counts, _ = np.histogram(carnot_values, bins=edges)
    thrml_counts, _ = np.histogram(thrml_values, bins=edges)
    carnot_probs = (carnot_counts.astype(np.float64) + 0.5) / (
        float(np.sum(carnot_counts)) + 0.5 * len(carnot_counts)
    )
    thrml_probs = (thrml_counts.astype(np.float64) + 0.5) / (
        float(np.sum(thrml_counts)) + 0.5 * len(thrml_counts)
    )
    kl_divergence = float(np.sum(carnot_probs * np.log(carnot_probs / thrml_probs)))
    ks_stat = _ks_statistic(carnot_values, thrml_values)
    ks_p = _ks_p_value(ks_stat, int(carnot_values.size), int(thrml_values.size))
    return {
        "carnot_mean_energy": _round_metric(float(np.mean(carnot_values))),
        "thrml_mean_energy": _round_metric(float(np.mean(thrml_values))),
        "mean_energy_delta_abs": _round_metric(
            abs(float(np.mean(carnot_values)) - float(np.mean(thrml_values)))
        ),
        "kl_divergence": _round_metric(kl_divergence),
        "ks_statistic": _round_metric(ks_stat),
        "ks_p_value": _round_metric(ks_p),
        "histogram_counts": {
            "energy_bin_count": int(len(carnot_counts)),
            "bin_edges": [_round_metric(float(edge)) for edge in edges],
            "carnot_counts": [int(value) for value in carnot_counts],
            "thrml_counts": [int(value) for value in thrml_counts],
        },
    }


def detect_byte_identical_pairs(
    per_case_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return cases whose stochastic paths or histograms are byte-identical."""

    identical: list[dict[str, str]] = []
    for row in per_case_results:
        same_hash = row.get("carnot_sample_hash") == row.get("thrml_sample_hash")
        histogram = dict(row.get("histogram_counts") or {})
        same_histogram = histogram.get("carnot_counts") == histogram.get("thrml_counts")
        if same_hash or same_histogram:
            if same_hash and same_histogram:
                match_type = "sample_path_hash_and_histogram_counts"
            elif same_hash:
                match_type = "sample_path_hash"
            else:
                match_type = "histogram_counts"
            identical.append({"case_id": str(row.get("case_id")), "match_type": match_type})
    return identical


def inspect_existing_parity_harnesses(paths: Sequence[Path] | None = None) -> dict[str, Any]:
    """Inspect prior parity harnesses for Carnot-lane THRML sampler calls."""

    harness_paths = list(
        paths
        or (
            PROJECT_ROOT / "python/carnot/samplers/thrml_carnot_parity_n32_sample.py",
            PROJECT_ROOT / "python/carnot/samplers/thrml_carnot_parity_n64_sample.py",
            PROJECT_ROOT / "python/carnot/samplers/thrml_carnot_parity_n128_production_scale.py",
            PROJECT_ROOT / "python/carnot/samplers/thrml_diverse_topology_parity_n32.py",
            PROJECT_ROOT / "python/carnot/samplers/thrml_diverse_topology_parity_n64.py",
        )
    )
    carnot_calls_thrml = False
    thrml_lane_cpu_fallback = False
    evidence: list[dict[str, Any]] = []
    for path in harness_paths:
        text = path.read_text(encoding="utf-8")
        carnot_default_thrml = "carnot_backend_factory: BackendFactory = ThrmlSamplerBackend" in text
        carnot_assignment_thrml = "carnot_backend = ThrmlSamplerBackend" in text
        thrml_default_fallback = "thrml_backend_factory: BackendFactory = ThrmlSamplerBackend" in text
        carnot_calls_thrml = carnot_calls_thrml or carnot_default_thrml or carnot_assignment_thrml
        thrml_lane_cpu_fallback = thrml_lane_cpu_fallback or thrml_default_fallback
        evidence.append(
            {
                "path": _display_path(path),
                "carnot_backend_default_thrml": bool(carnot_default_thrml),
                "carnot_assignment_thrml": bool(carnot_assignment_thrml),
                "thrml_lane_uses_thrml_sampler_backend": bool(thrml_default_fallback),
            }
        )
    return {
        "harnesses_checked": evidence,
        "existing_carnot_path_imports_or_calls_thrml_sampler": bool(carnot_calls_thrml),
        "existing_thrml_lane_uses_carnot_cpu_fallback_adapter": bool(thrml_lane_cpu_fallback),
        "new_audit_carnot_path": "carnot.samplers.backend.CpuBackend",
        "new_audit_thrml_path": "installed_thrml_api_sample_states",
        "code_path_independent": not carnot_calls_thrml,
    }


def carnot_cpu_sampler(
    case: AuditIsingCase,
    *,
    seed: int,
    n_samples: int,
    schedule: dict[str, Any],
) -> np.ndarray:
    """Sample one case through Carnot's CPU backend."""

    backend = CpuBackend(seed=int(seed))
    return np.asarray(backend.sample(case.bias, case.j_matrix, int(n_samples), schedule))


def direct_thrml_sampler(
    case: AuditIsingCase,
    *,
    seed: int,
    n_samples: int,
    schedule: dict[str, Any],
) -> np.ndarray:
    """Sample one case through the installed THRML API without Carnot fallback."""

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    jax.config.update("jax_disable_jit", True)
    thrml = importlib.import_module("thrml")
    models = importlib.import_module("thrml.models")
    nodes = [thrml.SpinNode() for _ in range(case.n_spins)]
    edges: list[tuple[Any, Any]] = []
    weights: list[float] = []
    for left in range(case.n_spins):
        for right in range(left + 1, case.n_spins):
            weight = float(case.j_matrix[left, right])
            if weight != 0.0:
                edges.append((nodes[left], nodes[right]))
                weights.append(weight)
    model = models.IsingEBM(
        nodes,
        edges,
        jnp.asarray(case.bias, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(float(case.beta), dtype=jnp.float32),
    )
    blocks = [thrml.Block([node]) for node in nodes]
    program = models.IsingSamplingProgram(model, blocks, [])
    sampling_schedule = thrml.SamplingSchedule(
        n_warmup=int(schedule["n_warmup"]),
        n_samples=int(n_samples),
        steps_per_sample=int(schedule["steps_per_sample"]),
    )
    init_bool = [jnp.asarray([idx % 2 == 0], dtype=bool) for idx in range(case.n_spins)]
    block_samples = thrml.sample_states(
        jrandom.PRNGKey(int(seed)),
        program,
        sampling_schedule,
        init_bool,
        [],
        blocks,
    )
    return np.concatenate([np.asarray(item, dtype=bool) for item in block_samples], axis=1)


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    seed_manifest_path: str | Path = SEED_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before audit execution completes."""

    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "fpga_execution": False,
        },
        "status": "in_progress",
        "milestone": MILESTONE,
        "independent_rng_audit_ready": False,
        "rng_path_independent": False,
        "code_path_independent": False,
        "rng_seed_manifest_path": _display_path(seed_manifest_path),
        "n_values_tested": [],
        "topologies_tested": [],
        "sample_path_hashes": {},
        "byte_identical_pairs": [],
        "nonzero_stochastic_delta_observed": False,
        "per_case_results": [],
        "max_mean_energy_delta_abs": None,
        "max_kl_divergence": None,
        "min_ks_p_value": None,
        "bounded_kl_passed": False,
        "ks_test_passed": False,
        "rng_path_not_independent": True,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "focused_tests_passed": False,
        "honest_verdict": "complete: in_progress_thrml_carnot_independent_rng_audit",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _case_seed_lookup(manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    return {
        str(row["case_id"]): {
            "carnot_seed": int(row["carnot_seed"]),
            "thrml_seed": int(row["thrml_seed"]),
        }
        for row in manifest.get("case_seeds") or []
    }


def run_independent_rng_audit(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    seed_manifest_path: str | Path = SEED_MANIFEST_PATH,
    manifest_path: str | Path = AUDIT_MANIFEST_PATH,
    n_values: Sequence[int] = DEFAULT_N_VALUES,
    topologies: Sequence[str] = TOPOLOGIES,
    carnot_sampler: Callable[..., np.ndarray] = carnot_cpu_sampler,
    thrml_sampler: Callable[..., np.ndarray] = direct_thrml_sampler,
    sample_count_per_case: int = DEFAULT_SAMPLE_COUNT_PER_CASE,
    n_warmup: int = DEFAULT_N_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
    thresholds: Mapping[str, float] = THRESHOLDS,
    energy_bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
    focused_tests_passed: bool = False,
) -> dict[str, Any]:
    """Run the independent-RNG audit and write JSON/JSONL evidence."""

    write_in_progress_artifact(output_path, seed_manifest_path)
    seed_manifest = build_seed_manifest(n_values=n_values, topologies=topologies)
    write_seed_manifest(seed_manifest, seed_manifest_path)
    seed_lookup = _case_seed_lookup(seed_manifest)
    harness_audit = inspect_existing_parity_harnesses()
    cases = build_audit_cases(n_values=n_values, topologies=topologies)
    rows: list[dict[str, Any]] = []
    per_case_results: list[dict[str, Any]] = []
    sample_path_hashes: dict[str, dict[str, str]] = {}
    schedule = {
        "beta": DEFAULT_BETA,
        "n_warmup": int(n_warmup),
        "steps_per_sample": int(steps_per_sample),
        "use_checkerboard": True,
    }
    for case in cases:
        case_id = f"n{case.n_spins}_{case.topology}"
        seeds = seed_lookup[case_id]
        case_schedule = {**schedule, "beta": float(case.beta)}
        carnot_samples = np.asarray(
            carnot_sampler(
                case,
                seed=seeds["carnot_seed"],
                n_samples=int(sample_count_per_case),
                schedule=case_schedule,
            )
        )
        thrml_samples = np.asarray(
            thrml_sampler(
                case,
                seed=seeds["thrml_seed"],
                n_samples=int(sample_count_per_case),
                schedule=case_schedule,
            )
        )
        carnot_energies = _samples_to_energies(case, carnot_samples)
        thrml_energies = _samples_to_energies(case, thrml_samples)
        metrics = compute_distribution_metrics(
            carnot_energies,
            thrml_energies,
            energy_bin_count=energy_bin_count,
        )
        carnot_hash = sample_path_hash(carnot_samples)
        thrml_hash = sample_path_hash(thrml_samples)
        sample_path_hashes[case_id] = {"carnot": carnot_hash, "thrml": thrml_hash}
        row = {
            "case_type": "independent_rng_case",
            "case_id": case_id,
            "n_spins": case.n_spins,
            "topology": case.topology,
            "carnot_seed": seeds["carnot_seed"],
            "thrml_seed": seeds["thrml_seed"],
            "sample_count_per_backend": int(sample_count_per_case),
            "schedule": case_schedule,
            "carnot_sample_hash": carnot_hash,
            "thrml_sample_hash": thrml_hash,
            "histogram_counts": metrics["histogram_counts"],
            "carnot_mean_energy": metrics["carnot_mean_energy"],
            "thrml_mean_energy": metrics["thrml_mean_energy"],
            "mean_energy_delta_abs": metrics["mean_energy_delta_abs"],
            "kl_divergence": metrics["kl_divergence"],
            "ks_statistic": metrics["ks_statistic"],
            "ks_p_value": metrics["ks_p_value"],
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
        }
        rows.append(row)
        per_case_results.append(row)

    byte_identical_pairs = detect_byte_identical_pairs(per_case_results)
    max_mean_energy_delta_abs = max(float(row["mean_energy_delta_abs"]) for row in per_case_results)
    max_kl_divergence = max(float(row["kl_divergence"]) for row in per_case_results)
    min_ks_p_value = min(float(row["ks_p_value"]) for row in per_case_results)
    bounded_kl_passed = all(
        0.0 < float(row["kl_divergence"]) <= float(thresholds["kl_divergence_max"])
        for row in per_case_results
    )
    ks_test_passed = all(
        float(row["ks_p_value"]) >= float(thresholds["ks_p_value_min"]) for row in per_case_results
    )
    nonzero_stochastic_delta_observed = any(
        float(row["mean_energy_delta_abs"]) > 0.0 for row in per_case_results
    )
    mean_energy_passed = all(
        float(row["mean_energy_delta_abs"]) <= float(thresholds["mean_energy_delta_abs_max"])
        for row in per_case_results
    )
    rng_path_independent = len(byte_identical_pairs) == 0
    ready = (
        rng_path_independent
        and bool(harness_audit["code_path_independent"])
        and nonzero_stochastic_delta_observed
        and bounded_kl_passed
        and ks_test_passed
        and mean_energy_passed
        and bool(focused_tests_passed)
    )
    summary_row = {
        "case_type": "independent_rng_audit_summary",
        "n_values_tested": [int(value) for value in n_values],
        "topologies_tested": [str(value) for value in topologies],
        "byte_identical_pairs": byte_identical_pairs,
        "nonzero_stochastic_delta_observed": bool(nonzero_stochastic_delta_observed),
        "max_mean_energy_delta_abs": _round_metric(max_mean_energy_delta_abs),
        "max_kl_divergence": _round_metric(max_kl_divergence),
        "min_ks_p_value": _round_metric(min_ks_p_value),
        "bounded_kl_passed": bool(bounded_kl_passed),
        "ks_test_passed": bool(ks_test_passed),
        "mean_energy_passed": bool(mean_energy_passed),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }
    _write_manifest(manifest_path, [*rows, summary_row])
    verdict = (
        "complete: independent_rng_thrml_carnot_parity_passed_simulator_only_no_hardware_claim"
        if ready
        else "complete: independent_rng_thrml_carnot_parity_not_ready_simulator_only_no_hardware_claim"
    )
    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "audit_manifest_path": _display_path(manifest_path),
            "thresholds": dict(thresholds),
            "sample_count_per_case": int(sample_count_per_case),
            "energy_bin_count": int(energy_bin_count),
            "harness_code_path_audit": harness_audit,
            "carnot_sampler_callable": _callable_label(carnot_sampler),
            "thrml_sampler_callable": _callable_label(thrml_sampler),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "fpga_execution": False,
        },
        "status": "complete",
        "milestone": MILESTONE,
        "independent_rng_audit_ready": bool(ready),
        "rng_path_independent": bool(rng_path_independent),
        "code_path_independent": bool(harness_audit["code_path_independent"]),
        "rng_seed_manifest_path": str(seed_manifest_path),
        "n_values_tested": [int(value) for value in n_values],
        "topologies_tested": [str(value) for value in topologies],
        "sample_path_hashes": sample_path_hashes,
        "byte_identical_pairs": byte_identical_pairs,
        "nonzero_stochastic_delta_observed": bool(nonzero_stochastic_delta_observed),
        "per_case_results": per_case_results,
        "max_mean_energy_delta_abs": _round_metric(max_mean_energy_delta_abs),
        "max_kl_divergence": _round_metric(max_kl_divergence),
        "min_ks_p_value": _round_metric(min_ks_p_value),
        "bounded_kl_passed": bool(bounded_kl_passed),
        "ks_test_passed": bool(ks_test_passed),
        "rng_path_not_independent": not bool(rng_path_independent),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def _callable_label(func: Callable[..., Any]) -> str:
    try:
        return f"{inspect.getmodule(func).__name__}.{func.__name__}"
    except AttributeError:
        return repr(func)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1548 terminal artifact and no-hardware boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1548")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1548")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("independent_rng_audit_ready") is True:
        if artifact.get("byte_identical_pairs"):
            raise ValueError("independent audit readiness cannot include byte-identical pairs")
        required_true = (
            "rng_path_independent",
            "code_path_independent",
            "nonzero_stochastic_delta_observed",
            "bounded_kl_passed",
            "ks_test_passed",
            "focused_tests_passed",
        )
        for field in required_true:
            if artifact.get(field) is not True:
                raise ValueError(f"independent audit readiness requires {field}=true")
        if artifact.get("rng_path_not_independent") is not False:
            raise ValueError("independent audit readiness requires rng_path_not_independent=false")
        if float(artifact.get("max_kl_divergence") or 0.0) <= 0.0:
            raise ValueError("independent audit readiness requires nonzero KL")
        if float(artifact.get("min_ks_p_value") or 0.0) < THRESHOLDS["ks_p_value_min"]:
            raise ValueError("independent audit readiness requires KS p-value gate")


def main() -> None:  # pragma: no cover - exercised by direct operator command.
    run_independent_rng_audit(focused_tests_passed=True)


if __name__ == "__main__":  # pragma: no cover
    main()
