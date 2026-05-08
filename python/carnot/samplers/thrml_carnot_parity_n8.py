"""Exp 1526 exact n=8 THRML/Carnot simulator parity.

This helper runs the first scaled THRML/Carnot parity case beyond the n=4
smoke evidence. It is intentionally narrow: one deterministic eight-spin
signed ring-chord Ising model, exact enumeration over all 256 states, and a
fixed-seed sampling row that is recorded only as secondary software evidence.
No Extropic TSU hardware path is executed or claimed.

Spec traces: REQ-SAMPLE-047, SCENARIO-SAMPLE-075.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.analysis.pbit_sampler_portability import enumerate_spin_states, ising_energy
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1526_thrml_carnot_parity_n8.json"
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_parity_n8_1526.jsonl"
EXP1515_PATH = (
    PROJECT_ROOT / "results" / "experiment_1515_thrml_samplerbackend_conformance_pack.json"
)

EXPERIMENT_ID = 1526
RUN_DATE = "20260508"
SCHEMA = "thrml_carnot_parity_n8_v1"
DEFAULT_SEED = 20260508
DEFAULT_SAMPLE_COUNT = 128
DEFAULT_N_WARMUP = 64
DEFAULT_STEPS_PER_SAMPLE = 4

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
THRESHOLDS = {
    "partition_relative_error_max": 1.0e-6,
    "mean_energy_delta_abs_max": 1.0e-6,
    "kl_divergence_max": 1.0e-6,
    "sample_mean_energy_delta_abs_max": 0.35,
}
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_parity_n8_passed",
    "simulator_only",
    "no_tsu_hardware_claim",
    "n_spins",
    "exact_states_enumerated",
    "topology",
    "seed",
    "carnot_partition_function",
    "thrml_partition_function",
    "partition_relative_error",
    "mean_energy_delta",
    "kl_divergence",
    "parity_manifest_path",
    "blockers",
    "honest_verdict",
}

ImportModule = Callable[[str], Any]
BackendFactory = Callable[[int], Any]


@dataclass(frozen=True)
class ParityIsingCase:
    """A deterministic Ising case small enough for exact THRML/Carnot checks."""

    name: str
    topology: str
    j_matrix: np.ndarray
    bias: np.ndarray
    beta: float

    @property
    def n_spins(self) -> int:
        """Return the spin count used by enumeration and backend sample shapes."""
        return int(self.bias.shape[0])


class MissingThrmlApi(RuntimeError):
    """Raised when local THRML imports but lacks the Ising energy API needed."""


def n8_signed_ring_chord_case() -> ParityIsingCase:
    """Return the deterministic eight-spin signed ring-chord parity case.

    The graph uses one signed ring edge and one signed distance-two chord per
    spin. That keeps exact enumeration cheap while exercising enough mixed
    ferro/antiferromagnetic structure to catch sign-convention drift.

    Spec traces: REQ-SAMPLE-047.
    """

    n_spins = 8
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    ring_weights = np.array([0.62, -0.44, 0.51, -0.38, 0.47, -0.53, 0.36, -0.41])
    chord_weights = np.array([-0.31, 0.27, -0.46, 0.22, -0.35, 0.18, -0.29, 0.26])
    for idx, weight in enumerate(ring_weights):
        left, right = idx, (idx + 1) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    for idx, weight in enumerate(chord_weights):
        left, right = idx, (idx + 2) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    bias = np.array([0.12, -0.07, 0.05, -0.10, 0.09, -0.03, 0.04, -0.02])
    return ParityIsingCase(
        name="n8_signed_ring_chord",
        topology="signed_ring_chord",
        j_matrix=j_matrix,
        bias=bias.astype(np.float64),
        beta=1.10,
    )


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


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before import probing or parity execution.

    Spec traces: REQ-SAMPLE-047.
    """

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
        },
        "status": "in_progress",
        "thrml_parity_n8_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 8,
        "exact_states_enumerated": 0,
        "topology": "signed_ring_chord",
        "seed": DEFAULT_SEED,
        "carnot_partition_function": None,
        "thrml_partition_function": None,
        "partition_relative_error": None,
        "mean_energy_delta": None,
        "kl_divergence": None,
        "thresholds": dict(THRESHOLDS),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_carnot_parity_n8_simulator_only",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _load_exp1515_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1515_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1515_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_import_ready") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1515_thrml_import_not_ready",
                "detail": "Exp1515 must be complete, import-ready, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def _import_thrml(
    importer: ImportModule,
) -> tuple[Any | None, dict[str, Any], dict[str, str] | None]:
    try:
        thrml_module = importer("thrml")
        models_module = importer("thrml.models")
    except Exception as exc:
        return (
            None,
            {},
            {
                "blocker": "thrml_local_import_unavailable",
                "detail": f"{exc.__class__.__name__}: {exc}",
            },
        )
    return (
        (thrml_module, models_module),
        {
            "thrml_version": str(getattr(thrml_module, "__version__", "unknown")),
            "thrml_import_path": str(getattr(thrml_module, "__file__", "<unknown>")),
        },
        None,
    )


def _ising_edge_payload(case: ParityIsingCase) -> tuple[list[tuple[int, int]], np.ndarray]:
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for left in range(case.n_spins):
        for right in range(left + 1, case.n_spins):
            weight = float(case.j_matrix[left, right])
            if weight != 0.0:
                edges.append((left, right))
                weights.append(weight)
    return edges, np.asarray(weights, dtype=np.float64)


def _build_thrml_model(thrml_modules: Any, case: ParityIsingCase) -> tuple[Any, list[Any], Any]:
    thrml_module, models_module = thrml_modules
    spin_node_cls = getattr(thrml_module, "SpinNode", None)
    ising_cls = getattr(models_module, "IsingEBM", None)
    if spin_node_cls is None or ising_cls is None:
        raise MissingThrmlApi("local THRML API lacks SpinNode or models.IsingEBM")
    nodes = [spin_node_cls() for _ in range(case.n_spins)]
    edge_indices, weights = _ising_edge_payload(case)
    node_edges = [(nodes[left], nodes[right]) for left, right in edge_indices]
    model = ising_cls(
        nodes,
        node_edges,
        np.asarray(case.bias, dtype=np.float64),
        weights,
        1.0,
    )
    if getattr(model, "energy", None) is None:
        raise MissingThrmlApi("local THRML IsingEBM lacks an energy(spins) method")
    return model, nodes, thrml_module


def _thrml_energy_for_state(
    model: Any, nodes: list[Any], thrml_module: Any, state: np.ndarray
) -> float:
    try:
        return float(model.energy(np.asarray(state, dtype=np.float64)))
    except TypeError as exc:
        block_cls = getattr(thrml_module, "Block", None)
        if block_cls is None:
            raise MissingThrmlApi("local THRML block energy API lacks Block") from exc
        bool_state = np.asarray(state == 1, dtype=bool)
        return float(model.energy([bool_state], [block_cls(nodes)]))


def _boltzmann_from_energies(energies: np.ndarray, beta: float) -> tuple[float, np.ndarray, float]:
    shifted = -float(beta) * np.asarray(energies, dtype=np.float64)
    max_shift = float(np.max(shifted))
    weights = np.exp(shifted - max_shift)
    partition = float(np.exp(max_shift) * np.sum(weights))
    distribution = weights / float(np.sum(weights))
    mean_energy = float(distribution @ np.asarray(energies, dtype=np.float64))
    return partition, distribution, mean_energy


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_norm = np.asarray(p, dtype=np.float64) / float(np.sum(p))
    q_norm = np.asarray(q, dtype=np.float64) / float(np.sum(q))
    mask = p_norm > 0.0
    return float(np.sum(p_norm[mask] * np.log(p_norm[mask] / np.maximum(q_norm[mask], 1.0e-15))))


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def exact_parity_metrics(thrml_modules: Any, case: ParityIsingCase) -> dict[str, Any]:
    """Enumerate all states and compare exact Carnot/THRML distributions."""

    states = enumerate_spin_states(case.n_spins)
    thrml_model, nodes, thrml_module = _build_thrml_model(thrml_modules, case)
    carnot_energies = np.asarray([ising_energy(case, state) for state in states], dtype=np.float64)
    thrml_energies = np.asarray(
        [_thrml_energy_for_state(thrml_model, nodes, thrml_module, state) for state in states],
        dtype=np.float64,
    )
    carnot_z, carnot_distribution, carnot_mean_energy = _boltzmann_from_energies(
        carnot_energies, case.beta
    )
    thrml_z, thrml_distribution, thrml_mean_energy = _boltzmann_from_energies(
        thrml_energies, case.beta
    )
    partition_relative_error = abs(carnot_z - thrml_z) / max(abs(carnot_z), 1.0e-15)
    return {
        "case_id": f"exp1526:{case.name}:exact_distribution",
        "case_type": "exact_distribution_parity",
        "n_spins": case.n_spins,
        "state_count": int(len(states)),
        "topology": case.topology,
        "beta": float(case.beta),
        "carnot_partition_function": _round_metric(carnot_z),
        "thrml_partition_function": _round_metric(thrml_z),
        "partition_relative_error": _round_metric(partition_relative_error),
        "carnot_mean_energy": _round_metric(carnot_mean_energy),
        "thrml_mean_energy": _round_metric(thrml_mean_energy),
        "mean_energy_delta": _round_metric(abs(carnot_mean_energy - thrml_mean_energy)),
        "kl_divergence": _round_metric(_kl_divergence(carnot_distribution, thrml_distribution)),
        "max_energy_abs_delta": _round_metric(
            float(np.max(np.abs(carnot_energies - thrml_energies)))
        ),
        "thresholds": dict(THRESHOLDS),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def _sample_energies(case: ParityIsingCase, samples: np.ndarray) -> np.ndarray:
    spin_samples = np.where(np.asarray(samples, dtype=bool), 1, -1).astype(np.int8)
    return np.asarray([ising_energy(case, state) for state in spin_samples], dtype=np.float64)


def sampling_metrics(
    case: ParityIsingCase,
    *,
    seed: int,
    carnot_backend_factory: BackendFactory,
    thrml_backend_factory: BackendFactory,
    sample_count: int,
    n_warmup: int,
    steps_per_sample: int,
) -> dict[str, Any]:
    """Record fixed-seed software sampling as secondary parity evidence."""

    config = {
        "beta": float(case.beta),
        "n_warmup": int(n_warmup),
        "steps_per_sample": int(steps_per_sample),
        "use_checkerboard": True,
    }
    carnot_backend = carnot_backend_factory(int(seed))
    thrml_backend = thrml_backend_factory(int(seed))
    carnot_samples = np.asarray(
        carnot_backend.sample(case.bias, case.j_matrix, int(sample_count), config)
    )
    thrml_samples = np.asarray(
        thrml_backend.sample(case.bias, case.j_matrix, int(sample_count), config)
    )
    carnot_energies = _sample_energies(case, carnot_samples)
    thrml_energies = _sample_energies(case, thrml_samples)
    return {
        "case_id": f"exp1526:{case.name}:fixed_seed_sampling",
        "case_type": "fixed_seed_sampling_secondary_check",
        "seed": int(seed),
        "sample_count": int(sample_count),
        "schedule": config,
        "carnot_backend": str(getattr(carnot_backend, "backend_name", "<unknown>")),
        "thrml_backend": str(getattr(thrml_backend, "backend_name", "<unknown>")),
        "carnot_mean_energy": _round_metric(float(np.mean(carnot_energies))),
        "thrml_mean_energy": _round_metric(float(np.mean(thrml_energies))),
        "mean_energy_delta": _round_metric(abs(float(np.mean(carnot_energies - thrml_energies)))),
        "carnot_best_energy": _round_metric(float(np.min(carnot_energies))),
        "thrml_best_energy": _round_metric(float(np.min(thrml_energies))),
        "thresholds": {
            "sample_mean_energy_delta_abs_max": THRESHOLDS["sample_mean_energy_delta_abs_max"]
        },
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def _write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _blocked_artifact(
    *,
    manifest_path: str | Path,
    seed: int,
    blockers: list[dict[str, str]],
    verdict: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            **dict(metadata or {}),
        },
        "status": "blocked",
        "thrml_parity_n8_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 8,
        "exact_states_enumerated": 0,
        "topology": "signed_ring_chord",
        "seed": int(seed),
        "carnot_partition_function": None,
        "thrml_partition_function": None,
        "partition_relative_error": None,
        "mean_energy_delta": None,
        "kl_divergence": None,
        "thresholds": dict(THRESHOLDS),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def _passed_exact_thresholds(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["partition_relative_error"]) <= THRESHOLDS["partition_relative_error_max"]
        and float(metrics["mean_energy_delta"]) <= THRESHOLDS["mean_energy_delta_abs_max"]
        and float(metrics["kl_divergence"]) <= THRESHOLDS["kl_divergence_max"]
    )


def run_parity_n8(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1515_path: str | Path = EXP1515_PATH,
    importer: ImportModule = importlib.import_module,
    carnot_backend_factory: BackendFactory = CpuBackend,
    thrml_backend_factory: BackendFactory = ThrmlSamplerBackend,
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    n_warmup: int = DEFAULT_N_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
) -> dict[str, Any]:
    """Run exact n=8 THRML/Carnot parity and write JSON/JSONL evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    exp1515_ready, exp1515_payload, exp1515_blocker = _load_exp1515_ready(exp1515_path)
    if not exp1515_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seed=seed,
                blockers=[
                    exp1515_blocker
                    or {
                        "blocker": "exp1515_thrml_import_not_ready",
                        "detail": "unknown Exp1515 blocker",
                    }
                ],
                verdict="complete_thrml_carnot_parity_n8_blocked_exp1515_no_tsu_hardware_claim",
                metadata={"exp1515_status": exp1515_payload.get("status")},
            ),
        )
    thrml_modules, thrml_details, import_blocker = _import_thrml(importer)
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seed=seed,
                blockers=[import_blocker],
                verdict="complete_thrml_carnot_parity_n8_blocked_simulator_dependency_no_tsu_hardware_claim",
                metadata={"exp1515_status": exp1515_payload.get("status")},
            ),
        )
    case = n8_signed_ring_chord_case()
    try:
        exact_row = exact_parity_metrics(thrml_modules, case)
    except MissingThrmlApi as exc:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seed=seed,
                blockers=[{"blocker": "thrml_ising_energy_api_unavailable", "detail": str(exc)}],
                verdict="complete_thrml_carnot_parity_n8_blocked_thrml_api_no_tsu_hardware_claim",
                metadata=thrml_details,
            ),
        )
    sample_row = sampling_metrics(
        case,
        seed=seed,
        carnot_backend_factory=carnot_backend_factory,
        thrml_backend_factory=thrml_backend_factory,
        sample_count=sample_count,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
    )
    rows = [exact_row, sample_row]
    _write_manifest(manifest_path, rows)
    passed = _passed_exact_thresholds(exact_row)
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "exp1515_status": exp1515_payload.get("status"),
            **thrml_details,
        },
        "status": "complete",
        "thrml_parity_n8_passed": passed,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": case.n_spins,
        "exact_states_enumerated": int(exact_row["state_count"]),
        "topology": case.topology,
        "seed": int(seed),
        "carnot_partition_function": exact_row["carnot_partition_function"],
        "thrml_partition_function": exact_row["thrml_partition_function"],
        "partition_relative_error": exact_row["partition_relative_error"],
        "mean_energy_delta": exact_row["mean_energy_delta"],
        "kl_divergence": exact_row["kl_divergence"],
        "thresholds": dict(THRESHOLDS),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": []
        if passed
        else [{"blocker": "exact_parity_threshold_failed", "detail": str(exact_row)}],
        "honest_verdict": (
            "complete_thrml_carnot_parity_n8_passed_no_tsu_hardware_claim"
            if passed
            else "complete_thrml_carnot_parity_n8_failed_exact_thresholds_no_tsu_hardware_claim"
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, exact pass gates, and no-TSU claim boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1526")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1526")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("thrml_parity_n8_passed") is True:
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        pass_metrics = (
            int(artifact.get("exact_states_enumerated") or 0) == 256
            and float(artifact.get("partition_relative_error") or 0.0)
            <= float(thresholds["partition_relative_error_max"])
            and float(artifact.get("mean_energy_delta") or 0.0)
            <= float(thresholds["mean_energy_delta_abs_max"])
            and float(artifact.get("kl_divergence") or 0.0)
            <= float(thresholds["kl_divergence_max"])
        )
        if not pass_metrics:
            raise ValueError("thrml_parity_n8_passed requires exact pass metrics")
