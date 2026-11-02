"""Exp 1531 sampled n=32 THRML/Carnot parity across diverse topologies.

This module keeps the Exp 1528 sampled parity protocol at the same 32-spin
scale, but varies the graph family so the evidence is not anchored to a single
ring-chord case. The comparison is still software/simulator-only: both backends
receive identical Ising parameters and schedules, and the artifact explicitly
does not claim Extropic TSU hardware execution.

Spec traces: REQ-SAMPLE-052, SCENARIO-SAMPLE-080.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.samplers import thrml_carnot_parity_n32_sample as parity_n32
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = parity_n32.PROJECT_ROOT
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1531_thrml_diverse_topology_parity_n32.json"
)
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_diverse_topology_parity_n32_1531.jsonl"
EXP1528_PATH = PROJECT_ROOT / "results" / "experiment_1528_thrml_carnot_parity_n32_sample.json"

EXPERIMENT_ID = 1531
RUN_DATE = "20260508"
SCHEMA = "thrml_diverse_topology_parity_n32_v1"
TOPOLOGIES = ("complete", "sparse_random", "lattice", "scale_free")
TOPOLOGY_SEEDS = {"sparse_random": 20260508, "scale_free": 20260509}
DEFAULT_SEEDS = parity_n32.DEFAULT_SEEDS
DEFAULT_SAMPLE_COUNT_PER_SEED = parity_n32.DEFAULT_SAMPLE_COUNT_PER_SEED
DEFAULT_N_WARMUP = parity_n32.DEFAULT_N_WARMUP
DEFAULT_STEPS_PER_SAMPLE = parity_n32.DEFAULT_STEPS_PER_SAMPLE
DEFAULT_ENERGY_BIN_COUNT = parity_n32.DEFAULT_ENERGY_BIN_COUNT
THRESHOLDS = dict(parity_n32.THRESHOLDS)
TERMINAL_VERDICT_PREFIXES = parity_n32.TERMINAL_VERDICT_PREFIXES
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "diverse_topology_parity_ready",
    "simulator_only",
    "no_tsu_hardware_claim",
    "n_spins",
    "topologies_tested",
    "topologies_passed",
    "topology_results",
    "mean_energy_delta_by_topology",
    "kl_divergence_by_topology",
    "parity_manifest_path",
    "blockers",
    "honest_verdict",
}

ImportModule = parity_n32.ImportModule
BackendFactory = parity_n32.BackendFactory
ParityIsingCase = parity_n32.ParityIsingCase


def _display_path(path: str | Path) -> str:
    return parity_n32._display_path(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return parity_n32._write_json(path, payload)


def _write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    parity_n32._write_manifest(path, [dict(row) for row in rows])


def _set_edge(j_matrix: np.ndarray, left: int, right: int, weight: float) -> None:
    j_matrix[int(left), int(right)] = float(weight)
    j_matrix[int(right), int(left)] = float(weight)


def _shared_bias(n_spins: int = 32) -> np.ndarray:
    return np.tile(np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float64), n_spins // 4)


def _complete_case() -> ParityIsingCase:
    n_spins = 32
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for left in range(n_spins):
        for right in range(left + 1, n_spins):
            sign = -1.0 if (left * 31 + right * 17) % 2 else 1.0
            magnitude = 0.008 + 0.002 * ((left + 2 * right) % 4)
            _set_edge(j_matrix, left, right, sign * magnitude)
    return ParityIsingCase(
        name="n32_complete",
        topology="complete",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.10,
    )


def _sparse_random_case(seed: int = TOPOLOGY_SEEDS["sparse_random"]) -> ParityIsingCase:
    n_spins = 32
    target_edges = 80
    rng = np.random.default_rng(int(seed))
    edge_set = {tuple(sorted((idx, (idx + 1) % n_spins))) for idx in range(n_spins)}
    candidates = [
        (left, right)
        for left in range(n_spins)
        for right in range(left + 1, n_spins)
        if (left, right) not in edge_set
    ]
    rng.shuffle(candidates)
    edge_set.update(candidates[: target_edges - len(edge_set)])
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for edge_index, (left, right) in enumerate(sorted(edge_set)):
        sign = -1.0 if edge_index % 3 == 0 else 1.0
        magnitude = 0.05 + 0.01 * ((left + right + edge_index) % 5)
        _set_edge(j_matrix, left, right, sign * magnitude)
    return ParityIsingCase(
        name=f"n32_sparse_random_seed_{int(seed)}",
        topology="sparse_random",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.10,
    )


def _lattice_case() -> ParityIsingCase:
    rows = 4
    cols = 8
    n_spins = rows * cols
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            right = row * cols + ((col + 1) % cols)
            down = ((row + 1) % rows) * cols + col
            horizontal = 0.11 if (row + col) % 2 == 0 else -0.07
            vertical = -0.09 if (row * 2 + col) % 3 == 0 else 0.06
            _set_edge(j_matrix, node, right, horizontal)
            _set_edge(j_matrix, node, down, vertical)
    return ParityIsingCase(
        name="n32_periodic_4x8_lattice",
        topology="lattice",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.10,
    )


def _scale_free_case(seed: int = TOPOLOGY_SEEDS["scale_free"]) -> ParityIsingCase:
    n_spins = 32
    m_edges = 2
    rng = np.random.default_rng(int(seed))
    degrees = [0 for _ in range(n_spins)]
    edges = {(0, 1), (0, 2), (1, 2)}
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    for new_node in range(3, n_spins):
        existing = np.arange(new_node)
        weights = np.asarray(degrees[:new_node], dtype=np.float64)
        probabilities = weights / float(np.sum(weights))
        targets = rng.choice(existing, size=m_edges, replace=False, p=probabilities)
        for target in sorted(int(value) for value in targets):
            edge = tuple(sorted((new_node, target)))
            edges.add(edge)
            degrees[new_node] += 1
            degrees[target] += 1
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for edge_index, (left, right) in enumerate(sorted(edges)):
        sign = -1.0 if (left + right + edge_index) % 4 == 0 else 1.0
        magnitude = 0.055 + 0.015 * ((left * 3 + right) % 4)
        _set_edge(j_matrix, left, right, sign * magnitude)
    return ParityIsingCase(
        name=f"n32_scale_free_seed_{int(seed)}",
        topology="scale_free",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.10,
    )


def n32_diverse_topology_cases() -> tuple[ParityIsingCase, ...]:
    """Return deterministic complete, sparse-random, lattice, and scale-free cases.

    Spec traces: REQ-SAMPLE-052.
    """

    return (_complete_case(), _sparse_random_case(), _lattice_case(), _scale_free_case())


def _load_exp1528_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1528_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1528_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_parity_n32_passed") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1528_parity_not_passed",
                "detail": "Exp1528 must be complete, n=32-passed, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before THRML probing or parity execution.

    Spec traces: REQ-SAMPLE-052.
    """

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "topology_seeds": dict(TOPOLOGY_SEEDS),
        },
        "status": "in_progress",
        "diverse_topology_parity_ready": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 32,
        "topologies_tested": list(TOPOLOGIES),
        "topologies_passed": [],
        "topology_results": {},
        "mean_energy_delta_by_topology": {},
        "kl_divergence_by_topology": {},
        "seeds": list(DEFAULT_SEEDS),
        "n_samples_per_backend": 0,
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_diverse_topology_parity_n32_simulator_only",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _blocked_artifact(
    *,
    manifest_path: str | Path,
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
            "topology_seeds": dict(TOPOLOGY_SEEDS),
            **dict(metadata or {}),
        },
        "status": "blocked",
        "diverse_topology_parity_ready": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 32,
        "topologies_tested": list(TOPOLOGIES),
        "topologies_passed": [],
        "topology_results": {},
        "mean_energy_delta_by_topology": {},
        "kl_divergence_by_topology": {},
        "seeds": list(DEFAULT_SEEDS),
        "n_samples_per_backend": 0,
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def sampled_topology_backend_row(
    case: ParityIsingCase,
    *,
    seed: int,
    backend_label: str,
    backend_name: str,
    samples: np.ndarray,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one topology/backend/seed sampled chain group.

    Spec traces: REQ-SAMPLE-052.
    """

    row = parity_n32.sampled_backend_row(
        case,
        seed=seed,
        backend_label=backend_label,
        backend_name=backend_name,
        samples=samples,
        schedule=schedule,
    )
    row.update(
        {
            "case_id": f"exp1531:{case.topology}:seed_{int(seed)}:{backend_label}",
            "case_type": "sampled_topology_seed_backend",
            "topology": case.topology,
        }
    )
    return row


def summarize_diverse_topology_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    topologies: Sequence[str],
    seeds: Sequence[int],
    thresholds: Mapping[str, float],
    energy_bin_count: int,
) -> dict[str, Any]:
    """Aggregate topology-level sampled parity rows into the Exp 1531 summary row.

    Spec traces: REQ-SAMPLE-052, SCENARIO-SAMPLE-080.
    """

    topology_results: dict[str, dict[str, Any]] = {}
    for topology in topologies:
        topology_rows = [dict(row) for row in rows if row.get("topology") == topology]
        summary = parity_n32.summarize_sampled_rows(
            topology_rows,
            seeds=seeds,
            thresholds=thresholds,
            energy_bin_count=energy_bin_count,
        )
        summary["case_id"] = f"exp1531:{topology}:sampled_summary"
        summary["topology"] = topology
        topology_results[str(topology)] = summary

    topologies_passed = [
        topology
        for topology in topologies
        if bool(topology_results[str(topology)]["passed_thresholds"])
    ]
    mean_delta = {
        topology: topology_results[str(topology)]["mean_energy_delta"] for topology in topologies
    }
    kl_delta = {topology: topology_results[str(topology)]["kl_divergence"] for topology in topologies}
    return {
        "case_id": "exp1531:diverse_topology_n32:summary",
        "case_type": "diverse_topology_summary",
        "seeds": [int(seed) for seed in seeds],
        "n_spins": 32,
        "topologies_tested": [str(topology) for topology in topologies],
        "topologies_passed": [str(topology) for topology in topologies_passed],
        "topology_results": topology_results,
        "mean_energy_delta_by_topology": mean_delta,
        "kl_divergence_by_topology": kl_delta,
        "diverse_topology_parity_ready": len(topologies_passed) >= 3,
        "thresholds": dict(thresholds),
        "energy_bin_count": int(energy_bin_count),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def run_diverse_topology_parity_n32(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1528_path: str | Path = EXP1528_PATH,
    importer: ImportModule = importlib.import_module,
    carnot_backend_factory: BackendFactory = CpuBackend,
    thrml_backend_factory: BackendFactory = ThrmlSamplerBackend,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sample_count_per_seed: int = DEFAULT_SAMPLE_COUNT_PER_SEED,
    n_warmup: int = DEFAULT_N_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
    thresholds: Mapping[str, float] = THRESHOLDS,
    energy_bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
) -> dict[str, Any]:
    """Run n=32 sampled parity across four topology families and write evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    exp1528_ready, exp1528_payload, exp1528_blocker = _load_exp1528_ready(exp1528_path)
    if not exp1528_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[
                    exp1528_blocker
                    or {
                        "blocker": "exp1528_parity_not_passed",
                        "detail": "unknown Exp1528 blocker",
                    }
                ],
                verdict="complete_thrml_diverse_topology_parity_n32_blocked_exp1528_no_tsu_hardware_claim",
                metadata={"exp1528_status": exp1528_payload.get("status")},
            ),
        )

    _thrml_modules, thrml_details, import_blocker = parity_n32.parity_n16.parity_n8._import_thrml(
        importer
    )
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[import_blocker],
                verdict=(
                    "complete_thrml_diverse_topology_parity_n32_blocked_simulator_dependency_"
                    "no_tsu_hardware_claim"
                ),
                metadata={"exp1528_status": exp1528_payload.get("status")},
            ),
        )

    cases = n32_diverse_topology_cases()
    rows: list[dict[str, Any]] = []
    for case in cases:
        schedule = {
            "beta": float(case.beta),
            "n_warmup": int(n_warmup),
            "steps_per_sample": int(steps_per_sample),
            "use_checkerboard": True,
        }
        for seed in seeds:
            carnot_backend = carnot_backend_factory(int(seed))
            thrml_backend = thrml_backend_factory(int(seed))
            carnot_samples = np.asarray(
                carnot_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
            )
            thrml_samples = np.asarray(
                thrml_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
            )
            rows.append(
                sampled_topology_backend_row(
                    case,
                    seed=int(seed),
                    backend_label="carnot",
                    backend_name=str(getattr(carnot_backend, "backend_name", "<unknown>")),
                    samples=carnot_samples,
                    schedule=schedule,
                )
            )
            rows.append(
                sampled_topology_backend_row(
                    case,
                    seed=int(seed),
                    backend_label="thrml",
                    backend_name=str(getattr(thrml_backend, "backend_name", "<unknown>")),
                    samples=thrml_samples,
                    schedule=schedule,
                )
            )

    summary_row = summarize_diverse_topology_rows(
        rows,
        topologies=TOPOLOGIES,
        seeds=seeds,
        thresholds=thresholds,
        energy_bin_count=energy_bin_count,
    )
    _write_manifest(manifest_path, [*rows, summary_row])
    ready = bool(summary_row["diverse_topology_parity_ready"])
    failed_topologies = [
        topology
        for topology in summary_row["topologies_tested"]
        if topology not in summary_row["topologies_passed"]
    ]
    sample_counts = [
        int(result["n_samples_per_backend"])
        for result in summary_row["topology_results"].values()
    ]
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "topology_seeds": dict(TOPOLOGY_SEEDS),
            "exp1528_status": exp1528_payload.get("status"),
            **thrml_details,
        },
        "status": "complete",
        "diverse_topology_parity_ready": ready,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 32,
        "topologies_tested": summary_row["topologies_tested"],
        "topologies_passed": summary_row["topologies_passed"],
        "topology_results": summary_row["topology_results"],
        "mean_energy_delta_by_topology": summary_row["mean_energy_delta_by_topology"],
        "kl_divergence_by_topology": summary_row["kl_divergence_by_topology"],
        "seeds": [int(seed) for seed in seeds],
        "n_samples_per_backend": int(min(sample_counts) if sample_counts else 0),
        "sample_count_per_seed": int(sample_count_per_seed),
        "warmup": int(n_warmup),
        "thinning": int(steps_per_sample),
        "thresholds": dict(thresholds),
        "energy_bin_count": int(energy_bin_count),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": []
        if ready
        else [
            {
                "blocker": "topology_sampled_parity_threshold_failed",
                "detail": ",".join(failed_topologies),
            }
        ],
        "honest_verdict": (
            "complete_thrml_diverse_topology_parity_n32_passed_no_tsu_hardware_claim"
            if ready
            else "complete_thrml_diverse_topology_parity_n32_failed_thresholds_no_tsu_hardware_claim"
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, readiness gates, and simulator-only boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1531")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1531")
    if int(artifact.get("n_spins") or 0) != 32:
        raise ValueError("Exp 1531 artifacts must remain at n_spins=32")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("diverse_topology_parity_ready") is True:
        topologies_tested = list(artifact.get("topologies_tested") or [])
        topologies_passed = list(artifact.get("topologies_passed") or [])
        topology_results = dict(artifact.get("topology_results") or {})
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        gates_ok = (
            topologies_tested == list(TOPOLOGIES)
            and len(topologies_passed) >= 3
            and set(topologies_passed).issubset(set(topologies_tested))
            and set(topology_results) == set(topologies_tested)
        )
        for topology in topologies_passed:
            result = dict(topology_results.get(topology) or {})
            gates_ok = gates_ok and bool(result.get("passed_thresholds"))
            gates_ok = gates_ok and float(result.get("mean_energy_delta") or 0.0) <= float(
                thresholds["mean_energy_delta_abs_max"]
            )
            gates_ok = gates_ok and float(result.get("magnetization_delta") or 0.0) <= float(
                thresholds["magnetization_delta_abs_max"]
            )
            gates_ok = gates_ok and float(result.get("kl_divergence") or 0.0) <= float(
                thresholds["kl_divergence_max"]
            )
            gates_ok = gates_ok and bool(result.get("kl_estimate_stable"))
        if not gates_ok:
            raise ValueError("diverse_topology_parity_ready requires at least three passing topologies")
