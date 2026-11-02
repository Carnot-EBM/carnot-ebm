"""Exp 1544 sampled n=64 THRML/Carnot parity across diverse topologies.

This module scales the Exp 1531 diverse-topology parity lane from 32 to 64
spins after the Exp 1543 n=256 schedule-stress gate. It remains a software
simulator comparison: Carnot and the THRML adapter receive identical Ising
models and comparable fixed-temperature schedules, while the artifact
explicitly disallows TSU, Z1, XTR-0, board, synthesis, and bitstream claims.

Spec traces: REQ-SAMPLE-054, SCENARIO-SAMPLE-082.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.samplers import thrml_carnot_parity_n64_sample as parity_n64
from carnot.samplers import thrml_carnot_parity_n256_schedule_stress as parity_n256
from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = parity_n64.PROJECT_ROOT
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1544_thrml_diverse_topology_parity_n64.json"
)
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_diverse_topology_parity_n64_1544.jsonl"
EXP1531_PATH = PROJECT_ROOT / "results" / "experiment_1531_thrml_diverse_topology_parity_n32.json"
EXP1543_PATH = (
    PROJECT_ROOT / "results" / "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json"
)

EXPERIMENT_ID = 1544
RUN_DATE = "20260508"
MILESTONE = "2026.04.118"
SCHEMA = "thrml_diverse_topology_parity_n64_v1"
TOPOLOGIES = ("complete", "sparse_random", "lattice", "scale_free")
TOPOLOGY_SEEDS = {"sparse_random": 20260510, "scale_free": 20260511}
DEFAULT_SEEDS = parity_n64.DEFAULT_SEEDS
DEFAULT_THRML_SEED_OFFSET = 100_000
DEFAULT_SAMPLE_COUNT_PER_SEED = parity_n64.DEFAULT_SAMPLE_COUNT_PER_SEED
DEFAULT_N_WARMUP = parity_n64.DEFAULT_N_WARMUP
DEFAULT_STEPS_PER_SAMPLE = parity_n64.DEFAULT_STEPS_PER_SAMPLE
DEFAULT_ENERGY_BIN_COUNT = parity_n64.DEFAULT_ENERGY_BIN_COUNT
THRESHOLDS = {
    "mean_energy_delta_abs_max": parity_n64.THRESHOLDS["mean_energy_delta_abs_max"],
    "mean_energy_delta_percent_max": parity_n64.THRESHOLDS["mean_energy_delta_percent_max"],
    "max_energy_delta_abs_max": parity_n64.THRESHOLDS["mean_energy_delta_abs_max"],
    "magnetization_delta_abs_max": parity_n64.THRESHOLDS["magnetization_delta_abs_max"],
    "kl_divergence_max": parity_n64.THRESHOLDS["kl_divergence_max"],
    "kl_min_samples_per_backend": parity_n64.THRESHOLDS["kl_min_samples_per_backend"],
    "autocorrelation_lag1_delta_abs_max": parity_n64.THRESHOLDS[
        "autocorrelation_lag1_delta_abs_max"
    ],
}
TERMINAL_VERDICT_PREFIXES = parity_n64.TERMINAL_VERDICT_PREFIXES
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "diverse_topology_parity_n64_ready",
    "n_spins",
    "topologies_tested",
    "per_topology_results",
    "mean_energy_delta",
    "max_energy_delta",
    "kl_divergence",
    "parity_passed",
    "simulator_only",
    "no_tsu_hardware_claim",
    "parity_report_path",
    "focused_tests_passed",
    "honest_verdict",
}

ImportModule = parity_n64.ImportModule
BackendFactory = parity_n64.BackendFactory
ParityIsingCase = parity_n64.ParityIsingCase


def _display_path(path: str | Path) -> str:
    return parity_n64._display_path(path)


def _round_metric(value: float) -> float:
    return parity_n64._round_metric(value)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return parity_n64._write_json(path, payload)


def _write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    parity_n64._write_manifest(path, [dict(row) for row in rows])


def _set_edge(j_matrix: np.ndarray, left: int, right: int, weight: float) -> None:
    j_matrix[int(left), int(right)] = float(weight)
    j_matrix[int(right), int(left)] = float(weight)


def _shared_bias(n_spins: int = 64) -> np.ndarray:
    return np.tile(
        np.array([0.006, -0.010, 0.008, -0.012, 0.010, -0.008, 0.004, -0.006], dtype=np.float64),
        n_spins // 8,
    )


def _complete_case() -> ParityIsingCase:
    n_spins = 64
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for left in range(n_spins):
        for right in range(left + 1, n_spins):
            sign = -1.0 if (left * 37 + right * 19) % 2 else 1.0
            magnitude = 0.0015 + 0.0005 * ((left + 3 * right) % 5)
            _set_edge(j_matrix, left, right, sign * magnitude)
    return ParityIsingCase(
        name="n64_complete",
        topology="complete",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.05,
    )


def _sparse_random_case(seed: int = TOPOLOGY_SEEDS["sparse_random"]) -> ParityIsingCase:
    n_spins = 64
    target_edges = 192
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
        sign = -1.0 if edge_index % 4 == 0 else 1.0
        magnitude = 0.024 + 0.004 * ((left + right + edge_index) % 5)
        _set_edge(j_matrix, left, right, sign * magnitude)
    return ParityIsingCase(
        name=f"n64_sparse_random_seed_{int(seed)}",
        topology="sparse_random",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.05,
    )


def _lattice_case() -> ParityIsingCase:
    rows = 8
    cols = 8
    n_spins = rows * cols
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            right = row * cols + ((col + 1) % cols)
            down = ((row + 1) % rows) * cols + col
            horizontal = 0.055 if (row + col) % 2 == 0 else -0.040
            vertical = -0.050 if (row * 2 + col) % 3 == 0 else 0.035
            _set_edge(j_matrix, node, right, horizontal)
            _set_edge(j_matrix, node, down, vertical)
    return ParityIsingCase(
        name="n64_periodic_8x8_lattice",
        topology="lattice",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.05,
    )


def _scale_free_case(seed: int = TOPOLOGY_SEEDS["scale_free"]) -> ParityIsingCase:
    n_spins = 64
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
        sign = -1.0 if (left + right + edge_index) % 5 == 0 else 1.0
        magnitude = 0.026 + 0.005 * ((left * 3 + right) % 5)
        _set_edge(j_matrix, left, right, sign * magnitude)
    return ParityIsingCase(
        name=f"n64_scale_free_seed_{int(seed)}",
        topology="scale_free",
        j_matrix=j_matrix,
        bias=_shared_bias(n_spins),
        beta=1.05,
    )


def n64_diverse_topology_cases() -> tuple[ParityIsingCase, ...]:
    """Return deterministic complete, sparse-random, lattice, and scale-free cases.

    Spec traces: REQ-SAMPLE-054.
    """

    return (_complete_case(), _sparse_random_case(), _lattice_case(), _scale_free_case())


def _load_exp1531_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1531_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1531_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("diverse_topology_parity_ready") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1531_parity_not_ready",
                "detail": "Exp1531 must be complete, diverse-ready, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def _load_exp1543_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1543_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1543_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_parity_n256_schedule_ready") is True
        and payload.get("parity_passed") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1543_parity_not_ready",
                "detail": "Exp1543 must be complete, schedule-ready, parity-passed, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


def _tolerance_sources(exp1531: Mapping[str, Any], exp1543: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exp1531_path": _display_path(EXP1531_PATH),
        "exp1543_path": _display_path(EXP1543_PATH),
        "exp1531_thresholds": dict(exp1531.get("thresholds") or {}),
        "exp1543_thresholds": dict(exp1543.get("thresholds") or {}),
        "local_reason_for_n64_thresholds": (
            "Exp1544 reuses the n=64 sampled parity thresholds for per-topology "
            "energy, magnetization, KL, and autocorrelation gates, with an "
            "explicit max-energy aggregate gate copied from the n=64 mean-energy bound."
        ),
    }


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before topology parity execution.

    Spec traces: REQ-SAMPLE-054.
    """

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "board_execution": False,
            "synthesis_run": False,
            "bitstream_generated": False,
            "topology_seeds": dict(TOPOLOGY_SEEDS),
        },
        "status": "in_progress",
        "milestone": MILESTONE,
        "diverse_topology_parity_n64_ready": False,
        "n_spins": 64,
        "topologies_tested": [],
        "topologies_passed": [],
        "per_topology_results": {},
        "topology_results": {},
        "mean_energy_delta": None,
        "max_energy_delta": None,
        "kl_divergence": None,
        "parity_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "parity_report_path": _display_path(manifest_path),
        "focused_tests_passed": False,
        "seeds": list(DEFAULT_SEEDS),
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_diverse_topology_parity_n64_simulator_only",
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
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "board_execution": False,
            "synthesis_run": False,
            "bitstream_generated": False,
            "topology_seeds": dict(TOPOLOGY_SEEDS),
            **dict(metadata or {}),
        },
        "status": "blocked",
        "milestone": MILESTONE,
        "diverse_topology_parity_n64_ready": False,
        "n_spins": 64,
        "topologies_tested": list(TOPOLOGIES),
        "topologies_passed": [],
        "per_topology_results": {},
        "topology_results": {},
        "mean_energy_delta": None,
        "max_energy_delta": None,
        "kl_divergence": None,
        "parity_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "parity_report_path": _display_path(manifest_path),
        "focused_tests_passed": False,
        "seeds": list(DEFAULT_SEEDS),
        "sample_count_per_seed": DEFAULT_SAMPLE_COUNT_PER_SEED,
        "warmup": DEFAULT_N_WARMUP,
        "thinning": DEFAULT_STEPS_PER_SAMPLE,
        "thresholds": dict(THRESHOLDS),
        "energy_bin_count": DEFAULT_ENERGY_BIN_COUNT,
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
    """Summarize one n=64 topology/backend/seed sampled chain group.

    Spec traces: REQ-SAMPLE-054.
    """

    row = parity_n64.sampled_backend_row(
        case,
        seed=seed,
        backend_label=backend_label,
        backend_name=backend_name,
        samples=samples,
        schedule=schedule,
    )
    row.update(
        {
            "case_id": f"exp1544:{case.topology}:seed_{int(seed)}:{backend_label}",
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
    """Aggregate topology rows into the Exp 1544 terminal summary row.

    Spec traces: REQ-SAMPLE-054, SCENARIO-SAMPLE-082.
    """

    per_topology_results: dict[str, dict[str, Any]] = {}
    for topology in topologies:
        topology_rows = [dict(row) for row in rows if row.get("topology") == topology]
        summary = parity_n64.summarize_sampled_rows(
            topology_rows,
            seeds=seeds,
            thresholds=thresholds,
            energy_bin_count=energy_bin_count,
        )
        summary["case_id"] = f"exp1544:{topology}:sampled_summary"
        summary["case_type"] = "diverse_topology_n64_topology_summary"
        summary["topology"] = str(topology)
        per_topology_results[str(topology)] = summary

    aggregate = parity_n64.summarize_sampled_rows(
        rows,
        seeds=seeds,
        thresholds=thresholds,
        energy_bin_count=energy_bin_count,
    )
    topologies_passed = [
        topology
        for topology in topologies
        if bool(per_topology_results[str(topology)]["passed_thresholds"])
    ]
    max_energy_delta = max(
        float(result["mean_energy_delta"]) for result in per_topology_results.values()
    )
    aggregate_passed = (
        float(aggregate["mean_energy_delta"]) <= float(thresholds["mean_energy_delta_abs_max"])
        and max_energy_delta <= float(thresholds["max_energy_delta_abs_max"])
        and float(aggregate["kl_divergence"]) <= float(thresholds["kl_divergence_max"])
        and bool(aggregate["passed_thresholds"])
    )
    parity_passed = bool(len(topologies_passed) >= 3 and aggregate_passed)
    return {
        "case_id": "exp1544:diverse_topology_n64:summary",
        "case_type": "diverse_topology_n64_summary",
        "seeds": [int(seed) for seed in seeds],
        "n_spins": 64,
        "topologies_tested": [str(topology) for topology in topologies],
        "topologies_passed": [str(topology) for topology in topologies_passed],
        "per_topology_results": per_topology_results,
        "topology_results": per_topology_results,
        "mean_energy_delta": aggregate["mean_energy_delta"],
        "max_energy_delta": _round_metric(max_energy_delta),
        "kl_divergence": aggregate["kl_divergence"],
        "aggregate_summary": aggregate,
        "thresholds": dict(thresholds),
        "energy_bin_count": int(energy_bin_count),
        "parity_passed": parity_passed,
        "diverse_topology_parity_n64_ready": parity_passed,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
    }


def run_diverse_topology_parity_n64(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1531_path: str | Path = EXP1531_PATH,
    exp1543_path: str | Path = EXP1543_PATH,
    importer: ImportModule = importlib.import_module,
    carnot_backend_factory: BackendFactory = CpuBackend,
    thrml_backend_factory: BackendFactory = ThrmlSamplerBackend,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sample_count_per_seed: int = DEFAULT_SAMPLE_COUNT_PER_SEED,
    n_warmup: int = DEFAULT_N_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
    thresholds: Mapping[str, float] = THRESHOLDS,
    energy_bin_count: int = DEFAULT_ENERGY_BIN_COUNT,
    thrml_seed_offset: int = DEFAULT_THRML_SEED_OFFSET,
    focused_tests_passed: bool = False,
) -> dict[str, Any]:
    """Run n=64 sampled parity across four topology families and write evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    exp1531_ready, exp1531_payload, exp1531_blocker = _load_exp1531_ready(exp1531_path)
    if not exp1531_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[exp1531_blocker or {"blocker": "exp1531_parity_not_ready", "detail": ""}],
                verdict="complete_thrml_diverse_topology_parity_n64_blocked_exp1531_no_tsu_hardware_claim",
                metadata={"exp1531_status": exp1531_payload.get("status")},
            ),
        )

    exp1543_ready, exp1543_payload, exp1543_blocker = _load_exp1543_ready(exp1543_path)
    if not exp1543_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[exp1543_blocker or {"blocker": "exp1543_parity_not_ready", "detail": ""}],
                verdict="complete_thrml_diverse_topology_parity_n64_blocked_exp1543_no_tsu_hardware_claim",
                metadata={
                    "exp1531_status": exp1531_payload.get("status"),
                    "exp1543_status": exp1543_payload.get("status"),
                },
            ),
        )

    thrml_details, import_blocker = parity_n256._import_thrml(importer)
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                blockers=[import_blocker],
                verdict=(
                    "complete_thrml_diverse_topology_parity_n64_blocked_simulator_dependency_"
                    "no_tsu_hardware_claim"
                ),
                metadata={
                    "exp1531_status": exp1531_payload.get("status"),
                    "exp1543_status": exp1543_payload.get("status"),
                },
            ),
        )

    cases = n64_diverse_topology_cases()
    rows: list[dict[str, Any]] = []
    for case in cases:
        schedule = {
            "beta": float(case.beta),
            "n_warmup": int(n_warmup),
            "steps_per_sample": int(steps_per_sample),
            "use_checkerboard": True,
        }
        for seed in seeds:
            carnot_seed = int(seed)
            thrml_seed = carnot_seed + int(thrml_seed_offset)
            carnot_backend = carnot_backend_factory(carnot_seed)
            thrml_backend = thrml_backend_factory(thrml_seed)
            carnot_samples = np.asarray(
                carnot_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
            )
            thrml_samples = np.asarray(
                thrml_backend.sample(case.bias, case.j_matrix, int(sample_count_per_seed), schedule)
            )
            rows.append(
                sampled_topology_backend_row(
                    case,
                    seed=carnot_seed,
                    backend_label="carnot",
                    backend_name=str(getattr(carnot_backend, "backend_name", "<unknown>")),
                    samples=carnot_samples,
                    schedule=schedule,
                )
            )
            rows.append(
                sampled_topology_backend_row(
                    case,
                    seed=thrml_seed,
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
    passed = bool(summary_row["parity_passed"])
    failed_topologies = [
        topology
        for topology in summary_row["topologies_tested"]
        if topology not in summary_row["topologies_passed"]
    ]
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "board_execution": False,
            "synthesis_run": False,
            "bitstream_generated": False,
            "exp1531_status": exp1531_payload.get("status"),
            "exp1543_status": exp1543_payload.get("status"),
            "thrml_execution_path": "local_software_simulator_or_cpu_fallback",
            "independent_rng_streams": int(thrml_seed_offset) != 0,
            "topology_seeds": dict(TOPOLOGY_SEEDS),
            **thrml_details,
        },
        "status": "complete",
        "milestone": MILESTONE,
        "diverse_topology_parity_n64_ready": passed,
        "n_spins": 64,
        "topologies_tested": summary_row["topologies_tested"],
        "topologies_passed": summary_row["topologies_passed"],
        "per_topology_results": summary_row["per_topology_results"],
        "topology_results": summary_row["topology_results"],
        "mean_energy_delta": summary_row["mean_energy_delta"],
        "max_energy_delta": summary_row["max_energy_delta"],
        "kl_divergence": summary_row["kl_divergence"],
        "parity_passed": passed,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "parity_report_path": _display_path(manifest_path),
        "focused_tests_passed": bool(focused_tests_passed),
        "seeds": [int(seed) for seed in seeds],
        "thrml_seed_offset": int(thrml_seed_offset),
        "sample_count_per_seed": int(sample_count_per_seed),
        "warmup": int(n_warmup),
        "thinning": int(steps_per_sample),
        "thresholds": dict(thresholds),
        "tolerance_sources": _tolerance_sources(exp1531_payload, exp1543_payload),
        "energy_bin_count": int(energy_bin_count),
        "n_samples_per_backend": int(summary_row["aggregate_summary"]["n_samples_per_backend"]),
        "aggregate_summary": summary_row["aggregate_summary"],
        "mean_energy_delta_by_topology": {
            topology: result["mean_energy_delta"]
            for topology, result in summary_row["per_topology_results"].items()
        },
        "kl_divergence_by_topology": {
            topology: result["kl_divergence"]
            for topology, result in summary_row["per_topology_results"].items()
        },
        "blockers": []
        if passed
        else [
            {
                "blocker": "topology_parity_threshold_failed",
                "detail": ",".join(failed_topologies) if failed_topologies else "aggregate_metrics",
            }
        ],
        "honest_verdict": (
            "complete_thrml_diverse_topology_parity_n64_passed_simulator_only_no_tsu_hardware_claim"
            if passed
            else "complete_thrml_diverse_topology_parity_n64_failed_thresholds_simulator_only_no_tsu_hardware_claim"
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 1544 schema, metric gates, and no-hardware boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1544")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1544")
    if int(artifact.get("n_spins") or 0) != 64:
        raise ValueError("Exp 1544 artifacts must remain at n_spins=64")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("diverse_topology_parity_n64_ready") is True or artifact.get("parity_passed") is True:
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        topologies_tested = list(artifact.get("topologies_tested") or [])
        topologies_passed = list(artifact.get("topologies_passed") or [])
        per_topology_results = dict(artifact.get("per_topology_results") or {})
        gates_ok = (
            artifact.get("diverse_topology_parity_n64_ready") is True
            and artifact.get("parity_passed") is True
            and topologies_tested == list(TOPOLOGIES)
            and len(topologies_passed) >= 3
            and set(topologies_passed).issubset(set(topologies_tested))
            and set(per_topology_results) == set(topologies_tested)
            and bool(artifact.get("parity_report_path"))
            and float(artifact.get("mean_energy_delta") or 0.0)
            <= float(thresholds["mean_energy_delta_abs_max"])
            and float(artifact.get("max_energy_delta") or 0.0)
            <= float(thresholds["max_energy_delta_abs_max"])
            and float(artifact.get("kl_divergence") or 0.0)
            <= float(thresholds["kl_divergence_max"])
        )
        for topology in topologies_passed:
            result = dict(per_topology_results.get(topology) or {})
            autocorr = result.get("autocorrelation_summary")
            autocorr_present = isinstance(autocorr, Mapping) and "lag1_delta" in autocorr
            mean_energy_gate = (
                float(result.get("mean_energy_delta") or 0.0)
                <= float(thresholds["mean_energy_delta_abs_max"])
                or float(result.get("mean_energy_delta_percent") or 0.0)
                <= float(thresholds["mean_energy_delta_percent_max"])
            )
            gates_ok = (
                gates_ok
                and bool(result.get("passed_thresholds"))
                and int(result.get("n_samples_per_backend") or 0)
                >= int(thresholds["kl_min_samples_per_backend"])
                and bool(result.get("kl_estimate_stable"))
                and bool(result.get("stability_diagnostics_present"))
                and autocorr_present
                and mean_energy_gate
                and float(result.get("magnetization_delta") or 0.0)
                <= float(thresholds["magnetization_delta_abs_max"])
                and float(result.get("kl_divergence") or 0.0)
                <= float(thresholds["kl_divergence_max"])
                and float(dict(autocorr).get("lag1_delta") or 0.0)
                <= float(thresholds["autocorrelation_lag1_delta_abs_max"])
            )
        if not gates_ok:
            raise ValueError("n=64 readiness requires passing diverse-topology parity metrics")
