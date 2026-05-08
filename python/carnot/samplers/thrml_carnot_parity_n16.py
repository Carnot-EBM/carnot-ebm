"""Exp 1527 exact n=16 THRML/Carnot simulator parity.

This module is the second THRML/Carnot parity scaling step. It reuses the
Exp 1526 parity machinery for THRML model construction, exact Boltzmann
comparison, and fixed-seed sampling rows, while changing the artifact contract
and Ising case to the n=16 run requested for 20260508. The run is deliberately
software-only: it compares Carnot's CPU simulator path with the local THRML
software API/fallback and records no Extropic TSU hardware execution claim.

Spec traces: REQ-SAMPLE-048, SCENARIO-SAMPLE-076.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from carnot.samplers.backend import CpuBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend
from carnot.samplers import thrml_carnot_parity_n8 as parity_n8

PROJECT_ROOT = parity_n8.PROJECT_ROOT
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1527_thrml_carnot_parity_n16.json"
PARITY_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_carnot_parity_n16_1527.jsonl"
EXP1526_PATH = PROJECT_ROOT / "results" / "experiment_1526_thrml_carnot_parity_n8.json"

EXPERIMENT_ID = 1527
RUN_DATE = "20260508"
SCHEMA = "thrml_carnot_parity_n16_v1"
DEFAULT_SEED = parity_n8.DEFAULT_SEED
DEFAULT_SAMPLE_COUNT = parity_n8.DEFAULT_SAMPLE_COUNT
DEFAULT_N_WARMUP = parity_n8.DEFAULT_N_WARMUP
DEFAULT_STEPS_PER_SAMPLE = parity_n8.DEFAULT_STEPS_PER_SAMPLE

THRESHOLDS = dict(parity_n8.THRESHOLDS)
TERMINAL_VERDICT_PREFIXES = parity_n8.TERMINAL_VERDICT_PREFIXES
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_parity_n16_passed",
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
    "sample_mean_energy_delta",
    "parity_manifest_path",
    "blockers",
    "honest_verdict",
}

ImportModule = parity_n8.ImportModule
BackendFactory = parity_n8.BackendFactory
ParityIsingCase = parity_n8.ParityIsingCase
MissingThrmlApi = parity_n8.MissingThrmlApi
enumerate_spin_states = parity_n8.enumerate_spin_states
ising_energy = parity_n8.ising_energy


def n16_signed_ring_chord_case() -> ParityIsingCase:
    """Return the deterministic 16-spin signed ring-chord parity case.

    The topology intentionally mirrors Exp 1526: each spin has one signed ring
    edge and one signed distance-two chord. That gives 32 non-zero undirected
    edges, enough mixed-sign structure to catch convention drift while keeping
    exact enumeration bounded at 65,536 states.

    Spec traces: REQ-SAMPLE-048.
    """

    n_spins = 16
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    ring_weights = np.array(
        [
            0.62,
            -0.44,
            0.51,
            -0.38,
            0.47,
            -0.53,
            0.36,
            -0.41,
            0.58,
            -0.49,
            0.43,
            -0.34,
            0.55,
            -0.46,
            0.39,
            -0.32,
        ],
        dtype=np.float64,
    )
    chord_weights = np.array(
        [
            -0.31,
            0.27,
            -0.46,
            0.22,
            -0.35,
            0.18,
            -0.29,
            0.26,
            -0.24,
            0.33,
            -0.28,
            0.21,
            -0.37,
            0.19,
            -0.25,
            0.30,
        ],
        dtype=np.float64,
    )
    for idx, weight in enumerate(ring_weights):
        left, right = idx, (idx + 1) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    for idx, weight in enumerate(chord_weights):
        left, right = idx, (idx + 2) % n_spins
        j_matrix[left, right] = j_matrix[right, left] = float(weight)
    bias = np.array(
        [
            0.12,
            -0.07,
            0.05,
            -0.10,
            0.09,
            -0.03,
            0.04,
            -0.02,
            0.08,
            -0.06,
            0.03,
            -0.11,
            0.07,
            -0.04,
            0.06,
            -0.05,
        ],
        dtype=np.float64,
    )
    return ParityIsingCase(
        name="n16_signed_ring_chord",
        topology="signed_ring_chord",
        j_matrix=j_matrix,
        bias=bias,
        beta=1.10,
    )


def _display_path(path: str | Path) -> str:
    return parity_n8._display_path(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return parity_n8._write_json(path, payload)


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before import probing or parity execution.

    Spec traces: REQ-SAMPLE-048.
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
        "thrml_parity_n16_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 16,
        "exact_states_enumerated": 0,
        "topology": "signed_ring_chord",
        "seed": DEFAULT_SEED,
        "carnot_partition_function": None,
        "thrml_partition_function": None,
        "partition_relative_error": None,
        "mean_energy_delta": None,
        "kl_divergence": None,
        "sample_mean_energy_delta": None,
        "thresholds": dict(THRESHOLDS),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [{"blocker": "parity_run_not_completed", "detail": "bootstrap artifact only"}],
        "honest_verdict": "success_in_progress_thrml_carnot_parity_n16_simulator_only",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _load_exp1526_ready(path: str | Path) -> tuple[bool, dict[str, Any], dict[str, str] | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return False, {}, {"blocker": "exp1526_evidence_missing", "detail": str(exc)}
    except json.JSONDecodeError as exc:
        return False, {}, {"blocker": "exp1526_evidence_malformed", "detail": str(exc)}
    ready = (
        payload.get("status") == "complete"
        and payload.get("thrml_parity_n8_passed") is True
        and payload.get("simulator_only") is True
        and payload.get("no_tsu_hardware_claim") is True
    )
    if not ready:
        return (
            False,
            payload,
            {
                "blocker": "exp1526_parity_not_passed",
                "detail": "Exp1526 must be complete, n=8-passed, simulator-only, and no-TSU-claim",
            },
        )
    return True, payload, None


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
        "thrml_parity_n16_passed": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "n_spins": 16,
        "exact_states_enumerated": 0,
        "topology": "signed_ring_chord",
        "seed": int(seed),
        "carnot_partition_function": None,
        "thrml_partition_function": None,
        "partition_relative_error": None,
        "mean_energy_delta": None,
        "kl_divergence": None,
        "sample_mean_energy_delta": None,
        "thresholds": dict(THRESHOLDS),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def exact_parity_metrics(thrml_modules: Any, case: ParityIsingCase) -> dict[str, Any]:
    """Enumerate all n=16 states and compare exact Carnot/THRML distributions."""

    row = dict(parity_n8.exact_parity_metrics(thrml_modules, case))
    row["case_id"] = f"exp1527:{case.name}:exact_distribution"
    row["thresholds"] = dict(THRESHOLDS)
    return row


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
    """Record fixed-seed n=16 software sampling as secondary parity evidence."""

    row = dict(
        parity_n8.sampling_metrics(
            case,
            seed=seed,
            carnot_backend_factory=carnot_backend_factory,
            thrml_backend_factory=thrml_backend_factory,
            sample_count=sample_count,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
        )
    )
    row["case_id"] = f"exp1527:{case.name}:fixed_seed_sampling"
    return row


def _write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    parity_n8._write_manifest(path, rows)


def _passed_thresholds(exact_row: Mapping[str, Any], sample_row: Mapping[str, Any]) -> bool:
    return (
        parity_n8._passed_exact_thresholds(exact_row)
        and float(sample_row["mean_energy_delta"]) <= THRESHOLDS["sample_mean_energy_delta_abs_max"]
    )


def run_parity_n16(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = PARITY_MANIFEST_PATH,
    exp1526_path: str | Path = EXP1526_PATH,
    importer: ImportModule = importlib.import_module,
    carnot_backend_factory: BackendFactory = CpuBackend,
    thrml_backend_factory: BackendFactory = ThrmlSamplerBackend,
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    n_warmup: int = DEFAULT_N_WARMUP,
    steps_per_sample: int = DEFAULT_STEPS_PER_SAMPLE,
) -> dict[str, Any]:
    """Run exact n=16 THRML/Carnot parity and write JSON/JSONL evidence."""

    write_in_progress_artifact(output_path, manifest_path)
    exp1526_ready, exp1526_payload, exp1526_blocker = _load_exp1526_ready(exp1526_path)
    if not exp1526_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seed=seed,
                blockers=[
                    exp1526_blocker
                    or {
                        "blocker": "exp1526_parity_not_passed",
                        "detail": "unknown Exp1526 blocker",
                    }
                ],
                verdict="complete_thrml_carnot_parity_n16_blocked_exp1526_no_tsu_hardware_claim",
                metadata={"exp1526_status": exp1526_payload.get("status")},
            ),
        )
    thrml_modules, thrml_details, import_blocker = parity_n8._import_thrml(importer)
    if import_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                seed=seed,
                blockers=[import_blocker],
                verdict="complete_thrml_carnot_parity_n16_blocked_simulator_dependency_no_tsu_hardware_claim",
                metadata={"exp1526_status": exp1526_payload.get("status")},
            ),
        )
    case = n16_signed_ring_chord_case()
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
                verdict="complete_thrml_carnot_parity_n16_blocked_thrml_api_no_tsu_hardware_claim",
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
    passed = _passed_thresholds(exact_row, sample_row)
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            "exp1526_status": exp1526_payload.get("status"),
            **thrml_details,
        },
        "status": "complete",
        "thrml_parity_n16_passed": passed,
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
        "sample_mean_energy_delta": sample_row["mean_energy_delta"],
        "thresholds": dict(THRESHOLDS),
        "parity_manifest_path": _display_path(manifest_path),
        "blockers": [] if passed else [{"blocker": "parity_threshold_failed", "detail": str(rows)}],
        "honest_verdict": (
            "complete_thrml_carnot_parity_n16_passed_no_tsu_hardware_claim"
            if passed
            else "complete_thrml_carnot_parity_n16_failed_thresholds_no_tsu_hardware_claim"
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, pass gates, and no-TSU claim boundaries."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("status") not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {artifact.get('status')!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1527")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1527")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("thrml_parity_n16_passed") is True:
        thresholds = dict(artifact.get("thresholds") or THRESHOLDS)
        pass_metrics = (
            int(artifact.get("exact_states_enumerated") or 0) == 65536
            and float(artifact.get("partition_relative_error") or 0.0)
            <= float(thresholds["partition_relative_error_max"])
            and float(artifact.get("mean_energy_delta") or 0.0)
            <= float(thresholds["mean_energy_delta_abs_max"])
            and float(artifact.get("kl_divergence") or 0.0)
            <= float(thresholds["kl_divergence_max"])
            and float(artifact.get("sample_mean_energy_delta") or 0.0)
            <= float(thresholds["sample_mean_energy_delta_abs_max"])
        )
        if not pass_metrics:
            raise ValueError("thrml_parity_n16_passed requires exact and sample pass metrics")
