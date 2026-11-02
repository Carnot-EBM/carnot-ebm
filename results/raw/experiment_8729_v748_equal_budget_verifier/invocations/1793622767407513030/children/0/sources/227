"""Exp 1347 THRML compatibility parity audit.

**Researcher summary:**
    This module checks whether Carnot's tiny hardware-portability energy cases
    can be represented by whatever THRML Python API is already available on the
    local machine. It deliberately does not install packages, fetch code, or
    label CPU/JAX simulation as Extropic TSU hardware execution.

**Detailed explanation for engineers:**
    The audit has two paths. If ``import thrml`` fails, it records a complete
    blocked artifact with mapping notes only. If THRML imports and exposes a
    tiny Ising energy API, the audit builds the same four-spin Ising case used
    by Exp 1320 and compares every enumerated spin-state energy against Carnot's
    local Hamiltonian. The KAN case is recorded as a portability note because
    Carnot's current tiny KAN energy is spline/unary-factor shaped, and this
    audit should not invent a THRML hypergraph API that is not locally present.

Spec refs: REQ-SAMPLE-041, SCENARIO-SAMPLE-069.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.analysis.pbit_sampler_portability import (
    IsingCase,
    enumerate_spin_states,
    ising_energy,
    tiny_ising_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1347_thrml_compatibility_parity_audit.json"

EXPERIMENT_ID = 1347
SCHEMA = "thrml_compatibility_parity_audit_v1"
DEFAULT_RUN_DATE = "20260505"

HONEST_VERDICTS = {
    "blocked_thrml_not_importable_no_hardware_claim",
    "blocked_thrml_api_missing_no_hardware_claim",
    "local_thrml_parity_measured_no_tsu_execution_claim",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "thrml_import_available",
    "cases_attempted",
    "energy_parity_max_abs_error",
    "sample_quality_proxy",
    "missing_api_or_dependency",
    "tsu_mapping_notes",
    "hardware_claim_allowed",
    "honest_verdict",
}

ImportModule = Callable[[str], Any]


@dataclass(frozen=True)
class ThrmlProbe:
    """Result of probing the already-installed THRML Python API."""

    import_available: bool
    module: Any | None
    models_module: Any | None
    version: str | None
    missing_api_or_dependency: str | None


class MissingThrmlApi(RuntimeError):
    """Raised when local THRML imports but lacks the API this audit needs."""


def probe_thrml(import_module: ImportModule = importlib.import_module) -> ThrmlProbe:
    """Probe local THRML import availability without installing anything.

    Spec: REQ-SAMPLE-041
    """
    try:
        thrml_module = import_module("thrml")
    except ModuleNotFoundError:
        return ThrmlProbe(
            import_available=False,
            module=None,
            models_module=None,
            version=None,
            missing_api_or_dependency=(
                "thrml Python package is not importable in the local environment; "
                "no package installation or network access was attempted"
            ),
        )

    version = str(getattr(thrml_module, "__version__", "unknown"))
    try:
        models_module = import_module("thrml.models")
    except ModuleNotFoundError:
        return ThrmlProbe(
            import_available=True,
            module=thrml_module,
            models_module=None,
            version=version,
            missing_api_or_dependency="thrml imports, but thrml.models is not importable locally",
        )

    return ThrmlProbe(
        import_available=True,
        module=thrml_module,
        models_module=models_module,
        version=version,
        missing_api_or_dependency=None,
    )


def _round_metric(value: float) -> float:
    return round(float(value), 12)


def _ising_edge_payload(case: IsingCase) -> tuple[list[tuple[int, int]], np.ndarray]:
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(case.n_spins):
        for j in range(i + 1, case.n_spins):
            weight = float(case.j_matrix[i, j])
            if weight != 0.0:
                edges.append((i, j))
                weights.append(weight)
    return edges, np.asarray(weights, dtype=np.float64)


def _build_thrml_ising_model(
    probe: ThrmlProbe, case: IsingCase
) -> tuple[Any, list[Any], list[tuple[int, int]]]:
    if probe.module is None or probe.models_module is None:
        raise MissingThrmlApi("THRML model construction requires thrml and thrml.models")

    spin_node_cls = getattr(probe.module, "SpinNode", None)
    ising_cls = getattr(probe.models_module, "IsingEBM", None)
    if spin_node_cls is None or ising_cls is None:
        raise MissingThrmlApi("local THRML API lacks SpinNode or models.IsingEBM")

    nodes = [spin_node_cls() for _ in range(case.n_spins)]
    edge_indices, weights = _ising_edge_payload(case)
    node_edges = [(nodes[i], nodes[j]) for i, j in edge_indices]
    model = ising_cls(
        nodes,
        node_edges,
        np.asarray(case.bias, dtype=np.float64),
        weights,
        float(case.beta),
    )
    return model, nodes, edge_indices


def measure_tiny_ising_thrml_parity(probe: ThrmlProbe, case: IsingCase | None = None) -> dict[str, Any]:
    """Measure exact energy parity on the Exp 1320 four-spin Ising case.

    Spec: REQ-SAMPLE-041, SCENARIO-SAMPLE-069
    """
    ising_case = tiny_ising_case() if case is None else case
    model, _nodes, edge_indices = _build_thrml_ising_model(probe, ising_case)
    energy_method = getattr(model, "energy", None)
    if energy_method is None:
        raise MissingThrmlApi("local THRML IsingEBM lacks an energy(spins) method")

    states = enumerate_spin_states(ising_case.n_spins)
    errors = []
    for state in states:
        local_energy = ising_energy(ising_case, state)
        thrml_energy = float(energy_method(np.asarray(state, dtype=np.float64)))
        errors.append(abs(local_energy - thrml_energy))

    max_error = max(errors) if errors else 0.0
    return {
        "case": f"tiny_ising:{ising_case.name}",
        "status": "parity_measured",
        "n_spins": ising_case.n_spins,
        "state_count": int(len(states)),
        "edge_count": int(len(edge_indices)),
        "max_abs_error": _round_metric(max_error),
    }


def _kan_mapping_note(thrml_available: bool) -> dict[str, Any]:
    reason = (
        "THRML unavailable, so the tiny KAN energy is recorded as mapping notes only"
        if not thrml_available
        else (
            "tiny KAN/KAEM energy is a spline or unary-factor energy; this audit found "
            "only the local Ising parity surface and did not assume a generic THRML "
            "hypergraph factor API"
        )
    )
    return {
        "case": "tiny_kan:univariate_kaem_note",
        "status": "mapping_notes_only",
        "reason": reason,
    }


def _blocked_ising_case(reason: str) -> dict[str, Any]:
    return {
        "case": "tiny_ising:n4_signed_ring_chord",
        "status": "blocked_missing_api_or_dependency",
        "reason": reason,
    }


def _sample_quality_proxy(max_error: float, state_count: int) -> dict[str, Any]:
    proxy_value = 1.0 / (1.0 + abs(float(max_error)))
    return {
        "proxy_name": "exact_energy_parity_score",
        "proxy_value": _round_metric(proxy_value),
        "state_count": int(state_count),
        "sample_count": 0,
        "notes": "Exact enumerated energy parity proxy; not a stochastic sample-quality measurement.",
    }


def _mapping_notes(parity_measured: bool, missing: str | None) -> str:
    if parity_measured:
        return (
            "Tiny Ising mapped through the local THRML IsingEBM energy API with Carnot's "
            "Hamiltonian sign convention checked on every enumerated state. Tiny KAN "
            "remains a mapping note because no local generic THRML factor API was used. "
            "No TSU hardware execution or acceleration claim is made."
        )
    return (
        "No THRML-backed parity execution occurred. Carnot's tiny Ising case would map "
        "bias terms to unary fields and non-zero upper-triangle J entries to pairwise "
        "SpinNode edges. The tiny KAN/KAEM case needs unary spline or generic factor "
        f"support before parity can be attempted. Missing local prerequisite: {missing}."
    )


def build_artifact(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = DEFAULT_RUN_DATE,
    import_module: ImportModule = importlib.import_module,
) -> dict[str, Any]:
    """Build the complete Exp 1347 audit artifact.

    Spec: REQ-SAMPLE-041, SCENARIO-SAMPLE-069
    """
    probe = probe_thrml(import_module=import_module)
    cases_attempted: list[dict[str, Any]]
    parity_error: float | None = None
    sample_proxy: dict[str, Any] | None = None
    missing = probe.missing_api_or_dependency

    if not probe.import_available:
        cases_attempted = [
            _blocked_ising_case(str(missing)),
            _kan_mapping_note(thrml_available=False),
        ]
        verdict = "blocked_thrml_not_importable_no_hardware_claim"
    else:
        try:
            ising_result = measure_tiny_ising_thrml_parity(probe)
        except MissingThrmlApi as exc:
            missing = str(exc)
            cases_attempted = [
                _blocked_ising_case(missing),
                _kan_mapping_note(thrml_available=True),
            ]
            verdict = "blocked_thrml_api_missing_no_hardware_claim"
        else:
            parity_error = float(ising_result["max_abs_error"])
            sample_proxy = _sample_quality_proxy(parity_error, int(ising_result["state_count"]))
            cases_attempted = [ising_result, _kan_mapping_note(thrml_available=True)]
            verdict = "local_thrml_parity_measured_no_tsu_execution_claim"

    parity_measured = parity_error is not None
    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "tsu_hardware_execution_confirmed": False,
        },
        "status": "complete",
        "thrml_import_available": probe.import_available,
        "thrml_version": probe.version,
        "cases_attempted": cases_attempted,
        "energy_parity_max_abs_error": (
            _round_metric(parity_error) if parity_error is not None else None
        ),
        "sample_quality_proxy": sample_proxy,
        "missing_api_or_dependency": None if parity_measured else missing,
        "tsu_mapping_notes": _mapping_notes(parity_measured, missing),
        "hardware_claim_allowed": bool(parity_measured and probe.import_available),
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the public schema and hardware-claim gate.

    Spec: REQ-SAMPLE-041
    """
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["honest_verdict"] not in HONEST_VERDICTS:
        raise ValueError(f"unknown honest_verdict: {artifact['honest_verdict']}")
    if artifact["hardware_claim_allowed"] and (
        not artifact["thrml_import_available"]
        or artifact["energy_parity_max_abs_error"] is None
        or artifact["sample_quality_proxy"] is None
    ):
        raise ValueError("hardware_claim_allowed requires local THRML parity measurement")
    if not artifact["thrml_import_available"] and (
        artifact["energy_parity_max_abs_error"] is not None
        or artifact["sample_quality_proxy"] is not None
        or artifact["hardware_claim_allowed"]
    ):
        raise ValueError("unavailable THRML artifacts must leave parity and claims unset")


def write_artifact(
    path: str | Path = DELIVERABLE_PATH, artifact: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Write a validated audit artifact JSON and return the payload.

    Spec: SCENARIO-SAMPLE-069
    """
    payload = dict(artifact or build_artifact())
    validate_artifact(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

