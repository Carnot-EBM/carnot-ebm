"""Exp 1347 THRML compatibility audit for tiny Ising/KAN energy cases.

This module is deliberately conservative: it checks only what is available on
the local filesystem and import path, attempts a tiny THRML Ising energy mapping
only when the public Python API can be imported, and records KAN mapping limits
instead of inventing an acceleration result.

Spec refs: REQ-SAMPLE-041, SCENARIO-SAMPLE-069.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from carnot.analysis.pbit_sampler_portability import (
    enumerate_spin_states,
    exact_boltzmann_distribution,
    ising_energy,
    kl_divergence,
    tiny_ising_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1347_thrml_compatibility_parity_audit.json"

EXPERIMENT_ID = 1347
SCHEMA = "thrml_compatibility_parity_audit_v1"
RUN_DATE = "20260505"

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

HONEST_VERDICTS = {
    "thrml_unavailable_mapping_notes_only_no_hardware_claim",
    "thrml_import_available_but_api_incompatible_no_hardware_claim",
    "thrml_energy_parity_measured_no_tsu_hardware_execution",
}


@dataclass(frozen=True)
class ThrmlProbeResult:
    """Result of checking for an importable local THRML package.

    The probe separates "a checkout exists" from "Python can import it". That
    distinction matters because a local THRML source tree still cannot back a
    Carnot parity run when one of its runtime dependencies is missing.

    Spec refs: REQ-SAMPLE-041.
    """

    import_available: bool
    module: ModuleType | Any | None
    version: str | None
    import_source: str
    local_package_path: str | None
    missing_api_or_dependency: str | None


def _default_local_package_candidates(project_root: str | Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """Return likely local THRML checkout paths without scanning the network."""
    root = Path(project_root).resolve()
    return (root.parent / "thrml",)


def _describe_import_exception(exc: BaseException) -> str:
    """Convert an import exception into an artifact-safe dependency note."""
    if isinstance(exc, importlib.metadata.PackageNotFoundError):
        return f"missing Python package metadata while importing THRML: {exc}"
    if isinstance(exc, ModuleNotFoundError):
        missing_name = exc.name or "unknown"
        return f"missing Python module while importing THRML: {missing_name}"
    text = str(exc).strip()
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__


def _module_version(module: ModuleType | Any) -> str:
    """Read a THRML version string without requiring package installation."""
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        return str(importlib.metadata.version("thrml"))
    except Exception:
        return "unknown"


def _import_from_local_path(
    candidate: Path,
    importer: Callable[[str], ModuleType | Any],
) -> tuple[ModuleType | Any | None, BaseException | None]:
    """Try importing THRML from a local source checkout and restore sys.path."""
    old_path = list(sys.path)
    old_modules = {
        name: module for name, module in sys.modules.items() if name == "thrml" or name.startswith("thrml.")
    }
    try:
        sys.path.insert(0, str(candidate))
        module = importer("thrml")
        return module, None
    except BaseException as exc:
        for name in [name for name in sys.modules if name == "thrml" or name.startswith("thrml.")]:
            if name not in old_modules:
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        return None, exc
    finally:
        sys.path[:] = old_path


def probe_thrml(
    *,
    importer: Callable[[str], ModuleType | Any] = importlib.import_module,
    project_root: str | Path = PROJECT_ROOT,
    local_package_candidates: Iterable[str | Path] | None = None,
) -> ThrmlProbeResult:
    """Probe THRML availability without installing packages or using the network.

    Spec refs: REQ-SAMPLE-041.
    """
    try:
        module = importer("thrml")
        return ThrmlProbeResult(
            import_available=True,
            module=module,
            version=_module_version(module),
            import_source="python_import_path",
            local_package_path=None,
            missing_api_or_dependency=None,
        )
    except BaseException as import_exc:
        initial_error = import_exc

    candidates = (
        tuple(Path(path) for path in local_package_candidates)
        if local_package_candidates is not None
        else _default_local_package_candidates(project_root)
    )
    for candidate in candidates:
        package_init = candidate / "thrml" / "__init__.py"
        if not package_init.exists():
            continue
        module, local_error = _import_from_local_path(candidate, importer)
        if module is not None:
            return ThrmlProbeResult(
                import_available=True,
                module=module,
                version=_module_version(module),
                import_source="local_source_checkout",
                local_package_path=str(candidate),
                missing_api_or_dependency=None,
            )
        return ThrmlProbeResult(
            import_available=False,
            module=None,
            version=None,
            import_source="local_source_checkout_import_failed",
            local_package_path=str(candidate),
            missing_api_or_dependency=(
                f"local THRML package found at {candidate}, but import failed: "
                f"{_describe_import_exception(local_error or initial_error)}"
            ),
        )

    return ThrmlProbeResult(
        import_available=False,
        module=None,
        version=None,
        import_source="not_importable",
        local_package_path=None,
        missing_api_or_dependency=_describe_import_exception(initial_error),
    )


def _require_thrml_api(module: ModuleType | Any) -> tuple[type, type, type]:
    """Return the THRML objects needed for the tiny Ising parity run."""
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


def _as_float(value: Any) -> float:
    """Convert scalar NumPy/JAX/Python values to plain float for JSON metrics."""
    return float(np.asarray(value))


def _measure_tiny_ising_parity(module: ModuleType | Any) -> tuple[dict[str, Any], float, dict[str, Any]]:
    """Compare local tiny Ising energies against a THRML Ising representation."""
    spin_node, block_cls, ising_ebm = _require_thrml_api(module)
    case = tiny_ising_case()
    states = enumerate_spin_states(case.n_spins)
    nodes = [spin_node() for _ in range(case.n_spins)]
    edge_pairs: list[tuple[int, int]] = []
    edges: list[tuple[Any, Any]] = []
    weights: list[float] = []
    for i in range(case.n_spins):
        for j in range(i + 1, case.n_spins):
            weight = float(case.j_matrix[i, j])
            edge_pairs.append((i, j))
            edges.append((nodes[i], nodes[j]))
            weights.append(weight)

    model = ising_ebm(
        nodes,
        edges,
        np.asarray(case.bias, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        1.0,
    )
    read_block = block_cls(nodes)

    local_energies = np.array([ising_energy(case, state) for state in states], dtype=np.float64)
    thrml_energies = []
    for state in states:
        bool_state = np.asarray(state == 1, dtype=bool)
        thrml_energies.append(_as_float(model.energy([bool_state], [read_block])))
    thrml_energy_array = np.asarray(thrml_energies, dtype=np.float64)

    max_abs_error = float(np.max(np.abs(local_energies - thrml_energy_array)))
    local_distribution = exact_boltzmann_distribution(case, states)
    shifted = -case.beta * thrml_energy_array
    shifted -= float(np.max(shifted))
    thrml_distribution = np.exp(shifted)
    thrml_distribution = thrml_distribution / float(thrml_distribution.sum())
    kl_to_local = kl_divergence(thrml_distribution, local_distribution)

    case_record = {
        "case_type": "ising",
        "name": case.name,
        "status": "parity_measured",
        "n_spins": case.n_spins,
        "state_count": int(states.shape[0]),
        "edge_count": len(edge_pairs),
        "edge_pairs": [[int(i), int(j)] for i, j in edge_pairs],
        "max_abs_energy_error": round(max_abs_error, 15),
        "energy_convention": "THRML beta set to 1.0; Carnot beta applied only in distribution proxy.",
    }
    sample_quality_proxy = {
        "proxy": "exact_boltzmann_kl_from_energy_parity",
        "kl_to_local_exact": round(float(kl_to_local), 15),
        "state_count": int(states.shape[0]),
        "sampling_executed": False,
    }
    return case_record, max_abs_error, sample_quality_proxy


def _kan_case_record(status: str) -> dict[str, Any]:
    """Record the tiny KAN mapping conclusion without pretending it ran on THRML."""
    return {
        "case_type": "kan",
        "name": "tiny_local_kan_spline_energy",
        "status": status,
        "local_reference": "python/carnot/models/kan.py::KANEnergyFunction.energy",
        "notes": (
            "Current local KAN energy uses spline edge and bias functions; the "
            "checked THRML Ising API represents linear spin biases and pairwise "
            "spin products, so this audit records KAN mapping notes only."
        ),
    }


def _missing_case_records(reason: str) -> list[dict[str, Any]]:
    """Return case records for a notes-only audit path."""
    return [
        {
            "case_type": "ising",
            "name": tiny_ising_case().name,
            "status": "not_attempted_missing_thrml",
            "reason": reason,
            "local_reference": "python/carnot/analysis/pbit_sampler_portability.py::tiny_ising_case",
        },
        _kan_case_record("mapping_notes_only"),
    ]


def _mapping_notes(probe: ThrmlProbeResult, parity_measured: bool) -> str:
    """Build the artifact's top-level TSU mapping note."""
    if parity_measured:
        return (
            "Tiny Ising energy parity was measured through a local THRML-compatible "
            "Python API. No Extropic TSU hardware execution occurred; this supports "
            "API compatibility only, not a hardware-acceleration claim. Tiny KAN "
            "spline terms still need a THRML hypergraph/spline representation."
        )
    if probe.local_package_path:
        return (
            "A local THRML checkout was found, but it was not importable with the "
            "current Python environment. The audit therefore records Ising/KAN "
            "mapping notes only and disallows hardware claims."
        )
    return (
        "THRML was not importable on the current Python path and no importable local "
        "package was used. The audit records Ising/KAN mapping notes only and "
        "disallows hardware claims."
    )


def build_artifact(
    *,
    probe: ThrmlProbeResult | None = None,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the complete Exp 1347 THRML compatibility audit artifact."""
    probe_result = probe or probe_thrml(project_root=project_root)
    cases_attempted: list[dict[str, Any]]
    energy_parity_max_abs_error: float | None = None
    sample_quality_proxy: dict[str, Any] | None = None
    missing_api_or_dependency = probe_result.missing_api_or_dependency
    hardware_claim_allowed = False

    if not probe_result.import_available or probe_result.module is None:
        cases_attempted = _missing_case_records(missing_api_or_dependency or "THRML not importable")
        honest_verdict = "thrml_unavailable_mapping_notes_only_no_hardware_claim"
    else:
        try:
            ising_record, max_error, quality_proxy = _measure_tiny_ising_parity(probe_result.module)
            cases_attempted = [ising_record, _kan_case_record("mapping_notes_only")]
            energy_parity_max_abs_error = round(float(max_error), 15)
            sample_quality_proxy = quality_proxy
            missing_api_or_dependency = None
            hardware_claim_allowed = True
            honest_verdict = "thrml_energy_parity_measured_no_tsu_hardware_execution"
        except Exception as exc:
            missing_api_or_dependency = _describe_import_exception(exc)
            cases_attempted = [
                {
                    "case_type": "ising",
                    "name": tiny_ising_case().name,
                    "status": "not_attempted_missing_thrml_api",
                    "reason": missing_api_or_dependency,
                    "local_reference": "python/carnot/analysis/pbit_sampler_portability.py::tiny_ising_case",
                },
                _kan_case_record("mapping_notes_only"),
            ]
            honest_verdict = "thrml_import_available_but_api_incompatible_no_hardware_claim"

    artifact = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": run_date,
            "project_root": str(project_root),
            "thrml_version": probe_result.version,
            "thrml_import_source": probe_result.import_source,
            "local_thrml_package_path": probe_result.local_package_path,
            "tsu_hardware_execution": False,
        },
        "status": "complete",
        "thrml_import_available": bool(probe_result.import_available),
        "cases_attempted": cases_attempted,
        "energy_parity_max_abs_error": energy_parity_max_abs_error,
        "sample_quality_proxy": sample_quality_proxy,
        "missing_api_or_dependency": missing_api_or_dependency,
        "tsu_mapping_notes": _mapping_notes(probe_result, energy_parity_max_abs_error is not None),
        "hardware_claim_allowed": hardware_claim_allowed,
        "honest_verdict": honest_verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required schema fields and the conservative hardware-claim gate."""
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["status"] not in {"in_progress", "complete"}:
        raise ValueError(f"invalid status: {artifact['status']!r}")
    if artifact["status"] == "in_progress":
        return
    if artifact["honest_verdict"] not in HONEST_VERDICTS:
        raise ValueError(f"invalid honest_verdict: {artifact['honest_verdict']!r}")
    cases = artifact["cases_attempted"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases_attempted must be a non-empty list")
    parity_measured = any(case.get("status") == "parity_measured" for case in cases)
    if artifact["hardware_claim_allowed"] and (
        artifact["energy_parity_max_abs_error"] is None or not parity_measured
    ):
        raise ValueError("hardware_claim_allowed requires measured parity")
    if not artifact["thrml_import_available"]:
        if artifact["energy_parity_max_abs_error"] is not None:
            raise ValueError("energy parity cannot be present when THRML import is unavailable")
        if artifact["sample_quality_proxy"] is not None:
            raise ValueError("sample_quality_proxy cannot be present when THRML import is unavailable")
        if not artifact["missing_api_or_dependency"]:
            raise ValueError("missing_api_or_dependency is required when THRML import is unavailable")
    if artifact["energy_parity_max_abs_error"] is not None and float(
        artifact["energy_parity_max_abs_error"]
    ) < 0.0:
        raise ValueError("energy_parity_max_abs_error must be non-negative")


def write_artifact(
    path: str | Path = DELIVERABLE_PATH,
    artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a validated Exp 1347 audit artifact."""
    payload = dict(artifact or build_artifact())
    validate_artifact(payload)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_experiment(
    *,
    deliverable_path: str | Path = DELIVERABLE_PATH,
    project_root: str | Path = PROJECT_ROOT,
    run_date: str = RUN_DATE,
    probe_func: Callable[[], ThrmlProbeResult] | None = None,
) -> dict[str, Any]:
    """Run the local THRML audit and persist the deliverable JSON."""
    probe_result = probe_func() if probe_func is not None else probe_thrml(project_root=project_root)
    artifact = build_artifact(probe=probe_result, project_root=project_root, run_date=run_date)
    return write_artifact(deliverable_path, artifact)
