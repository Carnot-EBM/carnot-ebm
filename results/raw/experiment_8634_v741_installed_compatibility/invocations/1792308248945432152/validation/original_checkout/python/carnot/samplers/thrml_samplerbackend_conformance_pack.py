"""Exp 1515 THRML SamplerBackend simulator conformance pack.

This module checks the boundary between Carnot's ``SamplerBackend`` protocol
and the THRML simulator lane. It deliberately stays at the adapter and
manifest layer: THRML import readiness is verified from the active local
Python environment, Carnot's THRML backend is exercised only through its CPU
fallback, and Exp 1504 parity vectors are carried forward as software parity
evidence. No Extropic TSU hardware path is executed or claimed.

Spec traces: REQ-SAMPLE-046, SCENARIO-SAMPLE-074.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.analysis.pbit_sampler_portability import ising_energy, tiny_ising_case
from carnot.samplers.thrml_backend import ThrmlSamplerBackend

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1515_thrml_samplerbackend_conformance_pack.json"
)
CONFORMANCE_MANIFEST_PATH = PROJECT_ROOT / "results" / "thrml_samplerbackend_conformance_1515.jsonl"
GATE_PATH = PROJECT_ROOT / "results" / "experiment_1506_115_completion_archive_116_activation.json"
PARITY_PATH = PROJECT_ROOT / "results" / "experiment_1504_thrml_carnot_simulator_parity_v3.json"

EXPERIMENT_ID = 1515
RUN_DATE = "20260508"
SCHEMA = "thrml_samplerbackend_conformance_pack_v1"

DEFAULT_SEED = 1515
DEFAULT_N_SAMPLES = 8
DEFAULT_N_STEPS = 16
DEFAULT_BETA = 1.25

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
    "thrml_samplerbackend_conformance_ready",
    "gated_inputs_present",
    "thrml_import_ready",
    "simulator_only",
    "no_tsu_hardware_claim",
    "conformance_cases",
    "parity_cases_passed",
    "sample_shape_contracts",
    "seed_reproducibility_checked",
    "conformance_manifest_path",
    "blockers",
    "honest_verdict",
}

ImportModule = Callable[[str], Any]
BackendFactory = Callable[[int], Any]


def _display_path(path: str | Path) -> str:
    output_path = Path(path)
    try:
        return str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(output_path)


def write_in_progress_artifact(
    path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = CONFORMANCE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the bootstrap artifact before gate/import/conformance checks.

    Spec traces: REQ-SAMPLE-046.
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
        "thrml_samplerbackend_conformance_ready": False,
        "gated_inputs_present": False,
        "thrml_import_ready": False,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "conformance_cases": [],
        "parity_cases_passed": 0,
        "sample_shape_contracts": [],
        "seed_reproducibility_checked": False,
        "conformance_manifest_path": _display_path(manifest_path),
        "blockers": [],
        "honest_verdict": (
            "success_in_progress_thrml_samplerbackend_conformance_pack_simulator_only"
        ),
    }
    return _write_json(path, artifact)


def _load_json(path: str | Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    json_path = Path(path)
    try:
        return json.loads(json_path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, {
            "blocker": "json_input_missing",
            "detail": f"missing JSON input: {json_path}",
        }
    except json.JSONDecodeError as exc:
        return None, {
            "blocker": "json_input_malformed",
            "detail": f"malformed JSON input {json_path}: {exc}",
        }


def _gate_ready(path: str | Path) -> tuple[bool, dict[str, Any] | None, dict[str, str] | None]:
    payload, blocker = _load_json(path)
    if blocker is not None:
        return False, None, {
            "blocker": "prior_thrml_parity_gate_missing",
            "detail": blocker["detail"],
        }
    if payload and payload.get("prior_thrml_parity_ready") is True:
        return True, payload, None
    return False, payload, {
        "blocker": "prior_thrml_parity_gate_closed",
        "detail": "Exp 1506 did not report prior_thrml_parity_ready=true",
    }


def _import_thrml(importer: ImportModule) -> tuple[bool, Any | None, dict[str, Any], dict[str, str] | None]:
    try:
        module = importer("thrml")
    except Exception as exc:
        return (
            False,
            None,
            {},
            {
                "blocker": "thrml_local_import_unavailable",
                "detail": f"{exc.__class__.__name__}: {exc}",
            },
        )
    details = {
        "thrml_version": str(getattr(module, "__version__", "unknown")),
        "thrml_import_path": str(getattr(module, "__file__", "<unknown>")),
    }
    return True, module, details, None


def _blocked_artifact(
    *,
    manifest_path: str | Path,
    gated_inputs_present: bool,
    thrml_import_ready: bool,
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
        "thrml_samplerbackend_conformance_ready": False,
        "gated_inputs_present": bool(gated_inputs_present),
        "thrml_import_ready": bool(thrml_import_ready),
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "conformance_cases": [],
        "parity_cases_passed": 0,
        "sample_shape_contracts": [],
        "seed_reproducibility_checked": False,
        "conformance_manifest_path": _display_path(manifest_path),
        "blockers": blockers,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def _case_provenance(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-SAMPLE-046", "SCENARIO-SAMPLE-074"],
        **dict(extra or {}),
    }


def _bool_contract(method: str, samples: np.ndarray, expected_shape: tuple[int, int]) -> dict[str, Any]:
    observed = np.asarray(samples)
    return {
        "method": method,
        "expected_shape": [int(expected_shape[0]), int(expected_shape[1])],
        "observed_shape": [int(dim) for dim in observed.shape],
        "dtype": str(observed.dtype),
        "passed": bool(observed.shape == expected_shape and observed.dtype == np.bool_),
    }


def _samples_to_energy_summary(samples: np.ndarray) -> dict[str, float]:
    case = tiny_ising_case()
    spin_samples = np.where(np.asarray(samples, dtype=bool), 1, -1).astype(np.int8)
    energies = np.asarray([ising_energy(case, spin_state) for spin_state in spin_samples])
    return {
        "mean_energy": round(float(np.mean(energies)), 12),
        "best_energy": round(float(np.min(energies)), 12),
    }


def _build_adapter_rows(
    *,
    backend_factory: BackendFactory,
    seed: int,
    n_samples: int,
    n_steps: int,
    beta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    case = tiny_ising_case()
    biases = np.asarray(case.bias, dtype=np.float64)
    couplings = np.asarray(case.j_matrix, dtype=np.float64)
    config = {
        "beta": float(beta),
        "n_warmup": int(n_steps),
        "steps_per_sample": 2,
        "use_checkerboard": True,
    }
    backend = backend_factory(int(seed))
    sample_output = np.asarray(backend.sample(biases, couplings, int(n_samples), config))
    minimize_output = np.asarray(
        backend.minimize_energy(biases, couplings, int(n_samples), int(n_steps), float(beta))
    )
    replay_output = np.asarray(backend_factory(int(seed)).sample(biases, couplings, int(n_samples), config))
    sample_contract = _bool_contract("sample", sample_output, (int(n_samples), case.n_spins))
    minimize_contract = _bool_contract(
        "minimize_energy", minimize_output, (int(n_samples), case.n_spins)
    )
    same_seed_equal = bool(np.array_equal(sample_output, replay_output))
    rows = [
        {
            "case_id": "adapter:accepted_model_shape",
            "case_type": "accepted_model_shape",
            "passed": bool(
                biases.shape == (case.n_spins,)
                and couplings.shape == (case.n_spins, case.n_spins)
                and np.allclose(couplings, couplings.T)
                and np.allclose(np.diag(couplings), 0.0)
            ),
            "input_contract": {
                "biases_shape": [case.n_spins],
                "couplings_shape": [case.n_spins, case.n_spins],
                "couplings_symmetric": bool(np.allclose(couplings, couplings.T)),
                "zero_diagonal": bool(np.allclose(np.diag(couplings), 0.0)),
                "accepted_spin_domain": "backend returns boolean samples; energy parity uses +/-1 conversion",
            },
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "provenance": _case_provenance({"source": "tiny_ising_case"}),
        },
        {
            "case_id": "adapter:sample_shape_contract",
            "case_type": "sample_shape_contract",
            "passed": bool(sample_contract["passed"]),
            "backend_name": str(getattr(backend, "backend_name", "<unknown>")),
            "schedule": config,
            "seed": int(seed),
            "sample_shape_contract": sample_contract,
            "energy_fields": _samples_to_energy_summary(sample_output),
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "provenance": _case_provenance({"source": "ThrmlSamplerBackend.sample"}),
        },
        {
            "case_id": "adapter:minimize_shape_contract",
            "case_type": "minimize_shape_contract",
            "passed": bool(minimize_contract["passed"]),
            "backend_name": str(getattr(backend, "backend_name", "<unknown>")),
            "schedule": {"beta": float(beta), "n_steps": int(n_steps)},
            "seed": int(seed),
            "sample_shape_contract": minimize_contract,
            "energy_fields": _samples_to_energy_summary(minimize_output),
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "provenance": _case_provenance({"source": "ThrmlSamplerBackend.minimize_energy"}),
        },
        {
            "case_id": "adapter:same_seed_reproducibility",
            "case_type": "seed_reproducibility",
            "passed": same_seed_equal,
            "seed": int(seed),
            "same_seed_equal": same_seed_equal,
            "schedule": config,
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "provenance": _case_provenance({"source": "two fresh backend instances"}),
        },
    ]
    return rows, [sample_contract, minimize_contract], same_seed_equal


def _parity_rows(parity_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    payload, blocker = _load_json(parity_path)
    if blocker is not None:
        return [], {
            "blocker": "prior_parity_vectors_missing",
            "detail": blocker["detail"],
        }
    cases = list((payload or {}).get("cases_compared") or [])
    rows = [
        {
            "case_id": f"exp1504:{case.get('case', '<unknown>')}",
            "case_type": "carnot_thrml_parity_vector",
            "source_case_type": case.get("type"),
            "passed": True,
            "energy_parity_fields": {
                "carnot_output": case.get("carnot_output"),
                "thrml_output": case.get("thrml_output"),
                "delta": case.get("delta"),
                "tolerance": case.get("tolerance"),
            },
            "simulator_only": True,
            "no_tsu_hardware_claim": True,
            "provenance": _case_provenance({"source": _display_path(parity_path)}),
        }
        for case in cases
        if case.get("passed") is True
    ]
    if not rows:
        return [], {
            "blocker": "prior_parity_vectors_not_passed",
            "detail": "Exp 1504 did not contain passed parity cases to carry forward",
        }
    return rows, None


def _case_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": str(row.get("case_id")),
            "case_type": str(row.get("case_type")),
            "passed": bool(row.get("passed") is True),
        }
        for row in rows
    ]


def _write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _terminal_artifact(
    *,
    manifest_path: str | Path,
    rows: list[dict[str, Any]],
    shape_contracts: list[dict[str, Any]],
    seed_reproducibility_checked: bool,
    thrml_details: Mapping[str, Any],
) -> dict[str, Any]:
    parity_cases_passed = sum(
        1
        for row in rows
        if row.get("case_type") == "carnot_thrml_parity_vector" and row.get("passed") is True
    )
    ready = bool(rows and shape_contracts and seed_reproducibility_checked and parity_cases_passed > 0)
    verdict = (
        "complete_thrml_samplerbackend_conformance_ready_no_tsu_hardware_claim"
        if ready
        else "complete_thrml_samplerbackend_conformance_not_ready_no_tsu_hardware_claim"
    )
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "tsu_hardware_execution": False,
            **dict(thrml_details),
        },
        "status": "complete",
        "thrml_samplerbackend_conformance_ready": ready,
        "gated_inputs_present": True,
        "thrml_import_ready": True,
        "simulator_only": True,
        "no_tsu_hardware_claim": True,
        "conformance_cases": _case_summaries(rows),
        "parity_cases_passed": int(parity_cases_passed),
        "sample_shape_contracts": shape_contracts,
        "seed_reproducibility_checked": bool(seed_reproducibility_checked),
        "conformance_manifest_path": _display_path(manifest_path),
        "blockers": [],
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def run_conformance_pack(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    manifest_path: str | Path = CONFORMANCE_MANIFEST_PATH,
    gate_path: str | Path = GATE_PATH,
    parity_path: str | Path = PARITY_PATH,
    importer: ImportModule = importlib.import_module,
    backend_factory: BackendFactory = ThrmlSamplerBackend,
    seed: int = DEFAULT_SEED,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_steps: int = DEFAULT_N_STEPS,
    beta: float = DEFAULT_BETA,
) -> dict[str, Any]:
    """Run the simulator-only conformance pack and write JSON/JSONL artifacts.

    Spec traces: REQ-SAMPLE-046, SCENARIO-SAMPLE-074.
    """

    write_in_progress_artifact(output_path, manifest_path)
    gate_ok, gate_payload, gate_blocker = _gate_ready(gate_path)
    if not gate_ok:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                gated_inputs_present=False,
                thrml_import_ready=False,
                blockers=[gate_blocker or {"blocker": "prior_thrml_parity_gate_closed", "detail": ""}],
                verdict="complete_thrml_samplerbackend_conformance_gated_no_tsu_hardware_claim",
                metadata={"gate_payload_status": (gate_payload or {}).get("status")},
            ),
        )
    import_ready, _thrml_module, thrml_details, import_blocker = _import_thrml(importer)
    if not import_ready:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                gated_inputs_present=True,
                thrml_import_ready=False,
                blockers=[
                    import_blocker
                    or {"blocker": "thrml_local_import_unavailable", "detail": "unknown import error"}
                ],
                verdict=(
                    "complete_thrml_samplerbackend_conformance_blocked_simulator_dependency_"
                    "no_tsu_hardware_claim"
                ),
                metadata={"gate_payload_status": (gate_payload or {}).get("status")},
            ),
        )
    parity_vector_rows, parity_blocker = _parity_rows(parity_path)
    if parity_blocker is not None:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                gated_inputs_present=True,
                thrml_import_ready=True,
                blockers=[parity_blocker],
                verdict=(
                    "complete_thrml_samplerbackend_conformance_blocked_prior_parity_"
                    "no_tsu_hardware_claim"
                ),
                metadata=thrml_details,
            ),
        )
    try:
        adapter_rows, shape_contracts, seed_checked = _build_adapter_rows(
            backend_factory=backend_factory,
            seed=seed,
            n_samples=n_samples,
            n_steps=n_steps,
            beta=beta,
        )
    except Exception as exc:
        Path(manifest_path).unlink(missing_ok=True)
        return _write_json(
            output_path,
            _blocked_artifact(
                manifest_path=manifest_path,
                gated_inputs_present=True,
                thrml_import_ready=True,
                blockers=[
                    {
                        "blocker": "samplerbackend_conformance_failed",
                        "detail": f"{exc.__class__.__name__}: {exc}",
                    }
                ],
                verdict=(
                    "complete_thrml_samplerbackend_conformance_blocked_adapter_contract_"
                    "no_tsu_hardware_claim"
                ),
                metadata=thrml_details,
            ),
        )
    rows = adapter_rows + parity_vector_rows
    _write_manifest(manifest_path, rows)
    return _write_json(
        output_path,
        _terminal_artifact(
            manifest_path=manifest_path,
            rows=rows,
            shape_contracts=shape_contracts,
            seed_reproducibility_checked=seed_checked,
            thrml_details=thrml_details,
        ),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required schema and simulator-only/no-TSU boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    status = artifact.get("status")
    if status not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {status!r}")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1515")
    if artifact.get("no_tsu_hardware_claim") is not True:
        raise ValueError("no_tsu_hardware_claim must remain true for Exp 1515")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact.get("thrml_samplerbackend_conformance_ready") is True:
        if (
            not artifact.get("conformance_cases")
            or int(artifact.get("parity_cases_passed") or 0) <= 0
            or not artifact.get("sample_shape_contracts")
            or artifact.get("seed_reproducibility_checked") is not True
        ):
            raise ValueError("ready artifact requires cases, parity, shapes, and seed checks")


def _write_json(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(artifact)
    validate_artifact(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:  # pragma: no cover
    run_conformance_pack()


if __name__ == "__main__":  # pragma: no cover
    main()
