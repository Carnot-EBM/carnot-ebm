"""Exp 1504 gated THRML/Carnot simulator parity audit.

This module runs only the simulator lane opened by Exp 1503. It compares a
tiny Ising case through Carnot's local software simulator and the installed
THRML software API, records numeric tolerances, and keeps Extropic TSU hardware
claims disabled in every outcome.

Spec traces: REQ-SAMPLE-045, SCENARIO-SAMPLE-073.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from carnot.samplers.thrml_npim_microprobe import _default_carnot_sample_func

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1504_thrml_carnot_simulator_parity_v3.json"
)
GATE_PATH = PROJECT_ROOT / "results" / "experiment_1503_thrml_import_readiness_repair_gate.json"

EXPERIMENT_ID = 1504
RUN_DATE = "20260507"
SCHEMA = "thrml_carnot_simulator_parity_v3"

DEFAULT_TOLERANCE = {
    "exact_energy_abs": 1.0e-6,
    "stochastic_mean_energy_abs": 0.35,
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
    "parity_experiment_ran",
    "thrml_import_ready",
    "simulator_only",
    "cases_compared",
    "parity_pass_count",
    "parity_fail_count",
    "tolerance",
    "max_observed_delta",
    "hardware_claim_allowed",
    "blockers",
    "honest_verdict",
}

ImportModule = Callable[[str], Any]


def _round_metric(value: float | None, digits: int = 12) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _load_gate(gate_path: str | Path = GATE_PATH) -> tuple[bool, dict[str, Any]]:
    gate_file = Path(gate_path)
    try:
        payload = json.loads(gate_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, {"error": f"missing gate artifact: {gate_file}"}
    except json.JSONDecodeError as exc:
        return False, {"error": f"malformed gate artifact: {exc}"}
    return bool(payload.get("thrml_import_ready") is True), payload


def _blocked_artifact(
    *,
    thrml_import_ready: bool,
    blockers: list[dict[str, str]],
    gate_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "gate_artifact_path": str(GATE_PATH),
            "gate_payload_status": dict(gate_payload or {}).get("status"),
            "tsu_hardware_execution": False,
        },
        "status": "blocked",
        "parity_experiment_ran": False,
        "thrml_import_ready": bool(thrml_import_ready),
        "simulator_only": True,
        "cases_compared": [],
        "parity_pass_count": 0,
        "parity_fail_count": 0,
        "tolerance": DEFAULT_TOLERANCE.copy(),
        "max_observed_delta": None,
        "hardware_claim_allowed": False,
        "blockers": blockers,
        "honest_verdict": "complete_thrml_carnot_simulator_parity_gated_no_hardware_claim",
    }
    validate_artifact(artifact)
    return artifact


def write_in_progress_artifact(path: str | Path = DELIVERABLE_PATH) -> dict[str, Any]:
    """Write the required bootstrap artifact before gate inspection finishes.

    Spec traces: REQ-SAMPLE-045.
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
        "parity_experiment_ran": False,
        "thrml_import_ready": None,
        "simulator_only": True,
        "cases_compared": [],
        "parity_pass_count": 0,
        "parity_fail_count": 0,
        "tolerance": DEFAULT_TOLERANCE.copy(),
        "max_observed_delta": None,
        "hardware_claim_allowed": False,
        "blockers": [],
        "honest_verdict": "success_in_progress_thrml_carnot_simulator_parity_v3",
    }
    validate_artifact(artifact)
    return _write_json(path, artifact)


def _case_edges(case: IsingCase) -> tuple[list[tuple[int, int]], np.ndarray]:
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(case.n_spins):
        for j in range(i + 1, case.n_spins):
            weight = float(case.j_matrix[i, j])
            if weight != 0.0:
                edges.append((i, j))
                weights.append(weight)
    return edges, np.asarray(weights, dtype=np.float32)


def _require_thrml_api(module: Any) -> tuple[Any, Any, Any, Any]:
    spin_node = getattr(module, "SpinNode", None)
    block_cls = getattr(module, "Block", None)
    models = getattr(module, "models", None)
    ising_ebm = getattr(models, "IsingEBM", None) if models is not None else None
    ising_program = getattr(models, "IsingSamplingProgram", None) if models is not None else None
    missing = [
        name
        for name, value in (
            ("SpinNode", spin_node),
            ("Block", block_cls),
            ("models.IsingEBM", ising_ebm),
            ("models.IsingSamplingProgram", ising_program),
        )
        if value is None
    ]
    if missing:
        raise AttributeError(f"THRML import lacks required Ising APIs: {', '.join(missing)}")
    return spin_node, block_cls, ising_ebm, ising_program


def _build_thrml_model(module: Any, case: IsingCase, *, beta: float) -> tuple[Any, list[Any]]:
    import jax.numpy as jnp

    spin_node, _block_cls, ising_ebm, _ising_program = _require_thrml_api(module)
    nodes = [spin_node() for _ in range(case.n_spins)]
    edge_indices, weights = _case_edges(case)
    edges = [(nodes[i], nodes[j]) for i, j in edge_indices]
    model = ising_ebm(
        nodes,
        edges,
        jnp.asarray(case.bias, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(float(beta), dtype=jnp.float32),
    )
    return model, nodes


def _sample_energies(case: IsingCase, samples: np.ndarray) -> np.ndarray:
    return np.asarray([ising_energy(case, state) for state in samples], dtype=np.float64)


def _measure_exact_energy_case(
    module: Any,
    case: IsingCase,
    tolerance: Mapping[str, float],
) -> dict[str, Any]:
    model, nodes = _build_thrml_model(module, case, beta=1.0)
    _spin_node, block_cls, _ising_ebm, _ising_program = _require_thrml_api(module)
    block = block_cls(nodes)
    states = enumerate_spin_states(case.n_spins)
    deltas: list[float] = []
    local_energies: list[float] = []
    thrml_energies: list[float] = []
    for state in states:
        local_energy = float(ising_energy(case, state))
        bool_state = np.asarray(state == 1, dtype=bool)
        thrml_energy = float(model.energy([bool_state], [block]))
        local_energies.append(local_energy)
        thrml_energies.append(thrml_energy)
        deltas.append(abs(local_energy - thrml_energy))
    max_delta = max(deltas) if deltas else 0.0
    limit = float(tolerance["exact_energy_abs"])
    return {
        "case": f"tiny_ising:{case.name}:exact_energy",
        "type": "exact_enumerated_energy",
        "seed": None,
        "state_count": int(states.shape[0]),
        "carnot_output": {
            "min_energy": _round_metric(min(local_energies)),
            "max_energy": _round_metric(max(local_energies)),
        },
        "thrml_output": {
            "min_energy": _round_metric(min(thrml_energies)),
            "max_energy": _round_metric(max(thrml_energies)),
        },
        "delta": _round_metric(max_delta),
        "tolerance": limit,
        "passed": bool(max_delta <= limit),
        "api_limitation": None,
    }


def _measure_stochastic_case(
    module: Any,
    case: IsingCase,
    tolerance: Mapping[str, float],
    *,
    seed: int,
    n_samples: int,
    n_warmup: int,
    steps_per_sample: int,
) -> dict[str, Any]:
    import jax.numpy as jnp
    import jax.random as jrandom

    _spin_node, block_cls, _ising_ebm, ising_program = _require_thrml_api(module)
    model, nodes = _build_thrml_model(module, case, beta=case.beta)
    blocks = [block_cls([node]) for node in nodes]
    program = ising_program(model, blocks, [])
    schedule = module.SamplingSchedule(
        n_warmup=int(n_warmup),
        n_samples=int(n_samples),
        steps_per_sample=int(steps_per_sample),
    )
    init_bool = [jnp.asarray([idx % 2 == 0], dtype=bool) for idx in range(case.n_spins)]
    thrml_blocks = module.sample_states(
        jrandom.PRNGKey(seed),
        program,
        schedule,
        init_bool,
        [],
        blocks,
    )
    thrml_bool = np.concatenate([np.asarray(item, dtype=bool) for item in thrml_blocks], axis=1)
    thrml_samples = np.where(thrml_bool, 1, -1).astype(np.int8)
    carnot_samples = _default_carnot_sample_func(
        case,
        seed=seed,
        n_samples=n_samples,
        n_warmup=n_warmup,
        steps_per_sample=steps_per_sample,
    )
    thrml_energies = _sample_energies(case, thrml_samples)
    carnot_energies = _sample_energies(case, carnot_samples)
    carnot_mean = float(np.mean(carnot_energies))
    thrml_mean = float(np.mean(thrml_energies))
    delta = abs(carnot_mean - thrml_mean)
    limit = float(tolerance["stochastic_mean_energy_abs"])
    return {
        "case": f"tiny_ising:{case.name}:fixed_seed_sample_mean_energy",
        "type": "fixed_seed_sample_mean_energy",
        "seed": int(seed),
        "sample_count": int(n_samples),
        "n_warmup": int(n_warmup),
        "steps_per_sample": int(steps_per_sample),
        "carnot_output": {
            "mean_energy": _round_metric(carnot_mean),
            "best_energy": _round_metric(float(np.min(carnot_energies))),
        },
        "thrml_output": {
            "mean_energy": _round_metric(thrml_mean),
            "best_energy": _round_metric(float(np.min(thrml_energies))),
        },
        "delta": _round_metric(delta),
        "tolerance": limit,
        "passed": bool(delta <= limit),
        "api_limitation": None,
    }


def _measure_cases(
    module: Any,
    *,
    seed: int,
    n_samples: int,
    n_warmup: int,
    steps_per_sample: int,
    tolerance: Mapping[str, float],
) -> list[dict[str, Any]]:
    case = tiny_ising_case()
    return [
        _measure_exact_energy_case(module, case, tolerance),
        _measure_stochastic_case(
            module,
            case,
            tolerance,
            seed=seed,
            n_samples=n_samples,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
        ),
    ]


def _terminal_artifact(
    *,
    cases: list[dict[str, Any]],
    tolerance: Mapping[str, float],
    thrml_module: Any,
    seed: int,
) -> dict[str, Any]:
    pass_count = sum(1 for case in cases if case.get("passed") is True)
    fail_count = len(cases) - pass_count
    deltas = [float(case["delta"]) for case in cases if case.get("delta") is not None]
    verdict = (
        "complete_thrml_carnot_simulator_parity_passed_no_hardware_claim"
        if fail_count == 0
        else "complete_thrml_carnot_simulator_parity_failed_no_hardware_claim"
    )
    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": str(PROJECT_ROOT),
            "thrml_version": str(getattr(thrml_module, "__version__", "unknown")),
            "fixed_seed": int(seed),
            "tsu_hardware_execution": False,
        },
        "status": "complete",
        "parity_experiment_ran": True,
        "thrml_import_ready": True,
        "simulator_only": True,
        "cases_compared": cases,
        "parity_pass_count": int(pass_count),
        "parity_fail_count": int(fail_count),
        "tolerance": dict(tolerance),
        "max_observed_delta": _round_metric(max(deltas) if deltas else None),
        "hardware_claim_allowed": False,
        "blockers": [],
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def run_parity_audit(
    *,
    output_path: str | Path = DELIVERABLE_PATH,
    gate_path: str | Path = GATE_PATH,
    importer: ImportModule = importlib.import_module,
    seed: int = 1504,
    n_samples: int = 64,
    n_warmup: int = 64,
    steps_per_sample: int = 4,
    tolerance: Mapping[str, float] = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Run the complete gated simulator-only parity audit and write JSON.

    Spec traces: REQ-SAMPLE-045, SCENARIO-SAMPLE-073.
    """

    write_in_progress_artifact(output_path)
    gate_ready, gate_payload = _load_gate(gate_path)
    if not gate_ready:
        detail = str(gate_payload.get("error") or "Exp 1503 did not report thrml_import_ready=true")
        return _write_json(
            output_path,
            _blocked_artifact(
                thrml_import_ready=False,
                blockers=[{"blocker": "thrml_import_gate_closed", "detail": detail}],
                gate_payload=gate_payload,
            ),
        )
    try:
        thrml_module = importer("thrml")
        cases = _measure_cases(
            thrml_module,
            seed=seed,
            n_samples=n_samples,
            n_warmup=n_warmup,
            steps_per_sample=steps_per_sample,
            tolerance=tolerance,
        )
    except Exception as exc:
        return _write_json(
            output_path,
            _blocked_artifact(
                thrml_import_ready=True,
                blockers=[
                    {
                        "blocker": "thrml_api_incompatible",
                        "detail": f"{exc.__class__.__name__}: {exc}",
                    }
                ],
                gate_payload=gate_payload,
            ),
        )
    return _write_json(
        output_path,
        _terminal_artifact(cases=cases, tolerance=tolerance, thrml_module=thrml_module, seed=seed),
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the public artifact schema and no-hardware-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact.get("hardware_claim_allowed") is not False:
        raise ValueError("hardware_claim_allowed must remain false for Exp 1504")
    if artifact.get("simulator_only") is not True:
        raise ValueError("simulator_only must remain true for Exp 1504")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_VERDICT_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal prefix")
    status = artifact.get("status")
    if status not in {"in_progress", "blocked", "complete"}:
        raise ValueError(f"invalid status: {status!r}")
    if status != "complete":
        return
    cases = list(artifact.get("cases_compared") or [])
    if not cases or artifact.get("parity_experiment_ran") is not True:
        raise ValueError("complete parity artifact must contain executed cases")
    pass_count = int(artifact.get("parity_pass_count", -1))
    fail_count = int(artifact.get("parity_fail_count", -1))
    if pass_count + fail_count != len(cases):
        raise ValueError("parity pass/fail counts must match cases_compared")


def _write_json(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(artifact)
    validate_artifact(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:  # pragma: no cover
    run_parity_audit()


if __name__ == "__main__":  # pragma: no cover
    main()
