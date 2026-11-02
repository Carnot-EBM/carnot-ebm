"""Exp 3006 fixed-point energy diagnostic over cached validator trajectories.

Spec refs: REQ-VERIFY-3006, SCENARIO-VERIFY-3006.

This module is intentionally diagnostic-only.  Equilibrium Reasoners and
attractor models motivate the questions we ask, but the implementation below
does not add a learned recurrence module, a native EqR architecture, or a local
attractor-model capability.  It replays the cached Exp 3005 exact-validator
evidence and asks whether the existing feedback states look like descent toward
a zero-violation fixed point.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import solver_to_validator_tree_expansion_v1 as exp3005


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_NAME = "experiment_3006_eqr_fixed_point_energy_diagnostic_v1"
OUTPUT_FILENAME = f"{ARTIFACT_NAME}.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILENAME
DIAGNOSTIC_TABLE_REL_PATH = Path("results/eqr_fixed_point_energy_diagnostic_3006/diagnostic_table.jsonl")
DEFAULT_TABLE_PATH = REPO_ROOT / DIAGNOSTIC_TABLE_REL_PATH
MIN_TRAJECTORIES = 1
TERMINAL_PREFIXES = ("ready:", "flagged:", "blocked:")
CONTROL_REJECTION_THRESHOLD = 1.0
REQUIRED_ARTIFACT_FIELDS = (
    "fixed_point_diagnostic_ready",
    "n_trajectories",
    "energy_definition",
    "convergence_rate",
    "energy_monotonicity_rate",
    "basin_sensitivity_summary",
    "negative_control_rejection_rate",
    "diagnostic_table_path",
    "native_eqr_claim_made",
    "honest_verdict",
)
ENERGY_DEFINITION = (
    "Scalar diagnostic energy = exact-validator violation penalties: "
    "runtime node failure=1, generic rejected state=1 per reason, "
    "partial prefix violation=2, candidate expected-status mismatch=2, "
    "Z3 execution failure=2, Z3/reference status mismatch=3, "
    "swapped incompatible validator=3, injected contradiction node=4, plus "
    "0.5 * remaining-reference-assertion fraction for accepted partial states. "
    "Zero energy means the cached full candidate cleared runtime and Z3 exact checks."
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Paths and clock hooks for deterministic Exp 3006 runs.

    Tests pass temporary paths so the diagnostic can be exercised without
    rewriting the repository artifact.  The default paths are the conductor
    contract: consume the cached Exp 3005 manifest and write the Exp 3006
    artifact plus a reproducible JSONL table under ``results/``.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    table_path: Path | None = None
    manifest_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_table_path(self) -> Path:
        return self.table_path or self.repo_root / DIAGNOSTIC_TABLE_REL_PATH

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / exp3005.VALIDATOR_MANIFEST_REL_PATH


def load_manifest(path: Path) -> list[JsonDict]:
    """Load the Exp 3005 JSONL validator manifest."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_diagnostic_table(path: Path) -> list[JsonDict]:
    """Load the reproducible Exp 3006 diagnostic table."""

    return load_manifest(path)


def load_cached_trajectories(manifest_path: Path) -> list[JsonDict]:
    """Reconstruct fixed-point diagnostic trajectories from cached validator rows."""

    return [build_trajectory(row) for row in load_manifest(manifest_path)]


def build_trajectory(row: Mapping[str, Any]) -> JsonDict:
    """Build the invalid-partial -> valid-partial -> exact-full energy path."""

    reference_count = _reference_assertion_count(row)
    prefix_count = _partial_prefix_count(reference_count)
    states = [
        score_state(
            "invalid_partial_feedback",
            row["partial_viability"]["invalid_partial"],
            reference_assertion_count=reference_count,
            partial_assertion_count=prefix_count + 1,
        ),
        score_state(
            "valid_extendable_partial",
            row["partial_viability"]["valid_partial"],
            reference_assertion_count=reference_count,
            partial_assertion_count=prefix_count,
        ),
        score_state(
            "full_exact_candidate",
            row["full_validation"],
            reference_assertion_count=reference_count,
        ),
    ]
    energies = [state["energy"] for state in states]
    return {
        "item_id": row["item_id"],
        "source_family": row.get("source_family"),
        "states": states,
        "energy_sequence": energies,
        "converged_to_fixed_point": energies[-1] == 0.0 and energies[-1] < energies[0],
        "energy_monotonic": _is_nonincreasing(energies),
        "native_eqr_claim_made": False,
    }


def score_state(
    state_name: str,
    feedback: Mapping[str, Any],
    *,
    reference_assertion_count: int,
    partial_assertion_count: int | None = None,
    contradiction_injected: bool = False,
    incompatible_validator: bool = False,
) -> JsonDict:
    """Assign auditable scalar energy to one cached or synthetic validator state."""

    node_results = list(feedback.get("node_results", []))
    reasons = _rejection_reasons(feedback, node_results)
    accepted = bool(feedback.get("accepted"))
    runtime_failures = _count_node_failures(node_results, "runtime_json_parser")
    z3_failures = _count_z3_execution_failures(node_results)
    z3_mismatches = _count_node_failures(node_results, "z3_solver")
    generic_rejection_count = len(reasons) if not accepted else 0
    partial_prefix_violations = int("partial_assertions_not_reference_prefix" in reasons)
    expected_status_mismatches = int("candidate_expected_status_mismatch" in reasons)

    energy = 0.0
    energy += float(runtime_failures)
    energy += float(generic_rejection_count)
    energy += 2.0 * partial_prefix_violations
    energy += 2.0 * expected_status_mismatches
    energy += 2.0 * z3_failures
    energy += 3.0 * z3_mismatches
    energy += 4.0 if contradiction_injected else 0.0
    energy += 3.0 if incompatible_validator else 0.0
    if accepted and partial_assertion_count is not None:
        remaining = max(reference_assertion_count - partial_assertion_count, 0)
        energy += 0.5 * (remaining / max(reference_assertion_count, 1))

    return {
        "state": state_name,
        "accepted": accepted,
        "energy": round(energy, 6),
        "rejection_reasons": reasons,
        "components": {
            "runtime_failures": runtime_failures,
            "generic_rejection_count": generic_rejection_count,
            "partial_prefix_violations": partial_prefix_violations,
            "expected_status_mismatches": expected_status_mismatches,
            "z3_execution_failures": z3_failures,
            "z3_status_mismatches": z3_mismatches,
            "contradiction_injected": contradiction_injected,
            "incompatible_validator": incompatible_validator,
        },
    }


def measure_basin_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    z3_module: Any = exp3005._z3,
) -> JsonDict:
    """Measure deterministic small-perturbation sensitivity around full candidates."""

    perturbations: list[JsonDict] = []
    sensitive_item_ids: set[str] = set()
    accepted_count = 0
    for row in rows:
        item_perturbations = _basin_perturbations(row, z3_module=z3_module)
        perturbations.extend(item_perturbations)
        accepted_count += sum(1 for item in item_perturbations if item["accepted"])
        if any(item["energy_delta"] > 0.0 for item in item_perturbations):
            sensitive_item_ids.add(str(row["item_id"]))

    deltas = [item["energy_delta"] for item in perturbations]
    return {
        "trajectory_count": len(rows),
        "perturbation_count": len(perturbations),
        "mean_energy_delta": _mean(deltas),
        "max_energy_delta": round(max(deltas), 6) if deltas else 0.0,
        "min_energy_delta": round(min(deltas), 6) if deltas else 0.0,
        "sensitive_trajectory_rate": _rate(len(sensitive_item_ids), len(rows)),
        "accepted_perturbation_rate": _rate(accepted_count, len(perturbations)),
    }


def build_negative_controls(
    row: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
    *,
    z3_module: Any = exp3005._z3,
) -> list[JsonDict]:
    """Create diagnostic negative controls for one trajectory."""

    reference_count = _reference_assertion_count(row)
    controls = [
        _permuted_partial_control(row, reference_count, z3_module=z3_module),
        _swapped_validator_control(row, all_rows, reference_count, z3_module=z3_module),
        _contradiction_control(row, reference_count, z3_module=z3_module),
    ]
    return [_mark_control_rejection(control) for control in controls]


def run_diagnostic(
    config: ExperimentConfig | None = None,
    *,
    z3_module: Any = exp3005._z3,
) -> JsonDict:
    """Run the cached fixed-point diagnostic and persist the artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    rows = load_manifest(active.resolved_manifest_path())
    trajectories = [build_trajectory(row) for row in rows]
    controls_by_item = [build_negative_controls(row, rows, z3_module=z3_module) for row in rows]
    basin_summary = measure_basin_sensitivity(rows, z3_module=z3_module)
    table_rows = _diagnostic_table_rows(rows, trajectories, controls_by_item)
    _write_jsonl(active.resolved_table_path(), table_rows)
    artifact = build_artifact(
        active,
        trajectories,
        controls_by_item,
        basin_summary,
        duration_s=round(active.clock() - started, 6),
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def build_artifact(
    config: ExperimentConfig,
    trajectories: Sequence[Mapping[str, Any]],
    controls_by_item: Sequence[Sequence[Mapping[str, Any]]],
    basin_summary: Mapping[str, Any],
    *,
    duration_s: float,
) -> JsonDict:
    """Build the terminal Exp 3006 artifact from measured diagnostic rows."""

    convergence_rate = _rate(
        sum(1 for trajectory in trajectories if trajectory["converged_to_fixed_point"]),
        len(trajectories),
    )
    monotonicity_rate = _energy_monotonicity_rate(trajectories)
    control_rows = [control for controls in controls_by_item for control in controls]
    negative_rate = _rate(
        sum(1 for control in control_rows if control["diagnostic_rejected"]),
        len(control_rows),
    )
    ready = bool(
        trajectories
        and str(ENERGY_DEFINITION)
        and config.resolved_table_path().exists()
        and 0.0 <= convergence_rate <= 1.0
        and 0.0 <= monotonicity_rate <= 1.0
        and int(basin_summary.get("perturbation_count", 0)) > 0
        and negative_rate > 0.0
    )
    return {
        "schema": "carnot.eqr_fixed_point_energy_diagnostic.v1",
        "artifact": ARTIFACT_NAME,
        "run_date": RUN_DATE,
        "fixed_point_diagnostic_ready": ready,
        "n_trajectories": len(trajectories),
        "energy_definition": ENERGY_DEFINITION,
        "convergence_rate": convergence_rate,
        "energy_monotonicity_rate": monotonicity_rate,
        "basin_sensitivity_summary": dict(basin_summary),
        "negative_control_rejection_rate": negative_rate,
        "diagnostic_table_path": str(_relative_to(config.repo_root, config.resolved_table_path())),
        "native_eqr_claim_made": False,
        "honest_verdict": (
            "ready: fixed-point diagnostic over cached validator trajectories complete"
            if ready
            else "flagged: fixed-point diagnostic evidence incomplete"
        ),
        "duration_s": duration_s,
        "source_manifest_path": str(_relative_to(config.repo_root, config.resolved_manifest_path())),
        "n_negative_controls": len(control_rows),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3006 artifact violates its terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("native_eqr_claim_made") is not False:
        raise ValueError("native_eqr_claim_made must remain false")
    if int(artifact.get("n_trajectories") or 0) < MIN_TRAJECTORIES:
        raise ValueError("n_trajectories must be positive")
    if not str(artifact.get("energy_definition") or "").strip():
        raise ValueError("energy_definition must be non-empty")
    _validate_rate("convergence_rate", artifact.get("convergence_rate"))
    _validate_rate("energy_monotonicity_rate", artifact.get("energy_monotonicity_rate"))
    _validate_rate(
        "negative_control_rejection_rate",
        artifact.get("negative_control_rejection_rate"),
    )
    if not isinstance(artifact.get("basin_sensitivity_summary"), dict):
        raise ValueError("basin_sensitivity_summary must be a dict")
    if not str(artifact.get("diagnostic_table_path") or "").strip():
        raise ValueError("diagnostic_table_path must be non-empty")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must state ready, flagged, or blocked")


def _basin_perturbations(
    row: Mapping[str, Any],
    *,
    z3_module: Any,
) -> list[JsonDict]:
    base_energy = build_trajectory(row)["states"][-1]["energy"]
    reference_count = _reference_assertion_count(row)
    perturbations = [
        ("drop_last_assertion", _candidate_from_row(row, drop_last=True), False, False),
        ("flip_expected_status", _candidate_from_row(row, flip_status=True), False, False),
        ("duplicate_first_assertion", _candidate_from_row(row, duplicate_first=True), False, False),
    ]
    results = []
    for name, candidate, contradiction, incompatible in perturbations:
        feedback = exp3005.evaluate_validator_tree(
            row["validator_tree"],
            json.dumps(candidate, sort_keys=True),
            z3_module=z3_module,
        )
        state = score_state(
            name,
            feedback,
            reference_assertion_count=reference_count,
            contradiction_injected=contradiction,
            incompatible_validator=incompatible,
        )
        results.append(
            {
                "perturbation": name,
                "accepted": state["accepted"],
                "energy": state["energy"],
                "energy_delta": round(state["energy"] - base_energy, 6),
            }
        )
    return results


def _permuted_partial_control(
    row: Mapping[str, Any],
    reference_count: int,
    *,
    z3_module: Any,
) -> JsonDict:
    assertions = list(row["validator_tree"]["reference"]["assertions"])
    prefix_count = min(max(_partial_prefix_count(reference_count), 2), len(assertions))
    candidate = {
        "assertions": list(reversed(assertions[:prefix_count])),
        "query": "(check-sat)",
        "expected_status": row["validator_tree"]["reference"]["expected_solver_status"],
    }
    feedback = exp3005.evaluate_partial_candidate(
        row["validator_tree"],
        json.dumps(candidate, sort_keys=True),
        z3_module=z3_module,
    )
    state = score_state(
        "permuted_partial_constraints",
        feedback,
        reference_assertion_count=reference_count,
        partial_assertion_count=prefix_count,
    )
    return _control("permuted_partial_constraints", state)


def _swapped_validator_control(
    row: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
    reference_count: int,
    *,
    z3_module: Any,
) -> JsonDict:
    other = _find_incompatible_row(row, all_rows)
    candidate = _candidate_from_row(row)
    feedback = exp3005.evaluate_validator_tree(
        other["validator_tree"],
        json.dumps(candidate, sort_keys=True),
        z3_module=z3_module,
    )
    state = score_state(
        "swapped_incompatible_validator",
        feedback,
        reference_assertion_count=reference_count,
        incompatible_validator=True,
    )
    return _control("swapped_incompatible_validator", state, swapped_with=other["item_id"])


def _contradiction_control(
    row: Mapping[str, Any],
    reference_count: int,
    *,
    z3_module: Any,
) -> JsonDict:
    candidate = _candidate_from_row(row)
    candidate["assertions"] = [*candidate["assertions"], "(assert false)"]
    feedback = exp3005.evaluate_validator_tree(
        row["validator_tree"],
        json.dumps(candidate, sort_keys=True),
        z3_module=z3_module,
    )
    state = score_state(
        "contradiction_node_injection",
        feedback,
        reference_assertion_count=reference_count,
        contradiction_injected=True,
    )
    return _control("contradiction_node_injection", state)


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    drop_last: bool = False,
    flip_status: bool = False,
    duplicate_first: bool = False,
) -> JsonDict:
    reference = row["validator_tree"]["reference"]
    assertions = list(reference["assertions"])
    if drop_last and assertions:
        assertions = assertions[:-1]
    if duplicate_first and assertions:
        assertions = [*assertions, assertions[0]]
    status = str(reference["expected_solver_status"])
    if flip_status:
        status = "sat" if status == "unsat" else "unsat"
    return {
        "assertions": assertions,
        "query": "(check-sat)",
        "expected_status": status,
        "answer_extraction": {"expected_answer_values": {}},
    }


def _diagnostic_table_rows(
    rows: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    controls_by_item: Sequence[Sequence[Mapping[str, Any]]],
) -> list[JsonDict]:
    table_rows = []
    for row, trajectory, controls in zip(rows, trajectories, controls_by_item, strict=True):
        rejected = sum(1 for control in controls if control["diagnostic_rejected"])
        table_rows.append(
            {
                "item_id": row["item_id"],
                "source_family": row.get("source_family"),
                "energy_sequence": list(trajectory["energy_sequence"]),
                "converged_to_fixed_point": trajectory["converged_to_fixed_point"],
                "energy_monotonic": trajectory["energy_monotonic"],
                "negative_controls": [dict(control) for control in controls],
                "negative_controls_rejected": rejected,
                "native_eqr_claim_made": False,
            }
        )
    return table_rows


def _control(name: str, state: Mapping[str, Any], **extra: Any) -> JsonDict:
    payload = {
        "control": name,
        "accepted": state["accepted"],
        "energy": state["energy"],
        "rejection_reasons": list(state["rejection_reasons"]),
    }
    payload.update(extra)
    return payload


def _mark_control_rejection(control: Mapping[str, Any]) -> JsonDict:
    return dict(control) | {
        "diagnostic_rejected": float(control["energy"]) >= CONTROL_REJECTION_THRESHOLD
    }


def _find_incompatible_row(
    row: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    status = row["validator_tree"]["reference"]["expected_solver_status"]
    for other in all_rows:
        other_status = other["validator_tree"]["reference"]["expected_solver_status"]
        if other["item_id"] != row["item_id"] and other_status != status:
            return other
    for other in all_rows:  # pragma: no cover - Exp 3005 has both sat and unsat rows.
        if other["item_id"] != row["item_id"]:
            return other
    return row  # pragma: no cover - a one-row manifest is rejected by downstream gates.


def _reference_assertion_count(row: Mapping[str, Any]) -> int:
    return len(row["validator_tree"]["reference"]["assertions"])


def _partial_prefix_count(reference_count: int) -> int:
    return max(1, min(2, reference_count - 1))


def _is_nonincreasing(values: Sequence[float]) -> bool:
    return all(next_value <= value for value, next_value in zip(values[:-1], values[1:], strict=True))


def _energy_monotonicity_rate(trajectories: Sequence[Mapping[str, Any]]) -> float:
    total = 0
    nonincreasing = 0
    for trajectory in trajectories:
        sequence = list(trajectory["energy_sequence"])
        pairs = list(zip(sequence[:-1], sequence[1:], strict=True))
        total += len(pairs)
        nonincreasing += sum(1 for left, right in pairs if right <= left)
    return _rate(nonincreasing, total)


def _rejection_reasons(
    feedback: Mapping[str, Any],
    node_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons = list(feedback.get("rejection_reasons", []))
    reasons.extend(
        str(row["rejection_reason"]) for row in node_results if row.get("rejection_reason")
    )
    return list(dict.fromkeys(str(reason) for reason in reasons if reason))


def _count_node_failures(node_results: Sequence[Mapping[str, Any]], authority: str) -> int:
    return sum(
        1
        for row in node_results
        if row.get("authority") == authority and row.get("accepted") is not True
    )


def _count_z3_execution_failures(node_results: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in node_results
        if row.get("z3_result") and row["z3_result"].get("z3_executed") is not True
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _validate_rate(name: str, value: Any) -> None:
    if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a rate in [0, 1]")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:  # pragma: no cover - external output paths are not used by this experiment.
        return path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for Exp 3006."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--table", default=str(DEFAULT_TABLE_PATH))
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / exp3005.VALIDATOR_MANIFEST_REL_PATH),
    )
    args = parser.parse_args(argv)
    artifact = run_diagnostic(
        ExperimentConfig(
            output_path=Path(args.output),
            table_path=Path(args.table),
            manifest_path=Path(args.manifest),
        )
    )
    print(
        "[exp3006] "
        f"verdict={artifact['honest_verdict']} "
        f"trajectories={artifact['n_trajectories']} "
        f"convergence={artifact['convergence_rate']:.3f}"
    )
    return 0 if artifact["fixed_point_diagnostic_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
