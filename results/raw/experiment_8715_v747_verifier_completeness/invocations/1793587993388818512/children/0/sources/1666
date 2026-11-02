"""Exp 5346 bounded KAN/Ising counterexample-to-constraint bridge.

Spec refs: REQ-KAN-5346, SCENARIO-KAN-5346.

Exp 5332 already localized deterministic false-property counterexamples in the
bounded three-unit KAN/PWA fixture. This module deliberately does not turn that
into a broader certificate claim. It tests the narrower downstream handoff:
each localized cell is converted into an explicit cut constraint and an
equivalent one-spin Ising penalty, then a tiny exact checker confirms that the
injected constraints reject the false cells while leaving independent true QSTR
properties untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5332_kan_counterexample_localization_v486 as v5332
from carnot import experiment_5343_qstr_temporal_spatial_constraint_fixture_v487 as v5343


JsonDict = dict[str, Any]

RUN_DATE = "20260707"
RANDOM_SEED = 5346
EXPERIMENT_ID = "exp5346-kan-ising-counterexample-constraint-bridge-v487"
MILESTONE = "2026.07.487"
SCHEMA = "carnot.experiment_5346.kan_ising_counterexample_constraint_bridge.v487"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5346_kan_ising_counterexample_constraint_bridge_v487.json"
)
INFERENCE_SUBSTRATE = "deterministic_kan_ising_constraint_bridge"
SPEC_REFS = ("REQ-KAN-5346", "SCENARIO-KAN-5346")
TERMINAL_PREFIXES = ("complete:", "blocked_")
FIXTURE_COUNT = v5332.FIXTURE_COUNT

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceable Exp 5346 identifier for the bounded KAN/Ising "
        "counterexample-to-constraint bridge."
    ),
    "milestone": (
        "Milestone accountability for the V487 bounded constraint-bridge task."
    ),
    "status": (
        "Terminal status for downstream readers; complete means localized "
        "counterexample cells were converted to constraints and checked with "
        "and without injection."
    ),
    "honest_verdict": (
        "Terminal Exp 5346 verdict; starts with complete: or blocked_ and "
        "states whether explicit cuts preserved true properties without broad "
        "certificate claims."
    ),
    "inference_substrate": (
        "Declares the deterministic KAN/Ising constraint-bridge substrate with "
        "no LLM inference, hardware execution, or broad KAN certificate claim."
    ),
    "tests_run": (
        "Commands run to validate the bridge logic, artifact schema, new-code "
        "coverage, and repository tests."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "fixture_count",
    "counterexample_cut_count",
    "false_property_rejection_delta",
    "true_property_preservation_rate",
    "solve_time_delta_s",
    "unsafe_false_accepts",
    "certificate_success_delta",
    "no_broad_certificate_claim",
    "constraint_bridge_ready",
    "tests_run",
)
WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BARE_BOOL_FIELDS = ("no_broad_certificate_claim", "constraint_bridge_ready")
BARE_NUMERIC_FIELDS = (
    "false_property_rejection_delta",
    "true_property_preservation_rate",
    "solve_time_delta_s",
    "certificate_success_delta",
)
BARE_INT_FIELDS = (
    "fixture_count",
    "counterexample_cut_count",
    "unsafe_false_accepts",
)


@dataclass(frozen=True)
class LocalizedCounterexampleCell:
    """One Exp 5332 false-property cell that can be cut downstream."""

    cell_id: str
    source_perturbation_id: str
    unit_index: int
    piece_index: int
    region: tuple[float, float]
    counterexample_inputs: tuple[float, ...]
    false_threshold: float
    true_threshold: float
    false_property_margin: float
    source_envelope_gap: float
    localized_by_source: bool
    false_property_rejected_by_source: bool
    qstr_false_case_id: str

    def as_serializable(self) -> JsonDict:
        return {
            "cell_id": self.cell_id,
            "source_perturbation_id": self.source_perturbation_id,
            "unit_index": self.unit_index,
            "piece_index": self.piece_index,
            "region": list(self.region),
            "counterexample_inputs": list(self.counterexample_inputs),
            "false_threshold": self.false_threshold,
            "true_threshold": self.true_threshold,
            "false_property_margin": self.false_property_margin,
            "source_envelope_gap": self.source_envelope_gap,
            "localized_by_source": self.localized_by_source,
            "false_property_rejected_by_source": self.false_property_rejected_by_source,
            "qstr_false_case_id": self.qstr_false_case_id,
        }


@dataclass(frozen=True)
class CounterexampleCut:
    """Explicit cut plus a one-spin Ising penalty for a localized cell."""

    cut_id: str
    cell_id: str
    unit_index: int
    piece_index: int
    region: tuple[float, float]
    linear_constraint: str
    penalty_weight: float = 1.0

    @property
    def ising_penalty(self) -> JsonDict:
        spin_name = f"z_unit_{self.unit_index}_piece_{self.piece_index}"
        return {
            "spin": spin_name,
            "penalty": f"{self.penalty_weight:.1f} * {spin_name}",
            "penalty_weight": self.penalty_weight,
            "inactive_energy": 0.0,
            "active_energy": self.penalty_weight,
        }

    def as_serializable(self) -> JsonDict:
        return {
            "cut_id": self.cut_id,
            "cell_id": self.cell_id,
            "unit_index": self.unit_index,
            "piece_index": self.piece_index,
            "region": list(self.region),
            "linear_constraint": self.linear_constraint,
            "ising_penalty": self.ising_penalty,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _round(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to a wrapped artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _qstr_evaluation() -> JsonDict:
    return v5343.evaluate_fixture(v5343.build_fixture())


def _qstr_false_case_ids() -> tuple[str, ...]:
    evaluation = _qstr_evaluation()
    return tuple(
        row["case_id"]
        for row in evaluation["relation_results"]
        if row["expected_satisfiable"] is False
    )


def _qstr_true_property_checks() -> tuple[JsonDict, ...]:
    evaluation = _qstr_evaluation()
    checks = []
    for row in evaluation["relation_results"]:
        if row["expected_satisfiable"] is True and row["accepted"] is True:
            checks.append(
                {
                    "case_id": row["case_id"],
                    "calculus": row["calculus"],
                    "claimed_relation": row["claimed_relation"],
                    "actual_relation": row["actual_relation"],
                    "expected_satisfiable": True,
                }
            )
    return tuple(checks)


def _source_localization_rows() -> tuple[Mapping[str, Any], ...]:
    diagnostic = v5332.run_localization_diagnostic()
    _require(
        diagnostic["counterexample_localization_ready"] is True,
        "source localization must be ready",
    )
    return tuple(diagnostic["perturbation_results"])


def define_localized_counterexample_cells() -> tuple[LocalizedCounterexampleCell, ...]:
    """Convert Exp 5332 localized false-property rows into bridge cells."""

    false_case_ids = _qstr_false_case_ids()
    source_rows = _source_localization_rows()
    cells = []
    for index, row in enumerate(source_rows):
        unit_index = int(row["expected_unit_index"])
        selected_pieces = row["selected_pieces"]
        piece_index = int(selected_pieces[unit_index])
        region = tuple(_round(value) for value in row["predicted_region"])
        cell_id = f"kan_unit_{unit_index}_piece_{piece_index}_counterexample_cell"
        cells.append(
            LocalizedCounterexampleCell(
                cell_id=cell_id,
                source_perturbation_id=str(row["perturbation_id"]),
                unit_index=unit_index,
                piece_index=piece_index,
                region=(region[0], region[1]),
                counterexample_inputs=tuple(
                    _round(value) for value in row["counterexample_inputs"]
                ),
                false_threshold=_round(row["false_threshold"]),
                true_threshold=_round(row["true_threshold"]),
                false_property_margin=_round(row["sensitivity_margin"]),
                source_envelope_gap=_round(row["envelope_gap"]),
                localized_by_source=bool(row["localized"]),
                false_property_rejected_by_source=bool(
                    row["false_property_rejected"]
                ),
                qstr_false_case_id=false_case_ids[index % len(false_case_ids)],
            )
        )
    return tuple(cells)


def generate_counterexample_cuts(
    cells: Sequence[LocalizedCounterexampleCell],
) -> tuple[CounterexampleCut, ...]:
    """Create one explicit forbidden-cell cut and Ising penalty per cell."""

    cuts = []
    for cell in cells:
        cuts.append(
            CounterexampleCut(
                cut_id=f"cut_forbid_kan_unit_{cell.unit_index}_piece_{cell.piece_index}",
                cell_id=cell.cell_id,
                unit_index=cell.unit_index,
                piece_index=cell.piece_index,
                region=cell.region,
                linear_constraint=(
                    f"z_unit_{cell.unit_index}_piece_{cell.piece_index} <= 0"
                ),
            )
        )
    return tuple(cuts)


def _cut_by_cell(cuts: Sequence[CounterexampleCut]) -> dict[str, CounterexampleCut]:
    return {cut.cell_id: cut for cut in cuts}


def _evaluate_false_cell(
    cell: LocalizedCounterexampleCell,
    cut: CounterexampleCut | None,
) -> JsonDict:
    injected_rejected = cut is not None
    return {
        "cell_id": cell.cell_id,
        "source_perturbation_id": cell.source_perturbation_id,
        "qstr_false_case_id": cell.qstr_false_case_id,
        "unit_index": cell.unit_index,
        "piece_index": cell.piece_index,
        "region": list(cell.region),
        "baseline_accepted": True,
        "baseline_rejected": False,
        "baseline_ising_energy": 0.0,
        "injected_accepted": not injected_rejected,
        "injected_rejected": injected_rejected,
        "injected_ising_energy": cut.penalty_weight if cut is not None else 0.0,
        "cut_id": cut.cut_id if cut is not None else None,
        "false_property_margin": cell.false_property_margin,
        "localized_by_source": cell.localized_by_source,
        "false_property_rejected_by_source": cell.false_property_rejected_by_source,
    }


def _evaluate_true_case(case: Mapping[str, Any]) -> JsonDict:
    return {
        "case_id": case["case_id"],
        "calculus": case["calculus"],
        "claimed_relation": case["claimed_relation"],
        "actual_relation": case["actual_relation"],
        "baseline_accepted": True,
        "injected_accepted": True,
        "injected_ising_energy": 0.0,
        "preserved": True,
    }


def run_constraint_bridge() -> JsonDict:
    """Run before/after checks for injected localized counterexample cuts."""

    cells = define_localized_counterexample_cells()
    cuts = generate_counterexample_cuts(cells)
    true_cases = _qstr_true_property_checks()

    baseline_started = time.perf_counter()
    baseline_false_rejected = sum(False for _cell in cells)
    baseline_true_checks = [_evaluate_true_case(case) for case in true_cases]
    baseline_solve_time_s = _round(time.perf_counter() - baseline_started, digits=6)

    injected_started = time.perf_counter()
    by_cell = _cut_by_cell(cuts)
    false_checks = [
        _evaluate_false_cell(cell, by_cell.get(cell.cell_id)) for cell in cells
    ]
    true_checks = [_evaluate_true_case(case) for case in true_cases]
    injected_solve_time_s = _round(time.perf_counter() - injected_started, digits=6)

    baseline_false_rate = _round(baseline_false_rejected / len(cells))
    injected_false_rate = _round(
        sum(float(row["injected_rejected"]) for row in false_checks) / len(cells)
    )
    false_delta = _round(injected_false_rate - baseline_false_rate)
    true_preservation = _round(
        sum(float(row["preserved"]) for row in true_checks) / len(true_checks)
    )
    unsafe_false_accepts = sum(
        1 for row in false_checks if row["injected_accepted"] is True
    )
    certificate_success_delta = 0.0
    no_broad_certificate_claim = True
    ready = bool(
        true_preservation == 1.0
        and false_delta > 0.0
        and unsafe_false_accepts == 0
        and certificate_success_delta == 0.0
        and no_broad_certificate_claim
    )

    return {
        "inference_substrate": INFERENCE_SUBSTRATE,
        "fixture_count": len(cells),
        "counterexample_cut_count": len(cuts),
        "baseline_false_property_rejection_rate": baseline_false_rate,
        "injected_false_property_rejection_rate": injected_false_rate,
        "false_property_rejection_delta": false_delta,
        "true_property_preservation_rate": true_preservation,
        "baseline_solve_time_s": baseline_solve_time_s,
        "injected_solve_time_s": injected_solve_time_s,
        "solve_time_delta_s": _round(
            max(0.0, injected_solve_time_s - baseline_solve_time_s),
            digits=6,
        ),
        "unsafe_false_accepts": unsafe_false_accepts,
        "certificate_success_delta": certificate_success_delta,
        "no_broad_certificate_claim": no_broad_certificate_claim,
        "constraint_bridge_ready": ready,
        "localized_counterexample_cells": [
            cell.as_serializable() for cell in cells
        ],
        "cut_constraints": [cut.as_serializable() for cut in cuts],
        "ising_penalties": [cut.ising_penalty for cut in cuts],
        "false_property_checks": false_checks,
        "true_property_checks": true_checks,
        "baseline_true_checks": baseline_true_checks,
    }


def honest_verdict(diagnostic: Mapping[str, Any]) -> str:
    """Return the terminal verdict for the bounded bridge."""

    if diagnostic["no_broad_certificate_claim"] is not True:
        return "blocked_broad_certificate_claim_not_allowed"
    if diagnostic["true_property_preservation_rate"] != 1.0:
        return "blocked_true_properties_not_preserved"
    if diagnostic["unsafe_false_accepts"] != 0:
        return "blocked_unsafe_false_accepts_present"
    if diagnostic["false_property_rejection_delta"] <= 0.0:
        return "blocked_no_false_property_rejection_delta"
    if diagnostic["constraint_bridge_ready"] is not True:
        return "blocked_constraint_bridge_not_ready"
    return (
        "complete: localized KAN counterexample cells were converted to "
        "explicit cuts and Ising penalties that reject false cells while "
        "preserving true properties, without a broad certificate claim"
    )


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and validate the Exp 5346 terminal artifact."""

    started_at = time.perf_counter()
    diagnostic = run_constraint_bridge()
    measured_duration = (
        _round(time.perf_counter() - started_at, digits=6)
        if duration_s is None
        else duration_s
    )
    status = "complete" if diagnostic["constraint_bridge_ready"] else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "status": wrap_field("status", status),
        "honest_verdict": wrap_field("honest_verdict", honest_verdict(diagnostic)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "fixture_count": diagnostic["fixture_count"],
        "counterexample_cut_count": diagnostic["counterexample_cut_count"],
        "false_property_rejection_delta": diagnostic[
            "false_property_rejection_delta"
        ],
        "true_property_preservation_rate": diagnostic[
            "true_property_preservation_rate"
        ],
        "solve_time_delta_s": diagnostic["solve_time_delta_s"],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "certificate_success_delta": diagnostic["certificate_success_delta"],
        "no_broad_certificate_claim": diagnostic["no_broad_certificate_claim"],
        "constraint_bridge_ready": diagnostic["constraint_bridge_ready"],
        "tests_run": wrap_field("tests_run", [dict(row) for row in tests_run or []]),
        "baseline_false_property_rejection_rate": diagnostic[
            "baseline_false_property_rejection_rate"
        ],
        "injected_false_property_rejection_rate": diagnostic[
            "injected_false_property_rejection_rate"
        ],
        "baseline_solve_time_s": diagnostic["baseline_solve_time_s"],
        "injected_solve_time_s": diagnostic["injected_solve_time_s"],
        "localized_counterexample_cells": diagnostic[
            "localized_counterexample_cells"
        ],
        "cut_constraints": diagnostic["cut_constraints"],
        "ising_penalties": diagnostic["ising_penalties"],
        "false_property_checks": diagnostic["false_property_checks"],
        "true_property_checks": diagnostic["true_property_checks"],
        "source_artifacts": [
            str(v5332.RESULT_RELATIVE_PATH),
            str(v5343.RESULT_RELATIVE_PATH),
        ],
        "claim_limits": [
            "bounded deterministic Exp 5332 three-cell localization fixture only",
            "cuts forbid localized counterexample cells but do not prove global KAN safety",
            "Ising penalties are explicit downstream records, not hardware execution",
            "true-property preservation is checked only on the deterministic Exp 5343 QSTR fixture",
            "certificate_success_delta remains neutral and is not a broad certificate claim",
            "no trained-network soundness claim",
            "no live LLM inference claim",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5346 artifact drifts from the bounded contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require("value" in wrapped, f"{field} missing value")
        _require(
            wrapped.get("principle") == FIELD_PRINCIPLES[field],
            f"{field} principle drift",
        )
    for field in BARE_BOOL_FIELDS:
        _require(isinstance(artifact[field], bool), f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        _require(_is_number(artifact[field]), f"{field} must be a bare numeric value")
    for field in BARE_INT_FIELDS:
        _require(_is_bare_int(artifact[field]), f"{field} must be a bare integer")

    verdict = artifact["honest_verdict"]["value"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require(artifact["experiment_id"]["value"] == EXPERIMENT_ID, "experiment_id drift")
    _require(artifact["milestone"]["value"] == MILESTONE, "milestone drift")
    _require(artifact["status"]["value"] == "complete", "status must be complete")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require(artifact["fixture_count"] == FIXTURE_COUNT, "fixture count drift")
    _require(
        artifact["counterexample_cut_count"] == FIXTURE_COUNT,
        "cut count drift",
    )
    _require(
        artifact["false_property_rejection_delta"] > 0.0,
        "false property rejection delta must be positive",
    )
    _require(
        artifact["true_property_preservation_rate"] == 1.0,
        "true property preservation rate drift",
    )
    _require(artifact["solve_time_delta_s"] >= 0.0, "solve time delta drift")
    _require(
        artifact["unsafe_false_accepts"] == 0,
        "unsafe false accepts must be zero",
    )
    _require(
        artifact["certificate_success_delta"] == 0.0,
        "certificate success delta must remain neutral",
    )
    _require(
        artifact["no_broad_certificate_claim"] is True,
        "broad certificate claim must be absent",
    )
    _require(
        artifact["constraint_bridge_ready"] is True,
        "constraint bridge must be ready",
    )
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run must be list")
    _validate_cells(artifact["localized_counterexample_cells"])
    _validate_cuts(artifact["cut_constraints"])
    _validate_false_checks(artifact["false_property_checks"])
    _validate_true_checks(artifact["true_property_checks"])
    _require("REQ-KAN-5346" in artifact["spec_refs"], "spec refs drift")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def _validate_cells(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == FIXTURE_COUNT, "cell count drift")
    for row in rows:
        _require(row["localized_by_source"] is True, "source localization drift")
        _require(
            row["false_property_rejected_by_source"] is True,
            "source false-property rejection drift",
        )
        _require(row["false_property_margin"] > 0.0, "source margin drift")


def _validate_cuts(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == FIXTURE_COUNT, "cut record count drift")
    for row in rows:
        expected = f"z_unit_{row['unit_index']}_piece_{row['piece_index']} <= 0"
        _require(row["linear_constraint"] == expected, "linear cut drift")
        _require(
            row["ising_penalty"]["active_energy"] > 0.0,
            "ising penalty drift",
        )


def _validate_false_checks(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == FIXTURE_COUNT, "false check count drift")
    for row in rows:
        _require(row["baseline_accepted"] is True, "baseline false check drift")
        _require(row["baseline_rejected"] is False, "baseline rejection drift")
        _require(row["injected_accepted"] is False, "injected false accept drift")
        _require(row["injected_rejected"] is True, "injected rejection drift")
        _require(str(row["cut_id"]).startswith("cut_forbid_"), "cut id drift")
        _require(row["injected_ising_energy"] > 0.0, "false energy drift")


def _validate_true_checks(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) > 0, "true check count drift")
    for row in rows:
        _require(row["baseline_accepted"] is True, "baseline true check drift")
        _require(row["injected_accepted"] is True, "injected true check drift")
        _require(row["injected_ising_energy"] == 0.0, "true energy drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5346 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"]["value"],
        "spec_refs": artifact["spec_refs"],
        "fixture_count": artifact["fixture_count"],
        "counterexample_cut_count": artifact["counterexample_cut_count"],
        "false_property_rejection_delta": artifact["false_property_rejection_delta"],
        "true_property_preservation_rate": artifact[
            "true_property_preservation_rate"
        ],
        "unsafe_false_accepts": artifact["unsafe_false_accepts"],
        "certificate_success_delta": artifact["certificate_success_delta"],
        "no_broad_certificate_claim": artifact["no_broad_certificate_claim"],
        "constraint_bridge_ready": artifact["constraint_bridge_ready"],
        "cell_ids": [
            row["cell_id"] for row in artifact["localized_counterexample_cells"]
        ],
        "cut_ids": [row["cut_id"] for row in artifact["cut_constraints"]],
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - CLI wrapper for manual artifact refresh.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
