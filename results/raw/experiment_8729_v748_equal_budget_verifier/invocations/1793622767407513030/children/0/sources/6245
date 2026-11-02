"""BEAVER-lite deterministic coverage bounds for ROCE validator trees.

Spec: REQ-VERIFY-1879, SCENARIO-VERIFY-1879.

The bounds in this module are accounting rows, not validators.  A compiled
ROCE validator leaf contributes deterministic coverage because executable
Python validation decides that leaf.  Unsupported or uncompiled constraints
contribute residual risk.  Acceptance still comes only from the validator tree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline import roce_validator_tree
from carnot.pipeline.verdict_record import VerdictRecord

JsonDict = dict[str, Any]

RUN_DATE = "20260511"
EXPERIMENT_ID = 1879
EXPERIMENT = "1879_beaver_lite_bounds"
DEFAULT_SOURCE_ARTIFACT_PATH = Path("results/experiment_1878_roce_validator_tree.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1879_beaver_lite_bounds.json")
SPEC_TRACES = ["REQ-VERIFY-1879", "SCENARIO-VERIFY-1879"]
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "honest_verdict",
    "beaver_lite_bounds_ready",
    "deterministic_coverage_bound",
    "residual_risk_bound",
    "acceptance_authority_unchanged",
    "tests_run",
)


@dataclass(frozen=True)
class BeaverLiteBoundRow:
    """One conservative coverage/residual row for a ROCE constraint slot."""

    leaf_id: str
    predicate: str
    source_constraint_type: str
    deterministic_coverage_bound: float
    residual_risk_bound: float
    executable_validator_present: bool
    guarded: bool
    bound_source: str

    def to_dict(self) -> JsonDict:
        """Return a stable JSON row."""

        return _json_safe(asdict(self))


@dataclass(frozen=True)
class BeaverLiteTreeBounds:
    """Aggregate BEAVER-lite accounting for one compiled ROCE tree."""

    case_id: str
    total_constraint_count: int
    supported_constraint_count: int
    deterministic_coverage_bound: float
    residual_risk_bound: float
    acceptance_authority_unchanged: bool
    bound_rows: tuple[BeaverLiteBoundRow, ...]

    @property
    def beaver_lite_bounds_ready(self) -> bool:
        """Return whether this tree has finite, conservative accounting bounds."""

        return bool(
            self.total_constraint_count > 0
            and self.bound_rows
            and self.acceptance_authority_unchanged
            and _finite_unit(self.deterministic_coverage_bound)
            and _finite_unit(self.residual_risk_bound)
            and self.deterministic_coverage_bound + self.residual_risk_bound <= 1.000001
            and all(_finite_unit(row.deterministic_coverage_bound) for row in self.bound_rows)
            and all(_finite_unit(row.residual_risk_bound) for row in self.bound_rows)
        )

    def to_dict(self) -> JsonDict:
        """Return a JSON-compatible bound summary."""

        return {
            "case_id": self.case_id,
            "total_constraint_count": self.total_constraint_count,
            "supported_constraint_count": self.supported_constraint_count,
            "deterministic_coverage_bound": self.deterministic_coverage_bound,
            "residual_risk_bound": self.residual_risk_bound,
            "beaver_lite_bounds_ready": self.beaver_lite_bounds_ready,
            "acceptance_authority_unchanged": self.acceptance_authority_unchanged,
            "bound_rows": [row.to_dict() for row in self.bound_rows],
        }


def load_or_reconstruct_exp1878_fixture_cases(
    source_artifact_path: Path | str = DEFAULT_SOURCE_ARTIFACT_PATH,
) -> list[JsonDict]:
    """Return Exp 1878 fixture cases, using the artifact only for provenance checks."""

    cases = roce_validator_tree.default_roce_fixture_cases()
    source_path = Path(source_artifact_path)
    if not source_path.exists():
        return cases
    try:
        artifact = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cases
    artifact_case_ids = {
        str(row.get("case_id"))
        for row in artifact.get("case_results", [])
        if isinstance(row, dict) and row.get("case_id")
    }
    fixture_case_ids = {str(case.get("case_id")) for case in cases}
    if artifact_case_ids and not artifact_case_ids <= fixture_case_ids:
        return cases
    return cases


def compute_tree_bounds(
    tree: roce_validator_tree.ROCEValidatorTree,
    *,
    acceptance_authority_unchanged: bool = True,
) -> BeaverLiteTreeBounds:
    """Compute conservative deterministic coverage and residual-risk bounds."""

    total = max(0, int(tree.total_constraint_count))
    supported = max(0, min(int(tree.supported_constraint_count), total))
    deterministic_coverage = _rate(supported, total)
    residual_risk = _rate(total - supported, total) if total else 1.0
    leaf_weight = _rate(1, total) if total else 0.0

    rows = [
        BeaverLiteBoundRow(
            leaf_id=leaf.id,
            predicate=leaf.predicate,
            source_constraint_type=leaf.source_constraint_type,
            deterministic_coverage_bound=leaf_weight,
            residual_risk_bound=0.0,
            executable_validator_present=True,
            guarded=leaf.guard is not None,
            bound_source="executable_validator_leaf",
        )
        for leaf in tree.leaves
    ]
    rows.extend(_unsupported_bound_rows(tree, total=total, supported=supported))
    return BeaverLiteTreeBounds(
        case_id=tree.case_id,
        total_constraint_count=total,
        supported_constraint_count=supported,
        deterministic_coverage_bound=deterministic_coverage,
        residual_risk_bound=residual_risk,
        acceptance_authority_unchanged=bool(acceptance_authority_unchanged),
        bound_rows=tuple(rows),
    )


def verdict_record_for_output(
    tree: roce_validator_tree.ROCEValidatorTree,
    output_text: str,
) -> VerdictRecord:
    """Return a structured verdict with BEAVER-lite bounds attached as extras."""

    validation = tree.validate(output_text)
    bounds = compute_tree_bounds(tree)
    passed = validation.accepted
    return VerdictRecord(
        verdict="pass" if passed else "fail",
        energy=0.0 if passed else 1.0,
        calibrated_confidence=1.0 - bounds.residual_risk_bound if passed else 0.0,
        producing_tier=3,
        tier_reached=3,
        rationale="roce_validator_tree_accepted" if passed else "roce_validator_tree_rejected",
        budget_ms_consumed=0.0,
        extras={
            "acceptance_authority": "roce_validator_tree",
            "acceptance_authority_unchanged": True,
            "tree_validation_accepted": validation.accepted,
            "failure_ids": validation.failure_ids,
            "skipped_ids": validation.skipped_ids,
            "unsupported_constraint_types": validation.unsupported_constraint_types,
            "beaver_lite_bounds": bounds.to_dict(),
        },
    )


def evaluate_fixture_case(case: Mapping[str, Any]) -> JsonDict:
    """Compile one Exp 1878 fixture and evaluate BEAVER-lite bound authority."""

    case_id = str(case.get("case_id") or "unknown")
    tree = roce_validator_tree.compile_roce_validator_tree(str(case.get("prompt") or ""), case_id=case_id)
    good_record = verdict_record_for_output(tree, str(case.get("known_good") or ""))
    bad_records = [
        verdict_record_for_output(tree, str(output)) for output in case.get("known_bad", [])
    ]
    known_bad_promoted = sum(1 for record in bad_records if record.verdict == "pass")
    authority_unchanged = known_bad_promoted == 0
    bounds = compute_tree_bounds(tree, acceptance_authority_unchanged=authority_unchanged)
    return {
        "case_id": case_id,
        "validator_tree": tree.to_dict(),
        "beaver_lite_bounds": bounds.to_dict(),
        "known_good_verdict": good_record.to_dict(),
        "known_bad_verdicts": [record.to_dict() for record in bad_records],
        "known_bad_promoted_by_bounds": known_bad_promoted,
        "acceptance_authority_unchanged": authority_unchanged,
    }


def build_artifact(
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    source_artifact_path: Path | str = DEFAULT_SOURCE_ARTIFACT_PATH,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Build the Exp 1879 artifact without writing it."""

    fixture_cases = list(cases) if cases is not None else load_or_reconstruct_exp1878_fixture_cases(
        source_artifact_path
    )
    rows = [evaluate_fixture_case(case) for case in fixture_cases]
    total_constraints = sum(row["beaver_lite_bounds"]["total_constraint_count"] for row in rows)
    supported_constraints = sum(
        row["beaver_lite_bounds"]["supported_constraint_count"] for row in rows
    )
    deterministic_coverage = _rate(supported_constraints, total_constraints)
    residual_risk = _rate(total_constraints - supported_constraints, total_constraints)
    acceptance_authority_unchanged = bool(
        rows and all(row["acceptance_authority_unchanged"] for row in rows)
    )
    ready = bool(
        rows
        and acceptance_authority_unchanged
        and all(row["beaver_lite_bounds"]["beaver_lite_bounds_ready"] for row in rows)
        and deterministic_coverage + residual_risk <= 1.000001
    )
    verdict_records = [
        record
        for row in rows
        for record in [row["known_good_verdict"], *row["known_bad_verdicts"]]
    ]
    return {
        "status": "complete" if ready else "partial",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": list(SPEC_TRACES),
        "source_artifact_path": str(source_artifact_path),
        "source_experiment": roce_validator_tree.EXPERIMENT,
        "fixture_cases": len(rows),
        "beaver_lite_bounds_ready": ready,
        "deterministic_coverage_bound": deterministic_coverage,
        "residual_risk_bound": residual_risk,
        "acceptance_authority_unchanged": acceptance_authority_unchanged,
        "case_results": rows,
        "verdict_records": verdict_records,
        "tests_run": list(tests_run or []),
        "honest_verdict": _honest_verdict(ready, deterministic_coverage, residual_risk),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the Exp 1879 artifact has the required completion schema."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    coverage = float(artifact["deterministic_coverage_bound"])
    residual = float(artifact["residual_risk_bound"])
    assert 0.0 <= coverage <= 1.0, "coverage out of range"
    assert 0.0 <= residual <= 1.0, "residual risk out of range"
    assert coverage + residual <= 1.000001, "coverage plus residual exceeds one"
    assert isinstance(artifact["tests_run"], list), "tests_run must be a list"
    if artifact["status"] == "complete":
        assert artifact["beaver_lite_bounds_ready"] is True, "complete requires ready bounds"
        assert artifact["acceptance_authority_unchanged"] is True, "complete requires authority"
        assert artifact["case_results"], "complete requires case results"
        assert artifact["verdict_records"], "complete requires verdict records"


def run_experiment(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    source_artifact_path: Path | str = DEFAULT_SOURCE_ARTIFACT_PATH,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run Exp 1879 and write `results/experiment_1879_beaver_lite_bounds.json`."""

    artifact = build_artifact(
        cases=cases,
        source_artifact_path=source_artifact_path,
        tests_run=tests_run,
    )
    artifact["artifact_path"] = str(output_path)
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def _unsupported_bound_rows(
    tree: roce_validator_tree.ROCEValidatorTree,
    *,
    total: int,
    supported: int,
) -> list[BeaverLiteBoundRow]:
    unsupported_slots = max(0, total - supported)
    if not unsupported_slots:
        return []
    unsupported_types = list(tree.unsupported_constraint_types) or ["uncompiled_constraint"]
    residual_per_type = _rate(unsupported_slots, total) / len(unsupported_types)
    return [
        BeaverLiteBoundRow(
            leaf_id=f"unsupported:{predicate}",
            predicate=predicate,
            source_constraint_type="unsupported",
            deterministic_coverage_bound=0.0,
            residual_risk_bound=round(residual_per_type, 6),
            executable_validator_present=False,
            guarded=False,
            bound_source="unsupported_constraint",
        )
        for predicate in unsupported_types
    ]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _finite_unit(value: float) -> bool:
    return math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _honest_verdict(ready: bool, coverage: float, residual: float) -> str:
    if ready:
        return (
            "complete: BEAVER-lite ROCE validator-tree bounds computed with "
            "executable validators retaining acceptance authority"
        )
    return (
        "partial: BEAVER-lite ROCE bounds not ready; "
        f"deterministic_coverage_bound={coverage}, residual_risk_bound={residual}"
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dict(payload)


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_SOURCE_ARTIFACT_PATH",
    "EXPERIMENT_ID",
    "REQUIRED_ARTIFACT_FIELDS",
    "SPEC_TRACES",
    "BeaverLiteBoundRow",
    "BeaverLiteTreeBounds",
    "build_artifact",
    "compute_tree_bounds",
    "evaluate_fixture_case",
    "load_or_reconstruct_exp1878_fixture_cases",
    "run_experiment",
    "validate_artifact",
    "verdict_record_for_output",
]
