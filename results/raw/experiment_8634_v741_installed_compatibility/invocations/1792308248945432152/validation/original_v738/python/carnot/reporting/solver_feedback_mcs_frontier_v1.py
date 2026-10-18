"""Exp 2979 deterministic solver-feedback frontier for NL-to-Z3 repair.

The goal is to turn prior exact-verifier failures into bounded, machine-readable
feedback. Exp 2966 supplies accepted SMT-LIB references; Exp 2967 supplies live
model parse and solver failures. This module combines them without fresh model
calls so Exp 2980 can consume concrete failure fields instead of free-form prose.

Spec: REQ-VERIFY-2979, SCENARIO-VERIFY-2979.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

try:  # pragma: no cover - dependency absence is represented by z3_module=None tests.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None

from carnot.eval import logic_frontier_materializer as exp2966


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2979_solver_feedback_mcs_frontier_v1.json"
EXP2966_FILENAME = "experiment_2966_logic_frontier_materializer_v1.json"
EXP2967_FILENAME = "experiment_2967_sota_nl_to_z3_dccd_formalization_v1.json"
INFERENCE_SUBSTRATE = "deterministic_z3_and_artifact_generation"
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "mcs_feedback_schema_ready",
    "frontier_upgrade_ready",
    "reference_z3_execution_rate",
    "reference_solver_verified_accuracy",
    "feedback_schema",
    "frontier_items",
    "failure_categories_from_exp2967",
    "mcs_mus_examples",
    "exp2980_input_path",
    "inference_substrate",
    "duration_s",
)
REQUIRED_FEEDBACK_FIELDS: tuple[str, ...] = (
    "parse_error",
    "z3_exception",
    "model_counterexample",
    "unsat_core_or_mus",
    "minimal_correction_hint",
    "skill_label",
    "accepted_reference_formalization",
)
FAILURE_CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "parse_errors": ("unparseable",),
    "execution_errors": ("z3_exception",),
    "solver_wrong": ("wrong_formula", "wrong_answer"),
}
SKILL_HINTS: dict[str, str] = {
    "symbolization": "Use the reference predicate and constant inventory before writing assertions.",
    "quantifier handling": "Keep universal and existential scopes explicit in SMT-LIB.",
    "countermodel construction": "When the target is satisfiable, preserve a model-producing negated goal.",
    "satisfiability": "Separate contradictory assertions so a minimal unsat subset can name the conflict.",
    "validity": "Encode validity by asserting the premises plus the negated conclusion.",
    "answer extraction": "Declare answer variables and constrain them before reading model values.",
}


@dataclass(frozen=True)
class SolverFeedbackConfig:
    """Runtime paths for the deterministic Exp 2979 artifact builder."""

    repo_root: Path = Path(__file__).resolve().parents[3]
    output_path: Path | None = None
    exp2966_path: Path | None = None
    exp2967_path: Path | None = None
    started_at: float | None = None
    clock: ClockFn = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_exp2966_path(self) -> Path:
        return self.exp2966_path or self.repo_root / "results" / EXP2966_FILENAME

    def resolved_exp2967_path(self) -> Path:
        return self.exp2967_path or self.repo_root / "results" / EXP2967_FILENAME


def build_artifact(
    config: SolverFeedbackConfig | None = None,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Build the Exp 2979 payload without writing it to disk."""

    active = config or SolverFeedbackConfig()
    started_at = active.start_time()
    output_path = active.resolved_output_path()
    if z3_module is None:
        return _blocked_artifact(
            active,
            started_at,
            reason="blocked_dependency: z3 import failed",
            diagnostics=_environment_diagnostics(False, "z3_module_is_none"),
        )

    exp2966_artifact = _read_json(active.resolved_exp2966_path())
    exp2967_artifact = _read_json(active.resolved_exp2967_path())
    manifest_path = _manifest_path(active.repo_root, exp2966_artifact)
    manifest = _read_json(manifest_path)
    manifest_items = [item for item in manifest.get("items") or [] if isinstance(item, Mapping)]
    selected = _select_frontier_items_by_skill(manifest_items)
    failure_rows = _failure_rows(exp2967_artifact)
    failure_categories = _failure_categories_from_exp2967(exp2967_artifact, failure_rows)

    frontier_items: list[JsonDict] = []
    for skill_label, item in selected:
        reference = _accepted_reference_formalization(item)
        reference_result = _execute_reference(item, z3_module=z3_module)
        source_failure = _failure_row_for_skill(failure_rows, skill_label)
        feedback = _solver_feedback(
            item=item,
            skill_label=skill_label,
            reference=reference,
            source_failure=source_failure,
            z3_module=z3_module,
        )
        frontier_items.append(
            {
                "item_id": str(item.get("item_id")),
                "prompt": str(item.get("prompt")),
                "skill_label": skill_label,
                "skill_labels": list(item.get("skill_labels") or []),
                "expected_solver_status": str(item.get("expected_solver_status")),
                "accepted_reference_formalization": reference,
                "reference_z3_result": reference_result,
                "solver_feedback": feedback,
            }
        )

    execution_rate, accuracy = _reference_rates(frontier_items)
    schema_ready = _feedback_schema_ready(frontier_items)
    fixture_ready = (
        schema_ready
        and {row["skill_label"] for row in frontier_items} == set(exp2966.SKILL_LABELS)
        and execution_rate == 1.0
        and accuracy == 1.0
        and str(output_path)
    )
    artifact = {
        "honest_verdict": (
            "complete: deterministic solver-feedback MCS/MUS frontier ready for Exp 2980"
            if fixture_ready
            else "blocked_or_incomplete: solver-feedback frontier is not consumable"
        ),
        "mcs_feedback_schema_ready": bool(fixture_ready),
        "frontier_upgrade_ready": bool(fixture_ready),
        "reference_z3_execution_rate": execution_rate,
        "reference_solver_verified_accuracy": accuracy,
        "feedback_schema": feedback_schema(ready=bool(fixture_ready)),
        "frontier_items": frontier_items,
        "failure_categories_from_exp2967": failure_categories,
        "mcs_mus_examples": _mcs_mus_examples(frontier_items),
        "exp2980_input_path": str(output_path),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": active.clock() - started_at,
        "run_date": RUN_DATE,
        "source_artifacts": {
            "exp2966": str(active.resolved_exp2966_path()),
            "exp2967": str(active.resolved_exp2967_path()),
            "manifest": str(manifest_path),
        },
        "preconditions_checked": [
            {"name": "z3_import", "ok": True, "detail": _z3_version(z3_module)},
            {
                "name": "exp2966_logic_frontier_materialized",
                "ok": bool(exp2966_artifact.get("logic_frontier_materialized")),
                "detail": str(active.resolved_exp2966_path()),
            },
            {"name": "exp2967_failure_categories_loaded", "ok": True, "detail": str(active.resolved_exp2967_path())},
        ],
        "environment_diagnostics": _environment_diagnostics(True, None),
    }
    return artifact


def write_artifact(config: SolverFeedbackConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2979 terminal artifact."""

    active = config or SolverFeedbackConfig()
    payload = build_artifact(active)
    output_path = active.resolved_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def feedback_schema(*, ready: bool = True) -> JsonDict:
    """Return the machine-readable schema for one `solver_feedback` object."""

    return {
        "object": "solver_feedback",
        "version": 1,
        "ready_for_exp2980": ready,
        "required": list(REQUIRED_FEEDBACK_FIELDS),
        "fields": {
            "parse_error": {
                "type": "string|null",
                "description": "Structured parse failure such as no_json_object or missing_schema_field.",
            },
            "z3_exception": {
                "type": "string|null",
                "description": "Solver compilation or execution exception for parseable proposals.",
            },
            "model_counterexample": {
                "type": "object|null",
                "description": "A bounded Z3 model for satisfiable references or wrong-formula repair.",
            },
            "unsat_core_or_mus": {
                "type": "object|null",
                "description": "Named unsat core and minimal unsat subsets for local contradiction repair.",
            },
            "minimal_correction_hint": {
                "type": "string",
                "description": "Skill-specific deterministic repair hint, not a verifier decision.",
            },
            "skill_label": {"type": "string", "description": "One Exp 2966 skill label."},
            "accepted_reference_formalization": {
                "type": "object",
                "description": "The accepted SMT-LIB reference Exp 2980 should preserve.",
            },
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise if the artifact violates the Exp 2979 terminal contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    schema = artifact.get("feedback_schema")
    fields = schema.get("fields") if isinstance(schema, Mapping) else {}
    if artifact.get("mcs_feedback_schema_ready") and not set(REQUIRED_FEEDBACK_FIELDS) <= set(fields):
        raise ValueError("schema ready requires all feedback fields")
    if artifact.get("frontier_upgrade_ready"):
        if artifact.get("reference_z3_execution_rate") != 1.0:
            raise ValueError("frontier upgrade requires full reference Z3 execution")
        if artifact.get("reference_solver_verified_accuracy") != 1.0:
            raise ValueError("frontier upgrade requires perfect reference solver accuracy")


def _blocked_artifact(
    active: SolverFeedbackConfig,
    started_at: float,
    *,
    reason: str,
    diagnostics: JsonDict,
) -> JsonDict:
    return {
        "honest_verdict": reason,
        "mcs_feedback_schema_ready": False,
        "frontier_upgrade_ready": False,
        "reference_z3_execution_rate": 0.0,
        "reference_solver_verified_accuracy": 0.0,
        "feedback_schema": feedback_schema(ready=False),
        "frontier_items": [],
        "failure_categories_from_exp2967": {
            "raw": {},
            "parse_errors": {},
            "execution_errors": {},
            "solver_wrong": {},
        },
        "mcs_mus_examples": [],
        "exp2980_input_path": str(active.resolved_output_path()),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": active.clock() - started_at,
        "run_date": RUN_DATE,
        "preconditions_checked": [
            {"name": "z3_import", "ok": False, "detail": diagnostics["z3_import_error"]}
        ],
        "environment_diagnostics": diagnostics,
    }


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(repo_root: Path, exp2966_artifact: Mapping[str, Any]) -> Path:
    raw = Path(str(exp2966_artifact.get("manifest_path") or ""))
    return raw if raw.is_absolute() else repo_root / raw


def _select_frontier_items_by_skill(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    selected: list[tuple[str, Mapping[str, Any]]] = []
    used: set[str] = set()
    for skill_label in exp2966.SKILL_LABELS:
        candidates = [item for item in items if skill_label in set(item.get("skill_labels") or [])]
        fresh = [item for item in candidates if str(item.get("item_id")) not in used]
        item = fresh[0] if fresh else candidates[0]
        used.add(str(item.get("item_id")))
        selected.append((skill_label, item))
    return selected


def _failure_rows(exp2967_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in exp2967_artifact.get("per_item_results") or [] if isinstance(row, Mapping)]


def _failure_categories_from_exp2967(
    exp2967_artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    raw = dict(exp2967_artifact.get("failure_categories") or Counter(_failure_category(row) for row in rows))
    return {
        "raw": raw,
        "parse_errors": {name: int(raw.get(name, 0)) for name in FAILURE_CATEGORY_GROUPS["parse_errors"]},
        "execution_errors": {
            name: int(raw.get(name, 0)) for name in FAILURE_CATEGORY_GROUPS["execution_errors"]
        },
        "solver_wrong": {name: int(raw.get(name, 0)) for name in FAILURE_CATEGORY_GROUPS["solver_wrong"]},
        "solver_verified_correct": int(raw.get("solver_verified_correct", 0)),
    }


def _failure_row_for_skill(
    rows: Sequence[Mapping[str, Any]],
    skill_label: str,
) -> Mapping[str, Any] | None:
    skill_rows = [
        row
        for row in rows
        if skill_label in set(row.get("skill_labels") or [])
        and _failure_category(row) != "solver_verified_correct"
    ]
    fallback_rows = [row for row in rows if _failure_category(row) != "solver_verified_correct"]
    for category in ("unparseable", "z3_exception", "wrong_formula", "wrong_answer"):
        for row in skill_rows or fallback_rows:
            if _failure_category(row) == category:
                return row
    return None


def _solver_feedback(
    *,
    item: Mapping[str, Any],
    skill_label: str,
    reference: JsonDict,
    source_failure: Mapping[str, Any] | None,
    z3_module: Any,
) -> JsonDict:
    category = _failure_category(source_failure or {})
    return {
        "parse_error": str(source_failure.get("parse_error")) if category == "unparseable" and source_failure else None,
        "z3_exception": _z3_error(source_failure) if category == "z3_exception" else None,
        "model_counterexample": _model_counterexample(item, z3_module)
        if str(item.get("expected_solver_status")) == "sat" or category in {"wrong_formula", "wrong_answer"}
        else None,
        "unsat_core_or_mus": _unsat_core_or_mus(skill_label, z3_module)
        if str(item.get("expected_solver_status")) == "unsat"
        else None,
        "minimal_correction_hint": SKILL_HINTS[skill_label],
        "skill_label": skill_label,
        "accepted_reference_formalization": reference,
    }


def _accepted_reference_formalization(item: Mapping[str, Any]) -> JsonDict:
    reference_z3 = item.get("reference_z3") or {}
    return {
        "format": "smt2",
        "assertions": str(reference_z3.get("assertions") or ""),
        "expected_solver_status": str(item.get("expected_solver_status")),
        "expected_answer_values": dict(item.get("expected_answer_values") or {}),
    }


def _execute_reference(item: Mapping[str, Any], *, z3_module: Any) -> JsonDict:
    reference = _accepted_reference_formalization(item)
    logic_item = exp2966.LogicFrontierItem(
        item_id=str(item.get("item_id")),
        prompt=str(item.get("prompt")),
        expected_label=str(item.get("expected_label")),
        check_kind=str(item.get("check_kind")),
        expected_solver_status=reference["expected_solver_status"],
        skill_labels=tuple(str(label) for label in item.get("skill_labels") or ()),
        reference_smt2=reference["assertions"],
        expected_answer_values=reference["expected_answer_values"],
    )
    return exp2966.execute_reference_formalization(logic_item, z3_module=z3_module)


def _reference_rates(frontier_items: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    if not frontier_items:
        return 0.0, 0.0
    executed = sum(1 for row in frontier_items if row["reference_z3_result"]["z3_executed"])
    accurate = sum(
        1
        for row in frontier_items
        if row["reference_z3_result"]["solver_status_matches_expected"]
        and row["reference_z3_result"]["answer_extraction_matches_expected"]
    )
    return executed / len(frontier_items), accurate / len(frontier_items)


def _feedback_schema_ready(frontier_items: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        frontier_items
        and all(set(REQUIRED_FEEDBACK_FIELDS) <= set(row["solver_feedback"]) for row in frontier_items)
    )


def _mcs_mus_examples(frontier_items: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    examples = []
    for row in frontier_items:
        feedback = row["solver_feedback"]
        if feedback["unsat_core_or_mus"]:
            examples.append(
                {
                    "item_id": row["item_id"],
                    "skill_label": row["skill_label"],
                    "unsat_core_or_mus": feedback["unsat_core_or_mus"],
                    "minimal_correction_hint": feedback["minimal_correction_hint"],
                }
            )
    return examples


def _model_counterexample(item: Mapping[str, Any], z3_module: Any) -> JsonDict | None:
    solver = z3_module.Solver()
    solver.add(z3_module.parse_smt2_string(_accepted_reference_formalization(item)["assertions"]))
    if str(solver.check()) != "sat":
        return None
    model = solver.model()
    assignments = {str(decl): str(model[decl]) for decl in model.decls()[:6]}
    return {
        "solver_status": "sat",
        "assignments": assignments,
        "note": "bounded model from accepted reference formalization",
    }


def _unsat_core_or_mus(skill_label: str, z3_module: Any) -> JsonDict:
    premise = z3_module.Bool(f"{_safe_name(skill_label)}_premise")
    constraints = (("premise", premise), ("negated_goal", z3_module.Not(premise)))
    solver = z3_module.Solver()
    for name, expr in constraints:
        solver.assert_and_track(expr, name)
    status = str(solver.check())
    core = [str(name) for name in solver.unsat_core()] if status == "unsat" else []
    return {
        "solver_status": status,
        "unsat_core": core,
        "minimal_unsat_subsets": _minimal_unsat_subsets(constraints, z3_module),
        "mcs_candidates": [[name] for name in core],
    }


def _minimal_unsat_subsets(
    constraints: Sequence[tuple[str, Any]],
    z3_module: Any,
) -> list[list[str]]:
    subsets: list[list[str]] = []
    for size in range(1, len(constraints) + 1):
        for combo in combinations(constraints, size):
            solver = z3_module.Solver()
            solver.add([expr for _name, expr in combo])
            if str(solver.check()) == "unsat":
                subsets.append([name for name, _expr in combo])
        if subsets:
            return subsets
    return subsets


def _failure_category(row: Mapping[str, Any]) -> str:
    return str(row.get("failure_category") or "")


def _z3_error(row: Mapping[str, Any] | None) -> str | None:
    if row is None:
        return None
    z3_result = row.get("z3_result") if isinstance(row.get("z3_result"), Mapping) else {}
    return str(z3_result.get("z3_error") or row.get("z3_error") or "")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "skill"


def _z3_version(z3_module: Any) -> str:
    getter = getattr(z3_module, "get_version_string", None)
    return str(getter()) if callable(getter) else "unknown"


def _environment_diagnostics(z3_import_ok: bool, z3_import_error: str | None) -> JsonDict:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "z3_import_ok": z3_import_ok,
        "z3_import_error": z3_import_error,
    }


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    write_artifact()


if __name__ == "__main__":  # pragma: no cover
    main()
