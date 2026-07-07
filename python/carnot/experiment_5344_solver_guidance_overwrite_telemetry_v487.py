"""Exp5344: deterministic solver-guidance overwrite telemetry.

Spec refs: REQ-VERIFY-5344, SCENARIO-VERIFY-5344.

Hints in this module are deliberately advisory. They can change the first
search path, but final labels come from exact QSTR facts or from CDCL after
fallback. That makes the measurement about recovery from bad guidance rather
than about trusting guidance as a solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as cdcl
from carnot import experiment_5343_qstr_temporal_spatial_constraint_fixture_v487 as qstr


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_5344_solver_guidance_overwrite_telemetry_v487"
EXPERIMENT_NUMBER = 5344
MILESTONE = "2026.07.487"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5344.solver_guidance_overwrite_telemetry.v487"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5344_solver_guidance_overwrite_telemetry_v487.json"
)
INFERENCE_SUBSTRATE = "deterministic_solver_guidance_telemetry"
SPEC_REFS = ("REQ-VERIFY-5344", "SCENARIO-VERIFY-5344")
TERMINAL_PREFIXES = ("complete:", "blocked_")
HINT_CLASS_NAMES = (
    "perfect_hints",
    "partial_hints",
    "stale_hints",
    "misleading_hints",
    "no_hints",
)
SAT_STALE_HINT = (2, 8)
SAT_MISLEADING_HINT = (4, 7)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceability for the Exp5344 deterministic solver-guidance overwrite "
        "telemetry diagnostic."
    ),
    "milestone": (
        "Milestone accountability for the V487 solver-guidance overwrite "
        "telemetry gate."
    ),
    "status": (
        "Machine-readable terminal state for downstream solver-guidance telemetry "
        "consumers."
    ),
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether "
        "solver-guidance telemetry preserved authoritative fallback safety."
    ),
    "inference_substrate": (
        "Declares deterministic_solver_guidance_telemetry so the artifact is read "
        "as exact QSTR and CDCL guidance telemetry, not live model quality."
    ),
    "tests_run": (
        "Commands run to validate solver-guidance telemetry, artifact schema, "
        "new-code coverage, repository tests, and applicable offline e2e checks."
    ),
}

WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "solver_authoritative",
    "hint_class_count",
    "hint_validity_rate",
    "hint_overwrite_rate",
    "fallback_completeness_rate",
    "conflict_delta_vs_no_hint",
    "misleading_hint_false_accepts",
    "blocked_instance_class_count",
    "solver_guidance_telemetry_ready",
    "tests_run",
)


@dataclass(frozen=True)
class HintClass:
    """One deterministic advisory hint family used in both fixture domains."""

    name: str
    description: str


@dataclass(frozen=True)
class DiagnosticInstance:
    """One authoritative fixture row that can be probed with advisory hints."""

    domain: str
    instance_id: str
    instance_class: str
    baseline_status: str
    expected_satisfiable: bool
    baseline_solution: tuple[Any, ...]
    qstr_row: JsonDict | None = None
    sat_instance: cdcl.GuidanceInstance | None = None


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _load_json(path: Path) -> JsonDict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def load_source_fixtures() -> JsonDict:
    """Load the checked-in QSTR artifact and the bounded SAT/CDCL fixture."""

    qstr_artifact = _load_json(qstr.RESULT_RELATIVE_PATH)
    cdcl_artifact = _load_json(cdcl.RESULT_RELATIVE_PATH)
    qstr.validate_artifact(qstr_artifact)
    cdcl.validate_artifact(cdcl_artifact)
    qstr_fixture = qstr.build_fixture()
    qstr_evaluation = qstr.evaluate_fixture(qstr_fixture)
    sat_instance = cdcl.build_factor_guidance_instances()[0]
    return {
        "qstr_ready": bool(qstr_artifact["qstr_fixture_ready"]),
        "sat_cdcl_available": True,
        "qstr_fixture": qstr_fixture,
        "qstr_evaluation": qstr_evaluation,
        "sat_instance": sat_instance,
        "source_artifacts": [
            str(qstr.RESULT_RELATIVE_PATH),
            str(cdcl.RESULT_RELATIVE_PATH),
        ],
    }


def build_hint_classes() -> tuple[HintClass, ...]:
    """Return the five deterministic hint classes required by the spec."""

    return (
        HintClass("perfect_hints", "exact current solution hint"),
        HintClass("partial_hints", "valid but incomplete current solution hint"),
        HintClass("stale_hints", "formerly plausible but invalid current hint"),
        HintClass("misleading_hints", "actively wrong hint that must not decide"),
        HintClass("no_hints", "solver-only or checker-only control"),
    )


def build_diagnostic_instances(fixtures: JsonDict | None = None) -> tuple[DiagnosticInstance, ...]:
    """Build QSTR relation rows plus one existing SAT/CDCL bounded fixture."""

    source = load_source_fixtures() if fixtures is None else fixtures
    qstr_instances = tuple(
        DiagnosticInstance(
            domain="qstr",
            instance_id=row["case_id"],
            instance_class=f"qstr:{row['case_type']}",
            baseline_status=row["actual_label"],
            expected_satisfiable=bool(row["expected_satisfiable"]),
            baseline_solution=tuple(row["actual_relations"]),
            qstr_row=row,
        )
        for row in source["qstr_evaluation"]["relation_results"]
    )
    sat_instance = source["sat_instance"]
    sat_baseline = cdcl.run_cdcl(sat_instance.clauses, n_vars=sat_instance.n_vars)
    sat = DiagnosticInstance(
        domain="sat_cdcl",
        instance_id=sat_instance.instance_id,
        instance_class=sat_instance.instance_class,
        baseline_status=sat_baseline.status,
        expected_satisfiable=sat_instance.expected_status == "sat",
        baseline_solution=tuple(sat_baseline.model),
        sat_instance=sat_instance,
    )
    return qstr_instances + (sat,)


def run_diagnostic() -> JsonDict:
    """Run all hint classes and summarize overwrite/fallback telemetry."""

    fixtures = load_source_fixtures()
    instances = build_diagnostic_instances(fixtures)
    hints = build_hint_classes()
    rows = [
        _run_hint(instance, hint)
        for instance in instances
        for hint in hints
    ]
    summary = _summarize_rows(rows, hints, fixtures)
    summary["per_hint_results"] = rows
    return summary


def build_artifact(*, tests_run: list[JsonDict]) -> JsonDict:
    """Build the Exp5344 terminal artifact from deterministic telemetry."""

    diagnostic = run_diagnostic()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = bool(
        diagnostic["solver_guidance_telemetry_ready"]
        and tests_run
        and not blockers
    )
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_NAME),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap(
            "status",
            "solver_guidance_telemetry_ready"
            if ready
            else "blocked_solver_guidance_telemetry_not_ready",
        ),
        "honest_verdict": _wrap(
            "honest_verdict",
            (
                "complete: solver-guidance telemetry preserved authoritative "
                "fallback safety while recording stale and misleading hint overwrites"
            )
            if ready
            else "blocked_solver_guidance_telemetry_not_ready",
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "solver_authoritative": diagnostic["solver_authoritative"],
        "hint_class_count": diagnostic["hint_class_count"],
        "hint_validity_rate": diagnostic["hint_validity_rate"],
        "hint_overwrite_rate": diagnostic["hint_overwrite_rate"],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "conflict_delta_vs_no_hint": diagnostic["conflict_delta_vs_no_hint"],
        "misleading_hint_false_accepts": diagnostic["misleading_hint_false_accepts"],
        "blocked_instance_class_count": diagnostic["blocked_instance_class_count"],
        "solver_guidance_telemetry_ready": ready,
        "solved_rate": diagnostic["solved_rate"],
        "fallback_rate": diagnostic["fallback_rate"],
        "search_delta_vs_no_hint": diagnostic["search_delta_vs_no_hint"],
        "misleading_hint_harm": diagnostic["misleading_hint_harm"],
        "blocked_instance_classes": diagnostic["blocked_instance_classes"],
        "domain_summaries": diagnostic["domain_summaries"],
        "source_fixtures": diagnostic["source_fixtures"],
        "per_hint_results": diagnostic["per_hint_results"],
        "readiness_blockers": blockers,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "tests_run": _wrap("tests_run", tests_run),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: list[JsonDict] | None = None,
) -> JsonDict:
    """Run the offline diagnostic and write the result artifact."""

    artifact = build_artifact(tests_run=[] if tests_run is None else tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Validate schema fields that downstream telemetry gates depend on."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        _require(isinstance(artifact[field], dict), f"{field} must be principle wrapped")
        _require(artifact[field].get("principle") == FIELD_PRINCIPLES[field], field)
        _require("value" in artifact[field], f"{field} missing value")

    _require(
        artifact["honest_verdict"]["value"].startswith(TERMINAL_PREFIXES),
        "honest_verdict",
    )
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        "inference_substrate",
    )
    _require(artifact["solver_authoritative"] is True, "solver_authoritative")
    _require(_is_bare_int(artifact["hint_class_count"]), "hint_class_count")
    for field in (
        "hint_validity_rate",
        "hint_overwrite_rate",
        "fallback_completeness_rate",
        "conflict_delta_vs_no_hint",
    ):
        _require(_is_bare_numeric(artifact[field]), field)
    _require(_is_bare_int(artifact["misleading_hint_false_accepts"]), "misleading_hint_false_accepts")
    _require(_is_bare_int(artifact["blocked_instance_class_count"]), "blocked_instance_class_count")
    _require(
        _is_bare_bool(artifact["solver_guidance_telemetry_ready"]),
        "solver_guidance_telemetry_ready",
    )
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run")
    if artifact["solver_guidance_telemetry_ready"]:
        _require(artifact["status"]["value"] == "solver_guidance_telemetry_ready", "status")
        _require(artifact["hint_class_count"] == len(HINT_CLASS_NAMES), "hint_class_count")
        _require(artifact["fallback_completeness_rate"] == 1.0, "fallback_completeness_rate")
        _require(artifact["misleading_hint_false_accepts"] == 0, "misleading_hint_false_accepts")
        _require(bool(artifact["tests_run"]["value"]), "tests_run")


def _run_hint(instance: DiagnosticInstance, hint: HintClass) -> JsonDict:
    if instance.domain == "qstr":
        return _run_qstr_hint(instance, hint)
    return _run_sat_hint(instance, hint)


def _run_qstr_hint(instance: DiagnosticInstance, hint: HintClass) -> JsonDict:
    row = instance.qstr_row
    assert row is not None
    hinted_relations = _qstr_hint_payload(row, hint.name)
    hint_valid = not hinted_relations or bool(
        set(hinted_relations).intersection(row["actual_relations"])
    )
    fallback_used = bool(hinted_relations and not hint_valid)
    final_status = row["actual_label"]
    final_solution = list(row["actual_relations"])
    overwrite_count = len(hinted_relations) if fallback_used else 0
    false_accept = bool(
        hint.name == "misleading_hints"
        and not row["expected_satisfiable"]
        and row["accepted"]
    )
    return _base_row(
        instance,
        hint,
        hint_payload={"relations": hinted_relations},
        hint_valid=hint_valid,
        fallback_used=fallback_used,
        overwrite_count=overwrite_count,
        fallback_preserved_baseline=final_status == instance.baseline_status
        and tuple(final_solution) == instance.baseline_solution,
        final_status=final_status,
        final_solution=final_solution,
        final_model_valid=True,
        solved=bool(row["label_matches_expected"]),
        false_accept=false_accept,
        conflicts=0,
        search_steps=1 + int(fallback_used),
        conflict_delta_vs_no_hint=0,
        search_delta_vs_no_hint=0,
    )


def _run_sat_hint(instance: DiagnosticInstance, hint: HintClass) -> JsonDict:
    sat_instance = instance.sat_instance
    assert sat_instance is not None
    baseline = cdcl.run_cdcl(sat_instance.clauses, n_vars=sat_instance.n_vars)
    assumptions = _sat_hint_payload(baseline.model, hint.name)
    primary = (
        baseline
        if not assumptions
        else cdcl.run_cdcl(
            sat_instance.clauses,
            n_vars=sat_instance.n_vars,
            assumptions=assumptions,
        )
    )
    fallback_used = bool(assumptions and primary.status != baseline.status)
    fallback = (
        cdcl.run_cdcl(sat_instance.clauses, n_vars=sat_instance.n_vars)
        if fallback_used
        else None
    )
    final = fallback or primary
    final_metrics = (
        _add_count_metrics(primary.metrics, fallback.metrics)
        if fallback is not None
        else _count_metrics(primary.metrics)
    )
    baseline_metrics = _count_metrics(baseline.metrics)
    overwrite_count = (
        _overwrite_count(assumptions, final.model)
        if fallback_used
        else 0
    )
    final_model_valid = final.status == "unsat" or cdcl.verify_model(
        sat_instance.clauses,
        final.model,
    )
    fallback_preserved = (
        final.status == baseline.status
        and tuple(final.model) == tuple(baseline.model)
        and final_model_valid
    )
    search_steps = _search_steps(final_metrics)
    no_hint_search_steps = _search_steps(baseline_metrics)
    return _base_row(
        instance,
        hint,
        hint_payload={"assumptions": list(assumptions)},
        hint_valid=not assumptions or primary.status == baseline.status,
        fallback_used=fallback_used,
        overwrite_count=overwrite_count,
        fallback_preserved_baseline=fallback_preserved,
        final_status=final.status,
        final_solution=list(final.model),
        final_model_valid=final_model_valid,
        solved=final.status == sat_instance.expected_status and final_model_valid,
        false_accept=False,
        conflicts=final_metrics["conflicts"],
        search_steps=search_steps,
        conflict_delta_vs_no_hint=baseline_metrics["conflicts"] - final_metrics["conflicts"],
        search_delta_vs_no_hint=no_hint_search_steps - search_steps,
    )


def _base_row(
    instance: DiagnosticInstance,
    hint: HintClass,
    *,
    hint_payload: JsonDict,
    hint_valid: bool,
    fallback_used: bool,
    overwrite_count: int,
    fallback_preserved_baseline: bool,
    final_status: str,
    final_solution: list[Any],
    final_model_valid: bool,
    solved: bool,
    false_accept: bool,
    conflicts: int,
    search_steps: int,
    conflict_delta_vs_no_hint: int,
    search_delta_vs_no_hint: int,
) -> JsonDict:
    return {
        "domain": instance.domain,
        "instance_id": instance.instance_id,
        "instance_class": instance.instance_class,
        "hint_class": hint.name,
        "hint_payload": hint_payload,
        "hint_valid": hint_valid,
        "fallback_used": fallback_used,
        "overwrite_count": overwrite_count,
        "fallback_preserved_baseline": fallback_preserved_baseline,
        "baseline_status": instance.baseline_status,
        "final_status": final_status,
        "final_matches_baseline": final_status == instance.baseline_status,
        "final_solution": final_solution,
        "final_model_valid": final_model_valid,
        "solved": solved,
        "false_accept": false_accept,
        "conflicts": conflicts,
        "search_steps": search_steps,
        "conflict_delta_vs_no_hint": conflict_delta_vs_no_hint,
        "search_delta_vs_no_hint": search_delta_vs_no_hint,
    }


def _qstr_hint_payload(row: JsonDict, hint_name: str) -> list[str]:
    actual = row["actual_relation"]
    invalid = _qstr_invalid_relation(row["calculus"], set(row["actual_relations"]))
    if hint_name == "perfect_hints":
        return [actual]
    if hint_name == "partial_hints":
        return [actual, invalid]
    if hint_name == "misleading_hints":
        return [row["allowed_relations"][0] if not row["expected_satisfiable"] else invalid]
    if hint_name == "stale_hints":
        return [invalid]
    return []


def _qstr_invalid_relation(calculus: str, actual_relations: set[str]) -> str:
    order = qstr.TEMPORAL_RELATION_ORDER if calculus == qstr.TEMPORAL else qstr.SPATIAL_RELATION_ORDER
    return next(relation for relation in reversed(order) if relation not in actual_relations)


def _sat_hint_payload(model: tuple[int, ...], hint_name: str) -> tuple[int, ...]:
    positives = tuple(literal for literal in model if literal > 0)
    if hint_name == "perfect_hints":
        return positives[:2]
    if hint_name == "partial_hints":
        return positives[:1]
    if hint_name == "stale_hints":
        return SAT_STALE_HINT
    if hint_name == "misleading_hints":
        return SAT_MISLEADING_HINT
    return ()


def _summarize_rows(rows: list[JsonDict], hints: tuple[HintClass, ...], fixtures: JsonDict) -> JsonDict:
    total = len(rows)
    hinted_rows = [row for row in rows if row["hint_class"] != "no_hints"]
    fallback_rows = [row for row in rows if row["fallback_used"]]
    blocked_classes = sorted(
        {
            f"{row['domain']}:{row['hint_class']}"
            for row in fallback_rows
            if row["hint_class"] in {"stale_hints", "misleading_hints"}
        }
    )
    misleading_rows = [row for row in rows if row["hint_class"] == "misleading_hints"]
    sat_rows = [row for row in rows if row["domain"] == "sat_cdcl"]
    fallback_completeness_rate = _rate(
        sum(row["fallback_preserved_baseline"] for row in fallback_rows),
        len(fallback_rows),
    )
    misleading_false_accepts = sum(row["false_accept"] for row in misleading_rows)
    ready = bool(
        fixtures["qstr_ready"]
        and fixtures["sat_cdcl_available"]
        and len(hints) == len(HINT_CLASS_NAMES)
        and all(row["final_matches_baseline"] for row in rows)
        and all(row["final_model_valid"] for row in rows)
        and fallback_completeness_rate == 1.0
        and misleading_false_accepts == 0
    )
    return {
        "solver_authoritative": True,
        "hint_class_count": len(hints),
        "hint_validity_rate": _rate(sum(row["hint_valid"] for row in rows), total),
        "hint_overwrite_rate": _rate(
            sum(row["overwrite_count"] > 0 for row in hinted_rows),
            len(hinted_rows),
        ),
        "fallback_completeness_rate": fallback_completeness_rate,
        "conflict_delta_vs_no_hint": sum(row["conflict_delta_vs_no_hint"] for row in sat_rows),
        "search_delta_vs_no_hint": sum(row["search_delta_vs_no_hint"] for row in sat_rows),
        "misleading_hint_false_accepts": misleading_false_accepts,
        "blocked_instance_class_count": len(blocked_classes),
        "solver_guidance_telemetry_ready": ready,
        "solved_rate": _rate(sum(row["solved"] for row in rows), total),
        "fallback_rate": _rate(len(fallback_rows), total),
        "misleading_hint_harm": {
            "conflicts_added": sum(
                max(0, -row["conflict_delta_vs_no_hint"])
                for row in misleading_rows
            ),
            "search_steps_added": sum(
                max(0, -row["search_delta_vs_no_hint"])
                for row in misleading_rows
            ),
        },
        "blocked_instance_classes": blocked_classes,
        "domain_summaries": _domain_summaries(rows),
        "source_fixtures": {
            "qstr_ready": fixtures["qstr_ready"],
            "sat_cdcl_available": fixtures["sat_cdcl_available"],
            "source_artifacts": fixtures["source_artifacts"],
        },
    }


def _domain_summaries(rows: list[JsonDict]) -> JsonDict:
    counts = Counter(row["domain"] for row in rows)
    return {
        domain: {
            "hint_runs": counts[domain],
            "fallback_runs": sum(row["fallback_used"] for row in rows if row["domain"] == domain),
            "overwritten_runs": sum(row["overwrite_count"] > 0 for row in rows if row["domain"] == domain),
            "false_accepts": sum(row["false_accept"] for row in rows if row["domain"] == domain),
        }
        for domain in sorted(counts)
    }


def _readiness_blockers(diagnostic: JsonDict, tests_run: list[JsonDict]) -> list[str]:
    checks = (
        (not diagnostic["source_fixtures"]["qstr_ready"], "qstr_fixture_not_ready"),
        (not diagnostic["source_fixtures"]["sat_cdcl_available"], "sat_cdcl_fixture_missing"),
        (diagnostic["hint_class_count"] != len(HINT_CLASS_NAMES), "hint_class_count_mismatch"),
        (diagnostic["fallback_completeness_rate"] != 1.0, "fallback_completeness_incomplete"),
        (diagnostic["misleading_hint_false_accepts"] != 0, "misleading_hint_false_accepts"),
        (not diagnostic["solver_guidance_telemetry_ready"], "telemetry_not_ready"),
        (not tests_run, "tests_not_recorded"),
    )
    return [blocker for failed, blocker in checks if failed]


def _count_metrics(metrics: JsonDict) -> JsonDict:
    return {
        "conflicts": int(metrics["conflicts"]),
        "decisions": int(metrics["decisions"]),
        "propagations": int(metrics["propagations"]),
        "restarts": int(metrics["restarts"]),
    }


def _add_count_metrics(left: JsonDict, right: JsonDict) -> JsonDict:
    return {
        key: int(left[key]) + int(right[key])
        for key in ("conflicts", "decisions", "propagations", "restarts")
    }


def _search_steps(metrics: JsonDict) -> int:
    return int(metrics["conflicts"]) + int(metrics["decisions"]) + int(metrics["propagations"])


def _overwrite_count(assumptions: tuple[int, ...], final_model: tuple[int, ...]) -> int:
    final_literals = set(final_model)
    return sum(1 for literal in assumptions if literal not in final_literals)


def _rate(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _checksum_payload(artifact: JsonDict) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"]["value"],
        "spec_refs": artifact["spec_refs"],
        "source_fixtures": artifact["source_fixtures"],
        "metrics": {
            field: artifact[field]
            for field in (
                "hint_validity_rate",
                "hint_overwrite_rate",
                "fallback_completeness_rate",
                "conflict_delta_vs_no_hint",
                "misleading_hint_false_accepts",
                "blocked_instance_class_count",
                "solver_guidance_telemetry_ready",
            )
        },
        "per_hint_results": artifact["per_hint_results"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _is_bare_bool(value: Any) -> bool:
    return type(value) is bool


def _is_bare_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
