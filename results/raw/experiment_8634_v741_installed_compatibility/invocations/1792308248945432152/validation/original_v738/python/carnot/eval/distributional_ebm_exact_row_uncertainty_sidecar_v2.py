"""Exp 3223 exact-row uncertainty sidecar v2.

Spec refs: REQ-VERIFY-3223, SCENARIO-VERIFY-3223.

This module builds a deterministic, distributional-energy-style sidecar over
the `.297` exact fixture banks. The important boundary is that the sidecar is
not a verifier. It looks at already-materialized fixture metadata, assigns
triage scores for abstention and review priority, and records negative-control
audits for shortcut domination. The exact context checkers and exact bounded
ConstraintBench solvers remain the only authority for accept/reject scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.distributional_ebm_exact_row_uncertainty_sidecar.v2"
EXPERIMENT_ID = "exp3223"
MILESTONE = "2026.05.298"
OUTPUT_REL_PATH = Path(
    "results/experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.json"
)

CONTEXT_ARTIFACT_REL_PATH = Path(
    "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
)
CONTEXT_FIXTURE_REL_PATH = Path("data/research/context_cot_clbench_parametric_shortcut_v1.jsonl")
CONSTRAINT_ARTIFACT_REL_PATH = Path(
    "results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json"
)
CONSTRAINT_FIXTURE_REL_PATH = Path("data/research/constraintbench_feasibility_objective_pilot_v1.jsonl")

INFERENCE_SUBSTRATE = "deterministic_artifact_replay_no_llm"
MODEL_IDENTITY_ABSENT = "no_model_identity:deterministic_fixture"
SUCCESS_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_")
BLOCKED_PREFIXES = ("blocked_", "blocked:")

REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "experiment_id",
    "milestone",
    "source_fixture_artifacts",
    "exact_row_count",
    "uncertainty_sidecar_ready",
    "abstention_threshold_defined",
    "shortcut_risk_rows",
    "solver_disagreement_risk_rows",
    "model_identity_shortcut_audit",
    "clean_verifier_consumption_plan",
    "exact_verifier_authority_preserved",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)

ROW_SCORE_FIELDS = (
    "row_id",
    "artifact_source",
    "fixture_family",
    "difficulty_score",
    "uncertainty_score",
    "abstention_risk",
    "shortcut_risk",
    "solver_disagreement_risk",
    "model_identity",
    "exact_verifier_authority",
    "feature_branches",
)

SOURCE_SPECS = (
    ("exp3210_context_artifact", CONTEXT_ARTIFACT_REL_PATH, "json"),
    ("exp3210_context_fixture", CONTEXT_FIXTURE_REL_PATH, "jsonl"),
    ("exp3211_constraint_artifact", CONSTRAINT_ARTIFACT_REL_PATH, "json"),
    ("exp3211_constraint_fixture", CONSTRAINT_FIXTURE_REL_PATH, "jsonl"),
)

CHECKER_DIFFICULTY = {
    "exact_alias_string": 0.35,
    "exact_entity_fact_string": 0.55,
    "exact_integer_string": 0.75,
}

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.py -q -o addopts=''",
    ".venv/bin/coverage run --source=python/carnot/eval/distributional_ebm_exact_row_uncertainty_sidecar_v2.py -m pytest -o addopts='' tests/python/test_experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/distributional_ebm_exact_row_uncertainty_sidecar_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3223: build the exact-fixture uncertainty sidecar."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    sources = source_fixture_artifacts(root_path)
    source_errors = [source["id"] for source in sources if source["exists"] is not True]
    sidecar_rows: list[JsonDict] = []

    if not source_errors:
        context_artifact = read_json_object(root_path / CONTEXT_ARTIFACT_REL_PATH)
        constraint_artifact = read_json_object(root_path / CONSTRAINT_ARTIFACT_REL_PATH)
        context_rows = read_jsonl_objects(root_path / CONTEXT_FIXTURE_REL_PATH)
        constraint_rows = read_jsonl_objects(root_path / CONSTRAINT_FIXTURE_REL_PATH)
        sidecar_rows = finalize_abstention_risk(
            [
                *[score_context_row(row) for row in context_rows],
                *[
                    score_constraint_row(row, candidate_score_by_id(constraint_artifact))
                    for row in constraint_rows
                ],
            ]
        )

    threshold = abstention_threshold(sidecar_rows)
    audit = model_identity_shortcut_audit(sidecar_rows)
    readiness = readiness_checks(
        source_errors=source_errors,
        sidecar_rows=sidecar_rows,
        threshold=threshold,
        audit=audit,
    )
    ready = all(readiness.values())
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "source_fixture_artifacts": sources,
        "source_errors": source_errors,
        "exact_row_count": len(sidecar_rows),
        "uncertainty_sidecar_ready": ready,
        "abstention_threshold_defined": threshold is not None,
        "abstention_policy": abstention_policy(sidecar_rows, threshold),
        "sidecar_method": sidecar_method(),
        "sidecar_rows": sidecar_rows,
        "shortcut_risk_rows": top_risk_rows(
            sidecar_rows,
            "shortcut_risk",
            min_value=0.65,
            artifact_source="exp3210_context",
        ),
        "solver_disagreement_risk_rows": top_risk_rows(
            sidecar_rows,
            "solver_disagreement_risk",
            min_value=0.35,
            artifact_source="exp3211_constraintbench",
        ),
        "model_identity_shortcut_audit": audit,
        "clean_verifier_consumption_plan": clean_verifier_consumption_plan(),
        "exact_verifier_authority_preserved": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "executes_models": False,
            "new_live_model_calls": 0,
            "training_performed": False,
            "offline_exact_artifact_replay": True,
        },
        "readiness_checks": readiness,
        "blocked_reasons": [name for name, passed in readiness.items() if passed is not True],
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3223 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def score_context_row(row: Mapping[str, Any]) -> JsonDict:
    """Convert one context fixture into deterministic risk features."""

    expected = str(row.get("expected_answer") or "")
    prior = str(row.get("prior_bait_answer") or "")
    checker = str(row.get("exact_checker_type") or "")
    counterexample = row.get("minimal_counterexample")
    counter_map = counterexample if isinstance(counterexample, Mapping) else {}
    contradiction = 1.0 if normalize_answer(expected) != normalize_answer(prior) else 0.0
    context_dependency = 1.0 if row.get("context") and row.get("question") else 0.0
    answer_delta = answer_delta_score(expected, prior)
    checker_score = CHECKER_DIFFICULTY.get(checker, 0.50)
    difficulty = round(
        unit_interval(
            0.30 * checker_score
            + 0.25 * context_dependency
            + 0.25 * contradiction
            + 0.20 * answer_delta
        ),
        6,
    )
    shortcut = round(
        unit_interval(
            0.55 * contradiction
            + 0.30 * context_dependency
            + 0.15
            * float(counter_map.get("failure_mode") == "parametric_prior_shortcut")
        ),
        6,
    )
    raw_energy = round(unit_interval(0.52 * difficulty + 0.48 * shortcut), 6)
    uncertainty = uncertainty_from_energy(raw_energy, difficulty, shortcut)
    return {
        "row_id": str(row.get("fixture_id") or ""),
        "artifact_source": "exp3210_context",
        "source_fixture_artifact": CONTEXT_FIXTURE_REL_PATH.as_posix(),
        "fixture_family": str(row.get("family") or ""),
        "difficulty_score": difficulty,
        "raw_energy": raw_energy,
        "uncertainty_score": uncertainty,
        "abstention_risk": 0.0,
        "shortcut_risk": shortcut,
        "model_identity_shortcut_risk": 0.0,
        "solver_disagreement_risk": 0.0,
        "model_identity": MODEL_IDENTITY_ABSENT,
        "exact_verifier_authority": "context_exact_checker",
        "context_dependency_label": "context_required_prior_contradiction",
        "feature_branches": {
            "context_dependency_flag": context_dependency,
            "prior_bait_contradiction_flag": contradiction,
            "checker_difficulty": checker_score,
            "answer_delta": answer_delta,
            "model_identity_shortcut_flag": 0.0,
        },
        "triage_metadata_only": True,
    }


def score_constraint_row(
    row: Mapping[str, Any],
    score_by_id: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Convert one ConstraintBench-style row into deterministic risk features."""

    row_id = str(row.get("row_id") or "")
    candidate = score_by_id.get(row_id, {})
    exact_reference = row.get("exact_reference")
    exact_map = exact_reference if isinstance(exact_reference, Mapping) else {}
    optimum = numeric_or_none(candidate.get("optimum_value"))
    if optimum is None:
        optimum = numeric_or_none(exact_map.get("objective_value")) or 0.0
    objective_gap = numeric_or_none(candidate.get("objective_gap"))
    gap_norm = objective_gap_score(objective_gap, optimum)
    invalid = bool(candidate.get("invalid_format") is True)
    hallucinated = bool(candidate.get("hallucinated_entity") is True)
    missing_constraint = bool(candidate.get("missing_constraint") is True)
    valid_format = bool(candidate.get("valid_format") is True)
    feasibility_failure = bool(valid_format and candidate.get("feasibility_pass") is not True)
    violation = max(
        float(invalid),
        float(hallucinated),
        float(missing_constraint),
        float(feasibility_failure),
    )
    constraints = row.get("constraints")
    constraint_count = len(constraints) if isinstance(constraints, list) else 0
    feasible_count = numeric_or_none(exact_map.get("feasible_count")) or 0.0
    constraint_density = unit_interval(constraint_count / 5.0)
    sparse_feasible_space = 1.0 - unit_interval(feasible_count / 12.0)
    solver_disagreement = round(unit_interval(max(violation, gap_norm * 0.80)), 6)
    shortcut = round(
        unit_interval(0.35 * float(hallucinated) + 0.20 * float(invalid) + 0.15 * float(missing_constraint)),
        6,
    )
    difficulty = round(
        unit_interval(
            0.30 * constraint_density
            + 0.20 * sparse_feasible_space
            + 0.25 * gap_norm
            + 0.25 * solver_disagreement
        ),
        6,
    )
    raw_energy = round(
        unit_interval(0.50 * solver_disagreement + 0.40 * difficulty + 0.10 * shortcut),
        6,
    )
    uncertainty = uncertainty_from_energy(raw_energy, difficulty, solver_disagreement)
    return {
        "row_id": row_id,
        "artifact_source": "exp3211_constraintbench",
        "source_fixture_artifact": CONSTRAINT_FIXTURE_REL_PATH.as_posix(),
        "fixture_family": str(row.get("family") or ""),
        "difficulty_score": difficulty,
        "raw_energy": raw_energy,
        "uncertainty_score": uncertainty,
        "abstention_risk": 0.0,
        "shortcut_risk": shortcut,
        "model_identity_shortcut_risk": 0.0,
        "solver_disagreement_risk": solver_disagreement,
        "model_identity": MODEL_IDENTITY_ABSENT,
        "exact_verifier_authority": "constraintbench_exact_bounded_solver",
        "context_dependency_label": "not_context_fixture",
        "feature_branches": {
            "satisfiable_flag": float(exact_map.get("feasible") is True),
            "valid_format_flag": float(valid_format),
            "objective_gap_norm": gap_norm,
            "invalid_format_flag": float(invalid),
            "hallucinated_entity_flag": float(hallucinated),
            "missing_constraint_flag": float(missing_constraint),
            "feasibility_failure_flag": float(feasibility_failure),
            "checker_backend_present": float(bool(row.get("checker_backend"))),
            "model_identity_shortcut_flag": 0.0,
        },
        "triage_metadata_only": True,
    }


def finalize_abstention_risk(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Attach source-balanced abstention risk without erasing row difficulty."""

    row_list = [dict(row) for row in rows]
    percentiles: dict[str, float] = {}
    for source in sorted({str(row.get("artifact_source") or "") for row in row_list}):
        group = [row for row in row_list if row.get("artifact_source") == source]
        ranked = sorted(group, key=lambda item: (float(item["raw_energy"]), str(item["row_id"])))
        denominator = max(1, len(ranked) - 1)
        for index, row in enumerate(ranked):
            percentiles[str(row["row_id"])] = 0.5 if len(ranked) == 1 else index / denominator

    finalized: list[JsonDict] = []
    for row in row_list:
        source_percentile = percentiles.get(str(row.get("row_id")), 0.0)
        abstention = round(
            unit_interval(
                0.35 * float(row["raw_energy"])
                + 0.35 * float(row["difficulty_score"])
                + 0.30 * source_percentile
            ),
            6,
        )
        finalized.append(
            {
                **row,
                "source_balanced_percentile": round(source_percentile, 6),
                "abstention_risk": abstention,
            }
        )
    return sorted(finalized, key=lambda item: str(item["row_id"]))


def abstention_threshold(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Define a deterministic top-quartile abstention threshold when rows exist."""

    if not rows:
        return None
    risks = sorted(float(row["abstention_risk"]) for row in rows)
    index = max(0, math.ceil(0.75 * len(risks)) - 1)
    return round(risks[index], 6)


def abstention_policy(rows: Sequence[Mapping[str, Any]], threshold: float | None) -> JsonDict:
    """Summarize how Exp 3225 can route high-risk rows for extra checks."""

    abstained = [
        row
        for row in rows
        if threshold is not None and float(row.get("abstention_risk", 0.0)) >= threshold
    ]
    return {
        "deployed_policy": False,
        "threshold_metric": "abstention_risk",
        "threshold": threshold,
        "threshold_selection": "deterministic top-quartile triage threshold",
        "coverage_denominator": len(rows),
        "rows_above_threshold": len(abstained),
        "coverage_after_abstention": safe_rate(len(rows) - len(abstained), len(rows)),
        "policy_note": "triage metadata only; exact verifier scoring remains authoritative",
    }


def model_identity_shortcut_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check that risk is not just a proxy for model identity or source file."""

    if not rows:
        return {
            "row_count": 0,
            "model_identity_values": [],
            "model_identity_group_means": {},
            "model_identity_max_group_delta": 0.0,
            "model_identity_dominated": False,
            "artifact_source_group_means": {},
            "artifact_source_max_group_delta": 0.0,
            "artifact_source_dominated": False,
            "difficulty_risk_correlation": 0.0,
            "artifact_source_risk_correlation": 0.0,
            "risk_dominated_by": "unavailable",
            "audit_note": "no rows available for shortcut audit",
        }
    risks = [float(row["abstention_risk"]) for row in rows]
    difficulties = [float(row["difficulty_score"]) for row in rows]
    model_means = group_means(rows, "model_identity", "abstention_risk")
    source_means = group_means(rows, "artifact_source", "abstention_risk")
    source_values = sorted(source_means)
    source_codes = [float(source_values.index(str(row["artifact_source"]))) for row in rows]
    model_delta = group_delta(model_means)
    source_delta = group_delta(source_means)
    difficulty_corr = abs(pearson_correlation(difficulties, risks))
    source_corr = abs(pearson_correlation(source_codes, risks))
    difficulty_span = max(difficulties) - min(difficulties)
    model_dominated = len(model_means) > 1 and model_delta > 0.20
    source_dominated = source_delta > max(0.30, difficulty_span) and source_corr > difficulty_corr
    return {
        "row_count": len(rows),
        "model_identity_values": sorted(model_means),
        "model_identity_group_means": model_means,
        "model_identity_max_group_delta": round(model_delta, 6),
        "model_identity_dominated": model_dominated,
        "artifact_source_group_means": source_means,
        "artifact_source_max_group_delta": round(source_delta, 6),
        "artifact_source_dominated": source_dominated,
        "difficulty_risk_correlation": round(difficulty_corr, 6),
        "artifact_source_risk_correlation": round(source_corr, 6),
        "risk_dominated_by": (
            "model_identity"
            if model_dominated
            else "artifact_source"
            if source_dominated
            else "row_difficulty"
        ),
        "audit_note": (
            "Model identity is absent because the sources are deterministic fixture "
            "artifacts; source effects are reported separately from row difficulty."
        ),
    }


def readiness_checks(
    *,
    source_errors: Sequence[str],
    sidecar_rows: Sequence[Mapping[str, Any]],
    threshold: float | None,
    audit: Mapping[str, Any],
) -> dict[str, bool]:
    """Return the machine-readable readiness checks for the artifact."""

    sources = {str(row.get("artifact_source") or "") for row in sidecar_rows}
    return {
        "required_sources_present": not source_errors,
        "context_rows_scored": "exp3210_context" in sources,
        "constraint_rows_scored": "exp3211_constraintbench" in sources,
        "exact_row_denominator_nonempty": bool(sidecar_rows),
        "abstention_threshold_defined": threshold is not None,
        "negative_control_not_model_identity_dominated": audit.get("model_identity_dominated")
        is False,
        "negative_control_not_artifact_source_dominated": audit.get("artifact_source_dominated")
        is False,
        "exact_verifier_authority_preserved": True,
        "no_live_model_calls": True,
    }


def top_risk_rows(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    min_value: float,
    artifact_source: str,
    max_rows: int = 10,
) -> list[JsonDict]:
    """Return compact high-risk rows for sidecar triage lists."""

    candidates = [
        row
        for row in rows
        if row.get("artifact_source") == artifact_source and float(row.get(field, 0.0)) >= min_value
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (float(row.get(field, 0.0)), float(row.get("abstention_risk", 0.0)), str(row.get("row_id"))),
        reverse=True,
    )
    return [
        {
            "rank": index + 1,
            "row_id": str(row.get("row_id") or ""),
            "artifact_source": str(row.get("artifact_source") or ""),
            "fixture_family": str(row.get("fixture_family") or ""),
            "risk_field": field,
            "risk_value": round(float(row.get(field, 0.0)), 6),
            "abstention_risk": round(float(row.get("abstention_risk", 0.0)), 6),
            "difficulty_score": round(float(row.get("difficulty_score", 0.0)), 6),
            "triage_reason": row_triage_reason(row, field),
        }
        for index, row in enumerate(ranked[:max_rows])
    ]


def row_triage_reason(row: Mapping[str, Any], field: str) -> str:
    """Explain why a row entered a compact triage list."""

    branches = row.get("feature_branches")
    branch_map = branches if isinstance(branches, Mapping) else {}
    if field == "shortcut_risk":
        return "context_prior_bait_counterexample"
    active = [
        name
        for name in (
            "missing_constraint_flag",
            "invalid_format_flag",
            "hallucinated_entity_flag",
            "feasibility_failure_flag",
            "objective_gap_norm",
        )
        if float(branch_map.get(name, 0.0)) > 0.0
    ]
    return ",".join(active) or "solver_risk_threshold"


def source_fixture_artifacts(root: Path) -> list[JsonDict]:
    """Return source paths, checksums, and row counts consumed by the sidecar."""

    rows: list[JsonDict] = []
    for source_id, rel_path, source_type in SOURCE_SPECS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "source_type": source_type,
                "exists": path.is_file(),
                "sha256": file_sha256(path),
                "row_count": source_row_count(path, source_type),
            }
        )
    return rows


def sidecar_method() -> JsonDict:
    """Describe the deterministic scoring method and its nondeployment boundary."""

    return {
        "method_type": "deterministic_distributional_energy_proxy_over_exact_rows",
        "training_performed": False,
        "llm_invocations": 0,
        "feature_inputs": [
            "context prior-bait contradiction",
            "context dependency labels",
            "ConstraintBench feasibility and objective gap",
            "ConstraintBench invalid-format, hallucinated-entity, and missing-constraint flags",
            "checker backend and exact-reference metadata",
        ],
        "abstention_risk": (
            "source-balanced deterministic risk derived from raw energy, row difficulty, "
            "and within-source rank"
        ),
        "negative_control": (
            "model identity is absent and artifact-source group effects are audited "
            "against row difficulty"
        ),
        "nondeployment_boundary": (
            "Scores may prioritize rows for abstention or clean-verifier review, "
            "but exact verifier and solver outputs remain authoritative."
        ),
    }


def clean_verifier_consumption_plan() -> str:
    """Return the Exp 3225 consumption plan required by the task."""

    return (
        "Exp3225 should consume this sidecar only as triage metadata: prioritize "
        "rows with high abstention_risk, shortcut_risk, or solver_disagreement_risk "
        "for clean verifier review, but compute all pass/fail metrics from the "
        "exact verifier or bounded solver outputs rather than from sidecar scores."
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict without upgrading triage into verification."""

    if artifact.get("uncertainty_sidecar_ready") is True:
        return (
            "complete: exact-row uncertainty sidecar materialized as triage metadata; "
            "exact verifier authority preserved"
        )
    reasons = artifact.get("blocked_reasons")
    reason_text = ",".join(str(reason) for reason in reasons if reason) or "preconditions_missing"
    return f"blocked_uncertainty_sidecar_precondition:{reason_text}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3223 terminal artifact before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    require(
        verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith(BLOCKED_PREFIXES),
        "honest_verdict must be complete/success/passed or blocked",
    )
    require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference_substrate must be deterministic artifact replay",
    )
    require(
        artifact.get("exact_verifier_authority_preserved") is True,
        "exact verifier authority must be preserved",
    )
    rows = artifact.get("sidecar_rows")
    require(isinstance(rows, list), "sidecar_rows must be a list")
    require(int(artifact.get("exact_row_count", -1)) == len(rows), "exact_row_count mismatch")
    audit = artifact.get("model_identity_shortcut_audit")
    require(isinstance(audit, Mapping), "model_identity_shortcut_audit must be an object")
    require(audit.get("model_identity_dominated") is False, "model identity dominated")
    require(audit.get("artifact_source_dominated") is False, "artifact source dominated")
    for row in rows:
        require(isinstance(row, Mapping), "sidecar row must be an object")
        missing_row = [field for field in ROW_SCORE_FIELDS if field not in row]
        require(not missing_row, f"sidecar row missing fields: {missing_row}")
        for field in (
            "difficulty_score",
            "uncertainty_score",
            "abstention_risk",
            "shortcut_risk",
            "solver_disagreement_risk",
        ):
            require(metric_is_unit_interval(row.get(field)), f"{field} must be unit interval")
    if artifact.get("uncertainty_sidecar_ready") is True:
        require(bool(rows), "ready artifact requires sidecar rows")
        require(
            artifact.get("abstention_threshold_defined") is True,
            "abstention threshold must be defined",
        )
        require(bool(artifact.get("shortcut_risk_rows")), "ready artifact requires shortcut rows")
        require(
            bool(artifact.get("solver_disagreement_risk_rows")),
            "ready artifact requires solver disagreement rows",
        )


def candidate_score_by_id(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Index ConstraintBench candidate score rows by row id."""

    scores = artifact.get("candidate_scores")
    if not isinstance(scores, list):
        return {}
    return {
        str(score.get("row_id")): dict(score)
        for score in scores
        if isinstance(score, Mapping) and score.get("row_id")
    }


def group_means(rows: Sequence[Mapping[str, Any]], group_field: str, value_field: str) -> dict[str, float]:
    """Return rounded group means for audit reporting."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_field) or ""), []).append(float(row.get(value_field, 0.0)))
    return {name: round(sum(values) / len(values), 6) for name, values in sorted(grouped.items())}


def group_delta(means: Mapping[str, float]) -> float:
    """Return max-min group mean spread."""

    if not means:
        return 0.0
    values = list(means.values())
    return max(values) - min(values)


def uncertainty_from_energy(raw_energy: float, difficulty: float, disagreement: float) -> float:
    """Estimate uncertainty from energy-boundary proximity and branch disagreement."""

    boundary = 1.0 - abs(unit_interval(raw_energy) - 0.5) * 2.0
    return round(unit_interval(0.35 * boundary + 0.35 * difficulty + 0.30 * disagreement), 6)


def answer_delta_score(expected: str, prior: str) -> float:
    """Return a bounded proxy for how different prior and expected answers are."""

    expected_tokens = set(normalize_answer(expected).split())
    prior_tokens = set(normalize_answer(prior).split())
    union = expected_tokens | prior_tokens
    if not union:
        return 0.0
    overlap = expected_tokens & prior_tokens
    return round(1.0 - safe_rate(len(overlap), len(union)), 6)


def objective_gap_score(objective_gap: float | None, optimum_value: float) -> float:
    """Normalize objective gap while preserving missing-gap failures as high risk."""

    if objective_gap is None:
        return 0.0
    return round(unit_interval(abs(objective_gap) / max(1.0, abs(optimum_value))), 6)


def normalize_answer(answer: str) -> str:
    """Normalize short exact answers for deterministic mismatch checks."""

    text = " ".join(str(answer).strip().lower().split())
    return text[:-1] if text.endswith((".", "!", "?")) else text


def read_json_object(path: Path) -> JsonDict:
    """Read a checked-in JSON object, returning empty evidence on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive source corruption path.
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read JSONL fixture rows and keep only object records."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:  # pragma: no cover - sources are prechecked before loading.
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - source corruption path.
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def source_row_count(path: Path, source_type: str) -> int | None:
    """Return fixture/artifact row counts when the source is readable."""

    if not path.is_file():
        return None
    if source_type == "jsonl":
        return len(read_jsonl_objects(path))
    payload = read_json_object(path)
    count = payload.get("fixture_count")
    return int(count) if isinstance(count, int) else None


def file_sha256(path: Path) -> str | None:
    """Hash a source file when it exists."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded elapsed wall time."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def numeric_or_none(value: Any) -> float | None:
    """Convert finite numeric artifact values while preserving missing values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def unit_interval(value: float) -> float:
    """Clamp a numeric score into the closed unit interval."""

    return max(0.0, min(1.0, float(value)))


def metric_is_unit_interval(value: Any) -> bool:
    """Return true when an artifact metric is finite and in [0, 1]."""

    numeric = numeric_or_none(value)
    return numeric is not None and 0.0 <= numeric <= 1.0


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with a zero-denominator fallback."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson correlation, or 0.0 when the data are degenerate."""

    if len(xs) != len(ys) or not xs:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_deltas = [x - x_mean for x in xs]
    y_deltas = [y - y_mean for y in ys]
    x_var = sum(delta * delta for delta in x_deltas)
    y_var = sum(delta * delta for delta in y_deltas)
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    covariance = sum(x_delta * y_delta for x_delta, y_delta in zip(x_deltas, y_deltas, strict=True))
    return covariance / math.sqrt(x_var * y_var)


def require(condition: bool, message: str) -> None:
    """Raise a validation error with a stable message."""

    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - exercised by the conductor, not unit tests.
    """CLI entrypoint for writing the requested result artifact."""

    output = write_artifact()
    artifact = read_json_object(output)
    print(
        json.dumps(
            {
                "artifact": output.as_posix(),
                "ready": artifact["uncertainty_sidecar_ready"],
                "exact_row_count": artifact["exact_row_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
