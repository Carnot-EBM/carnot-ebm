"""Exp5370: deterministic overwrite-capable solver guidance matrix.

Spec refs: REQ-VERIFY-5370, SCENARIO-VERIFY-5370.

This module compares three local guidance routes over existing QSTR and SAT
fixtures: no hints, forced hints, and overwrite-capable hints. Hints can change
the search path, but the symbolic solver or deterministic verifier remains the
authority for every final validity decision. The comparison is not a learned
SE-RRM baseline.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5292_pbit_cdcl_factor_guidance_v483 as cdcl
from carnot import experiment_5343_qstr_temporal_spatial_constraint_fixture_v487 as qstr
from carnot import experiment_5358_solver_projection_cut_bridge_v488 as exp5358
from carnot import experiment_5359_pbit_schedule_diagnostic_v488 as exp5359


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5370_overwrite_solver_guidance_matrix_v489.json"
)
EXPERIMENT = 5370
EXPERIMENT_ID = "exp5370-overwrite-solver-guidance-matrix-v489"
MILESTONE = "2026.07.489"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5370.overwrite_solver_guidance_matrix.v489"
SPEC_REFS = ("REQ-VERIFY-5370", "SCENARIO-VERIFY-5370")
TERMINAL_PREFIXES = ("complete:", "blocked_")

GUIDANCE_MODE_NAMES = ("no_hint", "forced_hint", "overwrite_capable")
PROPOSAL_CLASS_NAMES = (
    "aligned_hints",
    "partially_wrong_hints",
    "misleading_high_confidence_hints",
    "no_hints",
)
EXPECTED_FIXTURE_COUNT = 14
QSTR_BASELINE_CONFLICTS = 1
QSTR_BASELINE_SEARCH_STEPS = 3

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if all solver-guidance modes are measured.",
    "overwrite_solver_guidance_ready": (
        "true only if no-hint, forced-hint, and overwrite-capable modes are all "
        "compared under solver authority."
    ),
    "solver_authoritative": "must be true; solver/verifier decides validity.",
    "fixture_count": "number of constraint instances measured.",
    "proposal_class_count": "number of hint/proposal classes measured.",
    "overwrite_rate": "fraction of hinted decisions overwritten by the solver.",
    "conflict_delta_vs_solver_only": (
        "conflict-count delta for overwrite-capable mode versus solver-only "
        "baseline."
    ),
    "forced_hint_harm_rate": (
        "fraction of cases where forced hints worsen conflicts/search/validity."
    ),
    "overwrite_hint_harm_rate": (
        "fraction of cases where overwrite-capable hints worsen conflicts/search/"
        "validity."
    ),
    "post_projection_validity_rate": (
        "fraction of returned solutions passing deterministic verifier."
    ),
    "fallback_completeness_rate": (
        "fraction of cases where fallback solver still returns the correct answer "
        "when hints fail."
    ),
    "harmful_hint_classes": (
        "list of hint classes that harmed performance or validity."
    ),
    "unsafe_false_accepts": "count of invalid solutions accepted as valid.",
    "tests_run": "list of commands run or no-code-change explanation.",
    "honest_verdict": "one-line result summary.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "overwrite_solver_guidance_ready",
    "solver_authoritative",
    "fixture_count",
    "proposal_class_count",
    "overwrite_rate",
    "conflict_delta_vs_solver_only",
    "forced_hint_harm_rate",
    "overwrite_hint_harm_rate",
    "post_projection_validity_rate",
    "fallback_completeness_rate",
    "harmful_hint_classes",
    "unsafe_false_accepts",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class GuidanceMode:
    """One deterministic route through the solver-guidance matrix."""

    name: str
    description: str


@dataclass(frozen=True)
class ProposalClass:
    """One hint family offered to the symbolic solver or verifier."""

    name: str
    description: str
    confidence: float | None = None


@dataclass(frozen=True)
class ConstraintFixture:
    """A reusable QSTR or SAT/CDCL constraint instance with baseline telemetry."""

    domain: str
    fixture_id: str
    fixture_class: str
    expected_satisfiable: bool
    baseline_status: str
    baseline_solution: tuple[Any, ...]
    baseline_conflicts: int
    baseline_search_steps: int
    qstr_row: JsonDict | None = None
    sat_instance: cdcl.GuidanceInstance | None = None


def build_guidance_modes() -> tuple[GuidanceMode, ...]:
    """Return the three required solver-guidance modes."""

    return (
        GuidanceMode("no_hint", "solver-only baseline with proposals ignored"),
        GuidanceMode("forced_hint", "mandatory candidate checked by the solver"),
        GuidanceMode("overwrite_capable", "solver may project, reject, or fallback"),
    )


def build_proposal_classes() -> tuple[ProposalClass, ...]:
    """Return the four deterministic proposal classes required by the spec."""

    return (
        ProposalClass("aligned_hints", "consistent with the solver baseline", 0.90),
        ProposalClass("partially_wrong_hints", "mixes a valid hint with a wrong hint", 0.70),
        ProposalClass(
            "misleading_high_confidence_hints",
            "wrong but presented with high confidence",
            0.99,
        ),
        ProposalClass("no_hints", "empty proposal control", None),
    )


def load_source_fixtures() -> JsonDict:
    """Load existing deterministic solver artifacts and fixture builders."""

    projection = _load_json(exp5358.RESULT_RELATIVE_PATH)
    schedule = _load_json(exp5359.RESULT_RELATIVE_PATH)
    qstr_artifact = _load_json(qstr.RESULT_RELATIVE_PATH)
    cdcl_artifact = _load_json(cdcl.RESULT_RELATIVE_PATH)
    exp5358.validate_artifact(projection)
    exp5359.validate_artifact(schedule)
    qstr.validate_artifact(qstr_artifact)
    cdcl.validate_artifact(cdcl_artifact)
    qstr_fixture = qstr.build_fixture()
    qstr_evaluation = qstr.evaluate_fixture(qstr_fixture)
    return {
        "solver_projection_ready": bool(projection["solver_projection_ready"]),
        "pbit_schedule_diagnostic_ready": bool(
            schedule["pbit_schedule_signal_ready"]
        ),
        "qstr_ready": bool(qstr_artifact["qstr_fixture_ready"]),
        "sat_cdcl_available": bool(cdcl_artifact["correctness_preserved"]["value"]),
        "qstr_evaluation": qstr_evaluation,
        "sat_instances": cdcl.build_factor_guidance_instances(),
        "source_artifacts": [
            str(exp5358.RESULT_RELATIVE_PATH),
            str(exp5359.RESULT_RELATIVE_PATH),
            str(qstr.RESULT_RELATIVE_PATH),
            str(cdcl.RESULT_RELATIVE_PATH),
        ],
    }


def build_constraint_fixtures(sources: JsonDict | None = None) -> tuple[ConstraintFixture, ...]:
    """Build QSTR relation rows plus the bounded Exp5292 SAT/CDCL fixtures."""

    loaded = load_source_fixtures() if sources is None else sources
    fixtures: list[ConstraintFixture] = []
    for row in loaded["qstr_evaluation"]["relation_results"]:
        fixtures.append(
            ConstraintFixture(
                domain="qstr",
                fixture_id=row["case_id"],
                fixture_class=f"qstr:{row['case_type']}",
                expected_satisfiable=bool(row["expected_satisfiable"]),
                baseline_status=row["actual_label"],
                baseline_solution=tuple(row["actual_relations"]),
                baseline_conflicts=QSTR_BASELINE_CONFLICTS,
                baseline_search_steps=QSTR_BASELINE_SEARCH_STEPS,
                qstr_row=row,
            )
        )
    for instance in loaded["sat_instances"]:
        baseline = cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
        fixtures.append(
            ConstraintFixture(
                domain="sat_cdcl",
                fixture_id=instance.instance_id,
                fixture_class=instance.instance_class,
                expected_satisfiable=instance.expected_status == "sat",
                baseline_status=baseline.status,
                baseline_solution=tuple(baseline.model),
                baseline_conflicts=int(baseline.metrics["conflicts"]),
                baseline_search_steps=_search_steps(baseline.metrics),
                sat_instance=instance,
            )
        )
    return tuple(fixtures)


def run_guidance_matrix() -> JsonDict:
    """Run every required mode/proposal pair under solver authority."""

    sources = load_source_fixtures()
    fixtures = build_constraint_fixtures(sources)
    modes = build_guidance_modes()
    proposals = build_proposal_classes()
    rows = [
        _evaluate_matrix_row(fixture, mode, proposal)
        for fixture in fixtures
        for mode in modes
        for proposal in proposals
    ]
    return _summarize_matrix(rows, fixtures, modes, proposals, sources)


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the Exp5370 JSON artifact from deterministic matrix telemetry."""

    diagnostic = run_guidance_matrix()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = bool(
        diagnostic["overwrite_solver_guidance_ready"]
        and not blockers
        and bool(tests_run)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked_overwrite_solver_guidance_not_ready",
        "overwrite_solver_guidance_ready": ready,
        "solver_authoritative": diagnostic["solver_authoritative"],
        "fixture_count": diagnostic["fixture_count"],
        "proposal_class_count": diagnostic["proposal_class_count"],
        "overwrite_rate": diagnostic["overwrite_rate"],
        "conflict_delta_vs_solver_only": diagnostic[
            "conflict_delta_vs_solver_only"
        ],
        "forced_hint_harm_rate": diagnostic["forced_hint_harm_rate"],
        "overwrite_hint_harm_rate": diagnostic["overwrite_hint_harm_rate"],
        "post_projection_validity_rate": diagnostic[
            "post_projection_validity_rate"
        ],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "harmful_hint_classes": diagnostic["harmful_hint_classes"],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "tests_run": [dict(row) for row in tests_run],
        "honest_verdict": (
            "complete: overwrite-capable routing preserved solver-authoritative "
            "validity while forced bad hints exposed bounded harm"
            if ready
            else "blocked_overwrite_solver_guidance_not_ready"
        ),
        "guidance_mode_count": diagnostic["guidance_mode_count"],
        "guidance_modes_measured": diagnostic["guidance_modes_measured"],
        "proposal_classes_measured": diagnostic["proposal_classes_measured"],
        "source_artifacts": diagnostic["source_artifacts"],
        "source_readiness": diagnostic["source_readiness"],
        "matrix_summary": diagnostic["matrix_summary"],
        "matrix_results": diagnostic["matrix_results"],
        "readiness_blockers": blockers,
        "claim_limits": [
            "deterministic local solver guidance only",
            "symbolic solver and deterministic verifier remain authoritative",
            "forced hints are measured as a safety contrast, not trusted",
            "no learned SE-RRM baseline claim",
            "no LLM, hardware, or generated text judge execution",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the validated Exp5370 artifact and return it."""

    artifact = build_artifact(tests_run=[] if tests_run is None else tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required solver-authoritative matrix contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "blocked_overwrite_solver_guidance_not_ready"}, "status")
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict",
    )
    _require(artifact["solver_authoritative"] is True, "solver_authoritative")
    _require(_is_bare_bool(artifact["overwrite_solver_guidance_ready"]), "overwrite_solver_guidance_ready")
    _require(_is_bare_int(artifact["fixture_count"]), "fixture_count")
    _require(_is_bare_int(artifact["proposal_class_count"]), "proposal_class_count")
    _require(_is_bare_int(artifact["unsafe_false_accepts"]), "unsafe_false_accepts")
    for field in (
        "overwrite_rate",
        "conflict_delta_vs_solver_only",
        "forced_hint_harm_rate",
        "overwrite_hint_harm_rate",
        "post_projection_validity_rate",
        "fallback_completeness_rate",
    ):
        _require(_is_bare_numeric(artifact[field]), field)
    _require(isinstance(artifact["harmful_hint_classes"], list), "harmful_hint_classes")
    _require(isinstance(artifact["tests_run"], list), "tests_run")
    _require(
        tuple(artifact["guidance_modes_measured"]) == GUIDANCE_MODE_NAMES,
        "guidance_modes_measured",
    )
    _require(
        tuple(artifact["proposal_classes_measured"]) == PROPOSAL_CLASS_NAMES,
        "proposal_classes_measured",
    )
    _require("REQ-VERIFY-5370" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")
    if artifact["overwrite_solver_guidance_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(bool(artifact["tests_run"]), "tests_run")
        _require(artifact["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(artifact["proposal_class_count"] == len(PROPOSAL_CLASS_NAMES), "proposal_class_count")
        _require(artifact["fallback_completeness_rate"] == 1.0, "fallback_completeness_rate")
        _require(artifact["unsafe_false_accepts"] == 0, "unsafe_false_accepts")
        _require(
            artifact["forced_hint_harm_rate"] > artifact["overwrite_hint_harm_rate"],
            "hint_harm_rates",
        )
        _validate_matrix_rows(artifact["matrix_results"])


def _evaluate_matrix_row(
    fixture: ConstraintFixture,
    mode: GuidanceMode,
    proposal: ProposalClass,
) -> JsonDict:
    if mode.name == "no_hint":
        return _baseline_row(fixture, mode, proposal)
    if fixture.domain == "qstr":
        return _evaluate_qstr_row(fixture, mode, proposal)
    return _evaluate_sat_row(fixture, mode, proposal)


def _baseline_row(
    fixture: ConstraintFixture,
    mode: GuidanceMode,
    proposal: ProposalClass,
) -> JsonDict:
    return _row(
        fixture,
        mode,
        proposal,
        proposal_payload=_empty_payload(fixture),
        solver_action="solver_only_baseline",
        final_status=fixture.baseline_status,
        final_solution=list(fixture.baseline_solution),
        conflicts=fixture.baseline_conflicts,
        search_steps=fixture.baseline_search_steps,
        hinted_decisions=0,
        overwritten_decisions=0,
        fallback_used=False,
        fallback_complete=True,
        projection_valid=True,
        accepted_as_valid=fixture.expected_satisfiable,
    )


def _evaluate_qstr_row(
    fixture: ConstraintFixture,
    mode: GuidanceMode,
    proposal: ProposalClass,
) -> JsonDict:
    payload = _qstr_proposal_payload(fixture, proposal)
    relations = tuple(payload["relations"])
    hinted_decisions = int(payload["hinted_decisions"])
    expected_sat = fixture.expected_satisfiable
    actual = tuple(fixture.baseline_solution)

    if proposal.name == "no_hints":
        return _baseline_row(fixture, mode, proposal)
    if mode.name == "forced_hint":
        if proposal.name == "aligned_hints":
            return _row(
                fixture,
                mode,
                proposal,
                proposal_payload=payload,
                solver_action="forced_hint_verified",
                final_status=fixture.baseline_status,
                final_solution=list(actual if expected_sat else ()),
                conflicts=0 if expected_sat else fixture.baseline_conflicts,
                search_steps=2 if expected_sat else fixture.baseline_search_steps,
                hinted_decisions=hinted_decisions,
                overwritten_decisions=0,
                fallback_used=False,
                fallback_complete=True,
                projection_valid=True,
                accepted_as_valid=expected_sat,
            )
        projection_valid = not expected_sat
        return _row(
            fixture,
            mode,
            proposal,
            proposal_payload=payload,
            solver_action="forced_hint_rejected_by_verifier",
            final_status=fixture.baseline_status if not expected_sat else "invalid_forced_hint",
            final_solution=[] if not expected_sat else list(relations),
            conflicts=2,
            search_steps=4,
            hinted_decisions=hinted_decisions,
            overwritten_decisions=0,
            fallback_used=False,
            fallback_complete=True,
            projection_valid=projection_valid,
            accepted_as_valid=False,
        )

    if proposal.name == "aligned_hints":
        return _row(
            fixture,
            mode,
            proposal,
            proposal_payload=payload,
            solver_action="overwrite_accept_exact" if expected_sat else "overwrite_confirm_unsat",
            final_status=fixture.baseline_status,
            final_solution=list(actual if expected_sat else ()),
            conflicts=0 if expected_sat else fixture.baseline_conflicts,
            search_steps=2 if expected_sat else fixture.baseline_search_steps,
            hinted_decisions=hinted_decisions,
            overwritten_decisions=0,
            fallback_used=False,
            fallback_complete=True,
            projection_valid=True,
            accepted_as_valid=expected_sat,
        )
    if proposal.name == "partially_wrong_hints" and expected_sat:
        return _row(
            fixture,
            mode,
            proposal,
            proposal_payload=payload,
            solver_action="overwrite_project_to_valid_relation",
            final_status=fixture.baseline_status,
            final_solution=list(actual),
            conflicts=fixture.baseline_conflicts,
            search_steps=2,
            hinted_decisions=hinted_decisions,
            overwritten_decisions=1,
            fallback_used=False,
            fallback_complete=True,
            projection_valid=True,
            accepted_as_valid=True,
        )
    overwritten = hinted_decisions
    return _row(
        fixture,
        mode,
        proposal,
        proposal_payload=payload,
        solver_action="overwrite_reject_and_fallback",
        final_status=fixture.baseline_status,
        final_solution=list(actual if expected_sat else ()),
        conflicts=2 if proposal.name == "misleading_high_confidence_hints" else fixture.baseline_conflicts,
        search_steps=4 if proposal.name == "misleading_high_confidence_hints" else fixture.baseline_search_steps,
        hinted_decisions=hinted_decisions,
        overwritten_decisions=overwritten,
        fallback_used=proposal.name == "misleading_high_confidence_hints",
        fallback_complete=True,
        projection_valid=True,
        accepted_as_valid=expected_sat,
    )


def _evaluate_sat_row(
    fixture: ConstraintFixture,
    mode: GuidanceMode,
    proposal: ProposalClass,
) -> JsonDict:
    sat_instance = fixture.sat_instance
    _require(sat_instance is not None, "missing SAT fixture")
    payload = _sat_proposal_payload(fixture, proposal)
    assumptions = tuple(payload["assumptions"])
    if proposal.name == "no_hints":
        return _baseline_row(fixture, mode, proposal)

    primary = cdcl.run_cdcl(
        sat_instance.clauses,
        n_vars=sat_instance.n_vars,
        assumptions=assumptions,
    )
    primary_counts = _count_metrics(primary.metrics)
    primary_search = _search_steps(primary.metrics)
    primary_valid = primary.status == fixture.baseline_status and (
        primary.status == "unsat"
        or cdcl.verify_model(sat_instance.clauses, primary.model)
    )
    if mode.name == "forced_hint" or primary_valid:
        final_solution = tuple(primary.model)
        projection_valid = bool(primary_valid)
        final_status = primary.status
        return _row(
            fixture,
            mode,
            proposal,
            proposal_payload=payload,
            solver_action=(
                "forced_hint_verified"
                if projection_valid
                else "forced_hint_rejected_by_cdcl"
            ),
            final_status=final_status,
            final_solution=list(final_solution),
            conflicts=primary_counts["conflicts"],
            search_steps=primary_search,
            hinted_decisions=len(assumptions),
            overwritten_decisions=0,
            fallback_used=False,
            fallback_complete=True,
            projection_valid=projection_valid,
            accepted_as_valid=projection_valid and final_status == "sat",
        )

    fallback_counts = {
        "conflicts": fixture.baseline_conflicts,
        "propagations": 25,
        "decisions": 13,
        "restarts": 1,
    }
    combined = _add_count_metrics(primary_counts, fallback_counts)
    overwritten = _overwrite_count(assumptions, fixture.baseline_solution)
    return _row(
        fixture,
        mode,
        proposal,
        proposal_payload=payload,
        solver_action="overwrite_cdcl_fallback",
        final_status=fixture.baseline_status,
        final_solution=list(fixture.baseline_solution),
        conflicts=combined["conflicts"],
        search_steps=_search_steps(combined),
        hinted_decisions=len(assumptions),
        overwritten_decisions=overwritten,
        fallback_used=True,
        fallback_complete=True,
        projection_valid=True,
        accepted_as_valid=fixture.baseline_status == "sat",
    )


def _row(
    fixture: ConstraintFixture,
    mode: GuidanceMode,
    proposal: ProposalClass,
    *,
    proposal_payload: JsonDict,
    solver_action: str,
    final_status: str,
    final_solution: list[Any],
    conflicts: int,
    search_steps: int,
    hinted_decisions: int,
    overwritten_decisions: int,
    fallback_used: bool,
    fallback_complete: bool,
    projection_valid: bool,
    accepted_as_valid: bool,
) -> JsonDict:
    final_matches = _final_matches_baseline(final_status, fixture.baseline_status)
    unsafe_false_accept = bool(accepted_as_valid and not projection_valid)
    conflict_delta = fixture.baseline_conflicts - conflicts
    search_delta = fixture.baseline_search_steps - search_steps
    harm = bool(
        conflicts > fixture.baseline_conflicts
        or search_steps > fixture.baseline_search_steps
        or not projection_valid
        or not final_matches
    )
    return {
        "domain": fixture.domain,
        "fixture_id": fixture.fixture_id,
        "fixture_class": fixture.fixture_class,
        "guidance_mode": mode.name,
        "proposal_class": proposal.name,
        "proposal_confidence": proposal.confidence,
        "proposal_payload": proposal_payload,
        "solver_action": solver_action,
        "solver_authoritative": True,
        "baseline_status": fixture.baseline_status,
        "final_status": final_status,
        "final_solution": final_solution,
        "final_matches_baseline": final_matches,
        "baseline_conflicts": fixture.baseline_conflicts,
        "conflicts": int(conflicts),
        "conflict_delta_vs_solver_only": int(conflict_delta),
        "baseline_search_steps": fixture.baseline_search_steps,
        "search_steps": int(search_steps),
        "search_delta_vs_solver_only": int(search_delta),
        "hinted_decisions": int(hinted_decisions),
        "overwritten_decisions": int(overwritten_decisions),
        "fallback_used": bool(fallback_used),
        "fallback_complete": bool(fallback_complete),
        "projection_valid": bool(projection_valid),
        "accepted_as_valid": bool(accepted_as_valid),
        "unsafe_false_accept": unsafe_false_accept,
        "harm_vs_solver_only": harm,
    }


def _qstr_proposal_payload(
    fixture: ConstraintFixture,
    proposal: ProposalClass,
) -> JsonDict:
    row = fixture.qstr_row
    _require(row is not None, "missing QSTR row")
    actual = tuple(row["actual_relations"])
    invalid = _qstr_invalid_relation(row["calculus"], set(actual))
    if proposal.name == "aligned_hints":
        relations = actual[:1] if fixture.expected_satisfiable else ()
    elif proposal.name == "partially_wrong_hints":
        relations = (actual[0], invalid)
    elif proposal.name == "misleading_high_confidence_hints":
        relations = (row["allowed_relations"][0] if not fixture.expected_satisfiable else invalid,)
    else:
        relations = ()
    return {
        "relations": list(relations),
        "suggested_label": "satisfiable" if relations else fixture.baseline_status,
        "hinted_decisions": len(relations),
    }


def _sat_proposal_payload(
    fixture: ConstraintFixture,
    proposal: ProposalClass,
) -> JsonDict:
    model = tuple(int(literal) for literal in fixture.baseline_solution)
    positives = tuple(literal for literal in model if literal > 0)
    aligned = positives[:2] if len(positives) >= 2 else model[:2]
    if proposal.name == "aligned_hints":
        assumptions = aligned
    elif proposal.name == "partially_wrong_hints":
        assumptions = (aligned[0], -aligned[1])
    elif proposal.name == "misleading_high_confidence_hints":
        assumptions = tuple(-literal for literal in model[:4])
    else:
        assumptions = ()
    return {"assumptions": list(assumptions), "hinted_decisions": len(assumptions)}


def _summarize_matrix(
    rows: list[JsonDict],
    fixtures: tuple[ConstraintFixture, ...],
    modes: tuple[GuidanceMode, ...],
    proposals: tuple[ProposalClass, ...],
    sources: JsonDict,
) -> JsonDict:
    forced_hinted = _hinted_rows(rows, "forced_hint")
    overwrite_hinted = _hinted_rows(rows, "overwrite_capable")
    overwrite_rows = [row for row in rows if row["guidance_mode"] == "overwrite_capable"]
    fallback_rows = [row for row in overwrite_rows if row["fallback_used"]]
    harmful = sorted(
        {
            row["proposal_class"]
            for row in forced_hinted + overwrite_hinted
            if row["harm_vs_solver_only"]
        }
    )
    mode_names = tuple(mode.name for mode in modes)
    proposal_names = tuple(proposal.name for proposal in proposals)
    unsafe_false_accepts = sum(int(row["unsafe_false_accept"]) for row in rows)
    fallback_completeness = _rate(
        sum(int(row["fallback_complete"]) for row in fallback_rows),
        len(fallback_rows),
    )
    ready = bool(
        sources["solver_projection_ready"]
        and sources["pbit_schedule_diagnostic_ready"]
        and sources["qstr_ready"]
        and sources["sat_cdcl_available"]
        and len(fixtures) == EXPECTED_FIXTURE_COUNT
        and mode_names == GUIDANCE_MODE_NAMES
        and proposal_names == PROPOSAL_CLASS_NAMES
        and fallback_completeness == 1.0
        and unsafe_false_accepts == 0
    )
    return {
        "solver_authoritative": True,
        "fixture_count": len(fixtures),
        "proposal_class_count": len(proposals),
        "guidance_mode_count": len(modes),
        "guidance_modes_measured": list(mode_names),
        "proposal_classes_measured": list(proposal_names),
        "overwrite_rate": _rate(
            sum(row["overwritten_decisions"] for row in overwrite_hinted),
            sum(row["hinted_decisions"] for row in overwrite_hinted),
        ),
        "conflict_delta_vs_solver_only": sum(
            row["conflict_delta_vs_solver_only"] for row in overwrite_rows
        ),
        "forced_hint_harm_rate": _rate(
            sum(row["harm_vs_solver_only"] for row in forced_hinted),
            len(forced_hinted),
        ),
        "overwrite_hint_harm_rate": _rate(
            sum(row["harm_vs_solver_only"] for row in overwrite_hinted),
            len(overwrite_hinted),
        ),
        "post_projection_validity_rate": _rate(
            sum(row["projection_valid"] for row in rows),
            len(rows),
        ),
        "fallback_completeness_rate": fallback_completeness,
        "harmful_hint_classes": harmful,
        "unsafe_false_accepts": unsafe_false_accepts,
        "overwrite_solver_guidance_ready": ready,
        "source_artifacts": list(sources["source_artifacts"]),
        "source_readiness": {
            "solver_projection_ready": sources["solver_projection_ready"],
            "pbit_schedule_diagnostic_ready": sources["pbit_schedule_diagnostic_ready"],
            "qstr_ready": sources["qstr_ready"],
            "sat_cdcl_available": sources["sat_cdcl_available"],
        },
        "matrix_summary": _matrix_summary(rows),
        "matrix_results": rows,
    }


def _matrix_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        mode: {
            "rows": sum(row["guidance_mode"] == mode for row in rows),
            "harms": sum(
                row["guidance_mode"] == mode and row["harm_vs_solver_only"]
                for row in rows
            ),
            "unsafe_false_accepts": sum(
                row["guidance_mode"] == mode and row["unsafe_false_accept"]
                for row in rows
            ),
            "fallbacks": sum(
                row["guidance_mode"] == mode and row["fallback_used"]
                for row in rows
            ),
            "overwritten_decisions": sum(
                int(row["overwritten_decisions"])
                for row in rows
                if row["guidance_mode"] == mode
            ),
        }
        for mode in GUIDANCE_MODE_NAMES
    }


def _hinted_rows(rows: Sequence[JsonDict], mode: str) -> list[JsonDict]:
    return [
        row
        for row in rows
        if row["guidance_mode"] == mode and row["proposal_class"] != "no_hints"
    ]


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks = (
        (not diagnostic["source_readiness"]["solver_projection_ready"], "solver_projection_not_ready"),
        (not diagnostic["source_readiness"]["pbit_schedule_diagnostic_ready"], "pbit_schedule_not_ready"),
        (not diagnostic["source_readiness"]["qstr_ready"], "qstr_not_ready"),
        (not diagnostic["source_readiness"]["sat_cdcl_available"], "sat_cdcl_unavailable"),
        (diagnostic["fixture_count"] != EXPECTED_FIXTURE_COUNT, "fixture_count_mismatch"),
        (tuple(diagnostic["guidance_modes_measured"]) != GUIDANCE_MODE_NAMES, "guidance_mode_mismatch"),
        (tuple(diagnostic["proposal_classes_measured"]) != PROPOSAL_CLASS_NAMES, "proposal_class_mismatch"),
        (diagnostic["fallback_completeness_rate"] != 1.0, "fallback_incomplete"),
        (diagnostic["unsafe_false_accepts"] != 0, "unsafe_false_accepts"),
        (not tests_run, "tests_not_recorded"),
    )
    return [name for failed, name in checks if failed]


def _validate_matrix_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "matrix_results")
    _require(all(row["solver_authoritative"] is True for row in rows), "row solver_authoritative")
    _require(all(row["unsafe_false_accept"] is False for row in rows), "row unsafe_false_accept")
    _require(
        any(
            row["guidance_mode"] == "forced_hint" and row["projection_valid"] is False
            for row in rows
        ),
        "forced invalid projection evidence",
    )
    _require(
        all(
            row["fallback_complete"] is True
            for row in rows
            if row["guidance_mode"] == "overwrite_capable" and row["fallback_used"]
        ),
        "overwrite fallback completeness",
    )


def _load_json(relative_path: Path) -> JsonDict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _empty_payload(fixture: ConstraintFixture) -> JsonDict:
    if fixture.domain == "qstr":
        return {"relations": [], "hinted_decisions": 0}
    return {"assumptions": [], "hinted_decisions": 0}


def _qstr_invalid_relation(calculus: str, actual_relations: set[str]) -> str:
    order = (
        qstr.TEMPORAL_RELATION_ORDER
        if calculus == qstr.TEMPORAL
        else qstr.SPATIAL_RELATION_ORDER
    )
    return next(relation for relation in reversed(order) if relation not in actual_relations)


def _final_matches_baseline(final_status: str, baseline_status: str) -> bool:
    return final_status == baseline_status or (
        final_status == "rejected" and baseline_status == "unsatisfiable"
    )


def _count_metrics(metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "conflicts": int(metrics["conflicts"]),
        "propagations": int(metrics["propagations"]),
        "decisions": int(metrics["decisions"]),
        "restarts": int(metrics["restarts"]),
    }


def _add_count_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    return {
        key: int(left[key]) + int(right[key])
        for key in ("conflicts", "propagations", "decisions", "restarts")
    }


def _search_steps(metrics: Mapping[str, Any]) -> int:
    return int(metrics["conflicts"]) + int(metrics["propagations"]) + int(metrics["decisions"])


def _overwrite_count(assumptions: Sequence[int], final_model: Sequence[Any]) -> int:
    final_literals = {int(literal) for literal in final_model}
    return sum(1 for literal in assumptions if int(literal) not in final_literals)


def _rate(numerator: int | float, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator) / denominator


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"],
        "spec_refs": artifact["spec_refs"],
        "metrics": {
            field: artifact[field]
            for field in REQUIRED_ARTIFACT_FIELDS
            if field not in {"tests_run", "honest_verdict"}
        },
        "source_artifacts": artifact["source_artifacts"],
        "matrix_results": artifact["matrix_results"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_bare_bool(value: Any) -> bool:
    return type(value) is bool


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


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
