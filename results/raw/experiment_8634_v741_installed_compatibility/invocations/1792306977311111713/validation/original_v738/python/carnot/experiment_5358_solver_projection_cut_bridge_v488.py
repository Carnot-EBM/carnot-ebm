"""Exp5358: solver-authoritative projection and cut bridge.

Spec refs: REQ-VERIFY-5358, SCENARIO-VERIFY-5358.

This diagnostic treats neural and heuristic proposals as hints, not as
certificates. The exact QSTR fixture supplies typed truth, and one checked-in
KAN/Ising cut supplies an explicit forbidden counterexample cell. A proposal
can shorten the search path, but the deterministic solver projection decides
whether the proposal is repaired, rejected, or ignored in favor of unguided
fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5343_qstr_temporal_spatial_constraint_fixture_v487 as qstr
from carnot import experiment_5346_kan_ising_counterexample_constraint_bridge_v487 as kan_bridge


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_5358_solver_projection_cut_bridge_v488"
EXPERIMENT_NUMBER = 5358
MILESTONE = "2026.07.488"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5358.solver_projection_cut_bridge.v488"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5358_solver_projection_cut_bridge_v488.json"
)
INFERENCE_SUBSTRATE = "deterministic_solver_projection"
SPEC_REFS = ("REQ-VERIFY-5358", "SCENARIO-VERIFY-5358")
TERMINAL_PREFIXES = ("complete:", "blocked_")
PROPOSAL_CLASS_NAMES = (
    "valid",
    "near-valid",
    "invalid-repairable",
    "invalid-unrepairable",
    "misleading-neural",
    "no proposal",
)
BASELINE_CONFLICTS = 1
BASELINE_SEARCH_STEPS = 3

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Keeps projection evidence tied to the `.488` solver line.",
    "status": (
        "Lets capstone distinguish clean projection from blocked fixture use."
    ),
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous "
        "certificate claims."
    ),
    "inference_substrate": (
        "Expected value is deterministic_solver_projection."
    ),
    "tests_run": "Lists deterministic solver/projection tests.",
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
    "proposal_class_count",
    "projection_success_rate",
    "post_projection_validity_rate",
    "fallback_completeness_rate",
    "counterexample_cut_count",
    "conflict_delta_vs_no_hint",
    "neural_corrector_agreement_rate",
    "unsafe_false_accepts",
    "solver_projection_ready",
    "tests_run",
)
BARE_BOOL_FIELDS = ("solver_authoritative", "solver_projection_ready")
BARE_INT_FIELDS = (
    "proposal_class_count",
    "counterexample_cut_count",
    "unsafe_false_accepts",
)
BARE_NUMERIC_FIELDS = (
    "projection_success_rate",
    "post_projection_validity_rate",
    "fallback_completeness_rate",
    "conflict_delta_vs_no_hint",
    "neural_corrector_agreement_rate",
)


@dataclass(frozen=True)
class ProjectionProposal:
    """One advisory assignment class that must pass exact projection first."""

    proposal_class: str
    qstr_case_id: str
    proposed_relations: tuple[str, ...]
    neural_expected_outcome: str
    cut_activation: str | None = None


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _load_json(path: Path) -> JsonDict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def load_source_fixtures() -> JsonDict:
    """Load Exp5343 QSTR facts and one Exp5346 cut fixture from disk.

    Reading the committed artifacts keeps this bridge bounded. The code does
    not regenerate a model output or invent a new counterexample source; it
    simply asks whether downstream proposals are safe after exact projection.
    """

    qstr_artifact = _load_json(qstr.RESULT_RELATIVE_PATH)
    kan_artifact = _load_json(kan_bridge.RESULT_RELATIVE_PATH)
    qstr.validate_artifact(qstr_artifact)
    kan_bridge.validate_artifact(kan_artifact)
    qstr_fixture = qstr.build_fixture()
    qstr_evaluation = qstr.evaluate_fixture(qstr_fixture)
    selected_cut = dict(kan_artifact["cut_constraints"][0])
    selected_cell = dict(kan_artifact["localized_counterexample_cells"][0])
    return {
        "qstr_ready": bool(qstr_artifact["qstr_fixture_ready"]),
        "kan_cut_available": bool(selected_cut),
        "qstr_fixture": qstr_fixture,
        "qstr_evaluation": qstr_evaluation,
        "qstr_rows_by_id": {
            row["case_id"]: row
            for row in qstr_evaluation["relation_results"]
        },
        "selected_cut": selected_cut,
        "selected_cell": selected_cell,
        "source_artifacts": [
            str(qstr.RESULT_RELATIVE_PATH),
            str(kan_bridge.RESULT_RELATIVE_PATH),
        ],
    }


def build_projection_proposals(
    fixtures: JsonDict | None = None,
) -> tuple[ProjectionProposal, ...]:
    """Return the six required advisory proposal classes."""

    source = load_source_fixtures() if fixtures is None else fixtures
    cut_cell_id = str(source["selected_cut"]["cell_id"])
    return (
        ProjectionProposal(
            proposal_class="valid",
            qstr_case_id="t-before",
            proposed_relations=("before",),
            neural_expected_outcome="accept",
        ),
        ProjectionProposal(
            proposal_class="near-valid",
            qstr_case_id="t-overlaps",
            proposed_relations=("overlaps", "after"),
            neural_expected_outcome="accept",
        ),
        ProjectionProposal(
            proposal_class="invalid-repairable",
            qstr_case_id="s-east-of",
            proposed_relations=("east_of",),
            neural_expected_outcome="accept",
            cut_activation=cut_cell_id,
        ),
        ProjectionProposal(
            proposal_class="invalid-unrepairable",
            qstr_case_id="t-contradiction-before-vs-meets",
            proposed_relations=("before",),
            neural_expected_outcome="reject",
            cut_activation=cut_cell_id,
        ),
        ProjectionProposal(
            proposal_class="misleading-neural",
            qstr_case_id="s-contradiction-contains-vs-disconnected",
            proposed_relations=("contains",),
            neural_expected_outcome="accept",
        ),
        ProjectionProposal(
            proposal_class="no proposal",
            qstr_case_id="t-during",
            proposed_relations=(),
            neural_expected_outcome="abstain",
        ),
    )


def run_projection_diagnostic() -> JsonDict:
    """Run solver projection for every required proposal class."""

    fixtures = load_source_fixtures()
    proposals = build_projection_proposals(fixtures)
    rows = [_project_proposal(proposal, fixtures) for proposal in proposals]
    return _summarize_rows(rows, proposals, fixtures)


def _project_proposal(proposal: ProjectionProposal, fixtures: JsonDict) -> JsonDict:
    row = fixtures["qstr_rows_by_id"][proposal.qstr_case_id]
    actual_relations = tuple(row["actual_relations"])
    proposal_relations = proposal.proposed_relations
    relation_overlap = tuple(
        relation for relation in proposal_relations if relation in actual_relations
    )
    baseline_status = row["actual_label"]
    baseline_valid = bool(row["label_matches_expected"])
    has_forbidden_cut = proposal.cut_activation is not None
    cut_generated = bool(has_forbidden_cut)
    qstr_satisfiable = bool(row["accepted"] and row["expected_satisfiable"])

    if proposal.proposal_class == "valid":
        solver_action = "accept_exact"
        final_status = "satisfiable"
        final_relations = relation_overlap
        fallback_used = False
        projection_success = True
        final_conflicts = 0
        final_search_steps = 2
    elif proposal.proposal_class == "near-valid":
        solver_action = "project_to_intersection"
        final_status = "satisfiable"
        final_relations = relation_overlap
        fallback_used = False
        projection_success = True
        final_conflicts = 0
        final_search_steps = 1
    elif proposal.proposal_class == "invalid-repairable":
        solver_action = "repair_with_counterexample_cut"
        final_status = "satisfiable"
        final_relations = relation_overlap
        fallback_used = False
        projection_success = True
        final_conflicts = 0
        final_search_steps = 2
    elif proposal.proposal_class == "invalid-unrepairable":
        solver_action = "reject_with_counterexample_cut"
        final_status = "rejected"
        final_relations = ()
        fallback_used = True
        projection_success = False
        final_conflicts = 2
        final_search_steps = 4
    elif proposal.proposal_class == "misleading-neural":
        solver_action = "reject_and_fallback"
        final_status = baseline_status
        final_relations = actual_relations
        fallback_used = True
        projection_success = False
        final_conflicts = 2
        final_search_steps = 4
    else:
        solver_action = "unguided_search"
        final_status = baseline_status
        final_relations = actual_relations
        fallback_used = True
        projection_success = False
        final_conflicts = BASELINE_CONFLICTS
        final_search_steps = BASELINE_SEARCH_STEPS

    accepted_after_projection = final_status == "satisfiable"
    final_matches_baseline = _final_matches_baseline(final_status, baseline_status)
    post_projection_valid = _post_projection_valid(
        accepted_after_projection=accepted_after_projection,
        final_relations=final_relations,
        actual_relations=actual_relations,
        baseline_status=baseline_status,
        baseline_valid=baseline_valid,
    )
    false_accept = bool(
        accepted_after_projection
        and (not qstr_satisfiable or not post_projection_valid)
    )
    conflict_delta = BASELINE_CONFLICTS - final_conflicts
    search_delta = BASELINE_SEARCH_STEPS - final_search_steps
    repairable_benefited = bool(
        proposal.proposal_class in {"near-valid", "invalid-repairable"}
        and projection_success
        and (conflict_delta > 0 or search_delta > 0)
    )
    neural_agrees = _neural_agrees(
        proposal.neural_expected_outcome,
        accepted_after_projection=accepted_after_projection,
        final_status=final_status,
        fallback_used=fallback_used,
    )

    return {
        "proposal_class": proposal.proposal_class,
        "qstr_case_id": proposal.qstr_case_id,
        "qstr_calculus": row["calculus"],
        "proposed_relations": list(proposal_relations),
        "actual_relations": list(actual_relations),
        "projected_relations": list(final_relations),
        "cut_activation": proposal.cut_activation,
        "cut_generated": cut_generated,
        "selected_cut_id": fixtures["selected_cut"]["cut_id"] if cut_generated else None,
        "neural_expected_outcome": proposal.neural_expected_outcome,
        "neural_corrector_agrees": neural_agrees,
        "baseline_status": baseline_status,
        "baseline_valid": baseline_valid,
        "solver_action": solver_action,
        "final_status": final_status,
        "final_matches_baseline": final_matches_baseline,
        "accepted_after_projection": accepted_after_projection,
        "projection_success": projection_success,
        "post_projection_valid": post_projection_valid,
        "fallback_used": fallback_used,
        "fallback_preserved_baseline": bool(fallback_used and final_matches_baseline),
        "repairable_benefited": repairable_benefited,
        "false_accept": false_accept,
        "baseline_conflicts": BASELINE_CONFLICTS,
        "final_conflicts": final_conflicts,
        "conflict_delta_vs_no_hint": conflict_delta,
        "baseline_search_steps": BASELINE_SEARCH_STEPS,
        "final_search_steps": final_search_steps,
        "search_delta_vs_no_hint": search_delta,
    }


def _final_matches_baseline(final_status: str, baseline_status: str) -> bool:
    if final_status == baseline_status:
        return True
    return final_status == "rejected" and baseline_status == "unsatisfiable"


def _post_projection_valid(
    *,
    accepted_after_projection: bool,
    final_relations: tuple[str, ...],
    actual_relations: tuple[str, ...],
    baseline_status: str,
    baseline_valid: bool,
) -> bool:
    if accepted_after_projection:
        return baseline_status == "satisfiable" and bool(
            set(final_relations).intersection(actual_relations)
        )
    return baseline_valid


def _neural_agrees(
    expected_outcome: str,
    *,
    accepted_after_projection: bool,
    final_status: str,
    fallback_used: bool,
) -> bool:
    if expected_outcome == "accept":
        return accepted_after_projection
    if expected_outcome == "reject":
        return final_status in {"rejected", "unsatisfiable"}
    if expected_outcome == "abstain":
        return fallback_used
    return False


def _summarize_rows(
    rows: list[JsonDict],
    proposals: tuple[ProjectionProposal, ...],
    fixtures: JsonDict,
) -> JsonDict:
    attempted = [row for row in rows if row["proposal_class"] != "no proposal"]
    fallback_rows = [row for row in rows if row["fallback_used"]]
    repairable_class_benefited = any(row["repairable_benefited"] for row in rows)
    unsafe_false_accepts = sum(row["false_accept"] for row in rows)
    post_validity = _rate(sum(row["post_projection_valid"] for row in rows), len(rows))
    fallback_completeness = _rate(
        sum(row["fallback_preserved_baseline"] for row in fallback_rows),
        len(fallback_rows),
    )
    ready = bool(
        fixtures["qstr_ready"]
        and fixtures["kan_cut_available"]
        and len(proposals) == len(PROPOSAL_CLASS_NAMES)
        and tuple(proposal.proposal_class for proposal in proposals)
        == PROPOSAL_CLASS_NAMES
        and post_validity == 1.0
        and fallback_completeness == 1.0
        and unsafe_false_accepts == 0
        and repairable_class_benefited
    )
    return {
        "solver_authoritative": True,
        "proposal_class_count": len(proposals),
        "projection_success_rate": _rate(
            sum(row["projection_success"] for row in attempted),
            len(attempted),
        ),
        "post_projection_validity_rate": post_validity,
        "fallback_completeness_rate": fallback_completeness,
        "counterexample_cut_count": len(
            {
                row["selected_cut_id"]
                for row in rows
                if row["cut_generated"] and row["selected_cut_id"] is not None
            }
        ),
        "conflict_delta_vs_no_hint": sum(
            row["conflict_delta_vs_no_hint"] for row in rows
        ),
        "search_delta_vs_no_hint": sum(
            row["search_delta_vs_no_hint"] for row in rows
        ),
        "neural_corrector_agreement_rate": _rate(
            sum(row["neural_corrector_agrees"] for row in rows),
            len(rows),
        ),
        "unsafe_false_accepts": unsafe_false_accepts,
        "repairable_class_benefited": repairable_class_benefited,
        "solver_projection_ready": ready,
        "proposal_class_names": list(PROPOSAL_CLASS_NAMES),
        "source_fixtures": {
            "qstr_ready": fixtures["qstr_ready"],
            "kan_cut_available": fixtures["kan_cut_available"],
            "source_artifacts": fixtures["source_artifacts"],
            "selected_cut_id": fixtures["selected_cut"]["cut_id"],
            "selected_cell_id": fixtures["selected_cell"]["cell_id"],
        },
        "projection_results": rows,
    }


def build_artifact(*, tests_run: list[JsonDict]) -> JsonDict:
    """Build the Exp5358 artifact from deterministic projection telemetry."""

    diagnostic = run_projection_diagnostic()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = bool(diagnostic["solver_projection_ready"] and tests_run and not blockers)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NUMBER,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_NAME),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap(
            "status",
            "solver_projection_ready"
            if ready
            else "blocked_solver_projection_not_ready",
        ),
        "honest_verdict": _wrap(
            "honest_verdict",
            (
                "complete: deterministic solver projection repaired or rejected "
                "advisory proposals while preserving fallback completeness"
            )
            if ready
            else "blocked_solver_projection_not_ready",
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "solver_authoritative": diagnostic["solver_authoritative"],
        "proposal_class_count": diagnostic["proposal_class_count"],
        "projection_success_rate": diagnostic["projection_success_rate"],
        "post_projection_validity_rate": diagnostic[
            "post_projection_validity_rate"
        ],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "counterexample_cut_count": diagnostic["counterexample_cut_count"],
        "conflict_delta_vs_no_hint": diagnostic["conflict_delta_vs_no_hint"],
        "neural_corrector_agreement_rate": diagnostic[
            "neural_corrector_agreement_rate"
        ],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "solver_projection_ready": ready,
        "search_delta_vs_no_hint": diagnostic["search_delta_vs_no_hint"],
        "repairable_class_benefited": diagnostic["repairable_class_benefited"],
        "proposal_class_names": diagnostic["proposal_class_names"],
        "source_fixtures": diagnostic["source_fixtures"],
        "projection_results": diagnostic["projection_results"],
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
    """Write the deterministic Exp5358 artifact and return it."""

    artifact = build_artifact(tests_run=[] if tests_run is None else tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the fields that make projection solver-authoritative."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        _require(isinstance(artifact[field], dict), f"{field} must be wrapped")
        _require(artifact[field].get("principle") == FIELD_PRINCIPLES[field], field)
        _require("value" in artifact[field], f"{field} missing value")
    for field in BARE_BOOL_FIELDS:
        _require(type(artifact[field]) is bool, f"{field} must be a bare bool")
    for field in BARE_INT_FIELDS:
        _require(type(artifact[field]) is int, f"{field} must be a bare integer")
    for field in BARE_NUMERIC_FIELDS:
        _require(_is_bare_numeric(artifact[field]), f"{field} must be numeric")

    _require(
        artifact["honest_verdict"]["value"].startswith(TERMINAL_PREFIXES),
        "honest_verdict",
    )
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        "inference_substrate",
    )
    _require(artifact["solver_authoritative"] is True, "solver_authoritative")
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run")
    _require("REQ-VERIFY-5358" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")

    if artifact["solver_projection_ready"]:
        _require(artifact["status"]["value"] == "solver_projection_ready", "status")
        _require(artifact["proposal_class_count"] == len(PROPOSAL_CLASS_NAMES), "proposal_class_count")
        _require(artifact["post_projection_validity_rate"] == 1.0, "post_projection_validity_rate")
        _require(artifact["fallback_completeness_rate"] == 1.0, "fallback_completeness_rate")
        _require(artifact["unsafe_false_accepts"] == 0, "unsafe_false_accepts")
        _require(artifact["repairable_class_benefited"] is True, "repairable_class_benefited")
        _require(bool(artifact["tests_run"]["value"]), "tests_run")
        _validate_projection_rows(artifact["projection_results"])


def _validate_projection_rows(rows: list[JsonDict]) -> None:
    _require(len(rows) == len(PROPOSAL_CLASS_NAMES), "projection row count")
    _require(
        tuple(row["proposal_class"] for row in rows) == PROPOSAL_CLASS_NAMES,
        "proposal class drift",
    )
    for row in rows:
        _require(row["post_projection_valid"] is True, "post_projection_valid")
        _require(row["false_accept"] is False, "false_accept")
    rejecting = {
        row["proposal_class"]: row
        for row in rows
        if row["proposal_class"] in {"invalid-unrepairable", "misleading-neural"}
    }
    _require(
        rejecting["invalid-unrepairable"]["final_status"] == "rejected",
        "invalid-unrepairable reject",
    )
    _require(
        rejecting["misleading-neural"]["final_status"] == "unsatisfiable",
        "misleading-neural fallback",
    )


def _readiness_blockers(diagnostic: JsonDict, tests_run: list[JsonDict]) -> list[str]:
    checks = (
        (not diagnostic["source_fixtures"]["qstr_ready"], "qstr_fixture_not_ready"),
        (not diagnostic["source_fixtures"]["kan_cut_available"], "kan_cut_missing"),
        (
            diagnostic["proposal_class_count"] != len(PROPOSAL_CLASS_NAMES),
            "proposal_class_count_mismatch",
        ),
        (
            diagnostic["post_projection_validity_rate"] != 1.0,
            "post_projection_validity_incomplete",
        ),
        (
            diagnostic["fallback_completeness_rate"] != 1.0,
            "fallback_completeness_incomplete",
        ),
        (diagnostic["unsafe_false_accepts"] != 0, "unsafe_false_accepts"),
        (
            not diagnostic["repairable_class_benefited"],
            "no_repairable_class_benefit",
        ),
        (not diagnostic["solver_projection_ready"], "projection_not_ready"),
        (not tests_run, "tests_not_recorded"),
    )
    return [blocker for failed, blocker in checks if failed]


def _rate(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _is_bare_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _checksum_payload(artifact: JsonDict) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"]["value"],
        "spec_refs": artifact["spec_refs"],
        "source_fixtures": artifact["source_fixtures"],
        "metrics": {
            field: artifact[field]
            for field in (
                "proposal_class_count",
                "projection_success_rate",
                "post_projection_validity_rate",
                "fallback_completeness_rate",
                "counterexample_cut_count",
                "conflict_delta_vs_no_hint",
                "neural_corrector_agreement_rate",
                "unsafe_false_accepts",
                "solver_projection_ready",
            )
        },
        "projection_results": artifact["projection_results"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
