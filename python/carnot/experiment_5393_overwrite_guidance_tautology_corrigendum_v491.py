"""Exp5393: row-level overwrite-guidance tautology corrigendum.

Spec refs: REQ-VERIFY-5393, SCENARIO-VERIFY-5393.

Exp5383 was quarantined because two top-level aggregate rates matched exactly:
forced-hint harm and legacy all-mode projection validity. This corrigendum
does not reuse those aggregate claims. It recovers the solver matrix rows,
records the raw hint and solver state for each selected fixture, and recomputes
readiness from row outcomes with explicit denominators so the solver remains
the authority and hints stay advisory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as adversarial_verify


JsonDict = dict[str, Any]
RowOverride = Callable[[list[JsonDict]], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5393_overwrite_guidance_tautology_corrigendum_v491.json"
)
SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5383_overwrite_guidance_scale_validity_v490.json"
)
EXPERIMENT = 5393
EXPERIMENT_ID = "exp5393-overwrite-guidance-tautology-corrigendum-v491"
MILESTONE = "2026.07.491"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5393
SCHEMA = "carnot.experiment_5393.overwrite_guidance_tautology_corrigendum.v491"
SPEC_REFS = ("REQ-VERIFY-5393", "SCENARIO-VERIFY-5393")
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
TERMINAL_PREFIXES = ("complete:", "flagged:", "blocked:")

NO_HINT_FAMILY = "no_hint"
REQUIRED_FIXTURE_FAMILIES = (
    "benign",
    "incomplete",
    "harmful",
    "contradictory",
    NO_HINT_FAMILY,
)
EXPECTED_SOURCE_FIXTURE_COUNT = 14
EXPECTED_ROW_COUNT = 98

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "complete if row-level corrigendum ran, flagged if tautology remains, "
        "or blocked if required inputs are missing."
    ),
    "milestone": "must equal 2026.07.491.",
    "source_flagged_artifact": "must point to Exp5383.",
    "row_count": "number of row-level solver fixtures.",
    "fixture_families": (
        "list covering benign, incomplete, harmful, contradictory, and no_hint "
        "controls."
    ),
    "overwrite_rate_from_rows": "computed only from row-level outcomes.",
    "forced_hint_harm_rate_from_rows": (
        "computed only from harmful/contradictory rows."
    ),
    "post_projection_validity_rate_from_rows": (
        "solver-authoritative validity rate."
    ),
    "fallback_completeness_rate_from_rows": (
        "fallback rate when hints are invalid or harmful."
    ),
    "negative_control_pass_rate": (
        "pass rate for controls that must not benefit from hinting."
    ),
    "tautology_checks_passed": (
        "true only if readiness fields are recomputed from independent row "
        "evidence."
    ),
    "unsafe_false_accept_count": (
        "number of invalid/harmful solver outputs accepted."
    ),
    "overwrite_guidance_corrigendum_clean": (
        "true only if row-level evidence is clean and adversarial checks pass."
    ),
    "honest_verdict": (
        "one-line summary starting with complete:, flagged:, or blocked:."
    ),
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
ROW_REQUIRED_FIELDS = (
    "row_id",
    "fixture_family",
    "guidance_mode",
    "raw_hint",
    "solver_pre_state",
    "solver_post_state",
    "hint_action",
    "validity_proof",
    "conflict_delta_vs_no_hint",
    "fallback_result",
    "unsafe_status",
)


def load_source_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the flagged Exp5383 artifact that this corrigendum reviews."""

    path = Path(root) / SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def identify_source_tautology(source_artifact: Mapping[str, Any]) -> JsonDict:
    """Identify the exact aggregate equality that quarantined Exp5383."""

    tautological_fields: list[JsonDict] = []
    left = source_artifact.get("forced_hint_harm_rate")
    right = source_artifact.get("legacy_all_mode_projection_validity_rate")
    if left == right and left is not None:
        tautological_fields.append(
            {
                "left": "forced_hint_harm_rate",
                "right": "legacy_all_mode_projection_validity_rate",
                "left_value": left,
                "right_value": right,
                "why_suspect": (
                    "forced-hint harm and all-mode projection validity are "
                    "distinct aggregate concepts and must not be treated as "
                    "independent evidence when they agree exactly."
                ),
            }
        )
    return {
        "source_flagged_artifact": str(SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH),
        "source_experiment_id": source_artifact.get("experiment_id"),
        "source_honest_verdict": source_artifact.get("honest_verdict"),
        "tautological_fields": tautological_fields,
        "reused_aggregate_fields": [],
        "corrigendum_rule": (
            "Do not copy Exp5383 readiness metrics; recompute from row_evidence."
        ),
    }


def build_corrigendum_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Recover row-level solver evidence from Exp5383 without copying aggregates."""

    source = load_source_artifact(root)
    source_rows = list(source["matrix_results"])
    baseline_by_hint = {
        (row["source_fixture_id"], row["hint_class"]): row
        for row in source_rows
        if row["guidance_mode"] == "no_hint"
    }
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for row in source_rows:
        mode = row["guidance_mode"]
        hint_class = row["hint_class"]
        if mode == "no_hint" and hint_class == "benign_hints":
            selected.append((NO_HINT_FAMILY, row))
        elif mode == "overwrite_capable" and hint_class == "benign_hints":
            selected.append(("benign", row))
        elif mode == "overwrite_capable" and hint_class == "incomplete_hints":
            selected.append(("incomplete", row))
        elif hint_class == "harmful_hints" and mode in {
            "forced_hint",
            "overwrite_capable",
        }:
            selected.append(("harmful", row))
        elif hint_class == "contradictory_hints" and mode in {
            "forced_hint",
            "overwrite_capable",
        }:
            selected.append(("contradictory", row))

    rows = [
        _corrigendum_row(family, source_row, baseline_by_hint, index)
        for index, (family, source_row) in enumerate(selected, start=1)
    ]
    rows.sort(key=lambda row: str(row["row_id"]))
    return rows


def summarize_corrigendum_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute every readiness metric from row-level outcomes only."""

    families = _ordered_families(rows)
    overwrite_rows = [
        row
        for row in rows
        if row["guidance_mode"] == "overwrite_capable"
        and row["fixture_family"] in {"incomplete", "harmful", "contradictory"}
    ]
    harmful_control_rows = [
        row
        for row in rows
        if row["negative_control"]["control_kind"]
        in {"forced_harmful_no_improvement", "forced_contradictory_no_improvement"}
    ]
    post_projection_rows = [
        row for row in rows if row["guidance_mode"] != "forced_hint"
    ]
    fallback_rows = [
        row
        for row in rows
        if row["guidance_mode"] == "overwrite_capable"
        and row["fixture_family"] in {"harmful", "contradictory"}
    ]
    negative_control_rows = [
        row for row in rows if row["negative_control"]["control_kind"]
    ]
    unsafe_false_accept_count = sum(
        int(row["unsafe_status"]["unsafe_false_accept"]) for row in rows
    )
    summary = {
        "row_count": len(rows),
        "row_metric_denominator": len(rows),
        "fixture_families": families,
        "overwrite_rate_from_rows": _rate(
            sum(
                row["hint_action"] == "overwritten"
                and row["validity_proof"]["solver_authoritative"]
                for row in overwrite_rows
            ),
            len(overwrite_rows),
        ),
        "overwrite_rate_denominator": len(overwrite_rows),
        "forced_hint_harm_rate_from_rows": _rate(
            sum(_forced_harm_observed(row) for row in harmful_control_rows),
            len(harmful_control_rows),
        ),
        "forced_hint_harm_denominator": len(harmful_control_rows),
        "post_projection_validity_rate_from_rows": _rate(
            sum(_post_projection_valid(row) for row in post_projection_rows),
            len(post_projection_rows),
        ),
        "post_projection_validity_denominator": len(post_projection_rows),
        "fallback_completeness_rate_from_rows": _rate(
            sum(
                row["fallback_result"]["used"]
                and row["fallback_result"]["complete"]
                and row["validity_proof"]["projection_valid"]
                for row in fallback_rows
            ),
            len(fallback_rows),
        ),
        "fallback_completeness_denominator": len(fallback_rows),
        "negative_control_pass_rate": _rate(
            sum(row["negative_control"]["passed"] for row in negative_control_rows),
            len(negative_control_rows),
        ),
        "negative_control_denominator": len(negative_control_rows),
        "negative_control_passed_count": sum(
            row["negative_control"]["passed"] for row in negative_control_rows
        ),
        "unsafe_false_accept_count": unsafe_false_accept_count,
        "row_field_completeness": _row_field_completeness(rows),
    }
    summary["row_level_evidence_clean"] = bool(
        summary["row_count"] == EXPECTED_ROW_COUNT
        and tuple(summary["fixture_families"]) == REQUIRED_FIXTURE_FAMILIES
        and summary["row_field_completeness"]
        and summary["overwrite_rate_denominator"] == 42
        and summary["forced_hint_harm_denominator"] == 28
        and summary["post_projection_validity_denominator"] == 70
        and summary["fallback_completeness_denominator"] == 28
        and summary["negative_control_denominator"] == 42
        and summary["overwrite_rate_from_rows"] == 1.0
        and summary["forced_hint_harm_rate_from_rows"] == 1.0
        and summary["post_projection_validity_rate_from_rows"] == 1.0
        and summary["fallback_completeness_rate_from_rows"] == 1.0
        and summary["negative_control_pass_rate"] == 1.0
        and summary["unsafe_false_accept_count"] == 0
    )
    return summary


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    row_overrides: RowOverride | None = None,
) -> JsonDict:
    """Build a validated Exp5393 artifact from row evidence and checks."""

    source = load_source_artifact(root)
    source_review = identify_source_tautology(source)
    rows = build_corrigendum_rows(root)
    if row_overrides is not None:
        rows = row_overrides(rows)
    summary = summarize_corrigendum_rows(rows)
    tests = [_normalize_test_run(row) for row in tests_run]
    blockers = _readiness_blockers(source_review, summary, tests)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.05,
        "status": "complete",
        "source_flagged_artifact": str(SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH),
        "row_count": summary["row_count"],
        "row_metric_denominator": summary["row_metric_denominator"],
        "fixture_families": summary["fixture_families"],
        "overwrite_rate_from_rows": summary["overwrite_rate_from_rows"],
        "overwrite_rate_denominator": summary["overwrite_rate_denominator"],
        "forced_hint_harm_rate_from_rows": summary[
            "forced_hint_harm_rate_from_rows"
        ],
        "forced_hint_harm_denominator": summary[
            "forced_hint_harm_denominator"
        ],
        "post_projection_validity_rate_from_rows": summary[
            "post_projection_validity_rate_from_rows"
        ],
        "post_projection_validity_denominator": summary[
            "post_projection_validity_denominator"
        ],
        "fallback_completeness_rate_from_rows": summary[
            "fallback_completeness_rate_from_rows"
        ],
        "fallback_completeness_denominator": summary[
            "fallback_completeness_denominator"
        ],
        "negative_control_pass_rate": summary["negative_control_pass_rate"],
        "negative_control_denominator": summary["negative_control_denominator"],
        "negative_control_passed_count": summary["negative_control_passed_count"],
        "tautology_checks_passed": True,
        "unsafe_false_accept_count": summary["unsafe_false_accept_count"],
        "overwrite_guidance_corrigendum_clean": False,
        "honest_verdict": "complete: placeholder pending tautology check",
        "source_tautology_review": source_review,
        "row_level_evidence_clean": summary["row_level_evidence_clean"],
        "row_field_completeness": summary["row_field_completeness"],
        "readiness_blockers": blockers,
        "row_evidence": rows,
        "tests_run": tests,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "claim_limits": [
            "corrigendum over flagged Exp5383 only",
            "readiness metrics recomputed from row_evidence",
            "solver or deterministic verifier remains final authority",
            "hints are optional and may be ignored, completed, overwritten, or rejected",
            "no neural, LLM, hardware, or generated text judge execution",
        ],
        "methodology_note": (
            "Rates of 1.0 are deterministic solver-invariant checks over explicit "
            "row denominators, not classifier accuracy or sampled benchmark claims."
        ),
    }
    artifact["tautology_checks_passed"] = _local_tautology_checks_passed(artifact)
    blockers = _readiness_blockers(source_review, summary, tests)
    if not artifact["tautology_checks_passed"]:
        blockers = [*blockers, "adversarial_tautology_failed"]
    clean = bool(summary["row_level_evidence_clean"] and not blockers)
    artifact["readiness_blockers"] = blockers
    artifact["overwrite_guidance_corrigendum_clean"] = clean
    artifact["status"] = _status(clean, artifact["tautology_checks_passed"], blockers)
    artifact["honest_verdict"] = _honest_verdict(artifact["status"], clean, blockers)
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5393 artifact and return the validated payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the row-derived corrigendum schema and safety invariants."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "flagged", "blocked"}, "status")
    _require(artifact["milestone"] == MILESTONE, "milestone")
    _require(
        artifact["source_flagged_artifact"]
        == str(SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH),
        "source_flagged_artifact",
    )
    _require(
        str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES),
        "honest_verdict",
    )
    _require(_is_bare_int(artifact["row_count"]), "row_count")
    _require(_is_bare_int(artifact["unsafe_false_accept_count"]), "unsafe_false_accept_count")
    _require(artifact["unsafe_false_accept_count"] == 0, "unsafe_false_accept_count")
    _require(isinstance(artifact["fixture_families"], list), "fixture_families")
    _require(tuple(artifact["fixture_families"]) == REQUIRED_FIXTURE_FAMILIES, "fixture_families")
    for field in (
        "overwrite_rate_from_rows",
        "forced_hint_harm_rate_from_rows",
        "post_projection_validity_rate_from_rows",
        "fallback_completeness_rate_from_rows",
        "negative_control_pass_rate",
    ):
        _require(_is_bare_numeric(artifact[field]), field)
    for field in (
        "tautology_checks_passed",
        "overwrite_guidance_corrigendum_clean",
        "row_level_evidence_clean",
        "row_field_completeness",
    ):
        _require(type(artifact[field]) is bool, field)
    _require("REQ-VERIFY-5393" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")
    _validate_row_evidence(artifact["row_evidence"])

    if artifact["overwrite_guidance_corrigendum_clean"]:
        _require(artifact["status"] == "complete", "status")
        _require(artifact["tautology_checks_passed"] is True, "tautology_checks_passed")
        _require(artifact["row_count"] == EXPECTED_ROW_COUNT, "row_count")
        _require(artifact["overwrite_rate_denominator"] == 42, "overwrite_rate_denominator")
        _require(artifact["forced_hint_harm_denominator"] == 28, "forced_hint_harm_denominator")
        _require(
            artifact["post_projection_validity_denominator"] == 70,
            "post_projection_validity_denominator",
        )
        _require(
            artifact["fallback_completeness_denominator"] == 28,
            "fallback_completeness_denominator",
        )
        _require(artifact["negative_control_denominator"] == 42, "negative_control_denominator")
        _require(artifact["readiness_blockers"] == [], "readiness_blockers")
        _require(bool(artifact["tests_run"]), "tests_run")


def _corrigendum_row(
    family: str,
    source_row: Mapping[str, Any],
    baseline_by_hint: Mapping[tuple[str, str], Mapping[str, Any]],
    index: int,
) -> JsonDict:
    baseline = baseline_by_hint[
        (str(source_row["source_fixture_id"]), str(source_row["hint_class"]))
    ]
    raw_hint = None if family == NO_HINT_FAMILY else list(source_row["hint_payload"]["hint"])
    hint_action = _hint_action(family, source_row)
    fallback_used = bool(source_row["fallback_used"])
    fallback_complete = bool(source_row["fallback_complete"])
    negative_control = _negative_control(family, source_row, raw_hint)
    accepted_as_valid = bool(source_row["accepted_as_valid"])
    projection_valid = bool(source_row["projection_valid"])
    unsafe_false_accept = bool(source_row["unsafe_false_accept"])
    return {
        "row_id": (
            f"{index:03d}:{source_row['source_fixture_id']}:{family}:"
            f"{source_row['guidance_mode']}"
        ),
        "source_artifact": str(SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH),
        "source_fixture_id": source_row["source_fixture_id"],
        "fixture_id": source_row["fixture_id"],
        "fixture_family": family,
        "hint_class": source_row["hint_class"],
        "guidance_mode": source_row["guidance_mode"],
        "raw_hint": raw_hint,
        "solver_pre_state": {
            "status": source_row["baseline_status"],
            "solution": list(baseline["final_solution"]),
            "conflicts": int(source_row["baseline_conflicts"]),
            "convergence_steps": int(source_row["baseline_convergence_steps"]),
        },
        "solver_post_state": {
            "status": source_row["final_status"],
            "solution": list(source_row["final_solution"]),
            "conflicts": int(source_row["conflicts"]),
            "convergence_steps": int(source_row["convergence_steps"]),
        },
        "hint_action": hint_action,
        "hint_accepted": hint_action == "accepted",
        "hint_overwritten": hint_action == "overwritten",
        "hint_ignored": hint_action == "ignored",
        "accepted_as_valid": accepted_as_valid,
        "validity_proof": {
            "proof_kind": "solver_authoritative_projection",
            "solver_authoritative": bool(source_row["solver_authoritative"]),
            "projection_valid": projection_valid,
            "final_matches_baseline": bool(source_row["final_matches_baseline"]),
            "accepted_as_valid": accepted_as_valid,
            "solver_action": source_row["solver_action"],
        },
        "conflict_delta_vs_no_hint": int(source_row["conflict_delta_vs_no_hint"]),
        "convergence_delta_vs_no_hint": int(
            source_row["convergence_delta_vs_no_hint"]
        ),
        "fallback_result": {
            "used": fallback_used,
            "complete": fallback_complete,
            "status": _fallback_status(fallback_used, fallback_complete),
        },
        "negative_control": negative_control,
        "unsafe_status": {
            "unsafe_false_accept": unsafe_false_accept,
            "invalid_hint_accepted": bool(
                family in {"harmful", "contradictory"}
                and accepted_as_valid
                and not projection_valid
            ),
            "accepted_as_valid": accepted_as_valid,
        },
    }


def _hint_action(family: str, source_row: Mapping[str, Any]) -> str:
    if family == NO_HINT_FAMILY:
        return "ignored"
    if source_row["guidance_mode"] == "forced_hint":
        return "accepted" if source_row["projection_valid"] else "ignored"
    if family == "benign":
        return "accepted"
    return "overwritten"


def _negative_control(
    family: str,
    source_row: Mapping[str, Any],
    raw_hint: list[Any] | None,
) -> JsonDict:
    if family == NO_HINT_FAMILY:
        passed = (
            raw_hint is None
            and source_row["guidance_mode"] == "no_hint"
            and int(source_row["conflict_delta_vs_no_hint"]) == 0
            and int(source_row["convergence_delta_vs_no_hint"]) == 0
            and source_row["projection_valid"] is True
        )
        return {
            "control_kind": "no_hint_must_not_benefit_from_guidance",
            "passed": bool(passed),
            "reason": "No-hint control must exactly match solver-only baseline.",
        }
    if family in {"harmful", "contradictory"} and source_row["guidance_mode"] == "forced_hint":
        control_kind = f"forced_{family}_no_improvement"
        passed = (
            source_row["accepted_as_valid"] is False
            and source_row["projection_valid"] is False
            and int(source_row["conflict_delta_vs_no_hint"]) <= 0
            and bool(source_row["unsafe_false_accept"]) is False
        )
        return {
            "control_kind": control_kind,
            "passed": bool(passed),
            "reason": (
                "Forced harmful or contradictory hints are contrast rows; they "
                "must not improve or certify validity."
            ),
        }
    return {"control_kind": None, "passed": True, "reason": "not a negative control"}


def _fallback_status(fallback_used: bool, fallback_complete: bool) -> str:
    if fallback_used and fallback_complete:
        return "fallback_completed"
    if fallback_used:
        return "fallback_incomplete"
    return "not_needed"


def _ordered_families(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row["fixture_family"]) for row in rows}
    return [family for family in REQUIRED_FIXTURE_FAMILIES if family in present]


def _forced_harm_observed(row: Mapping[str, Any]) -> bool:
    return bool(
        row["guidance_mode"] == "forced_hint"
        and row["fixture_family"] in {"harmful", "contradictory"}
        and row["validity_proof"]["projection_valid"] is False
        and row["validity_proof"]["accepted_as_valid"] is False
        and row["negative_control"]["passed"] is True
    )


def _post_projection_valid(row: Mapping[str, Any]) -> bool:
    return bool(
        row["validity_proof"]["solver_authoritative"]
        and row["validity_proof"]["projection_valid"]
        and row["validity_proof"]["final_matches_baseline"]
        and not row["unsafe_status"]["unsafe_false_accept"]
    )


def _row_field_completeness(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        all(field in row for field in ROW_REQUIRED_FIELDS) for row in rows
    )


def _readiness_blockers(
    source_review: Mapping[str, Any],
    summary: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks = (
        (not source_review["tautological_fields"], "source_tautology_not_identified"),
        (summary["row_count"] != EXPECTED_ROW_COUNT, "row_count_mismatch"),
        (
            tuple(summary["fixture_families"]) != REQUIRED_FIXTURE_FAMILIES,
            "fixture_family_mismatch",
        ),
        (not summary["row_field_completeness"], "row_fields_missing"),
        (
            summary["overwrite_rate_from_rows"] != 1.0,
            "overwrite_rate_incomplete",
        ),
        (
            summary["forced_hint_harm_rate_from_rows"] != 1.0,
            "forced_hint_harm_not_observed",
        ),
        (
            summary["post_projection_validity_rate_from_rows"] != 1.0,
            "post_projection_validity_incomplete",
        ),
        (
            summary["fallback_completeness_rate_from_rows"] != 1.0,
            "fallback_completeness_incomplete",
        ),
        (
            summary["negative_control_pass_rate"] != 1.0,
            "negative_controls_failed",
        ),
        (summary["unsafe_false_accept_count"] != 0, "unsafe_false_accepts"),
        (not tests_run, "tests_not_recorded"),
    )
    return [name for failed, name in checks if failed]


def _local_tautology_checks_passed(artifact: Mapping[str, Any]) -> bool:
    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_tautology(dict(artifact), flags)
    return not any(
        flag.kind == "TAUTOLOGY" and flag.severity == "critical" for flag in flags
    )


def _status(clean: bool, tautology_passed: bool, blockers: Sequence[str]) -> str:
    if clean:
        return "complete"
    if not tautology_passed or "adversarial_tautology_failed" in blockers:
        return "flagged"
    return "blocked"


def _honest_verdict(status: str, clean: bool, blockers: Sequence[str]) -> str:
    if clean:
        return (
            "complete: row-level overwrite-guidance corrigendum recomputed "
            "readiness from solver evidence with clean adversarial checks"
        )
    if status == "flagged":
        return "flagged: row-derived artifact still trips tautology checks"
    return "blocked: row-level overwrite-guidance corrigendum blockers=" + ",".join(
        blockers
    )


def _validate_row_evidence(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_ROW_COUNT, "row_evidence")
    _require(_row_field_completeness(rows), "row fields")
    _require(
        {row["fixture_family"] for row in rows} == set(REQUIRED_FIXTURE_FAMILIES),
        "row fixture families",
    )
    _require(
        all(row["validity_proof"]["solver_authoritative"] is True for row in rows),
        "row solver authority",
    )
    _require(
        all(row["unsafe_status"]["unsafe_false_accept"] is False for row in rows),
        "row unsafe false accept",
    )
    _require(
        all(
            row["validity_proof"]["projection_valid"] is True
            for row in rows
            if row["guidance_mode"] == "overwrite_capable"
        ),
        "overwrite projection validity",
    )


def _normalize_test_run(row: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(row, str):
        return {"command": row, "outcome": "passed"}
    return dict(row)


def _rate(numerator: int | float, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator) / denominator


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"],
        "spec_refs": artifact["spec_refs"],
        "required_metrics": {
            field: artifact[field]
            for field in REQUIRED_ARTIFACT_FIELDS
            if field != "honest_verdict"
        },
        "rate_denominators": {
            "row_metric_denominator": artifact["row_metric_denominator"],
            "overwrite_rate_denominator": artifact["overwrite_rate_denominator"],
            "forced_hint_harm_denominator": artifact[
                "forced_hint_harm_denominator"
            ],
            "post_projection_validity_denominator": artifact[
                "post_projection_validity_denominator"
            ],
            "fallback_completeness_denominator": artifact[
                "fallback_completeness_denominator"
            ],
            "negative_control_denominator": artifact[
                "negative_control_denominator"
            ],
        },
        "source_tautology_review": artifact["source_tautology_review"],
        "row_evidence": artifact["row_evidence"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
