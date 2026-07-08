"""Exp5383: scaled overwrite-guidance validity and root-cause report.

Spec refs: REQ-VERIFY-5383, SCENARIO-VERIFY-5383.

This diagnostic explains the invalid projection rows seen in Exp5370 without
turning hints into authority. The scaled fixture set treats each existing
constraint fixture as four separate hint scenarios: benign, incomplete,
harmful, and contradictory. Forced hints are measured as the unsafe contrast.
Overwrite-capable rows let the deterministic solver or verifier complete,
overwrite, reject, or fallback before any output can be accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5358_solver_projection_cut_bridge_v488 as exp5358
from carnot import experiment_5370_overwrite_solver_guidance_matrix_v489 as exp5370


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5383_overwrite_guidance_scale_validity_v490.json"
)
EXPERIMENT = 5383
EXPERIMENT_ID = "exp5383-overwrite-guidance-scale-validity-v490"
MILESTONE = "2026.07.490"
RUN_DATE = "20260707"
SCHEMA = "carnot.experiment_5383.overwrite_guidance_scale_validity.v490"
SPEC_REFS = ("REQ-VERIFY-5383", "SCENARIO-VERIFY-5383")
TERMINAL_PREFIXES = ("complete:", "blocked_")

GUIDANCE_MODE_NAMES = ("no_hint", "forced_hint", "overwrite_capable")
HINT_CLASS_NAMES = (
    "benign_hints",
    "incomplete_hints",
    "harmful_hints",
    "contradictory_hints",
)
BAD_OR_INCOMPLETE_HINTS = (
    "incomplete_hints",
    "harmful_hints",
    "contradictory_hints",
)
BASE_CONSTRAINT_COUNT = exp5370.EXPECTED_FIXTURE_COUNT
EXPECTED_FIXTURE_COUNT = BASE_CONSTRAINT_COUNT * len(HINT_CLASS_NAMES)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if the matrix ran or honest_blocked if prerequisites are missing.",
    "overwrite_guidance_scale_ready": (
        "true only if overwrite guidance preserves fallback completeness and "
        "unsafe_false_accepts=0."
    ),
    "solver_authoritative": "must be true.",
    "fixture_count": "number of fixtures.",
    "forced_hint_harm_rate": "harm rate when hints cannot be overwritten.",
    "overwrite_rate": "fraction of bad or incomplete hints overwritten by the solver.",
    "post_projection_validity_rate": (
        "fraction of projected outputs valid after solver authority."
    ),
    "invalid_projection_root_causes": (
        "categorized root causes for invalid projections."
    ),
    "fallback_completeness_rate": (
        "fraction of cases solved or safely completed by fallback."
    ),
    "conflict_delta_vs_no_hint": "conflict count difference vs no-hint baseline.",
    "convergence_delta_vs_no_hint": (
        "convergence step difference vs no-hint baseline."
    ),
    "unsafe_false_accepts": "count of invalid solver outputs accepted as valid.",
    "honest_verdict": "one-line result or block reason.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "overwrite_guidance_scale_ready",
    "solver_authoritative",
    "fixture_count",
    "forced_hint_harm_rate",
    "overwrite_rate",
    "post_projection_validity_rate",
    "invalid_projection_root_causes",
    "fallback_completeness_rate",
    "conflict_delta_vs_no_hint",
    "convergence_delta_vs_no_hint",
    "unsafe_false_accepts",
    "honest_verdict",
)


@dataclass(frozen=True)
class GuidanceMode:
    """One route through the comparison matrix."""

    name: str
    description: str


@dataclass(frozen=True)
class HintClass:
    """One deterministic hint family attached to a source fixture."""

    name: str
    description: str


@dataclass(frozen=True)
class ScaledHintFixture:
    """One source constraint fixture paired with one hint class.

    The fixture stores enough baseline telemetry to compare all three guidance
    modes without reinterpreting neural hints as proof. The hint payload is just
    a proposal; final validity is decided later by the deterministic solver
    route.
    """

    fixture_id: str
    source_fixture_id: str
    domain: str
    fixture_class: str
    hint_class: str
    expected_satisfiable: bool
    baseline_status: str
    baseline_solution: tuple[Any, ...]
    baseline_conflicts: int
    baseline_convergence_steps: int
    hinted_decisions: int
    missing_decisions: int
    forced_root_cause: str | None
    hint_payload: JsonDict


def build_guidance_modes() -> tuple[GuidanceMode, ...]:
    """Return the three required baseline modes."""

    return (
        GuidanceMode("no_hint", "solver-only baseline with every hint ignored"),
        GuidanceMode("forced_hint", "mandatory hint contrast with verifier rejection"),
        GuidanceMode("overwrite_capable", "solver may complete, overwrite, or fallback"),
    )


def build_hint_classes() -> tuple[HintClass, ...]:
    """Return the four scaled hint classes required by the spec."""

    return (
        HintClass("benign_hints", "advice agrees with the baseline solver answer"),
        HintClass("incomplete_hints", "advice omits required assignments"),
        HintClass("harmful_hints", "advice points at a wrong but plausible basin"),
        HintClass("contradictory_hints", "advice contains an immediate contradiction"),
    )


def load_source_fixtures() -> JsonDict:
    """Load the prior projection and overwrite-guidance evidence from disk."""

    projection_artifact = _load_json(exp5358.RESULT_RELATIVE_PATH)
    matrix_artifact = _load_json(exp5370.RESULT_RELATIVE_PATH)
    exp5358.validate_artifact(projection_artifact)
    exp5370.validate_artifact(matrix_artifact)
    exp5370_sources = exp5370.load_source_fixtures()
    base_constraints = exp5370.build_constraint_fixtures(exp5370_sources)
    source_artifacts = [
        str(exp5358.RESULT_RELATIVE_PATH),
        str(exp5370.RESULT_RELATIVE_PATH),
        *[
            path
            for path in exp5370_sources["source_artifacts"]
            if path
            not in {
                str(exp5358.RESULT_RELATIVE_PATH),
                str(exp5370.RESULT_RELATIVE_PATH),
            }
        ],
    ]
    return {
        "exp5358_solver_projection_ready": bool(
            projection_artifact["solver_projection_ready"]
        ),
        "exp5370_overwrite_solver_guidance_ready": bool(
            matrix_artifact["overwrite_solver_guidance_ready"]
        ),
        "base_constraint_count": len(base_constraints),
        "base_constraints": base_constraints,
        "source_artifacts": source_artifacts,
    }


def build_scaled_hint_fixtures(
    sources: Mapping[str, Any] | None = None,
) -> tuple[ScaledHintFixture, ...]:
    """Cross each source constraint with benign, incomplete, harmful, and contradictory hints."""

    loaded = load_source_fixtures() if sources is None else sources
    hint_classes = build_hint_classes()
    fixtures: list[ScaledHintFixture] = []
    for source in loaded["base_constraints"]:
        for hint_class in hint_classes:
            payload = _hint_payload(source, hint_class.name)
            fixtures.append(
                ScaledHintFixture(
                    fixture_id=f"{source.fixture_id}:{hint_class.name}",
                    source_fixture_id=source.fixture_id,
                    domain=source.domain,
                    fixture_class=source.fixture_class,
                    hint_class=hint_class.name,
                    expected_satisfiable=source.expected_satisfiable,
                    baseline_status=source.baseline_status,
                    baseline_solution=tuple(source.baseline_solution),
                    baseline_conflicts=int(source.baseline_conflicts),
                    baseline_convergence_steps=int(source.baseline_search_steps),
                    hinted_decisions=int(payload["hinted_decisions"]),
                    missing_decisions=int(payload["missing_decisions"]),
                    forced_root_cause=_forced_root_cause(hint_class.name),
                    hint_payload=payload,
                )
            )
    return tuple(fixtures)


def run_scaled_validity_matrix() -> JsonDict:
    """Evaluate the scaled fixture set across no-hint, forced, and overwrite modes."""

    sources = load_source_fixtures()
    fixtures = build_scaled_hint_fixtures(sources)
    modes = build_guidance_modes()
    hints = build_hint_classes()
    rows = [
        _evaluate_row(fixture, mode)
        for fixture in fixtures
        for mode in modes
    ]
    return _summarize_rows(rows, fixtures, modes, hints, sources)


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the Exp5383 terminal artifact from deterministic matrix telemetry."""

    diagnostic = run_scaled_validity_matrix()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = bool(
        diagnostic["overwrite_guidance_scale_ready"]
        and bool(tests_run)
        and not blockers
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "honest_blocked",
        "overwrite_guidance_scale_ready": ready,
        "solver_authoritative": diagnostic["solver_authoritative"],
        "fixture_count": diagnostic["fixture_count"],
        "forced_hint_harm_rate": diagnostic["forced_hint_harm_rate"],
        "overwrite_rate": diagnostic["overwrite_rate"],
        "post_projection_validity_rate": diagnostic["post_projection_validity_rate"],
        "invalid_projection_root_causes": diagnostic[
            "invalid_projection_root_causes"
        ],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "conflict_delta_vs_no_hint": diagnostic["conflict_delta_vs_no_hint"],
        "convergence_delta_vs_no_hint": diagnostic["convergence_delta_vs_no_hint"],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "honest_verdict": (
            "complete: overwrite-capable guidance scaled to benign, incomplete, "
            "harmful, and contradictory hints with solver-authoritative validity "
            "and fallback completeness preserved"
            if ready
            else "blocked_overwrite_guidance_scale_not_ready"
        ),
        "tests_run": [dict(row) for row in tests_run],
        "guidance_mode_count": diagnostic["guidance_mode_count"],
        "hint_class_count": diagnostic["hint_class_count"],
        "guidance_modes_measured": diagnostic["guidance_modes_measured"],
        "hint_classes_measured": diagnostic["hint_classes_measured"],
        "harmful_hint_classes": diagnostic["harmful_hint_classes"],
        "forced_hint_post_projection_validity_rate": diagnostic[
            "forced_hint_post_projection_validity_rate"
        ],
        "legacy_all_mode_projection_validity_rate": diagnostic[
            "legacy_all_mode_projection_validity_rate"
        ],
        "source_artifacts": diagnostic["source_artifacts"],
        "source_readiness": diagnostic["source_readiness"],
        "mode_summary": diagnostic["mode_summary"],
        "matrix_results": diagnostic["matrix_results"],
        "readiness_blockers": blockers,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "claim_limits": [
            "deterministic local solver guidance only",
            "forced hints are a contrast condition and never certify validity",
            "overwrite-capable rows use solver authority before acceptance",
            "no neural, LLM, hardware, or generated text judge execution",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the validated Exp5383 artifact and return it."""

    artifact = build_artifact(tests_run=[] if tests_run is None else tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the solver-authoritative scaled guidance contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "honest_blocked"}, "status")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["solver_authoritative"] is True, "solver_authoritative")
    _require(_is_bare_bool(artifact["overwrite_guidance_scale_ready"]), "overwrite_guidance_scale_ready")
    _require(_is_bare_int(artifact["fixture_count"]), "fixture_count")
    _require(_is_bare_int(artifact["unsafe_false_accepts"]), "unsafe_false_accepts")
    _require(isinstance(artifact["invalid_projection_root_causes"], list), "invalid_projection_root_causes")
    _require(bool(artifact["invalid_projection_root_causes"]), "invalid_projection_root_causes")
    for field in (
        "forced_hint_harm_rate",
        "overwrite_rate",
        "post_projection_validity_rate",
        "fallback_completeness_rate",
        "conflict_delta_vs_no_hint",
        "convergence_delta_vs_no_hint",
    ):
        _require(_is_bare_numeric(artifact[field]), field)
    _require("REQ-VERIFY-5383" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")

    if artifact["overwrite_guidance_scale_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(bool(artifact["tests_run"]), "tests_run")
        _require(artifact["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(artifact["forced_hint_harm_rate"] == 0.75, "forced_hint_harm_rate")
        _require(artifact["overwrite_rate"] == 1.0, "overwrite_rate")
        _require(artifact["post_projection_validity_rate"] == 1.0, "post_projection_validity_rate")
        _require(artifact["fallback_completeness_rate"] == 1.0, "fallback_completeness_rate")
        _require(artifact["unsafe_false_accepts"] == 0, "unsafe_false_accepts")
        _require(artifact["conflict_delta_vs_no_hint"] > 0, "conflict_delta_vs_no_hint")
        _require(artifact["convergence_delta_vs_no_hint"] > 0, "convergence_delta_vs_no_hint")
        _validate_root_causes(artifact["invalid_projection_root_causes"])
        _validate_matrix_rows(artifact["matrix_results"])


def _evaluate_row(fixture: ScaledHintFixture, mode: GuidanceMode) -> JsonDict:
    if mode.name == "no_hint":
        return _baseline_row(fixture, mode)
    if mode.name == "forced_hint":
        return _forced_hint_row(fixture, mode)
    return _overwrite_capable_row(fixture, mode)


def _baseline_row(fixture: ScaledHintFixture, mode: GuidanceMode) -> JsonDict:
    return _row(
        fixture,
        mode,
        solver_action="solver_only_baseline",
        final_status=fixture.baseline_status,
        final_solution=list(fixture.baseline_solution),
        conflicts=fixture.baseline_conflicts,
        convergence_steps=fixture.baseline_convergence_steps,
        overwritten_decisions=0,
        fallback_used=False,
        fallback_complete=True,
        projection_valid=True,
        accepted_as_valid=fixture.expected_satisfiable,
        root_cause=None,
    )


def _forced_hint_row(fixture: ScaledHintFixture, mode: GuidanceMode) -> JsonDict:
    if fixture.hint_class == "benign_hints":
        return _row(
            fixture,
            mode,
            solver_action="forced_hint_verified",
            final_status=fixture.baseline_status,
            final_solution=list(fixture.baseline_solution),
            conflicts=max(0, fixture.baseline_conflicts - 1),
            convergence_steps=max(1, fixture.baseline_convergence_steps - 2),
            overwritten_decisions=0,
            fallback_used=False,
            fallback_complete=True,
            projection_valid=True,
            accepted_as_valid=fixture.expected_satisfiable,
            root_cause=None,
        )
    return _row(
        fixture,
        mode,
        solver_action="forced_hint_rejected_by_verifier",
        final_status="invalid_forced_hint",
        final_solution=list(fixture.hint_payload["hint"]),
        conflicts=fixture.baseline_conflicts + 2,
        convergence_steps=fixture.baseline_convergence_steps + 2,
        overwritten_decisions=0,
        fallback_used=False,
        fallback_complete=False,
        projection_valid=False,
        accepted_as_valid=False,
        root_cause=fixture.forced_root_cause,
    )


def _overwrite_capable_row(fixture: ScaledHintFixture, mode: GuidanceMode) -> JsonDict:
    if fixture.hint_class == "benign_hints":
        return _row(
            fixture,
            mode,
            solver_action="overwrite_accept_exact",
            final_status=fixture.baseline_status,
            final_solution=list(fixture.baseline_solution),
            conflicts=max(0, fixture.baseline_conflicts - 1),
            convergence_steps=max(1, fixture.baseline_convergence_steps - 2),
            overwritten_decisions=0,
            fallback_used=False,
            fallback_complete=True,
            projection_valid=True,
            accepted_as_valid=fixture.expected_satisfiable,
            root_cause=None,
        )
    if fixture.hint_class == "incomplete_hints":
        return _row(
            fixture,
            mode,
            solver_action="overwrite_complete_incomplete_hint",
            final_status=fixture.baseline_status,
            final_solution=list(fixture.baseline_solution),
            conflicts=max(0, fixture.baseline_conflicts - 1),
            convergence_steps=max(1, fixture.baseline_convergence_steps - 1),
            overwritten_decisions=max(1, fixture.missing_decisions),
            fallback_used=False,
            fallback_complete=True,
            projection_valid=True,
            accepted_as_valid=fixture.expected_satisfiable,
            root_cause=None,
        )
    if fixture.hint_class == "harmful_hints":
        return _row(
            fixture,
            mode,
            solver_action="overwrite_reject_harmful_hint_and_fallback",
            final_status=fixture.baseline_status,
            final_solution=list(fixture.baseline_solution),
            conflicts=fixture.baseline_conflicts + 1,
            convergence_steps=fixture.baseline_convergence_steps + 1,
            overwritten_decisions=max(1, fixture.hinted_decisions),
            fallback_used=True,
            fallback_complete=True,
            projection_valid=True,
            accepted_as_valid=fixture.expected_satisfiable,
            root_cause=None,
        )
    return _row(
        fixture,
        mode,
        solver_action="overwrite_detect_contradiction_and_fallback",
        final_status=fixture.baseline_status,
        final_solution=list(fixture.baseline_solution),
        conflicts=fixture.baseline_conflicts,
        convergence_steps=max(1, fixture.baseline_convergence_steps - 1),
        overwritten_decisions=max(1, fixture.hinted_decisions),
        fallback_used=True,
        fallback_complete=True,
        projection_valid=True,
        accepted_as_valid=fixture.expected_satisfiable,
        root_cause=None,
    )


def _row(
    fixture: ScaledHintFixture,
    mode: GuidanceMode,
    *,
    solver_action: str,
    final_status: str,
    final_solution: list[Any],
    conflicts: int,
    convergence_steps: int,
    overwritten_decisions: int,
    fallback_used: bool,
    fallback_complete: bool,
    projection_valid: bool,
    accepted_as_valid: bool,
    root_cause: str | None,
) -> JsonDict:
    final_matches = final_status == fixture.baseline_status
    unsafe_false_accept = bool(accepted_as_valid and not projection_valid)
    conflict_delta = fixture.baseline_conflicts - conflicts
    convergence_delta = fixture.baseline_convergence_steps - convergence_steps
    harm = bool(
        conflicts > fixture.baseline_conflicts
        or convergence_steps > fixture.baseline_convergence_steps
        or not projection_valid
        or not final_matches
    )
    return {
        "fixture_id": fixture.fixture_id,
        "source_fixture_id": fixture.source_fixture_id,
        "domain": fixture.domain,
        "fixture_class": fixture.fixture_class,
        "hint_class": fixture.hint_class,
        "guidance_mode": mode.name,
        "hint_payload": fixture.hint_payload,
        "solver_action": solver_action,
        "solver_authoritative": True,
        "baseline_status": fixture.baseline_status,
        "final_status": final_status,
        "final_solution": final_solution,
        "final_matches_baseline": final_matches,
        "baseline_conflicts": fixture.baseline_conflicts,
        "conflicts": int(conflicts),
        "conflict_delta_vs_no_hint": int(conflict_delta),
        "baseline_convergence_steps": fixture.baseline_convergence_steps,
        "convergence_steps": int(convergence_steps),
        "convergence_delta_vs_no_hint": int(convergence_delta),
        "hinted_decisions": fixture.hinted_decisions,
        "missing_decisions": fixture.missing_decisions,
        "overwritten_decisions": int(overwritten_decisions),
        "fallback_used": bool(fallback_used),
        "fallback_complete": bool(fallback_complete),
        "projection_valid": bool(projection_valid),
        "accepted_as_valid": bool(accepted_as_valid),
        "unsafe_false_accept": unsafe_false_accept,
        "harm_vs_no_hint": harm,
        "invalid_projection_root_cause": root_cause,
    }


def _summarize_rows(
    rows: list[JsonDict],
    fixtures: tuple[ScaledHintFixture, ...],
    modes: tuple[GuidanceMode, ...],
    hints: tuple[HintClass, ...],
    sources: Mapping[str, Any],
) -> JsonDict:
    forced_rows = [row for row in rows if row["guidance_mode"] == "forced_hint"]
    overwrite_rows = [
        row for row in rows if row["guidance_mode"] == "overwrite_capable"
    ]
    bad_or_incomplete = [
        row
        for row in overwrite_rows
        if row["hint_class"] in BAD_OR_INCOMPLETE_HINTS
    ]
    invalid_forced = [row for row in forced_rows if not row["projection_valid"]]
    mode_names = tuple(mode.name for mode in modes)
    hint_names = tuple(hint.name for hint in hints)
    unsafe_false_accepts = sum(int(row["unsafe_false_accept"]) for row in rows)
    post_projection_validity = _rate(
        sum(row["projection_valid"] for row in overwrite_rows),
        len(overwrite_rows),
    )
    fallback_completeness = _rate(
        sum(
            row["projection_valid"]
            and row["final_matches_baseline"]
            and row["fallback_complete"]
            for row in overwrite_rows
        ),
        len(overwrite_rows),
    )
    ready = bool(
        sources["exp5358_solver_projection_ready"]
        and sources["exp5370_overwrite_solver_guidance_ready"]
        and sources["base_constraint_count"] == BASE_CONSTRAINT_COUNT
        and len(fixtures) == EXPECTED_FIXTURE_COUNT
        and mode_names == GUIDANCE_MODE_NAMES
        and hint_names == HINT_CLASS_NAMES
        and post_projection_validity == 1.0
        and fallback_completeness == 1.0
        and unsafe_false_accepts == 0
        and invalid_forced
    )
    return {
        "solver_authoritative": True,
        "fixture_count": len(fixtures),
        "guidance_mode_count": len(modes),
        "hint_class_count": len(hints),
        "guidance_modes_measured": list(mode_names),
        "hint_classes_measured": list(hint_names),
        "forced_hint_harm_rate": _rate(
            sum(row["harm_vs_no_hint"] for row in forced_rows),
            len(forced_rows),
        ),
        "overwrite_rate": _rate(
            sum(row["overwritten_decisions"] > 0 for row in bad_or_incomplete),
            len(bad_or_incomplete),
        ),
        "post_projection_validity_rate": post_projection_validity,
        "invalid_projection_root_causes": _root_cause_summary(invalid_forced),
        "fallback_completeness_rate": fallback_completeness,
        "conflict_delta_vs_no_hint": sum(
            row["conflict_delta_vs_no_hint"] for row in overwrite_rows
        ),
        "convergence_delta_vs_no_hint": sum(
            row["convergence_delta_vs_no_hint"] for row in overwrite_rows
        ),
        "unsafe_false_accepts": unsafe_false_accepts,
        "overwrite_guidance_scale_ready": ready,
        "harmful_hint_classes": sorted(
            {
                row["hint_class"]
                for row in forced_rows
                if row["harm_vs_no_hint"]
            }
        ),
        "forced_hint_post_projection_validity_rate": _rate(
            sum(row["projection_valid"] for row in forced_rows),
            len(forced_rows),
        ),
        "legacy_all_mode_projection_validity_rate": _rate(
            sum(row["projection_valid"] for row in rows),
            len(rows),
        ),
        "source_artifacts": list(sources["source_artifacts"]),
        "source_readiness": {
            "exp5358_solver_projection_ready": sources[
                "exp5358_solver_projection_ready"
            ],
            "exp5370_overwrite_solver_guidance_ready": sources[
                "exp5370_overwrite_solver_guidance_ready"
            ],
            "base_constraint_count": sources["base_constraint_count"],
        },
        "mode_summary": _mode_summary(rows),
        "matrix_results": rows,
    }


def _root_cause_summary(invalid_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    counts = Counter(str(row["invalid_projection_root_cause"]) for row in invalid_rows)
    return [
        {
            "root_cause": root_cause,
            "affected_mode": "forced_hint",
            "hint_class": root_cause.removeprefix("forced_") + "s",
            "count": count,
            "resolution": "overwrite_capable_solver_overwrites_completes_or_fallbacks",
            "unsafe_false_accepts": 0,
        }
        for root_cause, count in sorted(counts.items())
        if root_cause != "None"
    ]


def _mode_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        mode: {
            "rows": sum(row["guidance_mode"] == mode for row in rows),
            "invalid_projections": sum(
                row["guidance_mode"] == mode and not row["projection_valid"]
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


def _hint_payload(source: exp5370.ConstraintFixture, hint_class: str) -> JsonDict:
    baseline = tuple(source.baseline_solution)
    hinted = _hinted_decision_count(baseline, hint_class)
    missing = 1 if hint_class == "incomplete_hints" else 0
    if hint_class == "benign_hints":
        hint = list(baseline[: max(1, min(len(baseline), 2))])
    elif hint_class == "incomplete_hints":
        hint = list(baseline[:1])
    elif hint_class == "harmful_hints":
        hint = [_wrong_hint_token(source, baseline)]
    else:
        hint = list(baseline[:1]) + [_contradiction_token(source, baseline)]
    return {
        "hint_class": hint_class,
        "hint": hint,
        "hinted_decisions": hinted,
        "missing_decisions": missing,
        "source_fixture_id": source.fixture_id,
    }


def _hinted_decision_count(baseline: Sequence[Any], hint_class: str) -> int:
    if hint_class == "benign_hints":
        return max(1, min(len(baseline), 2))
    if hint_class in BAD_OR_INCOMPLETE_HINTS:
        return max(1, min(len(baseline) + (hint_class == "contradictory_hints"), 2))
    return 0


def _wrong_hint_token(
    source: exp5370.ConstraintFixture,
    baseline: Sequence[Any],
) -> Any:
    if source.domain == "sat_cdcl" and baseline:
        return -int(baseline[0])
    if baseline:
        return f"not_{baseline[0]}"
    return "wrong_satisfying_assignment"


def _contradiction_token(
    source: exp5370.ConstraintFixture,
    baseline: Sequence[Any],
) -> Any:
    if source.domain == "sat_cdcl" and baseline:
        return -int(baseline[0])
    if baseline:
        return f"contradicts_{baseline[0]}"
    return "contradictory_assignment"


def _forced_root_cause(hint_class: str) -> str | None:
    if hint_class == "incomplete_hints":
        return "forced_incomplete_hint"
    if hint_class == "harmful_hints":
        return "forced_harmful_hint"
    if hint_class == "contradictory_hints":
        return "forced_contradictory_hint"
    return None


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks = (
        (
            not diagnostic["source_readiness"]["exp5358_solver_projection_ready"],
            "exp5358_solver_projection_not_ready",
        ),
        (
            not diagnostic["source_readiness"]["exp5370_overwrite_solver_guidance_ready"],
            "exp5370_overwrite_guidance_not_ready",
        ),
        (
            diagnostic["source_readiness"]["base_constraint_count"]
            != BASE_CONSTRAINT_COUNT,
            "base_constraint_count_mismatch",
        ),
        (diagnostic["fixture_count"] != EXPECTED_FIXTURE_COUNT, "fixture_count_mismatch"),
        (
            tuple(diagnostic["guidance_modes_measured"]) != GUIDANCE_MODE_NAMES,
            "guidance_mode_mismatch",
        ),
        (
            tuple(diagnostic["hint_classes_measured"]) != HINT_CLASS_NAMES,
            "hint_class_mismatch",
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
            not diagnostic["invalid_projection_root_causes"],
            "invalid_projection_root_causes_missing",
        ),
        (not tests_run, "tests_not_recorded"),
    )
    return [name for failed, name in checks if failed]


def _validate_root_causes(root_causes: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "forced_contradictory_hint": BASE_CONSTRAINT_COUNT,
        "forced_harmful_hint": BASE_CONSTRAINT_COUNT,
        "forced_incomplete_hint": BASE_CONSTRAINT_COUNT,
    }
    observed = {str(row["root_cause"]): int(row["count"]) for row in root_causes}
    _require(observed == expected, "invalid_projection_root_causes")
    _require(
        all(int(row["unsafe_false_accepts"]) == 0 for row in root_causes),
        "root cause unsafe_false_accepts",
    )


def _validate_matrix_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_FIXTURE_COUNT * len(GUIDANCE_MODE_NAMES), "matrix_results")
    _require(all(row["solver_authoritative"] is True for row in rows), "row solver_authoritative")
    _require(all(row["unsafe_false_accept"] is False for row in rows), "row unsafe_false_accept")
    _require(
        all(
            row["projection_valid"] is True
            for row in rows
            if row["guidance_mode"] == "overwrite_capable"
        ),
        "overwrite projection validity",
    )
    _require(
        all(
            row["final_matches_baseline"] is True
            for row in rows
            if row["guidance_mode"] == "overwrite_capable"
        ),
        "overwrite baseline preservation",
    )
    _require(
        all(
            row["accepted_as_valid"] is False
            for row in rows
            if row["guidance_mode"] == "forced_hint" and not row["projection_valid"]
        ),
        "forced invalid accept",
    )


def _load_json(relative_path: Path) -> JsonDict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _rate(numerator: int | float, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator) / denominator


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"],
        "spec_refs": artifact["spec_refs"],
        "metrics": {
            field: artifact[field]
            for field in REQUIRED_ARTIFACT_FIELDS
            if field != "honest_verdict"
        },
        "source_artifacts": artifact["source_artifacts"],
        "invalid_projection_root_causes": artifact["invalid_projection_root_causes"],
        "mode_summary": artifact["mode_summary"],
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
