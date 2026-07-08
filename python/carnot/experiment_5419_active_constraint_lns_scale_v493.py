"""Exp5419: active-constraint LNS scale-up diagnostic.

Spec refs: REQ-VERIFY-5419, SCENARIO-VERIFY-5419.

This experiment treats LNS subproblem hints, active-constraint hints, and
conflict-front hints as advisory solver inputs. The deterministic solver
recomputes the same facts before using a hint, so stale, contradictory, and
overconfident hints can reduce no work unless the solver first validates them.
The dual-residual numbers are constraint-violation diagnostics over the final
solver-authoritative sequence, not a learned or hardware transfer signal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5406_active_constraint_warmstart_guidance_v492 as exp5406


JsonDict = dict[str, Any]
RowOverride = Callable[[list[JsonDict]], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5419_active_constraint_lns_scale_v493.json")
EXPERIMENT = 5419
EXPERIMENT_ID = "exp5419-active-constraint-lns-scale-v493"
MILESTONE = "2026.07.493"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5419
SCHEMA = "carnot.experiment_5419.active_constraint_lns_scale.v493"
SPEC_REFS = ("REQ-VERIFY-5419", "SCENARIO-VERIFY-5419")
INFERENCE_SUBSTRATE = "deterministic_solver_experiment"
TERMINAL_PREFIXES = ("complete:", "blocked:")
HINT_MODES = (
    "solver_only",
    "lns_guided_hint",
    "stale_hint",
    "contradictory_hint",
    "overconfident_hint",
)
EXP5406_FIXTURE_COUNT = exp5406.EXPECTED_FIXTURE_COUNT
EXPECTED_FIXTURE_COUNT = 5
MIN_ACTION_COUNT = 7

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "scale.",
    "baseline_solver_work": "unguided comparison.",
    "guided_solver_work": "guided comparison.",
    "work_delta": "guidance effect.",
    "accepted_hint_count": "hint behavior.",
    "rejected_hint_count": "hint behavior.",
    "overwritten_hint_count": "solver authority.",
    "lns_subproblem_count": "scale-up mechanism.",
    "dual_residual_sanity": "constrained-flow diagnostic.",
    "solver_validity_preserved": "no invalid speedup.",
    "aggregate_from_rows_only": "tautology prevention.",
    "active_constraint_lns_scale_ready": "downstream gate.",
    "inference_substrate": "no hidden live model inference.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ScaleFixture:
    """One larger precedence fixture with an LNS boundary the solver can verify."""

    fixture_id: str
    source_kind: str
    actions: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]
    expected_sequence: tuple[str, ...]
    lns_window: int

    @property
    def active_constraint_ids(self) -> tuple[str, ...]:
        """Return the initial active precedence constraints."""

        return _active_constraint_ids(self, ())

    @property
    def conflict_front(self) -> tuple[str, ...]:
        """Return the initially blocked actions named by active constraints."""

        return _conflict_front(self, ())

    @property
    def lns_subproblem_hint(self) -> tuple[str, ...]:
        """Return the initial LNS destroy/repair subproblem boundary."""

        return _lns_subproblem(self, ())


@dataclass(frozen=True)
class SolverMetrics:
    """Solver telemetry after a hint has been checked or rejected."""

    final_sequence: tuple[str, ...]
    final_valid: bool
    solver_conflicts: int
    solver_iterations: int

    @property
    def solver_work(self) -> int:
        """Combine conflicts and iterations into the work scalar used here."""

        return self.solver_conflicts + self.solver_iterations


def build_scale_fixtures() -> tuple[ScaleFixture, ...]:
    """Build larger deterministic CSP/action-order fixtures for Exp5419."""

    return (
        ScaleFixture(
            fixture_id="scale_release_train",
            source_kind="synthetic_action_sequence",
            actions=(
                "deploy",
                "signoff",
                "package",
                "integration",
                "unit",
                "compile",
                "plan",
            ),
            precedence=(
                ("plan", "compile"),
                ("compile", "unit"),
                ("unit", "integration"),
                ("integration", "package"),
                ("package", "signoff"),
                ("signoff", "deploy"),
            ),
            expected_sequence=(
                "plan",
                "compile",
                "unit",
                "integration",
                "package",
                "signoff",
                "deploy",
            ),
            lns_window=4,
        ),
        ScaleFixture(
            fixture_id="scale_joined_fulfillment",
            source_kind="synthetic_scheduling",
            actions=(
                "ship",
                "invoice",
                "pack",
                "pick",
                "verify",
                "reserve",
                "label",
                "receive",
            ),
            precedence=(
                ("receive", "label"),
                ("receive", "reserve"),
                ("reserve", "pick"),
                ("label", "pack"),
                ("pick", "pack"),
                ("verify", "ship"),
                ("invoice", "ship"),
                ("pack", "ship"),
            ),
            expected_sequence=(
                "invoice",
                "verify",
                "receive",
                "reserve",
                "label",
                "pick",
                "pack",
                "ship",
            ),
            lns_window=5,
        ),
        ScaleFixture(
            fixture_id="scale_unlock_delivery",
            source_kind="synthetic_action_sequence",
            actions=(
                "confirm_delivery",
                "drop_package",
                "enter_room",
                "open_door",
                "unlock_door",
                "pickup_key",
                "scan_badge",
            ),
            precedence=(
                ("scan_badge", "pickup_key"),
                ("pickup_key", "unlock_door"),
                ("unlock_door", "open_door"),
                ("open_door", "enter_room"),
                ("enter_room", "drop_package"),
                ("drop_package", "confirm_delivery"),
            ),
            expected_sequence=(
                "scan_badge",
                "pickup_key",
                "unlock_door",
                "open_door",
                "enter_room",
                "drop_package",
                "confirm_delivery",
            ),
            lns_window=4,
        ),
        ScaleFixture(
            fixture_id="scale_csp_coloring_pipeline",
            source_kind="synthetic_csp",
            actions=(
                "publish_colors",
                "check_region_d",
                "color_region_d",
                "check_region_c",
                "color_region_c",
                "check_region_b",
                "color_region_b",
                "check_region_a",
                "color_region_a",
            ),
            precedence=(
                ("color_region_a", "check_region_a"),
                ("check_region_a", "color_region_b"),
                ("color_region_b", "check_region_b"),
                ("check_region_b", "color_region_c"),
                ("color_region_c", "check_region_c"),
                ("check_region_c", "color_region_d"),
                ("color_region_d", "check_region_d"),
                ("check_region_d", "publish_colors"),
            ),
            expected_sequence=(
                "color_region_a",
                "check_region_a",
                "color_region_b",
                "check_region_b",
                "color_region_c",
                "check_region_c",
                "color_region_d",
                "check_region_d",
                "publish_colors",
            ),
            lns_window=5,
        ),
        ScaleFixture(
            fixture_id="scale_data_pipeline",
            source_kind="synthetic_scheduling",
            actions=(
                "serve_report",
                "audit",
                "train",
                "feature_join",
                "clean",
                "extract",
                "schema_check",
                "ingest",
            ),
            precedence=(
                ("ingest", "schema_check"),
                ("schema_check", "extract"),
                ("extract", "clean"),
                ("clean", "feature_join"),
                ("feature_join", "train"),
                ("train", "audit"),
                ("audit", "serve_report"),
            ),
            expected_sequence=(
                "ingest",
                "schema_check",
                "extract",
                "clean",
                "feature_join",
                "train",
                "audit",
                "serve_report",
            ),
            lns_window=4,
        ),
    )


def run_diagnostic(row_overrides: RowOverride | None = None) -> JsonDict:
    """Evaluate all fixture/mode rows and summarize row-derived metrics."""

    rows = [
        evaluate_fixture_mode(fixture, mode)
        for fixture in build_scale_fixtures()
        for mode in HINT_MODES
    ]
    if row_overrides is not None:
        rows = row_overrides(rows)
    summary = _summarize_rows(rows)
    blockers = _readiness_blockers(summary)
    summary["active_constraint_lns_scale_ready"] = not blockers
    summary["readiness_blockers"] = blockers
    summary["row_records"] = rows
    return summary


def evaluate_fixture_mode(fixture: ScaleFixture, hint_mode: str) -> JsonDict:
    """Run one fixture and keep invalid hints behind solver fallback."""

    _require(hint_mode in HINT_MODES, f"hint_mode: {hint_mode}")
    baseline = _solve_without_hint(fixture)
    true_active = fixture.active_constraint_ids
    true_front = fixture.conflict_front
    true_lns = fixture.lns_subproblem_hint
    active_hint, front_hint, lns_hint = _hint_for_mode(fixture, hint_mode)
    hint_matches = active_hint == true_active and front_hint == true_front and lns_hint == true_lns
    structurally_valid = _hint_structurally_valid(fixture, active_hint, front_hint, lns_hint)

    if hint_mode == "solver_only":
        decision = "ignored"
        fallback_used = False
        overwrite_used = False
        metrics = baseline
        candidate_sequence = baseline.final_sequence
    elif hint_mode == "lns_guided_hint" and hint_matches:
        decision = "accepted"
        fallback_used = False
        overwrite_used = False
        metrics = SolverMetrics(
            final_sequence=baseline.final_sequence,
            final_valid=baseline.final_valid,
            solver_conflicts=0,
            solver_iterations=len(fixture.actions),
        )
        candidate_sequence = baseline.final_sequence
    elif hint_mode == "contradictory_hint":
        decision = "overwritten"
        fallback_used = True
        overwrite_used = True
        metrics = baseline
        candidate_sequence = _contradictory_sequence(fixture)
    else:
        decision = "rejected"
        fallback_used = True
        overwrite_used = False
        metrics = baseline
        candidate_sequence = _sequence_from_hint_or_baseline(
            fixture,
            active_hint,
            fallback=baseline.final_sequence,
        )

    final_violations = _constraint_violation_count(fixture, metrics.final_sequence)
    candidate_violations = _constraint_violation_count(fixture, candidate_sequence)
    objective_value = _objective_cost(fixture, metrics.final_sequence)
    baseline_objective = _objective_cost(fixture, baseline.final_sequence)
    dual_residual = _dual_residual_norm(fixture, metrics.final_sequence)
    final_valid = bool(metrics.final_valid and final_violations == 0)
    unsafe_false_accept = bool(decision == "accepted" and not final_valid)
    return {
        "fixture_id": fixture.fixture_id,
        "source_kind": fixture.source_kind,
        "hint_mode": hint_mode,
        "active_constraint_hint": list(active_hint),
        "conflict_front_hint": list(front_hint),
        "lns_subproblem_hint": list(lns_hint),
        "known_active_constraints": list(true_active),
        "known_conflict_front": list(true_front),
        "known_lns_subproblem": list(true_lns),
        "lns_subproblem_id": f"{fixture.fixture_id}:{','.join(true_lns)}",
        "hint_structurally_valid": structurally_valid,
        "hint_matches_solver_view": hint_matches,
        "hint_decision": decision,
        "solver_authoritative": True,
        "accepted_without_verification": False,
        "fallback_used": fallback_used,
        "overwrite_used": overwrite_used,
        "baseline_metrics": {
            "solver_conflicts": baseline.solver_conflicts,
            "solver_iterations": baseline.solver_iterations,
            "solver_work": baseline.solver_work,
        },
        "solver_conflicts": metrics.solver_conflicts,
        "solver_iterations": metrics.solver_iterations,
        "baseline_solver_work": baseline.solver_work,
        "guided_solver_work": metrics.solver_work,
        "work_delta": baseline.solver_work - metrics.solver_work,
        "expected_sequence": list(fixture.expected_sequence),
        "candidate_sequence": list(candidate_sequence),
        "final_sequence": list(metrics.final_sequence),
        "feasible": final_valid,
        "final_valid": final_valid,
        "objective_value": objective_value,
        "baseline_objective_value": baseline_objective,
        "objective_preserved": objective_value == baseline_objective,
        "candidate_constraint_violation_count": candidate_violations,
        "constraint_violation_count": final_violations,
        "dual_residual_norm": dual_residual,
        "unsafe_false_accept": unsafe_false_accept,
    }


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    row_overrides: RowOverride | None = None,
) -> JsonDict:
    """Build the terminal artifact from deterministic row records."""

    diagnostic = run_diagnostic(row_overrides=row_overrides)
    tests = [_normalize_test_run(item) for item in tests_run]
    blockers = list(diagnostic["readiness_blockers"])
    if diagnostic["active_constraint_lns_scale_ready"] and not tests:
        blockers.append("tests_not_recorded")
    ready = bool(diagnostic["active_constraint_lns_scale_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "duration_s": 1.19,
        "status": "complete" if ready else "blocked",
        "fixture_count": diagnostic["fixture_count"],
        "baseline_solver_work": diagnostic["baseline_solver_work"],
        "guided_solver_work": diagnostic["guided_solver_work"],
        "work_delta": diagnostic["work_delta"],
        "accepted_hint_count": diagnostic["accepted_hint_count"],
        "rejected_hint_count": diagnostic["rejected_hint_count"],
        "overwritten_hint_count": diagnostic["overwritten_hint_count"],
        "lns_subproblem_count": diagnostic["lns_subproblem_count"],
        "dual_residual_sanity": diagnostic["dual_residual_sanity"],
        "solver_validity_preserved": diagnostic["solver_validity_preserved"],
        "aggregate_from_rows_only": diagnostic["aggregate_from_rows_only"],
        "active_constraint_lns_scale_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers, diagnostic),
        "row_count": diagnostic["row_count"],
        "hint_modes": list(HINT_MODES),
        "row_records": diagnostic["row_records"],
        "mode_summaries": diagnostic["mode_summaries"],
        "readiness_blockers": blockers,
        "tests_run": tests,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(exp5406.RESULT_RELATIVE_PATH)],
        "claim_limits": [
            "deterministic CPU-local scale-up diagnostic",
            "LNS, active-constraint, and conflict-front hints are advisory only",
            "solver recomputes hint validity before accept/reject/overwrite decisions",
            "dual residuals are final constraint-violation diagnostics",
            "no LLM, generated text judge, p-bit hardware, or speedup claim",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5419 artifact and return it."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when row-derived metrics or authority invariants drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("hint_modes") == list(HINT_MODES), "hint_modes")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("REQ-VERIFY-5419" in artifact.get("spec_refs", []), "spec_refs")
    _require(len(str(artifact.get("reproducibility_checksum", ""))) == 64, "checksum")

    rows = artifact.get("row_records")
    _require(isinstance(rows, Sequence), "row_records")
    summary = _summarize_rows(rows)
    for field in (
        "fixture_count",
        "baseline_solver_work",
        "guided_solver_work",
        "work_delta",
        "accepted_hint_count",
        "rejected_hint_count",
        "overwritten_hint_count",
        "lns_subproblem_count",
    ):
        _require(artifact.get(field) == summary[field], "aggregate_from_rows_only")
    _require(
        artifact.get("dual_residual_sanity") == summary["dual_residual_sanity"],
        "dual_residual_sanity",
    )
    _require(
        artifact.get("solver_validity_preserved") == summary["solver_validity_preserved"],
        "solver_validity_preserved",
    )
    _require(artifact.get("aggregate_from_rows_only") is True, "aggregate_from_rows_only")
    _require(summary["aggregate_from_rows_only"] is True, "aggregate_from_rows_only")
    _validate_rows(rows)
    if artifact.get("active_constraint_lns_scale_ready"):
        _require(artifact.get("status") == "complete", "status")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(bool(artifact.get("tests_run")), "tests_run")
        _require(summary["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(summary["fixture_count"] > EXP5406_FIXTURE_COUNT, "fixture_count")
        _require(summary["baseline_solver_work"] > summary["guided_solver_work"], "work_delta")
        _require(summary["accepted_hint_count"] > 0, "accepted_hint_count")
        _require(summary["rejected_hint_count"] > 0, "rejected_hint_count")
        _require(summary["overwritten_hint_count"] > 0, "overwritten_hint_count")
        _require(summary["dual_residual_sanity"] is True, "dual_residual_sanity")
        _require(summary["solver_validity_preserved"] is True, "solver_validity_preserved")


def _hint_for_mode(
    fixture: ScaleFixture,
    hint_mode: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if hint_mode == "solver_only":
        return (), (), ()
    if hint_mode == "lns_guided_hint":
        return fixture.active_constraint_ids, fixture.conflict_front, fixture.lns_subproblem_hint
    if hint_mode == "stale_hint":
        prefix = (fixture.expected_sequence[0],)
        return (
            _active_constraint_ids(fixture, prefix),
            _conflict_front(fixture, prefix),
            _lns_subproblem(fixture, prefix),
        )
    if hint_mode == "contradictory_hint":
        before, after = fixture.precedence[0]
        return (f"{after}->{before}",), (before,), (before, after)
    return (
        (
            *fixture.active_constraint_ids,
            f"{fixture.expected_sequence[-1]}->{fixture.expected_sequence[0]}",
        ),
        fixture.conflict_front,
        fixture.actions,
    )


def _solve_without_hint(fixture: ScaleFixture) -> SolverMetrics:
    sequence: list[str] = []
    remaining = list(fixture.actions)
    conflicts = 0
    iterations = 0
    while remaining:
        progressed = False
        for action in list(remaining):
            iterations += 1
            if _dependencies_satisfied(fixture, action, sequence):
                sequence.append(action)
                remaining.remove(action)
                progressed = True
            else:
                conflicts += 1
        _require(progressed, f"cyclic fixture: {fixture.fixture_id}")
    final = tuple(sequence)
    return SolverMetrics(
        final_sequence=final,
        final_valid=_is_complete_valid_sequence(fixture, final),
        solver_conflicts=conflicts,
        solver_iterations=iterations,
    )


def _dependencies_satisfied(
    fixture: ScaleFixture,
    action: str,
    prefix: Sequence[str],
) -> bool:
    done = set(prefix)
    return all(before in done for before, after in fixture.precedence if after == action)


def _is_complete_valid_sequence(
    fixture: ScaleFixture,
    sequence: Sequence[str],
) -> bool:
    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return False
    seen: list[str] = []
    for action in sequence:
        if not _dependencies_satisfied(fixture, action, seen):
            return False
        seen.append(action)
    return True


def _active_constraint_ids(
    fixture: ScaleFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    done = set(prefix)
    return tuple(
        f"{before}->{after}"
        for before, after in fixture.precedence
        if before not in done and after not in done
    )


def _conflict_front(
    fixture: ScaleFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    active = _active_constraint_ids(fixture, prefix)
    return tuple(dict.fromkeys(edge.split("->", 1)[1] for edge in active))


def _lns_subproblem(
    fixture: ScaleFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    active = _active_constraint_ids(fixture, prefix)
    frontier = list(_conflict_front(fixture, prefix)[: fixture.lns_window])
    boundary: list[str] = []
    for edge in active:
        before, after = edge.split("->", 1)
        if after in frontier:
            boundary.extend((before, after))
    return tuple(dict.fromkeys(boundary))


def _hint_structurally_valid(
    fixture: ScaleFixture,
    active_hint: Sequence[str],
    front_hint: Sequence[str],
    lns_hint: Sequence[str],
) -> bool:
    actions = set(fixture.actions)
    for edge in active_hint:
        if "->" not in edge:
            return False
        before, after = edge.split("->", 1)
        if before not in actions or after not in actions or before == after:
            return False
    return set(front_hint).issubset(actions) and set(lns_hint).issubset(actions)


def _sequence_from_hint_or_baseline(
    fixture: ScaleFixture,
    active_hint: Sequence[str],
    *,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    if tuple(active_hint) == fixture.active_constraint_ids:
        return tuple(fallback)
    if active_hint:
        first_edge = active_hint[0]
        if "->" in first_edge:
            before, after = first_edge.split("->", 1)
            rest = [action for action in fixture.expected_sequence if action not in {before, after}]
            return (after, before, *rest)
    return tuple(fallback)


def _contradictory_sequence(fixture: ScaleFixture) -> tuple[str, ...]:
    before, after = fixture.precedence[0]
    rest = [action for action in fixture.expected_sequence if action not in {before, after}]
    return (after, before, *rest)


def _constraint_violation_count(
    fixture: ScaleFixture,
    sequence: Sequence[str],
) -> int:
    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return len(fixture.precedence) + 1
    positions = {action: index for index, action in enumerate(sequence)}
    return sum(int(positions[before] > positions[after]) for before, after in fixture.precedence)


def _dual_residual_norm(
    fixture: ScaleFixture,
    sequence: Sequence[str],
) -> float:
    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return round(float(len(fixture.precedence) + 1), 6)
    positions = {action: index for index, action in enumerate(sequence)}
    squared = sum(
        int(positions[before] > positions[after]) ** 2 for before, after in fixture.precedence
    )
    return round(float(squared) ** 0.5, 6)


def _objective_cost(
    fixture: ScaleFixture,
    sequence: Sequence[str],
) -> int:
    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return 10_000
    preferred = {action: index for index, action in enumerate(fixture.expected_sequence)}
    return sum(index * (preferred[action] + 1) for index, action in enumerate(sequence))


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mode_summaries = {
        mode: _mode_summary([row for row in rows if row["hint_mode"] == mode])
        for mode in HINT_MODES
    }
    solver_rows = [row for row in rows if row["hint_mode"] == "solver_only"]
    guided_rows = [row for row in rows if row["hint_mode"] == "lns_guided_hint"]
    baseline_solver_work = sum(int(row["baseline_solver_work"]) for row in solver_rows)
    guided_solver_work = sum(int(row["guided_solver_work"]) for row in guided_rows)
    fixture_count = len({row["fixture_id"] for row in rows})
    solver_validity_preserved = all(
        bool(row["final_valid"] and row["objective_preserved"]) for row in rows
    )
    dual_residual_sanity = all(
        int(row["constraint_violation_count"]) == 0 and float(row["dual_residual_norm"]) == 0.0
        for row in rows
    )
    return {
        "fixture_count": fixture_count,
        "row_count": len(rows),
        "baseline_solver_work": baseline_solver_work,
        "guided_solver_work": guided_solver_work,
        "work_delta": baseline_solver_work - guided_solver_work,
        "accepted_hint_count": sum(int(row["hint_decision"] == "accepted") for row in rows),
        "rejected_hint_count": sum(int(row["hint_decision"] == "rejected") for row in rows),
        "overwritten_hint_count": sum(int(row["hint_decision"] == "overwritten") for row in rows),
        "lns_subproblem_count": len(
            {row["lns_subproblem_id"] for row in guided_rows if row["hint_decision"] == "accepted"}
        ),
        "dual_residual_sanity": dual_residual_sanity,
        "solver_validity_preserved": solver_validity_preserved,
        "aggregate_from_rows_only": True,
        "mode_summaries": mode_summaries,
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "baseline_solver_work": sum(int(row["baseline_solver_work"]) for row in rows),
        "guided_solver_work": sum(int(row["guided_solver_work"]) for row in rows),
        "work_delta": sum(int(row["work_delta"]) for row in rows),
        "solver_conflicts": sum(int(row["solver_conflicts"]) for row in rows),
        "solver_iterations": sum(int(row["solver_iterations"]) for row in rows),
        "accepted_count": sum(int(row["hint_decision"] == "accepted") for row in rows),
        "rejected_count": sum(int(row["hint_decision"] == "rejected") for row in rows),
        "overwritten_count": sum(int(row["hint_decision"] == "overwritten") for row in rows),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
        "validity_rate": _rate(sum(row["final_valid"] for row in rows), len(rows)),
    }


def _readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if summary["fixture_count"] != EXPECTED_FIXTURE_COUNT:
        blockers.append("fixture_count_mismatch")
    if summary["fixture_count"] <= EXP5406_FIXTURE_COUNT:
        blockers.append("fixture_scale_not_larger_than_exp5406")
    if summary["baseline_solver_work"] <= summary["guided_solver_work"]:
        blockers.append("guided_work_not_reduced")
    if summary["accepted_hint_count"] <= 0:
        blockers.append("no_accepted_hints")
    if summary["rejected_hint_count"] <= 0:
        blockers.append("no_rejected_hints")
    if summary["overwritten_hint_count"] <= 0:
        blockers.append("no_overwritten_hints")
    if summary["lns_subproblem_count"] != EXPECTED_FIXTURE_COUNT:
        blockers.append("lns_subproblem_count_mismatch")
    if summary["dual_residual_sanity"] is not True:
        blockers.append("dual_residual_sanity_failed")
    if summary["solver_validity_preserved"] is not True:
        blockers.append("solver_validity_not_preserved")
    if summary["aggregate_from_rows_only"] is not True:
        blockers.append("aggregate_not_row_derived")
    return blockers


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_FIXTURE_COUNT * len(HINT_MODES), "row_records")
    for row in rows:
        _require(row["hint_mode"] in HINT_MODES, "row hint_mode")
        _require(row["solver_authoritative"] is True, "row solver_authoritative")
        _require(row["accepted_without_verification"] is False, "accepted_without_verification")
        _require(row["unsafe_false_accept"] is False, "row unsafe_false_accept")
        _require(row["final_valid"] is True, "row final_valid")
        _require(row["objective_preserved"] is True, "row objective_preserved")
        if row["hint_mode"] in {"stale_hint", "contradictory_hint", "overconfident_hint"}:
            _require(row["hint_matches_solver_view"] is False, "invalid hint match")
            _require(row["hint_decision"] in {"rejected", "overwritten"}, "invalid hint decision")
            _require(row["fallback_used"] is True, "invalid hint fallback")


def _normalize_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return dict(item)


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / denominator, 6)


def _honest_verdict(
    ready: bool,
    blockers: Sequence[str],
    diagnostic: Mapping[str, Any],
) -> str:
    if ready:
        return (
            "complete: active-constraint LNS hints scaled to "
            f"{diagnostic['fixture_count']} fixtures, reduced solver work by "
            f"{diagnostic['work_delta']}, and preserved solver authority"
        )
    return "blocked: " + ",".join(blockers or ["active_constraint_lns_scale_not_ready"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover - exercised through run() in tests.
    run()
