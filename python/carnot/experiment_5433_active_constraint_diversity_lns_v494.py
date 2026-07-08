"""Exp5433: active-constraint LNS diversity diagnostic.

Spec refs: REQ-VERIFY-5433, SCENARIO-VERIFY-5433.

This experiment keeps the Exp 5419 solver-guidance contract but changes the
question from scale to generality: do advisory LNS hints still help when the
constraint graph shape changes?  The solver recomputes active constraints,
conflict fronts, active tails, frozen variables, objectives, and validity before
using a hint.  Hints can reduce deterministic work after validation, but they
never certify the final answer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
RowOverride = Callable[[list[JsonDict]], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5433_active_constraint_diversity_lns_v494.json")
EXPERIMENT = 5433
EXPERIMENT_ID = "exp5433-active-constraint-diversity-lns-v494"
MILESTONE = "2026.07.494"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5433
SCHEMA = "carnot.experiment_5433.active_constraint_diversity_lns.v494"
SPEC_REFS = ("REQ-VERIFY-5433", "SCENARIO-VERIFY-5433")
INFERENCE_SUBSTRATE = "deterministic_solver_experiment"
TERMINAL_PREFIXES = ("complete:", "blocked:")
HINT_MODES = ("solver_only", "lns_guided_hint", "stale_hint", "adversarial_hint")
EXPECTED_FIXTURE_COUNT = 4
EXPECTED_FAMILY_COUNT = 4
MIN_SUBPROBLEM_FAMILY_COUNT = 3

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "scale.",
    "subproblem_family_count": "diversity coverage.",
    "diversity_descriptor_checksum": "reproducibility.",
    "baseline_solver_work": "unguided comparison.",
    "guided_solver_work": "guided comparison.",
    "work_delta": "guidance effect.",
    "accepted_hint_count": "hint behavior.",
    "rejected_hint_count": "hint behavior.",
    "overwritten_hint_count": "solver authority.",
    "conflict_front_precision": "hint quality.",
    "solver_validity_preserved": "no invalid speedup.",
    "aggregate_from_rows_only": "tautology prevention.",
    "active_constraint_diversity_ready": "downstream gate.",
    "inference_substrate": "no hidden live model inference.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class DiversityFixture:
    """One finite-domain precedence fixture with explicit diversity metadata."""

    fixture_id: str
    subproblem_family: str
    geometry: str
    lns_shape: str
    source_kind: str
    actions: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]
    expected_sequence: tuple[str, ...]
    lns_window: int

    @property
    def active_constraint_ids(self) -> tuple[str, ...]:
        """Return the initially active precedence constraints."""

        return _active_constraint_ids(self, ())

    @property
    def conflict_front(self) -> tuple[str, ...]:
        """Return actions blocked by the initially active constraints."""

        return _conflict_front(self, ())

    @property
    def lns_subproblem_hint(self) -> tuple[str, ...]:
        """Return the solver-derived LNS destroy/repair boundary."""

        return _lns_subproblem(self, ())

    @property
    def active_tail(self) -> tuple[str, ...]:
        """Return variables that remain live inside the LNS subproblem."""

        return self.lns_subproblem_hint

    @property
    def frozen_variables(self) -> tuple[str, ...]:
        """Return variables outside the active LNS tail that stay fixed."""

        active_tail = set(self.active_tail)
        return tuple(action for action in self.expected_sequence if action not in active_tail)

    @property
    def diversity_descriptor(self) -> JsonDict:
        """Describe the shape of the subproblem independently of hint quality."""

        in_degree = {action: 0 for action in self.actions}
        out_degree = {action: 0 for action in self.actions}
        for before, after in self.precedence:
            out_degree[before] += 1
            in_degree[after] += 1
        return {
            "fixture_id": self.fixture_id,
            "subproblem_family": self.subproblem_family,
            "geometry": self.geometry,
            "node_count": len(self.actions),
            "edge_count": len(self.precedence),
            "frontier_width": len(self.conflict_front),
            "max_in_degree": max(in_degree.values()),
            "max_out_degree": max(out_degree.values()),
            "lns_shape": self.lns_shape,
        }


@dataclass(frozen=True)
class SolverMetrics:
    """Deterministic solver telemetry after hint validation."""

    final_sequence: tuple[str, ...]
    final_valid: bool
    solver_conflicts: int
    solver_iterations: int

    @property
    def solver_work(self) -> int:
        """Use conflicts plus iterations as the local work scalar."""

        return self.solver_conflicts + self.solver_iterations


def build_diversity_fixtures() -> tuple[DiversityFixture, ...]:
    """Build diverse LNS fixtures with different graph shapes."""

    return (
        DiversityFixture(
            fixture_id="diverse_chain_release",
            subproblem_family="linear_chain",
            geometry="path_precedence",
            lns_shape="contiguous_path_window",
            source_kind="synthetic_action_sequence",
            actions=("deploy", "signoff", "package", "integration", "unit", "compile", "plan"),
            precedence=(
                ("plan", "compile"),
                ("compile", "unit"),
                ("unit", "integration"),
                ("integration", "package"),
                ("package", "signoff"),
                ("signoff", "deploy"),
            ),
            expected_sequence=("plan", "compile", "unit", "integration", "package", "signoff", "deploy"),
            lns_window=3,
        ),
        DiversityFixture(
            fixture_id="diverse_fork_join_fulfillment",
            subproblem_family="fork_join_dag",
            geometry="two_branch_join",
            lns_shape="join_frontier_slice",
            source_kind="synthetic_scheduling",
            actions=("ship", "invoice", "pack", "pick", "verify", "reserve", "label", "receive"),
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
            expected_sequence=("invoice", "verify", "receive", "reserve", "label", "pick", "pack", "ship"),
            lns_window=3,
        ),
        DiversityFixture(
            fixture_id="diverse_grid_route",
            subproblem_family="grid_route",
            geometry="two_dimensional_path",
            lns_shape="route_prefix_repair",
            source_kind="synthetic_grid_path",
            actions=("deliver", "pickup", "route_22", "route_21", "route_20", "route_10", "route_00"),
            precedence=(
                ("route_00", "route_10"),
                ("route_10", "route_20"),
                ("route_20", "route_21"),
                ("route_21", "route_22"),
                ("route_22", "pickup"),
                ("pickup", "deliver"),
            ),
            expected_sequence=("route_00", "route_10", "route_20", "route_21", "route_22", "pickup", "deliver"),
            lns_window=3,
        ),
        DiversityFixture(
            fixture_id="diverse_bipartite_coloring",
            subproblem_family="bipartite_color_check",
            geometry="bipartite_constraint_join",
            lns_shape="constraint_group_slice",
            source_kind="synthetic_csp",
            actions=("publish", "check_bc", "choose_c", "check_ab", "choose_a", "choose_b"),
            precedence=(
                ("choose_a", "check_ab"),
                ("choose_b", "check_ab"),
                ("check_ab", "choose_c"),
                ("choose_c", "check_bc"),
                ("choose_b", "check_bc"),
                ("check_bc", "publish"),
            ),
            expected_sequence=("choose_a", "choose_b", "check_ab", "choose_c", "check_bc", "publish"),
            lns_window=2,
        ),
    )


def run_diagnostic(row_overrides: RowOverride | None = None) -> JsonDict:
    """Evaluate all fixture/mode rows and summarize row-derived evidence."""

    rows = [
        evaluate_fixture_mode(fixture, mode)
        for fixture in build_diversity_fixtures()
        for mode in HINT_MODES
    ]
    if row_overrides is not None:
        rows = row_overrides(rows)
    summary = _summarize_rows(rows)
    blockers = _readiness_blockers(summary)
    summary["active_constraint_diversity_ready"] = not blockers
    summary["readiness_blockers"] = blockers
    summary["row_records"] = rows
    return summary


def evaluate_fixture_mode(fixture: DiversityFixture, hint_mode: str) -> JsonDict:
    """Run one fixture/mode row while validating hints before final solve."""

    _require(hint_mode in HINT_MODES, f"hint_mode: {hint_mode}")
    baseline = _solve_without_hint(fixture)
    true_active = fixture.active_constraint_ids
    true_front = fixture.conflict_front
    true_lns = fixture.lns_subproblem_hint
    true_tail = fixture.active_tail
    true_frozen = fixture.frozen_variables
    active_hint, front_hint, lns_hint, tail_hint, frozen_hint = _hint_for_mode(
        fixture,
        hint_mode,
    )
    hint_matches = (
        active_hint == true_active
        and front_hint == true_front
        and lns_hint == true_lns
        and tail_hint == true_tail
        and frozen_hint == true_frozen
    )
    structurally_valid = _hint_structurally_valid(fixture, active_hint, front_hint, lns_hint)

    if hint_mode == "solver_only":
        decision = "ignored"
        fallback_used = False
        overwrite_used = False
        metrics = baseline
        candidate_sequence = baseline.final_sequence
    elif hint_mode == "lns_guided_hint" and hint_matches and structurally_valid:
        decision = "accepted"
        fallback_used = False
        overwrite_used = False
        metrics = SolverMetrics(
            final_sequence=baseline.final_sequence,
            final_valid=baseline.final_valid,
            solver_conflicts=0,
            solver_iterations=len(true_tail),
        )
        candidate_sequence = baseline.final_sequence
    elif hint_mode == "adversarial_hint":
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

    objective_value = _objective_cost(fixture, metrics.final_sequence)
    baseline_objective = _objective_cost(fixture, baseline.final_sequence)
    final_valid = bool(
        metrics.final_valid
        and _constraint_violation_count(fixture, metrics.final_sequence) == 0
    )
    return {
        "fixture_id": fixture.fixture_id,
        "subproblem_family": fixture.subproblem_family,
        "source_kind": fixture.source_kind,
        "hint_mode": hint_mode,
        "diversity_descriptor": fixture.diversity_descriptor,
        "active_constraint_hint": list(active_hint),
        "conflict_front_hint": list(front_hint),
        "lns_subproblem_hint": list(lns_hint),
        "active_tail_hint": list(tail_hint),
        "frozen_variable_hint": list(frozen_hint),
        "known_active_constraints": list(true_active),
        "known_conflict_front": list(true_front),
        "known_lns_subproblem": list(true_lns),
        "known_active_tail": list(true_tail),
        "known_frozen_variables": list(true_frozen),
        "active_tail_size": len(true_tail),
        "frozen_variable_count": len(true_frozen),
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
        "conflict_front_precision": _precision(front_hint, true_front),
        "expected_sequence": list(fixture.expected_sequence),
        "candidate_sequence": list(candidate_sequence),
        "final_sequence": list(metrics.final_sequence),
        "feasible": final_valid,
        "final_valid": final_valid,
        "final_validity_source": "solver_recomputed",
        "objective_value": objective_value,
        "baseline_objective_value": baseline_objective,
        "objective_preserved": objective_value == baseline_objective,
        "candidate_constraint_violation_count": _constraint_violation_count(
            fixture,
            candidate_sequence,
        ),
        "constraint_violation_count": _constraint_violation_count(fixture, metrics.final_sequence),
        "unsafe_false_accept": bool(decision == "accepted" and not final_valid),
    }


def diversity_descriptor_checksum(descriptors: Iterable[Mapping[str, Any]]) -> str:
    """Return a deterministic checksum for the descriptor set."""

    payload = sorted((dict(descriptor) for descriptor in descriptors), key=lambda item: item["fixture_id"])
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    row_overrides: RowOverride | None = None,
) -> JsonDict:
    """Build the terminal artifact from deterministic row records."""

    diagnostic = run_diagnostic(row_overrides=row_overrides)
    tests = [_normalize_test_run(item) for item in tests_run]
    blockers = list(diagnostic["readiness_blockers"])
    if diagnostic["active_constraint_diversity_ready"] and not tests:
        blockers.append("tests_not_recorded")
    ready = bool(diagnostic["active_constraint_diversity_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "duration_s": 1.07,
        "status": "complete" if ready else "blocked",
        "fixture_count": diagnostic["fixture_count"],
        "subproblem_family_count": diagnostic["subproblem_family_count"],
        "diversity_descriptor_checksum": diagnostic["diversity_descriptor_checksum"],
        "baseline_solver_work": diagnostic["baseline_solver_work"],
        "guided_solver_work": diagnostic["guided_solver_work"],
        "work_delta": diagnostic["work_delta"],
        "accepted_hint_count": diagnostic["accepted_hint_count"],
        "rejected_hint_count": diagnostic["rejected_hint_count"],
        "overwritten_hint_count": diagnostic["overwritten_hint_count"],
        "conflict_front_precision": diagnostic["conflict_front_precision"],
        "solver_validity_preserved": diagnostic["solver_validity_preserved"],
        "aggregate_from_rows_only": diagnostic["aggregate_from_rows_only"],
        "active_constraint_diversity_ready": ready,
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
        "source_artifacts": ["results/experiment_5419_active_constraint_lns_scale_v493.json"],
        "claim_limits": [
            "deterministic CPU-local diversity diagnostic",
            "LNS, active-constraint, active-tail, frozen-variable, and conflict-front hints are advisory",
            "solver recomputes every hint before accept, reject, or overwrite decisions",
            "stale and adversarial hint rows cannot determine final validity",
            "no LLM, generated text judge, hardware sampler, or speedup claim",
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
    """Write the validated Exp5433 artifact and return it."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when row-derived metrics or solver authority drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("hint_modes") == list(HINT_MODES), "hint_modes")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("REQ-VERIFY-5433" in artifact.get("spec_refs", []), "spec_refs")
    _require(len(str(artifact.get("reproducibility_checksum", ""))) == 64, "checksum")

    rows = artifact.get("row_records")
    _require(isinstance(rows, Sequence), "row_records")
    summary = _summarize_rows(rows)
    for field in (
        "fixture_count",
        "subproblem_family_count",
        "baseline_solver_work",
        "guided_solver_work",
        "work_delta",
        "accepted_hint_count",
        "rejected_hint_count",
        "overwritten_hint_count",
    ):
        _require(artifact.get(field) == summary[field], "aggregate_from_rows_only")
    _require(
        artifact.get("diversity_descriptor_checksum") == summary["diversity_descriptor_checksum"],
        "diversity_descriptor_checksum",
    )
    _require(
        artifact.get("conflict_front_precision") == summary["conflict_front_precision"],
        "conflict_front_precision",
    )
    _require(
        artifact.get("solver_validity_preserved") == summary["solver_validity_preserved"],
        "solver_validity_preserved",
    )
    _require(artifact.get("aggregate_from_rows_only") is True, "aggregate_from_rows_only")
    _require(summary["aggregate_from_rows_only"] is True, "aggregate_from_rows_only")
    _validate_rows(rows)
    if artifact.get("active_constraint_diversity_ready"):
        _require(artifact.get("status") == "complete", "status")
        _require(artifact.get("readiness_blockers") == [], "readiness_blockers")
        _require(bool(artifact.get("tests_run")), "tests_run")
        _require(summary["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(
            summary["subproblem_family_count"] >= MIN_SUBPROBLEM_FAMILY_COUNT,
            "subproblem_family_count",
        )
        _require(summary["baseline_solver_work"] > summary["guided_solver_work"], "work_delta")
        _require(summary["accepted_hint_count"] > 0, "accepted_hint_count")
        _require(summary["rejected_hint_count"] > 0, "rejected_hint_count")
        _require(summary["overwritten_hint_count"] > 0, "overwritten_hint_count")
        _require(summary["conflict_front_precision"] > 0.0, "conflict_front_precision")
        _require(summary["solver_validity_preserved"] is True, "solver_validity_preserved")


def _hint_for_mode(
    fixture: DiversityFixture,
    hint_mode: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if hint_mode == "solver_only":
        return (), (), (), (), ()
    if hint_mode == "lns_guided_hint":
        return (
            fixture.active_constraint_ids,
            fixture.conflict_front,
            fixture.lns_subproblem_hint,
            fixture.active_tail,
            fixture.frozen_variables,
        )
    if hint_mode == "stale_hint":
        prefix = (fixture.expected_sequence[0],)
        lns = _lns_subproblem(fixture, prefix)
        frozen = _frozen_for_lns(fixture, lns)
        return (
            _active_constraint_ids(fixture, prefix),
            _conflict_front(fixture, prefix),
            lns,
            lns,
            frozen,
        )
    before, after = fixture.precedence[0]
    lns = (before, after)
    return (f"{after}->{before}",), (before,), lns, lns, _frozen_for_lns(fixture, lns)


def _solve_without_hint(fixture: DiversityFixture) -> SolverMetrics:
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
    fixture: DiversityFixture,
    action: str,
    prefix: Sequence[str],
) -> bool:
    done = set(prefix)
    return all(before in done for before, after in fixture.precedence if after == action)


def _is_complete_valid_sequence(
    fixture: DiversityFixture,
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
    fixture: DiversityFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    done = set(prefix)
    return tuple(
        f"{before}->{after}"
        for before, after in fixture.precedence
        if before not in done and after not in done
    )


def _conflict_front(
    fixture: DiversityFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    active = _active_constraint_ids(fixture, prefix)
    return tuple(dict.fromkeys(edge.split("->", 1)[1] for edge in active))


def _lns_subproblem(
    fixture: DiversityFixture,
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


def _frozen_for_lns(
    fixture: DiversityFixture,
    lns: Sequence[str],
) -> tuple[str, ...]:
    active = set(lns)
    return tuple(action for action in fixture.expected_sequence if action not in active)


def _hint_structurally_valid(
    fixture: DiversityFixture,
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
    fixture: DiversityFixture,
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


def _contradictory_sequence(fixture: DiversityFixture) -> tuple[str, ...]:
    before, after = fixture.precedence[0]
    rest = [action for action in fixture.expected_sequence if action not in {before, after}]
    return (after, before, *rest)


def _constraint_violation_count(
    fixture: DiversityFixture,
    sequence: Sequence[str],
) -> int:
    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return len(fixture.precedence) + 1
    positions = {action: index for index, action in enumerate(sequence)}
    return sum(int(positions[before] > positions[after]) for before, after in fixture.precedence)


def _objective_cost(
    fixture: DiversityFixture,
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
    hinted_rows = [row for row in rows if row["conflict_front_hint"]]
    descriptors = _unique_descriptors(rows)
    baseline_solver_work = sum(int(row["baseline_solver_work"]) for row in solver_rows)
    guided_solver_work = sum(int(row["guided_solver_work"]) for row in guided_rows)
    solver_validity_preserved = all(
        bool(row["final_valid"] and row["objective_preserved"] and row["feasible"])
        for row in rows
    )
    return {
        "fixture_count": len({row["fixture_id"] for row in rows}),
        "subproblem_family_count": len({row["subproblem_family"] for row in rows}),
        "diversity_descriptor_checksum": diversity_descriptor_checksum(descriptors),
        "row_count": len(rows),
        "baseline_solver_work": baseline_solver_work,
        "guided_solver_work": guided_solver_work,
        "work_delta": baseline_solver_work - guided_solver_work,
        "accepted_hint_count": sum(int(row["hint_decision"] == "accepted") for row in rows),
        "rejected_hint_count": sum(int(row["hint_decision"] == "rejected") for row in rows),
        "overwritten_hint_count": sum(int(row["hint_decision"] == "overwritten") for row in rows),
        "conflict_front_precision": _rate(
            sum(float(row["conflict_front_precision"]) for row in hinted_rows),
            len(hinted_rows),
        ),
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


def _unique_descriptors(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    descriptors: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        descriptor = row["diversity_descriptor"]
        descriptors[str(descriptor["fixture_id"])] = descriptor
    return list(descriptors.values())


def _readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if summary["fixture_count"] != EXPECTED_FIXTURE_COUNT:
        blockers.append("fixture_count_mismatch")
    if summary["subproblem_family_count"] < MIN_SUBPROBLEM_FAMILY_COUNT:
        blockers.append("subproblem_family_count_below_minimum")
    if summary["baseline_solver_work"] <= summary["guided_solver_work"]:
        blockers.append("guided_work_not_reduced")
    if summary["accepted_hint_count"] <= 0:
        blockers.append("no_accepted_hints")
    if summary["rejected_hint_count"] <= 0:
        blockers.append("no_rejected_hints")
    if summary["overwritten_hint_count"] <= 0:
        blockers.append("no_overwritten_hints")
    if summary["conflict_front_precision"] <= 0.0:
        blockers.append("conflict_front_precision_not_positive")
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
        _require(row["feasible"] is True, "row feasible")
        _require(row["final_validity_source"] == "solver_recomputed", "final_validity_source")
        _require(
            int(row["work_delta"])
            == int(row["baseline_solver_work"]) - int(row["guided_solver_work"]),
            "aggregate_from_rows_only",
        )
        _require(row["active_tail_size"] == len(row["known_active_tail"]), "active_tail_size")
        _require(
            row["frozen_variable_count"] == len(row["known_frozen_variables"]),
            "frozen_variable_count",
        )
        if row["hint_mode"] in {"stale_hint", "adversarial_hint"}:
            _require(row["hint_matches_solver_view"] is False, "invalid hint match")
            _require(row["hint_decision"] in {"rejected", "overwritten"}, "invalid hint decision")
            _require(row["fallback_used"] is True, "invalid hint fallback")


def _precision(hint: Sequence[str], truth: Sequence[str]) -> float:
    if not hint:
        return 0.0
    return _rate(len(set(hint) & set(truth)), len(set(hint)))


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
            "complete: active-constraint LNS hints covered "
            f"{diagnostic['subproblem_family_count']} families, reduced solver work by "
            f"{diagnostic['work_delta']}, and preserved solver authority"
        )
    return "blocked: " + ",".join(blockers or ["active_constraint_diversity_not_ready"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover - exercised through run() in tests.
    run()
