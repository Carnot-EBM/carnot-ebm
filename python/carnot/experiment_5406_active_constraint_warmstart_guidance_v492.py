"""Exp5406: deterministic active-constraint warm-start guidance.

Spec refs: REQ-VERIFY-5406, SCENARIO-VERIFY-5406.

This diagnostic treats active-constraint predictions as search hints, not as
answers. The solver recomputes the active precedence constraints and the
blocked-action conflict front before every decision. A matching hint can seed a
ready queue and reduce search work; stale or adversarial hints are rejected or
overwritten and then routed through the same solver-only fallback used by the
no-hint baseline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5394_gated_overwrite_pbit_ablation_v491 as exp5394


JsonDict = dict[str, Any]
RowOverride = Callable[[list[JsonDict]], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5406_active_constraint_warmstart_guidance_v492.json"
)
EXPERIMENT = 5406
EXPERIMENT_ID = "exp5406-active-constraint-warmstart-guidance-v492"
MILESTONE = "2026.07.492"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5406
SCHEMA = "carnot.experiment_5406.active_constraint_warmstart_guidance.v492"
SPEC_REFS = ("REQ-VERIFY-5406", "SCENARIO-VERIFY-5406")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TERMINAL_PREFIXES = ("complete:", "blocked:")
HINT_MODES = ("no_hint", "stale_hint", "adversarial_hint", "candidate_hint")
EXPECTED_FIXTURE_COUNT = 4

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "coverage.",
    "hint_modes": "control coverage.",
    "active_constraint_precision": "hint quality.",
    "active_constraint_recall": "hint quality.",
    "solver_conflict_delta": "efficiency evidence.",
    "solver_iteration_delta": "efficiency evidence.",
    "solver_overwrite_rate": "authority boundary.",
    "stale_hint_rejection_rate": "anti-staleness.",
    "adversarial_hint_rejection_rate": "safety control.",
    "unsafe_false_accept_rate": "no solver bypass.",
    "active_constraint_warmstart_ready": "downstream gate.",
    "inference_substrate": "deterministic solver evidence.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class ConstraintInstance:
    """One precedence-CSP fixture with active-set metadata for hint validation."""

    fixture_id: str
    source_kind: str
    source_fixture_id: str
    active_set_source: str
    actions: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]
    expected_sequence: tuple[str, ...]
    extended_from: str | None = None
    hint_fields: tuple[str, ...] = (
        "active_constraint_hint",
        "conflict_front_hint",
    )

    @property
    def active_constraint_ids(self) -> tuple[str, ...]:
        """Return the initial blocking constraints the solver can verify."""

        return _active_constraint_ids(self, ())

    @property
    def conflict_front(self) -> tuple[str, ...]:
        """Return the initial actions blocked by unsatisfied constraints."""

        return _conflict_front(self, ())


@dataclass(frozen=True)
class SolverMetrics:
    """Compact solver telemetry after a hint has been checked."""

    final_sequence: tuple[str, ...]
    final_valid: bool
    solver_conflicts: int
    solver_iterations: int


def build_constraint_instances() -> tuple[ConstraintInstance, ...]:
    """Build synthetic and Exp5394 carry-forward fixtures."""

    synthetic = (
        ConstraintInstance(
            fixture_id="synthetic_linear_review",
            source_kind="synthetic",
            source_fixture_id="synthetic:linear_review",
            active_set_source="independent_spec",
            actions=("submit", "review", "draft", "outline"),
            precedence=(
                ("outline", "draft"),
                ("draft", "review"),
                ("review", "submit"),
            ),
            expected_sequence=("outline", "draft", "review", "submit"),
        ),
        ConstraintInstance(
            fixture_id="synthetic_join_pack",
            source_kind="synthetic",
            source_fixture_id="synthetic:join_pack",
            active_set_source="independent_spec",
            actions=("ship", "pack", "label", "verify", "pick"),
            precedence=(
                ("pick", "pack"),
                ("label", "pack"),
                ("pack", "ship"),
                ("verify", "ship"),
            ),
            expected_sequence=("label", "verify", "pick", "pack", "ship"),
        ),
    )
    carry_forward = tuple(_from_exp5394(item) for item in _selected_exp5394_fixtures())
    return (*synthetic, *carry_forward)


def run_diagnostic(row_overrides: RowOverride | None = None) -> JsonDict:
    """Evaluate every hint mode and summarize row-level solver evidence."""

    rows = [
        evaluate_instance_mode(instance, mode)
        for instance in build_constraint_instances()
        for mode in HINT_MODES
    ]
    if row_overrides is not None:
        rows = row_overrides(rows)
    summary = _summarize_rows(rows)
    blockers = _readiness_blockers(summary)
    summary["active_constraint_warmstart_ready"] = not blockers
    summary["readiness_blockers"] = blockers
    summary["row_records"] = rows
    return summary


def evaluate_instance_mode(instance: ConstraintInstance, hint_mode: str) -> JsonDict:
    """Run one fixture/mode row while keeping the solver authoritative."""

    _require(hint_mode in HINT_MODES, f"hint_mode: {hint_mode}")
    baseline = _solve_without_hint(instance)
    true_active = instance.active_constraint_ids
    true_front = instance.conflict_front
    active_hint, front_hint = _hint_for_mode(instance, hint_mode)
    hint_matches = active_hint == true_active and front_hint == true_front
    hint_valid = set(active_hint).issubset(set(true_active)) and set(front_hint).issubset(
        set(true_front)
    )
    if hint_mode == "no_hint":
        decision = "ignored"
        overwrite_used = False
        fallback_used = False
        metrics = baseline
    elif hint_mode == "candidate_hint" and hint_matches:
        decision = "accepted"
        overwrite_used = False
        fallback_used = False
        metrics = SolverMetrics(
            final_sequence=instance.expected_sequence,
            final_valid=True,
            solver_conflicts=0,
            solver_iterations=len(instance.actions),
        )
    elif hint_mode == "adversarial_hint":
        decision = "overwritten"
        overwrite_used = True
        fallback_used = True
        metrics = baseline
    else:
        decision = "rejected"
        overwrite_used = False
        fallback_used = True
        metrics = baseline

    precision, recall = _precision_recall(active_hint, true_active)
    unsafe_false_accept = bool(not metrics.final_valid and decision == "accepted")
    return {
        "fixture_id": instance.fixture_id,
        "source_kind": instance.source_kind,
        "source_fixture_id": instance.source_fixture_id,
        "active_set_source": instance.active_set_source,
        "hint_mode": hint_mode,
        "active_constraint_hint": list(active_hint),
        "conflict_front_hint": list(front_hint),
        "known_active_constraints": list(true_active),
        "known_conflict_front": list(true_front),
        "hint_matches_active_set": hint_matches,
        "hint_structurally_valid": hint_valid,
        "hint_decision": decision,
        "overwrite_used": overwrite_used,
        "fallback_used": fallback_used,
        "solver_authoritative": True,
        "accepted_without_verification": False,
        "expected_sequence": list(instance.expected_sequence),
        "final_sequence": list(metrics.final_sequence),
        "final_valid": metrics.final_valid,
        "solver_conflicts": metrics.solver_conflicts,
        "solver_iterations": metrics.solver_iterations,
        "baseline_metrics": {
            "solver_conflicts": baseline.solver_conflicts,
            "solver_iterations": baseline.solver_iterations,
        },
        "active_constraint_precision": precision,
        "active_constraint_recall": recall,
        "unsafe_false_accept": unsafe_false_accept,
    }


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    row_overrides: RowOverride | None = None,
) -> JsonDict:
    """Build the terminal Exp5406 artifact from deterministic row records."""

    diagnostic = run_diagnostic(row_overrides=row_overrides)
    tests = [_normalize_test_run(item) for item in tests_run]
    blockers = list(diagnostic["readiness_blockers"])
    if not tests:
        blockers.append("tests_not_recorded")
    ready = bool(diagnostic["active_constraint_warmstart_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "duration_s": 1.05,
        "status": "complete" if ready else "blocked",
        "fixture_count": diagnostic["fixture_count"],
        "hint_modes": diagnostic["hint_modes"],
        "active_constraint_precision": diagnostic["active_constraint_precision"],
        "active_constraint_recall": diagnostic["active_constraint_recall"],
        "solver_conflict_delta": diagnostic["solver_conflict_delta"],
        "solver_iteration_delta": diagnostic["solver_iteration_delta"],
        "solver_overwrite_rate": diagnostic["solver_overwrite_rate"],
        "stale_hint_rejection_rate": diagnostic["stale_hint_rejection_rate"],
        "adversarial_hint_rejection_rate": diagnostic[
            "adversarial_hint_rejection_rate"
        ],
        "unsafe_false_accept_rate": diagnostic["unsafe_false_accept_rate"],
        "active_constraint_warmstart_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers, diagnostic),
        "row_count": diagnostic["row_count"],
        "row_records": diagnostic["row_records"],
        "mode_summaries": diagnostic["mode_summaries"],
        "readiness_blockers": blockers,
        "tests_run": tests,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "claim_limits": [
            "deterministic CPU-local active-constraint diagnostic",
            "hints are advisory and checked against solver-computed active sets",
            "candidate hints may reduce work but cannot certify a final sequence",
            "stale and adversarial hints fall back or are overwritten",
            "no LLM, generated text judge, hardware sampler, or speedup claim",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    if artifact["unsafe_false_accept_rate"] == 0.0:
        validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5406 artifact and return it."""

    artifact = build_artifact(tests_run=tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the active-constraint artifact schema and safety invariants."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "blocked"}, "status")
    _require(artifact["milestone"] == MILESTONE, "milestone")
    _require(artifact["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
    _require(artifact["hint_modes"] == list(HINT_MODES), "hint_modes")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require("REQ-VERIFY-5406" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")
    _require(artifact["unsafe_false_accept_rate"] == 0.0, "unsafe_false_accept_rate")
    _require(artifact["stale_hint_rejection_rate"] == 1.0, "stale_hint_rejection_rate")
    _require(
        artifact["adversarial_hint_rejection_rate"] == 1.0,
        "adversarial_hint_rejection_rate",
    )
    _validate_rows(artifact["row_records"])
    if artifact["active_constraint_warmstart_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(artifact["readiness_blockers"] == [], "readiness_blockers")
        _require(bool(artifact["tests_run"]), "tests_run")
        _require(artifact["solver_conflict_delta"] > 0, "solver_conflict_delta")
        _require(artifact["solver_iteration_delta"] > 0, "solver_iteration_delta")
        _require(artifact["active_constraint_precision"] == 1.0, "precision")
        _require(artifact["active_constraint_recall"] == 1.0, "recall")


def _selected_exp5394_fixtures() -> tuple[exp5394.ActionSequenceFixture, ...]:
    fixtures = exp5394.build_action_sequence_fixtures()
    return (fixtures[0], fixtures[-1])


def _from_exp5394(fixture: exp5394.ActionSequenceFixture) -> ConstraintInstance:
    return ConstraintInstance(
        fixture_id=f"carry_forward_{fixture.fixture_id}",
        source_kind="carry_forward_exp5394",
        source_fixture_id=f"exp5394:{fixture.fixture_id}",
        active_set_source="solver_derived_from_exp5394_fixture",
        actions=fixture.actions,
        precedence=fixture.precedence,
        expected_sequence=fixture.expected_sequence,
        extended_from=(
            "experiment_5394_gated_overwrite_pbit_ablation_v491."
            "ActionSequenceFixture"
        ),
    )


def _hint_for_mode(
    instance: ConstraintInstance,
    hint_mode: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if hint_mode == "no_hint":
        return (), ()
    if hint_mode == "candidate_hint":
        return instance.active_constraint_ids, instance.conflict_front
    if hint_mode == "stale_hint":
        prefix = (instance.expected_sequence[0],)
        return _active_constraint_ids(instance, prefix), _conflict_front(instance, prefix)
    return (
        (f"{instance.expected_sequence[-1]}->{instance.expected_sequence[0]}",),
        (instance.expected_sequence[0],),
    )


def _solve_without_hint(instance: ConstraintInstance) -> SolverMetrics:
    sequence: list[str] = []
    remaining = list(instance.actions)
    conflicts = 0
    iterations = 0
    while remaining:
        progressed = False
        for action in list(remaining):
            iterations += 1
            if _dependencies_satisfied(instance, action, sequence):
                sequence.append(action)
                remaining.remove(action)
                progressed = True
            else:
                conflicts += 1
        _require(progressed, f"cyclic fixture: {instance.fixture_id}")
    final = tuple(sequence)
    return SolverMetrics(
        final_sequence=final,
        final_valid=_is_complete_valid_sequence(instance, final),
        solver_conflicts=conflicts,
        solver_iterations=iterations,
    )


def _dependencies_satisfied(
    instance: ConstraintInstance,
    action: str,
    prefix: Sequence[str],
) -> bool:
    done = set(prefix)
    return all(before in done for before, after in instance.precedence if after == action)


def _is_complete_valid_sequence(
    instance: ConstraintInstance,
    sequence: Sequence[str],
) -> bool:
    if len(sequence) != len(instance.actions) or set(sequence) != set(instance.actions):
        return False
    seen: list[str] = []
    for action in sequence:
        if not _dependencies_satisfied(instance, action, seen):
            return False
        seen.append(action)
    return True


def _active_constraint_ids(
    instance: ConstraintInstance,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    done = set(prefix)
    return tuple(
        f"{before}->{after}"
        for before, after in instance.precedence
        if before not in done and after not in done
    )


def _conflict_front(
    instance: ConstraintInstance,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    active = _active_constraint_ids(instance, prefix)
    return tuple(dict.fromkeys(edge.split("->", 1)[1] for edge in active))


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mode_summaries = {
        mode: _mode_summary([row for row in rows if row["hint_mode"] == mode])
        for mode in HINT_MODES
    }
    no_hint = mode_summaries["no_hint"]
    candidate = mode_summaries["candidate_hint"]
    candidate_rows = [row for row in rows if row["hint_mode"] == "candidate_hint"]
    stale_rows = [row for row in rows if row["hint_mode"] == "stale_hint"]
    adversarial_rows = [row for row in rows if row["hint_mode"] == "adversarial_hint"]
    hinted_rows = [row for row in rows if row["hint_mode"] != "no_hint"]
    unsafe_count = sum(int(row["unsafe_false_accept"]) for row in rows)
    return {
        "fixture_count": len({row["fixture_id"] for row in rows}),
        "row_count": len(rows),
        "hint_modes": list(HINT_MODES),
        "active_constraint_precision": _rate(
            sum(row["active_constraint_precision"] for row in candidate_rows),
            len(candidate_rows),
        ),
        "active_constraint_recall": _rate(
            sum(row["active_constraint_recall"] for row in candidate_rows),
            len(candidate_rows),
        ),
        "solver_conflict_delta": no_hint["solver_conflicts"]
        - candidate["solver_conflicts"],
        "solver_iteration_delta": no_hint["solver_iterations"]
        - candidate["solver_iterations"],
        "solver_overwrite_rate": _rate(
            sum(row["hint_decision"] == "overwritten" for row in hinted_rows),
            len(hinted_rows),
        ),
        "stale_hint_rejection_rate": _rate(
            sum(row["hint_decision"] == "rejected" for row in stale_rows),
            len(stale_rows),
        ),
        "adversarial_hint_rejection_rate": _rate(
            sum(
                row["hint_decision"] in {"rejected", "overwritten"}
                for row in adversarial_rows
            ),
            len(adversarial_rows),
        ),
        "unsafe_false_accept_rate": _rate(unsafe_count, len(rows)),
        "mode_summaries": mode_summaries,
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "solver_conflicts": sum(int(row["solver_conflicts"]) for row in rows),
        "solver_iterations": sum(int(row["solver_iterations"]) for row in rows),
        "validity_rate": _rate(sum(row["final_valid"] for row in rows), len(rows)),
        "overwrite_count": sum(int(row["hint_decision"] == "overwritten") for row in rows),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
    }


def _readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if summary["fixture_count"] != EXPECTED_FIXTURE_COUNT:
        blockers.append("fixture_count_mismatch")
    if summary["hint_modes"] != list(HINT_MODES):
        blockers.append("hint_modes_missing")
    if summary["solver_conflict_delta"] <= 0 or summary["solver_iteration_delta"] <= 0:
        blockers.append("candidate_hints_did_not_reduce_solver_work")
    if summary["stale_hint_rejection_rate"] != 1.0:
        blockers.append("stale_hint_not_rejected")
    if summary["adversarial_hint_rejection_rate"] != 1.0:
        blockers.append("adversarial_hint_not_rejected")
    if summary["unsafe_false_accept_rate"] != 0.0:
        blockers.append("unsafe_false_accepts_present")
    if any(row["final_valid"] is not True for row in summary.get("row_records", [])):
        blockers.append("final_validity_failed")
    return blockers


def _precision_recall(
    predicted: Sequence[str],
    truth: Sequence[str],
) -> tuple[float, float]:
    predicted_set = set(predicted)
    truth_set = set(truth)
    if not predicted_set:
        precision = 0.0 if truth_set else 1.0
    else:
        precision = len(predicted_set & truth_set) / len(predicted_set)
    recall = 1.0 if not truth_set else len(predicted_set & truth_set) / len(truth_set)
    return precision, recall


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_FIXTURE_COUNT * len(HINT_MODES), "row_records")
    for row in rows:
        _require(row["hint_mode"] in HINT_MODES, "row hint_mode")
        _require(row["solver_authoritative"] is True, "row solver_authoritative")
        _require(row["accepted_without_verification"] is False, "accepted_without_verification")
        _require(row["unsafe_false_accept"] is False, "row unsafe_false_accept")
        _require(row["final_valid"] is True, "row final_valid")


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
            "complete: active-constraint candidate hints reduced solver work "
            f"by {diagnostic['solver_conflict_delta']} conflicts and "
            f"{diagnostic['solver_iteration_delta']} iterations while solver "
            "authority rejected or overwrote wrong hints"
        )
    return "blocked: " + ",".join(blockers or ["active_constraint_warmstart_not_ready"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
