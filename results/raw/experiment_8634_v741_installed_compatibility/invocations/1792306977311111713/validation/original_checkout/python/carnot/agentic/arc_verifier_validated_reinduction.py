"""Shared contracts for verifier-validated ARC-AGI-3 level re-induction.

Spec refs: REQ-PHASE4-021, SCENARIO-PHASE4-021.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_ARTIFACT_FIELDS = (
    "ACCURACY_levels_solved",
    "new_levels_solved_this_task",
    "verifier_validated_the_rule",
    "actions_saved_vs_openloop",
    "game_advanced",
    "per_level_actions",
    "baseline_actions_ref",
    "real_env_confirmed",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")


@dataclass(frozen=True)
class RuleValidation:
    """Verifier result for one re-induced level-local rule candidate.

    The rule is trusted only when it exactly matches demo transitions and has
    low held-out executed-consistency energy. Separating predicted and validated
    levels prevents a planner from claiming progress that copied-env execution
    did not reproduce.
    """

    candidate_id: str
    rule_name: str
    demo_fit: float
    heldout_energy: float | None
    heldout_n: int
    predicted_levels_after: int
    validated_levels_after: int
    planned_l2_actions: int


def choose_verified_candidate(
    candidates: list[RuleValidation],
    *,
    current_level: int,
    energy_threshold: float = 0.0,
) -> RuleValidation | None:
    """Choose the best candidate that passed demo-fit and held-out execution.

    Progress is primary: a candidate that actually advances farther in copied
    execution beats a cheaper no-progress candidate. Energy and action count are
    tie-breakers so equally advancing rules prefer the most consistent and then
    shortest plan.
    """

    valid = [
        candidate
        for candidate in candidates
        if candidate.demo_fit >= 1.0
        and candidate.heldout_energy is not None
        and candidate.heldout_energy <= energy_threshold
        and candidate.heldout_n > 0
        and candidate.predicted_levels_after == candidate.validated_levels_after
        and candidate.validated_levels_after >= current_level
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda candidate: (
            candidate.validated_levels_after,
            -float(candidate.heldout_energy),
            -candidate.planned_l2_actions,
            candidate.candidate_id,
        ),
    )


def executed_consistency_energy(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> float | None:
    """Return a simple held-out mismatch rate over executed transition fields.

    This is the GAP-4-style safety check used by Exp 3992: a candidate predicts
    concrete post-action state facts, copied-env execution observes them, and
    any disagreement raises energy before the real environment spends actions.
    """

    if not expected and not observed:
        return None

    mismatches = 0
    comparisons = 0
    n_rows = max(len(expected), len(observed))
    for index in range(n_rows):
        exp = expected[index] if index < len(expected) else {}
        obs = observed[index] if index < len(observed) else {}
        for key in sorted(set(exp) | set(obs)):
            comparisons += 1
            mismatches += int(exp.get(key) != obs.get(key))
    return round(mismatches / max(1, comparisons), 4)


def actions_saved_vs_openloop(*, openloop_actions: int, committed_rejected_actions: int) -> int:
    """Count Exp 3980-style rejected actions avoided by verifier pre-validation."""

    return max(0, int(openloop_actions) - int(committed_rejected_actions))


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Validate the bare-field terminal schema required by REQ-PHASE4-021."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    for field in ("ACCURACY_levels_solved", "new_levels_solved_this_task", "actions_saved_vs_openloop", "random_seed"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")

    for field in ("verifier_validated_the_rule", "real_env_confirmed"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")

    for field in ("game_advanced", "honest_verdict", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    for field in ("per_level_actions", "baseline_actions_ref"):
        if field not in artifact:
            continue
        value = artifact[field]
        if not isinstance(value, list) or any(type(item) is not int for item in value):
            errors.append(f"{field} must be a list of bare ints")

    if "duration_s" in artifact and type(artifact["duration_s"]) not in (int, float):
        errors.append("duration_s must be a bare number")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete:/success:/blocked_")
    return errors
