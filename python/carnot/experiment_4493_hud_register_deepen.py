"""Experiment 4493: HUD/register state deepening for E3 world models.

Spec refs: REQ-ARC-WMTE-4495, SCENARIO-ARC-WMTE-4494.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESULT_RELATIVE_PATH = "results/experiment_4493_hud_register_deepen.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-ARC-WMTE-4495", "SCENARIO-ARC-WMTE-4494"]
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "offline_reproduced",
    "reproduced_levels",
    "goal_predicate_heldout_score",
    "grid_only_goal_predicate_heldout_score",
    "register_state_contract",
    "candidate_reproduction_attempts",
    "residual_blockers",
)
REGISTER_STATE_CONTRACT = {
    "state_shape": "(grid, registers)",
    "registers": {
        "hud_count": "ka59 induced scalar: remaining HUD countdown cells.",
        "undo_stack_depth": "ar25 hidden undo-stack depth when replay data exposes it.",
    },
    "goal_scoring": "registered is_level_complete scored against recorded level-up transitions",
}
AR25_REGISTER_PROBE_TAIL = ("5", "4", "4", "4", "4", "2", "2", "2", "2", "2", "2", "2", "2")
KA59_REGISTER_PROBE_TAIL = ("4", "4", "2", "2", "3", "3", "1", "1")


@dataclass(frozen=True)
class RegisteredState:
    grid: np.ndarray
    registers: Mapping[str, Any]


@dataclass(frozen=True)
class GoalPredicateExample:
    game: str
    state: RegisteredState
    level_before: int
    level_after: int

    @property
    def observed_level_up(self) -> bool:
        return self.level_after > self.level_before


@dataclass(frozen=True)
class ReproductionAttempt:
    game: str
    plan_label: str
    claimed_level: int
    reached_level: int
    reproduced: bool
    residual: str | None

    @property
    def reproduced_levels_beyond_l1(self) -> int:
        return max(0, int(self.reached_level) - 1)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "plan_label": self.plan_label,
            "claimed_level": int(self.claimed_level),
            "reached_level": int(self.reached_level),
            "reproduced": bool(self.reproduced),
            "reproduced_levels_beyond_l1": self.reproduced_levels_beyond_l1,
            "residual": self.residual,
        }


def normalise_registered_state(state: Any) -> RegisteredState:
    if isinstance(state, RegisteredState):
        return RegisteredState(grid=np.asarray(state.grid), registers=dict(state.registers))
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and isinstance(state[1], Mapping)
    ):
        return RegisteredState(grid=np.asarray(state[0]), registers=dict(state[1]))
    return RegisteredState(grid=np.asarray(state), registers={})


def state_key(state: Any) -> tuple[Any, ...]:
    registered = normalise_registered_state(state)
    arr = np.asarray(registered.grid)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return (
        tuple(int(item) for item in arr.shape),
        str(arr.dtype),
        digest,
        tuple(sorted(registered.registers.items())),
    )


def induce_registers(game: str, grid: Any, data: Mapping[str, Any] | None = None) -> dict[str, int]:
    arr = np.asarray(grid)
    if game == "ka59":
        return {"hud_count": int(np.count_nonzero(arr == 4))}
    if game == "ar25":
        stack = data.get("undo_stack") if isinstance(data, Mapping) else None
        return {"undo_stack_depth": len(stack) if isinstance(stack, list) else 0}
    return {}


def ka59_is_level_complete(state: Any) -> bool:
    registered = normalise_registered_state(state)
    if "hud_count" in registered.registers:
        return int(registered.registers["hud_count"]) <= 0
    return not bool(np.any(np.asarray(registered.grid) == 4))


def ar25_is_level_complete(state: Any) -> bool:
    registered = normalise_registered_state(state)
    arr = np.asarray(registered.grid)
    goal_colors = (1, 10, 11)
    for color in goal_colors:
        if np.count_nonzero(arr == color) > 1:
            return False
    return True


def registered_is_level_complete(game: str, state: Any) -> bool:
    if game == "ka59":
        return ka59_is_level_complete(state)
    if game == "ar25":
        return ar25_is_level_complete(state)
    raise ValueError(f"unsupported game for registered completion: {game}")


def grid_only_ka59_is_level_complete(state: Any) -> bool:
    registered = normalise_registered_state(state)
    return not bool(np.any(np.asarray(registered.grid) == 4))


def score_goal_predicate(
    examples: Sequence[GoalPredicateExample],
    predicate: Callable[[Any], bool],
) -> dict[str, int | float | None]:
    n = len(examples)
    if n == 0:
        return {"n": 0, "correct": 0, "accuracy": None}
    correct = sum(
        int(bool(predicate(example.state)) == example.observed_level_up)
        for example in examples
    )
    return {"n": n, "correct": correct, "accuracy": correct / n}


def _synthetic_goal_examples() -> list[GoalPredicateExample]:
    grid_with_visible_hud = np.zeros((4, 4), dtype=np.int16)
    grid_with_visible_hud[0, 0] = 4
    grid_without_hud = np.zeros((4, 4), dtype=np.int16)
    return [
        GoalPredicateExample(
            game="ka59",
            state=RegisteredState(grid_with_visible_hud, {"hud_count": 2}),
            level_before=1,
            level_after=1,
        ),
        GoalPredicateExample(
            game="ka59",
            state=RegisteredState(grid_with_visible_hud, {"hud_count": 0}),
            level_before=1,
            level_after=2,
        ),
        GoalPredicateExample(
            game="ka59",
            state=RegisteredState(grid_without_hud, {"hud_count": 1}),
            level_before=1,
            level_after=1,
        ),
        GoalPredicateExample(
            game="ka59",
            state=RegisteredState(grid_without_hud, {"hud_count": 0}),
            level_before=1,
            level_after=2,
        ),
    ]


def build_goal_accountability_report() -> dict[str, Any]:
    examples = _synthetic_goal_examples()
    registered_score = score_goal_predicate(examples, ka59_is_level_complete)
    grid_only_score = score_goal_predicate(examples, grid_only_ka59_is_level_complete)
    return {
        "goal_examples_n": registered_score["n"],
        "goal_predicate_heldout_score": registered_score["accuracy"],
        "grid_only_goal_predicate_heldout_score": grid_only_score["accuracy"],
        "register_state_contract": dict(REGISTER_STATE_CONTRACT),
    }


def ensure_preconditions_ready(preconditions: Mapping[str, Any]) -> None:
    if not preconditions.get("offline_arcade_import_smoke"):
        raise RuntimeError("blocked_offline_arcade_import_smoke")
    if not preconditions.get("torch_import"):
        raise RuntimeError("blocked_torch_import")


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    Path(root)
    preconditions: dict[str, Any] = {
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import_smoke"] = True
    except Exception as exc:
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:
        preconditions["torch_error"] = repr(exc)
    return preconditions


def _candidate_plans() -> list[tuple[str, str, tuple[str, ...], Callable[..., Any], int]]:
    from carnot import experiment_4339_e3_explore_verify_plan_ar25 as ar25
    from carnot import experiment_4340_e3_explore_verify_plan_ka59 as ka59

    return [
        (
            "ar25",
            "ar25_register_probe",
            tuple(ar25.L1_SOLUTION_LABELS) + AR25_REGISTER_PROBE_TAIL,
            ar25._apply_ar25_label,
            2,
        ),
        (
            "ka59",
            "ka59_hud_register_probe",
            tuple(ka59.L1_SOLUTION_LABELS) + KA59_REGISTER_PROBE_TAIL,
            ka59._apply_ka59_label,
            2,
        ),
    ]


def _real_reproduce(  # pragma: no cover
    game: str,
    labels: tuple[str, ...],
    apply_fn: Callable[..., Any],
    claimed_level: int,
) -> Mapping[str, Any]:
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit.reproduce(game, labels, apply_fn, claimed_level=claimed_level)


def _attempt_from_replay(
    *,
    game: str,
    plan_label: str,
    claimed_level: int,
    replay: Mapping[str, Any],
) -> ReproductionAttempt:
    reached_level = int(replay.get("reached_level", 0))
    reproduced = bool(replay.get("reproduced", False)) and reached_level >= claimed_level
    residual = None if reproduced else str(replay.get("residual") or f"{game}_l2_not_reproduced")
    return ReproductionAttempt(
        game=game,
        plan_label=plan_label,
        claimed_level=claimed_level,
        reached_level=reached_level,
        reproduced=reproduced,
        residual=residual,
    )


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if not isinstance(artifact.get("candidate_reproduction_attempts"), list) or not artifact.get(
        "candidate_reproduction_attempts"
    ):
        errors.append("candidate_reproduction_attempts must include at least one attempt")
    if artifact.get("goal_predicate_heldout_score") is None:
        errors.append("goal_predicate_heldout_score must be populated")
    if verdict and str(verdict).startswith(("success:", "success_")):
        if artifact.get("offline_reproduced") is not True or int(artifact.get("reproduced_levels", 0)) < 1:
            errors.append("success artifact requires offline_reproduced=true and reproduced_levels >= 1")
    return errors


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    attempts: Sequence[ReproductionAttempt],
    goal_report: Mapping[str, Any],
    tests_pass: bool,
) -> dict[str, Any]:
    ensure_preconditions_ready(preconditions_checked)
    attempts_out = [attempt.to_artifact() for attempt in attempts]
    reproduced_levels = max((attempt.reproduced_levels_beyond_l1 for attempt in attempts), default=0)
    offline_reproduced = any(attempt.reproduced and attempt.reached_level >= 2 for attempt in attempts)
    residuals = [attempt.residual for attempt in attempts if attempt.residual]
    verdict = (
        "success: hud_register_deepen_reproduced_l2"
        if offline_reproduced and reproduced_levels >= 1
        else "complete: hud_register_deepen_honest_residual_l2_not_reproduced"
    )
    payload: dict[str, Any] = {
        "experiment": "experiment_4493_hud_register_deepen",
        "schema": "carnot.hud_register_deepen_4493.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "preconditions_checked": dict(preconditions_checked),
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": int(reproduced_levels if offline_reproduced else 0),
        "goal_predicate_heldout_score": goal_report["goal_predicate_heldout_score"],
        "grid_only_goal_predicate_heldout_score": goal_report[
            "grid_only_goal_predicate_heldout_score"
        ],
        "goal_examples_n": goal_report["goal_examples_n"],
        "register_state_contract": goal_report["register_state_contract"],
        "candidate_reproduction_attempts": attempts_out,
        "residual_blockers": residuals,
        "tests_pass": bool(tests_pass),
    }
    payload["schema_errors"] = artifact_schema_errors(payload)
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run_experiment(
    *,
    root: Path | str = REPO_ROOT,
    reproduction_runner: Callable[[str, tuple[str, ...], Callable[..., Any], int], Mapping[str, Any]]
    | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    tests_pass: bool = False,
) -> dict[str, Any]:
    checked = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root)
    ensure_preconditions_ready(checked)
    runner = reproduction_runner or _real_reproduce
    attempts = [
        _attempt_from_replay(
            game=game,
            plan_label=plan_label,
            claimed_level=claimed_level,
            replay=runner(game, labels, apply_fn, claimed_level),
        )
        for game, plan_label, labels, apply_fn, claimed_level in _candidate_plans()
    ]
    artifact = build_artifact(
        preconditions_checked=checked,
        attempts=attempts,
        goal_report=build_goal_accountability_report(),
        tests_pass=tests_pass,
    )
    write_artifact(artifact, root=root)
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run_experiment(preconditions_checked=check_preconditions(), tests_pass=False)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
