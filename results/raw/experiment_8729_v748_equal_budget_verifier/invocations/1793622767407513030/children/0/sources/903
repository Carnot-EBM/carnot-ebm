"""Experiment 4503: HUD/register E3 state deepening to one L2 gate.

Spec refs: REQ-ARC-WMTE-4503, SCENARIO-ARC-WMTE-4503.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_4493_hud_register_deepen as exp4493


RESULT_RELATIVE_PATH = "results/experiment_4503_hud_register_deepen_l2.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-ARC-WMTE-4503", "SCENARIO-ARC-WMTE-4503"]
PRIOR_REPRODUCED_LEVELS = 1
TARGET_LEVEL = 2
DEFAULT_TARGET_GAME = "ar25"
SUPPORTED_GAMES = ("ar25", "ka59")
AR25_L2_TAIL = ("3", "3", "5", "2", "2", "2", "2", "2", "2", "2", "2")
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
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_."
    ),
    "inference_substrate": (
        "explicit substrate so adversarial_verify applies the right duration floor."
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
    "target_game",
    "register_state_contract",
    "goal_predicate_heldout_score",
    "grid_only_goal_predicate_heldout_score",
    "reproduction_gate",
    "solution_labels",
    "residual_blockers",
)


@dataclass(frozen=True)
class E3RegisteredState:
    grid: np.ndarray
    registers: Mapping[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalise_e3_state(state: Any) -> E3RegisteredState:
    if isinstance(state, E3RegisteredState):
        return E3RegisteredState(grid=np.asarray(state.grid), registers=dict(state.registers))
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], Mapping):
        return E3RegisteredState(grid=np.asarray(state[0]), registers=dict(state[1]))
    return E3RegisteredState(grid=np.asarray(state), registers={})


def induce_hud_registers(
    game: str,
    grid: Any,
    data: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    if game == "ka59":
        return {"hud_count": int(np.count_nonzero(np.asarray(grid) == 4))}
    if game == "ar25":
        stack = data.get("undo_stack") if isinstance(data, Mapping) else None
        return {"undo_stack_depth": len(stack) if isinstance(stack, list) else 0}
    return {}


def make_e3_state(
    game: str,
    grid: Any,
    registers: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
) -> E3RegisteredState:
    payload = dict(registers) if registers is not None else induce_hud_registers(game, grid, data)
    return E3RegisteredState(grid=np.asarray(grid), registers=payload)


def e3_state_key(state: Any) -> tuple[Any, ...]:
    registered = normalise_e3_state(state)
    arr = np.asarray(registered.grid)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    registers = tuple(
        (str(key), _stable_json(value))
        for key, value in sorted(registered.registers.items(), key=lambda item: str(item[0]))
    )
    return (tuple(int(item) for item in arr.shape), str(arr.dtype), digest, registers)


def registered_is_level_complete(game: str, state: Any) -> bool:
    registered = normalise_e3_state(state)
    if game == "ka59":
        if "hud_count" in registered.registers:
            return int(registered.registers["hud_count"]) <= 0
        return not bool(np.any(np.asarray(registered.grid) == 4))
    if game == "ar25":
        arr = np.asarray(registered.grid)
        return all(np.count_nonzero(arr == color) <= 1 for color in (1, 10, 11))
    raise ValueError(f"unsupported game for registered completion: {game}")


def build_goal_accountability_report() -> dict[str, Any]:
    return exp4493.build_goal_accountability_report()


def ensure_preconditions_ready(preconditions: Mapping[str, Any]) -> None:
    if not preconditions.get("offline_arcade_import_smoke"):
        raise RuntimeError("blocked_offline_arcade_import_smoke")
    if not preconditions.get("torch_import"):
        raise RuntimeError("blocked_torch_import")


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    return exp4493.check_preconditions(root)


def _new_reproduced_levels(reproduction_gate: Mapping[str, Any]) -> int:
    if not bool(reproduction_gate.get("reproduced", False)):
        return 0
    reached = int(reproduction_gate.get("reached_level", 0) or 0)
    return max(0, reached - PRIOR_REPRODUCED_LEVELS)


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
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
    if artifact.get("target_game") not in SUPPORTED_GAMES:
        errors.append("target_game must be ar25 or ka59")
    if not isinstance(artifact.get("reproduction_gate"), Mapping):
        errors.append("reproduction_gate must be a mapping")
    if not isinstance(artifact.get("solution_labels"), list):
        errors.append("solution_labels must be a list")
    if artifact.get("goal_predicate_heldout_score") is None:
        errors.append("goal_predicate_heldout_score must be populated")
    if artifact.get("grid_only_goal_predicate_heldout_score") is None:
        errors.append("grid_only_goal_predicate_heldout_score must be populated")
    if verdict and str(verdict).startswith(("success:", "success_")):
        if artifact.get("offline_reproduced") is not True or int(artifact.get("reproduced_levels", 0)) < 1:
            errors.append("success artifact requires offline_reproduced=true and reproduced_levels >= 1")
    return errors


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    target_game: str,
    solution_labels: Sequence[str],
    reproduction_gate: Mapping[str, Any],
    goal_report: Mapping[str, Any],
    tests_pass: bool,
    residual_blockers: Sequence[str] | None = None,
) -> dict[str, Any]:
    ensure_preconditions_ready(preconditions_checked)
    new_levels = _new_reproduced_levels(reproduction_gate)
    gate_reached_level = int(reproduction_gate.get("reached_level", 0) or 0)
    offline_reproduced = (
        bool(reproduction_gate.get("reproduced", False))
        and gate_reached_level >= TARGET_LEVEL
        and new_levels >= 1
    )
    residuals = list(residual_blockers or [])
    if not offline_reproduced:
        residuals.append(str(reproduction_gate.get("residual") or f"{target_game}_l2_not_reproduced"))
    verdict = (
        f"success: {target_game}_hud_register_deepen_l2_offline_reproduced"
        if offline_reproduced
        else f"complete: {target_game}_hud_register_deepen_l2_honest_residual"
    )
    payload: dict[str, Any] = {
        "experiment": "experiment_4503_hud_register_deepen_l2",
        "schema": "carnot.hud_register_deepen_l2_4503.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "preconditions_checked": dict(preconditions_checked),
        "target_game": str(target_game),
        "prior_reproduced_levels": PRIOR_REPRODUCED_LEVELS,
        "claimed_total_level": TARGET_LEVEL,
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": int(new_levels if offline_reproduced else 0),
        "total_reproduced_levels": gate_reached_level,
        "goal_predicate_heldout_score": goal_report["goal_predicate_heldout_score"],
        "grid_only_goal_predicate_heldout_score": goal_report[
            "grid_only_goal_predicate_heldout_score"
        ],
        "goal_examples_n": goal_report["goal_examples_n"],
        "register_state_contract": goal_report["register_state_contract"],
        "reproduction_gate": dict(reproduction_gate),
        "solution_labels": list(solution_labels),
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


def _candidate_plan(
    target_game: str,
) -> tuple[tuple[str, ...], Callable[..., Any]]:
    if target_game == "ar25":
        from carnot import experiment_4339_e3_explore_verify_plan_ar25 as ar25

        return tuple(ar25.L1_SOLUTION_LABELS) + AR25_L2_TAIL, ar25._apply_ar25_label
    if target_game == "ka59":
        from carnot import experiment_4340_e3_explore_verify_plan_ka59 as ka59

        return tuple(ka59.L1_SOLUTION_LABELS) + exp4493.KA59_REGISTER_PROBE_TAIL, ka59._apply_ka59_label
    raise ValueError(f"unsupported target_game: {target_game}")  # pragma: no cover


def _real_reproduce(  # pragma: no cover
    game: str,
    labels: tuple[str, ...],
    apply_fn: Callable[..., Any],
    claimed_level: int,
) -> Mapping[str, Any]:
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit.reproduce(game, labels, apply_fn, claimed_level=claimed_level)


def run_experiment(
    *,
    root: Path | str = REPO_ROOT,
    reproduction_runner: Callable[[str, tuple[str, ...], Callable[..., Any], int], Mapping[str, Any]]
    | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    tests_pass: bool = False,
    target_game: str = DEFAULT_TARGET_GAME,
) -> dict[str, Any]:
    checked = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root)
    ensure_preconditions_ready(checked)
    labels, apply_fn = _candidate_plan(target_game)
    runner = reproduction_runner or _real_reproduce
    replay = runner(target_game, labels, apply_fn, TARGET_LEVEL)
    artifact = build_artifact(
        preconditions_checked=checked,
        target_game=target_game,
        solution_labels=labels,
        reproduction_gate=replay,
        goal_report=build_goal_accountability_report(),
        tests_pass=tests_pass,
        residual_blockers=[],
    )
    write_artifact(artifact, root=root)
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run_experiment(preconditions_checked=check_preconditions(), tests_pass=False)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
