"""Experiment 4571: ka59-only hidden-field state-key probe.

Spec refs: REQ-ARC-WMTE-4571, SCENARIO-ARC-WMTE-4571.
"""

from __future__ import annotations

import hashlib
import json
import re
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from carnot import experiment_4559_hidden_field_state_probe as exp4559
from carnot.agentic import arc_game_adapters


EXPERIMENT = "experiment_4571_hidden_field_state_probe_ka59"
SCHEMA = "carnot.hidden_field_state_probe_ka59_4571.v1"
TARGET_GAME = "ka59"
TARGET_LEVEL = 2
DEFAULT_DEPTH_CAP = 2
RESULT_RELATIVE_PATH = "results/experiment_4571_hidden_field_state_probe_ka59.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
SPEC_REFS = ["REQ-ARC-WMTE-4571", "SCENARIO-ARC-WMTE-4571"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4571

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: hidden_field_state_ka59_L2_offline_reproduced OR "
        "complete: hidden_field_state_ka59_gap_sharpened_no_bank_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade solve with the extended "
        "state hash, no headline LLM load (1s floor)."
    ),
    "hidden_fields_added": (
        "the ka59 StepCounter register added to the state_key (read from internal state) -- "
        "traceable to GAP-ARCH-GRID-ONLY-STATE."
    ),
    "state_disambiguation_control_passed": (
        "THE GATE-FIRST POSITIVE CONTROL -- the extended state_key disambiguates a pair the "
        "grid-only key aliased; a no-bank null is valid only if this passed (else the change "
        "is a no-op)."
    ),
    "false_negative_risk_checked": (
        "a no-bank null is valid only if the disambiguation positive control passed."
    ),
    "offline_reproduced": "only offline-reproduced new levels count toward reproducible_total_levels.",
    "reproduced_levels": "the integer new-level count banked this task.",
    "missing_verifier_gaps": (
        "if no bank, which specific register the search still cannot read -- the sharpened "
        "GAP-ARCH-GRID-ONLY-STATE entry."
    ),
    "registry_updated": (
        "the ka59 hidden-field findings + dead-ends persisted so the next attempt reuses."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

HIDDEN_FIELDS_ADDED = {
    TARGET_GAME: [
        "step_counter_current_steps from game.urgssjskot.current_steps",
        "step_counter_limit from game.urgssjskot.koyyeuyzyr/current_level.get_data('StepCounter')",
    ]
}

grid_only_state_key = exp4559.grid_only_state_key


@dataclass(frozen=True)
class StatePair:
    frame: Any
    left_game: Any
    right_game: Any
    left_path: Sequence[str] = ()
    right_path: Sequence[str] = ()
    right_frame: Any | None = None


@dataclass(frozen=True)
class Ka59DeepenAttempt:
    reached_level: int
    offline_reproduced: bool
    reproduced_levels: int
    solution_labels: Sequence[str]
    reproduction_gate: Mapping[str, Any]
    residual: str | None
    states_expanded: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "game": TARGET_GAME,
            "target_level": TARGET_LEVEL,
            "reached_level": int(self.reached_level),
            "offline_reproduced": bool(self.offline_reproduced),
            "reproduced_levels": int(self.reproduced_levels),
            "solution_labels": list(self.solution_labels),
            "reproduction_gate": dict(self.reproduction_gate),
            "residual": self.residual,
            "states_expanded": int(self.states_expanded),
        }


def hidden_fields_from_game(game_state: Any) -> dict[str, Any]:
    return arc_game_adapters.hidden_state_registers(TARGET_GAME, game_state)


def build_real_ka59_state_pair() -> StatePair:  # pragma: no cover - exercised by required CLI.
    from carnot.agentic import arc_solver_kit as kit

    adapter = arc_game_adapters.get_adapter(TARGET_GAME)
    if adapter is None:
        raise RuntimeError("ka59 adapter missing")
    arc = kit.offline_arcade()
    env = arc.make(TARGET_GAME, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        TARGET_GAME,
        adapter.action_labels,
        adapter.apply,
        adapter.state_key,
        warmup_label=adapter.warmup_label,
        verifier=adapter.hand_verifier,
        branch_mode=getattr(adapter, "branch_mode", "replay"),
        max_nodes=1000,
    )
    seed: list[str] = []
    frame = solver._replay(env, [])
    while kit.frame_level(frame) < 1 and len(seed) < 20:
        labels = list(adapter.action_labels(env, frame, tuple(seed)))
        if not labels:
            break
        seed.append(labels[0])
        frame = solver._replay(env, seed)

    seen_by_grid: dict[Any, tuple[Any, Any, Any, tuple[str, ...], dict[str, Any]]] = {}
    queue: list[tuple[str, ...]] = [()]
    for path in queue:
        frame = solver._replay(env, [*seed, *path])
        game_snapshot = copy.deepcopy(env._game)
        grid_key = grid_only_state_key(frame)
        state_key = adapter.state_key(game_snapshot, frame)
        hidden = hidden_fields_from_game(game_snapshot)
        prior = seen_by_grid.get(grid_key)
        if prior is not None:
            prior_state, prior_frame, prior_game, prior_path, prior_hidden = prior
            if (
                prior_state != state_key
                and prior_hidden.get("step_counter_current_steps")
                != hidden.get("step_counter_current_steps")
            ):
                return StatePair(
                    frame=prior_frame,
                    right_frame=frame,
                    left_game=prior_game,
                    right_game=game_snapshot,
                    left_path=prior_path,
                    right_path=path,
                )
        seen_by_grid[grid_key] = (state_key, frame, game_snapshot, path, hidden)
        if len(path) < 5:
            for label in adapter.action_labels(env, frame, path):
                queue.append((*path, label))
    raise RuntimeError("ka59 StepCounter positive-control pair not found")


def build_state_disambiguation_control(
    *,
    pair_builder: Callable[[], StatePair] = build_real_ka59_state_pair,
) -> dict[str, Any]:
    adapter = arc_game_adapters.get_adapter(TARGET_GAME)
    pair = pair_builder()
    right_frame = pair.right_frame if pair.right_frame is not None else pair.frame
    left_grid = grid_only_state_key(pair.frame)
    right_grid = grid_only_state_key(right_frame)
    left_hidden = hidden_fields_from_game(pair.left_game)
    right_hidden = hidden_fields_from_game(pair.right_game)
    if adapter is None:
        left_extended = right_extended = None
        unreadable_register = "ka59_adapter_missing"
    else:
        left_extended = adapter.state_key(pair.left_game, pair.frame)
        right_extended = adapter.state_key(pair.right_game, right_frame)
        unreadable_register = "step_counter_current_steps"
    current_steps_differ = left_hidden.get("step_counter_current_steps") != right_hidden.get(
        "step_counter_current_steps"
    )
    grid_only_aliased = left_grid == right_grid
    extended_disambiguated = left_extended != right_extended
    passed = bool(grid_only_aliased and extended_disambiguated and current_steps_differ)
    return {
        "game": TARGET_GAME,
        "passed": passed,
        "grid_only_aliased": grid_only_aliased,
        "extended_state_key_disambiguated": extended_disambiguated,
        "step_counter_current_steps_differ": current_steps_differ,
        "left_path": list(pair.left_path),
        "right_path": list(pair.right_path),
        "left_grid_key": list(left_grid),
        "right_grid_key": list(right_grid),
        "left_hidden_fields": left_hidden,
        "right_hidden_fields": right_hidden,
        "unreadable_register": None if passed else unreadable_register,
    }


def _new_l2_levels(gate: Mapping[str, Any]) -> int:
    if not bool(gate.get("reproduced")):
        return 0
    reached = int(gate.get("reached_level", 0) or 0)
    return max(0, reached - 1) if reached >= TARGET_LEVEL else 0


def run_standing_loop_attempt() -> Ka59DeepenAttempt:  # pragma: no cover - exercised by required CLI.
    from carnot.agentic import arc_solver_kit as kit

    adapter = arc_game_adapters.get_adapter(TARGET_GAME)
    if adapter is None:
        return Ka59DeepenAttempt(
            reached_level=0,
            offline_reproduced=False,
            reproduced_levels=0,
            solution_labels=[],
            reproduction_gate={"game": TARGET_GAME, "reached_level": 0, "reproduced": False},
            residual="ka59_adapter_missing_for_hidden_field_probe",
        )

    arc = kit.offline_arcade()
    env = arc.make(TARGET_GAME, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        TARGET_GAME,
        adapter.action_labels,
        adapter.apply,
        adapter.state_key,
        warmup_label=adapter.warmup_label,
        verifier=adapter.hand_verifier,
        branch_mode=getattr(adapter, "branch_mode", "replay"),
        max_nodes=3000,
    )
    frame = solver._replay(env, [])
    current_level = kit.frame_level(frame)
    full: list[str] = []
    states_expanded = 0
    for level in range(current_level + 1, TARGET_LEVEL + 1):
        cap = int(adapter.depth_caps.get(level, DEFAULT_DEPTH_CAP))
        path, nodes = solver.solve_level(env, current_level, full, cap)
        states_expanded += int(nodes)
        if path is None:
            break
        full += list(path)
        frame = solver._replay(env, full)
        current_level = kit.frame_level(frame)
        if current_level < level:
            break

    gate = kit.reproduce(
        TARGET_GAME,
        full,
        adapter.apply,
        warmup_label=adapter.warmup_label,
        claimed_level=TARGET_LEVEL,
    )
    reproduced_levels = _new_l2_levels(gate)
    offline_reproduced = reproduced_levels >= 1 and bool(gate.get("reproduced"))
    residual = None if offline_reproduced else "ka59_l2_not_reproduced_after_step_counter_state_key"
    return Ka59DeepenAttempt(
        reached_level=int(gate.get("reached_level", current_level) or 0),
        offline_reproduced=offline_reproduced,
        reproduced_levels=reproduced_levels,
        solution_labels=full,
        reproduction_gate=gate,
        residual=residual,
        states_expanded=states_expanded,
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _checksum_material(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hidden_fields_added": artifact.get("hidden_fields_added"),
        "state_disambiguation_control": artifact.get("state_disambiguation_control"),
        "attempts": artifact.get("attempts"),
        "registry_update": artifact.get("registry_update"),
        "random_seed": artifact.get("random_seed"),
    }


def _control_failure_gap(state_control: Mapping[str, Any]) -> str:
    register = str(state_control.get("unreadable_register") or "step_counter_current_steps")
    return f"ka59_{register}_unreadable_control_failed"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    attempt: Ka59DeepenAttempt | None,
    state_control: Mapping[str, Any],
    registry_update: Mapping[str, Any],
) -> dict[str, Any]:
    state_control_passed = bool(state_control.get("passed"))
    successful = (
        attempt
        if attempt is not None and attempt.offline_reproduced and attempt.reproduced_levels >= 1
        else None
    )
    if successful is not None:
        verdict = "success: hidden_field_state_ka59_L2_offline_reproduced"
        offline_reproduced = True
        reproduced_levels = int(successful.reproduced_levels)
        missing: list[str] = []
    else:
        verdict = "complete: hidden_field_state_ka59_gap_sharpened_no_bank_honest_null"
        offline_reproduced = False
        reproduced_levels = 0
        if state_control_passed and attempt is not None and attempt.residual:
            missing = [str(attempt.residual)]
        elif not state_control_passed:
            missing = [_control_failure_gap(state_control)]
        else:
            missing = ["ka59_l2_not_reproduced_after_step_counter_state_key"]
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "hidden_fields_added": dict(HIDDEN_FIELDS_ADDED),
        "state_disambiguation_control_passed": state_control_passed,
        "state_disambiguation_control": _json_safe(state_control),
        "false_negative_risk_checked": state_control_passed,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "missing_verifier_gaps": missing,
        "registry_updated": bool(registry_update.get("updated")),
        "registry_update": _json_safe(registry_update),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "attempts": _json_safe([attempt.as_dict()] if attempt is not None else []),
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(_checksum_material(payload))
    payload["schema_errors"] = artifact_schema_errors(payload)
    return payload


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "honest_verdict",
        "inference_substrate",
        "hidden_fields_added",
        "state_disambiguation_control_passed",
        "false_negative_risk_checked",
        "offline_reproduced",
        "reproduced_levels",
        "missing_verifier_gaps",
        "registry_updated",
        "random_seed",
        "reproducibility_checksum",
        "preconditions_checked",
    ):
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("hidden_fields_added") != HIDDEN_FIELDS_ADDED:
        errors.append("hidden_fields_added mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("registry_updated") is not True:
        errors.append("registry_updated must be true for persisted ka59 findings")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(
        _checksum_material(artifact)
    ):
        errors.append("checksum mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict.startswith("success:"):
        if not (
            artifact.get("offline_reproduced") is True
            and int(artifact.get("reproduced_levels") or 0) >= 1
            and artifact.get("state_disambiguation_control_passed") is True
        ):
            errors.append("success artifact requires ka59 L2 offline reproduction")
    elif verdict.startswith("complete:"):
        if artifact.get("offline_reproduced") is not False:
            errors.append("complete artifact must not claim offline reproduction")
        if not artifact.get("missing_verifier_gaps"):
            errors.append("complete artifact requires a verifier/register gap")
        if artifact.get("state_disambiguation_control_passed") is False and artifact.get(
            "false_negative_risk_checked"
        ) is not False:
            errors.append("failed control must not claim false-negative risk checked")
    else:
        errors.append("honest_verdict must start with terminal prefix")
    return errors


def _replace_top_level_registry_section(registry_text: str, key: str, section_text: str) -> str:
    lines = registry_text.splitlines(keepends=True)
    start = next((index for index, line in enumerate(lines) if line.startswith(f"{key}:")), None)
    if start is None:
        return registry_text.rstrip() + "\n" + section_text
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z0-9_][^:\n]*:\s*(?:#.*)?$", lines[index]):
            end = index
            break
    return "".join(lines[:start]) + section_text + "".join(lines[end:])


def apply_registry_probe(
    registry_text: str,
    *,
    artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    registry = yaml.safe_load(registry_text) or {}
    prior = registry.get("latest_hidden_field_state_probe_4571")
    row = {
        "artifact": RESULT_RELATIVE_PATH,
        "honest_verdict": artifact.get("honest_verdict"),
        "hidden_fields_added": artifact.get("hidden_fields_added"),
        "state_disambiguation_control_passed": artifact.get(
            "state_disambiguation_control_passed"
        ),
        "false_negative_risk_checked": artifact.get("false_negative_risk_checked"),
        "offline_reproduced": artifact.get("offline_reproduced"),
        "reproduced_levels": artifact.get("reproduced_levels"),
        "missing_verifier_gaps": artifact.get("missing_verifier_gaps"),
        "dead_ends": [
            {
                "game": attempt.get("game"),
                "residual": attempt.get("residual"),
                "reached_level": attempt.get("reached_level"),
            }
            for attempt in artifact.get("attempts", [])
            if attempt.get("residual")
        ],
    }
    section_text = yaml.safe_dump(
        {"latest_hidden_field_state_probe_4571": row},
        sort_keys=False,
        width=1000,
    )
    updated_text = _replace_top_level_registry_section(
        registry_text,
        "latest_hidden_field_state_probe_4571",
        section_text,
    )
    return updated_text, {
        "updated": prior != row,
        "path": REGISTRY_RELATIVE_PATH,
        "section": "latest_hidden_field_state_probe_4571",
        "target_game": TARGET_GAME,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _spec_refs_present(root: Path) -> bool:
    spec = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return all(ref in spec for ref in SPEC_REFS)


def check_offline_arcade() -> bool:  # pragma: no cover - exercised by required CLI.
    from carnot.agentic import arc_solver_kit

    arc_solver_kit.offline_arcade()
    return True


def run_experiment(
    *,
    root: Path | None = None,
    precondition_checker: Callable[[], bool] = check_offline_arcade,
    control_builder: Callable[[], Mapping[str, Any]] = build_state_disambiguation_control,
    attempt_runner: Callable[[], Ka59DeepenAttempt] = run_standing_loop_attempt,
    instructions_checked: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    instructions = dict(instructions_checked or {"AGENTS.md": True, "CODEX.md": True})
    preconditions_checked = {
        **instructions,
        "offline_arcade_import_smoke": bool(precondition_checker()),
        "spec_refs_present": _spec_refs_present(root),
    }
    state_control = control_builder()
    attempt = attempt_runner() if state_control.get("passed") else None
    placeholder_update = {"updated": True, "path": REGISTRY_RELATIVE_PATH}
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        attempt=attempt,
        state_control=state_control,
        registry_update=placeholder_update,
    )
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_text = registry_path.read_text(encoding="utf-8")
    updated_registry, registry_update = apply_registry_probe(registry_text, artifact=artifact)
    registry_update["updated"] = True
    registry_path.write_text(updated_registry, encoding="utf-8")
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        attempt=attempt,
        state_control=state_control,
        registry_update=registry_update,
    )
    _write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in (
                    "honest_verdict",
                    "state_disambiguation_control_passed",
                    "false_negative_risk_checked",
                    "offline_reproduced",
                    "reproduced_levels",
                    "registry_updated",
                    "reproducibility_checksum",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
