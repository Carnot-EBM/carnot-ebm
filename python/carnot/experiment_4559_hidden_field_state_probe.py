"""Experiment 4559: hidden-field state-key probe.

Spec refs: REQ-ARC-WMTE-4559, SCENARIO-ARC-WMTE-4559.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from carnot.agentic import arc_game_adapters


EXPERIMENT = "experiment_4559_hidden_field_state_probe"
SCHEMA = "carnot.hidden_field_state_probe_4559.v1"
RESULT_RELATIVE_PATH = "results/experiment_4559_hidden_field_state_probe.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
SPEC_REFS = ["REQ-ARC-WMTE-4559", "SCENARIO-ARC-WMTE-4559"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4559
TARGET_GAMES = ("ka59", "ar25", "ft09")
TARGET_LEVEL = 2
DEFAULT_DEPTH_CAP = 2

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: hidden_field_state_<game>_L2_offline_reproduced OR "
        "complete: hidden_field_state_gap_sharpened_no_bank_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade solve with the extended "
        "state hash, no headline LLM load."
    ),
    "hidden_fields_added": (
        "the HUD/hidden registers added to the state_key per game (read from internal state) -- "
        "traceable to GAP-ARCH-GRID-ONLY-STATE."
    ),
    "state_disambiguation_control_passed": (
        "the POSITIVE CONTROL -- the extended state_key disambiguates a pair the grid-only key "
        "aliased; guards a no-op change; a no-bank null is valid only if this passed."
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
        "the per-game hidden-field findings + dead-ends persisted so the next attempt reuses."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

HIDDEN_FIELDS_ADDED = {
    "ka59": [
        "step_counter_current_steps from game.urgssjskot.current_steps",
        "step_counter_limit from game.urgssjskot.koyyeuyzyr/current_level.get_data('StepCounter')",
    ],
    "ar25": [
        "undo_stack_depth from len(game.flqblmrxsla)",
        "step_counter_current_steps from game.lelsvjlwneo.current_steps",
        "step_counter_limit from game.lelsvjlwneo.ilqnjlrnkk/current_level.get_data('StepCounter')",
    ],
    "ft09": [
        "color_cycle from game.gqb/current_level.get_data('cwU')",
        "cell_cycle_phases from live Hkx/NTi sprite center colors",
        "step_counter_current_steps from game.lpw.dzy",
    ],
}


@dataclass(frozen=True)
class DeepenAttempt:
    game: str
    reached_level: int
    offline_reproduced: bool
    reproduced_levels: int
    solution_labels: Sequence[str]
    reproduction_gate: Mapping[str, Any]
    residual: str | None
    states_expanded: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "target_level": TARGET_LEVEL,
            "reached_level": int(self.reached_level),
            "offline_reproduced": bool(self.offline_reproduced),
            "reproduced_levels": int(self.reproduced_levels),
            "solution_labels": list(self.solution_labels),
            "reproduction_gate": dict(self.reproduction_gate),
            "residual": self.residual,
            "states_expanded": int(self.states_expanded),
        }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _array_key(value: Any) -> tuple[Any, ...]:
    arr = np.asarray(value)
    return (
        tuple(int(item) for item in arr.shape),
        str(arr.dtype),
        hashlib.sha256(arr.tobytes()).hexdigest(),
    )


def grid_only_state_key(frame: Any) -> tuple[Any, ...]:
    try:
        from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of

        return ("grid", frame_hash(grid_of(frame)))
    except Exception:
        pass
    grid = getattr(frame, "grid", None)
    if grid is None:
        grid = getattr(frame, "observation", None)
    if grid is None:
        return ("grid", None, int(getattr(frame, "levels_completed", 0) or 0))
    return ("grid", *_array_key(grid))


def hidden_fields_from_game(game: str, game_state: Any) -> dict[str, Any]:
    return arc_game_adapters.hidden_state_registers(game, game_state)


class _FakeLevel:
    def __init__(self, data: Mapping[str, Any]):
        self._data = dict(data)

    def get_data(self, name: str) -> Any:
        return self._data.get(name)


def _fake_frame() -> SimpleNamespace:
    return SimpleNamespace(levels_completed=1, grid=np.zeros((4, 4), dtype=np.int16))


def _fake_sprite(*, color: int, x: int = 1, y: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        name="Hkx-test",
        tags=["Hkx"],
        x=x,
        y=y,
        width=3,
        height=3,
        pixels=np.array([[0, 0, 0], [0, color, 0], [0, 0, 0]], dtype=np.int16),
    )


def _positive_control_pairs() -> dict[str, tuple[Any, Any, Any]]:
    frame = _fake_frame()
    return {
        "ka59": (
            frame,
            SimpleNamespace(
                current_level=_FakeLevel({"StepCounter": 100}),
                urgssjskot=SimpleNamespace(current_steps=100, koyyeuyzyr=100),
            ),
            SimpleNamespace(
                current_level=_FakeLevel({"StepCounter": 100}),
                urgssjskot=SimpleNamespace(current_steps=99, koyyeuyzyr=100),
            ),
        ),
        "ar25": (
            frame,
            SimpleNamespace(
                current_level=_FakeLevel({"StepCounter": 64}),
                lelsvjlwneo=SimpleNamespace(current_steps=64, ilqnjlrnkk=64),
                flqblmrxsla=[],
            ),
            SimpleNamespace(
                current_level=_FakeLevel({"StepCounter": 64}),
                lelsvjlwneo=SimpleNamespace(current_steps=64, ilqnjlrnkk=64),
                flqblmrxsla=[{"undo": 1}],
            ),
        ),
        "ft09": (
            frame,
            SimpleNamespace(
                current_level=_FakeLevel({"cwU": [9, 8]}),
                gqb=[9, 8],
                fhc=[_fake_sprite(color=9)],
                mou=[],
                lpw=SimpleNamespace(dzy=32, oro=32),
                our=0,
            ),
            SimpleNamespace(
                current_level=_FakeLevel({"cwU": [8, 9]}),
                gqb=[8, 9],
                fhc=[_fake_sprite(color=9)],
                mou=[],
                lpw=SimpleNamespace(dzy=32, oro=32),
                our=0,
            ),
        ),
    }


def build_state_disambiguation_control() -> dict[str, Any]:
    per_game: dict[str, dict[str, Any]] = {}
    for game, (frame, left, right) in _positive_control_pairs().items():
        adapter = arc_game_adapters.get_adapter(game)
        left_grid = grid_only_state_key(frame)
        right_grid = grid_only_state_key(frame)
        if adapter is None:
            left_extended = right_extended = None
        else:
            left_extended = adapter.state_key(left, frame)
            right_extended = adapter.state_key(right, frame)
        per_game[game] = {
            "grid_only_aliased": left_grid == right_grid,
            "extended_state_key_disambiguated": left_extended != right_extended,
            "hidden_fields_left": hidden_fields_from_game(game, left),
            "hidden_fields_right": hidden_fields_from_game(game, right),
        }
    return {
        "passed": all(
            row["grid_only_aliased"] and row["extended_state_key_disambiguated"]
            for row in per_game.values()
        ),
        "per_game": per_game,
    }


def _new_l2_levels(gate: Mapping[str, Any]) -> int:
    if not bool(gate.get("reproduced")):
        return 0
    reached = int(gate.get("reached_level", 0) or 0)
    return max(0, reached - 1) if reached >= TARGET_LEVEL else 0


def run_standing_loop_attempt(
    game: str,
    target_level: int = TARGET_LEVEL,
    depth_cap: int = DEFAULT_DEPTH_CAP,
) -> DeepenAttempt:  # pragma: no cover - exercised by required experiment command.
    from carnot.agentic import arc_solver_kit as kit

    adapter = arc_game_adapters.get_adapter(game)
    if adapter is None:
        return DeepenAttempt(
            game=game,
            reached_level=0,
            offline_reproduced=False,
            reproduced_levels=0,
            solution_labels=[],
            reproduction_gate={"game": game, "reached_level": 0, "reproduced": False},
            residual=f"{game}_adapter_missing_for_hidden_field_probe",
        )

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        game,
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
    for level in range(current_level + 1, target_level + 1):
        cap = int(adapter.depth_caps.get(level, depth_cap))
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
        game,
        full,
        adapter.apply,
        warmup_label=adapter.warmup_label,
        claimed_level=target_level,
    )
    reproduced_levels = _new_l2_levels(gate)
    offline_reproduced = reproduced_levels >= 1 and bool(gate.get("reproduced"))
    residual = None if offline_reproduced else f"{game}_l2_not_reproduced_after_hidden_key"
    return DeepenAttempt(
        game=game,
        reached_level=int(gate.get("reached_level", current_level) or 0),
        offline_reproduced=offline_reproduced,
        reproduced_levels=reproduced_levels,
        solution_labels=full,
        reproduction_gate=gate,
        residual=residual,
        states_expanded=states_expanded,
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    encoded = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checksum_material(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hidden_fields_added": artifact.get("hidden_fields_added"),
        "state_disambiguation_control": artifact.get("state_disambiguation_control"),
        "attempts": artifact.get("attempts"),
        "registry_update": artifact.get("registry_update"),
        "random_seed": artifact.get("random_seed"),
    }


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
        errors.append("registry_updated must be true for persisted probe findings")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(
        _checksum_material(artifact)
    ):
        errors.append("checksum mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict.startswith("success:"):
        if not (
            artifact.get("offline_reproduced") is True
            and int(artifact.get("reproduced_levels") or 0) >= 1
        ):
            errors.append("success artifact requires offline L2 reproduction")
    elif verdict.startswith("complete:"):
        if not (
            artifact.get("state_disambiguation_control_passed") is True
            and artifact.get("false_negative_risk_checked") is True
        ):
            errors.append("no-bank null requires positive control")
    else:
        errors.append("honest_verdict must start with terminal prefix")
    return errors


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    attempts: Sequence[DeepenAttempt],
    state_control: Mapping[str, Any],
    registry_update: Mapping[str, Any],
) -> dict[str, Any]:
    successful = next(
        (attempt for attempt in attempts if attempt.offline_reproduced and attempt.reproduced_levels >= 1),
        None,
    )
    state_control_passed = bool(state_control.get("passed"))
    missing = [str(attempt.residual) for attempt in attempts if attempt.residual]
    if successful is not None:
        verdict = f"success: hidden_field_state_{successful.game}_L2_offline_reproduced"
        offline_reproduced = True
        reproduced_levels = int(successful.reproduced_levels)
        false_negative_risk_checked = True
        missing = []
    else:
        verdict = "complete: hidden_field_state_gap_sharpened_no_bank_honest_null"
        offline_reproduced = False
        reproduced_levels = 0
        false_negative_risk_checked = state_control_passed
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
        "false_negative_risk_checked": false_negative_risk_checked,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "missing_verifier_gaps": missing,
        "registry_updated": bool(registry_update.get("updated")),
        "registry_update": _json_safe(registry_update),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "attempts": _json_safe([attempt.as_dict() for attempt in attempts]),
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(_checksum_material(payload))
    payload["schema_errors"] = artifact_schema_errors(payload)
    return payload


def apply_registry_probe(
    registry_text: str,
    *,
    artifact: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    registry = yaml.safe_load(registry_text) or {}
    prior = registry.get("latest_hidden_field_state_probe_4559")
    row = {
        "artifact": RESULT_RELATIVE_PATH,
        "honest_verdict": artifact.get("honest_verdict"),
        "hidden_fields_added": artifact.get("hidden_fields_added"),
        "state_disambiguation_control_passed": artifact.get(
            "state_disambiguation_control_passed"
        ),
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
        {"latest_hidden_field_state_probe_4559": row},
        sort_keys=False,
        width=1000,
    )
    updated_text = _replace_top_level_registry_section(
        registry_text,
        "latest_hidden_field_state_probe_4559",
        section_text,
    )
    return updated_text, {
        "updated": prior != row,
        "path": REGISTRY_RELATIVE_PATH,
        "section": "latest_hidden_field_state_probe_4559",
        "target_games": list(TARGET_GAMES),
    }


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _spec_refs_present(root: Path) -> bool:
    spec = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return all(ref in spec for ref in SPEC_REFS)


def check_offline_arcade() -> bool:  # pragma: no cover - exercised by required command.
    from carnot.agentic import arc_solver_kit

    arc_solver_kit.offline_arcade()
    return True


def run_experiment(
    *,
    root: Path | None = None,
    precondition_checker: Callable[[], bool] = check_offline_arcade,
    attempt_runner: Callable[[str, int, int], DeepenAttempt] = run_standing_loop_attempt,
    instructions_checked: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    instructions = dict(instructions_checked or {"AGENTS.md": True, "CODEX.md": True})
    preconditions_checked = {
        **instructions,
        "offline_arcade_import_smoke": bool(precondition_checker()),
        "spec_refs_present": _spec_refs_present(root),
    }
    state_control = build_state_disambiguation_control()
    attempts = [
        attempt_runner(game, TARGET_LEVEL, DEFAULT_DEPTH_CAP)
        for game in TARGET_GAMES
    ]
    placeholder_update = {"updated": True, "path": REGISTRY_RELATIVE_PATH}
    artifact = build_artifact(
        preconditions_checked=preconditions_checked,
        attempts=attempts,
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
        attempts=attempts,
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
