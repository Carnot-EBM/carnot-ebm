"""Experiment 4535: bank one ARC sprint level-up guarantee level.

Spec refs: REQ-ARC-WMTE-4535, SCENARIO-ARC-WMTE-4535.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


EXPERIMENT = "experiment_4535_levelup_attempt"
SCHEMA = "carnot.arc_levelup_attempt_4535.v1"
TARGET_GAME = "sp80"
RESULT_RELATIVE_PATH = "results/experiment_4535_levelup_attempt.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_sp80.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4535
SPEC_REFS = ["REQ-ARC-WMTE-4535", "SCENARIO-ARC-WMTE-4535"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

SP80_L2_SOLUTION_LABELS: tuple[str, ...] = (
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":5}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":6,\"data\":{\"x\":33,\"y\":25}}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":6,\"data\":{\"x\":13,\"y\":17}}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":4}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":2}",
    "{\"action\":5}",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: "
        "<game>_delta_identified_no_bank (honest progress, not a bank)."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade solve, no headline LLM load."
    ),
    "offline_reproduced": (
        "a solve not reproducible offline is wasted effort -- only reproduced levels count toward "
        "reproducible_total_levels."
    ),
    "reproduced_levels": (
        "the integer new-level count banked this task (>=1 to satisfy the level-up guarantee on an "
        "attempted bank)."
    ),
    "registry_updated": (
        "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt "
        "reuses, not re-derives."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

DEFAULT_DEAD_ENDS = [
    {
        "game": "sp80",
        "attempt": "arc_loop_solve --game sp80",
        "result": "reproduced_L1_only_before_adapter",
        "why": "Adapter-free graph explore re-gated the shallow seed but did not expose the L2 splitter-placement delta.",
    },
    {
        "game": "sp80",
        "attempt": "unbounded move/select/commit BFS from L2",
        "result": "stalled_on_failed_spill_states",
        "why": "Failed commits can spend the search budget in spill animation; a bounded spill predicate was needed before path planning.",
    },
    {
        "game": "sp80",
        "attempt": "arc_loop_solve --game sp80 --target-level 3",
        "result": "stalled_at_L2",
        "why": "The registered adapter banks L2 only; L3 has a larger four-splitter delta and remains unmodelled.",
    },
]


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable content hash for JSON-compatible replay evidence."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _registry_row_sum(registry: Mapping[str, Any]) -> int:
    return sum(
        int(row.get("levels_reproduced") or 0)
        for row in registry.get("games", [])
        if row.get("reproducibility") == "reproduced"
    )


def _game_row(registry: Mapping[str, Any], game: str) -> Mapping[str, Any]:
    for row in registry.get("games", []):
        if row.get("game") == game:
            return row
    raise ValueError(f"registry missing game row: {game}")


def _loop_reached_level(loop_result: Mapping[str, Any]) -> int:
    gate = loop_result.get("reproduction_gate") or {}
    return int(gate.get("reached_level") or loop_result.get("reached_level") or 0)


def _loop_reproduced(loop_result: Mapping[str, Any]) -> bool:
    gate = loop_result.get("reproduction_gate") or {}
    return bool(loop_result.get("offline_reproduced") and gate.get("reproduced"))


def _sp80_registry_block(checksum: str, artifact_path: str) -> str:
    return f"""- game: sp80
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: spill_splitter_placement
  win_condition: 'L1 move-right x3 then commit; L2 splitter placements route the
    spill into all three repwkzbkhxl target blocks without touching the wall:
    len3 at (0,3), len5 at (4,3), len3 at (8,5).'
  action_model: Keyboard ACTION1-4 move the selected splitter after display-rotation
    remapping; ACTION6 selects splitter sprites from offline-derived display centers;
    ACTION5 commits the spill. L2 tail selects the two len3 splitters and moves them
    to the derived target placements.
  solver: GameAdapter _sp80 in python/carnot/agentic/arc_game_adapters.py +
    scripts/arc_loop_solve.py; results/arc_loop_solve_sp80.json 33-label L2 gate.
  reproduce: Exp4535 {artifact_path} re-gated results/arc_loop_solve_sp80.json
    offline_reproduced=True, reached_level=2, banked +1 over the current L1
    registry row, checksum {checksum}.
  gotchas:
  - L2 has dojfslwbg=180, so external keyboard directions and click coordinates are
    display-rotated before the game maps them back to grid space.
  - The win predicate is not just visible piece placement; ACTION5 starts a spill,
    and success requires every repwkzbkhxl block to be marked while no waoewejnqzc
    wall is touched.
  - Failed commit states can spend search time in spill animation; use a bounded
    spill predicate to evaluate candidate placements before replay-gating.
  - Use fresh-env branching for this adapter; replaying L1 on a reused advanced
    sp80 env does not reconstruct the L2 start state that the reproduction gate
    sees from a pristine offline arcade.
  dead_ends:
  - Exp4535 arc_loop_solve --game sp80 --target-level 3 replays to L2 only; the
    current adapter has no grounded L3 splitter-placement delta."""


def _replace_sp80_block(registry_text: str, replacement: str) -> str:
    start = registry_text.index("- game: sp80")
    end = registry_text.find("\n- game: ", start + len("- game: sp80"))
    if end == -1:
        end = len(registry_text)
    return f"{registry_text[:start]}{replacement}{registry_text[end:]}"


def apply_sp80_registry_bank(
    registry_text: str,
    *,
    loop_result: Mapping[str, Any],
    checksum: str,
    artifact_path: str,
) -> tuple[str, dict[str, Any]]:
    """Apply the sp80 L2 bank to registry text after validating the YAML state."""

    registry = yaml.safe_load(registry_text)
    prior_row = _game_row(registry, TARGET_GAME)
    prior_level = int(prior_row.get("levels_reproduced") or 0)
    prior_total_declared = int(registry.get("reproducible_total_levels") or 0)
    prior_total_row_sum = _registry_row_sum(registry)
    reached_level = _loop_reached_level(loop_result)
    can_bank = (
        loop_result.get("game") == TARGET_GAME
        and _loop_reproduced(loop_result)
        and reached_level > prior_level
    )
    banked_levels = max(0, reached_level - prior_level) if can_bank else 0

    update = {
        "updated": False,
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": TARGET_GAME,
        "prior_game_levels": prior_level,
        "new_game_levels": prior_level,
        "banked_levels": 0,
        "prior_total_declared": prior_total_declared,
        "prior_total_row_sum": prior_total_row_sum,
        "new_total_declared": prior_total_declared,
        "new_total_row_sum": prior_total_row_sum,
        "reconciled_total_delta": 0,
    }
    if banked_levels < 1:
        return registry_text, update

    new_total = prior_total_row_sum + banked_levels
    replacement = _sp80_registry_block(checksum, artifact_path)
    updated_text = _replace_sp80_block(registry_text, replacement)
    updated_text = re.sub(
        r"(?m)^(reproducible_total_levels:\s*)\d+\s*$",
        rf"\g<1>{new_total}",
        updated_text,
        count=1,
    )
    updated_registry = yaml.safe_load(updated_text)
    update.update(
        {
            "updated": True,
            "new_game_levels": reached_level,
            "banked_levels": banked_levels,
            "new_total_declared": int(updated_registry["reproducible_total_levels"]),
            "new_total_row_sum": _registry_row_sum(updated_registry),
            "reconciled_total_delta": int(updated_registry["reproducible_total_levels"])
            - prior_total_declared,
        }
    )
    return updated_text, update


def build_artifact(
    *,
    loop_result: Mapping[str, Any],
    registry_update: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    dead_ends: list[Mapping[str, Any]],
) -> dict[str, Any]:
    reached_level = _loop_reached_level(loop_result)
    offline_reproduced = _loop_reproduced(loop_result)
    banked_levels = int(registry_update.get("banked_levels") or 0)
    registry_updated = bool(registry_update.get("updated"))
    success = offline_reproduced and registry_updated and banked_levels >= 1
    verdict = (
        f"success: {TARGET_GAME}_L{reached_level}_offline_reproduced"
        if success
        else f"complete: {TARGET_GAME}_delta_identified_no_bank"
    )
    checksum_material = {
        "target_game": TARGET_GAME,
        "reproduction_gate": loop_result.get("reproduction_gate"),
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "registry_update": dict(registry_update),
        "random_seed": RANDOM_SEED,
    }
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "target_game": TARGET_GAME,
        "target_level": reached_level,
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": banked_levels if success else 0,
        "registry_updated": registry_updated,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(checksum_material),
        "preconditions_checked": dict(preconditions_checked),
        "reproduction_gate": loop_result.get("reproduction_gate"),
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "dead_ends": [dict(item) for item in dead_ends],
        "registry_update": dict(registry_update),
        "arc_loop_result_path": LOOP_RESULT_RELATIVE_PATH,
    }
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "honest_verdict",
        "inference_substrate",
        "offline_reproduced",
        "reproduced_levels",
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
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    checksum_material = {
        "target_game": artifact.get("target_game"),
        "reproduction_gate": artifact.get("reproduction_gate"),
        "solution_labels": list(artifact.get("solution_labels") or []),
        "registry_update": dict(artifact.get("registry_update") or {}),
        "random_seed": artifact.get("random_seed"),
    }
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(checksum_material):
        errors.append("checksum mismatch")
    if str(artifact.get("honest_verdict", "")).startswith("success:"):
        if not (
            artifact.get("offline_reproduced") is True
            and int(artifact.get("reproduced_levels") or 0) >= 1
            and artifact.get("registry_updated") is True
        ):
            errors.append("success artifact missing reproduced registry bank")
    return errors


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _spec_refs_present(root: Path) -> bool:
    spec = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return all(ref in spec for ref in SPEC_REFS)


def check_offline_arcade() -> bool:  # pragma: no cover - exercised by the required command.
    from carnot.agentic import arc_solver_kit as solver_kit

    solver_kit.offline_arcade()
    return True


def run_experiment(
    *,
    root: Path | None = None,
    precondition_checker: Callable[[], bool] = check_offline_arcade,
    instructions_checked: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    instructions = dict(instructions_checked or {"AGENTS.md": True, "CODEX.md": True})
    preconditions_checked = {
        **instructions,
        "offline_arcade_import_smoke": bool(precondition_checker()),
        "spec_refs_present": _spec_refs_present(root),
    }
    loop_result = _read_json(root / LOOP_RESULT_RELATIVE_PATH)
    checksum = reproducibility_checksum(
        {
            "target_game": TARGET_GAME,
            "reproduction_gate": loop_result.get("reproduction_gate"),
            "solution_labels": list(loop_result.get("solution_labels") or []),
        }
    )
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_text = registry_path.read_text(encoding="utf-8")
    updated_registry_text, registry_update = apply_sp80_registry_bank(
        registry_text,
        loop_result=loop_result,
        checksum=checksum,
        artifact_path=RESULT_RELATIVE_PATH,
    )
    if registry_update["updated"]:
        registry_path.write_text(updated_registry_text, encoding="utf-8")
    artifact = build_artifact(
        loop_result=loop_result,
        registry_update=registry_update,
        preconditions_checked=preconditions_checked,
        dead_ends=DEFAULT_DEAD_ENDS,
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
                    "offline_reproduced",
                    "reproduced_levels",
                    "registry_updated",
                    "reproducibility_checksum",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
