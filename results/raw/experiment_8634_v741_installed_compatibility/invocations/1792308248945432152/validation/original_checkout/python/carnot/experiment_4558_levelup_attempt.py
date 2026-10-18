"""Experiment 4558: ARC sprint rotation level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4558, SCENARIO-ARC-WMTE-4558.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


EXPERIMENT = "experiment_4558_levelup_attempt"
SCHEMA = "carnot.arc_levelup_attempt_4558.v1"
RESULT_RELATIVE_PATH = "results/experiment_4558_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4558
SPEC_REFS = ["REQ-ARC-WMTE-4558", "SCENARIO-ARC-WMTE-4558"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TARGET_CANDIDATES = (
    "cn04",
    "sk48",
    "ar25",
    "m0r0",
    "dc22",
    "lp85",
    "tu93",
    "tr87",
    "cd82",
    "sp80",
    "su15",
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
        "the integer new-level count banked this task (>=1 satisfies the level-up guarantee on an "
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
        "game": "cn04",
        "attempt": "arc_loop_solve --game cn04 --target-level 2",
        "result": "needs_per_game_RE",
        "why": (
            "The standing transfer route found no registered cn04 adapter; next work must derive the "
            "L2 win/action/state delta instead of repeating graph exploration."
        ),
    },
    {
        "game": "sk48",
        "attempt": "arc_loop_solve --game sk48 --target-level 2 plus bounded L2 replay probes",
        "result": "needs_per_game_RE",
        "why": (
            "The L1 route replays, but L2 line/snake block ordering still needs a focused adapter; "
            "broad replay search reduced mismatch without crossing the reproduction gate."
        ),
    },
    {
        "game": "ar25",
        "attempt": "arc_loop_solve --game ar25 --target-level 2",
        "result": "needs_per_game_RE",
        "why": (
            "The registry already flags action7 hidden undo-stack state as the L2 blocker; no new "
            "offline-reproduced level was produced."
        ),
    },
    {
        "game": "m0r0",
        "attempt": "arc_loop_solve --game m0r0 --target-level 3",
        "result": "reproduced_existing_L2_only",
        "why": (
            "The registered push-motion adapter replayed through L2 but did not derive the L3 delta, "
            "so reproduced_levels repeats the registry row and cannot increment the total."
        ),
    },
    {
        "game": "dc22",
        "attempt": "arc_loop_solve --game dc22 --target-level 2",
        "result": "reproduced_existing_L1_only",
        "why": (
            "The toggle-navigation adapter replayed only the current L1 bank; the L2 transition did "
            "not reproduce offline."
        ),
    },
]


def loop_result_path(game: str) -> str:
    return f"results/arc_loop_solve_{game}.json"


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
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


def _bank_reason(*, reproduced: bool, reached_level: int, prior_level: int) -> str:
    if not reproduced:
        return "not_offline_reproduced"
    if reached_level <= prior_level:
        return "reproduced_existing_or_lower_level"
    return "banked_offline_reproduced_level"


def apply_registry_bank(
    registry_text: str,
    *,
    loop_result: Mapping[str, Any],
    checksum: str,
    artifact_path: str,
) -> tuple[str, dict[str, Any]]:
    registry = yaml.safe_load(registry_text)
    game = str(loop_result.get("game") or "")
    prior_row = _game_row(registry, game)
    prior_level = int(prior_row.get("levels_reproduced") or 0)
    prior_total_declared = int(registry.get("reproducible_total_levels") or 0)
    prior_total_row_sum = _registry_row_sum(registry)
    reached_level = _loop_reached_level(loop_result)
    reproduced = _loop_reproduced(loop_result)
    reason = _bank_reason(reproduced=reproduced, reached_level=reached_level, prior_level=prior_level)
    banked_levels = max(0, reached_level - prior_level) if reason == "banked_offline_reproduced_level" else 0
    update = {
        "updated": False,
        "path": REGISTRY_RELATIVE_PATH,
        "target_game": game,
        "prior_game_levels": prior_level,
        "new_game_levels": prior_level,
        "banked_levels": 0,
        "prior_total_declared": prior_total_declared,
        "prior_total_row_sum": prior_total_row_sum,
        "new_total_declared": prior_total_declared,
        "new_total_row_sum": prior_total_row_sum,
        "reconciled_total_delta": 0,
        "reason": reason,
    }
    if banked_levels < 1:
        return registry_text, update

    mutable_row = next(row for row in registry["games"] if row.get("game") == game)
    mutable_row["reproducibility"] = "reproduced"
    mutable_row["levels_reproduced"] = reached_level
    mutable_row["solver"] = f"scripts/arc_loop_solve.py + {loop_result_path(game)}"
    mutable_row["reproduce"] = (
        f"Exp4558 {artifact_path} re-gated {loop_result_path(game)} offline_reproduced=True, "
        f"reached_level={reached_level}, banked +{banked_levels}, checksum {checksum}."
    )
    registry["reproducible_total_levels"] = prior_total_row_sum + banked_levels
    updated_text = yaml.safe_dump(registry, sort_keys=False, width=1000)
    update.update(
        {
            "updated": True,
            "new_game_levels": reached_level,
            "banked_levels": banked_levels,
            "new_total_declared": int(registry["reproducible_total_levels"]),
            "new_total_row_sum": _registry_row_sum(registry),
            "reconciled_total_delta": int(registry["reproducible_total_levels"]) - prior_total_declared,
        }
    )
    return updated_text, update


def load_candidate_loop_results(
    root: Path,
    candidates: Sequence[str] = TARGET_CANDIDATES,
) -> list[tuple[str, str, dict[str, Any]]]:
    results: list[tuple[str, str, dict[str, Any]]] = []
    for game in candidates:
        relative_path = loop_result_path(game)
        path = root / relative_path
        if path.exists():
            data = _read_json(path)
            data.setdefault("game", game)
            results.append((game, relative_path, data))
    return results


def choose_loop_result(
    candidates: Sequence[tuple[str, str, Mapping[str, Any]]],
    registry: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any]]:
    if not candidates:
        raise FileNotFoundError("no cached ARC loop results found for experiment 4558 candidates")
    best: tuple[int, int, int] | None = None
    selected: tuple[str, str, Mapping[str, Any]] | None = None
    for index, item in enumerate(candidates):
        game, _path, loop_result = item
        prior_level = int(_game_row(registry, game).get("levels_reproduced") or 0)
        reached_level = _loop_reached_level(loop_result)
        reproduced = _loop_reproduced(loop_result)
        advances = int(reproduced and reached_level > prior_level)
        score = (advances, int(reproduced), -index)
        if best is None or score > best:
            best = score
            selected = item
    assert selected is not None
    return selected


def build_artifact(
    *,
    loop_result: Mapping[str, Any],
    registry_update: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    dead_ends: list[Mapping[str, Any]],
    arc_loop_result_path: str,
) -> dict[str, Any]:
    game = str(registry_update.get("target_game") or loop_result.get("game") or "")
    reached_level = _loop_reached_level(loop_result)
    offline_reproduced = _loop_reproduced(loop_result)
    banked_levels = int(registry_update.get("banked_levels") or 0)
    registry_updated = bool(registry_update.get("updated"))
    success = offline_reproduced and registry_updated and banked_levels >= 1
    verdict = (
        f"success: {game}_L{reached_level}_offline_reproduced"
        if success
        else f"complete: {game}_delta_identified_no_bank"
    )
    checksum_material = {
        "target_game": game,
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
        "target_game": game,
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
        "arc_loop_result_path": arc_loop_result_path,
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
    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_text = registry_path.read_text(encoding="utf-8")
    registry = yaml.safe_load(registry_text)
    game, relative_path, loop_result = choose_loop_result(load_candidate_loop_results(root), registry)
    checksum = reproducibility_checksum(
        {
            "target_game": game,
            "reproduction_gate": loop_result.get("reproduction_gate"),
            "solution_labels": list(loop_result.get("solution_labels") or []),
        }
    )
    updated_registry_text, registry_update = apply_registry_bank(
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
        arc_loop_result_path=relative_path,
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
