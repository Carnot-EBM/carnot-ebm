"""Exp 4436: deepen one reproduced ARC game and consolidate solver primitives.

Spec refs: REQ-REPORT-4436, SCENARIO-REPORT-4436.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from carnot.agentic import arc_solver_kit as kit


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4436_deepen_plus_primitive_consolidation.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
DEEPENED_GAME = "tu93"
RANDOM_SEED = 4436
SPEC_REFS = ("REQ-REPORT-4436", "SCENARIO-REPORT-4436")
TU93_L4_SOURCE_RELATIVE_PATH = "results/experiment_4361_e3_deeper_high_headroom_games.json"
TU93_L5_SUFFIX_ACTIONS = (3, 3, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 2, 2, 4, 2, 2, 4, 4, 4, 1, 2, 1, 1, 2, 1, 3)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproduced_levels",
    "offline_reproduced",
    "primitives_consolidated",
    "no_regression",
    "random_seed",
    "reproducibility_checksum",
    "verifier_is_oracle",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed",
    "reproduced_levels": "bare int; the +1 deeper level, reproduction-gated",
    "offline_reproduced": "the gate",
    "primitives_consolidated": (
        "list of {operator, derived_from_games} -- the reusable generic "
        "operators the live solver now composes"
    ),
    "no_regression": (
        "bare bool: every prior reproducible solve still reproduces after the "
        "refactor -- additive-only guarantee"
    ),
    "random_seed": "determinism",
    "reproducibility_checksum": "content hash",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registry(root: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {"games": []}
    return loaded if isinstance(loaded, dict) else {"games": []}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("complete:", "success:", "passed:", "shipped:", "blocked_"))


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def primitives_as_rows(primitives: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for primitive in primitives:
        if hasattr(primitive, "as_dict"):
            raw = primitive.as_dict()
        elif isinstance(primitive, Mapping):
            raw = dict(primitive)
        else:
            raw = {"operator": str(primitive), "derived_from_games": []}
        rows.append(
            {
                "operator": str(raw.get("operator", "")),
                "derived_from_games": [str(game) for game in raw.get("derived_from_games", [])],
            }
        )
    return rows


def _no_regression(prior_reproductions: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        bool(row.get("reproduced")) and _as_int(row.get("reached_level")) >= _as_int(row.get("claimed_level"))
        for row in prior_reproductions
    )


def _verdict(
    *,
    deepened_game: str,
    target_level: int,
    offline_reproduced: bool,
    new_levels: int,
    no_regression: bool,
    preconditions_checked: Mapping[str, Any],
) -> str:
    if preconditions_checked.get("offline_env_files_present") is not True:
        return "blocked_offline_env_files_missing"
    if offline_reproduced and new_levels >= 1 and no_regression:
        return f"success: {deepened_game}_L{target_level}_deepened_primitives_consolidated"
    if offline_reproduced and new_levels >= 1:
        return f"complete: {deepened_game}_L{target_level}_deepened_but_regression_detected"
    return f"complete: {deepened_game}_no_deeper_level_primitives_consolidated"


def compute_reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "deepened_game": artifact.get("deepened_game"),
            "prior_reproduced_levels": artifact.get("prior_reproduced_levels"),
            "target_level": artifact.get("target_level"),
            "deepened_reproduction": artifact.get("deepened_reproduction"),
            "prior_reproductions": artifact.get("prior_reproductions"),
            "primitives_consolidated": artifact.get("primitives_consolidated"),
            "no_regression": artifact.get("no_regression"),
            "random_seed": artifact.get("random_seed"),
        }
    )


def preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:
    env_dir = root / "environment_files"
    return {
        "offline_env_files_present": env_dir.is_dir() and any(env_dir.iterdir()),
        "offline_env_files_path": str(env_dir),
        "registry_path": REGISTRY_RELATIVE_PATH,
        "no_3090_inference": True,
        "leaderboard_submission": False,
    }


def prior_best_level(root: Path = REPO_ROOT, game: str = DEEPENED_GAME) -> int:
    registry = _load_registry(root)
    for entry in registry.get("games", []) if isinstance(registry.get("games"), list) else []:
        if isinstance(entry, Mapping) and entry.get("game") == game:
            return _as_int(entry.get("levels_reproduced"))
    return 0


def build_artifact(
    *,
    deepened_game: str,
    prior_reproduced_levels: int,
    target_level: int,
    deepened_reproduction: Mapping[str, Any],
    prior_reproductions: Sequence[Mapping[str, Any]],
    primitives: Sequence[Any],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    reached = _as_int(deepened_reproduction.get("reached_level"))
    offline_reproduced = bool(deepened_reproduction.get("reproduced")) and reached >= int(target_level)
    new_levels = max(0, reached - int(prior_reproduced_levels)) if offline_reproduced else 0
    no_regression = _no_regression(prior_reproductions)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4436_deepen_plus_primitive_consolidation",
        "schema": "carnot.exp4436.deepen_plus_primitive_consolidation.v1",
        "deepened_game": deepened_game,
        "prior_reproduced_levels": int(prior_reproduced_levels),
        "target_level": int(target_level),
        "honest_verdict": _verdict(
            deepened_game=deepened_game,
            target_level=int(target_level),
            offline_reproduced=offline_reproduced,
            new_levels=new_levels,
            no_regression=no_regression,
            preconditions_checked=preconditions_checked,
        ),
        "reproduced_levels": reached if offline_reproduced else int(prior_reproduced_levels),
        "new_levels_reproduced": new_levels,
        "offline_reproduced": offline_reproduced,
        "deepened_reproduction": dict(deepened_reproduction),
        "prior_reproductions": [dict(row) for row in prior_reproductions],
        "primitives_consolidated": primitives_as_rows(primitives),
        "no_regression": no_regression,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "submitted_to_leaderboard": False,
        "no_3090_inference": True,
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("no_regression")) is not bool:
        errors.append("no_regression must be bare bool")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    rows = artifact.get("primitives_consolidated")
    if not isinstance(rows, list) or not rows:
        errors.append("primitives_consolidated must be non-empty list")
    elif any(
        not isinstance(row, Mapping) or not row.get("operator") or not row.get("derived_from_games")
        for row in rows
    ):
        errors.append("primitives_consolidated rows require derived_from_games")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("offline_reproduced") is not True:
            errors.append("success verdict requires offline_reproduced true")
        if artifact.get("no_regression") is not True:
            errors.append("success verdict requires no_regression true")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def deepened_solution_labels(root: Path = REPO_ROOT) -> list[str]:
    """Prior tu93 L4 replay plus the reproduction-gated L5 maze suffix."""

    source = _load_json(root / TU93_L4_SOURCE_RELATIVE_PATH)
    prefix: list[str] = []
    for row in source.get("per_target_scorecard", []):
        if isinstance(row, Mapping) and row.get("game") == DEEPENED_GAME:
            raw_plan = row.get("plan") or []
            prefix = [str(label) for label in raw_plan]
            break
    suffix = [json.dumps({"action": action}) for action in TU93_L5_SUFFIX_ACTIONS]
    return prefix + suffix


def reproduce_deepened_tu93(root: Path = REPO_ROOT, *, claimed_level: int | None = None) -> dict[str, Any]:
    """Replay the deepened tu93 solution through arc_solver_kit.reproduce."""

    from carnot.agentic.arc_game_adapters import get_adapter

    adapter = get_adapter(DEEPENED_GAME)
    solution = deepened_solution_labels(root)
    if adapter is None or not solution:
        return {
            "game": DEEPENED_GAME,
            "claimed_level": claimed_level,
            "reached_level": 0,
            "reproduced": False,
            "source": TU93_L4_SOURCE_RELATIVE_PATH,
            "mode": "missing_tu93_adapter_or_solution",
        }
    result = dict(
        kit.reproduce(
            DEEPENED_GAME,
            [str(label) for label in solution],
            adapter.apply,
            warmup_label=adapter.warmup_label,
            claimed_level=claimed_level,
        )
    )
    result["source"] = TU93_L4_SOURCE_RELATIVE_PATH
    result["solution_action_count"] = len(solution)
    result["deepened_suffix_action_count"] = len(TU93_L5_SUFFIX_ACTIONS)
    return result


def prior_reproduction_gate_results(root: Path = REPO_ROOT) -> list[dict[str, Any]]:  # pragma: no cover
    """Replay every prior counted registry solve through the existing audit gate."""

    from carnot import experiment_4426_arc_registry_repro_audit as audit

    rows: list[dict[str, Any]] = []
    registry = _load_registry(root)
    for entry in registry.get("games", []) if isinstance(registry.get("games"), list) else []:
        if not isinstance(entry, Mapping):
            continue
        claimed = _as_int(entry.get("levels_reproduced"))
        if entry.get("reproducibility") != "reproduced" or claimed <= 0:
            continue
        result = dict(audit.reproduce_registry_entry(entry, root))
        result.setdefault("game", entry.get("game"))
        result.setdefault("claimed_level", claimed)
        rows.append(result)
    return rows


def run(root: Path = REPO_ROOT, *, write: bool = True) -> dict[str, Any]:  # pragma: no cover
    checked = preconditions(root)
    prior = prior_best_level(root, DEEPENED_GAME)
    target = prior + 1
    if checked["offline_env_files_present"]:
        deepened = reproduce_deepened_tu93(root, claimed_level=target)
        prior_results = prior_reproduction_gate_results(root)
    else:
        deepened = {"game": DEEPENED_GAME, "claimed_level": target, "reached_level": prior, "reproduced": False}
        prior_results = []
    artifact = build_artifact(
        deepened_game=DEEPENED_GAME,
        prior_reproduced_levels=prior,
        target_level=target,
        deepened_reproduction=deepened,
        prior_reproductions=prior_results,
        primitives=kit.primitive_operator_registry(),
        preconditions_checked=checked,
    )
    if write:
        write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run(REPO_ROOT, write=True)
    print(f"{artifact['honest_verdict']} wrote {RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
