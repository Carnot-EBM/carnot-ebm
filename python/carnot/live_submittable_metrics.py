"""Shared live-submittable ARC metric helpers.

Spec refs: REQ-CAPSTONE-4586, SCENARIO-CAPSTONE-4586.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
DEFAULT_PACKAGE_RELATIVE_PATH = "results/experiment_4585_submission_package_integration_gate.json"
FALLBACK_PACKAGE_RELATIVE_PATHS = (
    DEFAULT_PACKAGE_RELATIVE_PATH,
    "results/experiment_4580_submission_package_live_gap_close.json",
)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive schema boundary
        return default


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive file boundary
        return {}
    return loaded if isinstance(loaded, dict) else {}  # pragma: no cover - defensive file boundary


def _read_yaml(path: Path) -> JsonDict:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary
        return {}
    return loaded if isinstance(loaded, dict) else {}  # pragma: no cover - defensive file boundary


def default_package_path(root: Path | str = REPO_ROOT) -> str:
    """Return the latest refreshed package path available in this checkout."""

    root_path = Path(root)
    for rel_path in FALLBACK_PACKAGE_RELATIVE_PATHS:
        if (root_path / rel_path).exists():
            return rel_path
    return DEFAULT_PACKAGE_RELATIVE_PATH


def _reproduced_registry_levels(registry: Mapping[str, Any]) -> dict[str, int]:
    rows = registry.get("games")
    if not isinstance(rows, list):
        return {}
    levels: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game") or "")
        reproduced = row.get("reproducibility") == "reproduced"
        level = _as_int(row.get("levels_reproduced"))
        if game and reproduced and level > 0:
            levels[game] = max(levels.get(game, 0), level)
    return levels


def _reproducible_total(registry: Mapping[str, Any], levels: Mapping[str, int]) -> int:
    total = _as_int(registry.get("reproducible_total_levels"))
    return total if total > 0 else sum(levels.values())


def _package_rows(package: Mapping[str, Any]) -> dict[str, JsonDict]:
    rows = package.get("package_manifest")
    if not isinstance(rows, list):
        rows = package.get("per_game_submittable")
    if not isinstance(rows, list):
        return {}
    out: dict[str, JsonDict] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game") or "")
        if game:
            out[game] = dict(row)
    return out


def _has_replayable_trajectory(row: Mapping[str, Any]) -> bool:
    if row.get("has_trajectory") is True:
        return True
    trajectory_path = str(row.get("trajectory_path") or "")
    action_count = _as_int(row.get("action_count") or row.get("trajectory_action_count"))
    return bool(trajectory_path and action_count > 0)


def _has_env_adaptive_resolver(row: Mapping[str, Any]) -> bool:
    if row.get("has_env_adaptive_resolver") is True:
        return True
    return bool(str(row.get("adaptive_solver") or row.get("adaptive_resolver") or ""))


def _env_matchable(row: Mapping[str, Any]) -> bool:
    if row.get("env_matched") is True or row.get("env_match") is True:
        return True
    return bool(str(row.get("env_match_basis") or ""))


def _candidate_levels(
    *,
    registry_level: int,
    package_row: Mapping[str, Any],
) -> tuple[int, int, int]:
    package_level = _as_int(package_row.get("levels") or package_row.get("submittable_level"))
    offline_level = _as_int(package_row.get("offline_reproduced_level"))
    return package_level, offline_level, min(registry_level, package_level, offline_level)


def _exclusion_reason(
    *,
    package_row: Mapping[str, Any] | None,
    offline_level: int,
    has_submission_path: bool,
    env_matchable: bool,
) -> str:
    if package_row is None:
        return "missing_package_row"
    if offline_level <= 0:
        return "not_offline_reproduced"
    if not has_submission_path:
        return "missing_trajectory_or_adaptive_resolver"
    if not env_matchable:
        return "env_not_matchable"
    return ""


def compute_live_submittable_metrics(
    root: Path | str = REPO_ROOT,
    *,
    registry: Mapping[str, Any] | None = None,
    package: Mapping[str, Any] | None = None,
    registry_path: str = REGISTRY_RELATIVE_PATH,
    package_path: str | None = None,
) -> JsonDict:
    """Compute the leaderboard-submittable subset from the registry and package."""

    root_path = Path(root)
    selected_package_path = package_path or default_package_path(root_path)
    registry_payload = dict(registry or _read_yaml(root_path / registry_path))
    package_payload = dict(package or _read_json(root_path / selected_package_path))
    registry_levels = _reproduced_registry_levels(registry_payload)
    package_by_game = _package_rows(package_payload)

    per_game: list[JsonDict] = []
    for game, registry_level in sorted(registry_levels.items()):
        package_row = package_by_game.get(game)
        package_level = 0
        offline_level = 0
        candidate = 0
        has_trajectory = False
        has_adaptive = False
        env_ok = False
        if package_row is not None:
            package_level, offline_level, candidate = _candidate_levels(
                registry_level=registry_level,
                package_row=package_row,
            )
            has_trajectory = _has_replayable_trajectory(package_row)
            has_adaptive = _has_env_adaptive_resolver(package_row)
            env_ok = _env_matchable(package_row)

        has_submission_path = has_trajectory or has_adaptive
        reason = _exclusion_reason(
            package_row=package_row,
            offline_level=offline_level,
            has_submission_path=has_submission_path,
            env_matchable=env_ok,
        )
        included = not reason and candidate > 0
        per_game.append(
            {
                "game": game,
                "registry_reproduced_level": registry_level,
                "package_claimed_level": package_level,
                "offline_reproduced_level": offline_level,
                "has_replayable_trajectory": has_trajectory,
                "has_env_adaptive_resolver": has_adaptive,
                "env_matchable": env_ok,
                "included": bool(included),
                "submittable_level": candidate if included else 0,
                "exclusion_reason": reason,
            }
        )

    live_count = sum(row["submittable_level"] for row in per_game)
    reproducible_total = _reproducible_total(registry_payload, registry_levels)
    gap = reproducible_total - live_count
    return {
        "reproducible_total_levels": reproducible_total,
        "live_submittable_level_count": live_count,
        "reproducible_vs_submittable_gap": gap,
        "live_submittable_subset_of_reproducible": live_count <= reproducible_total,
        "per_game_live_submittable": per_game,
        "registry_path": registry_path,
        "refreshed_package_path": selected_package_path,
    }
