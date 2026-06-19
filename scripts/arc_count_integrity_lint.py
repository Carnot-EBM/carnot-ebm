#!/usr/bin/env python3
"""Lint ARC reproduced counts and submission-package replay claims.

Spec refs: REQ-REPORT-4462, SCENARIO-REPORT-4462, SCENARIO-REPORT-4462-SUBMISSION.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SUBMISSION_PACKAGE_RELATIVE_PATH = Path("results/experiment_4460_submission_package_prep.json")
DEFAULT_REGISTRY_REPLAY_SPOT_CHECK = 3

ReplayEntryFn = Callable[[Mapping[str, Any], Path], Mapping[str, Any] | None]
ReplayRowFn = Callable[[Mapping[str, Any], Path], Mapping[str, Any] | None]


@dataclass(frozen=True)
class ArcCountIntegrityIssue:
    """One count-integrity lint failure for one path and optional game row."""

    path: Path
    kind: str
    detail: str
    game: str | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        payload = {
            "path": str(self.path),
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
        }
        if self.game is not None:
            payload["game"] = self.game
        return payload


def lint_registry_path(
    path: Path | str,
    *,
    replay_entry_fn: ReplayEntryFn | None = None,
    max_replay_games: int | None = DEFAULT_REGISTRY_REPLAY_SPOT_CHECK,
    root: Path | None = None,
) -> list[ArcCountIntegrityIssue]:
    """REQ-REPORT-4462: lint registry totals and sampled banked replay rows."""

    registry_path = Path(path)
    registry = _read_yaml_mapping(registry_path)
    return lint_registry_payload(
        registry_path,
        registry,
        replay_entry_fn=replay_entry_fn,
        max_replay_games=max_replay_games,
        root=root or _infer_repo_root(registry_path),
    )


def lint_registry_payload(
    path: Path | str,
    registry: Mapping[str, Any],
    *,
    replay_entry_fn: ReplayEntryFn | None = None,
    max_replay_games: int | None = DEFAULT_REGISTRY_REPLAY_SPOT_CHECK,
    root: Path = REPO_ROOT,
) -> list[ArcCountIntegrityIssue]:
    """Lint an already-loaded ARC solve registry mapping."""

    registry_path = Path(path)
    issues: list[ArcCountIntegrityIssue] = []
    entries = _reproduced_entries(registry)
    expected_total = sum(_as_int(entry.get("levels_reproduced")) for entry in entries)
    actual_total = _as_int(registry.get("reproducible_total_levels"))
    if actual_total != expected_total:
        issue_kind = "REGISTRY_TOTAL_MISMATCH"
        if _looks_like_provisional_inflation(registry, entries, actual_total, expected_total):
            issue_kind = "PROVISIONAL_INFLATION"
        issues.append(
            ArcCountIntegrityIssue(
                path=registry_path,
                kind=issue_kind,
                detail=(
                    "reproducible_total_levels must equal "
                    f"sum(levels_reproduced)={expected_total}; got {actual_total}. "
                    "Do not include levels_live_recorded or provisional_total_levels."
                ),
            )
        )

    if replay_entry_fn is None:
        replay_entry_fn = _default_registry_replay
    for entry in _registry_replay_sample(entries, max_replay_games):
        game = str(entry.get("game") or "")
        claimed = _as_int(entry.get("levels_reproduced"))
        try:
            replay = replay_entry_fn(entry, root)
        except Exception as exc:  # pragma: no cover - defensive real-env boundary
            issues.append(
                ArcCountIntegrityIssue(
                    path=registry_path,
                    kind="REGISTRY_REPLAY_EXCEPTION",
                    game=game,
                    detail=f"registry replay raised {type(exc).__name__}: {exc}",
                )
            )
            continue
        if replay is None:
            continue
        reached = _as_int(replay.get("reached_level"))
        reproduced = bool(replay.get("reproduced")) and reached >= claimed
        if not reproduced:
            issues.append(
                ArcCountIntegrityIssue(
                    path=registry_path,
                    kind="REGISTRY_REPLAY_OVERCLAIM",
                    game=game,
                    detail=f"{game} claimed {claimed} reproduced levels but replay reached {reached}.",
                )
            )
    return issues


def lint_submission_package_path(
    path: Path | str,
    *,
    replay_row_fn: ReplayRowFn | None = None,
    max_package_replays: int | None = None,
    root: Path | None = None,
) -> list[ArcCountIntegrityIssue]:
    """SCENARIO-REPORT-4462-SUBMISSION: lint a package artifact path."""

    package_path = Path(path)
    payload = _read_json_mapping(package_path)
    return lint_submission_package_payload(
        package_path,
        payload,
        replay_row_fn=replay_row_fn,
        max_package_replays=max_package_replays,
        root=root or _infer_repo_root(package_path),
    )


def lint_submission_package_payload(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    replay_row_fn: ReplayRowFn | None = None,
    max_package_replays: int | None = None,
    root: Path = REPO_ROOT,
) -> list[ArcCountIntegrityIssue]:
    """Lint an already-loaded A6 submission-package artifact mapping."""

    package_path = Path(path)
    issues: list[ArcCountIntegrityIssue] = []
    if payload.get("submitted_to_leaderboard") is not False:
        issues.append(
            ArcCountIntegrityIssue(
                path=package_path,
                kind="SUBMISSION_SUBMITTED_TO_LEADERBOARD",
                detail="submission-package guard requires submitted_to_leaderboard=false.",
            )
        )

    rows = payload.get("per_game_replay_validation")
    if not isinstance(rows, list):
        issues.append(
            ArcCountIntegrityIssue(
                path=package_path,
                kind="SUBMISSION_ROWS_NOT_LIST",
                detail="per_game_replay_validation must be a list.",
            )
        )
        return issues

    if replay_row_fn is None:
        replay_row_fn = _default_submission_replay

    valid_total = 0
    replay_budget = len(rows) if max_package_replays is None else max(0, int(max_package_replays))
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            issues.append(
                ArcCountIntegrityIssue(
                    path=package_path,
                    kind="SUBMISSION_ROW_NOT_MAPPING",
                    detail=f"per_game_replay_validation[{index}] must be a mapping.",
                )
            )
            continue
        game = str(row.get("game") or "")
        counted = _as_int(row.get("reproduced_levels"))
        if counted <= 0:
            continue
        metadata_ok, metadata_reason = _submission_row_metadata_ok(row, counted)
        replay_ok = metadata_ok
        if metadata_ok and replay_budget > 0:
            replay_budget -= 1
            try:
                replay = replay_row_fn(row, root)
            except Exception as exc:  # pragma: no cover - defensive real-env boundary
                issues.append(
                    ArcCountIntegrityIssue(
                        path=package_path,
                        kind="SUBMISSION_REPLAY_EXCEPTION",
                        game=game,
                        detail=f"package replay raised {type(exc).__name__}: {exc}",
                    )
                )
                replay_ok = False
            else:
                if replay is not None:
                    reached = _as_int(replay.get("reached_level"))
                    replay_ok = bool(replay.get("reproduced")) and reached >= counted
                    if not replay_ok:
                        issues.append(
                            ArcCountIntegrityIssue(
                                path=package_path,
                                kind="SUBMISSION_REPLAY_OVERCLAIM",
                                game=game,
                                detail=(
                                    f"{game} package counted {counted} levels but "
                                    f"replay reached {reached}."
                                ),
                            )
                        )
                    expected_sequence = replay.get("expected_action_sequence")
                    if expected_sequence is not None and _action_sequence(row) != [
                        str(label) for label in expected_sequence
                    ]:
                        replay_ok = False
                        issues.append(
                            ArcCountIntegrityIssue(
                                path=package_path,
                                kind="SUBMISSION_ACTION_SEQUENCE_MISMATCH",
                                game=game,
                                detail=f"{game} package action_sequence differs from banked replay plan.",
                            )
                        )
        if not metadata_ok:
            issues.append(
                ArcCountIntegrityIssue(
                    path=package_path,
                    kind="SUBMISSION_ROW_NOT_COUNTABLE",
                    game=game,
                    detail=f"{game} counts {counted} levels but {metadata_reason}.",
                )
            )
        if metadata_ok and replay_ok:
            valid_total += counted

    actual_total = _as_int(payload.get("total_reproduced_levels_in_package"))
    if actual_total != valid_total:
        issues.append(
            ArcCountIntegrityIssue(
                path=package_path,
                kind="SUBMISSION_TOTAL_MISMATCH",
                detail=(
                    "total_reproduced_levels_in_package must equal the sum of "
                    f"replay-valid rows ({valid_total}); got {actual_total}."
                ),
            )
        )
    return issues


def lint_paths(
    paths: Iterable[Path | str],
    *,
    skip_replay: bool = False,
    max_registry_replays: int | None = DEFAULT_REGISTRY_REPLAY_SPOT_CHECK,
    max_package_replays: int | None = None,
) -> list[ArcCountIntegrityIssue]:
    """Lint explicit registry and package paths."""

    issues: list[ArcCountIntegrityIssue] = []
    replay_entry_fn: ReplayEntryFn | None = (lambda _entry, _root: None) if skip_replay else None
    replay_row_fn: ReplayRowFn | None = (lambda _row, _root: None) if skip_replay else None
    for raw_path in paths:
        path = Path(raw_path)
        if path.name == REGISTRY_RELATIVE_PATH.name:
            issues.extend(
                lint_registry_path(
                    path,
                    replay_entry_fn=replay_entry_fn,
                    max_replay_games=0 if skip_replay else max_registry_replays,
                )
            )
        elif path.name == SUBMISSION_PACKAGE_RELATIVE_PATH.name:
            issues.extend(
                lint_submission_package_path(
                    path,
                    replay_row_fn=replay_row_fn,
                    max_package_replays=0 if skip_replay else max_package_replays,
                )
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    """Run the count-integrity lint CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Registry/package paths to lint.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--skip-replay", action="store_true", help="Run metadata checks only.")
    parser.add_argument(
        "--max-registry-replays",
        type=int,
        default=DEFAULT_REGISTRY_REPLAY_SPOT_CHECK,
        help="Maximum reproduced registry rows to spot-check through reproduce().",
    )
    parser.add_argument(
        "--max-package-replays",
        type=int,
        default=None,
        help="Maximum package rows to replay; default replays every counted row.",
    )
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.paths]
    if not paths:
        paths = [
            REPO_ROOT / REGISTRY_RELATIVE_PATH,
            REPO_ROOT / SUBMISSION_PACKAGE_RELATIVE_PATH,
        ]
    issues = lint_paths(
        paths,
        skip_replay=args.skip_replay,
        max_registry_replays=args.max_registry_replays,
        max_package_replays=args.max_package_replays,
    )
    report = {"ok": not issues, "issue_count": len(issues), "issues": [i.to_dict() for i in issues]}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:  # pragma: no cover - JSON mode is the conductor-facing path.
        for issue in issues:
            game = f" [{issue.game}]" if issue.game else ""
            print(f"{issue.path}{game}: {issue.kind}: {issue.detail}")
    return 1 if issues else 0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _infer_repo_root(path: Path) -> Path:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "ops").is_dir() and (parent / "results").is_dir():
            return parent
    if resolved.parent.name in {"ops", "results"}:
        return resolved.parent.parent
    return REPO_ROOT


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _reproduced_entries(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = registry.get("games")
    if not isinstance(rows, list):
        return []
    entries: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if _as_int(row.get("levels_reproduced")) > 0:
            entries.append(dict(row))
    return entries


def _looks_like_provisional_inflation(
    registry: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    actual_total: int,
    expected_total: int,
) -> bool:
    if actual_total <= expected_total:
        return False
    if _as_int(registry.get("provisional_total_levels")) > 0:
        return True
    live_augmented_total = sum(
        max(_as_int(entry.get("levels_reproduced")), _as_int(entry.get("levels_live_recorded")))
        for entry in entries
    )
    return actual_total == live_augmented_total and live_augmented_total > expected_total


def _registry_replay_sample(
    entries: Sequence[Mapping[str, Any]],
    max_replay_games: int | None,
) -> list[Mapping[str, Any]]:
    ordered = sorted(
        entries,
        key=lambda entry: (
            _as_int(entry.get("levels_live_recorded")) <= _as_int(entry.get("levels_reproduced")),
            -_as_int(entry.get("levels_reproduced")),
            str(entry.get("game") or ""),
        ),
    )
    if max_replay_games is None:
        return ordered
    return ordered[: max(0, int(max_replay_games))]


def _submission_row_metadata_ok(row: Mapping[str, Any], counted: int) -> tuple[bool, str]:
    if row.get("replays_ok") is not True:
        return False, "replays_ok is not true"
    if row.get("env_matched") is not True:
        return False, "env_matched is not true"
    if not _action_sequence(row):
        return False, "action_sequence is empty"
    result = row.get("reproduction_result")
    if not isinstance(result, Mapping):
        return False, "reproduction_result is missing"
    reached = _as_int(result.get("reached_level"))
    if result.get("reproduced") is not True or reached < counted:
        return False, f"embedded reproduction_result reached {reached}, below counted {counted}"
    return True, ""


def _action_sequence(row: Mapping[str, Any]) -> list[str]:
    sequence = row.get("action_sequence")
    if not isinstance(sequence, list):
        return []
    return [str(label) for label in sequence]


def _default_registry_replay(  # pragma: no cover - real offline ARC env boundary.
    entry: Mapping[str, Any],
    root: Path,
) -> Mapping[str, Any]:
    from carnot import experiment_4460_submission_package_prep as exp4460

    return exp4460.reproduce_registry_entry(entry, root)


def _registry_entry_for_game(root: Path, game: str) -> dict[str, Any]:
    registry = _read_yaml_mapping(root / REGISTRY_RELATIVE_PATH)
    for entry in _reproduced_entries(registry):
        if entry.get("game") == game:
            return entry
    return {"game": game}


def _default_submission_replay(  # pragma: no cover - real offline ARC env boundary.
    row: Mapping[str, Any],
    root: Path,
) -> Mapping[str, Any]:
    from carnot import experiment_4460_submission_package_prep as exp4460
    from carnot.agentic import arc_solver_kit

    game = str(row.get("game") or "")
    counted = _as_int(row.get("reproduced_levels"))
    sequence = _action_sequence(row)
    entry = _registry_entry_for_game(root, game)
    if counted > 0:
        entry["levels_reproduced"] = counted
    plan = exp4460.resolve_replay_plan(entry, root)
    result = dict(
        arc_solver_kit.reproduce(
            game,
            sequence,
            plan.apply_fn,
            warmup_label=plan.warmup_label,
            claimed_level=counted,
        )
    )
    result["expected_action_sequence"] = list(sequence)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
