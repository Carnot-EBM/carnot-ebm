"""Exp 4477: per-game online discriminative pruning for ARC StepwiseExplorer.

Spec refs: REQ-PHASE4-4477, SCENARIO-PHASE4-4477.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4477_per_game_online_discriminative.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4477
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
DEFAULT_HELD_OUT_GAMES = ("bp35", "lp85", "tu93")
DEFAULT_ACTION_BUDGET = 80

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "offline_reproduced",
    "reproduced_levels",
    "preconditions_checked",
    "baseline_solve_rate",
    "online_solve_rate",
    "solve_rate_delta",
    "baseline_actions_to_first_levelup",
    "online_actions_to_first_levelup",
    "actions_to_first_levelup_delta",
    "per_game_results",
    "online_verifier",
    "random_seed",
    "reproducibility_checksum",
    "submitted_to_leaderboard",
    "field_principles",
    "spec_refs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "MUST start with a terminal prefix complete:/complete_/success:/success_/"
            "passed:/passed_/shipped:/shipped_ so the reconciler classifies it as "
            "terminal (Verdict Terminal-Prefix Discipline)."
        )
    },
    "inference_substrate": {
        "principle": (
            "explicit declaration (live_llm_inference | "
            "verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) "
            "so adversarial_verify applies the right floor."
        )
    },
    "offline_reproduced": {
        "principle": (
            "a solve not reproducible offline is wasted effort -- only reproduced levels "
            "count (ARC Solve Reproducibility)."
        )
    },
    "reproduced_levels": {
        "principle": (
            "headline metric reproducible_total_levels grows monotonically; report the "
            "count banked, real-env-confirmed."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified before launching; pre-empts the "
            "silent-missing-resource fabrication mode."
        )
    },
}


@dataclass(frozen=True)
class Preconditions:
    offline_fixtures_present: bool = False
    arc_solver_kit_importable: bool = False
    arcengine_importable: bool = False
    submitted_to_leaderboard: bool = False
    action_budget: int = DEFAULT_ACTION_BUDGET
    held_out_games: tuple[str, ...] = DEFAULT_HELD_OUT_GAMES
    blocker: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.offline_fixtures_present
            and self.arc_solver_kit_importable
            and self.arcengine_importable
            and not self.submitted_to_leaderboard
            and not self.blocker
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["held_out_games"] = list(self.held_out_games)
        return data


@dataclass(frozen=True)
class PerGameComparison:
    game: str
    baseline_solved: bool
    online_solved: bool
    baseline_actions_to_first_levelup: int | None
    online_actions_to_first_levelup: int | None
    baseline_reached_level: int
    online_reached_level: int
    baseline_actions_spent: int
    online_actions_spent: int
    online_verifier: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "baseline_solved": bool(self.baseline_solved),
            "online_solved": bool(self.online_solved),
            "baseline_actions_to_first_levelup": self.baseline_actions_to_first_levelup,
            "online_actions_to_first_levelup": self.online_actions_to_first_levelup,
            "baseline_reached_level": int(self.baseline_reached_level),
            "online_reached_level": int(self.online_reached_level),
            "baseline_actions_spent": int(self.baseline_actions_spent),
            "online_actions_spent": int(self.online_actions_spent),
            "online_verifier": dict(self.online_verifier),
        }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _rate(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if value) / len(values), 10)


def _mean_action_metric(
    rows: Sequence[PerGameComparison], field: str, spent_field: str
) -> int | None:
    if not rows:
        return None
    values = []
    for row in rows:
        value = getattr(row, field)
        values.append(int(value) if value is not None else int(getattr(row, spent_field)))
    return int(round(sum(values) / len(values)))


def _first_blocker(preconditions: Preconditions) -> str | None:
    if preconditions.offline_fixtures_present is not True:
        return "offline_fixtures_missing"
    if preconditions.arc_solver_kit_importable is not True:
        return "arc_solver_kit_unavailable"
    if preconditions.arcengine_importable is not True:
        return "arcengine_unavailable"
    if preconditions.submitted_to_leaderboard is True:
        return "leaderboard_submission_policy"
    if preconditions.blocker:
        return str(preconditions.blocker)
    return None


def _verdict(*, solve_rate_delta: float, preconditions: Preconditions) -> str:
    blocker = _first_blocker(preconditions)
    if blocker is not None:
        return f"complete: per_game_online_discriminative_blocked_{blocker}"
    if solve_rate_delta > 0.0:
        return "success: per_game_online_discriminative_improves_solve_rate"
    return "complete: per_game_online_discriminative_no_solve_rate_gain"


def _aggregate_online_verifier(rows: Sequence[PerGameComparison]) -> dict[str, Any]:
    diagnostics = [dict(row.online_verifier) for row in rows]
    return {
        "per_game_head": "DiscriminativeVerifier",
        "trained_games": sum(1 for row in diagnostics if row.get("trained") is True),
        "positive_samples": sum(int(row.get("positive_samples") or 0) for row in diagnostics),
        "negative_samples": sum(int(row.get("negative_samples") or 0) for row in diagnostics),
        "frontier_pruned": sum(int(row.get("frontier_pruned") or 0) for row in diagnostics),
        "training_scope": "per_game_online_only",
    }


def build_artifact(
    *,
    per_game_results: Sequence[PerGameComparison] = (),
    preconditions_checked: Preconditions | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    preconditions = preconditions_checked or Preconditions()
    rows = list(per_game_results)
    baseline_rate = _rate([row.baseline_solved for row in rows])
    online_rate = _rate([row.online_solved for row in rows])
    solve_delta = round(online_rate - baseline_rate, 10)
    baseline_actions = _mean_action_metric(
        rows,
        "baseline_actions_to_first_levelup",
        "baseline_actions_spent",
    )
    online_actions = _mean_action_metric(
        rows,
        "online_actions_to_first_levelup",
        "online_actions_spent",
    )
    action_delta = (
        None
        if baseline_actions is None or online_actions is None
        else int(online_actions - baseline_actions)
    )
    row_json = [row.to_json() for row in rows]
    online_verifier = _aggregate_online_verifier(rows)
    checksum_payload = {
        "rows": row_json,
        "preconditions": preconditions.to_json(),
        "online_verifier": online_verifier,
        "random_seed": int(random_seed),
    }
    artifact = {
        "honest_verdict": _verdict(solve_rate_delta=solve_delta, preconditions=preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "preconditions_checked": preconditions.to_json(),
        "baseline_solve_rate": baseline_rate,
        "online_solve_rate": online_rate,
        "solve_rate_delta": solve_delta,
        "baseline_actions_to_first_levelup": baseline_actions,
        "online_actions_to_first_levelup": online_actions,
        "actions_to_first_levelup_delta": action_delta,
        "per_game_results": row_json,
        "online_verifier": online_verifier,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _sha256(checksum_payload),
        "submitted_to_leaderboard": False,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-PHASE4-4477", "SCENARIO-PHASE4-4477"],
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if isinstance(verdict, str) and verdict.startswith("partial:"):
        errors.append("honest_verdict must not use partial prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be verifier_ensemble_against_cached_candidates")
    if type(artifact.get("offline_reproduced")) is not bool:
        errors.append("offline_reproduced must be bare bool")
    if type(artifact.get("reproduced_levels")) is not int:
        errors.append("reproduced_levels must be bare int")
    if (
        artifact.get("offline_reproduced") is True
        and int(artifact.get("reproduced_levels") or 0) < 1
    ):
        errors.append("offline_reproduced true requires reproduced_levels >= 1")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a dict")
    for field in ("baseline_solve_rate", "online_solve_rate", "solve_rate_delta"):
        if not isinstance(artifact.get(field), (float, int)):
            errors.append(f"{field} must be numeric")
    for field in (
        "baseline_actions_to_first_levelup",
        "online_actions_to_first_levelup",
        "actions_to_first_levelup_delta",
    ):
        if artifact.get(field) is not None and type(artifact.get(field)) is not int:
            errors.append(f"{field} must be int or null")
    if not isinstance(artifact.get("per_game_results"), list):
        errors.append("per_game_results must be a list")
    if not isinstance(artifact.get("online_verifier"), Mapping):
        errors.append("online_verifier must be a dict")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not _checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be 64-char sha256 hex")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be dict")
    else:
        for field, expected in FIELD_PRINCIPLES.items():
            if principles.get(field) != expected:
                errors.append(f"field_principles.{field} must match REQ-PHASE4-4477")
    if artifact.get("spec_refs") != ["REQ-PHASE4-4477", "SCENARIO-PHASE4-4477"]:
        errors.append("spec_refs must cite REQ-PHASE4-4477 and SCENARIO-PHASE4-4477")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_preconditions(
    root: Path = REPO_ROOT,
    *,
    games: Sequence[str] = DEFAULT_HELD_OUT_GAMES,
    action_budget: int = DEFAULT_ACTION_BUDGET,
) -> Preconditions:  # pragma: no cover - exercised by the real artifact run.
    fixtures = [
        (Path(root) / "environment_files" / game).is_dir()
        and any((Path(root) / "environment_files" / game).iterdir())
        for game in games
    ]
    try:
        from carnot.agentic import arc_solver_kit  # noqa: F401

        solver_importable = True
    except Exception:
        solver_importable = False
    try:
        import arcengine  # noqa: F401

        arcengine_importable = True
    except Exception:
        arcengine_importable = False
    blocker = "" if all(fixtures) else "offline_fixtures_missing"
    return Preconditions(
        offline_fixtures_present=all(fixtures),
        arc_solver_kit_importable=solver_importable,
        arcengine_importable=arcengine_importable,
        submitted_to_leaderboard=False,
        action_budget=int(action_budget),
        held_out_games=tuple(str(game) for game in games),
        blocker=blocker,
    )


def _play_one(
    game: str,
    *,
    online_discriminative: bool,
    action_budget: int,
) -> dict[str, Any]:  # pragma: no cover - exercised by the real artifact run.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import StepwiseExplorer, _level_of

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    explorer = StepwiseExplorer(
        target_levels=1,
        value_head=None,
        value_weight=0.0,
        search_mode="depth_first_ride",
        online_discriminative=online_discriminative,
        discriminative_min_positives=1,
        discriminative_min_negatives=1,
        discriminative_prune_threshold=0.12,
    )
    frames: list[Any] = []
    latest = None
    start_level = 0
    actions = 0
    first_levelup_actions = None
    reached = 0
    for _ in range(action_budget):
        if explorer.is_done(frames, latest):
            break
        kind, data = explorer.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            start_level = _level_of(latest)
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            actions += 1
        frames.append(latest)
        reached = _level_of(latest)
        if first_levelup_actions is None and reached > start_level:
            first_levelup_actions = actions
            break
        if latest is None:
            break
    return {
        "solved": first_levelup_actions is not None,
        "actions_to_first_levelup": first_levelup_actions,
        "reached_level": int(reached),
        "actions_spent": int(actions),
        "online_verifier": explorer.online_discriminator_diagnostics(),
    }


def evaluate_offline_ab(
    *,
    games: Sequence[str] = DEFAULT_HELD_OUT_GAMES,
    action_budget: int = DEFAULT_ACTION_BUDGET,
) -> list[PerGameComparison]:  # pragma: no cover - exercised by the real artifact run.
    rows: list[PerGameComparison] = []
    for game in games:
        baseline = _play_one(game, online_discriminative=False, action_budget=action_budget)
        online = _play_one(game, online_discriminative=True, action_budget=action_budget)
        rows.append(
            PerGameComparison(
                game=str(game),
                baseline_solved=bool(baseline["solved"]),
                online_solved=bool(online["solved"]),
                baseline_actions_to_first_levelup=baseline["actions_to_first_levelup"],
                online_actions_to_first_levelup=online["actions_to_first_levelup"],
                baseline_reached_level=int(baseline["reached_level"]),
                online_reached_level=int(online["reached_level"]),
                baseline_actions_spent=int(baseline["actions_spent"]),
                online_actions_spent=int(online["actions_spent"]),
                online_verifier=dict(online["online_verifier"]),
            )
        )
    return rows


def run(
    *,
    root: Path = REPO_ROOT,
    games: Sequence[str] = DEFAULT_HELD_OUT_GAMES,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    write: bool = True,
) -> dict[str, Any]:  # pragma: no cover - exercised by the real artifact run.
    preconditions = check_preconditions(root, games=games, action_budget=action_budget)
    rows = (
        []
        if not preconditions.ok
        else evaluate_offline_ab(games=games, action_budget=action_budget)
    )
    artifact = build_artifact(per_game_results=rows, preconditions_checked=preconditions)
    if write:
        write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
