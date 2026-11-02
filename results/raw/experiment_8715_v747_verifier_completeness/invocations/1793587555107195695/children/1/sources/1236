"""Experiment 4841: object-identity perception probe on real ARC frames.

Spec refs: REQ-ARC-WMTE-4841,
SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE,
SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL,
SCENARIO-ARC-WMTE-4841-LIVE-PATH-REACHABLE.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from carnot.agentic.arc_object_identity_perception import (
    TrackerConfig,
    config_fingerprint,
    measure_correspondence,
    track_object_identities,
    track_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4841_object_identity_perception_probe"
RESULT_RELATIVE_PATH = "results/experiment_4841_object_identity_perception_probe.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
TARGET_GAMES = ("lp85", "r11l", "tu93")
RANDOM_SEED = 4841
REAL_SOURCE_KINDS = {"banked_replay", "transition_corpus"}
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; recovery is success_object_identity_perception_recovers_goal_grounding, "
            "a null is complete_object_identity_unrecoverable_from_rendered_grid_deeper_finding."
        )
    },
    "measured_on_real_frames": {
        "principle": (
            "true iff the correspondence scores are measured on REAL lp85/r11l/tu93 frames, NOT "
            "synthetic -- the mechanic-template synthetic-only-pass trap; a false here is a NON-TEST."
        )
    },
    "per_game_correspondence": {
        "principle": (
            "per-game mapping game -> (shape_motion_score, color_centroid_baseline_score, n_frames) "
            "-- the quantitative recovery measure vs the color-centroid comparator on the same frames."
        )
    },
    "positive_control_tu93_passed": {
        "principle": (
            "the tu93 visible-goal control must yield stable player+goal identity tracks -- a "
            "Phase-Prototype positive control so a global null is not a harness artifact."
        )
    },
    "games_with_recovery": {
        "principle": (
            "count of the three test games (lp85, r11l, tu93) where the shape/motion tracker materially "
            "beats color-centroid -- >=2 for a PASS."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "true -- execution-grounded structural perception, NOT an oracle-distinct moat "
            "(circularity discipline); this probe is about grounding."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the module is importable by the live agent (arc_orphan_solver_lint passes) -- a "
            "perception layer the live agent cannot reach is wasted effort."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- an offline perception probe on banked frames, NOT a live first-win; "
            "declared honestly (not live_agent_self_discovery)."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference if any LLM induction runs, else "
            "verifier_ensemble_against_cached_candidates -- declare what actually ran (the probe is "
            "mostly CPU segmentation; do not claim live inference if none ran)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/frame-availability checks so a missing-resource run emits blocked_, never "
            "a fabricated correspondence score."
        )
    },
    "random_seed": {
        "principle": "determinism for any stochastic segmentation/tracking step + the baseline."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (frames, tracker params, baseline) so a replication catches drift."
        )
    },
}


@dataclass(frozen=True)
class FrameSequence:
    """Real rendered frames and the source used to obtain them."""

    game: str
    frames: list[np.ndarray]
    source_path: str
    source_kind: str


def _as_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _load_actions(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    for keys in (
        ("solution",),
        ("trajectory",),
        ("solve_trace", "actions"),
        ("solver_trace", "actions"),
        ("plan_executed_detail", "plan_result", "executed_steps"),
    ):
        current: Any = payload
        for key in keys:
            current = current.get(key) if isinstance(current, dict) else None
            if current is None:
                break
        if isinstance(current, list) and current:
            return [row for row in current if isinstance(row, dict)]
    return []


def _normalize_action(action: Mapping[str, Any]) -> tuple[int | None, dict[str, int] | None]:
    action_id = action.get("action")
    if "data" in action:
        data = action.get("data")
        clean_data = (
            {"x": int(data["x"]), "y": int(data["y"])}
            if isinstance(data, Mapping) and "x" in data and "y" in data
            else None
        )
        return (int(action_id) if action_id is not None else None), clean_data
    x = action.get("x", action.get("world_x"))
    y = action.get("y", action.get("world_y"))
    has_xy = x is not None and y is not None
    if action_id is None and has_xy:
        action_id = 6
    data = {"x": int(x), "y": int(y)} if has_xy else None
    return (int(action_id) if action_id is not None else None), data


def _frame_checksum(sequence: FrameSequence) -> str:
    digest = hashlib.sha256()
    digest.update(sequence.game.encode())
    digest.update(sequence.source_kind.encode())
    digest.update(sequence.source_path.encode())
    for frame in sequence.frames:
        arr = np.ascontiguousarray(np.asarray(frame, dtype=np.int16))
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return "sha256:" + digest.hexdigest()


def load_banked_replay_frames(root: Path, game: str) -> FrameSequence | None:
    """Replay the banked solution through the offline environment and return rendered grids."""

    action_path = root / "results" / "arc3_live_banked_trajectories" / f"{game}.json"
    actions = _load_actions(action_path)
    if not actions:
        return _load_transition_corpus_frames(root, game)

    try:
        from arcengine import GameAction
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_agi3_world_model import grid_of

        arcade = kit.offline_arcade()
        env = arcade.make(game, scorecard_id=arcade.open_scorecard())
        frame = env.reset()
        frames = [np.asarray(grid_of(frame), dtype=np.int16).copy()]
        for action in actions:
            action_id, data = _normalize_action(action)
            if action_id is None:
                continue
            frame = env.step(
                getattr(GameAction, f"ACTION{action_id}"),
                data=data,
                reasoning={"policy": "experiment_4841_replay"},
            )
            if frame is None:
                break
            frames.append(np.asarray(grid_of(frame), dtype=np.int16).copy())
    except Exception:
        return _load_transition_corpus_frames(root, game)

    if len(frames) < 2:
        return _load_transition_corpus_frames(root, game)
    return FrameSequence(
        game=game,
        frames=frames,
        source_path=str(action_path.relative_to(root)),
        source_kind="banked_replay",
    )


def _load_transition_corpus_frames(root: Path, game: str) -> FrameSequence | None:
    corpus_path = root / "data" / "arc_transition_corpus" / f"{game}.npz"
    if not corpus_path.exists():
        return None
    try:
        data = np.load(corpus_path, allow_pickle=False)
        grids = data["grids"]
        next_grids = data["next_grids"]
    except (OSError, KeyError, ValueError):
        return None
    frames: list[np.ndarray] = []
    for before, after in zip(grids[:64], next_grids[:64], strict=False):
        frames.append(np.asarray(before, dtype=np.int16).copy())
        frames.append(np.asarray(after, dtype=np.int16).copy())
    if len(frames) < 2:
        return None
    return FrameSequence(
        game=game,
        frames=frames,
        source_path=str(corpus_path.relative_to(root)),
        source_kind="transition_corpus",
    )


def _offline_arcade_available() -> bool:
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def run_arc_orphan_solver_lint(root: Path) -> dict[str, Any]:
    command = [sys.executable, str(root / "scripts" / "arc_orphan_solver_lint.py")]
    proc = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def default_live_path_checker(root: Path) -> bool:
    return bool(run_arc_orphan_solver_lint(root)["passed"])


def evaluate_tu93_positive_control(
    frames: Iterable[Any],
    config: TrackerConfig | None = None,
) -> dict[str, Any]:
    """Identify one moving player-like track and one stable goal-like track."""

    cfg = config or TrackerConfig()
    tracked = track_object_identities(frames, cfg)
    summary = track_summary(tracked)
    if not tracked.frames:
        return {
            "passed": False,
            "player_track_id": None,
            "goal_track_id": None,
            "player_motion": 0.0,
            "goal_persistence": 0.0,
            "track_summary": {},
        }

    player_candidates = [
        (track_id, row)
        for track_id, row in summary.items()
        if float(row["persistence"]) >= cfg.positive_control_min_persistence
        and float(row["total_motion"]) >= 1.0
        and float(row["mean_pixels"]) <= 128.0
    ]
    player_candidates.sort(
        key=lambda item: (
            -float(item[1]["total_motion"]),
            -float(item[1]["persistence"]),
            float(item[1]["mean_pixels"]),
            item[0],
        )
    )
    player = player_candidates[0] if player_candidates else None

    goal_candidates = [
        (track_id, row)
        for track_id, row in summary.items()
        if (player is None or track_id != player[0])
        and float(row["persistence"]) >= cfg.positive_control_min_persistence
        and float(row["total_motion"]) <= 0.25
        and float(row["mean_pixels"]) <= 256.0
    ]
    goal_candidates.sort(
        key=lambda item: (
            -float(item[1]["persistence"]),
            float(item[1]["mean_pixels"]),
            item[0],
        )
    )
    goal = goal_candidates[0] if goal_candidates else None

    return {
        "passed": bool(player and goal),
        "player_track_id": player[0] if player else None,
        "goal_track_id": goal[0] if goal else None,
        "player_motion": float(player[1]["total_motion"]) if player else 0.0,
        "goal_persistence": float(goal[1]["persistence"]) if goal else 0.0,
        "track_summary": {str(track_id): row for track_id, row in sorted(summary.items())},
    }


def _build_preconditions(
    *,
    offline_ok: bool,
    offline_error: str | None,
    frame_sequences: Mapping[str, FrameSequence],
    missing_games: Iterable[str],
    live_path_result: bool | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(live_path_result, Mapping):
        live_path_payload = dict(live_path_result)
    else:
        live_path_payload = {"passed": bool(live_path_result)}
    return {
        "offline_arcade": {
            "ok": bool(offline_ok),
            "check": 'from carnot.agentic import arc_solver_kit as k; k.offline_arcade()',
            "error": offline_error,
        },
        "frame_sources": {
            game: {
                "present": game in frame_sequences,
                "source_path": frame_sequences[game].source_path if game in frame_sequences else None,
                "source_kind": frame_sequences[game].source_kind if game in frame_sequences else None,
                "n_frames": len(frame_sequences[game].frames) if game in frame_sequences else 0,
            }
            for game in TARGET_GAMES
        },
        "available_games": sorted(frame_sequences),
        "missing_games": sorted(missing_games),
        "arc_orphan_solver_lint": live_path_payload,
    }


def _empty_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "honest_verdict": verdict,
        "measured_on_real_frames": False,
        "per_game_correspondence": {},
        "positive_control_tu93_passed": False,
        "positive_control_tu93": {
            "passed": False,
            "player_track_id": None,
            "goal_track_id": None,
        },
        "games_with_recovery": 0,
        "verifier_is_oracle": True,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "tracker_params": config_fingerprint(),
        "baseline": "same-color nearest-centroid over active changed components",
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(1.0, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    frame_sequences: Mapping[str, FrameSequence],
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    config: TrackerConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TrackerConfig()
    per_game: dict[str, dict[str, Any]] = {}
    frame_checksums: dict[str, str] = {}
    for game in TARGET_GAMES:
        sequence = frame_sequences.get(game)
        if sequence is None:
            continue
        measurement = measure_correspondence(sequence.frames, cfg).as_dict()
        measurement["source_path"] = sequence.source_path
        measurement["source_kind"] = sequence.source_kind
        measurement["frame_checksum"] = _frame_checksum(sequence)
        per_game[game] = measurement
        frame_checksums[game] = measurement["frame_checksum"]

    positive = (
        evaluate_tu93_positive_control(frame_sequences["tu93"].frames, cfg)
        if "tu93" in frame_sequences
        else {
            "passed": False,
            "player_track_id": None,
            "goal_track_id": None,
            "player_motion": 0.0,
            "goal_persistence": 0.0,
            "track_summary": {},
        }
    )
    games_with_recovery = sum(1 for row in per_game.values() if bool(row.get("recovered")))
    measured_on_real_frames = bool(per_game) and all(
        row.get("source_kind") in REAL_SOURCE_KINDS for row in per_game.values()
    )
    passed = bool(positive.get("passed")) and games_with_recovery >= 2
    verdict = (
        "success_object_identity_perception_recovers_goal_grounding"
        if passed
        else "complete_object_identity_unrecoverable_from_rendered_grid_deeper_finding"
    )

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "honest_verdict": verdict,
        "measured_on_real_frames": measured_on_real_frames,
        "per_game_correspondence": per_game,
        "positive_control_tu93_passed": bool(positive.get("passed")),
        "positive_control_tu93": positive,
        "games_with_recovery": games_with_recovery,
        "verifier_is_oracle": True,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "tracker_params": config_fingerprint(cfg),
        "baseline": "same-color nearest-centroid over active changed components",
        "retire_if_same_verdict": True,
        "frame_checksums": frame_checksums,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(1.0, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean["reproducibility_checksum"] = ""
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "honest_verdict",
        "measured_on_real_frames",
        "per_game_correspondence",
        "positive_control_tu93_passed",
        "games_with_recovery",
        "verifier_is_oracle",
        "live_path_reachable",
        "solve_provenance",
        "inference_substrate",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
    )
    for field in required:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    blocked = verdict.startswith("blocked_")
    if not (
        verdict.startswith("success_")
        or verdict.startswith("complete_")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance must be development_proxy")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match REQ-ARC-WMTE-4841")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must be 4841")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles drifted")

    per_game = artifact.get("per_game_correspondence")
    if not isinstance(per_game, Mapping):
        errors.append("per_game_correspondence must be a mapping")
        per_game = {}
    if not blocked and artifact.get("measured_on_real_frames") is not True:
        errors.append("measured_on_real_frames must be true for non-blocked artifacts")
    if not blocked and not per_game:
        errors.append("per_game_correspondence must be non-empty for non-blocked artifacts")
    if not blocked and artifact.get("live_path_reachable") is not True:
        errors.append("live_path_reachable must be true for non-blocked artifacts")

    recovered_count = 0
    for game, row in per_game.items():
        if game not in TARGET_GAMES:
            errors.append(f"unexpected per-game row {game}")
        if not isinstance(row, Mapping):
            errors.append(f"{game} row must be a mapping")
            continue
        shape = row.get("shape_motion_score")
        color = row.get("color_centroid_baseline_score")
        n_frames = row.get("n_frames")
        if not isinstance(shape, (int, float)) or not 0.0 <= float(shape) <= 1.0:
            errors.append(f"{game} shape_motion_score must be in [0, 1]")
        if not isinstance(color, (int, float)) or not 0.0 <= float(color) <= 1.0:
            errors.append(f"{game} color_centroid_baseline_score must be in [0, 1]")
        if not isinstance(n_frames, int) or n_frames < 2:
            errors.append(f"{game} n_frames must be >= 2")
        if row.get("source_kind") not in REAL_SOURCE_KINDS:
            errors.append(f"{game} source_kind must be real-frame backed")
        if bool(row.get("recovered")):
            recovered_count += 1
    if artifact.get("games_with_recovery") != recovered_count:
        errors.append("games_with_recovery must match per-game recovered rows")
    if verdict.startswith("success_"):
        if artifact.get("positive_control_tu93_passed") is not True:
            errors.append("success requires positive_control_tu93_passed=true")
        if recovered_count < 2:
            errors.append("success requires games_with_recovery >= 2")

    checksum = artifact.get("reproducibility_checksum")
    expected = "sha256:" + payload_checksum(artifact)
    if checksum != expected:
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    offline_arcade_checker: Callable[[], bool] | None = None,
    frame_sequence_provider: Callable[[str], FrameSequence | None] | None = None,
    live_path_checker: Callable[[Path], bool | dict[str, Any]] | None = None,
    now: Callable[[], float] | None = None,
    write: bool = True,
    config: TrackerConfig | None = None,
) -> dict[str, Any]:
    clock = now or time.monotonic
    start = clock()
    root = Path(root)
    checker = offline_arcade_checker or _offline_arcade_available
    provider = frame_sequence_provider or (lambda game: load_banked_replay_frames(root, game))
    live_checker = live_path_checker or (lambda path: run_arc_orphan_solver_lint(path))

    offline_ok = False
    offline_error: str | None = None
    try:
        offline_ok = bool(checker())
    except Exception as exc:
        offline_error = f"{type(exc).__name__}: {exc}"

    live_path_result = live_checker(root)
    live_path_reachable = (
        bool(live_path_result.get("passed")) if isinstance(live_path_result, Mapping) else bool(live_path_result)
    )

    if not offline_ok:
        checks = _build_preconditions(
            offline_ok=False,
            offline_error=offline_error,
            frame_sequences={},
            missing_games=TARGET_GAMES,
            live_path_result=live_path_result,
        )
        artifact = _empty_artifact(
            "blocked_offline_arcade_missing",
            preconditions_checked=checks,
            live_path_reachable=live_path_reachable,
            duration_s=clock() - start,
        )
        if write:
            write_artifact(artifact, root=root)
        return artifact

    sequences: dict[str, FrameSequence] = {}
    missing: list[str] = []
    for game in TARGET_GAMES:
        try:
            sequence = provider(game)
        except Exception:
            sequence = None
        if sequence is None or len(sequence.frames) < 2:
            missing.append(game)
            continue
        sequences[game] = sequence

    checks = _build_preconditions(
        offline_ok=True,
        offline_error=None,
        frame_sequences=sequences,
        missing_games=missing,
        live_path_result=live_path_result,
    )

    if not sequences:
        artifact = _empty_artifact(
            "blocked_no_offline_frames",
            preconditions_checked=checks,
            live_path_reachable=live_path_reachable,
            duration_s=clock() - start,
        )
    else:
        artifact = build_artifact(
            frame_sequences=sequences,
            preconditions_checked=checks,
            live_path_reachable=live_path_reachable,
            duration_s=clock() - start,
            config=config,
        )
    if write:
        write_artifact(artifact, root=root)
    return artifact


def main() -> int:
    artifact = run(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
