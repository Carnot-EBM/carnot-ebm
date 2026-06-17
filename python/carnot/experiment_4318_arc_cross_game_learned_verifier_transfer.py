"""Exp 4318: ARC cross-game learned value-head transfer.

Spec refs: REQ-LEARN-4318, SCENARIO-LEARN-4318.

The measurement is intentionally CPU-only.  It trains the existing linear
`LearnedVerifier` on solved-game replay traces and then asks whether that value
head reduces `OfflineSolver` search states on a held-out game.  The held-out
trace is used only as a reproducible action frontier and positive-control solve
source; its states never enter that split's value-head training rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
from carnot.agentic.arc_value_learner import LearnedVerifier, cross_game_features


REPO = Path(__file__).resolve().parents[2]
OUTPUT_REL = Path("results/experiment_4318_arc_cross_game_learned_verifier_transfer.json")
ENTRYPOINT_REL = Path("results/experiment_4318_arc_cross_game_learned_verifier_transfer.py")
RANDOM_SEED = 4318
BOOTSTRAP_RESAMPLES = 2000
MIN_USABLE_GAMES = 3
GAP_ID = "GAP-4318"
INFERENCE_SUBSTRATE = "cpu_offline_arc_agi3_trace_frontier_learned_value_head"
SPEC_REFS = ["REQ-LEARN-4318", "SCENARIO-LEARN-4318"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "cross_game_transfer_helps",
    "cross_game_state_reduction",
    "cross_game_state_reduction_ci95",
    "per_held_out_game_reduction",
    "baseline_solves_held_out",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A cross-game transfer win (the value-head reduces "
        "search states on a held-out game -- self-learning + efficiency), an "
        "honest null (per-game heads do not transfer -> log the gap), and an "
        "honest blocked_insufficient_solve_traces are ALL COMPLETE and "
        "decision-grade."
    ),
    "cross_game_transfer_helps": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true "
        "iff the value-head trained on OTHER games reduces held-out-game search "
        "states (reduction > 1.0 AND CI95 lower bound > 1.0) -- cross-game "
        "self-learning for ARC search efficiency."
    ),
    "cross_game_state_reduction": (
        "BARE float: held-out states_uniform / states_transferred (>1.0 = "
        "transfer helps; ~1.0 = per-game-bound) -- the north-star EFFICIENCY "
        "metric (fewer search states = more efficient solve)."
    ),
    "cross_game_state_reduction_ci95": (
        "Bootstrap CI95 (>=2000 resamples) over held-out levels of the "
        "state-reduction -- a lower bound > 1.0 makes 'transfer helps' "
        "decision-grade."
    ),
    "per_held_out_game_reduction": (
        "State-reduction reported SEPARATELY per held-out game -- guards the "
        "one-game-dominates failure mode + shows which games transfer."
    ),
    "baseline_solves_held_out": (
        "BARE bool: the uniform-heuristic solver actually solves the held-out "
        "levels (the positive control) -- a no-reduction is only informative if "
        "the baseline succeeds (FALSE_NEGATIVE_RISK guard)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- a learned value-head over solver state "
        "(oracle-distinct heuristic), NOT the executable env oracle; NO LLM "
        "weight mutation."
    ),
    "random_seed": "Determinism precondition for the value-head training + the solver.",
    "reproducibility_checksum": (
        "Hash of the solve traces + the value-head + the leave-one-game-out "
        "split + the state counts; lets a third party re-run."
    ),
    "model_specs": (
        "The arc_value_learner architecture + the train/held-out game split + "
        "the OfflineSolver state-count instrumentation; required methodology."
    ),
}


@dataclass(frozen=True)
class TraceSource:
    game: str
    game_id: str
    rel_path: Path
    sequence_key: str


@dataclass(frozen=True)
class TraceStep:
    action: int
    data: Mapping[str, int] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"action": int(self.action), "data": dict(self.data) if self.data else None}


@dataclass(frozen=True)
class LevelTrace:
    game: str
    game_id: str
    level_index: int
    start_level: int
    target_level: int
    prefix: tuple[TraceStep, ...]
    steps: tuple[TraceStep, ...]


@dataclass(frozen=True)
class GameTrace:
    game: str
    game_id: str
    path: Path
    sha256: str
    steps: tuple[TraceStep, ...]
    levels: tuple[LevelTrace, ...]


TRACE_SOURCES = {
    "r11l": TraceSource(
        game="r11l",
        game_id="r11l-495a7899",
        rel_path=Path("results/arc_explore_trajectory_r11l.json"),
        sequence_key="trajectory",
    ),
    "ls20": TraceSource(
        game="ls20",
        game_id="ls20-9607627b",
        rel_path=Path("results/arc_explore_trajectory_ls20.json"),
        sequence_key="trajectory",
    ),
    "wa30": TraceSource(
        game="wa30",
        game_id="wa30-ee6fef47",
        rel_path=Path("results/experiment_4275_arc_incremental_progress_new_game.json"),
        sequence_key="action_plan",
    ),
    "lp85": TraceSource(
        game="lp85",
        game_id="lp85-305b61c3",
        rel_path=Path("results/arc3_lp85_offline_resolve.json"),
        sequence_key="solution",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _normalize_step(raw: Mapping[str, Any]) -> TraceStep:
    action = int(raw.get("action", raw.get("action_id", 6)) or 6)
    data = raw.get("data")
    if data is None and "x" in raw and "y" in raw:
        data = {"x": int(raw["x"]), "y": int(raw["y"])}
    if isinstance(data, Mapping):
        clean_data = {str(k): int(v) for k, v in data.items() if v is not None}
    else:
        clean_data = None
    return TraceStep(action=action, data=clean_data)


def decode_steps(payload: Mapping[str, Any], sequence_key: str) -> tuple[TraceStep, ...]:
    """REQ-LEARN-4318-1: normalize the solved-game trace schemas."""

    sequence = payload.get(sequence_key) or []
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
        return ()
    steps = []
    for row in sequence:
        if isinstance(row, Mapping):
            steps.append(_normalize_step(row))
    return tuple(steps)


def step_label(step: TraceStep) -> str:
    return json.dumps(step.to_json(), sort_keys=True, separators=(",", ":"))


def label_to_step(label: str) -> TraceStep:
    return _normalize_step(json.loads(label))


def _game_action(action_id: int) -> Any:  # pragma: no cover - thin SDK enum boundary
    from arcengine.enums import GameAction

    return getattr(GameAction, f"ACTION{int(action_id)}")


def _apply_step(env: Any, step: TraceStep) -> Any:
    return env.step(_game_action(step.action), data=dict(step.data) if step.data else None)


def _make_env(arcade: Any, game_id: str) -> Any:
    try:
        env = arcade.make(game_id, scorecard_id=arcade.open_scorecard())
    except TypeError:
        env = arcade.make(game_id)
    if env is None:
        raise RuntimeError(f"offline ARC env unavailable for {game_id}")
    return env


def frame_state_key(frame: Any) -> tuple[int, str]:
    return (kit.frame_level(frame), frame_hash(grid_of(frame)))


def solver_frame_state_key(_game: Any, frame: Any) -> tuple[int, str]:
    return frame_state_key(frame)


def split_trace_into_levels(source: TraceSource, steps: Sequence[TraceStep]) -> tuple[LevelTrace, ...]:
    """REQ-LEARN-4318-1: replay a solve trace and split it at level increments."""

    arcade = kit.offline_arcade()
    env = _make_env(arcade, source.game_id)
    frame = env.reset()
    current_level = kit.frame_level(frame)
    level_start = 0
    levels: list[LevelTrace] = []
    for index, step in enumerate(steps):
        frame = _apply_step(env, step)
        next_level = kit.frame_level(frame)
        if next_level > current_level:
            levels.append(
                LevelTrace(
                    game=source.game,
                    game_id=source.game_id,
                    level_index=len(levels) + 1,
                    start_level=current_level,
                    target_level=next_level,
                    prefix=tuple(steps[:level_start]),
                    steps=tuple(steps[level_start : index + 1]),
                )
            )
            current_level = next_level
            level_start = index + 1
    return tuple(levels)


def load_usable_traces(repo: Path = REPO) -> tuple[dict[str, GameTrace], list[str]]:
    """REQ-LEARN-4318-1: load all solved traces that replay to at least one level."""

    traces: dict[str, GameTrace] = {}
    missing: list[str] = []
    for game, source in TRACE_SOURCES.items():
        path = repo / source.rel_path
        try:
            payload = _read_json(path)
            steps = decode_steps(payload, source.sequence_key)
            levels = split_trace_into_levels(source, steps) if steps else ()
        except Exception:
            missing.append(game)
            continue
        if not levels:
            missing.append(game)
            continue
        traces[game] = GameTrace(
            game=game,
            game_id=source.game_id,
            path=path,
            sha256=_sha256_file(path),
            steps=steps,
            levels=levels,
        )
    return traces, missing


class TraceFrontierActionModel:
    """Trace-derived action frontier for positive-control held-out solves."""

    def __init__(
        self,
        trace: GameTrace,
        next_label_by_state: Mapping[tuple[int, str], str],
        *,
        branch_decoys: int = 2,
        offtrace_branching: int = 1,
        offtrace_depth_limit: int = 2,
    ) -> None:
        self.trace = trace
        self.next_label_by_state = dict(next_label_by_state)
        self.labels = sorted({step_label(step) for step in trace.steps})
        self.branch_decoys = int(branch_decoys)
        self.offtrace_branching = int(offtrace_branching)
        self.offtrace_depth_limit = int(offtrace_depth_limit)

    def action_labels(self, _env: Any, frame: Any, path: tuple[str, ...]) -> list[str]:
        key = frame_state_key(frame)
        correct = self.next_label_by_state.get(key)
        if correct is not None:
            decoys = [label for label in self.labels if label != correct]
            return decoys[: self.branch_decoys] + [correct]
        if len(path) < self.offtrace_depth_limit:
            return self.labels[: self.offtrace_branching]
        return []


def build_next_label_map(trace: GameTrace) -> dict[tuple[int, str], str]:
    arcade = kit.offline_arcade()
    env = _make_env(arcade, trace.game_id)
    frame = env.reset()
    out: dict[tuple[int, str], str] = {}
    for step in trace.steps:
        out[frame_state_key(frame)] = step_label(step)
        frame = _apply_step(env, step)
    return out


def apply_label(env: Any, label: str, _frame: Any) -> Any:
    return _apply_step(env, label_to_step(label))


def collect_training_rows(
    traces: Mapping[str, GameTrace],
    train_games: Sequence[str],
) -> tuple[list[list[float]], list[float]]:
    """REQ-LEARN-4318-3: collect state -> steps-to-go rows excluding held-out games."""

    arcade = kit.offline_arcade()
    X: list[list[float]] = []
    y: list[float] = []
    for game in train_games:
        trace = traces[game]
        env = _make_env(arcade, trace.game_id)
        frame = env.reset()
        for level in trace.levels:
            for index, step in enumerate(level.steps):
                X.append([float(value) for value in cross_game_features(frame)])
                y.append(float(len(level.steps) - index))
                frame = _apply_step(env, step)
    return X, y


def _uniform_verifier(_game: Any, _frame: Any) -> float:
    return 0.0


def _learned_frame_verifier(value_head: LearnedVerifier):
    def verifier(_game: Any, frame: Any) -> float:
        return float(value_head(frame)) if frame is not None else 0.0

    return verifier


def run_solver_arm(
    trace: GameTrace,
    verifier: Any,
    *,
    max_nodes: int = 60000,
) -> list[dict[str, Any]]:
    """REQ-LEARN-4318: solve held-out levels and count OfflineSolver states."""

    arcade = kit.offline_arcade()
    env = _make_env(arcade, trace.game_id)
    action_model = TraceFrontierActionModel(trace, build_next_label_map(trace))
    solver = kit.OfflineSolver(
        trace.game_id,
        action_model.action_labels,
        apply_label,
        solver_frame_state_key,
        max_nodes=max_nodes,
        verifier=verifier,
    )
    prefix: list[str] = []
    rows: list[dict[str, Any]] = []
    for level in trace.levels:
        start_frame = solver._replay(env, prefix)
        start_level = kit.frame_level(start_frame)
        path, states = solver.solve_level(
            env,
            start_level,
            prefix,
            depth_cap=len(level.steps) + 3,
        )
        solved = path is not None
        if solved:
            prefix.extend(path)
        rows.append(
            {
                "held_out_game": trace.game,
                "level_index": int(level.level_index),
                "target_level": int(level.target_level),
                "solved": bool(solved),
                "states": int(states),
                "path_len": int(len(path or [])),
            }
        )
    return rows


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def summarize_state_reduction(
    level_rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = RANDOM_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """REQ-LEARN-4318: aggregate held-out state reductions and bootstrap CI."""

    rows = [dict(row) for row in level_rows]
    total_uniform = sum(int(row.get("states_uniform", 0) or 0) for row in rows)
    total_transferred = sum(int(row.get("states_transferred", 0) or 0) for row in rows)
    reduction = _safe_ratio(total_uniform, total_transferred)
    baseline_solves = bool(rows) and all(bool(row.get("baseline_solved")) for row in rows)

    rng = np.random.default_rng(int(random_seed))
    boot: list[float] = []
    if rows and n_resamples > 0:
        for _ in range(int(n_resamples)):
            sample = rng.integers(0, len(rows), size=len(rows))
            u = sum(int(rows[int(i)].get("states_uniform", 0) or 0) for i in sample)
            t = sum(int(rows[int(i)].get("states_transferred", 0) or 0) for i in sample)
            boot.append(_safe_ratio(u, t))
    if boot:
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    else:
        ci = [0.0, 0.0]

    per_game: dict[str, dict[str, Any]] = {}
    for game in sorted({str(row.get("held_out_game")) for row in rows}):
        game_rows = [row for row in rows if str(row.get("held_out_game")) == game]
        u = sum(int(row.get("states_uniform", 0) or 0) for row in game_rows)
        t = sum(int(row.get("states_transferred", 0) or 0) for row in game_rows)
        per_game[game] = {
            "states_uniform": int(u),
            "states_transferred": int(t),
            "state_reduction": float(_safe_ratio(u, t)),
            "baseline_solved": all(bool(row.get("baseline_solved")) for row in game_rows),
            "transferred_solved": all(bool(row.get("transferred_solved")) for row in game_rows),
            "levels": [
                {
                    "level_index": int(row.get("level_index", 0) or 0),
                    "states_uniform": int(row.get("states_uniform", 0) or 0),
                    "states_transferred": int(row.get("states_transferred", 0) or 0),
                    "state_reduction": float(
                        _safe_ratio(row.get("states_uniform", 0) or 0, row.get("states_transferred", 0) or 0)
                    ),
                    "baseline_solved": bool(row.get("baseline_solved")),
                    "transferred_solved": bool(row.get("transferred_solved")),
                }
                for row in game_rows
            ],
        }

    helps = bool(baseline_solves and reduction > 1.0 and ci[0] > 1.0)
    return {
        "cross_game_transfer_helps": helps,
        "cross_game_state_reduction": float(reduction),
        "cross_game_state_reduction_ci95": [float(ci[0]), float(ci[1])],
        "per_held_out_game_reduction": per_game,
        "baseline_solves_held_out": baseline_solves,
        "n_held_out_levels": len(rows),
        "n_bootstrap_resamples": int(n_resamples),
    }


def _round_weights(value_head: LearnedVerifier) -> list[float]:
    if value_head.w is None:
        return []
    return [round(float(v), 12) for v in value_head.w.tolist()]


def _verdict(summary: Mapping[str, Any]) -> str:
    if not summary.get("baseline_solves_held_out"):
        return "complete: arc_cross_game_transfer_positive_control_failed"
    if summary.get("cross_game_transfer_helps") is True:
        reduction = float(summary.get("cross_game_state_reduction", 0.0) or 0.0)
        return f"success: arc_cross_game_value_head_transfer_{reduction:.3f}x"
    return "complete: arc_cross_game_value_head_no_improvement_positive_control_passed"


def build_blocked_artifact(
    *,
    usable_games: Sequence[str],
    missing_games: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4318-BLOCKED: terminal artifact for insufficient traces."""

    return {
        "experiment": "experiment_4318_arc_cross_game_learned_verifier_transfer",
        "title": "arc_cross_game_learned_value_head_transfer",
        "honest_verdict": "blocked_insufficient_solve_traces",
        "cross_game_transfer_helps": False,
        "cross_game_state_reduction": 0.0,
        "cross_game_state_reduction_ci95": [0.0, 0.0],
        "per_held_out_game_reduction": {},
        "baseline_solves_held_out": False,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(
            {"usable_games": list(usable_games), "missing_games": list(missing_games)}
        ),
        "model_specs": {
            "blocked_reason": "insufficient_solve_traces",
            "usable_games": list(usable_games),
            "missing_games": list(missing_games),
            "minimum_usable_games": MIN_USABLE_GAMES,
            "value_head": "not_trained",
            "llm_weight_mutation": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "acceptance_gate_passed": True,
    }


def build_complete_artifact(
    *,
    level_rows: Sequence[Mapping[str, Any]],
    split_specs: Mapping[str, Any],
    model_weight_specs: Mapping[str, Any],
    trace_checksums: Mapping[str, str],
    duration_s: float,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    summary = summarize_state_reduction(
        level_rows,
        random_seed=RANDOM_SEED,
        n_resamples=n_resamples,
    )
    missing_gaps = []
    if summary["baseline_solves_held_out"] and not summary["cross_game_transfer_helps"]:
        missing_gaps.append(
            {
                "gap_id": GAP_ID,
                "failure_mode": "transferred linear value-head did not reduce held-out OfflineSolver states",
                "missing_discriminator": "game-invariant ARC value representation",
                "candidate_design": "learned frame encoder or per-game adapter-conditioned value head",
                "priority": "medium",
            }
        )
    checksum_payload = {
        "level_rows": list(level_rows),
        "split_specs": split_specs,
        "model_weight_specs": model_weight_specs,
        "trace_checksums": trace_checksums,
        "summary": summary,
        "random_seed": RANDOM_SEED,
    }
    artifact = {
        "experiment": "experiment_4318_arc_cross_game_learned_verifier_transfer",
        "title": "arc_cross_game_learned_value_head_transfer",
        **summary,
        "honest_verdict": _verdict(summary),
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "value_head": {
                "module": "python/carnot/agentic/arc_value_learner.py",
                "class": "LearnedVerifier",
                "architecture": "linear least-squares value head with bias",
                "features": "cross_game_features(frame): nonzero density, n colors, object count, spread, dominant color share",
                "target": "raw steps_remaining within each solved level",
                "training_compute": "CPU numpy.linalg.lstsq",
                "llm_weight_mutation": False,
            },
            "splits": dict(split_specs),
            "model_weights_by_held_out_game": dict(model_weight_specs),
            "state_count_instrumentation": (
                "arc_solver_kit.OfflineSolver.solve_level counts generated child "
                "states in last_states_expanded; lower is better"
            ),
            "held_out_action_frontier": (
                "trace-derived reproducible action frontier used only to make the "
                "uniform positive-control solve finite; held-out states are excluded "
                "from value-head training"
            ),
            "trace_checksums": dict(trace_checksums),
            "bootstrap_resamples": int(n_resamples),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "held_out_level_rows": [dict(row) for row in level_rows],
        "missing_verifier_gaps": missing_gaps,
        "methodology_note": (
            "This is a CPU-only value-head search-efficiency measurement over the "
            "currently reproduced solved-game traces. The CI is over held-out "
            "levels, so a null is logged as a representation gap rather than "
            "promoted as a broad impossibility claim."
        ),
        "acceptance_gate_passed": True,
    }
    return artifact


def evaluate_leave_one_game_out(repo: Path = REPO) -> dict[str, Any]:
    started = time.time()
    traces, missing_games = load_usable_traces(repo)
    if len(traces) < MIN_USABLE_GAMES:
        return build_blocked_artifact(
            usable_games=sorted(traces),
            missing_games=sorted(set(missing_games) | (set(TRACE_SOURCES) - set(traces))),
            duration_s=time.time() - started,
        )

    level_rows: list[dict[str, Any]] = []
    split_specs: dict[str, Any] = {}
    weight_specs: dict[str, Any] = {}
    trace_checksums = {game: trace.sha256 for game, trace in sorted(traces.items())}
    for held_out in sorted(traces):
        train_games = [game for game in sorted(traces) if game != held_out]
        X, y = collect_training_rows(traces, train_games)
        value_head = LearnedVerifier(cross_game_features).fit(X, y)
        uniform_rows = run_solver_arm(traces[held_out], _uniform_verifier)
        transferred_rows = run_solver_arm(traces[held_out], _learned_frame_verifier(value_head))
        split_specs[held_out] = {
            "held_out_game": held_out,
            "train_games": train_games,
            "n_train_samples": int(value_head.n_samples),
            "n_held_out_levels": len(traces[held_out].levels),
        }
        weight_specs[held_out] = {
            "n_samples": int(value_head.n_samples),
            "weights": _round_weights(value_head),
        }
        for uniform, transferred in zip(uniform_rows, transferred_rows, strict=True):
            level_rows.append(
                {
                    "held_out_game": held_out,
                    "level_index": int(uniform["level_index"]),
                    "target_level": int(uniform["target_level"]),
                    "states_uniform": int(uniform["states"]),
                    "states_transferred": int(transferred["states"]),
                    "baseline_solved": bool(uniform["solved"]),
                    "transferred_solved": bool(transferred["solved"]),
                    "uniform_path_len": int(uniform["path_len"]),
                    "transferred_path_len": int(transferred["path_len"]),
                }
            )

    return build_complete_artifact(
        level_rows=level_rows,
        split_specs=split_specs,
        model_weight_specs=weight_specs,
        trace_checksums=trace_checksums,
        duration_s=time.time() - started,
        n_resamples=BOOTSTRAP_RESAMPLES,
    )


def _is_bare_float(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_ci(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_bare_float(item) for item in value)
    )


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if not isinstance(artifact.get("honest_verdict"), str):
        errors.append("honest_verdict must be a string")
    if type(artifact.get("cross_game_transfer_helps")) is not bool:
        errors.append("cross_game_transfer_helps must be a bare bool")
    if type(artifact.get("baseline_solves_held_out")) is not bool:
        errors.append("baseline_solves_held_out must be a bare bool")
    if not _is_bare_float(artifact.get("cross_game_state_reduction")):
        errors.append("cross_game_state_reduction must be a bare float")
    if not _is_ci(artifact.get("cross_game_state_reduction_ci95")):
        errors.append("cross_game_state_reduction_ci95 must be a two-float list")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be the bare bool false")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be an object")
    if not isinstance(artifact.get("per_held_out_game_reduction"), Mapping):
        errors.append("per_held_out_game_reduction must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"field_principles missing {field}")
    if artifact.get("cross_game_transfer_helps") is True:
        ci = artifact.get("cross_game_state_reduction_ci95")
        if not _is_ci(ci) or float(ci[0]) <= 1.0:
            errors.append("cross_game_transfer_helps requires CI95 lower bound > 1.0")
        if not _is_bare_float(artifact.get("cross_game_state_reduction")) or float(
            artifact["cross_game_state_reduction"]
        ) <= 1.0:
            errors.append("cross_game_transfer_helps requires reduction > 1.0")
        if artifact.get("baseline_solves_held_out") is not True:
            errors.append("cross_game_transfer_helps requires baseline_solves_held_out=true")
    return errors


def ensure_gap_logged(repo: Path, artifact: Mapping[str, Any]) -> None:
    if artifact.get("cross_game_transfer_helps") is True:
        return
    if artifact.get("baseline_solves_held_out") is not True:
        return
    gap_path = repo / "ops" / "verifier_gaps.md"
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    if gap_path.exists():
        text = gap_path.read_text(encoding="utf-8")
    else:
        text = "# Verifier Gaps\n\n"
    if GAP_ID in text:
        return
    reduction = float(artifact.get("cross_game_state_reduction", 0.0) or 0.0)
    entry = (
        f"\n### {GAP_ID}: Game-invariant ARC value representation\n"
        "- status: open\n"
        f"- evidence: `{OUTPUT_REL.as_posix()}` reports cross_game_state_reduction="
        f"{reduction:.6g} with baseline_solves_held_out=true.\n"
        "- failure mode: a value-head trained on other solved games did not "
        "produce a decision-grade held-out search-state reduction.\n"
        "- missing discriminator: game-invariant ARC value representation that "
        "recognizes progress across navigation, click-placement, and rotation "
        "mechanics.\n"
        "- candidate design: learned frame encoder or adapter-conditioned value "
        "head trained on more reproduced solve traces, with hardware-portable CPU "
        "features first and an accelerator path later.\n"
        "- priority: medium\n"
    )
    gap_path.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")


def _write_artifact(repo: Path, artifact: Mapping[str, Any]) -> None:
    output = repo / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(*, repo: Path = REPO, write: bool = True) -> dict[str, Any]:
    artifact = evaluate_leave_one_game_out(repo)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(repo, artifact)
        ensure_gap_logged(repo, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
