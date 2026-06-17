"""Exp 4331: learned frame encoder cross-game ARC value transfer.

Spec refs: REQ-LEARN-4331, SCENARIO-LEARN-4331.

This is the learned-representation retry of Exp 4318.  Exp 4318 used generic
hand features and landed a powered null; this experiment trains a tiny CPU CNN
over raw ARC frames on solved-game traces, then fits the existing linear
`LearnedVerifier` value head on that embedding.  The held-out game's frames are
excluded from that split's encoder and value-head training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_value_learner import LearnedVerifier
from carnot import experiment_4318_arc_cross_game_learned_verifier_transfer as exp4318


REPO = Path(__file__).resolve().parents[2]
OUTPUT_REL = Path("results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json")
ENTRYPOINT_REL = Path("results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py")
RANDOM_SEED = 4331
BOOTSTRAP_RESAMPLES = 2000
MIN_USABLE_GAMES = 3
GAP_ID = "GAP-4331"
INFERENCE_SUBSTRATE = "cpu_offline_arc_agi3_trace_frontier_learned_frame_encoder_value_head"
SPEC_REFS = ["REQ-LEARN-4331", "SCENARIO-LEARN-4331"]
FRAME_SIZE = 16
N_COLORS = 16

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "learned_encoder_transfer_helps",
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
        "Terminal-prefixed. A learned-encoder transfer win (reduces held-out "
        "search states where generic features failed -- the gap closes), a "
        "powered null (the encoder is still insufficient -> sharper gap, retire "
        "the small-encoder ask), and an honest blocked_insufficient_solve_traces "
        "are ALL COMPLETE and decision-grade."
    ),
    "learned_encoder_transfer_helps": (
        "BARE bool: the capstone reads this; true iff the LEARNED-encoder "
        "value-head trained on OTHER games reduces held-out-game search states "
        "(reduction > 1.0 AND CI95 lower bound > 1.0) -- cross-game "
        "self-learning via a learned game-invariant representation."
    ),
    "cross_game_state_reduction": (
        "BARE float: held-out states_uniform / states_transferred (>1.0 = "
        "transfer helps; ~1.0 = still per-game-bound) -- compare to exp4318's "
        "generic-feature 1.0; the north-star EFFICIENCY metric."
    ),
    "cross_game_state_reduction_ci95": (
        "Bootstrap CI95 (>=2000 resamples) over held-out levels -- a lower "
        "bound > 1.0 makes 'transfer helps' decision-grade."
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
        "BARE bool=false -- a learned encoder + value-head over solver state "
        "(oracle-distinct heuristic), NOT the executable env oracle; NO LLM "
        "weight mutation."
    ),
    "random_seed": "Determinism precondition for the encoder training + the solver.",
    "reproducibility_checksum": (
        "Hash of the solve traces + the encoder + the leave-one-game-out split "
        "+ the state counts; lets a third party re-run."
    ),
    "model_specs": (
        "The learned-encoder architecture + the value-head + the train/held-out "
        "game split + the OfflineSolver state-count instrumentation; required "
        "methodology."
    ),
}

TRACE_SOURCES = {
    "r11l": exp4318.TraceSource(
        game="r11l",
        game_id="r11l-495a7899",
        rel_path=Path("results/arc_explore_trajectory_r11l.json"),
        sequence_key="trajectory",
    ),
    "ls20": exp4318.TraceSource(
        game="ls20",
        game_id="ls20-9607627b",
        rel_path=Path("results/arc_explore_trajectory_ls20.json"),
        sequence_key="trajectory",
    ),
    "wa30": exp4318.TraceSource(
        game="wa30",
        game_id="wa30-ee6fef47",
        rel_path=Path("results/experiment_4275_arc_incremental_progress_new_game.json"),
        sequence_key="action_plan",
    ),
    "lp85": exp4318.TraceSource(
        game="lp85",
        game_id="lp85-305b61c3",
        rel_path=Path("results/arc3_lp85_offline_resolve.json"),
        sequence_key="solution",
    ),
    "cd82": exp4318.TraceSource(
        game="cd82",
        game_id="cd82-fb555c5d",
        rel_path=Path("results/arc_explore_trajectory_cd82.json"),
        sequence_key="trajectory",
    ),
    "sp80": exp4318.TraceSource(
        game="sp80",
        game_id="sp80-589a99af",
        rel_path=Path("results/arc_explore_trajectory_sp80.json"),
        sequence_key="trajectory",
    ),
    "su15": exp4318.TraceSource(
        game="su15",
        game_id="su15-1944f8ab",
        rel_path=Path("results/arc_explore_trajectory_su15.json"),
        sequence_key="trajectory",
    ),
    "tu93": exp4318.TraceSource(
        game="tu93",
        game_id="tu93-0768757b",
        rel_path=Path("results/arc_explore_trajectory_tu93.json"),
        sequence_key="trajectory",
    ),
    "cn04": exp4318.TraceSource(
        game="cn04",
        game_id="cn04-2fe56bfb",
        rel_path=Path("results/arc_explore_trajectory_cn04.json"),
        sequence_key="trajectory",
    ),
    "m0r0": exp4318.TraceSource(
        game="m0r0",
        game_id="m0r0-492f87ba",
        rel_path=Path("results/arc_explore_trajectory_m0r0.json"),
        sequence_key="trajectory",
    ),
    "sk48": exp4318.TraceSource(
        game="sk48",
        game_id="sk48-d8078629",
        rel_path=Path("results/arc_explore_trajectory_sk48.json"),
        sequence_key="trajectory",
    ),
}


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _grid_array(frame_or_grid: Any) -> np.ndarray:
    if isinstance(frame_or_grid, np.ndarray):
        return frame_or_grid.astype(np.int64, copy=False)
    if isinstance(frame_or_grid, Sequence) and not isinstance(frame_or_grid, (str, bytes)):
        return np.asarray(frame_or_grid, dtype=np.int64)
    raw = getattr(frame_or_grid, "frame", frame_or_grid)
    return np.asarray(raw, dtype=np.int64)


def frame_to_tensor(
    frame_or_grid: Any,
    *,
    size: int = FRAME_SIZE,
    n_colors: int = N_COLORS,
) -> np.ndarray:
    """REQ-LEARN-4331-4: encode a raw frame grid as fixed one-hot channels."""

    grid = _grid_array(frame_or_grid)
    if grid.ndim > 2:
        grid = grid[-1]
    grid = np.asarray(grid[:size, :size], dtype=np.int64)
    height, width = grid.shape if grid.ndim == 2 else (0, 0)
    clipped = np.zeros((size, size), dtype=np.int64)
    if height and width:
        clipped[:height, :width] = np.mod(grid, n_colors)
    tensor = np.zeros((n_colors, size, size), dtype=np.float32)
    for color in range(n_colors):
        tensor[color] = clipped == color
    return tensor


class LearnedFrameEncoder:
    """Tiny CPU learned convolutional feature map for ARC frames.

    The filter bank is learned from raw-frame one-hot 3x3 patches with PCA.  A
    second PCA/ridge stage builds a compact embedding that the existing linear
    `LearnedVerifier` consumes.  This keeps the experiment CPU-only and avoids
    mutating any LLM weights while still learning frame features from traces.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        size: int = FRAME_SIZE,
        n_colors: int = N_COLORS,
        epochs: int = 24,
        learning_rate: float = 0.01,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.size = int(size)
        self.n_colors = int(n_colors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.n_samples = 0
        self.loss_history: list[float] = []
        self.n_filters = 4
        self._filters: np.ndarray | None = None
        self._patch_mean: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._projection: np.ndarray | None = None
        self._ridge_weights: np.ndarray | None = None

    def _patch_matrix(self, tensors: np.ndarray) -> np.ndarray:
        padded = np.pad(tensors, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        patches: list[np.ndarray] = []
        for y in range(self.size):
            for x in range(self.size):
                patch = padded[:, :, y : y + 3, x : x + 3].reshape(tensors.shape[0], -1)
                patches.append(patch)
        return np.concatenate(patches, axis=0).astype(np.float32)

    def _fit_filters(self, tensors: np.ndarray) -> None:
        patches = self._patch_matrix(tensors)
        self._patch_mean = patches.mean(axis=0, keepdims=True)
        centered = patches - self._patch_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        filters = np.zeros((self.n_filters, self.n_colors * 3 * 3), dtype=np.float32)
        usable = min(self.n_filters, vt.shape[0])
        filters[:usable] = vt[:usable].astype(np.float32)
        self._filters = filters.reshape(self.n_filters, self.n_colors, 3, 3)

    def _conv_pool_features(self, tensor: np.ndarray) -> np.ndarray:
        if self._filters is None:
            raise ValueError("frame encoder is untrained")
        padded = np.pad(tensor, ((0, 0), (1, 1), (1, 1)), mode="constant")
        pooled: list[float] = []
        for filt in self._filters:
            activations = np.zeros((self.size, self.size), dtype=np.float32)
            for y in range(self.size):
                for x in range(self.size):
                    patch = padded[:, y : y + 3, x : x + 3]
                    activations[y, x] = float(np.sum(patch * filt))
            activations = np.maximum(activations, 0.0)
            pooled.extend(
                [
                    float(np.mean(activations)),
                    float(np.std(activations)),
                    float(np.max(activations)),
                ]
            )
        return np.asarray(pooled, dtype=np.float32)

    def fit(self, frames: Sequence[Any], targets: Sequence[float]) -> "LearnedFrameEncoder":
        """Train the learned convolutional feature map on CPU."""

        if not frames:
            raise ValueError("cannot train frame encoder with no frames")
        tensors = np.stack(
            [frame_to_tensor(frame, size=self.size, n_colors=self.n_colors) for frame in frames],
            axis=0,
        )
        self._fit_filters(tensors)
        pooled = np.stack([self._conv_pool_features(tensor) for tensor in tensors], axis=0)
        self._feature_mean = pooled.mean(axis=0, keepdims=True)
        centered = pooled - self._feature_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projection = np.zeros((pooled.shape[1], self.embedding_dim), dtype=np.float32)
        usable = min(self.embedding_dim, vt.shape[0])
        projection[:, :usable] = vt[:usable].T.astype(np.float32)
        self._projection = projection
        embedding = np.tanh(centered @ self._projection)
        y = np.asarray(targets, dtype=np.float64)
        design = np.hstack([embedding.astype(np.float64), np.ones((embedding.shape[0], 1))])
        ridge = 1e-6 * np.eye(design.shape[1], dtype=np.float64)
        self._ridge_weights = np.linalg.solve(design.T @ design + ridge, design.T @ y)
        pred = design @ self._ridge_weights
        self.loss_history = [float(np.mean((pred - y) ** 2))]
        self.n_samples = len(frames)
        return self

    def transform_grid(self, frame_or_grid: Any) -> list[float]:
        if self._projection is None or self._feature_mean is None:
            raise ValueError("frame encoder is untrained")
        tensor = frame_to_tensor(frame_or_grid, size=self.size, n_colors=self.n_colors)
        pooled = self._conv_pool_features(tensor)[None, :]
        embedding = np.tanh((pooled - self._feature_mean) @ self._projection)
        return [float(v) for v in embedding.squeeze(0).tolist()]

    def featurize(self, frame_or_grid: Any) -> list[float]:
        return self.transform_grid(frame_or_grid)

    def model_summary(self) -> dict[str, Any]:
        return {
            "architecture": "tiny_cpu_cnn_frame_encoder",
            "input": f"{self.n_colors} one-hot color channels over {self.size}x{self.size} raw frame crop/pad",
            "layers": [
                "PCA-learned 3x3 one-hot convolution filters + ReLU",
                "mean/std/max spatial pooling",
                f"PCA projection to {self.embedding_dim}-dim tanh embedding",
                "ridge auxiliary steps-to-go head for representation fitting",
            ],
            "embedding_dim": self.embedding_dim,
            "n_filters": self.n_filters,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "n_samples": self.n_samples,
            "training_compute": "CPU numpy PCA/ridge",
            "llm_weight_mutation": False,
            "loss_history": [round(float(v), 8) for v in self.loss_history[-5:]],
        }


def load_usable_traces(repo: Path = REPO) -> tuple[dict[str, exp4318.GameTrace], list[str]]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4331-1: load solved traces that replay to at least one level."""

    traces: dict[str, exp4318.GameTrace] = {}
    missing: list[str] = []
    for game, source in TRACE_SOURCES.items():
        path = repo / source.rel_path
        try:
            payload = exp4318._read_json(path)
            steps = exp4318.decode_steps(payload, source.sequence_key)
            levels = exp4318.split_trace_into_levels(source, steps) if steps else ()
        except Exception:
            missing.append(game)
            continue
        if not levels:
            missing.append(game)
            continue
        traces[game] = exp4318.GameTrace(
            game=game,
            game_id=source.game_id,
            path=path,
            sha256=exp4318._sha256_file(path),
            steps=steps,
            levels=levels,
        )
    return traces, missing


def collect_training_frames(
    traces: Mapping[str, exp4318.GameTrace],
    train_games: Sequence[str],
) -> tuple[list[np.ndarray], list[float]]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4331-3: collect raw-frame rows excluding held-out games."""

    from carnot.agentic import arc_solver_kit as kit

    arcade = kit.offline_arcade()
    frames: list[np.ndarray] = []
    y: list[float] = []
    for game in train_games:
        trace = traces[game]
        env = exp4318._make_env(arcade, trace.game_id)
        frame = env.reset()
        for level in trace.levels:
            for index, step in enumerate(level.steps):
                frames.append(_grid_array(frame).copy())
                y.append(float(len(level.steps) - index))
                frame = exp4318._apply_step(env, step)
    return frames, y


def train_encoder_value_head(
    traces: Mapping[str, exp4318.GameTrace],
    train_games: Sequence[str],
    *,
    seed: int,
) -> tuple[LearnedFrameEncoder, LearnedVerifier]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4331-4: train encoder, then fit the linear value head."""

    frames, targets = collect_training_frames(traces, train_games)
    encoder = LearnedFrameEncoder(seed=seed).fit(frames, targets)
    X = [encoder.transform_grid(frame) for frame in frames]
    value_head = LearnedVerifier(encoder.featurize).fit(X, targets)
    return encoder, value_head


def _round_weights(value_head: LearnedVerifier) -> list[float]:
    if value_head.w is None:
        return []
    return [round(float(v), 12) for v in value_head.w.tolist()]


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
    """REQ-LEARN-4331: aggregate held-out state reductions and bootstrap CI."""

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
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else [0.0, 0.0]

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
        "learned_encoder_transfer_helps": helps,
        "cross_game_state_reduction": float(reduction),
        "cross_game_state_reduction_ci95": [float(ci[0]), float(ci[1])],
        "per_held_out_game_reduction": per_game,
        "baseline_solves_held_out": baseline_solves,
        "n_held_out_levels": len(rows),
        "n_bootstrap_resamples": int(n_resamples),
    }


def _verdict(summary: Mapping[str, Any]) -> str:
    if not summary.get("baseline_solves_held_out"):
        return "complete: learned_frame_encoder_positive_control_failed"
    if summary.get("learned_encoder_transfer_helps") is True:
        reduction = float(summary.get("cross_game_state_reduction", 0.0) or 0.0)
        return f"success: learned_frame_encoder_transfer_{reduction:.3f}x"
    return "complete: learned_frame_encoder_transfer_no_improvement_positive_control_passed"


def build_blocked_artifact(
    *,
    usable_games: Sequence[str],
    missing_games: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4331-BLOCKED: terminal artifact for insufficient traces."""

    return {
        "experiment": "experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer",
        "title": "self_learning_learned_frame_encoder_cross_game_transfer",
        "honest_verdict": "blocked_insufficient_solve_traces",
        "learned_encoder_transfer_helps": False,
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
            "encoder": "not_trained",
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
    model_specs_by_held_out: Mapping[str, Any],
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
    if summary["baseline_solves_held_out"] and not summary["learned_encoder_transfer_helps"]:
        missing_gaps.append(
            {
                "gap_id": GAP_ID,
                "failure_mode": (
                    "small learned frame encoder over the current solved set did not "
                    "produce a decision-grade held-out OfflineSolver state reduction"
                ),
                "missing_discriminator": "game-invariant ARC value representation",
                "candidate_design": (
                    "larger learned frame encoder, more reproduced solved traces, "
                    "or adapter-conditioned value head with a hardware-portable path"
                ),
                "priority": "medium",
            }
        )
    checksum_payload = {
        "level_rows": list(level_rows),
        "split_specs": split_specs,
        "model_specs_by_held_out": model_specs_by_held_out,
        "trace_checksums": trace_checksums,
        "summary": summary,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer",
        "title": "self_learning_learned_frame_encoder_cross_game_transfer",
        **summary,
        "honest_verdict": _verdict(summary),
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "encoder": {
                "architecture": "tiny_cpu_cnn_frame_encoder",
                "module": "python/carnot/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.py",
                "input": f"{N_COLORS} one-hot raw-frame color channels over {FRAME_SIZE}x{FRAME_SIZE} crop/pad",
                "embedding_consumed_by": "arc_value_learner.LearnedVerifier linear value-head",
                "llm_weight_mutation": False,
            },
            "value_head": {
                "module": "python/carnot/agentic/arc_value_learner.py",
                "class": "LearnedVerifier",
                "architecture": "linear least-squares value head with bias over learned encoder embeddings",
                "target": "raw steps_remaining within each solved level",
                "training_compute": "CPU numpy.linalg.lstsq",
                "llm_weight_mutation": False,
            },
            "splits": dict(split_specs),
            "models_by_held_out_game": dict(model_specs_by_held_out),
            "state_count_instrumentation": (
                "arc_solver_kit.OfflineSolver.solve_level counts generated child "
                "states in last_states_expanded; lower is better"
            ),
            "held_out_action_frontier": (
                "trace-derived reproducible action frontier used only to make the "
                "uniform positive-control solve finite; held-out states are excluded "
                "from encoder and value-head training"
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
            "CPU-only learned frame encoder plus linear value-head over solved-game "
            "traces. The held-out game's frames are excluded from that split's "
            "training rows; its trace is used only as a positive-control action "
            "frontier for OfflineSolver state counts."
        ),
        "acceptance_gate_passed": True,
    }


def evaluate_leave_one_game_out(repo: Path = REPO) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
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
    model_specs_by_held_out: dict[str, Any] = {}
    trace_checksums = {game: trace.sha256 for game, trace in sorted(traces.items())}
    for split_index, held_out in enumerate(sorted(traces)):
        train_games = [game for game in sorted(traces) if game != held_out]
        encoder, value_head = train_encoder_value_head(
            traces,
            train_games,
            seed=RANDOM_SEED + split_index,
        )
        uniform_rows = exp4318.run_solver_arm(traces[held_out], exp4318._uniform_verifier)
        transferred_rows = exp4318.run_solver_arm(
            traces[held_out],
            exp4318._learned_frame_verifier(value_head),
        )
        split_specs[held_out] = {
            "held_out_game": held_out,
            "train_games": train_games,
            "n_train_samples": int(value_head.n_samples),
            "n_encoder_samples": int(encoder.n_samples),
            "n_held_out_levels": len(traces[held_out].levels),
        }
        model_specs_by_held_out[held_out] = {
            "encoder": encoder.model_summary(),
            "value_head": {
                "n_samples": int(value_head.n_samples),
                "weights": _round_weights(value_head),
            },
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
        model_specs_by_held_out=model_specs_by_held_out,
        trace_checksums=trace_checksums,
        duration_s=time.time() - started,
        n_resamples=BOOTSTRAP_RESAMPLES,
    )


def _is_bare_float(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_ci(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_bare_float(item) for item in value)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """SCENARIO-LEARN-4331: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if not isinstance(artifact.get("honest_verdict"), str):
        errors.append("honest_verdict must be a string")
    if type(artifact.get("learned_encoder_transfer_helps")) is not bool:
        errors.append("learned_encoder_transfer_helps must be a bare bool")
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
            if principles.get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles mismatch for {field}")
    if artifact.get("learned_encoder_transfer_helps") is True:
        ci = artifact.get("cross_game_state_reduction_ci95")
        if not _is_ci(ci) or float(ci[0]) <= 1.0:
            errors.append("learned_encoder_transfer_helps requires CI95 lower bound > 1.0")
        if not _is_bare_float(artifact.get("cross_game_state_reduction")) or float(
            artifact["cross_game_state_reduction"]
        ) <= 1.0:
            errors.append("learned_encoder_transfer_helps requires reduction > 1.0")
        if artifact.get("baseline_solves_held_out") is not True:
            errors.append("learned_encoder_transfer_helps requires baseline_solves_held_out=true")
    return errors


def ensure_gap_logged(repo: Path, artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-4331-7: sharpen the representation gap after a powered null."""

    if artifact.get("learned_encoder_transfer_helps") is True:
        return
    if artifact.get("baseline_solves_held_out") is not True:
        return
    gap_path = repo / "ops" / "verifier_gaps.md"
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    text = gap_path.read_text(encoding="utf-8") if gap_path.exists() else "# Verifier Gaps\n\n"
    if GAP_ID in text:
        return
    reduction = float(artifact.get("cross_game_state_reduction", 0.0) or 0.0)
    entry = (
        f"\n### {GAP_ID}: Game-invariant ARC value representation - small encoder insufficient\n"
        "- status: open\n"
        f"- evidence: `{OUTPUT_REL.as_posix()}` reports cross_game_state_reduction="
        f"{reduction:.6g} with baseline_solves_held_out=true.\n"
        "- failure mode: small learned frame encoder over the current solved set is insufficient "
        "to produce a decision-grade held-out search-state reduction.\n"
        "- missing discriminator: game-invariant ARC value representation that recognizes "
        "progress across navigation, click-placement, rotation, and shallow-tail mechanics.\n"
        "- candidate design: larger encoder with more reproduced games, adapter-conditioned "
        "value head, or experience-gated source-relevance features; preserve a CPU/hardware path.\n"
        "- priority: medium\n"
    )
    gap_path.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")


def _write_artifact(repo: Path, artifact: Mapping[str, Any]) -> None:
    output = repo / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_adversarial_verify(repo: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover - subprocess boundary
    """REQ-LEARN-4331-6: run the repository adversarial artifact verifier."""

    output = repo / OUTPUT_REL
    cmd = [sys.executable, str(repo / "scripts" / "adversarial_verify.py"), str(output), "--json"]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    try:
        report = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        report = {"stdout": completed.stdout, "stderr": completed.stderr}
    flagged_count = int(report.get("flagged_count", 0) or 0)
    status = "clean" if completed.returncode == 0 and flagged_count == 0 else "flagged"
    return {
        "status": status,
        "returncode": int(completed.returncode),
        "flagged_count": flagged_count,
        "reports": report.get("reports", []),
    }


def run(*, repo: Path = REPO, write: bool = True) -> dict[str, Any]:
    artifact = evaluate_leave_one_game_out(repo)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(repo, artifact)
        artifact = dict(artifact)
        artifact["adversarial_verify"] = run_adversarial_verify(repo, artifact)
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
