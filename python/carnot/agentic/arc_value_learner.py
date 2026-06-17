"""Learned verifier for the ARC solver harness — the self-improving loop: train a
value/energy function on our OWN solve traces (state -> steps-to-go), then use it
as the verifier that routes (best-first prunes) the search for the NEXT level/game.
Each solve makes the verifier better, which makes the next search faster.

This is the LEARNED counterpart of the hand-computed verifier in
arc3_verifier_routed_search_demo: instead of being told "descend total
goal-distance", it LEARNS from data which state-features predict proximity to the
win. The verifier itself is the project's energy/verifier thesis — here it learns
from successes (solve trajectories) and routes the search.

Plug a per-game `featurize(game) -> sequence[float]` in; the rest is general. The
value is trained by least-squares on (features, steps_remaining) collected from
solved trajectories. `LearnedVerifier.__call__(game)` returns predicted
steps-to-go (LOWER = closer to win), the score OfflineSolver descends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


def cross_game_features(frame: Any) -> list[float]:
    """GAME-AGNOSTIC frame features for a CROSS-GAME verifier (transfer the value
    head across games). Computable from any ARC-AGI-3 frame grid; normalized so the
    scale is comparable across games. The bet: some of these correlate with
    progress-toward-win regardless of game, giving a deep/sparse game (e.g. wa30) a
    search heuristic before it has its own solved trajectory."""
    from collections import Counter
    from carnot.agentic.arc_agi3_world_model import grid_of, objects
    g = grid_of(frame)
    total = max(1, g.size)
    flat = g.flatten().tolist()
    cnt = Counter(flat)
    n_nonzero = total - cnt.get(0, 0)
    n_colors = len([c for c in cnt if c != 0])
    n_objects = len(objects(g))
    nz = (g != 0)
    ys, xs = np.where(nz)
    spread = float((np.std(xs) + np.std(ys)) / max(1, max(g.shape))) if len(xs) else 0.0
    dom = (max((v for k, v in cnt.items() if k != 0), default=0) / max(1, n_nonzero))
    return [n_nonzero / total, float(n_colors), float(n_objects) / 32.0, spread, float(dom)]


def collect_trajectory_data(env: Any, solver: Any, prefix: Sequence[str],
                            level_path: Sequence[str], featurize: Callable[[Any], Sequence[float]]):
    """Replay a solved LEVEL path and emit (features, steps_remaining) per state —
    the supervision: states near the level-up are low-cost, far ones high-cost."""
    X, y = [], []
    n = len(level_path)
    for i in range(n):
        solver._replay(env, list(prefix) + list(level_path[:i]))  # state before action i
        X.append([float(v) for v in featurize(env._game)])
        y.append(float(n - i))  # steps remaining to the level-up
    return X, y


class LearnedVerifier:
    """A linear value function v(state) ≈ steps-to-go, trained on solve traces.

    Linear keeps it interpretable + needs almost no data (a value head, not a deep
    net); swap in any regressor by overriding fit/predict. Use as the `verifier`
    arg of arc_solver_kit.OfflineSolver."""

    def __init__(self, featurize: Callable[[Any], Sequence[float]]) -> None:
        self.featurize = featurize
        self.w: np.ndarray | None = None  # last entry is the bias
        self.n_samples = 0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "LearnedVerifier":
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        A = np.hstack([A, np.ones((A.shape[0], 1))])  # bias column
        self.w, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.n_samples = A.shape[0]
        return self

    def __call__(self, game: Any) -> float:
        if self.w is None:
            return 0.0  # untrained ⇒ neutral (solver degrades to BFS)
        f = np.asarray([float(v) for v in self.featurize(game)] + [1.0], dtype=float)
        return float(max(0.0, f @ self.w))  # predicted steps-to-go; clamp ≥ 0

    # --- checkpointing: capture the trained weights as a versionable, MIRROR-READY
    # artifact so the learning loop's output is never lost in a demo file. Mirror to
    # HuggingFace (Carnot-EBM) + IPFS per CLAUDE.md Decentralization Rule 3 once the
    # verifier is substantial; the PUBLIC release is operator-only (External Publication).
    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        if self.w is None:
            raise ValueError("nothing to save: verifier is untrained")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "schema": "carnot_arc_learned_verifier_v1",
            "kind": "linear_value_head",
            "weights": self.w.tolist(),          # [feature weights..., bias]
            "feature_names": (meta or {}).get("feature_names"),
            "trained_games": (meta or {}).get("trained_games"),
            "n_samples": self.n_samples,
            "provenance": (meta or {}).get("provenance"),
        }, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path, featurize: Callable[[Any], Sequence[float]]) -> "LearnedVerifier":
        d = json.loads(Path(path).read_text())
        v = cls(featurize)
        v.w = np.asarray(d["weights"], dtype=float)
        v.n_samples = int(d.get("n_samples", 0))
        return v
