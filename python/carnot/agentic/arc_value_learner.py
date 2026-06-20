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


def cross_game_features_v2(frame: Any) -> list[float]:
    """RICHER frame-only cross-game features (v2): the 5 v1 scalars PLUS a coarse SPATIAL occupancy map
    (6x6 pooled nonzero-fraction) so the value head can locate WHERE activity is, not just how much.
    The v1 head (5 scalars) was inert at routing because it threw away all spatial structure; this keeps
    a small spatial signature (41 features) while staying game-agnostic + cheap. Frame-only (live-legal)."""
    import numpy as np
    from carnot.agentic.arc_agi3_world_model import grid_of
    base = cross_game_features(frame)
    g = np.asarray(grid_of(frame))
    if g.ndim == 1:
        s = int(round(g.size ** 0.5))
        g = g.reshape(s, s) if s * s == g.size else g.reshape(1, -1)
    nz = (g != 0).astype(float)
    h, w = nz.shape
    # pool into a 6x6 grid of nonzero-fraction (handles any input size; empty blocks -> 0)
    grid = []
    for by in range(6):
        for bx in range(6):
            y0, y1 = h * by // 6, max(h * by // 6 + 1, h * (by + 1) // 6)
            x0, x1 = w * bx // 6, max(w * bx // 6 + 1, w * (bx + 1) // 6)
            block = nz[y0:y1, x0:x1]
            grid.append(float(block.mean()) if block.size else 0.0)
    return base + grid


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


class DiscriminativeVerifier:
    """A logistic WIN-REACHABILITY classifier: P(this state is on a winning path).

    DISTINCT from LearnedVerifier (which regresses steps-to-go along the ONE banked winning path). A
    distance-along-the-gold-path value cannot tell an off-path TRAP from a near-win: a state 3 steps from a
    dead-end and a state 3 steps from the win can carry identical marginal features and identical regressed
    value, yet one can no longer win. That is the discrimination the regression head structurally lacks
    (and the reason the linear value head is too weak to route with weight>0). This head learns it directly
    from the negatives the solver already produces (off-path / game-over states) -- the "off-path negatives"
    the corpus builder was always supposed to add.

    Logistic regression with standardized features + L2, numpy gradient descent (no sklearn dep). Returns a
    probability in [0,1]; the search can prune low-P(on-path) states the steps-to-go value would miss."""

    def __init__(self, featurize: Callable[[Any], Sequence[float]]) -> None:
        self.featurize = featurize
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.n_samples = 0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float],
            iters: int = 800, lr: float = 0.5, l2: float = 1e-3) -> "DiscriminativeVerifier":
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)              # 1 = on winning path, 0 = off-path / dead-end
        self.mu = A.mean(axis=0)
        self.sd = A.std(axis=0) + 1e-8
        Z = np.hstack([(A - self.mu) / self.sd, np.ones((A.shape[0], 1))])  # standardized + bias
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w)))
            grad = Z.T @ (p - b) / len(b) + l2 * np.r_[w[:-1], 0.0]
            w -= lr * grad
        self.w = w
        self.n_samples = len(b)
        return self

    def proba(self, frame: Any) -> float:
        if self.w is None:
            return 0.5
        z = (np.asarray([float(v) for v in self.featurize(frame)]) - self.mu) / self.sd
        z = np.r_[z, 1.0]
        return float(1.0 / (1.0 + np.exp(-(z @ self.w))))

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        if self.w is None:
            raise ValueError("nothing to save: classifier is untrained")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "schema": "carnot_arc_discriminative_verifier_v1",
            "kind": "logistic_win_reachability",
            "weights": self.w.tolist(),
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "feature_names": (meta or {}).get("feature_names"),
            "trained_games": (meta or {}).get("trained_games"),
            "n_samples": self.n_samples,
            "provenance": (meta or {}).get("provenance"),
        }, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path, featurize: Callable[[Any], Sequence[float]]) -> "DiscriminativeVerifier":
        d = json.loads(Path(path).read_text())
        v = cls(featurize)
        v.w = np.asarray(d["weights"], dtype=float)
        v.mu = np.asarray(d["mu"], dtype=float)
        v.sd = np.asarray(d["sd"], dtype=float)
        v.n_samples = int(d.get("n_samples", 0))
        return v
