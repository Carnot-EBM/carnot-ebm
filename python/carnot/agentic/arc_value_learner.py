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
    nz = g != 0
    ys, xs = np.where(nz)
    spread = float((np.std(xs) + np.std(ys)) / max(1, max(g.shape))) if len(xs) else 0.0
    dom = max((v for k, v in cnt.items() if k != 0), default=0) / max(1, n_nonzero)
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
        s = int(round(g.size**0.5))
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


_V3_V2_LEN = 41
_V3_OBJECT_RELATIONAL_NAMES = [
    "pair_count_norm",
    "pair_manhattan_min_norm",
    "pair_manhattan_mean_norm",
    "pair_manhattan_max_norm",
    "pair_manhattan_std_norm",
    "correspondence_known",
    "correspondence_match_fraction",
    "correspondence_unmatched_prev_fraction",
    "correspondence_unmatched_cur_fraction",
    "correspondence_displacement_min_norm",
    "correspondence_displacement_mean_norm",
    "correspondence_displacement_max_norm",
    "correspondence_displacement_std_norm",
    "correspondence_area_delta_mean_norm",
]
_V3_FRAME_DELTA_NAMES = [
    "delta_known",
    "changed_fraction",
    "nonzero_delta_signed_fraction",
    "nonzero_delta_abs_fraction",
    "color_hist_l1_delta",
    "object_count_delta_signed_norm",
    "centroid_shift_norm",
    "level_delta_norm",
    "shape_changed",
]
_V3_ACTION_CONDITIONED_NAMES = [
    "action_known",
    "action_1",
    "action_2",
    "action_3",
    "action_4",
    "action_5",
    "action_6",
]
_V3_PREDICATE_DISTANCE_NAMES = [
    "goal_known",
    "goal_grid_mismatch_fraction",
    "goal_nonzero_delta_abs_fraction",
    "goal_color_hist_l1",
    "goal_object_count_delta_abs_norm",
    "goal_centroid_distance_norm",
    "goal_pairwise_mean_delta_abs_norm",
    "goal_pairwise_max_delta_abs_norm",
]


def cross_game_feature_slices_v3() -> dict[str, tuple[int, int]]:
    """REQ-LEARN-4476: stable feature-class slices for v3 ablations."""
    lengths = {
        "v2": _V3_V2_LEN,
        "object_relational": len(_V3_OBJECT_RELATIONAL_NAMES),
        "frame_delta": len(_V3_FRAME_DELTA_NAMES),
        "action_conditioned": len(_V3_ACTION_CONDITIONED_NAMES),
        "predicate_distance": len(_V3_PREDICATE_DISTANCE_NAMES),
    }
    out: dict[str, tuple[int, int]] = {}
    cur = 0
    for name, n in lengths.items():
        out[name] = (cur, cur + n)
        cur += n
    return out


def cross_game_feature_names_v3() -> list[str]:
    """REQ-LEARN-4476: human-readable v3 feature names for artifacts."""
    return (
        [f"v2_{i}" for i in range(_V3_V2_LEN)]
        + [f"object_relational_{n}" for n in _V3_OBJECT_RELATIONAL_NAMES]
        + [f"frame_delta_{n}" for n in _V3_FRAME_DELTA_NAMES]
        + [f"action_conditioned_{n}" for n in _V3_ACTION_CONDITIONED_NAMES]
        + [f"predicate_distance_{n}" for n in _V3_PREDICATE_DISTANCE_NAMES]
    )


def cross_game_feature_names_v3_value_routing() -> list[str]:
    """REQ-LEARN-4652: feature names for the cheap live value-routing subset."""

    return [f"v2_{i}" for i in range(_V3_V2_LEN)] + [
        f"frame_delta_{name}" for name in _V3_FRAME_DELTA_NAMES
    ]


def _grid2d(frame: Any) -> np.ndarray:
    from carnot.agentic.arc_agi3_world_model import grid_of

    g = np.asarray(grid_of(frame), dtype=float)
    if g.ndim == 1:
        side = int(round(g.size**0.5))
        g = g.reshape(side, side) if side * side == g.size else g.reshape(1, -1)
    return g


def _component_stats_from_grid(g: np.ndarray) -> list[dict[str, float]]:
    """4-connectivity non-background components -> per-component stats (centroid / area / dominant color /
    bbox). ``scipy.ndimage.label`` fast path (~34x faster than the pure-python flood fill it replaces);
    falls back to the original flood fill when scipy is absent (the live Kaggle kernel may lack it). The
    downstream features are order-invariant (min/mean/max/std + all-pairs), so only the SET of component
    stats must match -- verified identical over 40 random grids (2026-06-23)."""
    vals, counts = np.unique(g, return_counts=True)
    bg = float(vals[counts.argmax()]) if vals.size else 0.0
    mask = g != bg
    if not mask.any():
        return []
    try:
        from scipy import ndimage  # fast path
    except Exception:
        ndimage = None
    comps: list[dict[str, float]] = []
    if ndimage is not None:
        labels, n = ndimage.label(mask)  # default = 4-connectivity (same cross neighbourhood)
        ys_idx, xs_idx = np.indices(g.shape)
        for k in range(1, n + 1):
            m = labels == k
            cy_, cx_, colors = ys_idx[m], xs_idx[m], g[m]
            cvals, ccounts = np.unique(colors, return_counts=True)
            comps.append(
                {
                    "cy": float(cy_.mean()),
                    "cx": float(cx_.mean()),
                    "area": float(int(m.sum())),
                    "color": float(cvals[ccounts.argmax()]),
                    "y0": float(cy_.min()),
                    "y1": float(cy_.max()),
                    "x0": float(cx_.min()),
                    "x1": float(cx_.max()),
                }
            )
        return comps
    # ---- pure-python fallback (original 4-neighbour flood fill; identical output) ----
    h, w = g.shape
    seen = np.zeros_like(mask, dtype=bool)
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            colors = [float(g[y, x]) for y, x in cells]
            vals2, counts2 = np.unique(colors, return_counts=True)
            comps.append(
                {
                    "cy": float(np.mean(ys)),
                    "cx": float(np.mean(xs)),
                    "area": float(len(cells)),
                    "color": float(vals2[counts2.argmax()]),
                    "y0": float(min(ys)),
                    "y1": float(max(ys)),
                    "x0": float(min(xs)),
                    "x1": float(max(xs)),
                }
            )
    return comps


def _safe_norm(g: np.ndarray) -> float:
    return float(max(1, g.shape[0] + g.shape[1] - 2))


def _summary4(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.asarray(values, dtype=float)
    return [float(arr.min()), float(arr.mean()), float(arr.max()), float(arr.std())]


def _pairwise_manhattan(comps: Sequence[dict[str, float]], norm: float) -> list[float]:
    ds = []
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ds.append(
                (abs(comps[i]["cy"] - comps[j]["cy"]) + abs(comps[i]["cx"] - comps[j]["cx"])) / norm
            )
    return ds


def _weighted_centroid(comps: Sequence[dict[str, float]]) -> tuple[float, float] | None:
    total = sum(c["area"] for c in comps)
    if total <= 0:
        return None
    cy = sum(c["cy"] * c["area"] for c in comps) / total
    cx = sum(c["cx"] * c["area"] for c in comps) / total
    return float(cy), float(cx)


def _color_hist_l1(a: np.ndarray, b: np.ndarray) -> float:
    vals = sorted(set(a.flatten().tolist()) | set(b.flatten().tolist()))
    if not vals:
        return 0.0
    aa = {v: float((a == v).sum()) / max(1, a.size) for v in vals}
    bb = {v: float((b == v).sum()) / max(1, b.size) for v in vals}
    return float(sum(abs(aa[v] - bb[v]) for v in vals) / 2.0)


def _level(frame: Any) -> float:
    return float(getattr(frame, "levels_completed", 0) or 0)


def _object_relational_features(g: np.ndarray, previous_frame: Any | None) -> list[float]:
    comps = _component_stats_from_grid(g)
    norm = _safe_norm(g)
    pair_ds = _pairwise_manhattan(comps, norm)
    pair = _summary4(pair_ds)
    pair_count_norm = float(min(len(pair_ds), 64) / 64.0)
    if previous_frame is None:
        return [pair_count_norm, *pair, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    prev_g = _grid2d(previous_frame)
    prev = _component_stats_from_grid(prev_g)
    if not prev and not comps:
        return [pair_count_norm, *pair, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    unused = set(range(len(prev)))
    disp: list[float] = []
    area_delta: list[float] = []
    for cur in comps:
        if not unused:
            break
        same_color = [i for i in unused if prev[i]["color"] == cur["color"]]
        candidates = same_color or list(unused)
        best = min(
            candidates,
            key=lambda i: abs(prev[i]["cy"] - cur["cy"]) + abs(prev[i]["cx"] - cur["cx"]),
        )
        unused.remove(best)
        disp.append((abs(prev[best]["cy"] - cur["cy"]) + abs(prev[best]["cx"] - cur["cx"])) / norm)
        area_delta.append(abs(prev[best]["area"] - cur["area"]) / max(1.0, g.size))

    denom = float(max(1, max(len(prev), len(comps))))
    unmatched_prev = len(unused) / denom
    unmatched_cur = max(0, len(comps) - len(disp)) / denom
    disp_stats = _summary4(disp)
    return [
        pair_count_norm,
        *pair,
        1.0,
        float(len(disp) / denom),
        float(unmatched_prev),
        float(unmatched_cur),
        *disp_stats,
        float(np.mean(area_delta)) if area_delta else 0.0,
    ]


def _frame_delta_features(g: np.ndarray, frame: Any, previous_frame: Any | None) -> list[float]:
    if previous_frame is None:
        return [0.0] * len(_V3_FRAME_DELTA_NAMES)
    prev = _grid2d(previous_frame)
    shape_changed = float(prev.shape != g.shape)
    if prev.shape == g.shape:
        changed = float((prev != g).sum()) / max(1, g.size)
        hist = _color_hist_l1(prev, g)
    else:
        changed = 1.0
        hist = 1.0
    nz_cur = float((g != 0).sum())
    nz_prev = float((prev != 0).sum())
    cur_comps = _component_stats_from_grid(g)
    prev_comps = _component_stats_from_grid(prev)
    cur_cent = _weighted_centroid(cur_comps)
    prev_cent = _weighted_centroid(prev_comps)
    if cur_cent is None or prev_cent is None:
        centroid_shift = 0.0
    else:
        centroid_shift = (
            abs(cur_cent[0] - prev_cent[0]) + abs(cur_cent[1] - prev_cent[1])
        ) / _safe_norm(g)
    return [
        1.0,
        changed,
        float((nz_cur - nz_prev) / max(1, g.size)),
        float(abs(nz_cur - nz_prev) / max(1, g.size)),
        hist,
        float((len(cur_comps) - len(prev_comps)) / 32.0),
        float(centroid_shift),
        float((_level(frame) - _level(previous_frame)) / 32.0),
        shape_changed,
    ]


def _action_features(action_id: Any | None) -> list[float]:
    out = [0.0] * len(_V3_ACTION_CONDITIONED_NAMES)
    if action_id is None:
        return out
    try:
        if isinstance(action_id, (tuple, list)):
            aid = int(action_id[0])
        elif hasattr(action_id, "name") and str(action_id.name).startswith("ACTION"):
            aid = int(str(action_id.name).replace("ACTION", ""))
        elif isinstance(action_id, str) and action_id.startswith("ACTION"):
            aid = int(action_id.replace("ACTION", ""))
        else:
            aid = int(action_id)
    except (TypeError, ValueError):
        return out
    out[0] = 1.0
    if 1 <= aid <= 6:
        out[aid] = 1.0
    return out


def _predicate_distance_features(g: np.ndarray, goal_frame: Any | None) -> list[float]:
    if goal_frame is None:
        return [0.0] * len(_V3_PREDICATE_DISTANCE_NAMES)
    goal = _grid2d(goal_frame)
    if goal.shape == g.shape:
        mismatch = float((goal != g).sum()) / max(1, g.size)
        hist = _color_hist_l1(g, goal)
    else:
        mismatch = 1.0
        hist = 1.0
    nz_delta = abs(float((goal != 0).sum()) - float((g != 0).sum())) / max(
        1, max(goal.size, g.size)
    )
    comps = _component_stats_from_grid(g)
    goal_comps = _component_stats_from_grid(goal)
    obj_delta = abs(len(goal_comps) - len(comps)) / 32.0
    cur_cent = _weighted_centroid(comps)
    goal_cent = _weighted_centroid(goal_comps)
    if cur_cent is None or goal_cent is None:
        centroid = 0.0
    else:
        centroid = (abs(cur_cent[0] - goal_cent[0]) + abs(cur_cent[1] - goal_cent[1])) / _safe_norm(
            g
        )
    cur_pairs = _pairwise_manhattan(comps, _safe_norm(g))
    goal_pairs = _pairwise_manhattan(goal_comps, _safe_norm(goal))
    cur_mean = float(np.mean(cur_pairs)) if cur_pairs else 0.0
    cur_max = float(np.max(cur_pairs)) if cur_pairs else 0.0
    goal_mean = float(np.mean(goal_pairs)) if goal_pairs else 0.0
    goal_max = float(np.max(goal_pairs)) if goal_pairs else 0.0
    return [
        1.0,
        mismatch,
        float(nz_delta),
        hist,
        float(obj_delta),
        float(centroid),
        abs(cur_mean - goal_mean),
        abs(cur_max - goal_max),
    ]


def cross_game_features_v3(
    frame: Any,
    previous_frame: Any | None = None,
    action_id: Any | None = None,
    goal_frame: Any | None = None,
) -> list[float]:
    """REQ-LEARN-4476: v2 plus relational, delta, action, and predicate-distance context.

    The optional context is used by the offline trainer. With only `frame`, the
    function still emits a stable vector so existing live loaders degrade to a
    frame-only v3 view instead of breaking.
    """
    g = _grid2d(frame)
    v2 = cross_game_features_v2(frame)
    return [
        *v2,
        *_object_relational_features(g, previous_frame),
        *_frame_delta_features(g, frame, previous_frame),
        *_action_features(action_id),
        *_predicate_distance_features(g, goal_frame),
    ]


def cross_game_features_v3_value_routing(
    frame: Any,
    previous_frame: Any | None = None,
    action_id: Any | None = None,
    goal_frame: Any | None = None,
) -> list[float]:
    """REQ-LEARN-4652: cheap live routing features: v2 + frame-delta only.

    The function accepts the full v3 context signature so callers can swap it in
    for `cross_game_features_v3`, but it intentionally ignores action and goal
    context. Those classes were measured as dead weight for live routing; the
    previous frame is the only optional context used here.
    """

    _ = action_id, goal_frame
    g = _grid2d(frame)
    return [
        *cross_game_features_v2(frame),
        *_frame_delta_features(g, frame, previous_frame),
    ]


def collect_trajectory_data(
    env: Any,
    solver: Any,
    prefix: Sequence[str],
    level_path: Sequence[str],
    featurize: Callable[[Any], Sequence[float]],
):
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
        p.write_text(
            json.dumps(
                {
                    "schema": "carnot_arc_learned_verifier_v1",
                    "kind": "linear_value_head",
                    "weights": self.w.tolist(),  # [feature weights..., bias]
                    "feature_names": (meta or {}).get("feature_names"),
                    "trained_games": (meta or {}).get("trained_games"),
                    "n_samples": self.n_samples,
                    "provenance": (meta or {}).get("provenance"),
                },
                indent=2,
            )
        )
        return p

    @classmethod
    def load(
        cls, path: str | Path, featurize: Callable[[Any], Sequence[float]]
    ) -> "LearnedVerifier":
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

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        iters: int = 800,
        lr: float = 0.5,
        l2: float = 1e-3,
    ) -> "DiscriminativeVerifier":
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)  # 1 = on winning path, 0 = off-path / dead-end
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
        return self.proba_features(self.featurize(frame))

    def proba_features(self, features: Sequence[float]) -> float:
        """Score a cached feature vector without needing to keep the original frame.

        The step-wise live explorer stores compact frame-only feature vectors on
        frontier nodes. Re-scoring those cached vectors after an online refit is
        cheaper and less fragile than keeping full rendered frames around just so
        the logistic head can call its featurizer again.
        """
        if self.w is None or self.mu is None or self.sd is None:
            return 0.5
        z = (np.asarray([float(v) for v in features], dtype=float) - self.mu) / self.sd
        z = np.r_[z, 1.0]
        return float(1.0 / (1.0 + np.exp(-(z @ self.w))))

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        if self.w is None:
            raise ValueError("nothing to save: classifier is untrained")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "schema": "carnot_arc_discriminative_verifier_v1",
                    "kind": "logistic_win_reachability",
                    "weights": self.w.tolist(),
                    "mu": self.mu.tolist(),
                    "sd": self.sd.tolist(),
                    "feature_names": (meta or {}).get("feature_names"),
                    "trained_games": (meta or {}).get("trained_games"),
                    "n_samples": self.n_samples,
                    "provenance": (meta or {}).get("provenance"),
                },
                indent=2,
            )
        )
        return p

    @classmethod
    def load(
        cls, path: str | Path, featurize: Callable[[Any], Sequence[float]]
    ) -> "DiscriminativeVerifier":
        d = json.loads(Path(path).read_text())
        v = cls(featurize)
        v.w = np.asarray(d["weights"], dtype=float)
        v.mu = np.asarray(d["mu"], dtype=float)
        v.sd = np.asarray(d["sd"], dtype=float)
        v.n_samples = int(d.get("n_samples", 0))
        return v


DAGGER_WIN_REACHABILITY_VALUE_HEAD_SCHEMA = (
    "carnot_arc_dagger_win_reachability_value_head_v1"
)
DAGGER_WIN_REACHABILITY_FEATURE_SUBSET = "cross_game_features_v3:v2_plus_frame_delta"


class DaggerWinReachabilityValueHead:
    """REQ-LEARN-4665: cheap DAgger-corrected win-reachability value route.

    The underlying model is the existing logistic `DiscriminativeVerifier`, but
    the live frontier wants a lower-is-better value term. This wrapper exposes
    `1 - P(win-reachable)` as a tiny CPU cost while keeping the learned head
    oracle-distinct from the executable reproduction oracle used for labels.
    """

    feature_subset = DAGGER_WIN_REACHABILITY_FEATURE_SUBSET
    verifier_is_oracle = False

    def __init__(self, verifier: DiscriminativeVerifier) -> None:
        self.verifier = verifier

    @property
    def n_samples(self) -> int:
        return int(self.verifier.n_samples)

    def proba_features(self, features: Sequence[float]) -> float:
        return float(self.verifier.proba_features(features))

    def cost_features(self, features: Sequence[float]) -> float:
        return float(max(0.0, min(1.0, 1.0 - self.proba_features(features))))

    def __call__(self, frame: Any, previous_frame: Any | None = None) -> float:
        features = cross_game_features_v3_value_routing(frame, previous_frame=previous_frame)
        return self.cost_features(features)

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        if self.verifier.w is None or self.verifier.mu is None or self.verifier.sd is None:
            raise ValueError("nothing to save: DAgger value head is untrained")
        payload = {
            "schema": DAGGER_WIN_REACHABILITY_VALUE_HEAD_SCHEMA,
            "kind": "dagger_win_reachability_value_head",
            "weights": self.verifier.w.tolist(),
            "mu": self.verifier.mu.tolist(),
            "sd": self.verifier.sd.tolist(),
            "feature_subset": self.feature_subset,
            "feature_names": cross_game_feature_names_v3_value_routing(),
            "n_samples": self.verifier.n_samples,
            "verifier_is_oracle": False,
            "provenance": (meta or {}).get("provenance"),
            "trained_games": (meta or {}).get("trained_games"),
            "spec_refs": (meta or {}).get("spec_refs"),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DaggerWinReachabilityValueHead":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != DAGGER_WIN_REACHABILITY_VALUE_HEAD_SCHEMA:
            raise ValueError("not a DAgger win-reachability value-head checkpoint")
        verifier = DiscriminativeVerifier(lambda row: row)
        verifier.w = np.asarray(payload["weights"], dtype=float)
        verifier.mu = np.asarray(payload["mu"], dtype=float)
        verifier.sd = np.asarray(payload["sd"], dtype=float)
        verifier.n_samples = int(payload.get("n_samples", 0))
        return cls(verifier)


def fit_dagger_win_reachability_value_head(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
    *,
    iters: int = 800,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> DaggerWinReachabilityValueHead:
    """REQ-LEARN-4665: fit the DAgger-lite live-frontier discriminator."""

    x_list = [[float(v) for v in row] for row in x_rows]
    y_list = [float(v) for v in y_rows]
    if not x_list:
        raise ValueError("DAgger value head requires at least one feature row")
    positives = sum(1 for value in y_list if value >= 0.5)
    negatives = len(y_list) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("DAgger value head requires positive and negative labels")
    width = len(x_list[0])
    if any(len(row) != width for row in x_list):
        raise ValueError("DAgger value head feature rows must have a stable width")
    verifier = DiscriminativeVerifier(lambda row: row).fit(
        x_list,
        y_list,
        iters=iters,
        lr=lr,
        l2=l2,
    )
    return DaggerWinReachabilityValueHead(verifier)
