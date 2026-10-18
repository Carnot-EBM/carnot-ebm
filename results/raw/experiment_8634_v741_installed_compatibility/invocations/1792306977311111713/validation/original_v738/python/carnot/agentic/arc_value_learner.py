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

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

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
    bbox). ``scipy.ndimage.label`` fast path; falls back to the original flood fill when scipy is absent.
    CORRECTED 2026-06-30: the original "~34x faster" claim was never measured on real target hardware and
    was wrong. Verified on the actual Kaggle sandbox (carnot-arc-scipy-diag kernel, scipy 1.16.3 confirmed
    present): scipy path 663us/call vs pure-python fallback 936us/call -- a 1.41x difference, not 34x.
    The fallback risk itself is CLOSED (scipy confirmed present on Kaggle; the slow path never triggers
    live), but neither path is a large lever on its own -- see ops/known-issues.md "energy distillation"
    task for where the real per-node cost is (the O(components^2) greedy frame-matching loop on top of
    this, not the labeling step itself). The downstream features are order-invariant (min/mean/max/std +
    all-pairs), so only the SET of component stats must match -- verified identical over 40 random grids
    (2026-06-23)."""
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


class CrossGameFrameContextV3(NamedTuple):
    """The (frame, previous_frame, goal_frame)-only pieces of `cross_game_features_v3` -- every
    piece except `_action_features(action_id)`. Callers that score MULTIPLE candidate actions
    against the SAME frame (e.g. `CrossGameDiscriminativeCandidateRouter.rank()`) compute this
    ONCE via `cross_game_frame_context_v3()` and pass it to `cross_game_features_v3(...,
    frame_context=...)` for every candidate, instead of paying `_object_relational_features`'s
    O(components^2) greedy frame-matching loop (see this module's 2026-06-30 docstring note: the
    real per-node cost here, not the scipy-vs-fallback labeling step) once per candidate. Found
    2026-07-15 during ARC-AGI-3 submission-prep: this per-candidate recomputation was the
    dominant remaining cause of a local-submission-gate regression (lp85/m0r0/sp80 losing to the
    verified baseline) even after fixing the structurally-identical bug in
    arc_color_blob_salience.py."""

    v2: list[float]
    object_relational: list[float]
    frame_delta: list[float]
    predicate_distance: list[float]


def cross_game_frame_context_v3(
    frame: Any, previous_frame: Any | None, goal_frame: Any | None
) -> CrossGameFrameContextV3:
    g = _grid2d(frame)
    return CrossGameFrameContextV3(
        v2=cross_game_features_v2(frame),
        object_relational=_object_relational_features(g, previous_frame),
        frame_delta=_frame_delta_features(g, frame, previous_frame),
        predicate_distance=_predicate_distance_features(g, goal_frame),
    )


def cross_game_features_v3(
    frame: Any,
    previous_frame: Any | None = None,
    action_id: Any | None = None,
    goal_frame: Any | None = None,
    *,
    frame_context: CrossGameFrameContextV3 | None = None,
) -> list[float]:
    """REQ-LEARN-4476: v2 plus relational, delta, action, and predicate-distance context.

    The optional context is used by the offline trainer. With only `frame`, the
    function still emits a stable vector so existing live loaders degrade to a
    frame-only v3 view instead of breaking.

    `frame_context`: an optional pre-computed `cross_game_frame_context_v3(frame,
    previous_frame, goal_frame)` result -- everything in the output EXCEPT the
    `_action_features(action_id)` slice is independent of `action_id`, so a caller scoring many
    candidates against the same frame can compute it once (see `CrossGameFrameContextV3`'s
    docstring for the O(components^2)-per-candidate incident this fixes). When omitted, computed
    fresh exactly as before (unchanged behavior/output for every other existing caller).
    """
    if frame_context is None:
        frame_context = cross_game_frame_context_v3(frame, previous_frame, goal_frame)
    return [
        *frame_context.v2,
        *frame_context.object_relational,
        *frame_context.frame_delta,
        *_action_features(action_id),
        *frame_context.predicate_distance,
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


@dataclass
class ObjectCentricProposalConfig:
    """REQ-ARC-WMTE-4700: deployable object-slot proposal conditioning."""

    enabled: bool = True
    neighborhood_radius: int = 2
    max_slots: int = 256
    max_augmented_clicks: int = 192
    slot_score_weight: float = 1.0
    offpath_effect_bonus: float = 0.25
    no_op_penalty: float = 0.4
    surfacing_ranker_enabled: bool = False
    surfacing_ranker_weight: float = 1.0
    surfacing_refit_min_samples: int = 4


def _dominant_background(g: np.ndarray) -> float:
    vals, counts = np.unique(g, return_counts=True)
    return float(vals[counts.argmax()]) if vals.size else 0.0


def _clip_point(x: float, y: float, g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    return max(0, min(w - 1, int(round(x)))), max(0, min(h - 1, int(round(y))))


def _candidate_action_id(candidate: Any) -> int:
    try:
        if isinstance(candidate, Mapping):
            return int(candidate.get("action", candidate.get("action_id", 0)) or 0)
        return int(getattr(candidate, "action", getattr(candidate, "action_id", 0)) or 0)
    except Exception:
        return 0


def _candidate_data(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get("data")
    return getattr(candidate, "data", None)


def _proposal_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    action = int(candidate.get("action") or 0)
    data = candidate.get("data")
    if action == 6 and isinstance(data, Mapping):
        return (6, int(data.get("x", -1)), int(data.get("y", -1)))
    return (action,)


def _as_candidate_row(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        row = dict(candidate)
        row["action"] = _candidate_action_id(candidate)
        row["data"] = _candidate_data(candidate)
        return row
    return {"action": _candidate_action_id(candidate), "data": _candidate_data(candidate)}


def object_centric_proposal_features(
    frame: Any,
    *,
    previous_frame: Any | None = None,
    action_id: Any | None = None,
    goal_frame: Any | None = None,
) -> list[float]:
    """REQ-ARC-WMTE-4700: object/relational proposal features with action context.

    This is the deployable proposal-side view: v2 context plus object
    correspondences, frame delta, and the candidate action. Goal context remains
    optional and is normally absent in live exploration.
    """

    return cross_game_features_v3(
        frame,
        previous_frame=previous_frame,
        action_id=action_id,
        goal_frame=goal_frame,
    )


def object_centric_slots(
    frame: Any,
    *,
    previous_frame: Any | None = None,
    neighborhood_radius: int = 2,
    max_slots: int = 256,
) -> list[dict[str, Any]]:
    """REQ-ARC-WMTE-4700: connected-component slots plus relational gap keypoints."""

    _ = previous_frame
    g = _grid2d(frame)
    comps = _component_stats_from_grid(g)
    if not comps:
        return []
    bg = _dominant_background(g)
    color_counts = {float(v): int((g == v).sum()) for v in np.unique(g)}
    slots: dict[tuple[int, int], dict[str, Any]] = {}

    def add_slot(
        x: float,
        y: float,
        *,
        slot_type: str,
        color: float,
        base_score: float,
        distance: float = 0.0,
    ) -> None:
        px, py = _clip_point(x, y, g)
        rarity = 1.0 / (1.0 + float(color_counts.get(float(color), 0)))
        local = g[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2]
        local_density = float((local != bg).mean()) if local.size else 0.0
        score = float(base_score + rarity + 0.35 * local_density - 0.05 * distance)
        key = (px, py)
        existing = slots.get(key)
        if existing is not None and float(existing["score"]) >= score:
            return
        slots[key] = {
            "x": int(px),
            "y": int(py),
            "slot_type": slot_type,
            "support_color": int(color),
            "score": score,
            "local_object_density": local_density,
        }

    radius = max(0, int(neighborhood_radius))
    for comp in comps:
        color = float(comp["color"])
        area = float(comp["area"])
        base = 1.0 + min(area, 64.0) / 64.0
        add_slot(
            comp["cx"],
            comp["cy"],
            slot_type="component_centroid",
            color=color,
            base_score=base,
        )
        add_slot(
            (comp["x0"] + comp["x1"]) / 2.0,
            (comp["y0"] + comp["y1"]) / 2.0,
            slot_type="component_bbox_center",
            color=color,
            base_score=base - 0.05,
        )
        if radius <= 0:
            continue
        cx, cy = _clip_point(comp["cx"], comp["cy"], g)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                distance = abs(dx) + abs(dy)
                if distance == 0 or distance > radius:
                    continue
                px, py = cx + dx, cy + dy
                if not (0 <= py < g.shape[0] and 0 <= px < g.shape[1]):
                    continue
                if float(g[py, px]) != bg:
                    continue
                add_slot(
                    px,
                    py,
                    slot_type="object_neighborhood_gap",
                    color=color,
                    base_score=0.85 + (radius - distance + 1) / max(1.0, radius + 1.0),
                    distance=float(distance),
                )

    by_color: dict[float, list[dict[str, float]]] = {}
    for comp in comps:
        by_color.setdefault(float(comp["color"]), []).append(comp)
    for color, color_comps in by_color.items():
        if len(color_comps) < 2:
            continue
        for i, left in enumerate(color_comps):
            for right in color_comps[i + 1 :]:
                dy = abs(left["cy"] - right["cy"])
                dx = abs(left["cx"] - right["cx"])
                if max(dx, dy) > 12 or min(dx, dy) > 2:
                    continue
                add_slot(
                    (left["cx"] + right["cx"]) / 2.0,
                    (left["cy"] + right["cy"]) / 2.0,
                    slot_type="object_constellation_gap",
                    color=color,
                    base_score=1.2,
                    distance=float(dx + dy),
                )

    return sorted(
        slots.values(),
        key=lambda row: (-float(row["score"]), row["slot_type"], int(row["y"]), int(row["x"])),
    )[: max(1, int(max_slots))]


STRUCTURAL_ALIGNMENT_GOAL_EXPRESSION = "structural_piece_sprite_alignment_over_detected_objects"


def _same_color_components(g: np.ndarray) -> list[dict[str, Any]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rows: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            color = int(g[y, x])
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(g[ny, nx]) == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            rows.append(
                {
                    "color": color,
                    "area": len(cells),
                    "x0": min(xs),
                    "y0": min(ys),
                    "x1": max(xs),
                    "y1": max(ys),
                    "cx": float(sum(xs)) / float(len(xs)),
                    "cy": float(sum(ys)) / float(len(ys)),
                }
            )
    return rows


def _is_solid_2x2_component(component: Mapping[str, Any]) -> bool:
    return (
        int(component.get("area") or 0) == 4
        and int(component.get("x1") or 0) - int(component.get("x0") or 0) == 1
        and int(component.get("y1") or 0) - int(component.get("y0") or 0) == 1
    )


def _corner_marker_pieces(components: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_color: dict[int, set[tuple[int, int]]] = {}
    for component in components:
        if int(component.get("area") or 0) != 1:
            continue
        color = int(component.get("color") or 0)
        by_color.setdefault(color, set()).add(
            (int(component.get("x0") or 0), int(component.get("y0") or 0))
        )

    pieces: list[dict[str, Any]] = []
    for color, points in by_color.items():
        for x0, y0 in sorted(points):
            corners = {(x0, y0), (x0 + 3, y0), (x0, y0 + 3), (x0 + 3, y0 + 3)}
            if not corners.issubset(points):
                continue
            pieces.append(
                {
                    "kind": "corner_marker_piece",
                    "color": int(color),
                    "bbox": [int(x0), int(y0), int(x0 + 3), int(y0 + 3)],
                    "target_goal_bbox": [int(x0 + 1), int(y0 + 1), int(x0 + 2), int(y0 + 2)],
                    "center": [float(x0 + 1.5), float(y0 + 1.5)],
                }
            )
    return pieces


def _marker_goal_distance(piece: Mapping[str, Any], goal: Mapping[str, Any]) -> int:
    target = list(piece.get("target_goal_bbox") or [])
    bbox = list(goal.get("bbox") or [])
    if len(target) < 2 or len(bbox) < 2:
        return 1_000_000
    return int(abs(int(bbox[0]) - int(target[0])) + abs(int(bbox[1]) - int(target[1])))


def _pair_corner_markers_to_goals(
    pieces: Sequence[Mapping[str, Any]],
    candidate_goals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any] | None]:
    matched: list[dict[str, Any] | None] = [None for _ in pieces]
    used_goals: set[int] = set()
    piece_indices_by_color: dict[int, list[int]] = {}
    goal_indices_by_color: dict[int, list[int]] = {}
    for index, piece in enumerate(pieces):
        piece_indices_by_color.setdefault(int(piece.get("color") or 0), []).append(index)
    for index, goal in enumerate(candidate_goals):
        goal_indices_by_color.setdefault(int(goal.get("color") or 0), []).append(index)

    for color, piece_indices in piece_indices_by_color.items():
        goal_indices = goal_indices_by_color.get(color, [])
        scored: list[tuple[int, int, int, int]] = []
        for piece_index in piece_indices:
            target = list(pieces[piece_index].get("target_goal_bbox") or [])
            for goal_index in goal_indices:
                goal = candidate_goals[goal_index]
                exact = list(goal.get("bbox") or []) == target
                scored.append(
                    (
                        0 if exact else 1,
                        _marker_goal_distance(pieces[piece_index], goal),
                        piece_index,
                        goal_index,
                    )
                )
        for _rank, _distance, piece_index, goal_index in sorted(scored):
            if matched[piece_index] is not None or goal_index in used_goals:
                continue
            matched[piece_index] = dict(candidate_goals[goal_index])
            used_goals.add(goal_index)
    return matched


def detect_marker_pair_shape_alignment(frame: Any) -> dict[str, Any]:
    """REQ-ARC-WMTE-4712: detect corner-marker pieces and same-color goal sprites.

    The A1 object-centric slot builder still runs first, but the alignment
    objects are segmented by same-color components so board backgrounds do not
    merge the moveable corner markers into one global component.
    """

    g = _grid2d(frame).astype(np.int16, copy=False)
    try:
        slots = object_centric_slots(g)
    except Exception:
        slots = []
    components = _same_color_components(g)
    raw_goals = [
        {
            "kind": "goal_sprite",
            "color": int(component["color"]),
            "bbox": [
                int(component["x0"]),
                int(component["y0"]),
                int(component["x1"]),
                int(component["y1"]),
            ],
            "center": [float(component["cx"]), float(component["cy"])],
        }
        for component in components
        if _is_solid_2x2_component(component)
    ]
    colors_with_goals = {int(goal["color"]) for goal in raw_goals}
    pieces = [
        piece
        for piece in _corner_marker_pieces(components)
        if int(piece["color"]) in colors_with_goals
    ]
    piece_colors = {int(piece["color"]) for piece in pieces}
    candidate_goals = [goal for goal in raw_goals if int(goal["color"]) in piece_colors]
    matched_goals = _pair_corner_markers_to_goals(pieces, candidate_goals)
    goals = [goal for goal in matched_goals if goal is not None]

    pairs: list[dict[str, Any]] = []
    for piece, aligned_goal in zip(pieces, matched_goals, strict=True):
        target = list(piece["target_goal_bbox"])
        aligned = aligned_goal is not None and list(aligned_goal["bbox"]) == target
        distance = None
        if aligned_goal is not None:
            distance = _marker_goal_distance(piece, aligned_goal)
        pairs.append(
            {
                "piece": piece,
                "goal": aligned_goal,
                "aligned": bool(aligned),
                "alignment_distance": distance,
            }
        )

    complete = bool(pieces) and all(bool(pair["aligned"]) for pair in pairs)
    return {
        "goal_expression": STRUCTURAL_ALIGNMENT_GOAL_EXPRESSION,
        "detected": bool(pieces and goals),
        "complete": bool(complete),
        "piece_count": int(len(pieces)),
        "goal_count": int(len(goals)),
        "raw_goal_count": int(len(raw_goals)),
        "aligned_piece_count": int(sum(1 for pair in pairs if pair["aligned"])),
        "object_centric_slot_count": int(len(slots)),
        "pieces": pieces,
        "goals": goals,
        "pairs": pairs,
        "verifier_is_oracle": False,
    }


def structural_piece_sprite_alignment_goal(grid: Any) -> bool:
    """REQ-ARC-WMTE-4712: structural piece->sprite goal predicate."""

    return bool(detect_marker_pair_shape_alignment(grid).get("complete"))


def structural_alignment_goal_candidate(grid: Any) -> dict[str, Any] | None:
    """REQ-ARC-WMTE-4712: return an oracle-distinct structural goal candidate."""

    diagnostics = detect_marker_pair_shape_alignment(grid)
    if not diagnostics.get("detected"):
        return None
    return {
        "name": STRUCTURAL_ALIGNMENT_GOAL_EXPRESSION,
        "goal_expression": STRUCTURAL_ALIGNMENT_GOAL_EXPRESSION,
        "predicate": structural_piece_sprite_alignment_goal,
        "diagnostics": diagnostics,
        "verifier_is_oracle": False,
    }


def _float_features(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.astype(float).reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(0.0)
        return out
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _align_features(features: Sequence[float], dim: int) -> list[float]:
    row = [float(value) for value in features]
    if len(row) < dim:
        return row + [0.0] * (dim - len(row))
    return row[:dim]


class OffPathCalibratedProposalRanker:
    """REQ-ARC-WMTE-4713: oracle-distinct ranker over live off-path candidates."""

    verifier_is_oracle = False

    def __init__(self, *, iters: int = 300, lr: float = 0.4, l2: float = 1e-3) -> None:
        self.iters = max(1, int(iters))
        self.lr = float(lr)
        self.l2 = float(l2)
        self._samples: list[tuple[list[float], float]] = []
        self._verifier: DiscriminativeVerifier | None = None
        self._feature_dim = 0
        self._fit_count = 0

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "OffPathCalibratedProposalRanker":
        samples: list[tuple[list[float], float]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            features = _float_features(row.get("features", row.get("surfacing_features")))
            if not features:
                continue
            label = 1.0 if float(row.get("label", row.get("y", 0.0)) or 0.0) >= 0.5 else 0.0
            samples.append((features, label))
        self._samples = samples
        self._fit_from_samples()
        return self

    def record_sample(self, features: Sequence[float], label: float) -> None:
        clean = _float_features(features)
        if not clean:
            return
        self._samples.append((clean, 1.0 if float(label) >= 0.5 else 0.0))
        self._fit_from_samples()

    def _fit_from_samples(self) -> None:
        labels = [label for _features, label in self._samples]
        if not labels or len(set(labels)) < 2:
            return
        self._feature_dim = max(len(features) for features, _label in self._samples)
        X = [_align_features(features, self._feature_dim) for features, _label in self._samples]
        y = list(labels)
        verifier = DiscriminativeVerifier(lambda _frame: [])
        verifier.fit(X, y, iters=self.iters, lr=self.lr, l2=self.l2)
        self._verifier = verifier
        self._fit_count += 1

    def score_features(self, features: Sequence[float]) -> float:
        if self._verifier is None or self._feature_dim <= 0:
            return 0.5
        return float(self._verifier.proba_features(_align_features(features, self._feature_dim)))

    def rank_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        feature_key: str = "surfacing_features",
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, float, int, dict[str, Any]]] = []
        for index, row in enumerate(rows):
            out = dict(row)
            features = _float_features(out.get(feature_key))
            score = self.score_features(features)
            out["surfacing_verifier_score"] = float(score)
            out["surfacing_ranker_oracle_distinct"] = True
            base = float(
                out.get(
                    "object_centric_combined_score", out.get("object_centric_proposal_score", 0.0)
                )
                or 0.0
            )
            scored.append((float(score), base, index, out))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [row for _score, _base, _index, row in scored]

    def diagnostics(self) -> dict[str, Any]:
        positives = sum(1 for _features, label in self._samples if label >= 0.5)
        negatives = len(self._samples) - positives
        return {
            "enabled": True,
            "offpath_calibrated": self._verifier is not None,
            "samples": int(len(self._samples)),
            "positive_samples": int(positives),
            "negative_samples": int(negatives),
            "feature_dim": int(self._feature_dim),
            "fit_count": int(self._fit_count),
            "verifier_is_oracle": False,
        }


def _candidate_surfacing_features(
    frame: Any,
    row: Mapping[str, Any],
    *,
    slot: Mapping[str, Any] | None,
    object_score: float,
    calibration_score: float,
    previous_frame: Any | None = None,
) -> list[float]:
    explicit = _float_features(row.get("surfacing_features"))
    if explicit:
        return explicit
    try:
        g = _grid2d(frame)
    except Exception:
        g = np.zeros((1, 1), dtype=float)
    h, w = g.shape
    data = row.get("data")
    x_norm = 0.0
    y_norm = 0.0
    is_click = 0.0
    if int(row.get("action") or 0) == 6 and isinstance(data, Mapping):
        is_click = 1.0
        x_norm = float(int(data.get("x", 0))) / float(max(1, w - 1))
        y_norm = float(int(data.get("y", 0))) / float(max(1, h - 1))
    slot_type = str((slot or {}).get("slot_type") or "")
    slot_score = float((slot or {}).get("score") or 0.0)
    density = float((slot or {}).get("local_object_density") or 0.0)
    color = float((slot or {}).get("support_color") or 0.0) / 16.0
    type_names = (
        "component_centroid",
        "component_bbox_center",
        "object_neighborhood_gap",
        "object_constellation_gap",
    )
    try:
        frame_features = cross_game_features_v2(frame)
    except Exception:
        frame_features = [0.0] * _V3_V2_LEN
    return [
        *frame_features,
        *_action_features(row.get("action")),
        is_click,
        x_norm,
        y_norm,
        float(bool(row.get("object_centric_augmented"))),
        float(object_score),
        float(calibration_score),
        slot_score,
        density,
        color,
        *[1.0 if slot_type == name else 0.0 for name in type_names],
        float(previous_frame is not None),
    ]


class ObjectCentricProposalPolicy:
    """REQ-ARC-WMTE-4700: proposal augmenter/ranker for live StepwiseExplorer."""

    verifier_is_oracle = False

    def __init__(
        self,
        config: ObjectCentricProposalConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if config is None:
            self.config = ObjectCentricProposalConfig()
        elif isinstance(config, ObjectCentricProposalConfig):
            self.config = config
        else:
            self.config = ObjectCentricProposalConfig(**dict(config))
        self._candidate_scores = 0
        self._augmented_candidates = 0
        self._last_slot_count = 0
        self._transition_observations = 0
        self._effect_by_key: dict[tuple[Any, ...], list[int]] = {}
        self._surfacing_features_by_key: dict[tuple[Any, ...], list[float]] = {}
        self._surfacing_samples_since_fit = 0
        self.surfacing_ranker = (
            OffPathCalibratedProposalRanker() if self.config.surfacing_ranker_enabled else None
        )

    def calibrate_surfacing_ranker(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> "ObjectCentricProposalPolicy":
        """REQ-ARC-WMTE-4713: fit the proposal ranker on off-path rows."""

        if self.surfacing_ranker is None:
            self.surfacing_ranker = OffPathCalibratedProposalRanker()
            self.config.surfacing_ranker_enabled = True
        self.surfacing_ranker.fit(rows)
        return self

    def _calibration_score(self, row: Mapping[str, Any]) -> float:
        stats = self._effect_by_key.get(_proposal_key(row))
        if not stats:
            return 0.0
        changed, total = stats
        if total <= 0:
            return 0.0
        effect_rate = float(changed) / float(total)
        return self.config.offpath_effect_bonus * effect_rate - self.config.no_op_penalty * (
            1.0 - effect_rate
        )

    def _slot_lookup(
        self, slots: Sequence[Mapping[str, Any]]
    ) -> dict[tuple[int, int], Mapping[str, Any]]:
        return {(int(slot["x"]), int(slot["y"])): slot for slot in slots}

    def rank_candidates(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        previous_frame: Any | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return [_as_candidate_row(candidate) for candidate in candidates]
        rows = [_as_candidate_row(candidate) for candidate in candidates]
        seen = {_proposal_key(row) for row in rows}
        slots = object_centric_slots(
            frame,
            previous_frame=previous_frame,
            neighborhood_radius=self.config.neighborhood_radius,
            max_slots=self.config.max_slots,
        )
        self._last_slot_count = len(slots)
        for slot in slots[: max(0, int(self.config.max_augmented_clicks))]:
            row = {"action": 6, "data": {"x": int(slot["x"]), "y": int(slot["y"])}}
            key = _proposal_key(row)
            if key in seen:
                continue
            row["object_centric_augmented"] = True
            rows.append(row)
            seen.add(key)
            self._augmented_candidates += 1

        slot_by_xy = self._slot_lookup(slots)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, row in enumerate(rows):
            score = 0.0
            slot = None
            if int(row.get("action") or 0) == 6 and isinstance(row.get("data"), Mapping):
                x = int(row["data"].get("x", -1))
                y = int(row["data"].get("y", -1))
                slot = slot_by_xy.get((x, y))
                if slot is not None:
                    score += self.config.slot_score_weight * float(slot.get("score") or 0.0)
            action_features = _action_features(row.get("action"))
            if action_features and action_features[0] > 0:
                score += 0.01
            calibration_score = self._calibration_score(row)
            score += calibration_score
            surfacing_features = _candidate_surfacing_features(
                frame,
                row,
                slot=slot,
                object_score=score,
                calibration_score=calibration_score,
                previous_frame=previous_frame,
            )
            combined = float(score)
            out = dict(row)
            out["object_centric_proposal_score"] = float(score)
            if slot is not None:
                out["object_centric_slot"] = dict(slot)
            if self.surfacing_ranker is not None:
                out["surfacing_features"] = list(surfacing_features)
                surfacing_score = self.surfacing_ranker.score_features(surfacing_features)
                out["surfacing_verifier_score"] = float(surfacing_score)
                out["surfacing_ranker_oracle_distinct"] = True
                combined += self.config.surfacing_ranker_weight * (float(surfacing_score) - 0.5)
                self._surfacing_features_by_key[_proposal_key(out)] = list(surfacing_features)
            out["object_centric_combined_score"] = float(combined)
            scored.append((float(combined), index, out))
            self._candidate_scores += 1
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [row for _score, _index, row in scored]

    def record_transition(
        self,
        previous_frame: Any,
        next_frame: Any,
        action: Mapping[str, Any],
    ) -> None:
        if not self.config.enabled:
            return
        try:
            before = _grid2d(previous_frame)
            after = _grid2d(next_frame)
            changed = int(before.shape != after.shape or bool((before != after).any()))
        except Exception:
            changed = 0
        row = _as_candidate_row(action)
        key = _proposal_key(row)
        stats = self._effect_by_key.setdefault(key, [0, 0])
        stats[0] += changed
        stats[1] += 1
        self._transition_observations += 1
        if self.surfacing_ranker is not None:
            features = self._surfacing_features_by_key.get(key)
            if features is not None:
                self.surfacing_ranker.record_sample(features, float(changed))
                self._surfacing_samples_since_fit += 1

    def diagnostics(self) -> dict[str, Any]:
        surfacing = (
            {"enabled": False, "offpath_calibrated": False, "verifier_is_oracle": False}
            if self.surfacing_ranker is None
            else self.surfacing_ranker.diagnostics()
        )
        return {
            "enabled": bool(self.config.enabled),
            "representation": "connected_components_object_slots_plus_correspondence_action_context",
            "candidate_scores": int(self._candidate_scores),
            "augmented_candidates": int(self._augmented_candidates),
            "last_slot_count": int(self._last_slot_count),
            "offpath_transition_observations": int(self._transition_observations),
            "offpath_calibrated": self._transition_observations > 0,
            "neighborhood_radius": int(self.config.neighborhood_radius),
            "max_augmented_clicks": int(self.config.max_augmented_clicks),
            "surfacing_ranker": surfacing,
            "verifier_is_oracle": False,
        }


def coerce_object_centric_proposal_policy(
    value: Any,
) -> ObjectCentricProposalPolicy | None:
    """REQ-ARC-WMTE-4700: normalize opt-in object-centric proposal config."""

    if value is None or value is False:
        return None
    if isinstance(value, ObjectCentricProposalPolicy):
        return value if value.config.enabled else None
    if value is True:
        return ObjectCentricProposalPolicy()
    if isinstance(value, ObjectCentricProposalConfig):
        return ObjectCentricProposalPolicy(value) if value.enabled else None
    if isinstance(value, Mapping):
        config = ObjectCentricProposalConfig(**dict(value))
        return ObjectCentricProposalPolicy(config) if config.enabled else None
    return None


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


DAGGER_WIN_REACHABILITY_VALUE_HEAD_SCHEMA = "carnot_arc_dagger_win_reachability_value_head_v1"
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
