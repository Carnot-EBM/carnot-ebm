"""Coordinate-aware click-TARGET features for the live ARC candidate router.

Spec refs: REQ-ARC-FCP-5904, SCENARIO-ARC-FCP-5904.

WHY THIS MODULE EXISTS (the verified defect it repairs)
-------------------------------------------------------
The live candidate router (``arc_discriminative_router.CrossGameDiscriminativeCandidateRouter``)
is COORDINATE-BLIND. It scores a candidate by calling ``cross_game_features_v3(...,
action_id=_action_id(action))``, and ``_action_id`` returns the action *TYPE* integer -- ``6``
for every single click, no matter where the click lands. Downstream, ``cross_game_features_v3``
feeds that integer through ``_action_features()``, a 7-dimensional one-hot; coordinates are not
merely ignored, they are structurally unrepresentable in that vector.

Measured on this machine (offline arcade, public game ``lp85``, ``max_click=48``):

    37 distinct click targets available (``candidate_action_key`` sees ``(6, x, y)``)
     1 distinct value reaches the router (``_action_id`` returns ``6`` for all of them)
     1 distinct score across all 37 targets
    ``rank()`` preserved the input order EXACTLY -- a stable no-op for clicks

The same collapse reproduces on all 19 click-capable public games. So the "learned router"
contributes exactly nothing to click ordering; click order in fact falls back to a static
area x colour-rarity salience sort, whose output is consumed by ``lst.pop(0)``.

This module supplies the missing half: a cheap, coordinate-AWARE description of *what the
agent would be clicking on*, so a ranker can distinguish "click this small salient button"
from "click this large dull background region" -- which is the whole content of a click
decision, and precisely the content the incumbent featurization discards.

ONLINE / WITHIN-GAME ONLY -- AND WHY CROSS-GAME TRANSFER IS DELIBERATELY EXCLUDED
--------------------------------------------------------------------------------
Nothing here is trained across games. Nothing is persisted to disk. No cross-game checkpoint
is loaded, and the featurizer itself is a pure function of (frame, x, y) plus optional
*in-episode* novelty counters that are discarded when the episode ends.

That is a deliberate scope decision, not an oversight:

1. ``ops/exclusion_manifest.yaml`` id ``cross_game_value_transfer_retired_exp4342_v401``
   RETIRES the "ARC cross-game learned value-transfer" direction after three consecutive
   nulls (exp4318, exp4331, exp4342), with ``operator_reopen_required: true`` and
   ``blocked_patterns`` including "cross-game value transfer". Building a cross-game-trained
   click value head would re-run a retired direction.
2. More fundamentally, it is the only thing a hidden-game agent can actually do. Per
   CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent", the scored deliverable is an
   agent handed games it has NEVER seen, which must induce that game's perception, dynamics
   and goal AT RUNTIME. It has no prior exposure to the hidden game to transfer from. The
   only legal fit is from the agent's OWN observed transitions, within the episode.

WHY THE WORK IS SPLIT PER-FRAME / PER-CANDIDATE
----------------------------------------------
This repo has already paid for the lesson twice (see ``CrossGameFrameContextV3``'s docstring
and ``arc_color_blob_salience``'s ``_blob_cache`` comment): a per-CANDIDATE call to
``connected_color_blobs`` turns candidate ranking into an O(candidates x components^2)-class
cost. Profiling a real ``lp85`` episode found 8176 blob segmentations for 500 actions -- once
per candidate rather than once per frame -- 23.1s of a 43.3s run.

So this module is explicitly split:

* ``ClickTargetFrameContext`` / ``click_target_frame_context(frame)`` -- everything that
  depends only on the FRAME. Computed once per ``rank()`` call, content-cached. Measured
  0.9-4.8 ms per frame (worst case ``bp35``, 190 blobs).
* ``click_target_features(ctx, x, y)`` -- per CANDIDATE. Given the context, it does an
  ``O(1)`` numpy index into a precomputed ``(H, W)`` blob-index map, one 5x5 slice, a handful
  of arithmetic ops, and at most 8 anchor distances. Measured 14-20 us per candidate.

``blob_topology()`` is DELIBERATELY NOT USED, at either granularity. It measures 322 ms on a
single ``bp35`` frame -- roughly 100x everything else in the context combined, or ~161 s of
pure overhead across a 500-action episode. A test asserts the name does not appear in this
module so the measurement is encoded as a contract rather than a comment.

RELATION TO PRIOR ART (Failed-Experiment Rerun Discipline)
----------------------------------------------------------
``arc_inert_click_pruner.InertClickSigPruner`` already fits an online, within-episode,
coordinate-aware click model from the agent's own clicks (default off, wired into
``StepwiseExplorer`` and ``E3AgentPolicy``). The forward difference: that is a BINARY VETO
keyed on an EXACT 4-tuple signature ``(color, size, is_rect, twin_count)`` -- an unseen
signature has no counts, so it abstains, and it never ORDERS the survivors. This module is a
CONTINUOUS, 21-feature description that generalizes across similar-but-not-identical blobs
and yields a total order. Complementary, not a rerun. ``arc_hazard_pruner.HazardMovePruner``
is the same shape for deaths rather than clicks.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np

from carnot.agentic.arc_color_blob_salience import (
    ColorBlob,
    ColorBlobSaliencePrior,
    connected_color_blobs,
    object_hash,
)


# The feature vector's ORDER IS A CONTRACT: an online head fitted on index i must keep
# scoring index i. Tests assert the length and the names together so a reordering that
# silently invalidates a fitted head cannot land quietly.
CLICK_TARGET_FEATURE_NAMES: tuple[str, ...] = (
    "on_blob",  # 0  did the click land inside a real component at all
    "tier_norm",  # 1  ColorBlobSaliencePrior.tier / 4 (lower tier == higher priority)
    "button_likelihood_norm",  # 2  morphology score / 1.5
    "is_button_like",  # 3
    "is_large_flat",  # 4
    "is_status_bar_like",  # 5
    "pixel_count_norm",  # 6  min(1, pixels / 64)
    "area_fraction",  # 7  blob pixels / frame pixels
    "aspect_norm",  # 8  min(1, max(w/h, h/w) / 8)
    "color_rarity",  # 9  1 / (1 + count of this colour in the frame)
    "twin_count_norm",  # 10 min(1, identical-object multiplicity / 8)
    "local_distinct_colors",  # 11 distinct colours in the 5x5 neighbourhood / 8
    "local_dominant_fraction",  # 12 dominant-colour share of the 5x5 neighbourhood
    "local_heterogeneity",  # 13 1 - local_dominant_fraction
    "edge_distance_norm",  # 14 distance to the nearest frame edge, normalized
    "rare_anchor_min_distance",  # 15 L1 distance to the nearest rare-colour anchor / diagonal
    "rare_anchor_mean_distance",  # 16 mean L1 distance to rare-colour anchors / diagonal
    "target_novelty",  # 17 min(1, times this exact (x, y) clicked this episode / 3)
    "object_novelty",  # 18 min(1, times this object identity clicked this episode / 3)
    "x_norm",  # 19 x / width
    "y_norm",  # 20 y / height
)
CLICK_TARGET_FEATURE_DIM = len(CLICK_TARGET_FEATURE_NAMES)

# Number of rarest-colour blobs used as spatial anchors. Bounded so feature 15/16 stay O(1)
# per candidate rather than O(blobs).
MAX_RARE_ANCHORS = 8

# Half-width of the local neighbourhood window (5x5 total for radius 2).
LOCAL_WINDOW_RADIUS = 2

_DEFAULT_PRIOR = ColorBlobSaliencePrior()

# Bounded per-frame context cache. Keyed on grid CONTENT (shape + bytes), not object identity:
# candidate-generation code re-wraps the same underlying grid in a new frame object per call,
# so identity keying would miss every time. Same rationale and same bound as
# ``arc_color_blob_salience._blob_cache`` -- within one ranking pass all candidates share the
# same current frame, so only a handful of distinct frames are ever live at once.
_CONTEXT_CACHE_MAX_SIZE = 8
_context_cache: "OrderedDict[tuple[Any, ...], ClickTargetFrameContext]" = OrderedDict()


class ClickTargetFrameContext(NamedTuple):
    """Everything a click featurizer needs that depends only on the FRAME.

    Built once per ranking pass. Every field is either a small numpy array or a plain
    Python container, so per-candidate featurization is pure indexing and arithmetic.
    """

    grid: np.ndarray  # (H, W) int16 settled grid
    height: int
    width: int
    blobs: tuple[ColorBlob, ...]
    blob_index: np.ndarray  # (H, W) int32; blob list index, or -1 for "no blob here"
    blob_table: np.ndarray  # (n_blobs, 9) float; precomputed per-blob morphology, see below
    object_hashes: tuple[str, ...]  # per blob, translation-invariant identity
    twin_counts: Mapping[str, int]  # object_hash -> multiplicity within this frame
    color_counts: Mapping[int, int]  # colour value -> pixel count in this frame
    rare_anchors: np.ndarray  # (k, 2) float centroids (y, x) of the rarest-colour blobs
    diag_norm: float  # hypot(H, W), used to normalize L1 distances


# Column layout of ClickTargetFrameContext.blob_table. Precomputing these per BLOB (there are
# tens of blobs) rather than per CANDIDATE (there can be ~50, many landing on the same blob)
# is what keeps the per-candidate cost at ~15 us.
_BT_TIER = 0
_BT_BUTTON_LIKELIHOOD = 1
_BT_IS_BUTTON_LIKE = 2
_BT_IS_LARGE_FLAT = 3
_BT_IS_STATUS_BAR = 4
_BT_PIXEL_COUNT = 5
_BT_AREA_FRACTION = 6
_BT_ASPECT = 7
_BT_COLOR = 8
_BLOB_TABLE_WIDTH = 9


def _as_grid(frame: Any) -> np.ndarray:
    """Extract the settled 2-D grid from a frame object, an animation stack, or a raw array.

    Re-derived locally rather than importing ``arc_color_blob_salience._as_grid`` because that
    is a private helper; duplicating five lines is cheaper than coupling to a private name.
    ARC frames arrive as a list of animation layers whose LAST layer is the settled grid.
    """

    arr = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D ARC grid, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def settled_grid_of(frame: Any) -> np.ndarray:
    """Public alias for the settled-grid extraction, for callers outside this module."""

    return _as_grid(frame)


def _blob_table(blobs: Sequence[ColorBlob], prior: ColorBlobSaliencePrior) -> np.ndarray:
    table = np.zeros((len(blobs), _BLOB_TABLE_WIDTH), dtype=np.float64)
    for index, blob in enumerate(blobs):
        aspect = max(
            float(blob.width) / max(1.0, float(blob.height)),
            float(blob.height) / max(1.0, float(blob.width)),
        )
        table[index, _BT_TIER] = float(prior.tier(blob))
        table[index, _BT_BUTTON_LIKELIHOOD] = float(prior.button_likelihood(blob))
        table[index, _BT_IS_BUTTON_LIKE] = 1.0 if prior.is_button_like_blob(blob) else 0.0
        table[index, _BT_IS_LARGE_FLAT] = 1.0 if prior.is_large_flat_blob(blob) else 0.0
        table[index, _BT_IS_STATUS_BAR] = 1.0 if prior.is_status_bar_like(blob) else 0.0
        table[index, _BT_PIXEL_COUNT] = float(blob.pixel_count)
        table[index, _BT_AREA_FRACTION] = float(blob.area_fraction)
        table[index, _BT_ASPECT] = aspect
        table[index, _BT_COLOR] = float(blob.color)
    return table


def _nearest_blob_index(context: "ClickTargetFrameContext", x: int, y: int) -> int:
    """Index of the nearest-centroid blob, or -1 when the frame has no blobs.

    Same semantics as ``arc_color_blob_salience.blob_at_click``'s fallback branch, but it
    returns the INDEX so callers can look up the precomputed ``blob_table`` row. Resolving by
    index also avoids ``list.index(blob)``, which would compare ``ColorBlob.cells``
    frozensets -- a needlessly expensive equality for a positional lookup.
    """

    if not context.blobs:
        return -1
    best_index = -1
    best_distance = float("inf")
    for index, blob in enumerate(context.blobs):
        centroid_y, centroid_x = blob.centroid
        distance = (centroid_y - float(y)) ** 2 + (centroid_x - float(x)) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _rare_anchors(
    blobs: Sequence[ColorBlob], color_counts: Mapping[int, int], limit: int
) -> np.ndarray:
    """Centroids of the blobs whose colour is rarest in the frame.

    WHY: a click's meaning is partly relational -- "the piece next to the odd-coloured
    marker". Rare colours are the cheapest available proxy for a designed landmark (the same
    intuition ``ColorBlobSaliencePrior.salient_colors`` encodes, but as a spatial reference
    rather than a per-blob flag). Bounded to ``limit`` so the per-candidate distance loop
    stays O(1).
    """

    if not blobs:
        return np.zeros((0, 2), dtype=np.float64)
    ordered = sorted(
        range(len(blobs)),
        key=lambda i: (int(color_counts.get(int(blobs[i].color), 0)), blobs[i].bbox),
    )
    picked = ordered[: max(0, int(limit))]
    return np.array([blobs[i].centroid for i in picked], dtype=np.float64)


def click_target_frame_context(
    frame: Any,
    *,
    prior: ColorBlobSaliencePrior | None = None,
    use_cache: bool = True,
) -> ClickTargetFrameContext:
    """Build (or reuse) the per-FRAME half of the click featurization.

    Measured 0.9-4.8 ms per frame. Call this ONCE per ranking pass and hand the result to
    every ``click_target_features`` call -- never once per candidate.
    """

    grid = _as_grid(frame)
    active_prior = prior if prior is not None else _DEFAULT_PRIOR
    key: tuple[Any, ...] | None = None
    if use_cache:
        key = (grid.shape, grid.tobytes(), id(active_prior))
        cached = _context_cache.get(key)
        if cached is not None:
            _context_cache.move_to_end(key)
            return cached

    height, width = int(grid.shape[0]), int(grid.shape[1])
    blobs = tuple(
        connected_color_blobs(
            grid,
            min_pixels=active_prior.min_pixels,
            max_component_fraction=active_prior.max_component_fraction,
        )
    )

    # One pass over blob cells builds the (H, W) -> blob-index lookup. This replaces a linear
    # ``blob_at_click`` scan per candidate (measured 1.7-9.2 us) with a single numpy index
    # (0.34-0.46 us) at a one-off build cost of 0.13-0.35 ms per frame.
    blob_index = np.full((height, width), -1, dtype=np.int32)
    for index, blob in enumerate(blobs):
        for cell_y, cell_x in blob.cells:
            blob_index[int(cell_y), int(cell_x)] = index

    object_hashes = tuple(object_hash(blob) for blob in blobs)
    twin_counts = Counter(object_hashes)
    color_counts = Counter(int(value) for value in grid.flatten().tolist())
    rare_anchors = _rare_anchors(blobs, color_counts, MAX_RARE_ANCHORS)

    context = ClickTargetFrameContext(
        grid=grid,
        height=height,
        width=width,
        blobs=blobs,
        blob_index=blob_index,
        blob_table=_blob_table(blobs, active_prior),
        object_hashes=object_hashes,
        twin_counts=twin_counts,
        color_counts=color_counts,
        rare_anchors=rare_anchors,
        diag_norm=float(np.hypot(height, width)) or 1.0,
    )
    if use_cache and key is not None:
        _context_cache[key] = context
        if len(_context_cache) > _CONTEXT_CACHE_MAX_SIZE:
            _context_cache.popitem(last=False)
    return context


def clear_click_target_frame_context_cache() -> None:
    """Drop the per-frame context cache (used by tests that count segmentation calls)."""

    _context_cache.clear()


class ClickEpisodeState:
    """In-episode novelty counters. Created per episode, discarded at episode end.

    WHY this is a separate object rather than module state: a router instance can be reused
    across games (``scripts/arc_leaderboard_eval.py`` module-caches ONE router for a whole
    sweep). Counters stored on such a shared instance without episode keying would leak
    across games -- de-facto cross-game transfer, the direction retired by
    ``ops/exclusion_manifest.yaml`` id ``cross_game_value_transfer_retired_exp4342_v401``.
    Keying this object on the frame's own ``(game_id, guid)`` makes the isolation structural.
    """

    __slots__ = ("click_counts", "hash_counts")

    def __init__(self) -> None:
        self.click_counts: Counter[tuple[int, int]] = Counter()
        self.hash_counts: Counter[str] = Counter()

    def observe_click(self, x: int, y: int, object_identity: str | None = None) -> None:
        self.click_counts[(int(x), int(y))] += 1
        if object_identity:
            self.hash_counts[object_identity] += 1


def click_coordinates(candidate: Any) -> tuple[int, int] | None:
    """Return ``(x, y)`` for a click candidate, or ``None`` for anything else.

    Accepts both the ``ArcAction``-like object shape (``.action_id`` / ``.data``) and the
    Mapping row shape (``{"action" | "action_id", "data"}``) -- mirroring
    ``ColorBlobSaliencePrior._candidate_action_id`` / ``._candidate_data``, because both
    shapes really do reach the ranking path.

    Returning ``None`` (rather than a zero vector) matters: a keyboard action has no
    coordinate signal at all, and a zero vector would be a *scoreable* input that an online
    head could learn to prefer or avoid on no evidence. The router treats ``None`` as
    "contribute exactly 0.0".
    """

    if isinstance(candidate, Mapping):
        action_id = candidate.get("action", candidate.get("action_id", 0))
        data = candidate.get("data")
    else:
        action_id = getattr(candidate, "action_id", getattr(candidate, "action", 0))
        data = getattr(candidate, "data", None)
    try:
        if int(action_id or 0) != 6:
            return None
    except (TypeError, ValueError):
        return None
    if not isinstance(data, Mapping):
        return None
    if "x" not in data or "y" not in data:
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (TypeError, ValueError):
        return None


def click_target_features(
    context: ClickTargetFrameContext,
    x: int,
    y: int,
    *,
    episode_state: ClickEpisodeState | None = None,
) -> list[float]:
    """Describe the click target at ``(x, y)`` as a fixed-length, finite feature vector.

    Length is exactly ``CLICK_TARGET_FEATURE_DIM``; every entry is finite and roughly in
    ``[0, 1]`` so an online logistic head sees comparably-scaled inputs without needing a
    well-conditioned standardization from a handful of in-episode samples.

    Measured 14-20 us per candidate given a prebuilt context.
    """

    x = int(x)
    y = int(y)
    height, width = context.height, context.width
    out = [0.0] * CLICK_TARGET_FEATURE_DIM

    in_bounds = 0 <= y < height and 0 <= x < width
    blob_i = int(context.blob_index[y, x]) if in_bounds else -1

    if blob_i >= 0:
        out[0] = 1.0
        row = context.blob_table[blob_i]
        identity: str | None = context.object_hashes[blob_i]
    else:
        # Rare fallback (measured 0/48 on bp35, 0/47 tn36, 0/37 lp85, 2/7 sp80): the click
        # landed on a shared-colour gap between components. ``blob_at_click`` resolves it to
        # the nearest-centroid blob -- O(blobs), but paid only on the miss.
        out[0] = 0.0
        blob_i = _nearest_blob_index(context, x, y)
        if blob_i < 0:
            row = None
            identity = None
        else:
            row = context.blob_table[blob_i]
            identity = context.object_hashes[blob_i]

    if row is not None:
        out[1] = float(row[_BT_TIER]) / 4.0
        out[2] = min(1.0, float(row[_BT_BUTTON_LIKELIHOOD]) / 1.5)
        out[3] = float(row[_BT_IS_BUTTON_LIKE])
        out[4] = float(row[_BT_IS_LARGE_FLAT])
        out[5] = float(row[_BT_IS_STATUS_BAR])
        out[6] = min(1.0, float(row[_BT_PIXEL_COUNT]) / 64.0)
        out[7] = min(1.0, float(row[_BT_AREA_FRACTION]))
        out[8] = min(1.0, float(row[_BT_ASPECT]) / 8.0)
        color = int(row[_BT_COLOR])
        out[9] = 1.0 / (1.0 + float(context.color_counts.get(color, 0)))
        out[10] = min(1.0, float(context.twin_counts.get(identity or "", 0)) / 8.0)

    # Local texture: a 5x5 numpy slice, clipped at the frame border. Cheap, and it separates
    # "click in the middle of a uniform field" from "click on a boundary between objects" --
    # a distinction no per-blob feature carries.
    if in_bounds:
        r = LOCAL_WINDOW_RADIUS
        window = context.grid[
            max(0, y - r) : min(height, y + r + 1), max(0, x - r) : min(width, x + r + 1)
        ]
        values, counts = np.unique(window, return_counts=True)
        total = float(counts.sum()) or 1.0
        out[11] = min(1.0, float(len(values)) / 8.0)
        out[12] = float(counts.max()) / total
        out[13] = 1.0 - out[12]

        half_min_dim = max(1.0, float(min(height, width)) / 2.0)
        out[14] = min(1.0, float(min(x, y, width - 1 - x, height - 1 - y)) / half_min_dim)

    anchors = context.rare_anchors
    if anchors.shape[0] > 0:
        # L1 rather than Euclidean: grid worlds move in 4-connected steps, so L1 is the
        # distance the agent's own actions actually traverse.
        distances = np.abs(anchors[:, 0] - float(y)) + np.abs(anchors[:, 1] - float(x))
        out[15] = min(1.0, float(distances.min()) / context.diag_norm)
        out[16] = min(1.0, float(distances.mean()) / context.diag_norm)

    if episode_state is not None:
        out[17] = min(1.0, float(episode_state.click_counts.get((x, y), 0)) / 3.0)
        if identity:
            out[18] = min(1.0, float(episode_state.hash_counts.get(identity, 0)) / 3.0)

    out[19] = float(x) / float(max(1, width))
    out[20] = float(y) / float(max(1, height))
    return out


def click_target_object_identity(context: ClickTargetFrameContext, x: int, y: int) -> str | None:
    """Object-identity hash of whatever the click at ``(x, y)`` lands on, or ``None``."""

    if not (0 <= int(y) < context.height and 0 <= int(x) < context.width):
        return None
    blob_i = int(context.blob_index[int(y), int(x)])
    if blob_i < 0:
        blob_i = _nearest_blob_index(context, int(x), int(y))
    if blob_i < 0:
        return None
    return context.object_hashes[blob_i]


class OnlineClickTargetDiscriminator:
    """A tiny logistic head fitted ONLINE, within one episode, from the agent's own outcomes.

    Design constraints, each measured or cited rather than assumed:

    * **Cold start MUST be a no-op.** ``proba`` returns exactly ``0.5`` until the sample gate
      is met, so a router blending ``weight * (proba - 0.5)`` adds exactly ``0.0`` and its
      output is bit-identical to the unmodified router. The gate mirrors the live agent's own
      ``discriminative_min_positives`` / ``min_negatives`` of 3 and
      ``InertClickSigPruner.min_observations`` of 4.
    * **Variance FLOOR, not ``std + 1e-8``.** ``DiscriminativeVerifier.fit`` standardizes with
      ``sd = std + 1e-8``. Measured failure with that in a small in-episode sample: a column of
      19 zeros and one ``1e-6`` gives ``sd = 2.28e-7`` and weight ``1.473``; a later value of
      ``0.75`` then produces ``z = 3.29e6`` and ``proba`` EXACTLY ``1.0`` -- which collapses
      the ranking back into a tie AND emits ``IMPLAUSIBLE_PERFECT``-shaped values. A floor of
      ``1e-2`` plus a clip on the standardized value is the entire fix. (Exactly-constant
      columns are already safe: their weight stays exactly 0.)
    * **numpy, not sklearn.** sklearn is installed and fast enough, but this project
      deliberately avoids it in shipped paths (see ``models/hallusal_sparse_ae.py`` "Why not
      sklearn") and the Kaggle image is size-constrained. A ~40-line numpy head also matches
      the shape of the incumbent ``DiscriminativeVerifier`` that reviewers already know.
    * **Bounded memory.** Observations are capped, so a long episode cannot grow without
      limit; the cap keeps the whole head at a sub-2 ms refit (measured 0.91-1.57 ms at
      n <= 120).

    THE LABEL. The caller supplies it, and it MUST be causally downstream of the click --
    typically "did this executed click change the frame / produce a level-up", read from the
    agent's OWN observed ``(before, action, after)`` transition. Do NOT use the human-replay
    corpus's ``level_progress(row, step_index)``: it is a pure function of the step index
    (``arc_human_replay_corpus.py:134-145``), which is exactly how exp5835's zero-perception
    step-index predictor scored 0.9234 while its "perception" arms scored 0.66-0.69.
    """

    def __init__(
        self,
        *,
        dim: int = CLICK_TARGET_FEATURE_DIM,
        min_positives: int = 3,
        min_negatives: int = 3,
        min_total: int = 8,
        max_samples: int = 256,
        refit_every: int = 4,
        iters: int = 200,
        lr: float = 0.5,
        l2: float = 1e-2,
        sd_floor: float = 1e-2,
        z_clip: float = 8.0,
    ) -> None:
        self.dim = int(dim)
        self.min_positives = int(min_positives)
        self.min_negatives = int(min_negatives)
        self.min_total = int(min_total)
        self.max_samples = int(max_samples)
        self.refit_every = max(1, int(refit_every))
        self.iters = int(iters)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.sd_floor = float(sd_floor)
        self.z_clip = float(z_clip)

        self._x: list[list[float]] = []
        self._y: list[float] = []
        self._weights: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._pending = 0
        self.n_fits = 0
        self.saturation_clips = 0
        self.n_level_up_labels = 0

    # ---------------------------------------------------------------- observation

    @property
    def n_positives(self) -> int:
        return int(sum(1 for label in self._y if label >= 0.5))

    @property
    def n_negatives(self) -> int:
        return int(sum(1 for label in self._y if label < 0.5))

    @property
    def fitted(self) -> bool:
        return self._weights is not None

    @property
    def gate_met(self) -> bool:
        return (
            self.n_positives >= self.min_positives
            and self.n_negatives >= self.min_negatives
            and len(self._y) >= self.min_total
        )

    def observe(self, features: Sequence[float], label: float, *, leveled_up: bool = False) -> None:
        vector = [float(value) for value in features]
        if len(vector) != self.dim:
            raise ValueError(
                f"expected {self.dim} features, got {len(vector)}"
            )  # pragma: no cover - guarded by tests
        self._x.append(vector)
        self._y.append(1.0 if float(label) >= 0.5 else 0.0)
        if leveled_up:
            self.n_level_up_labels += 1
        if len(self._x) > self.max_samples:
            self._x.pop(0)
            self._y.pop(0)
        self._pending += 1

    def maybe_fit(self) -> bool:
        """Refit if the gate is met and enough new observations have accumulated."""

        if not self.gate_met:
            return False
        if self._weights is not None and self._pending < self.refit_every:
            return False
        return self.fit()

    def fit(self) -> bool:
        if not self.gate_met:
            return False
        X = np.asarray(self._x, dtype=np.float64)
        y = np.asarray(self._y, dtype=np.float64)
        mu = X.mean(axis=0)
        sd = np.maximum(X.std(axis=0), self.sd_floor)
        Z = np.clip((X - mu) / sd, -self.z_clip, self.z_clip)
        design = np.hstack([Z, np.ones((Z.shape[0], 1))])
        w = np.zeros(design.shape[1], dtype=np.float64)
        n = float(design.shape[0])
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -30.0, 30.0)))
            grad = design.T @ (p - y) / n
            grad[:-1] += self.l2 * w[:-1]  # no L2 on the bias
            w -= self.lr * grad
        self._weights = w
        self._mu = mu
        self._sd = sd
        self._pending = 0
        self.n_fits += 1
        return True

    # ---------------------------------------------------------------- scoring

    def proba(self, features: Sequence[float]) -> float:
        """P(label=1) for this click target, or exactly 0.5 when not yet trustworthy."""

        if self._weights is None or self._mu is None or self._sd is None:
            return 0.5
        vector = np.asarray([float(value) for value in features], dtype=np.float64)
        if vector.shape[0] != self.dim:
            return 0.5
        z = (vector - self._mu) / self._sd
        clipped = np.clip(z, -self.z_clip, self.z_clip)
        if not np.array_equal(z, clipped):
            self.saturation_clips += 1
        logit = float(np.dot(np.append(clipped, 1.0), self._weights))
        return float(1.0 / (1.0 + np.exp(-max(-30.0, min(30.0, logit)))))

    def stats(self) -> dict[str, Any]:
        return {
            "n_samples": len(self._y),
            "n_positives": self.n_positives,
            "n_negatives": self.n_negatives,
            "n_fits": self.n_fits,
            "fitted": self.fitted,
            "gate_met": self.gate_met,
            "saturation_clips": self.saturation_clips,
            "n_level_up_labels": self.n_level_up_labels,
            "min_positives": self.min_positives,
            "min_negatives": self.min_negatives,
            "min_total": self.min_total,
        }
