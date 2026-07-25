"""Orientation-complete HUD status-bar detection + a hard collapse guard on its application.

Spec refs: REQ-ARC-WMTE-5960, SCENARIO-ARC-WMTE-5960.

=====================================================================================
WHY THIS MODULE EXISTS -- the measured defect it repairs
=====================================================================================
``StepwiseExplorer`` collapses HUD cells out of NODE IDENTITY (see its ``_hash``): a game
that renders a ticking score/timer/step counter into the frame makes every single action
look like it reached a brand-new state, so the search's dedup never fires and it has no
memory of where it has already been.

On the public game ``r11l`` that failure was measured end to end (offline dev twin,
``development_proxy`` provenance -- see ``ops/known-issues.md`` 2026-07-25):

  * r11l renders a MONOTONE step counter into frame COLUMN 0, so every action mutates
    the frame.
  * ``SUBMITTED_AUTO_HUD_MASK_ENABLED`` is already True and the mask IS attempted, but the
    shipped classifier resolves to **None** on r11l -- nothing is excluded from identity.
  * Consequence: 1956 actions produced 1392 graph nodes over only 31 true game states =
    **44.9x node inflation** (that ratio is an ARM-A number; on the current live config,
    arm B2, the same game inflates ~22-23x -- cite the per-arm baseline, never the headline).
  * The livelock it creates: salience rank 0 is a WALL-blocked, game-state-INERT click that
    changes exactly one cell -- in column 0. Each such action therefore mints a fresh node
    and is re-popped forever: 1371 of 1956 actions (70%) on arm A.
  * CAUSAL PROOF: masking column 0 out of node identity and changing NOTHING else makes
    r11l WIN on 3 of 3 seeds (arm A 0->1 level in 1013 actions; arm B2 0->1 in 222-232),
    inflation 44.9x -> ~1.0-1.5x.

That proof used a hand-built column-0 mask derived from the public game's SOURCE. That is a
DIAGNOSTIC ONLY and must never ship. This module is the hidden-game-legal replacement: it
derives the same mask from FRAME STATISTICS ALONE -- no per-game constants, no source
reading, no oracle.

=====================================================================================
WHY THE SHIPPED CLASSIFIER MISSES IT (the exact predicate, measured)
=====================================================================================
``ColorBlobSaliencePrior.is_status_bar_like`` requires, for a non-status-COLOURED blob::

    touches_frame_edge = (bbox.y0 == 0) or (bbox.y1 == frame_height - 1)
    spans_status_width = blob.width >= 0.75 * frame_width
    thin               = blob.height <= 2

r11l's counter is one 4-connected blob: colour 0, 64 px, bbox (y0=0, x0=0, y1=63, x1=0),
width 1, height 64. Instrumented on the real reset frame: ``touches_frame_edge`` True (only
incidentally -- the bar happens to span the full height), ``spans_status_width`` False,
``thin`` False. So it never fires, and the mask is None.

Two INDEPENDENT gaps, both structural rather than mis-thresholded:

1. ``width >= 0.75*W AND height <= 2`` is a HORIZONTAL-bar template. By definition no
   vertical bar can satisfy that conjunction at any frame size -- the detector is
   orientation-blind, not merely mis-tuned.
2. The edge test reads ``bbox[0]``/``bbox[2]``, which are both Y coordinates, so only the
   TOP and BOTTOM rows are ever tested. A vertical bar in column 63 spanning rows 10..50
   would fail the edge test too.

This module's ``is_edge_bar_like`` fixes both: all FOUR edges with a symmetric tolerance,
and a SCALE-FREE orientation-aware elongation ratio instead of an absolute width floor plus
an absolute thickness cap.

=====================================================================================
THE OVER-MASKING TRAP -- why this is detection-only and deliberately conservative
=====================================================================================
The 3rd-place hidden-leaderboard reference solver (arXiv:2512.24156, "just-explore") DOES
detect r11l's bar; its ``FrameProcessor.identify_status_bars``, run on the IDENTICAL r11l
reset frame, returns a 64-cell column-0 mask byte-for-byte equal to the oracle diagnostic.
So detection is provably possible legally. But that same reference is also a worked example
of how the APPLICATION goes wrong: on lf52/tu93/su15 its masked-hash graph shows up to 88x
hash COLLAPSE, its graph records contradictory successors, it trips its own assert, and it
livelocks (72-97% of its ``choose_action`` calls raise).

Two things were measured about that, and both shape this module:

* The reference's over-collapse is NOT detection sensitivity. On lf52/tu93/su15 our
  detector and the reference's produce BYTE-IDENTICAL masks at the reset frame. The damage
  lives in its application: it overwrites masked cells with colour 16 BEFORE segmentation
  (so the mask also rewrites the candidate set), and it RECOMPUTES the mask on every
  level-up from whatever mid-game frame is current -- measured on r11l as a mid-run mask
  change from 64 to 93 cells, silently re-keying every node that already existed.
  Carnot's application is the conservative one (mask for IDENTITY only, computed ONCE from
  the first observed frame, never refreshed) and this module does not touch it.
* Real over-masking nevertheless EXISTS in our own pipeline. Injecting the reference's mask
  into our conservative application produced proven aliasing violations on tu93 (2 of 58
  observable keys; unmasked control 0 of 14) and lf52 (1/9, 0/8, 1/40). Every violation
  inspected was ONE class: two antecedents differing ONLY inside the mask, one successor a
  normal frame and one TERMINAL -- i.e. a monotone clock that GATES the game-over
  transition. That is a decision-relevant state variable hiding inside a textbook HUD.

THE ASYMMETRY THIS MODULE IS BUILT AROUND: under-masking costs search efficiency;
over-masking destroys correctness. A mask that collapses genuinely distinct states is WORSE
than no mask at all. Hence: take the reference's sensitivity, keep our conservative
application, and add a runtime guard that can HARD-REFUSE a mask that is proven to alias.

=====================================================================================
WHAT IS IN HERE -- three stages, each independently unit-testable and pure where possible
=====================================================================================
STAGE 1 -- ``edge_bar_hud_mask(frame)``: single-frame geometry. Zero action cost, available
  at first contact, and it is what actually finds r11l. Verified on all 25 public games to
  be a strict SUPERSET of today's mask (drops nothing on 25/25) and a strict SUBSET of the
  reference's mask (0 extra px on 25/25; 72 px fewer, 100% of which is the reference's
  "twins"/dotted-bar rule masking scattered single pixels that are candidate CLICK targets
  -- deliberately NOT ported).

STAGE 2 -- ``region_hud_evidence(grids, mask, ...)``: multi-frame behavioural confirmation
  from transitions the agent ALREADY takes (zero extra actions, unlike
  ``arc_graph_explore.discover_hud_mask`` which burns up to 4 real actions from reset and is
  not viable under RHAE live scoring). Geometry is a SHAPE prior and provably cannot
  distinguish a live counter from a static decorative border; this stage is what does.
  A MEASUREMENT TRAP recorded here so it is not repeated: the per-CELL statistic "this cell
  changed on every action" returns 0 on every game measured, INCLUDING r11l, because a
  monotone FILLING bar changes only ONE cell per action. The correct statistic is
  REGION-level -- "the region ticks even when the rest of the board does not".

STAGE 3 -- ``MaskCollapseGuard``: the runtime hard refusal. A ``(masked_node,
  concrete_action)`` key that is observed to produce TWO DIFFERENT masked successors proves
  one masked hash is covering two behaviourally distinct true states. Every such proof is
  counted and the offending node is un-masked (locally split); past a small cap the mask is
  globally revoked and identity falls back to unmasked. It carries a MANDATORY unmasked
  CONTROL: if the same key also produces two different UNMASKED successors, the environment
  is simply non-deterministic at that node and the violation is NOT attributable to the mask
  (measured: sc25 shows 2-3 masked violations AND 2-3 unmasked-control violations, so
  nothing there is the mask's fault).

=====================================================================================
CO-CHANGE AUDIT -- everything that could have interacted, checked one by one
=====================================================================================
Changing NODE IDENTITY touches more than dedup, and a previous experiment in this area
nearly credited a fix to the wrong mechanism. So each component that plausibly interacts was
audited rather than assumed:

* ``ColorBlobSaliencePrior.is_status_bar_like`` -- DELIBERATELY NOT TOUCHED. It feeds
  ``tier()``, ``is_button_like_blob()``, ``button_likelihood()`` (candidate RANKING order) and
  is feature 5 of the already-TRAINED click-target CNN feature table
  (``arc_click_target_features``). Widening it in place would have changed candidate ordering
  AND shifted a trained model's input distribution simultaneously with node identity. That is
  the whole reason ``is_edge_bar_like`` is a NEW predicate consumed only by the mask path, and
  it is why the ranking arms of an A/B against this change are byte-identical.
* The FRONTIER TIER BARRIER (``arc_frontier_discipline.click_tier_map``) -- no interaction.
  Verified by reading it: it uses its own ``colour == 16`` status-bar test and never calls
  ``is_status_bar_like`` at all. So the three frontier flags flipped live on 2026-07-25 are
  untouched by this change.
* The FRAME-CHANGE SCORER and the ACTION PRIOR -- no interaction. Their ``observe_transition``
  hooks take raw before/after frames plus the action; neither ever sees a node hash.
* The INERT-CLICK PRUNER -- no interaction (same raw-frame observe signature). Worth naming
  explicitly because it is the SECOND, INDEPENDENT lever on the same r11l pathology and it
  must stay OFF while this one is measured, or the two become inseparable.
* The DISCRIMINATIVE ROUTER -- no interaction; it featurises frames and candidates, with no
  node-hash-keyed state.
* SIMILARITY RETRIEVAL (``_index_similarity_state``) -- DOES key on the node hash, so a split
  node gains one extra bucket entry. Additive and harmless (it is a retrieval index, not an
  identity), and its own flag is off by default. Documented rather than changed.
* The A* VALUE CACHE (``_value_cache[node_hash]``) -- keys on the node hash, so a split node
  gets a fresh entry. That is CORRECT: post-split it genuinely is a different node.
* The CLICK-PIXEL REDRAW BUDGET -- stored on the node dict itself, not in a hash-keyed map, so
  a split node starts with a fresh budget. Benign.
* ADJACENCY (``self.adj[origin]``) -- edges recorded before a split stay under the old masked
  key while new edges land under the compound key, i.e. the graph briefly holds two identity
  conventions for that one node. This is the real, bounded cost of local retraction, taken
  deliberately (see ``MaskCollapseGuard``) instead of the rehash-everything approach that
  makes the reference solver re-initialise its graph hundreds of times per run.

=====================================================================================
HONEST LIMITS
=====================================================================================
* A Stage-3 violation is only OBSERVABLE for a key tried at least twice, and the count grows
  with budget (lf52: 1 of 9 observable keys at 394 actions -> 3 of 30 at 765). Zero
  violations is therefore never proof of zero aliasing. ``observable_keys`` is always
  reported as the honest denominator.
* Stage 2's thresholds were calibrated on 9 public games x 5 seeds. They separate that
  corpus cleanly but could be over-fit to it; they should be re-checked, not hand-tuned,
  when a full-corpus run happens.
* Stage 2's ubiquity conjunct is FOOLED by exactly the terminal-gating counter described
  above (it admits it, and it does alias) -- which is precisely why Stage 3 is mandatory
  whenever a mask is applied, rather than optional.
* Stage 2's non-reciprocity conjunct has a BLIND SPOT created by its own restart fix: a region
  whose values CYCLE back through the segment's first value on every lap is read as a series of
  restarts, not as revisits, so a periodic region can pass. Refusing to treat that as a restart
  is worse (it false-refuses the mask that provably wins r11l), so the blind spot is accepted
  and Stage 3 is again the backstop. Found while writing this module's own tests, not assumed.
* Stage 1 masks a background STRIP that an inset bar leaves behind: a bar one cell in from an
  edge splits the background, and the leftover 1-cell strip is itself edge-adjacent and
  elongated, so both are masked and the union doubles. On a small frame that can trip the area
  ceiling and refuse an otherwise-fine mask. Conservative in the right direction (it refuses
  rather than over-masks) but it does mean the mask can be larger than the visible bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1 -- single-frame geometry
# ---------------------------------------------------------------------------

# Symmetric distance from a frame edge, in cells, within which a blob counts as
# edge-adjacent. WHY 2 AND NOT 3: the reference's own tolerance is ASYMMETRIC by an
# off-by-one -- with threshold 3 on a 64-wide frame it admits rows/cols 0..2 on the
# top/left (`max < 3`) but only 62..63 on the bottom/right (`min > 64-3 = 61`). A SYMMETRIC
# tolerance of 2 (0..1 and 62..63) is therefore contained in the reference's admission band
# on ALL FOUR edges, which is what makes our mask a provable subset of theirs. Tolerance 3
# breaks that containment and immediately misbehaves in practice: measured on ls20 it fires
# on a colour-11 blob at bbox (61,13)-(62,54) and masks 84 px where the reference -- using
# its own threshold of 3 -- masks NOTHING at all.
EDGE_BAR_EDGE_TOLERANCE = 2

# Scale-free elongation required in the orientation implied by the edge. Matches the
# reference's ratio threshold exactly, so an A/B difference against it cannot be a
# threshold difference. Replaces the shipped classifier's `width >= 0.75*W AND height <= 2`,
# which is a horizontal-bar TEMPLATE no vertical bar can ever satisfy.
EDGE_BAR_MIN_ELONGATION = 5.0

# Hard ceiling on the TOTAL fraction of the frame a candidate mask may claim. The largest
# real status bar measured across the 25 public games is 128 cells of a 64x64 frame = 3.1%
# (sc25), so 5% is loose enough to admit every real bar while structurally refusing a mask
# that would swallow the board. This is the blast-radius bound: over-masking destroys
# correctness, so the detector must be unable to propose an unbounded mask even if the
# geometry predicate is fooled.
EDGE_BAR_MAX_MASK_AREA_FRACTION = 0.05


@dataclass(frozen=True)
class EdgeBarThresholds:
    """The three Stage-1 knobs, grouped so a test can vary one without monkeypatching."""

    edge_tolerance: int = EDGE_BAR_EDGE_TOLERANCE
    min_elongation: float = EDGE_BAR_MIN_ELONGATION
    max_mask_area_fraction: float = EDGE_BAR_MAX_MASK_AREA_FRACTION


def _as_grid(frame: Any) -> Optional[np.ndarray]:
    """Coerce whatever the caller has (Frame object, list-of-grids, ndarray) to one 2-D grid."""

    if frame is None:
        return None
    arr = np.asarray(frame.frame if hasattr(frame, "frame") else frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def is_edge_bar_like(blob: Any, *, thresholds: EdgeBarThresholds | None = None) -> bool:
    """Return True for an edge-adjacent, elongated strip in EITHER orientation.

    This is the repaired predicate. It is deliberately a NEW predicate rather than a
    widening of ``ColorBlobSaliencePrior.is_status_bar_like``, because that method is also
    consumed by ``tier()`` / ``is_button_like_blob()`` / ``button_likelihood()`` (candidate
    RANKING order) and is feature 5 of the already-TRAINED click-target CNN feature table
    (``arc_click_target_features``). Widening it in place would change candidate ordering AND
    shift a trained model's input distribution at the same time as changing node identity --
    i.e. it would make the experiment un-attributable, the exact failure mode where a fix
    gets credited to the wrong mechanism.

    The predicate, on frame statistics only::

        y0, x0, y1, x1 = blob.bbox ; h, w = blob.frame_shape ; tol = thresholds.edge_tolerance
        on_top    = y1 < tol            on_bottom = y0 > h - 1 - tol
        on_left   = x1 < tol            on_right  = x0 > w - 1 - tol
        long_h    = width  / max(1, height) >= ratio
        long_v    = height / max(1, width)  >= ratio
        fire      = ((on_top or on_bottom) and long_h) or ((on_left or on_right) and long_v)

    The orientation is coupled to the edge on purpose: a long HORIZONTAL strip is only a
    status bar when it hugs the top or bottom, and a long VERTICAL strip only when it hugs
    the left or right. Decoupling them (accepting either orientation on any edge) is what
    the reference's ``direction == "any"`` corner case does, and it is strictly looser.
    """

    thr = thresholds or EdgeBarThresholds()
    shape = getattr(blob, "frame_shape", None)
    if shape is None:
        return False
    try:
        height, width = int(shape[0]), int(shape[1])
        y0, x0, y1, x1 = (int(v) for v in blob.bbox)
        blob_h, blob_w = int(blob.height), int(blob.width)
    except (TypeError, ValueError, IndexError):  # pragma: no cover - defensive
        return False
    if height <= 0 or width <= 0:
        return False

    tol = max(0, int(thr.edge_tolerance))
    on_top = y1 < tol
    on_bottom = y0 > height - 1 - tol
    on_left = x1 < tol
    on_right = x0 > width - 1 - tol

    ratio = float(thr.min_elongation)
    long_horizontal = (float(blob_w) / float(max(1, blob_h))) >= ratio
    long_vertical = (float(blob_h) / float(max(1, blob_w))) >= ratio

    return bool(
        ((on_top or on_bottom) and long_horizontal)
        or ((on_left or on_right) and long_vertical)
    )


def edge_bar_hud_mask(
    frame: Any,
    *,
    thresholds: EdgeBarThresholds | None = None,
    include_status_bar_like: bool = True,
) -> Optional[np.ndarray]:
    """Stage 1: propose a HUD mask from ONE frame. ``None`` means "found nothing" (a no-op).

    ``include_status_bar_like=True`` ORs in the currently-shipped
    ``ColorBlobSaliencePrior.is_status_bar_like`` predicate, which makes the returned mask a
    SUPERSET of today's mask BY CONSTRUCTION. That is what lets the A/B attribute any
    difference to the newly-detected cells rather than to cells that silently stopped being
    masked. Verified across all 25 public games: 0 games lose a cell they mask today.

    The total-area ceiling is applied to the UNION, not per blob, and a mask that exceeds it
    is refused ENTIRELY (returns None -> today's behaviour) rather than truncated, because a
    partially-applied over-broad mask is exactly the correctness hazard.
    """

    grid = _as_grid(frame)
    if grid is None:
        return None
    thr = thresholds or EdgeBarThresholds()
    try:
        from carnot.agentic.arc_color_blob_salience import (
            ColorBlobSaliencePrior,
            connected_color_blobs,
        )
    except Exception:  # pragma: no cover - defensive import guard
        return None

    prior = ColorBlobSaliencePrior() if include_status_bar_like else None
    mask = np.zeros(grid.shape, dtype=bool)
    fired = False
    for blob in connected_color_blobs(grid):
        hit = is_edge_bar_like(blob, thresholds=thr)
        if not hit and prior is not None:
            hit = bool(prior.is_status_bar_like(blob))
        if not hit:
            continue
        fired = True
        for y, x in blob.cells:
            mask[y, x] = True

    if not fired or not bool(mask.any()):
        return None
    area = int(grid.shape[0]) * int(grid.shape[1])
    if area <= 0:
        return None
    if float(mask.sum()) / float(area) > float(thr.max_mask_area_fraction):
        # Refuse wholesale. See the docstring: truncating would ship a partially-applied
        # over-broad mask, and over-masking destroys correctness.
        return None
    return mask


# ---------------------------------------------------------------------------
# Stage 2 -- multi-frame behavioural confirmation
# ---------------------------------------------------------------------------

# Below this many usable transitions the evidence ABSTAINS rather than deciding. Calibrated
# on 9 games x 5 seeds; an abstain must leave the caller on its previous behaviour.
REGION_EVIDENCE_MIN_TRANSITIONS = 16

# A real HUD clock ticks regardless of WHICH action was taken. Required per action class
# that was tried at least twice, not pooled: pooling lets a frequently-tried action class
# carry a class that never moves the region at all (measured: su15's row 63 responds only to
# clicks -- action a7 never changes it while a6 changes it 75-89% of the time, so its pooled
# rate looks respectable while its per-class minimum is 0.0).
REGION_EVIDENCE_MIN_UBIQUITY = 0.95

# A monotone counter is IRREVERSIBLE; genuine game state is revisitable. Any in-episode
# revisit of a prior region value therefore refuses the mask.
REGION_EVIDENCE_MAX_REVISITS = 0


def _region_signature(grid: np.ndarray, mask: np.ndarray) -> bytes:
    return grid[mask].tobytes()


def _complement_signature(grid: np.ndarray, mask: np.ndarray) -> bytes:
    return grid[~mask].tobytes()


def region_hud_evidence(
    grids: Sequence[Any],
    mask: Any,
    *,
    actions: Optional[Sequence[Any]] = None,
    min_transitions: int = REGION_EVIDENCE_MIN_TRANSITIONS,
    min_ubiquity: float = REGION_EVIDENCE_MIN_UBIQUITY,
    max_revisits: int = REGION_EVIDENCE_MAX_REVISITS,
) -> dict:
    """Stage 2: does ``mask``'s region behave like a HUD across observed transitions?

    ``grids`` is the observed frame sequence (anything ``_as_grid`` accepts; a None entry is
    read as an episode break, e.g. a terminal frame). ``actions`` is the parallel sequence of
    action labels for the transition INTO each frame -- ``actions[i]`` produced ``grids[i]``
    from ``grids[i-1]``. Any hashable label works; the harness passes ``(action_id, x, y)``.

    Returns a verdict dict. The three measured statistics:

    ``independent_tick_rate``
        Of the transitions where the COMPLEMENT of the region did not change at all, the
        fraction where the region changed anyway. This is the decisive HUD signature --
        "it ticks when the game state does not" is literally the node-inflation mechanism.
        Measured: r11l 18/18, lf52 94/94, tn36 57/57, tu93 30/30, bp35 9/9; versus
        su15 28/70, sc25 10/24, cn04 7/18, ar25 10/25.

    ``ubiquity``
        The MINIMUM, over action classes tried at least twice, of the fraction of that
        class's transitions in which the region changed. A HUD clock ticks whatever you do.
        Measured 1.0 for r11l/lf52/tu93/tn36/sp80 and 0.0-0.29 for su15/sc25/cn04.
        When ``actions`` is None this degenerates to the pooled rate and
        ``ubiquity_is_pooled`` is set, so a caller cannot mistake it for the per-class test.

    ``revisits``
        In-episode revisits of a prior region value. Segments break on a None/terminal frame
        AND on a return to the segment's FIRST region value -- that second rule is a real
        fix, not a nicety: without it, restart-induced revisits made r11l REFUSE on seed 0
        (13 revisits) while seeds 1-2 admitted, a false refusal of the mask that provably
        wins r11l. With it, r11l admits on 5/5 seeds and no refuser regresses.

    A verdict of ``"abstain"`` is NOT a pass. The caller must keep its previous behaviour.
    """

    mask_arr = np.asarray(mask, dtype=bool) if mask is not None else None
    out: dict[str, Any] = {
        "verdict": "abstain",
        "reason": "no_mask",
        "n_transitions": 0,
        "n_complement_static": 0,
        "n_complement_static_region_changed": 0,
        "independent_tick_rate": None,
        "ubiquity": None,
        "ubiquity_is_pooled": actions is None,
        "per_action_change_rate": {},
        "revisits": 0,
        "n_distinct_region_values": 0,
        "n_distinct_complement_values": 0,
        "min_transitions": int(min_transitions),
    }
    if mask_arr is None or not bool(mask_arr.any()):
        return out

    prepared: list[Optional[np.ndarray]] = []
    for item in grids or ():
        g = _as_grid(item)
        if g is None or g.shape != mask_arr.shape:
            prepared.append(None)
        else:
            prepared.append(g)

    labels: list[Any] = list(actions or ())

    region_values: set[bytes] = set()
    complement_values: set[bytes] = set()
    n_transitions = 0
    n_complement_static = 0
    n_complement_static_region_changed = 0
    per_action_total: dict[Any, int] = {}
    per_action_changed: dict[Any, int] = {}
    revisits = 0
    segment_seen: set[bytes] = set()
    segment_first: Optional[bytes] = None

    for index, grid in enumerate(prepared):
        if grid is None:
            # Episode break: a terminal / unusable frame ends the monotonicity segment.
            segment_seen = set()
            segment_first = None
            continue
        region = _region_signature(grid, mask_arr)
        complement = _complement_signature(grid, mask_arr)
        region_values.add(region)
        complement_values.add(complement)

        if segment_first is None:
            segment_first = region
            segment_seen = {region}
        elif region == segment_first:
            # A monotone counter that returned to its reset value is a NEW segment, not a
            # revisit -- this is the restart case that produced the r11l false refusal.
            segment_seen = {region}
        elif region in segment_seen:
            revisits += 1
        else:
            segment_seen.add(region)

        if index == 0 or prepared[index - 1] is None:
            continue
        previous = prepared[index - 1]
        n_transitions += 1
        region_changed = region != _region_signature(previous, mask_arr)
        complement_changed = complement != _complement_signature(previous, mask_arr)
        if not complement_changed:
            n_complement_static += 1
            if region_changed:
                n_complement_static_region_changed += 1
        label = labels[index] if index < len(labels) else None
        per_action_total[label] = per_action_total.get(label, 0) + 1
        if region_changed:
            per_action_changed[label] = per_action_changed.get(label, 0) + 1

    out["n_transitions"] = int(n_transitions)
    out["n_complement_static"] = int(n_complement_static)
    out["n_complement_static_region_changed"] = int(n_complement_static_region_changed)
    out["revisits"] = int(revisits)
    out["n_distinct_region_values"] = int(len(region_values))
    out["n_distinct_complement_values"] = int(len(complement_values))
    if n_complement_static:
        out["independent_tick_rate"] = round(
            n_complement_static_region_changed / n_complement_static, 4
        )

    rates = {
        label: (per_action_changed.get(label, 0) / total)
        for label, total in per_action_total.items()
        if total >= 2
    }
    out["per_action_change_rate"] = {
        (str(label) if label is not None else "unlabelled"): round(rate, 4)
        for label, rate in rates.items()
    }
    if rates:
        out["ubiquity"] = round(min(rates.values()), 4)

    if n_transitions < int(min_transitions):
        out["reason"] = "insufficient_transitions"
        return out
    if out["ubiquity"] is None:
        out["reason"] = "no_action_class_tried_twice"
        return out
    if float(out["ubiquity"]) < float(min_ubiquity):
        out["verdict"] = "refuse"
        out["reason"] = "region_not_action_ubiquitous"
        return out
    if int(revisits) > int(max_revisits):
        out["verdict"] = "refuse"
        out["reason"] = "region_value_revisited_in_episode"
        return out
    out["verdict"] = "admit"
    out["reason"] = "action_ubiquitous_and_monotone"
    return out


# ---------------------------------------------------------------------------
# Stage 3 -- the runtime collapse guard (HARD refusal)
# ---------------------------------------------------------------------------

# How many nodes may be individually un-masked before the mask is revoked outright. Small
# on purpose: a handful of proven aliases is a local defect worth splitting, but many is
# evidence the mask itself is wrong. NOT zero, because a strict zero-violation rule would
# throw away the mask that wins tu93 on 3/3 seeds over 2 bad nodes out of 58 -- the
# asymmetry cuts both ways once the guard exists to bound the damage.
HUD_MASK_GUARD_MAX_SPLIT_NODES = 3


@dataclass
class MaskCollapseGuard:
    """Proves, at runtime, whether an applied HUD mask is collapsing DISTINCT states.

    THE PROOF OBLIGATION. In a deterministic environment, one ``(state, action)`` pair has
    exactly one successor. So if a ``(masked_hash, concrete_action)`` key is observed to
    produce TWO DIFFERENT masked successors, the masked hash must be covering at least two
    behaviourally distinct true states -- a genuine collapse, not a dedup win. That is a
    causal proof from the agent's OWN transitions: no oracle, no source reading, no per-game
    knowledge.

    THE MANDATORY CONTROL. The same statistic is kept on UNMASKED hashes. If the unmasked
    key ALSO shows two successors then the environment (or our observation of it) is simply
    non-deterministic there, and the violation is NOT attributable to the mask. Without this
    control the guard would fire spuriously: measured on sc25, 2-3 masked violations per seed
    are matched by 2-3 unmasked-control violations, so none of them is the mask's fault.

    THE RESPONSE, in escalating order:
      1. LOCAL SPLIT (default). The offending masked hash joins ``split_hashes``; the caller
         then hashes frames at that node by masked+unmasked, i.e. un-masks exactly the node
         that provably failed and keeps dedup everywhere else.
      2. GLOBAL REVOCATION, past ``max_split_nodes``. ``globally_revoked`` goes True and the
         caller falls back to unmasked identity for every subsequent frame.

    An HONEST NOTE ON REVOCATION. Nodes created BEFORE a split/revocation keep their old
    keys, so the graph briefly carries two identity conventions. That is a real cost, taken
    deliberately and bounded by ``max_split_nodes``: the alternative -- rehashing the whole
    graph -- is precisely what makes the reference solver re-initialise its graph hundreds of
    times per run and livelock. Every split and every revocation is COUNTED and surfaced in
    ``diagnostics()`` so the guard's activity is never invisible.

    OBSERVABILITY LIMIT. A violation can only be seen for a key tried at least twice.
    ``observable_keys`` is reported as the honest denominator; zero violations out of zero
    observable keys says nothing at all.
    """

    max_split_nodes: int = HUD_MASK_GUARD_MAX_SPLIT_NODES
    _masked_successors: dict[tuple, set[str]] = field(default_factory=dict)
    _unmasked_successors: dict[tuple, set[str]] = field(default_factory=dict)
    _controlled_keys: set[tuple] = field(default_factory=set)
    split_hashes: set[str] = field(default_factory=set)
    violations: int = 0
    refusals: int = 0
    non_deterministic_keys: int = 0
    uncontrolled_observations: int = 0
    uncontrolled_branchings_declined: int = 0
    globally_revoked: bool = False
    observations: int = 0

    def observe(
        self,
        *,
        origin_masked: Optional[str],
        origin_unmasked: Optional[str],
        action_key: Any,
        successor_masked: Optional[str],
        successor_unmasked: Optional[str],
    ) -> bool:
        """Record one realized transition. Returns True iff this call PROVED a new collapse.

        FAIL-SAFE ON A MISSING CONTROL (fixed 2026-07-25, found by this module's own smoke run
        rather than assumed): if the caller cannot supply the unmasked antecedent, there is NO
        control, and an uncontrolled masked branching is NOT a proof -- it is indistinguishable
        from ordinary environment non-determinism. The first version treated a missing control
        as a PASSED control and therefore un-masked 6 nodes on tu93 and 4 on lf52 on zero
        evidence, destroying both wins, while reporting
        ``non_deterministic_keys_excluded_by_control: 0`` -- a dead control channel reading as a
        clean one, which is precisely the uninstrumented-arm failure this project has already
        been burned by. Root cause: the antecedent frame was read from the graph node, and the
        bare explorer only RETAINS node frames when one of several optional components is
        enabled, so it was None on 1952 of 1952 transitions. Declined observations are counted
        (``uncontrolled_branchings_declined``) so a dead control can never again look clean.
        """

        if not origin_masked or not successor_masked:
            return False
        self.observations += 1
        masked_key = (str(origin_masked), _hashable(action_key))
        masked_seen = self._masked_successors.setdefault(masked_key, set())
        masked_seen.add(str(successor_masked))

        controlled = bool(origin_unmasked and successor_unmasked)
        if controlled:
            self._controlled_keys.add(masked_key)
            unmasked_key = (str(origin_unmasked), _hashable(action_key))
            unmasked_seen = self._unmasked_successors.setdefault(unmasked_key, set())
            unmasked_seen.add(str(successor_unmasked))
        else:
            self.uncontrolled_observations += 1
            unmasked_seen = set()

        if len(masked_seen) < 2:
            return False
        if not controlled or masked_key not in self._controlled_keys:
            # No control on this key -> no proof. Counted, never acted on.
            self.uncontrolled_branchings_declined += 1
            return False
        if len(unmasked_seen) >= 2:
            # The unmasked control also branches: non-determinism, not a mask collapse.
            self.non_deterministic_keys += 1
            return False
        if str(origin_masked) in self.split_hashes:
            return False

        self.violations += 1
        self.split_hashes.add(str(origin_masked))
        self.refusals += 1
        if len(self.split_hashes) > int(self.max_split_nodes):
            self.globally_revoked = True
        return True

    def observable_key_count(self) -> int:
        """Keys observed with >=2 DISTINCT masked successors -- where a collapse was provable.

        This is the honest denominator for ``violations``: a key with only one observed
        successor could not have exhibited a violation no matter how many times it was tried,
        so counting it would understate the rate. Reported alongside ``violations`` always,
        because zero violations out of zero provable keys says nothing at all.
        """

        return int(sum(1 for seen in self._masked_successors.values() if len(seen) >= 2))

    def is_split(self, masked_hash: Optional[str]) -> bool:
        if self.globally_revoked:
            return True
        return bool(masked_hash) and str(masked_hash) in self.split_hashes

    def diagnostics(self) -> dict:
        return {
            "observations": int(self.observations),
            "distinct_keys": int(len(self._masked_successors)),
            # Keys observed with >=2 DISTINCT successors -- i.e. where a collapse was provable.
            "keys_with_multiple_successors": int(self.observable_key_count()),
            "collapse_violations": int(self.violations),
            "collapse_refusals": int(self.refusals),
            "split_node_count": int(len(self.split_hashes)),
            "non_deterministic_keys_excluded_by_control": int(self.non_deterministic_keys),
            # THE CONTROL-CHANNEL HEALTH FIELDS. `control_live` False means the guard had no
            # control at all and therefore could not have proved anything, whatever the other
            # counters say. Emitted so a dead control can never read as a clean one again.
            "uncontrolled_observations": int(self.uncontrolled_observations),
            "uncontrolled_branchings_declined": int(self.uncontrolled_branchings_declined),
            "controlled_keys": int(len(self._controlled_keys)),
            "control_live": bool(self._controlled_keys),
            "globally_revoked": bool(self.globally_revoked),
            "max_split_nodes": int(self.max_split_nodes),
        }


def _hashable(value: Any) -> Any:
    """Coerce an action payload (possibly a dict) into a stable hashable key."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_hashable(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def mask_summary(mask: Any) -> dict:
    """Small serialisable description of a mask, for artifact rows."""

    if mask is None:
        return {"resolved": False, "cell_count": 0, "rows": [], "cols": []}
    arr = np.asarray(mask, dtype=bool)
    rows = sorted({int(y) for y in np.nonzero(arr.any(axis=1))[0].tolist()})
    cols = sorted({int(x) for x in np.nonzero(arr.any(axis=0))[0].tolist()})
    return {
        "resolved": bool(arr.any()),
        "cell_count": int(arr.sum()),
        "rows": rows,
        "cols": cols,
    }


__all__ = [
    "EdgeBarThresholds",
    "MaskCollapseGuard",
    "EDGE_BAR_EDGE_TOLERANCE",
    "EDGE_BAR_MIN_ELONGATION",
    "EDGE_BAR_MAX_MASK_AREA_FRACTION",
    "HUD_MASK_GUARD_MAX_SPLIT_NODES",
    "REGION_EVIDENCE_MIN_TRANSITIONS",
    "REGION_EVIDENCE_MIN_UBIQUITY",
    "edge_bar_hud_mask",
    "is_edge_bar_like",
    "mask_summary",
    "region_hud_evidence",
]
