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

  STAGE 1 ALONE IS NOT SAFE TO SHIP, and that is not a hypothetical (measured 2026-07-25).
  Across the 25 public games the repair changes the mask on exactly SIX -- ar25 63->127
  cells (adds column 63), sc25 0->128 (columns 62-63), lp85 0->64 (column 0), r11l 0->64
  (column 0), tn36 0->61 (row 1), cn04 0->32 (row 0) -- and on ar25 the newly-masked column
  is a FILL-LEVEL GAUGE, i.e. a decision-relevant state variable, not a clock. Masking it
  turned 1554 distinct raw frames into 233 nodes and the Stage-3 guard proved 4 aliasing
  keys on the first seed measured. Stage 1 must therefore be gated by Stage 2 before it is
  applied to identity; see ``DeferredMaskActivation``.

STAGE 2 -- ``region_hud_evidence(grids, mask, ...)``: multi-frame behavioural confirmation
  from transitions the agent ALREADY takes (zero extra actions, unlike
  ``arc_graph_explore.discover_hud_mask`` which burns up to 4 real actions from reset and is
  not viable under RHAE live scoring). Geometry is a SHAPE prior and provably cannot
  distinguish a live counter from a static decorative border; this stage is what does.
  A MEASUREMENT TRAP recorded here so it is not repeated: the per-CELL statistic "this cell
  changed on every action" returns 0 on every game measured, INCLUDING r11l, because a
  monotone FILLING bar changes only ONE cell per action. The correct statistic is
  REGION-level -- "the region ticks even when the rest of the board does not".

STAGE 2b -- ``DeferredMaskActivation``: the SEQUENCING that makes Stage 2 usable. Stage 1
  proposes a candidate; identity stays UNMASKED (today's shipped behaviour) until Stage 2 has
  >=16 observed transitions to judge it on; only an ``admit`` verdict ever applies the mask.
  Measured on the reset frames + real transitions: refuses ar25/sc25/lp85/cn04, admits
  r11l/tn36. This is the component that prevents Stage 1's ar25 over-mask from shipping.

STAGE 3 -- ``MaskCollapseGuard``: the runtime hard refusal. A ``(masked_node,
  concrete_action)`` key that is observed to produce TWO DIFFERENT masked successors shows
  one masked hash is covering two behaviourally distinct true states. Every such branching is
  counted and the offending node is un-masked (locally split, UNBOUNDED -- there is no global
  revocation, because flipping the hash convention mid-run was measured to corrupt 97.7% of
  the graph and to be strictly worse than no guard at all). It carries a MANDATORY unmasked
  CONTROL: if the same key also produces two different UNMASKED successors, the environment
  is simply non-deterministic at that node and the violation is NOT attributable to the mask
  (measured: sc25 shows 2-3 masked violations AND 2-3 unmasked-control violations, so
  nothing there is the mask's fault). The control's POWER is reported separately from its
  liveness: it can only exonerate when the unmasked antecedent REPEATS, which never happens
  on a monotone-counter region, so branchings on such keys are reported as
  ``unproven_masked_branchings`` (acted on conservatively, but not called proofs).

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
  key while new edges land under the compound key, i.e. the graph holds two identity
  conventions for THAT ONE NODE. This is the real, bounded cost of local retraction, taken
  deliberately (see ``MaskCollapseGuard``) instead of the rehash-everything approach that
  makes the reference solver re-initialise its graph hundreds of times per run.
  CORRECTION 2026-07-25: an earlier version of this bullet described the same cost as applying
  to GLOBAL revocation, calling it "brief" and "bounded by max_split_nodes". That was wrong and
  it understated a measured 97.7%: revocation flipped the hash convention for every subsequent
  frame while leaving all pre-revocation nodes under the old convention, so 640 of 655 nodes on
  tu93 and 1100 of 1161 on ar25 landed on the far side of the switch, 58 of 658 distinct raw
  frames held BOTH key forms, and the pre-revocation subgraph became unreachable. Global
  revocation is no longer the default and no longer changes hashing.

=====================================================================================
HONEST LIMITS
=====================================================================================
* A Stage-3 violation is only OBSERVABLE for a key tried at least twice, and the count grows
  with budget (lf52: 1 of 9 observable keys at 394 actions -> 3 of 30 at 765). Zero
  violations is therefore never proof of zero aliasing. ``observable_keys`` is always
  reported as the honest denominator.
* The Stage-3 control cannot exonerate on a monotone-counter region (its unmasked antecedent
  never repeats), so ``non_deterministic_keys_excluded_by_control: 0`` on such a key is a
  CONSTRUCTIONAL zero, not evidence. Consequence for the claim this module made earlier about
  the ALREADY-SHIPPED mask on tu93/lf52: "the shipped mask collapses provably-distinct states"
  is a HYPOTHESIS with a named confound (masked content causal vs. hidden state that is never
  rendered into the frame), not an established property of the live flag. The evidence is
  attributed, not asserted -- see ``unproven_masked_branchings``.
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

import hashlib
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
        ((on_top or on_bottom) and long_horizontal) or ((on_left or on_right) and long_vertical)
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
    if area <= 0:  # pragma: no cover - _as_grid rejects empty frames before this point
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
# Stage 2b -- DEFERRED ACTIVATION: Stage 1 proposes, Stage 2 confirms, only then apply
# ---------------------------------------------------------------------------

# Hard cap on how many frames the deferred-activation buffer holds while waiting for enough
# transitions to run Stage 2. Bounded because this buffer lives for the whole episode on a
# game that never accumulates `min_transitions` usable transitions, and an unbounded buffer on
# a 64x64 uint8 frame stream is a memory leak in a live agent. Past the cap the candidate is
# DISCARDED (never applied) rather than admitted on partial evidence -- the conservative
# direction, since under-masking only costs search efficiency.
DEFERRED_ACTIVATION_MAX_BUFFERED_FRAMES = 64


@dataclass
class DeferredMaskActivation:
    """Hold a Stage-1 candidate mask UNAPPLIED until Stage 2 confirms it behaves like a HUD.

    WHY THIS EXISTS -- the measured defect it repairs (2026-07-25 adversarial review).
    Stage 1 is single-frame GEOMETRY. Geometry is a shape prior and provably cannot tell a live
    counter from a decision-relevant state variable that happens to be drawn as an edge strip.
    On the public game ``ar25`` that distinction is load-bearing and Stage 1 gets it wrong: a
    colour-11 blob at bbox (0,63)-(63,63), h=64 w=1, fires ``is_edge_bar_like`` via
    ``on_right and long_vertical`` and newly masks all 64 cells of COLUMN 63. Column 63 is a
    FILL-LEVEL GAUGE, not a clock. Masking it collapsed provably-distinct states: with the mask
    applied, 1554 distinct raw frames became 233 graph nodes, and the Stage-3 guard proved 4
    aliasing keys (independent post-hoc analysis over a full 1168-transition arm-G log found 17
    observable keys and 17 proven collapses, 0 non-deterministic, with the environment's
    determinism separately confirmed on 20 repeated raw keys). Two antecedents differing in 6
    cells INSIDE the mask and 0 cells outside produced successors differing in 144 NON-masked
    cells, and ``col63_fill_height -> successor`` was a 1:1 function over fill heights 7..61.
    That is a state variable, and masking it is the CARDINAL SIN this module's own docstring
    names: over-masking destroys correctness, under-masking only costs efficiency.

    Stage 2 (``region_hud_evidence``) SEPARATES that corpus correctly -- measured on the same
    reset frames and the agent's own transitions: it REFUSES ar25 (``region_not_action_ubiquitous``,
    ubiquity 0.0, 110-115 in-episode revisits), sc25, lp85 and cn04, and ADMITS exactly the two
    clean winners r11l and tn36 (tick rate 1.0, ubiquity 1.0, 0 revisits). Its only blocker was
    SEQUENCING: it needs >=16 transitions and therefore cannot run at the single-frame,
    first-contact point ``REQ-ARC-WMTE-5583`` mandates the mask be computed at.

    THE FIX IS THE SEQUENCING, NOT THE STATISTIC. Identity stays UNMASKED (i.e. exactly today's
    shipped behaviour) until Stage 2 has enough evidence to decide. Admit -> the mask activates
    from that frame on. Refuse -> the candidate is discarded permanently and the run continues
    unmasked. Abstain past the buffer cap -> discarded.

    WHY LATE ACTIVATION IS SAFE BUT LATE REVOCATION IS NOT (the asymmetry that decides the
    design). Nodes created before activation carry unmasked keys; frames after it carry masked
    keys. The same true state can therefore appear under two keys -- that is DUPLICATION, i.e.
    under-dedup, i.e. a bounded search-efficiency cost. The reverse move (applying a mask early
    and withdrawing it mid-run, which is what ``MaskCollapseGuard``'s legacy global revocation
    did) leaves already-created nodes keyed by a COLLAPSING convention while new frames use
    another, and measured 97.7% of the graph on the wrong side of the switch. Under-masking is
    recoverable; over-masking is not.
    """

    min_transitions: int = REGION_EVIDENCE_MIN_TRANSITIONS
    max_buffered_frames: int = DEFERRED_ACTIVATION_MAX_BUFFERED_FRAMES
    min_ubiquity: float = REGION_EVIDENCE_MIN_UBIQUITY
    max_revisits: int = REGION_EVIDENCE_MAX_REVISITS
    candidate: Optional[np.ndarray] = None
    # The mask the SHIPPED classifier resolves on the same frame -- the FALLBACK, and the region
    # Stage 2 must NOT judge. See `propose`.
    baseline: Optional[np.ndarray] = None
    verdict: str = "no_candidate"
    reason: str = "stage1_proposed_nothing"
    evidence: dict = field(default_factory=dict)
    activated_after_transitions: Optional[int] = None
    _added_region: Optional[np.ndarray] = None
    _grids: list = field(default_factory=list)
    _actions: list = field(default_factory=list)

    def propose(self, candidate: Any, baseline: Any = None) -> None:
        """Register the Stage-1 candidate and the shipped-mask BASELINE it must never undercut.

        STAGE 2 JUDGES ONLY THE REPAIR-ADDED REGION (fixed 2026-07-25, found by the full-corpus
        A/B rather than assumed). The first version fed Stage 2 the whole candidate mask -- which
        is the UNION of the shipped mask and the repair's additions -- so a refusal threw away the
        SHIPPED mask too, on games where no repair-added cell exists at all:

            su15  1 level / 534 actions / 21 nodes  ->  0 levels / 1947 actions / 238 nodes
            dc22  1 level / 1769 actions / 120 nodes ->  0 levels / 1967 actions / 1028 nodes

        Both have a 64-cell shipped mask and ZERO repair-added cells, so the treatment arm should
        have been byte-identical to its control there. That is a direct violation of the
        superset-by-construction property this requirement rests on: the repair may only ADD
        masked cells, never remove the dedup the live configuration already gets.

        With the baseline supplied, the semantics become: apply the SHIPPED mask immediately
        (exactly today's behaviour), and let Stage 2 decide only whether to WIDEN it. A refusal
        returns to -- never below -- today's behaviour. It is also the more precise question: the
        thing under test is whether the newly-detected cells behave like a HUD, and mixing the
        shipped bar's statistics into that measurement is what fooled it.
        """

        candidate_arr = None if candidate is None else np.asarray(candidate, dtype=bool)
        baseline_arr = None if baseline is None else np.asarray(baseline, dtype=bool)
        if baseline_arr is not None and not baseline_arr.any():
            baseline_arr = None
        self.baseline = baseline_arr
        if candidate_arr is None or not candidate_arr.any():
            self.candidate = None
            self.verdict = "no_candidate"
            self.reason = "stage1_proposed_nothing"
            self._added_region = None
            return
        self.candidate = candidate_arr
        added = (
            candidate_arr
            if baseline_arr is None or baseline_arr.shape != candidate_arr.shape
            else (candidate_arr & ~baseline_arr)
        )
        if not added.any():
            # Nothing was ADDED, so there is nothing for Stage 2 to judge and no reason to defer:
            # the candidate IS today's mask. Applying it immediately keeps the arm byte-identical
            # to its control on every inert game.
            self._added_region = None
            self.verdict = "no_added_region"
            self.reason = "repair_added_no_cell_so_the_shipped_mask_applies_unchanged"
            return
        self._added_region = added
        self.verdict = "pending"
        self.reason = "awaiting_stage2_transitions"

    @property
    def pending(self) -> bool:
        return self.verdict == "pending" and self.candidate is not None

    def observe(self, grid: Any, action_label: Any = None) -> Optional[np.ndarray]:
        """Record one observed frame. Returns the mask to ACTIVATE, or None.

        A non-None return happens at most ONCE per instance (the frame on which Stage 2
        admitted). Every other call returns None, which the caller reads as "keep the current
        identity convention" -- so a caller that ignores the return value silently keeps
        today's unmasked behaviour rather than silently masking.
        """

        if not self.pending:
            return None
        prepared = _as_grid(grid)
        self._grids.append(prepared)
        self._actions.append(action_label)

        usable = sum(1 for g in self._grids if g is not None)
        if usable < int(self.min_transitions) + 1:
            if len(self._grids) >= int(self.max_buffered_frames):
                self.verdict = "discarded"
                self.reason = "buffer_cap_reached_without_enough_usable_transitions"
                self.evidence = {
                    "buffered_frames": len(self._grids),
                    "usable_frames": usable,
                }
                self._release()
            return None

        evidence = region_hud_evidence(
            self._grids,
            # THE REPAIR-ADDED REGION ONLY -- never the union with the shipped mask. See `propose`.
            self._added_region,
            actions=self._actions,
            min_transitions=self.min_transitions,
            min_ubiquity=self.min_ubiquity,
            max_revisits=self.max_revisits,
        )
        self.evidence = evidence
        if evidence.get("verdict") == "admit":
            self.verdict = "admitted"
            self.reason = str(evidence.get("reason") or "action_ubiquitous_and_monotone")
            self.activated_after_transitions = int(evidence.get("n_transitions") or 0)
            mask = self.candidate
            self._release()
            return mask
        if evidence.get("verdict") == "refuse":
            self.verdict = "refused"
            self.reason = str(evidence.get("reason") or "stage2_refused")
            self._release()
            return None
        # abstain: keep buffering until the cap, then discard rather than guess.
        if len(self._grids) >= int(self.max_buffered_frames):
            self.verdict = "discarded"
            self.reason = "buffer_cap_reached_while_stage2_still_abstaining"
            self._release()
        return None

    def _release(self) -> None:
        """Drop the frame buffer. A decided instance must not keep holding frames."""

        self._grids = []
        self._actions = []

    def fallback_mask(self) -> Optional[np.ndarray]:
        """What identity should use when Stage 2 is pending, refusing, or discarding.

        The SHIPPED mask -- i.e. exactly today's live behaviour -- never None-when-a-shipped-mask-
        exists. This is what makes a Stage-2 refusal a return to the baseline rather than a
        regression below it.
        """

        return self.baseline

    def diagnostics(self) -> dict:
        return {
            "stage2_verdict": str(self.verdict),
            "stage2_reason": str(self.reason),
            "stage2_min_transitions": int(self.min_transitions),
            "candidate_cell_count": (
                int(np.asarray(self.candidate, dtype=bool).sum())
                if self.candidate is not None
                else 0
            ),
            # The two numbers that make a refusal readable: how many cells the repair would have
            # ADDED, and how many the run keeps regardless (the shipped baseline).
            "repair_added_cell_count": (
                int(self._added_region.sum()) if self._added_region is not None else 0
            ),
            "baseline_cell_count": (int(self.baseline.sum()) if self.baseline is not None else 0),
            "activated_after_transitions": self.activated_after_transitions,
            "buffered_frames": len(self._grids),
            "evidence": dict(self.evidence),
            # The honest reading of an unapplied candidate: identity is EXACTLY today's live
            # behaviour (the shipped mask, if the shipped classifier resolved one), not "the
            # detector failed" and not "no mask at all".
            "identity_convention_while_pending": "shipped_mask_same_as_live_default",
        }


# ---------------------------------------------------------------------------
# Stage 3 -- the runtime collapse guard (HARD refusal)
# ---------------------------------------------------------------------------

# Local splits are UNBOUNDED by default (None). Each split un-masks exactly the one node that
# branched, so N splits degrade the run gracefully toward unmasked identity -- one identity
# convention per node, no global switch.
#
# WHY THE OLD CAP OF 3 WAS REMOVED (measured harm, 2026-07-25 adversarial review). Past the cap
# the guard set `globally_revoked`, after which `is_split()` returned True UNCONDITIONALLY and
# `_hash` emitted the compound `masked|u:unmasked` key for EVERY frame -- while nodes created
# BEFORE revocation kept their plain masked keys. That is not "fall back to the unmasked
# baseline"; it is two identity conventions in one graph, and it was measured to be strictly
# WORSE than shipping no guard at all:
#
#   * tu93, where the repaired mask is IDENTICAL to the shipped one (64 cells both), so arming
#     the guard was the ONLY difference from the live config: 1 level / 361 actions -> 0 levels
#     / 1953 actions on 3 of 3 seeds. Same on lf52 seed 20260726 (1 -> 0).
#   * Instrumented on tu93 seed 20260724: 72 hashes computed pre-revocation, 1927 post, and 58
#     of 658 distinct RAW frames ended up holding BOTH a plain masked key and a compound key --
#     the same true state present twice under two conventions. 640 of 655 graph nodes (97.7%)
#     were post-revocation; on ar25, 1100 of 1161.
#   * The pre-revocation subgraph becomes structurally UNREACHABLE (`_hash` can never re-emit a
#     plain key), so its accumulated `path`/`adj` knowledge is dead while navigation can still
#     target it.
#
# The module docstring previously described this as "the graph BRIEFLY carries two identity
# conventions ... BOUNDED BY max_split_nodes". That is true of a LOCAL split and false of
# revocation, which is unbounded. Corrected here rather than left understated.
HUD_MASK_GUARD_MAX_SPLIT_NODES: Optional[int] = None

# Purely for REPORTING: past this many splits the mask is probably wrong wholesale and the
# artifact should say so. It changes NO behaviour (see `split_budget_exceeded`), because the
# only correctness-preserving way to withdraw a mask mid-run would be to REHASH the whole graph
# under the unmasked key, and the explorer does not retain node frames to rehash from.
HUD_MASK_GUARD_SPLIT_REPORTING_THRESHOLD = 3

# The two revocation modes. `local_split_only` is the default and the only one that keeps one
# identity convention per node. `global_hash_flip` is the measured-harmful legacy behaviour,
# retained ONLY so a regression test can demonstrate the corruption; it must not be enabled on
# any live or flip-candidate configuration.
HUD_MASK_GUARD_REVOCATION_LOCAL = "local_split_only"
HUD_MASK_GUARD_REVOCATION_GLOBAL_HASH_FLIP = "global_hash_flip_measured_harmful_do_not_ship"


@dataclass
class MaskCollapseGuard:
    """Proves, at runtime, whether an applied HUD mask is collapsing DISTINCT states.

    THE PROOF OBLIGATION. In a deterministic environment, one ``(state, action)`` pair has
    exactly one successor. So if a ``(masked_hash, concrete_action)`` key is observed to
    produce TWO DIFFERENT masked successors, the masked hash must be covering at least two
    behaviourally distinct true states -- a genuine collapse, not a dedup win. That is a
    causal proof from the agent's OWN transitions: no oracle, no source reading, no per-game
    knowledge.

    THE MANDATORY CONTROL, AND ITS POWER LIMIT. The same statistic is kept on UNMASKED hashes.
    If the unmasked key ALSO shows two successors then the environment (or our observation of
    it) is simply non-deterministic there, and the violation is NOT attributable to the mask.
    Without this control the guard would fire spuriously: measured on sc25, 2-3 masked
    violations per seed are matched by 2-3 unmasked-control violations, so none of them is the
    mask's fault.

    But LIVENESS IS NOT POWER, and conflating them overstated every conclusion this guard has
    ever supported (found 2026-07-25 by adversarial review, then confirmed by direct probe).
    The exoneration branch can only fire when the UNMASKED antecedent REPEATS -- and if the
    masked region is a monotone counter, the unmasked antecedent never repeats by construction,
    so the branch is unreachable. On such a key ``non_deterministic_keys_excluded_by_control ==
    0`` is a CONSTRUCTIONAL zero, not evidence of determinism. Probe result: a genuinely
    non-deterministic node with a ticking masked region was CONVICTED (excluded_by_control 0,
    control_live True), while the identical scenario with a non-ticking region was correctly
    exonerated (excluded 1). So this class now reports three distinct outcomes:

      * ``proven_collapses`` -- the control had POWER (that unmasked key was observed 2+ times)
        and showed exactly ONE successor. Non-determinism is genuinely ruled out.
      * ``unproven_masked_branchings`` -- the control was live but POWERLESS (the unmasked
        antecedent never repeated). The branching is still strong evidence that the masked
        content is decision-relevant (two frames identical OUTSIDE the mask, differing INSIDE
        it, produced different successors), but "unobserved hidden state that is never rendered
        into the frame" is an equally consistent explanation, so it is NOT a proof.
      * ``non_deterministic_keys`` -- the control had power and branched: excluded, no action.

    WHY AN UNPROVEN BRANCHING IS COUNTED BUT NOT ACTED ON.

    THE INTUITION, WHICH IS WRONG: "splitting costs only search efficiency while not splitting
    risks correctness, so under this module's stated asymmetry the conservative move is to
    un-mask the node in BOTH the proven and the unproven case." That was the first version's
    reasoning and it is preserved here because it is the natural reading of the asymmetry -- it
    is simply not what the corpus does.

    THE MEASUREMENT THAT REFUTED IT (full 25-game corpus, 3 seeds, budget 2000, 2026-07-25).
    Acting on unproven branchings shattered the graph on games where the mask is the
    ALREADY-SHIPPED one, unchanged by any repair:

        tu93   60 nodes / 1 level / 361 actions  ->  578 nodes / 0 levels / 1957 actions
               (28 refusals: 8 proven, 20 UNPROVEN)
        dc22  120 nodes / 1 level / 1769 actions ->  311 nodes / 0 levels / 1943 actions
               (35 refusals: 3 proven, 32 UNPROVEN)
        su15   21 nodes / 1 level / 534 actions  ->  270 nodes / 0 levels / 1947 actions
               (12 refusals: 2 proven, 10 UNPROVEN)

    The guard, armed as the SAFETY mechanism, was itself the single largest source of lost wins
    in that run. The unproven majority is not incidental: on a monotone-counter region the
    unmasked antecedent never repeats BY CONSTRUCTION, so most branchings there can NEVER be
    proven, and treating each one as a reason to un-mask reverses the dedup the mask exists to
    provide. Splitting therefore does NOT cost "only efficiency" -- past a handful of splits it
    costs the win outright.

    Default is PROVEN-ONLY, with unproven branchings counted, attributed, and reported. Set
    ``act_on_unproven_branchings=True`` to restore the aggressive behaviour (kept so the
    measurement above stays reproducible).

    THE HONEST COST OF THAT CHOICE: the guard is now weak on exactly the region class where the
    control has no power. That gap is why ``DeferredMaskActivation`` (Stage 2) is MANDATORY --
    it refuses a bad mask BEFORE it is ever applied, using a statistic that does not depend on
    the control repeating. Stage 3 alone was never sufficient, and this measurement is why.

    THE RESPONSE: LOCAL SPLIT, unbounded. The offending masked hash joins ``split_hashes``; the
    caller then hashes frames at that node by masked+unmasked, i.e. un-masks exactly the node
    that failed and keeps dedup everywhere else. There is NO global revocation by default -- see
    ``HUD_MASK_GUARD_MAX_SPLIT_NODES`` above for the measured 97.7%-of-the-graph corruption that
    mode caused, and ``revocation_mode`` for why it survives only as a test fixture.

    An HONEST NOTE ON LOCAL SPLITS. Nodes created BEFORE a split keep their old key, so that ONE
    node's edges live under the plain masked key while new edges land under the compound key.
    That cost is genuinely bounded (one node per split, one convention per node) -- unlike the
    global flip, which re-keyed everything. Every split is COUNTED and surfaced in
    ``diagnostics()`` so the guard's activity is never invisible.

    OBSERVABILITY LIMIT. A violation can only be seen for a key tried at least twice.
    ``observable_keys`` is reported as the honest denominator; zero violations out of zero
    observable keys says nothing at all.
    """

    max_split_nodes: Optional[int] = HUD_MASK_GUARD_MAX_SPLIT_NODES
    revocation_mode: str = HUD_MASK_GUARD_REVOCATION_LOCAL
    # Act on branchings the unmasked control could not have exonerated? Default False -- see the
    # class docstring for the measured harm (tu93 1 level -> 0, dc22 1 -> 0, su15 1 -> 0, all on
    # games whose mask no repair touched, with 20/32/10 of those refusals unproven).
    act_on_unproven_branchings: bool = False
    # Retract only cells THIS REPAIR ADDED, never cells the live configuration already masks?
    # Default True. WHY (measured, full corpus, 2026-07-25): restricting the guard to PROVEN
    # collapses was not enough. The ALREADY-SHIPPED mask aliases heavily on its own -- 441 proven
    # collapses across the corpus with the guard armed -- and splitting those nodes cost the run
    # dc22, su15, tu93 and lf52, i.e. wins the LIVE configuration currently holds BECAUSE the
    # shipped mask collapses those states. Acting on them means this experiment regresses the
    # baseline to fix a defect it did not introduce, which is out of scope and strictly worse
    # than shipping nothing. Those branchings are still COUNTED and ATTRIBUTED (they are a real,
    # operator-visible finding about `SUBMITTED_AUTO_HUD_MASK_ENABLED`), just not acted on here.
    restrict_action_to_repair_added_region: bool = True
    split_reporting_threshold: int = HUD_MASK_GUARD_SPLIT_REPORTING_THRESHOLD
    # REGION ATTRIBUTION (optional). When the caller supplies the mask the SHIPPED classifier
    # would have produced and the mask actually applied, every acted-on branching is attributed
    # to the REPAIR-ADDED region or to the already-shipped region -- computed from the guard's
    # own antecedent frames, so it needs no win to have been lost and covers every game in a
    # run. See `_hud_aliasing_attribution`'s window defect for why this exists.
    applied_mask: Optional[np.ndarray] = None
    shipped_mask: Optional[np.ndarray] = None
    _masked_successors: dict[tuple, set[str]] = field(default_factory=dict)
    _unmasked_successors: dict[tuple, set[str]] = field(default_factory=dict)
    _unmasked_key_counts: dict[tuple, int] = field(default_factory=dict)
    _controlled_keys: set[tuple] = field(default_factory=set)
    _masked_key_grids: dict[tuple, Any] = field(default_factory=dict)
    split_hashes: set[str] = field(default_factory=set)
    violations: int = 0
    refusals: int = 0
    proven_collapses: int = 0
    unproven_masked_branchings: int = 0
    non_deterministic_keys: int = 0
    uncontrolled_observations: int = 0
    uncontrolled_branchings_declined: int = 0
    keys_with_repeated_unmasked_antecedent: int = 0
    globally_revoked: bool = False
    observations: int = 0
    attribution_added_region: int = 0
    attribution_shipped_region: int = 0
    attribution_outside_mask: int = 0
    attribution_unavailable: int = 0
    branchings_in_shipped_region_not_acted_on: int = 0

    # Bound on how many antecedent grids are retained for region attribution. One 64x64 uint8
    # grid is 4 KiB, so 512 keys is ~2 MiB -- enough to cover every observable key in every
    # measured cell (max seen: 25) with three orders of magnitude of headroom, while still being
    # a hard bound rather than an unbounded per-run leak.
    max_retained_antecedent_grids: int = 512

    def observe(
        self,
        *,
        origin_masked: Optional[str],
        origin_unmasked: Optional[str],
        action_key: Any,
        successor_masked: Optional[str],
        successor_unmasked: Optional[str],
        origin_grid: Any = None,
    ) -> bool:
        """Record one realized transition. Returns True iff this call ACTED on a new branching.

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

        ``origin_grid`` is optional and is used ONLY for region attribution (which part of the
        mask the differing antecedent cells fall in). Omitting it costs a counted
        ``attribution_unavailable``, never a changed decision.
        """

        if not origin_masked or not successor_masked:
            return False
        self.observations += 1
        masked_key = (str(origin_masked), _hashable(action_key))
        masked_seen = self._masked_successors.setdefault(masked_key, set())
        first_successor_for_key = not masked_seen
        masked_seen.add(str(successor_masked))
        if (
            first_successor_for_key
            and origin_grid is not None
            and len(self._masked_key_grids) < int(self.max_retained_antecedent_grids)
        ):
            retained = _as_grid(origin_grid)
            if retained is not None:
                self._masked_key_grids[masked_key] = retained.copy()

        controlled = bool(origin_unmasked and successor_unmasked)
        control_had_power = False
        if controlled:
            self._controlled_keys.add(masked_key)
            unmasked_key = (str(origin_unmasked), _hashable(action_key))
            unmasked_seen = self._unmasked_successors.setdefault(unmasked_key, set())
            unmasked_seen.add(str(successor_unmasked))
            seen_count = self._unmasked_key_counts.get(unmasked_key, 0) + 1
            self._unmasked_key_counts[unmasked_key] = seen_count
            if seen_count == 2:
                # First time this unmasked antecedent+action REPEATED: from here on the control
                # is capable of exonerating, which is what "power" means.
                self.keys_with_repeated_unmasked_antecedent += 1
            control_had_power = seen_count >= 2
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

        if control_had_power:
            self.proven_collapses += 1
        else:
            # The control was live but could not have fired (the unmasked antecedent never
            # repeated), so this branching is NOT a proof -- counted and attributed, but by
            # default not acted on (see `act_on_unproven_branchings`).
            self.unproven_masked_branchings += 1
            if not self.act_on_unproven_branchings:
                self._attribute(masked_key, origin_grid)
                return False
        region = self._attribute(masked_key, origin_grid)
        if self.restrict_action_to_repair_added_region and region in {"shipped", "outside"}:
            # The differing cells lie in the mask the LIVE configuration already applies, not in
            # anything this repair added. Un-masking here would regress the baseline to fix a
            # pre-existing defect -- out of scope, and measured to cost dc22/su15/tu93/lf52.
            self.branchings_in_shipped_region_not_acted_on += 1
            return False
        self.violations += 1
        self.split_hashes.add(str(origin_masked))
        self.refusals += 1
        if (
            self.revocation_mode == HUD_MASK_GUARD_REVOCATION_GLOBAL_HASH_FLIP
            and self.max_split_nodes is not None
            and len(self.split_hashes) > int(self.max_split_nodes)
        ):
            self.globally_revoked = True
        return True

    def _attribute(self, masked_key: tuple, origin_grid: Any) -> Optional[str]:
        """Which REGION do the two antecedents differ in -- repair-added, or already-shipped?

        Computed from the guard's own retained antecedent grid for this key, so it needs no win
        to have been lost and covers every game in a run (the defect in the harness's
        win/loss-keyed attribution window).

        Returns ``"added"`` / ``"shipped"`` / ``"outside"``, or ``None`` when it cannot tell.
        The caller uses that to decide whether acting would retract a cell the LIVE
        configuration already masks. A ``None`` return is treated as UNATTRIBUTABLE, and the
        caller's default is to leave the baseline alone rather than act on evidence it could not
        place -- except where there is no shipped mask at all, in which case every masked cell is
        repair-added by construction.
        """

        if self.applied_mask is None:
            self.attribution_unavailable += 1
            return None
        previous = self._masked_key_grids.get(masked_key)
        current = _as_grid(origin_grid)
        if previous is None or current is None or previous.shape != current.shape:
            self.attribution_unavailable += 1
            return None if self.shipped_mask is not None else "added"
        applied = np.asarray(self.applied_mask, dtype=bool)
        if applied.shape != previous.shape:
            self.attribution_unavailable += 1
            return None if self.shipped_mask is not None else "added"
        differing = previous != current
        shipped = (
            np.asarray(self.shipped_mask, dtype=bool)
            if self.shipped_mask is not None
            and np.asarray(self.shipped_mask).shape == applied.shape
            else np.zeros_like(applied)
        )
        added = applied & ~shipped
        if bool((differing & added).any()):
            self.attribution_added_region += 1
            return "added"
        if bool((differing & shipped).any()):
            self.attribution_shipped_region += 1
            return "shipped"
        if True:
            # Differing cells lie entirely OUTSIDE the mask. That cannot be the mask's fault --
            # two frames differing outside the mask would not share a masked hash -- so it is
            # recorded rather than attributed.
            self.attribution_outside_mask += 1
            return "outside"

    def observable_key_count(self) -> int:
        """Keys observed with >=2 DISTINCT masked successors -- where a collapse was provable.

        This is the honest denominator for ``violations``: a key with only one observed
        successor could not have exhibited a violation no matter how many times it was tried,
        so counting it would understate the rate. Reported alongside ``violations`` always,
        because zero violations out of zero provable keys says nothing at all.
        """

        return int(sum(1 for seen in self._masked_successors.values() if len(seen) >= 2))

    def is_split(self, masked_hash: Optional[str]) -> bool:
        """Should this masked hash be keyed by masked+unmasked instead?

        In the default ``local_split_only`` mode this is TRUE only for hashes that actually
        branched -- one identity convention per node. The legacy ``global_hash_flip`` mode
        returns True for EVERYTHING once revoked, which is the measured-harmful behaviour kept
        only as a test fixture; see ``HUD_MASK_GUARD_MAX_SPLIT_NODES``.
        """

        if self.globally_revoked and self.revocation_mode == (
            HUD_MASK_GUARD_REVOCATION_GLOBAL_HASH_FLIP
        ):
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
            # THE CONTROL-POWER FIELDS (added 2026-07-25). `control_live` says an unmasked
            # antecedent was SUPPLIED; it does NOT say the control could have fired. The
            # exoneration branch needs the unmasked antecedent to REPEAT, which never happens
            # when the masked region is a monotone counter -- so on those keys
            # `non_deterministic_keys_excluded_by_control: 0` is a constructional zero.
            # `proven_collapses` counts only branchings where the control genuinely had power.
            "proven_collapses": int(self.proven_collapses),
            "unproven_masked_branchings": int(self.unproven_masked_branchings),
            "acted_on_unproven_branchings": bool(self.act_on_unproven_branchings),
            "restricted_action_to_repair_added_region": bool(
                self.restrict_action_to_repair_added_region
            ),
            # Proven collapses in the ALREADY-SHIPPED mask that this run deliberately did NOT act
            # on. A real, operator-visible property of the live configuration -- surfaced here
            # rather than silently fixed by regressing the baseline.
            "branchings_in_shipped_region_not_acted_on": int(
                self.branchings_in_shipped_region_not_acted_on
            ),
            "keys_with_repeated_unmasked_antecedent": int(
                self.keys_with_repeated_unmasked_antecedent
            ),
            "control_had_power_on_any_key": bool(self.keys_with_repeated_unmasked_antecedent),
            "refusals_are_all_proven": bool(self.unproven_masked_branchings == 0),
            # THE CONTROL-CHANNEL HEALTH FIELDS. `control_live` False means the guard had no
            # control at all and therefore could not have proved anything, whatever the other
            # counters say. Emitted so a dead control can never read as a clean one again.
            "uncontrolled_observations": int(self.uncontrolled_observations),
            "uncontrolled_branchings_declined": int(self.uncontrolled_branchings_declined),
            "controlled_keys": int(len(self._controlled_keys)),
            "control_live": bool(self._controlled_keys),
            # REGION ATTRIBUTION: whose mask is aliasing. Zero-filled when the caller supplied
            # no `applied_mask`/`origin_grid`, and that state is COUNTED
            # (`attribution_unavailable`) rather than reported as "no added-region aliasing".
            "attribution": {
                "branchings_differing_in_repair_added_region": int(self.attribution_added_region),
                "branchings_differing_in_already_shipped_region": int(
                    self.attribution_shipped_region
                ),
                "branchings_differing_outside_the_mask": int(self.attribution_outside_mask),
                "attribution_unavailable": int(self.attribution_unavailable),
                "regions_supplied": bool(self.applied_mask is not None),
            },
            "revocation_mode": str(self.revocation_mode),
            # Reporting only: past the threshold the mask is probably wrong wholesale. Changes
            # NO behaviour -- withdrawing a mask mid-run corrupts the graph (measured 97.7%).
            "split_budget_exceeded": bool(
                len(self.split_hashes) > int(self.split_reporting_threshold)
            ),
            "split_reporting_threshold": int(self.split_reporting_threshold),
            "globally_revoked": bool(self.globally_revoked),
            "max_split_nodes": self.max_split_nodes,
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


def mask_cell_digest(mask: Any) -> Optional[str]:
    """A content digest of the mask's CELL SET -- the identity two masks must be compared on.

    WHY A DIGEST AND NOT ``cell_count`` (fixed 2026-07-25). The harness compared a control mask
    to a treatment mask by equality of ``cell_count``. Equal counts DO imply "the repair added
    nothing" on the games measured so far, but the converse is not sound in general: two masks of
    equal size can occupy different cells, and reading equal counts as "same mask" would silently
    exonerate a repair that MOVED the mask instead of widening it. The digest is exact.
    """

    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return None
    return hashlib.sha256(
        b"|".join([str(arr.shape).encode("ascii"), np.packbits(arr).tobytes()])
    ).hexdigest()[:16]


def mask_summary(mask: Any) -> dict:
    """Small serialisable description of a mask, for artifact rows."""

    if mask is None:
        return {"resolved": False, "cell_count": 0, "rows": [], "cols": [], "digest": None}
    arr = np.asarray(mask, dtype=bool)
    rows = sorted({int(y) for y in np.nonzero(arr.any(axis=1))[0].tolist()})
    cols = sorted({int(x) for x in np.nonzero(arr.any(axis=0))[0].tolist()})
    return {
        "resolved": bool(arr.any()),
        "cell_count": int(arr.sum()),
        "rows": rows,
        "cols": cols,
        "digest": mask_cell_digest(arr),
    }


__all__ = [
    "DeferredMaskActivation",
    "EdgeBarThresholds",
    "MaskCollapseGuard",
    "DEFERRED_ACTIVATION_MAX_BUFFERED_FRAMES",
    "EDGE_BAR_EDGE_TOLERANCE",
    "EDGE_BAR_MIN_ELONGATION",
    "EDGE_BAR_MAX_MASK_AREA_FRACTION",
    "HUD_MASK_GUARD_MAX_SPLIT_NODES",
    "HUD_MASK_GUARD_REVOCATION_GLOBAL_HASH_FLIP",
    "HUD_MASK_GUARD_REVOCATION_LOCAL",
    "HUD_MASK_GUARD_SPLIT_REPORTING_THRESHOLD",
    "REGION_EVIDENCE_MIN_TRANSITIONS",
    "REGION_EVIDENCE_MIN_UBIQUITY",
    "edge_bar_hud_mask",
    "is_edge_bar_like",
    "mask_cell_digest",
    "mask_summary",
    "region_hud_evidence",
]
