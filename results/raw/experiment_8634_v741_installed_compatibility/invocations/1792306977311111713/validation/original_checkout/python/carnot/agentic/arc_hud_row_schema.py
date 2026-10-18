"""ONE projection of `StepwiseExplorer.hud_mask_diagnostics()` onto a result ROW, shared by every
harness that records it, plus the compatibility read that makes already-recorded rows of EITHER
historical schema interchangeable.

WHY THIS MODULE EXISTS -- THE MEASURED DEFECT (2026-07-26)
==========================================================
The same quantity had two different addresses on a result row, depending on which harness wrote it,
and each reader saw a clean, error-free `None` on the other writer's rows:

  * `scripts/arc_scored_path_lever_harness.py` wrote the diagnostics NESTED, under
    `row["lever2_hud_fire"]`. All 805 recorded rows of
    `results/outer_loop_scored_path_lever_ab_llm_on_20260726.json` read `None` at the FLAT address
    `row["hud_mask_resolved"]` while the nested copy was populated in all 805 (True 598 / False 207).
  * `python/carnot/experiment_5836_frontier_discipline_ab.py:run_cell` wrote them FLAT, at row top
    level, with NO nested copy. All 1713 rows of `results/cptb_20260726_cells/*.jsonl.gz` therefore
    scored `lever2_fired = False` under a nested-only reader.

`None` rather than `False` is the tell in both directions, and it is the same defect class as the
exp5836 dead observe channel: a diagnostic that reads as a measured zero when in fact nothing was
consulted. Two writers projecting the same source dict independently IS the defect; one shared
function is the fix. Consumers that read a row must go through `backfill_hud_flat_fields` at a
single chokepoint so no call site can be the forgotten one.

WHAT BACK-FILL CANNOT FIX (state this, never paper over it)
===========================================================
Back-fill recovers a field that was WRITTEN at the other address. It cannot recover a field that was
never written at all. exp5836's hand-rolled projection omitted `hud_shipped_mask_digest`, which is
the comparator `hud_lever_fired` needs, so all 1713 already-recorded cptb rows remain PERMANENTLY
unscoreable for lever 2 -- `lever2_scoreable(row)` is False for every one of them. Those rows must
be excluded from a lever-2 denominator, NEVER counted as 1713 non-fires. Only a re-run captures the
datum. This module exports `lever2_scoreable` precisely so that distinction is available to every
reader instead of being re-derived (or forgotten) per consumer.
"""

from __future__ import annotations

# The FLAT hud_* key names that both writers must emit. Named here -- once -- so the two harnesses
# are PROVABLY comparable rather than accidentally similar, and so a future added key has exactly
# one place to be added. `tests/python/test_arc_hud_row_schema.py` asserts both writers' rows carry
# this exact set.
HUD_ROW_KEYS: tuple[str, ...] = (
    # --- the detector's own resolution -------------------------------------------------------
    "hud_mask_resolved",
    "hud_mask_cell_count",
    "hud_mask_digest",
    "hud_mask_source",
    # --- stage 2's verdict on this cell -------------------------------------------------------
    "hud_mask_stage2_verdict",
    "hud_mask_stage2_reason",
    "hud_mask_stage2_candidate_cells",
    # --- node identity / dedup ---------------------------------------------------------------
    "unique_frames",
    "graph_nodes",
    "node_inflation_vs_unique_frames",
    "collapse_guard_refusals",
    # --- the shipped-side comparator the fire predicate needs ---------------------------------
    # exp5836's hand-rolled projection omitted BOTH of these, which is why its 1713 recorded rows
    # cannot answer "did the repaired detector's mask differ from the shipped classifier's".
    "hud_shipped_mask_cell_count",
    "hud_shipped_mask_digest",
    # --- populated-vs-unread witness ----------------------------------------------------------
    "hud_diagnostics_readable",
    "hud_diagnostics_error",
    # --- the derived flag + the predicate version that produced it ----------------------------
    "lever2_fired",
    "lever2_fired_predicate",
)

# The DEPRECATED short alias for `node_inflation_vs_unique_frames`. exp5836 renamed the field on its
# own rows and reads it back under the short name in three places, so the shared projection emits
# BOTH for one release rather than breaking those readers in the same commit that unifies the
# schema. Removing the alias is a separate, greppable change.
NODE_INFLATION_ALIAS = "node_inflation"

# Stamped onto every row so a row produced by the PRE-FIX predicate is distinguishable from one
# produced by the current predicate. Without it, `lever2_fired: False` on an already-recorded row is
# ambiguous between "the lever did not fire" and "the broken predicate could not see it fire" -- and
# the 430 recorded budget-400 cells are all of the second kind. Bump on any predicate change.
LEVER2_FIRE_PREDICATE_VERSION = "2026-07-26.appearing-mask-counts-digest-compared"


def hud_lever_fired(hud: dict | None) -> bool:
    """Did the HUD edge-bar trio (lever 2) actually DO SOMETHING in this cell?

    THE DEFECT THIS PREDICATE EXISTS TO FIX (2026-07-26, measured). The first version ANDed in
    `hud_shipped_mask_digest` -- i.e. it required the ALREADY-SHIPPED `auto_hud_mask` classifier to
    have produced a mask before the repaired detector's mask could count as a difference. That is
    ANTI-CORRELATED with the lever it measures: the whole reason REQ-ARC-WMTE-5960 exists is that
    the shipped classifier returns None on r11l and tn36, and those are the only two games in the
    corpus where the repaired detector resolves a mask the shipped one does not. So `lever2_fired`
    was False in ALL 430 cells of the first scored-path run while the lever was demonstrably firing:
    on r11l the resolved mask goes None -> 64 cells (`hud_mask_source=
    edge_bar_detector_req5960_stage2_confirmed`, Stage 2 `admitted`) and states_expanded goes
    319 -> 41; on tn36 None -> 61 cells and 49 -> 17.

    THE CORRECT PREDICATE. A mask APPEARING where the shipped configuration had none is the lever's
    strongest possible firing, not a non-event, so a falsy shipped digest is treated as a real
    difference. Digests are compared, never cell COUNTS -- the 2026-07-25 gate compared counts and
    therefore read a same-size different mask as "no change".

    A MISSING `hud_shipped_mask_digest` KEY IS *UNKNOWN*, NOT None. `digest != None` is true for
    every resolved mask, so treating an unrecorded shipped digest as None would arithmetically FORCE
    "fired" on every resolved cell -- 1058 of the 1713 recorded cptb rows, none of which ever made a
    shipped-side comparison. Such a row returns False here, and `lever2_scoreable` is what tells a
    reader it carries NO evidence rather than negative evidence.
    """
    h = hud or {}
    if h.get("error") or h.get("hud_diagnostics_error"):
        return False
    if not h.get("hud_mask_resolved"):
        return False
    digest = h.get("hud_mask_digest")
    if not digest:
        return False
    if "hud_shipped_mask_digest" not in h:
        return False
    return bool(digest != h.get("hud_shipped_mask_digest"))


def lever2_scoreable(hud: dict | None) -> bool:
    """Does this row carry enough recorded state to answer the lever-2 question AT ALL?

    THE POINT. `hud_lever_fired` returns False for two structurally different reasons: the lever did
    not fire (evidence), or the row never recorded the shipped-side comparator (no evidence). A
    denominator that pools them reports "the lever fired in 0 of 1713 cells" when the honest
    statement is "1713 cells cannot say". Every rate over lever 2 must divide by the count of rows
    where THIS returns True.
    """
    h = hud or {}
    if h.get("error") or h.get("hud_diagnostics_error"):
        return False
    if h.get("hud_mask_resolved") is None:
        return False
    return "hud_shipped_mask_digest" in h


def hud_row_fields(hud: dict | None) -> dict:
    """Project a live `hud_mask_diagnostics()` dict onto FLAT row keys. THE single projection.

    Both writers call this. `hud=None` (no explorer, or a reference solver that owns its own mask)
    yields the full key set with `None` values and `hud_diagnostics_readable: False` -- an explicit
    "no evidence" row rather than a set of ABSENT keys, because a missing key is indistinguishable
    from a measured zero at read time.
    """
    h = hud or {}
    err = h.get("error")
    stage2 = h.get("stage2") or {}
    out = {
        "hud_mask_resolved": h.get("hud_mask_resolved"),
        "hud_mask_cell_count": h.get("hud_mask_cell_count"),
        # THE MASK'S IDENTITY, not its size. Two masks must be compared on this digest: equal cell
        # COUNTS do not imply the same cells, so a repair that MOVED a mask instead of widening it
        # read as inert under the previous count comparison.
        "hud_mask_digest": h.get("hud_mask_digest"),
        "hud_mask_source": h.get("hud_mask_source"),
        # Stage 2's verdict: admitted / refused / discarded / pending / no_candidate. A `refused`
        # row is the safety mechanism WORKING (the candidate was never applied) and is reported
        # distinctly from "the detector found nothing".
        "hud_mask_stage2_verdict": stage2.get("stage2_verdict"),
        "hud_mask_stage2_reason": stage2.get("stage2_reason"),
        "hud_mask_stage2_candidate_cells": stage2.get("candidate_cell_count"),
        "unique_frames": h.get("unique_frames"),
        "graph_nodes": h.get("graph_nodes"),
        # graph_nodes / distinct UNMASKED frames. 1.0 = every distinct raw frame became its own node
        # (no dedup at all -- the measured r11l pathology).
        "node_inflation_vs_unique_frames": h.get("node_inflation_vs_unique_frames"),
        # THE GUARD'S ACTIVITY WITNESS. >0 means the guard PROVED the mask was collapsing distinct
        # states and un-masked those nodes.
        "collapse_guard_refusals": h.get("collapse_guard_refusals"),
        "hud_shipped_mask_cell_count": h.get("hud_shipped_mask_cell_count"),
        "hud_shipped_mask_digest": h.get("hud_shipped_mask_digest"),
        # True only if `hud_mask_diagnostics()` was actually called AND returned a real payload. A
        # row with `hud_diagnostics_readable: False` carries NO evidence about lever 2 in either
        # direction and must be excluded from a denominator, never counted as a non-fire.
        "hud_diagnostics_readable": bool(err is None and h.get("hud_mask_resolved") is not None),
        "hud_diagnostics_error": err,
        "lever2_fired": hud_lever_fired(hud),
        "lever2_fired_predicate": LEVER2_FIRE_PREDICATE_VERSION,
    }
    # Deprecated short alias, emitted for exp5836's own three readers. Removing it is a separate
    # commit; keeping it here means the unification does not have to break them in the same change.
    out[NODE_INFLATION_ALIAS] = out["node_inflation_vs_unique_frames"]
    return out


# The subset of `HUD_ROW_KEYS` that can be lifted out of a NESTED `lever2_hud_fire` dict verbatim.
# The stage-2 keys are excluded because they live one level deeper in the nested spelling.
_NESTED_LIFTABLE = tuple(
    k
    for k in HUD_ROW_KEYS
    if not k.startswith("hud_mask_stage2_")
    and k not in ("hud_diagnostics_readable", "hud_diagnostics_error", "lever2_fired_predicate")
)


def backfill_hud_flat_fields(r: dict) -> str:
    """Make an already-recorded row readable at BOTH addresses, and REPORT which schema it arrived
    in. THE single chokepoint every consumer must call when loading rows.

    Returns one of:
      `both`        -- the row already carried both addresses (written after the unification).
      `nested_only` -- a pre-unification scored-path row; the flat keys were back-filled from the
                       nested dict and `hud_flat_fields_backfilled_from_nested` is stamped True.
      `flat_only`   -- an exp5836 / cptb-schema row; left as-is apart from the node-inflation alias.
      `absent`      -- no HUD diagnostics at all. Carries no lever-2 evidence in either direction.

    Returning the tag rather than silently transforming is deliberate: a silent back-fill is one
    more unwitnessed step between measurement and claim, and the artifact must be able to state how
    many rows needed it.
    """
    nested = r.get("lever2_hud_fire")
    has_nested = isinstance(nested, dict) and bool(nested)
    has_flat = "hud_mask_resolved" in r
    tag = "absent"
    if has_nested and has_flat:
        tag = "both"
    elif has_nested:
        for k in _NESTED_LIFTABLE:
            if k in nested:
                r[k] = nested[k]
        stage2 = nested.get("stage2") or {}
        r.setdefault("hud_mask_stage2_verdict", stage2.get("stage2_verdict"))
        r.setdefault("hud_mask_stage2_reason", stage2.get("stage2_reason"))
        r.setdefault("hud_mask_stage2_candidate_cells", stage2.get("candidate_cell_count"))
        r["hud_flat_fields_backfilled_from_nested"] = True
        tag = "nested_only"
    elif has_flat:
        tag = "flat_only"

    # THE NAME COLLISION, both directions. exp5836 wrote `node_inflation`; everything else writes
    # `node_inflation_vs_unique_frames`. A reader of either name must find the value regardless of
    # which writer produced the row, or the SAME quantity reads None on half the corpus -- the
    # identical defect this module exists to close, one field deeper.
    if r.get("node_inflation_vs_unique_frames") is None and r.get(NODE_INFLATION_ALIAS) is not None:
        r["node_inflation_vs_unique_frames"] = r[NODE_INFLATION_ALIAS]
    if r.get(NODE_INFLATION_ALIAS) is None and r.get("node_inflation_vs_unique_frames") is not None:
        r[NODE_INFLATION_ALIAS] = r["node_inflation_vs_unique_frames"]

    # THE U6 GUARD. Stamped on the row so no consumer has to re-derive it, and so a lever-2 rate can
    # never silently divide by rows that were structurally unable to answer.
    r["_hud_row_schema"] = tag
    r["_lever2_scoreable"] = lever2_scoreable(nested if isinstance(nested, dict) and nested else r)
    return tag
