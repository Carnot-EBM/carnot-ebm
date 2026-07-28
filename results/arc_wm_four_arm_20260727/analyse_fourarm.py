#!/usr/bin/env python
"""Analyser for the REQ-ARC-WMTE-6010/-6011/-6013 four-arm live matrix.

WHAT THIS READS AND WHAT IT DOES NOT
------------------------------------
It reads the per-cell row files this run wrote under `cells/`. It does NOT re-run the
agent, re-score an engine, or recompute any gate: every quantity below is a projection of
a field the LIVE agent wrote onto its own attempt dict at decision time. That distinction
is load-bearing for this project -- the canonical measurement failure here is two
independent reimplementations of a wrong formula agreeing 44/44 with each other and both
being wrong about the system. So the analyser reads the real object and never models it.

Consequently this analyser declares `inference_substrate:
aggregation_from_upstream_artifacts` and publishes `measurement_wall_s` from the SUM OF
EACH ROW FILE'S OWN `elapsed_s`, never from its own wall clock. The analyser clock is not
the measurement clock.

THE FOUR ARMS AND WHY FOUR
--------------------------
  A0 control    mask=0 gate=0
  A1 mask-only  mask=1 gate=0
  A2 gate-only  mask=0 gate=1
  A3 both       mask=1 gate=1

The two fixes push in OPPOSITE directions. Masking the HUD can only DELETE cells from the
exact-match comparison, so it can only RAISE a measured score and can only ADMIT more
engines. The change gate can only REJECT engines, so it can only LOWER the admission rate.
A two-arm before/after therefore cannot distinguish "both worked and cancelled" from
"neither did" -- and the offline prior says the cancellation is near-exact. Every arm is
reported against the control SEPARATELY, and the interaction is stated explicitly rather
than left implicit in a combined-arm number.

PAIRING AND TESTING
-------------------
Unit of pairing is the (game, variant) cell, matched across all four arms -- never an
any-seed union, which would let an arm win on cells its control never ran. The test is the
exact two-sided sign test on DISCORDANT pairs (concordant pairs carry no information about
direction). Both tails are reported, and so is the MINIMUM REACHABLE p at the achieved
discordant count: with n discordant pairs the smallest two-sided p obtainable is 2^(1-n),
so at n=4 no result can reach p<0.05 no matter how lopsided it is. Reporting that bound
alongside the p-value is what stops an underpowered null from being read as evidence of
absence.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
OUT = REPO / "results" / "arc_wm_four_arm_20260727"
CELLS = OUT / "cells"

from carnot.agentic.arc_world_model_trust_energy import (  # noqa: E402
    HIDDEN_STATE_GAME_IDS,
)

ARMS = ["wm_A0_control", "wm_A1_mask", "wm_A2_gate", "wm_A3_both"]
CONTROL = "wm_A0_control"
TREATMENTS = ["wm_A1_mask", "wm_A2_gate", "wm_A3_both"]

# What each arm is DECLARED to set. Checked against what the run file says the resolver
# actually returned, so a declared arm that did not take is a hard failure, not a silent
# null. This is the declared-vs-actual gap the whole measurement exists to close.
ARM_DECLARED = {
    "wm_A0_control": {"hud_mask": False, "change_gate": False},
    "wm_A1_mask": {"hud_mask": True, "change_gate": False},
    "wm_A2_gate": {"hud_mask": False, "change_gate": True},
    "wm_A3_both": {"hud_mask": True, "change_gate": True},
}


# ------------------------------------------------------------------ statistics


def sign_test(pairs: list[tuple[float, float]]) -> dict:
    """Exact two-sided sign test on discordant pairs, with the power bound attached.

    `pairs` is [(control_value, treatment_value)]. Concordant pairs (exact ties) are
    EXCLUDED, which is the standard sign test and also the honest thing here: a cell where
    the arm changed nothing carries no evidence about direction, and counting ties as
    support would let a structurally-inert arm manufacture significance from cells it
    provably could not have touched.
    """

    up = sum(1 for a, b in pairs if b > a)
    down = sum(1 for a, b in pairs if b < a)
    n = up + down
    if n == 0:
        return {
            "n_pairs_total": len(pairs),
            "n_discordant": 0,
            "n_up": 0,
            "n_down": 0,
            "p_two_sided": None,
            "p_one_sided_greater": None,
            "p_one_sided_less": None,
            "min_reachable_two_sided_p": None,
            "note": "zero discordant pairs -- NO test is possible; this is not a null result",
        }

    def _binom_tail_ge(k: int, n_: int) -> float:
        from math import comb

        return sum(comb(n_, i) for i in range(k, n_ + 1)) / (2.0**n_)

    p_greater = _binom_tail_ge(up, n)
    p_less = _binom_tail_ge(down, n)
    p_two = min(1.0, 2.0 * min(p_greater, p_less))
    return {
        "n_pairs_total": len(pairs),
        "n_discordant": n,
        "n_up": up,
        "n_down": down,
        "p_two_sided": round(p_two, 6),
        "p_one_sided_greater": round(p_greater, 6),
        "p_one_sided_less": round(p_less, 6),
        # 2^(1-n): the smallest two-sided p an n-discordant sign test can ever produce.
        # If this is above 0.05 the arm is UNDERPOWERED BY CONSTRUCTION and a non-significant
        # result says nothing about the effect.
        "min_reachable_two_sided_p": round(2.0 ** (1 - n), 6),
        "underpowered_by_construction": bool(2.0 ** (1 - n) > 0.05),
    }


def dist(vals: list[float]) -> dict:
    """Distribution summary. `n_exactly_zero` and the full sorted list are included because
    the headline question for the mask arm is whether the trust distribution MOVES OFF THE
    FLOOR -- a mean or a median alone cannot answer that, and the baseline's median was
    0.0 with a non-zero mean."""
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return {"n": 0, "empty": True}
    return {
        "n": len(v),
        "empty": False,
        "min": round(v[0], 6),
        "median": round(statistics.median(v), 6),
        "max": round(v[-1], 6),
        "mean": round(statistics.fmean(v), 6),
        "n_exactly_zero": sum(1 for x in v if x == 0.0),
        "n_at_or_above_0p5": sum(1 for x in v if x >= 0.5),
        "all_values_sorted": [round(x, 6) for x in v],
    }


# ------------------------------------------------------------------ cell reading


def read_cells() -> dict:
    """arm -> variant_signature -> compact per-cell record."""
    out: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    for f in sorted(CELLS.glob("*.json")):
        d = json.loads(f.read_text())
        arm = d.get("arm")
        if arm not in out:
            continue
        w = d.get("liveness_witness") or {}
        diags = w.get("induction_attempt_gate_diagnostics") or []
        # The FIRST induction attempt is the primary unit: it is the attempt whose gate
        # decision determines whether a plan is installed at all. Later attempts are kept
        # in `n_attempts` so a cell that retried is distinguishable from one that did not.
        d0 = diags[0] if diags else {}
        game = str(d.get("game"))
        out[arm][str(d.get("variant_signature"))] = {
            "game": game,
            "branch": "hidden_state" if game in HIDDEN_STATE_GAME_IDS else "plain",
            "elapsed_s": float(d.get("elapsed_s") or 0.0),
            "cell_error": d.get("cell_error") or "",
            "first_win": d.get("first_win"),
            "reached_level": d.get("reached_level"),
            "actions": d.get("actions"),
            "n_attempts": int(w.get("induction_attempts_n") or 0),
            "planned": int(w.get("induction_attempts_planned") or 0),
            "skipped_reasons": list(w.get("induction_attempts_skipped") or []),
            "llm_calls": (w.get("llm") or {}).get("calls"),
            "llm_responses": (w.get("llm") or {}).get("responses"),
            "llm_errors": (w.get("llm") or {}).get("errors"),
            # --- the per-attempt witnesses (REQ-6014) ---
            "skipped": d0.get("skipped"),
            "verify_accuracy": d0.get("verify_accuracy"),
            "verify_cell_recall": d0.get("verify_cell_recall"),
            "heldout_accuracy": d0.get("heldout_accuracy"),
            "heldout_change_consistency": d0.get("heldout_change_consistency"),
            "correct_changed_cells": d0.get("correct_changed_cells"),
            "binary_gate_pass": d0.get("binary_gate_pass"),
            "change_fidelity": d0.get("verify_change_fidelity"),
            "spurious_changed_cells": d0.get("verify_spurious_changed_cells"),
            "hud_mask_status": d0.get("hud_mask_status"),
            "hud_mask_cells": d0.get("hud_mask_cells"),
            "hud_mask_reason": d0.get("hud_mask_reason"),
            "change_gate_hidden_state_enabled": d0.get("change_gate_hidden_state_enabled"),
            "n_diags": len(diags),
        }
    return out


def gated_quantity(rec: dict):
    """The quantity the INCUMBENT gate reads on this cell's branch.

    Plain branch: `verify_accuracy` -- full-grid exact match, thresholded at 0.5.
    Hidden-state branch: `heldout_accuracy` -- the quantity behind `binary_gate_pass`.

    These are branch-specific on purpose. Pooling them into one column would compare an
    exact-match rate against a held-out accuracy as if they were the same measurement.
    """
    return rec["verify_accuracy"] if rec["branch"] == "plain" else rec["heldout_accuracy"]


# ------------------------------------------------------------------ analysis


def analyse() -> dict:
    t_an0 = time.time()
    cells = read_cells()
    sigs_all = sorted(set().union(*[set(cells[a]) for a in ARMS]) if any(cells.values()) else [])
    # MATCHED support only: a signature missing from any arm is excluded from every paired
    # test and reported separately. An any-seed union would let an arm be scored on cells
    # its control never ran.
    matched = [s for s in sigs_all if all(s in cells[a] for a in ARMS)]
    unmatched = [s for s in sigs_all if s not in matched]

    runs = {}
    for a in ARMS:
        rf = OUT / f"run_{a}.json"
        runs[a] = json.loads(rf.read_text()) if rf.exists() else None

    # ---- arm integrity: did the declared arm actually take? -------------------
    arm_integrity = {}
    for a in ARMS:
        r = runs[a] or {}
        resolved = r.get("resolved_flags") or {}
        decl = ARM_DECLARED[a]
        arm_integrity[a] = {
            "declared": decl,
            "resolved": resolved,
            "resolver_matches_declaration": bool(
                resolved
                and resolved.get("hud_mask") == decl["hud_mask"]
                and resolved.get("change_gate") == decl["change_gate"]
            ),
            # REQ-6013's follow-the-6011 default, OBSERVED. `None` means the run file
            # predates the field, which is itself reportable -- not silently "fine".
            "change_gate_hidden_state_resolved": resolved.get("change_gate_hidden_state"),
            "follow_default_took": (
                None
                if not resolved
                else resolved.get("change_gate_hidden_state") == decl["change_gate"]
            ),
            "shipped_defaults_unchanged": r.get("shipped_defaults_unchanged"),
            "server_device": ((r.get("server") or {}).get("device") or {}).get("verdict"),
            "server_vram_mib": ((r.get("server") or {}).get("device") or {}).get("my_vram_mib"),
            "vram_summary": r.get("vram_summary"),
            "measurement_wall_s_from_rows": r.get("measurement_wall_s_from_rows"),
        }

    # ---- per-arm distributions ------------------------------------------------
    per_arm = {}
    for a in ARMS:
        recs = [cells[a][s] for s in matched]
        plain = [r for r in recs if r["branch"] == "plain"]
        hidden = [r for r in recs if r["branch"] == "hidden_state"]
        per_arm[a] = {
            "n_cells": len(recs),
            "n_plain": len(plain),
            "n_hidden_state": len(hidden),
            "n_cell_errors": sum(1 for r in recs if r["cell_error"]),
            "n_first_win": sum(1 for r in recs if r["first_win"]),
            "first_win_cells": sorted(s for s in matched if cells[a][s]["first_win"]),
            "n_planned_gt_0": sum(1 for r in recs if r["planned"] > 0),
            "planned_cells": sorted(s for s in matched if cells[a][s]["planned"] > 0),
            "n_attempts_total": sum(r["n_attempts"] for r in recs),
            "llm_calls_total": sum(int(r["llm_calls"] or 0) for r in recs),
            "llm_responses_total": sum(int(r["llm_responses"] or 0) for r in recs),
            "llm_errors_total": sum(int(r["llm_errors"] or 0) for r in recs),
            # THE HEADLINE DISTRIBUTION for the mask arm.
            "gated_quantity_plain_verify_accuracy": dist(
                [r["verify_accuracy"] for r in plain if r["verify_accuracy"] is not None]
            ),
            "gated_quantity_hidden_heldout_accuracy": dist(
                [r["heldout_accuracy"] for r in hidden if r["heldout_accuracy"] is not None]
            ),
            "change_fidelity_all": dist(
                [r["change_fidelity"] for r in recs if r["change_fidelity"] is not None]
            ),
            "hud_mask_status_census": _census(r["hud_mask_status"] for r in recs),
            "skip_reason_census": _census(r["skipped"] for r in recs),
            "n_mask_applied": sum(
                1
                for r in recs
                if r["hud_mask_status"] == "applied" and int(r["hud_mask_cells"] or 0) > 0
            ),
        }

    # ---- paired arm-vs-control comparisons -----------------------------------
    comparisons = {}
    for a in TREATMENTS:
        comparisons[a] = _compare(cells, matched, a)

    # ---- the interaction, stated explicitly ----------------------------------
    interaction = _interaction(per_arm, comparisons)

    # ---- witnesses + acceptance ----------------------------------------------
    witnesses = _witnesses(cells, matched, per_arm, arm_integrity)

    measurement_wall_s = round(sum(cells[a][s]["elapsed_s"] for a in ARMS for s in cells[a]), 3)

    art = {
        "matched_signatures": matched,
        "unmatched_signatures": unmatched,
        "n_matched_cells_per_arm": len(matched),
        "arm_integrity": arm_integrity,
        "per_arm": per_arm,
        "comparisons_vs_control": comparisons,
        "interaction": interaction,
        "witnesses": witnesses,
        "measurement_wall_s": measurement_wall_s,
        "analyser_duration_s": round(time.time() - t_an0, 4),
    }
    return art


def _census(vals) -> dict:
    out: dict[str, int] = {}
    for v in vals:
        k = "(absent)" if v is None else str(v)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _compare(cells: dict, matched: list[str], arm: str) -> dict:
    ctrl = cells[CONTROL]
    trt = cells[arm]

    q_pairs, q_sigs = [], []
    for s in matched:
        a, b = gated_quantity(ctrl[s]), gated_quantity(trt[s])
        if a is not None and b is not None:
            q_pairs.append((float(a), float(b)))
            q_sigs.append(s)

    # Admission (planned>0) is the DECISION-level endpoint; the gated quantity is the
    # score-level one. They are reported separately because the mask can move the score
    # without crossing the 0.5 threshold, and that distinction is the whole question.
    adm_ctrl = {s for s in matched if ctrl[s]["planned"] > 0}
    adm_trt = {s for s in matched if trt[s]["planned"] > 0}
    win_ctrl = {s for s in matched if ctrl[s]["first_win"]}
    win_trt = {s for s in matched if trt[s]["first_win"]}

    # Bit-identity check: how many cells did this arm leave EXACTLY unchanged? The
    # 2026-07-27 first-win measurement's fatal finding was that every LLM-on arm was
    # bit-identical to its control on 74/74 cells, which made p=1.0 an arithmetic identity
    # rather than a measurement. Reporting the identity count up front makes that failure
    # mode impossible to miss again.
    identical = [
        s
        for s in matched
        if ctrl[s]["first_win"] == trt[s]["first_win"]
        and ctrl[s]["actions"] == trt[s]["actions"]
        and ctrl[s]["reached_level"] == trt[s]["reached_level"]
        and ctrl[s]["planned"] == trt[s]["planned"]
        and ctrl[s]["skipped"] == trt[s]["skipped"]
    ]

    moved = [
        {
            "sig": s,
            "game": ctrl[s]["game"],
            "branch": ctrl[s]["branch"],
            "control": gated_quantity(ctrl[s]),
            "treatment": gated_quantity(trt[s]),
            "control_skip": ctrl[s]["skipped"],
            "treatment_skip": trt[s]["skipped"],
            "treatment_hud_mask_status": trt[s]["hud_mask_status"],
            "treatment_hud_mask_cells": trt[s]["hud_mask_cells"],
            "control_change_fidelity": ctrl[s]["change_fidelity"],
            "treatment_change_fidelity": trt[s]["change_fidelity"],
        }
        for s in q_sigs
        if gated_quantity(ctrl[s]) != gated_quantity(trt[s])
    ]

    return {
        "gated_quantity_sign_test": sign_test(q_pairs),
        "gated_quantity_control_dist": dist([a for a, _ in q_pairs]),
        "gated_quantity_treatment_dist": dist([b for _, b in q_pairs]),
        "cells_whose_gated_quantity_moved": moved,
        "n_moved": len(moved),
        "admission": {
            "n_control": len(adm_ctrl),
            "n_treatment": len(adm_trt),
            "admitted_only_by_treatment": sorted(adm_trt - adm_ctrl),
            "admitted_only_by_control": sorted(adm_ctrl - adm_trt),
            "mcnemar_sign_test": sign_test(
                [(1.0, 0.0) for _ in (adm_ctrl - adm_trt)]
                + [(0.0, 1.0) for _ in (adm_trt - adm_ctrl)]
            ),
        },
        "first_win": {
            "n_control": len(win_ctrl),
            "n_treatment": len(win_trt),
            "lost_vs_control": sorted(win_ctrl - win_trt),
            "gained_vs_control": sorted(win_trt - win_ctrl),
            "regression_clause_holds": not (win_ctrl - win_trt),
        },
        "n_bit_identical_to_control": len(identical),
        "bit_identical_fraction": (round(len(identical) / len(matched), 4) if matched else None),
    }


def _interaction(per_arm: dict, comparisons: dict) -> dict:
    """State the interaction explicitly rather than leaving it implicit in A3.

    The two fixes push in opposite directions, so A3's delta is NOT the sum of A1's and
    A2's unless they are independent. Reporting the additive prediction next to the
    observed A3 is what makes "both worked and cancelled" distinguishable from "neither
    did" -- if A1 admits +k, A2 removes -m, and A3 lands at neither the sum nor either
    single-arm value, the fixes interact and neither single arm generalises.
    """
    a0 = per_arm[CONTROL]["n_planned_gt_0"]
    a1 = per_arm["wm_A1_mask"]["n_planned_gt_0"]
    a2 = per_arm["wm_A2_gate"]["n_planned_gt_0"]
    a3 = per_arm["wm_A3_both"]["n_planned_gt_0"]
    additive = a0 + (a1 - a0) + (a2 - a0)
    return {
        "endpoint": "n_cells_with_induction_attempts_planned_gt_0 (the admission decision)",
        "A0_control": a0,
        "A1_mask_only": a1,
        "A2_gate_only": a2,
        "A3_both_observed": a3,
        "A3_additive_prediction": additive,
        "delta_mask": a1 - a0,
        "delta_gate": a2 - a0,
        "A3_minus_additive_prediction": a3 - additive,
        "fixes_are_additive_on_this_endpoint": a3 == additive,
        # REQ-ARC-WMTE-6019: AN EMPTY PASS REGION, named instead of reported as a clean
        # False. The test needs `delta_gate < 0` -- the gate arm admitting FEWER cells than
        # control. The endpoint is a COUNT of admitting cells, so it is floored at 0; when
        # the control admits 0, `delta_gate` cannot be negative at ANY outcome and the flag
        # is unreachable-False rather than measured-False. That is exactly this run's shape
        # (A0_control = 0), so a reader taking `cancellation_detected: false` as evidence
        # would be reading a gate that could not have fired -- and it read as clean while
        # the artifact's own prose described the gate rejecting precisely the engine the
        # mask admitted on lf52, which IS a per-cell cancellation.
        #
        # Reported as three-valued: None when the arithmetic cannot express the condition,
        # with the measurability flag and the reason beside it, so an absent measurement is
        # never a negative finding. Same discipline as `noop_ok_is_vacuous` in
        # `change_gate_decision` and `hud_mask_swallow_clean`'s affirmative-`ok` rule.
        "cancellation_measurable": bool(a0 > 0),
        "cancellation_detected": (
            bool((a1 - a0) > 0 and (a2 - a0) < 0 and a3 == a0) if a0 > 0 else None
        ),
        "cancellation_unmeasurable_reason": (
            ""
            if a0 > 0
            else (
                "requires delta_gate < 0, but the endpoint is a count floored at 0 and "
                f"A0_control == {a0}, so delta_gate >= 0 at every reachable outcome; the "
                "flag is unreachable-False, not measured-False"
            )
        ),
        "why_this_matters": (
            "the mask can only ADMIT more and the gate can only REJECT more; if A3 returns "
            "to the control value while A1 and A2 both moved, a two-arm before/after would "
            "have reported a clean null that is in fact two real effects cancelling"
        ),
        "cancellation_detected_is_aggregate_not_per_cell": (
            "this flag is an ARM-LEVEL count comparison. A PER-CELL cancellation is a "
            "different and weaker claim, and this run has one: on lf52 the mask admitted an "
            "engine and the gate rejected it, so A1 moved and A3 did not. The arm-level flag "
            "cannot express that, which is why the per-cell rejection detail is reported "
            "separately rather than inferred from this field"
        ),
    }


def _witnesses(cells: dict, matched: list[str], per_arm: dict, integrity: dict) -> dict:
    """Computed witnesses AT EACH GATE'S OWN AGGREGATION LEVEL.

    A gate that reports a PASS which could not have failed is not evidence. Each witness
    below is a computed quantity whose failing value is reachable from this data.
    """
    # (1) Every cell the mask arm claims to have masked must carry the proof.
    mask_claims = []
    for arm in ("wm_A1_mask", "wm_A3_both"):
        for s in matched:
            r = cells[arm][s]
            if r["hud_mask_status"] == "applied":
                mask_claims.append(
                    {
                        "arm": arm,
                        "sig": s,
                        "hud_mask_cells": r["hud_mask_cells"],
                        "proof_ok": int(r["hud_mask_cells"] or 0) > 0,
                    }
                )
    # (2) THE EQUALITY CONTROL: on cells where no mask resolved, the mask arm MUST be
    # identical to the control. A difference there means the arm is changing something
    # other than the mask, and every mask-arm number would be uninterpretable.
    inert_violations = []
    n_inert_checked = 0
    for arm in ("wm_A1_mask",):
        for s in matched:
            c, t = cells[CONTROL][s], cells[arm][s]
            if t["hud_mask_status"] != "applied":
                n_inert_checked += 1
            if t["hud_mask_status"] != "applied" and gated_quantity(c) != gated_quantity(t):
                inert_violations.append(
                    {
                        "arm": arm,
                        "sig": s,
                        "hud_mask_status": t["hud_mask_status"],
                        "control": gated_quantity(c),
                        "treatment": gated_quantity(t),
                    }
                )
    # (3) The gate arm must reject DEGENERATES specifically, not everything. A rejection
    # whose own change_fidelity is high would be a false reject.
    gate_rejections = []
    for arm in ("wm_A2_gate", "wm_A3_both"):
        for s in matched:
            t = cells[arm][s]
            sk = str(t["skipped"] or "")
            if "change_gate" in sk:
                gate_rejections.append(
                    {
                        "arm": arm,
                        "sig": s,
                        "branch": t["branch"],
                        "reason": sk,
                        "change_fidelity": t["change_fidelity"],
                        "spurious_changed_cells": t["spurious_changed_cells"],
                        "correct_changed_cells": t["correct_changed_cells"],
                        "control_would_have_admitted": cells[CONTROL][s]["planned"] > 0,
                        "control_incumbent_binary_gate_pass": cells[CONTROL][s]["binary_gate_pass"],
                        # A rejection is JUSTIFIED when the engine really did get ~none of
                        # the change region right. This is the field that makes a
                        # false-reject visible instead of assumed away.
                        "rejection_justified_by_low_fidelity": (
                            t["change_fidelity"] is not None and float(t["change_fidelity"]) < 0.5
                        ),
                    }
                )
    return {
        "mask_application_proof": {
            "n_claims": len(mask_claims),
            "n_with_positive_cells": sum(1 for m in mask_claims if m["proof_ok"]),
            "all_claims_proved": all(m["proof_ok"] for m in mask_claims),
            "detail": mask_claims,
        },
        "mask_inert_equality_control": {
            "n_violations": len(inert_violations),
            # The SUPPORT for this control. `all([])` is True, so a control checked over an
            # empty set passes without constraining anything -- a pass that could not have
            # failed. Reporting the count is what makes that visible.
            "n_cells_checked": n_inert_checked,
            "holds": not inert_violations,
            "detail": inert_violations,
            "principle": (
                "on cells where no mask resolved, the mask arm must be identical to the "
                "control; a difference means the arm changes something other than the mask"
            ),
        },
        "gate_rejections": {
            "n": len(gate_rejections),
            "n_justified_low_fidelity": sum(
                1 for g in gate_rejections if g["rejection_justified_by_low_fidelity"]
            ),
            "n_where_incumbent_would_have_passed": sum(
                1 for g in gate_rejections if g["control_incumbent_binary_gate_pass"] is True
            ),
            "detail": gate_rejections,
        },
        "generator_liveness": {
            arm: {
                "llm_calls_total": per_arm[arm]["llm_calls_total"],
                "llm_responses_total": per_arm[arm]["llm_responses_total"],
                "llm_errors_total": per_arm[arm]["llm_errors_total"],
                # A dead generator would make every arm an LLM-off control and every
                # comparison an identity. This is the channel that proves otherwise.
                "generator_answered": per_arm[arm]["llm_responses_total"] > 0,
            }
            for arm in ARMS
        },
        "device_residency": {
            arm: {
                "verdict": integrity[arm]["server_device"],
                "vram_mib_at_launch": integrity[arm]["server_vram_mib"],
                "vram_summary": integrity[arm]["vram_summary"],
            }
            for arm in ARMS
        },
    }


if __name__ == "__main__":
    a = analyse()
    (OUT / "analysis_fourarm.json").write_text(json.dumps(a, indent=1, default=str))
    a["sha256_of_analysis"] = hashlib.sha256(
        json.dumps(a, sort_keys=True, default=str).encode()
    ).hexdigest()
    print(
        json.dumps({k: v for k, v in a.items() if k != "witnesses"}, indent=1, default=str)[:6000]
    )
    print("\nwrote", OUT / "analysis_fourarm.json")
