#!/usr/bin/env python
"""Build results/experiment_6015_wm_hud_mask_change_gate_four_arm_live.json.

WHY THIS FILE EXISTS SEPARATELY FROM `build_artifact.py`
-------------------------------------------------------
`build_artifact.py` was written at 16:23 local for the run that was later DISCARDED for
engine-store contamination (see `cells_PRE_ISOLATION_FIX_DISCARDED/WHY_DISCARDED.md`), and
line 36 of it overwrites the TRACKED `analysis_fourarm.json`, which still holds that
discarded run's analysis. Overwriting it would destroy a historical record, which the
never-prune rule forbids. So this builder is additive: it writes `analysis_fourarm_live.json`
and leaves every pre-existing file alone.

It reuses `analyse_fourarm.py` rather than reimplementing its statistics. That is deliberate:
this project's canonical measurement failure is two independent reimplementations of one wrong
formula agreeing 44/44 with each other and both being wrong about the system. Agreement
between reconstructions is not evidence, so the analyser is IMPORTED, not rebuilt.

WHAT THIS BUILDER ADDS ON TOP OF THE IMPORTED ANALYSER
------------------------------------------------------
The imported analyser was designed before this run existed. Reading its output against the
real 100 cells surfaced five things it cannot express, each of which is a finding rather than
a formatting gap. All five are computed here from `A.read_cells()` -- the same projection of
the same row files -- and never modelled:

  1. THE WIN ENDPOINT. The analyser reports win SETS but no win-level test. The headline of
     this run is a win-level null, so the test (and, load-bearing, its MINIMUM REACHABLE p)
     has to be computed at that endpoint explicitly.

  2. PER-CELL ARM-TREATMENT EVIDENCE. `run_wm_A2_gate.json` does not exist, so the analyser's
     resolver read-back for the gate-only arm reads as `{}` -> `resolver_matches_declaration:
     False`. That is an UNMEASURABLE, not a failure: the cells themselves carry
     `hud_mask_reason` and `change_gate_hidden_state_enabled`, which prove what each arm's
     flags actually were. Both are reported -- the rollup gap as a gap, the per-cell evidence
     as evidence -- because an unmeasurable read as clean is this project's recurring bug and
     an unmeasurable read as BROKEN is the same error with the sign flipped.

  3. THE MASK-DECISION RECORD IS TWO DEAD CHANNELS. `hud_mask_swallow` is None in 104 of 104
     diagnostics: the swallow guard's auditable record is computed and discarded.
     `hud_mask_status` is absent on every hidden-state cell in every arm, so on the 11
     hidden-state games the mask decision is unrecoverable from the record even though
     `hud_mask_reason` shows the mask RESOLVED on 8 of them.

  4. THE INERT-EQUALITY CONTROL IS MIS-SCOPED, AND FAILS ANYWAY. The analyser treats
     `hud_mask_status != "applied"` as "no mask resolved", which silently swallows the 11
     hidden-state cells where the status key is simply absent. Re-scoped to cells where the
     mask PROVABLY did not apply, the control still fails -- which is the load-bearing
     methodological finding of this run, because it means per-cell score differences are not
     attributable to the mask.

  5. ONE GATE REJECTION IS UNMEASURABLE, NOT JUSTIFIED. The analyser's
     `rejection_justified_by_low_fidelity` reads `change_fidelity < 0.5`, and 0.0 is also the
     INITIALISED value on a rejection whose reason is `no_changing_transitions` -- there were
     no changing transitions to judge. Counting that as "justified by measured low fidelity"
     is the same unmeasurable-read-as-clean shape the swallow guard had, one level up, in the
     auditor. Recounted here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "results" / "arc_wm_four_arm_20260727"))
sys.path.insert(0, str(REPO / "python"))
OUT = REPO / "results" / "arc_wm_four_arm_20260727"
CELLS = OUT / "cells"

import analyse_fourarm as A  # noqa: E402, N812

EXP = 6015
ART = REPO / "results" / f"experiment_{EXP}_wm_hud_mask_change_gate_four_arm_live.json"
FIXTURES = REPO / "tests" / "fixtures" / "arc_hud_mask_swallow"

# A rejection reason that names the ABSENCE of anything to judge. `change_fidelity == 0.0` on
# such a rejection is the field's initialised value, not a measurement, so the rejection is
# UNMEASURABLE and must not be banked as "justified by low change fidelity".
_UNMEASURABLE_REJECTION_MARKERS = ("no_changing_transitions",)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------- raw diagnostic access


def _diags() -> list[tuple[str, str, dict]]:
    """(arm, variant_signature, diagnostic) for EVERY induction attempt in EVERY cell.

    The imported analyser keeps only the FIRST diagnostic per cell (`d0`), which is right for
    the admission decision but wrong for a record-coverage census: a channel that is dead on
    the second attempt is just as dead. So the census below reads every diagnostic.
    """
    rows: list[tuple[str, str, dict]] = []
    for f in sorted(CELLS.glob("*.json")):
        d = json.loads(f.read_text())
        arm = str(d.get("arm"))
        sig = str(d.get("variant_signature"))
        w = d.get("liveness_witness") or {}
        for dg in w.get("induction_attempt_gate_diagnostics") or []:
            rows.append((arm, sig, dg))
    return rows


def _first_diag_map() -> dict[tuple[str, str], dict]:
    """(arm, sig) -> the FIRST diagnostic, matching the analyser's `d0`.

    This must be the first and not the last. `A.gated_quantity` reads the analyser's `d0`, so
    any per-cell block that pairs a mask status against a gated quantity has to read the SAME
    attempt or it compares two different decisions. Concretely: vc33 has two attempts in every
    arm, and its SECOND attempt carries no mask fields at all -- a last-wins map silently
    reclassified that cell's mask status from `applied` to unrecorded. Caught by the
    applied-count disagreeing with the independent per-branch census (9 vs 10).
    """
    out: dict[tuple[str, str], dict] = {}
    for arm, sig, dg in _diags():
        out.setdefault((arm, sig), dg)
    return out


# --------------------------------------------------------------- (1) the win endpoint


def win_endpoint(a: dict, cells: dict) -> dict:
    """The headline endpoint, tested where it actually lives.

    A win is the only endpoint on this lane that cannot be inflated by a measurement change:
    the mask can raise a measured accuracy and the gate can lower an admission rate, but
    neither can manufacture a level-up. So the win endpoint is where the four-arm question is
    actually decided, and it is the one the artifact leads with.
    """
    matched = a["matched_signatures"]
    per_arm = {}
    for arm in A.ARMS:
        wins = sorted(s for s in matched if cells[arm][s]["first_win"])
        per_arm[arm] = {
            "n_wins": len(wins),
            "win_cells": wins,
            "max_reached_level": max((cells[arm][s]["reached_level"] or 0) for s in matched),
        }

    comparisons = {}
    for arm in A.TREATMENTS:
        w_c = {s for s in matched if cells[A.CONTROL][s]["first_win"]}
        w_t = {s for s in matched if cells[arm][s]["first_win"]}
        gained = sorted(w_t - w_c)
        lost = sorted(w_c - w_t)
        st = A.sign_test(
            [(1.0, 0.0) for _ in lost] + [(0.0, 1.0) for _ in gained],
        )
        comparisons[arm] = {
            "n_wins_control": len(w_c),
            "n_wins_treatment": len(w_t),
            "won_only_by_treatment": gained,
            "won_only_by_control": lost,
            "n_win_discordant_pairs": len(gained) + len(lost),
            "sign_test": st,
            # The honest phrasing, computed. With zero discordant pairs the sign test has no
            # data at all, so "no significant difference" is FALSE -- no test was performed.
            # 2^(1-0) = 2.0 is clipped to the probability ceiling of 1.0.
            "min_reachable_two_sided_p": 1.0
            if (len(gained) + len(lost)) == 0
            else st["min_reachable_two_sided_p"],
            "test_was_possible": bool(len(gained) + len(lost)),
            "correct_reading": (
                "ZERO win-discordant pairs: the minimum reachable two-sided p is 1.0, so NO "
                "test is possible at any outcome. This is an absence of discordance, NOT a "
                "measured 'no significant difference'."
                if (len(gained) + len(lost)) == 0
                else "win discordance present; see sign_test"
            ),
        }

    # Was the one win produced by an INDUCED WORLD MODEL, or by exploration alone? If every
    # arm admitted zero engines on the winning cell, the win carries no information about the
    # induce->plan path in any arm, and the four-arm comparison of that path is a comparison
    # of a path that never fired.
    all_wins = sorted({s for arm in A.ARMS for s in per_arm[arm]["win_cells"]})
    win_provenance = []
    for s in all_wins:
        win_provenance.append(
            {
                "sig": s,
                "game": cells[A.CONTROL][s]["game"],
                "won_in_arms": [arm for arm in A.ARMS if s in per_arm[arm]["win_cells"]],
                "planned_per_arm": {arm: cells[arm][s]["planned"] for arm in A.ARMS},
                "actions_per_arm": {arm: cells[arm][s]["actions"] for arm in A.ARMS},
                "reached_level_per_arm": {arm: cells[arm][s]["reached_level"] for arm in A.ARMS},
                "identical_across_all_four_arms": len(
                    {
                        (
                            cells[arm][s]["first_win"],
                            cells[arm][s]["actions"],
                            cells[arm][s]["reached_level"],
                        )
                        for arm in A.ARMS
                    }
                )
                == 1,
                "produced_by_induced_world_model_in_any_arm": any(
                    cells[arm][s]["planned"] > 0 for arm in A.ARMS
                ),
            }
        )

    return {
        "endpoint": "row.first_win (a real level-up in the live env), per matched cell",
        "per_arm": per_arm,
        "comparisons_vs_control": comparisons,
        "n_wins_identical_in_every_arm": bool(
            len({tuple(per_arm[arm]["win_cells"]) for arm in A.ARMS}) == 1
        ),
        "win_provenance": win_provenance,
        "no_win_came_from_an_induced_world_model": not any(
            wp["produced_by_induced_world_model_in_any_arm"] for wp in win_provenance
        ),
    }


# ------------------------------------------------- (2) per-cell arm-treatment evidence


def arm_treatment_evidence(cells: dict, a: dict) -> dict:
    """Prove each arm's treatment from the CELLS, independent of the run rollup.

    `hud_mask_reason` is written by the same helper the agent calls: a disabled flag yields
    `flag_disabled` and can never yield `resolved`. `change_gate_hidden_state_enabled` is the
    gate flag read back at decision time on the hidden-state branch. Together they establish
    what each arm's flags actually were WITHOUT the resolver read-back, which is what the
    missing `run_wm_A2_gate.json` costs.
    """
    matched = set(a["matched_signatures"])
    per_arm: dict[str, dict] = {}
    for arm in A.ARMS:
        reasons: dict[str, int] = {}
        gate_flags: dict[str, int] = {}
        for arm_i, sig, dg in _diags():
            if arm_i != arm or sig not in matched:
                continue
            r = str(dg.get("hud_mask_reason"))
            reasons[r] = reasons.get(r, 0) + 1
            if "change_gate_hidden_state_enabled" in dg:
                k = repr(dg["change_gate_hidden_state_enabled"])
                gate_flags[k] = gate_flags.get(k, 0) + 1
        mask_on = any(k in ("resolved", "explorer_mask_unresolved") for k in reasons)
        mask_off = reasons.get("flag_disabled", 0) > 0
        gate_on = gate_flags.get("True", 0) > 0
        gate_off = gate_flags.get("False", 0) > 0
        decl = A.ARM_DECLARED[arm]
        per_arm[arm] = {
            "declared": decl,
            "hud_mask_reason_census": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "change_gate_hidden_state_enabled_census": gate_flags,
            "mask_evidenced_on": mask_on,
            "mask_evidenced_off": mask_off,
            "gate_evidenced_on": gate_on,
            "gate_evidenced_off": gate_off,
            # A cell census that shows BOTH on and off evidence for one flag would mean the
            # arm was not homogeneous -- a reachable failing value, not a rubber stamp.
            "mask_evidence_unambiguous": mask_on != mask_off,
            "gate_evidence_unambiguous": gate_on != gate_off,
            "per_cell_evidence_matches_declaration": bool(
                (mask_on if decl["hud_mask"] else mask_off)
                and (mask_on != mask_off)
                and (gate_on if decl["change_gate"] else gate_off)
                and (gate_on != gate_off)
            ),
        }
    return per_arm


def rollup_availability() -> dict:
    """Which arms have a run rollup, and what its absence costs -- named, not implied."""
    out = {}
    for arm in A.ARMS:
        p = OUT / f"run_{arm}.json"
        out[arm] = {
            "rollup_path": f"results/arc_wm_four_arm_20260727/run_{arm}.json",
            "rollup_present": p.exists(),
            "resolver_readback_available": p.exists(),
            "device_residency_record_available": p.exists(),
        }
    missing = sorted(k for k, v in out.items() if not v["rollup_present"])
    return {
        "per_arm": out,
        "arms_missing_rollup": missing,
        "consequence": (
            "for an arm with no rollup the resolver read-back and the per-PID VRAM residency "
            "record are UNAVAILABLE -- unmeasurable, not clean and not broken. That arm's "
            "treatment is established instead from per-cell evidence (see "
            "arm_treatment_evidence), and its device residency is NOT claimed."
            if missing
            else "every arm has a rollup; resolver read-back and device residency are available"
        ),
        "corroborating_observation": (
            "the two arms that DID write rollups while a sibling arm was loading recorded that "
            "sibling's server as a foreign PID on the other card (A0 saw pid 2617566 on GPU 0, "
            "A1 saw pid 2621885 on GPU 0), so the two-chains-one-card-each layout is "
            "corroborated by a third party's per-PID record even for the arm whose own rollup "
            "is missing -- but this is corroboration of LAYOUT, not a residency claim for that "
            "arm"
        ),
    }


# ------------------------------------------------- (3) the mask-decision record channels


def mask_record_coverage(a: dict) -> dict:
    """Can the mask decision be AUDITED from the record? Two channels; both are dead."""
    matched = set(a["matched_signatures"])
    rows = [(arm, sig, dg) for arm, sig, dg in _diags() if sig in matched]

    n_diags = len(rows)
    swallow_present = sum(1 for _, _, dg in rows if dg.get("hud_mask_swallow") is not None)

    # `hud_mask_status` coverage, split by branch, because the absence is branch-structured.
    from carnot.agentic.arc_world_model_trust_energy import HIDDEN_STATE_GAME_IDS

    def _branch(sig: str) -> str:
        return "hidden_state" if sig.split("~")[0] in HIDDEN_STATE_GAME_IDS else "plain"

    per = {}
    for arm in A.ARMS:
        for br in ("plain", "hidden_state"):
            sel = [dg for a_i, s, dg in rows if a_i == arm and _branch(s) == br]
            present = [dg for dg in sel if "hud_mask_status" in dg]
            resolved_but_statusless = [
                dg
                for dg in sel
                if "hud_mask_status" not in dg and dg.get("hud_mask_reason") == "resolved"
            ]
            per[f"{arm}|{br}"] = {
                "n_diagnostics": len(sel),
                "n_with_hud_mask_status": len(present),
                "n_mask_resolved_but_status_absent": len(resolved_but_statusless),
            }

    n_resolved_statusless = sum(v["n_mask_resolved_but_status_absent"] for v in per.values())

    return {
        "hud_mask_swallow_channel": {
            "n_diagnostics": n_diags,
            "n_with_hud_mask_swallow_populated": swallow_present,
            "channel_is_dead": swallow_present == 0,
            "what_is_lost": (
                "the swallow guard's own auditable record -- swallows / changed_cell_overlap / "
                "raw vs masked changing-transition counts / reason. It is computed at decision "
                "time and discarded, so WHY a mask was applied or refused cannot be read off "
                "this run's record at all."
            ),
            "principle": (
                "a field computed and discarded is a dead channel; the decision it documents "
                "becomes unauditable after the fact, which is how a guard failure can be dated "
                "but not explained"
            ),
        },
        "hud_mask_status_channel": {
            "per_arm_branch": per,
            "n_mask_resolved_but_status_absent": n_resolved_statusless,
            "hidden_state_branch_records_no_status_in_any_arm": all(
                v["n_with_hud_mask_status"] == 0
                for k, v in per.items()
                if k.endswith("|hidden_state")
            ),
            "what_is_lost": (
                "on the 11 hidden-state games -- the games that carry the live 0.08 wall -- the "
                "mask arms show hud_mask_reason=resolved on 8 cells with NO hud_mask_status, so "
                "whether the mask was APPLIED or REFUSED there is unrecoverable from the record"
            ),
        },
        "both_channels_dead": bool(swallow_present == 0 and n_resolved_statusless > 0),
        "fixed_in_same_session": True,
        "fix_reference": (
            "the guard-fix work in this same session lifted hud_mask_swallow onto the attempt "
            "dict in BOTH branches and added it to the projection tuple, and made "
            "select_trusted_world_model report its own refusal status instead of the central "
            "null; see cited_code_fixes. Those fixes are NOT retro-applied to these cells -- "
            "this run's record stays as measured (never-prune)."
        ),
    }


# ------------------------------- (4) the inert-equality control, correctly scoped


def inert_equality_rescoped(a: dict, cells: dict) -> dict:
    """Re-scope the analyser's equality control to cells where the mask PROVABLY did not apply.

    The analyser's version tests `hud_mask_status != "applied"`, which lumps together three
    very different states: the mask provably did not apply, the mask provably DID apply, and
    the status was never recorded. The third state is 11 of 25 cells per arm, so the pooled
    control is uninterpretable in both directions. Split three ways here.

    Why this control is load-bearing: if the mask arm differs from the control on cells the
    mask never touched, then the arms differ for a reason other than the mask, and NO per-cell
    score comparison on this run can be attributed to the mask. That is a statement about
    what this run can and cannot support, so it has to be computed, not assumed.
    """
    matched = a["matched_signatures"]
    d_by = _first_diag_map()

    out = {}
    for arm in ("wm_A1_mask", "wm_A3_both"):
        provable_no_mask, applied, unknown = [], [], []
        for s in matched:
            dg = d_by.get((arm, s), {})
            status = dg.get("hud_mask_status")
            reason = dg.get("hud_mask_reason")
            if status == "applied":
                applied.append(s)
            elif status is not None or reason == "explorer_mask_unresolved":
                # A recorded non-applied status, or an unresolved mask on either branch: the
                # mask provably did not reach the comparison on this cell.
                provable_no_mask.append(s)
            else:
                unknown.append(s)

        violations = []
        for s in provable_no_mask:
            c, t = A.gated_quantity(cells[A.CONTROL][s]), A.gated_quantity(cells[arm][s])
            if c is not None and t is not None and c != t:
                violations.append(
                    {
                        "sig": s,
                        "game": cells[arm][s]["game"],
                        "branch": cells[arm][s]["branch"],
                        "hud_mask_status": d_by.get((arm, s), {}).get("hud_mask_status"),
                        "hud_mask_reason": d_by.get((arm, s), {}).get("hud_mask_reason"),
                        "control": c,
                        "treatment": t,
                        "abs_delta": round(abs(float(t) - float(c)), 6),
                    }
                )
        # INDEPENDENT PATH to the applied-count: census `status == "applied"` over EVERY
        # diagnostic rather than over the first-attempt map. The two paths must agree, and
        # they did not before the first-vs-last diagnostic bug was fixed (9 vs 10). Keeping
        # the cross-check as a recorded field is what stops that bug returning silently.
        applied_all_diags = {
            s
            for arm_i, s, dg in _diags()
            if arm_i == arm and dg.get("hud_mask_status") == "applied"
        }
        out[arm] = {
            "n_provably_no_mask": len(provable_no_mask),
            "provably_no_mask_cells": sorted(provable_no_mask),
            "n_mask_applied": len(applied),
            "n_mask_applied_cross_check_over_all_diagnostics": len(applied_all_diags),
            "applied_count_cross_check_agrees": len(applied) == len(applied_all_diags),
            "n_mask_applicability_UNKNOWN": len(unknown),
            "mask_applicability_unknown_cells": sorted(unknown),
            "n_violations_on_provably_no_mask": len(violations),
            "violations": sorted(violations, key=lambda v: -v["abs_delta"]),
            "control_holds": not violations,
            "control_has_support": len(provable_no_mask) > 0,
            "largest_violation_abs_delta": (
                max(v["abs_delta"] for v in violations) if violations else 0.0
            ),
        }

    any_violation = any(v["n_violations_on_provably_no_mask"] > 0 for v in out.values())
    return {
        "per_arm": out,
        "applied_count_cross_check_agrees_in_every_mask_arm": all(
            v["applied_count_cross_check_agrees"] for v in out.values()
        ),
        "control_holds_in_every_mask_arm": not any_violation,
        "consequence_if_it_fails": (
            "per-cell gated-quantity differences between a mask arm and the control are NOT "
            "attributable to the mask: the arms differ on cells the mask provably never "
            "touched. Any sign test on that quantity is confounded and must not be read as a "
            "mask effect."
        ),
        "confound_named": (
            "the generator is stochastic and each arm induced its own engines (per-arm LLM "
            "response counts differ: see generator_nondeterminism), so a cell's score varies "
            "arm-to-arm for reasons that have nothing to do with the treatment"
        ),
        "endpoints_NOT_affected": (
            "the win endpoint and the admission endpoint are decision-level and are reported "
            "as set differences, so they are unaffected by score-level nondeterminism; the win "
            "was additionally BIT-IDENTICAL in all four arms"
        ),
    }


def generator_nondeterminism(a: dict, cells: dict) -> dict:
    """Evidence that arms are not replicates of one another at the engine level."""
    matched = a["matched_signatures"]
    per = {}
    for arm in A.ARMS:
        per[arm] = {
            "llm_calls_total": sum(int(cells[arm][s]["llm_calls"] or 0) for s in matched),
            "llm_responses_total": sum(int(cells[arm][s]["llm_responses"] or 0) for s in matched),
            "llm_errors_total": sum(int(cells[arm][s]["llm_errors"] or 0) for s in matched),
        }
    counts = {v["llm_responses_total"] for v in per.values()}
    return {
        "per_arm": per,
        "arms_share_identical_response_counts": len(counts) == 1,
        "reading": (
            "different response counts per arm mean different engines were induced per arm; "
            "the arms are matched on the CELL, not on the engine, so score-level per-cell "
            "comparisons carry generator variance in addition to any treatment effect"
        ),
    }


# ------------------------------- (5) gate rejections, unmeasurable separated out


def gate_rejections_rescoped(a: dict) -> dict:
    """Recount the analyser's rejection justification, separating unmeasurable from justified.

    `rejection_justified_by_low_fidelity` is `change_fidelity < 0.5`. On a rejection whose
    reason is `no_changing_transitions` there was nothing to measure, and 0.0 is the field's
    initialised value -- so that rejection is UNMEASURABLE. Banking it as justified is the
    same unmeasurable-read-as-clean shape as the swallow guard's, one level up, inside the
    auditor. Recounted rather than inherited.
    """
    detail = a["witnesses"]["gate_rejections"]["detail"]
    unmeasurable, justified, unjustified = [], [], []
    for g in detail:
        reason = str(g.get("reason") or "")
        if any(m in reason for m in _UNMEASURABLE_REJECTION_MARKERS):
            unmeasurable.append(g)
        elif g.get("rejection_justified_by_low_fidelity"):
            justified.append(g)
        else:
            unjustified.append(g)

    changed_an_admission = [g for g in detail if g.get("control_would_have_admitted")]
    return {
        "n_rejections": len(detail),
        "n_justified_by_MEASURED_low_change_fidelity": len(justified),
        "n_UNMEASURABLE_nothing_to_judge": len(unmeasurable),
        "unmeasurable_detail": unmeasurable,
        "n_unjustified": len(unjustified),
        "unjustified_detail": unjustified,
        "analyser_uncorrected_n_justified": a["witnesses"]["gate_rejections"][
            "n_justified_low_fidelity"
        ],
        "why_this_recount_is_NOT_an_acceptance_gate": (
            "a gate over this partition would be unfalsifiable: the classification and any "
            "check of it both read the same rejection-reason string, so mutating the marker "
            "list moves both sides together and the gate can never fail (confirmed by mutation "
            "M2). Reported as a finding instead of banked as a passed gate."
        ),
        "n_rejections_that_changed_an_admission_decision": len(changed_an_admission),
        "gate_changed_no_admission_decision": not changed_an_admission,
        # REQ-ARC-WMTE-6019: SAY WHAT THIS COUNTERFACTUAL IS. `control_would_have_admitted`
        # is read off the CONTROL ARM's row, i.e. a DIFFERENT engine -- this artifact's own
        # `generator_nondeterminism` witness records per-arm response counts (93/91/94/89) and
        # states the arms induced different engines. So "the control also failed to admit" is
        # a cross-arm statement, not "this arm's engine would have been admitted by the
        # incumbent".
        #
        # The IN-ARM counterfactual exists in code and was simply not recorded:
        # `change_gate_decision` emits `legacy_accuracy_would_pass_at_live_threshold` on
        # every attempt, but `change_gate` was absent from the diagnostics projection, so the
        # field is absent from all 104 attempts of this run (computed and discarded -- the
        # same shape as `hud_mask_swallow`, one field over). REQ-6019 adds it to the
        # projection, so a RE-MEASUREMENT can make this claim in-arm. Until then the label
        # below is the honest scope of what is on disk.
        "admission_counterfactual_scope": "cross_arm_different_engine",
        "admission_counterfactual_caveat": (
            "control_would_have_admitted is read from the CONTROL arm's row, which induced a "
            "DIFFERENT engine (see generator_nondeterminism: per-arm LLM response counts "
            "93/91/94/89). The in-arm quantity -- "
            "legacy_accuracy_would_pass_at_live_threshold, this arm's own engine judged at the "
            "threshold the agent ships (1.0) -- is computed by change_gate_decision but was "
            "NOT projected onto any cell in this run, so it is unavailable here. REQ-6019 "
            "adds it to the projection for the next measurement; this reading is not upgraded "
            "retroactively."
        ),
        "reading": (
            "CROSS-ARM READING (see admission_counterfactual_scope): the change gate rejected "
            "engines that the incumbent gate, AS RUN IN THE CONTROL ARM ON ITS OWN ENGINE, "
            "had also failed to admit -- so the gate's measured effect on the admission "
            "endpoint is nil. Stated in-arm this would be the stronger claim, and the in-arm "
            "field is absent from this run's cells, so it is NOT claimed here."
        ),
        "one_exception_examined": (
            "sc25 is the only cell where the control's own hidden-state binary gate PASSED "
            "(heldout_accuracy 0.875, change_fidelity 0.111, 4 correct changed cells) -- and "
            "the control still did not admit it, because the trust-energy threshold rejected "
            "it. So even there the change gate did not overturn an admission; and the engine "
            "it judged was a different engine from the control's (see generator_nondeterminism)."
        ),
    }


# ------------------------------------------------------------- the lf52 finding


def lf52_finding(a: dict, cells: dict) -> dict:
    """The single plan-discordant cell, stated as the laundering hazard it is.

    IMPORTANT CORRECTION carried forward from the guard-fix work in this same session: the
    1.0000 changed-cell overlap figure for lf52 belongs to a DIFFERENT corpus (120 random
    actions offline), where nothing but the counter ever moved. On THIS run's live corpus the
    swallow guard measured overlap 0.3086 and cleared the mask with reason `ok`, and that
    verdict was CORRECT -- 56 of 81 changed cells lie outside the mask and 2 changing
    transitions survive. So the admission here is NOT explained by "the mask deleted all
    dynamics". It is explained by no-op dominance: masking turns 23 of 25 transitions into
    no-ops, and a full-grid exact-match accuracy over a 92%-no-op corpus is passed by an
    engine that predicts nothing changing.
    """
    sig = None
    for s in a["matched_signatures"]:
        if cells["wm_A1_mask"][s]["planned"] > 0 and cells[A.CONTROL][s]["planned"] == 0:
            sig = s
            break
    if sig is None:
        return {"plan_discordant_cell": None, "note": "no plan-discordant cell in this run"}

    d_by = _first_diag_map()
    table = {}
    for arm in A.ARMS:
        dg = d_by.get((arm, sig), {})
        table[arm] = {
            "planned": cells[arm][sig]["planned"],
            "skipped": cells[arm][sig]["skipped"],
            "verify_accuracy": dg.get("verify_accuracy"),
            "verify_cell_recall": dg.get("verify_cell_recall"),
            "verify_change_fidelity": dg.get("verify_change_fidelity"),
            "verify_spurious_changed_cells": dg.get("verify_spurious_changed_cells"),
            "hud_mask_status": dg.get("hud_mask_status"),
            "hud_mask_cells": dg.get("hud_mask_cells"),
            "hud_mask_reason": dg.get("hud_mask_reason"),
            "first_win": cells[arm][sig]["first_win"],
            "reached_level": cells[arm][sig]["reached_level"],
            "actions": cells[arm][sig]["actions"],
        }

    # The frozen live corpus for this exact game+cell, captured by the guard-fix work from the
    # SAME entrypoint this run used. Cited by sha256 so the arithmetic below is traceable.
    man = json.loads((FIXTURES / "MANIFEST.json").read_text())
    live = man.get("lf52_live_episode") or {}
    sc = live.get("swallow_check_at_capture") or {}
    n_t = int(sc.get("n_transitions") or 0)
    deleted = int(sc.get("changing_transitions_deleted") or 0)
    surviving = int(sc.get("masked_changing_transitions") or 0)
    noop_fraction_after_mask = round((n_t - surviving) / n_t, 6) if n_t else None

    observed_acc = table["wm_A1_mask"]["verify_accuracy"]
    _noop_pct = f"{noop_fraction_after_mask or 0.0:.0%}"
    return {
        "plan_discordant_cell": sig,
        "game": cells[A.CONTROL][sig]["game"],
        "four_arm_table": table,
        "what_the_mask_did": (
            "took an engine from verify_accuracy 0.0 (rejected) to 0.88 (ADMITTED) while its "
            "verify_cell_recall and verify_change_fidelity both stayed at 0.0 -- the engine got "
            "ZERO of the changed cells right in both arms. The score moved; the engine did not."
        ),
        "why_the_score_moved_noop_dominance": {
            "frozen_live_corpus": {
                "fixture": "tests/fixtures/arc_hud_mask_swallow/lf52_live_episode.npz",
                "sha256": live.get("sha256"),
                "corpus": live.get("corpus"),
                "captured_by": (
                    "the guard-fix work in this same session, from the same "
                    "experiment_4605.run_variant_attempt entrypoint this run used"
                ),
            },
            "n_transitions": n_t,
            "changing_transitions_raw": sc.get("raw_changing_transitions"),
            "changing_transitions_deleted_by_mask": deleted,
            "changing_transitions_surviving_mask": surviving,
            "changing_transition_survival": sc.get("changing_transition_survival"),
            "noop_fraction_of_masked_corpus": noop_fraction_after_mask,
            "observed_verify_accuracy_in_mask_arm": observed_acc,
            "arithmetic": (
                f"a masked corpus that is {_noop_pct} no-ops is passed at ~that rate by an "
                f"engine predicting nothing changes; the observed verify_accuracy "
                f"{observed_acc} sits just under that no-op fraction while "
                "change_fidelity is 0.0, so the score is no-op dominance, not dynamics learned"
            ),
        },
        "the_swallow_guard_was_RIGHT_here": {
            "swallow_check_reason_at_capture": sc.get("reason"),
            "changed_cell_overlap": sc.get("changed_cell_overlap"),
            "overlap_threshold": sc.get("overlap_threshold"),
            "verdict": (
                "the guard measured overlap 0.3086 and cleared the mask (`ok`). On the LIVE "
                "corpus that is correct: 56 of 81 changed cells lie OUTSIDE the mask. The "
                "1.0000-overlap figure recorded for lf52 elsewhere is from a different corpus "
                "(120 random offline actions) and does not describe this run."
            ),
            "so_the_defect_is_NOT_the_mask_or_the_guard": (
                "it is the legacy full-grid `accuracy >= 0.5` admission gate, which is no-op "
                "dominated once the mask removes the only cells that were changing"
            ),
        },
        "the_change_gate_rejected_it": {
            "arm": "wm_A3_both",
            "skipped": table["wm_A3_both"]["skipped"],
            "gate_protected_against_the_mask": bool(
                table["wm_A1_mask"]["planned"] > 0 and table["wm_A3_both"]["planned"] == 0
            ),
            "reading": (
                "mask-only ADMITTED this engine; mask+gate REJECTED it. The gate is protecting "
                "against the mask -- the opposite sign from 'the gate cancels the mask's "
                "benefit'. There was no benefit to cancel."
            ),
        },
        "and_the_admission_bought_NOTHING": {
            "skipped_after_admission": table["wm_A1_mask"]["skipped"],
            "first_win": table["wm_A1_mask"]["first_win"],
            "reached_level": table["wm_A1_mask"]["reached_level"],
            "reading": (
                "the admitted engine yielded `no_reachable_plan_after_refinement`: no plan was "
                "ever produced from it, no action was taken from it, and the cell did not win. "
                "So even the single admission is not a plan that did anything."
            ),
        },
        "NOT_AN_IMPROVEMENT": (
            "1 admission of 25 cells, on an engine with change_fidelity 0.0, that produced no "
            "plan, no action, no level and no win, and which the change gate rejected. This "
            "must not be read as the mask working."
        ),
    }


# ---------------------------------------------------------------------------- main


def main() -> int:
    t0 = time.time()
    a = A.analyse()
    cells = A.read_cells()
    n = a["n_matched_cells_per_arm"]

    win = win_endpoint(a, cells)
    treat = arm_treatment_evidence(cells, a)
    rollups = rollup_availability()
    record = mask_record_coverage(a)
    inert = inert_equality_rescoped(a, cells)
    nondet = generator_nondeterminism(a, cells)
    gates_rej = gate_rejections_rescoped(a)
    lf52 = lf52_finding(a, cells)

    per = a["per_arm"]
    cmp_ = a["comparisons_vs_control"]
    wit = a["witnesses"]
    integ = a["arm_integrity"]

    # ---- the headline, computed ------------------------------------------------
    n_wins = {arm: win["per_arm"][arm]["n_wins"] for arm in A.ARMS}
    n_planned = {arm: per[arm]["n_planned_gt_0"] for arm in A.ARMS}
    max_level = {arm: win["per_arm"][arm]["max_reached_level"] for arm in A.ARMS}
    all_win_discordance_zero = all(
        win["comparisons_vs_control"][arm]["n_win_discordant_pairs"] == 0 for arm in A.TREATMENTS
    )
    any_test_possible = any(
        win["comparisons_vs_control"][arm]["test_was_possible"] for arm in A.TREATMENTS
    )

    headline = {
        "result": "NULL at the win endpoint -- and no statistical test was possible",
        "per_arm_summary": {
            arm: {
                "planned_admissions": n_planned[arm],
                "wins": n_wins[arm],
                "max_reached_level": max_level[arm],
            }
            for arm in A.ARMS
        },
        "the_one_win": {
            "cell": (win["win_provenance"][0]["sig"] if win["win_provenance"] else None),
            "won_in_all_four_arms": (
                win["win_provenance"][0]["won_in_arms"] if win["win_provenance"] else []
            ),
            "bit_identical_across_arms": (
                win["win_provenance"][0]["identical_across_all_four_arms"]
                if win["win_provenance"]
                else None
            ),
            "produced_by_an_induced_world_model": (
                win["win_provenance"][0]["produced_by_induced_world_model_in_any_arm"]
                if win["win_provenance"]
                else None
            ),
            "reading": (
                "one win, the same cell, in every arm, with identical actions and level, and "
                "with ZERO world-model admissions on that cell in any arm. The win came from "
                "exploration, not from the induce->plan path, so it carries no information "
                "about the treatments -- and no arm gained or lost a win anywhere."
            ),
        },
        "win_discordance_is_zero_in_every_arm": all_win_discordance_zero,
        "min_reachable_two_sided_p": {
            arm: win["comparisons_vs_control"][arm]["min_reachable_two_sided_p"]
            for arm in A.TREATMENTS
        },
        "no_test_was_possible": not any_test_possible,
        "REQUIRED_PHRASING": (
            "With zero win-discordant pairs in every arm the minimum reachable two-sided p is "
            "1.0, so NO significance is available at ANY outcome. This is NOT 'no significant "
            "difference between arms' -- no test was performed, because there was no "
            "discordance to test."
        ),
        "measurement_artifact_hypothesis": "REFUTED",
        "measurement_artifact_hypothesis_detail": (
            "the hypothesis was that the induced-world-model wall is an artifact of HUD "
            "contamination inflating the exact-match denominator, and that removing it would "
            "open the induce->plan path. Removing it did not: mask-only produced ONE admission "
            "of 25 (an engine with change_fidelity 0.0 that then yielded no reachable plan), "
            "zero additional plans that executed, zero additional levels and zero additional "
            "wins. The wall is induction CAPABILITY, not measurement."
        ),
        "what_the_mask_DID_do": (
            "it raised the measured score. The gated quantity moved on 13 of 25 cells in the "
            "mask arm (11 up, 2 down) -- but the inert-equality control FAILS on cells the "
            "mask provably never touched, so that movement is confounded with generator "
            "nondeterminism and is NOT attributable to the mask. Either way it bought no "
            "decision: score inflation without capability is the laundering hazard, not a gain."
        ),
        "what_the_GATE_did": (
            "it rejected {} engine-attempts, none of which the incumbent gate would have "
            "admitted, so its live effect on the admission endpoint is nil -- EXCEPT that in "
            "the both-arm it rejected precisely the engine the mask had admitted (lf52). The "
            "gate is protecting against the mask.".format(gates_rej["n_rejections"])
        ),
        "both_stay_DEFAULT_OFF": (
            "the mask and the change gate are GUARDS, not levers. This run measures them and "
            "does not flip them: SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED and "
            "SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED remain False."
        ),
    }

    # ---- acceptance gates, each with a reachable failing value -----------------
    g_matched = bool(n > 0 and not a["unmatched_signatures"])
    g_cells_complete = all(per[arm]["n_cells"] == n for arm in A.ARMS)
    g_no_cell_errors = all(per[arm]["n_cell_errors"] == 0 for arm in A.ARMS)
    g_treatment_evidenced = all(
        treat[arm]["per_cell_evidence_matches_declaration"] for arm in A.ARMS
    )
    g_generator_live = all(wit["generator_liveness"][arm]["generator_answered"] for arm in A.ARMS)
    # Residency is asserted ONLY for arms that recorded it. An arm with no rollup is excluded
    # and named in `rollup_availability`, so this gate cannot pass by silently ignoring a gap.
    arms_with_residency = [arm for arm in A.ARMS if rollups["per_arm"][arm]["rollup_present"]]
    g_device = bool(arms_with_residency) and all(
        (integ[arm]["server_device"] or "").startswith("CONFIRMED_GPU")
        for arm in arms_with_residency
    )
    g_no_card_drop = bool(arms_with_residency) and all(
        not ((integ[arm]["vram_summary"] or {}).get("dropped_below_1gib_mid_run"))
        for arm in arms_with_residency
    )
    g_mask_proof = wit["mask_application_proof"]["all_claims_proved"]
    g_no_regression = all(cmp_[arm]["first_win"]["regression_clause_holds"] for arm in A.TREATMENTS)
    g_defaults_off = all(
        (
            json.loads((OUT / f"run_{arm}.json").read_text()).get("shipped_defaults_unchanged")
            or {}
        ).get("SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED")
        is False
        for arm in arms_with_residency
    )
    # The honesty gates: this artifact must NOT report the null as a tested difference, and
    # must NOT report an unmeasurable as clean. Both are computed, both can fail.
    g_null_not_dressed_as_test = bool(all_win_discordance_zero and not any_test_possible)
    # THERE IS DELIBERATELY NO GATE ON THE UNMEASURABLE-REJECTION RECOUNT. Two drafts of one
    # were written and both were rejected as unfalsifiable:
    #   (a) `channel_is_dead AND recount_correct` -- would FAIL once the dead swallow channel
    #       is fixed, i.e. a gate asserting the continued existence of a defect.
    #   (b) `recount == independently_recomputed_count` -- TAUTOLOGICAL: the classification and
    #       the check both read the same reason string through the same marker list, so
    #       emptying the marker list moves both sides together and the gate cannot fail.
    #       Confirmed by mutation M2, which failed to flip it.
    # With only (reason, change_fidelity, counts) on a rejection record, the reason string IS
    # the classifier, so any gate over it is circular. The recount is therefore reported as a
    # FINDING (49 justified / 1 unmeasurable, against the analyser's uncorrected 50) and is not
    # dressed up as a passed gate.

    gates = {
        "acceptance_gate_every_cell_matched_across_all_four_arms": g_matched,
        "acceptance_gate_all_arms_have_full_cell_count": g_cells_complete,
        "acceptance_gate_zero_cell_errors": g_no_cell_errors,
        "acceptance_gate_every_arm_treatment_evidenced_per_cell": g_treatment_evidenced,
        "acceptance_gate_generator_answered_in_every_arm": g_generator_live,
        "acceptance_gate_device_confirmed_by_per_pid_residency_where_recorded": g_device,
        "acceptance_gate_no_card_dropped_off_bus_where_recorded": g_no_card_drop,
        "acceptance_gate_every_mask_claim_carries_positive_cell_count": g_mask_proof,
        "acceptance_gate_no_control_win_lost_by_any_treatment_arm": g_no_regression,
        "acceptance_gate_shipped_defaults_still_off": g_defaults_off,
        "acceptance_gate_null_reported_as_untestable_not_as_no_difference": (
            g_null_not_dressed_as_test
        ),
        "acceptance_gate_mask_applied_count_agrees_across_two_independent_paths": inert[
            "applied_count_cross_check_agrees_in_every_mask_arm"
        ],
    }
    passed = all(gates.values())

    # ---- vacuity disclosure: a pass over an empty set is not evidence ----------
    n_control_wins = per[A.CONTROL]["n_first_win"]
    vacuity = {
        "acceptance_gate_no_control_win_lost_by_any_treatment_arm": {
            "vacuous": n_control_wins == 0,
            "support": n_control_wins,
            "meaning_if_vacuous": (
                "the control won on zero cells, so no arm COULD have lost a win and the "
                "regression clause has no support"
            ),
        },
        "acceptance_gate_every_mask_claim_carries_positive_cell_count": {
            "vacuous": wit["mask_application_proof"]["n_claims"] == 0,
            "support": wit["mask_application_proof"]["n_claims"],
            "meaning_if_vacuous": "the mask applied on zero cells, so the proof is empty",
        },
        "acceptance_gate_device_confirmed_by_per_pid_residency_where_recorded": {
            "vacuous": len(arms_with_residency) < len(A.ARMS),
            "support": len(arms_with_residency),
            "meaning_if_vacuous": (
                "{} of {} arms recorded residency; the arm(s) {} are NOT covered by this gate "
                "and their device placement is not claimed".format(
                    len(arms_with_residency), len(A.ARMS), rollups["arms_missing_rollup"]
                )
            ),
        },
        "acceptance_gate_no_card_dropped_off_bus_where_recorded": {
            "vacuous": len(arms_with_residency) < len(A.ARMS),
            "support": len(arms_with_residency),
            "meaning_if_vacuous": (
                "an arm with no VRAM timeseries could have dropped off the bus undetected "
                "rather than provably not have done"
            ),
        },
        "acceptance_gate_shipped_defaults_still_off": {
            "vacuous": len(arms_with_residency) == 0,
            "support": len(arms_with_residency),
            "meaning_if_vacuous": "no rollup recorded the shipped defaults",
        },
    }
    n_vacuous = sum(1 for v in vacuity.values() if v["vacuous"])

    verdict = (
        "complete_four_arm_live_null_zero_win_discordance_min_reachable_p_1.0_"
        "measurement_artifact_hypothesis_refuted_mask_admitted_1_of_25_"
        "change_fidelity_zero_engine_that_produced_no_plan"
    )

    art = {
        "experiment": EXP,
        "experiment_id": f"exp{EXP}",
        "title": (
            "REQ-ARC-WMTE-6010/-6011/-6013 four-arm LIVE matrix: HUD-mask and change-gate on "
            "the scored induce->plan path -- a win-endpoint NULL, and the mask's one admission "
            "is a laundering hazard"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "outer-loop 2026-07-27",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_note": (
            "the ROWS were produced by live GPU inference (Qwen3.5-9B-MTP via a CUDA "
            "llama-server, one server per arm, per-PID VRAM residency recorded for the three "
            "arms that wrote rollups). THIS artifact is an analyser pass over those persisted "
            "rows: it loads no model and runs no inference, so duration_s is the analyser "
            "clock and measurement_wall_s comes from the rows' own elapsed_s."
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "no moat, efficiency or headline capability claim is made. The quantity under test "
            "is a world-model verifier's agreement with OBSERVED env transitions used as an "
            "admission gate; this artifact reports that the treatments changed no decision "
            "that mattered. A null needs no oracle-distinctness claim to be honest, and none "
            "is made."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "public games through the shipped E3AgentPolicy via experiment_4605's real "
            "run_variant_attempt. NO solve is claimed by this artifact. One cell (vc33~color01) "
            "reached level 1 identically in all four arms; its own row records "
            "reproduction_gate.reproduced=true under mode `offline_reproduction_gate_no_quota`, "
            "which means the reproduction gate did NOT run -- so offline_reproduced is False "
            "and reproduced_levels is 0 here. That row field is itself another instance of the "
            "unmeasurable-read-as-clean shape and is reported, not relied on."
        ),
        "submitted_to_leaderboard": False,
        "submitted_to_leaderboard_note": (
            "no submission was made or prepared. The operator-only quota gate (an offline "
            "result beating BOTH a TRM baseline AND the best prior submitted run) is NOT met "
            "by a null."
        ),
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "random_seed": 0,
        "random_seeds_used": [0],
        "random_seed_note": (
            "the pairing unit is the (game, variant) cell, not an RNG seed: all four arms ran "
            "the SAME 25 games at variant 1 and are compared PER CELL on the 25 signatures "
            "present in all four arms -- never an any-seed union, which would let an arm be "
            "scored on cells its control never ran. Note that the generator is stochastic, so "
            "the arms are matched on the cell and NOT on the induced engine; see "
            "generator_nondeterminism for what that costs."
        ),
        "model_specs": {
            "generator": "unsloth/Qwen3.5-9B-MTP-GGUF (Qwen3.5-9B-Q4_K_M.gguf)",
            "n_ctx": 81920,
            "max_tokens": 4096,
            "server": "~/.cache/llama.cpp-master/build/bin/llama-server (CUDA build)",
            "device": (
                "one server per arm, ~13.5 GiB resident; GPU 1 for A0/A1 and GPU 0 for A3 by "
                "per-PID residency. Never two servers on one card."
            ),
            "note": (
                "the 9B, not the 31B: a 21GB gemma-4-31B on a 24GB card triggered the "
                "documented eGPU PCI-bus fault on 2026-07-24 (GPU 1 dropped 21GB -> 4MiB "
                "mid-run). A per-PID VRAM timeseries was sampled DURING each arm so a card "
                "falling off the bus would be recorded rather than surfacing as a hang."
            ),
        },
        "duration_s": None,  # filled below, after every field is computed
        "duration_s_provenance": "analyser wall time only; NOT the measurement clock",
        "measurement_wall_s": a["measurement_wall_s"],
        "measurement_wall_s_provenance": (
            "sum of every cited cell row file's OWN elapsed_s across all 100 cells; the arms "
            "ran with k=4 concurrency in two parallel chains, so this exceeds the wall-clock "
            "span (see measurement_wall_clock_span_s)"
        ),
        "measurement_wall_clock_span_s": {
            arm: (json.loads((OUT / f"run_{arm}.json").read_text()).get("wall_s"))
            for arm in arms_with_residency
        },
        "measurement_wall_s_per_arm_from_rows": {
            arm: round(sum(cells[arm][s]["elapsed_s"] for s in a["matched_signatures"]), 3)
            for arm in A.ARMS
        },
        "preconditions_checked": [
            {
                "resource": "100 cell row files present under cells/ (4 arms x 25 games)",
                "available": g_cells_complete,
            },
            {
                "resource": "every signature matched across all four arms",
                "available": g_matched,
            },
            {
                "resource": "per-arm engine-store isolation (CARNOT_ARC_E3_DIR per arm)",
                "available": True,
                "evidence": (
                    "fourarm.py:573-592 sets and asserts a per-arm store; the pre-isolation "
                    "cells are quarantined in cells_PRE_ISOLATION_FIX_DISCARDED/ and are NOT "
                    "read by this analyser"
                ),
            },
            {
                "resource": "per-arm run rollup (resolver read-back + device residency)",
                "available": not rollups["arms_missing_rollup"],
                "note": (
                    "MISSING for {}; treatment established from per-cell evidence instead, and "
                    "device residency is NOT claimed for that arm".format(
                        rollups["arms_missing_rollup"]
                    )
                ),
            },
            {
                "resource": "frozen lf52 live-corpus fixture for the laundering arithmetic",
                "available": (FIXTURES / "lf52_live_episode.npz").exists(),
            },
        ],
        "four_arms": A.ARM_DECLARED,
        "headline": headline,
        "win_endpoint": win,
        "lf52_laundering_finding": lf52,
        "arm_treatment_evidence": treat,
        "rollup_availability": rollups,
        "mask_decision_record_coverage": record,
        "mask_inert_equality_control_rescoped": inert,
        "generator_nondeterminism": nondet,
        "gate_rejections_rescoped": gates_rej,
        "arm_integrity_from_rollups": integ,
        "per_arm": per,
        "comparisons_vs_control": cmp_,
        "interaction": a["interaction"],
        "witnesses_from_analyser": wit,
        "matched_signatures": a["matched_signatures"],
        "unmatched_signatures": a["unmatched_signatures"],
        "n_matched_cells_per_arm": n,
        "guard_gap_found_by_this_run_and_fixed_same_session": {
            "found_how": (
                "this run's single plan-discordant cell (lf52) sent the session to read the "
                "swallow guard's decision -- and the record was empty: hud_mask_swallow is None "
                "in 104 of 104 diagnostics. The failure could be DATED but not EXPLAINED."
            ),
            "defects_fixed": [
                "hud_mask_swallow_check's unmeasurable verdict (`no_dynamics_to_swallow`, "
                "which leaves `swallows` at its initialised False) was read by both consumers "
                "as truthiness, so an unmeasurable verdict applied the mask as if it had been "
                "checked and cleared; now requires an affirmative reason == 'ok'",
                "select_trusted_world_model reported hud_mask_status 'disabled' for its own "
                "refusals, indistinguishable from the flag being off, on all 11 hidden-state "
                "games",
                "hud_mask_swallow was computed and discarded (the dead channel this run "
                "exhibits); now lifted onto the attempt dict in both branches and added to the "
                "projection",
                "the swallow verdict was unreadable without its corpus; transition-level "
                "counts and survival/overlap statistics were added to the record",
            ],
            "measured_scope_of_the_fix": (
                "on the measured corpus the fix flips ZERO admissions -- its one on-disk "
                "instance (ft09, all 3 seeds in exp6011) has an all-no-op corpus where identity "
                "scores 1.0 masked or not, so the correction is to 3 records, not to a metric. "
                "The laundering it closes needs an engine erring only inside masked cells."
            ),
            "no_new_gate_was_added": (
                "neither transition-level statistic separates honest from swallowing masks "
                "across the 25 games, so a threshold would have to be fitted in a <0.05 window "
                "between two games. The statistics are RECORDED and not gated, and a test "
                "asserts that."
            ),
            "not_retro_applied": (
                "the fixed guard is NOT applied to these 100 cells. This run's record stays as "
                "measured; the corrections live in new code plus an append-only corrigendum."
            ),
        },
        "what_this_run_refutes": {
            "hypothesis": (
                "the induced-world-model wall on the live scored path is a MEASUREMENT "
                "ARTIFACT: HUD/counter cells contaminate the exact-match denominator, the trust "
                "score is pinned near the floor as a result, and removing that contamination "
                "will open the induce->plan path."
            ),
            "verdict": "REFUTED",
            "evidence": (
                "with the mask on, admissions went 0 -> 1 of 25 and that single admission was "
                "an engine with change_fidelity 0.0 that then produced no reachable plan. Wins "
                "went 1 -> 1 (the same cell, bit-identical, and with zero admissions on it in "
                "every arm). Max level 1 -> 1. Zero win-discordant pairs in every arm."
            ),
            "what_remains": (
                "the wall is induction CAPABILITY -- the generator does not produce engines "
                "that predict the changing cells. change_fidelity is 0.0 on essentially every "
                "rejected engine in every arm, which is a statement about the engines, not "
                "about the measurement of them."
            ),
        },
        "field_provenance": {
            "measurement_wall_s": {
                "principle": (
                    "the analyser clock is not the measurement clock; republishing each row's "
                    "own elapsed_s is what makes a seconds-long analyser pass over a "
                    "hours-long measurement honest instead of a DURATION_TOO_SHORT signal"
                ),
                "satisfied_by": "sum of cells/*.json elapsed_s over all 100 cells",
            },
            "headline.min_reachable_two_sided_p": {
                "principle": (
                    "with n discordant pairs the smallest reachable two-sided p is 2^(1-n); at "
                    "n=0 no test exists at all, so stating the bound is what stops a "
                    "zero-discordance result being written up as 'no significant difference'"
                ),
                "satisfied_by": "computed per arm from the win-discordant pair count",
            },
            "arm_treatment_evidence": {
                "principle": (
                    "an arm whose rollup is missing is UNMEASURABLE at the resolver, not "
                    "clean and not broken; proving the treatment from per-cell fields the "
                    "agent wrote at decision time is what keeps the gap from being papered "
                    "over in either direction"
                ),
                "satisfied_by": (
                    "hud_mask_reason (flag_disabled can never read `resolved`) plus "
                    "change_gate_hidden_state_enabled, censused per arm from the cells"
                ),
            },
            "mask_decision_record_coverage": {
                "principle": (
                    "a field computed and discarded is a dead channel: the decision it "
                    "documents becomes unauditable, which is exactly why this run's guard "
                    "failure could be dated but not explained"
                ),
                "satisfied_by": (
                    "census of hud_mask_swallow and hud_mask_status presence over all 104 "
                    "diagnostics, split by arm and branch"
                ),
            },
            "mask_inert_equality_control_rescoped": {
                "principle": (
                    "if a treatment arm differs from control on cells the treatment provably "
                    "never touched, then no per-cell comparison on that run is attributable "
                    "to the treatment; scoping the control to PROVABLY-untouched cells is what "
                    "makes that inference valid instead of pooling in cells whose "
                    "applicability was never recorded"
                ),
                "satisfied_by": (
                    "three-way split of each mask arm's cells into provably-no-mask / applied "
                    "/ applicability-UNKNOWN, with violations counted only on the provable set"
                ),
            },
            "gate_rejections_rescoped.n_UNMEASURABLE_nothing_to_judge": {
                "principle": (
                    "change_fidelity == 0.0 on a rejection whose reason is "
                    "`no_changing_transitions` is the field's INITIALISED value, not a "
                    "measurement; counting it as justified-by-low-fidelity is the same "
                    "unmeasurable-read-as-clean bug the swallow guard had, one level up in the "
                    "auditor itself"
                ),
                "satisfied_by": (
                    "recount of the analyser's rejection detail, partitioning on the reason "
                    "string rather than on the fidelity value"
                ),
            },
            "lf52_laundering_finding.why_the_score_moved_noop_dominance": {
                "principle": (
                    "a score that rises because the denominator lost the only cells that were "
                    "changing is laundering, not learning; the arithmetic has to be shown "
                    "against a frozen corpus with a sha256 so the claim traces to data"
                ),
                "satisfied_by": (
                    "the frozen lf52 live-episode fixture's own captured swallow-check "
                    "statistics, cited by sha256, against the observed verify_accuracy"
                ),
            },
        },
        **gates,
        "acceptance_gate_passed": passed,
        # REQ-ARC-WMTE-6019: `unmet_gates` used to be derived from `gates` ALONE, while the
        # composite `acceptance_gate_passed_and_none_vacuous` is computed OUTSIDE that dict
        # -- so the one gate that actually fails on this run could never appear in the list.
        # On disk that produced `acceptance_gate_passed: true` + `unmet_gates: []` beside
        # `acceptance_gate_passed_and_none_vacuous: false`: a consumer reading the project's
        # documented pair (pass flag + unmet list) saw "passed, nothing unmet" while a gate
        # was failing, and only `summarize_artifact.py`'s key-prefix scan surfaced it. The
        # composite is now included in the derivation, and each vacuous gate is named in the
        # list too -- a gate whose pass region is empty is not a met gate.
        "unmet_gates": sorted(
            [k for k, v in gates.items() if not v]
            + (
                []
                if bool(passed and n_vacuous == 0)
                else ["acceptance_gate_passed_and_none_vacuous"]
            )
            + [f"{k}__VACUOUS" for k, v in vacuity.items() if v["vacuous"]]
        ),
        "unmet_gates_includes_vacuous_and_composite": True,
        "gate_vacuity_disclosure": vacuity,
        "n_vacuous_gates": n_vacuous,
        "vacuous_gates": sorted(k for k, v in vacuity.items() if v["vacuous"]),
        "acceptance_gate_passed_and_none_vacuous": bool(passed and n_vacuous == 0),
        "honest_verdict": verdict,
        "cited_upstream_artifacts": [
            {
                "path": "results/experiment_6011_world_model_change_gate_four_arm.json",
                "role": (
                    "the OFFLINE four-arm matrix this run tests live; also the artifact holding "
                    "the swallow guard's one on-disk unmeasurable-read-as-clean instance (ft09)"
                ),
                "sha256": _sha(
                    REPO / "results" / "experiment_6011_world_model_change_gate_four_arm.json"
                ),
            },
            {
                "path": "results/experiment_6012_hidden_state_trust_gate_hole.json",
                "role": "the hidden-state trust-gate hole that motivated the change gate",
                "sha256": _sha(
                    REPO / "results" / "experiment_6012_hidden_state_trust_gate_hole.json"
                ),
            },
            {
                "path": "results/experiment_6013_hidden_state_change_gate_closure.json",
                "role": "the hidden-state change-gate closure this run exercises live",
                "sha256": _sha(
                    REPO / "results" / "experiment_6013_hidden_state_change_gate_closure.json"
                ),
            },
            {
                "path": "tests/fixtures/arc_hud_mask_swallow/MANIFEST.json",
                "role": (
                    "frozen real corpora captured by the guard-fix work; source of the lf52 "
                    "live-corpus no-op-dominance arithmetic"
                ),
                "sha256": _sha(FIXTURES / "MANIFEST.json"),
            },
            {
                "path": (
                    "results/arc_wm_four_arm_20260727/cells_PRE_ISOLATION_FIX_DISCARDED/"
                    "WHY_DISCARDED.md"
                ),
                "role": (
                    "the quarantined pre-isolation cells this analyser does NOT read; kept per "
                    "never-prune"
                ),
                "sha256": _sha(OUT / "cells_PRE_ISOLATION_FIX_DISCARDED" / "WHY_DISCARDED.md"),
            },
        ],
        "cited_code_fixes": [
            {
                "path": "python/carnot/agentic/arc_executable_world_model.py",
                "role": "hud_mask_swallow_clean() affirmative-reason requirement + record fields",
            },
            {
                "path": "python/carnot/agentic/arc_world_model_trust_energy.py",
                "role": "select_trusted_world_model refusal-status visibility",
            },
            {
                "path": "python/carnot/agentic/arc_competition_agent.py",
                "role": "hud_mask_swallow lifted onto the attempt dict + projection key",
            },
        ],
        "row_files_dir": "results/arc_wm_four_arm_20260727/cells/",
        "runner": "results/arc_wm_four_arm_20260727/fourarm.py",
        "analyser": (
            "results/arc_wm_four_arm_20260727/analyse_fourarm.py (imported) + "
            "build_live_artifact.py (this builder's additive blocks)"
        ),
        "run_git_head": "a2222320c629ae14699ca795a4de27abf4c4e296",
        "git_head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }

    art["duration_s"] = round(time.time() - t0, 4)
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in art.items() if k not in ("run_date", "duration_s")},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    # Additive only: the DISCARDED run's analysis_fourarm.json is tracked and is left alone.
    (OUT / "analysis_fourarm_live.json").write_text(json.dumps(a, indent=1, default=str))
    ART.write_text(json.dumps(art, indent=1, default=str))

    print("wrote", ART)
    print("wrote", OUT / "analysis_fourarm_live.json")
    print()
    print("verdict:", verdict)
    print("per-arm:", json.dumps(headline["per_arm_summary"]))
    print("win discordance zero everywhere:", all_win_discordance_zero)
    print("min reachable two-sided p:", json.dumps(headline["min_reachable_two_sided_p"]))
    print("gates:", json.dumps(gates, indent=1))
    print("acceptance_gate_passed:", passed, " unmet:", art["unmet_gates"])
    print("vacuous:", art["vacuous_gates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
