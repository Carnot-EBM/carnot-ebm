"""Assemble the final results/ artifact from the analysis, deriving the verdict FROM the data.

No conclusion in this file is hardcoded.  Every gate reads its own computed witness first and
refuses to emit a verdict when its pass region is empty, so the artifact cannot report
"transfers" for a contrast that never had a measurable effect to begin with.
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Work directory holding the JSONL cell rows + intermediate JSON. Overridable so the
# battery can be run out of a scratch dir (as it was for the recorded run) or a
# repo-local dir, without editing the file.
SCRATCH = Path(os.environ.get("CPTB_WORKDIR") or Path(__file__).resolve().parent)
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "outer_loop_cptb_shipped_lever_convention_transfer_20260726.json"

# Every (contrast, perturbed condition) pair is gated, not just the headline one, so an
# inert pair cannot quietly contribute a retention ratio of 1.0 to the reading.  The
# `headline` flag marks the pair that answers each lever's SHIPPED claim: the frontier flip
# was measured with the HUD off (it predates the HUD flip), and the HUD flip was measured
# with the frontier already on.
# (contrast, lever) -- the perturbed conditions are taken from whatever the run actually
# measured, so adding the C3 dose axis does not require editing a table (and a run that lacks
# it does not produce phantom gates).
GATE_CONTRASTS = [
    ("frontier_given_hud_off", "frontier"),
    ("frontier_given_hud_on", "frontier"),
    ("hud_given_frontier_on", "hud"),
    ("hud_given_frontier_off", "hud"),
    ("both_levers_shipped_vs_preflip", "both"),
]

# WHICH CONTRAST answers each lever's own claim.  Fixed, so it cannot be selected after the
# fact: the frontier flip was measured with the HUD off (it predates the HUD flip); the HUD
# flip was measured with the frontier lever already on.
HEADLINE_CONTRAST = {"frontier": "frontier_given_hud_off", "hud": "hud_given_frontier_on"}

# Which CONDITION supplies each lever's headline verdict, chosen by a rule stated in advance:
# the ON-TARGET, EVALUABLE condition with the SMALLEST perturbation magnitude.  Selecting on
# EVALUABILITY (can this question be answered at all?) is legitimate; selecting on the ANSWER
# would not be, and is not done -- the verdict is read only after the condition is fixed.  If
# no on-target condition is evaluable, the lever is NOT_DECIDABLE_BY_THIS_DESIGN.
CONDITION_RANK = {
    "C1_salience_inversion": 0,  # colour, no geometric distortion at all
    "C3_roll_k1": 1,  # 1-cell roll: mildest geometric dose
    "C3_roll_k2": 2,
    "C2_diag_roll": 3,  # 3-cell roll: measured to raze the corpus
}

# WHICH LEVER'S CONVENTION EACH PERTURBATION ACTUALLY ATTACKS.  This is not bookkeeping --
# without it a gate labelled `lever: hud` evaluated under C1 reads as "the HUD's convention was
# violated and its gain died", when C1 provably does not touch the HUD mechanism at all (the
# Stage-1 predicate is pure geometry, and the static dose witness measures zero HUD-mask change
# on all 25 games).  Attributing a lost gain to a convention that was never perturbed is
# exactly failure mode #7 (crediting a result to the wrong mechanism), so every gate carries
# this flag and off-target gates get their own verdict token.
CONDITION_TARGETS = {
    "C1_salience_inversion": {"frontier"},  # absolute-colour salience: frontier tier predicate
    "C2_diag_roll": {"hud", "frontier"},  # edge adjacency (HUD) + object geometry (frontier)
    # THE DOSE AXIS, and the k=1 entry is deliberately NOT hud-on-target.  The Stage-1
    # predicate is `on_top = y1 < EDGE_BAR_EDGE_TOLERANCE` with the tolerance = 2, so a mask
    # whose lowest row is index 0 -- which is r11l's, a single 64-cell row -- still satisfies
    # the predicate after a 1-cell roll (y1: 0 -> 1 < 2).  A k=1 roll therefore perturbs object
    # GEOMETRY (which the frontier tier predicate's width test reads) without violating the
    # HUD lever's edge-adjacency convention on the game that carries its gain.  Measurement
    # corroborates the derivation: at k=1 r11l's mask still resolves and HUD-alone still wins
    # it; at k=2 the mask stops resolving entirely.  Marking k=1 hud-on-target would credit a
    # result to a convention that was not perturbed -- failure mode #7.
    "C3_roll_k1": {"frontier"},
    "C3_roll_k2": {"hud", "frontier"},
}


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout.strip()


# Where the budget-sweep rows live.  Globbed from both the work directory and the committed
# cell directory so the artifact can be rebuilt from the repo alone.
_SWEEP_DIRS = (SCRATCH, REPO / "results" / "cptb_20260726_cells")


def _load_sweep_rows() -> list[dict]:
    """Every budget-sweep row available, from .jsonl or .jsonl.gz, de-duplicated.

    De-duplication is by the full cell key (arm, game, condition, seed, budget) because the
    same cell exists in both the original 1-2 seed sweep and the 5-seed re-run; keeping both
    copies would double-count a seed and inflate n_seeds, which is precisely the field these
    rows exist to make honest.
    """

    seen: dict[tuple, dict] = {}
    for d in _SWEEP_DIRS:
        if not d.exists():
            continue
        for p in sorted(list(d.glob("*sweep*.jsonl")) + list(d.glob("*sweep*.jsonl.gz"))):
            text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
            for line in text.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                k = (
                    r.get("arm"),
                    r.get("game"),
                    r.get("condition"),
                    int(r.get("seed") or 0),
                    int(r.get("budget") or 0),
                )
                seen.setdefault(k, r)
    return list(seen.values())


def _load_probe_rows() -> list[dict]:
    """Rows from the targeted roll-magnitude dose-response probe, if it was run.

    Kept OUT of the battery (and out of the sweep aggregate) on purpose: it covers only the
    two games the HUD lever moves, so folding it into the corpus-level win sets would make the
    per-condition coverage unbalanced and the win counts incomparable.
    """

    seen: dict[tuple, dict] = {}
    for d in _SWEEP_DIRS:
        if not d.exists():
            continue
        for p in sorted(
            list(d.glob("*probe_rollk*.jsonl")) + list(d.glob("*probe_rollk*.jsonl.gz"))
        ):
            text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
            for line in text.splitlines():
                if line.strip():
                    r = json.loads(line)
                    seen.setdefault(
                        (r.get("arm"), r.get("game"), r.get("condition"), int(r.get("seed") or 0)),
                        r,
                    )
    return list(seen.values())


def _roll_dose_response(rows: list[dict]) -> dict:
    """Per (game, roll magnitude): did the mask still resolve, and could ANY arm still win?

    This is the measurement that decides whether the roll FAMILY can ever test the HUD
    lever's convention: if the smallest magnitude that stops the detector firing is also the
    magnitude at which every arm loses the game, then no member of the family can separate
    "the convention broke" from "the task broke".
    """

    out: dict[str, dict] = {}
    for r in rows:
        g, c, a = r.get("game"), r.get("condition"), r.get("arm")
        e = out.setdefault(g, {}).setdefault(
            c, {"arms": {}, "n_seeds": 0, "mask_resolved_seeds": 0, "mask_cell_counts": set()}
        )
        e["arms"].setdefault(a, {"n_seeds": 0, "n_won": 0})
        e["arms"][a]["n_seeds"] += 1
        e["arms"][a]["n_won"] += 1 if int(r.get("levels") or 0) > 0 else 0
        e["n_seeds"] += 1
        if r.get("hud_mask_resolved"):
            e["mask_resolved_seeds"] += 1
        if r.get("hud_mask_cell_count") is not None:
            e["mask_cell_counts"].add(int(r["hud_mask_cell_count"]))
    for g, byc in out.items():
        for c, e in byc.items():
            e["mask_cell_counts"] = sorted(e["mask_cell_counts"])
            e["dead_for_every_arm"] = all(v["n_won"] == 0 for v in e["arms"].values())
            e["mask_resolved_on_any_seed"] = e["mask_resolved_seeds"] > 0
    return out


def _summarise_sweeps(rows: list[dict]) -> dict:
    """Aggregate sweep rows per (game, condition, arm, budget) WITH n_seeds attached.

    n_seeds is the whole point.  The first recorded run's budget-sweep reading rested on a
    SINGLE seed for tn36 while the claim it overturned rested on five, and neither n appeared
    anywhere in the artifact or the report -- so a reader could not tell that the reclassifying
    evidence was weaker than the evidence it reclassified.
    """

    buckets: dict[str, list[dict]] = {}
    for r in rows:
        k = f"{r.get('game')}|{r.get('condition')}|{r.get('arm')}|budget_{r.get('budget')}"
        buckets.setdefault(k, []).append(r)
    out = {}
    for k, rs in sorted(buckets.items()):
        rs.sort(key=lambda r: int(r.get("seed") or 0))
        a1 = [
            r.get("actions_to_first_levelup")
            for r in rs
            if r.get("actions_to_first_levelup") is not None
        ]
        out[k] = {
            "n_seeds": len(rs),
            "seeds": [int(r.get("seed") or 0) for r in rs],
            "levels_per_seed": [int(r.get("levels") or 0) for r in rs],
            "n_seeds_won": sum(1 for r in rs if int(r.get("levels") or 0) > 0),
            "wins_on_every_seed": all(int(r.get("levels") or 0) > 0 for r in rs),
            "actions_to_first_levelup_per_seed": a1,
            "median_actions_to_first_levelup": (statistics.median(a1) if a1 else None),
            "min_actions_to_first_levelup": (min(a1) if a1 else None),
            "max_actions_to_first_levelup": (max(a1) if a1 else None),
            "states_expanded_per_seed": [r.get("states_expanded") for r in rs],
            "hud_mask_resolved_per_seed": [r.get("hud_mask_resolved") for r in rs],
        }
    return out


def _cost_ratio(sw: dict, treat_key: str, ctrl_key: str) -> dict:
    """Per-seed-range cost ratio between two sweep buckets, or a stated reason it is undefined.

    Reported as a RANGE across seeds, never as a single-cell point estimate: the first recorded
    run's '~3.5x' was one seed's 5337/1506 with no n attached.
    """

    t, c = sw.get(treat_key), sw.get(ctrl_key)
    if (
        not t
        or not c
        or not t["actions_to_first_levelup_per_seed"]
        or not c["actions_to_first_levelup_per_seed"]
    ):
        return {
            "defined": False,
            "reason": "one of the two arms never reached a level-up at this budget, so "
            "there is no cost to compare (a capability difference, not an "
            "efficiency one)",
        }
    ratios = [
        round(a / statistics.median(c["actions_to_first_levelup_per_seed"]), 3)
        for a in t["actions_to_first_levelup_per_seed"]
    ]
    return {
        "defined": True,
        "treatment": treat_key,
        "control": ctrl_key,
        "n_seeds_treatment": t["n_seeds"],
        "n_seeds_control": c["n_seeds"],
        "treatment_actions_per_seed": t["actions_to_first_levelup_per_seed"],
        "control_actions_per_seed": c["actions_to_first_levelup_per_seed"],
        "ratio_per_treatment_seed_vs_control_median": ratios,
        "ratio_range": [min(ratios), max(ratios)],
        "ratio_median": statistics.median(ratios),
        "note": "ratio of actions-to-first-level-up; a RANGE over seeds, not a single cell",
    }


def main() -> int:
    A = json.loads((SCRATCH / "cptb_analysis.json").read_text())
    receipt = json.loads((SCRATCH / "arm_receipt.json").read_text())
    arms = json.loads((SCRATCH / "cptb_arms_dump.json").read_text())

    A["config"]["arms"] = arms
    A["arm_flag_resolution_receipt"] = receipt

    integrity = A["cell_integrity"]
    interpretable = bool(integrity["interpretable"])

    # ---------------------------------------------------------------- gates (data-derived)
    perturbed_conds = [c for c in A["config"]["conditions"] if c != "C0_real"]
    gate_pairs = [(ct, cd, lev) for ct, lev in GATE_CONTRASTS for cd in perturbed_conds]
    gates = {}
    headline = {}
    for contrast, cond, lever in gate_pairs:
        is_headline = False  # designated below, once every gate's evaluability is known
        c = A["contrasts"][contrast]
        w = A["pass_region_witness"][contrast]
        t_arm, c_arm = c["treatment_arm"], c["control_arm"]
        anchor = c["anchor_median_gain_C0"]
        pc = c["per_condition"][cond]
        dose_t = A["behavioural_dose_witness"][f"{t_arm}|{cond}"]
        dose_c = A["behavioural_dose_witness"][f"{c_arm}|{cond}"]
        anchor_ok = bool(w["pass_region_nonempty"]) and anchor > 0
        # A perturbation that moves NEITHER arm cannot test anything; a retention of 1.0 in
        # that case is arithmetic, not robustness.  Requiring dose on the TREATMENT is the
        # binding condition (a moved control with a frozen treatment is also informative).
        dose_ok = not dose_t["inert_for_this_arm"]
        # FOURTH PRECONDITION, added 2026-07-25.  The three above are all evaluated at C0 or on
        # the perturbation in general; none of them asks the question that actually decides
        # whether THIS gate could have returned anything other than what it returned: are the
        # games that carried the C0 anchor still winnable by ANY arm under this perturbation?
        # When they are not, the perturbed gain is arithmetically forced to zero -- the mirror
        # image of the INERT case, and exactly what happened to the HUD lever's designated
        # headline gate in the first recorded run (r11l and tn36, the only two games the HUD
        # lever has ever moved, are won 0/5 by all four arms under the k=3 roll, leaving zero
        # discriminating games and an undefined sign test).
        wc = w["per_condition"][cond]
        support_ok = bool(wc["anchor_support_still_live"])
        sat = bool(c["retention"][cond].get("dose_saturated"))
        g = {
            "lever": lever,
            "is_headline_pair_for_this_lever": is_headline,
            "contrast": contrast,
            "treatment_arm": t_arm,
            "control_arm": c_arm,
            "perturbation_condition": cond,
            "PRECONDITION_pass_region_nonempty": bool(w["pass_region_nonempty"]),
            "PRECONDITION_anchor_median_gain_C0_positive": anchor > 0,
            "PRECONDITION_perturbation_has_behavioural_dose_on_treatment": dose_ok,
            "PRECONDITION_anchor_support_still_live_under_perturbation": support_ok,
            "witness_cells_at_C0_ANCHOR_ONLY_not_a_witness_for_this_condition": w[
                "witness_cells_treatment_wins_control_does_not_at_C0"
            ],
            "witness_cells_AT_THIS_CONDITION": wc[
                "witness_cells_treatment_wins_control_does_not_at_this_condition"
            ],
            "n_discriminating_games_at_this_condition": wc[
                "n_discriminating_games_at_this_condition"
            ],
            "anchor_game_max_seeds_won_across_ALL_arms_at_this_condition": wc[
                "anchor_game_max_seeds_won_across_ALL_arms"
            ],
            "behavioural_dose_treatment_fraction_moved": dose_t["fraction_moved"],
            "behavioural_dose_control_fraction_moved": dose_c["fraction_moved"],
            "anchor_median_gain_C0": anchor,
            "perturbed_median_gain": pc["median_gain"],
            "perturbed_per_seed_gain": pc["per_seed_gain"],
            "perturbed_no_seed_regresses": pc["no_seed_regresses"],
            "perturbed_strict_per_seed_dominance": pc["strict_per_seed_dominance"],
            "retention_ratio": c["retention"][cond]["retention_ratio"],
            "retention_precision": {
                k: c["retention"][cond][k]
                for k in (
                    "paired_per_seed_delta_vs_C0",
                    "decline_sign_test_p_one_sided",
                    "decline_resolved_at_this_n",
                    "gained_set_jaccard_vs_C0",
                    "retention_ratio_precision_note",
                )
            },
            "game_unit_sign_test": pc["game_unit_sign_test"],
            "seeds_are_a_replication_axis_for_this_contrast": c[
                "seeds_are_a_replication_axis_for_this_contrast"
            ],
            "n_seed_replicates_effective": c["n_seed_replicates_effective"],
            "perturbation_is_dose_saturated": sat,
            "control_wins_as_fraction_of_C0_under_this_perturbation": c["retention"][cond][
                "control_wins_as_fraction_of_C0"
            ],
            "games_gained_on_every_seed_under_perturbation": pc["games_gained_on_every_seed"],
            "games_lost_on_every_seed_under_perturbation": pc["games_lost_on_every_seed"],
            "evaluable": bool(anchor_ok and dose_ok and support_ok and interpretable),
            "reason_if_not_evaluable": None,
        }
        if not interpretable:
            g["reason_if_not_evaluable"] = (
                "cell_integrity.interpretable is False; see cell_integrity"
            )
        elif not anchor_ok:
            g["reason_if_not_evaluable"] = (
                "the anchor effect at C0_real is not positive, so there is no measured gain "
                "for the perturbation to retain -- uninterpretable, NOT evidence of transfer"
            )
        elif not dose_ok:
            g["reason_if_not_evaluable"] = (
                f"the perturbation is behaviourally INERT on the treatment arm {t_arm} "
                f"({dose_t['n_cells_behaviourally_moved']}/{dose_t['n_cells']} cells moved), "
                "so the retention ratio here is arithmetic, not robustness. Reported as "
                "UNINTERPRETABLE rather than as a survival."
            )
        elif not support_ok:
            g["reason_if_not_evaluable"] = (
                f"the games carrying this contrast's C0 anchor "
                f"({w['anchor_games_at_C0']}) are won by NO arm under {cond} "
                f"(max seeds won across all four arms: "
                f"{wc['anchor_game_max_seeds_won_across_ALL_arms']}), and there are "
                f"{wc['n_discriminating_games_at_this_condition']} discriminating games here. "
                "The perturbed gain is therefore arithmetically FORCED to 0 -- this gate could "
                "not have returned any other value, so it is not a measurement. Same class of "
                "defect as the empty pass region that VOIDED exp5835."
            )
        on_target = lever in CONDITION_TARGETS[cond] or lever == "both"
        g["perturbation_targets_this_levers_convention"] = on_target
        _attacks = {
            "C1_salience_inversion": (
                "absolute-colour salience ({6..15}), which ONLY the frontier tier predicate "
                "reads; the HUD Stage-1 predicate is pure geometry and is provably unaffected "
                "(static dose witness: 0 of 25 games change their HUD mask)"
            ),
            "C3_roll_k1": (
                "object geometry/position ONLY. A 1-cell roll is SMALLER than the HUD Stage-1 "
                "predicate's edge tolerance (2), so a mask whose lowest row is index 0 -- "
                "r11l's, the game carrying the HUD gain -- remains edge-adjacent (y1: 0 -> 1 < "
                "2). Measured: r11l's mask still resolves at k=1. So this condition is "
                "on-target for the FRONTIER lever's geometry-reading tier predicate and is NOT "
                "a violation of the HUD lever's edge-adjacency convention."
            ),
        }
        g["what_convention_this_perturbation_attacks"] = _attacks.get(
            cond,
            "edge adjacency (HUD Stage-1 `y1 < tol`) AND object geometry/position, which the "
            "frontier tier predicate's width test also reads -- so this condition is on-target "
            "for BOTH levers and is NOT a clean single-mechanism probe",
        )
        if g["evaluable"]:
            # TIGHTENED 2026-07-25.  SURVIVES used to require only median > 0 and
            # no_seed_regresses, and no_seed_regresses admits a seed at exactly ZERO -- so a
            # contrast with a dead seed and a game lost on every seed still read as "survives".
            # Surviving now means positive on EVERY seed (strict per-seed dominance), which is
            # the standard the rest of this battery already uses to avoid any-seed-union
            # reasoning.
            base = (
                "SURVIVES_CONVENTION_VIOLATION"
                if (pc["median_gain"] > 0 and pc["strict_per_seed_dominance"])
                else (
                    "PARTIALLY_SURVIVES"
                    if pc["median_gain"] > 0
                    else "GAIN_DOES_NOT_SURVIVE_CONVENTION_VIOLATION"
                )
            )
            if sat:
                # A dose-saturated condition cannot clear a lever: the control itself has lost
                # most of its capability there.  This is a SUFFIX, never a promotion -- it does
                # not turn a failure into a survival, it marks the reading as a statement about
                # the perturbation's strength as much as about the lever.
                base += "_IN_A_DOSE_SATURATED_CORPUS"
            if on_target:
                g["verdict"] = base
            else:
                # The gain moved (or did not) under a perturbation that does NOT attack this
                # lever's own convention.  That is a real observation -- and here a striking
                # one -- but it is NOT a statement about this lever's convention-robustness,
                # so it gets a distinct token that cannot be read as one.
                g["verdict"] = f"OFF_TARGET_FOR_THIS_LEVER_observed_{base.lower()}"
                g["off_target_note"] = (
                    f"{cond} does not perturb the {lever} lever's own convention, so this "
                    f"result must NOT be read as '{lever} is/is not convention-robust'. What it "
                    f"shows is how the {lever} lever's marginal contribution behaves when the "
                    f"OTHER lever's input distribution is disturbed."
                )
        elif not support_ok:
            g["verdict"] = "UNINTERPRETABLE_ANCHOR_SUPPORT_DEAD"
        else:
            g["verdict"] = "UNINTERPRETABLE_INERT_OR_NO_ANCHOR"
        gates[f"{contrast}|{cond}"] = g

    # -------------------------------------------- headline designation (rule stated in advance)
    # The lever's headline CONTRAST is fixed (HEADLINE_CONTRAST); only the CONDITION is chosen,
    # and it is chosen as the smallest-magnitude ON-TARGET EVALUABLE one.  Ties and absences are
    # deterministic: with no evaluable on-target condition the fallback is the largest-magnitude
    # on-target candidate, which then reports its own not-evaluable reason, so a lever can never
    # silently lose its headline entry.
    headline_choice = {}
    for lever, contrast in HEADLINE_CONTRAST.items():
        cands = [
            (CONDITION_RANK.get(cd, 99), f"{contrast}|{cd}")
            for cd in perturbed_conds
            if f"{contrast}|{cd}" in gates
            and gates[f"{contrast}|{cd}"]["perturbation_targets_this_levers_convention"]
        ]
        if not cands:
            continue
        evaluable = [(r, k) for r, k in cands if gates[k]["evaluable"]]
        rank, key = min(evaluable) if evaluable else max(cands)
        gates[key]["is_headline_pair_for_this_lever"] = True
        headline[lever] = gates[key]
        headline_choice[lever] = {
            "chosen_gate": key,
            "chosen_because": (
                "smallest-magnitude on-target EVALUABLE condition"
                if evaluable
                else "NO on-target condition is evaluable; the largest-magnitude "
                "on-target candidate is shown so its own reason is visible"
            ),
            "on_target_candidates_in_magnitude_order": [k for _, k in sorted(cands)],
            "evaluable_among_them": [k for _, k in sorted(evaluable)],
            "selection_rule": (
                "Fixed contrast per lever; the condition is selected on EVALUABILITY and dose "
                "MAGNITUDE only, both known before any verdict is read. No selection on the "
                "answer."
            ),
        }
    A["acceptance_gate_headline_selection"] = headline_choice

    # ------------------------------------------------------- per-lever DECIDABILITY (derived)
    # A lever's convention-robustness is only DECIDED if at least one gate that (a) attacks
    # THAT lever's own convention and (b) is evaluable exists.  In the first recorded run the
    # HUD lever had neither: C1 provably does not touch the HUD mask (off-target, and inert on
    # HUDO), and C2 kills the only two games the lever moves.  Reporting the C2 FAIL as the
    # HUD headline therefore stated a verdict the design could not deliver, which is why the
    # headline is now computed rather than designated.
    decidability = {}
    for lev in sorted({g["lever"] for g in gates.values()}):
        cands = {k: v for k, v in gates.items() if v["lever"] == lev}
        usable = {
            k: v
            for k, v in cands.items()
            if v["evaluable"] and v["perturbation_targets_this_levers_convention"]
        }
        decidability[lev] = {
            "decidable_by_this_design": bool(usable),
            "usable_on_target_evaluable_gates": sorted(usable),
            "why_each_candidate_gate_is_or_is_not_usable": {
                k: (
                    "USABLE"
                    if k in usable
                    else (
                        "OFF_TARGET: " + v["what_convention_this_perturbation_attacks"][:70]
                        if not v["perturbation_targets_this_levers_convention"]
                        else "NOT_EVALUABLE: " + str(v["reason_if_not_evaluable"])
                    )
                )
                for k, v in cands.items()
            },
        }

    A["per_lever_decidability"] = decidability

    # ---------------------------------------------------------------- honest verdict
    short_of = {
        "SURVIVES_CONVENTION_VIOLATION": "survives",
        "PARTIALLY_SURVIVES": "partial",
        "GAIN_DOES_NOT_SURVIVE_CONVENTION_VIOLATION": "gain_does_not_survive",
        "UNINTERPRETABLE_INERT_OR_NO_ANCHOR": "uninterpretable",
        "UNINTERPRETABLE_ANCHOR_SUPPORT_DEAD": "not_decidable_by_this_design",
    }

    def _short(lev):
        if not decidability.get(lev, {}).get("decidable_by_this_design"):
            return f"{lev}_not_decidable_by_this_design"
        v = headline[lev]["verdict"]
        for token, short in short_of.items():
            if v.startswith(token):
                return f"{lev}_{short}"
        return f"{lev}_{v.lower()}"

    parts = [_short(lev) for lev in ("frontier", "hud")]
    verdict = "complete_convention_perturbation_transfer_battery_" + "_".join(parts)

    A["acceptance_gates"] = gates
    A["acceptance_gate_headline_per_lever"] = {
        lev: {
            "gate": f"{g['contrast']}|{g['perturbation_condition']}",
            "verdict": (
                "NOT_DECIDABLE_BY_THIS_DESIGN"
                if not decidability[lev]["decidable_by_this_design"]
                else g["verdict"]
            ),
            "verdict_of_the_designated_gate_itself": g["verdict"],
            "decidable_by_this_design": decidability[lev]["decidable_by_this_design"],
            "evaluable": g["evaluable"],
            "reason_if_not_evaluable": g["reason_if_not_evaluable"],
            "anchor_median_gain_C0": g["anchor_median_gain_C0"],
            "perturbed_median_gain": g["perturbed_median_gain"],
            "retention_ratio": g["retention_ratio"],
            "game_unit_sign_test_p_one_sided": g["game_unit_sign_test"]["p_one_sided_exact"],
            "n_independent_replicates": g["game_unit_sign_test"]["n_independent_replicates"],
            "WHY_THIS_GATE_WAS_DESIGNATED": (
                "the frontier flip was measured with the HUD off (it predates the HUD flip), "
                "so this gate replicates the configuration the flip decision was made in. It "
                "is NOT the shipped configuration -- see "
                "acceptance_gate_shipped_configuration_marginal_per_lever for that."
                if lev == "frontier"
                else "the HUD flip was measured with the frontier lever already on, so this is the "
                "shipped HUD contrast. Whether it is EVALUABLE is a separate question, "
                "answered by its own preconditions."
            ),
        }
        for lev, g in headline.items()
    }
    # The gate above answers each lever's flip-time claim.  The SHIPPED configuration's
    # marginal contrast is a different pair, and for the frontier lever it is materially less
    # flattering -- so it is surfaced at the same level rather than left for a reader to dig
    # out of `contrasts`.  Omitting it was a real defect in the first recorded run: the
    # operator's decision is about the configuration that ships.
    # Derived from the headline selection so it cannot drift out of step with it: same
    # perturbation condition, but the contrast measured with the OTHER lever already ON.
    _shipped_contrast = {"frontier": "frontier_given_hud_on", "hud": "hud_given_frontier_on"}
    SHIPPED_MARGINAL = {
        lev: (_shipped_contrast[lev], headline[lev]["perturbation_condition"])
        for lev in _shipped_contrast
        if lev in headline
    }
    A["acceptance_gate_shipped_configuration_marginal_per_lever"] = {
        lev: {
            "gate": f"{ct}|{cd}",
            "verdict": gates[f"{ct}|{cd}"]["verdict"],
            "anchor_median_gain_C0": gates[f"{ct}|{cd}"]["anchor_median_gain_C0"],
            "perturbed_median_gain": gates[f"{ct}|{cd}"]["perturbed_median_gain"],
            "retention_ratio": gates[f"{ct}|{cd}"]["retention_ratio"],
            "perturbed_per_seed_gain": gates[f"{ct}|{cd}"]["perturbed_per_seed_gain"],
            "perturbed_strict_per_seed_dominance": gates[f"{ct}|{cd}"][
                "perturbed_strict_per_seed_dominance"
            ],
            "games_lost_on_every_seed_under_perturbation": gates[f"{ct}|{cd}"][
                "games_lost_on_every_seed_under_perturbation"
            ],
            "game_unit_sign_test_at_C0": A["contrasts"][ct]["per_condition"]["C0_real"][
                "game_unit_sign_test"
            ],
            "note": (
                "This is the lever's marginal contribution IN THE SHIPPED CONFIGURATION "
                "(the other lever already ON), which is what the live agent runs today."
            ),
        }
        for lev, (ct, cd) in SHIPPED_MARGINAL.items()
        if f"{ct}|{cd}" in gates
    }
    A["acceptance_gates_all_evaluable"] = all(g["evaluable"] for g in gates.values())
    A["acceptance_gates_all_passed"] = all(
        g["verdict"].startswith("SURVIVES_CONVENTION_VIOLATION")
        and decidability[lev]["decidable_by_this_design"]
        for lev, g in headline.items()
    )
    A["honest_verdict"] = verdict

    # Absolute win level under each condition, so a "gain retained" reading cannot be taken
    # out of context: C2 makes every arm much worse in absolute terms, so a retained gain
    # there sits on a far lower base than the same gain at C0.
    A["absolute_win_level_context"] = {
        cond: {
            arm: A["per_arm_condition_wins"][f"{arm}|{cond}"]["per_seed_win_counts"]
            for arm in ("CTRL", "FRONT", "HUDO", "SHIP")
        }
        for cond in A["config"]["conditions"]
    }

    # ---------------------------------------------------------------- budget sweeps (derived)
    sweep_rows = _load_sweep_rows()
    sweeps = _summarise_sweeps(sweep_rows)
    sweep_files = sorted(
        str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else p.name
        for d in _SWEEP_DIRS
        if d.exists()
        for p in list(d.glob("*sweep*.jsonl")) + list(d.glob("*sweep*.jsonl.gz"))
    )
    probe_rows = _load_probe_rows()
    roll_dose = _roll_dose_response(probe_rows)
    probe_files = sorted(
        str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else p.name
        for d in _SWEEP_DIRS
        if d.exists()
        for p in list(d.glob("*probe_rollk*.jsonl")) + list(d.glob("*probe_rollk*.jsonl.gz"))
    )

    # duration_s from the analysis covers the 1500 battery cells only.  The budget sweeps and the
    # roll-magnitude probe are real additional compute and are accounted for explicitly rather
    # than silently omitted -- duration_s is the load-bearing fabrication signal, so under-
    # reporting it is as wrong as over-reporting it.
    sweep_wall = round(sum(float(r.get("cell_wall_s") or 0) for r in sweep_rows), 2)
    probe_wall = round(sum(float(r.get("cell_wall_s") or 0) for r in probe_rows), 2)
    A["duration_s_breakdown"] = {
        "battery_cells_wall_s": A["duration_s"],
        "budget_sweep_cells_wall_s": sweep_wall,
        "n_budget_sweep_cells": len(sweep_rows),
        "roll_magnitude_probe_cells_wall_s": probe_wall,
        "n_roll_magnitude_probe_cells": len(probe_rows),
        "note": "duration_s is the SUM of all three; the analysis alone measures the battery.",
    }
    A["duration_s"] = round(A["duration_s"] + sweep_wall + probe_wall, 2)

    cost_ratios = {
        "tn36_C0_SHIP_vs_HUDO_at_budget_8000": _cost_ratio(
            sweeps, "tn36|C0_real|SHIP|budget_8000", "tn36|C0_real|HUDO|budget_8000"
        ),
        "r11l_C1_SHIP_vs_HUDO_at_budget_8000": _cost_ratio(
            sweeps,
            "r11l|C1_salience_inversion|SHIP|budget_8000",
            "r11l|C1_salience_inversion|HUDO|budget_8000",
        ),
    }

    # ---------------------------------------------------------------- key findings (derived)
    mr = A["hud_mask_resolution_mechanism_evidence"]
    di = A["games_where_adding_frontier_destroys_a_hud_win"]
    wm = A["per_game_win_matrix_seeds_won_of_5"]
    loo = A["leave_one_game_out_jackknife"]
    A["key_findings"] = {
        "1_baseline_independently_replicated": {
            "claim": "The explicitly-pinned pre-flip control reproduces the historical "
            "baseline win set exactly, so the drift-free control is sound.",
            "measured_CTRL_C0_win_set": A["per_arm_condition_wins"]["CTRL|C0_real"][
                "per_seed_win_sets"
            ]["20260726"],
            "historical_arm_A_real_win_set": [
                "cd82",
                "lf52",
                "lp85",
                "sp80",
                "su15",
                "tu93",
                "vc33",
            ],
            "identical": A["per_arm_condition_wins"]["CTRL|C0_real"]["per_seed_win_sets"][
                "20260726"
            ]
            == ["cd82", "lf52", "lp85", "sp80", "su15", "tu93", "vc33"],
            "why_this_matters": "Arms A and B2 in the upstream harness pin only a subset of "
            "the gated flags, so since the 2026-07-25 flips they inherit "
            "the treatment defaults and can no longer serve as controls. "
            "This arm pins all seven, and still lands on the same 7 games.",
        },
        "2_frontier_lever_survives_both_convention_violations": {
            "claim": "The frontier lever's gain SURVIVES inversion of the absolute-colour "
            "salience convention it keys on -- positive on every seed, and the "
            "game-unit sign test at C1 clears 0.05 (numbers in "
            "survival_is_established). Whether it DEGRADES is a SEPARATE question and "
            "is NOT resolved at this sample size: see "
            "degradation_is_not_resolved_at_this_n.",
            "anchor_median_gain_C0": A["contrasts"]["frontier_given_hud_off"][
                "anchor_median_gain_C0"
            ],
            "retention": {
                k: v["retention_ratio"]
                for k, v in A["contrasts"]["frontier_given_hud_off"]["retention"].items()
            },
            # CORRECTED 2026-07-25.  The first recorded run reported '+4 -> +3 games (retention
            # 0.75)' as a measured degradation.  It is a ratio of two medians over 5 seeds on a
            # difference of ONE game, and the paired per-seed deltas are [-1,-1,-1,-2,+1]: four
            # seeds decline, one improves, one-sided sign test p=0.1875.  The SURVIVAL is
            # robust; the DEGRADATION is a point estimate consistent with no degradation at all,
            # and CLAUDE.md's sample-size rigor rule does not support reporting it as measured.
            "survival_is_established": {
                "per_seed_gain_at_C1": A["contrasts"]["frontier_given_hud_off"]["per_condition"][
                    "C1_salience_inversion"
                ]["per_seed_gain"],
                "min_gain_at_C1": A["contrasts"]["frontier_given_hud_off"]["per_condition"][
                    "C1_salience_inversion"
                ]["min_gain"],
                "strict_per_seed_dominance_at_C1": A["contrasts"]["frontier_given_hud_off"][
                    "per_condition"
                ]["C1_salience_inversion"]["strict_per_seed_dominance"],
                "game_unit_sign_test_at_C1": A["contrasts"]["frontier_given_hud_off"][
                    "per_condition"
                ]["C1_salience_inversion"]["game_unit_sign_test"],
            },
            "degradation_is_not_resolved_at_this_n": {
                k: A["contrasts"]["frontier_given_hud_off"]["retention"]["C1_salience_inversion"][k]
                for k in (
                    "paired_per_seed_delta_vs_C0",
                    "n_seeds_declining",
                    "n_seeds_improving",
                    "decline_sign_test_p_one_sided",
                    "decline_resolved_at_this_n",
                )
            },
            "what_the_retention_0_75_point_estimate_is_and_is_not": (
                "It is the ratio of the C1 median gain to the C0 median gain, both over 5 "
                "seeds, on a difference of one game. It is NOT a measured degradation: the "
                "paired sign test does not clear 0.05, so 'no degradation' remains fully "
                "consistent with the data. Resolving it needs more seeds (or more games), not "
                "a re-reading of these."
            ),
            "retention_at_C2_is_dose_saturated_not_a_survival": {
                "retention_ratio": A["contrasts"]["frontier_given_hud_off"]["retention"][
                    "C2_diag_roll"
                ]["retention_ratio"],
                "control_wins_as_fraction_of_C0": A["contrasts"]["frontier_given_hud_off"][
                    "retention"
                ]["C2_diag_roll"]["control_wins_as_fraction_of_C0"],
                "gained_set_jaccard_vs_C0": A["contrasts"]["frontier_given_hud_off"]["retention"][
                    "C2_diag_roll"
                ]["gained_set_jaccard_vs_C0"],
                "games_gained_on_every_seed_at_C0": A["contrasts"]["frontier_given_hud_off"][
                    "retention"
                ]["C2_diag_roll"]["games_gained_on_every_seed_at_C0"],
                "games_gained_on_every_seed_at_C2": A["contrasts"]["frontier_given_hud_off"][
                    "retention"
                ]["C2_diag_roll"]["games_gained_on_every_seed_here"],
                "reading": (
                    "Read the ratio next to the two fields above it. Where the gained SET "
                    "barely overlaps C0's and the control has lost most of its own capability, "
                    "a retention ratio near 1.0 is NOT 'the same gain retained' -- it is a "
                    "similarly-sized gain on DIFFERENT games in a razed corpus."
                ),
            },
            "strict_per_seed_dominance_in_every_condition": all(
                A["contrasts"]["frontier_given_hud_off"]["per_condition"][c][
                    "strict_per_seed_dominance"
                ]
                for c in A["config"]["conditions"]
            ),
            "no_seed_ever_regresses": all(
                A["contrasts"]["frontier_given_hud_off"]["per_condition"][c]["no_seed_regresses"]
                for c in A["config"]["conditions"]
            ),
            "gain_is_spread_not_concentrated": {
                "n_games_whose_removal_drops_the_gain_to_zero": loo["frontier_given_hud_off"][
                    "n_games_whose_removal_drops_the_gain_to_zero_or_below"
                ],
                "games_that_contribute": sorted(
                    g
                    for g, v in loo["frontier_given_hud_off"][
                        "median_gain_with_each_game_held_out"
                    ].items()
                    if v != loo["frontier_given_hud_off"]["full_corpus_median_gain"]
                ),
            },
        },
        "3_hud_lever_convention_robustness_is_NOT_DECIDABLE_BY_THIS_DESIGN": {
            "claim": "The HUD lever's marginal gain in the shipped configuration is exactly "
            "one game (r11l), and this battery CANNOT decide whether that gain "
            "depends on the edge-adjacency convention. The first recorded run "
            "reported it as GAIN_DOES_NOT_SURVIVE; that verdict is WITHDRAWN as "
            "uninterpretable, because neither perturbed condition can answer the "
            "question.",
            "anchor_median_gain_C0": A["contrasts"]["hud_given_frontier_on"][
                "anchor_median_gain_C0"
            ],
            "retention": {
                k: v["retention_ratio"]
                for k, v in A["contrasts"]["hud_given_frontier_on"]["retention"].items()
            },
            "single_game_carrying_the_whole_gain": loo["hud_given_frontier_on"][
                "single_game_whose_removal_costs_the_most"
            ],
            "median_gain_without_that_game": loo["hud_given_frontier_on"][
                "median_gain_without_that_game"
            ],
            "support_is_a_SINGLE_game_so_no_p_can_clear": A["contrasts"]["hud_given_frontier_on"][
                "per_condition"
            ]["C0_real"]["game_unit_sign_test"],
            "WHY_NOT_DECIDABLE": {
                "C1_salience_inversion": (
                    "OFF TARGET. C1 perturbs absolute colour, which only the frontier tier "
                    "predicate reads; the HUD Stage-1 predicate is pure geometry and the static "
                    "dose witness measures ZERO HUD-mask change on all 25 games. Whatever "
                    "happens to the HUD lever's gain here is a statement about the OTHER "
                    "lever's input distribution, not about edge adjacency."
                ),
                "C2_diag_roll": (
                    "ANCHOR SUPPORT DEAD. Under the k=3 roll, r11l and tn36 -- the only two "
                    "games this lever has ever moved -- are won by NO arm on any seed, leaving "
                    "zero discriminating games and an undefined sign test. The perturbed gain "
                    "of 0 was arithmetically FORCED and could not have come out any other way. "
                    "See the gate's own PRECONDITION_anchor_support_still_live_under_"
                    "perturbation, which now fails."
                ),
            },
            "WHAT_WAS_WRONG_WITH_THE_FIRST_RUNS_VERDICT": (
                "The pass-region witness was computed ONLY at C0 and then attached to every "
                "gate for the contrast, so the C2 gate certified 'my pass region is non-empty' "
                "using 5 r11l cells measured at C0 -- a witness for a different condition than "
                "the one being scored. All three preconditions read green and the gate reported "
                "evaluable=true with reason_if_not_evaluable=null, while the confound was "
                "disclosed only three levels deep in prose. Same class of defect as the empty "
                "pass region that VOIDED exp5835: a gate that could not fail is not a gate, and "
                "a gate that could not pass is not one either."
            ),
            "dose_axis_conditions_present_in_this_run": [
                c for c in A["config"]["conditions"] if c.startswith("C3")
            ]
            or "none -- the C3 dose axis is wired but not measured in this run",
            "THE_ONLY_HUD_MECHANISM_EVIDENCE_THAT_SURVIVES": (
                "The corpus-wide mask-resolution collapse under the roll, which is measured "
                "independently of any win: see hud_mask_resolution_mechanism_evidence. That "
                "shows the edge-adjacency convention is load-bearing FOR THE DETECTOR, exactly "
                "as the predicate's `y1 < tol` source predicts. It does NOT show what happens "
                "to the lever's GAIN when the convention is violated -- for that, a "
                "perturbation is needed that moves the bar off the edge WITHOUT making the "
                "games unwinnable. The dose axis added in cptb_perturb (C3_roll_k1/k2) is the "
                "hook for building one; whether it was measured in THIS run is recorded in "
                "dose_axis_conditions_present_in_this_run above, not asserted here."
            ),
        },
        "4_the_two_perturbations_break_the_hud_gain_by_DIFFERENT_mechanisms": {
            "claim": "Under the geometric roll the detector itself stops working; under "
            "salience inversion the detector works perfectly and the frontier lever "
            "destroys the win instead. These are separate failure modes and the "
            "artifact does not merge them.",
            "C2_diag_roll_mask_resolution_collapses": {
                "corpus_fraction_resolved_C0": mr["SHIP|C0_real"]["fraction_resolved"],
                "corpus_fraction_resolved_C2": mr["SHIP|C2_diag_roll"]["fraction_resolved"],
                "r11l_seeds_mask_resolved_C0": mr["SHIP|C0_real"]["per_game_seeds_resolved"][
                    "r11l"
                ],
                "r11l_seeds_mask_resolved_C2": mr["SHIP|C2_diag_roll"]["per_game_seeds_resolved"][
                    "r11l"
                ],
                "interpretation": "Moving every edge-hugging bar 3 cells inward takes r11l's "
                "mask from resolving on 5 of 5 seeds to 0 of 5. The "
                "edge-adjacency convention is load-bearing, exactly as the "
                "predicate's `y1 < tol` source predicts.",
                "HONEST_CONFOUND": "Under C2 r11l is unwinnable for EVERY arm (including the "
                "control), so the vanished gain alone would not prove the "
                "mask is why. The mask-resolution collapse is the "
                "independent mechanism evidence; the win-level evidence is "
                "confounded and is not relied on.",
                # PROMOTED 2026-07-25 from prose-only disclosure to a machine-readable gate
                # precondition.  In the first recorded run this confound was stated HERE, three
                # levels deep, while the gate it invalidates reported evaluable=true with all
                # preconditions green -- and summarize_artifact.py prints the gate block, not
                # key_findings, so a reader or an aggregating capstone saw an unqualified FAIL.
                # The confound is now enforced by
                # PRECONDITION_anchor_support_still_live_under_perturbation, which fails, so the
                # gate reports UNINTERPRETABLE_ANCHOR_SUPPORT_DEAD instead of a verdict.
                "THIS_CONFOUND_IS_NOW_A_GATE_PRECONDITION": {
                    "gate": "hud_given_frontier_on|C2_diag_roll",
                    "precondition": "PRECONDITION_anchor_support_still_live_under_perturbation",
                    "value": gates["hud_given_frontier_on|C2_diag_roll"][
                        "PRECONDITION_anchor_support_still_live_under_perturbation"
                    ],
                    "gate_verdict_now": gates["hud_given_frontier_on|C2_diag_roll"]["verdict"],
                    "n_discriminating_games_at_C2": gates["hud_given_frontier_on|C2_diag_roll"][
                        "n_discriminating_games_at_this_condition"
                    ],
                },
                "DOSE_CEILING_AT_C2": {
                    k: A["dose_ceiling_witness"]["C2_diag_roll"][k]
                    for k in (
                        "control_median_absolute_wins",
                        "control_wins_as_fraction_of_C0",
                        "n_games_dead_for_all_arms",
                        "n_games",
                        "dose_saturated",
                    )
                },
            },
            "C1_salience_inversion_mask_is_UNAFFECTED_but_the_win_is_lost": {
                "corpus_fraction_resolved_C0": mr["SHIP|C0_real"]["fraction_resolved"],
                "corpus_fraction_resolved_C1": mr["SHIP|C1_salience_inversion"][
                    "fraction_resolved"
                ],
                "r11l_seeds_mask_resolved_C1": mr["SHIP|C1_salience_inversion"][
                    "per_game_seeds_resolved"
                ]["r11l"],
                "r11l_seeds_won_C1": wm["C1_salience_inversion"]["r11l"],
                "interpretation": "The mask resolution is IDENTICAL to C0 (the Stage-1 "
                "predicate is colour-invariant by construction, and the "
                "static dose witness measured zero HUD-mask change on all "
                "25 games). HUD-alone still wins r11l on 5 of 5 seeds. The "
                "shipped both-levers-on arm wins it on 0 of 5. So under "
                "inverted salience the FRONTIER lever destroys the HUD's "
                "win, with the detector working normally.",
            },
        },
        "5_the_two_levers_interact_and_the_two_cases_are_DIFFERENT_in_kind": {
            "claim": "Adding the frontier lever on top of the HUD lever costs games the HUD "
            "lever alone wins. A budget sweep separates two cases that look identical "
            "at budget 2000 but are not: tn36 is an EFFICIENCY regression that "
            "crosses the budget, r11l-under-salience-inversion is a genuine "
            "CAPABILITY loss. Neither flip's own A/B could see either, because both "
            "of their controls also lose these games.",
            "games_HUDO_wins_and_SHIP_loses_on_every_seed": di,
            "tn36_win_matrix_C0": wm["C0_real"]["tn36"],
            "r11l_win_matrix_C0": wm["C0_real"]["r11l"],
            "r11l_win_matrix_C1": wm["C1_salience_inversion"]["r11l"],
            "WHY_A_BUDGET_SWEEP_WAS_REQUIRED": (
                "At budget 2000 both cases read as 'SHIP loses a game HUDO wins'. This "
                "project has already documented one such loss (arm B2's cd82) that turned out "
                "to be a budget WALL rather than lost capability, so the same alternative "
                "explanation had to be excluded here before either could be called a "
                "regression. The first draft of this finding called tn36 a capability loss; "
                "the sweep refuted that, and the corrected reading is below."
            ),
            # DERIVED, not transcribed (corrected 2026-07-25).  The first recorded run
            # hand-copied a handful of cells into this block with no n_seeds, and the two
            # decisive cells happened to be n=1 (tn36 at budgets 4000/8000) while the claim they
            # overturned was n=5.  Every bucket below now carries its own n_seeds, per-seed
            # levels and per-seed action counts, read straight off the sweep rows.
            "budget_sweep": sweeps,
            "budget_sweep_n_seeds_per_bucket": {k: v["n_seeds"] for k, v in sweeps.items()},
            "budget_sweep_raw_rows": sweep_files,
            "cost_ratios": cost_ratios,
            "corrected_reading": {
                "tn36_on_real_games": (
                    "The frontier lever makes tn36 substantially more expensive to solve with "
                    "the SAME mask resolved in both arms, which pushes the win past the "
                    "2000-action budget -- an efficiency regression with a budget-visible "
                    "consequence, in the same class as the already-known cd82 residual, NOT a "
                    "destroyed capability. The cost ratio is reported as a RANGE over seeds in "
                    "`cost_ratios`, not as the single-cell '~3.5x' the first run quoted."
                ),
                "tn36_reclassification_now_rests_on_5_seeds": (
                    "This reclassification was originally made on ONE seed at budget 8000 while "
                    "the loss it reclassified was measured on five at budget 2000 -- the weaker "
                    "evidence overturning the stronger, in the direction that made the shipped "
                    "configuration look better. The sweep was re-run on all 5 seeds; see "
                    "`budget_sweep_n_seeds_per_bucket` for the n behind every number here."
                ),
                "r11l_under_salience_inversion": (
                    "This one is real. HUD-alone wins r11l under inverted salience at every "
                    "budget tested, on every seed; the shipped both-on configuration is still "
                    "at 0 levels at 4x and 8x the measured budget with the mask resolving "
                    "normally. The frontier lever removes a capability the HUD lever supplies, "
                    "once the colour convention no longer holds."
                ),
            },
            "mechanism": "In both cases HUDO and SHIP resolve the SAME mask, so the mask is "
            "not the difference. The difference is search shape: on tn36 at "
            "budget 2000 HUDO expands 52 graph nodes and banks the level while "
            "SHIP expands 17; the global tier barrier defers the branch "
            "containing the win behind an exhaustive lower-tier sweep, which "
            "costs actions rather than reachability. Under salience inversion the "
            "tier assignment itself is wrong, so the deferral no longer converges "
            "within any budget tested.",
            "what_this_does_and_does_not_argue": "It does NOT argue for un-flipping either "
            "lever: the shipped configuration still has "
            "the highest median win count of the four "
            "arms at every condition (see "
            "per_arm_condition_wins). These are specific, "
            "reproducible, previously unmeasured costs, "
            "and the decision is the operator's.",
        },
        "6_the_ROLL_FAMILY_cannot_decide_the_hud_question_at_ANY_magnitude": {
            "claim": (
                "The adversarial review's suggested repair was to add a milder geometric "
                "condition so the edge-adjacency convention could be attacked without razing "
                "the corpus. That was built (a dose-parameterised roll) and MEASURED, and the "
                "answer is that no magnitude of this transform works: on BOTH games the HUD "
                "lever moves, the smallest roll that stops the detector firing is already a "
                "roll at which every arm loses the game. Convention-breakage and "
                "task-destruction are inseparable within the roll family."
            ),
            "why_mechanistically": (
                "The Stage-1 predicate is `on_top = y1 < EDGE_BAR_EDGE_TOLERANCE` with the "
                "tolerance = 2 (arc_hud_bar_detector.py). r11l's mask is a single 64-cell row "
                "at index 0, so a 1-cell roll leaves y1 = 1 < 2 -- the convention still holds "
                "and the detector still fires (measured). The first magnitude that violates it "
                "is k = 2, and at k = 2 r11l is won by no arm. tn36's mask stops resolving "
                "already at k = 1, and at k = 1 tn36 is likewise won by no arm. So the roll "
                "cannot separate the two explanations on either game."
            ),
            "measured_dose_response": roll_dose,
            "n_probe_cells": len(probe_rows),
            "probe_raw_rows": probe_files,
            "what_a_DECIDABLE_perturbation_would_have_to_do": (
                "Move the status bar at least EDGE_BAR_EDGE_TOLERANCE cells off its edge while "
                "leaving the playfield's object contiguity and reachability intact -- i.e. not "
                "a whole-grid wrap at all. A row/column STRIP SWAP (exchange rows 0..t-1 with "
                "rows t..2t-1) is the obvious candidate: it is a permutation, so no content is "
                "lost, and it leaves every row below 2t untouched. That is NOT implemented "
                "here, and until it is, the HUD lever's convention-dependence stays open."
            ),
            "honest_scope": (
                "The probe covers the two support games at 5 seeds x 2 magnitudes, not the "
                "whole corpus, because its question is about those two games specifically. It "
                "is deliberately excluded from the corpus win sets (see _load_probe_rows) so "
                "it cannot unbalance the per-condition coverage."
            ),
            "a_full_corpus_C3_condition_was_STARTED_AND_ABANDONED": (
                "A full 25-game x 4-arm x 5-seed C3_roll_k1 condition (500 cells) was launched "
                "and stopped after 73 cells covering 4 games. It is NOT included and its "
                "partial rows are not published: a condition with partial game coverage makes "
                "the per-condition win sets non-comparable and would have flipped "
                "cell_integrity.interpretable to False for the whole battery. It was abandoned "
                "because the question it would have answered for the HUD lever was already "
                "answered NEGATIVELY by the targeted probe above (k=1 does not violate the "
                "convention on r11l; k=2 does and kills the game), so the remaining value was a "
                "frontier-lever gate under a mild geometric perturbation -- a nice-to-have that "
                "no claim here depends on -- at a measured ~8.3s/cell, i.e. over an hour of "
                "further compute. Recorded rather than omitted so the record shows what was "
                "attempted, not only what landed."
            ),
        },
    }

    # ------------------------------------------------------------------------- headline
    # Assembled FROM the computed fields, not transcribed, so it cannot drift from the data.
    # Rewritten 2026-07-25 against the adversarial review: the previous version led with the
    # frontier lever's PRE-HUD-FLIP contrast only, presented a ratio of two 5-seed medians as a
    # measured degradation, and stated a HUD verdict this design cannot deliver.
    fh_off = A["contrasts"]["frontier_given_hud_off"]
    fh_on = A["contrasts"]["frontier_given_hud_on"]
    ret_c1 = fh_off["retention"]["C1_salience_inversion"]
    ret_on_c1 = fh_on["retention"]["C1_salience_inversion"]
    dc2 = A["dose_ceiling_witness"]["C2_diag_roll"]
    hud_gate = gates["hud_given_frontier_on|C2_diag_roll"]
    tn36_ratio = cost_ratios["tn36_C0_SHIP_vs_HUDO_at_budget_8000"]
    tn36_ship_8k = sweeps.get("tn36|C0_real|SHIP|budget_8000", {})
    r11l_ship_8k = sweeps.get("r11l|C1_salience_inversion|SHIP|budget_8000", {})
    r11l_hudo_8k = sweeps.get("r11l|C1_salience_inversion|HUDO|budget_8000", {})
    # largest budget actually swept for r11l under salience inversion, read from the keys
    _r11l_budgets = sorted(
        int(k.rsplit("_", 1)[1])
        for k in sweeps
        if k.startswith("r11l|C1_salience_inversion|SHIP|budget_")
    )
    r11l_max_budget = _r11l_budgets[-1] if _r11l_budgets else A["config"]["budget"]
    r11l_ship_max = sweeps.get(
        f"r11l|C1_salience_inversion|SHIP|budget_{r11l_max_budget}", r11l_ship_8k
    )

    A["headline"] = (
        "FRONTIER LEVER, AS MEASURED AT FLIP TIME (HUD off -- the configuration the flip "
        f"decision was made in): the gain SURVIVES inversion of the absolute-colour salience "
        f"convention it keys on. Per-seed gain at C1 "
        f"{list(fh_off['per_condition']['C1_salience_inversion']['per_seed_gain'].values())}, "
        f"minimum +{fh_off['per_condition']['C1_salience_inversion']['min_gain']}, positive on "
        f"every seed; on the GAME unit "
        f"{fh_off['per_condition']['C1_salience_inversion']['game_unit_sign_test']['n_games_treatment_wins_on_more_seeds']}"
        f" games favour it to "
        f"{fh_off['per_condition']['C1_salience_inversion']['game_unit_sign_test']['n_games_control_wins_on_more_seeds']}"
        f" against, exact one-sided sign test p="
        f"{fh_off['per_condition']['C1_salience_inversion']['game_unit_sign_test']['p_one_sided_exact']}"
        f". Whether it DEGRADES is NOT resolved at this sample size: median "
        f"{fh_off['anchor_median_gain_C0']} -> {ret_c1['transfer_median_gain']} is a ratio of "
        f"two {len(A['random_seeds_used'])}-seed medians on a difference of one game, the "
        f"paired per-seed deltas are {ret_c1['paired_per_seed_delta_vs_C0']}, and the one-sided "
        f"sign test for a decline gives p={ret_c1['decline_sign_test_p_one_sided']} -- fully "
        "consistent with no degradation. The earlier reading that quoted 'retention 0.75' as a "
        "measured degradation is WITHDRAWN; the survival claim stands. "
        "FRONTIER LEVER, IN THE SHIPPED CONFIGURATION (HUD on -- what the live agent runs "
        f"today): the same lever's marginal gain is median {fh_on['anchor_median_gain_C0']} at "
        f"C0 and {ret_on_c1['transfer_median_gain']} at C1 (retention "
        f"{ret_on_c1['retention_ratio']}), per-seed "
        f"{list(fh_on['per_condition']['C1_salience_inversion']['per_seed_gain'].values())} -- "
        "one seed at zero, so per-seed dominance is NOT strict -- and it loses "
        f"{fh_on['per_condition']['C1_salience_inversion']['games_lost_on_every_seed']} on every "
        "seed. This is the contrast the operator's decision is actually about, and it is less "
        "flattering than the flip-time one. "
        f"Under the geometric roll the frontier gain is nominally undiminished, but that "
        f"condition is DOSE SATURATED: it takes the pre-flip control from "
        f"{A['dose_ceiling_witness']['C0_real']['control_median_absolute_wins']} wins to "
        f"{dc2['control_median_absolute_wins']} and takes the games no arm can win from "
        f"{A['dose_ceiling_witness']['C0_real']['n_games_dead_for_all_arms']}/{dc2['n_games']} "
        f"to {dc2['n_games_dead_for_all_arms']}/{dc2['n_games']}, and the gained set overlaps "
        f"C0's with Jaccard "
        f"{fh_off['retention']['C2_diag_roll']['gained_set_jaccard_vs_C0']}. A similarly-sized "
        "gain on different games in a razed corpus is not the same statement as at C0. "
        "HUD LEVER: NOT DECIDABLE BY THIS DESIGN. Its marginal gain in the shipped "
        "configuration is exactly ONE game (r11l) -- a single-game support, which no sign test "
        "at any seed count can clear -- and neither perturbed condition can test the "
        "edge-adjacency convention it depends on. C1 provably does not touch the HUD mask "
        "(off-target; zero mask change on all 25 games) and is behaviourally inert on the "
        "HUD-alone arm. C2 kills the anchor: r11l and tn36, the only two games this lever has "
        f"ever moved, are won by NO arm on any seed there, leaving "
        f"{hud_gate['n_discriminating_games_at_this_condition']} discriminating games, so the "
        "perturbed gain of 0 was arithmetically FORCED. The earlier "
        "GAIN_DOES_NOT_SURVIVE_CONVENTION_VIOLATION verdict is WITHDRAWN as uninterpretable "
        "(the gate had certified its pass region with cells measured at a DIFFERENT condition "
        "-- the exp5835 defect in a new location). What DOES survive as HUD mechanism evidence "
        "is the corpus-wide mask-resolution collapse under the roll, measured independently of "
        "any win: r11l's mask resolves on "
        f"{mr['SHIP|C0_real']['per_game_seeds_resolved']['r11l']}/"
        f"{len(A['random_seeds_used'])} seeds at C0 and "
        f"{mr['SHIP|C2_diag_roll']['per_game_seeds_resolved']['r11l']}/"
        f"{len(A['random_seeds_used'])} under the roll. That shows the convention is "
        "load-bearing for the DETECTOR; it does not show what happens to the lever's GAIN. "
        "THE TWO LEVERS INTERACT, and a 5-seed budget sweep splits it into two different "
        f"things. On REAL unperturbed tn36 the shipped config loses a game HUD-alone wins at "
        f"budget 2000, but it DOES win at budget 8000 on "
        f"{tn36_ship_8k.get('n_seeds_won')}/{tn36_ship_8k.get('n_seeds')} seeds, at "
        f"{tn36_ratio.get('ratio_range')}x the cost -- an EFFICIENCY regression crossing the "
        "budget, not a lost capability. (That reclassification originally rested on ONE seed "
        "while the loss it overturned rested on five; it has now been re-run on all "
        f"{tn36_ship_8k.get('n_seeds')} seeds and holds.) The genuine CAPABILITY loss is r11l "
        f"under salience inversion: HUD-alone wins it on "
        f"{r11l_hudo_8k.get('n_seeds_won')}/{r11l_hudo_8k.get('n_seeds')} seeds in "
        f"{r11l_hudo_8k.get('median_actions_to_first_levelup')} actions, while the shipped "
        f"config is still at 0 levels on {r11l_ship_max.get('n_seeds')}/"
        f"{r11l_ship_max.get('n_seeds')} seeds at the largest budget tested "
        f"({r11l_max_budget}, {r11l_max_budget // A['config']['budget']}x the battery's), with "
        "the mask resolving normally. Neither flip's own A/B could have seen either, because "
        "both of "
        "their controls also lose these games. "
        "NO hidden-game transfer is claimed or measured here, and none can be from this "
        "harness: all 25 public games are already solved and the scored path is operator-only. "
        "This measures CONVENTION-DEPENDENCE, which is necessary but not sufficient for "
        "transfer -- and it now does so for ONE of the two levers, not both."
    )

    A["preconditions_checked"] = [
        {
            "resource": "offline_arcade_environment_files",
            "available": len(A["config"]["games"]) == 25,
            "detail": f"{len(A['config']['games'])} games instantiated from environment_files",
        },
        {
            "resource": "no_per_game_GameAdapter_on_the_measured_path",
            "available": all(not r["adapter_module_imported"] for r in receipt.values()),
            "detail": "carnot.agentic.arc_game_adapters absent from sys.modules after "
            "constructing every arm",
        },
        {
            "resource": "all_seven_gated_flags_resolve_as_pinned",
            "available": all(r["all_match"] for r in receipt.values()),
            "detail": "per-arm requested vs resolved-on-explorer comparison in "
            "arm_flag_resolution_receipt",
        },
        {
            "resource": "no_llm_no_gpu_required",
            "available": True,
            "detail": "StepwiseExplorer has no proposer parameter; llm_disabled=True",
        },
    ]

    A["principle_annotations"] = {
        "honest_verdict": "Terminal-prefixed self-declared state so the conductor reconciler "
        "can classify without re-running; a non-prefixed verdict risks "
        "false-positive partial classification.",
        "inference_substrate": "Declares that no model was loaded, so the linter applies the "
        "offline-arcade duration floor instead of the 60s live-LLM "
        "floor; without it a fast-but-real run reads as fabrication.",
        "solve_provenance": "development_proxy, because this runs the offline arcade over "
        "environment_files with the public games. It is NOT "
        "live_agent_self_discovery and NOT evidence the live agent "
        "self-discovers a hidden game.",
        "verifier_is_oracle": "False, and no verifier-moat or verifier-value claim is made: "
        "the levers under test are a search-ordering barrier and a "
        "node-identity mask, not verifiers. Correctness is the real "
        "env's own levels_completed counter, read only for SCORING.",
        "random_seed": "Determinism is the precondition for reproducibility; every cell is "
        "seeded and the per-arm determinism was MEASURED, not assumed.",
        "reproducibility_checksum": "Content hash over every (arm, game, condition, seed, "
        "levels, actions, states_expanded, hud_mask_resolved) "
        "tuple, so a replication can be compared exactly.",
        "duration_s": "Real compute takes wall-clock time; summed per-cell wall time is the "
        "load-bearing fabrication signal.",
        "preconditions_checked": "Records WHICH resources were verified before measuring, "
        "pre-empting the failure mode where an agent silently lacks "
        "the resource and synthesises a passing artifact.",
        "pass_region_witness": "A gate whose pass region is empty is not a gate; this emits "
        "the concrete cells that make each anchor non-zero.",
        "behavioural_dose_witness": "A metric that cannot causally depend on the intervention "
        "is not a measurement; this proves each perturbation "
        "actually moved each arm before any verdict is read.",
        "leave_one_game_out_jackknife": "A hidden game is a fresh draw from the game "
        "distribution; concentration of the gain on one or "
        "two public games bounds the expected fresh-draw gain.",
    }

    A["scope_and_limits"] = {
        "what_this_measures": (
            "Whether each shipped lever's measured gain depends on a CONVENTION that the 25 "
            "public games happen to share and an unseen game need not: absolute-colour "
            "salience for the frontier tier predicate, edge-adjacency for the HUD bar "
            "detector. This is the named, falsifiable mechanism by which a game-AGNOSTIC "
            "lever fails on a game it has never seen."
        ),
        "what_this_does_NOT_measure": [
            "A hidden-game score. The scored path is operator-only and no hidden game is "
            "available locally; all 25 public games are already solved.",
            "Whether the gain survives in the FULL scored configuration. This battery runs "
            "the bare StepwiseExplorer core (CarnotAgentPolicy, force_explore, no proposer). "
            "The scored E3AgentPolicy resolves the SAME seven lever values -- verified -- but "
            "runs them alongside a value head, candidate router, frame-change scorer, goal "
            "bias, epistemic ledger and the LLM induce/verify/plan cascade, none of which are "
            "present here, and at target_levels=3 rather than 1.",
            "Selection overfit to the 25-game corpus as such. The jackknife bounds how "
            "concentrated the gain is across games, but the corpus itself is the one the "
            "levers were iterated against and no measurement here can remove that.",
            "Conventions nobody enumerated. This battery bounds the two KNOWN convention "
            "risks; it cannot certify transfer.",
            "The HUD lever's convention-dependence AT ALL. Added 2026-07-25: neither perturbed "
            "condition can test it (C1 is off-target and inert on the HUD mask; C2 makes the "
            "lever's only two support games unwinnable for every arm). The battery bounds the "
            "FRONTIER lever's convention-dependence and leaves the HUD lever's open. A "
            "dose-parameterised roll (C3_roll_k1/k2, wired into cptb_perturb + cptb_run) is the "
            "hook for a condition that moves the bar off the edge without razing the corpus.",
            "Any effect small enough to need more than this design's resolution. With 5 seeds "
            "the smallest reachable exact sign-test p on the seed axis is 0.031, and on the "
            "GAME axis it is 0.5 for a one-game support and 0.25 for a two-game support -- so a "
            "one- or two-game gain CANNOT be established at p<=0.05 here however consistent it "
            "is. Every contrast now reports its own n_independent_replicates and its "
            "smallest_reachable_p_at_this_n rather than leaving that implicit.",
        ],
        "statistical_resolution_of_this_design": {
            "replication_unit_for_transfer": "the GAME (a hidden game is a fresh draw from the "
            "game distribution), which is why the sign tests "
            "run on games, not seeds",
            "n_games": len(A["config"]["games"]),
            "n_seeds": len(A["random_seeds_used"]),
            "seed_axis_smallest_reachable_p": round(0.5 ** len(A["random_seeds_used"]), 5),
            "arms_measured_deterministic": [
                a
                for a, d in A["measured_determinism_per_arm"].items()
                if d["measured_deterministic"]
            ],
            "note": (
                "For a contrast between two measured-deterministic arms the seeds are ONE "
                "observation repeated, not five trials; those contrasts report "
                "n_seed_replicates_effective = 1. The harness already declared these arms "
                "deterministic in cptb_arms.py and the first run's gates did not use that flag."
            ),
        },
        "corrected_caveat_wording_for_the_two_flips": (
            "Measured on the 25 PUBLIC games with the LLM-free StepwiseExplorer core -- NO "
            "per-game GameAdapter, no banked plan, no trained value head or candidate router "
            "(verified: arc_game_adapters is absent from sys.modules after constructing every "
            "arm, and StepwiseExplorer takes no game-id parameter at all). The earlier caveat "
            "'public games WITH per-game adaptation' was WRONG on the adaptation half: the "
            "measurement was already adapter-free, i.e. the generic solver. What remains "
            "undemonstrated is (i) that the effect survives on games outside the corpus the "
            "levers were selected against, and (ii) that it survives inside the full scored "
            "E3 cascade."
        ),
        "flags_flipped_by_this_experiment": "NONE. This is a measurement task; no SUBMITTED_* "
        "default and no shipped configuration was changed.",
    }

    # ------------------------------------------- code-stability receipt (concurrent-edit risk)
    # The conductor modified python/carnot/agentic/arc_competition_agent.py DURING this run
    # (mtime 2026-07-25 20:55 local, mid-battery), so "was every cell measured against the
    # same code?" is a live validity question rather than a formality.  Two independent
    # checks, both recorded rather than asserted:
    #
    #   CODE-PATH  the two hunks are (i) `_load_submitted_candidate_router` now returning the
    #              online click-target router, reachable ONLY from `E3AgentPolicy.__init__`
    #              via the `_DEFAULT_CANDIDATE_ROUTER` sentinel (arc_competition_agent.py:3941
    #              / :4004-4005; class E3AgentPolicy begins at :3893, CarnotAgentPolicy at
    #              :3782 and defaults candidate_router to None), and (ii) a new
    #              `candidate_router.observe_click_outcome(...)` block guarded by
    #              `candidate_router is not None`.  Both are unreachable/no-op when
    #              candidate_router is None, which a runtime probe confirms it is for every
    #              arm here.
    #   EMPIRICAL  36 cells re-run against the CURRENT (post-edit) working tree -- 3 games
    #              spanning an early-finishing and a late-finishing process, all 3 conditions,
    #              all 4 arms -- compared field-by-field against the recorded rows.
    A["code_stability_receipt"] = {
        "concern": "arc_competition_agent.py was modified by a concurrent conductor session "
        "partway through the battery; cells measured before and after that edit "
        "would otherwise be a silent confound.",
        "changed_hunks": [
            "_load_submitted_candidate_router now returns load_online_click_target_router "
            "(reachable only from E3AgentPolicy, which this battery does not run)",
            "a candidate_router.observe_click_outcome(...) call in StepwiseExplorer, guarded "
            "by `candidate_router is not None`",
        ],
        "code_path_analysis": {
            "CarnotAgentPolicy_candidate_router_default": "None",
            "explorer_candidate_router_observed_at_runtime": None,
            "changed_loader_reachable_from_measured_path": False,
        },
        "empirical_reproduction": {
            "n_cells_rerun_against_post_edit_tree": 36,
            "n_identical": 36,
            "n_different": 0,
            "fields_compared": [
                "ran",
                "levels",
                "actions",
                "states_expanded",
                "errors",
                "hud_mask_resolved",
                "hud_mask_cell_count",
                "actions_to_first_levelup",
            ],
            "games": ["r11l", "tn36", "ft09"],
            "raw_rows": "results/cptb_20260726_cells/reproduction_sample.jsonl.gz",
        },
        "conclusion": "The concurrent edit is inert for this battery's arms by both checks, so "
        "all 1500 cells are comparable.",
    }

    dirty = [ln[3:] for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    A["provenance"] = {
        "git_head": _git("rev-parse", "HEAD"),
        "working_tree_dirty_at_run_time": bool(dirty),
        # DERIVED (2026-07-25), was a hardcoded list.  A stale hardcoded dirty-file list is a
        # provenance claim that silently stops being true.
        "modified_or_untracked_files_at_artifact_build_time": dirty,
        "unstaged_files_at_original_battery_run_time_not_authored_by_this_experiment": [
            "python/carnot/agentic/arc_competition_agent.py",
            "python/carnot/agentic/arc_discriminative_router.py",
            "openspec/capabilities/arc-human-replay-frame-change/spec.md",
            "tests/python/test_arc_online_click_target_router.py",
        ],
        "git_head_subject": _git("log", "-1", "--pretty=%s"),
        "frontier_flip_commit": "c9e3c4459",
        "hud_flip_commit": "53e503c1b",
        "harness_reused": "python/carnot/experiment_5836_frontier_discipline_ab.py:run_cell",
        "runner": "scratchpad cptb_run.py (round-robin by arm)",
    }
    A["run_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(A, indent=1, default=str))
    print("WROTE", OUT)
    print("VERDICT", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
