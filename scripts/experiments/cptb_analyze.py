"""ANALYSIS + ARTIFACT for the convention-perturbation transfer battery.

Every statistic below is computed PER SEED and matched against the control's row for the SAME
(game, seed, condition).  Nothing is unioned across seeds.  The reason is specific and
historical: this project has twice mis-read an any-seed union -- once as "no arm regressed"
(exp5836's own seed_fairness_corrigendum: a union over 3 seeds can never register a loss so
long as ONE seed wins) and once, subtler, by comparing an arm's PER-SEED set against the
control's any-seed UNION, which makes the control appear to lose to itself.  Matched-seed
set difference is immune to both.

Definitions
-----------
    W(arm, cond, seed) = { game : levels > 0 }                       -- a win SET, never a total
    gain(t, c, cond, seed) = |W(t)\\W(c)| - |W(c)\\W(t)|             -- net games, per seed
    anchor A0 = median over seeds of gain at C0_real
    transfer T = median over seeds of gain at the perturbed condition
    retention R = T / A0, reported ONLY when A0 > 0

Gates
-----
Every gate emits a computed WITNESS before any verdict, because a gate whose pass region is
empty is not a gate (exp5835 was VOIDED for exactly this).  Two witnesses are required:

  PASS-REGION WITNESS  a concrete (game, seed) list where the treatment wins and the control
                       does not, at C0.  If that list is empty the anchor is zero and no
                       retention statement is computable -- the run is declared
                       uninterpretable rather than reported as "transfers".

  DOSE WITNESS         (a) the static convention dose measured on each game's reset frame
                       (cptb_dose.json), and (b) a BEHAVIOURAL dose: the count of cells whose
                       row actually changes between C0 and the perturbed condition.  A
                       perturbation with zero behavioural dose on an arm cannot have moved
                       that arm, so a flat result under it is uninterpretable, not reassuring.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Work directory holding the JSONL cell rows + intermediate JSON. Overridable so the
# battery can be run out of a scratch dir (as it was for the recorded run) or a
# repo-local dir, without editing the file.
SCRATCH = Path(os.environ.get("CPTB_WORKDIR") or Path(__file__).resolve().parent)
REPO = Path(__file__).resolve().parents[2]

CONDS = ["C0_real", "C1_salience_inversion", "C2_diag_roll"]
ARMS = ["CTRL", "FRONT", "HUDO", "SHIP"]

# (name, treatment, control, what it isolates)
CONTRASTS = [
    ("frontier_given_hud_off", "FRONT", "CTRL"),
    ("frontier_given_hud_on", "SHIP", "HUDO"),
    ("hud_given_frontier_on", "SHIP", "FRONT"),
    ("hud_given_frontier_off", "HUDO", "CTRL"),
    ("both_levers_shipped_vs_preflip", "SHIP", "CTRL"),
]


def load_rows():
    rows = []
    for p in sorted((SCRATCH / "battery").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(r):
    return (r["arm"], r["game"], r["condition"], int(r["seed"]))


def main() -> int:
    t0 = time.time()
    rows = load_rows()
    by = {key(r): r for r in rows}
    games = sorted({r["game"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})

    # ---------------------------------------------------------------- coverage + integrity
    expected = len(ARMS) * len(games) * len(CONDS) * len(seeds)
    errored = defaultdict(int)
    ran = defaultdict(int)
    uninstrumented = 0
    for r in rows:
        ran[(r["arm"], r["condition"])] += 1
        if not r.get("ran") or int(r.get("errors") or 0) > 0:
            errored[(r["arm"], r["condition"])] += 1
        if r.get("ran") and r.get("states_expanded") is None:
            uninstrumented += 1
    cell_integrity = {
        "n_cells_expected": expected,
        "n_cells_recorded": len(rows),
        "coverage_complete": len(rows) == expected,
        "n_rows_ran_false_or_errors": sum(errored.values()),
        "errored_cell_rate": round(sum(errored.values()) / max(1, len(rows)), 4),
        "n_ran_rows_missing_states_expanded": uninstrumented,
        "per_arm_condition_errored": {f"{a}|{c}": errored[(a, c)] for a in ARMS for c in CONDS},
        "interpretable": (
            len(rows) == expected
            and sum(errored.values()) / max(1, len(rows)) <= 0.05
            and uninstrumented == 0
        ),
        "principle": (
            "An arm with no expansion count and no error count is an UNINSTRUMENTED arm; this "
            "project once read a 72-97%-crashed control as a legitimate null across 975 cells "
            "for exactly that reason. Every row of every arm carries states_expanded and an "
            "error count, and a run above a 5% errored-cell rate is refused interpretation."
        ),
        "HONEST_SCOPE_OF_THE_ERROR_COUNT": (
            "Do NOT read errored_cell_rate == 0.0 as 'nothing was swallowed anywhere'. For an "
            "explorer arm, run_cell's `errors` counts only the click-pixel sampler's two "
            "internal exception paths, and click_pixel_sampling is OFF in all four arms here, "
            "so that counter is 0 BY CONSTRUCTION and carries almost no information. The "
            "load-bearing integrity signals in this run are the ones that are not vacuous: "
            "coverage_complete (1500 of 1500 expected cells present), zero rows with "
            "ran=False (a run_game crash is attributed to the raising arm's own row), and "
            "zero ran-rows missing states_expanded."
        ),
    }

    # ---------------------------------------------------------------- determinism (measured)
    determinism = {}
    for arm in ARMS:
        distinct = 0
        total = 0
        for g in games:
            for c in CONDS:
                sig = {
                    (by[(arm, g, c, s)].get("levels"), by[(arm, g, c, s)].get("actions"),
                     by[(arm, g, c, s)].get("states_expanded"))
                    for s in seeds if (arm, g, c, s) in by
                }
                total += 1
                if len(sig) > 1:
                    distinct += 1
        determinism[arm] = {
            "n_game_condition_cells": total,
            "n_cells_varying_across_seeds": distinct,
            "measured_deterministic": distinct == 0,
        }

    # ---------------------------------------------------------------- win sets, per seed
    def winset(arm, cond, seed):
        out = set()
        for g in games:
            r = by.get((arm, g, cond, seed))
            if r and r.get("ran") and int(r.get("levels") or 0) > 0:
                out.add(g)
        return out

    wins = {(a, c, s): winset(a, c, s) for a in ARMS for c in CONDS for s in seeds}

    per_arm_condition_wins = {
        f"{a}|{c}": {
            "per_seed_win_counts": [len(wins[(a, c, s)]) for s in seeds],
            "per_seed_win_sets": {str(s): sorted(wins[(a, c, s)]) for s in seeds},
            "median_wins": statistics.median([len(wins[(a, c, s)]) for s in seeds]),
            "n_games_won_on_every_seed": len(set.intersection(*[wins[(a, c, s)] for s in seeds])),
            "union_win_count_DO_NOT_USE_FOR_COMPARISON": len(
                set.union(*[wins[(a, c, s)] for s in seeds])
            ),
        }
        for a in ARMS
        for c in CONDS
    }

    # ---------------------------------------------------------------- behavioural dose
    behavioural_dose = {}
    for a in ARMS:
        for c in CONDS[1:]:
            moved = 0
            total = 0
            for g in games:
                for s in seeds:
                    r0, rx = by.get((a, g, "C0_real", s)), by.get((a, g, c, s))
                    if not r0 or not rx:
                        continue
                    total += 1
                    if (r0.get("levels"), r0.get("actions"), r0.get("states_expanded")) != (
                        rx.get("levels"), rx.get("actions"), rx.get("states_expanded")
                    ):
                        moved += 1
            behavioural_dose[f"{a}|{c}"] = {
                "n_cells": total,
                "n_cells_behaviourally_moved": moved,
                "fraction_moved": round(moved / max(1, total), 4),
                "inert_for_this_arm": moved == 0,
            }

    # ---------------------------------------------------------------- contrasts
    def gain(t, c, cond, seed):
        wt, wc = wins[(t, cond, seed)], wins[(c, cond, seed)]
        return len(wt - wc) - len(wc - wt)

    contrasts = {}
    for name, t, c in CONTRASTS:
        entry = {"treatment_arm": t, "control_arm": c, "per_condition": {}}
        for cond in CONDS:
            g_by_seed = {str(s): gain(t, c, cond, s) for s in seeds}
            gained = {str(s): sorted(wins[(t, cond, s)] - wins[(c, cond, s)]) for s in seeds}
            lost = {str(s): sorted(wins[(c, cond, s)] - wins[(t, cond, s)]) for s in seeds}
            vals = list(g_by_seed.values())
            entry["per_condition"][cond] = {
                "per_seed_gain": g_by_seed,
                "median_gain": statistics.median(vals),
                "min_gain": min(vals),
                "max_gain": max(vals),
                "strict_per_seed_dominance": all(v > 0 for v in vals),
                "no_seed_regresses": all(v >= 0 for v in vals),
                "games_gained_per_seed": gained,
                "games_lost_per_seed": lost,
                "games_gained_on_every_seed": sorted(
                    set.intersection(*[wins[(t, cond, s)] - wins[(c, cond, s)] for s in seeds])
                ),
                "games_lost_on_every_seed": sorted(
                    set.intersection(*[wins[(c, cond, s)] - wins[(t, cond, s)] for s in seeds])
                ),
            }
        a0 = entry["per_condition"]["C0_real"]["median_gain"]
        entry["anchor_median_gain_C0"] = a0
        entry["retention"] = {}
        for cond in CONDS[1:]:
            tg = entry["per_condition"][cond]["median_gain"]
            entry["retention"][cond] = {
                "transfer_median_gain": tg,
                "retention_ratio": (round(tg / a0, 4) if a0 > 0 else None),
                "computable": a0 > 0,
                "reason_if_not_computable": (
                    None if a0 > 0 else
                    "anchor median gain at C0_real is <= 0, so a retention ratio is undefined; "
                    "there is no measured effect for the perturbation to retain"
                ),
            }
        contrasts[name] = entry

    # ---------------------------------------------------------------- pass-region witness
    witness = {}
    for name, t, c in CONTRASTS:
        cells = []
        for s in seeds:
            for g in sorted(wins[(t, "C0_real", s)] - wins[(c, "C0_real", s)]):
                cells.append({"game": g, "seed": s})
        witness[name] = {
            "pass_region_nonempty": bool(cells),
            "witness_cells_treatment_wins_control_does_not_at_C0": cells,
            "n_witness_cells": len(cells),
            "principle": (
                "A gate whose pass region is empty is not a gate. This emits the concrete "
                "(game, seed) cells that make the anchor non-zero, so a reader can verify the "
                "comparison could have registered an effect before reading any verdict."
            ),
        }

    # ------------------------------------------------- leave-one-game-out jackknife of the anchor
    loo = {}
    for name, t, c in CONTRASTS:
        per_game = {}
        for held in games:
            vals = []
            for s in seeds:
                wt = wins[(t, "C0_real", s)] - {held}
                wc = wins[(c, "C0_real", s)] - {held}
                vals.append(len(wt - wc) - len(wc - wt))
            per_game[held] = statistics.median(vals)
        full = contrasts[name]["anchor_median_gain_C0"]
        worst = min(per_game.items(), key=lambda kv: kv[1]) if per_game else (None, None)
        loo[name] = {
            "full_corpus_median_gain": full,
            "median_gain_with_each_game_held_out": per_game,
            "single_game_whose_removal_costs_the_most": worst[0],
            "median_gain_without_that_game": worst[1],
            "n_games_whose_removal_drops_the_gain_to_zero_or_below": sum(
                1 for v in per_game.values() if v <= 0
            ),
            "principle": (
                "A hidden game is a fresh draw from the game distribution. If the whole "
                "measured gain rests on one or two public games, the expected gain on a fresh "
                "draw is close to zero even though the corpus-level number looks large. This "
                "jackknife measures that concentration directly; it is NOT a threshold refit "
                "(the levers have no corpus-fitted threshold to refit -- the frontier tier "
                "constants are ported verbatim from the external reference, and the HUD "
                "thresholds sit in saturated margins where every fold refits identically)."
            ),
        }

    # ---------------------------------------------------------------- efficiency axis
    efficiency = {}
    for name, t, c in CONTRASTS:
        per_cond = {}
        for cond in CONDS:
            deltas_actions, deltas_states = [], []
            for g in games:
                for s in seeds:
                    rt, rc = by.get((t, g, cond, s)), by.get((c, g, cond, s))
                    if not (rt and rc and rt.get("ran") and rc.get("ran")):
                        continue
                    # only where BOTH won the same game -- otherwise "actions" is a
                    # budget-bound non-comparable quantity
                    if int(rt.get("levels") or 0) > 0 and int(rc.get("levels") or 0) > 0:
                        at, ac = rt.get("actions_to_first_levelup"), rc.get(
                            "actions_to_first_levelup")
                        if at is not None and ac is not None:
                            deltas_actions.append(int(at) - int(ac))
                        st, sc_ = rt.get("states_expanded"), rc.get("states_expanded")
                        if st is not None and sc_ is not None:
                            deltas_states.append(int(st) - int(sc_))
            per_cond[cond] = {
                "n_commonly_won_cells": len(deltas_actions),
                "median_delta_actions_to_first_levelup": (
                    statistics.median(deltas_actions) if deltas_actions else None),
                "median_delta_states_expanded": (
                    statistics.median(deltas_states) if deltas_states else None),
                "note": "negative = treatment cheaper. Only cells BOTH arms won are compared.",
            }
        efficiency[name] = per_cond

    # ------------------------------------------- MECHANISM attribution: did the mask resolve?
    # Failure mode #7 in this project's list is crediting a result to the wrong mechanism.
    # "The HUD gain vanished under C2" has two candidate explanations that must be separated:
    #   (a) the detector's edge-adjacency assumption broke, so the mask never resolved; or
    #   (b) the game became unwinnable for every arm, so there was no gain left to have.
    # These are distinguishable because `hud_mask_resolved` is recorded on every row: (a)
    # predicts the resolution rate collapses, (b) predicts it does not.  Both can be true at
    # once, and where they are, the artifact says so rather than picking the flattering one.
    mask_resolution = {}
    for a in ("HUDO", "SHIP"):
        for c in CONDS:
            res, tot, per_game = 0, 0, {}
            for g in games:
                hits = 0
                for s in seeds:
                    r = by.get((a, g, c, s))
                    if r is None:
                        continue
                    tot += 1
                    if r.get("hud_mask_resolved"):
                        res += 1
                        hits += 1
                per_game[g] = hits
            mask_resolution[f"{a}|{c}"] = {
                "n_cells": tot,
                "n_cells_mask_resolved": res,
                "fraction_resolved": round(res / max(1, tot), 4),
                "n_games_resolved_on_every_seed": sum(1 for v in per_game.values() if v == len(seeds)),
                "per_game_seeds_resolved": per_game,
            }

    # Per-game per-condition win matrix -- the failure SET view (#6), so a reader sees WHICH
    # game moved rather than a total.  Value = number of seeds (of 5) the arm won that game.
    win_matrix = {
        c: {
            g: {a: sum(1 for s in seeds if g in wins[(a, c, s)]) for a in ARMS}
            for g in games
        }
        for c in CONDS
    }
    # Games where the two levers INTERACT destructively: HUDO wins and SHIP does not.
    destructive_interaction = {
        c: sorted(
            g for g in games
            if all(g in wins[("HUDO", c, s)] for s in seeds)
            and all(g not in wins[("SHIP", c, s)] for s in seeds)
        )
        for c in CONDS
    }

    # ---------------------------------------------------------------- assemble
    dose_static = json.loads((SCRATCH / "cptb_dose.json").read_text())
    wall = round(sum(float(r.get("cell_wall_s") or 0) for r in rows), 2)

    payload_for_hash = json.dumps(
        sorted(
            [
                [r["arm"], r["game"], r["condition"], int(r["seed"]), r.get("levels"),
                 r.get("actions"), r.get("states_expanded"), r.get("hud_mask_resolved")]
                for r in rows
            ]
        ),
        sort_keys=True,
    )
    checksum = hashlib.sha256(payload_for_hash.encode()).hexdigest()

    out = {
        "experiment": "outer_loop_cptb_shipped_lever_convention_transfer",
        "title": (
            "Convention-perturbation transfer battery for the two levers flipped ON 2026-07-25 "
            "(frontier tier discipline; HUD edge-bar mask)"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": wall,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": 20260726,
        "random_seeds_used": seeds,
        "reproducibility_checksum": checksum,
        "config": {
            "games": games,
            "arms": {a: {"label": None, "kwargs": None} for a in ARMS},
            "conditions": CONDS,
            "budget": 2000,
            "n_seeds": len(seeds),
            "policy_kind": "explorer_force_explore_no_proposer",
            "llm_disabled": True,
            "adapters_loaded": False,
        },
        "cell_integrity": cell_integrity,
        "measured_determinism_per_arm": determinism,
        "static_convention_dose_witness": dose_static,
        "behavioural_dose_witness": behavioural_dose,
        "pass_region_witness": witness,
        "per_arm_condition_wins": per_arm_condition_wins,
        "contrasts": contrasts,
        "leave_one_game_out_jackknife": loo,
        "efficiency_axis": efficiency,
        "hud_mask_resolution_mechanism_evidence": mask_resolution,
        "per_game_win_matrix_seeds_won_of_5": win_matrix,
        "games_where_adding_frontier_destroys_a_hud_win": destructive_interaction,
    }
    p = SCRATCH / "cptb_analysis.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print("WROTE", p, "cells", len(rows), "elapsed", round(time.time() - t0, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
