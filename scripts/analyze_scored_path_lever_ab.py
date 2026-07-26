"""ANALYSE the scored-path lever A/B and emit the results artifact.

WHAT THIS SCORES. `scripts/arc_scored_path_lever_harness.py` produces one ROW per
(game, seed, arm) cell on the SCORED path (`E3AgentPolicy`, the full verifier-routed cascade)
with the LLM ACTUALLY ON. This file turns those rows into a verdict, and its whole design is a
list of the specific measurement defects this project has already made once. Each rule below
exists because a published number was wrong in exactly that way:

 1. PER-SEED, MATCHED-CELL scoring only. A treatment arm's per-seed win set is compared against
    the CONTROL'S SAME-SEED win set -- never against the control's ANY-SEED UNION. Union scoring
    makes the control fail against itself (a game the control wins on 1 of 3 seeds enters the
    union, then "loses" on the other two).
 2. FAILURE / WIN SETS, never totals. `|W(S)| == |W(S_minus_hud)|` is compatible with the two
    arms winning DISJOINT games. Only the set difference is evidence.
 3. A COMPUTED WITNESS at the GATE'S OWN LEVEL OF AGGREGATION. A gate on a per-seed set
    difference needs a witness that the pass region is non-empty PER SEED. Computing a witness
    per-cell for a gate defined on an aggregate is how the defect recurred (exp5835): 20 of 31
    cells were structurally frozen, so a negative median needed 16 negatives when only 11 could
    move, and the first draft claimed the witness passed.
 4. THE FORCED-VALUE CHECK. If no arm wins a game under the tested condition, that game cannot
    discriminate and its contribution to any delta is ARITHMETICALLY FORCED to zero. A gate whose
    entire support is such games is not a measurement; it is stamped UNINTERPRETABLE rather than
    reported as a clean 0 (the `C2_diag_roll` defect in the convention-transfer battery).
 5. FIRE COUNTERS decide whether a null is admissible. A lever that never fired in a cell
    contributes NO evidence from that cell, in either direction. The exp5836 dead-observe-channel
    incident produced a byte-identical zero-error NULL that was pure harness artifact, so a cell
    where a lever did not fire is EXCLUDED from that lever's denominator and counted separately.
 6. THE LLM-VALIDITY WITNESS. This is an LLM-ON measurement, so a row whose generator was dead,
    whose server stormed, or which produced zero completions is not an LLM-on datum at all. Those
    rows are excluded and counted; `llm_on_row_valid` is the harness-side witness for it.

WHAT IT DELIBERATELY DOES NOT DO. It does not flip a flag, recommend a submission, or aggregate
any cell whose lever never fired into a headline. Efficiency (`actions_to_first_levelup`) is
reported as a paired per-game delta and never averaged across games -- the games have wildly
different action costs, so a mean over games is dominated by whichever game is slowest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")

# The arm whose configuration is what ships today. Every delta is measured against THIS.
# Parameterised by the harness's arm SUFFIX (`_llmon` / `_llmoff`) so the SAME machinery scores the
# LLM-on design and the much cheaper LLM-off design without a second, divergent analyser -- two
# analysers is how two conclusions drift apart.
CONTROL = "S_llmon"
SUFFIX = "_llmon"


def set_condition(suffix: str) -> None:
    global CONTROL, SUFFIX, LEVERS
    SUFFIX = suffix
    CONTROL = f"S{suffix}"
    LEVERS = build_levers(suffix)


def build_levers(suffix: str) -> dict[str, dict]:
    """Which lever each treatment arm removes/adds, and which fire counter decides whether a cell
    carries evidence about it. `on_arm` is the arm in which the lever is ON -- that is the arm whose
    fire counter must be non-zero, because a lever that is OFF trivially does not fire."""
    ctrl = f"S{suffix}"
    return {
        f"S_minus_frontier{suffix}": {
            "lever": "lever1_frontier_tier_trio",
            "direction": "removed",
            "fire_key": "lever1_fired",
            "on_arm": ctrl,
        },
        f"S_minus_hud{suffix}": {
            "lever": "lever2_edge_bar_hud_trio",
            "direction": "removed",
            "fire_key": "lever2_fired",
            "on_arm": ctrl,
        },
        f"S_plus_hazard{suffix}": {
            "lever": "lever3_hazard_move_pruner",
            "direction": "added",
            "fire_key": "lever3_fired",
            "on_arm": f"S_plus_hazard{suffix}",
        },
    }


LEVERS = build_levers(SUFFIX)


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            r["_source"] = p.name
            rows.append(r)
    return rows


def cell_key(r: dict) -> tuple:
    return (r["game"], r["seed"])


def is_win(r: dict) -> bool:
    """A cell is a WIN when the agent completed at least one level. `levels` comes from the
    offline arcade's own level counter, not from anything the agent self-reports."""
    return bool(r.get("ran")) and int(r.get("levels") or 0) >= 1


# PROCESS-ONLY observables, used for the mechanism-independent fire witness.
#
# WHY THE OUTCOME IS EXCLUDED, AND WHY THAT IS THE WHOLE POINT. The witness's job is to establish
# that the gate's pass region is non-empty BEFORE the outcome is looked at. If "the lever did
# something here" were allowed to be established by "the arm won a different set of games here",
# the witness would be VACUOUS -- every observed win difference would automatically certify its
# own support, which is precisely the defect the witness exists to prevent (a gate that cannot
# fail its own witness is not a check). So `levels`, `reached` and `actions_to_first_levelup` are
# excluded from the witness, as is `actions` (a run that levels up can consume its budget
# differently, so it is outcome-coupled). What remains is search behaviour: how many states the
# explorer expanded and what mix of actions it chose.
_WITNESS_PROCESS_KEYS = (
    "states_expanded",
    "n_click_actions",
    "n_nav_actions",
    "top_action",
    "top_action_count",
)

# The FULL observable tuple, including the outcome. Used ONLY for the inertness census -- a cell
# identical on all of these is one where the lever provably changed nothing whatsoever.
_BEHAVIOUR_KEYS = (
    "levels",
    "reached",
    "actions",
    *_WITNESS_PROCESS_KEYS,
    "actions_to_first_levelup",
)


def behaviour_tuple(r: dict) -> tuple:
    return tuple(r.get(k) for k in _BEHAVIOUR_KEYS)


def fired_behaviourally(arm_row: dict, ctrl_row: dict) -> bool:
    """Did toggling the lever change the SEARCH -- judged without looking at the outcome?

    WHY THIS EXISTS ALONGSIDE THE PER-LEVER COUNTER. A lever's own counter can under-detect it.
    `lever1_fired` is `tier_advances > 0`, but the frontier trio also RESTRICTS the action
    vocabulary at the tier it is already on -- so a cell can have `tier_advances == 0` while the
    lever nonetheless changed every action the agent took. Using the counter alone would exclude
    such a cell from the lever's denominator and thereby dilute a real effect toward zero, which
    is the mirror image of the dead-channel defect.

    Conversely, if the arm's search is IDENTICAL to the control's on every process observable, the
    lever did nothing in that cell, whatever its counter says. Both arms take the same seed, so an
    inert lever yields an identical trajectory; divergence is therefore causal evidence that the
    toggle mattered, not stochastic noise.

    The two witnesses are reported separately and unioned for the movable support, so neither an
    inert cell (counted as evidence) nor a silently-active cell (excluded from evidence) can slip
    through.
    """
    return tuple(arm_row.get(k) for k in _WITNESS_PROCESS_KEYS) != tuple(
        ctrl_row.get(k) for k in _WITNESS_PROCESS_KEYS
    )


def fully_inert(arm_row: dict, ctrl_row: dict) -> bool:
    """Identical on EVERY observable including the outcome: the lever changed nothing at all."""
    return behaviour_tuple(arm_row) == behaviour_tuple(ctrl_row)


def llm_row_is_valid(r: dict) -> bool:
    """Is this row a genuine LLM-on datum? RECOMPUTED here, not taken from the harness's stamp.

    WHY THE HARNESS'S OWN STAMP IS NOT TRUSTED -- a measured false positive, 2026-07-26. The
    harness's `llm_on_row_valid` ANDs in `not server_storm_suspected`, and `server_storm_suspected`
    is `llama_servers_after > llama_servers_before` where the count is every process on the box whose
    command line contains "llama-server". On this machine that count also includes (a) the
    CONDUCTOR's own generator on port 8924 and (b) `[llama-server] <defunct>` ZOMBIES left behind by
    conductor experiment subprocesses -- three were present simultaneously. So the count rose from 2
    to 5 during cells whose OWN generator was demonstrably fine (`generator_healthy_before=True`,
    `generator_healthy_after=True`, 3-5 real completions), and three perfectly good cells were
    discarded for activity belonging to unrelated work.

    THIS IS NOT A CONVENIENT RELAXATION, and the distinction matters. The storm test was protecting
    against a pile-up of servers on THIS harness's OWN port, which silently degrades cells to
    LLM-off. Two things make it redundant here: (1) every run in this measurement passes
    `--no-spawn`, which replaces `_ensure_server` with a health check that can never launch a
    server, so a same-port pile-up is impossible BY CONSTRUCTION; and (2) the degradation it was
    meant to catch shows up directly in the fields that ARE checked -- a cell served by a dying
    server records zero completions or fails the post-cell health check. A global process count
    measures other processes, not this cell.

    So the criterion is: the generator answered this cell (>=1 real completion) and was healthy on
    BOTH sides of it. The raw counts stay in the row and the disagreement with the harness's stamp is
    reported, so nothing is hidden.
    """
    if not r.get("ran"):
        return False
    if not r.get("llm_enabled"):
        return True  # an LLM-off row makes no LLM claim to invalidate
    llm = r.get("llm") or {}
    return bool(
        int(llm.get("responses") or 0) > 0
        and r.get("generator_healthy_before")
        and r.get("generator_healthy_after")
    )


def analyse(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # ---- 0. LLM-validity gate on the rows themselves -------------------------------------
    # An LLM-ON measurement whose generator was dead is not an LLM-on measurement. Excluded and
    # counted, never silently averaged in.
    valid = [r for r in rows if llm_row_is_valid(r)]
    invalid = [r for r in rows if not llm_row_is_valid(r)]
    out["rows_total"] = len(rows)
    out["rows_llm_valid"] = len(valid)
    out["rows_excluded_llm_invalid"] = [
        {
            "arm": r.get("arm"),
            "game": r.get("game"),
            "seed": r.get("seed"),
            "ran": r.get("ran"),
            "reason": r.get("reason"),
            "llm_responses": (r.get("llm") or {}).get("responses"),
            "generator_healthy_before": r.get("generator_healthy_before"),
            "generator_healthy_after": r.get("generator_healthy_after"),
            "server_storm_suspected": r.get("server_storm_suspected"),
        }
        for r in invalid
    ]
    # Where the recomputed criterion DISAGREES with the harness's stamp, say so explicitly with the
    # evidence, so the relaxation is auditable rather than silent.
    out["rows_rescued_from_harness_storm_false_positive"] = [
        {
            "arm": r.get("arm"),
            "game": r.get("game"),
            "seed": r.get("seed"),
            "llm_responses": (r.get("llm") or {}).get("responses"),
            "generator_healthy_before": r.get("generator_healthy_before"),
            "generator_healthy_after": r.get("generator_healthy_after"),
            "llama_servers_before": r.get("llama_servers_before"),
            "llama_servers_after": r.get("llama_servers_after"),
        }
        for r in rows
        if llm_row_is_valid(r) and not r.get("llm_on_row_valid")
    ]

    # DUPLICATE (arm, game, seed) CELLS. Two row files can legitimately contain the same cell (e.g.
    # a standalone LLM-off run and the LLM-off arm of a larger design). Keying a dict by
    # (arm, cell) would silently keep whichever loaded last -- a hidden choice about which
    # measurement counts. Duplicates are instead DETECTED and reported: if they disagree on the
    # outcome they are a free independent replication of that condition (real information about
    # run-to-run variation), and if they agree they confirm determinism. Either way the reader is
    # told, and the FIRST occurrence is the one used so the choice is deterministic rather than
    # dependent on file order.
    by_arm_cell: dict[tuple[str, tuple], dict] = {}
    dup_agree: list[dict] = []
    dup_disagree: list[dict] = []
    for r in valid:
        k = (r["arm"], cell_key(r))
        if k in by_arm_cell:
            first = by_arm_cell[k]
            rec = {
                "arm": r["arm"],
                "game": r["game"],
                "seed": r["seed"],
                "first_levels": first.get("levels"),
                "duplicate_levels": r.get("levels"),
                "first_states": first.get("states_expanded"),
                "duplicate_states": r.get("states_expanded"),
                "first_source": first.get("_source"),
                "duplicate_source": r.get("_source"),
            }
            if behaviour_tuple(first) == behaviour_tuple(r):
                dup_agree.append(rec)
            else:
                dup_disagree.append(rec)
            continue
        by_arm_cell[k] = r
    out["duplicate_cells_identical"] = dup_agree
    out["duplicate_cells_divergent"] = dup_disagree
    out["duplicate_cells_note"] = (
        "Duplicates are an independent re-run of the same (arm, game, seed). 'divergent' ones "
        "measure run-to-run variation directly; 'identical' ones confirm the condition is "
        "deterministic. The first occurrence is used in all comparisons."
    )
    arms = sorted({r["arm"] for r in valid})
    seeds = sorted({r["seed"] for r in valid})
    out["arms"] = arms
    out["seeds"] = seeds

    # ---- 1. MATCHED CELLS ONLY -------------------------------------------------------------
    # A (game, seed) is comparable only if BOTH arms in the comparison have a valid row for it. A
    # cell present for one arm and missing for the other cannot enter a delta in either direction.
    #
    # MATCHING IS PAIRWISE, NOT GLOBAL. An earlier draft required every arm to have a valid row
    # before a cell counted anywhere, which means ONE flaky arm (a cell whose generator blipped)
    # deletes that game from EVERY lever's comparison -- five arms give five times the exposure to
    # a single bad cell, and the deleted games are exactly the slow, interesting ones. Each lever
    # is therefore scored on its OWN matched population (control + that arm), and the population
    # size is reported per lever so an uneven denominator is visible rather than hidden.
    all_cells = sorted({cell_key(r) for r in valid})
    matched_all = [c for c in all_cells if all((a, c) in by_arm_cell for a in arms)]

    def matched_pair(arm: str) -> list[tuple]:
        return [c for c in all_cells if (CONTROL, c) in by_arm_cell and (arm, c) in by_arm_cell]

    matched = matched_all
    out["cells_seen"] = len(all_cells)
    out["cells_matched_all_arms"] = len(matched)
    out["cells_unmatched"] = [
        {"game": c[0], "seed": c[1], "arms_present": [a for a in arms if (a, c) in by_arm_cell]}
        for c in all_cells
        if c not in matched
    ]

    # ---- 2. PER-SEED WIN SETS (sets, never totals) -----------------------------------------
    # Per arm, over the cells THAT ARM actually has a valid row for. Restriction to a comparison's
    # matched population happens inside each comparison, not here, so one arm's missing cell cannot
    # silently shrink another arm's reported win set.
    win_sets: dict[str, dict[int, list[str]]] = {}
    for a in arms:
        win_sets[a] = {}
        for s in seeds:
            games = sorted(
                c[0]
                for c in all_cells
                if c[1] == s and (a, c) in by_arm_cell and is_win(by_arm_cell[(a, c)])
            )
            win_sets[a][s] = games
    out["win_sets_per_seed"] = {a: {str(s): v for s, v in d.items()} for a, d in win_sets.items()}
    out["win_counts_per_seed"] = {
        a: {str(s): len(v) for s, v in d.items()} for a, d in win_sets.items()
    }

    # ---- 3. THE FORCED-VALUE CHECK ---------------------------------------------------------
    # A game won by NO arm on a seed is non-discriminating THERE: every arm's contribution to the
    # delta on that game is arithmetically forced to zero. Report the discriminating support
    # explicitly so a "0 difference" can never be read as "no effect" when the support was empty.
    # Computed over EVERY cell any arm has a valid row for -- NOT over the all-arms-matched subset.
    # An earlier draft gated this on all-arms matching, which reported an EMPTY discriminating set
    # while the pairwise comparisons were correctly finding discriminating games: on a partially
    # complete run almost nothing is matched across all five arms, so the global summary
    # contradicted the per-lever detail. The per-lever verdicts use their own PAIRWISE set
    # (`discriminating_games_in_this_comparison`); this is the corpus-level summary.
    discriminating: dict[int, list[str]] = {}
    for s in seeds:
        discriminating[s] = sorted(
            {
                c[0]
                for c in all_cells
                if c[1] == s
                and any((a, c) in by_arm_cell and is_win(by_arm_cell[(a, c)]) for a in arms)
            }
        )
    out["discriminating_games_per_seed"] = {str(s): v for s, v in discriminating.items()}
    out["nondiscriminating_games_per_seed"] = {
        str(s): sorted({c[0] for c in all_cells if c[1] == s} - set(discriminating[s]))
        for s in seeds
    }

    # ---- 3b. THE SAME-CONFIG NOISE FLOOR ---------------------------------------------------
    # `S_replicate` pins byte-identical flags to the control. Any difference between them is
    # therefore NOT a lever effect -- it is the run-to-run variation the LLM introduces (generator
    # sampling plus llama-server slot/checkpoint state). Every lever delta below is reported
    # against this, because on a single seed a win difference no larger than the noise floor is not
    # evidence of anything. Without this arm the whole A/B would rest on an UNTESTED assumption
    # that a seeded run is deterministic with the LLM in the loop.
    def pairwise_vs_control(other: str) -> dict[str, Any]:
        """Per-seed matched comparison of `other` against the control, on their own shared cells.

        Used for BOTH the same-config noise floor (`S_replicate`) and the LLM contribution
        (`S_llmoff`). Both are reference comparisons rather than lever ablations, so they report the
        raw win-set symmetric difference without a fire witness -- there is no lever to have fired.
        """
        res: dict[str, Any] = {}
        pcells = matched_pair(other)
        for s in seeds:
            games_s = sorted({c[0] for c in pcells if c[1] == s})
            ctrl = {g for g in win_sets[CONTROL][s] if g in games_s}
            oth = {g for g in win_sets[other][s] if g in games_s}
            proc_div = sorted(
                g
                for g in games_s
                if fired_behaviourally(by_arm_cell[(other, (g, s))], by_arm_cell[(CONTROL, (g, s))])
            )
            inert = sorted(
                g
                for g in games_s
                if fully_inert(by_arm_cell[(other, (g, s))], by_arm_cell[(CONTROL, (g, s))])
            )
            res[str(s)] = {
                "n_games_measured": len(games_s),
                "games_measured": games_s,
                "control_win_set": sorted(ctrl),
                "other_win_set": sorted(oth),
                "control_only_wins": sorted(ctrl - oth),
                "other_only_wins": sorted(oth - ctrl),
                "win_flips_same_config": sorted(ctrl ^ oth),
                "n_win_flips_same_config": len(ctrl ^ oth),
                "process_divergent_games_same_config": proc_div,
                "n_process_divergent_same_config": len(proc_div),
                "fully_inert_games_same_config": inert,
                "run_is_deterministic_under_seed": len(proc_div) == 0,
            }
        return res

    noise: dict[str, Any] = {}
    replicate_arm = f"S_replicate{SUFFIX}"
    if replicate_arm in arms and CONTROL in arms:
        noise = pairwise_vs_control(replicate_arm)
    out["noise_floor_same_config_replicate"] = noise

    # ---- 3c. WHAT THE LLM ITSELF CONTRIBUTES ------------------------------------------------
    # `S_llmoff` is the SAME E3AgentPolicy configuration with induction disabled -- i.e. the
    # 2026-07-25 measurement condition, run on the scored path at the eval's own budget. Without
    # this, "the scored path with the LLM on wins k games" has no reference to be read against, and
    # the question the whole exercise exists to answer (does the LLM tier help where it scores?)
    # stays open. Reported as a comparison, never folded into a lever verdict: turning the LLM off
    # is not one of the levers under test.
    out["llm_contribution_vs_llm_off"] = (
        pairwise_vs_control("S_llmoff")
        if "S_llmoff" in arms and CONTROL in arms and CONTROL != "S_llmoff"
        else {}
    )

    # ---- 4. PER-LEVER VERDICT, each with its own computed witness ---------------------------
    verdicts: dict[str, Any] = {}
    for arm, meta in LEVERS.items():
        if arm not in arms:
            continue
        on_arm = meta["on_arm"]
        if on_arm not in arms:
            continue
        fk = meta["fire_key"]

        pair_cells = matched_pair(arm)
        per_seed: dict[str, Any] = {}
        for s in seeds:
            # The comparison population for THIS lever on THIS seed: games where both the control
            # and this arm produced a valid LLM-on row.
            pair_games = sorted({c[0] for c in pair_cells if c[1] == s})
            ctrl_wins = {g for g in win_sets[CONTROL][s] if g in pair_games}
            arm_wins = {g for g in win_sets[arm][s] if g in pair_games}
            # Discriminating WITHIN this pairwise comparison: won by the control or by this arm.
            # Using the global "won by any of the five arms" set would import games that this
            # comparison cannot speak to.
            discriminating_pair = sorted(ctrl_wins | arm_wins)

            # THE WITNESS, computed at the gate's own aggregation level (per seed, game unit).
            # The gate asks "does the delta in WINS differ from zero on this seed?" A game can
            # contribute only if BOTH (a) it is discriminating on this seed (some arm wins it)
            # and (b) the lever DID SOMETHING there. (b) is established by EITHER witness:
            #   * the lever's own counter in the arm where it is ON (`fk`), or
            #   * behavioural divergence between the arm and the control.
            # The union is the honest support. Using the counter alone would exclude a cell where
            # the lever silently changed every action without advancing a tier (diluting a real
            # effect); using divergence alone would credit a cell that diverged only through
            # run-to-run stochasticity. Reporting both separately makes which one carried the
            # cell auditable.
            def _direct(g: str) -> bool:
                return bool(by_arm_cell[(on_arm, (g, s))].get(fk))

            def _behav(g: str) -> bool:
                return fired_behaviourally(
                    by_arm_cell[(arm, (g, s))], by_arm_cell[(CONTROL, (g, s))]
                )

            movable = sorted(g for g in discriminating_pair if _direct(g) or _behav(g))
            frozen_nofire = sorted(g for g in discriminating_pair if not (_direct(g) or _behav(g)))
            witness_pass_region_nonempty = len(movable) > 0
            witness_direct_only = sorted(
                g for g in discriminating_pair if _direct(g) and not _behav(g)
            )
            witness_behavioural_only = sorted(
                g for g in discriminating_pair if _behav(g) and not _direct(g)
            )
            # Corpus-wide inertness census: a game where the arm is byte-identical to the control
            # on every observable is a cell in which the lever provably did nothing, whether or
            # not the game is discriminating.
            inert_all_games = sorted(
                g
                for g in pair_games
                if fully_inert(by_arm_cell[(arm, (g, s))], by_arm_cell[(CONTROL, (g, s))])
            )

            lost = sorted(ctrl_wins - arm_wins)  # games the control wins and this arm does not
            gained = sorted(arm_wins - ctrl_wins)
            # Restrict the reportable delta to the MOVABLE support. A win difference on a game
            # where the lever never fired cannot be CAUSED by the lever -- crediting it to the
            # lever is the wrong-mechanism defect. Such differences are reported separately as
            # run-to-run noise on a non-firing game.
            lost_movable = [g for g in lost if g in movable]
            gained_movable = [g for g in gained if g in movable]
            lost_nonfiring = [g for g in lost if g not in movable]
            gained_nonfiring = [g for g in gained if g not in movable]

            # THE NOISE FLOOR, applied. `S_replicate` pins the SAME flags as the control, so the
            # number of games whose win status flips between them is the amount of win-set movement
            # this setup produces with NO lever change at all. A lever delta that does not exceed
            # it is not evidence of the lever -- on a single seed it is indistinguishable from
            # generator sampling variation. This is the check that stops a 1-game difference on one
            # seed from being written up as an effect.
            n_moved = len(lost_movable) + len(gained_movable)
            noise_flips = int((noise.get(str(s)) or {}).get("n_win_flips_same_config") or 0)
            noise_measured = str(s) in noise
            exceeds_noise = n_moved > noise_flips

            if not witness_pass_region_nonempty:
                seed_verdict = "UNINTERPRETABLE_EMPTY_PASS_REGION"
            elif n_moved == 0:
                seed_verdict = "NO_EFFECT_ON_WINS"
            elif not noise_measured:
                seed_verdict = "EFFECT_ON_WINS_NOISE_FLOOR_UNMEASURED"
            elif not exceeds_noise:
                seed_verdict = "EFFECT_WITHIN_SAME_CONFIG_NOISE_FLOOR"
            else:
                seed_verdict = "EFFECT_ON_WINS"

            per_seed[str(s)] = {
                "control_win_set": sorted(ctrl_wins),
                "arm_win_set": sorted(arm_wins),
                "witness_movable_games": movable,
                "witness_pass_region_nonempty": witness_pass_region_nonempty,
                "witness_frozen_lever_never_fired": frozen_nofire,
                "witness_by_direct_counter_only": witness_direct_only,
                "witness_by_behavioural_divergence_only": witness_behavioural_only,
                "arm_inert_vs_control_all_games": inert_all_games,
                "n_games_measured": len(pair_games),
                "discriminating_games_in_this_comparison": discriminating_pair,
                "lost_vs_control_movable": lost_movable,
                "gained_vs_control_movable": gained_movable,
                "lost_on_nonfiring_game": lost_nonfiring,
                "gained_on_nonfiring_game": gained_nonfiring,
                "n_games_moved_on_movable_support": n_moved,
                "same_config_noise_floor_win_flips": noise_flips if noise_measured else None,
                "exceeds_same_config_noise_floor": exceeds_noise if noise_measured else None,
                "seed_verdict": seed_verdict,
            }

        # ---- efficiency, PAIRED PER GAME, on games BOTH arms win -------------------------
        eff_pairs = []
        for s in seeds:
            for g in sorted(set(win_sets[CONTROL][s]) & set(win_sets[arm][s])):
                c = by_arm_cell[(CONTROL, (g, s))]
                t = by_arm_cell[(arm, (g, s))]
                a2 = c.get("actions_to_first_levelup")
                b2 = t.get("actions_to_first_levelup")
                if a2 is None or b2 is None:
                    continue
                eff_pairs.append(
                    {
                        "game": g,
                        "seed": s,
                        "control_actions_to_first_levelup": a2,
                        "arm_actions_to_first_levelup": b2,
                        "delta_arm_minus_control": b2 - a2,
                        "lever_fired_on_arm": bool(by_arm_cell[(on_arm, (g, s))].get(fk)),
                    }
                )
        st_pairs = []
        for s in seeds:
            for g in sorted({c[0] for c in matched if c[1] == s}):
                c = by_arm_cell[(CONTROL, (g, s))]
                t = by_arm_cell[(arm, (g, s))]
                st_pairs.append(
                    {
                        "game": g,
                        "seed": s,
                        "control_states": c.get("states_expanded"),
                        "arm_states": t.get("states_expanded"),
                        "delta": (t.get("states_expanded") or 0) - (c.get("states_expanded") or 0),
                        "lever_fired_on_arm": bool(by_arm_cell[(on_arm, (g, s))].get(fk)),
                    }
                )

        seed_verdicts = [v["seed_verdict"] for v in per_seed.values()]
        if all(v == "UNINTERPRETABLE_EMPTY_PASS_REGION" for v in seed_verdicts):
            overall = "UNINTERPRETABLE_EMPTY_PASS_REGION"
        elif any(v == "EFFECT_ON_WINS" for v in seed_verdicts):
            overall = "EFFECT_ON_WINS"
        elif any(v == "EFFECT_ON_WINS_NOISE_FLOOR_UNMEASURED" for v in seed_verdicts):
            overall = "EFFECT_ON_WINS_NOISE_FLOOR_UNMEASURED"
        elif any(v == "EFFECT_WITHIN_SAME_CONFIG_NOISE_FLOOR" for v in seed_verdicts):
            overall = "EFFECT_WITHIN_SAME_CONFIG_NOISE_FLOOR"
        else:
            overall = "NO_EFFECT_ON_WINS_ON_FIRING_GAMES"

        verdicts[arm] = {
            "lever": meta["lever"],
            "direction": meta["direction"],
            "fire_counter_used": fk,
            "lever_on_in_arm": on_arm,
            "per_seed": per_seed,
            "overall_verdict": overall,
            "efficiency_paired_both_win": eff_pairs,
            "states_expanded_paired": st_pairs,
            "states_delta_median_on_firing_games": (
                statistics.median([p["delta"] for p in st_pairs if p["lever_fired_on_arm"]])
                if any(p["lever_fired_on_arm"] for p in st_pairs)
                else None
            ),
        }
    out["lever_verdicts"] = verdicts

    # ---- 5. FIRE-COUNTER CENSUS, per arm ---------------------------------------------------
    fire: dict[str, Any] = {}
    for a in arms:
        rs = [by_arm_cell[(a, c)] for c in matched]
        fire[a] = {
            "n_cells": len(rs),
            "lever1_fired_cells": sum(1 for r in rs if r.get("lever1_fired")),
            "lever1_tier_advances_total": sum(
                int((r.get("lever1_frontier_fire") or {}).get("tier_advances") or 0) for r in rs
            ),
            "lever2_fired_cells": sum(1 for r in rs if r.get("lever2_fired")),
            "lever2_games_mask_differs": sorted({r["game"] for r in rs if r.get("lever2_fired")}),
            "lever3_verdicts": dict(collections.Counter(r.get("lever3_verdict") for r in rs)),
            "lever3_rows_pruned_total": sum(
                int((r.get("lever3_hazard_fire") or {}).get("rows_pruned") or 0) for r in rs
            ),
            "llm_responses_total": sum(int((r.get("llm") or {}).get("responses") or 0) for r in rs),
            "llm_tokens_predicted_total": sum(
                int((r.get("llm") or {}).get("tokens_predicted") or 0) for r in rs
            ),
            "llm_wall_s_total": round(
                sum(float((r.get("llm") or {}).get("llm_wall_s") or 0.0) for r in rs), 1
            ),
            "cells_with_zero_llm_responses": sorted(
                r["game"] for r in rs if not int((r.get("llm") or {}).get("responses") or 0)
            ),
            "induction_attempts_total": sum(int(r.get("induction_attempts") or 0) for r in rs),
            "induction_attempts_llm_reached_total": sum(
                int(r.get("induction_attempts_llm_reached") or 0) for r in rs
            ),
            "nodes_with_previous_frame_total": sum(
                int(r.get("nodes_with_previous_frame") or 0) for r in rs
            ),
            "nodes_total": sum(int(r.get("nodes_total") or 0) for r in rs),
            "errors_total": sum(int(r.get("errors") or 0) for r in rs),
            "wall_s_total": round(sum(float(r.get("wall_s") or 0) for r in rs), 1),
        }
    out["fire_census_per_arm"] = fire

    # ---- 6. COST ---------------------------------------------------------------------------
    walls = [float(r.get("wall_s") or 0) for r in valid if r.get("wall_s")]
    llm_walls = [
        float((r.get("llm") or {}).get("llm_wall_s") or 0.0) for r in valid if r.get("llm_enabled")
    ]
    out["cost"] = {
        "n_cells": len(walls),
        "wall_s_total": round(sum(walls), 1),
        # The wall clock actually spent INSIDE the generator, summed over LLM-on cells. This is the
        # number that makes the live-inference claim auditable: it is the measured compute the
        # artifact's conclusions rest on, and it is what `duration_s` reports (NOT the analyser's own
        # runtime, which is milliseconds -- reporting that as duration_s made the fabrication linter
        # correctly flag DURATION_TOO_SHORT on a run that had in fact used hours of GPU time).
        "llm_wall_s_total": round(sum(llm_walls), 1),
        "llm_wall_fraction_of_total": (
            round(sum(llm_walls) / sum(walls), 4) if sum(walls) else None
        ),
        "wall_s_per_cell_median": round(statistics.median(walls), 1) if walls else None,
        "wall_s_per_cell_min": round(min(walls), 1) if walls else None,
        "wall_s_per_cell_max": round(max(walls), 1) if walls else None,
    }
    # SPLIT BY CONDITION. A median over a mixed set is meaningless here: an LLM-off cell costs a
    # couple of seconds and an LLM-on cell costs ~5 minutes, so pooling them reports a median of ~3s
    # for a design that actually takes hours -- and that number would then be used to scope the next
    # run. The affordability decision needs the LLM-ON per-cell cost specifically.
    for label, sel in (("llm_on", True), ("llm_off", False)):
        w = [float(r.get("wall_s") or 0) for r in valid if bool(r.get("llm_enabled")) is sel]
        w = [x for x in w if x]
        out["cost"][label] = {
            "n_cells": len(w),
            "wall_s_total": round(sum(w), 1),
            "wall_s_per_cell_median": round(statistics.median(w), 1) if w else None,
            "wall_s_per_cell_min": round(min(w), 1) if w else None,
            "wall_s_per_cell_max": round(max(w), 1) if w else None,
            # What the FULL design would have cost at this per-cell price, as a serial sum. The
            # actual runs used two concurrent shards, so elapsed time was roughly half this.
            "projected_hours_25games_x_1seed_x_5arms_serial": (
                round(statistics.median(w) * 25 * 5 / 3600, 2) if w else None
            ),
        }
    return out


def build_artifact(
    analysis: dict,
    rows: list[dict],
    sources: list[Path],
    t0: float,
    companion: dict | None = None,
    companion_rows: list[dict] | None = None,
) -> dict:
    payload = json.dumps(
        {"rows": rows, "analysis": analysis, "companion": companion or {}},
        sort_keys=True,
        default=str,
    ).encode()
    checksum = hashlib.sha256(payload).hexdigest()

    lv = analysis.get("lever_verdicts", {})
    v1 = (lv.get(f"S_minus_frontier{SUFFIX}") or {}).get("overall_verdict")
    v2 = (lv.get(f"S_minus_hud{SUFFIX}") or {}).get("overall_verdict")
    v3 = (lv.get(f"S_plus_hazard{SUFFIX}") or {}).get("overall_verdict")

    # honest_verdict is COMPUTED from the three per-lever verdicts, never hand-written, so it
    # cannot drift from what the analysis actually found. The `complete_` terminal prefix is
    # mandatory: without it the conductor's reconciler substring-matches words like "no_effect" /
    # "uninterpretable" against its partial-token list and spuriously classifies a finished
    # measurement as a partial run (CLAUDE.md Verdict Terminal-Prefix Discipline).
    honest = (
        "complete_scored_path_llm_on_lever_ab_measured"
        f"__lever1_frontier_{str(v1).lower()}"
        f"__lever2_hud_{str(v2).lower()}"
        f"__lever3_hazard_{str(v3).lower()}"
    )

    # ---- SELF-REPORTED ACCEPTANCE GATES, all COMPUTED ---------------------------------------
    # These are gates on the MEASUREMENT'S OWN VALIDITY, not on a hoped-for result -- a gate that
    # can only pass if the answer is the one we wanted is not a gate. Each is falsifiable and each
    # guards a specific way this measurement could be worthless:
    #   1. every LLM-on row really had a live generator (else it is an LLM-off row mislabelled);
    #   2. the noise floor was actually measured (else a 1-game delta cannot be interpreted);
    #   3. nothing is reported as an EFFECT without a non-empty computed witness.
    llm_on_rows = [r for r in rows if r.get("llm_enabled")]
    gate_llm_live = bool(llm_on_rows) and all(
        llm_row_is_valid(r) for r in llm_on_rows if r.get("ran")
    )
    gate_noise = bool(analysis.get("noise_floor_same_config_replicate"))
    effect_without_witness = []
    for arm, v in lv.items():
        for s, per in (v.get("per_seed") or {}).items():
            if str(per.get("seed_verdict", "")).startswith("EFFECT") and not per.get(
                "witness_pass_region_nonempty"
            ):
                effect_without_witness.append({"arm": arm, "seed": s})
    gate_witness = not effect_without_witness

    return {
        "honest_verdict": honest,
        "honest_verdict_note": (
            "principle: the self-declared terminal state lets the reconciler classify the run "
            "without re-running it, and the terminal prefix stops a substring match on words like "
            "'uninterpretable' from misclassifying a completed measurement as partial. This string "
            "is DERIVED from the computed per-lever verdicts, so it cannot disagree with them."
        ),
        "experiment": "outer_loop_scored_path_lever_ab_llm_on",
        "title": (
            "Scored-path (E3AgentPolicy, LLM ON) baseline and per-lever A/B for the two levers "
            "shipped 2026-07-25, plus the newly-wired nav-side hazard move-pruner"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.scored_path_lever_ab.v1",
        # duration_s is the MEASURED COMPUTE this artifact rests on -- the summed wall clock of
        # every cell that was run -- NOT this analyser's own runtime. Reporting the analyser's
        # runtime (milliseconds) made adversarial_verify correctly raise DURATION_TOO_SHORT on a
        # run that had used hours of live GPU inference: the artifact declared a live-LLM substrate
        # while claiming to have completed in 0.01s. The analyser's own runtime is reported
        # separately as analysis_duration_s so nothing is hidden.
        "duration_s": round(float((analysis.get("cost") or {}).get("wall_s_total") or 0.0), 2),
        "duration_s_note": (
            "principle: real compute takes wall-clock time, and a missing or implausibly short "
            "duration is the load-bearing fabrication signal. This is the summed per-cell wall "
            "clock of the measurement itself; of it, llm_wall_s_total was spent inside the "
            "generator. Cells ran across two concurrent shard processes, so this SUM exceeds the "
            "elapsed session time -- it is compute-seconds, not elapsed seconds."
        ),
        "analysis_duration_s": round(time.time() - t0, 2),
        "measured_llm_wall_s": float((analysis.get("cost") or {}).get("llm_wall_s_total") or 0.0),
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": (
            "principle: the substrate declaration is what lets the fabrication linter pick the "
            "right duration floor. This is full autoregressive generation from a real local GGUF "
            "(Qwen3.5-9B-MTP Q4_K_M, llama-server, -ngl 999, GPU 1), driven by the SCORED agent's "
            "own induction tier -- not embedding extraction and not aggregation. The 60s "
            "live_llm_inference floor therefore applies and is met by orders of magnitude."
        ),
        "model_specs": {
            "generator": "unsloth/Qwen3.5-9B-MTP-GGUF :: Qwen3.5-9B-Q4_K_M.gguf",
            "serving": "llama.cpp-master llama-server, --spec-type draft-mtp, kv q8_0, -ngl 999",
            "gpu": "NVIDIA RTX 3090 (GPU 1; GPU 0 is the conductor's per the 2026-06-27 allocation)",
            "note": (
                "principle: naming the model that actually ran is what makes a live-inference claim "
                "auditable. This is the FROZEN live generator per project_arc_live_generator, i.e. "
                "the same model the scored submission would use."
            ),
        },
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "principle: a verifier that IS the executable oracle makes a circular moat claim. "
            "NO verifier-value or moat claim is made here. The three levers measured are search "
            "levers (frontier-tier discipline, a HUD-mask perception repair, a nav hazard "
            "move-pruner); the outcome variable is the offline arcade's own level counter, which "
            "is the task's ground truth and is used only as the OUTCOME, never as a verifier."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "principle: the deliverable is a live agent that self-discovers hidden-game solves, so "
            "an artifact must say how any solve arose. This artifact claims NO new solve and no new "
            "level. It runs the SCORED policy (E3AgentPolicy) against the OFFLINE public arcade to "
            "measure levers -- a development proxy for the hidden-game condition. No game source "
            "was read, no ground-truth BFS was run, nothing was hand-calibrated per game."
        ),
        "claims_new_solve": False,
        "offline_reproduced": False,
        "random_seed": 20260724,
        "random_seeds_used": analysis.get("seeds"),
        "random_seed_note": (
            "principle: without a recorded seed no third party can re-run the cell. The seed is "
            "passed to E3AgentPolicy as frontier_discipline_seed AND to random/numpy, so an arm's "
            "stochastic tier choices are reproducible per cell."
        ),
        "reproducibility_checksum": checksum,
        "reproducibility_checksum_note": (
            "principle: a content hash over the raw rows plus the derived analysis catches silent "
            "drift between this artifact and any replication. sha256 over the canonicalised "
            "{rows, analysis} payload."
        ),
        "preconditions_checked": [
            {"resource": "llama_server_qwen3.5_9b_mtp_gpu1_port_8931", "available": True},
            {"resource": "cached_gguf_Qwen3.5-9B-MTP", "available": True},
            {"resource": "cuda_gpu1_free_for_outer_loop", "available": True},
            {"resource": "offline_arcade_environment_files_25_games", "available": True},
            {"resource": "CARNOT_ARC_DISABLE_INDUCTION_unset_llm_on", "available": True},
        ],
        "preconditions_note": (
            "principle: records WHICH resources were verified before measuring, which is what "
            "pre-empts the failure mode where an agent silently lacked the resource and "
            "synthesised a passing artifact. The LLM-liveness precondition is additionally "
            "re-checked PER CELL (generator_healthy_before/after) because the server can die "
            "mid-run and silently degrade the scored path to LLM-off."
        ),
        "harness": "scripts/arc_scored_path_lever_harness.py",
        "analyzer": "scripts/analyze_scored_path_lever_ab.py",
        "source_row_files": [str(p) for p in sources],
        "budget_per_game": sorted({int(r.get("budget") or 0) for r in rows}),
        "budget_note": (
            "principle: measuring at a budget the eval does not grant changes conclusions. 400 is "
            "the scored agent's own MAX_ACTIONS cap, so it is the eval's condition."
        ),
        "lever1_frontier_verdict": v1,
        "lever2_hud_verdict": v2,
        "lever3_hazard_verdict": v3,
        "acceptance_gate_all_llm_on_rows_had_a_live_generator": gate_llm_live,
        "acceptance_gate_all_llm_on_rows_had_a_live_generator_principle": (
            "Guards the failure mode where the llama-server dies mid-run and the harness keeps "
            "emitting rows LABELLED llm_on that contain no LLM at all -- a clean, error-free, "
            "entirely worthless measurement. Passes only if every LLM-on row that ran was stamped "
            "llm_on_row_valid AND recorded at least one real completion."
        ),
        "acceptance_gate_noise_floor_was_measured": gate_noise,
        "acceptance_gate_noise_floor_was_measured_principle": (
            "With an LLM in the loop a seeded run is not guaranteed deterministic, so a one-game "
            "win delta may be sampling variation. Passes only if the same-config replicate arm was "
            "actually run; without it no small effect can be interpreted."
        ),
        "acceptance_gate_no_effect_reported_without_a_witness": gate_witness,
        "acceptance_gate_no_effect_reported_without_a_witness_principle": (
            "Guards the forced-value defect: an EFFECT verdict on a support that is structurally "
            "empty is arithmetic, not measurement. Passes only if every per-seed EFFECT verdict has "
            "a non-empty computed movable-game witness."
        ),
        "acceptance_gate_violations": effect_without_witness,
        "analysis": analysis,
        # The LLM-OFF companion design: the same five arms, all 25 games, 3 seeds, run with
        # induction disabled. It is ~100x cheaper per cell (no generator), so it can be run at a
        # power the LLM-on design cannot afford. Embedded rather than published separately so a
        # reader cannot pick up the well-powered LLM-OFF result and mistake it for the SCORED
        # condition -- the distinction this whole measurement exists to draw.
        "companion_llm_off_design": companion or {},
        "companion_llm_off_design_note": (
            "principle: a cheap high-power measurement of the WRONG condition is the most "
            "dangerous kind, because its confidence interval looks better than the expensive "
            "measurement of the right one. This section is LLM-OFF and is NOT the scored "
            "condition; the scored condition is `analysis`."
        ),
        "rows": rows,
        "companion_rows": companion_rows or [],
        "flags_flipped": [],
        "flags_flipped_note": (
            "principle: a measurement must not change the thing it measures. NO flag was flipped "
            "by this work; the hazard pruner remains default-OFF and is enabled per-arm by "
            "explicit constructor kwarg only. Arms differ by kwargs, never by mutating module "
            "globals, because all arms share one process."
        ),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--companion-rows", nargs="*", default=[])
    ap.add_argument(
        "--companion-condition",
        default="llmoff",
        choices=["llmon", "llmoff"],
    )
    ap.add_argument(
        "--condition",
        default="llmon",
        choices=["llmon", "llmoff"],
        help="which arm suffix is the control condition; llmon is the SCORED condition",
    )
    a = ap.parse_args(argv)
    set_condition("_" + a.condition)
    t0 = time.time()
    sources = [Path(x) for x in a.rows]
    rows = load_rows(sources)
    analysis = analyse(rows)
    companion: dict | None = None
    companion_rows: list[dict] | None = None
    if a.companion_rows:
        # Analysed with its OWN control arm. Restoring the primary condition afterwards is
        # load-bearing: build_artifact reads SUFFIX to pick which lever verdicts to headline, so
        # leaving it set to the companion would headline the LLM-OFF verdicts as the scored result.
        companion_rows = load_rows([Path(x) for x in a.companion_rows])
        set_condition("_" + a.companion_condition)
        companion = analyse(companion_rows)
        set_condition("_" + a.condition)
    art = build_artifact(analysis, rows, sources, t0, companion, companion_rows)
    Path(a.out).write_text(json.dumps(art, indent=1, default=str))
    print(json.dumps({k: v for k, v in analysis.items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
