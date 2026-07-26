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
 7. THE GAME-UNIT EXACT SIGN TEST, on every lever verdict (added 2026-07-26). The first version of
    this analyser emitted three lever verdicts and ZERO statistical inference -- no sign test, no
    p-value, no statement of what the design could reach -- while the project's own top
    known-issues entry, one day old, names the exact one-sided sign test ON THE GAME UNIT as the
    standard and had just used it to WITHDRAW a sibling HUD claim. A hidden game is a fresh draw
    from the game distribution, so seeds cannot widen a support: three seeds of the same two movers
    is still a two-mover support with a best-reachable p of 0.125. Every verdict therefore carries
    the test AND `smallest_reachable_p_at_this_n`, which turns "not significant" into the
    actionable "this design could not have been".
 8. FIRE COUNTERS ARE RECOMPUTED FROM RAW DIAGNOSTICS, not trusted (added 2026-07-26). Lever 2's
    harness stamp was measured to be ANTI-CORRELATED with its own lever, reading 0 in all 430 cells
    while the lever demonstrably fired. A counter can be broken in the direction that HIDES an
    effect just as easily as in the direction that invents one, and the hiding direction is worse
    because it looks like a clean null. See `recomputed_lever2_fired`.
 9. THE BUDGET IS PART OF EVERY CONCLUSION. Lever orderings REVERSE between budget 400 (the shipped
    agent's self-imposed loop guard) and budget 2000 (enough actions for the corpus's measured
    first-win costs, which span 20 to 1747 actions). Neither is "the eval's condition" in the sense
    of a constraint that cannot be changed. `budget_note` and
    `prior_measurements_that_must_be_reconciled_against` carry this, and no flag recommendation may
    rest on one budget alone.

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
# Needed for `carnot.agentic.arc_hud_row_schema`, the SHARED row-schema module this analyser and
# both writing harnesses now import instead of each keeping its own copy of the key list.
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

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


# THE COMPATIBILITY READ IS SHARED WITH THE WRITERS (2026-07-26).
# `backfill_hud_flat_fields` used to be DEFINED here, so the analyser's idea of the row schema and
# the harnesses' idea of it were two separate pieces of code that had to be kept in agreement by
# memory. They are now one module: `carnot.agentic.arc_hud_row_schema` owns the key list, the flat
# projection both writers emit, the fire predicate, the back-fill, and the `lever2_scoreable`
# distinction between "the lever did not fire" and "this row cannot say". Re-exported under the
# original name because the tests and the sibling harness import it from here.
from carnot.agentic.arc_hud_row_schema import (  # noqa: E402
    HUD_ROW_KEYS,
    backfill_hud_flat_fields,
    hud_lever_fired,
    lever2_scoreable,
)

_HUD_FLAT_KEYS = HUD_ROW_KEYS


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            r["_source"] = p.name
            # Single chokepoint: every row entering the analyser is made readable at both HUD
            # addresses here, so no downstream call site can be the one that was forgotten.
            r["_hud_row_schema"] = backfill_hud_flat_fields(r)
            rows.append(r)
    return rows


def cell_key(r: dict) -> tuple:
    return (r["game"], r["seed"])


def recomputed_lever2_fired(r: dict) -> bool:
    """Recompute lever 2's fire flag from the row's own RECORDED HUD diagnostics.

    WHY THIS IS RECOMPUTED HERE RATHER THAN TRUSTED (measured defect, 2026-07-26). The harness's
    first `lever2_fired` predicate required the ALREADY-SHIPPED `auto_hud_mask` classifier to have
    produced a mask before the REPAIRED detector's mask could count as a difference. That is
    anti-correlated with the lever: the shipped classifier returns None on exactly the two games
    (r11l, tn36) where the repaired detector resolves a mask, so the counter read 0 in all 430
    cells of the first scored-path run while the lever was demonstrably firing -- resolved mask
    None -> 64 cells on r11l with `states_expanded` 319 -> 41, and None -> 61 with 49 -> 17 on
    tn36. Recomputing here means every already-recorded row is corrected without re-running hours
    of GPU cells, and it follows the same defence-in-depth pattern as `llm_row_is_valid`: the
    harness's stamp is kept in the row as diagnostics, the disagreement is reported explicitly, and
    the analysis uses the recomputed value.

    A MASK APPEARING WHERE THE SHIPPED CONFIG HAD NONE IS THE LEVER'S STRONGEST FIRING, not a
    non-event, so a falsy shipped digest counts as a real difference. Digests are compared, never
    cell COUNTS -- the 2026-07-25 gate compared counts and therefore read a same-size different
    mask as "no change".

    READS EITHER ROW SCHEMA (added 2026-07-26 with the population fix). The nested
    `lever2_hud_fire` dict is preferred when present; otherwise the FLAT `hud_mask_*` keys are used,
    so exp5836-schema rows are scored instead of silently reading False -- which is what happened to
    all 1713 rows of `results/cptb_20260726_cells/*.jsonl.gz`.

    A MISSING `hud_shipped_mask_digest` KEY IS *UNKNOWN*, NOT None. `digest != None` is true for
    every resolved mask, so treating an unrecorded shipped digest as None would arithmetically force
    "fired" on every resolved cell -- 1058 of those 1713 cptb rows, none of which recorded a
    shipped-side comparison at all. Such a row is scored as NOT fired, and its
    `_hud_row_schema`/`hud_diagnostics_readable` tags are what tell a reader it carries no evidence
    rather than negative evidence.

    ONE PREDICATE, NOT TWO (2026-07-26). The rule used to be written out here AND in the harness,
    which meant the two could silently disagree about a row -- and a disagreement between the
    recorded stamp and the recomputed value is precisely what this function exists to surface, so
    it must not itself be a second implementation. It now delegates to the shared
    `carnot.agentic.arc_hud_row_schema.hud_lever_fired`; the only thing decided here is WHICH
    address on the row to hand it (nested when present, else the flat keys).
    """
    nested = r.get("lever2_hud_fire")
    return hud_lever_fired(nested if isinstance(nested, dict) and nested else r)


# Fire keys whose value the analyser RECOMPUTES from the row's raw diagnostics instead of trusting
# the harness's stamp. Anything not listed here is read straight off the row.
_RECOMPUTED_FIRE = {"lever2_fired": recomputed_lever2_fired}


def fire_flag(r: dict, fire_key: str) -> bool:
    """Read a lever's fire flag, recomputing it from raw diagnostics where we have a recomputation.

    Every place the analysis asks "did this lever fire in this cell?" goes through here, so a
    correction lands everywhere at once rather than in whichever call site was remembered.
    """
    fn = _RECOMPUTED_FIRE.get(fire_key)
    if fn is not None:
        return fn(r)
    return bool(r.get(fire_key))


def exact_one_sided_sign_test(n_favour: int, n_against: int) -> dict[str, Any]:
    """Exact one-sided sign test on the GAME unit, plus the smallest p this design could reach.

    WHY THE GAME IS THE UNIT, AND WHY SEEDS ARE NOT (this project's own standing rule, stated in
    `ops/known-issues.md`'s 2026-07-26 entry and in the convention-transfer battery's jackknife
    principle): a hidden game is a fresh draw from the game distribution, so generalisation is
    over GAMES. Seeds are re-runs of the same games; adding seeds can make a per-seed result more
    reproducible but cannot widen the support. The known-issues entry states the consequence
    directly -- "a one-game support has an exact sign-test floor of p=0.5. It cannot be established
    at p<=0.05 at ANY seed count ... needs MORE GAMES that it moves, not more seeds" -- and it was
    used ONE DAY EARLIER to withdraw a sibling HUD claim as forced rather than measured. Reporting
    a lever verdict without this test is therefore inconsistent with the standard this project
    already applies to its own siblings.

    `smallest_reachable_p_at_this_n` is the crucial second number: with n discordant games the best
    achievable p is 2**-n, so a design with 3 movers CANNOT clear 0.05 whatever the outcome. That
    turns "did not reach significance" into the actionable statement "this design cannot reach it".

    THE THIRD NUMBER, AND WHY IT WAS ADDED (2026-07-26, after the budget-2000 reconciliation).
    A one-sided test reports only the tail in the ARM'S favour, so a result that points the OTHER
    WAY comes back as a LARGE p -- and a large p reads to a human as "no effect". It is not: at
    budget 2000 the frontier arm's removal LOSES 4 games and gains 2, which this function reports
    as p=0.8906. Read alone that number invites exactly the wrong conclusion ("nothing here"), when
    the honest reading is "the data favours the CONTROL, 4 games to 2, and that is a REVERSAL of the
    budget-400 direction". So `p_one_sided_exact_opposite_direction` and `direction_favoured` are
    reported alongside, and a reversal becomes a number instead of an inference the reader has to
    make by inverting a tail probability in their head.
    """
    n = int(n_favour) + int(n_against)
    if n == 0:
        return {
            "n_games_favouring": 0,
            "n_games_against": 0,
            "n_discordant_games": 0,
            "p_one_sided_exact": None,
            "p_one_sided_exact_opposite_direction": None,
            "direction_favoured": "no_discordant_game",
            "clears_p_0_05": False,
            "opposite_direction_clears_p_0_05": False,
            "undefined_because_no_discordant_game": True,
            "smallest_reachable_p_at_this_n": None,
            "underpowered_support": True,
        }
    # P(X >= n_favour) under X ~ Binomial(n, 0.5), computed exactly with integer binomials so there
    # is no floating-point drift at the tiny n this project actually has.
    from math import comb

    tail = sum(comb(n, k) for k in range(int(n_favour), n + 1))
    p = tail / (2**n)
    # The mirror tail: P(X >= n_against), i.e. the same exact test run in the CONTROL'S favour.
    tail_opp = sum(comb(n, k) for k in range(int(n_against), n + 1))
    p_opp = tail_opp / (2**n)
    floor_p = 1.0 / (2**n)
    return {
        "n_games_favouring": int(n_favour),
        "n_games_against": int(n_against),
        "n_discordant_games": n,
        "p_one_sided_exact": round(p, 4),
        "p_one_sided_exact_opposite_direction": round(p_opp, 4),
        "direction_favoured": (
            "arm"
            if int(n_favour) > int(n_against)
            else ("control" if int(n_against) > int(n_favour) else "tie")
        ),
        "clears_p_0_05": bool(p <= 0.05),
        "opposite_direction_clears_p_0_05": bool(p_opp <= 0.05),
        "undefined_because_no_discordant_game": False,
        "smallest_reachable_p_at_this_n": round(floor_p, 4),
        # A support so small that NO outcome could have cleared 0.05. This is a property of the
        # design, not of the result, and it is the difference between "we measured no effect" and
        # "this measurement could not have found one".
        "underpowered_support": bool(floor_p > 0.05),
    }


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

    THE CRITERION, AND WHY `generator_healthy_before` IS NOT PART OF IT. A first draft required
    health on BOTH sides and was measured to be WRONG in the other direction: the `S_llmon` cell on
    tn36 -- the single most decision-relevant cell in the whole LLM-on design, because it is the
    control for the game where the frontier lever changes the outcome -- has
    `generator_healthy_before: False` (the server happened to be inside a systemd restart when the
    cell began) yet returned SIX completions totalling 15,514 predicted tokens with
    `generator_healthy_after: True`. The LLM provably ran. Excluding that cell would have deleted
    the control for the only game that discriminates, purely because of when the cell started.

    What actually needs establishing is that the generator SERVED this cell:
      * `responses > 0` -- real completions, counted from llama.cpp's own `timings` block in each
        response rather than estimated, so this cannot be satisfied without the server answering;
      * `generator_healthy_after` -- the server was alive when the cell ended, so the completions
        were not the truncated output of a dying server and the rest of the cell did not silently
        run LLM-off.
    Health BEFORE the cell says only whether the first induction attempt might have failed; it says
    nothing about whether the LLM ran. It stays in the row as diagnostics.
    """
    if not r.get("ran"):
        return False
    if not r.get("llm_enabled"):
        return True  # an LLM-off row makes no LLM claim to invalidate
    llm = r.get("llm") or {}
    return bool(int(llm.get("responses") or 0) > 0 and r.get("generator_healthy_after"))


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
    out["noise_floor_same_config_replicate_note"] = (
        "SAME-SEED replication. This establishes CAUSAL ATTRIBUTION -- a same-seed difference "
        "between two identically-flagged arms can only come from run-to-run variation, so if it is "
        "zero then any difference an ablation arm shows IS caused by the flag change. It does NOT "
        "bound the variance a win-set comparison generalises over: re-running the same seed is "
        "structurally near-deterministic here, so this floor is expected to be 0 and a gate that "
        "only asks 'did anything change at all' is weak. The generalisation-relevant floor is "
        "`noise_floor_control_across_seeds` below."
    )

    # ---- 3b-ii. THE CROSS-SEED NOISE FLOOR (the one a win delta must actually be read against) --
    # WHY BOTH FLOORS ARE NEEDED. The same-config replicate floor above is a SAME-SEED comparison,
    # so it is structurally near-zero and reduces `exceeds_same_config_noise_floor` to "did anything
    # change at all". But the quantity a lever verdict wants to generalise -- "this configuration
    # wins this SET of games" -- is measurably unstable ACROSS seeds. Directly measured here: the
    # control arm's own win set moves between seeds (e.g. 3 -> 4 -> 4 wins with DIFFERENT membership
    # in the budget-400 LLM-off design), and the convention-transfer battery independently records
    # the shipped and frontier arms as `measured_deterministic: false` with 54 of 75
    # game-condition cells varying across seeds (the uniform-random tier is seeded, so a different
    # seed is a different search). A 1-2 game lever delta is therefore the SAME MAGNITUDE as the
    # control's own seed-to-seed movement, which is exactly why the game-unit sign test below, not
    # the same-seed floor, is what decides whether the lever's direction is established.
    cross_seed: dict[str, Any] = {}
    if CONTROL in arms and len(seeds) > 1:
        ctrl_cells = [c for c in all_cells if (CONTROL, c) in by_arm_cell]
        games_all_seeds = sorted(
            {c[0] for c in ctrl_cells}
            - {
                c[0]
                for c in all_cells
                if any((CONTROL, (c[0], s)) not in by_arm_cell for s in seeds)
            }
        )
        per_pair = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                w1 = {g for g in win_sets[CONTROL][s1] if g in games_all_seeds}
                w2 = {g for g in win_sets[CONTROL][s2] if g in games_all_seeds}
                per_pair.append(
                    {
                        "seed_a": s1,
                        "seed_b": s2,
                        "win_set_a": sorted(w1),
                        "win_set_b": sorted(w2),
                        "win_flips_across_seeds": sorted(w1 ^ w2),
                        "n_win_flips_across_seeds": len(w1 ^ w2),
                    }
                )
        flips = [p["n_win_flips_across_seeds"] for p in per_pair]
        won_every = sorted(
            g for g in games_all_seeds if all(g in win_sets[CONTROL][s] for s in seeds)
        )
        won_some = sorted(
            g for g in games_all_seeds if any(g in win_sets[CONTROL][s] for s in seeds)
        )
        cross_seed = {
            "arm": CONTROL,
            "n_games_measured_on_every_seed": len(games_all_seeds),
            "seed_pairs": per_pair,
            "max_win_flips_across_any_seed_pair": max(flips) if flips else None,
            "median_win_flips_across_seed_pairs": (statistics.median(flips) if flips else None),
            "games_won_on_every_seed": won_every,
            "games_won_on_at_least_one_seed": won_some,
            "n_games_unstable_across_seeds": len(set(won_some) - set(won_every)),
            "control_is_stable_across_seeds": bool(flips) and max(flips) == 0,
        }
    else:
        # NOT MEASURABLE is not the same as MEASURED ZERO, and an empty dict cannot tell them apart.
        # The scored (LLM-on) design has ONE seed, so this floor is structurally unavailable there --
        # which is itself a limitation of that design, and a reader must be told so rather than
        # seeing a blank and assuming it was measured and clean.
        cross_seed = {
            "not_measurable": True,
            "reason": (
                f"needs >=2 seeds of the control arm {CONTROL!r}; this design has "
                f"{len(seeds)} seed(s). A single-seed design cannot bound its own "
                "seed-to-seed win-set movement, so no win delta measured here may be "
                "called larger than the noise it generalises over."
            ),
            "n_seeds_available": len(seeds),
            "control_is_stable_across_seeds": None,
        }
    out["noise_floor_control_across_seeds"] = cross_seed
    out["noise_floor_control_across_seeds_note"] = (
        "The CONTROL arm compared against ITSELF on different seeds, over games measured on every "
        "seed. This is the variance a win-set claim generalises over. A lever delta of the same "
        "magnitude as `max_win_flips_across_any_seed_pair` is not thereby refuted -- the lever "
        "comparison is matched-seed and so remains causally attributable -- but it means the delta "
        "is not larger than the setup's own seed-to-seed movement, and the game-unit sign test is "
        "what must carry the conclusion."
    )

    # ---- 3c. WHAT THE LLM ITSELF CONTRIBUTES ------------------------------------------------
    # `S_llmoff` is the SAME E3AgentPolicy configuration with induction disabled -- i.e. the
    # 2026-07-25 measurement condition, run on the scored path at the eval's own budget. Without
    # this, "the scored path with the LLM on wins k games" has no reference to be read against, and
    # the question the whole exercise exists to answer (does the LLM tier help where it scores?)
    # stays open. Reported as a comparison, never folded into a lever verdict: turning the LLM off
    # is not one of the levers under test.
    # ---- 3d. THE LLM PLAN-CHANNEL CENSUS ----------------------------------------------------
    # COMPUTED, because the prose version of this was WRONG. The first write-up of this measurement
    # said the induced world model is "rejected by a verifier gate every single time" with
    # "planned=0". The rows say otherwise: `induction_planned` is 1 in one of the 30 LLM-on rows.
    # That single cell is the ONLY place in the entire run where the LLM -> plan channel opened, and
    # it is therefore the POSITIVE CONTROL the inertness null needs -- without it, "the gate
    # correctly rejects a weak model" and "the plan path never influences behaviour" are
    # indistinguishable, which is the brief's own central lesson about nulls. It is emitted here so
    # the claim cannot be restated from memory, and so its pairing status (whether its matched
    # control row is valid) is visible rather than buried.
    llm_rows = [r for r in rows if r.get("llm_enabled") and r.get("ran")]
    planned_rows = [r for r in llm_rows if int(r.get("induction_planned") or 0) > 0]
    reason_counter: collections.Counter = collections.Counter()
    for r in llm_rows:
        reason_counter.update(r.get("induction_reasons") or {})
    out["llm_plan_channel_census"] = {
        "n_llm_on_rows_that_ran": len(llm_rows),
        "induction_planned_distribution": dict(
            collections.Counter(int(r.get("induction_planned") or 0) for r in llm_rows)
        ),
        "n_rows_where_plan_channel_opened": len(planned_rows),
        "induction_reason_counts": dict(reason_counter),
        "rows_where_plan_channel_opened": [
            {
                "arm": r.get("arm"),
                "game": r.get("game"),
                "seed": r.get("seed"),
                "levels": r.get("levels"),
                "induction_attempts": r.get("induction_attempts"),
                "induction_attempts_llm_reached": r.get("induction_attempts_llm_reached"),
                "induction_reasons": r.get("induction_reasons"),
                "llm_responses": (r.get("llm") or {}).get("responses"),
                # Is the matched control cell for this row a VALID LLM-on datum? If not, this
                # positive control is UNPAIRED and cannot yet be used to attribute the outcome.
                "matched_control_row_is_valid": bool((CONTROL, cell_key(r)) in by_arm_cell),
            }
            for r in planned_rows
        ],
        "note": (
            "The plan channel opening at all is what distinguishes 'the verifier gate rejects a "
            "weak induced model' (a real, reportable finding) from 'the plan path is structurally "
            "unreachable on this path' (a wiring defect masquerading as a finding). A run in which "
            "it NEVER opened has no positive control and its inertness claim is not admissible. "
            "`matched_control_row_is_valid=false` means the one opening cell has no comparable "
            "control, so re-running that cell with a healthy generator is the concrete next step."
        ),
    }

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
                return fire_flag(by_arm_cell[(on_arm, (g, s))], fk)

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

            # THE VERDICT NAME SAYS ONLY WHAT THE GATE ESTABLISHES. Renamed 2026-07-26 from
            # EFFECT_ON_WINS: the gate's noise floor is a SAME-SEED replicate, which is
            # structurally near-zero, so passing it establishes "this win difference is
            # ATTRIBUTABLE to the flag change" and nothing more. It does NOT establish that the
            # difference generalises to a fresh game draw -- that is what the per-lever game-unit
            # sign test below is for. "EFFECT_ON_WINS" read as a corpus-level effect claim, which
            # a 1-2 game matched-seed difference cannot support.
            if not witness_pass_region_nonempty:
                seed_verdict = "UNINTERPRETABLE_EMPTY_PASS_REGION"
            elif n_moved == 0:
                seed_verdict = "NO_WIN_DIFFERENCE"
            elif not noise_measured:
                seed_verdict = "ATTRIBUTABLE_WIN_DIFFERENCE_NOISE_FLOOR_UNMEASURED"
            elif not exceeds_noise:
                seed_verdict = "WIN_DIFFERENCE_WITHIN_SAME_CONFIG_NOISE_FLOOR"
            else:
                seed_verdict = "ATTRIBUTABLE_WIN_DIFFERENCE"

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
                        "lever_fired_on_arm": fire_flag(by_arm_cell[(on_arm, (g, s))], fk),
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
                        "lever_fired_on_arm": fire_flag(by_arm_cell[(on_arm, (g, s))], fk),
                    }
                )

        seed_verdicts = [v["seed_verdict"] for v in per_seed.values()]
        if all(v == "UNINTERPRETABLE_EMPTY_PASS_REGION" for v in seed_verdicts):
            overall = "UNINTERPRETABLE_EMPTY_PASS_REGION"
        elif any(v == "ATTRIBUTABLE_WIN_DIFFERENCE" for v in seed_verdicts):
            overall = "ATTRIBUTABLE_WIN_DIFFERENCE"
        elif any(v == "ATTRIBUTABLE_WIN_DIFFERENCE_NOISE_FLOOR_UNMEASURED" for v in seed_verdicts):
            overall = "ATTRIBUTABLE_WIN_DIFFERENCE_NOISE_FLOOR_UNMEASURED"
        elif any(v == "WIN_DIFFERENCE_WITHIN_SAME_CONFIG_NOISE_FLOOR" for v in seed_verdicts):
            overall = "WIN_DIFFERENCE_WITHIN_SAME_CONFIG_NOISE_FLOOR"
        else:
            overall = "NO_WIN_DIFFERENCE_ON_FIRING_GAMES"

        # ---- THE GAME-UNIT EXACT SIGN TEST -----------------------------------------------
        # A game is a MOVER for this lever if, pooled over the seeds where both arms have a valid
        # row, the arm wins it on strictly more seeds than the control (favours the arm) or on
        # strictly fewer (favours the control). Ties contribute nothing, exactly as a sign test
        # requires. Only games in the per-seed MOVABLE support can count, so a win difference on a
        # game where the lever never fired cannot enter the test -- same wrong-mechanism guard the
        # per-seed delta uses.
        #
        # WHY THIS AND NOT "CONSISTENT ACROSS SEEDS". Direction consistency across seeds is a
        # statement about re-running the SAME games; the claim being made is about a FRESH game.
        # Three seeds of the same two movers is still a two-mover support. This block is what makes
        # that visible, and `smallest_reachable_p_at_this_n` says outright when the design could not
        # have cleared 0.05 under ANY outcome.
        arm_seed_wins: dict[str, int] = collections.Counter()
        ctrl_seed_wins: dict[str, int] = collections.Counter()
        eligible_seeds: dict[str, int] = collections.Counter()
        for s in seeds:
            per = per_seed.get(str(s)) or {}
            mv = set(per.get("witness_movable_games") or [])
            for g in mv:
                eligible_seeds[g] += 1
                if g in set(per.get("arm_win_set") or []):
                    arm_seed_wins[g] += 1
                if g in set(per.get("control_win_set") or []):
                    ctrl_seed_wins[g] += 1
        movers_favouring_arm = sorted(
            g for g in eligible_seeds if arm_seed_wins[g] > ctrl_seed_wins[g]
        )
        movers_favouring_control = sorted(
            g for g in eligible_seeds if ctrl_seed_wins[g] > arm_seed_wins[g]
        )
        sign = exact_one_sided_sign_test(len(movers_favouring_arm), len(movers_favouring_control))
        sign.update(
            {
                "unit": "game",
                "movers_favouring_arm": movers_favouring_arm,
                "movers_favouring_control": movers_favouring_control,
                "per_game_seed_win_counts": {
                    g: {
                        "arm_wins_on_n_seeds": arm_seed_wins[g],
                        "control_wins_on_n_seeds": ctrl_seed_wins[g],
                        "eligible_seeds": eligible_seeds[g],
                    }
                    for g in sorted(eligible_seeds)
                },
                "principle": (
                    "The GAME is the replication unit because a hidden game is a fresh draw from "
                    "the game distribution; seeds are re-runs of the same games and cannot widen "
                    "the support. This project applied exactly this test one day earlier to "
                    "withdraw a sibling HUD claim. `smallest_reachable_p_at_this_n` states whether "
                    "the design could have cleared 0.05 at all."
                ),
            }
        )

        verdicts[arm] = {
            "lever": meta["lever"],
            "direction": meta["direction"],
            "fire_counter_used": fk,
            "lever_on_in_arm": on_arm,
            "per_seed": per_seed,
            "overall_verdict": overall,
            # SCOPE, printed beside the verdict so a 3-game / 1-seed comparison can never be read
            # as a corpus result just because it sits in a column headed with the condition name.
            "scope": {
                "n_seeds_compared": len([s for s in seeds if str(s) in per_seed]),
                "n_games_matched_min_across_seeds": (
                    min(int(p["n_games_measured"]) for p in per_seed.values()) if per_seed else 0
                ),
                "n_games_matched_max_across_seeds": (
                    max(int(p["n_games_measured"]) for p in per_seed.values()) if per_seed else 0
                ),
                "n_movable_games_union_over_seeds": len(
                    {g for p in per_seed.values() for g in (p["witness_movable_games"] or [])}
                ),
                "note": (
                    "Read these numbers before the verdict string. A comparison over a handful of "
                    "matched games on one seed is a spot check, not a corpus measurement, and "
                    "`n_movable_games_union_over_seeds` is the support the game-unit sign test "
                    "actually has -- not the number of games run."
                ),
            },
            "game_unit_sign_test": sign,
            # The verdict a reader should quote. The seed verdict says the difference is
            # ATTRIBUTABLE; this says whether its DIRECTION is established on the unit that
            # generalises. Both are needed: attributable-but-underpowered is the common case here
            # and it is NOT a basis for changing a shipped flag.
            "generalisation_verdict": (
                "UNINTERPRETABLE_EMPTY_PASS_REGION"
                if overall == "UNINTERPRETABLE_EMPTY_PASS_REGION"
                else (
                    "DIRECTION_ESTABLISHED_ON_GAME_UNIT"
                    if sign["clears_p_0_05"]
                    else (
                        "UNDERPOWERED_BY_DESIGN_NO_OUTCOME_COULD_CLEAR_P05"
                        if sign["underpowered_support"]
                        else "DIRECTION_NOT_ESTABLISHED_ON_GAME_UNIT"
                    )
                )
            ),
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
            "lever2_fired_cells": sum(1 for r in rs if recomputed_lever2_fired(r)),
            "lever2_games_mask_differs": sorted(
                {r["game"] for r in rs if recomputed_lever2_fired(r)}
            ),
            # The harness's own stamp, kept beside the recomputed value so the correction is
            # visible rather than silent. A non-empty disagreement list means the row files
            # predate the 2026-07-26 fire-counter repair.
            "lever2_fired_cells_per_harness_stamp": sum(1 for r in rs if r.get("lever2_fired")),
            "lever2_games_where_recomputed_disagrees_with_harness_stamp": sorted(
                {r["game"] for r in rs if recomputed_lever2_fired(r) != bool(r.get("lever2_fired"))}
            ),
            # WHICH PREDICATE VERSION STAMPED THE ROW. Absent (`unstamped_pre_2026_07_26`) means the
            # row was written by the pre-fix harness, so its `lever2_fired: False` is uninformative
            # -- it cannot be distinguished from "the broken predicate could not see it fire". This
            # is what makes the disagreement list above interpretable instead of alarming.
            "lever2_fired_predicate_versions": dict(
                collections.Counter(
                    r.get("lever2_fired_predicate") or "unstamped_pre_2026_07_26" for r in rs
                )
            ),
            # HOW THE HUD DIAGNOSTICS REACHED THIS ANALYSER. `nested_only` counts rows that needed
            # the flat back-fill (i.e. rows on which a flat reader saw None); `absent` counts rows
            # carrying NO lever-2 evidence in either direction, which must never be read as a
            # non-fire. A silent back-fill would be an unwitnessed transformation between the
            # measurement and the claim, so the census states it per arm.
            "hud_row_schema": dict(
                collections.Counter(r.get("_hud_row_schema") or "unknown" for r in rs)
            ),
            "hud_diagnostics_unreadable_cells": sorted(
                r["game"]
                for r in rs
                if not (
                    r.get("hud_diagnostics_readable")
                    if "hud_diagnostics_readable" in r
                    else r.get("hud_mask_resolved") is not None
                )
            ),
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


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """sha256 + size + mtime of a file this artifact's numbers DEPEND ON.

    Recorded so `--check-fresh` can answer "was this artifact built by the code and the inputs that
    are on disk right now?" mechanically, instead of a reader comparing mtimes by eye. The incident
    this closes: an artifact committed at 08:53 while its analyser was edited at 10:38 and committed
    without a rebuild. Nothing in the artifact said which analyser had produced it, so the only way
    to find out was to rebuild and diff -- which nobody does before quoting a number.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"path": str(path), "sha256": None, "unreadable": f"{type(exc).__name__}:{exc}"}
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
    }


# The code files whose CONTENT decides what this artifact's numbers mean. `--check-fresh` refuses an
# artifact whose recorded fingerprint for any of these no longer matches the working tree.
def _code_dependencies() -> list[Path]:
    return [
        Path(__file__).resolve(),
        REPO / "scripts" / "arc_scored_path_lever_harness.py",
        REPO / "python" / "carnot" / "agentic" / "arc_hud_row_schema.py",
    ]


def _git_head() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_artifact(
    analysis: dict,
    rows: list[dict],
    sources: list[Path],
    t0: float,
    companion: dict | None = None,
    companion_rows: list[dict] | None = None,
    alt_budget: dict | None = None,
    alt_budget_rows: list[dict] | None = None,
    companion_sources: list[Path] | None = None,
    alt_budget_sources: list[Path] | None = None,
) -> dict:
    # CHECKSUM SCOPE FIX (2026-07-26). This payload used to be {rows, analysis, companion,
    # alt_budget} where `companion` / `alt_budget` are the derived ANALYSIS dicts -- so the 375
    # companion ROWS and the 375 alt-budget ROWS were OUTSIDE the artifact's own integrity hash.
    # 750 of the 805 published rows, 93.2%, were uncovered. Proven rather than inferred: a rebuild
    # that changed `_source` on all 375 companion rows produced a BYTE-IDENTICAL checksum. Every
    # row the artifact publishes is now hashed.
    payload = json.dumps(
        {
            "rows": rows,
            "analysis": analysis,
            "companion": companion or {},
            "companion_rows": companion_rows or [],
            "alt_budget": alt_budget or {},
            "alt_budget_rows": alt_budget_rows or [],
        },
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
            # Any verdict that ASSERTS a difference. The prefix set is explicit rather than a
            # `startswith("EFFECT")` match, which silently stopped matching anything when the
            # verdicts were renamed to ATTRIBUTABLE_WIN_DIFFERENCE on 2026-07-26 -- a gate that
            # quietly becomes vacuous is worse than no gate.
            asserts_difference = str(per.get("seed_verdict", "")).startswith(
                ("EFFECT", "ATTRIBUTABLE_WIN_DIFFERENCE", "WIN_DIFFERENCE")
            )
            if asserts_difference and not per.get("witness_pass_region_nonempty"):
                effect_without_witness.append({"arm": arm, "seed": s})
    gate_witness = not effect_without_witness
    # Every lever that produced a verdict must also carry the GAME-UNIT sign test, so a verdict can
    # never be quoted without the number that says whether its direction generalises.
    levers_without_sign_test = sorted(
        arm for arm, v in lv.items() if not (v.get("game_unit_sign_test") or {})
    )
    gate_sign_test = not levers_without_sign_test

    # ---- FLAG-CHANGE DISPOSITION, DERIVED ---------------------------------------------------
    # Two independent conditions must hold before this measurement can support ANY advice about a
    # shipped flag, and both were violated by the first write-up:
    #   1. the lever's direction must be established on the GAME unit at p<=0.05 (it was p=0.5);
    #   2. more than one BUDGET must have been measured, because lever orderings reverse between
    #      budget 400 and budget 2000 (shipped config: 3-4 wins vs median 12).
    # Anything short of both yields NO_RECOMMENDATION with the reason named.
    all_measured_rows = list(rows) + list(companion_rows or []) + list(alt_budget_rows or [])
    single_budget = len({int(r.get("budget") or 0) for r in all_measured_rows}) <= 1
    flag_recs: dict[str, Any] = {}
    for arm, v in lv.items():
        st = v.get("game_unit_sign_test") or {}
        gv = v.get("generalisation_verdict")
        if gv == "UNINTERPRETABLE_EMPTY_PASS_REGION":
            rec, why = (
                "NO_RECOMMENDATION_UNINTERPRETABLE",
                "the lever's pass region was empty on every seed: no game it moved was won by "
                "either arm, so its contribution was arithmetically forced.",
            )
        elif not st.get("clears_p_0_05"):
            rec, why = (
                "NO_RECOMMENDATION_UNDERPOWERED_ON_GAME_UNIT",
                f"game-unit exact one-sided sign test p={st.get('p_one_sided_exact')} on "
                f"{st.get('n_discordant_games')} discordant game(s) "
                f"(favouring arm: {st.get('movers_favouring_arm')}, favouring control: "
                f"{st.get('movers_favouring_control')}); smallest reachable p at this support is "
                f"{st.get('smallest_reachable_p_at_this_n')}. The same exact test in the CONTROL'S "
                f"favour gives p={st.get('p_one_sided_exact_opposite_direction')}, and the "
                f"discordant games favour "
                f"{st.get('direction_favoured')}. The win-set difference is real and "
                "attributable, but its direction is not established on the unit that generalises.",
            )
        elif single_budget:
            rec, why = (
                "NO_RECOMMENDATION_SINGLE_BUDGET",
                "direction clears p<=0.05 on the game unit, but only ONE budget was measured and "
                "lever orderings reverse with the budget. Reproduce at the other budget first.",
            )
        else:
            rec, why = (
                "EVIDENCE_SUPPORTS_A_FLAG_REVIEW",
                "direction established on the game unit at p<=0.05 and reproduced across more than "
                "one budget. This is evidence for the OPERATOR to review, not a flip.",
            )
        flag_recs[arm] = {
            "lever": v.get("lever"),
            "recommendation": rec,
            "reason": why,
            "generalisation_verdict": gv,
            "p_one_sided_exact_game_unit": st.get("p_one_sided_exact"),
            "smallest_reachable_p_at_this_support": st.get("smallest_reachable_p_at_this_n"),
        }
    gate_no_rec = all(
        rec["recommendation"].startswith("NO_RECOMMENDATION")
        or (
            (lv.get(arm, {}).get("game_unit_sign_test") or {}).get("clears_p_0_05")
            and not single_budget
        )
        for arm, rec in flag_recs.items()
    )

    # ---- BUDGET-DIRECTION RECONCILIATION, COMPUTED ------------------------------------------
    # THE DEFECT THIS EXISTS TO CLOSE. The first write-up recommended un-flipping the shipped
    # frontier trio on a budget-400 result, without measuring the OTHER budget at all -- while the
    # convention-transfer battery had already measured the shipped configuration as the BEST arm at
    # budget 2000. `single_budget` above is the mechanical block on that class of advice, but a
    # block is not a finding: a reader still cannot see WHETHER the two budgets agree.
    #
    # This does. It compares the SAME lever's discordant-game direction at the two budgets, holding
    # the LLM condition FIXED at OFF -- the companion design (budget 400, LLM off) against the
    # alt-budget design (budget 2000, LLM off). That pairing is the only clean budget contrast
    # available: the headline rows are LLM-ON, so comparing them to the alt budget would confound
    # budget with the LLM. AGREES / REVERSES / NOT_COMPARABLE is stamped per lever, and a REVERSAL
    # is by itself sufficient to refuse any single-budget flag advice, independent of significance.
    budget_dir: dict[str, Any] = {}
    if companion and alt_budget:
        lo = companion.get("lever_verdicts") or {}
        hi = alt_budget.get("lever_verdicts") or {}
        for arm in sorted(set(lo) | set(hi)):
            st_lo = (lo.get(arm) or {}).get("game_unit_sign_test") or {}
            st_hi = (hi.get(arm) or {}).get("game_unit_sign_test") or {}
            d_lo, d_hi = st_lo.get("direction_favoured"), st_hi.get("direction_favoured")
            if not d_lo or not d_hi or "no_discordant_game" in (d_lo, d_hi):
                agree = "NOT_COMPARABLE_NO_DISCORDANT_GAME_AT_ONE_BUDGET"
            elif d_lo == "tie" or d_hi == "tie":
                agree = "NOT_COMPARABLE_TIED_SUPPORT_AT_ONE_BUDGET"
            elif d_lo == d_hi:
                agree = "AGREES"
            else:
                agree = "REVERSES"
            budget_dir[arm] = {
                "lever": (lo.get(arm) or hi.get(arm) or {}).get("lever"),
                "budget_agreement": agree,
                "low_budget": {
                    "budget": sorted({int(r.get("budget") or 0) for r in (companion_rows or [])}),
                    "direction_favoured": d_lo,
                    "movers_favouring_arm": st_lo.get("movers_favouring_arm"),
                    "movers_favouring_control": st_lo.get("movers_favouring_control"),
                    "p_one_sided_exact": st_lo.get("p_one_sided_exact"),
                    "p_one_sided_exact_opposite_direction": st_lo.get(
                        "p_one_sided_exact_opposite_direction"
                    ),
                },
                "high_budget": {
                    "budget": sorted({int(r.get("budget") or 0) for r in (alt_budget_rows or [])}),
                    "direction_favoured": d_hi,
                    "movers_favouring_arm": st_hi.get("movers_favouring_arm"),
                    "movers_favouring_control": st_hi.get("movers_favouring_control"),
                    "p_one_sided_exact": st_hi.get("p_one_sided_exact"),
                    "p_one_sided_exact_opposite_direction": st_hi.get(
                        "p_one_sided_exact_opposite_direction"
                    ),
                },
            }
    # A lever whose direction REVERSES between the two budgets cannot support a flag change from
    # either budget alone, whatever its p-value at one of them.
    levers_reversing_with_budget = sorted(
        a for a, v in budget_dir.items() if v.get("budget_agreement") == "REVERSES"
    )

    art: dict[str, Any] = {
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
            "{rows, analysis, companion, companion_rows, alt_budget, alt_budget_rows} payload. "
            "SCOPE CHANGED 2026-07-26: the two ROW lists were previously outside this hash (only "
            "the derived companion/alt_budget ANALYSIS dicts were in it), so 750 of the 805 "
            "published rows -- 93.2% -- were uncovered. A rebuild that mutated all 375 companion "
            "rows produced a byte-identical checksum under the old scope. Any checksum change "
            "dated 2026-07-26 is THIS SCOPE CHANGE, not a measurement change."
        ),
        # ---- STALENESS GUARD (2026-07-26) ---------------------------------------------------
        # THE INCIDENT: this artifact was committed at 08:53 and its analyser was then edited and
        # committed at 10:38 with no rebuild, leaving a ~1h56m window in which the on-disk artifact
        # was not the output of the on-disk analyser. Nothing in the artifact recorded WHICH
        # analyser had produced it, so the only way to find out was to rebuild and diff -- which is
        # exactly what nobody does before quoting a number. These fingerprints make the question
        # mechanical: `analyze_scored_path_lever_ab.py --check-fresh <artifact>` recomputes each
        # one and exits 3 on any mismatch, naming the file and the rebuild command.
        "provenance": {
            "git_head": _git_head(),
            "code": [_file_fingerprint(p) for p in _code_dependencies()],
            # ALL THREE row-source designs, not just `--rows`. `source_row_files` below records only
            # the headline design's paths, so a companion-row file could be swapped with nothing in
            # the artifact noticing -- the same blind spot the checksum scope had.
            "rows_sources": {
                "rows": [_file_fingerprint(p) for p in sources],
                "companion_rows": [_file_fingerprint(p) for p in (companion_sources or [])],
                "alt_budget_rows": [_file_fingerprint(p) for p in (alt_budget_sources or [])],
            },
            "rebuild_command": " ".join(
                [
                    "python scripts/analyze_scored_path_lever_ab.py",
                    "--rows",
                    *[str(p) for p in sources],
                    *(
                        ["--companion-rows", *[str(p) for p in companion_sources]]
                        if companion_sources
                        else []
                    ),
                    *(
                        ["--alt-budget-rows", *[str(p) for p in alt_budget_sources]]
                        if alt_budget_sources
                        else []
                    ),
                    "--out",
                    "<this file>",
                ]
            ),
            "note": (
                "principle: an artifact that cannot say which code produced it cannot be known to "
                "be current, and a stale artifact's numbers are quoted exactly as confidently as a "
                "fresh one's. Check with `--check-fresh <artifact>` (exit 3 = stale, and it names "
                "which dependency drifted)."
            ),
        },
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
        "budget_per_game_all_designs": sorted(
            {int(r.get("budget") or 0) for r in all_measured_rows}
        ),
        "budget_per_game_all_designs_note": (
            "`budget_per_game` is the HEADLINE (scored, LLM-on) rows only. This field is every "
            "budget measured anywhere in this artifact, including the companion and alt-budget "
            "designs -- so a reader can see at a glance whether the single-budget block on flag "
            "advice is in force."
        ),
        "budget_direction_agreement_per_lever": budget_dir,
        "budget_direction_agreement_note": (
            "principle: a lever conclusion is budget-conditional, so 'does the direction survive "
            "the other budget?' is a question with a computed answer, not a caveat. Compares each "
            "lever's discordant-game direction at budget 400 vs budget 2000 with the LLM condition "
            "held FIXED at OFF (the companion vs the alt-budget design) -- the only clean budget "
            "contrast available, since the headline rows are LLM-ON and comparing them would "
            "confound budget with the LLM. REVERSES means the two budgets disagree about which arm "
            "the discordant games favour; that alone refuses any single-budget flag advice for that "
            "lever, independent of its p-value at either budget."
        ),
        "levers_whose_direction_reverses_with_budget": levers_reversing_with_budget,
        "budget_note": (
            "principle: a lever conclusion is only meaningful together with the budget it was "
            "measured at, and this project has already misread its own source once here. 400 is "
            "the SHIPPED agent's self-imposed per-game MAX_ACTIONS loop guard, so a budget-400 row "
            "describes the CURRENT SUBMISSION'S configuration. It is NOT an eval-imposed bound: the "
            "comment above that constant in arc_competition_agent.py says the real bound is the "
            "eval's wall-clock budget (<=12h across all games) and that MAX_ACTIONS is an INTENDED "
            "OVERRIDE POINT (Playback sets it to 1e6). The distinction is load-bearing because "
            "lever orderings REVERSE with it: at budget 2000 the convention-transfer battery "
            "measures the shipped configuration as the BEST of four arms (median 12 wins vs 11 "
            "frontier-only, 9 HUD-only, 7 all-off), while at 400 it wins 3-4 of 25. The sibling "
            "harness experiment_5836 states the mechanism: measured first-win costs span 20 (lp85) "
            "to 1747 (cd82) actions, so a budget below ~2000 structurally cannot see most of the "
            "signal. Therefore NO flag recommendation may rest on one budget alone."
        ),
        "prior_measurements_that_must_be_reconciled_against": [
            {
                "artifact": (
                    "results/outer_loop_cptb_shipped_lever_convention_transfer_20260726.json"
                ),
                "condition": "CarnotAgentPolicy dev twin, LLM off, budget 2000, 5 seeds, C0_real",
                "what_it_measures_that_this_one_does_not": (
                    "The frontier lever's MAIN EFFECT against the all-off control on the game unit: "
                    "5 games to 0, exact one-sided sign test p=0.031 -- the only frontier result in "
                    "this project that clears p<=0.05. It also records the shipped configuration as "
                    "the top arm in every measured condition (median 12/10/5 wins by condition)."
                ),
                "what_it_already_recorded_that_this_run_replicates": (
                    "Its field `games_where_adding_frontier_destroys_a_hud_win` has C0_real value "
                    "['tn36'] -- the SAME game this run finds the frontier trio costing. So tn36 is "
                    "a REPLICATION of a known frontier x HUD antagonism, not new evidence, and the "
                    "correct reading of the combined evidence is a localised interaction (tn36, and "
                    "r11l under salience inversion), NOT that the frontier trio is net harmful."
                ),
                "cross_seed_variance_it_measured": (
                    "measured_determinism_per_arm: the FRONT and SHIP arms are "
                    "measured_deterministic=false with 54 of 75 game-condition cells varying across "
                    "seeds (the uniform-random tier is seeded). A single-seed win delta on those "
                    "arms is therefore not a stable quantity."
                ),
                "differences_that_prevent_a_direct_numeric_comparison": (
                    "Different POLICY (CarnotAgentPolicy vs E3AgentPolicy) and different BUDGET "
                    "(2000 vs 400). Both differences push the same way, so neither artifact can be "
                    "used to overturn the other without holding one of them fixed."
                ),
            }
        ],
        "lever1_frontier_verdict": v1,
        "lever2_hud_verdict": v2,
        "lever3_hazard_verdict": v3,
        # COMPUTED, so a recommendation cannot be written by hand at a bar the evidence does not
        # meet. The first version of this measurement recommended UN-FLIPPING the shipped frontier
        # trio on a 2-game support at p=0.5, one day after the project used the same test to
        # withdraw a sibling claim at p=0.5. This field derives the disposition mechanically from
        # (a) the game-unit sign test and (b) whether more than one budget was measured.
        "flag_change_recommendation_per_lever": flag_recs,
        "flag_change_recommendation_note": (
            "principle: a flag recommendation is a decision, and a decision derived from a "
            "measurement must be derivable FROM the measurement. NO_RECOMMENDATION_* is the default "
            "and the only outcome available when the game-unit sign test does not clear 0.05 or the "
            "result was measured at a single budget -- because lever orderings reverse with the "
            "budget. Nothing here flips anything: flag changes are the operator's call."
        ),
        "acceptance_gate_no_flag_recommendation_without_game_unit_significance": gate_no_rec,
        "acceptance_gate_no_flag_recommendation_without_game_unit_significance_principle": (
            "Guards the defect that produced this artifact's first headline: advising a change to "
            "the shipped submission configuration on a 2-game support at p=0.5. Passes only if "
            "every lever whose recommendation is anything other than NO_RECOMMENDATION_* has a "
            "game-unit sign test clearing p<=0.05 AND was measured at more than one budget."
        ),
        "budgets_measured": sorted({int(r.get("budget") or 0) for r in all_measured_rows}),
        "single_budget_measurement": single_budget,
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
        "acceptance_gate_every_lever_verdict_carries_a_game_unit_sign_test": gate_sign_test,
        "acceptance_gate_every_lever_verdict_carries_a_game_unit_sign_test_principle": (
            "Guards the defect this artifact's first version shipped with: three lever verdicts and "
            "ZERO statistical inference anywhere -- no sign test, no p-value, no statement of what "
            "the design could reach. The GAME is the replication unit (a hidden game is a fresh "
            "draw), so a verdict quoted without its game-unit test invites a 2-game support to be "
            "read as a corpus result. Passes only if every lever verdict carries the test."
        ),
        "acceptance_gate_lever_fire_counters_recomputed_from_raw_diagnostics": True,
        "acceptance_gate_lever_fire_counters_recomputed_from_raw_diagnostics_principle": (
            "Guards the dead/inverted fire-counter class directly. Lever 2's harness stamp was "
            "measured to be ANTI-CORRELATED with the lever (it required the shipped mask to exist, "
            "which is false on exactly the two games the lever moves), so it read 0 in all 430 "
            "cells while the lever fired. The analyser therefore recomputes the flag from each "
            "row's recorded HUD digests and reports every disagreement with the harness stamp in "
            "`fire_census_per_arm.*.lever2_games_where_recomputed_disagrees_with_harness_stamp`."
        ),
        # Assembled by SCANNING every acceptance_gate_* boolean below, not by copying one gate's
        # list. WHY: this field was previously assigned `effect_without_witness`, i.e. it reported
        # the witness gate ALONE -- so an artifact could carry
        # `acceptance_gate_all_llm_on_rows_had_a_live_generator=False` and
        # `acceptance_gate_violations=[]` simultaneously, and any consumer reading the
        # purpose-named field saw a clean run. Filled in after the dict is built.
        "acceptance_gate_violations": [],
        "acceptance_gate_violations_note": (
            "principle: a machine-readable violations list must enumerate EVERY failing gate, or a "
            "downstream aggregation reads a failing run as clean. Assembled by scanning all "
            "acceptance_gate_* booleans in this artifact. `witness_detail` carries the per-arm "
            "detail that this field used to hold alone."
        ),
        "acceptance_gate_violations_witness_detail": effect_without_witness,
        "acceptance_gate_violations_sign_test_detail": levers_without_sign_test,
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
        # THE SECOND BUDGET. The same five arms and all 25 games with induction disabled, run at
        # budget 2000 instead of 400. This section exists because the FATAL defect in this
        # measurement's first version was a budget misreading that inverted its recommendation:
        # lever orderings reverse between the shipped agent's self-imposed 400-action loop guard and
        # a budget large enough for the corpus's measured first-win costs (20 to 1747 actions). It is
        # embedded here, not published separately, so a reader cannot hold one budget's conclusion
        # without seeing the other's.
        "alt_budget_llm_off_design": alt_budget or {},
        "alt_budget_llm_off_design_note": (
            "principle: a lever conclusion is budget-conditional, so a single-budget measurement "
            "cannot support advice about a shipped flag. This section is LLM-OFF at the OTHER "
            "budget. It is not the scored condition either -- it isolates the budget variable while "
            "holding the POLICY fixed at E3AgentPolicy, which is what the convention-transfer "
            "battery (a different policy at budget 2000) could not do."
        ),
        "rows": rows,
        "companion_rows": companion_rows or [],
        "alt_budget_rows": alt_budget_rows or [],
        "flags_flipped": [],
        "flags_flipped_note": (
            "principle: a measurement must not change the thing it measures. NO flag was flipped "
            "by this work; the hazard pruner remains default-OFF and is enabled per-arm by "
            "explicit constructor kwarg only. Arms differ by kwargs, never by mutating module "
            "globals, because all arms share one process."
        ),
    }

    # SCAN every acceptance gate. Any `acceptance_gate_<name>` whose value is exactly False becomes
    # a named violation. Keys ending in `_principle`/`_note`/`_detail` are prose, and
    # `acceptance_gate_violations*` are the outputs themselves, so both are skipped. This is
    # deliberately a scan rather than a hand-maintained list: the previous version had to be
    # remembered when a gate was added, and it was not.
    violations = [
        k
        for k, v in art.items()
        if k.startswith("acceptance_gate_")
        and not k.startswith("acceptance_gate_violations")
        and not k.endswith(("_principle", "_note", "_detail"))
        and v is False
    ]
    art["acceptance_gate_violations"] = violations
    return art


# Where the freshness lint looks to find WHICH artifacts have a separate analyser step and can
# therefore go stale. An INDEX rather than a scan of results/: that directory is 6.1 GB / 6300+ files
# and a cold grep over it in a pre-commit hook would cost minutes for a check that concerns a
# handful of artifacts. Only artifacts written by an analyser that calls
# `register_analyzed_artifact` appear here, and an artifact whose file has since been deleted is
# pruned on the next registration rather than becoming a permanent lint failure.
ANALYZED_ARTIFACT_INDEX = REPO / "ops" / "analyzer_artifact_index.json"


def register_analyzed_artifact(out_path: Path, analyzer: Path | None = None) -> None:
    """Record that THIS artifact is analyser-produced, so the freshness lint knows to check it.

    Registration is idempotent and self-pruning. It is deliberately separate from the artifact's own
    `provenance` block: provenance answers "was this built from the current code" for an artifact
    someone already has in hand; the index answers "which artifacts should anyone be asking that
    about at all", which a commit-time hook needs before it has any artifact in hand.

    `analyzer` MUST be passed by any OTHER analyser reusing this helper. It defaults to this file for
    backwards compatibility, and that default is a trap when the function is imported: the first
    external reuser (`analyze_arc_early_stop_sweep.py`, 2026-07-26) registered its artifact under
    THIS analyser's name, which would have sent a future reader chasing the wrong rebuild command for
    a drifted artifact. The index's whole value is naming the code to re-run.
    """
    try:
        index = json.loads(ANALYZED_ARTIFACT_INDEX.read_text())
        if not isinstance(index, dict):
            index = {}
    except Exception:
        index = {}
    try:
        rel = str(out_path.resolve().relative_to(REPO))
    except ValueError:
        # An artifact written outside the repo (a scratchpad dry-run) is not a tracked deliverable
        # and must not be registered -- the lint would then fail forever on a temp file.
        return
    index[rel] = {
        "analyzer": str((analyzer or Path(__file__)).resolve().relative_to(REPO)),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    index = {k: v for k, v in index.items() if (REPO / k).exists()}
    ANALYZED_ARTIFACT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    ANALYZED_ARTIFACT_INDEX.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")


def check_fresh(artifact_path: Path) -> int:
    """Is this on-disk artifact the output of the code and inputs that are on disk RIGHT NOW?

    Exit codes, and WHY there are four rather than two -- an unknown is not a pass:
      0  FRESH        every recorded dependency's sha256 still matches the working tree.
      3  STALE        at least one dependency's CONTENT changed. This is the incident condition.
      4  UNKNOWN      the artifact predates this guard and records no fingerprints. Reporting that
                      as fresh would be the same false-clean-zero this whole change is about.
      5  UNVERIFIABLE a dependency could not be READ (typically a row-source file that lived in a
                      session scratchpad and has since been cleaned up). Distinguished from STALE
                      because "I cannot check" and "I checked and it drifted" are different facts,
                      and conflating them would train readers to ignore the check.

    Deliberately hash-only: no rows are loaded and no analysis runs, so this is cheap enough to wire
    into `scripts/summarize_artifact.py`, which CLAUDE.md's Reading-Results Discipline already
    mandates as the only legal way to read a result artifact.
    """
    try:
        art = json.loads(artifact_path.read_text())
    except Exception as exc:
        print(f"STALENESS-CHECK: cannot read {artifact_path}: {type(exc).__name__}:{exc}")
        return 4
    prov = art.get("provenance") or {}
    recorded = list(prov.get("code") or [])
    for group in (prov.get("rows_sources") or {}).values():
        recorded.extend(group or [])
    if not recorded:
        print(
            f"STALENESS-CHECK: UNKNOWN -- {artifact_path.name} carries no `provenance` "
            "fingerprints (built before 2026-07-26). Its freshness cannot be established "
            "mechanically; rebuild it to make it checkable."
        )
        return 4
    drift: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for entry in recorded:
        p = Path(entry.get("path", ""))
        now = _file_fingerprint(p)
        if now.get("unreadable"):
            unreadable.append({"path": str(p), "reason": now["unreadable"]})
        elif now.get("sha256") != entry.get("sha256"):
            drift.append(
                {
                    "path": str(p),
                    "recorded_sha256": entry.get("sha256"),
                    "on_disk_sha256": now.get("sha256"),
                }
            )
    cmd = (prov.get("rebuild_command") or "").replace("<this file>", str(artifact_path))
    if drift:
        print(f"STALENESS-CHECK: STALE -- {artifact_path.name} was built against different inputs:")
        for d in drift:
            print(f"  - {d['path']}: content changed since the artifact was built")
        for u in unreadable:
            print(f"  - {u['path']}: ALSO unreadable ({u['reason']})")
        if cmd:
            print(f"  rebuild with: {cmd}")
        return 3
    if unreadable:
        print(
            f"STALENESS-CHECK: UNVERIFIABLE -- {artifact_path.name}: "
            f"{len(recorded) - len(unreadable)} dependencies verified, "
            f"{len(unreadable)} could not be read:"
        )
        for u in unreadable:
            print(f"  - {u['path']}: {u['reason']}")
        print(
            "  This is NOT a staleness finding. It means the check cannot answer the question for "
            "those inputs -- do not read it as a pass."
        )
        return 5
    print(f"STALENESS-CHECK: FRESH -- {artifact_path.name}, {len(recorded)} dependencies verified.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check-fresh",
        metavar="ARTIFACT",
        help="hash-only freshness check of an already-built artifact against the current working "
        "tree. Exits 3 if any recorded code/row-source dependency has changed since it was built, "
        "4 if the artifact predates the guard. Runs no analysis and loads no rows.",
    )
    ap.add_argument("--rows", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--companion-rows", nargs="*", default=[])
    ap.add_argument(
        "--alt-budget-rows",
        nargs="*",
        default=[],
        help="row files measured at the OTHER budget (the same arms, LLM off). Required to lift "
        "the single-budget block on any flag recommendation, because lever orderings reverse "
        "between the shipped agent's 400-action loop guard and a budget large enough for the "
        "corpus's measured first-win costs.",
    )
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
    if a.check_fresh:
        return check_fresh(Path(a.check_fresh))
    if not a.rows or not a.out:
        ap.error("--rows and --out are required unless --check-fresh is given")
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
    alt_budget: dict | None = None
    alt_budget_rows: list[dict] | None = None
    if a.alt_budget_rows:
        # Scored with the SAME analyser and the same control-arm suffix as the companion, so the
        # two budgets are compared by identical machinery. Restoring the primary condition
        # afterwards is load-bearing for the same reason it is for the companion.
        alt_budget_rows = load_rows([Path(x) for x in a.alt_budget_rows])
        set_condition("_" + a.companion_condition)
        alt_budget = analyse(alt_budget_rows)
        set_condition("_" + a.condition)
    art = build_artifact(
        analysis,
        rows,
        sources,
        t0,
        companion,
        companion_rows,
        alt_budget,
        alt_budget_rows,
        companion_sources=[Path(x) for x in a.companion_rows],
        alt_budget_sources=[Path(x) for x in a.alt_budget_rows],
    )
    out_path = Path(a.out)
    out_path.write_text(json.dumps(art, indent=1, default=str))
    register_analyzed_artifact(out_path)
    print(json.dumps({k: v for k, v in analysis.items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
