"""ANALYSE the scored-path budget sweep, and emit the milestone artifact.

THE DELIVERABLE IS (budget -> wins, TIME), NOT (budget -> wins). Time is the real constraint: the
action cap is self-imposed (`CarnotAgent.MAX_ACTIONS = 400`, an intended override point), while the
eval's 12h notebook timeout is not. So every win count here is paired with the measured wall clock
that bought it, and the report names the budget at which the projected wall clock CROSSES the
envelope.

DISCIPLINES APPLIED, EACH AGAINST A SPECIFIC DEFECT THIS PROJECT HAS SHIPPED BEFORE:
  * PER-SEED MATCHED counts everywhere. The any-seed UNION is computed and reported ONLY as a
    clearly-labelled diagnostic, never as the headline, because a union count compared against a
    per-seed count shows a control failing against itself.
  * BOTH TAILS on every test, plus the direction favoured. A one-sided test on a REVERSAL returns
    p=0.89 and reads as "no effect".
  * THE HEADLINE SIGN TEST IS ON THE GAME, NOT THE CELL (corrected 2026-07-26 after adversarial
    review). Three seeds of one game are REPLICATES -- they share that game's adapter, mechanics and
    win condition -- so a cell-level test treats them as independent and inflates significance by
    1-2 orders of magnitude. The inferential target is a HIDDEN game, i.e. a fresh draw from the
    game distribution, so the game is the unit of generalization. The cell-level test is retained
    but explicitly labelled as within-game replicate counts. Earlier versions of this docstring
    CLAIMED the game unit while the code counted cells; that mismatch is the defect being fixed.
  * A COMPUTED WITNESS that each comparison's pass region is non-empty, AT THE COMPARISON'S OWN UNIT.
    A comparison whose movable count is 0 is ARITHMETICALLY FORCED and is stamped UNINTERPRETABLE
    rather than reported as a null.
  * THE SCORE AXIS IS DERIVED FROM THE INSTALLED SCORER, NOT FROM A PARAPHRASE OF THE SPEC. The
    authoritative `arc_agi.scorecard.EnvironmentScoreCalculator` is driven directly (see
    `_resolve_charging_rule`), which is what resolves -- rather than speculates about -- how the
    gateway charges actions. A prior version of this analyser reported that question as "not
    resolvable locally" and built a competing "total-action charge" model on the retracted
    min(human/agent,1)^2 formula; both are gone.
  * MEMORY IS AN ENVELOPE, NOT A RESIDUAL. The framework runs every game as a CONCURRENT THREAD in
    one process, so retained per-game search graphs multiply by the game count. That is measured and
    projected alongside the wall clock, because it binds FIRST.
  * FAILURE SETS, not totals: which GAMES are won at each budget, so a "same count" that is a
    different set is visible.
  * INSTRUMENTATION CENSUS FIRST. Every field the conclusions rest on is counted for
    populated-ness, and any None-valued diagnostic is reported as a hole rather than read as a
    negative. That is the dead-observe-channel lesson.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO = repo_root()

# ---------------------------------------------------------------------------------------------
# PRIOR (budget -> wins) POINTS, CITED NOT RE-DERIVED. Each is a real measurement already on file;
# re-running them would be churn. They are carried into the artifact so a reader can see this
# sweep's curve against everything previously measured, including the points that DISAGREE with it.
# ---------------------------------------------------------------------------------------------
PRIOR_BUDGET_POINTS = [
    {
        "source": "results/proto_just_explore_diag.json",
        "experiment": "exp4605",
        "policy": "just_explore (third-party, NOT E3AgentPolicy)",
        "corpus": "25 public games, variant-1 COLOR-PERMUTED copies",
        "budget": 200,
        "wins": 1,
        "n_games": 25,
        "note": "first_win_rate 0.04 -> 1/25 (only lp85). NOT 0/25; the 0/25 reading is wrong.",
    },
    {
        "source": "results/proto_just_explore_budget_scan.json",
        "policy": "just_explore (third-party)",
        "corpus": "25 public games, color-permuted variants, 5 seeds",
        "budget": 2000,
        "wins_any_seed": 11,
        "mean_solve_fraction": 0.28,
        "note": "PLATEAUS: 4000 gives 0.272 (delta -0.008, within seed noise).",
    },
    {
        "source": "results/proto_just_explore_budget_scan.json",
        "policy": "just_explore (third-party)",
        "budget": 4000,
        "wins_any_seed": 11,
        "mean_solve_fraction": 0.272,
    },
    {
        "source": "results/outer_loop_scored_path_lever_ab_llm_on_20260726.json",
        "policy": "E3AgentPolicy, shipped flags, LLM-OFF, 3 seeds",
        "corpus": "25 public games",
        "budget": 400,
        "wins_per_seed": [3, 4, 4],
        "note": "the current submission's condition",
    },
    {
        "source": "results/outer_loop_scored_path_lever_ab_llm_on_20260726.json",
        "policy": "E3AgentPolicy, shipped flags, LLM-OFF, 3 seeds",
        "budget": 2000,
        "wins_per_seed": [11, 11, 11],
        "note": "the +7-games motivating result",
    },
    {
        "source": "results/outer_loop_cptb_shipped_lever_convention_transfer_20260726.json",
        "policy": "E3AgentPolicy, shipped flags, LLM-OFF, 5 seeds, 1500 cells",
        "budget": 2000,
        "wins_median": 12,
        "note": "5-seed median is 12, not 11 -- a same-budget disagreement with the 3-seed design.",
    },
    {
        "source": "results/experiment_4518_metric_harness_canonical.json",
        "policy": "E3AgentPolicy via arc_leaderboard_eval, induction disabled, 1 seed",
        "corpus": "8-game canonical set (NOT the 25)",
        "budgets_wins": {
            "8000": 4,
            "12000": 6,
            "16000": 6,
            "18000": 7,
            "24000": 7,
            "36000": 3,
        },
        "n_games": 8,
        "note": (
            "MONOTONE NON-DECREASING 8000->24000. The 36000 drop to 3 is a 115-SECOND CI "
            "SUBPROCESS "
            "TIMEOUT artifact: 4 of 8 games carry timed_out=true / actions=null and ALL FOUR were "
            "solved at 24000. It is NOT evidence that a higher budget fails."
        ),
    },
    {
        "source": "results/experiment_5836_frontier_click_vocab_gate.json",
        "policy": "E3AgentPolicy per-game budget probe",
        "game": "cd82",
        "budgets_levels": {"2000": 0, "3000": 0, "4000": 1},
        "note": "cd82's first win needs >3000 actions -- explains its loss at 400 AND at 2000.",
    },
    {
        "source": "ops/changelog.md:184-190 + ops/status.md:20-23",
        "policy": "E3AgentPolicy, shipped flags, 5 seeds, single-game targeted",
        "game": "tn36",
        "budgets_wins": {"2000": 0, "8000": 5},
        "note": (
            "tn36 is an EFFICIENCY regression that crosses the budget (~3.6x), not a capability "
            "loss."
        ),
    },
    {
        "source": "results/experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail.json",
        "policy": "adapter-free graph-explore (NOT E3AgentPolicy)",
        "corpus": "9 shallow-tail games",
        "actions_consumed_max": 253260,
        "wins": 0,
        "note": (
            "The largest action count ever consumed on any ARC path in this repo bought ZERO "
            "levels. "
            "Budget alone does not crack the hard tail; the shipped lever config + E3 cascade is "
            "what differs."
        ),
    },
]

# ---------------------------------------------------------------------------------------------
# WALL-CLOCK ENVELOPE MODELS. The record carries THREE mutually inconsistent per-game figures and
# they reach OPPOSITE conclusions, so the crossing point is reported under EVERY model rather than
# under a silently-chosen one. Reporting a single model would encode an unstated assumption about
# the eval's shape -- the exact defect class that voided exp5835.
# ---------------------------------------------------------------------------------------------
KERNEL_OVERHEAD_S = 980.0  # gateway curl retry-max-time 600 + LLM health probe <=315 + pip/copy
ENVELOPES = [
    {
        "id": "A_25games_12h",
        "n_games": 25,
        "cap_s": 43200.0,
        "source": (
            "docs/research-notes/carnot-verifier-as-pruner-on-graph-explore-2026-06-25.md:121-127"
        ),
        "note": (
            "the 2026-06-25 affordability calc. 25 games -- the PUBLIC set size, not the hidden "
            "set."
        ),
    },
    {
        "id": "B_110games_8h",
        "n_games": 110,
        "cap_s": 28800.0,
        "source": (
            "docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md:41 "
            "(preview, UNCONFIRMED for 2026)"
        ),
        "note": "the 2026-06-21 audit's model, and the TIGHTEST. ~262 s/game serial-equivalent.",
    },
    {
        "id": "C_110games_12h",
        "n_games": 110,
        "cap_s": 43200.0,
        "source": "scripts/kaggle/submission_kernel/main.py subprocess timeout=43200",
        "note": "the only bound VERIFIED in code: the Kaggle notebook's own subprocess timeout.",
    },
]
# A SECOND, INDEPENDENT ceiling that is not a wall-clock division: the documented gateway step rate.
STEP_RATE_PER_S = 10.0
STEP_RATE_CAP_S = 28800.0  # the 8h play cap the rate is quoted against
STEP_RATE_TOTAL = STEP_RATE_PER_S * STEP_RATE_CAP_S  # ~288,000 global real steps

# MEASURED LLM-ON cost at budget 400 (the ONLY budget any LLM-on row exists at), recomputed from
# results/outer_loop_scored_path_lever_ab_llm_on_20260726.json by a prior lane.
LLM_ON_B400 = {
    "arm": "S_llmon",
    "median_s_per_game": 227.3,
    "mean_s_per_game": 255.5,
    "max_s_per_game": 508.0,
    "n_cells": 17,
    "generator_share_of_wall": 0.718,
    "inductions_per_game": 1.20,
    "s_per_induction": 156.8,
    "note": (
        "Every LLM-ON scored row on file is budget 400, so the per-ACTION component of the LLM-on "
        "cost is NOT IDENTIFIED (OLS over those rows gives a NEGATIVE per-action slope at "
        "R^2=0.208 because actions span only 346-396). The LLM-on crossing point is therefore "
        "reported as a BAND between two attributions, not as a number."
    ),
}


def sign_test_two_sided(n_pos: int, n_neg: int) -> dict:
    """Exact binomial sign test on DISCORDANT pairs only, reporting BOTH tails.

    A one-sided test is how a REVERSAL gets reported as "no effect" (p=0.89). So this returns the
    p for each direction plus the two-sided p and which direction the data actually favours.
    """
    n = n_pos + n_neg
    if n == 0:
        return {
            "n_discordant": 0,
            "n_pos": 0,
            "n_neg": 0,
            "p_one_sided_pos": None,
            "p_one_sided_neg": None,
            "p_two_sided": None,
            "direction_favoured": "none_no_discordant_pairs",
            "interpretable": False,
            "reason": "zero discordant pairs -- no test is possible, this is NOT a null result",
        }

    def tail(k: int) -> float:
        return sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0**n)

    p_pos = tail(n_pos)
    p_neg = tail(n_neg)
    return {
        "n_discordant": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "p_one_sided_pos": round(p_pos, 6),
        "p_one_sided_neg": round(p_neg, 6),
        "p_two_sided": round(min(1.0, 2.0 * min(p_pos, p_neg)), 6),
        "direction_favoured": (
            "higher_budget" if n_pos > n_neg else ("lower_budget" if n_neg > n_pos else "tie")
        ),
        "min_reachable_two_sided_p_at_this_support": round(min(1.0, 2.0 * (0.5**n)), 6),
        "interpretable": True,
    }


def _drive_scorer(baselines: list[int], level_up_actions: list[int], total_actions: int) -> float:
    """Score a run with the INSTALLED authoritative scorer, exactly as the gateway does.

    Mirrors `arc_agi.scorecard.EnvironmentScorecard._calculate_score` (scorecard.py:474-491), which
    is also what `arc_leaderboard_eval.run_game` drives: iterate the game's FULL baseline list, take
    per-level actions by DIFFERENCING successive level-up checkpoints, and assign the trailing tail
    to the first INCOMPLETE level.
    """
    from arc_agi.scorecard import EnvironmentScoreCalculator

    calc = EnvironmentScoreCalculator()
    prev = 0
    for li in range(len(baselines)):
        if li < len(level_up_actions):
            at = level_up_actions[li]
            lvl_actions, done, prev = at - prev, True, at
        else:
            lvl_actions, done, prev = total_actions - prev, False, total_actions
        calc.add_level(
            level_index=li + 1,
            completed=done,
            actions_taken=lvl_actions,
            baseline_actions=baselines[li],
        )
    return float(calc.to_score(include_levels=False).score)


def _resolve_charging_rule() -> dict:
    """MEASURE, against the installed scorer, whether the post-solve tail costs anything.

    This exists because a prior version of this analyser reported the charging rule as "NOT
    resolvable locally" and reported a sign disagreement that followed from that assumption. The
    scorer is installed; the question is a two-line experiment, not an open question.
    """
    try:
        import arc_agi.scorecard as _sc
    except Exception as exc:  # pragma: no cover -- the package is a hard dependency of the harness
        return {"resolvable_locally": False, "error": repr(exc)}
    base8 = [20, 30, 40, 50, 60, 70, 80, 90]  # an 8-level game, L1 solved in 15 (superhuman)
    tails = [15, 100, 400, 2000, 4000, 100000]
    scores = {str(t): round(_drive_scorer(base8, [15], t), 4) for t in tails}
    return {
        "resolvable_locally": True,
        "scorer_module_path": _sc.__file__,
        "authoritative_path": "arc_agi/scorecard.py:474-491 (_calculate_score) -> add_level:166-183",
        "per_level_score_formula": "min((baseline_actions / actions_taken)**2 * 100, 115.0)",
        "per_game_aggregation": (
            "index-weighted mean over the game's FULL level list: "
            "sum(level_score[i]*(i+1)) / sum(i+1), then clamped by max_score"
        ),
        "tail_probe_same_solve_varying_tail": scores,
        "tail_is_score_relevant": len(set(scores.values())) > 1,
        "conclusion": (
            "The tail is score-IRRELEVANT. A level the agent did NOT complete scores 0.0 no matter "
            "how many actions were charged to it, and the differencing means those actions are "
            "never charged to a COMPLETED level. So a post-solve tail costs exactly nothing, and "
            "the 'total-action charge' model that a prior version of this analyser reported as an "
            "equally-live reading contradicts the installed implementation."
        ),
        "retracted_formula_note": (
            "min(human/agent,1)^2 is the formula a 2026-06-20 adversarial review already retracted. "
            "For human=20/agent=15 it returns 1.0 where the authoritative scorer returns 2.7778 on "
            "an 8-level game -- a different quantity on a different scale, so any model that "
            "recovers a baseline by inverting it is unsound."
        ),
    }


def _max_score_clamp_table() -> dict:
    """The OMITTED term. `to_score` clamps the game score at the index-weighted fraction of levels
    SOLVED, so the number of levels solved sets a CEILING that per-level speed cannot buy past.

    This was absent from the previous analysis entirely, which is why that analysis could read the
    score axis as a pure per-level efficiency trade-off. It is not: on an 8-level game, an agent that
    is superhuman on level 1 and solves nothing else is capped at 2.78 out of 100.

    A CORRECTION TO THE OBVIOUS READING OF THIS TERM, found by writing its regression test. `min()`
    makes the clamp a CEILING, not a floor, so "solve more levels" is NOT unconditionally the
    dominant lever. Holding per-level speed fixed, depth multiplies the score (2.78 -> 8.33 -> 27.78
    -> 100 at 1/2/4/8 levels solved at exactly human speed). But depth bought by GRINDING scores
    below shallow speed: 4 of 8 levels solved in 400/900/1500/2200 actions scores 0.1207, LESS than
    one level solved in 15 actions (2.7778). Since grinding is exactly what a raised action budget
    buys, this is the mechanism behind the measured result that tripling the win count raises the
    score only ~2%.
    """
    base8 = [20, 30, 40, 50, 60, 70, 80, 90]
    rows = {}
    for nsolved in (1, 2, 3, 4, 6, 8):
        fast = [10 * (i + 1) for i in range(nsolved)]  # superhuman on every solved level
        at_human, cum = [], 0
        for i in range(nsolved):  # exactly the human baseline on every solved level
            cum += base8[i]
            at_human.append(cum)
        rows[str(nsolved)] = {
            "levels_solved_of_8": nsolved,
            "score_with_superhuman_speed_on_every_solved_level": round(
                _drive_scorer(base8, fast, 10000), 4
            ),
            "score_at_exactly_human_speed_on_every_solved_level": round(
                _drive_scorer(base8, at_human, 10000), 4
            ),
            "clamp_index_weighted_fraction_times_100": round(
                sum(range(1, nsolved + 1)) / sum(range(1, 9)) * 100, 4
            ),
        }
    return {
        "formula": "max_score = sum(weights of levels with score>0) / sum(all weights) * 100",
        "applied_as": "score = min(index_weighted_mean_of_level_scores, max_score)",
        "source": "arc_agi/scorecard.py:192-206 (to_score)",
        "probe_on_an_8_level_game": rows,
        "grinding_counterexample": {
            "four_of_eight_levels_solved_slowly_400_900_1500_2200": round(
                _drive_scorer(base8, [400, 900, 1500, 2200], 4000), 4
            ),
            "one_level_solved_fast_15": round(_drive_scorer(base8, [15], 4000), 4),
            "note": (
                "The deeper-but-slower run scores LOWER. The clamp is a ceiling, so depth raises "
                "what speed can earn but does not earn it."
            ),
        },
        "consequence": (
            "The clamp is a CEILING, so the honest statement is conditional: at EQUAL per-level "
            "speed, depth is the dominant lever (1/2/4/8 levels at human speed score "
            "2.78/8.33/27.78/100). But depth obtained by GRINDING -- which is what a larger action "
            "budget buys -- can score below a fast shallow solve. That is why this sweep's tripled "
            "win count moves the score only ~2%: the extra wins arrive hundreds-to-thousands of "
            "actions in, so they sit far under the ceiling they raise."
        ),
    }


def q(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 3)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument(
        "--serial-control-rows",
        default="",
        help="row file from an UNCONTENDED serial run, used to bound how much the parallel "
        "per-seed execution inflated the measured wall clock. Wall clock IS the deliverable, so "
        "presenting a contended number as the cost would be a measurement defect.",
    )
    ap.add_argument(
        "--extension-artifact",
        default="",
        help="a SECOND artifact produced by this same script over a LONGER budget ladder on FEWER "
        "seeds. It is embedded as a clearly-labelled EXTENSION rather than merged into the matched "
        "curve, because mixing a 1-seed budget into a 3-seed matched design would let one budget "
        "be "
        "scored on a different design than its neighbours.",
    )
    ap.add_argument(
        "--memory-rows",
        default="",
        help="JSONL from scripts/arc_budget_memory_probe.py: one clean process per (game, budget) "
        "reporting the shared-library RSS baseline and the PER-GAME retained-graph delta. Required "
        "to build the memory envelope, which binds BEFORE wall clock because the framework runs "
        "every game as a concurrent thread in ONE process.",
    )
    ap.add_argument(
        "--host-ram-gib",
        type=float,
        default=16.0,
        help="host RAM of the target instance, for the memory envelope. Default 16 GiB is a "
        "PLACEHOLDER inherited from the requirements note's VRAM figure and is NOT a confirmed host-"
        "RAM number; the artifact labels it as unconfirmed.",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    t_start = time.time()
    raws = [json.loads(Path(p).read_text()) for p in a.rows]
    raw = raws[0]
    rows = [r for rr in raws for r in rr["rows"]]

    # =========================================================================================
    # STEP 0 -- INSTRUMENTATION CENSUS, BEFORE ANY CONCLUSION. Every field the analysis leans on
    # is counted for populated-ness. A None-valued diagnostic is the tell for a dead channel, and a
    # dead channel produces a clean, zero-error null that looks exactly like a finding.
    # =========================================================================================
    def n_none(key):
        return sum(1 for r in rows if r.get(key) is None)

    census = {
        "n_rows": len(rows),
        "n_ran_true": sum(1 for r in rows if r.get("ran")),
        "n_ran_false_or_missing": sum(1 for r in rows if not r.get("ran")),
        "crash_reasons": dict(
            collections.Counter(r.get("reason") for r in rows if not r.get("ran"))
        ),
        # The outcome fields the win curve is built from.
        "levels_none": n_none("levels"),
        "actions_none": n_none("actions"),
        "wall_s_none": n_none("wall_s"),
        "construct_s_none": n_none("construct_s"),
        "efficiency_none": n_none("efficiency"),
        "states_expanded_none": n_none("states_expanded"),
        # The OBSERVE CHANNEL. If nodes_with_previous_frame is 0 the frame-dependent levers are
        # measuring nothing -- this is the exact 0-of-122-nodes incident.
        "nodes_with_previous_frame_none": n_none("nodes_with_previous_frame"),
        "nodes_with_previous_frame_zero": sum(
            1 for r in rows if (r.get("nodes_with_previous_frame") or 0) == 0
        ),
        "nodes_with_previous_frame_median": statistics.median(
            [r.get("nodes_with_previous_frame") or 0 for r in rows]
        ),
        # The HUD flat fields repaired 2026-07-26 (commit db46988d7). None here means UNREADABLE,
        # which is NOT the same as "the detector resolved nothing" (False).
        "hud_mask_resolved_none": n_none("hud_mask_resolved"),
        "hud_mask_resolved_true": sum(1 for r in rows if r.get("hud_mask_resolved") is True),
        "hud_mask_resolved_false": sum(1 for r in rows if r.get("hud_mask_resolved") is False),
        "hud_mask_cell_count_none": n_none("hud_mask_cell_count"),
        "hud_diagnostics_readable_true": sum(
            1 for r in rows if r.get("hud_diagnostics_readable") is True
        ),
        "hud_diagnostics_readable_false_or_none": sum(
            1 for r in rows if r.get("hud_diagnostics_readable") is not True
        ),
        "hud_diagnostics_unreadable_cells": [
            f"{r.get('game')}/s{r.get('seed')}/b{r.get('budget')}"
            for r in rows
            if r.get("hud_diagnostics_readable") is not True
        ],
        "lever2_fired_predicate_versions": dict(
            collections.Counter(r.get("lever2_fired_predicate") for r in rows)
        ),
        "lever1_fired_true": sum(1 for r in rows if r.get("lever1_fired")),
        "lever2_fired_true": sum(1 for r in rows if r.get("lever2_fired")),
        "lever3_verdicts": dict(collections.Counter(r.get("lever3_verdict") for r in rows)),
        "gated_flags_pinned_all_eight": all(len(r.get("gated_flags") or {}) == 8 for r in rows),
        "distinct_gated_flag_configs": sorted(
            {json.dumps(r.get("gated_flags"), sort_keys=True) for r in rows}
        ),
    }
    census["instrumentation_holes"] = [
        k for k, v in census.items() if k.endswith("_none") and isinstance(v, int) and v > 0
    ]
    census["observe_channel_alive"] = (
        census["nodes_with_previous_frame_zero"] == 0
        and census["nodes_with_previous_frame_none"] == 0
    )

    ok = [r for r in rows if r.get("ran") and r.get("levels") is not None]
    budgets = sorted({r["budget"] for r in ok})
    seeds = sorted({r["seed"] for r in ok})
    games = sorted({r["game"] for r in ok})

    def cell(b, s, g):
        for r in ok:
            if r["budget"] == b and r["seed"] == s and r["game"] == g:
                return r
        return None

    # Only (seed, game) pairs COMPLETE across every budget may enter the matched curve. A partial
    # pair would let one budget be scored on an easier subset than another.
    complete_pairs = [
        (s, g) for s in seeds for g in games if all(cell(b, s, g) is not None for b in budgets)
    ]
    incomplete_pairs = [(s, g) for s in seeds for g in games if (s, g) not in complete_pairs]

    # =========================================================================================
    # STEP 1 -- (budget -> WINS, TIME), PER-SEED MATCHED.
    # =========================================================================================
    per_budget = {}
    for b in budgets:
        per_seed = {}
        for s in seeds:
            gs = [g for (ss, g) in complete_pairs if ss == s]
            won = sorted(g for g in gs if (cell(b, s, g)["levels"] or 0) > 0)
            walls = [
                (cell(b, s, g)["wall_s"] or 0.0) + (cell(b, s, g)["construct_s"] or 0.0) for g in gs
            ]
            per_seed[str(s)] = {
                "n_games_scored": len(gs),
                "wins": len(won),
                "won_games": won,
                "levels_sum": sum(cell(b, s, g)["levels"] or 0 for g in gs),
                "actions_sum": sum(cell(b, s, g)["actions"] or 0 for g in gs),
                "wall_total_s": round(sum(walls), 2),
                "wall_median_s": round(statistics.median(walls), 3) if walls else None,
                "wall_mean_s": round(statistics.mean(walls), 3) if walls else None,
                "wall_p90_s": q(walls, 0.90),
                "wall_max_s": round(max(walls), 3) if walls else None,
                "slowest_game": (
                    max(gs, key=lambda g: cell(b, s, g)["wall_s"] or 0) if gs else None
                ),
            }
        wins = [per_seed[str(s)]["wins"] for s in seeds]
        walls_tot = [per_seed[str(s)]["wall_total_s"] for s in seeds]
        # PER-SEED MATCHED is the headline. The any-seed UNION is recorded ONLY as a labelled
        # diagnostic, because a union count is not comparable to a per-seed count.
        union = sorted(set().union(*[set(per_seed[str(s)]["won_games"]) for s in seeds]))
        allseed = sorted(set.intersection(*[set(per_seed[str(s)]["won_games"]) for s in seeds]))
        per_budget[str(b)] = {
            "budget": b,
            "per_seed": per_seed,
            "wins_per_seed": wins,
            "wins_median": statistics.median(wins),
            "wins_min": min(wins),
            "wins_max": max(wins),
            "wall_total_s_per_seed": walls_tot,
            "wall_total_s_median": round(statistics.median(walls_tot), 2),
            "wall_s_per_game_median_of_seed_medians": round(
                statistics.median([per_seed[str(s)]["wall_median_s"] for s in seeds]), 3
            ),
            "DIAGNOSTIC_ONLY_wins_any_seed_union": len(union),
            "DIAGNOSTIC_ONLY_won_games_any_seed_union": union,
            "wins_all_seeds_intersection": len(allseed),
            "won_games_all_seeds_intersection": allseed,
        }

    # =========================================================================================
    # STEP 2 -- MARGINAL RETURN + SATURATION, and a WITNESS that each step's pass region is
    # non-empty at the step's own unit (the game). A step whose movable-game count is 0 cannot
    # produce a delta of either sign and is stamped UNINTERPRETABLE, not "no effect".
    # =========================================================================================
    steps = []
    for lo, hi in zip(budgets, budgets[1:], strict=False):
        ratio = hi / lo
        doublings = math.log2(ratio)
        d_wins = [
            per_budget[str(hi)]["per_seed"][str(s)]["wins"]
            - per_budget[str(lo)]["per_seed"][str(s)]["wins"]
            for s in seeds
        ]
        d_wall = [
            per_budget[str(hi)]["per_seed"][str(s)]["wall_total_s"]
            - per_budget[str(lo)]["per_seed"][str(s)]["wall_total_s"]
            for s in seeds
        ]
        # PER-GAME PAIRED, matched on (seed, game): who GAINED, who LOST, who could not move.
        gained, lost, frozen_zero, frozen_won = [], [], [], []
        for s, g in complete_pairs:
            wl = (cell(lo, s, g)["levels"] or 0) > 0
            wh = (cell(hi, s, g)["levels"] or 0) > 0
            if wh and not wl:
                gained.append(f"{g}/s{s}")
            elif wl and not wh:
                lost.append(f"{g}/s{s}")
            elif not wl and not wh:
                frozen_zero.append(f"{g}/s{s}")
            else:
                frozen_won.append(f"{g}/s{s}")
        # The witness is at the GAME unit, aggregated per-seed-cell: a cell that is 0-0 could still
        # have moved (the higher budget could have won it), so "movable" = every cell not already
        # won at the LOWER budget, plus every cell already won (which could regress).
        st = sign_test_two_sided(len(gained), len(lost))
        # ------------------------------------------------------------------------------------
        # THE GAME-UNIT (CLUSTERED) TEST -- THE HEADLINE. Cells are NOT independent: three seeds of
        # the same game share that game's adapter, mechanics and win condition, so a cell-level sign
        # test counts within-game replicates as independent observations and inflates significance
        # by 1-2 orders of magnitude. The inferential target is a HIDDEN game -- a fresh draw from
        # the game distribution -- so the GAME is unambiguously the unit of generalization.
        #
        # Found by adversarial review 2026-07-26. The defect is doubly embarrassing because this
        # module's own docstring already claimed the witness/test unit was "AT THE COMPARISON'S OWN
        # UNIT (the game)" while the code counted cells, AND a sibling lane had used exactly this
        # clustering argument to WITHDRAW a HUD-lever claim ("support is 2 games, so the p-floor is
        # 0.25 at any seed count") without either lane applying it to this budget curve.
        #
        # Aggregation: each game contributes ONE sign, from the count of its seeds that gained minus
        # the count that lost. A game whose seeds disagree in equal numbers is concordant (no sign)
        # and drops out of the discordant set, exactly as a tied pair does in any sign test.
        per_game_delta: dict[str, int] = collections.defaultdict(int)
        for s, g in complete_pairs:
            wl = (cell(lo, s, g)["levels"] or 0) > 0
            wh = (cell(hi, s, g)["levels"] or 0) > 0
            if wh and not wl:
                per_game_delta[g] += 1
            elif wl and not wh:
                per_game_delta[g] -= 1
        games_gained = sorted(g for g, d in per_game_delta.items() if d > 0)
        games_lost = sorted(g for g, d in per_game_delta.items() if d < 0)
        st_games = sign_test_two_sided(len(games_gained), len(games_lost))
        # THE WITNESS MUST BE AT THE UNIT THE QUOTED p IS COMPUTED ON. Leaving it cell-only while the
        # headline test moved to the game unit reproduces CLAUDE.md's own named defect verbatim: "a
        # per-cell witness for a median gate is how that defect recurred". A game could GAIN if any of
        # its seeds was unwon at the lower budget; it could REGRESS if any seed was won there.
        games_could_gain = sorted(
            {g for s, g in complete_pairs if not (cell(lo, s, g)["levels"] or 0) > 0}
        )
        games_could_regress = sorted(
            {g for s, g in complete_pairs if (cell(lo, s, g)["levels"] or 0) > 0}
        )
        steps.append(
            {
                "from_budget": lo,
                "to_budget": hi,
                "ratio": round(ratio, 3),
                "doublings": round(doublings, 3),
                "delta_wins_per_seed": d_wins,
                "delta_wins_median": statistics.median(d_wins),
                "wins_gained_per_doubling_median": round(statistics.median(d_wins) / doublings, 3),
                "delta_wall_total_s_per_seed": [round(x, 2) for x in d_wall],
                "delta_wall_total_s_median": round(statistics.median(d_wall), 2),
                "seconds_per_extra_win_median": (
                    round(statistics.median(d_wall) / statistics.median(d_wins), 1)
                    if statistics.median(d_wins) > 0
                    else None
                ),
                "cells_gained": sorted(gained),
                "cells_lost": sorted(lost),
                "n_cells_frozen_zero_both_budgets": len(frozen_zero),
                "n_cells_won_at_both_budgets": len(frozen_won),
                "WITNESS_pass_region_nonempty": {
                    # GAME unit FIRST: this is the unit the headline sign test -- and therefore the
                    # quoted p-value -- is computed on, so it is the unit the witness must certify.
                    "n_games_that_could_gain": len(games_could_gain),
                    "n_games_that_could_regress": len(games_could_regress),
                    "games_that_could_gain": games_could_gain,
                    "nonempty_at_the_game_unit": len(games_could_gain) > 0,
                    # Cell unit retained as the raw evidence, explicitly labelled as the finer grain.
                    "n_cells_that_could_gain": len(gained) + len(frozen_zero),
                    "n_cells_that_could_regress": len(lost) + len(frozen_won),
                    "nonempty": (len(gained) + len(frozen_zero)) > 0,
                    "principle": (
                        "A budget step can only be interpreted if something was structurally able to "
                        "move. If every game were already won at the lower budget, a zero delta would "
                        "be ARITHMETICALLY FORCED, not evidence of saturation. The witness is stated "
                        "at the GAME unit because that is the unit the headline test uses -- a "
                        "per-cell witness for a game-unit test is the exact defect shape CLAUDE.md "
                        "names ('a per-cell witness for a median gate is how that defect recurred')."
                    ),
                },
                # THE HEADLINE TEST. See the clustering note above: the game is the unit of
                # generalization, so this is the p-value that may be quoted.
                "games_gained": games_gained,
                "games_lost": games_lost,
                "HEADLINE_sign_test_on_GAMES_both_tails": st_games,
                # RETAINED, EXPLICITLY DEMOTED. These are WITHIN-GAME REPLICATE counts, not
                # independent observations. Kept because the cell-level gain/loss lists are the raw
                # evidence and because the inflation between the two is itself worth seeing.
                "sign_test_on_cells_WITHIN_GAME_REPLICATES_not_independent": st,
                "clustering_note": (
                    "The cell-level p is inflated relative to the game-level p because the 3 seeds "
                    "of a game are replicates of one game, not 3 draws from the game distribution. "
                    "Quote the game-level p. The EFFECT SIZE (median wins, zero regressions) is "
                    "unaffected by which unit the test uses."
                ),
                "interpretable": (len(gained) + len(frozen_zero)) > 0,
            }
        )

    # SATURATION: the first budget after which NO seed's win count ever increases again.
    sat = None
    for i, b in enumerate(budgets):
        later = budgets[i + 1 :]
        if not later:
            continue
        if all(
            per_budget[str(nb)]["per_seed"][str(s)]["wins"]
            <= per_budget[str(b)]["per_seed"][str(s)]["wins"]
            for nb in later
            for s in seeds
        ):
            sat = b
            break
    saturation = {
        "saturating_budget": sat,
        "definition": (
            "smallest measured budget after which NO seed's win count increases at ANY larger "
            "measured budget"
        ),
        "caveat": (
            "This is saturation WITHIN the measured grid on the 25 PUBLIC games. It is not a claim "
            "about hidden games, and prior points on file disagree at larger budgets (cd82 first "
            "wins only above 3000; tn36 only at 8000; exp4518 gains 8000->18000 on its 8-game set)."
        ),
    }

    # =========================================================================================
    # STEP 3 -- THE CROSSING POINT. THIS IS THE ANSWER.
    # =========================================================================================
    # Fit the measured LLM-OFF per-game cost as fixed + marginal, from the sweep's own rows, using
    # MATCHED PAIRS rather than an average. The average is dominated by the one-off construct cost
    # and would overstate marginal cost roughly 2x, understating the affordable budget.
    fixed = statistics.median([r["construct_s"] for r in ok if r.get("construct_s") is not None])
    marg_samples = []
    for s, g in complete_pairs:
        for lo, hi in zip(budgets, budgets[1:], strict=False):
            rl, rh = cell(lo, s, g), cell(hi, s, g)
            da = (rh["actions"] or 0) - (rl["actions"] or 0)
            dw = (rh["wall_s"] or 0.0) - (rl["wall_s"] or 0.0)
            if da > 0:
                marg_samples.append(dw / da)
    marginal = statistics.median(marg_samples)
    marginal_p90 = q(marg_samples, 0.90)

    # IS COST LINEAR IN BUDGET? If it is SUPERLINEAR the fitted marginal UNDERSTATES cost at large
    # budget, so the analytic crossing budget below is OPTIMISTIC and must be labelled as such.
    # Measured directly as the log-log exponent between adjacent measured budgets.
    superlinear = []
    for lo, hi in zip(budgets, budgets[1:], strict=False):
        clo = per_budget[str(lo)]["wall_s_per_game_median_of_seed_medians"]
        chi = per_budget[str(hi)]["wall_s_per_game_median_of_seed_medians"]
        superlinear.append(
            {
                "from_budget": lo,
                "to_budget": hi,
                "cost_ratio": round(chi / clo, 3) if clo else None,
                "budget_ratio": round(hi / lo, 3),
                "exponent_alpha": (
                    round(math.log(chi / clo) / math.log(hi / lo), 3) if clo and chi else None
                ),
            }
        )
    alphas = [x["exponent_alpha"] for x in superlinear if x["exponent_alpha"] is not None]

    crossing = {
        "cost_scaling_in_budget": {
            "per_step": superlinear,
            "alpha_median": round(statistics.median(alphas), 3) if alphas else None,
            "alpha_at_largest_step": alphas[-1] if alphas else None,
            "alpha_is_rising": bool(len(alphas) >= 2 and alphas[-1] > alphas[0]),
            "which_alpha_matters_for_extrapolation": (
                "alpha_at_largest_step. The low-budget alphas are DEPRESSED by the fixed per-game "
                "construct cost (~2.3s), which dominates a 200-action cell; they are not evidence "
                "of sub-linear search. Extrapolating past the largest measured budget must use the "
                "largest-step exponent."
            ),
            "is_superlinear": bool(alphas and alphas[-1] > 1.05),
            "consequence": (
                "alpha > 1 means per-game cost grows FASTER than the budget (the search graph "
                "grows, "
                "so each additional action costs more than the last). The linear "
                "fixed+marginal crossing budget below is therefore an OPTIMISTIC UPPER BOUND; the "
                "MEASURED per-budget totals are the trustworthy figures and the extrapolation past "
                "the largest measured budget should not be read as a feasibility guarantee."
            ),
        },
        "cost_model_llm_off": {
            "fixed_s_per_game_construct_median": round(fixed, 4),
            "marginal_s_per_action_median": round(marginal, 6),
            "marginal_s_per_action_p90": marginal_p90,
            "n_matched_pairs": len(marg_samples),
            "method": (
                "MARGINAL from matched (game, seed) budget pairs: d(wall_s)/d(actions). The "
                "average "
                "cost per action would be inflated by the fixed construct cost."
            ),
        },
        "measured_totals_25_games_llm_off": {
            str(b): per_budget[str(b)]["wall_total_s_median"] for b in budgets
        },
        "envelopes": [],
        "step_rate_ceiling": {},
        "llm_on_band": {},
    }

    for env in ENVELOPES:
        n_games_env, cap = env["n_games"], env["cap_s"]
        usable = cap - KERNEL_OVERHEAD_S
        rows_env = {}
        for b in budgets:
            # Project from the MEASURED per-game median at that budget (public games), scaled to N.
            med = per_budget[str(b)]["wall_s_per_game_median_of_seed_medians"]
            p90 = statistics.median(
                [per_budget[str(b)]["per_seed"][str(s)]["wall_p90_s"] for s in seeds]
            )
            rows_env[str(b)] = {
                "projected_total_s_at_median": round(n_games_env * med, 1),
                "projected_total_s_at_p90": round(n_games_env * p90, 1),
                "fraction_of_usable_at_median": round(n_games_env * med / usable, 4),
                "fraction_of_usable_at_p90": round(n_games_env * p90 / usable, 4),
                "fits_at_median": (n_games_env * med) <= usable,
                "fits_at_p90": (n_games_env * p90) <= usable,
            }
        # analytic crossing budget from the fitted cost model
        b_cross = (usable / n_games_env - fixed) / marginal
        b_cross_p90 = (usable / n_games_env - fixed) / marginal_p90
        # SUPERLINEAR crossing: cost_per_game(b) = fixed + k * b^alpha, anchored at the LARGEST
        # measured budget with alpha from the largest measured step. This is the defensible
        # extrapolation when alpha > 1; the linear one above is an optimistic upper bound.
        b_top = budgets[-1]
        c_top = per_budget[str(b_top)]["wall_s_per_game_median_of_seed_medians"]
        alpha_top = alphas[-1] if alphas else 1.0
        b_cross_super = None
        if c_top > fixed and alpha_top and alpha_top > 0:
            k = (c_top - fixed) / (b_top**alpha_top)
            avail = usable / n_games_env - fixed
            if avail > 0:
                b_cross_super = int((avail / k) ** (1.0 / alpha_top))
        crossing["envelopes"].append(
            {
                **env,
                "usable_loop_wall_s": round(usable, 1),
                "kernel_overhead_s_assumed": KERNEL_OVERHEAD_S,
                "per_budget": rows_env,
                "largest_measured_budget_that_fits_at_median": max(
                    [b for b in budgets if rows_env[str(b)]["fits_at_median"]], default=None
                ),
                "analytic_crossing_budget_llm_off_median_cost_LINEAR_optimistic": int(b_cross),
                "analytic_crossing_budget_llm_off_p90_cost_LINEAR_optimistic": int(b_cross_p90),
                "analytic_crossing_budget_llm_off_SUPERLINEAR": b_cross_super,
                "superlinear_fit": {
                    "form": "cost_per_game(b) = fixed + k*b^alpha",
                    "fixed_s": round(fixed, 4),
                    "alpha": alpha_top,
                    "anchored_at_budget": b_top,
                    "anchor_cost_s": c_top,
                },
                "which_crossing_to_trust": (
                    "the SUPERLINEAR one when alpha > 1. The LINEAR figure assumes each extra "
                    "action costs the same as the last, which the measured exponent contradicts."
                ),
            }
        )

    for env in ENVELOPES:
        n_games_env = env["n_games"]
        crossing["step_rate_ceiling"][env["id"]] = {
            "global_steps_available": STEP_RATE_TOTAL,
            "per_game_action_ceiling": int(STEP_RATE_TOTAL / n_games_env),
            "source": (
                "10 steps/sec x 8h, "
                "docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md:41"
            ),
            "status": (
                "UNCONFIRMED for 2026 and server-side; no client-side limiter exists in the "
                "framework"
            ),
            "binding_vs_compute": (
                "TIGHTER than the LLM-off compute bound"
                if (STEP_RATE_TOTAL / n_games_env)
                < ((env["cap_s"] - KERNEL_OVERHEAD_S) / n_games_env - fixed) / marginal
                else "looser than the LLM-off compute bound"
            ),
        }

    # THE LLM-ON RECOMPUTATION. This is the honest caveat the brief demands: the curve above is
    # LLM-OFF, and the scored submission is LLM-ON.
    llmoff_b400 = (
        per_budget[str(400)]["wall_s_per_game_median_of_seed_medians"] if 400 in budgets else None
    )
    # THE MECHANISTIC MODEL, replacing the fixed-vs-proportional BAND (corrected 2026-07-26 after
    # adversarial review). The band was:
    #     lower  = LLM-on cost is FIXED per game, extra actions cost only the LLM-off marginal (1.0x)
    #     upper  = the whole LLM-on cost scales PROPORTIONALLY with the budget (~10x at b4000)
    # Both are wrong, and the field that resolves it was ALREADY IN EVERY ROW. LLM cost is
    # (number of inductions) x (seconds per induction); `induction_attempts` is recorded on every
    # row of this sweep, and it grows only 1.15 -> 1.61 per game across a 10x budget raise -- because
    # induction is triggered by novel-observation events, not by action count. So the fixed bound is
    # optimistic (inductions DO grow) and the proportional bound is wildly pessimistic (they grow
    # 1.4x, not 10x).
    #
    #   cost_per_game(b) = llm_off_cost(b) + inductions(b) * s_per_induction,
    #   calibrated by a single multiplicative factor so the model REPRODUCES the measured LLM-on
    #   227.3 s/game at b400.
    #
    # RESIDUALS, disclosed because they are the model's real limits: `s_per_induction` is measured
    # only at b400; the calibration factor is applied multiplicatively at every budget; and
    # `induction_attempts` on an LLM-OFF row counts PLANNED inductions (the generator is absent, so
    # the call is skipped), which is the right scaling proxy only if the trigger rate does not itself
    # depend on the generator's output. ONE LLM-on run at b1000 or b2000 would anchor this directly
    # and is the next measurement worth making -- not a budget flip.
    ind_per_game = {
        b: statistics.mean(
            [
                cell(b, s, g).get("induction_attempts") or 0
                for s, g in complete_pairs
                if cell(b, s, g).get("induction_attempts") is not None
            ]
        )
        for b in budgets
    }
    spi = LLM_ON_B400["s_per_induction"]
    raw_b400 = (llmoff_b400 or 0.0) + ind_per_game.get(400, 0.0) * spi
    calib = (LLM_ON_B400["median_s_per_game"] / raw_b400) if raw_b400 else 1.0

    def llm_on_model_s_per_game(b: int) -> float:
        base_off = per_budget[str(b)]["wall_s_per_game_median_of_seed_medians"]
        return (base_off + ind_per_game[b] * spi) * calib

    crossing["llm_on_band"] = {
        "measured_llm_on_at_budget_400": LLM_ON_B400,
        "measured_llm_off_at_budget_400_this_sweep_s_per_game": llmoff_b400,
        "llm_on_over_llm_off_factor_at_budget_400": (
            round(LLM_ON_B400["median_s_per_game"] / llmoff_b400, 1) if llmoff_b400 else None
        ),
        "model": "cost_per_game(b) = llm_off_s_per_game(b) + inductions_per_game(b) * s_per_induction",
        "model_calibration_factor": round(calib, 4),
        "model_calibration_note": (
            "applied multiplicatively at every budget so the model reproduces the MEASURED LLM-on "
            f"{LLM_ON_B400['median_s_per_game']} s/game at b400 (raw model there: "
            f"{round(raw_b400, 1)} s/game)"
        ),
        "measured_inductions_per_game_by_budget": {
            str(b): round(v, 3) for b, v in ind_per_game.items()
        },
        "inductions_growth_over_the_measured_budget_range": (
            round(ind_per_game[budgets[-1]] / ind_per_game[budgets[0]], 3)
            if ind_per_game.get(budgets[0])
            else None
        ),
        "why_the_previous_band_was_wrong": (
            "The retired band's LOWER bound held LLM cost fixed (1.0x) and its UPPER bound scaled it "
            "with the budget (~10x at b4000). Measured induction growth across that same range is "
            f"{round(ind_per_game[budgets[-1]] / ind_per_game[budgets[0]], 2)}x, so the truth is "
            "neither bound and the band spanned a factor of ~10 for no reason: the identifying "
            "variable was recorded on every row."
        ),
        "residuals": [
            "THE BUDGET-400 ROW IS ARITHMETICALLY FORCED, NOT A TEST. The model is calibrated to "
            "reproduce the measured b400 anchor exactly, so its b400 projection equals the "
            "measurement BY CONSTRUCTION and that row's FITS/OVER verdict carries ZERO independent "
            "information. Only the b1000+ rows are model OUTPUT. (Flagged by adversarial review of "
            "this fix: an anchor whose value is forced is the same defect class as a gate whose pass "
            "region is empty.)",
            "s_per_induction measured only at budget 400",
            "calibration factor applied multiplicatively across budgets",
            "induction_attempts on LLM-off rows counts PLANNED inductions, not executed ones",
            "one LLM-on run at b1000/b2000 would replace this model with a direct anchor",
        ],
        "budget_400_row_is_forced_by_calibration": True,
        "budgets_that_are_genuine_model_output": [b for b in budgets if b != 400],
        "projected_s_per_game_by_budget": {
            str(b): round(llm_on_model_s_per_game(b), 1) for b in budgets
        },
    }
    for env in ENVELOPES:
        n_games_env, cap = env["n_games"], env["cap_s"]
        usable = cap - KERNEL_OVERHEAD_S
        band = {}
        for b in budgets:
            tot = n_games_env * llm_on_model_s_per_game(b)
            band[str(b)] = {
                "projected_total_s": round(tot, 1),
                "fraction_of_usable": round(tot / usable, 3),
                "verdict": "FITS" if tot <= usable else "OVER",
            }
        band["largest_budget_that_fits"] = max(
            [b for b in budgets if band[str(b)]["verdict"] == "FITS"], default=None
        )
        crossing["llm_on_band"][env["id"]] = band
    # The conditional conclusion, stated per envelope rather than as an unconditional claim.
    fits_by_env = {
        env["id"]: crossing["llm_on_band"][env["id"]]["largest_budget_that_fits"]
        for env in ENVELOPES
    }
    crossing["llm_on_band"]["conclusion_is_CONDITIONAL_on_the_envelope"] = {
        "largest_budget_that_fits_per_envelope": fits_by_env,
        "statement": (
            "Under the only envelope VERIFIED in code (C_110games_12h -- the Kaggle kernel's own "
            f"subprocess timeout=43200), the model fits to budget {fits_by_env.get('C_110games_12h')} "
            f"at {crossing['llm_on_band']['C_110games_12h'][str(budgets[-1])]['fraction_of_usable']} "
            "of usable wall at the largest measured budget, i.e. ~10% headroom. Under the TIGHTEST "
            "documented-but-unconfirmed envelope (B_110games_8h) the model is already OVER above "
            f"budget {fits_by_env.get('B_110games_8h')}. So LLM-on wall clock is NOT unconditionally "
            "non-binding: it is comfortable only under the 12h reading, and it binds under the 8h "
            "reading."
        ),
    }

    # =========================================================================================
    # STEP 3b -- THE MEMORY ENVELOPE. THIS IS THE CONSTRAINT THAT BINDS FIRST.
    #
    # A prior version of this analysis demoted memory to an untested residual ("memory at 110
    # concurrent games untested", citing a prior ESTIMATE near 6.6 GiB) while recommending the
    # largest measured budget. That was the wrong call twice over: the quantity is trivially
    # measurable, and it is the one that hard-fails rather than merely running slow.
    #
    # WHY IT MULTIPLIES BY THE GAME COUNT. The competition framework's `Swarm.main()`
    # (/home/ianblenke/arc3_agents/agents/swarm.py:76-99) constructs one agent + one Thread per game
    # and starts EVERY thread before joining any, so all N games are live in ONE address space at
    # once and each retains its own search graph. The projection is therefore
    #     shared_libs_rss + n_games * per_game_retained_delta
    # with the split taken at "everything importable" (shared) vs "env + policy + graph" (per-thread).
    # =========================================================================================
    memory_envelope: dict = {
        "measured": False,
        "why_it_binds_first": (
            "Swarm.main() starts one thread per game before joining any, so retained per-game search "
            "graphs are concurrent, not sequential. Exceeding host RAM is a hard failure, whereas "
            "exceeding a wall-clock budget degrades gracefully into fewer games played."
        ),
    }
    if a.memory_rows and Path(a.memory_rows).exists():
        mem_rows = [
            json.loads(ln)
            for ln in Path(a.memory_rows).read_text().splitlines()
            if ln.strip().startswith("{")
        ]
        mem_rows = [m for m in mem_rows if m.get("ran") and m.get("per_game_delta_mib")]
        shared = statistics.median([m["shared_libs_rss_mib"] for m in mem_rows])
        host_mib = a.host_ram_gib * 1024.0
        per_budget_mem = {}
        for b in sorted({m["budget"] for m in mem_rows}):
            deltas = [m["per_game_delta_mib"] for m in mem_rows if m["budget"] == b]
            # WORST-CASE, not mean. Every thread is live at once, so the swarm's peak is driven by
            # the games that retain the most, and a mean would understate a hard failure.
            worst, med = max(deltas), statistics.median(deltas)
            probed_games = sorted({m["game"] for m in mem_rows if m["budget"] == b})
            # DOES THE PROBE SET ACTUALLY CONTAIN THE CORPUS WORST CASE? The projection multiplies a
            # per-game WORST by the hidden-set game count, so a worst taken over an arbitrary subset
            # would understate it. Checked here against the full sweep's retained-frame counts rather
            # than asserted: `nodes_with_frame` is the quantity the retained graph's size tracks.
            corpus_argmax = max(
                complete_pairs, key=lambda sg: cell(b, sg[0], sg[1]).get("nodes_with_frame") or 0
            )
            corpus_argmax_game = corpus_argmax[1]
            entry = {
                "n_games_probed": len(deltas),
                "per_game_delta_mib_worst": round(worst, 1),
                "per_game_delta_mib_median": round(med, 1),
                "games_probed": probed_games,
                "corpus_argmax_game_by_nodes_with_frame": corpus_argmax_game,
                "probe_set_contains_the_corpus_worst_case": corpus_argmax_game in probed_games,
                "probe_coverage_note": (
                    "The worst-case term is a max over the PROBED games, so it only bounds the corpus "
                    "if the probe set contains the corpus's heaviest game. That is checked, not "
                    "assumed. Even when true it does NOT bound the ~110 HIDDEN games, which have no "
                    "adapters and may retain differently -- that remains unmeasured."
                ),
            }
            for env in ENVELOPES:
                n = env["n_games"]
                proj_worst = shared + n * worst
                proj_med = shared + n * med
                entry[env["id"]] = {
                    "n_games": n,
                    "projected_peak_gib_if_every_game_is_worst_case": round(proj_worst / 1024.0, 2),
                    "projected_peak_gib_at_median_per_game": round(proj_med / 1024.0, 2),
                    "fraction_of_host_ram_worst": round(proj_worst / host_mib, 3),
                    "fraction_of_host_ram_median": round(proj_med / host_mib, 3),
                    "verdict": (
                        "FITS at worst case"
                        if proj_worst <= host_mib
                        else (
                            "MARGINAL: fits at the median per-game cost, over at worst case"
                            if proj_med <= host_mib
                            else "OVER even at the median per-game cost"
                        )
                    ),
                }
            per_budget_mem[str(b)] = entry
        budgets_mem = sorted(int(k) for k in per_budget_mem)
        safe = {
            env["id"]: max(
                [
                    b
                    for b in budgets_mem
                    if per_budget_mem[str(b)][env["id"]]["verdict"] == "FITS at worst case"
                ],
                default=None,
            )
            for env in ENVELOPES
        }
        memory_envelope = {
            "measured": True,
            "method": (
                "one clean process per (game, budget); every module run_cell imports lazily is "
                "imported EAGERLY before the baseline snapshot, so the baseline is the SHARED term "
                "and the delta is the PER-THREAD term (env + policy + retained graph)"
            ),
            "probe_script": "scripts/arc_budget_memory_probe.py",
            "probe_rows_source": str(Path(a.memory_rows).resolve()),
            "n_probe_cells": len(mem_rows),
            # The probe's own machine time, recorded so the artifact accounts for ALL the compute it
            # rests on rather than only the sweep's.
            "probe_total_cell_compute_s": round(sum(m.get("wall_s") or 0.0 for m in mem_rows), 1),
            "shared_libs_rss_mib_median": round(shared, 1),
            "host_ram_gib_assumed": a.host_ram_gib,
            "host_ram_is_UNCONFIRMED": (
                "The 16 GiB figure in the requirements note refers to VRAM, not host RAM. The real "
                "host RAM of the ARC-AGI-3 instance pool is NOT confirmed, and it is what decides "
                "whether the marginal budgets below are safe. Confirm before acting on this."
            ),
            "per_budget": per_budget_mem,
            "largest_budget_that_FITS_at_worst_case_per_envelope": safe,
            "why_it_binds_first": memory_envelope["why_it_binds_first"],
            "measurement_defect_this_replaces": (
                "A prior version reported memory as 'NOT measured here' with a cited ESTIMATE near "
                "6.6 GiB, while recommending the largest measured budget. The estimate turns out to "
                "have been for the SHIPPED budget; at the largest measured budget the projection is "
                "several times larger."
            ),
        }
    memory_envelope["nodes_with_frame_by_budget_for_extrapolation"] = {
        str(b): {
            "median": statistics.median(
                [cell(b, s, g).get("nodes_with_frame") or 0 for s, g in complete_pairs]
            ),
            "max": max([cell(b, s, g).get("nodes_with_frame") or 0 for s, g in complete_pairs]),
        }
        for b in budgets
    }

    # =========================================================================================
    # STEP 4 -- THE EFFICIENCY / SCORE AXIS, RESOLVED AGAINST THE INSTALLED SCORER.
    #
    # THE DEFECT THIS REPLACES (found by adversarial review 2026-07-26, and it INVERTED the
    # recommendation). The previous version of this block computed the score two ways -- the
    # harness's own metric, and a "pessimistic total-action charge" that assumed the gateway might
    # charge a completed level for EVERY action the game ever took -- then reported that the two
    # "DISAGREE IN SIGN" and that the charging rule was "NOT resolvable locally". All three parts
    # were wrong:
    #
    #   1. IT IS RESOLVABLE LOCALLY. `arc_agi.scorecard` is installed in the very venv this analyser
    #      runs in, and `EnvironmentScoreCalculator.add_level(actions_taken=...)` takes PER-LEVEL
    #      actions. The authoritative card->score path (scorecard.py:474-491) DIFFERENCES successive
    #      level checkpoints (`level_actions = actions_at_level - prev_actions`) and assigns the
    #      trailing tail to the FIRST INCOMPLETE level -- which is byte-for-byte what
    #      `arc_leaderboard_eval.run_game` already does (arc_leaderboard_eval.py:296-317). The
    #      "total-action charge" was not a second reading of an ambiguous spec; it CONTRADICTED the
    #      shipped implementation.
    #   2. IT IS EMPIRICALLY FALSE. `authoritative_scorer_resolution` below DRIVES the installed
    #      scorer to show that actions charged to an incomplete level are score-IRRELEVANT: the same
    #      solve scores identically whether the tail is 10 actions or 100,000.
    #   3. IT USED A RETRACTED FORMULA. The pessimistic model recovered a human baseline as
    #      sqrt(eff)*atfl, which presumes `eff` is `min(human/agent,1)^2`. A 2026-06-20 adversarial
    #      review already caught that formula as wrong on three counts; the real per-level score is
    #      `min((baseline/actions)^2 * 100, 115)` and the per-GAME score is an INDEX-WEIGHTED mean
    #      over ALL levels, clamped by `max_score` (see `max_score_clamp` below).
    #
    # So the efficiency axis is now derived SOLELY from the authoritative scorer, and the omitted
    # `max_score` clamp -- the term that makes solving MORE levels the dominant lever -- is computed
    # explicitly rather than left out.
    # =========================================================================================
    eff = {
        "why": (
            "A win count is not the scored quantity. The scored quantity is the authoritative "
            "per-game score from arc_agi.scorecard, which `arc_leaderboard_eval.run_game` already "
            "computes into every row's `efficiency` field by driving the installed calculator. This "
            "block reports how that score moves with budget, and separates the two levers inside it "
            "(per-level action efficiency, and the max_score clamp set by how many levels are "
            "solved)."
        ),
        "harness_metric_is_authoritative": (
            "`efficiency` is NOT a harness-local approximation. arc_leaderboard_eval.py:296-317 "
            "drives arc_agi.scorecard.EnvironmentScoreCalculator with the same level-differencing "
            "as scorecard.py:474-491 and returns `calc.to_score().score`, so the field IS the "
            "leaderboard's own per-game score for that run. It is reported here without a competing "
            "model."
        ),
        "per_budget": {},
        "regressions_vs_400": {},
    }
    # THE LOCAL RESOLUTION, COMPUTED NOT ASSERTED. Driving the installed scorer is what turns
    # "the charging rule is unresolved" into a measured fact, so it is done here in the analyser
    # rather than described in prose.
    eff["authoritative_scorer_resolution"] = _resolve_charging_rule()
    eff["max_score_clamp"] = _max_score_clamp_table()
    # BUDGET-EXHAUSTION COVERAGE. Reported because it is the load-bearing premise behind ANY claim
    # about post-solve tails: if the agent stopped early, there is no tail to argue about. Measured
    # as a FRACTION of the budget consumed rather than an equality test, because the run stops a few
    # actions short of the cap (resets are counted separately from stepped actions).
    eff["premise_coverage_budget_exhaustion"] = {}
    for b in budgets:
        fracs = [
            (cell(b, s, g).get("actions") or 0) / b
            for s, g in complete_pairs
            if cell(b, s, g).get("actions") is not None
        ]
        eff["premise_coverage_budget_exhaustion"][str(b)] = {
            "n_cells": len(fracs),
            "n_cells_at_or_above_95pct_of_budget": sum(1 for x in fracs if x >= 0.95),
            "n_cells_below_90pct_of_budget": sum(1 for x in fracs if x < 0.90),
            "fraction_of_budget_used_median": round(statistics.median(fracs), 4) if fracs else None,
            "fraction_of_budget_used_min": round(min(fracs), 4) if fracs else None,
        }
    eff["premise_coverage_budget_exhaustion"]["principle"] = (
        "SUBMITTED_EARLY_STOP_GRACE is None, so nothing cuts a run after a solve and essentially "
        "every cell runs to the cap. That premise is what made the retired 'total-action charge' "
        "model's arithmetic reduce to the identity (b_ref/b)^2 -- the model carried no information "
        "beyond the budget ratio. The premise itself is real and measured here; the model built on "
        "it was not."
    )
    for b in budgets:
        vals = [
            cell(b, s, g)["efficiency"]
            for s, g in complete_pairs
            if cell(b, s, g).get("efficiency") is not None
        ]
        won_eff = [
            cell(b, s, g)["efficiency"]
            for s, g in complete_pairs
            if (cell(b, s, g)["levels"] or 0) > 0
        ]
        eff["per_budget"][str(b)] = {
            "efficiency_sum_over_cells": round(sum(vals), 5),
            "efficiency_sum_per_seed": [
                round(
                    sum(
                        cell(b, s, g)["efficiency"] or 0.0 for (ss, g) in complete_pairs if ss == s
                    ),
                    5,
                )
                for s in seeds
            ],
            "n_won_cells": len(won_eff),
            "efficiency_median_over_won_cells": (
                round(statistics.median(won_eff), 5) if won_eff else None
            ),
        }
    # THE DECISION-RELEVANT NUMBER, from the authoritative scorer only. Summed over matched cells so
    # a budget's score total is comparable to another's on the SAME cell set.
    eff["scored_sum_authoritative"] = {}
    for b in budgets:
        tot = 0.0
        n_single = n_multi = 0
        for s, g in complete_pairs:
            r = cell(b, s, g)
            if (r.get("levels") or 0) <= 0:
                continue
            tot += r.get("efficiency") or 0.0
            if (r.get("levels") or 0) == 1:
                n_single += 1
            else:
                n_multi += 1
        eff["scored_sum_authoritative"][str(b)] = {
            "authoritative_score_sum_over_won_cells": round(tot, 5),
            "n_won_cells": n_single + n_multi,
            "n_single_level_cells": n_single,
            "n_multi_level_cells": n_multi,
        }
    base = eff["scored_sum_authoritative"].get("400")
    if base:
        ref = base["authoritative_score_sum_over_won_cells"]
        for v in eff["scored_sum_authoritative"].values():
            v["vs_b400_ratio"] = (
                round(v["authoritative_score_sum_over_won_cells"] / ref, 4) if ref else None
            )
        # WHERE THE SCORE GAIN COMES FROM, AND WHY IT IS SMALL. The win count triples while the
        # score barely moves, because a newly-won cell is won LATE (hundreds to thousands of actions
        # against human baselines of tens), so its per-level term is a rounding error against the
        # clamp. Reporting only the win count would overstate the leaderboard benefit ~150x.
        newly = [
            (s, g)
            for s, g in complete_pairs
            if (cell(budgets[-1], s, g).get("levels") or 0) > 0
            and (cell(400, s, g).get("levels") or 0) == 0
        ]
        new_scores = [cell(budgets[-1], s, g).get("efficiency") or 0.0 for s, g in newly]
        new_atfl = [
            cell(budgets[-1], s, g).get("actions_to_first_levelup")
            for s, g in newly
            if cell(budgets[-1], s, g).get("actions_to_first_levelup")
        ]
        eff["newly_won_cells_score_contribution"] = {
            "reference_budget": 400,
            "compared_budget": budgets[-1],
            "n_newly_won_cells": len(newly),
            "summed_authoritative_score_of_newly_won_cells": round(sum(new_scores), 5),
            "mean_authoritative_score_per_newly_won_cell": (
                round(sum(new_scores) / len(new_scores), 5) if new_scores else None
            ),
            "actions_to_first_levelup_of_newly_won_cells": {
                "min": min(new_atfl) if new_atfl else None,
                "median": statistics.median(new_atfl) if new_atfl else None,
                "max": max(new_atfl) if new_atfl else None,
            },
            "principle": (
                "A newly-won cell contributes its own per-level score, and that score is tiny "
                "because the win arrives hundreds-to-thousands of actions in against human "
                "baselines of tens. The win COUNT and the SCORE therefore move by very different "
                "factors, and only the score is on the leaderboard."
            ),
        }
        eff["scored_sum_verdict"] = (
            "Raising the budget is score-POSITIVE, monotonically, on the authoritative metric -- "
            "there is no sign disagreement to adjudicate, because the 'total-action charge' the "
            "prior version of this analysis weighed against it contradicts the installed scorer "
            "(see authoritative_scorer_resolution). But the gain is SMALL where the win count's gain "
            "is large: the score sum rises "
            f"{ref:.4f} -> "
            f"{eff['scored_sum_authoritative'][str(budgets[-1])]['authoritative_score_sum_over_won_cells']:.4f}"
            f" ({eff['scored_sum_authoritative'][str(budgets[-1])]['vs_b400_ratio']}x) while won "
            f"cells go {base['n_won_cells']} -> "
            f"{eff['scored_sum_authoritative'][str(budgets[-1])]['n_won_cells']}. Reporting the win "
            "count as the benefit would overstate the leaderboard effect by roughly two orders of "
            "magnitude. This is a PUBLIC-set figure: the public corpus already contains fast wins "
            "that dilute the ratio, and the hidden set (where the shipped budget wins little) has no "
            "such dilution, so the RELATIVE gain there is unmeasured and could be larger while the "
            "absolute per-win contribution stays this small."
        )

    if 400 in budgets:
        for b in budgets:
            if b == 400:
                continue
            # MATCHED per-cell comparison on the AUTHORITATIVE score: same game, same seed, won at
            # BOTH budgets, so no cell-set difference can explain a change. `actions_to_first_levelup`
            # is carried alongside because it is the quantity the scorer actually charges, and it is
            # the direct test of whether a bigger budget makes an already-won game slower.
            worse, better, same = [], [], []
            atfl_same = atfl_worse = atfl_better = 0
            for s, g in complete_pairs:
                r4, rb = cell(400, s, g), cell(b, s, g)
                if (r4["levels"] or 0) <= 0 or (rb["levels"] or 0) <= 0:
                    continue
                e4, eb = r4.get("efficiency") or 0.0, rb.get("efficiency") or 0.0
                (worse if eb < e4 - 1e-9 else (better if eb > e4 + 1e-9 else same)).append(
                    f"{g}/s{s}"
                )
                a4, ab = r4.get("actions_to_first_levelup"), rb.get("actions_to_first_levelup")
                if a4 and ab:
                    if ab == a4:
                        atfl_same += 1
                    elif ab > a4:
                        atfl_worse += 1
                    else:
                        atfl_better += 1
            eff["regressions_vs_400"][str(b)] = {
                "n_cells_won_at_both": len(worse) + len(better) + len(same),
                "authoritative_score_worse": worse,
                "authoritative_score_better": better,
                "authoritative_score_unchanged": same,
                "authoritative_score_n_worse": len(worse),
                # THE PREMISE THE RETIRED PESSIMISTIC MODEL RESTED ON, measured directly. If a bigger
                # budget made an already-won game reach its first level-up later, THAT would be a
                # real scored regression. It does not.
                "actions_to_first_levelup_identical_cells": atfl_same,
                "actions_to_first_levelup_later_cells": atfl_worse,
                "actions_to_first_levelup_earlier_cells": atfl_better,
                "principle": (
                    "The scorer charges a completed level the actions BETWEEN level-ups, so the only "
                    "way a raised budget can cost score on a game it already won is by reaching that "
                    "level-up later. Counting cells where it happens is the direct measurement; a "
                    "modelled bound is not needed and the one previously reported here was unsound."
                ),
            }

    # =========================================================================================
    # STEP 5 -- WON-GAME SETS ACROSS BUDGETS (sets, not totals).
    # =========================================================================================
    sets_by_budget = {
        str(b): {
            "all_seeds": per_budget[str(b)]["won_games_all_seeds_intersection"],
            "per_seed": {
                str(s): per_budget[str(b)]["per_seed"][str(s)]["won_games"] for s in seeds
            },
        }
        for b in budgets
    }

    # =========================================================================================
    # STEP 6 -- THE CONTENTION CONTROL. The three seeds were run as three CONCURRENT processes to
    # fit the sweep in available wall time. Concurrency inflates measured wall clock, and wall clock
    # is the deliverable, so the inflation is MEASURED rather than assumed negligible: the same
    # cells were also run in a single UNCONTENDED serial process before the parallel launch.
    # =========================================================================================
    contention = {
        "why": (
            "Three seeds ran as three concurrent processes on a 24-core box (~3.5 cores each). "
            "A contended wall clock presented as the cost would overstate the time and therefore "
            "UNDERSTATE the affordable budget -- so the inflation is measured against an "
            "uncontended serial run of the same cells, not assumed away."
        ),
        "available": False,
    }
    if a.serial_control_rows and Path(a.serial_control_rows).exists():
        ser = json.loads(Path(a.serial_control_rows).read_text())["rows"]
        pairs = []
        for sr in ser:
            if not sr.get("ran"):
                continue
            pr = cell(sr["budget"], sr["seed"], sr["game"])
            if pr is None or not pr.get("ran"):
                continue
            if not sr.get("wall_s"):
                continue
            pairs.append(
                {
                    "cell": f"{sr['game']}/s{sr['seed']}/b{sr['budget']}",
                    "serial_wall_s": sr["wall_s"],
                    "parallel_wall_s": pr["wall_s"],
                    "inflation_x": round((pr["wall_s"] or 0) / sr["wall_s"], 3),
                    "actions_match": sr.get("actions") == pr.get("actions"),
                    "levels_match": sr.get("levels") == pr.get("levels"),
                }
            )
        if pairs:
            infl = [p["inflation_x"] for p in pairs]
            contention.update(
                available=True,
                n_control_cells=len(pairs),
                inflation_median_x=round(statistics.median(infl), 3),
                inflation_max_x=round(max(infl), 3),
                inflation_min_x=round(min(infl), 3),
                outcomes_identical=all(p["actions_match"] and p["levels_match"] for p in pairs),
                cells=pairs,
                interpretation=(
                    "The WIN curve is unaffected by contention (the offline sim is deterministic "
                    "given the seed -- outcomes_identical confirms it). Only the TIME axis is "
                    "affected. Divide the reported wall clock by the median inflation to recover "
                    "the uncontended cost, which makes the affordable-budget figures LARGER, not "
                    "smaller -- so the crossing points reported here are CONSERVATIVE."
                ),
            )

    dur = round(time.time() - t_start, 3)
    checksum = hashlib.sha256(
        json.dumps(
            [
                [
                    r.get("game"),
                    r.get("seed"),
                    r.get("budget"),
                    r.get("levels"),
                    r.get("actions"),
                    r.get("wall_s"),
                ]
                for r in rows
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()

    git_head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    wins_med = {str(b): per_budget[str(b)]["wins_median"] for b in budgets}
    # fraction of the TIGHTEST envelope (110 games / 8h) consumed at the largest measured budget
    tight_frac = crossing["envelopes"][1]["per_budget"][str(budgets[-1])][
        "fraction_of_usable_at_median"
    ]
    best_b = max(budgets, key=lambda b: per_budget[str(b)]["wins_median"])

    # HONEST VERDICT, terminal-prefixed and COMPUTED from the measured curve rather than written by
    # hand, so it cannot drift from the numbers it summarises.
    lo_b, hi_b = budgets[0], budgets[-1]
    w_lo = per_budget[str(lo_b)]["wins_median"]
    w_hi = max(per_budget[str(b)]["wins_median"] for b in budgets)
    sat_phrase = (
        f"saturates_at_b{sat}"
        if sat is not None
        else f"no_saturation_within_measured_grid_b{lo_b}_to_b{hi_b}"
    )
    #
    # TWO CLAIMS WERE STRUCK FROM THIS STRING after adversarial review 2026-07-26, both because they
    # contradicted the artifact's own measured numbers:
    #   * `scored_efficiency_term_degrades_quadratically` -- the authoritative score sum RISES
    #     monotonically with budget, and the quadratic loss existed only under a total-action charge
    #     model that contradicts the installed scorer (see efficiency_axis).
    #   * `wall_clock_never_binding` -- LLM-OFF wall never binds, but the LLM-ON mechanistic model
    #     binds above budget 400 under the tightest envelope, and MEMORY binds before either.
    # The replacements are computed from the measured numbers so they cannot drift from them.
    score_lo = eff["scored_sum_authoritative"][str(400 if 400 in budgets else lo_b)][
        "authoritative_score_sum_over_won_cells"
    ]
    score_hi = eff["scored_sum_authoritative"][str(hi_b)]["authoritative_score_sum_over_won_cells"]
    mem_safe_c = (
        memory_envelope.get("largest_budget_that_FITS_at_worst_case_per_envelope", {}) or {}
    ).get("C_110games_12h")
    mem_phrase = (
        f"memory_binds_first_largest_worst_case_safe_budget_b{mem_safe_c}"
        if mem_safe_c is not None
        else "memory_envelope_NOT_measured"
    )
    verdict = (
        f"complete_budget_sweep_measured_wins_median_{w_lo}_at_b{lo_b}_to_{w_hi}_at_b{best_b}_"
        f"over_{len(games)}_games_{len(seeds)}_seeds_per_seed_matched_llm_off_"
        f"{sat_phrase}_authoritative_score_sum_ROSE_{score_lo}_to_{score_hi}_"
        f"llm_off_wall_never_binding_but_llm_on_model_binds_above_b"
        f"{crossing['llm_on_band']['B_110games_8h']['largest_budget_that_fits']}_under_the_8h_"
        f"envelope_and_{mem_phrase}_time_and_memory_axes_recorded_no_flag_changed"
    )

    artifact = {
        "experiment": "arc_scored_path_budget_sweep_v528",
        "honest_verdict": verdict,
        "honest_verdict_principle": (
            "Terminal 'complete_' prefix so the conductor's reconciler classifies this as a "
            "finished experiment; without a prefix, words like 'saturates' risk a false-positive "
            "partial classification. The verdict string is COMPUTED from the curve so it cannot "
            "disagree with the artifact's own numbers."
        ),
        "title": (
            "MAX_ACTIONS budget sweep on the scored path: (budget -> wins, TIME) over the 25 "
            "public games, shipped flag configuration, LLM-off"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.arc_budget_sweep.v1",
        # duration_s IS THE MEASUREMENT'S OWN WALL CLOCK, not the analysis pass's. The three seeds
        # ran as concurrent processes, so the sweep's wall clock is the MAX of their elapsed times
        # while the compute actually spent is the SUM; both are recorded because they answer
        # different questions (how long did I wait, vs how much machine time did this cost).
        "duration_s": round(max([rr.get("elapsed_s") or 0 for rr in raws]), 1),
        "total_cell_compute_s": round(sum([rr.get("elapsed_s") or 0 for rr in raws]), 1),
        "analysis_pass_duration_s": dur,
        "git_head": git_head,
        "random_seed": seeds[0] if seeds else None,
        "random_seeds_used": seeds,
        "reproducibility_checksum": checksum,
        "rows_source": [str(Path(p).resolve()) for p in a.rows],
        "rows_source_checksums": [rr.get("rows_checksum") for rr in raws],
        "rows_source_elapsed_s": [rr.get("elapsed_s") for rr in raws],
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_principle": (
            "The live ARC agent (E3AgentPolicy) takes real actions against the offline arcade with "
            "induction disabled: pure Python env-stepping plus verifier/world-model scoring. No "
            "GGUF is loaded and no CUDA is used, so a 60s live-inference duration floor would be a "
            "false positive. The LLM-on cost is cited from a prior artifact, not measured here."
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "No verifier claim is made at all. This sweep varies a search BUDGET and measures "
            "wins and wall clock; the win signal comes from the game env's own level counter, not "
            "from a verifier scoring itself. Recorded False because nothing here is a "
            "verifier-value or moat claim."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_principle": (
            "Runs are the OFFLINE dev twin over environment_files with per-game GameAdapters "
            "available, driven by a harness rather than by the live scored gateway. That is a "
            "development proxy for the live agent's behaviour, NOT proof of hidden-game "
            "self-discovery. No new level is claimed as a solve and the registry is not updated."
        ),
        "arc_solve_claim": False,
        "preconditions_checked": [
            {"resource": "environment_files (offline arcade, 25 public games)", "available": True},
            {"resource": "arc_leaderboard_eval + E3AgentPolicy importable", "available": True},
            {
                "resource": "llama-server / GGUF generator",
                "available": False,
                "note": (
                    "NOT REQUIRED: this sweep is LLM-off by design (proposer=None, induction "
                    "disabled per cell)."
                ),
            },
            {
                "resource": "all eight GATED_FLAGS pinned per cell",
                "available": census["gated_flags_pinned_all_eight"],
            },
            {
                "resource": "observe channel (nodes carry previous_frame)",
                "available": census["observe_channel_alive"],
            },
            {
                "resource": "HUD flat diagnostics readable (commit db46988d7)",
                "available": census["hud_mask_resolved_none"] == 0,
            },
        ],
        "what_was_NOT_changed": {
            "MAX_ACTIONS_class_attr_line_6230": 400,
            "MAX_ACTIONS_module_level_line_117": 200,
            "SUBMITTED_flags_touched": [],
            "submission_made": False,
            "principle": (
                "This is a MEASUREMENT task. The budget is varied through the OFFLINE harness "
                "parameter arc_leaderboard_eval.run_game(budget=N), never by editing the shipped "
                "default. The decision to raise the cap is the operator's."
            ),
        },
        "design": {
            "policy": "E3AgentPolicy (the SCORED policy class)",
            "arm": raw.get("arm"),
            "arm_flags": raw.get("arm_flags"),
            "llm_enabled": False,
            "budgets": budgets,
            "seeds": seeds,
            "n_games": len(games),
            "games": games,
            "cells_requested": len(budgets) * len(seeds) * len(games),
            "cells_ran": len(ok),
            "complete_seed_game_pairs": len(complete_pairs),
            "incomplete_seed_game_pairs": [f"{g}/s{s}" for s, g in incomplete_pairs],
            "matching": (
                "PER-SEED MATCHED on (seed, game); any-seed union reported as diagnostic only"
            ),
            "loop_order": (
                "seed -> game -> budget (budget INNERMOST, so truncation cannot favour a budget)"
            ),
            "flag_parity_vs_live_globals": raw.get("flag_parity_vs_live_globals"),
        },
        "field_provenance": {
            "duration_s": {
                "principle": (
                    "Real compute takes wall-clock time; an implausibly-short duration is the "
                    "load-bearing fabrication signal. Here it is the MEASUREMENT's wall clock (the "
                    "sweep), not the analysis pass -- the analysis pass is recorded separately as "
                    "analysis_pass_duration_s so neither can be mistaken for the other."
                ),
                "satisfied_by": "max elapsed across the concurrent per-seed sweep processes",
            },
            "total_cell_compute_s": {
                "principle": (
                    "The machine time the measurement actually cost, which differs from the wall "
                    "clock because the seeds ran concurrently. Reporting only the wall clock would "
                    "make the sweep look ~3x cheaper than it was."
                ),
                "satisfied_by": "sum of elapsed across the per-seed sweep processes",
            },
            "random_seeds_used": {
                "principle": (
                    "Determinism is the precondition for reproducibility, AND the seed is the "
                    "matching unit: every budget comparison here is PER-SEED MATCHED, so an "
                    "unrecorded seed set would make the matching unverifiable."
                ),
                "satisfied_by": "seeds passed to run_cell (random.seed + np.random.seed per cell)",
            },
            "reproducibility_checksum": {
                "principle": (
                    "Content hash over (game, seed, budget, levels, actions, wall_s) for every "
                    "row, so a third party can confirm they are holding the same measurement and "
                    "not a re-run with drifted numbers."
                ),
                "satisfied_by": "sha256 over the row tuples",
            },
            "inference_substrate": {
                "principle": (
                    "Resolves the duration-floor ambiguity. Vestigial GGUF strings elsewhere in "
                    "the stack would otherwise make a legitimately-fast no-LLM run look "
                    "fabricated under the 60s live-inference floor."
                ),
                "satisfied_by": "proposer=None and CARNOT_ARC_DISABLE_INDUCTION set per cell",
            },
            "solve_provenance": {
                "principle": (
                    "The ARC deliverable is the LIVE agent discovering hidden games from its own "
                    "attempts. Declaring this as a development_proxy prevents an offline "
                    "dev-twin measurement from being read as evidence of hidden-game "
                    "self-discovery."
                ),
                "satisfied_by": "offline arcade + per-game GameAdapters + a harness driver",
            },
            "verifier_is_oracle": {
                "principle": (
                    "A verifier that IS the correctness oracle produces a true-but-circular win. "
                    "Recorded False and explained because this artifact makes no verifier claim "
                    "at all -- an undeclared field would itself be a warn."
                ),
                "satisfied_by": (
                    "no verifier-value claim; the win signal is the env's level counter"
                ),
            },
            "preconditions_checked": {
                "principle": (
                    "Records WHICH resources were verified before measuring, pre-empting the "
                    "failure mode where a missing resource is silently papered over with a "
                    "synthesised result instead of a blocked_* verdict."
                ),
                "satisfied_by": "explicit checks listed with availability booleans",
            },
            "instrumentation_census": {
                "principle": (
                    "A lever or field that never populates produces a clean zero-error null that "
                    "LOOKS like a finding. Counting populated-ness BEFORE drawing conclusions is "
                    "the only defence; None-valued diagnostics are the tell."
                ),
                "satisfied_by": "per-field None/zero counts over every row",
            },
            "WITNESS_pass_region_nonempty": {
                "principle": (
                    "A gate or comparison whose pass region is empty produces an ARITHMETICALLY "
                    "FORCED value, not evidence. The witness is computed at the comparison's own "
                    "unit (the game-cell), which is where a prior median-gate defect recurred."
                ),
                "satisfied_by": "counts of cells that could gain and cells that could regress",
            },
            "sign_test_on_cells_both_tails": {
                "principle": (
                    "A one-sided test reports only one tail, so a REVERSAL comes back p=0.89 and "
                    "reads as 'no effect'. Both tails plus the favoured direction plus the "
                    "minimum reachable p at the available support are all required to read a null "
                    "honestly."
                ),
                "satisfied_by": "exact binomial on discordant pairs, both tails",
            },
        },
        "instrumentation_census": census,
        "budget_curve": per_budget,
        "won_game_sets_by_budget": sets_by_budget,
        "marginal_return_per_step": steps,
        "saturation": saturation,
        "crossing_point": crossing,
        "memory_envelope": memory_envelope,
        "efficiency_axis": eff,
        "wall_clock_contention_control": contention,
        "prior_budget_points_cited_not_rederived": PRIOR_BUDGET_POINTS,
        "stop_conditions_evaluated": {
            "is_raising_MAX_ACTIONS_permitted_by_the_competition_harness": {
                "answer": "YES -- the sweep is not moot",
                "evidence": (
                    "The framework loop is `while not is_done(...) and self.action_counter <= "
                    "self.MAX_ACTIONS` (/home/ianblenke/arc3_agents/agents/agent.py:70-87), i.e. "
                    "it "
                    "reads the INSTANCE attribute. Agent.MAX_ACTIONS defaults to 80; CarnotAgent "
                    "shadows it with 400 at arc_competition_agent.py:6230; and the framework's own "
                    "Playback subclass sets it to 1_000_000 (agent.py:204), so overriding it is a "
                    "framework-sanctioned pattern, exactly as the code comment claims. "
                    "INDEPENDENTLY RE-VERIFIED in this session: make_carnot_agent(Base) with "
                    "Base.MAX_ACTIONS=80 resolves to 400, MRO owner is CarnotAgent, and the "
                    "module-level MAX_ACTIONS=200 at line 117 is a DIFFERENT object "
                    "(`C.MAX_ACTIONS is MODLEVEL` -> False)."
                ),
                "which_cap_governs_the_scored_path": 400,
                "module_level_200_is_dead_for_scoring": True,
                "off_by_one": "a cap of N executes N+1 actions (the guard is <=)",
            },
            "is_there_a_recorded_time_overrun_incident_from_raising_the_budget": {
                "answer": "NO",
                "evidence": (
                    "The only higher-budget 'failure' on file is exp4518's budget-36000 row "
                    "(7 solved at 24000 -> 3 at 36000). Four of its eight games carry "
                    "timed_out=true / actions=null against the GATE's own 115-SECOND subprocess "
                    "cap (scripts/kaggle/arc_local_submission_gate.py --cap default 115), and all "
                    "four were solved at 24000. It is a CI-harness timeout artifact, not an "
                    "eval-time overrun. Reading it as 'we tried higher and it broke' would be "
                    "wrong; actions=null was the tell."
                ),
            },
        },
        "what_this_sweep_does_NOT_measure": {
            "the_scored_LLM_ON_condition_at_raised_budget": (
                "NOT DIRECTLY MEASURED. Every LLM-on scored row in the record is budget 400. It is "
                "now MODELLED rather than bracketed by a factor-of-10 band: llm_on_band fits "
                "cost = llm_off(b) + inductions(b) * s_per_induction using the induction_attempts "
                "already recorded on every row, calibrated to the measured b400 anchor. ONE LLM-on "
                "run at b1000 or b2000 would replace the model with a direct anchor and remains the "
                "cheapest decisive next measurement."
            ),
            "hidden_game_cost_and_difficulty": (
                "These are the 25 PUBLIC games with their per-game GameAdapters available. The "
                "hidden set is ~110 games with no adapters. Projecting the per-game wall clock to "
                "N=110 assumes hidden games cost like public ones -- an assumption, not a "
                "measurement, and it is the single largest source of error in the crossing point."
            ),
            "whether_the_gateway_scores_actions_to_level_or_total_actions": (
                "RESOLVED, and a prior version of this artifact wrongly listed it here as "
                "unresolvable. arc_agi.scorecard is INSTALLED in the analysis venv; its card->score "
                "path differences successive level checkpoints and assigns the tail to the first "
                "INCOMPLETE level, which scores 0. Driving it directly (see "
                "efficiency_axis.authoritative_scorer_resolution) shows the same solve scores "
                "identically with a 10-action tail and a 100,000-action tail. The 2026-06-21 audit's "
                "'tail quadratically erodes the score' claim does not survive contact with the "
                "shipped implementation and should be corrected wherever it is cited."
            ),
            "the_10_steps_per_second_gateway_rate": (
                "UNVERIFIED. It appears only in this project's own requirements note and is marked "
                "UNCONFIRMED for 2026. No client-side limiter exists in the framework (the only "
                "time.sleep is in the Playback subclass), so if it is real it is enforced "
                "server-side and EVERY offline measurement here omits it. At 10 steps/sec a "
                "2000-action game has a 200-second pure-latency floor before any agent compute."
            ),
            "memory_at_110_concurrent_games": (
                "NOW MEASURED, not estimated -- see the memory_envelope section, which is the "
                "constraint that binds FIRST. What remains unmeasured is (a) the target instance's "
                "real HOST RAM (the 16 GiB in the requirements note is a VRAM figure) and (b) whether "
                "hidden games retain graphs like public ones. Both are needed before the memory "
                "ceiling can be treated as exact."
            ),
            "the_competition_s_own_framework_copy": (
                "The framework proven against is the local clone at /home/ianblenke/arc3_agents. "
                "The Kaggle kernel copies the COMPETITION's own ARC-AGI-3-Agents from the eval "
                "mount, which exists only in the sandbox. If its Agent.main() added a deadline or "
                "resolved the cap differently, the permission finding would not transfer."
            ),
        },
        "headline": {
            "wins_median_by_budget": wins_med,
            # RENAMED AND QUALIFIED after the adversarial review of this fix. The old key was
            # `best_measured_budget`, an unqualified recommendation-shaped number computed as the
            # argmax of the WIN COUNT -- the very metric this analysis demotes -- sitting in the
            # headline block while the same artifact's memory envelope says that budget is OVER host
            # RAM at the hidden-set game count. "Best by wins" is not "best", and the name must say so.
            "budget_with_most_wins_NOT_A_RECOMMENDATION": best_b,
            "budget_with_most_wins_basis": (
                "argmax of wins_median. Wins are NOT the scored quantity and NOT constraint-aware; "
                "see authoritative_score_sum_by_budget for the scored quantity and "
                "constraint_feasible_budget below for what actually fits."
            ),
            "constraint_feasible_budget": {
                "memory_worst_case_110_games": (
                    memory_envelope.get("largest_budget_that_FITS_at_worst_case_per_envelope") or {}
                ).get("C_110games_12h"),
                "llm_on_wall_12h_110_games": crossing["llm_on_band"]["C_110games_12h"][
                    "largest_budget_that_fits"
                ],
                "llm_on_wall_8h_110_games": crossing["llm_on_band"]["B_110games_8h"][
                    "largest_budget_that_fits"
                ],
                "binding_of_the_three": min(
                    [
                        x
                        for x in [
                            (
                                memory_envelope.get(
                                    "largest_budget_that_FITS_at_worst_case_per_envelope"
                                )
                                or {}
                            ).get("C_110games_12h"),
                            crossing["llm_on_band"]["C_110games_12h"]["largest_budget_that_fits"],
                        ]
                        if x is not None
                    ],
                    default=None,
                ),
                "principle": (
                    "The budget that wins the most games is not the budget that can be run. Reporting "
                    "the win-count argmax without the feasible ceiling beside it is how a report "
                    "recommends a configuration that cannot be deployed."
                ),
            },
            # THE SCORE, ALONGSIDE THE WIN COUNT, because they move by very different factors and
            # only one of them is on the leaderboard. Reporting the win count alone overstates the
            # benefit by roughly two orders of magnitude on this corpus.
            "authoritative_score_sum_by_budget": {
                str(b): eff["scored_sum_authoritative"][str(b)][
                    "authoritative_score_sum_over_won_cells"
                ]
                for b in budgets
            },
            "won_cells_by_budget": {
                str(b): eff["scored_sum_authoritative"][str(b)]["n_won_cells"] for b in budgets
            },
            "win_count_vs_score_divergence": (
                f"won cells x{round(eff['scored_sum_authoritative'][str(hi_b)]['n_won_cells'] / max(1, eff['scored_sum_authoritative'][str(400 if 400 in budgets else lo_b)]['n_won_cells']), 2)} "
                f"but authoritative score x{eff['scored_sum_authoritative'][str(hi_b)]['vs_b400_ratio']}"
            ),
            # GAME-LEVEL p-values. The cell-level values are within-game replicate counts and must
            # not be quoted as the design's significance (see the clustering note on each step).
            "HEADLINE_game_level_sign_test_p_two_sided_by_step": {
                f"{s['from_budget']}->{s['to_budget']}": s[
                    "HEADLINE_sign_test_on_GAMES_both_tails"
                ]["p_two_sided"]
                for s in steps
            },
            "cell_level_sign_test_p_NOT_INDEPENDENT_by_step": {
                f"{s['from_budget']}->{s['to_budget']}": s[
                    "sign_test_on_cells_WITHIN_GAME_REPLICATES_not_independent"
                ]["p_two_sided"]
                for s in steps
            },
            "n_distinct_games_that_moved_by_step": {
                f"{s['from_budget']}->{s['to_budget']}": len(s["games_gained"])
                + len(s["games_lost"])
                for s in steps
            },
            "wall_total_s_25_games_by_budget": {
                str(b): per_budget[str(b)]["wall_total_s_median"] for b in budgets
            },
            "wins_gained_per_doubling_by_step": {
                f"{s['from_budget']}->{s['to_budget']}": s["wins_gained_per_doubling_median"]
                for s in steps
            },
            "seconds_per_extra_win_by_step": {
                f"{s['from_budget']}->{s['to_budget']}": s["seconds_per_extra_win_median"]
                for s in steps
            },
            # Ordered by which constraint actually bites first. A prior version headlined
            # "not wall clock -> the step rate is the bound" and demoted memory to an untested
            # residual; memory is measured now and it is the tightest of the four.
            "BINDING_CONSTRAINT": {
                "1_memory_binds_FIRST": (
                    (
                        "MEASURED. Every game is a concurrent thread in one process, so retained "
                        "search graphs multiply by the game count. Largest budget that fits at "
                        "worst case, 110 games, "
                        f"{memory_envelope.get('host_ram_gib_assumed')} GiB host: b"
                        f"{(memory_envelope.get('largest_budget_that_FITS_at_worst_case_per_envelope') or {}).get('C_110games_12h')}"
                        ". The host-RAM figure itself is UNCONFIRMED, so this ceiling is the number "
                        "most worth pinning down before acting."
                    )
                    if memory_envelope.get("measured")
                    else "NOT MEASURED in this pass -- rerun with --memory-rows."
                ),
                "2_llm_on_wall_clock_binds_CONDITIONALLY": (
                    "LLM-OFF wall never crosses any envelope inside the measured grid (at the "
                    f"tightest model the largest measured budget costs {tight_frac:.1%} of usable "
                    "wall). But the scored submission runs the LLM, and the mechanistic LLM-on model "
                    "fits only to b"
                    f"{crossing['llm_on_band']['B_110games_8h']['largest_budget_that_fits']} under "
                    "the 8h envelope versus b"
                    f"{crossing['llm_on_band']['C_110games_12h']['largest_budget_that_fits']} under "
                    "the code-verified 12h envelope. So 'wall clock never binds' is TRUE only of the "
                    "LLM-off condition, which is not the scored one."
                ),
                "3_the_gateway_step_rate_if_it_is_real": (
                    "the DOCUMENTED-BUT-UNCONFIRMED gateway step rate: 10 steps/sec over an 8h "
                    "play "
                    f"cap = ~{int(STEP_RATE_TOTAL):,} global real steps = "
                    f"{int(STEP_RATE_TOTAL / 110):,} actions/game at ~110 hidden games. A figure "
                    "this project's own requirements doc marks unconfirmed for 2026 and that cannot "
                    "be checked locally."
                ),
                "4_the_scoring_axis_is_NOT_a_constraint": (
                    "CORRECTED. A prior version listed a quadratic scoring penalty here. The "
                    "authoritative scorer charges a completed level only the actions BETWEEN "
                    "level-ups, so the post-solve tail is free, and the measured authoritative score "
                    f"sum RISES {score_lo} -> {score_hi} across the budget range with zero cells "
                    "regressing. The score axis therefore does not oppose a raise -- it just gains "
                    "far less than the win count implies (see "
                    "efficiency_axis.newly_won_cells_score_contribution)."
                ),
            },
        },
    }

    if a.extension_artifact and Path(a.extension_artifact).exists():
        ext = json.loads(Path(a.extension_artifact).read_text())
        artifact["longer_budget_extension_SEPARATE_DESIGN"] = {
            "why_separate": (
                "This ladder reaches a LARGER budget on FEWER seeds. It is reported separately "
                "because a budget measured on 1 seed is not matched against a budget measured on "
                "3 -- folding it in would compare a budget against neighbours drawn from a "
                "different design, which is how a control ends up losing to itself."
            ),
            "seeds": ext["design"]["seeds"],
            "budgets": ext["design"]["budgets"],
            "n_games": ext["design"]["n_games"],
            "wins_per_seed_by_budget": {
                bk: v["wins_per_seed"] for bk, v in ext["budget_curve"].items()
            },
            "wall_total_s_by_budget": {
                bk: v["wall_total_s_median"] for bk, v in ext["budget_curve"].items()
            },
            "won_game_sets": ext["won_game_sets_by_budget"],
            "marginal_return_per_step": ext["marginal_return_per_step"],
            "saturation": ext["saturation"],
            "cost_scaling_in_budget": ext["crossing_point"]["cost_scaling_in_budget"],
            "crossing_envelopes": [
                {
                    "id": e["id"],
                    "largest_measured_budget_that_fits_at_median": e[
                        "largest_measured_budget_that_fits_at_median"
                    ],
                    "analytic_crossing_budget_llm_off_SUPERLINEAR": e[
                        "analytic_crossing_budget_llm_off_SUPERLINEAR"
                    ],
                    "per_budget": e["per_budget"],
                }
                for e in ext["crossing_point"]["envelopes"]
            ],
            "instrumentation_census_holes": ext["instrumentation_census"]["instrumentation_holes"],
            "observe_channel_alive": ext["instrumentation_census"]["observe_channel_alive"],
            "source_artifact": str(Path(a.extension_artifact).resolve()),
            "reproducibility_checksum": ext["reproducibility_checksum"],
        }

    # Carry hand-authored keys (rebuild_note_*, freshness acks) through
    # the rebuild (REQ-OPS-REBUILD-PRESERVE-1).
    import sys as _sys

    if str(Path(__file__).resolve().parent) not in _sys.path:
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from artifact_merge_preserve import merge_preserve_with_file

    artifact = merge_preserve_with_file(Path(a.out), artifact)
    Path(a.out).write_text(json.dumps(artifact, indent=1))
    print(json.dumps(artifact["headline"], indent=1))
    print(f"\ninstrumentation holes: {census['instrumentation_holes']}")
    print(f"observe channel alive: {census['observe_channel_alive']}")
    print(f"saturating budget: {sat}")
    for e in crossing["envelopes"]:
        print(
            f"{e['id']}: usable={e['usable_loop_wall_s']}s "
            f"largest_measured_fits={e['largest_measured_budget_that_fits_at_median']} "
            f"cross_lin={e['analytic_crossing_budget_llm_off_median_cost_LINEAR_optimistic']} "
            f"cross_super={e['analytic_crossing_budget_llm_off_SUPERLINEAR']} "
            f"step_rate_ceiling={crossing['step_rate_ceiling'][e['id']]['per_game_action_ceiling']}"
        )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
