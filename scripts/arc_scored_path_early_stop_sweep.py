"""THE EARLY-STOP GRACE SWEEP on the SCORED path (E3AgentPolicy, shipped flag configuration).

WHAT IS BEING SWEPT, AND WHY IT IS A PARAMETER AND NOT A FLAG FLIP.
`arc_competition_agent.SUBMITTED_EARLY_STOP_GRACE` is currently DEAD CODE: it is declared at :128
and read NOWHERE. `StepwiseExplorer.__init__` accepts `early_stop_grace` (:1003) and `is_done`
implements the window (:3936-3946), but `E3AgentPolicy.__init__` never forwards it. So flipping the
global would have been a silent no-op -- an UNINSTRUMENTED ARM producing a clean, meaningless null.
This sweep therefore sets the parameter on the constructed explorer (`run_cell(early_stop_grace=)`)
and READS IT BACK per row, and touches no `SUBMITTED_*` global at all. The shipped configuration is
unchanged; the decision is the operator's.

WHAT THE MECHANISM DOES. After the first level-up, a new level-up RESETS a window of
`early_stop_grace` LOOP ITERATIONS (`len(frames)`, RESETs included -- not actions). If no new level
appears inside the window, the run stops. Riding consecutive level-ups keeps it alive, so reachable
deeper levels are still solved; what is cut is the tail after the LAST findable level.

WHAT IT CAN AND CANNOT BUY -- the gate design follows from this, and it is the reason the gate this
sweep was originally handed had to be restated before it was run.
The authoritative scorer (arc_agi.scorecard, the package the competition gateway itself runs)
charges a COMPLETED level only `actions_at_level - prev_actions` -- a difference of successive
level-up checkpoints (scorecard.py:479). Actions after the LAST level-up land in the first
NOT-completed level's bucket, and an incomplete level scores 0.0 no matter how many actions it was
charged (scorecard.py:178-183). THE POST-SOLVE TAIL IS THEREFORE EXACTLY SCORE-FREE. Cutting it can
only:
  - leave the score UNCHANGED (the expected case), or
  - LOSE score, if the window closes before a level-up that would otherwise have arrived.
There is no reachable path by which cutting a score-free tail RAISES the score. A gate of the form
"the efficiency sum must improve" therefore has an EMPTY PASS REGION and is arithmetically forced --
the same defect that made a prior experiment's "levels preserved" gate uninformative because both
arms topped out at the same level. It is restated here as NON-INFERIORITY, and the mechanism's real
payoff (wall clock and actions, which buy budget for other games inside the eval's 12h cap) is the
headline.

THE TWO GATES ACTUALLY EVALUATED:
  SAFETY        -- per-seed matched, per (game, seed, budget): levels[grace] >= levels[control].
                   The window resets on level-up, so this is a property to VERIFY, not assume.
  NON-INFERIORITY -- the authoritative per-game `efficiency` (arc_leaderboard_eval.run_game ->
                   arc_agi EnvironmentScoreCalculator) must not fall, per-seed matched.
  BENEFIT (headline, not a gate) -- actions and wall clock saved at non-regressing levels.

EACH GATE CARRIES A COMPUTED WITNESS AT ITS OWN LEVEL OF AGGREGATION. A safety gate over cells that
could never have regressed is forced: only cells that actually reached >=2 levels in the control, or
whose measured inter-level-up gap exceeds the grace, are MOVABLE. The witness counts them. If the
movable set is empty for an arm, that arm is stamped UNINTERPRETABLE rather than reported as a pass.

LOOP ORDER: seed -> game -> budget -> GRACE INNERMOST. A run cut short can therefore never leave one
grace value with more measured cells than another inside a completed (seed, game, budget) group.

BUDGETS. Two conditions, because grace values above the budget CANNOT FIRE (the window counts loop
iterations and the loop runs at most `budget` of them):
  b400  -- the SHIPPED scored cap (`CarnotAgent.MAX_ACTIONS = 400`, arc_competition_agent.py:6230).
           This is the decision-relevant condition.
  b2000 -- a raised budget where the tail is large and the measured inter-level-up gap distribution
           (median ~1200, max ~2800) actually bites, so the SAFETY gate has something to catch.

GRID CHOICE IS ITSELF A MEASUREMENT DECISION, and the first b400 grid got it wrong (adversarial
review, 2026-07-26). A grace is SAFE in-sample iff it exceeds every at-risk inter-level-up gap, and
FIRES iff it is below the largest post-level-up tail -- both in FRAMES, and both computable from the
CONTROL arm before any treatment arm runs. At b400 those bounds are 340.2 and 372.3 frames, and the
grid 50/100/150/200/400 contained NO point between them: four values that could only fail and one
(400) that equalled the budget and was therefore inert BY CONSTRUCTION. A grid like that can only
support "none of the values TESTED is safe" -- never the existence claim over the parameter space
that the gate's first name asserted. Grace 350 was run afterwards to measure the window rather than
extrapolate it. TWO RULES FOR ANY FUTURE GRID HERE:
  1. Compute the window from the control arm first, and put at least one point strictly inside it.
  2. Never place the top grid point AT the budget -- it is guaranteed inert and wastes the arm.
  3. But do not place a point ABOVE the largest tail either (e.g. 380 at b400): it is below the
     budget, so the census's budget rule would call it FIRING-CAPABLE, and a firing-capable arm that
     never fires stamps the whole artifact uninterpretable. The census now classifies inertness by
     MEASURED tail reachability for exactly this reason.

LLM-OFF by design: this is a search-behaviour parameter, and the LLM-on cost (~61x per cell) cannot
cover 2 budgets x 6 grace values x 25 games x 3 seeds. `run_cell` is IMPORTED, never reimplemented,
so a grace delta cannot be an instrumentation delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

import arc_scored_path_lever_harness as harness  # noqa: E402


def _parse_grace(s: str) -> list[int | None]:
    """'none,50,100' -> [None, 50, 100]. None must be first: it is the control."""
    out: list[int | None] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(None if tok.lower() in ("none", "null", "control") else int(tok))
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graces", default="none,50,100,150,200,400")
    ap.add_argument("--budgets", default="400")
    ap.add_argument("--seeds", default="20260724,20260725,20260726")
    ap.add_argument("--games", default="")
    ap.add_argument("--arm", default="S")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tag",
        default="",
        help="free-form label carried into every row (e.g. 'contention_control_serial')",
    )
    a = ap.parse_args(argv)

    graces = _parse_grace(a.graces)
    budgets = [int(x) for x in a.budgets.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]
    games = [g for g in a.games.split(",") if g] or list(harness.GAMES_25)

    # Pinning keeps an ARM stable across a later flip; only this check says whether the arm named
    # "S" is still the LIVE configuration. Recorded either way.
    parity = harness.assert_shipped_dict_matches_module_globals()
    if parity["pinned_vs_live_drift"]:
        print(f"[WARNING] pinned-vs-live flag drift: {parity['pinned_vs_live_drift']}", flush=True)

    # The parameter must be dead code for the "no SUBMITTED_* flag was touched" claim to mean
    # anything -- assert it rather than assert it by eye.
    from carnot.agentic import arc_competition_agent as comp

    submitted_grace_at_start = comp.SUBMITTED_EARLY_STOP_GRACE

    rows: list[dict] = []
    t0 = time.time()
    total = len(seeds) * len(games) * len(budgets) * len(graces)
    for seed in seeds:
        for game in games:
            for budget in budgets:
                for grace in graces:  # <-- INNERMOST
                    row = harness.run_cell(
                        game,
                        seed,
                        budget=budget,
                        proposer=None,
                        llm=False,
                        extra_kwargs=dict(harness.ARMS[a.arm]),
                        arm=f"{a.arm}_llmoff_b{budget}_g{'none' if grace is None else grace}",
                        early_stop_grace=grace,
                    )
                    row["sweep_tag"] = a.tag
                    rows.append(row)
                    print(
                        f"[{len(rows):4}/{total}] {game:5} s{seed} b{budget:<5} "
                        f"g={'none' if grace is None else grace:<5} "
                        f"lv={row.get('levels')} act={row.get('actions')} "
                        f"eff={row.get('efficiency')} "
                        f"stop={row.get('early_stopped')} "
                        f"applied={row.get('early_stop_grace_applied')} "
                        f"lua={row.get('level_up_actions')} "
                        f"tail={row.get('actions_after_last_levelup')} "
                        f"rst={row.get('n_resets')} wall={row.get('wall_s')}s "
                        f"cum={round(time.time() - t0, 1)}s",
                        flush=True,
                    )
                    Path(a.out).write_text(
                        json.dumps(
                            {
                                "sweep": "arc_scored_path_early_stop_sweep",
                                "tag": a.tag,
                                "rows": rows,
                                "graces_requested": ["none" if g is None else g for g in graces],
                                "budgets_requested": budgets,
                                "seeds_requested": seeds,
                                "games_requested": games,
                                "arm": a.arm,
                                "arm_flags": dict(harness.ARMS[a.arm]),
                                "llm_enabled": False,
                                "shipped_agent_max_actions": 400,
                                "submitted_early_stop_grace_at_start": submitted_grace_at_start,
                                "submitted_early_stop_grace_at_end": comp.SUBMITTED_EARLY_STOP_GRACE,
                                "flag_parity_vs_live_globals": parity,
                                "elapsed_s": round(time.time() - t0, 1),
                                "rows_checksum": hashlib.sha256(
                                    json.dumps(
                                        [
                                            [
                                                r.get("game"),
                                                r.get("seed"),
                                                r.get("budget"),
                                                r.get("early_stop_grace"),
                                                r.get("levels"),
                                                r.get("actions"),
                                                r.get("efficiency"),
                                            ]
                                            for r in rows
                                        ],
                                        sort_keys=True,
                                    ).encode()
                                ).hexdigest(),
                            },
                            indent=2,
                        )
                    )
    # THE FLAG MUST BE UNTOUCHED. This sweeps a parameter; if the module global moved, the run is
    # not the measurement it claims to be.
    assert comp.SUBMITTED_EARLY_STOP_GRACE == submitted_grace_at_start, (
        "SUBMITTED_EARLY_STOP_GRACE changed during the sweep -- this run swept a flag, not a "
        "parameter, and its rows are not comparable to the shipped configuration"
    )
    print(
        f"[done] {len(rows)} rows in {round(time.time() - t0, 1)}s -> {a.out} "
        f"(SUBMITTED_EARLY_STOP_GRACE still {comp.SUBMITTED_EARLY_STOP_GRACE!r})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
