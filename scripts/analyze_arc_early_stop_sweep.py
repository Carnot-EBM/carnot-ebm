"""ANALYSER for the early-stop grace sweep. Emits the milestone artifact.

READ THE SWEEP DRIVER'S DOCSTRING FIRST (`scripts/arc_scored_path_early_stop_sweep.py`) -- it
explains what is swept and why the gate handed to this measurement had to be restated before it
could be run. The short version, because it determines every design choice below:

  The authoritative scorer charges a COMPLETED level `actions_at_level - prev_actions`
  (arc_agi/scorecard.py:479), and an INCOMPLETE level scores 0.0 regardless of how many actions it
  was charged (:178-183). The post-solve tail therefore costs EXACTLY ZERO score. Cutting it can
  only leave the score unchanged or -- if the window closes before a level-up that would otherwise
  have arrived -- LOSE score. So "the efficiency sum must improve" is a gate with an EMPTY PASS
  REGION. Running it as stated would have manufactured a forced failure and called it a finding.

THE GATES THIS FILE ACTUALLY EVALUATES, each with a COMPUTED WITNESS at its own aggregation level:

  G-SAFETY   levels[grace] >= levels[control], per-seed matched, per (game, seed, budget).
             WITNESS: the MOVABLE set. A cell whose control reached <=1 level CANNOT regress -- the
             window only arms AFTER the first level-up, and stopping after a level-up cannot undo
             it. So only cells with control levels >= 2 are movable, and only those with a measured
             inter-level-up gap exceeding the grace are AT RISK. Both are counted. An arm with an
             empty movable set is stamped UNINTERPRETABLE, not reported as a pass.
             AND (2026-07-26) A **PASS** IS ADDITIONALLY GATED ON FALSIFIABILITY. `movable > 0` is
             the right guard for a FAIL; it is the wrong guard for a PASS. For "grace G is SAFE" to
             be falsifiable some cell must have been AT RISK -- with at_risk == 0 the pass restates
             its own precondition ("G exceeds every observed gap") and carries no independent
             information. Such an arm is stamped PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK; an arm where
             an at-risk cell survived is PASS_AT_RISK_CELLS_SURVIVED. The distinction is not academic:
             grace 1300 is an unfalsifiable pass at b2000 and REGRESSES SIX CELLS at b4000.

THREE THINGS THE 2026-07-26 ADVERSARIAL REVIEW ADDED, each closing a way a true statement here could
be read as a stronger one:

  SAFE-FIRING WINDOW (per condition). A grace is safe in-sample iff it exceeds every at-risk gap and
  fires iff it is below the largest post-level-up tail -- BOTH computable from the CONTROL arm. The
  original b400 grid (50/100/150/200/400) contained NO point strictly inside (340.2, 372.3), so its
  "no firing grace value is safe" answer was a GRID ARTIFACT, and the top grid point equalled the
  budget and was inert by construction. The gate is renamed to its measured scope (a **TESTED**
  value), the grid's adequacy is now a gate of its own, and the missing point (grace 350) was RUN:
  it fires on 1 of 75 cells, costs no level, moves no score, and saves 0.072% of corpus actions.
  So the safe window is nearly empty of VALUE, not empty of MEMBERS -- which is a different, and
  defensible, reason not to ship the mechanism at the shipped budget.

  SCOPE AND POWER (hoisted to the top level). At b400, 64 of 75 control cells reach zero levels and
  the 3 that reach two are all ONE game, so every level-regression is that game's and the decisive
  gap is one seed's. With at most 2 at-risk cells the smallest reachable two-sided p is 0.5 -- no
  significance is attainable at b400 no matter how many seeds are added. Separately 4 of 75 cells
  hold 91.35% of the corpus score total (one cell 52.58%), so a score delta is a statement about
  those cells. All of this sat in per-arm residuals under a headline that read corpus-wide.

  OUT-OF-SAMPLE FRAGILITY. For every grace tested at more than one budget, whether it passed in one
  condition and regressed levels in another. Three of four did. That is the strongest available
  evidence that no FIXED grace generalises, and it uses only arms already measured.

  G-NONINF   sum(efficiency[grace]) >= sum(efficiency[control]), per-seed matched.
             WITNESS: the set of cells whose LEVEL-UP CHECKPOINT VECTOR differs between arms. The
             score is a function of that vector alone (plus the human baselines), so a cell with an
             identical vector is FROZEN by construction and its equal efficiency is not evidence.

  BENEFIT (headline, deliberately NOT a gate): actions, tail actions and wall clock saved on cells
             that did not regress. This is what the mechanism can actually buy.

EVERY COMPARISON IS PER-SEED MATCHED on the (game, seed, budget) key. An arm's per-seed set is never
compared against a control any-seed UNION -- that comparison shows a control failing against itself.

BOTH TAILS ARE REPORTED for every test, with the MINIMUM REACHABLE p at the available support, so a
REVERSAL cannot read as "no effect" and a small movable set cannot read as "not significant" when no
p below the reported floor was reachable in the first place.

THE REFUTED CHARGE MODEL IS COMPUTED AS A SENSITIVITY CHECK, not weighed as an equal reading: the
gateway's own installed source settles it. It is reported so a reader can see what the conclusion
would have been under the alternative, and see that the DIRECTION of the safety verdict does not
depend on the choice.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")


# --------------------------------------------------------------------------------------------
# statistics -- both tails, always, plus the reachable floor
# --------------------------------------------------------------------------------------------
def sign_test_both_tails(deltas: list[float]) -> dict:
    """Exact two-sided sign test on paired deltas, reporting BOTH one-sided tails.

    Reports `p_min_reachable`: the smallest two-sided p ANY outcome could have produced at this
    support (all n non-zero deltas on one side). A test whose floor is above 0.05 cannot reject at
    0.05 no matter what the data said, and reporting p alone would hide that.
    """
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = sum(1 for d in deltas if d == 0)
    n = pos + neg

    def _binom_tail_ge(k: int, n_: int) -> float:
        if n_ == 0:
            return 1.0
        return sum(math.comb(n_, i) for i in range(k, n_ + 1)) / (2.0**n_)

    p_greater = _binom_tail_ge(pos, n)  # H1: deltas tend positive
    p_less = _binom_tail_ge(neg, n)  # H1: deltas tend negative
    p_two = min(1.0, 2.0 * min(p_greater, p_less)) if n else 1.0
    p_floor = min(1.0, 2.0 * (0.5**n)) if n else 1.0
    return {
        "n_pairs_total": len(deltas),
        "n_nonzero": n,
        "n_positive": pos,
        "n_negative": neg,
        "n_zero": zero,
        "p_one_sided_greater": round(p_greater, 6),
        "p_one_sided_less": round(p_less, 6),
        "p_two_sided": round(p_two, 6),
        "p_min_reachable_two_sided": round(p_floor, 6),
        "direction_favoured": ("positive" if pos > neg else ("negative" if neg > pos else "tie")),
        "can_reject_at_0.05": bool(p_floor <= 0.05),
        "note": (
            "p_min_reachable_two_sided is the smallest two-sided p ANY outcome could produce at "
            "this support. If it exceeds 0.05 the test could not have rejected regardless of the "
            "data, and 'not significant' carries no information."
        ),
    }


# --------------------------------------------------------------------------------------------
# the two charge models
# --------------------------------------------------------------------------------------------
def score_actions_to_level(level_up_actions, total_actions, baselines) -> float:
    """The AUTHORITATIVE charge model, reimplemented ONLY as a cross-check on the row's own
    `efficiency` field (which comes from the installed arc_agi scorer via run_game).

    Charges each completed level the DIFFERENCE of successive checkpoints; the tail lands in the
    first incomplete level, which scores 0.0. Mirrors arc_agi/scorecard.py:474-491.
    """
    if not baselines:
        return 0.0
    try:
        from arc_agi.scorecard import EnvironmentScoreCalculator
    except Exception:
        return 0.0
    calc = EnvironmentScoreCalculator()
    prev = 0
    for li, base in enumerate(baselines):
        if li < len(level_up_actions):
            at = level_up_actions[li]
            calc.add_level(
                level_index=li + 1, completed=True, actions_taken=at - prev, baseline_actions=base
            )
            prev = at
        else:
            calc.add_level(
                level_index=li + 1,
                completed=False,
                actions_taken=total_actions - prev,
                baseline_actions=base,
            )
            prev = total_actions
    return round(float(calc.to_score(include_levels=False).score), 6)


def score_total_action_charge(level_up_actions, total_actions, baselines) -> float:
    """The REFUTED alternative, computed as a sensitivity check only.

    Under this reading the agent is charged its TOTAL actions for the deepest level it completed --
    i.e. the post-solve tail is billed to the last completed level rather than to the incomplete one.
    This is the reading under which raising the budget looked catastrophic and the b200-beats-b400
    inversion appeared. It contradicts the installed scorer's source, which the gateway itself runs;
    it is retained here so the reader can see that the SAFETY verdict does not depend on the choice.
    """
    if not baselines:
        return 0.0
    try:
        from arc_agi.scorecard import EnvironmentScoreCalculator
    except Exception:
        return 0.0
    calc = EnvironmentScoreCalculator()
    prev = 0
    n_done = len(level_up_actions)
    for li, base in enumerate(baselines):
        if li < n_done:
            # the LAST completed level absorbs the tail under this model
            at = total_actions if li == n_done - 1 else level_up_actions[li]
            calc.add_level(
                level_index=li + 1, completed=True, actions_taken=at - prev, baseline_actions=base
            )
            prev = at
        else:
            calc.add_level(
                level_index=li + 1, completed=False, actions_taken=0, baseline_actions=base
            )
    return round(float(calc.to_score(include_levels=False).score), 6)


_BASELINE_CACHE: dict[str, list[int]] = {}


def baselines_for(game: str) -> list[int]:
    if game in _BASELINE_CACHE:
        return _BASELINE_CACHE[game]
    # USE THE HARNESS'S OWN GETTER, not a hand-rolled attribute read. The first draft did
    # `getattr(env, "baseline_actions")`, which returns nothing for these envs (the attribute lives
    # on `env.info`), so `baselines_for` returned [] for EVERY game and the sensitivity check
    # silently produced 0.0 -> 0.0 on the whole corpus -- a dead diagnostic that looked like "the
    # alternative charge model shows no difference". Reusing `_baseline_actions` means the
    # sensitivity check reads the SAME baselines the authoritative score does, by construction.
    try:
        sys.path.insert(0, str(REPO / "python"))
        sys.path.insert(0, str(REPO / "scripts"))
        import arc_leaderboard_eval as lb
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        base = lb._baseline_actions(env, game) or {}
        out = [int(base[i]) for i in sorted(base)] if base else []
    except Exception:
        out = []
    _BASELINE_CACHE[game] = out
    return out


# --------------------------------------------------------------------------------------------
def preconditions_checked(paths: list[str]) -> list[dict]:
    """The resources this measurement depends on, each verified by a command whose result is
    recorded -- never asserted from memory.

    Two of these are witnessed BY THE SWEEP FILES THEMSELVES (they were checked at run time, per
    row, and the result is in the rows): that the swept parameter actually applied on every row, and
    that the pinned arm matched the live SUBMITTED_* globals. The rest are re-verified here, live.
    """
    out: list[dict] = []

    def _chk(resource: str, fn, principle: str):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}:{exc}"
        out.append(
            {"resource": resource, "available": bool(ok), "detail": detail, "principle": principle}
        )

    def _scorer():
        from arc_agi.scorecard import EnvironmentScoreCalculator  # noqa: F401
        import arc_agi

        return True, f"arc_agi {getattr(arc_agi, '__version__', 'unknown')}"

    _chk(
        "authoritative arc_agi scorer importable",
        _scorer,
        "Every efficiency number in this artifact comes from the installed scorer -- the same "
        "package the competition gateway runs. Without it run_game silently falls back to 0.0 and "
        "every score would read as a tie.",
    )

    def _grace_param():
        sys.path.insert(0, str(REPO / "python"))
        import inspect

        from carnot.agentic.arc_competition_agent import StepwiseExplorer

        sig = inspect.signature(StepwiseExplorer.__init__)
        return "early_stop_grace" in sig.parameters, sorted(sig.parameters)[:0] or "present"

    _chk(
        "StepwiseExplorer implements early_stop_grace",
        _grace_param,
        "The mechanism must exist before it can be swept. If it did not, every treatment arm would "
        "silently equal the control and produce a clean, meaningless null.",
    )

    def _flag_wiring_state():
        """STALE-ASSERTION FIX (2026-08-08, live-agent-adversarial-review-2026-08-08.md, Gaps
        finding 1). This check used to require SUBMITTED_EARLY_STOP_GRACE to still equal None
        (true when this sweep first ran, 2026-07-26). On 2026-08-07 the operator set the
        constant to 400, but a wiring bug meant E3AgentPolicy never read or forwarded it -- the
        old check would have silently reported `available: False` for that reason alone, without
        saying why. Fixed 2026-08-08: E3AgentPolicy now accepts `early_stop_grace` and forwards
        it to StepwiseExplorer, and the value is pinned in SUBMITTED_AGENT_CONFIG. This check now
        confirms that wiring is genuinely in place, so a re-run of this analysis reads the
        mechanism's CURRENT state instead of an assumption baked in when the sweep was written.
        """
        sys.path.insert(0, str(REPO / "python"))
        import inspect

        from carnot.agentic import arc_competition_agent as comp

        e3_param = inspect.signature(comp.E3AgentPolicy.__init__).parameters.get("early_stop_grace")
        e3_default = None if e3_param is None else e3_param.default
        config_value = comp.SUBMITTED_AGENT_CONFIG.get("early_stop_grace")
        wired = (
            e3_param is not None
            and e3_default == comp.SUBMITTED_EARLY_STOP_GRACE
            and config_value == comp.SUBMITTED_EARLY_STOP_GRACE
        )
        return wired, (
            f"submitted_value={comp.SUBMITTED_EARLY_STOP_GRACE!r}, "
            f"e3_init_default={e3_default!r}, "
            f"submitted_agent_config_value={config_value!r}"
        )

    _chk(
        "SUBMITTED_EARLY_STOP_GRACE is wired through E3AgentPolicy (fixed 2026-08-08)",
        _flag_wiring_state,
        "The sweep driver (arc_scored_path_early_stop_sweep.py) sets the parameter directly on "
        "the constructed explorer instance, bypassing E3AgentPolicy entirely, so this analysis "
        "does not depend on the fix to read its own rows correctly. This check instead confirms "
        "the SHIPPED default now matches what the sweep measured as a deliberate change -- a "
        "silent regression back to dead code would otherwise go unnoticed by every other check "
        "in this file.",
    )

    def _rows_witness():
        applied, parity_drift, crashed, n = True, {}, 0, 0
        for p in paths:
            d = json.loads(Path(p).read_text())
            parity_drift.update(
                (d.get("flag_parity_vs_live_globals") or {}).get("pinned_vs_live_drift") or {}
            )
            for r in d.get("rows", []):
                n += 1
                applied = applied and bool(r.get("early_stop_grace_applied"))
                crashed += 0 if r.get("ran") else 1
        return (applied and not parity_drift and crashed == 0), (
            f"rows={n}, all_grace_applied={applied}, pinned_vs_live_drift={parity_drift}, "
            f"crashed_cells={crashed}"
        )

    _chk(
        "every row: swept parameter applied, arm == live shipped config, no crashed cell",
        _rows_witness,
        "Witnessed per row at RUN time, not asserted here. An arm whose parameter failed to apply, "
        "or whose pinned flags drifted from the live config, is not the measurement it claims.",
    )

    def _games():
        sys.path.insert(0, str(REPO / "python"))
        from carnot.agentic import arc_solver_kit as kit

        envs = kit.offline_arcade().get_environments()
        return len(envs) >= 25, f"offline_arcade envs={len(envs)}"

    _chk(
        "offline arcade exposes the 25 public games",
        _games,
        "The corpus must be the full public survey set for a corpus-level claim; a short corpus "
        "would silently narrow the denominator.",
    )
    return out


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        for r in d.get("rows", []):
            r = dict(r)
            r["_source"] = p
            rows.append(r)
    return rows


def measurement_wall_clock(paths: list[str]) -> dict:
    """Sum the TRUE wall clock of the live measurement, from each row file's own `elapsed_s`.

    WHY THIS EXISTS (2026-07-26 review finding). This analyser runs in ~8 seconds over rows that
    took hours to produce. Publishing only the analyser's `duration_s` next to a live-runtime
    substrate declaration understated the real measurement by three orders of magnitude, and a
    reader checking whether the run was plausibly real would have been looking at the wrong clock
    entirely -- the same class of misreading that `DURATION_TOO_SHORT` exists to catch, arrived at
    from the other direction.

    Prefers each file's `elapsed_s` (the driving process's own clock) over the sum of per-cell
    `wall_s`, because the driver's clock includes per-cell setup and env construction that the
    per-cell timer does not: measured 9126.0s by file elapsed vs 6861.1s by summed cell wall, a
    ~25% undercount. Falls back to the summed cell wall when a file predates `elapsed_s`, and says
    per-file which basis it used so the two are never silently mixed without disclosure.
    """
    per_file, total, fallbacks = [], 0.0, []
    for p in sorted(set(paths)):
        try:
            d = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError):
            per_file.append({"file": p, "elapsed_s": None, "basis": "unreadable"})
            continue
        rows = d.get("rows", []) if isinstance(d, dict) else []
        elapsed = d.get("elapsed_s") if isinstance(d, dict) else None
        basis = "file_elapsed_s"
        if elapsed is None:
            elapsed = sum(float(r.get("wall_s") or 0.0) for r in rows)
            basis = "summed_cell_wall_s_fallback"
            fallbacks.append(p)
        per_file.append(
            {"file": p, "n_rows": len(rows), "elapsed_s": round(float(elapsed), 1), "basis": basis}
        )
        total += float(elapsed)
    return {
        "total_s": round(total, 1),
        "total_h": round(total / 3600.0, 2),
        "n_cells": sum(f.get("n_rows") or 0 for f in per_file),
        "per_file": per_file,
        "files_using_fallback_basis": fallbacks,
        "all_files_report_their_own_elapsed": not fallbacks,
    }


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """De-duplicate ANALYSIS rows on the matched-comparison key, and report what was dropped.

    WHY THIS EXISTS. Cells are keyed (game, seed, budget, grace) and the matched comparison stores at
    most one row per key -- but the CORPUS-LEVEL aggregates (`control_cells`, the gap distribution,
    `mechanism_reach`, the score-concentration denominators) iterate the FLAT row list. So a second
    file re-running the SAME cells would double-count them in every denominator while the numerators
    stayed single-counted, silently halving percentages. Adding a follow-up arm file is the obvious
    thing to do -- this sweep did exactly that to probe the safe window -- so the trap is live.

    APPLIED ONLY TO THE ANALYSIS ROWS. The reproduction and contention row sets deliberately contain
    the SAME cells more than once (that is the point of a reproduction and of a 3-process contention
    control); those checks pair them by key themselves. De-duplicating their inputs here would have
    silently changed which row won and moved `concurrent_wall_s` -- a first version of this change did
    exactly that, because the report was a module global that the last `load_rows` call overwrote.
    Hence a separate, explicitly-called function taking and returning its own report.

    Duplicates that DISAGREE on an outcome field are reported separately: that is a determinism
    failure, not a bookkeeping detail, and it must never be resolved by last-write-wins in silence.
    """
    seen: dict[tuple, dict] = {}
    order: list[tuple] = []
    n_dupes, disagreeing = 0, []
    fields = ("levels", "actions", "efficiency", "level_up_actions", "early_stopped")
    for r in rows:
        k = (r.get("game"), r.get("seed"), r.get("budget"), r.get("early_stop_grace"))
        if k in seen:
            n_dupes += 1
            prev = seen[k]
            diff = {f: [prev.get(f), r.get(f)] for f in fields if prev.get(f) != r.get(f)}
            if diff:
                disagreeing.append(
                    {
                        "cell": [str(x) for x in k],
                        "sources": [prev.get("_source"), r.get("_source")],
                        "diff": diff,
                    }
                )
            continue  # keep the FIRST occurrence, deterministically
        seen[k] = r
        order.append(k)
    report = {
        "n_rows_in": len(rows),
        "n_rows_kept": len(order),
        "n_duplicate_rows_dropped": n_dupes,
        "n_duplicates_disagreeing_on_an_outcome": len(disagreeing),
        "duplicates_are_consistent": not disagreeing,
        "disagreeing_detail": disagreeing[:20],
        "scope": (
            "ANALYSIS rows only (--rows). The reproduction and contention row sets are NOT deduped: "
            "they contain repeated cells by design and their checks pair by key themselves."
        ),
        "principle": (
            "A duplicated cell would double-count in every corpus-level DENOMINATOR while the "
            "matched comparison kept one row, silently halving percentages. A duplicate that "
            "DISAGREES is a determinism failure and is reported, never resolved by last-write-wins."
        ),
    }
    return [seen[k] for k in order], report


def gkey(r) -> tuple:
    return (r.get("game"), r.get("seed"), r.get("budget"))


def garm(r):
    return r.get("early_stop_grace")


def instrumentation_census(rows: list[dict], dup_report: dict | None = None) -> dict:
    """PROVE EVERY FIELD POPULATES. An arm whose parameter silently failed to apply, or whose
    diagnostics are None, is an UNINSTRUMENTED arm -- and would report a clean, meaningless null."""
    req = [
        "levels",
        "actions",
        "efficiency",
        "level_up_actions",
        "actions_after_last_levelup",
        "n_resets",
        "n_frames",
        "early_stopped",
        "early_stop_grace_applied",
        "wall_s",
    ]
    none_counts = {f: sum(1 for r in rows if r.get(f) is None) for f in req}
    ran = [r for r in rows if r.get("ran")]
    per_arm_fired = collections.Counter()
    per_arm_n = collections.Counter()
    for r in ran:
        per_arm_n[garm(r)] += 1
        if r.get("early_stopped"):
            per_arm_fired[garm(r)] += 1
    arms = sorted((a for a in per_arm_n if a is not None), key=lambda x: x)
    # FLAG PARITY ACROSS ARMS. Every arm must pin the SAME eight gated flags, RESOLVED off the live
    # explorer -- an arm that pinned only a subset would inherit whatever the module globals said at
    # run time, so a later flip would silently redefine it. Read from the rows, not from the arm
    # dict, so a kwarg that failed to reach the explorer is caught too.
    flagsets = collections.Counter(
        json.dumps(r.get("gated_flags") or {}, sort_keys=True) for r in rows
    )
    # A NEVER-FIRING ARM HAS TWO DIFFERENT CAUSES AND THEY MUST NOT BE CONFLATED.
    #   INERT BY CONSTRUCTION -- the grace is >= the budget on every cell of that arm, so the window
    #     mathematically CANNOT close inside a loop that runs at most `budget` iterations. Such an
    #     arm is a deliberate control: it should be byte-identical to the control arm, and its
    #     equality WITH the control is a positive determinism check, not a defect.
    #   WIRING SUSPECT -- the grace is below the budget somewhere, so the window COULD have closed
    #     and did not. Given that SUBMITTED_EARLY_STOP_GRACE is dead code, this is the shape a
    #     failed wiring would take, and it must be called out rather than read as a clean null.
    min_budget_per_arm: dict = {}
    for r in ran:
        a_ = garm(r)
        b_ = int(r.get("budget") or 0)
        min_budget_per_arm[a_] = min(min_budget_per_arm.get(a_, b_), b_)
    # MEASURED REACHABILITY, not just the budget rule. `grace >= budget` is SUFFICIENT for inertness
    # but far from necessary: a grace BELOW the budget that still exceeds every control cell's
    # post-level-up tail also cannot fire, and the budget rule alone would mislabel it WIRING
    # SUSPECT -- which poisons the whole artifact's verdict (`every_firing_capable_arm_fired` gates
    # it). This bit us in the design of this very sweep: grace 380 at b400 is below the budget and
    # provably cannot fire on any measured cell, so putting it on the grid would have raised a false
    # alarm. The criterion below uses the control cells' tail in ACTIONS, which is a LOWER bound on
    # the tail in FRAMES (frames >= actions always), so "this arm could have fired" is only asserted
    # when it is certain -- errs toward inert, never toward a false wiring alarm.
    ctrl_by_budget: dict[int, list[dict]] = collections.defaultdict(list)
    for r in ran:
        if garm(r) is None:
            ctrl_by_budget[int(r.get("budget") or 0)].append(r)
    budgets_per_arm: dict = collections.defaultdict(set)
    for r in ran:
        budgets_per_arm[garm(r)].add(int(r.get("budget") or 0))

    def _could_fire(G: int) -> int:
        """How many CONTROL cells had a post-level-up tail long enough for grace G to close."""
        n = 0
        for b_ in budgets_per_arm.get(G, ()):  # only budgets this arm was actually run at
            for r in ctrl_by_budget.get(b_, ()):
                if (
                    int(r.get("levels") or 0) >= 1
                    and int(r.get("actions_after_last_levelup") or 0) > G
                ):
                    n += 1
        return n

    could_fire_cells = {a: _could_fire(a) for a in arms}
    inert = [
        a
        for a in arms
        if per_arm_fired[a] == 0 and (a >= min_budget_per_arm.get(a, 0) or could_fire_cells[a] == 0)
    ]
    wiring = [a for a in arms if per_arm_fired[a] == 0 and a not in inert]
    return {
        "n_rows": len(rows),
        "n_ran": len(ran),
        "n_crashed": len(rows) - len(ran),
        "none_valued_required_fields": none_counts,
        "zero_none_valued_diagnostics": all(v == 0 for v in none_counts.values()),
        "grace_applied_all_rows": all(r.get("early_stop_grace_applied") for r in rows),
        "early_stopped_cells_per_arm": {str(a): int(per_arm_fired[a]) for a in per_arm_n},
        "cells_per_arm": {str(a): int(per_arm_n[a]) for a in per_arm_n},
        "every_treatment_arm_fired_somewhere": all(per_arm_fired[a] > 0 for a in arms),
        "control_never_fired": per_arm_fired[None] == 0,
        "uninstrumented_arms": [str(a) for a in arms if per_arm_fired[a] == 0],
        "inert_by_construction_arms": [str(a) for a in inert],
        "uninstrumented_arms_wiring_suspect": [str(a) for a in wiring],
        "every_firing_capable_arm_fired": len(wiring) == 0,
        "min_budget_per_arm": {str(k): v for k, v in min_budget_per_arm.items()},
        "control_cells_whose_tail_could_close_this_grace": {
            str(k): v for k, v in could_fire_cells.items()
        },
        "inertness_criterion_note": (
            "An arm is INERT BY CONSTRUCTION if grace >= budget (it cannot close inside a loop of at "
            "most `budget` iterations) OR if no control cell at that arm's budget had a "
            "post-level-up tail longer than the grace. The second clause matters: a grace BELOW the "
            "budget can still be unfireable, and calling it WIRING SUSPECT would falsely stamp the "
            "whole artifact uninterpretable. Tail is measured in ACTIONS, a lower bound on the "
            "FRAMES the window counts, so 'could have fired' is asserted only when certain."
        ),
        "row_duplication": dict(dup_report or {}),
        "distinct_resolved_gated_flag_sets": len(flagsets),
        "all_arms_pin_the_same_eight_gated_flags": len(flagsets) == 1,
        "resolved_gated_flags": (json.loads(next(iter(flagsets))) if len(flagsets) == 1 else None),
        "resolved_gated_flag_set_counts": (
            {k: v for k, v in flagsets.items()} if len(flagsets) != 1 else None
        ),
        "n_gated_flags_resolved": (
            len(json.loads(next(iter(flagsets)))) if len(flagsets) == 1 else None
        ),
        "principle": (
            "An arm that never fired contributes NO evidence about the mechanism. Because "
            "SUBMITTED_EARLY_STOP_GRACE is dead code (read nowhere), a wiring mistake would have "
            "produced exactly such an arm and it would have looked like a clean null."
        ),
    }


def analyse_condition(rows: list[dict], budget: int) -> dict:
    rows = [r for r in rows if r.get("budget") == budget and r.get("ran")]
    by_cell: dict[tuple, dict] = collections.defaultdict(dict)
    for r in rows:
        by_cell[gkey(r)][garm(r)] = r
    arms = sorted({garm(r) for r in rows if garm(r) is not None})

    # PER-SEED MATCHED cells only: a cell missing either the control or the arm is dropped, and the
    # drop is reported. Never compare an arm's cells against a control union over other seeds.
    per_arm: dict[str, dict] = {}
    for G in arms:
        matched = [(k, c[None], c[G]) for k, c in by_cell.items() if None in c and G in c]
        unmatched = [k for k, c in by_cell.items() if (None in c) != (G in c)]

        # ---- G-SAFETY ------------------------------------------------------------------
        # MOVABLE: control reached >= 2 levels. A control with <= 1 level cannot regress -- the
        # window only arms after the FIRST level-up and cannot undo it.
        movable = [(k, c, t) for (k, c, t) in matched if int(c.get("levels") or 0) >= 2]
        # AT RISK: of the movable, those whose measured inter-level-up gap exceeds the grace. The
        # window counts FRAMES (RESETs included) while the gap is in ACTIONS, and frames >= actions,
        # so the gap is inflated by the cell's own frames/actions ratio -- the conservative
        # direction (it counts MORE cells as at-risk, never fewer).
        at_risk = []
        for k, c, t in movable:
            gaps = list(c.get("inter_levelup_gaps") or [])
            infl = (c.get("n_frames") or 0) / max(1, int(c.get("actions") or 1))
            if any(g * max(1.0, infl) > G for g in gaps):
                at_risk.append((k, c, t))
        regressed = [
            {
                "cell": list(k),
                "control_levels": int(c.get("levels") or 0),
                "arm_levels": int(t.get("levels") or 0),
                "control_level_up_actions": c.get("level_up_actions"),
                "arm_level_up_actions": t.get("level_up_actions"),
                "control_inter_levelup_gaps": c.get("inter_levelup_gaps"),
                # WHY IT REGRESSED, in the window's OWN UNIT. The window counts FRAMES; the gaps are
                # in ACTIONS. Without this a regression at a grace ABOVE the largest measured
                # action-gap looks inexplicable -- which is exactly what happens: r11l seed
                # 20260724 at b4000 has a 2775-ACTION gap, and grace 2800 still lost the level,
                # because 219 resets inflate that gap to ~2936 FRAMES. Reported per cell so the
                # cause is visible rather than guessed at.
                "control_frames_per_action": round(
                    (c.get("n_frames") or 0) / max(1, int(c.get("actions") or 1)), 4
                ),
                "control_gaps_estimated_in_frames": [
                    round(g * (c.get("n_frames") or 0) / max(1, int(c.get("actions") or 1)), 1)
                    for g in (c.get("inter_levelup_gaps") or [])
                ],
                "grace_frames": G,
                "control_n_resets": c.get("n_resets"),
            }
            for (k, c, t) in matched
            if int(t.get("levels") or 0) < int(c.get("levels") or 0)
        ]
        improved = [
            list(k)
            for (k, c, t) in matched
            if int(t.get("levels") or 0) > int(c.get("levels") or 0)
        ]
        safety_interpretable = len(movable) > 0
        # FALSIFIABILITY OF A **PASS**, which is a different question from interpretability of a
        # FAIL. `movable > 0` is the right guard for a FAIL (some cell had to be ABLE to regress),
        # but it is the WRONG guard for a PASS: for "this grace is SAFE" to have been falsifiable,
        # at least one cell must have been AT RISK -- movable AND holding an inter-level-up gap the
        # grace could have cut. With at_risk == 0 the pass is arithmetically forced: no cell could
        # have regressed, so "did not regress" adds nothing to "grace exceeds every observed gap".
        # In this corpus at_risk is a PERFECT predictor of regressions (b2000: 5->5, 4->4, 4->4,
        # 0->0, 0->0; b4000: 10->10, 6->6, 1->1), which is the mechanism, not a coincidence.
        # Empirically confirmed fragile: grace 1300 passes at b2000 with at_risk == 0 and REGRESSES
        # 6 cells at b4000, where the gap distribution is wider.
        safety_falsifiable = len(at_risk) > 0
        safety = {
            "passed": len(regressed) == 0,
            "interpretable": safety_interpretable,
            # A PASS is FALSIFIABLE only if some cell was at risk. Reported separately from
            # `passed` so a forced pass cannot be read as evidence.
            "falsifiable": safety_falsifiable,
            "verdict": (
                "FAIL_LEVEL_REGRESSION"
                if regressed
                else (
                    "UNINTERPRETABLE_EMPTY_MOVABLE_SET_NO_CELL_COULD_REGRESS"
                    if not safety_interpretable
                    else (
                        "PASS_AT_RISK_CELLS_SURVIVED"
                        if safety_falsifiable
                        else "PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK"
                    )
                )
            ),
            "safety_law": (
                "Safety of a FIXED grace reduces to `grace > the largest inter-level-up gap in the "
                "sample, measured in FRAMES`. It is therefore a property of the GAP DISTRIBUTION, "
                "not an independent empirical finding -- and the gap distribution grows with the "
                "action budget (max at-risk gap ~340 frames at b400, under 1300 at b2000, over 2800 "
                "at b4000) and varies by orders of magnitude across (game, seed). NO FIXED GRACE IS "
                "SAFE OUT OF SAMPLE. An adaptive window (scaled to the run's own observed gaps) is "
                "the only form that could be, and it is UNMEASURED here."
            ),
            "n_matched_cells": len(matched),
            "witness_movable_cells": len(movable),
            "witness_movable_games": sorted({k[0] for k, _, _ in movable}),
            "witness_at_risk_cells": len(at_risk),
            "witness_at_risk_games": sorted({k[0] for k, _, _ in at_risk}),
            # THE STATISTICAL POWER OF THIS GATE, at the gate's own support. A safety verdict decided
            # by 2 at-risk cells cannot reach p<=0.05 no matter what those cells did: the smallest
            # two-sided p any outcome could produce at n=2 is 0.5. Reporting the verdict without this
            # invites reading a single-observation existence claim as an estimate.
            "witness_support_power": {
                "n_at_risk_cells": len(at_risk),
                "n_at_risk_games": len(sorted({k[0] for k, _, _ in at_risk})),
                "p_min_reachable_two_sided_at_this_support": (
                    round(min(1.0, 2.0 * (0.5 ** len(at_risk))), 6) if at_risk else 1.0
                ),
                "any_p_below_0.05_reachable": bool(min(1.0, 2.0 * (0.5 ** len(at_risk))) <= 0.05)
                if at_risk
                else False,
                "decided_by_n_games": len(sorted({k[0] for k, _, _ in movable})),
                "principle": (
                    "The support is the AT-RISK set, not the matched set: only an at-risk cell can "
                    "carry information about safety. If no p at or below 0.05 was reachable, the "
                    "verdict is an EXISTENCE claim over a handful of observations, and adding seeds "
                    "-- not re-reading these -- is the only way to strengthen it."
                ),
            },
            "regressing_cells": regressed,
            "regressing_games": sorted({r["cell"][0] for r in regressed}),
            "improving_cells": improved,
            "witness_principle": (
                "MOVABLE = control reached >=2 levels (only such a cell HAS a later level-up to "
                "forgo). AT RISK = movable AND a measured inter-level-up gap, inflated to frames, "
                "exceeds the grace. An empty movable set makes the gate arithmetically forced."
            ),
            "witness_at_risk_approximation_note": (
                "STATED LIMITATION. The window is counted in FRAMES (loop iterations, RESETs "
                "included); the persisted gaps are in ACTIONS. The at-risk test therefore scales "
                "each gap by the cell's own whole-run frames/actions ratio rather than by the "
                "per-level ratio, which is an APPROXIMATION -- resets are not uniformly "
                "distributed across a run. It errs toward counting MORE cells at risk (frames >= "
                "actions always), so it cannot make an at-risk cell look safe. The AT-RISK count "
                "is an advisory witness; the SAFETY VERDICT itself is decided by the measured "
                "levels, not by this estimate."
            ),
        }

        # ---- G-NONINF (authoritative charge model) --------------------------------------
        eff_deltas, eff_moved = [], []
        for k, c, t in matched:
            d = float(t.get("efficiency") or 0.0) - float(c.get("efficiency") or 0.0)
            eff_deltas.append(d)
            if list(c.get("level_up_actions") or []) != list(t.get("level_up_actions") or []):
                eff_moved.append(
                    {
                        "cell": list(k),
                        "control_lua": c.get("level_up_actions"),
                        "arm_lua": t.get("level_up_actions"),
                        "control_eff": c.get("efficiency"),
                        "arm_eff": t.get("efficiency"),
                        "delta": round(d, 6),
                    }
                )
        ctrl_sum = sum(float(c.get("efficiency") or 0.0) for _, c, _ in matched)
        arm_sum = sum(float(t.get("efficiency") or 0.0) for _, _, t in matched)
        noninf = {
            "passed": arm_sum >= ctrl_sum - 1e-9,
            "interpretable": len(eff_moved) > 0,
            "verdict": (
                "PASS_NON_INFERIOR"
                if arm_sum >= ctrl_sum - 1e-9
                else "FAIL_SCORE_LOSS"
                if eff_moved
                else "FAIL_SCORE_LOSS"
            ),
            "control_efficiency_sum": round(ctrl_sum, 6),
            "arm_efficiency_sum": round(arm_sum, 6),
            "delta_sum": round(arm_sum - ctrl_sum, 6),
            "relative_delta": (round((arm_sum - ctrl_sum) / ctrl_sum, 6) if ctrl_sum else None),
            "witness_cells_whose_levelup_vector_moved": len(eff_moved),
            "witness_moved_detail": eff_moved[:40],
            "witness_principle": (
                "The per-game score is a function of the level-up CHECKPOINT VECTOR and the human "
                "baselines alone. A cell whose vector is identical between arms is FROZEN by "
                "construction and its equal efficiency is not evidence either way. If NO cell's "
                "vector moved, the score comparison is structurally forced to a tie."
            ),
            "sign_test": sign_test_both_tails(eff_deltas),
        }
        if not eff_moved:
            noninf["verdict"] = "PASS_STRUCTURALLY_FROZEN_NO_CELL_COULD_MOVE"

        # ---- BENEFIT --------------------------------------------------------------------
        act_d = [int(t.get("actions") or 0) - int(c.get("actions") or 0) for _, c, t in matched]
        tail_d = [
            int(t.get("actions_after_last_levelup") or 0)
            - int(c.get("actions_after_last_levelup") or 0)
            for _, c, t in matched
        ]
        wall_d = [
            float(t.get("wall_s") or 0.0) - float(c.get("wall_s") or 0.0) for _, c, t in matched
        ]
        ctrl_act = sum(int(c.get("actions") or 0) for _, c, _ in matched)
        arm_act = sum(int(t.get("actions") or 0) for _, _, t in matched)
        ctrl_wall = sum(float(c.get("wall_s") or 0.0) for _, c, _ in matched)
        arm_wall = sum(float(t.get("wall_s") or 0.0) for _, _, t in matched)
        ctrl_tail = sum(int(c.get("actions_after_last_levelup") or 0) for _, c, _ in matched)
        arm_tail = sum(int(t.get("actions_after_last_levelup") or 0) for _, _, t in matched)
        # Restricted to cells that did NOT regress -- a saving bought by losing a level is not a
        # saving worth reporting as a benefit.
        nonreg = [
            (k, c, t)
            for (k, c, t) in matched
            if int(t.get("levels") or 0) >= int(c.get("levels") or 0)
        ]
        benefit = {
            "total_actions_control": ctrl_act,
            "total_actions_arm": arm_act,
            "total_actions_saved": ctrl_act - arm_act,
            "total_actions_saved_pct": (
                round(100.0 * (ctrl_act - arm_act) / ctrl_act, 3) if ctrl_act else None
            ),
            "tail_actions_control": ctrl_tail,
            "tail_actions_arm": arm_tail,
            "tail_actions_saved": ctrl_tail - arm_tail,
            "tail_actions_saved_pct": (
                round(100.0 * (ctrl_tail - arm_tail) / ctrl_tail, 3) if ctrl_tail else None
            ),
            "total_wall_s_control": round(ctrl_wall, 2),
            "total_wall_s_arm": round(arm_wall, 2),
            "total_wall_s_saved": round(ctrl_wall - arm_wall, 2),
            "total_wall_s_saved_pct": (
                round(100.0 * (ctrl_wall - arm_wall) / ctrl_wall, 3) if ctrl_wall else None
            ),
            "median_actions_delta": round(statistics.median(act_d), 2) if act_d else None,
            "median_tail_delta": round(statistics.median(tail_d), 2) if tail_d else None,
            "median_wall_delta_s": round(statistics.median(wall_d), 3) if wall_d else None,
            "n_nonregressing_cells": len(nonreg),
            "actions_saved_on_nonregressing_cells": sum(
                int(c.get("actions") or 0) - int(t.get("actions") or 0) for _, c, t in nonreg
            ),
            "wall_s_saved_on_nonregressing_cells": round(
                sum(
                    float(c.get("wall_s") or 0.0) - float(t.get("wall_s") or 0.0)
                    for _, c, t in nonreg
                ),
                2,
            ),
            # IS THE WALL SAVING REAL, MEASURED WITHOUT A CHOSEN THRESHOLD? Wall clock is the axis
            # this mechanism's whole case rests on and the noisiest thing here, so the grace-350 arm
            # at b400 is a live trap: 21 actions saved (0.072%) against a MEASURED 2.42% wall saving.
            #
            # The decisive test needs no arbitrary constant, because this design supplies its own
            # control. On any cell where the two arms took the IDENTICAL number of actions, they did
            # byte-identical work (the run is a deterministic function of the seed -- verified by the
            # reproduction check), so ANY wall difference on those cells is measurement jitter with a
            # TRUE value of exactly zero. Summing it gives the jitter present in this very
            # comparison, at the same aggregation as the headline. For grace 350: 74 of 75 cells did
            # identical work and contribute +3.19s of spurious "saving", while the ONE cell that
            # actually stopped early contributes 0.89s -- so 78% of the reported 4.08s is noise, and
            # the arm is NOISE-DOMINATED on measurement rather than on a threshold.
            #
            # The seconds-per-action ratio is kept as a secondary, cruder signal and labelled as the
            # heuristic it is (its 3x cut-off is a choice; the jitter comparison is not).
            "wall_saving_attribution": (
                lambda rate, jit, real: {
                    "control_seconds_per_action": round(rate, 6),
                    "actions_saved": ctrl_act - arm_act,
                    "wall_s_measured_saved": round(ctrl_wall - arm_wall, 3),
                    # -- the threshold-free test --
                    "n_cells_doing_identical_work": sum(
                        1
                        for _, c2, t2 in matched
                        if int(c2.get("actions") or 0) == int(t2.get("actions") or 0)
                    ),
                    "wall_s_jitter_on_identical_work_cells": round(jit, 3),
                    "wall_s_saved_on_cells_whose_actions_changed": round(real, 3),
                    "jitter_share_of_measured_saving_pct": (
                        round(100.0 * jit / (ctrl_wall - arm_wall), 1)
                        if abs(ctrl_wall - arm_wall) > 1e-9
                        else None
                    ),
                    "noise_dominated": bool(abs(jit) >= abs(real)),
                    # -- the secondary heuristic, explicitly labelled --
                    "wall_s_attributable_to_actions_saved_heuristic": round(
                        (ctrl_act - arm_act) * rate, 3
                    ),
                    "measured_over_attributable_ratio_heuristic": (
                        round((ctrl_wall - arm_wall) / ((ctrl_act - arm_act) * rate), 2)
                        if (ctrl_act - arm_act) * rate > 1e-9
                        else None
                    ),
                    "principle": (
                        "`noise_dominated` is decided WITHOUT a chosen threshold: cells where both "
                        "arms took the same number of actions did identical work, so their summed "
                        "wall difference IS this comparison's jitter, with a true value of zero. If "
                        "that jitter is as large as the saving on the cells that actually changed, "
                        "the wall figure is not a benefit. The seconds-per-action ratio beside it is "
                        "a cruder heuristic (its 3x cut-off would be a judgement call) and is "
                        "labelled as such; `inert_arm_noise_floor` is the third, independent reading."
                    ),
                }
            )(
                ctrl_wall / max(1, ctrl_act),
                sum(
                    float(c2.get("wall_s") or 0.0) - float(t2.get("wall_s") or 0.0)
                    for _, c2, t2 in matched
                    if int(c2.get("actions") or 0) == int(t2.get("actions") or 0)
                ),
                sum(
                    float(c2.get("wall_s") or 0.0) - float(t2.get("wall_s") or 0.0)
                    for _, c2, t2 in matched
                    if int(c2.get("actions") or 0) != int(t2.get("actions") or 0)
                ),
            ),
            "sign_test_actions": sign_test_both_tails(act_d),
            "sign_test_wall": sign_test_both_tails(wall_d),
        }

        # ---- SENSITIVITY: the refuted total-action charge model --------------------------
        # WITH A LIVE CROSS-CHECK, because a sensitivity model computed from missing baselines
        # silently returns 0.0 -> 0.0, which reads as "the alternative shows no difference". The
        # guard: recompute the AUTHORITATIVE score from the same inputs and compare it against the
        # row's own `efficiency` (produced by the installed scorer inside run_game). If the
        # reimplementation matches on every cell, the baselines and the checkpoint vectors are both
        # right, and the sensitivity number computed beside it is trustworthy. If it does not, the
        # sensitivity number is stamped untrustworthy rather than reported.
        sens_c, sens_a, sens_d = 0.0, 0.0, []
        xcheck_n, xcheck_bad, cells_with_baselines = 0, [], 0
        for k, c, t in matched:
            b = baselines_for(k[0])
            if b:
                cells_with_baselines += 1
            for side in (c, t):
                recomputed = score_actions_to_level(
                    list(side.get("level_up_actions") or []), int(side.get("actions") or 0), b
                )
                xcheck_n += 1
                if abs(recomputed - float(side.get("efficiency") or 0.0)) > 5e-4:
                    xcheck_bad.append(
                        {
                            "cell": list(k),
                            "grace": side.get("early_stop_grace"),
                            "row_efficiency": side.get("efficiency"),
                            "recomputed": recomputed,
                        }
                    )
            sc = score_total_action_charge(
                list(c.get("level_up_actions") or []), int(c.get("actions") or 0), b
            )
            sa = score_total_action_charge(
                list(t.get("level_up_actions") or []), int(t.get("actions") or 0), b
            )
            sens_c += sc
            sens_a += sa
            sens_d.append(sa - sc)
        sensitivity = {
            "authoritative_reimplementation_crosscheck": {
                "n_cells_checked": xcheck_n,
                "n_mismatched": len(xcheck_bad),
                "matches_installed_scorer_on_every_cell": len(xcheck_bad) == 0,
                "mismatches": xcheck_bad[:20],
                "cells_with_nonempty_human_baselines": cells_with_baselines,
                "n_matched_cells": len(matched),
                "principle": (
                    "The reimplemented authoritative model is compared against the row's own "
                    "`efficiency`, which came from the INSTALLED scorer. Agreement on every cell is "
                    "the evidence that the persisted checkpoint vector and the human baselines are "
                    "both correct -- and therefore that the sensitivity model computed from the "
                    "same inputs means something. Zero cells with baselines would make both 0.0, "
                    "which is a DEAD diagnostic, not a null result."
                ),
            },
            "trustworthy": len(xcheck_bad) == 0 and cells_with_baselines > 0,
            "model": "total_action_charge (REFUTED by arc_agi/scorecard.py:479; sensitivity only)",
            "control_sum": round(sens_c, 6),
            "arm_sum": round(sens_a, 6),
            "delta_sum": round(sens_a - sens_c, 6),
            "relative_delta": (round((sens_a - sens_c) / sens_c, 6) if sens_c else None),
            "sign_test": sign_test_both_tails(sens_d),
            "note": (
                "Under the refuted reading the tail IS charged, so early-stop looks strongly "
                "positive. Reported to show the SAFETY verdict does not depend on the choice of "
                "charge model -- only the size and sign of the SCORE benefit does."
            ),
        }

        per_arm[str(G)] = {
            "grace": G,
            "n_matched_cells": len(matched),
            "n_unmatched_cells_dropped": len(unmatched),
            "unmatched_cells": [list(k) for k in unmatched],
            "gate_safety": safety,
            "gate_score_non_inferiority_authoritative": noninf,
            "benefit_actions_and_wall": benefit,
            "sensitivity_refuted_total_action_charge": sensitivity,
            "n_cells_early_stopped": sum(1 for _, _, t in matched if t.get("early_stopped")),
        }

    # ---- cross-arm: the corpus-level gap distribution the safety gate rests on -----------
    ctrl_rows = [r for r in rows if garm(r) is None]
    gaps = []
    for r in ctrl_rows:
        for g in r.get("inter_levelup_gaps") or []:
            gaps.append({"game": r["game"], "seed": r["seed"], "gap_actions": int(g)})
    gvals = sorted(x["gap_actions"] for x in gaps)

    # ---- THE SAFE-AND-FIRING WINDOW, computed rather than assumed to be spanned by the grid ------
    # WHY THIS BLOCK EXISTS (adversarial review, 2026-07-26). The decision gate below asks "is there
    # a firing grace value that is safe?" and the tested grid answered "no" at b400. But a gate over
    # a GRID can only ever support "none of the values TESTED", and the untested region was where the
    # answer lived: the safe region begins just above the largest at-risk gap and firing stops just
    # above the largest post-level-up tail, and at b400 the tested grid (50/100/150/200/400) jumped
    # straight over the resulting window. Worse, the top grid point EQUALLED the budget, so it was
    # inert BY CONSTRUCTION and wasted the one arm that could have probed the safe region. Both
    # bounds are computable from the control arm alone, so the grid's adequacy is now REPORTED, and
    # the missing point (grace 350 at b400) was subsequently RUN rather than extrapolated.
    def _tail_frames(r) -> float:
        infl = (r.get("n_frames") or 0) / max(1, int(r.get("actions") or 1))
        return int(r.get("actions_after_last_levelup") or 0) * max(1.0, infl)

    def _gap_frames(r) -> list[float]:
        infl = (r.get("n_frames") or 0) / max(1, int(r.get("actions") or 1))
        return [g * max(1.0, infl) for g in (r.get("inter_levelup_gaps") or [])]

    movable_ctrl = [r for r in ctrl_rows if int(r.get("levels") or 0) >= 2]
    all_gap_frames = [(v, r) for r in movable_ctrl for v in _gap_frames(r)]
    lower = max((v for v, _ in all_gap_frames), default=0.0)  # safe iff grace > this
    lower_cell = max(all_gap_frames, key=lambda t: t[0])[1] if all_gap_frames else None
    tails = [(_tail_frames(r), r) for r in ctrl_rows if int(r.get("levels") or 0) >= 1]
    upper = max((v for v, _ in tails), default=0.0)  # fires somewhere iff grace < this
    upper_cell = max(tails, key=lambda t: t[0])[1] if tails else None
    tested = sorted(a for a in per_arm if a not in (None, "None"))
    tested_int = sorted(int(a) for a in tested)
    inside = [g for g in tested_int if lower < g < upper]
    safe_firing_window = {
        "lower_bound_frames_exclusive": round(lower, 1),
        "lower_bound_is_the_largest_at_risk_gap_on_cell": (
            [lower_cell.get("game"), lower_cell.get("seed")] if lower_cell is not None else None
        ),
        "upper_bound_frames_exclusive": round(upper, 1),
        "upper_bound_is_the_largest_post_levelup_tail_on_cell": (
            [upper_cell.get("game"), upper_cell.get("seed")] if upper_cell is not None else None
        ),
        "window_is_non_empty": bool(upper > lower),
        "tested_graces": tested_int,
        "tested_graces_inside_the_window": inside,
        "grid_spans_the_window": bool(inside),
        "inert_by_construction_tested_graces": [g for g in tested_int if g >= upper],
        "unsafe_by_construction_tested_graces": [g for g in tested_int if g <= lower],
        "principle": (
            "A grace is SAFE in-sample iff it exceeds every at-risk inter-level-up gap (in FRAMES) "
            "and FIRES somewhere iff it is below the largest post-level-up tail (in FRAMES). A "
            "grid that contains no value strictly between those bounds cannot answer 'is a safe "
            "firing value available' -- only 'none of the values tested is'. Both bounds come from "
            "the CONTROL arm, so grid adequacy is checkable before any treatment arm is run."
        ),
        "frames_vs_actions_note": (
            "Both bounds scale ACTION-unit measurements by each cell's own whole-run frames/actions "
            "ratio, because the window counts frames and the persisted gaps/tails are in actions. "
            "That is an approximation (resets are not uniformly distributed within a run); it errs "
            "toward a WIDER unsafe region and a WIDER firing region, i.e. toward a NARROWER claimed "
            "safe-and-firing window. The window's members are still verified BY RUNNING them."
        ),
    }

    # ---- SCORE CONCENTRATION: the denominator behind every score delta in this condition ---------
    # WHY (adversarial review, 2026-07-26). The reported score deltas are SUMS over cells, and at
    # b400 four of 75 cells carry ~91% of the corpus efficiency total, one of them alone over half.
    # A delta framed as a corpus effect is, at that concentration, a statement about two or three
    # cells. Any future arm that perturbs the top cell will swamp the sum by itself, so this is a
    # STANDING diagnostic, not a one-off residual.
    eff_sorted = sorted(
        ((float(r.get("efficiency") or 0.0), r.get("game"), r.get("seed")) for r in ctrl_rows),
        reverse=True,
    )
    eff_total = sum(v for v, _, _ in eff_sorted)
    score_concentration = {
        "control_efficiency_sum": round(eff_total, 6),
        "n_control_cells": len(ctrl_rows),
        "n_cells_with_nonzero_score": sum(1 for v, _, _ in eff_sorted if v > 0),
        "top_1_share_pct": (round(100.0 * eff_sorted[0][0] / eff_total, 2) if eff_total else None),
        "top_4_share_pct": (
            round(100.0 * sum(v for v, _, _ in eff_sorted[:4]) / eff_total, 2)
            if eff_total
            else None
        ),
        "top_cells": [
            {"game": g, "seed": s, "efficiency": round(v, 6)} for v, g, s in eff_sorted[:6] if v > 0
        ],
        "max_single_cell_efficiency": (round(eff_sorted[0][0], 6) if eff_sorted else None),
        "principle": (
            "Score deltas are SUMS. At high concentration a 'corpus' delta is a statement about the "
            "handful of cells that hold the mass. Reported at the same aggregation level as the "
            "delta itself so the two cannot be read apart."
        ),
        "note_max_exceeds_one": (
            "A single cell's authoritative score can exceed 1.0 (the maximum here is "
            f"{round(eff_sorted[0][0], 4) if eff_sorted else None}); the per-game score is on a "
            "0-100 scale with a per-level cap of 115 clamped to the all-scoring-levels ceiling. This "
            "is direct empirical refutation of the retracted `min(human/agent,1)^2` paraphrase, "
            "which cannot exceed 1."
        ),
    }
    games_covered = sorted({r.get("game") for r in rows})
    return {
        "budget": budget,
        "games_covered": games_covered,
        "n_games_covered": len(games_covered),
        "is_full_25_game_corpus": len(games_covered) == 25,
        "score_sums_are_corpus_level": len(games_covered) == 25,
        "scope_note": (
            "FULL 25-game corpus: score sums are corpus-level."
            if len(games_covered) == 25
            else (
                "SUBSET condition, scoped to level-reaching games so the SAFETY gate has the "
                "largest available movable set. Its score SUMS are NOT corpus-level and must NOT "
                "be compared against a full-corpus condition's sums -- the subset was chosen for "
                "its level-reaching. Read this condition for the SAFETY verdict and the gap "
                "distribution; read the full-corpus conditions for score and corpus benefit."
            )
        ),
        "arms": per_arm,
        "safe_firing_window": safe_firing_window,
        "score_concentration": score_concentration,
        "control_cells": len(ctrl_rows),
        "control_cells_reaching_2plus_levels": sum(
            1 for r in ctrl_rows if int(r.get("levels") or 0) >= 2
        ),
        "control_levels_distribution": dict(
            collections.Counter(int(r.get("levels") or 0) for r in ctrl_rows)
        ),
        # THE MECHANISM'S STRUCTURAL CEILING, which is easy to overstate by looking only at won
        # cells. `is_done` arms the window ONLY after the first level-up, so a cell that never
        # levels up is UNREACHABLE by early-stop no matter how long its tail is -- and those cells
        # hold most of the corpus's actions. Reporting the tail fraction of WON cells (a large
        # number) as the available saving is the mistake this block exists to prevent.
        "mechanism_reach": {
            "control_cells": len(ctrl_rows),
            "cells_where_window_can_arm_levels_ge_1": sum(
                1 for r in ctrl_rows if int(r.get("levels") or 0) >= 1
            ),
            "cells_unreachable_levels_eq_0": sum(
                1 for r in ctrl_rows if int(r.get("levels") or 0) == 0
            ),
            "reachable_fraction_of_cells": (
                round(
                    sum(1 for r in ctrl_rows if int(r.get("levels") or 0) >= 1) / len(ctrl_rows), 4
                )
                if ctrl_rows
                else None
            ),
            "control_actions_total": sum(int(r.get("actions") or 0) for r in ctrl_rows),
            "control_actions_on_reachable_cells": sum(
                int(r.get("actions") or 0) for r in ctrl_rows if int(r.get("levels") or 0) >= 1
            ),
            "reachable_fraction_of_actions": (
                round(
                    sum(
                        int(r.get("actions") or 0)
                        for r in ctrl_rows
                        if int(r.get("levels") or 0) >= 1
                    )
                    / max(1, sum(int(r.get("actions") or 0) for r in ctrl_rows)),
                    4,
                )
            ),
            "principle": (
                "An upper bound on what this mechanism can ever save, computed from the CONTROL "
                "arm. The window arms only after the first level-up, so the action budget spent on "
                "never-levelling cells is out of its reach entirely. Any saving claim above "
                "`reachable_fraction_of_actions` is impossible by construction."
            ),
        },
        "inter_levelup_gap_distribution_actions": {
            "n": len(gvals),
            "values": gvals,
            "min": gvals[0] if gvals else None,
            "median": statistics.median(gvals) if gvals else None,
            "max": gvals[-1] if gvals else None,
            "by_cell": gaps,
            "principle": (
                "This is the quantity that decides whether a grace value can cost a level. It is "
                "measured on the CONTROL arm at this budget, not imported from another condition."
            ),
        },
    }


def reproduction_check(main_rows: list[dict], repro_rows: list[dict]) -> dict:
    """ROUND-ROBIN REPRODUCTION. LLM-off should be a deterministic function of the seed; this
    re-runs a whole seed in a FRESH process and diffs every matched cell rather than assuming it."""
    key = lambda r: (r.get("game"), r.get("seed"), r.get("budget"), r.get("early_stop_grace"))  # noqa: E731
    a = {key(r): r for r in main_rows if r.get("ran")}
    b = {key(r): r for r in repro_rows if r.get("ran")}
    shared = sorted(set(a) & set(b), key=lambda t: tuple(str(x) for x in t))
    fields = ["levels", "actions", "efficiency", "level_up_actions", "early_stopped", "n_resets"]
    mismatches = []
    for k in shared:
        diff = {f: [a[k].get(f), b[k].get(f)] for f in fields if a[k].get(f) != b[k].get(f)}
        if diff:
            mismatches.append({"cell": [str(x) for x in k], "diff": diff})
    return {
        "n_cells_compared": len(shared),
        "n_mismatched": len(mismatches),
        "deterministic": len(mismatches) == 0,
        "fields_compared": fields,
        "mismatches": mismatches[:40],
        "principle": (
            "A sweep whose cells are not reproducible cannot support a per-cell matched comparison: "
            "an arm delta would be indistinguishable from run-to-run variation."
        ),
    }


def contention_check(serial_rows: list[dict], conc_rows: list[dict]) -> dict:
    """CONTENTION CONTROL -- measured, not assumed away. Same cells, run serially and then with N
    concurrent processes. Wall clock is expected to inflate; OUTCOMES must be identical.

    DEFECT FIXED 2026-07-26 (found while reconciling a 0.4% discrepancy between the changelog's
    reported 80.1s and a rebuild's 79.76s). The N concurrent processes each ran the SAME 30 cells, so
    `conc_rows` holds N rows per cell. The old implementation collapsed them with
    `{key(r): r for r in conc_rows}` -- LAST WRITE WINS -- which meant:

      1. `concurrent_wall_s` reported ONE ARBITRARY PROCESS and silently discarded the other two.
         The three processes measured 80.08 / 81.10 / 79.76 s, so the published figure and its
         inflation factor (1.547 / 1.553 / 1.573) depended entirely on the ORDER the files were
         listed on the command line. That is an arbitrary selection wearing the clothes of a
         measurement -- and it is what made the changelog and the rebuild disagree.
      2. Worse, `outcomes_identical: true` was checked against that ONE surviving process only. The
         other two processes' outcomes were never compared to serial at all, so the claim covered
         1 of 3 processes while reading as though it covered all of them. (Re-checked properly:
         all three are in fact identical to serial, 0 mismatches each -- the claim was true but
         under-evidenced, which is luck, not method.)

    Now every process is kept separate: outcomes are compared per process (the gate is the WORST
    process), and the wall figure is reported as the MEAN across processes with min/max and the
    per-process detail, so no ordering can change the headline number.
    """
    key = lambda r: (r.get("game"), r.get("seed"), r.get("budget"), r.get("early_stop_grace"))  # noqa: E731
    a = {key(r): r for r in serial_rows if r.get("ran")}
    fields = ["levels", "actions", "efficiency", "level_up_actions", "early_stopped"]

    # Group the concurrent rows by SOURCE FILE = one process. `_source` is stamped by load_rows; if a
    # caller ever passes rows without it, everything lands in one bucket and the behaviour degrades
    # to the old single-process reading rather than crashing -- but the bucket count is reported, so
    # a silent collapse to 1 is visible.
    procs: dict[str, dict] = collections.defaultdict(dict)
    for r in conc_rows:
        if r.get("ran"):
            procs[str(r.get("_source", "unknown"))][key(r)] = r

    per_process, all_mism = [], []
    for src in sorted(procs):
        b = procs[src]
        shared_p = sorted(set(a) & set(b), key=lambda t: tuple(str(x) for x in t))
        mism_p = []
        for k in shared_p:
            d = {f: [a[k].get(f), b[k].get(f)] for f in fields if a[k].get(f) != b[k].get(f)}
            if d:
                mism_p.append({"process": src, "cell": [str(x) for x in k], "diff": d})
        all_mism.extend(mism_p)
        ws_p = sum(float(a[k].get("wall_s") or 0) for k in shared_p)
        wc_p = sum(float(b[k].get("wall_s") or 0) for k in shared_p)
        per_process.append(
            {
                "process": src,
                "n_cells_compared": len(shared_p),
                "serial_wall_s": round(ws_p, 2),
                "concurrent_wall_s": round(wc_p, 2),
                "wall_inflation_factor": (round(wc_p / ws_p, 3) if ws_p else None),
                "n_outcome_mismatches": len(mism_p),
            }
        )

    shared = sorted(
        set(a) & set().union(*[set(p) for p in procs.values()]) if procs else set(),
        key=lambda t: tuple(str(x) for x in t),
    )
    ws = sum(float(a[k].get("wall_s") or 0) for k in shared)
    concs = [p["concurrent_wall_s"] for p in per_process if p["concurrent_wall_s"]]
    infl = [p["wall_inflation_factor"] for p in per_process if p["wall_inflation_factor"]]
    return {
        "n_concurrent_processes": len(per_process),
        "n_cells_compared": len(shared),
        "serial_wall_s": round(ws, 2),
        # THE HEADLINE IS THE MEAN ACROSS PROCESSES, and the spread is published beside it so nobody
        # has to trust a single process. min/max are the honest bounds on "what concurrency cost".
        "concurrent_wall_s_mean": (round(statistics.mean(concs), 2) if concs else None),
        "concurrent_wall_s_min": (round(min(concs), 2) if concs else None),
        "concurrent_wall_s_max": (round(max(concs), 2) if concs else None),
        "wall_inflation_factor_mean": (round(statistics.mean(infl), 3) if infl else None),
        "wall_inflation_factor_min": (round(min(infl), 3) if infl else None),
        "wall_inflation_factor_max": (round(max(infl), 3) if infl else None),
        "per_process": per_process,
        "n_outcome_mismatches": len(all_mism),
        "n_processes_with_a_mismatch": sum(1 for p in per_process if p["n_outcome_mismatches"]),
        "outcomes_identical": len(all_mism) == 0,
        "outcomes_identical_scope": (
            f"every one of {len(per_process)} concurrent processes compared against serial on all "
            "shared cells -- not just whichever process happened to be listed last"
        ),
        "mismatches": all_mism[:20],
        "principle": (
            "The BENEFIT headline is a wall-clock claim, so the wall-clock measurement condition "
            "has to be stated. The primary sweep ran SERIALLY for exactly this reason; this "
            "control quantifies what concurrency would have done to that number, and confirms it "
            "does not change any outcome. Reported per process AND as a mean+range, because "
            "collapsing N processes to one row makes the figure depend on argument order and leaves "
            "N-1 processes' outcomes unchecked."
        ),
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--reproduction-rows", nargs="*", default=[])
    ap.add_argument("--contention-serial-rows", nargs="*", default=[])
    ap.add_argument("--contention-concurrent-rows", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    rows, dup_report = dedupe_rows(load_rows(a.rows))
    budgets = sorted({r.get("budget") for r in rows if r.get("budget")})
    conditions = {f"b{b}": analyse_condition(rows, b) for b in budgets}

    repro = (
        reproduction_check(rows, load_rows(a.reproduction_rows)) if a.reproduction_rows else None
    )
    cont = (
        contention_check(
            load_rows(a.contention_serial_rows), load_rows(a.contention_concurrent_rows)
        )
        if a.contention_serial_rows and a.contention_concurrent_rows
        else None
    )

    census = instrumentation_census(rows, dup_report)
    # EVERY row file, not just --rows: the reproduction, contention and stress arms are real live
    # measurement wall clock too, and omitting them would undercount the run by ~30%.
    measurement_wall_s = measurement_wall_clock(
        list(a.rows)
        + list(a.reproduction_rows)
        + list(a.contention_serial_rows)
        + list(a.contention_concurrent_rows)
    )

    # ---- HEADLINE ----------------------------------------------------------------------
    # Chosen strictly by the gates, never by eye.
    per_arm_summary = []
    for cname, cond in conditions.items():
        for aname, arm in cond["arms"].items():
            per_arm_summary.append(
                {
                    "condition": cname,
                    "corpus_level": cond["is_full_25_game_corpus"],
                    "n_games": cond["n_games_covered"],
                    "grace": arm["grace"],
                    "n_matched_cells": arm["n_matched_cells"],
                    "safety_verdict": arm["gate_safety"]["verdict"],
                    "safety_falsifiable": arm["gate_safety"]["falsifiable"],
                    "safety_movable_cells": arm["gate_safety"]["witness_movable_cells"],
                    "safety_at_risk_cells": arm["gate_safety"]["witness_at_risk_cells"],
                    "safety_p_min_reachable_two_sided": arm["gate_safety"]["witness_support_power"][
                        "p_min_reachable_two_sided_at_this_support"
                    ],
                    "regressing_games": arm["gate_safety"]["regressing_games"],
                    "score_verdict": arm["gate_score_non_inferiority_authoritative"]["verdict"],
                    "score_delta_sum": arm["gate_score_non_inferiority_authoritative"]["delta_sum"],
                    "actions_saved_pct": arm["benefit_actions_and_wall"]["total_actions_saved_pct"],
                    "wall_saved_pct": arm["benefit_actions_and_wall"]["total_wall_s_saved_pct"],
                    "cells_early_stopped": arm["n_cells_early_stopped"],
                }
            )
    # A SAFE ARM: both gates pass AND it fired somewhere. The SAFETY pass is now split by
    # FALSIFIABILITY (see gate_safety.falsifiable): `PASS_AT_RISK_CELLS_SURVIVED` is a cell that
    # COULD have regressed and did not -- real evidence; `PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK` is
    # a pass forced by "the grace exceeds every observed gap", which is the DEFINITION of in-sample
    # safety for a fixed grace and carries no independent information. Both are reported; they are
    # never summed into one list, because the second does not generalise (grace 1300 is an
    # unfalsifiable pass at b2000 and regresses 6 cells at b4000).
    safe_arms = [
        s
        for s in per_arm_summary
        if s["safety_verdict"].startswith("PASS")
        and s["score_verdict"].startswith("PASS")
        and s["cells_early_stopped"] > 0
    ]
    safe_arms_falsifiable = [
        s for s in safe_arms if s["safety_verdict"] == "PASS_AT_RISK_CELLS_SURVIVED"
    ]
    safe_arms_by_construction = [
        s for s in safe_arms if s["safety_verdict"] == "PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK"
    ]
    # The headline benefit number must come from a CORPUS-LEVEL condition -- a subset condition's
    # savings are sums over games chosen for their level-reaching and are not a corpus claim.
    corpus_safe = [s for s in safe_arms if s["corpus_level"]]
    # PREFER A FALSIFIABLE PASS. Only if none exists does the headline fall back to a
    # pass-by-construction, and it is stamped as such rather than presented as "the safe value".
    corpus_safe_falsifiable = [s for s in corpus_safe if s["safety_falsifiable"]]
    pool = corpus_safe_falsifiable or corpus_safe
    best = max(pool, key=lambda s: s["actions_saved_pct"] or 0.0) if pool else None

    # ---- OUT-OF-SAMPLE FRAGILITY OF A FIXED GRACE, computed from the sweep's own conditions -------
    # The strongest available check on "grace G is safe" is whether G survives at ANOTHER budget,
    # since a raised budget widens the inter-level-up gap distribution. Any G that passes in one
    # condition and FAILS in another is direct evidence that the pass did not generalise -- which is
    # the whole argument against shipping a fixed value, and it is a fact about arms already run.
    by_grace: dict[int, list[dict]] = collections.defaultdict(list)
    for s in per_arm_summary:
        by_grace[int(s["grace"])].append(s)
    fragility = []
    for g in sorted(by_grace):
        vs = by_grace[g]
        if len(vs) < 2:
            continue
        passes = [v["condition"] for v in vs if v["safety_verdict"].startswith("PASS")]
        fails = [v["condition"] for v in vs if v["safety_verdict"] == "FAIL_LEVEL_REGRESSION"]
        fragility.append(
            {
                "grace": g,
                "conditions_tested": [v["condition"] for v in vs],
                "passed_in": passes,
                "failed_in": fails,
                "regressing_games_where_it_failed": sorted(
                    {gm for v in vs for gm in (v["regressing_games"] or [])}
                ),
                "passes_somewhere_and_fails_elsewhere": bool(passes and fails),
            }
        )
    out_of_sample = {
        "per_grace": fragility,
        "n_graces_tested_at_more_than_one_budget": len(fragility),
        "n_graces_that_pass_somewhere_and_fail_elsewhere": sum(
            1 for f in fragility if f["passes_somewhere_and_fails_elsewhere"]
        ),
        "any_fixed_grace_safe_at_every_budget_it_was_tested_at": bool(
            fragility and all(not f["failed_in"] for f in fragility)
        ),
        "principle": (
            "A fixed grace's safety is a property of the gap distribution, which widens with the "
            "action budget. A value that passes at one budget and regresses levels at another has "
            "been shown NOT to generalise -- the nearest thing to an out-of-sample test this sweep "
            "can run, and it uses only arms already measured."
        ),
    }

    art = {
        "experiment": "arc_early_stop_grace_sweep",
        "experiment_id": "outer_loop_early_stop_grace_sweep_20260726",
        "title": (
            "SUBMITTED_EARLY_STOP_GRACE parameter sweep on the scored path: a WALL-CLOCK lever, "
            "not a score lever"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "outer-loop 2026-07-26",
        "random_seed": 20260724,
        "random_seeds_used": sorted({r.get("seed") for r in rows if r.get("seed")}),
        "duration_s": None,  # filled below
        # WHY aggregation AND NOT the live-agent substrate (fixed 2026-07-26, review finding).
        # This script does not step a single ARC env. It reads the persisted row files that
        # scripts/arc_scored_path_early_stop_sweep.py wrote and computes gates/deltas over them --
        # exactly CLAUDE.md's `aggregation_from_upstream_artifacts` ("reads upstream JSON, computes
        # deltas / formats tables"). Declaring the live-agent substrate here made `duration_s`
        # (this analyser's ~8s pass) read as the cost of the live run, understating the real
        # measurement by three orders of magnitude. The live-agent substrate belongs on the ROW
        # files, which is where the env-stepping actually happened; their wall clock is summed
        # into `measurement_wall_s` below so the reader gets the true figure and cannot mistake
        # the analyser's runtime for it. The 0.0001s aggregation floor is honest for an 8s read.
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "measurement_wall_s": measurement_wall_s,
        "measurement_wall_s_note": (
            "TRUE wall clock of the underlying LIVE measurement, summed from each row file's own "
            "`elapsed_s` (the driving process's clock, which includes per-cell setup the summed "
            "per-cell `wall_s` omits). This is the number to quote for 'how long did the sweep "
            "take' -- NOT the top-level `duration_s`, which times only this analyser's pass over "
            "the already-persisted rows."
        ),
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "preconditions_checked": preconditions_checked(list(a.rows)),
        "instrumentation_census": census,
        "charge_model_resolution": {
            "resolved": "actions_to_level",
            "how": (
                "From the INSTALLED scorer's source, which is the package the competition gateway "
                "itself runs -- not inferred from an experiment. A completed level is charged "
                "actions_at_level - prev_actions (arc_agi/scorecard.py:479); an incomplete level "
                "scores 0.0 regardless of actions charged (:178-183), so the post-solve tail is "
                "billed to a zero-scoring bucket."
            ),
            "consequence_for_this_sweep": (
                "Cutting a score-free tail can only leave the score unchanged or LOSE it. The "
                "originally-specified EFFICIENCY gate ('the per-level sum must improve') therefore "
                "has an EMPTY PASS REGION and was restated as NON-INFERIORITY before the run, "
                "rather than run as specified and reported as a forced failure."
            ),
            "refuted_alternative": "total_action_charge -- computed per arm as a sensitivity check",
        },
        "conditions": conditions,
        "per_arm_summary": per_arm_summary,
        "reproduction": repro,
        "contention_control": cont,
        "headline": {
            "safe_arms": safe_arms,
            "safe_arms_falsifiable_at_risk_cells_survived": safe_arms_falsifiable,
            "safe_arms_pass_by_construction_no_cell_at_risk": safe_arms_by_construction,
            "corpus_level_safe_arms": corpus_safe,
            "best_corpus_safe_arm_by_actions_saved": best,
            "best_arm_safety_is_falsifiable": (bool(best["safety_falsifiable"]) if best else None),
            # THE CAVEAT ATTACHED TO THE HEADLINE ARM ITSELF, not left to a reader to go find. If the
            # headline grace value regressed levels at another budget, that fact belongs next to the
            # number, because the number is what gets quoted.
            "best_arm_failed_at_another_budget": (
                next(
                    (f["failed_in"] for f in fragility if f["grace"] == int(best["grace"])),
                    [],
                )
                if best
                else None
            ),
            "best_arm_caveat": (
                (
                    f"grace {best['grace']} passes safety at {best['condition']} ONLY BY "
                    "CONSTRUCTION (no cell was at risk) "
                    + (
                        "and it REGRESSES LEVELS at "
                        + ",".join(
                            next(
                                (
                                    f["failed_in"]
                                    for f in fragility
                                    if f["grace"] == int(best["grace"])
                                ),
                                [],
                            )
                        )
                        + " -- it is demonstrably not safe out of sample."
                        if any(
                            f["grace"] == int(best["grace"]) and f["failed_in"] for f in fragility
                        )
                        else "and it was not tested at any other budget, so nothing is known about "
                        "its out-of-sample behaviour."
                    )
                )
                if best
                else None
            ),
            "fixed_grace_out_of_sample_fragility": out_of_sample,
            "selection_rule": (
                "An arm is SAFE only if BOTH gates pass AND it fired somewhere (a never-firing arm "
                "is uninstrumented, not safe). A FALSIFIABLE safety pass (some cell was at risk and "
                "survived) is preferred over a pass BY CONSTRUCTION (no cell was at risk, so the "
                "pass is forced by 'grace exceeds every observed gap'); the headline arm is the "
                "corpus-level safe arm with the largest action saving from the preferred pool, and "
                "`best_arm_safety_is_falsifiable` says which pool it came from. The choice is made "
                "by the gates, never by eye."
            ),
            # A WALL-CLOCK NOISE FLOOR THAT COSTS NOTHING TO MEASURE. An arm whose grace exceeds its
            # budget cannot fire, so its TRUE saving is exactly zero on every axis. Its measured
            # `actions_saved_pct` must therefore be 0.0 (a determinism check: the action trace is a
            # deterministic function of the seed), and its measured `wall_saved_pct` is PURE NOISE.
            # Without this, a small wall saving could not be told from timing jitter -- and wall
            # clock is the axis this mechanism's whole case rests on.
            "inert_arm_noise_floor": {
                "inert_arms": [
                    {
                        "condition": s["condition"],
                        "grace": s["grace"],
                        "actions_saved_pct": s["actions_saved_pct"],
                        "wall_saved_pct": s["wall_saved_pct"],
                    }
                    for s in per_arm_summary
                    if s["cells_early_stopped"] == 0
                    and s["grace"] is not None
                    and s["grace"] >= conditions[s["condition"]]["budget"]
                ],
                "max_abs_wall_saved_pct_on_an_inert_arm": (
                    max(
                        (
                            abs(s["wall_saved_pct"] or 0.0)
                            for s in per_arm_summary
                            if s["cells_early_stopped"] == 0
                            and s["grace"] is not None
                            and s["grace"] >= conditions[s["condition"]]["budget"]
                        ),
                        default=None,
                    )
                ),
                "all_inert_arms_saved_exactly_zero_actions": all(
                    (s["actions_saved_pct"] or 0.0) == 0.0
                    for s in per_arm_summary
                    if s["cells_early_stopped"] == 0
                    and s["grace"] is not None
                    and s["grace"] >= conditions[s["condition"]]["budget"]
                ),
                "principle": (
                    "An inert arm's true saving is zero by construction. Zero measured ACTION "
                    "saving on every inert arm is a determinism witness; its measured WALL saving "
                    "is the noise floor any wall-clock benefit claim must clear."
                ),
            },
        },
        "rows_analysed": len(rows),
        "rows_sources": list(a.rows),
        "field_provenance": {
            "efficiency": {
                "principle": (
                    "The AUTHORITATIVE per-game score, produced by the installed arc_agi "
                    "EnvironmentScoreCalculator via arc_leaderboard_eval.run_game -- never a "
                    "re-derived formula. A prior project paraphrase (min(human/agent,1)^2) is wrong "
                    "on three counts and is retracted."
                ),
                "satisfied_by": "run_game -> arc_agi.scorecard.EnvironmentScoreCalculator",
            },
            "level_up_actions": {
                "principle": (
                    "The cumulative checkpoint vector the scorer DIFFERENCES. Persisted so any "
                    "score claim in this artifact is recomputable without re-running a cell, and "
                    "so a score delta can be attributed to a specific level."
                ),
                "satisfied_by": "reconstructed from run_game's per_level agent_actions",
            },
            "early_stopped": {
                "principle": (
                    "Read off the live explorer, not inferred from the action count. An arm that "
                    "never fired contributes no evidence; without this the dead-code wiring would "
                    "have produced a clean null."
                ),
                "satisfied_by": "StepwiseExplorer.early_stopped",
            },
            "n_resets": {
                "principle": (
                    "The live gateway charges a RESET one action; this offline harness charges "
                    "zero. Offline efficiency is therefore OPTIMISTIC by this many actions per "
                    "level, and the grace window is counted in frames (RESETs included) while the "
                    "gaps are in actions. Recorded so the gap is visible, not assumed away."
                ),
                "satisfied_by": "count of RESET moves in run_game's frame_sequence",
            },
            "preconditions_checked": {
                "principle": (
                    "Records WHICH resources were verified before the measurement, pre-empting the "
                    "failure mode where the agent silently lacked one and synthesised a result."
                ),
                "satisfied_by": "explicit checks recorded at run time",
            },
        },
    }

    # ---- TOP-LEVEL ACCEPTANCE GATES ------------------------------------------------------
    # Hoisted to the top level ON PURPOSE. The project's Reading-Results Discipline mandates
    # reading an artifact through `scripts/summarize_artifact.py`, which surfaces
    # `acceptance_gate_*` fields and lets a FAILED gate override a celebratory verdict. Gates
    # buried inside `conditions[...].arms[...]` are invisible to that reader, so a per-arm failure
    # could sit under a headline that sounds like a win. These mirror the per-arm gates exactly;
    # they add no new judgement.
    shipped = conditions.get("b400")
    shipped_safe_firing = [
        s
        for s in per_arm_summary
        if s["condition"] == "b400"
        and s["safety_verdict"].startswith("PASS")
        and s["cells_early_stopped"] > 0
    ]
    art["acceptance_gate_swept_parameter_applied_on_every_row"] = bool(
        census["grace_applied_all_rows"]
    )
    art["acceptance_gate_every_firing_capable_arm_fired_somewhere"] = bool(
        census["every_firing_capable_arm_fired"]
    )
    art["acceptance_gate_zero_none_valued_diagnostics"] = bool(
        census["zero_none_valued_diagnostics"]
    )
    art["acceptance_gate_all_arms_pin_the_same_eight_gated_flags"] = bool(
        census["all_arms_pin_the_same_eight_gated_flags"]
    )
    art["acceptance_gate_no_crashed_cell"] = census["n_crashed"] == 0
    art["acceptance_gate_submitted_early_stop_grace_flag_untouched"] = all(
        p["available"]
        for p in art["preconditions_checked"]
        if "SUBMITTED_EARLY_STOP" in p["resource"]
    )
    art["acceptance_gate_reproducible_across_a_fresh_process"] = (
        bool(repro["deterministic"]) if repro else None
    )
    art["acceptance_gate_contention_changes_no_outcome"] = (
        bool(cont["outcomes_identical"]) if cont else None
    )
    art["acceptance_gate_reimplemented_scorer_matches_installed_scorer_on_every_cell"] = all(
        arm["sensitivity_refuted_total_action_charge"]["authoritative_reimplementation_crosscheck"][
            "matches_installed_scorer_on_every_cell"
        ]
        for cond in conditions.values()
        for arm in cond["arms"].values()
    )
    # THE DECISION GATE, RENAMED TO ITS MEASURED SCOPE (adversarial review, 2026-07-26). It asks
    # "does a grace value we TESTED fire and cost no level at the shipped budget?" -- NOT the
    # existence claim over the whole parameter space that the previous name
    # (`a_firing_grace_value_is_safe_...`) asserted. A grid can only answer the former. The window
    # bounds are computed separately (`conditions.b400.safe_firing_window`), and the value the
    # original grid jumped over (grace 350) was subsequently RUN.
    art["acceptance_gate_a_TESTED_firing_grace_value_is_safe_at_the_shipped_budget_b400"] = (
        len(shipped_safe_firing) > 0
    )
    art["acceptance_gate_shipped_budget_safety_witness_non_empty"] = (
        (
            max(
                (s["safety_movable_cells"] for s in per_arm_summary if s["condition"] == "b400"),
                default=0,
            )
            > 0
        )
        if shipped
        else None
    )
    # DOES THE TESTED GRID EVEN SPAN THE DECISION-RELEVANT REGION? Without this, a "no value is
    # safe" answer cannot be told apart from "the grid missed the window". Computed from the control
    # arm, so it is checkable before any treatment arm runs.
    art["acceptance_gate_shipped_budget_grid_spans_the_safe_firing_window"] = (
        bool(shipped["safe_firing_window"]["grid_spans_the_window"]) if shipped else None
    )
    # IS THE HEADLINE ARM'S WALL SAVING REAL? The mechanism's entire case is wall clock, which is the
    # noisiest quantity measured. A saving larger than the actions saved can account for -- at the
    # control's own seconds-per-action -- is jitter. The grace-350 arm at b400 measures 2.42% wall off
    # 0.072% actions, a ratio of ~34, so this gate FAILS for it, correctly.
    _hb = (
        conditions[best["condition"]]["arms"][str(best["grace"])]["benefit_actions_and_wall"][
            "wall_saving_attribution"
        ]
        if best
        else None
    )
    art["acceptance_gate_headline_arm_wall_saving_is_attributable_not_noise"] = (
        (not _hb["noise_dominated"]) if _hb else None
    )
    _sb = (
        conditions["b400"]["arms"][
            str(max(shipped_safe_firing, key=lambda s: s["actions_saved_pct"] or 0.0)["grace"])
        ]["benefit_actions_and_wall"]["wall_saving_attribution"]
        if shipped_safe_firing
        else None
    )
    art["acceptance_gate_shipped_budget_safe_arm_wall_saving_is_attributable_not_noise"] = (
        (not _sb["noise_dominated"]) if _sb else None
    )
    art["gate_definitions_and_deliberate_exclusions"] = {
        "a_TESTED_firing_grace_value_is_safe_at_the_shipped_budget_b400": (
            "THE DECISION GATE, scoped to the values actually measured. True means at least one "
            "TESTED grace fires at the shipped MAX_ACTIONS=400 and costs no level on any seed. It "
            "deliberately does NOT claim anything about untested values: the earlier name asserted "
            "an existence property of the whole parameter space, which a grid cannot support. Read "
            "it together with `..._grid_spans_the_safe_firing_window` and with "
            "`conditions.b400.safe_firing_window`, which computes the region from the control arm."
        ),
        "shipped_budget_safety_witness_non_empty": (
            "Guards the gate above against being forced. If NO cell at b400 reached >=2 levels, no "
            "arm could have regressed and the decision gate would carry no information."
        ),
        "shipped_budget_grid_spans_the_safe_firing_window": (
            "The grid-adequacy guard. A 'no safe value' answer from a grid that contains no point "
            "strictly between the largest at-risk gap and the largest post-level-up tail is a GRID "
            "ARTIFACT, not a property of the mechanism. This is why grace 350 was run."
        ),
        "headline_arm_wall_saving_is_attributable_not_noise": (
            "Wall clock is this mechanism's entire case AND the noisiest thing measured. A saving "
            "larger than the actions saved can buy at the control's own seconds-per-action is "
            "jitter, and would be quoted as a benefit if unchecked. FAILS for the b400 safe arm on "
            "purpose: 0.072% actions saved against a 2.42% measured wall saving is a ratio of ~34."
        ),
        "shipped_budget_safe_arm_wall_saving_is_attributable_not_noise": (
            "Same test applied to the SHIPPED-budget safe arm specifically, since that is the "
            "configuration the decision is about. Read it with `inert_arm_noise_floor`: an arm that "
            "cannot fire measures the same jitter with a TRUE saving of exactly zero."
        ),
        "why_a_PASS_here_is_still_not_a_recommendation": (
            "A safe firing value existing at b400 does NOT make the mechanism worth shipping. Its "
            "safety is a pass BY CONSTRUCTION (no cell was at risk -- see gate_safety.falsifiable) "
            "and its entire measured benefit is a fraction of a percent of corpus actions on a "
            "single cell. The operator-facing conclusion is unchanged: a wall-clock lever with "
            "negligible value at the shipped budget, whose safe window is nearly empty of value "
            "rather than empty of members."
        ),
        "excluded_deliberately": (
            "There is NO 'the efficiency sum improved' gate. Under the resolved charge model the "
            "post-solve tail costs zero score, so that gate's pass region is empty and reporting "
            "it would manufacture a forced failure. Score is gated on NON-INFERIORITY per arm."
        ),
    }
    # ---- SCOPE AND POWER OF THE SHIPPED-BUDGET VERDICT, hoisted (adversarial review 2026-07-26) ---
    # The b400 safety verdict is decided by ONE GAME and its decisive gap by ONE SEED, and 4 of 75
    # cells carry ~91% of the score total. Those were disclosed only as per-arm residuals, where a
    # headline sentence reading as a corpus-wide finding could sit on top of them. Hoisted so
    # `summarize_artifact.py` surfaces them beside the verdict.
    if shipped:
        b4 = shipped
        sw = b4["safe_firing_window"]
        sc = b4["score_concentration"]
        b4_arms = list(b4["arms"].values())
        movable_games = sorted(
            {g for a_ in b4_arms for g in a_["gate_safety"]["witness_movable_games"]}
        )
        regressing_games = sorted(
            {g for a_ in b4_arms for g in a_["gate_safety"]["regressing_games"]}
        )
        at_risk_max = max((a_["gate_safety"]["witness_at_risk_cells"] for a_ in b4_arms), default=0)
        p_floor = min(
            (
                a_["gate_safety"]["witness_support_power"][
                    "p_min_reachable_two_sided_at_this_support"
                ]
                for a_ in b4_arms
            ),
            default=1.0,
        )
        art["shipped_budget_scope_and_power"] = {
            "condition": "b400",
            "control_cells": b4["control_cells"],
            "control_levels_distribution": b4["control_levels_distribution"],
            "n_cells_reaching_any_level": b4["mechanism_reach"][
                "cells_where_window_can_arm_levels_ge_1"
            ],
            "n_cells_reaching_2plus_levels_movable": b4["control_cells_reaching_2plus_levels"],
            "movable_games": movable_games,
            "n_movable_games": len(movable_games),
            "regressing_games": regressing_games,
            "max_at_risk_cells_on_any_arm": at_risk_max,
            "best_reachable_two_sided_p_on_any_arm": p_floor,
            "score_top_1_share_pct": sc["top_1_share_pct"],
            "score_top_4_share_pct": sc["top_4_share_pct"],
            "score_top_cells": sc["top_cells"],
            "safe_firing_window_frames": [
                sw["lower_bound_frames_exclusive"],
                sw["upper_bound_frames_exclusive"],
            ],
            "grid_spans_the_window": sw["grid_spans_the_window"],
            "tested_graces_inside_the_window": sw["tested_graces_inside_the_window"],
            "honest_scope_statement": (
                "AT THE SHIPPED BUDGET THIS IS A SINGLE-GAME, SINGLE-OBSERVATION RESULT, NOT A "
                f"CORPUS ESTIMATE. Of {b4['control_cells']} control cells "
                f"{b4['control_levels_distribution'].get(0, 0)} reach ZERO levels and only "
                f"{b4['control_cells_reaching_2plus_levels']} reach two -- and all of those are the "
                f"same game ({', '.join(movable_games) or 'none'}), so every level-regression at "
                "b400 is one game's. The largest at-risk gap that sets the safe-window boundary "
                f"comes from ONE seed of that game. With at most {at_risk_max} at-risk cells the "
                f"smallest two-sided p reachable is {p_floor}"
                ", so NO significance on the regression question is attainable at b400 no matter how "
                f"many seeds are added to these games. Separately, {sc['top_4_share_pct']}% of the "
                "corpus score total sits in 4 of "
                f"{sc['n_control_cells']} cells (the top cell alone {sc['top_1_share_pct']}%), so a "
                "reported score delta is a statement about those cells, not a corpus effect. For a "
                "safety claim intended to generalise, read the b4000 condition, whose movable set is "
                "an order of magnitude larger -- and note its `score_sums_are_corpus_level: false`."
            ),
        }
    violations = [
        k
        for k, v in art.items()
        if k.startswith("acceptance_gate_") and isinstance(v, bool) and v is False
    ]
    art["acceptance_gate_violations"] = violations
    art["acceptance_gates_all_passed"] = not violations

    # honest_verdict, terminal-prefixed, chosen by the gates
    n_reg = sum(1 for s in per_arm_summary if s["safety_verdict"] == "FAIL_LEVEL_REGRESSION")
    # Keyed on the WIRING-SUSPECT set, not on "every arm fired": an arm whose grace exceeds its
    # budget cannot fire by construction and is a deliberate inert control, so counting it as an
    # instrumentation failure would suppress a real, well-measured result.
    if not census["every_firing_capable_arm_fired"]:
        art["honest_verdict"] = (
            "complete_early_stop_grace_sweep_uninterpretable_firing_capable_arms_never_fired_"
            f"{'_'.join(census['uninstrumented_arms_wiring_suspect'])}"
        )
    elif not art["acceptance_gate_a_TESTED_firing_grace_value_is_safe_at_the_shipped_budget_b400"]:
        # THE VERDICT LEADS WITH THE SHIPPED-BUDGET ANSWER, because that is the configuration the
        # operator's decision is about. A verdict that led with "safe at grace 1300" would be true
        # of a RAISED budget and read as a recommendation for the shipped one -- the same
        # condition-substitution that a raised-budget headline invites. Named explicitly here so a
        # future edit cannot quietly promote the friendlier condition. SCOPED to the tested grid, and
        # it states whether the grid even spanned the safe window, so a grid artifact cannot read as
        # a property of the mechanism.
        art["honest_verdict"] = (
            "complete_early_stop_grace_sweep_no_TESTED_firing_grace_value_is_level_safe_at_the_"
            f"shipped_budget_b400_all_{n_reg}_firing_arms_regressed_a_level_grid_spanned_window_"
            f"{art['acceptance_gate_shipped_budget_grid_spans_the_safe_firing_window']}_"
            "mechanism_is_a_wall_clock_lever_not_a_score_lever"
            + (
                f"_only_safe_firing_point_found_was_grace_{best['grace']}_at_{best['condition']}_"
                f"saving_{best['actions_saved_pct']}pct_actions_falsifiable_"
                f"{best['safety_falsifiable']}"
                if best is not None
                else "_and_no_raised_budget_arm_was_safe_either"
            )
        )
    elif best is None:
        art["honest_verdict"] = (
            "complete_early_stop_grace_sweep_no_arm_cleared_both_gates_"
            f"{n_reg}_arms_regressed_levels"
        )
    else:
        # A SAFE FIRING VALUE EXISTS AT THE SHIPPED BUDGET -- and the verdict must not let that read
        # as a recommendation. It carries the benefit SIZE and the FALSIFIABILITY of the safety
        # pass, because the honest summary is "safe, and worth almost nothing": the safe window at
        # b400 is bounded below by the largest at-risk gap and above by the largest tail, and the
        # only value inside it fires on a single cell.
        shipped_best = max(
            (s for s in shipped_safe_firing),
            key=lambda s: s["actions_saved_pct"] or 0.0,
        )
        art["honest_verdict"] = (
            "complete_early_stop_grace_sweep_measured_wall_clock_lever_not_score_lever_"
            f"shipped_budget_b400_safe_firing_grace_{shipped_best['grace']}_saves_only_"
            f"{shipped_best['actions_saved_pct']}pct_actions_on_"
            f"{shipped_best['cells_early_stopped']}_of_{shipped_best['n_matched_cells']}_cells_"
            f"score_delta_{shipped_best['score_delta_sum']}_safety_pass_falsifiable_"
            f"{shipped_best['safety_falsifiable']}_wall_saving_noise_dominated_"
            f"{(_sb or {}).get('noise_dominated')}_not_worth_shipping_at_b400"
        )

    # ---- STALENESS PROVENANCE + INDEX REGISTRATION -----------------------------------------
    # REUSED, not reimplemented. `scripts/analyze_scored_path_lever_ab.py` already owns the
    # fingerprint helpers and the index the pre-commit `artifact-freshness-lint` reads; a second copy
    # here would be a second thing to keep in step, which is the defect class that guard exists for.
    # WITHOUT THIS BLOCK THIS ARTIFACT WOULD BE INVISIBLE TO THE GUARD: the lint checks an INDEX
    # rather than scanning the 6.1 GB results/ tree, so an unregistered artifact is never checked and
    # a future edit to this analyser would leave its numbers stale and unflagged.
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        import analyze_scored_path_lever_ab as sibling

        code_deps = [
            Path(__file__).resolve(),
            REPO / "scripts" / "arc_scored_path_early_stop_sweep.py",
            REPO / "scripts" / "arc_scored_path_lever_harness.py",
            REPO / "scripts" / "arc_leaderboard_eval.py",
            REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py",
        ]
        art["provenance"] = {
            "git_head": sibling._git_head(),
            "code": [sibling._file_fingerprint(p) for p in code_deps],
            "rows_sources": {
                "rows": [sibling._file_fingerprint(Path(p)) for p in a.rows],
                "reproduction_rows": [
                    sibling._file_fingerprint(Path(p)) for p in a.reproduction_rows
                ],
                "contention_serial_rows": [
                    sibling._file_fingerprint(Path(p)) for p in a.contention_serial_rows
                ],
                "contention_concurrent_rows": [
                    sibling._file_fingerprint(Path(p)) for p in a.contention_concurrent_rows
                ],
            },
            "rebuild_command": " ".join(
                [
                    "python scripts/analyze_arc_early_stop_sweep.py",
                    "--rows",
                    *[str(p) for p in a.rows],
                    *(
                        ["--reproduction-rows", *[str(p) for p in a.reproduction_rows]]
                        if a.reproduction_rows
                        else []
                    ),
                    *(
                        ["--contention-serial-rows", *[str(p) for p in a.contention_serial_rows]]
                        if a.contention_serial_rows
                        else []
                    ),
                    *(
                        [
                            "--contention-concurrent-rows",
                            *[str(p) for p in a.contention_concurrent_rows],
                        ]
                        if a.contention_concurrent_rows
                        else []
                    ),
                    "--out",
                    str(a.out),
                ]
            ),
            "note": (
                "principle: an artifact that cannot say which code produced it cannot be known to "
                "be current, and a stale artifact's numbers are quoted exactly as confidently as a "
                "fresh one's. Every ROW-SOURCE DESIGN is fingerprinted, not just --rows, so a "
                "swapped contention or reproduction file cannot pass unnoticed."
            ),
        }
        sibling.register_analyzed_artifact(Path(a.out), analyzer=Path(__file__).resolve())
    except Exception as exc:  # provenance must never block the measurement itself
        art["provenance"] = {"error": f"{type(exc).__name__}:{exc}"}

    # PRESERVE the append-only freshness-acknowledgement audit trail across this overwrite (see
    # analyze_scored_path_lever_ab.py:preserve_freshness_acknowledgements for the full incident).
    # Reuses the SAME `sibling` alias imported above; fails open if that import didn't succeed.
    try:
        sibling.preserve_freshness_acknowledgements(art, Path(a.out))
    except Exception:
        pass

    art["duration_s"] = round(time.time() - t0, 3)
    payload = json.dumps(
        {k: art[k] for k in art if k not in ("run_date", "duration_s")}, sort_keys=True
    ).encode()
    art["reproducibility_checksum"] = hashlib.sha256(payload).hexdigest()
    art["reproducibility_checksum_note"] = (
        "sha256 over the WHOLE artifact except run_date and duration_s (wall-clock noise). Every "
        "analysed condition, gate, witness and row-source list is inside the hash."
    )
    try:
        art["git_head"] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
            ).stdout.strip()
            or None
        )
    except Exception:
        art["git_head"] = None
    Path(a.out).write_text(json.dumps(art, indent=2))
    print(json.dumps(art["headline"], indent=2))
    print("verdict:", art["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
