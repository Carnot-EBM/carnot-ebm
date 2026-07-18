"""Bounded actions-to-progress metric harness for ARC-AGI-3 capability A/Bs (REQ-ARC-WMTE-5720).

WHY THIS EXISTS
---------------
Two capability A/Bs run in 2026-07 -- the `/think` vs `/no_think` re-test
(REQ-ARC-WMTE-5714) and the playbook-exemplar retrieval re-test
(REQ-ARC-WMTE-5716/5717/5718) -- BOTH landed on honest-inconclusive verdicts,
and BOTH named the SAME root cause: their metric was *single-shot world-model
induction quality* (graded cell-recall / goal-predicate accuracy on ONE fixed
transition window). That proxy is:

  * FLOORED: exact-match reproduction accuracy is 0 for almost every arm, so the
    only discriminator is graded cell-recall, which is itself at its floor for
    most (game, arm) pairs (exp5719: `metric_floored` sometimes true; deltas
    `outlier_fragile`).
  * HIGH-VARIANCE: one lucky/unlucky stochastic induction dominates a per-arm
    mean of N=2-8 (exp5714 comparison_a had 10 games x 1 window; exp5719 N=8/arm
    with a single 0.71 outlier flipping the sign).
  * NOT WHAT MATTERS: predicting ONE grid transition accurately is not the
    deliverable. The deliverable (CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game
    Discovery Agent") is whether a capability change helps the agent actually
    make PROGRESS -- reach a level-up, or at least get materially closer to one
    -- inside a bounded live-style solve.

This harness measures REAL PROGRESS in a bounded solve driven by the ACTUAL
scored `E3AgentPolicy` cascade (the same `next_move`/`is_done` loop
`scripts/arc_leaderboard_eval.py:run_game` and `make_carnot_agent.choose_action`
run in production), so a capability change is activated through its REAL live
env-var wiring and the result is directly relevant to a graduation decision.

WHAT IT MEASURES (three floor-busting signals, not one)
-------------------------------------------------------
Per bounded run we record, in decreasing decisiveness:

  1. `levels_gained` / `actions_to_first_solve` -- the RHAE-relevant ground
     truth: did a REAL level-up fire (`frame.levels_completed` advanced) within
     the action budget, and after how many actions. This is oracle-distinct (the
     win gate is the level counter, never a heuristic) but SPARSE -- on the live
     frame-only path most public games do not level up (consistent with the
     ~0.08 live score), so this is confirmatory, not the primary discriminator.

  2. `hv_progress` -- a DENSE goal-distance-reduction proxy: the per-game
     GameAdapter `hand_verifier` is a hand-written distance-to-goal (LOWER =
     closer to the win; the OfflineSolver uses it as its best-first search
     heuristic). We track its MINIMUM over the run and report
     `(start_hv - best_hv) / max(start_hv, 1)`, the fraction of the way the agent
     closed toward the goal. This CHANGES every time the board gets closer to a
     win, so it discriminates even when no arm achieves a full level-up -- the
     exact floor the single-shot binary/cell-recall proxy could not clear. It is
     a MEASUREMENT, not the win oracle (`verifier_is_oracle=False`), and it is a
     DIFFERENT function from the agent's own routing verifier (oracle-distinct).

  3. Induction-quality aggregates over the run's MULTIPLE induction attempts:
     `n_inductions`, `plan_found_rate` (fraction of attempts that produced a
     reachable plan), `mean_heldout_accuracy`, `mean_prefix_accuracy`. Averaging
     the induction signal over every attempt the agent makes inside ONE solve is
     already lower-variance than the prior single-shot induction, and it is the
     signal the capability changes most directly move.

TRACTABILITY TRADEOFF (stated explicitly per the task brief)
------------------------------------------------------------
A full live solve loop per (game, arm, seed) is expensive; the dominant cost is
LLM induction (measured ~30-100s per attempt on a warm CUDA server; a genuine
`/think` reasoning attempt is ~2-3 min). We bound each run THREE ways: an action
`budget` (the real eval constraint -- live MAX_ACTIONS is 400, per-level
ceilings are smaller), a `max_inductions` cap (the PRIMARY cost/fairness bound:
each arm gets the SAME number of induction attempts, so a slower arm is not
penalized by wall time -- the live ARC eval has no per-action time limit), and a
`wall_s` safety cap (a tractability guard, recorded honestly as `timed_out`).
Paired design: the SAME (game, seed) is run under each arm, so shared
exploration luck cancels in the within-pair delta.

FAITHFULNESS / DISCIPLINE
-------------------------
This drives `E3AgentPolicy` -- one of the two live entrypoints (CLAUDE.md "ARC
Live-Path Reachability Discipline") -- so it is the live mechanism, not a
parallel solver. It runs on the PUBLIC offline arcade for development
measurement, so `solve_provenance=development_proxy`, `read_game_source=False`
(we never read a game's `.py` source; the `hand_verifier` progress proxy reads
the live runtime game object via the adapter's public callable, `used_env_source`
declared True). It NEVER flips the frozen live default and NEVER submits.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Arm configuration -- each arm is activated through the REAL live env-var wiring
# ---------------------------------------------------------------------------

# Every knob below is read by the PRODUCTION agent, so configuring an arm here is
# exactly what graduating the feature to the live default would do:
#   * CARNOT_ARC_CODEONLY_INDUCE  -> arc_executable_world_model.py:1604 (the fence)
#   * proposer.no_think_prefix    -> arc_executable_world_model.py:1608 (/think vs /no_think)
#   * CARNOT_ARC_PLAYBOOK_RETRIEVAL / _EXEMPLARS_ENABLED -> arc_competition_agent
#     _induce_and_plan playbook wiring (retrieval > static > none)
#   * proposer.max_tokens         -> the n_predict budget an induction gets
ARM_CONFIGS: dict[str, dict[str, Any]] = {
    # --- MTP+reasoning question (re-test of REQ-ARC-WMTE-5714) ---
    "frozen": {
        "desc": "frozen live default: codeonly fence ON + /no_think, 4096 n_predict",
        "codeonly": "1",
        "no_think_prefix": "/no_think\n",
        "max_tokens": 4096,
        "tries": 3,
        "retrieval": "0",
        "static": "0",
    },
    "reason": {
        "desc": "genuine reasoning: codeonly fence OFF + /think, 8192 n_predict",
        "codeonly": "0",
        "no_think_prefix": "/think\n",
        "max_tokens": 8192,
        "tries": 1,  # a /think overrun that never emits code is the honest finding, not a retry bug
        "retrieval": "0",
        "static": "0",
    },
    # --- playbook-retrieval question (re-test of REQ-ARC-WMTE-5716/5717/5718) ---
    # "none" is byte-identical to "frozen" (the retrieval control); named separately for clarity.
    "none": {
        "desc": "retrieval control: frozen default, no playbook injection",
        "codeonly": "1",
        "no_think_prefix": "/no_think\n",
        "max_tokens": 4096,
        "tries": 3,
        "retrieval": "0",
        "static": "0",
    },
    "retrieval": {
        "desc": "frozen default + top-K retrieved playbook exemplars injected at induction",
        "codeonly": "1",
        "no_think_prefix": "/no_think\n",
        "max_tokens": 4096,
        "tries": 3,
        "retrieval": "1",
        "static": "0",
    },
    "static": {
        "desc": "frozen default + the fixed static playbook exemplar block injected",
        "codeonly": "1",
        "no_think_prefix": "/no_think\n",
        "max_tokens": 4096,
        "tries": 3,
        "retrieval": "0",
        "static": "1",
    },
}


def apply_arm(proposer: Any, arm: str) -> Callable[[], None]:
    """Configure ``proposer`` + process env for ``arm``; return a restore closure.

    The restore closure MUST be called (in a finally) to reset the env vars and
    proposer fields so successive arms on the same reused proposer/server do not
    leak configuration into each other.
    """
    cfg = ARM_CONFIGS[arm]
    saved_env = {
        k: os.environ.get(k)
        for k in (
            "CARNOT_ARC_CODEONLY_INDUCE",
            "CARNOT_ARC_PLAYBOOK_RETRIEVAL",
            "CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED",
        )
    }
    saved_prop = {
        "no_think_prefix": getattr(proposer, "no_think_prefix", None),
        "max_tokens": getattr(proposer, "max_tokens", None),
        "tries": getattr(proposer, "tries", None),
        "include_playbook_exemplars": getattr(proposer, "include_playbook_exemplars", None),
    }

    os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = cfg["codeonly"]
    os.environ["CARNOT_ARC_PLAYBOOK_RETRIEVAL"] = cfg["retrieval"]
    os.environ["CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED"] = cfg["static"]
    proposer.no_think_prefix = cfg["no_think_prefix"]
    proposer.max_tokens = cfg["max_tokens"]
    proposer.tries = cfg["tries"]
    # E3._induce_and_plan resets include_playbook_exemplars itself; clear it defensively too.
    proposer.include_playbook_exemplars = False

    def _restore() -> None:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in saved_prop.items():
            try:
                setattr(proposer, k, v)
            except Exception:
                pass

    return _restore


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------


@dataclass
class ProgressResult:
    game: str
    arm: str
    seed: int
    variant: int
    start_level: int
    reached_level: int
    levels_gained: int
    solved: bool
    actions_to_first_solve: Optional[int]
    total_actions: int
    noop_frac: Optional[float]
    revisit_frac: Optional[float]
    start_hv: Optional[float]
    best_hv: Optional[float]
    hv_progress: Optional[float]
    n_inductions: int
    n_plans_found: int
    plan_found_rate: Optional[float]
    mean_heldout_accuracy: Optional[float]
    mean_prefix_accuracy: Optional[float]
    playbook_injection_modes: list[str]
    wall_s: float
    timed_out: bool
    hit_induction_cap: bool
    error: Optional[str] = None
    induction_events: list[dict[str, Any]] = field(default_factory=list)

    def to_row(self, *, include_events: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_events:
            d.pop("induction_events", None)
        return d


def _hand_verifier_fn(game: str) -> Optional[Callable[[Any, Any], Optional[float]]]:
    """Return a callable (game_obj, frame) -> float goal-distance, or None if the game
    has no adapter hand_verifier. LOWER = closer to the win (adapter convention; the
    OfflineSolver best-first-searches on it). Robust to hand_verifier signatures that
    take one arg (game only) or two (game, frame)."""
    from carnot.agentic import arc_game_adapters as adapters

    ad = adapters.get_adapter(game)
    if ad is None or ad.hand_verifier is None:
        return None
    hv = ad.hand_verifier

    def _call(game_obj: Any, frame: Any) -> Optional[float]:
        for args in ((game_obj, frame), (game_obj,)):
            try:
                v = hv(*args)
                return float(v)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    return _call


def _summarize_inductions(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-induction diagnostics E3AgentPolicy records on
    ``self.level_induction_events``. Each event is a dict with keys such as
    ``planned`` (bool), ``playbook_injection_mode`` (str), and ``refinement_rounds``
    (list of dicts carrying ``heldout_accuracy`` / ``prefix_accuracy``)."""
    n = len(events)
    if n == 0:
        return {
            "n_inductions": 0,
            "n_plans_found": 0,
            "plan_found_rate": None,
            "mean_heldout_accuracy": None,
            "mean_prefix_accuracy": None,
            "playbook_injection_modes": [],
        }
    n_plans = sum(1 for e in events if e.get("planned"))
    modes = [str(e.get("playbook_injection_mode") or "none") for e in events]
    heldouts: list[float] = []
    prefixes: list[float] = []
    for e in events:
        rounds = e.get("refinement_rounds") or []
        best_h = None
        best_p = None
        for r in rounds:
            h = r.get("heldout_accuracy")
            p = r.get("prefix_accuracy")
            if isinstance(h, (int, float)):
                best_h = h if best_h is None else max(best_h, h)
            if isinstance(p, (int, float)):
                best_p = p if best_p is None else max(best_p, p)
        if best_h is not None:
            heldouts.append(float(best_h))
        if best_p is not None:
            prefixes.append(float(best_p))
    return {
        "n_inductions": n,
        "n_plans_found": n_plans,
        "plan_found_rate": round(n_plans / n, 4),
        "mean_heldout_accuracy": round(statistics.mean(heldouts), 4) if heldouts else None,
        "mean_prefix_accuracy": round(statistics.mean(prefixes), 4) if prefixes else None,
        "playbook_injection_modes": modes,
    }


def run_bounded_progress(
    game: str,
    arm: str,
    *,
    proposer: Any,
    seed: int,
    budget: int = 120,
    max_inductions: int = 3,
    wall_s: float = 600.0,
    explore_budget: int = 24,
    variant: int = 0,
    reflect: Optional[str] = None,
) -> ProgressResult:
    """Drive the REAL e3 cascade on ``game`` under ``arm`` for a bounded solve.

    Mirrors `scripts/arc_leaderboard_eval.py:run_game` (the faithful driver) but adds
    the dense `hand_verifier` progress track, the per-arm config, and cost/fairness
    bounds. Returns a populated ProgressResult; a policy crash is captured as `error`
    (a crash on a game is itself a datum, not a harness bug).

    ``explore_budget`` (default 24 = ``SUBMITTED_ROUTED_EXPLORE_BUDGET``, a real shipped
    value) sets how many transitions the agent collects before it STALLS and induces.
    It is set explicitly so every bounded run reliably reaches the induce->plan->execute
    phase -- the ONLY phase the induction arms differ in -- within a small action budget.
    Without it, the graph-explore route's default 80-transition budget means a short run
    never induces and all arms are identical (observed: ls20 at budget=50 did 0 inductions).

    ``max_inductions`` bounds the number of stall->induce cycles (the dominant cost). The
    cap is enforced so the plan from the LAST permitted induction still executes (we only
    stop before a NEW explore->induce cycle, never mid-plan-execution), so the progress
    signal of every induction we pay for is actually measured.
    """
    import random

    import numpy as np
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    random.seed(seed)
    np.random.seed(seed)

    hv_fn = _hand_verifier_fn(game)
    restore = apply_arm(proposer, arm)
    t0 = time.time()
    err: Optional[str] = None
    timed_out = False
    hit_cap = False
    frames: list[Any] = []
    latest: Any = None
    actions = noop = revisit = 0
    seen: set = set()
    start = best = None
    level_up_actions: list[int] = []
    start_hv = best_hv = None

    try:
        pol = E3AgentPolicy(game, proposer=proposer, explore_budget=explore_budget)
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        if variant:
            from carnot.agentic.arc_variant_generator import VariantEnv

            env = VariantEnv(env, game, variant, reflect=reflect)

        for _ in range(budget):
            if time.time() - t0 > wall_s:
                timed_out = True
                break
            # Induction cap: stop before a NEW explore->induce cycle, but let the plan of
            # the last permitted induction finish executing (phase == "execute"). Count
            # `induction_attempts` (the per-stall record), NOT `level_induction_events`.
            if (
                len(getattr(pol, "induction_attempts", []) or []) >= max_inductions
                and getattr(pol, "phase", None) != "execute"
            ):
                hit_cap = True
                break
            if pol.is_done(frames, latest):
                break
            kind, data = pol.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
            elif kind is None:
                break
            else:
                before = latest
                latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
                actions += 1
                if latest is not None:
                    h = frame_hash(grid_of(latest))
                    if before is not None and frame_hash(grid_of(before)) == h:
                        noop += 1
                    if h in seen:
                        revisit += 1
                    seen.add(h)
            lvl = _level_of(latest)
            if start is None:
                start = best = lvl
            if best is not None and lvl > best:
                for _lv in range(best, lvl):
                    level_up_actions.append(actions)
                best = lvl
            if hv_fn is not None and latest is not None:
                hv = hv_fn(getattr(env, "_game", None), latest)
                if hv is not None:
                    if start_hv is None:
                        start_hv = hv
                    if best_hv is None or hv < best_hv:
                        best_hv = hv
            frames.append(latest)
            if latest is None:
                break
        reached = _level_of(latest) if latest is not None else (best or 0)
    except Exception as exc:  # a policy crash on a game is a datum
        err = f"{type(exc).__name__}: {exc}"[:300]
        reached = _level_of(latest) if latest is not None else (best or 0)
        pol = locals().get("pol")
    finally:
        restore()

    events = list(getattr(pol, "induction_attempts", []) or []) if pol is not None else []
    ind = _summarize_inductions(events)
    levels_gained = max(0, (reached or 0) - (start or 0))
    hv_progress = None
    if start_hv is not None and best_hv is not None:
        hv_progress = round(max(0.0, (start_hv - best_hv)) / max(abs(start_hv), 1.0), 4)

    return ProgressResult(
        game=game,
        arm=arm,
        seed=seed,
        variant=variant,
        start_level=start or 0,
        reached_level=reached or 0,
        levels_gained=levels_gained,
        solved=levels_gained >= 1,
        actions_to_first_solve=level_up_actions[0] if level_up_actions else None,
        total_actions=actions,
        noop_frac=round(noop / actions, 4) if actions else None,
        revisit_frac=round(revisit / actions, 4) if actions else None,
        start_hv=start_hv,
        best_hv=best_hv,
        hv_progress=hv_progress,
        n_inductions=ind["n_inductions"],
        n_plans_found=ind["n_plans_found"],
        plan_found_rate=ind["plan_found_rate"],
        mean_heldout_accuracy=ind["mean_heldout_accuracy"],
        mean_prefix_accuracy=ind["mean_prefix_accuracy"],
        playbook_injection_modes=ind["playbook_injection_modes"],
        wall_s=round(time.time() - t0, 1),
        timed_out=timed_out,
        hit_induction_cap=hit_cap,
        error=err,
        induction_events=events,
    )


# ---------------------------------------------------------------------------
# Paired A/B analysis
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Seeded induce->plan->execute mode (the ROBUST primary mode for the re-tests)
# ---------------------------------------------------------------------------
#
# WHY a second mode. run_bounded_progress drives the WHOLE live loop (explore ->
# stall -> induce -> plan -> execute), which is maximally faithful but has one fatal
# reliability flaw for an INDUCTION A/B: on some (game, seed) pairs the explorer is
# replay-heavy and never accumulates enough transitions to STALL within a bounded
# budget, so induction never fires and the induction arms are identical for the WRONG
# reason (observed: ls20 did 0 inductions at budget=136, explore_budget=8). This mode
# removes that confound by SEEDING with a real level-up-straddling `build_window` and calling
# the LLM induction DIRECTLY (`proposer.induce`) with EXPLICIT per-arm injection (the exp5719
# mechanism) -- so induction ALWAYS fires, the arm's prompt is guaranteed applied, and the arm
# config is the only thing that varies. It then uses the LIVE `plan_in_model` planner + executes
# the plan against the real env. Reuses the prior experiments' `build_window` input, so results
# are directly comparable to REQ-ARC-WMTE-5714/5719 with ONLY the metric upgraded from
# single-shot cell-recall to actions-to-progress (does the induced model's OWN plan reach a win).
# (`run_bounded_progress` above remains the maximally-faithful whole-loop mode for other uses.)


@dataclass
class SeededProgressResult:
    game: str
    arm: str
    trial: int
    induction_ok: bool
    # induce_ok: did THIS cell's proposer.induce actually succeed and (over)write the engine? The
    # stale-engine attribution bug (found 2026-07-18 by the exp5722 generator-swap adversarial review)
    # was that load_engine reads a per-GAME world_model.py which a FAILED induce leaves untouched, so a
    # later run silently re-reads (and is scored on) an EARLIER run's engine. induction_ok is now gated
    # on induce_ok AND the stale file is deleted before each induce, so a failed induce -> engine=None
    # -> induction_ok=False (honest), never a mis-attributed stale score. induce_ok is surfaced so an
    # auditor can confirm the scored engine belongs to this cell's own generator/arm.
    induce_ok: bool
    plan_found: bool
    plan_len: int
    reached_levelup: bool
    actions_to_levelup: Optional[int]
    start_hv: Optional[float]
    best_hv: Optional[float]
    hv_progress: Optional[float]
    heldout_accuracy: Optional[float]
    cell_recall: Optional[float]
    goal_predicate_accuracy: Optional[float]
    levelup_positive_recall: Optional[float]
    playbook_injection_mode: str
    n_refinement_rounds: int
    wall_s: float
    error: Optional[str] = None

    def to_row(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


# in-process cache: build_window solves the game offline (slow); do it once per game.
_WINDOW_CACHE: dict[str, Any] = {}


def build_progress_window(game: str) -> Optional[tuple[list, list, int]]:
    """Return (level-up window, full winning trajectory, cell) for ``game`` -- the same
    level-up-straddling induction input the prior A/Bs used (reuses exp5717.build_window),
    cached per game. None if the game cannot be solved to L1 offline."""
    if game in _WINDOW_CACHE:
        return _WINDOW_CACHE[game]
    from carnot.experiment_5717_playbook_exemplars_stall_induction_ab import build_window

    out = build_window(game)
    _WINDOW_CACHE[game] = out
    return out


def _attribution_ok(induce_ok: Any, engine: Any, is_done: Any) -> bool:
    """A cell's induction is attributable to THIS cell's own generator/arm ONLY if its
    proposer.induce SUCCEEDED (induce_ok) AND produced a loadable engine + goal predicate.

    Gating on induce_ok -- not merely "a per-game world_model.py loaded" -- is the fix for the
    stale-engine attribution bug the exp5722 generator-swap adversarial review found: a FAILED
    induce leaves the prior run's engine on disk, which load_engine would otherwise silently
    re-read and score as if this cell had produced it. See run_seeded_progress (which also
    deletes the stale engine before inducing, a complementary guard)."""
    return bool(induce_ok) and engine is not None and is_done is not None


def _levelup_positive_recall(is_level_complete, window: list) -> Optional[float]:
    """Fraction of real level-up transitions in ``window`` whose next_grid the induced
    is_level_complete RECOGNIZES as a win (the win-recognition signal, mirroring
    REQ-ARC-WMTE-5714). None if the window has no real level-up."""
    if is_level_complete is None:
        return None
    hits = total = 0
    for t in window:
        if getattr(t, "level_after", 0) > getattr(t, "level_before", 0):
            total += 1
            try:
                hits += bool(is_level_complete(t.next_grid))
            except Exception:
                pass
    return round(hits / total, 4) if total else None


def _plan_step(step: Any) -> Optional[tuple[int, Any]]:
    """Normalize one plan step to (action_int, data). Handles {"action","data"} dicts
    and (action, data) tuples; returns None on an unrecognized shape."""
    if isinstance(step, dict) and "action" in step:
        return int(step["action"]), step.get("data")
    if isinstance(step, (list, tuple)) and step:
        return int(step[0]), (step[1] if len(step) > 1 else None)
    return None


def _execute_plan_measure(game: str, plan: list, hv_fn) -> dict[str, Any]:
    """Execute ``plan`` against a FRESH real offline env from reset (+ warmup) and measure
    whether a REAL level-up fires and after how many actions, tracking the dense
    hand_verifier goal-distance. The win oracle is the level counter, never a heuristic."""
    from arcengine import GameAction
    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _level_of

    ad = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    if ad is not None and ad.warmup_label is not None:
        frame = ad.apply(env, ad.warmup_label, frame)
    start_level = _level_of(frame)
    start_hv = best_hv = None
    if hv_fn is not None:
        h = hv_fn(getattr(env, "_game", None), frame)
        if h is not None:
            start_hv = best_hv = h
    reached = False
    actions_to = None
    for i, raw in enumerate(plan):
        step = _plan_step(raw)
        if step is None:
            break
        action, data = step
        frame = env.step(getattr(GameAction, f"ACTION{action}"), data=data)
        if frame is None:
            break
        if hv_fn is not None:
            h = hv_fn(getattr(env, "_game", None), frame)
            if h is not None and (best_hv is None or h < best_hv):
                best_hv = h
        if _level_of(frame) > start_level:
            reached = True
            actions_to = i + 1
            break
    hv_progress = None
    if start_hv is not None and best_hv is not None:
        hv_progress = round(max(0.0, (start_hv - best_hv)) / max(abs(start_hv), 1.0), 4)
    return {
        "reached_levelup": reached,
        "actions_to_levelup": actions_to,
        "start_hv": start_hv,
        "best_hv": best_hv,
        "hv_progress": hv_progress,
    }


def run_seeded_progress(
    game: str, arm: str, *, proposer: Any, trial: int, window: list, full_traj: list, cell: int
) -> SeededProgressResult:
    """Seed the LIVE induction with a real level-up window, induce under ``arm`` (guaranteed
    to fire), then execute the induced model's OWN plan against the real env and measure
    whether it reaches a real level-up + how dense the goal-distance progress was.

    Mechanism (DIRECT induce with EXPLICIT injection, the exp5719 pattern + live planner):
    the live `E3AgentPolicy._induce_and_plan` auto-arms playbook injection only on its LLM
    fall-through path, which is skipped when its TTT-CNN tier short-circuits -- so it does NOT
    reliably inject retrieval (observed: retrieval arm reused a stale world_model.py, mode=none).
    Instead we (1) build the retrieval block with the live `_retrieve_playbook_block` (same
    embed+retrieve+format the scored agent uses), (2) set it on the proposer + apply the arm's
    codeonly/think config, (3) call `proposer.induce(...)` DIRECTLY so the fresh LLM induction
    ALWAYS runs with the arm's exact prompt (guaranteed injection, reliable), then (4) load the
    engine and use the LIVE `plan_in_model` planner + execute the plan against the real env.
    This isolates the LLM-induction-content question (what the arms change) and is directly
    comparable to exp5714/5719 (same induce input), upgrading only the downstream metric.
    """
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import (
        E3_DIR,
        WorldModelVerifier,
        load_engine,
        plan_in_model,
        score_goal_predicate_consistency,
    )

    hv_fn = _hand_verifier_fn(game)
    restore = apply_arm(proposer, arm)
    t0 = time.time()
    err: Optional[str] = None
    engine = is_done = None
    plan: list = []
    root_grid = full_traj[0].grid if full_traj else None
    injection_mode = "none"
    induce_ok = False
    try:
        # Build the injection block for the retrieval/static arms via the LIVE retrieval method.
        block: Any = False
        if arm == "retrieval":
            helper = E3AgentPolicy(game, proposer=proposer)
            helper.cell = int(cell)
            b = helper._retrieve_playbook_block(list(full_traj))
            if b:
                block, injection_mode = b, "retrieval"
            else:
                injection_mode = "retrieval_unavailable"  # honest: embedder/index failed
        elif arm == "static":
            block, injection_mode = True, "static"
        proposer.include_playbook_exemplars = block
        # DELETE the prior engine BEFORE inducing so a FAILED induce cannot leave a stale, earlier-run
        # engine on disk for load_engine to silently re-read and mis-attribute to this cell (the
        # exp5722 generator-swap stale-engine attribution bug -- proposer.induce writes world_model.py
        # ONLY on success, and load_engine reads a per-GAME path shared across generators/arms/trials).
        _wm = E3_DIR / game / "world_model.py"
        try:
            _wm.unlink()
        except FileNotFoundError:
            pass
        # Fresh, guaranteed LLM induction with the arm's exact prompt (codeonly/think + injection).
        induce_ok, _detail = proposer.induce(game, list(window), int(cell))
        try:
            engine, is_done = load_engine(game)
        except Exception as exc:
            err = f"load_engine: {type(exc).__name__}: {exc}"[:200]
        if engine is not None and is_done is not None and root_grid is not None:
            plan = list(
                plan_in_model(engine, is_done, root_grid, max_nodes=20000, max_depth=40) or []
            )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        restore()

    heldout = cell_recall = goal_pred = levelup_rec = None
    if engine is not None and window:
        try:
            vr = WorldModelVerifier(window).score(engine)
            heldout, cell_recall = round(float(vr.accuracy), 4), round(float(vr.cell_recall), 4)
        except Exception:
            pass
    if is_done is not None and window:
        try:
            goal_pred = round(float(score_goal_predicate_consistency(is_done, window).accuracy), 4)
        except Exception:
            pass
        levelup_rec = _levelup_positive_recall(is_done, window)

    exe = {
        "reached_levelup": False,
        "actions_to_levelup": None,
        "start_hv": None,
        "best_hv": None,
        "hv_progress": None,
    }
    if plan and err is None:
        try:
            exe = _execute_plan_measure(game, plan, hv_fn)
        except Exception as exc:
            err = (err or "") + f" | execute: {type(exc).__name__}: {exc}"[:150]

    return SeededProgressResult(
        game=game,
        arm=arm,
        trial=trial,
        # Gate on induce_ok (via _attribution_ok): an engine that loaded WITHOUT this cell's induce
        # succeeding would be a stale re-read (now also prevented by the pre-induce unlink above);
        # both guards together mean a True induction_ok is attributable to THIS cell's own generator.
        induction_ok=_attribution_ok(induce_ok, engine, is_done),
        induce_ok=bool(induce_ok),
        plan_found=bool(plan),
        plan_len=len(plan),
        reached_levelup=exe["reached_levelup"],
        actions_to_levelup=exe["actions_to_levelup"],
        start_hv=exe["start_hv"],
        best_hv=exe["best_hv"],
        hv_progress=exe["hv_progress"],
        heldout_accuracy=heldout,
        cell_recall=cell_recall,
        goal_predicate_accuracy=goal_pred,
        levelup_positive_recall=levelup_rec,
        playbook_injection_mode=injection_mode,
        n_refinement_rounds=0,
        wall_s=round(time.time() - t0, 1),
        error=err,
    )


def _sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact sign-test p-value for `wins` vs `losses` (ties dropped).
    Returns None when there are no discordant pairs."""
    n = wins + losses
    if n == 0:
        return None
    from math import comb

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return round(min(1.0, 2.0 * tail), 4)


def paired_by_game(
    rows: list[dict[str, Any]], arm_treat: str, arm_base: str, *, metric: str
) -> dict[str, Any]:
    """Paired comparison where the MATCHED UNIT is the GAME: average ``metric`` over the
    per-game trials for each arm, then pair the two arms by game. This is the lowest-variance
    paired design for the stochastic seeded mode (trials are replicates within a game, the
    game is the thing held constant across arms). ``rows`` are plain dicts (from
    SeededProgressResult.to_row); a game contributes a pair only if BOTH arms have >=1 non-None
    value for ``metric``. Returns mean_delta, per-game wins/ties/losses, exact sign-test p, and
    the outlier-fragility flag."""
    by_game_arm: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        by_game_arm.setdefault((r["game"], r["arm"]), []).append(float(v))

    games = sorted({g for (g, a) in by_game_arm})
    pairs: list[tuple[str, float, float]] = []
    for g in games:
        tv = by_game_arm.get((g, arm_treat))
        bv = by_game_arm.get((g, arm_base))
        if not tv or not bv:
            continue
        pairs.append((g, statistics.mean(tv), statistics.mean(bv)))

    if not pairs:
        return {
            "metric": metric,
            "treat": arm_treat,
            "base": arm_base,
            "n_game_pairs": 0,
            "unit": "game",
            "note": "no comparable game pairs",
        }
    deltas = [t - b for _, t, b in pairs]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses
    mean_delta = statistics.mean(deltas)
    outlier_fragile = False
    if len(deltas) >= 2:
        imax = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
        remaining = [d for i, d in enumerate(deltas) if i != imax]
        if remaining and (
            statistics.mean(remaining) * mean_delta < 0
            or (abs(statistics.mean(remaining)) < 1e-9 < abs(mean_delta))
        ):
            outlier_fragile = True
    return {
        "metric": metric,
        "treat": arm_treat,
        "base": arm_base,
        "unit": "game",
        "n_game_pairs": len(pairs),
        "mean_delta": round(mean_delta, 4),
        "mean_treat": round(statistics.mean([t for _, t, _ in pairs]), 4),
        "mean_base": round(statistics.mean([b for _, _, b in pairs]), 4),
        "wins_treat": wins,
        "ties": ties,
        "losses_treat": losses,
        "sign_test_p": _sign_test_p(wins, losses),
        "outlier_fragile": outlier_fragile,
        "per_game": [
            {"game": g, "treat": round(t, 4), "base": round(b, 4), "delta": round(t - b, 4)}
            for g, t, b in pairs
        ],
    }


def paired_summary(
    results: list[ProgressResult], arm_treat: str, arm_base: str, *, metric: str
) -> dict[str, Any]:
    """Paired within-(game,seed) comparison of ``arm_treat`` vs ``arm_base`` on ``metric``.

    ``metric`` is one of the numeric ProgressResult fields (e.g. ``hv_progress``,
    ``levels_gained``, ``mean_heldout_accuracy``, ``actions_to_first_solve``). Returns
    the paired mean delta, the win/tie/loss counts, an exact sign-test p-value, and an
    ``outlier_fragile`` flag (does dropping the single largest-magnitude pair flip the
    sign of the mean delta) -- the same honesty guards the prior experiments used.
    """
    by_key: dict[tuple[str, int, int], dict[str, ProgressResult]] = {}
    for r in results:
        by_key.setdefault((r.game, r.seed, r.variant), {})[r.arm] = r
    pairs: list[tuple[str, float, float]] = []
    for key, arms in by_key.items():
        rt, rb = arms.get(arm_treat), arms.get(arm_base)
        if rt is None or rb is None:
            continue
        vt, vb = getattr(rt, metric), getattr(rb, metric)
        if vt is None or vb is None:
            continue
        pairs.append((f"{key[0]}#s{key[1]}v{key[2]}", float(vt), float(vb)))

    if not pairs:
        return {
            "metric": metric,
            "treat": arm_treat,
            "base": arm_base,
            "n_pairs": 0,
            "note": "no comparable pairs (metric None or arm missing)",
        }

    deltas = [t - b for _, t, b in pairs]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses
    mean_delta = statistics.mean(deltas)
    outlier_fragile = False
    if len(deltas) >= 2:
        imax = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
        remaining = [d for i, d in enumerate(deltas) if i != imax]
        if remaining and (
            statistics.mean(remaining) * mean_delta < 0
            or abs(statistics.mean(remaining)) < 1e-9 < abs(mean_delta)
        ):
            outlier_fragile = True
    return {
        "metric": metric,
        "treat": arm_treat,
        "base": arm_base,
        "n_pairs": len(pairs),
        "mean_delta": round(mean_delta, 4),
        "mean_treat": round(statistics.mean([t for _, t, _ in pairs]), 4),
        "mean_base": round(statistics.mean([b for _, _, b in pairs]), 4),
        "wins_treat": wins,
        "ties": ties,
        "losses_treat": losses,
        "sign_test_p": _sign_test_p(wins, losses),
        "outlier_fragile": outlier_fragile,
        "per_pair": [
            {"key": k, "treat": round(t, 4), "base": round(b, 4), "delta": round(t - b, 4)}
            for k, t, b in pairs
        ],
    }
