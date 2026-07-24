"""Experiment 5836 -- A/B for the just-explore FRONTIER-DISCIPLINE graft (arXiv:2512.24156).

REQ-ARC-WMTE-5836. Measures whether the two search-ORDER mechanisms grafted in
``carnot.agentic.arc_frontier_discipline`` (a strict GLOBAL priority-tier exhaustion barrier,
and a multi-source frontier-distance gradient) move the LIVE agent's first-win rate and
action efficiency on the offline arcade, against the reference solver ITSELF as a positive
control.

=====================================================================================
WHY THIS EXPERIMENT (and why it is not another non-test)
=====================================================================================
Four recent experiments targeted the live click decision and could not move it, because that
decision is ``lst.pop(0)`` over a statically-sorted list and the learned router meant to
reorder it is coordinate-blind (one identical score for every click target -> a stable
no-op). So the lever under test here is the SEARCH ORDER, not the model.

ARMS
  A  baseline                        -- today's live explorer, both flags off
  B  + tier exhaustion              -- the GLOBAL priority barrier, greedy within-tier draw
  B2 + tier exhaustion, uniform     -- the barrier with the reference's UNIFORM within-tier draw
  C  + distance gradient            -- navigation-nearest frontier target
  D  B + C                          -- both mechanisms
  E  just-explore ITSELF            -- MANDATORY POSITIVE CONTROL (the real reference solver,
                                       via the existing shim over our offline arcade)

Arm B2 is not decoration. Three prior Carnot experiments replaced the reference's uniform
within-tier draw with a Carnot-scored order (argmax / eps-greedy / softmax / percentile-defer)
and EVERY arm lost solves. Carnot's live ``pop(0)`` is exactly such a replacement (the fully
greedy one), so "the greedy draw is itself part of the defect" is a live hypothesis. Without
B2, a null on B would not distinguish "the barrier does not help" from "the barrier only helps
when paired with a uniform draw", and the experiment would be uninterpretable.

Arm E is mandatory and must never be faked. If the reference cannot be loaded, the artifact
records ``{"arm": "E", "ran": false, "reason": ...}`` and the A/B is declared NOT
interpretable, because without a positive control a flat result cannot be distinguished from
a broken harness.

=====================================================================================
THREE PARAMETER CHOICES THAT DEVIATE FROM THE ORIGINAL SPEC -- and the measurements why
=====================================================================================
These are recorded in the artifact as ``spec_deviations`` so the record is auditable rather
than silently "corrected".

1. BUDGET 2000, NOT 200. Budget 200 was measured DEGENERATE: 0 of 25 games reach a first win
   within 200 actions, so every arm scores 0 and no arm can be distinguished from any other.
   Measured first-win costs on the baseline span 20 (lp85) to 1747 (cd82) actions, so a budget
   below ~2000 structurally cannot see most of the signal. ``--budget`` still overrides.

2. SEEDS ARE NOT A REPLICATION AXIS FOR THE DETERMINISTIC ARMS. Arms A/B/C/D and the baseline
   explorer are bit-deterministic -- the global RNG seed provably changes nothing (verified
   across three seeds on three games). Running "3 seeds" on them would produce three identical
   rows and a fake-tight confidence interval. So deterministic arms run ONCE and declare
   ``deterministic: true, n_runs: 1``; only the genuinely stochastic arms (B2, which draws
   uniformly within a tier, and E, whose reference solver draws uniformly by construction) get
   multiple seeds and a real CI.

3. COLOUR PERMUTATION IS A NEGATIVE CONTROL, NOT THE GENERALIZATION AXIS. The original spec
   called for colour-permuted variants "so we measure generalization rather than public-game
   memorization". For THIS agent that premise is empirically void: colour permutation is
   behaviourally INERT (28 of 28 runs across 7 winning games x 4 variants produced byte-
   identical action counts), because the salience sort is geometric plus a colour-COUNT rarity
   term, and a bijective recolour preserves per-colour counts exactly. Running it as specified
   would produce a confident-looking "generalization" number that is the real-game number
   relabelled. So: ``recolor`` is kept as a zero-delta control FOR THE BASELINE-ORDER ARMS
   (A, C) -- where a non-zero delta means colour leakage crept in -- and REFLECTION is used as
   the axis that actually varies (measured: vc33 60 -> 21 actions, cd82 win -> loss).

   IMPORTANT EXCEPTION, found by this experiment's own smoke run: recolour is NOT inert for the
   TIER arms (B / B2 / D). just-explore's tier predicate keys on ABSOLUTE colour values
   (``salient = colour in {6..15}``), which a permutation does not preserve -- measured arm-B
   mean actions-to-first-win 13.5 (real) -> 168.5 (recoloured) on the same two games. This is a
   genuine LIMITATION of the mechanism being grafted, not a harness defect: the barrier buys its
   ordering by assuming a fixed colour->salience convention that a hidden game need not share.
   ``vs_baseline.recolor_control`` records per arm whether inertness was even predicted, so a
   tier arm's delta is neither mis-read as a broken harness nor quietly hidden.
   Reflection mirrors the observation but not directional move actions, so it is a
   self-consistent MIRROR world -- a legitimate re-induction test, but a DIFFERENT difficulty,
   reported separately and never averaged into the real-game headline.

=====================================================================================
WHAT THIS EXPERIMENT DOES NOT DO
=====================================================================================
* No ARC/Kaggle submission. Everything here is the OFFLINE dev twin.
* No LLM. Arms A-D use ``CarnotAgentPolicy(force_explore=True)``, which builds a bare
  ``StepwiseExplorer`` with no proposer parameter at all. NOTE the trap this avoids: passing
  ``proposer=None`` to ``E3AgentPolicy`` does NOT disable the LLM -- it lazily constructs one
  on first use. Hence ``policy="explorer"``, never ``"e3"``.
* No new banked levels. All 25 public games are already recorded fully cleared in the solve
  registry, so this is a SEARCH-DISCIPLINE / EFFICIENCY measurement whose provenance is
  ``development_proxy``; it deliberately does not emit solve-claim fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "python", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXPERIMENT_ID = 5836
ARTIFACT = REPO / "results" / "experiment_5836_frontier_discipline_ab.json"
RANDOM_SEED = 20260724
DEFAULT_BUDGET = 2000
JE_ROOT = Path("/home/ianblenke/arc-sota-refs/arc-agi-3-just-explore")

# Games exposing ACTION6 (click). The barrier and the within-tier draw both act on CLICK
# candidates, so they are STRUCTURALLY INERT on nav-only games -- reporting one pooled number
# over both strata would dilute a real effect into a null. Measured at reset across all 25.
CLICK_GAMES = (
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "ka59", "lf52", "lp85", "m0r0",
    "r11l", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "vc33",
)
NAV_ONLY_GAMES = ("g50t", "ls20", "re86", "tr87", "tu93", "wa30")
ALL_GAMES = tuple(sorted(CLICK_GAMES + NAV_ONLY_GAMES))
# Baseline first-win games at budget 2000 (measured). These are the REGRESSION GUARD: an arm
# that gains a new win but loses one of these has not obviously improved anything.
BASELINE_WIN_GAMES = ("cd82", "lf52", "lp85", "sp80", "su15", "tu93", "vc33")

# (label, variant, reflect). variant=0 is the real game; variant>0 wraps VariantEnv.
CONDITIONS: tuple[tuple[str, int, Optional[int]], ...] = (
    ("real", 0, None),
    ("recolor_negative_control", 1, None),
    ("reflect_axis0", 1, 0),
)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _explorer_policy(game: str, **explorer_kwargs: Any):
    """The LLM-FREE live explorer with explicit frontier-discipline flags.

    Flags are passed as CONSTRUCTOR ARGS, never by mutating module globals or os.environ: all
    arms run in one process, and a global flip would silently leak into every later arm.
    """

    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy

    return CarnotAgentPolicy(game, {}, force_explore=True, **explorer_kwargs)


ARMS: dict[str, dict[str, Any]] = {
    "A": {
        "label": "baseline_live_explorer",
        "kwargs": {"tier_exhaustion": False, "frontier_gradient": False},
        "deterministic": True,
    },
    "B": {
        "label": "tier_exhaustion_greedy_draw",
        "kwargs": {"tier_exhaustion": True, "tier_uniform_random": False,
                   "frontier_gradient": False},
        "deterministic": True,
    },
    "B2": {
        "label": "tier_exhaustion_uniform_draw",
        "kwargs": {"tier_exhaustion": True, "tier_uniform_random": True,
                   "frontier_gradient": False},
        "deterministic": False,
    },
    "C": {
        "label": "frontier_distance_gradient",
        "kwargs": {"tier_exhaustion": False, "frontier_gradient": True},
        "deterministic": True,
    },
    "D": {
        "label": "tier_exhaustion_plus_gradient",
        "kwargs": {"tier_exhaustion": True, "tier_uniform_random": False,
                   "frontier_gradient": True},
        "deterministic": True,
    },
    "E": {
        "label": "just_explore_reference_positive_control",
        "kwargs": {},
        "deterministic": False,
    },
}


# ---------------------------------------------------------------------------
# Arm E -- the just-explore reference, loaded through the existing shim
# ---------------------------------------------------------------------------


def load_just_explore_runner() -> tuple[Optional[Callable[..., dict]], str]:
    """Return ``(runner, reason)``; ``runner`` is None when the reference cannot be run.

    The reference is a vendored MIT clone OUTSIDE the repo, so its absence is a legitimate
    blocked state, not a failure to paper over. We deliberately return a reason string instead
    of raising, because the calling code must record ``{"ran": false, "reason": ...}`` and
    declare the A/B uninterpretable rather than silently dropping the positive control.
    """

    if not (JE_ROOT / "graph_explorer.py").exists():
        return None, f"reference_clone_absent:{JE_ROOT}"
    try:
        import importlib.util
        import types

        import numpy as np

        if str(JE_ROOT) not in sys.path:
            sys.path.insert(0, str(JE_ROOT))
        agents_dir = str(JE_ROOT / "agents")
        if agents_dir not in sys.path:
            sys.path.insert(0, agents_dir)
        # agents/__init__.py eagerly imports langgraph/smolagents, which are not installed and
        # are not needed by the explorer core. Pre-register a bare package with the right
        # __path__ so submodule imports resolve without executing that __init__.
        if not isinstance(sys.modules.get("agents"), types.ModuleType) or not hasattr(
            sys.modules.get("agents"), "__path__"
        ):
            pkg = types.ModuleType("agents")
            pkg.__path__ = [agents_dir]  # type: ignore[attr-defined]
            sys.modules["agents"] = pkg
        if "agents.tracing" not in sys.modules:
            tr = types.ModuleType("agents.tracing")
            tr.trace_agent_session = lambda fn: fn  # type: ignore[attr-defined]
            sys.modules["agents.tracing"] = tr

        shim_path = REPO / "scripts" / "experiments" / "proto_h2h_just_explore.py"
        if not shim_path.exists():
            return None, f"shim_absent:{shim_path}"
        spec = importlib.util.spec_from_file_location("_je_shim_5836", shim_path)
        assert spec is not None and spec.loader is not None
        shim = importlib.util.module_from_spec(spec)
        sys.modules["_je_shim_5836"] = shim
        spec.loader.exec_module(shim)
    except Exception as exc:  # pragma: no cover -- environment-dependent
        return None, f"shim_import_failed:{type(exc).__name__}:{exc}"

    def _run(game: str, *, budget: int, seed: int, variant: int, reflect) -> dict:
        from carnot.agentic import arc_solver_kit as kit

        random.seed(seed)
        np.random.seed(seed % (2**32))
        arc = kit.offline_arcade()
        gid = _resolve_game_id(arc, game)
        env = arc.make(gid, scorecard_id=arc.open_scorecard())
        if variant:
            from carnot.agentic.arc_variant_generator import VariantEnv

            env = VariantEnv(env, game, variant, reflect=reflect)
        agent = shim.OfflineHeuristicAgent(env, gid, budget)
        # Instrument the FIRST level-up's action count. Without this, arm E reports only its
        # budget-bound total (the reference does NOT early-stop on a level-up, so its `actions`
        # is always ~= budget and is therefore NOT comparable to arms A-D, which stop at the
        # first level-up). The action count AT the first level-up IS comparable, and it is the
        # efficiency axis the live scorer squares -- so it is worth capturing rather than
        # reporting an incomparable number and calling it a comparison.
        first_levelup: dict[str, Optional[int]] = {"actions": None}
        _inner_take_action = agent.take_action

        def _counting_take_action(action):
            fd_ = _inner_take_action(action)
            try:
                if first_levelup["actions"] is None and int(getattr(fd_, "score", 0) or 0) > 0:
                    first_levelup["actions"] = int(getattr(agent, "action_counter", 0) or 0)
            except Exception:
                pass
            return fd_

        agent.take_action = _counting_take_action  # type: ignore[method-assign]
        t0 = time.time()
        try:
            agent.main()
        except Exception:
            # The reference terminates by raising in some end states; its max_score is still
            # the honest measurement, so we keep it rather than discarding the run.
            pass
        return {
            "reached": int(agent.max_score),
            "levels": int(agent.max_score),
            "actions": int(getattr(agent, "action_counter", 0) or 0),
            "actions_to_first_levelup": first_levelup["actions"],
            "duration_s": round(time.time() - t0, 3),
        }

    return _run, "ok"


def _resolve_game_id(arc: Any, game: str) -> str:
    for e in arc.get_environments():
        gid = str(getattr(e, "game_id", ""))
        if gid.startswith(game):
            return gid
    return game


# ---------------------------------------------------------------------------
# One cell = (arm, game, condition, seed)
# ---------------------------------------------------------------------------


def run_cell(
    arm: str,
    game: str,
    *,
    budget: int,
    seed: int,
    variant: int,
    reflect: Optional[int],
    je_runner: Optional[Callable[..., dict]] = None,
) -> dict:
    """Run one (arm, game, condition, seed) cell and return its measured row."""

    import arc_leaderboard_eval as lb

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass

    if arm == "E":
        if je_runner is None:
            return {"arm": arm, "game": game, "ran": False, "reason": "no_reference_runner"}
        out = je_runner(game, budget=budget, seed=seed, variant=variant, reflect=reflect)
        return {
            "arm": arm, "game": game, "seed": seed, "ran": True,
            "levels": int(out["levels"]), "reached": int(out["reached"]),
            "actions": int(out["actions"]),
            "actions_to_first_levelup": out.get("actions_to_first_levelup"),
            # NOT comparable to arms A-D's `actions`: the reference does not early-stop on a
            # level-up, so its total is budget-bound. Its LEVELS and its
            # actions_to_first_levelup are the comparable quantities.
            "actions_total_is_budget_bound": True,
            "states_expanded": None, "efficiency": None,
            "duration_s": out["duration_s"],
            "frontier_discipline": None,
        }

    t0 = time.time()
    # Thread the SEED into the explorer's own RNG. Without this the within-tier uniform draw
    # runs off a fixed constructor default, so the "seeds" of arm B2 would produce byte-identical
    # rows and its confidence interval would be fabricated width-zero certainty. (Caught by the
    # smoke run: B2's two seeds were initially identical.) Harmless for the deterministic arms,
    # which never draw from this RNG.
    policy = _explorer_policy(game, frontier_discipline_seed=seed, **ARMS[arm]["kwargs"])
    r = lb.run_game(game, policy, budget=budget, variant=variant, reflect=reflect)
    explorer = getattr(policy, "explorer", None)
    fd_diag = None
    if explorer is not None and hasattr(explorer, "frontier_discipline_diagnostics"):
        fd_diag = explorer.frontier_discipline_diagnostics()
    return {
        "arm": arm, "game": game, "seed": seed, "ran": True,
        "levels": int(r["levels"]), "reached": int(r["reached"]), "actions": int(r["actions"]),
        "actions_to_first_levelup": r["actions_to_first_levelup"],
        # states_expanded = distinct graph nodes the explorer built. The search-effort axis:
        # a discipline that reaches the same level with fewer expanded states is a real win
        # even when the binary win count is flat.
        "states_expanded": (len(explorer.graph) if explorer is not None else None),
        "efficiency": r["efficiency"],
        "duration_s": round(time.time() - t0, 3),
        "frontier_discipline": fd_diag,
    }


def _seeds_for(arm: str, n_seeds: int) -> list[int]:
    """Deterministic arms get ONE run. See spec-deviation 2: seeding them fakes replication."""

    if ARMS[arm]["deterministic"]:
        return [RANDOM_SEED]
    return [RANDOM_SEED + i for i in range(max(1, int(n_seeds)))]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean_ci95(values: Sequence[float]) -> dict:
    """Mean with a normal-approximation 95% CI. ``ci95`` is None when n < 2 -- an honest
    "no interval available" rather than a zero-width interval that would look like certainty."""

    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    if len(vals) < 2:
        return {"n": 1, "mean": vals[0], "ci95": None}
    sd = statistics.stdev(vals)
    half = 1.96 * sd / (len(vals) ** 0.5)
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
            "sd": round(sd, 4), "ci95": [round(statistics.fmean(vals) - half, 4),
                                         round(statistics.fmean(vals) + half, 4)]}


def aggregate(rows: Sequence[dict], games: Sequence[str]) -> dict:
    """Per (arm, condition) aggregates, stratified by action vocabulary."""

    out: dict[str, Any] = {}
    click = set(CLICK_GAMES)
    for arm in ARMS:
        for cond, _v, _r in CONDITIONS:
            cell = [r for r in rows if r.get("arm") == arm and r.get("condition") == cond
                    and r.get("ran")]
            if not cell:
                continue
            # Per game: won if ANY seed reached a level (existence), plus the per-seed rate.
            by_game: dict[str, list[dict]] = {}
            for r in cell:
                by_game.setdefault(r["game"], []).append(r)
            any_win = sorted(g for g, rs in by_game.items() if any(x["levels"] > 0 for x in rs))
            per_seed_rates = []
            seeds = sorted({r["seed"] for r in cell})
            for s in seeds:
                srows = [r for r in cell if r["seed"] == s]
                if srows:
                    per_seed_rates.append(sum(1 for r in srows if r["levels"] > 0) / len(srows))
            eff_costs = [r["actions_to_first_levelup"] for r in cell
                         if r.get("actions_to_first_levelup")]
            states = [r["states_expanded"] for r in cell if r.get("states_expanded") is not None]
            out[f"{arm}|{cond}"] = {
                "arm": arm, "arm_label": ARMS[arm]["label"], "condition": cond,
                "deterministic_arm": bool(ARMS[arm]["deterministic"]),
                "n_runs_per_game": len(seeds),
                "games_measured": len(by_game),
                "games_won_any_seed": any_win,
                "n_games_won_any_seed": len(any_win),
                "first_win_rate_over_seeds": _mean_ci95(per_seed_rates),
                "click_stratum_wins": sorted(g for g in any_win if g in click),
                "nav_stratum_wins": sorted(g for g in any_win if g not in click),
                "levels_banked_total": sum(max(x["levels"] for x in rs) for rs in by_game.values()),
                "mean_actions_to_first_win": _mean_ci95(eff_costs),
                "mean_states_expanded": _mean_ci95(states),
            }
    return out


def compare_to_baseline(agg: dict, games: Sequence[str]) -> dict:
    """Arm-vs-A deltas on the REAL condition, plus the regression guard and the control check."""

    base_key = "A|real"
    base = agg.get(base_key)
    if base is None:
        return {"available": False, "reason": "baseline_arm_A_real_missing"}
    base_wins = set(base["games_won_any_seed"])
    guard = [g for g in BASELINE_WIN_GAMES if g in set(games)]
    out: dict[str, Any] = {"available": True, "baseline_wins": sorted(base_wins),
                           "regression_guard_games": guard}
    for arm in ARMS:
        key = f"{arm}|real"
        if key not in agg or arm == "A":
            continue
        wins = set(agg[key]["games_won_any_seed"])
        lost = sorted(g for g in base_wins if g not in wins)
        out[arm] = {
            "new_wins": sorted(wins - base_wins),
            "lost_wins": lost,
            "regressed_baseline_win": bool([g for g in lost if g in guard]),
            "n_win_delta": len(wins) - len(base_wins),
        }
    # RECOLOUR CONTROL -- and an important asymmetry found by this experiment's own smoke run.
    #
    # For the BASELINE candidate order, recolour is provably inert: the salience key is geometric
    # plus a colour-COUNT rarity term, and a bijection preserves per-colour counts exactly. So on
    # arms A and C, recolour SHOULD be a zero-delta control, and a delta there means something
    # introduced a colour dependence (a real defect worth investigating).
    #
    # For the TIER arms (B / B2 / D) recolour is NOT a null control, because just-explore's tier
    # predicate keys on ABSOLUTE colour values (`salient = colour in {6..15}`), which a
    # permutation does NOT preserve. Measured in the smoke run: arm B's mean actions-to-first-win
    # went 13.5 (real) -> 168.5 (recoloured) on the same two games. That is a genuine property of
    # the grafted mechanism, not a harness bug -- and it is a real LIMITATION to report: the
    # barrier buys its ordering by assuming a fixed colour->salience convention, which a hidden
    # game need not share. `expected_inert` records which arms the zero-delta assertion actually
    # applies to, so a tier arm's delta is never mis-read as a harness failure and, equally
    # important, is never hidden.
    control: dict[str, Any] = {}
    colour_dependent = {a for a in ARMS if ARMS[a]["kwargs"].get("tier_exhaustion")} | {"E"}
    for arm in ARMS:
        rk, ck = f"{arm}|real", f"{arm}|recolor_negative_control"
        if rk in agg and ck in agg:
            same_wins = agg[rk]["games_won_any_seed"] == agg[ck]["games_won_any_seed"]
            real_cost = agg[rk]["mean_actions_to_first_win"]["mean"]
            rec_cost = agg[ck]["mean_actions_to_first_win"]["mean"]
            same_cost = real_cost == rec_cost
            expected_inert = arm not in colour_dependent
            control[arm] = {
                "real_wins": agg[rk]["n_games_won_any_seed"],
                "recolor_wins": agg[ck]["n_games_won_any_seed"],
                "same_win_set": same_wins,
                "mean_actions_real": real_cost,
                "mean_actions_recolor": rec_cost,
                "same_cost": same_cost,
                "expected_inert": expected_inert,
                # A control VIOLATION only for the arms where inertness is actually predicted.
                "control_violated": bool(expected_inert and not (same_wins and same_cost)),
                "colour_dependent_by_construction": not expected_inert,
                "note": (
                    "tier predicate keys on absolute colour values ({6..15} salient), which a "
                    "permutation does not preserve -- a real limitation of the mechanism"
                    if not expected_inert
                    else "baseline salience key is permutation-invariant -> zero delta expected"
                ),
            }
    out["recolor_control"] = control
    out["recolor_control_violations"] = sorted(
        a for a, v in control.items() if v["control_violated"]
    )
    return out


# ---------------------------------------------------------------------------
# Reproduction gate
# ---------------------------------------------------------------------------


def replay_validate(
    rows: Sequence[dict],
    *,
    budget: int,
    je_runner: Optional[Callable[..., dict]] = None,
    limit: int = 6,
) -> dict:
    """Re-run winning cells and confirm the same level is reached.

    HONEST SCOPE NOTE (this is deliberately NOT called a kit.reproduce gate):
    ``arc_solver_kit.reproduce`` replays a banked list of string action LABELS, which the live
    explorer never produces -- it emits ``(action_id, data)`` tuples. So reproduce() is not
    applicable to a live-explorer trajectory and claiming it ran would be false. What IS a
    genuine, checkable gate for this offline harness is re-execution: run the identical
    (arm, game, condition, seed) cell a second time in a fresh process-local env and assert
    the same level is reached. That is what this does. Level-ups themselves are read from the
    env's own ``levels_completed``, i.e. frame truth, not a self-report.
    """

    wins = [r for r in rows if r.get("ran") and r.get("levels", 0) > 0][: max(0, int(limit))]
    checks = []
    for r in wins:
        cond = next((c for c in CONDITIONS if c[0] == r["condition"]), None)
        if cond is None:
            continue
        again = run_cell(r["arm"], r["game"], budget=budget, seed=r["seed"],
                         variant=cond[1], reflect=cond[2], je_runner=je_runner)
        checks.append({
            "arm": r["arm"], "game": r["game"], "condition": r["condition"], "seed": r["seed"],
            "levels_first": r["levels"], "levels_replay": again.get("levels"),
            "reproduced": bool(again.get("levels") == r["levels"]),
        })
    n_ok = sum(1 for c in checks if c["reproduced"])
    return {
        "method": "re_execution_of_the_same_cell_not_kit_reproduce",
        "why_not_kit_reproduce": (
            "kit.reproduce consumes banked string action LABELS; the live explorer emits "
            "(action_id, data) tuples, so it is not applicable to this trajectory shape"
        ),
        "level_source": "env.levels_completed (frame truth, not a self-report)",
        "n_checked": len(checks), "n_reproduced": n_ok,
        "all_reproduced": bool(checks) and n_ok == len(checks),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def check_preconditions() -> list[dict]:
    """PRECONDITIONS, checked BEFORE any measurement (CLAUDE.md Pre-Launch Preconditions).

    Every entry is a real observation, not an assumption. A missing resource yields a
    ``blocked_*`` verdict and no numbers, rather than a plausible-looking fabricated run.
    """

    out: list[dict] = []

    def _add(resource: str, fn: Callable[[], bool], detail: str = "") -> bool:
        try:
            ok = bool(fn())
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}:{exc}"
        out.append({"resource": resource, "available": ok, "detail": detail})
        return ok

    _add("offline_arcade_environment_files", lambda: (REPO / "environment_files").is_dir(),
         str(REPO / "environment_files"))
    _add("arc_leaderboard_eval_harness", lambda: (REPO / "scripts" / "arc_leaderboard_eval.py").is_file())
    _add("frontier_discipline_module", lambda: __import__(
        "carnot.agentic.arc_frontier_discipline", fromlist=["TierExhaustionPolicy"]
    ) is not None)
    _add("live_explorer_flags_wired", lambda: all(
        hasattr(
            __import__("carnot.agentic.arc_competition_agent",
                       fromlist=["StepwiseExplorer"]).StepwiseExplorer(),
            attr,
        )
        for attr in ("tier_exhaustion_enabled", "frontier_gradient_enabled")
    ))
    _add("just_explore_reference_clone", lambda: (JE_ROOT / "graph_explorer.py").is_file(),
         str(JE_ROOT))
    _add("variant_generator", lambda: __import__(
        "carnot.agentic.arc_variant_generator", fromlist=["VariantEnv"]
    ) is not None)
    # Explicitly asserted ABSENT: no LLM is loaded on any arm. Arms A-D use
    # CarnotAgentPolicy(force_explore=True), which has no proposer parameter at all; arm E is
    # the reference solver, which uses no model by design.
    out.append({"resource": "llm_proposer_deliberately_absent", "available": True,
                "detail": "arms A-D use CarnotAgentPolicy(force_explore=True); arm E uses no model"})
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _reproducibility_checksum(payload: dict) -> str:
    """Content hash over the measured rows + the run configuration.

    Anchors the artifact to its inputs: a later replication that produces different rows
    produces a different checksum, so silent drift in the corpus, the arms, or the budget
    cannot masquerade as the same experiment.
    """

    blob = json.dumps(
        {"rows": payload.get("per_cell_rows"), "config": payload.get("config")},
        sort_keys=True, default=str,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Self-declared terminal state, terminal-prefixed so the conductor's reconciler cannot "
        "misclassify it from a nuance word in the descriptive tail."
    ),
    "inference_substrate": (
        "Declares WHAT compute actually ran, so the fabrication linter applies the right "
        "duration floor instead of assuming a 60s live-LLM floor. Here: the live agent takes "
        "real actions against the offline arcade with NO LLM loaded on any arm."
    ),
    "duration_s": (
        "Real compute takes wall-clock time; a missing or implausibly short duration is the "
        "load-bearing fabrication signal."
    ),
    "random_seed": (
        "Determinism is the precondition for reproducibility; without a seed no third party "
        "can re-run this and confirm or refute it."
    ),
    "reproducibility_checksum": (
        "Content hash of the measured rows + config; catches silent corpus/arm/budget drift "
        "between this artifact and any replication attempt."
    ),
    "preconditions_checked": (
        "Records WHICH resources were verified before measuring, pre-empting the failure mode "
        "where the agent silently lacked a resource and synthesized a passing artifact."
    ),
    "verifier_is_oracle": (
        "True iff the thing judging success IS the executable oracle that defines correctness. "
        "Here TRUE and disclosed: level-ups are read from the environment's own "
        "levels_completed. That makes this an execution-grounded SEARCH-EFFICIENCY measurement, "
        "NOT an oracle-distinct verifier-moat claim, and it is therefore not headline-eligible "
        "as a moat result."
    ),
    "solve_provenance": (
        "development_proxy: the offline dev twin, not the live agent self-discovering a hidden "
        "game. Banks no new levels -- all 25 public games are already registry-cleared."
    ),
    "positive_control_ran": (
        "Without the reference solver as a positive control, a flat A/B cannot be distinguished "
        "from a broken harness, so the A/B is declared uninterpretable when this is false."
    ),
    "arms": (
        "Names every measured arm including the uniform-within-tier arm; omitting that arm "
        "would leave a null on the barrier confounded with the greedy-draw hypothesis."
    ),
    "spec_deviations": (
        "Records where measured reality forced a departure from the original parameters "
        "(budget 200 degenerate, seeds non-replicating, recolour inert), so the record is "
        "auditable instead of silently corrected."
    ),
    "states_expanded": (
        "Search effort. A discipline that reaches the same level with fewer expanded states is "
        "a real efficiency win even when the binary win count is flat."
    ),
}


def run(
    *,
    games: Sequence[str] = ALL_GAMES,
    arms: Sequence[str] = tuple(ARMS),
    conditions: Sequence[str] = tuple(c[0] for c in CONDITIONS),
    budget: int = DEFAULT_BUDGET,
    n_seeds: int = 3,
    artifact_path: Optional[Path] = None,
    replay_limit: int = 6,
) -> dict:
    """Execute the A/B and write the artifact. Returns the artifact dict."""

    t0 = time.time()
    pre = check_preconditions()
    blocking = [p for p in pre if not p["available"]
                and p["resource"] in ("offline_arcade_environment_files",
                                      "arc_leaderboard_eval_harness",
                                      "frontier_discipline_module",
                                      "live_explorer_flags_wired")]
    out_path = Path(artifact_path) if artifact_path else ARTIFACT
    if blocking:
        art = {
            "experiment": EXPERIMENT_ID,
            "experiment_id": EXPERIMENT_ID,
            "honest_verdict": "blocked_" + ",".join(p["resource"] for p in blocking),
            "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "preconditions_checked": pre,
            "duration_s": round(time.time() - t0, 4),
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(art, indent=2, default=str))
        return art

    je_runner, je_reason = load_just_explore_runner()
    cond_specs = [c for c in CONDITIONS if c[0] in set(conditions)]
    rows: list[dict] = []
    for arm in arms:
        if arm not in ARMS:
            continue
        if arm == "E" and je_runner is None:
            rows.append({"arm": "E", "ran": False, "reason": je_reason})
            continue
        for cond_label, variant, reflect in cond_specs:
            for seed in _seeds_for(arm, n_seeds):
                for game in games:
                    try:
                        row = run_cell(arm, game, budget=budget, seed=seed, variant=variant,
                                       reflect=reflect, je_runner=je_runner)
                    except Exception as exc:
                        # Record the error rather than silently dropping a game: a dropped game
                        # changes the denominator and quietly biases every rate.
                        row = {"arm": arm, "game": game, "seed": seed, "ran": False,
                               "reason": f"{type(exc).__name__}:{exc}"}
                    row["condition"] = cond_label
                    rows.append(row)

    errored = [r for r in rows if not r.get("ran")]
    agg = aggregate(rows, games)
    cmp_ = compare_to_baseline(agg, games)
    repro = replay_validate(rows, budget=budget, je_runner=je_runner, limit=replay_limit)

    positive_control_ran = any(r.get("arm") == "E" and r.get("ran") for r in rows)
    # An A/B is interpretable only if the positive control ran AND most cells actually produced a
    # measurement. A run where a third of the cells errored is a broken harness reporting numbers,
    # not a result -- an earlier smoke of this very file silently produced "complete" with 36 of
    # 72 cells errored on a signature mismatch, which is exactly the failure this gate closes.
    error_rate = (len(errored) / len(rows)) if rows else 1.0
    interpretable = bool(positive_control_ran and error_rate <= 0.05)

    verdict = "complete_frontier_discipline_ab_measured"
    if not positive_control_ran:
        verdict = "complete_frontier_discipline_ab_measured_but_uninterpretable_no_positive_control"
    elif error_rate > 0.05:
        verdict = (
            "complete_frontier_discipline_ab_measured_but_uninterpretable_"
            f"errored_cell_rate_{error_rate:.2f}"
        )

    config = {
        "games": list(games), "arms": list(arms), "conditions": [c[0] for c in cond_specs],
        "budget_actions_per_game": int(budget), "n_seeds_stochastic_arms": int(n_seeds),
        "llm_disabled": True, "policy_kind": "explorer_force_explore_no_proposer",
    }
    art: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "title": "Frontier-discipline A/B: just-explore tier exhaustion + distance gradient",
        "requirement": "REQ-ARC-WMTE-5836",
        "reference": "arXiv:2512.24156 (just-explore, ARC-AGI-3 Preview private leaderboard 3rd)",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": True,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - t0, 3),
        "preconditions_checked": pre,
        "config": config,
        "arms": {k: {"label": v["label"], "kwargs": v["kwargs"],
                     "deterministic": v["deterministic"]} for k, v in ARMS.items()},
        "positive_control_ran": positive_control_ran,
        "positive_control_reason": je_reason,
        "ab_interpretable": interpretable,
        "per_cell_rows": rows,
        "aggregates": agg,
        "vs_baseline": cmp_,
        "reproduction_gate": repro,
        "n_errored_cells": len(errored),
        "errored_cell_rate": round(error_rate, 4),
        "errored_cells": errored[:40],
        "action_vocabulary_strata": {"click_games": list(CLICK_GAMES),
                                     "nav_only_games": list(NAV_ONLY_GAMES)},
        "spec_deviations": [
            {"spec": "budget 200",
             "actual": f"budget {budget} (default {DEFAULT_BUDGET})",
             "why": "budget 200 measured 0/25 first-wins -> no arm distinguishable; measured "
                    "first-win costs span 20 (lp85) to 1747 (cd82) actions"},
            {"spec": ">=3 seeds on every arm",
             "actual": "1 run for the bit-deterministic arms (A/B/C/D); >=3 seeds only for the "
                       "stochastic arms (B2 uniform draw, E reference)",
             "why": "the explorer is bit-deterministic (verified across 3 seeds x 3 games), so "
                    "seeding it produces identical rows and a fake-tight CI"},
            {"spec": "colour-permuted variants measure generalization",
             "actual": "recolour kept as a ZERO-DELTA NEGATIVE CONTROL; reflection used as the "
                       "generalization axis, reported separately",
             "why": "recolour is behaviourally inert for the BASELINE candidate order (28/28 "
                    "byte-identical runs) because that salience key is geometric plus a "
                    "permutation-invariant colour-COUNT rarity term. NOTE the exception this "
                    "experiment's own smoke run found: recolour is NOT inert for the TIER arms, "
                    "because just-explore's tier predicate keys on ABSOLUTE colour values "
                    "(salient = colour in {6..15}); measured arm-B mean actions-to-first-win "
                    "13.5 (real) -> 168.5 (recoloured). That is a real LIMITATION of the grafted "
                    "mechanism (it assumes a fixed colour->salience convention a hidden game need "
                    "not share), reported in vs_baseline.recolor_control, not a harness bug"},
            {"spec": "solves validated with kit.reproduce",
             "actual": "re-execution of the same cell; level-ups read from env.levels_completed",
             "why": "kit.reproduce consumes banked string action LABELS, which the live explorer "
                    "never emits -- claiming it ran would be false"},
        ],
        "field_provenance": {k: {"principle": v} for k, v in FIELD_PRINCIPLES.items()},
        "caveats": [
            "Arm E is NOT equal-env-step with arms A-D: Carnot's RESET-replay navigation does "
            "uncounted env.step+reset per expansion (measured ~4.2-4.4x more real env "
            "interaction at nominal-equal budget), which HANDICAPS arm E. A just-explore win "
            "here is therefore a conservative LOWER bound.",
            "Reflection mirrors the observation but not directional move actions, so a nav game "
            "becomes a self-consistent MIRROR world -- a legitimate re-induction test but a "
            "DIFFERENT difficulty. Never averaged into the real-game headline.",
            "This experiment banks NO new levels and makes no solve claim: all 25 public games "
            "are already recorded fully cleared in ops/arc_solve_registry.yaml.",
            "The 5-tier predicate is shared with the already-nulled CARNOT_ARC_TIER_SCHEDULE "
            "sort by design, so a difference measured here is a difference in the DISCIPLINE "
            "(global exhaustion) and not in the tier predicate.",
        ],
    }
    art["reproducibility_checksum"] = _reproducibility_checksum(art)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art, indent=2, default=str))
    return art


def _arg(argv: Sequence[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv else default


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    budget = int(_arg(argv, "--budget", str(DEFAULT_BUDGET)))
    n_seeds = int(_arg(argv, "--seeds", "3"))
    games_arg = _arg(argv, "--games", "")
    arms_arg = _arg(argv, "--arms", "")
    conds_arg = _arg(argv, "--conditions", "")
    out = _arg(argv, "--out", "")
    games = tuple(g.strip() for g in games_arg.split(",") if g.strip()) or ALL_GAMES
    arms = tuple(a.strip() for a in arms_arg.split(",") if a.strip()) or tuple(ARMS)
    conds = tuple(c.strip() for c in conds_arg.split(",") if c.strip()) or tuple(
        c[0] for c in CONDITIONS
    )
    art = run(games=games, arms=arms, conditions=conds, budget=budget, n_seeds=n_seeds,
              artifact_path=Path(out) if out else None,
              replay_limit=int(_arg(argv, "--replay-limit", "6")))
    print(json.dumps({
        "honest_verdict": art["honest_verdict"],
        "ab_interpretable": art.get("ab_interpretable"),
        "positive_control_ran": art.get("positive_control_ran"),
        "duration_s": art["duration_s"],
        "n_errored_cells": art.get("n_errored_cells"),
        "aggregates": {k: {"n_wins": v["n_games_won_any_seed"],
                           "wins": v["games_won_any_seed"],
                           "mean_actions_to_first_win": v["mean_actions_to_first_win"]["mean"],
                           "mean_states_expanded": v["mean_states_expanded"]["mean"]}
                       for k, v in art.get("aggregates", {}).items()},
        "vs_baseline": art.get("vs_baseline"),
        "reproduction_gate": {k: art["reproduction_gate"][k]
                              for k in ("n_checked", "n_reproduced", "all_reproduced")},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
