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

B2's draw is UNRESTRICTED-uniform over every tier-admitted row, which is what the reference
does (``random.choice(untested_edges)`` accumulated over groups ``0..active_group``, no top-k).
This is worth stating because it was WRONG until 2026-07-24: the wiring passed the unrelated
hybrid-diversity knob's top-8 default, so B2 was a top-8 draw AND was silently coupled to a
foreign env var. On r11l a node carries ~34 candidates, so top-8 vs all-34 is a materially
different distribution -- i.e. the arm added specifically to de-confound the greedy-draw
hypothesis was not testing the reference's draw at all.

=====================================================================================
WHAT THE HEADLINE OF THIS EXPERIMENT IS (and what it is NOT)
=====================================================================================
The decision-relevant quantity is CAPABILITY: ``new_wins_vs_baseline`` for the grafted arms,
against ``positive_control_new_wins`` for the reference. An efficiency delta on games the
baseline already solves does NOT justify flipping a flag, and reporting it first buries the
result. The efficiency axis is reported strictly after the capability numbers, and its
inferential statistic is the PAIRED per-game comparison (``paired_efficiency_vs_baseline``),
never the pooled-across-games mean whose interval is dominated by between-game difficulty.

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
# The SECOND requirement this module hosts. A run that includes the sampler arms (F/F1) is a
# REQ-ARC-WMTE-5950 measurement and must declare that identity, not 5836's -- see the identity
# derivation in `run()`.
CLICK_PIXEL_EXPERIMENT_ID = 5950
ARTIFACT = REPO / "results" / "experiment_5836_frontier_discipline_ab.json"
RANDOM_SEED = 20260724
DEFAULT_BUDGET = 2000
JE_ROOT = Path("/home/ianblenke/arc-sota-refs/arc-agi-3-just-explore")

# Games exposing ACTION6 (click). The barrier and the within-tier draw both act on CLICK
# candidates, so they are STRUCTURALLY INERT on nav-only games -- reporting one pooled number
# over both strata would dilute a real effect into a null. Measured at reset across all 25.
CLICK_GAMES = (
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "dc22",
    "ft09",
    "ka59",
    "lf52",
    "lp85",
    "m0r0",
    "r11l",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "tn36",
    "vc33",
)
NAV_ONLY_GAMES = ("g50t", "ls20", "re86", "tr87", "tu93", "wa30")
ALL_GAMES = tuple(sorted(CLICK_GAMES + NAV_ONLY_GAMES))
# REGRESSION-GUARD FALLBACK ONLY -- *ASSUMED*, NOT MEASURED BY THIS FILE.
#
# This list was previously commented "measured at budget 2000" with no artifact behind it, which
# an adversarial review correctly flagged as fabrication-adjacent (the nearest real artifact,
# results/arc3_offline_eval_carnot.json, is a DIFFERENT policy at a different budget and reports
# a different 15-game win set, so it is NOT the source). The guard is now DERIVED AT RUNTIME
# from arm A's own measured rows in this very run -- see `_guard_games_from_rows` -- which is
# strictly better anyway: the guard should be self-consistent with the baseline it guards. This
# tuple survives only as the fallback used when arm A was not run in a given invocation (e.g.
# `--arms B,E`), and as the input to the power-ceiling calculation when no arm-A rows exist.
# TODO: delete once every invocation path is guaranteed to include arm A.
ASSUMED_BASELINE_WIN_GAMES = ("cd82", "lf52", "lp85", "sp80", "su15", "tu93", "vc33")


def _guard_games_from_rows(rows: Sequence[dict]) -> tuple[list[str], str]:
    """The regression guard, derived from arm A's MEASURED wins in this run.

    Returns ``(games, provenance)``. Falls back to ``ASSUMED_BASELINE_WIN_GAMES`` (clearly
    labelled as such in the artifact) only when this run has no arm-A real-condition rows to
    derive from, because a guard is still better than no guard when someone runs a subset.
    """

    won = sorted(
        {
            r["game"]
            for r in rows
            if r.get("arm") == "A"
            and r.get("ran")
            and r.get("condition") == "real"
            and int(r.get("levels") or 0) > 0
        }
    )
    if won:
        return won, "derived_from_arm_A_real_condition_rows_in_this_run"
    return list(ASSUMED_BASELINE_WIN_GAMES), "ASSUMED_fallback_arm_A_not_measured_in_this_run"


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
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": False,
            "frontier_gradient": False,
        },
        "deterministic": True,
    },
    "B2": {
        "label": "tier_exhaustion_uniform_draw",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": True,
            "frontier_gradient": False,
        },
        "deterministic": False,
    },
    # 2026-07-25: isolates the runtime click-vocabulary gate. B2 now inherits the gate from its
    # default (SUBMITTED_FRONTIER_TIER_CLICK_VOCAB_ONLY_ENABLED=True); this arm turns it OFF, so
    # B2 - B2_nofix attributes any change to the gate alone rather than to drift in the barrier.
    "B2_nofix": {
        "label": "tier_exhaustion_uniform_draw_no_click_vocab_gate",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": True,
            "tier_click_vocab_only": False,
            "frontier_gradient": False,
        },
        "deterministic": False,
    },
    # REQ-ARC-WMTE-5950 (2026-07-25): per-object CLICK-PIXEL SAMPLING, the just-explore
    # GENERATION rule -- a uniform random pixel OF the chosen object instead of that
    # object's truncated centroid.
    #
    # THE CONTROL FOR THESE TWO ARMS IS B2, NOT A. Arm A pins tier_exhaustion=False and
    # frontier_gradient=False as explicit constructor kwargs, and _fd_gate gives an explicit
    # kwarg precedence over both the env override and the SUBMITTED_* default -- so after the
    # 2026-07-25 flip of tier_exhaustion / tier_uniform_random / tier_click_vocab_only, arm A
    # is the PRE-FLIP agent, not the live one. B2 pins exactly the flipped configuration, so
    # F - B2 isolates the sampler against what actually ships today. (A is still worth
    # running: it is what the historical rows in this artifact's own record were measured on.)
    #
    # F vs F1 separates the mechanism's two halves, which is the whole reason there are two:
    #   F1 -- CORRECT the coordinate only (redraw budget 1 = one shot per object, exactly as
    #         today, but on a real member pixel instead of a possibly-off-object centroid).
    #   F  -- also permit bounded WITH-REPLACEMENT revisiting (budget 3).
    # The offline isolation that motivated this predicted the difference matters: a fixed
    # member pixel improved the median but produced the WORST tail of any arm, while uniform
    # redraws compressed the tail. If F beats F1 here, that reproduces; if not, the cheaper
    # F1 is the flip candidate.
    "F": {
        "label": "click_pixel_sampling_with_bounded_redraw_on_live_config",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": True,
            "frontier_gradient": False,
            "click_pixel_sampling": True,
            "click_pixel_redraw_budget": 3,
        },
        "deterministic": False,
    },
    "F1": {
        "label": "click_pixel_sampling_one_shot_on_live_config",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": True,
            "frontier_gradient": False,
            "click_pixel_sampling": True,
            "click_pixel_redraw_budget": 1,
        },
        "deterministic": False,
    },
    # REQ-ARC-WMTE-5960 (2026-07-25): the REPAIRED, orientation-complete HUD status-bar
    # detector, and the same detector with its runtime collapse guard armed.
    #
    # THE CONTROL FOR THESE TWO ARMS IS B2, NOT A -- identical reasoning to arms F/F1 above:
    # arm A pins tier_exhaustion=False / frontier_gradient=False as explicit constructor
    # kwargs and _fd_gate ranks an explicit kwarg above the SUBMITTED_* default, so after the
    # 2026-07-25 flip arm A is the PRE-flip agent. B2 pins exactly the flipped configuration,
    # so G - B2 isolates the detector against what actually ships today.
    #
    # WHAT THE DETECTOR CHANGES, precisely: `_compute_hud_mask_from_frame` switches to the
    # orientation-complete predicate in `carnot.agentic.arc_hud_bar_detector`, which ORs IN the
    # currently-shipped `is_status_bar_like` rule -- so the mask is a strict SUPERSET of today's
    # by construction (verified on all 25 public games: 0 games lose a cell). Nothing else
    # moves: candidate ranking, the tier map, the trained click-target feature table and the
    # click sampler are all untouched, because the repair is a NEW predicate consumed only by
    # the mask path rather than a widening of the shared one.
    #
    # G vs G2 separates the two halves, which is the whole reason there are two:
    #   G  -- detection only. Measures the DEDUP effect (the r11l pathology) with no guard.
    #   G2 -- detection + the runtime collapse guard, which HARD-REFUSES (un-masks) any node
    #         proven to alias two behaviourally distinct states. G2 - G is the guard's cost:
    #         if the guard fires on games G already wins, the mask is over-broad there; if G2
    #         matches G with zero refusals, the mask is clean on this corpus.
    # Over-masking destroys correctness while under-masking only costs efficiency, so the
    # asymmetry is why the guard exists at all and why G alone is not the flip candidate.
    "G": {
        "label": "edge_bar_hud_mask_detection_only_on_live_config",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": True,
            "frontier_gradient": False,
            "edge_bar_hud_mask": True,
            "hud_mask_collapse_guard": False,
        },
        "deterministic": False,
    },
    "G2": {
        "label": "edge_bar_hud_mask_with_collapse_guard_on_live_config",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": True,
            "frontier_gradient": False,
            "edge_bar_hud_mask": True,
            "hud_mask_collapse_guard": True,
        },
        "deterministic": False,
    },
    "C": {
        "label": "frontier_distance_gradient",
        "kwargs": {"tier_exhaustion": False, "frontier_gradient": True},
        "deterministic": True,
    },
    "D": {
        "label": "tier_exhaustion_plus_gradient",
        "kwargs": {
            "tier_exhaustion": True,
            "tier_uniform_random": False,
            "frontier_gradient": True,
        },
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


def _seed_global_rngs(seed: int) -> None:
    """Seed the two global RNGs the arms actually draw from."""

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except Exception:  # pragma: no cover -- numpy is a hard dependency of the harness
        pass


def construct_reference_agent_seeded(factory: Callable[[], Any], seed: int) -> Any:
    """Build the reference agent, then RESTORE the seed its ``__init__`` clobbers.

    ``HeuristicAgent.__init__`` ends with ``random.seed(int(time.time() * 1000000) +
    hash(self.game_id) % 1000000)`` (heuristic_agent.py:66-69). Because construction happens
    AFTER the harness seeds the RNG, that line silently overwrote the harness seed and made
    every arm-E cell an unrepeatable draw -- which is why the "reference wins r11l 9/9" row
    could not be reproduced and had to be retracted.

    Extracted as a named function so the ORDER (construct, THEN re-seed) is directly
    testable without needing the vendored reference clone present: a test can pass a factory
    that clobbers the seed the same way and assert the seed is restored afterwards.
    """

    agent = factory()
    _seed_global_rngs(seed)
    return agent


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
        agent = construct_reference_agent_seeded(
            lambda: shim.OfflineHeuristicAgent(env, gid, budget), seed
        )
        # ARM-E SEEDING FIX (2026-07-25, REQ-ARC-WMTE-5950 co-change).
        #
        # THE DEFECT: HeuristicAgent.__init__ does `seed = int(time.time() * 1000000) +
        # hash(self.game_id) % 1000000; random.seed(seed)` (heuristic_agent.py:66-69). It runs
        # AFTER the harness seeded the global RNG above, so it CLOBBERS that seed with the wall
        # clock. Every arm-E cell was therefore an independent unseeded draw: its "9 of 9 r11l
        # wins" were nine unrepeatable coin flips, and re-running a cell could not reproduce
        # it. A positive control that cannot be reproduced cannot support a claim about the
        # reference, which is exactly what the 2026-07-24 corrigendum had to retract.
        #
        # THE FIX: re-seed AFTER construction, so the harness's seed is the one in force for
        # every draw the agent actually makes (all of its randomness is drawn during main(),
        # not in __init__ -- __init__ only seeds). Deliberately NOT done by patching the
        # reference's own file: it is a read-only vendored clone under its own licence, and
        # editing it would make our measurements non-reproducible against a fresh clone.
        # (The re-seed itself lives in construct_reference_agent_seeded, above.)
        # Instrument the FIRST level-up's action count. Without this, arm E reports only its
        # budget-bound total (the reference does NOT early-stop on a level-up, so its `actions`
        # is always ~= budget and is therefore NOT comparable to arms A-D, which stop at the
        # first level-up). The action count AT the first level-up IS comparable, and it is the
        # efficiency axis the live scorer squares -- so it is worth capturing rather than
        # reporting an incomparable number and calling it a comparison.
        #
        # RESET ACCOUNTING (fixed 2026-07-24 after adversarial review). The reference's
        # `action_counter` (agents/agent.py: incremented once per main-loop iteration) counts
        # RESET as an action; `arc_leaderboard_eval.run_game`, which measures arms A-D, does NOT
        # (`actions += 1` sits in the non-RESET branch). Reporting the raw counter therefore
        # CHARGED arm E for resets that arms A-D got for free -- a bias against the positive
        # control on the very axis the comparison headlines. We now count resets and report the
        # reset-EXCLUDED figure as the comparable `actions_to_first_levelup`, keeping the raw
        # counter alongside it under an explicit name. The +1 is the same off-by-one convention
        # run_game uses: run_game increments `actions` BEFORE testing the level, whereas this
        # hook fires inside take_action, before main() increments the counter for this action.
        first_levelup: dict[str, Optional[int]] = {"actions": None, "actions_incl_reset": None}
        resets = {"n": 0}
        # ARM-E INSTRUMENTATION (2026-07-25). Arm E previously reported states_expanded=None on
        # every row, which is how a control that livelocked on 72-97% of its decisions read as a
        # legitimate null across 975 cells. Two counters close that:
        #   errors        -- the shim's take_action swallows exceptions and returns a GAME_OVER
        #                    frame; that swallow IS the livelock/crash signal, so count it.
        #   choose_errors -- exceptions raised by the reference's own choose_action (its
        #                    "no available actions" ValueError, which it uses as control flow).
        # states_expanded comes from the reference's own per-frame bookkeeping table, which is
        # its node set -- the closest analogue to our explorer's graph size.
        errors = {"take_action": 0, "choose_action": 0}
        _inner_take_action = agent.take_action
        _inner_choose_action = agent.choose_action

        def _counting_choose_action(*args, **kwargs):
            try:
                return _inner_choose_action(*args, **kwargs)
            except Exception:
                errors["choose_action"] += 1
                raise

        agent.choose_action = _counting_choose_action  # type: ignore[method-assign]

        def _counting_take_action(action):
            is_reset = str(getattr(action, "name", "")) == "RESET"
            if is_reset:
                resets["n"] += 1
            fd_ = _inner_take_action(action)
            try:
                # The shim returns an empty-frame GAME_OVER FrameData when the env step raised.
                if not getattr(fd_, "frame", None) and str(getattr(fd_, "state", "")).endswith(
                    "GAME_OVER"
                ):
                    errors["take_action"] += 1
            except Exception:
                pass
            try:
                if first_levelup["actions"] is None and int(getattr(fd_, "score", 0) or 0) > 0:
                    raw = int(getattr(agent, "action_counter", 0) or 0) + 1
                    first_levelup["actions_incl_reset"] = raw
                    first_levelup["actions"] = max(0, raw - resets["n"])
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
            # Reset-EXCLUDED = the convention arms A-D are measured in (run_game does not count
            # RESET as an action). This is the number that is comparable across arms.
            "actions_to_first_levelup": first_levelup["actions"],
            "actions_to_first_levelup_incl_reset": first_levelup["actions_incl_reset"],
            "resets_taken": int(resets["n"]),
            # Distinct frame hashes the reference built bookkeeping for = its node set.
            "states_expanded": int(len(getattr(agent, "hashed_frame2action_results", {}) or {})),
            "errors": int(errors["take_action"] + errors["choose_action"]),
            "error_breakdown": {
                "take_action_swallowed": int(errors["take_action"]),
                "choose_action_raised": int(errors["choose_action"]),
            },
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
            "arm": arm,
            "game": game,
            "seed": seed,
            "ran": True,
            "levels": int(out["levels"]),
            "reached": int(out["reached"]),
            "actions": int(out["actions"]),
            "actions_to_first_levelup": out.get("actions_to_first_levelup"),
            "actions_to_first_levelup_incl_reset": out.get("actions_to_first_levelup_incl_reset"),
            # WHOLE-RUN total resets (the reference keeps resetting after the first level-up
            # because it does not early-stop). The resets charged BEFORE the first level-up --
            # the quantity that makes the two conventions differ -- is
            # actions_to_first_levelup_incl_reset - actions_to_first_levelup, NOT this field.
            "resets_taken": out.get("resets_taken"),
            "action_count_convention": "resets_excluded_matching_run_game",
            # NOT comparable to arms A-D's `actions`: the reference does not early-stop on a
            # level-up, so its total is budget-bound. Its LEVELS and its
            # actions_to_first_levelup are the comparable quantities.
            "actions_total_is_budget_bound": True,
            # Arm E runs to budget, so its `levels` is NOT capped the way arms A-D's is
            # (target_levels=1 stops them at the first level-up). See
            # `levels_capped_by_early_stop` on the explorer arms.
            "levels_capped_by_early_stop": False,
            # 2026-07-25: was hardcoded None. An arm with no expansion count and no error count
            # is an UNINSTRUMENTED arm, and this one was silently livelocking.
            "states_expanded": out.get("states_expanded"),
            "errors": int(out.get("errors") or 0),
            "error_breakdown": out.get("error_breakdown"),
            # NAMED for what it actually is (2026-07-25). 100% of the observed count is the
            # reference's OWN choose_action raising ValueError("No available actions found")
            # (heuristic_agent.py:343); its main() catches ANY exception, sets failed=True and
            # level_up=True, and REPLAYS self.last_action_object. So a high rate here does not
            # mean "the reference is using exceptions as benign control flow" and does not mean
            # "the reference livelocks through our shim" -- it means the reference spends that
            # fraction of its budget in a self-flagged degenerate repeat-last-action loop. Any
            # cell above the threshold is NOT a measurement of reference behaviour.
            "reference_choose_action_raised": int(
                (out.get("error_breakdown") or {}).get("choose_action_raised") or 0
            ),
            "degenerate_fallback_fraction": round(
                int(out.get("errors") or 0) / max(1, int(out["actions"])), 4
            ),
            "reference_degenerate": bool(
                int(out.get("errors") or 0) / max(1, int(out["actions"]))
                > REFERENCE_DEGENERACY_THRESHOLD
            ),
            "efficiency": None,
            "duration_s": out["duration_s"],
            "frontier_discipline": None,
            # REQ-ARC-WMTE-5960: arm E is the reference solver through our shim; it keeps its
            # OWN mask and its own bookkeeping table, so our explorer's HUD/dedup counters do
            # not exist for it. Explicitly stated with a reason rather than left absent, so
            # "field missing" can never be read as "measured zero".
            "hud_mask_resolved": None,
            "hud_mask_cell_count": None,
            "hud_mask_source": "not_applicable_reference_solver_owns_its_own_mask",
            "unique_frames": None,
            "graph_nodes": out.get("states_expanded"),
            "node_inflation": None,
            "collapse_guard_refusals": None,
            "hud_mask": None,
        }

    t0 = time.time()
    # Thread the SEED into the explorer's own RNG. Without this the within-tier uniform draw
    # runs off a fixed constructor default, so the "seeds" of arm B2 would produce byte-identical
    # rows and its confidence interval would be fabricated width-zero certainty. (Caught by the
    # smoke run: B2's two seeds were initially identical.) Harmless for the deterministic arms,
    # which never draw from this RNG.
    policy = _explorer_policy(game, frontier_discipline_seed=seed, **ARMS[arm]["kwargs"])
    # 2026-07-25: an exception out of run_game used to propagate to the caller's blanket
    # handler, which recorded {"ran": False} WITHOUT an error count on the arm's own row -- so
    # a partially-crashed arm could still average out to a plausible-looking null. The
    # exception is now attributed to the arm that raised it, on that arm's row.
    try:
        r = lb.run_game(game, policy, budget=budget, variant=variant, reflect=reflect)
    except Exception as exc:
        explorer = getattr(policy, "explorer", None)
        return {
            "arm": arm,
            "game": game,
            "seed": seed,
            "ran": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "errors": 1,
            "states_expanded": (len(explorer.graph) if explorer is not None else None),
            # REQ-ARC-WMTE-5960: a crashed cell still reports whatever HUD state it reached, so
            # a partial crash cannot be mistaken for "the detector never resolved a mask".
            "hud_mask_resolved": (
                bool(getattr(explorer, "hud_mask", None) is not None)
                if explorer is not None
                else None
            ),
            "graph_nodes": (len(explorer.graph) if explorer is not None else None),
        }
    explorer = getattr(policy, "explorer", None)
    fd_diag = None
    if explorer is not None and hasattr(explorer, "frontier_discipline_diagnostics"):
        fd_diag = explorer.frontier_discipline_diagnostics()
    # REQ-ARC-WMTE-5960 per-cell HUD/identity instrumentation. Emitted on EVERY explorer row
    # of EVERY arm -- not just the treatment arms -- because an arm with no mask/dedup counters
    # is an UNINSTRUMENTED arm, and this project has already once read a 72-97%-crashed control
    # as a legitimate null across 975 cells for exactly that reason.
    hud_diag = None
    if explorer is not None and hasattr(explorer, "hud_mask_diagnostics"):
        try:
            hud_diag = explorer.hud_mask_diagnostics()
        except Exception as exc:  # pragma: no cover - instrumentation must never fail a run
            hud_diag = {"error": f"{type(exc).__name__}:{exc}"}
    return {
        "arm": arm,
        "game": game,
        "seed": seed,
        "ran": True,
        "levels": int(r["levels"]),
        "reached": int(r["reached"]),
        "actions": int(r["actions"]),
        "actions_to_first_levelup": r["actions_to_first_levelup"],
        "action_count_convention": "resets_excluded_run_game_native",
        # StepwiseExplorer defaults to target_levels=1, so is_done() stops these arms at the
        # FIRST level-up and their `levels` is capped at 1 BY CONSTRUCTION. Arm E has no such
        # early stop (it runs to budget), so `levels` is NOT an arm-comparable quantity -- this
        # flag exists so no reader or downstream capstone compares it across arms.
        "levels_capped_by_early_stop": True,
        "target_levels": int(getattr(explorer, "target_levels", 1) if explorer is not None else 1),
        # states_expanded = distinct graph nodes the explorer built. The search-effort axis:
        # a discipline that reaches the same level with fewer expanded states is a real win
        # even when the binary win count is flat.
        "states_expanded": (len(explorer.graph) if explorer is not None else None),
        # Emitted on EVERY row of EVERY arm so a crashed arm cannot read as a clean null.
        # HONEST SCOPE of what this counts for an explorer arm: exceptions the SAMPLER
        # swallowed internally -- BOTH halves, the redraw path (`click_pixel_errors`) and the
        # generation path (`click_pixel_generation_errors`; omitting the latter is how a
        # totally dead generation-path sampler reported errors=0) -- plus (on the not-ran row
        # above) a run_game crash. It does NOT instrument every exception the explorer
        # swallows elsewhere in its own pipeline -- those would need their own counters, which
        # do not exist yet. So errors == 0 means "no crash and no sampler error", not
        # "provably nothing was swallowed anywhere".
        "errors": int((fd_diag or {}).get("click_pixel_errors") or 0)
        + int((fd_diag or {}).get("click_pixel_generation_errors") or 0),
        # THE MECHANISM'S ACTIVITY WITNESS, lifted onto the row so no reader has to dig into
        # frontier_discipline to answer "did the sampler actually replace anything". A sampler
        # arm reporting 0 here is a CONTROL, not a treatment, whatever its label says.
        "click_pixel_coordinates_changed": (fd_diag or {}).get("click_pixel_coordinates_changed"),
        "click_pixel_generation_errors": (fd_diag or {}).get("click_pixel_generation_errors"),
        "efficiency": r["efficiency"],
        "duration_s": round(time.time() - t0, 3),
        "frontier_discipline": fd_diag,
        # --- REQ-ARC-WMTE-5960 HUD / node-identity axis -------------------------------------
        # Lifted onto the row (not buried in `hud_mask`) so a reader never has to dig to answer
        # "did the detector fire, and did dedup actually happen".
        "hud_mask_resolved": (hud_diag or {}).get("hud_mask_resolved"),
        "hud_mask_cell_count": (hud_diag or {}).get("hud_mask_cell_count"),
        "hud_mask_source": (hud_diag or {}).get("hud_mask_source"),
        "unique_frames": (hud_diag or {}).get("unique_frames"),
        "graph_nodes": (hud_diag or {}).get("graph_nodes"),
        # graph_nodes / distinct UNMASKED frames. 1.0 = every distinct raw frame became its own
        # node (no dedup at all -- the measured r11l pathology). This is the oracle-free stand-in
        # for true node inflation, which needs a per-game count of real game states and is
        # therefore available only from public game source, i.e. diagnostic-only.
        "node_inflation": (hud_diag or {}).get("node_inflation_vs_unique_frames"),
        # THE GUARD'S ACTIVITY WITNESS. >0 means the guard PROVED the mask was collapsing
        # distinct states and un-masked those nodes. An arm with the guard armed and 0 refusals
        # is evidence the mask is clean on that cell, not evidence the guard is dead -- read it
        # together with `collapse_guard.keys_with_multiple_successors`, the honest denominator.
        "collapse_guard_refusals": (hud_diag or {}).get("collapse_guard_refusals"),
        "hud_mask": hud_diag,
    }


def _seeds_for(arm: str, n_seeds: int) -> list[int]:
    """Deterministic arms get ONE run. See spec-deviation 2: seeding them fakes replication."""

    if ARMS[arm]["deterministic"]:
        return [RANDOM_SEED]
    return [RANDOM_SEED + i for i in range(max(1, int(n_seeds)))]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


# Two-sided 95% t-critical values by degrees of freedom (n-1). A normal-approximation 1.96 at
# n=2 understates the interval by ~6.5x (t=12.706), which is exactly the regime this experiment
# runs in, so the small-n correction is not optional here.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    29: 2.045,
}


def _t_crit95(df: int) -> float:
    """Two-sided 95% t critical value, conservative (rounds UP to the nearest tabled df)."""

    if df <= 0:
        return float("inf")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def _mean_ci95(values: Sequence[float], *, clamp_min: Optional[float] = None) -> dict:
    """Mean with a small-sample t-based 95% CI, or an explicit "no interval available".

    ``ci95`` is None when n < 2 AND when the sample standard deviation is exactly 0. The
    zero-sd case is the one the original version got wrong: it emitted ``[x, x]``, a
    zero-width interval indistinguishable from certainty, which is precisely what this
    function's own docstring promised not to do (observed in the smoke run: arm E's
    first_win_rate ci95 came out [1.0, 1.0] from two identical values).

    ``clamp_min`` clamps the interval (used with 0 for action counts, where the naive
    normal/t interval can otherwise report a physically impossible negative bound).

    THIS IS DESCRIPTIVE ONLY when the values are pooled across games. Between-game difficulty
    variance dominates such a pool, so a wide overlapping interval here is NOT evidence of no
    effect -- see ``paired_efficiency_vs_baseline`` for the inferential statistic.
    """

    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    if len(vals) < 2:
        return {"n": 1, "mean": vals[0], "ci95": None, "ci95_absent_reason": "n_lt_2"}
    sd = statistics.stdev(vals)
    mean = statistics.fmean(vals)
    if sd == 0.0:
        return {
            "n": len(vals),
            "mean": round(mean, 4),
            "sd": 0.0,
            "ci95": None,
            "ci95_absent_reason": "zero_variance_interval_would_be_fake_certainty",
        }
    half = _t_crit95(len(vals) - 1) * sd / (len(vals) ** 0.5)
    lo, hi = mean - half, mean + half
    clamped = False
    if clamp_min is not None and lo < clamp_min:
        lo, clamped = float(clamp_min), True
    return {
        "n": len(vals),
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci95_method": "t_distribution_small_sample",
        "ci95_clamped_at_min": clamped,
    }


def _sign_test_p(deltas: Sequence[float]) -> dict:
    """Exact two-sided sign test over PAIRED deltas (ties dropped, as the sign test requires).

    Why a sign test rather than a t-test: n is tiny (at most one delta per game that has a
    baseline win), the per-game delta distribution is not remotely normal (first-win costs span
    20 to 1747 actions), and the question being asked is directional ("does the arm beat the
    baseline more often than chance"), which the sign test answers without a distributional
    assumption. Wilcoxon signed-rank would use magnitude too, but at n<=7 its exact
    null distribution buys almost nothing over the sign test and costs an implementation whose
    correctness is harder to see by eye.
    """

    nz = [float(d) for d in deltas if float(d) != 0.0]
    n = len(nz)
    n_pos = sum(1 for d in nz if d > 0)
    if n == 0:
        return {
            "n_pairs_nonzero": 0,
            "n_favouring_arm": 0,
            "p_value": None,
            "p_absent_reason": "all_paired_deltas_are_ties",
        }
    # Exact binomial two-sided p at q=0.5, summing the tail at least as extreme as observed.
    k = max(n_pos, n - n_pos)

    def _c(nn: int, rr: int) -> int:
        out = 1
        for i in range(rr):
            out = out * (nn - i) // (i + 1)
        return out

    tail = sum(_c(n, i) for i in range(k, n + 1))
    p = min(1.0, 2.0 * tail / (2**n))
    return {
        "n_pairs_nonzero": n,
        "n_favouring_arm": n_pos,
        "p_value": round(p, 5),
        "test": "exact_two_sided_sign_test_q0.5",
        "smallest_attainable_p_at_this_n": round(min(1.0, 2.0 / (2**n)), 5),
    }


def _bootstrap_median_ci(
    deltas: Sequence[float], *, iters: int = 4000, seed: int = 20260724
) -> dict:
    """Percentile bootstrap 95% CI on the MEDIAN paired delta.

    The median (not the mean) because a single 1747-action game would otherwise dominate a
    7-game mean. Reported as None below n=3, where a bootstrap of a bootstrap-degenerate
    sample would look far more informative than it is.
    """

    vals = [float(d) for d in deltas]
    if len(vals) < 3:
        return {
            "n": len(vals),
            "median": (round(statistics.median(vals), 4) if vals else None),
            "ci95": None,
            "ci95_absent_reason": "n_lt_3_bootstrap_uninformative",
        }
    rng = random.Random(seed)
    meds = []
    for _ in range(int(iters)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[int(0.025 * len(meds))]
    hi = meds[min(len(meds) - 1, int(0.975 * len(meds)))]
    return {
        "n": len(vals),
        "median": round(statistics.median(vals), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "method": f"percentile_bootstrap_{iters}_iters",
    }


def paired_efficiency_vs_baseline(rows: Sequence[dict]) -> dict:
    """THE PRIMARY EFFICIENCY STATISTIC: per-game PAIRED deltas against arm A.

    WHY THIS REPLACED THE POOLED CI AS THE HEADLINE (adversarial review 2026-07-24). The
    previous primary statistic pooled ``actions_to_first_levelup`` across GAMES into one
    unpaired list, so its confidence interval was dominated by between-game difficulty variance
    rather than by the intervention. Measured on the real smoke: arm A mean 40.0 ci95
    (0.80, 79.20) vs arm B mean 13.5 ci95 (-3.16, 30.16) -- heavily overlapping, with a
    physically impossible negative lower bound -- WHILE the paired per-game deltas were
    [+15, +38], both the same sign. In other words the pooled statistic could not see an effect
    that the data showed unanimously. Since the harness runs every arm on the SAME games with
    the SAME seeds and differs only in policy, the pairing is free and discarding it was pure
    loss of power.

    delta = a2f(A) - a2f(arm), so POSITIVE means the arm reached its first level-up in FEWER
    actions than the baseline (an improvement). Pairs exist only where BOTH arms won that game
    (a game only one arm wins is a CAPABILITY difference, counted by
    ``compare_to_baseline``/``new_wins``, and folding it in here as an efficiency number would
    double-count it as both).
    """

    def _a2f(r: dict) -> Optional[float]:
        v = r.get("actions_to_first_levelup")
        return float(v) if v else None

    base: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.get("arm") != "A" or not r.get("ran"):
            continue
        v = _a2f(r)
        if v is not None:
            base.setdefault((r["game"], r.get("condition", "?")), []).append(v)

    out: dict[str, Any] = {
        "definition": "delta = actions_to_first_levelup(A) - actions_to_first_levelup(arm); "
        "positive = arm is MORE efficient than the baseline",
        "pairing": "same game, same condition, same seed set; arms differ only in policy",
        "primary": True,
        "note": "arms A-D are bit-deterministic (1 run); B2/E average their seeds within a game "
        "before pairing, so one pair per (game, condition) either way",
    }
    for arm in ARMS:
        if arm == "A":
            continue
        arm_rows: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            if r.get("arm") != arm or not r.get("ran"):
                continue
            v = _a2f(r)
            if v is not None:
                arm_rows.setdefault((r["game"], r.get("condition", "?")), []).append(v)
        per_cond: dict[str, Any] = {}
        for cond, _v, _r in CONDITIONS:
            pairs = []
            for (game, c), bvals in sorted(base.items()):
                if c != cond:
                    continue
                avals = arm_rows.get((game, c))
                if not avals:
                    continue
                b = statistics.fmean(bvals)
                a = statistics.fmean(avals)
                pairs.append({"game": game, "baseline": b, "arm": a, "delta": round(b - a, 4)})
            if not pairs:
                continue
            deltas = [p["delta"] for p in pairs]
            per_cond[cond] = {
                "n_paired_games": len(pairs),
                "pairs": pairs,
                "sign_test": _sign_test_p(deltas),
                "median_paired_delta": _bootstrap_median_ci(deltas),
                "mean_paired_delta_descriptive_only": round(statistics.fmean(deltas), 4),
            }
        if per_cond:
            out[arm] = per_cond
    return out


def power_ceiling(games: Sequence[str], baseline_win_games: Sequence[str]) -> dict:
    """How small a p-value this corpus can even attain -- stated up front, not discovered later.

    With k baseline-win games there are at most k paired deltas, so the smallest two-sided
    exact sign-test p is ``2 * 0.5**k`` and only if EVERY pair favours the arm. At k=6 that is
    0.031: it clears 0.05 with zero margin, and a single contrary game makes significance
    unreachable. Publishing this alongside the result stops a null being read as evidence of
    absence when it is really absence of power.
    """

    in_corpus = [g for g in baseline_win_games if g in set(games)]
    k = len(in_corpus)
    # The barrier and the within-tier draw act on CLICK candidates only, so on a nav-only game
    # they are structurally inert and their paired delta is a tie -- which the sign test drops.
    # The EFFECTIVE n for the efficiency test is therefore the CLICK stratum, and that is the
    # number the power ceiling has to be computed from; using the pooled count would overstate
    # attainable power by counting games the mechanism cannot move.
    click = [g for g in in_corpus if g in set(CLICK_GAMES)]
    kc = len(click)
    return {
        "n_baseline_win_games_in_corpus": k,
        "baseline_win_games_in_corpus": sorted(in_corpus),
        "n_baseline_win_click_games": kc,
        "baseline_win_click_games": sorted(click),
        "max_paired_deltas": k,
        "max_effective_paired_deltas_click_stratum": kc,
        "smallest_attainable_two_sided_p": (round(min(1.0, 2.0 * 0.5**kc), 5) if kc else None),
        "smallest_attainable_two_sided_p_pooled_all_strata": (
            round(min(1.0, 2.0 * 0.5**k), 5) if k else None
        ),
        "clears_0.05_only_if_unanimous": bool(kc and 2.0 * 0.5**kc <= 0.05),
        "interpretation": "a flat sign test at this n is UNDERPOWERED, not evidence of no "
        "effect; widening the corpus (more budget so more games enter the baseline-win set, or "
        "deeper per-game levels) is the only way to raise this ceiling",
    }


def aggregate(rows: Sequence[dict], games: Sequence[str]) -> dict:
    """Per (arm, condition) aggregates, stratified by action vocabulary."""

    out: dict[str, Any] = {}
    click = set(CLICK_GAMES)
    for arm in ARMS:
        for cond, _v, _r in CONDITIONS:
            cell = [
                r
                for r in rows
                if r.get("arm") == arm and r.get("condition") == cond and r.get("ran")
            ]
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
            eff_costs = [
                r["actions_to_first_levelup"] for r in cell if r.get("actions_to_first_levelup")
            ]
            states = [r["states_expanded"] for r in cell if r.get("states_expanded") is not None]
            out[f"{arm}|{cond}"] = {
                "arm": arm,
                "arm_label": ARMS[arm]["label"],
                "condition": cond,
                "deterministic_arm": bool(ARMS[arm]["deterministic"]),
                "n_runs_per_game": len(seeds),
                "games_measured": len(by_game),
                "games_won_any_seed": any_win,
                "n_games_won_any_seed": len(any_win),
                "first_win_rate_over_seeds": _mean_ci95(per_seed_rates),
                # STRUCTURAL LIMIT, not a fixable sampling gap: arms A/B/C/D are bit-deterministic
                # and run once by design (spec-deviation 2), so this rate has n=1 and can NEVER
                # carry a CI for them no matter how many seeds are requested. It is a point
                # estimate over games, and the cross-arm inference lives in
                # `paired_efficiency_vs_baseline` + `capability_summary`, not here.
                "first_win_rate_ci_unavailable_by_construction": bool(ARMS[arm]["deterministic"]),
                "click_stratum_wins": sorted(g for g in any_win if g in click),
                "nav_stratum_wins": sorted(g for g in any_win if g not in click),
                # PER-ARM ONLY -- NEVER COMPARE THIS ACROSS ARMS. Arms A-D run with
                # target_levels=1, so StepwiseExplorer.is_done stops them at the first level-up
                # and this sum is capped at 1 per game by construction; arm E runs to budget and
                # reports its true max_score. Measured on the same 3 games: A-D = 2, E = 4, and
                # half of that gap is the stopping rule, not capability. The two sibling fields
                # below carry that caveat with the number so it cannot travel without it.
                "levels_banked_total": sum(max(x["levels"] for x in rs) for rs in by_game.values()),
                "levels_capped_by_early_stop": bool(
                    any(r.get("levels_capped_by_early_stop") for r in cell)
                ),
                "levels_banked_total_cross_arm_comparable": False,
                "mean_actions_to_first_win": _mean_ci95(eff_costs, clamp_min=0.0),
                # A pooled-across-games mean is dominated by between-game difficulty; it is kept
                # as descriptive colour only. The inferential statistic is the PAIRED per-game
                # comparison in `paired_efficiency_vs_baseline`.
                "mean_actions_to_first_win_is_inferential": False,
                "mean_states_expanded": _mean_ci95(states, clamp_min=0.0),
            }
    return out


def compare_to_baseline(agg: dict, games: Sequence[str], rows: Sequence[dict] = ()) -> dict:
    """Arm-vs-A deltas on the REAL condition, plus the regression guard and the control check."""

    base_key = "A|real"
    base = agg.get(base_key)
    if base is None:
        return {"available": False, "reason": "baseline_arm_A_real_missing"}
    base_wins = set(base["games_won_any_seed"])
    guard, guard_provenance = _guard_games_from_rows(rows)
    guard = [g for g in guard if g in set(games)]
    out: dict[str, Any] = {
        "available": True,
        "baseline_wins": sorted(base_wins),
        "regression_guard_games": guard,
        "regression_guard_provenance": guard_provenance,
    }
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


# DELIBERATELY UNCHANGED when the REQ-ARC-WMTE-5950 sampler arms (F, F1) were added. This
# tuple feeds capability_summary + acceptance_gate_capability, whose numbers are already on
# the record for the frontier-discipline graft; folding new arms into it would silently
# redefine an existing published quantity (a milestone-over-milestone comparison would then
# be comparing two different definitions). The sampler arms get their OWN gate below.
GRAFTED_ARMS = ("B", "B2", "C", "D")

# Above this fraction of a cell's actions ending in the reference's own choose_action raising,
# the cell is NOT a measurement of reference behaviour: the reference's main() catches the
# exception, sets failed=True / level_up=True, and replays last_action_object, so the remaining
# budget is a degenerate repeat-last-action loop rather than its tier assignment and
# within-tier draw. 5% is deliberately strict -- the observed cells are 78-96%, and a healthy
# cell measured exactly 0. Origin: the smoke reported errored_cell_rate 0.0 and
# positive_control_ran true while 4 of 6 arm-E cells were 78-96% degenerate.
REFERENCE_DEGENERACY_THRESHOLD = 0.05

# REQ-ARC-WMTE-5950: the sampler arms and the arm that is their MATCHED control (identical
# flags except the sampler itself). Named here so the gate cannot silently drift onto arm A,
# which is the PRE-flip baseline rather than the live configuration.
CLICK_PIXEL_ARMS = ("F", "F1")
CLICK_PIXEL_CONTROL_ARM = "B2"

# REQ-ARC-WMTE-5960: the repaired HUD detector arms. Same matched control (B2 = the current
# live configuration), same one-flag-difference discipline as the sampler arms above.
HUD_MASK_ARMS = ("G", "G2")
HUD_MASK_CONTROL_ARM = "B2"


def positive_control_health(rows: Sequence[dict]) -> dict:
    """Is arm E's measurement actually a measurement OF THE REFERENCE?

    Origin (2026-07-25 adversarial review, second pass): arm E was instrumented in the first
    pass -- but the instrumentation GATED NOTHING. ``errored_cell_rate`` is computed from
    ``not row["ran"]`` only, so a cell that ran to completion with 96% of its decisions ending
    in the reference's own ``choose_action`` raising still counted as a clean cell:
    ``errored_cell_rate: 0.0``, ``positive_control_ran: true``, ``positive_control_reason:
    "ok"``, and the headline credited the reference with a new win "under identical
    conditions". Read from the reference's source, that error count is not benign control
    flow: ``heuristic_agent.py:343`` raises ``ValueError("No available actions found")`` and
    ``main()`` (465-469) catches ANY exception, sets ``failed=True`` / ``level_up=True``, and
    reuses ``last_action_object``. So a degenerate cell spends most of its budget replaying one
    action, self-flagged as failed. Its LOSSES are artifacts of that state, not evidence about
    the reference -- which matters because ``capability_summary.diagnostic_target`` was
    directing all forward work at "instrument what the reference does differently".

    What survives: a cell's WIN is still real when it lands before degeneration (the win is
    read from the env's own frame score), so a degenerate cell's win is kept and only its
    losses are marked uninterpretable.
    """

    cells = [r for r in rows if r.get("arm") == "E" and r.get("ran")]
    if not cells:
        return {
            "measured": False,
            "n_cells": 0,
            "n_degenerate_cells": 0,
            "degenerate_fraction": 0.0,
            "healthy": False,
            "reason": "no arm-E cell produced a measurement",
        }
    degenerate = []
    for row in cells:
        frac = row.get("degenerate_fallback_fraction")
        if frac is None:
            frac = int(row.get("errors") or 0) / max(1, int(row.get("actions") or 1))
        if float(frac) > REFERENCE_DEGENERACY_THRESHOLD:
            degenerate.append(
                {
                    "game": row.get("game"),
                    "seed": row.get("seed"),
                    "degenerate_fallback_fraction": round(float(frac), 4),
                    "choose_action_raised": row.get("reference_choose_action_raised"),
                    "levels": row.get("levels"),
                }
            )
    share = len(degenerate) / len(cells)
    worst = max((d["degenerate_fallback_fraction"] for d in degenerate), default=0.0)
    return {
        "measured": True,
        "n_cells": len(cells),
        "n_degenerate_cells": len(degenerate),
        "degenerate_fraction": round(share, 4),
        "worst_cell_fallback_fraction": worst,
        "degenerate_cells": degenerate,
        "threshold": REFERENCE_DEGENERACY_THRESHOLD,
        "mechanism": (
            "the reference's own choose_action raises ValueError('No available actions found'); "
            "its main() catches any exception, sets failed=True/level_up=True and replays "
            "last_action_object, so the remaining budget is a repeat-last-action loop"
        ),
        # A majority-degenerate positive control is not a usable positive control. Its wins
        # still stand (frame-score truth, landed before degeneration) but its losses are not
        # reference behaviour, so it cannot underwrite a "the reference does X, we do not" claim.
        "healthy": bool(share <= 0.5),
        "reason": (
            "ok"
            if share <= 0.5
            else f"reference_degenerate_in_{len(degenerate)}_of_{len(cells)}_cells_"
            f"worst_fallback_fraction_{worst:.2f}"
        ),
    }


def capability_summary(agg: dict, cmp_: dict, *, control_healthy: bool = True) -> dict:
    """THE DECISION-RELEVANT RESULT: did the graft transfer any CAPABILITY, vs the control?

    Filed as a top-level field on adversarial-review instruction (2026-07-24) because the first
    write-up of this experiment led with an efficiency delta on games Carnot already solves while
    burying the capability null: every grafted arm won the IDENTICAL game set as the baseline
    (n_win_delta = 0 across all three conditions), while the just-explore positive control won a
    game NO Carnot arm won, on every run. An efficiency number that travels without that null
    attached is a misleading summary of this experiment, so the null is computed here and named
    in the verdict rather than left for a reader to reconstruct from the aggregates.
    """

    if not cmp_.get("available"):
        return {"available": False, "reason": cmp_.get("reason", "no_baseline")}
    base_wins = set(cmp_.get("baseline_wins") or [])
    graft_new: set[str] = set()
    graft_lost: set[str] = set()
    per_arm: dict[str, Any] = {}
    for arm in GRAFTED_ARMS:
        info = cmp_.get(arm)
        if not isinstance(info, dict):
            continue
        graft_new |= set(info.get("new_wins") or [])
        graft_lost |= set(info.get("lost_wins") or [])
        per_arm[arm] = {
            "new_wins": info.get("new_wins"),
            "lost_wins": info.get("lost_wins"),
            "n_win_delta": info.get("n_win_delta"),
        }
    e_info = cmp_.get("E") if isinstance(cmp_.get("E"), dict) else {}
    e_new = set((e_info or {}).get("new_wins") or [])
    e_key = agg.get("E|real") or {}
    e_wins = set(e_key.get("games_won_any_seed") or [])
    only_control = sorted(e_wins - base_wins - graft_new)
    return {
        "available": True,
        "baseline_wins": sorted(base_wins),
        # The headline capability numbers.
        "new_wins_vs_baseline": len(graft_new),
        "new_win_games_vs_baseline": sorted(graft_new),
        "lost_wins_vs_baseline": sorted(graft_lost),
        "positive_control_new_wins": len(e_new),
        "positive_control_new_win_games": sorted(e_new),
        "games_won_only_by_positive_control": only_control,
        "per_grafted_arm": per_arm,
        # GATED on the positive control being non-degenerate (2026-07-25). Directing forward
        # work at "what the reference does differently" is only sound if arm E was actually
        # exercising the reference. When 4 of 6 cells spend 78-96% of their budget in the
        # reference's own failed=True repeat-last-action fallback, the discrepancy is not yet
        # attributable to the reference at all -- the shim's swallowed exceptions have to be
        # fixed before any "the reference does X" datum measured through it can be trusted.
        "positive_control_healthy": bool(control_healthy),
        "diagnostic_target": (
            (
                "the games in games_won_only_by_positive_control are now the ONLY "
                "evidence-bearing discrepancy: instrument what the reference does differently "
                "there (its tier assignment and its within-tier draw sequence) rather than "
                "reporting the efficiency delta as the outcome"
                if control_healthy
                else "NOT YET ATTRIBUTABLE TO THE REFERENCE. The positive control is degenerate "
                "on most cells (its own choose_action raises, main() sets failed=True and "
                "replays last_action_object), so its losses are shim artifacts and its "
                "tier/draw behaviour was largely not exercised. FIX THE SHIM's swallowed "
                "choose_action exceptions before borrowing any further reference mechanism or "
                "treating this discrepancy as a Carnot capability gap"
            )
            if only_control
            else "no game is won by the control alone -- no capability discrepancy to diagnose"
        ),
    }


def _per_seed_win_sets(
    rows: Sequence[dict], arm: str, condition: str = "real"
) -> dict[int, set[str]]:
    """``{seed: {games won}}`` for one arm. PER SEED, never an any-seed union.

    An any-seed union cannot register a regression: if a treatment loses a game on 2 of 3
    seeds and keeps it on 1, the union still says "won". This project has already made that
    mistake once, so the union is not offered as an option here.
    """

    out: dict[int, set[str]] = {}
    for row in rows:
        if row.get("arm") != arm or row.get("condition") != condition or not row.get("ran"):
            continue
        seed = int(row.get("seed"))
        out.setdefault(seed, set())
        if int(row.get("levels") or 0) > 0:
            out[seed].add(str(row.get("game")))
    return out


def _per_seed_measured_games(
    rows: Sequence[dict], arm: str, condition: str = "real"
) -> dict[int, set[str]]:
    """``{seed: {games that produced a real measurement}}`` for one arm.

    The DENOMINATOR the headroom calculation needs. ``_per_seed_win_sets`` only reports what
    was WON, which cannot distinguish "the control lost this game (a gain is possible)" from
    "this game was never run for the control (nothing is knowable)".
    """

    out: dict[int, set[str]] = {}
    for row in rows:
        if row.get("arm") != arm or row.get("condition") != condition or not row.get("ran"):
            continue
        out.setdefault(int(row.get("seed")), set()).add(str(row.get("game")))
    return out


def click_pixel_sampling_gate(rows: Sequence[dict], *, condition: str = "real") -> dict:
    """REQ-ARC-WMTE-5950's PRE-REGISTERED gate, stated before any sweep was run.

    THE CONDITION (fixed in advance, one line, no hidden conjuncts):

        On the REAL condition, on at least one shared seed, a sampler arm wins a game its
        MATCHED CONTROL (arm B2 = the current live configuration) does not win on that same
        seed, AND on no shared seed does that sampler arm lose a game the control wins.

    WHY IT IS SHAPED THIS WAY -- each clause answers a specific past measurement failure:

    * MATCHED CONTROL IS B2, NOT ARM A. Arm A pins the pre-flip flags explicitly, so
      comparing against it would credit the sampler with the already-flipped tier barrier's
      effect. B2 differs from F by exactly one flag.
    * PER SEED, PER GAME. Not a pooled rate and not an any-seed union, so a regression on
      one seed cannot be hidden by a win on another.
    * NO CONJUNCT ENCODES AN ASSUMED VALUE. Both clauses are differences MEASURED IN THIS
      SAME RUN. The gate that had to be voided (exp5835) failed because a conjunct asserted
      something about the baseline arm's value, which made the region unpassable for every
      possible treatment value. To make that failure mode impossible to repeat silently, this
      function COMPUTES a witness -- a concrete synthetic assignment of win sets that
      satisfies the condition -- and reports it. A gate whose pass region is empty cannot
      produce a witness, so ``pass_region_nonempty`` is a checked property, not a claim.
    * HEADROOM IS MEASURED, NOT ASSUMED (added 2026-07-25 after adversarial review). The
      witness above is SYNTHETIC: it proves the predicate is satisfiable in the abstract, and
      it CANNOT detect that the pass region was empty ON THE CORPUS ACTUALLY MEASURED. The
      first smoke run hit exactly that: the control already won 2 of the 3 games on both
      seeds, so the only game where any treatment value could have gained was r11l -- which
      the same session had independently diagnosed as blocked by state-identity aliasing, a
      defect this mechanism explicitly disclaims fixing. A "0 new wins" result there is an
      UNINFORMATIVE TEST, not evidence of no effect, and reporting it as a capability NULL is
      the FALSE_NEGATIVE_RISK pattern CLAUDE.md forbids propagating without a
      headroom-present positive control. So the gate now computes, per seed,
      ``control_unwon_games`` (the games where a gain is even possible) and reports
      ``headroom_present`` / ``informative``. With zero headroom the verdict is
      ``uninformative_no_headroom``, never a confident-looking ``passed: False``.
    * ITS OWN GATE, NOT THE FRONTIER-DISCIPLINE ONE. Folding these arms into
      ``acceptance_gate_capability`` would redefine a number already on the record.

    A missing arm is reported as ``passed: False`` with a reason, never as a silent pass.
    """

    witness = _click_pixel_gate_pass_region_witness()
    out: dict[str, Any] = {
        "condition": (
            "on the REAL condition and on shared seeds: (sampler arm wins >=1 game its "
            "matched control B2 does not win on that seed) AND (sampler arm loses no game "
            "B2 wins on that seed)"
        ),
        "principle": (
            "a generation-rule change earns a flag flip by making a game REACHABLE that the "
            "shipped configuration cannot reach, without giving one back; measured per seed "
            "against the arm that differs by exactly one flag"
        ),
        "matched_control_arm": CLICK_PIXEL_CONTROL_ARM,
        "why_not_arm_A": (
            "arm A passes tier_exhaustion=False / frontier_gradient=False as explicit "
            "constructor kwargs and _fd_gate ranks an explicit kwarg above the SUBMITTED_* "
            "default, so arm A is the PRE-flip agent -- not the live one"
        ),
        "pass_region_nonempty": bool(witness.get("passes")),
        "pass_region_witness": witness,
        "per_arm": {},
    }
    control = _per_seed_win_sets(rows, CLICK_PIXEL_CONTROL_ARM, condition)
    if not control:
        out["passed"] = False
        out["informative"] = False
        out["reason"] = f"matched control arm {CLICK_PIXEL_CONTROL_ARM} not measured in this run"
        return out
    control_ran = _per_seed_measured_games(rows, CLICK_PIXEL_CONTROL_ARM, condition)
    any_pass = False
    reachable_all: set[str] = set()
    for arm in CLICK_PIXEL_ARMS:
        treat = _per_seed_win_sets(rows, arm, condition)
        treat_ran = _per_seed_measured_games(rows, arm, condition)
        shared = sorted(set(treat) & set(control))
        if not shared:
            out["per_arm"][arm] = {
                "measured": False,
                "reason": "no seed shared with the matched control",
                "passed": False,
                "headroom_present": False,
                "informative": False,
            }
            continue
        per_seed = []
        reachable_arm: set[str] = set()
        for seed in shared:
            new = sorted(treat[seed] - control[seed])
            lost = sorted(control[seed] - treat[seed])
            # THE HEADROOM. A new win is only POSSIBLE on a game that both arms measured on
            # this seed and that the control did NOT win. If this set is empty the gate is a
            # test of nothing: every game is already at ceiling for the control.
            unwon = sorted(
                (control_ran.get(seed, set()) & treat_ran.get(seed, set())) - control[seed]
            )
            reachable_arm |= set(unwon)
            per_seed.append(
                {
                    "seed": seed,
                    "new_wins": new,
                    "lost_wins": lost,
                    "n_treatment_wins": len(treat[seed]),
                    "n_control_wins": len(control[seed]),
                    "control_unwon_games": unwon,
                    "max_attainable_new_wins": len(unwon),
                    "n_games_at_ceiling": len(control[seed]),
                }
            )
        gained = any(s["new_wins"] for s in per_seed)
        regressed = any(s["lost_wins"] for s in per_seed)
        passed = bool(gained and not regressed)
        any_pass = any_pass or passed
        reachable_all |= reachable_arm
        out["per_arm"][arm] = {
            "measured": True,
            "shared_seeds": shared,
            "per_seed": per_seed,
            "any_seed_gained": gained,
            "any_seed_regressed": regressed,
            "reachable_new_win_games": sorted(reachable_arm),
            "headroom_present": bool(reachable_arm),
            # A failed gate over ZERO reachable games says nothing about the mechanism. This
            # flag is what stops such a run being written up as a capability NULL.
            "informative": bool(reachable_arm) or passed,
            "passed": passed,
            # An arm whose activity witness is zero replaced no coordinate and is therefore a
            # control regardless of its label. Reported per arm so a silent no-op cannot be
            # read as a measured treatment.
            "mechanism_active": _arm_mechanism_active(rows, arm, condition),
        }
    out["reachable_new_win_games"] = sorted(reachable_all)
    out["headroom_present"] = bool(reachable_all)
    out["passed"] = bool(any_pass)
    out["informative"] = bool(reachable_all) or any_pass
    # NARROW HEADROOM is not the same as headroom. A pass region of ONE game means the gate's
    # entire win axis is a single cell, so its null is a statement about that one game rather
    # than about the mechanism -- and if that game has an independently-diagnosed blocker the
    # mechanism explicitly disclaims fixing (r11l's state-identity aliasing is exactly this
    # case), the null carries almost no information. Disclosed rather than left for a reader to
    # reconstruct from the per-seed sets, because the first write-up of this experiment did not
    # reconstruct it and called the result a capability NULL.
    out["n_reachable_new_win_games"] = len(reachable_all)
    out["headroom_narrow"] = bool(0 < len(reachable_all) <= 1)
    if out["headroom_narrow"]:
        out["headroom_narrow_note"] = (
            "the win axis had exactly ONE candidate game "
            f"({', '.join(sorted(reachable_all))}); every other game measured is already won by "
            "the matched control. A failed gate here is a statement about that single game, not "
            "a corpus-level null -- and it is uninformative about the mechanism if that game has "
            "a separately-diagnosed blocker this mechanism does not address. Run the full 25-game "
            "corpus (baseline wins ~7/25, so ~18 games of real headroom) before recording any "
            "capability null"
        )
    if not out["informative"]:
        out["verdict"] = "uninformative_no_headroom"
        out["reason"] = (
            "the matched control already wins every game measured on every shared seed, so the "
            "set of games where ANY treatment value could have gained a win is EMPTY. This run "
            "is therefore not a test of the sampler: report it as UNTESTED for capability, not "
            "as a capability NULL (CLAUDE.md FALSE_NEGATIVE_RISK)"
        )
    else:
        out["verdict"] = "passed" if out["passed"] else "failed_with_headroom_present"
    return out


def _arm_mechanism_active(rows: Sequence[dict], arm: str, condition: str = "real") -> dict:
    """Did the sampler on ``arm`` actually replace any coordinate? The activity witness.

    A treatment arm whose mechanism never fired is a duplicate of its control, and reading
    its result as "the mechanism had no effect" is a category error. ``instrumented: False``
    means the rows carry no witness at all -- the state this project previously shipped, in
    which a dead sampler was indistinguishable from a working one.
    """

    changed = 0
    gen_errors = 0
    instrumented = False
    for row in rows:
        if row.get("arm") != arm or row.get("condition") != condition or not row.get("ran"):
            continue
        value = row.get("click_pixel_coordinates_changed")
        if value is None:
            continue
        instrumented = True
        changed += int(value)
        gen_errors += int(row.get("click_pixel_generation_errors") or 0)
    return {
        "instrumented": instrumented,
        "coordinates_changed": changed,
        "generation_errors": gen_errors,
        "active": bool(instrumented and changed > 0),
    }


def _click_pixel_gate_pass_region_witness() -> dict:
    """Prove the gate above CAN pass, by evaluating it on a synthetic run that should.

    This is a guard against the exp5835 defect class (a gate whose conjunction is
    unsatisfiable for every possible treatment value). It builds two synthetic rows -- a
    control that wins one game and a treatment that wins that game plus one more, on the
    same seed -- and applies the SAME predicate the real gate applies. If the predicate is
    ever tightened into an unsatisfiable shape, this witness turns False and the artifact
    says so instead of reporting a confident-looking null.
    """

    seed = 1
    synthetic = [
        {
            "arm": CLICK_PIXEL_CONTROL_ARM,
            "game": "gA",
            "condition": "real",
            "seed": seed,
            "ran": True,
            "levels": 1,
        },
        {
            "arm": CLICK_PIXEL_ARMS[0],
            "game": "gA",
            "condition": "real",
            "seed": seed,
            "ran": True,
            "levels": 1,
        },
        {
            "arm": CLICK_PIXEL_ARMS[0],
            "game": "gB",
            "condition": "real",
            "seed": seed,
            "ran": True,
            "levels": 1,
        },
    ]
    control = _per_seed_win_sets(synthetic, CLICK_PIXEL_CONTROL_ARM)
    treat = _per_seed_win_sets(synthetic, CLICK_PIXEL_ARMS[0])
    gained = bool(treat.get(seed, set()) - control.get(seed, set()))
    regressed = bool(control.get(seed, set()) - treat.get(seed, set()))
    return {
        "construction": "control wins {gA}; treatment wins {gA, gB} on the same seed",
        "gained": gained,
        "regressed": regressed,
        "passes": bool(gained and not regressed),
    }


def _hud_mask_gate_pass_region_witness() -> dict:
    """Prove the HUD gate below CAN pass, by evaluating its predicate on a synthetic run.

    Same guard against the exp5835 defect class as `_click_pixel_gate_pass_region_witness`: a
    gate whose conjunction is unsatisfiable for EVERY possible treatment value is a
    measurement of nothing, and it reads as a confident null. The construction here is the
    minimum that should pass the HEADLINE clause -- a treatment that wins one game the control
    does not, on the same seed, while losing none -- plus the two SIGNAL quantities evaluated
    on synthetic rows so a tightened predicate turns this False instead of silently voiding
    the gate.
    """

    seed = 1
    synthetic = [
        {"arm": HUD_MASK_CONTROL_ARM, "game": "gA", "condition": "real", "seed": seed,
         "ran": True, "levels": 1, "node_inflation": 1.0, "hud_mask_resolved": False},
        {"arm": HUD_MASK_ARMS[0], "game": "gA", "condition": "real", "seed": seed,
         "ran": True, "levels": 1, "node_inflation": 0.2, "hud_mask_resolved": True},
        {"arm": HUD_MASK_ARMS[0], "game": "gB", "condition": "real", "seed": seed,
         "ran": True, "levels": 1, "node_inflation": 0.1, "hud_mask_resolved": True},
    ]
    control = _per_seed_win_sets(synthetic, HUD_MASK_CONTROL_ARM)
    treat = _per_seed_win_sets(synthetic, HUD_MASK_ARMS[0])
    gained = bool(treat.get(seed, set()) - control.get(seed, set()))
    regressed = bool(control.get(seed, set()) - treat.get(seed, set()))
    # The SIGNAL half, evaluated with the same helper the real gate uses, so the witness fails
    # if the signal computation is broken rather than only if the win predicate is.
    signal = _hud_mask_signal(synthetic, HUD_MASK_ARMS[0], game="gA", condition="real")
    return {
        "construction": (
            "control wins {gA} with node_inflation 1.0 and no mask; treatment wins {gA, gB} "
            "with a resolved mask and node_inflation 0.2 on the same seed"
        ),
        "gained": gained,
        "regressed": regressed,
        "signal_computable": bool(signal.get("measured")),
        "signal_inflation_fell": bool(signal.get("inflation_improved")),
        "passes": bool(gained and not regressed),
    }


def _hud_mask_signal(
    rows: Sequence[dict], arm: str, *, game: str = "r11l", condition: str = "real"
) -> dict:
    """The PRE-REGISTERED SIGNAL for the HUD repair -- and explicitly NOT the gate.

    The measured diagnosis predicts two specific things on r11l: the mask RESOLVES (it does not
    today), and node inflation collapses (arm A 44.9x -> ~1.0x with the oracle mask; on the B2
    control the same game inflates ~22-23x, so the per-arm baseline is what this compares
    against, never the headline number). Both are reported.

    WHY THIS IS NOT THE GATE. r11l is ONE public game. A flag flipped on the strength of one
    public game is hand-fitted to something worth ~0 on the hidden leaderboard, and this
    project has an explicit rule against exactly that. So the gate is the per-seed
    full-corpus regression clause; this is the mechanism-worked-as-diagnosed evidence that
    makes a corpus result interpretable rather than the thing being decided.
    """

    control_rows = [
        r
        for r in rows
        if r.get("arm") == HUD_MASK_CONTROL_ARM
        and r.get("game") == game
        and r.get("condition") == condition
        and r.get("ran")
    ]
    treat_rows = [
        r
        for r in rows
        if r.get("arm") == arm
        and r.get("game") == game
        and r.get("condition") == condition
        and r.get("ran")
    ]
    if not control_rows or not treat_rows:
        return {"measured": False, "reason": f"{game}/{condition} not measured for {arm} and control"}

    def _inflations(subset: Sequence[dict]) -> list[float]:
        return [float(r["node_inflation"]) for r in subset if r.get("node_inflation") is not None]

    control_inf = _inflations(control_rows)
    treat_inf = _inflations(treat_rows)
    per_seed = []
    for treat_row in sorted(treat_rows, key=lambda r: int(r.get("seed") or 0)):
        seed = int(treat_row.get("seed") or 0)
        match = next((r for r in control_rows if int(r.get("seed") or 0) == seed), None)
        per_seed.append(
            {
                "seed": seed,
                "control_levels": (int(match["levels"]) if match else None),
                "treatment_levels": int(treat_row.get("levels") or 0),
                "control_node_inflation": (match or {}).get("node_inflation"),
                "treatment_node_inflation": treat_row.get("node_inflation"),
                "treatment_mask_resolved": treat_row.get("hud_mask_resolved"),
                "treatment_mask_cell_count": treat_row.get("hud_mask_cell_count"),
                # PER SEED, never an any-seed union: a win on one seed must not conceal a
                # flat result on another.
                "level_gained_this_seed": bool(
                    match is not None
                    and int(treat_row.get("levels") or 0) > int(match.get("levels") or 0)
                ),
            }
        )
    return {
        "measured": True,
        "game": game,
        "condition": condition,
        "arm": arm,
        "control_arm": HUD_MASK_CONTROL_ARM,
        "per_seed": per_seed,
        "mask_resolved_all_seeds": all(bool(s["treatment_mask_resolved"]) for s in per_seed),
        "level_gained_all_seeds": bool(per_seed) and all(
            s["level_gained_this_seed"] for s in per_seed
        ),
        "level_gained_any_seed": any(s["level_gained_this_seed"] for s in per_seed),
        "mean_control_node_inflation": (
            round(statistics.fmean(control_inf), 4) if control_inf else None
        ),
        "mean_treatment_node_inflation": (
            round(statistics.fmean(treat_inf), 4) if treat_inf else None
        ),
        "inflation_improved": bool(
            control_inf and treat_inf and statistics.fmean(treat_inf) < statistics.fmean(control_inf)
        ),
        "baseline_note": (
            "the 44.9x figure in ops/known-issues.md is an ARM-A number (1392 nodes / 31 "
            "oracle-known true states). The control here is B2, whose measured r11l inflation "
            "is roughly half that. node_inflation reported here is graph_nodes / distinct "
            "unmasked frames -- oracle-free and therefore hidden-game legal, NOT the "
            "oracle-normalised ratio"
        ),
    }


def hud_mask_gate(rows: Sequence[dict], *, condition: str = "real") -> dict:
    """REQ-ARC-WMTE-5960's PRE-REGISTERED gate, stated before any sweep was run.

    THE HEADLINE CONDITION (the thing that decides the flag):

        Per seed, on the REAL condition, against the MATCHED CONTROL (arm B2 = the current
        live configuration): the treatment arm loses NO game the control wins on that same
        seed, and gains at least one.

    THE PRE-REGISTERED SIGNAL (reported, deliberately NOT gated on): r11l's mask RESOLVES and
    its node inflation falls. See `_hud_mask_signal` for why gating on one public game would
    be hand-fitting to something worth ~0 on the hidden set.

    EACH CLAUSE ANSWERS A SPECIFIC PAST MEASUREMENT FAILURE:

    * MATCHED CONTROL IS B2, NOT ARM A -- arm A pins the pre-flip flags as explicit
      constructor kwargs and `_fd_gate` ranks an explicit kwarg above the SUBMITTED_* default,
      so arm A is the PRE-flip agent. B2 differs from G by exactly one flag.
    * PER SEED, PER GAME -- not a pooled rate, not an any-seed union.
    * NO CONJUNCT ENCODES AN ASSUMED VALUE -- every clause is a difference measured in this
      same run, and the pass region's non-emptiness is COMPUTED
      (`_hud_mask_gate_pass_region_witness`) rather than asserted. That is the exp5835 defect
      this shape exists to prevent.
    * HEADROOM IS MEASURED -- `control_unwon_games` per seed, so a run where the control
      already wins everything is reported as UNINFORMATIVE rather than as a capability null
      (CLAUDE.md FALSE_NEGATIVE_RISK).
    * THE MECHANISM'S ACTIVITY IS A WITNESS -- an arm whose mask never resolved on any game is
      a CONTROL whatever its label says, and `mechanism_active` says so.
    * THE SAFETY AXIS IS REPORTED SEPARATELY AND IS ASYMMETRIC -- `collapse_guard_refusals`
      from arm G2. Over-masking destroys correctness; under-masking only costs efficiency. A
      guard that fires on a game the treatment wins is a reason to distrust that win, so the
      count is surfaced on the gate rather than left in the rows.
    """

    witness = _hud_mask_gate_pass_region_witness()
    out: dict[str, Any] = {
        "condition": (
            "on the REAL condition, PER SEED, against matched control B2: the treatment arm "
            "loses no game B2 wins on that seed AND gains at least one"
        ),
        "principle": (
            "a node-identity change earns a flag flip by making games REACHABLE that the "
            "shipped configuration cannot reach, on every seed, without giving one back -- "
            "because a mask that collapses genuinely distinct states is worse than no mask, "
            "so the regression clause is the load-bearing half"
        ),
        "matched_control_arm": HUD_MASK_CONTROL_ARM,
        "why_not_arm_A": (
            "arm A passes tier_exhaustion=False / frontier_gradient=False as explicit "
            "constructor kwargs and _fd_gate ranks an explicit kwarg above the SUBMITTED_* "
            "default, so arm A is the PRE-flip agent -- not the live one"
        ),
        "pass_region_nonempty": bool(witness.get("passes")),
        "pass_region_witness": witness,
        "signal_is_not_the_gate": (
            "r11l 0->1 and node_inflation falling are the PRE-REGISTERED SIGNAL that the "
            "mechanism worked as diagnosed. They are reported, never gated on: one public "
            "game is worth ~0 on the hidden set and gating on it would be hand-fitting"
        ),
        "per_arm": {},
    }
    control = _per_seed_win_sets(rows, HUD_MASK_CONTROL_ARM, condition)
    if not control:
        out["passed"] = False
        out["informative"] = False
        out["reason"] = f"matched control arm {HUD_MASK_CONTROL_ARM} not measured in this run"
        return out
    control_ran = _per_seed_measured_games(rows, HUD_MASK_CONTROL_ARM, condition)
    any_pass = False
    reachable_all: set[str] = set()
    for arm in HUD_MASK_ARMS:
        treat = _per_seed_win_sets(rows, arm, condition)
        treat_ran = _per_seed_measured_games(rows, arm, condition)
        shared = sorted(set(treat_ran) & set(control_ran))
        if not shared:
            out["per_arm"][arm] = {
                "measured": False,
                "reason": "no seed shared with the matched control",
                "passed": False,
                "headroom_present": False,
                "informative": False,
            }
            continue
        per_seed = []
        reachable_arm: set[str] = set()
        for seed in shared:
            treat_wins = treat.get(seed, set())
            control_wins = control.get(seed, set())
            new = sorted(treat_wins - control_wins)
            lost = sorted(control_wins - treat_wins)
            unwon = sorted(
                (control_ran.get(seed, set()) & treat_ran.get(seed, set())) - control_wins
            )
            reachable_arm |= set(unwon)
            per_seed.append(
                {
                    "seed": seed,
                    "new_wins": new,
                    "lost_wins": lost,
                    "n_treatment_wins": len(treat_wins),
                    "n_control_wins": len(control_wins),
                    "control_unwon_games": unwon,
                    "max_attainable_new_wins": len(unwon),
                    "n_games_at_ceiling": len(control_wins),
                }
            )
        gained = any(s["new_wins"] for s in per_seed)
        regressed = any(s["lost_wins"] for s in per_seed)
        passed = bool(gained and not regressed)
        any_pass = any_pass or passed
        reachable_all |= reachable_arm
        out["per_arm"][arm] = {
            "measured": True,
            "shared_seeds": shared,
            "per_seed": per_seed,
            "any_seed_gained": gained,
            "any_seed_regressed": regressed,
            "reachable_new_win_games": sorted(reachable_arm),
            "headroom_present": bool(reachable_arm),
            "informative": bool(reachable_arm) or passed,
            "passed": passed,
            "mechanism_active": _hud_arm_mechanism_active(rows, arm, condition),
            "signal_r11l": _hud_mask_signal(rows, arm, condition=condition),
            "safety": _hud_arm_safety(rows, arm, condition),
        }
    out["reachable_new_win_games"] = sorted(reachable_all)
    out["headroom_present"] = bool(reachable_all)
    out["n_reachable_new_win_games"] = len(reachable_all)
    out["headroom_narrow"] = bool(0 < len(reachable_all) <= 1)
    out["passed"] = bool(any_pass)
    out["informative"] = bool(reachable_all) or any_pass
    out["aliasing_attribution"] = _hud_aliasing_attribution(rows, out, condition)
    if out["headroom_narrow"]:
        out["headroom_narrow_note"] = (
            "the win axis had exactly ONE candidate game "
            f"({', '.join(sorted(reachable_all))}); every other game measured is already won by "
            "the matched control, so a failed gate here is a statement about that one game, not "
            "a corpus-level null. Run the full 25-game corpus (the baseline wins ~7/25, leaving "
            "~18 games of real headroom) before recording any capability null"
        )
    if not out["informative"]:
        out["verdict"] = "uninformative_no_headroom"
        out["reason"] = (
            "the matched control already wins every game measured on every shared seed, so the "
            "set of games where ANY treatment value could have gained a win is EMPTY. Report "
            "this run as UNTESTED for capability, not as a capability NULL (CLAUDE.md "
            "FALSE_NEGATIVE_RISK)"
        )
    else:
        out["verdict"] = "passed" if out["passed"] else "failed_with_headroom_present"
    return out


def _hud_aliasing_attribution(rows: Sequence[dict], gate: dict, condition: str = "real") -> dict:
    """WHOSE mask is aliasing -- the repair's newly-masked cells, or the ALREADY-SHIPPED mask?

    This distinction decides who owns the problem, and getting it wrong would credit (or
    blame) the wrong mechanism -- the failure mode this experiment family has already made
    once. It is computed, not asserted: for every game where a guard-armed arm LOST a win the
    matched control holds, compare that game's ``hud_mask_cell_count`` between control and
    treatment.

      * EQUAL cell counts  -> the repair added NOTHING on that game, so the aliasing the guard
        proved is in the mask the SHIPPED classifier already resolves there. That is a defect
        in the currently-live `SUBMITTED_AUTO_HUD_MASK_ENABLED=True` configuration, surfaced by
        this experiment but NOT introduced by it -- and it is operator-visible.
      * DIFFERENT cell counts -> the repair widened the mask on that game and the widening is
        the prime suspect.
    """

    def _cells(arm: str, game: str) -> set[int]:
        return {
            int(r["hud_mask_cell_count"])
            for r in rows
            if r.get("arm") == arm
            and r.get("game") == game
            and r.get("condition") == condition
            and r.get("ran")
            and r.get("hud_mask_cell_count") is not None
        }

    shipped_mask_games: list[dict] = []
    repair_widened_games: list[dict] = []
    for arm, detail in (gate.get("per_arm") or {}).items():
        if not detail.get("measured"):
            continue
        if not (detail.get("safety") or {}).get("guard_armed"):
            continue
        lost = sorted({g for s in detail.get("per_seed", []) for g in s.get("lost_wins", [])})
        for game in lost:
            control_cells = _cells(HUD_MASK_CONTROL_ARM, game)
            treat_cells = _cells(arm, game)
            row = {
                "arm": arm,
                "game": game,
                "control_mask_cells": sorted(control_cells),
                "treatment_mask_cells": sorted(treat_cells),
            }
            if control_cells and control_cells == treat_cells:
                shipped_mask_games.append(row)
            else:
                repair_widened_games.append(row)
    return {
        "regressions_attributable_to_the_ALREADY_SHIPPED_mask": shipped_mask_games,
        "regressions_attributable_to_the_REPAIR_widening_the_mask": repair_widened_games,
        "interpretation": (
            "a regression on a game where control and treatment mask the SAME number of cells "
            "cannot have been caused by the repair -- the repair added no cell there. It means "
            "the guard proved that the mask the SHIPPED classifier already applies on that game "
            "collapses behaviourally distinct states, and that the win on that game DEPENDS on "
            "that collapse. That is a pre-existing property of the live configuration, made "
            "visible here rather than created here"
        ),
        "shipped_mask_aliasing_detected": bool(shipped_mask_games),
    }


def _hud_arm_mechanism_active(rows: Sequence[dict], arm: str, condition: str = "real") -> dict:
    """Did the repaired detector actually resolve a mask this arm's control did not?

    An arm whose mask never resolved anywhere is a DUPLICATE of its control, and reading its
    result as "the detector had no effect" is a category error. `instrumented: False` means the
    rows carry no witness at all -- the state that previously let a dead mechanism look like a
    measured one.
    """

    resolved_games: set[str] = set()
    control_resolved_games: set[str] = set()
    instrumented = False
    total_cells = 0
    for row in rows:
        if row.get("condition") != condition or not row.get("ran"):
            continue
        if row.get("hud_mask_resolved") is None:
            continue
        if row.get("arm") == arm:
            instrumented = True
            total_cells += 1
            if row.get("hud_mask_resolved"):
                resolved_games.add(str(row.get("game")))
        elif row.get("arm") == HUD_MASK_CONTROL_ARM and row.get("hud_mask_resolved"):
            control_resolved_games.add(str(row.get("game")))
    newly = sorted(resolved_games - control_resolved_games)
    return {
        "instrumented": instrumented,
        "cells_instrumented": total_cells,
        "games_with_mask_resolved": sorted(resolved_games),
        "control_games_with_mask_resolved": sorted(control_resolved_games),
        # THE activity witness: games where the repair found a bar the shipped classifier missed.
        "games_newly_masked_vs_control": newly,
        "active": bool(instrumented and newly),
    }


def _hud_arm_safety(rows: Sequence[dict], arm: str, condition: str = "real") -> dict:
    """The SAFETY axis: did the collapse guard ever prove the mask was aliasing?

    Reported with its honest denominator. A violation is only OBSERVABLE for a (node, action)
    key tried at least twice, and the count grows with budget (measured on lf52: 1 of 9
    observable keys at 394 actions -> 3 of 30 at 765). So zero refusals is NOT proof of zero
    aliasing, and `guard_armed` must be checked before reading a zero at all.
    """

    armed = False
    refusals = 0
    provable_keys = 0
    excluded_by_control = 0
    declined = 0
    cells_with_live_control = 0
    cells_seen = 0
    revoked_cells: list[str] = []
    for row in rows:
        if row.get("arm") != arm or row.get("condition") != condition or not row.get("ran"):
            continue
        guard = (row.get("hud_mask") or {}).get("collapse_guard")
        if not guard:
            continue
        armed = True
        cells_seen += 1
        refusals += int(guard.get("collapse_refusals") or 0)
        provable_keys += int(guard.get("keys_with_multiple_successors") or 0)
        excluded_by_control += int(guard.get("non_deterministic_keys_excluded_by_control") or 0)
        declined += int(guard.get("uncontrolled_branchings_declined") or 0)
        if guard.get("control_live"):
            cells_with_live_control += 1
        if guard.get("globally_revoked"):
            revoked_cells.append(f"{row.get('game')}@seed{row.get('seed')}")
    return {
        "guard_armed": armed,
        "collapse_refusals": refusals if armed else None,
        "keys_with_multiple_successors": provable_keys if armed else None,
        "non_deterministic_keys_excluded_by_control": excluded_by_control if armed else None,
        # CONTROL-CHANNEL HEALTH. Every refusal above is only meaningful if the unmasked control
        # was live on that cell. The first version of this guard had a DEAD control (the graph
        # node's antecedent frame is not retained on a bare explorer run) and un-masked 6 nodes
        # on tu93 on zero evidence while reporting a clean-looking
        # `non_deterministic_keys_excluded_by_control: 0`. This field is what makes that state
        # visible instead of silent.
        "cells_with_live_control": cells_with_live_control if armed else None,
        "cells_instrumented": cells_seen if armed else None,
        "control_live_on_all_cells": bool(armed and cells_with_live_control == cells_seen),
        "uncontrolled_branchings_declined": declined if armed else None,
        "globally_revoked_cells": revoked_cells,
        "zero_refusals_is_not_proof_of_no_aliasing": True,
    }


def acceptance_gates(
    cap: dict, paired: dict, power: dict, rows: Sequence[dict] | None = None
) -> dict:
    """Explicit, COMPARATIVE, falsifiable gates -- self-reported pass/fail.

    WHY NOT an absolute first-win-rate threshold (the original spec's 0.12). A gate the NEGATIVE
    CONTROL passes measures nothing: the baseline arm already wins ~7 of 25 games (rate 0.28),
    so "rate >= 0.12" is cleared by doing nothing at all. Both gates below are therefore
    differences against the baseline measured in this same run.

    G-CAPABILITY: at least one NEW win vs the baseline, and no lost win among the
    regression-guard games. This is the gate that actually decides whether the graft is worth
    flipping on, and it is the one the measured result fails.

    G-EFFICIENCY: a PAIRED sign test p < 0.05 on actions-to-first-win over at least 6 paired
    games. The n>=6 floor is not arbitrary -- below it the exact sign test cannot reach 0.05 at
    all (see `power_ceiling`), so a "significant" result at n<6 would be an artifact of the
    test, and a null at n<6 is uninformative rather than negative.
    """

    gates: dict[str, Any] = {}
    if cap.get("available"):
        new_wins = int(cap.get("new_wins_vs_baseline") or 0)
        lost = list(cap.get("lost_wins_vs_baseline") or [])
        gates["acceptance_gate_capability"] = {
            "condition": "new_wins_vs_baseline >= 1 AND no lost baseline win",
            "principle": "a search-discipline graft earns a flag flip by SOLVING something the "
            "baseline cannot, not by being marginally cheaper on games already solved",
            "new_wins": new_wins,
            "lost_wins": lost,
            "passed": bool(new_wins >= 1 and not lost),
        }
    else:
        gates["acceptance_gate_capability"] = {
            "condition": "new_wins_vs_baseline >= 1 AND no lost baseline win",
            "passed": False,
            "reason": "no baseline comparison available",
        }

    best: dict[str, Any] = {"arm": None, "p_value": None, "n": 0}
    for arm in GRAFTED_ARMS:
        cond = (paired.get(arm) or {}).get("real")
        if not cond:
            continue
        p = (cond.get("sign_test") or {}).get("p_value")
        n = int(cond.get("n_paired_games") or 0)
        if p is None:
            continue
        if best["p_value"] is None or p < best["p_value"]:
            best = {"arm": arm, "p_value": p, "n": n}
    gates["acceptance_gate_efficiency"] = {
        "condition": "paired sign-test p < 0.05 on actions-to-first-win, n_paired_games >= 6",
        "principle": "pairs the arms on the SAME game so the test sees the intervention rather "
        "than between-game difficulty; the n floor keeps a 'significant' result from being an "
        "artifact of a sample too small for the test to attain 0.05",
        "best_arm": best["arm"],
        "best_p_value": best["p_value"],
        "n_paired_games": best["n"],
        "min_n_required": 6,
        "passed": bool(best["p_value"] is not None and best["p_value"] < 0.05 and best["n"] >= 6),
        "power_ceiling": power,
    }
    # Flat BOOLEAN mirrors. scripts/summarize_artifact.py marks a gate PASS/FAIL only when the
    # field's value is literally True/False; a dict-valued gate renders as "?" and would not trip
    # its ">> a FAILED gate overrides any celebratory verdict" line. The structured dicts above
    # carry the reasoning; these three carry the verdict the reading tool can act on.
    gates["acceptance_gate_capability_passed"] = bool(
        gates["acceptance_gate_capability"].get("passed")
    )
    gates["acceptance_gate_efficiency_passed"] = bool(
        gates["acceptance_gate_efficiency"].get("passed")
    )
    gates["acceptance_gates_all_passed"] = bool(
        gates["acceptance_gate_capability_passed"] and gates["acceptance_gate_efficiency_passed"]
    )
    # REQ-ARC-WMTE-5950's own gate, kept OUT of acceptance_gates_all_passed on purpose: that
    # field is already on the record as the frontier-discipline verdict, and folding a second
    # mechanism into it would redefine it. The flat mirror is what summarize_artifact reads.
    if rows is not None:
        cps_gate = click_pixel_sampling_gate(rows)
        gates["acceptance_gate_click_pixel_sampling"] = cps_gate
        gates["acceptance_gate_click_pixel_sampling_passed"] = bool(cps_gate.get("passed"))
        # REQ-ARC-WMTE-5960's own gate, kept OUT of acceptance_gates_all_passed for the same
        # reason as the sampler's: that field is already on the record as the
        # frontier-discipline verdict, and folding a third mechanism into it would redefine a
        # published number.
        hud_gate = hud_mask_gate(rows)
        gates["acceptance_gate_hud_mask"] = hud_gate
        gates["acceptance_gate_hud_mask_passed"] = bool(hud_gate.get("passed"))
    return gates


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

    # SELECTION BUG FIX (2026-07-25, found by the operator on the first full run): this previously
    # took the FIRST `limit` winning rows in row order. Arm A (baseline) is measured first, so all
    # 6 slots were consumed by baseline wins and the NEW wins -- the entire headline claim -- were
    # never eligible for reproduction. That silently violated this project's own rule that only
    # reproduced levels count. Now the sample is ROUND-ROBIN BY ARM, so every arm that won anything
    # is represented and the grafted/control arms carrying the claim are always checked.
    # LIMIT-TRUNCATION FIX (2026-07-25, second pass of the same defect class): round-robin
    # alone was not enough. The smoke ran with limit=4 while FIVE arms had wins, and the
    # round-robin walks sorted(by_arm) and simply STOPS at the limit -- so arm F1, one of the
    # two arms carrying the sampler claim and the CLEAN single-variable arm, was never
    # reproduction-checked, while the artifact presented the gate as fixed. The effective
    # limit is therefore floored at the number of arms with wins, which guarantees every such
    # arm gets at least one slot no matter how the caller sets --replay-limit.
    all_wins = [r for r in rows if r.get("ran") and r.get("levels", 0) > 0]
    by_arm: dict[str, list[dict]] = {}
    for row in all_wins:
        by_arm.setdefault(str(row.get("arm")), []).append(row)
    arms_with_wins = sorted(by_arm)
    requested_limit = max(0, int(limit))
    effective_limit = max(requested_limit, len(arms_with_wins)) if arms_with_wins else 0
    wins: list[dict] = []
    while len(wins) < effective_limit and any(by_arm.values()):
        for arm_key in sorted(by_arm):
            if len(wins) >= effective_limit:
                break
            if by_arm[arm_key]:
                wins.append(by_arm[arm_key].pop(0))
    checks = []
    for r in wins:
        cond = next((c for c in CONDITIONS if c[0] == r["condition"]), None)
        if cond is None:
            continue
        again = run_cell(
            r["arm"],
            r["game"],
            budget=budget,
            seed=r["seed"],
            variant=cond[1],
            reflect=cond[2],
            je_runner=je_runner,
        )
        checks.append(
            {
                "arm": r["arm"],
                "game": r["game"],
                "condition": r["condition"],
                "seed": r["seed"],
                "levels_first": r["levels"],
                "levels_replay": again.get("levels"),
                "reproduced": bool(again.get("levels") == r["levels"]),
            }
        )
    n_ok = sum(1 for c in checks if c["reproduced"])
    checked_arms = {str(c["arm"]) for c in checks}
    # EXPLICIT, not implied by n_checked: which arms that won something were NOT reproduced.
    # The claim-carrying arms are named so a silent drop is impossible to present as a pass.
    not_reproduced = sorted(a for a in arms_with_wins if a not in checked_arms)
    # REQ-ARC-WMTE-5960: the HUD arms carry a claim too, so they must be reproduction-eligible.
    # Without this they would be silently omitted from `claim_carrying_arms_not_reproduced`,
    # which is the exact defect class already fixed twice above for the sampler arms.
    claim_arms = (
        tuple(CLICK_PIXEL_ARMS) + tuple(HUD_MASK_ARMS) + (CLICK_PIXEL_CONTROL_ARM,)
    )
    return {
        "method": "re_execution_of_the_same_cell_not_kit_reproduce",
        "replay_limit_requested": requested_limit,
        "replay_limit_effective": effective_limit,
        "n_arms_with_wins": len(arms_with_wins),
        "arms_with_wins": arms_with_wins,
        "arms_reproduced": sorted(checked_arms),
        "arms_not_reproduced": not_reproduced,
        "claim_carrying_arms": list(claim_arms),
        "claim_carrying_arms_not_reproduced": sorted(
            a for a in not_reproduced if a in set(claim_arms)
        ),
        "all_arms_with_wins_reproduced": not not_reproduced,
        "why_not_kit_reproduce": (
            "kit.reproduce consumes banked string action LABELS; the live explorer emits "
            "(action_id, data) tuples, so it is not applicable to this trajectory shape"
        ),
        "level_source": "env.levels_completed (frame truth, not a self-report)",
        "n_checked": len(checks),
        "n_reproduced": n_ok,
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

    _add(
        "offline_arcade_environment_files",
        lambda: (REPO / "environment_files").is_dir(),
        str(REPO / "environment_files"),
    )
    _add(
        "arc_leaderboard_eval_harness",
        lambda: (REPO / "scripts" / "arc_leaderboard_eval.py").is_file(),
    )
    _add(
        "frontier_discipline_module",
        lambda: (
            __import__("carnot.agentic.arc_frontier_discipline", fromlist=["TierExhaustionPolicy"])
            is not None
        ),
    )
    _add(
        "live_explorer_flags_wired",
        lambda: all(
            hasattr(
                __import__(
                    "carnot.agentic.arc_competition_agent", fromlist=["StepwiseExplorer"]
                ).StepwiseExplorer(),
                attr,
            )
            for attr in ("tier_exhaustion_enabled", "frontier_gradient_enabled")
        ),
    )
    _add(
        "just_explore_reference_clone",
        lambda: (JE_ROOT / "graph_explorer.py").is_file(),
        str(JE_ROOT),
    )
    _add(
        "variant_generator",
        lambda: (
            __import__("carnot.agentic.arc_variant_generator", fromlist=["VariantEnv"]) is not None
        ),
    )
    # Explicitly asserted ABSENT: no LLM is loaded on any arm. Arms A-D use
    # CarnotAgentPolicy(force_explore=True), which has no proposer parameter at all; arm E is
    # the reference solver, which uses no model by design.
    out.append(
        {
            "resource": "llm_proposer_deliberately_absent",
            "available": True,
            "detail": "arms A-D use CarnotAgentPolicy(force_explore=True); arm E uses no model",
        }
    )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_scope(
    games: Sequence[str],
    arms: Sequence[str],
    cond_specs: Sequence[Any],
    budget: int,
) -> dict:
    """Did this invocation cover the FULL declared spec, or is it a reduced smoke?

    Filed on adversarial-review instruction (2026-07-24): every number this experiment has ever
    quoted came from a 3-game smoke at one fifth of the declared budget, while the artifact's
    verdict said "measured". Scope is now computed mechanically and stamped on the artifact, so a
    smoke number cannot be mistaken for the full result by a reader or by a capstone.
    """

    full = (
        set(games) == set(ALL_GAMES)
        and set(a for a in arms if a in ARMS) == set(ARMS)
        and len(cond_specs) == len(CONDITIONS)
        and int(budget) >= DEFAULT_BUDGET
    )
    return {
        "full_declared_spec": bool(full),
        "n_games": len(games),
        "n_games_declared": len(ALL_GAMES),
        "n_arms": len([a for a in arms if a in ARMS]),
        "n_arms_declared": len(ARMS),
        "n_conditions": len(cond_specs),
        "budget": int(budget),
        "budget_declared": DEFAULT_BUDGET,
    }


def verdict_for(
    scope: dict,
    cap: dict,
    *,
    positive_control_ran: bool,
    error_rate: float,
    control_health: Optional[dict] = None,
    cps_gate: Optional[dict] = None,
) -> str:
    """The honest_verdict string. States SCOPE and the CAPABILITY result, not the efficiency delta.

    Terminal-prefixed per CLAUDE.md's Verdict Terminal-Prefix Discipline. A reduced-scope run gets
    a ``partial_`` prefix on purpose: it IS a partial execution of the declared experiment, and
    the reconciler classifying it as such is the correct outcome -- better than a ``complete_``
    verdict whose descriptive tail nobody reads.

    Two branches added 2026-07-25, both because the previous verdict was silently confident
    about things the run had not established:

    * ``uninterpretable_arm_error_rate_*`` -- a NON-run-blocking error rate. ``error_rate``
      above counts only cells that failed to RUN; a cell that ran with 96% of its decisions
      raising is invisible to it. That is how a majority-degenerate positive control produced a
      verdict crediting it with a new win "under identical conditions".
    * ``sampler_untested_no_headroom`` -- the sampler gate had ZERO games where a new win was
      even attainable, so its ``passed: False`` is an uninformative test, not a null. Naming it
      in the verdict is what keeps the artifact's FIRST line from implying a measured null.
    """

    cap_tail = ""
    if cap.get("available"):
        cap_tail = (
            f"_graft_new_wins_{int(cap.get('new_wins_vs_baseline') or 0)}"
            f"_control_new_wins_{int(cap.get('positive_control_new_wins') or 0)}"
        )
    if not positive_control_ran:
        return "complete_frontier_discipline_ab_measured_but_uninterpretable_no_positive_control"
    if error_rate > 0.05:
        return (
            "complete_frontier_discipline_ab_measured_but_uninterpretable_"
            f"errored_cell_rate_{error_rate:.2f}"
        )
    if control_health is not None and not control_health.get("healthy", True):
        return (
            "complete_frontier_discipline_ab_measured_but_uninterpretable_arm_error_rate_"
            f"positive_control_degenerate_{control_health.get('n_degenerate_cells')}"
            f"_of_{control_health.get('n_cells')}_cells_worst_"
            f"{float(control_health.get('worst_cell_fallback_fraction') or 0.0):.2f}"
        )
    cps_tail = ""
    if cps_gate is not None and "informative" in cps_gate:
        cps_tail = (
            "_sampler_untested_no_headroom"
            if not cps_gate.get("informative")
            else (
                "_sampler_gate_passed"
                if cps_gate.get("passed")
                else (
                    "_sampler_gate_failed_but_headroom_was_"
                    f"{int(cps_gate.get('n_reachable_new_win_games') or 0)}_game_only"
                    if cps_gate.get("headroom_narrow")
                    else "_sampler_gate_failed_with_headroom"
                )
            )
        )
    if scope.get("full_declared_spec"):
        return "complete_frontier_discipline_ab_measured" + cap_tail + cps_tail
    return (
        f"partial_frontier_discipline_ab_smoke_scale_{int(scope.get('n_games') or 0)}games_"
        f"budget{int(scope.get('budget') or 0)}_not_full_spec" + cap_tail + cps_tail
    )


def build_headline(
    cap: dict,
    *,
    control_health: dict,
    cps_gate: Optional[dict] = None,
    sampler_run: bool = False,
) -> str:
    """The artifact's one-line summary. Shared by ``run()`` and the ``--summarize`` path.

    Extracted 2026-07-25 because the two paths had DIVERGED into two hand-maintained f-string
    ladders, and both led with the frontier-discipline graft even on a run whose whole purpose
    was the click-pixel sampler -- so the sampler's own result did not appear in the field a
    reader sees first. Two disclosures the previous headline omitted entirely and which changed
    what the run means:

    * the positive control's DEGENERACY, so "under identical conditions" is not asserted for a
      control that spent most of its budget in its own failed=True fallback; and
    * the sampler gate's HEADROOM, so a failed gate over a one-game (or empty) win axis is not
      read as a corpus-level capability null.
    """

    parts: list[str] = []
    if sampler_run and isinstance(cps_gate, dict):
        active = [
            arm
            for arm, info in (cps_gate.get("per_arm") or {}).items()
            if (info.get("mechanism_active") or {}).get("active")
        ]
        changed = sum(
            int((info.get("mechanism_active") or {}).get("coordinates_changed") or 0)
            for info in (cps_gate.get("per_arm") or {}).values()
        )
        if not cps_gate.get("informative"):
            parts.append(
                "REQ-ARC-WMTE-5950 click-pixel sampling: mechanism ACTIVE "
                f"({changed} click coordinates replaced across arms {sorted(active)}) but the "
                "gate is UNINFORMATIVE on this corpus -- 0 games where a new win was attainable "
                "(the matched control already wins everything it measured), so the sampler is "
                "UNTESTED for capability here, NOT a measured null"
            )
        else:
            reach = cps_gate.get("reachable_new_win_games") or []
            parts.append(
                "REQ-ARC-WMTE-5950 click-pixel sampling: mechanism ACTIVE "
                f"({changed} click coordinates replaced across arms {sorted(active)}); gate "
                f"{'PASSED' if cps_gate.get('passed') else 'FAILED'} with a win axis of "
                f"{len(reach)} attainable game(s) {sorted(reach)}"
                + (
                    " -- a ONE-game win axis, so this is a statement about that game rather "
                    "than a corpus-level null"
                    if cps_gate.get("headroom_narrow")
                    else ""
                )
            )
    if not cap.get("available"):
        parts.append("no baseline comparison available -- no capability claim")
        return "; ".join(parts)
    graft = (
        "grafted tier exhaustion + distance gradient produced "
        f"{int(cap.get('new_wins_vs_baseline') or 0)} new win(s) vs the baseline on the measured "
        "corpus"
    )
    control = (
        "the just-explore reference positive control produced "
        f"{int(cap.get('positive_control_new_wins') or 0)} new win(s)"
    )
    if control_health.get("healthy"):
        control += " under identical conditions"
    else:
        control += (
            ", but was DEGENERATE on "
            f"{int(control_health.get('n_degenerate_cells') or 0)} of "
            f"{int(control_health.get('n_cells') or 0)} cells (worst "
            f"{float(control_health.get('worst_cell_fallback_fraction') or 0.0):.0%} of actions in "
            "its own failed=True repeat-last-action fallback), so its LOSSES are shim artifacts "
            "and 'under identical conditions' does NOT hold"
        )
    parts.extend([graft, control])
    return "; ".join(parts)


def _reproducibility_checksum(payload: dict) -> str:
    """Content hash over the measured rows + the run configuration.

    Anchors the artifact to its inputs: a later replication that produces different rows
    produces a different checksum, so silent drift in the corpus, the arms, or the budget
    cannot masquerade as the same experiment.
    """

    blob = json.dumps(
        {"rows": payload.get("per_cell_rows"), "config": payload.get("config")},
        sort_keys=True,
        default=str,
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
    "new_wins_vs_baseline": (
        "THE decision-relevant number: games the grafted arms win that the baseline does not. "
        "Published at top level so no downstream aggregation can carry this experiment's "
        "efficiency delta forward without the capability result attached to it."
    ),
    "positive_control_new_wins": (
        "The same number for the just-explore reference. Its whole job is to make the graft's "
        "capability number interpretable: graft=0 with control=0 means the corpus had no "
        "headroom, whereas graft=0 with control>0 means the graft failed to reproduce a "
        "transfer the reference demonstrably achieves under identical conditions."
    ),
    "paired_efficiency_vs_baseline": (
        "The inferential efficiency statistic. Arms run on the SAME games with the SAME seeds, "
        "so pairing is free; pooling across games instead lets between-game difficulty variance "
        "hide an effect the per-game deltas show unanimously."
    ),
    "power_ceiling": (
        "The smallest p-value this corpus can attain at all. Stated up front so an underpowered "
        "null is never read as evidence of absence."
    ),
    "acceptance_gate_capability": (
        "A gate must be able to FAIL for the intervention and PASS for a real improvement. An "
        "absolute win-rate threshold the baseline already clears cannot; a new-wins-vs-baseline "
        "gate can."
    ),
    "run_scope": (
        "Records whether this run covered the full declared spec (all games/arms/conditions at "
        "the declared budget) or was a reduced smoke, so a number quoted from a smoke can never "
        "be mistaken for the full result."
    ),
    "levels_capped_by_early_stop": (
        "Arms A-D stop at the first level-up (target_levels=1) while arm E runs to budget, so "
        "banked-level counts are not arm-comparable; the flag travels with the number."
    ),
    "click_pixel_coordinates_changed": (
        "The mechanism's ACTIVITY WITNESS: how many click coordinates the sampler actually "
        "replaced. A treatment arm reporting zero here is a control regardless of its label, so "
        "its null cannot be read as 'the mechanism had no effect'. click_pixel_rows_sampled is "
        "NOT this quantity -- it counts rows present and is identical for a dead sampler."
    ),
    "positive_control_health": (
        "Whether arm E was a measurement OF THE REFERENCE at all. A cell that returns while most "
        "of its decisions end in the reference's own choose_action raising is spending its budget "
        "in a self-flagged repeat-last-action fallback, so its losses are shim artifacts. Without "
        "this, a 96%-degenerate control read as a clean null and set the forward diagnostic."
    ),
    "reachable_new_win_games": (
        "The games where a NEW win was even attainable (measured by both arms on a shared seed "
        "and NOT already won by the control). Empty means the gate's pass region was empty ON "
        "THIS CORPUS, so a failed gate is an uninformative test, not evidence of no effect "
        "(CLAUDE.md FALSE_NEGATIVE_RISK)."
    ),
    "arms_not_reproduced": (
        "Arms that won something but were never re-executed by the reproduction gate. Named "
        "explicitly because a --replay-limit below the number of winning arms silently truncated "
        "a claim-carrying arm out of the sample while the gate still reported n/n reproduced."
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
    blocking = [
        p
        for p in pre
        if not p["available"]
        and p["resource"]
        in (
            "offline_arcade_environment_files",
            "arc_leaderboard_eval_harness",
            "frontier_discipline_module",
            "live_explorer_flags_wired",
        )
    ]
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
                        row = run_cell(
                            arm,
                            game,
                            budget=budget,
                            seed=seed,
                            variant=variant,
                            reflect=reflect,
                            je_runner=je_runner,
                        )
                    except Exception as exc:
                        # Record the error rather than silently dropping a game: a dropped game
                        # changes the denominator and quietly biases every rate.
                        row = {
                            "arm": arm,
                            "game": game,
                            "seed": seed,
                            "ran": False,
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    row["condition"] = cond_label
                    rows.append(row)

    errored = [r for r in rows if not r.get("ran")]
    agg = aggregate(rows, games)
    cmp_ = compare_to_baseline(agg, games, rows)
    guard_games, guard_provenance = _guard_games_from_rows(rows)
    paired = paired_efficiency_vs_baseline(rows)
    power = power_ceiling(games, guard_games)
    control_health = positive_control_health(rows)
    cap = capability_summary(agg, cmp_, control_healthy=bool(control_health.get("healthy")))
    gates = acceptance_gates(cap, paired, power, rows)
    repro = replay_validate(rows, budget=budget, je_runner=je_runner, limit=replay_limit)

    positive_control_ran = any(r.get("arm") == "E" and r.get("ran") for r in rows)
    # An A/B is interpretable only if the positive control ran AND most cells actually produced a
    # measurement. A run where a third of the cells errored is a broken harness reporting numbers,
    # not a result -- an earlier smoke of this very file silently produced "complete" with 36 of
    # 72 cells errored on a signature mismatch, which is exactly the failure this gate closes.
    error_rate = (len(errored) / len(rows)) if rows else 1.0
    # positive_control_ran is now gated on the control being a MEASUREMENT OF THE REFERENCE, not
    # merely on the cell returning. A majority-degenerate arm E cannot underwrite the "the
    # reference does X under identical conditions" claim the headline was making.
    positive_control_usable = bool(positive_control_ran and control_health.get("healthy"))
    interpretable = bool(positive_control_usable and error_rate <= 0.05)

    scope = run_scope(games, arms, cond_specs, budget)
    cps_gate_for_verdict = gates.get("acceptance_gate_click_pixel_sampling")
    verdict = verdict_for(
        scope,
        cap,
        positive_control_ran=positive_control_ran,
        error_rate=error_rate,
        control_health=control_health,
        cps_gate=cps_gate_for_verdict,
    )

    config = {
        "games": list(games),
        "arms": list(arms),
        "conditions": [c[0] for c in cond_specs],
        "budget_actions_per_game": int(budget),
        "n_seeds_stochastic_arms": int(n_seeds),
        # Was ABSENT, so "reproduction gate 4/4" was unauditable from the artifact -- the reader
        # could not tell that the sample size had truncated a claim-carrying arm out of it.
        "replay_limit": int(replay_limit),
        "llm_disabled": True,
        "policy_kind": "explorer_force_explore_no_proposer",
    }
    # ARTIFACT IDENTITY (2026-07-25). This module hosts TWO requirements: the frontier-discipline
    # graft (REQ-ARC-WMTE-5836, arms A-E) and the click-pixel sampler (REQ-ARC-WMTE-5950, arms
    # F/F1). The identity fields were hardcoded to 5836, so a 5950 run wrote an artifact
    # declaring experiment_id 5836 -- which folds it into an already-published record (four
    # existing artifacts carry experiment=5836) and leaves REQ-ARC-WMTE-5950 with no artifact
    # claiming it. Identity is therefore derived from WHICH ARMS RAN, not hardcoded.
    sampler_arms_present = [a for a in arms if a in set(CLICK_PIXEL_ARMS)]
    graft_arms_present = [a for a in arms if a in set(GRAFTED_ARMS) - set(CLICK_PIXEL_ARMS)]
    sampler_run = bool(sampler_arms_present)
    exp_id = CLICK_PIXEL_EXPERIMENT_ID if sampler_run else EXPERIMENT_ID
    requirement = "REQ-ARC-WMTE-5950" if sampler_run else "REQ-ARC-WMTE-5836"
    title = (
        "Click-pixel sampling A/B (REQ-ARC-WMTE-5950): per-object uniform pixel vs truncated "
        "centroid, against the live configuration"
        if sampler_run
        else "Frontier-discipline A/B: just-explore tier exhaustion + distance gradient"
    )
    art: dict[str, Any] = {
        "experiment": exp_id,
        "experiment_id": exp_id,
        "title": title,
        "requirement": requirement,
        "requirements_exercised": (["REQ-ARC-WMTE-5950"] if sampler_run else [])
        + (["REQ-ARC-WMTE-5836"] if graft_arms_present or not sampler_run else []),
        "identity_derivation": (
            "experiment_id/requirement are derived from which arms ran: arms F/F1 present -> "
            "REQ-ARC-WMTE-5950 (the click-pixel sampler); otherwise REQ-ARC-WMTE-5836 (the "
            "frontier-discipline graft). Hardcoding 5836 previously made a 5950 run write an "
            "artifact that folded into the already-published 5836 record"
        ),
        "reference": "arXiv:2512.24156 (just-explore, ARC-AGI-3 Preview private leaderboard 3rd)",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": True,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - t0, 3),
        "preconditions_checked": pre,
        "config": config,
        "arms": {
            k: {"label": v["label"], "kwargs": v["kwargs"], "deterministic": v["deterministic"]}
            for k, v in ARMS.items()
        },
        # GATED on the control being a measurement OF THE REFERENCE (see
        # positive_control_health): a cell that returned while 78-96% of its decisions ended in
        # the reference's own choose_action raising is not a usable positive control.
        "positive_control_ran": positive_control_usable,
        "positive_control_cell_returned": positive_control_ran,
        "positive_control_reason": (
            je_reason if positive_control_usable else str(control_health.get("reason"))
        ),
        "positive_control_health": control_health,
        "ab_interpretable": interpretable,
        "run_scope": scope,
        # ---- HEADLINE: the sampler's own result FIRST on a sampler run, then the capability
        # result, then the control's health caveat. See build_headline for why. ----
        "headline": build_headline(
            cap,
            control_health=control_health,
            cps_gate=cps_gate_for_verdict,
            sampler_run=sampler_run,
        ),
        "new_wins_vs_baseline": cap.get("new_wins_vs_baseline"),
        "positive_control_new_wins": cap.get("positive_control_new_wins"),
        "capability_summary": cap,
        **gates,
        "paired_efficiency_vs_baseline": paired,
        "power_ceiling": power,
        "regression_guard_provenance": guard_provenance,
        "per_cell_rows": rows,
        "aggregates": agg,
        "vs_baseline": cmp_,
        "reproduction_gate": repro,
        "n_errored_cells": len(errored),
        "errored_cell_rate": round(error_rate, 4),
        # CLAUDE.md: an exactly-0.0 metric needs an explanation, or it is indistinguishable from
        # an uninitialised stub default. errored_cell_rate is a HARNESS-health counter, not a
        # capability metric: 0.0 means every (arm, game, condition, seed) cell returned a real
        # measurement. It is legitimately zero on a healthy run and is NOT a performance claim.
        # A non-zero value is the interesting case (>0.05 flips ab_interpretable to false).
        "methodology_note": (
            "errored_cell_rate counts ONLY cells that failed to RUN. It is NOT a measure of "
            "within-cell health: a cell that ran to completion with most of its decisions "
            "raising internally still contributes 0 here. Read positive_control_health (arm E's "
            "own degenerate-fallback fraction) and each row's click_pixel_coordinates_changed "
            "alongside it -- the earlier claim that 'errored_cell_rate == 0.0 means every cell "
            "produced a real measurement' was misleading, and is retracted. errored_cell_rate is "
            "a harness-health counter, not a capability metric, so zero is the expected healthy "
            "value rather than an implausibly-perfect result. Level counts come from the "
            "environment's own levels_completed (execution-grounded), which is why "
            "verifier_is_oracle is declared True: this is a SEARCH-EFFICIENCY measurement, not "
            "an oracle-distinct verifier claim."
        ),
        "errored_cells": errored[:40],
        "action_vocabulary_strata": {
            "click_games": list(CLICK_GAMES),
            "nav_only_games": list(NAV_ONLY_GAMES),
        },
        "spec_deviations": [
            {
                "spec": "budget 200",
                "actual": f"budget {budget} (default {DEFAULT_BUDGET})",
                "why": "budget 200 measured 0/25 first-wins -> no arm distinguishable; measured "
                "first-win costs span 20 (lp85) to 1747 (cd82) actions",
            },
            {
                "spec": ">=3 seeds on every arm",
                "actual": "1 run for the bit-deterministic arms (A/B/C/D); >=3 seeds only for the "
                "stochastic arms (B2 uniform draw, E reference)",
                "why": "the explorer is bit-deterministic (verified across 3 seeds x 3 games), so "
                "seeding it produces identical rows and a fake-tight CI",
            },
            {
                "spec": "colour-permuted variants measure generalization",
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
                "not share), reported in vs_baseline.recolor_control, not a harness bug",
            },
            {
                "spec": "solves validated with kit.reproduce",
                "actual": "re-execution of the same cell; level-ups read from env.levels_completed",
                "why": "kit.reproduce consumes banked string action LABELS, which the live explorer "
                "never emits -- claiming it ran would be false",
            },
            {
                "spec": "reference tier-advancement trigger (graph_explorer._maybe_advance_group: "
                "advance while the CURRENT node's distance to any frontier node == INFINITY)",
                "actual": "GLOBAL SET-EXHAUSTION (advance only when NO node anywhere still has "
                "work at the active tier) -- TierExhaustionPolicy.next_active_tier",
                "why": "the reference's literal trigger can never fire in Carnot: Carnot can "
                "RESET and replay any node's recorded path from the root, so no node is ever "
                "unreachable and no distance is ever INFINITY. Global set-exhaustion is the "
                "faithful analogue of 'no open node is reachable' in a graph where everything "
                "is reachable",
                "consequence_for_interpretation": "the graft is a STRICTER variant than the "
                "reference runs: the reference advances while lower-tier work still exists "
                "elsewhere-but-unreachable-forward, this graft does not. A NULL ON ARMS "
                "B/B2/D THEREFORE DOES NOT FALSIFY THE REFERENCE MECHANISM -- it falsifies "
                "this stricter substitution. Testing the literal trigger needs a further arm "
                "gated on `nearest_open_node(...) is None` (the module already computes it)",
            },
            {
                "spec": "reference within-tier draw (random.choice over ALL untested edges in "
                "groups 0..active_group -- no top-k)",
                "actual": "faithful: top_k defaults to None (unrestricted uniform draw)",
                "why": "this was a REAL DEVIATION until 2026-07-24: the wiring passed the "
                "unrelated hybrid-diversity knob CARNOT_ARC_EXPLORE_DIV_TOPK (default 8), which "
                "silently made arm B2 a top-8 draw AND coupled the arm to a foreign env var. "
                "Fixed; the draw now has its own knob (CARNOT_ARC_FRONTIER_TIER_DRAW_TOPK), "
                "unset by default. If that env var IS set, this run is a top-k deviation -- see "
                "per_cell_rows[*].frontier_discipline.tier_draw_top_k, which records the value "
                "actually used",
            },
            {
                "spec": "arm E action count as reported by the reference (agent.action_counter)",
                "actual": "reset-EXCLUDED count, matching arc_leaderboard_eval.run_game's "
                "convention for arms A-D; the raw counter is kept as "
                "actions_to_first_levelup_incl_reset",
                "why": "the reference increments action_counter once per main-loop iteration "
                "INCLUDING RESET, while run_game increments `actions` only in its non-RESET "
                "branch. Reporting the raw counter charged the positive control for resets that "
                "arms A-D got free, biasing the efficiency axis against the control",
            },
            {
                "spec": "acceptance gate: first-win rate >= 0.12",
                "actual": "replaced by two COMPARATIVE gates (see acceptance_gate_capability / "
                "acceptance_gate_efficiency)",
                "why": "0.12 x 25 games = 3 wins, but the BASELINE arm already wins ~7 of 25 "
                "(rate 0.28) -- a gate the negative control passes cannot separate the "
                "intervention from doing nothing, so it was non-discriminative in principle",
            },
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
            "levels_banked_total is PER-ARM ONLY. Arms A-D run target_levels=1 so is_done stops "
            "them at the first level-up (levels capped at 1 per game by construction); arm E runs "
            "to budget. Comparing that column across arms attributes the stopping rule to "
            "capability. Every explorer row carries levels_capped_by_early_stop=true.",
            "mean_actions_to_first_win pools across GAMES and is descriptive only: between-game "
            "difficulty (20 to 1747 baseline actions) dominates its interval. The inferential "
            "statistic is paired_efficiency_vs_baseline (per-game paired deltas + exact sign "
            "test + bootstrap median CI).",
            "POWER: the paired test has at most one delta per baseline-win game, so the smallest "
            "attainable two-sided p is 2*0.5^k for k such games -- 0.031 at k=6, i.e. it clears "
            "0.05 only if EVERY pair favours the arm. See power_ceiling; a flat sign test at this "
            "n is underpowered, not evidence of absence.",
            "MECHANISM (b) SCOPE: the distance gradient no longer returns a depth-capped current "
            "node (fixed 2026-07-24). Before that fix it did so on 100% of its picks on deep-graph "
            "games, which meant arms C/D were partly measuring the REMOVAL of the max_depth=45 "
            "backtrack cap rather than the gradient. Any C/D number from before that fix is "
            "confounded and must not be compared with these.",
        ],
    }
    art["reproducibility_checksum"] = _reproducibility_checksum(art)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art, indent=2, default=str))
    return art


def recompute_derived(art: dict) -> dict:
    """Re-derive every ANALYSIS section from the artifact's own measured rows. Measures nothing.

    Why this exists: the measured rows (``per_cell_rows``) are expensive to produce and are the
    only thing in the artifact that required running the agent, whereas the aggregates, paired
    statistics, capability summary, gates and verdict are pure functions OF those rows. When the
    analysis code changes -- e.g. because a review found the primary statistic was the wrong one
    -- re-running hours of identical measurement to pick up the new analysis would be waste, and
    hand-editing the artifact would be indefensible. This recomputes the derived sections with
    the CURRENT code and leaves every measured field (rows, durations, preconditions) untouched.

    It CANNOT invent a measurement: with no rows it returns the artifact unchanged. The
    reproducibility checksum is computed over rows + config only, so a recompute does not change
    it -- which is the point: same measurement, better analysis.
    """

    rows = art.get("per_cell_rows")
    if not rows:
        return art
    cfg = art.get("config") or {}
    games = list(cfg.get("games") or sorted({r.get("game") for r in rows if r.get("game")}))
    arms = list(cfg.get("arms") or sorted({r.get("arm") for r in rows if r.get("arm")}))
    conds = list(cfg.get("conditions") or sorted({r.get("condition") for r in rows}))
    cond_specs = [c for c in CONDITIONS if c[0] in set(conds)]
    budget = int(cfg.get("budget_actions_per_game") or DEFAULT_BUDGET)

    errored = [r for r in rows if not r.get("ran")]
    agg = aggregate(rows, games)
    cmp_ = compare_to_baseline(agg, games, rows)
    guard_games, guard_provenance = _guard_games_from_rows(rows)
    paired = paired_efficiency_vs_baseline(rows)
    power = power_ceiling(games, guard_games)
    control_health = positive_control_health(rows)
    cap = capability_summary(agg, cmp_, control_healthy=bool(control_health.get("healthy")))
    gates = acceptance_gates(cap, paired, power, rows)
    positive_control_ran = any(r.get("arm") == "E" and r.get("ran") for r in rows)
    positive_control_usable = bool(positive_control_ran and control_health.get("healthy"))
    error_rate = (len(errored) / len(rows)) if rows else 1.0
    scope = run_scope(games, arms, cond_specs, budget)
    cps_gate_for_verdict = gates.get("acceptance_gate_click_pixel_sampling")

    art.update(
        {
            "aggregates": agg,
            "vs_baseline": cmp_,
            "paired_efficiency_vs_baseline": paired,
            "power_ceiling": power,
            "capability_summary": cap,
            "new_wins_vs_baseline": cap.get("new_wins_vs_baseline"),
            "positive_control_new_wins": cap.get("positive_control_new_wins"),
            "positive_control_ran": positive_control_usable,
            "positive_control_cell_returned": positive_control_ran,
            "positive_control_health": control_health,
            "regression_guard_provenance": guard_provenance,
            "run_scope": scope,
            "n_errored_cells": len(errored),
            "errored_cell_rate": round(error_rate, 4),
            "ab_interpretable": bool(positive_control_usable and error_rate <= 0.05),
            "honest_verdict": verdict_for(
                scope,
                cap,
                positive_control_ran=positive_control_ran,
                error_rate=error_rate,
                control_health=control_health,
                cps_gate=cps_gate_for_verdict,
            ),
            "headline": build_headline(
                cap,
                control_health=control_health,
                cps_gate=cps_gate_for_verdict,
                sampler_run=bool(set(arms) & set(CLICK_PIXEL_ARMS)),
            ),
            "field_provenance": {k: {"principle": v} for k, v in FIELD_PRINCIPLES.items()},
            "derived_sections_recomputed_from_measured_rows": True,
        }
    )
    art.update(gates)
    art["reproducibility_checksum"] = _reproducibility_checksum(art)
    return art


def _arg(argv: Sequence[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv else default


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--recompute" in argv:
        # Re-derive the analysis from an existing artifact's measured rows. Never measures.
        path = Path(_arg(argv, "--recompute", str(ARTIFACT)))
        art = recompute_derived(json.loads(path.read_text()))
        path.write_text(json.dumps(art, indent=2, default=str))
        print(
            json.dumps(
                {
                    "recomputed": str(path),
                    "honest_verdict": art["honest_verdict"],
                    "new_wins_vs_baseline": art.get("new_wins_vs_baseline"),
                    "positive_control_new_wins": art.get("positive_control_new_wins"),
                    "acceptance_gates_all_passed": art.get("acceptance_gates_all_passed"),
                },
                indent=2,
            )
        )
        return 0
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
    art = run(
        games=games,
        arms=arms,
        conditions=conds,
        budget=budget,
        n_seeds=n_seeds,
        artifact_path=Path(out) if out else None,
        replay_limit=int(_arg(argv, "--replay-limit", "6")),
    )
    print(
        json.dumps(
            {
                "honest_verdict": art["honest_verdict"],
                "headline": art.get("headline"),
                "new_wins_vs_baseline": art.get("new_wins_vs_baseline"),
                "positive_control_new_wins": art.get("positive_control_new_wins"),
                "games_won_only_by_positive_control": (art.get("capability_summary") or {}).get(
                    "games_won_only_by_positive_control"
                ),
                "acceptance_gate_capability_passed": (
                    art.get("acceptance_gate_capability") or {}
                ).get("passed"),
                "acceptance_gate_efficiency_passed": (
                    art.get("acceptance_gate_efficiency") or {}
                ).get("passed"),
                "acceptance_gate_click_pixel_sampling_passed": art.get(
                    "acceptance_gate_click_pixel_sampling_passed"
                ),
                "click_pixel_sampling_gate": art.get("acceptance_gate_click_pixel_sampling"),
                "acceptance_gate_hud_mask_passed": art.get("acceptance_gate_hud_mask_passed"),
                "hud_mask_gate": art.get("acceptance_gate_hud_mask"),
                "run_scope": art.get("run_scope"),
                "ab_interpretable": art.get("ab_interpretable"),
                "positive_control_ran": art.get("positive_control_ran"),
                "duration_s": art["duration_s"],
                "n_errored_cells": art.get("n_errored_cells"),
                "aggregates": {
                    k: {
                        "n_wins": v["n_games_won_any_seed"],
                        "wins": v["games_won_any_seed"],
                        "mean_actions_to_first_win": v["mean_actions_to_first_win"]["mean"],
                        "mean_states_expanded": v["mean_states_expanded"]["mean"],
                    }
                    for k, v in art.get("aggregates", {}).items()
                },
                "vs_baseline": art.get("vs_baseline"),
                "reproduction_gate": {
                    k: art["reproduction_gate"][k]
                    for k in ("n_checked", "n_reproduced", "all_reproduced")
                },
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
