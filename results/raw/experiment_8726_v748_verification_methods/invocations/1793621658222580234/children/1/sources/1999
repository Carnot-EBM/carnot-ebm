"""Experiment 5756: properly-powered live A/B for ``InertClickSigPruner`` across
the SAME 11-game roster + real capability metrics (levels / states_expanded),
driving ``E3AgentPolicy``/``StepwiseExplorer`` directly against the offline
arcade -- per CLAUDE.md's Phase Prototype + Empirical Validation + Adversarial
Check Discipline, the Failed-Experiment Rerun Discipline, and the ARC Live-Path
Reachability Discipline.

WHAT THIS CLOSES. ``InertClickSigPruner`` (arc_inert_click_pruner.py) was built,
unit-tested, and LIVE-WIRED into ``E3AgentPolicy``/``StepwiseExplorer`` on
2026-07-13 (a ``rank_candidates`` drop-filter in ``StepwiseExplorer._candidates``
+ a real ``observe()`` from ``_ingest``'s per-transition OBSERVE hook), but gated
OFF by default (``SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False``). Its own
docstring states flipping it on for the SCORED agent "needs its own matched-budget
offline A/B first, per the ``solve_rate_dropped`` guardrail." This is that A/B on
the full roster with a real capability metric.

WHY THIS IS NOT A DOOMED RERUN (Failed-Experiment Rerun Discipline). Two prior
artifacts already touched this mechanism, BOTH single-game (m0r0) nulls at a low
budget:

  * exp5595 (offline prototype): ``inert_click_sig_pruner_prototype_ran_but_no_
    signature_cleared_evidence_floor`` -- m0r0, 32 click transitions, 12
    signatures tracked, 0 confidently inert at that budget.
  * exp5602 (matched-budget A/B): ``inert_click_pruner_ab_no_op_offline_and_live_
    at_this_budget_on_m0r0`` -- m0r0, target_level 2, states_expanded_reduction=0.
    CRITICALLY its OfflineSolver arm's pruner observed=0 (the directed
    verifier-guided OfflineSolver search never even EXERCISED the pruner, so that
    arm is structurally uninformative), and its live E3 supplementary check
    observed 32 clicks / 9 signatures but pruned 0 (no signature crossed the
    evidence floor at m0r0's ~37-transition budget).

The shared root cause of both nulls: a SINGLE game (m0r0) at a LOW budget (~37
transitions) never accumulates ``min_observations=4`` inert observations of the
same ``(color, size, is_rect, twin_count)`` signature with high enough specificity
to fire a prune. A single-game null at that budget does not generalize -- it is
"m0r0 at budget~37 has no confidently-inert repeated click signature", not "the
pruner is inert roster-wide at every budget".

WHAT IS DIFFERENT HERE (must be true, not asserted): (a) the SAME 11-game roster
exp5729/exp5732/exp5740 used, not one game; (b) budget=200 (vs exp5602's ~37
transitions on m0r0) -- directly testing whether a MUCH larger budget crosses the
``min_observations`` evidence floor on more games/signatures; (c) REAL capability
metrics -- ``levels_gained_total``, ``states_expanded_total``
(``len(policy.explorer.graph)``), per-game breakdown -- via the exp5740/exp5729
harness pattern (drive E3AgentPolicy/StepwiseExplorer directly, matched action
budget, no-LLM), which is the ACTUAL live-scored mechanism the pruner is wired into
(``rank_candidates`` + ``observe``), unlike exp5602's OfflineSolver arm where the
pruner was never exercised; (d) real pruner diagnostics per game (signatures
reaching the evidence floor, candidates actually pruned) so a null is
distinguished from a never-fired guard; (e) an EMPIRICAL missed-win safety check
(any game where a treatment arm banks FEWER levels than baseline = the pruner
suppressed a winnable click), not an assumption that the trust+specificity gate
worked.

ARMS (3, same 11-game roster + same 200-action budget + CARNOT_ARC_DISABLE_
INDUCTION=1 no-LLM harness as exp5740, single-threaded, per-game fixed seed so
each arm's game-X starts from an identical RNG state -> any delta isolates the
pruner, not RNG drift):

  1. baseline            -- the CURRENT shipped default: E3AgentPolicy(game) with
                            inert_click_pruner OFF (SUBMITTED_INERT_CLICK_PRUNER_
                            ENABLED=False -> coerce_inert_click_pruner(False)=None).
  2. treatment_default   -- THE PRIMARY comparison + the faithful live flip: exactly
                            what flipping SUBMITTED_INERT_CLICK_PRUNER_ENABLED
                            False->True does. inert_click_pruner=True ->
                            coerce_inert_click_pruner builds InertClickSigPruner
                            (grid_of) at the shipped default params
                            (min_observations=4, min_specificity=0.9).
  3. treatment_aggressive -- a cheap sensitivity arm: min_observations=2 (Reki's
                            original K=2 evidence floor, which the shipped pruner
                            deliberately raised to 4), min_specificity=0.9. Tests
                            whether the shipped default's higher evidence floor is
                            WHY the pruner never fires (exp5595/exp5602's null) --
                            i.e. does a lower floor make pruning ENGAGE at
                            budget=200, and if so does that help, hurt, or no-op the
                            banked levels? This is the sensitivity read exp5595's
                            "0 confidently inert" null asks for.

CACHE-HYGIENE PREREQUISITE (done before this A/B; NOT the exp5740 per-candidate
flood). Unlike ObjectHistorySaliencePrior.score() -- which exp5740 fixed because it
recomputed the frame decomposition once per CANDIDATE (O(candidates x grid_cells)) --
InertClickSigPruner's live path already decomposes ONCE per rank_candidates call
(then loops candidates via the cheap blob_at_click lookup) and ONCE per observe()
transition, NOT once per candidate. So it does NOT have exp5740's severe flood. What
it DID do was call ``connected_color_blobs`` RAW, bypassing the shared
``_cached_blobs_and_counts`` per-frame LRU cache (the same key ColorBlobSaliencePrior
uses with min_pixels=1/max_component_fraction=1.0), so repeated same-frame
decomposition (observe after rank_candidates on the same frame, or a co-active blob
prior in the real live stack) recomputed the flood from scratch. Fixed to route
through ``_cached_blobs_and_counts`` (behavior-preserving: identical blob list,
verified below). Because run_game's budget is ACTION-count (``for step_index in
range(budget)``), not wall-clock, and the no-induction search path has no wall-clock
cutoff, this per-frame timing change cannot confound the capability read (levels/
states) EITHER WAY -- it is hygiene, not a capability-confound fix. See
``blob_cache_perf_fix`` in the artifact for the before/after timing + the
behavior-preservation assertion.

Spec refs: REQ-ARC-FCP-5756, SCENARIO-ARC-FCP-5756-ELEVEN-GAME-CAPABILITY-AB,
SCENARIO-ARC-FCP-5756-INERT-CLICK-CACHE-HYGIENE.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (PYTHON_ROOT, REPO_ROOT, SCRIPTS_ROOT):
    if str(_p) not in sys.path:  # pragma: no cover - direct script guard
        sys.path.insert(0, str(_p))

# No-LLM search substrate: guaranteed before any policy import/construction.
os.environ.setdefault("CARNOT_ARC_DISABLE_INDUCTION", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_color_blob_salience import (  # noqa: E402
    _cached_blobs_and_counts,
    blob_at_click,
    connected_color_blobs,
)
from carnot.agentic.arc_inert_click_pruner import (  # noqa: E402
    InertClickSigPruner,
    click_signature,
    coerce_inert_click_pruner,
)

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5756_inert_click_pruner_11game_ab"
RESULT_RELATIVE_PATH = "results/experiment_5756_inert_click_pruner_11game_ab.json"
SCHEMA = "carnot.exp5756.inert_click_pruner_11game_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5756
DEFAULT_BUDGET = 200
BASELINE_ARM = "baseline"
DEFAULT_MIN_OBSERVATIONS = 4
DEFAULT_MIN_SPECIFICITY = 0.9
AGGRESSIVE_MIN_OBSERVATIONS = 2
STATES_REGRESSION_REL = (
    0.20  # exp5729/exp5740 discipline: >20% states growth is a material regression
)

# (name, kind). kind in {"baseline","treatment_default","treatment_aggressive"}.
ARMS: tuple[tuple[str, str], ...] = (
    ("baseline", "baseline"),
    ("treatment_default", "treatment_default"),
    ("treatment_aggressive", "treatment_aggressive"),
)
TREATMENT_ARMS = ("treatment_default", "treatment_aggressive")

# Same 11-game roster as exp5729/exp5732/exp5740 for an apples-to-apples comparison.
DEFAULT_ROSTER = (
    "cd82",
    "cn04",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "sk48",
    "sp80",
    "su15",
    "tu93",
    "wa30",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "solve_provenance",
    "roster",
    "budget",
    "arms_tested",
    "baseline_arm",
    "per_arm_results",
    "per_arm_game_rows",
    "levels_gained_total_by_arm",
    "states_expanded_total_by_arm",
    "n_signatures_reaching_evidence_floor_per_game",
    "n_candidates_pruned_per_game",
    "trajectories_diverge_per_game",
    "blob_cache_perf_fix",
    "levels_gained_headroom_present",
    "any_config_beats_baseline_levels",
    "safety_regression_check",
    "prior_work_extended",
    "recommendation",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a level win, a search-behavior-only change (pruning fires and "
        "diverges but banks no extra level), a never-fired null (no signature crosses the evidence "
        "floor), and a missed-win regression are distinct real outcomes -- enabling the pruner only "
        "earns a live-default flip if it holds or raises banked levels on the same roster+budget "
        "WITHOUT suppressing a winnable click and WITHOUT a states blow-up"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_"
        "INDUCTION=1 guarantees no GGUF/LLM load; the pruner affects only the candidate-filter + "
        "frontier path, isolated from tier-3 induction"
    },
    "verifier_is_oracle": {
        "principle": "False -- a learned inert-click predictor fit from the search's OWN observed "
        "transitions (per-signature obs/inert/leveled tally), never the executable oracle that defines "
        "correctness"
    },
    "solve_provenance": {
        "principle": "development_proxy -- an offline-arcade live-path move-pruner A/B on the dev "
        "twin; does NOT fit the game-solve taxonomy (a component measurement, not a self-discovery "
        "solve), NO new level is banked (any level reached is a pre-existing registry solve reached "
        "incidentally), offline_reproduced is deliberately NOT claimed"
    },
    "arms_tested": {
        "principle": "baseline (shipped default, pruner OFF) + treatment_default (the faithful live "
        "flip at min_observations=4) + treatment_aggressive (min_observations=2, Reki's original K=2 "
        "floor) -- the aggressive arm tests whether the shipped default's higher evidence floor is WHY "
        "the pruner never fired in exp5595/exp5602"
    },
    "per_arm_results": {
        "principle": "per-arm levels/states/efficiency totals + per-game level deltas vs baseline so a "
        "third party can re-derive every headline number"
    },
    "per_arm_game_rows": {
        "principle": "full per-game rows (levels, states_expanded, actions, pruner stats, trajectory "
        "sha) so the capability comparison, the pruner-engagement diagnostics, and the divergence "
        "secondary are all independently auditable"
    },
    "levels_gained_total_by_arm": {
        "principle": "the raw capability answer: total banked levels per arm across the roster; the "
        "primary comparison is treatment_default vs baseline"
    },
    "states_expanded_total_by_arm": {
        "principle": "total search cost per arm; a pruner that drops candidates should REDUCE or hold "
        "states, mirroring HazardMovePruner's tu93 A/B (2947 -> 2859) -- a states INCREASE from a "
        "pruner would itself be a red flag"
    },
    "n_signatures_reaching_evidence_floor_per_game": {
        "principle": "the load-bearing engagement diagnostic: how many (color,size,is_rect,twin_count) "
        "signatures accumulated >= min_observations observations per game at budget=200. exp5595/"
        "exp5602 found 0 at budget~37 on m0r0; a null here is only meaningful (FALSE_NEGATIVE_RISK) if "
        "SOME signature crossed the floor somewhere -- else the pruner simply never had the evidence to "
        "fire and the A/B cannot distinguish 'inert-click pruning does not help' from 'the floor was "
        "never reached'"
    },
    "n_candidates_pruned_per_game": {
        "principle": "how many click candidates the pruner actually dropped per game (pruner.pruned) -- "
        "0 everywhere means treatment is byte-identical to baseline by construction (a never-fired "
        "guard), a positive count means the capability delta is a REAL pruning effect"
    },
    "trajectories_diverge_per_game": {
        "principle": "SECONDARY interpretation aid: if pruned=0 the treatment trajectory is byte-"
        "identical to baseline (confirming the null is a never-fired floor, not a fired-but-neutral "
        "prune); if it diverges but levels are equal, pruning fired and changed search without changing "
        "banked levels"
    },
    "blob_cache_perf_fix": {
        "principle": "documents the cache-hygiene change (route InertClickSigPruner's raw "
        "connected_color_blobs through the shared _cached_blobs_and_counts cache), the before/after "
        "per-frame timing, the behavior-preservation assertion (identical signatures), and WHY this is "
        "NOT exp5740's per-candidate flood (the pruner already decomposes once per frame, not per "
        "candidate) and cannot confound an action-budget A/B either way"
    },
    "levels_gained_headroom_present": {
        "principle": "FALSE_NEGATIVE_RISK discipline -- a no-delta result is only interpretable if some "
        "arm banks a nonzero level somewhere, else the null may just mean the roster had no level "
        "headroom for any candidate-filter at this budget"
    },
    "any_config_beats_baseline_levels": {
        "principle": "the load-bearing capability boolean: True iff some arm banks strictly more levels "
        "than baseline on the same roster+budget (a pruner is not EXPECTED to raise levels -- its job "
        "is efficiency -- so the key safety property is that it does not LOSE levels)"
    },
    "safety_regression_check": {
        "principle": "exp5729/exp5740 discipline PLUS the pruner-specific missed-win check -- a config "
        "that suppresses a winnable click (banks FEWER levels than baseline on any game) is the "
        "specific risk this component's trust+specificity gate exists to prevent, verified empirically "
        "via per-game level deltas, not assumed; a states_expanded inflation is also flagged"
    },
    "prior_work_extended": {
        "principle": "Failed-Experiment Rerun Discipline -- names exp5595 and exp5602 (the prior m0r0 "
        "nulls this re-tests) by id+verdict+root cause AND states precisely what is different (11-game "
        "roster + budget=200 vs their single-game budget~37 + real capability metrics on the actual "
        "live-wired E3 path, not the OfflineSolver arm where the pruner was never exercised), plus "
        "exp5740 (the harness precedent), with a retire condition"
    },
    "recommendation": {
        "principle": "reports whether to flip the live default (SUBMITTED_INERT_CLICK_PRUNER_ENABLED); "
        "the agent never self-authorizes flipping the live-stack default -- operator-only"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
    "duration_s": {
        "principle": "real wall-clock of the search A/B; the no-LLM substrate floor is 0.01s and this "
        "runs sequentially over 3 arms x 11 games, so a plausible multi-hundred-second total is "
        "expected"
    },
    "preconditions_checked": {
        "principle": "records the resources verified (offline arcade builds an env, the live policy + "
        "pruner + cache-fix import resolve) before the sweep, per Pre-Launch Preconditions Discipline"
    },
}

PRIOR_WORK_EXTENDED = {
    "experiments_extended": [
        {
            "experiment_id": "exp5595 (experiment_5595_inert_click_sig_pruner_offline_sim_prototype)",
            "prior_verdict": (
                "complete: inert_click_sig_pruner_prototype_ran_but_no_signature_cleared_evidence_floor"
            ),
            "diagnosed_root_cause": (
                "The offline prototype ran the pruner over a single game (m0r0, explore_budget=6, "
                "total_budget=40): 32 click transitions, 12 signatures tracked, 0 confidently inert. At "
                "that budget no (color,size,is_rect,twin_count) signature accumulated the "
                "min_observations=4 inert observations at min_specificity>=0.9 needed to fire a prune. "
                "An honest null from insufficient evidence at a low single-game budget."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "exp5602 (experiment_5602_inert_click_pruner_matched_budget_ab)",
            "prior_verdict": (
                "complete: inert_click_pruner_ab_no_op_offline_and_live_at_this_budget_on_m0r0"
            ),
            "diagnosed_root_cause": (
                "A matched-budget A/B on m0r0 target_level=2 with states_expanded_reduction=0. Its "
                "OfflineSolver arm's pruner observed=0 (the directed verifier-guided OfflineSolver "
                "search never EXERCISED the pruner -- structurally uninformative for a live-path "
                "capability read). Its live E3 supplementary check observed 32 clicks / 9 signatures "
                "but pruned 0 (no signature crossed the evidence floor at m0r0's ~37-transition "
                "budget). Same root cause as exp5595: single game (m0r0) at a low budget never "
                "accumulates enough repeated inert observations of one signature to fire. NOT re-"
                "litigated the same way -- exp5602 already settled that the OfflineSolver arm never "
                "engages the pruner, so this re-test targets the LIVE E3 path (rank_candidates + "
                "observe) exclusively, at a much larger budget, across the full roster."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "exp5740 (experiment_5740_object_history_salience_11game_ab)",
            "prior_verdict": (
                "complete: object_history_bonus_inert_over_colorblob_prior... (an 11-game live A/B of a "
                "sibling gated-off live-wired action_prior)"
            ),
            "role": (
                "THE HARNESS PRECEDENT (most recent, most refined). Solved the 'matched-budget A/B a "
                "live-path component with real levels/states metrics on the 11-game roster' problem: "
                "drive E3AgentPolicy/StepwiseExplorer directly via arc_leaderboard_eval.run_game, per-"
                "game fixed seed, states_expanded = len(policy.explorer.graph), the safety-regression "
                "discipline, the embedded cache-hygiene timing check. This script reuses that skeleton "
                "for the pruner (a rank_candidates drop-filter) instead of an action_prior (a score)."
            ),
        },
    ],
    "what_is_different_here": (
        "exp5595 and exp5602 both tested ONE game (m0r0) at a LOW budget (~37 transitions), where the "
        "min_observations=4 evidence floor was never crossed (0 signatures confidently inert / 0 "
        "pruned). This tests the SAME 11-game roster exp5729/exp5732/exp5740 used, at budget=200 "
        "(~5x m0r0's transition count), with REAL capability metrics (levels_gained_total, states_"
        "expanded_total, per-game breakdown) on the ACTUAL live-wired E3 path (rank_candidates + "
        "observe) -- NOT exp5602's OfflineSolver arm where the pruner was never exercised. It adds a "
        "min_observations=2 sensitivity arm to test whether the shipped default's higher floor is WHY "
        "the pruner never fired, and an EMPIRICAL missed-win safety check (per-game level regression). "
        "This is a roster-breadth + budget-increase + real-metric re-measurement, not a re-run of the "
        "same single-game low-budget setup."
    ),
    "retire_if_same_verdict": (
        "If at budget=200 across the full 11-game roster STILL no signature crosses the evidence floor "
        "in ANY arm (0 pruned everywhere, treatment byte-identical to baseline) -- reproducing exp5595/"
        "exp5602's never-fired null with full roster breadth and a real budget -- then the inert-click-"
        "signature mechanism does not engage at reachable live-agent budgets and adds no live-path "
        "capability: recommend the operator retire the live-default flip consideration for "
        "SUBMITTED_INERT_CLICK_PRUNER_ENABLED (operator-only; it stays False) and close the gap per "
        "Missing-Verifier Gap Logging. Do NOT re-propose another InertClickSigPruner budget/roster A/B "
        "without a NEW mechanism change (e.g. a different signature keying that accumulates evidence "
        "faster)."
    ),
}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Cache-hygiene timing sanity check (embedded so the artifact self-documents it).
# --------------------------------------------------------------------------- #
def _blob_cache_timing_check(game: str = "lp85", k: int = 400) -> JsonDict:
    """Time raw-per-decomposition vs cache-routed on a real frame; assert the fix is
    behavior-preserving (identical signatures). This is NOT exp5740's per-candidate
    flood -- the pruner already decomposes once per frame (rank_candidates) / once per
    transition (observe), not once per candidate. What the fix removes is the RAW
    (uncached) recompute of the SAME frame across those calls (and across a co-active
    ColorBlobSaliencePrior in the real live stack, which shares the cache key)."""

    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    pruner = InertClickSigPruner(grid_of)
    grid = pruner._g2d(frame)
    assert grid is not None
    h, w = grid.shape
    rng = np.random.default_rng(RANDOM_SEED)
    clicks = [(int(rng.integers(0, w)), int(rng.integers(0, h))) for _ in range(k)]

    g16 = np.asarray(grid).astype(np.int16, copy=False)

    def _raw_sig(x: int, y: int) -> Any:
        blobs = connected_color_blobs(g16, min_pixels=1, max_component_fraction=1.0)
        blob = blob_at_click(blobs, x, y)
        return None if blob is None else click_signature(blob, blobs)

    def _cached_sig(x: int, y: int) -> Any:
        # exactly the post-fix path in _signature_for_click, via the shared cache.
        blobs, _c = _cached_blobs_and_counts(g16, min_pixels=1, max_component_fraction=1.0)
        blob = blob_at_click(blobs, x, y)
        return None if blob is None else click_signature(blob, blobs)

    # behavior preservation: identical signatures for every click.
    identical = all(_raw_sig(x, y) == _cached_sig(x, y) for (x, y) in clicks)

    # timing: simulate the same frame being decomposed repeatedly (observe after
    # rank_candidates, plus a co-active blob prior) -- best of 3 reps.
    def _time(fn: Any) -> float:
        _blob_cache_clear()
        for x, y in clicks[:5]:
            fn(x, y)
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            for x, y in clicks:
                fn(x, y)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return float(best or 0.0)

    t_raw = _time(_raw_sig)
    t_cached = _time(_cached_sig)

    return {
        "note": (
            "InertClickSigPruner's live path decomposes ONCE per rank_candidates call (then loops "
            "candidates via cheap blob_at_click) and ONCE per observe() transition -- NOT once per "
            "candidate. So it does NOT have exp5740's severe per-candidate flood (O(candidates x "
            "grid_cells)). This is a cache-sharing hygiene fix, not a capability-confound fix."
        ),
        "bug": (
            "_signature_for_click and rank_candidates called connected_color_blobs(grid, min_pixels=1, "
            "max_component_fraction=1.0) RAW, bypassing the shared per-frame LRU cache "
            "(_cached_blobs_and_counts, arc_color_blob_salience.py) that ColorBlobSaliencePrior.score "
            "uses with the identical key. Repeated same-frame decomposition (observe after "
            "rank_candidates, or a co-active blob prior) recomputed the O(grid-cells) flood from "
            "scratch instead of reusing the warm cache entry."
        ),
        "fix": (
            "Added InertClickSigPruner._frame_blobs(grid) routing through _cached_blobs_and_counts "
            "(int16-normalized to match ColorBlobSaliencePrior's cache key); _signature_for_click and "
            "rank_candidates now call it. Behavior-preserving: the cache stores exactly "
            "connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)'s output."
        ),
        "timing_probe_game": game,
        "grid_shape": [int(h), int(w)],
        "n_decompositions": int(k),
        "raw_uncached_us_per_call": round(t_raw / k * 1e6, 2),
        "cache_routed_us_per_call": round(t_cached / k * 1e6, 2),
        "raw_vs_cached_slowdown_ratio": round(t_raw / t_cached, 3) if t_cached else None,
        "fixed_signatures_identical_to_raw": bool(identical),
        "fix_behavior_preserving": bool(identical),
        "does_not_confound_capability": (
            "run_game's budget is ACTION-count (for step_index in range(budget)), not wall-clock, and "
            "the no-induction search path has no wall-clock cutoff, so this per-frame timing change "
            "alters total runtime but never levels/states -- the capability read is invariant to it "
            "either way."
        ),
    }


def _blob_cache_clear() -> None:
    from carnot.agentic import arc_color_blob_salience as cbs

    cbs._blob_cache.clear()


# --------------------------------------------------------------------------- #
# Preconditions.
# --------------------------------------------------------------------------- #
def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401

        checks["e3_policy_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
    checks["inert_click_pruner_import"] = bool(InertClickSigPruner and coerce_inert_click_pruner)
    # The cache-fix import (the hygiene fix's precondition) must resolve.
    checks["blob_cache_import"] = bool(_cached_blobs_and_counts)
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _build_policy(game: str, kind: str) -> Any:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    if kind == "baseline":
        return E3AgentPolicy(game)
    if kind == "treatment_default":
        # Exactly what flipping SUBMITTED_INERT_CLICK_PRUNER_ENABLED True does.
        return E3AgentPolicy(game, inert_click_pruner=True)
    if kind == "treatment_aggressive":
        pruner = InertClickSigPruner(
            grid_of,
            min_observations=AGGRESSIVE_MIN_OBSERVATIONS,
            min_specificity=DEFAULT_MIN_SPECIFICITY,
        )
        return E3AgentPolicy(game, inert_click_pruner=pruner)
    raise ValueError(f"unknown arm kind: {kind}")  # pragma: no cover


def _trajectory(policy: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in getattr(policy, "transitions", []) or []:
        out.append({"action": int(getattr(t, "action", 0)), "data": getattr(t, "data", None)})
    return out


def _pruner_diagnostics(policy: Any) -> JsonDict:
    """Read the live pruner off the explorer and summarize its engagement."""

    pruner = getattr(getattr(policy, "explorer", None), "inert_click_pruner", None)
    if pruner is None:
        return {
            "pruner_present": False,
            "observed": 0,
            "pruned": 0,
            "signatures_tracked": 0,
            "signatures_reaching_evidence_floor": 0,
            "signatures_ever_leveled": 0,
            "pruned_signatures": 0,
            "min_observations": None,
            "min_specificity": None,
        }
    tally = getattr(pruner, "_tally", {}) or {}
    min_obs = int(getattr(pruner, "min_observations", 0))
    floor = sum(1 for t in tally.values() if int(t.get("obs", 0)) >= min_obs)
    ever_leveled = sum(1 for t in tally.values() if int(t.get("leveled", 0)) > 0)
    stats = pruner.stats()
    return {
        "pruner_present": True,
        "observed": int(stats.get("observed", 0)),
        "pruned": int(stats.get("pruned", 0)),
        "signatures_tracked": int(stats.get("signatures_tracked", 0)),
        "signatures_reaching_evidence_floor": int(floor),
        "signatures_ever_leveled": int(ever_leveled),
        "pruned_signatures": int(stats.get("pruned_signatures", 0)),
        "min_observations": min_obs,
        "min_specificity": float(getattr(pruner, "min_specificity", 0.0)),
    }


def _play_one_game(
    game: str, *, arm_name: str, kind: str, budget: int, game_index: int
) -> JsonDict:
    import arc_leaderboard_eval as lb

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    # Per-game fixed seed so baseline-game-X and arm-game-X start from an identical
    # RNG state -- any delta isolates the pruner, not RNG drift (exp5740 pattern).
    _seed_everything(RANDOM_SEED + game_index)

    t0 = time.time()
    policy = _build_policy(game, kind)
    row = lb.run_game(game, policy, budget=budget)
    dt = round(time.time() - t0, 3)
    traj = _trajectory(policy)
    traj_sha = hashlib.sha256(
        json.dumps(traj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    pruner_diag = _pruner_diagnostics(policy)

    return {
        "game": game,
        "arm": arm_name,
        "levels": int(row.get("levels", 0)),
        "reached": int(row.get("reached", 0)),
        "actions": int(row.get("actions", 0)),
        "states_expanded": int(len(policy.explorer.graph)),
        "efficiency": float(row.get("efficiency", 0.0) or 0.0),
        "actions_to_first_levelup": row.get("actions_to_first_levelup"),
        "gap_signature": (row.get("gap") or {}).get("signature") if row.get("gap") else None,
        "trajectory_len": len(traj),
        "trajectory_sha": traj_sha,  # auditable fingerprint: equal sha == identical action sequence
        "pruner": pruner_diag,
        "duration_s": dt,
        "_trajectory": traj,  # dropped from the artifact after divergence is computed
    }


def run_sweep(
    roster: tuple[str, ...],
    arms: tuple[tuple[str, str], ...],
    *,
    budget: int,
) -> dict[str, dict[str, JsonDict]]:
    out: dict[str, dict[str, JsonDict]] = {}
    for arm_name, kind in arms:
        per_game: dict[str, JsonDict] = {}
        for game_index, game in enumerate(roster):
            per_game[game] = _play_one_game(
                game, arm_name=arm_name, kind=kind, budget=budget, game_index=game_index
            )
        out[arm_name] = per_game
    return out


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    arms: tuple[tuple[str, str], ...] = ARMS,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    started_at = time.time()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    arm_names = [a[0] for a in arms]

    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "solve_provenance": "development_proxy",
            "roster": list(roster),
            "budget": int(budget),
            "arms_tested": arm_names,
            "baseline_arm": BASELINE_ARM,
            "per_arm_results": [],
            "per_arm_game_rows": {},
            "levels_gained_total_by_arm": {},
            "states_expanded_total_by_arm": {},
            "n_signatures_reaching_evidence_floor_per_game": {},
            "n_candidates_pruned_per_game": {},
            "trajectories_diverge_per_game": {},
            "blob_cache_perf_fix": {},
            "levels_gained_headroom_present": False,
            "any_config_beats_baseline_levels": False,
            "safety_regression_check": {},
            "prior_work_extended": PRIOR_WORK_EXTENDED,
            "recommendation": f"blocked precondition {miss}; sweep not run.",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    # Cache-hygiene timing sanity check first (cheap, self-documents the fix).
    blob_cache_perf_fix = _blob_cache_timing_check()

    sweep = run_sweep(roster, arms, budget=budget)

    baseline_rows = sweep[BASELINE_ARM]
    baseline_levels_total = sum(r["levels"] for r in baseline_rows.values())
    baseline_states_total = sum(r["states_expanded"] for r in baseline_rows.values())

    # Trajectory divergence vs baseline (secondary interpretation aid) computed BEFORE
    # dropping the heavy per-run trajectories from the rows.
    def _pairwise_diverge(rows_a: dict[str, JsonDict], rows_b: dict[str, JsonDict]) -> JsonDict:
        per_game = {g: bool(rows_a[g]["_trajectory"] != rows_b[g]["_trajectory"]) for g in roster}
        return {
            "per_game": per_game,
            "n_games_diverged": int(sum(1 for v in per_game.values() if v)),
            "games_diverged": sorted(g for g, v in per_game.items() if v),
        }

    treatment_vs_baseline_divergence = {
        arm: _pairwise_diverge(sweep[arm], baseline_rows) for arm in TREATMENT_ARMS
    }
    trajectories_diverge_per_game: JsonDict = {
        "treatment_vs_baseline": treatment_vs_baseline_divergence,
        "note": (
            "If a treatment arm's pruner fired (n_candidates_pruned > 0), its trajectory should "
            "diverge from baseline on those games; if pruned=0, the trajectory is byte-identical "
            "(equal trajectory_sha) -- confirming the null is a never-fired evidence floor, not a "
            "fired-but-neutral prune."
        ),
    }

    # Per-game pruner engagement diagnostics (the load-bearing 'did the guard fire' read).
    n_floor_per_game: JsonDict = {}
    n_pruned_per_game: JsonDict = {}
    for arm_name in arm_names:
        n_floor_per_game[arm_name] = {
            g: int(sweep[arm_name][g]["pruner"]["signatures_reaching_evidence_floor"])
            for g in roster
        }
        n_pruned_per_game[arm_name] = {
            g: int(sweep[arm_name][g]["pruner"]["pruned"]) for g in roster
        }

    # Drop the heavy trajectories from the persisted rows.
    per_arm_game_rows: JsonDict = {}
    for arm_name in arm_names:
        per_arm_game_rows[arm_name] = {
            g: {k: v for k, v in sweep[arm_name][g].items() if k != "_trajectory"} for g in roster
        }

    per_arm_results: list[JsonDict] = []
    levels_by_arm: JsonDict = {}
    states_by_arm: JsonDict = {}
    any_headroom = False
    for arm_name, _kind in arms:
        rows = sweep[arm_name]
        levels_total = sum(r["levels"] for r in rows.values())
        states_total = sum(r["states_expanded"] for r in rows.values())
        eff_sum = round(sum(r["efficiency"] for r in rows.values()), 4)
        wall_total = round(sum(r["duration_s"] for r in rows.values()), 3)
        pruned_total = sum(r["pruner"]["pruned"] for r in rows.values())
        floor_total = sum(r["pruner"]["signatures_reaching_evidence_floor"] for r in rows.values())
        per_game_levels = {g: rows[g]["levels"] for g in roster}
        per_game_levels_delta = {g: rows[g]["levels"] - baseline_rows[g]["levels"] for g in roster}
        per_game_states = {g: rows[g]["states_expanded"] for g in roster}
        if any(r["levels"] > 0 for r in rows.values()):
            any_headroom = True
        levels_by_arm[arm_name] = int(levels_total)
        states_by_arm[arm_name] = int(states_total)
        per_arm_results.append(
            {
                "arm": arm_name,
                "is_baseline": arm_name == BASELINE_ARM,
                "levels_gained_total": int(levels_total),
                "levels_delta_vs_baseline": int(levels_total - baseline_levels_total),
                "states_expanded_total": int(states_total),
                "states_delta_vs_baseline": int(states_total - baseline_states_total),
                "efficiency_sum": float(eff_sum),
                "wall_clock_total_s": wall_total,
                "candidates_pruned_total": int(pruned_total),
                "signatures_reaching_evidence_floor_total": int(floor_total),
                "per_game_levels": per_game_levels,
                "per_game_levels_delta_vs_baseline": per_game_levels_delta,
                "per_game_states_expanded": per_game_states,
            }
        )

    # Safety-regression check: states inflation + the pruner-specific MISSED-WIN check.
    per_config_safety: dict[str, JsonDict] = {}
    for res in per_arm_results:
        arm = res["arm"]
        delta = int(res["states_delta_vs_baseline"])
        rel = (delta / baseline_states_total) if baseline_states_total else 0.0
        states_regression = bool(arm != BASELINE_ARM and rel > STATES_REGRESSION_REL)
        # MISSED-WIN: any game where this arm banks strictly FEWER levels than baseline.
        missed_win_games = sorted(
            g for g, d in res["per_game_levels_delta_vs_baseline"].items() if d < 0
        )
        per_config_safety[arm] = {
            "states_expanded_total": int(res["states_expanded_total"]),
            "states_delta_vs_baseline": delta,
            "states_rel_change_vs_baseline": round(float(rel), 4),
            "states_expanded_regression": states_regression,
            "missed_win_games": missed_win_games,
            "suppressed_a_winnable_click": bool(missed_win_games),
        }
    any_missed_win = bool(
        any(
            v["suppressed_a_winnable_click"]
            for a, v in per_config_safety.items()
            if a != BASELINE_ARM
        )
    )
    safety_regression_check = {
        "baseline_states_expanded_total": int(baseline_states_total),
        "states_regression_relative_threshold": STATES_REGRESSION_REL,
        "per_config": per_config_safety,
        "any_config_states_regression": bool(
            any(v["states_expanded_regression"] for v in per_config_safety.values())
        ),
        "any_config_suppressed_a_winnable_click": any_missed_win,
        "interpretation": (
            "The pruner's job is EFFICIENCY (drop clicks that never change the frame), not raising "
            "levels; the load-bearing safety property is that it never suppresses a WINNABLE click. "
            "suppressed_a_winnable_click is measured empirically: a treatment arm banking FEWER levels "
            "than baseline on any game means the pruner dropped a click that led to progress -- the "
            "exact failure its trust+specificity+sacred-if-ever-leveled gate exists to prevent. A "
            "states_expanded INCREASE from a drop-filter would itself be anomalous (a pruner should "
            "reduce or hold search cost)."
        ),
    }

    # Capability booleans.
    treatment_levels = {a: levels_by_arm[a] for a in TREATMENT_ARMS}
    best_treatment_arm = max(treatment_levels, key=lambda a: treatment_levels[a])
    best_treatment_levels = treatment_levels[best_treatment_arm]
    any_beats = any(
        levels_by_arm[a] > baseline_levels_total for a in arm_names if a != BASELINE_ARM
    )
    total_pruned_by_arm = {
        a: sum(sweep[a][g]["pruner"]["pruned"] for g in roster) for a in arm_names
    }
    total_floor_by_arm = {
        a: sum(sweep[a][g]["pruner"]["signatures_reaching_evidence_floor"] for g in roster)
        for a in arm_names
    }
    any_pruner_fired = any(total_pruned_by_arm[a] > 0 for a in TREATMENT_ARMS)
    any_floor_reached = any(total_floor_by_arm[a] > 0 for a in TREATMENT_ARMS)
    any_treatment_diverges = any(
        treatment_vs_baseline_divergence[a]["n_games_diverged"] > 0 for a in TREATMENT_ARMS
    )
    # Efficiency direction among arms whose pruner actually fired: a pruner is an efficiency
    # tool, so the load-bearing question once it fires + holds levels is whether it REDUCED
    # search cost (states_expanded) like HazardMovePruner's tu93 win (2947 -> 2859), or made
    # the search LESS efficient (states went up because dropping clicks reshaped the frontier).
    fired_arms = [a for a in TREATMENT_ARMS if total_pruned_by_arm[a] > 0]
    best_fired_states = min((states_by_arm[a] for a in fired_arms), default=baseline_states_total)
    pruner_reduced_states = bool(fired_arms) and best_fired_states < baseline_states_total
    best_fired_states_delta = int(best_fired_states - baseline_states_total)

    # Verdict selection.
    if any_missed_win:
        verdict = (
            "complete: inert_click_pruner_SUPPRESSED_A_WINNABLE_CLICK_treatment_banks_fewer_levels_"
            f"than_baseline_{baseline_levels_total}_on_some_game_do_not_flip_live_default"
        )
    elif any_beats:
        verdict = (
            "complete: inert_click_pruner_raises_levels_to_"
            f"{best_treatment_levels}_at_{best_treatment_arm}_vs_baseline_{baseline_levels_total}_"
            "unexpected_for_an_efficiency_filter_review_before_flip"
        )
    elif any_pruner_fired and pruner_reduced_states:
        verdict = (
            "complete: inert_click_pruner_fires_at_budget200_prunes_"
            f"{sum(total_pruned_by_arm[a] for a in TREATMENT_ARMS)}_candidates_holds_levels_at_"
            f"{baseline_levels_total}_no_missed_win_and_reduces_states_expanded_efficiency_win"
        )
    elif any_pruner_fired:
        verdict = (
            "complete: inert_click_pruner_fires_at_budget200_prunes_"
            f"{sum(total_pruned_by_arm[a] for a in TREATMENT_ARMS)}_candidates_holds_levels_at_"
            f"{baseline_levels_total}_no_missed_win_but_states_expanded_did_not_decrease_delta_"
            f"{best_fired_states_delta:+d}_no_efficiency_benefit"
        )
    elif any_floor_reached:
        verdict = (
            "complete: inert_click_pruner_signatures_reach_evidence_floor_at_budget200_but_specificity_"
            f"gate_prunes_zero_treatment_identical_to_baseline_levels_{baseline_levels_total}_"
            "generalizes_exp5595_exp5602_null_with_partial_engagement"
        )
    else:
        verdict = (
            "complete: inert_click_pruner_never_fires_no_signature_crosses_evidence_floor_at_budget200_"
            f"on_11game_roster_treatment_byte_identical_to_baseline_levels_{baseline_levels_total}_"
            "generalizes_exp5595_exp5602_m0r0_null_to_full_roster"
        )

    # Recommendation.
    if any_missed_win:
        offenders = {
            a: v["missed_win_games"]
            for a, v in per_config_safety.items()
            if a != BASELINE_ARM and v["suppressed_a_winnable_click"]
        }
        recommendation = (
            f"DO NOT flip SUBMITTED_INERT_CLICK_PRUNER_ENABLED. The pruner SUPPRESSED A WINNABLE CLICK: "
            f"arm(s) {offenders} banked fewer levels than baseline ({baseline_levels_total}) on the "
            "listed game(s) -- the exact failure the trust+specificity gate is meant to prevent. "
            "Operator-only whether to investigate the gate (a higher min_observations / min_specificity "
            "or a longer sacred-protection window); the live default stays False."
        )
    elif any_beats:
        recommendation = (
            f"Arm {best_treatment_arm} banked MORE levels ({best_treatment_levels}) than baseline "
            f"({baseline_levels_total}). This is unexpected for an efficiency filter (a pruner removes "
            "clicks; it does not add reachable states) and warrants scrutiny for an RNG/seed artifact "
            "before any flip. Operator-only; do NOT self-authorize flipping the live default on a "
            "single-seed surprise."
        )
    elif any_pruner_fired and pruner_reduced_states:
        pruned_total_all = sum(total_pruned_by_arm[a] for a in TREATMENT_ARMS)
        clean = [
            a
            for a in TREATMENT_ARMS
            if not per_config_safety[a]["states_expanded_regression"]
            and not per_config_safety[a]["suppressed_a_winnable_click"]
        ]
        recommendation = (
            f"At budget=200 the pruner ENGAGES (pruned {pruned_total_all} candidates across treatment "
            f"arms), HOLDS banked levels at baseline's {baseline_levels_total} with NO missed win, AND "
            f"REDUCES states_expanded (best fired arm delta {best_fired_states_delta:+d}). Arm(s) "
            f"{clean} prune cleanly. This is a genuine candidate efficiency win (fewer wasted clicks at "
            "equal capability, lower search cost), mirroring HazardMovePruner's tu93 result. Operator-"
            "only whether to flip SUBMITTED_INERT_CLICK_PRUNER_ENABLED; the agent does not self-"
            "authorize, and a single-seed states reduction should be confirmed across more seeds first."
        )
    elif any_pruner_fired:
        pruned_total_all = sum(total_pruned_by_arm[a] for a in TREATMENT_ARMS)
        fired_games = sorted(
            {g for a in fired_arms for g in roster if sweep[a][g]["pruner"]["pruned"] > 0}
        )
        recommendation = (
            f"At budget=200 the pruner ENGAGES (pruned {pruned_total_all} candidates across treatment "
            f"arms, on game(s) {fired_games}) and HOLDS banked levels at baseline's "
            f"{baseline_levels_total} with NO missed win -- the trust+specificity+sacred gate correctly "
            "avoided suppressing any winnable click. HOWEVER, states_expanded did NOT decrease (best "
            f"fired arm delta {best_fired_states_delta:+d}); on the game(s) where it actually pruned, "
            "dropping click candidates reshaped the search frontier so the search expanded MORE states "
            "for the same banked levels -- the OPPOSITE of HazardMovePruner's tu93 efficiency win. So "
            "at this budget the inert-click pruner buys no capability AND no efficiency: do NOT flip "
            "SUBMITTED_INERT_CLICK_PRUNER_ENABLED. Operator-only whether to investigate why pruning "
            "these signatures grows the frontier (a click that is frame-inert may still be a necessary "
            "traversal step the search re-routes around), but on this evidence the live default should "
            "stay False. Per the retire condition, another plain budget/roster A/B is not warranted "
            "without a NEW mechanism that makes pruning reduce rather than reshape search cost."
        )
    elif any_floor_reached:
        recommendation = (
            f"At budget=200 some signatures cross the min_observations evidence floor (partial "
            "engagement beyond exp5595/exp5602's zero), but the min_specificity=0.9 gate prunes ZERO "
            f"candidates -- treatment is identical to baseline at {baseline_levels_total} levels. The "
            "inert-click mechanism does not add live-path capability at this budget; do NOT flip the "
            "live default. Operator-only whether a lower specificity bar is worth a follow-up, but "
            "per the retire condition this is close to a same-verdict outcome -- prefer a NEW keying "
            "mechanism over another threshold sweep."
        )
    else:
        recommendation = (
            f"NO signature crosses the evidence floor in ANY arm at budget=200 across the full 11-game "
            f"roster (0 pruned, treatment byte-identical to baseline at {baseline_levels_total} "
            "levels). This GENERALIZES exp5595/exp5602's single-game m0r0 null to the full roster at a "
            "5x budget: the inert-click-signature mechanism does not engage at reachable live-agent "
            "budgets. Per the retire condition, recommend the operator retire the live-default flip "
            "consideration for SUBMITTED_INERT_CLICK_PRUNER_ENABLED (operator-only; it stays False) and "
            "close the gap per Missing-Verifier Gap Logging. Do NOT re-propose another budget/roster "
            "A/B without a NEW signature-keying mechanism that accumulates evidence faster."
        )

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "roster": list(roster),
        "budget": int(budget),
        "arms_tested": arm_names,
        "baseline_arm": BASELINE_ARM,
        "baseline_levels_total": int(baseline_levels_total),
        "baseline_states_expanded_total": int(baseline_states_total),
        "per_arm_results": per_arm_results,
        "per_arm_game_rows": per_arm_game_rows,
        "levels_gained_total_by_arm": levels_by_arm,
        "states_expanded_total_by_arm": states_by_arm,
        "candidates_pruned_total_by_arm": {a: int(total_pruned_by_arm[a]) for a in arm_names},
        "signatures_reaching_evidence_floor_total_by_arm": {
            a: int(total_floor_by_arm[a]) for a in arm_names
        },
        "n_signatures_reaching_evidence_floor_per_game": n_floor_per_game,
        "n_candidates_pruned_per_game": n_pruned_per_game,
        "trajectories_diverge_per_game": trajectories_diverge_per_game,
        "any_pruner_fired": bool(any_pruner_fired),
        "pruner_reduced_states_expanded": bool(pruner_reduced_states),
        "best_fired_arm_states_delta_vs_baseline": int(best_fired_states_delta)
        if any_pruner_fired
        else 0,
        "any_signature_reached_evidence_floor": bool(any_floor_reached),
        "any_treatment_diverges_from_baseline": bool(any_treatment_diverges),
        "blob_cache_perf_fix": blob_cache_perf_fix,
        "levels_gained_headroom_present": bool(any_headroom),
        "any_config_beats_baseline_levels": bool(any_beats),
        "safety_regression_check": safety_regression_check,
        "prior_work_extended": PRIOR_WORK_EXTENDED,
        "recommendation": recommendation,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
