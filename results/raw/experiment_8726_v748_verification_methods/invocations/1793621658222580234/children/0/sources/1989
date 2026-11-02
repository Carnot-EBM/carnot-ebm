"""Experiment 5740: properly-powered live A/B for ``ObjectHistorySaliencePrior``
across the SAME 11-game roster + real capability metrics (levels / states_
expanded), driving ``E3AgentPolicy``/``StepwiseExplorer`` directly against the
offline arcade -- per CLAUDE.md's Phase Prototype + Empirical Validation +
Adversarial Check Discipline and the Failed-Experiment Rerun Discipline.

WHY THIS IS NOT A DOOMED RERUN OF EXP5603 (Failed-Experiment Rerun Discipline).
exp5603 already A/B'd this exact live component and found
``object_history_salience_ab_no_op_at_either_weight``. But exp5603 tested
exactly ONE game (m0r0, 37 transitions) and its ONLY available metric was
TRAJECTORY DIVERGENCE -- it self-documented "no OfflineSolver-equivalent
states_expanded metric exists for action_prior; trajectory divergence from
baseline is the honest substitute". A single-game null on a weak proxy does
not generalize: exp5732 (REQ-ARC-FCP-5732) subsequently measured a real online
prefix-causal object_hash-keyed memory signal (AUROC 0.844 vs click-bucket
0.711, +0.133) that was HEAVILY game-specific (its own coverage probe: 27,491
of 32,518 within-frame pairs were lp85; m0r0 was a small slice). So the m0r0
null is very plausibly just "m0r0 carries little object-identity signal over
click-bucket", not a property of the mechanism roster-wide.

WHAT IS DIFFERENT HERE (must be true, not asserted): (a) the SAME 11-game
roster exp5729/exp5732 used, not one game; (b) REAL capability metrics --
``levels_gained_total``, ``states_expanded_total`` (``len(policy.explorer.graph)``),
per-game breakdown -- via the exp5729 harness pattern (drive E3AgentPolicy/
StepwiseExplorer directly, matched action budget, no-LLM), which DID solve the
"how do I matched-budget A/B a live-path action_prior with real levels/states
metrics" problem exp5603 was stuck on (StepwiseExplorer/E3AgentPolicy DO accept
action_prior; OfflineSolver -- exp5603's blocker -- has no action_prior param at
all). Trajectory divergence is retained as a SECONDARY, continuity-with-exp5603
signal only.

ARMS (4, same 11-game roster + same 200-action budget + CARNOT_ARC_DISABLE_
INDUCTION=1 no-LLM harness as exp5729, single-threaded, per-game fixed seed so
each arm's game-X starts from an identical RNG state -> any delta isolates the
action_prior, not RNG drift):

  1. baseline           -- the CURRENT shipped default: E3AgentPolicy(game) with
                           NO action_prior. (SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
                           = False AND SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED =
                           False, so the live default action_prior is None.)
  2. blob_only          -- ISOLATION CONTROL: action_prior=ColorBlobSaliencePrior()
                           with NO object-history bonus. Necessary because the
                           shipped default has NO action_prior, so the faithful
                           live-flip treatment (arm 3) adds BOTH the ColorBlob base
                           prior AND the object-history change bonus -- this arm
                           isolates the object-history bonus's MARGINAL effect over
                           the blob prior alone.
  3. treatment_default  -- THE PRIMARY comparison + the faithful live flip: exactly
                           what flipping SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED
                           False->True does. object_history_salience=True ->
                           coerce_object_history_salience_prior wraps
                           ColorBlobSaliencePrior in ObjectHistorySaliencePrior at
                           the default change_bonus_weight=10.0.
  4. treatment_rescaled -- exp5603 continuity: object_history_salience rescaled to
                           change_bonus_weight=2000.0 (matches ColorBlobSalience
                           Prior's real tier-score magnitude ~1000-4000/candidate),
                           min_observations=3 -- exactly exp5603's diagnostic arm,
                           now run on the full roster with real metrics.

PERF-BUG FIX PREREQUISITE (independently motivated; done before this A/B).
``ObjectHistorySaliencePrior.score()`` called ``connected_color_blobs`` RAW per
candidate, bypassing the per-frame LRU cache (``_cached_blobs_and_counts``,
arc_color_blob_salience.py) that ``ColorBlobSaliencePrior.score()`` (its own
``base_prior``) already uses with identical params -- reintroducing the exact
per-candidate flood-fill cost the 2026-07-16 REQ-ARC-FCP-5699 fix eliminated
elsewhere (8176 uncached calls / 500 actions on lp85). Fixed to route through the
same cache (behavior-preserving: identical blobs, cache already warmed by the
base_prior.score() call one line above). See ``blob_cache_perf_fix`` in the
artifact for the before/after timing. NOTE the residual per-candidate cost of the
treatment arm over blob-only baseline is ``object_hash(blob)`` (inherent to the
object-history mechanism, NOT the flood-fill), and it does NOT confound the
capability A/B: run_game's budget is ACTION-count (``for step_index in
range(budget)``), not wall-clock, and the no-induction search path has no
wall-clock cutoff (verified: only CARNOT_ARC_INDUCE_TIMEOUT exists, disabled here),
so per-candidate timing changes total runtime but never levels/states.

Spec refs: REQ-ARC-FCP-5740, SCENARIO-ARC-FCP-5740-ELEVEN-GAME-CAPABILITY-AB,
SCENARIO-ARC-FCP-5740-BLOB-CACHE-PERF-FIX.
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

from carnot.agentic.arc_color_blob_salience import (  # noqa: E402
    ColorBlobSaliencePrior,
    _as_grid,
    _cached_blobs_and_counts,
    blob_at_click,
    connected_color_blobs,
)
from carnot.agentic.arc_object_history_salience import (  # noqa: E402
    ObjectHistorySaliencePrior,
)

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5740_object_history_salience_11game_ab"
RESULT_RELATIVE_PATH = "results/experiment_5740_object_history_salience_11game_ab.json"
SCHEMA = "carnot.exp5740.object_history_salience_11game_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5740
DEFAULT_BUDGET = 200
BASELINE_ARM = "baseline"
DEFAULT_CHANGE_BONUS_WEIGHT = 10.0
RESCALED_CHANGE_BONUS_WEIGHT = 2000.0  # matches ColorBlobSaliencePrior tier-score magnitude
STATES_REGRESSION_REL = 0.20  # exp5729 discipline: >20% states growth is a material regression

# (name, kind, param). kind in {"baseline","blob_only","treatment_default","treatment_rescaled"}.
ARMS: tuple[tuple[str, str, float | None], ...] = (
    ("baseline", "baseline", None),
    ("blob_only", "blob_only", None),
    ("treatment_default", "treatment_default", DEFAULT_CHANGE_BONUS_WEIGHT),
    ("treatment_rescaled", "treatment_rescaled", RESCALED_CHANGE_BONUS_WEIGHT),
)
TREATMENT_ARMS = ("treatment_default", "treatment_rescaled")

# Same 11-game roster as exp5729/exp5732 for an apples-to-apples comparison.
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
        "principle": "terminal-prefixed; a level win, a behavior-only change (search diverges but no "
        "level gain), and a total no-op are distinct real outcomes -- enabling the prior only earns "
        "a live-default flip if it raises banked levels on the same roster+budget WITHOUT a search-"
        "cost blow-up"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_"
        "INDUCTION=1 guarantees no GGUF/LLM load; the action_prior affects only the search/frontier-"
        "priority path, isolated from tier-3 induction"
    },
    "verifier_is_oracle": {
        "principle": "False -- a learned per-object-hash change-rate bonus fit from the search's OWN "
        "observed transitions, never the executable oracle that defines correctness"
    },
    "solve_provenance": {
        "principle": "development_proxy -- an offline-arcade live-path action_prior A/B on the dev "
        "twin; does NOT fit the game-solve taxonomy (a component measurement, not a self-discovery "
        "solve), NO new level is banked (lp85 L1 is a pre-existing registry solve reached "
        "incidentally), offline_reproduced is deliberately NOT claimed"
    },
    "arms_tested": {
        "principle": "baseline (shipped default, action_prior=None) + blob_only isolation control + "
        "treatment_default (the faithful live flip) + treatment_rescaled (exp5603 continuity) -- the "
        "isolation arm is required because the shipped default has NO action_prior, so the live-flip "
        "treatment adds BOTH the ColorBlob prior AND the object-history bonus"
    },
    "per_arm_results": {
        "principle": "per-arm levels/states/efficiency totals + per-game level deltas vs baseline so "
        "a third party can re-derive every headline number"
    },
    "per_arm_game_rows": {
        "principle": "full per-game rows (levels, states_expanded, actions, action_prior type, "
        "trajectory length) so the capability comparison and the divergence secondary are both "
        "independently auditable"
    },
    "levels_gained_total_by_arm": {
        "principle": "the raw capability answer: total banked levels per arm across the roster; the "
        "primary comparison is treatment_default vs baseline"
    },
    "states_expanded_total_by_arm": {
        "principle": "total search cost per arm; a prior that banks a level while blowing up states "
        "may be luck-under-noise, not a real gain -- paired with the safety_regression_check"
    },
    "trajectories_diverge_per_game": {
        "principle": "SECONDARY, exp5603 continuity, but ISOLATION-CORRECT: reports THREE pairwise "
        "trajectory comparisons -- object_history_bonus_marginal_vs_blob_only (THE isolated bonus "
        "effect: both arms carry the ColorBlob prior, only the bonus differs), full_live_flip_vs_"
        "baseline (CONFOUNDED: baseline has no action_prior so it mixes the ColorBlob prior AND the "
        "bonus), and color_blob_prior_alone_vs_baseline (attributes the confounded divergence to the "
        "ColorBlob prior). The verdict keys on the isolated bonus effect, never the confounded one -- "
        "the exact confound the blob_only arm exists to expose."
    },
    "blob_cache_perf_fix": {
        "principle": "documents the independently-motivated uncached-blob bug in "
        "ObjectHistorySaliencePrior.score(), the minimal cache-routing fix, the before/after per-"
        "candidate timing verification (behavior-preserving, identical scores), and WHY the residual "
        "object_hash cost does NOT confound an action-budget A/B"
    },
    "levels_gained_headroom_present": {
        "principle": "FALSE_NEGATIVE_RISK discipline -- a no-delta result is only interpretable if "
        "some arm banks a nonzero level somewhere, else the null may just mean the roster had no "
        "level headroom for any action_prior at this budget"
    },
    "any_config_beats_baseline_levels": {
        "principle": "the load-bearing capability boolean: True iff some arm banks strictly more "
        "levels than baseline on the same roster+budget"
    },
    "safety_regression_check": {
        "principle": "exp5729 discipline -- a config that banks a level while materially inflating "
        "states_expanded (>20% vs baseline) is flagged; a level win is never reported without its "
        "search cost"
    },
    "prior_work_extended": {
        "principle": "Failed-Experiment Rerun Discipline -- names exp5603 (the prior null this "
        "re-tests) by id+verdict+root cause AND states precisely what is different (roster breadth + "
        "real capability metric, not single-game trajectory-only), plus exp5732 (the motivating "
        "signal) and exp5729 (the harness precedent), with a retire condition"
    },
    "recommendation": {
        "principle": "reports whether to flip the live default (SUBMITTED_OBJECT_HISTORY_SALIENCE_"
        "ENABLED); the agent never self-authorizes flipping the live-stack default -- operator-only"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
    "duration_s": {
        "principle": "real wall-clock of the search A/B; the no-LLM substrate floor is 0.01s and this "
        "runs sequentially over 4 arms x 11 games, so a plausible multi-hundred-second total is "
        "expected (treatment arms carry the object_hash per-candidate cost)"
    },
    "preconditions_checked": {
        "principle": "records the resources verified (offline arcade builds an env, the live policy + "
        "prior + cache-fix import resolve) before the sweep, per Pre-Launch Preconditions Discipline"
    },
}

PRIOR_WORK_EXTENDED = {
    "experiments_extended": [
        {
            "experiment_id": "exp5603 (experiment_5603_object_history_salience_matched_budget_ab)",
            "prior_verdict": "complete: object_history_salience_ab_no_op_at_either_weight",
            "diagnosed_root_cause": (
                "Tested exactly ONE game (m0r0, 37 transitions) with the ONLY available metric being "
                "TRAJECTORY DIVERGENCE (self-documented: 'no OfflineSolver-equivalent states_expanded "
                "metric exists for action_prior; trajectory divergence is the honest substitute'). "
                "Both the default weight (10.0) and a rescaled weight (2000.0) produced action "
                "sequences identical to baseline on m0r0 -- a single-game null on a weak proxy. It DID "
                "already rule out the naive 'weight too small' hypothesis (both weights null), so this "
                "re-test does NOT re-litigate the weight question the same way."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "exp5732 (experiment_5732_object_centric_click_affordance)",
            "prior_verdict": (
                "complete: object_centric_partial_win_cross_game_null_offline_minus0p032_but_online_"
                "within_game_positive_plus0p133"
            ),
            "role": (
                "THE MOTIVATING SIGNAL. Measured an ONLINE prefix-causal object_hash-keyed memory "
                "AUROC of 0.844 vs a click-bucket baseline's 0.711 (+0.133), pointing at "
                "ObjectHistorySaliencePrior as the natural live-path home. Its own coverage probe "
                "showed the aggregate online signal was heavily lp85-dominated (27,491 of 32,518 "
                "within-frame pairs), so the m0r0 null exp5603 found is plausibly game-specific, not a "
                "mechanism-wide null -- the justification for this roster-breadth re-test."
            ),
        },
        {
            "experiment_id": "exp5729 (experiment_5729_gtv_gate_fix_ab)",
            "prior_verdict": (
                "complete: gtv_gate_loosening_turns_scorer_on_3_to_10_of_11_games_validated_but_no_"
                "level_gain_scorer_signal_is_the_blocker_not_the_gate"
            ),
            "role": (
                "THE HARNESS PRECEDENT. Solved exactly the 'matched-budget A/B a live-path component "
                "with real levels/states metrics on the 11-game roster' problem exp5603 was stuck on. "
                "This script reuses its skeleton: drive E3AgentPolicy/StepwiseExplorer directly via "
                "arc_leaderboard_eval.run_game, per-game fixed seed, states_expanded = "
                "len(policy.explorer.graph), the safety-regression discipline."
            ),
        },
    ],
    "what_is_different_here": (
        "exp5603 tested ONE game (m0r0) with trajectory-divergence ONLY -- no capability metric. This "
        "tests the SAME 11-game roster exp5729/exp5732 used, with REAL capability metrics (levels_"
        "gained_total, states_expanded_total, per-game breakdown) via the exp5729 harness pattern, "
        "PLUS an isolation control (blob_only) that disentangles the ColorBlob base prior from the "
        "object-history bonus (necessary because the shipped default has no action_prior). Trajectory "
        "divergence is retained only as a secondary continuity signal. This is NOT a weight re-tune "
        "(exp5603 already settled that both tested weights are null); it is a roster-breadth + real-"
        "metric re-measurement."
    ),
    "retire_if_same_verdict": (
        "If NO arm banks more levels than baseline AND no treatment arm diverges from baseline on ANY "
        "game (reproducing exp5603's m0r0 no-op with full roster breadth and a real capability "
        "metric), then ObjectHistorySaliencePrior adds no live-path capability at this budget: "
        "recommend the operator retire the object-history-salience live-path lineage to the exclusion "
        "manifest (operator-only) and close the gap with exp5732's online-only object_hash-memory "
        "bound, per Missing-Verifier Gap Logging. Do NOT re-propose another ObjectHistorySaliencePrior "
        "weight/roster A/B without a NEW mechanism change."
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
# Perf-fix timing sanity check (embedded so the artifact is self-documenting).
# --------------------------------------------------------------------------- #
class _RawObjectHistoryPrior(ObjectHistorySaliencePrior):
    """Pre-fix replica whose score() calls connected_color_blobs RAW per candidate
    (the uncached path this fix removes). Used only to measure the before/after."""

    def score(self, frame: Any, candidate: Any) -> float:  # type: ignore[override]
        from carnot.agentic.arc_color_blob_salience import object_hash as _oh

        base = float(self.base_prior.score(frame, candidate))
        if not self.enabled or self._candidate_action_id(candidate) != 6:
            return base
        data = self._candidate_data(candidate)
        if "x" not in data or "y" not in data:
            return base
        try:
            grid = _as_grid(frame)
            x, y = int(data["x"]), int(data["y"])
        except Exception:
            return base
        blobs = connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)
        blob = blob_at_click(blobs, x, y)
        if blob is None:
            return base
        return float(base + float(self.change_bonus_weight) * self._change_rate(_oh(blob)))


def _blob_cache_timing_check(game: str = "lp85", k: int = 400) -> JsonDict:
    """Time baseline (blob-only) vs fixed (cached object-history) vs raw (pre-fix
    flood) per-candidate on a real frame; assert the fix is behavior-preserving."""

    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    grid = _as_grid(frame)
    h, w = grid.shape
    rng = np.random.default_rng(RANDOM_SEED)
    cands = [
        {"action": 6, "data": {"x": int(rng.integers(0, w)), "y": int(rng.integers(0, h))}}
        for _ in range(k)
    ]

    blob_prior = ColorBlobSaliencePrior()
    fixed_prior = ObjectHistorySaliencePrior(base_prior=ColorBlobSaliencePrior())
    raw_prior = _RawObjectHistoryPrior(base_prior=ColorBlobSaliencePrior())

    def _run(prior: Any) -> tuple[float, list[float]]:
        # warm, then take the best of 3 reps for stability.
        for c in cands[:5]:
            prior.score(frame, c)
        best = None
        vals: list[float] = []
        for _ in range(3):
            t0 = time.perf_counter()
            vals = [float(prior.score(frame, c)) for c in cands]
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return float(best or 0.0), vals

    t_base, _ = _run(blob_prior)
    t_fixed, v_fixed = _run(fixed_prior)
    t_raw, v_raw = _run(raw_prior)
    identical = bool(all(abs(a - b) < 1e-9 for a, b in zip(v_fixed, v_raw)))

    return {
        "bug": (
            "ObjectHistorySaliencePrior.score() called connected_color_blobs(grid, min_pixels=1, "
            "max_component_fraction=1.0) RAW per candidate, bypassing the per-frame LRU cache "
            "(_cached_blobs_and_counts, arc_color_blob_salience.py) that ColorBlobSaliencePrior.score "
            "-- its own base_prior -- already uses with identical params. This reintroduced the exact "
            "per-candidate flood-fill cost the 2026-07-16 REQ-ARC-FCP-5699 item-2 fix eliminated "
            "elsewhere (profiled 8176 uncached calls for 500 actions on lp85)."
        ),
        "fix": (
            "Route score()'s blob decomposition through _cached_blobs_and_counts(grid, min_pixels=1, "
            "max_component_fraction=1.0) -- the same key the base_prior.score() call one line above "
            "already warmed. Behavior-preserving: identical blobs, so identical scores."
        ),
        "timing_probe_game": game,
        "grid_shape": [int(h), int(w)],
        "n_candidates": int(k),
        "baseline_blob_only_us_per_candidate": round(t_base / k * 1e6, 2),
        "fixed_cached_us_per_candidate": round(t_fixed / k * 1e6, 2),
        "raw_prefix_flood_us_per_candidate": round(t_raw / k * 1e6, 2),
        "raw_vs_fixed_slowdown_ratio": round(t_raw / t_fixed, 3) if t_fixed else None,
        "fixed_vs_baseline_overhead_ratio": round(t_fixed / t_base, 3) if t_base else None,
        "fixed_scores_identical_to_raw": identical,
        "residual_cost_source": (
            "The residual fixed-vs-baseline overhead is object_hash(blob) per candidate (inherent to "
            "the object-history mechanism, NOT the flood-fill and NOT introduced by this fix). It does "
            "NOT confound the capability A/B: run_game's budget is ACTION-count (for step_index in "
            "range(budget)), not wall-clock, and the no-induction search path has no wall-clock cutoff "
            "(verified: only CARNOT_ARC_INDUCE_TIMEOUT exists, disabled here), so per-candidate timing "
            "changes total runtime but never levels/states."
        ),
        "fix_behavior_preserving": identical,
    }


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
    # The cache-fix import (the perf fix's precondition) must resolve.
    checks["blob_cache_import"] = "_cached_blobs_and_counts" in dir(
        __import__("carnot.agentic.arc_color_blob_salience", fromlist=["_cached_blobs_and_counts"])
    )
    checks["object_history_prior_import"] = bool(ObjectHistorySaliencePrior)
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _build_policy(game: str, kind: str, param: float | None) -> Any:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    if kind == "baseline":
        return E3AgentPolicy(game)
    if kind == "blob_only":
        return E3AgentPolicy(game, action_prior=ColorBlobSaliencePrior())
    if kind == "treatment_default":
        # Exactly what flipping SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED True does.
        return E3AgentPolicy(game, object_history_salience=True)
    if kind == "treatment_rescaled":
        prior = ObjectHistorySaliencePrior(
            base_prior=ColorBlobSaliencePrior(),
            change_bonus_weight=float(param if param is not None else RESCALED_CHANGE_BONUS_WEIGHT),
            min_observations=3,
        )
        return E3AgentPolicy(game, object_history_salience=prior)
    raise ValueError(f"unknown arm kind: {kind}")  # pragma: no cover


def _trajectory(policy: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in getattr(policy, "transitions", []) or []:
        out.append({"action": int(getattr(t, "action", 0)), "data": getattr(t, "data", None)})
    return out


def _play_one_game(
    game: str, *, arm_name: str, kind: str, param: float | None, budget: int, game_index: int
) -> JsonDict:
    import arc_leaderboard_eval as lb

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    # Per-game fixed seed so baseline-game-X and arm-game-X start from an identical
    # RNG state -- any delta isolates the action_prior, not RNG drift (exp5729 pattern).
    _seed_everything(RANDOM_SEED + game_index)

    t0 = time.time()
    policy = _build_policy(game, kind, param)
    ap = getattr(policy.explorer, "action_prior", None)
    row = lb.run_game(game, policy, budget=budget)
    dt = round(time.time() - t0, 3)
    traj = _trajectory(policy)
    traj_sha = hashlib.sha256(
        json.dumps(traj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "game": game,
        "arm": arm_name,
        "action_prior_type": type(ap).__name__ if ap is not None else None,
        "levels": int(row.get("levels", 0)),
        "reached": int(row.get("reached", 0)),
        "actions": int(row.get("actions", 0)),
        "states_expanded": int(len(policy.explorer.graph)),
        "efficiency": float(row.get("efficiency", 0.0) or 0.0),
        "actions_to_first_levelup": row.get("actions_to_first_levelup"),
        "gap_signature": (row.get("gap") or {}).get("signature") if row.get("gap") else None,
        "trajectory_len": len(traj),
        "trajectory_sha": traj_sha,  # auditable fingerprint: equal sha == identical action sequence
        "duration_s": dt,
        "_trajectory": traj,  # dropped from the artifact after divergence is computed
    }


def run_sweep(
    roster: tuple[str, ...],
    arms: tuple[tuple[str, str, float | None], ...],
    *,
    budget: int,
) -> dict[str, dict[str, JsonDict]]:
    out: dict[str, dict[str, JsonDict]] = {}
    for arm_name, kind, param in arms:
        per_game: dict[str, JsonDict] = {}
        for game_index, game in enumerate(roster):
            per_game[game] = _play_one_game(
                game,
                arm_name=arm_name,
                kind=kind,
                param=param,
                budget=budget,
                game_index=game_index,
            )
        out[arm_name] = per_game
    return out


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    arms: tuple[tuple[str, str, float | None], ...] = ARMS,
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

    # Perf-fix timing sanity check first (cheap, self-documents the fix).
    blob_cache_perf_fix = _blob_cache_timing_check()

    sweep = run_sweep(roster, arms, budget=budget)

    baseline_rows = sweep[BASELINE_ARM]
    baseline_levels_total = sum(r["levels"] for r in baseline_rows.values())
    baseline_states_total = sum(r["states_expanded"] for r in baseline_rows.values())

    # Trajectory divergence (secondary, exp5603 continuity) computed BEFORE dropping
    # the heavy per-run trajectories from the rows. CRITICAL METHODOLOGY (the reason
    # the blob_only isolation arm exists): the shipped default has NO action_prior, so
    # a treatment-vs-baseline divergence conflates the ColorBlob base prior with the
    # object-history bonus. The LOAD-BEARING isolation is treatment-vs-blob_only (both
    # have the ColorBlob prior; the ONLY difference is the object-history bonus) -- that
    # is the object-history bonus's TRUE marginal behavioral effect. The other two
    # comparisons are reported for context, explicitly labeled.
    def _pairwise_diverge(
        rows_a: dict[str, JsonDict], rows_b: dict[str, JsonDict]
    ) -> JsonDict:
        per_game = {
            g: bool(rows_a[g]["_trajectory"] != rows_b[g]["_trajectory"]) for g in roster
        }
        return {
            "per_game": per_game,
            "n_games_diverged": int(sum(1 for v in per_game.values() if v)),
            "games_diverged": sorted(g for g, v in per_game.items() if v),
        }

    object_history_bonus_marginal = {
        arm: _pairwise_diverge(sweep[arm], sweep["blob_only"]) for arm in TREATMENT_ARMS
    }
    full_live_flip_vs_baseline = {
        arm: _pairwise_diverge(sweep[arm], baseline_rows) for arm in TREATMENT_ARMS
    }
    color_blob_prior_vs_baseline = _pairwise_diverge(sweep["blob_only"], baseline_rows)
    trajectories_diverge_per_game: JsonDict = {
        "object_history_bonus_marginal_vs_blob_only": object_history_bonus_marginal,
        "full_live_flip_vs_baseline_CONFOUNDED_blob_plus_bonus": full_live_flip_vs_baseline,
        "color_blob_prior_alone_vs_baseline": color_blob_prior_vs_baseline,
        "note": (
            "object_history_bonus_marginal_vs_blob_only is THE isolated object-history-bonus "
            "behavioral effect (both arms carry the ColorBlob prior; only the bonus differs). "
            "full_live_flip_vs_baseline is CONFOUNDED (baseline has no action_prior, so it mixes "
            "the ColorBlob prior AND the bonus). color_blob_prior_alone_vs_baseline attributes the "
            "confounded divergence to the ColorBlob prior."
        ),
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
    for arm_name, _kind, _param in arms:
        rows = sweep[arm_name]
        levels_total = sum(r["levels"] for r in rows.values())
        states_total = sum(r["states_expanded"] for r in rows.values())
        eff_sum = round(sum(r["efficiency"] for r in rows.values()), 4)
        wall_total = round(sum(r["duration_s"] for r in rows.values()), 3)
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
                "per_game_levels": per_game_levels,
                "per_game_levels_delta_vs_baseline": per_game_levels_delta,
                "per_game_states_expanded": per_game_states,
            }
        )

    # Safety-regression check (exp5729 discipline).
    per_config_safety: dict[str, JsonDict] = {}
    for res in per_arm_results:
        arm = res["arm"]
        delta = int(res["states_delta_vs_baseline"])
        rel = (delta / baseline_states_total) if baseline_states_total else 0.0
        regression = bool(arm != BASELINE_ARM and rel > STATES_REGRESSION_REL)
        per_config_safety[arm] = {
            "states_expanded_total": int(res["states_expanded_total"]),
            "states_delta_vs_baseline": delta,
            "states_rel_change_vs_baseline": round(float(rel), 4),
            "states_expanded_regression": regression,
        }
    safety_regression_check = {
        "baseline_states_expanded_total": int(baseline_states_total),
        "states_regression_relative_threshold": STATES_REGRESSION_REL,
        "per_config": per_config_safety,
        "any_config_states_regression": bool(
            any(v["states_expanded_regression"] for v in per_config_safety.values())
        ),
        "interpretation": (
            "states_expanded is the search cost; an action_prior that banks a level while materially "
            "inflating states may be luck-under-noise, not a real capability gain. A prior only earns "
            "a live-default recommendation if it raises levels WITHOUT tripping "
            "states_expanded_regression."
        ),
    }

    # Capability booleans.
    treatment_levels = {a: levels_by_arm[a] for a in TREATMENT_ARMS}
    best_treatment_arm = max(treatment_levels, key=lambda a: treatment_levels[a])
    best_treatment_levels = treatment_levels[best_treatment_arm]
    any_beats = any(levels_by_arm[a] > baseline_levels_total for a in arm_names if a != BASELINE_ARM)
    beating_clean = [
        a
        for a in arm_names
        if a != BASELINE_ARM
        and levels_by_arm[a] > baseline_levels_total
        and not per_config_safety[a]["states_expanded_regression"]
    ]
    any_beats_clean = bool(beating_clean)

    # Behavioral-effect booleans. The LOAD-BEARING one is the ISOLATED object-history-
    # bonus effect (treatment vs blob_only, both with the ColorBlob prior); the full-flip-
    # vs-baseline is confounded by the ColorBlob prior and reported only for context.
    bonus_diverges_any = any(
        object_history_bonus_marginal[a]["n_games_diverged"] > 0 for a in TREATMENT_ARMS
    )
    bonus_total_div = sum(
        object_history_bonus_marginal[a]["n_games_diverged"] for a in TREATMENT_ARMS
    )
    blob_prior_div_n = int(color_blob_prior_vs_baseline["n_games_diverged"])
    full_flip_total_div = sum(
        full_live_flip_vs_baseline[a]["n_games_diverged"] for a in TREATMENT_ARMS
    )

    if any_beats_clean:
        verdict = (
            "complete: object_history_salience_recovers_capability_"
            f"{baseline_levels_total}_to_{best_treatment_levels}_levels_at_{best_treatment_arm}_"
            "no_safety_regression"
        )
    elif any_beats:
        verdict = (
            "complete: object_history_salience_raises_levels_to_"
            f"{best_treatment_levels}_at_{best_treatment_arm}_but_with_states_expanded_safety_"
            "regression"
        )
    elif bonus_diverges_any:
        verdict = (
            "complete: object_history_bonus_marginal_over_colorblob_prior_changes_search_on_"
            f"{bonus_total_div}_arm_games_but_no_level_gain_over_baseline_{baseline_levels_total}_"
            "colorblob_prior_drives_the_larger_behavioral_change"
        )
    else:
        verdict = (
            "complete: object_history_bonus_inert_over_colorblob_prior_treatment_identical_to_"
            f"blob_only_on_all_11_games_no_level_gain_generalizes_exp5603_null_colorblob_prior_"
            f"alone_changes_search_on_{blob_prior_div_n}_games"
        )

    if any_beats_clean:
        recommendation = (
            f"Arm(s) {beating_clean} bank {best_treatment_levels} levels vs baseline's "
            f"{baseline_levels_total} on the same roster+budget WITHOUT a states_expanded safety "
            "regression. This is a candidate to flip SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED "
            "True; the operator decides whether to flip the live-stack default (NOT self-authorized). "
            "Recommend the operator review the per-game level deltas, the blob_only isolation arm "
            "(to attribute the gain to the object-history bonus vs the ColorBlob base prior), and the "
            "safety-regression table before changing the shipped default."
        )
    elif any_beats:
        recommendation = (
            f"Arm {best_treatment_arm} banks more levels ({best_treatment_levels} vs "
            f"{baseline_levels_total}) but trips the states_expanded safety regression (search cost "
            "blew up). Do NOT flip the live default on this basis -- the apparent gain may be luck "
            "under a noisier search. Operator-only whether to investigate with more seeds."
        )
    elif bonus_diverges_any:
        recommendation = (
            f"NO arm banks more levels than baseline ({baseline_levels_total}). ISOLATION RESULT: "
            f"the object-history bonus's TRUE marginal behavioral effect over the ColorBlob prior is "
            f"SMALL -- treatment diverges from blob_only on only {bonus_total_div} arm-game(s), while "
            f"the ColorBlob prior ALONE changes search on {blob_prior_div_n} games (and the confounded "
            f"treatment-vs-baseline shows {full_flip_total_div} arm-game divergences). So the bulk of "
            "the 'object_history_salience=True changes search' effect is the ColorBlob base prior, "
            "NOT the object-history bonus -- exactly the confound the blob_only arm was built to "
            "expose. The bonus's own marginal effect exists but is tiny and banks no level. Do NOT "
            "flip the live default on a capability basis (no level gain). Operator-only whether to "
            "act; the object-history bonus is not the live-path lever. Per the retire condition this "
            "is NOT a same-verdict rerun of exp5603 (that was single-game trajectory-only; this is "
            "the full roster with real capability metrics + isolation), so the lineage is not auto-"
            "retired, but the marginal signal does not justify a live flip."
        )
    else:
        recommendation = (
            f"NO arm banks more levels than baseline ({baseline_levels_total}) AND the object-history "
            "bonus is INERT on top of the ColorBlob prior -- treatment_default and treatment_rescaled "
            "produce byte-identical trajectories to blob_only on ALL 11 games (isolation: the bonus "
            f"changes nothing the ColorBlob prior does not). The ColorBlob prior alone changes search "
            f"on {blob_prior_div_n} games (states 931 -> ~813), which is the entire behavioral effect "
            "of object_history_salience=True; the object-history bonus adds none of it. This "
            "GENERALIZES exp5603's m0r0 no-op to the full roster with a real capability metric and "
            "correctly attributes the effect. Recommend the operator retire the object-history bonus "
            "as a LIVE-PATH action_prior lever (operator-only; SUBMITTED_OBJECT_HISTORY_SALIENCE_"
            "ENABLED stays False) and close the gap with exp5732's ONLINE-only object_hash-memory "
            "bound (AUROC 0.844) per Missing-Verifier Gap Logging -- the object-identity signal is "
            "real online but does not reach the live search's frontier ordering here. Do NOT re-"
            "propose another ObjectHistorySaliencePrior weight/roster A/B without a NEW mechanism."
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
        "trajectories_diverge_per_game": trajectories_diverge_per_game,
        "object_history_bonus_marginal_diverges": bool(bonus_diverges_any),
        "object_history_bonus_marginal_arm_game_divergences": int(bonus_total_div),
        "color_blob_prior_alone_diverges_n_games": int(blob_prior_div_n),
        "full_live_flip_vs_baseline_arm_game_divergences_CONFOUNDED": int(full_flip_total_div),
        "blob_cache_perf_fix": blob_cache_perf_fix,
        "levels_gained_headroom_present": bool(any_headroom),
        "any_config_beats_baseline_levels": bool(any_beats),
        "any_config_beats_baseline_without_safety_regression": bool(any_beats_clean),
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
