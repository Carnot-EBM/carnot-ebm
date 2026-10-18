"""Experiment 5729: matched-budget A/B testing whether LOOSENING the
``GroundTruthValidatedFrameChangeScorer`` validation gate -- while still
providing real protection against a miscalibrated scorer -- recovers live-agent
search capability, per CLAUDE.md's Phase Prototype + Empirical Validation +
Adversarial Check Discipline.

WHY THIS IS NOT A RERUN OF EXP5590 OR EXP5728 (Failed-Experiment Rerun
Discipline). exp5590 fixed a dict-candidate CRASH bug in the CNN scorer (clean
null at the default weight). exp5728 swept the CNN blend WEIGHT 0.05->2.0 and
found ANOTHER clean null -- and its calibration instrumentation localized the
real blocker precisely: the ``GroundTruthValidatedFrameChangeScorer`` returns
0.0 upstream of every downstream multiply whenever it is NOT validated, and it
never validated post-run on 7/11 games (cd82, cn04, lp85, m0r0, sk48, su15,
wa30). This experiment changes NEITHER the bug NOR the weight; it changes the
GATE's validation criterion itself -- the variable exp5728's retire condition
explicitly named as "the right next lever".

ROOT CAUSE THE GATE CHANGE ADDRESSES. The shipped gate
(``arc_frame_change_predictor.GroundTruthValidatedFrameChangeScorer``) is
validated iff ``agreement_count >= required_agreements (=1) AND
contradiction_count == 0`` -- zero contradictions FOREVER, with a ``reset()``
that forwards to the inner scorer but never clears its OWN agreement/
contradiction counters. Because the dominant memory term has an observed floor
>= 0.94 (exp5728 per-game memory_term_stats.min), the gate's predicted_changed
is nearly always True, so any ordinary no-op action (blocked move, inert click
-- normal ARC play) trips one contradiction and PERMANENTLY disqualifies
validation for the rest of the game run. This is stricter than the gate's own
originating spec (REQ-ARC-FCP-5373: "at least one observed transition validates
it") -- the zero-tolerance-forever requirement is an implementation overreach.

WHAT IS TESTED (5 configs, same 11-game roster + same 200-action budget + same
CARNOT_ARC_DISABLE_INDUCTION=1 no-LLM harness as exp5590/exp5728, single-
threaded for determinism):

  1. baseline           -- the CURRENT shipped gate, unmodified (control).
  2. rate_0.7 (primary) -- replace zero-tolerance-forever with an agreement-RATE
                           threshold after a minimum observation count:
                           validated := observed_count >= min_observations (10)
                           AND agreement_count/observed_count >= 0.7.
  3. rate_0.6           -- same, threshold 0.6 (sensitivity, more permissive).
  4. rate_0.8           -- same, threshold 0.8 (sensitivity, stricter).
  5. reset_on_levelup   -- keep the hard zero-tolerance criterion, but make
                           reset(reset_to_prior=True) (the level-up path,
                           arc_competition_agent.py:1798) also clear the gate's
                           own agreement/contradiction/observed counters, so
                           each level gets a fresh validation window instead of
                           accumulating contradictions across the whole run.

PROTECTION IS PRESERVED (the gate is not deleted). The rate gate still blocks a
genuinely miscalibrated scorer: a scorer whose predicted-changed matches
observed-changed <= 50% of the time (exp5728: sk48 0.056, lp85 0.279, cn04
0.526) stays gated OFF even at threshold 0.6. The docstring rationale the gate
exists for -- "prevent an unvalidated frame-diff prior from choosing probes
solely because it is self-consistent" -- is retained; only the
zero-tolerance-forever overreach is relaxed.

INJECTION IS NON-INVASIVE (measurement task, not a live-stack change). This
script does NOT edit arc_frame_change_predictor.py. Per game it constructs the
real E3AgentPolicy, then swaps the class of the single shared gate object
(policy.explorer.frame_change_scorer -- the SAME object referenced by the
ranking path, the ActionEffectExpansionPrior frontier path, the
observe_transition accounting, and the reset path) to a config-specific
subclass via ``__class__`` reassignment, preserving object identity so every
reference sees the modified behavior. Whether to change the live default is
reported as a recommendation for the OPERATOR; the agent never self-authorizes
flipping the live-stack default.

Spec refs: REQ-ARC-FCP-5729, SCENARIO-ARC-FCP-5729-RATE-TOLERANCE-GATE,
SCENARIO-ARC-FCP-5729-RESET-ON-LEVELUP.
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

from carnot.agentic.arc_frame_change_predictor import (  # noqa: E402
    GroundTruthValidatedFrameChangeScorer,
)

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5729_gtv_gate_fix_ab"
RESULT_RELATIVE_PATH = "results/experiment_5729_gtv_gate_fix_ab.json"
SCHEMA = "carnot.exp5729.gtv_gate_fix_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5729
DEFAULT_BUDGET = 200
# min_observations=10: below ~10 observations the running agreement rate is too
# noisy to trust (one disagreement swings a 3-obs rate from 1.0 to 0.67), while
# 10 gives the rate 0.1 resolution and is reached in the first ~5% of a 200-
# action budget (exp5728 gtv_observed_count ~= 194/run), so the gate warms up
# briefly then decides on data instead of on the single first contradiction.
MIN_OBSERVATIONS = 10
BASELINE_CONFIG = "baseline"
# (name, kind, param) -- kind in {"baseline","rate","reset_on_levelup"}.
CONFIGS: tuple[tuple[str, str, float | None], ...] = (
    ("baseline", "baseline", None),
    ("rate_0.7", "rate", 0.7),
    ("rate_0.6", "rate", 0.6),
    ("rate_0.8", "rate", 0.8),
    ("reset_on_levelup", "reset_on_levelup", None),
)
# Same 11-game roster as exp5590/exp5728 for an apples-to-apples comparison.
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
    "configs_tested",
    "baseline_config",
    "min_observations",
    "gate_config_results",
    "per_config_game_rows",
    "safety_regression_check",
    "n_games_validated_post_run_by_config",
    "levels_gained_headroom_present",
    "any_config_beats_baseline_levels",
    "any_config_beats_baseline_without_safety_regression",
    "prior_work_extended",
    "recommendation",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; recovered capability, an inert loosening, and a safety regression are distinct real outcomes -- loosening the gate is only a win if it raises banked levels on the same roster+budget WITHOUT blowing up search"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_INDUCTION=1 guarantees no GGUF/LLM load, isolating the gate's effect on the search/frontier-priority path from tier-3 induction"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures a validation-gate criterion's effect on live-path search, not an executable win-check; the gated scorer is oracle-distinct perception"
    },
    "solve_provenance": {
        "principle": "development_proxy -- an offline-arcade gate-mechanism A/B on the dev twin; this does NOT fit the game-solve taxonomy (it is gate infrastructure work, not a self-discovery solve), no NEW level is banked (lp85 L1 is a pre-existing registry solve reached incidentally), offline_reproduced is deliberately NOT claimed"
    },
    "configs_tested": {
        "principle": "baseline control + rate-tolerance at three thresholds (sensitivity) + reset-on-level-up -- spanning two independent ways to relax the overreach lets the real lever show rather than a single guess"
    },
    "min_observations": {
        "principle": "the rate gate's warm-up floor -- below it the running agreement rate is statistically meaningless; documents the one free parameter of the rate configs so a reviewer can judge sensitivity"
    },
    "gate_config_results": {
        "principle": "per-config levels/states/efficiency totals, per-game level deltas vs baseline, and n_games_validated_post_run -- n_games_validated is the mechanism this experiment directly manipulates, levels is the capability answer"
    },
    "per_config_game_rows": {
        "principle": "full per-game rows (levels, states_expanded, gtv agreement/contradiction/validated, ranking-scorer nonzero returns) so a third party can re-derive every headline number and confirm the gate actually turned the scorer ON"
    },
    "safety_regression_check": {
        "principle": "a loosened gate that lets a miscalibrated scorer steer search can make search WORSE/noisier even when it occasionally banks a level -- this compares states_expanded per config vs baseline and flags any blow-up, so a level win is never reported without its search cost"
    },
    "n_games_validated_post_run_by_config": {
        "principle": "the direct mechanism readout -- how many of 11 games end with the gate validated per config; a gate change that does not move this cannot have changed the search, and one that moves it a lot but banks no levels localizes the failure to a later stage"
    },
    "levels_gained_headroom_present": {
        "principle": "FALSE_NEGATIVE_RISK discipline -- a no-delta result is only interpretable if some game shows nonzero levels somewhere, else the null may just mean the roster had no headroom for any gate"
    },
    "any_config_beats_baseline_levels": {
        "principle": "the raw capability boolean: True iff some config banks strictly more levels than baseline on the same roster"
    },
    "any_config_beats_baseline_without_safety_regression": {
        "principle": "the load-bearing recommendation boolean: a level win only justifies changing the live default if it does not come with a search-cost blow-up (the safety-regression guard)"
    },
    "prior_work_extended": {
        "principle": "Failed-Experiment Rerun Discipline -- names exp5590 AND exp5728 by id+verdict and states precisely what is different (the gate's tolerance criterion, not the bug and not the blend weight), with a retire condition"
    },
    "recommendation": {
        "principle": "reports whether to change the live default gate criterion; the agent never self-authorizes flipping the live-stack default -- that is operator-only"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
    "duration_s": {
        "principle": "real wall-clock of the search A/B; the no-LLM substrate floor is 0.01s and this runs sequentially over 5 configs x 11 games, so a plausible multi-second total is expected"
    },
}

PRIOR_WORK_EXTENDED = {
    "experiments_extended": [
        {
            "experiment_id": "exp5590 (experiment_5590_frame_change_cnn_dict_candidate_fix_ab)",
            "prior_verdict": "complete: dict_candidate_fix_honest_null_headroom_present_no_delta",
            "prior_finding": (
                "Fixed a dict-shaped-candidate CRASH bug (_as_action_like) that silently zeroed "
                "the CNN term; control (CNN forced zero) and treatment were byte-identical at the "
                "default weight. The fix stopped a silent bug but produced no capability delta."
            ),
        },
        {
            "experiment_id": "exp5728 (experiment_5728_cnn_weight_sweep)",
            "prior_verdict": (
                "complete: cnn_weight_sweep_headroom_present_weight_change_yields_same_levels"
            ),
            "prior_finding": (
                "Swept CNN blend weight 0.05->2.0 holding memory_weight=1.0: no weight banked more "
                "levels than baseline (1). Calibration localized the blocker to the "
                "GroundTruthValidatedFrameChangeScorer validation gate, which returns 0.0 upstream "
                "of the weight multiply whenever unvalidated -- 7/11 games never validated post-run "
                "-- so the scorer was rarely/never consulted while validated. Its retire condition "
                "named the GTV validation gate as the right next lever."
            ),
        },
    ],
    "what_is_different_here": (
        "This changes NEITHER the dict-candidate bug (exp5590, already fixed) NOR the CNN blend "
        "weight (exp5728, settled null). It changes the GATE's validation criterion itself: the "
        "zero-tolerance-forever (contradiction_count==0 forever, no counter reset) requirement is "
        "relaxed two independent ways -- an agreement-rate threshold after a min observation count, "
        "and a per-level-up counter reset -- while still gating OFF a genuinely miscalibrated "
        "scorer (<=50% agreement). This is exactly the lever exp5728's retire condition named."
    ),
    "retire_if_same_verdict": (
        "If NO gate config banks more levels than baseline AND n_games_validated_post_run rises "
        "substantially (proving the gate change DID turn the scorer on) with no level gain, then "
        "the scorer's SIGNAL -- not the gate that admits it -- is the blocker: retire further gate-"
        "criterion experiments and redirect at the scorer's discriminative quality (the CNN/memory "
        "perception itself), per the Missing-Verifier Gap Logging discipline."
    ),
}


# --------------------------------------------------------------------------- #
# Config-specific gate subclasses. Injected by ``__class__`` reassignment onto
# the live gate instance (identity preserved -> every reference updated). Each
# overrides ONLY the validation criterion (rate) or the reset semantics
# (reset_on_levelup); all other behavior (candidate_score gating, observe_
# transition accounting, as_dict) is inherited unchanged from the shipped class.
# --------------------------------------------------------------------------- #
class _RateToleranceGTV(GroundTruthValidatedFrameChangeScorer):
    """Validated iff observed_count >= min_observations AND agreement rate >= threshold.

    Instance attributes ``min_observations`` / ``agreement_rate_threshold`` are
    set after the ``__class__`` swap (the base __init__ is never re-run).
    """

    min_observations: int = MIN_OBSERVATIONS
    agreement_rate_threshold: float = 0.7

    @property
    def validated(self) -> bool:
        obs = int(self._observed_count)
        if obs < int(self.min_observations):
            return False
        return (float(self._agreement_count) / float(obs)) >= float(
            self.agreement_rate_threshold
        )


class _ResetOnLevelUpGTV(GroundTruthValidatedFrameChangeScorer):
    """Zero-tolerance criterion unchanged, but reset(reset_to_prior=True) clears
    this wrapper's OWN counters (the live level-up path), giving each level a
    fresh validation window instead of accumulating contradictions run-wide."""

    def reset(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("reset_to_prior"):
            self._observed_count = 0
            self._agreement_count = 0
            self._contradiction_count = 0
            self._last_observed_delta = None
        # Forward to the inner scorer exactly as the shipped base reset does.
        if hasattr(self.scorer, "reset"):
            try:
                self.scorer.reset(*args, **kwargs)
            except Exception:
                pass


def _apply_config(gtv: Any, kind: str, param: float | None) -> bool:
    """Mutate the live gate object in place for the given config. Returns True
    iff a modification was applied (baseline returns False -- no change)."""

    if gtv is None or not isinstance(gtv, GroundTruthValidatedFrameChangeScorer):
        return False
    if kind == "baseline":
        return False
    if kind == "rate":
        gtv.__class__ = _RateToleranceGTV
        gtv.min_observations = int(MIN_OBSERVATIONS)
        gtv.agreement_rate_threshold = float(param if param is not None else 0.7)
        return True
    if kind == "reset_on_levelup":
        gtv.__class__ = _ResetOnLevelUpGTV
        return True
    return False


def _gate_self_test() -> JsonDict:
    """Adversarial check (Phase Prototype discipline): prove the injection
    subclasses actually change validation before trusting the sweep."""

    class _Dummy:
        def candidate_score(self, frame: Any, cand: Any) -> float:
            return 0.5

        def reset(self, *a: Any, **k: Any) -> None:
            pass

    ok = True
    # Base: one contradiction disqualifies forever.
    g = GroundTruthValidatedFrameChangeScorer(_Dummy())
    g._observed_count, g._agreement_count, g._contradiction_count = 20, 19, 1
    ok = ok and (g.validated is False)
    # Rate 0.7: 0.95 agreement despite a contradiction -> validated.
    _apply_config(g, "rate", 0.7)
    ok = ok and (g.validated is True)
    g._observed_count, g._agreement_count = 5, 5  # below warm-up floor
    ok = ok and (g.validated is False)
    g._observed_count, g._agreement_count = 20, 12  # 0.6 < 0.7
    ok = ok and (g.validated is False)
    # Reset-on-level-up: clears counters on reset_to_prior=True only.
    g2 = GroundTruthValidatedFrameChangeScorer(_Dummy())
    g2._observed_count, g2._agreement_count, g2._contradiction_count = 50, 30, 20
    _apply_config(g2, "reset_on_levelup", None)
    ok = ok and (g2.validated is False)
    g2.reset(level=1, reset_to_prior=True)
    ok = ok and (g2._observed_count == 0 and g2._contradiction_count == 0)
    g2._observed_count, g2._agreement_count, g2._contradiction_count = 1, 1, 0
    ok = ok and (g2.validated is True)
    # reset() WITHOUT reset_to_prior must NOT clear counters.
    g2._observed_count, g2._agreement_count, g2._contradiction_count = 9, 4, 5
    g2.reset()
    ok = ok and (g2._observed_count == 9 and g2._contradiction_count == 5)
    return {"gate_injection_self_test_passed": bool(ok)}


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
    # The CNN checkpoint must exist, else the gated scorer's CNN term is absent.
    checks["cnn_checkpoint_present"] = (
        root / "results" / "experiment_4629_live_frame_change_cnn.pt"
    ).exists()
    # The gate injection subclasses must actually change validation behavior.
    checks.update(_gate_self_test())
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _play_one_game(
    game: str, *, config_name: str, kind: str, param: float | None, budget: int, game_index: int
) -> JsonDict:
    """Run one game under one gate config, returning a compact row + gate diagnostics."""

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    # Per-game fixed seed so baseline-game-X and config-game-X start from an
    # identical RNG state -- any delta isolates the gate change, not RNG drift.
    _seed_everything(RANDOM_SEED + game_index)

    policy = E3AgentPolicy(game)
    gtv = getattr(policy.explorer, "frame_change_scorer", None)
    applied = _apply_config(gtv, kind, param)

    # Count RANKING consults that return nonzero -- the moments the gate admitted
    # the scorer to the frontier order (the direct readout that a loosened gate
    # turned the scorer ON). observe_transition accounting is NOT counted here.
    ranking = {"calls": 0, "nonzero": 0}
    if gtv is not None and hasattr(gtv, "candidate_score"):
        _orig_cs = gtv.candidate_score

        def _counting_candidate_score(frame: Any, candidate: Any, _o=_orig_cs, _r=ranking) -> float:
            val = float(_o(frame, candidate))
            _r["calls"] += 1
            if val != 0.0:
                _r["nonzero"] += 1
            return val

        gtv.candidate_score = _counting_candidate_score  # type: ignore[method-assign]

    row = lb.run_game(game, policy, budget=budget)
    gtv_diag = gtv.as_dict() if (gtv is not None and hasattr(gtv, "as_dict")) else {}

    return {
        "game": game,
        "config": config_name,
        "gate_config_applied": bool(applied),
        "scorer_present": bool(gtv is not None),
        "levels": int(row.get("levels", 0)),
        "reached": int(row.get("reached", 0)),
        "actions": int(row.get("actions", 0)),
        "states_expanded": int(len(policy.explorer.graph)),
        "efficiency": float(row.get("efficiency", 0.0) or 0.0),
        "actions_to_first_levelup": row.get("actions_to_first_levelup"),
        "gap_signature": (row.get("gap") or {}).get("signature") if row.get("gap") else None,
        "ranking_scorer_calls": int(ranking["calls"]),
        "ranking_scorer_nonzero_returns": int(ranking["nonzero"]),
        "gtv_observed_count": int(gtv_diag.get("observed_count", 0)),
        "gtv_agreement_count": int(gtv_diag.get("agreement_count", 0)),
        "gtv_contradiction_count": int(gtv_diag.get("contradiction_count", 0)),
        "gtv_validated_post_run": bool(gtv_diag.get("frame_diff_ground_truth_validated", False)),
    }


def run_sweep(
    roster: tuple[str, ...],
    configs: tuple[tuple[str, str, float | None], ...],
    *,
    budget: int,
) -> dict[str, dict[str, JsonDict]]:
    """Sequential single-threaded sweep: {config_name: {game: row}}."""

    out: dict[str, dict[str, JsonDict]] = {}
    for config_name, kind, param in configs:
        per_game: dict[str, JsonDict] = {}
        for game_index, game in enumerate(roster):
            per_game[game] = _play_one_game(
                game,
                config_name=config_name,
                kind=kind,
                param=param,
                budget=budget,
                game_index=game_index,
            )
        out[config_name] = per_game
    return out


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    configs: tuple[tuple[str, str, float | None], ...] = CONFIGS,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
    config_names = [c[0] for c in configs]
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
            "configs_tested": config_names,
            "baseline_config": BASELINE_CONFIG,
            "min_observations": int(MIN_OBSERVATIONS),
            "gate_config_results": [],
            "per_config_game_rows": {},
            "safety_regression_check": {},
            "n_games_validated_post_run_by_config": {},
            "levels_gained_headroom_present": False,
            "any_config_beats_baseline_levels": False,
            "any_config_beats_baseline_without_safety_regression": False,
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

    sweep = run_sweep(roster, configs, budget=budget)

    baseline_rows = sweep[BASELINE_CONFIG]
    baseline_levels_total = sum(r["levels"] for r in baseline_rows.values())
    baseline_states_total = sum(r["states_expanded"] for r in baseline_rows.values())

    gate_config_results: list[JsonDict] = []
    per_config_game_rows: JsonDict = {}
    n_validated_by_config: JsonDict = {}
    any_headroom = False
    for config_name, _kind, _param in configs:
        rows = sweep[config_name]
        per_config_game_rows[config_name] = rows
        levels_total = sum(r["levels"] for r in rows.values())
        states_total = sum(r["states_expanded"] for r in rows.values())
        eff_sum = round(sum(r["efficiency"] for r in rows.values()), 4)
        ranking_nonzero_total = sum(r["ranking_scorer_nonzero_returns"] for r in rows.values())
        n_validated = sum(1 for r in rows.values() if r["gtv_validated_post_run"])
        games_validated = sorted(g for g in roster if rows[g]["gtv_validated_post_run"])
        per_game_levels = {g: rows[g]["levels"] for g in roster}
        per_game_levels_delta = {g: rows[g]["levels"] - baseline_rows[g]["levels"] for g in roster}
        per_game_validated_at_end = {g: bool(rows[g]["gtv_validated_post_run"]) for g in roster}
        if any(r["levels"] > 0 for r in rows.values()):
            any_headroom = True
        n_validated_by_config[config_name] = int(n_validated)
        gate_config_results.append(
            {
                "config": config_name,
                "is_baseline": config_name == BASELINE_CONFIG,
                "levels_gained_total": int(levels_total),
                "levels_delta_vs_baseline": int(levels_total - baseline_levels_total),
                "states_expanded_total": int(states_total),
                "states_delta_vs_baseline": int(states_total - baseline_states_total),
                "efficiency_sum": float(eff_sum),
                "n_games_validated_post_run": int(n_validated),
                "games_validated_post_run": games_validated,
                "ranking_scorer_nonzero_returns_total": int(ranking_nonzero_total),
                "per_game_levels": per_game_levels,
                "per_game_levels_delta_vs_baseline": per_game_levels_delta,
                "per_game_validated_at_end": per_game_validated_at_end,
            }
        )

    # Safety-regression check: a loosened gate that lets a miscalibrated scorer
    # steer search can inflate states_expanded (noisier search) even without a
    # level change. Flag any config whose states blow up materially vs baseline.
    # 20% relative growth is the threshold for "material" -- below that is search
    # jitter, above it the gate change is making the search meaningfully worse.
    STATES_REGRESSION_REL = 0.20
    per_config_safety: dict[str, JsonDict] = {}
    for res in gate_config_results:
        cfg = res["config"]
        states = int(res["states_expanded_total"])
        delta = int(res["states_delta_vs_baseline"])
        rel = (delta / baseline_states_total) if baseline_states_total else 0.0
        regression = bool(cfg != BASELINE_CONFIG and rel > STATES_REGRESSION_REL)
        per_config_safety[cfg] = {
            "states_expanded_total": states,
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
            "states_expanded is the search cost; a gate loosening that banks a level while "
            "materially inflating states may be luck-under-noise, not a real capability gain. A "
            "config only earns a live-default recommendation if it raises levels WITHOUT tripping "
            "states_expanded_regression."
        ),
    }

    # Capability booleans.
    best = max(gate_config_results, key=lambda r: r["levels_gained_total"])
    best_levels = int(best["levels_gained_total"])
    best_config = str(best["config"])
    any_beats = best_levels > baseline_levels_total
    beating_configs_clean = [
        r["config"]
        for r in gate_config_results
        if r["config"] != BASELINE_CONFIG
        and r["levels_gained_total"] > baseline_levels_total
        and not per_config_safety[r["config"]]["states_expanded_regression"]
    ]
    any_beats_clean = bool(beating_configs_clean)

    # Did the gate change DO anything mechanically (turn the scorer on more)?
    baseline_validated = n_validated_by_config[BASELINE_CONFIG]
    max_validated = max(n_validated_by_config.values())
    gate_moved_validation = bool(max_validated > baseline_validated)

    if any_beats_clean:
        verdict = (
            f"complete: gtv_gate_loosening_recovers_capability_{baseline_levels_total}_to_"
            f"{best_levels}_levels_at_{best_config}_no_safety_regression"
        )
    elif any_beats:
        verdict = (
            f"complete: gtv_gate_loosening_raises_levels_to_{best_levels}_at_{best_config}_"
            "but_with_states_expanded_safety_regression"
        )
    elif gate_moved_validation:
        verdict = (
            "complete: gtv_gate_loosening_turns_scorer_on_"
            f"{baseline_validated}_to_{max_validated}_of_{len(roster)}_games_validated_"
            "but_no_level_gain_scorer_signal_is_the_blocker_not_the_gate"
        )
    elif any_headroom:
        verdict = "complete: gtv_gate_loosening_inert_headroom_present_no_validation_or_level_change"
    else:
        verdict = "complete: gtv_gate_loosening_zero_levels_across_entire_roster"

    if any_beats_clean:
        recommendation = (
            f"Gate config(s) {beating_configs_clean} bank {best_levels} levels vs baseline's "
            f"{baseline_levels_total} on the same roster+budget WITHOUT a states_expanded safety "
            f"regression. This is a candidate live-default change to "
            "GroundTruthValidatedFrameChangeScorer's validation criterion; the operator decides "
            "whether to flip the live-stack default (not self-authorized). Recommend the operator "
            "review the per-game level deltas and the safety-regression table before changing the "
            "shipped gate."
        )
    elif any_beats:
        recommendation = (
            f"Config {best_config} banks more levels ({best_levels} vs {baseline_levels_total}) but "
            "trips the states_expanded safety regression (search cost blew up). Do NOT change the "
            "live default on this basis -- the apparent gain may be luck under a noisier search. "
            "Operator-only whether to investigate further with more seeds."
        )
    elif gate_moved_validation:
        recommendation = (
            "NO gate config banks more levels than baseline "
            f"({baseline_levels_total}), BUT the loosening DID turn the scorer on "
            f"(games-validated-post-run rose {baseline_validated} -> {max_validated} of "
            f"{len(roster)}). This localizes the blocker DOWNSTREAM of the gate: the scorer's own "
            "signal (CNN/memory perception), not the gate that admits it, is now the limiting "
            "factor. Do NOT change the live default gate on a capability basis (no level gain); the "
            "right next lever is the scorer's discriminative quality, per exp5728's retire "
            "condition and Missing-Verifier Gap Logging. Operator-only whether to act."
        )
    else:
        recommendation = (
            "NO gate config banks more levels than baseline AND the loosening did not materially "
            "raise games-validated-post-run. The gate criterion is not the active lever on this "
            "roster+budget. Do NOT change the live default. Operator-only whether to act."
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
        "configs_tested": config_names,
        "baseline_config": BASELINE_CONFIG,
        "min_observations": int(MIN_OBSERVATIONS),
        "baseline_levels_total": int(baseline_levels_total),
        "baseline_states_expanded_total": int(baseline_states_total),
        "best_config": best_config,
        "best_levels_total": best_levels,
        "gate_config_results": gate_config_results,
        "per_config_game_rows": per_config_game_rows,
        "safety_regression_check": safety_regression_check,
        "n_games_validated_post_run_by_config": n_validated_by_config,
        "gate_change_moved_validation": bool(gate_moved_validation),
        "levels_gained_headroom_present": bool(any_headroom),
        "any_config_beats_baseline_levels": bool(any_beats),
        "any_config_beats_baseline_without_safety_regression": bool(any_beats_clean),
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
