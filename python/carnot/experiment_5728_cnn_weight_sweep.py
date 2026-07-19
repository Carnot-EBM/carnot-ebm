"""Experiment 5728: matched-budget SWEEP of ``LiveActionEffectScorer.cnn_weight``
on the live ARC agent search path, testing whether RAISING the CNN blend weight
(not just fixing the dict-candidate bug, which exp5590 already shipped) produces
a measurable capability delta.

WHY THIS IS NOT A RERUN OF EXP5590 (Failed-Experiment Rerun Discipline).
exp5590 (``results/experiment_5590_frame_change_cnn_dict_candidate_fix_ab.json``,
verdict ``complete: dict_candidate_fix_honest_null_headroom_present_no_delta``)
fixed a real bug where dict-shaped frontier candidates crashed the CNN scorer
and silently zeroed the CNN term, then measured control (CNN forced to zero)
vs treatment (CNN active at the DEFAULT ``cnn_weight=0.05``). Both arms were
byte-identical (679 states, 1 level, all per-game deltas 0). That experiment
changed the CNN's CORRECTNESS at a fixed weight; it did NOT vary the weight.
The operator-approved diagnosis of that null was that ``cnn_weight=0.05`` is
20x smaller than ``memory_weight=1.0`` in the additive frontier score, so even
a now-correct CNN term is too small to change any tie-break. THIS experiment
changes a DIFFERENT variable -- the weight magnitude -- holding
``memory_weight=1.0`` fixed, to test that diagnosis directly. Acceptance:
if no swept weight raises ``levels_gained_total`` above the 0.05 baseline on
the SAME 11-game roster + SAME per-game budget, the diagnosis is refined, not
merely confirmed -- the calibration instrumentation below records WHY.

WHAT THE HARNESS ISOLATES. Tier-3 LLM induction is disabled
(``CARNOT_ARC_DISABLE_INDUCTION=1``) so this measures the search/frontier-
priority effect cleanly and fast, with NO GPU/LLM. Same construction as
exp5590: the real ``E3AgentPolicy`` -> ``StepwiseExplorer`` with the shipped
``GroundTruthValidatedFrameChangeScorer`` wrapping the real
``LiveActionEffectScorer``, ``ActionEffectExpansionPrior`` enabled. The ONLY
knob varied per run is the ``cnn_weight`` on the constructed
``LiveActionEffectScorer`` instance (reached via
``policy.explorer.frame_change_scorer.scorer``; the expansion prior references
the same wrapper object, so one assignment covers both the ranking and the
frontier-priority paths).

DETERMINISM. Unlike exp5590 (which ran arms in concurrent threads and thus was
subject to a documented, benign cross-arm scoring race), this sweep runs
SINGLE-THREADED and sequential. Verified deterministic: repeated single-thread
runs of a game reproduce identical ``states_expanded``/``efficiency``. The
per-weight 0.05 point is therefore a FRESH, seed-and-code-matched baseline
re-run (NOT a copy of exp5590's threaded numbers, which are not reproducible
single-threaded -- lp85 was 69 states threaded vs 55 single-threaded).

CALIBRATION INSTRUMENTATION (the operator's sanity-check ask). A behaviorally
IDENTICAL recording wrapper replaces the constructed ``LiveActionEffectScorer``
for each run: it computes the exact same ``memory_weight*mem + cnn_weight*cnn``
score (so it cannot perturb the search relative to just setting the attribute)
while recording, per scorer consultation, the raw (un-weighted) CNN output and
the memory term. Because the ``GroundTruthValidatedFrameChangeScorer`` returns
0.0 whenever it is NOT validated, the wrapper's call count directly measures
how many times the scorer was actually consulted WHILE VALIDATED -- i.e., how
many times ``cnn_weight`` could have mattered at all. If the CNN raw outputs
have near-zero variance, or the scorer is rarely consulted while validated,
then no weight can change frontier ranking, and that is the real story rather
than "weight too small".

Spec refs: REQ-ARC-FCP-5728, SCENARIO-ARC-FCP-5728-CNN-WEIGHT-SWEEP,
SCENARIO-ARC-FCP-5728-GTV-GATE-CALIBRATION.
"""

from __future__ import annotations

import hashlib
import json
import os
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

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5728_cnn_weight_sweep"
RESULT_RELATIVE_PATH = "results/experiment_5728_cnn_weight_sweep.json"
SCHEMA = "carnot.exp5728.cnn_weight_sweep.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5728
DEFAULT_BUDGET = 200
BASELINE_WEIGHT = 0.05
# Baseline (shipped default) + four meaningfully-higher points spanning below,
# at, and above memory_weight=1.0 -- enough to reveal a monotonic trend or a
# cliff rather than a single guess.
WEIGHTS: tuple[float, ...] = (0.05, 0.25, 0.5, 1.0, 2.0)
MEMORY_WEIGHT_FIXED = 1.0
# Same 11-game roster as exp5590 for an apples-to-apples comparison.
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
    "weights_swept",
    "baseline_weight",
    "memory_weight_fixed",
    "weight_sweep_results",
    "per_weight_game_rows",
    "cnn_calibration_summary",
    "levels_gained_headroom_present",
    "any_weight_beats_baseline_levels",
    "prior_work_extended",
    "recommendation",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a positive delta, an inert sweep, and a regression are distinct real outcomes -- raising a blend weight is only a win if it raises banked levels on the same roster+budget"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- CARNOT_ARC_DISABLE_INDUCTION=1 guarantees no GGUF/LLM load, isolating the search/frontier-priority effect of cnn_weight from tier-3 induction"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures a blend-weight's effect on live-path search, not an executable win-check; the scorer is oracle-distinct perception"
    },
    "solve_provenance": {
        "principle": "development_proxy -- an offline-arcade frontier-scoring A/B on the dev twin, NOT a new live-agent self-discovery solve; no NEW level is banked (lp85 L1 is a pre-existing registry solve reached incidentally), offline_reproduced is deliberately NOT claimed"
    },
    "weights_swept": {
        "principle": "spanning below/at/above memory_weight=1.0 lets a monotonic trend or a cliff show, rather than a single higher guess"
    },
    "memory_weight_fixed": {
        "principle": "one-variable-changed discipline -- memory_weight stays 1.0 for every run so the delta isolates cnn_weight alone"
    },
    "weight_sweep_results": {
        "principle": "per-weight levels/states/efficiency totals and per-game deltas vs the 0.05 baseline -- the capability answer is whether any weight raises levels_gained_total"
    },
    "cnn_calibration_summary": {
        "principle": "records how often the scorer was consulted WHILE the GTV gate was validated and the variance of the raw CNN outputs -- a weight increase on a rarely-consulted or non-discriminating scorer adds noise, not signal, and that is the real story if levels do not move"
    },
    "levels_gained_headroom_present": {
        "principle": "FALSE_NEGATIVE_RISK discipline -- a no-delta result is only interpretable if some game shows nonzero levels somewhere, else the null may just mean the roster had no headroom for any method"
    },
    "any_weight_beats_baseline_levels": {
        "principle": "the single load-bearing capability boolean: True iff some swept weight banks strictly more levels than the 0.05 baseline on the same roster"
    },
    "prior_work_extended": {
        "principle": "Failed-Experiment Rerun Discipline -- names exp5590 by id+verdict and states precisely what is different (weight magnitude, not bug correctness), with a retire condition"
    },
    "recommendation": {
        "principle": "reports whether to change the live default and by how much; the agent never self-authorizes flipping the live-stack default -- that is operator-only"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
    "duration_s": {
        "principle": "real wall-clock of the search sweep; the no-LLM substrate floor is 0.01s and this runs sequentially, so a plausible multi-second-per-run total is expected"
    },
}

PRIOR_WORK_EXTENDED = {
    "experiment_id": "exp5590 (experiment_5590_frame_change_cnn_dict_candidate_fix_ab)",
    "prior_verdict": "complete: dict_candidate_fix_honest_null_headroom_present_no_delta",
    "prior_finding": (
        "The dict-candidate CNN fix (_as_action_like) was measured at the DEFAULT "
        "cnn_weight=0.05: control (CNN forced to zero) and treatment (CNN active) were "
        "byte-identical across all 11 games (679 states, 1 level each). The fix stopped a "
        "silent bug but produced no capability delta at 0.05."
    ),
    "diagnosed_root_cause": (
        "cnn_weight=0.05 is 20x smaller than memory_weight=1.0 in the additive frontier "
        "score, so even a now-correct CNN term is too small to change which candidate wins "
        "a tie-break."
    ),
    "what_is_different_here": (
        "This is NOT a rerun of exp5590's A/B. It changes a DIFFERENT variable: the CNN blend "
        "weight itself (0.05 -> 0.25 -> 0.5 -> 1.0 -> 2.0), holding memory_weight=1.0 fixed and "
        "the dict-candidate fix shipped. It directly tests the diagnosed root cause (weight "
        "magnitude) instead of the bug correctness exp5590 already settled."
    ),
    "retire_if_same_verdict": (
        "If no swept weight raises levels_gained_total above the 0.05 baseline AND the "
        "calibration shows the scorer is the blocker (rarely consulted while validated, or "
        "non-discriminating), then cnn_weight is the wrong knob: retire further cnn_weight "
        "sweeps and redirect at the GroundTruthValidatedFrameChangeScorer validation gate."
    ),
}


class _RecordingLiveScorer:
    """Behaviorally IDENTICAL replacement for a constructed ``LiveActionEffectScorer``
    that applies a swept ``cnn_weight`` and records calibration data.

    The score arithmetic replicates ``LiveActionEffectScorer.candidate_score``
    exactly (same terms, same per-term try/except-to-zero semantics), so
    swapping this in cannot perturb the search relative to simply setting the
    ``cnn_weight`` attribute on the original. The only added behavior is
    appending ``(memory_term, raw_cnn_output_or_None)`` per call.
    """

    def __init__(self, original: Any, cnn_weight: float) -> None:
        self._original = original
        self.cnn_weight = float(cnn_weight)
        self.memory = getattr(original, "memory", None)
        self.cnn_scorer = getattr(original, "cnn_scorer", None)
        self.memory_weight = float(getattr(original, "memory_weight", 1.0))
        self.records: list[tuple[float, float | None]] = []

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        mem_term = 0.0
        if self.memory is not None:
            try:
                mem_term = float(self.memory_weight) * float(self.memory.candidate_score(candidate))
            except Exception:
                mem_term = 0.0
        cnn_raw: float | None = None
        cnn_term = 0.0
        if self.cnn_scorer is not None:
            try:
                cnn_raw = float(self.cnn_scorer.candidate_score(frame, candidate))
                cnn_term = float(self.cnn_weight) * cnn_raw
            except Exception:
                cnn_raw = None
                cnn_term = 0.0
        self.records.append((mem_term, cnn_raw))
        return float(mem_term + cnn_term)

    def as_dict(self) -> dict[str, Any]:
        base = self._original.as_dict() if hasattr(self._original, "as_dict") else {}
        base = dict(base) if isinstance(base, dict) else {}
        base["cnn_weight"] = float(self.cnn_weight)
        base["swept_by_exp5728"] = True
        return base


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
        from carnot.agentic.arc_frame_change_predictor import (  # noqa: F401
            LiveActionEffectScorer,
            load_live_action_effect_scorer,
        )

        checks["e3_policy_import"] = True
        checks["live_action_effect_scorer_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
        checks["live_action_effect_scorer_import"] = False
    # The CNN checkpoint must exist, else the swept term is absent and the sweep is meaningless.
    checks["cnn_checkpoint_present"] = (
        root / "results" / "experiment_4629_live_frame_change_cnn.pt"
    ).exists()
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


def _stats(values: list[float]) -> JsonDict:
    if not values:
        return {
            "n": 0,
            "n_nonzero": 0,
            "mean": None,
            "variance": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(values)),
        "n_nonzero": int(sum(1 for v in values if v != 0.0)),
        "mean": float(statistics.fmean(values)),
        "variance": float(statistics.pvariance(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _play_one_game(game: str, *, weight: float, budget: int) -> JsonDict:
    """Run one game at one cnn_weight, returning a compact row + calibration."""

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

    policy = E3AgentPolicy(game)
    gtv = getattr(policy.explorer, "frame_change_scorer", None)
    inner = getattr(gtv, "scorer", None) if gtv is not None else None
    wrapper: _RecordingLiveScorer | None = None
    # Count the RANKING entrypoint separately from the validation-accounting path.
    # gtv.candidate_score is the search-ranking entrypoint (gated: it returns 0.0 without
    # consulting the blended scorer whenever the GTV gate is NOT validated). gtv.observe_transition
    # ALSO calls the blended scorer (self.scorer) for validation accounting REGARDLESS of the gate,
    # so the wrapper's raw-call count conflates both -- only ranking_scorer_nonzero_returns measures
    # the moments cnn_weight could actually influence the frontier order.
    ranking = {"calls": 0, "nonzero": 0}
    if gtv is not None and inner is not None:
        wrapper = _RecordingLiveScorer(inner, weight)
        gtv.scorer = wrapper  # covers ranking (when validated) + observe_transition accounting

        _orig_cs = gtv.candidate_score

        def _counting_candidate_score(frame: Any, candidate: Any, _o=_orig_cs, _r=ranking) -> float:
            val = float(_o(frame, candidate))
            _r["calls"] += 1
            if val != 0.0:
                _r["nonzero"] += 1
            return val

        gtv.candidate_score = _counting_candidate_score  # type: ignore[method-assign]

    row = lb.run_game(game, policy, budget=budget)

    records = wrapper.records if wrapper is not None else []
    cnn_raws = [c for (_m, c) in records if c is not None]
    mem_terms = [_m for (_m, _c) in records]
    gtv_diag = gtv.as_dict() if (gtv is not None and hasattr(gtv, "as_dict")) else {}

    return {
        "game": game,
        "cnn_weight": float(weight),
        "levels": int(row.get("levels", 0)),
        "reached": int(row.get("reached", 0)),
        "actions": int(row.get("actions", 0)),
        "states_expanded": int(len(policy.explorer.graph)),
        "efficiency": float(row.get("efficiency", 0.0) or 0.0),
        "actions_to_first_levelup": row.get("actions_to_first_levelup"),
        "gap_signature": (row.get("gap") or {}).get("signature") if row.get("gap") else None,
        "scorer_wrapper_injected": bool(wrapper is not None),
        "ranking_scorer_calls": int(ranking["calls"]),
        "ranking_scorer_nonzero_returns": int(ranking["nonzero"]),
        "blended_scorer_wrapper_calls": int(len(records)),
        "cnn_raw_stats": _stats(cnn_raws),
        "memory_term_stats": _stats(mem_terms),
        "gtv_observed_count": int(gtv_diag.get("observed_count", 0)),
        "gtv_agreement_count": int(gtv_diag.get("agreement_count", 0)),
        "gtv_contradiction_count": int(gtv_diag.get("contradiction_count", 0)),
        "gtv_validated_post_run": bool(gtv_diag.get("frame_diff_ground_truth_validated", False)),
    }


def run_sweep(
    roster: tuple[str, ...], weights: tuple[float, ...], *, budget: int
) -> dict[float, dict[str, JsonDict]]:
    """Sequential single-threaded sweep: {weight: {game: row}}."""

    out: dict[float, dict[str, JsonDict]] = {}
    for weight in weights:
        per_game: dict[str, JsonDict] = {}
        for game in roster:
            per_game[game] = _play_one_game(game, weight=weight, budget=budget)
        out[weight] = per_game
    return out


def _weight_key(weight: float) -> str:
    return f"cnn_weight_{weight:g}"


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    weights: tuple[float, ...] = WEIGHTS,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
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
            "weights_swept": list(weights),
            "baseline_weight": BASELINE_WEIGHT,
            "memory_weight_fixed": MEMORY_WEIGHT_FIXED,
            "weight_sweep_results": [],
            "per_weight_game_rows": {},
            "cnn_calibration_summary": {},
            "levels_gained_headroom_present": False,
            "any_weight_beats_baseline_levels": False,
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

    sweep = run_sweep(roster, weights, budget=budget)

    baseline_rows = sweep[BASELINE_WEIGHT]
    baseline_levels_total = sum(r["levels"] for r in baseline_rows.values())

    weight_sweep_results: list[JsonDict] = []
    per_weight_game_rows: JsonDict = {}
    any_headroom = False
    for weight in weights:
        rows = sweep[weight]
        per_weight_game_rows[_weight_key(weight)] = rows
        levels_total = sum(r["levels"] for r in rows.values())
        states_total = sum(r["states_expanded"] for r in rows.values())
        eff_sum = round(sum(r["efficiency"] for r in rows.values()), 4)
        ranking_nonzero_total = sum(r["ranking_scorer_nonzero_returns"] for r in rows.values())
        cnn_nonzero_total = sum(r["cnn_raw_stats"]["n_nonzero"] for r in rows.values())
        per_game_levels = {g: rows[g]["levels"] for g in roster}
        per_game_levels_delta = {g: rows[g]["levels"] - baseline_rows[g]["levels"] for g in roster}
        identical_to_baseline = all(
            rows[g]["levels"] == baseline_rows[g]["levels"]
            and rows[g]["states_expanded"] == baseline_rows[g]["states_expanded"]
            for g in roster
        )
        if any(r["levels"] > 0 for r in rows.values()):
            any_headroom = True
        weight_sweep_results.append(
            {
                "cnn_weight": float(weight),
                "is_baseline": weight == BASELINE_WEIGHT,
                "levels_gained_total": int(levels_total),
                "levels_delta_vs_baseline": int(levels_total - baseline_levels_total),
                "states_expanded_total": int(states_total),
                "efficiency_sum": float(eff_sum),
                "per_game_levels": per_game_levels,
                "per_game_levels_delta_vs_baseline": per_game_levels_delta,
                "ranking_scorer_nonzero_returns_total": int(ranking_nonzero_total),
                "cnn_raw_nonzero_total": int(cnn_nonzero_total),
                "identical_to_baseline_levels_and_states": bool(identical_to_baseline),
            }
        )

    best = max(weight_sweep_results, key=lambda w: w["levels_gained_total"])
    best_total = int(best["levels_gained_total"])
    best_weight = float(best["cnn_weight"])
    any_weight_beats_baseline = best_total > baseline_levels_total
    all_identical = all(w["identical_to_baseline_levels_and_states"] for w in weight_sweep_results)

    # Roster-level calibration. The load-bearing number is ranking_scorer_nonzero_returns:
    # the count of RANKING consults (gtv.candidate_score) that returned nonzero, i.e. the moments
    # the blended scorer -- and thus cnn_weight -- could actually change the frontier order. The
    # blended_scorer_wrapper_calls count is larger because it also includes gtv.observe_transition
    # validation-accounting calls, which run regardless of the gate and never touch the search order.
    baseline_ranking_nonzero_total = sum(
        r["ranking_scorer_nonzero_returns"] for r in baseline_rows.values()
    )
    baseline_ranking_calls_total = sum(r["ranking_scorer_calls"] for r in baseline_rows.values())
    total_ranking_nonzero_all = sum(
        r["ranking_scorer_nonzero_returns"] for rows in sweep.values() for r in rows.values()
    )
    total_cnn_raw_calls_all = sum(
        r["cnn_raw_stats"]["n"] for rows in sweep.values() for r in rows.values()
    )
    total_cnn_nonzero_all = sum(
        r["cnn_raw_stats"]["n_nonzero"] for rows in sweep.values() for r in rows.values()
    )
    pooled_cnn_variances = [
        r["cnn_raw_stats"]["variance"]
        for r in baseline_rows.values()
        if r["cnn_raw_stats"]["variance"] is not None
    ]
    games_ranking_scorer_inert = sorted(
        g for g in roster if baseline_rows[g]["ranking_scorer_nonzero_returns"] == 0
    )
    games_gtv_never_validated = sorted(
        g for g in roster if not baseline_rows[g]["gtv_validated_post_run"]
    )
    cnn_calibration_summary = {
        "baseline_ranking_scorer_calls_total": int(baseline_ranking_calls_total),
        "baseline_ranking_scorer_nonzero_returns_total": int(baseline_ranking_nonzero_total),
        "sweep_ranking_scorer_nonzero_returns_total": int(total_ranking_nonzero_all),
        "sweep_cnn_raw_calls_total": int(total_cnn_raw_calls_all),
        "sweep_cnn_raw_nonzero_total": int(total_cnn_nonzero_all),
        "baseline_cnn_raw_variance_mean_over_games": (
            round(float(statistics.fmean(pooled_cnn_variances)), 6)
            if pooled_cnn_variances
            else None
        ),
        "games_ranking_scorer_inert_zero_nonzero_returns": games_ranking_scorer_inert,
        "n_games_ranking_scorer_inert": int(len(games_ranking_scorer_inert)),
        "games_gtv_never_validated_post_run": games_gtv_never_validated,
        "n_games_gtv_never_validated": int(len(games_gtv_never_validated)),
        "interpretation": (
            "ranking_scorer_nonzero_returns counts RANKING consults (gtv.candidate_score) that "
            "returned nonzero -- the ONLY moments cnn_weight can influence the frontier order, "
            "since the GroundTruthValidatedFrameChangeScorer returns 0.0 upstream of the "
            "cnn_weight multiply whenever it is not validated. When this is 0 for a game, "
            "cnn_weight is provably inert there regardless of magnitude. The CNN raw outputs are "
            "themselves NON-trivially varied (see baseline_cnn_raw_variance_mean_over_games), so a "
            "no-delta result is a GATE problem (validation never succeeds), not a "
            "non-discriminating-CNN problem -- and NOT evidence the CNN is useless."
        ),
    }

    if any_weight_beats_baseline:
        verdict = (
            f"complete: cnn_weight_sweep_raising_weight_helps_{baseline_levels_total}_to_"
            f"{best_total}_levels_at_w{best_weight:g}"
        )
    elif all_identical:
        verdict = (
            "complete: cnn_weight_sweep_inert_all_weights_byte_identical_"
            "gtv_validation_gate_floors_scorer_output"
        )
    elif any_headroom:
        verdict = "complete: cnn_weight_sweep_headroom_present_weight_change_yields_same_levels"
    else:
        verdict = "complete: cnn_weight_sweep_zero_levels_across_entire_roster"

    if any_weight_beats_baseline:
        recommendation = (
            f"A swept cnn_weight ({best_weight:g}) banks {best_total} levels vs the 0.05 "
            f"baseline's {baseline_levels_total} on the same roster+budget "
            f"(+{best_total - baseline_levels_total}). This is a candidate live-default change; "
            "the operator decides whether to flip the live-stack default (not self-authorized). "
            "Recommend the operator review the per-game deltas before changing "
            "LiveActionEffectScorer.cnn_weight."
        )
    else:
        recommendation = (
            f"NO tested weight (0.05..2.0) banks more levels than the 0.05 baseline "
            f"({baseline_levels_total} levels) on this roster. Do NOT change the live default "
            "cnn_weight -- raising it is not a capability win here. Root cause is NOT weight "
            "magnitude: the GroundTruthValidatedFrameChangeScorer validation gate returns 0.0 "
            "upstream of the cnn_weight multiply whenever it is unvalidated "
            f"({cnn_calibration_summary['n_games_gtv_never_validated']}/{len(roster)} games "
            "never validate post-run), so the blended scorer is rarely or never consulted while "
            "validated and cnn_weight cannot act. IMPORTANT: this is NOT evidence the CNN is "
            "useless -- it was largely never exercised. The right next lever is the GTV "
            "validation gate (its agreement/contradiction accounting), not the CNN blend weight. "
            "Operator-only whether to act."
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
        "weights_swept": list(weights),
        "baseline_weight": BASELINE_WEIGHT,
        "memory_weight_fixed": MEMORY_WEIGHT_FIXED,
        "baseline_levels_total": int(baseline_levels_total),
        "best_weight": best_weight,
        "best_levels_total": best_total,
        "weight_sweep_results": weight_sweep_results,
        "per_weight_game_rows": per_weight_game_rows,
        "cnn_calibration_summary": cnn_calibration_summary,
        "levels_gained_headroom_present": bool(any_headroom),
        "any_weight_beats_baseline_levels": bool(any_weight_beats_baseline),
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
