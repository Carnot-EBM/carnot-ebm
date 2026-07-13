"""Experiment 5585: strong-config offline matched-budget A/B for auto_hud_mask.

Direct follow-up to `exp5584` (`results/experiment_5584_hud_mask_offline_ab.json`).
exp5584 confirmed `auto_hud_mask` collapses cosmetic HUD-tick duplicate states (a
real, CI-excludes-zero effect on `distinct_states_discovered`) but could not test
whether that collapse translates into reaching MORE levels, because its bare
explorer config (`proposer=None, value_head=lambda _f: 0.0` -- the SAME config
`exp5578` also used) never reached a single level-up in EITHER arm on ANY roster
game at budget=300 (a floor-effect, per exp5584's own `false_negative_risk_note`
and `adversarial_verify.py`'s independent FALSE_NEGATIVE_RISK flag on that
artifact). This script is the "re-run with a stated difference" exp5584's own
recommendation called for.

What's different from exp5584 (the ONE change, everything else held matched):
  - `E3AgentPolicy` is constructed with its OWN real defaults instead of the
    stripped-down override -- `value_head` resolves to
    `DaggerWinReachabilityValueHead` (not a flat lambda), and `candidate_router` /
    `frame_change_scorer` / `action_prior` all resolve to the actual submitted
    components (`CrossGameDiscriminativeCandidateRouter`,
    `GroundTruthValidatedFrameChangeScorer` wrapping a small CNN + persistent
    action-effect memory, `ColorBlobSaliencePrior`). `proposer` stays `None` --
    the LLM/GGUF proposer is only invoked on STALL escalation to tier-3 induction,
    not needed for basic exploration, and keeping it out avoids a GPU/GGUF
    precondition this validation doesn't need.
  - Roster trimmed to 3 games (`s5i5, g50t, sk48`) instead of exp5578/exp5584's 6,
    and budget lowered to 150 (from 300). Both cuts are WALL-CLOCK, not scientific,
    decisions: a smoke test (`carnot-baseline-worktree`-adjacent, see the 2026-07-12
    session record) measured this stronger config at ~11.5s/action -- roughly 6.5x
    slower than exp5584's bare config's ~1.75s/action, because the frame-change
    scorer now runs a real (if small, CPU-only) CNN forward pass per candidate. A
    naive 6-game/budget=300 rerun would cost multiple hours; this script picks the
    3 roster games with the SHORTEST known-minimal scripted solve lengths
    (`ops/arc_solve_registry.yaml`: s5i5=39, g50t=48, sk48=44) to maximize the
    chance that budget=150 (roughly 3-3.8x each game's minimal length) actually
    gives the stronger, guided search a real shot at a level-up, while keeping one
    HUD-negative game (sk48) as the harmlessness control. This SHRINKS statistical
    power (N=3, not N=6) -- an even wider CI is expected than exp5584's already-wide
    one -- but a null on N=3 with real headroom is still more informative than a
    null on N=6 with NO headroom anywhere (exp5584's actual result).

Everything else matches exp5584 exactly: development-proxy discipline (fresh
env.reset(), no registry-prefix consultation), threaded per-game-run purely for
wall-clock (no shared state, so nothing about correctness depends on the
concurrent execution shape), matched compute between arms (identical roster,
budget, and E3AgentPolicy construction except `auto_hud_mask`), paired bootstrap
CI on the delta, the same two positive controls (`mask_fired_matches_survey` and
`levels_gained_headroom_present`), and the same FALSE_NEGATIVE_RISK acknowledgment
path if headroom is STILL absent even with the stronger config.

Spec refs: REQ-ARC-WMTE-5583 (states the validation obligation this and exp5584
both fulfill).
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5585_hud_mask_strong_config_ab"
RESULT_RELATIVE_PATH = "results/experiment_5585_hud_mask_strong_config_ab.json"
SCHEMA = "carnot.exp5585.hud_mask_strong_config_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5585
DEFAULT_BUDGET_PER_GAME = 150
DEFAULT_BOOTSTRAPS = 2000
# Shortest known-minimal scripted solve lengths on the exp5578/exp5584 roster
# (s5i5=39, g50t=48, sk48=44 actions; ops/arc_solve_registry.yaml), maximizing
# headroom per wall-clock action at this config's ~6.5x-slower-per-action cost.
# sk48 is the roster's sole HUD-negative game (see exp5584's pre-registered
# survey) -- kept as the in-roster harmlessness control.
DEFAULT_ROSTER = ("s5i5", "g50t", "sk48")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "budget_per_game",
    "hud_survey",
    "control_results",
    "treatment_results",
    "levels_gained_delta_mean",
    "levels_gained_delta_ci",
    "distinct_states_delta_mean",
    "distinct_states_delta_ci",
    "mask_fired_matches_survey",
    "levels_gained_headroom_present",
    "recommendation",
    "no_regression_fallback",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; a measured null is complete (negative-but-real), never partial"
    },
    "inference_substrate": {
        "principle": "no LLM/proposer invoked (StepwiseExplorer explore-phase only, proposer=None) -- pure CPU search including a small local CNN frame-change scorer, declared honestly"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures whether a state-identity masking mechanism adds value, not whether the executable win-check agrees with itself"
    },
    "roster": {
        "principle": "3-game subset of exp5578/exp5584's roster, chosen for shortest known-minimal solve length to fit this config's higher per-action cost within an interactive session; sk48 kept as the HUD-negative harmlessness control"
    },
    "hud_survey": {
        "principle": "pre-registered per-game mask-detection result on the initial reset frame -- lets the reader confirm treatment-arm results are consistent with the mechanism actually firing where expected"
    },
    "levels_gained_delta_ci": {
        "principle": "paired bootstrap CI on (treatment - control) per game; a claim requires the CI to exclude 0.0, same bar exp4582/exp5578/exp5584 used"
    },
    "distinct_states_delta_ci": {
        "principle": "paired bootstrap CI on the dedup-collapse proxy (graph node count); the mechanism's DIRECT effect, separate from whether that translates into a levels_gained win"
    },
    "mask_fired_matches_survey": {
        "principle": "positive control -- confirms the mechanism actually activates on treatment-arm HUD-positive games rather than silently no-op-ing the whole roster"
    },
    "levels_gained_headroom_present": {
        "principle": "SECOND positive control (CLAUDE.md FALSE_NEGATIVE_RISK) -- confirms at least one arm reached a level-up somewhere on the roster; the specific gap exp5584 hit and this script exists to close"
    },
    "no_regression_fallback": {
        "principle": "SUBMITTED_AUTO_HUD_MASK_ENABLED stays False regardless of this result until an operator reviews it -- this script never flips the flag"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


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
        from carnot.agentic.arc_competition_agent import E3AgentPolicy, _compute_hud_mask_from_frame

        checks["e3_policy_import"] = True
        checks["hud_mask_fn_import"] = True
    except Exception:
        checks["e3_policy_import"] = False
        checks["hud_mask_fn_import"] = False
    try:
        # The strong config's frame_change_scorer loads a checkpointed CNN via torch
        # (CPU); confirm the checkpoint/import path resolves before committing to a
        # long run, rather than discovering a missing checkpoint 90 minutes in.
        from carnot.agentic.arc_competition_agent import _load_submitted_frame_change_scorer

        checks["frame_change_scorer_loadable"] = _load_submitted_frame_change_scorer() is not None
    except Exception:
        checks["frame_change_scorer_loadable"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def survey_hud_masks(roster: tuple[str, ...]) -> JsonDict:
    """Pre-registered: does each roster game's INITIAL reset frame trigger the mask?"""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    survey: JsonDict = {}
    for game in roster:
        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        frame = env.reset()
        mask = _compute_hud_mask_from_frame(frame)
        survey[game] = {
            "has_hud": mask is not None,
            "n_masked_cells": int(mask.sum()) if mask is not None else 0,
        }
    return survey


def _play_one_game(
    game: str,
    *,
    budget: int,
    auto_hud_mask: bool,
    results: dict[str, JsonDict],
    lock: threading.Lock,
) -> None:
    """Run one game to its action budget, fresh env.reset() only (development-proxy
    discipline), using E3AgentPolicy's OWN real defaults (the one deliberate
    difference from exp5584 -- see module docstring). Runs on its own thread purely
    for wall-clock; no shared mutable state besides the lock-protected results dict."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    frames = [frame]
    # NOTE the absence of value_head=... here versus exp5584 -- this is the
    # entire experimental manipulation beyond auto_hud_mask itself.
    policy = E3AgentPolicy(game, proposer=None, auto_hud_mask=auto_hud_mask)

    start_level = int(getattr(frame, "levels_completed", 0) or 0)
    first_levelup_action: int | None = None
    max_level = start_level
    error: str | None = None
    try:
        for step in range(int(budget)):
            kind, data = policy.next_move(frames, frame)
            if kind == "RESET" or kind is None:
                frame = env.reset()
            else:
                frame = env.step(_game_action(GameAction, int(kind)), data=data)
            frames.append(frame)
            level = int(getattr(frame, "levels_completed", 0) or 0)
            if level > max_level:
                max_level = level
                if first_levelup_action is None:
                    first_levelup_action = step + 1
    except Exception as exc:  # pragma: no cover - live boundary
        error = f"{type(exc).__name__}: {exc}"

    explorer = policy.explorer
    row = {
        "game": game,
        "start_level": start_level,
        "max_level_reached": max_level,
        "levels_gained": max_level - start_level,
        "first_levelup_action": first_levelup_action,
        "actions_used": budget,
        "distinct_states_discovered": len(explorer.graph) if explorer is not None else None,
        "hud_mask_fired": bool(explorer.hud_mask is not None) if explorer is not None else False,
        "error": error,
    }
    with lock:
        results[game] = row


def run_both_conditions(
    roster: tuple[str, ...], *, budget: int
) -> tuple[JsonDict, JsonDict, float]:
    """Launch all 2*len(roster) independent game-runs (control + treatment) as one
    thread batch. Each run is a fully independent env/policy instance."""

    control: JsonDict = {}
    treatment: JsonDict = {}
    lock = threading.Lock()
    t0 = time.time()
    threads = []
    for game in roster:
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={
                    "budget": budget,
                    "auto_hud_mask": False,
                    "results": control,
                    "lock": lock,
                },
            )
        )
        threads.append(
            threading.Thread(
                target=_play_one_game,
                args=(game,),
                kwargs={
                    "budget": budget,
                    "auto_hud_mask": True,
                    "results": treatment,
                    "lock": lock,
                },
            )
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return control, treatment, time.time() - t0


def _paired_bootstrap_delta_ci(
    deltas: list[float], *, random_seed: int, n_bootstrap: int
) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if n_bootstrap <= 0:
        mean = sum(deltas) / len(deltas)
        return [round(mean, 10), round(mean, 10)]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = []
    for _ in range(int(n_bootstrap)):
        total = 0.0
        for _s in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 10), round(float(hi), 10)]


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    budget: int = DEFAULT_BUDGET_PER_GAME,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
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
            "roster": list(roster),
            "budget_per_game": int(budget),
            "hud_survey": {},
            "control_results": {},
            "treatment_results": {},
            "levels_gained_delta_mean": 0.0,
            "levels_gained_delta_ci": [0.0, 0.0],
            "distinct_states_delta_mean": 0.0,
            "distinct_states_delta_ci": [0.0, 0.0],
            "mask_fired_matches_survey": False,
            "levels_gained_headroom_present": False,
            "false_negative_risk_note": None,
            "recommendation": "blocked -- preconditions failed, no measurement taken",
            "no_regression_fallback": True,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    hud_survey = survey_hud_masks(roster)
    control_results, treatment_results, combined_wall_clock_s = run_both_conditions(
        roster, budget=budget
    )

    levels_gained_deltas: list[float] = []
    states_deltas: list[float] = []
    mask_fired_ok = True
    for game in roster:
        c = control_results[game]
        t = treatment_results[game]
        levels_gained_deltas.append(float(t["levels_gained"] - c["levels_gained"]))
        if (
            c["distinct_states_discovered"] is not None
            and t["distinct_states_discovered"] is not None
        ):
            states_deltas.append(
                float(t["distinct_states_discovered"] - c["distinct_states_discovered"])
            )
        expected_fire = bool(hud_survey[game]["has_hud"])
        if t["hud_mask_fired"] and not expected_fire:
            mask_fired_ok = False
        if expected_fire and not t["hud_mask_fired"]:
            mask_fired_ok = False

    levels_gained_delta_mean = (
        statistics.mean(levels_gained_deltas) if levels_gained_deltas else 0.0
    )
    levels_gained_delta_ci = _paired_bootstrap_delta_ci(
        levels_gained_deltas, random_seed=RANDOM_SEED, n_bootstrap=n_bootstrap
    )
    distinct_states_delta_mean = statistics.mean(states_deltas) if states_deltas else 0.0
    distinct_states_delta_ci = _paired_bootstrap_delta_ci(
        states_deltas, random_seed=RANDOM_SEED + 1, n_bootstrap=n_bootstrap
    )

    levels_ci_excludes_zero = levels_gained_delta_ci[0] > 0.0 or levels_gained_delta_ci[1] < 0.0
    states_ci_excludes_zero = distinct_states_delta_ci[0] > 0.0 or distinct_states_delta_ci[1] < 0.0
    any_error = any(control_results[g]["error"] or treatment_results[g]["error"] for g in roster)
    any_levelup_either_arm = any(
        control_results[g]["levels_gained"] > 0 or treatment_results[g]["levels_gained"] > 0
        for g in roster
    )
    levels_gained_headroom_present = any_levelup_either_arm

    if any_error:
        verdict = "complete: hud_mask_strong_config_ab_run_error_see_error_fields"
        recommendation = "An arm errored mid-run; do not act on deltas until re-run clean."
        false_negative_risk_note = None
    elif levels_ci_excludes_zero and levels_gained_delta_mean > 0.0:
        verdict = "complete: hud_mask_strong_config_ab_positive_ci_excludes_zero"
        recommendation = (
            "levels_gained_delta_ci excludes 0.0 and the mean is positive -- masking "
            "shows a real levels_gained effect with a real explorer config. Still "
            "N=3 (small even relative to exp5584's already-small N=6); consider a "
            "larger roster before flipping SUBMITTED_AUTO_HUD_MASK_ENABLED, but this "
            "clears the CI-excludes-baseline bar exp4582/exp5578/exp5584 used, and "
            "resolves exp5584's open false_negative_risk_note."
        )
        false_negative_risk_note = None
    elif levels_ci_excludes_zero and levels_gained_delta_mean < 0.0:
        verdict = "complete: hud_mask_strong_config_ab_negative_ci_excludes_zero"
        recommendation = (
            "levels_gained_delta_ci excludes 0.0 and the mean is NEGATIVE -- masking "
            "appears to HURT with a real explorer config. Do not flip "
            "SUBMITTED_AUTO_HUD_MASK_ENABLED; investigate the mechanism for a "
            "false-positive mask hiding real board state before any further attempt."
        )
        false_negative_risk_note = None
    elif not levels_gained_headroom_present:
        verdict = "complete: hud_mask_strong_config_ab_still_no_headroom"
        false_negative_risk_note = (
            "FALSE_NEGATIVE_RISK acknowledgment: EVEN with E3AgentPolicy's real "
            "defaults (DaggerWinReachabilityValueHead, CrossGameDiscriminativeCandidateRouter, "
            "GroundTruthValidatedFrameChangeScorer, ColorBlobSaliencePrior) and a "
            "roster pre-selected for the shortest known-minimal solve lengths "
            "(s5i5=39, g50t=48, sk48=44 actions), budget=150 (3-3.8x those lengths) "
            "STILL reached zero level-ups in either arm on any roster game. This is "
            "a stronger, harder-to-dismiss null than exp5584's (that config was "
            "already known-weak; this one is the actual submitted default component "
            "set) -- it suggests undirected/lightly-guided from-scratch search on "
            "this roster may need either an LLM proposer (tier-3 escalation, not "
            "used here) or substantially more budget to find these solve paths, "
            "independent of HUD masking. distinct_states_delta remains the only "
            "measured effect of the mechanism; see that field for whether it still "
            "holds at this budget/roster."
        )
        recommendation = (
            "Do NOT treat this as resolving exp5584's open question -- see "
            "false_negative_risk_note. Two honest options going forward: (a) invoke "
            "the tier-3 LLM proposer (requires a cached local SOTA GGUF and GPU/iGPU "
            "-- a materially bigger precondition and wall-clock commitment than this "
            "script's scope), or (b) accept that levels_gained cannot be cheaply "
            "measured for this mechanism offline and defer the flip decision to "
            "live-submission telemetry instead. Keep SUBMITTED_AUTO_HUD_MASK_ENABLED "
            "off by default either way."
        )
    else:
        verdict = "complete: hud_mask_strong_config_ab_honest_null_ci_includes_zero"
        recommendation = (
            "No detectable effect on levels_gained above noise on this roster/budget, "
            "with headroom confirmed present (at least one arm reached a level-up "
            "somewhere on the roster) -- a genuinely informative null. Per the "
            "Failed-Experiment Rerun Discipline: do not re-propose this exact "
            "measurement without a stated difference. Keep "
            "SUBMITTED_AUTO_HUD_MASK_ENABLED off by default."
        )
        false_negative_risk_note = None

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "budget_per_game": int(budget),
        "hud_survey": hud_survey,
        "control_results": control_results,
        "treatment_results": treatment_results,
        "levels_gained_delta_mean": round(levels_gained_delta_mean, 6),
        "levels_gained_delta_ci": levels_gained_delta_ci,
        "levels_gained_ci_excludes_zero": levels_ci_excludes_zero,
        "distinct_states_delta_mean": round(distinct_states_delta_mean, 6),
        "distinct_states_delta_ci": distinct_states_delta_ci,
        "distinct_states_ci_excludes_zero": states_ci_excludes_zero,
        "mask_fired_matches_survey": mask_fired_ok,
        "levels_gained_headroom_present": levels_gained_headroom_present,
        "false_negative_risk_note": false_negative_risk_note,
        "recommendation": recommendation,
        "no_regression_fallback": True,
        "combined_wall_clock_s": round(combined_wall_clock_s, 3),
        "n_bootstrap": int(n_bootstrap),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
        "predecessor_experiment": "experiment_5584_hud_mask_offline_ab",
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
