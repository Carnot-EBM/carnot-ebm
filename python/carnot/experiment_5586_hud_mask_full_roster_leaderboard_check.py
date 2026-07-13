"""Experiment 5586: full-CLAIMED-roster leaderboard check for auto_hud_mask.

The "next check" after `REQ-ARC-WMTE-5583`'s 2026-07-12 RESOLUTION deferred the
`levels_gained` question to live-submission telemetry (operator directive, after
`exp5584`/`exp5585` both hit a floor effect on their narrow 3-6 game rosters). No
actual scored submission has run yet, so this uses the project's OWN standing
pre-submission measurement tool (`scripts/arc_leaderboard_eval.py`, "the
measurement engine for rapid leaderboard progress" -- the harness used to sanity-
check the agent before any operator-gated Kaggle submission) at FULL scale: all
11 `CLAIMED` games (a materially larger and more diverse roster than exp5584's 6
or exp5585's 3), matching the SAME `--policy explorer` / `--budget 2500` shape
already checked into `results/arc_leaderboard_eval.json` from an earlier commit
(`1b909a4a3`), for continuity with the project's own established measurement.

Unlike exp5584/exp5585, this is NOT a repeat of the same narrow question --
"a re-run with a stated difference" per the Failed-Experiment Rerun Discipline:
the difference here is roster SIZE and DIVERSITY (11 games vs 3-6, entirely
disjoint from exp5584/exp5585's roster) and BUDGET (2500 vs 150-300), both of
which directly address exp5584's own diagnosis ("this roster/budget/explorer-
config combination has no observable headroom"). It turned out this cheap,
already-built harness runs at ~0.02s/action for the bare "explorer" policy
(CarnotAgentPolicy with no value_head/candidate_router/frame_change_scorer
override) -- nothing like the 1.75-11.5s/action of exp5584/exp5585's richer
configs -- so a genuinely larger, more diverse roster at a realistic budget was
affordable within a single interactive check, no background/multi-hour run
needed.

Design:
  - Both arms use `CarnotAgentPolicy(game, {}, force_explore=True,
    auto_hud_mask=...)` -- IDENTICAL construction except the one flag under
    test, matching `arc_leaderboard_eval.py --policy explorer`'s own default
    construction (verified: `_build_policy`'s fallback branch).
  - Roster = `sorted(CLAIMED)` (11 games, deterministic order) -- NOT hand-picked
    for known-short solve lengths this time (unlike exp5585); this is deliberately
    the project's OWN standing "claimed" roster, so the result is directly
    comparable to the historical baseline and any future re-run of the same
    standing tool.
  - Budget=2500, matching the checked-in historical baseline's own budget
    (`results/arc_leaderboard_eval.json` from commit `1b909a4a3`) for maximum
    comparability, though that baseline predates HUD-masking work by many
    commits and many OTHER unrelated changes, so it is cited as historical
    context, not a clean control.
  - Pre-registered `hud_survey` (same pattern as exp5584/exp5585) on this
    roster, confirming which games are even capable of showing an effect.
  - Deterministic, single run per arm (no bootstrap needed): every game is
    independent, `force_explore=True` disables any banked-plan branching, and
    the search itself has no intentional randomness at this config (no
    value_weight-driven tie-breaking beyond depth, no proposer). A positive or
    negative result here is a DIRECT observation, not a statistical estimate --
    still worth cross-checking against the pre-registered hud_survey for
    directional consistency (the adversarial-check habit this project applies
    throughout), which is the `mask_fired_matches_gain_pattern` field below.

Spec refs: REQ-ARC-WMTE-5583 (the deferred question this closes for the offline-
measurable portion; live-submission telemetry remains the authority for the
TRUE hidden-eval numbers, which this offline harness only approximates).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5586_hud_mask_full_roster_leaderboard_check"
RESULT_RELATIVE_PATH = "results/experiment_5586_hud_mask_full_roster_leaderboard_check.json"
SCHEMA = "carnot.exp5586.hud_mask_full_roster_leaderboard_check.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5586
DEFAULT_BUDGET = 2500

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "budget",
    "hud_survey",
    "control_results",
    "treatment_results",
    "levels_gained_control_total",
    "levels_gained_treatment_total",
    "efficiency_sum_control",
    "efficiency_sum_treatment",
    "mask_fired_matches_gain_pattern",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; either outcome (effect found or not) is a complete, real result"
    },
    "inference_substrate": {
        "principle": "no LLM/proposer invoked (bare CarnotAgentPolicy explorer, force_explore=True) -- pure CPU search, declared honestly"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures a state-identity masking mechanism's effect on live-path capability, not an executable win-check"
    },
    "roster": {
        "principle": "the project's own standing CLAIMED game roster (11 games) -- not hand-picked for this check, so results generalize to the project's existing measurement conventions"
    },
    "hud_survey": {
        "principle": "pre-registered per-game mask-detection result -- lets the reader verify any observed levels_gained difference is directionally consistent with which games can even be affected"
    },
    "mask_fired_matches_gain_pattern": {
        "principle": "adversarial-check field -- true only if EVERY per-game levels_gained delta (treatment minus control) is zero on HUD-negative games, ruling out an unrelated confound producing the observed effect"
    },
    "solve_provenance": {
        "principle": "development_proxy -- this is a live-path CAPABILITY measurement via the offline arcade with no per-game adapter or banked plan (force_explore=True), not a registry-eligible reproduced solve; ops/arc_solve_registry.yaml is NOT touched by this experiment"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_competition_agent import CLAIMED

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["claimed_roster_nonempty"] = len(CLAIMED) > 0
        checks["offline_arcade_makes_env"] = False
        try:
            first_game = sorted(CLAIMED)[0]
            env = arc.make(first_game, scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
        checks["claimed_roster_nonempty"] = False
    try:
        from carnot.agentic.arc_competition_agent import (
            CarnotAgentPolicy,
            _compute_hud_mask_from_frame,
        )

        checks["carnot_agent_policy_import"] = True
        checks["hud_mask_fn_import"] = True
    except Exception:
        checks["carnot_agent_policy_import"] = False
        checks["hud_mask_fn_import"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def survey_hud_masks(roster: list[str]) -> JsonDict:
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


def run_condition(roster: list[str], *, budget: int, auto_hud_mask: bool) -> JsonDict:
    """Sequential (matches arc_leaderboard_eval.py's own shape -- each game's run is
    fast enough at this bare config, ~0.02s/action, that threading buys nothing)."""

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy

    results: JsonDict = {}
    for game in roster:
        policy = CarnotAgentPolicy(game, {}, force_explore=True, auto_hud_mask=auto_hud_mask)
        row = lb.run_game(game, policy, budget=budget)
        results[game] = row
    return results


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_artifact(*, budget: int = DEFAULT_BUDGET, root: Path = REPO_ROOT) -> JsonDict:
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
            "roster": [],
            "budget": int(budget),
            "hud_survey": {},
            "control_results": {},
            "treatment_results": {},
            "levels_gained_control_total": 0,
            "levels_gained_treatment_total": 0,
            "efficiency_sum_control": 0.0,
            "efficiency_sum_treatment": 0.0,
            "mask_fired_matches_gain_pattern": False,
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    from carnot.agentic.arc_competition_agent import CLAIMED

    roster = sorted(CLAIMED)
    hud_survey = survey_hud_masks(roster)
    control_results = run_condition(roster, budget=budget, auto_hud_mask=False)
    treatment_results = run_condition(roster, budget=budget, auto_hud_mask=True)

    levels_gained_control_total = sum(r["levels"] for r in control_results.values())
    levels_gained_treatment_total = sum(r["levels"] for r in treatment_results.values())
    efficiency_sum_control = sum(r["efficiency"] for r in control_results.values())
    efficiency_sum_treatment = sum(r["efficiency"] for r in treatment_results.values())

    # Adversarial check: every per-game gain/loss must be on a HUD-positive game.
    # A delta on a HUD-negative game would mean the observed effect is NOT
    # attributable to auto_hud_mask -- an unrelated confound (env nondeterminism,
    # a stale cache, a different code path) would be the honest explanation, and
    # the headline claim below would need to be withdrawn.
    mask_fired_matches_gain_pattern = True
    per_game_deltas: JsonDict = {}
    for game in roster:
        delta = treatment_results[game]["levels"] - control_results[game]["levels"]
        per_game_deltas[game] = delta
        if delta != 0 and not hud_survey[game]["has_hud"]:
            mask_fired_matches_gain_pattern = False

    total_delta = levels_gained_treatment_total - levels_gained_control_total
    if total_delta > 0 and mask_fired_matches_gain_pattern:
        verdict = (
            f"complete: hud_mask_full_roster_positive_{levels_gained_control_total}_to_"
            f"{levels_gained_treatment_total}_levels_adversarial_check_clean"
        )
    elif total_delta > 0 and not mask_fired_matches_gain_pattern:
        verdict = (
            "complete: hud_mask_full_roster_positive_but_adversarial_check_failed_do_not_trust"
        )
    elif total_delta < 0:
        verdict = "complete: hud_mask_full_roster_negative_regression_found"
    else:
        verdict = "complete: hud_mask_full_roster_honest_null_no_delta"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": roster,
        "budget": int(budget),
        "hud_survey": hud_survey,
        "control_results": control_results,
        "treatment_results": treatment_results,
        "per_game_levels_delta": per_game_deltas,
        "levels_gained_control_total": levels_gained_control_total,
        "levels_gained_treatment_total": levels_gained_treatment_total,
        "efficiency_sum_control": round(efficiency_sum_control, 4),
        "efficiency_sum_treatment": round(efficiency_sum_treatment, 4),
        "mask_fired_matches_gain_pattern": mask_fired_matches_gain_pattern,
        "solve_provenance": "development_proxy",
        "historical_baseline_context": {
            "path": "results/arc_leaderboard_eval.json (as of commit 1b909a4a3)",
            "live_levels": 1,
            "note": (
                "predates HUD-masking work and many other unrelated changes -- cited as "
                "historical context only, NOT a clean control for this experiment's "
                "paired control_results (which is the actual apples-to-apples comparison)"
            ),
        },
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
