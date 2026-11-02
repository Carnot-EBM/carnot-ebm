"""Experiment 5587: does auto_hud_mask's levels_gained win replicate under the REAL
submitted E3AgentPolicy cascade (not the bare tier-1 explorer exp5586 tested)?

exp5586 found a clean, adversarially-checked levels_gained win (1->5 total levels
across 11 CLAIMED games) using CarnotAgentPolicy's BARE explorer (no value_head,
candidate_router, frame_change_scorer, or action_prior override). That is NOT what
actually gets submitted -- `make_carnot_agent(Agent)` (cascade=True, the shipped
default) constructs `E3AgentPolicy`, which resolves ALL of those components to
their real defaults (`DaggerWinReachabilityValueHead`,
`CrossGameDiscriminativeCandidateRouter`, `GroundTruthValidatedFrameChangeScorer`
with a small CNN, `ColorBlobSaliencePrior`). exp5585 tested THIS richer config but
found no levels_gained headroom at all on its narrow 3-game/budget=150 slice (a
floor effect, not a negative result). This script closes that gap directly: same
richer E3AgentPolicy construction as exp5585, but targeted at the SPECIFIC games
exp5586 already showed a bare-explorer effect on, at a larger budget.

Roster (6 games, NOT the full 11 -- see budget/wall-clock note below):
  - cd82, sp80, su15, tu93: the four games where exp5586's bare explorer went
    from 0 to 1 level with auto_hud_mask on. If the richer cascade ALSO shows
    the same gain on these specific games, that is strong, targeted evidence
    the effect transfers to what actually ships (not proof for the full
    roster, but a much sharper test than a blind re-run of all 11).
  - m0r0: a HUD-positive game that stayed flat in BOTH arms of exp5586 --
    included as an in-roster check that the richer cascade does not somehow
    unlock it either way (a change here would be a genuinely new, unexpected
    finding, not just noise).
  - sk48: the HUD-negative harmlessness control, reused from exp5584/exp5585's
    roster for continuity -- must stay flat in both arms.

Budget=400, NOT exp5586's 2500: a calibration run
(python r11l via E3AgentPolicy, proposer=None) measured 6.3s/action for this
richer config -- between exp5584's 1.75s/action (bare-ish) and exp5585's
11.5s/action (a different, apparently more candidate-dense game). At 6
games x 2 arms x 400 actions x ~7s/action serial estimate, even with the
~3.5x threading speedup exp5585 observed empirically, this is still a
multi-hour run -- run in the BACKGROUND, not blocking. proposer=None (no
LLM/GGUF): the tier-3 induction-on-stall escalation is NOT exercised here,
so this still does not test the absolute full cascade end-to-end, only
tiers 1-2 (explorer + real value/router/scorer/prior guidance) -- the
tier-3 LLM path would need a cached local SOTA GGUF and materially more
wall-clock, out of scope for this check.

Spec refs: REQ-ARC-WMTE-5583 (the same deferred-then-partially-closed
question exp5586 addressed; this narrows the remaining "does it transfer to
the real cascade" gap).
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
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

EXPERIMENT_ID = "experiment_5587_hud_mask_full_cascade_check"
RESULT_RELATIVE_PATH = "results/experiment_5587_hud_mask_full_cascade_check.json"
SCHEMA = "carnot.exp5587.hud_mask_full_cascade_check.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5587
DEFAULT_BUDGET = 400
DEFAULT_ROSTER = ("cd82", "sp80", "su15", "tu93", "m0r0", "sk48")

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
    "per_game_levels_delta",
    "mask_fired_matches_gain_pattern",
    "replicates_exp5586_pattern",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; either outcome (replicates, doesn't, or still no headroom) is a complete, real result"
    },
    "inference_substrate": {
        "principle": "no LLM/proposer invoked (E3AgentPolicy tiers 1-2 only, proposer=None) -- real value/router/scorer/prior components, but not the tier-3 LLM induction escalation; declared honestly"
    },
    "verifier_is_oracle": {
        "principle": "False -- this measures a state-identity masking mechanism's effect on live-path capability, not an executable win-check"
    },
    "roster": {
        "principle": "targeted subset of exp5586's roster (the 4 games that gained + 1 HUD-positive-flat + 1 HUD-negative control), not a blind full re-run -- a sharper test of transfer-to-the-real-cascade than a diffuse one, chosen for wall-clock feasibility"
    },
    "hud_survey": {
        "principle": "pre-registered per-game mask-detection result, reused from exp5584/exp5586 for these specific games -- lets the reader verify any observed delta is directionally consistent"
    },
    "mask_fired_matches_gain_pattern": {
        "principle": "adversarial-check field -- true only if every per-game levels_gained delta is zero on the HUD-negative control (sk48), ruling out an unrelated confound"
    },
    "replicates_exp5586_pattern": {
        "principle": "true only if AT LEAST ONE of the four exp5586-positive games also gains under the real cascade -- the specific, falsifiable claim this experiment exists to test"
    },
    "solve_provenance": {
        "principle": "development_proxy -- live-path CAPABILITY measurement via the offline arcade, no per-game adapter or banked plan; ops/arc_solve_registry.yaml is NOT touched"
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
    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    policy = E3AgentPolicy(game, proposer=None, auto_hud_mask=auto_hud_mask)
    row = lb.run_game(game, policy, budget=budget)
    with lock:
        results[game] = row


def run_both_conditions(
    roster: tuple[str, ...], *, budget: int
) -> tuple[JsonDict, JsonDict, float]:
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
                kwargs={"budget": budget, "auto_hud_mask": False, "results": control, "lock": lock},
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


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
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
            "roster": list(roster),
            "budget": int(budget),
            "hud_survey": {},
            "control_results": {},
            "treatment_results": {},
            "levels_gained_control_total": 0,
            "levels_gained_treatment_total": 0,
            "per_game_levels_delta": {},
            "mask_fired_matches_gain_pattern": False,
            "replicates_exp5586_pattern": False,
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

    hud_survey = survey_hud_masks(roster)
    control_results, treatment_results, combined_wall_clock_s = run_both_conditions(
        roster, budget=budget
    )

    levels_gained_control_total = sum(r["levels"] for r in control_results.values())
    levels_gained_treatment_total = sum(r["levels"] for r in treatment_results.values())

    mask_fired_matches_gain_pattern = True
    per_game_deltas: JsonDict = {}
    exp5586_positive_games = {"cd82", "sp80", "su15", "tu93"}
    for game in roster:
        delta = treatment_results[game]["levels"] - control_results[game]["levels"]
        per_game_deltas[game] = delta
        if delta != 0 and not hud_survey[game]["has_hud"]:
            mask_fired_matches_gain_pattern = False

    replicates_exp5586_pattern = any(
        per_game_deltas.get(g, 0) > 0 for g in exp5586_positive_games if g in per_game_deltas
    )
    total_delta = levels_gained_treatment_total - levels_gained_control_total
    any_headroom = any(
        control_results[g]["levels"] > 0 or treatment_results[g]["levels"] > 0 for g in roster
    )

    if not any_headroom:
        verdict = "complete: hud_mask_full_cascade_still_no_headroom"
    elif total_delta > 0 and mask_fired_matches_gain_pattern and replicates_exp5586_pattern:
        verdict = (
            f"complete: hud_mask_full_cascade_replicates_{levels_gained_control_total}_to_"
            f"{levels_gained_treatment_total}_levels_adversarial_check_clean"
        )
    elif total_delta > 0 and not mask_fired_matches_gain_pattern:
        verdict = (
            "complete: hud_mask_full_cascade_positive_but_adversarial_check_failed_do_not_trust"
        )
    elif total_delta < 0:
        verdict = "complete: hud_mask_full_cascade_negative_regression_found"
    else:
        verdict = "complete: hud_mask_full_cascade_honest_null_headroom_present_no_delta"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "budget": int(budget),
        "hud_survey": hud_survey,
        "control_results": control_results,
        "treatment_results": treatment_results,
        "per_game_levels_delta": per_game_deltas,
        "levels_gained_control_total": levels_gained_control_total,
        "levels_gained_treatment_total": levels_gained_treatment_total,
        "mask_fired_matches_gain_pattern": mask_fired_matches_gain_pattern,
        "replicates_exp5586_pattern": replicates_exp5586_pattern,
        "levels_gained_headroom_present": any_headroom,
        "solve_provenance": "development_proxy",
        "predecessor_experiments": [
            "experiment_5584_hud_mask_offline_ab",
            "experiment_5585_hud_mask_strong_config_ab",
            "experiment_5586_hud_mask_full_roster_leaderboard_check",
        ],
        "combined_wall_clock_s": round(combined_wall_clock_s, 3),
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
