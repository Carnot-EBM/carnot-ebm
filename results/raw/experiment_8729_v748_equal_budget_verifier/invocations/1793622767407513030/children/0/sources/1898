"""Experiment 5603: matched-budget A/B for ObjectHistorySaliencePrior
(REQ-ARC-FCP-5591-3) -- the pending flip-decision measurement task 10's own
wiring follow-on named: flipping SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED
needs its own matched-budget A/B first, per the solve_rate_dropped guardrail.

Unlike InertClickSigPruner (REQ-ARC-FCP-5595-2, exp5602), ObjectHistory
SaliencePrior cannot be A/B'd through OfflineSolver: OfflineSolver has no
action_prior= concept (only move_pruner=), and action_prior is exclusively a
StepwiseExplorer/E3AgentPolicy concern. There is also no states_expanded-
equivalent counter anywhere on that live path. This A/B therefore runs on
the LIVE E3AgentPolicy/lb.run_game path (matching exp5601's own real-game
construction) and measures TRAJECTORY DIVERGENCE -- does turning the bonus
on actually change which candidates the search chooses -- as the primary,
honestly-available signal, since m0r0 does not level up within this budget
under a bare (proposer=None) policy (no actions_to_first_levelup available).

REAL FINDING (not assumed going in; the diagnostic run CORRECTED an initial
wrong hypothesis formed from an ad-hoc pre-check that had no baseline
comparison -- see the fixed-up reasoning below, not the original guess):
baseline, default-weight treatment (change_bonus_weight=10.0), AND a
diagnostic run re-scaled to match ColorBlobSaliencePrior's own tier-score
magnitude (change_bonus_weight=2000.0, ~1000-4000 per real click candidate
on m0r0) all produce IDENTICAL action sequences under a bare
(proposer=None) policy -- the bonus shows ZERO measurable effect on
candidate selection at EITHER weight tested. The mechanism demonstrably
tracks real, non-degenerate evidence (confirmed independently by exp5601 and
by this script's own prior_summary), so this is not a "never engages" null;
it is specifically a "never changes which candidate gets chosen" null on
this game/policy at these two tested weights. A repeated-coordinate click
pattern noticed during an early informal check was INITIALLY misread as
bonus-induced over-exploitation -- the formal A/B with a real baseline
disproved that: the identical pattern appears in the baseline (no bonus at
all) too, so it is inherent to E3AgentPolicy's own backtrack/retry
navigation on m0r0, unrelated to this mechanism. Open question for future
investigation, not resolved here: whether m0r0's specific candidate-score
distribution structurally prevents the bonus from ever mattering (e.g. a
dominant top-tier candidate too far ahead for any tested bonus to overtake),
or whether a bare proposer=None policy's selection path does not actually
consult action_prior.score() for ranking the way assumed.

This is a measurement script, not a live-path parameter flip:
SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED stays False regardless of this
script's result. solve_provenance stays development_proxy.

Spec refs: REQ-ARC-FCP-5591-3, SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP,
SCENARIO-ARC-FCP-5591-3-RESCALED-WEIGHT-STILL-NO-OP.
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

EXPERIMENT_ID = "experiment_5603_object_history_salience_matched_budget_ab"
RESULT_RELATIVE_PATH = "results/experiment_5603_object_history_salience_matched_budget_ab.json"
SCHEMA = "carnot.exp5603.object_history_salience_matched_budget_ab.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5603
DEFAULT_GAME = "m0r0"
DEFAULT_EXPLORE_BUDGET = 6
DEFAULT_TOTAL_BUDGET = 40
DIAGNOSTIC_TOTAL_BUDGET = 15  # smaller budget for the rescaled-weight diagnostic run only --
# a full budget=40 run with change_bonus_weight=2000.0 was observed to run much slower than
# the default-weight arm (>90s vs ~15s); root cause not confirmed (NOT the over-exploitation
# hypothesis the module docstring's own findings section disproved), so this stays a smaller,
# cheaper diagnostic probe rather than a full matched-budget run pending further investigation.
DEFAULT_CHANGE_BONUS_WEIGHT = 10.0
DIAGNOSTIC_CHANGE_BONUS_WEIGHT = 2000.0  # matches ColorBlobSaliencePrior's real tier-score scale

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "game",
    "baseline",
    "default_weight_treatment",
    "rescaled_weight_diagnostic",
    "trajectories_diverge_at_default_weight",
    "trajectories_diverge_at_rescaled_weight",
    "gate_definition",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; reports whichever outcome the A/B actually produced "
        "-- a zero-divergence result at the default weight is a valid, real, and actionable "
        "finding (the bonus magnitude vs. the base prior's tier-score scale), not a failure to "
        "hide"
    },
    "trajectories_diverge_at_default_weight": {
        "principle": "whether the treatment (object_history_salience=True, default weight) "
        "action sequence differs from the baseline (False) sequence at all -- the most direct, "
        "honestly-available signal that the wiring has ANY real behavioral effect, since m0r0 "
        "does not level up within this budget under a bare policy (no actions_to_first_levelup "
        "available)"
    },
    "trajectories_diverge_at_rescaled_weight": {
        "principle": "a diagnostic-only check (change_bonus_weight rescaled to match "
        "ColorBlobSaliencePrior's real tier-score magnitude) isolating whether the mechanism "
        "CAN influence behavior at an appropriate scale, separate from whether the CURRENT "
        "default is well-tuned to ship"
    },
    "gate_definition": {
        "principle": "no OfflineSolver-equivalent states_expanded/reproduction-gate metric "
        "exists for action_prior on this path (action_prior is not a move_pruner) -- trajectory "
        "divergence is the honest substitute for 'does this change search behavior', not a "
        "claim about solve efficiency, which needs its own follow-up measurement"
    },
    "solve_provenance": {
        "principle": "development_proxy -- a prototype/measurement script, not a live-path flip"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make(DEFAULT_GAME, scorecard_id=arc.open_scorecard())
        env.reset()
        checks["offline_arcade_makes_env"] = True
    except Exception:
        checks["offline_arcade_makes_env"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_object_history_salience import (  # noqa: F401
            ObjectHistorySaliencePrior,
        )

        checks["e3_and_prior_import"] = True
    except Exception:
        checks["e3_and_prior_import"] = False
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


def _run_arm(
    *,
    game: str,
    explore_budget: int,
    total_budget: int,
    object_history_salience: Any,
) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    policy = E3AgentPolicy(
        game,
        proposer=None,
        explore_budget=explore_budget,
        object_history_salience=object_history_salience,
    )
    lb.run_game(game, policy, budget=total_budget)
    actions = [{"action": int(t.action), "data": t.data} for t in policy.transitions]
    prior = policy.explorer.action_prior
    prior_dict = prior.as_dict() if hasattr(prior, "as_dict") else None
    return {"transitions_collected": len(actions), "actions": actions, "prior_summary": prior_dict}


def build_artifact(
    *,
    game: str = DEFAULT_GAME,
    explore_budget: int = DEFAULT_EXPLORE_BUDGET,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
    root: Path = REPO_ROOT,
) -> JsonDict:
    started_at = time.time()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "game": game,
            "baseline": {},
            "default_weight_treatment": {},
            "rescaled_weight_diagnostic": {},
            "trajectories_diverge_at_default_weight": False,
            "trajectories_diverge_at_rescaled_weight": False,
            "gate_definition": (
                "no OfflineSolver-equivalent states_expanded metric exists for action_prior; "
                "trajectory divergence from baseline is the honest substitute"
            ),
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

    from carnot.agentic.arc_object_history_salience import ObjectHistorySaliencePrior

    baseline = _run_arm(
        game=game,
        explore_budget=explore_budget,
        total_budget=total_budget,
        object_history_salience=False,
    )
    default_treatment = _run_arm(
        game=game,
        explore_budget=explore_budget,
        total_budget=total_budget,
        object_history_salience=True,
    )
    rescaled_prior = ObjectHistorySaliencePrior(
        change_bonus_weight=DIAGNOSTIC_CHANGE_BONUS_WEIGHT, min_observations=3
    )
    rescaled_diagnostic = _run_arm(
        game=game,
        explore_budget=explore_budget,
        total_budget=DIAGNOSTIC_TOTAL_BUDGET,
        object_history_salience=rescaled_prior,
    )

    diverges_default = (
        baseline["actions"][: len(default_treatment["actions"])] != (default_treatment["actions"])
    )
    diverges_rescaled = (
        baseline["actions"][: len(rescaled_diagnostic["actions"])]
        != (rescaled_diagnostic["actions"])
    )

    if diverges_default:
        verdict = "complete: object_history_salience_ab_default_weight_changes_behavior"
    elif diverges_rescaled:
        verdict = (
            "complete: object_history_salience_ab_default_weight_no_op_"
            "rescaled_weight_diverges_needs_retune"
        )
    else:
        verdict = "complete: object_history_salience_ab_no_op_at_either_weight"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "game": game,
        "baseline": baseline,
        "default_weight_treatment": default_treatment,
        "rescaled_weight_diagnostic": rescaled_diagnostic,
        "trajectories_diverge_at_default_weight": bool(diverges_default),
        "trajectories_diverge_at_rescaled_weight": bool(diverges_rescaled),
        "gate_definition": (
            "no OfflineSolver-equivalent states_expanded metric exists for action_prior; "
            "trajectory divergence from baseline is the honest substitute"
        ),
        "solve_provenance": "development_proxy",
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
