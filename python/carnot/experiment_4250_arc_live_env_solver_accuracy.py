"""Exp 4250: ARC-AGI-3 live completion-targeting solver accuracy rerun.

Spec refs: REQ-PHASE4-066, SCENARIO-PHASE4-066.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Optional

import carnot.experiment_4237_arc_live_env_solver_accuracy as exp4237
from carnot.agentic.arc_agi3_live_adapter import BASE_URL
from carnot.experiment_4237_arc_live_env_solver_accuracy import (
    MARGIN_TRIGGER_THRESHOLD,
    MarginTriggeredOverrideConfig,
)


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_4250_arc_live_env_solver_accuracy"
RESULT_NAME = f"{EXPERIMENT_NAME}.json"
SOURCE_EXPERIMENT_ARTIFACT = "results/experiment_4237_arc_live_env_solver_accuracy.json"
RANDOM_SEED = 4250
INFERENCE_SUBSTRATE = (
    "official_arc_agi3_online_anonymous_key_margin_trigger_solver_accuracy_completion_probe"
)
REQUIREMENTS = ["REQ-PHASE4-066", "SCENARIO-PHASE4-066"]
REQUIRED_ARTIFACT_FIELDS = exp4237.REQUIRED_ARTIFACT_FIELDS
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'solver completes 0 levels but is efficient' or "
        "'blocked_arc_live_unreachable' is a COMPLETE grounding verdict."
    ),
    "solver_completes_level": (
        "BARE bool: levels_completed>=1 on the live env -- the ACCURACY win exp4237 "
        "lacked; the north-star's primary axis."
    ),
    "solver_beats_floor": (
        "{accuracy: solver_score vs floor_score, efficiency: solver_actions vs "
        "floor_actions} -- the real-env read on the two north-star axes vs the random floor."
    ),
    "live_env_metrics": (
        "{score, levels_completed, actions_taken, baseline_actions} from the live "
        "EnvironmentScore -- falsifiable real-env evidence, not synthetic-scaffold."
    ),
    "no_leaderboard_submission": (
        "BARE bool: zero scorecards submitted (Operator-Only External Publication; "
        "the online quota gate)."
    ),
    "preconditions_checked": (
        "Records the SDK + network reachability checks; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
}


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-066: validate the 4250 terminal artifact contract."""

    errors: list[str] = []
    if artifact.get("requirements") != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-066 and SCENARIO-PHASE4-066")

    compat = copy.deepcopy(artifact)
    compat["requirements"] = exp4237.REQUIREMENTS
    errors.extend(exp4237.artifact_schema_errors(compat))
    return errors


def retarget_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """REQ-PHASE4-066: preserve Exp 4237 live evidence under the Exp 4250 contract."""

    out = copy.deepcopy(artifact)
    out.update(
        {
            "experiment": EXPERIMENT_NAME,
            "title": "arc3_live_env_solver_accuracy_margin_trigger_completion_probe",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
            "requirements": list(REQUIREMENTS),
            "random_seed": RANDOM_SEED,
            "source_experiment_artifact": SOURCE_EXPERIMENT_ARTIFACT,
        }
    )
    out.setdefault("scorecard_closed", False)
    errors = artifact_schema_errors(out)
    if errors:
        raise ValueError("; ".join(errors))
    return out


def run(
    *,
    write: bool = True,
    action_budget: Optional[int] = None,
    base_url: str = BASE_URL,
    margin_config: Optional[MarginTriggeredOverrideConfig] = None,
) -> dict[str, Any]:
    """Run Exp 4250 by reusing the Exp 4237 live solver path and retargeting its artifact."""

    upstream = exp4237.run(
        write=False,
        action_budget=action_budget,
        base_url=base_url,
        margin_config=margin_config,
    )
    artifact = retarget_artifact(upstream)
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--action-budget", type=int, default=None)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--learned-margin", type=float, default=0.20)
    parser.add_argument("--verifier-margin", type=float, default=0.20)
    parser.add_argument("--margin-threshold", type=float, default=MARGIN_TRIGGER_THRESHOLD)
    args = parser.parse_args()
    artifact = run(
        write=not args.no_write,
        action_budget=args.action_budget,
        base_url=args.base_url,
        margin_config=MarginTriggeredOverrideConfig(
            learned_margin=args.learned_margin,
            verifier_margin=args.verifier_margin,
            margin_threshold=args.margin_threshold,
        ),
    )
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
