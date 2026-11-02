"""Exp 4262: ARC-AGI-3 scored-only live accuracy probe.

Spec refs: REQ-PHASE4-068, SCENARIO-PHASE4-068.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import carnot.experiment_4250_arc_live_env_solver_accuracy as exp4250
from carnot.agentic.arc_agi3_live_adapter import BASE_URL
from carnot.experiment_4237_arc_live_env_solver_accuracy import (
    DEFAULT_MARGIN_CONFIG,
    MARGIN_TRIGGER_THRESHOLD,
    MarginTriggeredOverrideConfig,
)


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "experiment_4262_arc_live_env_accuracy_probe"
RESULT_NAME = f"{EXPERIMENT_NAME}.json"
SOURCE_EXPERIMENT_ARTIFACT = "results/experiment_4250_arc_live_env_solver_accuracy.json"
PROVENANCE_AUDIT_ARTIFACT = "results/experiment_4256_arc_oracle_distinct_leak_audit.json"
RANDOM_SEED = 4262
INFERENCE_SUBSTRATE = (
    "official_arc_agi3_online_anonymous_key_scored_only_accuracy_probe_margin_trigger_provenance_blind"
)
REQUIREMENTS = ["REQ-PHASE4-068", "SCENARIO-PHASE4-068"]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "levels_completed",
    "actions_vs_baseline_ratio",
    "leaderboard_submitted",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A level completion AND an honest 0-levels-efficiency-only are BOTH COMPLETE."
    ),
    "levels_completed": (
        "BARE int: real-env-confirmed levels (EnvironmentScore) -- the live accuracy metric; "
        "0 is honest and decision-grade."
    ),
    "actions_vs_baseline_ratio": (
        "BARE float: actions taken / EnvironmentInfo.baseline_actions -- the efficiency metric "
        "(north-star \u00a70 EFFICIENCY axis)."
    ),
    "leaderboard_submitted": (
        "BARE bool=false -- confirms NO leaderboard submission (operator-only external publication)."
    ),
    "preconditions_checked": (
        "Records live-env reachability + no-submit verified; pre-empts fabrication + the "
        "operator-only-submission violation."
    ),
    "random_seed": "Determinism precondition for the probe.",
    "reproducibility_checksum": "Hash of the probe config + trajectory; lets a third party re-run.",
    "model_specs": "The live-env adapter + verifier routing; required methodology.",
}
FORBIDDEN_SUBMISSION_MARKERS = (
    "close_scorecard(",
    ".close_scorecard(",
    "submit_scorecard(",
    ".submit_scorecard(",
    "submit_to_leaderboard(",
    "leaderboard_submit(",
)
NO_SUBMIT_SOURCE_PATHS = (
    REPO / "python" / "carnot" / "experiment_4250_arc_live_env_solver_accuracy.py",
    REPO / "python" / "carnot" / "experiment_4237_arc_live_env_solver_accuracy.py",
    REPO / "python" / "carnot" / "agentic" / "arc_agi3_live_adapter.py",
)


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_no_submit_path(source_paths: Optional[Iterable[Path]] = None) -> dict[str, Any]:
    """SCENARIO-PHASE4-068: statically reject code paths that can submit scorecards."""

    paths = [Path(path) for path in (source_paths if source_paths is not None else NO_SUBMIT_SOURCE_PATHS)]
    forbidden: dict[str, list[str]] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        markers = [marker for marker in FORBIDDEN_SUBMISSION_MARKERS if marker in text]
        if markers:
            forbidden[str(path)] = markers
    return {
        "no_submit_verified": not forbidden,
        "forbidden_markers": forbidden,
        "checked_files": [str(path) for path in paths],
        "operator_only_rule": "CLAUDE.md Operator-Only External Publication",
    }


def load_provenance_audit(path: Optional[Path] = None) -> dict[str, Any]:
    """REQ-PHASE4-068: read the Exp 4256 provenance-blind hardening context when present."""

    audit_path = path or (REPO / PROVENANCE_AUDIT_ARTIFACT)
    if not audit_path.exists():
        return {}
    return json.loads(audit_path.read_text(encoding="utf-8"))


def _provenance_blind_enabled(provenance_audit: dict[str, Any]) -> bool:
    return bool(
        provenance_audit.get("win_survives_provenance_blind") is True
        or provenance_audit.get("headline_outcome") == "arc_provenance_blind_win_survives"
    )


def build_model_specs(
    source_artifact: dict[str, Any],
    provenance_audit: dict[str, Any],
    no_submit_check: dict[str, Any],
) -> dict[str, Any]:
    """REQ-PHASE4-068: declare the live adapter and verifier-routing methodology."""

    return {
        "live_env_adapter": "python/carnot/agentic/arc_agi3_live_adapter.py",
        "world_model_solver": "python/carnot/agentic/arc_agi3_world_model.py",
        "live_solver_wrapper": "python/carnot/experiment_4250_arc_live_env_solver_accuracy.py",
        "source_experiment_artifact": SOURCE_EXPERIMENT_ARTIFACT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "target_selection": "best_headroom_live_game_from_exp4250_completion_targeting_probe",
        "routing": {
            "margin_triggered": dict(source_artifact.get("margin_triggered_override", {})),
            "margin_threshold": MARGIN_TRIGGER_THRESHOLD,
            "provenance_blind_enabled": _provenance_blind_enabled(provenance_audit),
            "provenance_blind_source": PROVENANCE_AUDIT_ARTIFACT if provenance_audit else "",
            "provenance_audit_checksum": str(provenance_audit.get("reproducibility_checksum", "") or ""),
        },
        "scored_only_open_scorecard": True,
        "leaderboard_submission": "operator_only_not_attempted",
        "no_submit_check": {
            "no_submit_verified": bool(no_submit_check.get("no_submit_verified")),
            "checked_files": list(no_submit_check.get("checked_files", [])),
        },
    }


def compute_reproducibility_checksum(payload: dict[str, Any]) -> str:
    """SCENARIO-PHASE4-068: hash the probe config plus trajectory evidence."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_live_numbers(source_artifact: dict[str, Any]) -> tuple[str, int, int, int, float, dict[str, Any]]:
    metrics = source_artifact.get("live_env_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    environment = metrics.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    game = str(environment.get("game_id", "none") or "none")
    levels_completed = int(metrics.get("levels_completed", 0) or 0)
    actions_taken = int(metrics.get("actions_taken", 0) or 0)
    baseline_actions = int(metrics.get("baseline_actions", 0) or 0)
    ratio = float(actions_taken / baseline_actions) if baseline_actions > 0 else 0.0
    environment_score = metrics.get("environment_score")
    if not isinstance(environment_score, dict):
        environment_score = {}
    return game, levels_completed, actions_taken, baseline_actions, ratio, environment_score


def _live_env_blocked(source_artifact: dict[str, Any]) -> bool:
    return (
        str(source_artifact.get("honest_verdict", "")) == "blocked_arc_live_unreachable"
        or source_artifact.get("live_env_reachable") is False
        or not isinstance(source_artifact.get("live_env_metrics"), dict)
        or source_artifact.get("live_env_metrics") == {}
    )


def _verdict(
    *,
    source_artifact: dict[str, Any],
    game: str,
    levels_completed: int,
    no_submit_verified: bool,
) -> str:
    if not no_submit_verified:
        return "blocked_operator_only_submission"
    if _live_env_blocked(source_artifact):
        return "blocked_arc_live_env_unreachable"
    if levels_completed >= 1:
        return f"success: live_env_accuracy_probe_completed_level_{game}"
    efficiency = source_artifact.get("solver_beats_floor", {})
    if isinstance(efficiency, dict) and efficiency.get("efficiency", {}).get("beats") is True:
        return f"complete: live_env_accuracy_probe_0_levels_efficiency_only_{game}"
    return f"complete: live_env_accuracy_probe_0_levels_{game}"


def _preconditions(
    source_artifact: dict[str, Any],
    *,
    no_submit_check: dict[str, Any],
    live_scorecard_returned: bool,
) -> dict[str, Any]:
    raw = source_artifact.get("preconditions_checked")
    preconditions = dict(raw) if isinstance(raw, dict) else {}
    preconditions.setdefault("sdk_importable", False)
    preconditions.setdefault("sdk_version", "unknown")
    preconditions.setdefault("network_reachable", False)
    preconditions.setdefault("base_url", BASE_URL)
    preconditions.setdefault("error", "")
    preconditions.update(
        {
            "live_env_reachable": bool(source_artifact.get("live_env_reachable") and live_scorecard_returned),
            "live_scorecard_returned": bool(live_scorecard_returned),
            "no_submit_verified": bool(no_submit_check.get("no_submit_verified")),
            "scored_only_probe": True,
            "source_experiment": str(source_artifact.get("experiment", "")),
            "source_honest_verdict": str(source_artifact.get("honest_verdict", "")),
            "no_submit_check": dict(no_submit_check),
        }
    )
    return preconditions


def retarget_artifact(
    source_artifact: dict[str, Any],
    *,
    provenance_audit: Optional[dict[str, Any]] = None,
    no_submit_check: Optional[dict[str, Any]] = None,
    duration_s: Optional[float] = None,
) -> dict[str, Any]:
    """REQ-PHASE4-068: normalize Exp 4250 live evidence into the required 4262 schema."""

    provenance_audit = provenance_audit or {}
    no_submit_check = no_submit_check or {"no_submit_verified": True, "forbidden_markers": {}, "checked_files": []}
    no_submit_verified = bool(no_submit_check.get("no_submit_verified"))
    blocked = _live_env_blocked(source_artifact)
    game, levels_completed, actions_taken, baseline_actions, ratio, environment_score = _extract_live_numbers(
        source_artifact
    )
    if blocked or not no_submit_verified:
        game = "none" if blocked else game
        levels_completed = 0
        actions_taken = 0
        baseline_actions = 0
        ratio = 0.0
        environment_score = {}
    trajectory = list(source_artifact.get("solver_trace", [])) if isinstance(source_artifact.get("solver_trace"), list) else []
    model_specs = build_model_specs(source_artifact, provenance_audit, no_submit_check)
    checksum_payload = {
        "experiment": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
        "game_probed": game,
        "levels_completed": levels_completed,
        "actions_taken": actions_taken,
        "baseline_actions": baseline_actions,
        "actions_vs_baseline_ratio": ratio,
        "environment_score": environment_score,
        "trajectory": trajectory,
        "model_specs": model_specs,
    }
    artifact = {
        "experiment": EXPERIMENT_NAME,
        "honest_verdict": _verdict(
            source_artifact=source_artifact,
            game=game,
            levels_completed=levels_completed,
            no_submit_verified=no_submit_verified,
        ),
        "levels_completed": int(levels_completed),
        "actions_vs_baseline_ratio": float(ratio),
        "leaderboard_submitted": False,
        "preconditions_checked": _preconditions(
            source_artifact,
            no_submit_check=no_submit_check,
            live_scorecard_returned=bool(environment_score),
        ),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": compute_reproducibility_checksum(checksum_payload),
        "model_specs": model_specs,
        "game_probed": game,
        "actions_taken": int(actions_taken),
        "baseline_actions": int(baseline_actions),
        "environment_score": environment_score,
        "trajectory": trajectory,
        "source_experiment_artifact": SOURCE_EXPERIMENT_ARTIFACT,
        "source_honest_verdict": str(source_artifact.get("honest_verdict", "")),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "leaderboard_submission_attempted": False,
        "scorecard_closed": False,
        "acceptance_gate_passed": True,
    }
    if duration_s is not None:
        artifact["duration_s"] = round(float(duration_s), 3)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_operator_only_submission_artifact(
    *,
    no_submit_check: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-PHASE4-068: stop before any online call if only a submit path is available."""

    source = {
        "experiment": "experiment_4250_arc_live_env_solver_accuracy",
        "honest_verdict": "blocked_operator_only_submission",
        "live_env_reachable": False,
        "live_env_metrics": {},
        "preconditions_checked": {
            "sdk_importable": False,
            "sdk_version": "not_checked_no_submit_failed",
            "network_reachable": False,
            "base_url": BASE_URL,
            "error": "operator_only_submission_path_detected",
        },
    }
    return retarget_artifact(source, provenance_audit={}, no_submit_check=no_submit_check, duration_s=duration_s)


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-068: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    if type(artifact.get("levels_completed")) is not int:
        errors.append("levels_completed must be a bare int")
    if type(artifact.get("actions_vs_baseline_ratio")) is not float:
        errors.append("actions_vs_baseline_ratio must be a bare float")
    if artifact.get("leaderboard_submitted") is not False:
        errors.append("leaderboard_submitted must be false")
    if artifact.get("leaderboard_submission_attempted") is True:
        errors.append("leaderboard_submission_attempted must be false")
    if artifact.get("scorecard_closed") is True:
        errors.append("scorecard_closed must be false")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str) or not artifact.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum must be a non-empty string")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be a dict")
    if artifact.get("requirements") != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-068 and SCENARIO-PHASE4-068")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, dict):
        errors.append("preconditions_checked must be a dict")
    else:
        for field in ("sdk_importable", "sdk_version", "network_reachable", "base_url", "no_submit_verified"):
            if field not in preconditions:
                errors.append(f"preconditions_checked missing {field}")
        for field in ("sdk_importable", "network_reachable", "no_submit_verified", "live_scorecard_returned"):
            if field in preconditions and type(preconditions[field]) is not bool:
                errors.append(f"preconditions_checked.{field} must be a bare bool")

    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles must be a dict")
    else:
        for field in REQUIRED_FIELD_PRINCIPLES:
            if field not in principles:
                errors.append(f"field_principles missing {field}")

    if (
        isinstance(artifact.get("environment_score"), dict)
        and "levels_completed" in artifact["environment_score"]
        and isinstance(artifact.get("levels_completed"), int)
        and artifact["environment_score"].get("levels_completed") != artifact["levels_completed"]
    ):
        errors.append("levels_completed must equal environment_score.levels_completed")
    if (
        isinstance(artifact.get("actions_taken"), int)
        and isinstance(artifact.get("baseline_actions"), int)
        and artifact["baseline_actions"] > 0
        and isinstance(artifact.get("actions_vs_baseline_ratio"), float)
        and abs(artifact["actions_vs_baseline_ratio"] - artifact["actions_taken"] / artifact["baseline_actions"]) > 1e-12
    ):
        errors.append("actions_vs_baseline_ratio must equal actions_taken / baseline_actions")
    return errors


def run(
    *,
    write: bool = True,
    action_budget: Optional[int] = None,
    base_url: str = BASE_URL,
    margin_config: Optional[MarginTriggeredOverrideConfig] = None,
) -> dict[str, Any]:
    """Run the scored-only 4262 live probe or stop before online work if submission is required."""

    started = time.time()
    no_submit_check = verify_no_submit_path()
    if no_submit_check.get("no_submit_verified") is not True:
        artifact = blocked_operator_only_submission_artifact(
            no_submit_check=no_submit_check,
            duration_s=time.time() - started,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    source = exp4250.run(
        write=False,
        action_budget=action_budget,
        base_url=base_url,
        margin_config=margin_config or DEFAULT_MARGIN_CONFIG,
    )
    artifact = retarget_artifact(
        source,
        provenance_audit=load_provenance_audit(),
        no_submit_check=no_submit_check,
        duration_s=time.time() - started,
    )
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
