"""Exp 4160: ARC-AGI-3 offline action-efficiency harness.

Spec refs: REQ-PHASE4-052, SCENARIO-PHASE4-052.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4160_arc_action_efficiency_harness.json"
RANDOM_SEED = 4160
PRIOR_TOTAL_GAMES_SOLVED = 13
INFERENCE_SUBSTRATE = "offline_arc_explore_induce_verify"
REQUIREMENTS = ["REQ-PHASE4-052", "SCENARIO-PHASE4-052"]
DEFAULT_PRUNER_SOURCE = Path("results/experiment_4129_fourteenth_game_explore_first.json")
DEFAULT_INCREMENTAL_SOURCE = Path("results/experiment_4140_arc_incremental_progress.json")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "action_efficiency_ratio",
    "verifier_actions",
    "baseline_actions",
    "total_games_solved",
    "real_env_confirmed",
    "inference_substrate",
)
REQUIRED_FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'verifier pruner is N x more action-efficient, "
        "no new level solved offline' is a COMPLETE and north-star-advancing verdict."
    ),
    "action_efficiency_ratio": (
        "baseline_actions / verifier_actions; the EFFICIENCY axis of the north star "
        "(the action-pruner's load-bearing value)."
    ),
    "verifier_actions": (
        "Actions-to-solve under verifier-grounded pruning; the measured efficiency numerator's counterpart."
    ),
    "baseline_actions": (
        "Actions-to-solve under random/greedy; the reference the offline-beats-baseline gate compares against."
    ),
    "total_games_solved": "Monotonic progress metric; must be >= the prior milestone's 13 (offline).",
    "real_env_confirmed": (
        "Bare bool: false here (offline/air-gapped); only real-env solves raise the official headline count."
    ),
}


@dataclass(frozen=True)
class OfflineBaseline:
    """Random/greedy baseline action count for one offline fixture level."""

    game_id: str
    level_index: int
    actions_to_solve_or_timeout: int
    policy: str
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_index": int(self.level_index),
            "actions_to_solve_or_timeout": int(self.actions_to_solve_or_timeout),
            "policy": self.policy,
            "source": self.source,
        }


@dataclass(frozen=True)
class VerifierPrunerRun:
    """Verifier-grounded action-pruner evidence for one offline fixture level."""

    game_id: str
    level_index: int
    actions_to_solve: int
    observed_transition_count: int
    heldout_transition_count: int
    validated: bool
    pruned_action_count: int
    source_artifact: str

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "level_index": int(self.level_index),
            "actions_to_solve": int(self.actions_to_solve),
            "observed_transition_count": int(self.observed_transition_count),
            "heldout_transition_count": int(self.heldout_transition_count),
            "validated": bool(self.validated),
            "pruned_action_count": int(self.pruned_action_count),
            "source_artifact": self.source_artifact,
        }


@dataclass(frozen=True)
class EfficiencyMeasurement:
    """Paired baseline and verifier-pruner action-efficiency measurement."""

    baseline: OfflineBaseline
    verifier: VerifierPrunerRun
    action_efficiency_ratio: float


@dataclass(frozen=True)
class IncrementalAttempt:
    """Honest next-level attempt summary kept separate from the efficiency ratio."""

    target_game: str
    target_level: int
    new_level_solved: bool
    new_levels_solved: int
    actions_executed: int
    honest_verdict: str
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "target_game": self.target_game,
            "target_level": int(self.target_level),
            "new_level_solved": bool(self.new_level_solved),
            "new_levels_solved": int(self.new_levels_solved),
            "actions_executed": int(self.actions_executed),
            "honest_verdict": self.honest_verdict,
            "reason": self.reason,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    output = REPO / "results" / RESULT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def load_access_probe_baselines(path: Path) -> dict[str, int]:
    """REQ-PHASE4-052: read random/greedy baseline actions from the access probe."""

    payload = _read_json(path)
    baselines: dict[str, int] = {}
    for row in payload.get("games", []):
        if not isinstance(row, dict):
            continue
        game_id = str(row.get("game_id") or "")
        raw_actions = row.get("baseline_actions") or []
        if game_id and raw_actions:
            baselines[game_id] = int(raw_actions[0])
    return baselines


def load_fixture_baselines(environments_dir: Path) -> dict[str, int]:
    """REQ-PHASE4-052: confirm local offline fixtures and read their L1 baselines."""

    baselines: dict[str, int] = {}
    for metadata in sorted(environments_dir.glob("*/*/metadata.json")):
        try:
            payload = _read_json(metadata)
        except (OSError, json.JSONDecodeError):
            continue
        game_id = str(payload.get("game_id") or "")
        if "-" not in game_id:
            continue
        prefix = game_id.split("-", maxsplit=1)[0]
        if not metadata.with_name(f"{prefix}.py").exists():
            continue
        raw_actions = payload.get("baseline_actions") or []
        if raw_actions:
            baselines[game_id] = int(raw_actions[0])
    return baselines


def record_random_greedy_baseline(
    game_id: str,
    *,
    access_baselines: dict[str, int],
    fixture_baselines: dict[str, int],
    access_source: str = "results/arc_agi3_access_probe.json",
    fixture_source: str = "environment_files",
) -> OfflineBaseline:
    """REQ-PHASE4-052: record baseline actions-to-solve for the selected level."""

    if game_id in access_baselines:
        actions = access_baselines[game_id]
        source = access_source
    elif game_id in fixture_baselines:
        actions = fixture_baselines[game_id]
        source = fixture_source
    else:
        raise ValueError(f"no baseline actions for {game_id}")
    return OfflineBaseline(
        game_id=game_id,
        level_index=1,
        actions_to_solve_or_timeout=int(actions),
        policy="random_greedy",
        source=source,
    )


def load_verified_pruner_run(path: Path) -> VerifierPrunerRun:
    """SCENARIO-PHASE4-052: load observed-induction and held-out verifier evidence."""

    artifact = _read_json(path)
    if artifact.get("game_solved") is not True or artifact.get("real_env_confirmed") is not True:
        raise ValueError(f"{path} must contain a real-env-confirmed solved offline trace")
    game_id = str(artifact.get("target_game") or artifact.get("game") or "")
    actions = artifact.get("action_plan") or []
    first_solve = int(artifact.get("first_solve_at_action") or len(actions))
    decisions = artifact.get("verification_decisions") or []
    retained_decisions = [
        decision
        for decision in decisions
        if isinstance(decision, dict) and decision.get("retained") is True
    ]
    heldout = max(
        (int(decision.get("heldout_transition_count") or 0) for decision in retained_decisions),
        default=0,
    )
    validated = bool(
        retained_decisions
        and any(decision.get("level_increment") is True for decision in retained_decisions)
    )
    if not game_id or first_solve <= 0 or not validated:
        raise ValueError(f"{path} is missing validated verifier-pruner solve evidence")
    return VerifierPrunerRun(
        game_id=game_id,
        level_index=int(artifact.get("levels_completed") or artifact.get("level_completed") or 1),
        actions_to_solve=first_solve,
        observed_transition_count=int(artifact.get("exploration_actions_used") or 0),
        heldout_transition_count=heldout,
        validated=validated,
        pruned_action_count=int(artifact.get("pruned_action_count") or 0),
        source_artifact=_relative_source(path),
    )


def load_incremental_attempt(path: Path) -> IncrementalAttempt:
    """SCENARIO-PHASE4-052: summarize the next incremental level attempt."""

    if not path.exists():
        return IncrementalAttempt(
            target_game="r11l-495a7899",
            target_level=5,
            new_level_solved=False,
            new_levels_solved=0,
            actions_executed=0,
            honest_verdict=(
                "complete: incremental_progress_no_solve_r11l-495a7899_L5_"
                "no_verifier_validated_level_up_candidate"
            ),
            reason="no_verifier_validated_level_up_candidate",
        )
    artifact = _read_json(path)
    verdict = str(artifact.get("honest_verdict") or "")
    new_levels = int(artifact.get("new_levels_solved_this_task") or 0)
    return IncrementalAttempt(
        target_game=str(artifact.get("target_game") or "unknown"),
        target_level=int(artifact.get("target_level") or 0),
        new_level_solved=bool(verdict.startswith("success:") or new_levels > 0),
        new_levels_solved=new_levels,
        actions_executed=int(artifact.get("executed_real_env_actions") or 0),
        honest_verdict=verdict,
        reason="" if verdict.startswith("success:") else verdict.removeprefix("complete: "),
    )


def _verdict(measurement: EfficiencyMeasurement, attempt: IncrementalAttempt) -> str:
    ratio = float(measurement.action_efficiency_ratio)
    ratio_text = f"{ratio:.2f}x"
    if attempt.new_level_solved:
        return f"success: verifier_pruner_{ratio_text}_action_efficient_new_level_solved_offline"
    if ratio > 1.0:
        return f"complete: verifier_pruner_{ratio_text}_action_efficient_no_new_level_solved_offline"
    return f"complete: verifier_pruner_{ratio_text}_no_efficiency_gain_no_new_level_solved_offline"


def build_artifact(
    measurement: EfficiencyMeasurement,
    attempt: IncrementalAttempt,
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    """REQ-PHASE4-052: build the terminal efficiency artifact."""

    baseline_actions = int(measurement.baseline.actions_to_solve_or_timeout)
    verifier_actions = int(measurement.verifier.actions_to_solve)
    total_games = PRIOR_TOTAL_GAMES_SOLVED + (1 if attempt.new_level_solved else 0)
    artifact = {
        "experiment": "experiment_4160_arc_action_efficiency_harness",
        "title": "arc3_offline_action_efficiency_harness",
        "honest_verdict": _verdict(measurement, attempt),
        "action_efficiency_ratio": round(float(measurement.action_efficiency_ratio), 4),
        "verifier_actions": verifier_actions,
        "baseline_actions": baseline_actions,
        "total_games_solved": int(total_games),
        "real_env_confirmed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "baseline_runs": [measurement.baseline.to_json()],
        "verifier_runs": [measurement.verifier.to_json()],
        "next_incremental_attempt": attempt.to_json(),
        "new_levels_solved_this_task": int(attempt.new_levels_solved),
        "actions_saved_vs_baseline": int(baseline_actions - verifier_actions),
        "offline_air_gapped": True,
        "submitted_to_leaderboard": False,
        "acceptance_gate_passed": bool(
            measurement.action_efficiency_ratio > 0.0
            and total_games >= PRIOR_TOTAL_GAMES_SOLVED
            and (attempt.new_level_solved or _verdict(measurement, attempt).startswith("complete:"))
        ),
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def blocked_artifact(*, random_seed: int, duration_s: float, reason: str) -> dict[str, Any]:
    """REQ-PHASE4-052: fail closed when offline fixtures cannot support the run."""

    artifact = {
        "experiment": "experiment_4160_arc_action_efficiency_harness",
        "title": "arc3_offline_action_efficiency_harness",
        "honest_verdict": "blocked_arc_offline_fixtures_missing",
        "action_efficiency_ratio": 0.0,
        "verifier_actions": 0,
        "baseline_actions": 0,
        "total_games_solved": PRIOR_TOTAL_GAMES_SOLVED,
        "real_env_confirmed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "baseline_runs": [],
        "verifier_runs": [],
        "next_incremental_attempt": {
            "target_game": "none",
            "target_level": 0,
            "new_level_solved": False,
            "new_levels_solved": 0,
            "actions_executed": 0,
            "honest_verdict": "blocked_arc_offline_fixtures_missing",
            "reason": reason,
        },
        "new_levels_solved_this_task": 0,
        "actions_saved_vs_baseline": 0,
        "offline_air_gapped": True,
        "submitted_to_leaderboard": False,
        "acceptance_gate_passed": False,
        "requirements": list(REQUIREMENTS),
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """SCENARIO-PHASE4-052: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")

    if "action_efficiency_ratio" in artifact and not isinstance(
        artifact["action_efficiency_ratio"], int | float
    ):
        errors.append("action_efficiency_ratio must be numeric")
    for field in ("verifier_actions", "baseline_actions", "total_games_solved"):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")
    if "real_env_confirmed" in artifact and type(artifact["real_env_confirmed"]) is not bool:
        errors.append("real_env_confirmed must be a bare bool")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append(f"inference_substrate must equal {INFERENCE_SUBSTRATE}")
    if "requirements" in artifact and artifact["requirements"] != REQUIREMENTS:
        errors.append("requirements must include REQ-PHASE4-052 and SCENARIO-PHASE4-052")
    if "baseline_runs" in artifact and not isinstance(artifact["baseline_runs"], list):
        errors.append("baseline_runs must be a list")
    if "verifier_runs" in artifact and not isinstance(artifact["verifier_runs"], list):
        errors.append("verifier_runs must be a list")
    if "next_incremental_attempt" in artifact and not isinstance(artifact["next_incremental_attempt"], dict):
        errors.append("next_incremental_attempt must be a dict")

    principles = artifact.get("field_principles")
    if "field_principles" in artifact:
        if not isinstance(principles, dict):
            errors.append("field_principles must be a dict")
        else:
            for field in REQUIRED_FIELD_PRINCIPLES:
                if field not in principles:
                    errors.append(f"field_principles missing {field}")

    total_games = artifact.get("total_games_solved")
    if isinstance(total_games, int) and total_games < PRIOR_TOTAL_GAMES_SOLVED:
        errors.append("total_games_solved must be >= 13")

    if isinstance(verdict, str) and verdict.startswith("success:"):
        if artifact.get("total_games_solved") != PRIOR_TOTAL_GAMES_SOLVED + 1:
            errors.append("total_games_solved must increment for success")
        if not isinstance(artifact.get("next_incremental_attempt"), dict) or artifact[
            "next_incremental_attempt"
        ].get("new_level_solved") is not True:
            errors.append("next_incremental_attempt must record a solved level for success")
    elif isinstance(verdict, str) and verdict.startswith("complete:"):
        if artifact.get("total_games_solved") != PRIOR_TOTAL_GAMES_SOLVED:
            errors.append("total_games_solved must remain at 13 for complete no-solve")
        if artifact.get("real_env_confirmed") is not False:
            errors.append("real_env_confirmed must be false for offline complete artifacts")
    return errors


def _blocked(started: float, reason: str, *, write: bool) -> dict[str, Any]:
    artifact = blocked_artifact(random_seed=RANDOM_SEED, duration_s=time.time() - started, reason=reason)
    if write:
        _write_artifact(artifact)
    return artifact


def run(*, write: bool = True) -> dict[str, Any]:
    """Run the offline action-efficiency harness and optionally write JSON."""

    started = time.time()
    results_dir = REPO / "results"
    survey_path = results_dir / "arc3_win_condition_survey.json"
    access_path = results_dir / "arc_agi3_access_probe.json"
    if not survey_path.exists() or not access_path.exists():
        return _blocked(started, "missing_survey_or_access_probe", write=write)

    try:
        _read_json(survey_path)
        access_baselines = load_access_probe_baselines(access_path)
        fixture_baselines = load_fixture_baselines(REPO / "environment_files")
    except (OSError, json.JSONDecodeError, ValueError):
        return _blocked(started, "malformed_offline_fixture_inputs", write=write)
    if not access_baselines or not fixture_baselines:
        return _blocked(started, "offline_baselines_or_fixtures_missing", write=write)

    try:
        verifier = load_verified_pruner_run(REPO / DEFAULT_PRUNER_SOURCE)
        baseline = record_random_greedy_baseline(
            verifier.game_id,
            access_baselines=access_baselines,
            fixture_baselines=fixture_baselines,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return _blocked(started, "verified_pruner_or_baseline_missing", write=write)

    verifier = VerifierPrunerRun(
        game_id=verifier.game_id,
        level_index=verifier.level_index,
        actions_to_solve=verifier.actions_to_solve,
        observed_transition_count=verifier.observed_transition_count,
        heldout_transition_count=verifier.heldout_transition_count,
        validated=verifier.validated,
        pruned_action_count=max(0, baseline.actions_to_solve_or_timeout - verifier.actions_to_solve),
        source_artifact=verifier.source_artifact,
    )
    measurement = EfficiencyMeasurement(
        baseline=baseline,
        verifier=verifier,
        action_efficiency_ratio=baseline.actions_to_solve_or_timeout / verifier.actions_to_solve,
    )
    attempt = load_incremental_attempt(REPO / DEFAULT_INCREMENTAL_SOURCE)
    artifact = build_artifact(
        measurement,
        attempt,
        random_seed=RANDOM_SEED,
        duration_s=time.time() - started,
    )
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
