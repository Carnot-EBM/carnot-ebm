"""Exp 1398 CPU-only NGRPO Advantage Calibration theory probe.

Spec: REQ-LEARN-1398, SCENARIO-LEARN-1398.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1398_ngrpo_theory_probe"
SCHEMA = "ngrpo_theory_probe_v1"
OUTPUT_FILE = "experiment_1398_ngrpo_theory_probe.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_EXP1383_PATH = (
    REPO_ROOT / "results" / "experiment_1383_grpo_v7_jury_rl_formal_verifier_rewards.json"
)
REAL_ROLLOUT_COUNT = 4
VIRTUAL_MAX_REWARD = 1.0
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp1383_rollout_data_used",
    "original_resZero_advantage_variance",
    "ngrpo_virtual_sample_reward",
    "ngrpo_augmented_advantage_variance",
    "ngrpo_advantage_calibration_verified",
    "ngrpo_expected_gradient_magnitude",
    "theory_supports_exp1393",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1398-1: persist the bootstrap artifact before source loading.

    This probe is intentionally small, but the bootstrap write still matters:
    it lets the conductor distinguish "not started" from "started, then
    interrupted while reading the prior artifact or writing the terminal JSON."
    """

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-LEARN-1398", "SCENARIO-LEARN-1398"],
            "source_experiments": ["exp1383"],
        },
        "honest_verdict": "in_progress",
    }
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def load_exp1383_artifact(path: Path | str = DEFAULT_EXP1383_PATH) -> dict[str, Any]:
    """Load the Exp 1383 artifact used as the fixed rollout-data source."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_exp1383_rollouts(exp1383_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-LEARN-1398-2: summarize Exp 1383's all-UNKNOWN reward collapse.

    The theory probe needs the recorded reward distribution, not generated
    tokens.  Summarizing the source artifact up front makes the final result
    auditable: a reader can see exactly which Exp 1383 failure mode the closed
    form NGRPO calculation was applied to.
    """

    training_rows = _rows(exp1383_artifact, "training_reward_rows")
    heldout_rows = _rows(exp1383_artifact, "heldout_evaluation_rows")
    training_rewards = [
        float(reward)
        for row in training_rows
        for reward in _sequence(row.get("rewards") or row.get("raw_rewards"))
    ]
    training_answers = [
        str(answer) for row in training_rows for answer in _sequence(row.get("rollout_answers"))
    ]
    heldout_answers = [
        str(answer) for row in heldout_rows for answer in _sequence(row.get("rollout_answers"))
    ]
    all_recorded_answers = [*training_answers, *heldout_answers]
    rollout_counts = [
        len(_sequence(row.get("rollout_answers")))
        for row in training_rows
        if _sequence(row.get("rollout_answers"))
    ]
    per_group_count = (
        rollout_counts[0] if rollout_counts and len(set(rollout_counts)) == 1 else None
    )

    return {
        "source_experiment": exp1383_artifact.get("experiment"),
        "source_run_date": exp1383_artifact.get("run_date"),
        "source_honest_verdict": exp1383_artifact.get("honest_verdict"),
        "formal_reward_pass_rate": _float(exp1383_artifact.get("formal_reward_pass_rate")),
        "grpo_v7_improvement_pp": _float(exp1383_artifact.get("grpo_v7_improvement_pp")),
        "training_group_count": len(training_rows),
        "rollouts_per_training_group": per_group_count,
        "training_reward_distribution": _float_distribution(training_rewards),
        "training_rollout_answer_distribution": _string_distribution(training_answers),
        "heldout_rollout_answer_distribution": _string_distribution(heldout_answers),
        "all_training_rewards_zero": bool(training_rewards)
        and all(abs(reward) <= 1e-12 for reward in training_rewards),
        "all_rollouts_unknown": bool(all_recorded_answers)
        and all(answer == "UNKNOWN" for answer in all_recorded_answers),
    }


def simulate_reszero_advantages(rewards: Sequence[float]) -> dict[str, Any]:
    """REQ-LEARN-1398-3: simulate ResZero centering for a same-reward group."""

    real_rewards = [float(reward) for reward in rewards]
    mean_reward = _mean(real_rewards)
    advantages = [round(reward - mean_reward, 12) for reward in real_rewards]
    variance = _population_variance(advantages)
    return {
        "real_rewards": [round(reward, 12) for reward in real_rewards],
        "mean_reward": round(mean_reward, 12),
        "advantages": advantages,
        "advantage_variance": variance,
        "zero_gradient_signal": variance == 0.0
        and all(abs(value) <= 1e-12 for value in advantages),
    }


def simulate_ngrpo_advantage_calibration(
    rewards: Sequence[float],
    *,
    virtual_reward: float = VIRTUAL_MAX_REWARD,
) -> dict[str, Any]:
    """REQ-LEARN-1398-4: inject one virtual max-reward sample before centering.

    The virtual sample changes only the group mean.  It is included in the
    augmented advantage variance because that is the symmetry-breaking signal,
    but callers can keep real rollout advantages separate from the virtual
    advantage to avoid treating the virtual sample as generated data.
    """

    real_rewards = [float(reward) for reward in rewards]
    virtual = float(virtual_reward)
    augmented_rewards = [*real_rewards, virtual]
    augmented_mean = _mean(augmented_rewards)
    real_advantages = [round(reward - augmented_mean, 12) for reward in real_rewards]
    virtual_advantage = round(virtual - augmented_mean, 12)
    augmented_advantages = [*real_advantages, virtual_advantage]
    return {
        "real_rewards": [round(reward, 12) for reward in real_rewards],
        "virtual_reward": round(virtual, 12),
        "augmented_rewards": [round(reward, 12) for reward in augmented_rewards],
        "augmented_mean_reward": round(augmented_mean, 12),
        "real_advantages": real_advantages,
        "virtual_advantage": virtual_advantage,
        "augmented_advantages": augmented_advantages,
        "augmented_advantage_variance": _population_variance(augmented_advantages),
    }


def build_theory_probe_artifact(
    exp1383_artifact: Mapping[str, Any],
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the Exp 1398 terminal artifact from Exp 1383 rollout rewards."""

    source_summary = summarize_exp1383_rollouts(exp1383_artifact)
    real_rewards = _representative_training_rewards(exp1383_artifact)
    reszero = simulate_reszero_advantages(real_rewards)
    ngrpo = simulate_ngrpo_advantage_calibration(real_rewards, virtual_reward=VIRTUAL_MAX_REWARD)
    augmented_variance = float(ngrpo["augmented_advantage_variance"])
    verified = augmented_variance > 0.0
    verdict = (
        "cpu_only_theory_probe_supports_exp1393_nonzero_advantage_signal"
        if verified
        else "cpu_only_theory_probe_no_nonzero_advantage_signal"
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-LEARN-1398", "SCENARIO-LEARN-1398"],
            "source_experiments": ["exp1383"],
            "cpu_only": True,
            "model_inference_performed": False,
        },
        "started_at": _utc_now_iso(),
        "finished_at": _utc_now_iso(),
        "status": "complete",
        "exp1383_rollout_data_used": source_summary,
        "simulation_settings": {
            "real_rollout_count": len(real_rewards),
            "real_rewards": [round(reward, 12) for reward in real_rewards],
            "virtual_sample_count": 1,
            "advantage_variance_definition": "population_variance",
        },
        "resZero_simulation": reszero,
        "ngrpo_simulation": ngrpo,
        "original_resZero_advantage_variance": float(reszero["advantage_variance"]),
        "ngrpo_virtual_sample_reward": float(ngrpo["virtual_reward"]),
        "ngrpo_augmented_advantage_variance": augmented_variance,
        "ngrpo_advantage_calibration_verified": verified,
        "ngrpo_expected_gradient_magnitude": augmented_variance,
        "gradient_magnitude_estimator": "proportional_to_augmented_advantage_variance",
        "theory_supports_exp1393": verified,
        "headline_result_allowed": False,
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    exp1383_path: Path | str = DEFAULT_EXP1383_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1398-1/5: write bootstrap, compute theory probe, write final JSON."""

    output = Path(output_path)
    write_in_progress_artifact(
        output,
        project_root=project_root,
        run_date=run_date,
        write_observer=write_observer,
    )
    exp1383_artifact = load_exp1383_artifact(exp1383_path)
    artifact = build_theory_probe_artifact(
        exp1383_artifact,
        project_root=project_root,
        run_date=run_date,
    )
    _write_json(output, artifact, write_observer=write_observer)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields required by the Exp 1398 task contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS).difference(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] != "complete":
        raise AssertionError("terminal artifact status must be complete")
    if not isinstance(artifact["exp1383_rollout_data_used"], Mapping):
        raise AssertionError("exp1383_rollout_data_used must be an object")
    for field_name in (
        "original_resZero_advantage_variance",
        "ngrpo_virtual_sample_reward",
        "ngrpo_augmented_advantage_variance",
        "ngrpo_expected_gradient_magnitude",
    ):
        if not isinstance(artifact[field_name], (int, float)):
            raise AssertionError(f"{field_name} must be numeric")
    for field_name in (
        "ngrpo_advantage_calibration_verified",
        "theory_supports_exp1393",
    ):
        if not isinstance(artifact[field_name], bool):
            raise AssertionError(f"{field_name} must be boolean")
    verified = bool(artifact["ngrpo_augmented_advantage_variance"] > 0.0)
    if artifact["ngrpo_advantage_calibration_verified"] is not verified:
        raise AssertionError("verification flag must match positive augmented variance")
    if artifact["theory_supports_exp1393"] is not artifact["ngrpo_advantage_calibration_verified"]:
        raise AssertionError("theory_supports_exp1393 must match calibration verification")
    if float(artifact["original_resZero_advantage_variance"]) != 0.0:
        raise AssertionError("ResZero same-reward variance must remain zero")


def _representative_training_rewards(exp1383_artifact: Mapping[str, Any]) -> list[float]:
    training_rows = _rows(exp1383_artifact, "training_reward_rows")
    for row in training_rows:
        rewards = [
            float(reward) for reward in _sequence(row.get("rewards") or row.get("raw_rewards"))
        ]
        if len(rewards) == REAL_ROLLOUT_COUNT:
            return rewards
    return [0.0] * REAL_ROLLOUT_COUNT


def _population_variance(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    mean_value = _mean(numeric)
    return round(sum((value - mean_value) ** 2 for value in numeric) / len(numeric), 12)


def _mean(values: Sequence[float]) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


def _rows(payload: Mapping[str, Any], field_name: str) -> list[Mapping[str, Any]]:
    rows = payload.get(field_name)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list | tuple):
        return list(value)
    return []


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_distribution(values: Sequence[float]) -> dict[str, int]:
    return dict(sorted(Counter(str(float(value)) for value in values).items()))


def _string_distribution(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _write_json(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, payload)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1383-path", default=str(DEFAULT_EXP1383_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--run-date", default=RUN_DATE)
    args = parser.parse_args()
    run_experiment(
        exp1383_path=args.exp1383_path,
        output_path=args.output_path,
        project_root=args.project_root,
        run_date=args.run_date,
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
