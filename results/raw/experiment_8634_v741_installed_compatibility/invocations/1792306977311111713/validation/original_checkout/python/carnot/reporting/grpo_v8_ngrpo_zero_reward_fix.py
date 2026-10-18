"""Exp 1393 GRPO v8 with NGRPO Advantage Calibration.

Spec: REQ-LEARN-1393, SCENARIO-LEARN-1393, SCENARIO-LEARN-1394.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import grpo_v7_jury_rl_formal_verifier_rewards as v7


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260505"
EXPERIMENT = "1393_grpo_v8_ngrpo_zero_reward_fix"
SCHEMA = "grpo_v8_ngrpo_zero_reward_fix_v1"
OUTPUT_FILE = "experiment_1393_grpo_v8_ngrpo_zero_reward_fix.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_FOVER_PATH = v7.DEFAULT_FOVER_PATH
EXP1383_PATH = (
    REPO_ROOT / "results" / "experiment_1383_grpo_v7_jury_rl_formal_verifier_rewards.json"
)
WALL_BUDGET_S = 3600
ROLLOUTS_PER_CASE = 4
DEFAULT_TRAIN_CASES = 4
DEFAULT_HELDOUT_CASES = 4
TENSOR_SPLIT = [0.5, 0.5]
MAX_POSSIBLE_REWARD = 1.0
MANDATED_HEADLINE_MODEL_IDS = v7.MANDATED_HEADLINE_MODEL_IDS
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "grpo_version",
    "reward_mechanism",
    "ngrpo_advantage_calibration_applied",
    "virtual_max_reward_sample_injected",
    "models_used",
    "wall_budget_s",
    "wall_time_used_s",
    "training_steps_completed",
    "jury_acceptance_rate",
    "formal_reward_pass_rate",
    "unknown_rollout_rate",
    "grpo_v8_improvement_pp",
    "wall_budget_exhausted",
    "terminal_blocker",
    "headline_result_allowed",
    "honest_verdict",
}

FoVerJuryCase = v7.FoVerJuryCase
CachedPairFn = Callable[..., Sequence[Mapping[str, Any]] | None]
RolloutGenerator = Callable[
    [FoVerJuryCase, int, Mapping[str, Any], Mapping[str, Any]], Sequence[str]
]
VerifierFn = Callable[[FoVerJuryCase, str], str]
WriteObserver = Callable[[Path, dict[str, Any]], None]


@dataclass(frozen=True)
class NgrpoCalibration:
    """NGRPO advantage-centering result for one rollout group.

    The virtual sample is only a denominator/mean-shift device.  Keeping it
    separate from ``advantages`` prevents later policy code from accidentally
    treating the sample as a fifth generated answer.
    """

    advantages: list[float]
    augmented_rewards: list[float]
    virtual_reward: float | None
    virtual_advantage: float | None
    applied: bool
    virtual_sample_injected: bool


@dataclass(frozen=True)
class NgrpoRewardResult:
    """Reward and NGRPO advantage signal for one JURY-RL rollout group."""

    case_id: str
    rollout_answers: list[str]
    candidate_answer: str
    verifier_result: str
    raw_rewards: list[float]
    advantages: list[float]
    augmented_rewards: list[float]
    ngrpo_advantage_calibration_applied: bool
    virtual_max_reward_sample_injected: bool
    virtual_reward: float | None
    virtual_advantage: float | None

    @property
    def verified(self) -> bool:
        """Return whether Carnot's semantic verifier accepted the candidate."""

        return self.verifier_result == "VERIFIED"


class NgrpoJuryPolicy(v7.JuryPolicy):
    """Tiny policy-reranker updated from NGRPO advantages instead of raw rewards."""

    def update(self, reward_result: NgrpoRewardResult) -> None:  # type: ignore[override]
        """Apply real-rollout advantages; ignore the virtual sample."""

        if not reward_result.rollout_answers:
            return
        scale = self.learning_rate / len(reward_result.rollout_answers)
        for answer, advantage in zip(
            reward_result.rollout_answers,
            reward_result.advantages,
            strict=True,
        ):
            if answer in self.weights:
                self.weights[answer] += float(advantage) * scale


class LlamaCppRolloutGenerator:
    """Generate FoVer answer rollouts through llama.cpp with dual-GPU split."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def __call__(
        self,
        case: FoVerJuryCase,
        n_rollouts: int,
        model_spec: Mapping[str, Any],
        runtime_settings: Mapping[str, Any],
    ) -> list[str]:
        model = self._model_for(model_spec, runtime_settings)
        prompt = v7.build_fover_prompt(case)
        answers: list[str] = []
        for _ in range(n_rollouts):
            response = model(
                prompt,
                max_tokens=int(runtime_settings.get("max_tokens", 16)),
                temperature=float(runtime_settings.get("temperature", 0.7)),
                top_p=float(runtime_settings.get("top_p", 0.95)),
                stop=list(runtime_settings.get("stop", ["\n", "</s>", "<eos>"])),
            )
            answers.append(v7._response_text(response))
        return answers

    def _model_for(self, model_spec: Mapping[str, Any], runtime_settings: Mapping[str, Any]) -> Any:
        key = str(model_spec.get("model_path") or model_spec.get("hf_id") or "")
        if key in self._models:
            return self._models[key]
        model_path = str(model_spec.get("model_path") or "")
        if not model_path:
            raise RuntimeError("selected MODEL_SPECS entry has no model_path")
        llama_cls = _import_llama_class()
        model = llama_cls(
            model_path=model_path,
            n_ctx=int(runtime_settings.get("n_ctx", 2048)),
            n_gpu_layers=int(runtime_settings.get("n_gpu_layers", -1)),
            main_gpu=0,
            tensor_split=list(runtime_settings.get("tensor_split", TENSOR_SPLIT)),
            verbose=bool(runtime_settings.get("verbose", False)),
        )
        self._models[key] = model
        return model


def base_artifact(
    *,
    project_root: str | Path = REPO_ROOT,
    status: str,
    wall_budget_s: int = WALL_BUDGET_S,
) -> dict[str, Any]:
    """Return the Exp 1393 artifact skeleton with every required field present."""

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": RUN_DATE,
            "spec": ["REQ-LEARN-1393", "SCENARIO-LEARN-1393", "SCENARIO-LEARN-1394"],
            "source_experiments": ["exp1383", "exp1388"],
        },
        "run_date": RUN_DATE,
        "started_at": v7.utc_now_iso(),
        "finished_at": None,
        "status": status,
        "grpo_version": "v8",
        "reward_mechanism": "NGRPO Advantage Calibration with virtual max-reward sample",
        "ngrpo_advantage_calibration_applied": False,
        "virtual_max_reward_sample_injected": False,
        "models_used": [],
        "MODEL_SPECS": [],
        "runtime_settings_used": {
            "gpu_indices": [0, 1],
            "tensor_split": list(TENSOR_SPLIT),
            "rollouts_per_case": ROLLOUTS_PER_CASE,
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 16,
            "wall_budget_s": int(wall_budget_s),
        },
        "wall_budget_s": int(wall_budget_s),
        "wall_time_used_s": 0.0,
        "training_steps_completed": 0,
        "jury_acceptance_rate": 0.0,
        "formal_reward_pass_rate": 0.0,
        "unknown_rollout_rate": 0.0,
        "grpo_v8_improvement_pp": 0.0,
        "wall_budget_exhausted": False,
        "terminal_blocker": None,
        "headline_result_allowed": False,
        "retire_if_same_verdict": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "training_mode": "jury_rl_policy_reranker_ngrpo_no_base_weight_write",
        "base_model_weight_update_performed": False,
        "n_training_cases": 0,
        "n_heldout_cases": 0,
        "baseline_heldout_pass_rate": 0.0,
        "post_grpo_heldout_pass_rate": 0.0,
        "training_reward_rows": [],
        "heldout_evaluation_rows": [],
        "source_exp1383_failure": _load_exp1383_failure(),
    }


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    wall_budget_s: int = WALL_BUDGET_S,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1393-1: write the bootstrap artifact before live work."""

    artifact = base_artifact(
        project_root=project_root,
        status="in_progress",
        wall_budget_s=wall_budget_s,
    )
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def resolve_model_specs(*, cached_pair_fn: CachedPairFn | None = None) -> dict[str, Any]:
    """Resolve mandated SOTA GGUF specs and attach the required tensor split."""

    resolver = cached_pair_fn or _cached_sota_pair
    try:
        raw_specs = list(resolver(gpu_indices=(0, 1)) or [])
        error = None
    except Exception as exc:  # pragma: no cover - environment-dependent path.
        raw_specs = []
        error = f"{type(exc).__name__}:{v7._short_error(exc)}"

    specs: list[dict[str, Any]] = []
    for raw_spec in raw_specs:
        spec = dict(raw_spec)
        spec["tensor_split"] = list(TENSOR_SPLIT)
        specs.append(spec)

    models_used = [
        {
            "name": str(spec.get("name") or ""),
            "hf_id": str(spec.get("hf_id") or ""),
            "gpu": spec.get("gpu"),
            "model_path": spec.get("model_path"),
            "tensor_split": list(spec.get("tensor_split", TENSOR_SPLIT)),
            "available": bool(spec.get("model_path")),
            "headline_eligible": str(spec.get("hf_id") or "") in MANDATED_HEADLINE_MODEL_IDS,
            "generation_source": None,
            "selected_for_generation": False,
            "fallback_reason": error,
        }
        for spec in specs
    ]
    return {
        "MODEL_SPECS": specs,
        "models_used": models_used,
        "cached_sota_available": len(specs) >= 2,
        "resolved_model_ids": [str(spec.get("hf_id") or "") for spec in specs],
        "resolution_error": error,
        "tensor_split": list(TENSOR_SPLIT),
    }


def ngrpo_advantage_calibration(
    raw_rewards: Sequence[float],
    *,
    verifier_result: str,
    max_possible_reward: float = MAX_POSSIBLE_REWARD,
) -> NgrpoCalibration:
    """REQ-LEARN-1393-3: inject a virtual max reward before centering.

    This targets the Exp 1383 failure mode exactly: UNKNOWN verifier groups
    emitted all-zero rewards, so ResZero produced no update.  NGRPO shifts the
    group mean by adding one virtual reward of 1.0, making every real zero
    rollout negative while keeping the augmented advantages zero-sum.
    """

    rewards = [float(reward) for reward in raw_rewards]
    inject = bool(
        rewards and verifier_result == "UNKNOWN" and all(abs(reward) <= 1e-12 for reward in rewards)
    )
    virtual_reward = float(max_possible_reward) if inject else None
    augmented = [*rewards, virtual_reward] if virtual_reward is not None else list(rewards)
    if not augmented:
        return NgrpoCalibration([], [], None, None, False, False)
    mean_reward = sum(augmented) / len(augmented)
    advantages = [round(reward - mean_reward, 12) for reward in rewards]
    virtual_advantage = (
        round(float(virtual_reward) - mean_reward, 12) if virtual_reward is not None else None
    )
    return NgrpoCalibration(
        advantages=advantages,
        augmented_rewards=[round(reward, 12) for reward in augmented],
        virtual_reward=virtual_reward,
        virtual_advantage=virtual_advantage,
        applied=inject,
        virtual_sample_injected=inject,
    )


def jury_ngrpo_reward_for_case(
    case: FoVerJuryCase,
    rollouts: Sequence[str],
    *,
    verifier_fn: VerifierFn | None = None,
) -> NgrpoRewardResult:
    """REQ-LEARN-1393-2/3: compute verifier rewards and NGRPO advantages."""

    answers = [v7._normalise_answer(rollout) for rollout in rollouts]
    candidate = v7.majority_vote(answers)
    verifier = verifier_fn or v7.semantic_verifier
    verifier_result = verifier(case, candidate)
    if verifier_result == "VERIFIED":
        raw_rewards = [1.0 if answer == candidate else -1.0 for answer in answers]
    elif verifier_result == "UNKNOWN":
        raw_rewards = [0.0 for _ in answers]
    else:
        raw_rewards = [-1.0 for _ in answers]
    calibration = ngrpo_advantage_calibration(
        raw_rewards,
        verifier_result=verifier_result,
        max_possible_reward=MAX_POSSIBLE_REWARD,
    )
    return NgrpoRewardResult(
        case_id=case.case_id,
        rollout_answers=answers,
        candidate_answer=candidate,
        verifier_result=verifier_result,
        raw_rewards=raw_rewards,
        advantages=calibration.advantages,
        augmented_rewards=calibration.augmented_rewards,
        ngrpo_advantage_calibration_applied=calibration.applied,
        virtual_max_reward_sample_injected=calibration.virtual_sample_injected,
        virtual_reward=calibration.virtual_reward,
        virtual_advantage=calibration.virtual_advantage,
    )


def run(
    *,
    results_dir: Path | str | None = None,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: Path | str = REPO_ROOT,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    cases: Sequence[FoVerJuryCase] | None = None,
    cached_pair_fn: CachedPairFn | None = None,
    rollout_generator: RolloutGenerator | None = None,
    train_case_count: int = DEFAULT_TRAIN_CASES,
    heldout_case_count: int = DEFAULT_HELDOUT_CASES,
    wall_budget_s: int = WALL_BUDGET_S,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run the bounded GRPO v8 NGRPO loop and write its artifact."""

    if results_dir is not None:
        out_path = Path(results_dir) / OUTPUT_FILE
    output = Path(out_path)
    write_in_progress_artifact(
        output,
        project_root=project_root,
        wall_budget_s=wall_budget_s,
        write_observer=write_observer,
    )
    started = time.perf_counter()
    artifact = base_artifact(
        project_root=project_root, status="complete", wall_budget_s=wall_budget_s
    )

    model_resolution = resolve_model_specs(cached_pair_fn=cached_pair_fn)
    model_specs = [dict(spec) for spec in model_resolution["MODEL_SPECS"]]
    models_used = [dict(model) for model in model_resolution["models_used"]]
    artifact["MODEL_SPECS"] = model_specs
    artifact["models_used"] = models_used

    selected_spec = v7._select_generation_spec(model_specs)
    if selected_spec is None:
        artifact = _blocked_artifact(
            artifact,
            models_used=models_used,
            wall_time_used_s=time.perf_counter() - started,
            terminal_blocker="cached_sota_pair_unavailable",
        )
        _write_json(output, artifact, write_observer=write_observer)
        validate_artifact(artifact)
        return artifact

    models_used = v7._mark_selected_model(models_used, selected_spec, fallback_reason=None)
    artifact["models_used"] = models_used
    selected_cases = list(cases) if cases is not None else v7.load_fover_cases(fover_path)
    train_cases, heldout_cases = v7._split_cases(
        selected_cases,
        train_case_count=train_case_count,
        heldout_case_count=heldout_case_count,
    )
    artifact["n_training_cases"] = len(train_cases)
    artifact["n_heldout_cases"] = len(heldout_cases)
    if not train_cases or not heldout_cases:
        artifact = _blocked_artifact(
            artifact,
            models_used=models_used,
            wall_time_used_s=time.perf_counter() - started,
            terminal_blocker="fover_case_count_insufficient",
        )
        _write_json(output, artifact, write_observer=write_observer)
        validate_artifact(artifact)
        return artifact

    generator = rollout_generator or LlamaCppRolloutGenerator()
    runtime_settings = dict(artifact["runtime_settings_used"])
    policy = NgrpoJuryPolicy()
    training_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    verified_count = 0
    ngrpo_count = 0
    virtual_count = 0
    unknown_rollouts = 0
    total_rollouts = 0
    baseline_passes = 0
    final_passes = 0

    try:
        for case in train_cases:
            if _wall_budget_hit(started, wall_budget_s):
                return _finish_wall_budget(
                    output,
                    artifact,
                    models_used,
                    started,
                    training_rows,
                    heldout_rows,
                    unknown_rollouts,
                    total_rollouts,
                    write_observer,
                )
            rollouts = generator(case, ROLLOUTS_PER_CASE, selected_spec, runtime_settings)
            reward = jury_ngrpo_reward_for_case(case, rollouts)
            policy.update(reward)
            verified_count += int(reward.verified)
            ngrpo_count += int(reward.ngrpo_advantage_calibration_applied)
            virtual_count += int(reward.virtual_max_reward_sample_injected)
            unknown_rollouts += sum(answer == "UNKNOWN" for answer in reward.rollout_answers)
            total_rollouts += len(reward.rollout_answers)
            training_rows.append(_reward_row(case, reward))

        for case in heldout_cases:
            if _wall_budget_hit(started, wall_budget_s):
                return _finish_wall_budget(
                    output,
                    artifact,
                    models_used,
                    started,
                    training_rows,
                    heldout_rows,
                    unknown_rollouts,
                    total_rollouts,
                    write_observer,
                )
            rollouts = generator(case, ROLLOUTS_PER_CASE, selected_spec, runtime_settings)
            answers = [v7._normalise_answer(rollout) for rollout in rollouts]
            unknown_rollouts += sum(answer == "UNKNOWN" for answer in answers)
            total_rollouts += len(answers)
            baseline_answer = v7.majority_vote(answers)
            final_answer = policy.select(answers)
            baseline_result = v7.semantic_verifier(case, baseline_answer)
            final_result = v7.semantic_verifier(case, final_answer)
            baseline_passes += int(baseline_result == "VERIFIED")
            final_passes += int(final_result == "VERIFIED")
            heldout_rows.append(
                {
                    "case_id": case.case_id,
                    "expected_answer": case.expected_answer,
                    "rollout_answers": answers,
                    "baseline_answer": baseline_answer,
                    "post_grpo_answer": final_answer,
                    "baseline_verifier_result": baseline_result,
                    "post_grpo_verifier_result": final_result,
                    "improved": final_result == "VERIFIED" and baseline_result != "VERIFIED",
                }
            )
    except Exception as exc:
        blocker = f"llama_cpp_generation_failed:{type(exc).__name__}:{v7._short_error(exc)}"
        artifact["training_reward_rows"] = training_rows
        artifact["heldout_evaluation_rows"] = heldout_rows
        artifact = _blocked_artifact(
            artifact,
            models_used=v7._mark_selected_model(
                models_used, selected_spec, fallback_reason=blocker
            ),
            wall_time_used_s=time.perf_counter() - started,
            terminal_blocker=blocker,
        )
        _write_json(output, artifact, write_observer=write_observer)
        validate_artifact(artifact)
        return artifact

    training_steps = len(training_rows)
    formal_reward_pass_rate = v7._rate(verified_count, training_steps)
    baseline_rate = v7._rate(baseline_passes, len(heldout_rows))
    final_rate = v7._rate(final_passes, len(heldout_rows))
    improvement_pp = round(100.0 * (final_rate - baseline_rate), 6)
    unknown_rate = v7._rate(unknown_rollouts, total_rollouts)
    headline_allowed = bool(
        improvement_pp > 0.0
        and not _wall_budget_hit(started, wall_budget_s)
        and v7._selected_model_is_mandated(selected_spec)
    )
    terminal_blocker, retire = _retirement_gate(improvement_pp, unknown_rate)
    artifact.update(
        {
            "finished_at": v7.utc_now_iso(),
            "status": "complete",
            "models_used": v7._mark_selected_model(
                models_used,
                selected_spec,
                fallback_reason=None,
                headline_result_allowed=headline_allowed,
            ),
            "wall_time_used_s": round(time.perf_counter() - started, 6),
            "training_steps_completed": training_steps,
            "jury_acceptance_rate": formal_reward_pass_rate,
            "formal_reward_pass_rate": formal_reward_pass_rate,
            "unknown_rollout_rate": unknown_rate,
            "ngrpo_advantage_calibration_applied": ngrpo_count > 0,
            "virtual_max_reward_sample_injected": virtual_count > 0,
            "virtual_max_reward_sample_injected_count": virtual_count,
            "baseline_heldout_pass_rate": baseline_rate,
            "post_grpo_heldout_pass_rate": final_rate,
            "grpo_v8_improvement_pp": improvement_pp,
            "wall_budget_exhausted": False,
            "terminal_blocker": terminal_blocker,
            "headline_result_allowed": headline_allowed,
            "retire_if_same_verdict": retire,
            "honest_verdict": _honest_verdict(improvement_pp, headline_allowed, retire),
            "training_reward_rows": training_rows,
            "heldout_evaluation_rows": heldout_rows,
            "policy_weights": {key: round(value, 6) for key, value in policy.weights.items()},
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields required by the Exp 1393 task contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS.difference(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError("terminal artifact status must be complete or blocked")
    for field_name in (
        "wall_budget_s",
        "wall_time_used_s",
        "training_steps_completed",
        "jury_acceptance_rate",
        "formal_reward_pass_rate",
        "unknown_rollout_rate",
        "grpo_v8_improvement_pp",
    ):
        if not isinstance(artifact[field_name], (int, float)):
            raise AssertionError(f"{field_name} must be numeric")
    for field_name in (
        "ngrpo_advantage_calibration_applied",
        "virtual_max_reward_sample_injected",
        "wall_budget_exhausted",
        "headline_result_allowed",
    ):
        if not isinstance(artifact[field_name], bool):
            raise AssertionError(f"{field_name} must be boolean")
    if artifact["headline_result_allowed"]:
        if float(artifact["grpo_v8_improvement_pp"]) <= 0.0:
            raise AssertionError("headline claims require positive improvement")
        model_ids = {str(model.get("hf_id") or "") for model in artifact.get("models_used", [])}
        if not model_ids.intersection(MANDATED_HEADLINE_MODEL_IDS):
            raise AssertionError("headline claims require a mandated SOTA GGUF model")
    if (
        float(artifact["grpo_v8_improvement_pp"]) == 0.0
        and float(artifact["unknown_rollout_rate"]) >= 0.95
    ):
        if artifact.get("retire_if_same_verdict") is not True:
            raise AssertionError("all-UNKNOWN zero-delta run requires retirement")
        if artifact.get("terminal_blocker") != (
            "ngrpo_still_zero_reward_fover_incompatible_with_jury_rl"
        ):
            raise AssertionError("all-UNKNOWN zero-delta run requires terminal blocker")


def _reward_row(case: FoVerJuryCase, reward: NgrpoRewardResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "expected_answer": case.expected_answer,
        "rollout_answers": reward.rollout_answers,
        "candidate_answer": reward.candidate_answer,
        "verifier_result": reward.verifier_result,
        "raw_rewards": reward.raw_rewards,
        "advantages": reward.advantages,
        "augmented_rewards": reward.augmented_rewards,
        "reward_mean": round(sum(reward.raw_rewards) / len(reward.raw_rewards), 6)
        if reward.raw_rewards
        else 0.0,
        "augmented_reward_mean": round(
            sum(reward.augmented_rewards) / len(reward.augmented_rewards), 6
        )
        if reward.augmented_rewards
        else 0.0,
        "ngrpo_advantage_calibration_applied": reward.ngrpo_advantage_calibration_applied,
        "virtual_max_reward_sample_injected": reward.virtual_max_reward_sample_injected,
        "virtual_reward": reward.virtual_reward,
        "virtual_advantage": reward.virtual_advantage,
    }


def _blocked_artifact(
    artifact: Mapping[str, Any],
    *,
    models_used: Sequence[Mapping[str, Any]],
    wall_time_used_s: float,
    terminal_blocker: str,
) -> dict[str, Any]:
    blocked = dict(artifact)
    blocked.update(
        {
            "finished_at": v7.utc_now_iso(),
            "status": "complete",
            "models_used": [dict(model) for model in models_used],
            "wall_time_used_s": round(float(wall_time_used_s), 6),
            "wall_budget_exhausted": False,
            "terminal_blocker": terminal_blocker,
            "headline_result_allowed": False,
            "retire_if_same_verdict": False,
            "honest_verdict": f"blocked_{terminal_blocker}",
        }
    )
    return blocked


def _finish_wall_budget(
    output: Path,
    artifact: dict[str, Any],
    models_used: Sequence[Mapping[str, Any]],
    started: float,
    training_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    unknown_rollouts: int,
    total_rollouts: int,
    write_observer: WriteObserver | None,
) -> dict[str, Any]:
    artifact.update(
        {
            "finished_at": v7.utc_now_iso(),
            "status": "complete",
            "models_used": [dict(model) for model in models_used],
            "wall_time_used_s": round(time.perf_counter() - started, 6),
            "training_steps_completed": len(training_rows),
            "unknown_rollout_rate": v7._rate(unknown_rollouts, total_rollouts),
            "wall_budget_exhausted": True,
            "terminal_blocker": "wall_budget_exhausted",
            "headline_result_allowed": False,
            "retire_if_same_verdict": True,
            "honest_verdict": "wall_budget_exhausted",
            "training_reward_rows": training_rows,
            "heldout_evaluation_rows": heldout_rows,
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    validate_artifact(artifact)
    return artifact


def _retirement_gate(improvement_pp: float, unknown_rollout_rate: float) -> tuple[str | None, bool]:
    if float(improvement_pp) == 0.0 and float(unknown_rollout_rate) >= 0.95:
        return "ngrpo_still_zero_reward_fover_incompatible_with_jury_rl", True
    return None, False


def _honest_verdict(improvement_pp: float, headline_result_allowed: bool, retired: bool) -> str:
    if retired:
        return "grpo_v8_ngrpo_no_improvement_all_unknown_retired"
    if improvement_pp > 0.0:
        label = str(round(float(improvement_pp), 1)).replace(".", "_")
        return (
            f"grpo_v8_ngrpo_positive_improvement_{label}pp"
            if headline_result_allowed
            else f"grpo_v8_ngrpo_positive_non_headline_{label}pp"
        )
    if improvement_pp < 0.0:
        return "grpo_v8_ngrpo_regression"
    return "grpo_v8_ngrpo_no_improvement"


def _wall_budget_hit(started: float, wall_budget_s: int) -> bool:
    return (time.perf_counter() - started) >= float(wall_budget_s)


def _load_exp1383_failure() -> dict[str, Any]:
    try:
        payload = json.loads(EXP1383_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "experiment": payload.get("experiment"),
        "formal_reward_pass_rate": payload.get("formal_reward_pass_rate"),
        "unknown_rollout_rate": payload.get("unknown_rollout_rate", 1.0),
        "grpo_v7_improvement_pp": payload.get("grpo_v7_improvement_pp"),
        "honest_verdict": payload.get("honest_verdict"),
    }


def _cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    try:
        from carnot.inference.sota_models import cached_sota_pair

        return cached_sota_pair(**kwargs)
    except Exception:
        spec_path = REPO_ROOT / "python" / "carnot" / "inference" / "sota_models.py"
        spec = importlib.util.spec_from_file_location("_carnot_sota_models_direct", spec_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.cached_sota_pair(**kwargs)


def _ensure_cuda_runtime_on_ld_path() -> None:  # pragma: no cover - live GGUF only.
    sentinel = "CARNOT_LDPATH_PATCHED"
    if os.environ.get(sentinel) == "1":
        return
    venv_site = (
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvidia_root = venv_site / "nvidia"
    if not nvidia_root.is_dir():
        return
    nvidia_dirs = [
        str(sub / "lib") for sub in sorted(nvidia_root.iterdir()) if (sub / "lib").is_dir()
    ]
    if not nvidia_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join([*nvidia_dirs, existing]) if existing else ":".join(nvidia_dirs)
    )
    os.environ[sentinel] = "1"
    if sys.argv and sys.argv[0] != "-c":
        os.execv(sys.executable, [sys.executable, *sys.argv])


def _import_llama_class() -> type[Any]:  # pragma: no cover - live GGUF only.
    _ensure_cuda_runtime_on_ld_path()
    from llama_cpp import Llama

    return Llama


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if write_observer is not None:
        write_observer(path, dict(payload))


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    run(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
