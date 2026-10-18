"""Exp 1383 GRPO v7 with JURY-RL formal verifier rewards.

Spec: REQ-LEARN-1383, SCENARIO-LEARN-1383, SCENARIO-LEARN-1384.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260505"
EXPERIMENT = "1383_grpo_v7_jury_rl_formal_verifier_rewards"
SCHEMA = "grpo_v7_jury_rl_formal_verifier_rewards_v1"
OUTPUT_FILE = "experiment_1383_grpo_v7_jury_rl_formal_verifier_rewards.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
WALL_BUDGET_S = 2400
ROLLOUTS_PER_CASE = 4
DEFAULT_TRAIN_CASES = 4
DEFAULT_HELDOUT_CASES = 4
TENSOR_SPLIT = [0.5, 0.5]
CANONICAL_ANSWERS = ("SAT", "REPAIR_HINT", "UNKNOWN")
MANDATED_HEADLINE_MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "grpo_version",
    "reward_mechanism",
    "models_used",
    "wall_budget_s",
    "wall_time_used_s",
    "training_steps_completed",
    "jury_acceptance_rate",
    "formal_reward_pass_rate",
    "resZero_applied_count",
    "grpo_v7_improvement_pp",
    "wall_budget_exhausted",
    "terminal_blocker",
    "headline_result_allowed",
    "honest_verdict",
}


CachedPairFn = Callable[..., Sequence[Mapping[str, Any]] | None]
RolloutGenerator = Callable[
    ["FoVerJuryCase", int, Mapping[str, Any], Mapping[str, Any]], Sequence[str]
]
VerifierFn = Callable[["FoVerJuryCase", str], str]
WriteObserver = Callable[[Path, dict[str, Any]], None]


@dataclass(frozen=True)
class FoVerJuryCase:
    """One FoVer reasoning row normalized for JURY-RL reward assignment.

    ``label=0`` means the row is locally accepted as correct reasoning, while
    ``label=1`` means Carnot should route the row to repair.  The live reward
    path still asks the local semantic verifier to decide the candidate answer;
    the label is the verifier fixture used by the existing FoVer corpus.
    """

    case_id: str
    question: str
    response: str
    label: int
    source: str

    @property
    def expected_answer(self) -> str:
        """Return the verifier answer implied by the local FoVer fixture."""

        return "REPAIR_HINT" if int(self.label) == 1 else "SAT"


@dataclass(frozen=True)
class JuryRewardResult:
    """Reward signal for one JURY-RL group of rollouts."""

    case_id: str
    rollout_answers: list[str]
    candidate_answer: str
    verifier_result: str
    rewards: list[float]
    reszero_applied: bool

    @property
    def verified(self) -> bool:
        """Return whether the formal verifier accepted the voted candidate."""

        return self.verifier_result == "VERIFIED"


@dataclass
class JuryPolicy:
    """Tiny GRPO-style answer policy updated from group-relative rewards.

    The GGUF base model is not written back to disk by llama.cpp.  This policy
    captures the experiment's trainable state as a verifier-reward reranker over
    the model's four sampled answers, which is the smallest closed-loop update
    this runner can execute safely inside the wall budget.
    """

    learning_rate: float = 0.25
    weights: dict[str, float] = field(
        default_factory=lambda: {answer: 0.0 for answer in CANONICAL_ANSWERS}
    )

    def update(self, reward_result: JuryRewardResult) -> None:
        """Apply the mean reward for each answer token to the reranker."""

        if not reward_result.rollout_answers:
            return
        scale = self.learning_rate / len(reward_result.rollout_answers)
        for answer, reward in zip(
            reward_result.rollout_answers, reward_result.rewards, strict=True
        ):
            if answer in self.weights:
                self.weights[answer] += float(reward) * scale

    def select(self, rollout_answers: Sequence[str]) -> str:
        """Select the answer with majority support plus learned reward weight."""

        answers = [_normalise_answer(answer) for answer in rollout_answers]
        counts = Counter(answers)
        if not counts:
            return "UNKNOWN"
        return max(
            CANONICAL_ANSWERS,
            key=lambda answer: (
                counts.get(answer, 0) + self.weights.get(answer, 0.0),
                -CANONICAL_ANSWERS.index(answer),
            ),
        )


class LlamaCppRolloutGenerator:
    """Generate FoVer answer rollouts through the mandated local GGUF path."""

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
        prompt = build_fover_prompt(case)
        answers: list[str] = []
        for _ in range(n_rollouts):
            response = model(
                prompt,
                max_tokens=int(runtime_settings.get("max_tokens", 16)),
                temperature=float(runtime_settings.get("temperature", 0.7)),
                top_p=float(runtime_settings.get("top_p", 0.95)),
                stop=list(runtime_settings.get("stop", ["\n", "</s>", "<eos>"])),
            )
            answers.append(_response_text(response))
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


def utc_now_iso() -> str:
    """Return an artifact-friendly UTC timestamp."""

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def base_artifact(
    *,
    project_root: str | Path = REPO_ROOT,
    status: str,
    wall_budget_s: int = WALL_BUDGET_S,
) -> dict[str, Any]:
    """Return the Exp 1383 artifact skeleton with every required field present."""

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": RUN_DATE,
            "spec": ["REQ-LEARN-1383", "SCENARIO-LEARN-1383", "SCENARIO-LEARN-1384"],
            "source_experiments": ["exp1366", "exp1369"],
        },
        "run_date": RUN_DATE,
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": status,
        "grpo_version": "v7",
        "reward_mechanism": "JURY-RL majority-vote proposal plus Carnot formal verifier rewards",
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
        },
        "wall_budget_s": int(wall_budget_s),
        "wall_time_used_s": 0.0,
        "training_steps_completed": 0,
        "jury_acceptance_rate": 0.0,
        "formal_reward_pass_rate": 0.0,
        "resZero_applied_count": 0,
        "grpo_v7_improvement_pp": 0.0,
        "wall_budget_exhausted": False,
        "terminal_blocker": None,
        "headline_result_allowed": False,
        "retire_if_same_verdict": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "training_mode": "jury_rl_policy_reranker_no_base_weight_write",
        "base_model_weight_update_performed": False,
        "n_training_cases": 0,
        "n_heldout_cases": 0,
        "baseline_heldout_pass_rate": 0.0,
        "post_grpo_heldout_pass_rate": 0.0,
        "training_reward_rows": [],
        "heldout_evaluation_rows": [],
    }


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    wall_budget_s: int = WALL_BUDGET_S,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1383-1: write the bootstrap artifact before loading inputs."""

    artifact = base_artifact(
        project_root=project_root, status="in_progress", wall_budget_s=wall_budget_s
    )
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def resolve_model_specs(*, cached_pair_fn: CachedPairFn | None = None) -> dict[str, Any]:
    """Resolve mandated headline GGUF specs and attach dual-GPU tensor split."""

    resolver = cached_pair_fn or _cached_sota_pair
    try:
        raw_specs = list(resolver(gpu_indices=(0, 1)) or [])
        error = None
    except Exception as exc:  # pragma: no cover - depends on local environment.
        raw_specs = []
        error = f"{type(exc).__name__}:{_short_error(exc)}"
    specs = []
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


def load_fover_cases(
    path: Path | str = DEFAULT_FOVER_PATH,
    *,
    target_cases: int = DEFAULT_TRAIN_CASES + DEFAULT_HELDOUT_CASES,
) -> list[FoVerJuryCase]:
    """Load a deterministic balanced FoVer subset from local JSONL rows."""

    rows = _read_rows(Path(path))
    cases: list[FoVerJuryCase] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        label = _label_from_row(row)
        response = _row_text(row)
        if label is None or not response:
            continue
        raw_id = str(
            row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or row.get("question_index")
            or f"fover_{index}"
        )
        ordinal = seen.get(raw_id, 0)
        seen[raw_id] = ordinal + 1
        case_id = raw_id if ordinal == 0 else f"{raw_id}_{ordinal}"
        cases.append(
            FoVerJuryCase(
                case_id=case_id,
                question=str(row.get("question") or row.get("prompt") or ""),
                response=response,
                label=label,
                source=str(row.get("source") or "fover_corpus"),
            )
        )
    return _balanced_subset(cases, target_cases)


def build_fover_prompt(case: FoVerJuryCase) -> str:
    """Build the label-only verifier prompt used for each GGUF rollout."""

    question = _truncate(case.question, 500)
    response = _truncate(case.response, 1400)
    return (
        "You are Carnot's local FoVer verifier. Classify the reasoning step.\n"
        "Return exactly one label and no explanation:\n"
        "SAT = the reasoning step is correct enough to accept.\n"
        "REPAIR_HINT = the reasoning step is incorrect and needs repair.\n"
        "UNKNOWN = the local evidence is insufficient.\n"
        f"Question: {question}\n"
        f"Reasoning step: {response}\n"
        "Label:"
    )


def jury_reward_for_case(
    case: FoVerJuryCase,
    rollouts: Sequence[str],
    *,
    verifier_fn: VerifierFn | None = None,
) -> JuryRewardResult:
    """REQ-LEARN-1383-2/3: compute JURY-RL rewards for one rollout group."""

    answers = [_normalise_answer(rollout) for rollout in rollouts]
    candidate = majority_vote(answers)
    verifier = verifier_fn or semantic_verifier
    verifier_result = verifier(case, candidate)
    if verifier_result == "VERIFIED":
        rewards = [1.0 if answer == candidate else -1.0 for answer in answers]
        reszero = False
    elif verifier_result == "UNKNOWN":
        rewards = _reszero_rewards(answers)
        reszero = True
    else:
        rewards = [-1.0 for _ in answers]
        reszero = False
    return JuryRewardResult(
        case_id=case.case_id,
        rollout_answers=answers,
        candidate_answer=candidate,
        verifier_result=verifier_result,
        rewards=rewards,
        reszero_applied=reszero,
    )


def majority_vote(answers: Sequence[str]) -> str:
    """Return the most common normalized answer, preserving rollout order on ties."""

    normalised = [_normalise_answer(answer) for answer in answers]
    if not normalised:
        return "UNKNOWN"
    return Counter(normalised).most_common(1)[0][0]


def semantic_verifier(case: FoVerJuryCase, candidate_answer: str) -> str:
    """Carnot semantic verifier wrapper for the FoVer answer states."""

    answer = _normalise_answer(candidate_answer)
    if answer == "UNKNOWN":
        return "UNKNOWN"
    return "VERIFIED" if answer == case.expected_answer else "REJECTED"


def wall_budget_terminal_artifact(
    *,
    base_artifact: Mapping[str, Any],
    models_used: Sequence[Mapping[str, Any]],
    wall_time_used_s: float,
    training_steps_completed: int,
    grpo_v7_improvement_pp: float,
) -> dict[str, Any]:
    """REQ-LEARN-1383-6: produce the terminal wall-budget artifact."""

    artifact = dict(base_artifact)
    artifact.update(
        {
            "status": "complete",
            "finished_at": utc_now_iso(),
            "models_used": [dict(model) for model in models_used],
            "wall_time_used_s": round(float(wall_time_used_s), 6),
            "training_steps_completed": int(training_steps_completed),
            "grpo_v7_improvement_pp": round(float(grpo_v7_improvement_pp), 6),
            "wall_budget_exhausted": True,
            "terminal_blocker": "wall_budget_exhausted",
            "headline_result_allowed": False,
            "retire_if_same_verdict": True,
            "honest_verdict": "wall_budget_exhausted",
        }
    )
    return artifact


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
    """Run the bounded Exp 1383 JURY-RL reward loop and write its artifact."""

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
    artifact["runtime_settings_used"]["tensor_split"] = list(TENSOR_SPLIT)
    artifact["runtime_settings_used"]["wall_budget_s"] = int(wall_budget_s)

    selected_spec = _select_generation_spec(model_specs)
    if selected_spec is None:
        blocker = "cached_sota_pair_unavailable"
        artifact = _blocked_artifact(
            artifact,
            models_used=models_used,
            wall_time_used_s=time.perf_counter() - started,
            terminal_blocker=blocker,
        )
        _write_json(output, artifact, write_observer=write_observer)
        validate_artifact(artifact)
        return artifact

    models_used = _mark_selected_model(models_used, selected_spec, fallback_reason=None)
    artifact["models_used"] = models_used
    selected_cases = list(cases) if cases is not None else load_fover_cases(fover_path)
    train_cases, heldout_cases = _split_cases(
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
    policy = JuryPolicy()
    training_rows: list[dict[str, Any]] = []
    verified_count = 0
    reszero_count = 0

    try:
        for case in train_cases:
            if _wall_budget_hit(started, wall_budget_s):
                artifact["training_reward_rows"] = training_rows
                terminal = wall_budget_terminal_artifact(
                    base_artifact=artifact,
                    models_used=models_used,
                    wall_time_used_s=time.perf_counter() - started,
                    training_steps_completed=len(training_rows),
                    grpo_v7_improvement_pp=0.0,
                )
                _write_json(output, terminal, write_observer=write_observer)
                validate_artifact(terminal)
                return terminal
            rollouts = generator(case, ROLLOUTS_PER_CASE, selected_spec, runtime_settings)
            reward = jury_reward_for_case(case, rollouts)
            policy.update(reward)
            verified_count += int(reward.verified)
            reszero_count += int(reward.reszero_applied)
            training_rows.append(_reward_row(case, reward))

        heldout_rows = []
        baseline_passes = 0
        final_passes = 0
        for case in heldout_cases:
            if _wall_budget_hit(started, wall_budget_s):
                artifact["training_reward_rows"] = training_rows
                artifact["heldout_evaluation_rows"] = heldout_rows
                partial_improvement = _improvement_pp(
                    baseline_passes,
                    final_passes,
                    len(heldout_rows),
                )
                terminal = wall_budget_terminal_artifact(
                    base_artifact=artifact,
                    models_used=models_used,
                    wall_time_used_s=time.perf_counter() - started,
                    training_steps_completed=len(training_rows),
                    grpo_v7_improvement_pp=partial_improvement,
                )
                _write_json(output, terminal, write_observer=write_observer)
                validate_artifact(terminal)
                return terminal
            rollouts = generator(case, ROLLOUTS_PER_CASE, selected_spec, runtime_settings)
            answers = [_normalise_answer(rollout) for rollout in rollouts]
            baseline_answer = majority_vote(answers)
            final_answer = policy.select(answers)
            baseline_result = semantic_verifier(case, baseline_answer)
            final_result = semantic_verifier(case, final_answer)
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
        blocker = f"llama_cpp_generation_failed:{type(exc).__name__}:{_short_error(exc)}"
        artifact["training_reward_rows"] = training_rows
        artifact = _blocked_artifact(
            artifact,
            models_used=_mark_selected_model(models_used, selected_spec, fallback_reason=blocker),
            wall_time_used_s=time.perf_counter() - started,
            terminal_blocker=blocker,
        )
        _write_json(output, artifact, write_observer=write_observer)
        validate_artifact(artifact)
        return artifact

    training_steps = len(training_rows)
    jury_acceptance_rate = _rate(verified_count, training_steps)
    formal_reward_pass_rate = jury_acceptance_rate
    baseline_rate = _rate(baseline_passes, len(heldout_rows))
    final_rate = _rate(final_passes, len(heldout_rows))
    improvement_pp = round(100.0 * (final_rate - baseline_rate), 6)
    headline_allowed = bool(
        improvement_pp > 0.0
        and not _wall_budget_hit(started, wall_budget_s)
        and _selected_model_is_mandated(selected_spec)
    )
    artifact.update(
        {
            "finished_at": utc_now_iso(),
            "status": "complete",
            "models_used": _mark_selected_model(
                models_used,
                selected_spec,
                fallback_reason=None,
                headline_result_allowed=headline_allowed,
            ),
            "wall_time_used_s": round(time.perf_counter() - started, 6),
            "training_steps_completed": training_steps,
            "jury_acceptance_rate": jury_acceptance_rate,
            "formal_reward_pass_rate": formal_reward_pass_rate,
            "resZero_applied_count": reszero_count,
            "baseline_heldout_pass_rate": baseline_rate,
            "post_grpo_heldout_pass_rate": final_rate,
            "grpo_v7_improvement_pp": improvement_pp,
            "wall_budget_exhausted": False,
            "terminal_blocker": None,
            "headline_result_allowed": headline_allowed,
            "retire_if_same_verdict": False,
            "honest_verdict": _honest_verdict(improvement_pp, headline_allowed),
            "training_reward_rows": training_rows,
            "heldout_evaluation_rows": heldout_rows,
            "policy_weights": {key: round(value, 6) for key, value in policy.weights.items()},
        }
    )
    _write_json(output, artifact, write_observer=write_observer)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields required by the Exp 1383 task contract."""

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
        "resZero_applied_count",
        "grpo_v7_improvement_pp",
    ):
        if not isinstance(artifact[field_name], (int, float)):
            raise AssertionError(f"{field_name} must be numeric")
    if not isinstance(artifact["wall_budget_exhausted"], bool):
        raise AssertionError("wall_budget_exhausted must be boolean")
    if not isinstance(artifact["headline_result_allowed"], bool):
        raise AssertionError("headline_result_allowed must be boolean")
    if artifact["wall_budget_exhausted"]:
        if artifact.get("terminal_blocker") != "wall_budget_exhausted":
            raise AssertionError("wall exhaustion requires terminal_blocker")
        if artifact.get("retire_if_same_verdict") is not True:
            raise AssertionError("wall exhaustion requires retire_if_same_verdict=true")
    if artifact["headline_result_allowed"]:
        if float(artifact["grpo_v7_improvement_pp"]) <= 0.0:
            raise AssertionError("headline claims require positive improvement")
        model_ids = {str(model.get("hf_id") or "") for model in artifact.get("models_used", [])}
        if not model_ids.intersection(MANDATED_HEADLINE_MODEL_IDS):
            raise AssertionError("headline claims require a mandated SOTA GGUF model")


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
            "finished_at": utc_now_iso(),
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


def _reward_row(case: FoVerJuryCase, reward: JuryRewardResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "expected_answer": case.expected_answer,
        "rollout_answers": reward.rollout_answers,
        "candidate_answer": reward.candidate_answer,
        "verifier_result": reward.verifier_result,
        "rewards": reward.rewards,
        "reward_mean": round(sum(reward.rewards) / len(reward.rewards), 6)
        if reward.rewards
        else 0.0,
        "reszero_applied": reward.reszero_applied,
    }


def _normalise_answer(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if "REPAIR_HINT" in text or "REPAIR HINT" in text or re.search(r"\bREPAIR\b", text):
        return "REPAIR_HINT"
    if re.search(r"\bUNKNOWN\b", text):
        return "UNKNOWN"
    if re.search(r"\bSAT\b", text):
        return "SAT"
    return "UNKNOWN"


def _reszero_rewards(answers: Sequence[str]) -> list[float]:
    if not answers:
        return []
    counts = Counter(answers)
    support = [counts[answer] / len(answers) for answer in answers]
    mean_support = sum(support) / len(support)
    rewards = [round(value - mean_support, 12) for value in support]
    if rewards:
        rewards[-1] = round(rewards[-1] - sum(rewards), 12)
    return rewards


def _split_cases(
    cases: Sequence[FoVerJuryCase],
    *,
    train_case_count: int,
    heldout_case_count: int,
) -> tuple[list[FoVerJuryCase], list[FoVerJuryCase]]:
    selected = list(cases)
    train = selected[: max(0, train_case_count)]
    heldout_start = len(train)
    heldout_end = heldout_start + max(0, heldout_case_count)
    return train, selected[heldout_start:heldout_end]


def _select_generation_spec(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for spec in model_specs:
        if str(spec.get("hf_id") or "") in MANDATED_HEADLINE_MODEL_IDS and spec.get("model_path"):
            return dict(spec)
    return None


def _mark_selected_model(
    models_used: Sequence[Mapping[str, Any]],
    selected_spec: Mapping[str, Any],
    *,
    fallback_reason: str | None,
    headline_result_allowed: bool = False,
) -> list[dict[str, Any]]:
    selected_key = str(selected_spec.get("model_path") or selected_spec.get("hf_id") or "")
    marked: list[dict[str, Any]] = []
    for model in models_used:
        row = dict(model)
        key = str(row.get("model_path") or row.get("hf_id") or "")
        selected = key == selected_key
        row["selected_for_generation"] = selected
        row["generation_source"] = "live_sota_llamacpp" if selected else None
        row["headline_result_allowed"] = bool(headline_result_allowed and selected)
        row["fallback_reason"] = fallback_reason
        marked.append(row)
    return marked


def _selected_model_is_mandated(selected_spec: Mapping[str, Any]) -> bool:
    return str(selected_spec.get("hf_id") or "") in MANDATED_HEADLINE_MODEL_IDS


def _honest_verdict(improvement_pp: float, headline_result_allowed: bool) -> str:
    if improvement_pp > 0.0:
        label = str(round(float(improvement_pp), 1)).replace(".", "_")
        return (
            f"grpo_v7_jury_rl_positive_improvement_{label}pp"
            if headline_result_allowed
            else f"grpo_v7_jury_rl_positive_non_headline_{label}pp"
        )
    if improvement_pp < 0.0:
        return "grpo_v7_jury_rl_regression"
    return "grpo_v7_jury_rl_no_improvement"


def _improvement_pp(baseline_passes: int, final_passes: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * ((final_passes / total) - (baseline_passes / total)), 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _wall_budget_hit(started: float, wall_budget_s: int) -> bool:
    return (time.perf_counter() - started) >= float(wall_budget_s)


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                return str(first.get("text") or first.get("message") or "")
    return str(response or "")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "pairs", "items", "examples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _label_from_row(row: Mapping[str, Any]) -> int | None:
    if "is_correct" in row:
        return 0 if bool(row["is_correct"]) else 1
    raw = row.get("label", row.get("step_correct", row.get("sc_energy_label")))
    if isinstance(raw, bool):
        return 0 if raw else 1
    if isinstance(raw, (int, float)):
        return 0 if int(raw) == 1 else 1
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"correct", "true", "supported", "entailed", "1"}:
            return 0
        if normalized in {"incorrect", "wrong", "false", "violated", "violation", "0"}:
            return 1
    return None


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


def _balanced_subset(cases: Sequence[FoVerJuryCase], target_cases: int) -> list[FoVerJuryCase]:
    if target_cases <= 0 or len(cases) <= target_cases:
        return list(cases)
    incorrect = [idx for idx, case in enumerate(cases) if int(case.label) == 1]
    correct = [idx for idx, case in enumerate(cases) if int(case.label) == 0]
    if not incorrect or not correct:
        return list(cases[:target_cases])
    target_incorrect = min(len(incorrect), max(1, target_cases // 2))
    target_correct = min(len(correct), max(1, target_cases - target_incorrect))
    selected = set(incorrect[:target_incorrect] + correct[:target_correct])
    for idx in range(len(cases)):
        if len(selected) >= target_cases:
            break
        selected.add(idx)
    return [cases[idx] for idx in sorted(selected)]


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    return compact[:limit]


def _cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _import_llama_class() -> type[Any]:  # pragma: no cover - live GGUF only.
    from carnot.reporting.triggered_certificate_v7_truncproof_sota import _import_llama_class

    return _import_llama_class()


def _short_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:240]


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
