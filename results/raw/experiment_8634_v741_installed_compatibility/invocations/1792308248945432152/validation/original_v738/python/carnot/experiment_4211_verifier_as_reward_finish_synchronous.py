"""Exp 4211 synchronous verifier-as-reward A-vs-B finish.

Spec refs: REQ-CODE-4211, SCENARIO-CODE-4211-BLOCKED-PRECONDITION,
SCENARIO-CODE-4211-SYNC-ACCUMULATE, SCENARIO-CODE-4211-VERDICT-GATES.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4211_verifier_as_reward_finish_synchronous.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_LAUNCH_ARTIFACT = REPO_ROOT / "results" / "experiment_4198_verifier_reward_3arm_rft_launch.json"
DEFAULT_STABLE_CHECKPOINT = (
    REPO_ROOT
    / "results"
    / "verifier_reward_3arm_lora_rft"
    / "code_verifier_reward_lora_rft_a83b52882c198954"
)
THREE_ARM_RUNNER = REPO_ROOT / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"
DEFAULT_GENERATION_CHECKPOINT = REPO_ROOT / "results" / "offarc_power_sync_gemma12b_evalplus_k5.checkpoint.json"
RANDOM_SEED = 4198
BOOTSTRAP_RESAMPLES = 2000
MAX_ALLOWED_TRUNCATION_RATE = 0.05
APPROVED_NONQWEN_BASES = (
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "openbmb/MiniCPM5-1B",
)
SPEC_REFS = [
    "REQ-CODE-4211",
    "SCENARIO-CODE-4211-BLOCKED-PRECONDITION",
    "SCENARIO-CODE-4211-SYNC-ACCUMULATE",
    "SCENARIO-CODE-4211-VERDICT-GATES",
]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_label_carries_signal",
    "a_vs_b_delta",
    "a_vs_b_ci95",
    "positive_control_confirmed",
    "youden_j",
    "accumulated_n",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean 'label carries signal', a clean 'A~=B distillation null', "
        "and an honest 'accumulating/underpowered' are ALL COMPLETE -- the project's first "
        "clean verifier-as-reward read either way."
    ),
    "verifier_label_carries_signal": (
        "BARE bool: a_vs_b_ci95 excludes 0 AND delta>0 -- the de-confounded answer to the "
        "operator's pivot. NOT a_vs_cold_base (that conflates the verifier's label with the "
        "spurious-reward confound)."
    ),
    "a_vs_b_delta": (
        "pass@1(certified A) - pass@1(random-label B) on held-out hidden tests -- isolates "
        "the verifier's LABEL from the generator's intelligence (arXiv:2506.10947 / "
        "2509.20837 open question)."
    ),
    "a_vs_b_ci95": (
        "Task-level bootstrap CI95 of the A-vs-B delta -- excluding 0 distinguishes a real "
        "lift from the spurious-reward floor."
    ),
    "positive_control_confirmed": (
        "BARE bool: gold-control (Arm C >= base) AND truncation_rate<5% on all arms -- "
        "without these the A-vs-B number is INVALID, not a finding."
    ),
    "youden_j": (
        "BARE float TPR-FPR of the execution certifier vs hidden-pass -- the arXiv:2601.04411 "
        "precondition; an exact execution label should show J>>0."
    ),
    "accumulated_n": (
        "Running total eval N across windows (resume-accumulate) -- a single window adding "
        "tasks is PROGRESS; the accumulate-floor auto-retires only after 3 no-usable windows."
    ),
    "verifier_is_oracle": (
        "BARE bool=true -- HONEST: the reward is the execution oracle (RLVR/RLEF reward axis), "
        "NOT a moat claim; declaring it true keeps the result out of the oracle-distinct "
        "headline lane (Circularity Discipline)."
    ),
    "model_specs": (
        "The NON-Qwen base + the on-policy generator; required methodology for a live-LLM "
        "training artifact."
    ),
    "random_seed": (
        "Determinism precondition; torch generation + LoRA init seeded so the run is "
        "reproducible across windows."
    ),
    "reproducibility_checksum": (
        "Hash of the corpora + LoRA config; lets a third party confirm the same training inputs."
    ),
}


@dataclass(frozen=True)
class CachedBase:
    """One approved non-Qwen model cache selected for training."""

    model_id: str
    cache_path: Path


@dataclass(frozen=True)
class TrainingContext:
    """Everything the synchronous trainer needs to resume the stable checkpoint."""

    stable_checkpoint_path: Path
    manifest: Mapping[str, Any]
    corpus_paths: Mapping[str, Path]
    corpus_sizes: Mapping[str, int]
    cached_base: CachedBase
    random_seed: int
    mode: str = "in_process"
    progress_interval_s: float = 30.0


@dataclass(frozen=True)
class TrainingOutcome:
    """Serializable summary of a synchronous train/resume attempt."""

    status: str
    per_arm: Mapping[str, Any]
    accumulated_train_examples: Mapping[str, int]
    runner_artifact_path: Path
    progress_events: Sequence[Mapping[str, Any]]
    used_detached_process: bool
    error: str | None = None


@dataclass(frozen=True)
class EvaluationOutcome:
    """Held-out hidden-test eval summary for the A/B/C/D contrast."""

    status: str
    pass_at_1: Mapping[str, float]
    truncation_rate: Mapping[str, float]
    task_rows: Sequence[Mapping[str, Any]]
    seeds: Sequence[int]
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    memorization_shortcut_diagnostic: Mapping[str, Any] | None = None
    error: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    return default


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _hf_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def find_cached_nonqwen_base(
    *,
    model_ids: Sequence[str] = APPROVED_NONQWEN_BASES,
    hub_root: str | Path | None = None,
) -> CachedBase | None:
    root = Path(hub_root) if hub_root is not None else Path.home() / ".cache" / "huggingface" / "hub"
    for model_id in model_ids:
        if "qwen" in model_id.lower():
            continue
        cache_path = root / _hf_cache_name(model_id)
        if cache_path.is_dir():
            return CachedBase(model_id=model_id, cache_path=cache_path)
    return None


def _cuda_is_available() -> bool:  # pragma: no cover - live environment probe
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _seed_torch(seed: int) -> None:  # pragma: no cover - torch install/GPU dependent
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def load_checkpoint_context(stable_checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, int]]:
    stable = Path(stable_checkpoint_path)
    manifest = load_json(stable / "checkpoint_manifest.json")
    corpus_paths = {
        "A": stable / "corpora" / "arm_A.jsonl",
        "B": stable / "corpora" / "arm_B.jsonl",
        "C": stable / "corpora" / "arm_C.jsonl",
    }
    missing = [str(path) for path in corpus_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing stable corpora: {missing}")
    corpus_sizes = {arm: sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) for arm, path in corpus_paths.items()}
    corpus_sizes["D"] = 0
    return manifest, corpus_paths, corpus_sizes


def reproducibility_checksum(
    *,
    stable_checkpoint_path: str | Path,
    manifest: Mapping[str, Any],
    corpus_paths: Mapping[str, Path],
    random_seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(_jsonable(manifest), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("utf-8"))
    for arm in sorted(corpus_paths):
        digest.update(str(arm).encode("utf-8"))
        digest.update(Path(corpus_paths[arm]).read_bytes())
    digest.update(str(Path(stable_checkpoint_path)).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _truthy_score(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return 1.0 if _finite_float(value) > 0.0 else 0.0


def a_vs_b_delta(task_rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not task_rows:
        return None
    deltas = [_truthy_score(row, "A") - _truthy_score(row, "B") for row in task_rows]
    return sum(deltas) / len(deltas)


def a_vs_b_ci95(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> list[float] | None:
    if not task_rows:
        return None
    rng = random.Random(seed)
    deltas = [_truthy_score(row, "A") - _truthy_score(row, "B") for row in task_rows]
    n = len(deltas)
    if n == 0:
        return None
    means: list[float] = []
    for _ in range(int(resamples)):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_index = max(0, min(len(means) - 1, int(0.025 * len(means))))
    high_index = max(0, min(len(means) - 1, int(0.975 * len(means)) - 1))
    return [round(float(means[low_index]), 12), round(float(means[high_index]), 12)]


def _max_truncation_rate(truncation_rate: Mapping[str, float]) -> float:
    if not truncation_rate:
        return 1.0
    return max(_finite_float(value, default=1.0) for value in truncation_rate.values())


def _positive_control_confirmed(evaluation: EvaluationOutcome, manifest: Mapping[str, Any]) -> bool:
    pass_at_1 = evaluation.pass_at_1
    operating_point = manifest.get("operating_point") if isinstance(manifest.get("operating_point"), Mapping) else {}
    base = _finite_float(pass_at_1.get("D"), default=_finite_float(operating_point.get("base_passrate"), default=0.0))
    arm_c = _finite_float(pass_at_1.get("C"), default=-1.0)
    return arm_c >= base and _max_truncation_rate(evaluation.truncation_rate) < MAX_ALLOWED_TRUNCATION_RATE


def _memorization_diagnostic(
    *,
    evaluation: EvaluationOutcome | None,
    stable_checkpoint_path: Path,
) -> dict[str, Any]:
    if evaluation is not None and evaluation.memorization_shortcut_diagnostic is not None:
        return dict(evaluation.memorization_shortcut_diagnostic)
    return {
        "status": "pending_eval",
        "answer_token_perplexity_proxy": None,
        "wrong_label_shortcut_probe": None,
        "source": str(stable_checkpoint_path),
    }


def _model_specs(
    *,
    manifest: Mapping[str, Any],
    cached_base: CachedBase | None,
) -> dict[str, Any]:
    raw = manifest.get("model_specs") if isinstance(manifest.get("model_specs"), Mapping) else {}
    trainable_base = cached_base.model_id if cached_base is not None else str(raw.get("trainable_base") or "")
    specs = dict(raw)
    specs.update(
        {
            "trainable_base": trainable_base,
            "trainable_base_cache_path": str(cached_base.cache_path) if cached_base is not None else "",
            "trainable_base_is_non_qwen": bool(trainable_base and "qwen" not in trainable_base.lower()),
            "on_policy_generator": str(raw.get("on_policy_generator") or raw.get("trainable_base") or trainable_base),
            "qwen_train_base_forbidden": True,
        }
    )
    return _jsonable(specs)


def _empty_training(stable_checkpoint_path: Path) -> TrainingOutcome:
    return TrainingOutcome(
        status="not_started",
        per_arm={},
        accumulated_train_examples={"A": 0, "B": 0, "C": 0, "D": 0},
        runner_artifact_path=stable_checkpoint_path / "runner_artifact.json",
        progress_events=[],
        used_detached_process=False,
    )


def _empty_evaluation(status: str = "not_run", error: str | None = None) -> EvaluationOutcome:
    return EvaluationOutcome(
        status=status,
        pass_at_1={},
        truncation_rate={},
        task_rows=[],
        seeds=[],
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        error=error,
    )


def _artifact_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_jsonable(filtered), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_result_artifact(
    *,
    preconditions: Mapping[str, Any],
    stable_checkpoint_path: Path,
    manifest: Mapping[str, Any],
    corpus_sizes: Mapping[str, int],
    cached_base: CachedBase | None,
    training: TrainingOutcome,
    evaluation: EvaluationOutcome | None,
    adversarial_report: Mapping[str, Any] | None,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    evaluation = evaluation or _empty_evaluation()
    delta = a_vs_b_delta(evaluation.task_rows) if evaluation.status == "complete" else None
    ci95 = (
        a_vs_b_ci95(evaluation.task_rows, resamples=evaluation.bootstrap_resamples, seed=random_seed)
        if evaluation.status == "complete"
        else None
    )
    positive_control = evaluation.status == "complete" and _positive_control_confirmed(evaluation, manifest)
    label_carries_signal = bool(
        positive_control
        and delta is not None
        and delta > 0.0
        and ci95 is not None
        and ci95[0] > 0.0
    )
    if not preconditions.get("nonqwen_base_cached", True):
        verdict = "blocked_no_nonqwen_base_cached"
    elif not preconditions.get("cuda_available", True):
        verdict = "blocked_cuda_unavailable"
    elif evaluation.status != "complete":
        verdict = "progress: accumulating_verifier_reward_training_no_eval_yet"
    elif not positive_control:
        verdict = "invalid: TRAINING_INVALID_gold_control_or_truncation_failed"
    elif label_carries_signal:
        verdict = "complete: verifier_label_carries_signal"
    else:
        verdict = "complete: a_vs_b_distillation_null"

    pass_at_1 = dict(evaluation.pass_at_1)
    a_vs_c = None
    a_vs_d = None
    if evaluation.status == "complete":
        if "A" in pass_at_1 and "C" in pass_at_1:
            a_vs_c = _finite_float(pass_at_1.get("A")) - _finite_float(pass_at_1.get("C"))
        if "A" in pass_at_1 and "D" in pass_at_1:
            a_vs_d = _finite_float(pass_at_1.get("A")) - _finite_float(pass_at_1.get("D"))

    accumulated_train = dict(training.accumulated_train_examples)
    accumulated_n = {
        "train_A": int(accumulated_train.get("A", 0)),
        "train_B": int(accumulated_train.get("B", 0)),
        "train_C": int(accumulated_train.get("C", 0)),
        "train_D": int(accumulated_train.get("D", 0)),
        "eval": len(evaluation.task_rows) if evaluation.status == "complete" else 0,
    }
    payload = {
        "experiment": "experiment_4211_verifier_as_reward_finish_synchronous",
        "schema": "carnot.experiment_4211_verifier_as_reward_finish_synchronous.v1",
        "honest_verdict": verdict,
        "verifier_label_carries_signal": label_carries_signal,
        "a_vs_b_delta": None if delta is None else float(delta),
        "a_vs_b_ci95": ci95,
        "positive_control_confirmed": bool(positive_control),
        "youden_j": _finite_float(
            manifest.get("youden_j")
            or (manifest.get("preconditions", {}) if isinstance(manifest.get("preconditions"), Mapping) else {}).get("youden_j"),
            default=0.0,
        ),
        "accumulated_n": accumulated_n,
        "verifier_is_oracle": True,
        "model_specs": _model_specs(manifest=manifest, cached_base=cached_base),
        "random_seed": int(random_seed),
        "reproducibility_checksum": str(manifest.get("reproducibility_checksum") or ""),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "arm_corpus_sizes": {str(key): int(value) for key, value in corpus_sizes.items()},
        "preconditions": _jsonable(preconditions),
        "training": _jsonable(training),
        "evaluation": _jsonable(evaluation),
        "pass_at_1": _jsonable(pass_at_1),
        "a_vs_c_delta": a_vs_c,
        "a_vs_d_delta": a_vs_d,
        "truncation_guard": {
            "max_allowed_truncation_rate": MAX_ALLOWED_TRUNCATION_RATE,
            "max_observed_truncation_rate": _max_truncation_rate(evaluation.truncation_rate)
            if evaluation.status == "complete"
            else None,
        },
        "memorization_shortcut_diagnostic": _memorization_diagnostic(
            evaluation=evaluation,
            stable_checkpoint_path=stable_checkpoint_path,
        ),
        "adversarial_verify": _jsonable(adversarial_report),
        "acceptance_gate": {
            "condition": (
                "positive_control_confirmed true AND a_vs_b_delta + a_vs_b_ci95 reported "
                "(verifier_label_carries_signal resolved), OR an honest accumulating/invalid/retired verdict"
            ),
            "satisfied": bool(
                (positive_control and delta is not None and ci95 is not None)
                or verdict.startswith(("blocked_", "progress:", "invalid:", "complete_verifier_reward_retired"))
            ),
        },
        "created_at": _utc_now(),
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": "live_gpu_synchronous_lora_sft" if training.status != "not_started" else "deterministic_checkpoint_preflight",
    }
    if not payload["reproducibility_checksum"]:
        payload["reproducibility_checksum"] = _artifact_checksum(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _load_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location("verifier_reward_code_lora_rft_3arm", THREE_ARM_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner {THREE_ARM_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verifier_reward_code_lora_rft_3arm", module)
    spec.loader.exec_module(module)
    return module


def train_in_process(context: TrainingContext) -> TrainingOutcome:  # pragma: no cover - live GPU path
    _seed_torch(context.random_seed)
    runner = _load_runner_module()
    runner_artifact = Path(context.manifest.get("runner_artifact_path") or context.stable_checkpoint_path / "runner_artifact.json")
    raw_specs = context.manifest.get("model_specs") if isinstance(context.manifest.get("model_specs"), Mapping) else {}
    generation_checkpoint = Path(raw_specs.get("generation_checkpoint") or DEFAULT_GENERATION_CHECKPOINT)
    artifact = runner.run(
        checkpoint=generation_checkpoint,
        seed=context.random_seed,
        smoke=False,
        train=True,
        output_path=runner_artifact,
        train_root=context.stable_checkpoint_path / "arms",
        progress_interval_s=context.progress_interval_s,
    )
    training = artifact.get("training") if isinstance(artifact.get("training"), Mapping) else {}
    per_arm = {
        "A": training.get("arm_a", {}),
        "B": training.get("arm_b", {}),
        "C": training.get("arm_c", {}),
        "D": training.get("arm_d", {}),
    }
    accumulated = {
        "A": int((per_arm["A"] or {}).get("completed_steps") or (per_arm["A"] or {}).get("steps") or 0),
        "B": int((per_arm["B"] or {}).get("completed_steps") or (per_arm["B"] or {}).get("steps") or 0),
        "C": int((per_arm["C"] or {}).get("completed_steps") or (per_arm["C"] or {}).get("steps") or 0),
        "D": 0,
    }
    statuses = [str(row.get("status") or "") for row in per_arm.values() if isinstance(row, Mapping)]
    complete = statuses and all(status in {"trained", "trained_resume_complete", "cold_base_eval_only"} for status in statuses)
    return TrainingOutcome(
        status="complete" if complete else "partial",
        per_arm=per_arm,
        accumulated_train_examples=accumulated,
        runner_artifact_path=runner_artifact,
        progress_events=artifact.get("progress_events", []) if isinstance(artifact.get("progress_events"), list) else [],
        used_detached_process=False,
    )


def load_eval_if_available(stable_checkpoint_path: Path) -> EvaluationOutcome:
    for name in ("heldout_eval.json", "eval.json", "runner_eval.json"):
        path = stable_checkpoint_path / name
        if not path.is_file():
            continue
        payload = load_json(path)
        return EvaluationOutcome(
            status=str(payload.get("status") or "complete"),
            pass_at_1=payload.get("pass_at_1", {}) if isinstance(payload.get("pass_at_1"), Mapping) else {},
            truncation_rate=payload.get("truncation_rate", {}) if isinstance(payload.get("truncation_rate"), Mapping) else {},
            task_rows=payload.get("task_rows", []) if isinstance(payload.get("task_rows"), list) else [],
            seeds=payload.get("seeds", []) if isinstance(payload.get("seeds"), list) else [],
            bootstrap_resamples=int(payload.get("bootstrap_resamples") or BOOTSTRAP_RESAMPLES),
            memorization_shortcut_diagnostic=payload.get("memorization_shortcut_diagnostic")
            if isinstance(payload.get("memorization_shortcut_diagnostic"), Mapping)
            else None,
            error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        )
    return _empty_evaluation(status="pending_heldout_eval")


def _merge_launch_metadata(manifest: dict[str, Any], launch_artifact_path: str | Path) -> dict[str, Any]:
    path = Path(launch_artifact_path)
    if not path.is_file():
        return manifest
    launch = load_json(path)
    merged = dict(manifest)
    if "youden_j" not in merged:
        merged["youden_j"] = launch.get("youden_j") or (
            launch.get("preconditions", {}) if isinstance(launch.get("preconditions"), Mapping) else {}
        ).get("youden_j")
    if "preconditions" not in merged and isinstance(launch.get("preconditions"), Mapping):
        merged["preconditions"] = launch["preconditions"]
    if "model_specs" not in merged and isinstance(launch.get("model_specs"), Mapping):
        merged["model_specs"] = launch["model_specs"]
    if "operating_point" not in merged and isinstance(launch.get("operating_point"), Mapping):
        merged["operating_point"] = launch["operating_point"]
    return merged


def run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_CHECKPOINT,
    launch_artifact_path: str | Path = DEFAULT_LAUNCH_ARTIFACT,
    random_seed: int = RANDOM_SEED,
    train_callback: Callable[[TrainingContext], TrainingOutcome] | None = None,
) -> dict[str, Any]:
    started = time.time()
    stable = Path(stable_checkpoint_path)
    cached_base = find_cached_nonqwen_base()
    manifest: dict[str, Any] = {}
    corpus_paths: dict[str, Path] = {}
    corpus_sizes = {"A": 0, "B": 0, "C": 0, "D": 0}
    preconditions: dict[str, Any] = {
        "nonqwen_base_cached": cached_base is not None,
        "cached_base": _jsonable(cached_base),
        "stable_checkpoint_path": str(stable),
    }
    try:
        manifest, corpus_paths, corpus_sizes = load_checkpoint_context(stable)
        manifest = _merge_launch_metadata(manifest, launch_artifact_path)
        preconditions["stable_checkpoint_readable"] = True
        preconditions["arm_corpus_sizes"] = dict(corpus_sizes)
        preconditions["arms_n_matched"] = corpus_sizes.get("A", 0) > 0 and corpus_sizes.get("A") == corpus_sizes.get("B")
        if preconditions["arms_n_matched"]:
            manifest["reproducibility_checksum"] = reproducibility_checksum(
                stable_checkpoint_path=stable,
                manifest=manifest,
                corpus_paths=corpus_paths,
                random_seed=random_seed,
            )
    except Exception as exc:
        preconditions["stable_checkpoint_readable"] = False
        preconditions["stable_checkpoint_error"] = f"{type(exc).__name__}: {exc}"

    cuda_available = _cuda_is_available()
    preconditions["cuda_available"] = cuda_available
    training = _empty_training(stable)
    evaluation = _empty_evaluation(status="not_run")

    if cached_base is not None and cuda_available and preconditions.get("stable_checkpoint_readable") and preconditions.get("arms_n_matched"):
        context = TrainingContext(
            stable_checkpoint_path=stable,
            manifest=manifest,
            corpus_paths=corpus_paths,
            corpus_sizes=corpus_sizes,
            cached_base=cached_base,
            random_seed=random_seed,
            mode="in_process",
        )
        callback = train_callback or train_in_process
        try:
            training = callback(context)
        except Exception as exc:  # live training failures must still leave an honest artifact
            training = TrainingOutcome(
                status="failed",
                per_arm={},
                accumulated_train_examples={"A": 0, "B": 0, "C": 0, "D": 0},
                runner_artifact_path=stable / "runner_artifact.json",
                progress_events=[],
                used_detached_process=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        evaluation = load_eval_if_available(stable)

    artifact = build_result_artifact(
        preconditions=preconditions,
        stable_checkpoint_path=stable,
        manifest=manifest,
        corpus_sizes=corpus_sizes,
        cached_base=cached_base,
        training=training,
        evaluation=evaluation,
        adversarial_report=None,
        random_seed=random_seed,
        duration_s=time.time() - started,
    )
    write_artifact(artifact, output_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by result script
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_STABLE_CHECKPOINT)
    parser.add_argument("--launch-artifact", type=Path, default=DEFAULT_LAUNCH_ARTIFACT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    artifact = run(
        output_path=args.out,
        stable_checkpoint_path=args.stable_checkpoint,
        launch_artifact_path=args.launch_artifact,
        random_seed=args.seed,
    )
    print(f"-> {artifact['honest_verdict']}")
    print(f"   verifier_label_carries_signal={artifact['verifier_label_carries_signal']}")
    print(f"   a_vs_b_delta={artifact['a_vs_b_delta']} ci95={artifact['a_vs_b_ci95']}")
    print(f"   accumulated_n={artifact['accumulated_n']}")
    return 0 if artifact["acceptance_gate"]["satisfied"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
