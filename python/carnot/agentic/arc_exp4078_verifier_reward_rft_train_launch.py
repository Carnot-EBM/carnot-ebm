"""Exp 4078 detached LoRA train launcher for verifier-reward RFT arms.

Spec refs: REQ-LEARN-4078, SCENARIO-LEARN-4078-BLOCKED,
SCENARIO-LEARN-4078-LAUNCH.

This module separates launch accounting from the expensive HF training work.
The launcher first records every resource it needs, then either writes an
honest blocked artifact or starts detached workers with stable checkpoint
directories. The stable directories matter because conductor windows are
shorter than a full ladder run; progress must accumulate across windows rather
than vanish when the agent exits.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot.agentic.arc_exp4077_verifier_reward_rft_corpus_build import (
    RESULT_FILENAME as EXP4077_RESULT_FILENAME,
    check_cuda_visible,
    check_hf_safetensors_model,
    check_trainer_imports,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4078_verifier_reward_rft_train_launch.json"
CHECKPOINT_ROOT = Path("results/checkpoints/experiment_4078_verifier_reward_rft_train")
LOG_ROOT = Path("results/logs/experiment_4078_verifier_reward_rft_train")
INFERENCE_SUBSTRATE = "gpu_hf_trl_peft_lora_sfttrainer_detached"
TERMINAL_PREFIXES = ("complete:", "blocked_", "failed:")
ARMS = ("rft_correct", "rft_ablation", "gold_sft")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "train_launched",
    "checkpoint_paths",
    "epochs_completed",
    "inference_substrate",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check that decides whether training may launch."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaseModelSpec:
    """One trainable HF base in the 4078 ladder."""

    key: str
    model_id: str
    trust_remote_code: bool


@dataclass(frozen=True)
class TrainingConfig:
    """Shared LoRA/SFT knobs; every arm uses identical values."""

    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    random_seed: int = 4078

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["save_strategy"] = "epoch"
        return payload


@dataclass(frozen=True)
class WorkerSpec:
    """Launch-time command and paths for one base+arm training worker."""

    repo_root: Path
    base: BaseModelSpec
    arm: str
    corpus_path: Path
    checkpoint_path: Path
    log_path: Path
    command: list[str]
    training_config: TrainingConfig


@dataclass(frozen=True)
class LaunchedWorker:
    """Detached process metadata recorded in the terminal artifact."""

    base_key: str
    arm: str
    pid: int
    command: list[str]
    checkpoint_path: str
    log_path: str
    detached: bool
    started: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BASE_MODELS = (
    BaseModelSpec("qwen35_08b", "Qwen/Qwen3.5-0.8B", False),
    BaseModelSpec("minicpm5_1b", "openbmb/MiniCPM5-1B", True),
)
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def _run_key(base_key: str, arm: str) -> str:
    return f"{base_key}:{arm}"


def _corpus_paths(repo_root: str | Path = REPO_ROOT) -> dict[str, Path]:
    root = Path(repo_root) / "results"
    return {arm: root / f"experiment_4077_{arm}.jsonl" for arm in ARMS}


def _jsonl_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def check_exp4077_corpora(*, repo_root: str | Path = REPO_ROOT) -> PreconditionCheck:
    """REQ-LEARN-4078-1: require complete Exp 4077 corpora before training."""

    root = Path(repo_root)
    artifact_path = root / "results" / EXP4077_RESULT_FILENAME
    if not artifact_path.exists():
        return PreconditionCheck("exp4077_corpora_missing", False, f"missing artifact {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return PreconditionCheck("exp4077_corpora_missing", False, f"malformed artifact: {exc}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith("complete:"):
        return PreconditionCheck("exp4077_corpora_missing", False, f"Exp 4077 not complete: {verdict}")

    counts = {arm: _jsonl_count(path) for arm, path in _corpus_paths(root).items()}
    missing = [arm for arm, count in counts.items() if count <= 0]
    if missing:
        return PreconditionCheck(
            "exp4077_corpora_missing",
            False,
            "missing or empty " + ",".join(missing),
        )
    detail = ", ".join(f"{arm}={count}" for arm, count in sorted(counts.items()))
    return PreconditionCheck("exp4077_corpora", True, detail)


def check_preconditions(*, repo_root: str | Path = REPO_ROOT) -> list[PreconditionCheck]:  # pragma: no cover
    """REQ-LEARN-4078-1: check compute resources before any worker launch."""

    return [
        PreconditionCheck(**check_hf_safetensors_model("Qwen/Qwen3.5-0.8B").to_dict()),
        PreconditionCheck(
            **check_hf_safetensors_model(
                "openbmb/MiniCPM5-1B",
                trust_remote_code=True,
            ).to_dict()
        ),
        PreconditionCheck(**check_trainer_imports().to_dict()),
        PreconditionCheck(**check_cuda_visible().to_dict()),
        check_exp4077_corpora(repo_root=repo_root),
    ]


def _first_missing(checks: Sequence[PreconditionCheck]) -> PreconditionCheck | None:
    return next((check for check in checks if not check.available), None)


def stable_checkpoint_paths(
    *,
    repo_root: str | Path = REPO_ROOT,
    bases: Sequence[BaseModelSpec] = BASE_MODELS,
    arms: Sequence[str] = ARMS,
) -> dict[str, Path]:
    """REQ-LEARN-4078-3: return stable base+arm checkpoint directories."""

    root = Path(repo_root) / CHECKPOINT_ROOT
    return {
        _run_key(base.key, arm): root / base.key / arm
        for base in bases
        for arm in arms
    }


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def epochs_completed_from_checkpoint(checkpoint_path: str | Path) -> int:
    """REQ-LEARN-4078: recover completed epoch progress from HF checkpoints."""

    root = Path(checkpoint_path)
    checkpoints = sorted(root.glob("checkpoint-*"), key=_checkpoint_step)
    if not checkpoints:
        return 0
    latest = checkpoints[-1]
    state_path = latest / "trainer_state.json"
    if not state_path.exists():
        return len(checkpoints)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return max(0, int(float(state.get("epoch", 0))))
    except (json.JSONDecodeError, TypeError, ValueError):
        return len(checkpoints)


def collect_epochs_completed(paths: Mapping[str, Path]) -> dict[str, int]:
    return {key: epochs_completed_from_checkpoint(path) for key, path in paths.items()}


def _default_python_executable(repo_root: Path) -> Path:
    candidate = repo_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path("python")


def build_worker_specs(
    *,
    repo_root: str | Path = REPO_ROOT,
    bases: Sequence[BaseModelSpec] = BASE_MODELS,
    arms: Sequence[str] = ARMS,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    python_executable: str | Path | None = None,
) -> list[WorkerSpec]:
    """Build one detached worker command per base+arm in priority order."""

    root = Path(repo_root)
    script = root / "scripts" / "experiments" / "exp4078_verifier_reward_rft_train_launch.py"
    executable = Path(python_executable) if python_executable is not None else _default_python_executable(root)
    corpora = _corpus_paths(root)
    checkpoint_paths = stable_checkpoint_paths(repo_root=root, bases=bases, arms=arms)
    specs: list[WorkerSpec] = []
    for base in bases:
        for arm in arms:
            checkpoint_path = checkpoint_paths[_run_key(base.key, arm)]
            log_path = root / LOG_ROOT / base.key / f"{arm}.log"
            command = [
                str(executable),
                str(script),
                "--worker",
                "--base-key",
                base.key,
                "--base-model",
                base.model_id,
                "--arm",
                arm,
                "--corpus-path",
                str(corpora[arm]),
                "--checkpoint-path",
                str(checkpoint_path),
                "--log-path",
                str(log_path),
                "--trust-remote-code",
                "true" if base.trust_remote_code else "false",
                "--training-config-json",
                json.dumps(training_config.to_dict(), sort_keys=True),
            ]
            specs.append(
                WorkerSpec(
                    repo_root=root,
                    base=base,
                    arm=arm,
                    corpus_path=corpora[arm],
                    checkpoint_path=checkpoint_path,
                    log_path=log_path,
                    command=command,
                    training_config=training_config,
                )
            )
    return specs


def launch_worker(spec: WorkerSpec) -> LaunchedWorker:
    """REQ-LEARN-4078-2: start one worker in a new POSIX session."""

    spec.checkpoint_path.mkdir(parents=True, exist_ok=True)
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            spec.command,
            cwd=str(spec.repo_root),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    started = process.poll() is None
    return LaunchedWorker(
        base_key=spec.base.key,
        arm=spec.arm,
        pid=int(process.pid),
        command=list(spec.command),
        checkpoint_path=str(spec.checkpoint_path),
        log_path=str(spec.log_path),
        detached=True,
        started=started,
    )


def _preconditions_payload(checks: Sequence[PreconditionCheck]) -> list[dict[str, object]]:
    return [check.to_dict() for check in checks]


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal state prevents a missing resource from becoming a fabricated training claim.",
        "train_launched": "Separates an actual detached launch from a precondition-only or blocked artifact.",
        "checkpoint_paths": "Stable base+arm paths let future windows resume accumulated LoRA progress.",
        "epochs_completed": "Progress is credit even when a later window must resume before final eval.",
        "inference_substrate": "Declares that this is HF TRL/PEFT GPU training, not an offline proxy.",
    }


def artifact_schema_errors(artifact: Mapping[str, object]) -> list[str]:
    """Validate the bare Exp 4078 launch artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")

    if "train_launched" in artifact and type(artifact["train_launched"]) is not bool:
        errors.append("train_launched must be a bare bool")

    checkpoint_paths = artifact.get("checkpoint_paths")
    if not isinstance(checkpoint_paths, dict):
        errors.append("checkpoint_paths must be a dict")
    elif any(not isinstance(key, str) or not isinstance(value, str) for key, value in checkpoint_paths.items()):
        errors.append("checkpoint_paths keys and values must be strings")

    epochs_completed = artifact.get("epochs_completed")
    if not isinstance(epochs_completed, dict):
        errors.append("epochs_completed must be a dict")
    elif any(not isinstance(key, str) or type(value) is not int for key, value in epochs_completed.items()):
        errors.append("epochs_completed values must be bare ints")

    if isinstance(checkpoint_paths, dict) and isinstance(epochs_completed, dict):
        if set(checkpoint_paths) != set(epochs_completed):
            errors.append("checkpoint_paths and epochs_completed keys must match")

    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the Exp 4078 substrate")

    launched_workers = artifact.get("launched_workers", [])
    if artifact.get("train_launched") is True and not launched_workers:
        errors.append("train_launched artifacts must include launched_workers")
    if isinstance(launched_workers, list):
        for worker in launched_workers:
            if not isinstance(worker, Mapping):
                errors.append("launched_workers entries must be dicts")
                break
            if worker.get("detached") is not True:
                errors.append("launched workers must be detached")
                break
    return errors


def _base_artifact(
    *,
    honest_verdict: str,
    train_launched: bool,
    checkpoint_paths: Mapping[str, Path],
    epochs_completed: Mapping[str, int],
    preconditions_checked: Sequence[PreconditionCheck],
    launched_workers: Sequence[LaunchedWorker],
    training_config: TrainingConfig,
    duration_s: float,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": "experiment_4078_verifier_reward_rft_train_launch",
        "schema": "carnot.experiment_4078_verifier_reward_rft_train_launch.v1",
        "honest_verdict": honest_verdict,
        "train_launched": bool(train_launched),
        "checkpoint_paths": {key: str(path) for key, path in checkpoint_paths.items()},
        "epochs_completed": dict(epochs_completed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _preconditions_payload(preconditions_checked),
        "launched_workers": [worker.to_dict() for worker in launched_workers],
        "training_config": training_config.to_dict(),
        "field_principles": _field_principles(),
        "duration_s": float(duration_s),
        "spec_refs": ["REQ-LEARN-4078", "SCENARIO-LEARN-4078-BLOCKED", "SCENARIO-LEARN-4078-LAUNCH"],
    }
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover
        raise ValueError("; ".join(errors))
    return artifact


def build_blocked_artifact(
    *,
    missing: PreconditionCheck,
    checkpoint_paths: Mapping[str, Path],
    epochs_completed: Mapping[str, int],
    preconditions_checked: Sequence[PreconditionCheck],
    training_config: TrainingConfig,
    duration_s: float,
) -> dict[str, object]:
    """REQ-LEARN-4078-1: write a blocked terminal artifact before launch."""

    return _base_artifact(
        honest_verdict=f"blocked_{missing.resource}",
        train_launched=False,
        checkpoint_paths=checkpoint_paths,
        epochs_completed=epochs_completed,
        preconditions_checked=preconditions_checked,
        launched_workers=(),
        training_config=training_config,
        duration_s=duration_s,
    )


def build_launched_artifact(
    *,
    checkpoint_paths: Mapping[str, Path],
    epochs_completed: Mapping[str, int],
    preconditions_checked: Sequence[PreconditionCheck],
    launched_workers: Sequence[LaunchedWorker],
    training_config: TrainingConfig,
    duration_s: float,
) -> dict[str, object]:
    """SCENARIO-LEARN-4078-LAUNCH: summarize detached worker launch."""

    all_started = bool(launched_workers) and all(worker.started and worker.detached for worker in launched_workers)
    suffix = "" if all_started else "_smoke_pending"
    return _base_artifact(
        honest_verdict=f"complete: rft_3arm_train_launched_detached{suffix}",
        train_launched=bool(launched_workers),
        checkpoint_paths=checkpoint_paths,
        epochs_completed=epochs_completed,
        preconditions_checked=preconditions_checked,
        launched_workers=launched_workers,
        training_config=training_config,
        duration_s=duration_s,
    )


def write_result_artifact(artifact: Mapping[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    preconditions_checker: Callable[..., Sequence[PreconditionCheck]] = check_preconditions,
    worker_launcher: Callable[[WorkerSpec], LaunchedWorker] = launch_worker,
    bases: Sequence[BaseModelSpec] = BASE_MODELS,
    arms: Sequence[str] = ARMS,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> dict[str, object]:
    """REQ-LEARN-4078: check resources, launch detached workers, and write JSON."""

    start = time.perf_counter()
    root = Path(repo_root)
    output = Path(output_path) if output_path is not None else root / "results" / RESULT_FILENAME
    checkpoint_paths = stable_checkpoint_paths(repo_root=root, bases=bases, arms=arms)
    epochs_completed = collect_epochs_completed(checkpoint_paths)
    checks = list(preconditions_checker(repo_root=root))
    missing = _first_missing(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing=missing,
            checkpoint_paths=checkpoint_paths,
            epochs_completed=epochs_completed,
            preconditions_checked=checks,
            training_config=training_config,
            duration_s=time.perf_counter() - start,
        )
        write_result_artifact(artifact, output)
        return artifact

    specs = build_worker_specs(
        repo_root=root,
        bases=bases,
        arms=arms,
        training_config=training_config,
    )
    launched_workers = [worker_launcher(spec) for spec in specs]
    artifact = build_launched_artifact(
        checkpoint_paths=checkpoint_paths,
        epochs_completed=epochs_completed,
        preconditions_checked=checks,
        launched_workers=launched_workers,
        training_config=training_config,
        duration_s=time.perf_counter() - start,
    )
    write_result_artifact(artifact, output)
    return artifact


def _latest_checkpoint(path: Path) -> str | None:  # pragma: no cover
    checkpoints = sorted(path.glob("checkpoint-*"), key=_checkpoint_step)
    return str(checkpoints[-1]) if checkpoints else None


def run_worker(  # pragma: no cover
    *,
    base_key: str,
    base_model: str,
    arm: str,
    corpus_path: str | Path,
    checkpoint_path: str | Path,
    log_path: str | Path,
    trust_remote_code: bool,
    training_config_json: str,
) -> None:
    """Train one LoRA arm with TRL SFTTrainer and epoch checkpoints."""

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    config_payload = json.loads(training_config_json)
    checkpoint = Path(checkpoint_path)
    log = Path(log_path)
    checkpoint.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    launch_manifest = checkpoint / "launch_manifest.json"
    launch_manifest.write_text(
        json.dumps(
            {
                "base_key": base_key,
                "base_model": base_model,
                "arm": arm,
                "corpus_path": str(corpus_path),
                "checkpoint_path": str(checkpoint),
                "save_strategy": "epoch",
                "training_config": config_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    records = []
    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records.append({"text": f"[{arm}]\n{row['text']}"})
    if not records:
        raise RuntimeError(f"empty corpus for {arm}: {corpus_path}")

    set_seed(int(config_payload["random_seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    peft_config = LoraConfig(
        r=int(config_payload["lora_rank"]),
        lora_alpha=int(config_payload["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    args = SFTConfig(
        output_dir=str(checkpoint),
        logging_dir=str(log.parent),
        num_train_epochs=int(config_payload["num_train_epochs"]),
        max_steps=int(config_payload["max_steps"]),
        per_device_train_batch_size=int(config_payload["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config_payload["gradient_accumulation_steps"]),
        learning_rate=float(config_payload["learning_rate"]),
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        max_length=int(config_payload["max_length"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(records),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=_latest_checkpoint(checkpoint))
    trainer.save_model(str(checkpoint / "final_adapter"))
