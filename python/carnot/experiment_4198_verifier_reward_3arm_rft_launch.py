"""Exp 4198 detached launch for code verifier-reward 3-arm LoRA-RFT.

Spec refs: REQ-CODE-4198, SCENARIO-CODE-4198-GATED-LAUNCH,
SCENARIO-CODE-4198-HONEST-DEFERRAL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot import experiment_4197_verifier_reward_phase0_headroom as exp4197


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4198_verifier_reward_3arm_rft_launch.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_A1_ARTIFACT = REPO_ROOT / "results" / exp4197.RESULT_FILENAME
DEFAULT_GENERATION_CHECKPOINT = exp4197.DEFAULT_PHASE0_CHECKPOINT
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "results" / "verifier_reward_3arm_lora_rft"
RUNNER_PATH = REPO_ROOT / "scripts" / "experiments" / "verifier_reward_code_lora_rft_3arm.py"
RANDOM_SEED = 4198
PHASE0_PRECISION_THRESHOLD = 0.85
SPEC_REFS = [
    "REQ-CODE-4198",
    "SCENARIO-CODE-4198-GATED-LAUNCH",
    "SCENARIO-CODE-4198-HONEST-DEFERRAL",
]
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "training_launched",
    "stable_checkpoint_path",
    "arm_corpus_sizes",
    "gold_control_early_read",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean launch + live checkpointing is COMPLETE for this split half; "
        "an honest gated deferral when A1 found no clean operating point is also COMPLETE."
    ),
    "training_launched": (
        "BARE bool: the 3-arm LoRA-RFT is running detached and checkpointing; A3's collect gate depends on it."
    ),
    "stable_checkpoint_path": (
        "The corpus+base+config-keyed checkpoint A3 reads + future windows resume "
        "(resume-not-restart, NOT exp-id-keyed)."
    ),
    "arm_corpus_sizes": (
        "{A,B,C} N-matched counts -- |A|=|B| is the de-confound precondition "
        "(the spurious-reward control must be size-matched to the certified arm)."
    ),
    "gold_control_early_read": (
        "Arm C vs base early signal -- if gold-SFT degrades below base the HARNESS is broken "
        "(over-fit/forgetting); A3 must NOT report A-vs-B until the gold control passes."
    ),
    "model_specs": "The NON-Qwen base + the on-policy generator; required methodology for a live-LLM training artifact.",
    "random_seed": "Determinism precondition; torch generation + LoRA init seeded so the run is reproducible across windows.",
    "reproducibility_checksum": "Hash of the corpora + LoRA config; lets A3 / a third party confirm the same training inputs.",
}
LORA_CONFIG = {
    "method": "LoRA-SFT",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "max_length": 1024,
    "target_modules": [
        "q_proj.linear",
        "k_proj.linear",
        "v_proj.linear",
        "o_proj.linear",
        "gate_proj.linear",
        "up_proj.linear",
        "down_proj.linear",
    ],
}


@dataclass(frozen=True)
class DetachedProcess:
    """A detached process handle that can be serialized into the launch artifact."""

    pid: int
    returncode: int | None
    log_path: Path


@dataclass(frozen=True)
class LaunchPlan:
    """All stable paths and inputs needed to launch or resume the 3-arm run."""

    stable_checkpoint_path: Path
    command: list[str]
    corpora: exp4197.ThreeArmCorpora
    arm_corpus_sizes: dict[str, int]
    model_specs: dict[str, Any]
    operating_point: dict[str, Any]
    reproducibility_checksum: str
    runner_artifact_path: Path
    log_path: Path


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


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else 0.0
    return 0.0


def _a1_gate_clears(a1_payload: Mapping[str, Any]) -> bool:
    return (
        _float(a1_payload.get("phase0_precision")) >= PHASE0_PRECISION_THRESHOLD
        and _float(a1_payload.get("youden_j")) > 0.0
        and bool(a1_payload.get("harness_ready")) is True
    )


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


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_detached_process(
    command: Sequence[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
    env: Mapping[str, str],
) -> DetachedProcess:  # pragma: no cover - exercised by the real launch, unit tests monkeypatch it
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log.open("ab")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(env),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    time.sleep(30.0)
    return DetachedProcess(pid=int(process.pid), returncode=process.poll(), log_path=log)


def _arm_sizes(corpora: exp4197.ThreeArmCorpora) -> dict[str, int]:
    return {
        "A": len(corpora.arm_a_certified),
        "B": len(corpora.arm_b_random_control),
        "C": len(corpora.arm_c_hidden_gold),
        "D": len(corpora.arm_d_cold_base),
    }


def _corpora_payload(corpora: exp4197.ThreeArmCorpora) -> dict[str, Any]:
    return {
        "A": _jsonable(corpora.arm_a_certified),
        "B": _jsonable(corpora.arm_b_random_control),
        "C": _jsonable(corpora.arm_c_hidden_gold),
        "D": _jsonable(corpora.arm_d_cold_base),
    }


def _model_specs(a1_payload: Mapping[str, Any], generation_checkpoint: str | Path) -> dict[str, Any]:
    a1_specs = a1_payload.get("model_specs") if isinstance(a1_payload.get("model_specs"), Mapping) else {}
    operating_point = a1_payload.get("operating_point") if isinstance(a1_payload.get("operating_point"), Mapping) else {}
    trainable_base = str(a1_specs.get("trainable_base") or operating_point.get("base") or exp4197.TRAINABLE_BASE)
    on_policy_generator = str(operating_point.get("base") or trainable_base)
    return {
        "trainable_base": trainable_base,
        "trainable_base_is_non_qwen": "qwen" not in trainable_base.lower(),
        "on_policy_generator": on_policy_generator,
        "generation_checkpoint": str(generation_checkpoint),
        "a1_operating_point": _jsonable(operating_point),
        "a1_artifact_checksum": str(a1_payload.get("reproducibility_checksum") or ""),
        "runner": str(RUNNER_PATH.relative_to(REPO_ROOT)),
        "lora_config": _jsonable(LORA_CONFIG),
        "qwen_train_base_forbidden": True,
    }


def _operating_point(a1_payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = a1_payload.get("operating_point")
    return _jsonable(raw) if isinstance(raw, Mapping) else {}


def _checksum_payload(
    *,
    corpora: exp4197.ThreeArmCorpora,
    model_specs: Mapping[str, Any],
    operating_point: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    return {
        "corpora": _corpora_payload(corpora),
        "lora_config": LORA_CONFIG,
        "model_specs": model_specs,
        "operating_point": operating_point,
        "random_seed": int(random_seed),
        "spec_refs": SPEC_REFS,
    }


def reproducibility_checksum(
    *,
    corpora: exp4197.ThreeArmCorpora,
    model_specs: Mapping[str, Any],
    operating_point: Mapping[str, Any],
    random_seed: int,
) -> str:
    encoded = json.dumps(
        _checksum_payload(
            corpora=corpora,
            model_specs=model_specs,
            operating_point=operating_point,
            random_seed=random_seed,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def stable_run_key(checksum: str) -> str:
    digest = checksum.split(":", 1)[-1]
    return f"code_verifier_reward_lora_rft_{digest[:16]}"


def build_training_command(
    *,
    generation_checkpoint: str | Path,
    stable_checkpoint_path: str | Path,
    runner_artifact_path: str | Path,
    random_seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--train",
        "--checkpoint",
        str(generation_checkpoint),
        "--seed",
        str(int(random_seed)),
        "--out",
        str(runner_artifact_path),
        "--train-root",
        str(Path(stable_checkpoint_path) / "arms"),
    ]


def prepare_launch(
    *,
    a1_payload: Mapping[str, Any],
    generation_checkpoint: str | Path,
    checkpoint_root: str | Path,
    random_seed: int,
) -> LaunchPlan:
    tasks = exp4197.load_checkpoint_tasks(generation_checkpoint)
    corpora = exp4197.build_three_arm_corpora(tasks, seed=random_seed)
    sizes = _arm_sizes(corpora)
    model_specs = _model_specs(a1_payload, generation_checkpoint)
    operating_point = _operating_point(a1_payload)
    checksum = reproducibility_checksum(
        corpora=corpora,
        model_specs=model_specs,
        operating_point=operating_point,
        random_seed=random_seed,
    )
    stable_path = Path(checkpoint_root) / stable_run_key(checksum)
    runner_artifact = stable_path / "runner_artifact.json"
    command = build_training_command(
        generation_checkpoint=generation_checkpoint,
        stable_checkpoint_path=stable_path,
        runner_artifact_path=runner_artifact,
        random_seed=random_seed,
    )
    return LaunchPlan(
        stable_checkpoint_path=stable_path,
        command=command,
        corpora=corpora,
        arm_corpus_sizes=sizes,
        model_specs=model_specs,
        operating_point=operating_point,
        reproducibility_checksum=checksum,
        runner_artifact_path=runner_artifact,
        log_path=stable_path / "train.log",
    )


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def materialize_checkpoint_inputs(plan: LaunchPlan) -> None:
    plan.stable_checkpoint_path.mkdir(parents=True, exist_ok=True)
    corpora_dir = plan.stable_checkpoint_path / "corpora"
    _write_jsonl(corpora_dir / "arm_A.jsonl", plan.corpora.arm_a_certified)
    _write_jsonl(corpora_dir / "arm_B.jsonl", plan.corpora.arm_b_random_control)
    _write_jsonl(corpora_dir / "arm_C.jsonl", plan.corpora.arm_c_hidden_gold)
    manifest = {
        "stable_checkpoint_path": str(plan.stable_checkpoint_path),
        "arm_corpus_sizes": plan.arm_corpus_sizes,
        "model_specs": plan.model_specs,
        "operating_point": plan.operating_point,
        "lora_config": LORA_CONFIG,
        "reproducibility_checksum": plan.reproducibility_checksum,
        "runner_artifact_path": str(plan.runner_artifact_path),
        "log_path": str(plan.log_path),
    }
    (plan.stable_checkpoint_path / "checkpoint_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _existing_live_run(stable_checkpoint_path: Path) -> dict[str, Any] | None:
    state_path = stable_checkpoint_path / "launch_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    pid = int(state.get("pid") or 0)
    if not _pid_is_alive(pid):
        return None
    return {
        "status": "existing_live_run",
        "pid": pid,
        "command": _jsonable(state.get("command") or []),
        "log_path": str(state.get("log_path") or stable_checkpoint_path / "train.log"),
        "detached": True,
    }


def _write_launch_state(plan: LaunchPlan, launch_status: Mapping[str, Any]) -> None:
    state = {
        "updated_at": _utc_now(),
        "pid": int(launch_status.get("pid") or 0),
        "status": str(launch_status.get("status") or ""),
        "command": plan.command,
        "log_path": str(plan.log_path),
        "stable_checkpoint_path": str(plan.stable_checkpoint_path),
        "reproducibility_checksum": plan.reproducibility_checksum,
    }
    (plan.stable_checkpoint_path / "launch_state.json").write_text(
        json.dumps(_jsonable(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gold_control_early_read(stable_checkpoint_path: Path, base_passrate: float) -> dict[str, Any]:
    path = stable_checkpoint_path / "gold_control_early_read.json"
    if path.is_file():
        return load_json(path)
    return {
        "available": False,
        "status": "pending_training_checkpoint",
        "arm_c_passrate": None,
        "base_passrate": float(base_passrate),
        "arm_c_minus_base": None,
        "gold_control_gate": "pending; A3 must not report A-vs-B until Arm C >= base",
    }


def _blocked_artifact(
    *,
    verdict: str,
    training_launched: bool,
    random_seed: int,
    duration_s: float,
    preconditions: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "experiment": "experiment_4198_verifier_reward_3arm_rft_launch",
        "schema": "carnot.experiment_4198_verifier_reward_3arm_rft_launch.v1",
        "honest_verdict": verdict,
        "training_launched": bool(training_launched),
        "stable_checkpoint_path": "",
        "arm_corpus_sizes": {"A": 0, "B": 0, "C": 0, "D": 0},
        "gold_control_early_read": {"available": False, "status": "not_launched"},
        "model_specs": {"trainable_base": exp4197.TRAINABLE_BASE, "trainable_base_is_non_qwen": True},
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "preconditions": _jsonable(preconditions),
        "launch_status": {"status": verdict, "pid": None, "detached": False},
        "duration_s": round(float(duration_s), 6),
    }
    payload["reproducibility_checksum"] = _payload_checksum(payload)
    return payload


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_jsonable(filtered), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _complete_artifact(
    *,
    plan: LaunchPlan,
    launch_status: Mapping[str, Any],
    training_launched: bool,
    random_seed: int,
    duration_s: float,
    preconditions: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = (
        "complete: verifier_reward_3arm_lora_rft_launched"
        if training_launched
        else "blocked_training_process_exited_before_checkpoint"
    )
    base_passrate = _float(plan.operating_point.get("base_passrate"))
    return {
        "experiment": "experiment_4198_verifier_reward_3arm_rft_launch",
        "schema": "carnot.experiment_4198_verifier_reward_3arm_rft_launch.v1",
        "honest_verdict": verdict,
        "training_launched": bool(training_launched),
        "stable_checkpoint_path": str(plan.stable_checkpoint_path),
        "arm_corpus_sizes": dict(plan.arm_corpus_sizes),
        "gold_control_early_read": _gold_control_early_read(plan.stable_checkpoint_path, base_passrate),
        "model_specs": _jsonable(plan.model_specs),
        "random_seed": int(random_seed),
        "reproducibility_checksum": plan.reproducibility_checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "operating_point": _jsonable(plan.operating_point),
        "lora_config": _jsonable(LORA_CONFIG),
        "preconditions": _jsonable(preconditions),
        "launch_status": _jsonable(launch_status),
        "accumulated_N": dict(plan.arm_corpus_sizes),
        "truncation_guard": {
            "max_allowed_truncation_rate": exp4197.MAX_ALLOWED_TRUNCATION,
            "a1_truncation_rate": _float(plan.operating_point.get("truncation_rate")),
            "status": "instrumented_in_runner_artifact",
        },
        "runner_artifact_path": str(plan.runner_artifact_path),
        "duration_s": round(float(duration_s), 6),
    }


def write_artifact(artifact: Mapping[str, Any], path: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    a1_artifact_path: str | Path = DEFAULT_A1_ARTIFACT,
    generation_checkpoint: str | Path = DEFAULT_GENERATION_CHECKPOINT,
    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    started = time.time()
    a1_payload = load_json(a1_artifact_path)
    gate_clears = _a1_gate_clears(a1_payload)
    preconditions: dict[str, Any] = {
        "a1_gate_clears": gate_clears,
        "phase0_precision": _float(a1_payload.get("phase0_precision")),
        "youden_j": _float(a1_payload.get("youden_j")),
        "harness_ready": bool(a1_payload.get("harness_ready")),
    }
    if not gate_clears:
        artifact = _blocked_artifact(
            verdict="complete_verifier_reward_train_deferred_no_clean_operating_point",
            training_launched=False,
            random_seed=random_seed,
            duration_s=time.time() - started,
            preconditions=preconditions,
        )
        write_artifact(artifact, output_path)
        return artifact

    cuda_available = _cuda_is_available()
    preconditions["cuda_available"] = cuda_available
    if not cuda_available:
        artifact = _blocked_artifact(
            verdict="blocked_cuda_unavailable",
            training_launched=False,
            random_seed=random_seed,
            duration_s=time.time() - started,
            preconditions=preconditions,
        )
        write_artifact(artifact, output_path)
        return artifact

    _seed_torch(random_seed)
    plan = prepare_launch(
        a1_payload=a1_payload,
        generation_checkpoint=generation_checkpoint,
        checkpoint_root=checkpoint_root,
        random_seed=random_seed,
    )
    materialize_checkpoint_inputs(plan)
    preconditions["arm_corpus_sizes"] = dict(plan.arm_corpus_sizes)
    preconditions["arms_n_matched"] = plan.arm_corpus_sizes["A"] > 0 and plan.arm_corpus_sizes["A"] == plan.arm_corpus_sizes["B"]
    preconditions["gold_control_present"] = plan.arm_corpus_sizes["C"] > 0
    if not preconditions["arms_n_matched"] or not preconditions["gold_control_present"]:
        artifact = _blocked_artifact(
            verdict="blocked_3arm_corpus_unmatched",
            training_launched=False,
            random_seed=random_seed,
            duration_s=time.time() - started,
            preconditions=preconditions,
        )
        artifact["stable_checkpoint_path"] = str(plan.stable_checkpoint_path)
        artifact["arm_corpus_sizes"] = dict(plan.arm_corpus_sizes)
        artifact["model_specs"] = _jsonable(plan.model_specs)
        artifact["reproducibility_checksum"] = plan.reproducibility_checksum
        write_artifact(artifact, output_path)
        return artifact

    existing = _existing_live_run(plan.stable_checkpoint_path)
    if existing is not None:
        artifact = _complete_artifact(
            plan=plan,
            launch_status=existing,
            training_launched=True,
            random_seed=random_seed,
            duration_s=time.time() - started,
            preconditions=preconditions,
        )
        write_artifact(artifact, output_path)
        return artifact

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CARNOT_VERIFIER_REWARD_STABLE_CHECKPOINT_PATH"] = str(plan.stable_checkpoint_path)
    process = _start_detached_process(plan.command, cwd=REPO_ROOT, log_path=plan.log_path, env=env)
    alive = process.returncode is None and _pid_is_alive(process.pid)
    launch_status = {
        "status": "launched_detached_alive" if alive else "process_exited_early",
        "pid": process.pid,
        "returncode": process.returncode,
        "log_path": str(process.log_path),
        "command": plan.command,
        "detached": True,
        "detached_method": "setsid_start_new_session",
    }
    _write_launch_state(plan, launch_status)
    artifact = _complete_artifact(
        plan=plan,
        launch_status=launch_status,
        training_launched=alive,
        random_seed=random_seed,
        duration_s=time.time() - started,
        preconditions=preconditions,
    )
    write_artifact(artifact, output_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by result script launch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--a1-artifact", type=Path, default=DEFAULT_A1_ARTIFACT)
    parser.add_argument("--generation-checkpoint", type=Path, default=DEFAULT_GENERATION_CHECKPOINT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    artifact = run(
        output_path=args.out,
        a1_artifact_path=args.a1_artifact,
        generation_checkpoint=args.generation_checkpoint,
        checkpoint_root=args.checkpoint_root,
        random_seed=args.seed,
    )
    print(f"-> {artifact['honest_verdict']}")
    print(f"   training_launched={artifact['training_launched']}")
    print(f"   stable_checkpoint_path={artifact['stable_checkpoint_path']}")
    print(f"   arm_corpus_sizes={artifact['arm_corpus_sizes']}")
    return 0 if artifact["training_launched"] or artifact["honest_verdict"].startswith("complete_verifier_reward_train_deferred") else 1
