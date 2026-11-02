"""Cross-model typed-memory transfer artifact builder for Exp 5249.

Spec refs: REQ-LEARN-5249, SCENARIO-LEARN-5249-BLOCKED-PRECONDITION,
SCENARIO-LEARN-5249-LIVE-TRANSFER.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5249_cross_model_typed_memory_transfer_v480"
EXPERIMENT_ID = 5249
SCHEMA = "carnot.cross_model_typed_memory_transfer.v480"
RUN_DATE = "2026-07-05"
RANDOM_SEED = 5249
RESULT_RELATIVE_PATH = "results/experiment_5249_cross_model_typed_memory_transfer_v480.json"
SPEC_REFS = (
    "REQ-LEARN-5249",
    "SCENARIO-LEARN-5249-BLOCKED-PRECONDITION",
    "SCENARIO-LEARN-5249-LIVE-TRANSFER",
)
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
MEMORY_HEADS = ("constraints", "provenance", "failure_modes", "skill_rubric_hints")
ARM_NAMES = (
    "aligned_memory",
    "shuffled_memory",
    "no_memory",
    "stale_memory",
    "rollback_triggered_memory",
)
RUNTIME_COMMAND = (
    ".venv/bin/python -m carnot.pipeline.cross_model_typed_memory_transfer "
    "--run-live-local-gguf-sota"
)
MIN_MATERIALIZED_GGUF_BYTES = 1_000_000
LIVE_INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
BLOCKED_SUBSTRATE = "precondition_check_only"

PROMPTS = {
    "producer_memory_extraction": (
        "From solved and verifier-accepted traces, extract typed memories under "
        "constraints, provenance, failure_modes, and skill_rubric_hints. Do not "
        "include held-out labels or oracle answers."
    ),
    "consumer_verifier_task": (
        "Solve the held-out verifier task using at most one supplied typed-memory "
        "entry. Report the selected action and whether rollback should fire."
    ),
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether cross-model "
        "typed memory was useful."
    ),
    "inference_substrate": (
        "Uses live_llm_inference_local_gguf_sota only when model calls run; "
        "otherwise precondition_check_only avoids a false live-inference claim."
    ),
    "model_specs": (
        "Names mandated headline GGUF models, quantization, runtime command, "
        "seeds, prompt checksums, and precondition receipts."
    ),
    "producer_model": "Producer model that would generate typed memories from solved traces.",
    "consumer_model": "Different consumer model evaluated on held-out verifier tasks.",
    "aligned_vs_shuffled_delta": "Aligned-memory accuracy minus shuffled-memory accuracy.",
    "aligned_vs_no_memory_delta": "Aligned-memory accuracy minus no-memory accuracy.",
    "stale_memory_delta": "Aligned-memory accuracy minus stale-memory accuracy.",
    "rollback_exercised": (
        "True only when rollback-triggered memory changes a risky action to a "
        "rollback, quarantine, retire, or block action."
    ),
    "retention_check_passed": (
        "True only when promoted and rolled-back memories remain retrievable and "
        "correct on held-out verifier tasks."
    ),
    "no_model_training": "True because the experiment transfers typed memories without weight updates.",
    "leakage_checks": (
        "Held-out split, prompt checksum, label visibility, and rollback-on-"
        "degradation checks prevent hidden training or answer leakage."
    ),
    "tests_run": "Commands run to verify the module, coverage, and artifact schema.",
}

REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "producer_model",
    "consumer_model",
    "aligned_vs_shuffled_delta",
    "aligned_vs_no_memory_delta",
    "stale_memory_delta",
    "rollback_exercised",
    "retention_check_passed",
    "no_model_training",
    "leakage_checks",
    "tests_run",
)
ROLLBACK_ACTION_PREFIXES = ("rollback_", "block_", "quarantine_", "retire_")


@dataclass(frozen=True)
class TypedMemory:
    """One typed-memory action candidate used by the deterministic arm evaluator."""

    subject: str
    head: str
    promotion_state: str
    action: str


@dataclass(frozen=True)
class TransferTask:
    """One held-out verifier task and its expected typed-memory action."""

    task_id: str
    query: str
    expected_subject: str
    expected_head: str
    expected_state: str
    expected_action: str
    default_action: str
    stale_subject: str
    degradation_trigger: bool = False


def check_preconditions(
    *,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    runtime_probe: Mapping[str, Any] | None = None,
    command_paths: Mapping[str, str | None] | None = None,
    min_model_bytes: int = MIN_MATERIALIZED_GGUF_BYTES,
) -> JsonDict:
    """Check local GPU, llama runtime, and materialized mandated GGUF files."""

    specs = [dict(spec) for spec in (model_specs or _default_model_specs())]
    runtime = dict(runtime_probe or _detect_runtime_probe())
    commands = dict(command_paths or _detect_command_paths())
    receipts = [
        model_file_receipt(spec, min_model_bytes=min_model_bytes) for spec in specs[:2]
    ]

    blockers: list[str] = []
    if not runtime.get("cuda_available") or int(runtime.get("cuda_device_count") or 0) < 1:
        blockers.append("blocked_cuda_gpu_unavailable")
    llama_access = bool(
        runtime.get("llama_cpp_import_ok")
        or commands.get("llama-server")
        or commands.get("llama-cli")
    )
    if not llama_access:
        blockers.append("blocked_llama_runtime_missing")
    if runtime.get("llama_cpp_import_ok") and not runtime.get("llama_cpp_supports_gpu_offload"):
        blockers.append("blocked_llama_cpp_gpu_offload")
    if len(specs) < 2:
        blockers.append("blocked_two_mandated_models_missing")
    if not _has_two_mandated_models(specs):
        blockers.append("blocked_mandated_model_pair_missing")
    if not all(receipt["materialized"] for receipt in receipts):
        blockers.append("blocked_model_file_not_materialized")

    return {
        "all_passed": not blockers,
        "blockers": sorted(set(blockers)),
        "runtime_probe": runtime,
        "command_paths": commands,
        "model_file_receipts": receipts,
        "min_model_bytes": int(min_model_bytes),
    }


def model_file_receipt(
    spec: Mapping[str, Any],
    *,
    min_model_bytes: int = MIN_MATERIALIZED_GGUF_BYTES,
) -> JsonDict:
    """Return a bounded checksum/size receipt without hashing huge GGUF weights."""

    path_text = str(spec.get("model_path") or "")
    path = Path(path_text).expanduser() if path_text else None
    present = bool(path and path.is_file())
    size = path.stat().st_size if present and path is not None else 0
    prefix = path.read_bytes()[:256] if present and path is not None else b""
    lfs_pointer = prefix.startswith(b"version https://git-lfs.github.com/spec/")
    materialized = bool(present and size >= min_model_bytes and not lfs_pointer)
    return {
        "name": str(spec.get("name") or ""),
        "hf_id": str(spec.get("hf_id") or ""),
        "quantization": str(spec.get("quantization") or _quantization_from_path(path_text)),
        "role": str(spec.get("role") or ""),
        "model_path": path_text or None,
        "present": present,
        "size_bytes": size,
        "materialized": materialized,
        "lfs_pointer": bool(lfs_pointer),
        "first_256_sha256": hashlib.sha256(prefix).hexdigest() if prefix else None,
    }


def evaluate_transfer_arms(
    tasks: Sequence[TransferTask],
    memories: Sequence[TypedMemory],
    *,
    seed: int = RANDOM_SEED,
) -> JsonDict:
    """Evaluate aligned, shuffled, no-memory, stale, and rollback arms."""

    by_subject = {memory.subject: memory for memory in memories}
    aligned = [_required_memory(by_subject, task.expected_subject) for task in tasks]
    stale = [_required_memory(by_subject, task.stale_subject) for task in tasks]
    shuffled = _seeded_derangement(aligned, seed)
    arm_metrics = {
        "aligned_memory": _arm_summary("aligned_memory", tasks, aligned),
        "shuffled_memory": _arm_summary("shuffled_memory", tasks, shuffled),
        "no_memory": _arm_summary("no_memory", tasks, [None] * len(tasks)),
        "stale_memory": _arm_summary("stale_memory", tasks, stale),
        "rollback_triggered_memory": _rollback_arm_summary(tasks, aligned),
    }
    aligned_accuracy = arm_metrics["aligned_memory"]["accuracy"]
    shuffled_accuracy = arm_metrics["shuffled_memory"]["accuracy"]
    no_memory_accuracy = arm_metrics["no_memory"]["accuracy"]
    stale_accuracy = arm_metrics["stale_memory"]["accuracy"]
    rollback_exercised = any(
        row["rollback_applied"] for row in arm_metrics["rollback_triggered_memory"]["rows"]
    )
    retention_passed = _retention_passed(tasks, aligned, arm_metrics["aligned_memory"]["rows"])
    leakage_checks = _leakage_checks(tasks, rollback_exercised=rollback_exercised)
    aligned_vs_shuffled = _delta(aligned_accuracy, shuffled_accuracy)
    aligned_vs_no = _delta(aligned_accuracy, no_memory_accuracy)
    stale_delta = _delta(aligned_accuracy, stale_accuracy)
    pass_condition_met = bool(
        aligned_vs_shuffled > 0.0
        and aligned_vs_no > 0.0
        and stale_delta >= 0.0
        and rollback_exercised
        and retention_passed
        and leakage_checks["passed"]
    )
    return {
        "arm_metrics": arm_metrics,
        "aligned_vs_shuffled_delta": aligned_vs_shuffled,
        "aligned_vs_no_memory_delta": aligned_vs_no,
        "stale_memory_delta": stale_delta,
        "rollback_exercised": rollback_exercised,
        "retention_check_passed": retention_passed,
        "leakage_checks": leakage_checks,
        "pass_condition_met": pass_condition_met,
        "no_model_training": True,
    }


def build_artifact(
    *,
    precondition_audit: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
    transfer_result: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 5249 result artifact from real preflight/evaluation receipts."""

    audit = dict(precondition_audit)
    transfer = dict(transfer_result or _neutral_transfer_result())
    live_ran = bool(audit.get("all_passed") and transfer_result)
    eligible = bool(live_ran and transfer.get("pass_condition_met"))
    substrate = LIVE_INFERENCE_SUBSTRATE if live_ran else BLOCKED_SUBSTRATE
    headline_models = [dict(row) for row in audit.get("model_file_receipts", [])]
    producer = headline_models[0] if headline_models else {}
    consumer = headline_models[1] if len(headline_models) > 1 else {}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": round(float(duration_s), 6),
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": _source_artifacts(),
        "typed_memory_heads": list(MEMORY_HEADS),
        "arms": list(ARM_NAMES),
        "predeclared_pass_condition": (
            "aligned_vs_shuffled_delta > 0, aligned_vs_no_memory_delta > 0, "
            "stale_memory_delta >= 0, rollback exercised, retention passed, "
            "all leakage checks passed, and no model training"
        ),
        "cross_model_memory_eligible": eligible,
        "cross_model_memory_eligible_principle": (
            "Bare gate for Exp5250; true only after live cross-model SOTA GGUF "
            "transfer beats controls with retention, rollback, and leakage checks."
        ),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(audit, transfer, live_ran)),
        "inference_substrate": _wrap("inference_substrate", substrate),
        "model_specs": _wrap(
            "model_specs",
            {
                "headline_models": headline_models,
                "tiny_smoke_tests": [],
                "quantization": [row.get("quantization") for row in headline_models],
                "runtime_command": RUNTIME_COMMAND,
                "seeds": [RANDOM_SEED],
                "prompts": dict(PROMPTS),
                "prompt_checksums": _prompt_checksums(),
                "precondition_audit": audit,
            },
        ),
        "producer_model": _wrap("producer_model", producer),
        "consumer_model": _wrap("consumer_model", consumer),
        "aligned_vs_shuffled_delta": _wrap(
            "aligned_vs_shuffled_delta", float(transfer["aligned_vs_shuffled_delta"])
        ),
        "aligned_vs_no_memory_delta": _wrap(
            "aligned_vs_no_memory_delta", float(transfer["aligned_vs_no_memory_delta"])
        ),
        "stale_memory_delta": _wrap("stale_memory_delta", float(transfer["stale_memory_delta"])),
        "rollback_exercised": _wrap("rollback_exercised", bool(transfer["rollback_exercised"])),
        "retention_check_passed": _wrap(
            "retention_check_passed", bool(transfer["retention_check_passed"])
        ),
        "no_model_training": _wrap("no_model_training", bool(transfer["no_model_training"])),
        "leakage_checks": _wrap("leakage_checks", dict(transfer["leakage_checks"])),
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise on malformed Exp 5249 artifacts; return True on success."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if not isinstance(artifact.get("cross_model_memory_eligible"), bool):
        raise ValueError("cross_model_memory_eligible must be bool")  # pragma: no cover
    if not artifact.get("cross_model_memory_eligible_principle"):
        raise ValueError("missing cross_model_memory_eligible_principle")  # pragma: no cover
    for field in (
        "aligned_vs_shuffled_delta",
        "aligned_vs_no_memory_delta",
        "stale_memory_delta",
    ):
        value = _wrapped_value(artifact, field)
        if not isinstance(value, float):
            raise ValueError(f"{field} must wrap a float")  # pragma: no cover
    leakage = _wrapped_value(artifact, "leakage_checks")
    if not isinstance(leakage, Mapping) or "passed" not in leakage:
        raise ValueError("leakage_checks missing passed")  # pragma: no cover
    return True


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    runtime_probe: Mapping[str, Any] | None = None,
    command_paths: Mapping[str, str | None] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    min_model_bytes: int = MIN_MATERIALIZED_GGUF_BYTES,
) -> JsonDict:
    """Run preconditions and write the honest Exp 5249 artifact."""

    started = time.monotonic()
    audit = check_preconditions(
        model_specs=model_specs,
        runtime_probe=runtime_probe,
        command_paths=command_paths,
        min_model_bytes=min_model_bytes,
    )
    artifact = build_artifact(
        precondition_audit=audit,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    _write_json(Path(result_path), artifact)
    return artifact


def _default_model_specs() -> list[JsonDict]:  # pragma: no cover - environment-dependent
    specs = cached_sota_pair() or [
        {
            "name": model["name"],
            "hf_id": model["hf_id"],
            "quantization": model["quantization"],
            "role": model["role"],
            "model_path": resolve_cached_gguf(model["hf_id"]),
        }
        for model in SOTA_GGUF_MODELS[:2]
    ]
    enriched = []
    by_hf_id = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}
    for index, spec in enumerate(specs[:2]):
        model = by_hf_id.get(str(spec.get("hf_id")), {})
        row = dict(spec)
        row.setdefault("quantization", model.get("quantization"))
        row["role"] = "producer" if index == 0 else "consumer"
        enriched.append(row)
    return enriched


def _detect_runtime_probe() -> JsonDict:  # pragma: no cover - environment-dependent
    probe: JsonDict = {}
    try:
        import torch

        probe["cuda_available"] = bool(torch.cuda.is_available())
        probe["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        probe.update({"cuda_available": False, "cuda_device_count": 0, "torch_error": repr(exc)})
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as low

        probe["llama_cpp_import_ok"] = True
        probe["llama_cpp_version"] = getattr(llama_cpp, "__version__", None)
        probe["llama_cpp_supports_gpu_offload"] = bool(low.llama_supports_gpu_offload())
    except Exception as exc:
        probe.update({"llama_cpp_import_ok": False, "llama_cpp_error": repr(exc)})
    probe["gpu_names"] = _gpu_names()
    return probe


def _detect_command_paths() -> JsonDict:  # pragma: no cover - environment-dependent
    return {"llama-server": shutil.which("llama-server"), "llama-cli": shutil.which("llama-cli")}


def _gpu_names() -> list[str]:  # pragma: no cover - environment-dependent
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _has_two_mandated_models(specs: Sequence[Mapping[str, Any]]) -> bool:
    hf_ids = [str(spec.get("hf_id") or "") for spec in specs[:2]]
    return len(set(hf_ids)) >= 2 and all(hf_id in MANDATED_MODEL_IDS for hf_id in hf_ids)


def _quantization_from_path(path_text: str) -> str:
    name = Path(path_text).name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0", "BF16"):
        if token.lower() in name.lower():
            return token
    return "unknown"


def _required_memory(by_subject: Mapping[str, TypedMemory], subject: str) -> TypedMemory:
    memory = by_subject.get(subject)
    if memory is None:
        raise ValueError(f"missing memory subject: {subject}")  # pragma: no cover
    return memory


def _arm_summary(
    arm: str,
    tasks: Sequence[TransferTask],
    selected: Sequence[TypedMemory | None],
) -> JsonDict:
    rows = [_row(task, memory) for task, memory in zip(tasks, selected, strict=True)]
    correct_n = sum(1 for row in rows if row["correct"])
    return {
        "arm": arm,
        "n": len(rows),
        "correct_n": correct_n,
        "accuracy": _rate(correct_n, len(rows)),
        "rows": rows,
    }


def _rollback_arm_summary(tasks: Sequence[TransferTask], aligned: Sequence[TypedMemory]) -> JsonDict:
    rows = []
    for task, memory in zip(tasks, aligned, strict=True):
        row = _row(task, memory)
        row["rollback_applied"] = bool(
            task.degradation_trigger
            and memory.promotion_state == "rolled_back"
            and row["selected_action"].startswith(ROLLBACK_ACTION_PREFIXES)
        )
        rows.append(row)
    correct_n = sum(1 for row in rows if row["correct"])
    return {
        "arm": "rollback_triggered_memory",
        "n": len(rows),
        "correct_n": correct_n,
        "accuracy": _rate(correct_n, len(rows)),
        "rows": rows,
    }


def _row(task: TransferTask, memory: TypedMemory | None) -> JsonDict:
    selected_action = task.default_action if memory is None else memory.action
    return {
        **asdict(task),
        "selected_subject": None if memory is None else memory.subject,
        "selected_head": None if memory is None else memory.head,
        "selected_state": None if memory is None else memory.promotion_state,
        "selected_action": selected_action,
        "correct": selected_action == task.expected_action,
    }


def _seeded_derangement(entries: Sequence[TypedMemory], seed: int) -> list[TypedMemory]:
    if len(entries) < 2:
        return list(entries)  # pragma: no cover
    offset = seed % (len(entries) - 1) + 1
    return list(entries[offset:]) + list(entries[:offset])


def _retention_passed(
    tasks: Sequence[TransferTask],
    aligned: Sequence[TypedMemory],
    rows: Sequence[Mapping[str, Any]],
) -> bool:
    states = {memory.promotion_state for memory in aligned}
    subjects_match = all(task.expected_subject == memory.subject for task, memory in zip(tasks, aligned, strict=True))
    return bool(subjects_match and {"promoted", "rolled_back"}.issubset(states) and all(row["correct"] for row in rows))


def _leakage_checks(tasks: Sequence[TransferTask], *, rollback_exercised: bool) -> JsonDict:
    label_visible = any(task.expected_action in task.query for task in tasks)
    held_out_ok = all(not _looks_like_train_split(task.task_id) for task in tasks)
    prompt_checksums = _prompt_checksums()
    checks = {
        "held_out_split_check": {"passed": held_out_ok},
        "prompt_checksum_check": {"passed": all(prompt_checksums.values()), "checksums": prompt_checksums},
        "label_visibility_check": {"passed": not label_visible},
        "rollback_on_degradation_check": {"passed": rollback_exercised},
    }
    checks["passed"] = all(bool(value["passed"]) for value in checks.values())
    return checks


def _looks_like_train_split(task_id: str) -> bool:
    lowered = task_id.lower()
    tokens = lowered.replace("-", "_").split("_")
    return "train" in tokens or "training" in tokens


def _neutral_transfer_result() -> JsonDict:
    return {
        "arm_metrics": {arm: {"arm": arm, "n": 0, "correct_n": 0, "accuracy": 0.0, "rows": []} for arm in ARM_NAMES},
        "aligned_vs_shuffled_delta": 0.0,
        "aligned_vs_no_memory_delta": 0.0,
        "stale_memory_delta": 0.0,
        "rollback_exercised": False,
        "retention_check_passed": False,
        "leakage_checks": {
            "passed": False,
            "held_out_split_check": {"passed": False, "reason": "blocked_precondition"},
            "prompt_checksum_check": {"passed": True, "checksums": _prompt_checksums()},
            "label_visibility_check": {"passed": False, "reason": "blocked_precondition"},
            "rollback_on_degradation_check": {"passed": False, "reason": "blocked_precondition"},
        },
        "pass_condition_met": False,
        "no_model_training": True,
    }


def _honest_verdict(
    audit: Mapping[str, Any],
    transfer: Mapping[str, Any],
    live_ran: bool,
) -> str:
    if not live_ran:
        blockers = ",".join(str(item) for item in audit.get("blockers", [])) or "model_calls_not_run"
        return (
            "blocked_precondition_cross_model_memory_not_measured: "
            f"cross-model memory usefulness not measured; blockers={blockers}"
        )
    if transfer.get("pass_condition_met"):
        return "complete: cross-model typed memory is useful across mandated SOTA GGUF families"
    return "complete: cross-model typed memory was measured but not useful under controls"


def _source_artifacts() -> list[str]:
    return [
        "research-program.md#continuous-self-learning-core-architectural-goal",
        "results/experiment_5227_continuous_self_learning_multihead_memory_v478.json",
        "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json",
        "ops/exclusion_manifest.yaml",
    ]


def _prompt_checksums() -> dict[str, str]:
    return {key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in PROMPTS.items()}


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    return value.get("value") if isinstance(value, Mapping) else None


def _rate(correct_n: int, total_n: int) -> float:
    return round(correct_n / total_n, 6) if total_n else 0.0


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
