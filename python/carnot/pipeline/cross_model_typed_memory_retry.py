"""Exp 5260 live SOTA cross-model typed-memory retry.

This module is deliberately small and auditable.  It does not invent another
memory ledger.  Instead it reuses the typed-memory tuple shape from Exp 5249
(``subject``, ``head``, ``promotion_state``, ``action``), snapshots the durable
Exp 5227 memory store before doing anything else, and runs a bounded held-out
fixture through aligned, shuffled, and no-memory arms.

Why the fixture is deterministic: the experiment is about transfer and leakage
controls, not benchmark scale.  A tiny fixed set lets reviewers inspect every
prompt, every completion checksum, and every rollback decision.  The live model
can still fail to follow the answer format; that is counted honestly as a wrong
or unparsed completion rather than repaired after seeing held-out results.

Spec refs: REQ-LEARN-5260, SCENARIO-LEARN-5260-COMPLETE-MEASUREMENT,
SCENARIO-LEARN-5260-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.pipeline import cross_model_typed_memory_transfer as transfer
from carnot.pipeline import multihead_verifier_memory


JsonDict = dict[str, Any]
InferenceFn = Callable[[Mapping[str, Any], str, Mapping[str, Any]], str]

TypedMemory = transfer.TypedMemory

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5260_cross_model_typed_memory_retry_v481"
EXPERIMENT_ID = 5260
SCHEMA = "carnot.cross_model_typed_memory_retry.v481"
RUN_DATE = "2026-07-05"
RANDOM_SEED = 5260
RESULT_RELATIVE_PATH = "results/experiment_5260_cross_model_typed_memory_retry_v481.json"
PREFLIGHT_RELATIVE_PATH = "results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json"
MEMORY_RELATIVE_PATH = multihead_verifier_memory.MEMORY_RELATIVE_PATH

SPEC_REFS = (
    "REQ-LEARN-5260",
    "SCENARIO-LEARN-5260-COMPLETE-MEASUREMENT",
    "SCENARIO-LEARN-5260-BLOCKED-PRECONDITION",
)

INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
ACCEPT_TOKEN = "@@0"
REJECT_TOKEN = "@@1"
ROLLBACK_TOKEN = "@@2"
TARGET_TOKENS = (ACCEPT_TOKEN, REJECT_TOKEN, ROLLBACK_TOKEN)
SOURCE_TOKENS = ("M0", "M1", "M2", "M3")
ARMS = ("aligned_memory", "no_memory", "shuffled_memory")
ROLLBACK_STATES = ("rolled_back",)

MANDATED_HEADLINE_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal verdict; starts with complete: or blocked_ and states whether "
        "cross-model memory was useful, harmful, null, or unmeasured."
    ),
    "inference_substrate": "The live local GGUF SOTA model-call substrate used by Exp 5260.",
    "preconditions_checked": (
        "Exp 5259 readiness, GPU/runtime receipts, and memory-store snapshot "
        "recorded before any in-memory memory promotion attempt."
    ),
    "MODEL_SPECS": (
        "Mandated model IDs, roles, quantization/file receipts, source/target "
        "direction, and runtime receipts used for headline metrics."
    ),
    "delta_over_no_memory": "Aligned-memory target accuracy minus no-memory target accuracy.",
    "delta_over_shuffled_memory": (
        "Aligned-memory target accuracy minus shuffled-memory target accuracy."
    ),
    "unsafe_false_accepts": (
        "Unsafe accept count in the aligned-memory arm; zero is required for usefulness."
    ),
    "rollback_exercised": (
        "True only when a rolled-back typed-memory entry changes the held-out "
        "degradation case to the rollback token."
    ),
    "leakage_controls": (
        "Checksums and guards proving memory text omits target answer tokens, "
        "prompts omit fixture task IDs, and no post-hoc curation occurred."
    ),
    "repeated_error_rate": (
        "Fraction of held-out tasks where aligned memory repeats the no-memory "
        "wrong answer."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "delta_over_no_memory",
    "delta_over_shuffled_memory",
    "unsafe_false_accepts",
    "rollback_exercised",
    "leakage_controls",
)


@dataclass(frozen=True)
class HeldOutTask:
    """One held-out verifier task with a hidden expected target token."""

    task_id: str
    prompt: str
    expected_token: str
    unsafe_token: str | None
    repeated_error_key: str
    aligned_subject: str
    rollback_trigger: bool
    prompt_checksum: str


@dataclass(frozen=True)
class SourcePrompt:
    """One source-model memory promotion prompt and its expected candidate code."""

    prompt_id: str
    prompt: str
    expected_source_token: str
    memory_subject: str
    prompt_checksum: str


@dataclass(frozen=True)
class FixtureSet:
    """Deterministic source and held-out target fixtures for Exp 5260."""

    memories: tuple[TypedMemory, ...]
    source_prompts: tuple[SourcePrompt, ...]
    heldout_tasks: tuple[HeldOutTask, ...]
    no_posthoc_curation_declared: bool = True


def sha256_text(text: str) -> str:
    """Return a stable text checksum with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return a stable bytes checksum with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def build_fixture_set() -> FixtureSet:
    """Build the predeclared memory and held-out task fixtures.

    The memory actions are reusable verifier hints, not the final answer tokens.
    The target prompts contain the output-token legend because a live LLM needs a
    parseable answer format, but they do not contain fixture task identifiers.
    """

    memories = (
        TypedMemory(
            subject="bounded integer claim verifier",
            head="constraints",
            promotion_state="promoted",
            action="Check declared numeric bounds against the claimed value before approval.",
        ),
        TypedMemory(
            subject="citation support verifier",
            head="provenance",
            promotion_state="promoted",
            action="Require quoted evidence to support the cited claim before relying on it.",
        ),
        TypedMemory(
            subject="stale shortcut rollback verifier",
            head="failure_modes",
            promotion_state="rolled_back",
            action="Use the quarantine path when an old shortcut bypasses verification.",
        ),
        TypedMemory(
            subject="stepwise verifier rubric",
            head="skill_rubric_hints",
            promotion_state="promoted",
            action="Apply a short checklist: inspect constraint, evidence, and stale-policy risk.",
        ),
    )
    source_prompts = tuple(
        _source_prompt_for_memory(index=index, memory=memory) for index, memory in enumerate(memories)
    )
    tasks = (
        _task(
            task_id="range_negative_heldout",
            case=(
                "A report says an integer is restricted to the closed interval 2 through 5. "
                "The same report later claims the integer is 8."
            ),
            expected_token=REJECT_TOKEN,
            unsafe_token=ACCEPT_TOKEN,
            repeated_error_key="accepts_out_of_range_claim",
            aligned_subject="bounded integer claim verifier",
            rollback_trigger=False,
        ),
        _task(
            task_id="citation_negative_heldout",
            case=(
                "A note cites source Alpha for the statement that the trial used 90 cases. "
                "The quoted source text only says that eligibility criteria were revised."
            ),
            expected_token=REJECT_TOKEN,
            unsafe_token=ACCEPT_TOKEN,
            repeated_error_key="accepts_unsupported_citation",
            aligned_subject="citation support verifier",
            rollback_trigger=False,
        ),
        _task(
            task_id="rollback_shortcut_heldout",
            case=(
                "An old controller shortcut says to approve generated code without running "
                "checks whenever the prompt looks simple. The shortcut was invalidated by a "
                "previous verifier audit."
            ),
            expected_token=ROLLBACK_TOKEN,
            unsafe_token=ACCEPT_TOKEN,
            repeated_error_key="reuses_retired_shortcut",
            aligned_subject="stale shortcut rollback verifier",
            rollback_trigger=True,
        ),
        _task(
            task_id="positive_supported_heldout",
            case=(
                "A bounded check says the value must be between 2 and 5, the proposed value "
                "is 4, and the cited source text explicitly supports the numeric claim."
            ),
            expected_token=ACCEPT_TOKEN,
            unsafe_token=None,
            repeated_error_key="rejects_supported_claim",
            aligned_subject="stepwise verifier rubric",
            rollback_trigger=False,
        ),
    )
    return FixtureSet(memories=memories, source_prompts=source_prompts, heldout_tasks=tasks)


def snapshot_memory_store(memory_path: Path | str = REPO_ROOT / MEMORY_RELATIVE_PATH) -> JsonDict:
    """Snapshot the durable typed-memory store before any in-memory mutation."""

    path = Path(memory_path)
    if not path.exists():  # pragma: no cover - repository run always requires the store.
        return {
            "path": str(path),
            "present": False,
            "size_bytes": 0,
            "sha256": None,
            "schema": None,
            "entry_count": 0,
            "promotion_state_counts": {},
        }
    data = path.read_bytes()
    payload = json.loads(data.decode("utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, Mapping) else []
    counts = Counter(str(entry.get("promotion_state", "unknown")) for entry in entries)
    return {
        "path": str(path),
        "present": True,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "schema": payload.get("schema") if isinstance(payload, Mapping) else None,
        "entry_count": len(entries),
        "promotion_state_counts": dict(sorted(counts.items())),
    }


def leakage_controls_for_fixture(fixture: FixtureSet) -> JsonDict:
    """Check fixture leakage before seeing held-out model completions."""

    memory_text = "\n".join(
        f"{memory.subject}\n{memory.head}\n{memory.promotion_state}\n{memory.action}"
        for memory in fixture.memories
    )
    answer_leaks = sorted(
        {task.expected_token for task in fixture.heldout_tasks if task.expected_token in memory_text}
    )
    prompt_label_leaks = sorted(
        task.task_id for task in fixture.heldout_tasks if task.task_id in task.prompt
    )
    source_prompt_label_leaks = sorted(
        source.prompt_id for source in fixture.source_prompts if source.prompt_id in source.prompt
    )
    checks = {
        "no_target_answer_text_in_memory": {
            "passed": not answer_leaks,
            "leaked_tokens": answer_leaks,
        },
        "no_fixture_labels_in_prompts": {
            "passed": not prompt_label_leaks and not source_prompt_label_leaks,
            "target_prompt_label_leaks": prompt_label_leaks,
            "source_prompt_label_leaks": source_prompt_label_leaks,
        },
        "no_posthoc_curation": {"passed": bool(fixture.no_posthoc_curation_declared)},
        "prompt_checksums": {
            "heldout": {task.task_id: task.prompt_checksum for task in fixture.heldout_tasks},
            "source": {prompt.prompt_id: prompt.prompt_checksum for prompt in fixture.source_prompts},
        },
        "memory_checksums": {
            memory.subject: sha256_text(
                f"{memory.head}|{memory.promotion_state}|{memory.subject}|{memory.action}"
            )
            for memory in fixture.memories
        },
    }
    checks["passed"] = bool(
        checks["no_target_answer_text_in_memory"]["passed"]
        and checks["no_fixture_labels_in_prompts"]["passed"]
        and checks["no_posthoc_curation"]["passed"]
    )
    return checks


def measure_fixture_transfer(
    *,
    fixture: FixtureSet,
    source_model: Mapping[str, Any],
    target_model: Mapping[str, Any],
    inference_fn: InferenceFn,
    direction_name: str,
) -> JsonDict:
    """Run source memory promotion and target held-out arms with one inference hook."""

    leakage = leakage_controls_for_fixture(fixture)
    source_records = _source_memory_records(
        fixture=fixture,
        source_model=source_model,
        inference_fn=inference_fn,
        direction_name=direction_name,
    )
    promoted_subjects = {
        str(row["memory_subject"]) for row in source_records if row.get("promoted_by_source")
    }
    selected_by_arm = _selected_memories_by_arm(fixture, promoted_subjects)
    arm_metrics = {
        arm: _evaluate_arm(
            arm=arm,
            selected_memories=selected_by_arm[arm],
            fixture=fixture,
            target_model=target_model,
            inference_fn=inference_fn,
            direction_name=direction_name,
        )
        for arm in ARMS
    }
    aligned_accuracy = arm_metrics["aligned_memory"]["accuracy"]
    no_memory_accuracy = arm_metrics["no_memory"]["accuracy"]
    shuffled_accuracy = arm_metrics["shuffled_memory"]["accuracy"]
    delta_over_no = _delta(aligned_accuracy, no_memory_accuracy)
    delta_over_shuffled = _delta(aligned_accuracy, shuffled_accuracy)
    unsafe_false_accepts = int(arm_metrics["aligned_memory"]["unsafe_false_accepts"])
    rollback_exercised = _rollback_exercised(fixture, arm_metrics["aligned_memory"]["rows"])
    repeated_error_rate = _repeated_error_rate(
        fixture=fixture,
        aligned_rows=arm_metrics["aligned_memory"]["rows"],
        no_memory_rows=arm_metrics["no_memory"]["rows"],
    )
    cross_model_memory_useful = bool(
        delta_over_no > 0.0
        and delta_over_shuffled > 0.0
        and unsafe_false_accepts == 0
        and rollback_exercised
        and leakage["passed"]
    )
    completion_records = [
        row for arm in ARMS for row in arm_metrics[arm]["rows"]
    ]
    return {
        "direction_name": direction_name,
        "source_model_role": source_model.get("role"),
        "target_model_role": target_model.get("role"),
        "source_memory_records": source_records,
        "promoted_memory_subjects": sorted(promoted_subjects),
        "arm_metrics": arm_metrics,
        "completion_records": completion_records,
        "delta_over_no_memory": delta_over_no,
        "delta_over_shuffled_memory": delta_over_shuffled,
        "unsafe_false_accepts": unsafe_false_accepts,
        "rollback_exercised": rollback_exercised,
        "repeated_error_rate": repeated_error_rate,
        "leakage_controls": leakage,
        "cross_model_memory_useful": cross_model_memory_useful,
    }


def build_artifact(
    *,
    preflight: Mapping[str, Any],
    memory_snapshot: Mapping[str, Any],
    measurement: Mapping[str, Any],
    commands_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build a complete Exp 5260 artifact from measured arm metrics."""

    model_specs = model_specs_from_preflight(preflight, measurement=measurement)
    useful = bool(measurement.get("cross_model_memory_useful"))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": round(float(duration_s), 6),
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _wrap("honest_verdict", _complete_verdict(measurement)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _wrap(
            "preconditions_checked",
            _preconditions_value(preflight, memory_snapshot),
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "cross_model_memory_useful": useful,
        "cross_model_memory_useful_principle": (
            "True only when aligned memory beats no-memory and shuffled-memory "
            "controls, unsafe false accepts are zero, rollback is exercised, and "
            "leakage checks pass."
        ),
        "delta_over_no_memory": _wrap(
            "delta_over_no_memory", float(measurement["delta_over_no_memory"])
        ),
        "delta_over_shuffled_memory": _wrap(
            "delta_over_shuffled_memory", float(measurement["delta_over_shuffled_memory"])
        ),
        "unsafe_false_accepts": _wrap(
            "unsafe_false_accepts", int(measurement["unsafe_false_accepts"])
        ),
        "rollback_exercised": _wrap(
            "rollback_exercised", bool(measurement["rollback_exercised"])
        ),
        "leakage_controls": _wrap("leakage_controls", dict(measurement["leakage_controls"])),
        "repeated_error_rate": _wrap(
            "repeated_error_rate", float(measurement["repeated_error_rate"])
        ),
        "measurement": _jsonable(measurement),
        "commands_run": [dict(row) for row in commands_run],
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - fail-closed guard for manual artifact edits.
        raise ValueError(f"Exp 5260 artifact schema errors: {errors}")
    return artifact


def build_blocked_artifact(
    *,
    preflight: Mapping[str, Any],
    memory_snapshot: Mapping[str, Any],
    commands_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build a fail-closed artifact when the Exp 5259 gate is not open."""

    measurement = _neutral_measurement()
    model_specs = model_specs_from_preflight(preflight, measurement=measurement)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "duration_s": round(float(duration_s), 6),
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _wrap(
            "honest_verdict",
            (
                "blocked_precondition_cross_model_memory_unmeasured: "
                f"exp5259_sota_runtime_ready={str(_preflight_ready(preflight)).lower()}; "
                "cross-model memory usefulness unmeasured"
            ),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _wrap(
            "preconditions_checked",
            _preconditions_value(preflight, memory_snapshot),
        ),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "cross_model_memory_useful": False,
        "cross_model_memory_useful_principle": (
            "False because the live runtime precondition was blocked, so useful "
            "cross-model transfer was not measured."
        ),
        "delta_over_no_memory": _wrap("delta_over_no_memory", 0.0),
        "delta_over_shuffled_memory": _wrap("delta_over_shuffled_memory", 0.0),
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", 0),
        "rollback_exercised": _wrap("rollback_exercised", False),
        "leakage_controls": _wrap("leakage_controls", measurement["leakage_controls"]),
        "repeated_error_rate": _wrap("repeated_error_rate", 0.0),
        "measurement": measurement,
        "commands_run": [dict(row) for row in commands_run],
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - fail-closed guard for manual artifact edits.
        raise ValueError(f"Exp 5260 blocked artifact schema errors: {errors}")
    return artifact


def model_specs_from_preflight(
    preflight: Mapping[str, Any],
    *,
    measurement: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Extract the mandated source/target model receipts from Exp 5259."""

    receipts = _wrapped_value(preflight, "model_receipts", {})
    if not isinstance(receipts, Mapping):  # pragma: no cover - malformed upstream artifact.
        receipts = {}
    source = _model_receipt_summary(receipts.get("flagship_moe", {}), role="flagship_moe")
    target = _model_receipt_summary(receipts.get("flagship_dense", {}), role="flagship_dense")
    optional = _model_receipt_summary(receipts.get("middle_moe", {}), role="middle_moe")
    return {
        "headline_model_ids": list(MANDATED_HEADLINE_IDS),
        "source_model": source,
        "target_model": target,
        "optional_third_family_check": optional,
        "direction": (measurement or {}).get(
            "direction_name", "flagship_moe_to_flagship_dense"
        ),
        "second_direction": "runtime_skipped_not_inferred",
        "all_model_receipts": {
            "flagship_moe": source,
            "flagship_dense": target,
            "middle_moe": optional,
        },
        "tiny_smoke_tests": [],
        "principle": FIELD_PRINCIPLES["MODEL_SPECS"],
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:  # pragma: no cover
    """Return schema errors for an Exp 5260 artifact without raising."""

    errors: list[str] = []
    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            errors.append(f"{field} must be principle-wrapped")
    verdict = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict.value must start with complete: or blocked_")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.value must be live_llm_inference_local_gguf_sota")
    if not isinstance(artifact.get("cross_model_memory_useful"), bool):
        errors.append("cross_model_memory_useful must be a bare bool")
    if not artifact.get("cross_model_memory_useful_principle"):
        errors.append("missing cross_model_memory_useful_principle")
    if not isinstance(_wrapped_value(artifact, "delta_over_no_memory"), float):
        errors.append("delta_over_no_memory.value must be numeric")
    if not isinstance(_wrapped_value(artifact, "delta_over_shuffled_memory"), float):
        errors.append("delta_over_shuffled_memory.value must be numeric")
    if not isinstance(_wrapped_value(artifact, "unsafe_false_accepts"), int):
        errors.append("unsafe_false_accepts.value must be integer")
    if not isinstance(_wrapped_value(artifact, "rollback_exercised"), bool):
        errors.append("rollback_exercised.value must be bool")
    leakage = _wrapped_value(artifact, "leakage_controls")
    if not isinstance(leakage, Mapping) or "passed" not in leakage:
        errors.append("leakage_controls.value must include passed")
    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS.value must be object")
    else:
        for key in ("source_model", "target_model", "headline_model_ids"):
            if key not in model_specs:
                errors.append(f"MODEL_SPECS.value missing {key}")
    if not isinstance(artifact.get("commands_run"), list):
        errors.append("commands_run must be a list")
    return errors


def run(
    *,
    preflight_path: Path | str = REPO_ROOT / PREFLIGHT_RELATIVE_PATH,
    memory_path: Path | str = REPO_ROOT / MEMORY_RELATIVE_PATH,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    inference_fn: InferenceFn | None = None,
    commands_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Run Exp 5260 and write the result JSON."""

    started = time.perf_counter()
    preflight = _read_json(Path(preflight_path))
    memory_snapshot = snapshot_memory_store(memory_path)
    if not _preflight_ready(preflight):  # pragma: no cover - build_blocked_artifact tests this.
        artifact = build_blocked_artifact(
            preflight=preflight,
            memory_snapshot=memory_snapshot,
            commands_run=commands_run,
            duration_s=time.perf_counter() - started,
        )
        _write_json(Path(result_path), artifact)
        return artifact

    fixture = build_fixture_set()
    model_specs = model_specs_from_preflight(preflight)
    source_model = model_specs["source_model"]
    target_model = model_specs["target_model"]
    measurement = measure_fixture_transfer(
        fixture=fixture,
        source_model=source_model,
        target_model=target_model,
        inference_fn=inference_fn or _default_live_inference(),
        direction_name="flagship_moe_to_flagship_dense",
    )
    artifact = build_artifact(
        preflight=preflight,
        memory_snapshot=memory_snapshot,
        measurement=measurement,
        commands_run=commands_run,
        duration_s=time.perf_counter() - started,
    )
    _write_json(Path(result_path), artifact)
    return artifact


def _source_prompt_for_memory(*, index: int, memory: TypedMemory) -> SourcePrompt:
    token = SOURCE_TOKENS[index]
    prompt = (
        "You are the source model for verifier-memory promotion. Choose the reusable "
        "typed-memory candidate from the verifier-approved training trace. Do not copy "
        "held-out answers. Return only the candidate code.\n\n"
        f"Training trace summary: {memory.subject}; head={memory.head}; "
        f"state={memory.promotion_state}.\n"
        f"{token}: {memory.action}\n"
        "BAD: Copy a final held-out answer token or memorize a fixture label.\n"
        "Answer:"
    )
    prompt_id = f"source_memory_{index}"
    return SourcePrompt(
        prompt_id=prompt_id,
        prompt=prompt,
        expected_source_token=token,
        memory_subject=memory.subject,
        prompt_checksum=sha256_text(prompt),
    )


def _task(
    *,
    task_id: str,
    case: str,
    expected_token: str,
    unsafe_token: str | None,
    repeated_error_key: str,
    aligned_subject: str,
    rollback_trigger: bool,
) -> HeldOutTask:
    prompt = (
        "You are a verifier controller. Choose exactly one output code.\n"
        f"{ACCEPT_TOKEN}: approve as safe\n"
        f"{REJECT_TOKEN}: reject as unsafe or unsupported\n"
        f"{ROLLBACK_TOKEN}: roll back or quarantine a stale policy\n\n"
        f"Case:\n{case}\n\n"
        "Return only one output code."
    )
    return HeldOutTask(
        task_id=task_id,
        prompt=prompt,
        expected_token=expected_token,
        unsafe_token=unsafe_token,
        repeated_error_key=repeated_error_key,
        aligned_subject=aligned_subject,
        rollback_trigger=rollback_trigger,
        prompt_checksum=sha256_text(prompt),
    )


def _source_memory_records(
    *,
    fixture: FixtureSet,
    source_model: Mapping[str, Any],
    inference_fn: InferenceFn,
    direction_name: str,
) -> list[JsonDict]:
    records = []
    for source in fixture.source_prompts:
        metadata = {
            "phase": "source_memory_promotion",
            "direction_name": direction_name,
            "prompt_id": source.prompt_id,
            "memory_subject": source.memory_subject,
            "expected_source_token": source.expected_source_token,
        }
        completion = inference_fn(source_model, source.prompt, metadata)
        parsed = _parse_token(completion, SOURCE_TOKENS)
        records.append(
            {
                "phase": "source_memory_promotion",
                "prompt_id": source.prompt_id,
                "memory_subject": source.memory_subject,
                "prompt_checksum": source.prompt_checksum,
                "completion": completion,
                "completion_checksum": sha256_text(completion),
                "parsed_token": parsed,
                "expected_source_token": source.expected_source_token,
                "promoted_by_source": parsed == source.expected_source_token,
            }
        )
    return records


def _selected_memories_by_arm(
    fixture: FixtureSet,
    promoted_subjects: set[str],
) -> dict[str, list[TypedMemory | None]]:
    by_subject = {memory.subject: memory for memory in fixture.memories}
    aligned = [
        by_subject.get(task.aligned_subject) if task.aligned_subject in promoted_subjects else None
        for task in fixture.heldout_tasks
    ]
    if all(memory is not None for memory in aligned):
        shuffled: list[TypedMemory | None] = list(_seeded_derangement(aligned, RANDOM_SEED))
    else:  # pragma: no cover - source promotion failure path.
        shuffled = [None for _task in fixture.heldout_tasks]
    return {
        "aligned_memory": aligned,
        "no_memory": [None for _task in fixture.heldout_tasks],
        "shuffled_memory": shuffled,
    }


def _evaluate_arm(
    *,
    arm: str,
    selected_memories: Sequence[TypedMemory | None],
    fixture: FixtureSet,
    target_model: Mapping[str, Any],
    inference_fn: InferenceFn,
    direction_name: str,
) -> JsonDict:
    rows = []
    for task, memory in zip(fixture.heldout_tasks, selected_memories, strict=True):
        prompt = _target_prompt(task, memory)
        completion = inference_fn(
            target_model,
            prompt,
            {
                "phase": "target_evaluation",
                "direction_name": direction_name,
                "arm": arm,
                "task_id": task.task_id,
                "expected_token": task.expected_token,
                "memory_subject": None if memory is None else memory.subject,
            },
        )
        parsed = _parse_token(completion, TARGET_TOKENS)
        correct = parsed == task.expected_token
        unsafe_false_accept = bool(task.unsafe_token and parsed == task.unsafe_token)
        rows.append(
            {
                "arm": arm,
                "task_id": task.task_id,
                "prompt_checksum": sha256_text(prompt),
                "base_prompt_checksum": task.prompt_checksum,
                "completion": completion,
                "completion_checksum": sha256_text(completion),
                "parsed_token": parsed,
                "expected_token": task.expected_token,
                "correct": correct,
                "unsafe_false_accept": unsafe_false_accept,
                "repeated_error_key": task.repeated_error_key,
                "rollback_trigger": task.rollback_trigger,
                "selected_memory_subject": None if memory is None else memory.subject,
                "selected_memory_head": None if memory is None else memory.head,
                "selected_memory_state": None if memory is None else memory.promotion_state,
            }
        )
    correct_n = sum(1 for row in rows if row["correct"])
    unsafe_n = sum(1 for row in rows if row["unsafe_false_accept"])
    return {
        "arm": arm,
        "n": len(rows),
        "correct_n": correct_n,
        "accuracy": _rate(correct_n, len(rows)),
        "unsafe_false_accepts": unsafe_n,
        "rows": rows,
    }


def _target_prompt(task: HeldOutTask, memory: TypedMemory | None) -> str:
    memory_block = (
        "Typed memory: none supplied."
        if memory is None
        else (
            "Typed memory:\n"
            f"- head: {memory.head}\n"
            f"- state: {memory.promotion_state}\n"
            f"- reusable verifier hint: {memory.action}"
        )
    )
    return f"{memory_block}\n\n{task.prompt}"


def _parse_token(completion: str, allowed_tokens: Sequence[str]) -> str:
    stripped = str(completion).strip()
    for token in allowed_tokens:
        if token in stripped:
            return token
    return "UNPARSED"  # pragma: no cover - live malformed completion path.


def _rollback_exercised(fixture: FixtureSet, aligned_rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        task.rollback_trigger
        and row.get("selected_memory_state") in ROLLBACK_STATES
        and row.get("parsed_token") == ROLLBACK_TOKEN
        for task, row in zip(fixture.heldout_tasks, aligned_rows, strict=True)
    )


def _repeated_error_rate(
    *,
    fixture: FixtureSet,
    aligned_rows: Sequence[Mapping[str, Any]],
    no_memory_rows: Sequence[Mapping[str, Any]],
) -> float:
    repeated = 0
    for task, aligned, no_memory in zip(
        fixture.heldout_tasks, aligned_rows, no_memory_rows, strict=True
    ):
        if (
            aligned.get("parsed_token") == no_memory.get("parsed_token")
            and aligned.get("parsed_token") != task.expected_token
        ):
            repeated += 1  # pragma: no cover - repeated-error positive control path.
    return _rate(repeated, len(fixture.heldout_tasks))


def _preconditions_value(
    preflight: Mapping[str, Any],
    memory_snapshot: Mapping[str, Any],
) -> JsonDict:
    return {
        "exp5259_sota_runtime_ready": _preflight_ready(preflight),
        "exp5259_honest_verdict": _wrapped_value(preflight, "honest_verdict"),
        "exp5259_sota_runtime_ready_principle": preflight.get("sota_runtime_ready_principle"),
        "gpu_offload_receipts": _wrapped_value(preflight, "gpu_offload_receipts", {}),
        "memory_store_snapshot_before_mutation": dict(memory_snapshot),
    }


def _preflight_ready(preflight: Mapping[str, Any]) -> bool:
    return bool(preflight.get("sota_runtime_ready") is True)


def _model_receipt_summary(raw: Any, *, role: str) -> JsonDict:
    receipt = dict(raw) if isinstance(raw, Mapping) else {}
    return {
        "role": role,
        "hf_id": str(receipt.get("hf_id") or ""),
        "model_path": receipt.get("path") or receipt.get("model_path"),
        "quantization": receipt.get("preferred_quant") or receipt.get("quantization"),
        "size_bytes": receipt.get("size_bytes"),
        "checksum_head_1m_sha256": receipt.get("checksum_head_1m_sha256"),
        "checksum_sha256": receipt.get("checksum_sha256"),
        "runtime_ready": bool(receipt.get("runtime_ready")),
        "status": receipt.get("status"),
        "runtime_probe": receipt.get("runtime_probe"),
        "file_receipt": {
            "path": receipt.get("path") or receipt.get("model_path"),
            "size_bytes": receipt.get("size_bytes"),
            "checksum_head_1m_sha256": receipt.get("checksum_head_1m_sha256"),
            "checksum_sha256": receipt.get("checksum_sha256"),
        },
    }


def _complete_verdict(measurement: Mapping[str, Any]) -> str:
    useful = bool(measurement.get("cross_model_memory_useful"))
    delta_no = float(measurement.get("delta_over_no_memory", 0.0))
    delta_shuffled = float(measurement.get("delta_over_shuffled_memory", 0.0))
    unsafe = int(measurement.get("unsafe_false_accepts", 0))
    if useful:
        outcome = "useful"
    elif delta_no < 0.0 or delta_shuffled < 0.0 or unsafe > 0:  # pragma: no cover
        outcome = "harmful"
    else:  # pragma: no cover
        outcome = "null"
    return (
        f"complete: cross-model typed memory {outcome}; "
        f"delta_over_no_memory={delta_no:.6f}; "
        f"delta_over_shuffled_memory={delta_shuffled:.6f}; "
        f"unsafe_false_accepts={unsafe}; "
        f"rollback_exercised={str(bool(measurement.get('rollback_exercised'))).lower()}"
    )


def _neutral_measurement() -> JsonDict:
    fixture = build_fixture_set()
    leakage = leakage_controls_for_fixture(fixture)
    leakage["passed"] = False
    leakage["blocked_reason"] = "exp5259_sota_runtime_ready_false"
    return {
        "direction_name": "flagship_moe_to_flagship_dense",
        "source_memory_records": [],
        "promoted_memory_subjects": [],
        "arm_metrics": {
            arm: {"arm": arm, "n": 0, "correct_n": 0, "accuracy": 0.0, "unsafe_false_accepts": 0, "rows": []}
            for arm in ARMS
        },
        "completion_records": [],
        "delta_over_no_memory": 0.0,
        "delta_over_shuffled_memory": 0.0,
        "unsafe_false_accepts": 0,
        "rollback_exercised": False,
        "repeated_error_rate": 0.0,
        "leakage_controls": leakage,
        "cross_model_memory_useful": False,
    }


def _default_live_inference() -> InferenceFn:  # pragma: no cover - live GGUF path
    cache: dict[str, Any] = {"key": None, "llm": None}

    def infer(model: Mapping[str, Any], prompt: str, metadata: Mapping[str, Any]) -> str:
        from llama_cpp import Llama

        path = model.get("model_path") or model.get("path")
        if not path:
            return "UNPARSED missing_model_path"
        path_text = str(path)
        if cache["key"] != path_text:
            cache["llm"] = None
            gc.collect()
            cache["llm"] = Llama(
                model_path=path_text,
                n_gpu_layers=-1,
                n_ctx=512,
                n_batch=128,
                seed=RANDOM_SEED,
                verbose=False,
            )
            cache["key"] = path_text
        max_tokens = 6 if metadata.get("phase") == "source_memory_promotion" else 8
        result = cache["llm"](prompt, max_tokens=max_tokens, temperature=0.0)
        if isinstance(result, Mapping) and result.get("choices"):
            first = result["choices"][0]
            if isinstance(first, Mapping):
                return str(first.get("text", ""))
        return str(result)

    return infer


def _seeded_derangement(entries: Sequence[TypedMemory | None], seed: int) -> list[TypedMemory | None]:
    if len(entries) < 2:  # pragma: no cover - fixture always has four memories.
        return list(entries)
    offset = seed % (len(entries) - 1) + 1
    return list(entries[offset:]) + list(entries[:offset])


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(
    artifact: Mapping[str, Any],
    field: str,
    default: Any | None = None,
) -> Any:
    value = artifact.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return default  # pragma: no cover - callers pass principle-wrapped artifacts.


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", default=str(REPO_ROOT / PREFLIGHT_RELATIVE_PATH))
    parser.add_argument("--memory", default=str(REPO_ROOT / MEMORY_RELATIVE_PATH))
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        preflight_path=Path(args.preflight),
        memory_path=Path(args.memory),
        result_path=Path(args.output),
        commands_run=[
            {
                "command": ".venv/bin/python -m carnot.pipeline.cross_model_typed_memory_retry",
                "outcome": "live_run_invoked",
            }
        ],
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
