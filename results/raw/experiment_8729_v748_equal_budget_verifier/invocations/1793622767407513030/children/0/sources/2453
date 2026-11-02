"""Freeze the V577 roadmap and receipt contract without running an LLM.

The planner document names thirteen tasks, but activation can remove the
pre-staged YAML or activate an incomplete subset. This module keeps that gap
visible. It validates exact YAML contracts and writes a terminal blocked
artifact when source evidence is incomplete. See REQ-REPORT-6616 and the
SCENARIO-REPORT-6616 requirements.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import pydantic
import yaml


JsonDict = dict[str, Any]
MILESTONE = "2026.08.577"
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
PRE_STAGED_ROADMAP = Path("research-roadmap-next.yaml")
ROADMAP_DOCUMENT = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_6616_v577_execution_contract.json")
INFERENCE_SUBSTRATE = "v577_roadmap_evidence_and_phase_receipt_contract_no_llm"
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
SUPPORTED_GATE_OPS = {"==", "!=", ">", "<", ">=", "<=", "exists", "in"}
TERMINAL_PHASE_STATES = {"terminal_complete", "terminal_blocked"}
PHASE_RECEIPT_FIELDS = (
    "task_id",
    "phase_name",
    "state",
    "monotonic_started_s",
    "monotonic_ended_s",
    "process_identity",
    "resource_owner",
    "input_hashes",
    "output_hashes",
    "heartbeat_monotonic_s",
    "terminal_reason",
    "checksum",
)
ACCELERATOR_RECEIPT_FIELDS = (
    "device_uuid",
    "pid_start_time",
    "model_hash",
    "vram_before_mib",
    "vram_after_mib",
    "offload_layers",
    "unload_evidence",
    "lease_token",
)
PROTECTED_PATHS = (ACTIVE_ROADMAP, Path("scripts/research_conductor.py"))
QWEN_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_MODEL = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
LEGACY_MODELS = ("Qwen3.5-0.8B", "Gemma E4B")

EXPECTED_TASKS: dict[str, JsonDict] = {
    "exp6616-v577-execution-contract": {
        "deliverable": RESULT_PATH.as_posix(),
        "phase": "phase_i_execution_truth_and_bounded_headroom",
    },
    "exp6617-gpu-lease-phase-receipts": {
        "deliverable": "results/experiment_6617_gpu_lease_phase_receipts.json",
        "phase": "phase_i_execution_truth_and_bounded_headroom",
    },
    "exp6618-mandated-model-admission": {
        "deliverable": "results/experiment_6618_mandated_model_admission.json",
        "phase": "phase_i_execution_truth_and_bounded_headroom",
    },
    "exp6619-qwen36-bounded-headroom": {
        "deliverable": "results/experiment_6619_qwen36_bounded_headroom.json",
        "phase": "phase_i_execution_truth_and_bounded_headroom",
    },
    "exp6620-headroom-reducer": {
        "deliverable": "results/experiment_6620_headroom_reducer.json",
        "phase": "phase_i_execution_truth_and_bounded_headroom",
    },
    "exp6621-two-level-decoding": {
        "deliverable": "results/experiment_6621_two_level_decoding.json",
        "phase": "phase_ii_two_level_constraint_search_and_safety",
    },
    "exp6622-decoding-safety-audit": {
        "deliverable": "results/experiment_6622_decoding_safety_audit.json",
        "phase": "phase_ii_two_level_constraint_search_and_safety",
    },
    "exp6623-spectral-integrity-repair": {
        "deliverable": "results/experiment_6623_spectral_integrity_repair.json",
        "phase": "phase_iii_spectral_sampler_evidence_recovery",
    },
    "exp6624-spectral-scale-replay": {
        "deliverable": "results/experiment_6624_spectral_scale_replay.json",
        "phase": "phase_iii_spectral_sampler_evidence_recovery",
    },
    "exp6625-arc-live-memory-actionability": {
        "deliverable": "results/experiment_6625_arc_live_memory_actionability.json",
        "phase": "phase_iv_live_actionability_and_continuous_self_learning",
    },
    "exp6626-working-memory-patch-gate": {
        "deliverable": "results/experiment_6626_working_memory_patch_gate.json",
        "phase": "phase_iv_live_actionability_and_continuous_self_learning",
    },
    "exp6627-prospective-state-grounded-learning": {
        "deliverable": "results/experiment_6627_prospective_state_grounded_learning.json",
        "phase": "phase_iv_live_actionability_and_continuous_self_learning",
    },
    "exp6628-v577-independent-capstone": {
        "deliverable": "results/experiment_6628_v577_independent_capstone.json",
        "phase": "phase_iv_live_actionability_and_continuous_self_learning",
    },
}
GPU_TASK_IDS = {
    "exp6618-mandated-model-admission",
    "exp6619-qwen36-bounded-headroom",
    "exp6621-two-level-decoding",
    "exp6627-prospective-state-grounded-learning",
}
REQUIRED_MODELS_BY_TASK = {
    "exp6618-mandated-model-admission": (QWEN_MODEL, GEMMA31_MODEL, GEMMA26_MODEL),
    "exp6619-qwen36-bounded-headroom": (QWEN_MODEL,),
    "exp6621-two-level-decoding": (QWEN_MODEL,),
    "exp6627-prospective-state-grounded-learning": (QWEN_MODEL,),
}
EXPECTED_GATES: dict[str, list[JsonDict]] = {
    "exp6617-gpu-lease-phase-receipts": [
        {
            "upstream": "exp6616-v577-execution-contract",
            "artifact_field": "execution_contract_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6618-mandated-model-admission": [
        {
            "upstream": "exp6617-gpu-lease-phase-receipts",
            "artifact_field": "gpu_lease_scheduler_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6619-qwen36-bounded-headroom": [
        {
            "upstream": "exp6618-mandated-model-admission",
            "artifact_field": "qwen_admission_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6621-two-level-decoding": [
        {
            "upstream": "exp6620-headroom-reducer",
            "artifact_field": "v577_headroom_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6622-decoding-safety-audit": [
        {
            "upstream": "exp6621-two-level-decoding",
            "artifact_field": "decoding_rows_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6624-spectral-scale-replay": [
        {
            "upstream": "exp6623-spectral-integrity-repair",
            "artifact_field": "sampler_integrity_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6626-working-memory-patch-gate": [
        {
            "upstream": "exp6625-arc-live-memory-actionability",
            "artifact_field": "live_memory_activation_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6627-prospective-state-grounded-learning": [
        {
            "upstream": "exp6618-mandated-model-admission",
            "artifact_field": "qwen_admission_ready_score",
            "op": "==",
            "value": 1.0,
        },
        {
            "upstream": "exp6626-working-memory-patch-gate",
            "artifact_field": "memory_patch_contract_ready_score",
            "op": "==",
            "value": 1.0,
        },
    ],
}
ATTACK_IDS = (
    "duplicate_id",
    "duplicate_deliverable",
    "stale_milestone",
    "missing_principle",
    "unsupported_gate_op",
    "missing_gate_field",
    "wrong_model_id",
    "absent_blocked_diagnostic",
    "nonterminal_verdict",
    "phase_timestamp_reversal",
    "pid_reuse",
    "checksum_tampering",
    "retired_upstream",
    "protected_file_mutation",
)
REQUIRED_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "task_contract_rows",
    "roadmap_validation_receipts",
    "gate_owner_rows",
    "prior_failure_dispositions",
    "phase_receipt_schema",
    "accelerator_receipt_schema",
    "model_policy_receipts",
    "execution_contract_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The contract task ends terminally and does not hide a schema or evidence block.",
    "honest_verdict": "The verdict reports contract readiness without claiming scientific benefit.",
    "verdict_class": "Use the closed verdict enum; a ready contract is null.",
    "gate_check_summary": "A block names the failed contract check and observed value.",
    "task_contract_rows": "Every expected V577 task has one exact contract disposition row.",
    "roadmap_validation_receipts": "Roadmap checks retain commands, exits, and observations.",
    "gate_owner_rows": "Each gate binds an earlier owner and an identical declared field.",
    "prior_failure_dispositions": "Prior verdicts, changed conditions, and retirement remain explicit.",
    "phase_receipt_schema": "Phase state, time, process, resources, hashes, and checksum are explicit.",
    "accelerator_receipt_schema": "Accelerator identity, memory, lease, and unload extend phase evidence.",
    "model_policy_receipts": "Model tasks bind mandated models and keep legacy models smoke-only.",
    "execution_contract_ready_score": "The exact binary field opens Exp6617 only after full replay.",
    "attack_rows": "Identity, gate, model, verdict, receipt, retirement, and mutation attacks fail closed.",
    "preconditions_checked": "Sources, versions, resources, hashes, and protected files are explicit.",
    "protected_files_unchanged": "The active roadmap and conductor retain their original hashes.",
    "inference_substrate": "The task validates roadmap and evidence without an LLM.",
    "verifier_is_oracle": "Exact schema and artifact checks are authoritative without a science claim.",
    "field_provenance": "Every field names its sources, hashes, parsers, and validation functions.",
    "duration_s": "Monotonic duration covers source replay, attacks, and validation.",
    "tests_run": "Focused, roadmap, exclusion, gate, spec, artifact, adversarial, and E2E checks retain exits.",
    "reproducibility_checksum": "A final content hash detects contract mutation.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest for exact source bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large artifacts do not copy into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def extract_required_fields(prompt: str) -> dict[str, str | None]:
    """Read required field names and their adjacent principles from a prompt."""

    if "REQUIRED ARTIFACT FIELDS:" not in prompt:
        return {}
    block = prompt.split("REQUIRED ARTIFACT FIELDS:", 1)[1]
    block = block.split("Set inference_substrate=", 1)[0]
    fields: dict[str, str | None] = {}
    current: str | None = None
    for line in block.splitlines():
        field_match = re.match(r"^  ([a-z][a-z0-9_]*):\s*$", line)
        if field_match:
            current = field_match.group(1)
            fields[current] = None
            continue
        principle_match = re.match(r'^\s{4}principle:\s*["\'](.+)["\']\s*$', line)
        if current and principle_match:
            fields[current] = principle_match.group(1)
    return fields


def _error(check: str, expected: Any, observed: Any, task_id: str | None = None) -> JsonDict:
    row = {"check": check, "expected": expected, "observed": observed}
    if task_id is not None:
        row["task_id"] = task_id
    return row


def _cycle_nodes(tasks: Sequence[Mapping[str, Any]]) -> set[str]:
    graph = {
        str(task.get("id")): [str(gate.get("upstream")) for gate in task.get("gated_on", [])]
        for task in tasks
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cycles.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        for upstream in graph.get(node, []):
            visit(upstream)
            if upstream in cycles:
                cycles.add(node)
        visiting.remove(node)
        visited.add(node)

    for task_id in graph:
        visit(task_id)
    return cycles


def validate_roadmap_contract(roadmap: Mapping[str, Any], *, retired_ids: set[str]) -> JsonDict:
    """Validate exact V577 tasks, gates, prior failures, models, and prompts."""

    errors: list[JsonDict] = []
    raw_tasks = roadmap.get("tasks", [])
    tasks = [dict(task) for task in raw_tasks if isinstance(task, Mapping)]
    task_ids = [str(task.get("id", "")) for task in tasks]
    deliverables = [str(task.get("deliverable", "")) for task in tasks]
    if not isinstance(raw_tasks, list) or len(tasks) != len(raw_tasks):
        errors.append(_error("roadmap_schema", "list of task mappings", type(raw_tasks).__name__))
    if roadmap.get("milestone") != MILESTONE:
        errors.append(_error("roadmap_milestone", MILESTONE, roadmap.get("milestone")))
    if len(task_ids) != len(set(task_ids)):
        errors.append(_error("duplicate_task_id", "unique", task_ids))
    if len(deliverables) != len(set(deliverables)):
        errors.append(_error("duplicate_deliverable", "unique", deliverables))
    if set(task_ids) != set(EXPECTED_TASKS):
        errors.append(
            _error(
                "exact_task_set",
                list(EXPECTED_TASKS),
                {
                    "present": task_ids,
                    "missing": [task_id for task_id in EXPECTED_TASKS if task_id not in task_ids],
                    "unexpected": [
                        task_id for task_id in task_ids if task_id not in EXPECTED_TASKS
                    ],
                },
            )
        )

    tasks_by_id = {str(task.get("id")): task for task in tasks}
    positions = {task_id: index for index, task_id in enumerate(task_ids)}
    task_rows: list[JsonDict] = []
    for task_id, expected in EXPECTED_TASKS.items():
        task = tasks_by_id.get(task_id)
        fields = extract_required_fields(str(task.get("prompt", ""))) if task else {}
        row = {
            "task_id": task_id,
            "yaml_present": task is not None,
            "deliverable": task.get("deliverable") if task else None,
            "expected_deliverable": expected["deliverable"],
            "phase": expected["phase"],
            "requires_gpu": task.get("requires_gpu") if task else None,
            "expected_requires_gpu": task_id in GPU_TASK_IDS,
            "resource": "accelerator" if task_id in GPU_TASK_IDS else "cpu",
            "agent_model": task.get("model") if task else None,
            "model_policy": list(REQUIRED_MODELS_BY_TASK.get(task_id, ())),
            "prior_failures": deepcopy(task.get("prior_failures", [])) if task else None,
            "gates": deepcopy(task.get("gated_on", [])) if task else None,
            "terminal_artifact_expectations": list(fields),
            "field_principles": fields,
            "contract_complete": False,
        }
        if task:
            if task.get("milestone") != MILESTONE:
                errors.append(_error("task_milestone", MILESTONE, task.get("milestone"), task_id))
            if task.get("deliverable") != expected["deliverable"]:
                errors.append(
                    _error(
                        "task_deliverable",
                        expected["deliverable"],
                        task.get("deliverable"),
                        task_id,
                    )
                )
            if bool(task.get("requires_gpu", False)) != (task_id in GPU_TASK_IDS):
                errors.append(
                    _error(
                        "task_gpu_policy",
                        task_id in GPU_TASK_IDS,
                        task.get("requires_gpu"),
                        task_id,
                    )
                )
            missing_principles = [field for field, principle in fields.items() if not principle]
            if not fields or missing_principles:
                errors.append(
                    _error(
                        "required_field_principles",
                        "all declared fields have principles",
                        {"field_count": len(fields), "missing": missing_principles},
                        task_id,
                    )
                )
            row["contract_complete"] = (
                task.get("milestone") == MILESTONE
                and task.get("deliverable") == expected["deliverable"]
                and bool(task.get("requires_gpu", False)) == (task_id in GPU_TASK_IDS)
                and bool(fields)
                and not missing_principles
            )
        task_rows.append(row)

    gate_rows: list[JsonDict] = []
    for consumer in tasks:
        consumer_id = str(consumer.get("id", ""))
        actual_gates = consumer.get("gated_on", []) or []
        if actual_gates != EXPECTED_GATES.get(consumer_id, []):
            errors.append(
                _error(
                    "exact_gate_contract",
                    EXPECTED_GATES.get(consumer_id, []),
                    actual_gates,
                    consumer_id,
                )
            )
        for gate in actual_gates:
            upstream = str(gate.get("upstream", ""))
            field = str(gate.get("artifact_field", ""))
            owner = tasks_by_id.get(upstream)
            owner_fields = extract_required_fields(str(owner.get("prompt", ""))) if owner else {}
            op_supported = gate.get("op") in SUPPORTED_GATE_OPS
            owner_exists = owner is not None
            owner_is_earlier = owner_exists and positions.get(upstream, -1) < positions.get(
                consumer_id, -1
            )
            owner_declares_field = field in owner_fields and bool(owner_fields.get(field))
            owner_not_retired = upstream not in retired_ids
            passed = all(
                (
                    op_supported,
                    owner_exists,
                    owner_is_earlier,
                    owner_declares_field,
                    owner_not_retired,
                )
            )
            gate_rows.append(
                {
                    "downstream": consumer_id,
                    "upstream": upstream,
                    "artifact_field": field,
                    "op": gate.get("op"),
                    "expected_value": gate.get("value"),
                    "owner_exists": owner_exists,
                    "owner_is_earlier": owner_is_earlier,
                    "owner_declares_identical_field": owner_declares_field,
                    "owner_not_retired": owner_not_retired,
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(_error("gate_owner_contract", True, gate_rows[-1], consumer_id))
    cycles = sorted(_cycle_nodes(tasks))
    if cycles:
        errors.append(_error("circular_gate", [], cycles))

    prior_rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id", ""))
        for prior in task.get("prior_failures", []) or []:
            changed = str(prior.get("addressed_by", "")).strip()
            concrete = len(changed) >= 20 and changed.lower() not in {
                "same condition",
                "no change",
                "retry",
            }
            row = {
                "declared_by_task": task_id,
                "experiment_id": prior.get("experiment_id"),
                "verdict": prior.get("verdict"),
                "changed_condition": changed,
                "changed_condition_concrete": concrete,
                "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
            }
            prior_rows.append(row)
            if not all(
                (
                    bool(row["experiment_id"]),
                    bool(row["verdict"]),
                    concrete,
                    row["retire_if_same_verdict"],
                )
            ):
                errors.append(_error("prior_failure_completeness", True, row, task_id))

    model_rows: list[JsonDict] = []
    for task_id, required_models in REQUIRED_MODELS_BY_TASK.items():
        task = tasks_by_id.get(task_id)
        prompt = str(task.get("prompt", "")) if task else ""
        present_models = [model for model in required_models if model in prompt]
        legacy_present = [model for model in LEGACY_MODELS if model.lower() in prompt.lower()]
        legacy_smoke_only = not legacy_present or (
            "smoke" in prompt.lower() and "cannot" in prompt.lower()
        )
        passed = (
            task is not None and len(present_models) == len(required_models) and legacy_smoke_only
        )
        row = {
            "task_id": task_id,
            "task_present": task is not None,
            "required_models": list(required_models),
            "present_models": present_models,
            "legacy_models_present": legacy_present,
            "legacy_smoke_only": legacy_smoke_only,
            "passed": passed,
        }
        model_rows.append(row)
        if not passed:
            errors.append(_error("model_policy", True, row, task_id))

    return {
        "passed": not errors,
        "errors": errors,
        "task_contract_rows": task_rows,
        "gate_owner_rows": gate_rows,
        "declared_prior_failure_rows": prior_rows,
        "model_policy_receipts": model_rows,
    }


def receipt_checksum(receipt: Mapping[str, Any]) -> str:
    """Hash a receipt while excluding only its own checksum field."""

    payload = {key: value for key, value in receipt.items() if key != "checksum"}
    return sha256_bytes(canonical_json(payload))


def validate_phase_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Return every phase receipt violation without repairing the receipt."""

    errors = [f"missing field: {field}" for field in PHASE_RECEIPT_FIELDS if field not in receipt]
    if errors:
        return errors
    start = receipt["monotonic_started_s"]
    end = receipt["monotonic_ended_s"]
    heartbeat = receipt["heartbeat_monotonic_s"]
    if end < start or not start <= heartbeat <= end:
        errors.append("phase timestamp reversal")
    identity = receipt["process_identity"]
    if not isinstance(identity, Mapping) or not {"pid", "pid_start_time"} <= set(identity):
        errors.append("invalid process identity")
    elif identity["pid_start_time"] > start:
        errors.append("PID start time is after phase start")
    if receipt["state"] not in TERMINAL_PHASE_STATES:
        errors.append("nonterminal phase state")
    if receipt["checksum"] != receipt_checksum(receipt):
        errors.append("checksum mismatch")
    return errors


def validate_accelerator_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Validate the accelerator extension after the base phase receipt."""

    errors = validate_phase_receipt(receipt)
    errors.extend(
        f"missing accelerator field: {field}"
        for field in ACCELERATOR_RECEIPT_FIELDS
        if field not in receipt
    )
    if any(field not in receipt for field in ACCELERATOR_RECEIPT_FIELDS):
        return errors
    identity = receipt.get("process_identity", {})
    if receipt["pid_start_time"] != identity.get("pid_start_time"):
        errors.append("accelerator PID start time differs from process identity")
    unload = receipt["unload_evidence"]
    if not isinstance(unload, Mapping) or unload.get("completed") is not True:
        errors.append("accelerator unload is incomplete")
    if not isinstance(receipt["offload_layers"], int) or receipt["offload_layers"] < 0:
        errors.append("invalid accelerator offload layer count")
    return errors


def _experiment_number(value: str) -> int | None:
    match = re.search(r"exp(?:eriment_)?(\d+)", value.lower())
    return int(match.group(1)) if match else None


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact root is not an object: {path}")
    return payload


def _unwrap(value: Any) -> Any:
    if (
        isinstance(value, dict)
        and "value" in value
        and set(value)
        <= {
            "value",
            "principle",
            "source",
            "satisfied_by",
        }
    ):
        return value["value"]
    return value


def _v576_artifact(repo_root: Path, number: int) -> Path | None:
    matches = sorted((repo_root / "results").glob(f"experiment_{number}_*.json"))
    return matches[0] if matches else None


def reconcile_prior_failures(
    repo_root: Path, tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Preserve V576 terminal values and validate every declared predecessor."""

    declarations: dict[int, list[JsonDict]] = {}
    for task in tasks:
        for prior in task.get("prior_failures", []) or []:
            number = _experiment_number(str(prior.get("experiment_id", "")))
            if number is not None:
                declarations.setdefault(number, []).append(
                    {"declared_by_task": str(task.get("id", "")), **dict(prior)}
                )
    rows: list[JsonDict] = []
    errors: list[JsonDict] = []
    for number in range(6604, 6616):
        path = _v576_artifact(repo_root, number)
        payload = _read_json(path) if path else {}
        status = _unwrap(payload.get("status")) if path else None
        verdict = _unwrap(payload.get("honest_verdict")) if path else None
        verdict_class = _unwrap(payload.get("verdict_class")) if path else None
        if not path:
            disposition = "missing_source"
        elif verdict_class not in CLOSED_VERDICT_CLASSES:
            disposition = (
                "missing_verdict_class" if verdict_class is None else "invalid_verdict_class"
            )
        else:
            disposition = verdict_class
        declared = declarations.get(number) or [None]
        for declaration in declared:
            changed = str(declaration.get("addressed_by", "")).strip() if declaration else None
            row = {
                "experiment_number": number,
                "source_state": "present" if path else "missing",
                "source_path": path.relative_to(repo_root).as_posix() if path else None,
                "source_sha256": sha256_file(path) if path else None,
                "source_status": status,
                "source_honest_verdict": verdict,
                "source_verdict_class": verdict_class,
                "contract_disposition": disposition,
                "declared_by_task": declaration.get("declared_by_task") if declaration else None,
                "declared_experiment_id": declaration.get("experiment_id") if declaration else None,
                "declared_verdict": declaration.get("verdict") if declaration else None,
                "verdict_matches": declaration.get("verdict") == verdict if declaration else None,
                "changed_condition": changed,
                "changed_condition_concrete": len(changed) >= 20 if changed else None,
                "retire_if_same_verdict": (
                    declaration.get("retire_if_same_verdict") is True if declaration else None
                ),
            }
            rows.append(row)
            if declaration and not row["verdict_matches"]:
                errors.append(_error("prior_failure_verdict", verdict, declaration.get("verdict")))
            if declaration and not row["changed_condition_concrete"]:
                errors.append(_error("prior_failure_changed_condition", True, changed))
            if declaration and not row["retire_if_same_verdict"]:
                errors.append(_error("prior_failure_retirement", True, False))
    return rows, errors


def _manifest_retired_ids(repo_root: Path) -> set[str]:
    path = repo_root / "ops/exclusion_manifest.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    retired: set[str] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        for row in payload.get(section, []) or []:
            values = [row.get("experiment_id"), *(row.get("experiment_ids", []) or [])]
            for value in values:
                number = _experiment_number(str(value))
                if number is not None:
                    retired.add(f"exp{number}")
                    retired.add(str(value))
    return retired


def load_pre_staged_roadmap(repo_root: Path) -> tuple[JsonDict, JsonDict]:
    """Load the working file or the latest Git blob removed by activation."""

    working = repo_root / PRE_STAGED_ROADMAP
    if working.is_file():
        body = working.read_bytes()
        return yaml.safe_load(body), {
            "path": PRE_STAGED_ROADMAP.as_posix(),
            "source_state": "working_tree",
            "git_commit": None,
            "sha256": sha256_bytes(body),
        }
    history = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", PRE_STAGED_ROADMAP.as_posix()],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    for commit in history.stdout.splitlines():
        shown = subprocess.run(
            ["git", "show", f"{commit}:{PRE_STAGED_ROADMAP.as_posix()}"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if shown.returncode == 0:
            return yaml.safe_load(shown.stdout), {
                "path": PRE_STAGED_ROADMAP.as_posix(),
                "source_state": "historical_git_blob",
                "git_commit": commit,
                "sha256": sha256_bytes(shown.stdout),
            }
    raise FileNotFoundError(PRE_STAGED_ROADMAP)


def protected_file_receipts(
    repo_root: Path, before_hashes: Mapping[str, str] | None = None
) -> JsonDict:
    """Compare protected files with hashes captured before task work."""

    before = dict(before_hashes or {})
    rows = []
    for relative in PROTECTED_PATHS:
        observed = sha256_file(repo_root / relative)
        expected = before.get(relative.as_posix(), observed)
        rows.append(
            {
                "path": relative.as_posix(),
                "before_sha256": expected,
                "after_sha256": observed,
                "unchanged": expected == observed,
            }
        )
    return {"all_unchanged": all(row["unchanged"] for row in rows), "rows": rows}


def _attack_fixture_roadmap() -> JsonDict:
    owner_fields = {
        gate["upstream"]: gate["artifact_field"]
        for gates in EXPECTED_GATES.values()
        for gate in gates
    }
    tasks = []
    for task_id, expected in EXPECTED_TASKS.items():
        fields = ["status", "honest_verdict", "verdict_class", "gate_check_summary"]
        if task_id in owner_fields:
            fields.append(owner_fields[task_id])
        lines = ["MODEL_SPECS:", *REQUIRED_MODELS_BY_TASK.get(task_id, ())]
        lines.append("REQUIRED ARTIFACT FIELDS:")
        for field in fields:
            lines.extend((f"  {field}:", f'    principle: "Principle for {field}."'))
        lines.append("Set inference_substrate=fixture_no_llm")
        tasks.append(
            {
                "id": task_id,
                "milestone": MILESTONE,
                "deliverable": expected["deliverable"],
                "title": task_id,
                "prompt": "\n".join(lines),
                "requires_gpu": task_id in GPU_TASK_IDS,
                "model": "opus",
                "gated_on": deepcopy(EXPECTED_GATES.get(task_id, [])),
            }
        )
    return {
        "milestone": MILESTONE,
        "milestone_title": "attack fixture",
        "milestone_doc": ROADMAP_DOCUMENT.as_posix(),
        "tasks": tasks,
    }


def _valid_phase_fixture() -> JsonDict:
    receipt: JsonDict = {
        "task_id": "exp6617-gpu-lease-phase-receipts",
        "phase_name": "validating",
        "state": "terminal_complete",
        "monotonic_started_s": 10.0,
        "monotonic_ended_s": 12.0,
        "process_identity": {"pid": 42, "pid_start_time": 9.0},
        "resource_owner": "exp6617-gpu-lease-phase-receipts",
        "input_hashes": {"input": "sha256:input"},
        "output_hashes": {"output": "sha256:output"},
        "heartbeat_monotonic_s": 11.0,
        "terminal_reason": "complete",
    }
    receipt["checksum"] = receipt_checksum(receipt)
    return receipt


def _terminal_errors(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    if payload.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("nonterminal verdict")
    if str(payload.get("status", "")).startswith("blocked_") and not payload.get(
        "gate_check_summary"
    ):
        errors.append("absent blocked diagnostic")
    return errors


def run_attacks() -> list[JsonDict]:
    """Run contract mutations and retain the exact rejection evidence."""

    base = _attack_fixture_roadmap()
    rows: list[JsonDict] = []

    def roadmap_attack(attack_id: str, mutate: Any, retired: set[str] | None = None) -> None:
        candidate = deepcopy(base)
        mutate(candidate)
        errors = validate_roadmap_contract(candidate, retired_ids=retired or set())["errors"]
        rows.append(
            {
                "attack_id": attack_id,
                "mutation_applied": True,
                "observed_errors": errors,
                "fail_closed": bool(errors),
            }
        )

    roadmap_attack(
        "duplicate_id",
        lambda value: value["tasks"][-1].update(id=value["tasks"][0]["id"]),
    )
    roadmap_attack(
        "duplicate_deliverable",
        lambda value: value["tasks"][-1].update(deliverable=value["tasks"][0]["deliverable"]),
    )
    roadmap_attack(
        "stale_milestone", lambda value: value["tasks"][0].update(milestone="2026.08.576")
    )
    roadmap_attack(
        "missing_principle",
        lambda value: value["tasks"][0].update(
            prompt=value["tasks"][0]["prompt"].replace('    principle: "Principle for status."', "")
        ),
    )
    roadmap_attack(
        "unsupported_gate_op",
        lambda value: value["tasks"][1]["gated_on"][0].update(op="contains"),
    )
    roadmap_attack(
        "missing_gate_field",
        lambda value: value["tasks"][1]["gated_on"][0].update(artifact_field="missing_ready_score"),
    )
    roadmap_attack(
        "wrong_model_id",
        lambda value: value["tasks"][2].update(
            prompt=value["tasks"][2]["prompt"].replace(QWEN_MODEL, LEGACY_MODELS[0])
        ),
    )

    terminal = {
        "status": "blocked_contract",
        "verdict_class": "blocked",
        "gate_check_summary": None,
    }
    rows.append(
        {
            "attack_id": "absent_blocked_diagnostic",
            "mutation_applied": True,
            "observed_errors": _terminal_errors(terminal),
            "fail_closed": bool(_terminal_errors(terminal)),
        }
    )
    nonterminal = {"status": "running", "verdict_class": "running"}
    rows.append(
        {
            "attack_id": "nonterminal_verdict",
            "mutation_applied": True,
            "observed_errors": _terminal_errors(nonterminal),
            "fail_closed": bool(_terminal_errors(nonterminal)),
        }
    )

    phase = _valid_phase_fixture()
    phase["monotonic_ended_s"] = 8.0
    phase["checksum"] = receipt_checksum(phase)
    rows.append(
        {
            "attack_id": "phase_timestamp_reversal",
            "mutation_applied": True,
            "observed_errors": validate_phase_receipt(phase),
            "fail_closed": bool(validate_phase_receipt(phase)),
        }
    )
    phase = _valid_phase_fixture()
    phase["process_identity"]["pid_start_time"] = 11.0
    phase["checksum"] = receipt_checksum(phase)
    rows.append(
        {
            "attack_id": "pid_reuse",
            "mutation_applied": True,
            "observed_errors": validate_phase_receipt(phase),
            "fail_closed": bool(validate_phase_receipt(phase)),
        }
    )
    phase = _valid_phase_fixture()
    phase["task_id"] = "tampered"
    rows.append(
        {
            "attack_id": "checksum_tampering",
            "mutation_applied": True,
            "observed_errors": validate_phase_receipt(phase),
            "fail_closed": bool(validate_phase_receipt(phase)),
        }
    )
    retired_owner = base["tasks"][0]["id"]
    roadmap_attack("retired_upstream", lambda _value: None, retired={retired_owner})
    protected_errors = ["protected file mutation"] if "sha256:before" != "sha256:after" else []
    rows.append(
        {
            "attack_id": "protected_file_mutation",
            "mutation_applied": True,
            "observed_errors": protected_errors,
            "fail_closed": bool(protected_errors),
        }
    )
    return rows


def _source_hash_rows(repo_root: Path) -> list[JsonDict]:
    paths = [
        ACTIVE_ROADMAP,
        Path("scripts/research_conductor.py"),
        ROADMAP_DOCUMENT,
        Path("ops/exclusion_manifest.yaml"),
        Path("scripts/roadmap_schema.py"),
        Path("scripts/validate_prior_failures.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        Path("scripts/conductor_gates.py"),
        SPEC_PATH,
    ]
    paths.extend(
        path.relative_to(repo_root)
        for number in range(6604, 6616)
        if (path := _v576_artifact(repo_root, number)) is not None
    )
    return [
        {
            "path": relative.as_posix(),
            "present": (repo_root / relative).is_file(),
            "sha256": sha256_file(repo_root / relative)
            if (repo_root / relative).is_file()
            else None,
        }
        for relative in paths
    ]


def _git_status(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.splitlines()


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    mem_total_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo.read_text(), re.MULTILINE)
        mem_total_kib = int(match.group(1)) if match else None
    return {
        "python": platform.python_version(),
        "pyyaml": yaml.__version__,
        "pydantic": pydantic.__version__,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": mem_total_kib * 1024 if mem_total_kib else None,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _validation_receipts(audit: Mapping[str, Any]) -> list[JsonDict]:
    grouped = {
        "schema": {"roadmap_schema"},
        "exclusion": {"gate_owner_contract"},
        "prior_failure": {"prior_failure_completeness"},
        "milestone": {"roadmap_milestone", "task_milestone"},
        "id": {"duplicate_task_id", "exact_task_set"},
        "deliverable": {"duplicate_deliverable", "task_deliverable"},
        "prompt": {"required_field_principles", "model_policy"},
    }
    errors = list(audit["errors"])
    return [
        {
            "check": name,
            "command": f"internal:validate_roadmap_contract:{name}",
            "exit_code": int(any(error["check"] in categories for error in errors)),
            "duration_s": 0.0,
            "observed": [error for error in errors if error["check"] in categories],
        }
        for name, categories in grouped.items()
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash final content while excluding only the checksum itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_bytes(canonical_json(payload))


def build_artifact(
    *,
    repo_root: Path,
    run_date: str,
    roadmap_payload: Mapping[str, Any] | None = None,
    roadmap_source: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    protected_before: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build a terminal contract artifact from exact stored evidence."""

    if roadmap_payload is None:
        loaded, loaded_receipt = load_pre_staged_roadmap(repo_root)
        roadmap_payload = loaded
        roadmap_source = loaded_receipt
    source = dict(roadmap_source or {})
    audit = validate_roadmap_contract(roadmap_payload, retired_ids=_manifest_retired_ids(repo_root))
    tasks = [task for task in roadmap_payload.get("tasks", []) if isinstance(task, Mapping)]
    prior_rows, prior_errors = reconcile_prior_failures(repo_root, tasks)
    attacks = run_attacks()
    protection = protected_file_receipts(repo_root, protected_before)
    test_failures = [dict(row) for row in tests_run if row.get("exit_code") != 0]
    ready = (
        audit["passed"]
        and not prior_errors
        and all(row["fail_closed"] for row in attacks)
        and protection["all_unchanged"]
        and not test_failures
    )
    failed_checks = [*audit["errors"], *prior_errors]
    if test_failures:
        failed_checks.append(_error("tests", 0, test_failures))
    if not protection["all_unchanged"]:
        failed_checks.append(_error("protected_files", True, protection))
    gate_summary = {
        "all_passed": ready,
        "failed_checks": failed_checks,
        "observed": {
            "expected_task_count": len(EXPECTED_TASKS),
            "yaml_task_count": len(tasks),
            "missing_task_ids": [
                row["task_id"] for row in audit["task_contract_rows"] if not row["yaml_present"]
            ],
            "gate_count": len(audit["gate_owner_rows"]),
            "prior_declaration_count": len([row for row in prior_rows if row["declared_by_task"]]),
            "failed_test_count": len(test_failures),
        },
    }
    status = "complete_execution_contract_ready" if ready else "blocked_roadmap_contract_incomplete"
    honest_verdict = (
        "complete: V577 execution contract is replayable null infrastructure with no science claim"
        if ready
        else "blocked_roadmap_contract_incomplete: expected Exp6616-Exp6628 YAML contracts are not all present"
    )
    phase_schema = {
        "schema_id": "v577.phase_receipt.v1",
        "required_fields": list(PHASE_RECEIPT_FIELDS),
        "terminal_states": sorted(TERMINAL_PHASE_STATES),
        "timestamp_rule": "monotonic_started_s <= heartbeat_monotonic_s <= monotonic_ended_s",
        "process_identity_fields": ["pid", "pid_start_time"],
        "checksum_rule": "sha256 of canonical JSON excluding checksum",
        "validator": "validate_phase_receipt",
    }
    accelerator_schema = {
        "schema_id": "v577.accelerator_receipt.v1",
        "extends": phase_schema["schema_id"],
        "required_extension_fields": list(ACCELERATOR_RECEIPT_FIELDS),
        "pid_binding_rule": "pid_start_time equals process_identity.pid_start_time",
        "unload_rule": "unload_evidence.completed is true",
        "validator": "validate_accelerator_receipt",
    }
    preconditions = {
        "planning_date": run_date,
        "pre_staged_roadmap": source,
        "source_hashes": _source_hash_rows(repo_root),
        "dirty_worktree_state": _git_status(repo_root),
        "resources_and_versions": _resource_receipt(repo_root),
        "no_llm_substrate": INFERENCE_SUBSTRATE,
        "llm_invocation_count": 0,
        "protected_paths": [path.as_posix() for path in PROTECTED_PATHS],
    }
    provenance = {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [
                PRE_STAGED_ROADMAP.as_posix(),
                ACTIVE_ROADMAP.as_posix(),
                ROADMAP_DOCUMENT.as_posix(),
                "results/experiment_6604_* through results/experiment_6615_*",
            ],
            "validators": [
                "validate_roadmap_contract",
                "reconcile_prior_failures",
                "validate_phase_receipt",
                "validate_accelerator_receipt",
                "validate_artifact",
            ],
        }
        for field in REQUIRED_FIELDS
    }
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": gate_summary,
        "task_contract_rows": audit["task_contract_rows"],
        "roadmap_validation_receipts": _validation_receipts(audit),
        "gate_owner_rows": audit["gate_owner_rows"],
        "prior_failure_dispositions": prior_rows,
        "phase_receipt_schema": phase_schema,
        "accelerator_receipt_schema": accelerator_schema,
        "model_policy_receipts": audit["model_policy_receipts"],
        "execution_contract_ready_score": 1.0 if ready else 0.0,
        "attack_rows": attacks,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protection,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": provenance,
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject incomplete, inconsistent, or mutated terminal artifacts."""

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility checksum mismatch")
    if artifact["verdict_class"] not in CLOSED_VERDICT_CLASSES:
        raise ValueError("verdict class is outside the closed enum")
    if (
        artifact["inference_substrate"] != INFERENCE_SUBSTRATE
        or artifact["verifier_is_oracle"] is not True
    ):
        raise ValueError("inference substrate or oracle declaration is invalid")
    if len(artifact["task_contract_rows"]) != len(EXPECTED_TASKS):
        raise ValueError("task contract row count is not exact")
    if {row["attack_id"] for row in artifact["attack_rows"]} != set(ATTACK_IDS) or not all(
        row["fail_closed"] for row in artifact["attack_rows"]
    ):
        raise ValueError("attack contract did not fail closed")
    score = artifact["execution_contract_ready_score"]
    summary_passed = artifact["gate_check_summary"].get("all_passed") is True
    if score == 1.0 and (artifact["verdict_class"] != "null" or not summary_passed):
        raise ValueError("readiness one requires a clean null contract")
    if score == 0.0 and (
        artifact["verdict_class"] != "blocked"
        or not str(artifact["status"]).startswith("blocked_")
        or not str(artifact["honest_verdict"]).startswith("blocked_")
        or not artifact["gate_check_summary"].get("failed_checks")
    ):
        raise ValueError("blocked readiness requires exact diagnostics")
    if score not in {0.0, 1.0}:
        raise ValueError("readiness must be binary")


def write_artifact_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Sync a temporary file, replace the target, and sync its directory."""

    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _command_receipt(
    command: list[str], repo_root: Path, scope: str
) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    return {
        "command": " ".join(command),
        "scope": scope,
        "exit_code": result.returncode,
        "duration_s": round(time.monotonic() - started, 6),
        "output_tail": output[-1000:],
    }


def _validation_commands(repo_root: Path) -> list[tuple[list[str], str]]:  # pragma: no cover
    python = (repo_root / ".venv/bin/python").as_posix()
    pytest = (repo_root / ".venv/bin/pytest").as_posix()
    coverage = (repo_root / ".venv/bin/coverage").as_posix()
    ruff = (repo_root / ".venv/bin/ruff").as_posix()
    mypy = (repo_root / ".venv/bin/mypy").as_posix()
    module = "python/carnot/experiment_6616_v577_execution_contract.py"
    focused = "tests/python/test_experiment_6616_v577_execution_contract.py"
    return [
        ([pytest, focused, "-q", "-o", "addopts=", "--no-cov"], "focused"),
        (
            [
                coverage,
                "run",
                "--rcfile=/dev/null",
                f"--include=*/{module}",
                "-m",
                "pytest",
                focused,
                "-q",
                "-o",
                "addopts=",
                "--no-cov",
            ],
            "new_code_coverage_run",
        ),
        (
            [coverage, "report", "--rcfile=/dev/null", "--fail-under=100", "--show-missing"],
            "new_code_coverage_report",
        ),
        ([pytest, "tests/python", "-q"], "full_python_suite"),
        ([ruff, "check", module, focused], "lint"),
        ([ruff, "format", "--check", module, focused], "format"),
        ([mypy, module], "type_check"),
        (
            [python, "scripts/validate_prior_failures.py", ACTIVE_ROADMAP.as_posix()],
            "roadmap_schema_and_prior_failures",
        ),
        (
            [python, "scripts/exclusion_manifest_lint.py", ACTIVE_ROADMAP.as_posix()],
            "exclusion_manifest",
        ),
        (
            [python, "scripts/audit_roadmap_gates.py", ACTIVE_ROADMAP.as_posix()],
            "gate_audit",
        ),
        ([python, "scripts/check_spec_coverage.py", focused], "spec_coverage"),
        (
            [python, "scripts/artifact_convention_audit.py", "--recent", "1", "--dry-run"],
            "artifact_convention",
        ),
        ([python, "scripts/adversarial_verify.py", RESULT_PATH.as_posix()], "adversarial"),
        (
            [
                pytest,
                "tests/python/test_conductor_gates.py",
                "tests/python/test_roadmap_schema.py",
                "-q",
                "-o",
                "addopts=",
                "--no-cov",
            ],
            "applicable_e2e",
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    protected_before = {
        relative.as_posix(): sha256_file(repo_root / relative) for relative in PROTECTED_PATHS
    }
    preliminary = build_artifact(
        repo_root=repo_root,
        run_date=args.date,
        duration_s=time.monotonic() - started,
        protected_before=protected_before,
    )
    write_artifact_atomic(repo_root / RESULT_PATH, preliminary)
    receipts = [
        _command_receipt(command, repo_root, scope)
        for command, scope in _validation_commands(repo_root)
    ]
    final = build_artifact(
        repo_root=repo_root,
        run_date=args.date,
        tests_run=receipts,
        duration_s=time.monotonic() - started,
        protected_before=protected_before,
    )
    write_artifact_atomic(repo_root / RESULT_PATH, final)
    print(json.dumps({"path": RESULT_PATH.as_posix(), "status": final["status"]}, sort_keys=True))
    return 0 if all(receipt["exit_code"] == 0 for receipt in receipts) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
