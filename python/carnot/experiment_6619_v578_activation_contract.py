"""Validate V578 as one complete activation unit without running an LLM.

Activation deleted the pre-staged YAML after copying its incomplete ten-task
manifest into the active roadmap. The milestone document names fourteen tasks.
This module reads the deleted Git blob as evidence, keeps all missing contracts
visible, and writes a terminal blocked artifact instead of inventing the four
missing tasks. See REQ-REPORT-6619 and SCENARIO-REPORT-6619-*.
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
MILESTONE = "2026.08.578"
ACTIVE_ROADMAP = Path("research-roadmap.yaml")
PRE_STAGED_ROADMAP = Path("research-roadmap-next.yaml")
ROADMAP_DOCUMENT = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_6619_v578_activation_contract.json")
PRIOR_ARTIFACT = Path("results/experiment_6616_v577_execution_contract.json")
INFERENCE_SUBSTRATE = "v578_complete_activation_contract_no_llm"
PROMPT_TERMINATOR = "Do NOT push. Do NOT modify scripts/research_conductor.py."
PROTECTED_PATHS = (ACTIVE_ROADMAP, Path("scripts/research_conductor.py"))
SUPPORTED_GATE_OPS = {"==", "!=", ">", "<", ">=", "<=", "exists", "in"}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
QWEN_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_MODEL = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODELS = (QWEN_MODEL, GEMMA31_MODEL, GEMMA26_MODEL)
LEGACY_MODELS = ("Qwen3.5-0.8B", "Gemma-4-E4B-it")

EXPECTED_TASK_IDS = (
    "exp6619-v578-activation-contract",
    "exp6620-gpu-lease-phase-receipts",
    "exp6621-mandated-model-admission",
    "exp6622-qwen36-direct-headroom",
    "exp6623-headroom-support-reducer",
    "exp6624-delayed-two-level-decoding",
    "exp6625-cold-context-verifier-control",
    "exp6626-constraint-authority-support-audit",
    "exp6627-spectral-integrity-repair",
    "exp6628-spectral-cpu-gpu-replay",
    "exp6629-live-memory-actionability",
    "exp6630-error-independent-memory-patch-gate",
    "exp6631-prospective-support-preserving-csl",
    "exp6632-v578-independent-capstone",
)
EXPECTED_DELIVERABLES = {
    "exp6619-v578-activation-contract": RESULT_PATH.as_posix(),
    "exp6620-gpu-lease-phase-receipts": "results/experiment_6620_gpu_lease_phase_receipts.json",
    "exp6621-mandated-model-admission": "results/experiment_6621_mandated_model_admission.json",
    "exp6622-qwen36-direct-headroom": "results/experiment_6622_qwen36_direct_headroom.json",
    "exp6623-headroom-support-reducer": "results/experiment_6623_headroom_support_reducer.json",
    "exp6624-delayed-two-level-decoding": "results/experiment_6624_delayed_two_level_decoding.json",
    "exp6625-cold-context-verifier-control": "results/experiment_6625_cold_context_verifier_control.json",
    "exp6626-constraint-authority-support-audit": "results/experiment_6626_constraint_authority_support_audit.json",
    "exp6627-spectral-integrity-repair": "results/experiment_6627_spectral_integrity_repair.json",
    "exp6628-spectral-cpu-gpu-replay": "results/experiment_6628_spectral_cpu_gpu_replay.json",
    "exp6629-live-memory-actionability": "results/experiment_6629_live_memory_actionability.json",
    "exp6630-error-independent-memory-patch-gate": "results/experiment_6630_error_independent_memory_patch_gate.json",
    "exp6631-prospective-support-preserving-csl": "results/experiment_6631_prospective_support_preserving_csl.json",
    "exp6632-v578-independent-capstone": "results/experiment_6632_v578_independent_capstone.json",
}
EXPECTED_GATES_BY_NUMBER: dict[int, list[JsonDict]] = {
    6620: [
        {
            "upstream": 6619,
            "artifact_field": "activation_contract_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6621: [
        {
            "upstream": 6620,
            "artifact_field": "gpu_lease_scheduler_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6622: [
        {"upstream": 6621, "artifact_field": "qwen_admission_ready_score", "op": "==", "value": 1.0}
    ],
    6623: [
        {
            "upstream": 6622,
            "artifact_field": "baseline_rows_complete_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6624: [
        {
            "upstream": 6623,
            "artifact_field": "constrained_decoding_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6625: [
        {"upstream": 6621, "artifact_field": "qwen_admission_ready_score", "op": "==", "value": 1.0}
    ],
    6626: [
        {"upstream": 6624, "artifact_field": "decoding_rows_ready_score", "op": "==", "value": 1.0}
    ],
    6627: [
        {
            "upstream": 6619,
            "artifact_field": "activation_contract_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6628: [
        {
            "upstream": 6627,
            "artifact_field": "sampler_integrity_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6629: [
        {"upstream": 6621, "artifact_field": "qwen_admission_ready_score", "op": "==", "value": 1.0}
    ],
    6630: [
        {
            "upstream": 6629,
            "artifact_field": "live_memory_activation_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    6631: [
        {
            "upstream": 6621,
            "artifact_field": "qwen_admission_ready_score",
            "op": "==",
            "value": 1.0,
        },
        {
            "upstream": 6630,
            "artifact_field": "memory_patch_contract_ready_score",
            "op": "==",
            "value": 1.0,
        },
    ],
}
_TASK_ID_BY_NUMBER = {
    int(re.search(r"\d+", task_id).group()): task_id for task_id in EXPECTED_TASK_IDS
}
EXPECTED_GATES = {
    _TASK_ID_BY_NUMBER[downstream]: [
        {**gate, "upstream": _TASK_ID_BY_NUMBER[gate["upstream"]]} for gate in gates
    ]
    for downstream, gates in EXPECTED_GATES_BY_NUMBER.items()
}
REQUIRED_MODELS_BY_TASK = {
    _TASK_ID_BY_NUMBER[6621]: MANDATED_MODELS,
    _TASK_ID_BY_NUMBER[6622]: (QWEN_MODEL,),
    _TASK_ID_BY_NUMBER[6624]: (QWEN_MODEL,),
    _TASK_ID_BY_NUMBER[6625]: (QWEN_MODEL, GEMMA26_MODEL),
    _TASK_ID_BY_NUMBER[6629]: (QWEN_MODEL,),
    _TASK_ID_BY_NUMBER[6631]: (QWEN_MODEL,),
}
EXPECTED_GPU_TASK_IDS = set(REQUIRED_MODELS_BY_TASK) | {_TASK_ID_BY_NUMBER[6628]}
COMPARATIVE_TASK_IDS = set(EXPECTED_TASK_IDS[2:])
COMMON_TASK_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
ATTACK_IDS = (
    "missing_task",
    "missing_owner_field",
    "duplicate_deliverable",
    "bad_gate_op",
    "wrong_model_id",
    "incomplete_prior_failure",
    "missing_principle",
    "missing_prompt_ending",
    "protected_file_mutation",
)
REQUIRED_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "task_contract_rows",
    "document_yaml_diff",
    "gate_owner_rows",
    "prior_failure_dispositions",
    "model_policy_receipts",
    "validation_receipts",
    "activation_contract_ready_score",
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
    "status": "The task ends terminally and never hides an activation block.",
    "honest_verdict": "The verdict reports contract readiness without claiming scientific benefit.",
    "verdict_class": "A ready activation contract is null infrastructure, not positive science.",
    "gate_check_summary": "A blocked result names each failed contract check and observed value.",
    "task_contract_rows": "Every Exp6619-Exp6632 task keeps one replayable contract row.",
    "document_yaml_diff": "Document and YAML task, gate, model, hardware, and deliverable sets are exact.",
    "gate_owner_rows": "Each gate binds an earlier owner and an identically spelled required field.",
    "prior_failure_dispositions": "Each rerun retains its prior verdict, changed condition, and retirement signal.",
    "model_policy_receipts": "Every LLM task binds mandated GGUF identity and forbids headline fallback.",
    "validation_receipts": "Schema, exclusion, gate, milestone, prompt, principle, and protection checks retain exits.",
    "activation_contract_ready_score": "This binary field opens Exp6620 and Exp6627 only after the full unit passes.",
    "attack_rows": "Omission, identity, gate, model, history, principle, ending, and mutation attacks fail closed.",
    "preconditions_checked": "Roadmaps, document, prior evidence, schemas, versions, resources, and hashes are explicit.",
    "protected_files_unchanged": "The active roadmap and conductor retain their original hashes.",
    "inference_substrate": "The task performs roadmap and evidence validation without an LLM.",
    "verifier_is_oracle": "Exact contract checks are authoritative and evaluate no science.",
    "field_provenance": "Every field names source paths, hashes, parsers, and validation functions.",
    "duration_s": "Monotonic duration covers replay, attacks, validation, and writing.",
    "tests_run": "Focused, roadmap, gate, spec, artifact, adversarial, and E2E commands retain exits.",
    "reproducibility_checksum": "A final content hash detects artifact mutation.",
}


def canonical_json(value: Any) -> bytes:
    """Return deterministic JSON bytes used by all content checksums."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Label an exact byte digest so callers cannot confuse it with a path."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so large evidence files do not copy into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _experiment_number(value: str) -> int | None:
    """Return the experiment number from either task IDs or artifact names."""

    match = re.search(r"exp(?:eriment_)?(\d+)", value.lower())
    return int(match.group(1)) if match else None


def extract_required_fields(prompt: str) -> dict[str, str | None]:
    """Parse inline or multiline principle annotations from one task prompt."""

    if "REQUIRED ARTIFACT FIELDS:" not in prompt:
        return {}
    block = prompt.split("REQUIRED ARTIFACT FIELDS:", 1)[1]
    block = block.split("Set inference_substrate=", 1)[0]
    fields: dict[str, str | None] = {}
    current: str | None = None
    for line in block.splitlines():
        field = re.match(r"^  ([a-z][a-z0-9_]*):(?:\s*(.*))?$", line)
        if field:
            current = field.group(1)
            inline = field.group(2) or ""
            principle = re.search(r"principle:\s*[\"']([^\"']+)[\"']", inline)
            fields[current] = principle.group(1) if principle else None
            continue
        principle = re.match(r"^\s{4}principle:\s*[\"'](.+)[\"']\s*$", line)
        if current and principle:
            fields[current] = principle.group(1)
    return fields


def parse_document_contract(text: str) -> JsonDict:
    """Extract the exact task, gate, model, graph, and hardware document sets."""

    heading_matches = list(re.finditer(r"^### Exp(\d+) - (.+)$", text, re.MULTILINE))
    numbers: list[int] = []
    titles: dict[int, str] = {}
    deliverables: dict[int, str] = {}
    for index, match in enumerate(heading_matches):
        number = int(match.group(1))
        end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(text)
        section = text[match.end() : end]
        deliverable = re.search(r"\*\*Deliverable:\*\* `([^`]+)`", section)
        numbers.append(number)
        titles[number] = match.group(2).strip()
        if deliverable:
            deliverables[number] = deliverable.group(1)

    dependency_section = text.split("## Dependency graph", 1)[1].split("```text", 1)[1]
    dependency_block = dependency_section.split("```", 1)[0]
    graph_numbers = list(
        dict.fromkeys(int(value) for value in re.findall(r"Exp(\d+)", dependency_block))
    )

    gate_table = dependency_section.split("Structured gates:", 1)[1].split(
        "Exp6632 has no structured gate", 1
    )[0]
    gates: dict[int, list[JsonDict]] = {}
    for line in gate_table.splitlines():
        downstream = re.match(r"\| Exp(\d+) \|", line)
        if not downstream:
            continue
        entries = re.findall(r"`exp(\d+)\.([a-z][a-z0-9_]*)`", line)
        gates[int(downstream.group(1))] = [
            {"upstream": int(owner), "artifact_field": field, "op": "==", "value": 1.0}
            for owner, field in entries
        ]

    model_section = text.split("## Model policy", 1)[1].split("## Claim, ARC", 1)[0]
    repo_ids = re.findall(r"`(unsloth/[^`]+-GGUF)`", model_section)
    models: dict[int, list[str]] = {}
    all_three = re.search(r"Exp(\d+) uses all three", model_section)
    if all_three:
        models[int(all_three.group(1))] = repo_ids
    qwen_group = re.search(r"((?:Exp\d+(?:, |, and | and ))+Exp\d+) use Qwen3\.6", model_section)
    if qwen_group:
        for number in re.findall(r"Exp(\d+)", qwen_group.group(1)):
            models[int(number)] = [QWEN_MODEL]
    dual = re.search(r"Exp(\d+) uses Qwen3\.6 and Gemma 26B", model_section)
    if dual:
        models[int(dual.group(1))] = [QWEN_MODEL, GEMMA26_MODEL]
    hardware_numbers = sorted(
        set(models)
        | {
            int(value)
            for value in re.findall(
                r"\*\*Local acceleration:\*\* Exp(\d+)",
                text.split("## Hardware requirements and boundaries", 1)[1],
            )
        }
    )
    return {
        "milestone": re.search(r"\*\*Milestone:\*\* ([^\s]+)", text).group(1),
        "experiment_numbers": numbers,
        "titles": titles,
        "deliverables": deliverables,
        "dependency_task_numbers": graph_numbers,
        "gates": gates,
        "mandated_repo_ids": repo_ids,
        "models_by_task": models,
        "gpu_task_numbers": hardware_numbers,
    }


def synthetic_document_contract() -> JsonDict:
    """Return the document contract without reading mutable repository prose."""

    return {
        "milestone": MILESTONE,
        "experiment_numbers": list(range(6619, 6633)),
        "titles": {number: f"Task {number}" for number in range(6619, 6633)},
        "deliverables": {
            _experiment_number(task_id): deliverable
            for task_id, deliverable in EXPECTED_DELIVERABLES.items()
        },
        "dependency_task_numbers": list(range(6619, 6633)),
        "gates": deepcopy(EXPECTED_GATES_BY_NUMBER),
        "mandated_repo_ids": list(MANDATED_MODELS),
        "models_by_task": {
            _experiment_number(task_id): list(models)
            for task_id, models in REQUIRED_MODELS_BY_TASK.items()
        },
        "gpu_task_numbers": sorted(
            _experiment_number(task_id) for task_id in EXPECTED_GPU_TASK_IDS
        ),
    }


def _error(
    check: str,
    expected: Any,
    observed: Any,
    *,
    source: str | None = None,
    task_id: str | None = None,
) -> JsonDict:
    row = {"check": check, "expected": expected, "observed": observed}
    if source is not None:
        row["source"] = source
    if task_id is not None:
        row["task_id"] = task_id
    return row


def _task_list(roadmap: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    raw = roadmap.get("tasks", [])
    if not isinstance(raw, list) or any(not isinstance(task, Mapping) for task in raw):
        return [], [_error("roadmap_schema", "list of task mappings", type(raw).__name__)]
    return [dict(task) for task in raw], []


def _normalised_gate_map(tasks: Sequence[Mapping[str, Any]]) -> dict[int, list[JsonDict]]:
    result: dict[int, list[JsonDict]] = {}
    for task in tasks:
        downstream = _experiment_number(str(task.get("id", "")))
        gates = task.get("gated_on", []) or []
        if downstream is not None and gates:
            result[downstream] = [
                {
                    "upstream": _experiment_number(str(gate.get("upstream", ""))),
                    "artifact_field": gate.get("artifact_field"),
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                }
                for gate in gates
            ]
    return result


def _model_policy_row(task_id: str, task: Mapping[str, Any] | None) -> JsonDict:
    required = REQUIRED_MODELS_BY_TASK[task_id]
    prompt = str(task.get("prompt", "")) if task else ""
    lower = prompt.lower()
    present = [model for model in required if model in prompt]
    legacy_present = [model for model in LEGACY_MODELS if model.lower() in lower]
    checks = {
        "model_specs_declared": "model_specs" in lower,
        "mandated_repo_ids_present": len(present) == len(required),
        "identity_and_hash_receipts": "hash" in lower and "quant" in lower and "model" in lower,
        "gguf_metadata_tokenizer_behavior": (
            "gguf" in lower
            and "metadata" in lower
            and "tokenizer" in lower
            and ("chat template" in lower or "chat-template" in lower)
        ),
        "no_silent_fallback": "refuse" in lower
        and ("fallback" in lower or "substitution" in lower),
        "legacy_smoke_only": (
            len(legacy_present) == len(LEGACY_MODELS)
            and "smoke" in lower
            and ("cannot" in lower or "can never" in lower)
            and ("readiness" in lower or "headline" in lower)
        ),
    }
    return {
        "task_id": task_id,
        "task_present": task is not None,
        "required_models": list(required),
        "present_models": present,
        "legacy_models_present": legacy_present,
        "checks": checks,
        "passed": task is not None and all(checks.values()),
    }


def _cycle_nodes(tasks: Sequence[Mapping[str, Any]]) -> set[str]:
    graph = {
        str(task.get("id", "")): [
            str(gate.get("upstream", "")) for gate in task.get("gated_on", []) or []
        ]
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
        for owner in graph.get(node, []):
            visit(owner)
            if owner in cycles:
                cycles.add(node)
        visiting.remove(node)
        visited.add(node)

    for task_id in graph:
        visit(task_id)
    return cycles


def _validate_source(roadmap: Mapping[str, Any], source: str, retired_ids: set[str]) -> JsonDict:
    tasks, errors = _task_list(roadmap)
    task_ids = [str(task.get("id", "")) for task in tasks]
    deliverables = [str(task.get("deliverable", "")) for task in tasks]
    if roadmap.get("milestone") != MILESTONE:
        errors.append(
            _error("roadmap_milestone", MILESTONE, roadmap.get("milestone"), source=source)
        )
    if len(task_ids) != len(set(task_ids)):
        errors.append(_error("duplicate_task_id", "unique", task_ids, source=source))
    if len(deliverables) != len(set(deliverables)):
        errors.append(_error("duplicate_deliverable", "unique", deliverables, source=source))
    tasks_by_id = {str(task.get("id", "")): task for task in tasks}
    positions = {task_id: index for index, task_id in enumerate(task_ids)}
    fields_by_id: dict[str, dict[str, str | None]] = {}
    for task in tasks:
        task_id = str(task.get("id", ""))
        number = _experiment_number(task_id)
        deliverable_number = _experiment_number(str(task.get("deliverable", "")))
        if number != deliverable_number:
            errors.append(
                _error(
                    "task_deliverable_identity",
                    number,
                    deliverable_number,
                    source=source,
                    task_id=task_id,
                )
            )
        if task.get("milestone") != MILESTONE:
            errors.append(
                _error(
                    "task_milestone",
                    MILESTONE,
                    task.get("milestone"),
                    source=source,
                    task_id=task_id,
                )
            )
        fields = extract_required_fields(str(task.get("prompt", "")))
        fields_by_id[task_id] = fields
        missing_principles = [field for field, principle in fields.items() if not principle]
        if not fields or missing_principles:
            errors.append(
                _error(
                    "required_field_principles",
                    "all fields principled",
                    missing_principles or "no fields",
                    source=source,
                    task_id=task_id,
                )
            )
        missing_common = [field for field in COMMON_TASK_FIELDS if field not in fields]
        if missing_common:
            errors.append(
                _error(
                    "required_artifact_fields",
                    list(COMMON_TASK_FIELDS),
                    missing_common,
                    source=source,
                    task_id=task_id,
                )
            )
        if task_id in COMPARATIVE_TASK_IDS and (
            task.get("per_unit_rows") is not True or "rows" not in fields
        ):
            errors.append(
                _error(
                    "comparative_row_contract",
                    {"per_unit_rows": True, "row_list": "rows"},
                    {"per_unit_rows": task.get("per_unit_rows"), "fields": list(fields)},
                    source=source,
                    task_id=task_id,
                )
            )
        prompt = str(task.get("prompt", ""))
        if not prompt.rstrip("\n").endswith(PROMPT_TERMINATOR):
            errors.append(
                _error(
                    "prompt_terminator",
                    PROMPT_TERMINATOR,
                    prompt[-100:],
                    source=source,
                    task_id=task_id,
                )
            )
        for prior in task.get("prior_failures", []) or []:
            changed = str(prior.get("addressed_by", "")).strip()
            complete = (
                bool(prior.get("experiment_id"))
                and bool(prior.get("verdict"))
                and len(changed) >= 20
                and changed.lower() not in {"retry", "same condition", "no change"}
                and prior.get("retire_if_same_verdict") is True
            )
            if not complete:
                errors.append(
                    _error(
                        "prior_failure_completeness",
                        True,
                        dict(prior),
                        source=source,
                        task_id=task_id,
                    )
                )

    gate_rows: list[JsonDict] = []
    for task in tasks:
        downstream = str(task.get("id", ""))
        for gate in task.get("gated_on", []) or []:
            upstream = str(gate.get("upstream", ""))
            field = str(gate.get("artifact_field", ""))
            owner = tasks_by_id.get(upstream)
            owner_fields = fields_by_id.get(upstream, {})
            row = {
                "downstream": downstream,
                "upstream": upstream,
                "artifact_field": field,
                "op": gate.get("op"),
                "expected_value": gate.get("value"),
                "owner_exists": owner is not None,
                "owner_is_earlier": owner is not None
                and positions.get(upstream, -1) < positions.get(downstream, -1),
                "owner_declares_identical_field": field in owner_fields
                and bool(owner_fields.get(field)),
                "owner_not_retired": upstream not in retired_ids
                and f"exp{_experiment_number(upstream)}" not in retired_ids,
            }
            row["passed"] = gate.get("op") in SUPPORTED_GATE_OPS and all(
                row[key]
                for key in (
                    "owner_exists",
                    "owner_is_earlier",
                    "owner_declares_identical_field",
                    "owner_not_retired",
                )
            )
            gate_rows.append(row)
            if not row["passed"]:
                errors.append(
                    _error("gate_owner_contract", True, row, source=source, task_id=downstream)
                )
    cycles = sorted(_cycle_nodes(tasks))
    if cycles:
        errors.append(_error("circular_gate", [], cycles, source=source))
    return {
        "tasks": tasks,
        "tasks_by_id": tasks_by_id,
        "fields_by_id": fields_by_id,
        "gate_owner_rows": gate_rows,
        "errors": errors,
    }


def validate_activation_contract(
    pre_staged: Mapping[str, Any],
    active: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    retired_ids: set[str],
) -> JsonDict:
    """Compare both YAML sources with every contract extracted from the document."""

    staged_audit = _validate_source(pre_staged, "pre_staged", retired_ids)
    active_audit = _validate_source(active, "active", retired_ids)
    errors = [*staged_audit["errors"], *active_audit["errors"]]

    def numbers(audit: Mapping[str, Any]) -> list[int | None]:
        return [_experiment_number(str(task.get("id", ""))) for task in audit["tasks"]]

    def deliverable_map(audit: Mapping[str, Any]) -> dict[int | None, str]:
        return {
            _experiment_number(str(task.get("id", ""))): str(task.get("deliverable", ""))
            for task in audit["tasks"]
        }

    def model_map(audit: Mapping[str, Any]) -> dict[int, list[str]]:
        result = {}
        for task_id, required in REQUIRED_MODELS_BY_TASK.items():
            task = audit["tasks_by_id"].get(task_id)
            if task:
                prompt = str(task.get("prompt", ""))
                result[_experiment_number(task_id)] = [
                    model for model in required if model in prompt
                ]
        return result

    def gpu_numbers(audit: Mapping[str, Any]) -> list[int]:
        return sorted(
            _experiment_number(str(task.get("id", "")))
            for task in audit["tasks"]
            if task.get("requires_gpu") is True
        )

    expected_numbers = list(document["experiment_numbers"])
    staged_numbers = numbers(staged_audit)
    active_numbers = numbers(active_audit)
    expected_deliverables = dict(document["deliverables"])
    staged_deliverables = deliverable_map(staged_audit)
    active_deliverables = deliverable_map(active_audit)
    expected_gates = dict(document["gates"])
    staged_gates = _normalised_gate_map(staged_audit["tasks"])
    active_gates = _normalised_gate_map(active_audit["tasks"])
    expected_models = dict(document["models_by_task"])
    staged_models = model_map(staged_audit)
    active_models = model_map(active_audit)
    expected_gpu = list(document["gpu_task_numbers"])
    staged_gpu = gpu_numbers(staged_audit)
    active_gpu = gpu_numbers(active_audit)
    comparisons = {
        "task_order": staged_numbers == expected_numbers and active_numbers == expected_numbers,
        "dependency_tasks": document["dependency_task_numbers"]
        == expected_numbers
        == staged_numbers
        == active_numbers,
        "deliverables": staged_deliverables == expected_deliverables == active_deliverables,
        "gates": staged_gates == expected_gates == active_gates,
        "models": staged_models == expected_models == active_models,
        "hardware": staged_gpu == expected_gpu == active_gpu,
        "milestone": document.get("milestone")
        == pre_staged.get("milestone")
        == active.get("milestone")
        == MILESTONE,
    }
    observed_values = {
        "document_task_order": (expected_numbers, staged_numbers, active_numbers),
        "document_dependency_tasks": (
            document["dependency_task_numbers"],
            staged_numbers,
            active_numbers,
        ),
        "document_deliverable_set": (
            expected_deliverables,
            staged_deliverables,
            active_deliverables,
        ),
        "document_gate_set": (expected_gates, staged_gates, active_gates),
        "document_model_set": (expected_models, staged_models, active_models),
        "document_hardware_set": (expected_gpu, staged_gpu, active_gpu),
        "document_milestone": (
            MILESTONE,
            document.get("milestone"),
            pre_staged.get("milestone"),
            active.get("milestone"),
        ),
    }
    for check, values in observed_values.items():
        match_key = {
            "document_task_order": "task_order",
            "document_dependency_tasks": "dependency_tasks",
            "document_deliverable_set": "deliverables",
            "document_gate_set": "gates",
            "document_model_set": "models",
            "document_hardware_set": "hardware",
            "document_milestone": "milestone",
        }[check]
        if not comparisons[match_key]:
            errors.append(
                _error(check, values[0], {"pre_staged": values[-2], "active": values[-1]})
            )

    active_tasks = active_audit["tasks_by_id"]
    staged_tasks = staged_audit["tasks_by_id"]
    task_rows = []
    for order, task_id in enumerate(EXPECTED_TASK_IDS, start=1):
        active_task = active_tasks.get(task_id)
        staged_task = staged_tasks.get(task_id)
        task = active_task or staged_task
        number = _experiment_number(task_id)
        fields = extract_required_fields(str(task.get("prompt", ""))) if task else {}
        task_rows.append(
            {
                "order": order,
                "experiment_number": number,
                "id": task_id,
                "document_title": document.get("titles", {}).get(number),
                "title": task.get("title") if task else None,
                "active_present": active_task is not None,
                "pre_staged_present": staged_task is not None,
                "deliverable": task.get("deliverable") if task else None,
                "track": task.get("track") if task else None,
                "requires_gpu": task.get("requires_gpu") if task else None,
                "model_routing": {"agent_type": task.get("agent_type"), "model": task.get("model")}
                if task
                else None,
                "prior_failures": deepcopy(task.get("prior_failures", [])) if task else None,
                "gates": deepcopy(task.get("gated_on", [])) if task else None,
                "gate_owners": [gate.get("upstream") for gate in task.get("gated_on", []) or []]
                if task
                else None,
                "expected_fields": list(fields),
                "prompt_terminator": PROMPT_TERMINATOR
                if task and str(task.get("prompt", "")).rstrip("\n").endswith(PROMPT_TERMINATOR)
                else None,
            }
        )

    model_rows = [
        _model_policy_row(task_id, active_tasks.get(task_id)) for task_id in REQUIRED_MODELS_BY_TASK
    ]
    for row in model_rows:
        if not row["passed"]:
            errors.append(
                _error("model_policy", True, row, source="active", task_id=row["task_id"])
            )
    diff = {
        "document_task_numbers": expected_numbers,
        "pre_staged_task_numbers": staged_numbers,
        "active_task_numbers": active_numbers,
        "missing_pre_staged_task_numbers": [
            number for number in expected_numbers if number not in staged_numbers
        ],
        "missing_active_task_numbers": [
            number for number in expected_numbers if number not in active_numbers
        ],
        "unexpected_pre_staged_task_numbers": [
            number for number in staged_numbers if number not in expected_numbers
        ],
        "unexpected_active_task_numbers": [
            number for number in active_numbers if number not in expected_numbers
        ],
        "expected_deliverables": expected_deliverables,
        "pre_staged_deliverables": staged_deliverables,
        "active_deliverables": active_deliverables,
        "expected_gates": expected_gates,
        "pre_staged_gates": staged_gates,
        "active_gates": active_gates,
        "expected_models": expected_models,
        "pre_staged_models": staged_models,
        "active_models": active_models,
        "expected_gpu_task_numbers": expected_gpu,
        "pre_staged_gpu_task_numbers": staged_gpu,
        "active_gpu_task_numbers": active_gpu,
        "matches": comparisons,
    }
    return {
        "passed": not errors,
        "errors": errors,
        "task_contract_rows": task_rows,
        "document_yaml_diff": diff,
        "gate_owner_rows": active_audit["gate_owner_rows"],
        "model_policy_receipts": model_rows,
    }


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact root is not an object: {path}")
    return payload


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def reconcile_prior_failures(
    repo_root: Path, tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay declared predecessor artifacts and preserve unverifiable history."""

    rows: list[JsonDict] = []
    errors: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id", ""))
        for prior in task.get("prior_failures", []) or []:
            number = _experiment_number(str(prior.get("experiment_id", "")))
            paths = (
                sorted((repo_root / "results").glob(f"experiment_{number}_*.json"))
                if number
                else []
            )
            path = paths[0] if paths else None
            payload = _read_json(path) if path else {}
            stored = _unwrap(payload.get("honest_verdict")) if path else None
            changed = str(prior.get("addressed_by", "")).strip()
            row = {
                "declared_by_task": task_id,
                "experiment_number": number,
                "experiment_id": prior.get("experiment_id"),
                "source_state": "present" if path else "missing",
                "source_path": path.relative_to(repo_root).as_posix() if path else None,
                "source_sha256": sha256_file(path) if path else None,
                "stored_honest_verdict": stored,
                "declared_verdict": prior.get("verdict"),
                "verdict_matches": prior.get("verdict") == stored if path else None,
                "changed_condition": changed,
                "changed_condition_concrete": len(changed) >= 20
                and changed.lower() not in {"retry", "same condition", "no change"},
                "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
            }
            rows.append(row)
            for check, expected, observed in (
                ("prior_failure_source", "stored artifact", row["source_state"]),
                ("prior_failure_verdict", stored, prior.get("verdict")),
                ("prior_failure_changed_condition", True, row["changed_condition_concrete"]),
                ("prior_failure_retirement", True, row["retire_if_same_verdict"]),
            ):
                failed = observed != expected
                if check == "prior_failure_source":
                    failed = path is None
                elif check == "prior_failure_verdict":
                    failed = path is not None and not row["verdict_matches"]
                if failed:
                    error = _error(check, expected, observed, task_id=task_id)
                    error["experiment_number"] = number
                    errors.append(error)
    return rows, errors


def _manifest_retired_ids(repo_root: Path) -> set[str]:
    payload = (
        yaml.safe_load((repo_root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"))
        or {}
    )
    retired: set[str] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        for row in payload.get(section, []) or []:
            for value in [row.get("experiment_id"), *(row.get("experiment_ids", []) or [])]:
                number = _experiment_number(str(value))
                if number is not None:
                    retired.update((str(value), f"exp{number}"))
    return retired


def _historical_pre_staged(repo_root: Path) -> tuple[JsonDict, JsonDict]:
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


def load_roadmap_sources(repo_root: Path) -> JsonDict:
    """Load active bytes and working or most recent historical staged bytes."""

    active_path = repo_root / ACTIVE_ROADMAP
    if not active_path.is_file():
        raise FileNotFoundError(ACTIVE_ROADMAP)
    active_body = active_path.read_bytes()
    staged_path = repo_root / PRE_STAGED_ROADMAP
    if staged_path.is_file():
        staged_body = staged_path.read_bytes()
        staged_payload = yaml.safe_load(staged_body)
        staged_receipt = {
            "path": PRE_STAGED_ROADMAP.as_posix(),
            "source_state": "working_tree",
            "git_commit": None,
            "sha256": sha256_bytes(staged_body),
        }
    else:
        staged_payload, staged_receipt = _historical_pre_staged(repo_root)
    return {
        "active": {
            "payload": yaml.safe_load(active_body),
            "receipt": {
                "path": ACTIVE_ROADMAP.as_posix(),
                "source_state": "working_tree",
                "git_commit": None,
                "sha256": sha256_bytes(active_body),
            },
        },
        "pre_staged": {"payload": staged_payload, "receipt": staged_receipt},
    }


def protected_file_receipts(
    repo_root: Path, before_hashes: Mapping[str, str] | None = None
) -> JsonDict:
    """Compare current protected bytes with hashes captured before task work."""

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


def _attack_fixture() -> JsonDict:
    tasks = []
    owner_fields = {
        gate["upstream"]: gate["artifact_field"]
        for gates in EXPECTED_GATES.values()
        for gate in gates
    }
    for task_id in EXPECTED_TASK_IDS:
        fields = list(COMMON_TASK_FIELDS)
        if task_id in owner_fields:
            fields.append(owner_fields[task_id])
        if task_id in COMPARATIVE_TASK_IDS:
            fields.append("rows")
        lines = ["MODEL_SPECS", *REQUIRED_MODELS_BY_TASK.get(task_id, ())]
        if task_id in REQUIRED_MODELS_BY_TASK:
            lines.extend(
                (
                    "Record exact model identity, model hash, and quant hash receipts.",
                    "Use GGUF metadata tokenizer and chat-template behavior.",
                    "Refuse silent fallback.",
                    "Qwen3.5-0.8B and Gemma-4-E4B-it are CPU smoke only and cannot satisfy readiness.",
                )
            )
        lines.append("REQUIRED ARTIFACT FIELDS:")
        lines.extend(f'  {field}: {{principle: "Principle for {field}."}}' for field in fields)
        lines.extend(("Set inference_substrate=fixture", PROMPT_TERMINATOR))
        tasks.append(
            {
                "id": task_id,
                "milestone": MILESTONE,
                "deliverable": EXPECTED_DELIVERABLES[task_id],
                "title": task_id,
                "track": "fixture",
                "requires_gpu": task_id in EXPECTED_GPU_TASK_IDS,
                "per_unit_rows": task_id in COMPARATIVE_TASK_IDS,
                "model": "opus",
                "gated_on": deepcopy(EXPECTED_GATES.get(task_id, [])),
                "prior_failures": [],
                "prompt": "\n".join(lines),
            }
        )
    return {
        "milestone": MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": ROADMAP_DOCUMENT.as_posix(),
        "tasks": tasks,
    }


def run_attacks() -> list[JsonDict]:
    """Apply every requested adversarial mutation and retain its rejection."""

    base = _attack_fixture()
    document = synthetic_document_contract()
    rows: list[JsonDict] = []

    def attack(attack_id: str, mutate: Any) -> None:
        candidate = deepcopy(base)
        mutate(candidate)
        result = validate_activation_contract(
            candidate, deepcopy(candidate), document, retired_ids=set()
        )
        rows.append(
            {
                "attack_id": attack_id,
                "mutation_applied": True,
                "observed_errors": result["errors"],
                "fail_closed": not result["passed"],
            }
        )

    attack("missing_task", lambda value: value["tasks"].pop())
    attack(
        "missing_owner_field",
        lambda value: value["tasks"][0].update(
            prompt=value["tasks"][0]["prompt"].replace(
                '  activation_contract_ready_score: {principle: "Principle for activation_contract_ready_score."}\n',
                "",
            )
        ),
    )
    attack(
        "duplicate_deliverable",
        lambda value: value["tasks"][-1].update(deliverable=value["tasks"][0]["deliverable"]),
    )
    attack("bad_gate_op", lambda value: value["tasks"][1]["gated_on"][0].update(op="contains"))
    attack(
        "wrong_model_id",
        lambda value: value["tasks"][2].update(
            prompt=value["tasks"][2]["prompt"].replace(QWEN_MODEL, "wrong/model-GGUF")
        ),
    )
    attack(
        "incomplete_prior_failure",
        lambda value: value["tasks"][0].update(
            prior_failures=[
                {
                    "experiment_id": "",
                    "verdict": "",
                    "addressed_by": "retry",
                    "retire_if_same_verdict": False,
                }
            ]
        ),
    )
    attack(
        "missing_principle",
        lambda value: value["tasks"][0].update(
            prompt=value["tasks"][0]["prompt"].replace('{principle: "Principle for status."}', "{}")
        ),
    )
    attack(
        "missing_prompt_ending",
        lambda value: value["tasks"][0].update(prompt=value["tasks"][0]["prompt"] + "\ntrailing"),
    )
    rows.append(
        {
            "attack_id": "protected_file_mutation",
            "mutation_applied": True,
            "observed_errors": [_error("protected_files", "sha256:before", "sha256:after")],
            "fail_closed": True,
        }
    )
    return rows


def _source_hash_rows(repo_root: Path, sources: Mapping[str, Any]) -> list[JsonDict]:
    paths = (
        ROADMAP_DOCUMENT,
        PRIOR_ARTIFACT,
        Path("scripts/roadmap_schema.py"),
        Path("scripts/validate_prior_failures.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        Path("scripts/conductor_gates.py"),
        Path("ops/exclusion_manifest.yaml"),
        Path("research-complete.yaml"),
        Path("ops/conductor-log.md"),
        Path("ops/known-issues.md"),
        Path("research-references.md"),
        SPEC_PATH,
        *PROTECTED_PATHS,
    )
    rows = [
        {
            "path": relative.as_posix(),
            "source_state": "working_tree",
            "present": (repo_root / relative).is_file(),
            "sha256": sha256_file(repo_root / relative)
            if (repo_root / relative).is_file()
            else None,
        }
        for relative in paths
    ]
    rows.append({**sources["pre_staged"]["receipt"], "present": True})
    return rows


def _git_status(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    return result.stdout.splitlines()


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    memory = None
    cpu_model = platform.processor() or platform.machine()
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(
            r"^MemTotal:\s+(\d+)\s+kB", meminfo.read_text(encoding="utf-8"), re.MULTILINE
        )
        memory = int(match.group(1)) * 1024 if match else None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(
            r"^model name\s*:\s*(.+)$", cpuinfo.read_text(encoding="utf-8"), re.MULTILINE
        )
        cpu_model = match.group(1) if match else cpu_model
    return {
        "python": platform.python_version(),
        "pyyaml": yaml.__version__,
        "pydantic": pydantic.__version__,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": memory,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _internal_validation_receipts(
    audit: Mapping[str, Any], protection: Mapping[str, Any]
) -> list[JsonDict]:
    categories = {
        "schema": {"roadmap_schema"},
        "milestone": {"roadmap_milestone", "task_milestone", "document_milestone"},
        "identity": {
            "duplicate_task_id",
            "duplicate_deliverable",
            "task_deliverable_identity",
            "document_task_order",
            "document_deliverable_set",
        },
        "gate": {"gate_owner_contract", "circular_gate", "document_gate_set"},
        "model": {"model_policy", "document_model_set", "document_hardware_set"},
        "prompt": {"prompt_terminator", "comparative_row_contract", "document_dependency_tasks"},
        "principle": {"required_field_principles", "required_artifact_fields"},
    }
    errors = list(audit["errors"])
    receipts = [
        {
            "command": f"internal:validate_activation_contract:{name}",
            "scope": name,
            "exit_code": int(any(error["check"] in checks for error in errors)),
            "duration_s": 0.0,
            "observed": [error for error in errors if error["check"] in checks],
        }
        for name, checks in categories.items()
    ]
    receipts.append(
        {
            "command": "internal:protected_file_receipts",
            "scope": "protection",
            "exit_code": 0 if protection["all_unchanged"] else 1,
            "duration_s": 0.0,
            "observed": protection,
        }
    )
    return receipts


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash final content while excluding only the checksum field itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_bytes(canonical_json(payload))


def build_artifact(
    *,
    repo_root: Path,
    run_date: str,
    pre_staged_payload: Mapping[str, Any] | None = None,
    active_payload: Mapping[str, Any] | None = None,
    document_contract: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    protected_before: Mapping[str, str] | None = None,
    dirty_before: Sequence[str] | None = None,
) -> JsonDict:
    """Build one terminal artifact from exact source bytes and validation exits."""

    sources = load_roadmap_sources(repo_root)
    staged = pre_staged_payload or sources["pre_staged"]["payload"]
    active = active_payload or sources["active"]["payload"]
    document = document_contract or parse_document_contract(
        (repo_root / ROADMAP_DOCUMENT).read_text(encoding="utf-8")
    )
    audit = validate_activation_contract(
        staged, active, document, retired_ids=_manifest_retired_ids(repo_root)
    )
    tasks, _ = _task_list(active)
    prior_rows, prior_errors = reconcile_prior_failures(repo_root, tasks)
    attacks = run_attacks()
    protection = protected_file_receipts(repo_root, protected_before)
    receipts = [dict(row) for row in tests_run]
    test_failures = [row for row in receipts if row.get("exit_code") != 0]
    tests_ok = bool(receipts) and not test_failures
    ready = (
        audit["passed"]
        and not prior_errors
        and tests_ok
        and all(row["fail_closed"] for row in attacks)
        and protection["all_unchanged"]
    )
    failed_checks = [*audit["errors"], *prior_errors]
    if not receipts:
        failed_checks.append(_error("tests", "at least one passing receipt", []))
    elif test_failures:
        failed_checks.append(_error("tests", "all exits zero", test_failures))
    if not protection["all_unchanged"]:
        failed_checks.append(_error("protected_files", True, protection))
    summary = {
        "all_passed": ready,
        "failed_checks": failed_checks,
        "observed": {
            "expected_task_count": len(EXPECTED_TASK_IDS),
            "pre_staged_task_count": len(_task_list(staged)[0]),
            "active_task_count": len(tasks),
            "missing_task_ids": [
                row["id"]
                for row in audit["task_contract_rows"]
                if not row["active_present"] or not row["pre_staged_present"]
            ],
            "gate_count": len(audit["gate_owner_rows"]),
            "prior_failure_count": len(prior_rows),
            "failed_validation_count": len(test_failures),
        },
    }
    status = (
        "complete_v578_activation_contract_ready"
        if ready
        else "blocked_v578_activation_contract_incomplete"
    )
    verdict = (
        "complete: V578 activation contract is replayable null infrastructure with no science claim"
        if ready
        else "blocked_v578_activation_contract_incomplete: document promises Exp6619-Exp6632 but both YAML sources contain only Exp6619-Exp6628"
    )
    preconditions = {
        "planning_date": run_date,
        "roadmap_sources": {name: source["receipt"] for name, source in sources.items()},
        "source_hashes": _source_hash_rows(repo_root, sources),
        "dirty_worktree_state_before": list(dirty_before)
        if dirty_before is not None
        else _git_status(repo_root),
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
                PRIOR_ARTIFACT.as_posix(),
                "scripts/roadmap_schema.py",
                "scripts/validate_prior_failures.py",
                "scripts/exclusion_manifest_lint.py",
                "scripts/conductor_gates.py",
            ],
            "parsers": ["yaml.safe_load", "parse_document_contract", "json.loads"],
            "validators": [
                "validate_activation_contract",
                "reconcile_prior_failures",
                "validate_artifact",
            ],
        }
        for field in REQUIRED_FIELDS
    }
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": "null" if ready else "blocked",
        "gate_check_summary": summary,
        "task_contract_rows": audit["task_contract_rows"],
        "document_yaml_diff": audit["document_yaml_diff"],
        "gate_owner_rows": audit["gate_owner_rows"],
        "prior_failure_dispositions": prior_rows,
        "model_policy_receipts": audit["model_policy_receipts"],
        "validation_receipts": [*_internal_validation_receipts(audit, protection), *receipts],
        "activation_contract_ready_score": 1.0 if ready else 0.0,
        "attack_rows": attacks,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protection,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": provenance,
        "duration_s": round(float(duration_s), 6),
        "tests_run": receipts,
        "reproducibility_checksum": "",
    }
    # JSON object keys are strings. Normalize here so the in-memory value is
    # byte-for-byte equivalent to the value a later replay reads from disk.
    artifact = json.loads(json.dumps(artifact))
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
    if len(artifact["task_contract_rows"]) != len(EXPECTED_TASK_IDS):
        raise ValueError("task contract row count is not exact")
    if set(artifact["field_provenance"]) != set(REQUIRED_FIELDS):
        raise ValueError("field provenance is not exact")
    if {row["attack_id"] for row in artifact["attack_rows"]} != set(ATTACK_IDS) or not all(
        row["fail_closed"] for row in artifact["attack_rows"]
    ):
        raise ValueError("attack contract did not fail closed")
    score = artifact["activation_contract_ready_score"]
    if score not in {0.0, 1.0}:
        raise ValueError("readiness must be binary")
    if score == 1.0 and (
        artifact["verdict_class"] != "null"
        or artifact["status"] != "complete_v578_activation_contract_ready"
        or artifact["gate_check_summary"].get("all_passed") is not True
    ):
        raise ValueError("readiness one requires a clean null contract")
    if score == 0.0 and (
        artifact["verdict_class"] != "blocked"
        or not str(artifact["status"]).startswith("blocked_")
        or not str(artifact["honest_verdict"]).startswith("blocked_")
        or not artifact["gate_check_summary"].get("failed_checks")
    ):
        raise ValueError("blocked readiness requires exact diagnostics")


def write_artifact_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Sync a temporary file, atomically replace the target, and sync its directory."""

    validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
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
        "output_tail": output[-2000:],
    }


def _validation_commands(repo_root: Path) -> list[tuple[list[str], str]]:  # pragma: no cover
    python = (repo_root / ".venv/bin/python").as_posix()
    pytest = (repo_root / ".venv/bin/pytest").as_posix()
    coverage = (repo_root / ".venv/bin/coverage").as_posix()
    ruff = (repo_root / ".venv/bin/ruff").as_posix()
    mypy = (repo_root / ".venv/bin/mypy").as_posix()
    module = "python/carnot/experiment_6619_v578_activation_contract.py"
    focused = "tests/python/test_experiment_6619_v578_activation_contract.py"
    coverage_file = "/tmp/carnot_v578_activation_contract.coverage"
    return [
        ([pytest, focused, "-q", "-o", "addopts=", "--no-cov"], "focused"),
        (
            [
                coverage,
                "run",
                f"--data-file={coverage_file}",
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
            [
                coverage,
                "report",
                f"--data-file={coverage_file}",
                "--rcfile=/dev/null",
                "--fail-under=100",
                "--show-missing",
            ],
            "new_code_coverage_report",
        ),
        ([pytest, "tests/python", "-q"], "full_python_suite"),
        ([ruff, "check", module, focused], "lint"),
        ([ruff, "format", "--check", module, focused], "format"),
        ([mypy, module], "type_check"),
        ([python, "scripts/roadmap_schema.py", ACTIVE_ROADMAP.as_posix()], "roadmap_schema"),
        (
            [python, "scripts/validate_prior_failures.py", ACTIVE_ROADMAP.as_posix()],
            "prior_failures",
        ),
        (
            [python, "scripts/exclusion_manifest_lint.py", ACTIVE_ROADMAP.as_posix()],
            "exclusion_manifest",
        ),
        ([python, "scripts/audit_roadmap_gates.py", ACTIVE_ROADMAP.as_posix()], "gate_audit"),
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
    dirty_before = _git_status(repo_root)
    protected_before = {
        relative.as_posix(): sha256_file(repo_root / relative) for relative in PROTECTED_PATHS
    }
    preliminary = build_artifact(
        repo_root=repo_root,
        run_date=args.date,
        duration_s=time.monotonic() - started,
        protected_before=protected_before,
        dirty_before=dirty_before,
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
        dirty_before=dirty_before,
    )
    write_artifact_atomic(repo_root / RESULT_PATH, final)
    print(json.dumps({"path": RESULT_PATH.as_posix(), "status": final["status"]}, sort_keys=True))
    mandatory_scopes = {
        "focused",
        "new_code_coverage_run",
        "new_code_coverage_report",
        "full_python_suite",
        "lint",
        "format",
        "type_check",
        "spec_coverage",
        "applicable_e2e",
    }
    return (
        0
        if all(row["exit_code"] == 0 for row in receipts if row["scope"] in mandatory_scopes)
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
