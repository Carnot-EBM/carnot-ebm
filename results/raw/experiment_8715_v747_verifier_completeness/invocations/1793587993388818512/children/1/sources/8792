#!/usr/bin/env python3
"""Audit the V595 owned contracts and active execution manifest.

The V594 plan failed before research because it named a capability that did
not exist. This module keeps the repair mechanical. It maps each contract to
an existing requirement, compares the design with the active YAML, and fails
closed when a gate or task cannot be checked from written evidence.

Spec refs: REQ-AGENTIC-6810-1, REQ-AGENTIC-6810-2,
REQ-CONSTRAINT-6810, and REQ-CL-6810.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import yaml


ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESULT_PATH = Path("results/experiment_6810_v595_contract_manifest_preflight.json")
EXPECTED_MILESTONE = "2026.08.595"
EXPECTED_EXPERIMENTS = tuple(range(6810, 6824))
AUDIT_SEED = 6810

PERMITTED_MODEL_SPECS = frozenset(
    {
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-12B-it-GGUF",
    }
)

SOURCE_PATHS = (
    DESIGN_PATH,
    ROADMAP_PATH,
    Path("openspec/capabilities/agentic-harness/spec.md"),
    Path("openspec/capabilities/constraint-verification/spec.md"),
    Path("openspec/capabilities/continuous-learning/spec.md"),
    Path("results/experiment_6802_operational_obligation_automaton_v2.json"),
    Path("results/experiment_6803_sota_operational_handoff_corpus.json"),
    Path("ops/exclusion_manifest.yaml"),
    Path("scripts/validate_prior_failures.py"),
)


@dataclass(frozen=True)
class ContractOwner:
    """Describe the written evidence needed to prove one contract has an owner."""

    spec_path: str
    requirement: str
    implementing_task: str
    artifact_path: str
    gate_field: str
    authority_boundary: str
    failure_behavior: str


@dataclass(frozen=True)
class DesignContract:
    """Hold the task and phase identities extracted from the design document."""

    phase_numbers: tuple[int, ...]
    task_numbers: tuple[int, ...]
    deliverables: dict[int, str]


CONTRACT_OWNERS = {
    "operational-obligation interface": ContractOwner(
        spec_path="openspec/capabilities/agentic-harness/spec.md",
        requirement="REQ-AGENTIC-6810-1",
        implementing_task="exp6811-operational-obligation-automaton-v3",
        artifact_path="results/experiment_6811_operational_obligation_automaton_v3.json",
        gate_field="operational_automaton_fixture_ready",
        authority_boundary="The exact post-action receipts remain",
        failure_behavior="complete_blocked_operational_obligation_automaton_v3",
    ),
    "exact priority arbiter": ContractOwner(
        spec_path="openspec/capabilities/constraint-verification/spec.md",
        requirement="REQ-CONSTRAINT-6810",
        implementing_task="exp6813-selective-priority-arbiter-ab",
        artifact_path="results/experiment_6813_selective_priority_arbiter_ab.json",
        gate_field="selective_arbiter_ab_completed",
        authority_boundary="transition evaluator remains authoritative.",
        failure_behavior="complete_blocked_selective_priority_arbiter_ab",
    ),
    "transactional verified memory": ContractOwner(
        spec_path="openspec/capabilities/continuous-learning/spec.md",
        requirement="REQ-CL-6810",
        implementing_task="exp6816-residual-pressure-route-learning-ab",
        artifact_path="results/experiment_6816_residual_pressure_route_learning_ab.json",
        gate_field="residual_route_learning_completed",
        authority_boundary="The exact local receipt remains authoritative.",
        failure_behavior="complete_blocked_residual_pressure_route_learning_ab",
    ),
    "live stepwise strategy path": ContractOwner(
        spec_path="openspec/capabilities/agentic-harness/spec.md",
        requirement="REQ-AGENTIC-6810-2",
        implementing_task="exp6819-arc-stepwise-strategy-accrual",
        artifact_path="results/experiment_6819_arc_stepwise_strategy_accrual.json",
        gate_field="stepwise_strategy_accrual_ready",
        authority_boundary="Exact next-action outcomes",
        failure_behavior="complete_blocked_arc_stepwise_strategy_accrual",
    ),
}


FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents later readers from guessing this artifact shape.",
    "experiment_id": "A stable identity binds the evidence to the task that produced it.",
    "run_date": "The execution date distinguishes this audit from later roadmap revisions.",
    "field_principles": "Each field states why it exists so compliance remains auditable.",
    "inference_substrate": "This is a CPU document audit and makes no model-inference claim.",
    "duration_s": "Measured wall time exposes fabricated or silently skipped work.",
    "random_seed": "A fixed seed keeps deterministic audit ordering reproducible.",
    "reproducibility_checksum": "The hash binds inputs, rows, commands, and output content.",
    "source_artifact_hashes": "Hashes bind the V594 evidence and every owned contract input.",
    "contract_owner_map": "Existing specs prevent another dependency on an invented capability.",
    "roadmap_identity": "The milestone and task range prevent stale-roadmap activation.",
    "rows": "Task and owner rows make every readiness claim independently checkable.",
    "task_count": "The exact count detects the five tasks omitted from the first active YAML.",
    "gate_resolution_results": "Every downstream gate must resolve to a written producer field.",
    "deliverable_uniqueness_results": "Unique paths prevent one task from overwriting another.",
    "model_spec_results": "Model checks prevent legacy or substituted models from supporting evidence.",
    "prior_failure_results": "Four-part failure records prevent unchanged blocked work from recurring.",
    "prompt_contract_results": "Prompt-tail and row checks keep execution commands and boundaries exact.",
    "verification_commands": "Recorded commands let another reader reproduce the contract audit.",
    "v595_contract_map_ready": "Exp6811 may start only after every owned contract and manifest check passes.",
    "gate_check_summary": "Blocked output names the failed check, expected value, and observed value.",
    "verifier_is_oracle": "This audit checks contracts; it does not define scientific correctness.",
    "verdict_class": "A closed result class prevents positive language from hiding a blocked audit.",
    "honest_verdict": "The terminal verdict states only what the written evidence supports.",
}


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject shapes the roadmap auditor cannot inspect."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"top-level roadmap value must be a mapping: {path}")
    return data


def requirement_section(spec_text: str, requirement: str) -> str:
    """Return one requirement section without borrowing text from a later owner."""

    heading = re.search(rf"^#{{2,3}} {re.escape(requirement)}(?::|$)", spec_text, re.MULTILINE)
    if heading is None:
        return ""
    following = re.search(
        r"^#{2,3} (?:REQ-|Implementation Status)",
        spec_text[heading.end() :],
        re.MULTILINE,
    )
    end = heading.end() + following.start() if following else len(spec_text)
    return spec_text[heading.start() : end]


def parse_design(path: Path) -> DesignContract:
    """Extract phase numbers, experiment numbers, and deliverables from V595 prose."""

    text = path.read_text(encoding="utf-8")
    phases = tuple(int(value) for value in re.findall(r"^## Phase (\d+):", text, re.MULTILINE))
    tasks = tuple(int(value) for value in re.findall(r"^### Exp(\d+):", text, re.MULTILINE))
    deliverables: dict[int, str] = {}
    task_matches = list(re.finditer(r"^### Exp(\d+):", text, re.MULTILINE))
    for index, match in enumerate(task_matches):
        section_end = task_matches[index + 1].start() if index + 1 < len(task_matches) else len(text)
        section = text[match.end() : section_end]
        artifact = re.search(r"\*\*Deliverable:\*\* `([^`]+)`", section)
        if artifact:
            deliverables[int(match.group(1))] = artifact.group(1)
    return DesignContract(phases, tasks, deliverables)


def _task_number(task_id: str) -> int | None:
    match = re.fullmatch(r"exp(\d+)-[a-z0-9-]+", task_id)
    return int(match.group(1)) if match else None


def _tasks(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(roadmap, dict):
        raise ValueError("top-level roadmap value must be a mapping")
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("top-level roadmap tasks must be a list")
    return [task for task in tasks if isinstance(task, dict)]


def audit_task_contracts(
    roadmap: dict[str, Any], design: DesignContract
) -> list[dict[str, Any]]:
    """Compare every active task identity and deliverable with the V595 design."""

    tasks = _tasks(roadmap)
    ids = [str(task.get("id") or "") for task in tasks]
    deliverables = [str(task.get("deliverable") or "") for task in tasks]
    id_counts = Counter(ids)
    deliverable_counts = Counter(deliverables)
    global_identity = (
        roadmap.get("milestone") == EXPECTED_MILESTONE
        and len(tasks) == len(EXPECTED_EXPERIMENTS)
        and design.phase_numbers == (1, 2, 3, 4)
        and design.task_numbers == EXPECTED_EXPERIMENTS
    )
    rows: list[dict[str, Any]] = []
    for position, task in enumerate(tasks):
        task_id = ids[position]
        number = _task_number(task_id)
        deliverable = deliverables[position]
        expected_number = EXPECTED_EXPERIMENTS[position] if position < len(EXPECTED_EXPERIMENTS) else None
        row = {
            "row_type": "task",
            "task_id": task_id,
            "experiment_number": number,
            "deliverable": deliverable,
            "unique_task_id": bool(task_id) and id_counts[task_id] == 1,
            "unique_deliverable": bool(deliverable) and deliverable_counts[deliverable] == 1,
            "consecutive_expected_id": number == expected_number,
            "design_has_task": number in design.task_numbers,
            "design_deliverable_matches": design.deliverables.get(number) == deliverable,
            "task_milestone_matches": task.get("milestone") == EXPECTED_MILESTONE,
            "ungated": not bool(task.get("gated_on")),
        }
        final_gate_shape = task_id != "exp6823-v595-branch-disposition" or row["ungated"]
        row["passed"] = bool(
            global_identity
            and row["unique_task_id"]
            and row["unique_deliverable"]
            and row["consecutive_expected_id"]
            and row["design_has_task"]
            and row["design_deliverable_matches"]
            and row["task_milestone_matches"]
            and final_gate_shape
        )
        rows.append(row)
    return rows


def retired_experiment_ids(repo_root: Path) -> set[str]:
    """Return normalized retired IDs from every exclusion-manifest section."""

    data = load_yaml_mapping(repo_root / "ops/exclusion_manifest.yaml")
    retired: set[str] = set()
    for section_name in ("retired", "retired_experiments", "retired_extras"):
        for entry in data.get(section_name, []) or []:
            if not isinstance(entry, dict):
                continue
            values = ([entry["experiment_id"]] if "experiment_id" in entry else []) + list(
                entry.get("experiment_ids", []) or []
            )
            for value in values:
                normalized = str(value).strip().lower()
                normalized = normalized if normalized.startswith("exp") else f"exp{normalized}"
                retired.add(normalized)
                match = re.match(r"exp\d+", normalized)
                if match:
                    retired.add(match.group(0))
    return retired


def _required_fields_block(prompt: str) -> str:
    marker = re.search(r"REQUIRED ARTIFACT FIELDS AND PRINCIPLES:", prompt, re.IGNORECASE)
    if marker is None:
        return ""
    block = prompt[marker.start() :]
    return re.sub(r"_\s+", "_", block.split("\n\nRun command:", 1)[0])


def _is_retired(upstream: str, retired_ids: set[str]) -> bool:
    normalized = upstream.lower()
    base = re.match(r"exp\d+", normalized)
    return normalized in retired_ids or bool(base and base.group(0) in retired_ids)


def audit_gate_resolution(
    roadmap: dict[str, Any], retired_ids: set[str]
) -> list[dict[str, Any]]:
    """Resolve each gate to a non-positive field declared by its producer."""

    tasks = _tasks(roadmap)
    tasks_by_id = {str(task.get("id") or ""): task for task in tasks}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for gate in task.get("gated_on", []) or []:
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            producer = tasks_by_id.get(upstream)
            upstream_exists = producer is not None
            declared = bool(producer and field and field in _required_fields_block(str(producer.get("prompt") or "")))
            positivity = bool(re.search(r"(?:positive|acceptance_gate|effect|improvement)", field, re.IGNORECASE))
            retired = _is_retired(upstream, retired_ids)
            exact_boolean_gate = gate.get("op") == "==" and gate.get("value") is True
            row = {
                "row_type": "gate",
                "task_id": str(task.get("id") or ""),
                "upstream": upstream,
                "artifact_field": field,
                "upstream_exists": upstream_exists,
                "producer_declares_field_verbatim": declared,
                "positivity_gate": positivity,
                "upstream_retired": retired,
                "exact_boolean_completion_gate": exact_boolean_gate,
            }
            row["passed"] = bool(
                upstream_exists and declared and not positivity and not retired and exact_boolean_gate
            )
            rows.append(row)
    return rows


def audit_model_specs(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    """Require each live LLM task to name its frozen permitted local model set."""

    rows: list[dict[str, Any]] = []
    for task in _tasks(roadmap):
        task_id = str(task.get("id") or "")
        prompt = str(task.get("prompt") or "")
        llm_task = bool(task.get("requires_gpu"))
        declared = sorted(model for model in PERMITTED_MODEL_SPECS if model in prompt)
        expected = set()
        if task_id == "exp6812-sota-operational-handoff-corpus-v2":
            expected = set(PERMITTED_MODEL_SPECS) - {"unsloth/gemma-4-12B-it-GGUF"}
        elif llm_task:
            expected = {"unsloth/Qwen3.6-35B-A3B-GGUF"}
        row = {
            "row_type": "model_spec",
            "task_id": task_id,
            "llm_task": llm_task,
            "declared_permitted_models": declared,
            "expected_models": sorted(expected),
            "model_specs_declared": "MODEL_SPECS" in prompt,
        }
        row["passed"] = bool(
            not llm_task
            or (row["model_specs_declared"] and expected.issubset(set(declared)))
        )
        rows.append(row)
    return rows


def audit_prior_failures(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    """Check each repeated scope has the complete four-part failure declaration."""

    rows: list[dict[str, Any]] = []
    for task in _tasks(roadmap):
        declarations = task.get("prior_failures")
        valid = isinstance(declarations, list) and bool(declarations)
        if valid:
            valid = all(
                isinstance(item, dict)
                and bool(str(item.get("experiment_id") or "").strip())
                and bool(str(item.get("verdict") or "").strip())
                and bool(str(item.get("addressed_by") or "").strip())
                and item.get("retire_if_same_verdict") is True
                for item in declarations
            )
        rows.append(
            {
                "row_type": "prior_failure",
                "task_id": str(task.get("id") or ""),
                "declaration_count": len(declarations) if isinstance(declarations, list) else 0,
                "four_part_declarations_valid": bool(valid),
                "passed": bool(valid),
            }
        )
    return rows


def audit_prompt_contracts(roadmap: dict[str, Any]) -> list[dict[str, Any]]:
    """Check per-unit evidence and the exact executable prompt ending for every task."""

    rows: list[dict[str, Any]] = []
    for task in _tasks(roadmap):
        prompt = str(task.get("prompt") or "").rstrip()
        deliverable = Path(str(task.get("deliverable") or ""))
        command = (
            "Run command: cd {project_root} && .venv/bin/python "
            f"scripts/experiments/{deliverable.stem}.py --date {{date}}"
        )
        required_tail = f"{command}\nDo NOT push. Do NOT modify scripts/research_conductor.py."
        comparison = bool(
            re.search(r"(?:\bA/B\b|\bcompare\b|\bversus\b|\bmatched\b)", prompt, re.IGNORECASE)
        )
        per_unit_rows = task.get("per_unit_rows") is True
        tail_matches = prompt.endswith(required_tail)
        rows.append(
            {
                "row_type": "prompt_contract",
                "task_id": str(task.get("id") or ""),
                "comparison_task": comparison,
                "per_unit_rows": per_unit_rows,
                "required_run_command": command,
                "prompt_tail_matches": tail_matches,
                "passed": per_unit_rows and tail_matches,
            }
        )
    return rows


def _audit_contract_owners(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract, owner in CONTRACT_OWNERS.items():
        spec = repo_root / owner.spec_path
        section = requirement_section(spec.read_text(encoding="utf-8"), owner.requirement) if spec.exists() else ""
        checks = {
            "requirement_present": bool(section),
            "implementing_task_present": owner.implementing_task in section,
            "artifact_path_present": owner.artifact_path in section,
            "gate_field_present": owner.gate_field in section,
            "authority_boundary_present": owner.authority_boundary in section,
            "failure_behavior_present": owner.failure_behavior in section,
        }
        rows.append(
            {
                "row_type": "requirement_owner",
                "contract": contract,
                "spec_path": owner.spec_path,
                "requirement": owner.requirement,
                "implementing_task": owner.implementing_task,
                "artifact_path": owner.artifact_path,
                "gate_field": owner.gate_field,
                **checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _sha256(path: Path) -> str | None:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}" if path.is_file() else None


def payload_checksum(artifact: dict[str, Any]) -> str:
    """Hash canonical artifact content while excluding the checksum field itself."""

    payload = deepcopy(artifact)
    payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _failed_check(check: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {"check": check, "expected": expected, "observed": observed, "passed": False}


def _base_artifact(repo_root: Path, run_date: str, duration_s: float) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_6810.v595_contract_manifest_preflight.v1",
        "experiment_id": "experiment_6810_v595_contract_manifest_preflight",
        "run_date": run_date,
        "field_principles": {},
        "inference_substrate": "CPU schema and document audit, no LLM",
        "duration_s": duration_s,
        "random_seed": AUDIT_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {
            str(path): _sha256(repo_root / path) for path in SOURCE_PATHS
        },
        "contract_owner_map": {},
        "roadmap_identity": {},
        "rows": [],
        "task_count": 0,
        "gate_resolution_results": [],
        "deliverable_uniqueness_results": [],
        "model_spec_results": [],
        "prior_failure_results": [],
        "prompt_contract_results": [],
        "verification_commands": [
            "cd {project_root} && .venv/bin/python "
            "scripts/experiments/experiment_6810_v595_contract_manifest_preflight.py --date {date}"
        ],
        "v595_contract_map_ready": False,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v595_contract_manifest_preflight",
    }


def _finish_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["field_principles"] = {
        field: FIELD_PRINCIPLES[field] for field in artifact
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    repo_root: Path,
    run_date: str,
    duration_s: float,
    *,
    roadmap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ready or blocked artifact from the checked-in V595 contracts."""

    active = roadmap if roadmap is not None else load_yaml_mapping(repo_root / ROADMAP_PATH)
    artifact = _base_artifact(repo_root, run_date, duration_s)
    milestone = active.get("milestone")
    if milestone != EXPECTED_MILESTONE:
        artifact["roadmap_identity"] = {
            "milestone": milestone,
            "first_experiment": None,
            "last_experiment": None,
        }
        artifact["gate_check_summary"] = [
            _failed_check("active_roadmap_milestone", EXPECTED_MILESTONE, milestone)
        ]
        return _finish_artifact(artifact)

    design = parse_design(repo_root / DESIGN_PATH)
    task_rows = audit_task_contracts(active, design)
    owner_rows = _audit_contract_owners(repo_root)
    gate_rows = audit_gate_resolution(active, retired_experiment_ids(repo_root))
    model_rows = audit_model_specs(active)
    prior_rows = audit_prior_failures(active)
    prompt_rows = audit_prompt_contracts(active)
    all_groups = (task_rows, owner_rows, gate_rows, model_rows, prior_rows, prompt_rows)
    ready = all(row["passed"] for group in all_groups for row in group)
    numbers = [row["experiment_number"] for row in task_rows if row["experiment_number"] is not None]

    artifact.update(
        {
            "contract_owner_map": {
                contract: {
                    "spec_path": owner.spec_path,
                    "requirement": owner.requirement,
                    "implementing_task": owner.implementing_task,
                    "artifact_path": owner.artifact_path,
                    "gate_field": owner.gate_field,
                    "authority_boundary": owner.authority_boundary,
                    "failure_behavior": owner.failure_behavior,
                }
                for contract, owner in CONTRACT_OWNERS.items()
            },
            "roadmap_identity": {
                "milestone": milestone,
                "first_experiment": min(numbers) if numbers else None,
                "last_experiment": max(numbers) if numbers else None,
            },
            "rows": task_rows + owner_rows,
            "task_count": len(task_rows),
            "gate_resolution_results": gate_rows,
            "deliverable_uniqueness_results": [
                {
                    "task_id": row["task_id"],
                    "deliverable": row["deliverable"],
                    "unique": row["unique_deliverable"],
                    "passed": row["unique_deliverable"],
                }
                for row in task_rows
            ],
            "model_spec_results": model_rows,
            "prior_failure_results": prior_rows,
            "prompt_contract_results": prompt_rows,
            "v595_contract_map_ready": ready,
            "verdict_class": "null" if ready else "blocked",
            "honest_verdict": (
                "complete: V595 owned contracts and active execution manifest agree"
                if ready
                else "complete_blocked_v595_contract_manifest_preflight"
            ),
        }
    )
    if not ready:
        artifact["gate_check_summary"] = [
            _failed_check(
                f"{row.get('row_type', 'contract')}:{row.get('task_id') or row.get('contract')}",
                True,
                row,
            )
            for group in all_groups
            for row in group
            if not row["passed"]
        ]
    return _finish_artifact(artifact)


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return stable validation errors without modifying the research record."""

    required = set(FIELD_PRINCIPLES)
    errors: list[str] = []
    missing = sorted(required - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(artifact):
        errors.append("field_principles_do_not_cover_artifact")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_outside_closed_enum")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("v595_contract_map_ready") is True:
        if artifact.get("task_count") != 14 or len(artifact.get("rows", [])) != 14 + len(CONTRACT_OWNERS):
            errors.append("ready_artifact_row_or_task_count_mismatch")
        if artifact.get("roadmap_identity") != {
            "milestone": EXPECTED_MILESTONE,
            "first_experiment": 6810,
            "last_experiment": 6823,
        }:
            errors.append("ready_artifact_roadmap_identity_mismatch")
        if artifact.get("gate_check_summary") != []:
            errors.append("ready_artifact_has_failed_gate_summary")
        if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
            errors.append("ready_artifact_verdict_is_not_terminal")
    else:
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_artifact_verdict_class_mismatch")
        if not artifact.get("gate_check_summary"):
            errors.append("blocked_artifact_missing_gate_summary")
        if artifact.get("honest_verdict") != "complete_blocked_v595_contract_manifest_preflight":
            errors.append("blocked_artifact_honest_verdict_mismatch")
    return errors


def _write_atomic(path: Path, artifact: dict[str, Any]) -> None:
    """Replace one requested artifact atomically so interrupted writes stay valid."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the preflight while allowing tests to isolate output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260831")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.project_root / RESULT_PATH

    if args.validate:
        errors = validate_artifact(json.loads(output.read_text(encoding="utf-8")))
        print("valid" if not errors else "\n".join(errors))
        return int(bool(errors))

    started = time.perf_counter()
    roadmap = load_yaml_mapping(args.project_root / ROADMAP_PATH)
    artifact = build_artifact(
        args.project_root,
        args.date,
        time.perf_counter() - started,
        roadmap=roadmap,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    _write_atomic(output, artifact)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the required run command.
    raise SystemExit(main())
