"""Build the deterministic V608 execution-contract preflight artifact.

The module compares two independent task manifests. It also runs the existing
roadmap guards. A failed check produces a blocked receipt instead of a partial
approval. See REQ-REPORT-6942.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6942_v608_contract_preflight.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
VERDICT_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
MILESTONE = "2026.09.608"
EXPECTED_NUMBERS = tuple(range(6941, 6953))
FINAL_PROHIBITIONS = "Do NOT push. Do NOT modify scripts/research_conductor.py."
INFERENCE_SUBSTRATE = "deterministic_roadmap_contract_preflight_no_llm"
RANDOM_SEED = 6942
BLOCKED_VERDICT = "blocked_v608_contract_preflight"
READY_VERDICT = "complete_circular_positive_v608_execution_contract_ready"
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
PROMPT_SECTIONS = ("CONTEXT:", "EXISTING CODE TO READ FIRST:", "TASK:", "CONCRETE STEPS:")
COMMON_PROMPT_FIELDS = {
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
MANDATED_GGUF_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-12B-it-GGUF",
)
MODEL_REQUIREMENTS = {
    6944: MANDATED_GGUF_MODELS[:3],
    6946: MANDATED_GGUF_MODELS[:3],
    6950: (MANDATED_GGUF_MODELS[0], MANDATED_GGUF_MODELS[2]),
}
REQUIRED_MUTATIONS = (
    "task_count",
    "id_order",
    "title",
    "deliverable",
    "gate_field",
    "model_specs",
    "prior_failure_field",
    "prompt_ending",
)


def _gate(upstream: str, artifact_field: str) -> JsonDict:
    """Return one exact equality gate from the locked task table."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": "==", "value": 1}


_TASK_VALUES = (
    (
        6941,
        "exp6941-v608-source-delta",
        "V608 post-marker source delta and compatibility audit",
        "results/experiment_6941_v608_source_delta.json",
        [],
    ),
    (
        6942,
        "exp6942-v608-contract-preflight",
        "V608 executable contract and bounded-shard preflight",
        "results/experiment_6942_v608_contract_preflight.json",
        [],
    ),
    (
        6943,
        "exp6943-verifier-density-prefix-corpus",
        "Verifier-density prefix corpus with exact first-error labels",
        "results/experiment_6943_verifier_density_prefix_corpus.json",
        [_gate("exp6942-v608-contract-preflight", "v608_execution_contract_ready_score")],
    ),
    (
        6944,
        "exp6944-three-family-prefix-bank",
        "Three-family bounded non-saturated prefix reasoning bank",
        "results/experiment_6944_three_family_prefix_bank.json",
        [_gate("exp6943-verifier-density-prefix-corpus", "prefix_corpus_ready_score")],
    ),
    (
        6945,
        "exp6945-prefix-credit-energy-canary",
        "Exact-prefix credit geometry and structural-energy canary",
        "results/experiment_6945_prefix_credit_energy_canary.json",
        [_gate("exp6944-three-family-prefix-bank", "reasoning_bank_complete_score")],
    ),
    (
        6946,
        "exp6946-gguf-causal-state-surface",
        "GGUF causal-state surface and replay receipt",
        "results/experiment_6946_gguf_causal_state_surface.json",
        [_gate("exp6944-three-family-prefix-bank", "reasoning_bank_complete_score")],
    ),
    (
        6947,
        "exp6947-causal-hidden-selection",
        "Causal hidden-state selection with random-direction controls",
        "results/experiment_6947_causal_hidden_selection.json",
        [_gate("exp6946-gguf-causal-state-surface", "causal_state_surface_ready_score")],
    ),
    (
        6948,
        "exp6948-arc-branch-corpus",
        "ARC live-attempt branching corpus audit",
        "results/experiment_6948_arc_branch_corpus.json",
        [_gate("exp6942-v608-contract-preflight", "v608_execution_contract_ready_score")],
    ),
    (
        6949,
        "exp6949-arc-branch-energy",
        "Within-game branch-discriminative world-state energy",
        "results/experiment_6949_arc_branch_energy.json",
        [_gate("exp6948-arc-branch-corpus", "arc_branch_corpus_ready_score")],
    ),
    (
        6950,
        "exp6950-trace-state-self-learning",
        "Prospective Trace-as-State continuous self-learning",
        "results/experiment_6950_trace_state_self_learning.json",
        [_gate("exp6943-verifier-density-prefix-corpus", "prefix_corpus_ready_score")],
    ),
    (
        6951,
        "exp6951-trace-memory-cold-audit",
        "Fresh-process trace-memory causal audit",
        "results/experiment_6951_trace_memory_cold_audit.json",
        [_gate("exp6950-trace-state-self-learning", "trace_state_run_complete_score")],
    ),
    (
        6952,
        "exp6952-v608-capstone",
        "V608 independent capstone and V609 handoff",
        "results/experiment_6952_v608_capstone.json",
        [],
    ),
)
EXPECTED_TASKS = tuple(
    {
        "order": order,
        "number": number,
        "task_id": task_id,
        "title": title,
        "deliverable": deliverable,
        "gates": gates,
    }
    for order, (number, task_id, title, deliverable, gates) in enumerate(_TASK_VALUES, 1)
)

ROW_FIELDS = (
    "document_task_rows",
    "yaml_task_rows",
    "task_parity_rows",
    "gate_contract_rows",
    "producer_field_rows",
    "prompt_contract_rows",
    "model_contract_rows",
    "prior_failure_rows",
    "exclusion_manifest_rows",
    "lint_command_rows",
    "mutation_rows",
    "bounded_scope_rows",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    *ROW_FIELDS,
    "random_seed",
    "reproducibility_checksum",
    "v608_execution_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}


def _principle(field: str) -> str:
    """Give every artifact field a plain reason that an auditor can inspect."""

    reasons = {
        "field_principles": "Field reasons expose why each recorded value is necessary.",
        "preconditions_checked": "Preflight rows prevent missing tools from looking like a clean audit.",
        "inference_substrate": "The substrate states that no language model produced the verdict.",
        "duration_s": "Measured duration distinguishes an executed audit from a copied result.",
        "source_artifact_hashes": "Source hashes bind each conclusion to exact contract bytes.",
        "rows": "A flat evidence ledger makes omissions and failed checks visible.",
        "document_task_rows": "Document rows preserve the design source independently.",
        "yaml_task_rows": "YAML rows preserve the executable source independently.",
        "task_parity_rows": "Parity rows reveal count, order, identity, and text drift.",
        "gate_contract_rows": "Gate rows prevent dependencies on absent or retired work.",
        "producer_field_rows": "Producer rows prove that each gate can read a declared value.",
        "prompt_contract_rows": "Prompt rows preserve executable instructions and prohibitions.",
        "model_contract_rows": "Model rows prevent legacy or invalid tokenizer plans from heading results.",
        "prior_failure_rows": "Failure rows make repeated methods and changed mechanisms auditable.",
        "exclusion_manifest_rows": "Exclusion rows stop retired work from re-entering a live dependency path.",
        "lint_command_rows": "Raw command receipts distinguish a run guard from an assumed guard.",
        "mutation_rows": "Mutation rows prove that each contract rule can fail when violated.",
        "bounded_scope_rows": "Scope rows prevent unbounded GPU work and silent non-artifacts.",
        "random_seed": "A fixed seed declares that this deterministic audit has no hidden randomness.",
        "reproducibility_checksum": "A content hash detects later changes to the receipt.",
        "v608_execution_contract_ready_score": "A binary score prevents partial approval of a failed contract.",
        "gate_check_summary": "Expected and observed values make every blocked check actionable.",
        "verifier_is_oracle": "True limits this oracle to contract conformance, not scientific truth.",
        "verdict_class": "A closed class keeps downstream status handling deterministic.",
        "honest_verdict": "A terminal prefix lets the conductor classify the result without guessing.",
    }
    return reasons.get(
        field, f"The {field} field binds this receipt to one reproducible audit state."
    )


FIELD_PRINCIPLES = {field: _principle(field) for field in REQUIRED_ARTIFACT_FIELDS}
PRECONDITION_PATHS = {
    "active_roadmap": ACTIVE_ROADMAP_PATH,
    "design_document": DESIGN_PATH,
    "roadmap_schema": SCHEMA_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "prior_failure_validator": PRIOR_VALIDATOR_PATH,
    "exclusion_manifest_lint": EXCLUSION_LINT_PATH,
    "roadmap_gate_audit": GATE_AUDIT_PATH,
    "verdict_row_consistency_lint": VERDICT_LINT_PATH,
}
SOURCE_PATHS = {
    **PRECONDITION_PATHS,
    "spec": SPEC_PATH,
    "module": Path("python/carnot/experiment_6942_v608_contract_preflight.py"),
    "wrapper": Path("scripts/experiments/experiment_6942_v608_contract_preflight.py"),
    "focused_tests": Path("tests/python/test_experiment_6942_v608_contract_preflight.py"),
}


def experiment_number(value: Any) -> int:
    """Extract an experiment number without accepting an unrelated integer."""

    match = re.match(r"^exp(\d+)(?:-|$)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _parse_scalar(text: str) -> Any:
    """Parse a design-table gate value with YAML scalar rules."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list)):
        raise ValueError(f"structured gate value must be scalar: {text}")
    return value


def _parse_design_gate(text: str) -> list[JsonDict]:
    """Turn the design table gate cell into the YAML gate shape."""

    if text.strip().lower() == "none":
        return []
    match = re.fullmatch(
        r"(exp\d+[a-z0-9-]*)\.([a-z][a-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+)",
        text.strip(),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"invalid structured gate: {text}")
    return [
        {
            "upstream": match.group(1),
            "artifact_field": match.group(2),
            "op": match.group(3),
            "value": _parse_scalar(match.group(4)),
        }
    ]


def parse_design(text: str) -> JsonDict:
    """Parse only the V608 task table without consulting YAML."""

    milestone = re.search(r"^\*\*Milestone:\*\*\s*`?([^`\s]+)`?\s*$", text, re.MULTILINE)
    if not milestone:
        raise ValueError("design milestone is required")
    section = re.search(
        r"^## Exact Task Contract\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not section:
        raise ValueError("design task table is required")
    rows = []
    for line in section.group("body").splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError(f"malformed design task row: {line}")
        order, task_id, title, deliverable, gate_text = cells
        rows.append(
            {
                "order": int(order),
                "number": experiment_number(task_id),
                "task_id": task_id,
                "title": title,
                "deliverable": deliverable,
                "gates": _parse_design_gate(gate_text),
                "milestone": milestone.group(1),
            }
        )
    if not rows:
        raise ValueError("design task table is required")
    return {"milestone": milestone.group(1), "tasks": rows}


def _required_fields_block(prompt: str) -> str:
    """Return the declared artifact-field block before the run command."""

    marker = re.search(r"REQUIRED ARTIFACT FIELDS:", prompt, re.IGNORECASE)
    if not marker:
        return ""
    tail = prompt[marker.start() :]
    return re.split(r"\n\s*Run command:", tail, maxsplit=1, flags=re.IGNORECASE)[0]


def parse_roadmap(document: Any) -> JsonDict:
    """Parse executable task values while retaining prompts and failure rows."""

    if not isinstance(document, Mapping):
        raise ValueError("roadmap mapping is required")
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("roadmap tasks list is required")
    rows = []
    for order, task in enumerate(tasks, 1):
        if not isinstance(task, Mapping):
            raise ValueError(f"malformed roadmap task at order {order}")
        gates = task.get("gated_on", []) or []
        priors = task.get("prior_failures", []) or []
        if not isinstance(gates, list) or not isinstance(priors, list):
            raise ValueError(f"malformed roadmap task lists at order {order}")
        prompt = str(task.get("prompt") or "")
        rows.append(
            {
                "order": order,
                "number": experiment_number(task.get("id")),
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "milestone": str(task.get("milestone") or ""),
                "gates": [dict(gate) if isinstance(gate, Mapping) else {} for gate in gates],
                "prior_failures": deepcopy(priors),
                "prompt": prompt,
                "required_fields_block": _required_fields_block(prompt),
                "requires_gpu": task.get("requires_gpu") is True,
                "per_unit_rows": task.get("per_unit_rows") is True,
                "estimated_wall_time_min": task.get("estimated_wall_time_min"),
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def retired_experiment_ids(document: Any) -> set[str]:
    """Read every numeric retirement section used by the manifest."""

    if not isinstance(document, Mapping):
        return set()
    ids: set[str] = set()
    for section in ("retired", "retired_experiments"):
        entries = document.get(section, []) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and isinstance(entry.get("experiment_id"), int):
                ids.add(f"exp{entry['experiment_id']}")
    return ids


def _is_retired(task_id: str, retired_ids: set[str]) -> bool:
    """Match a full task ID against numeric manifest IDs."""

    number = experiment_number(task_id)
    return number >= 0 and (f"exp{number}" in retired_ids or task_id in retired_ids)


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep every decision paired with expected and observed values."""

    return {"check": name, "expected": expected, "observed": observed, "passed": bool(passed)}


def _prompt_row(task: Mapping[str, Any]) -> JsonDict:
    """Check prompt sections, the task-owned command, and the final line."""

    prompt = str(task["prompt"])
    indices = [prompt.find(section) for section in PROMPT_SECTIONS]
    sections_ok = all(index >= 0 for index in indices) and indices == sorted(indices)
    expected_command = (
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        f"{Path(str(task['deliverable'])).stem}.py --date {{date}}"
    )
    run_commands = re.findall(r"^\s*Run command:.*$", prompt, re.MULTILINE)
    command_ok = len(run_commands) == 1 and run_commands[0].strip() == expected_command
    ending_ok = prompt.rstrip().endswith(FINAL_PROHIBITIONS)
    passed = sections_ok and command_ok and ending_ok
    return {
        "task_id": task["task_id"],
        "section_positions": dict(zip(PROMPT_SECTIONS, indices, strict=True)),
        "expected_run_command": expected_command,
        "observed_run_commands": [line.strip() for line in run_commands],
        "push_prohibition_present": "Do NOT push." in prompt,
        "conductor_prohibition_present": "Do NOT modify scripts/research_conductor.py." in prompt,
        "exact_terminal_line": ending_ok,
        "passed": passed,
    }


def _model_row(task: Mapping[str, Any]) -> JsonDict:
    """Check the complete model set and reject an invalid GGUF tokenizer call."""

    number = int(task["number"])
    prompt = str(task["prompt"])
    required = MODEL_REQUIREMENTS.get(number, ())
    model_bearing = bool(required) or bool(task["requires_gpu"])
    present = [model_id for model_id in MANDATED_GGUF_MODELS if model_id in prompt]
    missing = [model_id for model_id in required if model_id not in prompt]
    tokenizer_call = bool(
        re.search(r"AutoTokenizer\.from_pretrained\([^)]*GGUF[^)]*\)", prompt, re.IGNORECASE)
    )
    marker_present = "MODEL_SPECS" in prompt
    passed = not tokenizer_call
    if model_bearing:
        passed = passed and marker_present and bool(present) and not missing
    return {
        "task_id": task["task_id"],
        "model_bearing": model_bearing,
        "model_specs_present": marker_present,
        "required_models": list(required),
        "mandated_models_present": present,
        "missing_required_models": missing,
        "legacy_only_headline": model_bearing and not present,
        "gguf_autotokenizer_call": tokenizer_call,
        "passed": passed,
    }


def _prior_row(task: Mapping[str, Any]) -> JsonDict:
    """Check each declared prior failure without inventing missing matches."""

    required = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    failures = []
    for index, prior in enumerate(task["prior_failures"]):
        if not isinstance(prior, Mapping):
            failures.append({"index": index, "missing": sorted(required), "malformed": True})
            continue
        missing = sorted(
            field
            for field in required
            if field not in prior
            or (field != "retire_if_same_verdict" and not str(prior.get(field) or "").strip())
        )
        if missing:
            failures.append({"index": index, "missing": missing, "malformed": False})
    return {
        "task_id": task["task_id"],
        "prior_failure_count": len(task["prior_failures"]),
        "field_failures": failures,
        "passed": not failures,
    }


def _has_checkpoint_ceiling(prompt: str) -> tuple[bool, bool]:
    """Require a per-unit checkpoint phrase and a finite numeric ceiling."""

    compact = " ".join(prompt.split())
    checkpoint = bool(
        re.search(
            r"(?:after each|per[- ](?:row|event|unit|output|call|sequence)).{0,90}checkpoint|"
            r"checkpoint.{0,90}(?:after each|per[- ](?:row|event|unit|output|call|sequence))",
            compact,
            re.IGNORECASE,
        )
    )
    ceiling = bool(
        re.search(
            r"\b(?:exactly|at most|up to|maximum|max)\s+\d+\s+"
            r"(?:attempted\s+)?(?:rows?|outputs?|events?|calls?|sequences?|attempts?)\b",
            compact,
            re.IGNORECASE,
        )
    )
    return checkpoint, ceiling


def _scope_row(task: Mapping[str, Any]) -> JsonDict:
    """Check resource ceilings and the required blocked-artifact branch."""

    prompt = str(task["prompt"])
    estimate = task["estimated_wall_time_min"]
    wall_ok = isinstance(estimate, int) and not isinstance(estimate, bool) and 0 < estimate <= 720
    blocked_ok = bool(
        re.search(
            r"PRECONDITIONS:.*?\bwrite\s+blocked_[a-z0-9_]+\s+with\s+gate_check_summary",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )
    checkpoint, ceiling = _has_checkpoint_ceiling(prompt)
    gpu_ok = not task["requires_gpu"] or (checkpoint and ceiling and task["per_unit_rows"])
    return {
        "task_id": task["task_id"],
        "requires_gpu": task["requires_gpu"],
        "per_unit_rows": task["per_unit_rows"],
        "per_unit_checkpoint": checkpoint if task["requires_gpu"] else None,
        "finite_row_ceiling": ceiling if task["requires_gpu"] else None,
        "estimated_wall_time_min": estimate,
        "wall_time_at_most_720": wall_ok,
        "terminal_blocked_artifact_contract": blocked_ok,
        "passed": wall_ok and blocked_ok and gpu_ok,
    }


def evaluate_contract(design_text: str, roadmap_document: Any, retired_ids: set[str]) -> JsonDict:
    """Evaluate in-memory sources without invoking an external command."""

    design = parse_design(design_text)
    roadmap = parse_roadmap(roadmap_document)
    document_rows = design["tasks"]
    yaml_rows = roadmap["tasks"]
    parity_rows = []
    for order, (document, executable, expected) in enumerate(
        zip_longest(document_rows, yaml_rows, EXPECTED_TASKS), 1
    ):
        fields = ("order", "number", "task_id", "title", "deliverable", "gates")
        comparisons = {
            field: document is not None
            and executable is not None
            and expected is not None
            and document.get(field) == executable.get(field) == expected.get(field)
            for field in fields
        }
        parity_rows.append(
            {
                "order": order,
                "task_id": executable.get("task_id") if executable else None,
                "document": document,
                "yaml": executable,
                "expected": expected,
                "field_matches": comparisons,
                "passed": all(comparisons.values()),
            }
        )

    by_id = {task["task_id"]: task for task in yaml_rows}
    positions = {task["task_id"]: task["order"] for task in yaml_rows}
    gate_rows = []
    producer_rows = []
    for task in yaml_rows:
        for gate in task["gates"]:
            upstream = by_id.get(str(gate.get("upstream") or ""))
            retired = _is_retired(str(gate.get("upstream") or ""), retired_ids)
            precedes = upstream is not None and positions[upstream["task_id"]] < task["order"]
            gate_passed = upstream is not None and precedes and not retired
            gate_rows.append(
                {
                    "task_id": task["task_id"],
                    "gate": gate,
                    "upstream_exists": upstream is not None,
                    "upstream_precedes_consumer": precedes,
                    "upstream_retired": retired,
                    "passed": gate_passed,
                }
            )
            field = str(gate.get("artifact_field") or "")
            declared = bool(
                upstream
                and re.search(
                    rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])",
                    upstream["required_fields_block"],
                    re.IGNORECASE,
                )
            )
            producer_rows.append(
                {
                    "task_id": task["task_id"],
                    "upstream": gate.get("upstream"),
                    "artifact_field": field,
                    "producer_required_fields_block_present": bool(
                        upstream and upstream["required_fields_block"]
                    ),
                    "field_declared": declared,
                    "passed": upstream is not None and declared,
                }
            )

    prompt_rows = [_prompt_row(task) for task in yaml_rows]
    model_rows = [_model_row(task) for task in yaml_rows]
    prior_rows = [_prior_row(task) for task in yaml_rows]
    scope_rows = [_scope_row(task) for task in yaml_rows]
    exclusion_rows = [
        {
            "task_id": task["task_id"],
            "retired": _is_retired(task["task_id"], retired_ids),
            "operator_override": False,
            "passed": not _is_retired(task["task_id"], retired_ids),
        }
        for task in yaml_rows
    ]
    checks = [
        _check(
            "document_milestone", MILESTONE, design["milestone"], design["milestone"] == MILESTONE
        ),
        _check(
            "yaml_milestone", MILESTONE, roadmap["milestone"], roadmap["milestone"] == MILESTONE
        ),
        _check("document_task_count", 12, len(document_rows), len(document_rows) == 12),
        _check("yaml_task_count", 12, len(yaml_rows), len(yaml_rows) == 12),
        _check(
            "document_id_order",
            list(EXPECTED_NUMBERS),
            [row["number"] for row in document_rows],
            [row["number"] for row in document_rows] == list(EXPECTED_NUMBERS),
        ),
        _check(
            "yaml_id_order",
            list(EXPECTED_NUMBERS),
            [row["number"] for row in yaml_rows],
            [row["number"] for row in yaml_rows] == list(EXPECTED_NUMBERS),
        ),
    ]
    for name, rows in (
        ("task_parity", parity_rows),
        ("gate_contracts", gate_rows),
        ("producer_fields", producer_rows),
        ("prompt_contracts", prompt_rows),
        ("model_contracts", model_rows),
        ("prior_failure_fields", prior_rows),
        ("exclusion_manifest", exclusion_rows),
        ("bounded_scopes", scope_rows),
    ):
        checks.append(
            _check(
                name,
                "all rows pass",
                sum(row["passed"] for row in rows),
                all(row["passed"] for row in rows),
            )
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "document_task_rows": document_rows,
        "yaml_task_rows": yaml_rows,
        "task_parity_rows": parity_rows,
        "gate_contract_rows": gate_rows,
        "producer_field_rows": producer_rows,
        "prompt_contract_rows": prompt_rows,
        "model_contract_rows": model_rows,
        "prior_failure_rows": prior_rows,
        "exclusion_manifest_rows": exclusion_rows,
        "bounded_scope_rows": scope_rows,
    }


def build_mutation_rows(
    design_text: str, roadmap_document: Mapping[str, Any], retired_ids: set[str]
) -> list[JsonDict]:
    """Change each required dimension once and prove that it becomes invalid."""

    rows = []
    for mutation in REQUIRED_MUTATIONS:
        changed = deepcopy(roadmap_document)
        tasks = changed["tasks"]
        if mutation == "task_count":
            tasks.pop()
        elif mutation == "id_order":
            tasks[0], tasks[1] = tasks[1], tasks[0]
        elif mutation == "title":
            tasks[2]["title"] = f"{tasks[2]['title']} changed"
        elif mutation == "deliverable":
            tasks[3]["deliverable"] = "results/mutated_deliverable.json"
        elif mutation == "gate_field":
            tasks[2]["gated_on"][0]["artifact_field"] = "mutated_ready_score"
        elif mutation == "model_specs":
            tasks[3]["prompt"] = tasks[3]["prompt"].replace("MODEL_SPECS", "MODEL PLAN")
        elif mutation == "prior_failure_field":
            task = next(task for task in tasks if task.get("prior_failures"))
            task["prior_failures"][0].pop("addressed_by", None)
        else:
            tasks[0]["prompt"] = tasks[0]["prompt"].replace(FINAL_PROHIBITIONS, "Do NOT push.")
        result = evaluate_contract(design_text, changed, retired_ids)
        failed = [check["check"] for check in result["checks"] if not check["passed"]]
        rows.append(
            {
                "mutation": mutation,
                "contract_passed": result["passed"],
                "failed_checks": failed,
                "failed_as_expected": not result["passed"] and bool(failed),
            }
        )
    return rows


def _run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one bounded guard and retain its complete text receipt."""

    command = shlex.join(str(value) for value in argv)
    try:
        result = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        return {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "exit_code": 124,
            "stdout": str(exc.output or ""),
            "stderr": str(exc.stderr or ""),
            "passed": False,
        }


def run_lint_commands(root: Path, retired_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run the existing roadmap guards plus the deterministic retirement check."""

    python = sys.executable
    schema_program = (
        "from pathlib import Path; import sys,yaml; sys.path.insert(0, 'scripts'); "
        "from roadmap_schema import Roadmap; "
        "Roadmap.model_validate(yaml.safe_load(Path('research-roadmap.yaml').read_text())); "
        "print('roadmap schema validation clean')"
    )
    commands = (
        ("roadmap_schema", [python, "-c", schema_program]),
        ("prior_failure_validation", [python, str(PRIOR_VALIDATOR_PATH), str(ACTIVE_ROADMAP_PATH)]),
        ("exclusion_manifest_lint", [python, str(EXCLUSION_LINT_PATH), str(ACTIVE_ROADMAP_PATH)]),
        ("roadmap_gate_audit", [python, str(GATE_AUDIT_PATH), str(ACTIVE_ROADMAP_PATH)]),
    )
    rows = [_run_command(root, name, argv) for name, argv in commands]
    violations = [row["task_id"] for row in retired_rows if row.get("upstream_retired")]
    rows.append(
        {
            "name": "retired_upstream_check",
            "command": "internal retired_upstream_check research-roadmap.yaml ops/exclusion_manifest.yaml",
            "exit_code": 0 if not violations else 1,
            "stdout": json.dumps({"retired_upstream_tasks": violations}, sort_keys=True),
            "stderr": "",
            "passed": not violations,
        }
    )
    return rows


def _sha256(path: Path) -> str:
    """Hash one source without normalizing line endings or whitespace."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the scientific receipt while excluding duration and the hash itself."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _flat_rows(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Tag each evidence row so the combined ledger remains inspectable."""

    return [{"row_type": name, **dict(row)} for name, rows in groups.items() for row in rows]


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Read repository inputs, execute guards, and build one terminal receipt."""

    started = time.monotonic()
    preconditions = [
        {
            "resource": name,
            "path": str(path),
            "available": (root / path).is_file() and (root / path).stat().st_size > 0,
        }
        for name, path in PRECONDITION_PATHS.items()
    ]
    preconditions_ok = all(row["available"] for row in preconditions)
    source_hashes = {
        name: _sha256(root / path) if (root / path).is_file() else None
        for name, path in SOURCE_PATHS.items()
    }
    evaluation: JsonDict = {
        "passed": False,
        "checks": [],
        **{
            field: [] for field in ROW_FIELDS if field not in {"lint_command_rows", "mutation_rows"}
        },
    }
    mutation_rows: list[JsonDict] = []
    lint_rows: list[JsonDict] = []
    parse_error = ""
    if preconditions_ok:
        try:
            design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
            roadmap_document = yaml.safe_load(
                (root / ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8")
            )
            exclusion_document = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
            retired_ids = retired_experiment_ids(exclusion_document)
            evaluation = evaluate_contract(design_text, roadmap_document, retired_ids)
            mutation_rows = build_mutation_rows(design_text, roadmap_document, retired_ids)
            lint_rows = run_lint_commands(root, evaluation["gate_contract_rows"])
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    checks = [
        _check(
            "preconditions",
            "all required paths are readable non-empty files",
            preconditions,
            preconditions_ok,
        ),
        _check(
            "contract_parse",
            "both sources parse independently",
            parse_error or "parsed",
            preconditions_ok and not parse_error,
        ),
        *evaluation["checks"],
        _check(
            "lint_commands",
            "all command exit codes are zero",
            {row["name"]: row["exit_code"] for row in lint_rows},
            bool(lint_rows) and all(row["passed"] for row in lint_rows),
        ),
        _check(
            "required_mutations",
            list(REQUIRED_MUTATIONS),
            {row["mutation"]: row["failed_as_expected"] for row in mutation_rows},
            len(mutation_rows) == len(REQUIRED_MUTATIONS)
            and all(row["failed_as_expected"] for row in mutation_rows),
        ),
    ]
    ready = all(check["passed"] for check in checks)
    failures = [
        {"failed_check": row["check"], "expected": row["expected"], "observed": row["observed"]}
        for row in checks
        if not row["passed"]
    ]
    groups = {
        **{
            field: evaluation[field]
            for field in ROW_FIELDS
            if field not in {"lint_command_rows", "mutation_rows"}
        },
        "lint_command_rows": lint_rows,
        "mutation_rows": mutation_rows,
    }
    artifact: JsonDict = {
        "schema": "carnot.v608_execution_contract_preflight.v1",
        "experiment_id": 6942,
        "run_date": run_date,
        "status": "complete" if ready else "blocked",
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": source_hashes,
        "rows": _flat_rows(groups),
        **groups,
        "random_seed": RANDOM_SEED,
        "v608_execution_contract_ready_score": int(ready),
        "gate_check_summary": {
            "passed": ready,
            "failed_check": failures[0]["failed_check"] if failures else None,
            "expected": failures[0]["expected"] if failures else "all checks pass",
            "observed": failures[0]["observed"] if failures else "all checks pass",
            "failures": failures,
            "checks": checks,
        },
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "blocked",
        "honest_verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute all binary terminal claims from a serialized artifact."""

    errors = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not str(principles.get(field) or "").strip() for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    summary = artifact.get("gate_check_summary")
    checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
    expected_ready = int(bool(checks) and all(row.get("passed") is True for row in checks))
    if artifact.get("v608_execution_contract_ready_score") != expected_ready:
        errors.append("ready_score_mismatch")
    if expected_ready:
        if (
            artifact.get("status") != "complete"
            or artifact.get("verdict_class") != "circular_positive"
            or artifact.get("honest_verdict") != READY_VERDICT
        ):
            errors.append("ready_terminal_shape_mismatch")
    elif (
        artifact.get("status") != "blocked"
        or artifact.get("verdict_class") != "blocked"
        or artifact.get("honest_verdict") != BLOCKED_VERDICT
        or not isinstance(summary, Mapping)
        or not summary.get("failed_check")
        or "expected" not in summary
        or "observed" not in summary
    ):
        errors.append("blocked_terminal_shape_mismatch")
    mutations = artifact.get("mutation_rows")
    if not isinstance(mutations, list) or [row.get("mutation") for row in mutations] not in (
        [],
        list(REQUIRED_MUTATIONS),
    ):
        errors.append("mutation_rows_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _date_argument(value: str) -> str:
    """Reject ambiguous dates before any file write."""

    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the requested artifact only after a complete JSON write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preflight and write one validated terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"errors": errors}, indent=2), file=sys.stderr)
        return 1
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    _write_json_atomic(output, artifact)
    print(f"wrote {output}")
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the required entrypoint
    raise SystemExit(main())
