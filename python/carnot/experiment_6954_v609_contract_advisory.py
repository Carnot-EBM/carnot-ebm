"""Build the deterministic V609 advisory contract-audit artifact.

The module compares the design and executable roadmap as independent sources.
It reports contract defects without turning one global score into a science gate.
See REQ-REPORT-6954.
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
OUTPUT_PATH = Path("results/experiment_6954_v609_contract_advisory.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
VERDICT_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")
DOCS_AUDIT_PATH = Path("scripts/pages_adversarial_audit.py")
ARTIFACT_AUDIT_PATH = Path("scripts/artifact_convention_audit.py")
ROOT_CLUTTER_PATH = Path("scripts/root_clutter_sweep.py")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_PATH = Path("ops/known-issues.md")
PRIOR_ARTIFACT_PATH = Path("results/experiment_6942_v608_contract_preflight.json")
MILESTONE = "2026.09.609"
EXPECTED_NUMBERS = tuple(range(6953, 6965))
FINAL_PROHIBITIONS = "Do NOT push. Do NOT modify scripts/research_conductor.py."
INFERENCE_SUBSTRATE = "deterministic_advisory_contract_audit_no_llm"
RANDOM_SEED = 6954
BLOCKED_VERDICT = "blocked_v609_contract_advisory"
CONFORMS_VERDICT = "complete_circular_positive_v609_contract_conforms"
DEFECT_VERDICT = "complete_null_v609_contract_defects_found"
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
    6956: MANDATED_GGUF_MODELS[:3],
    6962: (MANDATED_GGUF_MODELS[0], MANDATED_GGUF_MODELS[2]),
}
ADVISORY_TASK_IDS = {
    "exp6953-v609-source-delta",
    "exp6954-v609-contract-advisory",
}
ADVISORY_SCORE_FIELDS = {
    "v609_source_delta_complete_score",
    "v609_contract_audit_complete_score",
    "v609_contract_conforms_score",
}
PRIOR_FAILURE_FIELDS = (
    "experiment_id",
    "verdict",
    "addressed_by",
    "retire_if_same_verdict",
)
REQUIRED_MUTATIONS = (
    "task_count",
    "id_order",
    "title",
    "deliverable",
    "gate_field",
    "producer_field",
    "model_specs",
    "prior_failure_experiment_id",
    "prior_failure_verdict",
    "prior_failure_addressed_by",
    "prior_failure_retire_if_same_verdict",
    "prompt_ending",
)


def _gate(upstream: str, artifact_field: str) -> JsonDict:
    """Return one exact equality gate for the locked task table."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": "==", "value": 1}


_TASK_VALUES = (
    (
        6953,
        "exp6953-v609-source-delta",
        "V609 post-marker source delta and compatibility audit",
        "results/experiment_6953_v609_source_delta.json",
        [],
    ),
    (
        6954,
        "exp6954-v609-contract-advisory",
        "V609 advisory execution-contract and gate-cascade audit",
        "results/experiment_6954_v609_contract_advisory.json",
        [],
    ),
    (
        6955,
        "exp6955-reformulation-fixture",
        "Exact optimization-reformulation mapping fixture",
        "results/experiment_6955_reformulation_fixture.json",
        [],
    ),
    (
        6956,
        "exp6956-three-family-reformulation-bank",
        "Three-family SOTA reformulation mapping bank",
        "results/experiment_6956_three_family_reformulation_bank.json",
        [_gate("exp6955-reformulation-fixture", "reformulation_fixture_ready_score")],
    ),
    (
        6957,
        "exp6957-smt-mapping-certification",
        "Independent SMT certification of SOTA mappings",
        "results/experiment_6957_smt_mapping_certification.json",
        [_gate("exp6956-three-family-reformulation-bank", "reformulation_bank_complete_score")],
    ),
    (
        6958,
        "exp6958-convex-factor-energy-canary",
        "Convex compositional factor-energy canary",
        "results/experiment_6958_convex_factor_energy_canary.json",
        [_gate("exp6955-reformulation-fixture", "reformulation_fixture_ready_score")],
    ),
    (
        6959,
        "exp6959-certified-energy-selection",
        "Causal certified-energy candidate selection",
        "results/experiment_6959_certified_energy_selection.json",
        [
            _gate(
                "exp6957-smt-mapping-certification",
                "smt_certification_run_complete_score",
            ),
            _gate(
                "exp6958-convex-factor-energy-canary",
                "convex_factor_run_complete_score",
            ),
        ],
    ),
    (
        6960,
        "exp6960-certified-selection-cold-audit",
        "Fresh-process certified-selection audit",
        "results/experiment_6960_certified_selection_cold_audit.json",
        [
            _gate(
                "exp6959-certified-energy-selection",
                "certified_selection_run_complete_score",
            )
        ],
    ),
    (
        6961,
        "exp6961-certified-event-sequence",
        "Sealed chronological outcome-certificate sequence",
        "results/experiment_6961_certified_event_sequence.json",
        [
            _gate(
                "exp6957-smt-mapping-certification",
                "smt_certification_run_complete_score",
            )
        ],
    ),
    (
        6962,
        "exp6962-queue-regulated-self-learning",
        "Queue-regulated continuous self-learning",
        "results/experiment_6962_queue_regulated_self_learning.json",
        [
            _gate(
                "exp6961-certified-event-sequence",
                "certified_event_sequence_ready_score",
            )
        ],
    ),
    (
        6963,
        "exp6963-queue-memory-cold-audit",
        "Fresh-process queue-memory safety audit",
        "results/experiment_6963_queue_memory_cold_audit.json",
        [
            _gate(
                "exp6962-queue-regulated-self-learning",
                "queue_learning_run_complete_score",
            )
        ],
    ),
    (
        6964,
        "exp6964-v609-capstone",
        "V609 independent capstone and V610 handoff",
        "results/experiment_6964_v609_capstone.json",
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
EXPECTED_TASK_IDS_BY_SHORT_ID = {
    f"exp{task['number']}": task["task_id"] for task in EXPECTED_TASKS
}
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
    "cascade_risk_rows",
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
    "v609_contract_audit_complete_score",
    "v609_contract_conforms_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
EVALUATION_CHECK_NAMES = {
    "document_milestone",
    "yaml_milestone",
    "document_task_count",
    "yaml_task_count",
    "document_id_order",
    "yaml_id_order",
    "yaml_task_milestones",
    "task_parity",
    "gate_contracts",
    "producer_fields",
    "prompt_contracts",
    "model_contracts",
    "prior_failure_fields",
    "exclusion_manifest",
    "bounded_scopes",
    "cascade_isolation",
}
LINT_NAMES = (
    "roadmap_schema",
    "prior_failure_validation",
    "exclusion_manifest_lint",
    "roadmap_gate_audit",
    "retired_upstream_check",
)


def _principle(field: str) -> str:
    """Give each field a short reason that makes omissions visible."""

    reasons = {
        "field_principles": "Field reasons expose why each recorded value is necessary.",
        "preconditions_checked": "Precondition rows stop missing tools from looking like a completed audit.",
        "inference_substrate": "The substrate states that no language model produced the verdict.",
        "duration_s": "Measured duration distinguishes an executed audit from a copied result.",
        "source_artifact_hashes": "Source hashes bind conclusions to exact contract bytes.",
        "rows": "A flat evidence ledger makes omissions and failed checks visible.",
        "document_task_rows": "Document rows preserve the design source independently.",
        "yaml_task_rows": "YAML rows preserve the executable source independently.",
        "task_parity_rows": "Parity rows expose count, order, identity, and text drift.",
        "gate_contract_rows": "Gate rows prevent dependencies on absent or retired work.",
        "producer_field_rows": "Producer rows prove that each gate can read a declared value.",
        "prompt_contract_rows": "Prompt rows preserve executable instructions and prohibitions.",
        "model_contract_rows": "Model rows keep legacy or invalid tokenizer plans out of headlines.",
        "prior_failure_rows": "Failure rows make repeated methods and changed mechanisms auditable.",
        "exclusion_manifest_rows": "Exclusion rows stop retired work from entering a live dependency path.",
        "lint_command_rows": "Raw command receipts separate contract findings from tool failures.",
        "mutation_rows": "Mutation rows prove that each contract rule can detect a violation.",
        "bounded_scope_rows": "Scope rows keep units, GPU ownership, and wall time finite.",
        "cascade_risk_rows": "Cascade rows prove that advisory scores do not control science tasks.",
        "random_seed": "A fixed seed declares that this deterministic audit has no hidden randomness.",
        "reproducibility_checksum": "A content hash detects later changes to the receipt.",
        "v609_contract_audit_complete_score": "Completion measures checked coverage, not contract success.",
        "v609_contract_conforms_score": "Conformance separates a clean contract from a completed defect report.",
        "gate_check_summary": "Expected and observed values make blocked or failed checks actionable.",
        "verifier_is_oracle": "True limits this oracle to contract conformance, not scientific truth.",
        "verdict_class": "A closed class keeps downstream status handling deterministic.",
        "honest_verdict": "A terminal prefix lets the conductor classify the result without guessing.",
    }
    return reasons.get(field, f"The {field} field binds this receipt to one audit state.")


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
    "openspec_coverage": SPEC_COVERAGE_PATH,
    "docs_audit": DOCS_AUDIT_PATH,
    "artifact_convention_audit": ARTIFACT_AUDIT_PATH,
    "root_clutter_check": ROOT_CLUTTER_PATH,
}
SOURCE_PATHS = {
    **PRECONDITION_PATHS,
    "known_issues": KNOWN_ISSUES_PATH,
    "prior_v608_artifact": PRIOR_ARTIFACT_PATH,
    "spec": SPEC_PATH,
    "module": Path("python/carnot/experiment_6954_v609_contract_advisory.py"),
    "wrapper": Path("scripts/experiments/experiment_6954_v609_contract_advisory.py"),
    "focused_tests": Path("tests/python/test_experiment_6954_v609_contract_advisory.py"),
}


def experiment_number(value: Any) -> int:
    """Extract an experiment number without accepting an unrelated integer."""

    match = re.match(r"^exp(\d+)(?:-|$)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _parse_scalar(text: str) -> Any:
    """Parse one design-table gate value with YAML scalar rules."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list)):
        raise ValueError(f"structured gate value must be scalar: {text}")
    return value


def _parse_design_gates(text: str) -> list[JsonDict]:
    """Turn every semicolon-separated design gate into the YAML gate shape."""

    if text.strip().lower() == "none":
        return []
    gates = []
    for part in text.split(";"):
        part = part.strip().strip("`")
        match = re.fullmatch(
            r"(exp\d+[a-z0-9-]*)\.?\s*`?([a-z][a-z0-9_]*)`?\s*(==|!=|>=|<=|>|<)\s*(.+)",
            part,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"invalid structured gate: {part}")
        upstream = match.group(1)
        gates.append(
            {
                "upstream": EXPECTED_TASK_IDS_BY_SHORT_ID.get(upstream, upstream),
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4)),
            }
        )
    return gates


def parse_design(text: str) -> JsonDict:
    """Parse only the V609 task table without consulting executable YAML."""

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
                "gates": _parse_design_gates(gate_text),
                "milestone": milestone.group(1),
            }
        )
    if not rows:
        raise ValueError("design task table is required")
    return {"milestone": milestone.group(1), "tasks": rows}


def _required_fields_block(prompt: str) -> str:
    """Return only the declared artifact fields before the run command."""

    marker = re.search(r"REQUIRED ARTIFACT FIELDS:", prompt, re.IGNORECASE)
    if not marker:
        return ""
    tail = prompt[marker.start() :]
    return re.split(r"\n\s*Run command:", tail, maxsplit=1, flags=re.IGNORECASE)[0]


def parse_roadmap(document: Any) -> JsonDict:
    """Parse executable values while retaining prompts and prior failures."""

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
    """Read numeric retirement entries from the complete exclusion manifest."""

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
    """Match a full task ID against numeric or full manifest IDs."""

    number = experiment_number(task_id)
    return number >= 0 and (f"exp{number}" in retired_ids or task_id in retired_ids)


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Pair each decision with the values needed to reproduce it."""

    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "terminal": True,
    }


def _field_declared(block: str, field: str) -> bool:
    """Find one complete field name without matching a longer name."""

    return bool(
        re.search(
            rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])",
            block,
            re.IGNORECASE,
        )
    )


def _prompt_row(task: Mapping[str, Any]) -> JsonDict:
    """Check prompt sections, the task-owned command, and exact ending."""

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
    return {
        "task_id": task["task_id"],
        "section_positions": dict(zip(PROMPT_SECTIONS, indices, strict=True)),
        "expected_run_command": expected_command,
        "observed_run_commands": [line.strip() for line in run_commands],
        "push_prohibition_present": "Do NOT push." in prompt,
        "conductor_prohibition_present": (
            "Do NOT modify scripts/research_conductor.py." in prompt
        ),
        "exact_terminal_line": ending_ok,
        "passed": sections_ok and command_ok and ending_ok,
    }


def _gguf_autotokenizer_call(prompt: str) -> bool:
    """Detect an executable tokenizer call that uses a literal GGUF repo ID."""

    return bool(
        re.search(
            r"AutoTokenizer\s*\.\s*from_pretrained\s*\(\s*(['\"])[^'\"]*-GGUF\1",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _model_row(task: Mapping[str, Any]) -> JsonDict:
    """Check current local models without treating warning prose as a call."""

    number = int(task["number"])
    prompt = str(task["prompt"])
    required = MODEL_REQUIREMENTS.get(number, ())
    model_bearing = bool(required)
    present = [model_id for model_id in MANDATED_GGUF_MODELS if model_id in prompt]
    missing = [model_id for model_id in required if model_id not in prompt]
    tokenizer_call = _gguf_autotokenizer_call(prompt)
    marker_present = "MODEL_SPECS" in prompt
    legacy_only = model_bearing and not present
    passed = not tokenizer_call
    if model_bearing:
        passed = passed and marker_present and bool(present) and not missing and not legacy_only
    return {
        "task_id": task["task_id"],
        "model_bearing": model_bearing,
        "model_specs_present": marker_present,
        "required_models": list(required),
        "mandated_models_present": present,
        "missing_required_models": missing,
        "legacy_only_headline": legacy_only,
        "gguf_autotokenizer_call": tokenizer_call,
        "passed": passed,
    }


def _prior_row(task: Mapping[str, Any]) -> JsonDict:
    """Check every prior row and require a real Boolean retirement choice."""

    failures = []
    for index, prior in enumerate(task["prior_failures"]):
        if not isinstance(prior, Mapping):
            failures.append(
                {"index": index, "missing": list(PRIOR_FAILURE_FIELDS), "malformed": True}
            )
            continue
        missing = [field for field in PRIOR_FAILURE_FIELDS if field not in prior]
        empty = [
            field
            for field in PRIOR_FAILURE_FIELDS[:-1]
            if field in prior and not str(prior.get(field) or "").strip()
        ]
        type_errors = []
        if (
            "retire_if_same_verdict" in prior
            and not isinstance(prior["retire_if_same_verdict"], bool)
        ):
            type_errors.append("retire_if_same_verdict")
        if missing or empty or type_errors:
            failures.append(
                {
                    "index": index,
                    "missing": missing,
                    "empty": empty,
                    "type_errors": type_errors,
                    "malformed": False,
                }
            )
    return {
        "task_id": task["task_id"],
        "prior_failure_count": len(task["prior_failures"]),
        "field_failures": failures,
        "passed": not failures,
    }


def _has_checkpoint_ceiling(prompt: str) -> tuple[bool, bool, str | None]:
    """Accept a numeric unit cap or a sealed finite upstream unit roster."""

    compact = " ".join(prompt.split())
    checkpoint = bool(
        re.search(
            r"(?:after each|per[- ](?:row|event|unit|output|call|sequence|attempt)).{0,250}"
            r"checkpoint|checkpoint.{0,250}(?:after each|per[- ](?:row|event|unit|output|"
            r"call|sequence|attempt))",
            compact,
            re.IGNORECASE,
        )
    )
    numeric = bool(
        re.search(
            r"\b(?:exactly|at most|up to|maximum|max)\s+\d+\s+"
            r"(?:attempted\s+)?(?:rows?|outputs?|events?|calls?|sequences?|attempts?|keys?)\b",
            compact,
            re.IGNORECASE,
        )
    )
    sealed_roster = all(
        phrase in compact.lower() for phrase in ("sealed", "sequence", "total attempted calls")
    )
    if numeric:
        boundary = "numeric_unit_ceiling"
    elif sealed_roster:
        boundary = "sealed_upstream_roster_and_fixed_call_total"
    else:
        boundary = None
    return checkpoint, numeric or sealed_roster, boundary


def _scope_row(task: Mapping[str, Any]) -> JsonDict:
    """Check unit rows, verdict shape, GPU ownership, and inclusive wall time."""

    prompt = str(task["prompt"])
    block = str(task["required_fields_block"])
    estimate = task["estimated_wall_time_min"]
    wall_ok = isinstance(estimate, int) and not isinstance(estimate, bool) and 1 <= estimate <= 720
    blocked_ok = bool(
        re.search(
            r"PRECONDITIONS:.*?\bwrite\s+blocked_[a-z0-9_]+\s+with\s+gate_check_summary",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )
    classes_ok = all(re.search(rf"\b{value}\b", prompt) for value in CLOSED_VERDICT_CLASSES)
    prefix_ok = bool(
        re.search(
            r"honest_verdict\s+with\s+a\s+terminal\s+prefix\s+consistent\s+with\s+"
            r"verdict_class",
            prompt,
            re.IGNORECASE,
        )
    )
    checkpoint, ceiling, boundary = _has_checkpoint_ceiling(prompt)
    receipt = _field_declared(block, "task_runtime_receipt")
    gpu_ok = not task["requires_gpu"] or (checkpoint and ceiling and receipt)
    passed = (
        task["per_unit_rows"] and wall_ok and blocked_ok and classes_ok and prefix_ok and gpu_ok
    )
    return {
        "task_id": task["task_id"],
        "requires_gpu": task["requires_gpu"],
        "per_unit_rows": task["per_unit_rows"],
        "task_owned_gpu_receipt": receipt if task["requires_gpu"] else None,
        "per_unit_checkpoint": checkpoint if task["requires_gpu"] else None,
        "finite_unit_ceiling": ceiling if task["requires_gpu"] else None,
        "scope_boundary": boundary if task["requires_gpu"] else None,
        "estimated_wall_time_min": estimate,
        "wall_time_inclusive_1_to_720": wall_ok,
        "terminal_blocked_artifact_contract": blocked_ok,
        "verdict_classes_present": classes_ok,
        "terminal_prefix_contract_present": prefix_ok,
        "passed": passed,
    }


def _cascade_rows(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Show each gate and prove that advisory scores do not control science."""

    rows = []
    all_gates = [
        (task["task_id"], gate) for task in tasks for gate in task.get("gates", [])
    ]
    for task_id, gate in all_gates:
        upstream = str(gate.get("upstream") or "")
        field = str(gate.get("artifact_field") or "")
        advisory_dependency = upstream in ADVISORY_TASK_IDS or field in ADVISORY_SCORE_FIELDS
        rows.append(
            {
                "row_kind": "gate",
                "task_id": task_id,
                "upstream": upstream,
                "artifact_field": field,
                "advisory_dependency": advisory_dependency,
                "passed": not advisory_dependency,
            }
        )
    for advisory_id in sorted(ADVISORY_TASK_IDS):
        consumers = [
            task_id
            for task_id, gate in all_gates
            if gate.get("upstream") == advisory_id
            or str(gate.get("artifact_field") or "") in ADVISORY_SCORE_FIELDS
        ]
        rows.append(
            {
                "row_kind": "advisory_isolation",
                "advisory_task_id": advisory_id,
                "science_consumers": consumers,
                "passed": not consumers,
            }
        )
    return rows


def evaluate_contract(design_text: str, roadmap_document: Any, retired_ids: set[str]) -> JsonDict:
    """Evaluate both sources without invoking an external command."""

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
            upstream_id = str(gate.get("upstream") or "")
            upstream = by_id.get(upstream_id)
            retired = _is_retired(upstream_id, retired_ids)
            precedes = upstream is not None and positions[upstream["task_id"]] < task["order"]
            advisory = upstream_id in ADVISORY_TASK_IDS or str(
                gate.get("artifact_field") or ""
            ) in ADVISORY_SCORE_FIELDS
            gate_passed = upstream is not None and precedes and not retired and not advisory
            gate_rows.append(
                {
                    "task_id": task["task_id"],
                    "gate": gate,
                    "upstream_exists": upstream is not None,
                    "upstream_precedes_consumer": precedes,
                    "upstream_retired": retired,
                    "advisory_global_dependency": advisory,
                    "passed": gate_passed,
                }
            )
            field = str(gate.get("artifact_field") or "")
            declared = bool(upstream and _field_declared(upstream["required_fields_block"], field))
            producer_rows.append(
                {
                    "task_id": task["task_id"],
                    "upstream": upstream_id,
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
    cascade_rows = _cascade_rows(yaml_rows)
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
        _check(
            "yaml_task_milestones",
            MILESTONE,
            [row["milestone"] for row in yaml_rows],
            all(row["milestone"] == MILESTONE for row in yaml_rows),
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
        ("cascade_isolation", cascade_rows),
    ):
        checks.append(
            _check(
                name,
                "all rows pass",
                sum(row["passed"] for row in rows),
                bool(rows) and all(row["passed"] for row in rows),
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
        "cascade_risk_rows": cascade_rows,
    }


def apply_mutation(roadmap_document: JsonDict, mutation: str) -> None:
    """Apply one named defect so tests and artifact rows share mutations."""

    tasks = roadmap_document["tasks"]
    if mutation == "task_count":
        tasks.pop()
    elif mutation == "id_order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "title":
        tasks[2]["title"] = f"{tasks[2]['title']} changed"
    elif mutation == "deliverable":
        tasks[3]["deliverable"] = "results/mutated_deliverable.json"
    elif mutation == "gate_field":
        tasks[3]["gated_on"][0]["artifact_field"] = "mutated_ready_score"
    elif mutation == "producer_field":
        gate = tasks[3]["gated_on"][0]
        producer = next(task for task in tasks if task["id"] == gate["upstream"])
        producer["prompt"] = producer["prompt"].replace(
            gate["artifact_field"], "mutated_producer_output"
        )
    elif mutation == "model_specs":
        tasks[3]["prompt"] = tasks[3]["prompt"].replace("MODEL_SPECS", "MODEL PLAN")
    elif mutation.startswith("prior_failure_"):
        field = mutation.removeprefix("prior_failure_")
        task = next(task for task in tasks if task.get("prior_failures"))
        task["prior_failures"][0].pop(field, None)
    elif mutation == "prompt_ending":
        tasks[0]["prompt"] = tasks[0]["prompt"].replace(FINAL_PROHIBITIONS, "Do NOT push.")
    else:
        raise ValueError(f"unknown mutation: {mutation}")


def build_mutation_rows(
    design_text: str, roadmap_document: Mapping[str, Any], retired_ids: set[str]
) -> list[JsonDict]:
    """Change each required dimension and prove that the evaluator rejects it."""

    rows = []
    baseline = json.dumps(roadmap_document, sort_keys=True, default=str)
    for mutation in REQUIRED_MUTATIONS:
        changed = deepcopy(roadmap_document)
        apply_mutation(changed, mutation)
        changed_text = json.dumps(changed, sort_keys=True, default=str)
        try:
            result = evaluate_contract(design_text, changed, retired_ids)
            contract_passed = result["passed"]
            failed = [check["check"] for check in result["checks"] if not check["passed"]]
            error = None
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            contract_passed = False
            failed = ["contract_parse"]
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "mutation": mutation,
                "input_changed": changed_text != baseline,
                "contract_passed": contract_passed,
                "failed_checks": failed,
                "error": error,
                "terminal": True,
                "failed_as_expected": (
                    changed_text != baseline and not contract_passed and bool(failed)
                ),
            }
        )
    return rows


def _run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one guard and retain output that explains any nonzero result."""

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
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "exit_code": 124,
            "stdout": str(exc.output or ""),
            "stderr": str(exc.stderr or ""),
            "outcome": "tool_failure",
            "terminal": False,
            "passed": False,
        }
    except OSError as exc:
        return {
            "name": name,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "outcome": "tool_failure",
            "terminal": False,
            "passed": False,
        }
    passed = result.returncode == 0
    return {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "outcome": "pass" if passed else "contract_defect",
        "terminal": True,
        "passed": passed,
    }


def run_lint_commands(root: Path, retired_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run the four repository guards and one deterministic retirement check."""

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
    passed = not violations
    rows.append(
        {
            "name": "retired_upstream_check",
            "command": (
                "internal retired_upstream_check research-roadmap.yaml "
                "ops/exclusion_manifest.yaml"
            ),
            "exit_code": 0 if passed else 1,
            "stdout": json.dumps({"retired_upstream_tasks": violations}, sort_keys=True),
            "stderr": "",
            "outcome": "pass" if passed else "contract_defect",
            "terminal": True,
            "passed": passed,
        }
    )
    return rows


def _sha256(path: Path) -> str:
    """Hash source bytes without normalizing whitespace."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the audit receipt while excluding duration and this hash."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _flat_rows(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Tag grouped rows so the combined ledger stays inspectable."""

    return [{"row_type": name, **dict(row)} for name, rows in groups.items() for row in rows]


def _score_state(artifact: Mapping[str, Any]) -> tuple[int, int]:
    """Recompute audit completion and conformance from terminal evidence rows."""

    summary = artifact.get("gate_check_summary")
    checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
    by_name = {
        str(row.get("check")): row for row in checks if isinstance(row, Mapping) and row.get("check")
    }
    lint_rows = artifact.get("lint_command_rows")
    mutation_rows = artifact.get("mutation_rows")
    lint_complete = (
        isinstance(lint_rows, list)
        and [row.get("name") for row in lint_rows] == list(LINT_NAMES)
        and all(row.get("terminal") is True for row in lint_rows)
    )
    mutations_complete = (
        isinstance(mutation_rows, list)
        and [row.get("mutation") for row in mutation_rows] == list(REQUIRED_MUTATIONS)
        and all(
            row.get("terminal") is True and row.get("input_changed") is True
            for row in mutation_rows
        )
    )
    evaluation_complete = EVALUATION_CHECK_NAMES <= by_name.keys()
    inputs_complete = all(
        by_name.get(name, {}).get("passed") is True for name in ("preconditions", "contract_parse")
    )
    complete = int(inputs_complete and evaluation_complete and lint_complete and mutations_complete)
    evaluation_passed = evaluation_complete and all(
        by_name[name].get("passed") is True for name in EVALUATION_CHECK_NAMES
    )
    lints_passed = lint_complete and all(row.get("passed") is True for row in lint_rows)
    mutations_passed = mutations_complete and all(
        row.get("failed_as_expected") is True for row in mutation_rows
    )
    conforms = int(bool(complete) and evaluation_passed and lints_passed and mutations_passed)
    return complete, conforms


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Read inputs, run guards, and preserve either a terminal audit or block."""

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
            field: []
            for field in ROW_FIELDS
            if field not in {"lint_command_rows", "mutation_rows"}
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
            parse_error or ("parsed" if preconditions_ok else "not attempted"),
            preconditions_ok and not parse_error,
        ),
        *evaluation["checks"],
        _check(
            "lint_commands_terminal",
            list(LINT_NAMES),
            {row["name"]: row["outcome"] for row in lint_rows},
            [row["name"] for row in lint_rows] == list(LINT_NAMES)
            and all(row["terminal"] for row in lint_rows),
        ),
        _check(
            "lint_commands_pass",
            "all completed commands report no contract defect",
            {row["name"]: row["exit_code"] for row in lint_rows},
            bool(lint_rows) and all(row["passed"] for row in lint_rows),
        ),
        _check(
            "required_mutations",
            list(REQUIRED_MUTATIONS),
            {row["mutation"]: row["failed_as_expected"] for row in mutation_rows},
            [row["mutation"] for row in mutation_rows] == list(REQUIRED_MUTATIONS)
            and all(row["failed_as_expected"] for row in mutation_rows),
        ),
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
        "schema": "carnot.v609_contract_advisory.v1",
        "experiment_id": 6954,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": source_hashes,
        "rows": _flat_rows(groups),
        **groups,
        "random_seed": RANDOM_SEED,
        "v609_contract_audit_complete_score": 0,
        "v609_contract_conforms_score": 0,
        "gate_check_summary": {
            "audit_complete": False,
            "passed": False,
            "failed_check": None,
            "expected": None,
            "observed": None,
            "failures": [],
            "checks": checks,
        },
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    complete, conforms = _score_state(artifact)
    failures = [
        {"failed_check": row["check"], "expected": row["expected"], "observed": row["observed"]}
        for row in checks
        if not row["passed"]
    ]
    artifact["v609_contract_audit_complete_score"] = complete
    artifact["v609_contract_conforms_score"] = conforms
    artifact["gate_check_summary"].update(
        {
            "audit_complete": bool(complete),
            "passed": bool(conforms),
            "failed_check": failures[0]["failed_check"] if failures else None,
            "expected": failures[0]["expected"] if failures else "all checks pass",
            "observed": failures[0]["observed"] if failures else "all checks pass",
            "failures": failures,
        }
    )
    if complete and conforms:
        artifact.update(
            status="complete",
            verdict_class="circular_positive",
            honest_verdict=CONFORMS_VERDICT,
        )
    elif complete:
        artifact.update(status="complete", verdict_class="null", honest_verdict=DEFECT_VERDICT)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute both scores and reject a forged terminal representation."""

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
    mutations = artifact.get("mutation_rows")
    if not isinstance(mutations, list) or [row.get("mutation") for row in mutations] not in (
        [],
        list(REQUIRED_MUTATIONS),
    ):
        errors.append("mutation_rows_mismatch")
    expected_complete, expected_conforms = _score_state(artifact)
    if artifact.get("v609_contract_audit_complete_score") != expected_complete:
        errors.append("audit_complete_score_mismatch")
    if artifact.get("v609_contract_conforms_score") != expected_conforms:
        errors.append("conforms_score_mismatch")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        errors.append("gate_check_summary_invalid")
    elif (
        summary.get("audit_complete") is not bool(expected_complete)
        or summary.get("passed") is not bool(expected_conforms)
    ):
        errors.append("gate_check_summary_score_mismatch")
    if expected_complete and expected_conforms:
        if (
            artifact.get("status") != "complete"
            or artifact.get("verdict_class") != "circular_positive"
            or artifact.get("honest_verdict") != CONFORMS_VERDICT
        ):
            errors.append("conforming_terminal_shape_mismatch")
    elif expected_complete:
        if (
            artifact.get("status") != "complete"
            or artifact.get("verdict_class") != "null"
            or artifact.get("honest_verdict") != DEFECT_VERDICT
        ):
            errors.append("advisory_null_terminal_shape_mismatch")
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
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _date_argument(value: str) -> str:
    """Reject ambiguous dates before any artifact write."""

    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the target only after a complete JSON write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the advisory audit and write one validated terminal artifact."""

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
