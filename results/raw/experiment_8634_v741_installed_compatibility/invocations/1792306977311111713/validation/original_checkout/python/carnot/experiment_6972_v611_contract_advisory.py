"""Build the deterministic V611 document and YAML contract audit.

The two contract files are parsed separately. The audit reports defects, but
its scores never control a scientific task. See REQ-REPORT-6972.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
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
OUTPUT_PATH = Path("results/experiment_6972_v611_contract_advisory.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
VERDICT_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
ARC_LIVE_PATH_LINT = Path("scripts/arc_llm_on_liveness_lint.py")
PRIOR_AUDIT_PATH = Path("results/experiment_6965_v610_contract_advisory.json")
MILESTONE = "2026.09.611"
EXPECTED_NUMBERS = tuple(range(6972, 6984))
FINAL_PROHIBITIONS = "Do NOT push. Do NOT modify scripts/research_conductor.py."
INFERENCE_SUBSTRATE = "deterministic_document_yaml_contract_audit"
RANDOM_SEED = 6972
BLOCKED_VERDICT = "blocked_v611_contract_advisory"
CONFORMS_VERDICT = "complete_circular_positive_v611_contract_conforms"
DEFECT_VERDICT = "complete_null_v611_contract_defects_found"
PROMPT_SECTIONS = ("CONTEXT:", "EXISTING CODE TO READ FIRST:", "TASK:", "CONCRETE STEPS:")
MANDATED_GGUF_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-12B-it-GGUF",
)
ARC_FALSE_FIELDS = (
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
)
PRIOR_FAILURE_FIELDS = (
    "experiment_id",
    "verdict",
    "addressed_by",
    "retire_if_same_verdict",
)
ADVISORY_TASK_IDS = {"exp6972-v611-contract-advisory"}
ADVISORY_SCORE_FIELDS = {"v611_contract_audit_complete_score", "contract_conforms_score"}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_MUTATIONS = (
    "task_count",
    "id_order",
    "title",
    "deliverable",
    "gate_field",
    "producer_field",
    "model_id",
    "prior_failure_experiment_id",
    "prior_failure_verdict",
    "prior_failure_addressed_by",
    "prior_failure_retire_if_same_verdict",
    "arc_solve_claim",
    "prompt_ending",
)
COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure_validation",
    "exclusion_manifest_lint",
    "roadmap_gate_audit",
    "verdict_row_consistency_lint",
    "retired_upstream_check",
    "arc_live_path_lint",
    "prompt_ending_check",
)
PRECONDITION_PATHS = {
    "active_roadmap": ACTIVE_ROADMAP_PATH,
    "design_document": DESIGN_PATH,
    "roadmap_schema": SCHEMA_PATH,
    "full_exclusion_manifest": EXCLUSION_PATH,
    "prior_failure_validation": PRIOR_VALIDATOR_PATH,
    "exclusion_manifest_lint": EXCLUSION_LINT_PATH,
    "roadmap_gate_audit": GATE_AUDIT_PATH,
    "verdict_row_consistency_lint": VERDICT_LINT_PATH,
    "arc_live_path_lint": ARC_LIVE_PATH_LINT,
}
SOURCE_PATHS = {
    **PRECONDITION_PATHS,
    "prior_v610_contract_audit": PRIOR_AUDIT_PATH,
    "spec": SPEC_PATH,
    "module": Path("python/carnot/experiment_6972_v611_contract_advisory.py"),
    "wrapper": Path("scripts/experiments/experiment_6972_v611_contract_advisory.py"),
    "focused_tests": Path("tests/python/test_experiment_6972_v611_contract_advisory.py"),
}
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
    "task_contract_rows",
    "gate_rows",
    "producer_field_rows",
    "model_contract_rows",
    "prior_failure_rows",
    "retired_scope_rows",
    "arc_solve_rule_rows",
    "prompt_ending_rows",
    "command_receipt_rows",
    "v611_contract_audit_complete_score",
    "contract_conforms_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}


def _principle(field: str) -> str:
    """State why one field is necessary scientific evidence."""

    reasons = {
        "field_principles": "Field reasons make each required value accountable to the audit method.",
        "preconditions_checked": "Precondition rows prevent missing tools from looking like a completed audit.",
        "inference_substrate": "The substrate limits the result to deterministic contract evidence.",
        "duration_s": "Measured duration distinguishes execution from a copied conclusion.",
        "source_artifact_hashes": "Hashes bind each finding to exact source bytes.",
        "rows": "The evidence ledger exposes every aggregate check and mutation.",
        "task_contract_rows": "Task rows preserve identity, order, title, deliverable, and gate parity.",
        "gate_rows": "Gate rows expose missing, later, retired, or advisory dependencies.",
        "producer_field_rows": "Producer rows prove each gate can read a declared field.",
        "model_contract_rows": "Model rows prevent invalid or legacy-only GGUF plans.",
        "prior_failure_rows": "Prior rows preserve the changed mechanism and retirement choice.",
        "retired_scope_rows": "Retirement rows keep closed work outside the active graph.",
        "arc_solve_rule_rows": "ARC rows prevent audit evidence from becoming solve credit.",
        "prompt_ending_rows": "Prompt rows detect missing sections, commands, and prohibitions.",
        "command_receipt_rows": "Raw receipts separate a contract defect from a tool failure.",
        "v611_contract_audit_complete_score": "Completion measures terminal evidence, not contract success.",
        "contract_conforms_score": "Conformance records whether every independent check passed.",
        "random_seed": "A fixed seed declares that the audit makes no sampling choice.",
        "reproducibility_checksum": "A checksum exposes later changes to the receipt.",
        "gate_check_summary": "Expected and observed values make each failure actionable.",
        "verifier_is_oracle": "The oracle flag applies only to document and YAML facts.",
        "verdict_class": "A closed class gives downstream readers one stable state.",
        "honest_verdict": "A terminal prefix states whether the audit completed or blocked.",
    }
    return reasons.get(field, f"The {field} field identifies one reproducible audit fact.")


FIELD_PRINCIPLES = {field: _principle(field) for field in REQUIRED_ARTIFACT_FIELDS}


def experiment_number(value: Any) -> int:
    """Extract a leading experiment number from a task identifier."""

    match = re.match(r"^exp(\d+)(?:-|$)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _parse_scalar(text: str) -> Any:
    """Parse a prerequisite value while refusing nested YAML values."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list)):
        raise ValueError(f"structured prerequisite value must be scalar: {text}")
    return value


def _parse_design_prerequisites(text: str, full_ids: Mapping[str, str]) -> list[JsonDict]:
    """Convert one Markdown prerequisite cell to executable gate values."""

    if text.strip().lower() == "none":
        return []
    gates = []
    for part in text.split(";"):
        candidate = part.strip()
        match = re.fullmatch(
            r"(exp\d+[a-z0-9-]*)\s+`([a-z][a-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(.+)`",
            candidate,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"invalid structured prerequisite: {candidate}")
        short_id = f"exp{experiment_number(match.group(1))}"
        gates.append(
            {
                "upstream": full_ids.get(short_id, match.group(1)),
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4)),
            }
        )
    return gates


def parse_design(text: str) -> JsonDict:
    """Parse only the milestone and Exact task contract Markdown table."""

    milestone = re.search(r"^\*\*Milestone:\*\*\s*`?([^`\s]+)`?\s*$", text, re.MULTILINE)
    if not milestone:
        raise ValueError("design milestone is required")
    section = re.search(
        r"^## Exact task contract\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if not section:
        raise ValueError("design task table is required")
    raw_rows = []
    for line in section.group("body").splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError(f"malformed design task row: {line}")
        order, task_id, title, deliverable, prerequisites = cells
        raw_rows.append(
            {
                "order": int(order),
                "number": experiment_number(task_id),
                "task_id": task_id.strip("`"),
                "title": title,
                "deliverable": deliverable.strip("`"),
                "prerequisites": prerequisites,
            }
        )
    if not raw_rows:
        raise ValueError("design task table is required")
    full_ids = {f"exp{row['number']}": str(row["task_id"]) for row in raw_rows}
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        row["gates"] = _parse_design_prerequisites(str(row.pop("prerequisites")), full_ids)
        row["milestone"] = milestone.group(1)
        rows.append(row)
    return {"milestone": milestone.group(1), "tasks": rows}


def _required_fields_block(prompt: str) -> str:
    """Return only the task's required artifact field declaration."""

    marker = re.search(r"REQUIRED ARTIFACT FIELDS:", prompt, re.IGNORECASE)
    if not marker:
        return ""
    tail = prompt[marker.start() :]
    return re.split(r"\n\s*Run command:", tail, maxsplit=1, flags=re.IGNORECASE)[0]


def parse_roadmap(document: Any) -> JsonDict:
    """Parse executable YAML fields without consulting the Markdown source."""

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
                "track": str(task.get("track") or ""),
                "agent_type": str(task.get("agent_type") or ""),
                "model": str(task.get("model") or ""),
                "requires_gpu": task.get("requires_gpu") is True,
                "gates": [dict(gate) if isinstance(gate, Mapping) else {} for gate in gates],
                "prior_failures": deepcopy(priors),
                "prompt": prompt,
                "required_fields_block": _required_fields_block(prompt),
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def _canonical_exp_id(value: Any) -> str | None:
    """Normalize integer and full identifiers to an experiment number key."""

    if isinstance(value, int) and not isinstance(value, bool):
        return f"exp{value}"
    number = experiment_number(value)
    return f"exp{number}" if number >= 0 else None


def retired_experiment_ids(document: Any) -> set[str]:
    """Read all retired IDs and honor explicit un-retirement entries."""

    if not isinstance(document, Mapping):
        return set()
    retired: set[str] = set()
    restored: set[str] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        entries = document.get(section, []) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            ids = [entry.get("experiment_id")]
            extra_ids = entry.get("experiment_ids", []) or []
            if isinstance(extra_ids, list):
                ids.extend(extra_ids)
            unretired = entry.get("un_retired_experiment_ids", []) or []
            if not isinstance(unretired, list):
                unretired = []
            retired.update(filter(None, (_canonical_exp_id(value) for value in ids)))
            restored.update(filter(None, (_canonical_exp_id(value) for value in unretired)))
    return retired - restored


def _is_retired(value: Any, retired_ids: set[str]) -> bool:
    """Check one identifier against normalized and full manifest values."""

    key = _canonical_exp_id(value)
    return str(value) in retired_ids or bool(key and key in retired_ids)


def _field_declared(block: str, field: str) -> bool:
    """Match a complete field name inside one required-fields block."""

    return bool(field and re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", block))


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one terminal aggregate contract check."""

    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "terminal": True,
    }


def _task_contract_rows(document_rows: list[JsonDict], yaml_rows: list[JsonDict]) -> list[JsonDict]:
    """Compare each YAML row with the independent design-table row."""

    rows = []
    for position, (design, active) in enumerate(zip_longest(document_rows, yaml_rows), 1):
        expected = design or {}
        observed = active or {}
        fields = {
            "order": (expected.get("order"), observed.get("order")),
            "task_id": (expected.get("task_id"), observed.get("task_id")),
            "title": (expected.get("title"), observed.get("title")),
            "deliverable": (expected.get("deliverable"), observed.get("deliverable")),
            "gates": (expected.get("gates"), observed.get("gates")),
        }
        matches = {name: left == right for name, (left, right) in fields.items()}
        rows.append(
            {
                "position": position,
                "document_present": design is not None,
                "yaml_present": active is not None,
                "expected": {name: values[0] for name, values in fields.items()},
                "observed": {name: values[1] for name, values in fields.items()},
                "field_matches": matches,
                "passed": design is not None and active is not None and all(matches.values()),
                "terminal": True,
            }
        )
    return rows


def _gate_and_producer_rows(
    document_rows: list[JsonDict], yaml_rows: list[JsonDict], retired_ids: set[str]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check exact gate parity and resolve each gate to an earlier producer."""

    by_id = {str(task["task_id"]): task for task in yaml_rows}
    positions = {str(task["task_id"]): int(task["order"]) for task in yaml_rows}
    gate_rows = []
    producer_rows = []
    for design in document_rows:
        consumer = by_id.get(str(design["task_id"]))
        observed_gates = consumer.get("gates", []) if consumer else []
        for gate_index, (expected, observed) in enumerate(
            zip_longest(design["gates"], observed_gates), 1
        ):
            gate = observed or expected or {}
            upstream_id = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            producer = by_id.get(upstream_id)
            precedes = bool(
                consumer
                and producer
                and positions[upstream_id] < positions[str(consumer["task_id"])]
            )
            retired = _is_retired(upstream_id, retired_ids)
            advisory = upstream_id in ADVISORY_TASK_IDS or field in ADVISORY_SCORE_FIELDS
            matches = expected is not None and observed == expected
            passed = bool(consumer and producer and matches and precedes and not retired and not advisory)
            gate_rows.append(
                {
                    "task_id": design["task_id"],
                    "gate_index": gate_index,
                    "expected_gate": expected,
                    "observed_gate": observed,
                    "gate_matches": matches,
                    "upstream_exists": producer is not None,
                    "upstream_precedes_consumer": precedes,
                    "upstream_retired": retired,
                    "advisory_dependency": advisory,
                    "passed": passed,
                    "terminal": True,
                }
            )
            declared = bool(producer and _field_declared(producer["required_fields_block"], field))
            producer_rows.append(
                {
                    "task_id": design["task_id"],
                    "gate_index": gate_index,
                    "upstream": upstream_id,
                    "artifact_field": field,
                    "producer_present": producer is not None,
                    "producer_required_fields_block_present": bool(
                        producer and producer["required_fields_block"]
                    ),
                    "field_declared": declared,
                    "passed": bool(consumer and producer and matches and declared),
                    "terminal": True,
                }
            )
    return gate_rows, producer_rows


def _gguf_autotokenizer_call(prompt: str) -> bool:
    """Detect an executable AutoTokenizer call on a GGUF repository."""

    return bool(
        re.search(
            r"AutoTokenizer\s*\.\s*from_pretrained\s*\(\s*(['\"])[^'\"]*-GGUF\1",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _model_row(task: JsonDict) -> JsonDict:
    """Check SOTA GGUF use and the declared execution-agent pairing."""

    prompt = task["prompt"]
    model_bearing = bool(task["requires_gpu"] or "live_local_" in prompt)
    present = [model_id for model_id in MANDATED_GGUF_MODELS if model_id in prompt]
    legacy = bool(re.search(r"Qwen3\.5|0\.8B|E4B|legacy", prompt, re.IGNORECASE))
    legacy_only = model_bearing and legacy and not present
    agent = task["agent_type"].lower()
    model = task["model"]
    agent_model_ok = (agent == "claude" and model in {"sonnet", "opus"}) or (
        agent == "codex" and model == "gpt-5.6-sol"
    )
    model_plan_ok = not model_bearing or ("MODEL_SPECS" in prompt and bool(present))
    tokenizer_call = _gguf_autotokenizer_call(prompt)
    return {
        "task_id": task["task_id"],
        "model_bearing": model_bearing,
        "model_specs_present": "MODEL_SPECS" in prompt,
        "mandated_models_present": present,
        "legacy_only_headline": legacy_only,
        "gguf_autotokenizer_call": tokenizer_call,
        "agent_type": agent,
        "model": model,
        "agent_model_coherent": agent_model_ok,
        "passed": bool(agent_model_ok and model_plan_ok and not legacy_only and not tokenizer_call),
        "terminal": True,
    }


def _prior_row(task: JsonDict) -> JsonDict:
    """Require every declared prior-failure subfield and its correct type."""

    failures = []
    for index, prior in enumerate(task["prior_failures"]):
        if not isinstance(prior, Mapping):
            failures.append({"index": index, "malformed": True, "missing": list(PRIOR_FAILURE_FIELDS)})
            continue
        missing = [field for field in PRIOR_FAILURE_FIELDS if field not in prior]
        empty = [
            field
            for field in PRIOR_FAILURE_FIELDS[:-1]
            if field in prior and not str(prior.get(field) or "").strip()
        ]
        type_errors = []
        if "retire_if_same_verdict" in prior and not isinstance(
            prior["retire_if_same_verdict"], bool
        ):
            type_errors.append("retire_if_same_verdict")
        if missing or empty or type_errors:
            failures.append(
                {
                    "index": index,
                    "malformed": False,
                    "missing": missing,
                    "empty": empty,
                    "type_errors": type_errors,
                }
            )
    return {
        "task_id": task["task_id"],
        "prior_failure_count": len(task["prior_failures"]),
        "field_failures": failures,
        "passed": not failures,
        "terminal": True,
    }


def _arc_row(task: JsonDict) -> JsonDict:
    """Require explicit false solve fields and no registry modification."""

    prompt = task["prompt"]
    block = task["required_fields_block"]
    false_fields = {
        field: bool(
            re.search(rf"\b{re.escape(field)}\s*(?:=|\()\s*false\b", prompt, re.IGNORECASE)
            and _field_declared(block, field)
        )
        for field in ARC_FALSE_FIELDS
    }
    positive_claim = bool(
        re.search(
            r"\b(?:solve_claimed|level_claimed|registry_updated|submitted_to_leaderboard)"
            r"\s*(?:=|\()\s*true\b",
            prompt,
            re.IGNORECASE,
        )
    )
    no_solve_claim = bool(
        re.search(r"(?:does not|do not|without).{0,90}(?:claim|banked).{0,30}solve", prompt, re.I | re.S)
    )
    no_registry_update = bool(
        re.search(r"do not modify\s+ops/arc_solve_registry\.yaml", prompt, re.I | re.S)
    )
    no_source_or_replacement = bool(
        re.search(r"does not generate an engine, inspect game\s+source", prompt, re.I | re.S)
    )
    passed = bool(
        all(false_fields.values())
        and not positive_claim
        and no_solve_claim
        and no_registry_update
        and no_source_or_replacement
    )
    return {
        "task_id": task["task_id"],
        "required_false_fields": false_fields,
        "positive_solve_claim": positive_claim,
        "solve_claim_prohibited": no_solve_claim,
        "registry_update_prohibited": no_registry_update,
        "replacement_and_source_use_prohibited": no_source_or_replacement,
        "passed": passed,
        "terminal": True,
    }


def _prompt_row(task: JsonDict) -> JsonDict:
    """Check section order, one task command, blocked path, and exact ending."""

    prompt = task["prompt"]
    indices = [prompt.find(section) for section in PROMPT_SECTIONS]
    sections_ok = all(index >= 0 for index in indices) and indices == sorted(indices)
    expected_command = (
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        f"{Path(task['deliverable']).stem}.py --date {{date}}"
    )
    commands = re.findall(r"^\s*Run command:.*$", prompt, re.MULTILINE)
    command_ok = len(commands) == 1 and commands[0].strip() == expected_command
    ending_ok = prompt.rstrip().endswith(FINAL_PROHIBITIONS)
    blocked_ok = bool(
        re.search(
            r"PRECONDITIONS:.*?\b(?:write|emit|yields?)\s+blocked_[a-z0-9_]+\s+with\s+gate_check_summary",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )
    positive_partial = False
    for match in re.finditer(r"\bpartial\b", prompt, re.IGNORECASE):
        context = prompt[max(0, match.start() - 80) : match.start()]
        has_instruction = bool(
            re.search(r"\b(?:declare|emit|write|treat|record|label)\b", context, re.IGNORECASE)
        )
        negated = bool(re.search(r"\b(?:do not|never|not)\b.{0,70}$", context, re.IGNORECASE))
        if has_instruction and not negated:
            positive_partial = True
            break
    partial_safe = blocked_ok and not positive_partial
    return {
        "task_id": task["task_id"],
        "section_positions": dict(zip(PROMPT_SECTIONS, indices, strict=True)),
        "expected_run_command": expected_command,
        "observed_run_commands": [line.strip() for line in commands],
        "blocked_precondition_artifact": blocked_ok,
        "external_absence_blocked_not_partial": partial_safe,
        "push_prohibition_present": "Do NOT push." in prompt,
        "conductor_prohibition_present": "Do NOT modify scripts/research_conductor.py." in prompt,
        "exact_terminal_line": ending_ok,
        "passed": bool(sections_ok and command_ok and blocked_ok and partial_safe and ending_ok),
        "terminal": True,
    }


def evaluate_contract(design_text: str, roadmap_document: Any, retired_ids: set[str]) -> JsonDict:
    """Evaluate both contract sources and retain every row-level defect."""

    design = parse_design(design_text)
    roadmap = parse_roadmap(roadmap_document)
    document_rows = design["tasks"]
    yaml_rows = roadmap["tasks"]
    task_rows = _task_contract_rows(document_rows, yaml_rows)
    gate_rows, producer_rows = _gate_and_producer_rows(document_rows, yaml_rows, retired_ids)
    model_rows = [_model_row(task) for task in yaml_rows]
    prior_rows = [_prior_row(task) for task in yaml_rows]
    prompt_rows = [_prompt_row(task) for task in yaml_rows]
    arc_rows = [_arc_row(task) for task in yaml_rows if task["track"].lower() == "arc"]
    retired_rows = []
    for task in yaml_rows:
        retired_upstreams = [
            str(gate.get("upstream") or "")
            for gate in task["gates"]
            if _is_retired(gate.get("upstream"), retired_ids)
        ]
        task_retired = _is_retired(task["task_id"], retired_ids)
        retired_rows.append(
            {
                "task_id": task["task_id"],
                "task_id_retired": task_retired,
                "retired_gate_upstreams": retired_upstreams,
                "passed": not task_retired and not retired_upstreams,
                "terminal": True,
            }
        )
    expected_numbers = list(EXPECTED_NUMBERS)
    checks = [
        _check("document_milestone", MILESTONE, design["milestone"], design["milestone"] == MILESTONE),
        _check("yaml_milestone", MILESTONE, roadmap["milestone"], roadmap["milestone"] == MILESTONE),
        _check("document_task_count", 12, len(document_rows), len(document_rows) == 12),
        _check("yaml_task_count", 12, len(yaml_rows), len(yaml_rows) == 12),
        _check(
            "document_id_order",
            expected_numbers,
            [row["number"] for row in document_rows],
            [row["number"] for row in document_rows] == expected_numbers,
        ),
        _check(
            "yaml_id_order",
            expected_numbers,
            [row["number"] for row in yaml_rows],
            [row["number"] for row in yaml_rows] == expected_numbers,
        ),
        _check(
            "document_declared_order",
            list(range(1, 13)),
            [row["order"] for row in document_rows],
            [row["order"] for row in document_rows] == list(range(1, 13)),
        ),
        _check(
            "yaml_task_milestones",
            MILESTONE,
            [row["milestone"] for row in yaml_rows],
            bool(yaml_rows) and all(row["milestone"] == MILESTONE for row in yaml_rows),
        ),
    ]
    for name, rows in (
        ("task_contracts", task_rows),
        ("gate_contracts", gate_rows),
        ("producer_fields", producer_rows),
        ("model_contracts", model_rows),
        ("prior_failure_contracts", prior_rows),
        ("retired_scopes", retired_rows),
        ("arc_solve_rules", arc_rows),
        ("prompt_endings", prompt_rows),
    ):
        checks.append(
            _check(
                name,
                "all rows pass",
                {"passed": sum(bool(row["passed"]) for row in rows), "total": len(rows)},
                bool(rows) and all(bool(row["passed"]) for row in rows),
            )
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "task_contract_rows": task_rows,
        "gate_rows": gate_rows,
        "producer_field_rows": producer_rows,
        "model_contract_rows": model_rows,
        "prior_failure_rows": prior_rows,
        "retired_scope_rows": retired_rows,
        "arc_solve_rule_rows": arc_rows,
        "prompt_ending_rows": prompt_rows,
    }


def apply_mutation(roadmap_document: JsonDict, mutation: str) -> None:
    """Inject one named defect for the required mutation proof."""

    tasks = roadmap_document["tasks"]
    if mutation == "task_count":
        tasks.pop()
    elif mutation == "id_order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "title":
        tasks[2]["title"] += " changed"
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
    elif mutation == "model_id":
        model_task = next(task for task in tasks if task.get("requires_gpu") is True)
        for model_id in MANDATED_GGUF_MODELS:
            model_task["prompt"] = model_task["prompt"].replace(model_id, "Qwen3.5-0.8B")
    elif mutation.startswith("prior_failure_"):
        field = mutation.removeprefix("prior_failure_")
        task = next(task for task in tasks if task.get("prior_failures"))
        task["prior_failures"][0].pop(field, None)
    elif mutation == "arc_solve_claim":
        task = next(task for task in tasks if str(task.get("track")).lower() == "arc")
        task["prompt"] = task["prompt"].replace("solve_claimed=false", "solve_claimed=true")
        task["prompt"] = task["prompt"].replace("solve_claimed (false)", "solve_claimed (true)")
    elif mutation == "prompt_ending":
        tasks[0]["prompt"] = tasks[0]["prompt"].replace(FINAL_PROHIBITIONS, "Do NOT push.")
    else:
        raise ValueError(f"unknown mutation: {mutation}")


def build_mutation_rows(
    design_text: str, roadmap_document: Mapping[str, Any], retired_ids: set[str]
) -> list[JsonDict]:
    """Prove each requested mutation changes input and fails evaluation."""

    baseline_input = json.dumps(roadmap_document, sort_keys=True, default=str)
    baseline_evidence = json.dumps(
        evaluate_contract(design_text, roadmap_document, retired_ids), sort_keys=True, default=str
    )
    rows = []
    for mutation in REQUIRED_MUTATIONS:
        changed = deepcopy(roadmap_document)
        apply_mutation(changed, mutation)
        changed_input = json.dumps(changed, sort_keys=True, default=str)
        try:
            result = evaluate_contract(design_text, changed, retired_ids)
            contract_passed = result["passed"]
            evaluation_changed = (
                json.dumps(result, sort_keys=True, default=str) != baseline_evidence
            )
            failed_checks = [check["check"] for check in result["checks"] if not check["passed"]]
            error = None
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            contract_passed = False
            evaluation_changed = True
            failed_checks = ["contract_parse"]
            error = f"{type(exc).__name__}: {exc}"
        input_changed = changed_input != baseline_input
        rows.append(
            {
                "mutation": mutation,
                "input_changed": input_changed,
                "evaluation_changed": evaluation_changed,
                "contract_passed": contract_passed,
                "failed_checks": failed_checks,
                "error": error,
                "terminal": True,
                "failed_as_expected": bool(
                    input_changed and evaluation_changed and not contract_passed and failed_checks
                ),
            }
        )
    return rows


def _preconditions(root: Path) -> tuple[list[JsonDict], Any | None]:
    """Check every required input and fully parse the exclusion manifest."""

    rows = []
    manifest: Any | None = None
    for resource, relative in PRECONDITION_PATHS.items():
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        row: JsonDict = {"resource": resource, "path": str(relative), "available": available}
        if resource == "full_exclusion_manifest" and available:
            try:
                manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
                complete_shape = bool(
                    isinstance(manifest, Mapping)
                    and all(isinstance(manifest.get(key), list) for key in (
                        "retired",
                        "retired_experiments",
                        "retired_extras",
                    ))
                )
                row["full_manifest_shape"] = complete_shape
                row["parse_error"] = None
                row["available"] = available and complete_shape
            except yaml.YAMLError as exc:
                row["full_manifest_shape"] = False
                row["parse_error"] = f"{type(exc).__name__}: {exc}"
                row["available"] = False
        rows.append(row)
    return rows, manifest


def _run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one lint and preserve raw output plus its failure kind."""

    command = shlex.join(str(value) for value in argv)
    try:
        result = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
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


def _internal_receipt(name: str, command: str, rows: list[JsonDict]) -> JsonDict:
    """Record an internal deterministic check in the command receipt shape."""

    failures = [row for row in rows if not row["passed"]]
    passed = not failures
    return {
        "name": name,
        "command": command,
        "exit_code": 0 if passed else 1,
        "stdout": json.dumps({"checked": len(rows), "failures": failures}, sort_keys=True),
        "stderr": "",
        "outcome": "pass" if passed else "contract_defect",
        "terminal": True,
        "passed": passed,
    }


def run_lint_commands(root: Path, evaluation: JsonDict) -> list[JsonDict]:
    """Run each named repository lint and retain its complete receipt."""

    python = root / ".venv/bin/python"
    roadmap = str(ACTIVE_ROADMAP_PATH)
    schema_code = (
        "from pathlib import Path; import sys,yaml; sys.path.insert(0,'scripts'); "
        "from roadmap_schema import Roadmap; "
        "Roadmap.model_validate(yaml.safe_load(Path('research-roadmap.yaml').read_text())); "
        "print('roadmap schema validation clean')"
    )
    receipts = [
        _run_command(root, "roadmap_schema", [str(python), "-c", schema_code]),
        _run_command(root, "prior_failure_validation", [
            str(python), str(PRIOR_VALIDATOR_PATH), roadmap
        ]),
        _run_command(root, "exclusion_manifest_lint", [
            str(python), str(EXCLUSION_LINT_PATH), roadmap
        ]),
        _run_command(root, "roadmap_gate_audit", [
            str(python), str(GATE_AUDIT_PATH), roadmap
        ]),
        _run_command(root, "verdict_row_consistency_lint", [
            str(python), str(VERDICT_LINT_PATH), str(PRIOR_AUDIT_PATH)
        ]),
        _internal_receipt(
            "retired_upstream_check",
            f"internal retired_upstream_check {ACTIVE_ROADMAP_PATH} {EXCLUSION_PATH}",
            evaluation["retired_scope_rows"],
        ),
        _run_command(root, "arc_live_path_lint", [
            str(python), str(ARC_LIVE_PATH_LINT), "--self-test"
        ]),
        _internal_receipt(
            "prompt_ending_check",
            f"internal prompt_ending_check {ACTIVE_ROADMAP_PATH}",
            evaluation["prompt_ending_rows"],
        ),
    ]
    return receipts


def _source_hashes(root: Path) -> JsonDict:
    """Hash each available source so later readers can reproduce the audit."""

    hashes = {}
    for name, relative in SOURCE_PATHS.items():
        path = root / relative
        if path.is_file():
            data = path.read_bytes()
            hashes[name] = {
                "path": str(relative),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    return hashes


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding its self-referential checksum."""

    body = dict(artifact)
    body.pop("reproducibility_checksum", None)
    data = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(data.encode('utf-8')).hexdigest()}"


def _blocked_artifact(
    root: Path,
    run_date: str,
    started: float,
    preconditions: list[JsonDict],
    failed_check: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Return the complete blocked shape used for unavailable audit inputs."""

    check = _check(failed_check, expected, observed, False)
    artifact: JsonDict = {
        "schema": "carnot.v611_contract_advisory.v1",
        "experiment_id": 6972,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": _source_hashes(root),
        "rows": [{"row_type": "contract_check", **check}],
        "task_contract_rows": [],
        "gate_rows": [],
        "producer_field_rows": [],
        "model_contract_rows": [],
        "prior_failure_rows": [],
        "retired_scope_rows": [],
        "arc_solve_rule_rows": [],
        "prompt_ending_rows": [],
        "command_receipt_rows": [],
        "v611_contract_audit_complete_score": 0,
        "contract_conforms_score": 0,
        "random_seed": RANDOM_SEED,
        "gate_check_summary": {
            "audit_complete": False,
            "passed": False,
            "failed_check": failed_check,
            "expected": expected,
            "observed": observed,
            "failures": [
                {"failed_check": failed_check, "expected": expected, "observed": observed}
            ],
            "checks": [check],
        },
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Build a terminal V611 audit artifact without writing tracked state."""

    started = time.monotonic()
    preconditions, manifest = _preconditions(root)
    if not all(row["available"] for row in preconditions):
        return _blocked_artifact(
            root,
            run_date,
            started,
            preconditions,
            "preconditions",
            "both contracts, schema, full manifest, and every named lint are readable",
            preconditions,
        )
    try:
        design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
        roadmap_document = yaml.safe_load((root / ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
        retired_ids = retired_experiment_ids(manifest)
        evaluation = evaluate_contract(design_text, roadmap_document, retired_ids)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return _blocked_artifact(
            root,
            run_date,
            started,
            preconditions,
            "contract_parse",
            "Markdown, YAML, and full manifest parse independently",
            f"{type(exc).__name__}: {exc}",
        )
    mutation_rows = build_mutation_rows(design_text, roadmap_document, retired_ids)
    receipts = run_lint_commands(root, evaluation)
    commands_terminal = all(row["terminal"] for row in receipts)
    mutations_terminal = all(row["terminal"] for row in mutation_rows)
    mutations_pass = all(row["failed_as_expected"] for row in mutation_rows)
    checks = list(evaluation["checks"])
    checks.extend(
        [
            _check(
                "commands_terminal",
                list(COMMAND_NAMES),
                {row["name"]: row["outcome"] for row in receipts},
                commands_terminal and [row["name"] for row in receipts] == list(COMMAND_NAMES),
            ),
            _check(
                "commands_pass",
                "all named checks report no contract defect",
                {row["name"]: row["exit_code"] for row in receipts},
                all(row["passed"] for row in receipts),
            ),
            _check(
                "required_mutations",
                list(REQUIRED_MUTATIONS),
                {row["mutation"]: row["failed_as_expected"] for row in mutation_rows},
                mutations_terminal and mutations_pass,
            ),
        ]
    )
    audit_complete = bool(commands_terminal and mutations_terminal)
    tool_failure = any(not row["terminal"] for row in receipts)
    conforms = bool(audit_complete and not tool_failure and all(check["passed"] for check in checks))
    if tool_failure:
        verdict_class = "blocked"
        honest_verdict = BLOCKED_VERDICT
        status = "blocked"
    elif conforms:
        verdict_class = "circular_positive"
        honest_verdict = CONFORMS_VERDICT
        status = "complete"
    else:
        verdict_class = "null"
        honest_verdict = DEFECT_VERDICT
        status = "complete"
    failures = [
        {
            "failed_check": check["check"],
            "expected": check["expected"],
            "observed": check["observed"],
        }
        for check in checks
        if not check["passed"]
    ]
    first = failures[0] if failures else {
        "failed_check": None,
        "expected": "all contract checks pass",
        "observed": "all contract checks pass",
    }
    rows = [{"row_type": "contract_check", **row} for row in checks]
    rows.extend({"row_type": "mutation", **row} for row in mutation_rows)
    artifact: JsonDict = {
        "schema": "carnot.v611_contract_advisory.v1",
        "experiment_id": 6972,
        "run_date": run_date,
        "status": status,
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": _source_hashes(root),
        "rows": rows,
        "task_contract_rows": evaluation["task_contract_rows"],
        "gate_rows": evaluation["gate_rows"],
        "producer_field_rows": evaluation["producer_field_rows"],
        "model_contract_rows": evaluation["model_contract_rows"],
        "prior_failure_rows": evaluation["prior_failure_rows"],
        "retired_scope_rows": evaluation["retired_scope_rows"],
        "arc_solve_rule_rows": evaluation["arc_solve_rule_rows"],
        "prompt_ending_rows": evaluation["prompt_ending_rows"],
        "command_receipt_rows": receipts,
        "v611_contract_audit_complete_score": int(audit_complete and not tool_failure),
        "contract_conforms_score": int(conforms),
        "random_seed": RANDOM_SEED,
        "gate_check_summary": {
            "audit_complete": audit_complete and not tool_failure,
            "passed": conforms,
            "failed_check": first["failed_check"],
            "expected": first["expected"],
            "observed": first["observed"],
            "failures": failures,
            "checks": checks,
        },
        "verifier_is_oracle": True,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the required shape, scores, verdict, and checksum."""

    errors = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
    errors.extend(f"missing_required_fields:{field}" for field in missing)
    if missing:
        return errors
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle_mismatch")
    if artifact["verdict_class"] not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    summary = artifact["gate_check_summary"]
    if not isinstance(summary, Mapping):
        errors.append("gate_check_summary_invalid")
        return errors
    complete = bool(summary.get("audit_complete"))
    passed = bool(summary.get("passed"))
    if artifact["v611_contract_audit_complete_score"] != int(complete):
        errors.append("audit_complete_score_mismatch")
    if artifact["contract_conforms_score"] != int(complete and passed):
        errors.append("contract_conforms_score_mismatch")
    if artifact["verdict_class"] == "blocked":
        valid = (
            not complete
            and not passed
            and artifact["honest_verdict"] == BLOCKED_VERDICT
            and artifact["status"] == "blocked"
            and summary.get("failed_check") is not None
            and "expected" in summary
            and "observed" in summary
        )
        if not valid:
            errors.append("blocked_terminal_shape_mismatch")
    elif complete and passed:
        if not (
            artifact["verdict_class"] == "circular_positive"
            and artifact["honest_verdict"] == CONFORMS_VERDICT
            and artifact["status"] == "complete"
        ):
            errors.append("conforming_terminal_shape_mismatch")
    elif complete:
        if not (
            artifact["verdict_class"] == "null"
            and artifact["honest_verdict"] == DEFECT_VERDICT
            and artifact["status"] == "complete"
        ):
            errors.append("defect_terminal_shape_mismatch")
    else:
        errors.append("nonterminal_artifact_shape")
    return errors


def _date_argument(value: str) -> str:
    """Accept only one real calendar date in compact form."""

    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD") from exc
    return parsed.strftime("%Y%m%d")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish the result only after a complete JSON object exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit, validate it, and write the requested artifact."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"validation_errors": errors}, indent=2), file=sys.stderr)
        return 1
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    _write_json_atomic(output, artifact)
    print(json.dumps({
        "output": str(output),
        "verdict_class": artifact["verdict_class"],
        "honest_verdict": artifact["honest_verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper is the supported entrypoint
    raise SystemExit(main())
