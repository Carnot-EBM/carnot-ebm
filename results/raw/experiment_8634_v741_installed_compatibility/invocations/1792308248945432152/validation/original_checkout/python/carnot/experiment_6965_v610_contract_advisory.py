"""Build the deterministic V610 advisory contract-audit artifact.

The audit compares Markdown and YAML as independent inputs. It reports contract
defects without turning its conformance score into a gate on scientific work.
See REQ-REPORT-6965.
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
OUTPUT_PATH = Path("results/experiment_6965_v610_contract_advisory.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
VERDICT_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
ARC_LIVE_PATH_LINT = Path("scripts/arc_llm_on_liveness_lint.py")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_PATH = Path("ops/known-issues.md")
PRIOR_AUDIT_PATH = Path("results/experiment_6954_v609_contract_advisory.json")
PRIOR_CAPSTONE_PATH = Path("results/experiment_6964_v609_capstone.json")
MILESTONE = "2026.09.610"
EXPECTED_NUMBERS = tuple(range(6965, 6979))
FINAL_PROHIBITIONS = "Do NOT push. Do NOT modify scripts/research_conductor.py."
INFERENCE_SUBSTRATE = "deterministic_document_yaml_contract_audit"
RANDOM_SEED = 6965
BLOCKED_VERDICT = "blocked_v610_contract_advisory"
CONFORMS_VERDICT = "complete_circular_positive_v610_contract_conforms"
DEFECT_VERDICT = "complete_null_v610_contract_defects_found"
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
LLM_TASK_NUMBERS = {6966, 6969, 6971, 6976}
ARC_TASK_NUMBERS = {6968}
ARC_FALSE_FIELDS = {
    "solve_claimed",
    "level_claimed",
    "registry_updated",
    "submitted_to_leaderboard",
}
ADVISORY_TASK_IDS = {"exp6965-v610-contract-advisory"}
ADVISORY_SCORE_FIELDS = {"v610_contract_audit_complete_score", "contract_conforms_score"}
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


def _gate(upstream: str, artifact_field: str, op: str = "==", value: Any = 1) -> JsonDict:
    """Return one locked structured gate in executable YAML shape."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": op, "value": value}


_TASK_VALUES = (
    (
        6965,
        "exp6965-v610-contract-advisory",
        "V610 advisory execution-contract and retirement audit",
        "results/experiment_6965_v610_contract_advisory.json",
        [],
    ),
    (
        6966,
        "exp6966-gguf-load-envelope-canary",
        "Three-family GGUF load-envelope and teardown canary",
        "results/experiment_6966_gguf_load_envelope_canary.json",
        [],
    ),
    (
        6967,
        "exp6967-certified-error-headroom-fixture",
        "Certified mapping-error and headroom fixture",
        "results/experiment_6967_certified_error_headroom_fixture.json",
        [],
    ),
    (
        6968,
        "exp6968-arc-post-refit-induction-audit",
        "ARC post-refit induction held-out quality audit",
        "results/experiment_6968_arc_post_refit_induction_audit.json",
        [],
    ),
    (
        6969,
        "exp6969-error-structured-prompt-bank",
        "Three-family error-structured prompt candidate bank",
        "results/experiment_6969_error_structured_prompt_bank.json",
        [
            _gate("exp6966-gguf-load-envelope-canary", "gguf_runtime_ready_score"),
            _gate("exp6967-certified-error-headroom-fixture", "error_fixture_ready_score"),
        ],
    ),
    (
        6970,
        "exp6970-bootstrap-prompt-policy-selection",
        "Bootstrap-stable prompt policy selection",
        "results/experiment_6970_bootstrap_prompt_policy_selection.json",
        [_gate("exp6969-error-structured-prompt-bank", "prompt_candidate_bank_complete_score")],
    ),
    (
        6971,
        "exp6971-heldout-prompt-policy-ab",
        "Three-family held-out prompt-policy causal A/B",
        "results/experiment_6971_heldout_prompt_policy_ab.json",
        [
            _gate("exp6966-gguf-load-envelope-canary", "gguf_runtime_ready_score"),
            _gate("exp6970-bootstrap-prompt-policy-selection", "stable_prompt_policy_ready_score"),
        ],
    ),
    (
        6972,
        "exp6972-prompt-policy-cold-audit",
        "Fresh-process prompt-policy audit",
        "results/experiment_6972_prompt_policy_cold_audit.json",
        [_gate("exp6971-heldout-prompt-policy-ab", "prompt_ab_run_complete_score")],
    ),
    (
        6973,
        "exp6973-constraint-affine-function-canary",
        "Hard constraint-affine local function canary",
        "results/experiment_6973_constraint_affine_function_canary.json",
        [
            _gate("exp6967-certified-error-headroom-fixture", "error_fixture_ready_score"),
            _gate("exp6969-error-structured-prompt-bank", "prompt_candidate_bank_complete_score"),
        ],
    ),
    (
        6974,
        "exp6974-headroom-projected-energy-selection",
        "Headroom-qualified constraint-affine energy selection",
        "results/experiment_6974_headroom_projected_energy_selection.json",
        [
            _gate("exp6971-heldout-prompt-policy-ab", "prompt_ab_run_complete_score"),
            _gate(
                "exp6971-heldout-prompt-policy-ab",
                "heldout_headroom_group_count",
                ">=",
                6,
            ),
            _gate(
                "exp6973-constraint-affine-function-canary",
                "constraint_affine_feasibility_score",
            ),
        ],
    ),
    (
        6975,
        "exp6975-projected-selection-cold-audit",
        "Fresh-process projected-selection audit",
        "results/experiment_6975_projected_selection_cold_audit.json",
        [
            _gate(
                "exp6974-headroom-projected-energy-selection",
                "projected_selection_run_complete_score",
            )
        ],
    ),
    (
        6976,
        "exp6976-dependency-fenced-self-learning",
        "Verifier-grounded dependency-fenced continuous self-learning",
        "results/experiment_6976_dependency_fenced_self_learning.json",
        [
            _gate("exp6966-gguf-load-envelope-canary", "gguf_runtime_ready_score"),
            _gate(
                "exp6967-certified-error-headroom-fixture",
                "chronological_event_stream_ready_score",
            ),
            _gate("exp6970-bootstrap-prompt-policy-selection", "stable_prompt_policy_ready_score"),
        ],
    ),
    (
        6977,
        "exp6977-self-learning-cold-audit",
        "Fresh-process dependency-fenced learning safety audit",
        "results/experiment_6977_self_learning_cold_audit.json",
        [_gate("exp6976-dependency-fenced-self-learning", "self_learning_run_complete_score")],
    ),
    (
        6978,
        "exp6978-v610-capstone",
        "V610 independent capstone and V611 handoff",
        "results/experiment_6978_v610_capstone.json",
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
EXPECTED_IDS_BY_SHORT_ID = {f"exp{task['number']}": task["task_id"] for task in EXPECTED_TASKS}
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
    "v610_contract_audit_complete_score",
    "contract_conforms_score",
    "random_seed",
    "reproducibility_checksum",
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
    "task_contracts",
    "gate_contracts",
    "producer_fields",
    "model_contracts",
    "prior_failure_contracts",
    "retired_scopes",
    "arc_solve_rules",
    "prompt_endings",
    "bounded_execution",
}


def _principle(field: str) -> str:
    """Explain why one required field is evidence rather than decoration."""

    reasons = {
        "field_principles": "Field reasons make every required value accountable to the audit method.",
        "preconditions_checked": "Precondition rows prevent absent tools from looking like a completed audit.",
        "inference_substrate": "The substrate limits the result to deterministic contract evidence.",
        "duration_s": "Measured duration distinguishes execution from a copied conclusion.",
        "source_artifact_hashes": "Hashes bind every finding to exact source bytes.",
        "rows": "A combined evidence ledger exposes every checked unit and defect.",
        "task_contract_rows": "Task rows preserve count, order, identity, title, deliverable, and gate parity.",
        "gate_rows": "Gate rows stop missing, forward, retired, or advisory dependencies.",
        "producer_field_rows": "Producer rows prove that a consumer can read a declared gate field.",
        "model_contract_rows": "Model rows prevent legacy-only or invalid GGUF loading plans.",
        "prior_failure_rows": "Prior rows make changed mechanisms and retirement decisions auditable.",
        "retired_scope_rows": "Retirement rows keep closed task IDs and upstreams out of the live graph.",
        "arc_solve_rule_rows": "ARC rows prevent offline quality evidence from becoming solve credit.",
        "prompt_ending_rows": "Prompt rows detect truncated sections, commands, and prohibitions.",
        "command_receipt_rows": "Raw receipts distinguish a contract finding from a tool failure.",
        "v610_contract_audit_complete_score": "Completion measures terminal coverage, not a clean contract.",
        "contract_conforms_score": "Conformance records whether all independently checked constraints pass.",
        "random_seed": "A fixed seed declares that this audit contains no hidden sampling choice.",
        "reproducibility_checksum": "A content checksum reveals later changes to the receipt.",
        "gate_check_summary": "Expected and observed values make every failed aggregate check actionable.",
        "verifier_is_oracle": "The oracle flag is true only for document and YAML contract facts.",
        "verdict_class": "A closed class keeps downstream status interpretation deterministic.",
        "honest_verdict": "A terminal prefix communicates the completed, null, or blocked state directly.",
    }
    return reasons.get(field, f"The {field} field identifies one reproducible audit fact.")


FIELD_PRINCIPLES = {field: _principle(field) for field in REQUIRED_ARTIFACT_FIELDS}
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
    "known_issues": KNOWN_ISSUES_PATH,
    "prior_v609_contract_audit": PRIOR_AUDIT_PATH,
    "prior_v609_capstone": PRIOR_CAPSTONE_PATH,
    "spec": SPEC_PATH,
    "module": Path("python/carnot/experiment_6965_v610_contract_advisory.py"),
    "wrapper": Path("scripts/experiments/experiment_6965_v610_contract_advisory.py"),
    "focused_tests": Path("tests/python/test_experiment_6965_v610_contract_advisory.py"),
}


def experiment_number(value: Any) -> int:
    """Extract only a leading ``exp<number>`` identifier."""

    match = re.match(r"^exp(\d+)(?:-|$)", str(value or ""), re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _parse_scalar(text: str) -> Any:
    """Parse a design-table gate value while refusing nested structures."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list)):
        raise ValueError(f"structured gate value must be scalar: {text}")
    return value


def _parse_design_gates(text: str) -> list[JsonDict]:
    """Convert semicolon-separated Markdown gates to the YAML gate shape."""

    if text.strip().lower() == "none":
        return []
    gates = []
    for part in text.split(";"):
        candidate = part.strip().strip("`")
        match = re.fullmatch(
            r"(exp\d+[a-z0-9-]*)\.?\s*`?([a-z][a-z0-9_]*)`?\s*(==|!=|>=|<=|>|<)\s*(.+)",
            candidate,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"invalid structured gate: {candidate}")
        short_id = match.group(1)
        gates.append(
            {
                "upstream": EXPECTED_IDS_BY_SHORT_ID.get(short_id, short_id),
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4)),
            }
        )
    return gates


def parse_design(text: str) -> JsonDict:
    """Parse only the V610 milestone and exact task table from Markdown."""

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
    """Return only text between the required-fields marker and run command."""

    marker = re.search(r"REQUIRED ARTIFACT FIELDS:", prompt, re.IGNORECASE)
    if not marker:
        return ""
    tail = prompt[marker.start() :]
    return re.split(r"\n\s*Run command:", tail, maxsplit=1, flags=re.IGNORECASE)[0]


def parse_roadmap(document: Any) -> JsonDict:
    """Parse executable task values without consulting the design document."""

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
                "gates": [dict(gate) if isinstance(gate, Mapping) else {} for gate in gates],
                "prior_failures": deepcopy(priors),
                "prompt": prompt,
                "required_fields_block": _required_fields_block(prompt),
                "requires_gpu": task.get("requires_gpu") is True,
                "per_unit_rows": task.get("per_unit_rows") is True,
                "max_turns": task.get("max_turns"),
                "estimated_wall_time_min": task.get("estimated_wall_time_min"),
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def _canonical_exp_id(value: Any) -> str | None:
    """Normalize integer and full task IDs to one numeric experiment key."""

    if isinstance(value, int) and not isinstance(value, bool):
        return f"exp{value}"
    number = experiment_number(value)
    return f"exp{number}" if number >= 0 else None


def retired_experiment_ids(document: Any) -> set[str]:
    """Read all retired IDs and honor explicit un-retirement corrections."""

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
            values = [entry.get("experiment_id"), *(entry.get("experiment_ids", []) or [])]
            restored_values = entry.get("un_retired_experiment_ids", []) or []
            for value in values:
                if normalized := _canonical_exp_id(value):
                    retired.add(normalized)
            for value in restored_values:
                if normalized := _canonical_exp_id(value):
                    restored.add(normalized)
    return retired - restored


def _is_retired(task_id: str, retired_ids: set[str]) -> bool:
    """Match full task IDs against normalized or exact retirement entries."""

    normalized = _canonical_exp_id(task_id)
    return task_id in retired_ids or (normalized is not None and normalized in retired_ids)


def _field_declared(block: str, field: str) -> bool:
    """Match a complete artifact field without accepting a longer identifier."""

    return bool(
        re.search(
            rf"(?<![a-z0-9_]){re.escape(field)}(?![a-z0-9_])",
            block,
            re.IGNORECASE,
        )
    )


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Record enough expected and observed evidence to reproduce one decision."""

    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "terminal": True,
    }


def _task_contract_rows(
    document_rows: Sequence[Mapping[str, Any]], yaml_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Retain every absent, extra, reordered, or changed task row."""

    rows = []
    for order, (document, executable, expected) in enumerate(
        zip_longest(document_rows, yaml_rows, EXPECTED_TASKS), 1
    ):
        fields = ("order", "number", "task_id", "title", "deliverable", "gates")
        matches = {
            field: document is not None
            and executable is not None
            and expected is not None
            and document.get(field) == executable.get(field) == expected.get(field)
            for field in fields
        }
        rows.append(
            {
                "order": order,
                "task_id": expected.get("task_id") if expected else None,
                "document_present": document is not None,
                "yaml_present": executable is not None,
                "document": dict(document) if document else None,
                "yaml": dict(executable) if executable else None,
                "expected": dict(expected) if expected else None,
                "field_matches": matches,
                "passed": all(matches.values()),
                "terminal": True,
            }
        )
    return rows


def _gate_and_producer_rows(
    yaml_rows: Sequence[Mapping[str, Any]], retired_ids: set[str]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check every locked gate and the field its producer promises to emit."""

    by_id = {str(task["task_id"]): task for task in yaml_rows}
    positions = {str(task["task_id"]): int(task["order"]) for task in yaml_rows}
    gate_rows = []
    producer_rows = []
    for expected in EXPECTED_TASKS:
        consumer = by_id.get(str(expected["task_id"]))
        observed_gates = consumer.get("gates", []) if consumer else []
        for gate_index, (locked_gate, observed_gate) in enumerate(
            zip_longest(expected["gates"], observed_gates), 1
        ):
            gate = observed_gate or locked_gate or {}
            upstream_id = str(gate.get("upstream") or "")
            upstream = by_id.get(upstream_id)
            retired = _is_retired(upstream_id, retired_ids)
            precedes = bool(
                consumer
                and upstream
                and positions[upstream_id] < positions[str(consumer["task_id"])]
            )
            field = str(gate.get("artifact_field") or "")
            advisory = upstream_id in ADVISORY_TASK_IDS or field in ADVISORY_SCORE_FIELDS
            matches = locked_gate is not None and observed_gate == locked_gate
            gate_passed = bool(
                consumer and upstream and matches and precedes and not retired and not advisory
            )
            gate_rows.append(
                {
                    "task_id": expected["task_id"],
                    "gate_index": gate_index,
                    "consumer_present": consumer is not None,
                    "expected_gate": locked_gate,
                    "observed_gate": observed_gate,
                    "gate_matches": matches,
                    "upstream_exists": upstream is not None,
                    "upstream_precedes_consumer": precedes,
                    "upstream_retired": retired,
                    "advisory_dependency": advisory,
                    "passed": gate_passed,
                    "terminal": True,
                }
            )
            declared = bool(
                upstream and _field_declared(str(upstream["required_fields_block"]), field)
            )
            producer_rows.append(
                {
                    "task_id": expected["task_id"],
                    "gate_index": gate_index,
                    "consumer_present": consumer is not None,
                    "upstream": upstream_id,
                    "artifact_field": field,
                    "producer_present": upstream is not None,
                    "producer_required_fields_block_present": bool(
                        upstream and upstream["required_fields_block"]
                    ),
                    "field_declared": declared,
                    "passed": bool(consumer and upstream and matches and declared),
                    "terminal": True,
                }
            )
    return gate_rows, producer_rows


def _gguf_autotokenizer_call(prompt: str) -> bool:
    """Detect an executable AutoTokenizer call on a literal GGUF repository."""

    return bool(
        re.search(
            r"AutoTokenizer\s*\.\s*from_pretrained\s*\(\s*(['\"])[^'\"]*-GGUF\1",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _model_row(expected: Mapping[str, Any], task: Mapping[str, Any] | None) -> JsonDict:
    """Check current SOTA names, GGUF loading, and agent/model coherence."""

    number = int(expected["number"])
    prompt = str(task.get("prompt") or "") if task else ""
    model_bearing = number in LLM_TASK_NUMBERS
    present = [model_id for model_id in MANDATED_GGUF_MODELS if model_id in prompt]
    tokenizer_call = _gguf_autotokenizer_call(prompt)
    legacy = bool(re.search(r"Qwen3\.5|0\.8B|E4B|legacy", prompt, re.IGNORECASE))
    legacy_only = model_bearing and legacy and not present
    agent = str(task.get("agent_type") or "").lower() if task else ""
    model = str(task.get("model") or "") if task else ""
    agent_model_ok = (agent == "codex" and model == "gpt-5.6-sol") or (
        agent == "claude" and model in {"sonnet", "opus"}
    )
    model_plan_ok = not model_bearing or ("MODEL_SPECS" in prompt and bool(present))
    return {
        "task_id": expected["task_id"],
        "task_present": task is not None,
        "model_bearing": model_bearing,
        "model_specs_present": "MODEL_SPECS" in prompt,
        "mandated_models_present": present,
        "legacy_only_headline": legacy_only,
        "gguf_autotokenizer_call": tokenizer_call,
        "agent_type": agent,
        "model": model,
        "agent_model_coherent": agent_model_ok,
        "passed": bool(
            task and agent_model_ok and model_plan_ok and not tokenizer_call and not legacy_only
        ),
        "terminal": True,
    }


def _prior_row(expected: Mapping[str, Any], task: Mapping[str, Any] | None) -> JsonDict:
    """Require complete text fields and a real Boolean retirement choice."""

    failures = []
    priors = task.get("prior_failures", []) if task else []
    for index, prior in enumerate(priors):
        if not isinstance(prior, Mapping):
            failures.append(
                {"index": index, "malformed": True, "missing": list(PRIOR_FAILURE_FIELDS)}
            )
            continue
        missing = [field for field in PRIOR_FAILURE_FIELDS if field not in prior]
        empty = [
            field
            for field in PRIOR_FAILURE_FIELDS[:-1]
            if field in prior and not str(prior.get(field) or "").strip()
        ]
        type_errors = [
            "retire_if_same_verdict"
            for _ in [0]
            if "retire_if_same_verdict" in prior
            and not isinstance(prior["retire_if_same_verdict"], bool)
        ]
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
        "task_id": expected["task_id"],
        "task_present": task is not None,
        "prior_failure_count": len(priors),
        "field_failures": failures,
        "passed": bool(task and not failures),
        "terminal": True,
    }


def _arc_row(expected: Mapping[str, Any], task: Mapping[str, Any] | None) -> JsonDict:
    """Require explicit false solve fields and forbid ground-truth calibration."""

    prompt = str(task.get("prompt") or "") if task else ""
    block = str(task.get("required_fields_block") or "") if task else ""
    false_fields = {
        field: bool(re.search(rf"\b{re.escape(field)}\s*(?:=|\()\s*false\b", prompt, re.IGNORECASE))
        and _field_declared(block, field)
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
    ground_truth_prohibited = bool(
        re.search(
            r"(?:do not|never).{0,100}outer-loop ground truth",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )
    passed = bool(
        task and all(false_fields.values()) and ground_truth_prohibited and not positive_claim
    )
    return {
        "task_id": expected["task_id"],
        "task_present": task is not None,
        "required_false_fields": false_fields,
        "outer_loop_ground_truth_prohibited": ground_truth_prohibited,
        "positive_solve_claim": positive_claim,
        "passed": passed,
        "terminal": True,
    }


def _prompt_row(expected: Mapping[str, Any], task: Mapping[str, Any] | None) -> JsonDict:
    """Check section order, one task-owned command, blocked path, and exact ending."""

    prompt = str(task.get("prompt") or "") if task else ""
    indices = [prompt.find(section) for section in PROMPT_SECTIONS]
    sections_ok = all(index >= 0 for index in indices) and indices == sorted(indices)
    expected_command = (
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        f"{Path(str(expected['deliverable'])).stem}.py --date {{date}}"
    )
    commands = re.findall(r"^\s*Run command:.*$", prompt, re.MULTILINE)
    command_ok = len(commands) == 1 and commands[0].strip() == expected_command
    ending_ok = prompt.rstrip().endswith(FINAL_PROHIBITIONS)
    blocked_ok = bool(
        re.search(
            r"PRECONDITIONS:.*?\bwrite\s+blocked_[a-z0-9_]+\s+with\s+gate_check_summary",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
    )
    return {
        "task_id": expected["task_id"],
        "task_present": task is not None,
        "section_positions": dict(zip(PROMPT_SECTIONS, indices, strict=True)),
        "expected_run_command": expected_command,
        "observed_run_commands": [line.strip() for line in commands],
        "blocked_precondition_artifact": blocked_ok,
        "push_prohibition_present": "Do NOT push." in prompt,
        "conductor_prohibition_present": "Do NOT modify scripts/research_conductor.py." in prompt,
        "exact_terminal_line": ending_ok,
        "passed": bool(task and sections_ok and command_ok and blocked_ok and ending_ok),
        "terminal": True,
    }


def _bounded_row(expected: Mapping[str, Any], task: Mapping[str, Any] | None) -> JsonDict:
    """Check declared limits and task-owned checkpoints for GPU work."""

    prompt = str(task.get("prompt") or "") if task else ""
    turns = task.get("max_turns") if task else None
    wall = task.get("estimated_wall_time_min") if task else None
    turns_ok = isinstance(turns, int) and not isinstance(turns, bool) and 1 <= turns <= 100
    wall_ok = isinstance(wall, int) and not isinstance(wall, bool) and 1 <= wall <= 720
    finite_boundary = bool(
        re.search(
            r"\b(?:exactly|at most|up to|fixed|freeze|frozen|every|all \d+|immutable)\b",
            prompt,
            re.IGNORECASE,
        )
    )
    checkpoint = bool(
        re.search(
            r"checkpoint.{0,80}(?:each|every)|(?:each|every).{0,80}checkpoint",
            prompt,
            re.IGNORECASE,
        )
    )
    task_owned = "task-owned" in prompt.lower()
    requires_gpu = task.get("requires_gpu") is True if task else False
    gpu_ok = not requires_gpu or (checkpoint and task_owned)
    classes_ok = all(re.search(rf"\b{value}\b", prompt) for value in CLOSED_VERDICT_CLASSES)
    prefix_ok = bool(
        re.search(
            r"honest_verdict\s+with\s+a\s+terminal\s+prefix\s+consistent\s+with\s+verdict_class",
            prompt,
            re.IGNORECASE,
        )
    )
    return {
        "task_id": expected["task_id"],
        "task_present": task is not None,
        "per_unit_rows": bool(task and task.get("per_unit_rows")),
        "max_turns": turns,
        "max_turns_bounded": turns_ok,
        "estimated_wall_time_min": wall,
        "wall_time_bounded": wall_ok,
        "finite_input_or_unit_boundary": finite_boundary,
        "requires_gpu": requires_gpu,
        "per_unit_checkpoint": checkpoint if requires_gpu else None,
        "task_owned_execution_evidence": task_owned if requires_gpu else None,
        "closed_verdict_classes": classes_ok,
        "terminal_prefix_contract": prefix_ok,
        "passed": bool(
            task
            and task.get("per_unit_rows") is True
            and turns_ok
            and wall_ok
            and finite_boundary
            and gpu_ok
            and classes_ok
            and prefix_ok
        ),
        "terminal": True,
    }


def evaluate_contract(design_text: str, roadmap_document: Any, retired_ids: set[str]) -> JsonDict:
    """Evaluate both contract sources and retain every row-level defect."""

    design = parse_design(design_text)
    roadmap = parse_roadmap(roadmap_document)
    document_rows = design["tasks"]
    yaml_rows = roadmap["tasks"]
    task_rows = _task_contract_rows(document_rows, yaml_rows)
    gate_rows, producer_rows = _gate_and_producer_rows(yaml_rows, retired_ids)
    yaml_by_id = {str(task["task_id"]): task for task in yaml_rows}
    model_rows = [
        _model_row(expected, yaml_by_id.get(str(expected["task_id"])))
        for expected in EXPECTED_TASKS
    ]
    prior_rows = [
        _prior_row(expected, yaml_by_id.get(str(expected["task_id"])))
        for expected in EXPECTED_TASKS
    ]
    prompt_rows = [
        _prompt_row(expected, yaml_by_id.get(str(expected["task_id"])))
        for expected in EXPECTED_TASKS
    ]
    bounded_rows = [
        _bounded_row(expected, yaml_by_id.get(str(expected["task_id"])))
        for expected in EXPECTED_TASKS
    ]
    arc_rows = [
        _arc_row(expected, yaml_by_id.get(str(expected["task_id"])))
        for expected in EXPECTED_TASKS
        if int(expected["number"]) in ARC_TASK_NUMBERS
    ]
    retired_rows = []
    for task in yaml_rows:
        retired_upstreams = [
            str(gate.get("upstream") or "")
            for gate in task["gates"]
            if _is_retired(str(gate.get("upstream") or ""), retired_ids)
        ]
        task_retired = _is_retired(str(task["task_id"]), retired_ids)
        retired_rows.append(
            {
                "task_id": task["task_id"],
                "task_id_retired": task_retired,
                "retired_gate_upstreams": retired_upstreams,
                "passed": not task_retired and not retired_upstreams,
                "terminal": True,
            }
        )
    checks = [
        _check(
            "document_milestone", MILESTONE, design["milestone"], design["milestone"] == MILESTONE
        ),
        _check(
            "yaml_milestone", MILESTONE, roadmap["milestone"], roadmap["milestone"] == MILESTONE
        ),
        _check("document_task_count", 14, len(document_rows), len(document_rows) == 14),
        _check("yaml_task_count", 14, len(yaml_rows), len(yaml_rows) == 14),
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
        ("bounded_execution", bounded_rows),
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
        "bounded_execution_rows": bounded_rows,
    }


def apply_mutation(roadmap_document: JsonDict, mutation: str) -> None:
    """Inject one named defect used by both tests and audit receipts."""

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
        tasks[4]["gated_on"][0]["artifact_field"] = "mutated_ready_score"
    elif mutation == "producer_field":
        gate = tasks[4]["gated_on"][0]
        producer = next(task for task in tasks if task["id"] == gate["upstream"])
        producer["prompt"] = producer["prompt"].replace(
            gate["artifact_field"], "mutated_producer_output"
        )
    elif mutation == "model_id":
        for model_id in MANDATED_GGUF_MODELS:
            tasks[1]["prompt"] = tasks[1]["prompt"].replace(model_id, "Qwen3.5-0.8B")
    elif mutation.startswith("prior_failure_"):
        field = mutation.removeprefix("prior_failure_")
        task = next(task for task in tasks if task.get("prior_failures"))
        task["prior_failures"][0].pop(field, None)
    elif mutation == "arc_solve_claim":
        task = next(task for task in tasks if experiment_number(task.get("id")) in ARC_TASK_NUMBERS)
        task["prompt"] = task["prompt"].replace("solve_claimed=false", "solve_claimed=true")
        task["prompt"] = task["prompt"].replace("solve_claimed (false)", "solve_claimed (true)")
    elif mutation == "prompt_ending":
        tasks[0]["prompt"] = tasks[0]["prompt"].replace(FINAL_PROHIBITIONS, "Do NOT push.")
    else:
        raise ValueError(f"unknown mutation: {mutation}")


def build_mutation_rows(
    design_text: str, roadmap_document: Mapping[str, Any], retired_ids: set[str]
) -> list[JsonDict]:
    """Prove each requested mutation changes observable evaluator evidence."""

    baseline_input = json.dumps(roadmap_document, sort_keys=True, default=str)
    baseline_result = evaluate_contract(design_text, roadmap_document, retired_ids)
    baseline_evidence = json.dumps(baseline_result, sort_keys=True, default=str)
    rows = []
    for mutation in REQUIRED_MUTATIONS:
        changed = deepcopy(roadmap_document)
        apply_mutation(changed, mutation)
        changed_input = json.dumps(changed, sort_keys=True, default=str)
        try:
            result = evaluate_contract(design_text, changed, retired_ids)
            contract_passed = result["passed"]
            changed_evidence = json.dumps(result, sort_keys=True, default=str)
            evaluation_changed = changed_evidence != baseline_evidence
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


def _run_command(root: Path, name: str, argv: Sequence[str]) -> JsonDict:
    """Run one lint and retain complete stdout, stderr, and failure kind."""

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


def _internal_receipt(name: str, command: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Represent a deterministic in-process lint with the same raw receipt shape."""

    failures = [dict(row) for row in rows if not row.get("passed")]
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


def run_lint_commands(root: Path, evaluation: Mapping[str, Any]) -> list[JsonDict]:
    """Run every named repository or internal V610 contract lint."""

    python = sys.executable
    schema_program = (
        "from pathlib import Path; import sys,yaml; sys.path.insert(0, 'scripts'); "
        "from roadmap_schema import Roadmap; "
        "Roadmap.model_validate(yaml.safe_load(Path('research-roadmap.yaml').read_text())); "
        "print('roadmap schema validation clean')"
    )
    rows = [
        _run_command(root, "roadmap_schema", [python, "-c", schema_program]),
        _run_command(
            root,
            "prior_failure_validation",
            [python, str(PRIOR_VALIDATOR_PATH), str(ACTIVE_ROADMAP_PATH)],
        ),
        _run_command(
            root,
            "exclusion_manifest_lint",
            [python, str(EXCLUSION_LINT_PATH), str(ACTIVE_ROADMAP_PATH)],
        ),
        _run_command(
            root,
            "roadmap_gate_audit",
            [python, str(GATE_AUDIT_PATH), str(ACTIVE_ROADMAP_PATH)],
        ),
        _run_command(
            root,
            "verdict_row_consistency_lint",
            [python, str(VERDICT_LINT_PATH), str(PRIOR_AUDIT_PATH)],
        ),
        _internal_receipt(
            "retired_upstream_check",
            "internal retired_upstream_check research-roadmap.yaml ops/exclusion_manifest.yaml",
            evaluation["retired_scope_rows"],
        ),
        _run_command(root, "arc_live_path_lint", [python, str(ARC_LIVE_PATH_LINT), "--self-test"]),
        _internal_receipt(
            "prompt_ending_check",
            "internal prompt_ending_check research-roadmap.yaml",
            evaluation["prompt_ending_rows"],
        ),
    ]
    return rows


def _sha256(path: Path) -> str:
    """Hash source bytes without normalizing their contract-significant text."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable receipt content while excluding duration and this checksum."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    payload.pop("duration_s", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _flat_rows(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Tag every grouped row so the combined evidence ledger remains searchable."""

    return [{"row_type": name, **dict(row)} for name, rows in groups.items() for row in rows]


def _score_state(artifact: Mapping[str, Any]) -> tuple[int, int]:
    """Recompute completion and conformance from terminal rows, never headlines."""

    summary = artifact.get("gate_check_summary")
    checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
    by_name = {
        str(row.get("check")): row
        for row in checks
        if isinstance(row, Mapping) and row.get("check")
    }
    commands = artifact.get("command_receipt_rows")
    mutations = artifact.get("mutation_rows")
    evaluation_complete = EVALUATION_CHECK_NAMES <= by_name.keys() and all(
        by_name[name].get("terminal") is True for name in EVALUATION_CHECK_NAMES
    )
    inputs_complete = all(
        by_name.get(name, {}).get("passed") is True for name in ("preconditions", "contract_parse")
    )
    commands_complete = (
        isinstance(commands, list)
        and [row.get("name") for row in commands] == list(COMMAND_NAMES)
        and all(row.get("terminal") is True for row in commands)
    )
    mutations_complete = (
        isinstance(mutations, list)
        and [row.get("mutation") for row in mutations] == list(REQUIRED_MUTATIONS)
        and all(
            row.get("terminal") is True
            and row.get("input_changed") is True
            and row.get("evaluation_changed") is True
            for row in mutations
        )
    )
    complete = int(
        inputs_complete and evaluation_complete and commands_complete and mutations_complete
    )
    evaluations_pass = evaluation_complete and all(
        by_name[name].get("passed") is True for name in EVALUATION_CHECK_NAMES
    )
    commands_pass = commands_complete and all(row.get("passed") is True for row in commands)
    mutations_pass = mutations_complete and all(
        row.get("failed_as_expected") is True for row in mutations
    )
    return complete, int(bool(complete) and evaluations_pass and commands_pass and mutations_pass)


def _manifest_is_full(document: Any) -> bool:
    """Require every canonical manifest section to parse as a list."""

    return isinstance(document, Mapping) and all(
        isinstance(document.get(section), list)
        for section in ("retired", "retired_experiments", "retired_extras")
    )


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Run the audit once and preserve complete, null, or blocked evidence."""

    started = time.monotonic()
    preconditions = [
        {
            "resource": name,
            "path": str(path),
            "available": (root / path).is_file() and (root / path).stat().st_size > 0,
        }
        for name, path in PRECONDITION_PATHS.items()
    ]
    manifest_document: Any = None
    manifest_error = ""
    manifest_path = root / EXCLUSION_PATH
    if manifest_path.is_file():
        try:
            manifest_document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not _manifest_is_full(manifest_document):
                manifest_error = "full exclusion manifest must contain three list sections"
        except (OSError, yaml.YAMLError) as exc:
            manifest_error = f"{type(exc).__name__}: {exc}"
    manifest_row = next(
        row for row in preconditions if row["resource"] == "full_exclusion_manifest"
    )
    manifest_row["parse_error"] = manifest_error or None
    manifest_row["full_manifest_shape"] = _manifest_is_full(manifest_document)
    manifest_row["available"] = bool(manifest_row["available"] and not manifest_error)
    preconditions_ok = all(row["available"] for row in preconditions)
    source_hashes = {
        name: _sha256(root / path) if (root / path).is_file() else None
        for name, path in SOURCE_PATHS.items()
    }
    empty_groups: JsonDict = {
        "task_contract_rows": [],
        "gate_rows": [],
        "producer_field_rows": [],
        "model_contract_rows": [],
        "prior_failure_rows": [],
        "retired_scope_rows": [],
        "arc_solve_rule_rows": [],
        "prompt_ending_rows": [],
        "bounded_execution_rows": [],
    }
    evaluation: JsonDict = {"passed": False, "checks": [], **empty_groups}
    mutation_rows: list[JsonDict] = []
    command_rows: list[JsonDict] = []
    parse_error = ""
    if preconditions_ok:
        try:
            design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
            roadmap_document = yaml.safe_load(
                (root / ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8")
            )
            retired_ids = retired_experiment_ids(manifest_document)
            evaluation = evaluate_contract(design_text, roadmap_document, retired_ids)
            mutation_rows = build_mutation_rows(design_text, roadmap_document, retired_ids)
            command_rows = run_lint_commands(root, evaluation)
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    checks = [
        _check(
            "preconditions",
            "both contracts, schema, full manifest, and every named lint are readable",
            preconditions,
            preconditions_ok,
        ),
        _check(
            "contract_parse",
            "Markdown, YAML, and the full manifest parse independently",
            parse_error or ("parsed" if preconditions_ok else "not attempted"),
            preconditions_ok and not parse_error,
        ),
        *evaluation["checks"],
        _check(
            "commands_terminal",
            list(COMMAND_NAMES),
            {row["name"]: row["outcome"] for row in command_rows},
            [row["name"] for row in command_rows] == list(COMMAND_NAMES)
            and all(row["terminal"] for row in command_rows),
        ),
        _check(
            "commands_pass",
            "all named checks report no contract defect",
            {row["name"]: row["exit_code"] for row in command_rows},
            bool(command_rows) and all(row["passed"] for row in command_rows),
        ),
        _check(
            "required_mutations",
            list(REQUIRED_MUTATIONS),
            {row["mutation"]: row["failed_as_expected"] for row in mutation_rows},
            [row["mutation"] for row in mutation_rows] == list(REQUIRED_MUTATIONS)
            and all(row["failed_as_expected"] for row in mutation_rows),
        ),
    ]
    groups = {key: evaluation[key] for key in empty_groups}
    groups["command_receipt_rows"] = command_rows
    groups["mutation_rows"] = mutation_rows
    artifact: JsonDict = {
        "schema": "carnot.v610_contract_advisory.v1",
        "experiment_id": 6965,
        "run_date": run_date,
        "status": "blocked",
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": source_hashes,
        "rows": _flat_rows(groups),
        **groups,
        "v610_contract_audit_complete_score": 0,
        "contract_conforms_score": 0,
        "random_seed": RANDOM_SEED,
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
    artifact["v610_contract_audit_complete_score"] = complete
    artifact["contract_conforms_score"] = conforms
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
            status="complete", verdict_class="circular_positive", honest_verdict=CONFORMS_VERDICT
        )
    elif complete:
        artifact.update(status="complete", verdict_class="null", honest_verdict=DEFECT_VERDICT)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute scores and reject an inconsistent terminal representation."""

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
    expected_complete, expected_conforms = _score_state(artifact)
    if artifact.get("v610_contract_audit_complete_score") != expected_complete:
        errors.append("audit_complete_score_mismatch")
    if artifact.get("contract_conforms_score") != expected_conforms:
        errors.append("conforms_score_mismatch")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        errors.append("gate_check_summary_invalid")
    elif summary.get("audit_complete") is not bool(expected_complete) or summary.get(
        "passed"
    ) is not bool(expected_conforms):
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
    """Reject ambiguous execution dates before any artifact write."""

    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the destination only after a complete JSON write."""

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


if __name__ == "__main__":  # pragma: no cover - the required entrypoint is the wrapper
    raise SystemExit(main())
