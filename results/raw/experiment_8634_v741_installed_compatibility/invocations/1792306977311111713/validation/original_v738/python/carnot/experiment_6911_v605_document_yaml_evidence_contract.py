"""Build the deterministic V605 document-YAML and evidence receipt.

The receipt checks execution metadata only. It cannot approve a scientific
claim. It also preserves the current V604 flag, block, and skip states so a
later task cannot mistake missing work for a successful result. See
REQ-REPORT-6911.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_6898_v604_evidence_admissibility_contract as base


JsonDict = dict[str, Any]
Verifier = Callable[[Path], Mapping[str, Any]]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6911_v605_document_yaml_evidence_contract.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
V604_ARTIFACT_PATHS = {
    6898: Path("results/experiment_6898_v604_evidence_admissibility_contract.json"),
    6899: Path("results/experiment_6899_live_relation_acquisition_canary.json"),
    6900: Path("results/experiment_6900_authentic_anchored_relation_corpus.json"),
    6901: Path("results/experiment_6901_independent_model_relation_qualification.json"),
}
V604_SKIP_TITLES = {
    6902: "Paraphrase-disjoint prospective relation learning",
    6903: "Prospective bounded relation-memory continuous sel",
    6904: "Sealed independent self-learning utility and suppo",
}
V604_SKIP_VERDICTS = {
    6902: "blocked_preemptively_upstream_retired",
    6903: "blocked_preemptively_upstream_retired",
    6904: "blocked_preemptively_upstream_retired",
}
V605_MILESTONE = "2026.09.605"
EXPECTED_NUMBERS = tuple(range(6911, 6923))
FINAL_SENTENCE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
INFERENCE_SUBSTRATE = "deterministic_manifest_and_evidence_audit_no_llm"
RANDOM_SEED = 6911
BLOCKED_VERDICT = "complete_blocked_v605_document_yaml_evidence_contract"
READY_VERDICT = "complete_null_v605_document_yaml_evidence_contract_advisory"
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
REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)


def _gate(upstream: str, field: str, op: str = "==", value: Any = 1) -> JsonDict:
    """Keep expected gate declarations compact and exact."""

    return {"upstream": upstream, "artifact_field": field, "op": op, "value": value}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "order": 1,
        "number": 6911,
        "task_id": "exp6911-v605-document-yaml-evidence-contract",
        "title": "V605 document-YAML execution and evidence contract",
        "deliverable": "results/experiment_6911_v605_document_yaml_evidence_contract.json",
        "design_gate": "none",
        "gates": [],
    },
    {
        "order": 2,
        "number": 6912,
        "task_id": "exp6912-alias-safe-relation-corpus-reducer",
        "title": "Alias-safe immutable relation-corpus reducer",
        "deliverable": "results/experiment_6912_alias_safe_relation_corpus_reducer.json",
        "design_gate": "none",
        "gates": [],
    },
    {
        "order": 3,
        "number": 6913,
        "task_id": "exp6913-relation-source-tuple-qualification",
        "title": "Exact relation source and tuple qualification shard",
        "deliverable": "results/experiment_6913_relation_source_tuple_qualification.json",
        "design_gate": "Exp6912 ready = 1",
        "gates": [
            _gate(
                "exp6912-alias-safe-relation-corpus-reducer",
                "clean_relation_corpus_ready_score",
            )
        ],
    },
    {
        "order": 4,
        "number": 6914,
        "task_id": "exp6914-relation-asp-isomorphic-qualification",
        "title": "Exact relation ASP and isomorphic qualification shard",
        "deliverable": "results/experiment_6914_relation_asp_isomorphic_qualification.json",
        "design_gate": "Exp6912 ready = 1",
        "gates": [
            _gate(
                "exp6912-alias-safe-relation-corpus-reducer",
                "clean_relation_corpus_ready_score",
            )
        ],
    },
    {
        "order": 5,
        "number": 6915,
        "task_id": "exp6915-qualified-relation-event-bank",
        "title": "Independent qualified relation event-bank merge",
        "deliverable": "results/experiment_6915_qualified_relation_event_bank.json",
        "design_gate": "Exp6913 ready = 1 and Exp6914 ready = 1",
        "gates": [
            _gate(
                "exp6913-relation-source-tuple-qualification",
                "source_tuple_shard_ready_score",
            ),
            _gate(
                "exp6914-relation-asp-isomorphic-qualification",
                "asp_isomorphic_shard_ready_score",
            ),
        ],
    },
    {
        "order": 6,
        "number": 6916,
        "task_id": "exp6916-isomorphic-prospective-relation-stream",
        "title": "Isomorphic prospective relation learning stream",
        "deliverable": "results/experiment_6916_isomorphic_prospective_relation_stream.json",
        "design_gate": "Exp6915 ready = 1 and event count >= 90",
        "gates": [
            _gate(
                "exp6915-qualified-relation-event-bank",
                "qualified_relation_event_bank_ready_score",
            ),
            _gate(
                "exp6915-qualified-relation-event-bank",
                "qualified_model_relation_event_count",
                ">=",
                90,
            ),
        ],
    },
    {
        "order": 7,
        "number": 6917,
        "task_id": "exp6917-bounded-relation-memory-continuous-learning",
        "title": "Bounded relation-memory continuous self-learning comparison",
        "deliverable": "results/experiment_6917_bounded_relation_memory_continuous_learning.json",
        "design_gate": "Exp6916 ready = 1",
        "gates": [
            _gate(
                "exp6916-isomorphic-prospective-relation-stream",
                "prospective_relation_stream_ready_score",
            )
        ],
    },
    {
        "order": 8,
        "number": 6918,
        "task_id": "exp6918-relation-learning-cold-support-audit",
        "title": "Independent relation-learning safety and support audit",
        "deliverable": "results/experiment_6918_relation_learning_cold_support_audit.json",
        "design_gate": "Exp6917 run complete = 1",
        "gates": [
            _gate(
                "exp6917-bounded-relation-memory-continuous-learning",
                "self_learning_run_complete_score",
            )
        ],
    },
    {
        "order": 9,
        "number": 6919,
        "task_id": "exp6919-exact-prefix-viability-fixture",
        "title": "Exact prefix-viability fixture and branch-cost canary",
        "deliverable": "results/experiment_6919_exact_prefix_viability_fixture.json",
        "design_gate": "none",
        "gates": [],
    },
    {
        "order": 10,
        "number": 6920,
        "task_id": "exp6920-sota-exact-guided-relation-generation",
        "title": "SOTA plain-text relation generation with exact prefix guidance",
        "deliverable": "results/experiment_6920_sota_exact_guided_relation_generation.json",
        "design_gate": "Exp6919 ready = 1",
        "gates": [
            _gate(
                "exp6919-exact-prefix-viability-fixture",
                "prefix_viability_canary_ready_score",
            )
        ],
    },
    {
        "order": 11,
        "number": 6921,
        "task_id": "exp6921-arc-dynamic-supervisor-banked-credit",
        "title": "ARC dynamic supervisor receipt and banked-progress audit",
        "deliverable": "results/experiment_6921_arc_dynamic_supervisor_banked_credit.json",
        "design_gate": "none",
        "gates": [],
    },
    {
        "order": 12,
        "number": 6922,
        "task_id": "exp6922-v605-independent-capstone",
        "title": "V605 independent evidence capstone and branch disposition",
        "deliverable": "results/experiment_6922_v605_independent_capstone.json",
        "design_gate": "none",
        "gates": [],
    },
)
CONTRACT_ROW_FIELDS = (
    "document_yaml_parity_rows",
    "gate_contract_rows",
    "required_field_rows",
    "prior_failure_contract_rows",
    "model_contract_rows",
    "prompt_ending_rows",
    "independent_root_rows",
    "ungated_tail_rows",
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
    "document_task_rows",
    "yaml_task_rows",
    *CONTRACT_ROW_FIELDS,
    "v604_artifact_rows",
    "adversarial_recheck_rows",
    "dependency_taint_rows",
    "task_count",
    "experiment_range",
    "random_seed",
    "reproducibility_checksum",
    "v605_execution_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets later audits reject incompatible receipts.",
    "experiment_id": "The fixed identity binds this receipt to Exp6911.",
    "run_date": "The date separates this audit from later roadmap states.",
    "status": "A terminal status shows that the audit completed.",
    "field_principles": "A reason for each field makes the receipt understandable.",
    "preconditions_checked": "Exact source failures prevent silent fallback.",
    "inference_substrate": "The no-LLM value keeps this audit separate from science.",
    "duration_s": "Measured wall time proves that the reducer ran.",
    "source_artifact_hashes": "Hashes bind findings to the primary source bytes.",
    "rows": "One task row and one V604 state row prevent hidden omissions.",
    "document_task_rows": "Document rows preserve the design independently.",
    "yaml_task_rows": "YAML rows preserve the conductor input independently.",
    "document_yaml_parity_rows": "One row per task exposes identity or order drift.",
    "gate_contract_rows": "Exact gate rows expose invalid or dead dependencies.",
    "required_field_rows": "Field rows prove that producers declare gate outputs.",
    "prior_failure_contract_rows": "Prior rows prevent unchanged failures from returning silently.",
    "model_contract_rows": "Model rows bind tasks to the current routing and GGUF rules.",
    "prompt_ending_rows": "Ending rows prevent changed or trailing commands.",
    "independent_root_rows": "Root rows keep advisory and science branches independent.",
    "ungated_tail_rows": "The tail row keeps the final capstone runnable.",
    "v604_artifact_rows": "One row per prior task separates evidence states.",
    "adversarial_recheck_rows": "Fresh verifier output prevents stale flag decisions.",
    "dependency_taint_rows": "Taint rows stop a blocked reducer from laundering a flagged source.",
    "task_count": "The fixed count catches partial activation.",
    "experiment_range": "The inclusive range catches gaps and reused numbers.",
    "random_seed": "The fixed seed states that this audit has no hidden randomness.",
    "reproducibility_checksum": "A content hash detects silent receipt changes.",
    "v605_execution_contract_ready_score": "This score covers metadata only and cannot approve science.",
    "gate_check_summary": "Expected and observed values make blocks actionable.",
    "verifier_is_oracle": "False prevents this structural audit from acting as scientific authority.",
    "verdict_class": "The closed class keeps the evidence boundary machine-readable.",
    "honest_verdict": "The terminal prefix lets the conductor classify completion safely.",
}
SOURCE_PATHS = {
    "v605_design": DESIGN_PATH,
    "v605_next_roadmap": NEXT_ROADMAP_PATH,
    "v605_active_roadmap": ACTIVE_ROADMAP_PATH,
    "conductor_log": CONDUCTOR_LOG_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "roadmap_schema": SCHEMA_PATH,
    "prior_failure_validator": PRIOR_VALIDATOR_PATH,
    "roadmap_gate_audit": GATE_AUDIT_PATH,
    "exclusion_manifest_lint": EXCLUSION_LINT_PATH,
    "adversarial_verifier": ADVERSARIAL_PATH,
    "module": Path("python/carnot/experiment_6911_v605_document_yaml_evidence_contract.py"),
    "wrapper": Path("scripts/experiments/experiment_6911_v605_document_yaml_evidence_contract.py"),
    "focused_tests": Path("tests/python/test_experiment_6911_v605_document_yaml_evidence_contract.py"),
    "spec": SPEC_PATH,
    **{f"v604_exp{number}": path for number, path in V604_ARTIFACT_PATHS.items()},
}
REQUIRED_SOURCE_NAMES = {
    "v605_design",
    "v605_next_roadmap",
    "v605_active_roadmap",
    "conductor_log",
    "exclusion_manifest",
    "roadmap_schema",
    "prior_failure_validator",
    "roadmap_gate_audit",
    "exclusion_manifest_lint",
    "adversarial_verifier",
    *{f"v604_exp{number}" for number in V604_ARTIFACT_PATHS},
}


experiment_number = base.experiment_number
write_json_atomic = base.write_json_atomic


def _isolated_subprocess_env() -> dict[str, str]:
    """Keep coverage control in the parent while preserving the runtime environment."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_"))
    }


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Create one check with enough detail to diagnose a block."""

    return {"check": name, "expected": expected, "observed": observed, "passed": bool(passed)}


def parse_design(text: str) -> JsonDict:
    """Read the V605 milestone and exact execution table without using YAML."""

    milestone = re.search(r"^\*\*Milestone:\*\*\s*`([^`]+)`", text, re.MULTILINE)
    if not milestone:
        raise ValueError("design milestone is required")
    deliverables = {
        int(number): path
        for number, path in re.findall(
            r"^### Exp(\d+)\s+[—-].*?^\*\*Deliverable:\*\*\s*`([^`]+)`",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if int(number) in EXPECTED_NUMBERS
    }
    row_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`(exp\d+[^`]*)`\s*\|\s*([^|]+?)\s*"
        r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    rows: list[JsonDict] = []
    for order, task_id, title, phase, gate_text in row_pattern.findall(text):
        number = experiment_number(task_id)
        if number not in EXPECTED_NUMBERS:
            continue
        rows.append(
            {
                "order": int(order),
                "number": number,
                "task_id": task_id.strip(),
                "title": title.strip(),
                "phase": phase.strip(),
                "structured_gate": gate_text.strip(),
                "milestone": milestone.group(1),
                "deliverable": deliverables.get(number),
            }
        )
    if not rows:
        raise ValueError("design execution table is required")
    return {"milestone": milestone.group(1), "tasks": rows}


def parse_roadmap(document: Any) -> JsonDict:
    """Read executable task values while retaining prompt and prior records."""

    if not isinstance(document, Mapping):
        raise ValueError("roadmap mapping is required")
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("roadmap tasks list is required")
    rows: list[JsonDict] = []
    for order, task in enumerate(tasks, 1):
        if not isinstance(task, Mapping):
            raise ValueError(f"malformed roadmap task at order {order}")
        gates = task.get("gated_on", []) or []
        priors = task.get("prior_failures", []) or []
        if not isinstance(gates, list) or not isinstance(priors, list):
            raise ValueError(f"malformed task lists at order {order}")
        prompt = str(task.get("prompt") or "")
        rows.append(
            {
                "order": order,
                "number": experiment_number(task.get("id")),
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "milestone": str(task.get("milestone") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "agent_type": str(task.get("agent_type") or ""),
                "model": str(task.get("model") or ""),
                "gates": [dict(gate) if isinstance(gate, Mapping) else {} for gate in gates],
                "prior_failures": deepcopy(priors),
                "prompt": prompt,
                "required_fields": base._required_fields(prompt),
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def current_model_rule(audit_source: str) -> tuple[str, str]:
    """Read the current Codex model and forbidden agent from the gate audit."""

    codex = re.search(r'agent_type\s*==\s*"codex"\s*and\s*model\s*!=\s*"([^"]+)"', audit_source)
    forbidden = re.search(r'agent_type\s*==\s*"([^"]+)"', audit_source)
    if not codex or not forbidden:
        raise ValueError("current model rule is missing from gate audit")
    forbidden_agents = re.findall(r'agent_type\s*==\s*"([^"]+)"', audit_source)
    blocked = next((agent for agent in forbidden_agents if agent != "codex"), "")
    if not blocked:
        raise ValueError("forbidden agent rule is missing from gate audit")
    return codex.group(1), blocked


def _task_shape(row: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only stable identity values in public task rows."""

    if row is None:
        return None
    return {
        "order": row.get("order"),
        "number": row.get("number"),
        "task_id": row.get("task_id"),
        "title": row.get("title"),
        "milestone": row.get("milestone"),
        "deliverable": row.get("deliverable"),
    }


def _public_yaml_row(row: Mapping[str, Any]) -> JsonDict:
    """Replace long prompt text with a stable content hash."""

    return {
        **(_task_shape(row) or {}),
        "agent_type": row.get("agent_type"),
        "model": row.get("model"),
        "gates": deepcopy(row.get("gates", [])),
        "prior_failure_count": len(row.get("prior_failures", [])),
        "prompt_sha256": base.sha256_bytes(str(row.get("prompt") or "").encode()),
    }


def _task_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Emit one exact comparison row for each required V605 position."""

    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        document = document_tasks[index] if index < len(document_tasks) else None
        executable = yaml_tasks[index] if index < len(yaml_tasks) else None
        expected_shape = {
            "order": expected["order"],
            "number": expected["number"],
            "task_id": expected["task_id"],
            "title": expected["title"],
            "milestone": V605_MILESTONE,
            "deliverable": expected["deliverable"],
        }
        document_shape = _task_shape(document)
        yaml_shape = _task_shape(executable)
        checks = {
            "document_matches_expected": document_shape == expected_shape,
            "yaml_matches_expected": yaml_shape == expected_shape,
            "document_yaml_equal": document_shape == yaml_shape,
        }
        rows.append(
            {
                "order": index + 1,
                "expected": expected_shape,
                "document": document_shape,
                "yaml": yaml_shape,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _prompt_rows(yaml_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Require all prompt sections, one command, and the exact ending."""

    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        prompt = str(task.get("prompt") or "") if task else ""
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        commands = [line for line in lines if line.startswith("Run command:")]
        expected_command = (
            "Run command: cd {project_root} && .venv/bin/python "
            f"scripts/experiments/{Path(expected['deliverable']).stem}.py --date {{date}}"
        )
        sections = {section: section in prompt for section in PROMPT_SECTIONS}
        checks = {
            "required_sections": all(sections.values()),
            "one_run_command": len(commands) == 1,
            "exact_run_command": bool(commands and commands[0] == expected_command),
            "exact_final_sentence": bool(lines and lines[-1] == FINAL_SENTENCE),
            "run_command_is_penultimate": bool(len(lines) >= 2 and lines[-2] == expected_command),
        }
        rows.append(
            {
                "number": expected["number"],
                "task_id": task.get("task_id") if task else None,
                "required_sections": sections,
                "expected_run_command": expected_command,
                "observed_run_commands": commands,
                "expected_final_sentence": FINAL_SENTENCE,
                "observed_final_sentence": lines[-1] if lines else None,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _required_field_rows(yaml_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Check common producer fields before resolving any gate owner."""

    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        fields = set(task.get("required_fields", set())) if task else set()
        prompt = str(task.get("prompt") or "") if task else ""
        missing = sorted(COMMON_PROMPT_FIELDS - fields)
        checks = {
            "required_block_present": "REQUIRED ARTIFACT FIELDS:" in prompt,
            "common_fields_present": not missing,
            "principle_rule_present": "one principle per required field" in prompt,
        }
        rows.append(
            {
                "number": expected["number"],
                "task_id": task.get("task_id") if task else None,
                "declared_fields": sorted(fields),
                "missing_common_fields": missing,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _gate_rows(
    document_tasks: Sequence[Mapping[str, Any]],
    yaml_tasks: Sequence[Mapping[str, Any]],
    valid_operators: set[str],
) -> list[JsonDict]:
    """Resolve every gate against its exact earlier producer declaration."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    yaml_by_id = {row.get("task_id"): row for row in yaml_tasks}
    rows: list[JsonDict] = []
    for expected in EXPECTED_TASKS:
        number = expected["number"]
        downstream = yaml_by_number.get(number)
        observed_gates = list(downstream.get("gates", [])) if downstream else []
        expected_gates = list(expected["gates"])
        for gate_index in range(max(len(expected_gates), len(observed_gates))):
            wanted = expected_gates[gate_index] if gate_index < len(expected_gates) else None
            observed = observed_gates[gate_index] if gate_index < len(observed_gates) else None
            upstream = yaml_by_id.get(observed.get("upstream")) if observed else None
            field = str(observed.get("artifact_field") or "") if observed else ""
            checks = {
                "design_gate_matches_expected": document_by_number.get(number, {}).get(
                    "structured_gate"
                )
                == expected["design_gate"],
                "gate_matches_expected": observed == wanted,
                "operator_valid": bool(observed and observed.get("op") in valid_operators),
                "upstream_present": upstream is not None,
                "upstream_earlier": bool(
                    upstream and downstream and upstream.get("order", 0) < downstream.get("order", 0)
                ),
                "producer_field_declared": bool(
                    upstream and field in upstream.get("required_fields", set())
                ),
                "advisory_contract_not_consumed": experiment_number(
                    observed.get("upstream") if observed else None
                )
                != 6911,
            }
            rows.append(
                {
                    "downstream_number": number,
                    "gate_index": gate_index,
                    "expected": deepcopy(wanted),
                    "observed": deepcopy(observed),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return rows


def _prior_rows(
    yaml_tasks: Sequence[Mapping[str, Any]], prior_evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare all four prior fields with primary artifact or skip evidence."""

    required = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
    rows: list[JsonDict] = []
    for task in yaml_tasks:
        for prior_index, value in enumerate(task.get("prior_failures", [])):
            prior = dict(value) if isinstance(value, Mapping) else {}
            prior_id = prior.get("experiment_id")
            primary = prior_evidence.get(str(prior_id), {})
            primary_verdict = primary.get("honest_verdict")
            failed = [field for field in required if field not in prior]
            if primary_verdict != prior.get("verdict"):
                failed.append("verdict")
            if not isinstance(prior.get("addressed_by"), str) or not prior.get(
                "addressed_by", ""
            ).strip():
                failed.append("addressed_by")
            if prior.get("retire_if_same_verdict") is not True:
                failed.append("retire_if_same_verdict")
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "prior_index": prior_index,
                    "experiment_id": prior_id,
                    "declared_verdict": prior.get("verdict"),
                    "primary_verdict": primary_verdict,
                    "primary_source": primary.get("source"),
                    "addressed_by": prior.get("addressed_by"),
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "failed_subfields": list(dict.fromkeys(failed)),
                    "passed": not failed,
                }
            )
    return rows


def _model_rows(yaml_tasks: Sequence[Mapping[str, Any]], audit_source: str) -> list[JsonDict]:
    """Apply current routing rules and the one V605 live-LLM model contract."""

    codex_model, forbidden_agent = current_model_rule(audit_source)
    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        agent = str(task.get("agent_type") or "") if task else ""
        model = str(task.get("model") or "") if task else ""
        prompt = str(task.get("prompt") or "") if task else ""
        fields = set(task.get("required_fields", set())) if task else set()
        routing_ok = bool(task)
        if agent == "codex":
            routing_ok = model == codex_model
        elif agent == forbidden_agent:
            routing_ok = False
        elif agent == "claude":
            routing_ok = model in {"sonnet", "opus"}
        elif agent == "opencode":
            routing_ok = bool(model)
        else:
            routing_ok = False
        llm_task = expected["number"] == 6920
        llm_checks = {
            "all_required_model_ids": not llm_task
            or all(model_id in prompt for model_id in REQUIRED_MODEL_IDS),
            "model_specs_declared": not llm_task or "model_specs" in fields,
            "gguf_native_tokenizer": not llm_task or "llama.cpp's native tokenizer" in prompt,
            "autotokenizer_gguf_forbidden": not llm_task
            or "Never pass a GGUF repository to AutoTokenizer.from_pretrained()." in prompt,
        }
        checks = {"current_agent_model_rule": routing_ok, **llm_checks}
        rows.append(
            {
                "number": expected["number"],
                "task_id": task.get("task_id") if task else None,
                "agent_type": agent,
                "model": model,
                "expected_codex_model": codex_model,
                "forbidden_agent": forbidden_agent,
                "llm_task": llm_task,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _root_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Prove the five independent roots, no advisory fanout, and the tail."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    consumers = [
        task.get("task_id")
        for task in yaml_tasks
        if any(experiment_number(gate.get("upstream")) == 6911 for gate in task.get("gates", []))
    ]
    roots = [
        {
            "kind": "advisory_contract_fanout",
            "number": 6911,
            "yaml_consumers": consumers,
            "passed": not consumers,
        }
    ]
    for number in (6911, 6912, 6919, 6921, 6922):
        document = document_by_number.get(number)
        executable = yaml_by_number.get(number)
        passed = bool(
            document
            and executable
            and document.get("structured_gate") == "none"
            and not executable.get("gates")
        )
        roots.append(
            {
                "kind": "independent_root",
                "number": number,
                "document_gate": document.get("structured_gate") if document else None,
                "yaml_gates": deepcopy(executable.get("gates", [])) if executable else None,
                "passed": passed,
            }
        )
    document_tail = document_by_number.get(6922)
    yaml_tail = yaml_by_number.get(6922)
    tails = [
        {
            "number": 6922,
            "document_present_ungated": bool(
                document_tail and document_tail.get("structured_gate") == "none"
            ),
            "yaml_present_ungated": bool(yaml_tail and not yaml_tail.get("gates")),
            "observed_gates": deepcopy(yaml_tail.get("gates", [])) if yaml_tail else None,
            "passed": bool(
                document_tail
                and yaml_tail
                and document_tail.get("structured_gate") == "none"
                and not yaml_tail.get("gates")
            ),
        }
    ]
    return roots, tails


def evaluate_contract(
    design_text: str,
    roadmap: Mapping[str, Any],
    prior_evidence: Mapping[str, Mapping[str, Any]],
    schema_source: str,
    gate_audit_source: str,
) -> JsonDict:
    """Evaluate V605 identity, prompts, gates, priors, models, and roots."""

    document = parse_design(design_text)
    executable = parse_roadmap(roadmap)
    document_tasks = document["tasks"]
    yaml_tasks = executable["tasks"]
    task_rows = _task_rows(document_tasks, yaml_tasks)
    prompt_rows = _prompt_rows(yaml_tasks)
    field_rows = _required_field_rows(yaml_tasks)
    gate_rows = _gate_rows(document_tasks, yaml_tasks, base.schema_gate_operators(schema_source))
    prior_rows = _prior_rows(yaml_tasks, prior_evidence)
    model_rows = _model_rows(yaml_tasks, gate_audit_source)
    root_rows, tail_rows = _root_rows(document_tasks, yaml_tasks)
    document_deliverables = [row.get("deliverable") for row in document_tasks]
    yaml_deliverables = [row.get("deliverable") for row in yaml_tasks]
    checks = [
        _check(
            "document_primary_contract",
            {"milestone": V605_MILESTONE, "task_count": 12, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": document["milestone"],
                "task_count": len(document_tasks),
                "numbers": [row.get("number") for row in document_tasks],
            },
            document["milestone"] == V605_MILESTONE
            and len(document_tasks) == 12
            and len(document_deliverables) == len(set(document_deliverables)) == 12,
        ),
        _check(
            "yaml_executable_contract",
            {"milestone": V605_MILESTONE, "task_count": 12, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": executable["milestone"],
                "task_count": len(yaml_tasks),
                "numbers": [row.get("number") for row in yaml_tasks],
            },
            executable["milestone"] == V605_MILESTONE
            and len(yaml_tasks) == 12
            and len({row.get("task_id") for row in yaml_tasks}) == 12
            and len(yaml_deliverables) == len(set(yaml_deliverables)) == 12,
        ),
    ]
    for name, rows in (
        ("document_yaml_task_contract", task_rows),
        ("prompt_contract", prompt_rows),
        ("required_field_contract", field_rows),
        ("gate_contract", gate_rows),
        ("prior_failure_contract", prior_rows),
        ("model_contract", model_rows),
        ("independent_root_contract", root_rows),
        ("ungated_tail_contract", tail_rows),
    ):
        passed = bool(rows) and all(row["passed"] for row in rows)
        checks.append(_check(name, True, passed, passed))
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "document_task_rows": deepcopy(document_tasks),
        "yaml_task_rows": [_public_yaml_row(row) for row in yaml_tasks],
        "document_yaml_parity_rows": task_rows,
        "gate_contract_rows": gate_rows,
        "required_field_rows": field_rows,
        "prior_failure_contract_rows": prior_rows,
        "model_contract_rows": model_rows,
        "prompt_ending_rows": prompt_rows,
        "independent_root_rows": root_rows,
        "ungated_tail_rows": tail_rows,
    }


def _dependency_ids(artifact: Mapping[str, Any]) -> set[int]:
    """Read experiment dependencies from source-hash keys and paths."""

    dependencies: set[int] = set()
    sources = artifact.get("source_artifact_hashes", {})
    if not isinstance(sources, Mapping):
        return dependencies
    for key, value in sources.items():
        number = experiment_number(key)
        if number is not None:
            dependencies.add(number)
        if isinstance(value, Mapping):
            number = experiment_number(value.get("path"))
            if number is not None:
                dependencies.add(number)
    return dependencies


def evaluate_v604_evidence(
    artifacts: Mapping[int, Mapping[str, Any]],
    rechecks: Mapping[int, Mapping[str, Any]],
    conductor_log: str,
) -> JsonDict:
    """Classify each V604 task without turning a skip into missing science."""

    recheck_rows: list[JsonDict] = []
    flags: list[JsonDict] = []
    warnings: list[JsonDict] = []
    critical_ids: set[int] = set()
    for number in V604_ARTIFACT_PATHS:
        report = dict(rechecks.get(number, {}))
        report_flags = [dict(row) for row in report.get("flags", []) if isinstance(row, Mapping)]
        for flag in report_flags:
            row = {"experiment_id": number, **flag}
            if str(flag.get("severity", "")).lower() == "critical":
                flags.append(row)
                critical_ids.add(number)
            else:
                warnings.append(row)
        if report.get("loaded") is not True:
            warnings.append({"experiment_id": number, "kind": "adversarial_recheck_not_loaded"})
        recheck_rows.append(
            {
                "experiment_id": number,
                "loaded": report.get("loaded") is True,
                "flag_count": len(report_flags),
                "flags": report_flags,
                "gate_version": report.get("gate_version"),
            }
        )
    rows: list[JsonDict] = []
    blocks: list[JsonDict] = []
    skips: list[JsonDict] = []
    for number in range(6898, 6905):
        artifact = artifacts.get(number)
        if number >= 6902:
            skip_found = V604_SKIP_TITLES[number] in conductor_log and "Pre-emptive skip" in conductor_log
            state = "skipped" if skip_found else "missing_skip_record"
            row = {
                "experiment_id": number,
                "artifact_present": False,
                "conductor_skip_record": skip_found,
                "evidence_state": state,
                "honest_verdict": V604_SKIP_VERDICTS[number] if skip_found else None,
                "science_success": False,
                "admissible": False,
            }
            rows.append(row)
            if skip_found:
                skips.append(deepcopy(row))
            else:
                warnings.append({"experiment_id": number, "kind": "missing_skip_record"})
            continue
        report = rechecks.get(number, {})
        verdict_class = artifact.get("verdict_class") if artifact else None
        loaded = report.get("loaded") is True
        if not loaded:
            state = "unverified"
        elif number in critical_ids:
            state = "flagged"
        elif verdict_class == "blocked" or str(artifact.get("status") if artifact else "").startswith(
            "blocked"
        ) or str(artifact.get("status") if artifact else "").endswith("blocked"):
            state = "blocked"
        else:
            state = "admissible"
        row = {
            "experiment_id": number,
            "artifact_present": artifact is not None,
            "conductor_skip_record": False,
            "evidence_state": state,
            "honest_verdict": artifact.get("honest_verdict") if artifact else None,
            "verdict_class": verdict_class,
            "science_success": False,
            "admissible": state == "admissible",
        }
        rows.append(row)
        if state == "blocked":
            blocks.append(deepcopy(row))
    taint_rows: list[JsonDict] = []
    for number, artifact in artifacts.items():
        tainted = sorted(_dependency_ids(artifact) & critical_ids)
        if tainted:
            taint_rows.append(
                {
                    "claim_experiment_id": number,
                    "source_experiment_id": tainted[0],
                    "all_tainted_sources": tainted,
                    "taint_kind": "transitive_source_flag",
                    "science_success": False,
                }
            )
    expected_states = {
        6898: "blocked",
        6899: "admissible",
        6900: "flagged",
        6901: "blocked",
        6902: "skipped",
        6903: "skipped",
        6904: "skipped",
    }
    passed = all(row["evidence_state"] == expected_states[row["experiment_id"]] for row in rows)
    return {
        "passed": passed,
        "v604_artifact_rows": rows,
        "adversarial_recheck_rows": recheck_rows,
        "dependency_taint_rows": taint_rows,
        "flags": flags,
        "blocks": blocks,
        "skips": skips,
        "warnings": warnings,
    }


def _empty_contract(error: str) -> JsonDict:
    """Return all contract collections when primary parsing cannot start."""

    return {
        "passed": False,
        "checks": [_check("contract_evaluation", "valid primary records", error, False)],
        "document_task_rows": [],
        "yaml_task_rows": [],
        **{field: [] for field in CONTRACT_ROW_FIELDS},
    }


def _read_text(path: Path) -> tuple[str, str | None]:
    """Read UTF-8 text and retain a stable missing or empty label."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return "", f"unreadable:{type(exc).__name__}"
    return (text, None) if text else ("", "empty")


def _read_artifact_summary(path: Path) -> tuple[JsonDict | None, str | None]:
    """Parse a primary artifact in a short process and retain contract fields."""

    if not path.is_file():
        return None, "missing"
    program = (
        "import json,sys; d=json.load(open(sys.argv[1], encoding='utf-8')); "
        "keys=('status','verdict_class','honest_verdict','source_artifact_hashes',"
        "'model_relation_qualification_ready_score'); "
        "print(json.dumps({k:d.get(k) for k in keys if k in d}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        capture_output=True,
        text=True,
        check=False,
        env=_isolated_subprocess_env(),
    )
    if result.returncode != 0:
        return None, f"invalid_json:exit_{result.returncode}"
    value = json.loads(result.stdout)
    return value, None


def _load_prior_evidence(
    root: Path, tasks: Sequence[Mapping[str, Any]], conductor_log: str
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Load prior verdicts from result artifacts or terminal skip records."""

    evidence: dict[str, JsonDict] = {}
    source_rows: list[JsonDict] = []
    for task in tasks:
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            prior_id = str(prior.get("experiment_id") or "")
            number = experiment_number(prior_id)
            if number in V604_SKIP_VERDICTS:
                found = V604_SKIP_TITLES[number] in conductor_log and "Pre-emptive skip" in conductor_log
                if found:
                    evidence[prior_id] = {
                        "honest_verdict": V604_SKIP_VERDICTS[number],
                        "source": CONDUCTOR_LOG_PATH.as_posix(),
                    }
                source_rows.append(
                    {
                        "experiment_id": prior_id,
                        "path": CONDUCTOR_LOG_PATH.as_posix(),
                        "sha256": base.sha256_path(root / CONDUCTOR_LOG_PATH),
                        "error": None if found else "terminal_skip_record_missing",
                    }
                )
                continue
            path = base._prior_artifact_path(root, prior_id)
            payload, error = _read_artifact_summary(path) if path else (None, "invalid_id")
            if payload is not None:
                evidence[prior_id] = {**payload, "source": path.relative_to(root).as_posix()}
            source_rows.append(
                {
                    "experiment_id": prior_id,
                    "path": path.relative_to(root).as_posix() if path else None,
                    "sha256": base.sha256_path(path) if path else None,
                    "error": error,
                }
            )
    return evidence, source_rows


def _external_validation_checks(root: Path, roadmap_path: Path, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Run the current schema, prior, gate, exclusion, and retired-pattern checks."""

    from pydantic import ValidationError
    from scripts.audit_roadmap_gates import audit_roadmap
    from scripts.roadmap_schema import Roadmap
    from scripts.validate_prior_failures import validate_roadmap

    try:
        Roadmap.model_validate(roadmap)
        schema_errors: list[str] = []
    except ValidationError as exc:
        schema_errors = [error["msg"] for error in exc.errors()]
    prior_schema, prior_failures = validate_roadmap(roadmap_path)
    gate_result = audit_roadmap(roadmap_path)
    exclusion = subprocess.run(
        [sys.executable, str(root / EXCLUSION_LINT_PATH), str(roadmap_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=_isolated_subprocess_env(),
    )
    exclusion_observed = exclusion.stdout.strip() or exclusion.stderr.strip()
    return [
        _check("roadmap_schema_validation", [], schema_errors, not schema_errors),
        _check(
            "prior_failure_validation",
            [],
            [*prior_schema, *prior_failures],
            not prior_schema and not prior_failures,
        ),
        _check(
            "gate_audit_validation",
            True,
            gate_result.to_artifact(),
            gate_result.roadmap_gate_audit_passed,
        ),
        _check(
            "exclusion_manifest_validation",
            "exit 0",
            {"exit_code": exclusion.returncode, "output": exclusion_observed},
            exclusion.returncode == 0,
        ),
    ]


def _source_hashes(root: Path, prior_rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Bind required, implementation, evidence, and prior sources to bytes."""

    hashes = {
        name: {"path": path.as_posix(), "sha256": base.sha256_path(root / path)}
        for name, path in SOURCE_PATHS.items()
    }
    for index, row in enumerate(prior_rows):
        hashes[f"prior_{index}_{row.get('experiment_id')}"] = {
            "path": row.get("path"),
            "sha256": row.get("sha256"),
            "error": row.get("error"),
        }
    return hashes


def _checksum_value(value: Any) -> Any:
    """Remove elapsed time and the checksum from reproducible content."""

    if isinstance(value, Mapping):
        return {
            key: _checksum_value(item)
            for key, item in value.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    if isinstance(value, list):
        return [_checksum_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all deterministic receipt content."""

    payload = json.dumps(
        _checksum_value(dict(artifact)), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _default_verifier(path: Path) -> Mapping[str, Any]:
    """Run the verifier in a fresh process so a large audit releases its RAM."""

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / ADVERSARIAL_PATH), str(path), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_isolated_subprocess_env(),
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"adversarial verifier exited {result.returncode}")
    payload = json.loads(result.stdout)
    return payload["reports"][0]


def build_artifact(root: Path, run_date: str, verifier: Verifier | None = None) -> JsonDict:
    """Build a complete advisory or blocked V605 receipt."""

    started = time.perf_counter()
    source_errors: list[JsonDict] = []
    texts: dict[str, str] = {}
    for name, path in SOURCE_PATHS.items():
        if name.startswith(("module", "wrapper", "focused_tests", "spec", "v604_exp")):
            continue
        texts[name], error = _read_text(root / path)
        if name in REQUIRED_SOURCE_NAMES and error:
            source_errors.append({"source": name, "error": error})
    next_roadmap, next_error = base.read_yaml_mapping(root / NEXT_ROADMAP_PATH)
    active_roadmap, active_error = base.read_yaml_mapping(root / ACTIVE_ROADMAP_PATH)
    if next_error:
        source_errors.append({"source": "v605_next_roadmap", "error": next_error})
    if active_error:
        source_errors.append({"source": "v605_active_roadmap", "error": active_error})
    roadmap = next_roadmap or active_roadmap or {}
    roadmap_path = root / (NEXT_ROADMAP_PATH if next_roadmap is not None else ACTIVE_ROADMAP_PATH)
    conductor_log = texts.get("conductor_log", "")
    raw_tasks = roadmap.get("tasks", []) if isinstance(roadmap, Mapping) else []
    prior_evidence, prior_source_rows = _load_prior_evidence(
        root, [task for task in raw_tasks if isinstance(task, Mapping)], conductor_log
    )
    source_errors.extend(
        {"source": row.get("path") or row.get("experiment_id"), "error": row["error"]}
        for row in prior_source_rows
        if row.get("error")
    )
    try:
        contract = evaluate_contract(
            texts.get("v605_design", ""),
            roadmap,
            prior_evidence,
            texts.get("roadmap_schema", ""),
            texts.get("roadmap_gate_audit", ""),
        )
    except (ValueError, SyntaxError) as exc:
        contract = _empty_contract(str(exc))
    external_checks: list[JsonDict] = []
    if roadmap and roadmap_path.is_file():
        external_checks = _external_validation_checks(root, roadmap_path, roadmap)
    else:
        external_checks = [
            _check(name, True, "roadmap unavailable", False)
            for name in (
                "roadmap_schema_validation",
                "prior_failure_validation",
                "gate_audit_validation",
                "exclusion_manifest_validation",
            )
        ]
    artifacts: dict[int, JsonDict] = {}
    reports: dict[int, Mapping[str, Any]] = {}
    checker = verifier or _default_verifier
    for number, path in V604_ARTIFACT_PATHS.items():
        payload, error = _read_artifact_summary(root / path)
        if error:
            source_errors.append({"source": path.as_posix(), "error": error})
            continue
        artifacts[number] = payload or {}
        try:
            reports[number] = checker(root / path)
        except Exception as exc:  # The receipt must preserve verifier failure as evidence.
            reports[number] = {"loaded": False, "flags": []}
            source_errors.append(
                {"source": path.as_posix(), "error": f"verifier_failed:{type(exc).__name__}"}
            )
    evidence = evaluate_v604_evidence(artifacts, reports, conductor_log)
    preconditions = base.gate_summary(
        [_check("required_source_readability", [], source_errors, not source_errors)]
    )
    checks = [
        _check(
            "preconditions",
            "all required sources readable",
            source_errors or "all required sources readable",
            preconditions["passed"],
        ),
        *contract["checks"],
        *external_checks,
        _check(
            "v604_evidence_state_contract",
            {
                6898: "blocked",
                6899: "admissible",
                6900: "flagged",
                6901: "blocked",
                6902: "skipped",
                6903: "skipped",
                6904: "skipped",
            },
            {row["experiment_id"]: row["evidence_state"] for row in evidence["v604_artifact_rows"]},
            evidence["passed"],
        ),
    ]
    ready = all(row["passed"] for row in checks)
    checks.append(_check("v605_execution_contract_ready_score", int(ready), int(ready), True))
    summary = base.gate_summary(checks)
    hard_failures = [
        {"hard_failure": row["check"], "expected": row["expected"], "observed": row["observed"]}
        for row in checks
        if not row["passed"]
    ]
    combined_rows = [
        {"row_type": "task_contract", **deepcopy(row)}
        for row in contract["document_yaml_parity_rows"]
    ]
    combined_rows.extend(
        {"row_type": "v604_evidence_admission", **deepcopy(row)}
        for row in evidence["v604_artifact_rows"]
    )
    artifact: JsonDict = {
        "schema": "carnot.experiment_6911.v605_document_yaml_evidence_contract.v1",
        "experiment_id": 6911,
        "run_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}",
        "status": "complete" if ready else "complete_blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root, prior_source_rows),
        "rows": combined_rows,
        "document_task_rows": contract["document_task_rows"],
        "yaml_task_rows": contract["yaml_task_rows"],
        **{field: contract[field] for field in CONTRACT_ROW_FIELDS},
        "v604_artifact_rows": evidence["v604_artifact_rows"],
        "adversarial_recheck_rows": evidence["adversarial_recheck_rows"],
        "dependency_taint_rows": evidence["dependency_taint_rows"],
        "task_count": 12,
        "experiment_range": [6911, 6922],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v605_execution_contract_ready_score": int(ready),
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "null" if ready else "blocked",
        "honest_verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
        "science_claim_approved": False,
        "hard_failures": hard_failures,
        "warnings": evidence["warnings"],
        "flags": evidence["flags"],
        "blocks": evidence["blocks"],
        "skips": evidence["skips"],
        "transitive_taint": deepcopy(evidence["dependency_taint_rows"]),
    }
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute readiness without trusting the advisory score."""

    summary = artifact.get("gate_check_summary", {})
    checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
    base_checks = [
        row
        for row in checks
        if isinstance(row, Mapping) and row.get("check") != "v605_execution_contract_ready_score"
    ]
    return bool(base_checks) and all(row.get("passed") is True for row in base_checks)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute fields, advisory readiness, terminal shape, and checksum."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not REQUIRED_ARTIFACT_FIELDS.issubset(principles):
        errors.append("field_principles_missing")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    ready = int(_ready_from_artifact(artifact))
    if artifact.get("v605_execution_contract_ready_score") != ready:
        errors.append("readiness_recomputation_mismatch")
    if artifact.get("task_count") != 12 or artifact.get("experiment_range") != [6911, 6922]:
        errors.append("task_identity_summary_mismatch")
    if artifact.get("science_claim_approved") is not False:
        errors.append("advisory_contract_cannot_approve_science")
    if ready:
        if (
            artifact.get("status") != "complete"
            or artifact.get("verdict_class") != "null"
            or artifact.get("honest_verdict") != READY_VERDICT
        ):
            errors.append("ready_terminal_shape_mismatch")
    elif (
        artifact.get("status") != "complete_blocked"
        or artifact.get("verdict_class") != "blocked"
        or artifact.get("honest_verdict") != BLOCKED_VERDICT
    ):
        errors.append("blocked_terminal_shape_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _date_argument(value: str) -> str:
    """Reject ambiguous execution dates before reading evidence."""

    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Write and validate the receipt at its append-only output path."""

    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_PATH)
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    write_json_atomic(args.output, artifact)
    if errors:
        print(json.dumps({"validation_errors": errors}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ready": artifact["v605_execution_contract_ready_score"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path
    raise SystemExit(main())
