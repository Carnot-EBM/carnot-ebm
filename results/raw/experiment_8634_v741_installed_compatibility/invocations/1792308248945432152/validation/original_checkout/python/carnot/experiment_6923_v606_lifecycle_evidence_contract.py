"""Build the deterministic V606 lifecycle and evidence contract receipt.

The receipt audits roadmap metadata. It does not execute a research method or
approve a science claim. See REQ-REPORT-6923.
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
from typing import Any, Mapping, Sequence

from carnot import experiment_6898_v604_evidence_admissibility_contract as base


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6923_v606_lifecycle_evidence_contract.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_VALIDATOR_PATH = Path("scripts/validate_prior_failures.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
V606_MILESTONE = "2026.09.606"
EXPECTED_NUMBERS = tuple(range(6923, 6937))
INFRASTRUCTURE_NUMBERS = frozenset({6923, 6924})
SOTA_INGESTION_NUMBERS = frozenset({6925})
MODEL_BEARING_NUMBERS = frozenset({6927, 6933, 6934, 6935})
INDEPENDENT_ROOT_NUMBERS = (6923, 6924, 6925, 6926, 6929, 6932, 6935, 6936)
UNGATED_NUMBERS = frozenset(INDEPENDENT_ROOT_NUMBERS)
UNGATED_TAIL_NUMBER = 6936
FINAL_SENTENCE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
INFERENCE_SUBSTRATE = "deterministic_lifecycle_manifest_and_evidence_audit_no_llm"
RANDOM_SEED = 6923
BLOCKED_VERDICT = "complete_blocked_v606_lifecycle_evidence_contract"
READY_VERDICT = "complete_null_v606_lifecycle_evidence_contract_advisory"
PROMPT_SECTIONS = ("CONTEXT:", "EXISTING CODE TO READ FIRST:", "TASK:", "CONCRETE STEPS:")
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
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


def _gate(upstream: str, artifact_field: str) -> JsonDict:
    """Return one exact equality gate from the locked V606 design."""

    return {"upstream": upstream, "artifact_field": artifact_field, "op": "==", "value": 1}


EXPECTED_TASKS: tuple[JsonDict, ...] = (
    {
        "order": 1,
        "number": 6923,
        "task_id": "exp6923-v606-lifecycle-evidence-contract",
        "title": "V606 lifecycle-aware execution and evidence contract",
        "deliverable": "results/experiment_6923_v606_lifecycle_evidence_contract.json",
        "design_gate": "None; advisory root",
        "gates": [],
    },
    {
        "order": 2,
        "number": 6924,
        "task_id": "exp6924-task-runtime-receipt-adoption",
        "title": "Task-owned runtime receipt adoption",
        "deliverable": "results/experiment_6924_task_runtime_receipt_adoption.json",
        "design_gate": "None; infrastructure root",
        "gates": [],
    },
    {
        "order": 3,
        "number": 6925,
        "task_id": "exp6925-v606-sota-ingestion",
        "title": "V606 SOTA ingestion and method map",
        "deliverable": "results/experiment_6925_v606_sota_ingestion.json",
        "design_gate": "None; research root",
        "gates": [],
    },
    {
        "order": 4,
        "number": 6926,
        "task_id": "exp6926-span-first-relation-fixture",
        "title": "Exact span-first relation fixture",
        "deliverable": "results/experiment_6926_span_first_relation_fixture.json",
        "design_gate": "None; acquisition root",
        "gates": [],
    },
    {
        "order": 5,
        "number": 6927,
        "task_id": "exp6927-sota-span-first-relation-acquisition",
        "title": "Three-family span-first relation acquisition",
        "deliverable": "results/experiment_6927_sota_span_first_relation_acquisition.json",
        "design_gate": "Exp6926 `span_relation_fixture_ready_score == 1`",
        "gates": [
            _gate("exp6926-span-first-relation-fixture", "span_relation_fixture_ready_score")
        ],
    },
    {
        "order": 6,
        "number": 6928,
        "task_id": "exp6928-independent-span-relation-qualification",
        "title": "Independent span-relation qualification",
        "deliverable": "results/experiment_6928_independent_span_relation_qualification.json",
        "design_gate": "Exp6927 `span_first_acquisition_complete_score == 1`",
        "gates": [
            _gate(
                "exp6927-sota-span-first-relation-acquisition",
                "span_first_acquisition_complete_score",
            )
        ],
    },
    {
        "order": 7,
        "number": 6929,
        "task_id": "exp6929-authentic-strategy-episode-fixture",
        "title": "Authentic operational strategy-episode fixture",
        "deliverable": "results/experiment_6929_authentic_strategy_episode_fixture.json",
        "design_gate": "None; self-learning root",
        "gates": [],
    },
    {
        "order": 8,
        "number": 6930,
        "task_id": "exp6930-selective-episodic-strategy-memory",
        "title": "Selective episodic strategy memory",
        "deliverable": "results/experiment_6930_selective_episodic_strategy_memory.json",
        "design_gate": "Exp6929 `strategy_episode_fixture_ready_score == 1`",
        "gates": [
            _gate(
                "exp6929-authentic-strategy-episode-fixture",
                "strategy_episode_fixture_ready_score",
            )
        ],
    },
    {
        "order": 9,
        "number": 6931,
        "task_id": "exp6931-independent-strategy-memory-audit",
        "title": "Independent cold strategy-memory audit",
        "deliverable": "results/experiment_6931_independent_strategy_memory_audit.json",
        "design_gate": "Exp6930 `episodic_strategy_memory_run_complete_score == 1`",
        "gates": [
            _gate(
                "exp6930-selective-episodic-strategy-memory",
                "episodic_strategy_memory_run_complete_score",
            )
        ],
    },
    {
        "order": 10,
        "number": 6932,
        "task_id": "exp6932-vera-verified-headroom-fixture",
        "title": "VeRA-style verified headroom fixture",
        "deliverable": "results/experiment_6932_vera_verified_headroom_fixture.json",
        "design_gate": "None; guidance root",
        "gates": [],
    },
    {
        "order": 11,
        "number": 6933,
        "task_id": "exp6933-sota-headroom-admission-canary",
        "title": "Three-family direct headroom admission canary",
        "deliverable": "results/experiment_6933_sota_headroom_admission_canary.json",
        "design_gate": "Exp6932 `verified_variant_fixture_ready_score == 1`",
        "gates": [
            _gate(
                "exp6932-vera-verified-headroom-fixture",
                "verified_variant_fixture_ready_score",
            )
        ],
    },
    {
        "order": 12,
        "number": 6934,
        "task_id": "exp6934-reward-guided-candidate-smc",
        "title": "Reward-guided complete-candidate SMC",
        "deliverable": "results/experiment_6934_reward_guided_candidate_smc.json",
        "design_gate": "Exp6933 `headroom_admission_ready_score == 1`",
        "gates": [
            _gate(
                "exp6933-sota-headroom-admission-canary",
                "headroom_admission_ready_score",
            )
        ],
    },
    {
        "order": 13,
        "number": 6935,
        "task_id": "exp6935-arc-postfix-induction-audit",
        "title": "ARC post-fix induction reachability and held-out accuracy",
        "deliverable": "results/experiment_6935_arc_postfix_induction_audit.json",
        "design_gate": "None; ARC root",
        "gates": [],
    },
    {
        "order": 14,
        "number": 6936,
        "task_id": "exp6936-v606-independent-capstone",
        "title": "V606 independent capstone",
        "deliverable": "results/experiment_6936_v606_independent_capstone.json",
        "design_gate": "None; ungated tail",
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
    "infrastructure_slot_rows",
    "sota_ingestion_slot_rows",
    "independent_root_rows",
    "ungated_tail_rows",
    "routing_audit_update_rows",
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
    "lifecycle_resolution_rows",
    "document_task_rows",
    "yaml_task_rows",
    *CONTRACT_ROW_FIELDS,
    "task_count",
    "experiment_range",
    "random_seed",
    "reproducibility_checksum",
    "v606_execution_contract_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "science_claim_approved",
}

FIELD_PRINCIPLES = {
    "schema": "A versioned shape lets later audits reject incompatible receipts.",
    "experiment_id": "The fixed identity binds this receipt to Exp6923.",
    "run_date": "The date separates this audit from later roadmap states.",
    "status": "A terminal status distinguishes ready and blocked audits.",
    "field_principles": "A reason for each field makes the receipt understandable.",
    "preconditions_checked": "Explicit input checks prevent a silent lifecycle fallback.",
    "inference_substrate": "The no-LLM value keeps metadata separate from science.",
    "duration_s": "Measured wall time proves that the deterministic audit ran.",
    "source_artifact_hashes": "Hashes bind findings to the source bytes.",
    "rows": "One row per hard check prevents hidden omissions.",
    "lifecycle_resolution_rows": "Lifecycle rows prove which single roadmap was selected.",
    "document_task_rows": "Document rows preserve the design parse independently.",
    "yaml_task_rows": "YAML rows preserve the execution parse independently.",
    "document_yaml_parity_rows": "Parity rows expose task identity and role drift.",
    "gate_contract_rows": "Gate rows reject aliases and undeclared producer fields.",
    "required_field_rows": "Field rows prove that each prompt declares common evidence.",
    "prior_failure_contract_rows": "Prior rows prove complete retirement mechanics.",
    "model_contract_rows": "Model rows preserve current vendor and GGUF policy.",
    "prompt_ending_rows": "Prompt rows preserve sections, command, and final sentence.",
    "infrastructure_slot_rows": "Slot rows reserve exactly two infrastructure tasks.",
    "sota_ingestion_slot_rows": "The SOTA row reserves exactly one ingestion task.",
    "independent_root_rows": "Root rows prevent advisory-to-science gate coupling.",
    "ungated_tail_rows": "The tail row keeps the capstone runnable after branch nulls.",
    "routing_audit_update_rows": "Routing rows prove the standalone Codex rule is current.",
    "task_count": "The observed count exposes partial activation.",
    "experiment_range": "The observed range exposes gaps and reused task numbers.",
    "random_seed": "The fixed seed states that this audit has no hidden randomness.",
    "reproducibility_checksum": "A content hash detects silent receipt changes.",
    "v606_execution_contract_ready_score": "This advisory score cannot approve science.",
    "gate_check_summary": "Expected and observed values make each block actionable.",
    "verifier_is_oracle": "False prevents structural checks from certifying science.",
    "verdict_class": "The closed class keeps the evidence boundary machine-readable.",
    "honest_verdict": "The terminal prefix lets the conductor classify completion.",
    "science_claim_approved": "False prevents metadata readiness from promoting a claim.",
}

SOURCE_PATHS = {
    "design_document": DESIGN_PATH,
    "active_roadmap": ACTIVE_ROADMAP_PATH,
    "next_roadmap": NEXT_ROADMAP_PATH,
    "roadmap_schema": SCHEMA_PATH,
    "prior_failure_validator": PRIOR_VALIDATOR_PATH,
    "roadmap_gate_audit": GATE_AUDIT_PATH,
    "exclusion_manifest_lint": EXCLUSION_LINT_PATH,
    "exclusion_manifest": EXCLUSION_PATH,
    "spec": SPEC_PATH,
    "module": Path("python/carnot/experiment_6923_v606_lifecycle_evidence_contract.py"),
    "wrapper": Path("scripts/experiments/experiment_6923_v606_lifecycle_evidence_contract.py"),
    "focused_tests": Path("tests/python/test_experiment_6923_v606_lifecycle_evidence_contract.py"),
}
REQUIRED_INPUTS = {
    "design_document",
    "roadmap_schema",
    "prior_failure_validator",
    "roadmap_gate_audit",
    "exclusion_manifest_lint",
    "exclusion_manifest",
}

experiment_number = base.experiment_number
write_json_atomic = base.write_json_atomic


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return one diagnostic check with exact expected and observed values."""

    return {"check": name, "expected": expected, "observed": observed, "passed": bool(passed)}


def _read_text(path: Path) -> tuple[str, str | None]:
    """Read one UTF-8 input and retain a stable failure label."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return "", f"unreadable:{type(exc).__name__}"
    return (text, None) if text else ("", "empty")


def resolve_roadmap_lifecycle(root: Path) -> JsonDict:
    """Select one V606 roadmap while preserving missing and ambiguous states."""

    candidates: list[JsonDict] = []
    for role, relative in (
        ("active", ACTIVE_ROADMAP_PATH),
        ("next", NEXT_ROADMAP_PATH),
    ):
        path = root / relative
        roadmap, error = base.read_yaml_mapping(path)
        milestone = str(roadmap.get("milestone") or "") if roadmap else None
        candidates.append(
            {
                "role": role,
                "path": relative.as_posix(),
                "exists": path.is_file(),
                "read_error": error,
                "milestone": milestone,
                "claims_v606": milestone == V606_MILESTONE,
                "selected": False,
            }
        )

    claims = [row for row in candidates if row["claims_v606"]]
    unreadable = [row for row in candidates if row["exists"] and row["read_error"]]
    selected: JsonDict | None = None
    if unreadable:
        state = "unresolved"
    elif len(claims) == 2:
        state = "ambiguous"
    elif len(claims) == 1:
        selected = claims[0]
        state = "pre_staged" if selected["role"] == "next" else "activated"
        selected["selected"] = True
    elif any(row["exists"] for row in candidates):
        state = "mismatched"
    else:
        state = "missing"

    selected_path = Path(selected["path"]) if selected else None
    roadmap = base.read_yaml_mapping(root / selected_path)[0] if selected_path else None
    for row in candidates:
        row["lifecycle_state"] = state
    return {
        "state": state,
        "selected_path": selected_path,
        "roadmap": roadmap,
        "rows": candidates,
        "passed": selected is not None and roadmap is not None,
    }


def _parse_document_gate(text: str, task_ids: Mapping[int, str]) -> list[JsonDict]:
    """Translate the design table's compact gate text into YAML gate form."""

    if text.startswith("None;"):
        return []
    match = re.fullmatch(r"Exp(\d+) `([a-z][a-z0-9_]*) (==|!=|>=|<=|>|<) ([^`]+)`", text)
    if not match:
        return [{"unparsed_document_gate": text}]
    upstream_number = int(match.group(1))
    raw_value = match.group(4).strip()
    try:
        value: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return [
        {
            "upstream": task_ids.get(upstream_number, f"exp{upstream_number}-missing"),
            "artifact_field": match.group(2),
            "op": match.group(3),
            "value": value,
        }
    ]


def parse_design(text: str) -> JsonDict:
    """Parse the locked design table without consulting either roadmap YAML."""

    milestone = re.search(r"^\*\*Milestone:\*\*\s*`([^`]+)`", text, re.MULTILINE)
    if not milestone:
        raise ValueError("design milestone is required")
    row_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`(exp\d+[^`]*)`\s*\|\s*([^|]+?)\s*"
        r"\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    raw_rows = row_pattern.findall(text)
    task_ids = {
        experiment_number(task_id): task_id.strip()
        for _order, task_id, _title, _deliverable, _gate in raw_rows
        if experiment_number(task_id) in EXPECTED_NUMBERS
    }
    model_section = re.search(
        r"The live model tasks are:\s*(.*?)(?:\nThe mandatory model IDs are:)",
        text,
        re.DOTALL,
    )
    model_numbers = (
        {int(value) for value in re.findall(r"^- Exp(\d+):", model_section.group(1), re.MULTILINE)}
        if model_section
        else set()
    )
    rows: list[JsonDict] = []
    for order, task_id, title, deliverable, gate_text in raw_rows:
        number = experiment_number(task_id)
        if number not in EXPECTED_NUMBERS:
            continue
        clean_gate = gate_text.strip()
        rows.append(
            {
                "order": int(order),
                "number": number,
                "task_id": task_id.strip(),
                "title": title.strip(),
                "deliverable": deliverable.strip(),
                "milestone": milestone.group(1),
                "design_gate": clean_gate,
                "gates": _parse_document_gate(clean_gate, task_ids),
                "model_bearing": number in model_numbers,
                "infrastructure_slot": number in INFRASTRUCTURE_NUMBERS,
                "sota_ingestion_slot": number in SOTA_INGESTION_NUMBERS,
                "ungated_tail": number == UNGATED_TAIL_NUMBER
                and clean_gate == "None; ungated tail",
            }
        )
    if not rows:
        raise ValueError("design task table is required")
    return {"milestone": milestone.group(1), "tasks": rows}


def parse_roadmap(document: Any) -> JsonDict:
    """Parse executable task fields while retaining prompts and prior rows."""

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
        number = experiment_number(task.get("id"))
        rows.append(
            {
                "order": order,
                "number": number,
                "task_id": str(task.get("id") or ""),
                "title": str(task.get("title") or ""),
                "track": str(task.get("track") or ""),
                "milestone": str(task.get("milestone") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "agent_type": str(task.get("agent_type") or ""),
                "model": str(task.get("model") or ""),
                "gates": [dict(gate) if isinstance(gate, Mapping) else {} for gate in gates],
                "prior_failures": deepcopy(priors),
                "prompt": prompt,
                "required_fields": base._required_fields(prompt),
                "model_bearing": "MODEL_SPECS" in prompt,
                "infrastructure_slot": str(task.get("track") or "").lower() == "infrastructure",
                "sota_ingestion_slot": number == 6925
                and "SOTA ingestion" in str(task.get("title") or ""),
                "ungated_tail": number == UNGATED_TAIL_NUMBER and not gates,
            }
        )
    return {"milestone": str(document.get("milestone") or ""), "tasks": rows}


def current_model_rule(audit_source: str) -> tuple[str, str]:
    """Read the Codex model and blocked backend from the standalone audit."""

    codex = re.search(r'agent_type\s*==\s*"codex"\s*and\s*model\s*!=\s*"([^"]+)"', audit_source)
    forbidden_agents = re.findall(r'agent_type\s*==\s*"([^"]+)"', audit_source)
    blocked = next((agent for agent in forbidden_agents if agent != "codex"), "")
    if not codex or not blocked:
        raise ValueError("current model rules are missing from gate audit")
    return codex.group(1), blocked


def _public_document_row(row: Mapping[str, Any]) -> JsonDict:
    """Copy the short independent design parse into the receipt."""

    return deepcopy(dict(row))


def _public_yaml_row(row: Mapping[str, Any]) -> JsonDict:
    """Replace long prompt text with a stable hash in public YAML rows."""

    return {
        key: deepcopy(row.get(key))
        for key in (
            "order",
            "number",
            "task_id",
            "title",
            "track",
            "milestone",
            "deliverable",
            "agent_type",
            "model",
            "gates",
            "model_bearing",
            "infrastructure_slot",
            "sota_ingestion_slot",
            "ungated_tail",
        )
    } | {
        "prior_failure_count": len(row.get("prior_failures", [])),
        "prompt_sha256": base.sha256_bytes(str(row.get("prompt") or "").encode()),
    }


def _expected_shape(expected: Mapping[str, Any]) -> JsonDict:
    """Return the fields shared by expected, document, and YAML task rows."""

    number = int(expected["number"])
    return {
        "order": expected["order"],
        "number": number,
        "task_id": expected["task_id"],
        "title": expected["title"],
        "deliverable": expected["deliverable"],
        "milestone": V606_MILESTONE,
        "gates": deepcopy(expected["gates"]),
        "model_bearing": number in MODEL_BEARING_NUMBERS,
        "infrastructure_slot": number in INFRASTRUCTURE_NUMBERS,
        "sota_ingestion_slot": number in SOTA_INGESTION_NUMBERS,
        "ungated_tail": number == UNGATED_TAIL_NUMBER,
    }


def _observed_shape(row: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only values that the document and YAML must share exactly."""

    if row is None:
        return None
    return {
        key: deepcopy(row.get(key))
        for key in (
            "order",
            "number",
            "task_id",
            "title",
            "deliverable",
            "milestone",
            "gates",
            "model_bearing",
            "infrastructure_slot",
            "sota_ingestion_slot",
            "ungated_tail",
        )
    }


def _task_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Emit one exact document/YAML parity row for each V606 position."""

    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        document = document_tasks[index] if index < len(document_tasks) else None
        executable = yaml_tasks[index] if index < len(yaml_tasks) else None
        wanted = _expected_shape(expected)
        document_shape = _observed_shape(document)
        yaml_shape = _observed_shape(executable)
        checks = {
            "document_matches_expected": document_shape == wanted,
            "yaml_matches_expected": yaml_shape == wanted,
            "document_yaml_equal": document_shape == yaml_shape,
            "design_gate_text_exact": bool(
                document and document.get("design_gate") == expected["design_gate"]
            ),
        }
        rows.append(
            {
                "order": index + 1,
                "expected": wanted,
                "document": document_shape,
                "yaml": yaml_shape,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _prompt_rows(yaml_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Require all prompt sections, one command, and the exact final sentence."""

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
            "exact_run_command": commands == [expected_command],
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
    """Require the common evidence fields and their principle instruction."""

    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        prompt = str(task.get("prompt") or "") if task else ""
        fields = set(task.get("required_fields", set())) if task else set()
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
    """Resolve every gate to an exact earlier producer and declared field."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    yaml_by_id = {row.get("task_id"): row for row in yaml_tasks}
    rows: list[JsonDict] = []
    for expected in EXPECTED_TASKS:
        number = expected["number"]
        document = document_by_number.get(number)
        downstream = yaml_by_number.get(number)
        document_gates = list(document.get("gates", [])) if document else []
        observed_gates = list(downstream.get("gates", [])) if downstream else []
        gate_checks: list[JsonDict] = []
        for gate_index, observed in enumerate(observed_gates):
            upstream = yaml_by_id.get(observed.get("upstream"))
            field = str(observed.get("artifact_field") or "")
            wanted = expected["gates"][gate_index] if gate_index < len(expected["gates"]) else None
            checks = {
                "exact_gate": observed == wanted,
                "operator_valid": observed.get("op") in valid_operators,
                "upstream_exact_task_id": upstream is not None,
                "upstream_earlier": bool(
                    upstream
                    and downstream
                    and upstream.get("order", 0) < downstream.get("order", 0)
                ),
                "producer_field_declared": bool(
                    upstream and field in upstream.get("required_fields", set())
                ),
            }
            gate_checks.append(
                {
                    "gate_index": gate_index,
                    "expected": deepcopy(wanted),
                    "observed": deepcopy(observed),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        checks = {
            "document_gate_matches_expected": document_gates == expected["gates"],
            "yaml_gate_matches_expected": observed_gates == expected["gates"],
            "all_gate_references_valid": all(row["passed"] for row in gate_checks),
        }
        rows.append(
            {
                "downstream_number": number,
                "task_id": downstream.get("task_id") if downstream else None,
                "expected": deepcopy(expected["gates"]),
                "document": deepcopy(document_gates),
                "observed": deepcopy(observed_gates),
                "gate_checks": gate_checks,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _prior_rows(
    yaml_tasks: Sequence[Mapping[str, Any]], prior_evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Check prior-failure shape and its primary verdict evidence."""

    required = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
    rows: list[JsonDict] = []
    for task in yaml_tasks:
        for prior_index, value in enumerate(task.get("prior_failures", [])):
            prior = dict(value) if isinstance(value, Mapping) else {}
            prior_id = prior.get("experiment_id")
            primary = prior_evidence.get(str(prior_id), {})
            primary_verdict = primary.get("honest_verdict")
            failed = [field for field in required if field not in prior]
            if not isinstance(prior_id, str) or not re.fullmatch(r"exp\d+-[^\s]+", prior_id):
                failed.append("experiment_id")
            if not isinstance(prior.get("verdict"), str) or not prior.get("verdict", "").strip():
                failed.append("verdict")
            if primary_verdict != prior.get("verdict"):
                failed.append("verdict")
            if (
                not isinstance(prior.get("addressed_by"), str)
                or not prior.get("addressed_by", "").strip()
            ):
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


def _routing_ok(agent: str, model: str, codex_model: str, forbidden_agent: str) -> bool:
    """Apply the current cross-vendor routing rules to one task."""

    if agent == "codex":
        return model == codex_model
    if agent == forbidden_agent:
        return False
    if agent == "claude":
        return model in {"sonnet", "opus"}
    if agent == "opencode":
        return bool(model)
    return False


def _model_rows(
    yaml_tasks: Sequence[Mapping[str, Any]], audit_source: str
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check task routing and the four live-model prompt contracts."""

    codex_model, forbidden_agent = current_model_rule(audit_source)
    rows: list[JsonDict] = []
    for index, expected in enumerate(EXPECTED_TASKS):
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        agent = str(task.get("agent_type") or "") if task else ""
        model = str(task.get("model") or "") if task else ""
        prompt = str(task.get("prompt") or "") if task else ""
        fields = set(task.get("required_fields", set())) if task else set()
        number = expected["number"]
        llm_task = number in MODEL_BEARING_NUMBERS
        required_ids = REQUIRED_MODEL_IDS[:1] if number == 6935 else REQUIRED_MODEL_IDS
        llm_checks = {
            "all_required_model_ids": not llm_task
            or all(model_id in prompt for model_id in required_ids),
            "production_model_declared": number != 6935 or "Qwen3.8-27B" in prompt,
            "model_specs_declared": not llm_task or "model_specs" in fields,
            "gguf_native_tokenizer": not llm_task or "llama.cpp's embedded tokenizer" in prompt,
            "autotokenizer_gguf_forbidden": not llm_task
            or "Never pass a GGUF repository to AutoTokenizer.from_pretrained()." in prompt,
        }
        checks = {
            "current_agent_model_rule": bool(task)
            and _routing_ok(agent, model, codex_model, forbidden_agent),
            **llm_checks,
        }
        rows.append(
            {
                "number": number,
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
    routing_checks = {
        "codex_model_updated": codex_model == "gpt-5.6-sol",
        "gemini_still_forbidden": forbidden_agent == "gemini",
        "claude_vendor_check_retained": all(
            row["checks"]["current_agent_model_rule"]
            for row in rows
            if row["agent_type"] == "claude"
        ),
    }
    routing_rows = [
        {
            "expected_codex_model": "gpt-5.6-sol",
            "observed_codex_model": codex_model,
            "expected_forbidden_agent": "gemini",
            "observed_forbidden_agent": forbidden_agent,
            "checks": routing_checks,
            "passed": all(routing_checks.values()),
        }
    ]
    return rows, routing_rows


def _slot_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Prove two infrastructure slots and one SOTA-ingestion slot."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    actual_infrastructure = {
        row.get("number") for row in yaml_tasks if row.get("infrastructure_slot")
    }
    infrastructure_rows: list[JsonDict] = []
    for number in sorted(INFRASTRUCTURE_NUMBERS):
        document = document_by_number.get(number)
        executable = yaml_by_number.get(number)
        checks = {
            "document_slot": bool(document and document.get("infrastructure_slot")),
            "yaml_slot": bool(executable and executable.get("infrastructure_slot")),
            "exact_slot_set": actual_infrastructure == INFRASTRUCTURE_NUMBERS,
        }
        infrastructure_rows.append(
            {
                "number": number,
                "expected_slots": sorted(INFRASTRUCTURE_NUMBERS),
                "observed_slots": sorted(value for value in actual_infrastructure if value),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    sota_candidates = {
        row.get("number") for row in yaml_tasks if "SOTA ingestion" in str(row.get("title") or "")
    }
    number = 6925
    document = document_by_number.get(number)
    executable = yaml_by_number.get(number)
    sota_checks = {
        "document_slot": bool(document and document.get("sota_ingestion_slot")),
        "yaml_slot": bool(executable and executable.get("sota_ingestion_slot")),
        "exact_slot_set": sota_candidates == SOTA_INGESTION_NUMBERS,
    }
    sota_rows = [
        {
            "number": number,
            "expected_slots": sorted(SOTA_INGESTION_NUMBERS),
            "observed_slots": sorted(value for value in sota_candidates if value),
            "checks": sota_checks,
            "passed": all(sota_checks.values()),
        }
    ]
    return infrastructure_rows, sota_rows


def _root_rows(
    document_tasks: Sequence[Mapping[str, Any]], yaml_tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Prove eight ungated tasks, no advisory consumers, and one tail."""

    document_by_number = {row.get("number"): row for row in document_tasks}
    yaml_by_number = {row.get("number"): row for row in yaml_tasks}
    consumers = {
        upstream: [
            task.get("task_id")
            for task in yaml_tasks
            if any(
                experiment_number(gate.get("upstream")) == upstream
                for gate in task.get("gates", [])
            )
        ]
        for upstream in (6923, 6925)
    }
    roots: list[JsonDict] = []
    for number in INDEPENDENT_ROOT_NUMBERS:
        document = document_by_number.get(number)
        executable = yaml_by_number.get(number)
        checks = {
            "document_ungated": bool(document and not document.get("gates")),
            "yaml_ungated": bool(executable and not executable.get("gates")),
            "advisory_has_no_consumers": number not in consumers or not consumers[number],
        }
        roots.append(
            {
                "number": number,
                "yaml_consumers": consumers.get(number, []),
                "document_gates": deepcopy(document.get("gates", [])) if document else None,
                "yaml_gates": deepcopy(executable.get("gates", [])) if executable else None,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    document_tail = document_by_number.get(UNGATED_TAIL_NUMBER)
    yaml_tail = yaml_by_number.get(UNGATED_TAIL_NUMBER)
    tail_checks = {
        "document_ungated_tail": bool(
            document_tail and document_tail.get("ungated_tail") and not document_tail.get("gates")
        ),
        "yaml_ungated_tail": bool(yaml_tail and yaml_tail.get("ungated_tail")),
    }
    tails = [
        {
            "number": UNGATED_TAIL_NUMBER,
            "document_gates": deepcopy(document_tail.get("gates", [])) if document_tail else None,
            "yaml_gates": deepcopy(yaml_tail.get("gates", [])) if yaml_tail else None,
            "checks": tail_checks,
            "passed": all(tail_checks.values()),
        }
    ]
    return roots, tails


def evaluate_contract(
    design_text: str,
    roadmap: Mapping[str, Any],
    schema_source: str,
    gate_audit_source: str,
    prior_evidence: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Evaluate V606 identity, prompts, gates, priors, models, and roots."""

    document = parse_design(design_text)
    executable = parse_roadmap(roadmap)
    document_tasks = document["tasks"]
    yaml_tasks = executable["tasks"]
    parity_rows = _task_rows(document_tasks, yaml_tasks)
    prompt_rows = _prompt_rows(yaml_tasks)
    field_rows = _required_field_rows(yaml_tasks)
    gate_rows = _gate_rows(document_tasks, yaml_tasks, base.schema_gate_operators(schema_source))
    prior_rows = _prior_rows(yaml_tasks, prior_evidence)
    model_rows, routing_rows = _model_rows(yaml_tasks, gate_audit_source)
    infrastructure_rows, sota_rows = _slot_rows(document_tasks, yaml_tasks)
    root_rows, tail_rows = _root_rows(document_tasks, yaml_tasks)
    document_deliverables = [row.get("deliverable") for row in document_tasks]
    yaml_deliverables = [row.get("deliverable") for row in yaml_tasks]
    checks = [
        _check(
            "document_primary_contract",
            {"milestone": V606_MILESTONE, "task_count": 14, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": document["milestone"],
                "task_count": len(document_tasks),
                "numbers": [row.get("number") for row in document_tasks],
            },
            document["milestone"] == V606_MILESTONE
            and len(document_tasks) == 14
            and len(document_deliverables) == len(set(document_deliverables)) == 14,
        ),
        _check(
            "yaml_executable_contract",
            {"milestone": V606_MILESTONE, "task_count": 14, "numbers": list(EXPECTED_NUMBERS)},
            {
                "milestone": executable["milestone"],
                "task_count": len(yaml_tasks),
                "numbers": [row.get("number") for row in yaml_tasks],
            },
            executable["milestone"] == V606_MILESTONE
            and len(yaml_tasks) == 14
            and len({row.get("task_id") for row in yaml_tasks}) == 14
            and len(yaml_deliverables) == len(set(yaml_deliverables)) == 14,
        ),
    ]
    row_groups = (
        ("document_yaml_task_contract", parity_rows),
        ("prompt_contract", prompt_rows),
        ("required_field_contract", field_rows),
        ("gate_contract", gate_rows),
        ("prior_failure_contract", prior_rows),
        ("model_contract", model_rows),
        ("infrastructure_slot_contract", infrastructure_rows),
        ("sota_ingestion_slot_contract", sota_rows),
        ("independent_root_contract", root_rows),
        ("ungated_tail_contract", tail_rows),
        ("routing_audit_update_contract", routing_rows),
    )
    for name, rows in row_groups:
        passed = all(row["passed"] for row in rows)
        checks.append(_check(name, True, passed, passed))
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "document_task_rows": [_public_document_row(row) for row in document_tasks],
        "yaml_task_rows": [_public_yaml_row(row) for row in yaml_tasks],
        "document_yaml_parity_rows": parity_rows,
        "gate_contract_rows": gate_rows,
        "required_field_rows": field_rows,
        "prior_failure_contract_rows": prior_rows,
        "model_contract_rows": model_rows,
        "prompt_ending_rows": prompt_rows,
        "infrastructure_slot_rows": infrastructure_rows,
        "sota_ingestion_slot_rows": sota_rows,
        "independent_root_rows": root_rows,
        "ungated_tail_rows": tail_rows,
        "routing_audit_update_rows": routing_rows,
    }


def _empty_contract(error: str) -> JsonDict:
    """Return all contract row families when source parsing cannot start."""

    return {
        "passed": False,
        "checks": [_check("contract_evaluation", "valid primary records", error, False)],
        "document_task_rows": [],
        "yaml_task_rows": [],
        **{field: [] for field in CONTRACT_ROW_FIELDS},
    }


def _isolated_subprocess_env() -> dict[str, str]:
    """Keep parent coverage controls out of validation subprocesses."""

    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_"))
    }


def _load_prior_evidence(
    root: Path, tasks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Load each declared prior verdict from its primary result artifact."""

    evidence: dict[str, JsonDict] = {}
    source_rows: list[JsonDict] = []
    for task in tasks:
        for prior in task.get("prior_failures", []) or []:
            if not isinstance(prior, Mapping):
                continue
            prior_id = str(prior.get("experiment_id") or "")
            number = experiment_number(prior_id)
            matches = (
                sorted((root / "results").glob(f"experiment_{number}_*.json")) if number else []
            )
            path = matches[0] if len(matches) == 1 else None
            payload, error = base.read_json_mapping(path) if path else (None, "artifact_not_unique")
            if payload is not None and path is not None:
                evidence[prior_id] = {
                    "honest_verdict": payload.get("honest_verdict"),
                    "source": path.relative_to(root).as_posix(),
                }
            source_rows.append(
                {
                    "experiment_id": prior_id,
                    "path": path.relative_to(root).as_posix() if path else None,
                    "sha256": base.sha256_path(path) if path else None,
                    "error": error,
                }
            )
    return evidence, source_rows


def _external_validation_checks(
    root: Path, roadmap_path: Path, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Run current schema, prior, gate, and exclusion validators."""

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
    exclusion_output = exclusion.stdout.strip() or exclusion.stderr.strip()
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
            {"exit_code": exclusion.returncode, "output": exclusion_output},
            exclusion.returncode == 0,
        ),
    ]


def _source_hashes(root: Path, prior_rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Bind lifecycle, validator, implementation, and prior inputs to bytes."""

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
        _checksum_value(dict(artifact)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def build_artifact(root: Path, run_date: str) -> JsonDict:
    """Build a complete advisory or blocked V606 lifecycle receipt."""

    started = time.perf_counter()
    texts: dict[str, str] = {}
    source_errors: list[JsonDict] = []
    for name in REQUIRED_INPUTS:
        relative = SOURCE_PATHS[name]
        texts[name], error = _read_text(root / relative)
        if error:
            source_errors.append({"source": name, "error": error})

    lifecycle = resolve_roadmap_lifecycle(root)
    roadmap = lifecycle["roadmap"] or {}
    selected_path = lifecycle["selected_path"]
    raw_tasks = roadmap.get("tasks", []) if isinstance(roadmap, Mapping) else []
    prior_evidence, prior_sources = _load_prior_evidence(
        root, [task for task in raw_tasks if isinstance(task, Mapping)]
    )
    try:
        contract = evaluate_contract(
            texts.get("design_document", ""),
            roadmap,
            texts.get("roadmap_schema", ""),
            texts.get("roadmap_gate_audit", ""),
            prior_evidence,
        )
    except (ValueError, SyntaxError) as exc:
        contract = _empty_contract(str(exc))

    precondition_checks = [
        _check("required_inputs", [], source_errors, not source_errors),
        _check(
            "lifecycle_resolution",
            "exactly one V606 roadmap",
            {
                "state": lifecycle["state"],
                "selected_path": selected_path.as_posix() if selected_path else None,
            },
            lifecycle["passed"],
        ),
    ]
    preconditions = base.gate_summary(precondition_checks)
    if selected_path and roadmap:
        external_checks = _external_validation_checks(root, root / selected_path, roadmap)
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
    checks = [
        _check(
            "preconditions",
            "all required inputs and one lifecycle roadmap",
            preconditions,
            preconditions["passed"],
        ),
        *contract["checks"],
        *external_checks,
    ]
    ready = all(row["passed"] for row in checks)
    checks.append(_check("v606_execution_contract_ready_score", int(ready), int(ready), True))
    parsed_numbers = [
        row.get("number") for row in contract["yaml_task_rows"] if row.get("number") is not None
    ]
    experiment_range = [min(parsed_numbers), max(parsed_numbers)] if parsed_numbers else []
    artifact: JsonDict = {
        "schema": "carnot.experiment_6923.v606_lifecycle_evidence_contract.v1",
        "experiment_id": 6923,
        "run_date": f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:8]}",
        "status": "complete" if ready else "complete_blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root, prior_sources),
        "rows": deepcopy(checks),
        "lifecycle_resolution_rows": deepcopy(lifecycle["rows"]),
        "document_task_rows": contract["document_task_rows"],
        "yaml_task_rows": contract["yaml_task_rows"],
        **{field: contract[field] for field in CONTRACT_ROW_FIELDS},
        "task_count": len(contract["yaml_task_rows"]),
        "experiment_range": experiment_range,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "v606_execution_contract_ready_score": int(ready),
        "gate_check_summary": base.gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "null" if ready else "blocked",
        "honest_verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
        "science_claim_approved": False,
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
        if isinstance(row, Mapping) and row.get("check") != "v606_execution_contract_ready_score"
    ]
    return bool(base_checks) and all(row.get("passed") is True for row in base_checks)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute identity, readiness, terminal shape, and checksum fields."""

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
    if artifact.get("v606_execution_contract_ready_score") != ready:
        errors.append("readiness_recomputation_mismatch")
    yaml_rows = artifact.get("yaml_task_rows", [])
    yaml_rows = yaml_rows if isinstance(yaml_rows, list) else []
    numbers = [
        row.get("number")
        for row in yaml_rows
        if isinstance(row, Mapping) and isinstance(row.get("number"), int)
    ]
    expected_range = [min(numbers), max(numbers)] if numbers else []
    if (
        artifact.get("task_count") != len(yaml_rows)
        or artifact.get("experiment_range") != expected_range
    ):
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
    """Write and validate the receipt at its requested output path."""

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
                "ready": artifact["v606_execution_contract_ready_score"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this path
    raise SystemExit(main())
