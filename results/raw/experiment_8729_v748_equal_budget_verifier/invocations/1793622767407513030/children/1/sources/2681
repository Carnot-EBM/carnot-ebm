"""Audit the V623 Markdown contract against the active execution YAML.

The Markdown and YAML parsers receive different source values. This separation
makes a roadmap mismatch visible. Missing inputs block the audit. A readable
contract mismatch disqualifies it without gating any science task.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
STAGING_ROADMAP_PATH = Path("research-roadmap-next.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HARNESS_SPEC_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
PROGRAM_PATH = Path("research-program.md")
CLAUDE_PATH = Path("CLAUDE.md")
CODEX_PATH = Path("CODEX.md")
ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
ROADMAP_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
HARNESS_LINT_PATH = Path("scripts/harness_fit_lint.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
EXP7091_PATH = Path("results/experiment_7091_v622_contract_preflight.json")
EXP7084_PATH = Path("results/experiment_7084_v621_contract_preflight.json")
EXP7076_PATH = Path("results/experiment_7076_v620_contract_preflight.json")
EXP7091_MODULE_PATH = Path("python/carnot/experiment_7091_v622_contract_preflight.py")
MODULE_PATH = Path("python/carnot/experiment_7097_v623_contract_preflight.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7097_v623_contract_preflight.py")
TEST_PATH = Path("tests/python/test_experiment_7097_v623_contract_preflight.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7097_v623_contract_preflight.json")

MILESTONE = "2026.09.623"
EXPECTED_TASK_IDS = (
    "exp7097-v623-contract-preflight",
    "exp7098-v623-execution-sota-ingestion",
    "exp7099-adapter-withheld-live-path-preflight",
    "exp7100-adapter-withheld-arc-loo-measurement",
    "exp7101-adapter-withheld-arc-cold-audit",
    "exp7102-feasibility-projected-action-energy",
    "exp7103-adapter-withheld-energy-live-ab",
    "exp7104-degree16-action-energy-portability",
    "exp7105-sealed-exact-constraint-stream",
    "exp7106-delayed-commit-procedural-memory-csl",
    "exp7107-continual-memory-cold-audit",
    "exp7108-v623-capstone",
)
EXPECTED_TASK_COUNT = 12
EXPECTED_GATE_COUNT = 8
EXPECTED_PRIOR_COUNT = 22
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts: independent Markdown and YAML contract replay"
)
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
PROMPT_FINAL_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
RANDOM_SEED = 709720260907

HEADLINE_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF"
ARC_LIVE_MODEL = "unsloth/Qwen3.8-27B-GGUF"
LLM_GENERATING_TASKS = frozenset({7099, 7100, 7103})
ARC_LIVE_GENERATING_TASKS = frozenset({7099, 7100, 7103})
MODEL_SUBSTRATE_CLASSES = frozenset({"model_bounded_generation", "model_full_generation"})
LEGAL_SUBSTRATE_CLASSES = frozenset(
    {
        "aggregation",
        "no_model_load",
        "model_load_no_generation",
        "model_bounded_generation",
        "model_full_generation",
        "blocked_no_run",
    }
)
EXPECTED_SUBSTRATE_CLASSES = {
    7097: "aggregation",
    7098: "no_model_load",
    7099: "model_full_generation",
    7100: "model_full_generation",
    7101: "aggregation",
    7102: "aggregation",
    7103: "model_full_generation",
    7104: "no_model_load",
    7105: "no_model_load",
    7106: "no_model_load",
    7107: "aggregation",
    7108: "aggregation",
}
RUN_SCRIPTS = {
    7097: "experiment_7097_v623_contract_preflight.py",
    7098: "experiment_7098_v623_sota_ingestion.py",
    7099: "experiment_7099_v623_adapter_withheld_preflight.py",
    7100: "experiment_7100_v623_adapter_withheld_loo.py",
    7101: "experiment_7101_v623_adapter_withheld_cold_audit.py",
    7102: "experiment_7102_v623_feasibility_action_energy.py",
    7103: "experiment_7103_v623_adapter_withheld_energy_live_ab.py",
    7104: "experiment_7104_v623_degree16_action_energy_portability.py",
    7105: "experiment_7105_v623_exact_constraint_stream.py",
    7106: "experiment_7106_v623_procedural_memory_csl.py",
    7107: "experiment_7107_v623_continual_memory_cold_audit.py",
    7108: "experiment_7108_v623_capstone.py",
}
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)

BLOCKED_VERDICT = "complete_blocked_v623_contract_preflight_prerequisite_missing"
POSITIVE_VERDICT = "complete_positive_v623_task_contract_conforms"
DISQUALIFIED_VERDICT = "complete_disqualified_v623_markdown_yaml_contract_mismatch"

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "source_artifact_hashes",
        "rows",
        "markdown_task_rows",
        "yaml_task_rows",
        "task_contract_rows",
        "title_parity_rows",
        "deliverable_parity_rows",
        "gate_contract_rows",
        "gate_producer_rows",
        "prior_failure_rows",
        "model_compliance_rows",
        "agent_routing_rows",
        "substrate_class_rows",
        "execution_venue_rows",
        "artifact_field_rows",
        "prompt_tail_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "active_roadmap_path",
        "staging_file_required_at_execution",
        "v623_task_contract_conforms_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)
CONTRACT_ROW_FIELDS = (
    "title_parity_rows",
    "deliverable_parity_rows",
    "gate_contract_rows",
    "gate_producer_rows",
    "prior_failure_rows",
    "model_compliance_rows",
    "agent_routing_rows",
    "substrate_class_rows",
    "execution_venue_rows",
    "artifact_field_rows",
    "prompt_tail_rows",
)
TASK_REQUIRED_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
TASK_METADATA_FIELDS = (
    "track",
    "priority",
    "agent_type",
    "model",
    "requires_gpu",
    "max_turns",
    "estimated_wall_time_min",
    "per_unit_rows",
)


def task_number(task_id: object) -> int | None:
    """Return the numeric experiment prefix without trusting its suffix."""

    match = re.match(r"exp(\d+)(?:-|$)", str(task_id), re.I)
    return int(match.group(1)) if match else None


def _normal_exp_id(value: object) -> str | None:
    """Normalize full or numeric experiment identities for comparisons."""

    candidate = f"exp{value}" if isinstance(value, int) else value
    number = task_number(candidate)
    return f"exp{number}" if number is not None else None


def _parse_scalar(text: str) -> Any:
    """Parse one gate value while rejecting hidden structured values."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("structured gate value must be scalar")
    return value


def _parse_markdown_gate_cell(
    cell: str, full_id_by_number: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Parse a Markdown gate without consulting any YAML task."""

    if cell.strip().lower() == "none":
        return []
    pattern = re.compile(
        r"^(exp\d+(?:-[A-Za-z0-9_-]+)?)\.([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(==|!=|>=|<=|>|<)\s*(.+)$",
        re.I,
    )
    gates: list[dict[str, Any]] = []
    for expression in re.split(r"\s*(?:;|\s+AND\s+)\s*", cell, flags=re.I):
        match = pattern.fullmatch(expression.strip().strip("`"))
        if match is None:
            raise ValueError(f"malformed Markdown structured gate: {cell}")
        upstream_number = task_number(match.group(1))
        upstream = full_id_by_number.get(upstream_number) if upstream_number is not None else None
        if upstream is None:
            upstream = match.group(1)
        gates.append(
            {
                "upstream": upstream,
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4).strip().strip("`")),
            }
        )
    return gates


def parse_markdown_contract(text: str) -> dict[str, Any]:
    """Parse the Markdown table and failure history without YAML objects."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\s]+)`?", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    section_match = re.search(
        r"^## Exact Task Contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.MULTILINE
    )
    if section_match is None:
        raise ValueError("Markdown task contract section is missing")

    tasks: list[dict[str, Any]] = []
    gate_cells: list[str] = []
    for line in section_match.group(1).splitlines():
        if re.match(r"^\|\s*\d+\s*\|", line) is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError(f"malformed Markdown task row: {line}")
        order_text, task_cell, title, deliverable_cell, gate_cell = cells
        task_id = task_cell.strip("`")
        number = task_number(task_id)
        deliverable = deliverable_cell.strip("`")
        order = int(order_text)
        if number is None or not title or not deliverable:
            raise ValueError(f"malformed Markdown task row: {line}")
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "number": number,
                "title": title,
                "deliverable": deliverable,
                "milestone": milestone_match.group(1),
                "gates": [],
                "prior_failure_ids": [],
            }
        )
        gate_cells.append(gate_cell)
    if not tasks:
        raise ValueError("Markdown task table is missing")

    full_id_by_number = {task["number"]: task["id"] for task in tasks}
    for task, gate_cell in zip(tasks, gate_cells, strict=True):
        task["gates"] = _parse_markdown_gate_cell(gate_cell, full_id_by_number)

    tasks_by_number = {task["number"]: task for task in tasks}
    for match in re.finditer(
        r"^### Exp(\d+)\s+-[^\n]*\n([\s\S]*?)(?=^### |^## |\Z)", text, re.MULTILINE
    ):
        number = int(match.group(1))
        prior_match = re.search(r"\*\*Prior failures:\*\*([\s\S]*?)(?=\n\*\*|\Z)", match.group(2))
        if number not in tasks_by_number or prior_match is None:
            continue
        failure_text = re.split(
            r"\b(?:V623|This attempt)\b", prior_match.group(1), maxsplit=1, flags=re.I
        )[0]
        prior_ids: list[str] = []
        for candidate in re.findall(r"\bExp\d+(?:-[A-Za-z0-9_-]+)?", failure_text, re.I):
            normalized = _normal_exp_id(candidate)
            if normalized is not None and normalized not in prior_ids:
                prior_ids.append(normalized)
        tasks_by_number[number]["prior_failure_ids"] = prior_ids
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def _required_block(prompt: str) -> str:
    """Isolate declarations so ordinary prose cannot satisfy field checks."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def parse_yaml_contract(document: object) -> dict[str, Any]:
    """Parse YAML through a path that never receives Markdown task rows."""

    if not isinstance(document, Mapping):
        raise ValueError("YAML roadmap mapping is required")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("YAML roadmap tasks list is required")
    tasks: list[dict[str, Any]] = []
    for order, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"YAML task {order} must be a mapping")
        raw_gates = raw_task.get("gated_on") or []
        raw_priors = raw_task.get("prior_failures") or []
        if not isinstance(raw_gates, list) or not isinstance(raw_priors, list):
            raise ValueError(f"YAML task {order} has a malformed list")
        gates: list[dict[str, Any]] = []
        for gate in raw_gates:
            if not isinstance(gate, Mapping):
                raise ValueError(f"YAML task {order} has a malformed gate")
            gates.append(
                {
                    "upstream": gate.get("upstream"),
                    "artifact_field": gate.get("artifact_field"),
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                }
            )
        prompt = str(raw_task.get("prompt", ""))
        task_id = str(raw_task.get("id") or "")
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "number": task_number(task_id),
                "title": raw_task.get("title"),
                "deliverable": raw_task.get("deliverable"),
                "milestone": raw_task.get("milestone", document.get("milestone")),
                **{field: raw_task.get(field) for field in TASK_METADATA_FIELDS},
                "gates": gates,
                "prior_failures": list(raw_priors),
                "prompt": prompt,
                "required_block": _required_block(prompt),
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a non-empty YAML mapping and reject every other shape."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document:
        raise ValueError(f"YAML mapping required at {path}")
    return document


def retired_experiment_ids(document: object) -> set[str]:
    """Collect retired IDs while honoring explicit restoration records."""

    if not isinstance(document, Mapping):
        return set()
    retired: set[str] = set()
    restored: set[str] = set()
    for key, value in document.items():
        if not str(key).startswith("retired") or not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            candidates = [item.get("experiment_id")]
            if isinstance(item.get("experiment_ids"), list):
                candidates.extend(item["experiment_ids"])
            retired.update(
                normalized
                for candidate in candidates
                if (normalized := _normal_exp_id(candidate)) is not None
            )
            if isinstance(item.get("un_retired_experiment_ids"), list):
                restored.update(
                    normalized
                    for candidate in item["un_retired_experiment_ids"]
                    if (normalized := _normal_exp_id(candidate)) is not None
                )
    return retired - restored


def _canonical_gates(gates: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Use ordered tuples so mapping key order cannot hide gate drift."""

    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
    ]


def _substrate_class(prompt: str) -> str | None:
    """Read the planned compute class before its blocked alternative."""

    match = re.search(r"inference_substrate_class\s*\(\s*([A-Za-z_]+)", prompt)
    return match.group(1) if match else None


def _execution_venue(prompt: str) -> str | None:
    """Read the venue from the required-fields declaration."""

    match = re.search(r"execution_venue\s*\(\s*([A-Za-z_]+)\s*\)", prompt)
    return match.group(1) if match else None


def _field_declared(block: str, field: str) -> bool:
    """Accept one bare field and reject a nested field with the same prefix."""

    pattern = rf"(?<![A-Za-z0-9_.]){re.escape(field)}(?![A-Za-z0-9_.])"
    return re.search(pattern, block) is not None


def _closed_verdict_enum(block: str) -> bool:
    """Require the six allowed verdict classes in their declared order."""

    normalized = re.sub(r"\s+", " ", block)
    values = r"\s*\|\s*".join(map(re.escape, VERDICT_CLASSES))
    return re.search(rf"verdict_class\s*\(\s*{values}\s*\)", normalized) is not None


def _comparison_prompt(prompt: str) -> bool:
    """Find comparisons that require preserved per-unit rows."""

    return (
        re.search(
            r"\b(?:compare|comparison|comparative|controls?|arms?|parity|paired|versus|A/B)\b",
            prompt,
            re.I,
        )
        is not None
    )


def _task_field_checks(task: Mapping[str, Any]) -> dict[str, bool]:
    """Check executable task metadata before prompt-specific policy."""

    return {
        "track": isinstance(task.get("track"), str) and bool(task["track"].strip()),
        "priority": isinstance(task.get("priority"), str) and bool(task["priority"].strip()),
        "agent_type": task.get("agent_type") == "codex",
        "model": task.get("model") == "gpt-5.6-sol",
        "requires_gpu": isinstance(task.get("requires_gpu"), bool),
        "max_turns": isinstance(task.get("max_turns"), int) and task["max_turns"] > 0,
        "estimated_wall_time_min": isinstance(task.get("estimated_wall_time_min"), int)
        and task["estimated_wall_time_min"] > 0,
        "per_unit_rows": isinstance(task.get("per_unit_rows"), bool),
    }


def _legacy_small_headline_substitution(prompt: str) -> bool:
    """Reject a small legacy model only when prose gives it a headline role."""

    legacy_models = ("qwen3.5-0.8b", "gemma-4-e4b", "gemma4-e4b", "legacy-small")
    for line in prompt.lower().splitlines():
        if not any(model in line for model in legacy_models):
            continue
        if not any(role in line for role in ("headline", "fallback", "substitut", "replace")):
            continue
        if not any(negation in line for negation in ("no ", "not ", "never", "forbid")):
            return True
    return False


def _model_compliance_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Check model execution only for tasks that actually generate with an LLM."""

    number = task["number"]
    prompt = task["prompt"]
    normalized = re.sub(r"\s+", " ", prompt)
    lowered = normalized.lower()
    substrate_class = _substrate_class(prompt)
    scheduled_generation = substrate_class in MODEL_SUBSTRATE_CLASSES
    expected_generation = number in LLM_GENERATING_TASKS
    model_specs_declared = _field_declared(task["required_block"], "MODEL_SPECS")
    specs_execute = re.search(r"MODEL_SPECS\b.{0,350}\bexecute\b", normalized, re.I) is not None
    headline_position = lowered.find(HEADLINE_MODEL.lower())
    headline_context = (
        lowered[max(0, headline_position - 180) : headline_position + 220]
        if headline_position >= 0
        else ""
    )
    headline_cell = headline_position >= 0 and "headline" in headline_context
    arc_position = lowered.find(ARC_LIVE_MODEL.lower())
    arc_context = (
        lowered[max(0, arc_position - 180) : arc_position + 220] if arc_position >= 0 else ""
    )
    arc_required = number in ARC_LIVE_GENERATING_TASKS
    arc_cell = arc_position >= 0 and any(
        marker in arc_context for marker in ("primary arm", "live-generator arm", "live generator")
    )
    legacy_substitution = _legacy_small_headline_substitution(prompt)
    if expected_generation:
        passed = bool(
            scheduled_generation
            and task["requires_gpu"] is True
            and model_specs_declared
            and specs_execute
            and headline_cell
            and (not arc_required or arc_cell)
            and not legacy_substitution
        )
    else:
        passed = not scheduled_generation and task["requires_gpu"] is False
    return {
        "order": task["order"],
        "number": number,
        "task_id": task["id"],
        "scheduled_llm_generation": scheduled_generation,
        "expected_llm_generation": expected_generation,
        "model_specs_declared": model_specs_declared,
        "model_specs_execute_declared": specs_execute,
        "headline_model": HEADLINE_MODEL if headline_position >= 0 else None,
        "headline_cell_declared": headline_cell,
        "arc_live_model_required": arc_required,
        "arc_live_model": ARC_LIVE_MODEL if arc_position >= 0 else None,
        "arc_live_cell_declared": arc_cell,
        "legacy_small_headline_substitution": legacy_substitution,
        "passed": passed,
    }


def _prior_rows(task: Mapping[str, Any], expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Check all four prior fields and compare Markdown failure identities."""

    priors = task["prior_failures"]
    observed_ids = [
        _normal_exp_id(prior.get("experiment_id")) if isinstance(prior, Mapping) else None
        for prior in priors
    ]
    identities_match = Counter(observed_ids) == Counter(expected_ids)
    if not priors:
        return [
            {
                "order": task["order"],
                "number": task["number"],
                "task_id": task["id"],
                "prior_failure_present": False,
                "expected_prior_failure_ids": list(expected_ids),
                "observed_prior_failure_ids": observed_ids,
                "identity_parity": identities_match,
                "passed": identities_match,
            }
        ]
    rows: list[dict[str, Any]] = []
    for prior in priors:
        values = dict(prior) if isinstance(prior, Mapping) else {}
        checks = {
            "experiment_id": isinstance(values.get("experiment_id"), str)
            and bool(values["experiment_id"].strip()),
            "verdict": isinstance(values.get("verdict"), str) and bool(values["verdict"].strip()),
            "addressed_by": isinstance(values.get("addressed_by"), str)
            and bool(values["addressed_by"].strip()),
            "retire_if_same_verdict": values.get("retire_if_same_verdict") is True,
        }
        rows.append(
            {
                "order": task["order"],
                "number": task["number"],
                "task_id": task["id"],
                "prior_failure_present": True,
                "experiment_id": values.get("experiment_id"),
                "expected_prior_failure_ids": list(expected_ids),
                "observed_prior_failure_ids": observed_ids,
                "identity_parity": identities_match,
                "field_checks": checks,
                "passed": identities_match and all(checks.values()),
            }
        )
    return rows


def _missing_policy_row(order: int, number: int | None, task_id: str | None) -> dict[str, Any]:
    """Keep each evidence table aligned when a YAML row is absent."""

    return {"order": order, "number": number, "task_id": task_id, "passed": False}


def _expected_prompt_tail(number: int | None) -> list[str] | None:
    """Return the two execution lines expected for one task."""

    script = RUN_SCRIPTS.get(number)
    if script is None:
        return None
    return [
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        f"{script} --date {{date}}",
        PROMPT_FINAL_LINE,
    ]


def evaluate_contract(
    markdown_text: str, yaml_document: object, retired_ids: set[str]
) -> dict[str, Any]:
    """Compare every V623 row and preserve each independent policy check."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    normalized_retired = {
        normalized for value in retired_ids if (normalized := _normal_exp_id(value)) is not None
    }
    number_counts = Counter(task["number"] for task in yaml_tasks)
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))

    title_rows: list[dict[str, Any]] = []
    deliverable_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    substrate_rows: list[dict[str, Any]] = []
    venue_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    id_checks: dict[int, bool] = {}

    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        task_id = (markdown_task or yaml_task or {}).get("id")
        number = (markdown_task or yaml_task or {}).get("number")
        same_identity = bool(
            markdown_task
            and yaml_task
            and markdown_task["id"] == yaml_task["id"]
            and markdown_task["order"] == yaml_task["order"] == order
        )
        title_rows.append(
            {
                "order": order,
                "number": number,
                "task_id": task_id,
                "expected": markdown_task.get("title") if markdown_task else None,
                "observed": yaml_task.get("title") if yaml_task else None,
                "passed": bool(same_identity and markdown_task["title"] == yaml_task["title"]),
            }
        )
        deliverable_rows.append(
            {
                "order": order,
                "number": number,
                "task_id": task_id,
                "expected": markdown_task.get("deliverable") if markdown_task else None,
                "observed": yaml_task.get("deliverable") if yaml_task else None,
                "passed": bool(
                    same_identity and markdown_task["deliverable"] == yaml_task["deliverable"]
                ),
            }
        )
        gate_rows.append(
            {
                "order": order,
                "number": number,
                "task_id": task_id,
                "expected": markdown_task.get("gates") if markdown_task else None,
                "observed": yaml_task.get("gates") if yaml_task else None,
                "passed": bool(
                    same_identity
                    and _canonical_gates(markdown_task["gates"])
                    == _canonical_gates(yaml_task["gates"])
                ),
            }
        )
        if yaml_task is None:
            missing = _missing_policy_row(order, number, task_id)
            prior_rows.append({**missing, "prior_failure_present": False})
            model_rows.append({**missing, "scheduled_llm_generation": None})
            routing_rows.append({**missing, "observed_route": None})
            substrate_rows.append({**missing, "observed_class": None})
            venue_rows.append({**missing, "observed_venue": None})
            artifact_rows.append({**missing, "comparison": None})
            tail_rows.append(
                {**missing, "expected": _expected_prompt_tail(number), "observed": None}
            )
            id_checks[order] = False
            continue

        expected_priors = markdown_task.get("prior_failure_ids", []) if markdown_task else []
        prior_rows.extend(_prior_rows(yaml_task, expected_priors))
        normalized_id = _normal_exp_id(yaml_task["id"])
        unique_id = yaml_task["number"] is not None and number_counts[yaml_task["number"]] == 1
        absent_from_retired = normalized_id not in normalized_retired
        id_checks[order] = unique_id and absent_from_retired

        prompt = yaml_task["prompt"]
        block = yaml_task["required_block"]
        substrate_class = _substrate_class(prompt)
        expected_class = EXPECTED_SUBSTRATE_CLASSES.get(yaml_task["number"])
        blocked_alternative = "blocked_no_run" in block
        substrate_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "observed_class": substrate_class,
                "expected_class": expected_class,
                "class_is_closed": substrate_class in LEGAL_SUBSTRATE_CLASSES,
                "blocked_no_run_declared": blocked_alternative,
                "passed": bool(
                    substrate_class == expected_class
                    and substrate_class in LEGAL_SUBSTRATE_CLASSES
                    and blocked_alternative
                ),
            }
        )
        venue = _execution_venue(prompt)
        venue_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "expected_venue": EXECUTION_VENUE,
                "observed_venue": venue,
                "passed": venue == EXECUTION_VENUE,
            }
        )
        model_rows.append(_model_compliance_row(yaml_task))

        field_checks = _task_field_checks(yaml_task)
        observed_route = [yaml_task["agent_type"], yaml_task["model"]]
        routing_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "observed_route": observed_route,
                "expected_route": ["codex", "gpt-5.6-sol"],
                "task_field_checks": field_checks,
                "passed": all(field_checks.values()),
            }
        )

        normalized_block = re.sub(r"\s+", " ", block.lower())
        diagnostics = all(
            phrase in normalized_block
            for phrase in ("failed check", "expected value", "observed value")
        )
        comparison = _comparison_prompt(prompt)
        capstone_ungated = not yaml_task["gates"] if yaml_task["number"] == 7108 else None
        advisory_ungated = not yaml_task["gates"] if yaml_task["number"] == 7097 else None
        generic_fields = all(_field_declared(block, field) for field in TASK_REQUIRED_FIELDS)
        artifact_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "comparison": comparison,
                "per_unit_rows": yaml_task["per_unit_rows"],
                "generic_fields_present": generic_fields,
                "blocked_diagnostics_present": diagnostics,
                "closed_verdict_enum": _closed_verdict_enum(block),
                "unique_id": unique_id,
                "absent_from_retired_ids": absent_from_retired,
                "capstone_ungated": capstone_ungated,
                "advisory_preflight_ungated": advisory_ungated,
                "passed": bool(
                    block
                    and generic_fields
                    and diagnostics
                    and _closed_verdict_enum(block)
                    and (not comparison or yaml_task["per_unit_rows"] is True)
                    and capstone_ungated is not False
                    and advisory_ungated is not False
                    and unique_id
                    and absent_from_retired
                ),
            }
        )
        observed_tail = prompt.rstrip().splitlines()[-2:] if prompt.rstrip() else []
        expected_tail = _expected_prompt_tail(yaml_task["number"])
        tail_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "expected": expected_tail,
                "observed": observed_tail,
                "passed": observed_tail == expected_tail,
            }
        )

    yaml_by_id = {task["id"]: task for task in yaml_tasks}
    gate_producer_rows: list[dict[str, Any]] = []
    for consumer in yaml_tasks:
        for gate in consumer["gates"]:
            upstream_id = gate.get("upstream")
            producer = yaml_by_id.get(upstream_id)
            field = gate.get("artifact_field")
            bare_field = (
                isinstance(field, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field) is not None
            )
            producer_precedes = bool(producer and producer["order"] < consumer["order"])
            same_milestone = bool(
                producer and producer["milestone"] == consumer["milestone"] == MILESTONE
            )
            field_declared = bool(
                producer and bare_field and _field_declared(producer["required_block"], field)
            )
            upstream_forbidden = _normal_exp_id(upstream_id) in normalized_retired
            advisory_preflight_used = task_number(upstream_id) == 7097
            gate_producer_rows.append(
                {
                    "order": consumer["order"],
                    "number": consumer["number"],
                    "consumer": consumer["id"],
                    "upstream": upstream_id,
                    "artifact_field": field,
                    "producer_present": producer is not None,
                    "producer_precedes_consumer": producer_precedes,
                    "same_milestone": same_milestone,
                    "bare_field": bare_field,
                    "field_declared": field_declared,
                    "upstream_forbidden": upstream_forbidden,
                    "advisory_preflight_used": advisory_preflight_used,
                    "passed": bool(
                        producer
                        and producer_precedes
                        and same_milestone
                        and bare_field
                        and field_declared
                        and not upstream_forbidden
                        and not advisory_preflight_used
                    ),
                }
            )

    supporting_rows = (
        title_rows,
        deliverable_rows,
        gate_rows,
        prior_rows,
        model_rows,
        routing_rows,
        substrate_rows,
        venue_rows,
        artifact_rows,
        tail_rows,
        gate_producer_rows,
    )
    checks_by_order: dict[int, list[bool]] = {}
    for evidence_rows in supporting_rows:
        for row in evidence_rows:
            checks_by_order.setdefault(row["order"], []).append(row["passed"] is True)

    task_rows: list[dict[str, Any]] = []
    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        expected_id = EXPECTED_TASK_IDS[index] if index < EXPECTED_TASK_COUNT else None
        task_rows.append(
            {
                "order": order,
                "expected_id": expected_id,
                "markdown_id": markdown_task.get("id") if markdown_task else None,
                "observed_id": yaml_task.get("id") if yaml_task else None,
                "expected_milestone": markdown_task.get("milestone") if markdown_task else None,
                "observed_milestone": yaml_task.get("milestone") if yaml_task else None,
                "markdown_present": markdown_task is not None,
                "yaml_present": yaml_task is not None,
                "passed": bool(
                    markdown_task
                    and yaml_task
                    and markdown_task["order"] == yaml_task["order"] == order
                    and markdown_task["id"] == yaml_task["id"] == expected_id
                    and markdown_task["milestone"] == yaml_task["milestone"] == MILESTONE
                    and id_checks.get(order, False)
                    and all(checks_by_order.get(order, [False]))
                ),
            }
        )

    global_checks = (
        markdown["milestone"] == MILESTONE,
        roadmap["milestone"] == MILESTONE,
        len(markdown_tasks) == EXPECTED_TASK_COUNT,
        len(yaml_tasks) == EXPECTED_TASK_COUNT,
        tuple(task["id"] for task in markdown_tasks) == EXPECTED_TASK_IDS,
        tuple(task["id"] for task in yaml_tasks) == EXPECTED_TASK_IDS,
        len(number_counts) == EXPECTED_TASK_COUNT,
        len(gate_producer_rows) == EXPECTED_GATE_COUNT,
        len(prior_rows) == EXPECTED_PRIOR_COUNT,
        all(task_number(row["upstream"]) != 7097 for row in gate_producer_rows),
    )
    passed = (
        all(global_checks)
        and len(task_rows) == EXPECTED_TASK_COUNT
        and all(row["passed"] for row in task_rows)
        and all(row["passed"] for row in gate_producer_rows)
    )
    return {
        "passed": passed,
        "markdown_task_rows": markdown_tasks,
        "yaml_task_rows": yaml_tasks,
        "task_contract_rows": task_rows,
        "title_parity_rows": title_rows,
        "deliverable_parity_rows": deliverable_rows,
        "gate_contract_rows": gate_rows,
        "gate_producer_rows": gate_producer_rows,
        "prior_failure_rows": prior_rows,
        "model_compliance_rows": model_rows,
        "agent_routing_rows": routing_rows,
        "substrate_class_rows": substrate_rows,
        "execution_venue_rows": venue_rows,
        "artifact_field_rows": artifact_rows,
        "prompt_tail_rows": tail_rows,
        "markdown_task_count": len(markdown_tasks),
        "observed_id_order": [task["id"] for task in yaml_tasks],
    }


def _sha256_path(path: Path) -> str:
    """Hash exact bytes so a later audit can detect source drift."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path_writable(path: Path) -> tuple[bool, str]:
    """Probe the destination before the audit depends on writing it."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7097-write-"):
            pass
        return True, "temporary_write_succeeded"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _preconditions(
    root: Path, output_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Read the three required sources and ignore staging content."""

    rows: list[dict[str, Any]] = []
    roadmap: dict[str, Any] | None = None
    exclusion: dict[str, Any] | None = None

    def add(check: str, path: Path, expected: str, available: bool, observed: str) -> None:
        rows.append(
            {
                "check": check,
                "path": str(path),
                "expected_value": expected,
                "observed_value": observed,
                "available": available,
                "terminal": True,
            }
        )

    design_path = root / DESIGN_PATH
    try:
        design_text = design_path.read_text(encoding="utf-8")
        available = bool(design_text.strip())
        add(
            "v623_markdown_readable",
            design_path,
            "readable_nonempty_source",
            available,
            "readable_nonempty_source" if available else "empty_source",
        )
    except OSError as exc:
        add(
            "v623_markdown_readable",
            design_path,
            "readable_nonempty_source",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    for check, relative, target in (
        ("v623_active_yaml_readable", ACTIVE_ROADMAP_PATH, "roadmap"),
        ("exclusion_manifest_readable", EXCLUSION_PATH, "exclusion"),
    ):
        path = root / relative
        try:
            document = load_yaml(path)
            if target == "roadmap":
                roadmap = document
            else:
                exclusion = document
            add(
                check,
                path,
                "readable_nonempty_yaml_mapping",
                True,
                "readable_nonempty_yaml_mapping",
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            add(
                check, path, "readable_nonempty_yaml_mapping", False, f"{type(exc).__name__}: {exc}"
            )

    staging_path = root / STAGING_ROADMAP_PATH
    add(
        "staging_file_not_required",
        staging_path,
        "not_read_at_execution",
        True,
        "present_but_not_read" if staging_path.exists() else "absent_and_not_read",
    )
    writable, observed = _artifact_path_writable(output_path)
    add("artifact_path_writable", output_path, "temporary_write_succeeds", writable, observed)
    return rows, roadmap, exclusion


def _source_hashes(root: Path) -> dict[str, str | None]:
    """Hash audit sources while preserving absent source identities."""

    paths = (
        CLAUDE_PATH,
        CODEX_PATH,
        PROGRAM_PATH,
        DESIGN_PATH,
        ACTIVE_ROADMAP_PATH,
        EXP7091_PATH,
        EXP7084_PATH,
        EXP7076_PATH,
        EXP7091_MODULE_PATH,
        EXCLUSION_PATH,
        ROADMAP_SCHEMA_PATH,
        ROADMAP_AUDIT_PATH,
        EXCLUSION_LINT_PATH,
        HARNESS_LINT_PATH,
        ADVERSARIAL_PATH,
        SPEC_PATH,
        HARNESS_SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        str(relative): _sha256_path(root / relative) if (root / relative).is_file() else None
        for relative in paths
    }


def _field_principles() -> dict[str, str]:
    """Attach one scientific reason to every required result field."""

    principles = {
        field: f"Preserving {field} lets an independent reviewer reproduce or falsify the V623 contract conclusion."
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }
    principles["field_principles"] = (
        "Scientific reasons make every recorded field an evidence control."
    )
    principles["preconditions_checked"] = (
        "A conclusion is valid only when every required runtime input was available."
    )
    principles["source_artifact_hashes"] = (
        "Content hashes reveal source drift during later replication."
    )
    principles["rows"] = "Per-task evidence prevents a pooled score from hiding a local failure."
    principles["v623_task_contract_conforms_score"] = (
        "The score is one only when all 12 rows and eight gate clauses pass."
    )
    principles["reproducibility_checksum"] = (
        "The checksum detects a change to any completed audit field."
    )
    principles["honest_verdict"] = (
        "A terminal class distinguishes a mismatch from a missing prerequisite."
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact fields except the field that stores this hash."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _empty_contract() -> dict[str, list[Any]]:
    """Keep blocked artifacts complete before contract parsing can run."""

    return {field: [] for field in CONTRACT_ROW_FIELDS} | {
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "task_contract_rows": [],
    }


def _base_artifact(root: Path, run_date: str) -> dict[str, Any]:
    """Create every required field before the terminal state is known."""

    return {
        "schema": "carnot.exp7097.v623_contract_preflight.v1",
        "experiment_id": 7097,
        "run_date": run_date,
        "status": "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root),
        "rows": [],
        **_empty_contract(),
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_TASK_IDS),
        "observed_id_order": [],
        "active_roadmap_path": str(ACTIVE_ROADMAP_PATH),
        "staging_file_required_at_execution": False,
        "v623_task_contract_conforms_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "preconditions_not_checked",
            "expected_value": "all_required_inputs_available",
            "observed_value": "not_checked",
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def _finish(artifact: dict[str, Any], started: float) -> dict[str, Any]:
    """Freeze measured duration and checksum after all other fields."""

    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _contract_failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while row tables keep local causes."""

    if contract["markdown_task_count"] != EXPECTED_TASK_COUNT:
        check, expected, observed = (
            "markdown_task_count",
            EXPECTED_TASK_COUNT,
            contract["markdown_task_count"],
        )
    elif len(contract["yaml_task_rows"]) != EXPECTED_TASK_COUNT:
        check, expected, observed = (
            "yaml_task_count",
            EXPECTED_TASK_COUNT,
            len(contract["yaml_task_rows"]),
        )
    else:
        first = next(row for row in contract["task_contract_rows"] if not row["passed"])
        check, expected, observed = f"task_contract_order_{first['order']}", True, False
    return {
        "failed_check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def build_artifact(root: Path, run_date: str, *, output_path: Path) -> dict[str, Any]:
    """Build a positive, disqualified, or prerequisite-blocked artifact."""

    started = time.monotonic()
    artifact = _base_artifact(root, run_date)
    preconditions, roadmap, exclusion = _preconditions(root, output_path)
    artifact["preconditions_checked"] = preconditions
    failed = next((row for row in preconditions if not row["available"]), None)
    if failed is not None:
        artifact["gate_check_summary"] = {
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
            "passed": False,
        }
        return _finish(artifact, started)

    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    try:
        contract = evaluate_contract(
            (root / DESIGN_PATH).read_text(encoding="utf-8"),
            roadmap,
            retired_experiment_ids(exclusion),
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        observed = f"{type(exc).__name__}: {exc}"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["rows"] = [
            {"row_kind": "contract_parse_error", "observed": observed, "passed": False}
        ]
        artifact["gate_check_summary"] = {
            "failed_check": "contract_parse",
            "expected_value": "two_parseable_independent_contracts",
            "observed_value": observed,
            "passed": False,
        }
        return _finish(artifact, started)

    for field in (
        "markdown_task_rows",
        "yaml_task_rows",
        "task_contract_rows",
        *CONTRACT_ROW_FIELDS,
    ):
        artifact[field] = contract[field]
    artifact["rows"] = [dict(row) for row in contract["task_contract_rows"]]
    artifact["observed_task_count"] = len(contract["yaml_task_rows"])
    artifact["observed_id_order"] = contract["observed_id_order"]
    artifact["v623_task_contract_conforms_score"] = int(contract["passed"])
    if contract["passed"]:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = POSITIVE_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": None,
            "expected_value": 1,
            "observed_value": 1,
            "passed": True,
        }
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = _contract_failure_summary(contract)
    return _finish(artifact, started)


def _contract_score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute conformance from preserved evidence instead of headlines."""

    markdown_rows = artifact.get("markdown_task_rows")
    yaml_rows = artifact.get("yaml_task_rows")
    task_rows = artifact.get("task_contract_rows")
    if not all(
        isinstance(rows, list) and len(rows) == EXPECTED_TASK_COUNT
        for rows in (markdown_rows, yaml_rows, task_rows)
    ):
        return 0
    if [row.get("id") for row in markdown_rows if isinstance(row, Mapping)] != list(
        EXPECTED_TASK_IDS
    ):
        return 0
    if [row.get("id") for row in yaml_rows if isinstance(row, Mapping)] != list(EXPECTED_TASK_IDS):
        return 0
    for order, row in enumerate(task_rows, start=1):
        expected_id = EXPECTED_TASK_IDS[order - 1]
        if (
            not isinstance(row, Mapping)
            or row.get("passed") is not True
            or row.get("order") != order
            or row.get("expected_id") != expected_id
            or row.get("markdown_id") != expected_id
            or row.get("observed_id") != expected_id
        ):
            return 0
    exact_counts = {
        "title_parity_rows": EXPECTED_TASK_COUNT,
        "deliverable_parity_rows": EXPECTED_TASK_COUNT,
        "gate_contract_rows": EXPECTED_TASK_COUNT,
        "gate_producer_rows": EXPECTED_GATE_COUNT,
        "prior_failure_rows": EXPECTED_PRIOR_COUNT,
        "model_compliance_rows": EXPECTED_TASK_COUNT,
        "agent_routing_rows": EXPECTED_TASK_COUNT,
        "substrate_class_rows": EXPECTED_TASK_COUNT,
        "execution_venue_rows": EXPECTED_TASK_COUNT,
        "artifact_field_rows": EXPECTED_TASK_COUNT,
        "prompt_tail_rows": EXPECTED_TASK_COUNT,
    }
    for field in CONTRACT_ROW_FIELDS:
        rows = artifact.get(field)
        if (
            not isinstance(rows, list)
            or not rows
            or len(rows) != exact_counts[field]
            or not all(isinstance(row, Mapping) and row.get("passed") is True for row in rows)
        ):
            return 0
    return 1


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the score, terminal class, diagnostics, and checksum."""

    errors = [
        f"missing_required_field:{field}"
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
    ]
    if errors:
        return errors
    if artifact.get("field_principles") != _field_principles():
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("expected_task_count") != EXPECTED_TASK_COUNT:
        errors.append("expected_task_count_invalid")
    if artifact.get("expected_id_order") != list(EXPECTED_TASK_IDS):
        errors.append("expected_id_order_invalid")
    if artifact.get("active_roadmap_path") != str(ACTIVE_ROADMAP_PATH):
        errors.append("active_roadmap_path_invalid")
    if artifact.get("staging_file_required_at_execution") is not False:
        errors.append("staging_file_required_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s_invalid")

    yaml_rows = artifact.get("yaml_task_rows")
    observed_count = len(yaml_rows) if isinstance(yaml_rows, list) else -1
    observed_order = (
        [row.get("id") for row in yaml_rows if isinstance(row, Mapping)]
        if isinstance(yaml_rows, list)
        else []
    )
    if artifact.get("observed_task_count") != observed_count:
        errors.append("observed_task_count_invalid")
    if artifact.get("observed_id_order") != observed_order:
        errors.append("observed_id_order_invalid")
    rows = artifact.get("rows")
    task_rows = artifact.get("task_contract_rows")
    parse_error_rows = (
        isinstance(rows, list)
        and len(rows) == 1
        and isinstance(rows[0], Mapping)
        and rows[0].get("row_kind") == "contract_parse_error"
        and task_rows == []
    )
    if rows != task_rows and not parse_error_rows:
        errors.append("rows_not_task_contract_rows")

    score = _contract_score_from_artifact(artifact)
    if artifact.get("v623_task_contract_conforms_score") != score:
        errors.append("v623_task_contract_conforms_score_invalid")
    preconditions = artifact.get("preconditions_checked")
    preconditions_ok = bool(preconditions) and all(
        isinstance(row, Mapping) and row.get("available") is True for row in preconditions
    )
    if not preconditions_ok:
        expected_class, expected_verdict, expected_substrate_class = (
            "blocked",
            BLOCKED_VERDICT,
            "blocked_no_run",
        )
    elif not score:
        expected_class, expected_verdict, expected_substrate_class = (
            "disqualified",
            DISQUALIFIED_VERDICT,
            INFERENCE_SUBSTRATE_CLASS,
        )
    else:
        expected_class, expected_verdict, expected_substrate_class = (
            "positive",
            POSITIVE_VERDICT,
            INFERENCE_SUBSTRATE_CLASS,
        )
    if artifact.get("inference_substrate_class") != expected_substrate_class:
        errors.append("inference_substrate_class_invalid")
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_not_derived")
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict_not_derived")

    summary = artifact.get("gate_check_summary")
    summary_valid = (
        isinstance(summary, Mapping)
        and {"failed_check", "expected_value", "observed_value", "passed"} <= summary.keys()
    )
    if not summary_valid:
        errors.append("gate_check_summary_invalid")
    else:
        if expected_class == "blocked":
            failed = next(
                row
                for row in preconditions
                if isinstance(row, Mapping) and row.get("available") is not True
            )
            expected_summary = {
                "failed_check": failed.get("check"),
                "expected_value": failed.get("expected_value"),
                "observed_value": failed.get("observed_value"),
                "passed": False,
            }
        elif expected_class == "positive":
            expected_summary = {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            }
        elif parse_error_rows:
            expected_summary = {
                "failed_check": "contract_parse",
                "expected_value": "two_parseable_independent_contracts",
                "observed_value": rows[0].get("observed"),
                "passed": False,
            }
        elif (
            isinstance(artifact.get("markdown_task_rows"), list)
            and len(artifact["markdown_task_rows"]) != EXPECTED_TASK_COUNT
        ):
            expected_summary = {
                "failed_check": "markdown_task_count",
                "expected_value": EXPECTED_TASK_COUNT,
                "observed_value": len(artifact["markdown_task_rows"]),
                "passed": False,
            }
        elif isinstance(yaml_rows, list) and len(yaml_rows) != EXPECTED_TASK_COUNT:
            expected_summary = {
                "failed_check": "yaml_task_count",
                "expected_value": EXPECTED_TASK_COUNT,
                "observed_value": len(yaml_rows),
                "passed": False,
            }
        else:
            first = next(
                row
                for row in task_rows
                if isinstance(row, Mapping) and row.get("passed") is not True
            )
            expected_summary = {
                "failed_check": f"task_contract_order_{first.get('order')}",
                "expected_value": True,
                "observed_value": False,
                "passed": False,
            }
        if dict(summary) != expected_summary:
            errors.append("gate_check_summary_not_derived")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _date_argument(value: str) -> str:
    """Require the compact date format used by result artifacts."""

    if re.fullmatch(r"\d{8}", value) is None:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the result only after the complete JSON reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the advisory audit or validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validate-artifact", type=Path)
    args = parser.parse_args(argv)

    if args.validate_artifact is not None:
        try:
            value = json.loads(args.validate_artifact.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"invalid artifact: {type(exc).__name__}: {exc}")
            return 1
        errors = (
            validate_artifact(value)
            if isinstance(value, Mapping)
            else ["artifact_mapping_required"]
        )
        if errors:
            print("invalid artifact: " + ", ".join(errors))
            return 1
        print(f"valid artifact: {args.validate_artifact}")
        return 0

    if args.date is None:
        parser.error("--date is required unless --validate-artifact is used")
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(root, args.date, output_path=output)
    errors = validate_artifact(artifact)
    _write_json_atomic(output, artifact)
    if errors:
        print("invalid generated artifact: " + ", ".join(errors))
        return 1
    print(f"{artifact['honest_verdict']}: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository uses the thin wrapper.
    raise SystemExit(main())
