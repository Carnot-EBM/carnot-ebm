"""Audit the V626 Markdown contract against the active execution YAML.

The two parsers receive separate source values and build separate row objects.
This keeps an active-roadmap omission visible instead of copying one parsed
contract into both sides of the comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
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
EXP7121_PATH = Path("results/experiment_7121_v625_contract_preflight.json")
EXP7121_WRAPPER_PATH = Path(
    "scripts/experiments/experiment_7121_v625_contract_preflight.py"
)
MODULE_PATH = Path("python/carnot/experiment_7124_v626_contract_preflight.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7124_v626_contract_preflight.py")
TEST_PATH = Path("tests/python/test_experiment_7124_v626_contract_preflight.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7124_v626_contract_preflight.json")

MILESTONE = "2026.09.626"
EXPECTED_TASK_IDS = (
    "exp7124-v626-contract-preflight",
    "exp7125-v626-source-delta",
    "exp7126-arc-loo-phase-receipt-forensics",
    "exp7127-adapter-withheld-arc-loo-cell",
    "exp7128-arc-loo-causal-provenance-audit",
    "exp7129-hardness-controlled-sota-constraint-bank",
    "exp7130-verifier-committed-uncertainty-routing",
    "exp7131-model-facing-fixed-schema-csl",
    "exp7132-directional-memory-portability-audit",
    "exp7133-wcrg-multiscale-sampler-prototype",
    "exp7134-frustrated-ising-sampler-benchmark",
    "exp7135-v626-capstone",
)
EXPECTED_TASK_COUNT = 12
EXPECTED_GATED_TASKS = frozenset({7130, 7131, 7132, 7134})
MAX_WALL_TIME_MIN = 70
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts: independent Markdown and YAML contract parsing"
)
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
PROMPT_FINAL_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
RANDOM_SEED = 712420260907

RUN_SCRIPTS = {
    7124: "experiment_7124_v626_contract_preflight.py",
    7125: "experiment_7125_v626_source_delta.py",
    7126: "experiment_7126_v626_arc_loo_phase_receipts.py",
    7127: "experiment_7127_v626_adapter_withheld_arc_loo.py",
    7128: "experiment_7128_v626_arc_loo_causal_audit.py",
    7129: "experiment_7129_v626_sota_constraint_bank.py",
    7130: "experiment_7130_v626_verifier_committed_routing.py",
    7131: "experiment_7131_v626_model_facing_csl.py",
    7132: "experiment_7132_v626_memory_portability_audit.py",
    7133: "experiment_7133_v626_multiscale_sampler_prototype.py",
    7134: "experiment_7134_v626_multiscale_sampler_benchmark.py",
    7135: "experiment_7135_v626_capstone.py",
}
MANDATED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_SUBSTRATE_CLASSES = frozenset(
    {"model_load_no_generation", "model_bounded_generation", "model_full_generation"}
)
LEGAL_SUBSTRATE_CLASSES = frozenset(
    {
        "aggregation",
        "no_model_load",
        "model_load_no_generation",
        "model_bounded_generation",
        "model_full_generation",
        "cpu_exact_solver_or_simulator",
        "blocked_no_run",
    }
)
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)

BLOCKED_VERDICT = "blocked_v626_contract_preflight_prerequisite_missing"
POSITIVE_VERDICT = "complete_positive_v626_task_contract_conforms"
DISQUALIFIED_VERDICT = "complete_disqualified_v626_markdown_yaml_contract_mismatch"

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "run_date",
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
        "existing_path_rows",
        "wall_time_rows",
        "artifact_field_rows",
        "prompt_tail_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "active_roadmap_path",
        "staging_file_required_at_execution",
        "v626_task_contract_conforms_score",
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
    "existing_path_rows",
    "wall_time_rows",
    "artifact_field_rows",
    "prompt_tail_rows",
)
TASK_REQUIRED_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
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

    match = re.fullmatch(r"exp(\d+)(?:-[A-Za-z0-9_-]+)?", str(task_id), re.I)
    return int(match.group(1)) if match else None


def _normal_exp_id(value: object) -> str | None:
    """Normalize a full or numeric experiment identity for manifest checks."""

    candidate = f"exp{value}" if isinstance(value, int) else value
    match = re.match(r"exp(\d+)(?:-|$)", str(candidate), re.I)
    return f"exp{int(match.group(1))}" if match else None


def _parse_scalar(text: str) -> Any:
    """Parse one gate value and reject hidden structured gate values."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("structured gate value must be scalar")
    return value


def _parse_markdown_gate(
    cell: str, full_id_by_number: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Parse a Markdown gate without consulting a YAML object."""

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
        number = task_number(match.group(1))
        upstream = full_id_by_number.get(number, match.group(1))
        gates.append(
            {
                "upstream": upstream,
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4).strip().strip("`")),
            }
        )
    return gates


def _markdown_model_contract(text: str) -> tuple[tuple[str, ...], str]:
    """Read headline repository IDs and task assignments from Markdown alone."""

    section = re.search(r"^## Model contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.I | re.M)
    if section is None:
        return (), ""
    found = re.findall(r"`(unsloth/[A-Za-z0-9_.-]+-GGUF)`", section.group(1))
    return tuple(dict.fromkeys(found)), section.group(1)


def parse_markdown_contract(text: str) -> dict[str, Any]:
    """Parse only the Markdown milestone, model section, and task table."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\s]+)`?", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    section_match = re.search(
        r"^## Exact task contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.I | re.M
    )
    if section_match is None:
        raise ValueError("Markdown task contract section is missing")

    tasks: list[dict[str, Any]] = []
    gate_cells: list[str] = []
    for line in section_match.group(1).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells[0].lower() == "order" or all(
            re.fullmatch(r":?-+:?", cell) for cell in cells
        ):
            continue
        if len(cells) != 5 or not cells[0].isdigit():
            raise ValueError(f"malformed Markdown task row: {line}")
        order = int(cells[0])
        task_id = cells[1].strip("`")
        number = task_number(task_id)
        title = cells[2]
        deliverable = cells[3].strip("`")
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
            }
        )
        gate_cells.append(cells[4])
    if not tasks:
        raise ValueError("Markdown task table is missing")
    full_id_by_number = {task["number"]: task["id"] for task in tasks}
    for task, gate_cell in zip(tasks, gate_cells, strict=True):
        task["gates"] = _parse_markdown_gate(gate_cell, full_id_by_number)
    headline_models, model_contract = _markdown_model_contract(text)
    return {
        "milestone": milestone_match.group(1),
        "tasks": tasks,
        "headline_models": headline_models,
        "model_contract": model_contract,
    }


def _required_block(prompt: str) -> str:
    """Isolate required fields so task prose cannot satisfy field checks."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def _existing_code_paths(prompt: str) -> list[str]:
    """Read only paths inside the named existing-input section."""

    marker = "EXISTING CODE TO READ FIRST:"
    if marker not in prompt:
        return []
    section = prompt.split(marker, 1)[1]
    section = re.split(r"\n\s*(?:TASK|CONCRETE STEPS):\s*", section, maxsplit=1)[0]
    return re.findall(r"\{project_root\}/([A-Za-z0-9_./-]+)", section)


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
        raw_requires = raw_task.get("requires") or []
        if not all(isinstance(value, list) for value in (raw_gates, raw_priors, raw_requires)):
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
        task_id = raw_task.get("id")
        prompt = raw_task.get("prompt")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            raise ValueError(f"YAML task {order} has malformed id or prompt")
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
                "requires": list(raw_requires),
                "prompt": prompt,
                "required_block": _required_block(prompt),
                "existing_paths": _existing_code_paths(prompt),
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one non-empty YAML mapping."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document:
        raise ValueError(f"YAML mapping required at {path}")
    return document


def retired_experiment_ids(document: object) -> set[str]:
    """Collect retired experiment numbers from the exclusion manifest."""

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
                normal
                for candidate in candidates
                if (normal := _normal_exp_id(candidate)) is not None
            )
            if isinstance(item.get("un_retired_experiment_ids"), list):
                restored.update(
                    normal
                    for candidate in item["un_retired_experiment_ids"]
                    if (normal := _normal_exp_id(candidate)) is not None
                )
    return retired - restored


def _canonical_gates(gates: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Make gate ordering and all gate fields explicit for comparison."""

    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
    ]


def _field_declared(block: str, field: str) -> bool:
    """Accept a bare declared field and reject a nested look-alike."""

    return (
        re.search(
            rf"(?<![A-Za-z0-9_.]){re.escape(field)}(?![A-Za-z0-9_]|\.[A-Za-z_])",
            block,
        )
        is not None
    )


def _planned_substrate_class(block: str) -> str | None:
    """Read the planned class before the blocked no-run alternative."""

    match = re.search(r"inference_substrate_class\s*\(\s*([A-Za-z_]+)", block)
    return match.group(1) if match else None


def _closed_verdict_enum(block: str) -> bool:
    """Require the six verdict classes in their fixed order."""

    normalized = re.sub(r"\s+", " ", block)
    values = r"\s*\|\s*".join(map(re.escape, VERDICT_CLASSES))
    return re.search(rf"verdict_class\s*\(\s*{values}\s*\)", normalized) is not None


def _comparison_prompt(prompt: str) -> bool:
    """Detect comparison work before the generic artifact field list."""

    prose = prompt.split("REQUIRED ARTIFACT FIELDS:", 1)[0]
    return (
        re.search(
            r"\b(?:compare|comparison|comparative|controls?|arms?|parity|paired|versus|A/B|audit)\b",
            prose,
            re.I,
        )
        is not None
    )


def _route_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Require the explicit V626 Codex route and task metadata."""

    agent = task.get("agent_type")
    model = task.get("model")
    route_valid = agent == "codex" and model == "gpt-5.6-sol"
    metadata = {
        "track": isinstance(task.get("track"), str) and bool(task["track"].strip()),
        "priority": task.get("priority") in {"critical", "high", "medium"},
        "requires_gpu": isinstance(task.get("requires_gpu"), bool),
        "max_turns": isinstance(task.get("max_turns"), int) and task["max_turns"] > 0,
        "per_unit_rows": isinstance(task.get("per_unit_rows"), bool),
    }
    return {
        "order": task["order"],
        "number": task["number"],
        "task_id": task["id"],
        "observed_route": [agent, model],
        "route_valid": route_valid,
        "metadata_checks": metadata,
        "passed": route_valid and all(metadata.values()),
    }


def _model_row(
    task: Mapping[str, Any],
    markdown_headline_models: Sequence[str],
    markdown_model_contract: str,
) -> dict[str, Any]:
    """Require each local-model task to run a permitted headline cell."""

    block = task["required_block"]
    prompt = task["prompt"]
    planned_class = _planned_substrate_class(block)
    is_local_llm = planned_class in MODEL_SUBSTRATE_CLASSES
    model_specs = _field_declared(block, "MODEL_SPECS")
    explicit_models = [model for model in MANDATED_MODELS if model in prompt]
    all_three_reference = re.search(r"all three mandated repositories", prompt, re.I) is not None
    resolved_models = (
        list(markdown_headline_models) if all_three_reference else explicit_models
    )
    resolved_models = [model for model in resolved_models if model in MANDATED_MODELS]
    execution_required = (
        re.search(r"\b(?:execute|run|invoke|load|prompt|generation|outputs?)\b", prompt, re.I)
        is not None
    )
    markdown_task_assignment = any(
        re.search(rf"\bExp{task['number']}\b", sentence, re.I)
        and re.search(r"\buses?\b", sentence, re.I)
        for sentence in re.split(r"[.!?](?:\s+|$)", markdown_model_contract)
    )
    headline_cell_declared = (
        re.search(r"\bheadline\b", prompt, re.I) is not None or markdown_task_assignment
    )
    headline_contract = bool(resolved_models) and all(
        model in markdown_headline_models for model in resolved_models
    )
    legacy_replacement = (
        re.search(
            r"(?:qwen3\.5-0\.8b|gemma-4-e4b|legacy.small).{0,80}(?:headline|replace|substitut)",
            prompt,
            re.I | re.S,
        )
        is not None
    )
    if is_local_llm:
        passed = bool(
            task.get("requires_gpu") is True
            and model_specs
            and resolved_models
            and execution_required
            and headline_cell_declared
            and headline_contract
            and not legacy_replacement
        )
    else:
        passed = True
    return {
        "order": task["order"],
        "number": task["number"],
        "task_id": task["id"],
        "local_llm_task": is_local_llm,
        "model_specs_declared": model_specs,
        "headline_models": resolved_models,
        "headline_contract_source": "independent_markdown_model_contract",
        "headline_cell_declared": headline_cell_declared,
        "headline_execution_required": execution_required,
        "legacy_small_headline_replacement": legacy_replacement,
        "passed": passed,
    }


def _prior_rows(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Record all four required fields for each declared prior failure."""

    priors = task["prior_failures"]
    if not priors:
        return [
            {
                "order": task["order"],
                "number": task["number"],
                "task_id": task["id"],
                "prior_failure_present": False,
                "passed": True,
            }
        ]
    rows: list[dict[str, Any]] = []
    for prior in priors:
        values = dict(prior) if isinstance(prior, Mapping) else {}
        checks = {
            "experiment_id": isinstance(values.get("experiment_id"), str)
            and bool(values["experiment_id"].strip()),
            "verdict": isinstance(values.get("verdict"), str)
            and bool(values["verdict"].strip()),
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
                "field_checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _expected_prompt_tail(number: int | None) -> list[str] | None:
    """Return the exact two terminal lines for one task."""

    script = RUN_SCRIPTS.get(number)
    if script is None:
        return None
    return [
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        f"{script} --date {{date}}",
        PROMPT_FINAL_LINE,
    ]


def _missing_row(order: int, number: int | None, task_id: str | None) -> dict[str, Any]:
    """Keep each evidence table aligned when one YAML row is absent."""

    return {"order": order, "number": number, "task_id": task_id, "passed": False}


def evaluate_contract(
    markdown_text: str,
    yaml_document: object,
    retired_ids: set[str],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Compare all V626 rows and preserve every supporting policy result."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    normalized_retired = {
        normal for value in retired_ids if (normal := _normal_exp_id(value)) is not None
    }
    number_counts = Counter(task["number"] for task in yaml_tasks)
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))

    title_rows: list[dict[str, Any]] = []
    deliverable_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    existing_rows: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    identity_checks: dict[int, bool] = {}

    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        source_task = markdown_task or yaml_task or {}
        task_id = source_task.get("id")
        number = source_task.get("number")
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
            missing = _missing_row(order, number, task_id)
            prior_rows.append({**missing, "prior_failure_present": None})
            model_rows.append({**missing, "local_llm_task": None})
            route_rows.append({**missing, "observed_route": None})
            existing_rows.append({**missing, "path": None})
            wall_rows.append({**missing, "observed_wall_time_min": None})
            artifact_rows.append({**missing, "comparison": None})
            tail_rows.append(
                {**missing, "expected": _expected_prompt_tail(number), "observed": None}
            )
            identity_checks[order] = False
            continue

        prior_rows.extend(_prior_rows(yaml_task))
        normalized_id = _normal_exp_id(yaml_task["id"])
        unique = yaml_task["number"] is not None and number_counts[yaml_task["number"]] == 1
        not_retired = normalized_id not in normalized_retired
        identity_checks[order] = unique and not_retired
        model_rows.append(
            _model_row(
                yaml_task,
                markdown["headline_models"],
                markdown["model_contract"],
            )
        )
        route_rows.append(_route_row(yaml_task))

        paths = yaml_task["existing_paths"]
        if not paths:
            existing_rows.append(
                {
                    "order": order,
                    "number": yaml_task["number"],
                    "task_id": yaml_task["id"],
                    "path": None,
                    "exists_at_plan_time": False,
                    "section_scoped": True,
                    "passed": False,
                }
            )
        for relative_text in paths:
            relative = Path(relative_text)
            exists = not relative.is_absolute() and (root / relative).exists()
            existing_rows.append(
                {
                    "order": order,
                    "number": yaml_task["number"],
                    "task_id": yaml_task["id"],
                    "path": relative_text,
                    "exists_at_plan_time": exists,
                    "section_scoped": True,
                    "passed": exists,
                }
            )

        wall = yaml_task.get("estimated_wall_time_min")
        wall_passed = isinstance(wall, int) and 0 < wall <= MAX_WALL_TIME_MIN
        wall_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "observed_wall_time_min": wall,
                "minimum_wall_time_min": 1,
                "maximum_wall_time_min": MAX_WALL_TIME_MIN,
                "passed": wall_passed,
            }
        )

        prompt = yaml_task["prompt"]
        block = yaml_task["required_block"]
        comparison = _comparison_prompt(prompt)
        planned_class = _planned_substrate_class(block)
        normalized_block = re.sub(r"\s+", " ", block.lower())
        diagnostics = all(
            phrase in normalized_block
            for phrase in ("failed check", "expected value", "observed value")
        )
        generic_fields = all(_field_declared(block, field) for field in TASK_REQUIRED_FIELDS)
        principles = "field_principles for every field below" in normalized_block
        class_valid = (
            planned_class in LEGAL_SUBSTRATE_CLASSES
            and planned_class != "blocked_no_run"
            and _field_declared(block, "blocked_no_run")
        )
        host_declared = re.search(r"execution_venue\s*\(\s*host\s*\)", block) is not None
        level_rows = _field_declared(block, "level_rows")
        solve_provenance = _field_declared(block, "solve_provenance")
        number_value = yaml_task["number"]
        capstone_ungated = not yaml_task["gates"] if number_value == 7135 else None
        advisory_ungated = not yaml_task["gates"] if number_value == 7124 else None
        arc_cell_ungated = not yaml_task["gates"] if number_value == 7127 else None
        arc_cell_no_requires = not yaml_task["requires"] if number_value == 7127 else None
        artifact_rows.append(
            {
                "order": order,
                "number": number_value,
                "task_id": yaml_task["id"],
                "comparison": comparison,
                "per_unit_rows": yaml_task.get("per_unit_rows"),
                "generic_fields_present": generic_fields,
                "field_principles_required": principles,
                "blocked_diagnostics_present": diagnostics,
                "closed_verdict_enum": _closed_verdict_enum(block),
                "planned_substrate_class": planned_class,
                "substrate_class_valid": class_valid,
                "host_venue_declared": host_declared,
                "level_rows_declared": level_rows,
                "solve_provenance_declared": solve_provenance,
                "unique_id": unique,
                "absent_from_retired_ids": not_retired,
                "capstone_ungated": capstone_ungated,
                "advisory_preflight_ungated": advisory_ungated,
                "arc_cell_ungated": arc_cell_ungated,
                "arc_cell_no_requires_chain": arc_cell_no_requires,
                "passed": bool(
                    block
                    and generic_fields
                    and principles
                    and diagnostics
                    and _closed_verdict_enum(block)
                    and class_valid
                    and host_declared
                    and (not comparison or yaml_task.get("per_unit_rows") is True)
                    and (
                        str(yaml_task.get("track", "")).lower() != "arc"
                        or not level_rows
                        or solve_provenance
                    )
                    and capstone_ungated is not False
                    and advisory_ungated is not False
                    and arc_cell_ungated is not False
                    and arc_cell_no_requires is not False
                    and unique
                    and not_retired
                ),
            }
        )
        observed_tail = prompt.rstrip().splitlines()[-2:] if prompt.rstrip() else []
        expected_tail = _expected_prompt_tail(number_value)
        tail_rows.append(
            {
                "order": order,
                "number": number_value,
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
            forbidden = _normal_exp_id(upstream_id) in normalized_retired
            advisory_used = task_number(upstream_id) == 7124
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
                    "upstream_forbidden": forbidden,
                    "advisory_preflight_used": advisory_used,
                    "passed": bool(
                        producer
                        and producer_precedes
                        and same_milestone
                        and bare_field
                        and field_declared
                        and not forbidden
                        and not advisory_used
                    ),
                }
            )

    supporting = (
        title_rows,
        deliverable_rows,
        gate_rows,
        prior_rows,
        model_rows,
        route_rows,
        existing_rows,
        wall_rows,
        artifact_rows,
        tail_rows,
        gate_producer_rows,
    )
    checks_by_order: dict[int, list[bool]] = {}
    for evidence_rows in supporting:
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
                    and identity_checks.get(order, False)
                    and all(checks_by_order.get(order, [False]))
                ),
            }
        )

    gated_numbers = {task["number"] for task in yaml_tasks if task["gates"]}
    global_checks = (
        markdown["milestone"] == MILESTONE,
        roadmap["milestone"] == MILESTONE,
        tuple(markdown["headline_models"]) == MANDATED_MODELS,
        len(markdown_tasks) == EXPECTED_TASK_COUNT,
        len(yaml_tasks) == EXPECTED_TASK_COUNT,
        tuple(task["id"] for task in markdown_tasks) == EXPECTED_TASK_IDS,
        tuple(task["id"] for task in yaml_tasks) == EXPECTED_TASK_IDS,
        len(number_counts) == EXPECTED_TASK_COUNT,
        gated_numbers == EXPECTED_GATED_TASKS,
        len(gate_producer_rows) == len(EXPECTED_GATED_TASKS),
    )
    passed = bool(
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
        "agent_routing_rows": route_rows,
        "existing_path_rows": existing_rows,
        "wall_time_rows": wall_rows,
        "artifact_field_rows": artifact_rows,
        "prompt_tail_rows": tail_rows,
        "markdown_task_count": len(markdown_tasks),
        "observed_id_order": [task["id"] for task in yaml_tasks],
    }


def _sha256_path(path: Path) -> str:
    """Hash exact bytes so later checks can detect source drift."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path_writable(path: Path) -> tuple[bool, str]:
    """Probe the destination directory before the audit depends on it."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7124-write-"):
            pass
        return True, "temporary_write_succeeded"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _preconditions(
    root: Path, output_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Read the required sources without reading the staging roadmap."""

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
            "v626_markdown_readable",
            design_path,
            "readable_nonempty_source",
            available,
            "readable_nonempty_source" if available else "empty_source",
        )
    except OSError as exc:
        add(
            "v626_markdown_readable",
            design_path,
            "readable_nonempty_source",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    for check, relative, target in (
        ("v626_active_yaml_readable", ACTIVE_ROADMAP_PATH, "roadmap"),
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
                check,
                path,
                "readable_nonempty_yaml_mapping",
                False,
                f"{type(exc).__name__}: {exc}",
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
    """Hash requested sources and preserve missing source identities."""

    paths = (
        CLAUDE_PATH,
        CODEX_PATH,
        PROGRAM_PATH,
        ACTIVE_ROADMAP_PATH,
        DESIGN_PATH,
        EXP7121_PATH,
        EXP7121_WRAPPER_PATH,
        ROADMAP_SCHEMA_PATH,
        ROADMAP_AUDIT_PATH,
        EXCLUSION_LINT_PATH,
        HARNESS_LINT_PATH,
        ADVERSARIAL_PATH,
        SPEC_PATH,
        HARNESS_SPEC_PATH,
        EXCLUSION_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        str(relative): _sha256_path(root / relative) if (root / relative).is_file() else None
        for relative in paths
    }


def _field_principles() -> dict[str, str]:
    """Attach one falsifiability reason to every required result field."""

    principles = {
        field: f"The {field} evidence lets a reviewer reproduce or falsify the V626 contract result."
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }
    principles["field_principles"] = "Each field records why its evidence is needed."
    principles["preconditions_checked"] = (
        "A no-run result must preserve the exact missing prerequisite."
    )
    principles["rows"] = "Per-task evidence prevents a pooled score from hiding one failed row."
    principles["existing_path_rows"] = (
        "Section-scoped path rows distinguish present inputs from future outputs."
    )
    principles["source_artifact_hashes"] = "Content hashes expose later source drift."
    principles["v626_task_contract_conforms_score"] = (
        "The score is one only when all 12 task contracts pass."
    )
    principles["reproducibility_checksum"] = "The checksum exposes any later artifact mutation."
    principles["honest_verdict"] = "The terminal verdict separates mismatch from missing input."
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the checksum storage field."""

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
        "schema": "carnot.exp7124.v626_contract_preflight.v1",
        "experiment_id": 7124,
        "status": "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "run_date": run_date,
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
        "v626_task_contract_conforms_score": 0,
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

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _contract_failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while row tables retain local causes."""

    if contract["markdown_task_count"] != EXPECTED_TASK_COUNT:
        check, observed = "markdown_task_count", contract["markdown_task_count"]
    elif len(contract["yaml_task_rows"]) != EXPECTED_TASK_COUNT:
        check, observed = "yaml_task_count", len(contract["yaml_task_rows"])
    else:
        first = next(row for row in contract["task_contract_rows"] if not row["passed"])
        return {
            "failed_check": f"task_contract_order_{first['order']}",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        }
    return {
        "failed_check": check,
        "expected_value": EXPECTED_TASK_COUNT,
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
            root=root,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": "contract_parse",
            "expected_value": "independently_parseable_markdown_and_yaml",
            "observed_value": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
        return _finish(artifact, started)

    for field in CONTRACT_ROW_FIELDS:
        artifact[field] = contract[field]
    artifact["markdown_task_rows"] = contract["markdown_task_rows"]
    artifact["yaml_task_rows"] = contract["yaml_task_rows"]
    artifact["task_contract_rows"] = contract["task_contract_rows"]
    artifact["rows"] = contract["task_contract_rows"]
    artifact["observed_task_count"] = len(contract["yaml_task_rows"])
    artifact["observed_id_order"] = contract["observed_id_order"]
    score = int(contract["passed"])
    artifact["v626_task_contract_conforms_score"] = score
    if score == 1:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = POSITIVE_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": None,
            "expected_value": "all_12_task_contracts_conform",
            "observed_value": "all_12_task_contracts_conform",
            "passed": True,
        }
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = _contract_failure_summary(contract)
    return _finish(artifact, started)


def _contract_score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute the headline score from stored row evidence."""

    markdown_rows = artifact.get("markdown_task_rows")
    yaml_rows = artifact.get("yaml_task_rows")
    task_rows = artifact.get("task_contract_rows")
    if not all(isinstance(rows, list) for rows in (markdown_rows, yaml_rows, task_rows)):
        return 0
    if not all(
        len(rows) == EXPECTED_TASK_COUNT for rows in (markdown_rows, yaml_rows, task_rows)
    ):
        return 0
    if [row.get("id") for row in markdown_rows] != list(EXPECTED_TASK_IDS):
        return 0
    if [row.get("id") for row in yaml_rows] != list(EXPECTED_TASK_IDS):
        return 0
    if not all(row.get("passed") is True for row in task_rows):
        return 0
    for field in CONTRACT_ROW_FIELDS:
        rows = artifact.get(field)
        if (
            not isinstance(rows, list)
            or not rows
            or not all(row.get("passed") is True for row in rows)
        ):
            return 0
    return 1


def validate_artifact(artifact: object) -> list[str]:
    """Recompute required fields, score, terminal class, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), str) or not principles[field].strip()
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles_invalid")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact["execution_venue"] != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if artifact["expected_task_count"] != EXPECTED_TASK_COUNT:
        errors.append("expected_task_count_invalid")
    if artifact["expected_id_order"] != list(EXPECTED_TASK_IDS):
        errors.append("expected_id_order_invalid")
    if artifact["active_roadmap_path"] != str(ACTIVE_ROADMAP_PATH):
        errors.append("active_roadmap_path_invalid")
    if artifact["staging_file_required_at_execution"] is not False:
        errors.append("staging_file_required_invalid")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if (
        not isinstance(artifact["duration_s"], (int, float))
        or isinstance(artifact["duration_s"], bool)
        or artifact["duration_s"] < 0
    ):
        errors.append("duration_s_invalid")
    if not re.fullmatch(r"\d{8}", str(artifact["run_date"])):
        errors.append("run_date_invalid")
    if not isinstance(artifact["source_artifact_hashes"], Mapping):
        errors.append("source_artifact_hashes_invalid")
    yaml_rows = artifact["yaml_task_rows"]
    observed_count = len(yaml_rows) if isinstance(yaml_rows, list) else None
    observed_order = [row.get("id") for row in yaml_rows] if isinstance(yaml_rows, list) else None
    if artifact["observed_task_count"] != observed_count:
        errors.append("observed_task_count_invalid")
    if artifact["observed_id_order"] != observed_order:
        errors.append("observed_id_order_invalid")
    if artifact["rows"] != artifact["task_contract_rows"]:
        errors.append("rows_not_task_contract_rows")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, list) or not all(
        isinstance(row, Mapping) and isinstance(row.get("available"), bool)
        for row in preconditions
    ):
        errors.append("preconditions_checked_invalid")
        blocked = False
    else:
        blocked = any(row.get("available") is False for row in preconditions)
    score = 0 if blocked else _contract_score_from_artifact(artifact)
    if artifact["v626_task_contract_conforms_score"] != score:
        errors.append("v626_task_contract_conforms_score_invalid")
    expected_class = "blocked" if blocked else ("positive" if score == 1 else "disqualified")
    expected_verdict = {
        "blocked": BLOCKED_VERDICT,
        "positive": POSITIVE_VERDICT,
        "disqualified": DISQUALIFIED_VERDICT,
    }[expected_class]
    expected_substrate_class = "blocked_no_run" if blocked else INFERENCE_SUBSTRATE_CLASS
    if artifact["inference_substrate_class"] != expected_substrate_class:
        errors.append("inference_substrate_class_invalid")
    if artifact["verdict_class"] != expected_class:
        errors.append("verdict_class_not_derived")
    if artifact["honest_verdict"] != expected_verdict:
        errors.append("honest_verdict_not_derived")
    summary = artifact["gate_check_summary"]
    if not isinstance(summary, Mapping) or not all(
        key in summary for key in ("failed_check", "expected_value", "observed_value", "passed")
    ):
        errors.append("gate_check_summary_invalid")
    elif blocked:
        failed = next(row for row in preconditions if row.get("available") is False)
        if (
            summary.get("failed_check") != failed.get("check")
            or summary.get("expected_value") != failed.get("expected_value")
            or summary.get("observed_value") != failed.get("observed_value")
            or summary.get("passed") is not False
        ):
            errors.append("gate_check_summary_invalid")
    elif summary.get("passed") is not (score == 1):
        errors.append("gate_check_summary_invalid")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _date_argument(value: str) -> str:
    """Accept only a real compact calendar date."""

    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be a real YYYYMMDD value") from exc
    return value


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write stable formatted JSON after its destination passed preflight."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V626 audit or validate one stored artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--validate-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.validate_artifact is not None:
        try:
            payload = json.loads(args.validate_artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"INVALID: {type(exc).__name__}: {exc}")
            return 1
        errors = validate_artifact(payload)
        print("VALID" if not errors else "INVALID: " + ", ".join(errors))
        return int(bool(errors))
    if args.date is None:
        parser.print_usage()
        return 2
    root = args.root.resolve()
    output_path = args.output if args.output.is_absolute() else root / args.output
    artifact = build_artifact(root, args.date, output_path=output_path)
    _write_artifact(output_path, artifact)
    errors = validate_artifact(artifact)
    print(output_path)
    if errors:
        print("INVALID: " + ", ".join(errors))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
