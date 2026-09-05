"""Audit the V616 Markdown contract against the active roadmap.

The audit reads the Markdown and YAML sources with separate parsers. It never
uses the staging roadmap. Missing inputs produce a blocked artifact. Contract
differences produce a disqualified artifact with the observed rows intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
STAGING_ROADMAP_PATH = Path("research-roadmap-next.yaml")
V615_PREFLIGHT_PATH = Path("results/experiment_7016_v615_source_contract_preflight.json")
V615_CAPSTONE_PATH = Path("results/experiment_7027_v615_capstone.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROGRAM_PATH = Path("research-program.md")
REFERENCE_PATH = Path("research-references.md")
CLAUDE_PATH = Path("CLAUDE.md")
CODEX_PATH = Path("CODEX.md")
CONDUCTOR_GATES_PATH = Path("scripts/conductor_gates.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
MODULE_PATH = Path("python/carnot/experiment_7028_v616_active_contract_preflight.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7028_v616_active_contract_preflight.py")
TEST_PATH = Path("tests/python/test_experiment_7028_v616_active_contract_preflight.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7028_v616_active_contract_preflight.json")

MILESTONE = "2026.09.616"
EXPECTED_TASK_IDS = (
    "exp7028-v616-active-contract-preflight",
    "exp7029-v616-sota-scope-audit",
    "exp7030-arc-gguf-model-identity-bridge",
    "exp7031-arc-model-identity-cold-audit",
    "exp7032-repaired-belief-shadow-live-trace",
    "exp7033-uniform-belief-live-ab",
    "exp7034-live-belief-cold-audit",
    "exp7035-selective-belief-csl",
    "exp7036-belief-release-or-retire",
    "exp7037-v616-capstone",
)
EXPECTED_TASK_COUNT = 10
EXPECTED_GATE_COUNT = 8
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PROMPT_TAIL = "Do NOT push. Do NOT modify scripts/research_conductor.py."
RANDOM_SEED = 702820260905

LEGAL_INFERENCE_SUBSTRATES = {
    "live_llm_inference",
    "verifier_ensemble_against_cached_candidates",
    "aggregation_from_upstream_artifacts",
    "hardware_smoke",
    "offline_arcade_live_agent_runtime_self_discovery_no_llm",
    "live_llm_embedding_extraction",
}
LIVE_LLM_SUBSTRATES = frozenset({"live_llm_inference", "live_llm_embedding_extraction"})
MANDATED_SOTA_GGUFS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-12B-it-GGUF",
)
LEGACY_SMALL_MODELS = ("Qwen3.5-0.8B", "gemma-4-E4B", "Gemma4-E4B")

BLOCKED_VERDICT = "complete_blocked_v616_active_contract_preflight_prerequisite_missing"
POSITIVE_VERDICT = "complete_positive_v616_task_contract_conforms"
DISQUALIFIED_VERDICT = "complete_disqualified_v616_markdown_yaml_contract_mismatch"

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
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
        "retired_id_rows",
        "model_compliance_rows",
        "substrate_compliance_rows",
        "artifact_field_rows",
        "prompt_tail_rows",
        "command_receipt_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "active_roadmap_path",
        "staging_file_required_at_execution",
        "v616_task_contract_conforms_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)

CONTRACT_ROW_FIELDS = (
    "task_contract_rows",
    "title_parity_rows",
    "deliverable_parity_rows",
    "gate_contract_rows",
    "gate_producer_rows",
    "prior_failure_rows",
    "retired_id_rows",
    "model_compliance_rows",
    "substrate_compliance_rows",
    "artifact_field_rows",
    "prompt_tail_rows",
)

COMPARATIVE_REQUIRED_FIELDS = (
    "field_principles",
    "verdict_class",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
)


def _task_number(task_id: object) -> int | None:
    """Return an experiment number so the fixed order stays machine-checkable."""

    match = re.match(r"exp(\d+)(?:-|$)", str(task_id), re.I)
    return int(match.group(1)) if match else None


def _normal_exp_id(value: object) -> str | None:
    """Normalize historical integer and full-string experiment identifiers."""

    candidate = f"exp{value}" if isinstance(value, int) else value
    number = _task_number(candidate)
    return f"exp{number}" if number is not None else None


def _parse_scalar(text: str) -> Any:
    """Parse one gate value and reject data structures that gates cannot compare."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("structured gate value must be scalar")
    return value


def parse_markdown_contract(text: str) -> dict[str, Any]:
    """Parse the Markdown table without using any active-YAML task objects."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    heading = "## 7. Phases and Exact Task Contract"
    if heading not in text:
        raise ValueError("Markdown task contract section is missing")
    section = text.split(heading, 1)[1].split("\n### Phase I", 1)[0]
    milestone = milestone_match.group(1)
    tasks: list[dict[str, Any]] = []
    prerequisites: list[str] = []
    for line in section.splitlines():
        if re.match(r"^\|\s*\d+\s*\|", line) is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError(f"malformed Markdown task row: {line}")
        order_text, task_cell, title, deliverable_cell, prerequisite = cells
        task_id = task_cell.strip("`")
        number = _task_number(task_id)
        deliverable = deliverable_cell.strip("`")
        if number is None or not title or not deliverable:
            raise ValueError(f"malformed Markdown task row: {line}")
        tasks.append(
            {
                "order": int(order_text),
                "id": task_id,
                "number": number,
                "title": title,
                "deliverable": deliverable,
                "milestone": milestone,
            }
        )
        prerequisites.append(prerequisite)
    if not tasks:
        raise ValueError("Markdown task table is missing")

    id_by_number = {task["number"]: task["id"] for task in tasks}
    gate_pattern = re.compile(
        r"Exp(\d+)\s+`([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*([^`]+)`",
        re.I,
    )
    for task, prerequisite in zip(tasks, prerequisites, strict=True):
        gates = []
        for part in prerequisite.split(";"):
            part = part.strip()
            if not part or part.lower().startswith("none") or part.lower() == "structurally ungated":
                continue
            match = gate_pattern.fullmatch(part)
            if match is None:
                raise ValueError(f"malformed Markdown structured gate: {part}")
            upstream_number = int(match.group(1))
            gates.append(
                {
                    "upstream": id_by_number.get(upstream_number, f"exp{upstream_number}-missing"),
                    "artifact_field": match.group(2),
                    "op": match.group(3),
                    "value": _parse_scalar(match.group(4).strip()),
                }
            )
        task["gates"] = gates
    return {"milestone": milestone, "tasks": tasks}


def _required_block(prompt: str) -> str:
    """Isolate declarations so ordinary prompt prose cannot satisfy a field check."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def _field_declared(block: str, field: str) -> bool:
    """Match a bare field and reject a nested or similarly named field."""

    pattern = rf"(?<![A-Za-z0-9_.]){re.escape(field)}(?![A-Za-z0-9_.])"
    return re.search(pattern, block) is not None


def _substrate(prompt: str) -> str | None:
    """Read the exact planned substrate even when the prompt wraps the line."""

    match = re.search(r"inference_substrate\s*\(([^)]+)\)", prompt, re.S)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def parse_yaml_contract(document: object) -> dict[str, Any]:
    """Parse active YAML without calling or sharing the Markdown parser."""

    if not isinstance(document, Mapping):
        raise ValueError("YAML roadmap mapping is required")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("YAML roadmap tasks list is required")
    tasks = []
    for order, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"YAML task {order} must be a mapping")
        raw_gates = raw_task.get("gated_on") or []
        raw_priors = raw_task.get("prior_failures") or []
        if not isinstance(raw_gates, list) or not isinstance(raw_priors, list):
            raise ValueError(f"YAML task {order} has a malformed list")
        gates = []
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
        task_id = str(raw_task.get("id", ""))
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "number": _task_number(task_id),
                "title": raw_task.get("title"),
                "deliverable": raw_task.get("deliverable"),
                "milestone": raw_task.get("milestone", document.get("milestone")),
                "requires_gpu": raw_task.get("requires_gpu"),
                "per_unit_rows": raw_task.get("per_unit_rows"),
                "agent_type": raw_task.get("agent_type"),
                "model": raw_task.get("model"),
                "gates": gates,
                "prior_failures": list(raw_priors),
                "operator_override": raw_task.get("operator_override"),
                "prompt": prompt,
                "required_block": _required_block(prompt),
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one non-empty YAML mapping and reject every other top-level shape."""

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
    """Convert gates to ordered tuples so no key-order difference affects parity."""

    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
    ]


def _prior_rows(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check all four failure-history fields without filling missing values."""

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
    rows = []
    for prior in priors:
        values = dict(prior) if isinstance(prior, Mapping) else {}
        field_checks = {
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
                "field_checks": field_checks,
                "passed": all(field_checks.values()),
            }
        )
    return rows


def _comparison_prompt(prompt: str) -> bool:
    """Find tasks whose claims need per-unit rows instead of pooled summaries."""

    pattern = r"\b(?:compare|comparison|comparative|controls?|arms?|parity|paired|versus|A/B)\b"
    return re.search(pattern, prompt, re.I) is not None


def _documented_override(value: object) -> bool:
    """Accept an override only when it cites a dated operator directive."""

    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return re.search(r"\d{4}-\d{2}-\d{2}.*operator directive", value, re.I | re.S) is not None


def _legacy_small_headline_path(prompt: str) -> bool:
    """Detect an allowed legacy-small headline path without flagging a prohibition."""

    for line in prompt.splitlines():
        lowered = line.lower()
        if not any(model.lower() in lowered for model in LEGACY_SMALL_MODELS):
            continue
        if "headline" not in lowered and "fallback" not in lowered:
            continue
        if not any(word in lowered for word in ("no ", "forbid", "not allowed", "must not")):
            return True
    return False


def evaluate_contract(
    markdown_text: str, yaml_document: object, retired_ids: set[str]
) -> dict[str, Any]:
    """Compare V616 rows and evaluate each active-YAML execution rule."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    normalized_retired = {
        normalized
        for value in retired_ids
        if (normalized := _normal_exp_id(value)) is not None
    }

    title_rows: list[dict[str, Any]] = []
    deliverable_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    retired_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    substrate_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))

    for index in range(width):
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        order = index + 1
        number = (
            markdown_task["number"]
            if markdown_task is not None
            else yaml_task["number"] if yaml_task is not None else None
        )
        task_id = (
            markdown_task["id"]
            if markdown_task is not None
            else yaml_task["id"] if yaml_task is not None else None
        )
        same_id = bool(
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
                "passed": bool(same_id and markdown_task["title"] == yaml_task["title"]),
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
                    same_id and markdown_task["deliverable"] == yaml_task["deliverable"]
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
                    same_id
                    and _canonical_gates(markdown_task["gates"])
                    == _canonical_gates(yaml_task["gates"])
                ),
            }
        )
        if yaml_task is None:
            missing = {"order": order, "number": number, "task_id": task_id, "passed": False}
            prior_rows.append({**missing, "prior_failure_present": False})
            retired_rows.append(dict(missing))
            model_rows.append({**missing, "live_llm_task": False})
            substrate_rows.append({**missing, "inference_substrate": None})
            artifact_rows.append(dict(missing))
            tail_rows.append({**missing, "expected": PROMPT_TAIL, "observed": None})
            continue

        prior_rows.extend(_prior_rows(yaml_task))
        current_id = _normal_exp_id(yaml_task["id"])
        retired_upstreams = sorted(
            normalized
            for gate in yaml_task["gates"]
            if (normalized := _normal_exp_id(gate.get("upstream"))) in normalized_retired
        )
        override_documented = _documented_override(yaml_task["operator_override"])
        retired_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "task_retired": current_id in normalized_retired,
                "retired_upstreams": retired_upstreams,
                "operator_override": yaml_task["operator_override"],
                "operator_override_documented": override_documented,
                "passed": current_id not in normalized_retired
                and not retired_upstreams
                and override_documented,
            }
        )

        prompt = yaml_task["prompt"]
        block = yaml_task["required_block"]
        substrate = _substrate(prompt)
        substrate_legal = substrate in LEGAL_INFERENCE_SUBSTRATES
        substrate_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "inference_substrate": substrate,
                "legal_values": sorted(LEGAL_INFERENCE_SUBSTRATES),
                "passed": substrate_legal,
            }
        )

        live_llm = substrate in LIVE_LLM_SUBSTRATES
        model_specs = _field_declared(block, "MODEL_SPECS")
        sota_models = [model for model in MANDATED_SOTA_GGUFS if model in prompt]
        legacy_headline = _legacy_small_headline_path(prompt)
        model_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "live_llm_task": live_llm,
                "model_specs_declared": model_specs,
                "mandated_sota_ggufs": sota_models,
                "legacy_small_headline_path": legacy_headline,
                "passed": not live_llm or (model_specs and bool(sota_models) and not legacy_headline),
            }
        )

        comparison = _comparison_prompt(prompt)
        fields_present = all(
            _field_declared(block, field) for field in COMPARATIVE_REQUIRED_FIELDS
        )
        normalized_block = re.sub(r"\s+", " ", block.lower())
        diagnostics = all(
            phrase in normalized_block
            for phrase in ("failed check", "expected value", "observed value")
        )
        capstone_ungated = not yaml_task["gates"] if yaml_task["number"] == 7037 else None
        advisory_ungated = not yaml_task["gates"] if yaml_task["number"] == 7028 else None
        artifact_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "comparison": comparison,
                "per_unit_rows": yaml_task["per_unit_rows"],
                "comparative_fields_present": fields_present,
                "blocked_diagnostics_present": diagnostics,
                "capstone_ungated": capstone_ungated,
                "advisory_preflight_ungated": advisory_ungated,
                "passed": bool(block)
                and diagnostics
                and (not comparison or (fields_present and yaml_task["per_unit_rows"] is True))
                and capstone_ungated is not False
                and advisory_ungated is not False,
            }
        )
        observed_tail = prompt.rstrip().splitlines()[-1] if prompt.rstrip() else ""
        tail_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "expected": PROMPT_TAIL,
                "observed": observed_tail,
                "passed": observed_tail == PROMPT_TAIL,
            }
        )

    yaml_by_id = {task["id"]: task for task in yaml_tasks}
    gate_producer_rows = []
    for consumer in yaml_tasks:
        for gate in consumer["gates"]:
            upstream_id = gate.get("upstream")
            producer = yaml_by_id.get(upstream_id)
            field = gate.get("artifact_field")
            bare_field = isinstance(field, str) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", field
            ) is not None
            producer_precedes = bool(producer and producer["order"] < consumer["order"])
            same_milestone = bool(
                producer and producer["milestone"] == consumer["milestone"] == MILESTONE
            )
            field_declared = bool(
                producer and bare_field and _field_declared(producer["required_block"], field)
            )
            upstream_retired = _normal_exp_id(upstream_id) in normalized_retired
            passed = bool(
                producer
                and producer_precedes
                and same_milestone
                and bare_field
                and field_declared
                and not upstream_retired
            )
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
                    "upstream_retired": upstream_retired,
                    "passed": passed,
                }
            )

    check_rows = (
        title_rows,
        deliverable_rows,
        gate_rows,
        prior_rows,
        retired_rows,
        model_rows,
        substrate_rows,
        artifact_rows,
        tail_rows,
        gate_producer_rows,
    )
    checks_by_order: dict[int, list[bool]] = {}
    for rows in check_rows:
        for row in rows:
            checks_by_order.setdefault(row["order"], []).append(row["passed"] is True)

    task_rows = []
    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        expected_id = EXPECTED_TASK_IDS[index] if index < EXPECTED_TASK_COUNT else None
        passed = bool(
            markdown_task
            and yaml_task
            and markdown_task["order"] == yaml_task["order"] == order
            and markdown_task["id"] == yaml_task["id"] == expected_id
            and markdown_task["milestone"] == yaml_task["milestone"] == MILESTONE
            and all(checks_by_order.get(order, [False]))
        )
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
                "passed": passed,
            }
        )

    global_checks = (
        markdown["milestone"] == MILESTONE,
        roadmap["milestone"] == MILESTONE,
        len(markdown_tasks) == EXPECTED_TASK_COUNT,
        len(yaml_tasks) == EXPECTED_TASK_COUNT,
        tuple(task["id"] for task in markdown_tasks) == EXPECTED_TASK_IDS,
        tuple(task["id"] for task in yaml_tasks) == EXPECTED_TASK_IDS,
        len(gate_producer_rows) == EXPECTED_GATE_COUNT,
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
        "retired_id_rows": retired_rows,
        "model_compliance_rows": model_rows,
        "substrate_compliance_rows": substrate_rows,
        "artifact_field_rows": artifact_rows,
        "prompt_tail_rows": tail_rows,
        "markdown_task_count": len(markdown_tasks),
        "observed_id_order": [task["id"] for task in yaml_tasks],
    }


def _sha256_path(path: Path) -> str:
    """Hash exact input bytes so later reviewers can detect source drift."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path_writable(path: Path) -> tuple[bool, str]:
    """Test the output directory before the audit depends on writing there."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7028-write-", delete=True):
            pass
        return True, "temporary_write_succeeded"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _preconditions(
    root: Path, output_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Check all execution inputs while treating the staging path as optional."""

    rows: list[dict[str, Any]] = []
    roadmap_document: dict[str, Any] | None = None
    exclusion_document: dict[str, Any] | None = None

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
        text = design_path.read_text(encoding="utf-8")
        available = bool(text.strip())
        add(
            "v616_markdown_readable",
            design_path,
            "readable_nonempty_source",
            available,
            "readable_nonempty_source" if available else "empty_source",
        )
    except OSError as exc:
        add(
            "v616_markdown_readable",
            design_path,
            "readable_nonempty_source",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    roadmap_path = root / ACTIVE_ROADMAP_PATH
    try:
        roadmap_document = load_yaml(roadmap_path)
        add(
            "v616_active_yaml_readable",
            roadmap_path,
            "readable_nonempty_yaml_mapping",
            True,
            "readable_nonempty_yaml_mapping",
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        add(
            "v616_active_yaml_readable",
            roadmap_path,
            "readable_nonempty_yaml_mapping",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    for check, relative in (
        ("v615_preflight_evidence_readable", V615_PREFLIGHT_PATH),
        ("v615_capstone_evidence_readable", V615_CAPSTONE_PATH),
    ):
        path = root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            available = isinstance(value, dict) and bool(value)
            add(
                check,
                path,
                "readable_nonempty_json_mapping",
                available,
                "readable_nonempty_json_mapping" if available else "nonempty_JSON_mapping_required",
            )
        except (OSError, ValueError) as exc:
            add(
                check,
                path,
                "readable_nonempty_json_mapping",
                False,
                f"{type(exc).__name__}: {exc}",
            )

    exclusion_path = root / EXCLUSION_PATH
    try:
        exclusion_document = load_yaml(exclusion_path)
        add(
            "exclusion_manifest_readable",
            exclusion_path,
            "readable_nonempty_yaml_mapping",
            True,
            "readable_nonempty_yaml_mapping",
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        add(
            "exclusion_manifest_readable",
            exclusion_path,
            "readable_nonempty_yaml_mapping",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    spec_path = root / SPEC_PATH
    try:
        spec_text = spec_path.read_text(encoding="utf-8")
        available = "REQ-REPORT-7028" in spec_text
        add(
            "reporting_spec_readable",
            spec_path,
            "REQ-REPORT-7028 present",
            available,
            "REQ-REPORT-7028 present" if available else "REQ-REPORT-7028 absent",
        )
    except OSError as exc:
        add(
            "reporting_spec_readable",
            spec_path,
            "REQ-REPORT-7028 present",
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
    add(
        "artifact_path_writable",
        output_path,
        "temporary_write_succeeds",
        writable,
        observed,
    )
    return rows, roadmap_document, exclusion_document


def _source_hashes(root: Path) -> dict[str, str | None]:
    """Hash every available input and keep absent optional paths explicit."""

    paths = (
        CLAUDE_PATH,
        CODEX_PATH,
        PROGRAM_PATH,
        REFERENCE_PATH,
        DESIGN_PATH,
        ACTIVE_ROADMAP_PATH,
        STAGING_ROADMAP_PATH,
        V615_PREFLIGHT_PATH,
        V615_CAPSTONE_PATH,
        EXCLUSION_PATH,
        CONDUCTOR_GATES_PATH,
        ROW_LINT_PATH,
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        str(relative): _sha256_path(root / relative) if (root / relative).is_file() else None
        for relative in paths
    }


def _field_principles() -> dict[str, str]:
    """Give each artifact field a falsifiability reason, not only a label."""

    principles = {
        field: (
            f"Preserving {field} lets an independent reviewer reproduce or falsify "
            "the contract conclusion."
        )
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }
    principles["field_principles"] = (
        "Scientific fields need stated reasons so compliance cannot replace understanding."
    )
    principles["preconditions_checked"] = (
        "A conclusion is valid only when every execution input was available before analysis."
    )
    principles["source_artifact_hashes"] = (
        "Content hashes expose input drift between this audit and a later replication."
    )
    principles["rows"] = (
        "Per-task observations prevent a pooled score from hiding a local contract failure."
    )
    principles["v616_task_contract_conforms_score"] = (
        "The score is one only when all 10 rows and all eight producer gates pass."
    )
    principles["reproducibility_checksum"] = (
        "The checksum detects any change to the completed audit record."
    )
    principles["honest_verdict"] = (
        "A terminal class prevents a missing input or mismatch from appearing positive."
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact fields except the checksum that contains the hash."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _empty_contract() -> dict[str, list[Any]]:
    """Keep blocked artifacts schema-complete before contract rows can exist."""

    return {field: [] for field in CONTRACT_ROW_FIELDS} | {
        "markdown_task_rows": [],
        "yaml_task_rows": [],
    }


def _base_artifact(root: Path, run_date: str) -> dict[str, Any]:
    """Create every required field before selecting a terminal state."""

    return {
        "schema": "carnot.exp7028.v616_active_contract_preflight.v1",
        "experiment_id": 7028,
        "run_date": run_date,
        "status": "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": _source_hashes(root),
        "rows": [],
        **_empty_contract(),
        "command_receipt_rows": [],
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_TASK_IDS),
        "observed_id_order": [],
        "active_roadmap_path": str(ACTIVE_ROADMAP_PATH),
        "staging_file_required_at_execution": False,
        "v616_task_contract_conforms_score": 0,
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
    """Freeze the measured duration and checksum after deriving every field."""

    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _contract_failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while detailed rows retain every difference."""

    if contract["markdown_task_count"] != EXPECTED_TASK_COUNT:
        check = "markdown_task_count"
        expected = EXPECTED_TASK_COUNT
        observed = contract["markdown_task_count"]
    elif len(contract["yaml_task_rows"]) != EXPECTED_TASK_COUNT:
        check = "observed_task_count"
        expected = EXPECTED_TASK_COUNT
        observed = len(contract["yaml_task_rows"])
    else:
        first = next(row for row in contract["task_contract_rows"] if not row["passed"])
        check = f"task_contract_order_{first['order']}"
        expected = True
        observed = False
    return {
        "failed_check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    command_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a positive, disqualified, or prerequisite-blocked V616 artifact."""

    started = time.monotonic()
    artifact = _base_artifact(root, run_date)
    preconditions, roadmap_document, exclusion_document = _preconditions(root, output_path)
    artifact["preconditions_checked"] = preconditions
    failed_precondition = next((row for row in preconditions if not row["available"]), None)
    if failed_precondition is not None:
        artifact["gate_check_summary"] = {
            "failed_check": failed_precondition["check"],
            "expected_value": failed_precondition["expected_value"],
            "observed_value": failed_precondition["observed_value"],
            "passed": False,
        }
        return _finish(artifact, started)

    try:
        markdown_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = evaluate_contract(
            markdown_text,
            roadmap_document,
            retired_experiment_ids(exclusion_document),
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["rows"] = [
            {
                "row_kind": "contract_parse_error",
                "observed": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        ]
        artifact["gate_check_summary"] = {
            "failed_check": "contract_parse",
            "expected_value": "two_parseable_independent_contracts",
            "observed_value": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
        return _finish(artifact, started)

    for field in ("markdown_task_rows", "yaml_task_rows", *CONTRACT_ROW_FIELDS):
        artifact[field] = contract[field]
    artifact["rows"] = [dict(row) for row in contract["task_contract_rows"]]
    artifact["observed_task_count"] = len(contract["yaml_task_rows"])
    artifact["observed_id_order"] = contract["observed_id_order"]
    artifact["v616_task_contract_conforms_score"] = int(contract["passed"])
    receipts = [dict(row) for row in command_rows] if command_rows is not None else []
    artifact["command_receipt_rows"] = receipts

    if not contract["passed"]:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = _contract_failure_summary(contract)
    else:
        failed_command = next(
            (
                row
                for row in receipts
                if row.get("terminal") is not True or row.get("passed") is not True
            ),
            None,
        )
        if failed_command is not None:
            artifact["verdict_class"] = "blocked"
            artifact["honest_verdict"] = BLOCKED_VERDICT
            artifact["gate_check_summary"] = {
                "failed_check": str(failed_command.get("name", "validation_command")),
                "expected_value": "terminal_passing_receipt",
                "observed_value": failed_command.get("outcome", "failed"),
                "passed": False,
            }
        else:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = POSITIVE_VERDICT
            artifact["gate_check_summary"] = {
                "failed_check": None,
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            }
    return _finish(artifact, started)


def _contract_score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute conformance from the preserved task and producer evidence."""

    task_rows = artifact.get("task_contract_rows")
    if not isinstance(task_rows, list) or len(task_rows) != EXPECTED_TASK_COUNT:
        return 0
    exact_counts = {
        "title_parity_rows": EXPECTED_TASK_COUNT,
        "deliverable_parity_rows": EXPECTED_TASK_COUNT,
        "gate_contract_rows": EXPECTED_TASK_COUNT,
        "gate_producer_rows": EXPECTED_GATE_COUNT,
        "retired_id_rows": EXPECTED_TASK_COUNT,
        "model_compliance_rows": EXPECTED_TASK_COUNT,
        "substrate_compliance_rows": EXPECTED_TASK_COUNT,
        "artifact_field_rows": EXPECTED_TASK_COUNT,
        "prompt_tail_rows": EXPECTED_TASK_COUNT,
    }
    for field in CONTRACT_ROW_FIELDS:
        rows = artifact.get(field)
        if not isinstance(rows, list) or not all(
            isinstance(row, Mapping) and row.get("passed") is True for row in rows
        ):
            return 0
        if field in exact_counts and len(rows) != exact_counts[field]:
            return 0
    prior_rows = artifact.get("prior_failure_rows")
    if not isinstance(prior_rows, list) or len(prior_rows) < EXPECTED_TASK_COUNT:
        return 0
    return 1


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, score, terminal verdict, diagnostics, and checksum."""

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

    contract_score = _contract_score_from_artifact(artifact)
    if artifact.get("v616_task_contract_conforms_score") != contract_score:
        errors.append("v616_task_contract_conforms_score_invalid")
    preconditions = artifact.get("preconditions_checked")
    preconditions_ok = bool(preconditions) and all(
        isinstance(row, Mapping) and row.get("available") is True for row in preconditions
    )
    receipts = artifact.get("command_receipt_rows")
    commands_ok = isinstance(receipts, list) and all(
        isinstance(row, Mapping)
        and row.get("terminal") is True
        and row.get("passed") is True
        for row in receipts
    )
    if not preconditions_ok:
        expected_class, expected_verdict = "blocked", BLOCKED_VERDICT
    elif not contract_score:
        expected_class, expected_verdict = "disqualified", DISQUALIFIED_VERDICT
    elif not commands_ok:
        expected_class, expected_verdict = "blocked", BLOCKED_VERDICT
    else:
        expected_class, expected_verdict = "positive", POSITIVE_VERDICT
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_not_derived")
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict_not_derived")

    summary = artifact.get("gate_check_summary")
    summary_valid = isinstance(summary, Mapping) and {
        "failed_check",
        "expected_value",
        "observed_value",
        "passed",
    } <= summary.keys()
    if not summary_valid:
        errors.append("gate_check_summary_invalid")
    elif (expected_class == "positive") != (summary.get("passed") is True):
        errors.append("gate_check_summary_not_derived")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _run_command(root: Path, name: str, argv: Sequence[str]) -> dict[str, Any]:
    """Run one bounded validation and preserve its output for later review."""

    command = [str(value) for value in argv]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        passed = completed.returncode == 0
        return {
            "name": name,
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
            "outcome": "pass" if passed else "failed",
            "terminal": True,
            "passed": passed,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": " ".join(command),
            "exit_code": None,
            "stdout": str(exc.stdout or "")[-8000:],
            "stderr": str(exc.stderr or "")[-8000:],
            "outcome": "timeout",
            "terminal": False,
            "passed": False,
        }
    except OSError as exc:
        return {
            "name": name,
            "command": " ".join(command),
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "outcome": "tool_error",
            "terminal": False,
            "passed": False,
        }


def run_validation_commands(root: Path, artifact_path: Path) -> list[dict[str, Any]]:
    """Run the focused coverage, contract, artifact, and repository checks."""

    python = root / ".venv/bin/python"
    coverage = root / ".venv/bin/coverage"
    coverage_data = Path(tempfile.gettempdir()) / "carnot-exp7028-v616-coverage"
    script = root / WRAPPER_PATH
    commands = (
        (
            "focused_tests",
            [
                coverage,
                "run",
                f"--data-file={coverage_data}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                TEST_PATH,
                "-q",
            ],
        ),
        (
            "new_code_100pct_coverage",
            [
                coverage,
                "report",
                f"--data-file={coverage_data}",
                "--include=*/experiment_7028_v616_active_contract_preflight.py",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "yaml_parsing",
            [python, "-c", "import yaml; yaml.safe_load(open('research-roadmap.yaml'))"],
        ),
        (
            "exclusion_manifest_pre_emit_lint",
            [python, "scripts/exclusion_manifest_lint.py", ACTIVE_ROADMAP_PATH],
        ),
        ("artifact_validation", [python, script, "--validate-artifact", artifact_path]),
        (
            "adversarial_verification",
            [python, "scripts/adversarial_verify.py", "--high-precision-only", artifact_path],
        ),
        (
            "row_consistency_lint",
            [python, "scripts/verdict_row_consistency_lint.py", artifact_path],
        ),
        (
            "openspec_coverage",
            [python, "scripts/check_spec_coverage.py", TEST_PATH],
        ),
        ("root_clutter_check", [python, "scripts/root_clutter_sweep.py"]),
    )
    return [_run_command(root, name, argv) for name, argv in commands]


def _date_argument(value: str) -> str:
    """Require the compact execution-date format used by result artifacts."""

    if re.fullmatch(r"\d{8}", value) is None:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the result only after a complete JSON document reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the advisory audit or validate one existing artifact."""

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
        errors = validate_artifact(value) if isinstance(value, Mapping) else [
            "artifact_mapping_required"
        ]
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
    if artifact["verdict_class"] != "blocked":
        _write_json_atomic(output, artifact)
        receipts = run_validation_commands(root, output)
        artifact = build_artifact(root, args.date, output_path=output, command_rows=receipts)
    errors = validate_artifact(artifact)
    _write_json_atomic(output, artifact)
    if errors:
        print("invalid generated artifact: " + ", ".join(errors))
        return 1
    print(f"{artifact['honest_verdict']}: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns direct execution.
    raise SystemExit(main())
