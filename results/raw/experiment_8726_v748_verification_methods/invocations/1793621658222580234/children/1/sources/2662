"""Audit the V619 Markdown contract against the active roadmap.

The two parsers consume different source values. Missing execution inputs
produce a blocked artifact. Contract differences produce a disqualified
artifact. The audit never reads a staging roadmap as an execution input.
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

from carnot.experiment_7038_v617_active_contract_preflight import (
    _closed_verdict_enum,
    _comparison_prompt,
    _field_declared,
    _legacy_small_headline_path,
    _normal_exp_id,
    _parse_scalar,
    _required_block,
    _substrate,
    load_yaml,
    parse_yaml_contract as _parse_yaml_contract,
    retired_experiment_ids,
    task_number,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
STAGING_ROADMAP_PATH = Path("research-roadmap-next.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROGRAM_PATH = Path("research-program.md")
REFERENCE_PATH = Path("research-references.md")
CLAUDE_PATH = Path("CLAUDE.md")
CODEX_PATH = Path("CODEX.md")
ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
ROADMAP_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
HARNESS_LINT_PATH = Path("scripts/harness_fit_lint.py")
EXP7050_PATH = Path("results/experiment_7050_v618_active_contract_preflight.json")
EXP7050_MODULE_PATH = Path("python/carnot/experiment_7050_v618_active_contract_preflight.py")
MODULE_PATH = Path("python/carnot/experiment_7063_v619_contract_preflight.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7063_v619_contract_preflight.py")
TEST_PATH = Path("tests/python/test_experiment_7063_v619_contract_preflight.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7063_v619_contract_preflight.json")

MILESTONE = "2026.09.619"
EXPECTED_TASK_IDS = (
    "exp7063-v619-contract-preflight",
    "exp7064-exact-entrance-constraint-fixture",
    "exp7065-three-family-entrance-proposal-bank",
    "exp7066-entrance-bank-independent-audit",
    "exp7067-hopfield-entrance-energy-selection",
    "exp7068-hierarchical-branch-fidelity-control",
    "exp7069-context-bound-experience-contract",
    "exp7070-bcit-prospective-self-learning",
    "exp7071-bcit-drift-rollback-audit",
    "exp7072-live-arc-compaction-ab",
    "exp7073-entrance-energy-ising-parity",
    "exp7074-degree16-placement-sampler-audit",
    "exp7075-v619-capstone",
)
EXPECTED_TASK_COUNT = 13
EXPECTED_GATE_COUNT = 9
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
PROMPT_TAIL = "Do NOT push. Do NOT modify scripts/research_conductor.py."
RANDOM_SEED = 706320260906

LEGAL_INFERENCE_SUBSTRATES = {
    "live_llm_inference",
    "aggregation_from_upstream_artifacts",
    "deterministic_verifier",
    "deterministic_verifier_plus_replay",
    "deterministicCPUchronologicalcomparison",
    "fresh_process_no_llm_transaction_audit",
}
LIVE_MODEL_SUBSTRATES = frozenset({"live_llm_inference"})
MANDATED_SOTA_GGUFS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-12B-it-GGUF",
)
EXPECTED_LLM_MODELS = {
    7065: (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ),
    7072: (
        "unsloth/Qwen3.8-27B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ),
}
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)

BLOCKED_VERDICT = "complete_blocked_v619_contract_preflight_prerequisite_missing"
POSITIVE_VERDICT = "complete_positive_v619_task_contract_conforms"
DISQUALIFIED_VERDICT = "complete_disqualified_v619_markdown_yaml_contract_mismatch"

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
        "model_compliance_rows",
        "artifact_field_rows",
        "prompt_tail_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "active_roadmap_path",
        "staging_file_required_at_execution",
        "v619_task_contract_conforms_score",
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
    "artifact_field_rows",
    "prompt_tail_rows",
)
GENERIC_REQUIRED_FIELDS = (
    "field_principles",
    "verdict_class",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
)


def parse_markdown_contract(text: str) -> dict[str, Any]:
    """Parse the Markdown table and gate sections without a YAML task value."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\s]+)`?", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    section_match = re.search(
        r"^## Exact Task Contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.MULTILINE
    )
    if section_match is None:
        raise ValueError("Markdown task contract section is missing")

    tasks: list[dict[str, Any]] = []
    for line in section_match.group(1).splitlines():
        if re.match(r"^\|\s*\d+\s*\|", line) is None:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            raise ValueError(f"malformed Markdown task row: {line}")
        order_text, task_cell, title, deliverable_cell = cells
        task_id = task_cell.strip("`")
        number = task_number(task_id)
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
                "milestone": milestone_match.group(1),
                "gates": [],
            }
        )
    if not tasks:
        raise ValueError("Markdown task table is missing")

    tasks_by_number = {task["number"]: task for task in tasks}
    task_sections = re.finditer(
        r"^### Exp(\d+)\s+-[^\n]*\n([\s\S]*?)(?=^### |^## |\Z)",
        text,
        re.MULTILINE,
    )
    gate_pattern = re.compile(
        r"`(exp\d+-[A-Za-z0-9_-]+)\.([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(==|!=|>=|<=|>|<)\s*([^`]+)`"
    )
    for match in task_sections:
        number = int(match.group(1))
        body = match.group(2)
        marker = re.search(r"\*\*Gates?:\*\*([\s\S]*?)(?=\n\*\*|\Z)", body)
        if marker is None or number not in tasks_by_number:
            continue
        gate_lines = [line for line in marker.group(1).splitlines() if "`" in line]
        gates = []
        for line in gate_lines:
            gate_match = gate_pattern.search(line)
            if gate_match is None:
                raise ValueError(f"malformed Markdown structured gate: {line.strip()}")
            gates.append(
                {
                    "upstream": gate_match.group(1),
                    "artifact_field": gate_match.group(2),
                    "op": gate_match.group(3),
                    "value": _parse_scalar(gate_match.group(4).strip()),
                }
            )
        if not gates:
            raise ValueError(f"Markdown gate block for Exp{number} is empty")
        tasks_by_number[number]["gates"] = gates
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def parse_yaml_contract(document: object) -> dict[str, Any]:
    """Parse active YAML through a path that does not receive Markdown rows."""

    return _parse_yaml_contract(document)


def _canonical_gates(gates: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Make ordered gate comparisons independent of mapping key order."""

    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
    ]


def _prior_rows(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check all four history fields and preserve each prior as one row."""

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
                "field_checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _legacy_small_fallback_forbidden(prompt: str) -> bool:
    """Require an explicit ban so silence cannot look like a model policy."""

    for line in prompt.splitlines():
        lowered = line.lower()
        names_legacy = "legacy-small" in lowered or "legacy small" in lowered
        names_claim_path = "headline" in lowered or "fallback" in lowered
        forbids_path = any(
            marker in lowered for marker in ("no ", "forbid", "not allowed", "must not", "never ")
        )
        if names_legacy and names_claim_path and forbids_path:
            return True
    return False


def _task_fields_valid(task: Mapping[str, Any]) -> tuple[dict[str, bool], bool]:
    """Validate executable metadata against the two active agent backends."""

    checks = {
        "track": isinstance(task["track"], str) and bool(task["track"].strip()),
        "priority": isinstance(task["priority"], str) and bool(task["priority"].strip()),
        "agent_type": task["agent_type"] in {"claude", "codex"},
        "model": isinstance(task["model"], str) and bool(task["model"].strip()),
        "requires_gpu": isinstance(task["requires_gpu"], bool),
        "max_turns": isinstance(task["max_turns"], int) and task["max_turns"] > 0,
        "estimated_wall_time_min": isinstance(task["estimated_wall_time_min"], int)
        and task["estimated_wall_time_min"] > 0,
        "per_unit_rows": isinstance(task["per_unit_rows"], bool),
    }
    coherent = (task["agent_type"], task["model"]) in {
        ("claude", "sonnet"),
        ("claude", "opus"),
        ("codex", "gpt-5.6-sol"),
    }
    return checks, all(checks.values()) and coherent


def evaluate_contract(
    markdown_text: str,
    yaml_document: object,
    retired_ids: set[str],
) -> dict[str, Any]:
    """Compare every V619 row and apply the active-YAML execution rules."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    normalized_retired = {
        normalized for value in retired_ids if (normalized := _normal_exp_id(value)) is not None
    }
    yaml_number_counts = Counter(task["number"] for task in yaml_tasks)
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))

    title_rows: list[dict[str, Any]] = []
    deliverable_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    id_checks_by_order: dict[int, bool] = {}

    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        task_id = (
            markdown_task["id"]
            if markdown_task is not None
            else yaml_task["id"]
            if yaml_task is not None
            else None
        )
        number = (
            markdown_task["number"]
            if markdown_task is not None
            else yaml_task["number"]
            if yaml_task is not None
            else None
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
            model_rows.append({**missing, "llm_task": False})
            artifact_rows.append({**missing, "inference_substrate": None})
            tail_rows.append({**missing, "expected": PROMPT_TAIL, "observed": None})
            id_checks_by_order[order] = False
            continue

        prior_rows.extend(_prior_rows(yaml_task))
        normalized_id = _normal_exp_id(yaml_task["id"])
        unique_id = yaml_task["number"] is not None and yaml_number_counts[yaml_task["number"]] == 1
        absent_from_retired = normalized_id not in normalized_retired
        id_checks_by_order[order] = unique_id and absent_from_retired

        prompt = yaml_task["prompt"]
        block = yaml_task["required_block"]
        substrate = _substrate(prompt)
        llm_task = yaml_task["number"] in EXPECTED_LLM_MODELS
        live_model = substrate in LIVE_MODEL_SUBSTRATES
        sota_models = [model for model in MANDATED_SOTA_GGUFS if model in prompt]
        required_models = list(EXPECTED_LLM_MODELS.get(yaml_task["number"], ()))
        required_models_present = all(model in prompt for model in required_models)
        model_specs = _field_declared(block, "MODEL_SPECS")
        cached_pattern = "cached_sota_pair()" in prompt
        legacy_headline = _legacy_small_headline_path(prompt)
        legacy_fallback_forbidden = _legacy_small_fallback_forbidden(prompt)
        model_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "llm_task": llm_task,
                "live_model_substrate": live_model,
                "model_specs_declared": model_specs,
                "cached_sota_pair_pattern": cached_pattern,
                "mandated_sota_ggufs": sota_models,
                "required_model_declarations": required_models,
                "required_models_present": required_models_present,
                "legacy_small_headline_path": legacy_headline,
                "legacy_small_headline_fallback_forbidden": legacy_fallback_forbidden,
                "passed": not llm_task
                or (
                    live_model
                    and model_specs
                    and cached_pattern
                    and bool(sota_models)
                    and required_models_present
                    and legacy_fallback_forbidden
                    and not legacy_headline
                ),
            }
        )

        fields_present = all(_field_declared(block, field) for field in GENERIC_REQUIRED_FIELDS)
        normalized_block = re.sub(r"\s+", " ", block.lower())
        diagnostics = all(
            phrase in normalized_block
            for phrase in ("failed check", "expected value", "observed value")
        )
        comparison = _comparison_prompt(prompt)
        task_field_checks, task_fields_valid = _task_fields_valid(yaml_task)
        capstone_ungated = not yaml_task["gates"] if yaml_task["number"] == 7075 else None
        advisory_ungated = not yaml_task["gates"] if yaml_task["number"] == 7063 else None
        artifact_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "inference_substrate": substrate,
                "legal_inference_substrate": substrate in LEGAL_INFERENCE_SUBSTRATES,
                "comparison": comparison,
                "per_unit_rows": yaml_task["per_unit_rows"],
                "generic_fields_present": fields_present,
                "blocked_diagnostics_present": diagnostics,
                "closed_verdict_enum": _closed_verdict_enum(block),
                "task_field_checks": task_field_checks,
                "agent_model_coherent": task_fields_valid,
                "unique_id": unique_id,
                "absent_from_retired_ids": absent_from_retired,
                "capstone_ungated": capstone_ungated,
                "advisory_preflight_ungated": advisory_ungated,
                "passed": bool(block)
                and substrate in LEGAL_INFERENCE_SUBSTRATES
                and fields_present
                and diagnostics
                and _closed_verdict_enum(block)
                and task_fields_valid
                and (not comparison or yaml_task["per_unit_rows"] is True)
                and capstone_ungated is not False
                and advisory_ungated is not False
                and unique_id
                and absent_from_retired,
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
            normalized_upstream = _normal_exp_id(upstream_id)
            upstream_forbidden = normalized_upstream in normalized_retired
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
                    "passed": bool(
                        producer
                        and producer_precedes
                        and same_milestone
                        and bare_field
                        and field_declared
                        and not upstream_forbidden
                    ),
                }
            )

    supporting_rows = (
        title_rows,
        deliverable_rows,
        gate_rows,
        prior_rows,
        model_rows,
        artifact_rows,
        tail_rows,
        gate_producer_rows,
    )
    checks_by_order: dict[int, list[bool]] = {}
    for rows in supporting_rows:
        for row in rows:
            checks_by_order.setdefault(row["order"], []).append(row["passed"] is True)

    task_rows = []
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
                    and id_checks_by_order.get(order, False)
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
        len(yaml_number_counts) == EXPECTED_TASK_COUNT,
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
        "model_compliance_rows": model_rows,
        "artifact_field_rows": artifact_rows,
        "prompt_tail_rows": tail_rows,
        "markdown_task_count": len(markdown_tasks),
        "observed_id_order": [task["id"] for task in yaml_tasks],
    }


def _sha256_path(path: Path) -> str:
    """Hash exact source bytes so a later audit can detect drift."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path_writable(path: Path) -> tuple[bool, str]:
    """Test the destination before the audit depends on writing there."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7063-write-"):
            pass
        return True, "temporary_write_succeeded"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _preconditions(
    root: Path, output_path: Path
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Read required sources and record that staging is not an input."""

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
            "v619_markdown_readable",
            design_path,
            "readable_nonempty_source",
            available,
            "readable_nonempty_source" if available else "empty_source",
        )
    except OSError as exc:
        add(
            "v619_markdown_readable",
            design_path,
            "readable_nonempty_source",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    for check, relative, target in (
        ("v619_active_yaml_readable", ACTIVE_ROADMAP_PATH, "roadmap"),
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
    """Hash each available source and preserve missing sources as null."""

    paths = (
        CLAUDE_PATH,
        CODEX_PATH,
        PROGRAM_PATH,
        REFERENCE_PATH,
        DESIGN_PATH,
        ACTIVE_ROADMAP_PATH,
        EXP7050_PATH,
        EXP7050_MODULE_PATH,
        EXCLUSION_PATH,
        ROADMAP_SCHEMA_PATH,
        ROADMAP_AUDIT_PATH,
        EXCLUSION_LINT_PATH,
        HARNESS_LINT_PATH,
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
    """Give every required artifact field one scientific reason."""

    principles = {
        field: (
            f"Preserving {field} lets an independent reviewer reproduce or falsify "
            "the V619 contract conclusion."
        )
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }
    principles["field_principles"] = (
        "Scientific reasons make each recorded field an evidence control, not a checklist."
    )
    principles["preconditions_checked"] = (
        "A conclusion is valid only when every required execution input was available."
    )
    principles["source_artifact_hashes"] = (
        "Content hashes reveal source drift between this audit and later replication."
    )
    principles["rows"] = (
        "Per-task evidence prevents one pooled score from hiding a local contract failure."
    )
    principles["v619_task_contract_conforms_score"] = (
        "The score is one only when all 13 independent rows and nine gates pass."
    )
    principles["reproducibility_checksum"] = (
        "The checksum detects a change to any completed audit field."
    )
    principles["honest_verdict"] = (
        "A terminal class prevents a missing input or mismatch from appearing positive."
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact fields except the checksum that contains the hash."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _empty_contract() -> dict[str, list[Any]]:
    """Keep blocked artifacts schema-complete before comparison rows exist."""

    return {field: [] for field in CONTRACT_ROW_FIELDS} | {
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "task_contract_rows": [],
    }


def _base_artifact(root: Path, run_date: str) -> dict[str, Any]:
    """Create each required field before the terminal state is known."""

    return {
        "schema": "carnot.exp7063.v619_contract_preflight.v1",
        "experiment_id": 7063,
        "run_date": run_date,
        "status": "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
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
        "v619_task_contract_conforms_score": 0,
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
    """Freeze measured duration and checksum after all derived fields."""

    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _contract_failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while the rows retain all details."""

    if contract["markdown_task_count"] != EXPECTED_TASK_COUNT:
        check = "markdown_task_count"
        expected = EXPECTED_TASK_COUNT
        observed = contract["markdown_task_count"]
    elif len(contract["yaml_task_rows"]) != EXPECTED_TASK_COUNT:
        check = "yaml_task_count"
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

    try:
        contract = evaluate_contract(
            (root / DESIGN_PATH).read_text(encoding="utf-8"),
            roadmap,
            retired_experiment_ids(exclusion),
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
    artifact["v619_task_contract_conforms_score"] = int(contract["passed"])
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
    """Recompute conformance only from preserved evidence rows."""

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
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            return 0
        if (
            row.get("order") != order
            or row.get("expected_id") != EXPECTED_TASK_IDS[order - 1]
            or row.get("markdown_id") != EXPECTED_TASK_IDS[order - 1]
            or row.get("observed_id") != EXPECTED_TASK_IDS[order - 1]
        ):
            return 0
    exact_counts = {
        "title_parity_rows": EXPECTED_TASK_COUNT,
        "deliverable_parity_rows": EXPECTED_TASK_COUNT,
        "gate_contract_rows": EXPECTED_TASK_COUNT,
        "gate_producer_rows": EXPECTED_GATE_COUNT,
        "model_compliance_rows": EXPECTED_TASK_COUNT,
        "artifact_field_rows": EXPECTED_TASK_COUNT,
        "prompt_tail_rows": EXPECTED_TASK_COUNT,
    }
    for field in CONTRACT_ROW_FIELDS:
        rows = artifact.get(field)
        if not isinstance(rows, list) or not rows:
            return 0
        if not all(isinstance(row, Mapping) and row.get("passed") is True for row in rows):
            return 0
        if field in exact_counts and len(rows) != exact_counts[field]:
            return 0
    prior_rows = artifact.get("prior_failure_rows")
    if not isinstance(prior_rows, list) or len(prior_rows) < EXPECTED_TASK_COUNT:
        return 0
    return 1


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute fields, score, terminal state, diagnostics, and checksum."""

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
    if artifact.get("v619_task_contract_conforms_score") != score:
        errors.append("v619_task_contract_conforms_score_invalid")
    preconditions = artifact.get("preconditions_checked")
    preconditions_ok = bool(preconditions) and all(
        isinstance(row, Mapping) and row.get("available") is True for row in preconditions
    )
    if not preconditions_ok:
        expected_class, expected_verdict = "blocked", BLOCKED_VERDICT
    elif not score:
        expected_class, expected_verdict = "disqualified", DISQUALIFIED_VERDICT
    else:
        expected_class, expected_verdict = "positive", POSITIVE_VERDICT
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
    elif (expected_class == "positive") != (summary.get("passed") is True):
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


if __name__ == "__main__":  # pragma: no cover - the wrapper owns execution.
    raise SystemExit(main())
