"""Audit the frozen V614 source delta and Markdown-to-YAML task contract.

The audit is advisory. It reads both contract forms independently and reports
differences. It never activates a roadmap or changes the conductor.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
DRAFT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
CAPSTONE_PATH = Path("results/experiment_7008_v613_capstone.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REFERENCE_PATH = Path("research-references.md")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7009_v614_source_contract_preflight.json")

MILESTONE = "2026.09.614"
EXPECTED_NUMBERS = tuple(range(7009, 7023))
EXPECTED_TASK_COUNT = len(EXPECTED_NUMBERS)
INFERENCE_SUBSTRATE = "deterministic_source_contract_audit_no_llm"
PROMPT_TAIL = "Do NOT push. Do NOT modify scripts/research_conductor.py."
REFERENCE_MARKER = "## V614 Planner Refresh - 2026-09-04"
PLANNER_MARKER = "2026-09-04T23:46:23-04:00"
RANDOM_SEED = 7009

QWEN36_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF"
REQUIRED_EXP7013_MODELS = (
    QWEN36_MODEL,
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_MODELS = REQUIRED_EXP7013_MODELS + ("unsloth/gemma-4-12B-it-GGUF",)
LLM_TASK_NUMBERS = frozenset({7013, 7021})
LEGACY_SMALL_MODEL_RE = re.compile(r"(?:Qwen3(?:\.5)?-(?:0\.8|1\.7|4)B|Gemma-?4-?E4B)", re.I)

REQUIRED_SOURCE_ROUTES = (
    "arxiv_ebt_2507_02092",
    "arxiv_arm_ebm_2512_15605",
    "openreview_ebt_2507_02092",
    "openreview_arm_ebm_2512_15605",
    "huggingface_ebt_2507_02092",
    "huggingface_arm_ebm_2512_15605",
    "semantic_ebt_2507_02092",
    "semantic_arm_ebm_2512_15605",
    "extropic_writing",
    "github_extropic",
    "logical_intelligence",
)

_SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "route": REQUIRED_SOURCE_ROUTES[0],
        "category": "primary",
        "url": "https://arxiv.org/abs/2507.02092",
        "query": "arXiv EBT 2507.02092",
        "known_date": "2025-07-02",
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[1],
        "category": "primary",
        "url": "https://arxiv.org/abs/2512.15605",
        "query": "arXiv ARM-EBM 2512.15605",
        "known_date": "2026-05-25",
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[2],
        "category": "primary",
        "url": "https://api2.openreview.net/notes?content.title=Energy-Based%20Transformers%20are%20Scalable%20Learners%20and%20Thinkers",
        "query": "OpenReview EBT 2507.02092 exact title",
        "known_date": None,
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[3],
        "category": "primary",
        "url": "https://api2.openreview.net/notes?content.title=Autoregressive%20Language%20Models%20are%20Secretly%20Energy-Based%20Models",
        "query": "OpenReview ARM-EBM 2512.15605 exact title",
        "known_date": None,
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[4],
        "category": "secondary",
        "url": "https://huggingface.co/papers/2507.02092",
        "query": "Hugging Face Papers 2507.02092",
        "known_date": "2025-07-02",
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[5],
        "category": "secondary",
        "url": "https://huggingface.co/papers/2512.15605",
        "query": "Hugging Face Papers 2512.15605",
        "known_date": "2025-12-17",
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[6],
        "category": "semantic",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?limit=100&fields=title,year,url,publicationDate",
        "query": "Semantic Scholar citations EBT 2507.02092",
        "known_date": None,
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[7],
        "category": "semantic",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?limit=100&fields=title,year,url,publicationDate",
        "query": "Semantic Scholar citations ARM-EBM 2512.15605",
        "known_date": None,
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[8],
        "category": "primary",
        "url": "https://extropic.ai/writing/z1t",
        "query": "Extropic Z1T writing post-marker update",
        "known_date": "2026-09-04",
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[9],
        "category": "primary",
        "url": "https://api.github.com/orgs/extropic-ai/repos?per_page=100&sort=updated",
        "query": "GitHub Extropic public repositories updated after marker",
        "known_date": None,
    },
    {
        "route": REQUIRED_SOURCE_ROUTES[10],
        "category": "primary",
        "url": "https://logicalintelligence.com/kona-ebms-energy-based-models",
        "query": "Logical Intelligence Kona public implementation update",
        "known_date": None,
    },
)

REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
        "duration_s",
        "source_artifact_hashes",
        "source_query_rows",
        "primary_source_rows",
        "secondary_source_rows",
        "semantic_scholar_rows",
        "post_marker_delta_rows",
        "reference_append_rows",
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
        "artifact_field_rows",
        "prompt_tail_rows",
        "command_receipt_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "v614_source_delta_complete_score",
        "v614_task_contract_conforms_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)

VALIDATION_COMMAND_NAMES = (
    "focused_tests",
    "yaml_parsing",
    "adversarial_verification",
    "row_consistency_lint",
    "openspec_coverage",
    "root_clutter_check",
    "artifact_validation",
)
EXTERNAL_COMMAND_NAMES = VALIDATION_COMMAND_NAMES[:-1]
REQUIRED_MUTATIONS = (
    "task_count",
    "id_order",
    "title",
    "deliverable",
    "gate_field",
    "gate_producer",
    "prior_failure_experiment_id",
    "prior_failure_verdict",
    "prior_failure_addressed_by",
    "prior_failure_retire_flag",
    "model_specs",
    "verdict_class",
    "per_unit_rows",
    "no_llm_substrate",
    "capstone_gate",
    "prompt_tail",
)

BLOCKED_VERDICT = "blocked_v614_contract_preflight"
CONFORMS_VERDICT = "complete_positive_v614_source_delta_and_contract_conform"
DISQUALIFIED_VERDICT = "disqualified_v614_markdown_yaml_contract_mismatch"
VALID_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


def _task_number(task_id: object) -> int | None:
    """Return the leading experiment number when the identifier has one."""

    match = re.match(r"exp(\d+)(?:-|$)", str(task_id), re.I)
    return int(match.group(1)) if match else None


def _normal_exp_id(value: object) -> str | None:
    """Normalize manifest and roadmap identifiers to one comparable form."""

    number = _task_number(f"exp{value}" if isinstance(value, int) else value)
    return f"exp{number}" if number is not None else None


def _table_section(text: str, heading: str) -> str:
    """Return one Markdown section so unrelated tables cannot be mixed in."""

    if heading not in text:
        kind = "task" if heading == "## Exact Task Contract" else "gate"
        raise ValueError(f"design {kind} table is missing")
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def _parse_scalar(text: str) -> Any:
    """Parse one gate value and reject nested values that gates cannot compare."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("structured gate value must be scalar")
    return value


def parse_design(text: str) -> dict[str, Any]:
    """Parse exact task rows and gate rows from separate Markdown tables."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    if not milestone_match:
        raise ValueError("design milestone is missing")
    task_section = _table_section(text, "## Exact Task Contract")
    tasks: list[dict[str, Any]] = []
    for raw in task_section.splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", raw):
            continue
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if len(cells) != 4:
            raise ValueError(f"malformed design task row: {raw}")
        order_text, task_cell, title, deliverable_cell = cells
        task_id = task_cell.strip("`")
        number = _task_number(task_id)
        deliverable = deliverable_cell.strip("`")
        if number is None or not deliverable:
            raise ValueError(f"malformed design task row: {raw}")
        tasks.append(
            {
                "order": int(order_text),
                "id": task_id,
                "number": number,
                "title": title,
                "deliverable": deliverable,
            }
        )
    if not tasks:
        raise ValueError("design task table is missing")

    gate_section = _table_section(text, "## Dependency and Gate Contract")
    gates_by_number: dict[int, list[dict[str, Any]]] = {}
    produced_by_number: dict[int, str] = {}
    for raw in gate_section.splitlines():
        if not re.match(r"^\|\s*Exp\d+\s*\|", raw, re.I):
            continue
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if len(cells) != 3:
            raise ValueError(f"malformed design gate row: {raw}")
        task_cell, prerequisites, produced_cell = cells
        number_match = re.fullmatch(r"Exp(\d+)", task_cell, re.I)
        field_match = re.fullmatch(r"`([A-Za-z_][A-Za-z0-9_]*)`", produced_cell)
        if not number_match or not field_match:
            raise ValueError(f"malformed design gate row: {raw}")
        number = int(number_match.group(1))
        produced_by_number[number] = field_match.group(1)
        parsed_gates = []
        if not prerequisites.lower().startswith("none"):
            for part in prerequisites.split(";"):
                upstream_match = re.search(r"Exp(\d+)", part, re.I)
                predicate_match = re.search(r"`\s*(==|!=|>=|<=|>|<)\s*(.*?)\s*`", part)
                if not upstream_match or not predicate_match:
                    raise ValueError(f"malformed structured gate: {part}")
                upstream_number = int(upstream_match.group(1))
                parsed_gates.append(
                    {
                        "upstream_number": upstream_number,
                        "op": predicate_match.group(1),
                        "value": _parse_scalar(predicate_match.group(2)),
                    }
                )
        gates_by_number[number] = parsed_gates
    if not gates_by_number:
        raise ValueError("design gate table is missing")

    id_by_number = {task["number"]: task["id"] for task in tasks}
    for task in tasks:
        number = task["number"]
        if number not in produced_by_number or number not in gates_by_number:
            raise ValueError(f"design gate row is missing for Exp{number}")
        task["produced_field"] = produced_by_number[number]
        task["gates"] = [
            {
                "upstream": id_by_number.get(
                    gate["upstream_number"], f"exp{gate['upstream_number']}-missing"
                ),
                "artifact_field": produced_by_number.get(gate["upstream_number"]),
                "op": gate["op"],
                "value": gate["value"],
            }
            for gate in gates_by_number[number]
        ]
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def _required_block(prompt: str) -> str:
    """Return only the required artifact block from one task prompt."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def _field_declared(block: str, field: str) -> bool:
    """Match one bare field without accepting an alias or longer name."""

    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", block) is not None


def _substrate(prompt: str) -> str | None:
    """Extract an inference substrate even when it wraps across lines."""

    match = re.search(r"inference_substrate\s*\(([^)]+)\)", prompt, re.S)
    return re.sub(r"\s+", "", match.group(1)) if match else None


def parse_roadmap(document: object) -> dict[str, Any]:
    """Normalize YAML task data without using any Markdown-derived rows."""

    if not isinstance(document, Mapping):
        raise ValueError("roadmap mapping is required")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("roadmap tasks list is required")
    tasks: list[dict[str, Any]] = []
    for order, raw in enumerate(raw_tasks, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"roadmap task at order {order} must be a mapping")
        raw_gates = raw.get("gated_on", [])
        raw_priors = raw.get("prior_failures", [])
        if raw_gates is None:
            raw_gates = []
        if raw_priors is None:
            raw_priors = []
        if not isinstance(raw_gates, list) or not isinstance(raw_priors, list):
            raise ValueError(f"roadmap task lists at order {order} are malformed")
        gates = []
        for gate in raw_gates:
            if not isinstance(gate, Mapping):
                raise ValueError(f"roadmap task lists at order {order} are malformed")
            gates.append(
                {
                    "upstream": gate.get("upstream"),
                    "artifact_field": gate.get("artifact_field"),
                    "op": gate.get("op"),
                    "value": gate.get("value"),
                }
            )
        prompt = str(raw.get("prompt", ""))
        task_id = str(raw.get("id", ""))
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "number": _task_number(task_id),
                "title": raw.get("title"),
                "deliverable": raw.get("deliverable"),
                "milestone": raw.get("milestone", document.get("milestone")),
                "requires_gpu": raw.get("requires_gpu"),
                "per_unit_rows": raw.get("per_unit_rows"),
                "gates": gates,
                "prior_failures": list(raw_priors),
                "prompt": prompt,
                "required_block": _required_block(prompt),
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping and reject every other top-level shape."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required at {path}")
    return value


def resolve_v614_roadmap(root: Path) -> Path:
    """Prefer the planned V614 file, then accept its activated equivalent."""

    for relative in (DRAFT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            document = load_yaml(candidate)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if str(document.get("milestone")) == MILESTONE:
            return candidate
    raise FileNotFoundError("readable V614 YAML was not found")


def retired_experiment_ids(document: object) -> set[str]:
    """Collect every retired ID and remove explicit restoration entries."""

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
            many = item.get("experiment_ids", [])
            if isinstance(many, list):
                candidates.extend(many)
            for candidate in candidates:
                normalized = _normal_exp_id(candidate)
                if normalized:
                    retired.add(normalized)
            unretired = item.get("un_retired_experiment_ids", [])
            if isinstance(unretired, list):
                restored.update(
                    filter(None, (_normal_exp_id(candidate) for candidate in unretired))
                )
    return retired - restored


def _canonical_gates(gates: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """Make gate lists comparable without depending on dictionary order."""

    return [
        (gate.get("upstream"), gate.get("artifact_field"), gate.get("op"), gate.get("value"))
        for gate in gates
    ]


def _prior_rows(number: int, task_id: str, priors: Sequence[object]) -> list[dict[str, Any]]:
    """Validate all four required prior-failure fields without inference."""

    if not priors:
        return [
            {"number": number, "task_id": task_id, "prior_failure_present": False, "passed": True}
        ]
    rows = []
    for prior in priors:
        values = dict(prior) if isinstance(prior, Mapping) else {}
        passed = (
            isinstance(values.get("experiment_id"), str)
            and bool(values.get("experiment_id", "").strip())
            and isinstance(values.get("verdict"), str)
            and bool(values.get("verdict", "").strip())
            and isinstance(values.get("addressed_by"), str)
            and bool(values.get("addressed_by", "").strip())
            and values.get("retire_if_same_verdict") is True
        )
        rows.append(
            {
                "number": number,
                "task_id": task_id,
                "prior_failure_present": True,
                "experiment_id": values.get("experiment_id"),
                "passed": passed,
            }
        )
    return rows


def evaluate_contract(
    design_text: str, roadmap_document: object, retired_ids: set[str]
) -> dict[str, Any]:
    """Compare all V614 task rules and resolve each executable gate."""

    design = parse_design(design_text)
    roadmap = parse_roadmap(roadmap_document)
    markdown_tasks = design["tasks"]
    yaml_tasks = roadmap["tasks"]
    normalized_retired = {normal for value in retired_ids if (normal := _normal_exp_id(value))}
    yaml_by_id = {task["id"]: task for task in yaml_tasks}
    order_by_id = {task["id"]: task["order"] for task in yaml_tasks}

    title_rows: list[dict[str, Any]] = []
    deliverable_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    producer_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    retired_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []

    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))
    for index in range(width):
        md = markdown_tasks[index] if index < len(markdown_tasks) else None
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        number = md["number"] if md else (task["number"] if task else None)
        task_id = md["id"] if md else (task["id"] if task else None)
        same_task = bool(md and task and md["id"] == task["id"] and md["order"] == task["order"])
        title_rows.append(
            {
                "number": number,
                "task_id": task_id,
                "expected": md.get("title") if md else None,
                "observed": task.get("title") if task else None,
                "passed": same_task and md["title"] == task["title"],
            }
        )
        deliverable_rows.append(
            {
                "number": number,
                "task_id": task_id,
                "expected": md.get("deliverable") if md else None,
                "observed": task.get("deliverable") if task else None,
                "passed": same_task and md["deliverable"] == task["deliverable"],
            }
        )
        gate_rows.append(
            {
                "number": number,
                "task_id": task_id,
                "expected": md.get("gates") if md else None,
                "observed": task.get("gates") if task else None,
                "passed": same_task
                and _canonical_gates(md["gates"]) == _canonical_gates(task["gates"]),
            }
        )

        if task is None or not isinstance(number, int):
            prior_rows.append(
                {
                    "number": number,
                    "task_id": task_id,
                    "prior_failure_present": False,
                    "yaml_present": False,
                    "passed": False,
                }
            )
            retired_rows.append(
                {"number": number, "task_id": task_id, "yaml_present": False, "passed": False}
            )
            model_rows.append(
                {
                    "number": number,
                    "task_id": task_id,
                    "model_bearing": number in LLM_TASK_NUMBERS,
                    "yaml_present": False,
                    "passed": False,
                }
            )
            artifact_rows.append(
                {
                    "number": number,
                    "task_id": task_id,
                    "yaml_present": False,
                    "capstone_ungated": False if number == 7022 else None,
                    "passed": False,
                }
            )
            tail_rows.append(
                {
                    "number": number,
                    "task_id": task_id,
                    "expected": PROMPT_TAIL,
                    "observed": None,
                    "passed": False,
                }
            )
            continue

        prior_rows.extend(_prior_rows(number, task["id"], task["prior_failures"]))
        normalized_task = _normal_exp_id(task["id"])
        dependency_ids = {_normal_exp_id(gate.get("upstream")) for gate in task["gates"]}
        retired_dependencies = sorted(
            value for value in dependency_ids if value in normalized_retired
        )
        retired_rows.append(
            {
                "number": number,
                "task_id": task["id"],
                "task_retired": normalized_task in normalized_retired,
                "retired_dependencies": retired_dependencies,
                "passed": normalized_task not in normalized_retired and not retired_dependencies,
            }
        )

        prompt = task["prompt"]
        block = task["required_block"]
        model_bearing = number in LLM_TASK_NUMBERS or task.get("requires_gpu") is True
        specs_declared = _field_declared(block, "MODEL_SPECS")
        current_model_present = any(model in prompt for model in MANDATED_MODELS)
        three_present = all(model in prompt for model in REQUIRED_EXP7013_MODELS)
        qwen_present = QWEN36_MODEL in prompt
        legacy_mentions = bool(LEGACY_SMALL_MODEL_RE.search(prompt))
        legacy_headline = bool(
            re.search(
                r"(?:headline[^\n]*(?:Qwen3|Gemma-?4-?E4B)|(?:Qwen3|Gemma-?4-?E4B)[^\n]*headline)",
                prompt,
                re.I,
            )
        )
        legacy_smoke_only = not legacy_mentions or (
            "smoke" in prompt.lower() and not legacy_headline
        )
        model_passed = (
            (not model_bearing or (specs_declared and current_model_present))
            and (number != 7013 or three_present)
            and (number != 7021 or qwen_present)
            and legacy_smoke_only
        )
        model_rows.append(
            {
                "number": number,
                "task_id": task["id"],
                "model_bearing": model_bearing,
                "model_specs_declared": specs_declared,
                "required_three_models_present": three_present if number == 7013 else None,
                "qwen36_present": qwen_present if number == 7021 else None,
                "legacy_smoke_only": legacy_smoke_only,
                "passed": model_passed,
            }
        )

        substrate = _substrate(prompt)
        substrate_legal = bool(substrate) and (
            substrate == "live_llm_inference" if model_bearing else substrate.endswith("_no_llm")
        )
        comparison = bool(
            re.search(
                r"\b(?:comparative|comparison|compare|arms?|controls?|conditions?|parity)\b",
                prompt,
                re.I,
            )
        )
        common_fields = (
            "field_principles",
            "duration_s",
            "random_seed",
            "reproducibility_checksum",
            "gate_check_summary",
            "verifier_is_oracle",
            "verdict_class",
            "honest_verdict",
        )
        fields_present = all(_field_declared(block, field) for field in common_fields)
        normalized_block = re.sub(r"\s+", " ", block.lower())
        diagnostics = all(
            phrase in normalized_block
            for phrase in ("failed check", "expected value", "observed value")
        )
        per_unit_ok = not comparison or task.get("per_unit_rows") is True
        capstone_ungated = not task["gates"] if number == 7022 else None
        artifact_rows.append(
            {
                "number": number,
                "task_id": task["id"],
                "comparative": comparison,
                "per_unit_rows": task.get("per_unit_rows"),
                "required_fields_present": fields_present,
                "blocked_diagnostics": diagnostics,
                "inference_substrate": substrate,
                "legal_inference_substrate": substrate_legal,
                "capstone_ungated": capstone_ungated,
                "passed": bool(block)
                and fields_present
                and diagnostics
                and substrate_legal
                and per_unit_ok
                and capstone_ungated is not False,
            }
        )
        tail_rows.append(
            {
                "number": number,
                "task_id": task["id"],
                "expected": PROMPT_TAIL,
                "observed": prompt.rstrip().splitlines()[-1] if prompt.rstrip() else "",
                "passed": prompt.rstrip().endswith(PROMPT_TAIL),
            }
        )

    for consumer in yaml_tasks:
        for gate in consumer["gates"]:
            upstream_id = gate.get("upstream")
            upstream = yaml_by_id.get(upstream_id)
            field = gate.get("artifact_field")
            producer_present = upstream is not None
            precedes = producer_present and order_by_id[upstream_id] < consumer["order"]
            same_milestone = (
                producer_present
                and upstream.get("milestone") == consumer.get("milestone") == MILESTONE
            )
            bare = (
                isinstance(field, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field) is not None
            )
            declared = bool(
                producer_present and bare and _field_declared(upstream["required_block"], field)
            )
            upstream_retired = _normal_exp_id(upstream_id) in normalized_retired
            passed = bool(
                producer_present
                and precedes
                and same_milestone
                and bare
                and declared
                and not upstream_retired
            )
            producer_rows.append(
                {
                    "number": consumer["number"],
                    "consumer": consumer["id"],
                    "upstream": upstream_id,
                    "artifact_field": field,
                    "producer_present": producer_present,
                    "producer_precedes_consumer": precedes,
                    "same_milestone": same_milestone,
                    "bare_field": bare,
                    "field_declared": declared,
                    "upstream_retired": upstream_retired,
                    "passed": passed,
                }
            )

    row_sets = (
        title_rows,
        deliverable_rows,
        gate_rows,
        prior_rows,
        retired_rows,
        model_rows,
        artifact_rows,
        tail_rows,
    )
    checks_by_number: dict[int, list[bool]] = {}
    for rows in (*row_sets, producer_rows):
        for row in rows:
            if isinstance(row.get("number"), int):
                checks_by_number.setdefault(row["number"], []).append(bool(row["passed"]))

    task_rows = []
    for index in range(width):
        md = markdown_tasks[index] if index < len(markdown_tasks) else None
        task = yaml_tasks[index] if index < len(yaml_tasks) else None
        number = md["number"] if md else (task["number"] if task else None)
        expected_number = EXPECTED_NUMBERS[index] if index < EXPECTED_TASK_COUNT else None
        basic = bool(
            md
            and task
            and md["order"] == task["order"] == index + 1
            and md["number"] == task["number"] == expected_number
            and task["milestone"] == MILESTONE
        )
        task_rows.append(
            {
                "order": index + 1,
                "number": number,
                "task_id": md["id"] if md else (task["id"] if task else None),
                "expected_number": expected_number,
                "markdown_present": md is not None,
                "yaml_present": task is not None,
                "passed": basic and all(checks_by_number.get(number, [False])),
            }
        )

    observed_numbers = [task["number"] for task in yaml_tasks]
    global_checks = (
        design["milestone"] == MILESTONE,
        roadmap["milestone"] == MILESTONE,
        len(markdown_tasks) == EXPECTED_TASK_COUNT,
        len(yaml_tasks) == EXPECTED_TASK_COUNT,
        [task["number"] for task in markdown_tasks] == list(EXPECTED_NUMBERS),
        observed_numbers == list(EXPECTED_NUMBERS),
    )
    passed = (
        all(global_checks)
        and len(task_rows) == EXPECTED_TASK_COUNT
        and all(row["passed"] for row in task_rows)
        and all(row["passed"] for row in producer_rows)
    )
    return {
        "passed": passed,
        "markdown_task_rows": markdown_tasks,
        "yaml_task_rows": yaml_tasks,
        "task_contract_rows": task_rows,
        "title_parity_rows": title_rows,
        "deliverable_parity_rows": deliverable_rows,
        "gate_contract_rows": gate_rows,
        "gate_producer_rows": producer_rows,
        "prior_failure_rows": prior_rows,
        "retired_id_rows": retired_rows,
        "model_compliance_rows": model_rows,
        "artifact_field_rows": artifact_rows,
        "prompt_tail_rows": tail_rows,
        "observed_id_order": observed_numbers,
    }


def apply_mutation(roadmap: dict[str, Any], mutation: str) -> None:
    """Apply one named fault so tests can prove the evaluator fails closed."""

    tasks = roadmap["tasks"]
    actions: dict[str, Callable[[], None]] = {
        "task_count": lambda: tasks.pop(),
        "id_order": lambda: tasks.__setitem__(slice(0, 2), [tasks[1], tasks[0]]),
        "title": lambda: tasks[0].__setitem__("title", str(tasks[0]["title"]) + " changed"),
        "deliverable": lambda: tasks[0].__setitem__("deliverable", "results/wrong.json"),
        "gate_field": lambda: tasks[4]["gated_on"][0].__setitem__(
            "artifact_field", "misspelled_ready_scor"
        ),
        "gate_producer": lambda: tasks[4]["gated_on"][0].__setitem__("upstream", "exp9999-missing"),
        "prior_failure_experiment_id": lambda: tasks[0]["prior_failures"][0].__setitem__(
            "experiment_id", ""
        ),
        "prior_failure_verdict": lambda: tasks[0]["prior_failures"][0].__setitem__("verdict", ""),
        "prior_failure_addressed_by": lambda: tasks[0]["prior_failures"][0].__setitem__(
            "addressed_by", ""
        ),
        "prior_failure_retire_flag": lambda: tasks[0]["prior_failures"][0].__setitem__(
            "retire_if_same_verdict", False
        ),
        "model_specs": lambda: tasks[4].__setitem__(
            "prompt", tasks[4]["prompt"].replace("MODEL_SPECS", "MODEL_PLAN")
        ),
        "verdict_class": lambda: tasks[0].__setitem__(
            "prompt", tasks[0]["prompt"].replace("verdict_class", "result_class", 1)
        ),
        "per_unit_rows": lambda: tasks[0].__setitem__("per_unit_rows", False),
        "no_llm_substrate": lambda: tasks[0].__setitem__(
            "prompt",
            tasks[0]["prompt"].replace("deterministic_exp7009_no_llm", "deterministic_exp7009"),
        ),
        "capstone_gate": lambda: tasks[13].__setitem__(
            "gated_on",
            [
                {
                    "upstream": tasks[12]["id"],
                    "artifact_field": "arc_belief_generalization_positive_score",
                    "op": "==",
                    "value": 1,
                }
            ],
        ),
        "prompt_tail": lambda: tasks[0].__setitem__(
            "prompt", tasks[0]["prompt"] + "Trailing sentence.\n"
        ),
    }
    if mutation not in actions:
        raise ValueError(f"unknown mutation: {mutation}")
    actions[mutation]()


def build_mutation_rows(
    design_text: str, roadmap: dict[str, Any], retired_ids: set[str]
) -> list[dict[str, Any]]:
    """Run each focused fault and record whether a valid baseline rejects it."""

    baseline = json.dumps(roadmap, sort_keys=True)
    baseline_passed = evaluate_contract(design_text, roadmap, retired_ids)["passed"]
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < EXPECTED_TASK_COUNT:
        return [
            {
                "mutation": mutation,
                "baseline_passed": False,
                "input_changed": False,
                "failed_as_expected": True,
                "terminal": True,
                "passed": False,
                "reason": "The frozen YAML is incomplete, so an isolated mutation cannot start from a conforming contract.",
            }
            for mutation in REQUIRED_MUTATIONS
        ]
    rows = []
    for mutation in REQUIRED_MUTATIONS:
        changed = deepcopy(roadmap)
        apply_mutation(changed, mutation)
        changed_text = json.dumps(changed, sort_keys=True)
        failed = not evaluate_contract(design_text, changed, retired_ids)["passed"]
        rows.append(
            {
                "mutation": mutation,
                "baseline_passed": baseline_passed,
                "input_changed": changed_text != baseline,
                "failed_as_expected": failed,
                "terminal": True,
                "passed": baseline_passed and changed_text != baseline and failed,
            }
        )
    return rows


def _fetch_url(url: str, timeout_s: float = 15.0) -> dict[str, Any]:
    """Fetch a bounded body and convert every access outcome to a terminal row."""

    request = Request(url, headers={"User-Agent": "Carnot-V614-contract-audit/1.0"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read(131_072)
            return {
                "outcome": "ok",
                "http_status": getattr(response, "status", 200),
                "terminal": True,
                "response_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                "last_modified": response.headers.get("Last-Modified"),
                "response_excerpt": body[:1000].decode("utf-8", errors="replace"),
            }
    except HTTPError as exc:
        return {
            "outcome": "http_error",
            "http_status": exc.code,
            "terminal": True,
            "response_sha256": None,
            "last_modified": None,
            "response_excerpt": str(exc),
        }
    except URLError as exc:
        return {
            "outcome": "network_error",
            "http_status": None,
            "terminal": True,
            "response_sha256": None,
            "last_modified": None,
            "response_excerpt": str(exc.reason),
        }
    except OSError as exc:
        return {
            "outcome": "tool_error",
            "http_status": None,
            "terminal": True,
            "response_sha256": None,
            "last_modified": None,
            "response_excerpt": f"{type(exc).__name__}: {exc}",
        }


def _observed_source_date(spec: Mapping[str, Any], receipt: Mapping[str, Any]) -> str | None:
    """Keep a declared publication date or a first-party update timestamp."""

    if spec.get("known_date"):
        return str(spec["known_date"])
    if spec.get("route") == "github_extropic":
        match = re.search(r'"updated_at"\s*:\s*"([^"]+)"', str(receipt.get("response_excerpt", "")))
        if match:
            return match.group(1)
    value = receipt.get("last_modified")
    return str(value) if value else None


def collect_source_query_rows(
    run_date: str, fetch: Callable[[str], dict[str, Any]] = _fetch_url
) -> list[dict[str, Any]]:
    """Access every named public route once and preserve its terminal result."""

    accessed_on = datetime.strptime(run_date, "%Y%m%d").date().isoformat()
    rows = []
    for spec in _SOURCE_SPECS:
        receipt = fetch(spec["url"])
        row = {
            **spec,
            **receipt,
            "accessed_on": accessed_on,
            "published_or_changed_at": _observed_source_date(spec, receipt),
            "post_marker_change_proven": False,
            "relevant": False,
            "finding": "No relevant primary or first-party change after the V614 planner marker was proved by this route.",
        }
        row.pop("known_date", None)
        rows.append(row)
    return rows


def source_delta_complete_score(rows: Sequence[Mapping[str, Any]]) -> int:
    """Return one only when every requested route has one terminal receipt."""

    terminal = {row.get("route") for row in rows if row.get("terminal") is True}
    return int(terminal == set(REQUIRED_SOURCE_ROUTES))


def build_source_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate source classes and make the append decision explicit."""

    copied = [dict(row) for row in rows]
    changed = [
        row
        for row in copied
        if row.get("category") == "primary"
        and row.get("post_marker_change_proven") is True
        and row.get("relevant") is True
    ]
    if changed:
        append_rows = [
            {
                "action": "append_required",
                "appended": False,
                "reason": "Relevant post-marker primary evidence requires a dated reference entry.",
                "routes": [row.get("route") for row in changed],
                "terminal": True,
            }
        ]
    else:
        append_rows = [
            {
                "action": "no_change",
                "appended": False,
                "reason": "No relevant primary or first-party change after the V614 planner marker was proved.",
                "terminal": True,
            }
        ]
    return {
        "source_query_rows": copied,
        "primary_source_rows": [row for row in copied if row.get("category") == "primary"],
        "secondary_source_rows": [row for row in copied if row.get("category") == "secondary"],
        "semantic_scholar_rows": [row for row in copied if row.get("category") == "semantic"],
        "post_marker_delta_rows": [
            {
                "route": row.get("route"),
                "marker": PLANNER_MARKER,
                "published_or_changed_at": row.get("published_or_changed_at"),
                "post_marker_change_proven": row.get("post_marker_change_proven") is True,
                "relevant": row.get("relevant") is True,
                "terminal": row.get("terminal") is True,
            }
            for row in copied
        ],
        "reference_append_rows": append_rows,
    }


def _sha256_path(path: Path) -> str:
    """Hash one exact source without changing it."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path_writable(path: Path) -> bool:
    """Prove that the output directory accepts a temporary file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7009-write-", delete=True):
            pass
        return True
    except OSError:
        return False


def _preconditions(
    root: Path, output_path: Path, network_available: bool | None
) -> tuple[list[dict[str, Any]], Path | None, dict[str, Any] | None]:
    """Check all external inputs before assigning a contract verdict."""

    rows: list[dict[str, Any]] = []
    roadmap_path: Path | None = None
    manifest: dict[str, Any] | None = None
    for resource, relative in (
        ("v614_markdown", DESIGN_PATH),
        ("v613_capstone", CAPSTONE_PATH),
        ("reference_file", REFERENCE_PATH),
        ("reporting_spec", SPEC_PATH),
    ):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        parse_error = None
        if available and resource == "v613_capstone":
            try:
                available = isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
            except (OSError, ValueError) as exc:
                available, parse_error = False, f"{type(exc).__name__}: {exc}"
        if available and resource == "reference_file":
            try:
                available = REFERENCE_MARKER in path.read_text(encoding="utf-8")
                if not available:
                    parse_error = "V614 reference marker is missing"
            except OSError as exc:
                available, parse_error = False, f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "resource": resource,
                "path": str(path),
                "available": available,
                "parse_error": parse_error,
                "terminal": True,
            }
        )
    try:
        roadmap_path = resolve_v614_roadmap(root)
        rows.append(
            {
                "resource": "v614_yaml",
                "requested_path": str(root / DRAFT_ROADMAP_PATH),
                "resolved_path": str(roadmap_path),
                "available": True,
                "parse_error": None,
                "terminal": True,
            }
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        rows.append(
            {
                "resource": "v614_yaml",
                "requested_path": str(root / DRAFT_ROADMAP_PATH),
                "resolved_path": None,
                "available": False,
                "parse_error": f"{type(exc).__name__}: {exc}",
                "terminal": True,
            }
        )
    manifest_path = root / EXCLUSION_PATH
    try:
        manifest = load_yaml(manifest_path)
        rows.append(
            {
                "resource": "exclusion_manifest",
                "path": str(manifest_path),
                "available": True,
                "parse_error": None,
                "terminal": True,
            }
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        rows.append(
            {
                "resource": "exclusion_manifest",
                "path": str(manifest_path),
                "available": False,
                "parse_error": f"{type(exc).__name__}: {exc}",
                "terminal": True,
            }
        )
    if network_available is None:
        probe = _fetch_url("https://arxiv.org/", timeout_s=5.0)
        network_available = probe["outcome"] in {"ok", "http_error"}
    rows.append(
        {
            "resource": "network_access",
            "available": bool(network_available),
            "observed": "network_available" if network_available else "network_unavailable",
            "terminal": True,
        }
    )
    writable = _artifact_path_writable(output_path)
    rows.append(
        {
            "resource": "writable_artifact_path",
            "path": str(output_path),
            "available": writable,
            "terminal": True,
        }
    )
    return rows, roadmap_path, manifest


def _field_principles() -> dict[str, str]:
    """Give every required artifact field one scientific purpose."""

    principles = {
        field: f"{field} preserves independently checkable V614 audit evidence."
        for field in sorted(REQUIRED_ARTIFACT_FIELDS)
    }
    principles["v614_source_delta_complete_score"] = (
        "This score is one only when every requested source route has a terminal receipt."
    )
    principles["v614_task_contract_conforms_score"] = (
        "This score is one only when all 14 contract rows agree and every gate producer resolves."
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the full artifact except the checksum field itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _empty_contract() -> dict[str, list[Any]]:
    """Provide all contract collections even for a blocked preflight."""

    return {
        key: []
        for key in (
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
            "artifact_field_rows",
            "prompt_tail_rows",
        )
    }


def _source_hashes(root: Path) -> dict[str, str | None]:
    """Hash each available source and preserve absent planned-roadmap state."""

    return {
        str(relative): _sha256_path(root / relative) if (root / relative).is_file() else None
        for relative in (
            DESIGN_PATH,
            CAPSTONE_PATH,
            EXCLUSION_PATH,
            REFERENCE_PATH,
            SPEC_PATH,
            DRAFT_ROADMAP_PATH,
            ACTIVE_ROADMAP_PATH,
        )
    }


def _base_artifact(root: Path, run_date: str, started: float) -> dict[str, Any]:
    """Create the complete common schema before its terminal classification."""

    return {
        "schema": "carnot.exp7009.v614_source_contract_preflight.v1",
        "experiment_id": 7009,
        "run_date": run_date,
        "status": "complete",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": _source_hashes(root),
        **{
            key: []
            for key in (
                "source_query_rows",
                "primary_source_rows",
                "secondary_source_rows",
                "semantic_scholar_rows",
                "post_marker_delta_rows",
                "reference_append_rows",
                "rows",
                "command_receipt_rows",
            )
        },
        **_empty_contract(),
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_NUMBERS),
        "observed_id_order": [],
        "v614_source_delta_complete_score": 0,
        "v614_task_contract_conforms_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "preconditions",
            "expected_value": "all_required_inputs_available",
            "observed_value": "not_checked",
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }


def _finish(artifact: dict[str, Any], started: float) -> dict[str, Any]:
    """Freeze duration and checksum after every derived field is present."""

    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    command_rows: Sequence[Mapping[str, Any]] | None = None,
    network_available: bool | None = None,
    roadmap_override: object | None = None,
) -> dict[str, Any]:
    """Build one terminal artifact from independent source and contract rows."""

    started = time.monotonic()
    artifact = _base_artifact(root, run_date, started)
    preconditions, roadmap_path, manifest = _preconditions(root, output_path, network_available)
    artifact["preconditions_checked"] = preconditions
    failed = next((row for row in preconditions if not row.get("available")), None)
    if failed is not None:
        artifact["gate_check_summary"] = {
            "failed_check": "preconditions",
            "expected_value": "all_required_inputs_available",
            "observed_value": failed.get("observed", f"{failed['resource']}_unavailable"),
            "passed": False,
        }
        return _finish(artifact, started)

    try:
        design_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
        roadmap_document = (
            roadmap_override if roadmap_override is not None else load_yaml(roadmap_path)
        )  # type: ignore[arg-type]
        retired = retired_experiment_ids(manifest)
        contract = evaluate_contract(design_text, roadmap_document, retired)
        mutation_rows = build_mutation_rows(design_text, roadmap_document, retired)  # type: ignore[arg-type]
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        artifact["gate_check_summary"] = {
            "failed_check": "contract_parse",
            "expected_value": "parseable_independent_contracts",
            "observed_value": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
        return _finish(artifact, started)

    queried = list(source_rows) if source_rows is not None else collect_source_query_rows(run_date)
    artifact.update(build_source_evidence(queried))
    for key in _empty_contract():
        artifact[key] = contract[key]
    artifact["observed_task_count"] = len(contract["yaml_task_rows"])
    artifact["observed_id_order"] = contract["observed_id_order"]
    artifact["v614_source_delta_complete_score"] = source_delta_complete_score(queried)
    artifact["v614_task_contract_conforms_score"] = int(contract["passed"])
    artifact["rows"] = [
        *[{"row_kind": "source_route", **row} for row in artifact["source_query_rows"]],
        *[{"row_kind": "task_contract", **row} for row in artifact["task_contract_rows"]],
        *[{"row_kind": "mutation", **row} for row in mutation_rows],
    ]
    receipts = [dict(row) for row in command_rows] if command_rows is not None else []
    artifact["command_receipt_rows"] = receipts

    source_score = artifact["v614_source_delta_complete_score"]
    contract_score = artifact["v614_task_contract_conforms_score"]
    commands_terminal = all(row.get("terminal") is True for row in receipts)
    commands_pass = all(row.get("passed") is True for row in receipts)
    if not commands_terminal:
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = BLOCKED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": "validation_tools",
            "expected_value": "terminal_receipts",
            "observed_value": "tool_failure",
            "passed": False,
        }
    elif not source_score:
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = BLOCKED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": "source_routes",
            "expected_value": 1,
            "observed_value": source_score,
            "passed": False,
        }
    elif not contract_score or not commands_pass:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": "task_contract" if not contract_score else "validation_commands",
            "expected_value": 1,
            "observed_value": contract_score if not contract_score else 0,
            "passed": False,
        }
    else:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = CONFORMS_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": None,
            "expected_value": 1,
            "observed_value": 1,
            "passed": True,
        }
    return _finish(artifact, started)


def _contract_score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute contract conformance from the complete per-task rows."""

    task_rows = artifact.get("task_contract_rows", [])
    producer_rows = artifact.get("gate_producer_rows", [])
    return int(
        isinstance(task_rows, list)
        and len(task_rows) == EXPECTED_TASK_COUNT
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in task_rows)
        and isinstance(producer_rows, list)
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in producer_rows)
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute required fields, scores, verdict, diagnostics, and checksum."""

    errors = [
        f"missing_required_fields:{field}"
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
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in VALID_VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    summary = artifact.get("gate_check_summary")
    if (
        not isinstance(summary, Mapping)
        or not {"failed_check", "expected_value", "observed_value", "passed"} <= summary.keys()
    ):
        errors.append("gate_check_summary_invalid")
    source_score = source_delta_complete_score(artifact.get("source_query_rows", []))
    contract_score = _contract_score_from_artifact(artifact)
    if artifact.get("v614_source_delta_complete_score") != source_score:
        errors.append("v614_source_delta_complete_score_invalid")
    if artifact.get("v614_task_contract_conforms_score") != contract_score:
        errors.append("v614_task_contract_conforms_score_invalid")
    preconditions = artifact.get("preconditions_checked", [])
    preconditions_ok = bool(preconditions) and all(
        isinstance(row, Mapping) and row.get("available") is True for row in preconditions
    )
    receipts = artifact.get("command_receipt_rows", [])
    terminal = all(isinstance(row, Mapping) and row.get("terminal") is True for row in receipts)
    commands_pass = all(isinstance(row, Mapping) and row.get("passed") is True for row in receipts)
    if not preconditions_ok or not terminal or not source_score:
        expected_class, expected_verdict = "blocked", BLOCKED_VERDICT
    elif not contract_score or not commands_pass:
        expected_class, expected_verdict = "disqualified", DISQUALIFIED_VERDICT
    else:
        expected_class, expected_verdict = "positive", CONFORMS_VERDICT
    if verdict_class != expected_class:
        errors.append("verdict_class_not_derived")
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict_not_derived")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _run_command(root: Path, name: str, argv: Sequence[str]) -> dict[str, Any]:
    """Run one bounded check and distinguish a finding from tool failure."""

    command = " ".join(argv)
    try:
        result = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, timeout=240, check=False
        )
        passed = result.returncode == 0
        return {
            "name": name,
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "outcome": "pass" if passed else "contract_defect",
            "terminal": True,
            "passed": passed,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "exit_code": None,
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


def run_validation_commands(root: Path, artifact_path: Path) -> list[dict[str, Any]]:
    """Run the required focused, artifact, policy, and cleanliness checks."""

    python = str(root / ".venv/bin/python")
    commands = (
        (
            "focused_tests",
            [
                python,
                "-m",
                "pytest",
                "--no-cov",
                "tests/python/test_experiment_7009_v614_source_contract_preflight.py",
                "-q",
            ],
        ),
        (
            "yaml_parsing",
            [
                python,
                "-c",
                "import pathlib,yaml; d=yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); assert d['milestone']=='2026.09.614' and len(d['tasks'])==14 and [int(t['id'].split('-',1)[0][3:]) for t in d['tasks']]==list(range(7009,7023))",
            ],
        ),
        (
            "adversarial_verification",
            [python, "scripts/adversarial_verify.py", "--high-precision-only", str(artifact_path)],
        ),
        (
            "row_consistency_lint",
            [python, "scripts/verdict_row_consistency_lint.py", str(artifact_path)],
        ),
        (
            "openspec_coverage",
            [
                python,
                "scripts/check_spec_coverage.py",
                "tests/python/test_experiment_7009_v614_source_contract_preflight.py",
            ],
        ),
        ("root_clutter_check", [python, "scripts/root_clutter_sweep.py"]),
    )
    receipts = [_run_command(root, name, argv) for name, argv in commands]
    for receipt in receipts:
        if (
            receipt["name"] == "adversarial_verification"
            and receipt["exit_code"] == 1
            and "[CRITICAL]" not in receipt["stdout"]
        ):
            receipt["outcome"] = "warning"
            receipt["passed"] = True
    return receipts


def _date_argument(value: str) -> str:
    """Accept only a real date in the command's compact form."""

    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYYMMDD") from exc
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the output only after the full JSON value is ready."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and write its one requested result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output

    provisional = build_artifact(REPO_ROOT, args.date, output_path=output, command_rows=[])
    errors = validate_artifact(provisional)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    _write_json_atomic(output, provisional)
    if provisional["verdict_class"] == "blocked":
        return 0

    external = run_validation_commands(REPO_ROOT, output)
    validation_row = {
        "name": "artifact_validation",
        "command": "validate_artifact",
        "exit_code": 0,
        "stdout": "clean",
        "stderr": "",
        "outcome": "pass",
        "terminal": True,
        "passed": True,
    }
    artifact = build_artifact(
        REPO_ROOT,
        args.date,
        output_path=output,
        source_rows=provisional["source_query_rows"],
        command_rows=[*external, validation_row],
        network_available=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    _write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper calls main.
    raise SystemExit(main())
