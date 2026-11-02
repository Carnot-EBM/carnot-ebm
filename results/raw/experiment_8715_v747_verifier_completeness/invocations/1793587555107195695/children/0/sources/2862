"""Build the V644 source and exact fourteen-task contract receipt.

The receipt compares two independently parsed authorities. It also exercises
the real conductor gate reader with fail-closed producer fixtures. The receipt
is advisory and cannot authorize a science task.

Spec refs: REQ-REPORT-7329 and SCENARIO-REPORT-7329-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_bytes,
    _atomic_write_json,
)
from carnot.experiment_7192_v634_source_contract import _fetch_url, reproducibility_checksum
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    reduce_required_checks,
    run_commands,
    run_scoped_validation,
)
from scripts.conductor_gates import evaluate_gates


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]
ValidationRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.644"
RUN_DATE = "20260915"
RETRIEVAL_DATE = "2026-09-15"
FIRST_TASK_ID = "exp7329-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7330-executor-isolation",
    "exp7331-learning-adapter",
    "exp7332-plan-canary",
    "exp7333-plan-capture",
    "exp7334-prospective-learning",
    "exp7335-learning-audit",
    "exp7336-arc-resume",
    "exp7337-arc-transfer",
    "exp7338-arc-causal-audit",
    "exp7339-native-binding",
    "exp7340-native-cost",
    "exp7341-board-continuity",
    "exp7342-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
CLOSED_TERMINAL_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
RANDOM_SEED = {
    "development": 7_329_202_609_15,
    "evaluation": 7_329_202_609_16,
    "resampling": 7_329_202_609_17,
}

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
STUDY_PATH = Path("research-studying.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")

MODULE_PATH = Path("python/carnot/experiment_7329_v644_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7329_v644_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7329_v644_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7329_v644_contract.json")
RAW_DIR = Path("results/raw/experiment_7329")
RAW_CANDIDATE_NAME = "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7329_v644_contract.json")

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")

HISTORY_PATHS = tuple(
    Path(f"results/experiment_{number}_v643_{suffix}.json")
    for number, suffix in (
        (7316, "contract"),
        (7317, "batch_harness"),
        (7318, "arc_authority"),
        (7319, "arc_session"),
        (7320, "batch_canary"),
        (7321, "batch_measurement"),
        (7322, "batch_audit"),
        (7323, "addition_prototype"),
        (7324, "addition_learning"),
        (7325, "addition_audit"),
        (7326, "constraint_kernel"),
        (7327, "board_continuity"),
        (7328, "capstone"),
    )
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    STUDY_PATH,
    EXCLUSION_PATH,
    E2E_PLAN_PATH,
    COMPLETE_PATH,
    SPEC_PATH,
    DESIGN_PATH,
    ROADMAP_SCHEMA_PATH,
    PRIOR_LINT_PATH,
    GATE_AUDIT_PATH,
    EXCLUSION_LINT_PATH,
    PROMPT_PATH_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    *HISTORY_PATHS,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

ROADMAP_CHECK_NAMES = (
    "roadmap_schema",
    "prior_failure_scope",
    "gate_audit",
    "exclusion_manifest",
    "prompt_paths",
)
TERMINAL_CHECK_NAMES = ("independent_reducer", "adversarial", "row_consistency")
ALL_VALIDATION_NAMES = (*ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the artifact while preserving ordinary top-level experiment_id and milestone.",
    "status": "Publish a terminal result only after current work and affected validation.",
    "run_date": "Use 20260915 and retain actual UTC timestamps.",
    "preconditions_checked": "Name input identity, availability and each failed check before work.",
    "MODEL_SPECS": "List actual intended/current executable identities; any LLM task includes unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True when any real load or generation is attempted, even if no usable answer arrives.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight loads and generations.",
    "inference_substrate": "Declare actual computation; historical model evidence is not a current invocation.",
    "inference_substrate_class": "Declare the real duration class; small fixed-token runs are model_bounded_generation.",
    "execution_venue": "Use host; old board receipts never imply current board execution.",
    "duration_s": "Measure real monotonic elapsed time; never pad a duration floor.",
    "phase_spans": "Disjoint stage spans, completed units, checkpoint positions and pending operations explain cost.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, public inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers rather than similarly named older results.",
    "rows": "Every comparative unit and arm carries metrics, costs, abstentions, failures and censoring.",
    "sample_size_budget": "Record planned, attempted, completed and censored units with a fixed stopping rule.",
    "acceptance_gate_results": "Each gate records expected, observed and passed; separate completion from value.",
    "gate_check_summary": "Every blocked_* must identify upstream, failed check, artifact field, expected and observed value.",
    "verifier_is_oracle": "True whenever the execution authority defines correctness; independent code alone does not remove circularity.",
    "honest_verdict": "Completed findings start complete_ or complete:; external absence starts blocked_ and names the check.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; external absence is blocked.",
    "validation_receipts": "Record exact command, affected scope, exit code, duration and log hash including failures.",
    "repository_health": "Keep unrelated repository failures as dated observations, separate from affected required checks.",
    "field_principles": "Explain each field without wrapping executable scores or ordinary dictionaries.",
    "contract_complete_score": "One means the exact fourteen-task contract and its adverse gate controls pass; it authorizes no science.",
    "source_dispositions": "Each source has URL, date, access outcome, adopted method, task mapping and limits.",
    "contract_rows": "Keep one independently compared row per exact task plus mutation and actual-gate-reader rows.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "access_id": "cca",
        "url": "https://arxiv.org/abs/2604.13283",
        "title": "Conservative Constraint Acquisition",
        "publication_or_version_date": "2026-04",
    },
    {
        "access_id": "memoir",
        "url": "https://arxiv.org/abs/2607.20792",
        "title": "Memoir: Should a Model Write to Its Memory While It Thinks?",
        "publication_or_version_date": "2026-07-22",
    },
    {
        "access_id": "solver_hard",
        "url": "https://arxiv.org/abs/2607.17047",
        "title": "Solver-Hard Is Not Model-Hard",
        "publication_or_version_date": "2026-07-19",
    },
    {
        "access_id": "amtfv",
        "url": "https://arxiv.org/abs/2607.29549",
        "title": "AMTFV: Agentic Mathematical Tool-Flow Verification",
        "publication_or_version_date": "2026-07-31",
    },
)

SOURCE_CHOICES: tuple[JsonDict, ...] = (
    {
        "method_family": "cca",
        "adopted_method": "Use restricted constraint acquisition through an isolated Boolean executor and charge every query.",
        "task_mapping": ["exp7330-executor-isolation", "exp7334-prospective-learning"],
        "limits": "The synthetic schedule language and shared executor authority do not establish live general competence.",
    },
    {
        "method_family": "memoir",
        "adopted_method": "Keep prediction read-only and commit verified memory updates between requests.",
        "task_mapping": [
            "exp7331-learning-adapter",
            "exp7334-prospective-learning",
            "exp7335-learning-audit",
        ],
        "limits": "One procedural-recall study does not prove a universal memory-write rule.",
    },
    {
        "method_family": "solver_hard",
        "adopted_method": "Freeze identifier relabeling and request-order controls before proposal evaluation.",
        "task_mapping": ["exp7330-executor-isolation", "exp7333-plan-capture"],
        "limits": "Solver runtime is not a proxy for model difficulty, and saturated cells retain no headroom claim.",
    },
    {
        "method_family": "amtfv",
        "adopted_method": "Trace tool request, exact execution, resumed reasoning, and the later policy action separately.",
        "task_mapping": [
            "exp7336-arc-resume",
            "exp7337-arc-transfer",
            "exp7338-arc-causal-audit",
        ],
        "limits": "Mathematical tool-flow results do not establish ARC transfer or later-action utility.",
    },
)


def progress(phase: int, event: str, detail: str) -> None:
    """Flush a short phase boundary so long work never looks stalled."""

    print(f"[exp7329] phase={phase} event={event} {detail}", flush=True)


def sha256_bytes(content: bytes) -> str:
    """Hash exact evidence bytes without changing their representation."""

    return "sha256:" + hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    """Hash one file exactly as it was read by this task."""

    return sha256_bytes(path.read_bytes())


def _load_yaml(content: bytes, path: Path) -> JsonDict:
    """Load one nonempty YAML mapping and keep the source in errors."""

    value = yaml.safe_load(content.decode("utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"YAML mapping required at {path}")
    return value


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Prefer a matching staged V644 roadmap, then the matching active file."""

    selected: tuple[Path, JsonDict, bytes] | None = None
    candidates: list[JsonDict] = []
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            content = path.read_bytes()
            document = _load_yaml(content, path)
            observed: object = document.get("milestone")
            matches = observed == MILESTONE
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            content, document, observed, matches = b"", None, f"{type(exc).__name__}: {exc}", False
        candidates.append(
            {
                "check": f"yaml_candidate:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "artifact_field": "milestone",
                "expected_value": MILESTONE,
                "observed_value": observed,
                "available": matches,
                "blocking": False,
            }
        )
        if selected is None and matches and document is not None:
            selected = (relative, document, content)
    if selected is None:
        return None, None, None, candidates
    return *selected, candidates


def _required_block(prompt: str) -> str:
    """Isolate declared artifact fields from unrelated prompt prose."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def _field_declared(prompt: str, field: object) -> bool:
    """Require one exact field declaration inside the producer contract."""

    if not isinstance(field, str):
        return False
    return bool(re.search(rf"(?m)^\s*-\s*{re.escape(field)}\s*:", _required_block(prompt)))


def _substrate_class(prompt: str) -> str | None:
    """Read the explicit computation-class assignment from task instructions."""

    values = re.findall(r"inference_substrate_class\s*=\s*([a-z_]+)", prompt)
    return values[0] if values else None


def _parse_gate_value(operator: str, text: str) -> Any:
    """Parse scalars and allow `in` only with a closed JSON class list."""

    if operator == "in":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Markdown in operator requires a JSON list") from exc
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) for item in value)
            or not set(value).issubset(CLOSED_TERMINAL_CLASSES)
        ):
            raise ValueError("Markdown in operator requires a JSON list of terminal classes")
        return value
    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("Markdown equality gate requires a scalar")
    return value


def _parse_markdown_gate(cell: str) -> list[JsonDict]:
    """Parse the V644 gate syntax without consulting YAML-derived values."""

    if cell.strip().lower() == "none":
        return []
    pattern = re.compile(
        r"^(exp\d+(?:-[A-Za-z0-9_-]+)?)\.([A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(==|in)\s*(.+)$",
        re.I,
    )
    gates: list[JsonDict] = []
    for expression in re.split(r"\s*(?:;|\s+AND\s+)\s*", cell, flags=re.I):
        match = pattern.fullmatch(expression.strip().strip("`"))
        if match is None:
            raise ValueError(f"malformed Markdown structured gate: {cell}")
        operator = match.group(3).lower()
        gates.append(
            {
                "upstream": match.group(1),
                "artifact_field": match.group(2),
                "op": operator,
                "value": _parse_gate_value(operator, match.group(4).strip().strip("`")),
            }
        )
    return gates


def parse_markdown_contract(text: str) -> JsonDict:
    """Parse the first exact-task table without using the YAML document."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\s]+)`?", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    section = re.search(r"^## Exact Task Contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.I | re.M)
    if section is None:
        raise ValueError("Markdown task contract section is missing")
    header: list[str] | None = None
    tasks: list[JsonDict] = []
    for line in section.group(1).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if header is None:
            header = [re.sub(r"\s+", " ", cell.lower()) for cell in cells]
            continue
        columns = dict(zip(header, cells, strict=False))
        try:
            order = int(columns.get("order", ""))
            phase = int(columns.get("phase", ""))
        except ValueError as exc:
            raise ValueError(f"malformed Markdown task row: {line}") from exc
        task_id = columns.get("task id", "").strip("`")
        title = columns.get("exact title", "")
        deliverable = columns.get("deliverable", "").strip("`")
        substrate = columns.get("substrate class", "").strip("`")
        if not re.fullmatch(r"exp\d+(?:-[A-Za-z0-9_-]+)?", task_id) or not all(
            (title, deliverable, substrate)
        ):
            raise ValueError(f"malformed Markdown task row: {line}")
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "title": title,
                "deliverable": deliverable,
                "phase": phase,
                "substrate": substrate,
                "milestone": milestone_match.group(1),
                "gates": _parse_markdown_gate(columns.get("structured gate", "none")),
            }
        )
    if not tasks:
        raise ValueError("Markdown task table is missing")
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def parse_yaml_contract(document: object) -> JsonDict:
    """Parse executable V644 rows without receiving the Markdown result."""

    if not isinstance(document, Mapping) or not isinstance(document.get("tasks"), list):
        raise ValueError("YAML roadmap mapping with tasks is required")
    tasks: list[JsonDict] = []
    for order, raw in enumerate(document["tasks"], 1):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("prompt"), str):
            raise ValueError(f"YAML task {order} must contain a prompt mapping")
        raw_gates = raw.get("gated_on") or []
        if not isinstance(raw_gates, list) or not all(
            isinstance(gate, Mapping) for gate in raw_gates
        ):
            raise ValueError(f"YAML task {order} has malformed gates")
        gates: list[JsonDict] = []
        for gate in raw_gates:
            operator = gate.get("op")
            value = gate.get("value")
            if operator not in {"==", "in"}:
                raise ValueError(f"YAML task {order} has malformed operator")
            if operator == "in" and (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) for item in value)
                or not set(value).issubset(CLOSED_TERMINAL_CLASSES)
            ):
                raise ValueError(f"YAML task {order} has malformed terminal class list")
            gates.append(
                {
                    "upstream": gate.get("upstream"),
                    "artifact_field": gate.get("artifact_field"),
                    "op": operator,
                    "value": value,
                }
            )
        phase = raw.get("phase")
        tasks.append(
            {
                "order": order,
                "id": raw.get("id"),
                "title": raw.get("title"),
                "deliverable": raw.get("deliverable"),
                "phase": phase if isinstance(phase, int) and not isinstance(phase, bool) else None,
                "substrate": _substrate_class(str(raw["prompt"])),
                "milestone": raw.get("milestone", document.get("milestone")),
                "gates": gates,
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def validate_gate_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Prove each consumed field is declared and the receipt gates no science."""

    tasks = roadmap.get("tasks", [])
    by_id = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    rows: list[JsonDict] = []
    receipt_is_not_upstream = True
    for consumer_index, consumer in enumerate(tasks):
        if not isinstance(consumer, Mapping):
            continue
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            receipt_is_not_upstream = receipt_is_not_upstream and upstream != FIRST_TASK_ID
            producer_entry = by_id.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            declared = _field_declared(str(producer.get("prompt", "")), gate.get("artifact_field"))
            rows.append(
                {
                    "consumer": consumer.get("id"),
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "producer_precedes_consumer": producer_index is not None
                    and producer_index < consumer_index,
                    "declared_verbatim": declared,
                    "passed": producer_index is not None
                    and producer_index < consumer_index
                    and declared
                    and upstream != FIRST_TASK_ID,
                }
            )
    return {
        "rows": rows,
        "receipt_is_not_upstream": receipt_is_not_upstream,
        "passed": bool(rows) and receipt_is_not_upstream and all(row["passed"] for row in rows),
    }


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only the six independently compared fields."""

    if task is None:
        return None
    return {
        key: task.get(key)
        for key in ("order", "id", "title", "phase", "substrate", "deliverable", "gates")
    }


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare task-owned Markdown and YAML parsers for every V644 field."""

    markdown = parse_markdown_contract(markdown_text)
    parsed_yaml = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    declarations = (
        validate_gate_declarations(yaml_document) if isinstance(yaml_document, Mapping) else {}
    )
    declaration_rows = declarations.get("rows", [])
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else None
        observed = yaml_tasks[index] if index < len(yaml_tasks) else None
        task_id = (observed or expected or {}).get("id")
        task_declarations = [row for row in declaration_rows if row.get("consumer") == task_id]
        checks = {
            "id": bool(expected and observed and expected.get("id") == observed.get("id")),
            "title": bool(expected and observed and expected.get("title") == observed.get("title")),
            "phase": bool(expected and observed and expected.get("phase") == observed.get("phase")),
            "substrate": bool(
                expected and observed and expected.get("substrate") == observed.get("substrate")
            ),
            "deliverable": bool(
                expected and observed and expected.get("deliverable") == observed.get("deliverable")
            ),
            "gates": bool(expected and observed and expected.get("gates") == observed.get("gates")),
            "gate_fields_declared": all(row.get("passed") is True for row in task_declarations),
        }
        failures = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "unit_id": task_id,
                "arm": "markdown_vs_yaml",
                "order": index + 1,
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "checks": checks,
                "metrics": {
                    "matched_field_count": sum(checks.values()),
                    "field_count": len(checks),
                },
                "costs": {"model_loads": 0, "generation_calls": 0, "network_requests": 0},
                "failures": failures,
                "abstentions": [],
                "censored": False,
                "passed": not failures,
            }
        )
    markdown_order = [task.get("id") for task in markdown_tasks]
    yaml_order = [task.get("id") for task in yaml_tasks]
    passed = bool(
        markdown.get("milestone") == parsed_yaml.get("milestone") == MILESTONE
        and markdown_order == yaml_order == list(EXPECTED_ID_ORDER)
        and len(rows) == EXPECTED_TASK_COUNT
        and declarations.get("passed") is True
        and all(row["passed"] for row in rows)
    )
    return {
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "markdown_task_rows": [_public_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task(task) for task in yaml_tasks],
        "contract_rows": rows,
        "markdown_id_order": markdown_order,
        "yaml_id_order": yaml_order,
        "gate_declaration_result": declarations,
        "passed": passed,
    }


def run_contract_mutations(
    markdown_text: str, roadmap: Mapping[str, Any], temp_root: Path
) -> list[JsonDict]:
    """Reject every named contract mutation through the task-owned parsers."""

    mutations: list[tuple[str, Callable[[JsonDict], None]]] = []

    def missing_task(document: JsonDict) -> None:
        document["tasks"] = document["tasks"][:-1]

    def reordered_id(document: JsonDict) -> None:
        document["tasks"][0], document["tasks"][1] = document["tasks"][1], document["tasks"][0]

    def stale_milestone(document: JsonDict) -> None:
        document["milestone"] = "2026.09.643"

    def changed_phase(document: JsonDict) -> None:
        document["tasks"][0]["phase"] = 4

    def changed_path(document: JsonDict) -> None:
        document["tasks"][0]["deliverable"] = "results/mutated.json"

    def changed_title(document: JsonDict) -> None:
        document["tasks"][0]["title"] += " changed"

    def changed_field(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["artifact_field"] += "_mutated"

    def malformed_operator(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["op"] = "contains"

    mutations.extend(
        [
            ("missing_task", missing_task),
            ("reordered_id", reordered_id),
            ("stale_milestone", stale_milestone),
            ("changed_phase", changed_phase),
            ("changed_path", changed_path),
            ("changed_title", changed_title),
            ("changed_field", changed_field),
            ("malformed_operator", malformed_operator),
        ]
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for name, mutate in mutations:
        changed = deepcopy(dict(roadmap))
        mutate(changed)
        path = temp_root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
        try:
            observed = evaluate_contract(markdown_text, yaml.safe_load(path.read_text()))["passed"]
            error = None
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            observed = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "mutation": name,
                "fixture_path": str(path),
                "expected": "contract_rejected",
                "observed": observed,
                "error": error,
                "rejected": observed is False,
            }
        )
    return rows


def upstream_precondition(data: object, field: str, expected: Any, op: str = "==") -> JsonDict:
    """Reject missing or ineligible evidence before reading a gate score."""

    mapping = data if isinstance(data, Mapping) else {}
    actual = mapping.get(field)
    field_missing = field not in mapping
    text = " ".join(
        str(mapping.get(name, "")) for name in ("status", "verdict_class", "honest_verdict")
    ).lower()
    quarantined = bool(mapping.get("flagged_adversarial"))
    disqualified = "disqualified" in text
    blocked = "blocked" in text
    partial = "partial" in text or not isinstance(data, Mapping)
    field_passed = actual == expected if op == "==" else actual in expected if op == "in" else False
    passed = bool(
        not field_missing
        and field_passed
        and not quarantined
        and not disqualified
        and not blocked
        and not partial
    )
    if not isinstance(data, Mapping):
        reason = "missing_upstream_rejected_before_field_consumption"
    elif field_missing:
        reason = "missing_field_rejected_before_field_consumption"
    elif quarantined:
        reason = "quarantined_upstream_rejected_before_field_consumption"
    elif disqualified:
        reason = "disqualified_upstream_rejected_before_field_consumption"
    elif blocked:
        reason = "blocked_upstream_rejected_before_field_consumption"
    elif partial:
        reason = "partial_upstream_rejected_before_field_consumption"
    elif not field_passed:
        reason = "upstream_field_did_not_match"
    else:
        reason = "authenticated_terminal_field_match"
    return {
        "field": field,
        "expected": expected,
        "actual": actual,
        "field_missing": field_missing,
        "quarantined": quarantined,
        "disqualified": disqualified,
        "blocked": blocked,
        "partial": partial,
        "field_passed": field_passed,
        "passed": passed,
        "reason": reason,
    }


def _passing_artifacts(gates: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Build one terminal passing artifact for all gates of one consumer."""

    artifacts: dict[str, JsonDict] = {}
    for gate in gates:
        upstream = str(gate["upstream"])
        data = artifacts.setdefault(
            upstream,
            {
                "status": "complete",
                "verdict_class": "circular_positive",
                "honest_verdict": "complete_fixture",
            },
        )
        expected = gate.get("value")
        data[str(gate["artifact_field"])] = expected[0] if gate.get("op") == "in" else expected
    return artifacts


def _write_gate_artifacts(directory: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    """Write isolated producer files using the conductor's filename convention."""

    directory.mkdir(parents=True, exist_ok=True)
    for upstream, data in artifacts.items():
        match = re.match(r"exp(\d+)", upstream)
        if match is None:
            raise ValueError(f"malformed upstream task id: {upstream}")
        _atomic_write_json(directory / f"experiment_{match.group(1)}_validation_input.json", data)


def run_gate_controls(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Run six real-reader cases for every declared V644 gate edge."""

    edges = [
        (str(task.get("id")), list(task.get("gated_on") or []), edge_index, gate)
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
        for edge_index, gate in enumerate(task.get("gated_on") or [])
        if isinstance(gate, Mapping)
    ]
    cases = (
        "passing",
        "zero_score",
        "missing_file",
        "missing_field",
        "missing_verdict_class",
        "disqualified_score_one",
    )
    rows: list[JsonDict] = []
    for global_index, (consumer, consumer_gates, local_index, gate) in enumerate(edges):
        for case in cases:
            progress(3, "gate_control_start", f"edge={global_index + 1}/{len(edges)} case={case}")
            directory = results_dir / f"edge_{global_index}_{case}"
            artifacts = _passing_artifacts(consumer_gates)
            upstream = str(gate["upstream"])
            field = str(gate["artifact_field"])
            target = artifacts[upstream]
            if case == "zero_score":
                target[field] = 0
            elif case == "missing_field":
                target.pop(field, None)
            elif case == "missing_verdict_class":
                target.pop("verdict_class", None)
            elif case == "disqualified_score_one":
                target["verdict_class"] = "disqualified"
                target["honest_verdict"] = "complete_disqualified_fixture"
                for candidate in consumer_gates:
                    candidate_field = str(candidate["artifact_field"])
                    if candidate_field.endswith("_score"):
                        artifacts[str(candidate["upstream"])][candidate_field] = 1
            if case == "missing_file":
                artifacts.pop(upstream, None)
            _write_gate_artifacts(directory, artifacts)
            single = evaluate_gates({"id": consumer, "gated_on": [dict(gate)]}, directory)
            combined = evaluate_gates({"id": consumer, "gated_on": consumer_gates}, directory)
            intake_rows = [
                upstream_precondition(
                    artifacts.get(str(candidate["upstream"])),
                    str(candidate["artifact_field"]),
                    candidate.get("value"),
                    str(candidate.get("op")),
                )
                for candidate in consumer_gates
            ]
            observed = single.gates_evaluated[0]
            rows.append(
                {
                    "edge_index": global_index,
                    "consumer_gate_index": local_index,
                    "consumer": consumer,
                    "upstream": upstream,
                    "case": case,
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate.get("value"),
                    "observed_value": observed.actual,
                    "conductor_passed": single.passed,
                    "conductor_reason": observed.reason,
                    "combined_conductor_passed": combined.passed,
                    "combined_conductor_summary": combined.summary,
                    "experiment_precondition_passed": all(row["passed"] for row in intake_rows),
                    "experiment_precondition_reasons": [row["reason"] for row in intake_rows],
                    "validation_input": True,
                    "research_result": False,
                }
            )
            progress(3, "gate_control_end", f"edge={global_index + 1}/{len(edges)} case={case}")
    return rows


def gate_controls_complete(rows: object, gate_count: int) -> bool:
    """Require one pass and five fail-closed cases for each gate edge."""

    if not isinstance(rows, list) or len(rows) != gate_count * 6:
        return False
    cases = [
        "passing",
        "zero_score",
        "missing_file",
        "missing_field",
        "missing_verdict_class",
        "disqualified_score_one",
    ]
    for edge_index in range(gate_count):
        edge = [row for row in rows if row.get("edge_index") == edge_index]
        if [row.get("case") for row in edge] != cases:
            return False
        expected = [True, False, False, False, False, False]
        if [row.get("combined_conductor_passed") for row in edge] != expected:
            return False
        if [row.get("experiment_precondition_passed") for row in edge] != expected:
            return False
    return True


def collect_source_dispositions(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check four primary pages in order and retain every access outcome."""

    access_rows: list[JsonDict] = []
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(4, "source_request_start", f"completed={index - 1}/4 source={spec['access_id']}")
        try:
            receipt = dict(fetcher(str(spec["url"])))
        except Exception as exc:  # noqa: BLE001 - the access failure is evidence.
            receipt = {
                "ok": False,
                "status_code": None,
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        body = str(receipt.get("body", ""))
        status_code = receipt.get("status_code")
        if receipt.get("ok") is True:
            outcome = "http_success"
        elif status_code == 429:
            outcome = "http_429"
        elif "challenge" in f"{body} {receipt.get('error', '')}".lower():
            outcome = "browser_challenge"
        else:
            outcome = "access_failed"
        row = {
            "access_id": spec["access_id"],
            "url": spec["url"],
            "retrieval_date": RETRIEVAL_DATE,
            "publication_or_version_date": spec["publication_or_version_date"],
            "timeout_s": 20,
            "status_code": status_code,
            "error": receipt.get("error"),
            "response_body": body,
            "response_sha256": sha256_bytes(body.encode("utf-8")),
            "title_verified": str(spec["title"]).lower() in body.lower()
            if receipt.get("ok") is True
            else None,
            "access_outcome": outcome,
        }
        access_rows.append(row)
        progress(4, "source_request_end", f"completed={index}/4 outcome={outcome}")
    by_id = {row["access_id"]: row for row in access_rows}
    dispositions: list[JsonDict] = []
    for choice in SOURCE_CHOICES:
        row = deepcopy(choice)
        access = by_id[row["method_family"]]
        row.update(
            url=access["url"],
            retrieval_date=access["retrieval_date"],
            publication_or_version_date=access["publication_or_version_date"],
            access_outcome=access["access_outcome"],
            access_status_code=access["status_code"],
            access_error=access["error"],
            local_evidence_claimed=False,
        )
        dispositions.append(row)
    return dispositions, access_rows


def source_dispositions_complete(rows: object) -> bool:
    """Require every source decision without requiring successful access."""

    expected = {row["method_family"] for row in SOURCE_CHOICES}
    outcomes = {"http_success", "http_429", "browser_challenge", "access_failed"}
    return bool(
        isinstance(rows, list)
        and len(rows) == len(SOURCE_CHOICES)
        and {row.get("method_family") for row in rows} == expected
        and all(
            row.get("url")
            and row.get("adopted_method")
            and row.get("task_mapping")
            and row.get("limits")
            and row.get("access_outcome") in outcomes
            for row in rows
        )
    )


def roadmap_command_specs(root: Path, authority: Path) -> list[CommandSpec]:
    """Build the shipped structural checks against the selected YAML only."""

    python = str(root / ".venv/bin/python")
    roadmap = str(root / authority)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    return [
        CommandSpec("roadmap_schema", (python, "-u", "-c", schema_code), "selected_yaml"),
        CommandSpec(
            "prior_failure_scope",
            (python, "-u", str(root / PRIOR_LINT_PATH), roadmap),
            "selected_yaml",
        ),
        CommandSpec(
            "gate_audit",
            (
                python,
                "-u",
                str(root / GATE_AUDIT_PATH),
                roadmap,
                "--complete",
                str(root / COMPLETE_PATH),
            ),
            "selected_yaml",
        ),
        CommandSpec(
            "exclusion_manifest",
            (python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap),
            "selected_yaml",
        ),
        CommandSpec(
            "prompt_paths",
            (python, "-u", str(root / PROMPT_PATH_LINT_PATH), "prompt-paths", roadmap),
            "selected_yaml",
        ),
    ]


def _failed_names(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[str]:
    """Name missing, duplicate, failed, or timed-out required commands."""

    failures: list[str] = []
    for name in names:
        rows = [row for row in receipts if row.get("name") == name]
        if (
            len(rows) != 1
            or rows[0].get("passed") is not True
            or rows[0].get("exit_code") != 0
            or rows[0].get("timed_out") is True
        ):
            failures.append(name)
    return failures


def run_affected_validation(
    root: Path,
    authority: Path,
    raw_dir: Path,
    *,
    test_paths: Sequence[str] | None = None,
    changed_modules: Sequence[str] | None = None,
    static_paths: Sequence[str] | None = None,
    historical_failures: Sequence[Mapping[str, Any]] = (),
    roadmap_runner: Callable[..., list[JsonDict]] = run_commands,
    scoped_runner: Callable[..., JsonDict] = run_scoped_validation,
) -> JsonDict:
    """Run structural audits and the shipped explicit-file validation set."""

    tests = list(test_paths or [str(TEST_PATH)])
    modules = list(changed_modules or [str(MODULE_PATH)])
    static = list(static_paths or [str(WRAPPER_PATH)])
    basetemp = Path("/tmp/exp7329-v644-scoped")
    basetemp.mkdir(parents=True, exist_ok=True)
    roadmap_receipts = roadmap_runner(
        root,
        roadmap_command_specs(root, authority),
        log_dir=raw_dir / "validation/roadmap",
    )
    scoped = scoped_runner(
        root,
        test_paths=tests,
        changed_modules=modules,
        static_paths=static,
        basetemp=basetemp,
        coverage_file=Path("/tmp/.coverage-exp7329-v644"),
        log_dir=raw_dir / "validation/scoped",
        historical_failures=historical_failures,
    )
    scoped_receipts = list(scoped.get("validation_receipts", []))
    scoped_reduction = reduce_required_checks(scoped_receipts)
    failed = _failed_names(roadmap_receipts, ROADMAP_CHECK_NAMES)
    failed.extend(scoped_reduction["failed_required_commands"])
    failed.extend(scoped_reduction["missing_required_commands"])
    failed.extend(scoped_reduction["duplicate_required_commands"])
    return {
        "validation_receipts": [*roadmap_receipts, *scoped_receipts],
        "required_checks_passed": not failed and scoped_reduction["required_checks_passed"],
        "failed_required_commands": failed,
        "repository_health": scoped.get("repository_health", {}),
    }


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Reload the candidate and run the independent and strict terminal tools."""

    python = str(root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib;"
        "from carnot.experiment_7329_v644_contract import independent_reduce;"
        f"errors=independent_reduce(json.loads(pathlib.Path({str(candidate)!r}).read_text()));"
        "print(errors,flush=True);raise SystemExit(bool(errors))"
    )
    commands = [
        CommandSpec(
            "independent_reducer", (python, "-u", "-c", reducer_code), "measured_candidate"
        ),
        CommandSpec(
            "adversarial",
            (python, "-u", str(root / ADVERSARIAL_PATH), str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "row_consistency",
            (python, "-u", str(root / ROW_LINT_PATH), "--strict", str(candidate)),
            "measured_candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def _historical_failures(root: Path) -> list[JsonDict]:
    """Keep failed V643 full-suite receipts separate from affected checks."""

    rows: list[JsonDict] = []
    for relative in HISTORY_PATHS:
        try:
            data = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for receipt in data.get("validation_receipts", []) if isinstance(data, Mapping) else []:
            if (
                isinstance(receipt, Mapping)
                and receipt.get("name") == "full_python_suite"
                and receipt.get("passed") is not True
            ):
                rows.append(
                    {
                        "observed_at": "2026-09-15",
                        "producer_path": str(relative),
                        "producer_sha256": sha256(root / relative),
                        "name": receipt.get("name"),
                        "command": receipt.get("command"),
                        "exit_code": receipt.get("exit_code"),
                        "duration_s": receipt.get("duration_s"),
                        "log_sha256": receipt.get("log_sha256"),
                        "collection_errors": [],
                        "resolved": False,
                    }
                )
    return rows


def _path_label(path: Path, root: Path) -> str:
    """Use stable repository paths while private fixtures stay absolute."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _hash_paths(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    """Hash each existing source, evaluator, sidecar, and frozen authority."""

    hashes: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        absolute = path if path.is_absolute() else root / path
        if absolute.is_file():
            hashes[_path_label(absolute, root)] = sha256(absolute)
    return hashes


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Record each local identity and exact failure before aggregation."""

    authority, roadmap, yaml_bytes, candidates = select_yaml_authority(root)
    rows = list(candidates)
    rows.append(
        {
            "check": "yaml_authority",
            "path": str(authority) if authority else None,
            "upstream": "research-roadmap-next.yaml|research-roadmap.yaml",
            "artifact_field": "milestone",
            "expected_value": MILESTONE,
            "observed_value": str(authority) if authority else None,
            "available": authority is not None,
            "blocking": True,
        }
    )
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "artifact_field": "bytes",
                "expected_value": "readable_nonempty_bytes",
                "observed_value": path.stat().st_size if available else "missing_or_empty",
                "available": available,
                "blocking": True,
            }
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    rows.append(
        {
            "check": "driving_requirement",
            "path": str(SPEC_PATH),
            "upstream": str(SPEC_PATH),
            "artifact_field": "REQ-*",
            "expected_value": "REQ-REPORT-7329",
            "observed_value": "REQ-REPORT-7329" if "REQ-REPORT-7329" in spec_text else "missing",
            "available": "REQ-REPORT-7329" in spec_text,
            "blocking": True,
        }
    )
    for name, directory in (
        ("result_directory", output_path.parent),
        ("raw_directory", raw_dir),
        ("checkpoint_directory", checkpoint_path.parent),
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            available, observed = True, "directory_writable"
        except OSError as exc:
            available, observed = False, f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "check": name,
                "path": str(directory),
                "upstream": str(directory),
                "artifact_field": "writable",
                "expected_value": True,
                "observed_value": observed,
                "available": available,
                "blocking": True,
            }
        )
    return rows, authority, roadmap, yaml_bytes


def _failure_summary(
    upstream: object, check: object, field: object, expected: object, observed: object
) -> JsonDict:
    """Keep the five exact values needed to diagnose a terminal failure."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def apply_terminal_state(artifact: JsonDict) -> None:
    """Derive the terminal score and verdict from stored checks only."""

    failed_precondition = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if row.get("blocking") and row.get("available") is not True
        ),
        None,
    )
    failed_gate = next(
        (
            row
            for row in artifact.get("acceptance_gate_results", [])
            if row.get("passed") is not True
        ),
        None,
    )
    artifact["contract_complete_score"] = int(
        failed_precondition is None
        and failed_gate is None
        and bool(artifact.get("acceptance_gate_results"))
    )
    artifact["science_readiness_score"] = 0
    artifact["science_value_score"] = 0
    artifact["science_promotion_score"] = 0
    artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    artifact["inference_substrate_class"] = "aggregation"
    if failed_precondition is not None:
        artifact.update(
            status="blocked",
            verdict_class="blocked",
            honest_verdict="blocked_v644_contract_required_input_missing",
            gate_check_summary=_failure_summary(
                failed_precondition.get("upstream"),
                failed_precondition.get("check"),
                failed_precondition.get("artifact_field"),
                failed_precondition.get("expected_value"),
                failed_precondition.get("observed_value"),
            ),
        )
    elif failed_gate is not None:
        artifact.update(
            status="complete",
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_v644_contract_mismatched_or_invalid",
            gate_check_summary=_failure_summary(
                FIRST_TASK_ID,
                failed_gate.get("criterion"),
                failed_gate.get("criterion"),
                failed_gate.get("expected"),
                failed_gate.get("observed"),
            ),
        )
    else:
        artifact.update(
            status="complete",
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_v644_exact_advisory_contract",
            gate_check_summary={
                "upstream": FIRST_TASK_ID,
                "failed_check": None,
                "artifact_field": "contract_complete_score",
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            },
        )


def _acceptance_gates(artifact: Mapping[str, Any], *, require_terminal: bool) -> list[JsonDict]:
    """Reduce contract, controls, sources, model state, and validation."""

    contract_rows = artifact.get("contract_rows", [])
    gate_rows = artifact.get("gate_control_rows", [])
    gate_count = len(gate_rows) // 6 if isinstance(gate_rows, list) else 0
    receipts = artifact.get("validation_receipts", [])
    raw_rows = artifact.get("raw_authority_rows", [])
    criteria: list[tuple[str, object, object, bool, str]] = [
        (
            "authority_selected",
            MILESTONE,
            artifact.get("yaml_milestone"),
            artifact.get("yaml_milestone") == MILESTONE,
            "Only the selected V644 roadmap can supply executable rows.",
        ),
        (
            "contract_exact",
            EXPECTED_TASK_COUNT,
            len(contract_rows) if isinstance(contract_rows, list) else None,
            artifact.get("contract_passed") is True,
            "All fourteen rows and all six compared fields must agree.",
        ),
        (
            "mutations_rejected",
            8,
            sum(row.get("rejected") is True for row in artifact.get("contract_mutation_rows", [])),
            len(artifact.get("contract_mutation_rows", [])) == 8
            and all(
                row.get("rejected") is True for row in artifact.get("contract_mutation_rows", [])
            ),
            "Named changes prove the contract can fail closed.",
        ),
        (
            "raw_authorities_hash_bound",
            True,
            [row.get("hash_matches") for row in raw_rows] if isinstance(raw_rows, list) else None,
            isinstance(raw_rows, list)
            and len(raw_rows) == 2
            and all(row.get("hash_matches") is True for row in raw_rows),
            "Frozen copies prevent later edits from changing this comparison.",
        ),
        (
            "gate_fields_declared",
            True,
            artifact.get("gate_declaration_result", {}).get("passed"),
            artifact.get("gate_declaration_result", {}).get("passed") is True,
            "Consumers can read only fields their producers promise.",
        ),
        (
            "gate_controls_complete",
            gate_count * 6,
            len(gate_rows) if isinstance(gate_rows, list) else None,
            gate_count > 0 and gate_controls_complete(gate_rows, gate_count),
            "The actual reader must reject every adverse producer case.",
        ),
        (
            "sources_complete",
            len(SOURCE_CHOICES),
            len(artifact.get("source_dispositions", [])),
            source_dispositions_complete(artifact.get("source_dispositions")),
            "Access can fail, but every bounded source decision remains visible.",
        ),
        (
            "model_contract",
            {"MODEL_SPECS": [], "model_invoked": False, "counts": ZERO_INVOCATION_COUNTS},
            {
                "MODEL_SPECS": artifact.get("MODEL_SPECS"),
                "model_invoked": artifact.get("model_invoked"),
                "counts": artifact.get("invocation_counts"),
            },
            artifact.get("MODEL_SPECS") == []
            and artifact.get("model_invoked") is False
            and artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS,
            "An aggregation receipt cannot claim a current model call.",
        ),
        (
            "affected_validation",
            True,
            artifact.get("required_checks_passed"),
            artifact.get("required_checks_passed") is True,
            "Any affected validation failure disqualifies current work.",
        ),
    ]
    if require_terminal:
        criteria.append(
            (
                "terminal_validation",
                list(TERMINAL_CHECK_NAMES),
                [row.get("name") for row in receipts if row.get("name") in TERMINAL_CHECK_NAMES],
                not _failed_names(receipts, TERMINAL_CHECK_NAMES),
                "Cold reduction and both strict terminal checks must pass.",
            )
        )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, observed, passed, principle in criteria
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every top-level value and preserve all required principles."""

    values = {field: "This field keeps the V644 advisory contract recheckable." for field in fields}
    values.update(REQUIRED_FIELD_PRINCIPLES)
    return values


def _phase_span(
    name: str,
    phase_start: float,
    run_start: float,
    units: int,
    checkpoint: str,
    pending: Sequence[str] = (),
) -> JsonDict:
    """Record one disjoint monotonic span without a duration floor."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_elapsed_s": phase_start - run_start,
        "end_elapsed_s": ended - run_start,
        "duration_s": ended - phase_start,
        "completed_units": units,
        "checkpoint_boundary": checkpoint,
        "pending_operations": list(pending),
    }


def _base_artifact(run_date: str, started_at: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint that cannot resemble terminal success."""

    return {
        "schema": "carnot.exp7329.v644_contract.v1",
        "experiment_id": FIRST_TASK_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {
                "planned": EXPECTED_TASK_COUNT,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
            },
            "gate_controls": {"planned": 0, "attempted": 0, "completed": 0, "censored": 0},
            "source_dispositions": {"planned": 4, "attempted": 0, "completed": 0, "censored": 0},
            "network_requests": {"planned": 4, "attempted": 0, "completed": 0, "censored": 0},
            "stopping_rule": "Stop after fourteen contract rows, eight named mutations, six controls per gate, four source requests, and one bounded validation sequence.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _failure_summary(
            FIRST_TASK_ID, "work_in_progress", "status", "terminal", "running"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_v644_contract",
        "verdict_class": "partial",
        "validation_receipts": [],
        "repository_health": {},
        "required_checks_passed": False,
        "contract_complete_score": 0,
        "science_readiness_score": 0,
        "science_value_score": 0,
        "science_promotion_score": 0,
        "contract_rows": [],
        "source_dispositions": [],
        "gate_control_rows": [],
        "checkpoint_path": str(checkpoint_path),
        "field_principles": {},
    }


def _finalize(artifact: JsonDict, started: float) -> None:
    """Seal actual duration, principles, and checksum after reduction."""

    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute evidence rows without trusting the positive score."""

    errors: list[str] = []
    rows = artifact.get("contract_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_TASK_COUNT
        or any(
            not isinstance(row.get("checks"), Mapping)
            or row.get("passed") != all(row["checks"].values())
            or row.get("censored") is not False
            for row in rows
        )
    ):
        errors.append("contract_row_reduction")
    gates = artifact.get("gate_control_rows")
    gate_count = len(gates) // 6 if isinstance(gates, list) else 0
    if gate_count == 0 or not gate_controls_complete(gates, gate_count):
        errors.append("gate_control_reduction")
    if not source_dispositions_complete(artifact.get("source_dispositions")):
        errors.append("source_disposition_reduction")
    raw = artifact.get("raw_authority_rows")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(row.get("source_sha256") != row.get("raw_sha256") for row in raw)
    ):
        errors.append("raw_authority_reduction")
    spans = artifact.get("phase_spans")
    if not isinstance(spans, list) or any(
        row.get("start_elapsed_s", 0) > row.get("end_elapsed_s", -1)
        or (index and row.get("start_elapsed_s", 0) < spans[index - 1].get("end_elapsed_s", 0))
        for index, row in enumerate(spans)
    ):
        errors.append("phase_span_reduction")
    return errors


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Recompute identity, hashes, rows, terminal state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != "carnot.exp7329.v644_contract.v1"
        or artifact.get("experiment_id") != FIRST_TASK_ID
        or artifact.get("milestone") != MILESTONE
    ):
        errors.append("identity_invalid")
    if artifact.get("run_date") != RUN_DATE or artifact.get("status") not in {
        "complete",
        "blocked",
    }:
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    if artifact.get("rows") != artifact.get("contract_rows"):
        errors.append("rows_invalid")
    principles = artifact.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or any(principles.get(field) != value for field, value in REQUIRED_FIELD_PRINCIPLES.items())
        or any(field not in principles for field in artifact)
    ):
        errors.append("field_principles_invalid")
    if artifact.get("verdict_class") != "blocked":
        errors.extend(independent_reduce(artifact))
        receipts = artifact.get("validation_receipts", [])
        receipt_names = [row.get("name") for row in receipts]
        malformed_receipts = any(receipt_names.count(name) != 1 for name in ALL_VALIDATION_NAMES)
        terminal_failures = _failed_names(receipts, TERMINAL_CHECK_NAMES)
        if malformed_receipts or terminal_failures:
            errors.append("validation_receipts_invalid")
    for label, expected_hash in artifact.get("source_artifact_hashes", {}).items():
        path = Path(str(label))
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file() or sha256(absolute) != expected_hash:
            errors.append("source_hash_mismatch")
            break
    expected = deepcopy(dict(artifact))
    apply_terminal_state(expected)
    for field in (
        "status",
        "contract_complete_score",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
        "science_readiness_score",
        "science_value_score",
        "science_promotion_score",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = _fetch_url,
    validation_runner: ValidationRunner = run_affected_validation,
) -> JsonDict:
    """Measure, validate, and atomically publish the V644 contract receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "write a running checkpoint before checks")
    artifact = _base_artifact(run_date, started_at, checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint_path, artifact)

    phase_start = time.monotonic()
    progress(1, "start", "authenticate inputs and resolve the YAML authority once")
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["phase_spans"].append(
        _phase_span(
            "preconditions", phase_start, started, len(preconditions), "running_checkpoint_written"
        )
    )
    if any(row.get("blocking") and row.get("available") is not True for row in preconditions):
        apply_terminal_state(artifact)
        _finalize(artifact, started)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_path, artifact)
        progress(6, "terminal_write", f"blocked output={output_path}")
        return artifact
    assert authority is not None and roadmap is not None and yaml_bytes is not None

    phase_start = time.monotonic()
    progress(2, "start", "freeze and compare independent Markdown and YAML bytes")
    raw_dir.mkdir(parents=True, exist_ok=True)
    design_path = root / str(roadmap.get("milestone_doc", DESIGN_PATH))
    design_bytes = design_path.read_bytes()
    design_copy = raw_dir / "selected-design.md"
    yaml_copy = raw_dir / "selected-roadmap.yaml"
    _atomic_write_bytes(design_copy, design_bytes)
    _atomic_write_bytes(yaml_copy, yaml_bytes)
    artifact["raw_authority_rows"] = [
        {
            "source_type": "markdown",
            "source_path": _path_label(design_path, root),
            "raw_path": _path_label(design_copy, root),
            "source_sha256": sha256_bytes(design_bytes),
            "raw_sha256": sha256(design_copy),
            "hash_matches": sha256_bytes(design_bytes) == sha256(design_copy),
        },
        {
            "source_type": "yaml",
            "source_path": str(authority),
            "raw_path": _path_label(yaml_copy, root),
            "source_sha256": sha256_bytes(yaml_bytes),
            "raw_sha256": sha256(yaml_copy),
            "hash_matches": sha256_bytes(yaml_bytes) == sha256(yaml_copy),
        },
    ]
    contract = evaluate_contract(design_bytes.decode("utf-8"), roadmap)
    artifact.update(contract)
    artifact["contract_passed"] = contract["passed"]
    with tempfile.TemporaryDirectory(prefix="exp7329-mutations-", dir="/tmp") as directory:
        artifact["contract_mutation_rows"] = run_contract_mutations(
            design_bytes.decode("utf-8"), roadmap, Path(directory)
        )
    artifact["rows"] = artifact["contract_rows"]
    artifact["sample_size_budget"]["contract_rows"].update(
        attempted=len(artifact["contract_rows"]), completed=len(artifact["contract_rows"])
    )
    artifact["phase_spans"].append(
        _phase_span(
            "contract", phase_start, started, len(artifact["contract_rows"]), "authorities_frozen"
        )
    )

    phase_start = time.monotonic()
    progress(3, "start", "exercise every gate edge through the real reader")
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    artifact["gate_control_rows"] = run_gate_controls(roadmap, raw_dir / "gate-controls")
    artifact["sample_size_budget"]["gate_controls"].update(
        planned=gate_count * 6,
        attempted=len(artifact["gate_control_rows"]),
        completed=len(artifact["gate_control_rows"]),
    )
    gate_sidecar = raw_dir / "gate-fixture-model-evidence.json"
    _atomic_write_json(
        gate_sidecar,
        {
            "schema": "carnot.exp7329.gate_fixture_model_evidence.v1",
            "current_model_invoked": False,
            "fixtures": [
                {
                    "case": "disqualified_score_one",
                    "MODEL_SPECS": ["injected/noncurrent-fixture"],
                    "accepted_as_dependency": False,
                }
            ],
        },
    )
    artifact["phase_spans"].append(
        _phase_span(
            "gate_controls",
            phase_start,
            started,
            len(artifact["gate_control_rows"]),
            "gate_controls_complete",
        )
    )

    phase_start = time.monotonic()
    progress(4, "start", "perform four bounded primary-source checks")
    dispositions, access_rows = collect_source_dispositions(fetcher)
    artifact["source_dispositions"] = dispositions
    access_sidecar = raw_dir / "source-access-receipts.json"
    _atomic_write_json(
        access_sidecar, {"schema": "carnot.exp7329.source_access.v1", "receipts": access_rows}
    )
    history_sidecar = raw_dir / "historical-model-receipts.json"
    history_rows = []
    for relative in HISTORY_PATHS:
        data = json.loads((root / relative).read_text(encoding="utf-8"))
        history_rows.append(
            {
                "path": str(relative),
                "sha256": sha256(root / relative),
                "status": data.get("status"),
                "verdict_class": data.get("verdict_class"),
                "honest_verdict": data.get("honest_verdict"),
                "MODEL_SPECS": data.get("MODEL_SPECS"),
                "model_invoked": data.get("model_invoked"),
                "accepted_as_dependency": False,
            }
        )
    _atomic_write_json(
        history_sidecar,
        {
            "schema": "carnot.exp7329.historical_model_receipts.v1",
            "current_invocation": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_artifacts": history_rows,
        },
    )
    artifact["sidecar_receipts"] = {
        "source_access": {
            "path": _path_label(access_sidecar, root),
            "sha256": sha256(access_sidecar),
        },
        "gate_fixture_models": {
            "path": _path_label(gate_sidecar, root),
            "sha256": sha256(gate_sidecar),
        },
        "historical_models": {
            "path": _path_label(history_sidecar, root),
            "sha256": sha256(history_sidecar),
        },
    }
    artifact["sample_size_budget"]["source_dispositions"].update(attempted=4, completed=4)
    artifact["sample_size_budget"]["network_requests"].update(
        attempted=4,
        completed=sum(row["access_outcome"] == "http_success" for row in access_rows),
        censored=sum(row["access_outcome"] != "http_success" for row in access_rows),
    )
    artifact["phase_spans"].append(
        _phase_span("sources", phase_start, started, 4, "source_sidecars_written")
    )

    phase_start = time.monotonic()
    progress(5, "start", "run selected-roadmap audits and affected validation")
    validation = validation_runner(
        root=root,
        authority=authority,
        raw_dir=raw_dir,
        test_paths=[str(TEST_PATH)],
        changed_modules=[str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        historical_failures=_historical_failures(root),
    )
    artifact["validation_receipts"] = list(validation.get("validation_receipts", []))
    artifact["required_checks_passed"] = validation.get("required_checks_passed") is True
    artifact["repository_health"] = validation.get("repository_health", {})
    artifact["source_artifact_hashes"] = _hash_paths(
        root,
        [
            *INPUT_PATHS,
            authority,
            design_path,
            design_copy,
            yaml_copy,
            access_sidecar,
            gate_sidecar,
            history_sidecar,
        ],
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=False)
    apply_terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "affected_validation",
            phase_start,
            started,
            len(artifact["validation_receipts"]),
            "measured_candidate_next",
            TERMINAL_CHECK_NAMES,
        )
    )
    _finalize(artifact, started)
    candidate = raw_dir / RAW_CANDIDATE_NAME
    _atomic_write_json(candidate, artifact)
    progress(5, "candidate_written", f"path={candidate}")

    terminal_start = time.monotonic()
    artifact["validation_receipts"].extend(run_terminal_validation(root, candidate, raw_dir))
    artifact["required_checks_passed"] = not _failed_names(
        artifact["validation_receipts"], ALL_VALIDATION_NAMES
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=True)
    apply_terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "terminal_validation",
            terminal_start,
            started,
            len(TERMINAL_CHECK_NAMES),
            "terminal_output_ready",
        )
    )
    _finalize(artifact, started)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, artifact)
    progress(6, "terminal_write", f"output={output_path} verdict={artifact['honest_verdict']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V644 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - required E2E entrypoint.
    """Run the advisory receipt and require a valid terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    artifact = build_artifact(
        REPO_ROOT, args.date, output_path=output, raw_dir=raw_dir, checkpoint_path=checkpoint
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7329] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7329] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0
