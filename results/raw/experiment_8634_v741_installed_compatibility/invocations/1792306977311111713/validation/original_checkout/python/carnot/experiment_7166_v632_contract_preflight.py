"""Compare the V632 Markdown contract with the V632 roadmap YAML.

The Markdown and YAML parsers receive separate inputs. Each parser creates its
own rows. This keeps a missing task visible instead of copying one omission to
both sides. The command never activates or repairs a roadmap.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import yaml

from carnot.experiment_7151_v629_contract_preflight import (
    _canonical_gates,
    _field_declared,
    parse_markdown_contract,
    parse_yaml_contract,
    retired_experiment_ids,
    task_number,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
PRIOR_ARTIFACT_PATH = Path("results/experiment_7159_v631_contract_preflight.json")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
HARNESS_LINT_PATH = Path("scripts/harness_fit_lint.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")
ROOT_CLUTTER_PATH = Path("scripts/root_clutter_sweep.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7166_v632_contract_preflight.json")
MODULE_PATH = Path("python/carnot/experiment_7166_v632_contract_preflight.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7166_v632_contract_preflight.py")
TEST_PATH = Path("tests/python/test_experiment_7166_v632_contract_preflight.py")

MILESTONE = "2026.09.632"
EXPECTED_TASK_COUNT = 13
EXPECTED_NUMBERS = tuple(range(7166, 7179))
EXPECTED_ID_ORDER = (
    "exp7166-v632-exact-contract-preflight",
    "exp7167-qwen38-claim-evidence-trace-capture",
    "exp7168-claim-evidence-energy-comparison",
    "exp7169-claim-evidence-causal-audit",
    "exp7170-supersession-aware-constraint-stream",
    "exp7171-procedural-graph-constraint-memory-csl",
    "exp7172-budgeted-stale-memory-allocation-ab",
    "exp7173-cold-retention-rollback-audit",
    "exp7174-live-arc-adapter-path-withheld-ab",
    "exp7175-fixed-magnetization-pair-swap-benchmark",
    "exp7176-rust-fixed-magnetization-parity",
    "exp7177-sampler-hardware-placement-audit",
    "exp7178-v632-independent-capstone",
)
FIRST_TASK_ID = "exp7166-v632-exact-contract-preflight"
LAST_TASK_ID = "exp7178-v632-independent-capstone"
PROMPT_FINAL_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
CURRENT_MODEL = "unsloth/Qwen3.8-27B-GGUF"
SUPERSEDED_HEADLINE_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/gemma-4-12B-it-GGUF",
)
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)
CANONICAL_SUBSTRATE_CLASSES = frozenset(
    {
        "aggregation",
        "model_bounded_generation",
        "model_full_generation",
        "cpu_exact_solver_or_simulator",
        "no_model_load",
        "hardware_smoke",
        "blocked_no_run",
    }
)
KNOWN_SUBSTRATE_CLASSES = {
    7166: "aggregation",
    7167: "model_full_generation",
    7168: "no_model_load",
    7169: "no_model_load",
    7170: "no_model_load",
    7171: "cpu_exact_solver_or_simulator",
    7172: "cpu_exact_solver_or_simulator",
    7173: "no_model_load",
    7174: "model_full_generation",
    7175: "cpu_exact_solver_or_simulator",
    7176: "cpu_exact_solver_or_simulator",
    7177: "aggregation",
    7178: "aggregation",
}
MODEL_TASK_NUMBERS = frozenset({7167, 7174})
INFERENCE_SUBSTRATE = "aggregation_from_independent_v632_contract_parsers"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
RANDOM_SEED = 716620260909
BLOCKED_VERDICT = "blocked_v632_contract_preflight_prerequisite_missing"
POSITIVE_VERDICT = "complete_positive_v632_task_contract_conforms"
DISQUALIFIED_VERDICT = "complete_disqualified_v632_markdown_yaml_contract_mismatch"

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": "Each required field explains the evidence it protects.",
    "status": "A terminal lifecycle value prevents a bootstrap shell from passing.",
    "preconditions_checked": "Named document and lint checks distinguish absence from mismatch.",
    "run_date": "The date binds the audit to the planning files inspected.",
    "inference_substrate": "Use aggregation_from_independent_v632_contract_parsers.",
    "inference_substrate_class": "Use aggregation, or blocked_no_run before comparison, because no model is loaded.",
    "execution_venue": "Host identity records where the contract was checked.",
    "duration_s": "Measured time exposes a copied or truncated audit.",
    "source_artifact_hashes": "Hashes bind the result to exact Markdown, YAML, schemas, and lints.",
    "rows": "One row per task and contract dimension makes the conclusion reconstructable.",
    "markdown_task_rows": "Independent Markdown rows expose an aspirational or stale design.",
    "yaml_task_rows": "Independent YAML rows expose partial emission.",
    "gate_contract_rows": "Gate rows preserve upstream, field, operator, and value parity.",
    "model_compliance_rows": "Model rows require Qwen3.8 only on LLM-bearing tasks and reject superseded mandates.",
    "progress_contract_rows": "Progress rows prove each prompt protects the 600-second silence boundary.",
    "expected_task_count": "The literal value 13 is the load-bearing contract.",
    "observed_task_count": "The independently observed count catches missing execution tasks.",
    "expected_id_order": "Full ordered IDs prevent partial roadmap activation.",
    "observed_id_order": "Observed IDs make drift explicit.",
    "v632_task_contract_conforms_score": "One means all thirteen rows agree and makes no science claim.",
    "random_seed": "A fixed seed makes mutation-test ordering reproducible.",
    "reproducibility_checksum": "The checksum detects silent contract-input drift.",
    "gate_check_summary": "A blocked or disqualified verdict names the failed check and values.",
    "verifier_is_oracle": "False records that this checks documents, not its own correctness.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial.",
    "honest_verdict": "Free text preserves the terminal contract conclusion.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)
TASK_COMMON_FIELDS = (
    "field_principles",
    "status",
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
SUPPORTING_ROW_FIELDS = (
    "authority_consistency_rows",
    "gate_contract_rows",
    "gate_producer_rows",
    "prior_failure_rows",
    "operator_override_rows",
    "model_compliance_rows",
    "agent_routing_rows",
    "substrate_contract_rows",
    "progress_contract_rows",
)
VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "gate",
    "exclusion_manifest",
    "harness_fit",
    "artifact",
    "adversarial",
    "row_consistency",
    "scoped_spec_coverage",
    "root_clutter",
)


def _progress(phase: int, state: str, detail: str) -> None:
    """Flush each phase boundary so the conductor can detect live work."""

    print(f"[exp7166] phase {phase} {state}: {detail}", flush=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one non-empty YAML mapping with a stable error message."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML at {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"YAML mapping required at {path}")
    return value


def _required_block(prompt: str) -> str:
    """Return only declarations, so normal prose cannot satisfy a field check."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def _valid_task_id(value: object, number: int) -> bool:
    """Require the published full V632 ID at this exact position."""

    text = str(value)
    if re.fullmatch(rf"exp{number}-[a-z0-9][a-z0-9-]*", text) is None:
        return False
    if number not in EXPECTED_NUMBERS:
        return False
    expected = EXPECTED_ID_ORDER[number - EXPECTED_NUMBERS[0]]
    return text == expected


def _closed_verdict_enum(block: str) -> bool:
    """Find the six verdict classes in their fixed order after the field name."""

    field = block.find("verdict_class")
    if field < 0:
        return False
    suffix = re.sub(r"\s+", " ", block[field:])
    values = r"\s*\|\s*".join(map(re.escape, VERDICT_CLASSES))
    return re.search(values, suffix) is not None


def _substrate_class(block: str) -> str | None:
    """Read the planned class before the blocked no-run alternative."""

    field = block.find("inference_substrate_class")
    if field < 0:
        return None
    suffix = block[field : field + 320]
    match = re.search(r"\bUse\s+([a-z_]+)", suffix, re.I)
    if match is None:
        match = re.search(r"inference_substrate_class\s*\(\s*([a-z_]+)", suffix)
    return match.group(1).lower() if match else None


def _public_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the parsed source evidence small and recheckable."""

    return {
        key: task.get(key)
        for key in ("order", "id", "number", "title", "deliverable", "milestone", "gates")
    }


def _prior_rows(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check all four required fields in each declared prior failure."""

    priors = task.get("prior_failures", [])
    if not priors:
        return [
            {
                "order": task["order"],
                "task_id": task["id"],
                "prior_order": None,
                "field_checks": {"no_prior_declared": True},
                "passed": True,
            }
        ]
    rows = []
    for index, prior in enumerate(priors, 1):
        values = dict(prior) if isinstance(prior, Mapping) else {}
        checks = {
            name: isinstance(values.get(name), str) and bool(values[name].strip())
            for name in ("experiment_id", "verdict", "addressed_by")
        }
        checks["retire_if_same_verdict"] = values.get("retire_if_same_verdict") is True
        rows.append(
            {
                "order": task["order"],
                "task_id": task["id"],
                "prior_order": index,
                "values": values,
                "field_checks": checks,
                "passed": all(checks.values()),
            }
        )
    return rows


def _override_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Accept no override or one auditable non-empty line."""

    value = task.get("operator_override")
    absent = value is None
    present = isinstance(value, str) and bool(value.strip())
    checks = {
        "absent_or_string": absent or present,
        "one_line": absent or (present and "\n" not in value),
        "minimum_length": absent or (present and len(value.strip()) >= 10),
    }
    return {
        "order": task["order"],
        "task_id": task["id"],
        "observed": value,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _route_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Allow the documented default route or the exact Codex route."""

    agent = task.get("agent_type")
    model = task.get("model")
    route_ok = (agent is None and model in {None, "opus"}) or (
        agent == "codex" and model == "gpt-5.6-sol"
    )
    metadata = {
        "track": isinstance(task.get("track"), str) and bool(task["track"].strip()),
        "priority": task.get("priority") in {"critical", "high", "medium"},
        "requires_gpu": isinstance(task.get("requires_gpu"), bool),
        "max_turns": isinstance(task.get("max_turns"), int) and task["max_turns"] > 0,
        "estimated_wall_time_min": isinstance(task.get("estimated_wall_time_min"), int)
        and task["estimated_wall_time_min"] > 0,
    }
    return {
        "order": task["order"],
        "task_id": task["id"],
        "observed_route": [agent, model],
        "metadata_checks": metadata,
        "passed": route_ok and all(metadata.values()),
    }


def _model_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Require Qwen3.8 for model work and reject old headline declarations."""

    block = task["required_block"]
    declared = _field_declared(block, "MODEL_SPECS")
    current = CURRENT_MODEL in block
    superseded = [model for model in SUPERSEDED_HEADLINE_MODELS if model in block]
    model_task = task.get("number") in MODEL_TASK_NUMBERS
    passed = declared and current and not superseded if model_task else not declared
    return {
        "order": task["order"],
        "task_id": task["id"],
        "model_task": model_task,
        "model_specs_declared": declared,
        "current_model_declared": current,
        "superseded_headline_models": superseded,
        "passed": passed,
    }


def _substrate_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Require one canonical planned class and known V632 assignments."""

    observed = _substrate_class(task["required_block"])
    expected = KNOWN_SUBSTRATE_CLASSES.get(task["number"])
    passed = observed in CANONICAL_SUBSTRATE_CLASSES and (expected is None or observed == expected)
    return {
        "order": task["order"],
        "task_id": task["id"],
        "expected": expected or "canonical_declared_class",
        "observed": observed,
        "passed": passed,
    }


def _progress_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Check every anti-stall, placeholder, run, and final-tail duty."""

    prompt = task["prompt"]
    normalized = re.sub(r"\s+", " ", prompt.lower())
    lines = prompt.rstrip().splitlines()
    run_line = lines[-2] if len(lines) >= 2 else ""
    checks = {
        "context_section": "context:" in normalized,
        "existing_code_section": "existing code to read first:" in normalized,
        "task_section": "task:" in normalized,
        "concrete_steps_section": "concrete steps:" in normalized,
        "phase_boundaries": "every numbered phase boundary, print a flushed progress line"
        in normalized,
        "slow_call_start_end": all(
            phrase in normalized
            for phrase in (
                "flushed start",
                "end lines immediately before and after",
                "model load",
                "generation",
                "benchmark",
                "subprocess",
            )
        ),
        "long_loop_heartbeat": "at least every 300 seconds" in normalized,
        "stdout_gap": "every stdout gap below 600 seconds" in normalized,
        "project_root_placeholder": "{project_root}" in prompt,
        "date_placeholder": "{date}" in prompt,
        "run_command": bool(
            re.fullmatch(
                r"Run command: cd \{project_root\} && \.venv/bin/python "
                r"scripts/experiments/experiment_\d+_[^ ]+\.py --date \{date\}",
                run_line,
            )
        ),
        "final_prohibition": bool(lines and lines[-1] == PROMPT_FINAL_LINE),
    }
    return {
        "order": task["order"],
        "task_id": task["id"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _discipline_checks(task: Mapping[str, Any], retired_ids: set[str]) -> dict[str, bool]:
    """Check fields shared by every active V632 task."""

    block = task["required_block"]
    normal_id = f"exp{task['number']}" if task.get("number") is not None else None
    normalized = re.sub(r"\s+", " ", block.lower())
    return {
        "required_block": bool(block),
        "common_fields": all(_field_declared(block, field) for field in TASK_COMMON_FIELDS),
        "field_principles": "principle:" in normalized,
        "blocked_diagnostics": "check" in normalized and "value" in normalized,
        "closed_verdict_enum": _closed_verdict_enum(block),
        "per_unit_rows": task.get("per_unit_rows") is True,
        "not_retired": normal_id not in retired_ids,
    }


def evaluate_contract(
    markdown_text: str, yaml_document: object, retired_ids: set[str]
) -> dict[str, Any]:
    """Compare independent V632 task views and keep each decision as evidence."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    retired = {
        f"exp{number}" for value in retired_ids if (number := task_number(str(value))) is not None
    }
    number_counts = Counter(task["number"] for task in yaml_tasks)
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))

    gate_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    substrate_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    discipline: dict[int, dict[str, bool]] = {}

    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        source = markdown_task or yaml_task or {}
        gate_rows.append(
            {
                "order": order,
                "task_id": source.get("id"),
                "expected": markdown_task.get("gates") if markdown_task else None,
                "observed": yaml_task.get("gates") if yaml_task else None,
                "passed": bool(
                    markdown_task
                    and yaml_task
                    and markdown_task["id"] == yaml_task["id"]
                    and _canonical_gates(markdown_task["gates"])
                    == _canonical_gates(yaml_task["gates"])
                ),
            }
        )
        if yaml_task is None:
            missing = {"order": order, "task_id": source.get("id"), "passed": False}
            prior_rows.append({**missing, "prior_order": None})
            override_rows.append({**missing, "checks": {}})
            model_rows.append({**missing, "model_specs_declared": False})
            route_rows.append({**missing, "observed_route": None})
            substrate_rows.append({**missing, "observed": None})
            progress_rows.append({**missing, "checks": {}})
            discipline[order] = {"yaml_present": False}
            continue
        yaml_task["required_block"] = _required_block(yaml_task["prompt"])
        prior_rows.extend(_prior_rows(yaml_task))
        override_rows.append(_override_row(yaml_task))
        model_rows.append(_model_row(yaml_task))
        route_rows.append(_route_row(yaml_task))
        substrate_rows.append(_substrate_row(yaml_task))
        progress_rows.append(_progress_row(yaml_task))
        discipline[order] = _discipline_checks(yaml_task, retired)

    yaml_by_id = {task["id"]: task for task in yaml_tasks}
    producer_rows: list[dict[str, Any]] = []
    for consumer in yaml_tasks:
        for gate in consumer["gates"]:
            producer = yaml_by_id.get(gate.get("upstream"))
            field = gate.get("artifact_field")
            bare = (
                isinstance(field, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field) is not None
            )
            declared = bool(
                producer and bare and _field_declared(_required_block(producer["prompt"]), field)
            )
            precedes = bool(producer and producer["order"] < consumer["order"])
            producer_rows.append(
                {
                    "order": consumer["order"],
                    "consumer": consumer["id"],
                    "upstream": gate.get("upstream"),
                    "artifact_field": field,
                    "producer_present": producer is not None,
                    "producer_precedes_consumer": precedes,
                    "bare_field": bare,
                    "field_declared": declared,
                    "passed": bool(
                        producer
                        and precedes
                        and bare
                        and declared
                        and producer["milestone"] == consumer["milestone"] == MILESTONE
                    ),
                }
            )

    supporting = (
        gate_rows,
        prior_rows,
        override_rows,
        model_rows,
        route_rows,
        substrate_rows,
        progress_rows,
        producer_rows,
    )
    checks_by_order: dict[int, list[bool]] = {}
    for evidence_rows in supporting:
        for row in evidence_rows:
            checks_by_order.setdefault(row["order"], []).append(row["passed"] is True)

    task_rows: list[dict[str, Any]] = []
    for index in range(width):
        order = index + 1
        number = EXPECTED_NUMBERS[index] if index < EXPECTED_TASK_COUNT else None
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        identity = {
            "markdown_present": markdown_task is not None,
            "yaml_present": yaml_task is not None,
            "markdown_full_id": bool(
                markdown_task and number and _valid_task_id(markdown_task["id"], number)
            ),
            "yaml_full_id": bool(yaml_task and number and _valid_task_id(yaml_task["id"], number)),
            "id_parity": bool(
                markdown_task and yaml_task and markdown_task["id"] == yaml_task["id"]
            ),
            "title_parity": bool(
                markdown_task and yaml_task and markdown_task["title"] == yaml_task["title"]
            ),
            "deliverable_parity": bool(
                markdown_task
                and yaml_task
                and markdown_task["deliverable"] == yaml_task["deliverable"]
            ),
            "milestone_parity": bool(
                markdown_task
                and yaml_task
                and markdown_task["milestone"] == yaml_task["milestone"] == MILESTONE
            ),
            "unique_number": bool(yaml_task and number_counts[yaml_task["number"]] == 1),
        }
        policy = discipline.get(order, {"yaml_present": False})
        task_rows.append(
            {
                "order": order,
                "expected_number": number,
                "expected_id": markdown_task.get("id") if markdown_task else None,
                "observed_id": yaml_task.get("id") if yaml_task else None,
                "identity_checks": identity,
                "discipline_checks": policy,
                "passed": all(identity.values())
                and all(policy.values())
                and all(checks_by_order.get(order, [False])),
            }
        )

    markdown_ids = [task["id"] for task in markdown_tasks]
    yaml_ids = [task["id"] for task in yaml_tasks]
    global_checks = (
        markdown["milestone"] == MILESTONE,
        roadmap["milestone"] == MILESTONE,
        len(markdown_tasks) == EXPECTED_TASK_COUNT,
        len(yaml_tasks) == EXPECTED_TASK_COUNT,
        [task["number"] for task in markdown_tasks] == list(EXPECTED_NUMBERS),
        [task["number"] for task in yaml_tasks] == list(EXPECTED_NUMBERS),
        len(number_counts) == EXPECTED_TASK_COUNT,
        markdown_ids == list(EXPECTED_ID_ORDER),
        yaml_ids == list(EXPECTED_ID_ORDER),
        markdown_ids == yaml_ids,
    )
    passed = bool(
        all(global_checks)
        and len(task_rows) == EXPECTED_TASK_COUNT
        and all(row["passed"] for row in task_rows)
    )
    return {
        "passed": passed,
        "markdown_task_rows": [_public_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task(task) for task in yaml_tasks],
        "task_contract_rows": task_rows,
        "gate_contract_rows": gate_rows,
        "gate_producer_rows": producer_rows,
        "prior_failure_rows": prior_rows,
        "operator_override_rows": override_rows,
        "model_compliance_rows": model_rows,
        "agent_routing_rows": route_rows,
        "substrate_contract_rows": substrate_rows,
        "progress_contract_rows": progress_rows,
        "authority_consistency_rows": [],
        "markdown_task_count": len(markdown_tasks),
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "observed_id_order": yaml_ids,
    }


def _sha256(path: Path) -> str:
    """Hash exact bytes so later source changes remain visible."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str | None]:
    """Hash each contract, schema, lint, implementation, and focused test."""

    paths = (
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        NEXT_ROADMAP_PATH,
        ACTIVE_ROADMAP_PATH,
        DESIGN_PATH,
        PRIOR_ARTIFACT_PATH,
        ROADMAP_SCHEMA_PATH,
        PRIOR_LINT_PATH,
        GATE_LINT_PATH,
        EXCLUSION_LINT_PATH,
        HARNESS_LINT_PATH,
        ADVERSARIAL_PATH,
        ROW_LINT_PATH,
        SPEC_COVERAGE_PATH,
        ROOT_CLUTTER_PATH,
        SPEC_PATH,
        EXCLUSION_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        str(relative): _sha256(root / relative) if (root / relative).is_file() else None
        for relative in paths
    }


def _field_principles() -> dict[str, str]:
    """Use the task's exact reasons and explain every supporting field."""

    fields = set(REQUIRED_ARTIFACT_FIELDS) | {
        "schema",
        "experiment_id",
        "task_contract_rows",
        "markdown_task_rows",
        "yaml_task_rows",
        "authority_consistency_rows",
        "gate_producer_rows",
        "prior_failure_rows",
        "operator_override_rows",
        "agent_routing_rows",
        "substrate_contract_rows",
        "validation_command_rows",
        "yaml_authority_path",
    }
    principles = {
        field: f"The {field} evidence lets an auditor recheck the V632 contract verdict."
        for field in sorted(fields)
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all fields except the location that stores this checksum."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result atomically so readers never see partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".exp7166-", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _checkpoint(path: Path, artifact: dict[str, Any], started: float) -> None:
    """Persist current duration and checksum at each safe boundary."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_artifact(path, artifact)


def _base_artifact(run_date: str) -> dict[str, Any]:
    """Create every schema field before the first prerequisite check."""

    empty_rows = {field: [] for field in SUPPORTING_ROW_FIELDS}
    return {
        "schema": "carnot.exp7166.v632_contract_preflight.v1",
        "experiment_id": FIRST_TASK_ID,
        "status": "running",
        "field_principles": _field_principles(),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "task_contract_rows": [],
        **empty_rows,
        "validation_command_rows": [],
        "yaml_authority_path": None,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "observed_id_order": [],
        "v632_task_contract_conforms_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "preconditions_not_checked",
            "expected_value": "all_required_inputs_available",
            "observed_value": "not_checked",
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_v632_contract_preflight",
    }


def _preconditions(
    root: Path, output_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, Path | None]:
    """Check every named input after the initial artifact is durable."""

    rows: list[dict[str, Any]] = []
    roadmap: dict[str, Any] | None = None
    authority: Path | None = None

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

    design = root / DESIGN_PATH
    try:
        readable = bool(design.read_text(encoding="utf-8").strip())
        add(
            "markdown_readable",
            design,
            "readable_nonempty_source",
            readable,
            "readable_nonempty_source" if readable else "empty_source",
        )
    except OSError as exc:
        add(
            "markdown_readable",
            design,
            "readable_nonempty_source",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    # A present next roadmap remains authoritative. After activation consumes
    # it, the active roadmap is eligible only when it declares V632.
    candidate = NEXT_ROADMAP_PATH if (root / NEXT_ROADMAP_PATH).is_file() else ACTIVE_ROADMAP_PATH
    path = root / candidate
    try:
        document = _load_yaml(path)
        milestone_ok = document.get("milestone") == MILESTONE
        if milestone_ok:
            roadmap = document
            authority = candidate
        observed = (
            "readable_v632_yaml_mapping"
            if milestone_ok
            else f"milestone={document.get('milestone')!r}"
        )
        add(
            "yaml_authority_readable",
            path,
            "readable_v632_yaml_mapping",
            milestone_ok,
            observed,
        )
    except ValueError as exc:
        add(
            "yaml_authority_readable",
            path,
            "readable_v632_yaml_mapping",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    exclusion_path = root / EXCLUSION_PATH
    try:
        _load_yaml(exclusion_path)
        add(
            "exclusion_manifest_readable",
            exclusion_path,
            "readable_nonempty_yaml_mapping",
            True,
            "readable_nonempty_yaml_mapping",
        )
    except ValueError as exc:
        add(
            "exclusion_manifest_readable",
            exclusion_path,
            "readable_nonempty_yaml_mapping",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    required_files = (
        PRIOR_ARTIFACT_PATH,
        SPEC_PATH,
        ROADMAP_SCHEMA_PATH,
        PRIOR_LINT_PATH,
        GATE_LINT_PATH,
        EXCLUSION_LINT_PATH,
        HARNESS_LINT_PATH,
        ADVERSARIAL_PATH,
        ROW_LINT_PATH,
        SPEC_COVERAGE_PATH,
        ROOT_CLUTTER_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    for relative in required_files:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        add(
            f"input_readable:{relative}",
            path,
            "readable_nonempty_file",
            available,
            "readable_nonempty_file" if available else "missing_or_empty",
        )

    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".exp7166-write-"):
            pass
        writable, observed = True, "temporary_write_succeeded"
    except OSError as exc:
        writable, observed = False, f"{type(exc).__name__}: {exc}"
    add("artifact_path_writable", output_path, "temporary_write_succeeds", writable, observed)
    return rows, roadmap, authority


def _authority_consistency_rows(
    root: Path, authority: Path, roadmap: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Compare an optional active file with the selected next-roadmap contract."""

    if authority != NEXT_ROADMAP_PATH or not (root / ACTIVE_ROADMAP_PATH).is_file():
        return []
    try:
        selected = parse_yaml_contract(roadmap)
        active = parse_yaml_contract(_load_yaml(root / ACTIVE_ROADMAP_PATH))
        passed = selected == active
        observed = "identical_v632_contract" if passed else "different_contract"
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        passed = False
        observed = f"{type(exc).__name__}: {exc}"
    return [
        {
            "check": "active_yaml_contract_parity",
            "authority": str(authority),
            "compared_path": str(ACTIVE_ROADMAP_PATH),
            "expected_value": "identical_v632_contract",
            "observed_value": observed,
            "passed": passed,
        }
    ]


def _failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while rows retain local causes."""

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
    elif any(not row["passed"] for row in contract.get("authority_consistency_rows", [])):
        check, expected, observed = "active_yaml_contract_parity", True, False
    else:
        first = next(row for row in contract["task_contract_rows"] if not row["passed"])
        check, expected, observed = f"task_contract_order_{first['order']}", True, False
    return {
        "failed_check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def _validation_commands(
    root: Path, output_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Build the ten read-only validation commands required by the task."""

    python = str(Path(sys.executable))
    roadmap = str(root / authority)
    output = str(output_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    artifact_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7166_v632_contract_preflight import validate_artifact;"
        f"errors=validate_artifact(json.loads(pathlib.Path({output!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    return (
        ("roadmap_schema", [python, "-c", schema_code]),
        ("prior_failure", [python, str(root / PRIOR_LINT_PATH), roadmap]),
        ("gate", [python, str(root / GATE_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, str(root / EXCLUSION_LINT_PATH), roadmap]),
        ("harness_fit", [python, str(root / HARNESS_LINT_PATH), roadmap]),
        ("artifact", [python, "-c", artifact_code]),
        ("adversarial", [python, str(root / ADVERSARIAL_PATH), output]),
        ("row_consistency", [python, str(root / ROW_LINT_PATH), output]),
        ("scoped_spec_coverage", [python, str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)]),
        ("root_clutter", [python, str(root / ROOT_CLUTTER_PATH)]),
    )


def run_validation_commands(
    root: Path, output_path: Path, artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run each bounded subprocess with flushed boundaries and receipts."""

    rows: list[dict[str, Any]] = []
    authority = Path(str(artifact.get("yaml_authority_path") or ACTIVE_ROADMAP_PATH))
    for index, (name, argv) in enumerate(_validation_commands(root, output_path, authority), 1):
        _progress(3, "subprocess start", f"{index}/10 {name}")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout[-8000:]
            stderr = completed.stderr[-8000:]
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = str(exc.stdout or "")[-8000:]
            stderr = (str(exc.stderr or "") + "\nsubprocess timed out after 300 seconds")[-8000:]
        row = {
            "name": name,
            "command": shlex.join(argv),
            "exit_code": exit_code,
            "duration_s": max(round(time.monotonic() - started, 6), 0.0001),
            "stdout": stdout,
            "stderr": stderr,
            "passed": exit_code == 0,
        }
        rows.append(row)
        artifact["validation_command_rows"] = list(rows)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        _write_artifact(output_path, artifact)
        _progress(3, "subprocess end", f"{index}/10 {name} exit_code={exit_code}")
    return rows


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    run_commands: bool = True,
) -> dict[str, Any]:
    """Build a positive, disqualified, or prerequisite-blocked result."""

    started = time.monotonic()
    _progress(0, "start", "initialize schema-complete artifact")
    artifact = _base_artifact(run_date)
    _checkpoint(output_path, artifact, started)
    _progress(0, "end", "initial artifact persisted")

    _progress(1, "start", "check active inputs, schemas, lints, and output path")
    preconditions, roadmap, authority = _preconditions(root, output_path)
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    artifact["source_artifact_hashes"] = _source_hashes(root)
    failed = next((row for row in preconditions if not row["available"]), None)
    if failed is not None:
        artifact["status"] = "complete"
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = BLOCKED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
            "passed": False,
        }
        _checkpoint(output_path, artifact, started)
        _progress(1, "end", f"blocked on {failed['check']}")
        _progress(4, "validation start", "validate final blocked artifact")
        validation_errors = validate_artifact(artifact)
        _progress(4, "validation end", f"errors={validation_errors}")
        _progress(5, "write start", "final blocked artifact boundary")
        _checkpoint(output_path, artifact, started)
        _progress(5, "write end", "blocked artifact complete")
        return artifact
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    _checkpoint(output_path, artifact, started)
    _progress(1, "end", "all named inputs are readable")

    _progress(2, "start", "parse independent sources and compare contracts")
    try:
        exclusion = _load_yaml(root / EXCLUSION_PATH)
        contract = evaluate_contract(
            (root / DESIGN_PATH).read_text(encoding="utf-8"),
            roadmap,
            retired_experiment_ids(exclusion),
        )
        contract["authority_consistency_rows"] = _authority_consistency_rows(
            root, authority, roadmap
        )
        if any(not row["passed"] for row in contract["authority_consistency_rows"]):
            contract["passed"] = False
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": "contract_parse",
            "expected_value": "independently_parseable_markdown_and_yaml",
            "observed_value": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
    else:
        for field in SUPPORTING_ROW_FIELDS:
            artifact[field] = contract[field]
        artifact["markdown_task_rows"] = contract["markdown_task_rows"]
        artifact["yaml_task_rows"] = contract["yaml_task_rows"]
        artifact["task_contract_rows"] = contract["task_contract_rows"]
        artifact["rows"] = contract["task_contract_rows"]
        artifact["observed_task_count"] = len(contract["yaml_task_rows"])
        artifact["expected_id_order"] = contract["expected_id_order"]
        artifact["observed_id_order"] = contract["observed_id_order"]
        score = int(contract["passed"])
        artifact["v632_task_contract_conforms_score"] = score
        if score:
            artifact["verdict_class"] = "positive"
            artifact["honest_verdict"] = POSITIVE_VERDICT
            artifact["gate_check_summary"] = {
                "failed_check": None,
                "expected_value": "all_13_task_contracts_conform",
                "observed_value": "all_13_task_contracts_conform",
                "passed": True,
            }
        else:
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = DISQUALIFIED_VERDICT
            artifact["gate_check_summary"] = _failure_summary(contract)
    artifact["status"] = "complete"
    _checkpoint(output_path, artifact, started)
    _progress(2, "end", f"contract verdict is {artifact['verdict_class']}")

    if run_commands:
        _progress(3, "start", "run required validation subprocesses")
        run_validation_commands(root, output_path, artifact)
        _progress(3, "end", "all validation subprocess receipts recorded")

    _progress(4, "validation start", "validate final contract artifact")
    validation_errors = validate_artifact(artifact)
    _progress(4, "validation end", f"errors={validation_errors}")
    _progress(5, "write start", "final contract artifact boundary")
    _checkpoint(output_path, artifact, started)
    _progress(5, "write end", f"{artifact['verdict_class']} artifact complete")
    return artifact


def _score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute the headline score only from stored contract evidence."""

    markdown = artifact.get("markdown_task_rows")
    yaml_rows = artifact.get("yaml_task_rows")
    task_rows = artifact.get("task_contract_rows")
    if not all(isinstance(rows, list) for rows in (markdown, yaml_rows, task_rows)):
        return 0
    if not all(len(rows) == EXPECTED_TASK_COUNT for rows in (markdown, yaml_rows, task_rows)):
        return 0
    markdown_ids = [row.get("id") for row in markdown]
    yaml_ids = [row.get("id") for row in yaml_rows]
    if markdown_ids != yaml_ids:
        return 0
    if any(
        not _valid_task_id(task_id, number)
        for task_id, number in zip(markdown_ids, EXPECTED_NUMBERS, strict=True)
    ):
        return 0
    if not all(row.get("passed") is True for row in task_rows):
        return 0
    for field in SUPPORTING_ROW_FIELDS:
        rows = artifact.get(field)
        if not isinstance(rows, list) or not all(row.get("passed") is True for row in rows):
            return 0
    return 1


def validate_artifact(artifact: object) -> list[str]:
    """Recompute required fields, score, lifecycle state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    principles = artifact["field_principles"]
    if (
        not isinstance(principles, Mapping)
        or any(
            principles.get(field) != principle
            for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
        )
        or any(field not in principles for field in artifact)
    ):
        errors.append("field_principles_invalid")
    if artifact.get("status") != "complete":
        errors.append("status_invalid")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact["execution_venue"] != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    if artifact["expected_task_count"] != EXPECTED_TASK_COUNT:
        errors.append("expected_task_count_invalid")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle_invalid")
    if not re.fullmatch(r"\d{8}", str(artifact["run_date"])):
        errors.append("run_date_invalid")
    if (
        not isinstance(artifact["duration_s"], (int, float))
        or isinstance(artifact["duration_s"], bool)
        or artifact["duration_s"] < 0
    ):
        errors.append("duration_s_invalid")
    if not isinstance(artifact["source_artifact_hashes"], Mapping):
        errors.append("source_artifact_hashes_invalid")

    yaml_rows = artifact.get("yaml_task_rows")
    markdown_rows = artifact.get("markdown_task_rows")
    observed_count = len(yaml_rows) if isinstance(yaml_rows, list) else None
    observed_order = [row.get("id") for row in yaml_rows] if isinstance(yaml_rows, list) else None
    expected_order = list(EXPECTED_ID_ORDER)
    if artifact["observed_task_count"] != observed_count:
        errors.append("observed_task_count_invalid")
    if artifact["observed_id_order"] != observed_order:
        errors.append("observed_id_order_invalid")
    if artifact["expected_id_order"] != expected_order:
        errors.append("expected_id_order_invalid")
    if artifact["rows"] != artifact.get("task_contract_rows"):
        errors.append("rows_not_task_contract_rows")

    preconditions = artifact["preconditions_checked"]
    if not isinstance(preconditions, list) or not all(
        isinstance(row, Mapping) and isinstance(row.get("available"), bool) for row in preconditions
    ):
        errors.append("preconditions_checked_invalid")
        blocked = False
    else:
        blocked = any(row["available"] is False for row in preconditions)
    score = 0 if blocked else _score_from_artifact(artifact)
    if artifact["v632_task_contract_conforms_score"] != score:
        errors.append("v632_task_contract_conforms_score_invalid")
    expected_class = "blocked" if blocked else ("positive" if score else "disqualified")
    expected_verdict = {
        "blocked": BLOCKED_VERDICT,
        "positive": POSITIVE_VERDICT,
        "disqualified": DISQUALIFIED_VERDICT,
    }[expected_class]
    expected_substrate = "blocked_no_run" if blocked else INFERENCE_SUBSTRATE_CLASS
    if artifact["inference_substrate_class"] != expected_substrate:
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
        failed = next(row for row in preconditions if row["available"] is False)
        expected_summary = {
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
            "passed": False,
        }
        if dict(summary) != expected_summary:
            errors.append("gate_check_summary_invalid")
    elif summary.get("passed") is not bool(score) or (
        score
        and dict(summary)
        != {
            "failed_check": None,
            "expected_value": "all_13_task_contracts_conform",
            "observed_value": "all_13_task_contracts_conform",
            "passed": True,
        }
    ):
        errors.append("gate_check_summary_invalid")
    commands = artifact.get("validation_command_rows")
    if not isinstance(commands, list) or (
        commands
        and (
            [row.get("name") for row in commands] != list(VALIDATION_COMMAND_NAMES[: len(commands)])
            or any(not isinstance(row.get("exit_code"), int) for row in commands)
        )
    ):
        errors.append("validation_command_rows_invalid")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the advisory preflight and fail only for an invalid artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--skip-validation-commands", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    artifact = build_artifact(
        REPO_ROOT,
        args.date,
        output_path=output,
        run_commands=not args.skip_validation_commands,
    )
    errors = validate_artifact(artifact)
    print(output, flush=True)
    if errors:
        print("INVALID: " + ", ".join(errors), flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
