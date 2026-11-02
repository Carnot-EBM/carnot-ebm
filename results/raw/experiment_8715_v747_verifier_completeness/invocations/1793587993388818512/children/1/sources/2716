"""Compare the V629 Markdown contract with the active roadmap YAML.

The two parsers receive different inputs and build different row objects. This
keeps an omission visible in its source instead of copying it into both views.
The check is advisory. It never edits or activates either roadmap.
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
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7151_v629_contract_preflight.json")
MODULE_PATH = Path("python/carnot/experiment_7151_v629_contract_preflight.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7151_v629_contract_preflight.py")
TEST_PATH = Path("tests/python/test_experiment_7151_v629_contract_preflight.py")

MILESTONE = "2026.09.629"
EXPECTED_TASK_IDS = (
    "exp7151-v629-active-contract-preflight",
    "exp7152-v629-source-and-cache-delta",
    "exp7153-grounding-runtime-postfix-qualification",
    "exp7154-qwen-dual-side-grounding-pilot",
    "exp7155-gemma-dual-side-grounding-replication",
    "exp7156-dual-side-grounding-causal-audit",
    "exp7157-flowbalance-transaction-preflight",
    "exp7158-procedure-family-memory-csl",
    "exp7159-memory-drift-poison-cold-audit",
    "exp7160-state-localized-arc-invariance",
    "exp7161-rust-multiscale-exact-parity",
    "exp7162-rust-multiscale-orchestration-profile",
    "exp7163-three-board-continuity",
    "exp7164-v629-capstone",
)
EXPECTED_TASK_COUNT = 14
EXPECTED_GATES = {
    7154: (EXPECTED_TASK_IDS[2], "grounding_runtime_ready_score"),
    7155: (EXPECTED_TASK_IDS[3], "qwen_dual_side_pilot_complete_score"),
    7156: (EXPECTED_TASK_IDS[4], "gemma_dual_side_replication_complete_score"),
    7158: (EXPECTED_TASK_IDS[6], "flowbalance_transaction_ready_score"),
    7159: (EXPECTED_TASK_IDS[7], "procedure_family_csl_complete_score"),
    7162: (EXPECTED_TASK_IDS[10], "rust_multiscale_exact_parity_score"),
}
EXPECTED_PRIOR_IDS = {
    7151: ("exp7148-v628-contract-preflight",),
    7152: (
        "exp7149-v628-source-and-cache-delta",
        "exp6461-v556-sota-source-and-benchmark-delta",
    ),
    7153: (
        "exp7150-source-grounding-preflight-repair",
        "exp7139-three-family-symbolic-grounding-ab",
    ),
    7154: ("exp7139-three-family-symbolic-grounding-ab",),
    7155: ("exp7139-three-family-symbolic-grounding-ab",),
    7156: (),
    7157: ("exp7142-flowbalance-external-memory-csl",),
    7158: (),
    7159: (),
    7160: (),
    7161: ("exp7145-rust-multiscale-sampler-parity",),
    7162: ("exp7145-rust-multiscale-sampler-parity",),
    7163: ("exp7146-gatemate-changed-state-continuity",),
    7164: (),
}
MANDATED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
EXPECTED_MODEL_TASKS = {
    7153: (MANDATED_MODELS[0],),
    7154: (MANDATED_MODELS[0],),
    7155: (MANDATED_MODELS[1], MANDATED_MODELS[2]),
    7158: (MANDATED_MODELS[0],),
    7159: (MANDATED_MODELS[0],),
    7160: (MANDATED_MODELS[0],),
}
EXPECTED_SUBSTRATE_CLASSES = {
    7151: "aggregation",
    7152: "aggregation",
    7153: "model_full_generation",
    7154: "model_full_generation",
    7155: "model_full_generation",
    7156: "aggregation",
    7157: "no_model_load",
    7158: "model_full_generation",
    7159: "model_full_generation",
    7160: "model_full_generation",
    7161: "cpu_exact_solver_or_simulator",
    7162: "cpu_exact_solver_or_simulator",
    7163: "hardware_smoke",
    7164: "aggregation",
}
FOCUSED_TESTS = {
    7151: "tests/python/test_experiment_7151_v629_contract_preflight.py",
    7152: "tests/python/test_experiment_7152_v629_source_delta.py",
    7153: "tests/python/test_experiment_7153_v629_grounding_runtime.py",
    7154: "tests/python/test_experiment_7154_v629_qwen_dual_side_grounding.py",
    7155: "tests/python/test_experiment_7155_v629_gemma_dual_side_grounding.py",
    7156: "tests/python/test_experiment_7156_v629_grounding_causal_audit.py",
    7157: "tests/python/test_experiment_7157_v629_flowbalance_transaction.py",
    7158: "tests/python/test_experiment_7158_v629_procedure_family_csl.py",
    7159: "tests/python/test_experiment_7159_v629_memory_drift_poison_audit.py",
    7160: "tests/python/test_experiment_7160_v629_arc_state_invariance.py",
    7161: "tests/python/test_experiment_7161_v629_rust_multiscale_parity.py",
    7162: "tests/python/test_experiment_7162_v629_rust_orchestration_profile.py",
    7163: "tests/python/test_experiment_7163_v629_three_board_continuity.py",
    7164: "tests/python/test_experiment_7164_v629_capstone.py",
}
RUN_LINES = {
    number: (
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/experiment_{number}_v629_{suffix}.py --date {{date}}"
    )
    for number, suffix in {
        7151: "contract_preflight",
        7152: "source_delta",
        7153: "grounding_runtime",
        7154: "qwen_dual_side_grounding",
        7155: "gemma_dual_side_grounding",
        7156: "grounding_causal_audit",
        7157: "flowbalance_transaction",
        7158: "procedure_family_csl",
        7159: "memory_drift_poison_audit",
        7160: "arc_state_invariance",
        7161: "rust_multiscale_parity",
        7162: "rust_orchestration_profile",
        7163: "three_board_continuity",
        7164: "capstone",
    }.items()
}
EXPECTED_OVERRIDE_TASKS = frozenset({7163, 7164})
PROMPT_FINAL_LINE = "Do NOT push. Do NOT modify scripts/research_conductor.py."
VERDICT_CLASSES = (
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
)
INFERENCE_SUBSTRATE = (
    "aggregation_from_active_contract: independent Markdown and active YAML parses"
)
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
RANDOM_SEED = 715120260909
BLOCKED_VERDICT = "blocked_v629_contract_preflight_prerequisite_missing"
POSITIVE_VERDICT = "complete_positive_v629_task_contract_conforms"
DISQUALIFIED_VERDICT = "complete_disqualified_v629_markdown_yaml_contract_mismatch"

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
        "deliverable_parity_rows",
        "gate_contract_rows",
        "gate_producer_rows",
        "prior_failure_rows",
        "operator_override_rows",
        "model_compliance_rows",
        "agent_routing_rows",
        "focused_test_rows",
        "prompt_tail_rows",
        "progress_contract_rows",
        "expected_task_count",
        "observed_task_count",
        "expected_id_order",
        "observed_id_order",
        "v629_task_contract_conforms_score",
        "random_seed",
        "reproducibility_checksum",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
)
SUPPORTING_ROW_FIELDS = (
    "deliverable_parity_rows",
    "gate_contract_rows",
    "gate_producer_rows",
    "prior_failure_rows",
    "operator_override_rows",
    "model_compliance_rows",
    "agent_routing_rows",
    "focused_test_rows",
    "prompt_tail_rows",
    "progress_contract_rows",
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


def _progress(phase: int, state: str, detail: str) -> None:
    """Flush phase state so the conductor can distinguish work from a stall."""

    print(f"[exp7151] phase {phase} {state}: {detail}", flush=True)


def task_number(task_id: object) -> int | None:
    """Return the experiment number only when the full prefix is valid."""

    match = re.fullmatch(r"exp(\d+)(?:-[A-Za-z0-9_-]+)?", str(task_id), re.I)
    return int(match.group(1)) if match else None


def _normal_exp_id(value: object) -> str | None:
    """Normalize an experiment reference for exclusion checks."""

    candidate = f"exp{value}" if isinstance(value, int) else str(value)
    match = re.match(r"exp(\d+)(?:-|$)", candidate, re.I)
    return f"exp{int(match.group(1))}" if match else None


def _parse_scalar(text: str) -> Any:
    """Parse a gate scalar and reject a hidden list or mapping."""

    value = yaml.safe_load(text)
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("structured gate value must be scalar")
    return value


def _parse_markdown_gate(
    cell: str, full_id_by_number: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Parse gates from Markdown without consulting the YAML source."""

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
        gates.append(
            {
                "upstream": full_id_by_number.get(number, match.group(1)),
                "artifact_field": match.group(2),
                "op": match.group(3),
                "value": _parse_scalar(match.group(4).strip().strip("`")),
            }
        )
    return gates


def parse_markdown_contract(text: str) -> dict[str, Any]:
    """Build task rows from the Markdown milestone and exact table only."""

    milestone_match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\s]+)`?", text)
    if milestone_match is None:
        raise ValueError("Markdown milestone is missing")
    section = re.search(
        r"^## Exact task contract\s*$([\s\S]*?)(?=^## |\Z)", text, re.I | re.M
    )
    if section is None:
        raise ValueError("Markdown task contract section is missing")
    tasks: list[dict[str, Any]] = []
    gate_cells: list[str] = []
    for line in section.group(1).splitlines():
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
        task_id = cells[1].strip("`")
        number = task_number(task_id)
        if number is None or not cells[2] or not cells[3].strip("`"):
            raise ValueError(f"malformed Markdown task row: {line}")
        tasks.append(
            {
                "order": int(cells[0]),
                "id": task_id,
                "number": number,
                "title": cells[2],
                "deliverable": cells[3].strip("`"),
                "milestone": milestone_match.group(1),
                "gates": [],
            }
        )
        gate_cells.append(cells[4])
    if not tasks:
        raise ValueError("Markdown task table is missing")
    full_ids = {task["number"]: task["id"] for task in tasks}
    for task, gate_cell in zip(tasks, gate_cells, strict=True):
        task["gates"] = _parse_markdown_gate(gate_cell, full_ids)
    return {"milestone": milestone_match.group(1), "tasks": tasks}


def _required_block(prompt: str) -> str:
    """Isolate declarations so normal prompt prose cannot satisfy them."""

    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].split("Run command:", 1)[0]


def parse_yaml_contract(document: object) -> dict[str, Any]:
    """Build YAML task rows without receiving a Markdown parser result."""

    if not isinstance(document, Mapping):
        raise ValueError("YAML roadmap mapping is required")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("YAML roadmap tasks list is required")
    tasks: list[dict[str, Any]] = []
    for order, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"YAML task {order} must be a mapping")
        task_id = raw_task.get("id")
        prompt = raw_task.get("prompt")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            raise ValueError(f"YAML task {order} has malformed id or prompt")
        raw_gates = raw_task.get("gated_on") or []
        raw_priors = raw_task.get("prior_failures") or []
        raw_requires = raw_task.get("requires") or []
        if not all(
            isinstance(value, list)
            for value in (raw_gates, raw_priors, raw_requires)
        ):
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
        tasks.append(
            {
                "order": order,
                "id": task_id,
                "number": task_number(task_id),
                "title": raw_task.get("title"),
                "deliverable": raw_task.get("deliverable"),
                "milestone": raw_task.get("milestone", document.get("milestone")),
                "track": raw_task.get("track"),
                "priority": raw_task.get("priority"),
                "agent_type": raw_task.get("agent_type"),
                "model": raw_task.get("model"),
                "requires_gpu": raw_task.get("requires_gpu"),
                "max_turns": raw_task.get("max_turns"),
                "estimated_wall_time_min": raw_task.get("estimated_wall_time_min"),
                "per_unit_rows": raw_task.get("per_unit_rows"),
                "gates": gates,
                "prior_failures": list(raw_priors),
                "requires": list(raw_requires),
                "operator_override": raw_task.get("operator_override"),
                "prompt": prompt,
                "required_block": _required_block(prompt),
            }
        )
    return {"milestone": document.get("milestone"), "tasks": tasks}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a non-empty YAML mapping and expose one stable error type."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML at {path}: {exc}") from exc
    if not isinstance(document, dict) or not document:
        raise ValueError(f"YAML mapping required at {path}")
    return document


def retired_experiment_ids(document: object) -> set[str]:
    """Read retired experiment numbers without treating prose as identity."""

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
    """Make all ordered gate parts explicit for comparison."""

    return [
        (
            gate.get("upstream"),
            gate.get("artifact_field"),
            gate.get("op"),
            gate.get("value"),
        )
        for gate in gates
    ]


def _field_declared(block: str, field: str) -> bool:
    """Accept a bare field and reject a nested look-alike."""

    return (
        re.search(
            rf"(?<![A-Za-z0-9_.]){re.escape(field)}(?![A-Za-z0-9_]|\.[A-Za-z_])",
            block,
        )
        is not None
    )


def _planned_substrate_class(block: str) -> str | None:
    """Read the planned class before its blocked no-run alternative."""

    match = re.search(r"inference_substrate_class\s*\(\s*([A-Za-z_]+)", block)
    return match.group(1) if match else None


def _closed_verdict_enum(block: str) -> bool:
    """Require the six verdict classes in their fixed order."""

    normalized = re.sub(r"\s+", " ", block)
    values = r"\s*\|\s*".join(map(re.escape, VERDICT_CLASSES))
    return re.search(rf"verdict_class\s*\(\s*{values}\s*\)", normalized) is not None


def _public_task_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Keep source evidence small while retaining contract data."""

    return {
        key: task[key]
        for key in (
            "order",
            "id",
            "number",
            "title",
            "deliverable",
            "milestone",
            "gates",
        )
    }


def _route_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Check the route and the shared execution metadata."""

    number = task["number"]
    observed = (task.get("agent_type"), task.get("model"))
    agent_type, model = observed
    route_allowed = bool(
        (agent_type is None and model in {None, "opus"})
        or (agent_type == "codex" and model == "gpt-5.6-sol")
    )
    metadata = {
        "track": isinstance(task.get("track"), str) and bool(task["track"].strip()),
        "priority": task.get("priority") in {"critical", "high", "medium"},
        "requires_gpu": isinstance(task.get("requires_gpu"), bool),
        "max_turns": isinstance(task.get("max_turns"), int)
        and task["max_turns"] > 0,
        "estimated_wall_time_min": isinstance(
            task.get("estimated_wall_time_min"), int
        )
        and task["estimated_wall_time_min"] > 0,
    }
    return {
        "order": task["order"],
        "number": number,
        "task_id": task["id"],
        "expected_route": "default_none_or_opus_or_codex_gpt_5_6_sol",
        "observed_route": list(observed),
        "metadata_checks": metadata,
        "passed": route_allowed and all(metadata.values()),
    }


def _prior_rows(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check all four fields and the declared predecessor identity."""

    expected_ids = EXPECTED_PRIOR_IDS.get(task["number"], ())
    priors = task["prior_failures"]
    rows: list[dict[str, Any]] = []
    for index in range(max(len(expected_ids), len(priors))):
        prior = priors[index] if index < len(priors) else {}
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
        expected_id = expected_ids[index] if index < len(expected_ids) else None
        rows.append(
            {
                "order": task["order"],
                "number": task["number"],
                "task_id": task["id"],
                "prior_order": index + 1,
                "expected_experiment_id": expected_id,
                "observed_experiment_id": values.get("experiment_id"),
                "field_checks": checks,
                "passed": values.get("experiment_id") == expected_id
                and all(checks.values()),
            }
        )
    if not rows:
        rows.append(
            {
                "order": task["order"],
                "number": task["number"],
                "task_id": task["id"],
                "prior_order": None,
                "expected_experiment_id": None,
                "observed_experiment_id": None,
                "field_checks": {"no_prior_expected_or_declared": True},
                "passed": True,
            }
        )
    return rows


def _override_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Require an auditable standing override only on the two continuations."""

    number = task["number"]
    value = task.get("operator_override")
    required = number in EXPECTED_OVERRIDE_TASKS
    present = isinstance(value, str) and bool(value.strip())
    text = value.strip() if present else ""
    checks = {
        "one_line": present and "\n" not in text,
        "minimum_length": len(text) >= 10,
        "standing_directive": "2026-05-29 operator directive (standing):" in text,
        "continuation_class": bool(re.search(r"standing\):\s*[^;—-]+[—-]", text)),
        "false_match_ids": "false-positive scope-match vs" in text,
        "forward_rationale": bool(re.search(r";\s*\S.+[.]$", text)),
    }
    passed = all(checks.values()) if required else not present
    return {
        "order": task["order"],
        "number": number,
        "task_id": task["id"],
        "required": required,
        "observed": value,
        "checks": checks if required or present else {},
        "passed": passed,
    }


def _model_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact GGUF declarations only on model tasks."""

    number = task["number"]
    expected_models = EXPECTED_MODEL_TASKS.get(number, ())
    block = task["required_block"]
    prompt = task["prompt"]
    declared = _field_declared(block, "MODEL_SPECS")
    observed_models = [model for model in MANDATED_MODELS if model in prompt]
    slow_action_named = bool(re.search(r"\b(?:model load|generation)\b", prompt, re.I))
    if expected_models:
        passed = bool(
            declared
            and task.get("requires_gpu") is True
            and tuple(observed_models) == expected_models
            and slow_action_named
        )
    else:
        passed = not declared and task.get("requires_gpu") is False
    return {
        "order": task["order"],
        "number": number,
        "task_id": task["id"],
        "expected_models": list(expected_models),
        "observed_models": observed_models,
        "model_specs_declared": declared,
        "live_execution_required": slow_action_named if expected_models else False,
        "passed": passed,
    }


def _progress_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Check the measured anti-stall contract without accepting vague prose."""

    prompt = re.sub(r"\s+", " ", task["prompt"].lower())
    checks = {
        "phase_boundaries": "every numbered phase boundary, print a flushed progress line"
        in prompt,
        "slow_call_start_end": all(
            phrase in prompt
            for phrase in (
                "flushed start",
                "end lines immediately before and after",
                "model load",
                "generation",
                "benchmark",
                "subprocess",
            )
        ),
        "long_loop_heartbeat": "at least every 300 seconds" in prompt,
        "stdout_gap": "every stdout gap below 600 seconds" in prompt,
    }
    return {
        "order": task["order"],
        "number": task["number"],
        "task_id": task["id"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _discipline_checks(task: Mapping[str, Any], retired_ids: set[str]) -> dict[str, bool]:
    """Check declarations that every V629 task must carry."""

    number = task["number"]
    block = task["required_block"]
    normalized = re.sub(r"\s+", " ", block.lower())
    gate_free = not task["gates"]
    checks = {
        "required_block": bool(block),
        "generic_fields": all(
            _field_declared(block, field) for field in TASK_REQUIRED_FIELDS
        ),
        "field_principles": "field_principles for every field below" in normalized,
        "blocked_diagnostics": all(
            phrase in normalized
            for phrase in ("failed check", "expected value", "observed value")
        ),
        "closed_verdict_enum": _closed_verdict_enum(block),
        "planned_substrate_class": _planned_substrate_class(block)
        == EXPECTED_SUBSTRATE_CLASSES.get(number),
        "blocked_no_run": _field_declared(block, "blocked_no_run"),
        "per_unit_rows": task.get("per_unit_rows") is True,
        "not_retired": _normal_exp_id(task["id"]) not in retired_ids,
        "advisory_ungated": gate_free if number == 7151 else True,
        "capstone_ungated": gate_free if number == 7164 else True,
    }
    return checks


def _missing_row(order: int, task_id: str | None = None) -> dict[str, Any]:
    """Keep evidence aligned when one YAML row is absent."""

    return {
        "order": order,
        "number": task_number(task_id),
        "task_id": task_id,
        "passed": False,
    }


def evaluate_contract(
    markdown_text: str, yaml_document: object, retired_ids: set[str]
) -> dict[str, Any]:
    """Compare V629 rows and retain each policy decision as evidence."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    normalized_retired = {
        normal
        for value in retired_ids
        if (normal := _normal_exp_id(value)) is not None
    }
    number_counts = Counter(task["number"] for task in yaml_tasks)
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))

    deliverable_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    focused_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    discipline_by_order: dict[int, dict[str, bool]] = {}

    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        source = markdown_task or yaml_task or {}
        task_id = source.get("id")
        number = source.get("number")
        same_identity = bool(
            markdown_task
            and yaml_task
            and markdown_task["id"] == yaml_task["id"]
            and markdown_task["order"] == yaml_task["order"] == order
        )
        deliverable_rows.append(
            {
                "order": order,
                "number": number,
                "task_id": task_id,
                "expected": markdown_task.get("deliverable") if markdown_task else None,
                "observed": yaml_task.get("deliverable") if yaml_task else None,
                "title_expected": markdown_task.get("title") if markdown_task else None,
                "title_observed": yaml_task.get("title") if yaml_task else None,
                "passed": bool(
                    same_identity
                    and markdown_task["title"] == yaml_task["title"]
                    and markdown_task["deliverable"] == yaml_task["deliverable"]
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
            missing = _missing_row(order, task_id)
            prior_rows.append({**missing, "prior_order": None})
            override_rows.append({**missing, "required": number in EXPECTED_OVERRIDE_TASKS})
            model_rows.append({**missing, "expected_models": []})
            route_rows.append({**missing, "expected_route": None})
            focused_rows.append({**missing, "expected": FOCUSED_TESTS.get(number)})
            tail_rows.append({**missing, "expected": RUN_LINES.get(number)})
            progress_rows.append({**missing, "checks": {}})
            discipline_by_order[order] = {"yaml_present": False}
            continue
        prior_rows.extend(_prior_rows(yaml_task))
        override_rows.append(_override_row(yaml_task))
        model_rows.append(_model_row(yaml_task))
        route_rows.append(_route_row(yaml_task))
        focused = FOCUSED_TESTS.get(yaml_task["number"])
        focused_rows.append(
            {
                "order": order,
                "number": yaml_task["number"],
                "task_id": yaml_task["id"],
                "expected": focused,
                "observed": focused if focused and focused in yaml_task["prompt"] else None,
                "passed": bool(focused and focused in yaml_task["prompt"]),
            }
        )
        observed_tail = yaml_task["prompt"].rstrip().splitlines()[-2:]
        expected_tail = [RUN_LINES.get(yaml_task["number"]), PROMPT_FINAL_LINE]
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
        progress_rows.append(_progress_row(yaml_task))
        discipline_by_order[order] = _discipline_checks(
            yaml_task, normalized_retired
        )

    yaml_by_id = {task["id"]: task for task in yaml_tasks}
    gate_producer_rows: list[dict[str, Any]] = []
    for consumer in yaml_tasks:
        for gate in consumer["gates"]:
            producer = yaml_by_id.get(gate.get("upstream"))
            field = gate.get("artifact_field")
            bare = isinstance(field, str) and bool(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field)
            )
            expected_gate = EXPECTED_GATES.get(consumer["number"])
            passed = bool(
                producer
                and producer["order"] < consumer["order"]
                and producer["milestone"] == consumer["milestone"] == MILESTONE
                and bare
                and _field_declared(producer["required_block"], field)
                and expected_gate == (producer["id"], field)
                and task_number(producer["id"]) != 7151
            )
            gate_producer_rows.append(
                {
                    "order": consumer["order"],
                    "number": consumer["number"],
                    "consumer": consumer["id"],
                    "upstream": gate.get("upstream"),
                    "artifact_field": field,
                    "producer_present": producer is not None,
                    "producer_precedes_consumer": bool(
                        producer and producer["order"] < consumer["order"]
                    ),
                    "bare_field": bare,
                    "field_declared": bool(
                        producer
                        and bare
                        and _field_declared(producer["required_block"], field)
                    ),
                    "passed": passed,
                }
            )

    supporting = (
        deliverable_rows,
        gate_rows,
        prior_rows,
        override_rows,
        model_rows,
        route_rows,
        focused_rows,
        tail_rows,
        progress_rows,
        gate_producer_rows,
    )
    checks_by_order: dict[int, list[bool]] = {}
    for evidence_rows in supporting:
        for row in evidence_rows:
            checks_by_order.setdefault(row["order"], []).append(
                row["passed"] is True
            )

    task_rows: list[dict[str, Any]] = []
    for index in range(width):
        order = index + 1
        markdown_task = markdown_tasks[index] if index < len(markdown_tasks) else None
        yaml_task = yaml_tasks[index] if index < len(yaml_tasks) else None
        expected_id = EXPECTED_TASK_IDS[index] if index < EXPECTED_TASK_COUNT else None
        discipline = discipline_by_order.get(order, {"yaml_present": False})
        identity_ok = bool(
            markdown_task
            and yaml_task
            and markdown_task["order"] == yaml_task["order"] == order
            and markdown_task["id"] == yaml_task["id"] == expected_id
            and markdown_task["milestone"] == yaml_task["milestone"] == MILESTONE
            and number_counts[yaml_task["number"]] == 1
        )
        task_rows.append(
            {
                "order": order,
                "expected_id": expected_id,
                "markdown_id": markdown_task.get("id") if markdown_task else None,
                "observed_id": yaml_task.get("id") if yaml_task else None,
                "expected_milestone": MILESTONE,
                "observed_milestone": yaml_task.get("milestone")
                if yaml_task
                else None,
                "discipline_checks": discipline,
                "passed": bool(
                    identity_ok
                    and all(discipline.values())
                    and all(checks_by_order.get(order, [False]))
                ),
            }
        )

    gated_numbers = {task["number"] for task in yaml_tasks if task["gates"]}
    global_checks = (
        markdown["milestone"] == MILESTONE,
        roadmap["milestone"] == MILESTONE,
        len(markdown_tasks) == EXPECTED_TASK_COUNT,
        len(yaml_tasks) == EXPECTED_TASK_COUNT,
        tuple(task["id"] for task in markdown_tasks) == EXPECTED_TASK_IDS,
        tuple(task["id"] for task in yaml_tasks) == EXPECTED_TASK_IDS,
        len(number_counts) == EXPECTED_TASK_COUNT,
        gated_numbers == set(EXPECTED_GATES),
        len(gate_producer_rows) == len(EXPECTED_GATES),
    )
    passed = bool(
        all(global_checks)
        and len(task_rows) == EXPECTED_TASK_COUNT
        and all(row["passed"] for row in task_rows)
    )
    return {
        "passed": passed,
        "markdown_task_rows": [_public_task_row(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task_row(task) for task in yaml_tasks],
        "task_contract_rows": task_rows,
        "deliverable_parity_rows": deliverable_rows,
        "gate_contract_rows": gate_rows,
        "gate_producer_rows": gate_producer_rows,
        "prior_failure_rows": prior_rows,
        "operator_override_rows": override_rows,
        "model_compliance_rows": model_rows,
        "agent_routing_rows": route_rows,
        "focused_test_rows": focused_rows,
        "prompt_tail_rows": tail_rows,
        "progress_contract_rows": progress_rows,
        "markdown_task_count": len(markdown_tasks),
        "observed_id_order": [task["id"] for task in yaml_tasks],
    }


def _sha256_path(path: Path) -> str:
    """Hash exact bytes so a later source change remains visible."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str | None]:
    """Hash contract sources and every named validation implementation."""

    paths = (
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        ACTIVE_ROADMAP_PATH,
        DESIGN_PATH,
        Path("results/experiment_7148_v628_contract_preflight.json"),
        Path("scripts/roadmap_schema.py"),
        Path("scripts/validate_prior_failures.py"),
        Path("scripts/audit_roadmap_gates.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        Path("scripts/harness_fit_lint.py"),
        Path("scripts/adversarial_verify.py"),
        SPEC_PATH,
        EXCLUSION_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        str(relative): _sha256_path(root / relative)
        if (root / relative).is_file()
        else None
        for relative in paths
    }


def _field_principles() -> dict[str, str]:
    """Give every result field a short reason for its presence."""

    fields = set(REQUIRED_ARTIFACT_FIELDS) | {"schema", "experiment_id", "status"}
    principles = {
        field: (
            f"The {field} evidence lets a reviewer reproduce or falsify "
            "the V629 contract result."
        )
        for field in sorted(fields)
    }
    principles["field_principles"] = "Each entry explains why its evidence is needed."
    principles["preconditions_checked"] = (
        "Prerequisite receipts separate a blocked run from a contract mismatch."
    )
    principles["rows"] = "Per-task rows stop one mismatch from hiding in a total."
    principles["source_artifact_hashes"] = "Content hashes expose later source drift."
    principles["v629_task_contract_conforms_score"] = (
        "The score is one only when all 14 contracts and supporting checks pass."
    )
    principles["reproducibility_checksum"] = "The checksum exposes artifact changes."
    principles["honest_verdict"] = "The verdict separates missing input from mismatch."
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact fields except the checksum storage field."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key != "reproducibility_checksum"
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _empty_contract() -> dict[str, list[Any]]:
    """Keep initial and blocked artifacts schema-complete."""

    fields = SUPPORTING_ROW_FIELDS + (
        "markdown_task_rows",
        "yaml_task_rows",
        "task_contract_rows",
    )
    return {field: [] for field in fields}


def _base_artifact(run_date: str) -> dict[str, Any]:
    """Create every required field before the first prerequisite check."""

    return {
        "schema": "carnot.exp7151.v629_contract_preflight.v1",
        "experiment_id": "exp7151-v629-active-contract-preflight",
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
        **_empty_contract(),
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_TASK_IDS),
        "observed_id_order": [],
        "v629_task_contract_conforms_score": 0,
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
        "honest_verdict": "partial_running_v629_contract_preflight",
    }


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write stable JSON so an interrupted phase leaves useful state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _checkpoint(
    output_path: Path, artifact: dict[str, Any], started: float, observed: str
) -> None:
    """Persist a running phase with a current duration and checksum."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.0001)
    if artifact["verdict_class"] == "partial":
        artifact["gate_check_summary"]["observed_value"] = observed
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_artifact(output_path, artifact)


def _artifact_path_writable(path: Path) -> tuple[bool, str]:
    """Probe the destination directory after the initial artifact exists."""

    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp7151-write-"):
            pass
        return True, "temporary_write_succeeded"
    except OSError as exc:  # pragma: no cover - operating system dependent
        return False, f"{type(exc).__name__}: {exc}"


def _preconditions(
    root: Path, output_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Read all prerequisites only after the initial artifact is durable."""

    rows: list[dict[str, Any]] = []
    roadmap: dict[str, Any] | None = None
    exclusion: dict[str, Any] | None = None

    def add(
        check: str, path: Path, expected: str, available: bool, observed: str
    ) -> None:
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
            "v629_markdown_readable",
            design_path,
            "readable_nonempty_source",
            available,
            "readable_nonempty_source" if available else "empty_source",
        )
    except OSError as exc:
        add(
            "v629_markdown_readable",
            design_path,
            "readable_nonempty_source",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    for check, relative, target in (
        ("v629_active_yaml_readable", ACTIVE_ROADMAP_PATH, "roadmap"),
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
        except ValueError as exc:
            add(
                check,
                path,
                "readable_nonempty_yaml_mapping",
                False,
                f"{type(exc).__name__}: {exc}",
            )
    writable, observed = _artifact_path_writable(output_path)
    add(
        "artifact_path_writable",
        output_path,
        "temporary_write_succeeds",
        writable,
        observed,
    )
    return rows, roadmap, exclusion


def _finish(
    output_path: Path, artifact: dict[str, Any], started: float
) -> dict[str, Any]:
    """Freeze terminal status, duration, checksum, and file content."""

    artifact["status"] = "complete"
    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_artifact(output_path, artifact)
    return artifact


def _failure_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Name the first broad mismatch while row evidence keeps local causes."""

    if contract["markdown_task_count"] != EXPECTED_TASK_COUNT:
        check = "markdown_task_count"
        observed = contract["markdown_task_count"]
        expected: Any = EXPECTED_TASK_COUNT
    elif len(contract["yaml_task_rows"]) != EXPECTED_TASK_COUNT:
        check = "yaml_task_count"
        observed = len(contract["yaml_task_rows"])
        expected = EXPECTED_TASK_COUNT
    else:
        first = next(row for row in contract["task_contract_rows"] if not row["passed"])
        check = f"task_contract_order_{first['order']}"
        observed = False
        expected = True
    return {
        "failed_check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def build_artifact(root: Path, run_date: str, *, output_path: Path) -> dict[str, Any]:
    """Build a positive, disqualified, or prerequisite-blocked artifact."""

    started = time.monotonic()
    _progress(0, "start", "initialize schema-complete running artifact")
    artifact = _base_artifact(run_date)
    _checkpoint(output_path, artifact, started, "artifact_initialized")
    _progress(0, "end", "running artifact persisted")

    _progress(1, "start", "check local prerequisites")
    preconditions, roadmap, exclusion = _preconditions(root, output_path)
    artifact["preconditions_checked"] = preconditions
    artifact["source_artifact_hashes"] = _source_hashes(root)
    failed = next((row for row in preconditions if not row["available"]), None)
    if failed is not None:
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = BLOCKED_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": failed["check"],
            "expected_value": failed["expected_value"],
            "observed_value": failed["observed_value"],
            "passed": False,
        }
        _checkpoint(output_path, artifact, started, "precondition_failed")
        _progress(1, "end", f"blocked on {failed['check']}")
        _progress(3, "start", "finalize blocked artifact")
        result = _finish(output_path, artifact, started)
        _progress(3, "end", "blocked artifact complete")
        return result
    artifact["inference_substrate_class"] = INFERENCE_SUBSTRATE_CLASS
    artifact["gate_check_summary"] = {
        "failed_check": None,
        "expected_value": "all_14_task_contracts_conform",
        "observed_value": "preconditions_complete",
        "passed": False,
    }
    _checkpoint(output_path, artifact, started, "preconditions_complete")
    _progress(1, "end", "all local prerequisites are readable")

    _progress(2, "start", "parse independent sources and compare rows")
    try:
        contract = evaluate_contract(
            (root / DESIGN_PATH).read_text(encoding="utf-8"),
            roadmap,
            retired_experiment_ids(exclusion),
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
        _checkpoint(output_path, artifact, started, "contract_parse_failed")
        _progress(2, "end", "contract parse disqualified")
        _progress(3, "start", "finalize disqualified artifact")
        result = _finish(output_path, artifact, started)
        _progress(3, "end", "disqualified artifact complete")
        return result
    for field in SUPPORTING_ROW_FIELDS:
        artifact[field] = contract[field]
    artifact["markdown_task_rows"] = contract["markdown_task_rows"]
    artifact["yaml_task_rows"] = contract["yaml_task_rows"]
    artifact["task_contract_rows"] = contract["task_contract_rows"]
    artifact["rows"] = contract["task_contract_rows"]
    artifact["observed_task_count"] = len(contract["yaml_task_rows"])
    artifact["observed_id_order"] = contract["observed_id_order"]
    score = int(contract["passed"])
    artifact["v629_task_contract_conforms_score"] = score
    if score == 1:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = POSITIVE_VERDICT
        artifact["gate_check_summary"] = {
            "failed_check": None,
            "expected_value": "all_14_task_contracts_conform",
            "observed_value": "all_14_task_contracts_conform",
            "passed": True,
        }
    else:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = _failure_summary(contract)
    _checkpoint(output_path, artifact, started, "contract_evaluated")
    _progress(2, "end", f"contract score is {score}")

    _progress(3, "start", "finalize terminal artifact")
    result = _finish(output_path, artifact, started)
    _progress(3, "end", f"{result['verdict_class']} artifact complete")
    return result


def _contract_score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Recompute the headline score from stored evidence rows."""

    markdown_rows = artifact.get("markdown_task_rows")
    yaml_rows = artifact.get("yaml_task_rows")
    task_rows = artifact.get("task_contract_rows")
    if not all(
        isinstance(rows, list) for rows in (markdown_rows, yaml_rows, task_rows)
    ):
        return 0
    if not all(
        len(rows) == EXPECTED_TASK_COUNT
        for rows in (markdown_rows, yaml_rows, task_rows)
    ):
        return 0
    if [row.get("id") for row in markdown_rows] != list(EXPECTED_TASK_IDS):
        return 0
    if [row.get("id") for row in yaml_rows] != list(EXPECTED_TASK_IDS):
        return 0
    if not all(row.get("passed") is True for row in task_rows):
        return 0
    for field in SUPPORTING_ROW_FIELDS:
        rows = artifact.get(field)
        if (
            not isinstance(rows, list)
            or not rows
            or not all(row.get("passed") is True for row in rows)
        ):
            return 0
    return 1


def validate_artifact(artifact: object) -> list[str]:
    """Recompute required fields, score, class, verdict, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(
        field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    )
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), str) or not principles[field].strip()
        for field in REQUIRED_ARTIFACT_FIELDS
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
    if artifact["expected_id_order"] != list(EXPECTED_TASK_IDS):
        errors.append("expected_id_order_invalid")
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
    observed_order = (
        [row.get("id") for row in yaml_rows] if isinstance(yaml_rows, list) else None
    )
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
    if artifact["v629_task_contract_conforms_score"] != score:
        errors.append("v629_task_contract_conforms_score_invalid")
    expected_class = "blocked" if blocked else ("positive" if score == 1 else "disqualified")
    expected_verdict = {
        "blocked": BLOCKED_VERDICT,
        "positive": POSITIVE_VERDICT,
        "disqualified": DISQUALIFIED_VERDICT,
    }[expected_class]
    expected_substrate_class = (
        "blocked_no_run" if blocked else INFERENCE_SUBSTRATE_CLASS
    )
    if artifact["inference_substrate_class"] != expected_substrate_class:
        errors.append("inference_substrate_class_invalid")
    if artifact["verdict_class"] != expected_class:
        errors.append("verdict_class_not_derived")
    if artifact["honest_verdict"] != expected_verdict:
        errors.append("honest_verdict_not_derived")
    summary = artifact["gate_check_summary"]
    if not isinstance(summary, Mapping) or not all(
        key in summary
        for key in ("failed_check", "expected_value", "observed_value", "passed")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the advisory preflight and fail only for an invalid artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    artifact = build_artifact(REPO_ROOT, args.date, output_path=output_path)
    errors = validate_artifact(artifact)
    print(output_path, flush=True)
    if errors:
        print("INVALID: " + ", ".join(errors), flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
